import os
from keras.models import Model
from keras.layers import Input, Lambda
from keras.optimizers import Adam
from keras_yolo3.yolo3.model import yolo_body, tiny_yolo_body, yolo_loss
from keras_yolo3.yolo3.utils import get_random_data
import tensorflow as tf
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow.keras.models import load_model
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.trainer.rewriting import converters
from tfx.components.trainer.rewriting import rewriter
from tfx.components.trainer.rewriting import rewriter_factory


def _build_yolo_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2, weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def _input_fn(file_pattern: List[str],
              data_accessor: DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              is_train: bool = False,
              batch_size: int = 200):
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )
    dataset = data_accessor.tf_dataset_factory(
      file_pattern,
      dataset_options.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_transformed_name(_LABEL_KEY)),
      tf_transform_output.transformed_metadata.schema)

    return dataset

def get_serve_image_fn(model):
    @tf.function
    def serve_image_fn(image_tensor):
        reshaped_tensor = tf.reshape(
            image_tensor, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]
        )
        return model(reshaped_tensor)

    return serve_image_fn

def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        fn_args.data_accessor,
        is_train=True,
        batch_size=_TRAIN_BATCH_SIZE
        )
    eval_dataset = _input_fn(
        fn_args.eval_files,
        tf_transform_output,
        fn_args.data_accessor,
        tf_transform_output,
      is_train=False,
      batch_size=_EVAL_BATCH_SIZE)

    model = _build_yolo_model()

    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=10,  # adjust as needed
        steps_per_epoch=100,  # adjust as needed
    )

    signatures = {
        'serving_default': get_serve_image_fn(model).get_concrete_function(
            tf.TensorSpec(shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], dtype=tf.float32)
        )
    }

    model.save(fn_args.serving_model_dir, signatures=signatures)