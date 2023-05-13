import os
from tfx.components import ImportExampleGen
from tfx.components import Evaluator
from tfx.components import Pusher
from tfx.components import Trainer
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.proto import pusher_pb2, trainer_pb2
from tfx.utils.dsl_utils import external_input

# Define the directory paths
_module_file = 'model.py'
_pipeline_name = "yolo_pipeline"
_pipeline_root = os.path.join('pipelines', _pipeline_name)
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_metadata_path = os.path.join('metadata', _pipeline_name, 'metadata.db')
_serving_model_dir = os.path.join('serving_model', _pipeline_name)

# Define the data path
data_root = './pipeline/tfrecords'

def _create_pipeline():
    # Define the ExampleGen component
    output = example_gen_pb2.Output(
                 split_config=example_gen_pb2.SplitConfig(splits=[
                     example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=4),
                     example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
                 ]))
    examples = external_input(data_root)
    example_gen = ImportExampleGen(input=examples, output_config=output)
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    # Define the schemagen, example_validator, tranfrom
        schema_gen = SchemaGen(
          statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)

    example_validator = ExampleValidator(
          statistics=statistics_gen.outputs['statistics'],
          schema=schema_gen.outputs['schema'])


    transform = Transform(
          examples=example_gen.outputs['examples'],
          schema=schema_gen.outputs['schema'],
          module_file=module_file)

    # Define the Trainer component
    trainer = Trainer(
        custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
        module_file=_module_file,
        examples=example_gen.outputs['examples'],
        train_args=trainer_pb2.TrainArgs(),
        eval_args=trainer_pb2.EvalArgs())

    # Define the Evaluator component
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'])

    # Define the Pusher component
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=_serving_model_dir)))

    # Return the pipeline definition
    return pipeline.Pipeline(
        pipeline_name=_pipeline_name,
        pipeline_root=_pipeline_root,
        components=[example_gen, trainer, evaluator, pusher],
        metadata_connection_config=metadata.sqlite_metadata_connection_config(_metadata_path),
    )

# Run the pipeline
context = InteractiveContext()
context.run(_create_pipeline())