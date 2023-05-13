import os
from typing import List
import absl
import tensorflow as tf
# Set the dataset directory path relative to the current working directory
dataset_dir = 'pothole_dataset'
tfrecords_dir = 'tfrecords'

# Create the tfrecords directory if it doesn't exist
os.makedirs(tfrecords_dir, exist_ok=True)

# Iterate over your dataset and convert images to TFRecords
splits = ['train', 'test', 'valid']



def convert_to_tfrecords(image_path: str, label_path: str, writer: tf.io.TFRecordWriter):
    # Read the image and label data from the files and perform any necessary preprocessing
    # Convert the data to the TFRecord format and write it to the writer

    # Example implementation (replace with your own logic):
    with open(image_path, 'rb') as f:
        image_data = f.read()

    with open(label_path, 'r') as f:
        label = f.readline().strip()  # Assuming the label is a single line in the file

    # Create a TFExample from the image and label data
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode()])),
            }
        )
    )

    # Write the TFExample to the TFRecord
    writer.write(example.SerializeToString())


for split in splits:
    split_images_dir = os.path.join(dataset_dir, 'images', split)
    split_labels_dir = os.path.join(dataset_dir, 'labels', split)
    split_tfrecords_dir = os.path.join(tfrecords_dir, split)
    os.makedirs(split_tfrecords_dir, exist_ok=True)
    tfrecords_writer = tf.io.TFRecordWriter(os.path.join(split_tfrecords_dir, f'{split}.tfrecords'))

    image_files = os.listdir(split_images_dir)

    for image_file in image_files:
        image_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(split_images_dir, image_file)
        label_path = os.path.join(split_labels_dir, f'{image_name}.txt')
        convert_to_tfrecords(image_path, label_path, tfrecords_writer)

    tfrecords_writer.close()