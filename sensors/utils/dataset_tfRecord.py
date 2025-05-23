from pathlib import Path

import tensorflow as tf

from sensors.config import LENGTH, INPUTS


def parse_tfrecord_fn(example_proto):
    feature_description = {
        'X': tf.io.FixedLenFeature([], tf.string),  # raw bytes
        'y': tf.io.FixedLenFeature([1], tf.float32),  # label as float
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    # Decode raw bytes into float32 and reshape
    X = tf.io.decode_raw(parsed['X'], tf.float32)
    X = tf.reshape(X, (LENGTH, len(INPUTS)))

    return X, parsed['y'][0]


def create_tfrecord_dataset(tfRecord_file: Path, batch_size: int = 64, shuffle: bool = True):
    raw_dataset = tf.data.TFRecordDataset(tfRecord_file)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

    if shuffle:
        parsed_dataset = parsed_dataset.shuffle(buffer_size=10000)
    return parsed_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
