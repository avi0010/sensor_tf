from pathlib import Path
from pickle import load

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.contrib.concurrent import process_map

from sensors.config import DATA_PROCESSED_DIR, INPUTS, LENGTH
from sensors.scripts.preprocess import (
    data_generated_dir,
    raw_data_train_dir,
    raw_data_val_dir,
)

data_processed_dir = Path(DATA_PROCESSED_DIR)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def serialize_example(X, y):
    feature = {
        'X': _bytes_feature(X.tobytes()),  # serialize the array
        'y': _float_feature([y])           # single float label
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def save_to_tfrecord(file_path: Path, tfrecord_writer: tf.io.TFRecordWriter):
    file_name = file_path.name

    sc = load(open("StandardScaler.pkl", "rb"))

    data_file = data_generated_dir / file_name
    df = pd.read_excel(data_file)
    df[INPUTS] = sc.transform(df[INPUTS])

    for window in df.rolling(window=LENGTH, step=1):
        if len(window) < LENGTH:
            continue

        X = window.drop(columns=["Exposure"]).to_numpy().astype(np.float32)
        y = np.array(window.Exposure.values[-1], dtype=np.float32)

        if np.isnan(X).any() or np.isnan(y).any():
            print(f"NaNs found in {file_path}")
            continue

        serialized = serialize_example(X, y)
        tfrecord_writer.write(serialized)


def process_split(split_dir: Path, tfrecord_file: Path):
    with tf.io.TFRecordWriter(str(tfrecord_file)) as writer:
        for file_path in split_dir.iterdir():
            save_to_tfrecord(file_path, writer)


def main():
    data_processed_dir.mkdir(parents=True, exist_ok=True)

    train_tfrecord = data_processed_dir / "train.tfrecord"
    val_tfrecord = data_processed_dir / "val.tfrecord"

    process_split(raw_data_train_dir, train_tfrecord)
    process_split(raw_data_val_dir, val_tfrecord)


if __name__ == "__main__":
    main()