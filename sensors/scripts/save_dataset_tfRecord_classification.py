from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from sensors.config import DATA_PROCESSED_DIR, LENGTH, CLASSIFICATION_MODEL_OUTPUTS
from sensors.scripts.preprocess import (
    data_generated_dir,
    raw_data_train_dir,
    raw_data_val_dir,
)
from sensors.scripts.save_dataset_tfRecord import serialize_example

data_processed_dir = Path(DATA_PROCESSED_DIR)


def extract_gas_from_filename(file_name, file_contains_gas: bool):
    if not file_contains_gas:
        return CLASSIFICATION_MODEL_OUTPUTS.index("NO_GAS")

    for idx, gas in enumerate(CLASSIFICATION_MODEL_OUTPUTS):
        if gas in file_name:
            return idx

    raise ValueError("No gas found in filename")


def save_to_tfrecord(file_path: Path, tfrecord_writer: tf.io.TFRecordWriter):
    file_name = file_path.name

    data_file = data_generated_dir / file_name
    df = pd.read_excel(data_file)

    # Extract class label from filename
    file_contains_gas = df["Exposure"].sum() > 0
    class_label = extract_gas_from_filename(file_name, file_contains_gas)

    for window in df.rolling(window=LENGTH, step=1):
        if len(window) < LENGTH:
            continue

        X = window.drop(columns=["Exposure"]).to_numpy().astype(np.float32)
        y = 0 if window.Exposure.values[-1] == 0 else class_label

        if np.isnan(X).any():
            print(f"NaNs found in {file_path}")
            continue

        serialized = serialize_example(X, y)
        tfrecord_writer.write(serialized)


def process_split(split_dir: Path, tfrecord_file: Path):
    with tf.io.TFRecordWriter(str(tfrecord_file)) as writer:
        for file_path in tqdm(split_dir.iterdir()):
            save_to_tfrecord(file_path, writer)


def main():
    data_processed_dir.mkdir(parents=True, exist_ok=True)

    train_tfrecord = data_processed_dir / "train_class.tfrecord"
    val_tfrecord = data_processed_dir / "val_class.tfrecord"

    process_split(raw_data_train_dir, train_tfrecord)
    process_split(raw_data_val_dir, val_tfrecord)


if __name__ == "__main__":
    main()
