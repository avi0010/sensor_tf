from itertools import islice
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
        save_to_tfrecord(Path("data_generated/68A_6685_Q1_NH3_50ppm_Response_1Rep_Outlet_connected_directly_to_Active_Unit_Pump_Inlet_250205091019_data_03112025_021447.xlsx"), writer)
        save_to_tfrecord(Path("data_generated/68A_6685_Q1_Cl2_1ppm_Response_1Rep_Active_Unit_w_Cover_no_sorbent_Smaller_Chamber_Bronkhorst_Setup_RH_21.3_T_22.1_200926115033_data_03112025_012834.xlsx"), writer)
        save_to_tfrecord(Path("data_generated/68A_6685_Q1_H2S_10ppm_Response_1Rep_Active_Unit_w_Cover_no_sorbent_Smaller_Chamber_Bronkhorst_Setup_RH_19.7_T_24.1_241010121613_data_03112025_012922.xlsx"), writer)
        save_to_tfrecord(Path("data_generated/68A_6685_Q1_HCN_50ppm_Response_1Rep_Active_Unit_w_Cover_no_sorbent_Smaller_Chamber_Bronkhorst_Setup_RH_19.3_T_22.9_241002134636_data_11072024_081414.xlsx"), writer)
        # for file_path in tqdm(islice(split_dir.iterdir(), 1)):
        #     save_to_tfrecord(file_path, writer)


def main():
    data_processed_dir.mkdir(parents=True, exist_ok=True)

    train_tfrecord = data_processed_dir / "train_class.tfrecord"
    val_tfrecord = data_processed_dir / "val_class.tfrecord"

    process_split(raw_data_train_dir, train_tfrecord)
    process_split(raw_data_val_dir, val_tfrecord)


if __name__ == "__main__":
    main()
