from pathlib import Path
from pickle import load

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from sensors.config import DATA_PROCESSED_DIR, INPUTS, LENGTH
from sensors.scripts.preprocess import (
    data_generated_dir,
    raw_data_train_dir,
    raw_data_val_dir,
)

data_processed_dir = Path(DATA_PROCESSED_DIR)


def save_dataset(file_path: Path):
    file_name = file_path.name
    file_dir = file_path.parent.name

    save_location = data_processed_dir / file_dir / file_name
    save_location.mkdir(exist_ok=True, parents=True)

    sc = load(open("StandardScaler.pkl", "rb"))

    data_file = data_generated_dir / file_name
    df = pd.read_excel(data_file)
    df[INPUTS] = sc.transform(df[INPUTS])

    for idx, window in enumerate(df.rolling(window=LENGTH, step=1)):
        if len(window) < LENGTH:
            continue

        X = window.drop(columns=["Exposure"]).to_numpy().astype(np.float32)
        y = np.array(window.Exposure.values[-1], dtype=np.float32)
        if np.isnan(X).any() or np.isnan(y).any():
            print(file_path)

        np.savez_compressed(save_location / f"{idx}", X=X, y=y)


def main():
    data_processed_dir.mkdir()
    process_map(save_dataset, list(raw_data_train_dir.iterdir()))
    process_map(save_dataset, list(raw_data_val_dir.iterdir()))


if __name__ == "__main__":
    main()
