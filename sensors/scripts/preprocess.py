from pathlib import Path
from pickle import dump

import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm.contrib.concurrent import process_map

from sensors.config import (
    DATA_GENERATED_DIR,
    ENVIRONMENT,
    INPUTS,
    OUTPUTS,
    PARAMETER,
    RAW_DATA_DIR,
    SHEETS,
    WARMUP,
)

data_generated_dir = Path(DATA_GENERATED_DIR)
raw_data_dir = Path(RAW_DATA_DIR)
raw_data_train_dir = raw_data_dir / "train"
raw_data_val_dir = raw_data_dir / "val"


def create_dataset(filename: Path):
    try:
        df = pd.DataFrame()

        S0_db = pd.read_excel(filename, sheet_name="S0")
        S0_db = S0_db.infer_objects()
        S0_filtered = S0_db[S0_db["Routine Counter"] >= WARMUP]
        df = S0_filtered[ENVIRONMENT + OUTPUTS].join(df)

        for sheet_name in SHEETS:
            sheet_df = pd.read_excel(filename, sheet_name=sheet_name)
            sheet_df = sheet_df.infer_objects()
            sheet_df[sheet_name] = sheet_df[PARAMETER]
            filtered_df = sheet_df[sheet_df["Routine Counter"] >= WARMUP]
            df[sheet_name] = filtered_df[sheet_name]

        df.replace(-999, 0, inplace=True)
        df.to_excel(data_generated_dir / filename.name, index=False)

    except Exception as e:
        print(f"{filename}: {e}")
    return


def main():
    data_generated_dir.mkdir(exist_ok=True)

    _ = process_map(create_dataset, list(raw_data_train_dir.iterdir()))
    _ = process_map(create_dataset, list(raw_data_val_dir.iterdir()))

    dfs: list[pd.DataFrame] = []
    for file in raw_data_train_dir.iterdir():
        df = pd.read_excel(data_generated_dir / file.name)
        dfs.append(df)

    dd = pd.concat(dfs)
    SC = StandardScaler()
    SC.fit(dd[INPUTS])
    dump(SC, open("StandardScaler.pkl", "wb"))


if __name__ == "__main__":
    main()
