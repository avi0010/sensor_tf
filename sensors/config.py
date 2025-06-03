SHEETS = [
    "S0",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8",
    "S9",
    "S10",
    "S11",
    "S12",
    "S13",
    "S14",
    "S15",
    "S16",
    "S17",
    "S18",
    "S19",
    "S20",
    "S21",
    "S22",
    "S23",
    "BME",
]
ENVIRONMENT = ["Temperature Deriv.", "Humidity Deriv."]
OUTPUTS = ["Exposure"]

INPUTS =  ENVIRONMENT + SHEETS
SENSORS = INPUTS + OUTPUTS

PARAMETER = "Raw Deriv."
WARMUP = 200
MM_FILE_NAME = f"scaler_{PARAMETER}.gz"

RAW_DATA_DIR = "data"
LENGTH = 101

DATA_GENERATED_DIR = "data_generated"

DATA_PROCESSED_DIR = "data_processed"
