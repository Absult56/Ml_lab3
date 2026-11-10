from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / r"D:\projects\python\ML_lab\lab3\data_12var\Lab3_Dataset.xlsx"
SHEET_NAME = "Sheet1"
OUT_DIR = BASE_DIR / r"D:\projects\python\ML_lab\lab3\results"
OUT_DIR.mkdir(exist_ok=True)

VARIANT = 12
COLUMN_MAP = {"a": 35, "b": 36, "c": 37, "d": 38, "e": 39, "y": 40}
DATA_START_ROW = 3
FEATURES = ["a", "b", "c", "d", "e"]
TARGET = "y"

TRAIN_RATIO, VAL_RATIO = 0.70, 0.15      # test = 0.15
RANDOM_STATE = 42

# Neural nets
SEQ_LEN = 1           # 1 шаг вперёд
SEQ_LEN_MULTI = 7     # 7 шагов вперёд
EPOCHS = 60
BATCH_SIZE = 32
LR = 1e-3