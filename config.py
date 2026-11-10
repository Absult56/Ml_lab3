"""Конфигурация лабораторной работы №3, вариант 12."""

from pathlib import Path

# Путь к исходному файлу
DATA_FILE = Path(r"D:\projects\python\ML_lab\lab3\data_12var\Lab3_Dataset.xlsx")
SHEET_NAME = "Sheet1"

# Номер варианта и соответствующие столбцы (0-based индексы pandas)
VARIANT = 12
# Столбцы варианта 12: AJ..AO  ->  индексы 35..40 (0-based)
COLUMN_MAP = {
    "a": 35,  # AJ
    "b": 36,  # AK
    "c": 37,  # AL
    "d": 38,  # AM
    "e": 39,  # AN
    "y": 40,  # AO
}

# Строка, с которой начинаются числовые данные (0-based после чтения без header)
# В файле шапка занимает 3 строки: заголовки, "Вариант N", a/b/c/d/e/y
DATA_START_ROW = 3

# Параметры разбиения
TEST_SIZE = 0.25
RANDOM_STATE = 42

# Порог значимости корреляции
CORR_THRESHOLD = 0.3
ALPHA = 0.05

# Директория для результатов
OUT_DIR = Path(r"D:\projects\python\ML_lab\lab3\`results")
OUT_DIR.mkdir(exist_ok=True)