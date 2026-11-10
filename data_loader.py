"""
ЗАГРУЗКА ИСХОДНЫХ ДАННЫХ (подготовка к п. 1.x).

Задача модуля:
1. Прочитать Lab3_Dataset.xlsx.
2. Оставить только 6 столбцов, соответствующих варианту 12 (a..e, y).
3. Привести к float, отбросить строки с NaN (в файле бывают хвостовые).

Пункты методички: подготовка данных для 1.1–5.x
"""

import pandas as pd
from config import (DATA_FILE, SHEET_NAME, COLUMN_MAP,
                    DATA_START_ROW, VARIANT, FEATURES)


def load_variant_data() -> pd.DataFrame:
    """Возвращает DataFrame с колонками a, b, c, d, e, y (только float)."""

    # header=None — читаем без интерпретации первой строки как названий.
    # Это нужно, потому что шапка многоуровневая (Вариант 1, Вариант 2, ...).
    raw = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME, header=None)

    # Берём только нужные столбцы варианта 12 и переименовываем их в a..y.
    df = raw.iloc[DATA_START_ROW:, list(COLUMN_MAP.values())].copy()
    df.columns = list(COLUMN_MAP.keys())

    # to_numeric с errors="coerce" превращает некорректные значения в NaN,
    # dropna() удаляет строки с любым NaN (обычно это пустые строки внизу файла).
    df = (df.apply(pd.to_numeric, errors="coerce")
            .dropna()
            .reset_index(drop=True))

    print(f"Вариант {VARIANT}: {df.shape[0]} наблюдений, признаки {FEATURES}")
    print(df.describe().T.round(3))     # описательные статистики для отчёта
    return df