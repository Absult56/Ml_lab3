"""Загрузка и подготовка данных варианта 12."""

import pandas as pd

from config import DATA_FILE, SHEET_NAME, COLUMN_MAP, DATA_START_ROW, VARIANT


def load_variant_data() -> pd.DataFrame:
    """Читает Excel и возвращает DataFrame с колонками a, b, c, d, e, y."""
    raw = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME, header=None)
    print(f"Исходный размер файла: {raw.shape}")

    # Выбираем нужные столбцы и переименовываем
    df = raw.iloc[DATA_START_ROW:, list(COLUMN_MAP.values())].copy()
    df.columns = list(COLUMN_MAP.keys())

    # Приводим к числу, отбрасываем строки с NaN (хвост файла)
    df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    df = df.astype(float)

    print(f"Вариант {VARIANT}: {df.shape[0]} наблюдений, {df.shape[1]} признаков")
    print(df.head())
    print("\nОписательные статистики:")
    print(df.describe().T)
    return df


if __name__ == "__main__":
    load_variant_data()