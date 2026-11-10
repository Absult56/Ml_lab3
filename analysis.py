"""Корреляционный анализ и отбор значимых признаков."""

import numpy as np
import pandas as pd
from scipy import stats

from config import CORR_THRESHOLD, ALPHA


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Матрица корреляций Пирсона."""
    corr = df.corr(method="pearson")
    return corr


def correlations_with_target(df: pd.DataFrame, target: str = "y") -> pd.DataFrame:
    """Корреляции каждого признака с целевой переменной + p-value."""
    rows = []
    y = df[target].values
    for col in df.columns:
        if col == target:
            continue
        r, p = stats.pearsonr(df[col].values, y)
        rows.append({"feature": col, "r": r, "p_value": p})
    out = pd.DataFrame(rows).sort_values("r", key=np.abs, ascending=False)
    return out


def select_features(df: pd.DataFrame, target: str = "y") -> list[str]:
    """Отбирает признаки, значимо коррелирующие с y (|r| >= порог и p < alpha)."""
    corr_df = correlations_with_target(df, target)
    print("\nКорреляции признаков с целевой переменной:")
    print(corr_df.to_string(index=False))

    selected = corr_df[
        (corr_df["r"].abs() >= CORR_THRESHOLD) & (corr_df["p_value"] < ALPHA)
    ]["feature"].tolist()

    if not selected:
        print(f"\nНи один признак не преодолел порог |r|>={CORR_THRESHOLD} — берём топ-2.")
        selected = corr_df.head(2)["feature"].tolist()

    print(f"\nОтобраны признаки: {selected}")
    return selected


def multicollinearity_check(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Проверка мультиколлинеарности между отобранными признаками."""
    sub = df[features]
    corr = sub.corr()
    print("\nМатрица корреляций отобранных признаков:")
    print(corr.round(3))
    return corr