"""
ПРЕДОБРАБОТКА ДАННЫХ.

Здесь реализовано:
- [п. 1.2] хронологическое разбиение train/val/test (70/15/15)
           + масштабирование (StandardScaler, фитится ТОЛЬКО на train)
- [п. 4]   подготовка сдвинутых таргетов y_{t+1..t+h} для нейросетей
           (одношаговый и многошаговый прогноз)

Важно: scaler обучается только на train, и его параметры применяются
к train/val/test. Это требование методички в п. 1.2.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import TRAIN_RATIO, VAL_RATIO, FEATURES, TARGET


# =========================================================
# [п. 1.2] ХРОНОЛОГИЧЕСКОЕ РАЗБИЕНИЕ 70/15/15
# =========================================================
def split_temporal(df: pd.DataFrame):
    """
    Делит DataFrame по индексам (без перемешивания), потому что данные
    представляют временной ряд — будущее не должно попадать в train.

    Возвращает (train, val, test) — три DataFrame с теми же колонками.
    """
    n = len(df)
    n_tr = int(n * TRAIN_RATIO)
    n_va = int(n * VAL_RATIO)
    return (df.iloc[:n_tr].reset_index(drop=True),
            df.iloc[n_tr:n_tr + n_va].reset_index(drop=True),
            df.iloc[n_tr + n_va:].reset_index(drop=True))


# =========================================================
# [п. 1.2] МАСШТАБИРОВАНИЕ С СОХРАНЕНИЕМ ПАРАМЕТРОВ TRAIN
# =========================================================
def scale_splits(train, val, test, scale: bool = True):
    """
    Возвращает словарь с X/y для трёх выборок.

    Если scale=True:
        StandardScaler.fit(train[FEATURES])  — обучается ТОЛЬКО на train
        transform применяется к train/val/test   — это требование п. 1.2
    Если scale=False:
        ничего не масштабируем (для сравнения «с нормализацией / без», п. 5.4).
    """
    if scale:
        sc = StandardScaler().fit(train[FEATURES])
    else:
        class _Identity:
            def transform(self, x): return np.asarray(x)
        sc = _Identity()

    return {
        "scaler": sc,
        "X_train": sc.transform(train[FEATURES]), "y_train": train[TARGET].values,
        "X_val":   sc.transform(val[FEATURES]),   "y_val":   val[TARGET].values,
        "X_test":  sc.transform(test[FEATURES]),  "y_test":  test[TARGET].values,
    }


# =========================================================
# [п. 4] СДВИНУТЫЕ ТАРГЕТЫ ДЛЯ МНОГОШАГОВОГО ПРОГНОЗА
# =========================================================
def build_shifted_targets(y: np.ndarray, horizon: int) -> np.ndarray:
    """
    Для нейросети:
        horizon=1  ->  y[t+1]                  (форма: [N, 1])
        horizon=7  ->  y[t+1..t+7]             (форма: [N, 7])

    Длина результата: len(y) - horizon.
    Это и есть требование п. 4: «выходной тензор дополняется измерением».
    """
    if horizon == 1:
        return y[1:].reshape(-1, 1)
    return np.stack([y[i + 1:i + 1 + horizon] for i in range(len(y) - horizon)])


def split_shifted(df: pd.DataFrame, horizon: int, scale: bool = True):
    """
    Хронологический split + сдвинутый таргет для NN.
    Используется в neural_nets.run_nn().

    Возвращает: scaler, X_tr, y_tr, X_va, y_va, X_te, y_te
    где y_* — матрицы формы [N, horizon].
    """
    n = len(df)
    n_tr = int(n * TRAIN_RATIO)
    n_va = int(n * VAL_RATIO)

    # scaler фитим только на train (требование п. 1.2)
    if scale:
        sc = StandardScaler().fit(df[FEATURES])
    else:
        class _Identity:
            def transform(self, x): return np.asarray(x)
        sc = _Identity()
    X = sc.transform(df[FEATURES])
    y = df[TARGET].values

    def _pack(lo, hi):
        """Берёт кусок [lo:hi), сдвигает X и y на horizon."""
        xx = X[lo:hi - horizon]
        yy = build_shifted_targets(y[lo:hi], horizon)
        return xx, yy

    X_tr, y_tr = _pack(0, n_tr)
    X_va, y_va = _pack(n_tr, n_tr + n_va)
    X_te, y_te = _pack(n_tr + n_va, n)
    return sc, X_tr, y_tr, X_va, y_va, X_te, y_te