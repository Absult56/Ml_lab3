"""
БАЗОВЫЕ СТАТИСТИЧЕСКИЕ МОДЕЛИ РЕГРЕССИИ.

Покрывает пункты:
    1.4 — модель парной линейной регрессии (один признак)
    1.5 — модель множественной линейной регрессии (все признаки)
    1.6 — модель нелинейной регрессии через scipy.optimize.curve_fit

Все три возвращают словарь единого формата:
    {
        "name":         строка-название модели,
        "model":        обученный объект (или (тип, параметры)),
        "preds":        {"train": ..., "val": ..., "test": ...},
        "test_metrics": {"MAE": ..., "MSE": ..., "R2": ..., "corr": ...},
        "best_params":  опционально (для GridSearchCV-моделей),
    }
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import curve_fit


def _metrics(y_true, y_pred) -> dict:
    """Единый набор метрик для п. 5.4 (MAE / MSE / R2 / corr)."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R2":  r2_score(y_true, y_pred),
        "corr": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


# =========================================================
# [п. 1.4] ПАРНАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ
# =========================================================
def pairwise_linear(d: dict, feature: str, features: list[str]) -> dict:
    """
    Модель y = w * feature + b, обученная только на одном признаке.

    feature — название признака (a/b/c/d/e),
              выбранного как «наибольшая |corr| с y» в main.py.
    """
    idx = features.index(feature)
    # Берём только один столбец из X_train/val/test
    Xtr = d["X_train"][:, [idx]]
    Xva = d["X_val"][:, [idx]]
    Xte = d["X_test"][:, [idx]]

    m = LinearRegression().fit(Xtr, d["y_train"])

    preds = {"train": m.predict(Xtr),
             "val":   m.predict(Xva),
             "test":  m.predict(Xte)}

    return {
        "name": f"Pairwise Linear ({feature})",
        "model": m,
        "preds": preds,
        "test_metrics": _metrics(d["y_test"], preds["test"]),
        "feature": feature,
    }


# =========================================================
# [п. 1.5] МНОЖЕСТВЕННАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ
# =========================================================
def multiple_linear(d: dict) -> dict:
    """
    Модель y = w1*a + w2*b + w3*c + w4*d + w5*e + bias
    (используются ВСЕ признаки).
    """
    m = LinearRegression().fit(d["X_train"], d["y_train"])
    preds = {"train": m.predict(d["X_train"]),
             "val":   m.predict(d["X_val"]),
             "test":  m.predict(d["X_test"])}

    # Печатаем коэффициенты — это удобно вставить в отчёт
    print("\n[1.5] Коэффициенты множественной регрессии:")
    for f, c in zip(["a", "b", "c", "d", "e"], m.coef_):
        print(f"  {f}: {c:+.6f}")
    print(f"  intercept: {m.intercept_:+.6f}")

    return {
        "name": "Multiple Linear",
        "model": m,
        "preds": preds,
        "test_metrics": _metrics(d["y_test"], preds["test"]),
    }


# =========================================================
# [п. 1.6] НЕЛИНЕЙНАЯ РЕГРЕССИЯ ЧЕРЕЗ curve_fit
# =========================================================
def nonlinear_curve_fit(d: dict) -> dict:
    """
    Модель:
        y = a*x0 + b*x1 + c*x2 + d*x3 + e*x4 + bias
            + a*b*x0*x1 + c*d*x2*x3
    где x0..x4 — это признаки [a, b, c, d, e] (после масштабирования).

    Гиперпараметры подбираются через scipy.optimize.curve_fit.
    Начальное приближение — вектор единиц.
    """

    # Функция-аппроксимант. curve_fit ожидает f(X, *params), где params — 6 чисел.
    def f(X, *p):
        w_a, w_b, w_c, w_d, w_e, bias = p
        return (w_a * X[:, 0] + w_b * X[:, 1] + w_c * X[:, 2]
                + w_d * X[:, 3] + w_e * X[:, 4] + bias
                + w_a * w_b * X[:, 0] * X[:, 1]
                + w_c * w_d * X[:, 2] * X[:, 3])

    # p0 — начальное приближение, maxfev — лимит итераций (иначе может упасть)
    popt, _ = curve_fit(
        lambda X, *p: f(np.atleast_2d(X), *p),
        d["X_train"], d["y_train"],
        p0=np.ones(6), maxfev=20000,
    )

    # Прогноз на всех трёх выборках
    preds = {k: f(d[f"X_{k}"], *popt) for k in ("train", "val", "test")}

    print(f"\n[1.6] curve_fit параметры: {np.round(popt, 4)}")
    return {
        "name": "Nonlinear (curve_fit)",
        "model": ("curve_fit", popt),       # сохраняем параметры
        "preds": preds,
        "test_metrics": _metrics(d["y_test"], preds["test"]),
    }