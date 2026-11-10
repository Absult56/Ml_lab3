"""
АНСАМБЛЕВЫЕ МЕТОДЫ (п. 3).

Покрывает:
    3.1 — Random Forest
    3.2 — Bagging (базовый алгоритм — дерево решений)
    3.3 — Gradient Boosting

Гиперпараметры подбираются через GridSearchCV, как требует методичка.
"""

import numpy as np
from sklearn.ensemble import (RandomForestRegressor,
                              BaggingRegressor,
                              GradientBoostingRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _metrics(y_true, y_pred):
    """Единый набор метрик для п. 5.4."""
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "R2":  r2_score(y_true, y_pred),
            "corr": float(np.corrcoef(y_true, y_pred)[0, 1])}


def _tune(name, base, param_grid, d):
    """Та же обёртка, что в ml_models._tune (повторяем, чтобы модули
    были независимы)."""
    gs = GridSearchCV(base, param_grid, cv=3,
                      scoring="neg_mean_squared_error", n_jobs=-1)
    gs.fit(d["X_train"], d["y_train"])

    preds = {"train": gs.predict(d["X_train"]),
             "val":   gs.predict(d["X_val"]),
             "test":  gs.predict(d["X_test"])}

    print(f"\n[{name}] best_params={gs.best_params_}")
    return {"name": name,
            "model": gs.best_estimator_,
            "preds": preds,
            "test_metrics": _metrics(d["y_test"], preds["test"]),
            "best_params": gs.best_params_}


# =========================================================
# [п. 3.1] RANDOM FOREST
# =========================================================
def random_forest(d):
    """Случайный лес: много деревьев, обученных на bootstrap-подвыборках."""
    return _tune(
        "Random Forest",
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {"n_estimators": [50, 100],
         "max_depth": [5, 10, None],
         "min_samples_leaf": [1, 2]},
        d,
    )


# =========================================================
# [п. 3.2] BAGGING
# =========================================================
def bagging(d):
    """
    Bagging = Bootstrap AGGregatING.
    Базовый алгоритм — DecisionTreeRegressor, обученный на случайных
    подвыборках признаков и объектов.
    """
    return _tune(
        "Bagging",
        BaggingRegressor(estimator=DecisionTreeRegressor(),
                         random_state=42, n_jobs=-1),
        {"n_estimators": [10, 30, 50],
         "max_samples": [0.7, 1.0],
         "max_features": [0.7, 1.0]},
        d,
    )


# =========================================================
# [п. 3.3] GRADIENT BOOSTING
# =========================================================
def gradient_boosting(d):
    """Градиентный бустинг на деревьях (sklearn-версия; XGBoost — опционально)."""
    return _tune(
        "Gradient Boosting",
        GradientBoostingRegressor(random_state=42),
        {"n_estimators": [50, 100],
         "learning_rate": [0.05, 0.1],
         "max_depth": [3, 5]},
        d,
    )