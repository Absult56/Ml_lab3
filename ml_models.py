"""
ОДИНОЧНЫЕ МЕТОДЫ МАШИННОГО ОБУЧЕНИЯ (п. 2).

Покрывает:
    2.1 — K-Nearest Neighbors (KNN)
    2.2 — Support Vector Machine (SVM / SVR)
    2.3 — Decision Tree (одиночное дерево решений)

Для каждой модели подбираются гиперпараметры через GridSearchCV (как
требует методичка: «для параметрических методов определить оптимальные
гиперпараметры»).
"""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
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
    """
    Универсальная обёртка:
        - GridSearchCV(cv=3) по сетке param_grid,
        - обучение на train,
        - прогноз для train/val/test,
        - возврат словаря в общем формате.
    """
    gs = GridSearchCV(
        base, param_grid,
        cv=3,                              # 3-fold CV — быстро
        scoring="neg_mean_squared_error",  # ищем минимум MSE
        n_jobs=-1,                         # все ядра
    )
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
# [п. 2.1] K-NEAREST NEIGHBORS
# =========================================================
def knn(d):
    """
    KNN — метод на основе расстояний. Чувствителен к масштабу,
    поэтому в main.py вызывается и для scaled=True, и для scaled=False
    (это закроет требование п. 5.4 «сравнить с/без нормализации»).
    """
    return _tune(
        "KNN",
        KNeighborsRegressor(),
        {"n_neighbors": [3, 5, 7, 11, 15],
         "weights": ["uniform", "distance"]},
        d,
    )


# =========================================================
# [п. 2.2] SVM (Support Vector Regression)
# =========================================================
def svm(d):
    """SVM-SVR: RBF/linear ядро. Также чувствителен к масштабированию."""
    return _tune(
        "SVM",
        SVR(),
        {"C": [0.1, 1, 10, 100],
         "kernel": ["rbf", "linear"],
         "epsilon": [0.01, 0.1]},
        d,
    )


# =========================================================
# [п. 2.3] DECISION TREE
# =========================================================
def decision_tree(d):
    """
    Одиночное дерево решений. Ограничиваем глубину,
    чтобы избежать переобучения. Дерево также используется
    в п. 5.3 — рисуем его в visualization.plot_tree_sklearn.
    """
    return _tune(
        "Decision Tree",
        DecisionTreeRegressor(random_state=42),
        {"max_depth": [3, 5, 10, 20, None],
         "min_samples_split": [2, 5, 10],
         "min_samples_leaf": [1, 2, 4]},
        d,
    )