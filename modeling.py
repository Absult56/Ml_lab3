"""Построение и оценка моделей регрессии."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

from config import TEST_SIZE, RANDOM_STATE


def split_data(df: pd.DataFrame, features: list[str], target: str = "y"):
    X = df[features].values
    y = df[target].values
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def evaluate(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "model": name,
        "R2_train": r2_score(y_train, y_pred_train),
        "R2_test": r2_score(y_test, y_pred_test),
        "RMSE_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "RMSE_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "MAE_test": mean_absolute_error(y_test, y_pred_test),
    }
    return metrics, model, y_pred_test


def build_linear(X_train, y_train, X_test, y_test):
    return evaluate("Linear", LinearRegression(), X_train, y_train, X_test, y_test)


def build_scaled_linear(X_train, y_train, X_test, y_test):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression()),
    ])
    return evaluate("Linear+Scaler", pipe, X_train, y_train, X_test, y_test)


def build_polynomial(X_train, y_train, X_test, y_test, degree: int = 2):
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lr", LinearRegression()),
    ])
    return evaluate(f"Poly(deg={degree})", pipe, X_train, y_train, X_test, y_test)


def run_all_models(df: pd.DataFrame, features: list[str]):
    """Обучает несколько моделей и сводит метрики в таблицу."""
    X_train, X_test, y_train, y_test = split_data(df, features)

    results = []
    fitted_models = {}

    for builder in (build_linear, build_scaled_linear,
                    lambda *a: build_polynomial(*a, degree=2),
                    lambda *a: build_polynomial(*a, degree=3)):
        metrics, model, y_pred = builder(X_train, y_train, X_test, y_test)
        results.append(metrics)
        fitted_models[metrics["model"]] = (model, y_pred)

    table = pd.DataFrame(results).round(4)
    print("\nСводная таблица метрик:")
    print(table.to_string(index=False))

    return table, fitted_models, (X_test, y_test)


def linear_coefficients(model, features: list[str]) -> pd.DataFrame:
    """Коэффициенты линейной модели (если модель — LinearRegression / Pipeline)."""
    lr = model
    if hasattr(lr, "named_steps"):
        lr = lr.named_steps["lr"]
    coefs = pd.DataFrame({
        "feature": features,
        "coef": lr.coef_,
    }).sort_values("coef", key=np.abs, ascending=False)
    print("\nКоэффициенты линейной модели:")
    print(coefs.to_string(index=False))
    print(f"Свободный член: {lr.intercept_:.6f}")
    return coefs