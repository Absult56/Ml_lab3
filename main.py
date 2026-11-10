"""Точка входа: полный пайплайн лабораторной работы №3 (вариант 12)."""

import numpy as np

from config import OUT_DIR
from data_loader import load_variant_data
from analysis import correlation_matrix, select_features, multicollinearity_check
from modeling import run_all_models, linear_coefficients, split_data
from visualization import (
    plot_target_distribution,
    plot_correlation_heatmap,
    plot_scatter_features,
    plot_actual_vs_pred,
    plot_residuals,
    plot_model_comparison,
)


def main():
    print("=" * 70)
    print("ЛАБОРАТОРНАЯ РАБОТА №3 — ВАРИАНТ 12")
    print("=" * 70)

    # 1. Загрузка
    df = load_variant_data()

    # 2. Визуализация целевой переменной
    plot_target_distribution(df)

    # 3. Корреляционный анализ
    corr = correlation_matrix(df)
    plot_correlation_heatmap(df, name="all")

    # 4. Отбор признаков
    features = select_features(df, target="y")
    multicollinearity_check(df, features)
    plot_scatter_features(df, features)

    # 5. Моделирование
    table, fitted, (X_test, y_test) = run_all_models(df, features)

    # 6. Коэффициенты лучшей линейной модели
    X_train, X_test, y_train, y_test = split_data(df, features)
    linear_coefficients(fitted["Linear"][0], features)

    # 7. Диагностика лучшей модели (по R2_test)
    best_row = table.sort_values("R2_test", ascending=False).iloc[0]
    best_name = best_row["model"]
    print(f"\nЛучшая модель по R²_test: {best_name} ({best_row['R2_test']:.4f})")

    model, y_pred = fitted[best_name]
    plot_actual_vs_pred(y_test, y_pred, f"{best_name}: факт vs прогноз",
                        "best_actual_vs_pred.png")
    plot_residuals(y_test, y_pred, f"{best_name}: анализ остатков",
                   "best_residuals.png")
    plot_model_comparison(table)

    # 8. Сохранение результатов
    df.to_csv(OUT_DIR / "variant12_data.csv", index=False)
    corr.to_csv(OUT_DIR / "corr_matrix.csv")
    table.to_csv(OUT_DIR / "model_metrics.csv", index=False)

    print(f"\nВсе результаты сохранены в папку: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()