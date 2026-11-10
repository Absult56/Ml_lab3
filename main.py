"""
ГЛАВНЫЙ ОРКЕСТРАТОР.

Здесь по порядку выполняются ВСЕ пункты методички:
    1.1  plot_all_features_target
    1.2  split_temporal + scale_splits (внутри цикла scaled in {True, False})
    1.3  plot_corr_matrix
    1.4  pairwise_linear          + 5.2 plot_fit_curve
    1.5  multiple_linear
    1.6  nonlinear_curve_fit
    2.1  knn
    2.2  svm
    2.3  decision_tree            + 5.3 plot_tree_sklearn
    3.1  random_forest
    3.2  bagging
    3.3  gradient_boosting
    4.1  run_nn(arch="single")  x horizon in (1, 7)
    4.2  run_nn(arch="mlp")
    4.3  run_nn(arch="deep")
    5.1  plot_split_template + plot_diff (для 1.4/1.5/1.6)
    5.4  scaling_effect.csv (MAE scaled vs unscaled)
    5.5  results_Comparison.xlsx + bar plots
    5.6  summary.txt (интерпретация)
"""

import numpy as np
import pandas as pd

from config import OUT_DIR, FEATURES, TARGET
from data_loader import load_variant_data
from preprocessing import split_temporal, scale_splits

# Базовые модели (п. 1.4–1.6)
from regression_basic import pairwise_linear, multiple_linear, nonlinear_curve_fit

# Одиночные ML-модели (п. 2)
from ml_models import knn, svm, decision_tree

# Ансамбли (п. 3)
from ensemble_models import random_forest, bagging, gradient_boosting

# Нейросети (п. 4)
from neural_nets import run_nn

# Визуализации (п. 1.1, 1.3, 5.1, 5.2, 5.3, learning curves)
from visualization import (
    plot_all_features_target, plot_corr_matrix, plot_split_template,
    plot_diff, plot_fit_curve, plot_tree_sklearn, plot_learning_curves,
    plot_model_bar,
)

# Сборка результатов (п. 5.4–5.6)
from comparison import collect_metrics, save_excel, interpret


def main():
    print("=" * 70)
    print("ЛАБОРАТОРНАЯ №3 — ВАРИАНТ 12")
    print("=" * 70)

    # ==== Загрузка ====
    df = load_variant_data()

    # ==== [п. 1.1] Все признаки + y на одном графике ====
    plot_all_features_target(df)

    # ==== [п. 1.2] Хронологическое разбиение 70/15/15 ====
    train, val, test = split_temporal(df)
    print(f"\n[1.2] train={len(train)}, val={len(val)}, test={len(test)}")

    # ================================================================
    # ЦИКЛ ПО ВАРИАНТАМ С НОРМАЛИЗАЦИЕЙ И БЕЗ (п. 5.4)
    # ================================================================
    all_results_scaled, all_results_unscaled = [], []

    for scaled in (True, False):
        print(f"\n{'=' * 70}\n=== ВАРИАНТ scaled={scaled} ===\n{'=' * 70}")

        # [п. 1.2] готовим X/y с масштабированием (параметры фитятся на train)
        d = scale_splits(train, val, test, scale=scaled)

        # ==== [п. 1.3] Матрица корреляций — один раз (не зависит от scaled) ====
        if scaled:
            plot_corr_matrix(df)

        # ==== [п. 1.4] Парная линейная по признаку с макс |corr| с y ====
        corr_with_y = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).abs()
        best_feat = corr_with_y.idxmax()
        print(f"[1.4] Признак с макс |corr| с y: {best_feat} "
              f"({corr_with_y[best_feat]:.3f})")
        res_1_4 = pairwise_linear(d, best_feat, FEATURES)
        (all_results_scaled if scaled else all_results_unscaled).append(res_1_4)

        # ==== [п. 1.5] Множественная линейная регрессия ====
        res_1_5 = multiple_linear(d)
        (all_results_scaled if scaled else all_results_unscaled).append(res_1_5)

        # ==== [п. 1.6] Нелинейная через curve_fit ====
        res_1_6 = nonlinear_curve_fit(d)
        (all_results_scaled if scaled else all_results_unscaled).append(res_1_6)

        # ==== [п. 2.1] KNN ====
        (all_results_scaled if scaled else all_results_unscaled).append(knn(d))
        # ==== [п. 2.2] SVM ====
        (all_results_scaled if scaled else all_results_unscaled).append(svm(d))
        # ==== [п. 2.3] Decision Tree ====
        res_dt = decision_tree(d)
        (all_results_scaled if scaled else all_results_unscaled).append(res_dt)

        # ==== [п. 3.1] Random Forest ====
        (all_results_scaled if scaled else all_results_unscaled).append(random_forest(d))
        # ==== [п. 3.2] Bagging ====
        (all_results_scaled if scaled else all_results_unscaled).append(bagging(d))
        # ==== [п. 3.3] Gradient Boosting ====
        (all_results_scaled if scaled else all_results_unscaled).append(gradient_boosting(d))

        # ---------- Визуализации для 1.4/1.5/1.6 (только scaled — чтобы не дублировать) ----------
        if scaled:
            # [п. 5.1] шаблон train/val/test + график разности
            for r in (res_1_4, res_1_5, res_1_6):
                safe = (r["name"].replace(" ", "_")
                              .replace("(", "").replace(")", ""))
                plot_split_template(d, r["preds"], r["name"],
                                    f"05_split_{safe}.png")
                plot_diff(d["y_test"], r["preds"]["test"], r["name"],
                          f"05_diff_{safe}.png")

            # [п. 5.2] аппроксимирующая кривая для парной регрессии
            x_1d = df[best_feat].values
            idx = FEATURES.index(best_feat)
            m = res_1_4["model"]
            lin_all = m.predict(np.column_stack([df[FEATURES].values[:, idx]]))
            plot_fit_curve(x_1d, df[TARGET].values, lin_all,
                           best_feat, f"Pairwise Linear: {best_feat}",
                           "04_pairwise_curve.png")

            # [п. 5.3] дендрограмма дерева решений (п. 2.3)
            plot_tree_sklearn(res_dt["model"], FEATURES,
                              "06_decision_tree.png", "Decision Tree")

    # ================================================================
    # [п. 4.1–4.3] НЕЙРОСЕТИ: 3 архитектуры × {h=1, h=7}
    # ================================================================
    nn_results = []
    for arch in ("single", "mlp", "deep"):
        for h in (1, 7):
            r = run_nn(df, arch=arch, horizon=h)
            nn_results.append(r)

    # Собираем NN в таблицу (всегда scaled=True — NN без масштабирования плохо учится)
    nn_rows = []
    for r in nn_results:
        m = r["test_metrics"]
        nn_rows.append({"model": r["name"], "scaled": True,
                        "MAE": m["MAE"], "MSE": m["MSE"],
                        "R2": m["R2"], "corr": m["corr"]})

    # [п. 4] Кривые обучения: отдельно для 1-step и для 7-step
    plot_learning_curves(
        [r for r in nn_results if r["horizon"] == 1],
        fname="10_nn_learning_curves_h1.png",
    )
    plot_learning_curves(
        [r for r in nn_results if r["horizon"] == 7],
        fname="10_nn_learning_curves_h7.png",
    )

    # ================================================================
    # [п. 5.4] Сравнение «scaled vs unscaled» для общих моделей
    # ================================================================
    cmp_scaled   = collect_metrics(all_results_scaled,   scaled_flag=True)
    cmp_unscaled = collect_metrics(all_results_unscaled, scaled_flag=False)
    cmp_nn       = pd.DataFrame(nn_rows)

    merged = cmp_scaled.merge(cmp_unscaled, on="model",
                              suffixes=("_scaled", "_unscaled"))
    merged["MAE_delta"] = merged["MAE_unscaled"] - merged["MAE_scaled"]
    print("\n[5.4] Влияние масштабирования (MAE):")
    print(merged[["model", "MAE_scaled", "MAE_unscaled", "MAE_delta"]]
          .to_string(index=False))
    merged.to_csv(OUT_DIR / "scaling_effect.csv", index=False)

    # ================================================================
    # [п. 5.5] Сводная таблица + Excel + барчарты
    # ================================================================
    full = pd.concat([cmp_scaled, cmp_nn], ignore_index=True)
    full_sorted = save_excel(full, "results_Comparison.xlsx")

    plot_model_bar(full_sorted, metric="MAE", fname="12_bar_mae.png")
    plot_model_bar(full_sorted, metric="MSE", fname="12_bar_mse.png")

    # ================================================================
    # [п. 5.6] Текстовая интерпретация
    # ================================================================
    report = interpret(full_sorted)
    print("\n" + report)
    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # [п. 4] Число внутренних параметров каждой сети — выносим в файл
    params_info = "\n".join(
        f"{r['name']}: {r['n_params']} параметров" for r in nn_results)
    print("\n[4] Параметры нейросетей:\n" + params_info)
    with open(OUT_DIR / "nn_params.txt", "w", encoding="utf-8") as f:
        f.write(params_info)

    print(f"\n✅ Готово. Все результаты — в {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()