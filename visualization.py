"""
ВСЕ ГРАФИКИ ЛАБОРАТОРНОЙ.

Покрывает:
    1.1 — все признаки + целевая переменная на одном графике
    1.3 — матрица корреляций
    5.1 — шаблон «Actual Train / Predicted Train / ... / Test» + вертикальные
          разделители train/val/test и отдельный график «разность»
    5.2 — аппроксимирующая кривая для парной регрессии
    5.3 — дендрограмма (визуализация дерева решений)
    п. 4 — кривые обучения для NN

Все функции сохраняют PNG в results/ и НЕ показывают окна (plt.close).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree

from config import OUT_DIR, FEATURES, TARGET

sns.set_style("whitegrid")


# =========================================================
# [п. 1.1] ВСЕ ПРИЗНАКИ И ЦЕЛЕВАЯ ПЕРЕМЕННАЯ
# =========================================================
def plot_all_features_target(df: pd.DataFrame, fname="01_all_features.png"):
    """На одном графике — 5 признаков (a..e) и y."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in FEATURES + [TARGET]:
        ax.plot(df.index, df[col], label=col, linewidth=1)
    ax.set_xlabel("Индекс наблюдения")
    ax.set_ylabel("Значение")
    ax.set_title("Все признаки и целевая переменная")
    ax.legend(ncol=6)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=140)
    plt.close(fig)


# =========================================================
# [п. 1.3] МАТРИЦА КОРРЕЛЯЦИЙ
# =========================================================
def plot_corr_matrix(df: pd.DataFrame, fname="02_corr.png"):
    """Тепловая карта корреляций между признаками и y."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, ax=ax)
    ax.set_title("Матрица корреляций")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=140)
    plt.close(fig)


# =========================================================
# [п. 5.1] ШАБЛОН: ACTUAL TRAIN / PREDICTED TRAIN / ... / TEST
# =========================================================
def plot_split_template(d: dict, preds: dict, name: str, fname: str):
    """
    Строит график по шаблону из методички (см. рис. 1 на стр. 3):
        - Actual Train      (синий)
        - Predicted Train   (оранжевый)
        - Actual Validation (зелёный)
        - Predicted Val     (красный)
        - Actual Test       (фиолетовый)
        - Predicted Test    (коричневый)
    + вертикальные пунктирные линии на границах train/val/test
    + подписи "Train", "Validation", "Test" сверху.
    """
    y_tr, y_va, y_te = d["y_train"], d["y_val"], d["y_test"]
    n_tr, n_va, n_te = len(y_tr), len(y_va), len(y_te)

    idx_tr = np.arange(n_tr)
    idx_va = np.arange(n_tr, n_tr + n_va)
    idx_te = np.arange(n_tr + n_va, n_tr + n_va + n_te)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(idx_tr, y_tr, label="Actual Train", color="tab:blue", lw=1)
    ax.plot(idx_tr, preds["train"], label="Predicted Train", color="tab:orange", lw=1)
    ax.plot(idx_va, y_va, label="Actual Validation", color="tab:green", lw=1)
    ax.plot(idx_va, preds["val"], label="Predicted Validation", color="tab:red", lw=1)
    ax.plot(idx_te, y_te, label="Actual Test", color="tab:purple", lw=1)
    ax.plot(idx_te, preds["test"], label="Predicted Test", color="tab:brown", lw=1)

    # Вертикальные разделители
    ax.axvline(n_tr - 0.5, color="k", ls="--", alpha=0.5)
    ax.axvline(n_tr + n_va - 0.5, color="k", ls="--", alpha=0.5)
    ax.text(n_tr * 0.45, ax.get_ylim()[1], "Train", fontsize=12)
    ax.text(n_tr + n_va * 0.4, ax.get_ylim()[1], "Validation", fontsize=12)
    ax.text(n_tr + n_va + n_te * 0.3, ax.get_ylim()[1], "Test", fontsize=12)

    ax.set_title(f"{name}: прогноз vs факт")
    ax.set_xlabel("Индекс")
    ax.set_ylabel(TARGET)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=140)
    plt.close(fig)


def plot_diff(y_true, y_pred, name: str, fname: str):
    """
    [п. 5.1] «Аналогичный график для разности реальных и прогнозных значений».
    Красная линия = y_true - y_pred; горизонтальный ноль — ориентир.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.asarray(y_true) - np.asarray(y_pred), color="crimson", lw=1)
    ax.axhline(0, color="k", ls="--")
    ax.set_title(f"{name}: разность (реальное − прогноз)")
    ax.set_xlabel("Индекс")
    ax.set_ylabel("Error")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=140)
    plt.close(fig)


# =========================================================
# [п. 5.2] АППРОКСИМИРУЮЩАЯ КРИВАЯ
# =========================================================
def plot_fit_curve(x_1d, y, y_pred_curve, feature, name, fname):
    """
    Точечное облако исходных данных + красная линия аппроксимации.
    По методичке — «отобразить разными цветами» (data=blue, fit=red).
    Используется для парной линейной и нелинейной моделей.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x_1d, y, s=20, color="tab:blue", label="data", alpha=0.6)
    ax.plot(x_1d, y_pred_curve, "r--", lw=2, label="fit")
    ax.set_xlabel(feature)
    ax.set_ylabel(TARGET)
    ax.set_title(name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=140)
    plt.close(fig)


# =========================================================
# [п. 5.3] ДЕНДРОГРАММА ДЕРЕВА РЕШЕНИЙ
# =========================================================
def plot_tree_sklearn(tree_model, feature_names, fname, title):
    """
    Визуализация одиночного дерева из п. 2.3.
    sklearn.tree.plot_tree рисует разбиения, gini/mse, число объектов в узле.
    """
    fig, ax = plt.subplots(figsize=(18, 10))
    plot_tree(tree_model, feature_names=feature_names, filled=True,
              rounded=True, fontsize=7, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=120)
    plt.close(fig)


# =========================================================
# [п. 4] КРИВЫЕ ОБУЧЕНИЯ ДЛЯ НЕЙРОСЕТЕЙ
# =========================================================
def plot_learning_curves(results, fname="10_nn_learning_curves.png"):
    """
    Каждый results[i] — словарь из neural_nets.run_nn().
    Строим train_loss и val_loss по эпохам для нескольких моделей рядом.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        ax.plot(r["train_losses"], label="train")
        ax.plot(r["val_losses"], label="val")
        ax.set_title(r["name"])
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE loss")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=140)
    plt.close(fig)


# =========================================================
# [п. 5.4] СТОЛБЧАТАЯ ДИАГРАММА СРАВНЕНИЯ МОДЕЛЕЙ
# =========================================================
def plot_model_bar(df_cmp: pd.DataFrame, metric="MAE", fname=None):
    """Горизонтальный барчарт — быстро оценить, кто лучше по MAE/MSE."""
    fig, ax = plt.subplots(figsize=(10, 4))
    df_plot = df_cmp.sort_values(metric)
    ax.barh(df_plot["model"], df_plot[metric])
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(f"Сравнение моделей по {metric} (test)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / (fname or f"12_bar_{metric.lower()}.png"), dpi=140)
    plt.close(fig)