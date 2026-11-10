"""Графики для отчёта."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import OUT_DIR

sns.set_style("whitegrid")


def plot_target_distribution(df: pd.DataFrame, target: str = "y"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(df[target], bins=30, color="steelblue", edgecolor="black")
    ax[0].set_title("Гистограмма y")
    ax[0].set_xlabel("y")

    ax[1].boxplot(df[target], vert=False)
    ax[1].set_title("Ящик с усами y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "target_distribution.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, name: str = "all"):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"Матрица корреляций ({name})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"corr_{name}.png", dpi=150)
    plt.close(fig)


def plot_actual_vs_pred(y_true, y_pred, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k")
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", label="y = x")
    ax.set_xlabel("Фактическое y")
    ax.set_ylabel("Предсказанное y")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)


def plot_residuals(y_true, y_pred, title: str, fname: str):
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(y_pred, residuals, alpha=0.6, edgecolors="k")
    ax[0].axhline(0, color="red", linestyle="--")
    ax[0].set_xlabel("Предсказание")
    ax[0].set_ylabel("Остаток")
    ax[0].set_title("Остатки vs предсказания")

    ax[1].hist(residuals, bins=30, color="steelblue", edgecolor="black")
    ax[1].set_title("Распределение остатков")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)


def plot_scatter_features(df: pd.DataFrame, features: list[str], target: str = "y"):
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, f in zip(axes, features):
        ax.scatter(df[f], df[target], alpha=0.6, edgecolors="k")
        ax.set_xlabel(f)
        ax.set_ylabel(target)
        ax.set_title(f"{f} vs {target}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "features_vs_target.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(table))
    w = 0.25
    ax.bar(x - w, table["R2_train"], w, label="R² train")
    ax.bar(x, table["R2_test"], w, label="R2 test")
    ax.bar(x + w, table["RMSE_test"], w, label="RMSE test")
    ax.set_xticks(x)
    ax.set_xticklabels(table["model"], rotation=15)
    ax.set_title("Сравнение моделей")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)