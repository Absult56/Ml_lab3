"""
СБОРКА И ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ.

Покрывает:
    5.4 — сравнение метрик scaled / unscaled (см. main.py: scaling_effect.csv)
    5.5 — сводная таблица всех моделей + экспорт в results_Comparison.xlsx
    5.6 — определение наилучшего/наихудшего метода и текстовое объяснение
"""

import pandas as pd

from config import OUT_DIR


# =========================================================
# [п. 5.4 / 5.5] СБОРКА МЕТРИК
# =========================================================
def collect_metrics(results: list[dict], scaled_flag: bool) -> pd.DataFrame:
    """
    Превращает список словарей (из *_models.py) в DataFrame
    с колонками model / scaled / MAE / MSE / R2 / corr.
    """
    rows = []
    for r in results:
        m = r["test_metrics"]
        rows.append({
            "model":  r["name"],
            "scaled": scaled_flag,     # True/False — для п. 5.4
            "MAE": m["MAE"],
            "MSE": m["MSE"],
            "R2":  m["R2"],
            "corr": m["corr"],
        })
    return pd.DataFrame(rows)


# =========================================================
# [п. 5.5] ЭКСПОРТ В EXCEL
# =========================================================
def save_excel(df: pd.DataFrame, fname="results_Comparison.xlsx"):
    """
    Сортирует по возрастанию MAE (лучшие сверху) и сохраняет в Excel.
    Возвращает отсортированный DataFrame — пригодится для барчартов.
    """
    df_out = df.sort_values("MAE").reset_index(drop=True)
    df_out.to_excel(OUT_DIR / fname, index=False)
    print(f"\n[5.5] {fname} сохранён ({len(df_out)} строк)")
    return df_out


# =========================================================
# [п. 5.6] ИНТЕРПРЕТАЦИЯ
# =========================================================
def interpret(df: pd.DataFrame) -> str:
    """
    Возвращает текстовый отчёт:
        - какая модель лучшая (по MAE),
        - какая худшая,
        - типовые объяснения (масштабирование, переобучение, шум).
    Файл сохраняется в main.py как results/summary.txt.
    """
    best = df.iloc[0]
    worst = df.iloc[-1]
    lines = [
        "=== Интерпретация результатов (тест) ===",
        f"Наилучшая модель:  {best['model']} | "
        f"MAE={best['MAE']:.3f}, MSE={best['MSE']:.3f}, "
        f"R2={best['R2']:.4f}, corr={best['corr']:.4f}",
        f"Наихудшая модель:  {worst['model']} | "
        f"MAE={worst['MAE']:.3f}, MSE={worst['MSE']:.3f}, "
        f"R2={worst['R2']:.4f}, corr={worst['corr']:.4f}",
        "",
        "Замечания:",
        " - Масштабирование критично для KNN и SVM "
        "(расстояния/ядра чувствительны к масштабу).",
        " - Деревья и ансамбли на их основе почти не зависят от масштабирования.",
        " - При R2, близком к 0.999, линейная модель выигрывает из-за почти "
        "линейной зависимости y от признаков.",
        " - Полиномиальные/глубокие модели склонны к переобучению при малом n.",
    ]
    return "\n".join(lines)