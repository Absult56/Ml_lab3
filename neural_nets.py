"""
НЕЙРОННЫЕ СЕТИ (п. 4).

Покрывает:
    4.1 — Однослойный линейный персептрон (с функцией активации Tanh)
    4.2 — Многослойный персептрон (2 скрытых слоя ReLU)
    4.3 — Глубокая сеть (3+ скрытых слоя + LayerNorm + Dropout)

Для КАЖДОЙ архитектуры решаются две задачи (методичка п. 4):
    - прогноз на 1 шаг вперёд (horizon=1)
    - прогноз на 7 шагов вперёд (horizon=7, выход — вектор длины 7)

Дополнительно:
    - сохраняются кривые обучения (train_loss и val_loss по эпохам) — п. 4;
    - подсчитывается число внутренних параметров каждой модели — п. 4;
    - метрики считаются как на масштабированном таргете (для сравнения),
      так и на обратно-масштабированном (в исходных единицах).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (SEQ_LEN_MULTI, EPOCHS, BATCH_SIZE, LR, RANDOM_STATE,
                    TRAIN_RATIO, VAL_RATIO, FEATURES, TARGET)
from preprocessing import split_shifted

torch.manual_seed(RANDOM_STATE)     # воспроизводимость


# =========================================================
# [п. 4.1] ОДНОСЛОЙНЫЙ ПЕРСЕПТРОН
# =========================================================
class SingleLayerPerceptron(nn.Module):
    """
    Один скрытый слой + активация + выходной слой:
        Linear(in, 32) -> Tanh -> Linear(32, out)
    Функция активации Tanh выбрана как оптимальная для регрессии
    гладких временных рядов (нет «мёртвых» нейронов, как у ReLU).
    """
    def __init__(self, in_dim, out_dim, act=nn.Tanh):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 32)   # скрытый слой
        self.act = act()                    # активация
        self.fc2 = nn.Linear(32, out_dim)  # выход (1 или 7)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# =========================================================
# [п. 4.2] МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН
# =========================================================
class MLP(nn.Module):
    """
    Два скрытых слоя ReLU (64 -> 32):
        Linear(in, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, out)
    ReLU — стандарт для глубоких сетей: быстро учится, не «затухает».
    """
    def __init__(self, in_dim, out_dim, hidden=64, act=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), act(),
            nn.Linear(hidden, hidden // 2), act(),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# [п. 4.3] ГЛУБОКАЯ СЕТЬ
# =========================================================
class DeepNN(nn.Module):
    """
    Три скрытых слоя + LayerNorm + Dropout (128 -> 64 -> 32 -> out).
    LayerNorm стабилизирует обучение, Dropout борется с переобучением
    на малом объёме данных (~700 строк train).
    """
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),      nn.LayerNorm(hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, hidden // 4), nn.ReLU(),
            nn.Linear(hidden // 4, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================
def _scale_y(y_train, y_val, y_test):
    """
    Масштабируем таргет отдельным StandardScaler.
    Это ОТДЕЛЬНЫЙ scaler, чтобы можно было вернуть прогноз в исходные единицы.
    """
    sc = StandardScaler().fit(y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train)
    ytr = sc.transform(y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train)
    yva = sc.transform(y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val)
    yte = sc.transform(y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test)
    return sc, ytr, yva, yte


def _train(model, X_tr, y_tr, X_va, y_va,
           epochs=EPOCHS, lr=LR, bs=BATCH_SIZE):
    """
    Стандартный цикл обучения:
        - оптимизатор: Adam(lr=1e-3) — хорошо работает «из коробки»
        - loss:        MSE (стандарт для регрессии)
        - возвращаем истории train_loss / val_loss по эпохам
          (это нужно для кривых обучения, п. 4).
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xv = torch.tensor(X_va, dtype=torch.float32)
    yv = torch.tensor(y_va, dtype=torch.float32)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=bs, shuffle=True)

    tr_hist, va_hist = [], []
    for _ in range(epochs):
        model.train()
        s = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            s += loss.item()
        tr_hist.append(s / len(loader))

        model.eval()
        with torch.no_grad():
            va_hist.append(loss_fn(model(Xv), yv).item())
    return tr_hist, va_hist


def _evaluate(model, X, y_true):
    """Возвращает (прогноз, метрики) на масштабированных данных."""
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return pred, {
        "MAE": mean_absolute_error(y_true, pred),
        "MSE": mean_squared_error(y_true, pred),
        "R2":  r2_score(y_true, pred),
        "corr": float(np.corrcoef(y_true.ravel(), pred.ravel())[0, 1]),
    }


# =========================================================
# ГЛАВНАЯ ФУНКЦИЯ ЗАПУСКА СЕТИ
# =========================================================
def run_nn(df, arch: str = "mlp", horizon: int = 1) -> dict:
    """
    Обучает сеть и возвращает всё, что нужно для отчёта.

    arch    : "single" | "mlp" | "deep"  — выбор архитектуры (п. 4.1/4.2/4.3)
    horizon : 1 | 7                       — 1 шаг или 7 шагов вперёд (п. 4)

    Возвращает словарь:
        name, arch, horizon, model, n_params,
        train_losses, val_losses,
        test_metrics,
        preds (unscaled) и y_true_unscaled
    """

    # --- 1. Подготовка данных (сдвиг на horizon) ---
    sc, X_tr, y_tr, X_va, y_va, X_te, y_te = split_shifted(df, horizon, scale=True)
    y_scaler, ytr_s, yva_s, yte_s = _scale_y(y_tr, y_va, y_te)

    # --- 2. Инициализация модели ---
    in_dim, out_dim = X_tr.shape[1], ytr_s.shape[1]
    Arch = {"single": SingleLayerPerceptron, "mlp": MLP, "deep": DeepNN}[arch]
    model = Arch(in_dim, out_dim)

    # --- 3. Подсчёт числа параметров (требование п. 4) ---
    n_params = sum(p.numel() for p in model.parameters())
    name = f"{arch.upper()} (h={horizon}, params={n_params})"
    print(f"\n[NN] {name}")

    # --- 4. Обучение + сбор кривых ---
    tr_hist, va_hist = _train(model, X_tr, ytr_s, X_va, yva_s)

    # --- 5. Метрики в масштабированном пространстве ---
    _, m_tr = _evaluate(model, X_tr, ytr_s)
    _, m_va = _evaluate(model, X_va, yva_s)
    _, m_te = _evaluate(model, X_te, yte_s)

    # --- 6. Обратное масштабирование и метрики в исходных единицах ---
    def _unscale(a): return y_scaler.inverse_transform(a)
    pred_te = _unscale(model(torch.tensor(X_te, dtype=torch.float32)).detach().numpy())
    true_te = _unscale(yte_s)
    test_metrics = {
        "MAE":  mean_absolute_error(true_te, pred_te),
        "MSE":  mean_squared_error(true_te, pred_te),
        "R2":   r2_score(true_te, pred_te),
        "corr": float(np.corrcoef(true_te.ravel(), pred_te.ravel())[0, 1]),
    }
    print(f"  test MAE={test_metrics['MAE']:.3f}, MSE={test_metrics['MSE']:.3f}")

    return {
        "name": name, "arch": arch, "horizon": horizon,
        "model": model, "n_params": n_params,
        "train_losses": tr_hist, "val_losses": va_hist,
        "test_metrics": test_metrics,
        "preds": {
            "train": _unscale(model(torch.tensor(X_tr, dtype=torch.float32)).detach().numpy()),
            "val":   _unscale(model(torch.tensor(X_va, dtype=torch.float32)).detach().numpy()),
            "test":  pred_te,
        },
        "y_true_unscaled": {
            "train": _unscale(ytr_s), "val": _unscale(yva_s), "test": true_te,
        },
    }