"""
forecasting.py
===============
Section 4.2 of the report: Yeo-Johnson transform + BiLSTM that watches the
last WINDOW timesteps of CPU utilisation and forecasts the next value.

Implementation note (deviation from the report, worth stating in your
paper): the report specifies TensorFlow/Keras for the BiLSTM. We
implement it in PyTorch instead, purely so the whole stack (forecaster +
PPO + DQN) shares one autodiff framework and one dependency footprint
-- Stable-Baselines3 is PyTorch-based, so this avoids installing both
TensorFlow and PyTorch. The architecture (bidirectional LSTM, same
hidden size, same lookback window) and the Yeo-Johnson preprocessing
are unchanged, so results are not expected to differ meaningfully from
a Keras re-implementation; if you need a Keras version for consistency
with the report, the class API here maps 1:1 onto
`tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(...))`.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import PowerTransformer

WINDOW = 50


class BiLSTMForecaster(nn.Module):
    def __init__(self, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x):  # x: (B, WINDOW, 1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # concat of final fwd+bwd hidden states
        return self.head(last).squeeze(-1)


def make_windows(series: np.ndarray, window: int = WINDOW):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)


def chronological_split(n: int, train_frac: float = 0.70, val_frac: float = 0.15) -> dict:
    """Half-open index ranges [start, end) into the series."""
    a = int(n * train_frac)
    b = int(n * (train_frac + val_frac))
    return dict(train=(0, a), val=(a, b), test=(b, n))


class LoadForecastingPipeline:
    """Bundles the Yeo-Johnson transformer with the BiLSTM so a caller can
    do: pipeline.fit(trace); pipeline.predict_next(last_50_raw_values)."""

    def __init__(self, window: int = WINDOW, hidden_size: int = 32, device: str = "cpu"):
        self.window = window
        self.pt = PowerTransformer(method="yeo-johnson")
        self.model = BiLSTMForecaster(hidden_size=hidden_size)
        self.device = device

    def fit_split(self, raw_series: np.ndarray, epochs: int = 20, lr: float = 1e-3,
                  batch_size: int = 64, train_frac: float = 0.70, val_frac: float = 0.15,
                  verbose: bool = True):
        """Leak-free fit. The series is split CHRONOLOGICALLY into
        train / validation / test segments (default 70/15/15).
          * the Yeo-Johnson transformer is fitted on the TRAIN segment only;
          * the BiLSTM is trained on windows from the TRAIN segment only;
          * the VALIDATION segment is used only to keep the best epoch;
          * the TEST segment is never touched here -- use evaluate().
        Windows never straddle a segment boundary. The split indices are
        stored in self.split so they can be reported in the paper."""
        raw = np.asarray(raw_series, dtype=np.float64).flatten()
        self.split = chronological_split(len(raw), train_frac, val_frac)
        (a0, a1), (b0, b1) = self.split["train"], self.split["val"]
        raw_tr, raw_va = raw[a0:a1], raw[b0:b1]

        tr_t = self.pt.fit_transform(raw_tr.reshape(-1, 1)).flatten()
        va_t = self.pt.transform(raw_va.reshape(-1, 1)).flatten()
        X_tr, y_tr = make_windows(tr_t, self.window)
        X_va, y_va = make_windows(va_t, self.window)
        if len(X_va) == 0 or len(X_tr) == 0:
            raise ValueError("Trace too short for a train/val/test split with this window.")

        Xt = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(-1)
        yt = torch.tensor(y_tr, dtype=torch.float32)
        Xv = torch.tensor(X_va, dtype=torch.float32).unsqueeze(-1)
        yv = torch.tensor(y_va, dtype=torch.float32)

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        n = Xt.shape[0]
        best_val, best_state = float("inf"), None

        for ep in range(epochs):
            self.model.train()
            perm = torch.randperm(n)
            total_loss = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(idx)
            self.model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model(Xv), yv).item()
            if val_loss < best_val:
                best_val, best_state = val_loss, copy.deepcopy(self.model.state_dict())
            if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
                print(f"[BiLSTM] epoch {ep+1}/{epochs}  train_mse={total_loss/n:.5f}  val_mse={val_loss:.5f}")
        self.model.load_state_dict(best_state)
        self.best_val_mse = best_val

        # Clip range / fallback come from the TRAIN segment only (see
        # _safe_inverse for why the clipping exists).
        self._t_lo, self._t_hi = float(tr_t.min()), float(tr_t.max())
        self._fallback_raw = float(np.median(raw_tr))
        return self

    @torch.no_grad()
    def evaluate(self, raw_segment: np.ndarray) -> dict:
        """One-step-ahead MAE/RMSE on a held-out segment (raw utilisation
        units), alongside a persistence baseline (predict the last value of
        each window) on exactly the same windows."""
        seg = np.asarray(raw_segment, dtype=np.float64).flatten()
        preds = self.rolling_forecast(seg)
        actual = seg[self.window:]
        persist = seg[self.window - 1:-1]
        err_m, err_p = preds - actual, persist - actual
        return dict(
            n_windows=int(len(actual)),
            bilstm_mae=float(np.mean(np.abs(err_m))), bilstm_rmse=float(np.sqrt(np.mean(err_m ** 2))),
            persistence_mae=float(np.mean(np.abs(err_p))), persistence_rmse=float(np.sqrt(np.mean(err_p ** 2))),
        )

    def _safe_inverse(self, pred_t: np.ndarray, fallback: float) -> np.ndarray:
        pred_t = np.clip(pred_t, self._t_lo, self._t_hi)
        with np.errstate(all="ignore"):
            out = self.pt.inverse_transform(pred_t.reshape(-1, 1)).flatten()
        bad = ~np.isfinite(out)
        if bad.any():
            out[bad] = fallback
        return out

    @torch.no_grad()
    def predict_next(self, last_window_raw: np.ndarray) -> float:
        """last_window_raw: length == self.window, raw (untransformed) values."""
        self.model.eval()
        raw = np.asarray(last_window_raw, dtype=np.float64).reshape(-1, 1)
        transformed = self.pt.transform(raw).flatten()
        x = torch.tensor(transformed, dtype=torch.float32).view(1, -1, 1)
        pred_t = self.model(x).numpy().reshape(1)
        fallback = float(last_window_raw[-1]) if len(last_window_raw) else self._fallback_raw
        pred_raw = self._safe_inverse(pred_t, fallback)[0]
        return float(np.clip(pred_raw, 0.0, 1.0))

    @torch.no_grad()
    def rolling_forecast(self, raw_series: np.ndarray) -> np.ndarray:
        """Vectorised one-step-ahead forecast over an entire series, used
        for the evaluation plot (predicted vs actual)."""
        self.model.eval()
        raw = np.asarray(raw_series, dtype=np.float64).reshape(-1, 1)
        transformed = self.pt.transform(raw).flatten()
        X, _ = make_windows(transformed, self.window)
        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        preds_t = self.model(Xt).numpy()
        preds_raw = self._safe_inverse(preds_t, self._fallback_raw)
        return np.clip(preds_raw, 0.0, 1.0)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_gen import synthetic_borg_like_trace

    trace = synthetic_borg_like_trace(3000).values
    pipe = LoadForecastingPipeline().fit_split(trace, epochs=10)
    s = pipe.split["test"]
    print("split:", pipe.split)
    print("held-out test metrics:", pipe.evaluate(trace[s[0]:s[1]]))