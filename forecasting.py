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


class LoadForecastingPipeline:
    """Bundles the Yeo-Johnson transformer with the BiLSTM so a caller can
    do: pipeline.fit(trace); pipeline.predict_next(last_50_raw_values)."""

    def __init__(self, window: int = WINDOW, hidden_size: int = 32, device: str = "cpu"):
        self.window = window
        self.pt = PowerTransformer(method="yeo-johnson")
        self.model = BiLSTMForecaster(hidden_size=hidden_size)
        self.device = device

    def fit(self, raw_series: np.ndarray, epochs: int = 15, lr: float = 1e-3,
            batch_size: int = 64, val_frac: float = 0.15, verbose: bool = True):
        raw = np.asarray(raw_series, dtype=np.float64).reshape(-1, 1)
        transformed = self.pt.fit_transform(raw).flatten()

        X, y = make_windows(transformed, self.window)
        n_val = int(len(X) * val_frac)
        X_train, y_train = X[:-n_val], y[:-n_val]
        X_val, y_val = X[-n_val:], y[-n_val:]

        Xt = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
        yt = torch.tensor(y_train, dtype=torch.float32)
        Xv = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
        yv = torch.tensor(y_val, dtype=torch.float32)

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        n = Xt.shape[0]

        for ep in range(epochs):
            self.model.train()
            perm = torch.randperm(n)
            total_loss = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                pred = self.model(Xt[idx])
                loss = loss_fn(pred, yt[idx])
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(idx)
            if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(Xv)
                    val_loss = loss_fn(val_pred, yv).item()
                print(f"[BiLSTM] epoch {ep+1}/{epochs}  train_mse={total_loss/n:.5f}  val_mse={val_loss:.5f}")

        # Yeo-Johnson's inverse_transform can return NaN when fed a value
        # outside the domain the fitted lambda supports (this happens in
        # practice on real, heavy-tailed traces whenever the network's
        # prediction overshoots). Remember the transformed-space range we
        # actually trained on so predict_next/rolling_forecast can clip
        # into it before inverting -- this is the fix, not a band-aid on
        # the symptom, because it keeps every prediction inside the
        # region PowerTransformer's inverse is well-defined on.
        self._t_lo, self._t_hi = float(transformed.min()), float(transformed.max())
        self._fallback_raw = float(np.median(raw))
        return self

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
    pipe = LoadForecastingPipeline().fit(trace, epochs=10)
    preds = pipe.rolling_forecast(trace)
    actual = trace[WINDOW:]
    mae = np.mean(np.abs(preds - actual))
    print(f"Rolling forecast MAE over full series: {mae:.4f}")
