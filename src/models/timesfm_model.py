"""TimesFM 2.5 (200M) zero-shot wrapper.

Zero-shot means exactly that: no fitting, no fine-tuning, no exposure to Indian demand
beyond whatever was in Google's pretraining corpus. `fit()` exists only so the model can be
swapped with the LightGBM tier behind one interface.

The API below was checked against the current model card for
`google/timesfm-2.5-200m-pytorch` rather than copied from tutorials, which are frequently
stale for this package.

Quantile output is documented as (batch, horizon, 10): index 0 is the mean and indices 1..9
are the deciles q10..q90. `_verify_quantile_layout` asserts that at load time rather than
trusting the docs -- silently reading the wrong column would corrupt every interval metric
in Phase 7.
"""
from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import pandas as pd

from src import config

#: Position of each quantile in the model's quantile output (index 0 is the mean).
_QUANTILE_COLUMN = {0.1: 1, 0.2: 2, 0.3: 3, 0.4: 4, 0.5: 5,
                    0.6: 6, 0.7: 7, 0.8: 8, 0.9: 9}

#: Loading and compiling costs ~3.5 minutes, and the context-length sweep would otherwise
#: pay it once per context. The compiled model is built with max_context=1024 and is
#: independent of how much history we choose to feed it, so one instance serves them all.
_MODEL_CACHE: dict[str, object] = {}


class TimesFMForecaster:
    """Zero-shot TimesFM 2.5, univariate, CPU."""

    def __init__(self, context_length: int = config.DEFAULT_CONTEXT,
                 checkpoint: str = config.TIMESFM_CHECKPOINT,
                 horizon: int = config.HORIZON,
                 batch_size: int = 32) -> None:
        self.context_length = int(context_length)
        self.checkpoint = checkpoint
        self.horizon = int(horizon)
        self.batch_size = int(batch_size)
        self.name = f"timesfm_ctx{self.context_length}"
        self._model = None

    @property
    def min_context(self) -> int:
        """Shortest history this model can forecast from."""
        return self.context_length

    # ----------------------------------------------------------------------------------
    def load(self):
        """Load and compile the checkpoint. Idempotent and lazy — loading costs ~30s."""
        if self._model is not None:
            return self._model
        if self.checkpoint in _MODEL_CACHE:
            self._model = _MODEL_CACHE[self.checkpoint]
            return self._model
        import timesfm

        t0 = time.time()
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.checkpoint)
        model.compile(
            timesfm.ForecastConfig(
                max_context=config.TIMESFM_MAX_CONTEXT,
                max_horizon=config.TIMESFM_MAX_HORIZON,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                fix_quantile_crossing=True,
            )
        )
        self._model = model
        print(f"  loaded {self.checkpoint} in {time.time()-t0:.1f}s")
        self._verify_quantile_layout()
        _MODEL_CACHE[self.checkpoint] = model
        return model

    def _verify_quantile_layout(self) -> None:
        """Assert the quantile axis really is [mean, q10..q90] before trusting it."""
        probe = np.linspace(100.0, 200.0, 512) + np.sin(np.arange(512) / 3.0) * 5.0
        point, quant = self._model.forecast(horizon=8, inputs=[probe])
        quant = np.asarray(quant)
        if quant.ndim != 3 or quant.shape[2] != 10:
            raise RuntimeError(f"unexpected quantile shape {quant.shape}; expected (b, h, 10)")
        q10, q50, q90 = quant[0, :, 1], quant[0, :, 5], quant[0, :, 9]
        if not (np.all(q10 <= q50 + 1e-6) and np.all(q50 <= q90 + 1e-6)):
            raise RuntimeError("quantile columns are not ordered q10 <= q50 <= q90")
        if not np.allclose(np.asarray(point)[0], quant[0, :, 0], rtol=0.05, atol=1.0):
            raise RuntimeError("column 0 of the quantile output is not the point forecast")

    def fit(self, *args, **kwargs) -> "TimesFMForecaster":  # noqa: ARG002
        """No-op. The model is evaluated strictly zero-shot."""
        return self

    # ----------------------------------------------------------------------------------
    def forecast(self, history: pd.Series | np.ndarray,
                 horizon: int | None = None) -> np.ndarray:
        """Point forecast for one window."""
        point, _ = self.forecast_with_quantiles([history], horizon)
        return point[0]

    def forecast_with_quantiles(self, histories: Iterable, horizon: int | None = None
                                ) -> tuple[np.ndarray, np.ndarray]:
        """Batched forecast. Returns (point (n, h), quantiles (n, h, n_quantiles))."""
        model = self.load()
        h = int(horizon or self.horizon)
        inputs = [np.asarray(x, dtype=np.float32).ravel()[-self.context_length:]
                  for x in histories]

        points: list[np.ndarray] = []
        quants: list[np.ndarray] = []
        for i in range(0, len(inputs), self.batch_size):
            chunk = inputs[i: i + self.batch_size]
            p, q = model.forecast(horizon=h, inputs=chunk)
            points.append(np.asarray(p, dtype=float))
            quants.append(np.asarray(q, dtype=float))

        point = np.concatenate(points, axis=0)
        quant_full = np.concatenate(quants, axis=0)
        cols = [_QUANTILE_COLUMN[q] for q in config.QUANTILES]
        return point, quant_full[:, :, cols]

    def forecast_windows(self, y: pd.Series, origins: pd.DatetimeIndex,
                         horizon: int = config.HORIZON) -> tuple[np.ndarray, np.ndarray]:
        """Rolling-origin interface: every origin's context sliced and forecast in batches."""
        values = y.to_numpy(dtype=float)
        positions = y.index.get_indexer(pd.DatetimeIndex(origins))
        if (positions < 0).any():
            raise ValueError("some origins are not present in the series index")

        histories = []
        for p in positions:
            start = p + 1 - self.context_length
            if start < 0:
                raise ValueError(
                    f"origin at position {p} has only {p+1} history points, "
                    f"needs {self.context_length}"
                )
            histories.append(values[start: p + 1])
        return self.forecast_with_quantiles(histories, horizon)
