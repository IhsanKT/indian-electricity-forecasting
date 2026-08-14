"""LightGBM tier — the comparison that makes the project defensible.

Without a properly tuned local model, a foundation-model win says nothing about whether
something simpler would have won by more.

Strategy is **direct multi-horizon**: one independent model per h = 1..24, each predicting
y(T+h) from features known at T. The alternative -- recursive one-step forecasting fed its
own output -- accumulates error over 24 steps and would understate this tier unfairly.
Direct models also make the error-by-horizon curve fall out for free.
"""
from __future__ import annotations

import time
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.models import features as F

#: Starting point; overridden by whatever the validation sweep picks.
BASE_PARAMS: dict = {
    "objective": "regression",
    "metric": "l1",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "n_estimators": 400,
    "n_jobs": 6,
    "verbose": -1,
    "seed": config.RANDOM_SEED,
}

#: Deliberately small — the spec asks for light tuning, not an exhaustive search.
#:
#: The first sweep ran 31/63/127 leaves and validation error rose monotonically with
#: capacity, so the best config sat on the edge of the grid. Reporting a tuned baseline
#: whose optimum is at a grid boundary is undertuning, so the grid was extended downward
#: to 7/15 leaves to bracket the minimum properly. More capacity overfits here: the useful
#: signal is a smooth daily profile, not fine interactions.
PARAM_GRID: tuple[dict, ...] = (
    {"num_leaves": 7, "learning_rate": 0.05},
    {"num_leaves": 15, "learning_rate": 0.05},
    {"num_leaves": 31, "learning_rate": 0.05},
    {"num_leaves": 63, "learning_rate": 0.05},
    {"num_leaves": 127, "learning_rate": 0.05},
    {"num_leaves": 15, "learning_rate": 0.03, "n_estimators": 800},
    {"num_leaves": 31, "learning_rate": 0.03, "n_estimators": 800},
    {"num_leaves": 15, "learning_rate": 0.10},
)

#: Horizons sampled during tuning. Fitting all 24 for every grid point is wasteful; these
#: five span the range, and one config is then used for all 24.
TUNING_HORIZONS: tuple[int, ...] = (1, 6, 12, 18, 24)


class LightGBMForecaster:
    """Direct multi-horizon LightGBM. One booster per horizon (per quantile, if asked)."""

    def __init__(self, horizon: int = config.HORIZON, params: dict | None = None,
                 quantile: float | None = None) -> None:
        self.horizon = int(horizon)
        self.params = {**BASE_PARAMS, **(params or {})}
        self.quantile = quantile
        if quantile is not None:
            self.params = {**self.params, "objective": "quantile", "alpha": float(quantile),
                           "metric": "quantile"}
        self.name = "lightgbm" if quantile is None else f"lightgbm_q{quantile:g}"
        self.models_: dict[int, lgb.LGBMRegressor] = {}

    @property
    def min_context(self) -> int:
        """Shortest history this model can forecast from."""
        return F.MAX_LAG + 1

    def fit(self, y_full: pd.Series, train_index: pd.DatetimeIndex,
            verbose: bool = False) -> "LightGBMForecaster":
        """Fit one booster per horizon.

        `y_full` is the whole series (lags near the start of a split need earlier values);
        `train_index` restricts which **origins** are trained on. Feature values do not
        depend on data after the origin, so passing the full series introduces no leakage --
        that property is asserted in tests/test_leakage.py.
        """
        for h in range(1, self.horizon + 1):
            X, target = F.training_frame(y_full, h)
            mask = X.index.isin(train_index)
            model = lgb.LGBMRegressor(**self.params)
            model.fit(X[mask], target[mask])
            self.models_[h] = model
            if verbose:
                print(f"    h={h:>2} fitted on {int(mask.sum()):,} origins", flush=True)
        return self

    def forecast(self, history: pd.Series, horizon: int | None = None) -> np.ndarray:
        """Forecast horizon steps beyond the end of `history`."""
        h_max = int(horizon or self.horizon)
        if not self.models_:
            raise RuntimeError("model is not fitted")
        preds = np.empty(h_max, dtype=float)
        for h in range(1, h_max + 1):
            row = F.features_at_origin(history, h)
            preds[h - 1] = float(self.models_[h].predict(row)[0])
        return preds

    def forecast_batch(self, y_full: pd.Series, origins: Iterable[pd.Timestamp],
                       horizon: int | None = None) -> np.ndarray:
        """Vectorised forecasts for many origins at once.

        Predicting origin-by-origin means rebuilding the whole feature frame per origin;
        here the frame is built once per horizon and the needed rows are selected. Values
        are identical -- features at T never depend on data after T.
        """
        h_max = int(horizon or self.horizon)
        origins = pd.DatetimeIndex(origins)
        out = np.full((len(origins), h_max), np.nan, dtype=float)
        for h in range(1, h_max + 1):
            # Built without training_frame because that drops rows with an unavailable
            # target; for inference only the features need to exist.
            base = F.origin_features(y_full)
            cal = F.calendar_features(y_full.index + pd.Timedelta(hours=h))
            cal.index = y_full.index
            Xf = pd.concat([base, cal], axis=1)[list(F.FEATURE_NAMES)]
            rows = Xf.reindex(origins)
            valid = rows.notna().all(axis=1).to_numpy()
            if valid.any():
                out[valid, h - 1] = self.models_[h].predict(rows[valid])
        return out


def tune(y_full: pd.Series, train_index: pd.DatetimeIndex, val_index: pd.DatetimeIndex,
         verbose: bool = True) -> tuple[dict, list[dict]]:
    """Small grid search on the validation split. Returns (best params, full record)."""
    records: list[dict] = []
    for i, override in enumerate(PARAM_GRID, 1):
        t0 = time.time()
        maes: list[float] = []
        for h in TUNING_HORIZONS:
            X, target = F.training_frame(y_full, h)
            tr = X.index.isin(train_index)
            va = X.index.isin(val_index)
            model = lgb.LGBMRegressor(**{**BASE_PARAMS, **override})
            model.fit(X[tr], target[tr])
            pred = model.predict(X[va])
            maes.append(float(np.mean(np.abs(target[va].to_numpy() - pred))))
        mean_mae = float(np.mean(maes))
        records.append({"params": override, "val_mae": mean_mae,
                        "per_horizon_mae": dict(zip(TUNING_HORIZONS, maes))})
        if verbose:
            print(f"  [{i}/{len(PARAM_GRID)}] {override} -> val MAE {mean_mae:,.1f} MW "
                  f"({time.time()-t0:.1f}s)", flush=True)
    best = min(records, key=lambda r: r["val_mae"])
    if verbose:
        print(f"  best: {best['params']}  (val MAE {best['val_mae']:,.1f} MW)")
    return best["params"], records
