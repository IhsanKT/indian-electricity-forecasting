"""Rolling-origin evaluation.

    context window -> forecast next 24h -> step the origin forward -> repeat

Origins step by `config.ORIGIN_STRIDE_HOURS` (6h) rather than 24h so that all four times of
day appear as forecast origins. Stepping by a whole day would measure skill at one clock
hour only and could flatter or punish a model by accident.

The output is long-format -- one row per (origin, horizon) -- because that is what both the
error-by-horizon curve and the dashboard need, and it keeps actual and forecast welded to
the same target timestamp instead of relying on positional alignment.
"""
from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src import config


def make_origins(y: pd.Series, start=None, end=None,
                 stride_hours: int = config.ORIGIN_STRIDE_HOURS,
                 min_context: int = 336, horizon: int = config.HORIZON) -> pd.DatetimeIndex:
    """Forecast origins inside [start, end] that have enough history and a full horizon.

    An origin T is usable only when y holds at least `min_context` observations up to T and
    all of T+1 .. T+horizon exist -- otherwise the window would be scored against a partial
    actual.
    """
    idx = y.index
    lo = idx[min_context - 1]
    hi = idx[-horizon - 1]
    if start is not None:
        lo = max(lo, pd.Timestamp(start))
    if end is not None:
        hi = min(hi, pd.Timestamp(end))
    candidates = idx[(idx >= lo) & (idx <= hi)]
    return pd.DatetimeIndex(candidates[::stride_hours])


def _forecast_all(model: Any, y: pd.Series, origins: pd.DatetimeIndex,
                  horizon: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Point forecasts (n_origins, horizon) plus optional quantiles (n, horizon, n_q)."""
    if hasattr(model, "forecast_windows"):
        return model.forecast_windows(y, origins, horizon)
    # Generic fallback: one call per origin, history sliced up to and including T.
    point = np.full((len(origins), horizon), np.nan)
    pos = y.index.get_indexer(origins)
    values = y.to_numpy()
    for i, p in enumerate(pos):
        point[i] = model.forecast(pd.Series(values[: p + 1], index=y.index[: p + 1]), horizon)
    return point, None


def rolling_forecast(model: Any, y: pd.Series, origins: pd.DatetimeIndex,
                     horizon: int = config.HORIZON, model_name: str | None = None,
                     verbose: bool = True) -> pd.DataFrame:
    """Run one model across all origins and return long-format forecasts vs actuals."""
    name = model_name or getattr(model, "name", type(model).__name__)
    t0 = time.time()
    point, quantiles = _forecast_all(model, y, origins, horizon)
    elapsed = time.time() - t0

    n = len(origins)
    h_index = np.arange(1, horizon + 1)
    origin_col = np.repeat(np.asarray(origins), horizon)
    h_col = np.tile(h_index, n)
    target_times = pd.DatetimeIndex(origin_col) + pd.to_timedelta(h_col, unit="h")

    # Look the actual up BY TIMESTAMP, never by position. Positional alignment is exactly
    # how an off-by-one silently produces plausible-looking metrics.
    actual = y.reindex(target_times).to_numpy()

    out = pd.DataFrame({
        "model": name,
        "origin": origin_col,
        "h": h_col,
        "target_time": target_times,
        "actual": actual,
        "prediction": point.reshape(-1),
    })
    if quantiles is not None:
        for j, q in enumerate(config.QUANTILES):
            out[f"q{q:g}"] = quantiles[:, :, j].reshape(-1)

    if verbose:
        rate = n / elapsed if elapsed > 0 else float("inf")
        print(f"  {name:<28} {n:>5,} origins in {elapsed:>7.1f}s ({rate:>6.1f} origins/s)",
              flush=True)
    return out


def error_by_horizon(forecasts: pd.DataFrame) -> pd.DataFrame:
    """MAE and RMSE at each horizon h = 1..24, per model.

    Accuracy degrades with horizon; reporting the curve rather than one averaged number
    shows the structure of the problem.
    """
    d = forecasts.dropna(subset=["actual", "prediction"]).copy()
    d["err"] = d["actual"] - d["prediction"]
    g = d.groupby(["model", "h"])
    return pd.DataFrame({
        "MAE": g["err"].apply(lambda e: float(np.mean(np.abs(e)))),
        "RMSE": g["err"].apply(lambda e: float(np.sqrt(np.mean(e ** 2)))),
    }).reset_index()


def summarise(forecasts: pd.DataFrame, scale: float,
              baseline_model: str | None = None) -> pd.DataFrame:
    """Headline metrics per model, with % improvement over the stronger baseline."""
    from src.evaluation import metrics as M

    rows = []
    for name, d in forecasts.groupby("model"):
        d = d.dropna(subset=["actual", "prediction"])
        row = {"model": name, "n_forecasts": int(len(d))}
        row.update(M.evaluate_forecast(d["actual"], d["prediction"], scale=scale))
        rows.append(row)
    out = pd.DataFrame(rows).set_index("model")

    if baseline_model is not None and baseline_model in out.index:
        base_mae = out.loc[baseline_model, "MAE"]
        out["improvement_vs_baseline_%"] = (1.0 - out["MAE"] / base_mae) * 100.0
    return out.sort_values("MAE")
