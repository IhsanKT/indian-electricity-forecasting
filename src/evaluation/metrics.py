"""Forecast accuracy metrics.

MASE needs a scale. Ours is the in-sample MAE of the *stronger* seasonal naive, computed
once on the training split and then frozen and applied identically to every model. An
unstated or per-model denominator makes MASE meaningless, so the value used is recorded in
results and quoted in the README.
"""
from __future__ import annotations

import numpy as np


def _as_arrays(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    if a.shape != p.shape:
        raise ValueError(f"shape mismatch: actual {a.shape} vs predicted {p.shape}")
    if a.size == 0:
        raise ValueError("empty input")
    return a, p


def mae(actual, predicted) -> float:
    """Mean absolute error."""
    a, p = _as_arrays(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual, predicted) -> float:
    """Root mean squared error."""
    a, p = _as_arrays(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def smape(actual, predicted) -> float:
    """Symmetric MAPE as a percentage, using the 0-200% convention.

    sMAPE = mean( 200 * |F - A| / (|A| + |F|) ). Terms where both are zero contribute zero.
    """
    a, p = _as_arrays(actual, predicted)
    denom = np.abs(a) + np.abs(p)
    ratio = np.divide(np.abs(a - p), denom, out=np.zeros_like(denom), where=denom != 0)
    return float(np.mean(200.0 * ratio))


def mape(actual, predicted) -> float:
    """Mean absolute percentage error. Undefined where actual is zero; those are skipped."""
    a, p = _as_arrays(actual, predicted)
    ok = a != 0
    if not ok.any():
        return float("nan")
    return float(np.mean(np.abs((a[ok] - p[ok]) / a[ok])) * 100.0)


def seasonal_naive_scale(train_series, seasonal_period: int) -> float:
    """In-sample MAE of the seasonal naive on the training data — the MASE denominator.

    This is mean(|y_t - y_{t-m}|) over the training split only. Computing it on test data
    would leak; recomputing it per model would make MASE incomparable across models.
    """
    y = np.asarray(train_series, dtype=float).ravel()
    m = int(seasonal_period)
    if y.size <= m:
        raise ValueError(f"need more than {m} observations to scale MASE, got {y.size}")
    diffs = np.abs(y[m:] - y[:-m])
    diffs = diffs[~np.isnan(diffs)]
    scale = float(np.mean(diffs))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("seasonal naive scale is zero or non-finite")
    return scale


def mase(actual, predicted, scale: float) -> float:
    """Mean absolute scaled error against a precomputed in-sample seasonal naive scale."""
    return mae(actual, predicted) / float(scale)


def evaluate_forecast(actual, predicted, scale: float | None = None) -> dict[str, float]:
    """Headline metric bundle for one set of forecast/actual pairs.

    `scale` is the frozen MASE denominator from `seasonal_naive_scale`. If omitted, MASE is
    returned as NaN rather than silently substituting a different scale.
    """
    out = {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "sMAPE": smape(actual, predicted),
        "MAPE": mape(actual, predicted),
    }
    out["MASE"] = mase(actual, predicted, scale) if scale is not None else float("nan")
    return out


# --------------------------------------------------------------------------------------
# Probabilistic metrics (Phase 7)
# --------------------------------------------------------------------------------------
def pinball_loss(actual, predicted_quantile, tau: float) -> float:
    """Pinball (quantile) loss at level tau. Lower is better."""
    a, p = _as_arrays(actual, predicted_quantile)
    diff = a - p
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def picp(actual, lower, upper) -> float:
    """Prediction Interval Coverage Probability: the fraction of actuals inside [lower, upper].

    An 80% nominal interval is *expected* to contain the actual about 80% of the time, not
    always. Coverage far below nominal means the interval is too narrow (overconfident);
    far above means it is too wide (uninformative).
    """
    a, lo = _as_arrays(actual, lower)
    _, hi = _as_arrays(actual, upper)
    return float(np.mean((a >= lo) & (a <= hi)))


def interval_width(lower, upper) -> float:
    """Mean width of the prediction interval, in the units of the series (MW)."""
    lo, hi = _as_arrays(lower, upper)
    return float(np.mean(hi - lo))
