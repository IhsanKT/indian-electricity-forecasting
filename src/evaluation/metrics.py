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


def winkler_score(actual, lower, upper, alpha: float = 0.2) -> float:
    """Winkler (interval) score for a central (1-alpha) interval. Lower is better.

    PICP alone is gameable: an interval spanning zero to infinity scores perfect coverage
    and says nothing. Winkler charges the width of the interval and adds a penalty of
    2/alpha times the shortfall whenever the actual lands outside, so it rewards the
    narrowest interval that still covers honestly.
    """
    a, lo = _as_arrays(actual, lower)
    _, hi = _as_arrays(actual, upper)
    width = hi - lo
    below = a < lo
    above = a > hi
    penalty = np.zeros_like(width)
    penalty[below] = (2.0 / alpha) * (lo[below] - a[below])
    penalty[above] = (2.0 / alpha) * (a[above] - hi[above])
    return float(np.mean(width + penalty))


def crps_from_quantiles(actual, quantile_preds, taus) -> float:
    """CRPS approximated from a discrete set of predictive quantiles. Lower is better.

    CRPS integrates the pinball loss over all quantile levels. With a finite grid the
    standard approximation is 2 * the mean pinball loss across the available levels, which
    is exact in the limit of a dense grid. Nine deciles is a reasonable grid; with only
    three it would be too coarse to trust, which is why the whole pipeline emits deciles.

    Unlike PICP or Winkler this scores the entire predictive distribution rather than one
    interval, so models cannot look good by tuning a single band.
    """
    a = np.asarray(actual, dtype=float).ravel()
    taus = np.asarray(taus, dtype=float).ravel()
    q = np.asarray(quantile_preds, dtype=float)
    if q.ndim != 2 or q.shape[1] != taus.size:
        raise ValueError(f"quantile_preds must be (n, {taus.size}), got {q.shape}")
    if q.shape[0] != a.size:
        raise ValueError(f"shape mismatch: actual {a.size} vs quantile rows {q.shape[0]}")
    losses = [pinball_loss(a, q[:, j], float(t)) for j, t in enumerate(taus)]
    return float(2.0 * np.mean(losses))


def rearrange_quantiles(quantile_preds) -> np.ndarray:
    """Sort each row's predicted quantiles into non-decreasing order.

    Fitting one model per quantile level gives no guarantee of monotonicity, so estimates
    routinely cross: a predicted 40th percentile above the predicted 60th. A crossed set is
    not a valid distribution, and it corrupts CRPS, coverage and interval width.

    Row-wise sorting is the standard rearrangement fix (Chernozhukov, Fernandez-Val &
    Galichon, 2010), which is guaranteed to weakly *reduce* estimation error against the
    true monotone quantile function -- it can only help the model it is applied to.

    TimesFM needs none of this: it is compiled with `fix_quantile_crossing=True` and its
    output is already monotone.
    """
    q = np.asarray(quantile_preds, dtype=float)
    if q.ndim != 2:
        raise ValueError(f"expected a 2-D (n, n_quantiles) array, got shape {q.shape}")
    return np.sort(q, axis=1)


def count_quantile_crossings(quantile_preds) -> dict[str, float]:
    """Diagnose how badly a set of quantile predictions violates monotonicity."""
    q = np.asarray(quantile_preds, dtype=float)
    diffs = np.diff(q, axis=1)
    rows = (diffs < 0).any(axis=1)
    return {
        "rows_with_crossing": int(rows.sum()),
        "fraction_of_rows": float(rows.mean()) if rows.size else 0.0,
        "max_inversion": float(np.abs(np.minimum(diffs, 0)).max()) if diffs.size else 0.0,
    }


def quantile_calibration(actual, quantile_preds, taus) -> dict[float, float]:
    """Empirical exceedance rate at each nominal quantile level.

    For a perfectly calibrated model the fraction of actuals falling at or below the
    predicted tau-quantile is tau. Deviations show *where* a model is miscalibrated -- a
    single coverage number cannot distinguish a tail problem from a systematic shift.
    """
    a = np.asarray(actual, dtype=float).ravel()
    q = np.asarray(quantile_preds, dtype=float)
    taus = np.asarray(taus, dtype=float).ravel()
    return {float(t): float(np.mean(a <= q[:, j])) for j, t in enumerate(taus)}
