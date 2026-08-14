"""Detailed evaluation report.

A single averaged MAE hides most of what matters about a forecaster: whether it holds up
across the horizon, whether its uncertainty is honest, whether it degrades in particular
months, and whether the gap to the next model is larger than noise. Each function here
produces one table answering one of those questions; `build_all` writes them to results/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.evaluation import metrics as M
from src.evaluation import statistical_tests as ST

QUANTILE_COLS = [f"q{q:g}" for q in config.QUANTILES]


def _valid(d: pd.DataFrame) -> pd.DataFrame:
    return d.dropna(subset=["actual", "prediction"])


def _has_quantiles(d: pd.DataFrame) -> bool:
    return all(c in d.columns for c in QUANTILE_COLS) and d[QUANTILE_COLS].notna().all(axis=None)


def _quantile_matrix(d: pd.DataFrame) -> np.ndarray:
    """Quantile columns as an array, rearranged to be monotone.

    A safety net: forecasts are rearranged before caching, but scoring a crossed set would
    silently produce a meaningless CRPS, so the guarantee is enforced again at use.
    """
    return M.rearrange_quantiles(d[QUANTILE_COLS].to_numpy(dtype=float))


def crossing_diagnostics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """How badly each model's raw quantile predictions violate monotonicity."""
    rows = []
    for name, d in forecasts.groupby("model"):
        if not _has_quantiles(d):
            continue
        rows.append({"model": name,
                     **M.count_quantile_crossings(d[QUANTILE_COLS].to_numpy(float))})
    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()


# --------------------------------------------------------------------------------------
def headline(forecasts: pd.DataFrame, scale: float, baseline: str) -> pd.DataFrame:
    """Point-forecast metrics per model, with skill against the stronger baseline."""
    rows = []
    for name, d in forecasts.groupby("model"):
        d = _valid(d)
        row = {"model": name, "n_forecasts": len(d)}
        row.update(M.evaluate_forecast(d["actual"], d["prediction"], scale=scale))
        rows.append(row)
    out = pd.DataFrame(rows).set_index("model")
    base_mae = out.loc[baseline, "MAE"]
    base_rmse = out.loc[baseline, "RMSE"]
    out["MAE_skill_%"] = (1.0 - out["MAE"] / base_mae) * 100.0
    out["RMSE_skill_%"] = (1.0 - out["RMSE"] / base_rmse) * 100.0
    return out.sort_values("MAE")


def by_horizon(forecasts: pd.DataFrame, scale: float, baseline: str) -> pd.DataFrame:
    """MAE, RMSE and MASE at each horizon, plus skill against the baseline at that horizon."""
    rows = []
    for (name, h), d in forecasts.groupby(["model", "h"]):
        d = _valid(d)
        rows.append({"model": name, "h": int(h),
                     "MAE": M.mae(d["actual"], d["prediction"]),
                     "RMSE": M.rmse(d["actual"], d["prediction"]),
                     "sMAPE": M.smape(d["actual"], d["prediction"]),
                     "MASE": M.mase(d["actual"], d["prediction"], scale)})
    out = pd.DataFrame(rows)
    base = out[out["model"] == baseline].set_index("h")["MAE"]
    out["MAE_skill_%"] = (1.0 - out["MAE"] / out["h"].map(base)) * 100.0
    return out.sort_values(["model", "h"])


def probabilistic(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Distributional scores: CRPS, Winkler, coverage and width at each interval level."""
    taus = np.asarray(config.QUANTILES, dtype=float)
    rows = []
    for name, d in forecasts.groupby("model"):
        d = d.dropna(subset=["actual"])
        if not _has_quantiles(d):
            continue
        qmat = _quantile_matrix(d)
        row = {
            "model": name,
            "CRPS": M.crps_from_quantiles(d["actual"], qmat, taus),
            "pinball_mean": float(np.mean([
                M.pinball_loss(d["actual"], qmat[:, j], float(t)) for j, t in enumerate(taus)
            ])),
        }
        col_of = {level: j for j, level in enumerate(config.QUANTILES)}
        for lo_q, hi_q, level in config.INTERVAL_LEVELS:
            lo = qmat[:, col_of[lo_q]]
            hi = qmat[:, col_of[hi_q]]
            pct = int(round(level * 100))
            row[f"PICP_{pct}"] = M.picp(d["actual"], lo, hi)
            row[f"width_{pct}"] = M.interval_width(lo, hi)
            row[f"winkler_{pct}"] = M.winkler_score(d["actual"], lo, hi, alpha=1.0 - level)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()


def calibration(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Empirical exceedance rate at each nominal decile — the reliability diagram as a table."""
    taus = np.asarray(config.QUANTILES, dtype=float)
    rows = []
    for name, d in forecasts.groupby("model"):
        d = d.dropna(subset=["actual"])
        if not _has_quantiles(d):
            continue
        emp = M.quantile_calibration(d["actual"], _quantile_matrix(d), taus)
        for tau, observed in emp.items():
            rows.append({"model": name, "nominal": tau, "empirical": observed,
                         "deviation": observed - tau})
    return pd.DataFrame(rows)


def coverage_by_horizon(forecasts: pd.DataFrame) -> pd.DataFrame:
    """80% interval coverage at each horizon — does calibration hold as the horizon grows?"""
    rows = []
    for (name, h), d in forecasts.groupby(["model", "h"]):
        d = d.dropna(subset=["actual"])
        if not _has_quantiles(d):
            continue
        qmat = _quantile_matrix(d)
        lo, hi = qmat[:, 0], qmat[:, -1]
        rows.append({"model": name, "h": int(h),
                     "PICP_80": M.picp(d["actual"], lo, hi),
                     "width_80": M.interval_width(lo, hi)})
    return pd.DataFrame(rows)


def by_month(forecasts: pd.DataFrame) -> pd.DataFrame:
    """MAE and bias per calendar month — where in the test period each model struggles."""
    d = _valid(forecasts).copy()
    d["month"] = pd.DatetimeIndex(d["target_time"]).tz_localize(None).to_period("M").astype(str)
    d["abs_err"] = (d["actual"] - d["prediction"]).abs()
    d["err"] = d["actual"] - d["prediction"]
    g = d.groupby(["model", "month"])
    return pd.DataFrame({"MAE": g["abs_err"].mean(), "bias": g["err"].mean(),
                         "n": g["err"].size()}).reset_index()


def peak_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Accuracy on the daily peak — the quantity that actually sizes generation reserve.

    For each 24-hour window: how far off was the predicted peak level, and did the model
    put the peak in the right hour? Average error can look fine while peak timing is wrong,
    and it is the peak that determines whether capacity is adequate.
    """
    d = _valid(forecasts)
    rows = []
    for name, sub in d.groupby("model"):
        g = sub.groupby("origin")
        peak_actual = g["actual"].max()
        peak_pred = g["prediction"].max()
        hour_actual = g.apply(lambda x: x.loc[x["actual"].idxmax(), "h"], include_groups=False)
        hour_pred = g.apply(lambda x: x.loc[x["prediction"].idxmax(), "h"], include_groups=False)
        hour_err = (hour_pred - hour_actual).abs()
        rows.append({
            "model": name,
            "peak_MAE": float((peak_pred - peak_actual).abs().mean()),
            # Signed as actual - predicted, matching `by_month`: positive = under-forecast.
            "peak_bias": float((peak_actual - peak_pred).mean()),
            "peak_MAPE_%": float(((peak_pred - peak_actual).abs() / peak_actual).mean() * 100),
            "peak_hour_exact_%": float((hour_err == 0).mean() * 100),
            "peak_hour_within_1h_%": float((hour_err <= 1).mean() * 100),
            "peak_hour_MAE_h": float(hour_err.mean()),
        })
    return pd.DataFrame(rows).set_index("model").sort_values("peak_MAE")


def by_time_of_day(forecasts: pd.DataFrame) -> pd.DataFrame:
    """MAE by target hour of day — which parts of the load curve are hard."""
    d = _valid(forecasts).copy()
    d["target_hour"] = pd.DatetimeIndex(d["target_time"]).hour
    d["abs_err"] = (d["actual"] - d["prediction"]).abs()
    return (d.groupby(["model", "target_hour"])["abs_err"].mean()
            .reset_index().rename(columns={"abs_err": "MAE"}))


# --------------------------------------------------------------------------------------
def build_all(forecasts: pd.DataFrame, scale: float, baseline: str,
              verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Compute every table and write it to results/."""
    tables = {
        "summary_detailed": headline(forecasts, scale, baseline),
        "error_by_horizon_detailed": by_horizon(forecasts, scale, baseline),
        "probabilistic": probabilistic(forecasts),
        "calibration": calibration(forecasts),
        "coverage_by_horizon": coverage_by_horizon(forecasts),
        "metrics_by_month": by_month(forecasts),
        "peak_metrics": peak_metrics(forecasts),
        "error_by_time_of_day": by_time_of_day(forecasts),
        "significance_dm": ST.pairwise_dm_table(forecasts),
        "quantile_crossing": crossing_diagnostics(forecasts),
    }
    for name, df in tables.items():
        if df is None or df.empty:
            continue
        path = config.RESULTS_DIR / f"{name}.csv"
        df.to_csv(path, index=not isinstance(df.index, pd.RangeIndex))
        if verbose:
            print(f"  wrote {path.name:<34} ({len(df)} rows)")
    return tables
