"""Significance testing for forecast comparisons.

A difference in MAE is not evidence on its own. With 832 origins each producing a 24-hour
window at a 6-hour stride, consecutive windows *share target hours* -- origin i and origins
i+1..i+3 all forecast some of the same timestamps. Those loss differentials are strongly
autocorrelated, so treating them as independent would shrink the standard error and turn
noise into significance.

The Diebold-Mariano test with a Newey-West (HAC) variance estimator handles exactly this.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def newey_west_variance(d: np.ndarray, max_lag: int | None = None) -> float:
    """Long-run variance of a serially correlated series (Newey-West, Bartlett kernel).

    Overlapping forecast windows make the naive variance far too small. Bartlett weights
    keep the estimate positive semi-definite.
    """
    d = np.asarray(d, dtype=float).ravel()
    n = d.size
    if max_lag is None:
        # Standard rule of thumb; also covers the overlap induced by the window structure.
        max_lag = int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    max_lag = max(0, min(int(max_lag), n - 2))

    d_centred = d - d.mean()
    gamma0 = float(np.dot(d_centred, d_centred) / n)
    total = gamma0
    for lag in range(1, max_lag + 1):
        cov = float(np.dot(d_centred[lag:], d_centred[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1.0)
        total += 2.0 * weight * cov
    # A HAC estimate can go negative in small samples; fall back to the contemporaneous term.
    return total if total > 0 else gamma0


def diebold_mariano(actual, pred_a, pred_b, power: int = 1,
                    max_lag: int | None = None,
                    harvey_correction: bool = True) -> dict[str, float]:
    """Test whether model A's forecast loss differs significantly from model B's.

    H0: the two models have equal expected loss. A negative statistic favours model A
    (lower loss); the p-value is two-sided.

    `power=1` compares absolute errors (matching MAE), `power=2` squared errors (RMSE).
    The Harvey-Leybourne-Newbold small-sample correction is applied by default -- the
    uncorrected statistic over-rejects at these sample sizes.
    """
    a = np.asarray(actual, dtype=float).ravel()
    pa = np.asarray(pred_a, dtype=float).ravel()
    pb = np.asarray(pred_b, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(pa) & np.isfinite(pb)
    a, pa, pb = a[ok], pa[ok], pb[ok]

    loss_a = np.abs(a - pa) ** power
    loss_b = np.abs(a - pb) ** power
    d = loss_a - loss_b
    n = d.size
    if n < 10:
        raise ValueError(f"too few paired observations for a DM test: {n}")

    var = newey_west_variance(d, max_lag=max_lag)
    stat = float(d.mean() / math.sqrt(var / n))

    used_lag = max_lag if max_lag is not None else int(
        math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    if harvey_correction:
        h = used_lag + 1
        adj = (n + 1 - 2 * h + h * (h - 1) / n) / n
        stat *= math.sqrt(max(adj, 1e-12))

    p_value = float(2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1)))
    return {
        "dm_stat": stat,
        "p_value": p_value,
        "mean_loss_diff": float(d.mean()),
        "n_pairs": int(n),
        "hac_lag": int(used_lag),
        "favours": "A" if d.mean() < 0 else "B",
    }


def pairwise_dm_table(forecasts: pd.DataFrame, models: list[str] | None = None,
                      power: int = 1, origin_level: bool = True) -> pd.DataFrame:
    """Diebold-Mariano for every model pair, on a common set of target timestamps.

    With `origin_level=True` the 24 hourly errors within each window are averaged into one
    loss per origin first. That is the conservative choice: it avoids counting a single
    window as 24 independent observations, and leaves only the milder overlap *between*
    windows for the HAC estimator to absorb.
    """
    models = models or sorted(forecasts["model"].unique())
    wide = forecasts.pivot_table(index=["origin", "h"], columns="model",
                                 values="prediction", aggfunc="first")
    actual = forecasts.pivot_table(index=["origin", "h"], columns="model",
                                   values="actual", aggfunc="first").iloc[:, 0]
    frame = wide.join(actual.rename("actual"))
    frame = frame.dropna(subset=["actual"] + [m for m in models if m in frame.columns])

    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            if a not in frame.columns or b not in frame.columns:
                continue
            if origin_level:
                g = frame.reset_index().groupby("origin")
                la = g.apply(lambda d: np.mean(np.abs(d["actual"] - d[a]) ** power),
                             include_groups=False)
                lb = g.apply(lambda d: np.mean(np.abs(d["actual"] - d[b]) ** power),
                             include_groups=False)
                d = (la - lb).to_numpy()
                n = d.size
                var = newey_west_variance(d)
                lag = int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
                stat = float(d.mean() / math.sqrt(var / n))
                h = lag + 1
                stat *= math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
                p = float(2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1)))
                res = {"dm_stat": stat, "p_value": p, "mean_loss_diff": float(d.mean()),
                       "n_pairs": n, "hac_lag": lag,
                       "favours": "A" if d.mean() < 0 else "B"}
            else:
                res = diebold_mariano(frame["actual"], frame[a], frame[b], power=power)
            rows.append({"model_A": a, "model_B": b,
                         "winner": a if res["favours"] == "A" else b, **res})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["significant_5pct"] = out["p_value"] < 0.05
        out["significant_1pct"] = out["p_value"] < 0.01
    return out
