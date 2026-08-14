"""Feature construction for the LightGBM tier.

Every feature is indexed by **forecast origin T** and is computed from y(<= T) only.
Calendar features describe the *target* timestamp T+h, which is legitimate: a calendar is
known arbitrarily far in advance, unlike demand.

That split is what makes the leakage test in tests/test_leakage.py meaningful — building
the frame from the full series and from the series truncated at T must give an identical
row at T. If a rolling window or lag ever reached forward, those two would differ.
"""
from __future__ import annotations

import functools

import holidays
import numpy as np
import pandas as pd

#: Lags in hours back from the forecast origin. The spec requires 24/48/168/336
#: (day, two days, week, fortnight); the short lags carry the most recent level.
LAGS: tuple[int, ...] = (0, 1, 2, 3, 24, 48, 168, 336)

#: Trailing windows for rolling statistics, in hours: one day and one week.
ROLL_WINDOWS: tuple[int, ...] = (24, 168)

MAX_LAG = max(max(LAGS), max(ROLL_WINDOWS))

ORIGIN_FEATURES: tuple[str, ...] = tuple(
    [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + [f"roll_std_{w}" for w in ROLL_WINDOWS]
    + ["diff_24", "diff_168"]
)

CALENDAR_FEATURES: tuple[str, ...] = (
    "hour", "dayofweek", "month", "dayofyear", "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
)

FEATURE_NAMES: tuple[str, ...] = ORIGIN_FEATURES + CALENDAR_FEATURES


@functools.lru_cache(maxsize=1)
def _india_holidays(first_year: int, last_year: int) -> set:
    """Indian public holidays as a set of dates, cached across calls."""
    cal = holidays.India(years=range(first_year, last_year + 1))
    return set(cal.keys())


def origin_features(y: pd.Series) -> pd.DataFrame:
    """Features known at each origin T, built only from y(<= T).

    Uses no shift(-k) anywhere: `rolling(w)` is trailing and right-closed, and `shift(k)`
    with k >= 0 only ever reaches backwards.
    """
    out = pd.DataFrame(index=y.index)
    for lag in LAGS:
        out[f"lag_{lag}"] = y.shift(lag)
    for w in ROLL_WINDOWS:
        roll = y.rolling(window=w, min_periods=w)
        out[f"roll_mean_{w}"] = roll.mean()
        out[f"roll_std_{w}"] = roll.std()
    # Change over the last day / week — cheap trend signal.
    out["diff_24"] = y - y.shift(24)
    out["diff_168"] = y - y.shift(168)
    return out


def calendar_features(times: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar attributes of the *target* timestamps.

    Deterministic and known in advance, so using target-time calendar values is not leakage.
    """
    idx = pd.DatetimeIndex(times)
    hol = _india_holidays(int(idx.year.min()), int(idx.year.max()))
    dates = idx.normalize()
    # tz-aware normalize keeps the tz; compare on plain dates.
    date_objs = pd.Index([d.date() for d in dates])

    out = pd.DataFrame(index=idx)
    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["dayofyear"] = idx.dayofyear
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["is_holiday"] = date_objs.isin(hol).astype(int)
    # Cyclic encodings so the model sees hour 23 and hour 0 as adjacent.
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    return out


def training_frame(y: pd.Series, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    """(X, target) for one horizon, indexed by origin T, with target y(T + horizon).

    Rows where any feature or the target is unavailable are dropped.
    """
    base = origin_features(y)
    target_times = y.index + pd.Timedelta(hours=horizon)
    cal = calendar_features(target_times)
    cal.index = y.index  # re-key onto the origin so the frames align

    X = pd.concat([base, cal], axis=1)[list(FEATURE_NAMES)]
    target = pd.Series(y.shift(-horizon).to_numpy(), index=y.index, name="target")

    ok = X.notna().all(axis=1) & target.notna()
    return X[ok], target[ok]


def features_at_origin(history: pd.Series, horizon: int) -> pd.DataFrame:
    """One feature row for the final origin in `history`, for the given horizon.

    Deliberately reuses `origin_features` so training and inference cannot drift apart.
    """
    if len(history) < MAX_LAG + 1:
        raise ValueError(f"need at least {MAX_LAG + 1} history points, got {len(history)}")
    base = origin_features(history).iloc[[-1]]
    origin = history.index[-1]
    cal = calendar_features(pd.DatetimeIndex([origin + pd.Timedelta(hours=horizon)]))
    cal.index = base.index
    return pd.concat([base, cal], axis=1)[list(FEATURE_NAMES)]
