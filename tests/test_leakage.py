"""Leakage assertions for the LightGBM feature pipeline.

The decisive test: build the feature frame from the whole series, then from the series
truncated at origin T, and require the row at T to be *identical*. If any lag or rolling
window reached forward -- even by one hour -- the two would differ, because the truncated
series simply has no future to reach into.

Eyeballing a feature list does not catch this. `rolling(w).mean()` with the wrong `center`
or a stray `shift(-1)` looks entirely reasonable in review and produces plausible metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import features as F


def _series(n: int = 1200, seed: int = 0) -> pd.Series:
    """Synthetic hourly demand with daily and weekly shape plus noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="Asia/Kolkata")
    t = np.arange(n)
    y = (150_000
         + 20_000 * np.sin(2 * np.pi * t / 24)
         + 8_000 * np.sin(2 * np.pi * t / 168)
         + rng.normal(0, 1_000, n))
    return pd.Series(y, index=idx, name="demand")


@pytest.mark.parametrize("offset", [400, 700, 1000])
def test_origin_features_do_not_depend_on_the_future(offset):
    """Truncating the series after T must not change the feature row at T."""
    y = _series()
    origin = y.index[offset]

    full_row = F.origin_features(y).loc[origin]
    truncated_row = F.origin_features(y.loc[:origin]).loc[origin]

    pd.testing.assert_series_equal(full_row, truncated_row, check_names=False)


def test_every_origin_feature_is_finite_and_identical_under_truncation():
    """Same property, checked across many origins and every feature column at once."""
    y = _series()
    full = F.origin_features(y)
    for offset in range(400, 1100, 97):
        origin = y.index[offset]
        trunc = F.origin_features(y.loc[:origin]).loc[origin]
        assert np.isfinite(trunc.to_numpy()).all(), f"non-finite features at {origin}"
        np.testing.assert_array_equal(full.loc[origin].to_numpy(), trunc.to_numpy())


@pytest.mark.parametrize("horizon", [1, 12, 24])
def test_training_frame_features_precede_the_target(horizon):
    """The full feature row for origin T must survive truncation of everything after T."""
    y = _series()
    X, target = F.training_frame(y, horizon)
    origin = X.index[len(X) // 2]

    trunc = F.features_at_origin(y.loc[:origin], horizon)
    # Compare only the origin-derived columns; calendar columns describe the target time,
    # which is legitimately known in advance.
    for col in F.ORIGIN_FEATURES:
        assert X.loc[origin, col] == pytest.approx(trunc.iloc[0][col]), f"{col} differs"


@pytest.mark.parametrize("horizon", [1, 6, 24])
def test_training_frame_target_is_the_value_h_hours_after_the_origin(horizon):
    """Guards against the target itself being misaligned, which no feature test would catch."""
    y = _series()
    X, target = F.training_frame(y, horizon)
    for origin in X.index[::250]:
        expected_time = origin + pd.Timedelta(hours=horizon)
        assert target.loc[origin] == pytest.approx(y.loc[expected_time])


def test_calendar_features_describe_the_target_not_the_origin():
    """At horizon 24 the target's hour equals the origin's; at horizon 1 it must not."""
    y = _series()
    X1, _ = F.training_frame(y, 1)
    origin = X1.index[500]
    assert X1.loc[origin, "hour"] == (origin + pd.Timedelta(hours=1)).hour

    X24, _ = F.training_frame(y, 24)
    assert X24.loc[origin, "hour"] == origin.hour


def test_features_at_origin_rejects_insufficient_history():
    y = _series(n=100)
    with pytest.raises(ValueError):
        F.features_at_origin(y, 1)


def test_rolling_std_is_trailing_not_centred():
    """A centred window is the classic accidental-leak; catch it directly.

    On a series that is flat then jumps, a trailing 24h std at the last pre-jump point must
    still be zero. A centred window would already 'see' the jump.
    """
    n = 400
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="Asia/Kolkata")
    values = np.concatenate([np.full(200, 100.0), np.full(n - 200, 500.0)])
    y = pd.Series(values, index=idx)

    feats = F.origin_features(y)
    last_flat = idx[199]
    assert feats.loc[last_flat, "roll_std_24"] == pytest.approx(0.0)
    assert feats.loc[last_flat, "roll_mean_24"] == pytest.approx(100.0)
