"""Forecast/actual timestamp alignment.

Off-by-one alignment is the most common silent bug in forecasting code: a project that
compares a forecast to the wrong hour still produces entirely plausible metrics, and
nothing in the output looks wrong. These tests pin the alignment with synthetic series
whose correct answer is known by construction rather than by inspection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.evaluation import rolling_evaluation as RE
from src.models import seasonal_naive as SN


def _ramp(n: int = 1000) -> pd.Series:
    """y(t) = t. Any misalignment by k hours shows up as a constant error of exactly k."""
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="Asia/Kolkata")
    return pd.Series(np.arange(n, dtype=float), index=idx, name="demand")


def test_seasonal_naive_daily_returns_the_value_24h_before_each_target():
    """f(T+h) must equal y(T+h-24) exactly -- not y(T+h-23) or y(T+h-25)."""
    y = _ramp()
    model = SN.daily()
    origin_pos = 500
    history = y.iloc[: origin_pos + 1]

    pred = model.forecast(history, horizon=24)
    expected = [y.iloc[origin_pos + h - 24] for h in range(1, 25)]
    np.testing.assert_array_equal(pred, np.array(expected, dtype=float))


def test_seasonal_naive_weekly_returns_the_value_168h_before_each_target():
    y = _ramp()
    model = SN.weekly()
    origin_pos = 500
    history = y.iloc[: origin_pos + 1]

    pred = model.forecast(history, horizon=24)
    expected = [y.iloc[origin_pos + h - 168] for h in range(1, 25)]
    np.testing.assert_array_equal(pred, np.array(expected, dtype=float))


def test_ramp_gives_constant_error_equal_to_the_seasonal_period():
    """On y(t)=t the daily naive must be wrong by exactly 24 at every horizon.

    Any other constant means the harness is pairing forecasts with the wrong actuals.
    """
    y = _ramp()
    origins = RE.make_origins(y, min_context=200, stride_hours=24, horizon=24)
    fc = RE.rolling_forecast(SN.daily(), y, origins, horizon=24, verbose=False)

    err = fc["actual"] - fc["prediction"]
    assert err.notna().all()
    np.testing.assert_allclose(err.to_numpy(), 24.0)


def test_weekly_naive_on_ramp_is_wrong_by_exactly_168():
    y = _ramp()
    origins = RE.make_origins(y, min_context=300, stride_hours=24, horizon=24)
    fc = RE.rolling_forecast(SN.weekly(), y, origins, horizon=24, verbose=False)
    np.testing.assert_allclose((fc["actual"] - fc["prediction"]).to_numpy(), 168.0)


def test_target_time_is_origin_plus_h():
    y = _ramp()
    origins = RE.make_origins(y, min_context=200, stride_hours=24, horizon=24)
    fc = RE.rolling_forecast(SN.daily(), y, origins, horizon=24, verbose=False)

    expected = pd.DatetimeIndex(
        pd.DatetimeIndex(fc["origin"]) + pd.to_timedelta(fc["h"].to_numpy(), unit="h")
    )
    pd.testing.assert_index_equal(pd.DatetimeIndex(fc["target_time"]), expected,
                                  check_names=False)


def test_actual_column_is_looked_up_by_timestamp():
    """actual at each row must equal y at that row's target_time."""
    y = _ramp()
    origins = RE.make_origins(y, min_context=200, stride_hours=24, horizon=24)
    fc = RE.rolling_forecast(SN.daily(), y, origins, horizon=24, verbose=False)

    expected = y.reindex(pd.DatetimeIndex(fc["target_time"])).to_numpy()
    np.testing.assert_array_equal(fc["actual"].to_numpy(), expected)


def test_forecast_never_reuses_the_origin_value_as_the_first_actual():
    """The first actual is y(T+1), not y(T) -- the classic off-by-one."""
    y = _ramp()
    origins = RE.make_origins(y, min_context=200, stride_hours=24, horizon=24)
    fc = RE.rolling_forecast(SN.daily(), y, origins, horizon=24, verbose=False)

    first = fc[fc["h"] == 1]
    origin_values = y.reindex(pd.DatetimeIndex(first["origin"])).to_numpy()
    assert not np.allclose(first["actual"].to_numpy(), origin_values)
    np.testing.assert_array_equal(first["actual"].to_numpy(), origin_values + 1.0)


def test_origins_leave_room_for_a_full_horizon():
    """No origin may sit so late that its 24h window runs past the end of the series."""
    y = _ramp(n=500)
    origins = RE.make_origins(y, min_context=200, stride_hours=6, horizon=24)
    assert len(origins) > 0
    assert origins.max() + pd.Timedelta(hours=24) <= y.index.max()


def test_origins_have_enough_history():
    y = _ramp(n=500)
    min_context = 200
    origins = RE.make_origins(y, min_context=min_context, stride_hours=6, horizon=24)
    assert origins.min() >= y.index[min_context - 1]


def test_error_by_horizon_is_flat_for_a_seasonal_naive_on_a_ramp():
    """Sanity check on the horizon aggregation itself."""
    y = _ramp()
    origins = RE.make_origins(y, min_context=200, stride_hours=24, horizon=24)
    fc = RE.rolling_forecast(SN.daily(), y, origins, horizon=24, verbose=False)
    curve = RE.error_by_horizon(fc)
    assert len(curve) == 24
    np.testing.assert_allclose(curve["MAE"].to_numpy(), 24.0)


def test_seasonal_naive_rejects_horizon_beyond_its_period():
    with pytest.raises(ValueError):
        SN.daily().forecast(_ramp(), horizon=25)
