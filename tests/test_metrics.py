"""Metric functions checked against values computed by hand.

Deliberately uses tiny arrays whose answers can be verified with arithmetic on paper -- a
metric that silently disagrees with its own definition would corrupt every number in the
results table.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation import metrics as M


ACTUAL = [100.0, 200.0, 300.0, 400.0]
PRED = [110.0, 190.0, 330.0, 360.0]
# errors: -10, +10, -30, +40  ->  |e| = 10, 10, 30, 40


def test_mae_matches_hand_calculation():
    # (10 + 10 + 30 + 40) / 4 = 22.5
    assert M.mae(ACTUAL, PRED) == pytest.approx(22.5)


def test_rmse_matches_hand_calculation():
    # (100 + 100 + 900 + 1600) / 4 = 675 ; sqrt(675) = 25.98076...
    assert M.rmse(ACTUAL, PRED) == pytest.approx(math.sqrt(675.0))


def test_mape_matches_hand_calculation():
    # (10/100 + 10/200 + 30/300 + 40/400) * 100 / 4 = (0.1+0.05+0.1+0.1)*100/4 = 8.75
    assert M.mape(ACTUAL, PRED) == pytest.approx(8.75)


def test_smape_matches_hand_calculation():
    # 200*|e|/(|a|+|p|) per term:
    #   200*10/210 = 9.523809...
    #   200*10/390 = 5.128205...
    #   200*30/630 = 9.523809...
    #   200*40/760 = 10.526315...
    expected = np.mean([200 * 10 / 210, 200 * 10 / 390, 200 * 30 / 630, 200 * 40 / 760])
    assert M.smape(ACTUAL, PRED) == pytest.approx(expected)


def test_perfect_forecast_scores_zero():
    assert M.mae(ACTUAL, ACTUAL) == 0.0
    assert M.rmse(ACTUAL, ACTUAL) == 0.0
    assert M.smape(ACTUAL, ACTUAL) == 0.0


def test_seasonal_naive_scale_matches_hand_calculation():
    # m=2: |30-10| + |40-20| + |50-30| = 20+20+20 -> mean 20
    y = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert M.seasonal_naive_scale(y, 2) == pytest.approx(20.0)


def test_mase_is_mae_divided_by_scale():
    scale = 4.5
    assert M.mase(ACTUAL, PRED, scale) == pytest.approx(22.5 / 4.5)


def test_mase_of_one_means_baseline_parity():
    """MASE = 1 exactly when the model's MAE equals the in-sample naive MAE."""
    scale = M.mae(ACTUAL, PRED)
    assert M.mase(ACTUAL, PRED, scale) == pytest.approx(1.0)


def test_evaluate_forecast_returns_nan_mase_without_scale():
    """Silently substituting a different scale would make MASE incomparable."""
    out = M.evaluate_forecast(ACTUAL, PRED)
    assert math.isnan(out["MASE"])
    assert out["MAE"] == pytest.approx(22.5)


def test_evaluate_forecast_uses_supplied_scale():
    out = M.evaluate_forecast(ACTUAL, PRED, scale=4.5)
    assert out["MASE"] == pytest.approx(5.0)


def test_seasonal_naive_scale_rejects_too_short_series():
    with pytest.raises(ValueError):
        M.seasonal_naive_scale([1.0, 2.0], 5)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        M.mae([1.0, 2.0], [1.0])


# --------------------------------------------------------------------------------------
# Probabilistic metrics
# --------------------------------------------------------------------------------------
def test_picp_counts_actuals_inside_the_interval():
    actual = [1.0, 5.0, 10.0, 20.0]
    lower = [0.0, 0.0, 0.0, 0.0]
    upper = [2.0, 4.0, 11.0, 21.0]  # inside, outside, inside, inside
    assert M.picp(actual, lower, upper) == pytest.approx(0.75)


def test_picp_is_inclusive_at_the_boundary():
    assert M.picp([1.0], [1.0], [2.0]) == pytest.approx(1.0)
    assert M.picp([2.0], [1.0], [2.0]) == pytest.approx(1.0)


def test_pinball_loss_matches_hand_calculation():
    # tau=0.9, actual 10, pred 8 -> diff=+2 -> max(0.9*2, -0.1*2) = 1.8
    assert M.pinball_loss([10.0], [8.0], 0.9) == pytest.approx(1.8)
    # tau=0.9, actual 8, pred 10 -> diff=-2 -> max(-1.8, 0.2) = 0.2
    assert M.pinball_loss([8.0], [10.0], 0.9) == pytest.approx(0.2)


def test_pinball_at_median_is_half_the_absolute_error():
    assert M.pinball_loss([10.0], [8.0], 0.5) == pytest.approx(1.0)


def test_interval_width():
    assert M.interval_width([1.0, 2.0], [3.0, 6.0]) == pytest.approx(3.0)
