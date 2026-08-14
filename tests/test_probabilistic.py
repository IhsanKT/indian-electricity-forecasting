"""Distributional metrics and the significance test, against values worked out by hand.

Proper scoring rules are easy to get subtly wrong -- a sign error in the Winkler penalty or
a missing factor of two in CRPS still produces plausible-looking numbers that rank models
in roughly the right order. These pin the definitions.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import metrics as M
from src.evaluation import statistical_tests as ST


# --------------------------------------------------------------------------------------
# Winkler score
# --------------------------------------------------------------------------------------
def test_winkler_inside_interval_is_just_the_width():
    # actual inside [90, 110] -> score is the width, 20
    assert M.winkler_score([100.0], [90.0], [110.0], alpha=0.2) == pytest.approx(20.0)


def test_winkler_below_interval_adds_scaled_shortfall():
    # actual 80, interval [90,110], alpha 0.2 -> 20 + (2/0.2)*(90-80) = 20 + 100 = 120
    assert M.winkler_score([80.0], [90.0], [110.0], alpha=0.2) == pytest.approx(120.0)


def test_winkler_above_interval_adds_scaled_excess():
    # actual 120 -> 20 + (2/0.2)*(120-110) = 120
    assert M.winkler_score([120.0], [90.0], [110.0], alpha=0.2) == pytest.approx(120.0)


def test_winkler_punishes_a_uselessly_wide_interval():
    """PICP alone would call the wide interval perfect; Winkler must not."""
    actual = [100.0, 105.0, 95.0]
    tight_lo, tight_hi = [90.0] * 3, [110.0] * 3
    wide_lo, wide_hi = [0.0] * 3, [1000.0] * 3
    assert M.picp(actual, wide_lo, wide_hi) == pytest.approx(1.0)
    assert M.picp(actual, tight_lo, tight_hi) == pytest.approx(1.0)
    assert M.winkler_score(actual, tight_lo, tight_hi) < M.winkler_score(actual, wide_lo, wide_hi)


def test_winkler_penalty_scales_with_confidence_level():
    """A miss on a 90% interval costs more than the same miss on a 50% one."""
    strict = M.winkler_score([80.0], [90.0], [110.0], alpha=0.1)
    loose = M.winkler_score([80.0], [90.0], [110.0], alpha=0.5)
    assert strict > loose


# --------------------------------------------------------------------------------------
# CRPS
# --------------------------------------------------------------------------------------
def test_crps_matches_twice_mean_pinball():
    rng = np.random.default_rng(0)
    actual = rng.normal(100, 10, 50)
    taus = np.array([0.1, 0.5, 0.9])
    q = np.column_stack([actual - 12, actual + 1, actual + 12])
    expected = 2.0 * np.mean([M.pinball_loss(actual, q[:, j], t) for j, t in enumerate(taus)])
    assert M.crps_from_quantiles(actual, q, taus) == pytest.approx(expected)


def test_crps_is_zero_for_a_perfect_point_mass():
    actual = np.array([100.0, 200.0])
    taus = np.array([0.1, 0.5, 0.9])
    q = np.column_stack([actual, actual, actual])  # all mass exactly on the truth
    assert M.crps_from_quantiles(actual, q, taus) == pytest.approx(0.0)


def test_crps_prefers_the_sharper_of_two_calibrated_forecasts():
    rng = np.random.default_rng(1)
    actual = rng.normal(0, 1, 400)
    taus = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    from scipy import stats as sps
    sharp = np.tile(sps.norm.ppf(taus, loc=0, scale=1), (400, 1))
    vague = np.tile(sps.norm.ppf(taus, loc=0, scale=5), (400, 1))
    assert M.crps_from_quantiles(actual, sharp, taus) < M.crps_from_quantiles(actual, vague, taus)


def test_crps_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        M.crps_from_quantiles([1.0, 2.0], np.zeros((2, 3)), [0.1, 0.9])


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------
def test_calibration_recovers_nominal_levels_for_a_correct_model():
    rng = np.random.default_rng(2)
    from scipy import stats as sps
    actual = rng.normal(0, 1, 20_000)
    taus = np.array([0.1, 0.5, 0.9])
    q = np.tile(sps.norm.ppf(taus), (20_000, 1))
    emp = M.quantile_calibration(actual, q, taus)
    for t in taus:
        assert emp[t] == pytest.approx(t, abs=0.02)


def test_calibration_detects_an_overconfident_model():
    """Too-narrow quantiles put too few actuals below q90 and too many below q10."""
    rng = np.random.default_rng(3)
    actual = rng.normal(0, 5, 5_000)
    taus = np.array([0.1, 0.9])
    q = np.tile(np.array([-0.2, 0.2]), (5_000, 1))  # far too tight
    emp = M.quantile_calibration(actual, q, taus)
    assert emp[0.9] < 0.9
    assert emp[0.1] > 0.1


# --------------------------------------------------------------------------------------
# Diebold-Mariano
# --------------------------------------------------------------------------------------
def test_dm_favours_the_genuinely_better_model():
    rng = np.random.default_rng(4)
    actual = rng.normal(100, 10, 500)
    good = actual + rng.normal(0, 1, 500)
    bad = actual + rng.normal(0, 8, 500)
    res = ST.diebold_mariano(actual, good, bad)
    assert res["favours"] == "A"
    assert res["dm_stat"] < 0
    assert res["p_value"] < 0.01


def test_dm_finds_no_difference_between_statistically_identical_models():
    rng = np.random.default_rng(5)
    actual = rng.normal(100, 10, 800)
    a = actual + rng.normal(0, 3, 800)
    b = actual + rng.normal(0, 3, 800)
    assert ST.diebold_mariano(actual, a, b)["p_value"] > 0.05


def test_dm_is_antisymmetric_in_its_arguments():
    rng = np.random.default_rng(6)
    actual = rng.normal(0, 1, 300)
    a = actual + rng.normal(0, 1, 300)
    b = actual + rng.normal(0, 2, 300)
    ab = ST.diebold_mariano(actual, a, b)
    ba = ST.diebold_mariano(actual, b, a)
    assert ab["dm_stat"] == pytest.approx(-ba["dm_stat"], rel=1e-9)
    assert ab["p_value"] == pytest.approx(ba["p_value"], rel=1e-9)


def test_newey_west_exceeds_naive_variance_under_positive_autocorrelation():
    """The whole point: overlapping windows inflate the true standard error."""
    rng = np.random.default_rng(7)
    n = 2_000
    e = rng.normal(0, 1, n)
    d = np.array([e[i] + 0.9 * e[i - 1] if i else e[i] for i in range(n)])
    naive = float(np.var(d))
    assert ST.newey_west_variance(d, max_lag=10) > naive


def test_newey_west_matches_naive_variance_for_white_noise():
    rng = np.random.default_rng(8)
    d = rng.normal(0, 1, 5_000)
    assert ST.newey_west_variance(d, max_lag=0) == pytest.approx(float(np.var(d)), rel=1e-9)


def test_dm_rejects_too_few_observations():
    with pytest.raises(ValueError):
        ST.diebold_mariano([1.0] * 5, [1.0] * 5, [1.0] * 5)


# --------------------------------------------------------------------------------------
# Quantile crossing / rearrangement
# --------------------------------------------------------------------------------------
def test_rearrangement_makes_quantiles_monotone():
    crossed = np.array([[10.0, 5.0, 7.0], [1.0, 2.0, 3.0]])
    fixed = M.rearrange_quantiles(crossed)
    assert np.all(np.diff(fixed, axis=1) >= 0)
    np.testing.assert_array_equal(fixed[0], [5.0, 7.0, 10.0])


def test_rearrangement_leaves_already_monotone_rows_untouched():
    good = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    np.testing.assert_array_equal(M.rearrange_quantiles(good), good)


def test_rearrangement_weakly_improves_pinball_loss():
    """The theoretical guarantee: sorting cannot make quantile estimates worse."""
    rng = np.random.default_rng(11)
    from scipy import stats as sps
    taus = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    actual = rng.normal(0, 1, 3_000)
    truth = sps.norm.ppf(taus)
    noisy = truth + rng.normal(0, 1.2, (3_000, taus.size))  # independent fits -> crossing
    fixed = M.rearrange_quantiles(noisy)

    before = np.mean([M.pinball_loss(actual, noisy[:, j], t) for j, t in enumerate(taus)])
    after = np.mean([M.pinball_loss(actual, fixed[:, j], t) for j, t in enumerate(taus)])
    assert after <= before


def test_crossing_counter_detects_and_measures_inversions():
    crossed = np.array([[1.0, 2.0, 3.0], [5.0, 1.0, 6.0]])
    diag = M.count_quantile_crossings(crossed)
    assert diag["rows_with_crossing"] == 1
    assert diag["fraction_of_rows"] == pytest.approx(0.5)
    assert diag["max_inversion"] == pytest.approx(4.0)


def test_crossing_counter_reports_clean_for_monotone_input():
    diag = M.count_quantile_crossings(np.array([[1.0, 2.0, 3.0]]))
    assert diag["rows_with_crossing"] == 0
    assert diag["max_inversion"] == pytest.approx(0.0)


def test_rearrange_rejects_non_2d_input():
    with pytest.raises(ValueError):
        M.rearrange_quantiles(np.array([1.0, 2.0, 3.0]))
