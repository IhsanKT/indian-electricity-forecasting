"""Seasonal naive baselines, in both the daily and weekly flavour.

    daily   f(t+h) = y(t+h-24)     h = 1..24
    weekly  f(t+h) = y(t+h-168)    h = 1..24

Both are implemented because the weekly variant is usually the stronger one on electricity
demand -- weekends do not look like weekdays -- and every improvement figure in this project
is quoted against whichever of the two actually wins. Beating only the weak baseline is the
easiest flaw for a reviewer to catch.

All model wrappers in this project share one interface:

    forecast(history, horizon) -> np.ndarray of length `horizon`

where `history` is everything known up to and including the forecast origin, and the
returned values correspond to origin+1h .. origin+horizon h.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


class SeasonalNaive:
    """Repeat the observation from one seasonal period ago."""

    def __init__(self, seasonal_period: int, name: str | None = None) -> None:
        self.m = int(seasonal_period)
        self.name = name or f"seasonal_naive_m{self.m}"

    @property
    def min_context(self) -> int:
        """Shortest history this model can forecast from."""
        return self.m

    def forecast(self, history: pd.Series | np.ndarray,
                 horizon: int = config.HORIZON) -> np.ndarray:
        """Forecast `horizon` steps beyond the end of `history`.

        For h = 1..horizon the prediction is y(T + h - m). With horizon <= m every
        referenced index falls inside `history`, so nothing is ever recursive.
        """
        y = np.asarray(history, dtype=float).ravel()
        if y.size < self.m:
            raise ValueError(f"{self.name}: need at least {self.m} observations, got {y.size}")
        if horizon > self.m:
            # Would need its own forecasts as inputs; not used at horizon 24.
            raise ValueError(f"{self.name}: horizon {horizon} exceeds seasonal period {self.m}")
        # y[-m:] holds y(T-m+1) .. y(T); element h-1 of that window is y(T+h-m).
        return y[-self.m:][:horizon].copy()

    def fit(self, train: pd.Series) -> "SeasonalNaive":  # noqa: ARG002 - nothing to learn
        """No-op; a naive baseline has no parameters. Present for interface symmetry."""
        return self


def daily() -> SeasonalNaive:
    """Seasonal naive with m=24."""
    return SeasonalNaive(config.SEASONAL_PERIOD_DAILY, name="seasonal_naive_daily")


def weekly() -> SeasonalNaive:
    """Seasonal naive with m=168."""
    return SeasonalNaive(config.SEASONAL_PERIOD_WEEKLY, name="seasonal_naive_weekly")
