"""Chronological train/validation/test splits.

**Never a random split.** Shuffling hours would put 3pm Tuesday in train and 4pm Tuesday in
test, so a model could reach a near-perfect score by interpolating between its neighbours
while learning nothing about forecasting. Every split here is a contiguous slice of time,
and test is strictly the most recent data.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config


@dataclass(frozen=True)
class Splits:
    """Index slices for each split, plus the boundary timestamps."""

    train: pd.DatetimeIndex
    val: pd.DatetimeIndex
    test: pd.DatetimeIndex

    @property
    def train_end(self) -> pd.Timestamp:
        return self.train[-1]

    @property
    def val_end(self) -> pd.Timestamp:
        return self.val[-1]

    def describe(self) -> pd.DataFrame:
        rows = []
        for name in ("train", "val", "test"):
            idx = getattr(self, name)
            rows.append({
                "split": name,
                "start": idx[0],
                "end": idx[-1],
                "hours": len(idx),
                "days": round(len(idx) / 24, 1),
                "share_%": round(100 * len(idx) / (len(self.train) + len(self.val) + len(self.test)), 1),
            })
        return pd.DataFrame(rows)


def chronological_splits(y: pd.Series,
                         train_frac: float = config.TRAIN_FRAC,
                         val_frac: float = config.VAL_FRAC) -> Splits:
    """Split the series 70/15/15 in time order."""
    n = len(y)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    idx = y.index
    splits = Splits(train=idx[:n_train],
                    val=idx[n_train:n_train + n_val],
                    test=idx[n_train + n_val:])
    # Guard the invariant the whole project depends on.
    assert splits.train[-1] < splits.val[0] < splits.val[-1] < splits.test[0], \
        "splits must be strictly ordered in time"
    return splits
