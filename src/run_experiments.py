"""Run the model comparison end to end and cache every forecast to results/.

Stages are separable so a slow one can be re-run without redoing the rest:

    python -m src.run_experiments --stage all
    python -m src.run_experiments --stage timesfm

Two rules hold across every tier, and they are what make the comparison fair:

1. **Identical origins.** Every model is scored on exactly the same forecast origins, set
   by the longest context any model needs (720h for TimesFM). Otherwise a model with a
   shorter warm-up would be judged on a different, easier stretch of the test set.
2. **Selection never touches test.** LightGBM hyperparameters and the TimesFM context
   length are both chosen on validation. Test numbers for the alternatives are reported for
   transparency, but the headline model is the one validation picked.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from src import config, splits as S
from src.data.preprocess import load_processed
from src.evaluation import metrics as M, rolling_evaluation as RE
from src.models import lightgbm_model as LGB, seasonal_naive as SN

#: Longest context needed by any model; fixes the common origin set.
MIN_CONTEXT = max(config.CONTEXT_LENGTHS + (337,))

FORECAST_PATH = config.RESULTS_DIR / "forecasts.parquet"
SUMMARY_PATH = config.RESULTS_DIR / "summary.csv"
HORIZON_PATH = config.RESULTS_DIR / "error_by_horizon.csv"
META_PATH = config.RESULTS_DIR / "run_metadata.json"
TUNING_PATH = config.RESULTS_DIR / "lightgbm_tuning.json"


def _load():
    y = load_processed()
    sp = S.chronological_splits(y)
    return y, sp


def _origins(y: pd.Series, sp: S.Splits, which: str) -> pd.DatetimeIndex:
    idx = {"val": sp.val, "test": sp.test}[which]
    return RE.make_origins(y, start=idx[0], end=idx[-1],
                           stride_hours=config.ORIGIN_STRIDE_HOURS,
                           min_context=MIN_CONTEXT, horizon=config.HORIZON)


def _append(frames: list[pd.DataFrame], path=FORECAST_PATH) -> None:
    """Merge new forecasts into the cache, replacing any earlier run of the same model."""
    new = pd.concat(frames, ignore_index=True)
    if path.exists():
        old = pd.read_parquet(path)
        old = old[~old["model"].isin(new["model"].unique())]
        new = pd.concat([old, new], ignore_index=True)
    new.to_parquet(path, index=False)
    print(f"  cached -> {path}  ({len(new):,} rows, models: "
          f"{sorted(new['model'].unique())})")


# --------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------
def stage_baselines() -> None:
    """Phase 3. Both seasonal naives, and the frozen MASE scale."""
    print("\n=== PHASE 3: baselines ===")
    y, sp = _load()
    train_vals = y.loc[sp.train]

    scale_daily = M.seasonal_naive_scale(train_vals, config.SEASONAL_PERIOD_DAILY)
    scale_weekly = M.seasonal_naive_scale(train_vals, config.SEASONAL_PERIOD_WEEKLY)
    print(f"  in-sample seasonal-naive MAE on train: m=24 {scale_daily:,.1f} MW | "
          f"m=168 {scale_weekly:,.1f} MW")

    # Which baseline is stronger is decided on VALIDATION, not on test.
    val_origins = _origins(y, sp, "val")
    print(f"  validation origins: {len(val_origins):,}")
    val_frames = [RE.rolling_forecast(m, y, val_origins, model_name=m.name)
                  for m in (SN.daily(), SN.weekly())]
    val_mae = {f["model"].iloc[0]: M.mae(*_clean(f)) for f in val_frames}
    print(f"  validation MAE: {({k: round(v, 1) for k, v in val_mae.items()})}")
    stronger = min(val_mae, key=val_mae.get)
    print(f"  -> stronger baseline: {stronger}")
    weekly_diag = _weekly_seasonality_check(train_vals)

    # MASE denominator = in-sample scale of whichever baseline is stronger. Frozen here and
    # applied identically to every model.
    scale = scale_weekly if stronger == "seasonal_naive_weekly" else scale_daily

    test_origins = _origins(y, sp, "test")
    print(f"  test origins: {len(test_origins):,}")
    frames = [RE.rolling_forecast(m, y, test_origins, model_name=m.name)
              for m in (SN.daily(), SN.weekly())]
    _append(frames)

    meta = {
        "mase_scale": scale,
        "mase_scale_basis": stronger,
        "scale_daily": scale_daily,
        "scale_weekly": scale_weekly,
        "stronger_baseline": stronger,
        "validation_mae": val_mae,
        "weekly_seasonality_check": weekly_diag,
        "n_test_origins": int(len(test_origins)),
        "n_val_origins": int(len(val_origins)),
        "test_start": str(sp.test[0]),
        "test_end": str(sp.test[-1]),
        "min_context": MIN_CONTEXT,
        "origin_stride_hours": config.ORIGIN_STRIDE_HOURS,
        "random_seed": config.RANDOM_SEED,
    }
    META_PATH.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"  frozen MASE scale = {scale:,.1f} MW (in-sample {stronger} on train)")


def _clean(f: pd.DataFrame):
    d = f.dropna(subset=["actual", "prediction"])
    return d["actual"], d["prediction"]


def _weekly_seasonality_check(train_vals: pd.Series) -> dict:
    """Confirm weekly seasonality exists even when the weekly naive loses to the daily one.

    On Western grids the weekly naive normally wins, so a daily win is the classic symptom
    of misaligned timestamps. On Indian all-India demand it is expected instead: industry
    and agriculture run seven days a week, so only Sunday differs materially and seven days
    of drift costs more than matching the weekday buys.

    Distinguishing the two cases needs a positive test rather than a comparison of the two
    baselines. If weekly structure is genuinely present, lag 168 is a *local minimum* --
    better than its neighbours at 144h and 192h -- even when lag 24 beats it outright.
    Losing to lag 24 is fine; losing to lag 144 as well would mean the weekly cycle is not
    in the data at all, which would point at preprocessing.
    """
    v = train_vals.to_numpy(dtype=float)
    lag_mae = {m: float(np.nanmean(np.abs(v[m:] - v[:-m]))) for m in (24, 120, 144, 168, 192)}
    is_local_min = lag_mae[168] < lag_mae[144] and lag_mae[168] < lag_mae[192]

    dow = train_vals.groupby(train_vals.index.dayofweek).mean()
    dow_spread_pct = float((dow.max() - dow.min()) / dow.mean() * 100)
    hod = train_vals.groupby(train_vals.index.hour).mean()
    hod_spread_pct = float((hod.max() - hod.min()) / hod.mean() * 100)

    print(f"  weekly-seasonality check: lag-168 MAE {lag_mae[168]:,.0f} vs "
          f"lag-144 {lag_mae[144]:,.0f} / lag-192 {lag_mae[192]:,.0f} -> "
          f"{'local minimum, weekly cycle present' if is_local_min else 'NO local minimum'}")
    print(f"    day-of-week spread {dow_spread_pct:.2f}% vs hour-of-day spread "
          f"{hod_spread_pct:.2f}% of mean")
    if not is_local_min:
        print("    WARNING: no weekly structure detected at all. Suspect timestamp "
              "misalignment in preprocessing before trusting any result below.")

    return {"lag_mae": lag_mae, "lag168_is_local_minimum": is_local_min,
            "dow_spread_pct": dow_spread_pct, "hod_spread_pct": hod_spread_pct}


def stage_lightgbm() -> None:
    """Phase 4 and the quantile half of Phase 7."""
    print("\n=== PHASE 4: LightGBM ===")
    y, sp = _load()

    print("  light grid search on validation ...")
    best, records = LGB.tune(y, sp.train, sp.val)
    TUNING_PATH.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")

    # Refit on train+val for the test run. The other tiers see all history up to each
    # origin, so withholding validation from LightGBM alone would handicap it.
    fit_index = sp.train.union(sp.val)
    test_origins = _origins(y, sp, "test")

    print(f"  fitting 24 horizon models on {len(fit_index):,} origins ...")
    t0 = time.time()
    model = LGB.LightGBMForecaster(params=best).fit(y, fit_index)
    print(f"    done in {time.time()-t0:.1f}s")

    point = model.forecast_batch(y, test_origins)
    frame = _frame_from_array("lightgbm", y, test_origins, point)

    n_q = len(config.QUANTILES)
    print(f"  fitting quantile models ({n_q} quantiles x {config.HORIZON} horizons) ...")
    t0 = time.time()
    for q in config.QUANTILES:
        qm = LGB.LightGBMForecaster(params=best, quantile=q).fit(y, fit_index)
        frame[f"q{q:g}"] = qm.forecast_batch(y, test_origins).reshape(-1)
        print(f"    q{q:g} done ({time.time()-t0:.1f}s elapsed)", flush=True)

    # Independently fitted quantile models are not monotone by construction, and here they
    # cross on most rows. Rearrange before caching so every downstream distributional score
    # sees a valid distribution.
    qcols = [f"q{q:g}" for q in config.QUANTILES]
    before = M.count_quantile_crossings(frame[qcols].to_numpy(float))
    frame[qcols] = M.rearrange_quantiles(frame[qcols].to_numpy(float))
    print(f"  quantile crossing before rearrangement: "
          f"{before['fraction_of_rows']*100:.1f}% of rows, "
          f"max inversion {before['max_inversion']:,.0f} MW -> rearranged")

    _append([frame])


def _frame_from_array(name: str, y: pd.Series, origins: pd.DatetimeIndex,
                      point: np.ndarray) -> pd.DataFrame:
    horizon = point.shape[1]
    h_col = np.tile(np.arange(1, horizon + 1), len(origins))
    origin_col = np.repeat(np.asarray(origins), horizon)
    target_times = pd.DatetimeIndex(origin_col) + pd.to_timedelta(h_col, unit="h")
    return pd.DataFrame({
        "model": name,
        "origin": origin_col,
        "h": h_col,
        "target_time": target_times,
        "actual": y.reindex(target_times).to_numpy(),
        "prediction": point.reshape(-1),
    })


def stage_timesfm(contexts: tuple[int, ...] = config.CONTEXT_LENGTHS,
                  all_contexts_on_test: bool = False) -> None:
    """Phase 5. Context length is chosen on validation; the winner is run on test.

    At ~1.4 origins/s on this CPU, scoring all three contexts over the full test set costs
    ~30 min for information the validation sweep already provides. Set
    `all_contexts_on_test=True` to spend it anyway.
    """
    from src.models.timesfm_model import TimesFMForecaster

    print("\n=== PHASE 5: TimesFM 2.5 (zero-shot) ===")
    y, sp = _load()
    val_origins = _origins(y, sp, "val")
    test_origins = _origins(y, sp, "test")

    # Choosing on validation, with a thinned origin set to keep CPU cost sane.
    thinned = val_origins[::4]
    print(f"  context sweep on {len(thinned):,} validation origins")
    val_scores: dict[int, float] = {}
    for ctx in contexts:
        m = TimesFMForecaster(context_length=ctx)
        f = RE.rolling_forecast(m, y, thinned, model_name=f"timesfm_ctx{ctx}")
        val_scores[ctx] = M.mae(*_clean(f))
        print(f"    ctx={ctx:<4} validation MAE {val_scores[ctx]:,.1f} MW")
    best_ctx = min(val_scores, key=val_scores.get)
    print(f"  -> selected context {best_ctx} on validation")

    run_on_test = contexts if all_contexts_on_test else (best_ctx,)
    print(f"  running {len(run_on_test)} context(s) on {len(test_origins):,} test origins")
    frames = []
    for ctx in run_on_test:
        m = TimesFMForecaster(context_length=ctx)
        name = "timesfm" if ctx == best_ctx else f"timesfm_ctx{ctx}"
        frames.append(RE.rolling_forecast(m, y, test_origins, model_name=name))
    _append(frames)

    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    meta["timesfm_context_val_mae"] = {str(k): v for k, v in val_scores.items()}
    meta["timesfm_selected_context"] = best_ctx
    META_PATH.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def stage_summary() -> None:
    """Phases 6 and 7: headline table, horizon curve, distributional scores, significance."""
    from src.evaluation import report as RPT

    print("\n=== PHASE 6/7: evaluation report ===")
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    scale = float(meta["mase_scale"])
    baseline = meta["stronger_baseline"]
    fc = pd.read_parquet(FORECAST_PATH)

    summary = RE.summarise(fc, scale=scale, baseline_model=baseline)
    summary.to_csv(SUMMARY_PATH)
    RE.error_by_horizon(fc).to_csv(HORIZON_PATH, index=False)

    tables = RPT.build_all(fc, scale=scale, baseline=baseline)

    pd.set_option("display.width", 240)
    print(f"\nMASE scale {scale:,.1f} MW (in-sample {baseline} on train); "
          f"skill quoted vs {baseline}\n")
    print(tables["summary_detailed"].round(3).to_string())

    if not tables["probabilistic"].empty:
        print("\n--- distributional scores ---")
        print(tables["probabilistic"].round(3).to_string())

    if not tables["significance_dm"].empty:
        print("\n--- Diebold-Mariano (origin-level absolute loss, Newey-West HAC) ---")
        cols = ["model_A", "model_B", "winner", "dm_stat", "p_value", "significant_1pct"]
        print(tables["significance_dm"][cols].round(4).to_string(index=False))

    print("\n--- peak-demand accuracy ---")
    print(tables["peak_metrics"].round(2).to_string())


STAGES = {
    "baselines": stage_baselines,
    "lightgbm": stage_lightgbm,
    "timesfm": stage_timesfm,
    "summary": stage_summary,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all", choices=[*STAGES, "all"])
    args = ap.parse_args()
    np.random.seed(config.RANDOM_SEED)

    names = list(STAGES) if args.stage == "all" else [args.stage]
    for n in names:
        STAGES[n]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
