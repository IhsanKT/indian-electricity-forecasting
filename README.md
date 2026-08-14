# Do zero-shot foundation models beat a tuned local model for Indian electricity demand?

**24-hour-ahead all-India electricity demand forecasting. Four model tiers, one
rolling-origin evaluation, 832 forecast origins over a 7-month held-out test set.**

**Answer: yes, and not narrowly.** Zero-shot TimesFM 2.5 cut MAE by **36%** against the
stronger seasonal naive, while a tuned LightGBM managed **8%**. The foundation model had
never seen Indian data specifically, was given no calendar, no holidays and no covariates,
and was not fine-tuned.

## Results

Test set: 2024-11-28 → 2025-06-24. 832 origins, every 6 hours, horizon 24h — 19,968
forecasts per model. Identical origins for every tier.

| Model | MAE (MW) | RMSE (MW) | sMAPE (%) | MASE | vs. baseline | PICP@80% |
|---|---|---|---|---|---|---|
| **TimesFM 2.5, ctx 896 (zero-shot)** | **3,642** | **5,838** | **1.88** | **0.819** | **+35.7%** | 0.820 |
| TimesFM 2.5, ctx 720 (zero-shot) | 3,719 | 5,894 | 1.92 | 0.837 | +34.3% | 0.809 |
| LightGBM (tuned, 24 direct models) | 5,231 | 7,311 | 2.71 | 1.177 | +7.6% | 0.552 |
| Seasonal naive, m=24 *(baseline)* | 5,662 | 8,616 | 2.95 | 1.274 | — | — |
| Seasonal naive, m=168 | 8,329 | 11,742 | 4.29 | 1.874 | −47.1% | — |

MASE is scaled by the in-sample MAE of the stronger seasonal naive on the training split
(**4,445.2 MW**, m=24), frozen once and applied identically to every model. Improvement is
quoted against that same stronger baseline — never against the weaker one.

**MASE below 1 means beating the in-sample naive.** Only TimesFM does. LightGBM's 1.177
means that despite winning on the test set, it is still worse than the naive was in-sample.

### Three things the headline number hides

**1. LightGBM overtakes TimesFM at long horizons.**

| Horizon | h=1 | h=6 | h=12 | h=18 | h=20 | h=21 | h=24 |
|---|---|---|---|---|---|---|---|
| TimesFM MAE | **1,272** | **3,163** | **3,789** | **4,222** | **4,283** | 4,417 | 4,327 |
| LightGBM MAE | 2,333 | 5,111 | 5,390 | 5,823 | 4,656 | **4,390** | **4,051** |

TimesFM is nearly twice as accurate at h=1 but degrades 3.4× across the day; LightGBM
degrades only 1.7× and wins outright from **h≥21**, where explicit hour-of-day and calendar
features let it anchor on "same hour tomorrow". If you only need day-ahead demand at a
fixed delivery hour, the cheap model is still competitive. Both seasonal naives are flat
across horizon (growth factors 1.02 and 0.98), which is the expected signature and a
useful check that the harness is sound.

**2. TimesFM's uncertainty is calibrated; LightGBM's is not.**
An 80% prediction interval is *expected* to contain the actual value about 80% of the
time — not always. TimesFM's native quantiles achieve **82.0% coverage against a nominal
80%**, marginally wide but very close to correct. LightGBM's quantile-regression intervals
reach only **55.2%** while being *narrower* (10,142 MW vs 11,666 MW average width): they
are overconfident, and would understate risk if used for reserve planning.

**3. LightGBM degrades out-of-sample; TimesFM does not.**
LightGBM scored 4,003 MW on validation but 5,231 MW on test (+31%). TimesFM was flat
(3,568 → 3,642 at ctx 896). LightGBM's test errors carry a systematic under-forecasting bias
(+1,423 MW in Feb 2025, +1,045 in March, +1,079 in June) in months whose demand level sits
above most of its training range. TimesFM normalises each context window, so it forecasts
*shape* rather than absolute level and never has to extrapolate a trend. This is the most
plausible reading of the gap, though the test period is a single stretch of time and we
cannot fully separate this from ordinary seasonal difficulty — January 2025 was hard for
every model, including both naives.

## Motivation

TimesFM 2.5 and Chronos-2 already top the GIFT-Eval leaderboard, which includes electricity
datasets, so "can a foundation model beat seasonal naive?" is close to settled and
reproducing it proves little.

The open question is whether models pretrained largely on Western data hold up on Indian
demand, which has features they plausibly under-represent: shifting festival dates,
monsoon-driven seasonality, and load-shedding artefacts. That is what this project tests.

A finding of "LightGBM wins" would have been reported just as plainly. It did not.

### An India-specific result worth stating separately

**The weekly seasonal naive loses badly to the daily one here** (8,329 vs 5,662 MW) — the
reverse of the usual pattern on Western grids, where weekends look nothing like weekdays.

The cause is structural, not a bug. In this data the **day-of-week spread is 4.41% of mean
against 20.29% for hour-of-day**, and nearly all of the weekly effect is Sunday alone
(−3.40%); Monday through Saturday sit within 1.4% of each other. Indian national demand is
dominated by industrial and agricultural load that runs seven days a week.

Weekly structure *is* present — lag-168 is a local minimum in error against its neighbours
at lag-144 (7,695 MW) and lag-192 (8,212 MW) — it is simply outweighed by seven days of
drift. `src/run_experiments.py` runs that local-minimum test automatically on every run,
because a *genuine* timestamp misalignment would also make the weekly naive look bad and
the two cases must be told apart.

## Data

**Source:** [Mendeley Data, DOI 10.17632/y58jknpgs8.2](https://doi.org/10.17632/y58jknpgs8.2)
— Mukherjee, Kalita & Kumar. **CC BY 4.0.** Originally Grid-India (NERLDC) SCADA, point
`NLDC_DEMAND|P`, i.e. all-India demand in **MW**.

**Coverage:** 2021-09-01 → 2025-06-24 IST, **33,432 hourly observations**, 46 months,
~4 annual cycles. Post-COVID throughout, so there is no pandemic demand collapse to model
around.

Full provenance, licence, anomalies and verification live in
**[`docs/data_audit.md`](docs/data_audit.md)**. The four findings that mattered:

- **The dataset's title says "1-hour interval", but only 1 of 29 files is hourly.** The
  rest are 5-minute (Sep 2021–Sep 2022) or 10-second (Oct 2022–Dec 2023) raw SCADA exports
  that must be aggregated, with demand in an unlabelled column under a two-row header.
- **The hourly file carries six trailing Excel summary rows** (Min/Max/Average/Sum). Read
  naively, the series reports a peak demand of 2.5 billion MW. Its timestamps are also
  `DD-MM-YYYY` strings, so `dayfirst=True` is mandatory.
- **Aggregation convention.** Hourly-mean and instantaneous-on-the-hour differ by 1.48%
  average / 6.08% worst — the same order as the accuracy being measured. Since one regime
  arrives pre-aggregated under an undocumented rule, a mismatch would have planted a
  structural break at 2024-01-01, inside the train/test span. A roughness test on adjacent
  months (Dec 2023 vs Jan 2024) identified the depositor's rule as hourly-mean, so all 46
  months are aggregated consistently.
- **SCADA zero-dropouts** (exact 0.0 MW sensor failures) are masked to NaN *before*
  averaging; leaving them in shifts hourly means by up to 2.3%.

**Quality after processing:** 12 missing hours out of 33,432 (**0.036%**), forward-filled
from past values only.

**External validity:** the series peaks at **248,688 MW on 2024-05-30 14:00 IST**, against
India's publicly reported record of ~250 GW on 30 May 2024 — same date, same magnitude.

## Methodology

**Splits are chronological, never random.** Random splitting would place 15:00 Tuesday in
train and 16:00 Tuesday in test, letting a model score well by interpolating between
neighbours while learning nothing about forecasting.

| Split | Period | Hours | Share |
|---|---|---|---|
| Train | 2021-09-01 → 2024-05-03 | 23,402 | 70% |
| Validation | 2024-05-03 → 2024-11-27 | 5,014 | 15% |
| Test | 2024-11-28 → 2025-06-24 | 5,016 | 15% |

**Rolling-origin evaluation:** context window → forecast 24h → step forward 6h → repeat.
The 6-hour stride (rather than 24) means all four times of day appear as origins; stepping
by a whole day would measure skill at one clock hour only.

**Every model sees identical origins**, fixed by the longest context any model needs
(720h). A model with a shorter warm-up would otherwise be judged on a different, easier
stretch of the test set.

**Selection never touches test.** LightGBM hyperparameters and the TimesFM context length
are both chosen on validation.

### Models

1. **Seasonal naive, daily** — `f(t+h) = y(t+h−24)`
2. **Seasonal naive, weekly** — `f(t+h) = y(t+h−168)`
3. **LightGBM** — direct multi-horizon: 24 independent models, one per h. Recursive
   one-step forecasting fed its own output would accumulate error over 24 steps and
   understate this tier. Features: lags 0/1/2/3/24/48/168/336h, rolling mean and std over
   trailing 24h and 168h, day/week differences, and target-time calendar (hour, weekday,
   month, weekend, `holidays.India`, cyclic encodings). Tuned on validation over an 8-point
   grid; the optimum sits in the interior at `num_leaves=15`.
4. **TimesFM 2.5 (200M)** — zero-shot, univariate, CPU. No fitting, no covariates, no
   calendar. Context length swept over 168/336/720, then extended to 896/1024 once 720
   proved to be the edge of the grid rather than the optimum.

**Chronos-2 was cut**, per the spec's scope-control order. TimesFM + LightGBM + two
baselines is already a complete comparison.

## Verification

```bash
pytest tests/ -q      # 41 tests
```

Three things are pinned, and they are what the results rest on:

- **Metrics against hand-computed values.** MAE, RMSE, sMAPE, MAPE, MASE, PICP and pinball
  loss on tiny arrays whose answers are checkable with arithmetic on paper.
- **Leakage** (`tests/test_leakage.py`). The feature frame is built from the whole series
  and again from the series truncated at origin T, and the row at T must be *identical*. A
  lag or rolling window that reached forward would differ, because the truncated series has
  no future to reach into. A separate test asserts the trailing 24h rolling std is zero at
  the last point before a step change — a centred window would already "see" the jump.
- **Timestamp alignment** (`tests/test_alignment.py`). On a synthetic ramp `y(t)=t`, the
  daily naive must be wrong by *exactly* 24 at every horizon and the weekly by exactly 168.
  Any other constant means forecasts are paired with the wrong actuals. Off-by-one
  alignment is the classic silent failure here: it still produces plausible metrics.

Actuals are joined to forecasts **by timestamp**, never by array position.

## Running it

```bash
py -3.12 -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m src.data.download        # 748 MB from Mendeley (~10 min)
python -m src.data.preprocess      # 29 workbooks -> hourly series (~7 min)
python -m src.eda                  # Plotly figures -> results/eda/
python -m src.run_experiments --stage all
streamlit run app/streamlit_app.py
```

Everything runs on CPU. Reference machine: Ryzen 5 5625U (6c/12t), 16 GB RAM. TimesFM
inference runs at ~1.4 origins/s, so the full test evaluation takes ~10 minutes per
context length; all forecasts are cached to `results/forecasts.parquet` so nothing is
recomputed. Seeds are fixed (`config.RANDOM_SEED = 42`).

`src/data/download.py` shells out to `curl` deliberately: Mendeley sits behind Cloudflare,
which rejects `python-requests` with HTTP 403 on TLS fingerprint regardless of User-Agent.

## Dashboard

```bash
streamlit run app/streamlit_app.py
```

Historical demand, a 24-hour forecast overlay with the P10–P90 band, a model selector
across all tiers, predicted peak demand and peak hour, and the results table. It reads only
the cached forecasts — it never invokes a foundation model, since seconds-per-window on CPU
would make the UI feel broken.

## Limitations

- **Single region.** All-India aggregate demand only. State and regional grids have sharper
  load-shedding artefacts and weaker aggregation smoothing, where the ranking may differ.
- **No covariates.** No temperature, no weather forecasts. Temperature is the single
  largest known driver of short-term demand, and its absence penalises LightGBM more than
  TimesFM, since LightGBM is the tier that could have used it.
- **Zero-shot only.** TimesFM was not fine-tuned. A fine-tuned foundation model or a
  LightGBM with weather inputs could reorder these results.
- **One test period.** 7 months, one winter-to-summer transition. The
  validation-to-test degradation seen in LightGBM cannot be fully separated from ordinary
  seasonal difficulty on a single stretch of time.
- **Aggregation inference.** The hourly-mean convention for the pre-aggregated regime was
  inferred from one adjacent-month comparison, not from documentation. It is the least
  certain claim in the data audit.
- **25 of 29 source files were verified by assertion, not by inspection.** The pipeline
  checks the sampling interval and grid completeness on every file it reads, but only four
  were opened and audited by hand.
- **No statistical significance testing.** Differences are large (TimesFM beats LightGBM by
  31% relative), but no Diebold-Mariano test was run, and the 832 origins overlap.

## Future work

- **Weather covariates**, using forecast-time-available values only. Using observed actuals
  would be a perfect-information upper bound and would have to be labelled as such.
- **Chronos-2**, which supports covariates natively.
- **Fine-tuning TimesFM** on Indian data, to separate "foundation model" from "zero-shot".
- **State-level forecasting**, where load-shedding is visible rather than averaged away.
- **Festival-date analysis** — Diwali and Holi shift against the Gregorian calendar, and
  isolating those days would test the original hypothesis about Western-pretrained models
  directly rather than in aggregate.
- **Diebold-Mariano tests** with an overlap-aware variance estimator.

## Project layout

```
data/{raw,processed}/            docs/data_audit.md
notebooks/01_data_exploration.ipynb   notebooks/02_model_evaluation.ipynb
src/config.py                    src/splits.py           src/eda.py
src/data/{download,preprocess}.py
src/models/{seasonal_naive,lightgbm_model,features,timesfm_model}.py
src/evaluation/{metrics,rolling_evaluation}.py
src/run_experiments.py           app/streamlit_app.py    tests/
results/{summary.csv,error_by_horizon.csv,run_metadata.json}
```

Notebooks display; they never compute. Every figure and metric comes from `src/`, so
everything the results depend on is reachable by the test suite.

## Licence and attribution

Code: MIT. Data: CC BY 4.0 — Mukherjee, Debanjan; Kalita, Karuna; Kumar, Subhash (2024),
*Electricity Demand, Solar and Wind Generation Data (September 2021 – June 2025) of India at
1-hour interval*, Mendeley Data, V2, doi: 10.17632/y58jknpgs8.2. Original data
Grid-India / NERLDC, Ministry of Power, Government of India.
