# Do zero-shot foundation models beat a tuned local model for Indian electricity demand?

**24-hour-ahead all-India electricity demand forecasting.** Four model tiers, rolling-origin
evaluation over 832 forecast origins on a 7-month held-out test set, with significance
testing and full distributional scoring.

> **Answer: yes, decisively — and the tuned local model turns out not to beat a seasonal
> naive at all.**
>
> Zero-shot TimesFM 2.5 cut MAE by **35.7%** against the stronger baseline
> (Diebold-Mariano *p* < 0.0001). A tuned LightGBM managed 7.6%, which **is not
> statistically significant** (*p* = 0.126). The foundation model was given no calendar, no
> holidays, no covariates and no fitting of any kind.

---

## Results

Test set **2024-11-28 → 2025-06-24**. 832 origins, every 6 hours, horizon 24h —
**19,968 forecasts per model**, on identical origins for every tier.

### Point forecast accuracy

| Model | MAE (MW) | RMSE (MW) | sMAPE (%) | MAPE (%) | MASE | MAE skill | RMSE skill |
|---|---|---|---|---|---|---|---|
| **TimesFM 2.5, ctx 896** (zero-shot) | **3,642** | **5,838** | **1.88** | **1.89** | **0.819** | **+35.7%** | **+32.2%** |
| TimesFM 2.5, ctx 720 (zero-shot) | 3,719 | 5,894 | 1.92 | 1.93 | 0.837 | +34.3% | +31.6% |
| LightGBM (tuned) | 5,231 | 7,311 | 2.71 | 2.72 | 1.177 | +7.6% | +15.1% |
| Seasonal naive, m=24 *(baseline)* | 5,662 | 8,616 | 2.95 | 2.96 | 1.274 | — | — |
| Seasonal naive, m=168 | 8,329 | 11,742 | 4.29 | 4.29 | 1.874 | −47.1% | −36.3% |

MASE is scaled by the in-sample MAE of the stronger seasonal naive on the **training** split
(**4,445.2 MW**, m=24), frozen once and applied identically to every model. Skill is quoted
against that same stronger baseline — never the weaker one.

**MASE < 1 means beating the in-sample naive. Only TimesFM does.** LightGBM's 1.177 says
that even while edging the naive on this test set, it remains worse than the naive was
in-sample.

### Statistical significance

Diebold-Mariano on origin-level absolute loss, Newey-West HAC variance, Harvey small-sample
correction. Consecutive 24h windows at a 6h stride **share target hours**, so their loss
differentials are autocorrelated; treating them as independent would manufacture
significance.

| Comparison | Winner | DM stat | *p* | Significant (1%) |
|---|---|---|---|---|
| TimesFM ctx896 vs LightGBM | **TimesFM** | 9.011 | <0.0001 | ✅ |
| TimesFM ctx896 vs naive m=24 | **TimesFM** | 8.883 | <0.0001 | ✅ |
| TimesFM ctx896 vs naive m=168 | **TimesFM** | 10.168 | <0.0001 | ✅ |
| **LightGBM vs naive m=24** | LightGBM | −1.533 | **0.126** | ❌ **not significant** |
| LightGBM vs naive m=168 | LightGBM | −5.703 | <0.0001 | ✅ |
| naive m=24 vs naive m=168 | naive m=24 | −4.829 | <0.0001 | ✅ |
| TimesFM ctx896 vs ctx720 | ctx896 | 1.986 | 0.047 | ❌ (5% only) |

**The single most important line in this table is the LightGBM one.** A carefully tuned
gradient-boosting model with lag, rolling, calendar and holiday features cannot be
distinguished from "yesterday, same hour" at any conventional significance level. The
foundation model can, overwhelmingly.

### Distributional accuracy

All models emit **9 deciles** (LightGBM via 9 quantile objectives × 24 horizons = 216
boosters; TimesFM natively). CRPS scores the entire predictive distribution; Winkler charges
interval width *plus* a penalty for misses, so an absurdly wide interval cannot win the way
raw coverage would allow.

| Model | CRPS | Mean pinball | Winkler@80 | Width@80 (MW) |
|---|---|---|---|---|
| **TimesFM ctx896** | **2,906** | **1,453** | **18,110** | 11,666 |
| TimesFM ctx720 | 2,971 | 1,486 | 18,508 | 11,891 |
| LightGBM | 4,100 | 2,050 | 26,941 | 10,638 |

TimesFM's CRPS is **29% better**. Note LightGBM's *narrower* 80% interval alongside its far
worse Winkler score — the interval is not sharp, it is overconfident.

> **Quantile crossing, found and fixed.** Fitting nine independent quantile regressors gives
> no monotonicity guarantee, and LightGBM's estimates crossed on **83% of rows**, with
> inversions up to **9,606 MW** — a predicted 40th percentile above the predicted 60th. That
> is not a valid distribution and it corrupts CRPS, coverage and width. All quantile
> predictions now pass through row-wise rearrangement (Chernozhukov, Fernández-Val &
> Galichon 2010), which is guaranteed to weakly *reduce* error. It improved LightGBM
> (CRPS 4,186 → 4,100, PICP@80 0.552 → 0.583), so the numbers above are the *charitable*
> ones. TimesFM crossed on 0% of rows — it is compiled with `fix_quantile_crossing=True`.

### Interval calibration

Empirical coverage against nominal, across four central intervals:

| Nominal | 20% | 40% | 60% | 80% |
|---|---|---|---|---|
| **TimesFM ctx896** | **0.209** | **0.413** | **0.614** | **0.820** |
| TimesFM ctx720 | 0.208 | 0.412 | 0.606 | 0.809 |
| LightGBM | 0.128 | 0.259 | 0.403 | 0.583 |

An 80% interval is *expected* to contain the actual about 80% of the time — **not always**.
TimesFM lands within 1.4 points of nominal at every level, so its stated uncertainty is
close to honest. LightGBM is badly overconfident everywhere: its "80%" interval covers only
**58.3%**, which would systematically understate risk if used to size operating reserve.

Coverage also holds up across the horizon for TimesFM but not LightGBM:

| 80% coverage at | h=1 | h=12 | h=24 |
|---|---|---|---|
| TimesFM ctx896 | 0.935 | 0.833 | 0.828 |
| LightGBM | 0.541 | 0.606 | 0.722 |

TimesFM is somewhat *over*-covered at h=1 (0.935) — its one-step intervals are wider than
they need to be.

---

## Three things the headline number hides

### 1. LightGBM overtakes TimesFM at long horizons

| Horizon | h=1 | h=3 | h=6 | h=12 | h=18 | h=20 | **h=21** | **h=24** |
|---|---|---|---|---|---|---|---|---|
| TimesFM ctx896 | **1,272** | **2,207** | **3,163** | **3,789** | **4,222** | **4,283** | 4,417 | 4,327 |
| LightGBM | 2,333 | 4,829 | 5,111 | 5,390 | 5,823 | 4,656 | **4,390** | **4,051** |
| Naive m=24 | 5,509 | 5,711 | 5,619 | 5,623 | 5,624 | 5,539 | 5,713 | 5,627 |

TimesFM is **1.8× more accurate at h=1** but degrades 3.4× across the day; LightGBM degrades
only 1.7× and **wins outright from h≥21**, where explicit hour-of-day and calendar features
let it anchor on "same hour tomorrow". If you only need day-ahead demand at a fixed delivery
hour, the cheap model remains competitive.

Both seasonal naives are essentially **flat** across horizon (growth factors 1.02 and 0.98),
which is the expected signature and a useful check that the harness is sound.

### 2. LightGBM systematically under-predicts the daily peak

The daily peak sizes generating reserve, so it deserves scoring separately from average
error.

| Model | Peak MAE (MW) | Peak bias (MW) | Peak MAPE | Peak hour exact | Within ±1h |
|---|---|---|---|---|---|
| **TimesFM ctx896** | **3,736** | +631 | **1.71%** | **72.5%** | **87.4%** |
| TimesFM ctx720 | 3,856 | +748 | 1.77% | 72.2% | 86.8% |
| Naive m=24 | 4,993 | **+75** | 2.31% | 65.3% | 82.9% |
| LightGBM | 6,000 | **+4,918** | 2.70% | 44.5% | 76.0% |
| Naive m=168 | 8,346 | +534 | 3.81% | 60.0% | 78.4% |

*(Positive bias = under-forecast.)*

LightGBM under-predicts the daily peak by **4,918 MW on average** — roughly 2.4% of national
demand, and an order of magnitude worse than every other tier including both naives. It also
identifies the correct peak hour only **44.5%** of the time, worse than the seasonal naive's
65.3%.

This is the expected failure mode of a regression model trained on a symmetric loss: it
regresses toward the conditional mean and smooths extremes. It matters operationally far
more than the MAE gap does — a model that quietly shaves 5 GW off every peak is dangerous
for capacity planning in a way its average error does not reveal.

### 3. LightGBM degrades out-of-sample; TimesFM does not

LightGBM scored **4,003 MW on validation** but **5,231 MW on test** (+31%). TimesFM was
essentially flat (3,568 → 3,642).

| MAE by month | Nov 24 | Dec 24 | Jan 25 | Feb 25 | Mar 25 | Apr 25 | May 25 | Jun 25 |
|---|---|---|---|---|---|---|---|---|
| TimesFM ctx896 | 2,185 | **2,127** | 4,343 | **2,026** | **3,426** | **4,071** | **4,659** | 5,204 |
| LightGBM | 2,589 | 3,396 | 6,470 | 6,864 | 5,171 | 4,588 | 5,439 | **4,987** |
| Naive m=24 | **1,951** | 4,278 | 6,366 | 4,081 | 5,558 | 5,994 | 7,489 | 6,155 |

LightGBM's worst months (Feb 6,864; Jan 6,470) carry systematic **under-forecasting bias**
(+1,423 MW in Feb, +1,045 in Mar, +1,079 in Jun) in months whose demand level sits above
most of its training range. TimesFM normalises each context window, so it forecasts *shape*
rather than absolute level and never has to extrapolate a trend.

This is the most plausible reading, but the test period is a single stretch of time and
cannot fully separate this from ordinary seasonal difficulty — **January 2025 was hard for
every model**, including both naives.

Hardest hour of day differs too: TimesFM's worst target hour is 16:00 (4,641 MW), LightGBM's
is 08:00 (6,876 MW) — the morning ramp.

---

## Motivation

TimesFM 2.5 and Chronos-2 already top the GIFT-Eval leaderboard, which includes electricity
datasets, so *"can a foundation model beat seasonal naive?"* is close to settled and
reproducing it proves little.

The open question is whether models pretrained largely on **Western** data hold up on
**Indian** demand, which has features they plausibly under-represent: shifting festival
dates, monsoon-driven seasonality, and load-shedding artefacts. That is what this tests.

A finding of "LightGBM wins" would have been reported just as plainly. It did not — and the
finding that LightGBM cannot beat a naive baseline significantly is reported here just as
plainly too.

### An India-specific result worth stating separately

**The weekly seasonal naive loses badly to the daily one here** (8,329 vs 5,662 MW,
*p* < 0.0001) — the reverse of the usual pattern on Western grids, where weekends look
nothing like weekdays.

The cause is structural, not a bug. In this data the **day-of-week spread is 4.41% of mean
against 20.29% for hour-of-day**, and nearly all of the weekly effect is Sunday alone
(−3.40%); Monday through Saturday sit within 1.4% of each other. Indian national demand is
dominated by industrial and agricultural load that runs seven days a week.

Weekly structure *is* present — lag-168 is a **local minimum** in error against its
neighbours at lag-144 (7,695 MW) and lag-192 (8,212 MW) — it is simply outweighed by seven
days of drift. `src/run_experiments.py` runs that local-minimum test automatically on every
run, because a *genuine* timestamp misalignment would also make the weekly naive look bad
and the two cases must be told apart.

---

## Data

**Source:** [Mendeley Data, DOI 10.17632/y58jknpgs8.2](https://doi.org/10.17632/y58jknpgs8.2)
— Mukherjee, Kalita & Kumar. **CC BY 4.0.** Originally Grid-India (NERLDC) SCADA, point
`NLDC_DEMAND|P`: all-India demand in **MW**.

**Coverage:** 2021-09-01 → 2025-06-24 IST, **33,432 hourly observations**, 46 months,
~4 annual cycles. Post-COVID throughout, so there is no pandemic demand collapse to model
around.

Full provenance, licence, anomalies and verification live in
**[`docs/data_audit.md`](docs/data_audit.md)**. The four findings that mattered:

- **The dataset's title says "1-hour interval", but only 1 of 29 files is hourly.** The rest
  are 5-minute (Sep 2021–Sep 2022) or 10-second (Oct 2022–Dec 2023) raw SCADA exports that
  must be aggregated, with demand in an unlabelled column under a two-row header.
- **The hourly file carries six trailing Excel summary rows** (Min/Max/Average/Sum). Read
  naively, the series reports a peak demand of **2.5 billion MW**. Its timestamps are also
  `DD-MM-YYYY` strings, so `dayfirst=True` is mandatory.
- **Aggregation convention.** Hourly-mean and instantaneous-on-the-hour differ by 1.48%
  average / 6.08% worst — the same order as the accuracy being measured. Since one regime
  arrives pre-aggregated under an undocumented rule, a mismatch would have planted a
  structural break at 2024-01-01, *inside the train/test span*. A roughness test on adjacent
  months (Dec 2023 vs Jan 2024) identified the depositor's rule as hourly-mean, so all 46
  months are aggregated consistently.
- **SCADA zero-dropouts** (exact 0.0 MW sensor failures) are masked to NaN *before*
  averaging; leaving them in shifts hourly means by up to 2.3%.

**Quality after processing:** 12 missing hours out of 33,432 (**0.036%**), forward-filled
from past values only.

**External validity:** the series peaks at **248,688 MW on 2024-05-30 14:00 IST**, against
India's publicly reported record of ~250 GW on 30 May 2024 — same date, same magnitude.

---

## Methodology

**Splits are chronological, never random.** Random splitting would place 15:00 Tuesday in
train and 16:00 Tuesday in test, letting a model score well by interpolating between
neighbours while learning nothing about forecasting.

| Split | Period | Hours | Share |
|---|---|---|---|
| Train | 2021-09-01 → 2024-05-03 | 23,402 | 70% |
| Validation | 2024-05-03 → 2024-11-27 | 5,014 | 15% |
| Test | 2024-11-28 → 2025-06-24 | 5,016 | 15% |

**Rolling-origin evaluation:** context window → forecast 24h → step forward 6h → repeat. The
6-hour stride (rather than 24) means all four times of day appear as origins; stepping by a
whole day would measure skill at one clock hour only.

**Every model sees identical origins**, fixed by the longest context any model needs. A
model with a shorter warm-up would otherwise be judged on a different, easier stretch of the
test set.

**Selection never touches test.** LightGBM hyperparameters and the TimesFM context length
are both chosen on validation.

### Models

1. **Seasonal naive, daily** — `f(t+h) = y(t+h−24)`
2. **Seasonal naive, weekly** — `f(t+h) = y(t+h−168)`
3. **LightGBM** — direct multi-horizon: 24 independent models, one per h. Recursive one-step
   forecasting fed its own output would accumulate error over 24 steps and understate this
   tier. Features: lags 0/1/2/3/24/48/168/336h; rolling mean and std over trailing 24h and
   168h; day and week differences; target-time calendar (hour, weekday, month, weekend,
   `holidays.India`, cyclic encodings). Tuned on validation over an 8-point grid.
4. **TimesFM 2.5 (200M)** — zero-shot, univariate, CPU. No fitting, no covariates, no
   calendar.

**Both grids were extended after hitting their boundary.** LightGBM's first sweep put the
optimum at the smallest `num_leaves` tried, and TimesFM's context sweep fell monotonically
across the specified 168/336/720. Reporting a "tuned" model whose optimum sits on a grid edge
is undertuning, so LightGBM was extended down to 7/15 leaves (optimum now interior at **15**)
and TimesFM out to 896/1024 (optimum **896**; 1024 is within 0.1% and effectively tied):

| TimesFM context | 168 | 336 | 720 | **896** | 1024 |
|---|---|---|---|---|---|
| Validation MAE (MW) | 4,513 | 3,961 | 3,701 | **3,568** | 3,571 |

**Chronos-2 was cut**, per the spec's scope-control order. TimesFM + LightGBM + two baselines
is already a complete comparison.

---

## Verification

```bash
pytest tests/ -q      # 58 tests
```

The results rest on four things being pinned:

- **Metrics against hand-computed values.** MAE, RMSE, sMAPE, MAPE, MASE, PICP, pinball,
  Winkler and CRPS on tiny arrays whose answers are checkable with arithmetic on paper.
  Includes a test that Winkler *punishes* a uselessly wide interval that scores perfect PICP,
  and that quantile rearrangement weakly reduces pinball loss.
- **Leakage** (`tests/test_leakage.py`). The feature frame is built from the whole series and
  again from the series truncated at origin T, and the row at T must be **identical**. A lag
  or rolling window that reached forward would differ, because the truncated series has no
  future to reach into. A separate test asserts the trailing 24h rolling std is zero at the
  last point before a step change — a centred window would already "see" the jump.
- **Timestamp alignment** (`tests/test_alignment.py`). On a synthetic ramp `y(t)=t`, the daily
  naive must be wrong by **exactly 24** at every horizon and the weekly by exactly 168. Any
  other constant means forecasts are paired with the wrong actuals. Off-by-one alignment is
  the classic silent failure here: it still produces entirely plausible metrics.
- **HAC variance** (`tests/test_probabilistic.py`). Newey-West must exceed the naive variance
  under positive autocorrelation and match it for white noise — otherwise the significance
  tests are decoration.

Actuals are joined to forecasts **by timestamp**, never by array position. The TimesFM
quantile-axis layout is asserted at load time rather than trusted from documentation, since
silently reading the wrong column would corrupt every interval metric above.

---

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

Everything runs on **CPU**. Reference machine: Ryzen 5 5625U (6c/12t), 16 GB RAM. TimesFM
inference runs at ~1.3 origins/s, so a full test evaluation takes ~11 minutes per context
length; all forecasts are cached to `results/forecasts.parquet` so nothing is recomputed.
Seeds fixed (`config.RANDOM_SEED = 42`).

`src/data/download.py` shells out to `curl` deliberately: Mendeley sits behind Cloudflare,
which rejects `python-requests` with HTTP 403 on TLS fingerprint regardless of User-Agent.

### Generated result tables

| File | Contents |
|---|---|
| `summary_detailed.csv` | Headline point metrics and skill scores |
| `error_by_horizon_detailed.csv` | MAE/RMSE/sMAPE/MASE at each h=1..24 |
| `significance_dm.csv` | Pairwise Diebold-Mariano tests |
| `probabilistic.csv` | CRPS, Winkler, PICP and width at four interval levels |
| `calibration.csv` | Empirical vs nominal at all nine deciles |
| `coverage_by_horizon.csv` | 80% coverage at each horizon |
| `peak_metrics.csv` | Daily-peak level and peak-hour accuracy |
| `quantile_crossing.csv` | Monotonicity violations per model |
| `metrics_by_month.csv` | MAE and bias per calendar month |
| `error_by_time_of_day.csv` | MAE by target hour |

---

## Dashboard

```bash
streamlit run app/streamlit_app.py
```

Historical demand, a 24-hour forecast overlay with a **decile fan chart**, model selector
across all tiers, predicted peak demand and peak hour, error-by-horizon curve, reliability
diagram and the significance table. It reads only the cached forecasts — it never invokes a
foundation model, since seconds-per-window on CPU would make the UI feel broken.

---

## Limitations

- **Single region.** All-India aggregate demand only. State and regional grids have sharper
  load-shedding artefacts and less aggregation smoothing, where the ranking may differ.
- **No covariates.** No temperature, no weather forecasts. Temperature is the single largest
  known driver of short-term demand, and its absence penalises **LightGBM more than
  TimesFM**, since LightGBM is the tier that could have exploited it. This is the most
  important caveat on the headline claim: a fair "tuned local model" in production would
  have weather inputs.
- **Zero-shot only.** TimesFM was not fine-tuned, so this compares *zero-shot foundation
  model* against *tuned local model*, not *foundation architecture* against *GBM*.
- **One test period.** 7 months, one winter-to-summer transition. The validation-to-test
  degradation in LightGBM cannot be fully separated from ordinary seasonal difficulty.
- **Aggregation inference.** The hourly-mean convention for the pre-aggregated regime was
  inferred from one adjacent-month comparison, not from documentation. It is the least
  certain claim in the data audit.
- **25 of 29 source files were verified by assertion, not by inspection.** The pipeline
  checks sampling interval and grid completeness on every file it reads; only four were
  opened and audited by hand.
- **CRPS is a 9-decile approximation**, not an exact integral over the predictive CDF.
- **LightGBM's quantiles are rearranged, not jointly estimated.** Rearrangement fixes
  monotonicity after the fact; a genuinely joint quantile model might do better than the
  numbers here.
- **Determinism was not re-verified** by running preprocessing twice end-to-end, though the
  pipeline contains no stochastic component.

## Future work

- **Weather covariates**, using forecast-time-available values only. Using observed actuals
  would be a perfect-information upper bound and must be labelled as such.
- **Chronos-2**, which supports covariates natively.
- **Fine-tuning TimesFM** on Indian data, to separate "foundation model" from "zero-shot".
- **Quantile-loss LightGBM for the peak specifically**, or an asymmetric loss, to address the
  4,918 MW peak under-prediction.
- **State-level forecasting**, where load-shedding is visible rather than averaged away.
- **Festival-date analysis** — Diwali and Holi shift against the Gregorian calendar;
  isolating those days would test the original hypothesis about Western-pretrained models
  directly rather than in aggregate.

---

## Project layout

```
data/{raw,processed}/                  docs/data_audit.md
notebooks/01_data_exploration.ipynb    notebooks/02_model_evaluation.ipynb
src/config.py       src/splits.py      src/eda.py       src/run_experiments.py
src/data/{download,preprocess}.py
src/models/{seasonal_naive,features,lightgbm_model,timesfm_model}.py
src/evaluation/{metrics,rolling_evaluation,report,statistical_tests}.py
app/streamlit_app.py                   tests/           results/
```

Notebooks display; they never compute. Every figure and metric comes from `src/`, so
everything the results depend on is reachable by the test suite.

## Licence and attribution

Code: MIT. Data: CC BY 4.0 — Mukherjee, Debanjan; Kalita, Karuna; Kumar, Subhash (2024),
*Electricity Demand, Solar and Wind Generation Data (September 2021 – June 2025) of India at
1-hour interval*, Mendeley Data, V2, doi: 10.17632/y58jknpgs8.2. Original data Grid-India /
NERLDC, Ministry of Power, Government of India.
