# Data Audit — Phase 0

**Verdict: VIABLE.** 46 months of continuous hourly all-India electricity demand,
comfortably clearing the 2-year requirement. Proceed to Phase 1.

Audit date: 2026-08-14. Four of 29 files were opened and verified in full
(one per regime, plus December 2023 for the boundary test); the remaining 25 are
inferred from file size and naming and **will be asserted, not assumed, in Phase 1**.

---

## 1. Source and licence

| Field | Value |
|---|---|
| Title | Electricity Demand, Solar and Wind Generation Data (September 2021 – June 2025) of India at 1-hour interval |
| Authors | Mukherjee, Debanjan; Kalita, Karuna; Kumar, Subhash |
| Repository | Mendeley Data, version 2 |
| DOI | [10.17632/y58jknpgs8.2](https://doi.org/10.17632/y58jknpgs8.2) |
| URL | https://data.mendeley.com/datasets/y58jknpgs8/2 |
| **Licence** | **CC BY 4.0** — reuse and redistribution permitted with attribution |
| Original provenance | North-Eastern Regional Load Despatch Centre (NERLDC), Grid-India (formerly POSOCO), Ministry of Power, Government of India |
| Access | Public HTTP, no registration or API key |
| Size | 29 files, 748 MB total |

The demand series originates from the SCADA point `NLDC_DEMAND|P` — National Load
Despatch Centre demand, i.e. **all-India**, not a single region.

> **Operational note.** Mendeley sits behind Cloudflare, which rejects `python-requests`
> with HTTP 403 on TLS fingerprint regardless of `User-Agent`. `curl` passes. The Phase 1
> downloader therefore shells out to `curl -L` (file URLs answer 302). Download URLs come
> from the public manifest endpoint and carry a 2126 expiry, so they are stable.

## 2. Coverage and the three format regimes

**The dataset title is misleading: only 1 of 29 files is actually hourly.** The other 28
are sub-hourly SCADA exports that must be aggregated. This is the single most important
structural fact about the source.

| Regime | Period | Months | Files | Native frequency | Rows/file | File size |
|---|---|---|---|---|---|---|
| **A** | Sep 2021 – Sep 2022 | 13 | 13 | **5-minute** | ~8,640 | ~1.6 MB |
| **B** | Oct 2022 – Dec 2023 | 15 | 15 | **10-second** | 242k–268k | ~48 MB |
| **C** | Jan 2024 – Jun 2025 | 18 | 1 | **1-hour** (pre-aggregated) | 12,984 | 0.66 MB |

Total span **2021-09-01 00:00 → 2025-06-24 23:00 IST**, 46 months, no missing months.
That is ~33,400 hourly observations spanning nearly four annual cycles.

The sampling interval is recorded inside each workbook in a hidden `_osi_config` sheet
(`{"r":300.0}` = 5 min, `{"r":10.0}` = 10 s), which Phase 1 uses to verify the regime
rather than trusting file size.

### Schema

**Regimes A and B** — sheet `Sheet1`, 15 columns, **two header rows** (row 1 = SCADA tag,
row 2 = friendly name); data begins at row 3. Demand is **column index 10**,
`NLDC_DEMAND|P`. Companion columns cover thermal, hydro, gas, nuclear, wind, solar and
system inertia. Column 8's label drifts between files (`Solar+Wind` vs `Wind+Solar`) —
cosmetic, and not a column we use.

**Regime C** — sheet `Report`, columns:
`Timestamp | Demand (MW) | Wind (MW) | Solar (MW) | Total Generation (MW)`.

## 3. Units

**MW (megawatts), confirmed** — not MU. Stated explicitly in the regime C header
(`Demand (MW)`), and the regime A/B values occupy the same 100,000–250,000 range, which is
correct for all-India instantaneous demand. No unit conversion is required at any point;
the only transformation is temporal aggregation.

## 4. Timestamps, timezone, DST

- **Regimes A/B**: genuine Excel datetimes, ISO-ordered.
- **Regime C**: **strings in `DD-MM-YYYY HH:MM:SS`** — parsing requires `dayfirst=True`.
  Without it pandas silently misreads the first 12 days of every month as months.
  This is a live footgun and is covered by a Phase 1 assertion.
- **No timezone is attached to any file.** IST (UTC+05:30) is assumed, per Grid-India
  convention and consistent with the observed diurnal shape (demand trough ~03:00,
  peak ~14:00 and ~20:00 local).
- **India observes no daylight saving time.** Confirmed against the data: no 23- or
  25-hour days, no repeated or skipped local hours anywhere in the verified files.
  Timestamps are localised to `Asia/Kolkata` without ambiguity handling.

## 5. Missing and duplicate timestamps

Every verified file sits on a **perfectly regular grid with no gaps and no duplicates**:

| File | Expected stamps | Present | Missing | Duplicates | Monotonic |
|---|---|---|---|---|---|
| September 2021 (A, 5-min) | 8,640 | 8,640 | **0** | 0 | yes |
| February 2023 (B, 10-s) | 241,920 | 241,920 | **0** | 0 | yes |
| December 2023 (B, 10-s) | 267,840 | 267,840 | **0** | 0 | yes |
| Jan 2024–Jun 2025 (C, hourly) | 12,984 | 12,984 | **0** | 0 | yes |

Regime C's figures are *after* removing the summary rows described below. This is
unusually clean data — the completeness assertion in Phase 1 is expected to pass, and if
it fails on an unverified file that is itself the finding.

## 6. Anomalies

**6.1 — Six trailing summary rows in regime C (must be dropped).**
Rows 12,985–12,990 hold an Excel summary block: `Minimum`, `Timestamp`, `Maximum`,
`Timestamp`, `Average`, `Sum`. Read naively, the series reports a maximum demand of
**2.5 billion MW** and a mean of 387,938 MW. After dropping them: max 248,688 MW,
mean 194,010 MW.

**6.2 — The file's own summary cells are internally inconsistent; ignore them.**
`Sum` and `Average` reconcile exactly (2,519,025,649.08 / 12,984 = 194,009.98, matching the
computed mean). But the stated `Maximum` of 291,851.75 MW **appears in no data row**
(actual max 248,688.24 MW), and the stated `Minimum` of 0 contradicts the actual minimum of
97,065 MW. Cause unknown — likely stale or computed over a different range. Our figures are
computed from the data rows, not read from these cells.

**6.3 — SCADA zero-dropouts in regimes A/B (must be masked before averaging).**
Instantaneous drops to exactly 0.0 MW: 17 in September 2021, 22 in February 2023. These are
telemetry failures, not demand. Left unmasked they drag affected hourly means down by up to
**4,099 MW (2.3%)**. Phase 1 replaces exact zeros with NaN *before* resampling.

**6.4 — Two anomalous hours in regime C.**
`2024-04-22 16:00` (132,189 MW) and `2025-04-04 16:00` (97,065 MW), both more than 4σ below
the hour-of-day median profile — 0.015% of the series. Both fall at 16:00, suggesting a
reporting glitch rather than a real event (97 GW on an April afternoon is not physically
plausible against a ~200 GW April mean). Treated as missing and filled from past
information only.

**6.5 — Not anomalies.** December night-time values of 127–133 GW are legitimate winter
minima. There are no negative demand values. **There is no COVID-era demand collapse** —
the series begins September 2021, well after it, which removes a confound the audit brief
anticipated.

## 7. The aggregation convention — key methodological finding

Regimes A/B are sub-hourly and must be aggregated **by us**. Regime C arrives
**pre-aggregated by the depositor under an undocumented rule**. That matters more than it
first appears: on regime B, hourly-mean and instantaneous-on-the-hour differ by

- **1.48% of level on average**, 3.36% at the 95th percentile, **6.08% at worst**
  (correlation 0.984).

That is the same order of magnitude as the forecast accuracy the project sets out to
measure. If the depositor used a different rule from ours, the resulting **structural break
would land at 2024-01-01 — inside the train/test span** — and would confound every model
comparison in the project.

The regimes do not overlap, so the rule cannot be checked directly. Instead, adjacent
months were compared (December 2023, regime B, against January 2024, regime C — both
winter, so seasonality is broadly comparable) using a roughness statistic: the standard
deviation of the second difference as a percentage of level. Instantaneous sampling retains
high-frequency noise that averaging suppresses, so the statistic discriminates the rules.

| Series | Roughness |
|---|---|
| Dec 2023, hourly-**mean** | 2.5733% |
| Dec 2023, **instantaneous** | 4.8037% |
| **Jan 2024, regime C (unknown rule)** | **2.4513%** |

Regime C sits 0.12 from hourly-mean and 2.35 from instantaneous. **Regime C is hourly-mean
aggregated.** Phase 1 therefore aggregates regimes A and B by hourly mean, making all 46
months consistent. **No structural break.**

This inference rests on one adjacent-month pair and is the least certain claim in this
audit. It is recorded as a limitation, and Phase 2's EDA should be read with an eye on the
Jan-2024 boundary.

## 8. External validity

Three independent checks that the series is what it claims to be:

1. **Peak demand: 248,688 MW at 2024-05-30 14:00 IST.** India's publicly reported
   all-India record peak was ~250 GW on **30 May 2024** — same date, same magnitude.
2. **Monthly profile** shows textbook Indian seasonality: summer peak May–Jun (~210 GW),
   monsoon dip Jul–Aug, November trough (~173 GW). Monsoon-driven seasonality is one of the
   features this project hypothesises foundation models may under-represent, and it is
   clearly present.
3. **Year-on-year growth** of +2.5% (Jan) and +6.0% (Feb) between 2024 and 2025 — plausible
   for Indian demand growth.

## 9. Rejected alternative

**Kaggle — "Hourly Load India - Electrical Load Forecasting"** (`shubhamvashisht`),
Jan 2019 – Apr 2024, 46,728 hourly rows, national plus regional grids, with temperature.

Rejected because: it requires an account and API token; its **licence and provenance are
not clearly stated** (third-party derivation from POSOCO reports by an undocumented
method), which is disqualifying for a project whose credibility rests on methodology; and
it spans the COVID demand collapse, needing special handling for no analytical gain here.

Retained as a **fallback** if a Phase 1 assertion fails on an unverified Mendeley file, and
noted as a candidate cross-validation source. Its temperature column is the natural input
for the "weather covariates" item in Future Work.

## 10. Consequences for Phase 1

- Parse cost is **~16 minutes one-time** (~60 s per regime-B file × 15, plus ~2 s per
  regime-A file × 13). Processed output is cached to Parquet so this is paid once.
- Disk: 748 MB raw against 13 GB free — comfortable.
- Required rules, each of which becomes an assertion: `dayfirst=True` for regime C; drop
  non-parseable-timestamp rows; mask exact zeros to NaN **before** resampling; aggregate by
  hourly mean; localise to `Asia/Kolkata`; verify the `_osi_config` interval matches the
  assumed regime.
- Expected output: a single `timestamp | demand` frame of ~33,400 hourly rows,
  2021-09-01 → 2025-06-24, in MW.

With a 70/15/15 chronological split that gives roughly: **train** Sep 2021 – Apr 2024,
**validation** May 2024 – Nov 2024, **test** Dec 2024 – Jun 2025. The test set spans winter
through the summer peak, so model comparison is not confined to a single season.
