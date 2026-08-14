"""Central configuration: paths, constants, split boundaries, seeds.

Nothing in this project hard-codes a path or a magic number; it all lives here.
"""
from __future__ import annotations

import pathlib

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MENDELEY_RAW_DIR = RAW_DIR / "mendeley"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

MANIFEST_PATH = RAW_DIR / "mendeley_manifest.json"
PROCESSED_SERIES_PATH = PROCESSED_DIR / "demand_hourly.parquet"
GAP_REPORT_PATH = PROCESSED_DIR / "preprocessing_report.json"

for _d in (RAW_DIR, MENDELEY_RAW_DIR, PROCESSED_DIR, RESULTS_DIR, DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Data source — see docs/data_audit.md
# --------------------------------------------------------------------------------------
MENDELEY_DATASET_ID = "y58jknpgs8"
MENDELEY_VERSION = 2
MENDELEY_DOI = "10.17632/y58jknpgs8.2"
MENDELEY_MANIFEST_URL = (
    f"https://data.mendeley.com/public-api/datasets/{MENDELEY_DATASET_ID}/files"
    f"?folder_id=root&version={MENDELEY_VERSION}"
)

#: Mendeley is behind Cloudflare, which 403s python-requests on TLS fingerprint
#: regardless of User-Agent. curl's fingerprint passes, so all HTTP goes via curl.
USE_CURL_FOR_DOWNLOAD = True

TIMEZONE = "Asia/Kolkata"  # IST, UTC+05:30. India observes no DST.

#: Column index of NLDC_DEMAND|P in the regime A/B SCADA sheets (0-based).
SCADA_DEMAND_COL = 10
SCADA_TIME_COL = 0
SCADA_HEADER_ROWS = 2  # row 1 = SCADA tag, row 2 = friendly name; data starts row 3
SCADA_SHEET = "Sheet1"
HOURLY_SHEET = "Report"

#: The single pre-aggregated hourly file (regime C). Everything else is sub-hourly SCADA.
HOURLY_FILE_NAME = "January 2024- June 2025.xlsx"

#: Demand outside this band is physically implausible for all-India and is treated as
#: missing. India's record peak is ~250 GW; the observed series minimum is ~127 GW.
PLAUSIBLE_MIN_MW = 20_000.0
PLAUSIBLE_MAX_MW = 300_000.0

# --------------------------------------------------------------------------------------
# Forecasting problem
# --------------------------------------------------------------------------------------
HORIZON = 24  # hours ahead
SEASONAL_PERIOD_DAILY = 24
SEASONAL_PERIOD_WEEKLY = 168

#: Context lengths swept for the foundation model (hours), capped by TimesFM's max_context
#: of 1024. The sweep began at 168/336/720; validation error fell monotonically across
#: those three, putting the optimum on the edge of the grid, so it was extended to 896 and
#: 1024 to bracket the minimum. It lands at 896 (1024 is within 0.1% and effectively tied).
CONTEXT_LENGTHS = (168, 336, 720, 896, 1024)
DEFAULT_CONTEXT = 896

#: Rolling-origin stride. 6h keeps ~830 origins over the test set and samples all four
#: times of day as origins, so the horizon curve is not measured at one clock hour only.
ORIGIN_STRIDE_HOURS = 6

#: Chronological splits. NEVER random — that would leak future into past.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

#: All nine deciles. TimesFM emits these natively, and matching them in LightGBM (nine
#: quantile objectives per horizon) buys a genuine distributional comparison: CRPS, Winkler
#: scores, and a calibration curve across five nominal interval levels rather than
#: coverage at a single 80% band.
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

#: Central intervals formed from symmetric decile pairs, as (lower, upper, nominal level).
INTERVAL_LEVELS = (
    (0.4, 0.6, 0.2),
    (0.3, 0.7, 0.4),
    (0.2, 0.8, 0.6),
    (0.1, 0.9, 0.8),
)
NOMINAL_COVERAGE = 0.8  # headline P10-P90 interval

# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
RANDOM_SEED = 42

# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
TIMESFM_CHECKPOINT = "google/timesfm-2.5-200m-pytorch"
TIMESFM_MAX_CONTEXT = 1024
TIMESFM_MAX_HORIZON = 256

MODEL_LABELS = {
    "seasonal_naive_daily": "Seasonal naive (m=24)",
    "seasonal_naive_weekly": "Seasonal naive (m=168)",
    "lightgbm": "LightGBM",
    "timesfm": "TimesFM 2.5 (zero-shot)",
    "timesfm_ctx896": "TimesFM 2.5, ctx 896 (zero-shot)",
}
