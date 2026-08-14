"""Minimal dashboard over the cached forecasts.

Reads only what `src/run_experiments.py` already wrote to results/. It never invokes a
foundation model: a 200M-parameter model on CPU takes seconds per window, which would make
every interaction feel broken. Run the experiments first, then this.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
import pathlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data.preprocess import load_processed  # noqa: E402

FORECASTS = config.RESULTS_DIR / "forecasts.parquet"
SUMMARY = config.RESULTS_DIR / "summary.csv"
HORIZON = config.RESULTS_DIR / "error_by_horizon.csv"

st.set_page_config(page_title="Indian Electricity Demand Forecasting", layout="wide")


@st.cache_data
def load_series() -> pd.Series:
    return load_processed()


@st.cache_data
def load_forecasts() -> pd.DataFrame:
    return pd.read_parquet(FORECASTS)


@st.cache_data
def load_table(path: pathlib.Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


st.title("Zero-shot foundation models vs. a tuned local model")
st.caption("24-hour-ahead all-India electricity demand. Data: Mendeley "
           f"{config.MENDELEY_DOI} (CC BY 4.0), via Grid-India SCADA.")

if not FORECASTS.exists():
    st.error("No cached forecasts found. Run `python -m src.run_experiments --stage all` first.")
    st.stop()

y = load_series()
fc = load_forecasts()
models = sorted(fc["model"].unique())

# --------------------------------------------------------------------------------------
# Results table
# --------------------------------------------------------------------------------------
st.header("Results")
summary = load_table(SUMMARY)
if summary is not None:
    st.dataframe(summary.round(3), use_container_width=True)
    st.caption("MASE is scaled by the in-sample MAE of the stronger seasonal naive on the "
               "training split, frozen and applied identically to every model.")

# --------------------------------------------------------------------------------------
# Forecast explorer
# --------------------------------------------------------------------------------------
st.header("24-hour forecast")
c1, c2 = st.columns([1, 2])
with c1:
    model = st.selectbox("Model", models,
                         index=models.index("timesfm") if "timesfm" in models else 0)
sub = fc[fc["model"] == model]
origins = sorted(sub["origin"].unique())
with c2:
    origin = st.select_slider("Forecast origin", options=origins,
                              value=origins[len(origins) // 2])

window = sub[sub["origin"] == origin].sort_values("h")
context = y.loc[:origin].iloc[-168:]

fig = go.Figure()
fig.add_trace(go.Scatter(x=context.index, y=context.to_numpy(), name="History (7 days)",
                         line=dict(color="#888", width=1.5)))
# Fan chart: nested decile bands, darkest in the middle.
for lo_q, hi_q, level in reversed(config.INTERVAL_LEVELS):
    lo_col, hi_col = f"q{lo_q:g}", f"q{hi_q:g}"
    if lo_col not in window.columns or window[lo_col].isna().all():
        continue
    shade = 0.10 + 0.16 * (1.0 - level)
    fig.add_trace(go.Scatter(x=window["target_time"], y=window[hi_col],
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=window["target_time"], y=window[lo_col],
                             name=f"P{int(lo_q*100)}–P{int(hi_q*100)}", fill="tonexty",
                             fillcolor=f"rgba(99,110,250,{shade:.2f})",
                             line=dict(width=0), hoverinfo="skip"))
fig.add_trace(go.Scatter(x=window["target_time"], y=window["actual"], name="Actual",
                         line=dict(color="#111", width=2.5)))
fig.add_trace(go.Scatter(x=window["target_time"], y=window["prediction"], name="Forecast",
                         line=dict(color="#EF553B", width=2.5, dash="dash")))
fig.update_layout(height=460, yaxis_title="Demand (MW)", xaxis_title="",
                  hovermode="x unified", margin=dict(t=30))
st.plotly_chart(fig, use_container_width=True)

peak_i = window["prediction"].idxmax()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Predicted peak", f"{window.loc[peak_i, 'prediction']:,.0f} MW")
m2.metric("Predicted peak hour", pd.Timestamp(window.loc[peak_i, "target_time"]).strftime("%H:%M"))
m3.metric("Actual peak", f"{window['actual'].max():,.0f} MW")
mae = (window["actual"] - window["prediction"]).abs().mean()
m4.metric("MAE this window", f"{mae:,.0f} MW")

# --------------------------------------------------------------------------------------
# Error by horizon
# --------------------------------------------------------------------------------------
st.header("Error by forecast horizon")
curve = load_table(HORIZON)
if curve is not None:
    fig2 = go.Figure()
    for name, d in curve.groupby("model"):
        fig2.add_trace(go.Scatter(x=d["h"], y=d["MAE"], name=name, mode="lines+markers"))
    fig2.update_layout(height=380, xaxis_title="Hours ahead", yaxis_title="MAE (MW)",
                       hovermode="x unified", margin=dict(t=30))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Accuracy degrades with horizon. A flat curve would suggest the model is "
               "ignoring recent information.")

# --------------------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------
cal = load_table(config.RESULTS_DIR / "calibration.csv")
if cal is not None and not cal.empty:
    st.header("Interval calibration")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect calibration",
                              line=dict(color="#999", dash="dash")))
    for name, d in cal.groupby("model"):
        d = d.sort_values("nominal")
        fig4.add_trace(go.Scatter(x=d["nominal"], y=d["empirical"], name=name,
                                  mode="lines+markers"))
    fig4.update_layout(height=420, xaxis_title="Nominal quantile",
                       yaxis_title="Empirical fraction below", hovermode="x unified",
                       margin=dict(t=30))
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("Points on the diagonal mean the stated quantiles are honest. Below the "
               "diagonal in the upper tail means the model is overconfident.")

    prob = load_table(config.RESULTS_DIR / "probabilistic.csv")
    if prob is not None:
        st.dataframe(prob.round(3), use_container_width=True)
        st.caption("CRPS scores the whole predictive distribution; Winkler charges interval "
                   "width plus a penalty for misses, so a uselessly wide interval cannot win.")

# --------------------------------------------------------------------------------------
# Significance
# --------------------------------------------------------------------------------------
dm = load_table(config.RESULTS_DIR / "significance_dm.csv")
if dm is not None and not dm.empty:
    st.header("Is the gap statistically significant?")
    cols = [c for c in ("model_A", "model_B", "winner", "dm_stat", "p_value",
                        "significant_1pct") if c in dm.columns]
    st.dataframe(dm[cols].round(4), use_container_width=True)
    st.caption("Diebold-Mariano on origin-level absolute loss, with a Newey-West HAC "
               "variance estimator — overlapping 24h windows are autocorrelated, and "
               "ignoring that would manufacture significance.")

st.header("Historical demand")
fig3 = go.Figure()
daily = y.resample("D").mean()
fig3.add_trace(go.Scatter(x=daily.index, y=daily.to_numpy(), name="Daily mean",
                          line=dict(color="#636EFA", width=1)))
fig3.update_layout(height=320, yaxis_title="Demand (MW)", margin=dict(t=30))
st.plotly_chart(fig3, use_container_width=True)
st.caption(f"{y.index.min():%Y-%m-%d} to {y.index.max():%Y-%m-%d} — {len(y):,} hourly "
           f"observations, IST.")
