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
if "q0.1" in window.columns and window["q0.1"].notna().any():
    fig.add_trace(go.Scatter(x=window["target_time"], y=window["q0.9"], name="P90",
                             line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=window["target_time"], y=window["q0.1"], name="P10-P90",
                             fill="tonexty", fillcolor="rgba(99,110,250,0.22)",
                             line=dict(width=0)))
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
st.header("Historical demand")
fig3 = go.Figure()
daily = y.resample("D").mean()
fig3.add_trace(go.Scatter(x=daily.index, y=daily.to_numpy(), name="Daily mean",
                          line=dict(color="#636EFA", width=1)))
fig3.update_layout(height=320, yaxis_title="Demand (MW)", margin=dict(t=30))
st.plotly_chart(fig3, use_container_width=True)
st.caption(f"{y.index.min():%Y-%m-%d} to {y.index.max():%Y-%m-%d} — {len(y):,} hourly "
           f"observations, IST.")
