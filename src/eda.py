"""Exploratory plots (Phase 2).

Figure-building lives here rather than in the notebook so that the notebook stays what the
spec asks for -- exploration and display only, no important logic -- and so the same
figures can be regenerated headlessly for the README.

    python -m src.eda        # writes results/eda/*.html
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src import config, splits as S
from src.data.preprocess import load_processed

EDA_DIR = config.RESULTS_DIR / "eda"
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_LAYOUT = dict(template="plotly_white", height=420, margin=dict(t=60))


def fig_full_period(y: pd.Series) -> go.Figure:
    """Whole series, with a daily mean overlaid so the trend is legible."""
    daily = y.resample("D").mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y.index, y=y.to_numpy(), name="Hourly",
                             line=dict(color="rgba(99,110,250,0.25)", width=0.5)))
    fig.add_trace(go.Scatter(x=daily.index, y=daily.to_numpy(), name="Daily mean",
                             line=dict(color="#EF553B", width=1.6)))
    fig.update_layout(title="All-India demand, full period",
                      yaxis_title="Demand (MW)", **_LAYOUT)
    return fig


def fig_hour_of_day(y: pd.Series) -> go.Figure:
    """Mean demand by hour, with the interquartile band."""
    g = y.groupby(y.index.hour)
    mean, q1, q3 = g.mean(), g.quantile(0.25), g.quantile(0.75)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q3.index, y=q3.to_numpy(), line=dict(width=0),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=q1.index, y=q1.to_numpy(), fill="tonexty", name="IQR",
                             fillcolor="rgba(99,110,250,0.20)", line=dict(width=0)))
    fig.add_trace(go.Scatter(x=mean.index, y=mean.to_numpy(), name="Mean",
                             line=dict(color="#636EFA", width=3)))
    fig.update_layout(title="Mean demand by hour of day (IST)",
                      xaxis_title="Hour", yaxis_title="Demand (MW)", **_LAYOUT)
    return fig


def fig_day_of_week(y: pd.Series) -> go.Figure:
    """Mean demand by weekday — the effect the weekly naive baseline exploits."""
    g = y.groupby(y.index.dayofweek).mean()
    fig = go.Figure(go.Bar(x=DAY_NAMES, y=g.to_numpy(), marker_color="#636EFA"))
    fig.update_layout(title="Mean demand by day of week",
                      yaxis_title="Demand (MW)", **_LAYOUT)
    fig.update_yaxes(range=[g.min() * 0.95, g.max() * 1.02])
    return fig


def fig_weekly_profile(y: pd.Series) -> go.Figure:
    """Hour-of-week profile: 168 points showing the daily cycle inside the weekly one."""
    how = y.index.dayofweek * 24 + y.index.hour
    prof = y.groupby(how).mean()
    fig = go.Figure(go.Scatter(x=prof.index, y=prof.to_numpy(), mode="lines",
                               line=dict(color="#636EFA", width=2)))
    fig.update_layout(title="Weekly seasonality (mean demand by hour of week)",
                      xaxis_title="Hour of week (0 = Monday 00:00)",
                      yaxis_title="Demand (MW)", **_LAYOUT)
    fig.update_xaxes(tickmode="array", tickvals=list(range(0, 168, 24)), ticktext=DAY_NAMES)
    return fig


def fig_monthly(y: pd.Series) -> go.Figure:
    """Monthly means per year — monsoon dip and summer peak."""
    df = pd.DataFrame({"demand": y.to_numpy(), "year": y.index.year, "month": y.index.month})
    piv = df.groupby(["year", "month"])["demand"].mean().reset_index()
    fig = px.line(piv, x="month", y="demand", color="year", markers=True)
    fig.update_layout(title="Monthly mean demand by year",
                      xaxis_title="Month", yaxis_title="Demand (MW)", **_LAYOUT)
    fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)),
                     ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    return fig


def fig_distribution(y: pd.Series) -> go.Figure:
    """Demand distribution."""
    fig = go.Figure(go.Histogram(x=y.to_numpy(), nbinsx=80, marker_color="#636EFA"))
    fig.update_layout(title="Distribution of hourly demand",
                      xaxis_title="Demand (MW)", yaxis_title="Hours", **_LAYOUT)
    return fig


def fig_missing_map(y: pd.Series) -> go.Figure:
    """Where the gaps were before forward-filling, from the preprocessing report."""
    missing: list[str] = []
    if config.GAP_REPORT_PATH.exists():
        report = json.loads(config.GAP_REPORT_PATH.read_text(encoding="utf-8"))
        missing = report.get("missing_hours_list", [])

    counts = pd.Series(0, index=pd.date_range(y.index.min().tz_localize(None).normalize(),
                                              y.index.max().tz_localize(None).normalize(),
                                              freq="D"))
    if missing:
        m = pd.to_datetime(pd.Series(missing)).dt.normalize()
        vc = m.value_counts()
        counts.loc[counts.index.isin(vc.index)] = vc.reindex(
            counts.index[counts.index.isin(vc.index)]).to_numpy()

    fig = go.Figure(go.Bar(x=counts.index, y=counts.to_numpy(), marker_color="#EF553B"))
    total = int(counts.sum())
    fig.update_layout(
        title=f"Missing hours before forward-fill ({total} of {len(y):,} = "
              f"{total/len(y)*100:.3f}%)",
        xaxis_title="", yaxis_title="Missing hours that day", **_LAYOUT)
    return fig


def fig_splits(y: pd.Series) -> go.Figure:
    """Chronological split boundaries drawn over the series."""
    sp = S.chronological_splits(y)
    daily = y.resample("D").mean()
    fig = go.Figure()
    for name, colour in (("train", "#636EFA"), ("val", "#FFA15A"), ("test", "#EF553B")):
        idx = getattr(sp, name)
        seg = daily.loc[(daily.index >= idx[0]) & (daily.index <= idx[-1])]
        fig.add_trace(go.Scatter(x=seg.index, y=seg.to_numpy(), name=name,
                                 line=dict(color=colour, width=1.6)))
    fig.update_layout(title="Chronological splits (70 / 15 / 15) — never random",
                      yaxis_title="Daily mean demand (MW)", **_LAYOUT)
    return fig


FIGURES = {
    "01_full_period": fig_full_period,
    "02_hour_of_day": fig_hour_of_day,
    "03_day_of_week": fig_day_of_week,
    "04_weekly_profile": fig_weekly_profile,
    "05_monthly": fig_monthly,
    "06_distribution": fig_distribution,
    "07_missing_map": fig_missing_map,
    "08_splits": fig_splits,
}


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    y = load_processed()
    print(f"series: {len(y):,} hours  {y.index.min()} -> {y.index.max()}")
    for name, builder in FIGURES.items():
        fig = builder(y)
        path = EDA_DIR / f"{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        print(f"  wrote {path.name}")

    sp = S.chronological_splits(y)
    print()
    print(sp.describe().to_string(index=False))


if __name__ == "__main__":
    main()
