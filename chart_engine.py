"""
chart_engine.py
---------------
Plotly chart factory — all charts use a consistent dark professional theme.
Each function returns a go.Figure ready to pass to st.plotly_chart().
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional


# ── Design System ───────────────────────────────────────────────────────────────
BG       = "#0A0E1A"
BG2      = "#0F1525"
PAPER    = "#111827"
FG       = "#E2E8F0"
FG2      = "#94A3B8"
GRID     = "#1E293B"
PALETTE  = ["#4F8EF7","#22D3A5","#F7874F","#A855F7","#F7C34F","#EF4444","#06B6D4","#84CC16","#F43F5E","#8B5CF6"]

LAYOUT_BASE = dict(
    paper_bgcolor=PAPER,
    plot_bgcolor =BG2,
    font         =dict(family="IBM Plex Sans, DM Sans, system-ui", color=FG2, size=12),
    title_font   =dict(family="Syne, DM Sans, system-ui", color=FG, size=16),
    margin       =dict(l=40, r=20, t=50, b=40),
    legend       =dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID, borderwidth=1, font_color=FG2),
    xaxis        =dict(gridcolor=GRID, zerolinecolor=GRID, tickfont_color=FG2, title_font_color=FG2),
    yaxis        =dict(gridcolor=GRID, zerolinecolor=GRID, tickfont_color=FG2, title_font_color=FG2),
    hoverlabel   =dict(bgcolor=PAPER, bordercolor=GRID, font_color=FG),
)


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0.01, xanchor="left"), **LAYOUT_BASE)
    return fig


def _fmt_currency(val: float) -> str:
    if val >= 1e6: return f"${val/1e6:.2f}M"
    if val >= 1e3: return f"${val/1e3:.1f}K"
    return f"${val:,.0f}"


# ── Chart Builders ──────────────────────────────────────────────────────────────

def bar_chart(df: pd.DataFrame, x_col: str, y_col: str,
              title: str = "", color_col: str = None,
              horizontal: bool = True, top_n: int = None) -> go.Figure:
    """Gradient bar chart with value annotations."""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return _empty_chart("No data available")

    df = df.copy()
    if top_n:
        df = df.nlargest(top_n, y_col)

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]

    if horizontal:
        fig = go.Figure(go.Bar(
            y=df[x_col].astype(str), x=df[y_col],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[_fmt_currency(v) for v in df[y_col]],
            textposition="outside", textfont=dict(color=FG2, size=11),
            hovertemplate=f"<b>%{{y}}</b><br>{y_col}: %{{x:,.0f}}<extra></extra>",
        ))
        fig.update_layout(yaxis=dict(autorange="reversed"))
    else:
        fig = go.Figure(go.Bar(
            x=df[x_col].astype(str), y=df[y_col],
            marker=dict(color=colors, line=dict(width=0)),
            text=[_fmt_currency(v) for v in df[y_col]],
            textposition="outside", textfont=dict(color=FG2, size=11),
            hovertemplate=f"<b>%{{x}}</b><br>{y_col}: %{{y:,.0f}}<extra></extra>",
        ))

    return _apply_theme(fig, title)


def line_chart(df: pd.DataFrame, x_col: str, y_cols: list[str],
               title: str = "", fill_area: bool = True) -> go.Figure:
    """Smooth line chart with optional area fill for trend visualization."""
    if df.empty:
        return _empty_chart("No data available")

    fig = go.Figure()
    for i, col in enumerate(y_cols):
        if col not in df.columns:
            continue
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x=df[x_col].astype(str), y=df[col],
            mode="lines+markers",
            name=col,
            line=dict(color=color, width=2.5, shape="spline"),
            marker=dict(size=6, color=color, symbol="circle"),
            fill="tozeroy" if (fill_area and i == 0) else "none",
            fillcolor=color.replace("#", "rgba(").rstrip(")") + ",0.08)" if color.startswith("#") else color,
            hovertemplate=f"<b>%{{x}}</b><br>{col}: $%{{y:,.0f}}<extra></extra>",
        ))

    return _apply_theme(fig, title)


def pie_donut_chart(df: pd.DataFrame, names_col: str, values_col: str,
                     title: str = "", donut: bool = True) -> go.Figure:
    """Professional donut / pie chart with % labels."""
    if df.empty or names_col not in df.columns or values_col not in df.columns:
        return _empty_chart("No data available")

    fig = go.Figure(go.Pie(
        labels=df[names_col].astype(str),
        values=df[values_col],
        hole=0.52 if donut else 0,
        marker=dict(colors=PALETTE[:len(df)], line=dict(color=BG, width=2.5)),
        textinfo="percent+label",
        textfont=dict(color=FG2, size=12),
        hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Share: %{percent}<extra></extra>",
        pull=[0.03 if i == 0 else 0 for i in range(len(df))],
    ))
    if donut:
        total = df[values_col].sum()
        fig.add_annotation(
            text=f"<b>{_fmt_currency(total)}</b><br><span style='font-size:11px'>Total</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(color=FG, size=16)
        )
    return _apply_theme(fig, title)


def grouped_bar_chart(df: pd.DataFrame, x_col: str,
                       bar_cols: list[str], title: str = "") -> go.Figure:
    """Multi-series grouped bar chart."""
    if df.empty:
        return _empty_chart("No data available")

    fig = go.Figure()
    for i, col in enumerate(bar_cols):
        if col not in df.columns:
            continue
        fig.add_trace(go.Bar(
            name=col, x=df[x_col].astype(str), y=df[col],
            marker=dict(color=PALETTE[i % len(PALETTE)], line=dict(width=0)),
            hovertemplate=f"<b>%{{x}}</b><br>{col}: $%{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(barmode="group")
    return _apply_theme(fig, title)


def area_chart(df: pd.DataFrame, x_col: str, y_col: str,
               title: str = "") -> go.Figure:
    """Smooth area chart — good for revenue trend."""
    if df.empty:
        return _empty_chart("No data available")

    color = PALETTE[0]
    fig = go.Figure(go.Scatter(
        x=df[x_col].astype(str), y=df[y_col],
        mode="lines+markers",
        line=dict(color=color, width=3, shape="spline"),
        marker=dict(size=7, color=color, symbol="circle",
                    line=dict(color=BG, width=2)),
        fill="tozeroy",
        fillcolor="rgba(79,142,247,0.10)",
        hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>",
    ))
    return _apply_theme(fig, title)


def heatmap_chart(df: pd.DataFrame, title: str = "") -> go.Figure:
    """Correlation or cross-tab heatmap."""
    if df.empty:
        return _empty_chart("No data available")

    colorscale = [[0, "#0A0E1A"], [0.5, "#1D4ED8"], [1, "#22D3A5"]]
    fig = go.Figure(go.Heatmap(
        z=df.values, x=df.columns.tolist(), y=df.index.tolist(),
        colorscale=colorscale,
        text=[[f"${v:,.0f}" for v in row] for row in df.values],
        texttemplate="%{text}", textfont=dict(size=10, color=FG),
        showscale=True,
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>$%{z:,.0f}<extra></extra>",
    ))
    return _apply_theme(fig, title)


def scatter_plot(df: pd.DataFrame, x_col: str, y_col: str,
                  color_col: str = None, size_col: str = None,
                  title: str = "") -> go.Figure:
    """Scatter plot with optional color and size dimensions."""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return _empty_chart("No data available")

    if color_col and color_col in df.columns:
        categories = df[color_col].unique()
        fig = go.Figure()
        for i, cat in enumerate(categories):
            sub = df[df[color_col] == cat]
            fig.add_trace(go.Scatter(
                x=sub[x_col], y=sub[y_col],
                mode="markers", name=str(cat),
                marker=dict(color=PALETTE[i % len(PALETTE)], size=9, opacity=0.8,
                            line=dict(color=BG, width=1)),
                hovertemplate=f"<b>{cat}</b><br>{x_col}: $%{{x:,.0f}}<br>{y_col}: $%{{y:,.0f}}<extra></extra>",
            ))
    else:
        fig = go.Figure(go.Scatter(
            x=df[x_col], y=df[y_col], mode="markers",
            marker=dict(color=PALETTE[0], size=9, opacity=0.8,
                        line=dict(color=BG, width=1)),
        ))

    return _apply_theme(fig, title)


def waterfall_chart(categories: list, values: list, title: str = "") -> go.Figure:
    """Waterfall chart for profit bridge or contribution analysis."""
    fig = go.Figure(go.Waterfall(
        name="", orientation="v",
        x=categories, y=values,
        connector=dict(line=dict(color=GRID, width=2)),
        increasing=dict(marker_color=PALETTE[1]),
        decreasing=dict(marker_color=PALETTE[5]),
        totals=dict(marker_color=PALETTE[0]),
        textposition="outside",
        text=[_fmt_currency(abs(v)) for v in values],
        hovertemplate="<b>%{x}</b><br>%{y:,.0f}<extra></extra>",
    ))
    return _apply_theme(fig, title)


def combo_bar_line(df: pd.DataFrame, x_col: str, bar_col: str,
                    line_col: str, title: str = "") -> go.Figure:
    """Combination chart: bars for one metric, line for another (e.g. revenue + margin)."""
    if df.empty:
        return _empty_chart("No data available")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=df[x_col].astype(str), y=df[bar_col],
        name=bar_col,
        marker=dict(color=PALETTE[0], opacity=0.8, line=dict(width=0)),
        hovertemplate=f"<b>%{{x}}</b><br>{bar_col}: $%{{y:,.0f}}<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df[x_col].astype(str), y=df[line_col],
        name=line_col, mode="lines+markers",
        line=dict(color=PALETTE[1], width=2.5),
        marker=dict(size=7),
        hovertemplate=f"<b>%{{x}}</b><br>{line_col}: %{{y:.1f}}%<extra></extra>",
    ), secondary_y=True)

    fig.update_yaxes(title_text=bar_col, secondary_y=False,
                     gridcolor=GRID, tickfont_color=FG2, title_font_color=FG2)
    fig.update_yaxes(title_text=line_col, secondary_y=True,
                     gridcolor=GRID, tickfont_color=FG2, title_font_color=FG2,
                     ticksuffix="%")
    fig.update_layout(**LAYOUT_BASE, title=dict(text=title, x=0.01))
    return fig


def forecast_chart(df: pd.DataFrame, date_col: str, metric_col: str,
                    title: str = "Revenue Forecast") -> go.Figure:
    """Dual-series chart: historical (solid) + forecast (dashed + CI band)."""
    if df.empty:
        return _empty_chart("No forecast data")

    hist = df[df.get("Type", pd.Series(["Historical"]*len(df))) == "Historical"]
    fore = df[df.get("Type", pd.Series([])) == "Forecast"]

    fig = go.Figure()
    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist["Period"].astype(str), y=hist[metric_col],
            mode="lines+markers", name="Historical",
            line=dict(color=PALETTE[0], width=2.5),
            marker=dict(size=6),
            fill="tozeroy", fillcolor="rgba(79,142,247,0.08)",
        ))
    if not fore.empty:
        std = hist[metric_col].std() * 0.5 if not hist.empty else 0
        fig.add_trace(go.Scatter(
            x=list(fore["Period"].astype(str)) + list(fore["Period"].astype(str))[::-1],
            y=list(fore[metric_col] + std) + list(fore[metric_col] - std)[::-1],
            fill="toself", fillcolor="rgba(168,85,247,0.10)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="CI Band",
        ))
        fig.add_trace(go.Scatter(
            x=fore["Period"].astype(str), y=fore[metric_col],
            mode="lines+markers+text", name="Forecast",
            line=dict(color=PALETTE[3], width=2.5, dash="dash"),
            marker=dict(size=8, symbol="diamond"),
            text=[f"${v:,.0f}" for v in fore[metric_col]],
            textposition="top center", textfont=dict(color=PALETTE[3], size=11),
        ))

    return _apply_theme(fig, title)


def kpi_gauge(value: float, max_value: float, title: str,
               color: str = "#4F8EF7") -> go.Figure:
    """Gauge chart for KPI visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title=dict(text=title, font=dict(color=FG2, size=13)),
        number=dict(font=dict(color=FG, size=28), prefix="$"),
        gauge=dict(
            axis=dict(range=[0, max_value], tickcolor=FG2),
            bar=dict(color=color),
            bgcolor=BG2,
            bordercolor=GRID,
            steps=[
                dict(range=[0, max_value*0.5], color=BG2),
                dict(range=[max_value*0.5, max_value*0.8], color="#1E293B"),
            ],
            threshold=dict(
                line=dict(color=PALETTE[2], width=3),
                thickness=0.8, value=max_value*0.8
            )
        )
    ))
    fig.update_layout(paper_bgcolor=PAPER, font=dict(color=FG2),
                       height=220, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def _empty_chart(message: str) -> go.Figure:
    """Return a placeholder chart when data is unavailable."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"⚠ {message}", x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="#EF4444"),
        xref="paper", yref="paper"
    )
    fig.update_layout(**LAYOUT_BASE, height=300)
    return fig
