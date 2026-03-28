"""
chart_engine.py — DataLensAI v2.0
Chart Configuration Engine

Responsibilities:
  - Auto-select chart types based on detected column semantics
  - Build Chart.js-compatible config dicts (for frontend rendering)
  - Build Plotly figure dicts (for server-side rendering / export)
  - Generate download-ready Plotly PNG charts via kaleido

Supported Chart Types:
  - Bar (vertical + horizontal)
  - Line with fill (trend)
  - Doughnut
  - Grouped / Stacked Bar (multi-series)
  - Scatter (correlation)
  - Area (monthly cumulative)

Author: Agam Kumar Verma
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from data_engine import DataEngine
from dataset_profiler import DatasetProfiler, _fmt, _trunc

log = logging.getLogger(__name__)

# ── Color palettes ────────────────────────────────────────────────────────────
PAL = [
    "#7C3AED", "#10B981", "#F59E0B", "#F87171",
    "#38BDF8", "#EC4899", "#F97316", "#A78BFA",
    "#34D399", "#60A5FA", "#FCD34D", "#FB7185",
]
PAL_ALPHA = [c + "BB" for c in PAL]


class ChartEngine:
    """
    Builds chart configurations from a DataEngine + DatasetProfiler pair.

    Usage
    -----
    engine  = DataEngine(...).load_from_bytes(content).clean()
    profiler = DatasetProfiler(engine)
    charts   = ChartEngine(engine, profiler).build_all_charts()
    """

    def __init__(self, engine: DataEngine, profiler: DatasetProfiler) -> None:
        self.engine   = engine
        self.profiler = profiler

    # ══════════════════════════════════════════════════════════════════════
    # BUILD ALL CHARTS (Chart.js format)
    # ══════════════════════════════════════════════════════════════════════
    def build_all_charts(self) -> list[dict]:
        """Return a list of Chart.js-compatible chart config dicts."""
        charts = []
        e = self.engine

        # ── 1. Revenue by Category (bar) ──────────────────────────────────
        if e.category_col and e.revenue_col:
            data = e.group_sum(e.category_col, e.revenue_col, top_n=8)
            if data:
                charts.append(self._bar_chart(
                    chart_id="revenue_by_category",
                    title=f"Revenue by {e.category_col.replace('_',' ')}",
                    sub="Total revenue per category",
                    icon="📊",
                    labels=[_trunc(d[0], 16) for d in data],
                    values=[d[1] for d in data],
                ))

        # ── 2. Monthly Revenue Trend (line) ───────────────────────────────
        if e.date_col and e.revenue_col:
            trend = e.monthly_trend(e.date_col, e.revenue_col, last_n=18)
            if len(trend) > 1:
                charts.append(self._line_chart(
                    chart_id="monthly_trend",
                    title=f"{e.revenue_col.replace('_',' ')} Monthly Trend",
                    sub="Revenue over time",
                    icon="📈",
                    wide=True,
                    labels=[t[0] for t in trend],
                    values=[t[1] for t in trend],
                ))

        # ── 3. Category Distribution (doughnut) ───────────────────────────
        if e.category_col:
            dist = e.value_counts_top(e.category_col, top_n=8)
            if len(dist) > 1:
                charts.append(self._doughnut_chart(
                    chart_id="category_distribution",
                    title=f"{e.category_col.replace('_',' ')} Distribution",
                    sub="Record count by category",
                    icon="🥧",
                    labels=[_trunc(d[0], 16) for d in dist],
                    values=[d[1] for d in dist],
                ))

        # ── 4. Top Categories Horizontal Bar ──────────────────────────────
        if e.category_col and e.revenue_col:
            top = e.group_sum(e.category_col, e.revenue_col, top_n=8)
            if top:
                charts.append(self._bar_chart(
                    chart_id="top_categories_horiz",
                    title="Top Categories Ranking",
                    sub="Horizontal revenue comparison",
                    icon="🏆",
                    labels=[_trunc(d[0], 20) for d in top],
                    values=[d[1] for d in top],
                    horizontal=True,
                    color=PAL[1],
                ))

        # ── 5. Revenue by Region (bar) ────────────────────────────────────
        if e.region_col and e.revenue_col:
            reg_data = e.group_sum(e.region_col, e.revenue_col, top_n=10)
            if len(reg_data) > 1:
                charts.append(self._bar_chart(
                    chart_id="revenue_by_region",
                    title="Revenue by Region",
                    sub="Regional performance comparison",
                    icon="🌍",
                    labels=[d[0] for d in reg_data],
                    values=[d[1] for d in reg_data],
                ))

        # ── 6. Revenue vs Profit grouped bar ──────────────────────────────
        if e.category_col and e.revenue_col and e.profit_col:
            cat_rev = e.group_sum(e.category_col, e.revenue_col, top_n=6)
            prf_map = dict(e.group_sum(e.category_col, e.profit_col, top_n=20))
            labels  = [_trunc(d[0], 14) for d in cat_rev]
            charts.append({
                "id":    "revenue_vs_profit",
                "type":  "bar",
                "title": "Revenue vs Profit by Category",
                "sub":   "Side-by-side comparison",
                "ico":   "⚖",
                "wide":  False,
                "labels": labels,
                "datasets": [
                    self._dataset_cfg(
                        label=e.revenue_col.replace("_", " "),
                        data=[d[1] for d in cat_rev],
                        color=PAL[0],
                        alpha="99",
                    ),
                    self._dataset_cfg(
                        label=e.profit_col.replace("_", " "),
                        data=[prf_map.get(d[0], 0) for d in cat_rev],
                        color=PAL[1],
                        alpha="99",
                    ),
                ],
            })

        # ── 7. Scatter — first two numeric columns ─────────────────────────
        num_cols = e.numeric_columns
        if len(num_cols) >= 2:
            c1, c2 = num_cols[0], num_cols[1]
            df2    = e.df[[c1, c2]].apply(pd.to_numeric, errors="coerce").dropna()
            # Sample for performance
            if len(df2) > 600:
                df2 = df2.sample(600, random_state=42)
            pts = [{"x": round(float(r[c1]), 4), "y": round(float(r[c2]), 4)}
                   for _, r in df2.iterrows()]
            if len(pts) > 10:
                charts.append({
                    "id":    "scatter_correlation",
                    "type":  "scatter",
                    "title": f"{c1.replace('_',' ')} vs {c2.replace('_',' ')}",
                    "sub":   "Correlation analysis",
                    "ico":   "⊕",
                    "wide":  False,
                    "xL":    c1,
                    "yL":    c2,
                    "labels": [],
                    "datasets": [{
                        "label":           "Data Points",
                        "data":            pts,
                        "backgroundColor": PAL[5] + "88",
                        "pointRadius":     4,
                        "pointHoverRadius": 6,
                    }],
                })

        log.info(f"Built {len(charts)} charts for '{e.filename}'")
        return charts

    # ══════════════════════════════════════════════════════════════════════
    # CHART BUILDERS (Chart.js)
    # ══════════════════════════════════════════════════════════════════════
    def _bar_chart(
        self,
        chart_id:   str,
        title:      str,
        sub:        str,
        icon:       str,
        labels:     list,
        values:     list,
        horizontal: bool = False,
        color:      Optional[str] = None,
    ) -> dict:
        colors = [c + "BB" for c in PAL[:len(values)]] if not color else [color + "99"] * len(values)
        return {
            "id":     chart_id,
            "type":   "bar",
            "title":  title,
            "sub":    sub,
            "ico":    icon,
            "wide":   False,
            "horiz":  horizontal,
            "labels": labels,
            "datasets": [self._dataset_cfg(
                label=title,
                data=[round(v, 2) for v in values],
                color=color or PAL[0],
                alpha="BB" if not color else "99",
                multi_color=(color is None),
                all_colors=PAL_ALPHA[:len(values)],
            )],
        }

    def _line_chart(
        self,
        chart_id: str,
        title:    str,
        sub:      str,
        icon:     str,
        labels:   list,
        values:   list,
        wide:     bool = True,
    ) -> dict:
        return {
            "id":     chart_id,
            "type":   "line",
            "title":  title,
            "sub":    sub,
            "ico":    icon,
            "wide":   wide,
            "labels": labels,
            "datasets": [{
                "label":            title,
                "data":             [round(v, 2) for v in values],
                "borderColor":      PAL[0],
                "backgroundColor":  PAL[0] + "28",
                "fill":             True,
                "tension":          0.42,
                "pointRadius":      4,
                "pointHoverRadius": 7,
                "borderWidth":      2.5,
                "pointBackgroundColor": PAL[0],
            }],
        }

    def _doughnut_chart(
        self,
        chart_id: str,
        title:    str,
        sub:      str,
        icon:     str,
        labels:   list,
        values:   list,
    ) -> dict:
        return {
            "id":     chart_id,
            "type":   "doughnut",
            "title":  title,
            "sub":    sub,
            "ico":    icon,
            "wide":   False,
            "labels": labels,
            "datasets": [{
                "data":            [int(v) for v in values],
                "backgroundColor": PAL_ALPHA[:len(values)],
                "borderColor":     "rgba(4,6,16,.8)",
                "borderWidth":     2,
                "hoverOffset":     10,
            }],
        }

    @staticmethod
    def _dataset_cfg(
        label:       str,
        data:        list,
        color:       str,
        alpha:       str = "BB",
        multi_color: bool = False,
        all_colors:  Optional[list] = None,
    ) -> dict:
        bg = all_colors if multi_color and all_colors else (color + alpha)
        return {
            "label":           label,
            "data":            data,
            "backgroundColor": bg,
            "borderColor":     color + "CC",
            "borderWidth":     0 if multi_color else 1.5,
            "borderRadius":    6,
            "borderSkipped":   False,
        }

    # ══════════════════════════════════════════════════════════════════════
    # PLOTLY CHARTS (server-side, export-quality)
    # ══════════════════════════════════════════════════════════════════════
    def build_plotly_charts(self) -> list[dict]:
        """
        Returns Plotly figure dicts (JSON-serialisable).
        Requires `plotly` to be installed.
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            log.warning("plotly not installed — skipping Plotly charts")
            return []

        charts = []
        e      = self.engine
        layout = dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans, sans-serif", color="#8892A8"),
            margin=dict(l=40, r=20, t=50, b=40),
        )

        # Revenue by Category
        if e.category_col and e.revenue_col:
            data  = e.group_sum(e.category_col, e.revenue_col, top_n=8)
            fig   = go.Figure(
                go.Bar(
                    x=[d[0] for d in data],
                    y=[d[1] for d in data],
                    marker_color=PAL[:len(data)],
                    text=[_fmt(d[1], "$") for d in data],
                    textposition="outside",
                )
            )
            fig.update_layout(title=f"Revenue by {e.category_col}", **layout)
            charts.append({"id": "plotly_rev_cat", "figure": fig.to_dict()})

        # Monthly Trend
        if e.date_col and e.revenue_col:
            trend = e.monthly_trend(e.date_col, e.revenue_col)
            if trend:
                fig = go.Figure(
                    go.Scatter(
                        x=[t[0] for t in trend],
                        y=[t[1] for t in trend],
                        mode="lines+markers",
                        fill="tozeroy",
                        line=dict(color=PAL[0], width=2.5),
                        fillcolor=PAL[0] + "28",
                    )
                )
                fig.update_layout(title="Monthly Revenue Trend", **layout)
                charts.append({"id": "plotly_trend", "figure": fig.to_dict()})

        return charts
