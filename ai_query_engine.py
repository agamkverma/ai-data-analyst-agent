"""
ai_query_engine.py — DataLensAI v2.0
Natural Language Query Engine

Responsibilities:
  - Intent detection via regex + keyword matching
  - Rule-based structured answers from pandas aggregations
  - AI-powered free-form answers via OpenAI or Gemini
  - Chart config suggestions for relevant query types
  - Graceful fallback: AI → rule-based → generic summary

Supported Intents:
  region_analysis      — "Which region has highest revenue?"
  category_analysis    — "Top 5 categories"
  trend_analysis       — "Monthly revenue trend"
  profit_analysis      — "What is the profit margin?"
  quality_check        — "Any missing values?"
  average_stats        — "Average sales per product"
  top_n_query          — "Top 10 products by revenue"
  correlation_query    — "Correlation between price and revenue"
  summary_query        — catch-all dataset summary

Author: Agam Kumar Verma
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import httpx
import numpy as np
import pandas as pd

from data_engine import DataEngine
from dataset_profiler import DatasetProfiler, _fmt, _trunc

log = logging.getLogger(__name__)


# ── Intent patterns ────────────────────────────────────────────────────────────
_INTENTS: list[tuple[str, re.Pattern]] = [
    ("region_analysis",   re.compile(r"region|geography|area|territory|location|zone|country|state|city", re.I)),
    ("category_analysis", re.compile(r"categor|product|type|brand|segment|class|group|department|dept",   re.I)),
    ("trend_analysis",    re.compile(r"trend|month|time|growth|over time|period|quarter|week|year",        re.I)),
    ("profit_analysis",   re.compile(r"profit|margin|net|earning|gain|ebit|ebitda|cost",                   re.I)),
    ("quality_check",     re.compile(r"missing|null|empty|clean|duplicate|quality|error",                  re.I)),
    ("average_stats",     re.compile(r"average|avg|mean|typical|median",                                   re.I)),
    ("top_n_query",       re.compile(r"top\s*\d+|best\s*\d+|highest|lowest|bottom|worst",                 re.I)),
    ("correlation_query", re.compile(r"correlat|relationship|vs\.|versus|compare|between",                 re.I)),
    ("revenue_summary",   re.compile(r"revenue|sales|total|sum|income|amount",                             re.I)),
]


class AIQueryEngine:
    """
    Processes natural language queries about the loaded dataset.

    Parameters
    ----------
    engine   : DataEngine
    profiler : DatasetProfiler
    """

    def __init__(self, engine: DataEngine, profiler: DatasetProfiler) -> None:
        self.engine   = engine
        self.profiler = profiler

    # ══════════════════════════════════════════════════════════════════════
    # INTENT DETECTION
    # ══════════════════════════════════════════════════════════════════════
    def detect_intent(self, query: str) -> str:
        """Return the most likely intent label for the query."""
        for intent, pattern in _INTENTS:
            if pattern.search(query):
                return intent
        return "summary_query"

    def extract_n(self, query: str, default: int = 5) -> int:
        """Extract a number from query like 'top 10' → 10."""
        match = re.search(r"\b(\d{1,2})\b", query)
        return int(match.group(1)) if match else default

    # ══════════════════════════════════════════════════════════════════════
    # RULE-BASED ANSWER
    # ══════════════════════════════════════════════════════════════════════
    def rule_answer(self, query: str) -> dict:
        """
        Generate a structured answer from pandas aggregations.
        Always succeeds — never raises.
        """
        intent = self.detect_intent(query)
        handler = getattr(self, f"_answer_{intent}", self._answer_summary_query)
        try:
            result = handler(query)
        except Exception as e:
            log.warning(f"Intent handler error ({intent}): {e}")
            result = self._answer_summary_query(query)
        result["intent"] = intent
        result["source"] = "Rule-Based Engine"
        return result

    # ── REGION ────────────────────────────────────────────────────────────
    def _answer_region_analysis(self, query: str) -> dict:
        e       = self.engine
        reg_col = e.region_col
        rev_col = e.revenue_col

        if not reg_col:
            return {"answer": "⚠️ No region column detected in this dataset.", "data": [], "chart_config": None}

        n       = self.extract_n(query, default=len(e.df[reg_col].unique()))
        data    = e.group_sum(reg_col, rev_col, top_n=n) if rev_col else e.value_counts_top(reg_col, n)
        rev_sum = self.profiler.get_numeric_series_stats(rev_col)["sum"] if rev_col else 0

        rows = [
            f"<strong>{i+1}. {d[0]}</strong> — {_fmt(d[1], '$') if rev_col else d[1]:,} "
            f"({'%.1f' % (d[1]/rev_sum*100)}%)" if rev_sum else f"<strong>{i+1}. {d[0]}</strong> — {d[1]:,} records"
            for i, d in enumerate(data)
        ]
        top_reg = data[0][0] if data else "—"
        answer  = f"🌍 <strong>Revenue by Region:</strong><br><br>" + "<br>".join(rows)
        if data:
            answer += f"<br><br>Top performer: <strong>{top_reg}</strong>"

        return {
            "answer":       answer,
            "data":         data,
            "chart_config": self._suggest_chart("bar", reg_col, rev_col or reg_col, data, horizontal=False),
        }

    # ── CATEGORY ─────────────────────────────────────────────────────────
    def _answer_category_analysis(self, query: str) -> dict:
        e       = self.engine
        cat_col = e.category_col
        rev_col = e.revenue_col

        if not cat_col:
            return {"answer": "⚠️ No category column detected in this dataset.", "data": [], "chart_config": None}

        n    = self.extract_n(query, default=5)
        data = e.group_sum(cat_col, rev_col, top_n=n) if rev_col else e.value_counts_top(cat_col, n)
        rev_sum = self.profiler.get_numeric_series_stats(rev_col)["sum"] if rev_col else 0

        rows = []
        for i, d in enumerate(data):
            pct = f" ({d[1]/rev_sum*100:.1f}%)" if rev_sum else ""
            val = _fmt(d[1], "$") if rev_col else f"{d[1]:,} records"
            rows.append(f"<strong>{i+1}. {d[0]}</strong> — {val}{pct}")

        answer = f"🏷 <strong>Top {n} {cat_col.replace('_',' ')}:</strong><br><br>" + "<br>".join(rows)
        return {
            "answer":       answer,
            "data":         data,
            "chart_config": self._suggest_chart("bar", cat_col, rev_col or cat_col, data, horizontal=True),
        }

    # ── TOP-N ─────────────────────────────────────────────────────────────
    def _answer_top_n_query(self, query: str) -> dict:
        e       = self.engine
        n       = self.extract_n(query, default=5)
        rev_col = e.revenue_col
        cat_col = e.category_col

        # Determine ascending/descending
        ascending = bool(re.search(r"bottom|worst|lowest|minimum", query, re.I))

        if cat_col and rev_col:
            data = e.group_sum(cat_col, rev_col, top_n=n, ascending=ascending)
            label = "Bottom" if ascending else "Top"
            rows  = [f"<strong>{i+1}. {d[0]}</strong>: {_fmt(d[1], '$')}" for i, d in enumerate(data)]
            answer = f"🏆 <strong>{label} {n} {cat_col.replace('_',' ')} by {rev_col.replace('_',' ')}:</strong><br><br>" + "<br>".join(rows)
            return {
                "answer":       answer,
                "data":         data,
                "chart_config": self._suggest_chart("bar", cat_col, rev_col, data, horizontal=True),
            }

        return self._answer_summary_query(query)

    # ── TREND ─────────────────────────────────────────────────────────────
    def _answer_trend_analysis(self, query: str) -> dict:
        e       = self.engine
        date_col = e.date_col
        rev_col  = e.revenue_col

        if not date_col:
            return {"answer": "⚠️ No date/time column detected in this dataset.", "data": [], "chart_config": None}

        trend = e.monthly_trend(date_col, rev_col or "", last_n=12)
        if not trend:
            return {"answer": "⚠️ Could not compute monthly trends. Ensure the date column is properly formatted.", "data": [], "chart_config": None}

        # Direction: last period vs first period
        direction = "↑ Growing" if len(trend) > 1 and trend[-1][1] > trend[0][1] else "↓ Declining"
        rows      = [f"<strong>{t[0]}</strong>: {_fmt(t[1], '$')}" for t in trend[-6:]]
        answer    = f"📅 <strong>Monthly Trend (last 6 periods) — {direction}:</strong><br><br>" + "<br>".join(rows)

        return {
            "answer":       answer,
            "data":         trend,
            "chart_config": self._suggest_chart("line", date_col, rev_col or "", trend),
        }

    # ── PROFIT ────────────────────────────────────────────────────────────
    def _answer_profit_analysis(self, query: str) -> dict:
        e       = self.engine
        prf_col = e.profit_col
        rev_col = e.revenue_col

        if not prf_col and not rev_col:
            return {"answer": "⚠️ No profit or revenue columns detected in this dataset.", "data": [], "chart_config": None}

        rev_stats = self.profiler.get_numeric_series_stats(rev_col) if rev_col else {}
        prf_stats = self.profiler.get_numeric_series_stats(prf_col) if prf_col else {}

        rev_sum = rev_stats.get("sum", 0)
        prf_sum = prf_stats.get("sum", 0)
        margin  = round(prf_sum / rev_sum * 100, 1) if rev_sum and prf_sum else None

        lines = []
        if rev_col:
            lines.append(f"Total Revenue: <strong>{_fmt(rev_sum, '$')}</strong>")
        if prf_col:
            lines.append(f"Total Profit: <strong>{_fmt(prf_sum, '$')}</strong>")
        if margin is not None:
            lines.append(f"Profit Margin: <strong>{margin}%</strong>")
            benchmark = "✅ Above" if margin >= 20 else "⚠️ Below"
            lines.append(f"Benchmark: {benchmark} 20% industry average")

        answer = "💵 <strong>Profit & Margin Analysis:</strong><br><br>" + "<br>".join(lines)
        return {"answer": answer, "data": [], "chart_config": None}

    # ── QUALITY ───────────────────────────────────────────────────────────
    def _answer_quality_check(self, query: str) -> dict:
        qs       = self.profiler.quality_score()
        e        = self.engine
        col_nulls = qs.get("col_nulls", {})
        null_cols = [(c, n) for c, n in col_nulls.items() if n > 0]

        if not null_cols:
            answer = (
                f"✅ <strong>Dataset is fully clean!</strong><br><br>"
                f"Zero missing values across all {len(e.columns)} columns "
                f"and {e.row_count:,} rows.<br>"
                f"Duplicates: <strong>{qs['duplicates']}</strong><br>"
                f"Quality Score: <strong>{qs['score']}% ({qs['grade']})</strong>"
            )
        else:
            rows = [f"• <strong>{c}</strong>: {n} nulls ({n/e.row_count*100:.1f}%)"
                    for c, n in sorted(null_cols, key=lambda x: -x[1])]
            answer = (
                f"⚠️ <strong>{qs['nulls']} missing values</strong> in {len(null_cols)} column(s):<br><br>"
                + "<br>".join(rows)
                + f"<br><br>Duplicates: <strong>{qs['duplicates']}</strong><br>"
                + f"Quality Score: <strong>{qs['score']}% ({qs['grade']})</strong>"
            )
        return {"answer": answer, "data": [], "chart_config": None}

    # ── AVERAGES ──────────────────────────────────────────────────────────
    def _answer_average_stats(self, query: str) -> dict:
        e        = self.engine
        num_cols = e.numeric_columns[:5]

        if not num_cols:
            return {"answer": "⚠️ No numeric columns detected in this dataset.", "data": [], "chart_config": None}

        rows = []
        for col in num_cols:
            s = self.profiler.get_numeric_series_stats(col)
            rows.append(
                f"<strong>{col.replace('_',' ')}</strong>: "
                f"avg={_fmt(s.get('mean',0),'$')}, "
                f"min={_fmt(s.get('min',0))}, "
                f"max={_fmt(s.get('max',0))}, "
                f"median={_fmt(s.get('median',0))}"
            )

        answer = "📊 <strong>Descriptive Statistics:</strong><br><br>" + "<br>".join(rows)
        return {"answer": answer, "data": [], "chart_config": None}

    # ── CORRELATION ───────────────────────────────────────────────────────
    def _answer_correlation_query(self, query: str) -> dict:
        e        = self.engine
        num_cols = e.numeric_columns
        if len(num_cols) < 2:
            return {"answer": "⚠️ At least 2 numeric columns are needed for correlation analysis.", "data": [], "chart_config": None}

        corr = (
            e.df[num_cols[:6]]
            .apply(pd.to_numeric, errors="coerce")
            .corr()
            .round(3)
        )
        # Find strongest pair
        corr_flat = corr.unstack().drop_duplicates()
        corr_flat = corr_flat[corr_flat.index.get_level_values(0) != corr_flat.index.get_level_values(1)]
        top_corr  = corr_flat.abs().nlargest(3)

        rows = []
        for (c1, c2), val in corr_flat[top_corr.index].items():
            label = "strong ↑ positive" if val > 0.7 else "moderate ↑" if val > 0.4 else "strong ↓ negative" if val < -0.7 else "weak"
            rows.append(f"<strong>{c1.replace('_',' ')} ↔ {c2.replace('_',' ')}</strong>: r={val:.3f} ({label})")

        answer = "🔗 <strong>Correlation Analysis (top pairs):</strong><br><br>" + "<br>".join(rows)
        return {"answer": answer, "data": corr.to_dict(), "chart_config": None}

    # ── REVENUE SUMMARY ───────────────────────────────────────────────────
    def _answer_revenue_summary(self, query: str) -> dict:
        e       = self.engine
        rev_col = e.revenue_col
        if not rev_col:
            return self._answer_summary_query(query)
        stats = self.profiler.get_numeric_series_stats(rev_col)
        answer = (
            f"💰 <strong>{rev_col.replace('_',' ')} Summary:</strong><br><br>"
            f"Total: <strong>{_fmt(stats.get('sum',0), '$')}</strong><br>"
            f"Average: <strong>{_fmt(stats.get('mean',0), '$')}</strong><br>"
            f"Median: <strong>{_fmt(stats.get('median',0), '$')}</strong><br>"
            f"Min: <strong>{_fmt(stats.get('min',0), '$')}</strong><br>"
            f"Max: <strong>{_fmt(stats.get('max',0), '$')}</strong><br>"
            f"Std Dev: <strong>{_fmt(stats.get('std',0), '$')}</strong>"
        )
        return {"answer": answer, "data": [], "chart_config": None}

    # ── CATCH-ALL SUMMARY ─────────────────────────────────────────────────
    def _answer_summary_query(self, query: str) -> dict:
        e   = self.engine
        qs  = self.profiler.quality_score()
        num = e.numeric_columns[:3]
        lines = [
            f"Records: <strong>{e.row_count:,} rows × {len(e.columns)} columns</strong>",
            f"Quality: <strong>{qs['score']}%</strong> ({qs['nulls']} nulls, {qs['duplicates']} duplicates)",
        ]
        for col in num:
            s = self.profiler.get_numeric_series_stats(col)
            lines.append(f"{col.replace('_',' ')}: avg <strong>{_fmt(s.get('mean',0),'$')}</strong>, max <strong>{_fmt(s.get('max',0),'$')}</strong>")

        answer = "📊 <strong>Dataset Summary:</strong><br><br>" + "<br>".join(lines)
        return {"answer": answer, "data": [], "chart_config": None}

    # ══════════════════════════════════════════════════════════════════════
    # AI-POWERED ANSWER
    # ══════════════════════════════════════════════════════════════════════
    async def ai_answer(
        self,
        query:    str,
        api_key:  str,
        provider: str = "gemini",
    ) -> dict:
        """
        Use AI to answer a query in natural language.
        Returns the same structure as rule_answer for consistency.
        """
        from insight_generator import InsightGenerator
        ig      = InsightGenerator(self.engine, self.profiler)
        summary = ig._build_summary()

        prompt = (
            f"You are a concise data analyst. Answer the following question about the dataset "
            f"in 2-4 sentences with specific numbers. Use <strong> tags around key values. "
            f"Be direct — never refuse.\n\n"
            f"Dataset context:\n{summary}\n\n"
            f"Question: {query}"
        )

        try:
            if provider == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
                payload = {
                    "contents":         [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": 350, "temperature": 0.4},
                }
                async with httpx.AsyncClient(timeout=20.0) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                url = "https://api.openai.com/v1/chat/completions"
                payload = {
                    "model":       "gpt-4o-mini",
                    "messages":    [{"role": "user", "content": prompt}],
                    "max_tokens":  350,
                    "temperature": 0.4,
                }
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                async with httpx.AsyncClient(timeout=20.0) as client:
                    resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"]

            return {
                "answer":       text,
                "data":         [],
                "chart_config": None,
                "intent":       self.detect_intent(query),
                "source":       f"{'Google Gemini' if provider == 'gemini' else 'OpenAI GPT-4o Mini'}",
            }
        except Exception as e:
            log.warning(f"AI query fallback triggered: {e}")
            return self.rule_answer(query)

    # ══════════════════════════════════════════════════════════════════════
    # CHART SUGGESTION
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _suggest_chart(
        chart_type: str,
        x_col:      str,
        y_col:      str,
        data:       list,
        horizontal: bool = False,
    ) -> Optional[dict]:
        """Return a minimal Chart.js config suggestion for the query result."""
        if not data:
            return None
        labels = [str(d[0]) for d in data]
        values = [float(d[1]) for d in data]
        from chart_engine import PAL, PAL_ALPHA
        if chart_type == "bar":
            return {
                "type":   "bar",
                "horiz":  horizontal,
                "labels": labels,
                "datasets": [{
                    "label":           y_col.replace("_", " "),
                    "data":            values,
                    "backgroundColor": PAL_ALPHA[:len(values)],
                    "borderRadius":    6,
                    "borderSkipped":   False,
                }],
            }
        if chart_type == "line":
            return {
                "type":   "line",
                "labels": labels,
                "datasets": [{
                    "label":           y_col.replace("_", " "),
                    "data":            values,
                    "borderColor":     PAL[0],
                    "backgroundColor": PAL[0] + "28",
                    "fill":            True,
                    "tension":         0.4,
                }],
            }
        return None
