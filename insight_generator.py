"""
insight_generator.py — DataLensAI v2.0
AI Insight Generation Engine

Responsibilities:
  - Rule-based insight generation from statistical heuristics
  - AI-powered insights via OpenAI GPT-4o-mini or Google Gemini
  - Structured output: summary, insights list, recommendations
  - Fallback chain: AI → rule-based → generic

Insight Types:
  positive  — strong performance, growth signals
  negative  — underperformance, declining metrics
  warning   — concentration risk, data quality issues
  neutral   — informational / contextual observations
  info      — dataset structure, column metadata

Author: Agam Kumar Verma
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import httpx
import numpy as np

from data_engine import DataEngine
from dataset_profiler import DatasetProfiler, _fmt, _trunc

log = logging.getLogger(__name__)

# ── AI prompt template ────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a senior Business Intelligence analyst. Your job is to analyze dataset statistics \
and generate actionable, specific, data-backed business insights. Always reference real numbers \
from the data. Be concise, direct, and strategic.
"""

_USER_PROMPT_TEMPLATE = """\
Analyze the following dataset summary and provide structured business insights.

{summary}

{custom_question}

Respond ONLY with valid JSON (no markdown fences, no extra text):
{{
  "summary": "3-5 sentences highlighting the most important findings. \
Use <strong> tags around key numbers, percentages, and entity names.",
  "insights": [
    {{
      "title": "Short, specific title",
      "description": "1-2 sentences with specific numbers from the data.",
      "type": "positive|negative|warning|neutral|info"
    }}
  ],
  "recommendations": [
    "Specific, actionable recommendation 1",
    "Specific, actionable recommendation 2",
    "Specific, actionable recommendation 3",
    "Specific, actionable recommendation 4",
    "Specific, actionable recommendation 5"
  ]
}}

Provide 7-9 insights covering revenue, profit, regions, categories, trends, data quality, \
and growth opportunities. All numbers must come from the dataset summary above.
"""


class InsightGenerator:
    """
    Generates AI or rule-based business insights from a DataEngine session.

    Parameters
    ----------
    engine   : DataEngine
    profiler : DatasetProfiler
    """

    def __init__(self, engine: DataEngine, profiler: DatasetProfiler) -> None:
        self.engine   = engine
        self.profiler = profiler

    # ══════════════════════════════════════════════════════════════════════
    # RULE-BASED INSIGHTS
    # ══════════════════════════════════════════════════════════════════════
    def rule_based_insights(self, custom_question: Optional[str] = None) -> dict:
        """
        Generate statistical heuristic-based insights.
        No API key required — runs entirely from pandas aggregations.
        """
        e       = self.engine
        df      = e.df
        rev_col = e.revenue_col
        prf_col = e.profit_col
        cat_col = e.category_col
        reg_col = e.region_col
        qs      = self.profiler.quality_score()

        # ── Aggregate data ────────────────────────────────────────────────
        rev_sum  = float(self.profiler.get_numeric_series_stats(rev_col).get("sum", 0)) if rev_col else 0
        prf_sum  = float(self.profiler.get_numeric_series_stats(prf_col).get("sum", 0)) if prf_col else 0
        margin   = round(prf_sum / rev_sum * 100, 1) if rev_sum and prf_sum else None

        region_data = e.group_sum(reg_col, rev_col, top_n=20) if reg_col and rev_col else []
        cat_data    = e.group_sum(cat_col, rev_col, top_n=20) if cat_col and rev_col else []

        top_reg  = region_data[0][0] if region_data else "—"
        bot_reg  = region_data[-1][0] if region_data else "—"
        top_cat  = cat_data[0][0] if cat_data else "—"

        top_reg_val = region_data[0][1] if region_data else 0
        bot_reg_val = region_data[-1][1] if region_data else 0
        top_cat_val = cat_data[0][1] if cat_data else 0

        top_reg_pct = round(top_reg_val / rev_sum * 100, 1) if rev_sum else 0
        top_cat_pct = round(top_cat_val / rev_sum * 100, 1) if rev_sum else 0
        bot_reg_pct = round(bot_reg_val / rev_sum * 100, 1) if rev_sum else 0

        top2_pct = (
            round((cat_data[0][1] + cat_data[1][1]) / rev_sum * 100, 1)
            if len(cat_data) >= 2 and rev_sum else 0
        )

        growth_gap = round((rev_sum / max(len(region_data), 1)) - bot_reg_val, 2) if region_data else 0

        # ── Summary ────────────────────────────────────────────────────────
        summary_parts = []
        if rev_sum:
            summary_parts.append(
                f"Total revenue stands at <strong>{_fmt(rev_sum, '$')}</strong>."
            )
        if top_reg != "—":
            summary_parts.append(
                f"<strong>{top_reg} region</strong> leads with <strong>{top_reg_pct}%</strong> of revenue."
            )
        if top_cat != "—":
            summary_parts.append(
                f"<strong>{top_cat}</strong> is the top-performing category at <strong>{top_cat_pct}%</strong>."
            )
        if margin is not None:
            benchmark = "above" if margin >= 20 else "below"
            summary_parts.append(
                f"Profit margin is <strong>{margin}%</strong> — {benchmark} the 20% industry benchmark."
            )
        summary_parts.append(
            f"Dataset quality score: <strong>{qs['score']}%</strong> ({qs['nulls']} nulls, {qs['duplicates']} duplicates)."
        )
        summary = " ".join(summary_parts)

        # ── Insights ───────────────────────────────────────────────────────
        insights = []

        if top_reg != "—":
            insights.append({
                "title":       f"Revenue Leader: {top_reg}",
                "description": f"{top_reg} generates {_fmt(top_reg_val, '$')} ({top_reg_pct}% of total). This region is the primary revenue driver and should receive continued investment.",
                "type":        "positive",
            })

        if top_cat != "—":
            second_cat = cat_data[1][0] if len(cat_data) > 1 else "—"
            second_pct = round(cat_data[1][1] / rev_sum * 100, 1) if len(cat_data) > 1 and rev_sum else 0
            insights.append({
                "title":       f"Top Category: {top_cat}",
                "description": f"{top_cat} accounts for {top_cat_pct}% of revenue ({_fmt(top_cat_val, '$')}). Second place: {second_cat} at {second_pct}%.",
                "type":        "positive",
            })

        if top2_pct > 0:
            risk = top2_pct > 65
            insights.append({
                "title":       "Revenue Concentration",
                "description": f"Top 2 categories account for {top2_pct}% of total revenue. " +
                               ("High concentration — diversification is recommended to reduce risk." if risk
                                else "Healthy distribution across the portfolio."),
                "type":        "warning" if risk else "neutral",
            })

        if bot_reg != "—" and bot_reg != top_reg:
            insights.append({
                "title":       f"Underperforming Region: {bot_reg}",
                "description": f"{bot_reg} contributes only {bot_reg_pct}% of revenue ({_fmt(bot_reg_val, '$')}). Performance gap vs top region: {_fmt(top_reg_val - bot_reg_val, '$')}.",
                "type":        "negative",
            })

        if margin is not None:
            insights.append({
                "title":       "Profit Margin Analysis",
                "description": f"{margin}% margin ({_fmt(prf_sum, '$')} profit on {_fmt(rev_sum, '$')} revenue). " +
                               ("Above 20% benchmark — efficient operations." if margin >= 20
                                else "Below 20% benchmark — review pricing and cost structure."),
                "type":        "positive" if margin >= 20 else "warning",
            })
        else:
            insights.append({
                "title":       "Profit Tracking",
                "description": "No profit column detected. Add a 'Profit' or 'Net Income' column for margin and profitability analysis.",
                "type":        "info",
            })

        insights.append({
            "title":       "Data Quality Assessment",
            "description": f"{e.row_count:,} records across {len(e.columns)} columns. {qs['nulls']} missing values, {qs['duplicates']} duplicates. Overall quality: {qs['score']}% ({qs['grade']}).",
            "type":        "info" if qs["score"] >= 80 else "warning",
        })

        if len(region_data) > 1:
            insights.append({
                "title":       "Regional Growth Opportunity",
                "description": f"If {bot_reg} reaches the portfolio average, estimated revenue uplift: {_fmt(max(0, growth_gap), '$')}. Focus marketing and distribution resources there.",
                "type":        "neutral",
            })

        if rev_col:
            rev_stats = self.profiler.get_numeric_series_stats(rev_col)
            skew      = rev_stats.get("skewness", 0)
            if abs(skew) > 1.0:
                insights.append({
                    "title":       "Revenue Distribution Skew",
                    "description": f"Revenue distribution is {'right' if skew > 0 else 'left'}-skewed (skewness={skew:.2f}). " +
                                   ("A few large transactions dominate — investigate Pareto effect." if skew > 1.0
                                    else "Most revenue clusters at higher values — a healthy sign."),
                    "type":        "neutral",
                })

        # ── Recommendations ────────────────────────────────────────────────
        recommendations = [
            f"Scale investment in {top_reg} region ({top_reg_pct}% revenue share) — expand marketing, distribution, and sales resources.",
        ]
        if bot_reg != "—" and bot_reg != top_reg:
            recommendations.append(
                f"Launch a targeted turnaround program in {bot_reg} to close the {_fmt(top_reg_val - bot_reg_val, '$')} performance gap."
            )
        recommendations.append(
            f"Grow {top_cat} category through cross-selling, bundling, and upsell campaigns to maintain market leadership."
        )
        if margin is not None and margin < 20:
            recommendations.append(
                f"Address the {margin}% profit margin by auditing supply chain costs and repricing low-margin products."
            )
        else:
            recommendations.append(
                "Introduce loyalty programs and subscription models to increase customer lifetime value and predictable recurring revenue."
            )
        if top2_pct > 65:
            recommendations.append(
                f"Reduce category concentration risk (top 2 = {top2_pct}%) by investing in campaigns for the bottom 3 categories."
            )
        else:
            recommendations.append(
                "Continue diversified category strategy — maintain balanced portfolio investment across all segments."
            )

        return {
            "summary":         summary,
            "insights":        insights[:9],
            "recommendations": recommendations[:5],
            "source":          "Rule-Based Intelligence Engine",
            "model":           None,
        }

    # ══════════════════════════════════════════════════════════════════════
    # AI-POWERED INSIGHTS
    # ══════════════════════════════════════════════════════════════════════
    async def ai_insights(
        self,
        api_key:         str,
        provider:        str = "gemini",
        custom_question: Optional[str] = None,
    ) -> dict:
        """
        Generate AI insights using OpenAI or Gemini.
        Falls back to rule-based on any error.

        Parameters
        ----------
        api_key         : str   API key for the chosen provider
        provider        : str   "gemini" | "openai"
        custom_question : str   Optional user-provided follow-up question
        """
        summary = self._build_summary()
        cq_text = f"\nUser question: {custom_question}" if custom_question else ""
        prompt  = _USER_PROMPT_TEMPLATE.format(summary=summary, custom_question=cq_text)

        raw = ""
        try:
            if provider == "gemini":
                raw = await self._call_gemini(api_key, prompt)
            else:
                raw = await self._call_openai(api_key, prompt)

            result = self._parse_ai_response(raw)
            result["source"] = (
                "Google Gemini 1.5 Flash" if provider == "gemini"
                else "OpenAI GPT-4o Mini"
            )
            result["model"] = (
                "gemini-1.5-flash" if provider == "gemini" else "gpt-4o-mini"
            )
            return result

        except json.JSONDecodeError:
            log.warning(f"AI returned invalid JSON. Raw: {raw[:300]}")
            raise ValueError("AI returned invalid JSON — switching to rule-based.")
        except Exception as e:
            log.warning(f"AI insight call failed: {e}")
            raise

    # ── Gemini API ─────────────────────────────────────────────────────────
    async def _call_gemini(self, api_key: str, prompt: str) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/gemini-1.5-flash:generateContent?key={api_key}"
        )
        payload = {
            "contents":       [{"parts": [{"text": _SYSTEM_PROMPT + "\n\n" + prompt}]}],
            "generationConfig": {"maxOutputTokens": 1800, "temperature": 0.35},
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RuntimeError(data["error"]["message"])
            return data["candidates"][0]["content"]["parts"][0]["text"]

    # ── OpenAI API ─────────────────────────────────────────────────────────
    async def _call_openai(self, api_key: str, prompt: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model":       "gpt-4o-mini",
            "messages":    [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens":  1800,
            "temperature": 0.35,
        }
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RuntimeError(data["error"]["message"])
            return data["choices"][0]["message"]["content"]

    # ── Parse AI JSON response ─────────────────────────────────────────────
    @staticmethod
    def _parse_ai_response(raw: str) -> dict:
        clean = raw.strip()
        # Strip markdown code fences if any
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)
        parsed = json.loads(clean)
        # Validate structure
        if not isinstance(parsed.get("insights"), list):
            raise ValueError("Missing 'insights' list in AI response.")
        return parsed

    # ── Build compact data summary for the prompt ──────────────────────────
    def _build_summary(self) -> str:
        e       = self.engine
        profile = self.profiler.full_profile()
        qs      = profile["quality"]
        lines   = [
            f'Dataset: "{e.filename}" | Rows: {e.row_count:,} | Columns: {len(e.columns)}',
            f'Detected fields: revenue={e.revenue_col}, profit={e.profit_col}, '
            f'date={e.date_col}, category={e.category_col}, region={e.region_col}',
            f'Quality: score={qs["score"]}%, nulls={qs["nulls"]}, duplicates={qs["duplicates"]}',
            '',
            'Column Statistics:',
        ]
        for col in profile["columns"][:14]:
            if col["type"] == "numeric":
                lines.append(
                    f'  {col["name"]}: sum={_fmt(col.get("sum",0))}, '
                    f'avg={_fmt(col.get("mean",0))}, min={_fmt(col.get("min",0))}, '
                    f'max={_fmt(col.get("max",0))}, nulls={col["null_count"]}'
                )
            elif col["type"] == "categorical":
                top3 = ", ".join(f'{v[0]}({v[1]})' for v in col.get("top_values", [])[:3])
                lines.append(
                    f'  {col["name"]}: {col.get("unique",0)} unique, top=[{top3}], '
                    f'nulls={col["null_count"]}'
                )
            else:
                lines.append(f'  {col["name"]}: {col["type"]}')

        lines.append('')
        lines.append(f'Sample rows: {json.dumps(e.sample(3), default=str)[:500]}')
        return "\n".join(lines)
