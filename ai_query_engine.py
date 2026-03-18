"""
query_engine.py
---------------
Natural Language Query Engine.
Parses user questions and dispatches to the correct analysis + chart.
Supports rule-based parsing + optional LLM (OpenAI / Gemini).
"""

import re
import json
import pandas as pd
from typing import Optional


# ── Intent Definitions ──────────────────────────────────────────────────────────
INTENTS = {
    "revenue_by_region":   ["revenue by region","region revenue","sales by region","regional revenue","region performance","region sales","which region"],
    "revenue_by_category": ["revenue by category","category revenue","sales by category","category performance","by category","category breakdown","category sales"],
    "top_products":        ["top product","best product","highest revenue product","top selling","most revenue","bestseller","top 5 product","top 10 product"],
    "monthly_trend":       ["monthly","over time","trend","by month","time series","monthly revenue","monthly sales","sales trend","revenue trend","month"],
    "quarterly_trend":     ["quarterly","by quarter","q1","q2","q3","q4","quarter"],
    "profit_margin":       ["margin","profitability","profit margin","most profitable","efficient","profit by"],
    "category_pie":        ["pie","distribution","breakdown","share","proportion","percentage","how much each"],
    "region_category_heat":["heatmap","heat map","matrix","cross","region and category","category and region"],
    "top_customers":       ["top customer","best customer","biggest customer","customer revenue","customer value"],
    "bottom_performers":   ["worst","bottom","underperform","lowest","weakest","poor"],
    "sales_rep":           ["sales rep","rep performance","salesperson","agent","who sold"],
    "forecast":            ["forecast","predict","future","next month","projection","will revenue"],
    "scatter_rev_profit":  ["scatter","correlation","revenue vs profit","profit vs revenue","relationship"],
    "kpi_summary":         ["kpi","summary","overview","total","overall","dashboard","key metrics"],
    "cost_analysis":       ["cost","expense","cogs","cost analysis","spending"],
    "growth":              ["growth","increase","change","mom","year over year","yoy"],
}


def detect_intent(query: str) -> str:
    """Return the best-matching intent key for a natural language query."""
    q = query.lower().strip()
    scores = {}
    for intent, phrases in INTENTS.items():
        score = sum(1 for ph in phrases if ph in q)
        if score > 0:
            scores[intent] = score
    if scores:
        return max(scores, key=scores.get)
    # Keyword fallback
    if any(w in q for w in ["region","area","territory"]): return "revenue_by_region"
    if any(w in q for w in ["category","segment","type"]): return "revenue_by_category"
    if any(w in q for w in ["product","item","sku"]):       return "top_products"
    if any(w in q for w in ["month","time","trend"]):       return "monthly_trend"
    if any(w in q for w in ["margin","profit"]):            return "profit_margin"
    return "kpi_summary"


def extract_top_n(query: str) -> int:
    """Extract a number from the query (e.g. 'top 10 products' → 10)."""
    word_map = {"one":1,"two":2,"three":3,"four":4,"five":5,
                "six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    for w, n in word_map.items():
        if w in query.lower():
            return n
    m = re.search(r"\b(\d+)\b", query)
    return int(m.group(1)) if m else 5


def parse_query(query: str, col_map: dict, df_columns: list) -> dict:
    """
    Parse a natural language query into a structured execution plan.
    Returns a dict with: intent, group_by, metric, chart_type, top_n, title, etc.
    """
    intent  = detect_intent(query)
    top_n   = extract_top_n(query)
    q       = query.lower()

    # Determine metric
    metric = None
    if any(w in q for w in ["profit","margin","earnings"]):
        metric = col_map.get("profit")
    elif any(w in q for w in ["cost","expense","cogs"]):
        metric = col_map.get("cost")
    elif any(w in q for w in ["units","quantity","qty","volume"]):
        metric = col_map.get("quantity")
    if not metric:
        metric = col_map.get("revenue")

    plan = {
        "intent":    intent,
        "metric":    metric,
        "group_by":  None,
        "chart":     "bar",
        "top_n":     top_n,
        "ascending": False,
        "title":     "",
        "subtitle":  "",
    }

    if intent == "revenue_by_region":
        plan.update({"group_by": col_map.get("region"), "chart": "bar",
                     "title": f"{_metric_name(metric)} by Region",
                     "subtitle": "Grouped bar — sorted by revenue descending"})

    elif intent == "revenue_by_category":
        plan.update({"group_by": col_map.get("category"), "chart": "bar",
                     "title": f"{_metric_name(metric)} by Category",
                     "subtitle": "Horizontal bar — category revenue comparison"})

    elif intent == "top_products":
        plan.update({"group_by": col_map.get("product"), "chart": "bar",
                     "title": f"Top {top_n} Products by {_metric_name(metric)}",
                     "subtitle": f"Ranked top {top_n} — highest revenue first"})

    elif intent == "monthly_trend":
        plan.update({"group_by": col_map.get("date"), "chart": "line",
                     "title": f"Monthly {_metric_name(metric)} Trend",
                     "subtitle": "Time series — monthly aggregation"})

    elif intent == "quarterly_trend":
        plan.update({"group_by": col_map.get("date"), "chart": "area",
                     "title": f"Quarterly {_metric_name(metric)} Trend",
                     "subtitle": "Time series — quarterly aggregation"})

    elif intent == "profit_margin":
        plan.update({"group_by": col_map.get("category") or col_map.get("region"),
                     "chart": "bar",
                     "title": "Profit Margin by Category",
                     "subtitle": "Profit % per revenue — higher is better"})

    elif intent == "category_pie":
        plan.update({"group_by": col_map.get("category"), "chart": "pie",
                     "title": f"{_metric_name(metric)} Distribution by Category",
                     "subtitle": "Donut chart — percentage share"})

    elif intent == "region_category_heat":
        plan.update({"chart": "heatmap",
                     "title": "Revenue Heatmap: Category × Region",
                     "subtitle": "Cross-tabulation of category and region revenue"})

    elif intent == "top_customers":
        plan.update({"group_by": col_map.get("customer"), "chart": "bar",
                     "title": f"Top {top_n} Customers by Revenue",
                     "subtitle": "Customer revenue leaderboard"})

    elif intent == "bottom_performers":
        plan.update({"group_by": col_map.get("region") or col_map.get("product"),
                     "ascending": True, "chart": "bar",
                     "title": f"Bottom {top_n} Performers by {_metric_name(metric)}",
                     "subtitle": "Lowest revenue — identify weak spots"})

    elif intent == "sales_rep":
        plan.update({"group_by": _find_rep_col(df_columns), "chart": "bar",
                     "title": "Sales Rep Performance",
                     "subtitle": "Revenue and margin by sales representative"})

    elif intent == "forecast":
        plan.update({"chart": "forecast",
                     "title": "Revenue Forecast — Next 3 Months",
                     "subtitle": "Polynomial regression forecast with confidence interval"})

    elif intent == "scatter_rev_profit":
        plan.update({"chart": "scatter",
                     "title": "Revenue vs Profit Scatter",
                     "subtitle": "Correlation analysis — colored by category"})

    elif intent == "kpi_summary":
        plan.update({"chart": "kpi",
                     "title": "Business KPI Summary",
                     "subtitle": "Key performance indicators overview"})

    elif intent == "cost_analysis":
        plan.update({"group_by": col_map.get("category"), "metric": col_map.get("cost"),
                     "chart": "bar",
                     "title": "Cost Analysis by Category",
                     "subtitle": "Total cost breakdown by category"})

    return plan


def _metric_name(col: Optional[str]) -> str:
    if not col: return "Revenue"
    return col.replace("_"," ").title()


def _find_rep_col(columns: list) -> Optional[str]:
    kws = ["rep","salesperson","agent","seller","staff","employee","sales rep"]
    for kw in kws:
        for col in columns:
            if kw.lower() in col.lower():
                return col
    return None


# ── LLM Query Parser (optional) ────────────────────────────────────────────────

def llm_parse_query(query: str, context: str, api_key: str, provider: str) -> dict:
    """
    Use LLM to parse query into structured plan.
    Falls back to rule-based if LLM fails.
    """
    system_prompt = """You are a BI analyst. Parse the user's question into a JSON analysis plan.
Respond ONLY with valid JSON, no markdown, no explanation.
Schema: {
  "intent": "revenue_by_region|revenue_by_category|top_products|monthly_trend|profit_margin|category_pie|heatmap|forecast|scatter|kpi_summary",
  "group_by": "column name or null",
  "metric": "column name or null",
  "chart": "bar|line|pie|heatmap|scatter|forecast|area",
  "top_n": 5,
  "ascending": false,
  "title": "chart title"
}"""
    try:
        if provider == "openai":
            import openai
            client = openai.OpenAI(api_key=api_key)
            r = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Dataset context:\n{context}\n\nUser question: {query}"}
                ],
                temperature=0.1, max_tokens=300,
                response_format={"type": "json_object"}
            )
            return json.loads(r.choices[0].message.content)

        elif provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            m = genai.GenerativeModel("gemini-1.5-pro", system_instruction=system_prompt)
            r = m.generate_content(
                f"Dataset context:\n{context}\n\nUser question: {query}",
                generation_config=genai.GenerationConfig(
                    temperature=0.1, response_mime_type="application/json"
                )
            )
            raw = re.sub(r"```json|```", "", r.text).strip()
            return json.loads(raw)

    except Exception:
        pass

    return {}  # Signal caller to use rule-based
