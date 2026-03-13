"""ai_query_engine.py — Translates natural language queries into structured analysis plans."""

import re
import json

DATE_KW     = ["date","time","month","year","day","period","quarter","week"]
REVENUE_KW  = ["revenue","sales","income","amount","price","total","turnover"]
PROFIT_KW   = ["profit","margin","earnings","net","gain"]
COST_KW     = ["cost","expense","cogs","spend"]
REGION_KW   = ["region","area","territory","zone","location","country","state","city","geo"]
CATEGORY_KW = ["category","segment","type","group","division","class","department"]
PRODUCT_KW  = ["product","item","sku","name","service","goods","brand"]
QUANTITY_KW = ["units","quantity","qty","count","volume","orders"]


def process_query(query: str, dataset_context: str,
                  api_key: str = "", provider: str = "openai") -> dict:
    if api_key:
        try:
            if provider == "openai":
                return _openai(query, dataset_context, api_key)
            elif provider == "gemini":
                return _gemini(query, dataset_context, api_key)
        except Exception as e:
            pass  # fall through to rule-based
    return _rule_based(query, dataset_context)


# ── LLM calls (optional) ─────────────────────────────────────────────────────

SYSTEM = """You are a BI analyst AI. Convert the user's question to a JSON analysis plan.
Respond ONLY with valid JSON, no markdown. Structure:
{"intent":"revenue_by_group|trend_over_time|top_n|bottom_n|distribution|margin_analysis|kpi_summary|forecast|anomaly|comparison",
 "group_by":"column or null","metric":"column or null","agg_func":"sum|mean|count",
 "sort_order":"desc|asc","top_n":null,"chart_type":"bar|line|area|pie|scatter|heatmap|table",
 "time_grouping":"month|quarter|year|null","title":"short title","insight_request":"what to highlight"}"""

def _openai(query, ctx, api_key):
    import openai
    client = openai.OpenAI(api_key=api_key)
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"system","content":SYSTEM},
                  {"role":"user","content":f"Question: {query}\nContext:\n{ctx}"}],
        temperature=0.1, max_tokens=400, response_format={"type":"json_object"})
    return json.loads(r.choices[0].message.content)

def _gemini(query, ctx, api_key):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel("gemini-1.5-pro", system_instruction=SYSTEM)
    r = m.generate_content(f"Question: {query}\nContext:\n{ctx}",
        generation_config=genai.GenerationConfig(temperature=0.1, response_mime_type="application/json"))
    raw = re.sub(r"```json|```","",r.text).strip()
    return json.loads(raw)


# ── Rule-based fallback ───────────────────────────────────────────────────────

def _rule_based(query: str, ctx: str) -> dict:
    q = query.lower()
    cols = _cols_from_ctx(ctx)

    # Detect group_by
    group_by = None
    for col in cols:
        if col.lower() in q:
            group_by = col; break
    if not group_by:
        if any(w in q for w in ["region","territory","area","geo"]):
            group_by = _match(cols, REGION_KW)
        elif any(w in q for w in ["category","segment","type"]):
            group_by = _match(cols, CATEGORY_KW)
        elif any(w in q for w in ["product","item","sku"]):
            group_by = _match(cols, PRODUCT_KW)
        elif any(w in q for w in ["rep","salesperson","agent"]):
            group_by = _match(cols, ["rep","sales rep"])
        elif any(w in q for w in ["customer","client","account"]):
            group_by = _match(cols, CUSTOMER_KW)

    # Detect metric
    metric = None
    if any(w in q for w in ["profit","margin","earnings"]):
        metric = _match(cols, PROFIT_KW)
    elif any(w in q for w in ["revenue","sales","income","amount"]):
        metric = _match(cols, REVENUE_KW)
    elif any(w in q for w in ["cost","expense"]):
        metric = _match(cols, COST_KW)
    elif any(w in q for w in ["units","quantity","volume"]):
        metric = _match(cols, QUANTITY_KW)
    if not metric:
        metric = _match(cols, REVENUE_KW) or _match(cols, PROFIT_KW)

    # Detect intent & chart
    intent = "revenue_by_group"; chart = "bar"; top_n = None
    sort_order = "desc"; time_grouping = None; time_col = None

    if any(w in q for w in ["trend","over time","monthly","quarterly","yearly","by month","by quarter","by year","time"]):
        intent = "trend_over_time"; chart = "line"
        time_col = _match(cols, DATE_KW)
        group_by = time_col
        time_grouping = "quarter" if "quarter" in q else "year" if ("year" in q and "quarterly" not in q) else "month"

    elif any(w in q for w in ["top","best","highest","leading","biggest"]):
        intent = "top_n"; chart = "bar"; top_n = _num(q) or 5; sort_order = "desc"

    elif any(w in q for w in ["bottom","worst","lowest","underperform","least"]):
        intent = "bottom_n"; chart = "bar"; top_n = _num(q) or 5; sort_order = "asc"

    elif any(w in q for w in ["distribution","breakdown","share","proportion","pie","percent","%"]):
        intent = "distribution"; chart = "pie"

    elif any(w in q for w in ["margin","profitability"]):
        intent = "margin_analysis"; chart = "bar"
        metric = _match(cols, PROFIT_KW) or metric

    elif any(w in q for w in ["heatmap","heat map","matrix"]):
        intent = "comparison"; chart = "heatmap"

    elif any(w in q for w in ["scatter","correlation","vs ","versus","compare"]):
        intent = "comparison"; chart = "scatter"

    elif any(w in q for w in ["kpi","summary","overview","dashboard","total","overall"]):
        intent = "kpi_summary"; chart = "table"

    elif any(w in q for w in ["forecast","predict","future","next month","projection"]):
        intent = "forecast"; chart = "line"

    elif any(w in q for w in ["anomaly","outlier","unusual","spike"]):
        intent = "anomaly"; chart = "scatter"

    title = _title(intent, metric, group_by, top_n, time_grouping)

    return {
        "intent": intent, "group_by": group_by, "metric": metric,
        "agg_func": "sum", "sort_order": sort_order, "top_n": top_n,
        "chart_type": chart, "time_column": time_col, "time_grouping": time_grouping,
        "filters": {}, "title": title,
        "insight_request": f"Analyze: {query}",
    }


def _cols_from_ctx(ctx: str):
    m = re.search(r"Columns:\s*(.+)", ctx)
    return [c.strip() for c in m.group(1).split(",")] if m else []

def _match(cols, kws):
    for kw in kws:
        for c in cols:
            if kw.lower() in c.lower(): return c
    return None

def _num(text: str):
    words = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    for w,n in words.items():
        if w in text: return n
    m = re.search(r"\b(\d+)\b", text)
    return int(m.group(1)) if m else None

def _title(intent, metric, group_by, top_n, tg):
    m = (metric or "Revenue").replace("_"," ").title()
    g = (group_by or "Segment").replace("_"," ").title()
    if intent == "trend_over_time":  return f"{m} Trend by {(tg or 'Period').title()}"
    if intent == "top_n":            return f"Top {top_n or 5} {g}s by {m}"
    if intent == "bottom_n":         return f"Bottom {top_n or 5} {g}s by {m}"
    if intent == "distribution":     return f"{m} Distribution by {g}"
    if intent == "margin_analysis":  return f"Profit Margin by {g}"
    if intent == "kpi_summary":      return "Business KPI Summary"
    if intent == "forecast":         return f"{m} Forecast"
    if intent == "anomaly":          return f"Anomaly Detection – {m}"
    return f"{m} by {g}"
