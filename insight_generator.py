"""
insight_generator.py
--------------------
Generates human-readable business insights from dataset analysis.
Supports: OpenAI GPT-4o, Google Gemini 1.5 Pro, and rule-based fallback (no API key needed).
"""

import re
import json
import pandas as pd
import numpy as np
from typing import Optional


# ── LLM Callers ─────────────────────────────────────────────────────────────────

def _call_openai(prompt: str, api_key: str) -> str:
    import openai
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert business analyst and data scientist. "
                    "Generate concise, actionable business insights from the data provided. "
                    "Write in clear professional English. "
                    "Format as numbered bullet points. Be specific with numbers and percentages."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


def _call_gemini(prompt: str, api_key: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        system_instruction=(
            "You are an expert business analyst. Generate concise, actionable insights. "
            "Use numbered bullet points and include specific numbers."
        )
    )
    response = model.generate_content(prompt)
    return response.text.strip()


# ── Rule-Based Insight Engine ───────────────────────────────────────────────────

def _rule_based_insights(df: pd.DataFrame, col_map: dict, kpis: dict) -> str:
    """Generate professional business insights without an API key."""
    insights = []

    rc  = col_map.get("revenue");   pc  = col_map.get("profit")
    cc  = col_map.get("category");  rgc = col_map.get("region")
    prc = col_map.get("product");   dc  = col_map.get("date")
    qc  = col_map.get("quantity")

    # Revenue & Profit overview
    if kpis.get("total_revenue") and kpis.get("total_profit"):
        rev  = kpis["total_revenue"]
        prof = kpis["total_profit"]
        margin = kpis.get("profit_margin_pct", 0)
        perf = "strong" if margin > 35 else "healthy" if margin > 20 else "below-average"
        insights.append(
            f"1. **Revenue Overview**: Total revenue of ${rev:,.0f} with a profit of ${prof:,.0f} "
            f"(margin: {margin:.1f}%). This is a {perf} profit margin "
            f"{'above the industry average of ~28%.' if margin > 28 else 'with room for improvement.'}"
        )

    # Category breakdown
    if cc and rc and cc in df.columns and rc in df.columns:
        cat_perf = df.groupby(cc)[rc].sum().sort_values(ascending=False)
        top_cat  = cat_perf.index[0]
        top_share = cat_perf.iloc[0] / cat_perf.sum() * 100
        insights.append(
            f"2. **Category Performance**: **{top_cat}** is the top revenue driver, "
            f"contributing **{top_share:.1f}%** of total revenue (${cat_perf.iloc[0]:,.0f}). "
            f"{cat_perf.index[-1]} is the weakest category at "
            f"{cat_perf.iloc[-1]/cat_perf.sum()*100:.1f}% share."
        )

    # Regional analysis
    if rgc and rc and rgc in df.columns and rc in df.columns:
        reg_perf   = df.groupby(rgc)[rc].sum().sort_values(ascending=False)
        best_reg   = reg_perf.index[0]
        worst_reg  = reg_perf.index[-1]
        best_share = reg_perf.iloc[0] / reg_perf.sum() * 100
        gap        = reg_perf.iloc[0] - reg_perf.iloc[-1]
        insights.append(
            f"3. **Regional Insights**: **{best_reg}** leads all regions with "
            f"${reg_perf.iloc[0]:,.0f} ({best_share:.1f}% share). "
            f"**{worst_reg}** underperforms with only ${reg_perf.iloc[-1]:,.0f}. "
            f"A ${gap:,.0f} gap between best and worst region represents a key growth opportunity."
        )

    # Profit margin by category
    if cc and rc and pc and all(c in df.columns for c in [cc, rc, pc]):
        margins = (df.groupby(cc)[pc].sum() / df.groupby(cc)[rc].sum() * 100).sort_values(ascending=False)
        best_m  = margins.index[0]
        worst_m = margins.index[-1]
        insights.append(
            f"4. **Margin Analysis**: **{best_m}** has the highest profit margin at "
            f"**{margins.iloc[0]:.1f}%**, making it the most profitable category per dollar of revenue. "
            f"**{worst_m}** has the lowest margin at {margins.iloc[-1]:.1f}%, "
            f"suggesting higher costs or lower pricing power."
        )

    # Top product
    if prc and rc and prc in df.columns and rc in df.columns:
        prod_perf  = df.groupby(prc)[rc].sum().sort_values(ascending=False)
        top_prod   = prod_perf.index[0]
        top_prod_r = prod_perf.iloc[0]
        n_prods    = len(prod_perf)
        top5_share = prod_perf.head(5).sum() / prod_perf.sum() * 100
        insights.append(
            f"5. **Product Insights**: **{top_prod}** is the bestseller with ${top_prod_r:,.0f} in revenue. "
            f"The top 5 products account for **{top5_share:.1f}%** of total revenue out of {n_prods} products total, "
            f"indicating a highly concentrated product portfolio."
        )

    # Time trend
    if dc and rc and dc in df.columns and rc in df.columns:
        try:
            tmp = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(tmp[dc]):
                tmp[dc] = pd.to_datetime(tmp[dc], errors="coerce")
            monthly = tmp.dropna(subset=[dc]).set_index(dc)[rc].resample("ME").sum()
            if len(monthly) >= 2:
                first_half = monthly.iloc[:len(monthly)//2].sum()
                second_half = monthly.iloc[len(monthly)//2:].sum()
                growth = (second_half - first_half) / first_half * 100 if first_half else 0
                peak_month = monthly.idxmax().strftime("%B %Y")
                trend_dir = "upward ↑" if growth > 5 else "declining ↓" if growth < -5 else "stable →"
                insights.append(
                    f"6. **Revenue Trend**: Business shows a **{trend_dir}** trajectory. "
                    f"Second-half revenue {'exceeded' if growth > 0 else 'fell below'} first-half by "
                    f"**{abs(growth):.1f}%**. Peak month was **{peak_month}** "
                    f"(${monthly.max():,.0f})."
                )
        except Exception:
            pass

    # Average order value
    if kpis.get("avg_order_value"):
        aov = kpis["avg_order_value"]
        insights.append(
            f"7. **Order Value**: Average order value is **${aov:,.0f}**. "
            f"{'Focused on high-value transactions.' if aov > 500 else 'Mix of small and large orders.'} "
            f"Upselling strategies could significantly impact total revenue."
        )

    # Growth opportunity
    if rgc and rc and rgc in df.columns and rc in df.columns:
        reg_perf = df.groupby(rgc)[rc].sum().sort_values(ascending=True)
        low_reg  = reg_perf.index[0]
        potential = reg_perf.mean() - reg_perf.iloc[0]
        if potential > 0:
            insights.append(
                f"8. **Growth Opportunity**: If **{low_reg}** reaches the average regional performance, "
                f"revenue could increase by **${potential:,.0f}** (~{potential/kpis.get('total_revenue',1)*100:.1f}% total growth). "
                f"Focus on targeted marketing or distribution improvements in this region."
            )

    return "\n\n".join(insights) if insights else "Load a dataset to generate insights."


# ── Main Entry Point ────────────────────────────────────────────────────────────

def generate_insights(df: pd.DataFrame, col_map: dict, kpis: dict,
                       context: str = "",
                       api_key: str = "", provider: str = "demo") -> str:
    """
    Generate business insights.
    Tries LLM first (if api_key provided), falls back to rule-based.
    """
    if api_key and provider != "demo":
        prompt = f"""Analyze this business dataset and generate 6–8 concise business insights.

Dataset context:
{context}

KPI Summary:
- Total Revenue: ${kpis.get('total_revenue', 0):,.0f}
- Total Profit: ${kpis.get('total_profit', 0):,.0f}
- Profit Margin: {kpis.get('profit_margin_pct', 0):.1f}%
- Best Region: {kpis.get('best_region', 'N/A')}
- Best Category: {kpis.get('best_category', 'N/A')}
- Best Product: {kpis.get('best_product', 'N/A')}

Format as numbered bullet points (1. 2. 3. ...).
Include specific numbers. Make insights actionable for business decisions.
Bold key findings using **text**."""
        try:
            if provider == "openai":
                return _call_openai(prompt, api_key)
            elif provider == "gemini":
                return _call_gemini(prompt, api_key)
        except Exception as e:
            return _rule_based_insights(df, col_map, kpis) + f"\n\n*Note: LLM unavailable ({e}). Using rule-based insights.*"

    return _rule_based_insights(df, col_map, kpis)


def generate_query_insight(df: pd.DataFrame, col_map: dict, kpis: dict,
                             query: str, result_summary: str,
                             api_key: str = "", provider: str = "demo") -> str:
    """Generate a short insight for a specific user query result."""
    if api_key and provider != "demo":
        prompt = (
            f"User asked: '{query}'\n\n"
            f"Analysis result:\n{result_summary}\n\n"
            "Provide a 2–3 sentence business insight about this result. "
            "Be specific, use numbers, and include one actionable recommendation."
        )
        try:
            if provider == "openai":
                return _call_openai(prompt, api_key)
            elif provider == "gemini":
                return _call_gemini(prompt, api_key)
        except Exception:
            pass

    # Rule-based per-query insights
    q = query.lower()
    rc = col_map.get("revenue"); pc = col_map.get("profit")
    rgc = col_map.get("region"); cc = col_map.get("category")
    prc = col_map.get("product")

    if "region" in q and rc and rgc and all(c in df.columns for c in [rgc, rc]):
        rg = df.groupby(rgc)[rc].sum()
        best = rg.idxmax(); worst = rg.idxmin()
        gap  = rg.max() - rg.min()
        return (f"**{best}** generates the most revenue at ${rg.max():,.0f}. "
                f"**{worst}** has the lowest at ${rg.min():,.0f}, creating a ${gap:,.0f} gap. "
                f"Consider reallocating sales resources to underperforming regions for higher ROI.")

    if any(w in q for w in ["product","top","best"]) and prc and rc and all(c in df.columns for c in [prc, rc]):
        pg = df.groupby(prc)[rc].sum()
        top5_share = pg.nlargest(5).sum() / pg.sum() * 100
        return (f"**{pg.idxmax()}** leads all products with ${pg.max():,.0f} in revenue. "
                f"Top 5 products drive **{top5_share:.1f}%** of total revenue. "
                f"Expand inventory and marketing for top performers to maximize returns.")

    if "margin" in q and cc and rc and pc and all(c in df.columns for c in [cc, rc, pc]):
        m = (df.groupby(cc)[pc].sum() / df.groupby(cc)[rc].sum() * 100)
        return (f"**{m.idxmax()}** is the most profitable category at {m.max():.1f}% margin. "
                f"**{m.idxmin()}** has the weakest margin at {m.min():.1f}%. "
                f"Improving pricing or reducing costs in low-margin categories could significantly boost profits.")

    if any(w in q for w in ["trend","monthly","time"]) and rc:
        rev = kpis.get("total_revenue", 0)
        avg = kpis.get("avg_order_value", 0)
        return (f"Total revenue stands at ${rev:,.0f} with an average order value of ${avg:,.0f}. "
                f"Seasonal patterns suggest targeted promotions in peak months could boost revenue further.")

    return f"Analysis complete. {result_summary[:200] if result_summary else 'Review the chart for detailed insights.'}"
