"""
app.py — DataLensAI v2.0
FastAPI Application Entry Point

Handles:
  - File upload & session management
  - API routing for all analytics endpoints
  - CORS, error handling, lifecycle events
  - Static file serving (HTML frontend)

Author: Agam Kumar Verma
"""

import os
import uuid
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from data_engine import DataEngine
from dataset_profiler import DatasetProfiler
from chart_engine import ChartEngine
from insight_generator import InsightGenerator
from ai_query_engine import AIQueryEngine
from prediction_engine import PredictionEngine

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("DataLensAI")

# ── In-memory session store ──────────────────────────────────────────────────
# Maps session_id -> { engine, profiler, charts, insights }
# For production: replace with Redis or PostgreSQL-backed cache
SESSIONS: dict[str, dict] = {}

# ── Config ───────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_MB", "50"))
MAX_ROWS         = int(os.getenv("MAX_ROWS", "500000"))
CORS_ORIGINS     = os.getenv("CORS_ORIGINS", "*").split(",")
PORT             = int(os.getenv("PORT", "8000"))
OPENAI_KEY       = os.getenv("OPENAI_API_KEY", "")
GEMINI_KEY       = os.getenv("GEMINI_API_KEY", "")
AI_PROVIDER      = os.getenv("AI_PROVIDER", "gemini")


# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 DataLensAI v2.0 starting up…")
    log.info(f"   AI Provider : {AI_PROVIDER}")
    log.info(f"   Gemini Key  : {'✓ set' if GEMINI_KEY else '✗ not set'}")
    log.info(f"   OpenAI Key  : {'✓ set' if OPENAI_KEY else '✗ not set'}")
    log.info(f"   CORS Origins: {CORS_ORIGINS}")
    yield
    log.info("🛑 DataLensAI shutting down — clearing sessions…")
    SESSIONS.clear()


# ── App instance ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="DataLensAI API",
    description="AI-powered Business Intelligence backend for DataLensAI v2.0",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    session_id: str
    query: str
    use_ai: bool = False
    api_key: Optional[str] = None
    provider: Optional[str] = None  # "gemini" | "openai"


class PredictRequest(BaseModel):
    session_id: str
    target_column: str
    periods: int = 6
    method: str = "linear"  # "linear" | "moving_avg" | "exponential"


class InsightRequest(BaseModel):
    session_id: str
    custom_question: Optional[str] = None
    use_ai: bool = False
    api_key: Optional[str] = None
    provider: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_session(session_id: str) -> dict:
    """Retrieve a session or raise 404."""
    if session_id not in SESSIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Upload a dataset first.",
        )
    return SESSIONS[session_id]


def resolve_api_key(req_key: Optional[str], req_provider: Optional[str]) -> tuple[str, str]:
    """Resolve API key and provider from request or environment."""
    provider = req_provider or AI_PROVIDER
    if req_key and len(req_key) > 8:
        return req_key, provider
    if provider == "gemini" and GEMINI_KEY:
        return GEMINI_KEY, "gemini"
    if provider == "openai" and OPENAI_KEY:
        return OPENAI_KEY, "openai"
    return "", provider


# ═══════════════════════════════════════════════════════════════════════════
# ROOT — serve the HTML frontend
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/", include_in_schema=False)
async def serve_frontend():
    html_path = Path("DataLensAI_v2.html")
    if html_path.exists():
        return FileResponse(html_path)
    return JSONResponse(
        {"message": "DataLensAI API v2.0 is running. Place DataLensAI_v2.html in this directory."}
    )


# ═══════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "sessions_active": len(SESSIONS),
        "ai_provider": AI_PROVIDER,
        "gemini_configured": bool(GEMINI_KEY),
        "openai_configured": bool(OPENAI_KEY),
    }


# ═══════════════════════════════════════════════════════════════════════════
# POST /api/upload — Ingest CSV and build full analytics session
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/api/upload", tags=["Dataset"])
async def upload_dataset(
    file: UploadFile = File(...),
    session_id: str = Form(default=""),
    max_rows: int = Form(default=MAX_ROWS),
):
    # Validate file type
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    # Validate file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Limit is {MAX_FILE_SIZE_MB} MB.",
        )

    # Generate session ID
    sid = session_id or str(uuid.uuid4())
    log.info(f"Upload: {file.filename} ({size_mb:.2f} MB) → session={sid}")

    try:
        # ── 1. Parse & clean data ──────────────────────────────────────────
        engine = DataEngine(filename=file.filename, max_rows=max_rows)
        engine.load_from_bytes(content)
        engine.clean()

        # ── 2. Profile dataset ─────────────────────────────────────────────
        profiler = DatasetProfiler(engine)
        profile  = profiler.full_profile()

        # ── 3. Build KPIs ──────────────────────────────────────────────────
        kpis = profiler.build_kpis()

        # ── 4. Build charts ────────────────────────────────────────────────
        chart_engine = ChartEngine(engine, profiler)
        charts       = chart_engine.build_all_charts()

        # ── 5. Rule-based insights ─────────────────────────────────────────
        ig      = InsightGenerator(engine, profiler)
        insights = ig.rule_based_insights()

        # ── Store session ──────────────────────────────────────────────────
        SESSIONS[sid] = {
            "engine":   engine,
            "profiler": profiler,
            "charts":   charts,
            "insights": insights,
            "kpis":     kpis,
            "filename": file.filename,
        }

        return {
            "session_id":      sid,
            "filename":        file.filename,
            "rows":            engine.row_count,
            "columns":         len(engine.columns),
            "quality_score":   profile["quality"]["score"],
            "detected_fields": engine.detected_fields,
            "kpis":            kpis,
            "profile":         profile,
            "charts_count":    len(charts),
            "insights_count":  len(insights.get("insights", [])),
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.exception(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/profile/{session_id} — Full dataset profile
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/profile/{session_id}", tags=["Dataset"])
async def get_profile(session_id: str):
    sess = get_session(session_id)
    return sess["profiler"].full_profile()


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/kpis/{session_id} — KPI cards
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/kpis/{session_id}", tags=["Analytics"])
async def get_kpis(session_id: str):
    sess = get_session(session_id)
    return {"kpis": sess["kpis"]}


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/charts/{session_id} — Chart configurations
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/charts/{session_id}", tags=["Analytics"])
async def get_charts(session_id: str, format: str = "chartjs"):
    """
    format: "chartjs" → Chart.js-compatible JSON (for frontend)
            "plotly"  → Plotly JSON (for server-side rendering)
    """
    sess = get_session(session_id)
    engine = ChartEngine(sess["engine"], sess["profiler"])
    if format == "plotly":
        return {"charts": engine.build_plotly_charts()}
    return {"charts": sess["charts"]}


# ═══════════════════════════════════════════════════════════════════════════
# POST /api/insights — AI or rule-based insights
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/api/insights", tags=["AI"])
async def get_insights(req: InsightRequest):
    sess = get_session(req.session_id)
    ig   = InsightGenerator(sess["engine"], sess["profiler"])

    if req.use_ai:
        api_key, provider = resolve_api_key(req.api_key, req.provider)
        if not api_key:
            log.warning("No API key — falling back to rule-based insights")
            insights = ig.rule_based_insights(custom_question=req.custom_question)
        else:
            try:
                insights = await ig.ai_insights(
                    api_key=api_key,
                    provider=provider,
                    custom_question=req.custom_question,
                )
            except Exception as e:
                log.warning(f"AI insight error ({e}) — falling back to rule-based")
                insights = ig.rule_based_insights(custom_question=req.custom_question)
    else:
        insights = ig.rule_based_insights(custom_question=req.custom_question)

    # Update cached insights
    SESSIONS[req.session_id]["insights"] = insights
    return insights


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/insights/{session_id} — Cached insights (fast)
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/insights/{session_id}", tags=["AI"])
async def get_cached_insights(session_id: str):
    sess = get_session(session_id)
    return sess.get("insights", {})


# ═══════════════════════════════════════════════════════════════════════════
# POST /api/query — Natural language query
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/api/query", tags=["AI"])
async def query_data(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    sess = get_session(req.session_id)
    qe   = AIQueryEngine(sess["engine"], sess["profiler"])

    if req.use_ai:
        api_key, provider = resolve_api_key(req.api_key, req.provider)
        if api_key:
            try:
                return await qe.ai_answer(req.query, api_key=api_key, provider=provider)
            except Exception as e:
                log.warning(f"AI query error ({e}) — falling back to rule-based")

    return qe.rule_answer(req.query)


# ═══════════════════════════════════════════════════════════════════════════
# POST /api/predict — Prediction / forecasting
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/api/predict", tags=["ML"])
async def predict(req: PredictRequest):
    sess   = get_session(req.session_id)
    engine = sess["engine"]

    if req.target_column not in engine.df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{req.target_column}' not found in dataset.",
        )

    pe     = PredictionEngine(engine)
    result = pe.forecast(
        target_col=req.target_column,
        periods=max(1, min(req.periods, 36)),
        method=req.method,
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/predict/anomalies/{session_id} — Anomaly detection
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/predict/anomalies/{session_id}", tags=["ML"])
async def detect_anomalies(session_id: str, column: Optional[str] = None):
    sess = get_session(session_id)
    pe   = PredictionEngine(sess["engine"])
    return pe.detect_anomalies(target_col=column)


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/predict/correlation/{session_id} — Correlation matrix
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/predict/correlation/{session_id}", tags=["ML"])
async def correlation_matrix(session_id: str):
    sess = get_session(session_id)
    pe   = PredictionEngine(sess["engine"])
    return pe.correlation_matrix()


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/export/csv/{session_id}", tags=["Export"])
async def export_csv(session_id: str):
    sess   = get_session(session_id)
    engine = sess["engine"]
    csv_bytes = engine.export_csv()
    filename  = engine.filename.replace(".csv", "_clean.csv")
    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/export/json/{session_id}", tags=["Export"])
async def export_json(session_id: str):
    import json
    sess     = get_session(session_id)
    engine   = sess["engine"]
    profiler = sess["profiler"]
    payload  = {
        "meta": {
            "file":        engine.filename,
            "rows":        engine.row_count,
            "columns":     len(engine.columns),
            "quality":     profiler.quality_score(),
            "detected":    engine.detected_fields,
            "generated_at": str(__import__("datetime").datetime.now().isoformat()),
        },
        "kpis":     sess["kpis"],
        "profile":  profiler.full_profile(),
        "insights": sess.get("insights"),
    }
    filename = engine.filename.replace(".csv", "_analysis.json")
    return StreamingResponse(
        iter([json.dumps(payload, indent=2, default=str).encode()]),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ═══════════════════════════════════════════════════════════════════════════
# DELETE /api/session/{session_id} — Clear session
# ═══════════════════════════════════════════════════════════════════════════
@app.delete("/api/session/{session_id}", tags=["System"])
async def delete_session(session_id: str):
    if session_id in SESSIONS:
        del SESSIONS[session_id]
        log.info(f"Session deleted: {session_id}")
        return {"message": f"Session '{session_id}' cleared."}
    raise HTTPException(status_code=404, detail="Session not found.")


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/sessions — List active sessions (dev only)
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/sessions", tags=["System"], include_in_schema=False)
async def list_sessions():
    return {
        "active": len(SESSIONS),
        "sessions": [
            {
                "id":       sid,
                "file":     s["filename"],
                "rows":     s["engine"].row_count,
                "columns":  len(s["engine"].columns),
            }
            for sid, s in SESSIONS.items()
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL ERROR HANDLER
# ═══════════════════════════════════════════════════════════════════════════
@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    log.error(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        log_level="info",
    )
