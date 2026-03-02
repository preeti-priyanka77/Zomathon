"""
SmartCart v2.0 — Production FastAPI Server
Full inference pipeline exposed as REST API.

Endpoints:
  POST /recommend         — personalised add-on recommendations
  POST /recommend/explain — recommendations + LLM explanations
  GET  /health            — liveness probe
  GET  /metrics           — runtime statistics

Latency budget: < 300ms P95 (200-300ms target per problem statement)
"""

import time
import uvicorn
from collections import deque
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import sys
from pathlib import Path

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference   import recommend
from cold_start  import recommend_with_fallback, get_user_tier
from explainer   import enrich_recommendations

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "SmartCart v2.0",
    description = "Context-Aware Add-On Recommendation API | Zomathon",
    version     = "2.0.0",
)

# ── Rolling latency tracker (last 1000 requests) ──────────────────────────────
_LATENCY_WINDOW:    deque = deque(maxlen=1000)
_REQUEST_COUNT      = 0
_ERROR_COUNT        = 0
_PREDICTIONS_SERVED = 0   # requests that returned ≥1 recommendation


# ── Request / Response schemas ────────────────────────────────────────────────
class RecommendContext(BaseModel):
    tier:         int   = Field(2,          ge=1, le=3)
    season:       str   = Field("Monsoon")
    zone_type:    str   = Field("CBD")
    hour:         int   = Field(13,         ge=0, le=23)
    day_of_week:  int   = Field(2,          ge=0, le=6)
    month:        int   = Field(6,          ge=1, le=12)
    distance_km:  float = Field(5.0,        ge=0)
    delivery_fee: float = Field(30.0,       ge=0)
    has_main:     int   = Field(1,          ge=0, le=1)
    has_side:     int   = Field(0,          ge=0, le=1)
    has_drink:    int   = Field(0,          ge=0, le=1)
    has_dessert:  int   = Field(0,          ge=0, le=1)


class RecommendRequest(BaseModel):
    user_id:    int          = Field(..., example=12345)
    cart_items: List[int]    = Field(..., min_length=1, example=[923, 936])
    context:    RecommendContext = Field(default_factory=RecommendContext)
    k:          int          = Field(8, ge=1, le=20)
    explain:    bool         = Field(False)   # set True for LLM explanations


class RecommendItem(BaseModel):
    item_id:     int
    score:       float
    strategy:    Optional[str]   = None
    explanation: Optional[str]   = None


class RecommendResponse(BaseModel):
    recommendations: List[RecommendItem]
    user_tier:       str
    latency_ms:      float
    request_id:      Optional[str] = None


class HealthResponse(BaseModel):
    status:      str
    uptime_ms:   float


class MetricsResponse(BaseModel):
    total_requests:     int
    error_count:        int
    predictions_served: int
    coverage_rate:      Optional[float]   # fraction of requests that received ≥1 prediction
    latency_p50_ms:     Optional[float]
    latency_p95_ms:     Optional[float]
    latency_p99_ms:     Optional[float]
    latency_mean_ms:    Optional[float]


# ── Startup timer ─────────────────────────────────────────────────────────────
_START_TIME = time.time()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status    = "ok",
        uptime_ms = round((time.time() - _START_TIME) * 1000, 1),
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    import numpy as np
    arr = list(_LATENCY_WINDOW)
    if arr:
        a = np.array(arr)
        p50, p95, p99, mean = (
            float(np.percentile(a, 50)),
            float(np.percentile(a, 95)),
            float(np.percentile(a, 99)),
            float(a.mean()),
        )
    else:
        p50 = p95 = p99 = mean = None

    coverage = round(_PREDICTIONS_SERVED / _REQUEST_COUNT, 4) if _REQUEST_COUNT > 0 else None

    return MetricsResponse(
        total_requests     = _REQUEST_COUNT,
        error_count        = _ERROR_COUNT,
        predictions_served = _PREDICTIONS_SERVED,
        coverage_rate      = coverage,
        latency_p50_ms     = p50,
        latency_p95_ms     = p95,
        latency_p99_ms     = p99,
        latency_mean_ms    = mean,
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(req: RecommendRequest):
    global _REQUEST_COUNT, _ERROR_COUNT, _PREDICTIONS_SERVED
    _REQUEST_COUNT += 1

    t0  = time.perf_counter()
    ctx = req.context.model_dump()

    try:
        tier = get_user_tier(req.user_id)

        # Route to appropriate strategy
        if tier == "warm":
            recs = recommend(req.user_id, req.cart_items, ctx, k=req.k)
            recs = [{**r, "strategy": "personalised_model"} for r in recs]
        else:
            recs = recommend_with_fallback(
                req.user_id, req.cart_items, ctx, k=req.k,
                full_model_fn=recommend,
            )

        # Optionally enrich with explanations
        if req.explain:
            recs = enrich_recommendations(recs, req.cart_items, ctx, use_llm=False)

        if recs:
            _PREDICTIONS_SERVED += 1

    except Exception as e:
        _ERROR_COUNT += 1
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = round((time.perf_counter() - t0) * 1000, 3)
    _LATENCY_WINDOW.append(latency_ms)

    return RecommendResponse(
        recommendations = [RecommendItem(**r) for r in recs],
        user_tier       = tier,
        latency_ms      = latency_ms,
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host      = "0.0.0.0",
        port      = 8000,
        workers   = 1,
        log_level = "info",
        app_dir   = str(Path(__file__).parent),
    )
