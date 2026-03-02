# SmartCart v2.0 — `src/`

All production Python modules for the CSAO recommendation system.
Run any script directly from the repo root with `python src/<module>.py`.

---

## Module Reference

```
src/
├── __init__.py               # Package initialisation
├── config.py                 # Centralised paths, constants, feature column lists
├── generate_data.py          # Synthetic dataset generator (50K users, 200K orders)
├── ranking_model.py          # LightGBM LambdaRank training + evaluation
├── inference.py              # Core recommendation pipeline (candidate gen → rank → rerank)
├── api.py                    # FastAPI production server  (/recommend /health /metrics)
├── cold_start.py             # Three-tier cold/cool/warm fallback strategy
├── explainer.py              # AI explainability (template + LLM mode)
├── evaluate.py               # Offline evaluation: NDCG@8, Precision@8, Recall@8, segment analysis
├── ab_testing.py             # A/B test design, statistical significance, guardrail checks
├── latency_test.py           # Live HTTP latency benchmark (500 requests, P50/P95/P99)
└── cart_transition_demo.py   # Cart progression demo: Main → Side → Drink → Dessert
```

---

## Quick Reference — Run Order

```bash
# 1. Generate synthetic data
python src/generate_data.py

# 2. Train the ranking model  ->  data/models/ranking_model.pkl
python src/ranking_model.py

# 3. Start the API server
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000

# 4. Benchmark live HTTP latency (run in a second terminal while server is up)
python src/latency_test.py

# 5. Demo cold-start strategy
python src/cold_start.py

# 6. Demo A/B test design + metric translation
python src/ab_testing.py

# 7. Demo AI explainer
python src/explainer.py

# 8. Offline segment error analysis (NDCG@8, Precision@8, Recall@8)
python src/evaluate.py

# 9. Cart transition demo
python src/cart_transition_demo.py
```

---

## Module Details

### `config.py`
Centralised constants — data paths, city tiers, cuisine lists, FP-Growth segment keys, feature column names, and the <=300 ms latency SLA budget.

### `generate_data.py`
Synthetic food delivery dataset generator. Produces realistic city-tier, cuisine, RFM, and seasonal variation. Outputs to `data/raw/` and `data/processed/`.

```bash
python src/generate_data.py                                   # defaults: 50K users, 200K orders
python src/generate_data.py --num-users 10000 --num-orders 50000
```

### `ranking_model.py`
Trains a LightGBM LambdaRank model optimised for NDCG@8. Outputs `data/models/ranking_model.pkl`, `feature_cols.pkl`, and a feature importance plot.

### `inference.py`
Core recommendation logic:
1. FP-Growth candidate lookup (<=50 candidates from pre-built segment index)
2. Feature assembly (user RFM + item features + cart state + context)
3. LightGBM scoring
4. Category-diversity re-ranking (max 3 per category in Top-8)

### `api.py`
FastAPI server exposing:

| Endpoint | Method | Description |
|---|---|---|
| `/recommend` | POST | Top-K personalised recommendations |
| `/health` | GET | Liveness probe |
| `/metrics` | GET | P50/P95/P99 latency, error rate, coverage rate |

### `cold_start.py`
Three-tier fallback:
- **Warm** (>=5 orders) — Full LightGBM model
- **Cool** (1-4 orders) — FP-Growth segment heuristic
- **Cold** (0 orders) — Global popularity + meal-time context

### `explainer.py`
Generates human-readable recommendation explanations.
- **Template mode** — zero latency, no API key required
- **LLM mode** — set `OPENAI_API_KEY` or `GEMINI_API_KEY` for GPT/Gemini enrichment

### `evaluate.py`
Full offline evaluation pipeline:
- Data quality & completeness checks
- Candidate generation efficiency (target: <50ms P95)
- Feature fetch efficiency (target: <60ms P95)
- Segment-level NDCG@8, Precision@8, Recall@8 (by tier / zone / age group / season)
- Production readiness checklist

### `ab_testing.py`
Statistical A/B test framework:
- Sample size + duration calculator
- Full metric tracking: AOV, attach rate, CTR, rail order share, coverage, C2O
- Guardrail enforcement (cart abandonment, P99 latency, complaint rate)
- Business metric translation (NDCG -> AOV lift)

### `latency_test.py`
Sends 500 live HTTP POST requests to the API server and reports:
- P50 / P95 / P99 end-to-end latency
- Server-side pipeline latency (from response body)
- Pass/fail vs 300ms SLA with headroom multiplier

### `cart_transition_demo.py`
Demonstrates dynamic recommendation evolution as items are added:
1. Main course only — system pushes sides + drinks
2. Main + side — system promotes drinks, surfaces desserts
3. Main + side + drink — system shifts to desserts + complements

---

## Importing from Notebooks

```python
import sys
sys.path.insert(0, '../src')

from inference import recommend
from cold_start import recommend_with_fallback
from explainer import enrich_recommendations
```
