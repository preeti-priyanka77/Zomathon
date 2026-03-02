"""
SmartCart v2.0 — Phase 2 / 3 / 4
Inference pipeline, latency measurement, baseline comparison.
Zomathon by Coding Ninjas
"""

import gc
import time
import pickle
import random
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR  = ROOT / "data" / "features"
MODELS_DIR    = ROOT / "data" / "models"


# ── Load all artifacts ONCE at import time ────────────────────────────────────
print("⏳  Loading artifacts …")

with open(PROCESSED_DIR / "candidate_lookup_fpgrowth.pkl", "rb") as f:
    _CANDIDATE_LOOKUP: dict = pickle.load(f)          # {(tier,season,zone): {item_id: [(cand,conf)]}}

_COOC_RAW: pd.DataFrame = pickle.load(open(PROCESSED_DIR / "cooccurrence.pkl", "rb"))
# Build fast O(1) co-occurrence dict: {(item_i, item_j): count}
_COOC_DICT: dict = {
    (int(row[0]), int(row[1])): int(row[2])
    for row in _COOC_RAW.itertuples(index=False)
}
del _COOC_RAW
gc.collect()

_USER_FEAT: pd.DataFrame = (
    pd.read_csv(PROCESSED_DIR / "user_features.csv")
    .set_index("user_id")
)

_ITEM_FEAT: pd.DataFrame = (
    pd.read_csv(PROCESSED_DIR / "item_features.csv")
    .set_index("item_id")
)

_SCALER     = joblib.load(FEATURES_DIR / "feature_scaler.pkl")
_BOOSTER    = joblib.load(MODELS_DIR   / "ranking_model.pkl")
_FEAT_COLS  = joblib.load(MODELS_DIR   / "feature_cols.pkl")   # 77 ordered columns

# Scaler column order (must match exactly what scaler was fit on)
_SCALE_COLS = list(_SCALER.feature_names_in_)

# Item category lookup for diversity post-processing
_item_cat_raw = pd.read_csv(
    ROOT / "data" / "raw" / "order_items_v2_full.csv",
    usecols=["item_id", "category"],
).drop_duplicates("item_id")
_ITEM_CATEGORY: dict[int, str] = dict(
    zip(_item_cat_raw["item_id"].astype(int), _item_cat_raw["category"])
)
del _item_cat_raw

# Build per-column mean/scale dicts for manual scaling (avoids sklearn name validation)
_SCALE_MEAN = dict(zip(_SCALE_COLS, _SCALER.mean_))
_SCALE_STD  = dict(zip(_SCALE_COLS, _SCALER.scale_))

print(f"✅  Artifacts loaded — {len(_CANDIDATE_LOOKUP)} segments, "
      f"{len(_COOC_DICT):,} co-occurrence pairs, "
      f"{len(_USER_FEAT):,} users, {len(_ITEM_FEAT):,} items")


# ── Candidate generation ──────────────────────────────────────────────────────
def _get_candidates(
    cart_items: list[int],
    tier: int,
    season: str,
    zone_type: str,
    max_candidates: int = 50,
) -> dict[int, float]:
    """
    Returns {item_id: confidence} for at most max_candidates items.
    Confidence = max FP-Growth confidence across cart items.
    """
    seg_key = (tier, season, zone_type)
    seg_lookup = _CANDIDATE_LOOKUP.get(seg_key, {})

    # Fall back to any available segment if this combo is missing
    if not seg_lookup:
        seg_lookup = next(iter(_CANDIDATE_LOOKUP.values()), {})

    candidates: dict[int, float] = {}
    cart_set = set(cart_items)

    for cart_item in cart_items:
        for cand_id, conf in seg_lookup.get(cart_item, []):
            if cand_id not in cart_set:
                # Keep max confidence per candidate
                if conf > candidates.get(cand_id, -1.0):
                    candidates[cand_id] = conf

    # Sort by confidence desc, take top max_candidates
    sorted_cands = sorted(candidates.items(), key=lambda x: -x[1])[:max_candidates]
    return dict(sorted_cands)


# ── Co-occurrence helpers ─────────────────────────────────────────────────────
def _cooc_with_cart(item_id: int, cart_items: list[int]) -> float:
    """Sum of co-occurrence counts of item with all cart items."""
    return float(sum(_COOC_DICT.get((item_id, ci), 0) for ci in cart_items))


def _item_cooc_strength(item_id: int, cart_items: list[int]) -> float:
    """Max co-occurrence count of item with any single cart item."""
    return float(max((_COOC_DICT.get((item_id, ci), 0) for ci in cart_items), default=0))


# ── Feature builder ───────────────────────────────────────────────────────────
def _build_features(
    user_id:    int,
    cart_items: list[int],
    candidates: dict[int, float],
    context:    dict,
) -> pd.DataFrame:
    """
    Build one row per candidate with all 77 model features.
    Missing data → 0 (safe for LightGBM trees).
    """
    n = len(candidates)
    cand_ids   = list(candidates.keys())
    cand_confs = list(candidates.values())

    # ── Base dict with all feature cols = 0.0 ─────────────────────────────
    rows: dict[str, np.ndarray] = {col: np.zeros(n, dtype="float32") for col in _FEAT_COLS}

    # ── User features ──────────────────────────────────────────────────────
    if user_id in _USER_FEAT.index:
        u = _USER_FEAT.loc[user_id]
        for col in ["total_orders", "avg_order_value", "max_order_value",
                    "days_since_last_order", "user_tenure_days"]:
            if col in rows and col in _USER_FEAT.columns:
                rows[col][:] = float(u[col])

    # ── Item features (per candidate) ──────────────────────────────────────
    item_ids_arr = np.array(cand_ids)
    for col in ["item_order_count", "avg_price", "popularity_rank"]:
        if col not in rows:
            continue
        mask = np.isin(item_ids_arr, _ITEM_FEAT.index)
        if mask.any():
            vals = _ITEM_FEAT.loc[item_ids_arr[mask], col].values.astype("float32")
            rows[col][mask] = vals

    # ── Context features ───────────────────────────────────────────────────
    hour       = int(context.get("hour", 12))
    dow        = int(context.get("day_of_week", 0))
    month      = int(context.get("month", 1))
    dist       = float(context.get("distance_km", 5.0))
    fee        = float(context.get("delivery_fee", 30.0))
    cart_sz    = len(cart_items)

    rows["hour"][:]              = hour
    rows["day_of_week"][:]       = dow
    rows["month"][:]             = month
    rows["is_weekend"][:]        = float(dow >= 5)
    rows["distance_km"][:]       = dist
    rows["delivery_fee"][:]      = fee
    rows["cart_size"][:]         = cart_sz
    rows["cart_size_log"][:]     = float(np.log1p(cart_sz))
    rows["distance_normalized"][:] = dist / 20.0       # approx max from training
    rows["delivery_fee_normalized"][:] = fee / 100.0   # approx max

    # Cart composition
    if "has_main" in rows:   rows["has_main"][:]    = float(context.get("has_main",    0))
    if "has_side" in rows:   rows["has_side"][:]    = float(context.get("has_side",    0))
    if "has_drink" in rows:  rows["has_drink"][:]   = float(context.get("has_drink",   0))
    if "has_dessert" in rows: rows["has_dessert"][:] = float(context.get("has_dessert", 0))

    # Cyclic time encodings
    rows["hour_sin"][:] = float(np.sin(2 * np.pi * hour / 24))
    rows["hour_cos"][:] = float(np.cos(2 * np.pi * hour / 24))
    rows["dow_sin"][:] = float(np.sin(2 * np.pi * dow  / 7))
    rows["dow_cos"][:] = float(np.cos(2 * np.pi * dow  / 7))

    # ── Co-occurrence features (per candidate) ─────────────────────────────
    cooc_w   = np.array([_cooc_with_cart(cid, cart_items) for cid in cand_ids], dtype="float32")
    cooc_str = np.array([_item_cooc_strength(cid, cart_items) for cid in cand_ids], dtype="float32")
    rows["cooc_with_cart"][:]    = cooc_w
    rows["item_cooc_strength"][:] = cooc_str
    rows["item_complementarity"][:] = cooc_str  # proxy

    # FP-Growth confidence as segment_cooc_score
    rows["segment_cooc_score"][:] = np.array(cand_confs, dtype="float32")

    # cart_diversity_score: fraction of unique categories, proxy via cart_size
    rows["cart_diversity_score"][:] = float(min(cart_sz / 4.0, 1.0))

    # ── Interaction features ───────────────────────────────────────────────
    avg_ov = float(_USER_FEAT.loc[user_id, "avg_order_value"]) if user_id in _USER_FEAT.index else 250.0
    item_prices = rows["avg_price"].copy()
    rows["affordability_ratio"][:] = np.where(avg_ov > 0, item_prices / avg_ov, 0.0)
    rows["tier_x_price"][:]        = float(context.get("tier", 2)) * item_prices

    # ── One-hot: zone_type ─────────────────────────────────────────────────
    zone = context.get("zone_type", "CBD")
    if zone == "Residential" and "zone_type_Residential" in rows:
        rows["zone_type_Residential"][:] = 1.0
    elif zone == "Student" and "zone_type_Student" in rows:
        rows["zone_type_Student"][:] = 1.0
    # CBD is the reference (all zeros)

    # ── One-hot: season ────────────────────────────────────────────────────
    season = context.get("season", "Monsoon")
    if season == "Summer" and "season_Summer" in rows:
        rows["season_Summer"][:] = 1.0
    elif season == "Winter" and "season_Winter" in rows:
        rows["season_Winter"][:] = 1.0
    # Monsoon is the reference (all zeros)

    # ── Assemble DataFrame in correct column order ─────────────────────────
    df = pd.DataFrame(rows, columns=_FEAT_COLS)
    return df, cand_ids


# ── Diversity re-ranking ─────────────────────────────────────────────────────
def _diversify(
    ranked: list[dict],
    k: int,
    max_per_category: int = 3,
) -> list[dict]:
    """
    Category-aware greedy re-ranking.
    Ensures at most `max_per_category` items per food category (drink/side/
    dessert/main) in the final top-k list.

    Algorithm:
      Pass 1 — add items in score order, skipping those that would exceed the
               per-category quota.
      Pass 2 — fill remaining slots with the highest-scoring leftovers.
    """
    selected: list[dict] = []
    cat_counts: dict[str, int] = {}
    leftover: list[dict] = []

    for rec in ranked:
        cat = _ITEM_CATEGORY.get(rec["item_id"], "unknown")
        if cat_counts.get(cat, 0) < max_per_category:
            selected.append({**rec, "category": cat})
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            if len(selected) >= k:
                break
        else:
            leftover.append({**rec, "category": cat})

    # Fill remaining slots from leftovers (already score-sorted)
    for rec in leftover:
        if len(selected) >= k:
            break
        selected.append(rec)

    return selected


# ── Scale + predict ───────────────────────────────────────────────────────────
def _score(df: pd.DataFrame) -> np.ndarray:
    """Apply per-column StandardScaler manually, then run booster."""
    # Scale only features that exist in both scaler AND model feature list
    # (tier is in scaler but was dropped before training — skip safely)
    for col in _SCALE_COLS:
        if col in df.columns and col in _FEAT_COLS:
            std = _SCALE_STD.get(col, 1.0) or 1.0
            df[col] = ((df[col] - _SCALE_MEAN.get(col, 0.0)) / std).astype("float32")
    X = df[_FEAT_COLS].values.astype("float32")
    return _BOOSTER.predict(X)


# ── Public API ────────────────────────────────────────────────────────────────
def recommend(
    user_id:    int,
    cart_items: list[int],
    context:    dict,
    k:          int = 8,
) -> list[dict]:
    """
    Return top-k recommended add-on items.

    Parameters
    ----------
    user_id    : int
    cart_items : list[int]   — item IDs already in cart
    context    : dict with keys:
                   tier        (int)   1 | 2 | 3
                   season      (str)   "Summer" | "Monsoon" | "Winter"
                   zone_type   (str)   "CBD" | "Residential" | "Student"
                   hour        (int)   0-23
                   day_of_week (int)   0-6
                   month       (int)   1-12
                   distance_km (float)
                   delivery_fee (float)
                   has_main / has_side / has_drink / has_dessert (int 0|1)
    k          : number of recommendations (default 8)

    Returns
    -------
    list of dicts: [{"item_id": int, "score": float}, ...]  sorted desc
    """
    if not cart_items:
        return []

    # Step 1 — Candidates
    candidates = _get_candidates(
        cart_items,
        tier      = context.get("tier",      2),
        season    = context.get("season",    "Monsoon"),
        zone_type = context.get("zone_type", "CBD"),
    )
    if not candidates:
        return []

    # Step 2 — Features
    df, cand_ids = _build_features(user_id, cart_items, candidates, context)

    # Step 3 — Score
    scores = _score(df)

    # Step 4 — Top-k
    top_idx = np.argpartition(scores, -min(k, len(scores)))[-min(k, len(scores)):]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    raw = [{"item_id": int(cand_ids[i]), "score": float(scores[i])} for i in top_idx]
    return _diversify(raw, k=k)


# ── Phase 3: Latency measurement ──────────────────────────────────────────────
def measure_latency(n_calls: int = 500) -> dict:
    """Measure end-to-end latency over n_calls realistic inputs."""
    # Sample real user_ids and item_ids
    user_ids  = _USER_FEAT.index.tolist()
    item_ids  = _ITEM_FEAT.index.tolist()
    seasons   = ["Summer", "Monsoon", "Winter"]
    zones     = ["CBD", "Residential", "Student"]

    rng = random.Random(42)
    latencies_ms = []

    for _ in range(n_calls):
        uid       = rng.choice(user_ids)
        cart      = rng.sample(item_ids, k=rng.randint(1, 4))
        ctx = {
            "tier":         rng.randint(1, 3),
            "season":       rng.choice(seasons),
            "zone_type":    rng.choice(zones),
            "hour":         rng.randint(0, 23),
            "day_of_week":  rng.randint(0, 6),
            "month":        rng.randint(1, 12),
            "distance_km":  round(rng.uniform(1, 20), 1),
            "delivery_fee": round(rng.uniform(10, 80), 0),
            "has_main":     rng.randint(0, 1),
            "has_side":     rng.randint(0, 1),
            "has_drink":    rng.randint(0, 1),
            "has_dessert":  rng.randint(0, 1),
        }
        t0 = time.perf_counter()
        recommend(uid, cart, ctx)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    arr    = np.array(latencies_ms)
    result = {
        "mean_ms": float(np.mean(arr)),
        "p50_ms":  float(np.percentile(arr, 50)),
        "p95_ms":  float(np.percentile(arr, 95)),
        "p99_ms":  float(np.percentile(arr, 99)),
        "passes_target": float(np.percentile(arr, 95)) < 300.0,
    }

    print("\nLATENCY RESULTS (n={:,})".format(n_calls))
    print("─" * 35)
    print(f"Mean:  {result['mean_ms']:6.2f} ms")
    print(f"P50:   {result['p50_ms']:6.2f} ms")
    print(f"P95:   {result['p95_ms']:6.2f} ms  ← target <300ms")
    print(f"P99:   {result['p99_ms']:6.2f} ms")
    print("─" * 35)
    status = "✓ PASS" if result["passes_target"] else "✗ FAIL"
    print(f"Status: {status}")
    return result


# ── Phase 4: Baseline comparison ──────────────────────────────────────────────
def baseline_comparison(test_path: str | None = None) -> None:
    """Compare SmartCart v2.0 vs popularity baseline on test set."""
    if test_path is None:
        test_path = FEATURES_DIR / "test_features.parquet"

    print("\n⏳  Loading test set …")
    test_df = pd.read_parquet(test_path)

    # Reduce memory
    for col in test_df.select_dtypes("float64").columns:
        test_df[col] = test_df[col].astype("float32")

    DROP_COLS = ["order_id", "user_id", "item_id", "split", "order_date",
                 "is_vegetarian", "tier", "city"]
    TARGET    = "label"
    FEAT_COLS = [c for c in _FEAT_COLS if c in test_df.columns]

    # ── SmartCart scores ───────────────────────────────────────────────────
    for col in _SCALE_COLS:
        if col in test_df.columns and col in _FEAT_COLS:
            std = _SCALE_STD.get(col, 1.0) or 1.0
            test_df[col] = ((test_df[col] - _SCALE_MEAN.get(col, 0.0)) / std).astype("float32")
    X_test             = test_df[FEAT_COLS].values.astype("float32")
    smartcart_scores   = _BOOSTER.predict(X_test)
    test_df["sc_score"] = smartcart_scores

    # ── Baseline: global popularity rank ───────────────────────────────────
    # Lower popularity_rank = more popular → invert for scoring
    max_rank = test_df["popularity_rank"].max() if "popularity_rank" in test_df.columns else 1.0
    test_df["bl_score"] = (max_rank - test_df.get("popularity_rank", max_rank)).fillna(0)

    # ── Compute Precision@8 and NDCG@8 ────────────────────────────────────
    def _metrics(df: pd.DataFrame, score_col: str, k: int = 8):
        # Sort by (order_id, score desc) once, then take top-k per group vectorized
        df_s = df[["order_id", TARGET, score_col]].copy()
        df_s = df_s.sort_values(["order_id", score_col], ascending=[True, False])
        df_s["rank"] = df_s.groupby("order_id").cumcount()
        top_k = df_s[df_s["rank"] < k]

        # Precision@k
        hits   = top_k.groupby("order_id")[TARGET].sum()
        p_at_k = float((hits / k).mean())

        # NDCG@k — vectorized discount sum
        top_k = top_k.copy()
        top_k["disc"] = top_k[TARGET] / np.log2(top_k["rank"] + 2)
        dcg   = top_k.groupby("order_id")["disc"].sum()

        # Ideal DCG: best possible ordering for each order
        ideal = df_s.sort_values(["order_id", TARGET], ascending=[True, False])
        ideal = ideal.copy()
        ideal["irank"] = ideal.groupby("order_id").cumcount()
        ideal = ideal[ideal["irank"] < k].copy()
        ideal["idisc"] = ideal[TARGET] / np.log2(ideal["irank"] + 2)
        idcg  = ideal.groupby("order_id")["idisc"].sum()

        # Align & compute NDCG (skip orders with no positives)
        mask   = idcg > 0
        n_at_k = float((dcg[mask] / idcg[mask]).mean())

        return p_at_k, n_at_k

    sc_p8, sc_n8 = _metrics(test_df, "sc_score")
    bl_p8, bl_n8 = _metrics(test_df, "bl_score")

    lift_p = (sc_p8 - bl_p8) / (bl_p8 + 1e-9) * 100
    lift_n = (sc_n8 - bl_n8) / (bl_n8 + 1e-9) * 100

    print("\nBASELINE vs SMARTCART v2.0 (Test Set)")
    print("═" * 52)
    print(f"{'Metric':<16} {'Baseline':>10} {'SmartCart':>10} {'Lift':>10}")
    print("─" * 52)
    print(f"{'Precision@8':<16} {bl_p8:>10.4f} {sc_p8:>10.4f} {lift_p:>+9.1f}%")
    print(f"{'NDCG@8':<16} {bl_n8:>10.4f} {sc_n8:>10.4f} {lift_n:>+9.1f}%")
    print("═" * 52)


# ── Runner ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Smoke test ─────────────────────────────────────────────────────────
    print("\n🧪  Smoke test …")
    # Find a cart item that has at least 2 distinct candidates
    seg_key  = list(_CANDIDATE_LOOKUP.keys())[0]
    seg_map  = _CANDIDATE_LOOKUP[seg_key]
    smoke_cart_item = next(
        (item for item, cands in seg_map.items() if len(cands) >= 2),
        list(seg_map.keys())[0]
    )
    sample_uid  = _USER_FEAT.index[0]
    sample_cart = [smoke_cart_item]   # single item → candidates can't all be excluded
    sample_ctx  = {
        "tier":         seg_key[0],
        "season":       seg_key[1],
        "zone_type":    seg_key[2],
        "hour":         13,
        "day_of_week":  2,
        "month":        6,
        "distance_km":  5.0,
        "delivery_fee": 30.0,
        "has_main": 1, "has_side": 0, "has_drink": 0, "has_dessert": 0,
    }
    recs = recommend(sample_uid, sample_cart, sample_ctx)
    print(f"  user_id={sample_uid}, cart={sample_cart}")
    print(f"  Top-{len(recs)} recommendations:")
    for r in recs:
        print(f"    item_id={r['item_id']:>6}  score={r['score']:.4f}")

    # Phase 3 — Latency
    print("\n📏  Phase 3: Latency measurement …")
    measure_latency(n_calls=500)

    # Phase 4 — Baseline comparison
    print("\n📊  Phase 4: Baseline comparison …")
    baseline_comparison()

    print("\n🎉  All phases complete!")
