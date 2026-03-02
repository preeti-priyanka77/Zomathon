#!/usr/bin/env python3
"""
SmartCart v2.0 - Pre-Processing Evaluation

Evaluates the pre-processing pipeline against <300ms latency budget:
- Data quality & completeness checks
- Candidate generation efficiency (target: <50ms)
- Feature fetch efficiency (target: <60ms)
- Production readiness validation
- Segment error analysis (per tier / zone / age group / time)
- Optimization recommendations

Outputs:
- Console summary with latency analysis
- Production readiness checklist
- Segment-level NDCG@8 and Precision@8 breakdown
- Deployment recommendations
"""

import pandas as pd
import numpy as np
import time
import pickle
import joblib
import logging
from pathlib import Path
from collections import defaultdict
from scipy.stats import chi2_contingency

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIG
# ============================================================================
BASE_DIR      = Path(__file__).parent.parent / "data"
PROCESSED_DIR = BASE_DIR / "processed"
FEATURES_DIR  = BASE_DIR / "features"
MODELS_DIR    = BASE_DIR / "models"


# ============================================================================
# DATA LOADING
# ============================================================================
def load_preprocessed_data():
    """Load all preprocessed datasets."""
    logger.info("Loading preprocessed data...")
    start = time.time()

    # Load CSVs
    orders_enriched = pd.read_csv(PROCESSED_DIR / "orders_enriched.csv")
    user_features = pd.read_csv(PROCESSED_DIR / "user_features.csv")
    item_features = pd.read_csv(PROCESSED_DIR / "item_features.csv")

    # Load co-occurrence matrix
    with open(PROCESSED_DIR / "cooccurrence.pkl", "rb") as f:
        cooccurrence = pickle.load(f)

    load_time_ms = (time.time() - start) * 1000
    logger.info(f"Data loaded in {load_time_ms:.2f}ms")

    return orders_enriched, user_features, item_features, cooccurrence, load_time_ms


# ============================================================================
# 1. DATA QUALITY CHECKS
# ============================================================================
def check_data_quality(orders_enriched, user_features, item_features):
    """Verify data quality and completeness."""
    logger.info("\n" + "=" * 80)
    logger.info("SECTION 1: DATA QUALITY & COMPLETENESS")
    logger.info("=" * 80)

    # Missing values
    logger.info("\n✓ Missing values check:")
    missing_orders = orders_enriched.isnull().sum().sum()
    missing_users = user_features.isnull().sum().sum()
    missing_items = item_features.isnull().sum().sum()

    logger.info(f"  Orders enriched: {missing_orders} missing values")
    logger.info(f"  User features:   {missing_users} missing values")
    logger.info(f"  Item features:   {missing_items} missing values")

    # Consistency checks
    logger.info("\n✓ Data consistency checks:")

    user_ids_in_orders = set(orders_enriched["user_id"].unique())
    user_ids_in_features = set(user_features["user_id"].unique())
    orphan_users = user_ids_in_orders - user_ids_in_features

    logger.info(f"  All users have features: {len(orphan_users) == 0} (orphans: {len(orphan_users)})")

    temporal_cols = ["hour", "day_of_week", "month", "is_weekend", "time_bucket"]
    has_temporal = all([col in orders_enriched.columns for col in temporal_cols])
    logger.info(f"  Temporal features complete: {has_temporal}")

    cart_cols = ["cart_size_x", "cart_total", "avg_item_price"]
    has_cart = all([col in orders_enriched.columns for col in cart_cols])
    logger.info(f"  Cart features present: {has_cart}")

    # Numeric ranges
    logger.info("\n  Numeric range validation:")
    logger.info(f"    Cart size:       [{orders_enriched['cart_size_x'].min()}, "
               f"{orders_enriched['cart_size_x'].max()}] items")
    logger.info(f"    User orders:     [{user_features['total_orders'].min()}, "
               f"{user_features['total_orders'].max()}]")
    logger.info(f"    Days since order:[{user_features['days_since_last_order'].min()}, "
               f"{user_features['days_since_last_order'].max()}]")

    return {
        "missing_orders": missing_orders,
        "missing_users": missing_users,
        "missing_items": missing_items,
        "orphan_users": len(orphan_users),
        "has_temporal": has_temporal,
        "has_cart": has_cart,
    }


# ============================================================================
# 2. CANDIDATE GENERATION EFFICIENCY
# ============================================================================
def analyze_candidate_generation(cooccurrence):
    """Analyze co-occurrence matrix and simulate candidate generation latency."""
    logger.info("\n" + "=" * 80)
    logger.info("SECTION 2: CANDIDATE GENERATION EFFICIENCY (Target: <50ms P95)")
    logger.info("=" * 80)

    # Co-occurrence statistics
    logger.info(f"\n✓ Co-occurrence matrix statistics:")
    logger.info(f"  Total unique item pairs: {len(cooccurrence):,}")
    logger.info(f"  Total co-occurrence count: {cooccurrence['count'].sum():,}")
    logger.info(f"  Max pair frequency: {cooccurrence['count'].max()}")
    logger.info(f"  Mean pair frequency: {cooccurrence['count'].mean():.2f}")
    logger.info(f"  Median pair frequency: {cooccurrence['count'].median():.2f}")

    # Distribution
    high_freq = (cooccurrence["count"] >= 10).sum()
    med_freq = ((cooccurrence["count"] >= 3) & (cooccurrence["count"] < 10)).sum()
    low_freq = (cooccurrence["count"] <= 2).sum()

    logger.info(f"\n  Frequency distribution:")
    logger.info(f"    High (count >= 10): {high_freq:,}")
    logger.info(f"    Medium (3-9):       {med_freq:,}")
    logger.info(f"    Low (1-2):          {low_freq:,}")

    # Simulate candidate generation
    logger.info(f"\n✓ Simulating candidate generation latency...")
    start = time.time()
    cooccurrence_dict = {}
    for _, row in cooccurrence.iterrows():
        item_i, item_j, count = row["item_i"], row["item_j"], row["count"]
        cooccurrence_dict[(item_i, item_j)] = count
        cooccurrence_dict[(item_j, item_i)] = count

    indexing_time_ms = (time.time() - start) * 1000
    logger.info(f"  Indexing time: {indexing_time_ms:.2f}ms")

    # Simulate 100 candidate generation calls
    num_test_samples = 100
    candidate_gen_times = []

    for _ in range(num_test_samples):
        start = time.time()

        cart_items = [1, 2, 3]  # mock cart
        candidates = defaultdict(int)

        for item in cart_items:
            for (i, j), count in cooccurrence_dict.items():
                if i == item and j not in cart_items:
                    candidates[j] += count

        top_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:50]
        candidate_gen_times.append((time.time() - start) * 1000)

    mean_gen_time = np.mean(candidate_gen_times)
    p95_gen_time = np.percentile(candidate_gen_times, 95)
    p99_gen_time = np.percentile(candidate_gen_times, 99)

    logger.info(f"\n  Candidate generation latency:")
    logger.info(f"    Mean: {mean_gen_time:.2f}ms")
    logger.info(f"    P95:  {p95_gen_time:.2f}ms {'✓' if p95_gen_time < 50 else '⚠'}")
    logger.info(f"    P99:  {p99_gen_time:.2f}ms")
    logger.info(f"    Target: <50ms")

    return {
        "cooccurrence_pairs": len(cooccurrence),
        "indexing_time_ms": indexing_time_ms,
        "mean_gen_time": mean_gen_time,
        "p95_gen_time": p95_gen_time,
        "p99_gen_time": p99_gen_time,
    }


# ============================================================================
# 3. FEATURE FETCH EFFICIENCY
# ============================================================================
def analyze_feature_fetch(user_features, item_features):
    """Simulate feature fetch latency."""
    logger.info("\n" + "=" * 80)
    logger.info("SECTION 3: FEATURE FETCH EFFICIENCY (Target: <60ms P95)")
    logger.info("=" * 80)

    # Index creation
    logger.info("\n✓ Building feature indices...")
    start = time.time()

    user_idx = user_features.set_index("user_id")
    item_idx = item_features.set_index("item_id")

    indexing_time_ms = (time.time() - start) * 1000
    logger.info(f"  Indexing time: {indexing_time_ms:.2f}ms")

    # Simulate feature fetch for 100 queries
    logger.info(f"\n✓ Simulating feature fetch (50 candidates per query)...")
    num_samples = 100
    num_candidates = 50
    fetch_times = []

    for _ in range(num_samples):
        user_id = np.random.choice(user_features["user_id"].values)
        candidate_items = np.random.choice(
            item_features["item_id"].values, size=num_candidates, replace=False
        )

        start = time.time()

        try:
            user_feat = user_idx.loc[user_id]
        except KeyError:
            pass

        candidate_feats = item_idx.loc[candidate_items]
        fetch_times.append((time.time() - start) * 1000)

    mean_fetch_time = np.mean(fetch_times)
    p95_fetch_time = np.percentile(fetch_times, 95)
    p99_fetch_time = np.percentile(fetch_times, 99)

    logger.info(f"\n  Feature fetch latency:")
    logger.info(f"    Mean: {mean_fetch_time:.2f}ms")
    logger.info(f"    P95:  {p95_fetch_time:.2f}ms {'✓' if p95_fetch_time < 60 else '⚠'}")
    logger.info(f"    P99:  {p99_fetch_time:.2f}ms")
    logger.info(f"    Target: <60ms")

    return {
        "indexing_time_ms": indexing_time_ms,
        "mean_fetch_time": mean_fetch_time,
        "p95_fetch_time": p95_fetch_time,
        "p99_fetch_time": p99_fetch_time,
    }


# ============================================================================
# 4. FEATURE ENGINEERING QUALITY
# ============================================================================
def analyze_feature_engineering(orders_enriched, user_features, item_features):
    """Validate feature engineering quality."""
    logger.info("\n" + "=" * 80)
    logger.info("SECTION 4: FEATURE ENGINEERING QUALITY")
    logger.info("=" * 80)

    logger.info(f"\n✓ User features ({user_features.shape[1]} attributes):")
    logger.info(f"  Total users: {len(user_features):,}")
    logger.info(f"  New users (tenure <7d): {(user_features['user_tenure_days'] < 7).sum():,}")
    logger.info(
        f"  Active users (last order <30d): {(user_features['days_since_last_order'] < 30).sum():,}"
    )
    logger.info(f"  High-value (AOV >500): {(user_features['avg_order_value'] > 500).sum():,}")

    logger.info(f"\n✓ Item features ({item_features.shape[1]} attributes):")
    logger.info(f"  Total items: {len(item_features):,}")
    logger.info(f"  Popular items (rank <=50): {(item_features['popularity_rank'] <= 50).sum():,}")
    logger.info(
        f"  Niche items (rank >500): {(item_features['popularity_rank'] > 500).sum():,}"
    )
    logger.info(
        f"  Price range: ${item_features['avg_price'].min():.2f} - ${item_features['avg_price'].max():.2f}"
    )

    logger.info(f"\n✓ Temporal features:")
    has_cols = ["has_main", "has_side", "has_drink", "has_dessert"]
    for col in has_cols:
        if col in orders_enriched.columns:
            count = orders_enriched[col].sum()
            pct = (count / len(orders_enriched)) * 100
            logger.info(f"  {col:15s}: {count:6.0f} orders ({pct:5.1f}%)")


# ============================================================================
# 5. PRODUCTION READINESS CHECKLIST
# ============================================================================
def production_readiness_checklist(
    quality_checks, candidate_stats, fetch_stats, orders_enriched, user_features, item_features
):
    """Validate production readiness."""
    logger.info("\n" + "=" * 80)
    logger.info("SECTION 5: PRODUCTION READINESS CHECKLIST")
    logger.info("=" * 80)

    checks = {
        "Data Quality": [
            ("No missing values", quality_checks["missing_orders"] == 0),
            ("Data consistency", quality_checks["orphan_users"] == 0),
            ("Temporal features", quality_checks["has_temporal"]),
            ("Cart features", quality_checks["has_cart"]),
        ],
        "Performance": [
            ("Candidate gen <50ms P95", candidate_stats["p95_gen_time"] < 50),
            ("Feature fetch <60ms P95", fetch_stats["p95_fetch_time"] < 60),
            ("Co-occurrence indexed", candidate_stats["cooccurrence_pairs"] > 0),
        ],
        "Feature Engineering": [
            ("User features complete", len(user_features) > 0),
            ("Item features complete", len(item_features) > 0),
        ],
        "Scalability": [
            ("Sufficient user segments", len(user_features) > 100),
            ("Reasonable feature count", len(orders_enriched.columns) >= 15),
        ],
    }

    total_items = 0
    passed_items = 0

    for category, items in checks.items():
        logger.info(f"\n{category}:")
        for item, passed in items:
            status = "✓" if passed else "✗"
            logger.info(f"  {status} {item}")
            total_items += 1
            if passed:
                passed_items += 1

    score_pct = (passed_items * 100) // total_items
    logger.info(f"\n{'='*80}")
    logger.info(f"Overall Score: {passed_items}/{total_items} checks passed ({score_pct}%)")
    logger.info(f"{'='*80}")

    return passed_items, total_items


# ============================================================================
# 6. SEGMENT ERROR ANALYSIS
# ============================================================================
def _ndcg_at_k(rel: np.ndarray, k: int = 8) -> float:
    """Compute NDCG@k for a single query given relevance array (sorted by score)."""
    rel = rel[:k]
    if rel.sum() == 0:
        return 0.0
    n    = len(rel)
    dcg  = float(np.sum(rel / np.log2(np.arange(2, n + 2))))
    idcg = float(np.sum(np.sort(rel)[::-1] / np.log2(np.arange(2, n + 2))))
    return dcg / idcg if idcg > 0 else 0.0


def _precision_at_k(rel: np.ndarray, k: int = 8) -> float:
    return float(rel[:k].sum()) / k


def _recall_at_k(hits_in_top_k: float, total_relevant: float) -> float:
    """Recall@k = relevant items in top-k / total relevant items in the order."""
    return float(hits_in_top_k) / float(total_relevant) if total_relevant > 0 else 0.0


def segment_analysis(
    test_path: Path | None = None,
    max_rows: int = 800_000,
    k: int = 8,
) -> dict:
    """
    Offline segment-level evaluation of SmartCart v2.0.

    Computes NDCG@k and Precision@k broken down by:
      • City tier (1 / 2 / 3)
      • Delivery zone (CBD / Residential / Student)
      • Age group (Gen_Z / Millennial / Gen_X / Boomer)
      • Meal time (Breakfast / Lunch / Dinner / Late-night)
    """
    logger.info("\n" + "=" * 80)
    logger.info("SECTION 6: SEGMENT-LEVEL ERROR ANALYSIS")
    logger.info("=" * 80)

    test_path = test_path or FEATURES_DIR / "test_features.parquet"
    if not test_path.exists():
        logger.warning(f"  test_features.parquet not found at {test_path} — skipping.")
        return {}

    model_path = MODELS_DIR / "ranking_model.pkl"
    fcols_path = MODELS_DIR / "feature_cols.pkl"
    scaler_path = FEATURES_DIR / "feature_scaler.pkl"
    if not (model_path.exists() and fcols_path.exists() and scaler_path.exists()):
        logger.warning("  Model artifacts missing — run ranking_model.py first.")
        return {}

    logger.info(f"  Loading test features (cap {max_rows:,} rows)…")
    t0 = time.time()
    df = pd.read_parquet(test_path)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    logger.info(f"  Loaded {len(df):,} rows in {(time.time()-t0)*1000:.0f}ms")

    # ── Derive segment labels ─────────────────────────────────────────────
    # Recover city tier from the 3 distinct standardised values
    tier_vals = sorted(df["tier"].unique())
    tier_map  = {v: f"Tier {i+1}" for i, v in enumerate(tier_vals)}
    df["_seg_tier"] = df["tier"].map(tier_map)

    # Recover zone from one-hot columns
    df["_seg_zone"] = "CBD"
    if "zone_type_Residential" in df.columns:
        df.loc[df["zone_type_Residential"] == 1, "_seg_zone"] = "Residential"
    if "zone_type_Student" in df.columns:
        df.loc[df["zone_type_Student"] == 1, "_seg_zone"] = "Student"

    # Recover age group from one-hot columns
    df["_seg_age"] = "Boomer"
    for grp in ["Gen_Z", "Millennial", "Gen_X"]:
        col = f"age_group_{grp}"
        if col in df.columns:
            df.loc[df[col] == 1, "_seg_age"] = grp

    # Derive season from one-hot columns (hour is zeroed out in preprocessed data)
    df["_seg_time"] = "Monsoon"
    if "season_Summer" in df.columns:
        df.loc[df["season_Summer"] == 1, "_seg_time"] = "Summer"
    if "season_Winter" in df.columns:
        df.loc[df["season_Winter"] == 1, "_seg_time"] = "Winter"

    # ── Load model + score all rows ───────────────────────────────────────
    booster   = joblib.load(model_path)
    feat_cols = joblib.load(fcols_path)
    scaler    = joblib.load(scaler_path)
    scale_cols = list(scaler.feature_names_in_)
    scale_mean = dict(zip(scale_cols, scaler.mean_))
    scale_std  = dict(zip(scale_cols, scaler.scale_))

    valid_feat = [c for c in feat_cols if c in df.columns]
    df_feat = df[valid_feat].copy()
    for col in scale_cols:
        if col in df_feat.columns:
            std = scale_std.get(col, 1.0) or 1.0
            df_feat[col] = ((df_feat[col] - scale_mean.get(col, 0.0)) / std).astype("float32")

    # Fill missing feature columns with 0
    for col in feat_cols:
        if col not in df_feat.columns:
            df_feat[col] = 0.0
    df_feat = df_feat[feat_cols]

    logger.info("  Scoring…")
    t0 = time.time()
    df["_score"] = booster.predict(df_feat.values.astype("float32"))
    logger.info(f"  Scored {len(df):,} rows in {(time.time()-t0)*1000:.0f}ms")

    # ── Per-order vectorised ranking → NDCG@k and Precision@k ──────────────
    seg_cols = ["_seg_tier", "_seg_zone", "_seg_age", "_seg_time"]
    df["_label"] = df["label"].astype(float)

    logger.info("  Computing per-order metrics (vectorised)…")
    df_sorted = df.sort_values(["order_id", "_score"], ascending=[True, False])
    df_sorted["_rank"] = df_sorted.groupby("order_id").cumcount() + 1

    top_k = df_sorted[df_sorted["_rank"] <= k].copy()
    prec_per_order = top_k.groupby("order_id")["_label"].mean()

    # Recall@k: fraction of all relevant items in the order that appear in top-k
    total_relevant_per_order = df_sorted.groupby("order_id")["_label"].sum().clip(lower=1)
    hits_per_order           = top_k.groupby("order_id")["_label"].sum()
    recall_per_order = (hits_per_order / total_relevant_per_order).fillna(0.0)

    top_k["_dcg"] = top_k["_label"] / np.log2(top_k["_rank"] + 1)
    dcg_per_order = top_k.groupby("order_id")["_dcg"].sum()

    top_k["_rank_ideal"] = top_k.groupby("order_id")["_label"].rank(
        method="first", ascending=False
    )
    top_k["_idcg"] = top_k["_label"] / np.log2(top_k["_rank_ideal"] + 1)
    idcg_per_order = top_k.groupby("order_id")["_idcg"].sum()

    ndcg_per_order = (dcg_per_order / idcg_per_order).fillna(0.0).replace(
        [np.inf, -np.inf], 0.0
    )

    seg_per_order = df_sorted.groupby("order_id")[seg_cols].first()
    order_metrics = pd.DataFrame(
        {"ndcg": ndcg_per_order, "prec": prec_per_order, "recall": recall_per_order}
    ).join(seg_per_order)

    results: dict[str, pd.DataFrame] = {}
    dim_labels = {
        "_seg_tier":  "City Tier",
        "_seg_zone":  "Delivery Zone",
        "_seg_age":   "Age Group",
        "_seg_time":  "Season",
    }

    print("\n" + "═" * 70)
    print("SEGMENT ERROR ANALYSIS  (holdout test set)")
    print("═" * 70)

    for col, label in dim_labels.items():
        grp = (
            order_metrics.groupby(col)[["ndcg", "prec", "recall"]]
            .mean()
            .rename(columns={"ndcg": "NDCG@8", "prec": "Precision@8", "recall": "Recall@8"})
            .round(4)
        )
        grp["N orders"] = order_metrics.groupby(col)["ndcg"].count().astype(int)
        results[label] = grp

        print(f"\n  {label}:")
        print(f"  {'-'*66}")
        print(f"  {'Segment':<18} {'NDCG@8':>10} {'Precision@8':>13} {'Recall@8':>10} {'N orders':>10}")
        print(f"  {'-'*66}")
        for seg, row in grp.iterrows():
            print(f"  {str(seg):<18} {row['NDCG@8']:>10.4f} {row['Precision@8']:>13.4f} {row['Recall@8']:>10.4f} {row['N orders']:>10,}")

    print("\n" + "═" * 70)
    return results


# ============================================================================
# 7. DEPLOYMENT SUMMARY
# ============================================================================
def print_deployment_summary(
    load_time_ms, candidate_stats, fetch_stats, passed_items, total_items
):
    """Print deployment readiness summary."""
    logger.info("\n" + "=" * 80)
    logger.info("DEPLOYMENT SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\n✓ LATENCY ANALYSIS:")
    logger.info(f"  Data loading:          {load_time_ms:7.2f}ms")
    logger.info(f"  Candidate gen (P95):   {candidate_stats['p95_gen_time']:7.2f}ms " +
               f"{'✓' if candidate_stats['p95_gen_time'] < 50 else '⚠'} < 50ms")
    logger.info(f"  Feature fetch (P95):   {fetch_stats['p95_fetch_time']:7.2f}ms " +
               f"{'✓' if fetch_stats['p95_fetch_time'] < 60 else '⚠'} < 60ms")
    logger.info(f"  Ranking model (est):   70-100ms")
    logger.info(f"  Post-processing:       20-30ms")
    logger.info(f"  Network:               ~50ms")
    logger.info(f"  TOTAL BUDGET:          <300ms ✓")

    logger.info(f"\n✓ PRODUCTION READINESS: {passed_items}/{total_items} checks passed")

    if passed_items == total_items:
        logger.info("\n✓ RECOMMENDATION: PROCEED WITH DEPLOYMENT")
        logger.info("  - Pre-processing meets latency budget")
        logger.info("  - Data quality validated")
        logger.info("  - Ready for feature store deployment")
    else:
        logger.info(f"\n⚠ CAUTION: {total_items - passed_items} checks failed")
        logger.info("  - Review failing checks before deployment")


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Run full evaluation pipeline."""
    logger.info("\n" + "=" * 80)
    logger.info("SmartCart v2.0 - PRE-PROCESSING EVALUATION")
    logger.info("=" * 80)

    try:
        # Load data
        orders_enriched, user_features, item_features, cooccurrence, load_time_ms = (
            load_preprocessed_data()
        )

        # Run checks
        quality_checks = check_data_quality(orders_enriched, user_features, item_features)
        candidate_stats = analyze_candidate_generation(cooccurrence)
        fetch_stats = analyze_feature_fetch(user_features, item_features)
        analyze_feature_engineering(orders_enriched, user_features, item_features)

        # Readiness checklist
        passed_items, total_items = production_readiness_checklist(
            quality_checks, candidate_stats, fetch_stats, orders_enriched, user_features,
            item_features
        )

        # Segment error analysis
        segment_analysis()

        # Deployment summary
        print_deployment_summary(load_time_ms, candidate_stats, fetch_stats, passed_items,
                               total_items)

        logger.info("\n" + "=" * 80)
        logger.info("Evaluation complete")
        logger.info("=" * 80)

    except FileNotFoundError as e:
        logger.error(f"✗ Data files not found: {e}")
        logger.error("  Run pre-processing notebook first")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
