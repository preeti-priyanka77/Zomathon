"""
SmartCart v2.0 — Phase 1: LightGBM Ranking Model
Zomathon by Coding Ninjas
"""

import gc
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, ndcg_score
import lightgbm as lgb

# ── Paths ────────────────────────────────────────────────────────────────────
FEATURES_DIR = Path("data/features")
MODELS_DIR   = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Columns to drop before training ─────────────────────────────────────────
DROP_COLS = [
    "order_id", "user_id", "item_id",
    "split", "order_date",
    "is_vegetarian",   # raw bool — is_vegetarian_int is the encoded version
    "tier",            # raw int — used in interaction features already
    "city",            # raw string — not encoded
]
TARGET_COL = "label"

# ── Model config (exact as per AGENT_INSTRUCTIONS) ───────────────────────────
MODEL_PARAMS = dict(
    objective         = "binary",
    metric            = "auc",
    n_estimators      = 300,
    learning_rate     = 0.05,
    max_depth         = 6,
    num_leaves        = 63,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_samples = 20,
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)


# ── Memory optimiser ─────────────────────────────────────────────────────────
def reduce_mem(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64→float32 and int64→int32. Saves ~50% RAM."""
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes("int64").columns:
        df[col] = df[col].astype("int32")
    return df


# ── Metric helpers ───────────────────────────────────────────────────────────
def precision_at_k(df_val: pd.DataFrame, score_col: str = "score", k: int = 8) -> float:
    """Mean Precision@k across all orders."""
    def _p_at_k(grp):
        top = grp.nlargest(k, score_col)
        return top[TARGET_COL].sum() / k
    return df_val.groupby("order_id").apply(_p_at_k).mean()


def ndcg_at_k(df_val: pd.DataFrame, score_col: str = "score", k: int = 8) -> float:
    """Mean NDCG@k across all orders."""
    scores = []
    for _, grp in df_val.groupby("order_id"):
        y_true  = grp[TARGET_COL].values.reshape(1, -1)
        y_score = grp[score_col].values.reshape(1, -1)
        if y_true.sum() == 0:
            continue
        scores.append(ndcg_score(y_true, y_score, k=k))
    return float(np.mean(scores))


# ── Data loading ─────────────────────────────────────────────────────────────
def load_split(split: str) -> pd.DataFrame:
    path = FEATURES_DIR / f"{split}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    df = pd.read_parquet(path)
    df = reduce_mem(df)
    mem = df.memory_usage(deep=True).sum() / 1024**3
    print(f"[{split}] loaded  shape={df.shape}  positives={df[TARGET_COL].sum():,}  RAM={mem:.2f} GB")
    return df


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # 1. Load train
    print("\n⏳  Loading train parquet …")
    train_df = load_split("train")

    cols_to_drop_train = [c for c in DROP_COLS if c in train_df.columns]
    feature_cols = [c for c in train_df.columns if c not in cols_to_drop_train + [TARGET_COL]]

    X_train = train_df[feature_cols].values.astype("float32")
    y_train = train_df[TARGET_COL].values

    # Build lgb.Dataset and immediately free the pandas DataFrame
    pos_count  = y_train.sum()
    neg_count  = len(y_train) - pos_count
    scale_pos  = neg_count / pos_count

    lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
    del train_df, X_train
    gc.collect()
    print(f"✅  lgb.Dataset built — freed train DataFrame from RAM")

    # 2. Load val (keep DataFrame for metric computation)
    print("\n⏳  Loading val parquet …")
    val_df  = load_split("val")
    cols_to_drop_val = [c for c in DROP_COLS if c in val_df.columns]

    val_meta = val_df[["order_id", TARGET_COL]].copy()
    X_val    = val_df[feature_cols].values.astype("float32")
    y_val    = val_df[TARGET_COL].values

    lgb_val  = lgb.Dataset(X_val, label=y_val, reference=lgb_train, free_raw_data=False)
    del val_df
    gc.collect()

    print(f"\n✅  Feature count: {len(feature_cols)}")
    print(f"    Class balance — scale_pos_weight={scale_pos:.1f}")

    # 3. Train
    print(f"\n⏳  Training LightGBM …")
    params = {**MODEL_PARAMS}
    params["scale_pos_weight"] = scale_pos
    n_estimators = params.pop("n_estimators")

    callbacks = [
        lgb.early_stopping(30, verbose=False),
        lgb.log_evaluation(50),
    ]

    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round   = n_estimators,
        valid_sets        = [lgb_val],
        callbacks         = callbacks,
    )

    # 4. Evaluate
    print("\n📊  Evaluating on val set …")
    y_pred_proba = booster.predict(X_val)

    auc = roc_auc_score(y_val, y_pred_proba)
    val_meta["score"] = y_pred_proba

    p8 = precision_at_k(val_meta, k=8)
    n8 = ndcg_at_k(val_meta,      k=8)

    print("\n" + "═" * 45)
    print("  EVALUATION RESULTS (Val Set)")
    print("═" * 45)
    print(f"  AUC           : {auc:.4f}  {'✓' if auc > 0.70 else '⚠ below target 0.70'}")
    print(f"  Precision@8   : {p8:.4f}  {'✓' if p8  > 0.30 else '⚠ below target 0.30'}")
    print(f"  NDCG@8        : {n8:.4f}  {'✓' if n8  > 0.50 else '⚠ below target 0.50'}")
    print("═" * 45)

    if auc < 0.60:
        print("⚠️  WARNING: AUC < 0.60 — saving model anyway")

    del X_val, y_val, lgb_train, lgb_val
    gc.collect()

    # 5. Save booster (used directly in inference)
    model_path = MODELS_DIR / "ranking_model.pkl"
    joblib.dump(booster, model_path)
    print(f"\n✅  Booster saved → {model_path}")

    # Also save feature column list so inference knows column order
    feat_path = MODELS_DIR / "feature_cols.pkl"
    joblib.dump(feature_cols, feat_path)
    print(f"✅  Feature columns saved → {feat_path}")

    # 6. Feature importance plot
    importance = booster.feature_importance(importance_type="split")
    fi_df = pd.DataFrame({"feature": feature_cols, "importance": importance})
    fi_df = fi_df.nlargest(20, "importance")

    plt.figure(figsize=(10, 7))
    plt.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="#f97316")
    plt.xlabel("Importance (split)")
    plt.title("SmartCart v2.0 — Top 20 Feature Importances")
    plt.tight_layout()
    fig_path = MODELS_DIR / "feature_importance.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"✅  Feature importance plot saved → {fig_path}")

    print("\n📋  Top 20 Features:")
    for _, row in fi_df.iterrows():
        print(f"    {row['feature']:<40} {row['importance']:>6.0f}")

    print("\n🎉  Phase 1 complete. Run Phase 2 next → src/inference.py")
    return booster, feature_cols


if __name__ == "__main__":
    main()
