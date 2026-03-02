"""
SmartCart v2.0 — A/B Testing Framework
Statistical design and analysis for live recommendation experiments.

Covers:
  1. Experiment design (sample size, duration)
  2. Online metric tracking
  3. Statistical significance testing
  4. Guardrail checks
  5. Business decision logic
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ── Experiment Configuration ─────────────────────────────────────────────────
@dataclass
class ABTestConfig:
    """
    Full A/B test configuration for SmartCart v2.0 launch.
    Aligned with Zomato's business objectives.
    """

    name:              str   = "smartcart_v2_vs_popularity_baseline"
    description:       str   = "SmartCart v2.0 personalised recommendations vs top-8 global popularity"

    # Traffic split
    control_pct:       float = 0.50            # 50% see popularity baseline
    treatment_pct:     float = 0.50            # 50% see SmartCart v2.0

    # Statistical parameters
    alpha:             float = 0.05            # significance level (5%)
    power:             float = 0.80            # statistical power (80%)
    mde:               float = 0.02            # minimum detectable effect (2% AOV lift)

    # Primary metrics
    primary_metric:    str   = "average_order_value"   # AOV
    secondary_metrics: list  = field(default_factory=lambda: [
        "addon_acceptance_rate",    # % of recommendations accepted
        "csao_attach_rate",         # % of CSAO rails shown where ≥1 item was added (target ≥35%)
        "csao_rail_order_share",    # % of ALL completed orders where ≥1 CSAO rail item was purchased
        "click_through_rate",       # % of users who tapped/clicked any item on the CSAO rail
        "cart_to_order_rate",       # % of sessions that complete checkout
        "items_per_order",          # avg number of items per order
        "session_revenue",          # total revenue per session
        "coverage_rate",            # % of eligible requests that received ≥1 prediction (target ≥95%)
    ])

    # Guardrail metrics — experiment STOPS if these are violated
    guardrail_metrics: dict = field(default_factory=lambda: {
        "cart_abandonment_rate":    {"direction": "increase", "threshold": 0.05},   # must NOT increase >5%
        "session_completion_rate":  {"direction": "decrease", "threshold": 0.03},   # must NOT drop >3%
        "p99_latency_ms":          {"direction": "increase", "threshold": 50.0},    # must NOT increase >50ms
        "user_complaint_rate":     {"direction": "increase", "threshold": 0.01},    # must NOT increase >1%
    })

    # Duration
    min_days:          int   = 14              # minimum 2 weeks (captures weekly patterns)
    max_days:          int   = 28              # cap at 4 weeks

    # Novelty effect — first 3 days excluded from analysis
    burn_in_days:      int   = 3


# ── Sample Size Calculator ────────────────────────────────────────────────────
def required_sample_size(
    baseline_mean:   float,
    mde_relative:    float = 0.02,   # 2% relative lift
    std_dev:         float | None = None,
    alpha:           float = 0.05,
    power:           float = 0.80,
) -> dict:
    """
    Calculate minimum sample size per arm for the A/B test.

    Returns dict with per-arm and total sample size + estimated days.
    """
    if std_dev is None:
        std_dev = baseline_mean * 0.5      # assume 50% CV (typical food delivery)

    mde_absolute = baseline_mean * mde_relative
    effect_size  = mde_absolute / std_dev

    # Two-sample t-test (two-tailed)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)
    n_per_arm = int(np.ceil(2 * ((z_alpha + z_beta) / effect_size) ** 2))

    # Zomato context: assume 50K active users, 30% daily active rate
    daily_users_per_arm = int(50_000 * 0.30 * 0.50)    # 50% traffic split
    estimated_days = int(np.ceil(n_per_arm / daily_users_per_arm))

    return {
        "n_per_arm":          n_per_arm,
        "n_total":            n_per_arm * 2,
        "baseline_mean":      baseline_mean,
        "mde_absolute":       round(mde_absolute, 4),
        "mde_relative_pct":   f"{mde_relative * 100:.1f}%",
        "alpha":              alpha,
        "power":              power,
        "estimated_days":     max(14, estimated_days),   # at least 2 weeks
        "daily_users_per_arm": daily_users_per_arm,
    }


# ── Statistical Analysis ──────────────────────────────────────────────────────
def analyse_experiment(
    control_data:   np.ndarray,
    treatment_data: np.ndarray,
    metric_name:    str = "AOV",
    alpha:          float = 0.05,
) -> dict:
    """
    Two-tailed Welch's t-test + effect size (Cohen's d).
    Returns full analysis dict.
    """
    ctrl_n,  ctrl_mean,  ctrl_std  = len(control_data),  control_data.mean(),  control_data.std()
    trt_n,   trt_mean,   trt_std   = len(treatment_data), treatment_data.mean(), treatment_data.std()

    t_stat, p_value = stats.ttest_ind(treatment_data, control_data, equal_var=False)

    # Cohen's d
    pooled_std = np.sqrt((ctrl_std**2 + trt_std**2) / 2)
    cohens_d   = (trt_mean - ctrl_mean) / pooled_std

    # Relative lift
    lift_pct = (trt_mean - ctrl_mean) / ctrl_mean * 100

    # 95% CI for the lift
    se_diff = np.sqrt(ctrl_std**2 / ctrl_n + trt_std**2 / trt_n)
    ci_lo   = (trt_mean - ctrl_mean) - 1.96 * se_diff
    ci_hi   = (trt_mean - ctrl_mean) + 1.96 * se_diff

    return {
        "metric":           metric_name,
        "control_mean":     round(float(ctrl_mean),  4),
        "treatment_mean":   round(float(trt_mean),   4),
        "absolute_lift":    round(float(trt_mean - ctrl_mean), 4),
        "relative_lift_pct": round(float(lift_pct),  2),
        "ci_95":            (round(float(ci_lo), 4), round(float(ci_hi), 4)),
        "t_stat":           round(float(t_stat), 4),
        "p_value":          round(float(p_value), 6),
        "significant":      bool(p_value < alpha),
        "cohens_d":         round(float(cohens_d), 4),
        "n_control":        int(ctrl_n),
        "n_treatment":      int(trt_n),
        "decision":         _decision(p_value, lift_pct, alpha),
    }


def _decision(p_value: float, lift_pct: float, alpha: float) -> str:
    if p_value < alpha and lift_pct > 0:
        return "✅ SHIP — statistically significant positive lift"
    elif p_value < alpha and lift_pct < 0:
        return "🚫 STOP — statistically significant negative impact"
    else:
        return "⏳ CONTINUE — not enough data yet"


# ── Guardrail Checker ─────────────────────────────────────────────────────────
def check_guardrails(
    metrics: dict[str, tuple[float, float]],   # {metric: (control_val, treatment_val)}
    config:  ABTestConfig | None = None,
) -> dict:
    """
    Check if any guardrail metric has been violated.
    Returns dict with status per metric and overall verdict.
    """
    config    = config or ABTestConfig()
    results   = {}
    stop_flag = False

    for metric, (ctrl_val, trt_val) in metrics.items():
        if metric not in config.guardrail_metrics:
            continue

        guard    = config.guardrail_metrics[metric]
        delta    = trt_val - ctrl_val
        violated = False

        if guard["direction"] == "increase":
            violated = delta > guard["threshold"]
        elif guard["direction"] == "decrease":
            violated = delta < -guard["threshold"]

        results[metric] = {
            "control":   round(ctrl_val, 6),
            "treatment": round(trt_val, 6),
            "delta":     round(delta, 6),
            "threshold": guard["threshold"],
            "direction": guard["direction"],
            "violated":  violated,
            "status":    "🚫 VIOLATED" if violated else "✅ OK",
        }
        if violated:
            stop_flag = True

    return {
        "guardrail_results": results,
        "stop_experiment":   stop_flag,
        "verdict":           "🚫 STOP — guardrail violated" if stop_flag else "✅ All guardrails healthy",
    }


# ── Offline → Online Metric Bridge ────────────────────────────────────────────
def project_business_impact(
    baseline_precision_at_k: float = 0.1485,   # measured from our test set
    model_precision_at_k:    float = 0.3818,   # measured from our test set
    baseline_ndcg_at_k:      float = 0.3086,   # measured from our test set
    model_ndcg_at_k:         float = 0.8764,   # measured from our test set
    k:                       int   = 8,
    avg_addon_price_inr:     float = 322.6,    # measured from raw item data
    incremental_fraction:    float = 0.20,     # industry: 20% of accepts are truly incremental
    baseline_aov:            float = 320.0,    # ₹ baseline AOV from dataset
    daily_orders:            int   = 200_000,
) -> dict:
    """
    Rigorous bottom-up translation of offline ML metrics to projected AOV lift.

    Two independent derivation paths are provided for cross-validation:

      Path 1 (Precision-based) — direct formula:
        accepted_model    = Precision@K × K
        accepted_baseline = Precision@K_baseline × K
        Δ_accepted        = (accepted_model − accepted_baseline)
        truly_incremental = Δ_accepted × incremental_fraction
        AOV_lift_inr      = truly_incremental × avg_addon_price
        AOV_lift_pct      = AOV_lift_inr / baseline_AOV

      Path 2 (NDCG-calibrated) — industry benchmark:
        10 pp NDCG lift ≈ 1.2 pp AOV lift (food-delivery benchmark)
    """
    # ── Path 1: Precision-based derivation ─────────────────────────────────
    accepted_baseline    = baseline_precision_at_k * k          # 1.19 items/order
    accepted_model       = model_precision_at_k    * k          # 3.05 items/order
    delta_accepted       = accepted_model - accepted_baseline   # 1.86 Δ items
    truly_incremental    = delta_accepted * incremental_fraction # 0.37 new items/order
    aov_lift_inr_prec    = truly_incremental * avg_addon_price_inr
    aov_lift_pct_prec    = aov_lift_inr_prec / baseline_aov * 100

    # ── Path 2: NDCG-calibrated benchmark ──────────────────────────────────
    ndcg_lift_pct        = (model_ndcg_at_k - baseline_ndcg_at_k) / baseline_ndcg_at_k * 100
    ndcg_to_aov_per_10pp = 1.2         # pp AOV lift per 10 pp NDCG lift (industry)
    aov_lift_pct_ndcg    = ndcg_lift_pct / 10.0 * ndcg_to_aov_per_10pp

    # ── Conservative estimate (lower of the two) ────────────────────────────
    aov_lift_pct_final   = min(aov_lift_pct_prec, aov_lift_pct_ndcg)
    aov_lift_inr_final   = baseline_aov * aov_lift_pct_final / 100

    # ── Revenue at scale ────────────────────────────────────────────────────
    daily_revenue_lift   = daily_orders * aov_lift_inr_final
    annual_revenue_lift  = daily_revenue_lift * 365

    # ── Acceptance rate lift ────────────────────────────────────────────────
    accept_lift_abs      = model_precision_at_k - baseline_precision_at_k
    accept_lift_pct      = accept_lift_abs / baseline_precision_at_k * 100

    return {
        "derivation_path_1_precision": {
            "formula":            "Δ_accepted × incremental_fraction × avg_addon_price / baseline_AOV",
            "accepted_baseline":  f"{accepted_baseline:.2f} items/order  (= {baseline_precision_at_k} × {k})",
            "accepted_model":     f"{accepted_model:.2f} items/order  (= {model_precision_at_k} × {k})",
            "delta_accepted":     f"{delta_accepted:.2f} Δ items",
            "truly_incremental":  f"{truly_incremental:.2f} new items/order  (× {incremental_fraction:.0%} factor)",
            "AOV_lift_INR":       f"₹{aov_lift_inr_prec:.1f}",
            "AOV_lift_pct":       f"+{aov_lift_pct_prec:.1f}%",
        },
        "derivation_path_2_ndcg": {
            "formula":            "NDCG_lift_pct / 10 × 1.2pp (industry calibration)",
            "NDCG_baseline":      f"{baseline_ndcg_at_k:.4f}",
            "NDCG_model":         f"{model_ndcg_at_k:.4f}",
            "NDCG_lift_pct":      f"+{ndcg_lift_pct:.1f}%",
            "AOV_lift_pct":       f"+{aov_lift_pct_ndcg:.1f}%",
        },
        "conservative_estimate": {
            "method":             "min(Precision-path, NDCG-path)",
            "AOV_lift_pct":       f"+{aov_lift_pct_final:.1f}%",
            "AOV_lift_INR":       f"₹{aov_lift_inr_final:.0f} per order",
            "addon_acceptance_lift_pct": f"+{accept_lift_pct:.1f}% (absolute +{accept_lift_abs:.4f})",
        },
        "revenue_at_scale": {
            "daily_orders":          f"{daily_orders:,}",
            "daily_revenue_lift":    f"₹{daily_revenue_lift:,.0f}",
            "annual_revenue_lift":   f"₹{annual_revenue_lift:,.0f}",
        },
        "assumptions": (
            f"incremental_fraction={incremental_fraction:.0%} "
            f"(industry: 15-25% of accepted recommendations are truly incremental); "
            f"avg_addon_price=₹{avg_addon_price_inr:.0f} (measured from raw data)"
        ),
    }


# ── Full Report Printer ───────────────────────────────────────────────────────
def print_ab_design_report() -> None:
    """Print the full A/B test design for submission."""
    config = ABTestConfig()

    print("\n" + "═" * 65)
    print("SmartCart v2.0 — A/B TEST DESIGN")
    print("═" * 65)
    print(f"\nExperiment: {config.name}")
    print(f"Description: {config.description}")
    print(f"\n{'─'*65}")
    print("TRAFFIC SPLIT")
    print(f"  Control   ({config.control_pct:.0%}): Top-8 globally popular items (baseline)")
    print(f"  Treatment ({config.treatment_pct:.0%}): SmartCart v2.0 personalised recommendations")

    print(f"\n{'─'*65}")
    print("SAMPLE SIZE (AOV as primary metric)")
    ss = required_sample_size(baseline_mean=320.0, mde_relative=0.02)
    for k, v in ss.items():
        print(f"  {k:<28}: {v}")

    print(f"\n{'─'*65}")
    print("PRIMARY METRIC")
    print(f"  {config.primary_metric.upper()}")

    print(f"\n{'─'*65}")
    print("SECONDARY METRICS")
    for m in config.secondary_metrics:
        print(f"  • {m}")

    print(f"\n{'─'*65}")
    print("GUARDRAIL METRICS (STOP if violated)")
    for metric, rule in config.guardrail_metrics.items():
        print(f"  • {metric}: must NOT {rule['direction']} by >{rule['threshold']:.0%}")

    print(f"\n{'─'*65}")
    print("DURATION")
    print(f"  Min: {config.min_days} days (captures full weekly cycle × 2)")
    print(f"  Max: {config.max_days} days")
    print(f"  Burn-in: First {config.burn_in_days} days excluded (novelty effect)")

    print(f"\n{'─'*65}")
    print("STATISTICAL PARAMETERS")
    print(f"  Significance level (α): {config.alpha}")
    print(f"  Statistical power:       {config.power}")
    print(f"  MDE (relative):         {config.mde:.0%}")

    print(f"\n{'─'*65}")
    print("PROJECTED BUSINESS IMPACT (from offline results)")
    impact = project_business_impact()
    for section, vals in impact.items():
        if section == "assumptions":
            continue
        print(f"\n  {section.upper().replace('_', ' ')}:")
        if isinstance(vals, dict):
            for key, v in vals.items():
                print(f"    {key:<40}: {v}")
        else:
            print(f"    {vals}")
    print(f"\n  ASSUMPTIONS: {impact['assumptions']}")
    print("\n" + "═" * 65)


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_ab_design_report()

    # Simulate statistical analysis with synthetic data
    rng = np.random.default_rng(42)
    print("\n\nSIMULATED ANALYSIS (synthetic data, 10K users per arm):")
    print("═" * 65)
    ctrl = rng.normal(320, 120, 10_000)
    trt  = rng.normal(326, 120, 10_000)   # +2% lift

    result = analyse_experiment(ctrl, trt, metric_name="AOV (₹)", alpha=0.05)
    for k, v in result.items():
        print(f"  {k:<25}: {v}")

    print("\nGUARDRAIL CHECK:")
    grd = check_guardrails({
        "cart_abandonment_rate":   (0.12, 0.118),   # improved (OK)
        "session_completion_rate": (0.68, 0.685),   # improved (OK)
        "p99_latency_ms":         (220.0, 245.0),   # increased but under threshold (OK)
    })
    print(f"  {grd['verdict']}")
    for m, r in grd["guardrail_results"].items():
        print(f"  {m}: {r['status']} (delta={r['delta']:+.4f}, threshold={r['threshold']})")
