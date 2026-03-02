"""
SmartCart v2.0 — LLM-Powered Recommendation Explainer (AI Edge)
Generates natural language explanations for recommendations.

Two modes:
  1. Template-based  → zero-latency, zero-cost, always works
  2. LLM-powered     → rich personalized explanations via OpenAI/Gemini
                       (requires OPENAI_API_KEY or GEMINI_API_KEY env var)

The AI Edge value:
  → Converts opaque ML scores into human-readable reasons
  → Drives ADD-TO-CART clicks (+12-18% CTR lift in industry studies)
  → Handles unstructured item descriptions for cold-start new items
"""

import os
import re
import random
import pandas as pd
from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

# Load item catalogue for rich descriptions
_ITEM_FEAT = pd.read_csv(PROCESSED_DIR / "item_features.csv").set_index("item_id")


# ── Template Library ──────────────────────────────────────────────────────────
TEMPLATES = {
    "high_cooc": [
        "{pct}% of customers who ordered this also added {item}.",
        "This pairs perfectly — {pct}% of similar orders include {item}.",
        "A top combo: {item} is ordered together {pct}% of the time.",
    ],
    "segment": [
        "Popular in your area — {item} is trending in {zone} right now.",
        "Customers like you in {zone} frequently add {item}.",
        "{item} is a bestseller for {tier} city customers.",
    ],
    "cart_complement": [
        "Your order looks complete with {item} — adds the perfect {category}.",
        "Missing a {category}? {item} is a crowd favourite.",
        "Complete your meal with {item} — pairs well with what's in your cart.",
    ],
    "seasonal": [
        "A great {season} pick — {item} is extra popular this season.",
        "Beat the {season} vibe with {item}!",
    ],
    "popular": [
        "{item} is one of our most ordered items today.",
        "Don't miss out — {item} is trending right now.",
        "Bestseller alert: {item} is loved by thousands.",
    ],
}


# ── Template-based explainer (zero latency) ───────────────────────────────────
def explain_template(
    item_id:     int,
    item_name:   str | None = None,
    cooc_score:  float = 0.0,
    segment:     dict  | None = None,
    category:    str   | None = None,
    season:      str   | None = None,
    strategy:    str   = "popular",
) -> str:
    """
    Generate a natural language explanation for a recommendation
    using rule-based templates. Zero latency, always available.
    """
    segment  = segment  or {}
    item_label = item_name or f"Item #{item_id}"

    pct  = int(min(cooc_score * 100, 99)) if cooc_score > 0 else random.randint(60, 90)
    zone = segment.get("zone_type", "your area")
    tier_map = {1: "metro", 2: "city", 3: "your"}
    tier_label = tier_map.get(segment.get("tier", 2), "your")

    if cooc_score > 0.3 and "high_cooc" in TEMPLATES:
        pool = TEMPLATES["high_cooc"]
        tmpl = random.choice(pool)
        return tmpl.format(pct=pct, item=item_label, zone=zone, tier=tier_label)

    if strategy == "segment_heuristic":
        pool = TEMPLATES["segment"]
        tmpl = random.choice(pool)
        return tmpl.format(item=item_label, zone=zone, tier=tier_label)

    if category:
        pool = TEMPLATES["cart_complement"]
        tmpl = random.choice(pool)
        return tmpl.format(item=item_label, category=category)

    if season:
        pool = TEMPLATES["seasonal"]
        tmpl = random.choice(pool)
        return tmpl.format(item=item_label, season=season)

    pool = TEMPLATES["popular"]
    return random.choice(pool).format(item=item_label)


# ── LLM-powered explainer (OpenAI / Gemini) ───────────────────────────────────
def explain_llm(
    cart_item_names:     list[str],
    recommended_name:    str,
    cooc_score:          float,
    segment:             dict,
) -> str:
    """
    Use LLM to generate personalised explanation.
    Falls back gracefully to template if no API key set.

    Set OPENAI_API_KEY or GEMINI_API_KEY in environment.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if openai_key:
        return _explain_openai(cart_item_names, recommended_name, cooc_score, segment, openai_key)
    elif gemini_key:
        return _explain_gemini(cart_item_names, recommended_name, cooc_score, segment, gemini_key)
    else:
        # Graceful fallback — always works at zero cost
        return explain_template(
            item_id    = 0,
            item_name  = recommended_name,
            cooc_score = cooc_score,
            segment    = segment,
        )


def _explain_openai(
    cart_items: list[str],
    rec_item: str,
    cooc_score: float,
    segment: dict,
    api_key: str,
) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        prompt = (
            f"A user in a {segment.get('zone_type','city')} area "
            f"({segment.get('tier',2)}-tier city) "
            f"has: {', '.join(cart_items)} in their cart.\n"
            f"We recommend: {rec_item}.\n"
            f"Co-occurrence support: {cooc_score:.0%}\n\n"
            f"Write ONE short recommendation reason (max 12 words). "
            f"Be specific and conversational. "
            f"Example: '87% of Biryani orders also include Raita.'"
        )

        resp = client.chat.completions.create(
            model    = "gpt-3.5-turbo",
            messages = [{"role": "user", "content": prompt}],
            max_tokens   = 40,
            temperature  = 0.4,
        )
        return resp.choices[0].message.content.strip().strip('"')

    except Exception as e:
        # Never crash the recommendation pipeline
        return explain_template(0, rec_item, cooc_score, segment)


def _explain_gemini(
    cart_items: list[str],
    rec_item: str,
    cooc_score: float,
    segment: dict,
    api_key: str,
) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-pro")

        prompt = (
            f"Cart: {', '.join(cart_items)}. "
            f"Recommended: {rec_item}. "
            f"Support: {cooc_score:.0%}. "
            f"One sentence explanation, max 12 words."
        )
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception:
        return explain_template(0, rec_item, cooc_score, segment)


# ── Batch explainer for API response ─────────────────────────────────────────
def enrich_recommendations(
    recommendations: list[dict],
    cart_items:      list[int],
    context:         dict,
    use_llm:         bool = False,
) -> list[dict]:
    """
    Add explanation field to each recommendation dict.

    Parameters
    ----------
    recommendations : output of inference.recommend()
    cart_items      : original cart item ids
    context         : request context
    use_llm         : whether to call LLM API (default False)

    Returns
    -------
    Same list with 'explanation' key added to each item
    """
    enriched = []
    for rec in recommendations:
        item_id   = rec["item_id"]
        cooc      = rec.get("cooc_score", rec.get("score", 0.0))
        item_name = f"Item #{item_id}"  # replace with real name lookup if available

        if use_llm:
            cart_names = [f"Item #{cid}" for cid in cart_items]
            explanation = explain_llm(cart_names, item_name, cooc, context)
        else:
            explanation = explain_template(
                item_id    = item_id,
                item_name  = item_name,
                cooc_score = cooc,
                segment    = context,
                season     = context.get("season"),
            )

        enriched.append({**rec, "explanation": explanation})
    return enriched


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("═" * 60)
    print("SmartCart v2.0 — LLM Explainer Demo")
    print("═" * 60)

    sample_recs = [
        {"item_id": 936,  "score": 0.92, "cooc_score": 0.74, "strategy": "personalised_model"},
        {"item_id": 1202, "score": 0.81, "cooc_score": 0.51, "strategy": "segment_heuristic"},
        {"item_id": 715,  "score": 0.65, "cooc_score": 0.12, "strategy": "global_popularity"},
    ]

    context = {
        "tier": 2, "season": "Summer", "zone_type": "CBD",
        "hour": 13,
    }

    enriched = enrich_recommendations(sample_recs, cart_items=[923], context=context)

    print("\nTemplate-based explanations (zero-cost, zero-latency):")
    for r in enriched:
        print(f"  Item #{r['item_id']} (score={r['score']:.2f})")
        print(f"    → \"{r['explanation']}\"")

    print("\nTo enable LLM-powered explanations:")
    print("  export OPENAI_API_KEY=sk-...")
    print("  enrich_recommendations(..., use_llm=True)")
    print("\n✅ AI Edge module ready.")
