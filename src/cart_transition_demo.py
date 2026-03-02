"""
SmartCart v2.0 — Cart Transition Demo
======================================
Demonstrates how recommendations evolve dynamically as items are added to
the cart, modelling the core challenge stated in the problem:

  "North Indian Main → salan/side → drink → dessert"

Each step shows the full Top-8 ranked list and highlights which categories
shift in and out as the cart becomes more complete.

Run:
    python src/cart_transition_demo.py
"""

import sys
from pathlib import Path

# Make src/ importable from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference import recommend  # noqa: E402 (import after path fix)

import pandas as pd

# ── Item name lookup ──────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parent.parent
_item_meta = (
    pd.read_csv(_ROOT / "data" / "raw" / "order_items_v2_full.csv",
                usecols=["item_id", "item_name", "category", "price"])
    .drop_duplicates("item_id")
    .set_index("item_id")
)

def _name(item_id: int) -> str:
    """Return a human-readable label for an item_id."""
    if item_id in _item_meta.index:
        row = _item_meta.loc[item_id]
        return f"{row['item_name']} ({row['category']}, ₹{row['price']:.0f})"
    return f"item_{item_id}"


# ── Scenario definition ───────────────────────────────────────────────────────
#   User 10 is a warm Tier-1 user (10+ orders, avg AOV ₹630)
#   Ordering from a North Indian / Biryani restaurant at 1 PM on a weekday.
USER_ID     = 10
BASE_CTX    = dict(
    tier         = 1,
    season       = "Summer",
    zone_type    = "CBD",
    hour         = 13,
    day_of_week  = 2,
    month        = 5,
    distance_km  = 3.5,
    delivery_fee = 30.0,
)

# Real item IDs present in the (Tier-1, Summer, CBD) FP-Growth segment
# These represent a typical North Indian meal progression:
#   main course (dal makhani / butter chicken) → salan/raita → cold drink
NORTH_MAIN  = 693   # North_Indian_main_12  (curry / butter chicken)
NORTH_SIDE  = 682   # North_Indian_side_1   (salan / raita / naan)
NORTH_DRINK = 684   # North_Indian_drink_3  (lassi / soft-drink)

STEPS = [
    {
        "label":      "Step 1 — Cart: [North_Indian_main (Curry/Biryani)]",
        "cart":       [NORTH_MAIN],
        "ctx_extra":  dict(has_main=1, has_side=0, has_drink=0, has_dessert=0),
        "narration":  "User adds a North Indian main. System should push sides (salan/raita) and drinks.",
    },
    {
        "label":      "Step 2 — Cart: [Main, Side/Salan]",
        "cart":       [NORTH_MAIN, NORTH_SIDE],
        "ctx_extra":  dict(has_main=1, has_side=1, has_drink=0, has_dessert=0),
        "narration":  "Salan added. System should now prominently surface drinks and start showing desserts.",
    },
    {
        "label":      "Step 3 — Cart: [Main, Side, Drink]",
        "cart":       [NORTH_MAIN, NORTH_SIDE, NORTH_DRINK],
        "ctx_extra":  dict(has_main=1, has_side=1, has_drink=1, has_dessert=0),
        "narration":  "Drink added. Cart is near-complete — desserts and complementary items should dominate.",
    },
]


# ── Runner ────────────────────────────────────────────────────────────────────
def _divider(char: str = "═", width: int = 72) -> str:
    return char * width


def run_demo(k: int = 8) -> None:
    print(_divider())
    print("  SmartCart v2.0 — Cart Transition Demo")
    print("  North Indian Main → Salan/Side → Drink → Dessert")
    print(_divider())
    print(f"\n  User: {USER_ID}  |  Context: Tier-1, CBD, 1 PM, Summer\n")

    prev_ids: set[int] = set()

    for step in STEPS:
        ctx = {**BASE_CTX, **step["ctx_extra"]}

        recs = recommend(USER_ID, step["cart"], ctx, k=k)

        print(_divider("-"))
        print(f"  {step['label']}")
        print(f"  Cart contents:")
        for cid in step["cart"]:
            print(f"    • {_name(cid)}")
        print(f"\n  💡 {step['narration']}")
        print()

        cur_ids = {r["item_id"] for r in recs}
        new_ids = cur_ids - prev_ids
        gone_ids = prev_ids - cur_ids

        print(f"  {'#':<4} {'Item':<40} {'Category':<12} {'Score':>8}  {'Change'}")
        print(f"  {'-'*75}")
        for rank, rec in enumerate(recs, 1):
            iid      = rec["item_id"]
            name     = _item_meta.loc[iid, "item_name"] if iid in _item_meta.index else f"item_{iid}"
            cat      = _item_meta.loc[iid, "category"]  if iid in _item_meta.index else "?"
            score    = rec.get("score", 0.0)
            change   = "🆕 NEW" if iid in new_ids else ("  ↑ ↑" if rank <= 3 else "")
            print(f"  {rank:<4} {name:<40} {cat:<12} {score:>8.4f}  {change}")

        if gone_ids and prev_ids:
            removed_names = [
                (_item_meta.loc[i, "item_name"] if i in _item_meta.index else f"item_{i}")
                for i in gone_ids
            ]
            print(f"\n  ⬇  Dropped from list: {', '.join(removed_names)}")

        # Category distribution summary
        cats = [_item_meta.loc[r["item_id"], "category"]
                if r["item_id"] in _item_meta.index else "?" for r in recs]
        from collections import Counter
        cat_counts = Counter(cats)
        print(f"\n  Category mix in Top-{k}: " +
              "  |  ".join(f"{c}: {n}" for c, n in sorted(cat_counts.items())))
        print()

        prev_ids = cur_ids

    print(_divider())
    print("  Demo complete.")
    print(_divider())


if __name__ == "__main__":
    run_demo()
