#!/usr/bin/env python3
"""
SmartCart v2.0 - Synthetic Data Generation Pipeline

Generates realistic food delivery order data with:
- User behavior (loyalty, preferences, repeat patterns)
- Restaurant & item attributes (pricing, cuisine, tier)
- Order context (distance, time, season)
- Economic justification patterns (delivery fee impacts)

Output files:
  - orders_v2_full.csv
  - order_items_v2_full.csv
"""

import pandas as pd
import random
from datetime import datetime
from faker import Faker
from collections import defaultdict
import time
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIG
# ============================================================================
SEED = 42
NUM_USERS = 50000
NUM_ORDERS = 200000

RESTAURANTS_PER_CITY = 10
ITEMS_PER_RESTAURANT = 20

CITIES = {
    "Delhi": {"tier": 1},
    "Mumbai": {"tier": 1},
    "Bangalore": {"tier": 1},
    "Hyderabad": {"tier": 1},
    "Pune": {"tier": 2},
    "Kolkata": {"tier": 2},
    "Chennai": {"tier": 2},
    "Ahmedabad": {"tier": 2},
    "Jaipur": {"tier": 2},
    "Lucknow": {"tier": 3},
    "Coimbatore": {"tier": 3},
    "Indore": {"tier": 3},
    "Bhopal": {"tier": 3},
    "Chandigarh": {"tier": 3},
}

ZONE_TYPES = ["CBD", "Residential", "Student"]
CUISINES = ["North_Indian", "South_Indian", "Biryani", "Chinese", "Veg", "Fast_Food"]

# Item attribute mappings
SPICE_LEVELS = ["mild", "medium", "hot"]
PRICE_TIERS = ["budget", "standard", "premium"]

# Vegetarian bias by cuisine
VEGETARIAN_BIAS = {
    "North_Indian": 0.4,
    "South_Indian": 0.35,
    "Biryani": 0.1,
    "Chinese": 0.3,
    "Veg": 0.95,
    "Fast_Food": 0.2,
}

# Spice level distribution by cuisine
SPICE_DISTRIBUTION = {
    "North_Indian": [0.2, 0.5, 0.3],  # [mild, medium, hot]
    "South_Indian": [0.1, 0.4, 0.5],
    "Biryani": [0.1, 0.3, 0.6],
    "Chinese": [0.3, 0.5, 0.2],
    "Veg": [0.5, 0.4, 0.1],
    "Fast_Food": [0.6, 0.3, 0.1],
}

# Addon rate by category
ADDON_RATE_BY_CATEGORY = {
    "main": 0.05,      # Mains are rarely add-ons
    "side": 0.35,      # Sides often paired as add-ons
    "drink": 0.50,     # Drinks frequently added
    "dessert": 0.45,   # Desserts often added post-selection
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def clamp(p, low=0.05, high=0.9):
    """Clamp probability to valid range."""
    return max(low, min(high, p))


def get_season(month):
    """Map month to season (Indian climate)."""
    if month in [3, 4, 5]:
        return "Summer"
    if month in [6, 7, 8, 9]:
        return "Monsoon"
    return "Winter"


def adjust_price(price, tier, price_band):
    """Adjust item price based on city tier and restaurant price band."""
    if tier == 1:
        multiplier = random.uniform(1.1, 1.3)
    elif tier == 2:
        multiplier = random.uniform(0.9, 1.1)
    else:
        multiplier = random.uniform(0.7, 0.9)

    if price_band == "premium":
        multiplier += 0.15

    return int(round(price * multiplier / 5) * 5)


# ============================================================================
# DATA GENERATION
# ============================================================================
def generate_dataset(num_users, num_orders, output_dir):
    """
    Generate synthetic food delivery dataset.

    Args:
        num_users: Number of unique users to generate
        num_orders: Number of orders to generate
        output_dir: Directory to save CSV files
    """
    logger.info(f"Starting data generation: {num_users} users, {num_orders} orders")
    start_time = time.time()

    # Set seeds
    random.seed(SEED)
    fake = Faker("en_IN")
    Faker.seed(SEED)

    # ========================================================================
    # 1. GENERATE USERS
    # ========================================================================
    logger.info(f"Generating {num_users} users...")
    users = {}

    for uid in range(1, num_users + 1):
        is_member = random.random() < 0.35  # 35% subscription penetration

        base_loyalty = random.uniform(0.3, 0.8)
        if is_member:
            base_loyalty += 0.1
            base_loyalty = min(base_loyalty, 0.95)

        users[uid] = {
            "birth_year": random.randint(1965, 2006),
            "preferred_cuisine": random.choice(CUISINES),
            "loyalty_score": base_loyalty,
            "recent_restaurants": [],
            "user_city": random.choice(list(CITIES.keys())),
            "is_member": is_member,
        }

    # ========================================================================
    # 2. GENERATE RESTAURANTS & ITEMS
    # ========================================================================
    logger.info(f"Generating restaurants and items...")
    restaurants = {}
    global_item_id = 1

    for city in CITIES:
        restaurants[city] = []
        for r in range(RESTAURANTS_PER_CITY):
            cuisine = random.choice(CUISINES)
            price_band = random.choice(["budget", "mid", "premium"])

            items = []
            for i in range(ITEMS_PER_RESTAURANT):
                category = random.choice(["main", "side", "drink", "dessert"])
                base_price = random.randint(120, 500)
                
                # ============================================================
                # NEW: Item Attributes for SmartCart v2.0
                # ============================================================
                # Vegetarian status (biased by cuisine)
                is_vegetarian = (
                    category == "dessert" or 
                    random.random() < VEGETARIAN_BIAS.get(cuisine, 0.3)
                )
                
                # Spice level (sampled from cuisine distribution)
                spice_dist = SPICE_DISTRIBUTION.get(cuisine, [0.33, 0.33, 0.34])
                spice_level = random.choices(SPICE_LEVELS, weights=spice_dist, k=1)[0]
                
                # Price tier (based on base price)
                if base_price < 250:
                    item_price_tier = "budget"
                elif base_price < 400:
                    item_price_tier = "standard"
                else:
                    item_price_tier = "premium"
                
                # Addon rate (biased by category, with some random variance)
                base_addon_rate = ADDON_RATE_BY_CATEGORY.get(category, 0.3)
                addon_rate = min(0.95, max(0.02, base_addon_rate + random.uniform(-0.1, 0.15)))

                items.append(
                    {
                        "item_id": global_item_id,
                        "name": f"{cuisine}_{category}_{i}",
                        "category": category,
                        "base_price": base_price,
                        "is_vegetarian": is_vegetarian,
                        "spice_level": spice_level,
                        "price_tier": item_price_tier,
                        "addon_rate": round(addon_rate, 3),
                    }
                )
                global_item_id += 1

            top_sellers = random.sample(items, 3)

            restaurants[city].append(
                {
                    "restaurant_id": f"{city}_R{r}",
                    "cuisine": cuisine,
                    "price_band": price_band,
                    "items": items,
                    "top_sellers": top_sellers,
                }
            )

    logger.info(f"Generated {global_item_id - 1} items across all restaurants")

    # ========================================================================
    # 3. GENERATE ORDERS & ITEMS
    # ========================================================================
    logger.info(f"Generating {num_orders} orders...")
    orders_output = []
    items_output = []
    user_stats = defaultdict(lambda: {"orders": 0, "spent": 0})

    start_date = datetime(2019, 1, 1)
    end_date = datetime(2025, 1, 1)

    for order_num in range(num_orders):
        if (order_num + 1) % 50000 == 0:
            logger.info(f"  Generated {order_num + 1}/{num_orders} orders...")

        order_id = random.randint(100000, 999999)
        user_id = random.randint(1, num_users)
        user = users[user_id]

        city = user["user_city"]
        tier = CITIES[city]["tier"]
        available = restaurants[city]

        # ====================================================================
        # RESTAURANT SELECTION (Loyalty + Cuisine Preference)
        # ====================================================================
        loyalty_prob = 0.5 + (user["loyalty_score"] * 0.5)

        if user["recent_restaurants"] and random.random() < loyalty_prob:
            loyal_candidates = [
                r for r in available if r["restaurant_id"] in user["recent_restaurants"]
            ]
            restaurant = (
                random.choice(loyal_candidates) if loyal_candidates else random.choice(available)
            )
        else:
            preferred = [r for r in available if r["cuisine"] == user["preferred_cuisine"]]
            restaurant = (
                random.choice(preferred)
                if preferred and random.random() < 0.6
                else random.choice(available)
            )

        user["recent_restaurants"].append(restaurant["restaurant_id"])
        user["recent_restaurants"] = user["recent_restaurants"][-5:]

        # ====================================================================
        # CONTEXT (Time, Season, Distance)
        # ====================================================================
        order_date = fake.date_between(start_date=start_date, end_date=end_date)
        season = get_season(order_date.month)
        user_age = order_date.year - user["birth_year"]
        zone = random.choice(ZONE_TYPES)
        distance_km = random.uniform(0.5, 6.0)

        # ====================================================================
        # DELIVERY FEE MODEL
        # ====================================================================
        if distance_km <= 2:
            base_fee = 20
        elif distance_km <= 4:
            base_fee = 40
        else:
            base_fee = 80

        if user["is_member"]:
            if distance_km <= 3:
                delivery_fee = 0
            else:
                delivery_fee = int(base_fee * 0.5)
        else:
            delivery_fee = base_fee

        # ====================================================================
        # CART COMPOSITION (Probabilistic Item Selection)
        # ====================================================================
        p_main = 0.75
        p_side = 0.5
        p_drink = 0.6
        p_dessert = 0.20

        # Age effect
        if user_age < 25:
            p_drink += 0.10
        elif user_age > 50:
            p_dessert -= 0.10

        # Tier effect
        if tier == 1:
            p_dessert += 0.10
        if tier == 3:
            p_dessert -= 0.10

        # Season effect
        if season == "Summer":
            p_drink += 0.15
        elif season == "Winter":
            p_drink -= 0.08

        # Distance gradient
        if 2 < distance_km <= 4:
            p_side += 0.10
            p_dessert += 0.05

        if distance_km > 4:
            p_side += 0.20
            p_dessert += 0.12

        # Noise
        p_main += random.uniform(-0.05, 0.05)
        p_side += random.uniform(-0.05, 0.05)
        p_drink += random.uniform(-0.05, 0.05)
        p_dessert += random.uniform(-0.05, 0.05)

        # Clamp probabilities
        p_main = clamp(p_main)
        p_side = clamp(p_side)
        p_drink = clamp(p_drink)
        p_dessert = clamp(p_dessert)

        cart = []

        def add_item(item):
            """Helper to add item to cart with adjusted price."""
            price = adjust_price(item["base_price"], tier, restaurant["price_band"])
            cart.append(
                {
                    "order_id": order_id,
                    "item_id": item["item_id"],
                    "item_name": item["name"],
                    "category": item["category"],
                    "price": price,
                    "restaurant_id": restaurant["restaurant_id"],
                    "cuisine": restaurant["cuisine"],
                    "is_vegetarian": item["is_vegetarian"],
                    "spice_level": item["spice_level"],
                    "price_tier": item["price_tier"],
                    "addon_rate": item["addon_rate"],
                }
            )

        # Main course (always included with high probability)
        if random.random() < p_main:
            item = (
                random.choice(restaurant["top_sellers"])
                if random.random() < 0.4
                else random.choice(restaurant["items"])
            )
            add_item(item)

            # Pair injection (side + drink with main)
            if random.random() < 0.6:
                for pair_cat in ["side", "drink"]:
                    candidates = [i for i in restaurant["items"] if i["category"] == pair_cat]
                    if candidates:
                        add_item(random.choice(candidates))

        # Dessert
        if random.random() < p_dessert:
            desserts = [i for i in restaurant["items"] if i["category"] == "dessert"]
            if desserts:
                add_item(random.choice(desserts))

        # Fallback: ensure cart is not empty
        if not cart:
            add_item(random.choice(restaurant["items"]))

        subtotal = sum(i["price"] for i in cart)

        # ====================================================================
        # CART COMPOSITION INDICATORS
        # ====================================================================
        cart_categories = {item["category"] for item in cart}
        has_main = "main" in cart_categories
        has_side = "side" in cart_categories
        has_drink = "drink" in cart_categories
        has_dessert = "dessert" in cart_categories

        # ====================================================================
        # ECONOMIC JUSTIFICATION BEHAVIOR
        # ====================================================================
        # High delivery fee incentivizes cart expansion
        if delivery_fee >= 60 and subtotal < 600:
            if random.random() < 0.65:
                extra_item = random.choice(restaurant["items"])
                add_item(extra_item)
                subtotal = sum(i["price"] for i in cart)

        # Update user statistics
        user_stats[user_id]["orders"] += 1
        user_stats[user_id]["spent"] += subtotal
        avg_user_value = user_stats[user_id]["spent"] / user_stats[user_id]["orders"]

        # Record the order
        orders_output.append(
            {
                "order_id": order_id,
                "user_id": user_id,
                "city": city,
                "tier": tier,
                "restaurant_id": restaurant["restaurant_id"],
                "cuisine": restaurant["cuisine"],
                "zone_type": zone,
                "distance_km": round(distance_km, 2),
                "order_date": order_date,
                "season": season,
                "user_age": user_age,
                "is_member": user["is_member"],
                "delivery_fee": delivery_fee,
                "subtotal": subtotal,
                "total_order_value": subtotal + delivery_fee,
                "avg_user_value": round(avg_user_value, 2),
                "cart_size": len(cart),
                "has_main": has_main,
                "has_side": has_side,
                "has_drink": has_drink,
                "has_dessert": has_dessert,
            }
        )

        items_output.extend(cart)

    # ========================================================================
    # 4. SAVE TO CSV
    # ========================================================================
    logger.info("Writing datasets to CSV...")
    output_dir.mkdir(parents=True, exist_ok=True)

    orders_df = pd.DataFrame(orders_output)
    items_df = pd.DataFrame(items_output)

    orders_path = output_dir / "orders_v2_full.csv"
    items_path = output_dir / "order_items_v2_full.csv"

    orders_df.to_csv(orders_path, index=False)
    items_df.to_csv(items_path, index=False)

    elapsed = time.time() - start_time

    logger.info(f"✓ Successfully generated data in {elapsed:.2f} seconds")
    logger.info(f"  Orders: {orders_path} ({len(orders_df):,} rows)")
    logger.info(f"  Items:  {items_path} ({len(items_df):,} rows)")

    return orders_df, items_df


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic food delivery order dataset"
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=NUM_USERS,
        help=f"Number of users (default: {NUM_USERS})",
    )
    parser.add_argument(
        "--num-orders",
        type=int,
        default=NUM_ORDERS,
        help=f"Number of orders (default: {NUM_ORDERS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../data/raw"),
        help="Output directory for CSV files",
    )

    args = parser.parse_args()

    generate_dataset(args.num_users, args.num_orders, args.output_dir)


if __name__ == "__main__":
    main()
