"""Configuration and constants for SmartCart v2.0"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data generation config
DATA_GENERATION_CONFIG = {
    "seed": 42,
    "num_users": 50000,
    "num_orders": 200000,
    "restaurants_per_city": 10,
    "items_per_restaurant": 20,
}

# Cities (tier-based segmentation)
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

# Categories and cuisines
CATEGORIES = ["main", "side", "drink", "dessert"]
CUISINES = ["North_Indian", "South_Indian", "Biryani", "Chinese", "Veg", "Fast_Food"]
ZONE_TYPES = ["CBD", "Residential", "Student"]
PRICE_BANDS = ["budget", "mid", "premium"]

# Latency targets (ms) for SmartCart v2.0
LATENCY_BUDGET = {
    "candidate_generation": 50,  # P95
    "feature_fetch": 60,          # P95
    "ranking_inference": 100,     # P95
    "post_processing": 30,        # P95
    "network": 50,                # estimate
    "total": 300,                 # budget
}

# Feature engineering constants
USER_FEATURE_COLUMNS = [
    "total_orders",
    "avg_order_value",
    "days_since_last_order",
    "user_tenure_days",
]

ITEM_FEATURE_COLUMNS = [
    "item_order_count",
    "avg_price",
    "popularity_rank",
]

# Probabilities for order composition
DEFAULT_PROBABILITIES = {
    "main": 0.75,
    "side": 0.50,
    "drink": 0.60,
    "dessert": 0.20,
}

# Economic parameters
DELIVERY_FEE_STRUCTURE = {
    "near": {"distance": 2, "fee": 20},
    "medium": {"distance": 4, "fee": 40},
    "far": {"distance": 10, "fee": 80},
}

MEMBERSHIP_BENEFITS = {
    "free_delivery_within": 3,  # km
    "discount_factor": 0.5,      # 50% delivery fee beyond
}

# Seasonal effects
SEASONS = {
    "Summer": [3, 4, 5],
    "Monsoon": [6, 7, 8, 9],
    "Winter": [10, 11, 12, 1, 2],
}
