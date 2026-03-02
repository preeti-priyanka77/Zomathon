**Production-Grade Context-Aware Add-On Recommendation System**

- **Project Type:** Real-time Learning-to-Rank Recommendation System
- **Domain:** Food Delivery / Dynamic Cart Intelligence
- **Version:** v2.0 (Enhanced Multi-Segment Architecture)

---
## 1. Executive Summary

SmartCart v2.0 is a production-ready, multi-segment, low-latency add-on recommendation system designed for dynamic shopping carts.

Unlike simple co-occurrence recommenders, v2.0 incorporates:

- City-tier purchasing power modeling
- State-level cuisine biases
- User demographics
- Behavioral history (RFM)
- Geospatial delivery zones
- Seasonal + temporal variations
- Contextual cart state modeling

The system operates in < 300ms and is architected for horizontal scalability.

**Projected business impact:**

- +2–4% AOV lift
- +5–8% add-on acceptance improvement

---
## 2. Problem Definition

**Given:**

- User `u`
- Dynamic cart `c`
- Context `x`
- Restaurant `r`

**Predict:**
Top-K add-on items ranked by probability of acceptance.

**Formally:**

```
f(u, c, i, x) → P(accept)
Rank items by predicted probability.
```

---

## 3. Dataset v2.0 Architecture

### 3.1 Geographic Hierarchy

**City Tiers:**

- **Tier 1:** High purchasing power metros
- **Tier 2:** Large balanced cities
- **Tier 3:** Emerging price-sensitive markets

**Impact:**

- 2–4x AOV variance
- Different cart complexity patterns
- Segment-specific co-occurrence rules

### 3.2 State-Level Cuisine Bias

Each city is mapped to:

- State
- Regional cuisine dominance
- Vegetarian percentage bias
- Cultural meal structure

This ensures co-occurrence is contextual, not global.

**Example:**

```
P(drink | main, tier=1, CBD, summer) ≠ P(drink | main, tier=3, residential, winter)
```

### 3.3 User Demographics

**Age Groups:**

- Gen_Z
- Millennial
- Gen_X
- Boomer

**Behavioral differences modeled:**

- Health consciousness
- Experimentation rate
- Repeat behavior
- AOV range
- Premium item preference

### 3.4 Behavioral History (RFM)

For each user:

**Recency:**

- `days_since_last_order`

**Frequency:**

- `orders_last_7_days`
- `orders_last_30_days`

**Monetary:**

- `avg_order_value`
- `max_order_value`

**Loyalty:**

- `repeat_restaurant_pct`
- `cuisine_consistency`
- `category_switch_rate`

**Cohorts:**

- New (0–7 days)
- Growing
- Established
- Loyal

### 3.5 Geospatial Context

**Delivery Zones:**

- CBD
- Residential
- Student
- Mixed
- Premium

**Distance modeling:**

- 0–1 km → snack orders
- 3–5 km → premium planned meals
- 5+ km → celebration-level AOV

### 3.6 Seasonal + Temporal Modeling

**Seasonal modifiers:**

- Summer → +30% drinks
- Winter → +30% butter/meat
- Festivals → +35–50% AOV

**Temporal:**

- Breakfast
- Lunch
- Dinner
- Late night
- Weekend vs weekday

---

## 4. Feature Engineering (36+ Features)

**Feature groups:**

### User Features (Offline)

- `age_group`
- `price_sensitivity`
- `vegetarian_pct`
- `avg_order_value`
- `user_tenure_days`
- `loyalty_metrics`

### Cart Features (Online)

- `cart_size`
- `cart_value`
- `has_main`
- `has_drink`
- `avg_item_price`
- `premium_item_count`

### Item Features (Offline)

- `category`
- `price_tier`
- `addon_rate`
- `popularity_rank`
- `is_vegetarian`
- `spice_level`

### Context Features (Online)

- `city_tier`
- `state`
- `delivery_zone`
- `distance`
- `season`
- `time_bucket`
- `is_holiday`

### Cross Features

- `city_tier × item_price`
- `user_avg_cart_value / item_price`
- `season × category`
- `zone × meal_type`

---

## 5. Modeling Architecture

### 5.1 Two-Stage Pipeline

```
Cart Event
   ↓
Candidate Generation (Heuristic + Segment-Aware)
   ↓
Feature Retrieval (Feature Store)
   ↓
GBDT Ranking Model
   ↓
Post-Processing (Diversity + Budget Guardrails)
   ↓
Top-K Recommendations
```

### 5.2 Candidate Generation

**Inputs:**

- Co-occurrence matrix segmented by:
  - `city_tier`
  - `age_group`
  - `season`
  - `zone`

**Reduces:** ~200 menu items → ~50 candidates

**Latency target:** < 50ms

### 5.3 Ranking Model

**Model:** LightGBM / XGBoost

**Why:**

- Handles tabular + categorical well
- Fast inference
- Interpretable
- Excellent for Learning-to-Rank

**Inference latency:** < 100ms for 50 candidates

---

## 6. Evaluation Strategy

**Split:**

- Temporal split (train on earlier weeks, test on latest)

**Primary metrics:**

- NDCG@8
- Precision@8

**Secondary:**

- AUC

**Segment evaluation:**

- Performance per city tier
- Performance per age group
- Performance per zone type

---

## 7. Business Experimentation

### A/B Test Design

**Control:**

- Popular items baseline

**Treatment:**

- SmartCart v2 recommendations

**Primary KPIs:**

- AOV
- Add-on acceptance rate

**Guardrails:**

- Cart abandonment
- Completion rate
- User satisfaction

---

## 8. Latency Budget

| Component       | Target      |
| --------------- | ----------- |
| Candidate Gen   | 30–50ms     |
| Feature Fetch   | 40–60ms     |
| Ranking         | 70–100ms    |
| Post-Processing | 20–30ms     |
| Network         | 50ms        |
| **Total**       | **< 300ms** |

---

## 9. Production Considerations

### Scalability

- Stateless ranking service
- Redis-based feature store
- Horizontal scaling

### Monitoring

- Drift detection
- Segment-wise acceptance rate
- Latency P50 / P95 / P99

### Fallback

- Cached popular items per restaurant + time bucket

---

## 10. Data Volume Scaling

**Projected:**

- 14 cities
- 50K–100K users
- 100K+ orders
- 36+ features
- Multi-segment co-occurrence matrices

**Architecture supports:**

- Monthly retraining
- Feature recalculation
- Shadow deployment