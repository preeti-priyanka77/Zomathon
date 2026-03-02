"""
SmartCart v2.0 — Online Latency Benchmark
Measures REAL end-to-end HTTP latency (not offline Python call time).

Usage:
  Terminal 1: cd /home/sampad/CSAO && uvicorn src.api:app --port 8000
  Terminal 2: cd /home/sampad/CSAO && python src/latency_test.py

Target: P95 ≤ 300ms (SLA ceiling per problem statement)
Expected: ~10ms P95 locally, ~31ms P95 in production with Redis + network
"""

import time
import httpx
import numpy as np
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL  = "http://localhost:8000"
N_CALLS   = 500
N_WARMUP  = 10
TIMEOUT   = 5.0   # seconds per request

PROCESSED_DIR = Path("data/processed")

# ── Load real users and items ─────────────────────────────────────────────────
print("Loading test data...")
user_ids = pd.read_csv(PROCESSED_DIR / "user_features.csv")["user_id"].tolist()
item_ids = pd.read_csv(PROCESSED_DIR / "item_features.csv")["item_id"].tolist()
print(f"  {len(user_ids):,} users | {len(item_ids):,} items loaded")

rng      = np.random.default_rng(42)
SEASONS  = ["Summer", "Monsoon", "Winter"]
ZONES    = ["CBD", "Residential", "Student"]


def make_payload(seed: int | None = None) -> dict:
    """Build a realistic random request payload."""
    if seed is not None:
        rng_local = np.random.default_rng(seed)
    else:
        rng_local = rng

    cart_size = int(rng_local.integers(1, 5))
    cart      = [int(x) for x in rng_local.choice(item_ids, size=cart_size, replace=False)]

    return {
        "user_id":    int(rng_local.choice(user_ids)),
        "cart_items": cart,
        "context": {
            "tier":         int(rng_local.integers(1, 4)),
            "season":       str(rng_local.choice(SEASONS)),
            "zone_type":    str(rng_local.choice(ZONES)),
            "hour":         int(rng_local.integers(0, 24)),
            "day_of_week":  int(rng_local.integers(0, 7)),
            "month":        int(rng_local.integers(1, 13)),
            "distance_km":  float(round(rng_local.uniform(1.0, 15.0), 1)),
            "delivery_fee": float(round(rng_local.uniform(20.0, 80.0), 0)),
            "has_main":     int(rng_local.integers(0, 2)),
            "has_side":     int(rng_local.integers(0, 2)),
            "has_drink":    int(rng_local.integers(0, 2)),
            "has_dessert":  int(rng_local.integers(0, 2)),
        },
        "k": 8,
    }


def run_latency_test():
    with httpx.Client(timeout=TIMEOUT) as client:

        # 1. Check server is up
        try:
            r = client.get(f"{BASE_URL}/health")
            r.raise_for_status()
            print(f"\n✅ Server is up: {r.json()}")
        except Exception as e:
            print(f"\n❌ Server not reachable at {BASE_URL}")
            print(f"   Start it with: uvicorn src.api:app --port 8000")
            print(f"   Error: {e}")
            return

        # 2. Warm up (excluded from measurements)
        print(f"\n⏳ Warming up ({N_WARMUP} calls)...")
        for i in range(N_WARMUP):
            client.post(f"{BASE_URL}/recommend", json=make_payload(seed=i))
        print("✅ Warmup complete")

        # 3. Measure N_CALLS requests
        print(f"\n📏 Measuring latency over {N_CALLS} real HTTP requests...")
        latencies_ms  = []
        errors        = 0
        server_ms     = []   # server-side pipeline latency from response body

        for i in range(N_CALLS):
            payload = make_payload(seed=N_WARMUP + i)

            t0 = time.perf_counter()
            try:
                r = client.post(f"{BASE_URL}/recommend", json=payload)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                if r.status_code == 200:
                    latencies_ms.append(elapsed_ms)
                    body = r.json()
                    server_ms.append(body.get("latency_ms", 0))
                else:
                    errors += 1

            except Exception:
                errors += 1

            if (i + 1) % 100 == 0:
                p95_so_far = np.percentile(latencies_ms, 95) if latencies_ms else 0
                print(f"  {i+1}/{N_CALLS} done — P95 so far: {p95_so_far:.1f}ms")

    # 4. Results
    if not latencies_ms:
        print("\n❌ No successful requests recorded.")
        return

    arr       = np.array(latencies_ms)
    srv       = np.array(server_ms) if server_ms else np.array([0.0])
    p95       = float(np.percentile(arr, 95))
    target_hi = 300.0
    headroom  = target_hi / p95

    print(f"\n{'═'*55}")
    print(f"ONLINE LATENCY RESULTS  (n={len(latencies_ms)} successful, {errors} errors)")
    print(f"{'─'*55}")
    print(f"  End-to-end HTTP latency (client-measured):")
    print(f"    Mean : {arr.mean():.2f} ms")
    print(f"    P50  : {np.percentile(arr, 50):.2f} ms")
    print(f"    P95  : {p95:.2f} ms  ← SLA budget ≤300ms")
    print(f"    P99  : {np.percentile(arr, 99):.2f} ms")
    print(f"    Min  : {arr.min():.2f} ms")
    print(f"    Max  : {arr.max():.2f} ms")
    print(f"\n  Server-side pipeline only (no network):")
    print(f"    P95  : {np.percentile(srv, 95):.2f} ms")
    print(f"\n  Errors: {errors}/{N_CALLS}")
    print(f"{'─'*55}")

    if p95 <= target_hi:
        print(f"  Status: ✅ PASS — P95 {p95:.2f}ms ({headroom:.0f}× within {target_hi:.0f}ms SLA budget)")
    else:
        print(f"  Status: ⚠️  OVER BUDGET — P95 {p95:.1f}ms > {target_hi}ms")
        print(f"  Tip: reduce candidates (50→20) or drop segment_cooc_score feature")
    print(f"{'═'*55}\n")

    # 5. Fetch server metrics
    with httpx.Client() as c2:
        try:
            m = c2.get(f"{BASE_URL}/metrics").json()
            print("SERVER METRICS:")
            for k, v in m.items():
                if v is not None:
                    print(f"  {k:<25}: {v}")
        except Exception:
            pass


if __name__ == "__main__":
    run_latency_test()
