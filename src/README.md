# SmartCart v2.0 — `src/`

Python utilities extracted from notebooks. Import from any notebook or run as CLI scripts.

## Structure

```
src/
├── __init__.py              # Package initialization
├── config.py                # Centralized configuration & constants
├── generate_data.py         # Synthetic data generation (CLI utility)
└── evaluate.py              # Pre-processing evaluation & validation
```

## Modules

### `config.py`
Centralised constants for the entire project: data paths, city tiers, cuisine lists, feature column names, and the <300 ms latency budget.

### `generate_data.py`
Synthetic food delivery dataset generator.

```bash
python src/generate_data.py                                        # defaults: 50k users, 200k orders
python src/generate_data.py --num-users 10000 --num-orders 50000
```

### `evaluate.py`
Validates the pre-processing pipeline against the <300 ms latency budget (candidate gen <50 ms P95, feature fetch <60 ms P95).

```bash
python src/evaluate.py
```

## Importing from notebooks

```python
import sys
sys.path.insert(0, '../src')
from config import PROCESSED_DATA_DIR, LATENCY_BUDGET
from generate_data import generate_dataset
```