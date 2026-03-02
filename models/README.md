# models/

Trained model artefacts are stored in **`data/models/`** (co-located with the data pipeline).

```
data/models/
├── ranking_model.pkl      # LightGBM LambdaRank — trained model object
├── feature_cols.pkl       # Ordered feature column list used at inference time
└── feature_importance.png # Top-20 feature importance bar chart
```

To regenerate:

```bash
python src/ranking_model.py
```

Large model files (>50 MB) are excluded from git via `.gitignore` patterns on `*.pkl` / `*.bin`.
A Google Drive link with pre-trained artefacts is provided in `data/Link.md`.
