# Boco Rock Wind Power Predictor

Streamlit web app for predicting hourly power output at Boco Rock Wind Farm (NSW, Australia) using an XGBoost model trained on ERA5 reanalysis wind data (2016–2023).

**ENGG2112 — Team 12**

## Model Performance (2024 Test Set)
| Metric | Value |
|--------|-------|
| R² | 0.7694 |
| MAE | 12.19 MW |
| RMSE | 17.17 MW |

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## CSV format

Upload a CSV with these columns:

| Column | Description |
|--------|-------------|
| `time` | Hourly timestamp |
| `u10` | U-component of wind at 10 m (m/s) |
| `v10` | V-component of wind at 10 m (m/s) |
| `u100` | U-component of wind at 100 m (m/s) |
| `v100` | V-component of wind at 100 m (m/s) |

A sample file (`sample_boco_week.csv`) is included — it contains the first week of January 2024 from the Boco Rock ERA5 dataset.

## Notes
- Model is site-specific to Boco Rock. Cross-site predictions (e.g. Silverton) show significantly degraded performance (R² ~0.35).
- Lag features (1h, 2h, 3h, 6h lookback) require sequential hourly rows to be most accurate.
