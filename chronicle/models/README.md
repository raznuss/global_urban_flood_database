# Models

Machine learning models for urban pluvial flood prediction using the Chronicle dataset.

## Scope

All models in this directory are trained on **pluvial urban floods only**:
- Urban filter: `urban_percentage >= 20%`
- Pluvial filter: `PFDI_p99 < 1.0`
- Rainfall filter: `60-min max > 7 mm/hr` AND `1440-min max > 1 mm/day`

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_baseline_training_and_evaluation.ipynb` | Random Forest + XGBoost baselines (9 features: urbanization + rainfall intensities) |

## Features

**Urbanization**
- `urban_percentage` – % built-up area within flood polygon (GHS-BUILT)
- `urban_built_up_area_m2` – absolute built-up area (m²)

**Rainfall Intensities (IMERG V07)**
- `30 / 60 / 120 / 240 / 360 / 720 / 1440_max_rainfall_intens` – max intensity (mm/hr) at each duration window

## Input Data

Loaded from: `D:\Development\RESEARCH\urban_flood_database\chronicle\unified_ml_dataset_LIGHTWEIGHT.pkl`
