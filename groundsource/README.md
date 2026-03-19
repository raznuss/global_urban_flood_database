# Groundsource Urban Flood Database — Processing Pipeline

This pipeline transforms raw satellite-detected flood polygons from the Groundsource dataset
into a machine-learning-ready dataset with georeferenced rainfall, urbanisation, and hydrology features.

---

## Input Dataset

- **`groundsource_2026.parquet`** — 2,646,302 raw flood event records
  - Columns: `uuid`, `geometry` (WKB bytes, WGS84), `start_date`, `end_date`, `area_km2`, and others
  - Source: Groundsource global flood database

---

## Pipeline Overview

The pipeline is split into **processing notebooks** (heavy computation, hours to days) and
**analysis notebooks** (EDA and visualisation, run manually in Jupyter).

```
00  Preliminary EDA
01a Urban Areas Detection        ← heavy, checkpointed
01b Urban Analysis               ← Jupyter EDA
02a Pluvial Flood Detection      ← heavy, GEE, checkpointed
02b Pluvial Analysis             ← Jupyter EDA
03a Extract IMERG Rain Matrices  ← heavy, GEE, checkpointed
03b Rain Intensity Standardisation
04  No-Flood Event Addition      ← heavy, GEE, checkpointed
05  Merge & Final EDA            ← Jupyter EDA
```

**Final output:** `outputs/groundsource_unified_ml_dataset.parquet`

---

## Running the Pipeline

### Option A — Fully automated (recommended for long runs)

```bash
# Run all heavy steps from scratch
python run_pipeline.py

# Resume from a specific step (e.g. after 01a is done)
python run_pipeline.py --from 02a

# Run only one step
python run_pipeline.py --only 03a
```

All steps are **idempotent** — already-completed work is automatically skipped.
Progress is logged to `outputs/pipeline.log`.

Each step runs in a separate subprocess, so memory is fully released between steps.

### Option B — Manual Jupyter (for visibility and graphs)

Run the `a`-notebooks manually cell by cell. The `b`-notebooks and `05` are
analysis-only and are always run manually in Jupyter.

---

## Checkpoint System

Every heavy step saves intermediate results to disk immediately and resumes from
the last saved point on restart. No work is lost on crash or kernel restart.

| Step | Checkpoint location |
|------|---------------------|
| 01a  | `outputs/urban_chunks/chunk_XXXX.parquet` (one file per 5,000-row chunk) |
| 02a  | `outputs/pfdi_batches/pfdi_batch_*.parquet` (one file per 50-event batch) |
| 03a  | `outputs/imerg_batches/imerg_batch_*.parquet` (one file per 1,000-event batch) |
| 04   | `outputs/no_flood_batches/no_flood_batch_*.parquet` (one file per 10-polygon batch) |

---

## Step-by-Step Details

### 00 — Preliminary Analysis
- Notebook: `00_groundsource_preliminary_analysis.ipynb`
- EDA only, no output files

### 01a — Urban Areas Detection
- Notebook: `01a_groundsource_urban_areas_detection.ipynb`
- Script:   `run_01a.py`
- Method: Zonal statistics of GHS-BUILT raster (100 m, Mollweide) over each polygon
- Parallelised with `ProcessPoolExecutor` (6 workers, 5,000 rows/chunk)
- Input:  `groundsource_2026.parquet` + GHS-BUILT raster
- Output: `outputs/groundsource_urban_df.parquet`
  - New columns: `urban_built_up_area_m2`, `polygon_total_area_m2`, `urban_percentage`

### 01b — Urban Analysis
- Notebook: `01b_groundsource_urban_analysis.ipynb`
- EDA only. Validates the 20% urban threshold (~32.5% of events pass).

### 02a — Pluvial Flood Detection (PFDI)
- Notebook: `02a_groundsource_pluvial_detection.ipynb`
- Script:   `run_02a.py`
- Method: PFDI = UPA / polygon_area via MERIT Hydro (GEE), 50 events/batch
- PFDI < 1 → pluvial (local rainfall); PFDI ≥ 1 → fluvial (river-driven)
- Input:  `outputs/groundsource_urban_df.parquet`
- Output: `outputs/groundsource_df_with_pfdi.parquet`
  - New columns: `upa_max`, `upa_p95`, `upa_p99`, `poly_area_km2`, `PFDI_p95`, `PFDI_p99`, `PFDI_max`
- **Requires GEE project ID** — set `GEE_PROJECT` in the config section

### 02b — Pluvial Analysis
- Notebook: `02b_groundsource_pluvial_analysis.ipynb`
- EDA only.

### 03a — Extract IMERG Precipitation Matrices
- Notebook: `03a_groundsource_extract_imerg_rain.ipynb`
- Script:   `run_03a.py`
- Method: Downloads 3-D rainfall arrays (T × H × W) from NASA IMERG V07 via GEE
  - Window: 72 h before to 24 h after event start, 0.1° resolution, 30-min timesteps
  - Falls back from IMERG 'final' to 'late' product if needed
  - Batches of 1,000 events
- Input:  `outputs/groundsource_df_with_pfdi.parquet`
- Output: `outputs/groundsource_with_imerg.parquet`
  - New columns: `imerg_matrix` (pickle bytes), `imerg_mask`, `imerg_meta`, `imerg_type`
- **Requires GEE project ID**

### 03b — Rain Intensity Standardisation
- Notebook: `03b_groundsource_rain_intensity_standardization.ipynb`
- Computes peak rolling-mean intensities for 7 durations: 30, 60, 120, 240, 360, 720, 1440 min
- Recalculates PFDI with corrected Mollweide area
- Input:  `outputs/groundsource_with_imerg.parquet`
- Output: `outputs/groundsource_rain_master.parquet`
  - New columns: `area_km2_mollweide`, `{30,60,120,240,360,720,1440}_max_rainfall_intens`

### 04 — No-Flood Event Addition
- Notebook: `04_groundsource_no_flood_event_addition.ipynb`
- Script:   `run_04.py`
- Method: Stratified sampling of 18 buckets (area × urban % × mechanism),
  scans IMERG for heavy-rain months without a recorded flood (±3-day buffer)
- Label: `is_flood = 0`
- Input:  `outputs/groundsource_rain_master.parquet`
- Output: `outputs/no_flood_batches/no_flood_batch_*.parquet`
- **Requires GEE project ID**

### 05 — Merge & Final EDA
- Notebook: `05_groundsource_merge_and_eda.ipynb`
- Concatenates flood events (`is_flood=1`) and no-flood events (`is_flood=0`)
- Output: `outputs/groundsource_unified_ml_dataset.parquet` (~35 columns, ML-ready)

---

## GEE Setup

Steps 02a, 03a, and 04 require Google Earth Engine access. Before running:

1. Authenticate: `earthengine authenticate`
2. Set `GEE_PROJECT = 'your-actual-project-id'` in each of `run_02a.py`, `run_03a.py`, `run_04.py`
   (or in the configuration cell of each notebook)

---

## Key Dependencies

```
pandas, geopandas, numpy, rasterstats, earthengine-api, shapely, rasterio, pyarrow
```

---

## Output Schema (Final Dataset)

| Column group | Columns |
|---|---|
| Identifiers | `uuid`, `event_id`, `start_date`, `end_date` |
| Geometry & land cover | `geometry`, `area_km2_original`, `area_km2_mollweide`, `urban_percentage` |
| Hydrology | `upa_max`, `upa_p95`, `upa_p99`, `poly_area_km2`, `PFDI_p95`, `PFDI_p99`, `PFDI_max` |
| Raw rainfall | `imerg_matrix`, `imerg_mask`, `imerg_meta`, `imerg_type` |
| Intensity features | `30_max_rainfall_intens` … `1440_max_rainfall_intens` |
| Label | `is_flood` (1 = flood, 0 = no-flood) |
