# Chronicle – Global Urban Flood Database

A machine learning pipeline for detecting and predicting urban pluvial floods, built on the Chronicle dataset (HUJI v3.1).

## Overview

The pipeline processes 882,972 global flood events (2000–2025) through a series of spatial enrichment, rainfall extraction, and sampling steps, ultimately producing a labeled dataset for supervised flood prediction models.

## Pipeline

```
Chronicle Raw Dataset (882,972 events)
         │
         ▼
[00] Preliminary Analysis
         │
         ▼
[01a] Urban Detection (GHS-BUILT, 100m)
         │  → 214,204 urban events (24.3%)
         ▼
[01b] Urban Analysis & Statistics
         │
         ▼
[02a] Pluvial Detection (MERIT Hydro → PFDI)
         │  → PFDI < 1.0 = pluvial | PFDI ≥ 1.0 = fluvial
         ▼
[02b] Pluvial Analysis & Statistics
         │
         ▼
[03a] IMERG V07 Rainfall Extraction (NASA GPM, 0.1°, 30-min)
         │
         ▼
[03b] Rainfall Intensity Standardization (7 duration windows)
         │
         ├──────────────────────────┐
         ▼                          ▼
[04] No-Flood Sampling        Rain Master Dataset
     (4,192 negatives)
         │
         ▼
[05] Merge & EDA
     → unified_ml_dataset_FULL.pkl       (with 3D IMERG matrices)
     → unified_ml_dataset_LIGHTWEIGHT.pkl (tabular only)
         │
         ▼
[models/] ML Training & Evaluation
```

## Directory Structure

```
chronicle/
├── data_processing/     # Pipeline notebooks (00–05)
├── models/              # ML model notebooks
├── archive/             # Deprecated/draft notebooks
├── README.md
└── .gitignore
```

## Key Concepts

| Concept | Definition |
|---------|-----------|
| Urban threshold | `urban_percentage >= 20%` (GHS-BUILT) |
| PFDI | Pluvial Flood Detectability Index = UPA_p95 / polygon_area |
| Pluvial filter | `PFDI_p99 < 1.0` |
| Rainfall windows | 30, 60, 120, 240, 360, 720, 1440 minutes (max intensity) |

## Data Sources

| Dataset | Source | Resolution |
|---------|--------|------------|
| Chronicle flood polygons | HUJI v3.1 | Vector |
| GHS-BUILT (built-up area) | ESA WorldCover | 100m |
| MERIT Hydro (UPA) | MERIT | 90m |
| IMERG V07 (rainfall) | NASA GPM | 0.1° / 30-min |

## Output Data

All processed datasets are stored externally at:
`D:\Development\RESEARCH\urban_flood_database\chronicle\`
