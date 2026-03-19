"""
Step 02a — Pluvial Flood Detection / PFDI (standalone script)

Calculates PFDI = UPA / polygon_area via Google Earth Engine (MERIT Hydro).
Already checkpointed per-batch; re-run to resume.

Usage:
    python run_02a.py
"""
import os, sys, glob, time, logging, warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import ee
from shapely import wkb as shapely_wkb

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_PATH    = r"D:\MY_CODES\global_urban_flood_database\groundsource\outputs\groundsource_urban_df.parquet"
BATCH_DIR     = r"D:\MY_CODES\global_urban_flood_database\groundsource\outputs\pfdi_batches"
OUTPUT_PATH   = r"D:\MY_CODES\global_urban_flood_database\groundsource\outputs\groundsource_df_with_pfdi.parquet"
LOG_PATH      = r"D:\MY_CODES\global_urban_flood_database\groundsource\outputs\02a_pluvial_detection.log"

GEE_PROJECT   = 'your-gee-project'   # ← set your GEE project ID here
BATCH_SIZE    = 50
SCALE_M       = 90
TILE_SCALE    = 8
EQUAL_AREA_CRS = "EPSG:6933"
WGS84_CRS     = "EPSG:4326"

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────
def fix_geometry_safe(geom):
    try:
        if not geom.is_valid:
            return geom.buffer(0)
    except Exception:
        pass
    return geom


def shapely_to_ee(geom):
    return ee.Geometry(geom.__geo_interface__)


def get_start_event_id(batch_dir):
    files = glob.glob(os.path.join(batch_dir, 'pfdi_batch_*.parquet'))
    if not files:
        return 0
    dfs = [pd.read_parquet(f, columns=['event_id']) for f in files]
    return int(pd.concat(dfs)['event_id'].max()) + 1


def ee_fc_to_df(fc):
    try:
        features = fc.getInfo()['features']
    except Exception as e:
        log.warning(f"GEE error: {e}")
        return None
    rows = [f['properties'] for f in features]
    return pd.DataFrame(rows) if rows else None


def pick_stat_column(row, base, suffixes=('', '_1', '_2')):
    for s in suffixes:
        col = f"{base}{s}"
        if col in row and not pd.isna(row[col]):
            return float(row[col])
    return np.nan


def main():
    os.makedirs(BATCH_DIR, exist_ok=True)

    ee.Initialize(project=GEE_PROJECT)
    log.info("GEE initialised")

    merit = ee.Image("MERIT/Hydro/v1_0_1").select('upa')

    df = pd.read_parquet(INPUT_PATH).reset_index(drop=True)
    df['event_id'] = df.index

    geoms_ea = gpd.GeoSeries.from_wkb(df['geometry'], crs=WGS84_CRS).to_crs(EQUAL_AREA_CRS)
    df['poly_area_km2'] = geoms_ea.area / 1e6
    log.info(f"Loaded {len(df):,} records")

    start_id = get_start_event_id(BATCH_DIR)
    subset   = df[df['event_id'] >= start_id].copy()
    log.info(f"Resuming from event_id={start_id} | remaining: {len(subset):,}")

    reducer = ee.Reducer.max().combine(ee.Reducer.percentile([95, 99]), sharedInputs=True)

    t_session  = time.time()
    batches_ok = 0

    for batch_start in range(0, len(subset), BATCH_SIZE):
        batch  = subset.iloc[batch_start : batch_start + BATCH_SIZE]
        id_min = int(batch['event_id'].min())
        id_max = int(batch['event_id'].max())
        out_file = os.path.join(BATCH_DIR, f'pfdi_batch_{id_min}_{id_max}.parquet')

        if os.path.exists(out_file):
            batches_ok += 1
            continue

        try:
            t0 = time.time()
            features = []
            for _, row in batch.iterrows():
                geom = shapely_wkb.loads(row['geometry'])
                geom = fix_geometry_safe(geom)
                features.append(ee.Feature(shapely_to_ee(geom), {'event_id': int(row['event_id'])}))

            fc     = ee.FeatureCollection(features)
            result = merit.reduceRegions(collection=fc, reducer=reducer,
                                         scale=SCALE_M, tileScale=TILE_SCALE)
            result_df = ee_fc_to_df(result)
            if result_df is None or result_df.empty:
                batches_ok += 1
                continue

            rows_out = []
            for _, r in result_df.iterrows():
                eid     = int(r.get('event_id', -1))
                upa_max = pick_stat_column(r, 'max')
                upa_p95 = pick_stat_column(r, 'p95')
                upa_p99 = pick_stat_column(r, 'p99')
                src     = batch[batch['event_id'] == eid]
                poly_a  = float(src['poly_area_km2'].values[0]) if len(src) else np.nan
                rows_out.append({
                    'event_id'    : eid,
                    'upa_max'     : upa_max,
                    'upa_p95'     : upa_p95,
                    'upa_p99'     : upa_p99,
                    'poly_area_km2': poly_a,
                    'PFDI_p95'    : upa_p95 / poly_a if poly_a and poly_a > 0 else np.nan,
                    'PFDI_p99'    : upa_p99 / poly_a if poly_a and poly_a > 0 else np.nan,
                    'PFDI_max'    : upa_max / poly_a if poly_a and poly_a > 0 else np.nan,
                })

            pd.DataFrame(rows_out).to_parquet(out_file, index=False)
            batches_ok += 1
            pct = (start_id + batch_start + len(batch)) / len(df) * 100
            log.info(f"Batch {batches_ok} | events {id_min}–{id_max} | {time.time()-t0:.1f}s | {pct:.1f}%")

        except Exception as e:
            log.warning(f"Batch {id_min}–{id_max} failed: {e}")
            batches_ok += 1
            continue

    log.info(f"Batch processing done in {time.time()-t_session:.0f}s")

    # ── Merge & save ───────────────────────────────────────────────────────────
    batch_files = sorted(glob.glob(os.path.join(BATCH_DIR, 'pfdi_batch_*.parquet')))
    log.info(f"Merging {len(batch_files)} batch files")
    pfdi_df = pd.concat([pd.read_parquet(f) for f in batch_files], ignore_index=True)
    merged  = df.merge(pfdi_df, on='event_id', how='inner')

    key_cols = ['upa_p95', 'upa_p99', 'upa_max', 'poly_area_km2']
    merged = merged.dropna(subset=key_cols)
    merged = merged[merged['poly_area_km2'] > 0]
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['PFDI_p95', 'PFDI_p99', 'PFDI_max'])

    merged.to_parquet(OUTPUT_PATH, index=False)
    log.info(f"Saved {len(merged):,} rows → {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
