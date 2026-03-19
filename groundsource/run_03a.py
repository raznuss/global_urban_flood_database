"""
Step 03a — Extract IMERG Precipitation Matrices (standalone script)

Downloads 3-D rainfall arrays from NASA IMERG V07 via GEE for each flood event.
Already checkpointed per-batch; re-run to resume.

Usage:
    python run_03a.py
"""
import os, sys, glob, time, pickle, logging, warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import geopandas as gpd
import ee
from shapely import wkb as shapely_wkb

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_PATH    = r"D:\MY_CODES\global_urban_flood_database\groundsource\outputs\groundsource_df_with_pfdi.parquet"
BATCH_DIR     = r"D:\MY_CODES\global_urban_flood_database\groundsource\outputs\imerg_batches"
OUTPUT_PATH   = r"D:\MY_CODES\global_urban_flood_database\groundsource\outputs\groundsource_with_imerg.parquet"
LOG_PATH      = r"D:\MY_CODES\global_urban_flood_database\groundsource\outputs\03a_extract_imerg.log"

GEE_PROJECT   = 'your-gee-project'   # ← set your GEE project ID here
IMERG_START   = '2000-06-01'
HOURS_BEFORE  = 72
HOURS_AFTER   = 24
SCALE_DEG     = 0.1
BATCH_SIZE    = 1_000
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
def get_bounding_box(wkb_bytes):
    geom = shapely_wkb.loads(wkb_bytes)
    minx, miny, maxx, maxy = geom.bounds
    buf = SCALE_DEG
    return minx - buf, miny - buf, maxx + buf, maxy + buf


def rasterize_polygon(wkb_bytes, origin_lon, origin_lat, n_lon, n_lat):
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.features import rasterize
    geom      = shapely_wkb.loads(wkb_bytes)
    transform = from_origin(origin_lon, origin_lat + n_lat * SCALE_DEG, SCALE_DEG, SCALE_DEG)
    return rasterize(
        [(geom.__geo_interface__, 1)],
        out_shape=(n_lat, n_lon),
        transform=transform,
        fill=0, dtype='uint8', all_touched=True
    )


def build_time_window(start_date):
    t_start = start_date - timedelta(hours=HOURS_BEFORE)
    t_end   = start_date + timedelta(hours=HOURS_AFTER)
    return t_start.strftime('%Y-%m-%dT%H:%M:%S'), t_end.strftime('%Y-%m-%dT%H:%M:%S')


def extract_imerg_for_event(wkb_bytes, start_date, product='final'):
    collection_id = (
        'NASA/GPM_L3/IMERG_V07/FINAL/HALF_HOURLY' if product == 'final'
        else 'NASA/GPM_L3/IMERG_V07/LATE/HALF_HOURLY'
    )
    t_start_str, t_end_str = build_time_window(start_date)
    min_lon, min_lat, max_lon, max_lat = get_bounding_box(wkb_bytes)
    region = ee.Geometry.BBox(min_lon, min_lat, max_lon, max_lat)

    ic = (ee.ImageCollection(collection_id)
          .filterDate(t_start_str, t_end_str)
          .filterBounds(region)
          .select('precipitationCal'))

    n_images = ic.size().getInfo()
    if n_images == 0:
        return None, None, None

    image_list = ic.toList(n_images)
    slices, timestamps = [], []
    n_lon = n_lat = origin_lon = origin_lat = None

    for i in range(n_images):
        img  = ee.Image(image_list.get(i))
        info = img.sampleRectangle(region=region, defaultValue=0).getInfo()
        arr  = np.array(info['properties']['precipitationCal'], dtype=np.float32)
        if n_lat is None:
            n_lat, n_lon = arr.shape
            coords = info.get('geometry', {}).get('coordinates', [[]])[0]
            if coords:
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                origin_lon, origin_lat = min(lons), max(lats)
            else:
                origin_lon, origin_lat = min_lon, max_lat
        ts = img.get('system:time_start').getInfo()
        timestamps.append(datetime.utcfromtimestamp(ts / 1000).strftime('%Y-%m-%dT%H:%M:%S'))
        slices.append(arr)

    matrix = np.stack(slices, axis=0)
    mask   = rasterize_polygon(wkb_bytes, origin_lon, origin_lat, n_lon, n_lat)
    meta   = {'origin_lon': origin_lon, 'origin_lat': origin_lat,
              'scale_deg': SCALE_DEG, 'shape': matrix.shape, 'timestamps': timestamps}
    return matrix, mask, meta


def get_completed_event_ids(batch_dir):
    files = glob.glob(os.path.join(batch_dir, 'imerg_batch_*.parquet'))
    if not files:
        return set()
    dfs = [pd.read_parquet(f, columns=['event_id']) for f in files]
    return set(pd.concat(dfs)['event_id'].tolist())


def main():
    os.makedirs(BATCH_DIR, exist_ok=True)

    ee.Initialize(project=GEE_PROJECT)
    log.info("GEE initialised")

    df = pd.read_parquet(INPUT_PATH).reset_index(drop=True)
    df['event_id']   = df.index
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date']   = pd.to_datetime(df['end_date'])
    df = df[df['start_date'] >= pd.Timestamp(IMERG_START)].copy()
    log.info(f"Events after IMERG cutoff: {len(df):,}")

    completed_ids = get_completed_event_ids(BATCH_DIR)
    remaining     = df[~df['event_id'].isin(completed_ids)].copy()
    log.info(f"Already processed: {len(completed_ids):,} | remaining: {len(remaining):,}")

    t_session  = time.time()
    batches_ok = 0

    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch   = remaining.iloc[batch_start : batch_start + BATCH_SIZE]
        id_min  = int(batch['event_id'].min())
        id_max  = int(batch['event_id'].max())
        out_file = os.path.join(BATCH_DIR, f'imerg_batch_{id_min}_{id_max}.parquet')

        if os.path.exists(out_file):
            batches_ok += 1
            continue

        t0      = time.time()
        records = []

        for _, row in batch.iterrows():
            matrix = mask = meta = imerg_type = None
            for product in ('final', 'late'):
                try:
                    matrix, mask, meta = extract_imerg_for_event(
                        row['geometry'], row['start_date'], product=product)
                    if matrix is not None:
                        imerg_type = product
                        break
                except Exception:
                    continue
            if matrix is None:
                continue
            records.append({
                'event_id'    : int(row['event_id']),
                'imerg_matrix': pickle.dumps(matrix),
                'imerg_mask'  : pickle.dumps(mask),
                'imerg_meta'  : pickle.dumps(meta),
                'imerg_type'  : imerg_type,
            })

        if records:
            pd.DataFrame(records).to_parquet(out_file, index=False)

        batches_ok += 1
        pct = (len(completed_ids) + batch_start + len(batch)) / len(df) * 100
        log.info(f"Batch {batches_ok} | events {id_min}–{id_max} | "
                 f"saved {len(records)} | {time.time()-t0:.1f}s | {pct:.1f}%")

    log.info(f"Batch processing done in {time.time()-t_session:.0f}s")

    # ── Merge & save ───────────────────────────────────────────────────────────
    batch_files = sorted(glob.glob(os.path.join(BATCH_DIR, 'imerg_batch_*.parquet')))
    log.info(f"Merging {len(batch_files)} batch files")
    imerg_df = pd.concat([pd.read_parquet(f) for f in batch_files], ignore_index=True)
    merged   = df.merge(imerg_df, on='event_id', how='inner')
    merged.to_parquet(OUTPUT_PATH, index=False)
    log.info(f"Saved {len(merged):,} rows → {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
