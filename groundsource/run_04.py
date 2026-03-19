"""
Step 04 — No-Flood Event Addition (standalone script)

Generates negative training samples from heavy-rain events that did NOT
coincide with recorded floods, using stratified polygon sampling.
Already checkpointed per-batch; re-run to resume.

Usage:
    python run_04.py
"""
import os, sys, glob, time, uuid, pickle, logging, warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import ee
from shapely import wkb as shapely_wkb

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_PATH    = r"D:\MY_CODES\global_urban_flood_database\groundsource\outputs\groundsource_rain_master.parquet"
BATCH_DIR     = r"D:\MY_CODES\global_urban_flood_database\groundsource\outputs\no_flood_batches"
LOG_PATH      = r"D:\MY_CODES\global_urban_flood_database\groundsource\outputs\04_no_flood.log"

GEE_PROJECT   = 'your-gee-project'   # ← set your GEE project ID here

MODERN_YEAR      = 2018
SEARCH_YEARS     = list(range(2018, 2026))
MAX_PER_BUCKET   = 8
THRESH_1H        = 7.0
THRESH_24H       = 1.0
BUFFER_DAYS      = 3
SCALE_DEG        = 0.1
HOURS_BEFORE     = 72
HOURS_AFTER      = 24
BATCH_SIZE_POLY  = 10
WGS84_CRS        = "EPSG:4326"
DURATIONS_MIN    = [30, 60, 120, 240, 360, 720, 1440]

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
def rasterize_polygon(wkb_bytes, origin_lon, origin_lat, n_lon, n_lat):
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


def extract_rain_matrix(wkb_bytes, storm_date, product='final'):
    collection_id = (
        'NASA/GPM_L3/IMERG_V07/FINAL/HALF_HOURLY' if product == 'final'
        else 'NASA/GPM_L3/IMERG_V07/LATE/HALF_HOURLY'
    )
    t_start = storm_date - timedelta(hours=HOURS_BEFORE)
    t_end   = storm_date + timedelta(hours=HOURS_AFTER)
    geom    = shapely_wkb.loads(wkb_bytes)
    minx, miny, maxx, maxy = geom.bounds
    buf    = SCALE_DEG
    region = ee.Geometry.BBox(minx - buf, miny - buf, maxx + buf, maxy + buf)

    ic = (ee.ImageCollection(collection_id)
          .filterDate(t_start.strftime('%Y-%m-%dT%H:%M:%S'),
                      t_end.strftime('%Y-%m-%dT%H:%M:%S'))
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
                origin_lon, origin_lat = minx - buf, maxy + buf
        ts = img.get('system:time_start').getInfo()
        timestamps.append(datetime.utcfromtimestamp(ts / 1000).strftime('%Y-%m-%dT%H:%M:%S'))
        slices.append(arr)

    matrix = np.stack(slices, axis=0)
    mask   = rasterize_polygon(wkb_bytes, origin_lon, origin_lat, n_lon, n_lat)
    meta   = {'origin_lon': origin_lon, 'origin_lat': origin_lat,
              'scale_deg': SCALE_DEG, 'shape': matrix.shape, 'timestamps': timestamps}
    return matrix, mask, meta


def compute_intensities(matrix, mask):
    n_pixels = mask.sum()
    if n_pixels == 0:
        return {f"{d}_max_rainfall_intens": np.nan for d in DURATIONS_MIN}
    series = (matrix * mask[np.newaxis, :, :]).sum(axis=(1, 2)) / n_pixels
    result = {}
    for dur_min in DURATIONS_MIN:
        n_steps = dur_min // 30
        kernel  = np.ones(n_steps) / n_steps
        rolled  = np.convolve(series, kernel, mode='valid')
        result[f"{dur_min}_max_rainfall_intens"] = float(rolled.max()) if len(rolled) > 0 else np.nan
    return result


def scan_for_storms(wkb_bytes, flood_dates_set):
    geom = shapely_wkb.loads(wkb_bytes)
    minx, miny, maxx, maxy = geom.bounds
    buf    = SCALE_DEG
    region = ee.Geometry.BBox(minx - buf, miny - buf, maxx + buf, maxy + buf)
    storm_dates = []

    for year in SEARCH_YEARS:
        for month in range(1, 13):
            t_start = f"{year}-{month:02d}-01"
            t_end   = f"{year + 1}-01-01" if month == 12 else f"{year}-{month+1:02d}-01"
            try:
                ic = (ee.ImageCollection('NASA/GPM_L3/IMERG_V07/FINAL/HALF_HOURLY')
                      .filterDate(t_start, t_end)
                      .filterBounds(region)
                      .select('precipitationCal'))
                n = ic.size().getInfo()
                if n == 0:
                    continue

                daily      = ic.map(lambda img: img.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=region,
                    scale=int(SCALE_DEG * 111_000), bestEffort=True
                ).set('system:time_start', img.get('system:time_start')))
                daily_info = daily.getInfo()
                if not daily_info:
                    continue

                best_ts, best_val = None, -1
                for feat in daily_info:
                    val = feat.get('precipitationCal', 0) or 0
                    if val > best_val:
                        best_val, best_ts = val, feat.get('system:time_start')

                if best_val < THRESH_1H or best_ts is None:
                    continue

                storm_dt   = datetime.utcfromtimestamp(best_ts / 1000)
                storm_date = storm_dt.date()
                if not any(abs((storm_date - fd).days) <= BUFFER_DAYS for fd in flood_dates_set):
                    storm_dates.append(storm_dt)
            except Exception:
                continue

    return storm_dates


def main():
    os.makedirs(BATCH_DIR, exist_ok=True)

    ee.Initialize(project=GEE_PROJECT)
    log.info("GEE initialised")

    df = pd.read_parquet(INPUT_PATH)
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date']   = pd.to_datetime(df['end_date'])
    log.info(f"Loaded {len(df):,} records")

    # ── Stratified sampling ────────────────────────────────────────────────────
    modern = df[df['start_date'].dt.year >= MODERN_YEAR].copy()
    modern['area_bin']      = pd.qcut(modern['area_km2_mollweide'], q=3,
                                       labels=['Small', 'Medium', 'Large'])
    modern['urban_bin']     = pd.cut(modern['urban_percentage'], bins=[0, 20, 60, 100],
                                      labels=['Low-Urban', 'Urban', 'Highly-Urban'],
                                      include_lowest=True)
    modern['mechanism_bin'] = np.where(modern['PFDI_max'] < 1, 'Pluvial', 'Fluvial')

    targets = (modern
               .groupby(['area_bin', 'urban_bin', 'mechanism_bin'], observed=True)
               .apply(lambda g: g.sample(n=min(MAX_PER_BUCKET, len(g)), random_state=42))
               .reset_index(drop=True))
    log.info(f"Stratified sample: {len(targets)} polygons")

    # ── Resume ─────────────────────────────────────────────────────────────────
    done_files = glob.glob(os.path.join(BATCH_DIR, 'no_flood_batch_*.parquet'))
    done_uuids = set()
    if done_files:
        done_uuids = set(
            pd.concat([pd.read_parquet(f, columns=['source_uuid']) for f in done_files])
            ['source_uuid'].tolist()
        )
    remaining_targets = targets[~targets['uuid'].isin(done_uuids)].copy()
    log.info(f"Already processed: {len(done_uuids)} | remaining: {len(remaining_targets)}")

    t_session       = time.time()
    total_generated = 0

    for batch_start in range(0, len(remaining_targets), BATCH_SIZE_POLY):
        batch       = remaining_targets.iloc[batch_start : batch_start + BATCH_SIZE_POLY]
        batch_label = f"{batch_start}_{batch_start + len(batch) - 1}"
        out_file    = os.path.join(BATCH_DIR, f'no_flood_batch_{batch_label}.parquet')

        if os.path.exists(out_file):
            continue

        records = []
        for _, row in batch.iterrows():
            flood_dates = set(df[df['uuid'] == row['uuid']]['start_date'].dt.date.tolist())
            try:
                storm_dates = scan_for_storms(row['geometry'], flood_dates)
            except Exception as e:
                log.warning(f"Scan failed uuid={row['uuid'][:8]}: {e}")
                continue

            for storm_dt in storm_dates:
                try:
                    matrix, mask, meta = extract_rain_matrix(row['geometry'], storm_dt, 'final')
                    if matrix is None:
                        matrix, mask, meta = extract_rain_matrix(row['geometry'], storm_dt, 'late')
                    if matrix is None:
                        continue

                    intensities = compute_intensities(matrix, mask)
                    if intensities.get('60_max_rainfall_intens', 0) < THRESH_1H:
                        continue
                    if intensities.get('1440_max_rainfall_intens', 0) < THRESH_24H:
                        continue

                    storm_date_str = storm_dt.strftime('%Y%m%d')
                    records.append({
                        'uuid'                  : str(uuid.uuid4()),
                        'source_uuid'           : row['uuid'],
                        'event_id'              : f"{row['uuid']}_noflood_{storm_date_str}",
                        'start_date'            : storm_dt.strftime('%Y-%m-%d'),
                        'end_date'              : storm_dt.strftime('%Y-%m-%d'),
                        'geometry'              : row['geometry'],
                        'area_km2_original'     : row['area_km2_original'],
                        'area_km2_mollweide'    : row['area_km2_mollweide'],
                        'urban_percentage'      : row['urban_percentage'],
                        'urban_built_up_area_m2': row.get('urban_built_up_area_m2', np.nan),
                        'polygon_total_area_m2' : row.get('polygon_total_area_m2', np.nan),
                        'upa_max'               : row.get('upa_max', np.nan),
                        'upa_p95'               : row.get('upa_p95', np.nan),
                        'upa_p99'               : row.get('upa_p99', np.nan),
                        'PFDI_p95'              : row.get('PFDI_p95', np.nan),
                        'PFDI_p99'              : row.get('PFDI_p99', np.nan),
                        'PFDI_max'              : row.get('PFDI_max', np.nan),
                        'imerg_matrix'          : pickle.dumps(matrix),
                        'imerg_mask'            : pickle.dumps(mask),
                        'imerg_meta'            : pickle.dumps(meta),
                        'imerg_type'            : 'final',
                        'is_flood'              : 0,
                        **intensities,
                    })
                except Exception as e:
                    log.warning(f"Matrix failed storm {storm_dt.date()}: {e}")
                    continue

        if records:
            pd.DataFrame(records).to_parquet(out_file, index=False)

        total_generated += len(records)
        pct = (batch_start + len(batch)) / len(remaining_targets) * 100
        log.info(f"Batch {batch_start // BATCH_SIZE_POLY + 1} | "
                 f"polygons {batch_start}–{batch_start+len(batch)-1} | "
                 f"generated {len(records)} | total: {total_generated} | {pct:.1f}%")

    log.info(f"Done. Total generated: {total_generated} | elapsed: {time.time()-t_session:.0f}s")


if __name__ == '__main__':
    main()
