"""
Step 01a — Urban Areas Detection (standalone script)

Computes urban built-up percentage for each flood polygon using the GHS-BUILT raster.
Fully checkpointed: each chunk is saved to disk immediately; re-run to resume.

Usage:
    python run_01a.py
"""
import os, sys, time, logging, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_PATH    = r"D:\Development\RESEARCH\urban_flood_database\Groundsource\groundsource_2026.parquet"
GHS_RASTER    = r"D:\Development\RESEARCH\global_datasets\GHS_BUILT\GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0.tif"
OUTPUT_DIR    = r"D:\MY_CODES\global_urban_flood_database\groundsource\outputs"
OUTPUT_PATH   = os.path.join(OUTPUT_DIR, "groundsource_urban_df.parquet")
CHUNK_DIR     = os.path.join(OUTPUT_DIR, "urban_chunks")
LOG_PATH      = os.path.join(OUTPUT_DIR, "01a_urban_detection.log")

MOLLWEIDE_CRS = "ESRI:54009"
WGS84_CRS     = "EPSG:4326"
N_WORKERS     = 6
CHUNK_SIZE    = 5_000

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


# ── Worker — must be module-level for ProcessPoolExecutor pickling ─────────────
def _zonal_stats_worker(args):
    wkb_bytes_list, raster_path, mollweide_crs = args
    import geopandas as gpd
    from rasterstats import zonal_stats
    geoms = gpd.GeoSeries.from_wkb(wkb_bytes_list, crs=mollweide_crs)
    results = zonal_stats(list(geoms), raster_path, stats=['sum'], all_touched=True, nodata=0)
    return [r.get('sum', 0) or 0 for r in results]


def main():
    os.makedirs(CHUNK_DIR, exist_ok=True)
    t_start = time.time()

    # ── Load & reproject ───────────────────────────────────────────────────────
    log.info(f"Loading {INPUT_PATH}")
    df  = pd.read_parquet(INPUT_PATH)
    log.info(f"Loaded {len(df):,} records")

    gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkb(df['geometry'], crs=WGS84_CRS))
    gdf = gdf.to_crs(MOLLWEIDE_CRS)
    gdf['polygon_total_area_m2'] = gdf.geometry.area
    log.info("Reprojected to Mollweide")

    wkb_mollweide = gdf.geometry.to_wkb()
    total_chunks  = (len(gdf) + CHUNK_SIZE - 1) // CHUNK_SIZE

    # ── Resume: find which chunks are already done ─────────────────────────────
    done_chunks = {
        int(f.replace("chunk_", "").replace(".parquet", ""))
        for f in os.listdir(CHUNK_DIR)
        if f.startswith("chunk_") and f.endswith(".parquet")
    }
    pending = [i for i in range(total_chunks) if i not in done_chunks]
    log.info(f"Chunks: total={total_chunks}  done={len(done_chunks)}  remaining={len(pending)}")

    # ── Process pending chunks ─────────────────────────────────────────────────
    if pending:
        tasks = {
            idx: (wkb_mollweide.iloc[idx * CHUNK_SIZE : (idx + 1) * CHUNK_SIZE].tolist(),
                  GHS_RASTER, MOLLWEIDE_CRS)
            for idx in pending
        }
        t2 = time.time()
        completed = len(done_chunks)

        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(_zonal_stats_worker, task): idx
                       for idx, task in tasks.items()}
            for future in as_completed(futures):
                idx     = futures[future]
                result  = future.result()
                i_start = idx * CHUNK_SIZE
                pd.DataFrame({
                    'row_idx'               : list(range(i_start, i_start + len(result))),
                    'urban_built_up_area_m2': result,
                }).to_parquet(os.path.join(CHUNK_DIR, f"chunk_{idx:04d}.parquet"), index=False)

                completed += 1
                elapsed = time.time() - t2
                pct  = completed / total_chunks * 100
                rate = (completed - len(done_chunks)) / elapsed * CHUNK_SIZE if elapsed > 0 else 0
                log.info(f"chunk {idx:04d} saved | {completed}/{total_chunks} ({pct:.1f}%) | "
                         f"{elapsed:.0f}s | ~{rate:.0f} rows/s")

        log.info(f"Zonal statistics complete in {time.time() - t2:.1f}s")
    else:
        log.info("All chunks already complete — loading from disk.")

    # ── Assemble final dataset ─────────────────────────────────────────────────
    log.info("Assembling final dataset from chunks...")
    urban_areas = []
    for idx in range(total_chunks):
        df_chunk = pd.read_parquet(os.path.join(CHUNK_DIR, f"chunk_{idx:04d}.parquet"))
        urban_areas.extend(df_chunk.sort_values('row_idx')['urban_built_up_area_m2'].tolist())

    output_df = df.copy()
    output_df['polygon_total_area_m2']  = gdf['polygon_total_area_m2'].values
    output_df['urban_built_up_area_m2'] = urban_areas
    output_df['urban_percentage'] = np.where(
        output_df['polygon_total_area_m2'] > 0,
        (output_df['urban_built_up_area_m2'] / output_df['polygon_total_area_m2']) * 100,
        0.0
    ).clip(0, 100)

    output_df.to_parquet(OUTPUT_PATH, index=False)
    log.info(f"Saved {len(output_df):,} rows → {OUTPUT_PATH}")
    log.info(f"Total runtime: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
