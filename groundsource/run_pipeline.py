"""
Pipeline Runner — executes all heavy processing steps in sequence.

Each step runs in a separate subprocess, so memory is fully released between steps.
On failure the pipeline stops and reports which step failed.

Usage:
    python run_pipeline.py              # run all steps from the beginning
    python run_pipeline.py --from 02a   # skip to step 02a (01a already done)
    python run_pipeline.py --only 03a   # run only step 03a

All steps are idempotent: already-completed work is skipped automatically.
Progress and timing are written to outputs/pipeline.log
"""
import subprocess, sys, time, logging, os, argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent
LOG_PATH = BASE_DIR / "outputs" / "pipeline.log"
os.makedirs(BASE_DIR / "outputs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

STEPS = [
    ("01a", "run_01a.py", "Urban Areas Detection"),
    ("02a", "run_02a.py", "Pluvial Flood Detection (PFDI)"),
    ("03a", "run_03a.py", "Extract IMERG Precipitation Matrices"),
    ("04",  "run_04.py",  "No-Flood Event Addition"),
]


def run_step(name, script, description):
    script_path = BASE_DIR / script
    if not script_path.exists():
        log.error(f"Script not found: {script_path}")
        return False

    log.info("=" * 65)
    log.info(f"  STEP {name}: {description}")
    log.info("=" * 65)
    t = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR),
    )
    elapsed = time.time() - t

    if result.returncode == 0:
        log.info(f"Step {name} DONE in {elapsed/60:.1f} min")
        return True
    else:
        log.error(f"Step {name} FAILED (exit code {result.returncode}) after {elapsed/60:.1f} min")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run the Groundsource pipeline.")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument('--from', dest='from_step', metavar='STEP',
                       help='Start from this step, skipping earlier ones (e.g. 02a)')
    group.add_argument('--only', dest='only_step', metavar='STEP',
                       help='Run only this single step (e.g. 03a)')
    args = parser.parse_args()

    step_names = [s[0] for s in STEPS]

    if args.only_step:
        if args.only_step not in step_names:
            log.error(f"Unknown step '{args.only_step}'. Available: {step_names}")
            sys.exit(1)
        steps_to_run = [s for s in STEPS if s[0] == args.only_step]
    elif args.from_step:
        if args.from_step not in step_names:
            log.error(f"Unknown step '{args.from_step}'. Available: {step_names}")
            sys.exit(1)
        idx = step_names.index(args.from_step)
        steps_to_run = STEPS[idx:]
        log.info(f"Skipping steps before {args.from_step}")
    else:
        steps_to_run = STEPS

    log.info(f"Steps to run: {[s[0] for s in steps_to_run]}")
    t_total = time.time()

    for name, script, description in steps_to_run:
        success = run_step(name, script, description)
        if not success:
            log.error(f"Pipeline stopped at step {name}. Fix the error and re-run with --from {name}")
            sys.exit(1)

    total_min = (time.time() - t_total) / 60
    log.info("=" * 65)
    log.info(f"  PIPELINE COMPLETE in {total_min:.1f} min")
    log.info("=" * 65)


if __name__ == '__main__':
    main()
