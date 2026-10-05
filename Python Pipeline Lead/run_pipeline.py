import subprocess
import sys
from pathlib import Path


# ============================================================
# NORTHSTAR END-TO-END REPRODUCIBLE PIPELINE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


def run_step(script_name):
    print("\n" + "=" * 70)
    print(f"RUNNING: {script_name}")
    print("=" * 70)

    script_path = BASE_DIR / script_name

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR
    )

    if result.returncode != 0:
        print("\nPIPELINE FAILED")
        print(f"Failed step: {script_name}")
        sys.exit(result.returncode)

    print(f"\nCompleted: {script_name}")


def main():

    print("=" * 70)
    print("NORTHSTAR END-TO-END REPRODUCIBLE PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: DATA CLEANING
    # --------------------------------------------------------

    run_step("clean_pipeline.py")

    # --------------------------------------------------------
    # STEP 2: DATA VALIDATION
    # --------------------------------------------------------

    run_step("validate_clean.py")

    # --------------------------------------------------------
    # STEP 3: FORECASTING
    # --------------------------------------------------------

    run_step("forecasting.py")

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nMain outputs:")
    print("- northstar_clean.csv")
    print("- cleaning_log.csv")
    print("- forecast_outputs/forecast_predictions.csv")
    print("- forecast_outputs/forecast_metrics.csv")
    print("- forecast_outputs/feature_importance.csv")
    print("- forecast_outputs/forecast_summary.txt")


if __name__ == "__main__":
    main()