#!/usr/bin/env python3
"""
Garmin Training Analysis for Ultramarathon Runners
===================================================
Main entry point that orchestrates data fetching, processing,
visualization, and pattern analysis.

Usage:
    # Set credentials via environment variables:
    export GARMIN_EMAIL="your@email.com"
    export GARMIN_PASSWORD="your_password"

    # Or create a .env file in the project root:
    echo 'GARMIN_EMAIL=your@email.com' > .env
    echo 'GARMIN_PASSWORD=your_password' >> .env

    # Run the analysis:
    python -m garmin_training_analysis.main

    # Or with options:
    python -m garmin_training_analysis.main --months 6 --output ./my_output

    # Skip data fetch and reuse cached CSVs:
    python -m garmin_training_analysis.main --skip-fetch --output ./my_output

Architecture:
    garmin_client.py     → API authentication and raw data fetching
    data_processing.py   → DataFrame transformation and metric calculation
    visualizations.py    → Chart generation (4 key training charts)
    pattern_analysis.py  → Statistical pattern detection and reporting
    main.py              → This file: orchestration and CLI interface
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from garmin_training_analysis.garmin_client import (
    authenticate,
    fetch_activities,
    fetch_activity_hr_zones,
    fetch_body_battery,
    fetch_hrv_data,
    fetch_sleep_data,
    fetch_stress_data,
    fetch_training_status,
)
from garmin_training_analysis.data_processing import (
    calculate_training_load,
    merge_all_data,
    process_activities,
    process_body_battery,
    process_hrv_data,
    process_sleep_data,
    process_stress_data,
    process_training_status,
)
from garmin_training_analysis.visualizations import generate_all_charts
from garmin_training_analysis.pattern_analysis import (
    format_report,
    run_all_analyses,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Garmin Training Analysis for Ultramarathon Runners",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--months", type=int, default=6,
        help="Number of months of historical data to analyze (default: 6)",
    )
    parser.add_argument(
        "--output", type=str, default="output",
        help="Output directory for charts and CSV files (default: output)",
    )
    parser.add_argument(
        "--skip-fetch", action="store_true",
        help="Skip API fetch and reuse cached CSV files from the output directory",
    )
    parser.add_argument(
        "--no-charts", action="store_true",
        help="Skip chart generation (useful for headless environments)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def fetch_all_data(client, months, output_dir):
    """
    Fetch all data types from Garmin Connect and save raw data to CSVs.

    Each data type is fetched independently with error handling so that
    a failure in one (e.g., no Body Battery on older devices) doesn't
    prevent the rest from completing.

    A small delay between API calls avoids rate limiting.

    Args:
        client: Authenticated Garmin client.
        months: Number of months of history.
        output_dir: Directory to save CSV files.

    Returns:
        dict: Raw data from each endpoint.
    """
    days = months * 30
    raw_data = {}

    # --- Activities ---
    logger.info("Fetching running activities...")
    try:
        raw_data["activities"] = fetch_activities(client, months=months)
        logger.info("  Found %d running activities.", len(raw_data["activities"]))

        # Fetch HR zones for each activity (with rate limiting)
        hr_zones = {}
        for act in raw_data["activities"][:100]:  # Cap at 100 to avoid rate limits
            act_id = act.get("activityId")
            if act_id:
                zones = fetch_activity_hr_zones(client, act_id)
                if zones:
                    hr_zones[act_id] = zones
                time.sleep(0.3)  # Gentle rate limiting
        raw_data["hr_zones"] = hr_zones
        logger.info("  Fetched HR zones for %d activities.", len(hr_zones))
    except Exception as e:
        logger.error("Failed to fetch activities: %s", e)
        raw_data["activities"] = []
        raw_data["hr_zones"] = {}

    time.sleep(1)

    # --- HRV ---
    logger.info("Fetching HRV data...")
    try:
        raw_data["hrv"] = fetch_hrv_data(client, days=days)
        logger.info("  Found %d HRV records.", len(raw_data["hrv"]))
    except Exception as e:
        logger.error("Failed to fetch HRV data: %s", e)
        raw_data["hrv"] = []

    time.sleep(1)

    # --- Sleep ---
    logger.info("Fetching sleep data...")
    try:
        raw_data["sleep"] = fetch_sleep_data(client, days=days)
        logger.info("  Found %d sleep records.", len(raw_data["sleep"]))
    except Exception as e:
        logger.error("Failed to fetch sleep data: %s", e)
        raw_data["sleep"] = []

    time.sleep(1)

    # --- Stress ---
    logger.info("Fetching stress data...")
    try:
        raw_data["stress"] = fetch_stress_data(client, days=days)
        logger.info("  Found %d stress records.", len(raw_data["stress"]))
    except Exception as e:
        logger.error("Failed to fetch stress data: %s", e)
        raw_data["stress"] = []

    time.sleep(1)

    # --- Body Battery ---
    logger.info("Fetching Body Battery data...")
    try:
        raw_data["body_battery"] = fetch_body_battery(client, days=days)
        logger.info("  Found %d Body Battery records.", len(raw_data["body_battery"]))
    except Exception as e:
        logger.error("Failed to fetch Body Battery data: %s", e)
        raw_data["body_battery"] = []

    time.sleep(1)

    # --- Training Status ---
    logger.info("Fetching Training Status data...")
    try:
        raw_data["training_status"] = fetch_training_status(client, days=days)
        logger.info("  Found %d training status records.", len(raw_data["training_status"]))
    except Exception as e:
        logger.error("Failed to fetch training status: %s", e)
        raw_data["training_status"] = []

    # Save raw data as JSON for debugging/reprocessing
    raw_path = Path(output_dir) / "raw_data.json"
    try:
        with open(raw_path, "w") as f:
            json.dump(raw_data, f, default=str, indent=2)
        logger.info("Raw data saved to %s", raw_path)
    except Exception as e:
        logger.warning("Could not save raw data JSON: %s", e)

    return raw_data


def process_all_data(raw_data):
    """
    Process all raw data into structured DataFrames.

    Returns:
        dict: Processed DataFrames for each data type.
    """
    processed = {}

    logger.info("Processing activities...")
    processed["activities"] = process_activities(
        raw_data.get("activities", []),
        hr_zones_map=raw_data.get("hr_zones", {}),
    )
    logger.info("  %d activities processed.", len(processed["activities"]))

    logger.info("Calculating training load metrics (ATL/CTL/TSB)...")
    processed["training_load"] = calculate_training_load(processed["activities"])
    if not processed["training_load"].empty:
        latest = processed["training_load"].iloc[-1]
        logger.info("  Latest ATL=%.1f, CTL=%.1f, TSB=%.1f",
                     latest.get("atl", 0), latest.get("ctl", 0), latest.get("tsb", 0))

    logger.info("Processing HRV data...")
    processed["hrv"] = process_hrv_data(raw_data.get("hrv", []))
    logger.info("  %d HRV records processed.", len(processed["hrv"]))

    logger.info("Processing sleep data...")
    processed["sleep"] = process_sleep_data(raw_data.get("sleep", []))
    logger.info("  %d sleep records processed.", len(processed["sleep"]))

    logger.info("Processing stress data...")
    processed["stress"] = process_stress_data(raw_data.get("stress", []))
    logger.info("  %d stress records processed.", len(processed["stress"]))

    logger.info("Processing Body Battery data...")
    processed["body_battery"] = process_body_battery(raw_data.get("body_battery", []))
    logger.info("  %d Body Battery records processed.", len(processed["body_battery"]))

    logger.info("Processing training status...")
    processed["training_status"] = process_training_status(
        raw_data.get("training_status", [])
    )
    logger.info("  %d training status records processed.",
                len(processed["training_status"]))

    return processed


def save_csvs(processed, output_dir):
    """
    Save all processed DataFrames to CSV files.

    CSV files use the date as the first column for easy import into
    spreadsheet tools, R, or other analysis environments.

    Args:
        processed: Dict of DataFrames from process_all_data().
        output_dir: Output directory path.

    Returns:
        list[Path]: Paths to saved CSV files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved = []

    csv_mapping = {
        "activities": "activities.csv",
        "training_load": "training_load_daily.csv",
        "hrv": "hrv_daily.csv",
        "sleep": "sleep_daily.csv",
        "stress": "stress_daily.csv",
        "body_battery": "body_battery_daily.csv",
        "training_status": "training_status.csv",
    }

    for key, filename in csv_mapping.items():
        df = processed.get(key, pd.DataFrame())
        if not df.empty:
            path = output_path / filename
            df.to_csv(path, index=False)
            saved.append(path)
            logger.info("Saved %s (%d rows)", path, len(df))

    return saved


def load_csvs(output_dir):
    """
    Load previously saved CSV files back into DataFrames.

    Used with --skip-fetch to reprocess data without hitting the API.

    Args:
        output_dir: Directory containing CSV files.

    Returns:
        dict: DataFrames loaded from CSV files.
    """
    output_path = Path(output_dir)
    processed = {}

    csv_mapping = {
        "activities": "activities.csv",
        "training_load": "training_load_daily.csv",
        "hrv": "hrv_daily.csv",
        "sleep": "sleep_daily.csv",
        "stress": "stress_daily.csv",
        "body_battery": "body_battery_daily.csv",
        "training_status": "training_status.csv",
    }

    for key, filename in csv_mapping.items():
        path = output_path / filename
        if path.exists():
            df = pd.read_csv(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            processed[key] = df
            logger.info("Loaded %s (%d rows)", path, len(df))
        else:
            processed[key] = pd.DataFrame()
            logger.warning("CSV not found: %s", path)

    return processed


def main():
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    output_dir = args.output
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Garmin Training Analysis for Ultramarathon Runners")
    print(f"  Analyzing {args.months} months of training data")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)
    print()

    # ---- Step 1: Fetch or load data ----
    if args.skip_fetch:
        logger.info("Skipping API fetch, loading from CSV files...")
        processed = load_csvs(output_dir)
    else:
        logger.info("Authenticating with Garmin Connect...")
        try:
            client = authenticate()
            logger.info("Authentication successful.")
        except Exception as e:
            logger.error("Authentication failed: %s", e)
            print(f"\nERROR: {e}")
            print("\nMake sure your credentials are set:")
            print("  export GARMIN_EMAIL='your@email.com'")
            print("  export GARMIN_PASSWORD='your_password'")
            sys.exit(1)

        logger.info("Fetching data from Garmin Connect...")
        raw_data = fetch_all_data(client, args.months, output_dir)

        logger.info("Processing data...")
        processed = process_all_data(raw_data)

    # ---- Step 2: Save CSVs ----
    logger.info("Saving CSV files...")
    csv_files = save_csvs(processed, output_dir)
    print(f"\nSaved {len(csv_files)} CSV files to {output_dir}/")
    for f in csv_files:
        print(f"  - {f.name}")

    # ---- Step 3: Merge all data ----
    logger.info("Merging all data sources...")
    merged = merge_all_data(
        processed.get("training_load", pd.DataFrame()),
        processed.get("hrv", pd.DataFrame()),
        processed.get("sleep", pd.DataFrame()),
        processed.get("stress", pd.DataFrame()),
        processed.get("body_battery", pd.DataFrame()),
        processed.get("training_status", pd.DataFrame()),
    )

    if not merged.empty:
        merged_path = Path(output_dir) / "merged_daily.csv"
        merged.to_csv(merged_path, index=False)
        logger.info("Saved merged dataset: %s (%d rows)", merged_path, len(merged))
    else:
        logger.warning("Merged dataset is empty — check data availability.")

    # ---- Step 4: Generate charts ----
    if not args.no_charts and not merged.empty:
        logger.info("Generating visualizations...")
        charts = generate_all_charts(merged, output_dir)
        print(f"\nGenerated {len(charts)} charts:")
        for c in charts:
            print(f"  - {c}")
    elif args.no_charts:
        print("\nChart generation skipped (--no-charts).")

    # ---- Step 5: Pattern analysis ----
    if not merged.empty:
        logger.info("Running pattern analysis...")
        report = run_all_analyses(merged)
        report_text = format_report(report)

        print("\n" + report_text)

        # Save report
        report_path = Path(output_dir) / "analysis_report.txt"
        with open(report_path, "w") as f:
            f.write(report_text)
            f.write(f"\n\nGenerated: {datetime.now().isoformat()}\n")
        logger.info("Report saved to %s", report_path)

        # Save report as JSON for programmatic access
        report_json_path = Path(output_dir) / "analysis_report.json"
        with open(report_json_path, "w") as f:
            json.dump(report, f, default=str, indent=2)
        logger.info("JSON report saved to %s", report_json_path)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  Analysis complete!")
    print(f"  All outputs saved to: {output_dir}/")
    print()
    print("  Files generated:")
    print("    CSV data files:      For import into Excel/R/Python")
    print("    PNG chart files:     Training analysis visualizations")
    print("    analysis_report.txt: Human-readable pattern analysis")
    print("    analysis_report.json: Machine-readable results")
    print("    raw_data.json:       Raw API responses for debugging")
    print()
    print("  Quick tips for 100km/week ultra training:")
    print("    - Re-run weekly to track trends")
    print("    - Use --skip-fetch to regenerate charts from cached data")
    print("    - Check TSB before planning race-pace workouts")
    print("    - Pair HRV drops with sleep data to find the root cause")
    print("=" * 60)


if __name__ == "__main__":
    main()
