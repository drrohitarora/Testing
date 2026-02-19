"""
Pattern Analysis
================
Statistical analysis of personal training data to identify actionable
patterns specific to the individual athlete.

This module answers three key questions for ultramarathon runners:

1. HRV Threshold: What HRV value predicts your bad training days?
   → Uses logistic regression on HRV vs. "bad day" classification

2. TSB Fatigue Limit: How many days of negative TSB before performance drops?
   → Analyzes cumulative negative TSB streaks vs. pace degradation

3. Sleep-Performance Correlation: How does sleep affect next-day pace?
   → Lagged correlation analysis between sleep metrics and running pace
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def analyze_hrv_threshold(merged_df, pace_col="pace_min_per_km", hrv_col="hrv_7day_avg"):
    """
    Determine the HRV threshold that predicts bad training days.

    Methodology:
    ------------
    1. Define "bad training day" as a run where pace is slower than
       your personal median pace by more than 1 standard deviation,
       excluding intentionally easy recovery runs (very low TRIMP).
    2. For each day with both HRV and pace data, classify as good/bad.
    3. Use a simple threshold search: find the HRV value that best
       separates good days from bad days (maximizes accuracy).
    4. Report the threshold and its predictive accuracy.

    Why this matters for ultra runners:
    - Knowing your personal HRV floor lets you auto-regulate training
    - On days below the threshold, switch planned quality sessions to easy runs
    - Over time, a rising threshold indicates improving fitness

    Args:
        merged_df: Unified daily DataFrame.
        pace_col: Column name for pace data.
        hrv_col: Column name for HRV data.

    Returns:
        dict: Analysis results including threshold, accuracy, and stats.
    """
    results = {
        "analysis": "HRV Threshold for Bad Training Days",
        "status": "insufficient_data",
    }

    # Filter to days with both HRV and running data
    run_days = merged_df.dropna(subset=[pace_col, hrv_col]).copy()
    run_days = run_days[run_days[pace_col] > 0]

    if len(run_days) < 15:
        results["message"] = f"Need at least 15 run days with HRV; found {len(run_days)}."
        return results

    # Classify "bad" days: pace > median + 0.5*std
    # (0.5 std is more sensitive than 1 std for detecting subtle performance drops)
    median_pace = run_days[pace_col].median()
    std_pace = run_days[pace_col].std()
    bad_threshold = median_pace + 0.5 * std_pace

    # Exclude very easy recovery runs (bottom 20% by TRIMP) to avoid
    # classifying intentional easy runs as "bad"
    if "daily_trimp" in run_days.columns:
        trimp_20th = run_days["daily_trimp"].quantile(0.2)
        run_days = run_days[run_days["daily_trimp"] > trimp_20th]

    run_days["is_bad_day"] = (run_days[pace_col] > bad_threshold).astype(int)

    # Search for optimal HRV threshold
    hrv_values = run_days[hrv_col].dropna().sort_values()
    best_threshold = None
    best_accuracy = 0.0

    for hrv_thresh in np.percentile(hrv_values, np.arange(10, 90, 5)):
        # Predict: HRV below threshold → bad day
        predicted_bad = (run_days[hrv_col] < hrv_thresh).astype(int)
        accuracy = (predicted_bad == run_days["is_bad_day"]).mean()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = hrv_thresh

    # Statistical significance: point-biserial correlation
    if len(run_days) > 5:
        corr, p_value = stats.pointbiserialr(
            run_days["is_bad_day"], run_days[hrv_col]
        )
    else:
        corr, p_value = 0, 1

    # Stats for good vs bad days
    good_days = run_days[run_days["is_bad_day"] == 0]
    bad_days = run_days[run_days["is_bad_day"] == 1]

    results.update({
        "status": "complete",
        "hrv_threshold": round(best_threshold, 1) if best_threshold else None,
        "prediction_accuracy": round(best_accuracy * 100, 1),
        "correlation": round(corr, 3),
        "p_value": round(p_value, 4),
        "statistically_significant": p_value < 0.05,
        "total_run_days": len(run_days),
        "bad_days_count": int(run_days["is_bad_day"].sum()),
        "good_day_avg_hrv": round(good_days[hrv_col].mean(), 1) if len(good_days) > 0 else None,
        "bad_day_avg_hrv": round(bad_days[hrv_col].mean(), 1) if len(bad_days) > 0 else None,
        "median_pace": round(median_pace, 2),
        "bad_pace_threshold": round(bad_threshold, 2),
        "interpretation": "",
    })

    if best_threshold and best_accuracy > 0.55:
        results["interpretation"] = (
            f"When your 7-day HRV average drops below {best_threshold:.0f} ms, "
            f"you have a {best_accuracy * 100:.0f}% chance of having a sub-par "
            f"training day (pace slower than {bad_threshold:.2f} min/km). "
            f"On good days your HRV averages {results['good_day_avg_hrv']} ms "
            f"vs {results['bad_day_avg_hrv']} ms on bad days."
        )
    else:
        results["interpretation"] = (
            "HRV alone is not a strong predictor of your bad training days. "
            "This could mean: (a) your training is well-regulated, (b) other "
            "factors (sleep, stress, nutrition) dominate, or (c) more data is needed."
        )

    return results


def analyze_tsb_fatigue_limit(merged_df):
    """
    Determine how many consecutive days of negative TSB lead to performance drops.

    Methodology:
    ------------
    1. Track streaks of consecutive negative TSB days.
    2. For runs that occur during these streaks, measure pace degradation
       relative to the athlete's baseline.
    3. Find the streak length where pace degradation becomes significant
       (> 5% slower than baseline).

    Why this matters for ultra runners:
    - Ultra training requires sustained overload, so some negative TSB is normal
    - But there's a personal tipping point where fatigue outpaces adaptation
    - Knowing your limit helps plan recovery weeks before performance collapses

    Returns:
        dict: Analysis results including fatigue limit and pace trends.
    """
    results = {
        "analysis": "TSB Fatigue Limit",
        "status": "insufficient_data",
    }

    if "tsb" not in merged_df.columns or "pace_min_per_km" not in merged_df.columns:
        results["message"] = "Need both TSB and pace data."
        return results

    df = merged_df.dropna(subset=["tsb"]).copy()
    if len(df) < 30:
        results["message"] = f"Need at least 30 days of TSB data; found {len(df)}."
        return results

    # Calculate negative TSB streaks
    df["neg_tsb"] = (df["tsb"] < 0).astype(int)
    # Group consecutive negative days
    df["streak_group"] = (df["neg_tsb"] != df["neg_tsb"].shift()).cumsum()
    df["streak_length"] = df.groupby("streak_group").cumcount() + 1
    df.loc[df["neg_tsb"] == 0, "streak_length"] = 0

    # Filter to running days
    run_days = df.dropna(subset=["pace_min_per_km"])
    run_days = run_days[run_days["pace_min_per_km"] > 0]

    if len(run_days) < 15:
        results["message"] = "Not enough running days with TSB data."
        return results

    baseline_pace = run_days["pace_min_per_km"].median()

    # Analyze pace by streak length buckets
    streak_buckets = [
        (0, 0, "Fresh (TSB >= 0)"),
        (1, 3, "1-3 days negative"),
        (4, 7, "4-7 days negative"),
        (8, 14, "8-14 days negative"),
        (15, 21, "15-21 days negative"),
        (22, 999, "22+ days negative"),
    ]

    bucket_analysis = []
    fatigue_limit = None

    for low, high, label in streak_buckets:
        bucket = run_days[(run_days["streak_length"] >= low) &
                          (run_days["streak_length"] <= high)]
        if len(bucket) >= 3:
            avg_pace = bucket["pace_min_per_km"].mean()
            pace_pct_diff = ((avg_pace - baseline_pace) / baseline_pace) * 100

            bucket_analysis.append({
                "streak_range": label,
                "num_runs": len(bucket),
                "avg_pace": round(avg_pace, 2),
                "pace_change_pct": round(pace_pct_diff, 1),
            })

            # Detect when pace degrades > 5% from baseline
            if pace_pct_diff > 5 and fatigue_limit is None:
                fatigue_limit = low

    # Correlation between streak length and pace
    valid = run_days.dropna(subset=["pace_min_per_km", "streak_length"])
    if len(valid) > 10:
        corr, p_val = stats.pearsonr(valid["streak_length"], valid["pace_min_per_km"])
    else:
        corr, p_val = 0, 1

    results.update({
        "status": "complete",
        "baseline_pace": round(baseline_pace, 2),
        "fatigue_limit_days": fatigue_limit,
        "streak_analysis": bucket_analysis,
        "streak_pace_correlation": round(corr, 3),
        "correlation_p_value": round(p_val, 4),
        "max_observed_streak": int(df["streak_length"].max()),
        "avg_negative_tsb": round(df[df["tsb"] < 0]["tsb"].mean(), 1),
        "interpretation": "",
    })

    if fatigue_limit:
        results["interpretation"] = (
            f"Your performance starts degrading significantly after "
            f"{fatigue_limit} consecutive days of negative TSB. Your baseline "
            f"pace is {baseline_pace:.2f} min/km, and after {fatigue_limit}+ "
            f"days of accumulated fatigue, pace slows by >5%. Consider "
            f"scheduling recovery days before reaching this threshold."
        )
    else:
        results["interpretation"] = (
            "No clear fatigue threshold detected in your data. This could mean: "
            "(a) you recover well between hard efforts, (b) you self-regulate "
            "effectively, or (c) more data with sustained high-load periods is needed."
        )

    return results


def analyze_sleep_performance_correlation(merged_df):
    """
    Analyze the correlation between sleep metrics and next-day running pace.

    Methodology:
    ------------
    1. Align each night's sleep data with the following day's training.
    2. Calculate Pearson correlations between sleep metrics and pace.
    3. Also test 2-day and 3-day lagged effects (accumulated sleep debt).
    4. Report which sleep metric has the strongest predictive power.

    Why this matters for ultra runners:
    - Sleep is the primary recovery mechanism for high-volume training
    - Understanding which sleep metric matters most for YOUR performance
      lets you prioritize (e.g., total duration vs. deep sleep %)
    - Lagged effects reveal how quickly sleep debt accumulates

    Returns:
        dict: Correlation analysis results.
    """
    results = {
        "analysis": "Sleep vs Next-Day Performance",
        "status": "insufficient_data",
    }

    sleep_metrics = ["sleep_score", "sleep_hours", "deep_sleep_hours",
                     "deep_sleep_pct", "rem_sleep_hours", "rem_sleep_pct"]

    available_metrics = [m for m in sleep_metrics if m in merged_df.columns]
    if not available_metrics:
        results["message"] = "No sleep metrics available."
        return results

    if "pace_min_per_km" not in merged_df.columns:
        results["message"] = "No pace data available."
        return results

    df = merged_df.copy()
    correlations = []

    for metric in available_metrics:
        for lag in [1, 2, 3]:
            # Shift sleep data: lag=1 means last night's sleep → today's run
            shifted_pace = df["pace_min_per_km"].shift(-lag)
            valid = df[[metric]].copy()
            valid["next_pace"] = shifted_pace
            valid = valid.dropna()
            valid = valid[valid["next_pace"] > 0]

            if len(valid) < 10:
                continue

            corr, p_val = stats.pearsonr(valid[metric], valid["next_pace"])

            correlations.append({
                "sleep_metric": metric,
                "lag_days": lag,
                "correlation": round(corr, 3),
                "p_value": round(p_val, 4),
                "significant": p_val < 0.05,
                "n_observations": len(valid),
                # Negative correlation means better sleep → faster pace (lower number)
                "direction": "better sleep → faster" if corr < 0 else "better sleep → slower",
            })

    if not correlations:
        results["message"] = "Not enough overlapping sleep and pace data."
        return results

    corr_df = pd.DataFrame(correlations)

    # Find the strongest predictor (most negative significant correlation,
    # since better sleep should predict lower/faster pace)
    significant = corr_df[corr_df["significant"]]
    if not significant.empty:
        best = significant.loc[significant["correlation"].idxmin()]
    else:
        best = corr_df.loc[corr_df["correlation"].abs().idxmax()]

    # Calculate practical effect: pace difference between good and bad sleep nights
    practical_effects = []
    for metric in available_metrics:
        valid = df[[metric, "pace_min_per_km"]].dropna()
        valid = valid[valid["pace_min_per_km"] > 0]
        if len(valid) < 10:
            continue

        q25 = valid[metric].quantile(0.25)
        q75 = valid[metric].quantile(0.75)
        poor_sleep = valid[valid[metric] <= q25]["pace_min_per_km"].mean()
        good_sleep = valid[valid[metric] >= q75]["pace_min_per_km"].mean()

        if poor_sleep > 0 and good_sleep > 0:
            pace_diff = poor_sleep - good_sleep
            practical_effects.append({
                "metric": metric,
                "poor_sleep_avg_pace": round(poor_sleep, 2),
                "good_sleep_avg_pace": round(good_sleep, 2),
                "pace_difference_sec_per_km": round(pace_diff * 60, 1),
            })

    results.update({
        "status": "complete",
        "all_correlations": correlations,
        "strongest_predictor": {
            "metric": best["sleep_metric"],
            "lag_days": int(best["lag_days"]),
            "correlation": best["correlation"],
            "p_value": best["p_value"],
        },
        "practical_effects": practical_effects,
        "interpretation": "",
    })

    metric_names = {
        "sleep_score": "overall sleep score",
        "sleep_hours": "total sleep duration",
        "deep_sleep_hours": "deep sleep hours",
        "deep_sleep_pct": "deep sleep percentage",
        "rem_sleep_hours": "REM sleep hours",
        "rem_sleep_pct": "REM sleep percentage",
    }

    best_name = metric_names.get(best["sleep_metric"], best["sleep_metric"])
    lag_desc = f"{'same' if best['lag_days'] == 1 else best['lag_days']}-day"

    if best["p_value"] < 0.05:
        results["interpretation"] = (
            f"Your strongest sleep-performance link is {best_name} "
            f"(r={best['correlation']:.3f}, p={best['p_value']:.4f}), "
            f"with a {lag_desc} lag effect. "
        )
        if practical_effects:
            pe = practical_effects[0]
            results["interpretation"] += (
                f"Practically, good sleep nights lead to paces averaging "
                f"{pe['good_sleep_avg_pace']:.2f} min/km vs {pe['poor_sleep_avg_pace']:.2f} "
                f"min/km after poor sleep — a difference of "
                f"{abs(pe['pace_difference_sec_per_km']):.0f} seconds per km."
            )
    else:
        results["interpretation"] = (
            f"No statistically significant sleep-performance correlation found. "
            f"Best candidate: {best_name} (r={best['correlation']:.3f}, "
            f"p={best['p_value']:.4f}). Your pace variation may be more "
            f"influenced by training plan structure than sleep quality."
        )

    return results


def run_all_analyses(merged_df):
    """
    Execute all pattern analyses and return a consolidated report.

    Args:
        merged_df: Unified daily DataFrame from merge_all_data().

    Returns:
        dict: Results from all three analyses.
    """
    report = {}

    analyses = [
        ("hrv_threshold", analyze_hrv_threshold),
        ("tsb_fatigue_limit", analyze_tsb_fatigue_limit),
        ("sleep_performance", analyze_sleep_performance_correlation),
    ]

    for name, fn in analyses:
        try:
            report[name] = fn(merged_df)
            logger.info("Completed analysis: %s — %s",
                        name, report[name].get("status", "unknown"))
        except Exception as e:
            logger.error("Failed analysis %s: %s", name, e)
            report[name] = {"analysis": name, "status": "error", "error": str(e)}

    return report


def format_report(report):
    """
    Format the analysis report as a human-readable string.

    Args:
        report: Dict from run_all_analyses().

    Returns:
        str: Formatted text report.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("  GARMIN TRAINING PATTERN ANALYSIS REPORT")
    lines.append("  Focused on Ultramarathon Endurance Metrics")
    lines.append("=" * 70)
    lines.append("")

    for key, analysis in report.items():
        title = analysis.get("analysis", key)
        status = analysis.get("status", "unknown")

        lines.append(f"--- {title} ---")
        lines.append(f"Status: {status}")
        lines.append("")

        if status == "complete":
            interpretation = analysis.get("interpretation", "")
            if interpretation:
                lines.append(f"  Finding: {interpretation}")
                lines.append("")

            # Print key metrics
            skip_keys = {"analysis", "status", "interpretation", "all_correlations",
                         "streak_analysis", "practical_effects", "strongest_predictor",
                         "message"}
            for k, v in analysis.items():
                if k not in skip_keys:
                    label = k.replace("_", " ").title()
                    lines.append(f"  {label}: {v}")

            # Print streak analysis table if present
            if "streak_analysis" in analysis:
                lines.append("")
                lines.append("  Pace by Negative TSB Streak Length:")
                lines.append(f"  {'Streak':25s} {'Runs':>5s} {'Avg Pace':>10s} {'Change':>8s}")
                for bucket in analysis["streak_analysis"]:
                    lines.append(
                        f"  {bucket['streak_range']:25s} "
                        f"{bucket['num_runs']:5d} "
                        f"{bucket['avg_pace']:10.2f} "
                        f"{bucket['pace_change_pct']:+7.1f}%"
                    )

            # Print practical sleep effects if present
            if "practical_effects" in analysis and analysis["practical_effects"]:
                lines.append("")
                lines.append("  Sleep Impact on Pace:")
                for pe in analysis["practical_effects"]:
                    lines.append(
                        f"  {pe['metric']:20s}: good sleep → "
                        f"{pe['good_sleep_avg_pace']:.2f}, poor sleep → "
                        f"{pe['poor_sleep_avg_pace']:.2f} min/km "
                        f"(Δ {abs(pe['pace_difference_sec_per_km']):.0f} sec/km)"
                    )

        elif status == "insufficient_data":
            lines.append(f"  {analysis.get('message', 'Not enough data.')}")

        elif status == "error":
            lines.append(f"  Error: {analysis.get('error', 'Unknown error')}")

        lines.append("")

    lines.append("=" * 70)
    lines.append("  Recommendations for 100km/week ultra training:")
    lines.append("  • Monitor HRV trend, not single-day values")
    lines.append("  • Schedule recovery weeks when TSB drops below your limit")
    lines.append("  • Prioritize sleep consistency over sleep duration")
    lines.append("  • Use Body Battery < 40 as a rest day trigger")
    lines.append("  • Track these patterns monthly — they shift with fitness")
    lines.append("=" * 70)

    return "\n".join(lines)
