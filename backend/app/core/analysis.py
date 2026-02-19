"""
Pattern analysis — ported from garmin_training_analysis/pattern_analysis.py.

Three analyses:
1. HRV threshold that predicts bad training days
2. TSB fatigue limit before performance drops
3. Sleep vs next-day performance correlation
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def analyze_hrv_threshold(merged_df, pace_col="pace_min_per_km", hrv_col="hrv_7day_avg") -> dict:
    """Find the HRV value that best separates good/bad training days."""
    results = {"analysis": "HRV Threshold for Bad Training Days", "status": "insufficient_data"}

    run_days = merged_df.dropna(subset=[pace_col, hrv_col]).copy()
    run_days = run_days[run_days[pace_col] > 0]

    if len(run_days) < 15:
        results["message"] = f"Need at least 15 run days with HRV; found {len(run_days)}."
        return results

    median_pace = run_days[pace_col].median()
    std_pace = run_days[pace_col].std()
    bad_threshold = median_pace + 0.5 * std_pace

    if "daily_trimp" in run_days.columns:
        trimp_20th = run_days["daily_trimp"].quantile(0.2)
        run_days = run_days[run_days["daily_trimp"] > trimp_20th]

    run_days["is_bad_day"] = (run_days[pace_col] > bad_threshold).astype(int)

    hrv_values = run_days[hrv_col].dropna().sort_values()
    best_threshold = None
    best_accuracy = 0.0

    for hrv_thresh in np.percentile(hrv_values, np.arange(10, 90, 5)):
        predicted_bad = (run_days[hrv_col] < hrv_thresh).astype(int)
        accuracy = (predicted_bad == run_days["is_bad_day"]).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = hrv_thresh

    if len(run_days) > 5:
        corr, p_value = stats.pointbiserialr(run_days["is_bad_day"], run_days[hrv_col])
    else:
        corr, p_value = 0, 1

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
            f"you have a {best_accuracy * 100:.0f}% chance of a sub-par training day."
        )
    else:
        results["interpretation"] = (
            "HRV alone is not a strong predictor of your bad training days. "
            "Other factors (sleep, stress, nutrition) may dominate."
        )

    return results


def analyze_tsb_fatigue_limit(merged_df) -> dict:
    """Find how many consecutive negative-TSB days cause pace degradation."""
    results = {"analysis": "TSB Fatigue Limit", "status": "insufficient_data"}

    if "tsb" not in merged_df.columns or "pace_min_per_km" not in merged_df.columns:
        results["message"] = "Need both TSB and pace data."
        return results

    df = merged_df.dropna(subset=["tsb"]).copy()
    if len(df) < 30:
        results["message"] = f"Need at least 30 days of TSB data; found {len(df)}."
        return results

    df["neg_tsb"] = (df["tsb"] < 0).astype(int)
    df["streak_group"] = (df["neg_tsb"] != df["neg_tsb"].shift()).cumsum()
    df["streak_length"] = df.groupby("streak_group").cumcount() + 1
    df.loc[df["neg_tsb"] == 0, "streak_length"] = 0

    run_days = df.dropna(subset=["pace_min_per_km"])
    run_days = run_days[run_days["pace_min_per_km"] > 0]

    if len(run_days) < 15:
        results["message"] = "Not enough running days with TSB data."
        return results

    baseline_pace = run_days["pace_min_per_km"].median()

    streak_buckets = [
        (0, 0, "Fresh (TSB >= 0)"), (1, 3, "1-3 days negative"),
        (4, 7, "4-7 days negative"), (8, 14, "8-14 days negative"),
        (15, 21, "15-21 days negative"), (22, 999, "22+ days negative"),
    ]

    bucket_analysis = []
    fatigue_limit = None

    for low, high, label in streak_buckets:
        bucket = run_days[(run_days["streak_length"] >= low) & (run_days["streak_length"] <= high)]
        if len(bucket) >= 3:
            avg_pace = bucket["pace_min_per_km"].mean()
            pct_diff = ((avg_pace - baseline_pace) / baseline_pace) * 100
            bucket_analysis.append({
                "streak_range": label, "num_runs": len(bucket),
                "avg_pace": round(avg_pace, 2), "pace_change_pct": round(pct_diff, 1),
            })
            if pct_diff > 5 and fatigue_limit is None:
                fatigue_limit = low

    valid = run_days.dropna(subset=["pace_min_per_km", "streak_length"])
    corr, p_val = stats.pearsonr(valid["streak_length"], valid["pace_min_per_km"]) if len(valid) > 10 else (0, 1)

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
            f"Performance degrades significantly after {fatigue_limit} consecutive "
            f"days of negative TSB. Schedule recovery before this threshold."
        )
    else:
        results["interpretation"] = "No clear fatigue threshold detected."

    return results


def analyze_sleep_performance_correlation(merged_df) -> dict:
    """Correlate sleep metrics with next-day running pace (1/2/3-day lags)."""
    results = {"analysis": "Sleep vs Next-Day Performance", "status": "insufficient_data"}

    sleep_metrics = ["sleep_score", "sleep_hours", "deep_sleep_hours",
                     "deep_sleep_pct", "rem_sleep_hours", "rem_sleep_pct"]
    available = [m for m in sleep_metrics if m in merged_df.columns]

    if not available:
        results["message"] = "No sleep metrics available."
        return results
    if "pace_min_per_km" not in merged_df.columns:
        results["message"] = "No pace data available."
        return results

    df = merged_df.copy()
    correlations = []

    for metric in available:
        for lag in [1, 2, 3]:
            shifted_pace = df["pace_min_per_km"].shift(-lag)
            valid = df[[metric]].copy()
            valid["next_pace"] = shifted_pace
            valid = valid.dropna()
            valid = valid[valid["next_pace"] > 0]
            if len(valid) < 10:
                continue
            corr, p_val = stats.pearsonr(valid[metric], valid["next_pace"])
            correlations.append({
                "sleep_metric": metric, "lag_days": lag,
                "correlation": round(corr, 3), "p_value": round(p_val, 4),
                "significant": p_val < 0.05, "n_observations": len(valid),
                "direction": "better sleep → faster" if corr < 0 else "better sleep → slower",
            })

    if not correlations:
        results["message"] = "Not enough overlapping sleep and pace data."
        return results

    corr_df = pd.DataFrame(correlations)
    sig = corr_df[corr_df["significant"]]
    best = sig.loc[sig["correlation"].idxmin()] if not sig.empty else corr_df.loc[corr_df["correlation"].abs().idxmax()]

    practical_effects = []
    for metric in available:
        valid = df[[metric, "pace_min_per_km"]].dropna()
        valid = valid[valid["pace_min_per_km"] > 0]
        if len(valid) < 10:
            continue
        poor = valid[valid[metric] <= valid[metric].quantile(0.25)]["pace_min_per_km"].mean()
        good = valid[valid[metric] >= valid[metric].quantile(0.75)]["pace_min_per_km"].mean()
        if poor > 0 and good > 0:
            practical_effects.append({
                "metric": metric,
                "poor_sleep_avg_pace": round(poor, 2),
                "good_sleep_avg_pace": round(good, 2),
                "pace_difference_sec_per_km": round((poor - good) * 60, 1),
            })

    results.update({
        "status": "complete",
        "all_correlations": correlations,
        "strongest_predictor": {
            "metric": best["sleep_metric"], "lag_days": int(best["lag_days"]),
            "correlation": best["correlation"], "p_value": best["p_value"],
        },
        "practical_effects": practical_effects,
        "interpretation": (
            f"Strongest link: {best['sleep_metric']} (r={best['correlation']:.3f}, p={best['p_value']:.4f})."
            if best["p_value"] < 0.05 else "No statistically significant sleep-performance link found."
        ),
    })

    return results


def run_all_analyses(merged_df) -> dict:
    """Execute all three pattern analyses."""
    report = {}
    for name, fn in [
        ("hrv_threshold", analyze_hrv_threshold),
        ("tsb_fatigue_limit", analyze_tsb_fatigue_limit),
        ("sleep_performance", analyze_sleep_performance_correlation),
    ]:
        try:
            report[name] = fn(merged_df)
        except Exception as e:
            logger.error("Analysis %s failed: %s", name, e)
            report[name] = {"analysis": name, "status": "error", "error": str(e)}
    return report
