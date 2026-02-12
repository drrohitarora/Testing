"""
Composite readiness score calculation.

Combines multiple recovery and load indicators into a single 0-100
readiness score tailored to ultramarathon runners.

Scoring model:
  readiness = weighted_sum(
      hrv_component,         # 30% — strongest single predictor of readiness
      tsb_component,         # 25% — training stress balance
      sleep_component,       # 20% — sleep quality + duration
      body_battery_component,# 15% — Garmin's own energy metric
      stress_component,      # 10% — daily stress levels
  )

Each component is normalized to 0-100 based on the athlete's personal
baseline (not population averages), since absolute HRV/HR values vary
hugely between individuals.
"""

import logging
from datetime import date

logger = logging.getLogger(__name__)


def compute_readiness_score(
    hrv_value: float | None = None,
    hrv_baseline: float | None = None,
    tsb: float | None = None,
    sleep_score: float | None = None,
    sleep_hours: float | None = None,
    body_battery: float | None = None,
    avg_stress: float | None = None,
    resting_hr: float | None = None,
) -> dict:
    """
    Calculate a composite readiness score from today's metrics.

    Returns a dict with:
      - readiness_score (0-100)
      - recommendation (text)
      - color ("green" / "yellow" / "red")
      - factors (breakdown of each input metric)
    """

    weights = {
        "hrv": 0.30,
        "tsb": 0.25,
        "sleep": 0.20,
        "body_battery": 0.15,
        "stress": 0.10,
    }

    components = {}
    available_weight = 0.0

    # --- HRV component ---
    # Score relative to personal baseline. HRV at baseline = 60 points,
    # HRV 20% above baseline = 100, HRV 30% below = 0.
    if hrv_value is not None:
        baseline = hrv_baseline if hrv_baseline else hrv_value  # Self-reference if no baseline
        if baseline > 0:
            ratio = hrv_value / baseline
            # Map ratio 0.7 → 0, 1.0 → 60, 1.2 → 100
            score = max(0, min(100, (ratio - 0.7) / 0.5 * 100))
            components["hrv"] = score
            available_weight += weights["hrv"]

    # --- TSB component ---
    # For ultra runners: TSB +15 = excellent (100), 0 = moderate (60),
    # -20 = tired (30), -40 = danger zone (0).
    if tsb is not None:
        # Map -40 → 0, 0 → 60, +15 → 100
        if tsb >= 0:
            score = min(100, 60 + (tsb / 15) * 40)
        else:
            score = max(0, 60 + (tsb / 40) * 60)
        components["tsb"] = score
        available_weight += weights["tsb"]

    # --- Sleep component ---
    # Blend of sleep score and sleep duration.
    sleep_scores = []
    if sleep_score is not None:
        sleep_scores.append(sleep_score)  # Already 0-100
    if sleep_hours is not None:
        # 8h = 80, 9h = 100, 6h = 40, <5h = 0
        hrs_score = max(0, min(100, (sleep_hours - 5) / 4 * 100))
        sleep_scores.append(hrs_score)
    if sleep_scores:
        components["sleep"] = sum(sleep_scores) / len(sleep_scores)
        available_weight += weights["sleep"]

    # --- Body Battery component ---
    # Already 0-100 from Garmin, use directly.
    if body_battery is not None:
        components["body_battery"] = max(0, min(100, body_battery))
        available_weight += weights["body_battery"]

    # --- Stress component ---
    # Invert: low stress = high readiness. Stress 0 → 100, stress 100 → 0.
    if avg_stress is not None:
        components["stress"] = max(0, min(100, 100 - avg_stress))
        available_weight += weights["stress"]

    # --- Compute weighted average ---
    if available_weight == 0:
        return {
            "readiness_score": 50.0,
            "recommendation": "Insufficient data — sync your Garmin to get a readiness score.",
            "color": "yellow",
            "factors": _build_factors(hrv_value, tsb, sleep_hours, sleep_score, body_battery, resting_hr, avg_stress),
        }

    # Re-normalize weights to sum to 1.0 for available components
    weighted_sum = 0.0
    for key, score in components.items():
        normalized_weight = weights[key] / available_weight
        weighted_sum += score * normalized_weight

    readiness = round(max(0, min(100, weighted_sum)), 1)

    # Classification
    if readiness >= 70:
        color = "green"
        recommendation = _green_recommendation(readiness, tsb, sleep_hours)
    elif readiness >= 40:
        color = "yellow"
        recommendation = _yellow_recommendation(readiness, components)
    else:
        color = "red"
        recommendation = _red_recommendation(readiness, components)

    return {
        "readiness_score": readiness,
        "recommendation": recommendation,
        "color": color,
        "factors": _build_factors(hrv_value, tsb, sleep_hours, sleep_score, body_battery, resting_hr, avg_stress),
    }


def _build_factors(hrv, tsb, sleep_hours, sleep_score, body_battery, resting_hr, avg_stress):
    return {
        "hrv": hrv,
        "tsb": tsb,
        "sleep_hours": sleep_hours,
        "sleep_score": sleep_score,
        "body_battery": body_battery,
        "resting_hr": resting_hr,
        "avg_stress": avg_stress,
    }


def _green_recommendation(readiness, tsb, sleep_hours):
    if readiness >= 85:
        return "Excellent readiness — great day for a quality session, long run, or race-pace effort."
    if tsb is not None and tsb > 10:
        return "Well rested. Good for tempo or interval work. Consider pushing the pace today."
    return "Ready for moderate-to-hard effort. A solid training day is possible."


def _yellow_recommendation(readiness, components):
    limiting = min(components, key=components.get) if components else "unknown"
    reasons = {
        "hrv": "HRV is below your baseline",
        "tsb": "accumulated training fatigue is high",
        "sleep": "sleep quality was subpar",
        "body_battery": "Body Battery is low",
        "stress": "stress levels are elevated",
    }
    reason = reasons.get(limiting, "multiple factors are moderate")
    return f"Moderate readiness — {reason}. Consider an easy run or zone 2 effort."


def _red_recommendation(readiness, components):
    if readiness < 20:
        return "Very low readiness — strong recommendation to rest or do light mobility work only."
    return "Low readiness — skip intensity today. Easy recovery jog at most, or take a full rest day."
