"""
Visualizations
==============
Creates training analysis charts from the merged Garmin data.

All charts are saved as PNG files and use a consistent dark theme
suitable for endurance training dashboards.

Chart Design Philosophy:
- Dual-axis plots overlay related metrics to reveal correlations
- 7-day rolling averages smooth out day-to-day noise
- Color coding: green = good/recovered, red = fatigued/stressed
- Shaded regions highlight danger zones for key metrics
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Consistent style for all charts
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "atl": "#FF6B6B",       # Red — acute fatigue
    "ctl": "#4ECDC4",       # Teal — chronic fitness
    "tsb": "#45B7D1",       # Blue — stress balance
    "hrv": "#96CEB4",       # Green — HRV
    "sleep": "#DDA0DD",     # Plum — sleep
    "stress": "#FF8C42",    # Orange — stress
    "bb": "#FFD93D",        # Yellow — body battery
    "pace": "#6C5CE7",      # Purple — pace
    "distance": "#A8E6CF",  # Light green — distance
    "danger": "#FF4757",    # Bright red — danger zone
}


def setup_figure(title, figsize=(16, 8)):
    """Create a consistently styled figure."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45, ha="right")
    return fig, ax


def save_figure(fig, output_dir, filename):
    """Save figure with tight layout."""
    path = Path(output_dir) / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved chart: %s", path)
    return path


def plot_training_load_vs_hrv(merged_df, output_dir):
    """
    Chart 1: Training Load vs HRV Trends
    =====================================
    This is the most important chart for an ultra runner. It overlays:
    - ATL (acute training load / fatigue) on the left axis
    - CTL (chronic training load / fitness) on the left axis
    - TSB (stress balance) as a filled area chart
    - 7-day HRV average on the right axis

    What to look for:
    - HRV dropping while ATL rises → approaching overreaching
    - HRV staying stable during ATL increases → good adaptation
    - TSB deeply negative + falling HRV → take a recovery day
    """
    if merged_df.empty:
        logger.warning("No data for training load vs HRV chart.")
        return None

    fig, ax1 = setup_figure("Training Load vs HRV Trends")

    dates = merged_df["date"]

    # Left axis: Training load metrics
    if "atl" in merged_df.columns:
        ax1.plot(dates, merged_df["atl"], color=COLORS["atl"],
                 linewidth=2, label="ATL (7-day fatigue)", alpha=0.9)
    if "ctl" in merged_df.columns:
        ax1.plot(dates, merged_df["ctl"], color=COLORS["ctl"],
                 linewidth=2, label="CTL (28-day fitness)", alpha=0.9)

    # TSB as filled area (green when positive/fresh, red when negative/fatigued)
    if "tsb" in merged_df.columns:
        tsb = merged_df["tsb"]
        ax1.fill_between(dates, 0, tsb, where=(tsb >= 0),
                         color="#4ECDC4", alpha=0.2, label="TSB (fresh)")
        ax1.fill_between(dates, 0, tsb, where=(tsb < 0),
                         color="#FF6B6B", alpha=0.2, label="TSB (fatigued)")
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        # Danger zone: TSB below -40
        ax1.axhline(y=-40, color=COLORS["danger"], linestyle=":",
                     alpha=0.4, label="Overtraining risk (TSB < -40)")

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Training Load (TRIMP)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=9)

    # Right axis: HRV
    hrv_col = "hrv_7day_avg" if "hrv_7day_avg" in merged_df.columns else "hrv_value"
    if hrv_col in merged_df.columns:
        ax2 = ax1.twinx()
        hrv_data = merged_df[hrv_col].dropna()
        if not hrv_data.empty:
            ax2.plot(dates, merged_df[hrv_col], color=COLORS["hrv"],
                     linewidth=2.5, label="HRV (7-day avg)", alpha=0.9)

            # Shade HRV baseline zone if we have baseline data
            if "hrv_baseline" in merged_df.columns:
                baseline = merged_df["hrv_baseline"].dropna()
                if not baseline.empty:
                    avg_baseline = baseline.mean()
                    ax2.axhline(y=avg_baseline, color=COLORS["hrv"],
                                linestyle="--", alpha=0.4, label=f"HRV baseline ({avg_baseline:.0f})")

            ax2.set_ylabel("HRV (ms)", fontsize=12, color=COLORS["hrv"])
            ax2.tick_params(axis="y", labelcolor=COLORS["hrv"])
            ax2.legend(loc="upper right", fontsize=9)

    return save_figure(fig, output_dir, "training_load_vs_hrv.png")


def plot_sleep_vs_performance(merged_df, output_dir):
    """
    Chart 2: Sleep Quality vs Next-Day Performance
    ===============================================
    Overlays sleep quality metrics against next-day running performance.

    For ultra runners, this reveals:
    - Minimum sleep hours needed before quality training
    - Whether deep sleep % correlates with faster paces
    - If accumulated sleep debt affects performance (shifted correlation)

    We shift performance data by 1 day so each night's sleep aligns
    with the following day's training.
    """
    if merged_df.empty:
        logger.warning("No data for sleep vs performance chart.")
        return None

    fig, ax1 = setup_figure("Sleep Quality vs Next-Day Running Performance")

    dates = merged_df["date"]

    # Left axis: Sleep metrics
    has_sleep = False
    if "sleep_score" in merged_df.columns:
        sleep_7day = merged_df.get("sleep_score_7day", merged_df["sleep_score"])
        ax1.plot(dates, sleep_7day, color=COLORS["sleep"],
                 linewidth=2, label="Sleep Score (7-day avg)")
        has_sleep = True

    if "sleep_hours" in merged_df.columns:
        sleep_hrs_7day = merged_df.get("sleep_hours_7day", merged_df["sleep_hours"])
        ax1_twin = ax1.twinx()
        ax1_twin.plot(dates, sleep_hrs_7day, color="#87CEEB",
                      linewidth=1.5, linestyle="--", label="Sleep Hours (7-day avg)",
                      alpha=0.7)
        ax1_twin.set_ylabel("Sleep Hours", fontsize=12, color="#87CEEB")
        ax1_twin.tick_params(axis="y", labelcolor="#87CEEB")
        # Target line: 8 hours for ultra runners
        ax1_twin.axhline(y=8.0, color="#87CEEB", linestyle=":",
                         alpha=0.3, label="Target: 8h sleep")
        ax1_twin.legend(loc="center right", fontsize=9)
        has_sleep = True

    if not has_sleep:
        plt.close(fig)
        logger.warning("No sleep data available for chart.")
        return None

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Sleep Score", fontsize=12, color=COLORS["sleep"])
    ax1.tick_params(axis="y", labelcolor=COLORS["sleep"])

    # Overlay: Next-day pace (shifted by 1 day)
    if "pace_min_per_km" in merged_df.columns:
        pace_data = merged_df["pace_min_per_km"].shift(-1)  # Shift to align
        # Only plot days that have runs
        mask = pace_data.notna() & (pace_data > 0)
        if mask.any():
            ax_pace = ax1.twinx()
            ax_pace.spines["right"].set_position(("outward", 60))
            ax_pace.scatter(dates[mask], pace_data[mask], color=COLORS["pace"],
                            s=30, alpha=0.6, label="Next-day pace (min/km)")
            ax_pace.invert_yaxis()  # Lower pace = faster = better (top of chart)
            ax_pace.set_ylabel("Pace (min/km) — lower is faster ↑",
                               fontsize=10, color=COLORS["pace"])
            ax_pace.tick_params(axis="y", labelcolor=COLORS["pace"])
            ax_pace.legend(loc="lower right", fontsize=9)

    ax1.legend(loc="upper left", fontsize=9)

    return save_figure(fig, output_dir, "sleep_vs_performance.png")


def plot_body_battery_vs_workout(merged_df, output_dir):
    """
    Chart 3: Body Battery vs Workout Quality
    =========================================
    Shows how morning Body Battery predicts training quality.

    For ultra runners doing 100km/week, Body Battery is a practical
    daily readiness indicator:
    - Morning BB > 70: Good for quality sessions (tempo, intervals, long runs)
    - Morning BB 40-70: Easy/moderate runs only
    - Morning BB < 40: Rest day or very easy recovery jog

    Scatter plot with morning BB on x-axis, TRIMP on y-axis, colored by pace.
    """
    if merged_df.empty:
        logger.warning("No data for body battery chart.")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Body Battery vs Training Quality", fontsize=16,
                 fontweight="bold", y=0.98)

    # --- Left panel: Time series ---
    dates = merged_df["date"]

    bb_col = "bb_morning_7day" if "bb_morning_7day" in merged_df.columns else "bb_morning"
    has_bb = bb_col in merged_df.columns and merged_df[bb_col].notna().any()

    if has_bb:
        ax1.plot(dates, merged_df[bb_col], color=COLORS["bb"],
                 linewidth=2, label="Morning Body Battery (7-day avg)")
        ax1.axhspan(0, 40, alpha=0.1, color="red", label="Low energy zone")
        ax1.axhspan(70, 100, alpha=0.1, color="green", label="High energy zone")
        ax1.set_ylim(0, 105)

    if "daily_trimp" in merged_df.columns:
        ax1_twin = ax1.twinx()
        ax1_twin.bar(dates, merged_df["daily_trimp"], color=COLORS["atl"],
                      alpha=0.3, width=1, label="Daily TRIMP")
        ax1_twin.set_ylabel("TRIMP", fontsize=12, color=COLORS["atl"])
        ax1_twin.tick_params(axis="y", labelcolor=COLORS["atl"])
        ax1_twin.legend(loc="upper right", fontsize=9)

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Body Battery", fontsize=12, color=COLORS["bb"])
    ax1.tick_params(axis="y", labelcolor=COLORS["bb"])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    ax1.legend(loc="upper left", fontsize=9)

    # --- Right panel: Scatter plot ---
    bb_morning_col = "bb_morning"
    if bb_morning_col in merged_df.columns and "daily_trimp" in merged_df.columns:
        scatter_df = merged_df.dropna(subset=[bb_morning_col, "daily_trimp"])
        scatter_df = scatter_df[scatter_df["daily_trimp"] > 0]  # Only training days

        if not scatter_df.empty:
            scatter = ax2.scatter(
                scatter_df[bb_morning_col],
                scatter_df["daily_trimp"],
                c=scatter_df.get("pace_min_per_km", scatter_df["daily_trimp"]),
                cmap="RdYlGn_r",
                s=50, alpha=0.7, edgecolors="gray", linewidth=0.5
            )
            plt.colorbar(scatter, ax=ax2, label="Pace (min/km)")

            # Add trend line
            valid = scatter_df[[bb_morning_col, "daily_trimp"]].dropna()
            if len(valid) > 5:
                z = np.polyfit(valid[bb_morning_col], valid["daily_trimp"], 1)
                p = np.poly1d(z)
                x_range = np.linspace(valid[bb_morning_col].min(),
                                      valid[bb_morning_col].max(), 100)
                ax2.plot(x_range, p(x_range), color="gray", linestyle="--",
                         alpha=0.5, label="Trend")

            ax2.set_xlabel("Morning Body Battery", fontsize=12)
            ax2.set_ylabel("Daily TRIMP", fontsize=12)
            ax2.set_title("Higher Body Battery → Higher Quality Training?")
            ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, "Insufficient Body Battery +\nTRIMP data for scatter",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=12)

    return save_figure(fig, output_dir, "body_battery_vs_workout.png")


def plot_hrv_trends_with_training(merged_df, output_dir):
    """
    Chart 4: HRV Trends with Training Load Overlay
    ================================================
    A detailed view of HRV behavior in the context of training volume.

    This chart helps identify:
    - HRV suppression from overload (HRV drops during high-volume weeks)
    - Supercompensation (HRV rises above baseline after recovery periods)
    - Your personal HRV "floor" that predicts bad training days

    Three panels stacked vertically:
    1. HRV with baseline and 7-day average
    2. Daily and weekly training volume
    3. Resting heart rate trends (inverse of HRV)
    """
    if merged_df.empty:
        logger.warning("No data for HRV trends chart.")
        return None

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle("HRV Trends with Training Load Context", fontsize=16,
                 fontweight="bold", y=0.98)

    dates = merged_df["date"]

    # --- Panel 1: HRV ---
    if "hrv_value" in merged_df.columns:
        ax1.scatter(dates, merged_df["hrv_value"], color=COLORS["hrv"],
                    s=15, alpha=0.3, label="Daily HRV")
    if "hrv_7day_avg" in merged_df.columns:
        ax1.plot(dates, merged_df["hrv_7day_avg"], color=COLORS["hrv"],
                 linewidth=2.5, label="HRV 7-day avg")
    if "hrv_baseline" in merged_df.columns:
        baseline = merged_df["hrv_baseline"]
        if baseline.notna().any():
            ax1.plot(dates, baseline, color="gray", linestyle="--",
                     alpha=0.6, label="HRV baseline")
    ax1.set_ylabel("HRV (ms)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("Heart Rate Variability", fontsize=13)

    # --- Panel 2: Training volume ---
    if "daily_trimp" in merged_df.columns:
        ax2.bar(dates, merged_df["daily_trimp"], color=COLORS["atl"],
                alpha=0.5, width=1, label="Daily TRIMP")
    if "weekly_distance_km" in merged_df.columns:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(dates, merged_df["weekly_distance_km"],
                      color=COLORS["distance"], linewidth=2,
                      label="Weekly Distance (km)")
        ax2_twin.set_ylabel("Weekly km", fontsize=12, color=COLORS["distance"])
        ax2_twin.tick_params(axis="y", labelcolor=COLORS["distance"])
        ax2_twin.legend(loc="upper right", fontsize=9)
    ax2.set_ylabel("TRIMP", fontsize=12)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_title("Training Volume", fontsize=13)

    # --- Panel 3: Resting HR (should mirror HRV inversely) ---
    if "resting_hr" in merged_df.columns:
        ax3.scatter(dates, merged_df["resting_hr"], color="#FF6B6B",
                    s=15, alpha=0.3, label="Daily RHR")
    if "rhr_7day_avg" in merged_df.columns:
        ax3.plot(dates, merged_df["rhr_7day_avg"], color="#FF6B6B",
                 linewidth=2.5, label="RHR 7-day avg")
    ax3.set_ylabel("Resting HR (bpm)", fontsize=12)
    ax3.set_xlabel("Date", fontsize=12)
    ax3.legend(loc="upper left", fontsize=9)
    ax3.set_title("Resting Heart Rate (rises with fatigue)", fontsize=13)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")

    return save_figure(fig, output_dir, "hrv_trends_with_training.png")


def generate_all_charts(merged_df, output_dir="output"):
    """
    Generate all visualization charts and save to output directory.

    Args:
        merged_df: The unified DataFrame from merge_all_data().
        output_dir: Directory to save chart PNGs.

    Returns:
        list[Path]: Paths to generated chart files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    charts = []

    chart_fns = [
        plot_training_load_vs_hrv,
        plot_sleep_vs_performance,
        plot_body_battery_vs_workout,
        plot_hrv_trends_with_training,
    ]

    for fn in chart_fns:
        try:
            path = fn(merged_df, output_dir)
            if path:
                charts.append(path)
        except Exception as e:
            logger.error("Failed to generate chart %s: %s", fn.__name__, e)

    logger.info("Generated %d charts in %s", len(charts), output_dir)
    return charts
