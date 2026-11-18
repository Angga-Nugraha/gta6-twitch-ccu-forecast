"""GTA VI Twitch viewer forecaster — entry for gta6-twitch-ccu-forecast-01.

Given the Twitch viewer count in hour 1 of a game's release day (UTC),
this program forecasts viewers for hours 2..24 as five quantiles
(5%, 25%, 50%, 75%, 95%), scored by pinball loss on the log scale.

How it works
-------------
The case file ships the release-day curves of comparable past launches.
Each comparable contributes a log-ratio curve

    r_h = ln(viewers_h / viewers_1),  h = 2..24,

which captures the *shape* of a release day (night dip, timezone ramp,
US prime-time peak) independent of the launch's size. Two views of the
conditional distribution of r_h given the target's hour-1 size v1 are
combined:

1. Kernel kNN — Gaussian-kernel weighted empirical quantiles of r_h,
   where a comparable's weight decays with the log-size distance between
   its hour-1 print and the target's. Launches of similar size tend to
   follow similar day shapes.
2. Log-size regression — a per-hour least-squares fit of r_h on
   ln(hour-1 viewers); the forecast is centred on the fitted value at the
   target's size, with spread taken from the residuals' quantiles.

The two views are blended in log space (0.75 kNN + 0.25 regression),
shifted by a +0.10 log bias correction, smoothed across neighbouring
hours (window +/-1), and forced non-decreasing. If the case ships no
usable comparable, the program falls back to the house persistence
baseline. Everything is a pure function of the case file: standard
library only, deterministic, no network, no clock, no randomness —
two runs write identical bytes.

Validation (leave-one-out over the 37 comparable launches with a full
24-hour release day, pinball loss on the log scale, the contest metric;
see MODEL.md for the full variant table):
  - GTA-like subset (comparable tier, hour-1 viewers >= 20k, n=17):
      this model 0.252 mean loss vs house baseline 0.347
  - hour-1 viewers >= 50k subset (n=13):
      this model 0.181 vs baseline 0.243
  - all 37 full-curve comparables: 0.396 vs baseline 0.930
  The model beats the house baseline on 14 of the 17 GTA-like launches.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0)
in python:3.13-slim, standard library only. It reads every <id>.in in
the input directory and writes <id>.out to the output directory:
{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}, 23 rows.
Outside the grader, MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in
for /input and /output.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports. (This entry is self-contained; the line
# is kept for parity with the kit starter.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard-normal quantiles matching LEVELS, for the persistence baseline.
BASELINE_Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Gaussian kernel bandwidth on ln(hour-1 viewers) for the kNN view.
KERNEL_BANDWIDTH = 0.8
# Blend weight of the kNN view vs the regression view (log space).
KNN_BLEND_WEIGHT = 0.75
# Log bias correction, from leave-one-out validation on the comparables.
LOG_BIAS_SHIFT = 0.10
# Hour-smoothing half-window for each quantile's log-curve.
SMOOTH_HALF_WINDOW = 1
# Minimum comparables at an hour for the regression view to be trusted;
# below this the hour falls back to the kNN view.
MIN_REGRESSION_POINTS = 8
# A comparable counts as usable with hour 1 plus at least this many hours.
MIN_CURVE_HOURS = 12


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def round_js_half_up(value: float) -> float | int:
    """Round to 2 decimals, half up, like JavaScript Math.round(x*100)/100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = value * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 100
    return int(rounded) if rounded.is_integer() else rounded


def interpolated_quantile(sorted_values, level: float) -> float:
    """Linear-interpolated quantile of an ascending list (type-7 style).

    Returns NaN for an empty list.
    """
    if not sorted_values:
        return float("nan")
    position = level * (len(sorted_values) - 1)
    lower = int(position)
    fraction = position - lower
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def weighted_empirical_quantile(value_weight_pairs, level: float) -> float:
    """Quantile of [(value, weight), ...] with values in ascending order.

    Walks the cumulative weight until it reaches level * total weight and
    returns that value. With no positive total weight, returns the median
    value instead.
    """
    total_weight = sum(weight for _, weight in value_weight_pairs)
    if total_weight <= 0:
        return value_weight_pairs[len(value_weight_pairs) // 2][0]
    threshold = level * total_weight
    accumulated = 0.0
    for value, weight in value_weight_pairs:
        accumulated += weight
        if accumulated >= threshold:
            return value
    return value_weight_pairs[-1][0]


def moving_average(values, half_window: int) -> list:
    """Each element replaced by the mean of itself and its neighbours.

    The window is +/-half_window, truncated at the ends of the list.
    """
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - half_window)
        stop = min(len(values), index + half_window + 1)
        smoothed.append(sum(values[start:stop]) / (stop - start))
    return smoothed


def fit_least_squares(xs, ys) -> tuple:
    """Least-squares line y ~ intercept + slope * x; returns (intercept, slope)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    spread_x = sum((x - mean_x) ** 2 for x in xs)
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov_xy / spread_x if spread_x else 0.0
    return mean_y - slope * mean_x, slope


# ---------------------------------------------------------------------------
# Case parsing
# ---------------------------------------------------------------------------

def parse_iso_timestamp(text: str) -> int:
    """'2026-09-15T00:00:00Z' -> seconds since the epoch (UTC), via calendar math."""
    import calendar

    year = int(text[0:4])
    month = int(text[5:7])
    day = int(text[8:10])
    hour = int(text[11:13])
    minute = int(text[14:16])
    second = int(text[17:19])
    return calendar.timegm((year, month, day, hour, minute, second, 0, 0, 0))


def release_day_curve(comparable: dict) -> dict:
    """{hour 1..24: viewers} for a comparable's release day, from its hourly series.

    Points are bucketed by whole hours since the comparable's day_start;
    non-positive viewer counts are skipped and only the first point in
    each hour is kept.
    """
    day_start = parse_iso_timestamp(comparable["day_start"])
    curve = {}
    for point in comparable.get("hourly") or []:
        viewers = point.get("viewers")
        if not viewers or viewers <= 0:
            continue
        hour = int((parse_iso_timestamp(point["from"]) - day_start) // 3600) + 1
        if 1 <= hour <= 24 and hour not in curve:
            curve[hour] = float(viewers)
    return curve


def usable_release_curves(case: dict) -> list:
    """Release-day curves of the case's comparables, filtered to usable ones.

    A curve is usable when it has an hour-1 print and at least
    MIN_CURVE_HOURS hours in total.
    """
    curves = []
    for comparable in case.get("comparables") or []:
        curve = release_day_curve(comparable)
        if 1 in curve and len(curve) >= MIN_CURVE_HOURS:
            curves.append(curve)
    return curves


def hour_one_viewers(case: dict, name: str) -> float:
    """The target's hour-1 viewer count, floored at 1.0."""
    first_hour = next((row for row in case["known"] if row["hour"] == 1), None)
    if first_hour is None:
        raise ValueError(f"{name} has no hour-1 print")
    return max(1.0, float(first_hour["viewers"]))


# ---------------------------------------------------------------------------
# The two views of the conditional distribution of log-ratios
# ---------------------------------------------------------------------------

def gaussian_kernel_weights(curves: list, log_target_size: float) -> list:
    """One Gaussian-kernel weight per curve, by log-size distance to the target."""
    weights = []
    for curve in curves:
        distance = (math.log(curve[1]) - log_target_size) / KERNEL_BANDWIDTH
        weights.append(math.exp(-0.5 * distance * distance))
    return weights


def kernel_knn_quantiles(curves: list, weights: list, hours: list) -> dict:
    """Per hour, kernel-weighted empirical quantiles of the log-ratio curves.

    Returns {hour: [log-ratio quantile for each level in LEVELS]}.
    """
    view = {}
    for hour in hours:
        weighted_ratios = sorted(
            (math.log(curve[hour] / curve[1]), weight)
            for curve, weight in zip(curves, weights)
            if hour in curve
        )
        view[hour] = [
            weighted_empirical_quantile(weighted_ratios, level) for level in LEVELS
        ]
    return view


def log_size_regression_quantiles(curves: list, log_target_size: float,
                                  hours: list, fallback: dict) -> dict:
    """Per hour, quantiles from a least-squares fit of log-ratio on log size.

    For each hour the log-ratio is regressed on ln(hour-1 viewers) across
    the comparables; the forecast is centred on the fitted value at the
    target's size, with spread taken from the residuals' empirical
    quantiles. Hours with too few comparables fall back to the given
    fallback view (the kNN quantiles).
    """
    view = {}
    for hour in hours:
        log_sizes = []
        log_ratios = []
        for curve in curves:
            if hour in curve:
                log_sizes.append(math.log(curve[1]))
                log_ratios.append(math.log(curve[hour] / curve[1]))
        if len(log_sizes) < MIN_REGRESSION_POINTS:
            view[hour] = fallback[hour]
            continue
        intercept, slope = fit_least_squares(log_sizes, log_ratios)
        residuals = sorted(
            ratio - (intercept + slope * size)
            for size, ratio in zip(log_sizes, log_ratios)
        )
        centre = intercept + slope * log_target_size
        view[hour] = [
            centre + interpolated_quantile(residuals, level) for level in LEVELS
        ]
    return view


# ---------------------------------------------------------------------------
# Blending, smoothing, and output formatting
# ---------------------------------------------------------------------------

def blend_views(knn_view: dict, regression_view: dict, hours: list,
                log_target_size: float) -> dict:
    """Blend the two views in log space, add the bias shift, smooth by hour.

    Returns {hour: [log viewer count per level]}. Smoothing runs along
    each quantile level's log-curve over neighbouring hours.
    """
    blended = {}
    for hour in hours:
        blended[hour] = [
            log_target_size
            + KNN_BLEND_WEIGHT * knn_view[hour][i]
            + (1.0 - KNN_BLEND_WEIGHT) * regression_view[hour][i]
            + LOG_BIAS_SHIFT
            for i in range(len(LEVELS))
        ]
    for level_index in range(len(LEVELS)):
        level_curve = [blended[hour][level_index] for hour in hours]
        smoothed_curve = moving_average(level_curve, SMOOTH_HALF_WINDOW)
        for hour, value in zip(hours, smoothed_curve):
            blended[hour][level_index] = value
    return blended


def log_quantiles_to_rows(log_quantiles: dict, hours: list) -> list:
    """Convert {hour: [log viewer counts]} to the contest's output rows.

    Exponentiates, rounds to 2 decimals (JavaScript style), enforces
    non-decreasing quantiles within each row, and floors at 1 viewer.
    """
    rows = []
    for hour in hours:
        quantiles = [round_js_half_up(math.exp(log_quantiles[hour][i]))
                     for i in range(len(LEVELS))]
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        quantiles = [max(1, q) for q in quantiles]
        rows.append({"hour": hour, "q": quantiles})
    return rows


def persistence_baseline_rows(target_size: float, hours: list) -> list:
    """House persistence baseline: q_tau = v1 * exp(z_tau * 0.15 * sqrt(h-1))."""
    return [
        {
            "hour": hour,
            "q": [
                round_js_half_up(
                    target_size * math.exp(z * 0.15 * math.sqrt(hour - 1))
                )
                for z in BASELINE_Z
            ],
        }
        for hour in hours
    ]


# ---------------------------------------------------------------------------
# Forecasting entry points
# ---------------------------------------------------------------------------

def forecast_from_curves(curves: list, target_size: float, hours: list) -> dict:
    """Forecast hours from comparable release-day curves and the hour-1 size.

    With no usable curves this is the house persistence baseline.
    """
    if not curves:
        return {"hours": persistence_baseline_rows(target_size, hours)}
    log_target_size = math.log(target_size)
    weights = gaussian_kernel_weights(curves, log_target_size)
    knn_view = kernel_knn_quantiles(curves, weights, hours)
    regression_view = log_size_regression_quantiles(
        curves, log_target_size, hours, knn_view
    )
    blended = blend_views(knn_view, regression_view, hours, log_target_size)
    return {"hours": log_quantiles_to_rows(blended, hours)}


def forecast_case(case: dict, name: str) -> dict:
    """Forecast one case dict, validating the requested quantile levels."""
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    target_size = hour_one_viewers(case, name)
    hours = [row["hour"] for row in case["forecast"]]
    curves = usable_release_curves(case)
    return forecast_from_curves(curves, target_size, hours)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
