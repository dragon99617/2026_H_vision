#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List


def percentile(values: List[float], percentage: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentage / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def finite(row: Dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
        return value if math.isfinite(value) else float("nan")
    except (KeyError, TypeError, ValueError):
        return float("nan")


def analyze(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    signed_errors = [
        finite(row, "position_error_m")
        if math.isfinite(finite(row, "position_error_m"))
        else finite(row, "x_ref_m") - finite(row, "x_m")
        for row in rows
        if (
            math.isfinite(finite(row, "position_error_m"))
            or (
                math.isfinite(finite(row, "x_m"))
                and math.isfinite(finite(row, "x_ref_m"))
            )
        )
    ]
    errors = [abs(value) for value in signed_errors]
    vision_ages = [
        finite(row, "vision_age_ms")
        for row in rows
        if math.isfinite(finite(row, "vision_age_ms"))
    ]
    theta = [
        abs(finite(row, "theta_cmd_rad"))
        for row in rows
        if math.isfinite(finite(row, "theta_cmd_rad"))
    ]
    inner_angle_errors = [
        abs(finite(row, "inner_angle_error_rad"))
        for row in rows
        if math.isfinite(finite(row, "inner_angle_error_rad"))
    ]
    theta_limit = math.radians(4.0)
    count = max(1, len(rows))
    settling_time_s = float("nan")
    if rows and len(errors) == len(rows):
        suffix_within_band = True
        for index in range(len(rows) - 1, -1, -1):
            suffix_within_band = suffix_within_band and errors[index] <= 0.004
            if suffix_within_band:
                settling_time_s = finite(rows[index], "time_s") - finite(rows[0], "time_s")
    steady_count = min(len(signed_errors), max(1, len(signed_errors) // 10))
    steady_errors = signed_errors[-steady_count:] if signed_errors else []
    return {
        "source": str(path),
        "samples": len(rows),
        "duration_s": (
            finite(rows[-1], "time_s") - finite(rows[0], "time_s") if len(rows) >= 2 else 0.0
        ),
        "position_error_p95_cm": percentile(errors, 95.0) * 100.0,
        "position_error_max_cm": max(errors, default=float("nan")) * 100.0,
        "position_error_rms_cm": (
            math.sqrt(sum(value * value for value in errors) / len(errors)) * 100.0
            if errors
            else float("nan")
        ),
        "steady_state_error_cm": (
            sum(steady_errors) / len(steady_errors) * 100.0
            if steady_errors
            else float("nan")
        ),
        "settling_time_4mm_s": settling_time_s,
        "angle_saturation_ratio": sum(value >= theta_limit * 0.99 for value in theta) / count,
        "pid_saturation_ratio": sum(
            int(row.get("pid_saturated", "0") or 0) != 0 for row in rows
        ) / count,
        "pid_integral_limit_ratio": sum(
            int(row.get("pid_integral_limited", "0") or 0) != 0 for row in rows
        ) / count,
        "pid_integrator_frozen_ratio": sum(
            int(row.get("pid_integrator_frozen", "0") or 0) != 0 for row in rows
        ) / count,
        "inner_angle_error_p95_deg": math.degrees(percentile(inner_angle_errors, 95.0)),
        "inner_angle_error_max_deg": math.degrees(
            max(inner_angle_errors, default=float("nan"))
        ),
        "inner_angle_warning_samples": sum(
            int(row.get("inner_angle_warning", "0") or 0) != 0 for row in rows
        ),
        "vision_age_p95_ms": percentile(vision_ages, 95.0),
        "slowdown_samples": sum(int(row.get("slow", "0") or 0) != 0 for row in rows),
        "stop_samples": sum(int(row.get("stop", "0") or 0) != 0 for row in rows),
    }


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize an NX control CSV log")
    parser.add_argument("log", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    metrics = analyze(args.log)
    payload = json.dumps(json_safe(metrics), ensure_ascii=False, indent=2) + "\n"
    print(payload, end="")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
