#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
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

    def categories(key: str) -> Dict[str, int]:
        return dict(Counter(row.get(key, "") or "<empty>" for row in rows))

    def counter_span(key: str) -> Dict[str, float]:
        values = [finite(row, key) for row in rows]
        values = [value for value in values if math.isfinite(value)]
        if not values:
            return {"first": float("nan"), "last": float("nan"), "increase": float("nan")}
        return {
            "first": values[0],
            "last": values[-1],
            "increase": max(0.0, values[-1] - values[0]),
        }

    positions = [
        finite(row, "x_m") for row in rows if math.isfinite(finite(row, "x_m"))
    ]
    transport_counter_keys = (
        "vision_received_frames",
        "vision_accepted_frames",
        "vision_status_lost_frames",
        "vision_duplicate_frames",
        "vision_missing_frames",
        "observer_rejected_measurements",
        "observer_too_old_measurements",
        "vision_datagrams",
        "vision_decode_errors",
        "vision_parser_crc_errors",
        "vision_parser_length_errors",
        "vision_parser_discarded_bytes",
        "dmmc_status_frames",
        "dmmc_duplicate_frames",
        "dmmc_missing_frames",
        "serial_connect_failures",
        "serial_read_failures",
        "serial_write_failures",
        "dmmc_unknown_frames",
        "dmmc_parser_crc_errors",
        "dmmc_parser_length_errors",
        "dmmc_parser_discarded_bytes",
    )
    return {
        "source": str(path),
        "task": rows[0].get("task", "") if rows else "",
        "task_run_id": finite(rows[0], "task_run_id") if rows else float("nan"),
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
        "max_position_cm": max(positions, default=float("nan")) * 100.0,
        "min_position_cm": min(positions, default=float("nan")) * 100.0,
        "task3_positive_overshoot_beyond_5cm": (
            max(0.0, max(positions) - 0.05) * 100.0 if positions else float("nan")
        ),
        "task3_negative_overshoot_beyond_minus5cm": (
            max(0.0, -0.05 - min(positions)) * 100.0 if positions else float("nan")
        ),
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
        "vision_health_samples_by_reason": categories("vision_health_reason"),
        "vision_update_samples_by_reason": categories("vision_update_reason"),
        "vision_status_samples": categories("vision_status"),
        "dmmc_health_samples_by_reason": categories("dmmc_health_reason"),
        "serial_disconnected_samples": sum(
            int(row.get("serial_connected", "0") or 0) == 0 for row in rows
        ),
        "transport_counters": {
            key: counter_span(key) for key in transport_counter_keys
        },
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
