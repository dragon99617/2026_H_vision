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
    errors = [
        abs(finite(row, "x_m") - finite(row, "x_ref_m"))
        for row in rows
        if math.isfinite(finite(row, "x_m")) and math.isfinite(finite(row, "x_ref_m"))
    ]
    solve_times = [
        finite(row, "mpc_ms") for row in rows if math.isfinite(finite(row, "mpc_ms"))
    ]
    vision_ages = [
        finite(row, "vision_age_ms")
        for row in rows
        if math.isfinite(finite(row, "vision_age_ms"))
    ]
    theta = [abs(finite(row, "theta_cmd_rad")) for row in rows]
    theta_limit = math.radians(2.0)
    count = max(1, len(rows))
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
        "angle_saturation_ratio": sum(value >= theta_limit * 0.99 for value in theta) / count,
        "mpc_solve_p95_ms": percentile(solve_times, 95.0),
        "mpc_solve_max_ms": max(solve_times, default=float("nan")),
        "mpc_over_3ms_ratio": sum(value > 3.0 for value in solve_times) / count,
        "mpc_over_10ms_count": sum(value > 10.0 for value in solve_times),
        "vision_age_p95_ms": percentile(vision_ages, 95.0),
        "fallback_ratio": sum(int(row.get("fallback", "0") or 0) != 0 for row in rows) / count,
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
