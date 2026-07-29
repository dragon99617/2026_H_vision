#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def percentile(values, fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("inf")
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate five-point, five-pitch half-pipe calibration CSVs"
    )
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--minimum-frames-per-point", type=int, default=100)
    parser.add_argument("--max-p95-error-cm", type=float, default=0.3)
    parser.add_argument("--minimum-valid-rate", type=float, default=0.99)
    args = parser.parse_args()

    groups = defaultdict(list)
    all_rows = []
    for path in args.csv:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    reference = float(row["reference_position_cm"])
                    pitch = float(row["reference_pitch_deg"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "%s lacks --reference-position-cm/--reference-pitch-deg metadata"
                        % path
                    ) from exc
                row["_reference"] = reference
                row["_pitch"] = pitch
                groups[(reference, pitch)].append(row)
                all_rows.append(row)
    if not all_rows:
        raise RuntimeError("no calibration rows")

    failures = []
    group_reports = []
    all_errors = []
    wrong_signs = 0
    valid_total = 0
    for (reference, pitch), rows in sorted(groups.items()):
        valid = []
        for row in rows:
            if row.get("status") not in ("1", "2") or not row.get("position_cm"):
                continue
            position = float(row["position_cm"])
            valid.append(position)
            error = abs(position - reference)
            all_errors.append(error)
            if abs(reference) > 0.5 and position * reference < 0.0:
                wrong_signs += 1
        valid_total += len(valid)
        valid_rate = len(valid) / float(len(rows))
        errors = [abs(value - reference) for value in valid]
        p95 = percentile(errors, 0.95)
        passed = (
            len(rows) >= args.minimum_frames_per_point
            and valid_rate >= args.minimum_valid_rate
            and p95 <= args.max_p95_error_cm
        )
        if not passed:
            failures.append(
                "reference=%+.2f pitch=%+.1f rows=%d valid=%.3f p95=%.3f"
                % (reference, pitch, len(rows), valid_rate, p95)
            )
        group_reports.append(
            {
                "reference_cm": reference,
                "reference_pitch_deg": pitch,
                "frames": len(rows),
                "valid_rate": valid_rate,
                "error_p95_cm": p95,
                "passed": passed,
            }
        )
    expected_positions = {-12.5, -6.25, 0.0, 6.25, 12.5}
    expected_pitches = {-20.0, -10.0, 0.0, 10.0, 20.0}
    actual_positions = {round(key[0], 2) for key in groups}
    actual_pitches = {round(key[1], 1) for key in groups}
    if actual_positions != expected_positions:
        failures.append(
            "reference positions incomplete: %s" % sorted(actual_positions)
        )
    if actual_pitches != expected_pitches:
        failures.append(
            "reference pitches incomplete: %s" % sorted(actual_pitches)
        )
    if wrong_signs:
        failures.append("%d valid samples have the wrong sign" % wrong_signs)
    report = {
        "passed": not failures,
        "frames": len(all_rows),
        "valid_rate": valid_total / float(len(all_rows)),
        "overall_error_p95_cm": percentile(all_errors, 0.95),
        "wrong_sign_samples": wrong_signs,
        "groups": group_reports,
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
