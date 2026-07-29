from __future__ import annotations

import unittest

from hardware_acceptance import check


class HardwareAcceptanceTests(unittest.TestCase):
    def test_performance_passes_at_limits(self) -> None:
        self.assertEqual(
            check(
                "performance",
                {
                    "duration_seconds": 600,
                    "capture_median_fps": 59,
                    "inference_average_fps": 58,
                    "latency_p95_ms": 35,
                },
            ),
            [],
        )

    def test_motion_reports_failed_coverage(self) -> None:
        failures = check(
            "motion",
            {
                "duration_seconds": 300,
                "measured_detection_coverage": 0.9,
                "effective_output_rate": 0.95,
                "max_predicted_streak_frames": 2,
            },
        )
        self.assertEqual(len(failures), 2)

    def test_static_checks_jitter(self) -> None:
        self.assertEqual(
            check(
                "static",
                {
                    "duration_seconds": 60,
                    "center_jitter_p95_radius_px": 2,
                },
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
