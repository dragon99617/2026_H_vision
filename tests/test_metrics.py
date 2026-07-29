from __future__ import annotations

import unittest
from pathlib import Path

from ball_runtime.metrics import (
    ImagePredictions,
    average_precision_50,
    score,
    select_threshold,
)
from ball_runtime.types import Detection


def box(x1, y1, x2, y2, confidence=1.0):
    return Detection(x1, y1, x2, y2, confidence)


class MetricsTests(unittest.TestCase):
    def test_matching_counts_false_positive_and_false_negative(self) -> None:
        records = [
            ImagePredictions(
                image=Path("one.jpg"),
                ground_truth=(box(0, 0, 10, 10), box(20, 20, 30, 30)),
                predictions=(
                    box(0, 0, 10, 10, 0.9),
                    box(50, 50, 60, 60, 0.8),
                ),
            )
        ]
        metrics = score(records, 0.5)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 1))
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)

    def test_threshold_rule_prefers_precision_at_recall_target(self) -> None:
        records = [
            ImagePredictions(
                image=Path("one.jpg"),
                ground_truth=(box(0, 0, 10, 10),),
                predictions=(
                    box(0, 0, 10, 10, 0.9),
                    box(50, 50, 60, 60, 0.2),
                ),
            )
        ]
        selected = select_threshold(records, target_recall=0.97)
        self.assertGreater(selected["threshold"], 0.2)
        self.assertAlmostEqual(selected["precision"], 1.0)
        self.assertAlmostEqual(selected["recall"], 1.0)

    def test_perfect_predictions_have_perfect_ap50(self) -> None:
        records = [
            ImagePredictions(
                image=Path("one.jpg"),
                ground_truth=(box(0, 0, 10, 10),),
                predictions=(box(0, 0, 10, 10, 0.9),),
            )
        ]
        self.assertAlmostEqual(average_precision_50(records), 1.0)


if __name__ == "__main__":
    unittest.main()
