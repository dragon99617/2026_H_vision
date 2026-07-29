from __future__ import annotations

import unittest

from select_default import eligible


def accepted_candidate():
    metrics = {"precision": 0.96, "recall": 0.96, "map50": 0.97}
    return {
        "metrics": dict(metrics),
        "pytorch_metrics": dict(metrics),
        "onnx_metrics": dict(metrics),
        "benchmark": {"pure_fps": 76.0},
        "comparison": {"tensorrt_recall_drop_vs_pytorch": 0.0},
    }


class ModelSelectionTests(unittest.TestCase):
    def test_offline_candidate_meeting_all_gates_is_provisionally_eligible(self):
        self.assertTrue(eligible(accepted_candidate()))

    def test_recall_conversion_drop_is_limited_to_one_point(self):
        candidate = accepted_candidate()
        candidate["comparison"]["tensorrt_recall_drop_vs_pytorch"] = 0.011
        self.assertFalse(eligible(candidate))

    def test_live_metrics_become_mandatory_when_present(self):
        candidate = accepted_candidate()
        candidate["live"] = {
            "capture_median_fps": 59.0,
            "inference_average_fps": 57.9,
            "latency_p95_ms": 30.0,
        }
        self.assertFalse(eligible(candidate))

    def test_each_backend_must_pass_accuracy_gate(self):
        candidate = accepted_candidate()
        candidate["onnx_metrics"]["map50"] = 0.949
        self.assertFalse(eligible(candidate))


if __name__ == "__main__":
    unittest.main()
