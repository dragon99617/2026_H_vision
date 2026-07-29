from __future__ import annotations

import unittest
from pathlib import Path

from prepare_dataset import DEFAULT_SOURCE, discover_samples, planned_split


class DatasetTests(unittest.TestCase):
    def test_relabelled_dataset_matches_manifest_contract(self) -> None:
        samples = discover_samples(DEFAULT_SOURCE)
        self.assertEqual(len(samples), 89)
        self.assertEqual(sum(sample.positive for sample in samples), 80)
        self.assertEqual(sum(sample.object_count for sample in samples), 483)
        split = planned_split(samples)
        self.assertEqual(
            {name: len(values) for name, values in split.items()},
            {"train": 55, "val": 17, "test": 17},
        )
        self.assertEqual(
            {
                name: (
                    sum(item.positive for item in values),
                    sum(not item.positive for item in values),
                )
                for name, values in split.items()
            },
            {"train": (50, 5), "val": (15, 2), "test": (15, 2)},
        )


if __name__ == "__main__":
    unittest.main()
