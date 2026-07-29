from __future__ import annotations

import unittest

from ball_runtime.latest import LatestValue


class LatestValueTests(unittest.TestCase):
    def test_overwrites_instead_of_queueing(self) -> None:
        slot = LatestValue()
        first = slot.publish("first")
        second = slot.publish("second")
        sequence, value = slot.get()
        self.assertEqual(first, 1)
        self.assertEqual(second, 2)
        self.assertEqual(sequence, 2)
        self.assertEqual(value, "second")
        self.assertEqual(slot.depth, 1)

    def test_wait_after_times_out_with_current_value(self) -> None:
        slot = LatestValue()
        slot.publish(123)
        sequence, value = slot.wait_after(1, timeout=0.001)
        self.assertEqual(sequence, 1)
        self.assertEqual(value, 123)


if __name__ == "__main__":
    unittest.main()
