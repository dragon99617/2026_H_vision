from __future__ import annotations

import unittest

from ball_runtime.datagram_writer import parse_udp_endpoint


class DatagramWriterTests(unittest.TestCase):
    def test_parse_endpoint(self) -> None:
        self.assertEqual(parse_udp_endpoint("127.0.0.1:29001"), ("127.0.0.1", 29001))

    def test_reject_invalid_endpoint(self) -> None:
        for value in ("127.0.0.1", ":29001", "127.0.0.1:0", "127.0.0.1:70000"):
            with self.assertRaises(ValueError):
                parse_udp_endpoint(value)


if __name__ == "__main__":
    unittest.main()
