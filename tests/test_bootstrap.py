import os
import sys
import unittest
from unittest import mock

from ball_runtime.bootstrap import ensure_gstreamer_tls_preload


class GStreamerTlsPreloadTests(unittest.TestCase):
    @mock.patch("ball_runtime.bootstrap.os.execve")
    @mock.patch(
        "ball_runtime.bootstrap._find_gldispatch",
        return_value="/lib/aarch64-linux-gnu/libGLdispatch.so.0",
    )
    def test_restarts_with_preload_and_preserves_arguments(
        self, _find_gldispatch, execve
    ):
        with mock.patch.dict(os.environ, {"LD_PRELOAD": "existing.so"}, clear=True):
            with mock.patch.object(sys, "argv", ["debug_detect_all.py", "--conf", "0.3"]):
                ensure_gstreamer_tls_preload()

        executable, arguments, environment = execve.call_args.args
        self.assertEqual(executable, sys.executable)
        self.assertEqual(
            arguments,
            [sys.executable, "debug_detect_all.py", "--conf", "0.3"],
        )
        self.assertEqual(
            environment["LD_PRELOAD"],
            "/lib/aarch64-linux-gnu/libGLdispatch.so.0:existing.so",
        )

    @mock.patch("ball_runtime.bootstrap.os.execve")
    @mock.patch(
        "ball_runtime.bootstrap._find_gldispatch",
        return_value="/lib/aarch64-linux-gnu/libGLdispatch.so.0",
    )
    def test_does_not_restart_when_already_preloaded(
        self, _find_gldispatch, execve
    ):
        with mock.patch.dict(
            os.environ, {"LD_PRELOAD": "/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0"},
            clear=True,
        ):
            ensure_gstreamer_tls_preload()

        execve.assert_not_called()

    @mock.patch("ball_runtime.bootstrap.os.execve")
    @mock.patch("ball_runtime.bootstrap._find_gldispatch", return_value=None)
    def test_does_not_restart_without_jetson_library(
        self, _find_gldispatch, execve
    ):
        ensure_gstreamer_tls_preload()

        execve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
