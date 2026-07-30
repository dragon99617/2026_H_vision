#!/usr/bin/env python3
"""Localhost-only PyPI bridge that uses curl for upstream TLS.

JetPack 5's Python/OpenSSL can fail against current PyPI CDNs while the system
curl remains compatible. pip talks plain HTTP to this ephemeral localhost
server; every upstream request is still HTTPS through curl.
"""

from __future__ import annotations

import argparse
import hashlib
import mimetypes
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


class CurlProxyHandler(BaseHTTPRequestHandler):
    server_version = "CurlPyPIProxy/1.0"

    def log_message(self, _format, *_args):
        return

    def do_HEAD(self):
        self._handle(send_body=False)

    def do_GET(self):
        self._handle(send_body=True)

    def _upstream_url(self) -> str:
        path = urlsplit(self.path).path
        if path.startswith("/simple/"):
            return "https://pypi.org" + path
        if path.startswith("/files/"):
            return "https://files.pythonhosted.org" + path[len("/files") :]
        if path.startswith("/pypi/") or path.startswith("/integrity/"):
            return "https://pypi.org" + path
        raise ValueError("unsupported path")

    def _handle(self, send_body: bool) -> None:
        try:
            upstream = self._upstream_url()
        except ValueError:
            self.send_error(404)
            return
        if upstream.startswith("https://pypi.org/simple/"):
            self._serve_index(upstream, send_body)
        else:
            self._serve_file(upstream, send_body)

    def _serve_index(self, upstream: str, send_body: bool) -> None:
        completed = subprocess.run(
            ["curl", "-fsSL", "--retry", "5", "--max-time", "120", upstream],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode:
            self.send_error(502, completed.stderr.decode("utf-8", "replace"))
            return
        base = "http://127.0.0.1:%d" % self.server.server_port
        body = completed.stdout.replace(
            b"https://files.pythonhosted.org",
            (base + "/files").encode("ascii"),
        ).replace(
            b"https://pypi.org/integrity/",
            (base + "/integrity/").encode("ascii"),
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=UTF-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def _serve_file(self, upstream: str, send_body: bool) -> None:
        cache_dir: Path = self.server.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        suffix = "".join(Path(urlsplit(upstream).path).suffixes)[-32:]
        cache = cache_dir / (hashlib.sha256(upstream.encode()).hexdigest() + suffix)
        if not cache.is_file():
            partial = Path(str(cache) + ".partial")
            completed = subprocess.run(
                [
                    "curl",
                    "-fsSL",
                    "--retry",
                    "5",
                    "--max-time",
                    "1800",
                    "-o",
                    str(partial),
                    upstream,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode:
                partial.unlink(missing_ok=True)
                self.send_error(502, completed.stderr.decode("utf-8", "replace"))
                return
            partial.replace(cache)
        self.send_response(200)
        content_type = mimetypes.guess_type(urlsplit(upstream).path)[0]
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(cache.stat().st_size))
        self.end_headers()
        if send_body:
            with cache.open("rb") as handle:
                while True:
                    block = handle.read(1024 * 1024)
                    if not block:
                        break
                    self.wfile.write(block)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-file", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "ball-yolo-pypi-cache",
    )
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", 0), CurlProxyHandler)
    server.cache_dir = args.cache_dir
    args.port_file.write_text(str(server.server_port), encoding="ascii")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
