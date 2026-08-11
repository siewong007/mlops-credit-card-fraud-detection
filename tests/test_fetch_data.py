"""The real-data download must survive an interrupted connection.

A single ``urlopen().read()`` of the ~70 MB source fails outright when the
transfer drops (``IncompleteRead``), which blocks reproduction on any throttled
or unstable link. These tests pin the resume contract using a local server; no
network access is required.
"""
import hashlib
import http.server
import os
import socketserver
import threading

import pytest

import src.fetch_data as fetch_data

PAYLOAD = os.urandom(600_000)
SHA = hashlib.sha256(PAYLOAD).hexdigest()
DROP_AFTER = 150_000


def _serve(handler):
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def _base_handler(*, drops: int, honour_range: bool = True):
    state = {"hits": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep pytest output clean
            pass

        def do_GET(self):  # noqa: N802 - stdlib naming
            state["hits"] += 1
            header = self.headers.get("Range")
            start = int(header.split("=")[1].split("-")[0]) if header and honour_range else 0
            body = PAYLOAD[start:]
            if header and honour_range:
                self.send_response(206)
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}"
                )
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if state["hits"] <= drops:
                self.wfile.write(body[:DROP_AFTER])
                self.wfile.flush()
                self.connection.close()
                return
            self.wfile.write(body)

    Handler.state = state
    return Handler


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    monkeypatch.setattr(fetch_data.time, "sleep", lambda _s: None)
    monkeypatch.setattr(fetch_data, "CHUNK_BYTES", 32 * 1024)


def test_resumes_after_dropped_connections(tmp_path):
    handler = _base_handler(drops=2)
    server, port = _serve(handler)
    try:
        dest = fetch_data.download(f"http://127.0.0.1:{port}/d.pq", dest=tmp_path / "d.pq")
    finally:
        server.shutdown()
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == SHA
    assert handler.state["hits"] == 3, "should have resumed twice, not restarted"
    assert not dest.with_suffix(dest.suffix + ".part").exists()


def test_resumes_a_partial_file_left_by_an_earlier_run(tmp_path):
    dest = tmp_path / "d.pq"
    part = dest.with_suffix(dest.suffix + ".part")
    part.write_bytes(PAYLOAD[:400_000])
    handler = _base_handler(drops=0)
    server, port = _serve(handler)
    try:
        out = fetch_data.download(f"http://127.0.0.1:{port}/d.pq", dest=dest)
    finally:
        server.shutdown()
    assert hashlib.sha256(out.read_bytes()).hexdigest() == SHA


def test_restarts_cleanly_when_the_server_ignores_range(tmp_path):
    dest = tmp_path / "d.pq"
    dest.with_suffix(dest.suffix + ".part").write_bytes(PAYLOAD[:200_000])
    handler = _base_handler(drops=0, honour_range=False)
    server, port = _serve(handler)
    try:
        out = fetch_data.download(f"http://127.0.0.1:{port}/d.pq", dest=dest)
    finally:
        server.shutdown()
    assert out.stat().st_size == len(PAYLOAD)
    assert hashlib.sha256(out.read_bytes()).hexdigest() == SHA


def test_does_not_retry_a_missing_url(tmp_path):
    class Missing(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):  # noqa: N802
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server, port = _serve(Missing)
    try:
        with pytest.raises(Exception) as excinfo:
            fetch_data.download(f"http://127.0.0.1:{port}/gone.pq", dest=tmp_path / "d.pq")
    finally:
        server.shutdown()
    assert "404" in str(excinfo.value)
