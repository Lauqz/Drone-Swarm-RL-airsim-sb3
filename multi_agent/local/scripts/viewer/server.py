"""Stdlib-only live viewer for the local drone sim.

Runs a ThreadingHTTPServer in a daemon thread. The env pushes a small JSON
state after every step; browsers receive it over Server-Sent Events (no
websocket dependency, one-way is all we need). Open http://localhost:<port>
from Windows — WSL2 forwards localhost automatically.

Endpoints:
  /             the viewer page (Three.js, vendored, works offline)
  /three.min.js vendored library
  /map          the map JSON the sim itself loaded (view == physics)
  /events       SSE stream of {episode, step, drones, cams, ...}

The server never back-pressures training: pushes just overwrite the latest
state, and each SSE client is throttled to ~15 updates/s.
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
MIN_SEND_INTERVAL = 0.066  # ~15 Hz per client


class _Hub:
    def __init__(self):
        self.cond = threading.Condition()
        self.payload = b"{}"
        self.version = 0

    def push(self, state):
        data = json.dumps(state).encode("utf-8")
        with self.cond:
            self.payload = data
            self.version += 1
            self.cond.notify_all()

    def wait_newer(self, version, timeout=1.0):
        with self.cond:
            if self.version == version:
                self.cond.wait(timeout)
            return self.version, self.payload


class DroneViewer:
    def __init__(self, map_data, port=8777, host="0.0.0.0", quiet=True):
        self.hub = _Hub()
        map_payload = json.dumps(map_data).encode("utf-8")
        hub = self.hub

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                if not quiet:
                    BaseHTTPRequestHandler.log_message(self, fmt, *args)

            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    with open(os.path.join(STATIC_DIR, "index.html"), "rb") as f:
                        self._send(200, "text/html; charset=utf-8", f.read())
                elif self.path == "/three.min.js":
                    with open(os.path.join(STATIC_DIR, "three.min.js"), "rb") as f:
                        self._send(200, "application/javascript", f.read())
                elif self.path == "/map":
                    self._send(200, "application/json", map_payload)
                elif self.path == "/events":
                    self._stream_events()
                else:
                    self._send(404, "text/plain", b"not found")

            def _stream_events(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                version = -1
                try:
                    while True:
                        new_version, payload = hub.wait_newer(version)
                        if new_version != version:
                            version = new_version
                            self.wfile.write(b"data: " + payload + b"\n\n")
                        else:
                            self.wfile.write(b": keep-alive\n\n")  # SSE comment
                        self.wfile.flush()
                        time.sleep(MIN_SEND_INTERVAL)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       name="drone-viewer", daemon=True)
        self.thread.start()
        print(f"[viewer] live view at http://localhost:{self.port}")

    def push(self, state):
        self.hub.push(state)

    def close(self):
        self.server.shutdown()
