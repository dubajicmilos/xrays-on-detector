"""Static server for web/ with a screenshot sink, for development only.

    python tools/devserver.py [port]

Serves the web/ folder, and accepts POST /_shot/<name>.png with a raw image
body, writing it to tools/_shots/. The app calls that from the console so
rendered frames can be inspected outside the browser. Nothing in the shipped
site depends on this.
"""
from __future__ import annotations

import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
SHOTS = os.path.join(ROOT, "tools", "_shots")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=WEB, **kw)

    def do_POST(self):
        if not self.path.startswith("/_shot/"):
            self.send_error(404)
            return
        name = os.path.basename(self.path[len("/_shot/"):]) or "shot.png"
        length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(length)
        os.makedirs(SHOTS, exist_ok=True)
        path = os.path.join(SHOTS, name)
        with open(path, "wb") as fh:
            fh.write(data)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(f"{len(data)} bytes -> {path}".encode())
        print(f"saved {len(data)} bytes to {path}", flush=True)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "_shot" in (args[0] if args else ""):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8777
    print(f"serving {WEB} on http://localhost:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
