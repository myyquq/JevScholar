"""`jev controller serve`: d1-3B's option probabilities over HTTP, for the controlled loop.

It runs in `.venv-train` (transformers 5); the runner's workers, in the main environment, call it
(controller/policy.py). Requests are served one at a time and one row per forward pass, as the
checkpoint's own `system_one` reads them (batching rows would only approximate that in bf16), and
answers are cached by request, so a re-run sees the same probabilities. Each question of a
request is answered on its own, as `system_one` answers a one-question request: its packing of
several questions over one state shifts their answers (by up to 0.03 on a loop state).

    GET  /info    {"model", "revision", "adapter"}: the adapter's sha256 (null: the base model)
    POST /choice  {"requests": [{"state", "questions": [...]}, ...], "render": {"style"}} ->
                  {"probabilities": [[[...] per question] per request], "tokens": n}, or 413 with
                  {"tokens": n} when a prompt is longer than d1's validated length
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from jevscholar.controller.d1 import MAX_TOKENS, MODEL

PORT = 9940


def adapter_sha(adapter: Path | None) -> str | None:
    from jevscholar.util import sha256_file

    return sha256_file(adapter / "adapter_model.safetensors") if adapter else None


def make_server(d1: Any, info: dict[str, Any], port: int = PORT) -> HTTPServer:
    """An HTTP server answering for `d1` (anything with D1's `encode` and `probabilities`)."""
    cache: dict[str, dict[str, Any]] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/info":
                return self._send(404, {"error": f"no route {self.path}"})
            self._send(200, info)

        def do_POST(self) -> None:
            if self.path != "/choice":
                return self._send(404, {"error": f"no route {self.path}"})
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            key = raw.decode()
            if key not in cache:
                body = json.loads(raw)
                render = body.get("render")
                rows = [(r["state"], q) for r in body["requests"] for q in r["questions"]]
                lengths = [len(d1.encode(s, q, render)[0]) for s, q in rows]
                if max(lengths) > MAX_TOKENS:
                    return self._send(413, {"tokens": max(lengths), "limit": MAX_TOKENS})
                probs = [d1.probabilities([row], render=render)[0] for row in rows]  # row by row
                sizes = [len(r["questions"]) for r in body["requests"]]
                grouped = [probs[sum(sizes[:i]) : sum(sizes[: i + 1])] for i in range(len(sizes))]
                cache[key] = {"probabilities": grouped, "tokens": sum(lengths)}
            self._send(200, cache[key])

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            out = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args: Any) -> None:
            pass

    return HTTPServer(("127.0.0.1", port), Handler)


def serve(adapter: Path | None = None, port: int = PORT) -> None:
    """Load d1 (with `adapter`, if given) and serve it until interrupted."""
    from jevscholar.controller.d1 import D1, REVISION

    info = {"model": MODEL, "revision": REVISION, "adapter": adapter_sha(adapter)}
    server = make_server(D1.load(adapter=adapter), info, port)
    print(f"d1 on http://127.0.0.1:{port} (adapter {info['adapter']})", flush=True)
    server.serve_forever()
