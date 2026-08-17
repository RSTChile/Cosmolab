#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Observatorio CG001 — compara CG001-A (ε=0) vs CG001-B (ε>0), análogo a anima-conversacion.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

A_URL = os.environ.get("CG_A_URL", "http://127.0.0.1:7888")
B_URL = os.environ.get("CG_B_URL", "http://127.0.0.1:7889")
PORT = int(os.environ.get("CG_OBS_PORT", "7900"))
POLL_S = float(os.environ.get("CG_POLL_S", "1.0"))
STATIC_DIR = Path(__file__).resolve().parents[1] / "server" / "static"

_EST = {"A": {}, "B": {}}
_LOCK = threading.Lock()


def _get_json(url: str, path: str = "/estado", timeout: float = 3.0) -> dict:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}{path}", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _poll_loop():
    while True:
        a = _get_json(A_URL)
        b = _get_json(B_URL)
        with _LOCK:
            _EST["A"] = a
            _EST["B"] = b
        time.sleep(POLL_S)


HTML = """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"/><title>CG001 Observatorio 3D</title>
<style>
body{margin:0;font-family:system-ui,sans-serif;background:#0b1020;color:#e8eefc}
header{padding:14px 18px;border-bottom:1px solid #243055}
main{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px}
.panel{background:#141c33;border:1px solid #243055;border-radius:12px;padding:12px}
.metric{display:inline-block;margin:4px 8px 4px 0;padding:6px 10px;background:#0f1730;border-radius:8px;font-size:.85rem}
.metric b{color:#6ee7ff;font-size:1rem}
.viewport{width:100%;height:min(42vh,400px);min-height:280px;border-radius:8px;overflow:hidden;background:#060a14}
.diff{margin:12px;padding:12px;background:#1a1430;border-radius:8px;border:1px solid #243055}
h2{margin:0 0 8px;font-size:1rem}
@media(max-width:900px){main{grid-template-columns:1fr}}
</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
</head><body>
<header><h1>Observatorio CG001 — Comparación 3D (ε=0 vs ε>0)</h1></header>
<main>
  <section class="panel">
    <h2>Universo A — CG001-A (ε=0)</h2>
    <div id="metrics-a"></div>
    <div class="viewport" id="view-a"></div>
  </section>
  <section class="panel">
    <h2>Universo B — CG001-B (ε>0)</h2>
    <div id="metrics-b"></div>
    <div class="viewport" id="view-b"></div>
  </section>
</main>
<section class="diff" id="diff">cargando divergencia…</section>
<script type="module">
import { CG001Viewer3D } from '/static/viewer3d.js';

const viewerA = new CG001Viewer3D(document.getElementById('view-a'), { gridSize: 64, showTrails: true });
const viewerB = new CG001Viewer3D(document.getElementById('view-b'), { gridSize: 64, showTrails: true });
viewerA.setFollowEpsilon(false);
viewerB.setFollowEpsilon(true);
document.getElementById('view-b').title = 'B sigue entidad ε (id=0)';

function card(s) {
  if (!s.ok && s.error) return '<span style="color:#f88">' + s.error + '</span>';
  const m = s.metrics || {};
  return `<span class="metric"><b>${s.N ?? '—'}</b> N</span>
    <span class="metric"><b>${m.IPD ?? '—'}</b> IPD</span>
    <span class="metric"><b>${m.IH ?? '—'}</b> IH</span>
    <span class="metric"><b>${m.IN ?? '—'}</b> IN</span>
    <span class="metric"><b>t=${s.t_sim ?? 0}</b></span>`;
}

async function refresh3D() {
  try {
    const [ra, rb] = await Promise.all([fetch('/proxy/a/entidades'), fetch('/proxy/b/entidades')]);
    const [da, db] = await Promise.all([ra.json(), rb.json()]);
    if (da.ok) viewerA.updateEntities(da.entidades, da.meta || {});
    if (db.ok) viewerB.updateEntities(db.entidades, db.meta || {});
  } catch (_) {}
}

function refreshHud() {
  fetch('/comparacion').then(r => r.json()).then(d => {
    document.getElementById('metrics-a').innerHTML = card(d.A || {});
    document.getElementById('metrics-b').innerHTML = card(d.B || {});
    const da = d.divergencia || {};
    document.getElementById('diff').innerHTML =
      `ΔIPD=${da.dIPD ?? '—'} · ΔIH=${da.dIH ?? '—'} · ΔN=${da.dN ?? '—'} · predicción: B > A en IPD/IH si ε basta`;
  });
}

setInterval(refreshHud, 1200);
setInterval(refresh3D, 400);
refreshHud(); refresh3D();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, HTML, "text/html; charset=utf-8")
        elif path == "/estado":
            with _LOCK:
                self._send(200, json.dumps({"ok": True, "A": _EST["A"], "B": _EST["B"]}))
        elif path == "/comparacion":
            with _LOCK:
                a, b = _EST["A"], _EST["B"]
            ma = (a.get("metrics") or {})
            mb = (b.get("metrics") or {})
            div = {
                "dIPD": round((mb.get("IPD", 1) or 1) - (ma.get("IPD", 1) or 1), 4),
                "dIH": round((mb.get("IH", 0) or 0) - (ma.get("IH", 0) or 0), 4),
                "dN": (b.get("N", 0) or 0) - (a.get("N", 0) or 0),
            }
            self._send(200, json.dumps({"A": a, "B": b, "divergencia": div}, ensure_ascii=False))
        elif path in ("/proxy/a/entidades", "/proxy/b/entidades"):
            base = A_URL if path.startswith("/proxy/a") else B_URL
            data = _get_json(base, "/entidades?limit=800", timeout=5.0)
            self._send(200, json.dumps(data, ensure_ascii=False))
        elif path.startswith("/static/"):
            rel = path[len("/static/"):]
            if ".." in rel:
                self._send(403, json.dumps({"error": "prohibido"}))
                return
            fpath = STATIC_DIR / rel
            if not fpath.is_file():
                self._send(404, json.dumps({"error": "no encontrado"}))
                return
            ctype = "application/javascript; charset=utf-8" if rel.endswith(".js") else "text/plain"
            self._send(200, fpath.read_text(encoding="utf-8"), ctype)
        else:
            self._send(404, json.dumps({"error": "no encontrado"}))


def main():
    threading.Thread(target=_poll_loop, daemon=True).start()
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[cg001-obs] observatorio → http://0.0.0.0:{PORT}  (A={A_URL} B={B_URL})")
    httpd.serve_forever()


if __name__ == "__main__":
    main()