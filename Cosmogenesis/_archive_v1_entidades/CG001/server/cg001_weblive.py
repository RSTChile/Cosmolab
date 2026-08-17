#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CG001 WebLive — servidor HTTP análogo a ANIMA WebLive.
Expone /estado, /metricas, /stream (SSE) y tablero web para observar el universo en vivo.
"""
from __future__ import annotations

import glob
import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CG001.core.universe import Universe  # noqa: E402

PUERTO = int(os.environ.get("CG_PUERTO", "7888"))
BIND = os.environ.get("CG_BIND", "0.0.0.0")
TICK_HZ = float(os.environ.get("CG_TICK_HZ", "10"))
AUTOSTART = os.environ.get("CG_AUTOSTART", "1") == "1"
HISTORY_DIR = os.environ.get("CG_HISTORY_DIR", "/data")
HISTORY_ENABLE = os.environ.get("CG_HISTORY_ENABLE", "true").lower() == "true"
VISUAL_LIMIT = int(os.environ.get("CG_VISUAL_LIMIT", "800"))
STATIC_DIR = Path(__file__).resolve().parent / "static"

UNI: Universe | None = None
RUN: "SimulationRunner | None" = None


class SimulationRunner:
    def __init__(self, universe: Universe):
        self.universe = universe
        self.q: queue.Queue = queue.Queue(maxsize=256)
        self.thread: threading.Thread | None = None
        self.stop = False
        self.done = False
        self.last_snap: dict = {}

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop = False
        self.done = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        interval = 1.0 / max(TICK_HZ, 0.1)
        while not self.stop:
            snap = self.universe.step()
            self.last_snap = snap
            try:
                self.q.put_nowait(snap)
            except queue.Full:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
                self.q.put_nowait(snap)
            if HISTORY_ENABLE:
                _maybe_log(snap)
            time.sleep(interval)
        self.done = True
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass

    def pause(self, paused: bool) -> None:
        self.universe.paused = paused


def _maybe_log(snap: dict) -> None:
    try:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        day = time.strftime("%Y-%m-%d")
        path = os.path.join(HISTORY_DIR, f"cg001_{snap.get('experiment_id', 'CG001')}_{day}.jsonl")
        if snap.get("t_sim", 0) % 10 != 0:
            return
        row = {
            "t": snap.get("t_sim"),
            "N": snap.get("N"),
            "metrics": snap.get("metrics"),
            "epsilon": snap.get("epsilon"),
            "ts": snap.get("ts"),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


_CSV_COLS = [
    "experiment_id", "t", "N", "epsilon", "ts",
    "IPD", "IH", "IN", "IPA", "ICG0", "S_mean", "S_max", "delta_mean", "H_delta",
]


def _history_csv() -> str:
    """Aplana el JSONL de métricas persistido (/data) a CSV para análisis del equipo."""
    exp = UNI.experiment_id if UNI else "CG001"
    lines = [",".join(_CSV_COLS)]
    try:
        files = sorted(glob.glob(os.path.join(HISTORY_DIR, f"cg001_{exp}_*.jsonl")))
    except Exception:
        files = []
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        row = json.loads(raw)
                    except Exception:
                        continue
                    m = row.get("metrics") or {}
                    vals = [
                        exp, row.get("t", ""), row.get("N", ""),
                        row.get("epsilon", ""), row.get("ts", ""),
                        m.get("IPD", ""), m.get("IH", ""), m.get("IN", ""),
                        m.get("IPA", ""), m.get("ICG0", ""), m.get("S_mean", ""),
                        m.get("S_max", ""), m.get("delta_mean", ""), m.get("H_delta", ""),
                    ]
                    lines.append(",".join(str(v) for v in vals))
        except Exception:
            continue
    return "\n".join(lines) + "\n"


HTML = """<!DOCTYPE html>
<html lang="es"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>CG001 WebLive 3D</title>
<style>
:root{--bg:#0b1020;--panel:#141c33;--accent:#6ee7ff;--text:#e8eefc}
*{box-sizing:border-box}body{margin:0;font-family:ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--text)}
header{padding:14px 18px;border-bottom:1px solid #243055;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
h1{margin:0;font-size:1.1rem}.tag{padding:4px 10px;border-radius:999px;background:#1f2b4d;font-size:.8rem}
main{display:grid;grid-template-columns:1.35fr .65fr;gap:12px;padding:12px;min-height:calc(100vh - 58px)}
.panel{background:var(--panel);border:1px solid #243055;border-radius:12px;padding:12px}
#viewport3d{width:100%;height:min(68vh,560px);min-height:380px;border-radius:8px;overflow:hidden;background:#060a14}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.metric{background:#0f1730;border-radius:8px;padding:10px}.metric b{display:block;font-size:1.15rem;color:var(--accent)}
.metric span{font-size:.75rem;opacity:.8}
.events{font-family:ui-monospace,monospace;font-size:.72rem;max-height:160px;overflow:auto;white-space:pre-wrap}
button,.btn{background:#22345f;color:var(--text);border:1px solid #3a4f82;border-radius:8px;padding:8px 12px;cursor:pointer;font-size:.85rem}
button:hover,.btn:hover{background:#2a4275}
.btn.active{background:#2d5a3d;border-color:#4a9e6a}
.hint{font-size:.78rem;opacity:.75;margin:8px 0 0}
.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}
@media(max-width:900px){main{grid-template-columns:1fr}}
</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
</head><body>
<header>
  <h1>CosmoGénesis CG001</h1>
  <span class="tag" id="exp">—</span>
  <span class="tag" id="eps">ε=—</span>
  <span class="tag" id="alive">N=—</span>
  <button onclick="cgControl('pause')">Pausa</button>
  <button onclick="cgControl('resume')">Reanudar</button>
  <button onclick="cgControl('reset')">Reset</button>
  <a class="btn" href="/csv" download title="Descargar series de métricas (CSV) para análisis del equipo">⬇ CSV</a>
</header>
<main>
  <section class="panel">
    <div class="toolbar">
      <strong>Universo 3D en vivo</strong>
      <button class="btn" id="btnFollow" onclick="toggleFollow()">Seguir ε (id=0)</button>
      <button class="btn" id="btnTrails" onclick="toggleTrails()">Estelas ON</button>
    </div>
    <div id="viewport3d"></div>
    <p class="hint">Arrastra para rotar · rueda para zoom · Color ← H (historia) · Tamaño y brillo ← S · Estelas por partícula</p>
  </section>
  <section class="panel">
    <h3>Observatorio</h3>
    <div class="metrics" id="metrics"></div>
    <h4 style="margin:14px 0 6px">Eventos recientes</h4>
    <div class="events" id="events">conectando…</div>
  </section>
</main>
<script type="module">
import { CG001Viewer3D } from '/static/viewer3d.js';

const viewer = new CG001Viewer3D(document.getElementById('viewport3d'), { gridSize: 64, showTrails: true });
let followOn = false;
let trailsOn = true;

window.toggleFollow = () => {
  followOn = !followOn;
  viewer.setFollowEpsilon(followOn);
  document.getElementById('btnFollow').classList.toggle('active', followOn);
};
window.toggleTrails = () => {
  trailsOn = !trailsOn;
  viewer.showTrails = trailsOn;
  if (!trailsOn) { viewer.trails.clear(); viewer._drawTrails(); }
  document.getElementById('btnTrails').classList.toggle('active', trailsOn);
};
document.getElementById('btnTrails').classList.add('active');

window.cgControl = (action) => fetch('/control', {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ action }),
});

function renderHud(s) {
  document.getElementById('exp').textContent = s.experiment_id || 'CG001';
  document.getElementById('eps').textContent = 'ε=' + (s.epsilon ?? 0);
  document.getElementById('alive').textContent = 'N=' + (s.N ?? 0) + ' / ' + (s.N0 ?? '—') + ' · t=' + s.t_sim;
  const m = s.metrics || {};
  const keys = ['IPD', 'IH', 'IN', 'IPA', 'ICG0', 'S_max', 'S_mean', 'H_delta'];
  document.getElementById('metrics').innerHTML = keys.map(k =>
    `<div class="metric"><b>${m[k] ?? '—'}</b><span>${k}</span></div>`).join('');
  const ev = (s.events_recent || []).map(e => `t${e.t} ${e.evento} ${JSON.stringify(e)}`).join('\\n');
  document.getElementById('events').textContent = ev || '(sin eventos)';
}

async function refresh3D() {
  try {
    const r = await fetch('/entidades?limit=800');
    const d = await r.json();
    if (d.ok) viewer.updateEntities(d.entidades, d.meta || {});
  } catch (_) {}
}

const es = new EventSource('/stream');
es.onmessage = e => { try { renderHud(JSON.parse(e.data)); } catch (_) {} };
es.onerror = () => { document.getElementById('events').textContent = 'stream desconectado — reintentando…'; };
fetch('/estado').then(r => r.json()).then(renderHud);
setInterval(refresh3D, 100);
refresh3D();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "CG001WebLive/0.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[cg001] " + (fmt % args) + "\n")

    def _send(self, code: int, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, HTML, "text/html; charset=utf-8")
        elif path == "/estado":
            snap = RUN.last_snap if RUN else (UNI.snapshot() if UNI else {"ok": False})
            brief = {
                "ok": snap.get("ok", True),
                "experiment_id": snap.get("experiment_id"),
                "t_sim": snap.get("t_sim", 0),
                "N": snap.get("N", 0),
                "epsilon": snap.get("epsilon", 0),
                "paused": snap.get("paused", False),
                "metrics": snap.get("metrics", {}),
                "ts": snap.get("ts", time.time()),
            }
            self._send(200, json.dumps(brief, ensure_ascii=False))
        elif path == "/metricas":
            snap = RUN.last_snap if RUN else (UNI.snapshot() if UNI else {})
            self._send(200, json.dumps(snap, ensure_ascii=False))
        elif path == "/entidades":
            qs = parse_qs(urlparse(self.path).query)
            limit = min(int((qs.get("limit") or [str(VISUAL_LIMIT)])[0]), 1200)
            data = UNI.sample_positions(limit) if UNI else []
            meta = {}
            if UNI:
                meta = {
                    "grid_size": UNI.grid_size,
                    "R": UNI.R,
                    "t_sim": UNI.t_sim,
                    "experiment_id": UNI.experiment_id,
                    "epsilon": UNI.epsilon,
                    "N_alive": len(UNI._alive()),
                }
            self._send(200, json.dumps({"ok": True, "entidades": data, "meta": meta}, ensure_ascii=False))
        elif path.startswith("/static/"):
            rel = path[len("/static/"):]
            if ".." in rel or rel.startswith("/"):
                self._send(403, json.dumps({"error": "prohibido"}))
                return
            fpath = STATIC_DIR / rel
            if not fpath.is_file():
                self._send(404, json.dumps({"error": "no encontrado"}))
                return
            ctype = "application/javascript; charset=utf-8" if rel.endswith(".js") else "text/plain"
            self._send(200, fpath.read_text(encoding="utf-8"), ctype)
        elif path == "/csv":
            exp = UNI.experiment_id if UNI else "CG001"
            body = _history_csv().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="cg001_{exp}_metricas.csv"')
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif path == "/stream":
            self._stream()
        else:
            self._send(404, json.dumps({"error": "no encontrado"}))

    def do_POST(self):
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        if path == "/control":
            action = body.get("action")
            if action == "pause" and RUN:
                RUN.pause(True)
            elif action == "resume" and RUN:
                RUN.pause(False)
            elif action == "reset" and UNI:
                seed = body.get("seed")
                UNI.reset(seed=int(seed) if seed is not None else None)
                if RUN:
                    RUN.last_snap = UNI.snapshot()
            self._send(200, json.dumps({"ok": True}))
        elif path == "/start":
            if RUN:
                RUN.start()
            self._send(200, json.dumps({"ok": True}))
        else:
            self._send(404, json.dumps({"error": "no encontrado"}))

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        run = RUN
        if run is None:
            return
        try:
            while not run.stop:
                try:
                    snap = run.q.get(timeout=1.0)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                if snap is None:
                    break
                payload = json.dumps(snap, ensure_ascii=False)
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    global UNI, RUN
    UNI = Universe()
    RUN = SimulationRunner(UNI)
    if AUTOSTART:
        RUN.start()
        RUN.last_snap = UNI.snapshot()
    httpd = ThreadingHTTPServer((BIND, PUERTO), Handler)
    print(f"[cg001] WebLive {UNI.experiment_id} ε={UNI.epsilon} → http://{BIND}:{PUERTO}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()