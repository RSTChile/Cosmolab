#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CG001 — Visor 3D nativo GPU (PyQt6 + pyqtgraph OpenGL, protocolo §145–154).

Render 3D real en ventana de escritorio (no proyección browser).
Conecta al laboratorio Docker o corre universo local.

  ./tools/run_cg001_3d.sh              # live → http://localhost:7888
  ./tools/run_cg001_3d.sh --local      # simulación embebida
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import Qt, QTimer  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg  # noqa: E402
import pyqtgraph.opengl as gl  # noqa: E402

from CG001.visualization.protocol_colors import display_position, encode_batch  # noqa: E402


class ParticleInterpolator:
    """Interpola posiciones entre polls para movimiento fluido (~60 fps)."""

    def __init__(self) -> None:
        self._from: dict[int, np.ndarray] = {}
        self._to: dict[int, dict] = {}
        self._blend_start = time.monotonic()
        self._blend_ms = 120.0

    def set_targets(self, entities: list[dict], grid_size: float) -> None:
        now = time.monotonic()
        new_to: dict[int, dict] = {}
        for e in entities:
            eid = int(e["id"])
            new_to[eid] = e
            disp = display_position(e.get("pos", [0, 0, 0]), grid_size)
            if eid in self._to:
                self._from[eid] = self._current_pos(eid, grid_size)
            else:
                self._from[eid] = disp.copy()
        self._to = new_to
        self._blend_start = now

    def _current_pos(self, eid: int, grid_size: float) -> np.ndarray:
        if eid not in self._to:
            return self._from.get(eid, np.zeros(3, dtype=np.float32))
        t = min(1.0, (time.monotonic() - self._blend_start) * 1000.0 / self._blend_ms)
        tgt = display_position(self._to[eid].get("pos", [0, 0, 0]), grid_size)
        src = self._from.get(eid, tgt)
        return src * (1.0 - t) + tgt * t

    def blended_entities(self, grid_size: float) -> list[dict]:
        out: list[dict] = []
        for eid, e in sorted(self._to.items()):
            ec = dict(e)
            p = self._current_pos(eid, grid_size)
            ec["pos"] = [float(p[0]), float(p[1]), float(p[2])]
            out.append(ec)
        return out


class TrailManager:
    """Estelas por partícula (trayectoria histórica, §152)."""

    def __init__(self, max_trail: int = 48, max_particles: int = 200) -> None:
        self.max_trail = max_trail
        self.max_particles = max_particles
        self._paths: dict[int, list[list[float]]] = {}
        self._items: dict[int, gl.GLLinePlotItem] = {}

    def reset(self) -> None:
        self._paths.clear()
        for item in self._items.values():
            item.setData(pos=np.zeros((0, 3)))
        self._items.clear()

    def update(self, entities: list[dict], parent: gl.GLViewWidget, enabled: bool) -> None:
        if not enabled:
            return
        active_ids = {int(e["id"]) for e in entities}
        for eid in list(self._paths):
            if eid not in active_ids:
                del self._paths[eid]
                if eid in self._items:
                    parent.removeItem(self._items[eid])
                    del self._items[eid]

        for e in entities[: self.max_particles]:
            eid = int(e["id"])
            p = e.get("pos", [0, 0, 0])
            pt = [float(p[0]), float(p[1]), float(p[2])]
            path = self._paths.setdefault(eid, [])
            if not path or np.linalg.norm(np.array(pt) - np.array(path[-1])) > 0.02:
                path.append(pt)
            if len(path) > self.max_trail:
                path.pop(0)
            if len(path) < 2:
                continue
            arr = np.array(path, dtype=np.float32)
            if eid not in self._items:
                h_norm = float(e.get("H") or 0)
                alpha = 0.25 + 0.45 * min(1.0, h_norm * 2.0)
                color = (0.35, 0.55, 0.95, alpha) if eid != 0 else (1.0, 0.75, 0.25, 0.9)
                item = gl.GLLinePlotItem(color=color, width=1.4 if eid != 0 else 2.2, antialias=True)
                parent.addItem(item)
                self._items[eid] = item
            self._items[eid].setData(pos=arr)


class GLUniverseWidget(gl.GLViewWidget):
    def __init__(self, grid_size: float = 64.0):
        super().__init__()
        self.grid_size = grid_size
        self.setBackgroundColor((6, 10, 20))
        self.opts["distance"] = grid_size * 2.5
        self.setCameraPosition(distance=grid_size * 2.5, elevation=25, azimuth=45)

        g = grid_size
        box_edges = [
            [(0, 0, 0), (g, 0, 0)], [(g, 0, 0), (g, g, 0)], [(g, g, 0), (0, g, 0)], [(0, g, 0), (0, 0, 0)],
            [(0, 0, g), (g, 0, g)], [(g, 0, g), (g, g, g)], [(g, g, g), (0, g, g)], [(0, g, g), (0, 0, g)],
            [(0, 0, 0), (0, 0, g)], [(g, 0, 0), (g, 0, g)], [(g, g, 0), (g, g, g)], [(0, g, 0), (0, g, g)],
        ]
        for a, b in box_edges:
            self.addItem(gl.GLLinePlotItem(
                pos=np.array([a, b], dtype=np.float32),
                color=(0.2, 0.35, 0.6, 0.9),
                width=1.5,
                antialias=True,
            ))

        self.scatter = gl.GLScatterPlotItem(pxMode=True)  # tamano en pixeles -> puntitos que no se hinchan al hacer zoom
        self.addItem(self.scatter)

        self._interp = ParticleInterpolator()
        self._trails = TrailManager()
        self.follow_epsilon = False
        self.show_trails = True
        self._last_entities: list[dict] = []

    def set_targets(self, entities: list[dict], meta: dict | None = None) -> None:
        if meta and meta.get("grid_size"):
            self.grid_size = float(meta["grid_size"])
        self._last_entities = entities
        self._interp.set_targets(entities, self.grid_size)

    def render_frame(self) -> None:
        entities = self._interp.blended_entities(self.grid_size)
        if not entities:
            return
        pos, colors, sizes = encode_batch(entities, self.grid_size)
        self.scatter.setData(pos=pos, color=colors, size=sizes)
        self._trails.update(entities, self, self.show_trails)
        if self.follow_epsilon:
            eps = next((e for e in entities if e.get("id") == 0), None)
            if eps:
                p = eps["pos"]
                self.opts["center"] = pg.Vector(float(p[0]), float(p[1]), float(p[2]))

    def reset_camera(self) -> None:
        g = self.grid_size
        self.setCameraPosition(distance=g * 2.5, elevation=25, azimuth=45)
        self.opts["center"] = pg.Vector(g / 2, g / 2, g / 2)

    def clear_trails(self) -> None:
        self._trails.reset()


class LiveFeed:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def fetch(self) -> tuple[list[dict], dict, dict]:
        ents = self._get("/entidades?limit=1000")
        estado = self._get("/estado")
        return (
            ents.get("entidades", []) if ents.get("ok") else [],
            ents.get("meta", {}),
            estado,
        )

    def _get(self, path: str) -> dict:
        try:
            with urllib.request.urlopen(f"{self.base}{path}", timeout=3) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return {"ok": False}


class LocalSim:
    def __init__(self, epsilon: float, seed: int, tick_hz: float):
        from CG001.core.universe import Universe
        import yaml

        os.environ["CG_EPSILON"] = str(epsilon)
        os.environ["CG_SEED"] = str(seed)
        os.environ["CG_QUIET_EVENTS"] = "1"
        os.environ["CG_FAST_METRICS"] = "1"
        cfg_path = ROOT / "CG001" / "config" / "CG001_default.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.universe = Universe(config=cfg)
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._loop, args=(1.0 / max(tick_hz, 1.0),), daemon=True).start()

    def _loop(self, interval: float) -> None:
        while self._running:
            with self._lock:
                self.universe.step()
            time.sleep(interval)

    def fetch(self) -> tuple[list[dict], dict, dict]:
        with self._lock:
            snap = self.universe.snapshot()
            ents = self.universe.sample_positions(1000)
        return ents, {"grid_size": self.universe.grid_size}, snap

    def stop(self) -> None:
        self._running = False


class CG001DesktopWindow(QMainWindow):
    def __init__(self, feed: LiveFeed | LocalSim, poll_ms: int = 100):
        super().__init__()
        self.feed = feed
        self.setWindowTitle("CosmoGénesis CG001 — Visor 3D nativo (OpenGL)")
        self.resize(1280, 720)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.gl_view = GLUniverseWidget()
        layout.addWidget(self.gl_view, stretch=4)

        panel = QVBoxLayout()
        layout.addLayout(panel, stretch=1)

        title = QLabel("Observatorio 3D — GPU")
        title.setStyleSheet("font-size:16px;font-weight:600;color:#6ee7ff")
        panel.addWidget(title)

        self.lbl_metrics = QLabel("conectando…")
        self.lbl_metrics.setWordWrap(True)
        self.lbl_metrics.setStyleSheet("font-family:monospace;font-size:12px;color:#e8eefc")
        panel.addWidget(self.lbl_metrics)

        self.chk_follow = QCheckBox("Seguir entidad ε (id=0)")
        self.chk_trails = QCheckBox("Estelas por partícula")
        self.chk_trails.setChecked(True)
        panel.addWidget(self.chk_follow)
        panel.addWidget(self.chk_trails)

        btn_cam = QPushButton("Reset cámara")
        btn_cam.clicked.connect(self.gl_view.reset_camera)
        panel.addWidget(btn_cam)
        panel.addStretch()

        hint = QLabel(
            "§152 — Color ← H (historia) · Tamaño ← S · Brillo ← S\n"
            "Estelas = trayectoria por partícula · id estable"
        )
        hint.setStyleSheet("font-size:11px;color:#8899bb")
        panel.addWidget(hint)

        self.chk_follow.toggled.connect(lambda v: setattr(self.gl_view, "follow_epsilon", v))
        self.chk_trails.toggled.connect(self._toggle_trails)

        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start(poll_ms)

        self._render_timer = QTimer()
        self._render_timer.timeout.connect(self.gl_view.render_frame)
        self._render_timer.start(16)

    def _toggle_trails(self, on: bool) -> None:
        self.gl_view.show_trails = on
        if not on:
            self.gl_view.clear_trails()

    def _poll(self) -> None:
        entities, meta, estado = self.feed.fetch()
        self.gl_view.set_targets(entities, meta)
        m = estado.get("metrics", {}) if estado.get("ok") else {}
        self.lbl_metrics.setText(
            f"exp: {estado.get('experiment_id', '—')}\n"
            f"ε: {estado.get('epsilon', '—')}\n"
            f"t_sim: {estado.get('t_sim', '—')}\n"
            f"N: {estado.get('N', '—')} / {estado.get('N0', '—')}\n"
            f"IPD: {m.get('IPD', '—')}\n"
            f"IH: {m.get('IH', '—')}\n"
            f"IN: {m.get('IN', '—')}\n"
            f"IPA: {m.get('IPA', '—')}\n"
            f"ICG₀: {m.get('ICG0', '—')}\n"
            f"S_max: {m.get('S_max', '—')}"
        )

    def closeEvent(self, event) -> None:
        if isinstance(self.feed, LocalSim):
            self.feed.stop()
        super().closeEvent(event)


def main() -> int:
    parser = argparse.ArgumentParser(description="CG001 visor 3D nativo OpenGL")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--url", default=os.environ.get("CG_LAB_URL", "http://127.0.0.1:7888"))
    parser.add_argument("--epsilon", type=float, default=float(os.environ.get("CG_EPSILON", "0.00001")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("CG_SEED", "42")))
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--poll-ms", type=int, default=100)
    args = parser.parse_args()

    if not args.local:
        args.live = True

    feed: LiveFeed | LocalSim
    if args.local:
        feed = LocalSim(args.epsilon, args.seed, args.hz)
    else:
        feed = LiveFeed(args.url)

    qapp = QApplication(sys.argv)
    win = CG001DesktopWindow(feed, poll_ms=args.poll_ms)
    win.show()
    return qapp.exec()


if __name__ == "__main__":
    raise SystemExit(main())