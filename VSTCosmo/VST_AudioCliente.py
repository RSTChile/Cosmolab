#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_AudioCliente — CONSUMIDOR (lado cliente): stream TCP → célula madre
================================================================================
El servidor (VST_AudioServer.py) entrega TODOS los canales. AQUÍ, del lado del
CLIENTE (la célula / el contenedor Docker), se hace:
  1) ELECCIÓN DE CANAL: el cliente decide qué canal del stream es oído izquierdo (L)
     y cuál es derecho (R). El servidor NO decide esto.
  2) ADAPTADOR DE IMPEDANCIA: el stream llega en bloques pequeños (~1024 samples,
     ~21 ms), pero un PASO METABÓLICO consume DT·SR = 4800 samples (0.1 s). Este
     consumidor BUFFERIZA los bloques entrantes y, cada vez que junta un "hop" de
     4800 samples por canal, realimenta el soma y avanza un paso metabólico.
  3) VIDA CONTINUA: usa OrganeloSoma.realimentar(), que cambia el bloque MANTENIENDO
     el estado del campo (Phi) → el organismo vive de forma continua con el audio en vivo.

USO
---
  # Con el servidor corriendo (VST_AudioServer.py) y la Rødecaster:
  venv/bin/python3 VST_AudioCliente.py --host 127.0.0.1 --port 8765 --left 3 --right 4
    (L = canal 3 = Combo 1 (L), R = canal 4 = Combo 1 (R); numeración 1-based RØDE)
    Tabla real Rødecaster Pro (pares L/R): 1-2 Main Mix · 3-4 Combo 1 · 5-6 Combo 2 ·
    7-8 Combo 3 · 9-10 Bluetooth · 11-12 USB 2 · 13-14 USB Main · 15-16 SMART Pads · 17-18 USB Chat
    --seg 0  → continuo hasta Ctrl+C   ·   --seg 30 → 30 s y termina

  # Prueba SIN servidor ni hardware (frames sintéticos, 16 canales):
  venv/bin/python3 VST_AudioCliente.py --demo --left 5 --right 10

Guarda un CSV de la sesión en CelulaMadre_logs/. Importable para el contenedor Docker:
  from VST_AudioCliente import ConsumidorCelula
================================================================================
"""
from __future__ import annotations
import argparse, os, sys, time
from datetime import datetime
import numpy as np

# motor + utilidades de la célula (reutiliza lo validado)
from VST_CelulaMadre_WebLive import cmf, _fila, COLS
from VST_AudioServer import AudioStreamClient

HOP = int(round(cmf.DT * cmf.SR))            # 4800 samples = un paso metabólico (DT·SR)


class ConsumidorCelula:
    """Adaptador (lado cliente): recibe bloques (block, n_canales), elige L/R, bufferiza
    a 'hops' de HOP samples, y alimenta una ÚNICA célula madre de forma continua.

    Cada hop de HOP samples → realimentar((L,R)) → un paso metabólico → una fila.
    Resuelve la impedancia bloque-de-audio (≈1024) ↔ paso-metabólico (4800)."""

    def __init__(self, idx_L: int, idx_R: int, toggles: dict | None = None) -> None:
        self.iL, self.iR = int(idx_L), int(idx_R)        # índices 0-based de canal
        self.toggles = toggles or {}
        self.cel = None; self.soma = None; self.rows = []
        self._bufL = np.zeros(0, dtype=np.float64)
        self._bufR = np.zeros(0, dtype=np.float64)

    def _construir(self, hopL, hopR) -> None:
        self.cel = cmf.celula_madre_funcional((hopL.copy(), hopR.copy()), binaural=True)
        for n, o in self.cel.organelos.items():          # ablación (lado cliente, opcional)
            if n != "soma" and not self.toggles.get(n, True):
                o.expresar = False
        self.soma = self.cel.organelos["soma"]

    def alimentar(self, bloque: np.ndarray) -> list:
        """Procesa un bloque (n_samples, n_canales). Devuelve las filas producidas (0 o más)."""
        nch = bloque.shape[1]
        if not (0 <= self.iL < nch and 0 <= self.iR < nch):
            raise IndexError(f"canal L={self.iL+1}/R={self.iR+1} fuera de rango (el stream trae {nch} canales)")
        self._bufL = np.concatenate([self._bufL, bloque[:, self.iL].astype(np.float64)])
        self._bufR = np.concatenate([self._bufR, bloque[:, self.iR].astype(np.float64)])
        filas = []
        while len(self._bufL) >= HOP:
            hopL, hopR = self._bufL[:HOP], self._bufR[:HOP]
            self._bufL, self._bufR = self._bufL[HOP:], self._bufR[HOP:]
            if self.cel is None:
                self._construir(hopL, hopR)
            else:
                self.soma.realimentar((hopL.copy(), hopR.copy()), binaural=True)
            self.cel.vivir_un_paso(cmf.DT)
            f = _fila(self.cel); self.rows.append(f); filas.append(f)
        return filas


def _frames_demo(nch=16, bs=1024, segundos=4.0):
    """Frames sintéticos (sin servidor ni hardware): cada canal una frecuencia distinta."""
    n = int(segundos * cmf.SR / bs)
    for k in range(n):
        t = (np.arange(bs) + k * bs) / cmf.SR
        blk = np.empty((bs, nch), dtype=np.float32)
        for c in range(nch):
            blk[:, c] = (0.3 * np.sin(2 * np.pi * (90 * (c + 1)) * t)).astype(np.float32)
        yield blk


def _guardar(rows, nota):
    if not rows:
        print("  (sin filas que guardar)"); return None
    os.makedirs("CelulaMadre_logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = f"CelulaMadre_logs/cliente_audio_{ts}.csv"
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(f"# {nota}\n" + ",".join(COLS) + "\n" +
                "\n".join(",".join(str(r[c]) for c in COLS) for r in rows))
    print(f"  📁 {ruta}  ({len(rows)} filas)")
    return ruta


def consumir(frames, idx_L, idx_R, toggles=None, max_seg=0.0, nota=""):
    """Recorre `frames` (iterable de bloques) alimentando la célula. max_seg=0 → sin límite."""
    cons = ConsumidorCelula(idx_L, idx_R, toggles)
    t0 = time.time(); ult = t0
    try:
        for blq in frames:
            cons.alimentar(blq)
            ahora = time.time()
            if ahora - ult >= 0.5 and cons.rows:
                r = cons.rows[-1]
                print(f"  t_vida={r['t']:7.1f}s  pasos={len(cons.rows):6d}  "
                      f"OI={r['OI']:.3f}  R2={r['R2']:.3f}  XE={r['XE']:.3f}  "
                      f"lat={r['lateralidad']:.3f}  bal={r['balance_LR']:+.3f}")
                ult = ahora
            if max_seg and (ahora - t0) >= max_seg:
                print("  (límite de tiempo alcanzado)"); break
    except KeyboardInterrupt:
        print("\n  detenido por el usuario.")
    _guardar(cons.rows, nota)
    return cons


def main() -> None:
    p = argparse.ArgumentParser(description="Consumidor: stream de audio TCP → célula madre.")
    p.add_argument("--host", default="127.0.0.1", help="host del servidor (en Docker: host.docker.internal)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--left", type=int, default=3, help="canal para oído IZQUIERDO (1-based, guía RØDE)")
    p.add_argument("--right", type=int, default=4, help="canal para oído DERECHO (1-based)")
    p.add_argument("--seg", type=float, default=0.0, help="segundos a consumir (0 = continuo hasta Ctrl+C)")
    p.add_argument("--apagar", default="", help="organelos a apagar (ablación), separados por coma")
    p.add_argument("--demo", action="store_true", help="frames sintéticos (sin servidor ni hardware)")
    a = p.parse_args()

    toggles = {n.strip(): False for n in a.apagar.split(",") if n.strip()}
    iL, iR = a.left - 1, a.right - 1                       # 1-based (RØDE) → 0-based

    print("=" * 78)
    print("VST_AudioCliente — stream → célula madre  (elección L/R = lado cliente)")
    print(f"  L = canal {a.left}  ·  R = canal {a.right}  ·  hop = {HOP} samples ({cmf.DT*1000:.0f} ms)")
    if toggles: print(f"  ablación: {', '.join(toggles)}")
    print("=" * 78)

    if a.demo:
        print("  MODO DEMO (frames sintéticos 16ch, sin servidor)")
        consumir(_frames_demo(nch=16, segundos=max(4.0, a.seg or 4.0)),
                 iL, iR, toggles, max_seg=a.seg, nota=f"demo L={a.left} R={a.right}")
        return

    print(f"  conectando a {a.host}:{a.port} ...")
    try:
        cli = AudioStreamClient(host=a.host, port=a.port, timeout=5.0)
    except OSError as e:
        print(f"ERROR: no se pudo conectar ({e}). ¿Está corriendo VST_AudioServer.py?"); sys.exit(1)
    hs = cli.handshake()
    print(f"  handshake: {hs}")
    nch = int(hs["channels"])
    if hs.get("sample_rate") and int(hs["sample_rate"]) != cmf.SR:
        print(f"  ⚠ el stream viene a {hs['sample_rate']} Hz pero la célula trabaja a {cmf.SR} Hz "
              "(no se remuestrea aquí; configura el servidor a 48000).")
    if not (0 <= iL < nch and 0 <= iR < nch):
        print(f"ERROR: canales L={a.left}/R={a.right} fuera de rango (el stream trae {nch}). Usa 1..{nch}."); sys.exit(1)
    print(f"  ▶ consumiendo {'continuo (Ctrl+C para parar)' if not a.seg else f'{a.seg:.0f}s'} ...")
    try:
        consumir(cli.frames(), iL, iR, toggles, max_seg=a.seg, nota=f"{hs.get('device','')} L={a.left} R={a.right}")
    finally:
        cli.cerrar()


if __name__ == "__main__":
    main()
