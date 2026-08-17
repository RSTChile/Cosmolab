#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_SDRPipeline — ciclo de vida del SDR ATADO al organismo (nace/muere con A o E)
==================================================================================
QUIÉN SOY (para retomar sin reprocesar):
  Enciendo y apago el pipeline físico del SDR — el receptor headless de SDRconnect (que abre
  la WS API en 5454) y el puente de audio (VST_SDRAudioBridge, que sirve el WFM demodulado en
  8771) — de forma que el SDR NO quede "permanentemente prendido" como daemon suelto, sino que
  VIVA MIENTRAS VIVE EL ORGANISMO que tiene el órgano de radio (A en la Mac, E en la Pi). El
  organismo lo arranca al nacer (si su órgano está activo) y lo detiene al cerrarse.

  Coherente con la economía metabólica del genoma y con "el órgano pertenece al organismo, no al
  hardware": el receptor de radio se enciende porque HAY un organismo que lo usa, y se apaga con él.
  (La demodulación de AUDIO, además, es bajo demanda dentro del puente: solo mientras alguien oye
  la fuente 📻 — ver VST_SDRAudioBridge. Aquí gobernamos el nivel de PROCESO, allí el de señal.)

IDEMPOTENTE Y RESPETUOSO: si el headless ya estaba corriendo (lo lanzaste tú a mano), NO lanzo otro
  ni lo mato al salir — solo detengo lo que YO arranqué. Así puedo convivir con tu trabajo manual.

MULTI-MÁQUINA: el binario del headless difiere Mac/Pi → configurable por ANIMA_SDR_HEADLESS_CMD.
  El puente se lanza con el MISMO intérprete que corre el organismo (sys.executable), así hereda su
  venv (websockets, numpy) sin adivinar rutas.

Config por entorno:
  ANIMA_SDRWS_URI          ws://127.0.0.1:5454   (de dónde sale la WS API; define host/puerto a sondear)
  ANIMA_SDR_AUDIO_PORT     8771                  (puerto del puente de audio)
  ANIMA_SDR_HEADLESS_CMD   /Applications/SDRconnect.app/Contents/MacOS/SDRconnect_headless
                                                 (en la Pi: ruta del SDRconnect headless de linux-arm64)
  ANIMA_SDR_HEADLESS_ESPERA 12                   (segundos máx. a esperar que abra el 5454)
  ANIMA_SDR_LOG_DIR        ~                     (dónde van los logs de headless/puente)
"""
from __future__ import annotations
import os
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse


def _puerto_escucha(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class PipelineSDR:
    """Arranca/detiene el receptor headless + el puente de audio, atados a la vida del organismo."""

    def __init__(self) -> None:
        uri = os.environ.get("ANIMA_SDRWS_URI", "ws://127.0.0.1:5454")
        pu = urlparse(uri)
        self.ws_host = pu.hostname or "127.0.0.1"
        self.ws_port = pu.port or 5454
        self.audio_host = "127.0.0.1"
        self.audio_port = int(os.environ.get("ANIMA_SDR_AUDIO_PORT", "8771"))
        self.headless_cmd = os.environ.get(
            "ANIMA_SDR_HEADLESS_CMD",
            "/Applications/SDRconnect.app/Contents/MacOS/SDRconnect_headless")
        self.headless_espera = float(os.environ.get("ANIMA_SDR_HEADLESS_ESPERA", "12"))
        self.log_dir = os.path.expanduser(os.environ.get("ANIMA_SDR_LOG_DIR", "~"))
        self._headless_proc: subprocess.Popen | None = None   # solo si LO arrancamos nosotros
        self._bridge_proc: subprocess.Popen | None = None

    def _log(self, nombre: str):
        try:
            return open(os.path.join(self.log_dir, nombre), "ab")
        except OSError:
            return subprocess.DEVNULL

    def arrancar(self) -> dict:
        """Enciende lo que falte: headless (si el 5454 no escucha) y puente (si el 8771 no escucha)."""
        estado = {"headless": "ya_estaba", "puente": "ya_estaba"}

        # 1) receptor headless (abre la WS API 5454) — solo si nadie lo sirve aún
        if not _puerto_escucha(self.ws_host, self.ws_port):
            if os.path.exists(self.headless_cmd):
                self._headless_proc = subprocess.Popen(
                    [self.headless_cmd], stdout=self._log("sdrconnect_headless.log"),
                    stderr=subprocess.STDOUT)
                estado["headless"] = "arrancado"
                # esperar a que abra el 5454 (o rendirse con gracia)
                t0 = time.time()
                while time.time() - t0 < self.headless_espera:
                    if _puerto_escucha(self.ws_host, self.ws_port):
                        break
                    time.sleep(0.4)
                else:
                    estado["headless"] = "arrancado_sin_5454"
            else:
                estado["headless"] = f"sin_binario ({self.headless_cmd})"

        # 2) puente de audio (WS→TCP en 8771) — solo si no escucha ya; con el intérprete del organismo
        if not _puerto_escucha(self.audio_host, self.audio_port):
            bridge = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VST_SDRAudioBridge.py")
            if os.path.exists(bridge):
                self._bridge_proc = subprocess.Popen(
                    [sys.executable, "-u", bridge], stdout=self._log("sdr_audio_bridge.log"),
                    stderr=subprocess.STDOUT)
                estado["puente"] = "arrancado"
            else:
                estado["puente"] = f"sin_script ({bridge})"

        return estado

    def detener(self) -> None:
        """Detiene SOLO lo que este pipeline arrancó (no toca procesos que ya estaban)."""
        for proc in (self._bridge_proc, self._headless_proc):
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=4)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                except OSError:
                    pass
        self._bridge_proc = None
        self._headless_proc = None


if __name__ == "__main__":
    # Prueba manual: arranca, informa, y al recibir Ctrl+C detiene lo que arrancó.
    p = PipelineSDR()
    print("[pipeline SDR] arrancando…", p.arrancar())
    print("[pipeline SDR] en marcha. Ctrl+C para detener lo que arranqué.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[pipeline SDR] deteniendo…")
        p.detener()
        print("[pipeline SDR] listo.")
