#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_NRF24Bridge — PUENTE NATIVO de la radio digital de A (Mac).
================================================================================
QUIÉN SOY
  El organismo A corre en Docker, y Docker Desktop en Mac NO puede ver el Arduino
  Uno (el nRF24 de A) por USB. Así que soy un proceso NATIVO en el Mac que:
    - LEE el Uno (estado del enlace nRF24: RF24/RFRX/RFTX),
    - lo expongo por HTTP para que A (en el contenedor) lo consuma,
    - y acepto órdenes de TRANSMITIR (A → E) por HTTP.
  Es el mismo patrón que el AudioServer: el hardware vive nativo, el organismo lo
  consume por red (host.docker.internal).

SERIE
  Uso pyserial con el python3 del SISTEMA (que sí lo trae; el venv-mac no). arduino-cli
  monitor bufferiza la tubería y no entrega líneas a tiempo → pyserial directo es mejor.
  Correr: `/usr/bin/python3 audio/VST_NRF24Bridge.py` (NO el venv-mac).

HTTP
  GET  /nrf   -> JSON con el estado del enlace (nrf_ok/connected/rx/tx/last_rx/vivo)
  POST /tx    -> body = texto; transmite "T<texto>" al Uno (paquete A → E)
"""
import os
import json
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT_ARDUINO = os.environ.get("ANIMA_UNO_PORT", "/dev/cu.usbmodem14101")
HTTP_PORT = int(os.environ.get("ANIMA_NRF24_BRIDGE_PORT", "8772"))

_estado = {"nrf_ok": 0, "nrf_connected": 0, "nrf_rx": 0, "nrf_tx": 0,
           "nrf_last_rx": "", "nrf_last_tx": "", "ts": 0.0}
_lock = threading.Lock()
_ser = None
_ser_lock = threading.Lock()


def _lector():
    """Abre el Uno por pyserial y parsea las líneas del nodo nRF24 (RF24/RFRX/RFTX)."""
    global _ser
    import serial
    while True:
        try:
            with _ser_lock:
                _ser = serial.Serial(PORT_ARDUINO, 115200, timeout=1)
            time.sleep(2.2)  # el Uno resetea al abrir; esperar boot
            _ser.reset_input_buffer()
            while True:
                line = _ser.readline().decode("ascii", "ignore").strip()
                if not line or line.startswith("#"):
                    continue
                p = line.split(",")
                tag = p[0].upper()
                with _lock:
                    if tag == "RF24" and len(p) >= 5:
                        _estado["nrf_ok"] = int(float(p[1]))
                        _estado["nrf_connected"] = int(float(p[2]))
                        _estado["nrf_rx"] = int(float(p[3]))
                        _estado["nrf_tx"] = int(float(p[4]))
                        _estado["ts"] = time.time()
                    elif tag == "RFRX":
                        _estado["nrf_last_rx"] = (",".join(p[1:]).replace(",", " ").strip())[:32]
                        _estado["ts"] = time.time()
                    elif tag == "RFTX":
                        _estado["nrf_last_tx"] = (p[1].strip() if len(p) > 1 else "")[:8]
                        _estado["ts"] = time.time()
        except Exception as e:
            print("[nrf-bridge] serie caído (%s), reintento..." % e, flush=True)
            with _ser_lock:
                try:
                    if _ser is not None:
                        _ser.close()
                except Exception:
                    pass
                _ser = None
            time.sleep(2)


def _tx(text: str) -> bool:
    """Transmite un paquete A → E (escribe 'T<texto>\\n' al Uno por el mismo serie)."""
    with _ser_lock:
        if _ser is None:
            return False
        try:
            _ser.write(("T" + text + "\n").encode("ascii", "ignore"))
            _ser.flush()
            return True
        except Exception:
            return False


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/nrf"):
            with _lock:
                d = dict(_estado)
            d["nrf_vivo"] = 1 if (d["ts"] and time.time() - d["ts"] < 8.0) else 0
            self._send(200, d)
        else:
            self._send(404, {"error": "no encontrado"})

    def do_POST(self):
        if self.path.startswith("/tx"):
            n = int(self.headers.get("Content-Length", 0) or 0)
            text = self.rfile.read(n).decode(errors="replace").strip()[:30]
            ok = _tx(text)
            self._send(200 if ok else 503, {"sent": ok, "text": text})
        else:
            self._send(404, {"error": "no encontrado"})


def main():
    threading.Thread(target=_lector, daemon=True).start()
    print("NRF24 bridge: Uno %s → HTTP :%d  (GET /nrf, POST /tx)" % (PORT_ARDUINO, HTTP_PORT), flush=True)
    HTTPServer(("0.0.0.0", HTTP_PORT), _Handler).serve_forever()


if __name__ == "__main__":
    main()
