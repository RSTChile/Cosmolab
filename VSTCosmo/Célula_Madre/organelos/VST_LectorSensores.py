#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_LectorSensores — lectores de hardware para E (Pi): ATmega (LUZ+GPS) y SDRplay RSP1.
Un hilo por fuente; inyectan campos CRUDOS en la fila. Los organelos solo consumen.
"""
from __future__ import annotations
import glob
import os
import socket
import threading
import time

WATCHDOG_S = 5.0


def _puerto_atmega() -> str:
    env = os.environ.get("ANIMA_ATMEGA_PORT", "").strip()
    if env:
        return env
    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return "/dev/ttyUSB0"


class LectorATmega:
    """Lee LUZ y GPS del ATmega por USB-serie (115200). Hilo daemon; no bloquea el loop de E."""

    def __init__(self, puerto: str | None = None, baud: int = 115200):
        self.puerto = puerto or _puerto_atmega()
        self.baud = int(baud)
        self._lock = threading.Lock()
        self._run = False
        self._hilo: threading.Thread | None = None
        self._ser = None
        self._ult = 0.0
        self._ok = False
        self._datos: dict = {
            "v_panel": None, "v_fuente": None, "v_lipo": None,
            "foto_adc_a0": None, "foto_adc_a1": None,
            "atmega_origen": "", "atmega_wifi_rx": 0,
            "gps_fix": 0, "gps_sats": 0, "gps_hdop": 99.0,
            "gps_lat": None, "gps_lon": None, "gps_alt": None,
            "gps_pps_count": None, "gps_pps_ms": None,
            "gps_time_utc": "", "gps_date_utc": "",
            "gps_vivo": 0, "atmega_vivo": 0,
            # radio DIGITAL nRF24 (la 3a radio, en el ATmega): estado del enlace con A (Uno)
            "nrf_ok": 0, "nrf_connected": 0, "nrf_rx": 0, "nrf_tx": 0,
            "nrf_last_rx": "", "nrf_last_tx": "",
            "nrf_tx_manual": 0, "nrf_backend": "",
        }
        self._nrf_rx_prev = 0
        self._pps_count = 0

    @property
    def vivo(self) -> bool:
        with self._lock:
            return self._ok and (time.time() - self._ult) < WATCHDOG_S

    def arrancar(self) -> bool:
        if self._run:
            return True
        try:
            import serial
            self._ser = serial.Serial(self.puerto, self.baud, timeout=2.0)
        except Exception as e:
            print(f"[LectorATmega] no abre {self.puerto}: {e}")
            return False
        self._run = True
        self._hilo = threading.Thread(target=self._loop, daemon=True, name="LectorATmega")
        self._hilo.start()
        print(f"[LectorATmega] escuchando {self.puerto} @ {self.baud}")
        return True

    def cerrar(self):
        self._run = False
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def transmitir(self, text: str) -> bool:
        """Envía un paquete nRF24 E→A: escribe 'T<texto>\\n' al ATmega, que hace RFTX.
        Contrato simétrico al LectorNRF24Remoto.transmitir de A (para la prueba de enlace)."""
        ser = self._ser
        if ser is None or not bool(self._datos.get("nrf_tx_manual")):
            return False
        try:
            ser.write(("T" + str(text) + "\n").encode("ascii", "ignore"))
            return True
        except Exception:
            return False

    def _parse_luz(self, parts: list[str]):
        if len(parts) < 3:
            return
        vp = float(parts[1])
        vl = float(parts[2])
        a0 = int(parts[3]) if len(parts) > 3 else None
        a1 = int(parts[4]) if len(parts) > 4 else None
        with self._lock:
            self._datos["v_panel"] = vp
            self._datos["v_fuente"] = vp
            self._datos["v_lipo"] = vl
            self._datos["foto_adc_a0"] = a0
            self._datos["foto_adc_a1"] = a1
            self._datos["atmega_origen"] = "usb"
            self._ok = True
            self._ult = time.time()

    def _parse_gps(self, parts: list[str]):
        if len(parts) < 8:
            return
        fix = int(float(parts[1]))
        sats = int(float(parts[2]))
        hdop = float(parts[3])
        lat = float(parts[4])
        lon = float(parts[5])
        alt = float(parts[6])
        pps = float(parts[9]) if len(parts) > 9 else None
        with self._lock:
            self._datos["gps_fix"] = fix
            self._datos["gps_sats"] = sats
            self._datos["gps_hdop"] = hdop
            self._datos["gps_lat"] = lat
            self._datos["gps_lon"] = lon
            self._datos["gps_alt"] = alt
            self._datos["gps_pps_count"] = pps
            if len(parts) > 11:
                self._datos["gps_time_utc"] = parts[10].strip()
                self._datos["gps_date_utc"] = parts[11].strip()
            self._datos["gps_vivo"] = 1 if fix >= 1 else 0
            self._datos["atmega_origen"] = "usb"
            self._ok = True
            self._ult = time.time()

    def _parse_rf24(self, parts: list[str]):
        # RF24,<ok>,<connected>,<rx_count>,<tx_count>  (estado 1 Hz del nodo digital)
        if len(parts) < 5:
            return
        with self._lock:
            self._datos["nrf_ok"] = int(float(parts[1]))
            self._datos["nrf_connected"] = int(float(parts[2]))
            self._datos["nrf_rx"] = int(float(parts[3]))
            self._datos["nrf_tx"] = int(float(parts[4]))
            self._datos["atmega_origen"] = "usb"
            self._ok = True
            self._ult = time.time()

    def _parse_rf24_config(self, parts: list[str]):
        # RF24_CONFIG,OK[,TXMANUAL]  (Arduino limpio con GPS + nRF24)
        ok = 1 if len(parts) > 1 and parts[1].strip().upper() == "OK" else 0
        caps = {p.strip().upper() for p in parts[2:]}
        with self._lock:
            self._datos["nrf_ok"] = ok
            self._datos["nrf_tx_manual"] = 1 if ("TXMANUAL" in caps or "TX" in caps) else 0
            self._datos["nrf_backend"] = "arduino" if ok else ""
            self._datos["atmega_origen"] = "usb"
            self._ok = True
            self._ult = time.time()

    def _parse_radio_rx(self, parts: list[str]):
        # RADIO_RX,<payload>  (paquete recibido desde la red nRF24)
        payload = ",".join(parts[1:]).replace(",", " ").strip()[:32]
        with self._lock:
            self._datos["nrf_last_rx"] = payload
            self._datos["nrf_rx"] = int(self._datos.get("nrf_rx") or 0) + 1
            self._datos["nrf_ok"] = 1
            self._datos["nrf_connected"] = 1
            self._datos["atmega_origen"] = "usb"
            self._ok = True
            self._ult = time.time()

    def _parse_rf24_status(self, parts: list[str]):
        """RF24_STATUS,<connected>,GPS_LAT,<lat>,GPS_LNG,<lon>,GPS_SAT,<sats>
        o RF24_STATUS,<connected>,GPS,Escaneando...[Satellites:n].
        """
        if len(parts) < 2:
            return
        connected = int(float(parts[1]))
        data = {}
        i = 2
        while i < len(parts):
            key = parts[i].strip().upper()
            val = parts[i + 1].strip() if i + 1 < len(parts) else ""
            data[key] = val
            i += 2
        with self._lock:
            self._datos["nrf_ok"] = 1
            self._datos["nrf_connected"] = connected
            self._datos["nrf_backend"] = self._datos.get("nrf_backend") or "arduino"
            if connected:
                # El firmware transmite un heartbeat DATOS_OK despues de cada status.
                self._datos["nrf_tx"] = int(self._datos.get("nrf_tx") or 0) + 1
                self._datos["nrf_last_tx"] = "DATOS_OK"
            if "GPS_LAT" in data and "GPS_LNG" in data:
                self._datos["gps_fix"] = 1
                self._datos["gps_lat"] = float(data["GPS_LAT"])
                self._datos["gps_lon"] = float(data["GPS_LNG"])
                self._datos["gps_sats"] = int(float(data.get("GPS_SAT", "0") or 0))
                self._datos["gps_time_utc"] = data.get("GPS_TIME", self._datos.get("gps_time_utc", ""))
                self._datos["gps_date_utc"] = data.get("GPS_DATE", self._datos.get("gps_date_utc", ""))
                self._datos["gps_vivo"] = 1
            else:
                sat_text = data.get("GPS", "")
                sats = 0
                if "Satellites:" in sat_text:
                    try:
                        sats = int(sat_text.rsplit("Satellites:", 1)[1].split("]", 1)[0])
                    except Exception:
                        sats = 0
                self._datos["gps_fix"] = 0
                self._datos["gps_sats"] = sats
                self._datos["gps_vivo"] = 0
            self._datos["atmega_origen"] = "usb"
            self._ok = True
            self._ult = time.time()

    def _parse_pps(self, line: str):
        # # PPS_PULSE_DETECTED_AT:<millis>
        try:
            ms = int(float(line.rsplit(":", 1)[1]))
        except Exception:
            ms = None
        with self._lock:
            self._pps_count += 1
            self._datos["gps_pps_count"] = self._pps_count
            self._datos["gps_pps_ms"] = ms
            self._datos["atmega_origen"] = "usb"
            self._ok = True
            self._ult = time.time()

    def _loop(self):
        while self._run and self._ser is not None:
            try:
                ln = self._ser.readline().decode("ascii", "ignore").strip()
                if not ln:
                    continue
                if ln.startswith("#"):
                    if ln.startswith("# PPS_PULSE_DETECTED_AT:"):
                        self._parse_pps(ln)
                    continue
                parts = ln.split(",")
                tag = parts[0].upper()
                if tag == "LUZ":
                    self._parse_luz(parts)
                elif tag == "GPS":
                    self._parse_gps(parts)
                elif tag == "RF24":
                    self._parse_rf24(parts)
                elif tag == "RF24_CONFIG":
                    self._parse_rf24_config(parts)
                elif tag == "RF24_STATUS":
                    self._parse_rf24_status(parts)
                elif tag == "RADIO_RX":
                    self._parse_radio_rx(parts)
                elif tag == "RFRX":                              # paquete recibido de A (Uno)
                    payload = ",".join(parts[1:]).replace(",", " ").strip()[:32]
                    with self._lock:
                        self._datos["nrf_last_rx"] = payload
                        self._ok = True
                        self._ult = time.time()
                elif tag == "RFTX":                              # RFTX,ok|fail[,payload]
                    estado = (parts[1].strip().lower() if len(parts) > 1 else "")
                    payload = ",".join(parts[2:]).replace(",", " ").strip() if len(parts) > 2 else ""
                    with self._lock:
                        self._datos["nrf_last_tx"] = (payload or estado)[:32]
                        if estado in ("ok", "1", "true"):
                            self._datos["nrf_tx"] = int(self._datos.get("nrf_tx") or 0) + 1
                            self._datos["nrf_connected"] = 1
                            self._datos["nrf_ok"] = 1
                        self._datos["atmega_origen"] = "usb"
                        self._ok = True
                        self._ult = time.time()
            except Exception:
                with self._lock:
                    self._ok = False
                time.sleep(0.5)

    def inyectar(self, fila: dict) -> None:
        with self._lock:
            d = dict(self._datos)
            alive = self._ok and (time.time() - self._ult) < WATCHDOG_S
            d["atmega_vivo"] = 1 if alive else 0
            if not alive:
                d["gps_vivo"] = 0
            # actividad de recepción digital desde el último tick (para spark/indicador)
            rx = int(d.get("nrf_rx") or 0)
            d["nrf_rx_delta"] = max(0, rx - self._nrf_rx_prev)
            self._nrf_rx_prev = rx
            # el enlace digital está vivo si el ATmega responde Y el chip está conectado por SPI
            d["nrf_vivo"] = 1 if (alive and int(d.get("nrf_connected") or 0) == 1) else 0
            fila.update(d)


class LectorATmegaWiFi:
    """Recibe paquetes TCP del RD-WiFi/ESP8266 del ATmega de E.

    Es un respaldo al USB: el firmware envia una linea CSV consolidada cada pocos
    segundos. Si esta fresca, inyecta las mismas columnas que LectorATmega para
    que el cloroplasto, GPS y radio digital sigan visibles aunque la caja este
    separada de la Pi.
    """

    def __init__(self, host: str | None = None, port: int | None = None):
        self.host = host if host is not None else os.environ.get("ANIMA_ATMEGA_WIFI_HOST", "0.0.0.0")
        self.port = int(port if port is not None else os.environ.get("ANIMA_ATMEGA_WIFI_PORT", "5000"))
        self._lock = threading.Lock()
        self._run = False
        self._hilo: threading.Thread | None = None
        self._ult = 0.0
        self._rx_total = 0
        self._nrf_rx_prev = 0
        self._datos: dict = {
            "v_panel": None, "v_fuente": None, "v_lipo": None,
            "foto_adc_a0": None, "foto_adc_a1": None,
            "atmega_origen": "", "atmega_wifi_rx": 0,
            "gps_fix": 0, "gps_sats": 0, "gps_hdop": 99.0,
            "gps_lat": None, "gps_lon": None, "gps_alt": None,
            "gps_pps_count": None, "gps_vivo": 0, "atmega_vivo": 0,
            "nrf_ok": 0, "nrf_connected": 0, "nrf_rx": 0, "nrf_tx": 0,
            "nrf_last_rx": "", "nrf_last_tx": "",
        }

    @property
    def vivo(self) -> bool:
        with self._lock:
            return (time.time() - self._ult) < WATCHDOG_S

    def arrancar(self) -> bool:
        if self._run:
            return True
        self._run = True
        self._hilo = threading.Thread(target=self._loop, daemon=True, name="LectorATmegaWiFi")
        self._hilo.start()
        print(f"[LectorATmegaWiFi] escuchando TCP {self.host}:{self.port}")
        return True

    def cerrar(self):
        self._run = False

    def _section(self, parts: list[str], tag: str) -> list[str]:
        try:
            i = parts.index(tag)
        except ValueError:
            return []
        j = len(parts)
        for nxt in ("LUZ", "GPS", "RF24"):
            if nxt == tag:
                continue
            try:
                k = parts.index(nxt, i + 1)
                j = min(j, k)
            except ValueError:
                pass
        return parts[i + 1:j]

    def _parse(self, text: str) -> None:
        parts = [p.strip() for p in text.strip().split(",") if p.strip()]
        if not parts:
            return

        luz = self._section(parts, "LUZ")
        gps = self._section(parts, "GPS")
        rf24 = self._section(parts, "RF24")

        with self._lock:
            if len(luz) >= 4:
                vp = float(luz[0])
                vl = float(luz[1])
                self._datos["v_panel"] = vp
                self._datos["v_fuente"] = vp
                self._datos["v_lipo"] = vl
                self._datos["foto_adc_a0"] = int(float(luz[2]))
                self._datos["foto_adc_a1"] = int(float(luz[3]))

            if len(gps) >= 10:
                fix = int(float(gps[0]))
                self._datos["gps_fix"] = fix
                self._datos["gps_sats"] = int(float(gps[1]))
                self._datos["gps_hdop"] = float(gps[2])
                self._datos["gps_lat"] = float(gps[3])
                self._datos["gps_lon"] = float(gps[4])
                self._datos["gps_alt"] = float(gps[5])
                self._datos["gps_pps_count"] = float(gps[8])
                self._datos["gps_vivo"] = 1 if fix >= 1 else 0

            if len(rf24) >= 4:
                self._datos["nrf_ok"] = int(float(rf24[0]))
                self._datos["nrf_connected"] = int(float(rf24[1]))
                self._datos["nrf_rx"] = int(float(rf24[2]))
                self._datos["nrf_tx"] = int(float(rf24[3]))

            self._rx_total += 1
            self._datos["atmega_origen"] = "wifi"
            self._datos["atmega_wifi_rx"] = self._rx_total
            self._ult = time.time()

    def _loop(self):
        while self._run:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    server.bind((self.host, self.port))
                    server.listen(5)
                    server.settimeout(1.0)
                    while self._run:
                        try:
                            conn, addr = server.accept()
                        except socket.timeout:
                            continue
                        with conn:
                            data = conn.recv(2048)
                        if not data:
                            continue
                        text = data.decode("utf-8", errors="replace").strip()
                        if text:
                            self._parse(text)
                            print(f"[LectorATmegaWiFi] {addr[0]} -> {text[:120]}", flush=True)
            except Exception as e:
                print(f"[LectorATmegaWiFi] no escucha {self.host}:{self.port}: {e}")
                time.sleep(2.0)

    def inyectar(self, fila: dict) -> None:
        with self._lock:
            fresh = (time.time() - self._ult) < WATCHDOG_S
            if not fresh:
                return
            d = dict(self._datos)
            d["atmega_vivo"] = 1
            rx = int(d.get("nrf_rx") or 0)
            d["nrf_rx_delta"] = max(0, rx - self._nrf_rx_prev)
            self._nrf_rx_prev = rx
            d["nrf_vivo"] = 1 if int(d.get("nrf_connected") or 0) == 1 else 0
            fila.update(d)


class LectorNRF24Remoto:
    """Radio DIGITAL de A leída desde el PUENTE nativo (VST_NRF24Bridge) por HTTP.
    A corre en Docker y no ve el Arduino Uno; el puente (nativo en el Mac) sí. Produce
    el MISMO set de columnas nrf_* que el LectorATmega de E → la caja nrf24.js sirve igual.
    También transmite (A → E) por POST /tx."""

    def __init__(self, url: str, periodo_s: float = 1.0):
        self.url = url.rstrip("/")
        self.periodo = float(periodo_s)
        self._lock = threading.Lock()
        self._run = False
        self._hilo: threading.Thread | None = None
        self._datos = {"nrf_ok": 0, "nrf_connected": 0, "nrf_rx": 0, "nrf_tx": 0,
                       "nrf_last_rx": "", "nrf_last_tx": "", "nrf_vivo": 0}
        self._rx_prev = 0
        self._ult = 0.0

    @property
    def vivo(self) -> bool:
        with self._lock:
            return (time.time() - self._ult) < WATCHDOG_S

    def arrancar(self) -> bool:
        if self._run:
            return True
        self._run = True
        self._hilo = threading.Thread(target=self._loop, daemon=True, name="LectorNRF24Remoto")
        self._hilo.start()
        print(f"[LectorNRF24Remoto] puente {self.url}")
        return True

    def cerrar(self):
        self._run = False

    def _loop(self):
        import json
        import urllib.request
        while self._run:
            try:
                with urllib.request.urlopen(self.url + "/nrf", timeout=2.0) as r:
                    d = json.loads(r.read().decode("utf-8", "ignore"))
                with self._lock:
                    for k in ("nrf_ok", "nrf_connected", "nrf_rx", "nrf_tx",
                              "nrf_last_rx", "nrf_last_tx", "nrf_vivo"):
                        if k in d:
                            self._datos[k] = d[k]
                    self._ult = time.time()
            except Exception:
                with self._lock:
                    self._datos["nrf_vivo"] = 0
            time.sleep(self.periodo)

    def inyectar(self, fila: dict) -> None:
        with self._lock:
            d = dict(self._datos)
            fresh = (time.time() - self._ult) < WATCHDOG_S
            rx = int(d.get("nrf_rx") or 0)
            d["nrf_rx_delta"] = max(0, rx - self._rx_prev)
            self._rx_prev = rx
            if not fresh:
                d["nrf_vivo"] = 0
            fila.update(d)

    def transmitir(self, text: str) -> bool:
        import urllib.request
        try:
            req = urllib.request.Request(self.url + "/tx", data=str(text).encode("ascii", "ignore"), method="POST")
            urllib.request.urlopen(req, timeout=2.0).read()
            return True
        except Exception:
            return False


class LectorSDR:
    """Barre el RSP1 por trozos (~8 MHz visibles) y pone espectro en la fila.
    Backend: SoapySDR (gr-sdrplay3) si está; si no, modo inactivo (sdr_vivo=0)."""

    def __init__(self):
        self.enabled = os.environ.get("ANIMA_SDR_ENABLE", "0").lower() in ("1", "true", "yes", "on")
        self.fmin = float(os.environ.get("ANIMA_SDR_FMIN_HZ", "88000000"))
        self.fmax = float(os.environ.get("ANIMA_SDR_FMAX_HZ", "108000000"))
        self.chunk_hz = float(os.environ.get("ANIMA_SDR_CHUNK_HZ", "7000000"))
        self.n_bins = int(os.environ.get("ANIMA_SDR_BINS", "256"))
        self.sample_rate = float(os.environ.get("ANIMA_SDR_SAMPLE_RATE", "2000000"))
        self._lock = threading.Lock()
        self._run = False
        self._hilo: threading.Thread | None = None
        self._espectro: list[float] | None = None
        self._fmin = self.fmin
        self._fmax = self.fmax
        self._vivo = False
        self._ult = 0.0
        self._orden_hz: float | None = None
        self._dev = None
        self._stream = None
        self._backend = "none"

    @property
    def vivo(self) -> bool:
        with self._lock:
            return self._vivo and (time.time() - self._ult) < 30.0

    def sintonizar(self, freq_hz: float | None) -> None:
        with self._lock:
            self._orden_hz = float(freq_hz) if freq_hz is not None else None

    def arrancar(self) -> bool:
        if not self.enabled:
            print("[LectorSDR] ANIMA_SDR_ENABLE=0 — inactivo")
            return False
        if self._run:
            return True
        # Intento abrir ahora; si el RSP1 está ocupado (típico justo tras un reinicio:
        # el proceso anterior aún no soltó el device) NO abortamos — arrancamos el hilo
        # igual y su auto-sanación reintenta abrir hasta lograrlo.
        abierto = self._abrir_dispositivo()
        self._run = True
        self._hilo = threading.Thread(target=self._loop_barrido, daemon=True, name="LectorSDR")
        self._hilo.start()
        estado = f"backend={self._backend}" if abierto else "esperando al RSP1 (reintentará)"
        print(f"[LectorSDR] barrido {self.fmin/1e6:.1f}–{self.fmax/1e6:.1f} MHz · {estado}")
        return True

    def cerrar(self):
        self._run = False
        self._cerrar_dispositivo()

    def reactivar(self) -> None:
        """Fuerza cerrar+reabrir el RSP1 en el próximo ciclo del barrido (para el botón
        'Reactivar radio' de la UI). El _loop_barrido reabre solo cuando _dev es None.
        La recuperación de hardware (reset de USB + servicio sdrplay) la hace un script
        aparte; esto solo obliga al lector a soltar el handle viejo y agarrar el device fresco."""
        with self._lock:
            self._vivo = False
        self._cerrar_dispositivo()

    def _abrir_dispositivo(self) -> bool:
        os.environ.setdefault("LD_LIBRARY_PATH", "/usr/local/lib")
        try:
            import SoapySDR
            from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
            self._SoapySDR = SoapySDR
            self._SOAPY_SDR_RX = SOAPY_SDR_RX
            self._SOAPY_SDR_CF32 = SOAPY_SDR_CF32
            kwargs = {"driver": "sdrplay"}
            self._dev = SoapySDR.Device(kwargs)
            self._dev.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
            self._dev.setBandwidth(SOAPY_SDR_RX, 0, min(self.sample_rate * 0.9, 8e6))
            self._dev.setGain(SOAPY_SDR_RX, 0, float(os.environ.get("ANIMA_SDR_GAIN", "40")))
            self._stream = self._dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
            self._dev.activateStream(self._stream)
            self._backend = "SoapySDR"
            return True
        except Exception as e:
            print(f"[LectorSDR] no abre SDR ({e}) — E sigue sin radio")
            return False

    def _cerrar_dispositivo(self):
        try:
            if self._dev is not None and self._stream is not None:
                self._dev.deactivateStream(self._stream)
                self._dev.closeStream(self._stream)
        except Exception:
            pass
        self._dev = None
        self._stream = None

    def _captura_psd(self, center_hz: float, n_fft: int) -> list[float]:
        import numpy as np
        SOAPY_SDR_RX = self._SOAPY_SDR_RX
        dev = self._dev
        stream = self._stream
        dev.setFrequency(SOAPY_SDR_RX, 0, center_hz)
        time.sleep(0.05)
        buf = np.zeros(n_fft, np.complex64)
        leidas = 0
        for _ in range(4):
            sr = dev.readStream(stream, [buf], n_fft, timeoutUs=500000)
            r = getattr(sr, "ret", sr)   # SoapySDR devuelve un objeto con .ret (nº muestras o error<0)
            if isinstance(r, int) and r > 0:
                leidas += r
        if leidas <= 0:
            # readStream dio timeout/error en las 4 pasadas → el stream está caído (el RSP1
            # clon se cuelga: abre pero no entrega muestras). ANTES esto se ignoraba y el
            # espectro salía TODO EN CERO reportando 'vivo'. Ahora lanzamos para que la
            # auto-sanación cierre y reabra el dispositivo (o el botón 'Reactivar radio' resetee el USB).
            raise RuntimeError("readStream sin muestras (stream caído / RSP1 colgado)")
        spec = np.abs(np.fft.fftshift(np.fft.fft(buf))) ** 2
        spec = spec / (np.max(spec) + 1e-12)
        return [float(x) for x in spec]

    def _loop_barrido(self):
        import numpy as np
        chunk_bins = max(16, self.n_bins // 8)
        fallos = 0          # fallos consecutivos → dispara la reapertura del dispositivo
        backoff = 3.0       # espera creciente entre reintentos (sensible a cerrar())
        while self._run:
            # asegurar el dispositivo abierto: tras un reinicio o una caída puede estar
            # cerrado (o el RSP1 aún ocupado por el proceso anterior). Reintenta con backoff.
            if self._dev is None or self._stream is None:
                if not self._abrir_dispositivo():
                    with self._lock:
                        self._vivo = False
                    t_end = time.time() + backoff
                    while self._run and time.time() < t_end:
                        time.sleep(0.2)
                    backoff = min(backoff * 2, 30.0)
                    continue
                print("[LectorSDR] RSP1 abierto — barriendo")
                fallos = 0
                backoff = 3.0
            try:
                t0 = time.time()
                with self._lock:
                    orden = self._orden_hz
                if orden is not None:
                    centers = [orden]
                    fmin, fmax = orden - self.chunk_hz / 2, orden + self.chunk_hz / 2
                else:
                    step = self.chunk_hz * 0.85
                    centers = []
                    c = self.fmin + self.chunk_hz / 2
                    while c < self.fmax:
                        centers.append(c)
                        c += step
                    fmin, fmax = self.fmin, self.fmax
                bloques: list[list[float]] = []
                for center in centers:
                    bloques.append(self._captura_psd(center, chunk_bins))
                espectro = []
                for b in bloques:
                    espectro.extend(b)
                if len(espectro) > self.n_bins:
                    idx = np.linspace(0, len(espectro) - 1, self.n_bins).astype(int)
                    espectro = [espectro[i] for i in idx]
                elif len(espectro) < self.n_bins:
                    espectro = espectro + [0.0] * (self.n_bins - len(espectro))
                with self._lock:
                    self._espectro = espectro
                    self._fmin = fmin
                    self._fmax = fmax
                    self._vivo = True
                    self._ult = time.time()
                fallos = 0          # un barrido bueno resetea el contador y el backoff
                backoff = 3.0
                dt = time.time() - t0
                if dt < 2.0:
                    time.sleep(2.0 - dt)
            except Exception as e:
                fallos += 1
                print(f"[LectorSDR] error barrido (#{fallos}): {e}")
                with self._lock:
                    self._vivo = False
                # espera con backoff, sensible a cerrar()
                t_end = time.time() + backoff
                while self._run and time.time() < t_end:
                    time.sleep(0.2)
                # AUTO-SANACIÓN: el RSP1/sdrplay es inestable (ServiceNotResponding,
                # ReleaseDevice que cuelga). Tras varios fallos seguidos el stream suele
                # estar muerto → cerramos el device; el TOPE del bucle lo reabre solo,
                # en vez de girar para siempre sobre un handle caído.
                if fallos >= 3:
                    self._cerrar_dispositivo()   # _dev=None → el tope del bucle reabre
                    fallos = 0

    def inyectar(self, fila: dict) -> None:
        with self._lock:
            fila["sdr_espectro"] = list(self._espectro) if self._espectro else None
            fila["sdr_freq_min_hz"] = self._fmin
            fila["sdr_freq_max_hz"] = self._fmax
            fila["sdr_vivo"] = 1 if (self._vivo and (time.time() - self._ult) < 30.0) else 0
