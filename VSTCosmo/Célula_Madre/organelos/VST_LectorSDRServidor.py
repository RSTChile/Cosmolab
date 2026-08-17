#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_LectorSDRServidor — lector de radio por la WebSocket API de SDRconnect (puerto 5454)
========================================================================================
QUIÉN SOY (para retomar sin reprocesar):
  Soy el "oído de radio" del organismo cuando el SDR lo sirve **SDRconnect** (la app de
  SDRplay) en vez de SoapySDR directo. Hablo su WebSocket API y entrego a la fila las MISMAS
  columnas que el LectorSDR clásico: sdr_espectro / sdr_freq_min_hz / sdr_freq_max_hz / sdr_vivo.
  Soy un REEMPLAZO DROP-IN de LectorSDR: mismo contrato (arrancar()->bool, inyectar(fila),
  sintonizar(hz), cerrar(), .vivo), así IntegracionRadioA(lector=...) me usa sin tocar nada.

POR QUÉ EXISTO (Alexis + CC, 3-jul-2026):
  En la Mac NO hay SoapySDR, pero SÍ está SDRconnect (RSPduo). Su puerto 50000 (--server) es
  un canal propietario TLS+RSA cerrado — inservible para un cliente propio. La vía buena es la
  **WebSocket API** (llegó en SDRconnect 1.0.6; validada aquí con 1.0.8 / api_version 1.0.3):
  texto JSON de control + binario con prefijo de tipo. El MISMO cliente sirve para la Mac
  (RSPduo) y para la Pi del animalito (RSP1) — "el órgano no pertenece al hardware sino al
  organismo". Requisito operativo: en SDRconnect, config.json → global_settings
  .WebSocketInterfaceEnabled=true (si no, el 5454 no abre).

CÓMO HABLO EL PROTOCOLO (SDRconnect_WebSocket_API.pdf):
  - Envío control JSON: {"event_type": "...", "property"?: "...", "value"?: "..."}.
  - Para arrancar el espectro: set_primary_device_enable=true  +  spectrum_enable=true.
  - Leo frecuencia con get_property de device_center_frequency (Uint64 Hz) y
    device_sample_rate (Double Hz). fmin/fmax = centro ∓ sample_rate/2.
  - Binario: los 2 primeros bytes = tipo (uint16 LE). Tipo 3 = "Unsigned 8-bit spectrum FFT
    bin normalised to visible range" (Primary Device) = EXACTAMENTE sdr_espectro (bins 0–255).
    Normalizo /255 → [0,1], coherente con el rango que espera OrganoRadio. (Tipo 1=audio,
    2=IQ; 4/5/6 = secundario del dual-tuner, ignorados aquí.)

DEGRADACIÓN ELEGANTE:
  Sin `websockets`, sin servidor en 5454, o sin espectro reciente → .vivo=False, sdr_vivo=0,
  sdr_espectro=None. El organismo sigue vivo; la caja de radio queda en "—". Nunca lanzo.

MODO HEADLESS (sin ventana — lo ideal para el organismo 24/7 y para la Pi):
  El binario `SDRconnect.app/Contents/MacOS/SDRconnect_headless` levanta la WS API en 5454
  SIN GUI, pero arranca SIN dispositivo abierto. A diferencia de la GUI (que ya trae el RSPduo
  seleccionado y sintonizado), en headless es EL CLIENTE quien abre el dispositivo por la propia
  API: selected_device → set device_center_frequency/sample_rate → set_primary_device_enable →
  spectrum_enable. Este lector hace esa secuencia al conectar (idempotente: contra una GUI ya
  configurada no molesta). Lanzar headless:  ./SDRconnect_headless   (Ctrl+C para salir).

Config por entorno (con defaults sensatos):
  ANIMA_SDRWS_URI          ws://127.0.0.1:5454   (en la Pi: ws://192.168.86.33:5454, etc.)
  ANIMA_SDRWS_BINS         512                   (nº de bins que expongo; remuestreo si difiere)
  ANIMA_SDRWS_STALE_S      30                    (antigüedad máx. del espectro para seguir "vivo")
  ANIMA_SDRWS_SECONDARY    0                     (1 = usar el tuner secundario: tipo 6 en vez de 3)
  ANIMA_SDRWS_DEVICE       0                     (índice de dispositivo a seleccionar en headless)
  ANIMA_SDRWS_FREQ_HZ      99500000              (frecuencia inicial; "" o "0" = no fijar, dejar la del server)
  ANIMA_SDRWS_SAMPLE_RATE  2000000               (sample rate inicial; "" o "0" = no fijar)
"""
from __future__ import annotations
import json
import os
import struct
import threading
import time


class LectorSDRServidor:
    """Cliente WebSocket de la API de SDRconnect. Drop-in de LectorSDR (mismo contrato)."""

    def __init__(self, uri: str | None = None):
        self.uri = uri or os.environ.get("ANIMA_SDRWS_URI", "ws://127.0.0.1:5454")
        self.n_bins = int(os.environ.get("ANIMA_SDRWS_BINS", "512"))
        self.stale_s = float(os.environ.get("ANIMA_SDRWS_STALE_S", "30"))
        secundario = os.environ.get("ANIMA_SDRWS_SECONDARY", "0").lower() in ("1", "true", "yes", "on")
        # tipo de payload de espectro según tuner (3 = primario, 6 = secundario del dual-tuner)
        self._tipo_espectro = 6 if secundario else 3
        self._dev_enable_evt = "set_secondary_device_enable" if secundario else "set_primary_device_enable"
        # arranque de dispositivo (necesario contra headless; idempotente contra la GUI)
        self._dev_index = os.environ.get("ANIMA_SDRWS_DEVICE", "0")
        def _num_env(k, d):
            v = os.environ.get(k, d)
            try:
                return int(float(v)) if v not in ("", "0", None) else None
            except (TypeError, ValueError):
                return None
        self._freq_init = _num_env("ANIMA_SDRWS_FREQ_HZ", "94500000")     # 94.5 MHz WFM por defecto
        self._sr_init = _num_env("ANIMA_SDRWS_SAMPLE_RATE", "2000000")

        self._lock = threading.Lock()
        self._run = False
        self._hilo: threading.Thread | None = None
        # estado compartido (protegido por _lock)
        self._espectro: list[float] | None = None
        self._center_hz: float | None = None
        self._sample_rate: float | None = None
        self._vivo = False
        self._ult = 0.0
        self._orden_hz: float | None = None
        self._orden_pendiente = False   # hay una sintonía que aún no se ha mandado al servidor
        self._api_version: str | None = None
        # telemetría de calidad de señal (property_changed del WS): la DINÁMICA REAL (no la del
        # espectro normalizado-a-visible). El órgano la usa para enganchar/soltar emisiones.
        self._snr_db: float | None = None
        self._power_dbm: float | None = None
        self._reconectar = False        # botón 'Reactivar radio': fuerza cerrar la sesión WS

    # ----- contrato público (idéntico a LectorSDR) -----------------------------------
    @property
    def vivo(self) -> bool:
        with self._lock:
            return self._vivo and (time.time() - self._ult) < self.stale_s

    def sintonizar(self, freq_hz: float | None) -> None:
        """El órgano ordena una frecuencia; el hilo la traslada al servidor (set_property)."""
        with self._lock:
            if freq_hz is None:
                return
            self._orden_hz = float(freq_hz)
            self._orden_pendiente = True

    def reactivar(self) -> None:
        """Botón 'Reactivar radio': fuerza cerrar la sesión WS actual para reconectar limpio
        (reabre el device en el servidor SDRconnect). El _loop reconecta solo."""
        with self._lock:
            self._reconectar = True
            self._vivo = False

    def arrancar(self) -> bool:
        try:
            import websockets  # noqa: F401
        except Exception as e:
            print(f"[LectorSDRServidor] falta la librería 'websockets' ({e}) — inactivo. "
                  f"Instala: pip install websockets")
            return False
        if self._run:
            return True
        self._run = True
        self._hilo = threading.Thread(target=self._loop, daemon=True, name="LectorSDRServidor")
        self._hilo.start()
        # damos un instante a que conecte y pida el primer espectro, para reportar arranque real
        t0 = time.time()
        while time.time() - t0 < 3.0:
            if self.vivo:
                break
            time.sleep(0.1)
        estado = "vivo" if self.vivo else "conectando (aún sin espectro)"
        print(f"[LectorSDRServidor] {self.uri} · tipo_espectro={self._tipo_espectro} · {estado}")
        # devolvemos True aunque aún no haya espectro: el hilo reintenta y degrada solo.
        return True

    def cerrar(self) -> None:
        self._run = False
        h = self._hilo
        if h is not None and h.is_alive():
            h.join(timeout=2.0)

    def inyectar(self, fila: dict) -> None:
        """Escribe en la fila las columnas sdr_* — MISMA firma/columnas que LectorSDR.inyectar."""
        with self._lock:
            vivo = self._vivo and (time.time() - self._ult) < self.stale_s
            esp = list(self._espectro) if self._espectro else None
            center = self._center_hz
            sr = self._sample_rate
        if center is not None and sr is not None:
            fmin = center - sr / 2.0
            fmax = center + sr / 2.0
        else:
            fmin = fmax = None
        with self._lock:
            snr = self._snr_db; pwr = self._power_dbm
        fila["sdr_espectro"] = esp if vivo else None
        fila["sdr_freq_min_hz"] = fmin
        fila["sdr_freq_max_hz"] = fmax
        fila["sdr_vivo"] = 1 if vivo else 0
        fila["sdr_snr_db"] = snr if vivo else None       # dinámica REAL (no del display normalizado)
        fila["sdr_power_dbm"] = pwr if vivo else None

    # ----- hilo de fondo: mantiene la conexión y consume el 5454 ----------------------
    def _loop(self) -> None:
        """Reconecta con backoff; dentro, corre el ciclo asyncio de la sesión WebSocket."""
        import asyncio
        backoff = 1.0
        while self._run:
            try:
                asyncio.run(self._sesion())
                backoff = 1.0  # una sesión que terminó limpia resetea el backoff
            except (asyncio.CancelledError, Exception) as e:
                # defensa en profundidad: cualquier caída (incl. CancelledError, que es
                # BaseException) marca no-vivo y REINTENTA; el hilo del lector nunca muere.
                with self._lock:
                    self._vivo = False
                if self._run:
                    print(f"[LectorSDRServidor] sesión caída ({type(e).__name__}: {e}); "
                          f"reintento en {backoff:.0f}s")
            # espera con backoff, pero sensible a cerrar()
            t_end = time.time() + backoff
            while self._run and time.time() < t_end:
                time.sleep(0.2)
            backoff = min(backoff * 2, 15.0)

    async def _sesion(self) -> None:
        import asyncio
        import websockets

        async with websockets.connect(self.uri, max_size=None, open_timeout=6,
                                       ping_interval=20, ping_timeout=20) as ws:

            async def send(d: dict) -> None:
                await ws.send(json.dumps(d))

            # arranque de dispositivo (headless llega sin dispositivo abierto; la GUI ya lo trae)
            await send({"event_type": "selected_device", "value": str(self._dev_index)})
            await asyncio.sleep(0.4)  # dar tiempo a que el server abra el RSPduo (started=true)
            if self._sr_init is not None:
                await send({"event_type": "set_property",
                            "property": "device_sample_rate", "value": str(self._sr_init)})
            if self._freq_init is not None:
                await send({"event_type": "set_property",
                            "property": "device_center_frequency", "value": str(self._freq_init)})
            # habilitar dispositivo + espectro; pedir freq/samplerate/version
            await send({"event_type": self._dev_enable_evt, "value": "true"})
            await send({"event_type": "spectrum_enable", "value": "true"})
            await send({"event_type": "get_property", "property": "device_center_frequency"})
            await send({"event_type": "get_property", "property": "device_sample_rate"})
            await send({"event_type": "get_property", "property": "api_version"})

            async def bombear_ordenes():
                """Traslada al servidor las sintonías pedidas por el órgano (set_property)."""
                while self._run:
                    orden = None
                    with self._lock:
                        if self._orden_pendiente and self._orden_hz is not None:
                            orden = self._orden_hz
                            self._orden_pendiente = False
                    if orden is not None:
                        await send({"event_type": "set_property",
                                    "property": "device_center_frequency",
                                    "value": str(int(orden))})
                        # tras mover el LO, re-preguntamos la freq real (por si el HW cuantiza)
                        await send({"event_type": "get_property",
                                    "property": "device_center_frequency"})
                    await asyncio.sleep(0.25)

            tarea_ordenes = asyncio.create_task(bombear_ordenes())
            try:
                while self._run:
                    with self._lock:
                        if self._reconectar:
                            self._reconectar = False
                            break   # cierra la sesión → el _loop reconecta limpio
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    except asyncio.TimeoutError:
                        continue
                    if isinstance(msg, (bytes, bytearray)):
                        self._on_binario(msg)
                    else:
                        self._on_texto(msg)
            finally:
                tarea_ordenes.cancel()
                try:
                    await tarea_ordenes
                except (asyncio.CancelledError, Exception):
                    # CancelledError hereda de BaseException (no de Exception): hay que
                    # atraparlo aquí explícitamente o escapa del hilo y lo mata al cancelar.
                    pass

    def _on_texto(self, msg: str) -> None:
        try:
            d = json.loads(msg)
        except Exception:
            return
        if d.get("event_type") not in ("get_property_response", "property_changed"):
            return
        prop = d.get("property")
        val = d.get("value")
        if prop == "device_center_frequency":
            try:
                with self._lock:
                    self._center_hz = float(val)
            except (TypeError, ValueError):
                pass
        elif prop == "device_sample_rate":
            try:
                with self._lock:
                    self._sample_rate = float(val)
            except (TypeError, ValueError):
                pass
        elif prop == "api_version":
            with self._lock:
                self._api_version = str(val)
        elif prop == "signal_snr":
            try:
                with self._lock:
                    self._snr_db = float(val)
            except (TypeError, ValueError):
                pass
        elif prop == "signal_power":
            try:
                with self._lock:
                    self._power_dbm = float(val)
            except (TypeError, ValueError):
                pass

    def _on_binario(self, msg: bytes) -> None:
        if len(msg) < 2:
            return
        tipo = struct.unpack("<H", msg[:2])[0]
        if tipo != self._tipo_espectro:
            return  # audio/IQ/tuner-secundario: no es nuestro espectro
        payload = msg[2:]
        if not payload:
            return
        # bins uint8 (0–255) normalizados al rango visible → /255 a [0,1]
        bins = [b / 255.0 for b in payload]
        bins = self._remuestrear(bins, self.n_bins)
        with self._lock:
            self._espectro = bins
            self._vivo = True
            self._ult = time.time()

    @staticmethod
    def _remuestrear(xs: list[float], n: int) -> list[float]:
        """Ajusta la lista a n bins (submuestreo/relleno) SIN depender de numpy."""
        m = len(xs)
        if m == n or n <= 0:
            return xs
        if m > n:
            # índices equiespaciados
            return [xs[(i * (m - 1)) // (n - 1)] if n > 1 else xs[0] for i in range(n)]
        return xs + [0.0] * (n - m)


# Fábrica registrable por ANIMA_RADIO_A_LECTOR (para no tocar IntegracionRadioA):
def crear_lector(*_a, **_k) -> "LectorSDRServidor":
    return LectorSDRServidor()


if __name__ == "__main__":
    # Prueba en vivo: arranca, consume el 5454 unos segundos e imprime una fila.
    lec = LectorSDRServidor()
    if not lec.arrancar():
        raise SystemExit("no arrancó (¿falta websockets o el 5454 cerrado?)")
    try:
        for _ in range(6):
            time.sleep(1.0)
            fila: dict = {}
            lec.inyectar(fila)
            esp = fila["sdr_espectro"]
            n = len(esp) if esp else 0
            pico = max(esp) if esp else 0.0
            fmin = fila["sdr_freq_min_hz"]
            fmax = fila["sdr_freq_max_hz"]
            fr = f"{fmin/1e6:.3f}–{fmax/1e6:.3f} MHz" if fmin and fmax else "—"
            print(f"vivo={lec.vivo} bins={n} pico={pico:.3f} banda={fr} api={lec._api_version}")
    finally:
        lec.cerrar()
