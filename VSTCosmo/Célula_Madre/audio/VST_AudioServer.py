#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_AudioServer — CAPTURA NATIVA DE AUDIO  ·  PUENTE POR RED HACIA EL DOCKER
================================================================================

QUÉ ES Y POR QUÉ EXISTE
-----------------------
Docker Desktop en Mac corre los contenedores dentro de una VM Linux que NO tiene
acceso al CoreAudio del Mac. Por eso un contenedor no puede leer la Rødecaster
(ni ningún dispositivo de audio del host) por más que se le pase `--device /dev/snd`:
ese dispositivo no existe dentro de la VM. (Confirmado: docker/for-mac #6789.)

La solución es no meter el audio AL contenedor, sino servírselo POR RED:

    [ Rødecaster ] --USB--> [ Mac, NATIVO ] --TCP--> [ contenedor Docker ]
                            (este programa)          (servidor cosmosemiótico)

Este programa corre NATIVO en el Mac (fuera de Docker), donde sounddevice sí ve
la Rødecaster. Lee dos canales (oído izquierdo / oído derecho), y los sirve como
un stream TCP. El contenedor se conecta como cliente y recibe los bloques L/R.
La frontera Mac↔contenedor es una conexión de red — eso Docker en Mac SÍ lo hace.

VENTAJA DE ESCALA: este mismo diseño sirve cuando se migre al DGX (Linux). Allí
el capturador puede correr nativo igual, o —si el DGX tiene la interfaz de audio
conectada— el contenedor podría usar /dev/snd directo (Linux sí lo permite). El
puente por red funciona en ambos mundos; el passthrough directo solo en Linux.

DEPENDENCIAS
------------
  · numpy        (ya instalado en el proyecto)
  · sounddevice  (venv/bin/pip install sounddevice ; requiere `brew install portaudio`)
  · Nada más: el servidor TCP usa solo la stdlib (socket, struct, threading).

PROTOCOLO DE RED (simple y robusto)
-----------------------------------
Todo mensaje va precedido por su longitud: [4 bytes big-endian][payload].
  1) HANDSHAKE (primer mensaje, JSON utf-8): describe el stream, p.ej.
     {"sample_rate":48000, "block_size":1024, "channels":2, "dtype":"<f4",
      "left":3, "right":4, "device":"RØDECaster Pro II Main"}
  2) FRAMES (resto): binario crudo, float32 little-endian, intercalado L,R,L,R,...
     de tamaño block_size*2 floats. El consumidor reconstruye con el handshake.

GARANTÍA DE TIEMPO REAL
-----------------------
El callback de audio NUNCA bloquea por red: deja cada bloque en una cola acotada
por cliente; si un cliente va lento y su cola se llena, se DESCARTAN bloques (se
prioriza mantener la captura en tiempo real sobre no perder datos de un cliente
lento). El envío por socket ocurre en hilos por-cliente, no en el callback.

USO (el servidor entrega TODOS los canales; la elección de L/R es del CLIENTE)
---
  # 1) Ver dispositivos disponibles:
  venv/bin/python3 VST_AudioServer.py --list

  # 2) Servir TODOS los canales de la Rødecaster (Main Multitrack, 20 canales):
  venv/bin/python3 VST_AudioServer.py --device "Main Multitrack"

  # 3) Verificar el stream SIN el contenedor (imprime RMS de cada canal con señal):
  venv/bin/python3 VST_AudioServer.py --test --port 8765

NUMERACIÓN DE CANALES (1-based) — la usa el CLIENTE para elegir L/R, NO el servidor
(que entrega todos). Tabla REAL Rødecaster Pro (verificada 23-jun-2026), pares L/R:
  1-2 Main Mix · 3-4 Combo 1 · 5-6 Combo 2 · 7-8 Combo 3 · 9-10 Bluetooth
  11-12 USB 2 · 13-14 USB Main · 15-16 SMART Pads · 17-18 USB Chat

CÓMO LO CONSUME EL CONTENEDOR
-----------------------------
El servidor cosmosemiótico (dentro de Docker) se conecta al host con:
    cliente = AudioStreamClient(host="host.docker.internal", port=8765)
    hs = cliente.handshake()                 # dict con sample_rate, block_size...
    for L, R in cliente.frames():            # arrays numpy float32 (block_size,)
        ...   # meter L y R al ciclo metabólico de la célula
`host.docker.internal` es el nombre con que un contenedor en Docker Desktop
alcanza al Mac anfitrión. La clase AudioStreamClient está al final de este archivo
para que el contenedor pueda importarla: `from VST_AudioServer import AudioStreamClient`.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import threading
import time
from queue import Queue, Full, Empty

import numpy as np

try:
    import sounddevice as sd
except Exception:  # noqa: BLE001 — queremos un mensaje claro, no un traceback
    sd = None


# ==============================================================================
# UTILIDADES DE PROTOCOLO  (longitud-prefijo de 4 bytes big-endian)
# ==============================================================================
_LEN = struct.Struct(">I")  # unsigned int de 4 bytes, big-endian


def enviar_mensaje(sock: socket.socket, payload: bytes) -> None:
    """Envía un mensaje con prefijo de longitud. Puede lanzar OSError si el peer cerró."""
    sock.sendall(_LEN.pack(len(payload)) + payload)


def _recv_exacto(sock: socket.socket, n: int) -> bytes:
    """Lee EXACTAMENTE n bytes del socket (o lanza ConnectionError si se corta)."""
    trozos = []
    faltan = n
    while faltan > 0:
        b = sock.recv(faltan)
        if not b:
            raise ConnectionError("conexión cerrada por el otro extremo")
        trozos.append(b)
        faltan -= len(b)
    return b"".join(trozos)


def recibir_mensaje(sock: socket.socket) -> bytes:
    """Lee un mensaje con prefijo de longitud."""
    cab = _recv_exacto(sock, _LEN.size)
    (n,) = _LEN.unpack(cab)
    return _recv_exacto(sock, n)


# ==============================================================================
# ENUMERACIÓN DE DISPOSITIVOS  (--list)
# ==============================================================================
def listar_dispositivos() -> None:
    if sd is None:
        print("ERROR: sounddevice no está instalado.\n"
              "  brew install portaudio\n"
              "  venv/bin/pip install sounddevice")
        sys.exit(1)
    print("=" * 78)
    print("DISPOSITIVOS DE ENTRADA DISPONIBLES (canales 1-based, como la guía RØDE)")
    print("=" * 78)
    disp = sd.query_devices()
    hostapis = sd.query_hostapis()
    for i, d in enumerate(disp):
        if d["max_input_channels"] > 0:
            api = hostapis[d["hostapi"]]["name"]
            print(f"  [{i:2d}] {d['name']}")
            print(f"       entradas={d['max_input_channels']}  "
                  f"sr_def={int(d['default_samplerate'])}Hz  api={api}")
    print("-" * 78)
    print("Para la Rødecaster Pro (multitrack ON), todos los canales aparecen como UN")
    print("dispositivo. El servidor los entrega todos; el CONSUMIDOR elige L/R. Pares L/R:")
    print("  1-2 Main Mix · 3-4 Combo 1 · 5-6 Combo 2 · 7-8 Combo 3 · 9-10 Bluetooth")
    print("  11-12 USB 2 · 13-14 USB Main · 15-16 SMART Pads · 17-18 USB Chat")
    print("=" * 78)


def resolver_dispositivo(nombre_o_indice: str) -> int:
    """Acepta un índice ('5') o una subcadena del nombre ('RØDECaster'). Devuelve índice."""
    disp = sd.query_devices()
    # ¿es un índice numérico? (distinguir "no es número" de "número inválido")
    if nombre_o_indice.strip().lstrip("-").isdigit():
        idx = int(nombre_o_indice)
        if 0 <= idx < len(disp) and disp[idx]["max_input_channels"] > 0:
            return idx
        print(f"ERROR: el índice {idx} no es un dispositivo de entrada válido. Usa --list.")
        sys.exit(1)
    # buscar por subcadena (case-insensitive)
    cand = [i for i, d in enumerate(disp)
            if d["max_input_channels"] > 0 and nombre_o_indice.lower() in d["name"].lower()]
    if not cand:
        print(f"ERROR: ningún dispositivo de entrada coincide con '{nombre_o_indice}'.")
        print("Usa --list para ver los disponibles.")
        sys.exit(1)
    if len(cand) > 1:
        nombres = ", ".join(f"[{i}] {disp[i]['name']}" for i in cand)
        print(f"AVISO: '{nombre_o_indice}' coincide con varios: {nombres}. Uso el primero ([{cand[0]}]).")
    return cand[0]


# ==============================================================================
# SERVIDOR DE FRAMES  (acepta clientes TCP y les difunde los bloques de audio)
# ==============================================================================
class ServidorFrames:
    """Servidor TCP que difunde bloques de audio a todos los clientes conectados.

    Cada cliente tiene su propia cola acotada. El productor (callback de audio)
    hace put_nowait en cada cola; si una cola está llena (cliente lento), descarta
    el bloque para ESE cliente, sin frenar la captura ni a los demás.
    """

    def __init__(self, host: str, port: int, handshake: dict, cola_max: int = 32) -> None:
        self.host = host
        self.port = port
        self.handshake_bytes = json.dumps(handshake).encode("utf-8")
        self.cola_max = cola_max
        self._clientes: dict[int, Queue] = {}   # id(sock) -> cola
        self._lock = threading.Lock()
        self._corriendo = threading.Event()
        self._corriendo.set()
        self._srv: socket.socket | None = None
        self.descartes = 0   # bloques descartados por colas llenas (diagnóstico)

    # ---- difusión: la llama el callback de audio (debe ser rápido) ----
    def difundir(self, frame: bytes) -> None:
        with self._lock:
            colas = list(self._clientes.values())
        for q in colas:
            try:
                q.put_nowait(frame)
            except Full:
                # cliente lento: descartar el bloque más viejo y meter el nuevo
                try:
                    q.get_nowait()
                    q.put_nowait(frame)
                except (Empty, Full):
                    pass
                self.descartes += 1

    # ---- aceptar conexiones ----
    def iniciar(self) -> None:
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.host, self.port))
        self._srv.listen(8)
        self._srv.settimeout(0.5)
        t = threading.Thread(target=self._loop_accept, daemon=True)
        t.start()
        print(f"  ► servidor TCP escuchando en {self.host}:{self.port} "
              f"(contenedor: host.docker.internal:{self.port})")

    def _loop_accept(self) -> None:
        while self._corriendo.is_set():
            try:
                cli, addr = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # baja latencia
            threading.Thread(target=self._servir_cliente, args=(cli, addr), daemon=True).start()

    def _servir_cliente(self, cli: socket.socket, addr) -> None:
        q: Queue = Queue(maxsize=self.cola_max)
        with self._lock:
            self._clientes[id(cli)] = q
        print(f"  + cliente conectado: {addr}  (total: {len(self._clientes)})")
        try:
            enviar_mensaje(cli, self.handshake_bytes)   # 1) handshake
            while self._corriendo.is_set():             # 2) frames
                try:
                    frame = q.get(timeout=0.5)
                except Empty:
                    continue
                enviar_mensaje(cli, frame)
        except (OSError, ConnectionError):
            pass
        finally:
            with self._lock:
                self._clientes.pop(id(cli), None)
            try:
                cli.close()
            except OSError:
                pass
            print(f"  - cliente desconectado: {addr}  (total: {len(self._clientes)})")

    @property
    def n_clientes(self) -> int:
        with self._lock:
            return len(self._clientes)

    def detener(self) -> None:
        self._corriendo.clear()
        if self._srv:
            try:
                self._srv.close()
            except OSError:
                pass


# ==============================================================================
# CAPTURA DE AUDIO  (sounddevice → extrae L/R → difunde)
# ==============================================================================
class CapturaAudio:
    """Abre la entrada del dispositivo y difunde TODOS sus canales.

    Filosofía: el servidor no decide qué es L y qué es R. Captura todos los
    canales disponibles del dispositivo y los sirve completos; la asignación
    de oído izquierdo/derecho (qué canal es cuál) la hace el CONSUMIDOR (el
    container), que recibe todos y elige. Así nunca falta un canal por no
    haberlo pedido, y se puede reasignar L/R sin reiniciar la captura.

    El handshake informa `channels` = nº de canales servidos. Los frames van
    intercalados c0,c1,...,c{n-1}, c0,c1,... por sample.
    """

    # Mapa verificado RØDECaster Pro II (pares L/R 1-based) → ENTRADA física.
    INPUT_PAIRS = [
        ("MainMix", (1, 2)), ("Combo1", (3, 4)), ("Combo2", (5, 6)), ("Combo3", (7, 8)),
        ("Bluetooth", (9, 10)), ("USB2", (11, 12)), ("USBMain", (13, 14)),
        ("SMARTPads", (15, 16)), ("USBChat", (17, 18)),
    ]

    def __init__(self, servidor: ServidorFrames, device_idx: int,
                 n_canales: int, sample_rate: int, block_size: int,
                 log_canales_path: "str | None" = None) -> None:
        self.servidor = servidor
        self.device_idx = device_idx
        self.n_canales = n_canales
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.bloques = 0
        self._stream: "sd.InputStream | None" = None
        # --- LOG por ENTRADA (Nivel 1): RMS de cada entrada del Rode cada ~0.25 s ---
        self.log_canales_path = log_canales_path
        self._logging = bool(log_canales_path)
        if self._logging:
            self.log_intervalo = 0.25
            self._pairs = [(nm, (a - 1, b - 1)) for nm, (a, b) in self.INPUT_PAIRS if a <= n_canales]
            self._ss = np.zeros(n_canales, dtype=np.float64)   # suma de cuadrados por canal
            self._nss = 0
            self._loglock = threading.Lock()
            self._logstop = False
            os.makedirs(os.path.dirname(log_canales_path), exist_ok=True)
            self._logf = open(log_canales_path, "w", buffering=1)
            self._logf.write("ts_real,hora," + ",".join(nm for nm, _ in self._pairs) + "\n")
            self._logthread = threading.Thread(target=self._logger_loop, daemon=True)

    def _callback(self, indata: np.ndarray, frames: int, t, status) -> None:
        # indata: (block_size, n_canales) float32. NO bloquear aquí.
        # tobytes() de un array C-contiguo (block, ch) ya intercala por sample:
        # [s0c0, s0c1, ..., s0c{n-1}, s1c0, ...] — exactamente lo que el cliente espera.
        if status:
            pass  # underflow/overflow ocasional: no abortamos
        self.servidor.difundir(np.ascontiguousarray(indata, dtype="<f4").tobytes())
        self.bloques += 1
        if self._logging:                       # acumula energía por canal (barato y vectorizado)
            x = indata.astype(np.float64)
            with self._loglock:
                self._ss += np.sum(x * x, axis=0)
                self._nss += frames

    def _logger_loop(self) -> None:
        # Cada ~0.25 s vuelca el RMS por ENTRADA del Rode (promedio de su par L/R) a un CSV.
        # Solo se guardan números (volumen por entrada) + timestamp: NO se guarda audio.
        while not self._logstop:
            time.sleep(self.log_intervalo)
            with self._loglock:
                n = self._nss; ss = self._ss.copy()
                self._ss[:] = 0.0; self._nss = 0
            if n <= 0:
                continue
            rms = np.sqrt(ss / n)
            vals = []
            for _nm, (a, b) in self._pairs:
                ch = [c for c in (a, b) if c < len(rms)]
                vals.append(f"{float(np.mean(rms[ch])) if ch else 0.0:.6f}")
            self._logf.write(f"{time.time():.3f},{time.strftime('%H:%M:%S')}," + ",".join(vals) + "\n")

    def iniciar(self) -> None:
        if sd is None:
            print("ERROR: sounddevice no está instalado.")
            sys.exit(1)
        info = sd.query_devices(self.device_idx)
        try:
            self._stream = sd.InputStream(
                device=self.device_idx,
                channels=self.n_canales,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
            if self._logging:
                self._logthread.start()
                print(f"  ► LOG por entrada activo → {self.log_canales_path}")
        except Exception as e:  # noqa: BLE001 — mensaje claro, no traceback
            print(f"\nERROR al abrir la entrada de '{info['name']}' "
                  f"(sr={self.sample_rate}, canales={self.n_canales}): {e}\n"
                  "  • PERMISO de Micrófono macOS: Ajustes → Privacidad y seguridad → Micrófono → Terminal/Python.\n"
                  "  • Rødecaster a 48 kHz y multitrack ON; o ajusta --rate / --canales.\n"
                  "  • Otro programa puede tener el dispositivo tomado.")
            sys.exit(1)
        print(f"  ► capturando de [{self.device_idx}] {info['name']}")
        print(f"    TODOS los {self.n_canales} canales (la asignación L/R la hace el consumidor)")
        print(f"    {self.sample_rate} Hz · bloque {self.block_size} samples "
              f"(~{1000.0*self.block_size/self.sample_rate:.1f} ms/bloque)")

    def detener(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass


# ==============================================================================
# CLIENTE DE STREAM  (lo importa el contenedor para consumir el audio)
# ==============================================================================
class AudioStreamClient:
    """Cliente que el servidor cosmosemiótico (en Docker) usa para recibir el audio.

    Uso típico dentro del contenedor:
        c = AudioStreamClient(host="host.docker.internal", port=8765)
        hs = c.handshake()                      # dict (incluye 'channels' = nº canales)
        for bloque in c.frames():               # array numpy (block_size, channels) float32
            L = bloque[:, idx_L]                # el container elige qué canal es L
            R = bloque[:, idx_R]                # y cuál es R
            metabolizar(L, R)
        c.cerrar()
    Reconecta no es automático: si la conexión cae, frames() termina; el llamador
    decide si reintenta (recomendado: re-crear el cliente con backoff).
    """

    def __init__(self, host: str = "host.docker.internal", port: int = 8765,
                 timeout: float = 10.0) -> None:
        self.host = host
        self.port = port
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._hs: dict | None = None

    def handshake(self) -> dict:
        if self._hs is None:
            self._hs = json.loads(recibir_mensaje(self._sock).decode("utf-8"))
        return self._hs

    def frames(self):
        """Generador que produce cada bloque como array numpy (block_size, channels) float32.

        El consumidor elige qué columnas usar como L y R. Ej.: para Combo 1 y Combo 2
        de la Rødecaster (canales 1-based 3 y 4) → bloque[:, 2] y bloque[:, 3].
        """
        hs = self.handshake()
        nch = int(hs["channels"])
        dtype = hs.get("dtype", "<f4")
        while True:
            try:
                payload = recibir_mensaje(self._sock)
            except (OSError, ConnectionError):
                return
            flat = np.frombuffer(payload, dtype=dtype)
            if flat.size == 0 or flat.size % nch != 0:
                continue                              # bloque vacío/corrupto: saltar
            yield flat.reshape(-1, nch).copy()        # acepta CUALQUIER tamaño de bloque (PortAudio puede variar)

    def cerrar(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


# ==============================================================================
# MODO PRUEBA  (--test): conecta como cliente e imprime el RMS de TODOS los canales
# ==============================================================================
def modo_prueba(host: str, port: int) -> None:
    print(f"  conectando a {host}:{port} ...")
    try:
        c = AudioStreamClient(host=host, port=port, timeout=5.0)
    except OSError as e:
        print(f"ERROR: no se pudo conectar ({e}). ¿Está corriendo el servidor?")
        sys.exit(1)
    hs = c.handshake()
    print(f"  handshake: {hs}")
    nch = int(hs["channels"]); bs = int(hs["block_size"]); sr = int(hs["sample_rate"])
    print(f"  midiendo RMS de los {nch} canales — solo se muestran los que tienen señal")
    print("  (mete sonido por donde sea y mira qué canal se enciende). Ctrl+C para salir:")
    t0 = time.time(); recibidos = 0; rms = np.zeros(nch)
    try:
        for bloque in c.frames():
            recibidos += 1
            inst = np.sqrt(np.mean(bloque.astype(np.float64) ** 2, axis=0))
            rms = 0.7 * rms + 0.3 * inst   # suavizado
            if time.time() - t0 >= 0.5:
                seg = recibidos * bs / sr
                activos = [(c_i + 1, rms[c_i]) for c_i in range(nch) if rms[c_i] > 1e-4]
                if activos:
                    print(f"  t={seg:6.1f}s  " + "   ".join(
                        f"ch{n}={v:.4f}{'#' * min(20, int(v * 120))}" for n, v in activos))
                else:
                    print(f"  t={seg:6.1f}s  (todos los canales en silencio)")
                t0 = time.time()
    except KeyboardInterrupt:
        print("\n  prueba terminada.")
    finally:
        c.cerrar()


# ==============================================================================
# MAIN
# ==============================================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description="Servidor de audio nativo: captura la Rødecaster y la sirve por red al Docker.")
    p.add_argument("--list", action="store_true", help="listar dispositivos de entrada y salir")
    p.add_argument("--test", action="store_true", help="modo cliente de prueba (imprime RMS L/R)")
    p.add_argument("--device", default="Main Multitrack",
                   help="índice o subcadena del nombre del dispositivo "
                        "(def: 'Main Multitrack' = Rødecaster de 16/20 canales, NO el Chat de 2)")
    p.add_argument("--canales", type=int, default=0,
                   help="nº de canales a capturar (def: 0 = TODOS los del dispositivo)")
    p.add_argument("--rate", type=int, default=48000, help="sample rate (def: 48000)")
    p.add_argument("--block", type=int, default=1024, help="tamaño de bloque en samples (def: 1024)")
    p.add_argument("--host", default="0.0.0.0",
                   help="interfaz donde escuchar (def: 0.0.0.0 = todas; el contenedor llega por host.docker.internal)")
    p.add_argument("--port", type=int, default=8765, help="puerto TCP (def: 8765)")
    p.add_argument("--log-canales", nargs="?", const="AUTO", default=None,
                   help="Nivel 1: registrar el RMS (volumen) de cada ENTRADA del Rode a un CSV "
                        "para cruzar con la fisiología. Opcional: ruta; sin ruta usa "
                        "~/Library/Logs/vstcosmo-audioserver/canales_rms_<fecha>.csv. "
                        "También con env VST_LOG_CANALES=1. No guarda audio, solo números.")
    args = p.parse_args()

    if args.list:
        listar_dispositivos()
        return

    if args.test:
        host = args.host if args.host != "0.0.0.0" else "127.0.0.1"
        modo_prueba(host, args.port)
        return

    if sd is None:
        print("ERROR: sounddevice no está instalado.\n"
              "  brew install portaudio\n"
              "  venv/bin/pip install sounddevice")
        sys.exit(1)

    device_idx = resolver_dispositivo(args.device)
    info = sd.query_devices(device_idx)
    max_ch = int(info["max_input_channels"])
    n_canales = max_ch if args.canales <= 0 else min(args.canales, max_ch)

    handshake = {
        "sample_rate": args.rate,
        "block_size": args.block,
        "channels": n_canales,       # TODOS los canales servidos; el consumidor elige L/R
        "dtype": "<f4",
        "device": info["name"],
    }

    print("=" * 78)
    print("VST_AudioServer — captura nativa → puente TCP al Docker")
    print("=" * 78)

    # Resolver destino del log por entrada (flag o env). 'AUTO' → ruta por defecto con fecha.
    log_path = args.log_canales
    if log_path is None and os.environ.get("VST_LOG_CANALES", "").strip():
        v = os.environ["VST_LOG_CANALES"].strip()
        log_path = "AUTO" if v.lower() in ("1", "true", "yes", "si", "sí") else v
    if log_path == "AUTO":
        _d = os.path.expanduser("~/Library/Logs/vstcosmo-audioserver")
        log_path = os.path.join(_d, "canales_rms_" + time.strftime("%Y%m%d_%H%M%S") + ".csv")

    servidor = ServidorFrames(args.host, args.port, handshake)
    captura = CapturaAudio(servidor, device_idx, n_canales, args.rate, args.block,
                           log_canales_path=log_path)

    servidor.iniciar()
    captura.iniciar()
    print("-" * 78)
    print(f"  Sirviendo {n_canales} canales. El consumidor elige cuál es L y cuál R.")
    print("  En el contenedor:  AudioStreamClient(host='host.docker.internal', "
          f"port={args.port})")
    print("    bloque = (block_size, channels);  L = bloque[:, idx_L],  R = bloque[:, idx_R]")
    print("  Para ver qué canal tiene señal, en otra terminal:")
    print(f"    venv/bin/python3 VST_AudioServer.py --test --port {args.port}")
    print("  Ctrl+C para detener.")
    print("=" * 78)

    try:
        while True:
            time.sleep(2.0)
            print(f"  · clientes={servidor.n_clientes}  bloques={captura.bloques}  "
                  f"descartes={servidor.descartes}")
    except KeyboardInterrupt:
        print("\n  deteniendo ...")
    finally:
        captura.detener()
        servidor.detener()
        print("  servidor detenido.")


if __name__ == "__main__":
    main()
