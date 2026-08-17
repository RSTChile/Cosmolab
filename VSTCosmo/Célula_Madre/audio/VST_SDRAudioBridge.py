#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_SDRAudioBridge — puente que convierte el AUDIO del SDR en un "servidor de audio" más
============================================================================================
QUIÉN SOY (para retomar sin reprocesar):
  Soy un segundo AudioServer, pero mi fuente NO es un dispositivo físico (Rødecaster): es el
  AUDIO DEMODULADO del SDR (WFM/NFM/AM) que sirve SDRconnect por su WebSocket API (puerto 5454,
  binario tipo 1 = "Signed 16-bit PCM Stereo Audio @ 48kHz LRLR", Primary Device). Lo recibo,
  lo convierto a float32 y lo DIFUNDO por TCP con EXACTAMENTE el mismo protocolo que
  VST_AudioServer.py (mismo handshake, mismos frames). Así, para el organismo, el SDR aparece
  como otro "📡 servidor" con 2 canales (L/R) en su selector de fuentes — sin código nuevo de
  consumo: reusa _LectorServidor tal cual.

POR QUÉ EXISTO (Alexis + CC, 3-jul-2026):
  El AudioServer clásico captura de un dispositivo nativo (Docker está aislado del audio, y el
  audio del SDR no es un dispositivo del sistema). En vez de rutear por un loopback virtual
  (BlackHole), traigo el audio directamente de la WS API y lo publico con el protocolo que el
  laboratorio ya sabe consumir. Un solo puerto extra, cero dependencias nuevas. El MISMO puente
  sirve para la Mac (RSPduo) y para la Pi (RSP1): solo cambia ANIMA_SDRWS_URI.

CÓMO HABLO CON SDRconnect (idéntico arranque que VST_LectorSDRServidor, modo headless):
  selected_device → set device_center_frequency/sample_rate → set_primary_device_enable →
  audio_stream_enable. Recibo tipo 1 (int16 LRLR) → float32 [-1,1] → difundir.

Config por entorno:
  ANIMA_SDRWS_URI          ws://127.0.0.1:5454   (Pi: ws://192.168.86.33:5454)
  ANIMA_SDR_AUDIO_PORT     8771                  (puerto TCP de ESTE servidor de audio SDR)
  ANIMA_SDR_AUDIO_HOST     0.0.0.0
  ANIMA_SDRWS_DEVICE       0                     (índice de dispositivo a abrir en headless)
  ANIMA_SDRWS_FREQ_HZ      94500000              (94.5 MHz WFM; "" o "0" = no fijar)
  ANIMA_SDRWS_SAMPLE_RATE  2000000               (sample rate del SDR; el AUDIO siempre sale a 48 kHz)
  ANIMA_SDR_AUDIO_SECONDARY 0                    (1 = tuner secundario: tipo 4 en vez de 1)
  ANIMA_SDR_AUDIO_ETIQUETA "SDR 94.5 MHz WFM"    (nombre del "device" en el handshake → etiqueta en el selector)

USO:
  .venv-mac/bin/python3 audio/VST_SDRAudioBridge.py
  (requiere: websockets, numpy; SDRconnect_headless corriendo con WebSocketInterfaceEnabled=true)
"""
from __future__ import annotations
import asyncio
import json
import os
import struct
import sys

import numpy as np

# Reusar el servidor de frames y el framing del AudioServer (mismo protocolo byte a byte)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from VST_AudioServer import ServidorFrames  # noqa: E402

AUDIO_SR = 48000  # la WS API entrega el audio demodulado a 48 kHz (= SR de la célula)


class PuenteAudioSDR:
    """Conecta a la WS API de SDRconnect, recibe audio PCM tipo 1 y lo difunde por ServidorFrames."""

    def __init__(self) -> None:
        self.uri = os.environ.get("ANIMA_SDRWS_URI", "ws://127.0.0.1:5454")
        self.port = int(os.environ.get("ANIMA_SDR_AUDIO_PORT", "8771"))
        self.host = os.environ.get("ANIMA_SDR_AUDIO_HOST", "0.0.0.0")
        self.dev_index = os.environ.get("ANIMA_SDRWS_DEVICE", "0")
        secundario = os.environ.get("ANIMA_SDR_AUDIO_SECONDARY", "0").lower() in ("1", "true", "yes", "on")
        self._tipo_audio = 4 if secundario else 1  # 1 = PCM estéreo primario; 4 = secundario
        self._dev_enable_evt = "set_secondary_device_enable" if secundario else "set_primary_device_enable"
        self.etiqueta = os.environ.get("ANIMA_SDR_AUDIO_ETIQUETA", "SDR 94.5 MHz WFM")

        def _num(k, d):
            v = os.environ.get(k, d)
            try:
                return int(float(v)) if v not in ("", "0", None) else None
            except (TypeError, ValueError):
                return None
        self._freq_init = _num("ANIMA_SDRWS_FREQ_HZ", "94500000")
        self._sr_init = _num("ANIMA_SDRWS_SAMPLE_RATE", "2000000")

        handshake = {"sample_rate": AUDIO_SR, "block_size": 0, "channels": 2,
                     "dtype": "<f4", "device": self.etiqueta}
        self.servidor = ServidorFrames(self.host, self.port, handshake)

    def iniciar(self) -> None:
        self.servidor.iniciar()
        print(f"  ► puente de AUDIO SDR: {self.uri}  →  TCP {self.host}:{self.port}  "
              f"(device='{self.etiqueta}', tipo_audio={self._tipo_audio})")
        # bucle de reconexión: si el SDR/servidor cae, reintenta sin morir
        backoff = 1.0
        while True:
            try:
                asyncio.run(self._sesion())
                backoff = 1.0
            except KeyboardInterrupt:
                print("\n[puente SDR] Ctrl+C — saliendo")
                break
            except Exception as e:
                print(f"[puente SDR] sesión caída ({type(e).__name__}: {e}); reintento en {backoff:.0f}s")
                import time
                time.sleep(backoff)
                backoff = min(backoff * 2, 15.0)

    async def _sesion(self) -> None:
        """Sesión WS con demodulación BAJO DEMANDA (economía metabólica): solo pide
        audio_stream_enable=true MIENTRAS haya ≥1 cliente escuchando el 8771. Cuando nadie
        escucha (el organismo no seleccionó la 📻), apaga el audio y SDRconnect deja de
        demodular — el SDR no queda 'permanentemente prendido'. El SELECCIONAR (un cliente
        conecta) es lo que enciende; el deseleccionar (se va el último) lo apaga."""
        import websockets
        async with websockets.connect(self.uri, max_size=None, open_timeout=6,
                                      ping_interval=20, ping_timeout=20) as ws:
            async def send(d):
                await ws.send(json.dumps(d))
            # arranque de dispositivo (headless llega sin dispositivo abierto). NOTA: la FRECUENCIA
            # la fija el ÓRGANO cuando está activo (radio_orden_hz → set_property por el lector del
            # espectro, mismo SDRconnect); aquí solo ponemos un default inicial si se configuró.
            await send({"event_type": "selected_device", "value": str(self.dev_index)})
            await asyncio.sleep(0.4)
            if self._sr_init is not None:
                await send({"event_type": "set_property", "property": "device_sample_rate",
                            "value": str(self._sr_init)})
            if self._freq_init is not None:
                await send({"event_type": "set_property", "property": "device_center_frequency",
                            "value": str(self._freq_init)})
            await send({"event_type": self._dev_enable_evt, "value": "true"})

            audio_on = False  # estado de la demodulación; se enciende/apaga según haya oyentes

            async def gestor_demanda():
                """Enciende/apaga la demodulación siguiendo el nº de clientes conectados al 8771."""
                nonlocal audio_on
                while True:
                    hay_oyentes = self.servidor.n_clientes > 0
                    if hay_oyentes and not audio_on:
                        await send({"event_type": "audio_stream_enable", "value": "true"})
                        audio_on = True
                        print(f"  ▶ oyente presente → audio SDR ENCENDIDO ({self.servidor.n_clientes} cliente/s)")
                    elif not hay_oyentes and audio_on:
                        await send({"event_type": "audio_stream_enable", "value": "false"})
                        audio_on = False
                        print("  ⏸ sin oyentes → audio SDR APAGADO (SDRconnect deja de demodular)")
                    await asyncio.sleep(0.5)

            tarea = asyncio.create_task(gestor_demanda())
            try:
                while True:
                    msg = await ws.recv()
                    if audio_on and isinstance(msg, (bytes, bytearray)) and len(msg) >= 2:
                        tipo = struct.unpack("<H", msg[:2])[0]
                        if tipo == self._tipo_audio:
                            self._difundir_pcm(msg[2:])
            finally:
                tarea.cancel()
                try:
                    await tarea
                except Exception:
                    pass

    def _difundir_pcm(self, payload: bytes) -> None:
        """int16 LRLR → float32 [-1,1] intercalado 2ch → difundir (mismo formato que CapturaAudio)."""
        if len(payload) < 4:
            return
        n = (len(payload) // 4) * 4  # múltiplo de 4 bytes (par de int16 = 1 sample estéreo)
        pcm = np.frombuffer(payload[:n], dtype="<i2").astype(np.float32) / 32768.0
        # ya viene LRLR intercalado; el consumidor reformatea a (-1, 2) por el handshake
        self.servidor.difundir(np.ascontiguousarray(pcm, dtype="<f4").tobytes())


if __name__ == "__main__":
    PuenteAudioSDR().iniciar()
