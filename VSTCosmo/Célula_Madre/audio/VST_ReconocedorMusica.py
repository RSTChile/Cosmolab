#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_ReconocedorMusica.py — NIVEL 3 (música): identifica QUÉ tema suena, EN VIVO, sin guardar audio.

Objetivo ÚNICO: darle a los analistas la información de qué está escuchando cada organismo,
timestamped, para cruzarla con la fisiología. Nada más.

Cómo: se conecta al VST_AudioServer por TCP (cliente — NO reabre el Rode, no toca a los organismos),
toma el canal de música (def 15 = SMART Pads), con compuerta por RMS (solo trabaja cuando hay música),
acumula una ventana EN MEMORIA, la manda como HUELLA a Shazam (shazamio) y escribe SOLO el resultado
(título/artista + timestamps) a JSONL. El audio se descarta tras analizarlo.

PRIVACIDAD: envía una HUELLA acústica del audio a Shazam (servicio externo, vía la librería no oficial
shazamio). No envía el audio crudo, pero es una llamada externa — autorizada explícitamente por el usuario.

USO (con el AudioServer corriendo):
  venv/bin/python3 Célula_Madre/audio/VST_ReconocedorMusica.py --canal 15 --entrada SMARTPads

Salida (JSONL): ~/Library/Logs/vstcosmo-audioserver/musica_<fecha>.jsonl
  {"ts_real":..,"hora":"HH:MM:SS","canal":15,"entrada":"SMARTPads","titulo":"..","artista":"..","shazam_key":".."}
"""
from __future__ import annotations
import argparse
import asyncio
import io
import json
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from VST_AudioServer import AudioStreamClient  # noqa: E402


def pcm_wav_bytes(x: np.ndarray, sr_in: int) -> bytes:
    """mono float32 [-1,1] @sr_in → WAV PCM 16-bit en memoria (stdlib, SIN ffmpeg).
    Shazam/shazamio-core re-muestrea internamente; no hace falta bajar a 16k aquí."""
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr_in)
        w.writeframes(pcm.tobytes())
    return bio.getvalue()


async def reconocer(shazam, wav_bytes: bytes) -> "dict | None":
    """Devuelve {titulo, artista, shazam_key} o None si no hay match."""
    out = await shazam.recognize(wav_bytes)              # shazamio acepta bytes de un archivo de audio
    track = (out or {}).get("track") or {}
    if not track:
        return None
    return {"titulo": track.get("title"), "artista": track.get("subtitle"),
            "shazam_key": track.get("key")}


def main() -> None:
    ap = argparse.ArgumentParser(description="Identifica el tema que suena en un canal del Rode (sin guardar audio).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--canal", type=int, default=15, help="canal 1-based de música (def 15 = SMART Pads)")
    ap.add_argument("--entrada", default="SMARTPads", help="nombre legible (para el JSONL)")
    ap.add_argument("--umbral", type=float, default=0.01, help="RMS mínimo para considerar que hay música")
    ap.add_argument("--ventana", type=float, default=12.0, help="seg de música por intento de reconocimiento")
    ap.add_argument("--cada", type=float, default=30.0, help="seg mínimos entre reconocimientos (evita spam a Shazam)")
    ap.add_argument("--salida", default=None)
    args = ap.parse_args()

    try:
        from shazamio import Shazam
    except ImportError:
        sys.exit("ERROR: falta shazamio. Instala:\n  venv/bin/pip install shazamio")

    salida = args.salida or os.path.expanduser(
        "~/Library/Logs/vstcosmo-audioserver/musica_" + time.strftime("%Y%m%d_%H%M%S") + ".jsonl")
    os.makedirs(os.path.dirname(salida), exist_ok=True)

    cli = AudioStreamClient(host=args.host, port=args.port)
    hs = cli.handshake()
    sr = int(hs["sample_rate"]); nch = int(hs["channels"]); ch = args.canal - 1
    if ch < 0 or ch >= nch:
        sys.exit(f"ERROR: canal {args.canal} fuera de rango (el stream trae {nch} canales)")

    shazam = Shazam()
    loop = asyncio.new_event_loop()
    out = open(salida, "a", buffering=1)
    print(f"[música] canal={args.canal} ({args.entrada}) sr={sr}Hz; salida={salida}", flush=True)
    print("[música] reconoce solo cuando hay música; envía huella a Shazam (externo).", flush=True)

    buf: list[np.ndarray] = []
    dur = 0.0
    ult_reco = 0.0
    ult_titulo = None

    try:
        for bloque in cli.frames():
            col = bloque[:, ch].astype(np.float32)
            rms = float(np.sqrt(np.mean(col * col))) if col.size else 0.0
            ahora = time.time()
            if rms < args.umbral:                        # sin música: vacía el buffer (no acumular silencio)
                buf = []; dur = 0.0
                continue
            buf.append(col); dur += col.size / sr
            if dur < args.ventana or (ahora - ult_reco) < args.cada:
                continue
            audio = np.concatenate(buf).astype(np.float32)
            buf = []; dur = 0.0; ult_reco = ahora
            try:
                res = loop.run_until_complete(reconocer(shazam, pcm_wav_bytes(audio, sr)))
            except Exception as e:                       # red caída, rate-limit, etc.: no abortar
                print(f"  [reco err] {e}", flush=True); continue
            if not res or not res.get("titulo"):
                continue
            if res["titulo"] == ult_titulo:              # mismo tema: no repetir log
                continue
            ult_titulo = res["titulo"]
            rec = {"ts_real": round(ahora, 3), "hora": time.strftime("%H:%M:%S"),
                   "canal": args.canal, "entrada": args.entrada, **res}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  [{rec['hora']}] {args.entrada}: ♪ {res['titulo']} — {res.get('artista')}", flush=True)
    except KeyboardInterrupt:
        print("\n[música] deteniendo…", flush=True)
    finally:
        out.close()
        try:
            cli.cerrar()
        except Exception:
            pass
        loop.close()


if __name__ == "__main__":
    main()
