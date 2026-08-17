#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_Transcriptor.py — NIVEL 3: voz → texto EN VIVO, sin guardar audio.

Se conecta al VST_AudioServer por TCP (como un cliente más — NO reabre el Rode, no
toca a los organismos), toma UN canal del multitrack (p.ej. Bluetooth = la película),
acumula ventanas de voz EN MEMORIA, las transcribe con faster-whisper (local, offline)
y escribe SOLO el transcript con timestamps (ts_real) para cruzarlo con la fisiología y
con el log de canales (Nivel 1). El audio se descarta tras analizarlo.

Engancha con el Nivel 1: una compuerta de energía (RMS) hace que SOLO se transcriba
cuando ese canal tiene señal; el silencio cierra la ventana y dispara la transcripción.

USO (con el AudioServer corriendo en el host):
  venv/bin/python3 Célula_Madre/audio/VST_Transcriptor.py --canal 9 --entrada Bluetooth \
      --modelo base --idioma es

Mapa de canales (1-based): 1-2 MainMix · 3-4/5-6/7-8 Combo1-3 · 9-10 Bluetooth ·
  11-12 USB2 · 13-14 USBMain · 15-16 SMARTPads · 17-18 USBChat

Salida (JSONL): ~/Library/Logs/vstcosmo-audioserver/transcript_<fecha>.jsonl
  {"ts_real_ini":..,"ts_real_fin":..,"hora":"HH:MM:SS","canal":9,"entrada":"Bluetooth","texto":".."}
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

# Reutiliza el cliente TCP del AudioServer (mismo directorio).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from VST_AudioServer import AudioStreamClient  # noqa: E402


def resample_48k_to_16k(x: np.ndarray, sr_in: int) -> np.ndarray:
    """48 kHz → 16 kHz para Whisper. Si sr_in/16000 es entero, decima con anti-alias
    suave (media de bloques); si no, interpola linealmente."""
    objetivo = 16000
    if sr_in == objetivo:
        return x.astype(np.float32)
    ratio = sr_in / objetivo
    if abs(ratio - round(ratio)) < 1e-6:                 # entero (48000/16000 = 3)
        k = int(round(ratio)); n = (len(x) // k) * k
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        return x[:n].reshape(-1, k).mean(axis=1).astype(np.float32)   # box-LPF + decima
    n_out = int(len(x) / ratio)
    if n_out <= 1:
        return np.zeros(0, dtype=np.float32)
    xp = np.linspace(0, len(x) - 1, n_out)
    return np.interp(xp, np.arange(len(x)), x).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcripción en vivo de un canal del Rode (sin guardar audio).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--canal", type=int, default=9, help="canal 1-based a transcribir (def 9 = Bluetooth)")
    ap.add_argument("--entrada", default="Bluetooth", help="nombre legible de la entrada (para el JSONL)")
    ap.add_argument("--modelo", default="base", help="faster-whisper: tiny/base/small/medium (def base)")
    ap.add_argument("--idioma", default=None, help="código es/en/... (def: autodetecta)")
    ap.add_argument("--umbral", type=float, default=0.01, help="RMS mínimo para considerar que hay voz")
    ap.add_argument("--silencio-corta", type=float, default=1.2, help="seg de silencio que cierran la ventana")
    ap.add_argument("--ventana-max", type=float, default=25.0, help="seg máximos de voz antes de forzar corte")
    ap.add_argument("--salida", default=None, help="ruta JSONL (def: ~/Library/Logs/vstcosmo-audioserver/...)")
    args = ap.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("ERROR: falta faster-whisper. Instala:\n  venv/bin/pip install faster-whisper")

    salida = args.salida or os.path.expanduser(
        "~/Library/Logs/vstcosmo-audioserver/transcript_" + time.strftime("%Y%m%d_%H%M%S") + ".jsonl")
    os.makedirs(os.path.dirname(salida), exist_ok=True)

    print(f"[transcriptor] cargando modelo faster-whisper '{args.modelo}' (CPU int8)…", flush=True)
    model = WhisperModel(args.modelo, device="cpu", compute_type="int8")
    print(f"[transcriptor] modelo listo. canal={args.canal} ({args.entrada})  salida={salida}", flush=True)

    cli = AudioStreamClient(host=args.host, port=args.port)
    hs = cli.handshake()
    sr = int(hs["sample_rate"]); nch = int(hs["channels"]); ch = args.canal - 1
    if ch < 0 or ch >= nch:
        sys.exit(f"ERROR: canal {args.canal} fuera de rango (el stream trae {nch} canales)")
    print(f"[transcriptor] conectado: sr={sr}Hz, {nch} canales. Esperando voz en el canal…", flush=True)

    out = open(salida, "a", buffering=1)
    buf: list[np.ndarray] = []      # audio (sr) acumulado MIENTRAS hay voz (en memoria, se descarta tras transcribir)
    t0: float | None = None         # ts_real del inicio de la ventana de voz
    ult_voz = 0.0                   # última vez que hubo señal
    dur_buf = 0.0                   # seg acumulados en buf

    def transcribir_y_volcar() -> None:
        nonlocal buf, t0, dur_buf
        if not buf or t0 is None:
            buf = []; t0 = None; dur_buf = 0.0; return
        audio48 = np.concatenate(buf).astype(np.float32)
        ini = t0
        buf = []; t0 = None; dur_buf = 0.0
        if len(audio48) < sr * 0.4:                       # <0.4 s: muy corto, ignora
            return
        audio16 = resample_48k_to_16k(audio48, sr)
        segs, _info = model.transcribe(audio16, language=args.idioma, vad_filter=True, beam_size=1)
        texto = " ".join(s.text.strip() for s in segs).strip()
        if not texto:
            return
        rec = {"ts_real_ini": round(ini, 3), "ts_real_fin": round(time.time(), 3),
               "hora": time.strftime("%H:%M:%S"), "canal": args.canal, "entrada": args.entrada,
               "texto": texto}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  [{rec['hora']}] {args.entrada}: {texto}", flush=True)

    try:
        for bloque in cli.frames():
            col = bloque[:, ch].astype(np.float32)
            rms = float(np.sqrt(np.mean(col * col))) if col.size else 0.0
            ahora = time.time()
            if rms >= args.umbral:                        # HAY voz → acumula
                if t0 is None:
                    t0 = ahora
                buf.append(col); dur_buf += col.size / sr; ult_voz = ahora
                if dur_buf >= args.ventana_max:           # ventana larga → corta y transcribe
                    transcribir_y_volcar()
            else:                                          # silencio
                if t0 is not None and (ahora - ult_voz) >= args.silencio_corta:
                    transcribir_y_volcar()                # fin de turno hablado → transcribe lo acumulado
    except KeyboardInterrupt:
        print("\n[transcriptor] deteniendo…", flush=True)
    finally:
        transcribir_y_volcar()
        out.close()
        try:
            cli.cerrar()
        except Exception:
            pass


if __name__ == "__main__":
    main()
