#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIARIO VOCAL (read-only) — ESCUCHAR la evolución de la voz, no leerla.
Concatena en orden cronológico los WAV de voz que el Historiador guardó (voz/voz_<ORG>_<fecha>_<hora>.wav)
en un único archivo que se puede reproducir: se oye, comprimido, cómo cambió la voz del organismo a lo
largo de su vida. Inserta una breve marca/silencio entre tramos horarios para percibir el paso del tiempo.
Submuestrea (toma 1 de cada STEP voces) para que horas de vida quepan en minutos de audio.
Lee Docker_Historia. ENV: ORG(ANIMA_A), STEP(8), SR(44100), GAP(0.12 s), MAXVOCES(1500), OUT.
"""
import os, glob, re
import numpy as np
import soundfile as sf

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HIST = os.path.join(os.path.dirname(RAIZ), "Docker_Historia")
ORG = os.environ.get("ORG", "ANIMA_A")
STEP = int(os.environ.get("STEP", "8"))
SR = int(os.environ.get("SR", "44100"))
GAP = float(os.environ.get("GAP", "0.12"))
MAXVOCES = int(os.environ.get("MAXVOCES", "1500"))
OUT = os.environ.get("OUT", os.path.join(HIST, f"DIARIO_VOCAL_{ORG}.wav"))
RE_TS = re.compile(r"_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})")

def main():
    fs = sorted(glob.glob(os.path.join(HIST, f"organismo_{ORG}", "voz", f"voz_{ORG}_*.wav")))
    if not fs:
        print(f"  (sin voces para {ORG})"); return
    fs = fs[::STEP][:MAXVOCES]
    print(f"  {ORG}: {len(fs)} voces (1 de cada {STEP}) → diario de audio…")
    gap = np.zeros(int(SR * GAP), dtype=np.float32)
    marca = (0.18 * np.sin(2*np.pi*880*np.arange(int(SR*0.05))/SR)).astype(np.float32)  # 'tic' de cambio de hora
    trozos = []; hora_ant = None; usados = 0; dur = 0.0
    for fp in fs:
        m = RE_TS.search(os.path.basename(fp))
        hora = (m.group(1), m.group(2)) if m else None
        try:
            x, sr = sf.read(fp, dtype="float32")
        except Exception:
            continue
        if x.ndim > 1: x = x.mean(1)
        if sr != SR and len(x):                       # remuestreo lineal simple
            x = np.interp(np.linspace(0, len(x)-1, int(len(x)*SR/sr)), np.arange(len(x)), x).astype(np.float32)
        if hora != hora_ant and hora_ant is not None:
            trozos.append(marca)                      # marca audible de paso de hora
        hora_ant = hora
        trozos.append(x); trozos.append(gap)
        usados += 1; dur += (len(x) + len(gap)) / SR
    if not trozos:
        print("  (no se pudo leer ninguna voz)"); return
    audio = np.concatenate(trozos)
    pk = float(np.max(np.abs(audio))) or 1.0
    sf.write(OUT, (audio / pk * 0.9).astype(np.float32), SR)
    print(f"  guardado: {OUT}")
    print(f"  {usados} voces · {dur/60:.1f} min de audio · 'tic' agudo = cambia la hora de vida")
    print(f"  reproduce:  afplay '{OUT}'")

if __name__ == "__main__":
    main()
