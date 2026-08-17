#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construye el WAV-secuencia del 'Experimento de saturación de estímulos'.
Concatena temas CORTOS de audio_binaural/ (sin BigBang/BlueMonday) con cortes de
silencio entre ellos, a 48 kHz mono. Resultado: audio_binaural/secuencia_saturacion.wav
→ se usa como fuente 'Mundo' (tipo archivo) de la díada; loopea solo (cursor).

NO se normaliza (decisión de Alexis: máxima variación a volumen natural).
"""
import os, sys, numpy as np, soundfile as sf

AUDIO = "/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/audio_binaural"
SR = 48000
CUT = 3.0   # segundos de silencio (corte) entre temas
SALIDA = os.path.join(AUDIO, "secuencia_saturacion.wav")

# Secuencia: cubre las 15 familias, mezclando variantes espaciales L(neg)/R(pos)/C.
# (etiqueta de familia, archivo)
SEC = [
    ("nota_sostenida_L",   "La_neg60deg_largo.wav"),
    ("escala_R",           "escala_do_mayor_piano_like_pos60deg.wav"),
    ("tono_440_R",         "freq_440_pos60deg_largo.wav"),
    ("tono_puro_C",        "Tono puro.wav"),
    ("ruido_blanco_C",     "Ruido blanco.wav"),
    ("viento_R",           "Viento_pos60deg.wav"),
    ("pulso_log_C",        "Pulso logaritmico.wav"),
    ("ritmos_aleat_C",     "Ritmos aleatorios.wav"),
    ("ondas_mixtas_C",     "Ondas mixtas.wav"),
    ("voz_C",              "Voz_Estudio.wav"),
    ("voz_viento_L",       "Voz+Viento_2_neg60deg.wav"),
    ("clasica_L",          "Brandemburgo_neg60deg.wav"),
    ("musica_R",           "musica_pos60deg.wav"),
]

def cargar_mono_48k(path):
    audio, sr = sf.read(path, dtype="float64", always_2d=True)
    mono = audio.mean(axis=1)                      # downmix a mono
    if sr != SR:                                   # resample lineal simple
        n = int(round(len(mono) * SR / sr))
        mono = np.interp(np.linspace(0, len(mono), n, endpoint=False),
                         np.arange(len(mono)), mono)
    return mono

partes, manifest, t = [], [], 0.0
silencio = np.zeros(int(CUT * SR), dtype=np.float64)
for fam, fn in SEC:
    p = os.path.join(AUDIO, fn)
    if not os.path.isfile(p):
        print(f"  ⚠ FALTA (omito): {fn}"); continue
    m = cargar_mono_48k(p)
    dur = len(m) / SR
    manifest.append((round(t, 2), round(t + dur, 2), fam, fn))
    partes.append(m); partes.append(silencio)
    t += dur + CUT

seq = np.concatenate(partes).astype(np.float32)
sf.write(SALIDA, seq, SR, subtype="PCM_16")
print(f"\n✓ secuencia: {SALIDA}")
print(f"  duración total: {len(seq)/SR:.1f}s ({len(seq)/SR/60:.2f} min) · {len(manifest)} temas · 48kHz mono")
print(f"\n  manifiesto (offset_inicio → offset_fin · familia · archivo):")
for ini, fin, fam, fn in manifest:
    print(f"    {ini:7.1f}s → {fin:7.1f}s  {fam:<18} {fn}")
# guarda el manifiesto para el análisis posterior
import json
with open(os.path.join(AUDIO, "secuencia_saturacion_manifest.json"), "w") as f:
    json.dump({"sr": SR, "cut_s": CUT, "temas": manifest}, f, ensure_ascii=False, indent=2)
