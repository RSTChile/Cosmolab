#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_RadioModulador — convierte AUDIO (voz del organismo + indicativo) en IQ NFM para transmitir
================================================================================================
QUIÉN SOY (para retomar sin reprocesar):
  Soy el puente entre la VOZ (audio) y la RADIO (IQ). Tomo el audio que el organismo quiere emitir
  —su voz R2D2 (WAV) precedida del indicativo legal en Morse— y lo modulo en FM de banda angosta
  (NFM) generando la señal IQ en banda base, que escribo como un .wav estéreo (I=canal0, Q=canal1)
  que el FileSource de SDRangel reproduce tal cual por el HackRF. El oído de radio (SDRconnect NFM)
  la demodula y se OYE la voz. Sin dependencias raras: numpy + wave (stdlib).

POR QUÉ ASÍ: los moduladores de SDRangel no aceptan un archivo por REST; FileSource sí reproduce
  IQ .wav. Modulando NOSOTROS controlamos todo (temporización del Morse, desviación, nivel), sin el
  jitter del keying por REST. La misma IQ sirve para Mac (A) y Pi (E).

IDENTIFICACIÓN LEGAL: toda emisión ABRE con el indicativo en Morse (CD3LZK por defecto, la licencia
  de Alexis). Es requisito de radioaficionado y queda 'firmada' la estación en cada transmisión.

Parámetros clave:
  iq_rate      250000  (tasa de la IQ; FileSource interpola ×2^log2Interp hasta el device)
  desviacion   3000 Hz (NFM de banda angosta; cabe en 70cm/canal angosto)
  tono_morse   700 Hz  · wpm 12
"""
from __future__ import annotations
import os
import wave
import numpy as np

MORSE = {'A':'.-','B':'-...','C':'-.-.','D':'-..','E':'.','F':'..-.','G':'--.','H':'....',
         'I':'..','J':'.---','K':'-.-','L':'.-..','M':'--','N':'-.','O':'---','P':'.--.',
         'Q':'--.-','R':'.-.','S':'...','T':'-','U':'..-','V':'...-','W':'.--','X':'-..-',
         'Y':'-.--','Z':'--..','0':'-----','1':'.----','2':'..---','3':'...--','4':'....-',
         '5':'.....','6':'-....','7':'--...','8':'---..','9':'----.','/':'-..-.'}


def morse_audio(texto: str, sr: int, wpm: int = 12, tono: float = 700.0) -> np.ndarray:
    """Audio del texto en Morse (tono keyado). Temporización estándar: dit=1200/wpm ms."""
    dit = 1.2 / wpm
    def tono_seg(dur):
        t = np.arange(int(sr * dur)) / sr
        return np.sin(2 * np.pi * tono * t)
    sil = lambda dur: np.zeros(int(sr * dur))
    partes = []
    for ch in texto.upper():
        if ch == ' ':
            partes.append(sil(dit * 7)); continue
        for el in MORSE.get(ch, ''):
            partes.append(tono_seg(dit * (3 if el == '-' else 1)))
            partes.append(sil(dit))            # gap intra-carácter
        partes.append(sil(dit * 2))            # gap entre caracteres (total 3 dit)
    return np.concatenate(partes) if partes else sil(0.1)


def _leer_wav_mono(path: str, sr_obj: int) -> np.ndarray:
    """Lee un WAV (cualquier canal/anchura común) → mono float [-1,1] remuestreado a sr_obj."""
    with wave.open(path, "rb") as w:
        nch, sw, fr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    dt = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    a = np.frombuffer(raw, dtype=dt).astype(np.float64)
    if sw == 1:  # 8-bit sin signo
        a = (a - 128) / 128.0
    else:
        a = a / float(np.iinfo(dt).max)
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    if fr != sr_obj and len(a) > 1:                      # remuestreo lineal (sin scipy)
        n_obj = int(round(len(a) * sr_obj / fr))
        a = np.interp(np.linspace(0, len(a) - 1, n_obj), np.arange(len(a)), a)
    m = np.max(np.abs(a)) or 1.0
    return (a / m) * 0.9


def nfm_iq(audio: np.ndarray, iq_rate: int, desviacion: float = 3000.0, amp: float = 0.9) -> np.ndarray:
    """FM de banda angosta: fase = 2π·desviación·∫audio dt; iq = amp·e^{jfase} (complejo banda base)."""
    fase = 2 * np.pi * desviacion * np.cumsum(audio) / iq_rate
    return amp * np.exp(1j * fase)


def escribir_iq_wav(iq: np.ndarray, iq_rate: int, path: str) -> None:
    """Escribe IQ como .wav estéreo int16 (I=canal0, Q=canal1) — lo que FileSource reproduce."""
    inter = np.empty(len(iq) * 2, dtype=np.int16)
    inter[0::2] = np.clip(iq.real, -1, 1) * 32767
    inter[1::2] = np.clip(iq.imag, -1, 1) * 32767
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(iq_rate)
        w.writeframes(inter.tobytes())


def voz_a_iq_wav(voz_wav: str, salida_iq: str, indicativo: str = None,
                 iq_rate: int = 250000, desviacion: float = 3000.0, wpm: int = 12,
                 offset_hz: float = 30000.0) -> dict:
    """PIPELINE COMPLETO: [indicativo Morse] + [voz del organismo] → IQ NFM → .wav para FileSource.
    offset_hz desplaza la señal del centro (evita el pico DC del receptor; el RX pone su VFO ahí).
    Devuelve metadatos (duración, dónde empieza la voz, offset) para validar la recepción."""
    indicativo = indicativo if indicativo is not None else os.environ.get("ANIMA_TX_INDICATIVO", "CD3LZK")
    tramos = []
    dur_id = 0.0
    if indicativo:
        idm = morse_audio(indicativo, iq_rate, wpm=wpm)
        tramos.append(idm); tramos.append(np.zeros(int(iq_rate * 0.4)))   # id + respiro
        dur_id = (len(idm) + int(iq_rate * 0.4)) / iq_rate
    voz = _leer_wav_mono(voz_wav, iq_rate)
    tramos.append(voz)
    audio = np.concatenate(tramos)
    iq = nfm_iq(audio, iq_rate, desviacion)
    if offset_hz:                                          # correr la señal fuera del DC
        t = np.arange(len(iq)) / iq_rate
        iq = iq * np.exp(2j * np.pi * offset_hz * t)
    escribir_iq_wav(iq, iq_rate, salida_iq)
    return {"salida": salida_iq, "iq_rate": iq_rate, "duracion_s": len(audio) / iq_rate,
            "voz_empieza_s": dur_id, "indicativo": indicativo, "desviacion": desviacion,
            "offset_hz": offset_hz}


if __name__ == "__main__":
    import sys, glob
    voces = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "..", "voces_r2d2", "*.wav")))
    voz = sys.argv[1] if len(sys.argv) > 1 else (voces[0] if voces else None)
    if not voz:
        raise SystemExit("no encuentro voces (voces_r2d2/*.wav)")
    meta = voz_a_iq_wav(voz, "/tmp/anima_voz_iq.wav")
    print("voz:", os.path.basename(voz))
    print("IQ generada:", meta)
