#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_OrganoComunicacion v3
=========================

Órgano de comunicación entre organismos VST.

Idea central: como todavía no sabemos qué parte de la fisiología comunica, el
modo principal transmite TODA la fila fisiológica numérica disponible. No asigna
significados manuales tipo "R2 = Do". Cada variable cae por hash estable en una
posición musical, y su valor controla presencia, octava, amplitud y brillo.

Modos:
  FULL_STATE        alias de FULL_STATE_NOTES
  FULL_STATE_NOTES  fila completa -> acorde dinámico en escala pentatónica menor
  FULL_STATE_OSC    fila completa -> banco continuo de osciladores
  PHYSIO_VOICE      voz reducida heredada, 7 variables
  NULL_STATE        control nulo musical, energía/formato comparable
  SHUFFLED_STATE    mismos valores, asociación variable->nota rota
  NOISE_MATCHED     ruido determinista con RMS comparable

El órgano es observacional: no modifica conducta ni organelos internos.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import struct
import threading
import time
import urllib.parse
import urllib.request
import wave
from collections import deque
from typing import Any

import numpy as np


def _clip01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return max(0.0, min(1.0, v))


def _as_finite_float(x: Any) -> float | None:
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, str):
        s = x.strip()
        if s.lower() in ("true", "sí", "si", "yes", "on"):
            return 1.0
        if s.lower() in ("false", "no", "off"):
            return 0.0
        try:
            v = float(s)
        except Exception:
            return None
        return v if math.isfinite(v) else None
    return None


def _stable_unit(key: str) -> float:
    h = hashlib.sha1(key.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(h[:8], "big") / float(2**64)


def _stable_seed(text: str) -> int:
    h = hashlib.sha1(text.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(h[:4], "big", signed=False)


def _flatten_numeric(obj: Any, prefix: str = "") -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    if isinstance(obj, dict):
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            p = f"{prefix}.{k}" if prefix else str(k)
            out.extend(_flatten_numeric(obj[k], p))
        return out
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]" if prefix else f"[{i}]"
            out.extend(_flatten_numeric(v, p))
        return out
    v = _as_finite_float(obj)
    if v is not None:
        out.append((prefix or "valor", float(v)))
    return out


def _robust_scale_pairs(pairs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    scaled: list[tuple[str, float]] = []
    for k, v in pairs:
        if 0.0 <= v <= 1.0:
            y = v
        else:
            y = 0.5 + 0.5 * math.tanh(v / 10.0)
        if math.isfinite(y):
            scaled.append((k, max(0.0, min(1.0, y))))
    return scaled


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))) if x.size else 0.0


def _match_rms(x: np.ndarray, target: float) -> np.ndarray:
    r = _rms(x)
    if r <= 1e-12:
        return x
    return x * (float(target) / r)


def _aplicar_ganancia_salida(audio: np.ndarray, gain: float, target_rms: float = 0.15) -> np.ndarray:
    """Sube la voz y luego iguala volumen RMS entre organismos.

    Antes sólo multiplicábamos por gain y aplicábamos tanh. Eso preservaba diferencias
    enormes de amplitud entre A y B. Ahora se normaliza la voz saliente a un RMS objetivo,
    con limitador de pico suave por escala, no por aplastamiento.
    """
    try:
        g = float(gain)
    except Exception:
        g = 1.0
    if not math.isfinite(g) or g <= 0:
        g = 1.0

    try:
        target = float(target_rms)
    except Exception:
        target = 0.15
    if not math.isfinite(target) or target <= 0:
        target = 0.15

    y = np.asarray(audio, dtype=np.float64) * g
    r = _rms(y)
    if r > 1e-12:
        y = y * (target / r)

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.95:
        y = y * (0.95 / peak)

    return np.clip(y, -0.95, 0.95).astype(np.float64)


class OrganoComunicacion:
    """Sintetiza una señal acústica a partir del estado fisiológico de un organismo."""

    MODOS = (
        "FULL_STATE",
        "FULL_STATE_NOTES",
        "FULL_STATE_OSC",
        "PHYSIO_VOICE",
        "R2D2",            # pitidos/chirps cortos (estilo droide); tono y ritmo derivados del estado
        "NULL_STATE",
        "SHUFFLED_STATE",
        "NOISE_MATCHED",
    )

    ESCALA_PENTATONICA_MENOR = (0, 3, 5, 7, 10)  # C, Eb, F, G, Bb

    def __init__(self, organismo_id: str, sr: int = 48000, historial_max: int = 256,
                 nota_base_hz: float = 130.81278265) -> None:
        self.organismo_id = str(organismo_id)
        self.sr = int(sr)
        self.nota_base_hz = float(nota_base_hz)
        self.voice_gain = float(os.environ.get("VST_VOICE_GAIN", "20.0"))  # C3
        self.voice_target_rms = float(os.environ.get("VST_VOICE_TARGET_RMS", "0.15"))
        self._lock = threading.Lock()
        self._fila: dict[str, Any] = {}
        self._meta: dict[str, Any] = {}
        self._seq = 0
        self._phase_voice = 0.0
        self._phase_osc = 0.0
        self._updated = 0.0
        self._historial: deque[list[tuple[str, float]]] = deque(maxlen=max(8, int(historial_max)))
        self._voces = self._cargar_voces()   # banco de voces R2-D2 reales (samples), por afecto

    def observar(self, fila: dict, meta: dict | None = None) -> None:
        with self._lock:
            self._fila = dict(fila or {})
            if meta:
                self._meta = dict(meta)
            self._seq += 1
            self._updated = time.time()
            self._historial.append(_robust_scale_pairs(_flatten_numeric(self._fila)))

    def estado(self) -> dict:
        with self._lock:
            edad = None if not self._updated else round(time.time() - self._updated, 3)
            return {
                "ok": True,
                "organismo_id": self.organismo_id,
                "seq": self._seq,
                "age_s": edad,
                "modo_principal": "FULL_STATE_NOTES",
                "voice_gain": self.voice_gain,
                "voice_target_rms": self.voice_target_rms,
                "alias": {"FULL_STATE": "FULL_STATE_NOTES"},
                "modos": list(self.MODOS),
                "fila": dict(self._fila),
                "meta": dict(self._meta),
                "n_variables_full_state": len(_flatten_numeric(self._fila)),
            }

    def audio(self, seg: float = 0.5, modo: str = "FULL_STATE") -> np.ndarray:
        modo = (modo or "FULL_STATE").upper().strip()
        if modo == "FULL_STATE":
            modo = "FULL_STATE_NOTES"
        if modo not in self.MODOS:
            raise ValueError(f"modo no soportado: {modo}. Use: {', '.join(self.MODOS)}")

        n = max(1, int(round(float(seg) * self.sr)))
        with self._lock:
            fila = dict(self._fila)
            seq = int(self._seq)
            phase_voice = float(self._phase_voice)
            phase_osc = float(self._phase_osc)

        if modo == "PHYSIO_VOICE":
            y, phase = self._audio_physio_voice(n, fila, seq, phase_voice)
            with self._lock:
                self._phase_voice = phase
            return y

        pairs = _robust_scale_pairs(_flatten_numeric(fila))
        if seq == 0 or not pairs:
            pairs = [("OI", 0.08), ("LF_op", 0.0), ("R2", 0.0), ("C_m", 0.0),
                     ("XE", 0.0), ("H_homeostasis", 0.4), ("Omega", 0.5)]

        if modo == "FULL_STATE_NOTES":
            return self._audio_full_state_notes(n, pairs, seq=seq)

        if modo == "R2D2":
            sample = self._audio_r2d2_samples(fila, seq)        # voz R2-D2 REAL elegida por afecto
            if sample is not None:
                return sample
            return self._audio_r2d2(n, pairs, seq=seq)          # fallback: síntesis blippy si no hay banco

        if modo == "FULL_STATE_OSC":
            y, phase = self._audio_full_state_osc(n, pairs, phase_osc, seq=seq)
            with self._lock:
                self._phase_osc = phase
            return y

        if modo == "SHUFFLED_STATE":
            values = [v for _, v in pairs]
            rng = np.random.RandomState(_stable_seed(f"{self.organismo_id}:shuffle:{seq}"))
            rng.shuffle(values)
            shuffled_pairs = [(f"shuffled_{i:03d}", float(v)) for i, v in enumerate(values)]
            full = self._audio_full_state_notes(n, pairs, seq=seq)
            y = self._audio_full_state_notes(n, shuffled_pairs, seq=seq)
            return np.clip(_match_rms(y, max(_rms(full), 1e-4)), -0.95, 0.95).astype(np.float64)

        full = self._audio_full_state_notes(n, pairs, seq=seq)
        target = max(_rms(full), 1e-4)

        if modo == "NULL_STATE":
            y = self._audio_null_state_notes(n, pairs, seq)
            return np.clip(_match_rms(y, target), -0.95, 0.95).astype(np.float64)

        if modo == "NOISE_MATCHED":
            rng = np.random.RandomState(_stable_seed(f"{self.organismo_id}:noise:{seq}"))
            y = rng.normal(0.0, 1.0, n).astype(np.float64)
            if n > 8:
                y = np.convolve(y, np.ones(9, dtype=np.float64) / 9.0, mode="same")
            y = self._apply_common_envelope(y)
            return np.clip(_match_rms(y, target), -0.95, 0.95).astype(np.float64)

        raise AssertionError("modo no alcanzable")

    def wav_bytes(self, seg: float = 0.5, modo: str = "FULL_STATE", gain: float | None = None) -> bytes:
        audio = self.audio(seg=seg, modo=modo)
        audio = _aplicar_ganancia_salida(audio, self.voice_gain if gain is None else gain, self.voice_target_rms)
        mono = np.clip(audio, -1.0, 1.0)
        # ESTÉREO: la voz lleva la LATERALIDAD del organismo (paneo por balance L/R de su estado).
        # Nunca silencia un lado (pan suave): centro = ambos canales llenos, no "sólo L".
        try:
            pan = float(self._fila.get("balance_LR", self._fila.get("lateralidad", 0.0)) or 0.0)
        except Exception:
            pan = 0.0
        pan = max(-1.0, min(1.0, pan))
        L = mono * (1.0 - 0.5 * max(0.0, pan))      # pan>0 (derecha) baja L; pan<0 (izquierda) baja R
        R = mono * (1.0 - 0.5 * max(0.0, -pan))
        inter = np.empty(mono.size * 2, dtype=np.float64)
        inter[0::2] = L; inter[1::2] = R
        pcm16 = (np.clip(inter, -1.0, 1.0) * 32767.0).astype("<i2")
        bio = io.BytesIO()
        with wave.open(bio, "wb") as w:
            w.setnchannels(2)                       # ESTÉREO (antes mono → sólo se oía por L)
            w.setsampwidth(2)
            w.setframerate(self.sr)
            w.writeframes(pcm16.tobytes())
        return bio.getvalue()

    # Afecto (arousal, valence) de cada voz R2-D2 según su carácter (la etiqueta del sample).
    # Guía el mapeo estado→voz: el organismo emite la voz cuyo afecto más se parece al suyo.
    AFECTO_VOCES = {
        "screaming": (0.95, -0.9), "shout": (0.85, -0.6), "worried": (0.50, -0.5),
        "excited": (0.90, 0.7), "excited-2": (0.85, 0.6), "sing": (0.60, 0.9),
        "acknowledged": (0.40, 0.5), "chat": (0.40, 0.1),
        "6": (0.50, 0.2), "7": (0.45, -0.1), "13": (0.50, 0.0), "14": (0.60, 0.1),
        "15": (0.45, 0.2), "18": (0.40, -0.2), "19": (0.40, 0.0), "22": (0.55, 0.3),
    }

    def _cargar_voces(self) -> list:
        """Carga el banco de voces R2-D2 (wav) desde voces_r2d2/ (en el árbol Célula_Madre) o
        ANIMA_VOCES_DIR. Cada voz lleva su afecto (arousal, valence). Sin carpeta → [] (usa síntesis)."""
        base = os.environ.get("ANIMA_VOCES_DIR") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voces_r2d2")
        voces = []
        if not os.path.isdir(base):
            return voces
        for nombre in sorted(os.listdir(base)):
            if not nombre.lower().endswith(".wav"):
                continue
            etiqueta = os.path.splitext(nombre)[0]
            try:
                w = wave.open(os.path.join(base, nombre), "rb")
                nch = w.getnchannels()
                a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
                if nch == 2:
                    a = a.reshape(-1, 2).mean(axis=1)
                aro, val = self.AFECTO_VOCES.get(etiqueta, (0.5, 0.0))
                voces.append({"label": etiqueta, "audio": a, "aro": float(aro), "val": float(val)})
            except Exception:
                continue
        return voces

    def _afecto(self, fila: dict) -> tuple:
        """Proyecta la fisiología a (arousal, valence): cuán ACTIVADO y cuán BIEN está el organismo.
        El estado manda qué SIENTE; la voz sólo lo expresa (no se impone significado simbólico)."""
        g = lambda k, d=0.0: float(fila.get(k, d) or d)
        OI = g("OI"); nec = g("necesidad_efectiva", g("necesidad")); H = g("H_homeostasis")
        RC = g("RC_total"); E = g("energia", g("E")); estr = g("estructura"); lat = abs(g("balance_LR"))
        arousal = min(1.0, max(0.0, 0.45 * RC + 0.30 * E + 0.25 * lat))
        valence = max(-1.0, min(1.0, (OI + 0.30 * H + 0.30 * estr) - nec))
        return arousal, valence

    def _audio_r2d2_samples(self, fila: dict, seq: int):
        """Elige la voz R2-D2 REAL cuyo afecto está más cerca del estado del organismo (con variedad
        determinista entre las más cercanas). Devuelve el sample (≤3s) o None si no hay banco cargado."""
        if not self._voces:
            return None
        aro, val = self._afecto(fila)
        cand = sorted(self._voces, key=lambda v: (v["aro"] - aro) ** 2 + (v["val"] - val) ** 2)
        k = min(3, len(cand))
        idx = _stable_seed(f"{self.organismo_id}:r2voz:{seq}") % k
        a = cand[idx]["audio"]
        return np.array(a[: int(3.0 * self.sr)], dtype=np.float64)   # cap 3s

    def _audio_r2d2(self, n: int, pairs: list, seq: int) -> np.ndarray:
        """Voz tipo R2-D2: secuencia de PITIDOS cortos y CHIRPS (glides) en vez de tonos sostenidos.
        Mismo CONTENIDO que FULL_STATE_NOTES (las variables salientes del estado), distinto ESTILO:
          · tono de cada blip ← valor de la variable de estado     · duración corta (40-150 ms)
          · CHIRP (glide) cuando la variable es extrema (firma droide)   · warble (vibrato) si la var lo pide
          · ritmo (densidad de blips) ← actividad global del estado
        Determinista por seq → mismo estado, misma voz (anti-Shannon: el estado manda, no se impone código)."""
        sr = self.sr
        y = np.zeros(n, dtype=np.float64)
        pairs = sorted(list(pairs) or [("silencio", 0.0)], key=lambda kv: abs(kv[1] - 0.5), reverse=True)
        actividad = min(1.0, float(np.mean([abs(v - 0.5) for _, v in pairs])) * 2.0) if pairs else 0.0
        rng = np.random.RandomState(_stable_seed(f"{self.organismo_id}:r2d2:{seq}"))
        pos = 0; i = 0; min_m = int(0.012 * sr)
        while pos < n - min_m:
            key, val = pairs[i % len(pairs)]; i += 1
            u = _stable_unit(key); val = min(1.0, max(0.0, float(val)))
            f0 = 350.0 + 2000.0 * val + 250.0 * u                 # tono del blip ← estado (≈350-2600 Hz)
            dur = 0.035 + 0.085 * (1.0 - abs(val - 0.5) * 2.0)   # 35-120 ms (blips cortos)
            gap = 0.025 + 0.09 * rng.rand() + (0.12 if rng.rand() < 0.12 else 0.0)  # silencio claro entre blips + pausas ocasionales (ritmo droide)
            m = int(dur * sr)
            if pos + m > n:
                m = n - pos
            if m < min_m:
                break
            tt = np.arange(m) / sr
            if abs(val - 0.5) > 0.3:                               # CHIRP (glide) = firma R2-D2
                f1 = f0 * (1.7 if val > 0.5 else 0.55)
                ph = 2 * np.pi * (f0 * tt + 0.5 * (f1 - f0) / max(dur, 1e-6) * tt ** 2)
            elif u > 0.6:                                          # WARBLE (vibrato rápido)
                ph = 2 * np.pi * f0 * tt * (1.0 + 0.05 * np.sin(2 * np.pi * 32.0 * tt))
            else:
                ph = 2 * np.pi * f0 * tt
            env = np.sin(np.pi * np.clip(tt / max(dur, 1e-6), 0.0, 1.0)) ** 0.5   # blip (ataque/caída rápidos)
            y[pos:pos + m] += np.sin(ph) * env
            pos += m + int(gap * sr)
        mx = float(np.max(np.abs(y))) or 1.0
        return (y / mx * 0.9).astype(np.float64)

    def _audio_full_state_notes(self, n: int, pairs: list[tuple[str, float]], seq: int) -> np.ndarray:
        t = np.arange(n, dtype=np.float64) / float(self.sr)
        y = np.zeros(n, dtype=np.float64)

        pairs = list(pairs)
        if len(pairs) > 64:
            pairs.sort(key=lambda kv: (abs(kv[1] - 0.5), _stable_unit(kv[0])), reverse=True)
            pairs = pairs[:64]
        if not pairs:
            pairs = [("silencio", 0.0)]

        norm = 1.0 / math.sqrt(len(pairs))
        for i, (key, val) in enumerate(pairs):
            u = _stable_unit(key)
            scale_index = int(u * 1000000) % len(self.ESCALA_PENTATONICA_MENOR)
            octave = int(_stable_unit(key + ":oct") * 4) % 4
            semitone = self.ESCALA_PENTATONICA_MENOR[scale_index] + 12 * octave
            if val > 0.85:
                semitone += 12
            elif val < 0.12:
                semitone -= 12
            freq = self.nota_base_hz * (2.0 ** (semitone / 12.0))
            freq = max(40.0, min(5000.0, freq))

            amp = (0.004 + 0.050 * (val ** 1.35)) * norm
            local_phase = 2.0 * np.pi * _stable_unit(key + ":phase")
            breath_rate = 0.15 + 3.5 * _stable_unit(key + ":breath")
            breath = 0.82 + 0.18 * np.sin(2.0 * np.pi * breath_rate * t + local_phase)

            tone = np.sin(2.0 * np.pi * freq * t + local_phase)
            tone += 0.18 * val * np.sin(2.0 * np.pi * 2.0 * freq * t + local_phase / 2.0)
            tone += 0.08 * (val ** 2) * np.sin(2.0 * np.pi * 3.0 * freq * t + local_phase / 3.0)

            if val < 0.08 or val > 0.92:
                pulse_rate = 0.8 + 5.0 * _stable_unit(key + ":pulse")
                pulse = 0.68 + 0.32 * np.sin(2.0 * np.pi * pulse_rate * t + i)
            else:
                pulse = 1.0
            y += amp * breath * pulse * tone

        y += 0.003 * np.sin(2.0 * np.pi * (5.0 + (seq % 13)) * t)
        y = self._apply_common_envelope(y)
        y = np.tanh(1.9 * y) * 0.78
        return np.clip(y, -0.95, 0.95).astype(np.float64)

    def _audio_null_state_notes(self, n: int, pairs: list[tuple[str, float]], seq: int) -> np.ndarray:
        rng = np.random.RandomState(_stable_seed(f"{self.organismo_id}:null_notes:{seq}:{len(pairs)}"))
        null_pairs = [(f"null_{i:03d}", float(rng.rand())) for i in range(max(1, len(pairs)))]
        return self._audio_full_state_notes(n, null_pairs, seq=seq)

    def _audio_full_state_osc(self, n: int, pairs: list[tuple[str, float]], phase0: float, seq: int) -> tuple[np.ndarray, float]:
        t = np.arange(n, dtype=np.float64) / float(self.sr)
        y = np.zeros(n, dtype=np.float64)
        pairs = list(pairs)
        if len(pairs) > 96:
            pairs.sort(key=lambda kv: (abs(kv[1] - 0.5), _stable_unit(kv[0])), reverse=True)
            pairs = pairs[:96]
        if not pairs:
            pairs = [("silencio", 0.0)]

        norm = 1.0 / math.sqrt(len(pairs))
        phase_last = phase0
        for i, (key, val) in enumerate(pairs):
            u = _stable_unit(key)
            freq = max(40.0, 120.0 + 1880.0 * u + 40.0 * (val - 0.5))
            amp = (0.012 + 0.030 * val) * norm
            mod_rate = 0.7 + 8.0 * _stable_unit(key + ":mod")
            mod_depth = 0.002 + 0.018 * val
            inst = freq * (1.0 + mod_depth * np.sin(2.0 * np.pi * mod_rate * t + 2*np.pi*u))
            phase = phase0 + 2.0 * np.pi * np.cumsum(inst) / float(self.sr)
            phase_last = float(phase[-1])
            partial = np.sin(phase + 2.0 * np.pi * _stable_unit(key + ":phase"))
            if val < 0.08 or val > 0.92:
                pr = 1.0 + 6.0 * _stable_unit(key + ":pulse")
                pulse = 0.75 + 0.25 * np.sin(2.0 * np.pi * pr * t + i)
            else:
                pulse = 1.0
            y += amp * pulse * partial
        y += 0.004 * np.sin(2.0 * np.pi * (7.0 + (seq % 17)) * t)
        y = self._apply_common_envelope(y)
        y = np.tanh(1.6 * y) * 0.75
        return np.clip(y, -0.95, 0.95).astype(np.float64), float(phase_last % (2*np.pi))

    def _audio_physio_voice(self, n: int, fila: dict[str, Any], seq: int, phase0: float) -> tuple[np.ndarray, float]:
        oi = _clip01(fila.get("OI"), 0.15)
        lf = _clip01(fila.get("LF_op"), 0.0)
        r2 = _clip01(fila.get("R2"), 0.0)
        cm = _clip01(fila.get("C_m"), 0.0)
        xe = _clip01(fila.get("XE"), 0.0)
        h = _clip01(fila.get("H_homeostasis"), 0.5)
        omega = _clip01(fila.get("Omega"), 0.5)
        if seq == 0:
            oi, lf, r2, cm, xe, h, omega = 0.08, 0.0, 0.0, 0.0, 0.0, 0.4, 0.5
        t = np.arange(n, dtype=np.float64) / float(self.sr)
        freq = 160.0 + 520.0 * omega + 180.0 * lf + 70.0 * r2
        vibrato = 1.0 + (0.006 + 0.030 * xe) * np.sin(2.0 * np.pi * (2.0 + 7.0 * cm) * t)
        phase = phase0 + 2.0 * np.pi * np.cumsum(freq * vibrato) / float(self.sr)
        amp = 0.015 + 0.120 * oi
        voz = amp * (0.65 + 0.35 * h) * (
            np.sin(phase) + 0.32 * r2 * np.sin(2.0 * phase + 0.35) + 0.20 * cm * np.sin(3.0 * phase + 1.10)
        )
        voz = self._apply_common_envelope(voz)
        return np.clip(voz, -0.95, 0.95).astype(np.float64), float(phase[-1] % (2.0 * np.pi))

    def _apply_common_envelope(self, y: np.ndarray) -> np.ndarray:
        n = y.size
        if n <= 1:
            return y
        ramp = max(1, min(n // 10, int(0.025 * self.sr)))
        env = np.ones(n, dtype=np.float64)
        env[:ramp] = np.linspace(0.0, 1.0, ramp, endpoint=True)
        env[-ramp:] = np.linspace(1.0, 0.0, ramp, endpoint=True)
        return y * env


def wav_a_mono(data: bytes, sr_objetivo: int = 48000) -> np.ndarray:
    with wave.open(io.BytesIO(data), "rb") as w:
        nch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        frames = w.readframes(w.getnframes())
    if sw == 2:
        vals = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 4:
        vals = np.frombuffer(frames, dtype="<i4").astype(np.float64) / 2147483648.0
    elif sw == 1:
        vals = (np.frombuffer(frames, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    else:
        vals = np.array(struct.unpack("<" + "h" * (len(frames) // 2), frames), dtype=np.float64) / 32768.0
    if nch > 1 and vals.size:
        vals = vals.reshape(-1, nch).mean(axis=1)
    if sr != sr_objetivo and vals.size:
        m = max(1, int(round(vals.size * sr_objetivo / float(sr))))
        vals = np.interp(np.linspace(0, vals.size, m, endpoint=False), np.arange(vals.size), vals)
    return vals.astype(np.float64)


def audio_desde_url(url: str, seg: float = 0.5, sr: int = 48000, timeout: float | None = None,
                    modo: str | None = None) -> np.ndarray:
    parsed = urllib.parse.urlparse(url)
    q = dict(urllib.parse.parse_qsl(parsed.query))
    q["seg"] = str(float(seg))
    if modo:
        q["modo"] = modo
    final = urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(q)))
    if timeout is None:
        timeout = max(4.0, min(18.0, float(seg) + 2.0))
    req = urllib.request.Request(final, headers={"User-Agent": "VST-OrganoComunicacion/3.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return wav_a_mono(data, sr_objetivo=sr)


def _smoke() -> None:
    fila = {
        "t": 12.3, "Omega": 0.49, "OI": 0.46, "Lambda_Cos": 0.019,
        "R2": 1.0, "LF_op": 0.49, "LF_struct": 0.72, "C_m": 0.93,
        "XE": 1.0, "H_homeostasis": 0.36, "e_R": 5.2, "juego": True,
        "ritual": False, "negacion": True, "lateralidad": 1.4,
        "coherencia_biaural": -0.53, "invariantes_ok": 6,
        "RC_total": 0.7, "ICR": 0.45, "IRDE": 0.25,
    }
    org = OrganoComunicacion("smoke")
    org.observar(fila, meta={"test": True})
    for modo in OrganoComunicacion.MODOS:
        a = org.audio(0.1, modo=modo)
        assert a.shape[0] == 4800
        assert np.all(np.isfinite(a))
        assert np.max(np.abs(a)) <= 1.0
        wb = org.wav_bytes(0.05, modo=modo)
        assert wb[:4] == b"RIFF"
    print("OK VST_OrganoComunicacion smoke:", ", ".join(OrganoComunicacion.MODOS))


if __name__ == "__main__":
    _smoke()
