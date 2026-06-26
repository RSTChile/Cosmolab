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
        # Ganancia de SALIDA del usuario (slider "Volumen de voz"): se aplica DESPUÉS de la
        # normalización/gobernanza, así sube la voz de verdad (la gobernanza capa el target_rms).
        self.voice_volumen = float(os.environ.get("VST_VOICE_VOLUMEN", "4.0"))
        self._lock = threading.Lock()
        self._fila: dict[str, Any] = {}
        self._meta: dict[str, Any] = {}
        self._seq = 0
        self._phase_voice = 0.0
        self._phase_osc = 0.0
        self._updated = 0.0
        self._historial: deque[list[tuple[str, float]]] = deque(maxlen=max(8, int(historial_max)))
        self._voces = self._cargar_voces()   # banco de voces R2-D2 reales (samples), por afecto
        # LIBERTAD EXPRESIVA: gesto vocal explorado por el OrganeloAlteridad (parámetros ACÚSTICOS).
        # Si es None → voz fisiológica pura (sin balbuceo). Lo fija WebLive cada paso desde ALTERIDAD.
        self.gesto: dict | None = None
        # SEGUNDA VÍA DE EMISIÓN — APARATO FONADOR (crear palabra propia, no reemplazar el banco).
        # Aparato ARP 2600 paramétrico: dado un afecto, SINTETIZA una vocalización R2-D2 nueva. Opcional:
        # si falta scipy, queda None y el organismo sigue con el banco (degradación elegante).
        self._fonador = None
        try:
            from VST_OrganoFonador import OrganoFonador
            self._fonador = OrganoFonador(self.sr)
        except Exception:
            self._fonador = None
        self._gap_reciente: deque = deque(maxlen=24)   # huecos recientes (target NO cubierto) → exige recurrencia
        self.ultima_voz_origen = "banco"               # banco | creado (de la última emisión real)
        self.ultimo_costo_voz = 0.0                    # coste de la última emisión (para mostrar)
        self._costo_pendiente = 0.0                    # coste acumulado desde la última lectura del metabolismo
        # PERSISTENCIA del vocabulario PROPIO: las palabras que el organismo acuña se guardan a disco y se
        # ACUMULAN entre vidas (volumen /data, sobrevive reinicios). El vocabulario base (voces_r2d2) NO se
        # toca. Al nacer, el organismo recupera las palabras que ya inventó.
        self._creadas_dir = os.environ.get("ANIMA_VOCES_CREADAS_DIR") or os.path.join(
            os.environ.get("ANIMA_ESTADO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "voces_creadas")
        self._creadas = 0          # mayor índice de palabra propia ACUÑADA emitido (monotónico, para etiquetar)
        self._aprendidas = 0       # mayor índice de palabra APRENDIDA (emulada del otro) emitido
        self._emision_seq = 0      # reloj de emisiones (para vida media / abandono de provisionales)
        self._cargar_creadas()     # recupera el vocabulario PROPIO consolidado de vidas anteriores

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
                "modo_principal": "R2D2",
                "voice_gain": self.voice_gain,
                "voice_target_rms": self.voice_target_rms,
                "voice_volumen": self.voice_volumen,
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

    def _aplicar_gesto(self, mono: np.ndarray) -> np.ndarray:
        """LIBERTAD EXPRESIVA (balbuceo): aplica el gesto vocal explorado — variaciones ACÚSTICAS pequeñas
        y reversibles, NO un significado: frecuencia/velocidad (resample), intensidad (ganancia), pausa
        (silencio inicial) y repetición (pequeñas secuencias). El espacio explorado es físico, no léxico."""
        g = self.gesto
        if not g:
            return mono
        try:
            mono = np.asarray(mono, dtype=np.float64)
            # frecuencia/velocidad: resample suave (±~28%), tipo "tono más agudo/grave y rápido/lento"
            fr = math.exp(0.25 * float(g.get("g_freq", 0.0)))
            if abs(fr - 1.0) > 0.01 and mono.size > 4:
                m = max(2, int(round(mono.size / fr)))
                idx = np.clip((np.arange(m) * fr).astype(int), 0, mono.size - 1)
                mono = mono[idx]
            # intensidad: ganancia pequeña (el clip final acota)
            mono = mono * (1.0 + 0.35 * float(g.get("g_intensidad", 0.0)))
            # pausa: silencio inicial (hasta ~0.35 s) → cambia el ritmo de la vocalización
            pa = int(self.sr * 0.35 * max(0.0, min(1.0, float(g.get("g_pausa", 0.0)))))
            if pa:
                mono = np.concatenate([np.zeros(pa, dtype=np.float64), mono])
            # repetición: 0..2 repeticiones con pequeños silencios (pequeñas SECUENCIAS, no palabras)
            rep = int(round(2.0 * max(0.0, min(1.0, float(g.get("g_repeticion", 0.0))))))
            if rep > 0:
                gap = np.zeros(int(self.sr * 0.08), dtype=np.float64)
                mono = np.concatenate([mono] + [np.concatenate([gap, mono]) for _ in range(rep)])
        except Exception:
            return np.asarray(mono, dtype=np.float64)
        return mono

    def wav_bytes(self, seg: float = 0.5, modo: str = "FULL_STATE", gain: float | None = None) -> bytes:
        audio = self.audio(seg=seg, modo=modo)
        audio = _aplicar_ganancia_salida(audio, self.voice_gain if gain is None else gain, self.voice_target_rms)
        # GANANCIA DE SALIDA del usuario (slider): sube la voz DE VERDAD, por encima del tope que la
        # gobernanza pone al target_rms. tanh = limitador suave (sin clip duro) para que llegue al rojo.
        mono = np.tanh(np.asarray(audio, dtype=np.float64) * max(0.0, float(self.voice_volumen))) * 0.97
        mono = self._aplicar_gesto(mono)   # LIBERTAD EXPRESIVA: imprime el gesto explorado sobre la voz
        # ESTÉREO: la voz lleva la LATERALIDAD del organismo (paneo SUAVE por balance L/R de su estado).
        # Siempre claramente en AMBOS canales (atenuación máx 25%): centro = ambos llenos, no "sólo L".
        try:
            pan = float(self._fila.get("balance_LR", self._fila.get("lateralidad", 0.0)) or 0.0)
        except Exception:
            pan = 0.0
        pan = max(-1.0, min(1.0, pan))
        L = mono * (1.0 - 0.25 * max(0.0, pan))     # pan>0 (derecha) baja L un poco; pan<0 baja R un poco
        R = mono * (1.0 - 0.25 * max(0.0, -pan))
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

    def nivel_voz_propia(self, seg: float = 0.3) -> float:
        """RMS de la voz que el organismo emite AHORA (para el medidor de 'voz propia'). Aplica el
        mismo voice_volumen del slider → al subir el volumen, el usuario VE subir este nivel."""
        try:
            a = self.audio(seg=seg, modo="R2D2")
            a = _aplicar_ganancia_salida(a, self.voice_gain, self.voice_target_rms)
            a = np.tanh(np.asarray(a, dtype=np.float64) * max(0.0, float(self.voice_volumen))) * 0.97
            return float(_rms(a))
        except Exception:
            return 0.0

    # SEGUNDA VÍA — parámetros de la decisión "usar banco vs. ACUÑAR palabra propia".
    GAP_CREAR = 0.22        # distancia afectiva mínima a la voz más cercana para considerar que el banco NO cubre
    COSTO_USAR = 0.0        # tirar del banco no añade coste extra: la voz YA es señal costosa (lo cobra el metabolismo)
    COSTO_CREAR = 0.04      # ACUÑAR añade un gasto energético REAL (más caro que reutilizar, como pidió el diseño)
    ENERGIA_MIN_CREAR = 0.25  # sin esta energía no se puede crear: el coste es REAL (si no, expresa con el banco)
    P_CREAR = 0.6           # libertad funcional: aun con hueco recurrente y energía, a veces NO crea
    RECURRENCIA_CREAR = 3   # el hueco debe REAPARECER ≥3 veces (no acuñar por un estado fugaz)
    MAX_CREADAS = 64        # tope del vocabulario propio (el umbral de gap ya limita; esto es un cinturón)
    # SELECCIÓN, no acumulación: acuñar ≠ incorporar. Una palabra nace PROVISIONAL (hipótesis del organismo
    # sobre cómo expresarse); sólo si la REUTILIZA se CONSOLIDA (pasa a patrimonio y se persiste); si no la
    # reusa, se ABANDONA (la historia, no la creación, decide qué queda). El vocabulario evoluciona por uso.
    USOS_CONSOLIDA = 4      # reusos para que una palabra provisional pase a ESTABLE (y se guarde a disco)
    VIDA_PROVISIONAL = 600  # emisiones sin reuso tras las que una palabra provisional se ABANDONA (no cuajó)
    # IMITACIÓN entre organismos: si el otro vocaliza algo que MI banco no cubre, puedo EMULARLO con mi propio
    # aparato (no copiar: re-sintetizo mi versión) → el vocabulario inventado puede CONVERGER (lenguaje
    # compartido) en vez de divergir en dos linajes. Emular es también una conducta libre y costosa.
    P_EMULAR = 0.5          # libertad funcional de emular la palabra del otro
    GAP_EMULAR = 0.18       # si la voz del otro cae en una región que MI banco ya cubre, no hace falta emular

    # Afecto (arousal, valence) de cada voz R2-D2 según su carácter (la etiqueta del sample).
    # Guía el mapeo estado→voz: el organismo emite la voz cuyo afecto más se parece al suyo.
    AFECTO_VOCES = {
        "screaming": (0.95, -0.9), "shout": (0.85, -0.6), "worried": (0.50, -0.5),
        "excited": (0.90, 0.7), "excited-2": (0.85, 0.6), "sing": (0.60, 0.9),
        "acknowledged": (0.40, 0.5), "chat": (0.40, 0.1),
        "6": (0.50, 0.2), "7": (0.45, -0.1), "13": (0.50, 0.0), "14": (0.60, 0.1),
        "15": (0.45, 0.2), "18": (0.40, -0.2), "19": (0.40, 0.0), "22": (0.55, 0.3),
    }

    # Nombre legible EN CASTELLANO de cada voz: una ETIQUETA para humanos (identificar y rastrear patrones
    # si surgen), NO un significado. El organismo elige la voz por AFECTO, jamás por el nombre. Los samples
    # R2-D2 originales venían en inglés; aquí su título provisional en español.
    NOMBRE_ES = {
        "screaming": "alarido", "shout": "grito", "worried": "inquietud",
        "excited": "euforia", "excited-2": "euforia 2", "sing": "canturreo",
        "acknowledged": "asentir", "chat": "parloteo",
    }
    # Acentos/títulos sugeridos para sonidos cuyo nombre de archivo va sin tilde (provisional, no semántico).
    TITULOS_ES = {
        "afirmacion": "afirmación", "alegria": "alegría", "atencion": "atención",
        "bateria_baja": "batería baja", "compania": "compañía", "comprension": "comprensión",
        "confusion": "confusión", "energia_plena": "energía plena", "exploracion": "exploración",
        "frustracion": "frustración", "negacion": "negación", "precaucion": "precaución",
        "satisfaccion": "satisfacción",
    }
    # TÍTULOS FIELES AL AFECTO MEDIDO (override prioritario). Varios nombres de archivo sugerían un
    # significado MUY específico (dolor, fatiga, hambre, negación…) que NO coincide con su afecto real:
    # estas voces caen en el rincón de BAJA ACTIVACIÓN / valencia casi neutra. Como el rótulo es sólo una
    # etiqueta para el humano (el organismo elige por afecto, no por el nombre), aquí se renombran a una
    # familia de REPOSO/CALMA fiel a lo medido: las levemente "pesadas" (val<0) → sopor/letargo/modorra…,
    # las levemente plácidas (val>0) → serenidad/placidez/sosiego…, las neutras → reposo/descanso/quietud.
    TITULOS_FIELES = {
        "dolor": "reposo", "confusion": "descanso", "fatiga": "sopor", "insistencia": "letargo",
        "hambre": "quietud", "respuesta": "languidez", "compania": "serenidad", "comprension": "placidez",
        "precaucion": "sosiego", "atencion": "calma", "negacion": "modorra", "llamada": "remanso",
    }

    def _titulo(self, label: str) -> str:
        """Título legible en castellano para una voz (provisional, sólo para que el humano la identifique)."""
        if label in self.TITULOS_FIELES:        # override fiel al afecto (corrige nombres engañosos)
            return self.TITULOS_FIELES[label]
        if label in self.NOMBRE_ES:
            return self.NOMBRE_ES[label]
        if label in self.TITULOS_ES:
            return self.TITULOS_ES[label]
        if str(label).isdigit():
            return f"tono {label}"
        return str(label).replace("_", " ").replace("-", " ").strip().capitalize() or label

    @staticmethod
    def _afecto_acustico(a: np.ndarray, sr: int) -> tuple:
        """Deriva (arousal, valence) de los RASGOS ACÚSTICOS del sonido mismo — no de una etiqueta.
        Así, un sonido NUEVO que se sube al repertorio adquiere afecto por lo que ES (cómo suena),
        no por un diccionario que lo nombre; el repertorio puede crecer sin límite y el organismo lo
        explora igual que a los demás. Anti-Shannon: el significado afectivo no se impone, se mide.
          · arousal ← energía (RMS) + brillo (centroide espectral) + 'agitación' (tasa de cruces por cero).
          · valence ← tonalidad (lo tonal/armónico tiende a +, lo ruidoso a −) + contorno de brillo
                       (sonido que se ABRE/sube → +, que se apaga/baja → −).
        Heurística honesta, no semántica: es un punto de partida que la HISTORIA puede recontextualizar."""
        if a is None or len(a) < 64:
            return 0.5, 0.0
        x = a.astype(np.float64)
        x = x - x.mean()
        pk = np.max(np.abs(x)) or 1.0
        x = x / pk
        rms = float(np.sqrt(np.mean(x * x)))                       # energía
        zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))      # cruces por cero (agitación/ruido)
        # espectro (una sola FFT global, suficiente para una etiqueta de afecto)
        win = np.hanning(len(x))
        S = np.abs(np.fft.rfft(x * win)) + 1e-9
        f = np.fft.rfftfreq(len(x), 1.0 / max(1, sr))
        centroide = float((f * S).sum() / S.sum())                 # brillo
        cen_n = min(1.0, centroide / (sr * 0.25))                  # normalizado a ~Nyquist/2
        # planitud espectral (Wiener): ~1 ruido, ~0 tonal
        gm = float(np.exp(np.mean(np.log(S))))
        am = float(np.mean(S))
        planitud = gm / am
        # contorno de brillo: ¿el sonido se abre (centroide sube) o se apaga (baja)?
        mid = len(x) // 2
        c1 = np.abs(np.fft.rfft(x[:mid] * np.hanning(mid))) + 1e-9 if mid > 32 else S
        c2 = np.abs(np.fft.rfft(x[mid:] * np.hanning(len(x) - mid))) + 1e-9 if mid > 32 else S
        f1 = np.fft.rfftfreq(mid, 1.0 / max(1, sr)); f2 = np.fft.rfftfreq(len(x) - mid, 1.0 / max(1, sr))
        cen1 = float((f1 * c1).sum() / c1.sum()); cen2 = float((f2 * c2).sum() / c2.sum())
        contorno = float(np.tanh((cen2 - cen1) / (sr * 0.1)))      # +sube / −baja
        arousal = min(1.0, max(0.0, 0.45 * rms + 0.35 * cen_n + 0.20 * zcr))
        valence = max(-1.0, min(1.0, 0.7 * (1.0 - 2.0 * planitud) + 0.3 * contorno))
        return round(arousal, 3), round(valence, 3)

    def _cargar_voces(self) -> list:
        """Carga el banco de voces R2-D2 (wav) desde voces_r2d2/ (en el árbol Célula_Madre) o
        ANIMA_VOCES_DIR. Cada voz lleva su afecto (arousal, valence). Sin carpeta → [] (usa síntesis).
        El afecto sale de AFECTO_VOCES si la etiqueta es conocida; si NO (sonido subido nuevo), se MIDE
        de la acústica (_afecto_acustico) → el repertorio crece sin límite y todo sonido es explorable."""
        base = os.environ.get("ANIMA_VOCES_DIR") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voces_r2d2")
        self._voces_dir = base
        voces = []
        if not os.path.isdir(base):
            return voces
        for nombre in sorted(os.listdir(base)):
            if not nombre.lower().endswith(".wav"):
                continue
            etiqueta = os.path.splitext(nombre)[0]
            ruta = os.path.join(base, nombre)
            try:
                try:
                    w = wave.open(ruta, "rb")
                    nch = w.getnchannels(); sr = w.getframerate() or 44100
                    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
                    if nch == 2:
                        a = a.reshape(-1, 2).mean(axis=1)
                except Exception:
                    import soundfile as _sf                        # fallback: WAV float32/24-bit (sonidos subidos)
                    a, sr = _sf.read(ruta, dtype="float64")
                    if getattr(a, "ndim", 1) > 1:
                        a = a.mean(axis=1)
                if etiqueta in self.AFECTO_VOCES:
                    aro, val = self.AFECTO_VOCES[etiqueta]          # voz R2-D2 conocida: afecto curado
                    origen = "curado"
                else:
                    aro, val = self._afecto_acustico(a, sr)         # sonido nuevo: afecto MEDIDO de su acústica
                    origen = "medido"
                voces.append({"label": etiqueta, "audio": a, "aro": float(aro), "val": float(val),
                              "afecto_origen": origen, "titulo": self._titulo(etiqueta)})
            except Exception:
                continue
        self._esparcir_afecto(voces)
        return voces

    def _esparcir_afecto(self, voces: list) -> None:
        """ESPARCE el afecto de los sonidos MEDIDOS por la región operativa del organismo, conservando su
        ORDEN acústico (más energía→más arousal; más brillo→más valencia) pero llenando el plano por RANGO.
        Por qué: las heurísticas acústicas tienden a AMONTONAR (p.ej. casi todo tonal cae en valencia alta),
        y entonces los sonidos nuevos quedan lejos de donde el organismo realmente está → no se emitirían nunca.
        Al esparcirlos por la zona alcanzable, cada estado puede mapear a un sonido DISTINTO y el repertorio
        se usa de verdad. NO afirmamos que estas coordenadas signifiquen alegría/miedo: son una colocación
        PROVISIONAL por huella acústica; el sentido, si surge, emerge del USO y la historia, no de aquí.
        Los sonidos CURADOS (R2-D2 originales) conservan su afecto a mano."""
        med = [v for v in voces if v.get("afecto_origen") == "medido"]
        n = len(med)
        if n < 3:
            return
        for r, v in enumerate(sorted(med, key=lambda v: v["aro"])):
            v["aro"] = round(0.12 + 0.56 * (r / (n - 1)), 3)        # 0.12 .. 0.68 (cubre la región del organismo)
        for r, v in enumerate(sorted(med, key=lambda v: v["val"])):
            v["val"] = round(-0.35 + 0.95 * (r / (n - 1)), 3)       # -0.35 .. 0.60

    def recargar_voces(self) -> int:
        """Re-explora la carpeta de voces y reconstruye el banco (para incorporar sonidos recién subidos
        sin reiniciar el organismo). Devuelve cuántas voces hay tras recargar. Idempotente y barato.
        Conserva el vocabulario PROPIO ya acuñado (lo vuelve a sumar tras recargar el banco base)."""
        self._voces = self._cargar_voces()
        self._creadas = self._cargar_creadas()
        return len(self._voces)

    def _guardar_wav_mono(self, audio, ruta: str) -> None:
        """Guarda audio float (~[-1,1]) como WAV mono PCM16 (mismo formato que el banco)."""
        a = np.asarray(audio, dtype=np.float64)
        pk = float(np.max(np.abs(a))) or 1.0
        pcm = (np.clip(a / pk * 0.9, -1.0, 1.0) * 32767.0).astype("<i2")
        with wave.open(ruta, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(self.sr); w.writeframes(pcm.tobytes())

    def _cargar_creadas(self) -> int:
        """Recupera del disco el vocabulario PROPIO CONSOLIDADO de vidas anteriores (sólo se persiste lo que
        cuajó: patrimonio) y lo suma al banco con su afecto EXACTO (sin re-medir ni esparcir: cada palabra
        cubre la región para la que se acuñó). Las recuperadas entran ya como ESTABLES. Continúa la numeración
        por tipo (creado/aprendida). Devuelve cuántas recuperó."""
        import json, re as _re
        man = os.path.join(self._creadas_dir, "manifiesto.json")
        if not os.path.isfile(man):
            return 0
        try:
            entradas = json.load(open(man, encoding="utf-8"))
        except Exception:
            return 0
        existentes = {v["label"] for v in self._voces}
        n = 0
        for e in entradas:
            label = e.get("label")
            if not label or label in existentes:
                continue
            ruta = os.path.join(self._creadas_dir, e.get("file", label + ".wav"))
            if not os.path.isfile(ruta):
                continue
            try:
                w = wave.open(ruta, "rb")
                a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
                origen = e.get("afecto_origen", "creado")     # "creado" (propia) | "aprendida" (emulada del otro)
                self._voces.append({"label": label, "audio": a, "aro": float(e["aro"]), "val": float(e["val"]),
                                    "afecto_origen": origen, "titulo": e.get("titulo", label),
                                    "estado": "estable", "usos": int(e.get("usos", self.USOS_CONSOLIDA)),
                                    "nacida": 0, "ultimo_uso": 0})
                existentes.add(label); n += 1
                m = _re.search(r"(\d+)$", label)              # continúa la numeración por tipo
                if m:
                    idx = int(m.group(1))
                    if origen == "aprendida":
                        self._aprendidas = max(self._aprendidas, idx)
                    else:
                        self._creadas = max(self._creadas, idx)
            except Exception:
                continue
        return n

    def _persistir_creada(self, voz: dict) -> None:
        """Escribe a disco una palabra CONSOLIDADA (patrimonio) — WAV + entrada en el manifiesto — para que
        SOBREVIVA reinicios y se ACUMULE. Sólo se llama al consolidar (no al acuñar): la persistencia es la
        marca de que la palabra CUAJÓ. Upsert por label (mantiene actualizado el contador de usos).
        Best-effort: si falla la escritura, la palabra sigue viva en RAM esta sesión."""
        import json
        try:
            os.makedirs(self._creadas_dir, exist_ok=True)
            fn = voz["label"] + ".wav"
            self._guardar_wav_mono(voz["audio"], os.path.join(self._creadas_dir, fn))
            man = os.path.join(self._creadas_dir, "manifiesto.json")
            entradas = []
            if os.path.isfile(man):
                try:
                    entradas = json.load(open(man, encoding="utf-8"))
                except Exception:
                    entradas = []
            entradas = [e for e in entradas if e.get("label") != voz["label"]]   # upsert
            entradas.append({"label": voz["label"], "file": fn, "aro": voz["aro"], "val": voz["val"],
                             "titulo": voz["titulo"], "afecto_origen": voz.get("afecto_origen", "creado"),
                             "usos": int(voz.get("usos", 0))})
            json.dump(entradas, open(man, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
        except Exception:
            pass

    def voz_actual(self, fila: dict) -> dict:
        """Qué voz R2-D2 emite el organismo para ESTE estado (la más cercana a su afecto) + el afecto.
        Determinista → sirve para REGISTRAR en el CSV la 'conversación': qué sonido usa en cada contexto."""
        aro, val = self._afecto(fila)
        label = "-"; titulo = "-"; origen = "banco"
        if self._voces:
            cand = sorted(self._voces, key=lambda v: (v["aro"] - aro) ** 2 + (v["val"] - val) ** 2)
            k = self._pool_k(len(cand))                 # MISMO pool exploratorio que la emisión real
            self._voz_seq = (getattr(self, "_voz_seq", 0) + 1)
            v = cand[_stable_seed(f"{self.organismo_id}:r2voz:lbl:{self._voz_seq}") % k]
            label = v["label"]; titulo = v.get("titulo", label)
            origen = {"creado": "creado", "aprendida": "aprendida"}.get(v.get("afecto_origen"), "banco")
        propias = [x for x in self._voces if x.get("afecto_origen") in ("creado", "aprendida")]
        estables = sum(1 for x in propias if x.get("estado") == "estable")
        aprendidas = sum(1 for x in propias if x.get("afecto_origen") == "aprendida")
        return {"voz_emitida": label, "voz_titulo": titulo, "voz_origen": origen,
                "voz_propias": len(propias), "voz_estables": estables, "voz_aprendidas": aprendidas,
                "voz_creadas": int(self._creadas), "voz_arousal": round(aro, 4), "voz_valence": round(val, 4)}

    def _pool_k(self, n_cand: int) -> int:
        """Tamaño del pool de voces entre las que el organismo EXPLORA: crece con su 'apertura' (act_perm /
        exploración del gesto). Más abierto → considera voces más lejanas del óptimo afectivo → usa más
        repertorio. En reposo (apertura≈0) se queda cerca (pool 3). 3..12 candidatos."""
        try:
            apertura = float(self._fila.get("act_perm", 0.0) or 0.0)
            apertura = max(apertura, abs(float(self.gesto.get("g_freq", 0.0))) * 0.5 if self.gesto else 0.0)
        except Exception:
            apertura = 0.0
        return min(max(3, int(round(3 + 9 * min(1.0, apertura)))), max(1, n_cand))

    def _afecto(self, fila: dict) -> tuple:
        """Proyecta la fisiología a (arousal, valence): cuán ACTIVADO y cuán BIEN está el organismo.
        El estado manda qué SIENTE; la voz sólo lo expresa (no se impone significado simbólico)."""
        g = lambda k, d=0.0: float(fila.get(k, d) or d)
        OI = g("OI"); nec = g("necesidad_efectiva", g("necesidad")); H = g("H_homeostasis")
        RC = g("RC_total"); E = g("energia", g("E")); estr = g("estructura"); lat = abs(g("balance_LR"))
        arousal = min(1.0, max(0.0, 0.45 * RC + 0.30 * E + 0.25 * lat))
        valence = max(-1.0, min(1.0, (OI + 0.30 * H + 0.30 * estr) - nec))
        return arousal, valence

    def _afecto_a_params(self, aro: float, val: float) -> dict:
        """Traduce el estado afectivo (+ la exploración del GESTO/balbuceo) a parámetros del aparato fonador.
        R2-D2 por construcción (todo pasa por el motor ARP 2600). NO es un código palabra↔significado: el
        mismo afecto produce una forma PARECIDA, pero el gesto introduce variación → la palabra acuñada es
        propia y novedosa, no una entrada de diccionario. Ejes: tono←arousal, contorno (sube/baja)←valencia,
        timbre metálico (FM)←arousal, tensión/ruido←valencia negativa, vibrato←repetición/arousal."""
        g = self.gesto or {}
        jf = float(g.get("g_freq", 0.0) or 0.0); ji = float(g.get("g_intensidad", 0.0) or 0.0)
        jp = float(g.get("g_pausa", 0.0) or 0.0); jr = float(g.get("g_repeticion", 0.0) or 0.0)
        a = max(0.0, min(1.0, aro))
        f0 = 300.0 + 1200.0 * a + 200.0 * jf            # tono base ← arousal (+ exploración)
        subir = val >= 0                                 # contorno ← valencia (sube=positivo, baja=negativo)
        f_ini = f0 * (0.85 if subir else 1.2)
        f_fin = f0 * (1.5 if subir else 0.6)
        fm_index = 0.5 + 6.0 * a + 2.0 * abs(ji)         # timbre metálico ← arousal
        tension = max(0.0, -val) * 0.8                   # tensión/ruido glotal ← malestar (valencia negativa)
        vibrato = (4.0 + 18.0 * abs(jr), 6.0 + 24.0 * a)  # LFO ← repetición / arousal
        dur = 0.30 + 0.30 * abs(jp)
        return dict(duracion=dur, f_ini=f_ini, f_fin=f_fin, fm_ratio=2.0, fm_index=(0.5, fm_index),
                    vibrato=vibrato, tension=tension, resonancia=0.3 * abs(val), res_centro=1200 + 800 * a,
                    ataque=0.01 + 0.1 * abs(jp), caida=0.1, sostiene=0.5)

    def _quizas_crear(self, fila: dict, aro: float, val: float, seq: int):
        """SEGUNDA VÍA: ¿el organismo ACUÑA una palabra nueva en vez de tirar una del banco?
        La crea cuando su repertorio NO cubre lo que necesita expresar —la voz más cercana queda LEJOS de su
        estado afectivo (un HUECO)— y ese hueco RECURRE (no por un estado fugaz), y tiene ENERGÍA para pagar
        el coste (mayor que reutilizar), y su libertad funcional lo decide. La palabra acuñada se SUMA al
        banco: cubre esa región y luego puede REUSARSE barata. Así el vocabulario crece desde la vida del
        organismo (sus necesidades expresivas no cubiertas), no desde nuestro diseño. Devuelve la voz o None."""
        activas = sum(1 for v in self._voces if v.get("afecto_origen") in ("creado", "aprendida"))
        if self._fonador is None or not self._voces or activas >= self.MAX_CREADAS:
            return None
        gap = min((v["aro"] - aro) ** 2 + (v["val"] - val) ** 2 for v in self._voces) ** 0.5
        if gap <= self.GAP_CREAR:
            self._gap_reciente.append(None)              # el banco cubre este estado
            return None
        self._gap_reciente.append((round(aro, 2), round(val, 2)))   # hueco: región no cubierta
        # RECURRENCIA: ¿la MISMA región no cubierta ha vuelto a aparecer? (no acuñar por algo pasajero)
        cerca = sum(1 for h in self._gap_reciente if h and abs(h[0] - aro) < 0.12 and abs(h[1] - val) < 0.12)
        if cerca < self.RECURRENCIA_CREAR:
            return None
        # ENERGÍA: el coste es real; sin energía no se puede crear (expresa imperfecto con el banco)
        energia = float(fila.get("energia", fila.get("met_energia", 0.0)) or 0.0)
        if energia < self.ENERGIA_MIN_CREAR:
            return None
        # LIBERTAD FUNCIONAL: aun con hueco recurrente y energía, a veces NO crea
        if (_stable_seed(f"{self.organismo_id}:crear:{seq}") % 1000) / 1000.0 > self.P_CREAR:
            return None
        # ACUÑA: afecto + exploración del gesto → parámetros del aparato → síntesis R2-D2 propia
        try:
            audio = np.asarray(self._fonador.vocalizar(**self._afecto_a_params(aro, val)), dtype=np.float64)
        except Exception:
            return None
        self._creadas += 1
        suf = self.organismo_id[-1] if self.organismo_id else "X"
        # NACE PROVISIONAL: es una HIPÓTESIS del organismo sobre cómo expresarse, no patrimonio todavía.
        # No se persiste aún: sólo si la REUTILIZA (→ _registrar_uso → consolida) pasará a disco. Si no la
        # reusa, _podar_provisionales la abandona. La historia, no la creación, decide qué queda.
        voz = {"label": f"palabra_{suf}{self._creadas:03d}",
               "audio": audio[: int(3.0 * self.sr)], "aro": float(aro), "val": float(val),
               "afecto_origen": "creado", "titulo": f"palabra propia {self._creadas}",
               "estado": "provisional", "usos": 0, "nacida": int(self._emision_seq), "ultimo_uso": int(self._emision_seq)}
        self._voces.append(voz)                          # se suma al banco como hipótesis provisional
        self._gap_reciente.clear()                       # esa región queda (tentativamente) cubierta
        self.ultima_voz_origen = "creado"; self.ultimo_costo_voz = self.COSTO_CREAR
        self._costo_pendiente += self.COSTO_CREAR         # gasto que el metabolismo cobrará una vez
        return voz

    def _registrar_uso(self, voz: dict) -> None:
        """REUSO de una palabra propia/aprendida (selección emergente). La reutilización es lo que CONSOLIDA:
        al alcanzar USOS_CONSOLIDA, la palabra deja de ser hipótesis provisional y pasa a ESTABLE (patrimonio
        → se guarda a disco). Así el vocabulario crece por SELECCIÓN (uso), no por simple acumulación."""
        if voz.get("afecto_origen") not in ("creado", "aprendida"):
            return
        voz["usos"] = int(voz.get("usos", 0)) + 1
        voz["ultimo_uso"] = int(self._emision_seq)
        if voz.get("estado") != "estable" and voz["usos"] >= self.USOS_CONSOLIDA:
            voz["estado"] = "estable"                    # CUAJÓ: la experiencia la incorporó
            self._persistir_creada(voz)                  # ahora sí es patrimonio → persiste y se acumula
        elif voz.get("estado") == "estable":
            self._persistir_creada(voz)                  # mantiene actualizado el contador de usos en disco

    def _podar_provisionales(self) -> int:
        """ABANDONA las palabras provisionales que NO se reusaron en VIDA_PROVISIONAL emisiones: la hipótesis
        no cuajó y la historia la deja caer (olvido por desuso). Las ESTABLES (patrimonio) no se podan.
        Devuelve cuántas se abandonaron. Esto hace que el vocabulario EVOLUCIONE por selección, no que crezca."""
        if not self._voces:
            return 0
        vivos = []; podadas = 0
        for v in self._voces:
            if (v.get("estado") == "provisional" and
                    self._emision_seq - int(v.get("ultimo_uso", 0)) > self.VIDA_PROVISIONAL):
                podadas += 1                              # no cuajó → se abandona (sólo estaba en RAM)
                continue
            vivos.append(v)
        if podadas:
            self._voces = vivos
        return podadas

    def quizas_emular(self, peer: dict, fila: dict, seq: int):
        """IMITACIÓN entre organismos (lenguaje compartido). Si el OTRO vocaliza una palabra DISTINTIVA que él
        inventó/aprendió y que MI banco no cubre, puedo EMULARLA: re-sintetizo MI versión con mi propio aparato
        a ese mismo afecto (NO copio su audio — la semejanza emerge de la historia, no por copia). Así la
        invención del otro puede entrar en MI vocabulario y el léxico propio CONVERGE entre A y B, en vez de
        divergir en dos linajes. Conducta libre (P_EMULAR) y costosa (como crear). Nace PROVISIONAL: deberá
        reusarse para cuajar, igual que una palabra propia. Devuelve la voz aprendida o None."""
        if self._fonador is None or not peer or not self._voces or not peer.get("vivo"):
            return None
        if peer.get("voz_origen", "banco") not in ("creado", "aprendida"):   # sólo emulo lo que el otro INVENTÓ
            return None
        try:
            aro = float(peer.get("voz_arousal", 0.0) or 0.0); val = float(peer.get("voz_valence", 0.0) or 0.0)
        except Exception:
            return None
        activas = sum(1 for v in self._voces if v.get("afecto_origen") in ("creado", "aprendida"))
        if activas >= self.MAX_CREADAS:
            return None
        gap = min((v["aro"] - aro) ** 2 + (v["val"] - val) ** 2 for v in self._voces) ** 0.5
        if gap <= self.GAP_EMULAR:                        # ya cubro esa región: no necesito emular
            return None
        energia = float(fila.get("energia", fila.get("met_energia", 0.0)) or 0.0)
        if energia < self.ENERGIA_MIN_CREAR:
            return None
        if (_stable_seed(f"{self.organismo_id}:emular:{seq}") % 1000) / 1000.0 > self.P_EMULAR:
            return None
        try:
            audio = np.asarray(self._fonador.vocalizar(**self._afecto_a_params(aro, val)), dtype=np.float64)
        except Exception:
            return None
        self._aprendidas += 1
        suf = self.organismo_id[-1] if self.organismo_id else "X"
        voz = {"label": f"apr_{suf}{self._aprendidas:03d}", "audio": audio[: int(3.0 * self.sr)],
               "aro": aro, "val": val, "afecto_origen": "aprendida",
               "titulo": f"eco de {peer.get('voz_titulo', 'el otro')}",
               "estado": "provisional", "usos": 0, "nacida": int(self._emision_seq),
               "ultimo_uso": int(self._emision_seq)}
        self._voces.append(voz)                           # entra en MI vocabulario como hipótesis aprendida
        self.ultima_voz_origen = "aprendida"; self.ultimo_costo_voz = self.COSTO_CREAR
        self._costo_pendiente += self.COSTO_CREAR
        return voz

    def vocabulario_propio(self) -> list:
        """Devuelve el vocabulario PROPIO (acuñado/aprendido) con sus métricas — para el estudio longitudinal:
        radio de uso, vida, reutilización, estable vs provisional, propio vs aprendido del otro."""
        out = []
        for v in self._voces:
            if v.get("afecto_origen") in ("creado", "aprendida"):
                out.append({"label": v["label"], "titulo": v.get("titulo"), "origen": v["afecto_origen"],
                            "estado": v.get("estado", "estable"), "usos": int(v.get("usos", 0)),
                            "aro": round(v["aro"], 3), "val": round(v["val"], 3),
                            "nacida": int(v.get("nacida", 0)), "ultimo_uso": int(v.get("ultimo_uso", 0))})
        return out

    def consumir_costo(self) -> float:
        """Devuelve (y resetea) el coste energético acumulado de las emisiones desde la última lectura.
        El bucle se lo pasa al metabolismo como gasto extra → ACUÑAR una palabra cuesta energía DE VERDAD
        (más que reutilizar el banco). Se cobra UNA vez por creación, no por paso."""
        c = self._costo_pendiente
        self._costo_pendiente = 0.0
        return c

    def _audio_r2d2_samples(self, fila: dict, seq: int):
        """Elige la voz que EMITE el organismo. Primero la SEGUNDA VÍA: ¿acuñar una palabra propia porque el
        banco no cubre su necesidad? Si no, tira del banco la voz más cercana a su afecto (con exploración).
        Devuelve el sample (≤3s) o None si no hay banco cargado."""
        if not self._voces:
            return None
        self._emision_seq += 1                            # reloj de emisiones (vida media / abandono)
        self._podar_provisionales()                       # deja caer las hipótesis que no cuajaron
        aro, val = self._afecto(fila)
        nueva = self._quizas_crear(fila, aro, val, seq)
        if nueva is not None:
            return np.array(nueva["audio"][: int(3.0 * self.sr)], dtype=np.float64)
        self.ultima_voz_origen = "banco"; self.ultimo_costo_voz = self.COSTO_USAR
        cand = sorted(self._voces, key=lambda v: (v["aro"] - aro) ** 2 + (v["val"] - val) ** 2)
        # EXPLORACIÓN del repertorio: no siempre el más cercano. El pool crece con la 'apertura' del
        # organismo (act_perm / exploración del gesto); más abierto → considera voces más lejanas del
        # óptimo afectivo → usa más repertorio. Determinista por seq (mismo estado+seq → misma voz).
        k = self._pool_k(len(cand))
        elegida = cand[_stable_seed(f"{self.organismo_id}:r2voz:{seq}") % k]
        self._registrar_uso(elegida)                      # REUSO: lo que consolida una palabra propia/aprendida
        if elegida.get("afecto_origen") in ("creado", "aprendida"):
            self.ultima_voz_origen = "creado" if elegida["afecto_origen"] == "creado" else "aprendida"
        return np.array(elegida["audio"][: int(3.0 * self.sr)], dtype=np.float64)   # cap 3s

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
