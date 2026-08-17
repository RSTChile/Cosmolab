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

# ESCALA COMPARTIDA (auditoría 4-ago-2026, regla 1 del plan): «un módulo compartido, no 168
# parches». Todo lo que aquí se relativiza usa rel/rel_contra de escala.py — no se reimplementa.
import sys as _sys
_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RAIZ not in _sys.path:
    _sys.path.insert(0, _RAIZ)
# `escala` vive en celula_madre/; esto permite importar el organelo suelto (pruebas y smokes)
# además de dentro del organismo. Unificado el 5-ago-2026: la revisión encontró CUATRO
# variantes del mismo arranque, que es el problema contra el que existe el módulo compartido.
import os as _os, sys as _sys
_RAIZ_CM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _RAIZ_CM not in _sys.path:
    _sys.path.insert(0, _RAIZ_CM)
from escala import Escala, rel, rel_contra, NEUTRO
# LO QUE CUESTA EXISTIR, para poder ponerle precio a hablar en las mismas unidades (ver COSTO_USAR).
# Sin ciclo: VST_Metabolismo sólo importa math, os y escala. Se toma de ahí y no se copia el número,
# que es justamente el defecto que este anclaje corrige.
from VST_Metabolismo import BASAL as _BASAL


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
    """OBSOLETO — conservado sólo para no romper a quien lo importe desde fuera.

    QUÉ ESTABA MAL (medido sobre 99.646 pasos del organismo ANIMA_5Z934MWHNNRH, 3–4 ago 2026):
    el `10.0` de `tanh(v/10)` declaraba, para 308 columnas de naturaleza distinta, que «10 unidades
    es mucho». Nadie midió ninguna de las 308. Consecuencia medida: el 14,11 % de todos los valores
    cantados salía exactamente 1,0000 y el 31,72 % exactamente 0,0000; 16 columnas quedaban clavadas
    en 1,0000 en más del 90 % de los pasos y 30 columnas en 0,0000. Entre las clavadas en 1,0000
    está `ts_real` (el reloj: ~1,78e9 → tanh satura → el organismo canta la hora como una constante
    máxima), `XE`, `A_soporte_fatiga`, `act_perm_energia`, `mem_persistencia`, `ove_confianza`.
    Es decir: casi la mitad de lo que el organismo canta no era estado, era el techo del tanh.

    El canto es lo que oye el OTRO organismo, y de aquí cuelgan OidoDigital, ValorEcologicoVoz,
    Expectativa y la métrica de convergencia léxica. Ver `OrganoComunicacion._escalar_pares`:
    la versión viva compara cada columna contra SU PROPIA historia (escala.py), sin ningún 10.
    """
    scaled: list[tuple[str, float]] = []
    for k, v in pairs:
        if 0.0 <= v <= 1.0:
            y = v
        else:
            y = 0.5 + 0.5 * math.tanh(v / 10.0)
        if math.isfinite(y):
            scaled.append((k, max(0.0, min(1.0, y))))
    return scaled


_SUPER = "⁰¹²³⁴⁵⁶⁷⁸⁹"


def _a_superindice(n: int) -> str:
    return "".join(_SUPER[int(d)] for d in str(int(n)))


def _de_superindice(s: str) -> int:
    return int("".join(str(_SUPER.index(c)) for c in s)) if s else 1


# Manipulacion experimental, en el mismo idioma que ANIMA_NO_ACUNAR y ANIMA_IMITAR_FORMA: se lee
# una sola vez al importar para que los dos brazos corran el MISMO binario y la unica diferencia
# este declarada en el compose. Apagado = comportamiento historico.
AFECTO_PROPIO_AL_APRENDER = os.environ.get(
    "ANIMA_AFECTO_PROPIO_AL_APRENDER", "").strip().lower() in ("1", "true", "yes", "on")


def _partir_eco(titulo: str) -> tuple:
    """Separa un titulo en (profundidad, raiz). El bucle, y no un solo corte, para que un titulo
    del formato viejo ("eco de eco de X") colapse entero y no un nivel."""
    resto = titulo
    n = 0
    while resto.startswith("eco"):
        r = resto[3:]
        i = 0
        while i < len(r) and r[i] in _SUPER:
            i += 1
        digitos, cola = r[:i], r[i:]
        if not (cola.startswith(" de ") and cola[4:].strip()):
            break
        n += _de_superindice(digitos)
        resto = cola[4:]
    return n, resto


def _componer_eco(n: int, raiz: str) -> str:
    if n <= 0:
        return raiz
    return "eco%s de %s" % (_a_superindice(n) if n > 1 else "", raiz)


def titulo_eco(titulo_ajeno: str) -> str:
    """Titulo de una palabra aprendida: la RAIZ del linaje y CUANTAS manos la han tocado.

    QUE ESTABA MAL (medido el 16-ago-2026). El titulo se construia concatenando
    f"eco de {titulo del otro}", y los cuatro organismos del lab estan en ANILLO
    (A->B->C->D->A), asi que una palabra da la vuelta y vuelve a entrar con un "eco de" mas
    encima, sin tope. Medido en la bitacora: 18 niveles de anidamiento, y 35.180 de las 60.032
    filas de habla aprendida eran la MISMA raiz -- "palabra propia 1" -- dando vueltas al anillo.
    El titulo ocupaba la linea entera y aun asi no decia lo unico que importa: de que palabra
    viene y por cuantas manos ha pasado. Habia que contar "eco de" a ojo.

    Ahora cabe en dos caracteres y dice mas:
        eco de palabra propia 1      (primera mano: identico a antes, no rompe la serie)
        eco{2} de palabra propia 1   (segunda, con el numero en superindice)
        eco{18} de palabra propia 1
    La raiz queda a la vista y la profundidad es un numero, no una longitud.

    NO cambia ninguna dinamica: `titulo` es metadato de presentacion y de bitacora. Lo que sigue
    pendiente es que el afecto de la palabra aprendida se HEREDA del par (quizas_emular toma
    peer["voz_arousal"]) en vez de ser el que ESA palabra le produce a quien la aprende. Mientras
    eso siga asi, lo que circula por el anillo es un numero congelado. Ver el criterio S=I<->E.
    """
    n, raiz = _partir_eco((titulo_ajeno or "").strip() or "el otro")
    return _componer_eco(n + 1, raiz)


def normalizar_titulo(titulo: str) -> str:
    """Colapsa un titulo del formato viejo SIN aumentar la profundidad.

    Las palabras ESTABLES se guardan a disco con su titulo, asi que un vocabulario acuniado
    antes del 16-ago-2026 vuelve con "eco de eco de eco de X" y se seguiria mostrando asi para
    siempre aunque nadie lo vuelva a imitar. Al restaurarlas se normalizan: el linaje es el
    mismo, sólo se escribe de otra forma. Un titulo que no sea un eco vuelve intacto.
    """
    base = (titulo or "").strip()
    if not base:
        return base
    n, raiz = _partir_eco(base)
    return _componer_eco(n, raiz) if n else base


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
        # ESCALAS PROPIAS DEL CANTO: una por columna cantada. Sustituyen al `tanh(v/10)` (ver
        # _robust_scale_pairs): cada magnitud se compara con lo habitual DE ESA MAGNITUD, no contra
        # un 10 que nadie midió. Aprenden UNA sola vez por paso, en observar().
        self._escala_canto: dict[str, dict] = {}
        # ESCALAS DEL AFECTO: una por pata del arousal (ver _afecto). Mismo motivo.
        self._escala_arousal: dict[str, Escala] = {}
        self._paso_cache: tuple[int, float] = (-1, 0.0)   # (nº de voces, resolución del repertorio)
        # DIAGNÓSTICO PUBLICABLE (lote E del plan): por qué NO se acuñó / NO se emuló en el último
        # intento. Sin esto, «cinco guardas cortan la vía y no se sabe cuál» — que es exactamente lo
        # que pasó durante 99.646 pasos con CERO emulaciones.
        self._bloqueo_crear = "sin_evaluar"
        self._bloqueo_emular = "sin_evaluar"
        self._gap_banco = 0.0        # distancia de MI estado a la voz más cercana de mi repertorio
        self._gap_peer = 0.0         # distancia de la palabra del OTRO a la voz más cercana mía
        self._creadas = 0          # mayor índice de palabra propia ACUÑADA emitido (monotónico, para etiquetar)
        self._aprendidas = 0       # mayor índice de palabra APRENDIDA (emulada del otro) emitido
        self._emision_seq = 0      # reloj de emisiones (para vida media / abandono de provisionales)
        self._cargar_creadas()     # recupera el vocabulario PROPIO consolidado de vidas anteriores

    def _escalar_pares(self, pares: list[tuple[str, float]],
                       aprender: bool = False) -> list[tuple[str, float]]:
        """NORMALIZADOR DEL CANTO — cada columna contra SU PROPIA historia, sin ningún número libre.

        QUÉ ESTABA MAL: `0.5 + 0.5*tanh(v/10)` (ver _robust_scale_pairs) fijaba una sola escala —el
        10— para las 308 columnas que el organismo canta. MEDIDO sobre 99.646 pasos: 14,11 % de los
        valores cantados salían exactamente 1,0000 y 31,72 % exactamente 0,0000; 16 columnas
        clavadas en 1,0000 en >90 % de los pasos (entre ellas `ts_real`, el reloj).

        POR QUÉ ESTO SÍ ES AUTORREGULADO: `rel(|v|, escala)` vale 0,5 cuando la magnitud está en su
        valor de siempre PARA ESTE ORGANISMO, y no tiene parámetro que elegir: la escala la pone la
        historia. El signo se conserva alrededor de 0,5, así que v=0 sigue cantando 0,5 (mismo
        centro acústico que antes) y ya no hay techo alcanzable por el mero tamaño de las unidades.

        DOS DECISIONES EXPLÍCITAS:
        · Una columna que SIEMPRE ha vivido dentro de [0,1] se canta en crudo: su escala ya está
          declarada (son fracciones, probabilidades, proporciones) y relativizarla borraría que el
          organismo lleve mucho tiempo alto o bajo. Sólo se relativiza la columna que alguna vez
          salió de [0,1], que es justo la que no tiene escala declarada — y la decisión es POR
          COLUMNA, no por valor, para no reintroducir el escalón que el `tanh` tenía en v=1
          (antes: v=1,0 cantaba 1,0000 y v=1,0001 cantaba 0,5500).
        · Mientras la escala no tiene historia se canta 0,5 (política de arranque única de
          escala.py): «todavía no sé si esto es mucho o poco» en vez de inventarlo.

        El canto es una PERCEPCIÓN (cómo suena mi estado), no una condición de viabilidad: por eso
        aquí el consigo-mismo es legítimo.
        """
        salida: list[tuple[str, float]] = []
        for k, v in pares:
            est = self._escala_canto.get(k)
            if est is None:
                est = {"esc": Escala(), "sin_escala": False}
                self._escala_canto[k] = est
            if v < 0.0 or v > 1.0:
                est["sin_escala"] = True          # esta columna no tiene escala declarada
            if not est["sin_escala"]:
                y = v                              # ya viene en [0,1]: escala declarada, se respeta
            else:
                if aprender:
                    est["esc"].observar(abs(v))
                if not est["esc"].madura:
                    y = NEUTRO                     # arranque: indeterminado, no inventado
                else:
                    y = 0.5 + 0.5 * math.copysign(rel(abs(v), est["esc"]), v)
            if math.isfinite(y):
                salida.append((k, max(0.0, min(1.0, y))))
        return salida

    def _aprender_escalas_afecto(self, fila: dict) -> None:
        """Alimenta las escalas de las patas del arousal UNA vez por paso (ver _afecto).

        Va aparte de _afecto porque _afecto se consulta varias veces por paso (voz_actual, la
        emisión y el cobro del coste): si aprendiera ahí, la media móvil correría al triple de
        velocidad que el organismo y dejaría de ser lo habitual de un paso.
        """
        g = lambda k, d=0.0: float(fila.get(k, d) or d)
        patas = {"RC_total": g("RC_total"),
                 "met_energia": g("met_energia", g("energia", g("E"))),
                 "lateralidad": abs(g("balance_LR"))}
        for nombre, x in patas.items():
            esc = self._escala_arousal.get(nombre)
            if esc is None:
                esc = Escala()
                self._escala_arousal[nombre] = esc
            esc.observar(x)

    def observar(self, fila: dict, meta: dict | None = None) -> None:
        with self._lock:
            self._fila = dict(fila or {})
            if meta:
                self._meta = dict(meta)
            self._seq += 1
            self._updated = time.time()
            self._aprender_escalas_afecto(self._fila)
            # aprender=True SÓLO aquí: observar() se llama una vez por paso; audio() puede llamarse
            # varias veces (el WAV, el medidor L/R, el nivel de voz propia) y no debe enseñarle nada.
            self._historial.append(self._escalar_pares(_flatten_numeric(self._fila), aprender=True))

    def estado(self) -> dict:
        with self._lock:
            edad = None if not self._updated else round(time.time() - self._updated, 3)
            # La fila publicada al PAR debe llevar el GESTO vocal actual (g_freq/g_intensidad/
            # g_pausa/g_repeticion): es la "boca" que el otro lee para IMITAR. self._fila se
            # captura en observar() ANTES de que el gesto se calcule en el paso, así que aquí
            # lo fusionamos desde self.gesto (poblado tras proximo_gesto). Sin esto, el OAO del
            # par lee g_*=None → memoria ecoica de ceros → imitación imposible.
            fila_pub = dict(self._fila)
            if isinstance(self.gesto, dict):
                for _k, _v in self.gesto.items():
                    if _v is not None:
                        fila_pub[_k] = _v
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
                "fila": fila_pub,
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

        pairs = self._escalar_pares(_flatten_numeric(fila))   # sin aprender: sólo observar() enseña
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

    def _mono_a_wav_estereo(self, mono: np.ndarray, gain: float | None = None,
                            aplicar_gesto: bool = True, pan: float | None = None) -> bytes:
        """Pipeline común: ganancia → volumen → (gesto) → paneo L/R → WAV estéreo PCM16."""
        audio = _aplicar_ganancia_salida(mono, self.voice_gain if gain is None else gain, self.voice_target_rms)
        # GANANCIA DE SALIDA del usuario (slider): sube la voz DE VERDAD, por encima del tope que la
        # gobernanza pone al target_rms. tanh = limitador suave (sin clip duro) para que llegue al rojo.
        mono = np.tanh(np.asarray(audio, dtype=np.float64) * max(0.0, float(self.voice_volumen))) * 0.97
        if aplicar_gesto:
            mono = self._aplicar_gesto(mono)   # LIBERTAD EXPRESIVA: imprime el gesto explorado sobre la voz
        # ESTÉREO: la voz lleva la LATERALIDAD del organismo (paneo SUAVE por balance L/R de su estado).
        # Siempre claramente en AMBOS canales (atenuación máx 25%): centro = ambos llenos, no "sólo L".
        if pan is None:
            try:
                pan = float(self._fila.get("balance_LR", self._fila.get("lateralidad", 0.0)) or 0.0)
            except Exception:
                pan = 0.0
        pan = max(-1.0, min(1.0, float(pan)))
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

    def wav_bytes(self, seg: float = 0.5, modo: str = "FULL_STATE", gain: float | None = None) -> bytes:
        return self._mono_a_wav_estereo(self.audio(seg=seg, modo=modo), gain=gain, aplicar_gesto=True)

    def buscar_voz(self, clave: str):
        """Busca una voz del repertorio por label o título (exacto, case-insensitive).
        Sirve al reproductor por-palabra de la UI (En vivo / Historia / repertorio)."""
        if not clave:
            return None
        k = str(clave).strip().lower()
        if not k or k in ("-", "—"):
            return None
        for v in (self._voces or []):
            if str(v.get("label", "")).strip().lower() == k:
                return v
            if str(v.get("titulo", "")).strip().lower() == k:
                return v
        return None

    def wav_bytes_por_label(self, clave: str, gain: float | None = None) -> bytes | None:
        """WAV de UNA palabra concreta del repertorio (sin gesto en vivo, centrada).
        Permite escuchar 'palabra propia 4' u otra etiqueta sin alterar la emisión actual."""
        v = self.buscar_voz(clave)
        if v is None:
            return None
        audio = v.get("audio")
        if audio is None:
            return None
        mono = np.asarray(audio[: int(3.0 * self.sr)], dtype=np.float64)
        if mono.size == 0:
            return None
        return self._mono_a_wav_estereo(mono, gain=gain, aplicar_gesto=False, pan=0.0)

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

    def nivel_voz_propia_estereo(self, seg: float = 0.3) -> dict:
        """L/R (rms, pico) de la voz ESTÉREO que el organismo emite AHORA. Reproduce el mismo
        pipeline que wav_bytes (ganancia → volumen → gesto → paneo L/R): el medidor 'Organismo
        L/R' enciende AMBOS canales porque la voz es estéreo, no mono. El paneo (lateralidad)
        sólo cambia el BALANCE entre lados (atenuación máx 25%), nunca apaga un canal."""
        vacio = {"L": {"rms": 0.0, "pico": 0.0}, "R": {"rms": 0.0, "pico": 0.0}}
        try:
            a = self.audio(seg=seg, modo="R2D2")
            a = _aplicar_ganancia_salida(a, self.voice_gain, self.voice_target_rms)
            mono = np.tanh(np.asarray(a, dtype=np.float64) * max(0.0, float(self.voice_volumen))) * 0.97
            mono = self._aplicar_gesto(mono)
            try:
                pan = float(self._fila.get("balance_LR", self._fila.get("lateralidad", 0.0)) or 0.0)
            except Exception:
                pan = 0.0
            pan = max(-1.0, min(1.0, pan))
            L = mono * (1.0 - 0.25 * max(0.0, pan))   # mismo paneo que wav_bytes
            R = mono * (1.0 - 0.25 * max(0.0, -pan))
            def _rp(x):
                return {"rms": float(_rms(x)),
                        "pico": float(np.max(np.abs(x))) if x.size else 0.0}
            return {"L": _rp(L), "R": _rp(R)}
        except Exception:
            return vacio

    # SEGUNDA VÍA — decisión "usar banco vs. ACUÑAR palabra propia".
    #
    # GAP_CREAR = 0.22 y GAP_EMULAR = 0.05 ESTABAN MAL: eran distancias en el plano afectivo
    # comparadas contra nada. MEDIDO sobre 99.646 pasos: la distancia real del estado del organismo
    # a la voz más cercana de su repertorio tiene mediana 0,108 (p25 0,060 · p75 0,166 · p95 0,214 ·
    # máximo 0,383). Con el corte en 0,22 la condición «hay hueco» se cumplía el 3,3 % de los pasos
    # —por debajo del 5 % que el propio detector de degeneración del plan declara sospechoso— y el
    # organismo acuñó 3 palabras por vida (26 eventos en 10 vidas). Con el corte en 0,05 se cumplía
    # el 85,3 %, que es lo mismo por el otro lado: una guarda que casi nunca decide nada.
    #
    # LA CORRECCIÓN es comparar la distancia contra OTRA MAGNITUD DEL PROPIO ORGANISMO CON LAS
    # MISMAS UNIDADES (advertencia 1 de la auditoría: rel_contra antes que r/(1+r)): la RESOLUCIÓN
    # de su propio repertorio, o sea a qué distancia están típicamente dos voces vecinas suyas
    # —ver _paso_repertorio(). Hay hueco cuando el estado cae más lejos de toda voz que lo que las
    # voces distan entre sí: entonces ninguna voz lo distingue, y por eso hace falta una nueva.
    # No es una analogía: es la definición operativa de «mi vocabulario no llega ahí».
    # Es además autolimitante: cada palabra acuñada añade un punto al repertorio y aprieta la
    # resolución, así que acuñar se vuelve más difícil a medida que el vocabulario cubre más.
    # MEDIDO en el repertorio real de hoy (16 voces curadas + 6 provisionales): la resolución vale
    # 0,114 — que es, con dos cifras, el 0,12 que estaba escrito a mano en la ventana de
    # recurrencia tres líneas más abajo. Que el número puesto a ojo coincida con la magnitud
    # derivada es la mejor evidencia de que la derivación es la correcta.
    #
    # SEÑAL COSTOSA (teoría CS / diseño basal 2026-07-08):
    # Vocalizar gasta energía. Antes COSTO_USAR=0 y solo se cobraba ACUÑAR → hablar del banco era gratis
    # y E podía quedarse en techo 1. Ahora emitir cobra (cinética), silencio no; acuñar sigue más caro.
    # Escala endógena (arousal/gesto), no por etiqueta del sample → anti-Shannon.
    # Env opcional: ANIMA_COSTO_VOZ_USAR (default = COSTO_USAR).
    # HABLAR CUESTA LO QUE CUESTA EXISTIR (7-ago-2026) — la fase 4 que el comentario de arriba
    # declaraba pendiente, resuelta.
    #
    # ANTES: `COSTO_USAR = 0.010`, un número puesto a mano en unidades que nadie reconcilió con el
    # metabolismo. Su escala real: 0,010 / basal = **3,33 veces lo que cuesta existir un paso** — y
    # el organismo vocaliza el 85 % de los pasos, así que se cobraba casi continuamente. MEDIDO
    # sobre 6.606 pasos del régimen actual, la voz es el **64,3 % de todo el gasto** (el comentario
    # viejo decía 41 %: había crecido). Una emisión cuesta 0,0074 contra una ingesta mediana de
    # 0,0084 por paso: hablar una vez se come lo que da de comer un paso entero.
    #
    # Es el mismo defecto que `costo_base = 1.0` en la homeostasis, que valía 26 veces el
    # metabolismo entero: un precio declarado en un organelo, en unidades que el metabolismo no
    # reconoce.
    #
    # EL CASO EN LA NATURALEZA. El costo de vocalizar se mide como múltiplo de la tasa metabólica
    # de reposo, y en un despliegue sostenido queda en el orden de una o dos veces esa tasa.
    # Ninguna conducta que un animal ejecuta el 85 % de su vida puede costar, por unidad de tiempo,
    # más que existir: si costara, no sería sostenible por definición — que es literalmente lo que
    # se estaba observando. Se copia la ESTRUCTURA («un múltiplo modesto del reposo»), no un número.
    #
    # SIN CONSTANTE NUEVA: `BASAL` ya existe en el metabolismo y ya es la referencia canónica del
    # proyecto para ponerle precio a algo. El 0,010 DESAPARECE; no se sustituye por otro número.
    # `COSTO_CREAR = USOS_CONSOLIDA · COSTO_USAR` no se toca: su derivación se conserva y pasa sola
    # de 0,040 a 0,012.
    #
    # EFECTO MEDIDO EN REPLAY del organelo real, mismo mundo grabado, sin tocar la ingesta:
    #   05-ago: met_energia p50 0,0000 → 0,1210  ·  E==0 exacto del 59,2 % al 1,2 %
    #   06-ago: 0,0000 → 0,1845                  ·  del 51,2 % al 5,6 %
    #   07-ago: 0,0000 → 0,1166                  ·  del 56,6 % al 0,9 %
    # Y la ingesta BAJA (0,0084 → 0,0070): es ahorro, no extracción. El organismo no saca más de lo
    # mismo —eso sería mejorar la digestión por tener hambre, que es lo que la norma prohíbe—:
    # deja de malgastar.
    #
    # FALSACIÓN, que es lo que impide que esto sea «bajar el precio hasta que cuadre»: en la
    # ventana sorda del 4-ago (43.900 pasos con la entrada muda al 100 %) el organismo sigue con la
    # reserva en cero el 98,8 % del tiempo — y con la voz GRATIS, el 98,6 %. El arreglo NO lo
    # vuelve inmortal: cuando no hay nada que oír, sigue vacío. La viabilidad sigue siendo
    # absoluta. Y el balance queda positivo en el 48-51 % de los pasos, no en el 100 %: la reserva
    # no se dispara, encuentra su punto fijo.
    COSTO_USAR = _BASAL     # emitir del banco / reutilizar (por paso de emisión; lo cobra el metabolismo)
    # SELECCIÓN, no acumulación: acuñar ≠ incorporar. Una palabra nace PROVISIONAL (hipótesis del organismo
    # sobre cómo expresarse); sólo si la REUTILIZA se CONSOLIDA (pasa a patrimonio y se persiste); si no la
    # reusa, se ABANDONA (la historia, no la creación, decide qué queda). El vocabulario evoluciona por uso.
    USOS_CONSOLIDA = 4      # reusos para que una palabra provisional pase a ESTABLE (y se guarde a disco)
    # ORIGEN: derivación aritmética de dos constantes ya declaradas (fase 4 del plan, literal:
    # «COSTO_CREAR = n · COSTO_USAR con n = USOS_CONSOLIDA»). Acuñar cuesta por adelantado lo que
    # costará usar la palabra las veces que hacen falta para que cuaje: el organismo paga de entrada
    # el precio de probar su hipótesis. Da 4 × 0,010 = 0,040, que es EXACTAMENTE el 0.04 que estaba
    # escrito a mano — el número no cambia, deja de ser arbitrario y ahora se mueve solo si se
    # mueven sus dos padres.
    #
    # AHORA ES EL PRECIO DE LA PRIMERA PALABRA, no el de todas (8-ago-2026): ver costo_crear().
    COSTO_CREAR = USOS_CONSOLIDA * COSTO_USAR
    # ENERGIA_MIN_CREAR = 0.25 ELIMINADA. Estaba mal: exigía una reserva 6,25 veces el precio de
    # acuñar sin que nadie dijera por qué 6,25. MEDIDO: met_energia vale 0 exacto el 50,3 % de los
    # pasos y sólo supera 0,25 el 24,0 %, así que la guarda cortaba 3 de cada 4 ocasiones por un
    # número inventado. La sustituye la comparación real, en _quizas_crear/quizas_emular: la reserva
    # contra EL PRECIO (rel_contra(energia, COSTO_CREAR)) — «¿me alcanza para pagarlo?». Es la
    # advertencia 1 de la auditoría aplicada al caso que ella misma nombró, y es una CONDICIÓN DE
    # VIABILIDAD, no una percepción: por eso se compara contra el precio absoluto y NO contra la
    # propia historia de energía (un organismo crónicamente vacío leería «energía normal» y se
    # arruinaría acuñando).
    P_CREAR = 0.6           # libertad funcional: aun con hueco recurrente y energía, a veces NO crea
    RECURRENCIA_CREAR = 3   # el hueco debe REAPARECER ≥3 veces (no acuñar por un estado fugaz)
    MAX_CREADAS = 64        # tope del vocabulario propio (el umbral de gap ya limita; esto es un cinturón)
    VIDA_PROVISIONAL = 600  # emisiones sin reuso tras las que una palabra provisional se ABANDONA (no cuajó)
    # IMITACIÓN entre organismos: si el otro vocaliza algo que MI banco no cubre, puedo EMULARLO con mi propio
    # aparato (no copiar: re-sintetizo mi versión) → el vocabulario inventado puede CONVERGER (lenguaje
    # compartido) en vez de divergir en dos linajes. Emular es también una conducta libre y costosa.
    P_EMULAR = 0.5          # libertad funcional de emular la palabra del otro

    # Afecto (arousal, valence) de cada voz R2-D2 según su carácter (la etiqueta del sample).
    # Guía el mapeo estado→voz: el organismo emite la voz cuyo afecto más se parece al suyo.
    AFECTO_VOCES = {
        "screaming": (0.95, -0.9), "shout": (0.786, -0.6), "worried": (0.214, -0.5),
        "excited": (0.868, 0.7), "excited-2": (0.786, 0.6), "sing": (0.377, 0.9),
        "acknowledged": (0.05, 0.5), "chat": (0.05, 0.1),
        "6": (0.214, 0.2), "7": (0.132, -0.1), "13": (0.214, 0.0), "14": (0.377, 0.1),
        "15": (0.132, 0.2), "18": (0.05, -0.2), "19": (0.05, 0.0), "22": (0.295, 0.3),
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
    def _diagnostico_fisico(a: np.ndarray, sr: int) -> dict:
        """DESACTIVADO COMO SEMÁNTICA (corrección anti-Shannon de Alexis, 29-jun-2026):
        los rasgos acústicos son DIAGNÓSTICO FÍSICO, no significado. El estatuto de una palabra NO se
        obtiene de su espectro sino por CALIBRACIÓN EXPERIENCIAL (lo que le hace al organismo al
        escucharla → VST_CalibradorLexicoExperiencial). Esta función vivía como `_afecto_acustico` y
        derivaba (arousal, valence) por FFT/centroide/planitud — eso era Shannon encubierto ("la palabra
        significa por cómo suena") y queda PROHIBIDO. Se conserva SÓLO como descriptor físico para el
        observador (duración, RMS, brillo, planitud, frecuencia dominante). Su salida NO PUEDE alimentar
        arousal/valence, OVE, OAO, LF, selección de palabra ni significado vocal: por eso devuelve un dict
        rotulado, no una tupla de afecto, y NO se llama en la carga del banco."""
        if a is None or len(a) < 64:
            return {"dur_s": 0.0, "rms": 0.0, "centroide_hz": 0.0, "planitud": 0.0, "frec_dominante_hz": 0.0,
                    "NOTA": "diagnóstico físico, NO semántico"}
        x = a.astype(np.float64); x = x - x.mean()
        pk = np.max(np.abs(x)) or 1.0; x = x / pk
        rms = float(np.sqrt(np.mean(x * x)))
        win = np.hanning(len(x))
        S = np.abs(np.fft.rfft(x * win)) + 1e-9
        f = np.fft.rfftfreq(len(x), 1.0 / max(1, sr))
        centroide = float((f * S).sum() / S.sum())
        planitud = float(np.exp(np.mean(np.log(S))) / np.mean(S))   # Wiener: ~1 ruido, ~0 tonal
        frec_dom = float(f[int(np.argmax(S))])
        return {"dur_s": round(len(x) / max(1, sr), 4), "rms": round(rms, 5),
                "centroide_hz": round(centroide, 1), "planitud": round(planitud, 4),
                "frec_dominante_hz": round(frec_dom, 1), "NOTA": "diagnóstico físico, NO semántico"}

    def _cargar_voces(self) -> list:
        """Carga el banco de voces R2-D2 (wav) desde voces_r2d2/ (en el árbol Célula_Madre) o
        ANIMA_VOCES_DIR. Cada voz lleva una colocación (arousal, valence) para la SELECCIÓN. Las R2-D2
        conocidas usan AFECTO_VOCES (curado a mano); un sonido subido NUEVO entra en colocación NEUTRA y
        PROVISIONAL (anti-Shannon: su sentido NO se lee del espectro), a la espera de calibración
        experiencial. Sin carpeta → [] (usa síntesis)."""
        base = os.environ.get("ANIMA_VOCES_DIR") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voces_r2d2")
        self._voces_dir = base
        voces = []
        if not os.path.isdir(base):
            return voces
        for nombre in sorted(os.listdir(base)):
            if not nombre.lower().endswith(".wav"):
                continue
            # IDENTIDAD = id original (último segmento tras '__'). El archivo puede llevar un PREFIJO
            # humano con el cuadrante experiencial (REGIMEN__nombre__id.wav) tras la calibración; la
            # etiqueta interna se queda con el id estable para que AFECTO_VOCES/persistencia resuelvan
            # y el organismo no cambie de identidad sólo porque el observador renombró el archivo.
            etiqueta = os.path.splitext(nombre)[0].split("__")[-1]
            # TÍTULO mostrado: si el archivo trae el prefijo de la CALIBRACIÓN EXPERIENCIAL
            # (REGIMEN__Nombre__id, p.ej. COLAPSO__Dolor__sing), el nombre legible refleja lo que la voz
            # le HACE SENTIR al organismo (régimen + nombre experiencial), NO el viejo título curado a mano.
            # La IDENTIDAD interna sigue siendo el id estable (etiqueta): el observador renombró el archivo,
            # el organismo no cambia de identidad. Sin prefijo → título legible clásico (_titulo).
            _segs = os.path.splitext(nombre)[0].split("__")
            if len(_segs) == 3:
                _regimen, _nombre_exp, _ = _segs
                titulo_voz = f"{_nombre_exp} · {_regimen.replace('_', ' ').lower()}"
            else:
                titulo_voz = self._titulo(etiqueta)
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
                    aro, val = self.AFECTO_VOCES[etiqueta]          # voz R2-D2 conocida: afecto curado (a mano)
                    origen = "curado"
                else:
                    # SONIDO NUEVO (anti-Shannon): NO se lee afecto de su espectro. Queda en colocación
                    # NEUTRA y PROVISIONAL (centro de la región operativa), SIN estatuto, hasta que la
                    # CALIBRACIÓN EXPERIENCIAL (VST_CalibradorLexicoExperiencial) le dé un régimen por lo
                    # que le HACE al organismo al escucharlo. La acústica no decide su sentido.
                    aro, val = 0.40, 0.10
                    origen = "provisional"
                voces.append({"label": etiqueta, "audio": a, "aro": float(aro), "val": float(val),
                              "afecto_origen": origen, "titulo": titulo_voz})
            except Exception:
                continue
        self._esparcir_afecto(voces)
        return voces

    def _esparcir_afecto(self, voces: list) -> None:
        """DESACTIVADO (anti-Shannon, 29-jun-2026). Antes esparcía por el plano afectivo los sonidos cuyo
        afecto se MEDÍA de la acústica (origen 'medido'), conservando su orden espectral. Eso era leer el
        sentido de la señal (Shannon). Ya no existen sonidos 'medido': los nuevos entran 'provisional' en
        colocación neutra y su estatuto lo da la CALIBRACIÓN EXPERIENCIAL, no el espectro. Se mantiene como
        no-op para no romper la llamada en _cargar_voces; la colocación experiencial se hará explícita aparte."""
        return

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
                                    "afecto_origen": origen, "titulo": normalizar_titulo(e.get("titulo", label)),
                                    "estado": "estable", "usos": int(e.get("usos", self.USOS_CONSOLIDA)),
                                    "afecto_regla": e.get("afecto_regla"),
                                    "afecto_par_aro": e.get("afecto_par_aro"),
                                    "afecto_par_val": e.get("afecto_par_val"),
                                    "imitacion": e.get("imitacion"),
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
            entrada = {"label": voz["label"], "file": fn, "aro": voz["aro"], "val": voz["val"],
                       "titulo": voz["titulo"], "afecto_origen": voz.get("afecto_origen", "creado"),
                       "usos": int(voz.get("usos", 0))}
            # Trazas del ENSAYO del afecto propio (16-ago-2026): sin esto, una palabra aprendida que
            # sobrevive a un reinicio vuelve del disco sin decir bajo QUE REGLA se aprendio, y el
            # ensayo dura mas que los reinicios. Se anaden solo si existen, asi que el manifiesto
            # viejo se sigue leyendo igual y una palabra acunada no se etiqueta de nada.
            for k in ("afecto_regla", "afecto_par_aro", "afecto_par_val", "imitacion"):
                if voz.get(k) is not None:
                    entrada[k] = voz[k]
            entradas.append(entrada)
            json.dump(entradas, open(man, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
        except Exception:
            pass

    def voz_actual(self, fila: dict) -> dict:
        """Qué voz R2-D2 emite el organismo para ESTE estado (la más cercana a su afecto) + el afecto.
        Determinista → sirve para REGISTRAR en el CSV la 'conversación': qué sonido usa en cada contexto."""
        aro, val = self._afecto(fila)
        label = "-"; titulo = "-"; origen = "banco"; emul_de = ""
        if self._voces:
            cand = sorted(self._voces, key=lambda v: (v["aro"] - aro) ** 2 + (v["val"] - val) ** 2)
            k = self._pool_k(len(cand))                 # MISMO pool exploratorio que la emisión real
            self._voz_seq = (getattr(self, "_voz_seq", 0) + 1)
            v = cand[_stable_seed(f"{self.organismo_id}:r2voz:lbl:{self._voz_seq}") % k]
            label = v["label"]; titulo = v.get("titulo", label)
            origen = {"creado": "creado", "aprendida": "aprendida"}.get(v.get("afecto_origen"), "banco")
            emul_de = v.get("emulada_de", "")           # de quién copió, si esta voz es una emulación (ruta léxica)
        propias = [x for x in self._voces if x.get("afecto_origen") in ("creado", "aprendida")]
        estables = sum(1 for x in propias if x.get("estado") == "estable")
        aprendidas = sum(1 for x in propias if x.get("afecto_origen") == "aprendida")
        # voz_id = ID GLOBAL de la palabra (lleva la letra del organismo: palabra_A001 / apr_B002) → permite
        # TRAZAR rutas léxicas entre organismos. voz_emulada_de = ID global de la palabra que se emuló (de quién).
        # De las aprendidas, cuántas entraron COPIANDO LA FORMA del otro y no
        # re-sintetizándola. Sin este contador el experimento de imitación corre a
        # ciegas: la etiqueta se queda dentro del organismo y no se puede observar.
        apr_forma = sum(1 for x in propias
                        if x.get("afecto_origen") == "aprendida" and x.get("imitacion") == "forma")
        # DIAGNÓSTICO DE LAS DOS VÍAS (lote E del plan: voz_gap_banco, voz_gap_peer,
        # voz_emular_bloqueo, voz_bloqueo_motivo). Sin estas columnas no se podía saber cuál de las
        # cinco guardas cortaba la emulación, y por eso el hecho —cero emulaciones en 99.646 pasos—
        # sólo se descubrió al contar filas del CSV. Se publican el motivo del último intento y las
        # dos distancias junto con la resolución del repertorio, que es la referencia contra la que
        # se juzgan: con esas tres cifras la decisión queda auditable desde fuera.
        # OJO al leerlas: voz_actual() se consulta ANTES que quizas_emular() en el paso del bucle,
        # así que voz_emular_bloqueo va un paso por detrás. Es diagnóstico, no dinámica.
        paso = self._paso_repertorio()
        return {"voz_emitida": label, "voz_titulo": titulo, "voz_origen": origen,
                "voz_id": label, "voz_emulada_de": emul_de,
                "voz_propias": len(propias), "voz_estables": estables, "voz_aprendidas": aprendidas,
                "voz_aprendidas_forma": apr_forma,
                "voz_creadas": int(self._creadas), "voz_arousal": round(aro, 4), "voz_valence": round(val, 4),
                "voz_paso_repertorio": round(paso, 4),
                "voz_gap_banco": round(float(self._gap_banco), 4),
                "voz_gap_peer": round(float(self._gap_peer), 4),
                "voz_bloqueo_motivo": self._bloqueo_crear,
                "voz_emular_bloqueo": self._bloqueo_emular}

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
        El estado manda qué SIENTE; la voz sólo lo expresa (no se impone significado simbólico).

        AROUSAL — DOS COSAS ESTABAN MAL, las dos medidas sobre 99.646 pasos:

        1) La pata de la energía valía SIEMPRE CERO. Decía `E = g("energia", g("E"))`, y `energia`
           no es una columna de la fisiología: es la clave fantasma que la bitácora graba en 0,0
           (la reserva real se llama met_energia). Como `fila.get("energia", ...)` sólo cae al
           respaldo si la clave FALTA, y el `or d` del lector convierte el 0,0 en 0,0, el 30 % del
           peso del arousal no aportaba nada. PRUEBA: la correlación del voz_arousal registrado con
           `0,45·RC_total + 0,25·|balance_LR|` —la fórmula SIN la pata de energía— es 1,0000, y el
           error absoluto mediano 2,5e-5, o sea el redondeo a cuatro decimales del CSV. Añadiendo
           met_energia la correlación BAJA a 0,8915: la energía nunca estuvo dentro.

        2) Los pesos 0,45 / 0,30 / 0,25 suponían tres magnitudes comparables en [0,1], y no lo son.
           RC_total tiene mediana 0,00207 (p95 0,138), así que su 0,45 aportaba 0,0009. RESULTADO
           MEDIDO: el arousal vivió en [0 , 0,5815] y NUNCA pasó de 0,60 en 99.646 pasos. Como el
           repertorio coloca `shout` en 0,786, `excited-2` en 0,786, `excited` en 0,868 y
           `screaming` en 0,95, cuatro de las dieciséis voces curadas quedaron en una región del
           plano a la que el organismo no puede llegar: fueron emitidas 98, 2, 0 y 1 veces
           respectivamente sobre 99.646. El organismo tenía un grito que no podía dar.

        CORRECCIÓN AUTORREGULADA: cada pata se compara con SU PROPIA historia (rel de escala.py,
        0,5 = lo de siempre) y el arousal es el promedio de las patas que ya tienen historia —el
        mismo patrón que el plan prescribe para el OI: «promediar sólo sobre las patas presentes,
        normalizar cada pata contra su historia». Desaparecen los tres pesos y la unidad de cada
        magnitud deja de decidir cuánto pesa. El arousal es una PERCEPCIÓN (cuán activado estoy
        PARA MÍ) y no una condición de viabilidad: relativizarlo es legítimo.

        MEDIDO DESPUÉS, reproduciendo el código nuevo contra los mismos pasos (replay offline):
        mediana 0,212 (antes 0,019), máximo 0,830 (antes 0,582) y un 5,3 % de pasos por encima de
        0,60 (antes CERO): el grito vuelve a estar al alcance. UNA SALVEDAD HONESTA: el 44,5 % de
        los pasos da arousal casi 0, porque las tres patas valen 0 exacto a la vez —met_energia es
        0 exacto el 50,3 % del tiempo—, y eso hace que la mediana caiga fuera de la banda
        [0,35 , 0,65] que el detector de degeneración exige a un mapa r/(1+r). No es el mapa: es que
        el organismo pasa de verdad la mitad de su vida inerte. Es el caso que el propio plan anota
        («las variables que son 0 exacto la mitad del tiempo rompen la EMA: tratar el cero como
        estado, no como valor pequeño») y queda escrito aquí para que nadie lo lea como una
        degeneración nueva introducida por esta corrección.

        VALENCE SE DEJA COMO ESTABA, a propósito. No es un consigo-mismo: ya es una COMPARACIÓN
        entre magnitudes del propio organismo —lo que sostiene (OI, H, estructura) menos lo que le
        falta (necesidad)—, que es la forma que la norma pide. MEDIDO: mediana 0,3356, recorrido
        [-0,647 , 0,972], que cubre de sobra el eje de valencia del repertorio ([-0,9 , 0,9]); no
        hay degeneración que corregir. Y es la pata que más se parece a una condición de vida
        («¿me está yendo bien?»): relativizarla borraría el malestar sostenido, que es justo lo que
        la advertencia 2 de la auditoría prohíbe. Sus 0,30 quedan pendientes de otra medición.
        """
        g = lambda k, d=0.0: float(fila.get(k, d) or d)
        OI = g("OI"); nec = g("necesidad_efectiva", g("necesidad")); H = g("H_homeostasis")
        estr = g("estructura")
        patas = (("RC_total", g("RC_total")),
                 ("met_energia", g("met_energia", g("energia", g("E")))),
                 ("lateralidad", abs(g("balance_LR"))))
        rels = [rel(x, esc) for nombre, x in patas
                for esc in (self._escala_arousal.get(nombre),) if esc is not None and esc.madura]
        # sin historia todavía → NEUTRO: el centro del repertorio, no un valor inventado
        arousal = min(1.0, max(0.0, sum(rels) / len(rels) if rels else NEUTRO))
        valence = max(-1.0, min(1.0, (OI + 0.30 * H + 0.30 * estr) - nec))
        return arousal, valence

    def _paso_repertorio(self) -> float:
        """RESOLUCIÓN del repertorio: a qué distancia están, típicamente, dos voces vecinas MÍAS.

        Es la magnitud del propio organismo, en las mismas unidades (distancia en el plano afecto),
        contra la que se mide si un estado cae en un hueco — sustituye a GAP_CREAR=0,22 y
        GAP_EMULAR=0,05, que no se comparaban contra nada. Mediana de la distancia de cada voz a su
        vecina más próxima; se ignoran las coincidencias exactas porque dos voces en el MISMO punto
        son un punto, no una resolución de cero (si no, seis sonidos provisionales colocados todos
        en (0,40 · 0,10) dejarían la resolución en 0 y ninguna vía volvería a abrirse).
        MEDIDO en el repertorio real de hoy: 0,114 con las provisionales, 0,129 sólo con las curadas.
        Se recalcula sólo cuando cambia el número de voces (una palabra nueva cambia la resolución).
        """
        n = len(self._voces)
        if n < 2:
            return 0.0
        if self._paso_cache[0] == n:
            return self._paso_cache[1]
        P = np.array([[float(v["aro"]), float(v["val"])] for v in self._voces], dtype=np.float64)
        D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=-1))
        np.fill_diagonal(D, np.inf)
        vecinas = D.min(axis=1)
        vecinas = vecinas[np.isfinite(vecinas) & (vecinas > 1e-9)]
        paso = float(np.median(vecinas)) if vecinas.size else 0.0
        self._paso_cache = (n, paso)
        return paso

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

    def _belleza_oida(self, audio) -> float:
        """Cuánto le GUSTARÍA al organismo OÍR este sonido: lo pasa por su PROPIO tímpano (a intensidad viable,
        para oírse sin lastimarse) y lo juzga con el mismo criterio físico que un sonido del mundo. BONITO =
        coherente (cono rígido de Von Békésy que transmite al martillo) + bien transmitido + que NO dispara el
        reflejo estapedial (no abruma). GARABATO = fragmentado/ruidoso/abrumador → baja belleza. Anti-Shannon:
        la belleza no es una regla impuesta, es cómo SUENA en su propia oreja."""
        try:
            from VST_OrganoMembrana import OrganoMembrana
        except Exception:
            return 0.0
        x = np.asarray(audio, dtype=np.float64); x = x - x.mean()
        pk = float(np.max(np.abs(x))) or 1.0; x = x / pk * 0.05
        m = OrganoMembrana(self.sr); s = None
        for j in range(0, max(1, len(x) - 4800), 4800):
            s = m.procesar(x[j:j + 4800], x[j:j + 4800])
        if s is None:
            return 0.0
        coher = float(s.get("mem_coherencia", 0.0))
        trans = min(1.0, float(s.get("mem_transmitido_L", 0.0)) * 5.0)
        refl = float(s.get("mem_reflejo", 0.0))
        return float(coher * (0.4 + 0.6 * trans) * (1.0 - refl))

    def _palabras_propias(self) -> int:
        """Cuántas palabras PROPIAS (acuñadas o emuladas) hay ahora mismo en el banco.

        Es el mismo recuento que ya usaban _quizas_crear/quizas_emular para MAX_CREADAS; se saca
        aquí porque ahora también pone el precio (ver costo_crear). Cuenta provisionales y estables:
        una hipótesis que todavía no cuajó ya está compitiendo por las emisiones."""
        return sum(1 for v in self._voces if v.get("afecto_origen") in ("creado", "aprendida"))

    def costo_crear(self) -> float:
        """PRECIO DE ACUÑAR AHORA. Crece con lo que ya hay en el banco. Sin constante nueva.

        QUÉ ESTABA MAL. `COSTO_CREAR = USOS_CONSOLIDA · COSTO_USAR` es un precio FIJO: acuñar la
        palabra 21 costaba exactamente lo mismo que acuñar la tercera. Su propia derivación dice
        «acuñar cuesta por adelantado lo que costará usarla las veces que hacen falta para que
        cuaje», y esa cuenta estaba a medias: suponía que la palabra nueva se emitiría en CADA
        emisión, es decir que el organismo no tiene más vocabulario propio que ella.

        NO LO TIENE: LO MIDO. Sobre 23.953 pasos del 7-ago (06:00→24:00, régimen actual), las
        palabras propias se llevan en conjunto una tajada casi constante de las emisiones —36,9 %
        con 10 palabras, 36,1 % con 14, 38,9 % con 11: no crece con el vocabulario— y por tanto la
        tajada DE CADA UNA cae como 1/k: 3,69 % con k=10 contra 2,58 % con k=14 (razón medida 1,43;
        razón de k, 1,40). Las palabras propias no se reparten el mundo: se lo quitan entre ellas,
        porque todas nacen colocadas donde el organismo pasa su tiempo afectivo.

        LA CORRECCIÓN es completar la derivación que ya estaba declarada, no inventar otra: si la
        palabra nueva sólo saldrá 1 de cada k veces que el organismo tire de su vocabulario propio,
        acumular USOS_CONSOLIDA usos le costará k veces más, y eso es lo que paga por adelantado.

            precio = USOS_CONSOLIDA · COSTO_USAR · k        k = palabras propias + la nueva

        k=1 devuelve 0,012 EXACTAMENTE, el precio de hoy: con vocabulario pobre no cambia nada, así
        que el arreglo del hambre del 7-ago (COSTO_USAR anclado a BASAL, gasto 0,01143 → 0,00733)
        queda intacto. Con vocabulario rico el precio sube solo: 0,012 la primera, 0,252 la 21ª.

        A CUÁNTAS PALABRAS CONVERGE, Y POR QUÉ ÉSE ES EL NÚMERO. La guarda de energía es
        `rel_contra(energia, precio) > 0,5`, o sea `reserva > precio`. Con este precio eso es

            k* = reserva alcanzable / (USOS_CONSOLIDA · BASAL)

        es decir: **el organismo sostiene tantas palabras propias como veces le alcanza la reserva
        para pagar la prueba completa de una más.** No es un tope elegido —MAX_CREADAS=64 sigue
        siendo un cinturón que no toca nada—: es su economía. Y respira: si una hipótesis no cuaja,
        _podar_provisionales la deja caer, k baja y acuñar vuelve a abrirse. Homeostasis sin
        setpoint. MEDIDO sobre esos mismos pasos, la reserva supera 0,012 el 9,0 % del tiempo,
        0,024 el 6,1 %, 0,048 el 5,1 %, 0,10 el 4,0 % y 0,25 el 0,9 %: el precio deja de encontrar
        reserva justo en el rango en que el vocabulario propio ronda la decena.

        POR QUÉ NO SE USA `_paso_repertorio` DE REFERENCIA, que era el candidato natural. Dos
        razones, ambas medidas. (1) NO APRIETA: el comentario de la segunda vía afirma que «cada
        palabra acuñada añade un punto al repertorio y aprieta la resolución»; en el organismo real
        pasa lo contrario, porque las palabras nacen EN LOS HUECOS y su vecina queda lejos —
        `voz_paso_repertorio` sube de 0,1301 con 8 palabras propias a 0,1472 con 15 (+13 %). Frena
        (con paso mayor cuesta más que un estado cuente como hueco), pero un 13 % en siete palabras
        no es un precio. (2) Y sobre todo: la resolución la decide lo mismo que el precio tendría
        que gobernar. Poner el precio contra ella es un criterio relativizado contra sí mismo —
        trinquete sin suelo—, que es exactamente lo que la norma prohíbe.

        AUDITABLE DESDE FUERA SIN COLUMNA NUEVA: el precio vigente en cualquier fila del CSV es
        `COSTO_CREAR · (1 + voz_propias)`, y `voz_propias` ya se publica. No se añade columna a
        propósito: cada clave que entra en `fila` entra también en lo que el organismo CANTA.

        EFECTO MEDIDO EN REPLAY (analisis/voz_precio.py: este organelo y el metabolismo de verdad,
        en lazo cerrado sobre el mismo mundo grabado, 157.587 pasos = la vida entera registrada del
        3 al 7 de agosto; lo único distinto entre brazos es el precio):

                        gasto p50   reserva en cero   acuñadas por cuarto   converge a
          0,010 fijo      0,00937        75,5 %            2/9/2/1              11
          0,012 fijo      0,00536        58,0 %            9/5/2/3              17
          creciente       0,00536        58,0 %            4/10/3/2             14

        En la ventana de 18 h del régimen actual (23.953 pasos, 7-ago 06:00→24:00): 11 → 9
        palabras, con el gasto IDÉNTICO (0,00678 en los dos) y la reserva en cero 28,1 % contra
        28,2 %. Eso último no es un fallo del cambio, es su alcance: acuñar cuesta el 0,19 % del
        gasto total, así que **este precio decide el TAMAÑO del vocabulario, no el presupuesto**.
        Quien busque la reserva tiene que mirar el gasto, no el precio de la palabra.

        FALSACIÓN (la que impide que esto sea aflojar hasta que cuadre, al revés que la del 7-ago:
        aquí el riesgo es que un precio más caro pase por bueno sin haber probado nada). En la
        ventana SORDA del 4-ago —44.699 pasos con `energia_L` y `energia_R` en 0,000— la reserva
        sigue vacía el 99,8 % del tiempo, exactamente igual que con el precio actual (99,8 %) y que
        con el viejo (99,9 %), con ingesta mediana 0,00000. Sin nada que oír sigue muriéndose: la
        viabilidad sigue siendo absoluta. Y el precio nunca BAJA del de hoy (k≥1 ⇒ ≥ 0,012), así
        que este cambio no puede, por construcción, comprarle vida a nadie.
        """
        return self.COSTO_CREAR * (1 + self._palabras_propias())

    def _quizas_crear(self, fila: dict, aro: float, val: float, seq: int):
        """SEGUNDA VÍA: ¿el organismo ACUÑA una palabra nueva en vez de tirar una del banco?
        La crea cuando su repertorio NO cubre lo que necesita expresar —la voz más cercana queda LEJOS de su
        estado afectivo (un HUECO)— y ese hueco RECURRE (no por un estado fugaz), y tiene ENERGÍA para pagar
        el coste (mayor que reutilizar), y su libertad funcional lo decide. La palabra acuñada se SUMA al
        banco: cubre esa región y luego puede REUSARSE barata. Así el vocabulario crece desde la vida del
        organismo (sus necesidades expresivas no cubiertas), no desde nuestro diseño. Devuelve la voz o None."""
        if os.environ.get("ANIMA_NO_ACUNAR", "").strip().lower() in ("1", "true", "yes", "on"):
            self._bloqueo_crear = "ablacion"
            return None    # ABLACIÓN experimental (condición C3 tríada): TERCERO ESTÉRIL — no re-acuña, no
            #                engendra ecosistema léxico (análogo del "mediador pasivo" de Cosmogénesis).
        activas = self._palabras_propias()
        if self._fonador is None or not self._voces or activas >= self.MAX_CREADAS:
            self._bloqueo_crear = ("sin_fonador" if self._fonador is None else
                                   ("sin_banco" if not self._voces else "vocabulario_lleno"))
            return None
        gap = min((v["aro"] - aro) ** 2 + (v["val"] - val) ** 2 for v in self._voces) ** 0.5
        self._gap_banco = gap
        paso = self._paso_repertorio()               # resolución de MI repertorio (misma unidad que gap)
        # HUECO = el estado cae más lejos de toda voz mía que lo que distan entre sí mis propias voces.
        # rel_contra(gap, paso) > 0,5 es exactamente gap > paso, escrito en el idioma de escala.py para
        # que se publique el cociente y se pueda auditar. Antes: gap > 0,22, un número contra nada.
        if paso <= 0.0 or rel_contra(gap, paso) <= NEUTRO:
            self._gap_reciente.append(None)              # el banco cubre este estado
            self._bloqueo_crear = "sin_hueco" if paso > 0.0 else "sin_resolucion"
            return None
        self._gap_reciente.append((round(aro, 2), round(val, 2)))   # hueco: región no cubierta
        # RECURRENCIA: ¿la MISMA región no cubierta ha vuelto a aparecer? (no acuñar por algo pasajero)
        # La ventana era 0,12 escrito a mano; es la MISMA unidad que la resolución del repertorio y
        # medía casi lo mismo (0,114): «la misma región» = más cerca que dos voces vecinas mías.
        cerca = sum(1 for h in self._gap_reciente if h and abs(h[0] - aro) < paso and abs(h[1] - val) < paso)
        if cerca < self.RECURRENCIA_CREAR:
            self._bloqueo_crear = "hueco_fugaz"
            return None
        # ENERGÍA: el coste es real; sin energía no se puede crear (expresa imperfecto con el banco).
        # La reserva se compara CONTRA EL PRECIO, no contra 0,25 (ver la nota de ENERGIA_MIN_CREAR):
        # rel_contra(energia, precio) > 0,5 es «me alcanza para pagarlo». Condición de viabilidad,
        # comparada contra una magnitud real del organismo, nunca contra su propia historia.
        # EL PRECIO YA NO ES FIJO (8-ago-2026): sube con las palabras propias que ya hay en el
        # banco, porque la nueva tendrá que compartir con ellas las emisiones que la consolidan.
        # Ver costo_crear(): k=1 devuelve el mismo 0,012 de antes, k=21 devuelve 0,252.
        energia = float(fila.get("energia", fila.get("met_energia", 0.0)) or 0.0)
        precio = self.costo_crear()
        if rel_contra(energia, precio) <= NEUTRO:
            self._bloqueo_crear = "sin_energia"
            return None
        # LIBERTAD FUNCIONAL: aun con hueco recurrente y energía, a veces NO crea
        if (_stable_seed(f"{self.organismo_id}:crear:{seq}") % 1000) / 1000.0 > self.P_CREAR:
            self._bloqueo_crear = "libertad"
            return None
        # ACUÑA expresando su ESTADO: _afecto_a_params ya hace BONITO cuando se siente bien (limpio) y
        # GARABATO cuando está MOLESTO (valencia<0 → tensión/ruido glotal). La molestia se expresa con
        # garabatos; NO se censura ninguna expresión. La belleza/fealdad la JUZGA al OÍR (propiocepción),
        # no se le impone al CREAR — el organismo emite lo que su afecto pide, lindo o feo.
        try:
            audio = np.asarray(self._fonador.vocalizar(**self._afecto_a_params(aro, val)), dtype=np.float64)
        except Exception:
            self._bloqueo_crear = "fonador_fallo"
            return None
        self._bloqueo_crear = "acunada"
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
        self.ultima_voz_origen = "creado"; self.ultimo_costo_voz = precio
        self._costo_pendiente += precio                  # gasto que el metabolismo cobrará una vez
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

    def _forma_oida_del_otro(self):
        """Trae el bloque de voz que el otro está emitiendo AHORA, tal cual suena.

        El organismo ya oye al par por ese mismo canal (`VST_COMUNICACION_PEER`);
        aquí simplemente se conserva la onda en vez de descartarla. Devuelve mono
        float a la tasa del organismo, o None si no se pudo (y entonces el llamador
        re-sintetiza como siempre).
        """
        url = (os.environ.get("VST_COMUNICACION_PEER") or "").strip()
        if not url:
            return None
        try:
            import io as _io
            import urllib.request as _url
            import scipy.io.wavfile as _wav
            pet = _url.Request(url, headers={"User-Agent": "ANIMA-Organismo/1.2"})
            with _url.urlopen(pet, timeout=1.5) as r:      # acotado: corre dentro del tick
                crudo = r.read(4_000_000)
            sr, datos = _wav.read(_io.BytesIO(crudo))
            x = np.asarray(datos, dtype=np.float64)
            if x.ndim > 1:                                 # a mono
                x = x.mean(axis=1)
            pico = float(np.max(np.abs(x))) if x.size else 0.0
            if pico <= 0 or x.size < int(0.05 * sr):        # silencio o demasiado corto
                return None
            x = x / pico * 0.9
            if sr != self.sr:                               # remuestreo lineal, suficiente aquí
                n = int(len(x) * self.sr / float(sr))
                if n < 2:
                    return None
                x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
            return x[: int(3.0 * self.sr)]
        except Exception:
            return None

    def quizas_emular(self, peer: dict, fila: dict, seq: int):
        """IMITACIÓN entre organismos (lenguaje compartido). Si el OTRO vocaliza una palabra DISTINTIVA que él
        inventó/aprendió y que MI banco no cubre, puedo EMULARLA: re-sintetizo MI versión con mi propio aparato
        a ese mismo afecto (NO copio su audio — la semejanza emerge de la historia, no por copia). Así la
        invención del otro puede entrar en MI vocabulario y el léxico propio CONVERGE entre A y B, en vez de
        divergir en dos linajes. Conducta libre (P_EMULAR) y costosa (como crear). Nace PROVISIONAL: deberá
        reusarse para cuajar, igual que una palabra propia. Devuelve la voz aprendida o None."""
        # DIAGNÓSTICO (motivo por el que se abandona el intento). Esta vía llevaba 99.646 pasos sin
        # ejecutarse NI UNA VEZ —voz_aprendidas = 0 en 99.646 de 99.646 filas, y 0 eventos
        # 'palabra_aprendida' contra 26 'palabra_propia'— y no había forma de saber cuál de las
        # guardas cortaba, porque ninguna dejaba rastro. Ahora cada salida escribe su motivo y
        # voz_actual() lo publica.
        if os.environ.get("ANIMA_NO_ACUNAR", "").strip().lower() in ("1", "true", "yes", "on"):
            self._bloqueo_emular = "ablacion"
            return None    # ABLACIÓN C3 (tercero ESTÉRIL): tampoco EMULA — no engendra ecosistema por imitación
        if self._fonador is None or not peer or not self._voces or not peer.get("vivo"):
            # SOSPECHOSO PRINCIPAL, y no es una constante: `peer` llega de _peer_voz_estado(), que
            # sondea VST_COMUNICACION_PEER — variable de entorno cuyo valor por defecto es "" y que
            # no la fija ningún lanzador del repositorio. El organismo SÍ oye vecinos (medido:
            # alt_otro_presente = 1 el 55,1 % de los pasos, presencia_vecinos_n mediana 6), pero los
            # oye por el ROSTER de Presencia, que es otro canal. Si es eso, la vía no está cortada
            # por ninguna guarda: está escuchando en una puerta por la que no entra nadie. Queda
            # medible en la próxima corrida como bloqueo 'sin_par'.
            self._bloqueo_emular = ("sin_fonador" if self._fonador is None else
                                    ("sin_par" if not peer else
                                     ("sin_banco" if not self._voces else "par_no_vivo")))
            return None
        if peer.get("voz_origen", "banco") not in ("creado", "aprendida"):   # sólo emulo lo que el otro INVENTÓ
            self._bloqueo_emular = "par_del_banco"   # MEDIDO en la traza propia: pasa el 20,8 % de los pasos
            return None
        try:
            aro = float(peer.get("voz_arousal", 0.0) or 0.0); val = float(peer.get("voz_valence", 0.0) or 0.0)
        except Exception:
            self._bloqueo_emular = "afecto_del_par_ilegible"
            return None
        # BRAZO EXPERIMENTAL (16-ago-2026, `ANIMA_AFECTO_PROPIO_AL_APRENDER`). Por defecto APAGADO:
        # el comportamiento historico es que la palabra aprendida se guarda con el afecto DEL OTRO,
        # y por eso lo que circula por el anillo es un numero congelado -- medido: 0,3082 / 0,2990 /
        # 0,3000 en los tres primeros niveles de eco, y una sola raiz en 35.180 de 60.032 filas.
        #
        # Encendido, se separa lo que hasta ahora estaba fundido en una sola variable:
        #   - el afecto DEL PAR (aro/val) sigue decidiendo si merece la pena imitar, porque es lo
        #     que YO percibo como region no cubierta, y sigue dando FORMA al sonido resintetizado:
        #     la palabra tiene que seguir sonando a la suya o no es imitacion, es invencion.
        #   - el afecto PROPIO (aro_mio/val_mio) es el que se GUARDA con la palabra, porque es lo
        #     que esa palabra me hace a MI al oirla. Ahi es donde la imitacion se anida como
        #     experiencia interior y no antes (criterio S=I<->E de Alexis, 4-ago-2026).
        # La forma viene de fuera; el sentido, de dentro. `afecto_regla` queda en la voz para poder
        # separar los dos brazos en la bitacora sin adivinar.
        if AFECTO_PROPIO_AL_APRENDER:
            try:
                aro_mio, val_mio = self._afecto(fila)
                aro_mio, val_mio = float(aro_mio), float(val_mio)
            except Exception:
                self._bloqueo_emular = "afecto_propio_ilegible"
                return None
            regla_afecto = "propio"
        else:
            aro_mio, val_mio, regla_afecto = aro, val, "heredado"
        activas = self._palabras_propias()
        if activas >= self.MAX_CREADAS:
            self._bloqueo_emular = "vocabulario_lleno"
            return None
        gap = min((v["aro"] - aro) ** 2 + (v["val"] - val) ** 2 for v in self._voces) ** 0.5
        self._gap_peer = gap
        # Antes: gap <= 0,05, un número contra nada (MEDIDO: lo pasaba el 85,3 % de los pasos, o sea
        # casi nunca decidía). Ahora se compara contra la RESOLUCIÓN de mi propio repertorio, igual
        # que al acuñar: la palabra del otro merece emularse si cae más lejos de toda voz mía que lo
        # que distan entre sí mis voces — es decir, si ninguna de las mías la distingue. Emular
        # sigue siendo más fácil que acuñar porque no exige que el hueco REAPAREZCA.
        paso = self._paso_repertorio()
        if paso <= 0.0 or rel_contra(gap, paso) <= NEUTRO:   # ya cubro esa región: no necesito emular
            self._bloqueo_emular = "ya_cubierto" if paso > 0.0 else "sin_resolucion"
            return None
        # La reserva contra EL PRECIO, no contra 0,25 (ver la nota de ENERGIA_MIN_CREAR). MEDIDO:
        # met_energia sólo superaba 0,25 el 24,0 % de los pasos, así que ese número por sí solo
        # tiraba abajo tres de cada cuatro ocasiones sin que nadie hubiera medido el precio real.
        # MISMO PRECIO CRECIENTE QUE AL ACUÑAR (8-ago-2026): una palabra emulada entra en el mismo
        # banco y compite por las mismas emisiones, así que se paga por el mismo criterio. Sería
        # incoherente que imitar al otro esquivara el precio de tener vocabulario.
        energia = float(fila.get("energia", fila.get("met_energia", 0.0)) or 0.0)
        precio = self.costo_crear()
        if rel_contra(energia, precio) <= NEUTRO:
            self._bloqueo_emular = "sin_energia"
            return None
        if (_stable_seed(f"{self.organismo_id}:emular:{seq}") % 1000) / 1000.0 > self.P_EMULAR:
            self._bloqueo_emular = "libertad"
            return None
        # ── IMITACIÓN DE FORMA (experimental, ANIMA_IMITAR_FORMA=1) ──────────────
        # Por defecto el organismo RE-SINTETIZA su versión del afecto del otro: la
        # semejanza emerge de compartir la función de síntesis, no de copiar.
        # Con el interruptor encendido, en cambio, se queda con la FORMA que oyó.
        #
        # La diferencia no es cosmética. En el lenguaje humano la estructura viaja
        # DENTRO de la forma copiada: un niño repite una secuencia que todavía no
        # sabe analizar, y la estructura llega con ella. Re-derivar desde coordenadas
        # afectivas no puede transportar nada que no esté ya en la función; copiar
        # la forma sí, incluidos encadenamientos de más de un gesto.
        #
        # Es una manipulación experimental de una decisión de diseño deliberada.
        # Falla hacia el comportamiento original: si no se puede oír al otro, se
        # re-sintetiza como siempre.
        audio = None
        forma_copiada = False
        if os.environ.get("ANIMA_IMITAR_FORMA", "").strip().lower() in ("1", "true", "yes", "on"):
            audio = self._forma_oida_del_otro()
            forma_copiada = audio is not None
        if audio is None:
            try:
                audio = np.asarray(self._fonador.vocalizar(**self._afecto_a_params(aro, val)), dtype=np.float64)
            except Exception:
                self._bloqueo_emular = "fonador_fallo"
                return None
        self._bloqueo_emular = "emulada"
        self._aprendidas += 1
        suf = self.organismo_id[-1] if self.organismo_id else "X"
        voz = {"label": f"apr_{suf}{self._aprendidas:03d}", "audio": audio[: int(3.0 * self.sr)],
               "aro": aro_mio, "val": val_mio, "afecto_origen": "aprendida",
               "afecto_regla": regla_afecto,
               "afecto_par_aro": round(aro, 4), "afecto_par_val": round(val, 4),
               "imitacion": "forma" if forma_copiada else "resintesis",
               "titulo": titulo_eco(peer.get("voz_titulo", "")),
               "emulada_de": peer.get("voz_emitida") or peer.get("voz_id") or "",   # RUTA: ID global de la palabra que se emuló (de quién)
               "estado": "provisional", "usos": 0, "nacida": int(self._emision_seq),
               "ultimo_uso": int(self._emision_seq)}
        self._voces.append(voz)                           # entra en MI vocabulario como hipótesis aprendida
        self.ultima_voz_origen = "aprendida"; self.ultimo_costo_voz = precio
        self._costo_pendiente += precio
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
        """Devuelve (y resetea) el coste energético acumulado desde la última lectura.
        El bucle lo inyecta en metabolismo como met_costo_extra (gasto real del paso).
        Incluye: emitir voz (COSTO_USAR·escala) y acuñar/emular (costo_crear(), una vez —
        el precio del momento, que sube con las palabras propias que ya hay)."""
        c = self._costo_pendiente
        self._costo_pendiente = 0.0
        return c

    def registrar_costo_emision(self, vocalizando: bool, fila: dict | None = None) -> float:
        """Cobra el coste de EMITIR este paso (señal costosa). Silencio → 0.

        Anti-Shannon: la escala depende de activación endógena (arousal del estado + intensidad
        del gesto), no de la etiqueta del sample ni de un código simbólico impuesto.
        Llamar DESPUÉS de decidir silencio (expr_vocalizando / voz_emitida='-').
        El cobro se aplica en el SIGUIENTE paso vía consumir_costo → metabolismo.
        """
        if not vocalizando:
            self.ultimo_costo_voz = 0.0
            return 0.0
        fila = fila if fila is not None else (getattr(self, "_fila", None) or {})
        try:
            aro, _val = self._afecto(fila)
        except Exception:
            aro = 0.5
        try:
            inten = abs(float((self.gesto or {}).get("g_intensidad", 0.0) or 0.0))
        except Exception:
            inten = 0.0
        # 0.5 … 1.0: más activado/intenso → señal más costosa (fisiología de esfuerzo, no de significado)
        escala = 0.5 + 0.5 * min(1.0, 0.65 * float(aro) + 0.35 * min(1.0, inten))
        try:
            base = float(os.environ.get("ANIMA_COSTO_VOZ_USAR", str(self.COSTO_USAR)))
        except Exception:
            base = float(self.COSTO_USAR)
        costo = max(0.0, base) * escala
        self._costo_pendiente += costo
        self.ultimo_costo_voz = costo
        return costo

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
