#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_APRENDIZAJE — Organelo de Aprendizaje Organísmico (OAO) [ANIMA-5]
================================================================================
QUÉ HAGO (y qué NO)
  Hoy el organismo escucha, recuerda y vocaliza, pero esas capacidades están DESACOPLADAS: la voz no
  recuerda lo escuchado. Yo NO decido ni controlo otros órganos: transformo la HISTORIA vivida en
  modificaciones LENTAS, PROBABILÍSTICAS y RECURSIVAS del comportamiento futuro. NO reemplazo la memoria,
  la percepción ni la expresión: LAS CONECTO. El aprendizaje no almacena respuestas; reorganiza
  progresivamente las PROBABILIDADES futuras. No aprende reglas, símbolos ni significados. Aprende historia.

CÓMO (todo emerge de la historia; sin tablas, sin diccionarios, sin mapeos sonido→significado):
  · MEMORIA ECOICA: conserva, por un tiempo LIMITADO (decae), una REPRESENTACIÓN INTERNA de lo oído — NO
    audio ni WAV: el vector de estructura (freq/intensidad/pausa/repetición) que el organismo percibe de la
    voz del otro / del mundo, sólo cuando de verdad OYE algo (energía de entrada sobre el mínimo perceptual).
  · IMITACIÓN = consecuencia EMERGENTE: nunca por copia. Lo oído modifica el PAISAJE PROBABILÍSTICO desde el
    cual luego emergen vocalizaciones (un atractor suave hacia lo escuchado). La semejanza surge de la
    historia, jamás de una instrucción. La VARIACIÓN se preserva (exploración constitutiva + jitter).
  · LIBERTAD FUNCIONAL: aprender es una CONDUCTA PROBABILÍSTICA, no una obligación. Siempre existe la
    posibilidad de NO incorporar una experiencia. La historia también incluye lo que el organismo NO aprende.
  · OLVIDO: lo no reutilizado se degrada. No hay recuerdos permanentes por diseño.
  El órgano sólo APORTA un sesgo de imitación a la expresión; la conducta vocal sigue emergiendo en el
  OrganoExpresion (estado→conducta) con silencio compitiendo en pie de igualdad.
================================================================================
"""
from __future__ import annotations
import os, math, hashlib
from collections import deque
import numpy as np

COLS_OAO = [
    "oao_oido", "oao_aprendio", "oao_echoica_n", "oao_imitacion_mag",
    "oao_eco_freq", "oao_eco_intensidad", "oao_eco_pausa", "oao_eco_repeticion",
]


def _num(v, d=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else d
    except Exception:
        return d


def _seed(s):
    return int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16)


class OrganeloAprendizajeOrganismico:
    """Conecta percepción↔memoria↔expresión: memoria ecoica de lo oído → sesgo de IMITACIÓN sobre el paisaje
    vocal futuro. Aprender es probabilístico (Libertad Funcional). Olvida. NO copia, NO impone. Simétrico A/B."""

    def __init__(self, organismo_id: str, vida_eco: float = 2.5, p_aprender: float = 0.7,
                 umbral_oido: float = 0.02):
        self.id = str(organismo_id)
        self.vida_eco = float(vida_eco)        # persistencia (s) de la memoria ecoica (decae con la antigüedad)
        self.p_aprender = float(p_aprender)    # probabilidad de INCORPORAR lo oído (Libertad Funcional: puede NO)
        self.umbral_oido = float(umbral_oido)  # energía de entrada mínima para contar como "oír algo"
        self._rng = np.random.RandomState(_seed(organismo_id))
        self.echoica = deque(maxlen=512)       # [t, gesto_vector(4), peso] — representación de lo oído, decae
        self.eventos = []

    @staticmethod
    def _resumen_par(par):
        """La estructura del gesto del OTRO tal como se percibe (la voz que llega = lo que el par produjo)."""
        par = par or {}
        f = par.get("fila") if isinstance(par.get("fila"), dict) else par
        return np.array([_num(f.get("g_freq")), _num(f.get("g_intensidad")),
                         _num(f.get("g_pausa")), _num(f.get("g_repeticion"))], dtype=float)

    def observar(self, fila: dict, par_estado: dict | None) -> dict:
        t = _num(fila.get("t"))
        e_oida = max(_num(fila.get("energia_L")), _num(fila.get("energia_R")))   # ¿se oye algo?
        aprendio = 0.0
        if e_oida > self.umbral_oido and par_estado is not None:
            oido = self._resumen_par(par_estado)
            # LIBERTAD FUNCIONAL: incorporar lo oído es probabilístico (a veces el organismo NO aprende)
            if self._rng.random() < self.p_aprender:
                self.echoica.append([t, oido, 1.0]); aprendio = 1.0
        # OLVIDO: la memoria ecoica decae con la antigüedad y se purga lo muy viejo
        for ent in self.echoica:
            ent[2] = math.exp(-(t - ent[0]) / max(0.1, self.vida_eco))
        while self.echoica and (t - self.echoica[0][0]) > 6.0 * self.vida_eco:
            self.echoica.popleft()
        bias = self.bias_imitacion()
        m = float(np.linalg.norm(bias)) if bias is not None else 0.0
        return {
            "oao_oido": round(e_oida, 4),
            "oao_aprendio": aprendio,
            "oao_echoica_n": float(len(self.echoica)),
            "oao_imitacion_mag": round(m, 4),
            "oao_eco_freq": round(float(bias[0]), 3) if bias is not None else 0.0,
            "oao_eco_intensidad": round(float(bias[1]), 3) if bias is not None else 0.0,
            "oao_eco_pausa": round(float(bias[2]), 3) if bias is not None else 0.0,
            "oao_eco_repeticion": round(float(bias[3]), 3) if bias is not None else 0.0,
        }

    def bias_imitacion(self):
        """Atractor de imitación = (promedio ponderado de los gestos OÍDOS) × FUERZA por recencia. La dirección
        es 'hacia dónde tira lo que he estado escuchando'; la fuerza DECAE cuando se deja de oír (sin oír →
        sesgo→0, el organismo deja de imitar). None si no hay memoria ecoica."""
        if not self.echoica:
            return None
        W = sum(e[2] for e in self.echoica)
        if W < 1e-6:
            return None
        avg = sum(e[1] * e[2] for e in self.echoica) / W
        fuerza = W / (W + 4.0)                    # mucho peso reciente → imita; memoria decaída → deja de imitar
        return avg * fuerza
