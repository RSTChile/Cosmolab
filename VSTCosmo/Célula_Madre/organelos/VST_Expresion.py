#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_EXPRESION — Órgano de EXPRESIÓN ORGANÍSMICA (rediseño del balbuceo, ANIMA-5)
================================================================================
DIAGNÓSTICO QUE LO MOTIVA
  El balbuceo anterior exploraba ≈225 gestos pero con covariación ~0 con el resto del organismo
  (η²<0.05 con orientación/atención/arousal/necesidad/OI…). Era un subsistema AISLADO: exploraba
  sonidos, NO expresaba al organismo.

QUÉ HAGO (y qué NO)
  Transformo la vocalización en una CONSECUENCIA EMERGENTE del estado GLOBAL del organismo y su historia,
  preservando exploración, plasticidad, aprendizaje, memoria, variabilidad y libertad. NO construyo lenguaje.

PRINCIPIOS (irrenunciables) — cada decisión los respeta:
  · SIN DETERMINISMO: nunca "si X → emite Y". No hay tablas, ni códigos vocales, ni asociaciones fijas.
  · SIN SEMÁNTICA: los gestos no significan hambre/peligro/etc.; sólo codifican una HISTORIA organísmica.
  · EL ORGANISMO COMPLETO DECIDE: el estado global (atención, orientación, arousal, valencia, necesidad,
    OI, H, expectativa, alteridad, energía, fatiga, MUNDO percibido, historia reciente) sesga — sin
    variable privilegiada y sin mapeo diseñado.
  · SELECCIÓN PROBABILÍSTICA: cada estado induce una DISTRIBUCIÓN sobre emisiones, nunca una respuesta fija.
    Estados parecidos → repertorios parecidos, jamás idénticos.
  · UNIDAD = CONDUCTA VOCAL: una SECUENCIA construida paso a paso; tras cada emisión se reconsulta al
    organismo y se decide (probabilístico) terminar o continuar. Sin frases predefinidas, sin longitud máxima.
  · COSTO: continuar consume recurso REAL (energía·(1−fatiga)); fatigado → termina antes; estable → más largo.
  · MEMORIA de EXPERIENCIAS (no de sonidos): (estado global aprox → conducta → consecuencia). La RECURRENCIA
    y la CONSECUENCIA (valor ecológico/contingencia, ya medidos por otros órganos) sólo SESGAN la probabilidad.
  · OLVIDO: toda asociación se degrada si deja de reutilizarse. Estabilidad = historia efectiva, no protección.
  · EL SILENCIO ES UNA CONDUCTA (no la ausencia de una): el PRIMER acto no es elegir QUÉ emitir, sino decidir
    probabilísticamente SI iniciar una conducta vocal. Silencio y voz COMPITEN bajo los mismos principios; el
    silencio se almacena en memoria, se refuerza y se olvida como cualquier conducta. Precedente: V176–V181
    (negación/afirmación operativa) — el "No" no era una salida aparte, sino una OPCIÓN que competía en la
    deliberación con valencia aprendida y elección probabilística. El silencio es la "negación operativa" del
    habla, parte constitutiva de la Libertad Funcional.
  El significado, si aparece, será emergente de la interacción organismo↔compañero↔mundo. Nunca del programador.
================================================================================
"""
from __future__ import annotations
import os, math, hashlib
from collections import defaultdict
import numpy as np

COLS_EXPR = [
    "expr_vocalizando", "expr_en_conducta", "expr_long_conducta", "expr_recurso",
    "expr_novedad", "expr_p_voz", "expr_familiaridad", "expr_consecuencia", "expr_estado_key",
    "expr_silencio", "expr_long_silencio", "expr_peso_silencio",
]


def _num(v, d=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else d
    except Exception:
        return d


def _seed(s):
    return int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16)


class OrganoExpresion:
    """La voz como manifestación probabilística, histórica y evolutiva del organismo COMPLETO (+ mundo).
    Genera el gesto acústico (g_*) y decide la conducta vocal (cuándo/cuánto vocalizar). Simétrico A/B."""

    # Variables que componen el ESTADO GLOBAL (ninguna privilegiada). El mundo entra por energia_L/R.
    _VARS = ["act_atencion_L", "act_atencion_R", "act_orientacion_deg", "act_confianza", "act_fatiga",
             "voz_arousal", "voz_valence", "necesidad", "OI", "H_homeostasis", "expectativa",
             "alt_intencion_comunicativa", "alt_otro_presente", "met_energia", "energia_L", "energia_R"]

    def __init__(self, organismo_id: str, explora: float = 0.12, lr_olvido: float = 0.0006,
                 baseline_voz: float = 0.05, baseline_sil: float = 1.0, k_imit: float = 0.12):
        self.id = str(organismo_id)
        self.explora = float(explora)           # magnitud de la exploración (constitutiva, nunca desaparece)
        self.k_imit = float(k_imit)             # fuerza del sesgo de IMITACIÓN (tira hacia lo oído; con variación)
        self.lr_olvido = float(lr_olvido)        # degradación lenta de asociaciones no reutilizadas
        self.baseline_voz = float(baseline_voz)  # tendencia constitutiva a VOCALIZAR (sin historia)
        self.baseline_sil = float(baseline_sil)  # tendencia constitutiva al SILENCIO (conducta por defecto, válida)
        self._rng = np.random.RandomState(_seed(organismo_id))
        self.gesto = np.zeros(4, dtype=float)    # gesto acústico continuo [freq, int, pausa, repeticion]
        # MEMORIA por región de estado: clave → {gesto_bucket: peso} (qué GESTO usar DENTRO de una conducta vocal)
        self.memoria = defaultdict(lambda: defaultdict(float))
        # COMPETENCIA de CONDUCTAS por región: clave → {"VOZ": peso, "SILENCIO": peso} — reforzada UNA vez por
        # conducta (no por gesto), para que silencio y voz compitan en pie de igualdad (el "primer acto").
        self.conducta_w = defaultdict(lambda: defaultdict(float))
        self.voc_freq = defaultdict(lambda: [0.0, 0.0])   # clave → [veces que vocalizó, veces visto]
        self.estado_media = None                 # EMA del estado global = "reposo" del organismo (emergente)
        self.act_media = None                    # EMA de la activación → "saliencia" relativa al propio típico
        # conducta VOCAL en curso
        self.en_conducta = False
        self.long_conducta = 0
        self.recurso = 0.0
        self._gestos_conducta = []
        self._key0 = None
        # conducta de SILENCIO en curso (el "No a hablar" como conducta operativa, no como ausencia)
        self.en_silencio = False
        self.long_silencio = 0
        self._key_sil0 = None
        self.eventos = []

    # ----------------------------------------------------------------- estado global
    def _estado_global(self, fila):
        # ACOTA cada variable con tanh (algunas, p.ej. act_fatiga, no están en 0..1): así el estado global
        # queda en un rango comparable sin asumir escalas ni privilegiar ninguna variable.
        v = []
        for k in self._VARS:
            x = _num(fila.get(k))
            if k == "act_orientacion_deg":
                x = x / 90.0
            v.append(math.tanh(x))
        return np.array(v, dtype=float)

    @staticmethod
    def _key(G):
        # discretización GRUESA del estado global → una REGIÓN (no asume las configuraciones posibles)
        return tuple(int(round(float(x) * 2)) for x in G)

    @staticmethod
    def _bucket(g):
        return "g%+d%+d%+d%+d" % tuple(int(round(float(x) * 2)) for x in g[:4])

    @staticmethod
    def _desbucket(b):
        import re
        nums = [int(x) for x in re.findall(r"[+-]\d+", str(b))]
        return (np.array(nums[:4], dtype=float) / 2.0) if len(nums) >= 4 else np.zeros(4)

    # ----------------------------------------------------------------- selección del gesto
    def _muestrear_gesto(self, key, bias_imit=None):
        """Exploración (random walk, reservorio) + SESGO por la distribución histórica de esta región +
        IMITACIÓN (atractor suave hacia lo recientemente OÍDO, vía la memoria ecoica del OAO). novedad = poca
        historia → explora; mucha → reutiliza (probabilístico, nunca idéntico). La imitación nunca copia."""
        dist = self.memoria.get(key)
        total = sum(dist.values()) if dist else 0.0
        novedad = 1.0 / (1.0 + total)            # emerge de la historia: no es un umbral fijo
        if dist and self._rng.random() > novedad:
            buckets = list(dist.keys()); pesos = np.array([dist[b] for b in buckets], dtype=float)
            pesos = pesos / pesos.sum()
            elegido = buckets[int(self._rng.choice(len(buckets), p=pesos))]
            centro = self._desbucket(elegido)
            gesto = centro + 0.5 * self.explora * self._rng.randn(4)   # jitter: parecido, jamás idéntico
        else:
            gesto = self.gesto + self.explora * self._rng.randn(4)     # explorar
        if bias_imit is not None:                # IMITACIÓN: tira hacia lo oído (suave), preservando variación
            gesto = gesto + self.k_imit * (np.asarray(bias_imit, dtype=float) - gesto)
        gesto = np.clip(gesto, -1.0, 1.0); gesto[2] = abs(gesto[2]); gesto[3] = abs(gesto[3])
        return gesto, novedad

    def _olvidar(self):
        # toda asociación (gestos Y conductas voz/silencio) se degrada lentamente; lo no reutilizado se desvanece
        for mapa in (self.memoria, self.conducta_w):
            for key, dist in list(mapa.items()):
                for b in list(dist.keys()):
                    dist[b] *= (1.0 - self.lr_olvido)
                    if dist[b] < 1e-4:
                        del dist[b]
                if not dist:
                    del mapa[key]

    @staticmethod
    def _consecuencia(fila):
        # consecuencia COMÚN a voz y silencio: valor ecológico + contingencia (ya medidos por otros órganos).
        # Sólo SESGA el peso (recurrencia + consecuencia); nunca impone una conducta.
        return max(0.0, _num(fila.get("voz_otro_valor_ecologico")) + _num(fila.get("alt_contingencia_social")))

    def _cerrar_conducta(self, fila):
        """Registra la conducta VOCAL: refuerza qué GESTOS se usaron (memoria) Y la conducta 'VOZ' de la región
        (UNA vez, para la competencia con el silencio). Recurrencia + consecuencia sólo sesgan."""
        consec = self._consecuencia(fila); peso = 1.0 + 6.0 * consec
        for b in self._gestos_conducta:
            self.memoria[self._key0][b] += peso
        self.conducta_w[self._key0]["VOZ"] += peso              # competencia de conductas: una vez por conducta
        if consec > 0.01:
            self.eventos.append(("expresion_consolida", f"conducta vocal útil (región {hash(self._key0) % 1000})",
                                 {"consec": round(consec, 4), "long": self.long_conducta}))
        return consec

    def _cerrar_silencio(self, fila):
        """Registra la conducta de SILENCIO como cualquier otra: región → 'SILENCIO', reforzada por recurrencia
        y consecuencia. Así el silencio compite, se almacena, aprende y se olvida en pie de igualdad con la voz."""
        consec = self._consecuencia(fila); peso = 1.0 + 6.0 * consec
        self.conducta_w[self._key_sil0]["SILENCIO"] += peso     # competencia de conductas: una vez por silencio
        return consec

    # ----------------------------------------------------------------- ciclo
    def proximo_gesto(self, fila: dict, bias_imit=None) -> dict:
        G = self._estado_global(fila)
        self.estado_media = G if self.estado_media is None else (0.99 * self.estado_media + 0.01 * G)
        activacion = float(np.linalg.norm(G - self.estado_media))   # cuán cambia el estado ahora
        self.act_media = activacion if self.act_media is None else (0.99 * self.act_media + 0.01 * activacion)
        exceso = max(0.0, activacion - self.act_media)
        saliencia = exceso / (self.act_media + exceso + 1e-6)       # 0..1: ~0 en estado TÍPICO, alto en INUSUAL
        key = self._key(G)
        consec_out = 0.0; p_voz = 0.0

        # (a) si una CONDUCTA VOCAL está en curso → continuar emitiendo gestos (decidir continuar/terminar)
        if self.en_conducta:
            gesto, novedad = self._muestrear_gesto(key, bias_imit)
            self.gesto = gesto; b = self._bucket(gesto)
            self._gestos_conducta.append(b); self.long_conducta += 1
            self.recurso -= 0.15                                   # costo modesto; la longitud emerge del recurso
            p_cont = 1.0 / (1.0 + math.exp(-8.0 * self.recurso))
            if self._rng.random() > p_cont:
                consec_out = self._cerrar_conducta(fila); self.en_conducta = False
            self._olvidar()
            return self._salida(b, key, novedad, 1.0, True, consec_out)

        # (b) PRIMER ACTO (no hay conducta vocal): DELIBERAR silencio vs voz. Compiten por sus pesos históricos
        #     (memoria de la región) + baseline constitutivo de cada uno; la activación del estado inclina —no
        #     impone— hacia expresar. El silencio es una conducta operativa (negación del habla), no una ausencia.
        cw = self.conducta_w.get(key, {})
        w_voz = self.baseline_voz + cw.get("VOZ", 0.0)
        w_sil = self.baseline_sil + cw.get("SILENCIO", 0.0)
        p_voz = w_voz / (w_voz + w_sil)
        p_voz = p_voz + (1.0 - p_voz) * min(0.6, saliencia)         # un estado SALIENTE inclina a expresar (no impone)
        self.voc_freq[key][1] += 1.0

        if self._rng.random() < p_voz:
            # INICIAR conducta VOCAL (si veníamos de un silencio, ciérralo y regístralo como conducta)
            if self.en_silencio:
                self._cerrar_silencio(fila); self.en_silencio = False; self.long_silencio = 0
            self.en_conducta = True; self.long_conducta = 1; self._gestos_conducta = []; self._key0 = key
            self.recurso = 0.15 + max(0.0, min(1.0, _num(fila.get("met_energia"))))
            self.voc_freq[key][0] += 1.0
            gesto, novedad = self._muestrear_gesto(key, bias_imit)
            self.gesto = gesto; b = self._bucket(gesto); self._gestos_conducta.append(b)
            self.recurso -= 0.15
            if self._rng.random() > 1.0 / (1.0 + math.exp(-8.0 * self.recurso)):
                consec_out = self._cerrar_conducta(fila); self.en_conducta = False
            self._olvidar()
            return self._salida(b, key, novedad, p_voz, True, consec_out)
        else:
            # CONDUCTA DE SILENCIO: se sostiene paso a paso (su longitud también emerge); se registra al terminar
            if not self.en_silencio:
                self.en_silencio = True; self.long_silencio = 0; self._key_sil0 = key
            self.long_silencio += 1
            self.gesto *= 0.92; self._olvidar()
            return self._salida("-", key, 0.0, p_voz, False, consec_out)

    def _salida(self, bucket, key, novedad, p_voz, vocaliza, consec):
        g = self.gesto
        fam = sum(self.memoria.get(key, {}).values()) if key in self.memoria else 0.0
        peso_sil = self.conducta_w.get(key, {}).get("SILENCIO", 0.0)
        return {
            "g_freq": round(float(g[0]), 3), "g_intensidad": round(float(g[1]), 3),
            "g_pausa": round(float(g[2]), 3), "g_repeticion": round(float(g[3]), 3),
            "g_bucket": bucket,
            "expr_vocalizando": 1.0 if vocaliza else 0.0,
            "expr_en_conducta": 1.0 if self.en_conducta else 0.0,
            "expr_long_conducta": float(self.long_conducta),
            "expr_recurso": round(float(self.recurso), 4),
            "expr_novedad": round(float(novedad), 4),
            "expr_p_voz": round(float(p_voz), 4),
            "expr_familiaridad": round(float(fam), 3),
            "expr_consecuencia": round(float(consec), 4),
            "expr_estado_key": (hash(key) % 100000),
            "expr_silencio": 1.0 if self.en_silencio else 0.0,
            "expr_long_silencio": float(self.long_silencio),
            "expr_peso_silencio": round(float(peso_sil), 3),
        }
