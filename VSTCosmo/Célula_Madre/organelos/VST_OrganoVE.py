#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_OrganoVE — ÓRGANO DE VALORACIÓN EXPERIENCIAL (OVE)
================================================================================
QUIÉN SOY
  No clasifico sonidos. No reconozco notas. No asigno significados. No tengo
  tablas ni etiquetas humanas. Respondo continuamente a UNA pregunta:
      ¿qué efecto produjo esta experiencia sobre mi organismo?
  No memorizo el estímulo: memorizo la TRANSFORMACIÓN organísmica que produjo.
  Soy una MEMORIA DE TRANSFORMACIONES. La valoración es una propiedad emergente
  de esa memoria, no el único dato que guardo (conservo el Δvector completo).

RESTRICCIÓN CONSTITUTIVA (cláusula fundacional, no negociable)
  El Órgano de Valoración Experiencial NO posee autoridad ejecutiva sobre el
  organismo. Ninguna valoración, preferencia, comparación histórica o
  recomendación podrá transformarse automáticamente en conducta. El OVE
  únicamente conserva y organiza la historia de las transformaciones
  experimentadas por el organismo. La decisión final pertenece siempre a la
  Libertad Funcional (LF), que podrá seguir, ignorar, invertir o explorar contra
  toda la evidencia acumulada. El organismo debe conservar en todo momento la
  capacidad de decir "No", incluso frente a experiencias históricamente muy
  favorables. La existencia de preferencias históricas nunca podrá eliminar la
  posibilidad de explorar lo improbable, lo desconocido o lo aparentemente
  desfavorable.
  → Estructuralmente: este órgano es READ-ONLY sobre el organismo. Lee la fila y
    escribe ÚNICAMENTE sus columnas ove_*. Jamás toca una variable de decisión,
    conducta ni LF. No existe camino automático valoración→conducta.

QUÉ SIGNIFICA EL ESCALAR DE VALORACIÓN
  El escalar de valoración no representa el valor verdadero de una experiencia.
  Representa únicamente la mejor hipótesis histórica que el organismo posee, en
  ese momento de su vida, acerca de cuánto contribuyó esa experiencia a su
  continuidad y reorganización. Esa hipótesis nunca es definitiva y podrá
  modificarse progresivamente conforme aumente la historia experiencial del
  organismo.

CRITERIO (A₁ — persistencia constitutiva, evolutiva)
  El objetivo constitutivo del organismo es seguir siendo un organismo. El
  escalar inicial se forma de: reorganización lograda + persistencia/estabilidad
  + libertad funcional preservada − coste, TODO normalizado contra su PROPIA
  historia (nunca valores absolutos). NO representa "gusto": representa capacidad
  histórica de seguir existiendo y reorganizándose. Los PESOS de esas dimensiones
  DERIVAN lentamente con la historia (lo que acompañó a su continuidad pesa más),
  salvo la LF, que tiene piso (es constitutiva). Una experiencia que persiste
  PERDIENDO libertad nunca recibe valoración máxima.
================================================================================
"""
from __future__ import annotations
import math
import hashlib
import numpy as np

# ════════════════════════════════════════════════════════════════════════════
# COLUMNAS OBSERVABLES (todo el proceso queda observable en el CSV)
# ════════════════════════════════════════════════════════════════════════════
COLS_OVE = [
    "ove_experiencias",      # nº de experiencias cerradas (histórico)
    "ove_valoracion_actual", # valoración de la última experiencia cerrada (hipótesis histórica)
    "ove_confianza",         # confianza en esa valoración (densidad de vecinos + duración)
    "ove_novedad",           # distancia al vecino más cercano del paisaje (alto = novedoso)
    "ove_reorganizacion",    # ‖Δvector‖ de la experiencia en curso / última
    "ove_coste",             # coste (energía consumida) de la última experiencia
    "ove_persistencia",      # estabilidad posterior alcanzada (1=muy estable)
    "ove_radio",             # radio del vecindario usado para comparar
    "ove_region",            # id de región emergente del paisaje (clúster por semejanza)
    "ove_memoria",           # tamaño de la memoria experiencial (registros guardados)
    "ove_comparaciones",     # nº de comparaciones históricas realizadas
    "ove_preferencia",       # valoración histórica media del vecindario (SOLO observable)
    # --- BOCA EVALUATIVA (indicador visual etológico, SIN influencia causal) ---
    "cara_valoracion",       # -1 desfavorable · 0 neutra · +1 favorable
    "cara_modo",             # "neutra" | "anticipada" | "final"
    "cara_confianza",        # confianza de la inferencia visual
    "cara_t_restante",       # s restantes manteniendo la expresión final antes de volver a neutra
    "cara_error_prediccion", # anticipada − final (corrección histórica por la experiencia actual)
]

# Dimensiones del snapshot de estado global (representativas, no exhaustivas).
DIMS = ["Omega", "OI", "A_sys_env", "LF_op", "R2", "gradiente", "presion_desacople",
        "H_homeostasis", "XE", "C_m", "necesidad", "met_energia", "voz_arousal", "voz_valence"]

MAX_REGISTROS = 1200   # tope de registros en disco (las estadísticas vivas retienen toda la historia)
MAX_DUR = 180.0        # tope: un sonido continuo de hasta 3 min es UNA experiencia (el silencio corta antes)
MIN_PASOS = 4          # mínimo de pasos para que una experiencia cuente
DEBOUNCE_PASOS = 3     # pasos sostenidos de cruce sonido↔silencio antes de cortar (anti-microbache)
# --- Boca evaluativa (indicador visual; NUNCA influye en el organismo) ---
MIN_MEM = 25           # experiencias mínimas antes de ANTICIPAR (recién nacido = neutra)
MIN_VECINOS = 5        # vecinos mínimos en la región para confiar en la anticipación
CONF_MIN = 0.35        # confianza mínima para mover la boca anticipadamente
HOLD_S = 4.0           # s que se mantiene la expresión FINAL antes de volver a neutra


def _num(v, d=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else d
    except Exception:
        return d


def _seed(s):
    return int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16)


class OrganoValoracionExperiencial:
    """Memoria de transformaciones. Read-only sobre el organismo: sólo observa y registra."""

    def __init__(self, organismo_id: str):
        self.id = str(organismo_id)
        self._rng = np.random.RandomState(_seed(organismo_id))
        nd = len(DIMS)
        # Estadística viva de los Δ por dimensión (para z-score contra la propia historia).
        self._mu = np.zeros(nd, dtype=np.float64)
        self._var = np.ones(nd, dtype=np.float64)   # var inicial 1 → z bien definido desde el inicio
        self._n_delta = 0
        # PESOS del prior constitutivo (evolutivos). LF con piso (no se des-pesa).
        self._pesos = {"reorg": 1.0, "persist": 1.0, "lf": 1.0, "coste": 1.0}
        self._lf_piso = 0.6
        # Paisaje experiencial (registros) + regiones emergentes.
        self.experiencias = []     # cada uno: dict con Δ y metadatos
        self._regiones = []        # centroides emergentes (en espacio Δ z-scoreado)
        self._comparaciones = 0
        self._exp_cerradas = 0     # nº de experiencias cerradas (histórico, persiste)
        # estadística viva de la valoración (para discretizar la boca contra su propia historia)
        self._val_mu = 0.0
        self._val_var = 0.04
        # BOCA EVALUATIVA: indicador visual, SIN influencia causal sobre el organismo.
        self._cara_hold = 0.0      # s restantes de la expresión final
        self._cara_final = 0       # cara final mantenida (-1/0/+1)
        self._cara_final_conf = 0.0
        self._anticipada = 0       # última cara anticipada (para el error de predicción)
        self._cara_err = 0.0
        # Experiencia EN CURSO + segmentación auto-referencial.
        self._exp = None
        self._pres_smooth = 0.0    # presencia suavizada (filtra microcortes dentro de un mismo sonido)
        self._pres_peak = 0.05     # pico reciente de presencia (auto-escala el umbral de silencio)
        self._en_sonido = False    # ¿la señal está por ENCIMA del umbral de silencio?
        self._cambio_cnt = 0       # pasos sostenidos de cruce sonido↔silencio (debounce)
        # Cache de salida (las cols se sostienen entre cierres de experiencia).
        self._out = {c: 0.0 for c in COLS_OVE}
        self.eventos = []

    # ---- transiente (NO la memoria/pesos, que persisten) ----
    def reset(self) -> None:
        self._exp = None
        self._pres_smooth = 0.0
        self._pres_peak = 0.05
        self._en_sonido = False
        self._cambio_cnt = 0
        self._cara_hold = 0.0       # la boca nace neutra (transitorio)
        self._cara_final = 0
        self._anticipada = 0
        self.eventos.clear()

    # ---- persistencia (biografía permanente) ----
    def snapshot(self) -> dict:
        return {
            "mu": self._mu.tolist(), "var": self._var.tolist(), "n_delta": self._n_delta,
            "pesos": self._pesos,
            "experiencias": self.experiencias[-MAX_REGISTROS:],
            "regiones": [r.tolist() for r in self._regiones],
            "comparaciones": self._comparaciones,
            "exp_cerradas": self._exp_cerradas,
            "val_mu": self._val_mu, "val_var": self._val_var,
        }

    def restore(self, d: dict) -> None:
        if not d:
            return
        try:
            self._mu = np.array(d.get("mu", self._mu), dtype=np.float64)
            self._var = np.array(d.get("var", self._var), dtype=np.float64)
            self._n_delta = int(d.get("n_delta", 0))
            self._pesos.update(d.get("pesos", {}) or {})
            self.experiencias = list(d.get("experiencias", []) or [])
            self._regiones = [np.array(r, dtype=np.float64) for r in (d.get("regiones", []) or [])]
            self._comparaciones = int(d.get("comparaciones", 0))
            self._exp_cerradas = int(d.get("exp_cerradas", 0))
            self._val_mu = float(d.get("val_mu", 0.0)); self._val_var = float(d.get("val_var", 0.04))
        except Exception:
            pass

    # ---- utilidades internas ----
    def _vector(self, fila: dict) -> np.ndarray:
        return np.array([_num(fila.get(k)) for k in DIMS], dtype=np.float64)

    def _z(self, delta: np.ndarray) -> np.ndarray:
        return (delta - self._mu) / np.sqrt(np.maximum(self._var, 1e-6))

    def _actualiza_stats(self, delta: np.ndarray) -> None:
        # EMA lenta de media/var de los Δ → normalización contra la propia historia.
        a = 0.02
        self._mu = (1 - a) * self._mu + a * delta
        self._var = (1 - a) * self._var + a * (delta - self._mu) ** 2
        self._n_delta += 1

    # ---- boca evaluativa (indicador visual; NUNCA influye en el organismo) ----
    def _disc(self, v) -> int:
        """Discretiza una valoración a -1/0/+1 contra la PROPIA historia de valoraciones
        (banda auto-referencial, no umbral absoluto)."""
        s = math.sqrt(max(self._val_var, 1e-6))
        if v > self._val_mu + 0.5 * s:
            return 1
        if v < self._val_mu - 0.5 * s:
            return -1
        return 0

    def _anticipar(self, dz):
        """Predicción visual desde la semejanza de la experiencia EN CURSO con el paisaje.
        Sólo si hay memoria y confianza suficientes; si no, neutra. NO modifica el paisaje."""
        if self._exp_cerradas < MIN_MEM or not self._regiones:
            return 0, 0.0
        dists = [float(np.linalg.norm(dz - r)) for r in self._regiones]
        j = int(np.argmin(dists)); nov = dists[j]
        radio = float(np.median(dists)) if len(dists) > 1 else nov
        if nov > radio:                     # región novedosa → sin anticipación
            return 0, 0.0
        vecinos = [r["valoracion"] for r in self.experiencias[-300:] if r.get("region") == j]
        if len(vecinos) < MIN_VECINOS:
            return 0, 0.0
        conf = float(min(1.0, len(vecinos) / 12.0))
        if conf < CONF_MIN:
            return 0, 0.0
        return self._disc(float(np.mean(vecinos))), conf

    # ════════════════════════════════════════════════════════════════════════
    # PASO: LEE la fila, segmenta la experiencia, y al cerrarla la valora/guarda.
    # ════════════════════════════════════════════════════════════════════════
    def observar(self, fila: dict, dt: float = 0.1) -> dict:
        t = _num(fila.get("t"))
        v = self._vector(fila)
        presencia = max(_num(fila.get("energia_L")), _num(fila.get("energia_R")))
        oi = _num(fila.get("OI"))

        # --- SEGMENTACIÓN por SILENCIO: una experiencia = un sonido COMPLETO (de inicio a fin) o un
        # silencio completo. El corte ocurre cuando la señal CRUZA el umbral de silencio (cae a ~0).
        # Un TEMA = una experiencia (todo el tema); una NOTA = una experiencia. No se mide por pedazos.
        self._pres_smooth = 0.6 * self._pres_smooth + 0.4 * presencia    # filtra microcortes internos
        self._pres_peak = max(presencia, 0.995 * self._pres_peak)        # pico reciente (decae lento)
        floor = max(0.02, 0.12 * self._pres_peak)                        # "silencio" = caída a ~0 (auto-escala)
        sonido_ahora = self._pres_smooth > floor
        self._cambio_cnt = 0 if (sonido_ahora == self._en_sonido) else (self._cambio_cnt + 1)

        if self._exp is None:
            self._abrir(t, v); self._en_sonido = sonido_ahora; self._cambio_cnt = 0
        else:
            e = self._exp
            e["pasos"] += 1
            e["jitter"] += abs(oi - e["_oi_prev"])
            e["_oi_prev"] = oi
            e["_coste"] += max(0.0, e["_met_prev"] - _num(fila.get("met_energia")))
            e["_met_prev"] = _num(fila.get("met_energia"))
            dur = t - e["t_inicio"]
            cruce = self._cambio_cnt >= DEBOUNCE_PASOS and e["pasos"] >= MIN_PASOS  # cruce sonido↔silencio SOSTENIDO
            if cruce or dur >= MAX_DUR:
                self._cerrar(v, t, fila)
                self._abrir(t, v); self._en_sonido = sonido_ahora; self._cambio_cnt = 0

        # reorganización EN CURSO (parcial), para que se vea en vivo
        if self._exp is not None:
            self._out["ove_reorganizacion"] = round(float(np.linalg.norm(self._z(v - self._exp["snap_ini"]))), 4)

        # --- BOCA EVALUATIVA: indicador visual, CERO influencia causal sobre el organismo ---
        self._cara_hold = max(0.0, self._cara_hold - dt)
        if self._cara_hold > 0.0:                          # FINAL tiene prioridad mientras dura
            cara, modo, conf = self._cara_final, "final", self._cara_final_conf
        elif self._exp is not None:                        # ANTICIPADA (sólo con memoria+confianza)
            c, conf = self._anticipar(self._z(v - self._exp["snap_ini"]))
            self._anticipada = c
            cara, modo = c, ("anticipada" if c != 0 else "neutra")
        else:
            cara, modo, conf = 0, "neutra", 0.0
        self._out["cara_valoracion"] = float(cara)
        self._out["cara_modo"] = modo
        self._out["cara_confianza"] = round(conf, 3)
        self._out["cara_t_restante"] = round(self._cara_hold, 2)
        self._out["cara_error_prediccion"] = round(self._cara_err, 4)

        self._out["ove_memoria"] = float(len(self.experiencias))
        self._out["ove_experiencias"] = float(self._exp_cerradas)
        self._out["ove_comparaciones"] = float(self._comparaciones)
        return dict(self._out)

    def _abrir(self, t, v):
        self._exp = {"t_inicio": t, "snap_ini": v.copy(), "pasos": 0, "jitter": 0.0,
                     "_oi_prev": float(v[DIMS.index("OI")]), "_coste": 0.0,
                     "_met_prev": float(v[DIMS.index("met_energia")])}

    def _cerrar(self, v_fin, t, fila):
        e = self._exp
        pasos = max(1, e["pasos"])
        delta = v_fin - e["snap_ini"]
        self._actualiza_stats(delta)
        dz = self._z(delta)

        # --- componentes (z-scoreados contra la propia historia) ---
        reorg = float(np.linalg.norm(dz))
        i_lf = DIMS.index("LF_op"); i_oi = DIMS.index("OI")
        libertad = float(dz[i_lf])                                  # ΔLF normalizado
        estabilidad = 1.0 / (1.0 + e["jitter"] / pasos)            # 1 = muy estable después
        coste_z = float((e["_coste"] - 0.0))                        # energía consumida (cruda; se escala abajo)
        coste = math.tanh(coste_z * 3.0)

        # --- valoración = hipótesis histórica (prior constitutivo, pesos evolutivos) ---
        p = self._pesos
        val = (p["reorg"] * math.tanh(0.5 * (reorg + dz[i_oi]))     # reorganización + integración
               + p["persist"] * (2 * estabilidad - 1.0)
               + max(p["lf"], self._lf_piso) * math.tanh(libertad)
               - p["coste"] * coste)
        # NORMALIZA a ~[-1,1] por la suma de pesos activos
        wsum = p["reorg"] + p["persist"] + max(p["lf"], self._lf_piso) + p["coste"]
        val = val / max(wsum, 1e-6)
        # GATE de LIBERTAD: persistir PERDIENDO libertad nunca recibe valoración máxima.
        gate = float(np.clip(0.4 + 0.6 * (1.0 / (1.0 + math.exp(-3.0 * libertad))), 0.4, 1.0))
        val = val * gate

        # --- paisaje: semejanza con experiencias pasadas (SOLO observación) ---
        novedad, region, pref, radio, conf = self._comparar(dz, val)

        # --- BOCA: estadística de valoración + expresión FINAL (prioritaria, se mantiene HOLD_S) ---
        av = 0.05
        self._val_mu = (1 - av) * self._val_mu + av * val
        self._val_var = (1 - av) * self._val_var + av * (val - self._val_mu) ** 2
        cara_fin = self._disc(val)
        self._cara_err = float(self._anticipada - cara_fin)   # corrección de la predicción
        self._cara_final = cara_fin
        self._cara_final_conf = conf
        self._cara_hold = HOLD_S

        registro = {
            "id": int(self._exp_cerradas), "t_inicio": round(e["t_inicio"], 2),
            "duracion": round(t - e["t_inicio"], 2), "pasos": pasos,
            "snap_ini": [round(x, 4) for x in e["snap_ini"].tolist()],
            "snap_fin": [round(x, 4) for x in v_fin.tolist()],
            "delta": [round(x, 4) for x in delta.tolist()],
            "valoracion": round(val, 4), "confianza": round(conf, 3),
            "reorganizacion": round(reorg, 4), "coste": round(coste, 4),
            "persistencia": round(estabilidad, 4), "libertad": round(libertad, 4),
            "region": region, "voz_ctx": fila.get("voz_emitida", "-"),
        }
        self.experiencias.append(registro)
        if len(self.experiencias) > MAX_REGISTROS:
            self.experiencias = self.experiencias[-MAX_REGISTROS:]
        self._exp_cerradas += 1
        self._adaptar_pesos(dz, estabilidad)

        # salida observable
        self._out.update({
            "ove_valoracion_actual": round(val, 4), "ove_confianza": round(conf, 3),
            "ove_novedad": round(novedad, 4), "ove_coste": round(coste, 4),
            "ove_persistencia": round(estabilidad, 4), "ove_radio": round(radio, 4),
            "ove_region": float(region), "ove_preferencia": round(pref, 4),
        })

    def _comparar(self, dz, val):
        """Distancia al paisaje → novedad, región, preferencia (media histórica del vecindario)."""
        if not self._regiones:
            self._regiones.append(dz.copy())
            return 1.0, 0, val, 1.0, 0.2
        dists = [float(np.linalg.norm(dz - r)) for r in self._regiones]
        j = int(np.argmin(dists)); novedad = dists[j]
        self._comparaciones += 1
        # radio adaptativo = mediana de distancias entre regiones (auto-escala del paisaje)
        radio = float(np.median(dists)) if len(dists) > 1 else max(novedad, 0.5)
        if novedad > radio:                       # región nueva emergente
            region = len(self._regiones); self._regiones.append(dz.copy())
        else:                                      # se asimila a la región más cercana
            region = j; self._regiones[j] = 0.98 * self._regiones[j] + 0.02 * dz
        # preferencia = valoración histórica media de las experiencias de regiones vecinas
        vecinos = [r["valoracion"] for r in self.experiencias[-300:] if r.get("region") == region]
        pref = float(np.mean(vecinos)) if vecinos else val
        conf = float(min(1.0, len(vecinos) / 12.0))   # más vecinos → más confianza
        return novedad, region, pref, radio, conf

    def _adaptar_pesos(self, dz, estabilidad):
        """Los pesos DERIVAN muy lentamente hacia lo que acompañó la continuidad (estabilidad).
        La LF nunca baja de su piso (constitutiva). v1: nudge conservador y documentado."""
        eta = 0.003
        senal = (2 * estabilidad - 1.0)            # continuidad alcanzada en esta experiencia
        i_oi = DIMS.index("OI"); i_lf = DIMS.index("LF_op")
        self._pesos["reorg"]   = max(0.2, self._pesos["reorg"]   + eta * senal * abs(float(dz[i_oi])))
        self._pesos["persist"] = max(0.2, self._pesos["persist"] + eta * senal)
        self._pesos["lf"]      = max(self._lf_piso, self._pesos["lf"] + eta * senal * abs(float(dz[i_lf])))
        # renormaliza para que no exploten
        s = sum(self._pesos.values()) / 4.0
        for k in self._pesos:
            self._pesos[k] = self._pesos[k] / max(s, 1e-6)


if __name__ == "__main__":
    # SMOKE TEST: simula pasos con régimen cambiante; debe cerrar experiencias y valorar.
    import random
    ove = OrganoValoracionExperiencial("TEST")
    random.seed(0)
    base = {"Omega": 0.9, "OI": 0.3, "A_sys_env": 0.3, "LF_op": 0.15, "R2": 0.8,
            "gradiente": 0.4, "presion_desacople": 170.0, "H_homeostasis": 0.0,
            "XE": 0.9, "C_m": 1.0, "necesidad": 0.2, "met_energia": 0.8,
            "voz_arousal": 0.5, "voz_valence": 0.1, "energia_L": 0.0, "energia_R": 0.0,
            "voz_emitida": "-", "t": 0.0}
    for i in range(400):
        f = dict(base); f["t"] = i * 0.1
        # régimen de presencia que cambia cada ~80 pasos
        fase = (i // 80) % 3
        pres = [0.0, 0.25, 0.12][fase] + random.uniform(-0.02, 0.02)
        f["energia_L"] = max(0.0, pres); f["OI"] = 0.3 + 0.1 * fase
        f["LF_op"] = 0.15 - 0.02 * fase; f["met_energia"] = 0.8 - 0.001 * i
        out = ove.observar(f, dt=0.1)
    print("experiencias cerradas:", out["ove_experiencias"], "| memoria:", out["ove_memoria"])
    print("última valoración:", out["ove_valoracion_actual"], "| preferencia:", out["ove_preferencia"],
          "| novedad:", out["ove_novedad"], "| region:", out["ove_region"])
    print("pesos evolutivos:", {k: round(v, 3) for k, v in ove._pesos.items()})
    print("snapshot keys:", list(ove.snapshot().keys()))
