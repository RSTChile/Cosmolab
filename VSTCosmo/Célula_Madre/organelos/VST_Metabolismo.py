#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_Metabolismo — OrganeloMetabolismo: la economía energética del organismo
================================================================================
El órgano que faltaba (veredicto 24-jun-2026): con memoria y necesidad ya vivas, la carencia
se desplazó al METABOLISMO. La necesidad crónica saturada (Cb 8→70) era su huella: un organismo
con hambre y memoria pero SIN cómo comer, digerir ni saciarse.

RE-TRANSCRIBE capacidades PROBADAS y perdidas (no inventa):
  · v034 metabolismo de experiencias — Índice Metabólico: NUTRITIVA / TÓXICA / NEUTRA.
  · v035 preferencia alimentaria      — preferencia aprendida por impacto (qué me alimenta).
  · v069 saciedad diferencial         — saciedad ESPECÍFICA por tipo de alimento (no comer siempre lo mismo).

LAS 4 FASES (las que Alexis nombró):
  · CONSUMO     — el organismo COME su experiencia presente: si es nutritiva, REPONE energía.
  · COSTO       — actuar/abrir membrana/computar GASTA energía (se debita de una reserva FINITA).
  · DEGRADACIÓN — metabolismo basal: la energía se fuga sola con el tiempo (no hay descanso gratis).
  · REPOSICIÓN  — comer experiencias nutritivas repone; las tóxicas dañan (cuestan extra).

QUÉ ES "ALIMENTO": la experiencia (audio→campo) se digiere en SENTIDO. Nutritiva = ICR convierte
ruido en orden mientras el acople (A) se sostiene; Tóxica = IRDE/riesgo domina y desacopla.
ABSORCIÓN gateada por A (estar acoplado = poder digerir). SACIEDAD diferencial frena comer lo mismo.

CONEXIÓN con [[memoria-organelo-ausente]]/[[act-perm-organo-conativo]]: secreta `met_nutricion`
(calidad de cada bocado) — la memoria la lee para SACIAR la necesidad. Así el lazo
necesidad→comer→saciedad por fin puede CERRARSE (a nivel diagnóstico; sin tocar el soma).

ANTI-SHANNON: NO hay setpoint de energía ni "comer para que E=X". E EMERGE de ingesta−gasto.
Constantes FISIOLÓGICAS declaradas (basal, tasas de ingesta/saciedad). BIOENERGÉTICA: reserva
finita, no se gasta lo no usado, costo basal real (sin alimento, el organismo decae).
================================================================================
"""
from __future__ import annotations
import math


def _c01(v, d=0.0):
    try:
        x = float(v)
    except Exception:
        return d
    return 0.0 if x != x else max(0.0, min(1.0, x))


def _num(v, d=0.0):
    try:
        x = float(v)
        return x if x == x else d
    except Exception:
        return d


COLS_MET = ["met_energia", "met_IM", "met_clase", "met_ingesta", "met_gasto", "met_balance",
            "met_hambre", "met_saciedad", "met_preferencia", "met_nutricion"]

CLASE = {"toxica": -1, "neutra": 0, "nutritiva": 1}


class OrganeloMetabolismo:
    """Economía energética: come la experiencia, paga el costo de vivir, se degrada y se repone.
    Constantes FISIOLÓGICAS declaradas (no para forzar un nivel de energía)."""

    def __init__(self,
                 E0: float = 0.6,            # energía inicial (reserva ∈ [0,1])
                 basal: float = 0.003,       # DEGRADACIÓN: metabolismo basal por paso (vivir cuesta)
                 k_ingesta: float = 0.30,    # cuánto repone CONVERTIR energía semiótica (ICES); calibrado a ES
                 k_trabajo: float = 0.008,   # COSTO de actuar/abrir membrana (act_perm) — < intake de buena comida
                 k_toxico: float = 0.04,     # daño extra de comer algo tóxico
                 umbral_nut: float = 0.10,   # IM por encima → nutritiva
                 umbral_tox: float = -0.10,  # IM por debajo → tóxica
                 tau_saciedad: float = 20.0, # cuánto dura la saciedad específica (v069)
                 base_saciedad: float = 0.03,# tasa base de saciarse al comer un tipo (variar > repetir)
                 sac_max: float = 0.6,       # la saciedad REDUCE el rendimiento, no lo ANULA (mono-dieta sostiene maintenance)
                 ema_IM: float = 0.1,        # digestión: se metaboliza el IM SOSTENIDO, no el instantáneo (mata transitorios)
                 ema_pref: float = 0.02) -> None:
        self.E0 = E0; self.basal = basal; self.k_ingesta = k_ingesta; self.k_trabajo = k_trabajo
        self.k_toxico = k_toxico; self.umbral_nut = umbral_nut; self.umbral_tox = umbral_tox
        self.tau_saciedad = tau_saciedad; self.base_saciedad = base_saciedad; self.ema_pref = ema_pref
        self.sac_max = sac_max; self.ema_IM = ema_IM
        self.es_ref = 0.10            # ES de referencia (RC_total con sonido ~0.26; silencio ~0.0008)
        self.reset()

    def reset(self) -> None:
        self.E = self.E0
        self.saciedad = {}      # saciedad específica por tipo de alimento (clave)  — v069
        self.preferencia = {}   # preferencia aprendida por tipo                    — v035
        self._IM_ema = 0.0      # IM SOSTENIDO (digestión: filtra transitorios del campo)
        self._ES_ema = 0.0      # energía semiótica sostenida (RC_total suavizado)

    # ----- PERSISTENCIA: las reservas y lo aprendido sobreviven al apagón (incremento 1) -----
    def snapshot(self) -> dict:
        """Estado metabólico a guardar: reservas de energía (E), saciedades específicas (v069) y
        preferencias aprendidas por alimento (v035). Las constantes (basal, k_*) vienen del genoma."""
        return {"E": self.E, "saciedad": self.saciedad, "preferencia": self.preferencia,
                "_IM_ema": self._IM_ema, "_ES_ema": self._ES_ema}

    def restore(self, d: dict) -> None:
        if not d:
            return
        self.E = float(d.get("E", self.E0))
        self.saciedad = d.get("saciedad", {}) or {}
        self.preferencia = d.get("preferencia", {}) or {}
        self._IM_ema = float(d.get("_IM_ema", 0.0) or 0.0)
        self._ES_ema = float(d.get("_ES_ema", 0.0) or 0.0)

    @staticmethod
    def _clave_alimento(lat: float, sabor: float):
        """Tipo de 'alimento' por su SABOR cosmosemiótico: dónde viene × dulce(ICR)/amargo(IRDE)."""
        lb = -1 if lat < -0.15 else (1 if lat > 0.15 else 0)
        sb = 1 if sabor > 0.15 else (-1 if sabor < -0.15 else 0)   # dulce(orden) / amargo(riesgo) / soso
        return (lb, sb)

    def actualizar(self, fila: dict, dt: float = 0.1) -> dict:
        icr_r = _c01(fila.get("ICR_ratio")); irde_r = _c01(fila.get("IRDE_ratio"))
        act_perm = _c01(fila.get("act_perm"))
        lat = _num(fila.get("lateralidad"))
        # ENERGÍA SEMIÓTICA en acto (RC ≡ ES, C-N2/NE23.1): RC_total = ES = ICES + IDES. En silencio ~0:
        # no hay nada que convertir ni comer. La absorción se rige por ESTO (estar EN ACTO), NO por A (calma).
        ES = max(0.0, _num(fila.get("RC_total")))
        self._ES_ema = (1.0 - self.ema_IM) * self._ES_ema + self.ema_IM * ES      # ES sostenida (digestión)
        es_norm = min(1.0, self._ES_ema / max(1e-6, self.es_ref))                 # ¿hay energía semiótica disponible?

        # ── ÍNDICE METABÓLICO (v034): ¿esta experiencia ALIMENTA o INTOXICA? ──────────────────
        # nutritiva = ICR convierte en sentido; tóxica = IRDE/riesgo domina. IM ∈ [−1,1].
        # DIGESTIÓN: se metaboliza el IM SOSTENIDO (EMA), no el instantáneo — un transitorio del campo no
        # es comida ni veneno; sólo lo que PERSISTE alimenta o intoxica. (También limpia la señal en movimiento.)
        IM_inst = icr_r - irde_r
        self._IM_ema = (1.0 - self.ema_IM) * self._IM_ema + self.ema_IM * IM_inst
        IM = self._IM_ema
        clase = ("nutritiva" if IM > self.umbral_nut else "toxica" if IM < self.umbral_tox else "neutra")
        clave = self._clave_alimento(lat, IM)

        # ── SACIEDAD diferencial (v069): específica por tipo; decae sola ─────────────────────
        d_sac = math.exp(-dt / self.tau_saciedad)
        for k in self.saciedad:
            self.saciedad[k] *= d_sac
        sac = self.saciedad.get(clave, 0.0)

        # ── CONSUMO = CONVERSIÓN DE ENERGÍA SEMIÓTICA EN SENTIDO (ICES) ──────────────────────
        # nutrición = conversión NETA en acto: calidad(IM = ICES−IDES) × energía semiótica presente (es_norm)
        # × hambre-de-eso. COMER es estar EN ACTO convirtiendo (enérgeia), NO estar calmo (por eso ya NO va A).
        # La saciedad reduce el rendimiento pero no lo anula (sac_max): se puede vivir de un alimento.
        sac_ef = min(self.sac_max, sac)
        nutricion = max(0.0, IM) * es_norm * (1.0 - sac_ef)
        ingesta = self.k_ingesta * nutricion
        if ingesta > 0:    # comer ESTE tipo sacia más de ESTE tipo (no de otros) — saciedad específica
            tasa = self.base_saciedad * (1.0 + (1.0 - icr_r))   # lo soso/predecible sacia más rápido (v069)
            self.saciedad[clave] = min(1.0, sac + tasa * (1.0 - sac))

        # ── COSTO + DEGRADACIÓN: vivir y actuar gastan; DISIPAR energía semiótica (IDES) daña ──
        toxicidad = self.k_toxico * max(0.0, -IM) * es_norm  # disipación de la energía presente (silencio no intoxica)
        gasto = self.basal + self.k_trabajo * act_perm + toxicidad
        balance = ingesta - gasto
        self.E = max(0.0, min(1.0, self.E + balance))
        hambre = 1.0 - self.E

        # ── PREFERENCIA aprendida (v035): qué tipo me ha alimentado mejor ────────────────────
        for k in self.preferencia:
            self.preferencia[k] *= (1.0 - 0.2 * self.ema_pref)   # olvido competitivo suave
        pref = self.preferencia.get(clave, 0.0)
        pref += self.ema_pref * (IM - pref)
        self.preferencia[clave] = max(-1.0, min(1.0, pref))

        return {
            "met_energia": round(self.E, 4), "met_IM": round(IM, 4), "met_clase": CLASE[clase],
            "met_ingesta": round(ingesta, 5), "met_gasto": round(gasto, 5), "met_balance": round(balance, 5),
            "met_hambre": round(hambre, 4), "met_saciedad": round(self.saciedad.get(clave, 0.0), 4),
            "met_preferencia": round(self.preferencia[clave], 4), "met_nutricion": round(nutricion, 4),
        }


if __name__ == "__main__":
    met = OrganeloMetabolismo()
    nutr = {"A_sys_env": 0.8, "ICR_ratio": 0.8, "IRDE_ratio": 0.2, "act_perm": 0.3, "lateralidad": 0.0}
    tox = {"A_sys_env": 0.3, "ICR_ratio": 0.2, "IRDE_ratio": 0.8, "act_perm": 0.6, "lateralidad": 0.0}
    for _ in range(60):
        rn = met.actualizar(nutr)
    print("tras comer NUTRITIVO:", rn)
    for _ in range(60):
        rt = met.actualizar(tox)
    print("tras comer TÓXICO:   ", rt)
