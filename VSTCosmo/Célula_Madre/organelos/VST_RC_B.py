#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_RC_B - E018 endogeno
==========================

Organelo observacional para O-N1:

    RC = ICR + IRDE

Esta version elimina las mezclas ponderadas manuales. Las magnitudes se forman por
simetria, normalizacion relativa y competencia entre bases endogenas:

  - RC_relacional / RC_externo nacen de union probabilistica de evidencias del canal.
  - ICR_ratio / IRDE_ratio nacen de competencia conversion-vulnerabilidad.
  - atencion / comprension / riesgo usan medias simetricas y pesos relativos del propio RC.
"""
from __future__ import annotations

import math
import threading


COLS_RC = [
    "RC_total", "RC_externo", "RC_relacional",
    "ICR", "IRDE", "ICR_ratio", "IRDE_ratio",
    "RC_delta_salud", "destino_RC",
    "RC_atencion_L", "RC_atencion_R",
    "RC_comprension_L", "RC_comprension_R",
    "RC_riesgo_L", "RC_riesgo_R",
    "RC_consenso_orientacion", "RC_confianza_comprension", "RC_freno_riesgo",
    "RC_base_relacional", "RC_base_externo",
    "RC_soporte_conversion", "RC_vulnerabilidad_desviacion",
    "RC_base_ICR", "RC_base_IRDE", "RC_peso_ICR", "RC_peso_IRDE",
]


def _num(x, default=0.0):
    try:
        v = float(x)
    except Exception:
        return float(default)
    return v if math.isfinite(v) else float(default)


def _clip01(x):
    return max(0.0, min(1.0, float(x)))


def _media(vals):
    xs = [_clip01(v) for v in vals if math.isfinite(float(v))]
    return _clip01(sum(xs) / max(1, len(xs)))


def _union(*vals):
    y = 1.0
    for v in vals:
        y *= 1.0 - _clip01(v)
    return _clip01(1.0 - y)


def _competir(a, b):
    a = max(0.0, float(a)); b = max(0.0, float(b))
    den = a + b
    if den <= 1e-12:
        return 0.0, 0.0
    return _clip01(a / den), _clip01(b / den)


class OrganoRC:
    """Metabolismo observacional de RC -> ICR/IRDE, sin ponderaciones manuales."""

    def __init__(self, ema: float = 0.20) -> None:
        self.ema = _clip01(ema)
        self._prev = None
        self._lock = threading.Lock()
        self._ema_rc = 0.0
        self._ema_icr = 0.0
        self._ema_irde = 0.0
        self._energy_ref = 1e-9

    def reset(self) -> None:
        with self._lock:
            self._prev = None
            self._ema_rc = self._ema_icr = self._ema_irde = 0.0
            self._energy_ref = 1e-9

    def _energia_relativa(self, raw_l: float, raw_r: float) -> tuple[float, float]:
        raw_l = max(0.0, _num(raw_l)); raw_r = max(0.0, _num(raw_r))
        pico = max(raw_l, raw_r, self._energy_ref)
        self._energy_ref = max(1e-9, max(pico, self._energy_ref * (1.0 - self.ema)))
        return _clip01(raw_l / (raw_l + self._energy_ref)), _clip01(raw_r / (raw_r + self._energy_ref))

    def observar(self, fila: dict, meta: dict | None = None) -> dict:
        with self._lock:
            prev = dict(self._prev) if self._prev else None

            energia_l, energia_r = self._energia_relativa(fila.get("energia_L", 0.0), fila.get("energia_R", 0.0))
            balance_signed = _num(fila.get("balance_LR", 0.0))
            balance = _clip01(abs(balance_signed))
            coher = _clip01(abs(_num(fila.get("coherencia_biaural", 0.0), 0.0)))

            w_l = _num(fila.get("omega_L", 0.0))
            w_r = _num(fila.get("omega_R", 0.0))
            w_a = _num(fila.get("omega_A", 0.0))
            sal_l = _clip01(abs(w_l - w_a)) * energia_l
            sal_r = _clip01(abs(w_r - w_a)) * energia_r

            novelty_l = novelty_r = novelty = 0.0
            if prev:
                novelty_l = _clip01(abs(w_l - _num(prev.get("omega_L", w_l)))) * energia_l
                novelty_r = _clip01(abs(w_r - _num(prev.get("omega_R", w_r)))) * energia_r
                novelty_terms = [
                    abs(_num(fila.get("Omega")) - _num(prev.get("Omega"))),
                    abs(_num(fila.get("R2")) - _num(prev.get("R2"))),
                    abs(_num(fila.get("LF_op")) - _num(prev.get("LF_op"))),
                    abs(_num(fila.get("C_m")) - _num(prev.get("C_m"))),
                    abs(_num(fila.get("XE")) - _num(prev.get("XE"))),
                    abs(_num(fila.get("H_homeostasis")) - _num(prev.get("H_homeostasis"))),
                ]
                novelty = _media(novelty_terms)

            incoher = 1.0 - coher
            base_rel = _media([energia_l, sal_l, balance * energia_l, novelty_l])
            base_ext = _media([energia_r, sal_r, incoher * energia_r, novelty_r])
            rc_rel = _union(base_rel, novelty_l)
            rc_ext = _union(base_ext, novelty_r)
            rc_total = _union(rc_rel, rc_ext, novelty)

            OI = _clip01(_num(fila.get("OI")))
            H = _clip01(_num(fila.get("H_homeostasis")))
            Aenv = _clip01(_num(fila.get("A_sys_env")))
            LF = _clip01(_num(fila.get("LF_op")))
            R2 = _clip01(_num(fila.get("R2")))
            C_m = _clip01(_num(fila.get("C_m")))
            XE = _clip01(_num(fila.get("XE")))
            e_R = _clip01(abs(_num(fila.get("e_R"))))
            Lambda = _clip01(_num(fila.get("Lambda_Cos")))

            if prev:
                mejoras = [
                    OI - _clip01(_num(prev.get("OI"))),
                    H - _clip01(_num(prev.get("H_homeostasis"))),
                    Lambda - _clip01(_num(prev.get("Lambda_Cos"))),
                    Aenv - _clip01(_num(prev.get("A_sys_env"))),
                    LF - _clip01(_num(prev.get("LF_op"))),
                    C_m - _clip01(_num(prev.get("C_m"))),
                    XE - _clip01(_num(prev.get("XE"))),
                    _clip01(abs(_num(prev.get("e_R")))) - e_R,
                ]
                conversion = sum(max(0.0, x) for x in mejoras)
                desviacion = sum(max(0.0, -x) for x in mejoras)
                peso_icr, peso_irde = _competir(conversion, desviacion)
                delta = conversion - desviacion
                if conversion + desviacion <= 1e-12:
                    soporte_conversion = _media([OI, H, Aenv, LF, R2, C_m, XE])
                    vulnerabilidad = _media([1.0 - OI, 1.0 - H, 1.0 - Aenv, 1.0 - LF, e_R, 1.0 - Lambda])
                    peso_icr, peso_irde = _competir(soporte_conversion, vulnerabilidad)
            else:
                soporte_conversion = _media([OI, H, Aenv, LF, R2, C_m, XE])
                vulnerabilidad = _media([1.0 - OI, 1.0 - H, 1.0 - Aenv, 1.0 - LF, e_R, 1.0 - Lambda])
                peso_icr, peso_irde = _competir(soporte_conversion, vulnerabilidad)
                delta = 0.0

            soporte_conversion = _media([OI, H, Aenv, LF, R2, C_m, XE])
            vulnerabilidad = _media([1.0 - OI, 1.0 - H, 1.0 - Aenv, 1.0 - LF, e_R, 1.0 - Lambda])
            # ESTRUCTURA del input (membrana sensorial): el SENTIDO nace del ORDEN. La conversión en sentido
            # (ICR=ICES) sólo RINDE si hay estructura que convertir; el ruido (estructura→0) se DISIPA (IRDE).
            estructura = _clip01(_num(fila.get("estructura", 1.0), 1.0))
            base_icr = _clip01(rc_total * soporte_conversion * peso_icr * estructura)
            base_irde = _clip01(rc_total * (vulnerabilidad * peso_irde + (1.0 - estructura) * peso_icr))
            icr_ratio, irde_ratio = _competir(base_icr, base_irde)

            icr = rc_total * icr_ratio
            irde = rc_total - icr

            at_l = _media([sal_l, rc_rel, energia_l, novelty_l])
            at_r = _media([sal_r, rc_ext, energia_r, novelty_r])
            rel_sum = rc_rel + rc_ext
            peso_l, peso_r = _competir(rc_rel, rc_ext)
            if rel_sum <= 1e-12:
                peso_l = peso_r = 0.0

            comp_l = _clip01(at_l * icr_ratio * _media([peso_l, soporte_conversion, rc_rel]))
            comp_r = _clip01(at_r * icr_ratio * _media([peso_r, soporte_conversion, rc_ext]))
            riesgo_l = _clip01(at_l * irde_ratio * _media([1.0 - peso_l, vulnerabilidad, rc_rel]))
            riesgo_r = _clip01(at_r * irde_ratio * _media([1.0 - peso_r, vulnerabilidad, rc_ext]))

            consenso = _clip01((comp_r - comp_l + 1.0) / 2.0) * 2.0 - 1.0
            confianza_comp = _clip01(icr_ratio * _media([comp_l + comp_r, soporte_conversion, rc_total]))
            freno_riesgo = _clip01(irde_ratio * _media([riesgo_l + riesgo_r, vulnerabilidad, rc_total]))

            self._ema_rc = (1.0 - self.ema) * self._ema_rc + self.ema * rc_total
            self._ema_icr = (1.0 - self.ema) * self._ema_icr + self.ema * icr
            self._ema_irde = (1.0 - self.ema) * self._ema_irde + self.ema * irde

            if rc_total <= 1e-12:
                destino = "silencio"
            elif abs(icr_ratio - irde_ratio) <= 1e-12:
                destino = "mixto"
            else:
                destino = "ICR" if icr_ratio > irde_ratio else "IRDE"

            fila.update({
                "RC_total": round(rc_total, 5),
                "RC_externo": round(rc_ext, 5),
                "RC_relacional": round(rc_rel, 5),
                "ICR": round(icr, 5),
                "IRDE": round(irde, 5),
                "ICR_ratio": round(icr_ratio, 5),
                "IRDE_ratio": round(irde_ratio, 5),
                "RC_delta_salud": round(delta, 6),
                "destino_RC": destino,
                "RC_atencion_L": round(at_l, 5),
                "RC_atencion_R": round(at_r, 5),
                "RC_comprension_L": round(comp_l, 5),
                "RC_comprension_R": round(comp_r, 5),
                "RC_riesgo_L": round(riesgo_l, 5),
                "RC_riesgo_R": round(riesgo_r, 5),
                "RC_consenso_orientacion": round(consenso, 5),
                "RC_confianza_comprension": round(confianza_comp, 5),
                "RC_freno_riesgo": round(freno_riesgo, 5),
                "RC_base_relacional": round(base_rel, 5),
                "RC_base_externo": round(base_ext, 5),
                "RC_soporte_conversion": round(soporte_conversion, 5),
                "RC_vulnerabilidad_desviacion": round(vulnerabilidad, 5),
                "RC_base_ICR": round(base_icr, 5),
                "RC_base_IRDE": round(base_irde, 5),
                "RC_peso_ICR": round(icr_ratio, 5),
                "RC_peso_IRDE": round(irde_ratio, 5),
            })
            self._prev = dict(fila)
            return fila
