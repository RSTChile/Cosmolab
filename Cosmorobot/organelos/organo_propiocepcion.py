#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
organelos.organo_propiocepcion — el robot SE SIENTE a sí mismo (bienestar).

PROCEDENCIA: adaptado de
    Célula_Madre/organelos/VST_OrganoPropiocepcion.py
Aclaración (Alexis, 2026-07-09): este organelo NO usa GPS ni ningún hardware
específico de otro cuerpo. Calcula el bienestar (W) a partir de campos
GENÉRICOS del estado interno (`fila`), cada uno con default 0 si falta —
por diseño funciona con lo que CosmoRobot realmente tiene, sin cambios.

En v1, CosmoRobot solo puede alimentar de forma honesta: `e_R` (error de
seguimiento del objetivo de distancia) y `costo_trabajo`/`fatiga` si se
implementan. El resto (energia, LF, H, OI, Lambda_Cos, IRDE...) quedan en 0
hasta que existan organelos que los calculen — el bienestar saldrá bajo pero
CORRECTO (una célula mínima aún no es organismo pleno; ver VST_Genoma.salud()).
No es necesario tocar este archivo para eso: es la MISMA honestidad que ya
tenía en ANIMA.
"""
from __future__ import annotations


def _g(fila: dict, *claves: str, default: float = 0.0) -> float:
    for k in claves:
        if k in fila and fila[k] is not None:
            try:
                return float(fila[k])
            except (TypeError, ValueError):
                pass
    return default


def _c01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


class OrganoPropiocepcion:
    """Siente la SUMA de los estados del organismo -> bienestar W."""

    def __init__(self, ema_base: float = 0.02) -> None:
        self.ema_base = float(ema_base)
        self.W_base: float | None = None
        self.ultimo: dict = {}

    def observar(self, fila: dict) -> dict:
        energia = _c01(_g(fila, "met_energia", "energia"))
        A = _c01(_g(fila, "A_sys_env"))
        LF = _c01(_g(fila, "LF_op", "LF"))
        H = _c01(_g(fila, "H_homeostasis", "H"))
        OI = _c01(_g(fila, "OI"))
        Lam = _c01(_g(fila, "Lambda_Cos") / 4.0)

        nec = _c01(_g(fila, "necesidad", "met_necesidad"))
        eR = _c01(_g(fila, "e_R") / 10.0)
        IRDE = _c01(_g(fila, "IRDE"))
        fat = _c01(_g(fila, "act_fatiga", "fatiga_activa"))
        reflejo = _c01(_g(fila, "mem_reflejo"))
        desac = _c01(_g(fila, "presion_desacople"))

        vigor = (energia + LF) / 2.0
        acople = (A + H) / 2.0
        sostienen = 0.85 * (energia + A + H + LF) / 4.0 + 0.15 * (OI + Lam) / 2.0
        malestar = (nec + eR + IRDE + fat + reflejo + desac) / 6.0
        placer = _g(fila, "placer_sensorial")
        W = _c01(0.5 + 0.7 * (sostienen - malestar) + 0.32 * placer)

        if self.W_base is None:
            self.W_base = W
        dW = W - self.W_base
        self.W_base += self.ema_base * (W - self.W_base)

        out = {"prop_bienestar": round(W, 4), "prop_vigor": round(_c01(vigor), 4),
               "prop_acople": round(_c01(acople), 4), "prop_malestar": round(_c01(malestar), 4),
               "prop_dW": round(dW, 5)}
        fila.update(out)
        self.ultimo = out
        return out

    def estado(self) -> dict:
        return dict(self.ultimo)
