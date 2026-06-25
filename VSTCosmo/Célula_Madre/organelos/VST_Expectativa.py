#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_EXPECTATIVA — Nodo de EXPECTATIVA (NO lenguaje, NO decisión, NO predictor perfecto)
================================================================================
QUIÉN SOY
  Diagnóstico previo: la voz del otro se oye (presencia ✅) pero NO afecta al otro (agencia ❌) ni ayuda
  a mi persistencia (valor ecológico ❌). Conectar voz→persistencia es demasiado directo (riesga Shannon).
  La biología no obtiene beneficio de una SEÑAL, sino de la CONDUCTA que realiza tras interpretarla:

      voz  →  EXPECTATIVA  →  exploración  →  resultado  →  persistencia

  Mi ÚNICA función es aprender, por historia de consecuencias:
      "después de esta firma acústica del otro, ¿las EXPLORACIONES posteriores tendieron a mejorar o
       empeorar mi situación?"
  No aprendo objetos ni etiquetas. Aprendo SÓLO capacidad predictiva. NO hay diccionario, ni reward
  externo, ni significado. Mi única SALIDA es muy leve: "vale un poco más la pena seguir observando"
  (disposición a explorar / segunda observación). Nunca oriento, nunca decido, nunca toco
  metabolismo/energía/RC/persistencia directamente.

  Falsable: bajo NULL no aparece; bajo SHUFFLED fuerte no se consolida; en REAL sólo si hay contingencia
  histórica. Hipótesis genealógica: expectativa ANTES que agencia, agencia ANTES que intención, intención
  ANTES que convención, convención ANTES que lenguaje.

  FRASE GUÍA: la voz del otro todavía no debe cambiar mi vida; primero debe enseñarme que, históricamente,
  después de escuchar ciertos patrones, vale la pena seguir explorando.
================================================================================
"""
from __future__ import annotations
import os, math
from collections import deque

COLS_EXP = [
    "expectativa", "expectativa_confianza", "expectativa_error", "expectativa_historia",
    "expectativa_utilidad", "expectativa_exploracion", "expectativa_confirmaciones", "expectativa_falsaciones",
]


def _num(v, d=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else d
    except Exception:
        return d


class OrganeloExpectativa:
    """Aprende si EXPLORAR tras una firma acústica del otro tiende a mejorar la situación. Salida única y
    leve: disposición a seguir observando. Falsable (NULL/SHUFFLED). NO significado, NO control. A/B simétrico."""

    def __init__(self, organismo_id: str, ventana: float = 1.5, lr: float = 0.05, ema: float = 0.04,
                 umbral_voz: float = 0.02, k_expl: float = 0.5, cap_expl: float = 0.20, thr: float = 0.005):
        self.id = str(organismo_id)
        self.ventana = float(ventana)        # s tras la firma para ver si la exploración mejoró la situación
        self.lr = float(lr)                  # aprendizaje LENTO de la expectativa
        self.ema = float(ema)
        self.umbral_voz = float(umbral_voz)
        self.k_expl = float(k_expl)          # cuánto la expectativa empuja la exploración
        self.cap_expl = float(cap_expl)      # tope de la salida (muy leve)
        self.thr = float(thr)                # umbral de "mejora real"
        self.expect = {}                     # firma -> expectativa (exceso PROMEDIO post−pre de la situación)
        self.modelo = {}                     # firma -> mejora POST media (EMA) = utilidad esperada
        self.modelo_pre = {}                 # firma -> mejora PRE media (línea-base; rectificar el PROMEDIO)
        self.n_firma = {}                    # firma -> conteo (novedad/repetición)
        self.confianza = 0.0
        self.error_ema = 0.0
        self.historia = 0.0                  # beneficio histórico atribuido a explorar tras la voz
        self.confirmaciones = 0.0            # veces que explorar tras la voz mejoró
        self.falsaciones = 0.0               # veces que NO mejoró
        self.exploracion = 0.0               # salida actual (disposición a seguir observando)
        self.pendientes = deque(maxlen=4000)
        self.sit_hist = deque(maxlen=512)    # (t, situacion) para la línea-base pre-firma
        self._e_hist = deque(maxlen=24)
        self._firma_prev = "·"
        self.eventos = []

    @staticmethod
    def _situacion(fila):
        """Situación del organismo (no etiqueta): ICES/ICR + acople − necesidad + homeostasis."""
        ICR = _num(fila.get("ICR", fila.get("RC_total")))
        A = _num(fila.get("A_sys_env"))
        nec = _num(fila.get("necesidad", fila.get("necesidad_efectiva")))
        H = _num(fila.get("H_homeostasis", fila.get("H_real", fila.get("H"))))
        return 0.5 * ICR + A - nec + 0.5 * H

    def _firma(self, e):
        if e <= self.umbral_voz:
            return "·"
        self._e_hist.append(e)
        if len(self._e_hist) >= 3:
            m = sum(self._e_hist) / len(self._e_hist)
            var = sum((x - m) ** 2 for x in self._e_hist) / len(self._e_hist)
            estable = 0 if var > (0.02 ** 2) else 1
        else:
            estable = 1
        return "e%d_%d" % (int(round(min(1.0, e) * 4)), estable)

    def observar(self, fila: dict, energia_voz_otro: float, dt: float = 0.1) -> dict:
        t = _num(fila.get("t"))
        e = max(0.0, _num(energia_voz_otro))
        sit = self._situacion(fila)
        self.sit_hist.append((t, sit))

        firma = self._firma(e)
        evento = (firma != "·") and (firma != self._firma_prev)
        if evento:
            sit_pre = next((s for (tt, s) in self.sit_hist if t - tt <= self.ventana), sit)
            mejora_pre = sit - sit_pre        # cuánto mejoraba YA mi situación antes de oír la firma
            self.pendientes.append({"t": t, "firma": firma, "sit0": sit, "mejora_pre": mejora_pre})
            self.n_firma[firma] = self.n_firma.get(firma, 0) + 1
            self.eventos.append(("expectativa_firma", f"firma {firma}", {"e": round(e, 4)}))
        self._firma_prev = firma

        # cuando pasa la ventana de EXPLORACIÓN: ¿mejoró la situación POR ENCIMA de mi línea-base?
        while self.pendientes and (t - self.pendientes[0]["t"]) >= self.ventana:
            ev = self.pendientes.popleft()
            f = ev["firma"]
            mejora_post = sit - ev["sit0"]
            mejora_cont = mejora_post - ev["mejora_pre"]
            pred = self.modelo.get(f, 0.0)
            self.error_ema = (1 - self.ema) * self.error_ema + self.ema * abs(mejora_post - pred)
            self.modelo[f] = (1 - self.lr) * pred + self.lr * mejora_post
            self.modelo_pre[f] = (1 - self.lr) * self.modelo_pre.get(f, 0.0) + self.lr * ev["mejora_pre"]
            # EXPECTATIVA = exceso del PROMEDIO post sobre el PROMEDIO pre (rectifica el PROMEDIO, no cada
            # evento → bajo SHUFFLED post≈pre y la expectativa NO converge; el ruido coincidente no se acumula).
            self.expect[f] = max(0.0, self.modelo[f] - self.modelo_pre[f])
            util = self.expect[f] > self.thr
            if mejora_cont > self.thr:
                self.confirmaciones += 1.0
                if util:
                    self.historia += mejora_cont
                self.eventos.append(("expectativa_confirma", f"{f}: explorar tras la voz mejoró", {"exp": round(self.expect[f], 4)}))
            else:
                self.falsaciones += 1.0
            self.confianza = (1 - self.ema) * self.confianza + self.ema * (1.0 if util else 0.0)

        # SALIDA ÚNICA Y LEVE: "vale un poco más la pena seguir observando" (disposición a explorar).
        exp_act = self.expect.get(firma, 0.0) if firma != "·" else 0.0
        self.exploracion = max(0.0, min(self.cap_expl, self.k_expl * exp_act))

        return {
            "expectativa": round(exp_act, 4),
            "expectativa_confianza": round(self.confianza, 4),
            "expectativa_error": round(self.error_ema, 4),
            "expectativa_historia": round(self.historia, 4),
            "expectativa_utilidad": round(self.modelo.get(firma, 0.0), 4),
            "expectativa_exploracion": round(self.exploracion, 4),
            "expectativa_confirmaciones": round(self.confirmaciones, 1),
            "expectativa_falsaciones": round(self.falsaciones, 1),
        }

    def expect_max(self):
        return max(self.expect.values()) if self.expect else 0.0
