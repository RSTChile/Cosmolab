#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quién soy / qué hago (código autodescriptivo):
  Módulo de referencia compartido para el ARREGLO 3 de la batería ENFOQUE 5
  (ver INSTRUCCION_ARREGLOS_antes_de_seguir_PARA_CC.md, sección "ARREGLO 3").

  Problema diagnosticado: cada uno de los ~30 experimentos de Enfoque 5 se diseñó
  por separado (regla "si la base común no está, define la tuya"), y terminaron
  usando fórmulas ligeramente distintas para "energía útil" (exergía X), "energía
  degradada" (E) y "desorden" (entropía S_ent). Consecuencia: las curvas de un
  experimento NO son comparables con las de otro, aunque midan el mismo campo phi.

  Fuente de la definición homologada (por instrucción explícita del director —
  "la del experimento 3 (E5.2-2), que es la más limpia y la que ya pasó bien"):

      X(t)     = (1/N) * sum_i (phi_i(t) - 1)^2
                 [exergía: desviación cuadrática media respecto al equilibrio
                  uniforme phi_eq=1, normalizada por N -> intensiva]

      S_ent(t) = -sum_i p_i(t) * ln(p_i(t)),   p_i(t) = phi_i(t)^2 / sum_j phi_j(t)^2
                 [entropía de Shannon espacial de la densidad phi^2; convención:
                  si el campo es idénticamente nulo, S_ent = ln(N) (uniforme)]

      Verbatim de E5_2_2_motor.py (exergia_X, entropia_S) — PASS casi perfecto
      (r≈-0.9999 a -1.0000, 44/44 celdas centrales), ya pre-registrado y corrido.

  Cierre de energía E(t) (no pedido explícitamente por E5.2-2, que no la usa, pero
  necesario para experimentos que auditan conservación / degradación): se adopta
  la definición ya usada y verificada por E5.5-1 (PASS en sus tres curvas, deriva
  de E1 medida ~eps^2), que es la extensión mínima consistente con la misma
  normalización de phi^2 que ya usa S_ent:

      E(t) = sum_i phi_i(t)^2     [energía total declarada, SIN normalizar por N
                                    -- es la misma suma que aparece sin dividir
                                    dentro de S_ent, antes de normalizar a p_i]

  Energía degradada (magnitud complementaria a X dentro del mismo presupuesto E,
  para experimentos que necesiten descomponer E = útil + degradada; no estaba
  definida por ningún hermano previo de forma explícita -- se deriva aquí, no se
  inventa un número nuevo, por diferencia respecto al total ya definido arriba):

      E_degradada(t) = E(t) - N * X(t)
                      = sum_i phi_i(t)^2 - sum_i (phi_i(t) - 1)^2
                      = sum_i [2*phi_i(t) - 1]

      (aparece solo por completitud algebraica; los experimentos existentes NO
      la usaron, así que no hay a qué re-homologar -- queda declarada por si
      algún pendiente de los 17 la necesita citar de un solo lugar).

  Uso: los experimentos que necesiten re-expresar sus resultados bajo la
  definición común, o que arranquen desde cero entre los 17 pendientes, deben
  importar estas tres funciones de aquí en vez de reimplementar su propia
  fórmula. No se edita `cs074_rcruz.py` (regla del proyecto) — este es un
  módulo nuevo, aditivo, y tampoco se editan los motores ya corridos (E5.2-2,
  E5.5-1): sus definiciones YA COINCIDEN con este módulo por construcción
  (es de donde se copiaron), así que no requieren re-cálculo.

  Re-expresión de los 13 ya corridos SIN volver a correrlos: solo es posible si
  el JSON crudo de un experimento guardó la trayectoria phi(t) completa (no solo
  los observables ya agregados). A la fecha de este módulo, la mayoría de los
  motores de Enfoque 5 NO persisten phi(t) crudo (guardan X(t)/S(t) ya calculados
  o solo resúmenes finales) -- verificar caso por caso antes de asumir que la
  re-expresión es posible sin recómputo.
"""
from __future__ import annotations

import numpy as np


def exergia_X(phi: np.ndarray) -> float:
    """X(t) = (1/N) * sum_i (phi_i(t) - 1)^2 -- exergía, intensiva, ref fija phi_eq=1.
    Verbatim de E5_2_2_motor.py::exergia_X (definición canónica, Arreglo 3)."""
    return float(np.mean((phi.astype(np.float64) - 1.0) ** 2))


def entropia_S(phi: np.ndarray) -> float:
    """S_ent(t) = -sum_i p_i ln p_i, p_i = phi_i(t)^2 / sum_j phi_j(t)^2 -- entropía de
    Shannon espacial de la densidad phi^2. Verbatim de E5_2_2_motor.py::entropia_S
    (definición canónica, Arreglo 3)."""
    e = phi.astype(np.float64) ** 2
    total = e.sum()
    if total <= 0:
        return float(np.log(phi.size))
    p = e / total
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def energia_E(phi: np.ndarray) -> float:
    """E(t) = sum_i phi_i(t)^2 -- energía total declarada, SIN normalizar por N.
    Verbatim de E5_5_1_motor.py::energia_E (cierre consistente con S_ent, Arreglo 3)."""
    return float(np.sum(phi.astype(np.float64) ** 2))


def energia_degradada(phi: np.ndarray) -> float:
    """E_degradada(t) = E(t) - N*X(t) -- complemento algebraico de X dentro del
    presupuesto E ya definido arriba (derivado aquí, no reimplementado en ningún
    motor previo -- ver docstring del módulo)."""
    N = phi.size
    return float(energia_E(phi) - N * exergia_X(phi))


def medir(phi: np.ndarray) -> dict:
    """Atajo: calcula las cuatro cantidades homologadas de una sola pasada por phi."""
    return {
        "E": energia_E(phi),
        "X": exergia_X(phi),
        "S_ent": entropia_S(phi),
        "E_degradada": energia_degradada(phi),
    }
