#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quién soy / qué hago (código autodescriptivo):
  Módulo de referencia compartido para el ARREGLO 2 de la batería ENFOQUE 5
  (ver INSTRUCCION_ARREGLOS_antes_de_seguir_PARA_CC.md, sección "ARREGLO 2").

  Problema diagnosticado (por E5.6-3, confirmado independientemente por E5.1-1,
  E5.2-1, E5.5-2, E5.5-4): el ruido dinámico por paso (exigido por la regla T7
  de la batería, "perturbación dinámica, no solo semilla") se implementó en
  cada experimento como:

      noise_amp = NOISE_REL * eps                 # CONSTANTE, no depende de N ni de pasos
      phi = phi + noise_amp * rng.standard_normal(phi.shape)   # cada paso

  Esto es un paseo aleatorio SIN amortiguar por nodo. Su varianza acumulada
  tras `pasos` pasos es ≈ pasos · noise_amp². Como `pasos_fijo(N)` se calibra
  para "lavar" el anillo (pasos_fijo ∝ 1/D(N) ∝ N², medido y confirmado por
  E5.6-3 con K_ref≈4.47, dispersión <1.3%), la varianza acumulada del ruido
  crece como N² · noise_amp² — SIN TOPE — y termina dominando la dinámica:
  a N≥2048 el NULL deja de discriminar (z→0) y la conservación declarada (E1)
  se viola hasta 98% en una sola celda (E5.6-3). Incluso a N=200 fijo, con
  ε=1.0, la deriva de E1 llega a 37% (E5.1-1) y hasta 101% (E5.5-2) — el
  problema NO es solo de escala con N, es que el ruido acumulado sobre
  `pasos_fijo` pasos nunca se compara contra `pasos_fijo` en su propia
  fórmula.

  Arreglo (este módulo): la amplitud de ruido POR PASO se escala con
  1/sqrt(pasos_fijo), de modo que la varianza acumulada total quede
  CONSTANTE — independiente de N y de cuántos pasos corra la simulación:

      noise_amp_por_paso = NOISE_REL * eps / sqrt(pasos_fijo)

      Var_acumulada ≈ pasos_fijo · noise_amp_por_paso²
                     = pasos_fijo · (NOISE_REL·eps)² / pasos_fijo
                     = (NOISE_REL·eps)²                              [constante]

  Esto resuelve TANTO el problema de escala con N (Arreglo 2, lo pedido
  explícitamente) COMO el problema de sobre-acumulación a `pasos_fijo` grande
  que ya aparecía incluso a N=200 fijo (mismo mecanismo matemático).

  Uso: los experimentos que necesiten volver a correrse tras el freno de
  arreglos deben importar `ruido_por_paso()` de aquí en vez de reimplementar
  su propia fórmula de amplitud fija. No se edita `cs074_rcruz.py` (regla del
  proyecto) — este es un módulo nuevo, aditivo.

  Verificación de que el arreglo funciona: ver
  `_verificacion_arreglo2_N_sweep.py` en esta misma carpeta — repite (a
  escala reducida) el barrido de N de E5.6-3 con este módulo y confirma que
  la deriva de conservación (E1) ya NO crece sin control con N.
"""
from __future__ import annotations

import numpy as np


def ruido_por_paso(NOISE_REL: float, eps: float, pasos_fijo: int) -> float:
    """Amplitud de ruido gaussiano POR PASO, calibrada para que la varianza
    acumulada total sobre `pasos_fijo` pasos sea ≈ (NOISE_REL·eps)²,
    independiente de N y de pasos_fijo.

    NOISE_REL: constante congelada del experimento (p.ej. 0.02, la misma que
               usaban los motores originales — NO se cambia el "tamaño" del
               ruido total pretendido, solo cómo se reparte en el tiempo).
    eps:       amplitud de la perturbación inicial del experimento (0 si no
               aplica — entonces el ruido es 0 exacto, igual que antes).
    pasos_fijo: número total de pasos que va a correr la simulación (medido,
               no impuesto — el mismo `pasos_fijo` que cada motor ya calibra
               por lavado).
    """
    if pasos_fijo <= 0:
        return 0.0
    return NOISE_REL * eps / np.sqrt(float(pasos_fijo))


def aplicar_ruido(phi: np.ndarray, NOISE_REL: float, eps: float, pasos_fijo: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Aplica un paso de ruido gaussiano ya calibrado (ver ruido_por_paso) a
    un array phi de cualquier forma (funciona igual para un solo campo o un
    lote/batch)."""
    amp = ruido_por_paso(NOISE_REL, eps, pasos_fijo)
    if amp <= 0.0:
        return phi
    return phi + amp * rng.standard_normal(phi.shape)
