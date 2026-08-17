#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_HomeostasisEmergente — H canónica (O-N9.14) + diagnóstico "¿qué sostiene A_sys-env?"
================================================================================
SIN Shannon encubierto. Capa POSTERIOR a _rc_observar (A y B la importan; simétrico).

CANON:
- O-N2.1: A_sys-env ∈ ℝ⁺ es la variable UNIFICADORA.
- O-N9.14: H se operacionaliza como A_sys-env EN RANGO ESTABLE.
- ⇒ H NO se calcula contra x0/kp/x_interna; mide si las dinámicas internas SOSTIENEN el acoplamiento.
- Homeostasis = estabilidad VIABLE a través del cambio (montañista: 0.40 ó 0.95 pueden ser sanos),
  no un setpoint. La BANDA viable EMERGE de la propia historia de A (EMA centro + varianza).

FASE 1 — HomeostasisEmergente: H real con ICR/IRDE LITERALES (de RC).
   acople_sostenido = calidad con que la competencia ICR↔IRDE sostiene A_sys-env estable en su banda.
FASE 2 — soporte_A_sys_env: descomposición DIAGNÓSTICA (no mando) de qué sostiene/degrada A.

Regla anti-Shannon: ninguna constante elegida "para que gire" ni "para que H sea alta" ni target_A.
Solo constantes FISIOLÓGICAS declaradas (banda, escalas, EMA, pisos de viabilidad/actividad).
================================================================================
"""
from __future__ import annotations
from typing import Optional


def _c01(v, d=0.0):
    try:
        x = float(v)
    except Exception:
        return d
    return 0.0 if x != x else max(0.0, min(1.0, x))   # NaN-safe


def _num(v, d=0.0):
    try:
        x = float(v)
        return x if x == x else d
    except Exception:
        return d


# Columnas exportadas (se añaden a COLS sin tocar las anteriores)
COLS_HOMEO_REAL = ["acople_sostenido", "acople_A_estabilidad", "acople_RC_vivo", "acople_competencia_ICR_IRDE",
                   "acople_recuperacion_A", "acople_autoencierro", "acople_anestesia",
                   "acople_banda_centro_A", "acople_banda_var_A", "acople_dA_sys_env"]
COLS_SOPORTE_A = ["A_soporte_LF", "A_soporte_comprension", "A_soporte_confianza", "A_soporte_S_shared",
                  "A_soporte_altruismo", "A_soporte_fatiga", "A_soporte_RC",
                  "A_soporte_total", "A_riesgo_desacople"]
# FASE 3 — act_perm (locus del manifiesto): permeabilidad activa de membrana (NO es el actuador de
# orientación). Latente por defecto: act_perm_alpha_sugerido NO se aplica al soma (preparar, no forzar).
COLS_PERM = ["act_perm", "act_perm_demanda", "act_perm_modo", "act_perm_energia", "act_perm_alpha_sugerido"]
COLS_HOMEO_EMERGENTE = COLS_HOMEO_REAL + COLS_SOPORTE_A + COLS_PERM


class HomeostasisEmergente:
    """FASE 1. H = ¿la competencia ICR↔IRDE (literal, de RC) sostiene A_sys-env estable en su banda
    viable EMERGENTE? Mantiene la historia de A (centro+varianza por EMA) — la banda no se fija.
    Constantes FISIOLÓGICAS declaradas (no para producir orientación ni estabilizar artificialmente)."""

    def __init__(self, ema: float = 0.05, banda: float = 0.15, escala_dA: float = 0.002,
                 rc_piso: float = 0.002, A_min=None) -> None:
        self.ema = ema              # memoria de la historia de A (banda emergente)
        self.banda = banda          # ancho viable fisiológico (como 36–37.5°C: tolerancia, no objetivo)
        self.escala_dA = escala_dA  # escala de la derivada dA/dt que cuenta como "moverse"
        # ── EL PISO DE VIABILIDAD DEL ACOPLE ES `banda`, NO UN NÚMERO PROPIO (8-ago-2026) ──
        # ERA `A_min = 0.1` y MEDIDO no discriminaba: sobre 157.421 pasos (108 CSV, 3–8 ago)
        # `A_sys_env` nunca bajó de 0,1638 ⇒ `viable = min(1, A/A_min)` valió 1,0 en el 100,00 %
        # de los pasos, y la pata baja de `anestesia` (A < 2·A_min = 0,20) se cumplió el 0,094 %,
        # con `acople_anestesia` = 0,0000 en el 99,996 %. Dos usos, ningún caso decidido.
        # NO SE RELATIVIZA CONTRA LA HISTORIA DE A: es una condición de vida, no una percepción
        # (advertencia 2 de escala.py). Se deriva de una relación YA DECLARADA aquí mismo y en las
        # MISMAS UNIDADES: `banda` es la anchura de fluctuación de A que este organelo considera
        # tolerable (estab_A = 1 − sd(A)/banda). Un acoplamiento menor que la tolerancia de sus
        # propias fluctuaciones ya no es un agarre. Un número menos; y si `banda` se mide, el piso
        # la sigue. Se llama `A_piso` y no `A_min` porque ya no es un mínimo elegido, sino un
        # derivado. `A_min` se sigue aceptando en la firma para no romper llamadas: se IGNORA.
        # QUÉ SE MUEVE, REPLAYADO fila por fila sobre los 157.421 pasos: `viable` sigue en 1,0 el
        # 100 % del tiempo (un piso de muerte se mide en que NO se cruza) pero el margen sobre el
        # peor acople registrado baja de 63,8 % a 9,2 %; `acople_sostenido` cambia en el 0,04 % de
        # los pasos (|ΔH| máximo 0,0825, medio 0,000008) y todo ese cambio viene de la anestesia,
        # que pasa de existir el 0,00 % del tiempo al 0,34 %. Deja de haber un término muerto.
        self.A_piso = banda         # piso de viabilidad: A∈ℝ⁺ (O-N2.1); A→0 = colapso
        # rc_piso: nivel de "razón cosmosemiótica mínima viva". CALIBRADO a la escala medida de
        # RC_total (silencio ≈ 0.0004; actividad típica ≈ 0.004) — NO elegido para subir H.
        self.rc_piso = rc_piso
        self.reset()

    def reset(self) -> None:
        self._A_ema = None; self._varA = 0.0; self._dA_ema = 0.0; self._A_prev = None

    def actualizar(self, fila: dict) -> dict:
        A = _c01(fila.get("A_sys_env"))
        RC = max(0.0, _num(fila.get("RC_total")))
        ICR = _c01(fila.get("ICR")); IRDE = _c01(fila.get("IRDE"))
        # ICR/IRDE en ESCALA NORMALIZADA [0,1] (la cruda es ~0.002): se usan los _ratio si existen,
        # que son las CUOTAS de cada rama de la razón cosmosemiótica (suman ~1). Fallback: del crudo.
        tot = ICR + IRDE
        icr_r = _c01(fila.get("ICR_ratio"), ICR / tot if tot > 1e-9 else 0.0)
        irde_r = _c01(fila.get("IRDE_ratio"), IRDE / tot if tot > 1e-9 else 0.0)
        # --- banda viable EMERGENTE (montañista): centro + dispersión salen de la historia de A ---
        if self._A_ema is None:
            self._A_ema = A; self._A_prev = A
        desv = A - self._A_ema
        self._A_ema += self.ema * desv
        self._varA = (1.0 - self.ema) * self._varA + self.ema * desv * desv
        self._dA_ema = (1.0 - self.ema) * self._dA_ema + self.ema * (A - self._A_prev)
        self._A_prev = A
        estab_A = max(0.0, 1.0 - (self._varA ** 0.5) / max(1e-6, self.banda))   # A estable en su banda
        viable = min(1.0, A / max(1e-6, self.A_piso))                           # A>0 (no colapso)
        tendencia = max(0.0, min(1.0, 1.0 + self._dA_ema / self.escala_dA))     # A se mantiene/mejora
        rc_gate = min(1.0, RC / max(1e-6, self.rc_piso))                        # razón cosmosemiótica VIVA (no anestesia)
        # competencia ICR↔IRDE VIVA: ambas ramas presentes (no igualdad fija). 2·min(cuotas): es 1
        # sólo si la rama débil tiene ≥0.5 de cuota, decae si una rama colapsa; 0.5 es su máximo teórico.
        competencia = max(0.0, min(1.0, 2.0 * min(icr_r, irde_r)))
        # recuperación: A subiendo, pero NO por apagar la percepción (gateada por RC vivo)
        recuperacion = max(0.0, min(1.0, (self._dA_ema / self.escala_dA))) * rc_gate
        # patologías (O-N2.1), en cuotas: AUTOENCIERRO = la rama de ORDEN domina mientras A cae;
        autoencierro = max(0.0, min(1.0, icr_r - irde_r)) * max(0.0, min(1.0, -self._dA_ema / self.escala_dA))
        # ANESTESIA = la razón cae a casi-silencio (deja de registrar el entorno) mientras A es baja.
        # La pata baja de la anestesia usa el MISMO piso (dos tolerancias de A): con el piso viejo
        # (2·0,10 = 0,20) se cumplía el 0,094 % de los pasos y `acople_anestesia` era 0,0000 en el
        # 99,996 % —otro término muerto—; con 2·banda = 0,30 se cumple el 20,75 %, y como sigue
        # exigiendo que la razón cosmosemiótica esté además apagada (RC < rc_piso), la anestesia
        # acaba existiendo en el 0,34 % de los pasos (máximo 0,3097 contra el 0,0250 de antes).
        anestesia = max(0.0, 1.0 - RC / max(1e-6, self.rc_piso)) * max(0.0, 1.0 - A / max(1e-6, 2.0 * self.A_piso))
        # H = ¿la competencia sostiene A estable y viable? (NO distancia a un punto)
        H = estab_A * viable * tendencia * competencia * rc_gate
        H = max(0.0, min(1.0, H - autoencierro - anestesia))
        return {
            "acople_sostenido": round(H, 4), "acople_A_estabilidad": round(estab_A, 4),
            "acople_RC_vivo": round(rc_gate, 4), "acople_competencia_ICR_IRDE": round(competencia, 4),
            "acople_recuperacion_A": round(recuperacion, 4), "acople_autoencierro": round(autoencierro, 4),
            "acople_anestesia": round(anestesia, 4), "acople_banda_centro_A": round(self._A_ema, 4),
            "acople_banda_var_A": round(self._varA, 6), "acople_dA_sys_env": round(self._dA_ema, 6),
        }


def soporte_A_sys_env(fila: dict, dA_sys_env: float = 0.0, escala_dA: float = 0.002) -> dict:
    """FASE 2 — DIAGNÓSTICO (no mando): descompone qué factores internos SOSTIENEN o DEGRADAN A_sys-env.
    No impone A, no calcula A objetivo, no fuerza movimiento. Solo mide. dA_sys_env = tendencia de A."""
    A = _c01(fila.get("A_sys_env"))
    ICR = _c01(fila.get("ICR")); IRDE = _c01(fila.get("IRDE"))
    tot = ICR + IRDE
    icr_r = _c01(fila.get("ICR_ratio"), ICR / tot if tot > 1e-9 else 0.0)   # cuota de conversión/orden
    irde_r = _c01(fila.get("IRDE_ratio"), IRDE / tot if tot > 1e-9 else 0.0)  # cuota de desviación/riesgo
    no_cae = max(0.0, min(1.0, 0.5 + dA_sys_env / (2.0 * escala_dA)))   # 1 si A sube, 0.5 estable, 0 si cae
    LF = max(_c01(fila.get("LF_op")), _c01(fila.get("LF_struct")))
    comp = _c01(fila.get("act_comprension_L")) + _c01(fila.get("act_comprension_R"))
    comp = min(1.0, comp)
    confianza = _c01(fila.get("act_confianza"))
    S_shared = max(_c01(fila.get("altruismo_S_shared")), _c01(fila.get("ME")))
    disp = _c01(fila.get("disposicion_cooperar"))
    psi = _c01(fila.get("altruismo_psi_alma"))
    fat = _num(fila.get("act_fatiga"))
    fat_norm = fat / (1.0 + fat)
    # SOPORTES (positivos sólo si se asocian a sostener A; LF/altruismo penalizados si suben riesgo)
    s_LF = LF * (1.0 - irde_r) * no_cae                     # LF sin acoplamiento = deriva (penalizada si A cae/sube IRDE)
    s_comp = comp * no_cae                                  # comprensión cuenta si A no cae
    s_conf = confianza                                      # soporte relacional
    s_S = S_shared                                          # memoria externalizable / coordinación
    s_altru = disp * psi * (1.0 - irde_r)                   # cooperación: soporte si recíproca y no sube riesgo
    s_fat = 1.0 - fat_norm                                  # fatiga alta resta soporte
    s_RC = icr_r * no_cae                                   # ICR convierte ruido en sentido si A se sostiene
    soportes = [s_LF, s_comp, s_conf, s_S, s_altru, s_fat, s_RC]
    total = sum(soportes) / len(soportes)
    riesgo = max(0.0, min(1.0, irde_r * max(0.0, -dA_sys_env / escala_dA)))   # cuota de riesgo + A cayendo
    return {
        "A_soporte_LF": round(s_LF, 4), "A_soporte_comprension": round(s_comp, 4),
        "A_soporte_confianza": round(s_conf, 4), "A_soporte_S_shared": round(s_S, 4),
        "A_soporte_altruismo": round(s_altru, 4), "A_soporte_fatiga": round(s_fat, 4),
        "A_soporte_RC": round(s_RC, 4), "A_soporte_total": round(total, 4),
        "A_riesgo_desacople": round(riesgo, 4),
    }


# Aplicar la permeabilidad al acople del soma cerraría el lazo conductual (A→H→OI→conducta→nuevo A).
# Por defecto FALSE: act_perm queda DEFINIDA por el circuito pero LATENTE (preparar, no forzar).
APLICAR_PERMEABILIDAD = False


def permeabilidad_activa(fila: dict, escala_alpha: float = 1.0) -> dict:
    """FASE 3 — act_perm DEFINIDA POR EL CIRCUITO (no constante externa, no setpoint, no target_A).
    Daisyworld de la MEMBRANA: ante DESACOPLE (H_real bajo = presión por actuar) el organismo ABRE
    para re-acoplarse SI tiene disposición de conversión (cuota ICR) y energía (no fatiga); se
    REPLIEGA si domina la desviación/riesgo o está exhausto. Anti-autoencierro: cerrarse mientras A
    cae sería patológico, así que el impulso sano ante desacople (con ICR vivo) es ABRIR.
    LATENTE: act_perm_alpha_sugerido es el multiplicador de rigidez de acople que el soma USARÍA si
    se cierra el lazo (APLICAR_PERMEABILIDAD); por defecto NO se aplica — sólo se mide."""
    icr_r = _c01(fila.get("ICR_ratio"))                                   # cuota de conversión (vs protección)
    fat = _num(fila.get("act_fatiga")); energia = 1.0 - fat / (1.0 + fat) # energía disponible
    demanda = max(0.0, min(1.0, 1.0 - _c01(fila.get("acople_sostenido"))))  # H baja ⇒ presión por re-acoplar
    modo = icr_r                                                          # disposición a abrir/convertir
    act_perm = max(0.0, min(1.0, demanda * modo * energia))              # abrir membrana para re-acoplar
    alpha_sugerido = 1.0 + escala_alpha * act_perm                       # multiplicador LATENTE (no aplicado)
    return {"act_perm": round(act_perm, 4), "act_perm_demanda": round(demanda, 4),
            "act_perm_modo": round(modo, 4), "act_perm_energia": round(energia, 4),
            "act_perm_alpha_sugerido": round(alpha_sugerido, 4)}


if __name__ == "__main__":
    he = HomeostasisEmergente()
    f = {"A_sys_env": 0.7, "RC_total": 0.5, "ICR": 0.5, "IRDE": 0.5, "LF_op": 0.5,
         "act_comprension_L": 0.3, "act_comprension_R": 0.2, "act_confianza": 0.4, "act_fatiga": 2.0}
    for _ in range(50):
        r = he.actualizar(f)
    s = soporte_A_sys_env(f, dA_sys_env=r["acople_dA_sys_env"])
    print("smoke H:", r)
    print("smoke soporte:", s)
