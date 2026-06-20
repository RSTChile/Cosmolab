#!/usr/bin/env python3
"""
V182A.4 — COMUNICACION BIDIRECCIONAL: TRANSFERENCIA MUTUA (cuerpo V180 real)
================================================================================
Sigue de V182A.3 (roles emergentes confirmados: A ancla -60°, B ancla +60°,
0° simetrico). Alli la incorporacion ya movia la valencia en AMBOS sentidos,
ponderada por confianza, pero no se MEDIA. Aqui se mide.

PREGUNTA: ¿la comunicacion es bidireccional de verdad? Es decir, tras el
acoplamiento, ¿B aprendio la banda de A (-60°) Y A aprendio la banda de B (+60°)?
Cada uno maestro en su banda, alumno en la del otro.

CRITERIO (transferencia mutua):
  ✅ B gana en -60° (banda de A): val_B(-60) sube por encima del umbral
  ✅ A gana en +60° (banda de B): val_A(+60) sube por encima del umbral
  -> ambos aprenden la banda del otro.

CONTRASTE (lo que aporta la memoria relacional, ablacion):
  ON  (confianza adaptativa): el alumno aprende y el MAESTRO casi no se degrada
      (su confianza en el novato es baja -> no copia hacia abajo). Transferencia
      dirigida: del que sabe al que no.
  OFF (peso fijo): ambos promedian hacia la media -> el alumno aprende algo PERO
      el maestro PIERDE competencia. Aprendizaje al costo del experto.

CUERPO: V180 importado VERBATIM. Mecanismo de V182A.3 sin cambios; solo se agrega
la medicion antes/despues. Config en FIDELIDAD (val llega a ~17). Un script.
================================================================================
"""
import os, json, time
import numpy as np
import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("V180", os.path.join(_here, "V180.py"))
V180 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V180)
DT = V180.DT

BANDAS = [-60.0, 0.0, 60.0]
PASOS_FUERTE   = 160000   # ~20 ciclos: val llega a ~17
PASOS_DEBIL    = 16000
PASOS_COMPART  = 80000
RONDAS_CPL     = 60
K_COMP   = 2.0
LR_CONF  = 0.10
ALFA_INC = 0.05
CONF_INI, CONF_MIN, CONF_MAX = 0.30, 0.02, 0.95
UMBRAL_TRANSFER = 3.0     # ganancia minima en la banda ajena para considerar "aprendio"
TS = time.strftime("%Y%m%d_%H%M%S")


def consolidar(org, banda, pasos):
    for _ in range(pasos):
        org.actualizar_setpoint(0.0, DT, DT, banda, target_reward=banda)


def fase_exposicion(A, B):
    consolidar(A, -60.0, PASOS_FUERTE); consolidar(A, 60.0, PASOS_DEBIL)
    consolidar(B,  60.0, PASOS_FUERTE); consolidar(B, -60.0, PASOS_DEBIL)
    consolidar(A, 0.0, PASOS_COMPART);  consolidar(B, 0.0, PASOS_COMPART)


class MemoriaRelacional:
    def __init__(self): self.conf = {}
    def _k(self, b): return round(b/5)*5 if b != 0 else 0
    def actualizar(self, b, val_otro, val_self):
        k = self._k(b)
        if k not in self.conf: self.conf[k] = CONF_INI
        s = 1.0/(1.0 + np.exp(-K_COMP*(val_otro - val_self)))
        objetivo = CONF_MIN + (CONF_MAX - CONF_MIN)*s
        self.conf[k] += LR_CONF*(objetivo - self.conf[k])
        self.conf[k] = float(np.clip(self.conf[k], CONF_MIN, CONF_MAX))
        return self.conf[k]


def _set_val(org, b, v):
    k = round(b/5)*5 if b != 0 else 0
    org.motor.valencia.valencia[k] = float(np.clip(v, -100, 100))


def acoplar(A, B, usar_memoria):
    memA, memB = MemoriaRelacional(), MemoriaRelacional()
    for _ in range(RONDAS_CPL):
        for b in BANDAS:
            vA = A.get_valencia(b); vB = B.get_valencia(b)
            if usar_memoria:
                cAB = memA.actualizar(b, vB, vA); cBA = memB.actualizar(b, vA, vB)
            else:
                cAB = cBA = CONF_INI
            nvA = vA + ALFA_INC*cAB*(vB - vA)
            nvB = vB + ALFA_INC*cBA*(vA - vB)
            _set_val(A, b, nvA); _set_val(B, b, nvB)


def correr(usar_memoria, etiqueta):
    A = V180.OrganismoV180(seed=44, memoria_episodica=V180.MemoriaEpisodicaV180())
    B = V180.OrganismoV180(seed=77, memoria_episodica=V180.MemoriaEpisodicaV180())
    A.set_modo_entrenamiento(False); B.set_modo_entrenamiento(False)
    fase_exposicion(A, B)
    pre = {b: (A.get_valencia(b), B.get_valencia(b)) for b in BANDAS}
    acoplar(A, B, usar_memoria)
    post = {b: (A.get_valencia(b), B.get_valencia(b)) for b in BANDAS}

    # A: nativo -60, ajeno +60 | B: nativo +60, ajeno -60
    A_nat_pre, A_nat_post = pre[-60.0][0], post[-60.0][0]
    A_aj_pre,  A_aj_post  = pre[60.0][0],  post[60.0][0]
    B_nat_pre, B_nat_post = pre[60.0][1],  post[60.0][1]
    B_aj_pre,  B_aj_post  = pre[-60.0][1], post[-60.0][1]

    print(f"\n{'='*92}\nRESULTADO — {etiqueta}\n{'='*92}")
    print(f"  {'organismo':<12} {'banda':>10} | {'antes':>7} -> {'despues':>7} | {'Δ':>7} | lectura")
    print(f"  {'-'*12}-{'-'*10}-+-{'-'*18}-+-{'-'*7}-+--------")
    print(f"  {'A (exp -60)':<12} {'+60 ajena':>10} | {A_aj_pre:>7.2f} -> {A_aj_post:>7.2f} | {A_aj_post-A_aj_pre:>+7.2f} | aprende de B")
    print(f"  {'A (exp -60)':<12} {'-60 propia':>10} | {A_nat_pre:>7.2f} -> {A_nat_post:>7.2f} | {A_nat_post-A_nat_pre:>+7.2f} | retiene")
    print(f"  {'B (exp +60)':<12} {'-60 ajena':>10} | {B_aj_pre:>7.2f} -> {B_aj_post:>7.2f} | {B_aj_post-B_aj_pre:>+7.2f} | aprende de A")
    print(f"  {'B (exp +60)':<12} {'+60 propia':>10} | {B_nat_pre:>7.2f} -> {B_nat_post:>7.2f} | {B_nat_post-B_nat_pre:>+7.2f} | retiene")

    A_aprende = (A_aj_post - A_aj_pre) > UMBRAL_TRANSFER
    B_aprende = (B_aj_post - B_aj_pre) > UMBRAL_TRANSFER
    A_retiene = A_nat_post > 0.6*A_nat_pre
    B_retiene = B_nat_post > 0.6*B_nat_pre
    print(f"\n  A aprende +60: {'✅' if A_aprende else '❌'}   B aprende -60: {'✅' if B_aprende else '❌'}   "
          f"-> transferencia mutua: {'✅ BIDIRECCIONAL' if (A_aprende and B_aprende) else '❌ no'}")
    print(f"  maestros preservados (no se degradan al enseñar): A {'✅' if A_retiene else '❌'}  B {'✅' if B_retiene else '❌'}")
    return {'pre': {str(k):v for k,v in pre.items()}, 'post': {str(k):v for k,v in post.items()},
            'A_aprende': bool(A_aprende), 'B_aprende': bool(B_aprende),
            'A_retiene': bool(A_retiene), 'B_retiene': bool(B_retiene)}


def main():
    print("="*92)
    print("V182A.4 — COMUNICACION BIDIRECCIONAL: TRANSFERENCIA MUTUA (cuerpo V180 real)")
    print("="*92)
    print("  ¿Ambos aprenden la banda del otro? A maestro en -60/alumno en +60; B al reves.")
    print(f"  Config FIDELIDAD: consolida {PASOS_FUERTE}/{PASOS_DEBIL}/{PASOS_COMPART}, {RONDAS_CPL} rondas.")
    print("="*92)
    t0 = time.time()
    on  = correr(True,  "MEMORIA RELACIONAL ON  (transferencia dirigida: del que sabe al que no)")
    off = correr(False, "MEMORIA RELACIONAL OFF (peso fijo: promedio que degrada al maestro)")
    print(f"\n{'#'*92}\n#  LECTURA\n{'#'*92}")
    bi_on  = on['A_aprende'] and on['B_aprende']
    print(f"  ON : bidireccional {'SÍ' if bi_on else 'NO'} | maestros preservados: A={on['A_retiene']} B={on['B_retiene']}")
    print(f"  OFF: bidireccional {'SÍ' if (off['A_aprende'] and off['B_aprende']) else 'NO'} | maestros preservados: A={off['A_retiene']} B={off['B_retiene']}")
    print("  La diferencia ON vs OFF en 'maestros preservados' dice si la memoria relacional")
    print("  hace transferencia DIRIGIDA (aprende el que no sabe, sin costar al que sabe)")
    print("  o solo promedia (todos hacia la media, el experto se degrada).")
    print(f"\n  tiempo {time.time()-t0:.1f}s")
    os.makedirs("V182_logs", exist_ok=True)
    with open(f"V182_logs/v182a4_bidireccional_{TS}.json","w") as f:
        json.dump({'on':on,'off':off}, f, indent=2)


if __name__ == "__main__":
    main()