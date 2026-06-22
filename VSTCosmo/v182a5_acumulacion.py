#!/usr/bin/env python3
"""
V182A.5 — ACUMULACION CON RECONSOLIDACION: ¿CULTURA O REGRESION A LA MEDIA?
================================================================================
Junta dos cosas en un experimento (cuerpo V180 real, verbatim):

  PARTE 1 (medidor corregido): la ronda 1 es V182A.4 (transferencia mutua), pero
  la RETENCION del maestro se mide como % real (post/pre), no con el binario de
  umbral 0.6 que dio el "A=True" espurio en OFF.

  PARTE 2 (acumulacion con reconsolidacion): rondas 2..N, cada una =
  [cada uno RECONSOLIDA su banda nativa] -> [intercambian]. Reconsolidar entre
  rondas es lo que hace honesto el test: hay algo nuevo que transferir cada vez
  (el experto sigue practicando lo suyo). Sin reconsolidar, encadenar solo
  confirma que ya convergieron.

PREGUNTA DECISIVA:
  ON  -> ¿ambos suben en AMBAS bandas ronda tras ronda (cultura acumulativa:
         cada uno termina experto en todo)?
  OFF -> ¿todos regresan a la media (entropia: cada intercambio borra lo que el
         experto reconsolido)?

Se traza val_A(-60), val_A(+60), val_B(-60), val_B(+60) ronda a ronda, ON y OFF.

CUERPO: V180 importado VERBATIM. Config FIDELIDAD. Un script, listo para correr.
ADVERTENCIA: corre la exposicion + reconsolidaciones en tiempo real de campo;
tarda VARIOS MINUTOS (mas que V182A.4, porque reconsolida en cada ronda).
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
PASOS_FUERTE     = 80000    # exposicion inicial banda propia (val ~10-12, brecha clara)
PASOS_DEBIL      = 8000
PASOS_COMPART    = 40000
PASOS_RECONSOLIDA= 15000    # top-up de la banda nativa entre rondas
RONDAS_ACUM      = 5        # rondas de acumulacion (ronda 1 = V182A.4)
RONDAS_CPL       = 30       # sub-rondas de intercambio por ronda de acumulacion
K_COMP   = 2.0
LR_CONF  = 0.10
ALFA_INC = 0.05
CONF_INI, CONF_MIN, CONF_MAX = 0.30, 0.02, 0.95
UMBRAL_EXPERTO = 10.0    # min de las 4 bandas por encima de esto = ambos expertos en AMBAS -> ✅
UMBRAL_RETEN   = 85.0    # % retencion del maestro por encima de esto = preservado -> ✅
TS = time.strftime("%Y%m%d_%H%M%S")


def consolidar(org, banda, pasos):
    for _ in range(pasos):
        org.actualizar_setpoint(0.0, DT, DT, banda, target_reward=banda)


def _set_val(org, b, v):
    k = round(b/5)*5 if b != 0 else 0
    org.motor.valencia.valencia[k] = float(np.clip(v, -100, 100))


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


def intercambiar(A, B, memA, memB, usar_memoria):
    for _ in range(RONDAS_CPL):
        for b in BANDAS:
            vA = A.get_valencia(b); vB = B.get_valencia(b)
            if usar_memoria:
                cAB = memA.actualizar(b, vB, vA); cBA = memB.actualizar(b, vA, vB)
            else:
                cAB = cBA = CONF_INI
            _set_val(A, b, vA + ALFA_INC*cAB*(vB - vA))
            _set_val(B, b, vB + ALFA_INC*cBA*(vA - vB))


def correr(usar_memoria, etiqueta):
    A = V180.OrganismoV180(seed=44, memoria_episodica=V180.MemoriaEpisodicaV180())
    B = V180.OrganismoV180(seed=77, memoria_episodica=V180.MemoriaEpisodicaV180())
    A.set_modo_entrenamiento(False); B.set_modo_entrenamiento(False)
    consolidar(A, -60.0, PASOS_FUERTE); consolidar(A, 60.0, PASOS_DEBIL)
    consolidar(B,  60.0, PASOS_FUERTE); consolidar(B, -60.0, PASOS_DEBIL)
    consolidar(A, 0.0, PASOS_COMPART);  consolidar(B, 0.0, PASOS_COMPART)
    memA, memB = MemoriaRelacional(), MemoriaRelacional()

    def snap():
        return (A.get_valencia(-60.), A.get_valencia(60.), B.get_valencia(-60.), B.get_valencia(60.))

    pre = snap()
    traza = [pre]
    for r in range(1, RONDAS_ACUM+1):
        if r > 1:
            consolidar(A, -60.0, PASOS_RECONSOLIDA)   # cada uno reconsolida SU banda nativa
            consolidar(B,  60.0, PASOS_RECONSOLIDA)
        intercambiar(A, B, memA, memB, usar_memoria)
        traza.append(snap())

    print(f"\n{'='*92}\n{etiqueta}\n{'='*92}")
    print(f"  ronda | A(-60) A(+60) | B(-60) B(+60) |   min  spread | estado")
    print(f"  {'-'*6}+{'-'*14}+{'-'*14}+{'-'*14}+{'-'*22}")
    for i, s in enumerate(traza):
        vals = list(s); mn = min(vals); sp = max(vals)-min(vals)
        et = "inicial" if i == 0 else ("V182A.4" if i == 1 else "")
        mk = '✅ expertos en ambas' if mn > UMBRAL_EXPERTO else ('❌' if i == len(traza)-1 else '·')
        print(f"  {i:>5} | {s[0]:>6.2f} {s[1]:>6.2f} | {s[2]:>6.2f} {s[3]:>6.2f} | {mn:>5.2f} {sp:>5.2f}  | {mk} {et}")

    # PARTE 1 (ronda 1 = V182A.4) con retencion en % real
    r1 = traza[1]
    ra = 100*r1[0]/pre[0]; rb = 100*r1[3]/pre[3]
    print(f"\n  PARTE 1 — transferencia mutua (ronda 1), retencion en % real:")
    print(f"     A aprende +60: {pre[1]:.2f} -> {r1[1]:.2f}  (Δ {r1[1]-pre[1]:+.2f})   |  A retiene -60: {ra:.0f}% [{'✅' if ra>=UMBRAL_RETEN else '❌'}]")
    print(f"     B aprende -60: {pre[2]:.2f} -> {r1[2]:.2f}  (Δ {r1[2]-pre[2]:+.2f})   |  B retiene +60: {rb:.0f}% [{'✅' if rb>=UMBRAL_RETEN else '❌'}]")

    fin = traza[-1]
    return {'traza': [list(s) for s in traza], 'min_final': float(min(fin)), 'spread_final': float(max(fin)-min(fin)),
            'A_ret_pct': float(100*r1[0]/pre[0]), 'B_ret_pct': float(100*r1[3]/pre[3])}


def main():
    print("="*92)
    print("V182A.5 — ACUMULACION CON RECONSOLIDACION: ¿CULTURA O REGRESION A LA MEDIA?")
    print("="*92)
    print("  Ronda 1 = V182A.4 (transferencia). Rondas 2..N = reconsolidar nativa + intercambiar.")
    print(f"  Config FIDELIDAD: exposicion {PASOS_FUERTE}/{PASOS_DEBIL}/{PASOS_COMPART}, "
          f"reconsolida {PASOS_RECONSOLIDA}, {RONDAS_ACUM} rondas. Tarda varios minutos.")
    print("="*92)
    t0 = time.time()
    on  = correr(True,  "MEMORIA RELACIONAL ON")
    off = correr(False, "MEMORIA RELACIONAL OFF (peso fijo)")

    print(f"\n{'#'*92}\n#  VEREDICTO\n{'#'*92}")
    on_cult = on['min_final'] > UMBRAL_EXPERTO
    off_cult = off['min_final'] > UMBRAL_EXPERTO
    print(f"  ON : min={on['min_final']:.2f}  spread={on['spread_final']:.2f}  retencion R1 {on['A_ret_pct']:.0f}%/{on['B_ret_pct']:.0f}% "
          f"-> {'✅ CULTURA ACUMULATIVA (expertos en ambas)' if on_cult else '❌ no llega a expertos en ambas'}")
    print(f"  OFF: min={off['min_final']:.2f}  spread={off['spread_final']:.2f}  retencion R1 {off['A_ret_pct']:.0f}%/{off['B_ret_pct']:.0f}% "
          f"-> {'✅ cultura' if off_cult else '❌ REGRESION A LA MEDIA'}")
    print()
    if on['min_final'] > off['min_final'] + 2.0:
        print("  ✅ La memoria relacional es la que permite acumular en vez de promediar:")
        print("     ON deja a ambos altos en AMBAS bandas; OFF los deja cerca de la media.")
        print("     Tu objecion al binario queda saldada: retencion en % real, sin artefacto.")
    else:
        print("  ❌ No hay separacion clara entre ON y OFF en el min final. Dato real, lo leemos.")
    print(f"\n  tiempo {time.time()-t0:.1f}s")
    os.makedirs("V182_logs", exist_ok=True)
    with open(f"V182_logs/v182a5_acumulacion_{TS}.json","w") as f:
        json.dump({'on':on,'off':off}, f, indent=2)


if __name__ == "__main__":
    main()
