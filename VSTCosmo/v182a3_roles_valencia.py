#!/usr/bin/env python3
"""
V182A.3 — ROLES CON COMPETENCIA = VALENCIA GANADA (cuerpo V180 real, verbatim)
================================================================================
Corrige V182A.2, que apuntaba al medidor equivocado. Dos errores de V182A.2:
  - media competencia como ACIERTO DE ORIENTACION -> se re-equilibra a ~2.4° para
    todos durante el probe; no retiene ventaja por banda. NO sirve.
  - "practicaba" por la ruta de deliberacion (reward=0) -> la valencia no crecia.

CORRECCION (verificada contra la corrida real de V180):
  - Competencia por banda = VALENCIA, ganada por la ruta de CONSOLIDACION real
    (`actualizar_setpoint(..., target_reward=banda)`): premia cuando el organismo
    esta asentado en la banda, y la valencia sube SOLO ahi (val de la banda ajena
    se queda en 0). En la corrida real: 20 ciclos -> val(-60) ~= 17.
  - Es campo, no contador: la valencia solo crece si el organismo orienta bien de
    verdad (|error| < zona_muerta). La logica de roles es la de V182A.1, ahora
    FIELD-GROUNDED sobre V180.

CUERPO: V180 importado VERBATIM. No se modifica (el acoplamiento es a nivel de
valencia, no necesita hook en el gradiente).

DISENO:
  F1 Exposicion diferencial: A consolida -60°, B consolida +60°, ambos 0°
     -> competencia EMERGENTE (valencia) asimetrica por banda.
  F2 Acoplamiento por banda con MEMORIA RELACIONAL: cada uno ajusta su confianza
     en el otro segun la valencia relativa observada (quien sabe mas en la banda),
     y incorpora la valencia del otro ponderada por esa confianza.
     ROL: el menos competente en la banda defiere; el mas competente ancla.
  ABLACION: ON (confianza adaptativa) vs OFF (peso fijo).

ESCALA: SMOKE verifica que la valencia se construye/diferencia y que los roles se
cablean. La fidelidad (val ~17, roles asentados) sale con los conteos -> IMAC.
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
# SMOKE (cablear)                              -> IMAC (fidelidad)
PASOS_FUERTE   = 160000  # ~20 ciclos: val llega a ~17
PASOS_DEBIL    = 16000
PASOS_COMPART  = 80000
RONDAS_CPL     = 60      # -> IMAC: 120
K_COMP   = 2.0           # sensibilidad de confianza a la valencia relativa (escala ~0-17: gap de 1 ya separa, gap de 17 satura)
LR_CONF  = 0.10
ALFA_INC = 0.05          # incorporacion suave: la brecha de competencia no se borra antes de que el rol se forme
CONF_INI, CONF_MIN, CONF_MAX = 0.30, 0.02, 0.95
UMBRAL_ROL = 0.20
TS = time.strftime("%Y%m%d_%H%M%S")


def consolidar(org, banda, pasos):
    """Consolidacion REAL: orienta a la banda con premio -> valencia (competencia)
    crece SOLO si el organismo se asienta ahi. Campo, no contador."""
    for _ in range(pasos):
        org.actualizar_setpoint(0.0, DT, DT, banda, target_reward=banda)


def fase_exposicion(A, B):
    consolidar(A, -60.0, PASOS_FUERTE); consolidar(A, 60.0, PASOS_DEBIL)
    consolidar(B,  60.0, PASOS_FUERTE); consolidar(B, -60.0, PASOS_DEBIL)
    consolidar(A, 0.0, PASOS_COMPART);  consolidar(B, 0.0, PASOS_COMPART)


class MemoriaRelacional:
    """Confianza por banda en el otro, segun valencia (competencia) relativa."""
    def __init__(self): self.conf = {}
    def _k(self, b): return round(b/5)*5 if b != 0 else 0
    def confianza_en(self, b): return self.conf.get(self._k(b), CONF_INI)
    def actualizar(self, b, val_otro, val_self):
        k = self._k(b)
        if k not in self.conf: self.conf[k] = CONF_INI
        s = 1.0/(1.0 + np.exp(-K_COMP*(val_otro - val_self)))   # el otro mas competente -> confianza sube
        objetivo = CONF_MIN + (CONF_MAX - CONF_MIN)*s
        self.conf[k] += LR_CONF*(objetivo - self.conf[k])
        self.conf[k] = float(np.clip(self.conf[k], CONF_MIN, CONF_MAX))
        return self.conf[k]


def _set_val(org, b, v):
    k = round(b/5)*5 if b != 0 else 0
    org.motor.valencia.valencia[k] = float(np.clip(v, -100, 100))


def acoplar(A, B, usar_memoria):
    memA, memB = MemoriaRelacional(), MemoriaRelacional()
    traza = {b: {'cAB': [], 'cBA': []} for b in BANDAS}
    for _ in range(RONDAS_CPL):
        for b in BANDAS:
            vA = A.get_valencia(b); vB = B.get_valencia(b)
            if usar_memoria:
                cAB = memA.actualizar(b, vB, vA)   # confianza de A en B
                cBA = memB.actualizar(b, vA, vB)
            else:
                cAB = cBA = CONF_INI
            # incorporacion ponderada por confianza: el que defiere mueve su valencia hacia el otro
            nvA = vA + ALFA_INC*cAB*(vB - vA)
            nvB = vB + ALFA_INC*cBA*(vA - vB)
            _set_val(A, b, nvA); _set_val(B, b, nvB)
            traza[b]['cAB'].append(cAB); traza[b]['cBA'].append(cBA)
    return traza


def analizar(A, B, comp_ini, traza):
    filas = []
    for b in BANDAS:
        vA, vB = comp_ini[b]
        cAB = traza[b]['cAB'][-1]; cBA = traza[b]['cBA'][-1]
        experto = "A" if vA > vB + 0.05 else ("B" if vB > vA + 0.05 else "—")
        if abs(cAB - cBA) <= UMBRAL_ROL:
            rol, lider = "simetrico (sin rol)", "—"
        elif cAB > cBA:
            rol, lider = "A defiere -> B ancla", "B"
        else:
            rol, lider = "B defiere -> A ancla", "A"
        coincide = (lider == experto) if (lider in ("A","B") and experto in ("A","B")) else (lider == "—" and experto == "—")
        filas.append({'banda': b, 'vA': vA, 'vB': vB, 'cAB': cAB, 'cBA': cBA,
                      'rol': rol, 'lider': lider, 'experto': experto, 'coincide': bool(coincide)})
    return filas


def imprimir(titulo, filas):
    print(f"\n{'='*92}")
    print(titulo); print('='*92)
    print(f"  {'banda':>6} | {'val A':>6} {'val B':>6} (competencia) | conf A->B  conf B->A | {'rol emergente':<22} | exp coincide")
    print(f"  {'-'*6}-+-{'-'*22}-+-{'-'*19}-+-{'-'*22}-+-{'-'*12}")
    for f in filas:
        mk = '✅' if f['coincide'] else '❌'
        print(f"  {f['banda']:>+6.0f} | {f['vA']:>6.2f} {f['vB']:>6.2f}             |   {f['cAB']:>5.2f}      {f['cBA']:>5.2f}   | {f['rol']:<22} |  {f['experto']:^3} [{mk}]")


def correr(usar_memoria, etiqueta):
    A = V180.OrganismoV180(seed=44, memoria_episodica=V180.MemoriaEpisodicaV180())
    B = V180.OrganismoV180(seed=77, memoria_episodica=V180.MemoriaEpisodicaV180())
    A.set_modo_entrenamiento(False); B.set_modo_entrenamiento(False)
    fase_exposicion(A, B)
    comp_ini = {b: (A.get_valencia(b), B.get_valencia(b)) for b in BANDAS}
    traza = acoplar(A, B, usar_memoria)
    filas = analizar(A, B, comp_ini, traza)
    print(f"\n  [competencia EMERGENTE tras consolidacion — valencia por banda]")
    for b in BANDAS:
        print(f"     {b:>+5.0f}°:  A={comp_ini[b][0]:+.2f}   B={comp_ini[b][1]:+.2f}")
    imprimir(f"RESULTADO — {etiqueta}", filas)
    roles = [f for f in filas if f['lider'] in ('A','B')]
    coin = [f for f in roles if f['coincide']]
    print(f"\n  Roles emergidos: {len(roles)}/{len(BANDAS)}   |   coinciden con competencia: {len(coin)}/{len(roles) if roles else 0}")
    return filas


def main():
    print("="*92)
    print("V182A.3 — ROLES CON COMPETENCIA = VALENCIA GANADA (cuerpo V180 real)")
    print("="*92)
    print("  Competencia = valencia construida por consolidacion real (premio cuando asentado).")
    print("  Roles = confianza por banda sobre valencia relativa. Cuerpo V180 verbatim (sin hook).")
    print(f"  Config FIDELIDAD: consolida {PASOS_FUERTE}/{PASOS_DEBIL}/{PASOS_COMPART}, {RONDAS_CPL} rondas.")
    print("  -> IMAC: subir PASOS_* a los valores marcados (val llega a ~17).")
    print("="*92)
    t0 = time.time()
    filas_on  = correr(True,  "MEMORIA RELACIONAL ON")
    filas_off = correr(False, "MEMORIA RELACIONAL OFF (peso fijo)")
    print(f"\n{'#'*92}\n#  LECTURA\n{'#'*92}")
    print(f"  ON : roles {sum(1 for f in filas_on if f['lider'] in ('A','B'))}/3, "
          f"coinciden {sum(1 for f in filas_on if f['lider'] in ('A','B') and f['coincide'])}")
    print(f"  OFF: roles {sum(1 for f in filas_off if f['lider'] in ('A','B'))}/3 (se espera 0)")
    print(f"  tiempo {time.time()-t0:.1f}s")
    os.makedirs("V182_logs", exist_ok=True)
    with open(f"V182_logs/v182a3_roles_valencia_{TS}.json","w") as f:
        json.dump({'on':filas_on,'off':filas_off}, f, indent=2)


if __name__ == "__main__":
    main()
