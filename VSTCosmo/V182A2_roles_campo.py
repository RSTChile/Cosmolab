#!/usr/bin/env python3
"""
V182A.2 — ROLES EMERGENTES CON COMPETENCIA ANCLADA EN EL CAMPO (cuerpo V180)
================================================================================
Arregla el hueco de V182A.1: alli la "competencia" la construia un contador de
recompensa y el campo solo consumia tiempo. Aqui:

  CUERPO: V180 REAL, importado verbatim (no se reescribe). Tiene aparato de
  orientacion con Kp plastico y valencia: la competencia en una banda = ACIERTO
  DE ORIENTACION (|banda - orientacion|), que MEJORA al practicar (plasticidad
  de Kp + valencia donde el error es bajo). Eso es campo, no contador.

  UNICO HOOK sobre el cuerpo: el socio entra al campo por el MISMO canal que el
  setpoint —el gradiente (V180: `gradiente += orientacion/90 * 0.3`)—. Se agrega
  `gradiente += (orient_socio/90) * confianza`. El otro es un estimulo de campo,
  ponderado por la confianza relacional. Nada mas se toca.

  COMPETENCIA EMERGENTE: exposicion diferencial (A practica -60°, B practica +60°).
  El acierto mejora por la dinamica del campo, no por asignacion.

  MEMORIA RELACIONAL (por banda): la confianza en el otro adapta segun el ACIERTO
  relativo observado en el campo (|err_propio| vs |err_socio| en la banda). Quien
  orienta mas certero en la banda gana deferencia. El peso del socio = esa confianza.

  ROLES: el menos certero en la banda defiere; el mas certero ancla. Por contexto.

  ABLACION: ON (confianza adaptativa) vs OFF (peso fijo).

NOTA DE TIEMPO: el campo es lento POR DISENO (tiempo real). No se puede comprimir.
La config SMOKE solo verifica el CABLEADO (orientacion responde, hook del socio
mueve, confianza lee acierto). El resultado de fidelidad (brecha de competencia +
roles asentados) sale con la config IMAC, que corre en minutos en tu maquina.
================================================================================
"""
import os, json, time
import numpy as np

# ---- importar el cuerpo V180 real (debe estar junto a este archivo) ----
import importlib.util
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("V180", os.path.join(_here, "V180.py"))
V180 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V180)
DT = V180.DT

# ---- config ----
BANDAS = [-60.0, 0.0, 60.0]
# SMOKE (verifica cableado, corre aqui).            -> IMAC (fidelidad, en tu maquina)
PASOS_PRACTICA   = 400     # -> IMAC: 8000   pasos reales por bloque de practica
PASOS_PROBE      = 300     # -> IMAC: 4000   pasos para medir acierto de una banda
N_BLOQUES_FUERTE = 3       # -> IMAC: 10     bloques de practica en la banda propia
N_BLOQUES_DEBIL  = 1       # bloques en la banda ajena
N_BLOQUES_COMPART= 1       # bloques en la banda compartida (0°)
RONDAS_CPL       = 6       # -> IMAC: 60     rondas de acoplamiento (ciclando bandas)
PASOS_POR_RONDA  = 200     # -> IMAC: 2000   pasos de campo por incorporacion del socio

K_COMP   = 0.20            # sensibilidad de la confianza al acierto relativo (grados)
LR_CONF  = 0.10
CONF_INI, CONF_MIN, CONF_MAX = 0.30, 0.02, 0.95
UMBRAL_ROL = 0.20
TS = time.strftime("%Y%m%d_%H%M%S")


# ============================================================
# ORGANISMO DE DIADA: V180 real + 1 hook (socio al gradiente)
# ============================================================
class OrganismoDiada(V180.OrganismoV180):
    def __init__(self, seed):
        super().__init__(seed, V180.MemoriaEpisodicaV180())
        self._bias_socio = 0.0
        self.set_modo_entrenamiento(False)   # LF activa: el motor actua

    def set_bias_socio(self, orient_socio, confianza):
        # el socio entra como estimulo de campo, por el canal del gradiente
        self._bias_socio = (orient_socio / 90.0) * confianza

    def paso_campo(self, banda):
        """V180.actualizar_con_opciones VERBATIM + la unica linea del hook."""
        self.izquierdo.actualizar(0.0, DT, DT, self.derecho)
        self.derecho.actualizar(0.0, DT, DT, self.izquierdo)
        self.sistema_B_izq.actualizar(0.0, DT, DT, self.sistema_B_der)
        self.sistema_B_der.actualizar(0.0, DT, DT, self.sistema_B_izq)
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        gradiente = omega_A - omega_B
        if abs(self.motor.orientacion) > 0.1:
            gradiente += (self.motor.orientacion / 90.0) * 0.3
        gradiente += self._bias_socio                       # <-- UNICO HOOK
        out = self.motor.ejecutar_con_deliberacion([banda], gradiente, 0.0, DT, trauma=False)
        return out[0]   # orientacion


def practicar(org, banda, pasos):
    """Practica = orientar a la banda durante tiempo real. Afina Kp (plasticidad)
    y construye valencia donde el error baja. La competencia EMERGE de aqui."""
    org._bias_socio = 0.0
    for _ in range(pasos):
        org.paso_campo(banda)


def acierto(org, banda, pasos):
    """Mide acierto de orientacion en la banda (campo): media de |banda - orient|
    en el ultimo tramo. Menor error = mas competente."""
    org._bias_socio = 0.0
    errs = []
    for i in range(pasos):
        o = org.paso_campo(banda)
        if i >= int(pasos*0.7):
            errs.append(abs(banda - o))
    return float(np.mean(errs)) if errs else float(abs(banda - org.motor.orientacion))


# ============================================================
# MEMORIA RELACIONAL (lee acierto de campo)
# ============================================================
class MemoriaRelacional:
    def __init__(self):
        self.conf = {}
    def _k(self, b): return round(b/5)*5 if b != 0 else 0
    def confianza_en(self, b): return self.conf.get(self._k(b), CONF_INI)
    def actualizar(self, b, err_propio, err_socio):
        k = self._k(b)
        if k not in self.conf: self.conf[k] = CONF_INI
        # el socio mas certero que yo en la banda -> objetivo de confianza alto
        s = 1.0/(1.0 + np.exp(-K_COMP*(err_propio - err_socio)))
        objetivo = CONF_MIN + (CONF_MAX-CONF_MIN)*s
        self.conf[k] += LR_CONF*(objetivo - self.conf[k])
        self.conf[k] = float(np.clip(self.conf[k], CONF_MIN, CONF_MAX))
        return self.conf[k]


# ============================================================
# FASES
# ============================================================
def fase_exposicion(A, B):
    for _ in range(N_BLOQUES_FUERTE): practicar(A, -60.0, PASOS_PRACTICA)
    for _ in range(N_BLOQUES_DEBIL):  practicar(A,  60.0, PASOS_PRACTICA)
    for _ in range(N_BLOQUES_FUERTE): practicar(B,  60.0, PASOS_PRACTICA)
    for _ in range(N_BLOQUES_DEBIL):  practicar(B, -60.0, PASOS_PRACTICA)
    for _ in range(N_BLOQUES_COMPART):
        practicar(A, 0.0, PASOS_PRACTICA); practicar(B, 0.0, PASOS_PRACTICA)


def competencia_emergente(A, B):
    """Acierto de campo por banda tras la exposicion (menor error = mas competente)."""
    comp = {}
    for b in BANDAS:
        comp[b] = (acierto(A, b, PASOS_PROBE), acierto(B, b, PASOS_PROBE))
    return comp


def acoplar(A, B, usar_memoria, comp):
    memA, memB = MemoriaRelacional(), MemoriaRelacional()
    traza = {b: {'cAB': [], 'cBA': []} for b in BANDAS}
    for _ in range(RONDAS_CPL):
        for b in BANDAS:
            # acierto observado en la banda (error de cada uno respecto a la banda)
            oA, oB = A.motor.orientacion, B.motor.orientacion
            errA, errB = abs(b - oA), abs(b - oB)
            if usar_memoria:
                cAB = memA.actualizar(b, errA, errB)   # confianza de A en B
                cBA = memB.actualizar(b, errB, errA)
            else:
                cAB = cBA = CONF_INI
            # cada uno incorpora la orientacion del socio por el campo, ponderada
            A.set_bias_socio(oB, cAB); B.set_bias_socio(oA, cBA)
            for _ in range(PASOS_POR_RONDA):
                A.paso_campo(b); B.paso_campo(b)
            traza[b]['cAB'].append(cAB); traza[b]['cBA'].append(cBA)
    return traza, comp


def analizar(comp, traza):
    filas = []
    for b in BANDAS:
        eA, eB = comp[b]                       # error de orientacion (menor = mas experto)
        cAB = traza[b]['cAB'][-1]; cBA = traza[b]['cBA'][-1]
        experto = "A" if eA < eB - 1.0 else ("B" if eB < eA - 1.0 else "—")
        if abs(cAB - cBA) <= UMBRAL_ROL:
            rol, lider = "simetrico (sin rol)", "—"
        elif cAB > cBA:
            rol, lider = "A defiere -> B ancla", "B"
        else:
            rol, lider = "B defiere -> A ancla", "A"
        coincide = (lider == experto) if (lider in ("A","B") and experto in ("A","B")) else (lider == "—" and experto == "—")
        filas.append({'banda': b, 'errA': eA, 'errB': eB, 'cAB': cAB, 'cBA': cBA,
                      'rol': rol, 'lider': lider, 'experto': experto, 'coincide': bool(coincide)})
    return filas


def imprimir(titulo, comp, filas):
    print(f"\n{'='*92}")
    print(titulo)
    print('='*92)
    print(f"  {'banda':>6} | {'errA':>6} {'errB':>6} (orient.) | conf A->B  conf B->A | {'rol emergente':<22} | experto coincide")
    print(f"  {'-'*6}-+-{'-'*20}-+-{'-'*19}-+-{'-'*22}-+-{'-'*15}")
    for f in filas:
        mk = '✅' if f['coincide'] else '❌'
        print(f"  {f['banda']:>+6.0f} | {f['errA']:>6.1f} {f['errB']:>6.1f}           |   {f['cAB']:>5.2f}      {f['cBA']:>5.2f}   | {f['rol']:<22} |   {f['experto']:^3}   [{mk}]")


def correr(usar_memoria, etiqueta):
    A = OrganismoDiada(seed=44); B = OrganismoDiada(seed=77)
    fase_exposicion(A, B)
    comp = competencia_emergente(A, B)
    traza, comp = acoplar(A, B, usar_memoria, comp)
    filas = analizar(comp, traza)
    imprimir(f"RESULTADO — {etiqueta}", comp, filas)
    roles = [f for f in filas if f['lider'] in ('A','B')]
    coin = [f for f in roles if f['coincide']]
    print(f"\n  Roles emergidos: {len(roles)}/{len(BANDAS)}   |   coinciden con competencia de campo: {len(coin)}/{len(roles) if roles else 0}")
    return filas


def main():
    print("="*92)
    print("V182A.2 — ROLES CON COMPETENCIA ANCLADA EN EL CAMPO (cuerpo V180 real)")
    print("="*92)
    print("  Competencia = acierto de orientacion por banda (mejora al practicar: Kp plastico + valencia).")
    print("  Socio inyectado al gradiente (canal del setpoint), ponderado por confianza relacional.")
    print(f"  Config SMOKE (verifica cableado): practica {PASOS_PRACTICA}, probe {PASOS_PROBE}, {RONDAS_CPL} rondas.")
    print("  -> IMAC: subir PASOS_* y conteos marcados para la corrida de fidelidad.")
    print("="*92)
    t0 = time.time()
    filas_on  = correr(True,  "MEMORIA RELACIONAL ON (confianza adaptativa por banda)")
    filas_off = correr(False, "MEMORIA RELACIONAL OFF (peso fijo)")
    print(f"\n{'#'*92}\n#  LECTURA\n{'#'*92}")
    print(f"  ON : roles {sum(1 for f in filas_on if f['lider'] in ('A','B'))}/3, "
          f"coinciden {sum(1 for f in filas_on if f['lider'] in ('A','B') and f['coincide'])}")
    print(f"  OFF: roles {sum(1 for f in filas_off if f['lider'] in ('A','B'))}/3 (se espera 0)")
    print(f"  tiempo {time.time()-t0:.1f}s")
    os.makedirs("V182_logs", exist_ok=True)
    with open(f"V182_logs/v182a2_roles_campo_{TS}.json","w") as f:
        json.dump({'on':filas_on,'off':filas_off}, f, indent=2)


if __name__ == "__main__":
    main()
