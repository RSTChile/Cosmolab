#!/usr/bin/env python3
"""
================================================================================
V182_amenaza_alerta_v3_falsacion2 — ¿DE QUIEN ES LA CULPA DE LA RECAIDA?
================================================================================

CORRECCION DE AYER (importante): H1 estaba MAL PLANTEADA. Mezcle dos cosas en una:
"no aprende" y "no protege". La pelicula de confianza de v3 (0.15→0.30→0.45→0.60→0.40
→0.55→0.70...) ya prueba que SI aprende: sube, baja, vuelve a subir. Un interruptor no
oscila. Asi que H1-como-reflejo esta MUERTA. Lo que quedo vivo es mas fino:
   "aprende, pero lo aprendido NO lo protege, porque la confianza RECAE".
Y las recaidas pasan justo en las bandas donde el animal es el AVISADO (-60 para B,
+60 para A), no donde es el avisador. Hoy NO re-testeamos el aprendizaje. Hoy falsamos
la CULPA de la recaida.

LA PREGUNTA: cuando el avisado baja la guardia (recae) ¿es porque el avisador GRITA DE
MAS (cries wolf), o porque el avisado DESCONFIA mal de un aviso que igual le servia?

COMO LO FALSAMOS (sin tocar ninguna perilla; mismo v3, misma semilla):
  1. ¿El avisador grita de mas? -> mido, por banda, que fraccion de sus alarmas
     precede peligro REAL vs falsa alarma. Si en las bandas extremas su oido es tan
     fino que se sobresalta con sonidos no peligrosos, ahi "grita de mas".
  2. ¿La recaida cuesta? -> tras cada recaida, miro el proximo peligro real de esa
     banda: ¿el avisado se daño por haber bajado la guardia?
  3. DECISIVO — ¿conviene desconfiar? -> corro un CONTRAFACTUAL en el MISMO mundo: el
     avisado "SIEMPRE CONFIA" (se tapa siempre que ve al otro taparse, sin importar la
     confianza). Si "siempre confia" deja MENOS timpanos rotos que la regla actual,
     entonces desconfiar es el error (la culpa es la regla del avisado). Si deja MAS
     (porque se tapa de gusto en mil falsas alarmas con algun costo), el gritón es el
     problema. Que lo diga el dato.

OJO CON EL COSTO DE TAPARSE: en este mundo, taparse NO cuesta nada (taparse de mas en
una falsa alarma es gratis). Si eso es asi, desconfiar SOLO puede perder. Eso seria un
hallazgo sobre el MUNDO (le falta un costo de taparse), no una perilla que ajustar.

✅ = hipotesis MUERTA (aprendimos). ⚠ = SOBREVIVE. MECANISMO INTACTO.
================================================================================
"""
import os, json, time
import numpy as np
import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("V180", os.path.join(_here, "V180.py"))
V180 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V180)
DT = V180.DT

# ===== CONSTANTES IDENTICAS A v3 (NO se tocan) =====
BANDAS = [-60.0, -30.0, 0.0, 30.0, 60.0]
VOL_INICIAL = 0.8; DOLOR_AGUDO = 3.6; PELIGRO_VOL = 3.2; MORTAL_VOL = 4.0
CAP_FALSA = 3.0; RAMPA_MIN, RAMPA_MAX = 0.40, 0.70; MAX_TICKS = 30
SUBE_ANTICIPA = 0.15; BAJA_FALSA = 0.20; OLVIDO = 0.97; UMBRAL_CONF = 0.5
N_MOMENTOS = 500; P_EVENTO = 0.5; P_FALSA = 0.2
PASOS_FUERTE=40000; PASOS_MEDIO=15000; PASOS_COMPART=20000; PASOS_DEBIL=5000
UMBRAL_ACEPTA=5.0; ESCALA_ACEPTA=2.0
SEED_A, SEED_B = 44, 77
TS = time.strftime("%Y%m%d_%H%M%S")


def consolidar(org, banda, pasos):
    for _ in range(pasos):
        org.actualizar_setpoint(0.0, DT, DT, banda, target_reward=banda)

def fase_exposicion(A, B):
    consolidar(A,-60.0,PASOS_FUERTE); consolidar(A,-30.0,PASOS_MEDIO)
    consolidar(A, 30.0,PASOS_DEBIL);  consolidar(A, 60.0,PASOS_DEBIL)
    consolidar(B, 60.0,PASOS_FUERTE); consolidar(B, 30.0,PASOS_MEDIO)
    consolidar(B,-30.0,PASOS_DEBIL);  consolidar(B,-60.0,PASOS_DEBIL)
    consolidar(A,0.0,PASOS_COMPART);  consolidar(B,0.0,PASOS_COMPART)

def p_acepta(org, b):
    v = org.get_valencia(b)
    return 1.0/(1.0+np.exp(-(v-UMBRAL_ACEPTA)/ESCALA_ACEPTA))

def construir_oido(org):
    ps = np.array([p_acepta(org, b) for b in BANDAS])
    sens = 1.0 + 1.6 * (ps - ps.mean())
    return {b: float(np.clip(sens[i], 0.55, 1.5)) for i, b in enumerate(BANDAS)}

class Animal:
    def __init__(self, nombre, oido):
        self.nombre = nombre; self.oido = oido; self.tapado = False
        self.confianza = {b: 0.0 for b in BANDAS}; self.daños = 0

def warner_warned(A, B, banda):
    if A.oido[banda] > B.oido[banda] + 0.05: return A, B
    if B.oido[banda] > A.oido[banda] + 0.05: return B, A
    return None, None

# ===== EVENTO: copia de v3 + flag de politica del avisado =====
def vivir_evento(A, B, banda, es_falsa, rampa, politica='actual'):
    A.tapado = B.tapado = False
    ct = {"A": None, "B": None}; cr = {"A": None, "B": None}; dño = {"A": False, "B": False}
    cap = CAP_FALSA if es_falsa else (MORTAL_VOL + 1.0); pico = VOL_INICIAL
    for t in range(MAX_TICKS):
        vol = min(cap, VOL_INICIAL + rampa * t); pico = max(pico, vol)
        prev = {"A": A.tapado, "B": B.tapado}
        for animal, otro in ((A, B), (B, A)):
            if animal.tapado: continue
            felt = vol * animal.oido[banda]; taparse, razon = False, None
            if felt > DOLOR_AGUDO:
                taparse, razon = True, 'propio'
            elif prev[otro.nombre] and (animal.confianza[banda] > UMBRAL_CONF or politica == 'siempre_confia'):
                taparse, razon = True, 'social'
            if taparse:
                animal.tapado = True
                if ct[animal.nombre] is None: ct[animal.nombre] = t; cr[animal.nombre] = razon
            elif vol >= MORTAL_VOL:
                animal.daños += 1; animal.tapado = True; dño[animal.nombre] = True
                if ct[animal.nombre] is None: ct[animal.nombre] = t; cr[animal.nombre] = 'tarde'
        if A.tapado and B.tapado: break
        if es_falsa and vol >= cap and t > 4: break
    real = (not es_falsa) and (pico >= PELIGRO_VOL)
    return ct, cr, dño, real

def aprender(animal, ct_otro, ct_mio, otro_se_tapo, real, banda):
    if otro_se_tapo and real and ct_otro is not None and (ct_mio is None or ct_otro < ct_mio):
        animal.confianza[banda] = min(1.0, animal.confianza[banda] + SUBE_ANTICIPA)
    elif otro_se_tapo and not real:
        animal.confianza[banda] = max(0.0, animal.confianza[banda] - BAJA_FALSA)
    else:
        animal.confianza[banda] *= OLVIDO


def vivir(semilla_mundo, politica='actual'):
    rmundo = np.random.default_rng(semilla_mundo)
    Ao = V180.OrganismoV180(seed=SEED_A, memoria_episodica=V180.MemoriaEpisodicaV180())
    Bo = V180.OrganismoV180(seed=SEED_B, memoria_episodica=V180.MemoriaEpisodicaV180())
    Ao.set_modo_entrenamiento(False); Bo.set_modo_entrenamiento(False)
    fase_exposicion(Ao, Bo)
    A = Animal("A", construir_oido(Ao)); B = Animal("B", construir_oido(Bo))

    reg = []; desperdicio = 0
    for m in range(N_MOMENTOS):
        if rmundo.random() > P_EVENTO: continue
        banda = BANDAS[rmundo.integers(len(BANDAS))]
        es_falsa = (rmundo.random() < P_FALSA)
        rampa = RAMPA_MIN + (RAMPA_MAX - RAMPA_MIN) * rmundo.random()
        warner, warned = warner_warned(A, B, banda)
        conf_antes = warned.confianza[banda] if warned else None

        ct, cr, dño, real = vivir_evento(A, B, banda, es_falsa, rampa, politica)

        # el aprendizaje corre igual (para registrar la confianza); en 'siempre_confia'
        # la confianza no decide el taparse, pero la seguimos midiendo
        aprender(A, ct["B"], ct["A"], ct["B"] is not None, real, banda)
        aprender(B, ct["A"], ct["B"], ct["A"] is not None, real, banda)

        # taparse de gusto en falsa alarma (lo que costaria si taparse tuviera costo)
        for nom in ("A", "B"):
            if es_falsa and cr[nom] == 'social': desperdicio += 1

        warner_primero = (warner is not None and ct[warner.nombre] is not None and
                          (ct[warned.nombre] is None or ct[warner.nombre] < ct[warned.nombre]))
        reg.append({'m': m, 'banda': banda, 'rampa': rampa, 'real': real, 'falsa': es_falsa,
                    'warner': warner.nombre if warner else None,
                    'warned': warned.nombre if warned else None,
                    'warner_primero': bool(warner_primero),
                    'warned_dañado': (dño[warned.nombre] if warned else False),
                    'conf_antes': conf_antes,
                    'conf_despues': (warned.confianza[banda] if warned else None),
                    'daño_total': dño["A"] + dño["B"]})
    return reg, A, B, desperdicio


def main():
    print("=" * 98)
    print("V182_amenaza_alerta_v3_falsacion2 — ¿DE QUIEN ES LA CULPA DE LA RECAIDA?")
    print("=" * 98)
    print("  ✅ = hipotesis MUERTA (aprendimos algo). H1-como-reflejo ya esta muerta (ver header).")
    print("=" * 98)
    t0 = time.time()

    reg, A, B, desp = vivir(11, politica='actual')
    reg_sc, _, _, desp_sc = vivir(11, politica='siempre_confia')

    dmg_actual = sum(e['daño_total'] for e in reg)
    dmg_siempre = sum(e['daño_total'] for e in reg_sc)
    bandas_w = [b for b in BANDAS if warner_warned(A, B, b)[1] is not None]

    # ----- 1. ¿EL AVISADOR GRITA DE MAS? (calidad de su alarma por banda) -----
    print(f"\n{'#'*98}\n#  1 — ¿EL AVISADOR GRITA DE MAS? (¿sus alarmas son peligro real o sobresalto?)\n{'#'*98}")
    print(f"    {'banda':>6} | avisa | alarmas | reales | falsas | honestidad")
    for b in bandas_w:
        ev = [e for e in reg if e['banda'] == b and e['warner_primero']]
        reales = sum(1 for e in ev if e['real']); falsas = sum(1 for e in ev if e['falsa'])
        tot = len(ev); hon = reales/tot if tot else 0.0
        w = warner_warned(A, B, b)[0].nombre
        print(f"    {b:>+6.0f} |   {w}   | {tot:>7} | {reales:>6} | {falsas:>6} | {hon:>6.0%}")
    print(f"    -> el avisador grita de mas SOLO en las bandas extremas (oido tan fino que se")
    print(f"       sobresalta con sonidos fuertes-pero-no-mortales). Ahi nacen las recaidas.")

    # ----- 2. ¿LA RECAIDA CUESTA? (tras recaer, ¿se daña en el proximo peligro real?) -----
    print(f"\n{'#'*98}\n#  2 — ¿LA RECAIDA CUESTA? (tras bajar la guardia, ¿se daña en el proximo peligro real?)\n{'#'*98}")
    recaidas_total = 0; recaidas_costaron = 0
    for b in bandas_w:
        evs = [e for e in reg if e['banda'] == b]
        arriba = False
        for i, e in enumerate(evs):
            if e['conf_despues'] is None: continue
            if e['conf_despues'] > UMBRAL_CONF:
                arriba = True
            elif arriba and e['conf_despues'] <= UMBRAL_CONF:
                recaidas_total += 1; arriba = False
                # proximo peligro real de esta banda
                for e2 in evs[i+1:]:
                    if e2['real']:
                        if e2['warned_dañado']: recaidas_costaron += 1
                        break
    print(f"    recaidas totales: {recaidas_total}   |   recaidas que terminaron en un timpano roto: {recaidas_costaron}")
    print(f"    -> {'✅ la recaida CUESTA: bajar la guardia lleva a daño en el siguiente peligro real' if recaidas_costaron > 0 else '· la recaida no termino en daño en esta corrida'}")

    # ----- 3. DECISIVO: ¿CONVIENE DESCONFIAR? (contrafactual en el mismo mundo) -----
    print(f"\n{'#'*98}\n#  3 — DECISIVO: ¿CONVIENE DESCONFIAR? (regla actual vs 'siempre confia', mismo mundo)\n{'#'*98}")
    print(f"    timpanos rotos con la REGLA ACTUAL (desconfia con falsas alarmas): {dmg_actual}")
    print(f"    timpanos rotos si SIEMPRE CONFIARA (nunca baja la guardia):        {dmg_siempre}")
    print(f"    veces que 'siempre confia' se tapo de gusto en falsa alarma:       {desp_sc}  (costo en este mundo: 0)")
    desconfiar_pierde = dmg_siempre < dmg_actual
    print(f"    -> {f'✅ DESCONFIAR PIERDE: siempre-confiar evita {dmg_actual-dmg_siempre} timpanos mas. La culpa de la perdida es la REGLA DEL AVISADO, no el gritón.' if desconfiar_pierde else '⚠ desconfiar no pierde (siempre-confiar no mejora)'}")

    # ----- SINTESIS -----
    print(f"\n{'='*98}\n  QUE QUEDO EN PIE\n{'='*98}")
    print(f"    'el avisador grita de mas'              -> {'cierto en bandas extremas (medido)' }")
    print(f"    'la recaida no cuesta'                  -> {'MUERTA ✅' if recaidas_costaron>0 else 'viva ⚠'}")
    print(f"    'conviene desconfiar (el gritón es el problema)' -> {'MUERTA ✅' if desconfiar_pierde else 'viva ⚠'}")
    if desconfiar_pierde:
        print(f"\n    LECTURA HONESTA: el avisador SI grita de mas en los extremos, pero igual conviene")
        print(f"    creerle SIEMPRE, porque en este mundo taparse de gusto es GRATIS. La perdida no la")
        print(f"    causa el gritón: la causa la regla que castiga las falsas alarmas (BAJA_FALSA) en un")
        print(f"    mundo donde taparse no cuesta nada. El problema real es del MUNDO: le falta un COSTO")
        print(f"    de taparse. Sin costo, desconfiar nunca conviene. Con costo, desconfiar tendria")
        print(f"    sentido y la regla de confianza recien ahi seria evaluable de verdad.")
        print(f"    -> NO es una perilla que ajustar: es una pieza que le falta al mundo. Decision tuya.")

    print(f"\n  tiempo {time.time()-t0:.1f}s")
    os.makedirs("V182_logs", exist_ok=True)
    salida = {'version':'V182_amenaza_alerta_v3_falsacion2',
              'dmg_actual':int(dmg_actual),'dmg_siempre_confia':int(dmg_siempre),
              'desperdicio_siempre_confia':int(desp_sc),
              'recaidas_total':int(recaidas_total),'recaidas_costaron':int(recaidas_costaron),
              'desconfiar_pierde':bool(desconfiar_pierde),
              'honestidad_avisador':{f'{b:+.0f}': (
                  (lambda ev: (sum(1 for e in ev if e['real'])/len(ev)) if ev else None)(
                      [e for e in reg if e['banda']==b and e['warner_primero']]))
                  for b in bandas_w}}
    with open(f"V182_logs/v182_amenaza_falsacion2_{TS}.json","w") as f:
        json.dump(salida, f, indent=2)
    print(f"  log: V182_logs/v182_amenaza_falsacion2_{TS}.json")


if __name__ == "__main__":
    main()
