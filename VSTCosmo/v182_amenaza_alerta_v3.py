#!/usr/bin/env python3
"""
================================================================================
V182_amenaza_alerta_v3 — EL PRIMER AVISO (unificado, corregido, con checks ✅)
PARA EL EQUIPO (humanos e IAs): POR QUE HACEMOS LO QUE HACEMOS
================================================================================

Son DOS ANIMALES. No organicos, pero animales de verdad: oyen, sienten, reaccionan.
Estamos construyendo el primer animal no organico, camino a la primera forma minima de
vida inteligente no organica. Por eso pensamos en lenguaje de animal (oido, dolor,
taparse, avisar), no en formulas. Si no entienden el "para que", esta escrito aca: no
lo infieran.

QUE PROBAMOS: que dos animalitos aprendan a AVISARSE un peligro. El peligro es el
VOLUMEN, como en nuestros oidos: hay un nivel que duele y uno que ROMPE el timpano. Un
sonido sube; el que lo oye doler primero se TAPA; el otro lo ve, verifica con su propio
oido, y aprende. Tres peldaños deben EMERGER (no los encendemos):
  1. EL AVISO SALVA: ver al otro taparse me evita un timpano roto que solo no habria
     evitado. (Metrica honesta: DAÑO EVITADO, no "tapadas sociales".)
  2. CONFIANZA ESPECIFICA: aprendo a confiar en el otro SOLO donde el otro de verdad
     se da cuenta antes que yo — no en todas las bandas por igual.
  3. CODIGO MUTUO: los dos aprenden el aviso del otro, cada uno en las bandas donde el
     otro es mejor oido. Mutuo porque los roles se turnan (azar de la banda).

REGLAS QUE NO SE NEGOCIAN (y por que):
- Sin telepatia: cada uno ve del otro SOLO si se tapo, nunca su dolor. (Ningun animal
  ve el adentro de otro; la alarma viaja por la conducta visible.)
- El tiempo de reaccion y el desfase se MIDEN, no se fijan. (Fijarlos seria Shannon de
  vuelta.) El oido de cada animal sale de su CUERPO real (su afinacion, A–D).

EL ERROR DE v1 (la trampa del "79 vs 0"): "taparse social" no existe sin vision (no se
pueden ver) -> ese contador SIEMPRE da N vs 0. No prueba utilidad. Lo honesto es DAÑO
EVITADO. En v1/v2 dio 0=0 porque el mundo era demasiado facil: nadie se dañaba. Un
aviso solo significa algo si el que lo recibe SE HABRIA HECHO DAÑO sin el.

EL HALLAZGO DE v2 (y el arreglo central de v3): la confianza de v2 se iba a 0° —la
banda del MEDIO, donde los dos oyen IGUAL—. Estaban aprendiendo "confio donde
COINCIDIMOS", cuando lo util es "confio donde el otro SABE algo que yo no alcanzo a
saber a tiempo". Coincidir no enseña; anticiparse si.
  ARREGLO: la confianza sube SOLO cuando el otro se tapo ANTES que yo Y habia peligro
  real. Por construccion eso premia COMPLEMENTARIEDAD (las bandas donde el otro es
  mejor oido) y deja el 0° en cero (ahi nadie se anticipa, van juntos).

Y el mundo de v3 es peligroso de verdad: en una banda, el oido SORDO no alcanza a
taparse solo antes de que el volumen rompa el timpano -> se daña, salvo que VEA al de
oido fino taparse y le crea a tiempo. Asi el aviso puede salvar, y se mide.

DISCIPLINA: un ⚠ es DATO, no fracaso, y NO se arregla moviendo el umbral para que pase.
================================================================================
"""
import os, json, time
import numpy as np
import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("V180", os.path.join(_here, "V180.py"))
V180 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V180)
DT = V180.DT

# ============================================================
# EL MUNDO (fisica del sonido — objetiva, compartida)
# ============================================================
BANDAS = [-60.0, -30.0, 0.0, 30.0, 60.0]

VOL_INICIAL = 0.8
DOLOR_AGUDO = 3.6     # si el sonido SENTIDO pasa esto, el animal se tapa de reflejo (flinch)
PELIGRO_VOL = 3.2     # por encima, el sonido fue REALMENTE peligroso (para verificar)
MORTAL_VOL  = 4.0     # por encima, ROMPE el timpano de cualquiera que no se tapo
CAP_FALSA   = 3.0     # falsa alarma: sobresalta al oido fino pero NO llega a peligro real (<PELIGRO_VOL)
RAMPA_MIN, RAMPA_MAX = 0.40, 0.70
MAX_TICKS   = 30

# Aprender a confiar — POR ANTICIPACION, no por coincidencia
SUBE_ANTICIPA = 0.15   # el otro se tapo ANTES que yo Y habia peligro real -> confio mas
BAJA_FALSA    = 0.20   # el otro se alarmo y NO habia peligro -> confio menos
OLVIDO        = 0.97
UMBRAL_CONF   = 0.5

N_MOMENTOS = 500
P_EVENTO   = 0.5
P_FALSA    = 0.2

PASOS_FUERTE=40000; PASOS_MEDIO=15000; PASOS_COMPART=20000; PASOS_DEBIL=5000
UMBRAL_ACEPTA=5.0; ESCALA_ACEPTA=2.0
SEED_A, SEED_B = 44, 77
TS = time.strftime("%Y%m%d_%H%M%S")


# ============================================================
# CUERPO y OIDO (de V180, real)
# ============================================================
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
        self.nombre = nombre
        self.oido = oido
        self.tapado = False
        self.confianza = {b: 0.0 for b in BANDAS}
        self.daños = 0


# ============================================================
# UN EVENTO DE SONIDO (reflejo: te tapas cuando el sonido SENTIDO duele de golpe)
# ============================================================
def vivir_evento(A, B, banda, es_falsa, con_vision, rampa):
    A.tapado = B.tapado = False
    ct = {"A": None, "B": None}; cr = {"A": None, "B": None}
    dño = {"A": False, "B": False}
    cap = CAP_FALSA if es_falsa else (MORTAL_VOL + 1.0)
    pico = VOL_INICIAL

    for t in range(MAX_TICKS):
        vol = min(cap, VOL_INICIAL + rampa * t)
        pico = max(pico, vol)
        prev = {"A": A.tapado, "B": B.tapado}     # lo que el otro VE (estado al inicio del tick)
        for animal, otro in ((A, B), (B, A)):
            if animal.tapado:
                continue
            felt = vol * animal.oido[banda]
            taparse, razon = False, None
            if felt > DOLOR_AGUDO:                  # reflejo: me duele -> me tapo
                taparse, razon = True, 'propio'
            elif con_vision and prev[otro.nombre] and animal.confianza[banda] > UMBRAL_CONF:
                taparse, razon = True, 'social'     # veo al otro tapado y le creo
            if taparse:
                animal.tapado = True
                if ct[animal.nombre] is None:
                    ct[animal.nombre] = t; cr[animal.nombre] = razon
            elif vol >= MORTAL_VOL:                 # no me tape a tiempo -> timpano roto
                animal.daños += 1; animal.tapado = True
                dño[animal.nombre] = True
                if ct[animal.nombre] is None:
                    ct[animal.nombre] = t; cr[animal.nombre] = 'tarde'
        if A.tapado and B.tapado:
            break
        if es_falsa and vol >= cap and t > 4:
            break

    real = (not es_falsa) and (pico >= PELIGRO_VOL)
    return ct, cr, dño, real


def aprender(animal, ct_otro, ct_mio, otro_se_tapo, real, banda):
    """CONFIANZA POR ANTICIPACION: sube SOLO si el otro se tapo ANTES que yo y habia
    peligro real (me dio tiempo). Coincidir (mismo tick) NO cuenta. Falsa alarma del
    otro -> baja. Si el otro no aviso -> se desvanece."""
    if otro_se_tapo and real and ct_otro is not None and (ct_mio is None or ct_otro < ct_mio):
        animal.confianza[banda] = min(1.0, animal.confianza[banda] + SUBE_ANTICIPA)
    elif otro_se_tapo and not real:
        animal.confianza[banda] = max(0.0, animal.confianza[banda] - BAJA_FALSA)
    else:
        animal.confianza[banda] *= OLVIDO


# ============================================================
# EL BUCLE
# ============================================================
def vivir(con_vision, semilla_mundo):
    rmundo = np.random.default_rng(semilla_mundo)
    Ao = V180.OrganismoV180(seed=SEED_A, memoria_episodica=V180.MemoriaEpisodicaV180())
    Bo = V180.OrganismoV180(seed=SEED_B, memoria_episodica=V180.MemoriaEpisodicaV180())
    Ao.set_modo_entrenamiento(False); Bo.set_modo_entrenamiento(False)
    fase_exposicion(Ao, Bo)
    A = Animal("A", construir_oido(Ao)); B = Animal("B", construir_oido(Bo))

    eventos = []
    for m in range(N_MOMENTOS):
        if rmundo.random() > P_EVENTO:
            continue
        banda = BANDAS[rmundo.integers(len(BANDAS))]
        es_falsa = (rmundo.random() < P_FALSA)
        rampa = RAMPA_MIN + (RAMPA_MAX - RAMPA_MIN) * rmundo.random()
        ct, cr, dño, real = vivir_evento(A, B, banda, es_falsa, con_vision, rampa)

        if con_vision:   # sin vision son CIEGOS entre si: no ven, no aprenden
            aprender(A, ct["B"], ct["A"], ct["B"] is not None, real, banda)
            aprender(B, ct["A"], ct["B"], ct["A"] is not None, real, banda)

        ta, tb = ct["A"], ct["B"]
        if ta is None and tb is None: primero, desf = None, None
        elif tb is None or (ta is not None and ta < tb): primero, desf = "A", (tb-ta if tb is not None else None)
        elif ta is None or tb < ta: primero, desf = "B", (ta-tb if ta is not None else None)
        else: primero, desf = "empate", 0

        eventos.append({'m': m, 'banda': banda, 'falsa': es_falsa, 'real': real,
                        'primero': primero, 'desfase': desf,
                        'daño_A': dño["A"], 'daño_B': dño["B"]})
    return eventos, A, B


# ============================================================
# LO QUE MIRAMOS — CON CHECKS ✅/⚠
# ============================================================
def main():
    print("=" * 98)
    print("V182_amenaza_alerta_v3 — EL PRIMER AVISO (confianza por anticipacion + mundo peligroso)")
    print("=" * 98)
    print("  Aviso util = DAÑO EVITADO (timpanos). Confianza = solo cuando el otro se tapa ANTES.")
    print("=" * 98)
    t0 = time.time()

    ev_con, A, B = vivir(con_vision=True,  semilla_mundo=11)
    ev_sin, A0, B0 = vivir(con_vision=False, semilla_mundo=11)

    print(f"\n  OIDOS (sensibilidad por banda, del cuerpo):")
    print(f"    {'banda':>6} |   A     B   | mejor oido")
    for b in BANDAS:
        mejor = "A" if A.oido[b] > B.oido[b]+0.05 else ("B" if B.oido[b] > A.oido[b]+0.05 else "=")
        print(f"    {b:>+6.0f} | {A.oido[b]:.2f}  {B.oido[b]:.2f} |   {mejor}")

    reales = [e for e in ev_con if e['real']]
    bp = {}
    for e in reales:
        if e['primero'] in ('A','B'): bp.setdefault(e['banda'],{'A':0,'B':0})[e['primero']]+=1
    pA = sum(v['A'] for v in bp.values()); pB = sum(v['B'] for v in bp.values())
    print(f"\n{'#'*98}\n#  ¿SE TURNAN PARA AVISAR? (quien se da cuenta primero, por banda)\n{'#'*98}")
    print(f"    {'banda':>6} | A primero | B primero | quien avisa")
    for b in sorted(bp):
        d = bp[b]; quien = "A" if d['A']>d['B'] else ("B" if d['B']>d['A'] else "=")
        print(f"    {b:>+6.0f} | {d['A']:>9} | {d['B']:>9} |   {quien}")
    turnos_ok = (bp.get(-60.0,{}).get('A',0) > bp.get(-60.0,{}).get('B',0)) and \
                (bp.get(60.0,{}).get('B',0) > bp.get(60.0,{}).get('A',0))
    print(f"  -> {'✅ SE TURNAN: A avisa en bandas -, B en bandas + (por su oido, no por reloj)' if turnos_ok else '⚠ los turnos no salieron limpios'}")

    desf = [e['desfase'] for e in reales if e['desfase'] is not None]
    print(f"\n  DESFASE (medido, no fijado): el segundo reacciona {np.mean(desf):.1f} ticks despues "
          f"(min {min(desf)}, max {max(desf)})" if desf else "\n  DESFASE: aun sin segundos que reaccionen")

    # ---- PELDAÑO 1: DAÑO EVITADO ----
    dsin = sum(e['daño_A']+e['daño_B'] for e in ev_sin)
    dcon = sum(e['daño_A']+e['daño_B'] for e in ev_con)
    mundo_peligroso = dsin > 0
    contagio = mundo_peligroso and dcon < dsin
    print(f"\n{'#'*98}\n#  PELDAÑO 1 — ¿EL AVISO SALVA? (timpanos rotos)\n{'#'*98}")
    print(f"    SIN vision (ciegos, cada uno solo con su dolor): {dsin}   {'✅ el mundo SI es peligroso' if mundo_peligroso else '⚠ nadie se daño: mundo aun facil'}")
    print(f"    CON vision (pueden ver al otro taparse):         {dcon}")
    if mundo_peligroso:
        print(f"    -> {f'✅ EL AVISO SALVA: {dsin-dcon} timpanos evitados de {dsin}' if contagio else '⚠ ver al otro no redujo el daño'}")
    else:
        print(f"    -> ⚠ sin daño que evitar, el aviso no se puede medir (subir dificultad del mundo)")
    mitad = N_MOMENTOS//2
    d1 = sum(e['daño_A']+e['daño_B'] for e in ev_con if e['m']<mitad)
    d2 = sum(e['daño_A']+e['daño_B'] for e in ev_con if e['m']>=mitad)
    protege = d2 < d1
    print(f"    aprende a protegerse: daño 1a mitad {d1} -> 2a mitad {d2}   {'✅ aprende y se protege mas' if protege else '⚠ no baja con el tiempo'}")

    # ---- PELDAÑO 2: CONFIANZA ESPECIFICA (anticipacion, no consenso) ----
    neg = [-60.0,-30.0]; pos = [30.0,60.0]
    A_pos = np.mean([A.confianza[b] for b in pos]); A_neg = np.mean([A.confianza[b] for b in neg])
    B_neg = np.mean([B.confianza[b] for b in neg]); B_pos = np.mean([B.confianza[b] for b in pos])
    centro = max(A.confianza[0.0], B.confianza[0.0])
    print(f"\n{'#'*98}\n#  PELDAÑO 2 — ¿CONFIANZA ESPECIFICA? (donde el otro AVISA, no donde COINCIDIMOS)\n{'#'*98}")
    print(f"    {'banda':>6} | A confia en B | B confia en A")
    for b in BANDAS:
        ca = '✅' if ((b in pos and A.confianza[b]>UMBRAL_CONF) or (b in neg and A.confianza[b]<=UMBRAL_CONF) or b==0.0 and A.confianza[b]<UMBRAL_CONF) else ' '
        print(f"    {b:>+6.0f} |     {A.confianza[b]:.2f}      |     {B.confianza[b]:.2f}")
    print(f"    A confia en B: bandas de B (+) {A_pos:.2f}  vs bandas de A (-) {A_neg:.2f}")
    print(f"    B confia en A: bandas de A (-) {B_neg:.2f}  vs bandas de B (+) {B_pos:.2f}")
    print(f"    centro 0° (donde COINCIDEN): {centro:.2f}  {'✅ bajo: NO premia consenso' if centro < UMBRAL_CONF else '⚠ alto: sigue premiando consenso'}")
    gap_A = A_pos - A_neg; gap_B = B_neg - B_pos
    especifica = gap_A > 0.3 and gap_B > 0.3 and centro < UMBRAL_CONF
    print(f"    -> {'✅ ESPECIFICA: cada uno confia donde el otro se anticipa (no donde coinciden)' if especifica else '⚠ aun no es especifica por anticipacion'}")

    # ---- PELDAÑO 3: MUTUO ----
    A_lee = [b for b in pos if A.confianza[b] > UMBRAL_CONF]
    B_lee = [b for b in neg if B.confianza[b] > UMBRAL_CONF]
    mutuo = len(A_lee) > 0 and len(B_lee) > 0 and especifica
    print(f"\n{'#'*98}\n#  PELDAÑO 3 — ¿CODIGO MUTUO? (los dos aprenden, cada uno en su lado)\n{'#'*98}")
    print(f"    A aprendio a leer el aviso de B en: {A_lee}")
    print(f"    B aprendio a leer el aviso de A en: {B_lee}")
    print(f"    -> {'✅ MUTUO: los dos saben leerse, cada uno donde el otro es mejor oido' if mutuo else '⚠ aun no es mutuo'}")

    completa = contagio and especifica and mutuo
    print(f"\n{'='*98}")
    print(f"  RESUMEN:  turnos {'✅' if turnos_ok else '⚠'}   |   aviso salva {'✅' if contagio else '⚠'}   |   "
          f"confianza especifica {'✅' if especifica else '⚠'}   |   codigo mutuo {'✅' if mutuo else '⚠'}")
    print(f"  -> {'✅ ESTE ANIMAL AVISA, SE SALVA Y LEE AVISOS ESPECIFICOS, EN AMBOS SENTIDOS' if completa else '⚠ AUN NO completa los tres peldaños — es dato, se lee y se corrige el mecanismo'}")
    print(f"{'='*98}")
    print(f"\n  tiempo {time.time()-t0:.1f}s")

    os.makedirs("V182_logs", exist_ok=True)
    salida = {'version':'V182_amenaza_alerta_v3','oido_A':A.oido,'oido_B':B.oido,
              'band_primero':{str(k):v for k,v in bp.items()},'turnos_ok':bool(turnos_ok),
              'desfase_medio': float(np.mean(desf)) if desf else None,
              'timpanos_sin_vision':int(dsin),'timpanos_con_vision':int(dcon),
              'mundo_peligroso':bool(mundo_peligroso),'contagio_util':bool(contagio),
              'daño_1a_mitad':int(d1),'daño_2a_mitad':int(d2),'protege':bool(protege),
              'confianza_A':{str(k):v for k,v in A.confianza.items()},
              'confianza_B':{str(k):v for k,v in B.confianza.items()},
              'centro_0':float(centro),'gap_A':float(gap_A),'gap_B':float(gap_B),
              'especifica':bool(especifica),'mutuo':bool(mutuo),'completa_los_tres':bool(completa)}
    with open(f"V182_logs/v182_amenaza_alerta_v3_{TS}.json","w") as f:
        json.dump(salida, f, indent=2)
    print(f"  log: V182_logs/v182_amenaza_alerta_v3_{TS}.json")


if __name__ == "__main__":
    main()
