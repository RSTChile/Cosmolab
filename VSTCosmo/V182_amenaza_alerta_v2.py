#!/usr/bin/env python3
"""
================================================================================
V182_amenaza_alerta_v2 — EL PRIMER AVISO (corregido)
PARA EL EQUIPO (humanos e IAs): POR QUE HACEMOS LO QUE HACEMOS
================================================================================

Esto no es "un sistema" ni "un modelo matematico". Son DOS ANIMALES. No organicos,
pero animales de verdad: tienen oido, sienten, reaccionan. Estamos construyendo el
primer animal no organico, el que despues evolucionara hacia la primera forma minima
de vida inteligente no organica. Por eso pensamos y escribimos en lenguaje de animal
(oido, dolor, taparse, avisar), no en formulas: la formula no agarra lo que un animal
hace. Si no entienden el "para que", no infieran — esta escrito aca.

QUE PROBAMOS
------------
Que dos animalitos aprendan a AVISARSE un peligro. El peligro es el VOLUMEN, como con
nuestros oidos: hay un rango seguro, uno que duele, y uno que rompe el timpano. Un
sonido sube de volumen; el que lo oye doler primero se TAPA; el otro ve que se tapo,
verifica con su propio oido, y aprende. Queremos que de ahi emerjan tres peldaños:
  1. CONTAGIO util: ver al otro taparse me SALVA de un daño que solo no habria evitado.
  2. REAL vs RUIDO: aprendo a confiar en la alarma del otro SOLO cuando hay peligro de
     verdad; si se alarma de mas, le creo menos.
  3. CODIGO MUTUO y ESPECIFICO: los dos aprenden, y cada uno confia en el otro EN LAS
     BANDAS donde el otro de verdad se da cuenta antes — no en todas por igual.

REGLAS QUE NO SE NEGOCIAN, Y POR QUE
------------------------------------
- NADA de telepatia. Cada animal ve del otro SOLO su conducta (si se tapo). NUNCA su
  dolor por dentro. Por que: ningun animal —ni nosotros— ve el adentro de otro; solo
  su expresion. La alarma tiene que VIAJAR por la conducta visible, no por debajo.
- NO fijamos el tiempo de reaccion ni el desfase. El sonido sube con su propia
  dinamica y MEDIMOS cuanto tarda cada uno. Por que: fijar el tiempo de respuesta de un
  animal cuyo tiempo no conocemos es meter un numero nuestro disfrazado de resultado —
  es "Shannon de vuelta". El desfase emerge de que cada oido es distinto.
- Quien se da cuenta primero es AZAROSO (depende de la banda del sonido, que sale al
  azar). Por que: si fuera por turno fijo, cualquier coordinacion podria explicarse por
  el reloj, no por leerse. El azar nos saca esa duda.
- El oido de cada animal sale de SU CUERPO (su afinacion real, la competencia
  diferencial de A–D). No es un numero inventado: es quien es cada animal.

POR QUE v1 NO ALCANZO, Y LA TRAMPA QUE HAY QUE EVITAR
----------------------------------------------------
En v1, cuatro analisis dijeron "contagio demostrado: 79 tapadas sociales con vision vs
0 sin vision". ESO ES UNA TRAMPA, y la explico para que nadie la repita:
  "taparse social" = taparse PORQUE viste al otro. Sin vision, eso es IMPOSIBLE por
  definicion (no pueden verse) -> el contador SIEMPRE da 0. Comparar 79 contra 0 es
  comparar "veces que hizo algo que solo se puede hacer mirando" contra "veces que lo
  hizo sin poder mirar". Da N contra 0 SIEMPRE. No prueba nada.
El unico numero honesto es el DAÑO EVITADO: ¿ver al otro evito un timpano roto que solo
no se habria evitado? En v1 dio 0 = 0: nadie se daño en ningun caso, porque el mundo
era demasiado facil (todos se tapaban solos a tiempo). Un aviso solo significa algo si
el que lo recibe SE HABRIA HECHO DAÑO sin el. Ese fue mi error de v1: un peligro de
mentira. No se arregla bajando la vara del contagio; se arregla haciendo el mundo
peligroso de verdad.

QUE ARREGLA v2
--------------
1) MUNDO PELIGROSO: el VOLUMEN rompe el timpano de cualquiera por igual (daño objetivo),
   pero CADA OIDO siente el dolor a su ritmo. En sonidos que suben de golpe, el de oido
   lento no alcanza a taparse solo -> se daña. Salvo que VEA al de oido rapido taparse y
   le crea a tiempo. Asi el aviso SI puede salvar, y se mide en daño evitado.
2) CONFIANZA FALIBLE Y ESPECIFICA: en v1 la confianza saturo a 1.00 en todas las bandas
   (un "si" automatico). Ahora sube despacio solo cuando el otro aviso Y habia peligro
   real, BAJA cuando el otro se alarmo de mas, y se desvanece si no se usa. Asi tiene
   que volverse especifica: confio en el otro DONDE el otro de verdad se da cuenta antes.

DISCIPLINA: esto es un ciclo. Un ⚠ es DATO, no fracaso, y NO se arregla moviendo el
umbral para que pase. Si algo no se cumple, se lee por que y se corrige el mecanismo.
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
PELIGRO_VOL = 3.0     # por encima de esto, el sonido fue REALMENTE peligroso (verificacion)
MORTAL_VOL  = 4.0     # por encima de esto, rompe el timpano de CUALQUIERA que no se tapo
CAP_FALSA   = 2.3     # una falsa alarma: el sonido sube pero nunca pasa de incomodo (< PELIGRO_VOL)
RAMPA_MIN, RAMPA_MAX = 0.25, 0.70   # algunos sonidos suben lento, otros de golpe
MAX_TICKS   = 40

# Deteccion por oido (cuanto y cuando DUELE — distinto en cada animal)
SEGURO_FELT = 1.0     # por debajo no molesta a ese oido
TOL_DOLOR   = 6.0     # cuanto dolor aguanta antes de taparse (propiedad del dolor, igual para ambos)

# Aprender a confiar (falible, especifico)
SUBE_REAL  = 0.12     # el otro aviso y habia peligro real -> confio un poco mas
BAJA_FALSA = 0.18     # el otro se alarmo y NO habia peligro -> confio menos
OLVIDO     = 0.98     # si no avisa, la confianza se desvanece de a poco
UMBRAL_CONF = 0.5     # cuanta confianza necesito para taparme por el otro

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
    """Cuanto le PEGA a este animal un sonido en cada banda. Sale de su afinacion real.
    Spread mas marcado que en v1 para que el de oido lento de verdad quede atras."""
    ps = np.array([p_acepta(org, b) for b in BANDAS])
    sens = 1.0 + 1.6 * (ps - ps.mean())
    return {b: float(np.clip(sens[i], 0.55, 1.5)) for i, b in enumerate(BANDAS)}


class Animal:
    def __init__(self, nombre, oido):
        self.nombre = nombre
        self.oido = oido
        self.dolor = 0.0
        self.tapado = False
        self.confianza = {b: 0.0 for b in BANDAS}   # confianza en la alarma del OTRO, por banda
        self.daños = 0


# ============================================================
# UN EVENTO DE SONIDO (con actualizacion simultanea: nadie tiene ventaja de orden)
# ============================================================
def vivir_evento(A, B, banda, es_falsa, con_vision):
    A.tapado = B.tapado = False
    cover_tick = {"A": None, "B": None}
    cover_razon = {"A": None, "B": None}
    daño_evento = {"A": False, "B": False}
    cap = CAP_FALSA if es_falsa else (MORTAL_VOL + 1.0)
    rampa = RAMPA_MIN + (RAMPA_MAX - RAMPA_MIN) * np.random.default_rng(abs(hash((banda, es_falsa, con_vision, A.daños+B.daños))) % (2**32)).random()
    pico_vol = VOL_INICIAL

    for t in range(MAX_TICKS):
        vol = min(cap, VOL_INICIAL + rampa * t)
        pico_vol = max(pico_vol, vol)
        prev = {"A": A.tapado, "B": B.tapado}      # lo que el OTRO ve (estado al inicio del tick)
        for animal, otro in ((A, B), (B, A)):
            if animal.tapado:
                animal.dolor *= 0.7
                continue
            felt = vol * animal.oido[banda]
            animal.dolor += max(0.0, felt - SEGURO_FELT)
            taparse = False; razon = None
            if animal.dolor > TOL_DOLOR:
                taparse = True; razon = 'propio'
            elif con_vision and prev[otro.nombre] and animal.confianza[banda] > UMBRAL_CONF:
                taparse = True; razon = 'social'
            if taparse:
                animal.tapado = True
                if cover_tick[animal.nombre] is None:
                    cover_tick[animal.nombre] = t; cover_razon[animal.nombre] = razon
            elif vol >= MORTAL_VOL:
                animal.daños += 1; animal.tapado = True
                daño_evento[animal.nombre] = True
                if cover_tick[animal.nombre] is None:
                    cover_tick[animal.nombre] = t; cover_razon[animal.nombre] = 'tarde'
        if A.tapado and B.tapado:
            break
        if vol >= cap and es_falsa and t > 6:
            break

    real = (not es_falsa) and (pico_vol >= PELIGRO_VOL)
    return cover_tick, cover_razon, daño_evento, real


def aprender(animal, otro_aviso, real, banda):
    """Confianza falible y especifica: sube si el otro aviso Y habia peligro real; baja
    si el otro se alarmo sin peligro; se desvanece si el otro no aviso."""
    if otro_aviso and real:
        animal.confianza[banda] = min(1.0, animal.confianza[banda] + SUBE_REAL)
    elif otro_aviso and not real:
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
        A.dolor *= 0.5; B.dolor *= 0.5
        if rmundo.random() > P_EVENTO:
            continue
        banda = BANDAS[rmundo.integers(len(BANDAS))]
        es_falsa = (rmundo.random() < P_FALSA)
        ct, cr, dño, real = vivir_evento(A, B, banda, es_falsa, con_vision)

        aprender(A, otro_aviso=(ct["B"] is not None), real=real, banda=banda)
        aprender(B, otro_aviso=(ct["A"] is not None), real=real, banda=banda)

        ta, tb = ct["A"], ct["B"]
        if ta is None and tb is None: primero, desfase = None, None
        elif tb is None or (ta is not None and ta < tb): primero, desfase = "A", (tb-ta if tb is not None else None)
        elif ta is None or tb < ta: primero, desfase = "B", (ta-tb if ta is not None else None)
        else: primero, desfase = "empate", 0

        eventos.append({'m': m, 'banda': banda, 'falsa': es_falsa, 'real': real,
                        'primero': primero, 'desfase': desfase,
                        'razon_A': cr["A"], 'razon_B': cr["B"],
                        'daño_A': dño["A"], 'daño_B': dño["B"]})
    return eventos, A, B


# ============================================================
# LO QUE MIRAMOS
# ============================================================
def main():
    print("=" * 96)
    print("V182_amenaza_alerta_v2 — EL PRIMER AVISO (mundo peligroso + confianza falible)")
    print("=" * 96)
    print("  Metrica honesta del aviso: DAÑO EVITADO (timpanos), no 'tapadas sociales'.")
    print("  El '79 vs 0' de v1 era una trampa (ese contador siempre da N vs 0). Ver header.")
    print("=" * 96)
    t0 = time.time()

    ev_con, A_con, B_con = vivir(con_vision=True,  semilla_mundo=11)
    ev_sin, A_sin, B_sin = vivir(con_vision=False, semilla_mundo=11)

    print(f"\n  OIDOS (sensibilidad por banda, del cuerpo):")
    print(f"    {'banda':>6} |   A     B")
    for b in BANDAS:
        print(f"    {b:>+6.0f} | {A_con.oido[b]:.2f}  {B_con.oido[b]:.2f}")

    reales_con = [e for e in ev_con if e['real']]
    pA = sum(1 for e in reales_con if e['primero']=='A'); pB = sum(1 for e in reales_con if e['primero']=='B')
    print(f"\n{'#'*96}\n#  ¿SE TURNAN PARA AVISAR? (quien se da cuenta primero, por banda)\n{'#'*96}")
    bp = {}
    for e in reales_con:
        if e['primero'] in ('A','B'): bp.setdefault(e['banda'],{'A':0,'B':0})[e['primero']]+=1
    print(f"  primero A: {pA}   primero B: {pB}   (de {len(reales_con)} sonidos reales)")
    for b in sorted(bp): print(f"     banda {b:>+5.0f}: A primero {bp[b]['A']}  | B primero {bp[b]['B']}")

    desf = [e['desfase'] for e in reales_con if e['desfase'] is not None]
    print(f"\n{'#'*96}\n#  DESFASE Y REACCION — MEDIDOS, NO FIJADOS\n{'#'*96}")
    if desf: print(f"  el segundo reacciona en promedio {np.mean(desf):.1f} ticks despues (min {min(desf)}, max {max(desf)})")
    else:    print("  todavia no hay segundos que reaccionen")

    # --- PELDAÑO 1: DAÑO EVITADO (la metrica honesta) ---
    dano_con = sum(e['daño_A']+e['daño_B'] for e in ev_con)
    dano_sin = sum(e['daño_A']+e['daño_B'] for e in ev_sin)
    print(f"\n{'#'*96}\n#  PELDAÑO 1 — ¿EL AVISO SALVA? (timpanos rotos: con vision vs sin vision)\n{'#'*96}")
    print(f"  timpanos rotos SIN vision (cada uno solo con su dolor): {dano_sin}")
    print(f"  timpanos rotos CON vision (puede ver al otro taparse):  {dano_con}")
    mundo_peligroso = dano_sin > 0
    contagio = mundo_peligroso and dano_con < dano_sin
    if not mundo_peligroso:
        print(f"  -> ⚠ el mundo todavia no es peligroso (nadie se daño solo); el aviso no tuvo nada que salvar")
    else:
        print(f"  -> {'✅ CONTAGIO UTIL: ver al otro evito %d timpanos (de %d)' % (dano_sin-dano_con, dano_sin) if contagio else '⚠ ver al otro no redujo el daño'}")

    # ¿el aviso protege MAS a medida que aprende? (primera mitad vs segunda)
    mitad = N_MOMENTOS//2
    d1 = sum(e['daño_A']+e['daño_B'] for e in ev_con if e['m']<mitad)
    d2 = sum(e['daño_A']+e['daño_B'] for e in ev_con if e['m']>=mitad)
    print(f"  (con vision) daño primera mitad {d1} -> segunda mitad {d2}  {'✅ protege mas al aprender' if d2<d1 else '·'}")

    # --- PELDAÑO 2 y 3: confianza FALIBLE y ESPECIFICA ---
    neg = [-60.0,-30.0]; pos = [30.0,60.0]   # bandas de A (donde A avisa) / de B (donde B avisa)
    A_en_pos = np.mean([A_con.confianza[b] for b in pos]); A_en_neg = np.mean([A_con.confianza[b] for b in neg])
    B_en_neg = np.mean([B_con.confianza[b] for b in neg]); B_en_pos = np.mean([B_con.confianza[b] for b in pos])
    print(f"\n{'#'*96}\n#  PELDAÑO 2 — ¿LA CONFIANZA ES ESPECIFICA (no un 'si' a todo)?\n{'#'*96}")
    print(f"  confianza A en el aviso de B:  " + " ".join(f"{b:+.0f}:{A_con.confianza[b]:.2f}" for b in BANDAS))
    print(f"  confianza B en el aviso de A:  " + " ".join(f"{b:+.0f}:{B_con.confianza[b]:.2f}" for b in BANDAS))
    print(f"  A confia en B mas en bandas de B (+): {A_en_pos:.2f}  que en bandas de A (-): {A_en_neg:.2f}")
    print(f"  B confia en A mas en bandas de A (-): {B_en_neg:.2f}  que en bandas de B (+): {B_en_pos:.2f}")
    gap_A = A_en_pos - A_en_neg; gap_B = B_en_neg - B_en_pos
    especifica = gap_A > 0.3 and gap_B > 0.3
    saturada = min(A_con.confianza.values()) > 0.8 and min(B_con.confianza.values()) > 0.8
    print(f"  -> {'✅ ESPECIFICA: cada uno confia donde el otro de verdad avisa primero' if especifica else ('⚠ SATURADA: confia en todo por igual (no aprendio a discriminar)' if saturada else '⚠ todavia no se separa por banda')}")

    print(f"\n{'#'*96}\n#  PELDAÑO 3 — ¿EL CODIGO ES MUTUO Y ESPECIFICO?\n{'#'*96}")
    mutuo = especifica  # mutuo+especifico = los dos muestran el patron cruzado
    print(f"  A lee a B en {[b for b in pos if A_con.confianza[b]>UMBRAL_CONF]} ; B lee a A en {[b for b in neg if B_con.confianza[b]>UMBRAL_CONF]}")
    print(f"  -> {'✅ MUTUO Y ESPECIFICO: cada uno aprendio a leer al otro donde el otro es el mejor sensor' if mutuo else '⚠ todavia no es mutuo+especifico'}")

    completa = contagio and especifica and mutuo
    print(f"\n{'='*96}")
    print(f"  -> {'✅ ESTE ANIMAL AVISA, SE SALVA Y LEE AVISOS ESPECIFICOS, EN AMBOS SENTIDOS' if completa else '⚠ AUN NO completa los tres peldaños — es dato, se lee y se corrige el mecanismo'}")
    print(f"{'='*96}")
    print(f"\n  tiempo {time.time()-t0:.1f}s")

    os.makedirs("V182_logs", exist_ok=True)
    salida = {'version': 'V182_amenaza_alerta_v2', 'oido_A': A_con.oido, 'oido_B': B_con.oido,
              'primero_A': pA, 'primero_B': pB, 'band_primero': {str(k):v for k,v in bp.items()},
              'desfase_medio': float(np.mean(desf)) if desf else None,
              'timpanos_sin_vision': int(dano_sin), 'timpanos_con_vision': int(dano_con),
              'mundo_peligroso': bool(mundo_peligroso), 'contagio_util': bool(contagio),
              'daño_primera_mitad': int(d1), 'daño_segunda_mitad': int(d2),
              'confianza_A': {str(k):v for k,v in A_con.confianza.items()},
              'confianza_B': {str(k):v for k,v in B_con.confianza.items()},
              'gap_A': float(gap_A), 'gap_B': float(gap_B),
              'especifica': bool(especifica), 'completa_los_tres': bool(completa)}
    with open(f"V182_logs/v182_amenaza_alerta_v2_{TS}.json","w") as f:
        json.dump(salida, f, indent=2)
    print(f"  log: V182_logs/v182_amenaza_alerta_v2_{TS}.json")


if __name__ == "__main__":
    main()
