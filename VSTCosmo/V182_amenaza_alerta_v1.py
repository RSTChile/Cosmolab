#!/usr/bin/env python3
"""
V182_amenaza_alerta_v1 — EL PRIMER AVISO: DOS ANIMALES QUE SE ADVIERTEN UN PELIGRO
================================================================================

QUE ES ESTO (en lenguaje de animal, no de modelo)
-------------------------------------------------
Dos animalitos que ESCUCHAN. No son organicos, pero son animales: tienen oido,
sienten, reaccionan. (El cuerpo es V180, que literalmente procesa sonido.)

En el aire, a veces un sonido empieza a SUBIR DE VOLUMEN hasta volverse dañino —el
mismo para los dos, porque comparten el aire—. Como ningun oido es igual, a uno le
empieza a doler ANTES y se TAPA las orejas; el otro lo ve taparse, le presta atencion
a su PROPIO oido, y comprueba que a el tambien le duele, un poco despues. Quien oye el
dolor primero depende del sonido, asi que SE TURNAN.

De ese taparse-y-verificar, repetido y en ambos sentidos, queremos que emerjan solos
TRES peldaños de comunicacion —no los encendemos nosotros, los miramos aparecer—:
  1. CONTAGIO: ver al otro taparse me hace taparme antes de lo que me taparia solo.
  2. REAL vs RUIDO: aprendo a confiar en la alarma del otro solo cuando MI propio
     oido la confirma (no por fe; por verificacion). El umbral de "esto va en serio"
     NO lo pongo yo: emerge de que las dos experiencias coincidan.
  3. EL PELIGRO ES ESE: aprendo que cuando el otro se tapa ante un sonido de tal
     banda, esa banda es peligrosa, y me anticipo. Y como los roles se turnan, los
     DOS terminan sabiendolo: el codigo compartido es mutuo.

REGLAS INNEGOCIABLES (lo que hace que esto sea honesto y no trampa)
------------------------------------------------------------------
- NADA de telepatia: cada animal ve del otro SOLO su conducta (si se tapo, y de que
  banda viene el sonido —que esta en el aire compartido—). NUNCA su dolor por dentro.
- La inferencia que A hace de B es interna de A y PUEDE estar equivocada.
- NO fijamos el desfase ni el tiempo de reaccion: el sonido SUBE con su propia
  dinamica, y CUANTO tarda cada oido en que le duela, y cuanto tarda el segundo en
  reaccionar, se MIDE. Fijarlo seria Shannon de vuelta.
- Quien se da cuenta primero es AZAROSO (depende de la banda del sonido, que es
  aleatoria), no por turno fijo: asi cualquier coordinacion no puede venir del reloj.
- El oido de cada animal —cuanto le pega un sonido en cada banda— sale de su PROPIO
  cuerpo (su valencia/afinacion real, la misma competencia diferencial de A–D). No es
  un numero inventado: es quien es cada animal.

QUE ES PELIGRO: el VOLUMEN. Como con nuestros oidos: hay un rango seguro, uno
riesgoso, uno que duele, y uno que rompe el timpano. Eso no se impone: es lo que le
pasa a cualquier cosa que oye.

ESTO ES UN PRIMER CICLO (v1). Puede que algun peldaño todavia no emerja. Si pasa, NO
es fracaso: es el dato de que el animal todavia no llega ahi, y se lee.

NOTA DE MAPA: su lugar en el roadmap (y el lio de nombre V182E/negociacion) sigue
pendiente de decidir. Aca solo construimos el animal.
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
# EL MUNDO (fisica del sonido — del ambiente, no de los animales)
# ============================================================
BANDAS = [-60.0, -30.0, 0.0, 30.0, 60.0]   # de donde puede venir un sonido

# Niveles de volumen SENTIDO (como los dB: propiedad del sonido, compartida)
SEGURO  = 1.0    # por debajo no molesta
RIESGO  = 2.0    # empieza a incomodar
PELIGRO = 3.0    # duele: hay que taparse
MORTAL  = 4.5    # rompe el timpano: daño permanente si no te tapaste

VOL_INICIAL = 0.8     # un sonido empieza bajito
VOL_RAMPA   = 0.12    # y SUBE asi de rapido por tick (dinamica del ambiente)
MAX_TICKS_EVENTO = 60 # cuanto dura un sonido como mucho

N_MOMENTOS   = 400    # cuantos instantes vive el animal
P_EVENTO     = 0.45   # con que frecuencia aparece un sonido fuerte
P_FALSA      = 0.15   # a veces uno se sobresalta sin peligro real (falsa alarma)

# Dolor: cuanto aguanta antes de taparse. Es propiedad del DOLOR, igual para ambos;
# el desfase NO sale de aca, sale de que cada OIDO siente distinto el mismo volumen.
TOL_DOLOR = 6.0

# Exposicion (da a cada animal su oido: su afinacion por banda, del cuerpo real)
PASOS_FUERTE=40000; PASOS_MEDIO=15000; PASOS_COMPART=20000; PASOS_DEBIL=5000
UMBRAL_ACEPTA=5.0; ESCALA_ACEPTA=2.0

SEED_A, SEED_B = 44, 77
TS = time.strftime("%Y%m%d_%H%M%S")
rng = np.random.default_rng(2026)


# ============================================================
# EL CUERPO y EL OIDO (de V180, real)
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
    """El oido: cuanto le PEGA a este animal un sonido en cada banda. Sale de su propia
    afinacion (valencia). La banda en que es experto la oye mas fuerte -> le duele
    antes. Centrado en ~1 y nunca chico, para que un sonido fuerte termine doliendo a
    cualquier oido (solo que al sensible le duele primero)."""
    ps = np.array([p_acepta(org, b) for b in BANDAS])
    centrado = ps - ps.mean()
    sens = 1.0 + 1.1 * centrado               # mas afinado -> mas sensible
    return {b: float(np.clip(sens[i], 0.6, 1.6)) for i, b in enumerate(BANDAS)}


# ============================================================
# UN ANIMAL (su estado en el bucle de amenaza)
# ============================================================
class Animal:
    def __init__(self, nombre, oido):
        self.nombre = nombre
        self.oido = oido                  # sensibilidad por banda (del cuerpo)
        self.dolor = 0.0
        self.tapado = False
        self.confianza = {b: 0.0 for b in BANDAS}   # cuanto confio en la alarma del otro, por banda (SE APRENDE)
        self.daños = 0                    # timpanos rotos (llego a MORTAL sin taparse)

    def reset_evento(self):
        self.tapado = False


# ============================================================
# UN EVENTO DE SONIDO (sube el volumen; los animales reaccionan tick a tick)
# ============================================================
def vivir_evento(A, B, banda, es_falsa, con_vision):
    """Devuelve el registro de quien se tapo primero, el desfase, si el segundo
    llego a tiempo, y la verificacion de cada uno. Cada animal solo VE si el otro
    esta tapado; nunca su dolor."""
    A.reset_evento(); B.reset_evento()
    vol = VOL_INICIAL
    tap_tick = {A.nombre: None, B.nombre: None}
    tap_razon = {A.nombre: None, B.nombre: None}
    # un evento "falso": hay un sonido que sobresalta pero NO sube a peligro
    techo = (RIESGO * 0.8) if es_falsa else (MORTAL + 1.0)

    for t in range(MAX_TICKS_EVENTO):
        for animal, otro in ((A, B), (B, A)):
            sentido = vol * animal.oido[banda]          # cuanto le suena a ESTE oido
            if not animal.tapado:
                # dolor sube si el sonido sentido pasa lo seguro
                animal.dolor += max(0.0, sentido - SEGURO)
                # timpano: si el sonido sentido llega a MORTAL sin taparse -> daño
                if sentido >= MORTAL:
                    animal.daños += 1
                    animal.tapado = True            # el daño lo obliga a taparse (tarde)
                    if tap_tick[animal.nombre] is None:
                        tap_tick[animal.nombre] = t; tap_razon[animal.nombre] = 'tarde'
                    continue
                # decision de taparse:
                # (a) por dolor propio
                if animal.dolor > TOL_DOLOR:
                    animal.tapado = True
                    tap_tick[animal.nombre] = t; tap_razon[animal.nombre] = 'propio'
                # (b) social: VE al otro tapado y CONFIA en su alarma para esta banda
                elif con_vision and otro.tapado and animal.confianza[banda] > 0.5:
                    animal.tapado = True
                    tap_tick[animal.nombre] = t; tap_razon[animal.nombre] = 'social'
            else:
                animal.dolor *= 0.7   # tapado: el dolor afloja
        vol += VOL_RAMPA
        if vol > techo and A.tapado and B.tapado:
            break
        if vol > techo:               # falsa alarma: el sonido nunca paso de riesgo
            break

    # ¿hubo peligro REAL para cada uno? (su oido sintio pasar PELIGRO en algun momento)
    pico = {A.nombre: VOL_INICIAL + VOL_RAMPA*MAX_TICKS_EVENTO, B.nombre: VOL_INICIAL + VOL_RAMPA*MAX_TICKS_EVENTO}
    real = {}
    for animal in (A, B):
        vol_max_sentido = (techo) * animal.oido[banda]
        real[animal.nombre] = (not es_falsa) and (vol_max_sentido >= PELIGRO)

    return tap_tick, tap_razon, real


def verificar_y_aprender(animal, banda, se_tapo_social, hubo_peligro_real):
    """Si me tape por ver al otro (social), VERIFICO con mi propio oido: ¿de verdad
    habia peligro? Si si, confio mas en el para esta banda; si no, confio menos.
    El umbral de 'confiar' NO lo puse yo: emerge de estas confirmaciones."""
    if se_tapo_social:
        if hubo_peligro_real:
            animal.confianza[banda] = min(1.0, animal.confianza[banda] + 0.25)
        else:
            animal.confianza[banda] = max(0.0, animal.confianza[banda] - 0.30)
    else:
        # tambien aprende del puro mirar: si el otro se tapo y a mi me dolio de verdad,
        # la proxima vez convendra hacerle caso antes (refuerzo suave)
        if hubo_peligro_real:
            animal.confianza[banda] = min(1.0, animal.confianza[banda] + 0.10)


# ============================================================
# EL BUCLE: la vida de los dos animales
# ============================================================
def vivir(con_vision, semilla_mundo):
    rmundo = np.random.default_rng(semilla_mundo)
    A_org = V180.OrganismoV180(seed=SEED_A, memoria_episodica=V180.MemoriaEpisodicaV180())
    B_org = V180.OrganismoV180(seed=SEED_B, memoria_episodica=V180.MemoriaEpisodicaV180())
    A_org.set_modo_entrenamiento(False); B_org.set_modo_entrenamiento(False)
    fase_exposicion(A_org, B_org)
    A = Animal("A", construir_oido(A_org))
    B = Animal("B", construir_oido(B_org))

    eventos = []
    for m in range(N_MOMENTOS):
        # el dolor afloja en los momentos de calma
        A.dolor *= 0.5; B.dolor *= 0.5
        if rmundo.random() > P_EVENTO:
            continue
        banda = BANDAS[rmundo.integers(len(BANDAS))]
        es_falsa = (rmundo.random() < P_FALSA)

        tap_tick, tap_razon, real = vivir_evento(A, B, banda, es_falsa, con_vision)

        # ¿quien primero? (puede ser ninguno en falsas)
        ta, tb = tap_tick["A"], tap_tick["B"]
        if ta is None and tb is None:
            primero = None; desfase = None
        elif tb is None or (ta is not None and ta < tb):
            primero = "A"; desfase = (tb - ta) if tb is not None else None
        elif ta is None or tb < ta:
            primero = "B"; desfase = (ta - tb) if ta is not None else None
        else:
            primero = "empate"; desfase = 0

        # verificacion y aprendizaje (cada uno con SU propio oido)
        for animal in (A, B):
            verificar_y_aprender(animal, banda,
                                 se_tapo_social=(tap_razon[animal.nombre] == 'social'),
                                 hubo_peligro_real=real[animal.nombre])

        eventos.append({'m': m, 'banda': banda, 'falsa': es_falsa, 'real_A': real["A"],
                        'real_B': real["B"], 'primero': primero, 'desfase': desfase,
                        'razon_A': tap_razon["A"], 'razon_B': tap_razon["B"],
                        'tap_A': tap_tick["A"], 'tap_B': tap_tick["B"]})
    return eventos, A, B


# ============================================================
# LO QUE MIRAMOS (no lo que imponemos)
# ============================================================
def medir(eventos):
    reales = [e for e in eventos if not e['falsa']]
    falsas = [e for e in eventos if e['falsa']]
    # quien se da cuenta primero (¿se turnan? ¿depende de la banda?)
    primeros = [e['primero'] for e in reales if e['primero'] in ('A', 'B')]
    pA = primeros.count('A'); pB = primeros.count('B')
    # ¿la banda manda quien es primero? (esperado: bandas - -> A, + -> B)
    band_primero = {}
    for e in reales:
        if e['primero'] in ('A', 'B'):
            band_primero.setdefault(e['banda'], {'A': 0, 'B': 0})[e['primero']] += 1
    # desfases medidos (no fijados)
    desf = [e['desfase'] for e in reales if e['desfase'] is not None]
    # cuanta gente se tapa SOCIAL (por ver al otro) — señal de contagio/codigo
    social = sum(1 for e in reales for r in (e['razon_A'], e['razon_B']) if r == 'social')
    tarde  = sum(1 for e in reales for r in (e['razon_A'], e['razon_B']) if r == 'tarde')
    # falsas alarmas: ¿alguien se tapo social en una falsa? (mal — deberia aprender a no)
    social_en_falsa = sum(1 for e in falsas for r in (e['razon_A'], e['razon_B']) if r == 'social')
    return {'n_reales': len(reales), 'n_falsas': len(falsas),
            'primero_A': pA, 'primero_B': pB, 'band_primero': band_primero,
            'desfase_medio': float(np.mean(desf)) if desf else None,
            'desfase_min': int(np.min(desf)) if desf else None,
            'desfase_max': int(np.max(desf)) if desf else None,
            'tapadas_social': social, 'tapadas_tarde': tarde,
            'social_en_falsa': social_en_falsa}


def main():
    print("=" * 96)
    print("V182_amenaza_alerta_v1 — EL PRIMER AVISO: dos animales que se advierten un peligro")
    print("=" * 96)
    print("  Dos animalitos que escuchan. Un sonido sube de volumen hasta doler. El que oye el")
    print("  dolor primero se TAPA; el otro lo ve, verifica con su propio oido, y aprende a leerlo.")
    print("  Nada de telepatia: solo se ven la conducta. El desfase y el tiempo de reaccion se MIDEN.")
    print("=" * 96)
    t0 = time.time()

    # CON vision (se ven taparse) vs SIN vision (control: cada uno solo con su dolor)
    ev_con, A_con, B_con = vivir(con_vision=True,  semilla_mundo=7)
    ev_sin, A_sin, B_sin = vivir(con_vision=False, semilla_mundo=7)   # mismo mundo
    m_con = medir(ev_con); m_sin = medir(ev_sin)

    print(f"\n  OIDOS (sensibilidad por banda, del cuerpo real):")
    print(f"    {'banda':>6} |   A     B")
    for b in BANDAS:
        print(f"    {b:>+6.0f} | {A_con.oido[b]:.2f}  {B_con.oido[b]:.2f}")

    print(f"\n{'#'*96}\n#  ¿SE TURNAN PARA AVISAR? (quien se da cuenta primero)\n{'#'*96}")
    print(f"  primero A: {m_con['primero_A']}   primero B: {m_con['primero_B']}   (de {m_con['n_reales']} sonidos reales)")
    print(f"  ¿depende de la banda del sonido? (esperado: bandas - -> A primero, + -> B primero)")
    for b in sorted(m_con['band_primero']):
        d = m_con['band_primero'][b]; print(f"     banda {b:>+5.0f}: A primero {d['A']}  | B primero {d['B']}")

    print(f"\n{'#'*96}\n#  EL DESFASE Y LA REACCION — MEDIDOS, NO FIJADOS\n{'#'*96}")
    if m_con['desfase_medio'] is not None:
        print(f"  el segundo reacciona, en promedio, {m_con['desfase_medio']:.1f} ticks despues del primero")
        print(f"  (minimo {m_con['desfase_min']}, maximo {m_con['desfase_max']})")
    else:
        print("  todavia no hay segundos que reaccionen tras el primero")

    print(f"\n{'#'*96}\n#  PELDAÑO 1 — ¿VER AL OTRO TAPARSE AYUDA? (contagio: con vision vs sin vision)\n{'#'*96}")
    print(f"  taparse SOCIAL (por ver al otro):  con vision {m_con['tapadas_social']}   sin vision {m_sin['tapadas_social']}")
    print(f"  timpanos rotos (se taparon TARDE): con vision {m_con['tapadas_tarde']}   sin vision {m_sin['tapadas_tarde']}")
    contagio = m_con['tapadas_social'] > 0 and m_con['tapadas_tarde'] < m_sin['tapadas_tarde']
    print(f"  -> {'✅ CONTAGIO: ver al otro hizo taparse antes y rompió menos timpanos' if contagio else '⚠ todavia no se ve que ver al otro ayude (leer por que)'}")

    print(f"\n{'#'*96}\n#  PELDAÑO 2 — ¿APRENDE A SEPARAR ALARMA REAL DE RUIDO? (verificacion)\n{'#'*96}")
    print(f"  taparse social en FALSAS alarmas: {m_con['social_en_falsa']}  (deberia tender a 0 al aprender)")
    print(f"  confianza aprendida por banda (A):  " + " ".join(f"{b:+.0f}:{A_con.confianza[b]:.2f}" for b in BANDAS))
    print(f"  confianza aprendida por banda (B):  " + " ".join(f"{b:+.0f}:{B_con.confianza[b]:.2f}" for b in BANDAS))
    aprende = (max(A_con.confianza.values()) > 0.5) and (max(B_con.confianza.values()) > 0.5)
    print(f"  -> {'✅ los dos aprendieron a confiar en la alarma del otro (en alguna banda)' if aprende else '⚠ la confianza aun no se consolida (leer por que)'}")

    print(f"\n{'#'*96}\n#  PELDAÑO 3 — ¿EL CODIGO ES MUTUO? (ambos aprenden, porque los roles se turnan)\n{'#'*96}")
    bandas_A = [b for b in BANDAS if A_con.confianza[b] > 0.5]
    bandas_B = [b for b in BANDAS if B_con.confianza[b] > 0.5]
    print(f"  A confia en B para las bandas: {bandas_A}")
    print(f"  B confia en A para las bandas: {bandas_B}")
    mutuo = len(bandas_A) > 0 and len(bandas_B) > 0
    print(f"  -> {'✅ MUTUO: los dos saben leer al otro (no es calle de un solo sentido)' if mutuo else '⚠ todavia no es mutuo (uno lee y el otro no -> falta inversion de roles real)'}")

    es_animal_que_avisa = contagio and aprende and mutuo
    print(f"\n{'='*96}")
    print(f"  -> {'✅ ESTE ANIMAL YA AVISA Y LEE AVISOS, EN AMBOS SENTIDOS' if es_animal_que_avisa else '⚠ AUN NO completa los tres peldaños — es un primer ciclo, es dato, no fracaso'}")
    print(f"{'='*96}")
    print(f"\n  tiempo {time.time()-t0:.1f}s")

    os.makedirs("V182_logs", exist_ok=True)
    salida = {'version': 'V182_amenaza_alerta_v1',
              'oido_A': A_con.oido, 'oido_B': B_con.oido,
              'medicion_con_vision': {k: v for k, v in m_con.items() if k != 'band_primero'},
              'medicion_sin_vision': {k: v for k, v in m_sin.items() if k != 'band_primero'},
              'band_primero': {str(k): v for k, v in m_con['band_primero'].items()},
              'confianza_A': {str(k): v for k, v in A_con.confianza.items()},
              'confianza_B': {str(k): v for k, v in B_con.confianza.items()},
              'contagio': bool(contagio), 'aprende': bool(aprende), 'mutuo': bool(mutuo),
              'completa_los_tres': bool(es_animal_que_avisa)}
    with open(f"V182_logs/v182_amenaza_alerta_v1_{TS}.json", "w") as f:
        json.dump(salida, f, indent=2)
    print(f"  log: V182_logs/v182_amenaza_alerta_v1_{TS}.json")


if __name__ == "__main__":
    main()
