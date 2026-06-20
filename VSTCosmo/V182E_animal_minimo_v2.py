#!/usr/bin/env python3
"""
V182E_animal_minimo_v2 — DE PLANTA A ANIMAL (corregido: descanso real + lectura limpia)
================================================================================

QUE SE ARREGLO RESPECTO A v1 (en simple)
----------------------------------------
v1 reporto ✅✅ pero tenia DOS goteras que el ✅ tapaba. Aca se arreglan de raiz.

GOTERA 1 — el "descanso" no descansaba (fatiga iba 461 -> 1996 -> 2246, subia siempre).
  Causa: el bloque de descanso igual hacia mover al organismo, y moverse cansa; y era
  demasiado corto para la recuperacion lenta de V180 (tau_recuperacion=300).
  Arreglo: descansa DE VERDAD — lo dejo quieto en su banda preferida (error ~0 -> rama
  de reposo de V180 -> la fatiga se recupera) y por MUCHOS pasos. Ahora la fatiga sube
  con trabajo y BAJA con descanso. Eso es lo que faltaba: un estado que oscila, no una
  pila que se descarga una sola vez.

GOTERA 2 — el experto NO aceptaba su banda; solo 1 de 5 bandas respondia.
  Causa (de fondo, no de calibracion): la "aceptacion" de v1 hacia VIAJAR al organismo
  hasta la banda y medsia si llegaba. Si venia de otra banda, no alcanzaba a llegar en
  el tiempo del probe -> media "que tan lejos quedo la banda anterior", no "cuanto le
  gusta esta banda". Por eso el experto salia 0%: el medidor estaba roto, no el gusto.
  Arreglo: aceptacion = GUSTO x CAPACIDAD, ambos cantidades reales de V180:
    - GUSTO    = p_acepta(valencia(banda))  [la disposicion limpia, como en D, que si
                 funcionaba: el experto SI acepta su banda]
    - CAPACIDAD= factor_gain de V180 = exp(-k_gain * fatiga)  [V180 YA baja la capacidad
                 cuando hay fatiga]
  => "le gusta x puede actuar". Fresco: puede -> acepta sus bandas. Cansado: su capacidad
     baja -> acepta menos en TODAS las bandas que le gustan (no en una sola). Y el probe
     ya no hace viajar al motor: lee el estado directo, asi que tampoco re-cansa.

HONESTIDAD (lo que el equipo paso por alto en v1)
-------------------------------------------------
El test "el otro lee el estado" de v1 era casi tautologico: si el comportamiento se
DEFINE como funcion de la fatiga, "leer la fatiga predice el comportamiento" es obvio,
no informa. La lectura MUTUA de verdad (A predice a B) solo tiene sentido si los dos
animales estan ACOPLADOS (mismo ambiente, fatigas correlacionadas), y eso es diseño de
V183, no de aca. Por eso este script NO inventa un numero de "legibilidad mutua". Mide
solo lo que corresponde: (a) el estado interno sube Y baja, (b) el comportamiento lo
sigue en VARIAS bandas, (c) el experto acepta su banda cuando esta fresco.

CUERPO: V180 importado VERBATIM. No se cambia ninguna constante del cuerpo. Solo se lee
su estado (valencia, fatiga, k_gain) y se lo hace trabajar/descansar.
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
# CONFIG
# ============================================================
BANDAS = [-60.0, -30.0, 0.0, 30.0, 60.0]

PASOS_FUERTE  = 40000
PASOS_MEDIO   = 15000
PASOS_COMPART = 20000
PASOS_DEBIL   = 5000

PASOS_REPOSO_INICIAL = 20000   # descanso post-exposicion -> linea base "fresco" limpia
PASOS_TRABAJO        = 6000    # bloque de esfuerzo -> sube fatiga
PASOS_REPOSO         = 20000   # descanso post-trabajo -> baja fatiga (la prueba de la gotera 1)

# Disposicion (igual que en D: sigmoide sobre valencia)
UMBRAL_ACEPTA = 5.0
ESCALA_ACEPTA = 2.0

# Umbrales del veredicto (declarados antes de correr)
SWING_MIN   = 0.10   # cambio minimo de aceptacion fresco->cansado para contar como "responde"
N_BANDAS_MIN = 2     # cuantas bandas deben responder para no ser "una sola puerta"
RECUP_MIN   = 0.20   # la fatiga recuperada debe bajar al menos 20% respecto a cansado

EXPERTO = {44: -60.0, 77: 60.0}   # banda nativa de cada organismo (descansa ahi)
SEED_A, SEED_B = 44, 77
TS = time.strftime("%Y%m%d_%H%M%S")


# ============================================================
# CUERPO
# ============================================================
def consolidar(org, banda, pasos):
    for _ in range(pasos):
        org.actualizar_setpoint(0.0, DT, DT, banda, target_reward=banda)

def fase_exposicion(A, B):
    consolidar(A, -60.0, PASOS_FUERTE); consolidar(A, -30.0, PASOS_MEDIO)
    consolidar(A,  30.0, PASOS_DEBIL);  consolidar(A,  60.0, PASOS_DEBIL)
    consolidar(B,  60.0, PASOS_FUERTE); consolidar(B,  30.0, PASOS_MEDIO)
    consolidar(B, -30.0, PASOS_DEBIL);  consolidar(B, -60.0, PASOS_DEBIL)
    consolidar(A, 0.0, PASOS_COMPART);  consolidar(B, 0.0, PASOS_COMPART)


# ============================================================
# TRABAJO (sube fatiga) y DESCANSO (baja fatiga) — REALES
# ============================================================
def trabajar_duro(org, t0):
    """Alterna blancos lejanos -> errores grandes -> esfuerzo -> la fatiga SUBE."""
    for s in range(PASOS_TRABAJO):
        t = t0 + s * DT
        objetivo = 60.0 if (s // 200) % 2 == 0 else -60.0
        org.actualizar_con_opciones(t, DT, t0 + PASOS_TRABAJO*DT, [objetivo], False, None)

def descansar(org, banda_querida, pasos, t0):
    """Lo deja quieto en una banda que le gusta. Llega (esfuerzo breve) y se queda dentro
    de la zona muerta -> rama de reposo de V180 -> la fatiga se RECUPERA el resto del
    tiempo. Por eso debe ser LARGO (la recuperacion de V180 es lenta)."""
    for s in range(pasos):
        t = t0 + s * DT
        org.actualizar_con_opciones(t, DT, t0 + pasos*DT, [banda_querida], False, None)


# ============================================================
# LECTURA LIMPIA DEL ESTADO  (gusto x capacidad, ambos de V180)
# ============================================================
def p_acepta(org, banda):
    v = org.get_valencia(banda)
    return 1.0 / (1.0 + np.exp(-(v - UMBRAL_ACEPTA) / ESCALA_ACEPTA))

def factor_gain(org):
    fa = org.motor.fatiga.get_fatiga()
    kg = org.motor.fatiga.k_gain
    return float(np.clip(np.exp(-kg * fa), 0.2, 1.0))

def leer_fase(org):
    fg = factor_gain(org)
    disp = {b: float(p_acepta(org, b)) for b in BANDAS}        # gusto (independiente de fatiga)
    will = {b: float(disp[b] * fg) for b in BANDAS}            # aceptacion = gusto x capacidad
    return {'fatiga': float(org.motor.fatiga.get_fatiga()), 'factor_gain': fg,
            'disposicion': disp, 'aceptacion': will}


def correr_organismo(org, experto):
    t = 0.0
    descansar(org, experto, PASOS_REPOSO_INICIAL, t); t += PASOS_REPOSO_INICIAL * DT
    fresco = leer_fase(org)
    trabajar_duro(org, t); t += PASOS_TRABAJO * DT
    cansado = leer_fase(org)
    descansar(org, experto, PASOS_REPOSO, t); t += PASOS_REPOSO * DT
    recuperado = leer_fase(org)
    return {'fresco': fresco, 'cansado': cansado, 'recuperado': recuperado}


# ============================================================
# VEREDICTO
# ============================================================
def evaluar(res, experto):
    fr, ca, re = res['fresco'], res['cansado'], res['recuperado']
    swing = {b: abs(fr['aceptacion'][b] - ca['aceptacion'][b]) for b in BANDAS}
    n_responden = sum(1 for b in BANDAS if swing[b] > SWING_MIN)
    experto_acepta_fresco = fr['aceptacion'][experto] > 0.5
    recupera = (ca['fatiga'] - re['fatiga']) / ca['fatiga'] if ca['fatiga'] > 1e-9 else 0.0
    return {'swing': swing, 'n_responden': n_responden,
            'experto_acepta_fresco': bool(experto_acepta_fresco),
            'frac_recuperacion': float(recupera),
            'recupera_ok': bool(recupera > RECUP_MIN)}


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 96)
    print("V182E_animal_minimo_v2 — DE PLANTA A ANIMAL (descanso real + lectura limpia)")
    print("=" * 96)
    print("  Aceptacion = GUSTO(valencia) x CAPACIDAD(factor_gain de V180). El estado interno")
    print("  (fatiga) sube con trabajo y baja con descanso; la aceptacion lo sigue en varias bandas.")
    print("=" * 96)
    t0 = time.time()

    A = V180.OrganismoV180(seed=SEED_A, memoria_episodica=V180.MemoriaEpisodicaV180())
    B = V180.OrganismoV180(seed=SEED_B, memoria_episodica=V180.MemoriaEpisodicaV180())
    A.set_modo_entrenamiento(False); B.set_modo_entrenamiento(False)
    fase_exposicion(A, B)

    resA = correr_organismo(A, EXPERTO[SEED_A])
    resB = correr_organismo(B, EXPERTO[SEED_B])
    evA = evaluar(resA, EXPERTO[SEED_A])
    evB = evaluar(resB, EXPERTO[SEED_B])

    def tabla(nombre, res, ev, experto):
        print(f"\n  {nombre} (experto en {experto:+.0f}°)")
        print(f"    fatiga:  fresco {res['fresco']['fatiga']:.0f}  ->  cansado {res['cansado']['fatiga']:.0f}  ->  recuperado {res['recuperado']['fatiga']:.0f}")
        print(f"    capacidad(factor_gain): {res['fresco']['factor_gain']:.2f} -> {res['cansado']['factor_gain']:.2f} -> {res['recuperado']['factor_gain']:.2f}")
        print(f"    {'banda':>6} | gusto | acept: fresco cansado recup | swing | responde")
        print(f"    {'-'*6}-+-{'-'*5}-+-{'-'*27}-+-{'-'*5}-+---------")
        for b in BANDAS:
            g = res['fresco']['disposicion'][b]
            af, ac, ar = res['fresco']['aceptacion'][b], res['cansado']['aceptacion'][b], res['recuperado']['aceptacion'][b]
            sw = ev['swing'][b]; resp = '✅' if sw > SWING_MIN else '·'
            print(f"    {b:>+6.0f} | {g:>4.0%} | {af:>11.0%} {ac:>6.0%} {ar:>6.0%} | {sw:>4.0%} |   [{resp}]")

    tabla("A", resA, evA, EXPERTO[SEED_A])
    tabla("B", resB, evB, EXPERTO[SEED_B])

    print(f"\n{'#'*96}\n#  VEREDICTO\n{'#'*96}")
    print(f"  GOTERA 1 (el descanso recupera la fatiga):")
    print(f"    A: baja {evA['frac_recuperacion']:+.0%} tras descansar  -> {'✅' if evA['recupera_ok'] else '❌ sigue sin recuperar'}")
    print(f"    B: baja {evB['frac_recuperacion']:+.0%} tras descansar  -> {'✅' if evB['recupera_ok'] else '❌ sigue sin recuperar'}")
    print(f"  GOTERA 2 (el experto acepta su banda fresco, y responden varias bandas):")
    print(f"    A: experto acepta fresco {'✅' if evA['experto_acepta_fresco'] else '❌'} ; bandas que responden: {evA['n_responden']}/5 {'✅' if evA['n_responden']>=N_BANDAS_MIN else '❌'}")
    print(f"    B: experto acepta fresco {'✅' if evB['experto_acepta_fresco'] else '❌'} ; bandas que responden: {evB['n_responden']}/5 {'✅' if evB['n_responden']>=N_BANDAS_MIN else '❌'}")

    es_animal = all([
        evA['recupera_ok'], evB['recupera_ok'],
        evA['experto_acepta_fresco'], evB['experto_acepta_fresco'],
        evA['n_responden'] >= N_BANDAS_MIN, evB['n_responden'] >= N_BANDAS_MIN,
    ])
    print(f"\n  -> {'✅ ANIMAL: el estado interno oscila (sube Y baja) y la conducta lo sigue en varias bandas' if es_animal else '⚠ AUN NO: alguna condicion no se cumple (ver arriba). No avanzar a V183 hasta cerrarlas'}")
    print(f"\n  NOTA: la lectura MUTUA (A predice a B) NO se mide aca — requiere acoplar los dos animales")
    print(f"  (mismo ambiente, fatigas correlacionadas). Eso es diseño de V183, no de este paso.")
    print(f"\n  tiempo {time.time()-t0:.1f}s")

    os.makedirs("V182_logs", exist_ok=True)
    salida = {'version': 'V182E_animal_minimo_v2', 'bandas': BANDAS,
              'A': resA, 'B': resB,
              'eval_A': {k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()}) for k, v in evA.items()},
              'eval_B': {k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()}) for k, v in evB.items()},
              'es_animal': bool(es_animal)}
    with open(f"V182_logs/v182e_animal_minimo_v2_{TS}.json", "w") as f:
        json.dump(salida, f, indent=2)
    print(f"  log: V182_logs/v182e_animal_minimo_v2_{TS}.json")


if __name__ == "__main__":
    main()
