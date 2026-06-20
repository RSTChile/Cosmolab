#!/usr/bin/env python3
"""
V182D_C — ¿PUEDE EL UMBRAL DE ALTERIDAD SURGIR DEL ORGANISMO, SIN QUE LO FIJEMOS?
================================================================================
Tres alternativas en paralelo, sobre la MISMA dinada de V182D_B (cuerpo V180 real).
La pregunta "¿el margen emerge del organismo?" se PARTE en dos preguntas distintas:
  (a) DONDE esta la frontera (que banda entra al test)
  (b) QUE TAN GRANDE debe ser el efecto (tamaño)
Cada alternativa endogena responde una; la tercera es un control que demuestra una
trampa. NO compiten por un unico puesto.

  ALT 1 — NULO POR PERMUTACION (fija el TAMAÑO desde el ruido del organismo):
    El umbral de tamaño deja de ser un numero nuestro. Se construye un nulo: se
    observa al otro pero se BARAJA el mapeo banda->disposicion, de modo que el
    modelo pierde la informacion banda-especifica. Se recomputa `genuino` muchas
    veces -> distribucion nula generada por el propio ruido de eleccion. Veredicto:
    genuino_real > percentil 95 del nulo. Lo unico convencional que queda es el
    percentil (95%), estadistica, no un tamaño de efecto magico.

  ALT 2 — FRONTERA ENDOGENA (fija la FRONTERA desde el modelo interno del otro):
    El organismo clasifica una banda como "con alteridad que modelar" cuando SU
    modelo observado del otro cae al OTRO lado del 0.5 que su propia disposicion.
    Prediccion: esto REPRODUCE el ⊘ estructural (mismo lado del 0.5) que impusimos
    en V182D_B, pero ahora derivado de lo que el organismo OBSERVA, no de lo que el
    experimentador sabe. La frontera emerge del organismo. NO fija el tamaño.

  ALT 3 — TAUTOLOGIA (CONTROL, NO CANDIDATA):
    Umbral leido de la MISMA cantidad que da el veredicto (umbral = genuino - eps).
    Pasa SIEMPRE —incluso con genuino ≈ 0 o negativo—. No falsea nada. Se implementa
    para DEMOSTRAR el confound (familia que mato los primeros disenos de C), no como
    criterio. Su "exito" es su prueba de invalidez.

SINTESIS ESPERADA (a confirmar con la corrida):
  ALT 1 y ALT 2 son COMPLEMENTARIAS: juntas dan un criterio de D totalmente
  endogeno (frontera del organismo + tamaño desde su ruido), sin numero nuestro.
  ALT 3 queda demostrada como no-falsable. El margen fijo (15/30/50) se vuelve
  innecesario: el organismo pone la frontera, su ruido pone el tamaño.

CUERPO: V180 importado VERBATIM. Maquinaria de V182D_B sin cambios; se agregan los
tres medidores de umbral.
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
# CONFIG (igual que V182D_B)
# ============================================================
BANDAS = [-60.0, 0.0, 60.0]
PASOS_FUERTE   = 80000
PASOS_DEBIL    = 8000
PASOS_COMPART  = 40000
N_OBS   = 200
N_PRED  = 200
UMBRAL_ACEPTA = 5.0
ESCALA_ACEPTA = 2.0
LADO = 0.5

# ALT 1: percentil del nulo (unica convencion que queda; NO es un tamaño de efecto)
N_PERM       = 200
PERCENTIL_NULO = 95

SEED_A, SEED_B = 44, 77
TS = time.strftime("%Y%m%d_%H%M%S")


# ============================================================
# CUERPO + CONDUCTA (identico a V182D_B)
# ============================================================
def consolidar(org, banda, pasos):
    for _ in range(pasos):
        org.actualizar_setpoint(0.0, DT, DT, banda, target_reward=banda)

def fase_exposicion(A, B):
    consolidar(A, -60.0, PASOS_FUERTE); consolidar(A, 60.0, PASOS_DEBIL)
    consolidar(B,  60.0, PASOS_FUERTE); consolidar(B, -60.0, PASOS_DEBIL)
    consolidar(A, 0.0, PASOS_COMPART);  consolidar(B, 0.0, PASOS_COMPART)

def p_acepta(org, banda):
    v = org.get_valencia(banda)
    return 1.0 / (1.0 + np.exp(-(v - UMBRAL_ACEPTA) / ESCALA_ACEPTA))

def lado_de(org, banda):
    return p_acepta(org, banda) >= LADO

def actuar(org, banda, rng):
    return 1 if rng.random() < p_acepta(org, banda) else 0

def observar_al_otro(otro, rng):
    return {b: float(np.mean([actuar(otro, b, rng) for _ in range(N_OBS)])) for b in BANDAS}

def observar_scramble(otro, rng):
    """Modelo NULO: observa al otro, pero baraja el mapeo banda->disposicion, de modo
    que pierde la informacion banda-especifica (conserva las frecuencias marginales)."""
    m = observar_al_otro(otro, rng)
    vals = [m[b] for b in BANDAS]
    rng.shuffle(vals)
    return {b: vals[i] for i, b in enumerate(BANDAS)}


# ============================================================
# genuino por banda dado un modelo del otro
# ============================================================
def genuino_por_banda(predictor, objetivo, modelo, rng_obj):
    out = {}
    for b in BANDAS:
        pred_modelo = 1 if modelo[b] >= 0.5 else 0
        pred_proy   = 1 if lado_de(predictor, b) else 0
        am = ap = 0
        for _ in range(N_PRED):
            v = actuar(objetivo, b, rng_obj)
            am += int(pred_modelo == v); ap += int(pred_proy == v)
        out[b] = (am / N_PRED) - (ap / N_PRED)
    return out


# ============================================================
# ALT 1 — NULO POR PERMUTACION (tamaño desde el ruido del organismo)
# ============================================================
def alt1_permutacion(predictor, objetivo, modelo_real, base_seed):
    g_obs = genuino_por_banda(predictor, objetivo, modelo_real, np.random.default_rng(base_seed))
    nulos = {b: [] for b in BANDAS}
    for i in range(N_PERM):
        m_scr = observar_scramble(objetivo, np.random.default_rng(base_seed + 1000 + i))
        g_n = genuino_por_banda(predictor, objetivo, m_scr, np.random.default_rng(base_seed + 5000 + i))
        for b in BANDAS:
            nulos[b].append(g_n[b])
    res = {}
    for b in BANDAS:
        p = float(np.percentile(nulos[b], PERCENTIL_NULO))
        res[b] = {'genuino': float(g_obs[b]), 'umbral_nulo_p95': p, 'pasa': bool(g_obs[b] > p)}
    return res


# ============================================================
# ALT 2 — FRONTERA ENDOGENA (frontera desde el modelo interno del otro)
# ============================================================
def alt2_frontera_endogena(predictor, objetivo, modelo_interno):
    res = {}
    for b in BANDAS:
        self_side  = lado_de(predictor, b)
        model_side = modelo_interno[b] >= 0.5
        obj_side   = lado_de(objetivo, b)   # verdad estructural (nivel experimentador)
        medible_endo    = (model_side != self_side)   # el organismo: "el otro difiere de mi"
        medible_estruct = (obj_side  != self_side)     # V182D_B
        res[b] = {'self_side': bool(self_side), 'model_side': bool(model_side),
                  'obj_side_true': bool(obj_side),
                  'medible_endogeno': bool(medible_endo),
                  'medible_estructural': bool(medible_estruct),
                  'coincide': bool(medible_endo == medible_estruct)}
    return res


# ============================================================
# ALT 3 — TAUTOLOGIA (CONTROL: pasa siempre -> no falsea nada)
# ============================================================
def alt3_tautologia(genuino_dict):
    res = {}
    for b in BANDAS:
        g = genuino_dict[b]
        umbral = g - 1e-9          # umbral leido del propio veredicto
        res[b] = {'genuino': float(g), 'umbral_tautologico': float(umbral), 'pasa': bool(g > umbral)}
    return res


# ============================================================
# CORRIDA
# ============================================================
def correr():
    A = V180.OrganismoV180(seed=SEED_A, memoria_episodica=V180.MemoriaEpisodicaV180())
    B = V180.OrganismoV180(seed=SEED_B, memoria_episodica=V180.MemoriaEpisodicaV180())
    A.set_modo_entrenamiento(False); B.set_modo_entrenamiento(False)
    fase_exposicion(A, B)

    disp = {b: (round(p_acepta(A, b), 2), round(p_acepta(B, b), 2)) for b in BANDAS}
    modelo_A_de_B = observar_al_otro(B, np.random.default_rng(101))
    modelo_B_de_A = observar_al_otro(A, np.random.default_rng(202))

    g_AB = genuino_por_banda(A, B, modelo_A_de_B, np.random.default_rng(303))
    g_BA = genuino_por_banda(B, A, modelo_B_de_A, np.random.default_rng(305))

    return {
        'disp': disp, 'modelo_A_de_B': modelo_A_de_B, 'modelo_B_de_A': modelo_B_de_A,
        'g_AB': g_AB, 'g_BA': g_BA,
        'alt1_AB': alt1_permutacion(A, B, modelo_A_de_B, 30000),
        'alt1_BA': alt1_permutacion(B, A, modelo_B_de_A, 40000),
        'alt2_AB': alt2_frontera_endogena(A, B, modelo_A_de_B),
        'alt2_BA': alt2_frontera_endogena(B, A, modelo_B_de_A),
        'alt3_AB': alt3_tautologia(g_AB),
        'alt3_BA': alt3_tautologia(g_BA),
    }


def _tabla_alt1(titulo, a1):
    print(f"\n  {titulo}")
    print(f"    {'banda':>6} | {'genuino':>8} {'umbral_nulo(p95)':>17} | veredicto")
    print(f"    {'-'*6}-+-{'-'*26}-+----------")
    for b in BANDAS:
        r = a1[b]; mk = '✅' if r['pasa'] else '·'
        print(f"    {b:>+6.0f} | {r['genuino']:>+8.0%} {r['umbral_nulo_p95']:>+17.0%} | [{mk}] {'supera el nulo' if r['pasa'] else 'dentro del ruido'}")

def _tabla_alt2(titulo, a2):
    print(f"\n  {titulo}")
    print(f"    {'banda':>6} | self  modelo  obj* | endogeno   estructural  coincide")
    print(f"    {'-'*6}-+-{'-'*18}-+-{'-'*30}")
    for b in BANDAS:
        r = a2[b]
        se = 'acep' if r['self_side'] else 'rech'
        mo = 'acep' if r['model_side'] else 'rech'
        ob = 'acep' if r['obj_side_true'] else 'rech'
        en = 'medible' if r['medible_endogeno'] else '⊘'
        es = 'medible' if r['medible_estructural'] else '⊘'
        ck = '✅' if r['coincide'] else '❌'
        print(f"    {b:>+6.0f} | {se:>4} {mo:>6} {ob:>5} | {en:>8}  {es:>11}   [{ck}]")

def _tabla_alt3(titulo, a3):
    print(f"\n  {titulo}")
    print(f"    {'banda':>6} | {'genuino':>8} {'umbral_tauto':>13} | pasa")
    print(f"    {'-'*6}-+-{'-'*22}-+------")
    for b in BANDAS:
        r = a3[b]
        print(f"    {b:>+6.0f} | {r['genuino']:>+8.0%} {r['umbral_tautologico']:>+13.2%} | {'SIEMPRE ✅' if r['pasa'] else '❌'}")


def main():
    print("=" * 98)
    print("V182D_C — ¿EL UMBRAL DE ALTERIDAD EMERGE DEL ORGANISMO?  (3 alternativas en paralelo)")
    print("=" * 98)
    print("  ALT 1 nulo-permutacion  -> fija el TAMAÑO desde el ruido del organismo")
    print("  ALT 2 frontera-endogena -> fija la FRONTERA desde el modelo interno del otro")
    print("  ALT 3 tautologia        -> CONTROL: pasa siempre, no falsea nada (no es candidata)")
    print("=" * 98)

    t0 = time.time()
    R = correr()

    print(f"\n  [disposicion emergente — P(acepta)]   A / B")
    for b in BANDAS:
        pa, pb = R['disp'][b]; print(f"     {b:>+6.0f} | {pa:.2f} / {pb:.2f}")

    print(f"\n{'='*98}\n  ALT 1 — NULO POR PERMUTACION (tamaño endogeno: el ruido del organismo pone el bar)\n{'='*98}")
    _tabla_alt1("A modela a B:", R['alt1_AB'])
    _tabla_alt1("B modela a A:", R['alt1_BA'])

    print(f"\n{'='*98}\n  ALT 2 — FRONTERA ENDOGENA (¿reproduce el ⊘ estructural de V182D_B?)\n{'='*98}")
    print("    (obj* = disposicion real del otro, nivel experimentador; 'coincide' = frontera del")
    print("     organismo == frontera estructural)")
    _tabla_alt2("A modela a B:", R['alt2_AB'])
    _tabla_alt2("B modela a A:", R['alt2_BA'])

    print(f"\n{'='*98}\n  ALT 3 — TAUTOLOGIA (CONTROL — demuestra el confound, NO es criterio)\n{'='*98}")
    _tabla_alt3("A modela a B:", R['alt3_AB'])
    _tabla_alt3("B modela a A:", R['alt3_BA'])

    # ---- SINTESIS ----
    def pasa_alt1(a1, a2):
        # banda valida = medible endogena (ALT2) Y supera el nulo (ALT1)
        return [b for b in BANDAS if a2[b]['medible_endogeno'] and a1[b]['pasa']]
    def medibles_endo(a2):
        return [b for b in BANDAS if a2[b]['medible_endogeno']]

    ver_AB = pasa_alt1(R['alt1_AB'], R['alt2_AB']); med_AB = medibles_endo(R['alt2_AB'])
    ver_BA = pasa_alt1(R['alt1_BA'], R['alt2_BA']); med_BA = medibles_endo(R['alt2_BA'])
    frontera_coincide = all(R['alt2_AB'][b]['coincide'] for b in BANDAS) and \
                        all(R['alt2_BA'][b]['coincide'] for b in BANDAS)
    alt3_siempre = all(R['alt3_AB'][b]['pasa'] for b in BANDAS) and \
                   all(R['alt3_BA'][b]['pasa'] for b in BANDAS)
    bidir_endogeno = (len(ver_AB) >= 1 and ver_AB == med_AB and
                      len(ver_BA) >= 1 and ver_BA == med_BA)

    print(f"\n{'#'*98}\n#  SINTESIS\n{'#'*98}")
    print(f"  ALT 1 (tamaño por nulo): A->B supera nulo en {[b for b in BANDAS if R['alt1_AB'][b]['pasa']]}, "
          f"B->A en {[b for b in BANDAS if R['alt1_BA'][b]['pasa']]}")
    print(f"  ALT 2 (frontera endogena) == ⊘ estructural de V182D_B: {'✅ SI' if frontera_coincide else '❌ NO (revisar)'}")
    print(f"  ALT 3 (tautologia) pasa SIEMPRE (incl. banda degenerada): {'✅ confirmado -> no falsable' if alt3_siempre else 'inesperado'}")
    print(f"\n  CRITERIO TOTALMENTE ENDOGENO (ALT2 frontera + ALT1 tamaño, sin margen nuestro):")
    print(f"     A->B: {len(ver_AB)}/{len(med_AB)} bandas endogenas con alteridad   {ver_AB}")
    print(f"     B->A: {len(ver_BA)}/{len(med_BA)} bandas endogenas con alteridad   {ver_BA}")
    print(f"     -> {'✅ ALTERIDAD BIDIRECCIONAL (umbral del organismo, sin numero fijado por nosotros)' if bidir_endogeno else '❌ revisar'}")
    print(f"\n  LECTURA: ALT1 y ALT2 son complementarias (tamaño + frontera). ALT3 es el control que")
    print(f"  muestra por que NO se puede leer el umbral de la misma cantidad que da el veredicto.")
    print(f"\n  tiempo {time.time()-t0:.1f}s")

    os.makedirs("V182_logs", exist_ok=True)
    salida = {'version': 'V182D_C-umbral-endogeno',
              'disposicion': {str(k): v for k, v in R['disp'].items()},
              'alt1_AB': {str(k): v for k, v in R['alt1_AB'].items()},
              'alt1_BA': {str(k): v for k, v in R['alt1_BA'].items()},
              'alt2_AB': {str(k): v for k, v in R['alt2_AB'].items()},
              'alt2_BA': {str(k): v for k, v in R['alt2_BA'].items()},
              'alt3_AB': {str(k): v for k, v in R['alt3_AB'].items()},
              'alt3_BA': {str(k): v for k, v in R['alt3_BA'].items()},
              'frontera_endogena_coincide_estructural': bool(frontera_coincide),
              'alt3_no_falsable': bool(alt3_siempre),
              'alteridad_bidireccional_endogena': bool(bidir_endogeno)}
    with open(f"V182_logs/v182d_C_umbral_endogeno_{TS}.json", "w") as f:
        json.dump(salida, f, indent=2)
    print(f"  log: V182_logs/v182d_C_umbral_endogeno_{TS}.json")


if __name__ == "__main__":
    main()