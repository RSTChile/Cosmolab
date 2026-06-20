#!/usr/bin/env python3
"""
V182D_C2 — UMBRAL ENDOGENO DE ALTERIDAD (ALT1 CORREGIDO: NULO PAREADO POR TRIAL)
================================================================================
Corrige V182D_C. El bug estaba SOLO en ALT1. ALT2 y ALT3 quedan identicos.

BUG DE ALT1 (V182D_C):
  El nulo barajaba el mapeo banda->disposicion. Con 3 bandas y un solo "acepta"
  (0.80), solo importa donde cae ese valor; en ~1/3 de las 6 permutaciones cae
  donde va de verdad -> el nulo REPRODUCE el modelo real. El nulo CONTENIA la
  hipotesis que debia excluir, asi que el p95 se sienta encima de la señal y el
  efecto real (real) casi nunca le gana. Falso negativo por nulo contaminado.

CORRECCION (ALT1):
  El nulo correcto para lo que D afirma —"el modelo le gana a la PROYECCION"— es un
  NULO PAREADO POR TRIAL. Por banda, cada trial produce (modelo_acierta,
  proyeccion_acierta). H0: las dos etiquetas son intercambiables (el modelo no
  aporta sobre la proyeccion). Se barajan las etiquetas por trial, se reconstruye
  la distribucion de `genuino`, y el umbral es su percentil 95.
    - El piso lo pone el RUIDO DE ELECCION del propio organismo (endogeno).
    - Lo unico convencional que queda es el percentil (95), estadistica.
    - Banda divergente: modelo y proyeccion predicen OPUESTO -> genuino refleja la
      disposicion real del objetivo, nulo ~0 -> pasa con holgura.
    - Banda degenerada 0°: modelo == proyeccion -> genuino = 0 exacto, nulo = 0 ->
      NO pasa -> ⊘ natural (coincide con ALT2).
    - SIGUE FALSABLE: si el modelo predice MAL en una banda divergente, genuino < 0
      y falla. No es sello de goma.

SEPARACION QUE NO HAY QUE MEZCLAR (error del analisis previo, corregido):
  Este nulo NO toca la cuestion de I(A;B) ≈ 0.15 < 0.3. Esa es informacion mutua
  (modelado banda-especifico SOBRE la base rate del otro); con disposicion 2-1
  solo la banda minoritaria aporta sobre la marginal. Es un limite del diseño de
  3 bandas y pertenece a V183, no a ALT1.

CUERPO: V180 importado VERBATIM. Maquinaria de V182D_B sin cambios.
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
# CONFIG (igual que V182D_C)
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
N_PERM       = 2000           # mas muestras: la permutacion pareada es barata
PERCENTIL_NULO = 95

SEED_A, SEED_B = 44, 77
TS = time.strftime("%Y%m%d_%H%M%S")


# ============================================================
# CUERPO + CONDUCTA (identico a V182D_B/C)
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


# ============================================================
# ALT 1 (CORREGIDO) — NULO PAREADO POR TRIAL
# ============================================================
def alt1_nulo_pareado(predictor, objetivo, modelo, base_seed):
    rng_obj = np.random.default_rng(base_seed)
    rng_perm = np.random.default_rng(base_seed + 7777)
    res = {}
    for b in BANDAS:
        pred_modelo = 1 if modelo[b] >= 0.5 else 0
        pred_proy   = 1 if lado_de(predictor, b) else 0
        mc = np.empty(N_PRED, dtype=int); pc = np.empty(N_PRED, dtype=int)
        for t in range(N_PRED):
            v = actuar(objetivo, b, rng_obj)
            mc[t] = int(pred_modelo == v)
            pc[t] = int(pred_proy == v)
        genuino = float(mc.mean() - pc.mean())
        # nulo pareado: intercambiar etiquetas modelo<->proyeccion por trial
        nulos = np.empty(N_PERM)
        for i in range(N_PERM):
            swap = rng_perm.random(N_PRED) < 0.5
            mc_p = np.where(swap, pc, mc)
            pc_p = np.where(swap, mc, pc)
            nulos[i] = mc_p.mean() - pc_p.mean()
        umbral = float(np.percentile(nulos, PERCENTIL_NULO))
        res[b] = {'genuino': genuino, 'umbral_nulo_p95': umbral, 'pasa': bool(genuino > umbral)}
    return res


# ============================================================
# ALT 2 — FRONTERA ENDOGENA (sin cambios)
# ============================================================
def alt2_frontera_endogena(predictor, objetivo, modelo_interno):
    res = {}
    for b in BANDAS:
        self_side  = lado_de(predictor, b)
        model_side = modelo_interno[b] >= 0.5
        obj_side   = lado_de(objetivo, b)
        res[b] = {'self_side': bool(self_side), 'model_side': bool(model_side),
                  'obj_side_true': bool(obj_side),
                  'medible_endogeno': bool(model_side != self_side),
                  'medible_estructural': bool(obj_side != self_side),
                  'coincide': bool((model_side != self_side) == (obj_side != self_side))}
    return res


# ============================================================
# ALT 3 — TAUTOLOGIA (CONTROL, sin cambios)
# ============================================================
def alt3_tautologia(genuino_dict):
    res = {}
    for b in BANDAS:
        g = genuino_dict[b]
        res[b] = {'genuino': float(g), 'umbral_tautologico': float(g - 1e-9), 'pasa': True}
    return res


# ============================================================
# genuino observado por banda (para ALT3 y reporte)
# ============================================================
def genuino_observado(predictor, objetivo, modelo, base_seed):
    rng = np.random.default_rng(base_seed)
    out = {}
    for b in BANDAS:
        pm = 1 if modelo[b] >= 0.5 else 0
        pp = 1 if lado_de(predictor, b) else 0
        am = ap = 0
        for _ in range(N_PRED):
            v = actuar(objetivo, b, rng)
            am += int(pm == v); ap += int(pp == v)
        out[b] = (am / N_PRED) - (ap / N_PRED)
    return out


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

    return {
        'disp': disp, 'modelo_A_de_B': modelo_A_de_B, 'modelo_B_de_A': modelo_B_de_A,
        'alt1_AB': alt1_nulo_pareado(A, B, modelo_A_de_B, 30000),
        'alt1_BA': alt1_nulo_pareado(B, A, modelo_B_de_A, 40000),
        'alt2_AB': alt2_frontera_endogena(A, B, modelo_A_de_B),
        'alt2_BA': alt2_frontera_endogena(B, A, modelo_B_de_A),
        'g_AB': genuino_observado(A, B, modelo_A_de_B, 50000),
        'g_BA': genuino_observado(B, A, modelo_B_de_A, 60000),
    }


def _tabla_alt1(titulo, a1):
    print(f"\n  {titulo}")
    print(f"    {'banda':>6} | {'genuino':>8} {'umbral_nulo(p95)':>17} | veredicto")
    print(f"    {'-'*6}-+-{'-'*26}-+----------")
    for b in BANDAS:
        r = a1[b]; mk = '✅' if r['pasa'] else '·'
        nota = 'supera el ruido endogeno' if r['pasa'] else 'dentro del ruido'
        print(f"    {b:>+6.0f} | {r['genuino']:>+8.0%} {r['umbral_nulo_p95']:>+17.0%} | [{mk}] {nota}")

def _tabla_alt2(titulo, a2):
    print(f"\n  {titulo}")
    print(f"    {'banda':>6} | self  modelo  obj* | endogeno   estructural  coincide")
    print(f"    {'-'*6}-+-{'-'*18}-+-{'-'*30}")
    for b in BANDAS:
        r = a2[b]
        se = 'acep' if r['self_side'] else 'rech'; mo = 'acep' if r['model_side'] else 'rech'
        ob = 'acep' if r['obj_side_true'] else 'rech'
        en = 'medible' if r['medible_endogeno'] else '⊘'; es = 'medible' if r['medible_estructural'] else '⊘'
        ck = '✅' if r['coincide'] else '❌'
        print(f"    {b:>+6.0f} | {se:>4} {mo:>6} {ob:>5} | {en:>8}  {es:>11}   [{ck}]")


def main():
    print("=" * 98)
    print("V182D_C2 — UMBRAL ENDOGENO DE ALTERIDAD (ALT1 corregido: nulo pareado por trial)")
    print("=" * 98)
    print("  ALT1 nulo-pareado  -> tamaño desde el ruido de eleccion del organismo (endogeno)")
    print("  ALT2 frontera      -> frontera desde el modelo interno del otro")
    print("  ALT3 tautologia    -> CONTROL (pasa siempre, no falsable)")
    print("=" * 98)

    t0 = time.time()
    R = correr()

    print(f"\n  [disposicion emergente — P(acepta)]   A / B")
    for b in BANDAS:
        pa, pb = R['disp'][b]; print(f"     {b:>+6.0f} | {pa:.2f} / {pb:.2f}")

    print(f"\n{'='*98}\n  ALT 1 — NULO PAREADO POR TRIAL (corregido)\n{'='*98}")
    _tabla_alt1("A modela a B:", R['alt1_AB'])
    _tabla_alt1("B modela a A:", R['alt1_BA'])

    print(f"\n{'='*98}\n  ALT 2 — FRONTERA ENDOGENA\n{'='*98}")
    _tabla_alt2("A modela a B:", R['alt2_AB'])
    _tabla_alt2("B modela a A:", R['alt2_BA'])

    a3_AB = alt3_tautologia(R['g_AB']); a3_BA = alt3_tautologia(R['g_BA'])
    print(f"\n{'='*98}\n  ALT 3 — TAUTOLOGIA (control)\n{'='*98}")
    print(f"    pasa SIEMPRE en ambas direcciones (incl. 0°): "
          f"{all(v['pasa'] for v in a3_AB.values()) and all(v['pasa'] for v in a3_BA.values())}  -> no falsable")

    # ---- SINTESIS ----
    def verdes(a1, a2):
        return [b for b in BANDAS if a2[b]['medible_endogeno'] and a1[b]['pasa']]
    def medibles(a2):
        return [b for b in BANDAS if a2[b]['medible_endogeno']]

    ver_AB = verdes(R['alt1_AB'], R['alt2_AB']); med_AB = medibles(R['alt2_AB'])
    ver_BA = verdes(R['alt1_BA'], R['alt2_BA']); med_BA = medibles(R['alt2_BA'])
    frontera_coincide = all(R['alt2_AB'][b]['coincide'] for b in BANDAS) and \
                        all(R['alt2_BA'][b]['coincide'] for b in BANDAS)
    bidir = (len(ver_AB) >= 1 and ver_AB == med_AB and len(ver_BA) >= 1 and ver_BA == med_BA)

    print(f"\n{'#'*98}\n#  SINTESIS\n{'#'*98}")
    print(f"  ALT1 (tamaño, nulo pareado): A->B pasa {[b for b in BANDAS if R['alt1_AB'][b]['pasa']]}, "
          f"B->A pasa {[b for b in BANDAS if R['alt1_BA'][b]['pasa']]}")
    print(f"  ALT2 (frontera) == ⊘ estructural de V182D_B: {'✅ SI' if frontera_coincide else '❌ NO'}")
    print(f"  ALT3 (tautologia): pasa siempre -> no falsable (control)")
    print(f"\n  CRITERIO TOTALMENTE ENDOGENO (frontera ALT2 + tamaño ALT1, sin margen nuestro):")
    print(f"     A->B: {len(ver_AB)}/{len(med_AB)} bandas endogenas con alteridad   {ver_AB}")
    print(f"     B->A: {len(ver_BA)}/{len(med_BA)} bandas endogenas con alteridad   {ver_BA}")
    print(f"     -> {'✅ ALTERIDAD BIDIRECCIONAL ENDOGENA (sin numero fijado por nosotros)' if bidir else '❌ revisar'}")
    print(f"\n  NOTA: este nulo NO mide I(A;B) (modelado sobre la base rate del otro). Esa cuestion")
    print(f"  —I(A;B) ≈ 0.15 < 0.3 con 3 bandas— es de diseño para V183, separada de ALT1.")
    print(f"\n  tiempo {time.time()-t0:.1f}s")

    os.makedirs("V182_logs", exist_ok=True)
    salida = {'version': 'V182D_C2-nulo-pareado',
              'disposicion': {str(k): v for k, v in R['disp'].items()},
              'alt1_AB': {str(k): v for k, v in R['alt1_AB'].items()},
              'alt1_BA': {str(k): v for k, v in R['alt1_BA'].items()},
              'alt2_AB': {str(k): v for k, v in R['alt2_AB'].items()},
              'alt2_BA': {str(k): v for k, v in R['alt2_BA'].items()},
              'frontera_endogena_coincide_estructural': bool(frontera_coincide),
              'alteridad_bidireccional_endogena': bool(bidir)}
    with open(f"V182_logs/v182d_C2_nulo_pareado_{TS}.json", "w") as f:
        json.dump(salida, f, indent=2)
    print(f"  log: V182_logs/v182d_C2_nulo_pareado_{TS}.json")


if __name__ == "__main__":
    main()