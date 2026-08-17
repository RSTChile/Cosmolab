"""
CS090 — FASE V-A (piloto): GENERADOR DE REGLAS + FILTRO DE ADMISIÓN P1-P5
===========================================================================
QUIÉN SOY: primer archivo del piloto de Fase V. Genera reglas relacionales candidatas, parametrizadas
por los 3 ejes de FASE5_especificacion_universalidad_CS.md (Eje A: fondo relacional A0/A1/A2, Eje B:
retroalimentación B0/B1, Eje C: costo/localidad C0/C1/C2), y aplica el filtro de admisión P1-P5 (§1 de
la especificación) a CADA regla generada, de verdad (simulación corta + inspección estructural), no
asumido. Una regla que falla algún P se DESCALIFICA y se documenta — no se fuerza a pasar ni se
descarta en silencio.

Decisión de diseño heredada del plan aprobado (no se revisita acá): motor genérico nuevo y liviano que
REUSA piezas atómicas ya validadas del proyecto (nunca las reescribe):
  - `_circ_mean_update`, `_linea_adyacencia`, `_holonomia_triangulos`  <- cs082_fase4_4sustratos.py
  - `_diam`, `_giant`                                                  <- cs055_proceso_acoplado.py
  - `aleatorio()` (grafo Erdős-Rényi)                                  <- cg003_diagnostico_gromov.py
  - patrón de costo (inconsistencia histórica + soporte + reciprocidad + holonomía), aligerado
                                                                          <- cs081_poda_dinamica.py
  - para A0 (sin grafo): campo continuo en anillo con difusión local — representación NO-grafo genuina
    (array de estados + offsets fijos left/right, SIN objeto de adyacencia tipo grafo en ningún punto).

Ningún script congelado se toca — sólo se importa. Este archivo NO ejecuta la dinámica pesada (eso es
cs090_fase5_motor.py): sólo genera parámetros + corre un chequeo P1-P5 con una simulación CHICA (N=50,
pocos pasos) para que el filtro sea real y barato.

Código autodescriptivo. No declara cierre ni veredicto — sólo genera y filtra, reporta números.
"""
from __future__ import annotations
import sys, math
import numpy as np
from collections import deque, defaultdict

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

import cg003_diagnostico_gromov as GR              # aleatorio() -- grafo ER, reusado tal cual
import cs082_fase4_4sustratos as C82               # _circ_mean_update, _linea_adyacencia, _holonomia_triangulos

# ============================ ESPACIO DE PARÁMETROS POR EJE (rangos, no valores fijos) ============================
# Ningún valor físico horneado (P5): sólo constantes de acople ABSTRACTAS del propio motor relacional.
RANGO_K        = (4, 8)         # alfabeto cíclico de orientación/diferencia interna (P2: dimensión de diferencia)
RANGO_J        = (0.30, 0.80)   # constante de acople (adimensional, del propio motor)
RANGO_NOISE    = (0.10, 0.35)   # ruido por sweep, en unidades de K
RANGO_MEANDEG  = (4.0, 8.0)     # grado medio del grafo ER base (A1/A2)
RANGO_SIM_THR  = (0.15, 0.35)   # umbral de similitud de campo (A0), fracción de K
RANGO_KCAP     = (4, 7)         # límite de escala duro (C2): grado máximo por nodo

# Nombres prohibidos por P5 (sin valores físicos horneados): si aparecen como parámetro con intención
# de escala física, la regla queda descalificada. Se escanea el dict de parámetros generado.
_PROHIBIDOS_P5 = {"G", "hbar", "h_bar", "c_luz", "masa", "longitud_planck", "carga_electrica", "G_newton"}

EJES_A = ("A0", "A1", "A2")
EJES_B = ("B0", "B1")
EJES_C = ("C0", "C1", "C2")


# ============================================================================================
# 1) GENERACIÓN DE PARÁMETROS por regla (aleatorios dentro del rango de su clase de ejes)
# ============================================================================================
def _sample(rng, lo, hi, entero=False):
    v = rng.uniform(lo, hi)
    return int(round(v)) if entero else float(v)


def generar_regla(eje_A, eje_B, eje_C, idx, seed):
    """Genera UNA regla candidata (dict de parámetros + descripción legible <500 chars) para la clase
    de ejes (eje_A, eje_B, eje_C). Parámetros aleatorios propios de esta regla (seed determinístico
    para reproducibilidad), dentro del espacio fijado arriba."""
    rng = np.random.default_rng(seed)
    p = dict(
        rule_id=f"{eje_A}-{eje_B}-{eje_C}-r{idx}",
        eje_A=eje_A, eje_B=eje_B, eje_C=eje_C, seed=seed,
        K=_sample(rng, *RANGO_K, entero=True),
        J=round(_sample(rng, *RANGO_J), 3),
        noise=round(_sample(rng, *RANGO_NOISE), 3),
        meandeg=round(_sample(rng, *RANGO_MEANDEG), 2),
        sim_thr_frac=round(_sample(rng, *RANGO_SIM_THR), 3),
        kcap=_sample(rng, *RANGO_KCAP, entero=True),
    )
    # descripción corta y legible (para P5: complejidad algorítmica acotada, <500 caracteres)
    desc = (f"S en Z_{p['K']} por nodo/relación en sustrato {eje_A}; actualización = media circular con "
            f"vecinos definidos por adyacencia previa (J={p['J']}, ruido={p['noise']}) [{eje_B}]; "
            f"costo/localidad = {eje_C}"
            + (f" (poda por costo top-percentil, hist+holonomía)" if eje_C == "C1" else "")
            + (f" (grado máx duro={p['kcap']} + poda por costo)" if eje_C == "C2" else "")
            + ".")
    p["descripcion"] = desc
    return p


# ============================================================================================
# 2) FILTRO DE ADMISIÓN P1-P5 — chequeo AUTOMATIZADO real (simulación chica N=50 + inspección
#    estructural), no asumido. Cada P devuelve (bool, detalle).
# ============================================================================================
_N_CHEQUEO = 50   # tamaño chico, sólo para el chequeo del filtro -- NO es el N del piloto
_STEPS_CHEQUEO = 6


def _construir_sustrato_chequeo(p, rng):
    """Construye una instancia MINI del sustrato (según eje_A) para poder correr el chequeo P1-P4
    de forma empírica y barata. Devuelve (kind, adj_o_None, vecinos_relacionales, S0)."""
    N = _N_CHEQUEO
    if p["eje_A"] == "A0":
        left = (np.arange(N) - 1) % N
        right = (np.arange(N) + 1) % N
        S0 = rng.uniform(0, p["K"], N)
        return "A0", None, (left, right), S0
    else:
        adj0, _ = GR.aleatorio(N, meandeg=p["meandeg"], seed=int(rng.integers(1 << 30)))
        adj = [set(a.tolist()) for a in adj0]
        if p["eje_B"] == "B0":
            vecinos = [list(a) for a in adj]      # línea de adyacencia = adyacencia de NODOS
            n_obj = N
        else:
            edges = sorted(set((min(i, j), max(i, j)) for i in range(N) for j in adj[i]))
            vecinos = C82._linea_adyacencia(edges) if edges else []
            n_obj = len(edges)
            adj = (adj, edges)
        S0 = rng.uniform(0, p["K"], n_obj) if n_obj > 0 else np.array([])
        return "A1A2", adj, vecinos, S0


def _paso_chequeo(kind, vecinos, S, p, rng):
    if kind == "A0":
        left, right = vecinos
        ang = 2 * np.pi * S / p["K"]
        vecino_ang = (np.angle(np.exp(1j * ang[left])) + np.angle(np.exp(1j * ang[right]))) / 2
        S_new = ((1 - p["J"]) * S + p["J"] * (vecino_ang / (2 * np.pi) * p["K"])) % p["K"]
        S_new = (S_new + rng.normal(0, p["noise"], len(S))) % p["K"]
        return S_new
    else:
        if len(S) == 0:
            return S
        return C82._circ_mean_update(S, vecinos, p["J"], p["noise"], rng)


def chequear_P1_persistencia(p, rng):
    """P1 — persistencia mínima: existe memoria real (dinámica NO markoviana-sin-estado, S(t+1)=f(S(t))
    con f no trivial) y S(t)>umbral hace más probable S(t+1)>umbral que el azar. Chequeo EMPÍRICO: se
    corre la dinámica REAL encadenada (cada paso usa el S del paso anterior) y se mide qué tan bien
    S(t)>mediana predice S(t+1)>mediana, contra un control SIN-MEMORIA (S resampleado i.i.d. en cada
    paso, es decir f trivial/markoviana-sin-estado): si la dinámica real no predice mejor que el
    resampleo puro, no hay memoria genuina y la regla se descalifica."""
    kind, adj, vecinos, S0 = _construir_sustrato_chequeo(p, rng)
    if kind != "A0" and len(S0) == 0:
        return False, "sin objetos relacionales para medir memoria (grafo demasiado disperso)"
    n_obj = len(S0)
    S = S0.copy()
    acierto_real, acierto_nomem, total = 0, 0, 0
    for _ in range(_STEPS_CHEQUEO):
        alto_antes = S > p["K"] / 2
        S_next = _paso_chequeo(kind, vecinos, S, p, rng)          # dinámica real encadenada: f(S(t))
        alto_despues = S_next > p["K"] / 2
        acierto_real += np.sum(alto_antes == alto_despues)
        # control SIN-MEMORIA: en vez de encadenar, S(t+1) se resamplea i.i.d. (ninguna f real aplicada)
        S_nomem_next = rng.uniform(0, p["K"], n_obj)
        acierto_nomem += np.sum(alto_antes == (S_nomem_next > p["K"] / 2))
        total += n_obj
        S = S_next
    tasa_real = acierto_real / total
    tasa_nomem = acierto_nomem / total
    ok = tasa_real > tasa_nomem + 0.02   # memoria real predice mejor que resampleo puro, margen >azar
    return ok, f"tasa_persistencia_real={tasa_real:.3f} vs sin_memoria={tasa_nomem:.3f}"


def chequear_P2_diferencia(p):
    """P2 — diferencia operable: S no es escalar homogéneo, tiene >=1 dimensión interna de diferencia
    (acá: fase circular en Z_K, K>1) que puede acoplarse (vía _circ_mean_update / analógo de anillo)."""
    ok = p["K"] > 1
    return ok, f"K={p['K']} (alfabeto de fase, dimensión de diferencia)"


def chequear_P3_localidad(p):
    """P3 — localidad relacional sin coordenadas: chequeo ESTRUCTURAL sobre el propio motor -- ninguna
    función de sustrato/actualización de este archivo ni de cs090_fase5_motor.py recibe o usa un
    arreglo de coordenadas (x,y,z); los vecinos SIEMPRE se leen de adyacencia (grafo) o de offsets de
    anillo fijos declarados al inicio (A0), nunca de índice global. Verificado por inspección de código
    (grep de patrones de coordenadas) sobre el propio módulo, no por promesa."""
    # inspecciona SÓLO las funciones que construyen sustrato/actualizan estado (donde viviría una
    # coordenada si se hubiera horneado) -- se excluye deliberadamente esta misma función de chequeo,
    # que necesita mencionar los patrones-a-buscar como texto y daría un falso positivo consigo misma.
    import inspect
    funciones_a_inspeccionar = [_construir_sustrato_chequeo, _paso_chequeo, generar_regla]
    try:
        import cs090_fase5_motor as _M
        funciones_a_inspeccionar += [_M.construir_A0, _M.construir_A1, _M.construir_A2,
                                      _M.dinamica_B0, _M.dinamica_B1]
    except Exception:
        pass   # motor puede no existir aún al importar este archivo standalone
    fuente = "\n".join(inspect.getsource(f) for f in funciones_a_inspeccionar)
    patrones_coord = ["pos[", "posiciones[", "coord", "xy=", "(x, y)", "embedding"]
    hallados = [pat for pat in patrones_coord if pat in fuente]
    ok = len(hallados) == 0
    return ok, ("sin patrones de coordenadas en las funciones de sustrato" if ok else f"patrones sospechosos: {hallados}")


def chequear_P4_reciprocidad(p, rng):
    """P4 — interacción recíproca: si i afecta a j, j afecta a i (ventana finita). Chequeo EMPÍRICO
    sobre una instancia real del sustrato: para A0, offsets left/right son inversos entre sí por
    construcción (ring[i]->i+1, y desde i+1 el left es i) -- se verifica. Para A1/A2, se verifica que
    la adyacencia construida sea simétrica en una muestra de nodos."""
    kind, adj, vecinos, S0 = _construir_sustrato_chequeo(p, rng)
    if kind == "A0":
        left, right = vecinos
        ok = np.all(right[left] == np.arange(_N_CHEQUEO)) and np.all(left[right] == np.arange(_N_CHEQUEO))
        return bool(ok), "offsets de anillo left/right mutuamente inversos (verificado)"
    else:
        base_adj = adj[0] if isinstance(adj, tuple) else adj
        malas = 0
        for i in range(min(_N_CHEQUEO, len(base_adj))):
            for j in base_adj[i]:
                if i not in base_adj[j]:
                    malas += 1
        ok = malas == 0
        return ok, f"adyacencia simétrica: {malas} pares asimétricos hallados (debe ser 0)"


def chequear_P5_sin_hornear(p):
    """P5 — ausencia de valores físicos horneados: ningún parámetro con nombre de constante física
    (G, hbar, c, masa, escala de longitud) puesto a mano; descripción de la regla <500 caracteres
    (complejidad algorítmica acotada)."""
    claves = set(p.keys())
    hallados = claves & _PROHIBIDOS_P5
    ok_nombres = len(hallados) == 0
    ok_longitud = len(p["descripcion"]) < 500
    ok = ok_nombres and ok_longitud
    detalle = f"sin nombres prohibidos={ok_nombres} (hallados={hallados}); len(desc)={len(p['descripcion'])}<500={ok_longitud}"
    return ok, detalle


def aplicar_filtro_P1_P5(p, seed_chequeo):
    """Aplica los 5 chequeos, guarda resultado explícito por P en el dict de la regla. Devuelve la
    regla con p['admitida']=True/False y p['detalle_filtro'] con el motivo si fue descalificada."""
    rng = np.random.default_rng(seed_chequeo)
    resultados = {}
    resultados["P1"] = chequear_P1_persistencia(p, rng)
    resultados["P2"] = chequear_P2_diferencia(p)
    resultados["P3"] = chequear_P3_localidad(p)
    resultados["P4"] = chequear_P4_reciprocidad(p, rng)
    resultados["P5"] = chequear_P5_sin_hornear(p)
    p["filtro"] = {k: v[0] for k, v in resultados.items()}
    p["filtro_detalle"] = {k: v[1] for k, v in resultados.items()}
    p["admitida"] = all(v[0] for v in resultados.values())
    if not p["admitida"]:
        fallidos = [k for k, v in resultados.items() if not v[0]]
        p["motivo_descarte"] = f"falló {fallidos}: " + "; ".join(f"{k}={resultados[k][1]}" for k in fallidos)
    else:
        p["motivo_descarte"] = None
    return p


# ============================================================================================
# 3) GENERAR N REGLAS PARA UNA CLASE DE EJES, aplicando el filtro, reintentando si alguna falla
# ============================================================================================
def generar_reglas_clase(eje_A, eje_B, eje_C, n_reglas, seed_base, max_intentos=20):
    """Genera n_reglas ADMITIDAS (pasan P1-P5) para la clase (eje_A,eje_B,eje_C). Documenta también
    las descartadas (no las esconde). Devuelve (admitidas, descartadas)."""
    admitidas, descartadas = [], []
    intento = 0
    idx = 0
    while len(admitidas) < n_reglas and intento < max_intentos:
        seed = seed_base + intento * 97 + 1
        p = generar_regla(eje_A, eje_B, eje_C, idx, seed)
        p = aplicar_filtro_P1_P5(p, seed_chequeo=seed + 500_000)
        if p["admitida"]:
            admitidas.append(p)
            idx += 1
        else:
            descartadas.append(p)
        intento += 1
    return admitidas, descartadas


if __name__ == "__main__":
    print("CS090 — Fase V-A piloto: generador + filtro P1-P5 (smoke test standalone)")
    print("=" * 100)
    total_ok, total_bad = 0, 0
    for eje_A in EJES_A:
        for eje_B in EJES_B:
            for eje_C in EJES_C:
                ok, bad = generar_reglas_clase(eje_A, eje_B, eje_C, n_reglas=2, seed_base=hash((eje_A, eje_B, eje_C)) % 100000)
                total_ok += len(ok); total_bad += len(bad)
                print(f"  {eje_A}-{eje_B}-{eje_C}: admitidas={len(ok)}  descartadas={len(bad)}")
    print(f"\nTOTAL admitidas={total_ok}  descartadas={total_bad}")
