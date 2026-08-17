"""
CS090 — FASE VI / O2-C: ¿`kcap=6` ES UN NÚMERO ESPECIAL, O CUALQUIER SATURACIÓN RELACIONAL SIRVE?
=====================================================================================================
QUIÉN SOY (archivo NUEVO; no toco ni una línea de ningún script anterior — sólo importo)
------------------------------------------------------------------------------------------------
La línea F5-C2-C→C5 dejó establecido que lo que produce geometría extendida en A2-B0-C2 es la
**rigidez del corte** (cupo exacto, sin excepción, en un paso) **+ el criterio de soporte local**, y no
la señal de costo ni la elasticidad del presupuesto (`FASE5_matriz_2x2_completa_CS.md`,
`FASE5_control_azar_elastico_CS.md`). La auditoría de `FASE5_auditoria_C2_resultado_CS.md` mostró
además que `kcap` es puramente topológico: no esconde coordenadas ni distancias (PASS).

Pero `kcap` sigue siendo un número puesto DESDE AFUERA. Esta tarea (propuesta F6-06 de GPT-5.6 Sol)
pregunta si ese número concreto importa, con tres pruebas:

  PRUEBA 1 — `kcap` ABSOLUTO fijo, N variable (500/1000/2000/4000; kcap = 4, 5, 6).
             ¿La geometría depende del valor absoluto de kcap o de kcap relativo al tamaño?
  PRUEBA 2 — capacidad NORMALIZADA al tamaño del sistema (dos normalizaciones, ver §2).
             ¿Alguna normalización conserva la familia de geometrías a través de escalas?
  PRUEBA 3 — capacidad HETEROGÉNEA entre nodos, MISMA media (5 distribuciones, ver §3).
             ¿Hace falta homogeneidad, o basta con que la capacidad sea finita?

LO QUE NO SE HACE ACÁ (y por qué): no se prueba una capacidad EMERGENTE derivada del historial/costo
del nodo. Ya se probó exhaustivamente en F5-C2-C→C5 y el presupuesto elástico NO reproduce el cupo
rígido (`FASE5_presupuesto_emergente_CS.md`, `FASE5_presupuesto_soporte_local_CS.md`,
`FASE5_control_azar_elastico_CS.md`). Repetirlo sería gastar cómputo en una pregunta ya contestada.

OBSERVABLE PRINCIPAL: LA PENDIENTE CONTINUA, MEDIDA CON EL DIÁMETRO CORREGIDO
------------------------------------------------------------------------------------------------
Consigna metodológica explícita del equipo para esta tarea: **NO usar "% Clase III" como endpoint**.
Motivo medido: el escalón de clasificación pierde información contra la rampa continua (R²=0.663 del
observable continuo vs 0.182 del escalón oficial, `FASE6_reanalisis_azar_continuo_CS.md`). Acá el
endpoint es la **pendiente de log(diámetro) vs log(N_cajas)** sobre el coarse-graining b=1,2,4,8,16
(la misma vara con que se calibraron los umbrales en `cs080_renormalizacion.py`), midiendo el diámetro
con `cs090_diam_corregido.diam_gigante` (componente gigante) — REGLA VIGENTE desde el 11-ago-2026, NO
el `_diam` de cs055. Las clases I-IV se calculan igual y se reportan como dato SECUNDARIO, sólo para
continuidad con los informes anteriores.

Analogía de la pendiente, en simple: agrupamos el grafo en "barrios" cada vez más grandes y medimos
cuánto se tarda en cruzar el mapa resultante. Si al agrandar el mapa el cruce se alarga proporcional-
mente (pendiente alta ≈ 0.7-1), la cosa tiene EXTENSIÓN, como una ciudad. Si el cruce casi no se
alarga (pendiente baja ≈ 0.3), todo está a dos pasos de todo: es un mundo-pequeño enrollado, sin
geometría propia.

QUÉ SE REUSA, TAL CUAL, SIN TOCAR
------------------------------------------------------------------------------------------------
  · `cs090_fase5_generador.generar_reglas_clase`      — reglas A2-B0-C2 con filtro P1-P5 real.
  · `cs090_fase5_motor.construir_A2`, `_recablear_A2`,
    `_costo_y_podar`, `_muestrear_triangulos`, `medir`,
    `medir_null_valor`, `GR.aleatorio`                — motor congelado.
  · `cs090_fase5_mecanismo_aislado.dinamica_B0_hibrido` — la dinámica A2-B0 con cupo POR NODO, ya
    escrita y usada en F5-C2-C3/C4. Con un vector de cupo CONSTANTE es idéntica al C2-hard del motor
    (mismo ranking por soporte, mismo corte estricto) — se verifica numéricamente en `--verificar`,
    no se asume.
  · `cs090_fase5_mecanismo_aislado._cupo_variable`    — la heterogeneidad "∝ grado inicial en el ER"
    que ya se probó en F5-C2-C3, reusada como UNA de las 5 formas de la Prueba 3 (pedido explícito).
  · `cs080_renormalizacion.cajas_bfs`, `grafo_grueso` — coarse-graining.
  · `cs090_diam_corregido.diam_gigante`, `giant`      — medición oficial del diámetro.
  · `cs090_fase5_clasificador.clasificar_regla`       — clases I-IV (dato secundario).

Único código genuinamente nuevo: las funciones que FABRICAN el vector de cupo por nodo (§1) y el
armado de la grilla. La dinámica, la medición y la clasificación son las de siempre.

CONTROL DE COMPARABILIDAD (por qué las comparaciones son pareadas y no confundidas)
------------------------------------------------------------------------------------------------
  · Para un mismo (regla, N), TODAS las condiciones parten del MISMO grafo Erdős-Rényi inicial y usan
    el MISMO flujo de números aleatorios de la dinámica (`rng` derivado de `p["seed"]*5000+N`). El
    sorteo del cupo heterogéneo usa un generador APARTE (`rng_cupo`), para que cambiar la distribución
    de capacidad no desplace el ruido de la dinámica. Así, la única diferencia entre condiciones es el
    vector de capacidad.
  · Todas las distribuciones heterogéneas se reescalan para que su media empírica sea ≈ `kcap_base`:
    la "masa total de cupo" es la misma, sólo cambia CÓMO se reparte (mismo principio que ya usó
    `_cupo_variable`). La media y el coeficiente de variación EFECTIVOS se guardan en el CSV, no se
    asumen.
  · Se sobreescribe `p["kcap"]` después del filtro P1-P5. Es legítimo: el filtro (generador, §2) corre
    una simulación mini SIN el enforcement de C2 — ninguno de los 5 chequeos lee `kcap`. Verificado
    por inspección del código del generador, se documenta acá para que no quede como supuesto.

No se corre Phantom. No se declara cierre ni veredicto: se reportan números. No se hacen commits.
"""
from __future__ import annotations
import argparse, csv, sys, time
import numpy as np
from collections import Counter

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cs090_fase5_generador as GEN
import cs090_fase5_motor as MOT
import cs090_fase5_mecanismo_aislado as MA            # dinamica_B0_hibrido + _cupo_variable — NO SE TOCA
import cs080_renormalizacion as CS80
import cs090_diam_corregido as DC                     # medición oficial del diámetro
from cs090_fase5_clasificador import clasificar_regla

EJE_A, EJE_B, EJE_C = "A2", "B0", "C2"
N_SWEEPS = 14                     # mismo que toda la línea Fase V/VI
ESCALAS_B = (1, 2, 4, 8, 16)      # mismas escalas con que se calibraron los umbrales (cs080)
N_SEEDS_NULL_TOPO = 3
SEED_BASE = 90210                 # MISMA base que F5-C2-C/C2/C3/C4 -> mismas reglas, comparable
N_ANCLA = 2000                    # tamaño de referencia de toda la línea A2-B0-C2
KCAP_ANCLA = 6                    # cupo de referencia (el "6" cuya especialidad se interroga)


# ============================================================================================
# 1) FÁBRICAS DE VECTOR DE CUPO POR NODO  (lo único genuinamente nuevo de este archivo)
# ============================================================================================
def _reescalar_a_media(v, objetivo, minimo=1):
    """Toma un vector real positivo `v` de "capacidades relativas" y lo convierte en cupos ENTEROS
    cuya media empírica sea lo más cercana posible a `objetivo`, con piso `minimo`.

    Por qué reescalar: si una distribución heterogénea tuviera, sin querer, más cupo total que el caso
    uniforme, cualquier diferencia de geometría podría deberse simplemente a "hay más aristas", no a la
    forma de la distribución. Reescalando la media, la única variable que cambia entre condiciones es
    la FORMA del reparto. Mismo principio que ya usa `MA._cupo_variable` (media empírica, no teórica).

    El piso `minimo=1` es la misma desviación declarada que `_cupo_variable`: un nodo con cupo 0 se
    quedaría sin ninguna relación para siempre tras la primera poda, y dejaría de ser un nodo
    relacional. Como el piso empuja la media hacia arriba, después se corrige el factor de escala
    iterando unas pocas veces (búsqueda simple, no una fórmula cerrada) hasta que la media empírica
    del vector entero final cae lo más cerca posible del objetivo.
    """
    v = np.asarray(v, dtype=float)
    v = np.maximum(v, 0.0)
    media_v = float(v.mean())
    if media_v <= 1e-12:
        return np.full(len(v), max(minimo, int(round(objetivo))), dtype=int)
    escala = objetivo / media_v
    mejor, mejor_err = None, np.inf
    for _ in range(40):                       # ajuste iterativo del factor por el sesgo del piso/redondeo
        k = np.maximum(np.round(v * escala).astype(int), minimo)
        err = float(k.mean()) - objetivo
        if abs(err) < mejor_err:
            mejor, mejor_err = k, abs(err)
        if abs(err) < 0.01:
            break
        escala *= objetivo / max(1e-9, float(k.mean()))
    return mejor


def cupo_uniforme(N, kcap_base, grado_inicial, rng):
    """HOMOGÉNEO — todos los nodos con el MISMO cupo. Es exactamente el C2-hard del motor congelado
    (`MOT._enforce_kcap` con `p["kcap"]`), expresado como vector. Coeficiente de variación = 0."""
    return np.full(N, max(1, int(round(kcap_base))), dtype=int)


def cupo_prop_grado(N, kcap_base, grado_inicial, rng):
    """HETEROGÉNEO #1 — cupo ∝ grado inicial en el grafo ER. Reusa `MA._cupo_variable` TAL CUAL (la
    misma forma ya probada en F5-C2-C3, que dio 35-40% Clase III, cerca del cupo fijo). La dispersión
    es la del muestreo Poisson del ER: modesta (CV ≈ 1/sqrt(meandeg) ≈ 0.4)."""
    return MA._cupo_variable(grado_inicial, kcap_base)


def cupo_uniforme_rango(N, kcap_base, grado_inicial, rng, semiancho=3):
    """HETEROGÉNEO #2 — cupo sorteado UNIFORME en un rango simétrico alrededor de la media
    (por defecto kcap_base-3 … kcap_base+3, con piso 1). Dispersión moderada y sin cola: es la forma
    más "plana" posible de heterogeneidad, el contraste directo contra el caso homogéneo."""
    lo = max(1, int(round(kcap_base)) - semiancho)
    hi = int(round(kcap_base)) + semiancho
    v = rng.integers(lo, hi + 1, size=N).astype(float)
    return _reescalar_a_media(v, kcap_base)


def cupo_bimodal(N, kcap_base, grado_inicial, rng, bajo=2):
    """HETEROGÉNEO #3 — BIMODAL: mitad de los nodos "pobres" en relación (cupo `bajo`), mitad "ricos"
    (cupo alto = 2*kcap_base - bajo, para que la media siga siendo kcap_base). Ningún nodo tiene el
    cupo promedio: es la heterogeneidad como SEGREGACIÓN, no como ruido. Analogía: en vez de que todos
    tengan 6 amigos, la mitad tiene 2 y la mitad tiene 10 — mismo total de amistades en el pueblo."""
    alto = max(1, int(round(2 * kcap_base - bajo)))
    # se asigna una CANTIDAD EXACTA de nodos "ricos" (no una moneda por nodo): con dos valores fijos,
    # el sorteo independiente deja la media empírica a merced del azar binomial (±3-4%) y el reescalado
    # entero no puede corregirlo (sólo hay 2 valores posibles). Fijar el conteo hace que la masa total
    # de cupo sea la misma que en el caso uniforme por construcción, no por suerte.
    v = np.full(N, float(max(1, bajo)))
    n_altos = int(round(N * (kcap_base - max(1, bajo)) / max(1e-9, alto - max(1, bajo))))
    n_altos = int(np.clip(n_altos, 0, N))
    v[rng.permutation(N)[:n_altos]] = float(alto)
    return _reescalar_a_media(v, kcap_base)


def cupo_powerlaw(N, kcap_base, grado_inicial, rng, alpha=2.5):
    """HETEROGÉNEO #4 — LEY DE POTENCIAS (Pareto, exponente alpha): la mayoría de los nodos con cupo
    chico y unos pocos "hubs" con cupo enorme. Es la heterogeneidad de cola pesada, la más lejana
    posible del caso uniforme con la MISMA media. Analogía: casi todos con 2-3 amigos y un puñado de
    personas con cientos."""
    u = rng.random(N)
    v = (1.0 - u) ** (-1.0 / (alpha - 1.0))       # Pareto(x_m=1, alpha) por inversión de la CDF
    return _reescalar_a_media(v, kcap_base)


def cupo_lognormal(N, kcap_base, grado_inicial, rng, sigma=0.8):
    """HETEROGÉNEO #5 — LOGNORMAL: cola derecha pero mucho más suave que la ley de potencias, sin
    hubs extremos. Sirve de punto INTERMEDIO en el eje de dispersión (entre el rango uniforme y la ley
    de potencias), para poder leer si hay dosis-respuesta y no sólo dos extremos."""
    v = rng.lognormal(mean=0.0, sigma=sigma, size=N)
    return _reescalar_a_media(v, kcap_base)


FABRICAS_CUPO = {
    "UNIF":       cupo_uniforme,
    "HET-grado":  cupo_prop_grado,
    "HET-rango":  cupo_uniforme_rango,
    "HET-lognor": cupo_lognormal,
    "HET-bimodal": cupo_bimodal,
    "HET-potencia": cupo_powerlaw,
}


# ============================================================================================
# 2) NORMALIZACIONES DE CAPACIDAD AL TAMAÑO DEL SISTEMA (Prueba 2)
# ============================================================================================
# La pregunta de la Prueba 2 es si existe alguna forma de "capacidad relativa" que, mantenida
# constante, conserve la familia de geometrías al cambiar N. Las candidatas y por qué:
#
#   ABS       — kcap constante en valor absoluto (es la Prueba 1; se incluye acá como referencia).
#   NORM-log  — kcap ∝ log N, anclado en kcap=KCAP_ANCLA a N=N_ANCLA. Justificación: en un grafo
#               disperso el diámetro típico escala como log N / log(grado); pedir "kcap ∝ log N" es
#               pedir que la capacidad crezca al mismo ritmo que la escala natural de recorrido del
#               sistema. Es la normalización más conservadora que sí varía con N.
#   NORM-pot  — kcap ∝ N^(1/3), mismo anclaje. Justificación: si uno espera que el sustrato tenga una
#               extensión de tipo volumen, la capacidad "por sitio" que mantiene constante la razón
#               entre capacidad y dimensión LINEAL del sistema crece como N^(1/d) con d=3. Es la
#               normalización agresiva; da una secuencia de cupos claramente distinta a NORM-log.
#   NORM-grado— kcap / ⟨grado inicial⟩ constante. RESULTADO ANALÍTICO, NO HACE FALTA CORRERLA: el
#               generador construye el ER con `meandeg` fijado por la regla, INDEPENDIENTE de N, así
#               que ⟨grado inicial⟩ no cambia con N y esta normalización COLAPSA exactamente sobre ABS.
#               Se documenta como degenerada en vez de gastar cómputo en re-medir la Prueba 1.
#   (nota) Una cuarta candidata, "factor de ramificación constante" (kcap tal que log N / log(kcap-1)
#          sea el mismo a toda escala), da en este rango de N la MISMA secuencia entera que NORM-log
#          (5,5,6,7) — se verifica en `kcap_normalizado` y por eso no se corre por separado.
def kcap_normalizado(norma, N, kcap_ancla=KCAP_ANCLA, n_ancla=N_ANCLA):
    """Devuelve el cupo ENTERO que corresponde a tamaño N bajo la normalización pedida, anclado en
    (n_ancla, kcap_ancla) — por construcción, todas las normalizaciones coinciden en N=n_ancla."""
    if norma == "ABS":
        return int(kcap_ancla)
    if norma == "NORM-log":
        return max(1, int(round(kcap_ancla * np.log(N) / np.log(n_ancla))))
    if norma == "NORM-pot":
        return max(1, int(round(kcap_ancla * (N / n_ancla) ** (1.0 / 3.0))))
    if norma == "NORM-ramif":
        # factor de ramificación constante: log N / log(kcap-1) = log n_ancla / log(kcap_ancla-1)
        D = np.log(n_ancla) / np.log(max(1.0001, kcap_ancla - 1.0))
        return max(1, int(round(np.exp(np.log(N) / D) + 1.0)))
    raise ValueError(f"normalización desconocida: {norma}")


# ============================================================================================
# 3) UNA CORRIDA: dinámica A2-B0 con vector de cupo + coarse-graining + diámetro CORREGIDO
# ============================================================================================
def correr_condicion(p, N, nombre_cupo, kcap_base, seed_cupo, n_sweeps=N_SWEEPS,
                     escalas_b=ESCALAS_B, n_seeds_null_topo=N_SEEDS_NULL_TOPO):
    """Copia estructural de `MA.correr_regla_coarse_hibrido` (misma construcción de sustrato, mismas
    semillas derivadas, mismo coarse-graining, mismos NULL emparejados) con DOS cambios explícitos:
      (a) el vector de cupo lo fabrica una de las funciones de §1 (no siempre `_cupo_variable`);
      (b) el diámetro se mide con `DC.diam_gigante` (regla vigente), tanto en REAL como en los NULL —
          corregir un solo lado inventaría una asimetría en el z-score.
    Devuelve (filas_por_escala, info_del_cupo)."""
    rng = np.random.default_rng(p["seed"] * 5000 + N)      # MISMO flujo que toda la línea A2-B0-C2
    sustrato = MOT.construir_A2(N, rng, p)
    grado_inicial = np.array([len(sustrato["adj"][i]) for i in range(N)], dtype=float)

    rng_cupo = np.random.default_rng(seed_cupo)            # stream APARTE: no desplaza el ruido de la dinámica
    cupo = FABRICAS_CUPO[nombre_cupo](N, kcap_base, grado_inicial, rng_cupo)
    cupo = np.asarray(cupo, dtype=int)
    info_cupo = dict(kcap_base=kcap_base, kcap_media_emp=float(cupo.mean()),
                     kcap_sd=float(cupo.std()),
                     kcap_cv=float(cupo.std() / cupo.mean()) if cupo.mean() > 0 else float("nan"),
                     kcap_min=int(cupo.min()), kcap_max=int(cupo.max()),
                     grado_inicial_medio=float(grado_inicial.mean()))

    sustrato = MA.dinamica_B0_hibrido(sustrato, p, rng, n_sweeps, cupo, "soporte")
    m = MOT.medir(sustrato, p, rng)
    adj_real = m["adj_final"]

    nv = MOT.medir_null_valor(m, p, np.random.default_rng(p["seed"] * 8000 + N))
    holon_real_native, holon_null_native = m["holonomia"], nv["holonomia"]

    meandeg_equiv = max(0.5, 2.0 * m["n_aristas"] / max(1, N))
    adjs_null = []
    for s in range(n_seeds_null_topo):
        adj0, _ = MOT.GR.aleatorio(N, meandeg=meandeg_equiv, seed=int(p["seed"] * 6000 + N * 13 + s))
        adjs_null.append([set(a.tolist()) for a in adj0])

    filas = []
    t0 = time.time()
    for b in escalas_b:
        if b == 1:
            adj_g, n_cajas = adj_real, N
        else:
            rng_b = np.random.default_rng(p["seed"] * 7000 + b * 31)
            asign, n_cajas = CS80.cajas_bfs(adj_real, N, b, rng_b)
            adj_g = CS80.grafo_grueso(adj_real, N, asign, n_cajas)
        diam_g = float(DC.diam_gigante(adj_g, n_cajas)) if n_cajas > 1 else float("nan")
        giant_g = float(DC.giant(adj_g, n_cajas)) if n_cajas > 1 else 0.0
        n_aristas_g = sum(len(a) for a in adj_g) // 2

        diam_nulls = []
        for k_null, adj_n in enumerate(adjs_null):
            if b == 1:
                adj_ng, n_cajas_n = adj_n, N
            else:
                rng_bn = np.random.default_rng(p["seed"] * 7500 + b * 37 + k_null)
                asign_n, n_cajas_n = CS80.cajas_bfs(adj_n, N, b, rng_bn)
                adj_ng = CS80.grafo_grueso(adj_n, N, asign_n, n_cajas_n)
            diam_nulls.append(float(DC.diam_gigante(adj_ng, n_cajas_n)) if n_cajas_n > 1 else float("nan"))

        filas.append(dict(
            rule_id=p["rule_id"], N=n_cajas, escala_b=b, dt=round(time.time() - t0, 2),
            diam_real=diam_g, giant_real=giant_g, holon_real=holon_real_native,
            n_aristas=n_aristas_g, n_triangulos=m["n_triangulos"],
            diam_null_topo=float(np.nanmean(diam_nulls)),
            diam_null_topo_std=float(np.nanstd(diam_nulls)) + 1e-9,
            holon_null_valor=holon_null_native,
        ))
    return filas, info_cupo


def fila_resumen(p, N, test, condicion, filas, info_cupo, dt):
    """Aplana una corrida a UNA fila de CSV: el observable principal (pendiente continua corregida)
    primero, la clase I-IV después, como dato secundario."""
    r = clasificar_regla(filas)
    f1 = next(f for f in filas if f["escala_b"] == 1)
    return dict(
        test=test, condicion=condicion, rule_id=p["rule_id"], seed=p["seed"], N_sistema=N,
        pendiente=r["pendiente_real"],                      # <-- OBSERVABLE PRINCIPAL
        pendiente_null=r["pendiente_null"], z_agg=r["z_agg"], clase=r["clase"],
        holon_ratio=r["holon_ratio"],
        n_aristas_b1=f1["n_aristas"], grado_medio_b1=round(2 * f1["n_aristas"] / f1["N"], 3),
        diam_b1=f1["diam_real"], giant_b1=f1["giant_real"],
        diam_por_escala="|".join(f"{f['diam_real']:.0f}" for f in filas),
        ncajas_por_escala="|".join(str(f["N"]) for f in filas),
        K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"],
        dt=round(dt, 2), **info_cupo,
    )


# ============================================================================================
# 4) LOTE DE REGLAS (mismas de siempre; sólo se sobreescribe kcap donde el diseño lo pide)
# ============================================================================================
def obtener_reglas(n_reglas, seed_base=SEED_BASE + 1):
    admitidas, descartadas = GEN.generar_reglas_clase(EJE_A, EJE_B, EJE_C, n_reglas=n_reglas,
                                                      seed_base=seed_base,
                                                      max_intentos=max(80, n_reglas * 4))
    print(f"reglas admitidas={len(admitidas)}  descartadas(P1-P5)={len(descartadas)}  "
          f"(seed_base={seed_base}, mismas que F5-C2-C/C2/C3/C4)")
    return admitidas


def guardar_csv(filas, ruta):
    if not filas:
        print(f"(sin filas para {ruta})")
        return
    campos = list(filas[0].keys())
    with open(ruta, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=campos)
        wr.writeheader()
        for f in filas:
            wr.writerow(f)
    print(f"CSV: {ruta}  ({len(filas)} filas)")


# ============================================================================================
# 5) LAS TRES PRUEBAS
# ============================================================================================
def prueba1(reglas, Ns, kcaps, presupuesto_seg=3600):
    """kcap ABSOLUTO fijo, N variando en varios órdenes."""
    out = []
    t_ini = time.time()
    for kc in kcaps:
        for N in Ns:
            for p in reglas:
                p2 = dict(p); p2["kcap"] = int(kc)
                t0 = time.time()
                filas, info = correr_condicion(p2, N, "UNIF", kc, seed_cupo=p["seed"] * 11 + N)
                fr = fila_resumen(p2, N, "P1-abs", f"kcap={kc}", filas, info, time.time() - t0)
                out.append(fr)
                print(f"  [P1] kcap={kc} N={N:<5} {p['rule_id']:<14} pend={fr['pendiente']:+.3f} "
                      f"grado={fr['grado_medio_b1']:.2f} diam_b1={fr['diam_b1']:.0f} clase={fr['clase'][:3]:<3} "
                      f"({fr['dt']:.1f}s) [t={time.time()-t_ini:.0f}s]", flush=True)
                if time.time() - t_ini > presupuesto_seg:
                    print("  *** SALVAGUARDA DE TIEMPO (P1) ***"); return out
    return out


def prueba2(reglas, Ns, normas, presupuesto_seg=3600):
    """Capacidad NORMALIZADA al tamaño del sistema."""
    out = []
    t_ini = time.time()
    for norma in normas:
        for N in Ns:
            kc = kcap_normalizado(norma, N)
            for p in reglas:
                p2 = dict(p); p2["kcap"] = int(kc)
                t0 = time.time()
                filas, info = correr_condicion(p2, N, "UNIF", kc, seed_cupo=p["seed"] * 11 + N)
                fr = fila_resumen(p2, N, "P2-norm", norma, filas, info, time.time() - t0)
                out.append(fr)
                print(f"  [P2] {norma:<10} N={N:<5} kcap={kc} {p['rule_id']:<14} "
                      f"pend={fr['pendiente']:+.3f} grado={fr['grado_medio_b1']:.2f} "
                      f"clase={fr['clase'][:3]:<3} ({fr['dt']:.1f}s) [t={time.time()-t_ini:.0f}s]", flush=True)
                if time.time() - t_ini > presupuesto_seg:
                    print("  *** SALVAGUARDA DE TIEMPO (P2) ***"); return out
    return out


def prueba3(reglas, Ns, condiciones, kcap_base=KCAP_ANCLA, presupuesto_seg=3600):
    """Capacidad HETEROGÉNEA entre nodos, misma media."""
    out = []
    t_ini = time.time()
    for cond in condiciones:
        for N in Ns:
            for p in reglas:
                p2 = dict(p); p2["kcap"] = int(kcap_base)
                t0 = time.time()
                filas, info = correr_condicion(p2, N, cond, kcap_base, seed_cupo=p["seed"] * 11 + N)
                fr = fila_resumen(p2, N, "P3-het", cond, filas, info, time.time() - t0)
                out.append(fr)
                print(f"  [P3] {cond:<13} N={N:<5} {p['rule_id']:<14} pend={fr['pendiente']:+.3f} "
                      f"cv={fr['kcap_cv']:.2f} media_kcap={fr['kcap_media_emp']:.2f} "
                      f"max={fr['kcap_max']:<4} grado={fr['grado_medio_b1']:.2f} clase={fr['clase'][:3]:<3} "
                      f"({fr['dt']:.1f}s) [t={time.time()-t_ini:.0f}s]", flush=True)
                if time.time() - t_ini > presupuesto_seg:
                    print("  *** SALVAGUARDA DE TIEMPO (P3) ***"); return out
    return out


# ============================================================================================
# 6) VERIFICACIONES (que las piezas reusadas hacen lo que digo que hacen — no se asume)
# ============================================================================================
def verificar():
    print("=" * 100)
    print("VERIFICACIÓN 1 — cupo uniforme por vector == C2-hard del motor congelado")
    print("=" * 100)
    rng = np.random.default_rng(1234)
    iguales = 0
    for t in range(12):
        N = 300
        adj0, _ = MOT.GR.aleatorio(N, meandeg=6.0, seed=int(rng.integers(1 << 30)))
        A = [set(a.tolist()) for a in adj0]
        B = [set(a) for a in A]
        k = int(rng.integers(3, 8))
        MOT._enforce_kcap(A, N, k)                                  # motor congelado
        MA._enforce_kcap_variable(B, N, np.full(N, k, dtype=int))   # vector constante
        ok = all(A[i] == B[i] for i in range(N))
        iguales += ok
    print(f"  grafos idénticos tras el corte: {iguales}/12  (esperado 12/12)")

    print("\n" + "=" * 100)
    print("VERIFICACIÓN 2 — las fábricas de cupo respetan la media pedida y difieren en dispersión")
    print("=" * 100)
    N = 2000
    gi = np.random.default_rng(7).poisson(6.0, N).astype(float)
    for nombre, fab in FABRICAS_CUPO.items():
        v = np.asarray(fab(N, 6, gi, np.random.default_rng(99)), dtype=int)
        print(f"  {nombre:<14} media={v.mean():6.3f}  sd={v.std():6.3f}  CV={v.std()/v.mean():5.3f}  "
              f"min={v.min():<3} max={v.max():<5} p99={int(np.percentile(v,99))}")

    print("\n" + "=" * 100)
    print("VERIFICACIÓN 3 — secuencias de cupo de cada normalización (Prueba 2)")
    print("=" * 100)
    for norma in ("ABS", "NORM-log", "NORM-pot", "NORM-ramif"):
        print(f"  {norma:<11} " + "  ".join(f"N={N}:kcap={kcap_normalizado(norma, N)}"
                                            for N in (500, 1000, 2000, 4000)))
    print("  (NORM-grado no se corre: ⟨grado inicial⟩=meandeg no depende de N -> colapsa sobre ABS)")


def cronometro(reglas, Ns):
    print("=" * 100)
    print("CRONÓMETRO — costo por corrida a cada N (antes de comprometer la grilla)")
    print("=" * 100)
    p = dict(reglas[0]); p["kcap"] = KCAP_ANCLA
    for N in Ns:
        t0 = time.time()
        filas, info = correr_condicion(p, N, "UNIF", KCAP_ANCLA, seed_cupo=1)
        r = clasificar_regla(filas)
        print(f"  N={N:<6} {time.time()-t0:6.2f}s   pendiente={r['pendiente_real']:+.3f}  "
              f"aristas_b1={filas[0]['n_aristas']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", required=True,
                    choices=["verificar", "cronometro", "p1", "p2", "p3"])
    ap.add_argument("--n-reglas", type=int, default=6)
    ap.add_argument("--Ns", type=str, default="500,1000,2000,4000")
    ap.add_argument("--kcaps", type=str, default="4,5,6")
    ap.add_argument("--normas", type=str, default="NORM-log,NORM-pot")
    ap.add_argument("--condiciones", type=str,
                    default="UNIF,HET-grado,HET-rango,HET-lognor,HET-bimodal,HET-potencia")
    ap.add_argument("--presupuesto-min", type=float, default=30.0)
    ap.add_argument("--sufijo", type=str, default="")
    args = ap.parse_args()

    Ns = [int(x) for x in args.Ns.split(",") if x]
    t0 = time.time()

    if args.modo == "verificar":
        verificar()
        sys.exit(0)

    reglas = obtener_reglas(args.n_reglas)

    if args.modo == "cronometro":
        cronometro(reglas, Ns)
    elif args.modo == "p1":
        filas = prueba1(reglas, Ns, [int(x) for x in args.kcaps.split(",")],
                        presupuesto_seg=args.presupuesto_min * 60)
        guardar_csv(filas, f"{_HERE}/cs090_fase6_o2c_p1_kcap_absoluto{args.sufijo}.csv")
    elif args.modo == "p2":
        filas = prueba2(reglas, Ns, args.normas.split(","), presupuesto_seg=args.presupuesto_min * 60)
        guardar_csv(filas, f"{_HERE}/cs090_fase6_o2c_p2_normalizado{args.sufijo}.csv")
    elif args.modo == "p3":
        filas = prueba3(reglas, Ns, args.condiciones.split(","),
                        presupuesto_seg=args.presupuesto_min * 60)
        guardar_csv(filas, f"{_HERE}/cs090_fase6_o2c_p3_heterogeneo{args.sufijo}.csv")

    print(f"\nTerminado en {(time.time()-t0)/60:.1f} min. No se declara cierre — números, la lectura es de Alexis.")
