"""
CS090 — FASE VI / tarea O2-B: GENEALOGÍAS INDEPENDIENTES ESCALADAS (de 4 a 20) EN A2-B0-C2
==============================================================================================

QUIÉN SOY
---------
Archivo NUEVO. No modifica ni una línea de ningún script existente. En particular NO toca:
`cs090_fase5_generador.py`, `cs090_fase5_motor.py`, `cs090_fase5_clasificador.py`,
`cs090_fase5_mecanismo_aislado.py`, `cs090_diam_corregido.py`, ni
`cs090_fase5_genealogias_independientes.py` (el script de la tarea anterior, que se reusa como patrón
de diseño; acá se re-implementa el mismo bucle porque hay que sharding + medición corregida, pero sus
piezas congeladas se importan iguales).

LA PREGUNTA (en simple, con analogía)
-------------------------------------
Una "genealogía" es una **red social entera y distinta**: un `seed_base` propio del generador, del que
salen 20 familias de parámetros (K, J, ruido, grado medio, tope de amigos, semilla) creadas desde cero.
Una "realización" es un **día distinto de la misma red social**: una de las 20 semillas dentro de ese
mismo `seed_base`.

La tarea anterior (`FASE5_genealogias_independientes_CS.md`) probó 4 redes sociales y encontró que el
patrón bimodal se sostiene en las 4 (45-75% Clase III en C2-hard), pero con **sólo 4 puntos** (3 grados
de libertad) no se podía distinguir:

  (a) "el mecanismo produce ~50-60% Clase III en CUALQUIER red social", de
  (b) "cada red social tiene su propio número verdadero y el promedio lo sostienen 2-3 familias fértiles".

Esta tarea sube a **20 genealogías** para ganar grados de libertad ENTRE grupos, que es exactamente lo
que pidieron los tres analistas del equipo (GPT-5.6 Sol F6-05 pedía 10-12; el segundo analista, 20).

Lo que compra responderla: si el efecto replica entre genealogías, se elimina una amenaza seria de
**pseudorreplicación** en los 40 pares de Fase V-B (tratar 40 reglas de pocas familias como 40 unidades
independientes) y se puede reportar un **N efectivo** defendible.

LA MEDICIÓN DE DIÁMETRO ES LA CORREGIDA
---------------------------------------
Regla vigente desde el 11-ago-2026 (`FASE6_adopcion_diam_corregido_CS.md`): todo cálculo NUEVO de
diámetro usa `cs090_diam_corregido.diam_gigante` (doble-BFS arrancando en la componente conexa más
grande), no el `_diam` de cs055 (que arranca en el nodo no aislado de índice más bajo del grafo entero
y, si ése cayó en un fragmento suelto, mide el fragmento — "el metro apoyado en el buzón de la vereda").

Cómo se aplica acá, sin tocar archivos: los motores llaman siempre a `cs090_fase5_motor._diam(...)`,
que es una búsqueda de atributo de módulo resuelta en el momento de la llamada. Se **sustituye ese
atributo en memoria** (`MOT._diam = DC.diam_gigante`) al arrancar el proceso, exactamente el mismo
mecanismo que ya usó y verificó `cs090_fase6_remedir_mecanismo.py`. Ningún archivo cambia en disco.
Esto importa para el resultado: con la vara vieja, varias reglas caían en "intermedio (sin clase clara)"
o en Clase I por pendiente NEGATIVA (geométricamente imposible), y al corregir se reclasifican — en el
lote de 430 la categoría "intermedio" pasó de 11 a 0.

LAS 20 GENEALOGÍAS Y POR QUÉ SE CONSIDERAN INDEPENDIENTES (§1 del informe)
---------------------------------------------------------------------------
1. **Ninguna repite** las 4 ya usadas (90210, 471829, 823001, 156644) ni los `seed_base` usados en otras
   tareas del proyecto (271828, 371828, 471828, 571828). Verificado por `grep` sobre `cs090*.py`.
2. **Separación garantizada de las cadenas de semillas individuales.** `generar_reglas_clase` deriva
   cada regla con `seed = seed_base + intento*97 + 1`, y con `max_intentos=80` la cadena de una
   genealogía ocupa como mucho el intervalo `[seed_base+1, seed_base+7761]`. Las 20 semillas base de
   acá están separadas entre sí (y de las 8 ya usadas) por **más de 20.000 unidades**, así que
   **ninguna genealogía comparte ni una sola semilla individual con otra** — no es una precaución
   estadística vaga, es una garantía aritmética verificable (`_verificar_separacion()` la comprueba en
   tiempo de ejecución y aborta si falla).
3. **Sin relación aritmética trivial entre ellas**: no son una progresión aritmética ni múltiplos unos
   de otros; los saltos entre semillas consecutivas de la lista son todos distintos.
4. Vale la aclaración honesta: PCG64 (`np.random.default_rng`) no tiene ninguna estructura conocida por
   la que semillas numéricamente cercanas produzcan secuencias correlacionadas, así que la separación
   numérica es una precaución *adicional*. Lo que garantiza independencia de verdad es el punto 2.

QUÉ CORRE
---------
Por genealogía: 20 reglas A2-B0-C2 admitidas por el filtro P1-P5 real (no reglas fabricadas a mano),
brazo **C2-hard** (`MOT.correr_regla_coarse`, el mejor caracterizado) y, si el presupuesto alcanza,
**C2-hibrido** (`MA.correr_regla_coarse_hibrido(modo="soporte")`). Clasificación con
`cs090_fase5_clasificador.clasificar_regla`, **sin cambiar ni un umbral**.

SHARDING (para que entre en el presupuesto)
-------------------------------------------
El motor es determinista y cada regla depende sólo de su `p["seed"]`, así que el barrido se puede
partir en procesos independientes sin cambiar un solo número: `--shard k --nshards M` corre las
genealogías cuyo índice cumple `i % M == k` y escribe su propio CSV. `--modo analisis` junta todos los
shards y hace las cuentas. Correr todo en un solo proceso (`--nshards 1`) da exactamente el mismo
resultado, sólo que más lento.

EL ANÁLISIS DE VARIANZA (§4 del encargo)
-----------------------------------------
Se separa la variación DENTRO de cada genealogía (entre sus 20 semillas) de la variación ENTRE
genealogías, de tres maneras complementarias:

  (a) **Descriptivo**: %Clase III por genealogía, media, desvío estándar y coeficiente de variación
      entre genealogías; comparación contra el error estándar binomial esperado dentro de una sola
      genealogía con n=20 (√(p(1-p)/n)) — el mismo cálculo que hizo la tarea anterior, para que sean
      comparables número a número.
  (b) **ANOVA de una vía sobre el indicador 0/1 de Clase III** (y también sobre la pendiente continua):
      MSB (entre grupos) vs MSW (dentro de grupos), F, y la **correlación intraclase**
      ICC = (MSB − MSW) / (MSB + (m−1)·MSW), truncada en 0 porque una ICC negativa significa
      "menos dispersión entre grupos que la esperada por azar", no una correlación real.
  (c) **N EFECTIVO**: con ICC estimada, el efecto de diseño de un muestreo por conglomerados es
      deff = 1 + (m−1)·ICC, y N_eff = N_total / deff. En simple: si las reglas de una misma familia se
      parecen entre sí, 400 reglas "no valen" 400 datos independientes; N_eff dice cuántos valen.
      Esto es lo que pidió GPT-5.6 Sol para poder defender los tests de Fase V-B.
  (d) **Test de permutación** (sin supuestos de normalidad, que con datos 0/1 y n=20 son dudosos): se
      barajan las etiquetas de genealogía entre todas las reglas y se mira cuántas veces el desvío
      estándar de los %Clase III por genealogía sale tan grande como el observado. Es la versión
      honesta de "¿la dispersión entre familias excede el ruido de muestreo?".

NO se corre Phantom. NO se declara cierre ni veredicto — se reportan números. NO se hacen commits.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from collections import Counter

import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cs090_fase5_generador as GEN                 # generar_reglas_clase() — congelado, NO SE TOCA
import cs090_fase5_motor as MOT                     # correr_regla_coarse()  — congelado, NO SE TOCA
import cs090_fase5_mecanismo_aislado as MA          # correr_regla_coarse_hibrido() — congelado
import cs090_diam_corregido as DC                   # la medición oficial de diámetro (11-ago-2026)
from cs090_fase5_clasificador import clasificar_regla   # umbrales pre-registrados, sin cambiar

# ---------------------------------------------------------------------------------------------
# SUSTITUCIÓN EN MEMORIA DE LA MEDICIÓN DE DIÁMETRO (ver docstring). Ningún archivo cambia en disco.
# Se hace UNA vez, al importar este módulo, para que TODO lo que corra debajo (C2-hard, C2-hibrido,
# la medición nativa de MOT.medir y los NULL_topo) use la misma vara. Corregir un solo lado del
# z-score REAL-vs-NULL inventaría una asimetría.
# ---------------------------------------------------------------------------------------------
_DIAM_HISTORICO = MOT._diam
MOT._diam = DC.diam_gigante

EJE_A, EJE_B, EJE_C = "A2", "B0", "C2"
N_GRANDE = 2000
ESCALAS_B = (1, 2, 4, 8, 16)
N_SWEEPS = 14
N_SEEDS_NULL_TOPO = 3
N_REGLAS_POR_GENEALOGIA = 20

# ---------------------------------------------------------------------------------------------
# LAS 20 GENEALOGÍAS. Etiqueta -> seed_base. Ver docstring §"LAS 20 GENEALOGÍAS" para la
# justificación de independencia (separación > 20.000, sin progresión aritmética, ninguna repetida).
# ---------------------------------------------------------------------------------------------
SEEDS_NUEVAS = [
    113477, 218903, 344251, 662819, 741037,
    905683, 1128409, 1357061, 1604923, 1889347,
    2043761, 2296589, 2571043, 2814697, 3102859,
    3389417, 3670213, 3948071, 4213589, 4507921,
]
GENEALOGIAS = [(f"H{i:02d}_{s}", s) for i, s in enumerate(SEEDS_NUEVAS)]

# Semillas base YA usadas en el proyecto — ninguna de las de arriba puede coincidir ni acercarse.
SEEDS_YA_USADAS = [90210, 156644, 271828, 371828, 471828, 471829, 571828, 823001]

# Ancho máximo de la cadena de semillas individuales de UNA genealogía:
#   seed = seed_base + intento*97 + 1, con max_intentos = max(80, n_reglas*4) = 80
ANCHO_CADENA = 80 * 97 + 1        # = 7761
SEPARACION_MINIMA = 20000          # holgura > 2.5x el ancho de cadena

BRAZOS_DISPONIBLES = ("C2-hard", "C2-hibrido")


def _verificar_separacion():
    """Comprueba, ANTES de gastar cómputo, que ninguna genealogía puede compartir una semilla
    individual con otra ni con las ya usadas en el proyecto. Aborta si falla: es más barato fallar acá
    que descubrir después que dos 'genealogías independientes' compartían reglas."""
    # OJO: entre las YA usadas hay pares casi pegados (471828 y 471829 difieren en 1) — eso es historia
    # del proyecto y no es cosa de esta tarea. Lo que se exige acá es (i) separación entre las NUEVAS,
    # y (ii) separación de cada NUEVA respecto de cada YA USADA.
    nuevas = sorted(SEEDS_NUEVAS)
    problemas = []
    d_min_nuevas = min(b - a for a, b in zip(nuevas, nuevas[1:]))
    for a, b in zip(nuevas, nuevas[1:]):
        if b - a < SEPARACION_MINIMA:
            problemas.append(("nueva-nueva", a, b, b - a))
    d_min_vs_usadas = min(abs(n - u) for n in nuevas for u in SEEDS_YA_USADAS)
    for n in nuevas:
        for u in SEEDS_YA_USADAS:
            if abs(n - u) < SEPARACION_MINIMA:
                problemas.append(("nueva-ya_usada", n, u, abs(n - u)))
    repetidas = [s for s, c in Counter(SEEDS_NUEVAS).items() if c > 1]
    if repetidas:
        problemas.append(("repetidas", repetidas, 0, 0))
    saltos = [b - a for a, b in zip(SEEDS_NUEVAS, SEEDS_NUEVAS[1:])]
    print(f"[verificación] {len(SEEDS_NUEVAS)} semillas nuevas; ancho de cadena por genealogía = "
          f"{ANCHO_CADENA}; separación mínima nueva-nueva = {d_min_nuevas}; "
          f"separación mínima nueva-vs-ya_usada = {d_min_vs_usadas}")
    print(f"[verificación] saltos consecutivos entre semillas nuevas, todos distintos = "
          f"{len(set(saltos)) == len(saltos)}  -> no hay progresión aritmética")
    if problemas:
        raise SystemExit(f"*** SEMILLAS MAL ELEGIDAS: {problemas} ***")
    print("[verificación] OK: ninguna genealogía puede compartir una semilla individual con otra.")


# ============================================================================================
# 1) CORRER UNA GENEALOGÍA
# ============================================================================================
def _correr_brazo(brazo, p):
    if brazo == "C2-hard":
        return MOT.correr_regla_coarse(p, N=N_GRANDE, n_sweeps=N_SWEEPS, escalas_b=ESCALAS_B,
                                       n_seeds_null_topo=N_SEEDS_NULL_TOPO)
    if brazo == "C2-hibrido":
        return MA.correr_regla_coarse_hibrido(p, modo="soporte")
    raise ValueError(brazo)


def correr_genealogia(etiqueta, seed_base, n_reglas, brazos):
    """Genera n_reglas admitidas por el filtro P1-P5 real y las corre en cada brazo. Devuelve
    (filas_resumen, n_admitidas, n_descartadas)."""
    t_ini = time.time()
    admitidas, descartadas = GEN.generar_reglas_clase(
        EJE_A, EJE_B, EJE_C, n_reglas=n_reglas, seed_base=seed_base,
        max_intentos=max(80, n_reglas * 4))
    print(f"[{etiqueta}] seed_base={seed_base}  admitidas={len(admitidas)}/{n_reglas}  "
          f"descartadas(P1-P5)={len(descartadas)}", flush=True)
    for d in descartadas:
        print(f"    descartada {d['rule_id']}: {d['motivo_descarte']}", flush=True)

    filas = []
    for p in admitidas:
        for brazo in brazos:
            t0 = time.time()
            try:
                fs = _correr_brazo(brazo, p)
            except Exception as e:                                    # noqa: BLE001
                print(f"    *** FALLO {brazo} en {p['rule_id']}: {type(e).__name__}: {e} ***",
                      flush=True)
                continue
            r = clasificar_regla(fs)
            f_b1 = next(f for f in fs if f["escala_b"] == 1)
            grado_medio_b1 = 2 * f_b1["n_aristas"] / f_b1["N"] if f_b1["N"] else float("nan")
            filas.append(dict(
                genealogia=etiqueta, seed_base=seed_base, rule_id=p["rule_id"], brazo=brazo,
                clase=r["clase"], es_clase_III=int(r["clase"] == "III"),
                es_clase_III_o_IV=int(r["clase"] in ("III", "IV")),
                pendiente=round(r["pendiente_real"], 6), z_agg=round(r["z_agg"], 4),
                holon_ratio=round(r["holon_ratio"], 4),
                diam_b1=f_b1["diam_real"], giant_b1=round(f_b1["giant_real"], 4),
                grado_medio_b1=round(grado_medio_b1, 3), n_aristas_b1=f_b1["n_aristas"],
                diams_por_escala="|".join(str(int(f["diam_real"])) for f in fs),
                K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"], kcap=p["kcap"],
                seed=p["seed"], seg=round(time.time() - t0, 2),
            ))
            print(f"  [{etiqueta}] {p['rule_id']:<16} {brazo:<11} clase={r['clase']:<26} "
                  f"pend={r['pendiente_real']:+.3f} diam_b1={f_b1['diam_real']:.0f} "
                  f"({time.time()-t0:.1f}s) [t={time.time()-t_ini:.0f}s]", flush=True)
    return filas, len(admitidas), len(descartadas)


def guardar_csv(filas, ruta):
    if not filas:
        print(f"(sin filas para {ruta})")
        return
    with open(ruta, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        wr.writeheader()
        wr.writerows(filas)
    print(f"CSV: {ruta}  ({len(filas)} filas)")


# ============================================================================================
# 2) ANÁLISIS DE VARIANZA ENTRE vs. DENTRO DE GENEALOGÍA
# ============================================================================================
def anova_una_via(grupos):
    """ANOVA de una vía clásica sobre una lista de listas (una por genealogía). Devuelve MSB, MSW, F,
    ICC y el n medio por grupo. Vale tanto para el indicador 0/1 de Clase III como para la pendiente
    continua (con datos 0/1 el ANOVA es el mismo cálculo que la descomposición de varianza de una
    proporción; el test F es aproximado y por eso además se hace la permutación de §(d))."""
    grupos = [np.asarray(g, dtype=float) for g in grupos if len(g) > 0]
    k = len(grupos)
    ns = np.array([len(g) for g in grupos], dtype=float)
    n_tot = ns.sum()
    medias = np.array([g.mean() for g in grupos])
    gran = np.concatenate(grupos).mean()
    ssb = float((ns * (medias - gran) ** 2).sum())
    ssw = float(sum(((g - g.mean()) ** 2).sum() for g in grupos))
    gl_b, gl_w = k - 1, n_tot - k
    msb = ssb / gl_b if gl_b > 0 else float("nan")
    msw = ssw / gl_w if gl_w > 0 else float("nan")
    F = msb / msw if msw > 1e-15 else float("inf")
    # n medio por grupo corregido (fórmula estándar cuando los grupos no son todos del mismo tamaño)
    m0 = (n_tot - (ns ** 2).sum() / n_tot) / (k - 1) if k > 1 else float("nan")
    icc_bruta = (msb - msw) / (msb + (m0 - 1) * msw) if (msb + (m0 - 1) * msw) > 1e-15 else 0.0
    return dict(k=k, n_total=int(n_tot), m0=m0, gran_media=float(gran),
                MSB=msb, MSW=msw, F=F, gl_b=int(gl_b), gl_w=int(gl_w),
                icc_bruta=float(icc_bruta), icc=float(max(0.0, icc_bruta)),
                var_entre=float(max(0.0, (msb - msw) / m0)) if m0 and m0 > 0 else float("nan"),
                var_dentro=float(msw))


def p_valor_F(F, gl1, gl2):
    """p de la F sin scipy: se integra la cola por la relación con la Beta incompleta, usando la
    aproximación de la distribución Beta vía la función beta incompleta regularizada calculada con
    integración numérica (suficiente para reportar el orden de magnitud; el test decisivo de esta
    tarea es la permutación, que no supone ninguna distribución)."""
    if not np.isfinite(F) or F <= 0:
        return float("nan")
    x = gl2 / (gl2 + gl1 * F)          # I_x(gl2/2, gl1/2) = p
    a, b = gl2 / 2.0, gl1 / 2.0
    ts = np.linspace(1e-12, max(1e-12, x), 20001)
    integrando = ts ** (a - 1) * (1 - ts) ** (b - 1)
    num = np.trapz(integrando, ts)
    ts2 = np.linspace(1e-12, 1 - 1e-12, 20001)
    den = np.trapz(ts2 ** (a - 1) * (1 - ts2) ** (b - 1), ts2)
    return float(min(1.0, max(0.0, num / den)))


def permutacion_dispersión(valores, etiquetas, n_perm=20000, semilla=20260811):
    """¿La dispersión de los promedios por genealogía excede lo que da barajar las reglas al azar entre
    genealogías? Se conserva el tamaño de cada genealogía y se permutan las reglas. Estadístico: desvío
    estándar de los promedios por grupo. Devuelve (std_obs, std_nulo_medio, p, percentil)."""
    rng = np.random.default_rng(semilla)
    valores = np.asarray(valores, dtype=float)
    etiquetas = np.asarray(etiquetas)
    grupos = [np.flatnonzero(etiquetas == e) for e in dict.fromkeys(etiquetas)]
    tam = [len(g) for g in grupos]

    def std_de_medias(v, tam):
        out, i = [], 0
        for t in tam:
            out.append(v[i:i + t].mean())
            i += t
        return float(np.std(out))

    orden = np.concatenate(grupos)
    v_ord = valores[orden]
    obs = std_de_medias(v_ord, tam)
    nulos = np.empty(n_perm)
    for i in range(n_perm):
        nulos[i] = std_de_medias(rng.permutation(v_ord), tam)
    p = float((np.sum(nulos >= obs) + 1) / (n_perm + 1))
    return dict(std_obs=obs, std_nulo_medio=float(nulos.mean()),
                std_nulo_p95=float(np.percentile(nulos, 95)), p=p,
                percentil_obs=float(100.0 * np.mean(nulos < obs)))


def analizar(filas, brazo, salida_prefijo):
    """Todo el §4 del encargo para UN brazo. Devuelve el dict del resumen e imprime las tablas."""
    fb = [f for f in filas if f["brazo"] == brazo]
    if not fb:
        return None
    etiquetas = sorted({f["genealogia"] for f in fb})

    print("\n" + "=" * 118)
    print(f"BRAZO {brazo} — %Clase III POR GENEALOGÍA (tabla completa)")
    print("=" * 118)
    print(f"{'genealogía':<16}{'seed_base':>10}{'n':>4}{'I':>4}{'II':>4}{'III':>5}{'IV':>4}"
          f"{'otro':>6}{'%III':>8}{'%III+IV':>9}{'SEbinom':>9}{'pend_med':>10}{'pend_mdn':>10}"
          f"{'diam_med':>10}{'grado':>8}")
    por_gen, fracs, pend_grupos, iii_grupos = [], [], [], []
    for e in etiquetas:
        g = [f for f in fb if f["genealogia"] == e]
        cnt = Counter(f["clase"] for f in g)
        n = len(g)
        n3, n4 = cnt.get("III", 0), cnt.get("IV", 0)
        frac = 100.0 * n3 / n
        p_ = n3 / n
        se_binom = 100.0 * np.sqrt(p_ * (1 - p_) / n)
        pends = [f["pendiente"] for f in g]
        fila = dict(
            genealogia=e, seed_base=g[0]["seed_base"], brazo=brazo, n=n,
            n_I=cnt.get("I", 0), n_II=cnt.get("II", 0), n_III=n3, n_IV=n4,
            n_otro=n - cnt.get("I", 0) - cnt.get("II", 0) - n3 - n4,
            pct_III=round(frac, 2), pct_III_IV=round(100.0 * (n3 + n4) / n, 2),
            se_binomial_pp=round(se_binom, 2),
            pendiente_media=round(float(np.mean(pends)), 4),
            pendiente_mediana=round(float(np.median(pends)), 4),
            pendiente_std_intra=round(float(np.std(pends, ddof=1)), 4),
            diam_medio_b1=round(float(np.mean([f["diam_b1"] for f in g])), 2),
            grado_medio_b1=round(float(np.mean([f["grado_medio_b1"] for f in g])), 3),
        )
        por_gen.append(fila)
        fracs.append(frac)
        pend_grupos.append(pends)
        iii_grupos.append([f["es_clase_III"] for f in g])
        print(f"{e:<16}{fila['seed_base']:>10}{n:>4}{fila['n_I']:>4}{fila['n_II']:>4}{n3:>5}"
              f"{n4:>4}{fila['n_otro']:>6}{frac:>7.1f}%{fila['pct_III_IV']:>8.1f}%"
              f"{se_binom:>9.2f}{fila['pendiente_media']:>10.3f}{fila['pendiente_mediana']:>10.3f}"
              f"{fila['diam_medio_b1']:>10.2f}{fila['grado_medio_b1']:>8.2f}")

    fracs = np.array(fracs)
    media, std = float(fracs.mean()), float(fracs.std(ddof=1))
    cv = 100.0 * std / media if media > 0 else float("nan")
    se_binom_prom = float(np.mean([f["se_binomial_pp"] for f in por_gen]))
    p_global = float(np.mean([f["es_clase_III"] for f in fb]))
    se_binom_global = 100.0 * np.sqrt(p_global * (1 - p_global) / N_REGLAS_POR_GENEALOGIA)

    print("-" * 118)
    print(f"ENTRE GENEALOGÍAS (n={len(fracs)}): media %III = {media:.2f}%   "
          f"std = {std:.2f} pp   CV = {cv:.1f}%   min = {fracs.min():.1f}%   max = {fracs.max():.1f}%   "
          f"rango = {fracs.max()-fracs.min():.1f} pp")
    print(f"RUIDO DE MUESTREO esperado DENTRO de una sola genealogía con n={N_REGLAS_POR_GENEALOGIA}: "
          f"SE binomial promedio (cada una con su p) = {se_binom_prom:.2f} pp   |   "
          f"con el p global ({100*p_global:.1f}%) = {se_binom_global:.2f} pp")
    print(f"  -> razón std_observado / SE_binomial = {std/se_binom_prom:.3f}  "
          f"(1.0 = la dispersión entre familias es EXACTAMENTE la esperada por puro muestreo)")

    # ---- (b) ANOVA sobre el indicador 0/1 de Clase III y sobre la pendiente continua ----
    print("\n" + "-" * 118)
    print(f"ANOVA de una vía — variación ENTRE genealogías vs DENTRO de cada genealogía [{brazo}]")
    print("-" * 118)
    resultados_anova = {}
    for nombre, grupos in (("indicador 0/1 de Clase III", iii_grupos), ("pendiente (continua)", pend_grupos)):
        a = anova_una_via(grupos)
        a["p_F"] = p_valor_F(a["F"], a["gl_b"], a["gl_w"])
        deff = 1.0 + (a["m0"] - 1.0) * a["icc"]
        a["deff"] = deff
        a["N_eff"] = a["n_total"] / deff if deff > 0 else float("nan")
        a["N_eff_genealogias_equiv"] = a["N_eff"] / a["m0"] if a["m0"] else float("nan")
        resultados_anova[nombre] = a
        print(f"  {nombre}:")
        print(f"    MSB (entre, gl={a['gl_b']}) = {a['MSB']:.5f}   "
              f"MSW (dentro, gl={a['gl_w']}) = {a['MSW']:.5f}   F = {a['F']:.3f}   p≈{a['p_F']:.4f}")
        print(f"    componente de varianza ENTRE = {a['var_entre']:.5f}   DENTRO = {a['var_dentro']:.5f}   "
              f"ICC = {a['icc']:.4f} (bruta {a['icc_bruta']:+.4f})")
        print(f"    efecto de diseño deff = 1+(m-1)·ICC = {deff:.3f}   ->   "
              f"N_total = {a['n_total']}  ->  **N EFECTIVO = {a['N_eff']:.1f}**  "
              f"(equivalente a {a['N_eff_genealogias_equiv']:.1f} genealogías completas)")

    # ---- (d) permutación ----
    print("\n" + "-" * 118)
    print(f"TEST DE PERMUTACIÓN (20.000 barajadas) — ¿la dispersión entre familias excede el azar? [{brazo}]")
    print("-" * 118)
    etq = [f["genealogia"] for f in fb]
    perms = {}
    for nombre, vals in (("indicador 0/1 de Clase III", [f["es_clase_III"] for f in fb]),
                         ("pendiente (continua)", [f["pendiente"] for f in fb])):
        pr = permutacion_dispersión(vals, etq)
        perms[nombre] = pr
        print(f"  {nombre}: std observado de las medias por genealogía = {pr['std_obs']:.4f}   "
              f"std esperado barajando = {pr['std_nulo_medio']:.4f} (p95={pr['std_nulo_p95']:.4f})   "
              f"p = {pr['p']:.4f}   percentil del observado = {pr['percentil_obs']:.1f}")

    # ---- ¿2-3 familias fértiles, o efecto repartido? ----
    print("\n" + "-" * 118)
    print(f"¿EFECTO REPARTIDO O SOSTENIDO POR POCAS FAMILIAS? [{brazo}]")
    print("-" * 118)
    ordenadas = sorted(por_gen, key=lambda f: -f["pct_III"])
    total_III = sum(f["n_III"] for f in por_gen)
    top3 = sum(f["n_III"] for f in ordenadas[:3])
    n_cero = sum(1 for f in por_gen if f["n_III"] == 0)
    n_mayor_30 = sum(1 for f in por_gen if f["pct_III"] >= 30.0)
    n_mayor_45 = sum(1 for f in por_gen if f["pct_III"] >= 45.0)
    print(f"  genealogías con 0% Clase III (familias 'estériles'): {n_cero}/{len(por_gen)}")
    print(f"  genealogías con >=30% Clase III: {n_mayor_30}/{len(por_gen)}   "
          f"con >=45% (el piso de las 4 anteriores): {n_mayor_45}/{len(por_gen)}")
    print(f"  las 3 genealogías más fértiles aportan {top3}/{total_III} de todas las Clase III = "
          f"{100.0*top3/max(1,total_III):.1f}%  (reparto perfectamente uniforme daría "
          f"{100.0*3/len(por_gen):.1f}%)")
    print(f"  mediana de %III = {float(np.median(fracs)):.1f}%   "
          f"cuartiles = {float(np.percentile(fracs,25)):.1f}% / {float(np.percentile(fracs,75)):.1f}%")

    guardar_csv(por_gen, f"{salida_prefijo}_por_genealogia_{brazo}.csv")
    return dict(brazo=brazo, por_gen=por_gen, media=media, std=std, cv=cv,
                se_binom_prom=se_binom_prom, anova=resultados_anova, perms=perms,
                fracs=fracs.tolist(), n_cero=n_cero, n_mayor_30=n_mayor_30,
                n_mayor_45=n_mayor_45, top3_share=100.0 * top3 / max(1, total_III))


def figura(filas, brazos, ruta):
    """Distribución de %Clase III por genealogía (barras ordenadas) + la nube de pendientes por
    genealogía, para ver de un vistazo si hay 2-3 familias fértiles o el efecto está repartido."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                            # noqa: BLE001
        print(f"(sin figura: {type(e).__name__}: {e})")
        return
    brazos = [b for b in brazos if any(f["brazo"] == b for f in filas)]
    fig, axes = plt.subplots(len(brazos), 2, figsize=(15, 4.6 * len(brazos)), squeeze=False)
    for i, brazo in enumerate(brazos):
        fb = [f for f in filas if f["brazo"] == brazo]
        etiquetas = sorted({f["genealogia"] for f in fb})
        datos = []
        for e in etiquetas:
            g = [f for f in fb if f["genealogia"] == e]
            datos.append((e, 100.0 * sum(f["es_clase_III"] for f in g) / len(g),
                          [f["pendiente"] for f in g]))
        datos.sort(key=lambda t: -t[1])
        ax = axes[i][0]
        ax.bar(range(len(datos)), [d[1] for d in datos], color="#3c6e9f")
        med = float(np.mean([d[1] for d in datos]))
        ax.axhline(med, color="#c0392b", ls="--", lw=1.4, label=f"media {med:.1f}%")
        ax.axhline(45, color="#7f8c8d", ls=":", lw=1.2, label="45% (piso de las 4 anteriores)")
        ax.set_xticks(range(len(datos)))
        ax.set_xticklabels([d[0].split("_")[0] for d in datos], rotation=60, fontsize=8)
        ax.set_ylabel("% Clase III (n=20 reglas)")
        ax.set_title(f"{brazo} — %Clase III por genealogía (diámetro corregido)")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 100)
        ax2 = axes[i][1]
        ax2.boxplot([d[2] for d in datos], showfliers=False)
        for j, d in enumerate(datos):
            ax2.scatter(np.full(len(d[2]), j + 1) + np.random.default_rng(j).normal(0, 0.06, len(d[2])),
                        d[2], s=9, alpha=0.55, color="#3c6e9f")
        ax2.axhline(0.7, color="#c0392b", ls="--", lw=1.2, label="umbral Clase III (0.7)")
        ax2.axhspan(0.35, 0.45, color="#f0ad4e", alpha=0.20, label="banda Clase II (0.35-0.45)")
        ax2.set_xticks(range(1, len(datos) + 1))
        ax2.set_xticklabels([d[0].split("_")[0] for d in datos], rotation=60, fontsize=8)
        ax2.set_ylabel("pendiente log(diám) vs log(N_cajas)")
        ax2.set_title(f"{brazo} — pendientes individuales por genealogía")
        ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    print(f"PNG: {ruta}")


def figura_kcap(filas, brazos, ruta):
    """La segunda figura responde 'por qué unas familias son más fértiles que otras': a la izquierda,
    %Clase III según el tope de amigos `kcap` de cada regla (nivel REGLA, sobre las 400); a la derecha,
    el %Clase III de cada genealogía contra el kcap medio que le tocó en el sorteo (nivel GENEALOGÍA).
    Si la nube de la derecha cae sobre una recta, la 'fertilidad de la familia' no es una propiedad de
    la familia: es cuántas reglas de kcap bajo le tocaron."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                            # noqa: BLE001
        print(f"(sin figura kcap: {type(e).__name__}: {e})")
        return
    brazos = [b for b in brazos if any(f["brazo"] == b for f in filas)]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    colores = {"C2-hard": "#3c6e9f", "C2-hibrido": "#c0752c"}
    for brazo in brazos:
        fb = [f for f in filas if f["brazo"] == brazo]
        kcaps = sorted({int(f["kcap"]) for f in fb})
        ys, ns = [], []
        for k in kcaps:
            s = [f for f in fb if int(f["kcap"]) == k]
            ys.append(100.0 * sum(f["es_clase_III"] for f in s) / len(s))
            ns.append(len(s))
        axes[0].plot(kcaps, ys, "o-", color=colores.get(brazo, "#555"), label=brazo)
        for k, y, n in zip(kcaps, ys, ns):
            axes[0].annotate(f"n={n}", (k, y), textcoords="offset points", xytext=(4, 6), fontsize=7)
        etiquetas = sorted({f["genealogia"] for f in fb})
        px, py = [], []
        for e in etiquetas:
            g = [f for f in fb if f["genealogia"] == e]
            px.append(float(np.mean([int(f["kcap"]) for f in g])))
            py.append(100.0 * sum(f["es_clase_III"] for f in g) / len(g))
        axes[1].scatter(px, py, s=42, color=colores.get(brazo, "#555"), label=brazo, alpha=0.85)
        if len(px) > 2:
            m, b = np.polyfit(px, py, 1)
            xs = np.linspace(min(px), max(px), 10)
            axes[1].plot(xs, m * xs + b, ls="--", lw=1.2, color=colores.get(brazo, "#555"))
    axes[0].set_xlabel("kcap (tope de vecinos de la regla)")
    axes[0].set_ylabel("% Clase III")
    axes[0].set_title("Nivel REGLA: kcap decide casi todo (400 reglas por brazo)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("kcap medio que le tocó a la genealogía (sorteo de 20 reglas)")
    axes[1].set_ylabel("% Clase III de la genealogía")
    axes[1].set_title("Nivel GENEALOGÍA: la 'fertilidad' sigue al kcap sorteado")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    print(f"PNG: {ruta}")


# ============================================================================================
# 3) DRIVER
# ============================================================================================
PREFIJO = f"{_HERE}/cs090_fase6_o2b_genealogias"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["correr", "analisis"], default="correr")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--brazos", default="C2-hard",
                    help="coma-separado; p.ej. 'C2-hard' o 'C2-hard,C2-hibrido'")
    ap.add_argument("--n-genealogias", type=int, default=len(GENEALOGIAS))
    ap.add_argument("--n-reglas", type=int, default=N_REGLAS_POR_GENEALOGIA)
    args = ap.parse_args()

    brazos = [b.strip() for b in args.brazos.split(",") if b.strip()]
    for b in brazos:
        assert b in BRAZOS_DISPONIBLES, b

    if args.modo == "correr":
        _verificar_separacion()
        print(f"[diámetro] MOT._diam sustituido en memoria: "
              f"{_DIAM_HISTORICO.__name__} -> {MOT._diam.__name__} "
              f"(medición oficial corregida, ningún archivo tocado en disco)")
        mias = [(i, e, s) for i, (e, s) in enumerate(GENEALOGIAS[:args.n_genealogias])
                if i % args.nshards == args.shard]
        print(f"[shard {args.shard}/{args.nshards}] genealogías: {[e for _, e, _ in mias]}  "
              f"brazos={brazos}")
        t0 = time.time()
        todas, info = [], []
        for i, etiqueta, seed_base in mias:
            fs, n_adm, n_desc = correr_genealogia(etiqueta, seed_base, args.n_reglas, brazos)
            todas += fs
            info.append((etiqueta, seed_base, n_adm, n_desc))
            guardar_csv(todas, f"{PREFIJO}_reglas_shard{args.shard}.csv")   # se guarda incremental
        print(f"\n[shard {args.shard}] filtro P1-P5:")
        for etiqueta, seed_base, n_adm, n_desc in info:
            print(f"   {etiqueta:<16} seed_base={seed_base:<9} admitidas={n_adm}/{args.n_reglas} "
                  f"descartadas={n_desc}")
        print(f"[shard {args.shard}] terminado en {(time.time()-t0)/60:.1f} min, "
              f"{len(todas)} filas.")
    else:
        filas = []
        for ruta in sorted(glob.glob(f"{PREFIJO}_reglas_shard*.csv")):
            with open(ruta) as fh:
                for r in csv.DictReader(fh):
                    for k in ("es_clase_III", "es_clase_III_o_IV", "seed_base", "n_aristas_b1",
                              "kcap", "K", "seed"):
                        r[k] = int(float(r[k]))
                    for k in ("pendiente", "z_agg", "holon_ratio", "diam_b1", "giant_b1",
                              "grado_medio_b1", "J", "noise", "meandeg", "seg"):
                        r[k] = float(r[k])
                    filas.append(r)
            print(f"leído {os.path.basename(ruta)}")
        if not filas:
            raise SystemExit("no hay CSVs de shard para analizar")
        guardar_csv(filas, f"{PREFIJO}_reglas_TODAS.csv")
        brazos_presentes = [b for b in BRAZOS_DISPONIBLES if any(f["brazo"] == b for f in filas)]
        print(f"\n{len(filas)} filas · genealogías = "
              f"{len(set(f['genealogia'] for f in filas))} · brazos = {brazos_presentes}")
        for b in brazos_presentes:
            analizar(filas, b, PREFIJO)
        figura(filas, brazos_presentes, f"{PREFIJO}_distribucion.png")
        figura_kcap(filas, brazos_presentes, f"{PREFIJO}_kcap.png")
        print("\nFin del análisis. No se declara cierre ni veredicto — los números están arriba; "
              "la lectura final es de Alexis.")
