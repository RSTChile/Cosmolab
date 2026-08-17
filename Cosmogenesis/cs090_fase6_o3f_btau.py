#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs090_fase6_o3f_btau.py — PASO 2 de O3-F: el observable B_τ y su test pareado
==============================================================================

Qué es B_τ, en una frase
------------------------
"Branching efectivo de futuros": cuánta variedad de movimiento le queda al gas que
TODAVÍA NO colapsó, por unidad de gas que le queda. La analogía: si la corrida fuera
una orquesta, los sumideros son los músicos que ya se sentaron y dejaron de tocar;
B_τ mide cuántas melodías distintas siguen sonando entre los que quedan de pie,
dividido por cuántos quedan de pie. La predicción de GPT-5.6 Sol (Fase VIII) es que
la Clase III tiene B_τ MAYOR que la Clase I *aunque* le quede MENOS gas: menos grados
de libertad, más futuros posibles (restricción exaptativa).

Las tres piezas del observable
------------------------------
1. **Gas difuso** = partículas de gas del volcado final con densidad por debajo de un
   umbral. Los sumideros ya son partículas aparte en Phantom, así que este corte saca
   además el gas que está cayendo dentro de los grumos.
   Uso SEIS umbrales de tres familias distintas, a propósito: la lección de tareas
   anteriores es que un umbral fijo puede fabricar el resultado.
     - Familia A (absoluta, común a todas las corridas): rho < P50 / P75 / P90 de la
       distribución de densidad AGRUPADA de las 76 corridas. Mismo corte físico para
       todos.
     - Familia B (anclada a la condición inicial de cada corrida): rho < k · mediana
       de rho en t=0 de ESA corrida, k = 1 y 3. "Difuso" = no más denso de lo que
       arrancó.
     - Familia C (control de conteo fijo): las 1000 partículas de menor densidad de
       cada corrida, exactamente. Aquí el número de partículas es idéntico en los dos
       brazos por construcción, así que cualquier diferencia NO puede venir de que a
       III le quede menos gas.

2. **Entropía de la distribución de velocidades** H(v). Shannon, en bits, con bordes
   de bin FIJOS y globales (calculados una sola vez sobre las velocidades agrupadas de
   las 76 corridas), para que dos corridas sean comparables bin a bin. Dos versiones:
     - `abs`  : sobre la velocidad cruda. Captura dispersión + forma.
     - `std`  : sobre la velocidad estandarizada por corrida ((v-media)/sigma en cada
       eje). Captura sólo la FORMA. Es el control contra el artefacto obvio: si III
       colapsa más, su gas se mueve más rápido y llena más bins sin que eso sea
       "más futuros".
   Y dos espacios: módulo de la velocidad en log (16 bins) y el vector (vx,vy,vz) en
   una grilla 4×4×4 = 64 celdas.
   Correcciones de sesgo: la entropía estimada con pocas partículas está sesgada hacia
   abajo, y los brazos tienen distinto número de partículas. Aplico (a) corrección de
   Miller-Madow, y (b) RAREFACCIÓN: recalculo la entropía submuestreando todas las
   corridas al mismo N (el mínimo de la tanda para ese umbral), promediando 200
   remuestreos. La rarefacción es la que deja el número realmente comparable.

3. **Número de filamentos** por friends-of-friends sobre las posiciones del gas
   difuso, con longitud de enlace GRANDE (que es lo que pide un FoF "de filamentos" y
   no "de halos"). La longitud de enlace se fija en múltiplos c ∈ {1.0, 1.5, 2.5} de
   la mediana agrupada de la distancia al vecino más cercano del gas difuso de ese
   umbral — o sea, un número absoluto, igual para todas las corridas. Cuento como
   filamento todo grupo con ≥5 miembros, y reporto además la fracción de partículas
   en el grupo más grande (indicador de percolación: si es ~1, el FoF colapsó todo en
   una sola cosa y el conteo pierde sentido).

B_τ propiamente dicho
---------------------
Tal como lo propuso el analista: B_τ = H(v_gas) / |Ω_gas|, con |Ω_gas| = masa del gas
difuso (masa por partícula × nº de partículas difusas; la masa por partícula es 9.4 y
es idéntica en las 76 corridas, verificado). Reporto además, por separado y sin
normalizar, H y |Ω_gas| — porque la duda central de esta tarea es si un eventual
B_τ(III) > B_τ(I) es sólo aritmética de dividir por un denominador más chico.

Salidas
-------
  cs090_fase6_o3f_btau_crudo.csv     — una fila por (corrida × umbral): todas las métricas
  cs090_fase6_o3f_btau_pares.csv     — una fila por (par × umbral): I, III y su diferencia
  cs090_fase6_o3f_btau_tests.csv     — test de signos + Wilcoxon por (umbral × métrica)
  cs090_fase6_o3f_correlaciones.csv  — correlación de B_τ con la fracción de masa acretada
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix

RAIZ = Path(__file__).resolve().parent
DIR_CACHE = RAIZ / "cs090_fase6_o3f_cache"
CSV_PARES_TOTAL = RAIZ / "cs090_fase5b_TOTAL_40pares.csv"
CSV_PARES_CORREGIDO = RAIZ / "cs090_fase6_reanalisis_40pares_corregido.csv"

RNG = np.random.default_rng(20260811)
N_REMUESTREOS = 200
MIN_MIEMBROS_FILAMENTO = 5
COEFS_ENLACE = (1.0, 1.5, 2.5)
BINS_MODULO = 16       # bins para log10|v|
BINS_EJE = 4           # bins por eje para la grilla 3D (4^3 = 64 celdas)


# ----------------------------------------------------------------------------
# Utilidades de entropía
# ----------------------------------------------------------------------------
def entropia_bits(conteos: np.ndarray) -> float:
    """Shannon en bits, estimador plug-in."""
    n = conteos.sum()
    if n == 0:
        return float("nan")
    p = conteos[conteos > 0] / n
    return float(-np.sum(p * np.log2(p)))


def entropia_miller_madow(conteos: np.ndarray) -> float:
    """Plug-in + corrección de Miller-Madow (K_observado - 1)/(2 N ln2).
    Corrige el sesgo hacia abajo de estimar entropía con pocas muestras."""
    n = conteos.sum()
    if n == 0:
        return float("nan")
    k_obs = int((conteos > 0).sum())
    return entropia_bits(conteos) + (k_obs - 1) / (2 * n * np.log(2))


def contar_en_bins_1d(valores: np.ndarray, bordes: np.ndarray) -> np.ndarray:
    idx = np.clip(np.digitize(valores, bordes), 0, len(bordes))
    return np.bincount(idx, minlength=len(bordes) + 1)


def contar_en_grilla_3d(v3: np.ndarray, bordes_por_eje: list[np.ndarray]) -> np.ndarray:
    """v3 es (N,3). Devuelve el vector de ocupación de la grilla aplanada."""
    n_eje = len(bordes_por_eje[0]) + 1
    idx = np.zeros(len(v3), dtype=np.int64)
    for eje in range(3):
        i = np.clip(np.digitize(v3[:, eje], bordes_por_eje[eje]), 0, n_eje - 1)
        idx = idx * n_eje + i
    return np.bincount(idx, minlength=n_eje ** 3)


def entropia_rarefaccion(valores_idx: np.ndarray, n_celdas: int, n_sub: int) -> float:
    """Entropía promedio submuestreando SIN reemplazo a n_sub elementos.
    Deja la estimación comparable entre corridas con distinto nº de partículas."""
    n = len(valores_idx)
    if n < n_sub:
        return float("nan")
    if n == n_sub:
        return entropia_bits(np.bincount(valores_idx, minlength=n_celdas))
    acum = 0.0
    for _ in range(N_REMUESTREOS):
        sub = RNG.choice(valores_idx, size=n_sub, replace=False)
        acum += entropia_bits(np.bincount(sub, minlength=n_celdas))
    return acum / N_REMUESTREOS


# ----------------------------------------------------------------------------
# Friends-of-friends
# ----------------------------------------------------------------------------
def fof(posiciones: np.ndarray, longitud_enlace: float):
    """Friends-of-friends clásico: dos partículas son amigas si están a menos de
    `longitud_enlace`; los grupos son las componentes conexas de esa amistad.
    Devuelve (nº de grupos con ≥MIN_MIEMBROS, fracción en el grupo mayor)."""
    if len(posiciones) < 2:
        return 0, float("nan")
    arbol = cKDTree(posiciones)
    pares = arbol.query_pairs(longitud_enlace, output_type="ndarray")
    n = len(posiciones)
    if len(pares) == 0:
        return 0, 1.0 / n
    filas = np.concatenate([pares[:, 0], pares[:, 1]])
    cols = np.concatenate([pares[:, 1], pares[:, 0]])
    adj = coo_matrix((np.ones(len(filas)), (filas, cols)), shape=(n, n))
    _, etiquetas = connected_components(adj, directed=False)
    tam = np.bincount(etiquetas)
    return int((tam >= MIN_MIEMBROS_FILAMENTO).sum()), float(tam.max() / n)


# ----------------------------------------------------------------------------
# Carga y definición de umbrales
# ----------------------------------------------------------------------------
def cargar_corridas() -> dict[str, dict]:
    corridas = {}
    for f in sorted(DIR_CACHE.glob("*.npz")):
        d = np.load(f)
        corridas[f.stem] = {k: d[k] for k in
                            ("x", "y", "z", "vx", "vy", "vz", "rho", "h", "rho0")}
    return corridas


def definir_umbrales(corridas: dict) -> dict:
    """Devuelve, para cada nombre de umbral, una función corrida -> máscara booleana."""
    rho_pool = np.concatenate([c["rho"] for c in corridas.values()])
    p50, p75, p90 = np.percentile(rho_pool, [50, 75, 90])

    def por_absoluto(valor):
        return lambda c: c["rho"] < valor

    def por_ic(k):
        return lambda c: c["rho"] < k * np.median(c["rho0"])

    def por_conteo(n_fijo):
        def f(c):
            m = np.zeros(len(c["rho"]), dtype=bool)
            m[np.argsort(c["rho"])[:n_fijo]] = True
            return m
        return f

    return {
        "A_P50_abs": (por_absoluto(p50), f"rho < {p50:.4g} (P50 agrupado)"),
        "A_P75_abs": (por_absoluto(p75), f"rho < {p75:.4g} (P75 agrupado)"),
        "A_P90_abs": (por_absoluto(p90), f"rho < {p90:.4g} (P90 agrupado)"),
        "B_IC_k1": (por_ic(1.0), "rho < 1x mediana de rho(t=0) de la propia corrida"),
        "B_IC_k3": (por_ic(3.0), "rho < 3x mediana de rho(t=0) de la propia corrida"),
        "C_N1000_fijo": (por_conteo(1000), "las 1000 partículas menos densas (N fijo)"),
    }


def bordes_globales(corridas: dict, mascaras: dict):
    """Bordes de bin fijos, calculados sobre el conjunto agrupado de TODAS las
    corridas con ESE umbral. Igual para las 76 corridas => comparables."""
    vel_abs, vel_std, mod_abs, mod_std, nn = [], [], [], [], []
    for nombre, c in corridas.items():
        m = mascaras[nombre]
        v = np.c_[c["vx"][m], c["vy"][m], c["vz"][m]]
        vel_abs.append(v)
        vs = (v - v.mean(0)) / v.std(0)
        vel_std.append(vs)
        mod_abs.append(np.log10(np.linalg.norm(v, axis=1)))
        mod_std.append(np.log10(np.linalg.norm(vs, axis=1) + 1e-12))
        pos = np.c_[c["x"][m], c["y"][m], c["z"][m]]
        if len(pos) > 2:
            dd, _ = cKDTree(pos).query(pos, k=2)
            nn.append(dd[:, 1])
    vel_abs = np.vstack(vel_abs)
    vel_std = np.vstack(vel_std)
    mod_abs = np.concatenate(mod_abs)
    mod_std = np.concatenate(mod_std)
    nn = np.concatenate(nn)

    cuantiles_eje = np.linspace(0, 100, BINS_EJE + 1)[1:-1]
    cuantiles_mod = np.linspace(0, 100, BINS_MODULO + 1)[1:-1]
    return dict(
        eje_abs=[np.percentile(vel_abs[:, i], cuantiles_eje) for i in range(3)],
        eje_std=[np.percentile(vel_std[:, i], cuantiles_eje) for i in range(3)],
        mod_abs=np.percentile(mod_abs, cuantiles_mod),
        mod_std=np.percentile(mod_std, cuantiles_mod),
        d_nn_mediana=float(np.median(nn)),
    )


# ----------------------------------------------------------------------------
# Cálculo de métricas por corrida y umbral
# ----------------------------------------------------------------------------
MASA_POR_PARTICULA = 9.4  # verificado idéntico en las 76 corridas (massoftype)


def metricas_de_corrida(c: dict, m: np.ndarray, bordes: dict, n_sub: int) -> dict:
    v = np.c_[c["vx"][m], c["vy"][m], c["vz"][m]]
    pos = np.c_[c["x"][m], c["y"][m], c["z"][m]]
    n_dif = int(m.sum())
    vs = (v - v.mean(0)) / v.std(0)

    n_celdas_3d = (BINS_EJE) ** 3
    n_celdas_1d = BINS_MODULO

    def idx_3d(vv, bordes_eje):
        n_eje = BINS_EJE
        idx = np.zeros(len(vv), dtype=np.int64)
        for eje in range(3):
            i = np.clip(np.digitize(vv[:, eje], bordes_eje[eje]), 0, n_eje - 1)
            idx = idx * n_eje + i
        return idx

    def idx_1d(mm, bordes_mod):
        return np.clip(np.digitize(mm, bordes_mod), 0, BINS_MODULO - 1)

    i3a = idx_3d(v, bordes["eje_abs"])
    i3s = idx_3d(vs, bordes["eje_std"])
    i1a = idx_1d(np.log10(np.linalg.norm(v, axis=1)), bordes["mod_abs"])
    i1s = idx_1d(np.log10(np.linalg.norm(vs, axis=1) + 1e-12), bordes["mod_std"])

    r = dict(n_difuso=n_dif, masa_difusa=n_dif * MASA_POR_PARTICULA)
    for etiq, idx, ncel in (("v3d_abs", i3a, n_celdas_3d), ("v3d_std", i3s, n_celdas_3d),
                            ("mod_abs", i1a, n_celdas_1d), ("mod_std", i1s, n_celdas_1d)):
        cont = np.bincount(idx, minlength=ncel)
        r[f"H_{etiq}"] = entropia_bits(cont)
        r[f"Hmm_{etiq}"] = entropia_miller_madow(cont)
        r[f"Hrar_{etiq}"] = entropia_rarefaccion(idx, ncel, n_sub)

    # dispersión de velocidades: escala cruda, para separar "más rápido" de "más variado"
    r["sigma_v"] = float(np.linalg.norm(v.std(0)))
    r["v_mediana"] = float(np.median(np.linalg.norm(v, axis=1)))

    # filamentos
    for coef in COEFS_ENLACE:
        n_fil, frac_may = fof(pos, coef * bordes["d_nn_mediana"])
        r[f"n_fil_c{coef}"] = n_fil
        r[f"frac_mayor_c{coef}"] = frac_may
    return r


# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def main() -> None:
    corridas = cargar_corridas()
    print(f"{len(corridas)} corridas en caché")

    # --- pares válidos (tras la corrección de diámetro) ---
    filas_total = list(csv.DictReader(open(CSV_PARES_TOTAL, newline="")))
    corregido = {r["par"]: r for r in csv.DictReader(open(CSV_PARES_CORREGIDO, newline=""))}
    pares = {}
    for f in filas_total:
        pares.setdefault(f["par"], {})[f["rol"]] = f
    pares_validos = []
    for par, brazos in pares.items():
        info = corregido.get(par)
        estado = info["estado_contraste"] if info else "SIN_INFO"
        pares_validos.append(dict(
            par=par, estado=estado,
            corrida_I=f"{brazos['I']['rule_id']}_I",
            corrida_III=f"{brazos['III']['rule_id']}_III",
            frac_masa_I=float(brazos["I"]["fraccion_masa_en_sumideros"]),
            frac_masa_III=float(brazos["III"]["fraccion_masa_en_sumideros"]),
        ))
    n_val = sum(1 for p in pares_validos if p["estado"] == "valido")
    print(f"{len(pares_validos)} pares totales; {n_val} con estado 'valido' tras "
          f"la corrección de diámetro")

    umbrales = definir_umbrales(corridas)

    filas_crudo, filas_pares = [], []
    for nombre_umbral, (fn_mascara, descripcion) in umbrales.items():
        mascaras = {n: fn_mascara(c) for n, c in corridas.items()}
        n_min = min(int(m.sum()) for m in mascaras.values())
        bordes = bordes_globales(corridas, mascaras)
        print(f"\n== umbral {nombre_umbral}: {descripcion}")
        print(f"   N_difuso min={n_min} (N de rarefacción), "
              f"d_nn mediana agrupada={bordes['d_nn_mediana']:.3f}")

        met = {}
        for n, c in corridas.items():
            met[n] = metricas_de_corrida(c, mascaras[n], bordes, n_min)
            filas_crudo.append(dict(corrida=n, umbral=nombre_umbral,
                                    descripcion_umbral=descripcion, **met[n]))

        for p in pares_validos:
            mi, miii = met[p["corrida_I"]], met[p["corrida_III"]]
            fila = dict(par=p["par"], estado=p["estado"], umbral=nombre_umbral,
                        frac_masa_I=p["frac_masa_I"], frac_masa_III=p["frac_masa_III"])
            for k in mi:
                fila[f"{k}_I"] = mi[k]
                fila[f"{k}_III"] = miii[k]
            # B_τ = H / masa de gas difuso, en las cuatro variantes de entropía
            for etiq in ("v3d_abs", "v3d_std", "mod_abs", "mod_std"):
                for pref in ("H", "Hmm", "Hrar"):
                    for rol, mm in (("I", mi), ("III", miii)):
                        fila[f"Btau_{pref}_{etiq}_{rol}"] = mm[f"{pref}_{etiq}"] / mm["masa_difusa"]
            filas_pares.append(fila)

    # --- escritura del crudo ---
    with open(RAIZ / "cs090_fase6_o3f_btau_crudo.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas_crudo[0].keys()))
        w.writeheader(); w.writerows(filas_crudo)
    with open(RAIZ / "cs090_fase6_o3f_btau_pares.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas_pares[0].keys()))
        w.writeheader(); w.writerows(filas_pares)

    # --- tests pareados ---
    metricas_a_testear = []
    for etiq in ("v3d_abs", "v3d_std", "mod_abs", "mod_std"):
        for pref in ("H", "Hmm", "Hrar"):
            metricas_a_testear.append(f"Btau_{pref}_{etiq}")
            metricas_a_testear.append(f"{pref}_{etiq}")
    metricas_a_testear += ["masa_difusa", "n_difuso", "sigma_v", "v_mediana"]
    metricas_a_testear += [f"n_fil_c{c}" for c in COEFS_ENLACE]
    metricas_a_testear += [f"frac_mayor_c{c}" for c in COEFS_ENLACE]

    filas_tests = []
    for nombre_umbral in umbrales:
        sub = [f for f in filas_pares
               if f["umbral"] == nombre_umbral and f["estado"] == "valido"]
        for met_nom in metricas_a_testear:
            a = np.array([f[f"{met_nom}_I"] for f in sub], dtype=float)
            b = np.array([f[f"{met_nom}_III"] for f in sub], dtype=float)
            d = b - a  # III menos I: positivo = la predicción
            fin = np.isfinite(d)
            d = d[fin]
            n_pos = int((d > 0).sum()); n_neg = int((d < 0).sum())
            p_signos = stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue if (n_pos + n_neg) else float("nan")
            try:
                p_wilcoxon = stats.wilcoxon(d, alternative="two-sided").pvalue
            except ValueError:
                p_wilcoxon = float("nan")
            filas_tests.append(dict(
                umbral=nombre_umbral, metrica=met_nom, n_pares=len(d),
                media_I=float(np.nanmean(a[fin])), media_III=float(np.nanmean(b[fin])),
                mediana_delta=float(np.median(d)), media_delta=float(np.mean(d)),
                n_III_mayor=n_pos, n_I_mayor=n_neg,
                p_signos=p_signos, p_wilcoxon=p_wilcoxon,
            ))
    with open(RAIZ / "cs090_fase6_o3f_btau_tests.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas_tests[0].keys()))
        w.writeheader(); w.writerows(filas_tests)

    # --- correlación de B_τ con la fracción de masa acretada (el confound obvio) ---
    filas_corr = []
    for nombre_umbral in umbrales:
        sub = [f for f in filas_pares if f["umbral"] == nombre_umbral and f["estado"] == "valido"]
        # una observación por brazo (2 x n_pares), sin distinguir clase
        for met_nom in ["Btau_Hrar_v3d_std", "Btau_Hrar_v3d_abs", "Btau_H_v3d_abs",
                        "Hrar_v3d_std", "Hrar_v3d_abs", "H_v3d_abs", "masa_difusa",
                        "sigma_v", "n_fil_c1.5"]:
            x, y = [], []
            for f in sub:
                for rol in ("I", "III"):
                    x.append(f[f"frac_masa_{rol}"]); y.append(f[f"{met_nom}_{rol}"])
            x = np.array(x, float); y = np.array(y, float)
            ok = np.isfinite(x) & np.isfinite(y)
            rp, pp = stats.pearsonr(x[ok], y[ok])
            rs, ps = stats.spearmanr(x[ok], y[ok])
            filas_corr.append(dict(umbral=nombre_umbral, metrica=met_nom, n=int(ok.sum()),
                                   pearson_r=rp, pearson_p=pp,
                                   spearman_rho=rs, spearman_p=ps))
    with open(RAIZ / "cs090_fase6_o3f_correlaciones.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas_corr[0].keys()))
        w.writeheader(); w.writerows(filas_corr)

    # --- impresión de lo esencial ---
    print("\n" + "=" * 100)
    print("TEST PAREADO III vs I  (delta = III - I; positivo = predicción de GPT-5.6 Sol)")
    print("=" * 100)
    clave = ["Btau_Hrar_v3d_std", "Btau_Hrar_v3d_abs", "Btau_Hrar_mod_std",
             "Hrar_v3d_std", "Hrar_v3d_abs", "masa_difusa", "sigma_v",
             "n_fil_c1.5", "n_fil_c2.5"]
    print(f"{'umbral':<14}{'metrica':<20}{'n':>3} {'media_I':>11} {'media_III':>11} "
          f"{'delta_med':>11} {'III>I':>7} {'p_signos':>10} {'p_wilcox':>10}")
    for f in filas_tests:
        if f["metrica"] in clave:
            print(f"{f['umbral']:<14}{f['metrica']:<20}{f['n_pares']:>3} "
                  f"{f['media_I']:>11.5g} {f['media_III']:>11.5g} "
                  f"{f['mediana_delta']:>11.4g} {f['n_III_mayor']:>3}/{f['n_pares']:<3} "
                  f"{f['p_signos']:>10.4g} {f['p_wilcoxon']:>10.4g}")
    print("\nArchivos escritos: btau_crudo / btau_pares / btau_tests / correlaciones")


if __name__ == "__main__":
    main()
