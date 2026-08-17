"""
cs089_on77_espectral.py — O-N7.7 con observable ESPECTRAL (sin Phantom), en vez de masa en sumideros
=======================================================================================================
Origen del encargo (Alexis, 9-ago-2026): `ON77_sistemaAB_cierre_CS.md` testeó O-N7.7 (acumulación
adaptativa vs. condensación exaptativa) con "masa en sumideros" como observable de Ω_op, y dio resultados
EN CONTRA de la predicción en ambos sistemas. Alexis hizo una CRÍTICA ya documentada a ese resultado: un
sumidero es un HORIZONTE -- la materia que cae ahí ya no tiene más historia posible, y el sumidero no
emite nada de vuelta. "Masa en sumideros" mide el FIN de la historia de esa materia, casi lo opuesto de
"dominio operativo creciente" que pide O-N7.7. Ese resultado anterior quedó en SUSPENSO (observable mal
elegido), no refutado ni confirmado.

LA IDEA DE ESTE SCRIPT: el espectro del grafo (ANTES de correr Phantom, sobre el gas que TODAVÍA NO
colapsó) evita el problema del horizonte por completo -- no depende de qué fracción cae en un sumidero,
sólo de la estructura RELACIONAL del sistema (quién quedó conectado con quién). Es un candidato mucho más
fiel a "espacio de procesamiento Ω_proc" / "dominio operativo Ω_op" que la masa acretada. Como no hace
falta correr Phantom NI el layout de resortes (las propiedades espectrales del laplaciano dependen sólo de
la ADYACENCIA del grafo, no de las posiciones en el espacio -- el layout es puramente cosmético para
sembrar la IC de Phantom), este script es mucho más barato que el intento anterior: se pueden correr
VARIAS SEMILLAS por punto en vez de una sola.

QUÉ SE REUSA, TAL CUAL, SIN TOCAR NINGÚN ARCHIVO DE ORIGEN (sólo import):
  - `ON77_sistemaA_cierre.py`: `dens_bar_para_n` (el atajo YA DOCUMENTADO de remuestreo bootstrap desde
    el pool REAL de bateria_n2000/dens_bar.npy), y las constantes D_CAUSAL/K_CAUSAL/SEED_EJES/N_LIST.
  - `ON77_sistemaB_cierre.py`: `construir_malla_historica_acotada` (el mecanismo de reorganización de
    presupuesto acotado con memoria genuina YA VALIDADA, Jaccard != 1.0 con H), DENS_BAR_N2000, H_LIST y
    sus constantes de semilla.
  - `cs072_modulos/piezas/p_semilla_causal.py`: `malla_causal_atomos` (regla generativa de Sistema A: kNN
    en D=3 ejes derivados de densidad, UN SOLO pase, sin barajado -- REAL exacto).
  - `cs084_espectro_laplaciano.py`: `dimension_espectral` (traza del núcleo de calor Tr(e^{-tL}) ~
    t^{-d_s/2}, LITERALMENTE la misma función, sin reescribirla) y la grilla `T_GRID`. El resto del
    método de diagonalización (denso, L=D-A, recorte de ruido numérico <0) se replica aquí porque cs084
    parte de un tejido distinto (`cs080_renormalizacion.construir_sustrato`, motor proceso066) -- acá el
    tejido es el de Sistema A/B de O-N7.7, no el de CS066/CS080.
  - `cs064_smoke.py`: `adj_sparse` (adyacencia dispersa desde una lista de sets), tal cual.

CUATRO CANTIDADES ESPECTRALES reportadas para AMBOS sistemas, sin elegir de antemano cuál es "la buena"
(instrucción explícita del encargo, para no sesgar la lectura):
  1. λ_max (el autovalor más grande -- "el armónico más agudo")
  2. λ2 (conectividad algebraica -- "qué tan fácil es cortar el grafo en dos pedazos", MÁS alto = MÁS
     difícil de desconectar)
  3. dispersión (desviación estándar de todo el espectro -- "qué tan ancho suena")
  4. dimensión espectral d_s(t) por núcleo de calor, en 4 tiempos de difusión (corto/medio/largo alcance)

MÉTODO: diagonalización DENSA completa (scipy.linalg.eigh), igual que CS084 -- necesaria para el núcleo de
calor (usa el espectro COMPLETO, no sólo los extremos). Costo medido en CS084 en esta misma máquina: N=8000
dense eigh ~65s. Ni Sistema A ni Sistema B corren Phantom ni el layout de resortes (O(N^2)/iter) -- sólo
se construye la ADYACENCIA (kNN barato, cKDTree) y se diagonaliza. Esto libera presupuesto de tiempo para
correr VARIAS SEMILLAS por punto (3 para Sistema A -- dominado por el costo de N=8000 --, 5 para Sistema B
-- N=2000 fijo, barato en cualquier caso).

No toca `ON77_sistemaA_cierre.py`, `ON77_sistemaB_cierre.py`, `cs084_espectro_laplaciano.py` -- sólo
importa/reusa funciones de ellos. No se declara cierre ni veredicto sobre O-N7.7 -- sólo se reportan
números; la lectura final es de Alexis.

Codea/ejecuta: CC (Claude).
"""
from __future__ import annotations
import csv
import json
import os
import sys
import time

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

# ---- reuso tal cual, SIN tocar ningún archivo de origen ----
from ON77_sistemaA_cierre import dens_bar_para_n, D_CAUSAL, K_CAUSAL, SEED_EJES, N_LIST
from ON77_sistemaB_cierre import (
    construir_malla_historica_acotada, DENS_BAR_N2000, H_LIST,
    SEED_ORDEN_HISTORIA, SEED_RECONSIDERACION, TAM_MUESTRA_RECONSIDERACION,
)
from cs072_modulos.piezas.p_semilla_causal import malla_causal_atomos
from cs084_espectro_laplaciano import dimension_espectral, T_GRID   # función reusada tal cual
import cs064_smoke as SM                                             # adj_sparse, tal cual

OUT_JSON = os.path.join(_HERE, "cs089_on77_espectral_resultado.json")
OUT_CSV_A = os.path.join(_HERE, "cs089_sistemaA_espectral.csv")
OUT_CSV_B = os.path.join(_HERE, "cs089_sistemaB_espectral.csv")

# Sistema A: 3 réplicas variando SÓLO el embedding D=3 (seed_ejes) -- dens_bar sigue siendo el mismo pool
# físico REAL (bootstrap determinista) por cada N, como en ON77_sistemaA_cierre. El costo lo domina N=8000
# (dense eigh), así que 3 réplicas caben cómodas en el presupuesto.
SEEDS_EJES_A = [SEED_EJES, SEED_EJES + 1, SEED_EJES + 2]

# Sistema B: 5 réplicas variando el ORDEN de revelación (seed_orden) -- N=2000 fijo en todos los H, barato
# en cualquier caso, así que se pueden dar más semillas (mismo criterio que CS084: 5 semillas por brazo).
SEEDS_ORDEN_B = [SEED_ORDEN_HISTORIA + i for i in range(5)]

T_PUNTOS = (0.05, 0.2, 1.0, 5.0)          # mismos 4 tiempos representativos que CS084, sin elegir "el bueno"
PRESUPUESTO_S_ANTES_DE_16000 = 20 * 60    # sólo se intenta N=16000 si, tras 2000/4000/8000, sobra tiempo


# ============================ NÚCLEO ESPECTRAL (replicado de cs084_espectro_laplaciano.laplaciano_denso,
#                               adaptado a que acá el tejido llega como dict-de-sets {nodo: vecinos}) =====
def laplaciano_denso_desde_adj(adj_dict, n):
    """Construye L = D - A denso desde un dict-de-sets {nodo: set(vecinos)} -- el formato que devuelven
    TANTO `malla_causal_atomos` (Sistema A) COMO `construir_malla_historica_acotada` (Sistema B) -- y
    diagonaliza COMPLETO. Mismo método exacto que `laplaciano_denso()` de cs084_espectro_laplaciano.py
    (denso, recorte de ruido numérico <0, orden ascendente); se reimplementa acá (en vez de importarla)
    porque esa función de cs084 arranca de `cs080_renormalizacion.construir_sustrato` -- un tejido
    distinto (motor proceso066 de CS066/CS080), no el de Sistema A/B de O-N7.7."""
    adj_lista = [adj_dict.get(i, set()) for i in range(n)]
    A = SM.adj_sparse(adj_lista, n)
    n_comp, labels = connected_components(A, directed=False)
    giant = float(np.max(np.bincount(labels))) / n
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sparse.diags(deg) - A
    Ld = L.toarray()
    w = eigh(Ld, eigvals_only=True)
    w = np.clip(w, 0.0, None)     # L es semidefinido positivo; recorta ruido numérico <0
    w.sort()
    return w, int(n_comp), giant


def resumen_espectral(w, n_comp, giant):
    """Las 4 cantidades pedidas por el encargo, sin elegir de antemano cuál es 'la buena': λ_max, λ2
    (conectividad algebraica -- el primer autovalor DESPUÉS de los n_comp triviales en 0, uno por
    componente conexa), dispersión (std de todo el espectro), y dimensión espectral d_s(t) en 4 tiempos."""
    lam2 = float(w[n_comp]) if n_comp < len(w) else float("nan")
    lam_max = float(w[-1])
    std_eig = float(np.std(w))
    mean_eig = float(np.mean(w))
    d_s_curve, _tr = dimension_espectral(w, t_grid=T_GRID)
    ds_puntos = {}
    for tt in T_PUNTOS:
        j = int(np.argmin(np.abs(T_GRID - tt)))
        ds_puntos[f"d_s_t{tt}"] = float(d_s_curve[j])
    out = dict(n_componentes=n_comp, giant_frac=round(giant, 4), lambda2=lam2, lambda_max=lam_max,
               std_eig=std_eig, mean_eig=mean_eig)
    out.update(ds_puntos)
    return out


# ============================ SISTEMA A -- N variable, regla fija (malla_causal_atomos), SIN Phantom =====
def correr_un_n_espectral(n, seeds_ejes=SEEDS_EJES_A):
    """Para un N dado: construye el pool de densidad (mismo atajo bootstrap YA DOCUMENTADO en
    ON77_sistemaA_cierre.dens_bar_para_n) UNA vez, y sobre él construye la malla causal (kNN, D=3, un solo
    pase, sin barajar -- REAL exacto) con varias semillas del embedding D=3, diagonalizando cada una."""
    dens_bar = dens_bar_para_n(n)
    filas = []
    for se in seeds_ejes:
        t0 = time.time()
        adj, m = malla_causal_atomos(dens_bar, D=D_CAUSAL, k=K_CAUSAL, seed_ejes=se)
        w, n_comp, giant = laplaciano_denso_desde_adj(adj, n)
        r = resumen_espectral(w, n_comp, giant)
        r.update(N=n, seed_ejes=se, t_s=round(time.time() - t0, 2))
        filas.append(r)
        print(f"[A-espectral] N={n:<6d} seed_ejes={se:<6d} n_comp={n_comp:<4} giant={giant:.3f} "
              f"lambda2={r['lambda2']:.5f} lambda_max={r['lambda_max']:.2f} std_eig={r['std_eig']:.3f} "
              f"d_s(t=1.0)={r['d_s_t1.0']:.3f}  ({r['t_s']}s)", flush=True)
    return filas


def _media_std(filas, campo):
    vals = np.array([f[campo] for f in filas], dtype=float)
    return float(np.nanmean(vals)), float(np.nanstd(vals))


CAMPOS_NUMERICOS = ["lambda2", "lambda_max", "std_eig"] + [f"d_s_t{tt}" for tt in T_PUNTOS]


def sistema_a(n_list=N_LIST):
    t_total = time.time()
    print(f"=== Sistema A espectral: N variable {n_list}, regla fija (malla_causal_atomos, sin barajar), "
          f"{len(SEEDS_EJES_A)} semillas de embedding por N, SIN Phantom ===", flush=True)
    filas_crudas = []
    resumen_por_n = []
    for n in n_list:
        filas_n = correr_un_n_espectral(n)
        filas_crudas.extend(filas_n)
        fila_media = dict(N=n, n_seeds=len(filas_n))
        for campo in CAMPOS_NUMERICOS:
            m, s = _media_std(filas_n, campo)
            fila_media[f"{campo}_media"] = round(m, 5)
            fila_media[f"{campo}_std"] = round(s, 5)
        resumen_por_n.append(fila_media)
        with open(OUT_JSON.replace(".json", "_parcial.json"), "w") as fp:
            json.dump(dict(sistema_a_crudo=filas_crudas, sistema_a_resumen=resumen_por_n,
                            tiempo_hasta_ahora_s=round(time.time() - t_total, 1)), fp, indent=2)
        if time.time() - t_total > PRESUPUESTO_S_ANTES_DE_16000 and 16000 not in n_list and n == n_list[-1]:
            print("[AVISO] presupuesto (20 min) ya usado en Sistema A -- no se intenta N=16000.", flush=True)

    # ganancia marginal por N, sobre TODAS las cantidades numéricas (no se elige una sola de antemano)
    for idx, fila in enumerate(resumen_por_n):
        if idx == 0:
            for campo in CAMPOS_NUMERICOS:
                fila[f"ganancia_marginal_{campo}"] = None
            continue
        prev = resumen_por_n[idx - 1]
        dN = fila["N"] - prev["N"]
        for campo in CAMPOS_NUMERICOS:
            dq = fila[f"{campo}_media"] - prev[f"{campo}_media"]
            fila[f"ganancia_marginal_{campo}"] = round(dq / dN, 8) if dN else None

    tiempo_total = time.time() - t_total
    print(f"\n[Sistema A espectral] completo en {tiempo_total/60:.1f} min", flush=True)
    return dict(filas_crudas=filas_crudas, resumen_por_n=resumen_por_n, tiempo_total_s=round(tiempo_total, 1))


def intentar_n16000(t_ya_usado_s):
    """N=16000 SÓLO si, tras 2000/4000/8000, sobra presupuesto claro -- dense eigh escala ~O(N^3), así que
    16000 cuesta ~8x el costo medido de 8000 (~65s en CS084) = varios minutos SÓLO en diagonalizar, más el
    riesgo de memoria (matriz densa 16000x16000 ~2GB). Se intenta con 1 sola semilla (no comparable 1:1 con
    el resto del sweep, que usa 3) y con try/except -- si falla o se pasa de tiempo, se documenta, no se
    fuerza."""
    if t_ya_usado_s > PRESUPUESTO_S_ANTES_DE_16000:
        print(f"[N=16000] NO se intenta -- ya se usaron {t_ya_usado_s/60:.1f} min, por encima del umbral "
              f"de {PRESUPUESTO_S_ANTES_DE_16000/60:.0f} min reservado para 2000/4000/8000.", flush=True)
        return None
    try:
        print("[N=16000] intentando con 1 sola semilla (fuera del sweep principal de 3 semillas)...",
              flush=True)
        filas = correr_un_n_espectral(16000, seeds_ejes=[SEED_EJES])
        fila_media = dict(N=16000, n_seeds=1, nota="1 sola semilla, no comparable 1:1 con 2000/4000/8000 (3 semillas)")
        for campo in CAMPOS_NUMERICOS:
            m, s = _media_std(filas, campo)
            fila_media[f"{campo}_media"] = round(m, 5)
            fila_media[f"{campo}_std"] = round(s, 5)
        return dict(filas_crudas=filas, resumen=fila_media)
    except Exception as e:
        print(f"[N=16000] ABORTADO ({type(e).__name__}: {e}) -- se documenta, no se fuerza.", flush=True)
        return dict(error=f"{type(e).__name__}: {e}")


# ============================ SISTEMA B -- N=2000 fijo, H variable, mecanismo YA VALIDADO, SIN Phantom ===
def correr_un_h_espectral(dens_bar, h, seeds_orden=SEEDS_ORDEN_B):
    n = len(dens_bar)
    filas = []
    for so in seeds_orden:
        t0 = time.time()
        adj, _n, n_reorg = construir_malla_historica_acotada(
            dens_bar, n_batches=h, seed_orden=so, tam_muestra=TAM_MUESTRA_RECONSIDERACION,
            seed_reconsideracion=SEED_RECONSIDERACION)
        w, n_comp, giant = laplaciano_denso_desde_adj(adj, n)
        r = resumen_espectral(w, n_comp, giant)
        r.update(H=h, N=n, seed_orden=so, n_reorganizaciones=n_reorg, t_s=round(time.time() - t0, 2))
        filas.append(r)
        print(f"[B-espectral] H={h:<3d} seed_orden={so:<6d} n_reorg={n_reorg:<5d} n_comp={n_comp:<4} "
              f"giant={giant:.3f} lambda2={r['lambda2']:.5f} lambda_max={r['lambda_max']:.2f} "
              f"std_eig={r['std_eig']:.3f} d_s(t=1.0)={r['d_s_t1.0']:.3f}  ({r['t_s']}s)", flush=True)
    return filas


def sistema_b(h_list=H_LIST):
    t_total = time.time()
    print(f"\n=== Sistema B espectral: N=2000 FIJO, H variable {h_list}, mecanismo de reorganización "
          f"acotada YA VALIDADO (memoria genuina), {len(SEEDS_ORDEN_B)} semillas de orden por H, SIN "
          f"Phantom ===", flush=True)
    dens_bar = np.load(DENS_BAR_N2000)
    n = len(dens_bar)
    print(f"[B] N={n} (pool REAL de bateria_n2000, mismo que Sistema B cierre)", flush=True)

    filas_crudas = []
    resumen_por_h = []
    for h in h_list:
        filas_h = correr_un_h_espectral(dens_bar, h)
        filas_crudas.extend(filas_h)
        fila_media = dict(H=h, N=n, n_seeds=len(filas_h))
        for campo in CAMPOS_NUMERICOS:
            m, s = _media_std(filas_h, campo)
            fila_media[f"{campo}_media"] = round(m, 5)
            fila_media[f"{campo}_std"] = round(s, 5)
        fila_media["n_reorganizaciones_media"] = round(
            float(np.mean([f["n_reorganizaciones"] for f in filas_h])), 2)
        resumen_por_h.append(fila_media)
        with open(OUT_JSON.replace(".json", "_parcial_B.json"), "w") as fp:
            json.dump(dict(sistema_b_crudo=filas_crudas, sistema_b_resumen=resumen_por_h,
                            tiempo_hasta_ahora_s=round(time.time() - t_total, 1)), fp, indent=2)

    tiempo_total = time.time() - t_total
    print(f"\n[Sistema B espectral] completo en {tiempo_total/60:.1f} min", flush=True)
    return dict(filas_crudas=filas_crudas, resumen_por_h=resumen_por_h, tiempo_total_s=round(tiempo_total, 1))


# ============================ ORQUESTADOR ============================
def main():
    t0 = time.time()
    resultado_a = sistema_a()
    resultado_n16000 = intentar_n16000(time.time() - t0)
    resultado_b = sistema_b()

    resultado = dict(
        sistema_a=resultado_a,
        sistema_a_n16000=resultado_n16000,
        sistema_b=resultado_b,
        seeds_ejes_a=SEEDS_EJES_A,
        seeds_orden_b=SEEDS_ORDEN_B,
        t_puntos_dim_espectral=list(T_PUNTOS),
        tiempo_total_s=round(time.time() - t0, 1),
    )
    with open(OUT_JSON, "w") as fp:
        json.dump(resultado, fp, indent=2)

    # CSVs planos para inspección rápida
    with open(OUT_CSV_A, "w", newline="") as f:
        campos = list(resultado_a["filas_crudas"][0].keys())
        wr = csv.DictWriter(f, fieldnames=campos)
        wr.writeheader()
        for fila in resultado_a["filas_crudas"]:
            wr.writerow(fila)
        if resultado_n16000 and "filas_crudas" in resultado_n16000:
            for fila in resultado_n16000["filas_crudas"]:
                wr.writerow({k: fila.get(k, "") for k in campos})
    with open(OUT_CSV_B, "w", newline="") as f:
        campos_b = list(resultado_b["filas_crudas"][0].keys())
        wr = csv.DictWriter(f, fieldnames=campos_b)
        wr.writeheader()
        for fila in resultado_b["filas_crudas"]:
            wr.writerow(fila)

    print(f"\n[TOTAL] {resultado['tiempo_total_s']/60:.1f} min -> {OUT_JSON}", flush=True)

    print("\n=== SISTEMA A -- resumen por N (media ± std sobre semillas) ===")
    print(f"{'N':<8}{'lambda2':<14}{'lambda_max':<14}{'std_eig':<12}{'d_s(t=1.0)':<14}{'gan.marg lambda_max':<22}")
    for fila in resultado_a["resumen_por_n"]:
        gm = fila["ganancia_marginal_lambda_max"]
        gm_str = f"{gm:+.5f}" if gm is not None else "--"
        print(f"{fila['N']:<8}{fila['lambda2_media']:<14.5f}{fila['lambda_max_media']:<14.2f}"
              f"{fila['std_eig_media']:<12.3f}{fila['d_s_t1.0_media']:<14.3f}{gm_str:<22}")
    if resultado_n16000 and "resumen" in resultado_n16000:
        r16 = resultado_n16000["resumen"]
        print(f"{r16['N']:<8}{r16['lambda2_media']:<14.5f}{r16['lambda_max_media']:<14.2f}"
              f"{r16['std_eig_media']:<12.3f}{r16['d_s_t1.0_media']:<14.3f}(1 semilla, ver nota)")

    print("\n=== SISTEMA B -- resumen por H (media ± std sobre semillas, N=2000 fijo) ===")
    print(f"{'H':<6}{'lambda2':<14}{'lambda_max':<14}{'std_eig':<12}{'d_s(t=1.0)':<14}")
    for fila in resultado_b["resumen_por_h"]:
        print(f"{fila['H']:<6}{fila['lambda2_media']:<14.5f}{fila['lambda_max_media']:<14.2f}"
              f"{fila['std_eig_media']:<12.3f}{fila['d_s_t1.0_media']:<14.3f}")

    return resultado


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
