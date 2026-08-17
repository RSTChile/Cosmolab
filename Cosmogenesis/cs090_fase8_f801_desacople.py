"""
cs090_fase8_f801_desacople.py — FASE VIII, tarea F8-01: DESACOPLAR LAS CUATRO MEDIDAS DE APIÑAMIENTO
====================================================================================================

QUÉ PREGUNTA CONTESTA
---------------------
F7-03 (`FASE7_F703_grados_y_triangulos_fijos_CS.md`) dejó probado que, con los grados clavados nodo por
nodo Y el número de triángulos clavado (diferencia máxima real: 1 triángulo), la masa que acreta Phantom
cambia +13.8% (12/12 grafos, p=4.9e-04, 28.7 partículas contra un grano de 1) según CÓMO estén
organizados esos triángulos.

Pero las cuatro medidas que capturan esa organización viajaron JUNTAS y no se sabe cuál manda:

  - `tri_por_arista_media`      (triángulos por arista que los sostiene)   ρ = +0.776 con la masa
  - `gini_tri_nodo`             (concentración por nodo)                   ρ = +0.736
  - `frac_aristas_multi_tri`    (solapamiento: aristas en >1 triángulo)    ρ = +0.696
  - `frac_aristas_en_triangulo` (tamaño del soporte en aristas)            ρ = −0.781

Esta tarea fabrica grafos donde UNA de esas medidas se mueve y las otras quedan lo más fijas posible, y
reporta CUÁNTO se desacopló de verdad (un desacople fallido también es información).

UN LÍMITE ALGEBRAICO QUE HAY QUE DECIR ANTES DE EMPEZAR
-------------------------------------------------------
Con el nº de aristas `m` fijo y el nº de triángulos `T` fijo, la suma de "carga" sobre las aristas es
exactamente `Σ_e t_e = 3T`. Entonces, llamando `D = frac_aristas_en_triangulo` (fracción de aristas con
al menos un triángulo) y `A = tri_por_arista_media` (media de `t_e` sobre esas aristas):

        A · D · m = 3T   →   A = 3T / (D·m)      con T y m FIJOS por diseño.

Es decir: **`tri_por_arista_media` y `frac_aristas_en_triangulo` NO son dos medidas, son una sola
medida y su inversa.** Ningún experimento puede separarlas mientras grados y triángulos estén fijos, y
eso explica sin misterio por qué en F7-03 salieron con ρ casi simétricos (+0.776 y −0.781). Se verifica
numéricamente en el CSV (columna `identidad_A_D_resid`, que debe dar ~0 en todos los brazos).

Quedan entonces, como mucho, TRES ejes realmente independientes:
  (1) el TAMAÑO DEL SOPORTE  D  (equivalentemente la carga media A = 3T/(D·m)),
  (2) la FORMA de la distribución de carga por arista a soporte dado: qué fracción de las aristas del
      soporte lleva ≥2 triángulos (`frac_aristas_multi_tri`) y cuánto llega a cargar la más cargada
      (`tri_por_arista_max`) — media contra cola,
  (3) la CONCENTRACIÓN POR NODO  `gini_tri_nodo` (y su pariente `frac_nodos_en_triangulo`).

LOS CINCO BRAZOS Y LOS CUATRO CONTRASTES
-----------------------------------------
Todos los brazos salen del MISMO piso (el grafo original con los triángulos destruidos por
`F702.bajar_clustering`, la misma función y dosis de F7-02/F7-03) y se rebobinan al MISMO número de
triángulos T* = mínimo de los cinco techos, exactamente con el mecanismo de F7-03 (lista de swaps
aceptados + replay).

  | brazo     | criterio de aceptación                                        | qué mueve |
  |-----------|---------------------------------------------------------------|-----------|
  | `abanico` | NUEVO. Empaquetamiento disjunto en aristas (mismo `_ok_disjunto` que el brazo `disj` de F7-03) PERO con el ápice sorteado de una bolsa de nodos ya calientes y exigiendo que el trío nuevo deje algún nodo con ≥2 triángulos. Resultado: molinos de viento — triángulos que comparten VÉRTICE y nunca ARISTA. | Gini ↑ con solapamiento clavado en 0 |
  | `disp`    | IMPORTADO TAL CUAL de F7-03 (`modo="disp"`): cupo de triángulos por nodo que sube por capas. Triángulos repartidos sobre la mayor cantidad de nodos distintos, también sin compartir aristas. | Gini ↓ con solapamiento ~0 |
  | `libro`   | NUEVO. Se sortea una ARISTA de una bolsa pesada por carga (rico se hace más rico) y se le apila un triángulo encima reusando el mismo par de nodos como "lomo". Resultado: pocas aristas MUY cargadas y muchas con exactamente 1. | cola ↑ (máximo por arista) con `frac_multi` bajo |
  | `malla`   | NUEVO. Se exige compartir arista con un triángulo existente (como `solap`) PERO con TOPE de 2 triángulos por arista tocada. Resultado: muchísimas aristas con exactamente 2, ninguna con más. | `frac_multi` ↑ con la cola cortada |
  | `solap`   | IMPORTADO TAL CUAL de F7-03: ancla para poder comparar con el +13.8% ya medido sobre los mismos grafos base. | referencia |

  CONTRASTE 1 — `abanico` vs `disp`  → **el único desacople GARANTIZADO POR CONSTRUCCIÓN**: los dos
    brazos son empaquetamientos disjuntos en aristas, así que con el mismo T tienen, exactamente,
    `frac_aristas_multi_tri = 0`, `tri_por_arista_media = 1.000` y `frac_aristas_en_triangulo = 3T/m`.
    Las tres medidas de arista quedan clavadas por álgebra, no por suerte. Lo único que cambia es la
    concentración por NODO. Si la masa se mueve acá, manda el nodo; si no se mueve, manda la arista.
  CONTRASTE 2 — `libro` vs `malla`  → misma familia (los dos apilan sobre aristas compartidas), media
    de carga parecida, pero MÁXIMO por arista muy distinto: ¿mandan unas pocas aristas supercargadas o
    la distribución entera?
  CONTRASTE 3 — `abanico` vs `libro` → concentración por nodo parecida (los dos hacen "barrios"), pero
    solapamiento 0.00 contra alto: aísla el solapamiento del Gini.
  CONTRASTE 4 (ancla) — `solap` vs `disp` → reproduce el eje de F7-03 sobre estos mismos grafos base.

UN TECHO ESTRUCTURAL QUE HAY QUE ANTICIPAR
-------------------------------------------
La propuesta del equipo pedía "máximo 6 contra máximo 30 triángulos por arista". **No es fabricable con
estos grafos:** una arista (a,b) tiene tantos triángulos como vecinos comunes, y con `kcap ≤ 8` eso es
como mucho 7. El mismo techo `d(d−1)/2` que ya limitó al brazo `conc` de F7-03. Lo que sí es fabricable
es "máximo 2 contra máximo 5-7", y así se reporta.

QUÉ REUSA Y QUÉ NO TOCA
------------------------
Sólo importa, jamás modifica: `cs090_fase7_f703_organizacion` (motor de brazos, `replay`,
`techo_conectado`, `medir_organizacion`, `_ok_disjunto`, `_tri_arista`, `_tri_nodo`),
`cs090_fase7_f702_escalera` (swap elemental que conserva grados + selección de los grafos base),
`cs090_fase6_o3b_rewiring`, `cs090_fase5b_phantom_adaptador`, `cs090_diam_corregido`.
Prefijo `f801`, jamás usado antes. No corre Phantom (eso es `cs090_fase8_f801_correr.py`).
No declara cierre ni veredicto.

GUARDADO DE LOS GRAFOS (regla de Fase VIII)
--------------------------------------------
F8-00 todavía no dejó su utilidad, así que cada brazo guarda su grafo en su propia carpeta como
`grafo_f801.npz` (numpy comprimido) con dos arrays: `aristas` (E×2, int32, i<j, ordenado) y `grados`
(N, int32). Se relee con `np.load(ruta)["aristas"]`. Formato documentado acá y en el informe.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

import cs090_fase6_o3b_rewiring as O3B                   # sólo import
import cs090_fase7_f702_escalera as F702                 # sólo import (swap que conserva grados)
import cs090_fase7_f703_organizacion as F703             # sólo import (motor de brazos de F7-03)
from cs090_fase5b_phantom_adaptador import (             # sólo import (pipeline validado Fase V-B)
    reconstruir_regla_a2b0c2, generar_ic_masa_fija_desde_grafo,
)

# ---------------------------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------------------------
N_NODOS = F703.N_NODOS               # 2000
N_SWEEPS = F703.N_SWEEPS             # 14
SEED_LAYOUT = F703.SEED_LAYOUT       # 12345 — MISMO layout en todos los brazos (regla de la fase)

MULT_SEED_DES = 10300                # multiplicador NUEVO. Ya usados en la línea:
                                     # 1000/2000/5000/6000/7000/7500/8000/9100/9700/9800.

FACTOR_INTENTOS_BAJAR = F702.FACTOR_INTENTOS_BAJAR       # mismo piso que F7-02/F7-03
FACTOR_INTENTOS_SUBIR = F703.FACTOR_INTENTOS_SUBIR       # 60 × aristas
PARADA_SIN_ACEPTAR = F703.PARADA_SIN_ACEPTAR             # 40000
ARRANQUE_LIBRE = F703.ARRANQUE_LIBRE                     # 12 primeros triángulos sin criterio

PACIENCIA_F801 = 12000               # tras tantos intentos seguidos sin aceptar, el brazo AFLOJA la
                                     # parte "de barrio" de su criterio (para no morir de hambre), pero
                                     # NUNCA afloja su invariante (disjunción en `abanico`, tope en
                                     # `malla`): el invariante es lo que hace válido el contraste.
PROB_BOLSA_ABANICO = 0.85            # prob. de sortear el ápice de la bolsa de nodos calientes
PROB_BOLSA_LIBRO = 0.85              # prob. de sortear el lomo de la bolsa de aristas cargadas
MIN_TRI_ABANICO = 2                  # `abanico`: el trío nuevo debe dejar algún nodo con ≥2 triángulos
MIN_LOMO_LIBRO = 2                   # `libro`: el lomo debe quedar con ≥2 triángulos
CAP_MALLA = 2                        # `malla`: NINGUNA arista afectada puede pasar de 2 triángulos
PROB_NUCLEO_COLA = 0.02              # `cola`: probabilidad de abrir un núcleo nuevo (una arista que
                                     # queda autorizada a cargar más de 1 triángulo). Baja a propósito:
                                     # la idea del brazo es que la carga extra se concentre en MUY
                                     # POCAS aristas, y el resto del grafo crezca disjunto.
CAP_COLA = 8                         # tope de un núcleo (por encima del techo estructural: una arista
                                     # no puede tener más triángulos que vecinos comunes, ≤ kcap−1 = 7)

CHECK_GIGANTE_CADA = F703.CHECK_GIGANTE_CADA             # 20
FRAC_GIGANTE_MINIMA = F702.FRAC_GIGANTE_MINIMA           # 0.97

BRAZOS = ("abanico", "disp", "cola", "malla", "solap")
BRAZOS_NUEVOS = ("abanico", "libro", "malla", "cola")    # los que usan el motor de este archivo
BRAZOS_F703 = ("disp", "solap")                          # los que se delegan TAL CUAL a F7-03

BASE_SALIDA = "/Users/alexis/phantom_cs073/bateria_fase8_f801_desacople"
RUTA_ESTRUCTURA = f"{HERE}/cs090_fase8_f801_estructura.csv"


# =============================================================================================
# 1) MOTOR de los tres brazos NUEVOS
#    (misma mecánica de F7-03: swap dirigido que conserva el grado de los cuatro nodos, historial de
#     componente gigante, lista de swaps aceptados para poder rebobinar; lo único propio es el
#     criterio de aceptación)
# =============================================================================================
def _cargas_tocadas(adj, u, x, y, p, q):
    """Carga (nº de triángulos) de las aristas que el swap acaba de tocar, con el swap YA aplicado.

    Las tres primeras son las del triángulo nuevo (u-x, u-y, x-y); la cuarta es la otra arista que el
    swap crea (p-q), que también puede haber cerrado triángulos."""
    return (F703._tri_arista(adj, u, x), F703._tri_arista(adj, u, y),
            F703._tri_arista(adj, x, y), F703._tri_arista(adj, p, q))


def _aristas_afectadas(adj, x, y, p, q):
    """TODAS las aristas cuya carga pudo AUMENTAR con este swap, con el swap ya aplicado.

    El swap saca (x-p) y (y-q) y pone (x-y) y (p-q). Sacar aristas nunca sube la carga de nadie, así
    que sólo pueden haber subido las aristas de los triángulos que las DOS aristas nuevas cierran: la
    propia arista nueva (a-b) y, por cada vecino común w, las dos aristas (a-w) y (b-w).

    (La primera versión de este archivo miraba sólo cuatro aristas fijas y se le escapaban las (a-w):
    el brazo `malla`, que debía tener tope 2, llegó a tener aristas con 4 en el piloto. Con esta
    versión el tope es exacto.)"""
    S = set()
    for a, b in ((x, y), (p, q)):
        if b not in adj[a]:
            continue
        S.add((min(a, b), max(a, b)))
        for w in (adj[a] & adj[b]):
            S.add((min(a, w), max(a, w)))
            S.add((min(b, w), max(b, w)))
    return S


def _max_carga_afectada(adj, x, y, p, q):
    """Carga máxima entre las aristas que este swap pudo cargar (0 si no cargó ninguna)."""
    S = _aristas_afectadas(adj, x, y, p, q)
    return max((F703._tri_arista(adj, a, b) for (a, b) in S), default=0)


def cerrar_triangulos_f801(adj, edges, idx, N, rng, modo, t_inicial, n_intentos):
    """Corre un brazo NUEVO (`abanico`, `libro`, `malla`) desde el piso hasta su techo.

    Devuelve exactamente la misma estructura que `F703.cerrar_triangulos`: (swaps, t_por_swap, hist),
    con `swaps` como lista de tuplas (x, p, q, y) reproducibles por `F703.replay`."""
    assert modo in BRAZOS_NUEVOS, modo
    t = t_inicial
    swaps, t_por_swap, hist = [], [], [(0, t_inicial, None)]
    bolsa_nodo = []                  # nodos calientes: 1 aparición por triángulo que ayudaron a cerrar
    bolsa_ar = []                    # aristas cargadas: 1 aparición por triángulo que sostienen
    nucleos = set()                  # brazo `cola`: las poquísimas aristas autorizadas a cargar >1
    nucleos_lista = []               # la misma cosa como lista, para sortear barato
    sin_aceptar = 0
    cand = np.array([i for i in range(N) if len(adj[i]) >= 2], dtype=int)
    if len(cand) == 0:
        return swaps, t_por_swap, hist

    for _ in range(n_intentos):
        sin_aceptar += 1
        if sin_aceptar > PARADA_SIN_ACEPTAR:
            break
        relajado = sin_aceptar > PACIENCIA_F801        # afloja SÓLO la parte "de barrio"

        # ---------- elegir el trío (u = ápice, x, y = los dos que se van a unir) ----------
        pila = bolsa_ar if modo == "libro" else (nucleos_lista if modo == "cola" else None)
        if modo in ("libro", "cola") and pila and rng.random() < PROB_BOLSA_LIBRO:
            # se apila sobre un LOMO ya cargado: u-y es el lomo, x el vecino nuevo del abanico
            a, b = pila[int(rng.integers(0, len(pila)))]
            if rng.random() < 0.5:
                a, b = b, a
            u, y = int(a), int(b)
            if y not in adj[u] or len(adj[u]) < 2:
                continue
            vx = [v for v in adj[u] if v != y and v not in adj[y]]
            if not vx:
                continue
            x = int(vx[int(rng.integers(0, len(vx)))])
        else:
            if modo == "abanico" and bolsa_nodo and rng.random() < PROB_BOLSA_ABANICO:
                u = int(bolsa_nodo[int(rng.integers(0, len(bolsa_nodo)))])
                if len(adj[u]) < 2:
                    continue
            else:
                u = int(cand[int(rng.integers(0, len(cand)))])
            vec_u = list(adj[u])
            if len(vec_u) < 2:
                continue
            i1, i2 = rng.integers(0, len(vec_u), size=2)
            if i1 == i2:
                continue
            x, y = int(vec_u[int(i1)]), int(vec_u[int(i2)])
            if y in adj[x]:
                continue                                  # ya estaba cerrado

        # ---------- elegir los dos donantes p, q (los que ceden la arista) ----------
        vx = [v for v in adj[x] if v != u and v != y]
        vy = [v for v in adj[y] if v != u and v != x]
        if not vx or not vy:
            continue
        p = int(vx[int(rng.integers(0, len(vx)))])
        q = int(vy[int(rng.integers(0, len(vy)))])
        if p == q or len({x, p, y, q}) < 4:
            continue
        if (min(p, q), max(p, q)) in idx:
            continue                                      # p-q ya existe: crearía arista doble

        delta = F702._delta_triangulos(adj, x, p, q, y)   # APLICA el swap
        aceptar = delta > 0

        if aceptar:
            if modo == "abanico":
                # INVARIANTE (nunca se relaja): empaquetamiento disjunto en aristas
                aceptar = F703._ok_disjunto(adj, x, y, p, q)
                if aceptar and t >= ARRANQUE_LIBRE and not relajado:
                    # criterio "de barrio": el trío nuevo se pega, por un VÉRTICE, a lo ya construido
                    aceptar = max(F703._tri_nodo(adj, u), F703._tri_nodo(adj, x),
                                  F703._tri_nodo(adj, y)) >= MIN_TRI_ABANICO
            elif modo == "libro":
                if t >= ARRANQUE_LIBRE and not relajado:
                    aceptar = F703._tri_arista(adj, u, y) >= MIN_LOMO_LIBRO
            elif modo == "malla":
                # INVARIANTE (nunca se relaja): tope EXACTO de carga por arista
                aceptar = _max_carga_afectada(adj, x, y, p, q) <= CAP_MALLA
                if aceptar and t >= ARRANQUE_LIBRE and not relajado:
                    cg = _cargas_tocadas(adj, u, x, y, p, q)
                    aceptar = max(cg[:3]) >= 2            # debe compartir arista con algo ya existente
            elif modo == "cola":
                # todo crece DISJUNTO salvo en unas poquísimas aristas "núcleo", que sí pueden
                # cargar mucho. Así la MEDIA de carga queda parecida a la de `malla` pero el MÁXIMO
                # es varias veces mayor: es el contraste media-contra-cola.
                cargadas = [e for e in _aristas_afectadas(adj, x, y, p, q)
                            if F703._tri_arista(adj, *e) >= 2]
                nuevas = [e for e in cargadas if e not in nucleos]
                if not cargadas:
                    aceptar = True                          # crecimiento disjunto: siempre bien
                elif all(F703._tri_arista(adj, *e) <= CAP_COLA for e in cargadas) and (
                        not nuevas or rng.random() < PROB_NUCLEO_COLA):
                    aceptar = True
                    for e in nuevas:
                        if e not in nucleos:
                            nucleos.add(e)
                            nucleos_lista.append(e)
                else:
                    aceptar = False

        if aceptar:
            F702._confirmar_en_listas(edges, idx, x, p, q, y)
            t += delta
            swaps.append((x, p, q, y))
            t_por_swap.append(t)
            sin_aceptar = 0
            if modo == "abanico":
                bolsa_nodo.extend((u, x, y))
            elif modo == "libro":
                bolsa_ar.extend(((min(u, x), max(u, x)), (min(u, y), max(u, y)),
                                 (min(x, y), max(x, y))))
            if len(swaps) % CHECK_GIGANTE_CADA == 0:
                hist.append((len(swaps), t, O3B.tam_gigante(adj, N)))
        else:
            F702._revertir(adj, x, p, q, y)

    hist.append((len(swaps), t, O3B.tam_gigante(adj, N)))
    return swaps, t_por_swap, hist


# =============================================================================================
# 2) MEDIDAS EXTRA (las de F7-03 no alcanzan: falta la COLA de la distribución por arista)
# =============================================================================================
def medir_carga_aristas(adj, N):
    """Distribución de la carga `t_e` (triángulos por arista) sobre las aristas que sostienen alguno.

    F7-03 medía sólo la MEDIA (`tri_por_arista_media`) y la fracción con ≥2 (`frac_aristas_multi_tri`).
    Para poder preguntar "¿manda la media o mandan unas pocas aristas supercargadas?" hace falta además
    el máximo, los cuantiles y el Gini de esa misma distribución."""
    tris = F703.enumerar_triangulos(adj, N)
    m = sum(len(adj[i]) for i in range(N)) // 2
    carga = {}
    for (a, b, c) in tris:
        for e in ((a, b), (a, c), (b, c)):
            carga[e] = carga.get(e, 0) + 1
    v = np.array(sorted(carga.values()), dtype=float) if carga else np.zeros(0)
    out = dict(n_aristas_soporte=len(v), suma_carga=float(v.sum()) if len(v) else 0.0)
    if len(v):
        out.update(
            tri_por_arista_max=float(v.max()),
            tri_por_arista_p90=float(np.percentile(v, 90)),
            tri_por_arista_p99=float(np.percentile(v, 99)),
            gini_tri_arista=F703._gini(v),
            frac_carga_en_top1pct=float(v[int(np.ceil(0.99 * len(v))) - 1:].sum() / v.sum()),
            frac_aristas_carga3mas=float((v >= 3).mean()),
        )
        # identidad algebraica A·D·m = 3T (debe dar ~0): control de que T y m están realmente fijos
        A = v.mean()
        D = len(v) / m if m else float("nan")
        out["identidad_A_D_resid"] = float(A * D * m - 3 * len(tris))
    else:
        for c in ("tri_por_arista_max", "tri_por_arista_p90", "tri_por_arista_p99", "gini_tri_arista",
                  "frac_carga_en_top1pct", "frac_aristas_carga3mas", "identidad_A_D_resid"):
            out[c] = float("nan")
    return out


def guardar_grafo(adj, N, ruta):
    """Guarda el grafo como npz comprimido: `aristas` (E×2 int32, i<j, ordenado) y `grados` (N int32).
    Se relee con `np.load(ruta)["aristas"]`. (Regla de Fase VIII: guardar los grafos de toda corrida.)"""
    ar = np.array(sorted((i, j) for i in range(N) for j in adj[i] if j > i), dtype=np.int32)
    gr = np.array([len(adj[i]) for i in range(N)], dtype=np.int32)
    np.savez_compressed(ruta, aristas=ar, grados=gr)
    return len(ar)


# =============================================================================================
# 3) PROCESAR un grafo base: piso común -> 5 brazos -> igualar triángulos -> medir -> IC
#    (misma estructura que `F703.procesar_una`; cambian los brazos y las medidas extra)
# =============================================================================================
def procesar_una(sel, generar_ic=True):
    rid, seed = sel["rule_id"], sel["seed"]
    t_ini = time.time()

    p, m = reconstruir_regla_a2b0c2(seed=seed, N=N_NODOS, n_sweeps=N_SWEEPS)
    adj_orig = O3B._a_lista(m["adj_final"], N_NODOS)
    E_orig = O3B.aristas_set(adj_orig, N_NODOS)
    g_orig = O3B.grados(adj_orig, N_NODOS)
    gig_orig = O3B.tam_gigante(adj_orig, N_NODOS)
    c_orig, tr_orig, t_orig = O3B.clustering(adj_orig, N_NODOS)
    piso_gigante = FRAC_GIGANTE_MINIMA * gig_orig

    # ---- piso común: se destruyen los triángulos (misma función y dosis que F7-02/F7-03) ----
    semilla_base = int(seed) * MULT_SEED_DES + 17
    rng_piso = np.random.default_rng(semilla_base)
    adj_p, edges_p, idx_p = F702._estado_desde_adj(adj_orig, N_NODOS)
    n_aristas = len(edges_p)
    t0 = time.time()
    d_baja, acc_baja = F702.bajar_clustering(adj_p, edges_p, idx_p, N_NODOS, rng_piso,
                                             FACTOR_INTENTOS_BAJAR * n_aristas)
    t_piso = t_orig + d_baja
    adj_piso = [set(s) for s in adj_p]
    t_bajar_s = time.time() - t0
    print(f"    piso: tri {t_orig} -> {t_piso}  ({acc_baja} swaps, {t_bajar_s:.1f}s)", flush=True)

    # ---- cada brazo, desde el MISMO piso, hasta su techo ----
    brazos = {}
    for j, modo in enumerate(BRAZOS):
        rng_b = np.random.default_rng(semilla_base + 101 * (j + 1))
        adj_b, edges_b, idx_b = F702._estado_desde_adj(adj_piso, N_NODOS)
        tb = time.time()
        motor = cerrar_triangulos_f801 if modo in BRAZOS_NUEVOS else F703.cerrar_triangulos
        swaps, t_por_swap, hist = motor(adj_b, edges_b, idx_b, N_NODOS, rng_b, modo, t_piso,
                                        FACTOR_INTENTOS_SUBIR * n_aristas)
        k_techo, t_techo = F703.techo_conectado(hist, piso_gigante)
        brazos[modo] = dict(swaps=swaps, t_por_swap=t_por_swap, hist=hist,
                            k_techo=k_techo, t_techo=t_techo,
                            t_techo_sin_restriccion=(t_por_swap[-1] if t_por_swap else t_piso),
                            n_aceptados=len(swaps), t_swaps_s=round(time.time() - tb, 1))
        print(f"    {modo:>7}: techo_conectado tri={t_techo} (k={k_techo}/{len(swaps)}) "
              f"techo_libre={brazos[modo]['t_techo_sin_restriccion']} [{brazos[modo]['t_swaps_s']}s]",
              flush=True)

    # ---- T* = el mínimo de los cinco techos; cada brazo se rebobina al swap más cercano ----
    T_obj = min(brazos[mo]["t_techo"] for mo in BRAZOS)
    filas, tris_conseguidos = [], {}
    for modo in BRAZOS:
        b = brazos[modo]
        tps, k_max = b["t_por_swap"], b["k_techo"]
        if k_max == 0:
            k_mejor = 0
        else:
            k_mejor = min(range(1, k_max + 1), key=lambda k: (abs(tps[k - 1] - T_obj), k))
        b["k_elegido"] = k_mejor
        b["adj"] = F703.replay(adj_piso, b["swaps"], k_mejor)
        tris_conseguidos[modo] = (tps[k_mejor - 1] if k_mejor else t_piso)
    dif_max = max(tris_conseguidos.values()) - min(tris_conseguidos.values())
    print(f"    T*={T_obj}  conseguidos={tris_conseguidos}  dif_max={dif_max}", flush=True)

    adjs_null = O3B.nulls_topo_de(seed, N_NODOS, len(E_orig))
    rng_med = np.random.default_rng(semilla_base + 7)

    for modo in BRAZOS:
        b = brazos[modo]
        adj_v = b["adj"]
        # --- VERIFICACIÓN NUMÉRICA nodo por nodo (no se asume: es el control que valida todo) ---
        g_v = O3B.grados(adj_v, N_NODOS)
        iguales = bool(np.array_equal(g_orig, g_v))
        n_dif = int((g_orig != g_v).sum())
        assert iguales, f"{rid}/{modo}: la secuencia de grados NO se conservó ({n_dif} nodos difieren)"
        E_v = O3B.aristas_set(adj_v, N_NODOS)
        assert len(E_v) == len(E_orig), f"{rid}/{modo}: cambió el nº de aristas"
        assert all(i not in adj_v[i] for i in range(N_NODOS)), f"{rid}/{modo}: hay un bucle i-i"

        cl, tr, ntri = O3B.clustering(adj_v, N_NODOS)
        org = F703.medir_organizacion(adj_v, N_NODOS, rng_med)
        assert org["n_triangulos_medido"] == ntri, f"{rid}/{modo}: dos conteos de triángulos distintos"
        carga = medir_carga_aristas(adj_v, N_NODOS)
        pc = O3B.pendiente_corregida_de_grafo(O3B.canonicalizar(adj_v, N_NODOS), N_NODOS, seed, adjs_null)

        fila = dict(
            rule_id=rid, seed=seed, lote=sel["lote"], K=sel["K"], kcap=sel["kcap"],
            brazo=modo, n_nodos=N_NODOS, n_aristas=len(E_v),
            grado_medio=2.0 * len(E_v) / N_NODOS,
            grados_identicos=iguales, n_nodos_grado_distinto=n_dif,
            solapamiento_aristas=len(E_orig & E_v) / len(E_orig),
            clustering_local=cl, transitividad=tr, n_triangulos=ntri,
            T_objetivo=T_obj, dif_max_triangulos=dif_max,
            gigante=O3B.tam_gigante(adj_v, N_NODOS),
            n_componentes=F702.n_componentes(adj_v, N_NODOS),
            asortatividad=F702.asortatividad_grados(adj_v, N_NODOS),
            pendiente_corr=pc["pendiente"], z_agg=pc["z_agg"], diams=pc["diams"], n_cajas=pc["n_cajas"],
            clustering_original=c_orig, transitividad_original=tr_orig, n_triangulos_original=t_orig,
            t_piso=t_piso, t_techo_brazo=b["t_techo"],
            t_techo_brazo_sin_restriccion=b["t_techo_sin_restriccion"],
            n_swaps_aceptados=b["n_aceptados"], k_elegido=b["k_elegido"],
            t_swaps_s=b["t_swaps_s"], acc_baja=acc_baja, t_bajar_s=round(t_bajar_s, 1),
            gigante_original=gig_orig, semilla_base=semilla_base,
            pendiente_corr_csv_ref=sel["pendiente_corregida"],
            frac_masa_fase5b=sel["frac_masa_fase5b"],
        )
        fila.update(org)
        fila.update(carga)

        if generar_ic:
            carpeta = f"{BASE_SALIDA}/{rid}_s{seed}_f801_{modo}"
            os.makedirs(carpeta, exist_ok=True)
            guardar_grafo(adj_v, N_NODOS, f"{carpeta}/grafo_f801.npz")
            t1 = time.time()
            info = generar_ic_masa_fija_desde_grafo(adj_v, N=N_NODOS, seed_layout=SEED_LAYOUT,
                                                    ruta_salida=f"{carpeta}/cosmogenesis_ic.txt")
            fila["t_ic_s"] = round(time.time() - t1, 2)
            meta = dict(
                tarea="FASE8_F801_desacople_apinamiento", brazo=modo, rule_id=rid, clase="III",
                seed=seed, lote=sel["lote"], N=N_NODOS, seed_layout=SEED_LAYOUT,
                K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"], kcap=p["kcap"],
                sim_thr_frac=p["sim_thr_frac"],
                n_aristas_grafo_final=len(E_v), grado_medio_grafo_final=2.0 * len(E_v) / N_NODOS,
                grados_identicos_al_original=iguales,
                semilla_base=semilla_base, solapamiento_aristas=fila["solapamiento_aristas"],
                clustering_local=cl, transitividad=tr, n_triangulos=ntri,
                T_objetivo=T_obj, dif_max_triangulos=dif_max,
                gigante=fila["gigante"], pendiente_corregida=pc["pendiente"],
                frac_aristas_multi_tri=org["frac_aristas_multi_tri"],
                tri_por_arista_media=org["tri_por_arista_media"],
                tri_por_arista_max=carga["tri_por_arista_max"],
                frac_aristas_en_triangulo=org["frac_aristas_en_triangulo"],
                gini_tri_nodo=org["gini_tri_nodo"], modularidad_tri=org["modularidad_tri"],
                masa_total_ic=info["masa_total"], carpeta=carpeta,
                grafo_guardado="grafo_f801.npz",
            )
            with open(f"{carpeta}/meta_regla.json", "w") as f:
                json.dump(meta, f, indent=2)
            fila["carpeta"] = carpeta

        filas.append(fila)
        print(f"    {modo:>7}  tri={ntri:<5} A={org['tri_por_arista_media']:.3f} "
              f"D={org['frac_aristas_en_triangulo']:.4f} multi={org['frac_aristas_multi_tri']:.3f} "
              f"max_e={carga['tri_por_arista_max']:.0f} gini_n={org['gini_tri_nodo']:.3f} "
              f"C={cl:.5f} nodos={org['frac_nodos_en_triangulo']:.3f} gig={fila['gigante']} "
              f"pend={pc['pendiente']:.4f}" + (f" ic={fila.get('t_ic_s')}s" if generar_ic else ""),
              flush=True)

    for f in filas:
        f["t_total_grafo_s"] = round(time.time() - t_ini, 1)
    return filas


def main(indices=None, generar_ic=True, sufijo_csv="", n_por_lote=2):
    elegidos = F702.seleccionar_grafos(n_por_lote=n_por_lote)   # mismos grafos base de F7-02/F7-03
    if indices is not None:
        elegidos = [elegidos[i] for i in indices]
    print(f"[f801] {len(elegidos)} grafos base (generar_ic={generar_ic}) brazos={BRAZOS}", flush=True)

    todas, t0 = [], time.time()
    for k, sel in enumerate(elegidos):
        print(f"[{k+1}/{len(elegidos)}] {sel['rule_id']} seed={sel['seed']} lote={sel['lote']}",
              flush=True)
        todas.extend(procesar_una(sel, generar_ic=generar_ic))

    ruta = RUTA_ESTRUCTURA.replace(".csv", f"{sufijo_csv}.csv")
    with open(ruta, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(todas[0].keys()))
        w.writeheader()
        w.writerows(todas)
    print(f"\n[f801] {len(todas)} filas -> {ruta.split('/')[-1]}  (total {time.time()-t0:.0f}s)")
    return todas


if __name__ == "__main__":
    # uso: python3.9 cs090_fase8_f801_desacople.py [idx0,idx1,...] [--sin-ic] [--sufijo=_s0] [--n-lote=2]
    idxs, gen_ic, suf, npl = None, True, "", 2
    for arg in sys.argv[1:]:
        if arg == "--sin-ic":
            gen_ic = False
        elif arg.startswith("--sufijo="):
            suf = arg.split("=", 1)[1]
        elif arg.startswith("--n-lote="):
            npl = int(arg.split("=", 1)[1])
        else:
            idxs = [int(x) for x in arg.split(",")]
    main(indices=idxs, generar_ic=gen_ic, sufijo_csv=suf, n_por_lote=npl)
