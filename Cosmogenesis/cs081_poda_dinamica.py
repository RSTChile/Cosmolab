"""
CS081 — Experimento 2 de Fase III: PODA DINÁMICA POR COSTO DE ENLACE (protocolo pre-registrado)
==================================================================================================
Este script ejecuta, TAL CUAL quedó diseñado en la sección 4 de FASE3_renormalizacion_resultado_CS.md
(pre-registrado antes de correr nada), el Experimento 2 de la Fase III. NO se rediseña el protocolo
acá — sólo se implementa.

PREGUNTA: en el Experimento 1 (cs080_renormalizacion.py, YA CERRADO) el tejido de CS066 (brazo
"local", k_local=6 fijo) resultó mundo-pequeño a TODAS las escalas de agrupamiento -- ninguna
condensación macroscópica. Ese resultado usó el tejido TAL CUAL lo deja proceso066. La pregunta de
HOY es distinta: ¿puede el propio SISTEMA, sin que nadie externo le diga "esto es un atajo, córtalo"
(ese criterio externo YA se probó y falló en CS068 Paso 1), descubrir un "lejos" macroscópico si cada
enlace paga un COSTO por lo que realmente aporta, y se podan los más caros?

SUSTRATO: el MISMO que el Experimento 1 -- motor de CS066 (proceso066), brazo "local", k_local=6
FIJO, N=8000, MISMAS 3 semillas (80100, 80200, 80300) -- para que los dos experimentos de Fase III
sean directamente comparables, manzana con manzana. No se toca cs066_localidad_geometrogenesis.py
NI cs080_renormalizacion.py -- ambos se usan sólo por import. El coarse-graining y las métricas de
juicio (cajas_bfs, grafo_grueso, propagar_spins, metricas_escala) se REUSAN tal cual de cs080 (mismo
método de "cajas" BFS declarado ahí, para que Exp1 y Exp2 midan con la MISMA vara).

COSTO POR ENLACE (4 componentes, pesos iguales 1/4, pre-registrado ANTES de correr, ver §4 del .md):
  1. Inconsistencia histórica -- cuántas veces el enlace aparece/desaparece durante los 20 pasos de
     proceso066. Requiere instrumentación NUEVA (no existía en cs066): `proceso066_instrumentado` de
     este archivo es una réplica del bucle de proceso066 -- llama a las MISMAS funciones importadas
     (C62._grav_peso, C7._confin/_em/_debil/_despliegue, gate_localidad, SM.alinear_nematico_fast) EN
     EL MISMO ORDEN, línea por línea -- y sólo AGREGA una foto del conjunto de aristas en cada paso
     para contar transiciones presente/ausente por arista. No se modifica ni se llama distinto ningún
     paso físico. (Ver nota al final de esta sección: se verificó que reproduce bit-a-bit el sustrato
     de cs080 con la misma semilla.)
  2. Conflicto de holonomía -- se usa C9._ciclos_fundamentales y C9._holonomia_ciclo (ya existen, sin
     tocar) sobre los espines finales; el costo del enlace es el promedio de la holonomía de todos los
     ciclos fundamentales muestreados que pasan por él (arista sin ciclo asignado -> se le da la media
     global, neutral, no se inventa señal donde no la hay).
  3. Baja contribución a persistencia (soporte local) -- nº de vecinos comunes de (i,j), EXACTAMENTE
     el mismo cálculo que ya vive adentro de gate_localidad (cs066), copiado como expresión de una
     línea, no reimplementado como función aparte. Costo = -soporte (menos vecinos comunes = más caro).
  4. Baja reciprocidad -- gate_localidad hoy usa OR ("sobrevive si algún extremo lo elige entre sus
     k_local más locales"); acá se recalcula, con el MISMO criterio de ranking por soporte que usa
     gate_localidad, qué eligió cada nodo, y se marca costo=1 si sólo un extremo la eligió, costo=0 si
     ambos (recíproca).
  Costo total = promedio de los z-scores de los 4 componentes (sobre TODAS las aristas del tejido en
  ese momento), pesos iguales. Fijado ANTES de mirar ningún resultado de poda.

POSA: se podan las aristas con costo total por ENCIMA del percentil P, para P en {50, 70, 90} (barrido,
no un solo valor, para no repetir el error ya cazado en CS068 Paso 1 de "un umbral externo arbitrario").

CONTROL NULL: mismo número exacto de aristas podadas, elegidas AL AZAR (no por costo), mismo P, mismo
tejido base -- aísla si importa QUÉ se corta (costo) o sólo CUÁNTO se corta (densidad).

MÉTRICA DE VEREDICTO: igual que el Experimento 1 -- pendiente log-log de diam(b) vs N_b(b) bajo el
MISMO coarse-graining (b=2,4,8,16,32 + b=1 sin agrupar), comparando poda-por-costo vs poda-aleatoria-
misma-cantidad vs sin-poda, en las MISMAS 3 semillas. Aprendizaje directo del Experimento 1: NO se
compara contra el umbral fijo 0.3 de CS068 (no discriminó bajo coarse-graining, ver Exp1 sección 3) --
se comparan las pendientes DIRECTAMENTE entre sí.

FALSACIÓN PRE-REGISTRADA (§4 del .md, verbatim):
  - poda-por-costo ≈ poda-aleatoria (misma pendiente dentro del rango de semillas) -> NEGATIVO: el
    costo elegido no descubre nada que el azar con la misma densidad no logre igual (resultado
    honesto, no fracaso).
  - poda-por-costo > poda-aleatoria de forma consistente en las 3 semillas -> hay algo que el costo
    captura que el azar no -- vale la pena escalar a N grande.

Codea/ejecuta: CC (Claude). No se declara cierre ni veredicto de arco -- se reportan números, la
lectura final es de Alexis.
"""
from __future__ import annotations
import os, sys, time, csv, math
from collections import deque, defaultdict, Counter
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

import cs057_paisaje_completo as C7          # _diam, _giant, _confin, _em, _debil, _despliegue, _T_de_paso
import cs059_espin_como_marco as C9          # _spins, _ciclos_fundamentales, _holonomia_ciclo, _frame_burgers
import cs062_paisaje_peso as C62             # _grav_peso
import cg003_diagnostico_gromov as GR        # aleatorio() -- generador de grafo caliente inicial (igual que CS066)
import cs064_smoke as SM                     # dim_volumen, adj_sparse, alinear_nematico_fast
import cs064_sistema_completo as C64         # _cataloga, DMAX_INT
import cs066_localidad_geometrogenesis as C66  # gate_localidad (reusada TAL CUAL) -- NO se toca el archivo
import cs080_renormalizacion as C80          # cajas_bfs, grafo_grueso, propagar_spins, metricas_escala -- NO se toca

RNG = np.random.default_rng
DMAX_INT = C64.DMAX_INT

N_NODOS   = int(os.environ.get("CS081_N", 8000))          # mismo N que Exp1
K_LOCAL   = int(os.environ.get("CS081_KLOC", 6))           # mismo k_local que Exp1
STEPS     = int(os.environ.get("CS081_STEPS", C66.STEPS))  # mismos 20 pasos que proceso066
ESCALAS_B = [int(x) for x in os.environ.get("CS081_B", "2,4,8,16,32").split(",")]  # mismas escalas que Exp1
SEEDS     = [int(x) for x in os.environ.get("CS081_SEEDS", "80100,80200,80300").split(",")]  # MISMAS 3 semillas
PERCENTILES = [int(x) for x in os.environ.get("CS081_P", "50,70,90").split(",")]
OUT       = os.environ.get("CS081_OUT", os.path.join(_HERE, "cs081_poda_dinamica.csv"))


# ======================================================================================
# 1) SUSTRATO INSTRUMENTADO: réplica exacta de C66.proceso066(arm="local"), + historial de aristas
# ======================================================================================
def proceso066_instrumentado(N, cat, k_local, rng):
    """Idéntico a C66.proceso066 con arm='local' (gate_on=True, barajado=False, marco_vivo=True) --
    MISMAS llamadas, MISMO orden, MISMOS parámetros -- salvo que, además, en cada paso fotografía el
    conjunto de aristas y cuenta cuántas veces cada arista CAMBIA de estado (aparece/desaparece) desde
    el paso anterior. Ese conteo es el componente 1 del costo ("inconsistencia histórica"), pedido en
    el protocolo pre-registrado y que proceso066 original no expone (sólo agrega D/G por paso)."""
    fam, color, carga, masa, es_anti = cat["fam"], cat["color"], cat["carga"], cat["masa"], cat["es_anti"]

    adj0, _ = GR.aleatorio(N, meandeg=6.0, seed=int(rng.integers(1 << 30)))
    adj = [set(a) for a in adj0]
    col = np.where(color >= 0, color, rng.integers(0, 3, N)).astype(np.int8)
    car = (carga > 0).astype(np.int8)
    deg0 = [len(a) for a in adj]
    t = np.zeros(N, dtype=np.int32)
    V = C9._spins(N, DMAX_INT, rng)
    R = getattr(C7, "R_GRAV", 1.0); T_CONF = getattr(C7, "T_CONF", 0.5); CAP_E = 12 * N
    D, G = [], []

    flip_count = Counter()
    prev_edges = set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i])

    def _foto_y_cuenta():
        cur = set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i])
        for e in prev_edges ^ cur:      # diferencia simétrica = aristas que cambiaron de estado
            flip_count[e] += 1
        prev_edges.clear(); prev_edges.update(cur)

    for step in range(STEPS):
        T = C7._T_de_paso(step, 1.0) if hasattr(C7, "_T_de_paso") else max(0.02, 1.6 * (1 - step / STEPS))
        E = sum(len(a) for a in adj) // 2
        if E > CAP_E:
            break
        w = masa * max(0.0, 1.0 - T); CAP = max(1, N // 4)
        try:
            rg = min(0.30 * R * (1 - T), CAP / max(1.0, (0.5 + T) * E))
            C62._grav_peso(adj, N, rng, rg, dmax=3, T=T, w=w + 1e-9)
        except Exception: pass
        if T < T_CONF:
            try: C7._confin(adj, N, col, t, rng, min(0.8, CAP / max(1.0, E)))
            except Exception: pass
        try: C7._em(adj, N, car, deg0, rng, 0.12)
        except Exception: pass
        try: C7._debil(N, col, car, rng, 0.05)
        except Exception: pass
        try: C7._despliegue(adj, N, rng, 0.14 * T)
        except Exception: pass
        if T > 0.4 and step % 2 == 0:
            for i in rng.choice(N, size=max(1, N // 60), replace=False):
                if adj[i] and es_anti[i] != es_anti[list(adj[i])[0]]:
                    j = list(adj[i])[0]; adj[i].discard(j); adj[j].discard(i)
        # EL ACTOR DE CS066: localidad en la persistencia (gate_localidad, reusada TAL CUAL, sin tocar)
        C66.gate_localidad(adj, N, rng, k_local, barajado=False)
        # co-evolución del marco nemático (marco_vivo=True, como arm="local")
        A = SM.adj_sparse(adj, N); deg = np.asarray(A.sum(axis=1)).ravel()
        V = SM.alinear_nematico_fast(V, A, deg, mezcla=0.35)
        D.append(C7._diam(adj, N)); G.append(C7._giant(adj, N))
        _foto_y_cuenta()   # <<< ÚNICO agregado sobre proceso066 original: instrumentación de historial

    return adj, V, flip_count


# ======================================================================================
# 2) COSTO POR ARISTA (4 componentes, z-score, pesos iguales -- pre-registrado)
# ======================================================================================
def costo_por_arista(adj, N, V, flip_count, k_local, rng):
    edges = [(i, j) if i < j else (j, i) for i in range(N) for j in adj[i] if j > i or j < i]
    edges = sorted(set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i]))

    # (3) soporte local -- MISMO cálculo de una línea que usa gate_localidad internamente
    soporte = {e: len(adj[e[0]] & adj[e[1]]) for e in edges}

    # (4) reciprocidad -- por nodo, MISMO ranking por soporte que gate_localidad para decidir "elegido"
    elegido = [set() for _ in range(N)]
    for i in range(N):
        nb = list(adj[i])
        if len(nb) <= k_local:
            elegido[i] = set(nb)
        else:
            sup = sorted(((len(adj[i] & adj[j]), j) for j in nb), reverse=True)
            elegido[i] = set(j for _, j in sup[:k_local])
    recip_cost = {e: (0.0 if (e[1] in elegido[e[0]] and e[0] in elegido[e[1]]) else 1.0) for e in edges}

    # (2) conflicto de holonomía -- ciclos fundamentales ya existentes (C9), sin tocar
    K = min(4, V.shape[1])
    spins = V[:, :K] / (np.linalg.norm(V[:, :K], axis=1, keepdims=True) + 1e-12)
    ciclos = C9._ciclos_fundamentales(adj, N, C9.N_CICLOS, rng)
    w0 = np.zeros(K); w0[0] = 1.0
    hol_por_arista = defaultdict(list)
    for ciclo in ciclos:
        h = C9._holonomia_ciclo(spins, ciclo, w0)
        L = len(ciclo)
        for k in range(L):
            a, b = ciclo[k], ciclo[(k + 1) % L]
            e = (a, b) if a < b else (b, a)
            hol_por_arista[e].append(h)
    todas_hol = [h for vs in hol_por_arista.values() for h in vs]
    media_global_hol = float(np.mean(todas_hol)) if todas_hol else 0.0
    hol_cost = {e: (float(np.mean(hol_por_arista[e])) if e in hol_por_arista else media_global_hol) for e in edges}

    # (1) inconsistencia histórica -- del proceso066_instrumentado; arista nunca vista en flip_count = 0 (estable)
    hist_cost = {e: float(flip_count.get(e, 0)) for e in edges}

    def z(d):
        arr = np.array([d[e] for e in edges], dtype=float)
        mu, sd = arr.mean(), arr.std()
        return (arr - mu) / sd if sd > 1e-9 else np.zeros_like(arr)

    z_hist, z_hol, z_sup, z_rec = z(hist_cost), z(hol_cost), z({e: -soporte[e] for e in edges}), z(recip_cost)
    total = 0.25 * (z_hist + z_hol + z_sup + z_rec)
    costo = dict(zip(edges, total))
    componentes = dict(hist=hist_cost, holonomia=hol_cost, soporte=soporte, reciprocidad=recip_cost)
    return edges, costo, componentes


# ======================================================================================
# 3) PODA: por costo (top percentil P) vs aleatoria (mismo nº de aristas)
# ======================================================================================
def podar_por_costo(adj, N, edges, costo, P):
    umbral = np.percentile([costo[e] for e in edges], P)
    a_podar = [e for e in edges if costo[e] > umbral]
    na = [set(a) for a in adj]
    for (i, j) in a_podar:
        na[i].discard(j); na[j].discard(i)
    return na, len(a_podar)

def podar_aleatorio(adj, N, edges, n_podar, rng):
    idx = rng.choice(len(edges), size=min(n_podar, len(edges)), replace=False)
    a_podar = [edges[t] for t in idx]
    na = [set(a) for a in adj]
    for (i, j) in a_podar:
        na[i].discard(j); na[j].discard(i)
    return na


# ======================================================================================
# 4) CORRIDA COMPLETA: 1 semilla -> sustrato + costo + 7 variantes (sin-poda, cost-P50/70/90, rand-P50/70/90)
#    x coarse-graining b=1,2,4,8,16,32 (funciones REUSADAS de cs080, sin tocarlo)
# ======================================================================================
def corre_semilla(seed):
    filas = []
    rng = RNG(seed)
    cat = C64._cataloga(N_NODOS, rng)
    r2 = RNG(seed * 137 + hash("local") % 9973 + 5)
    t0 = time.time()
    adj, V, flip_count = proceso066_instrumentado(N_NODOS, cat, K_LOCAL, r2)
    n_edges0 = sum(len(a) for a in adj) // 2
    print(f"  sustrato local construido: N={N_NODOS} aristas={n_edges0} ({time.time()-t0:.1f}s)", flush=True)

    rng_costo = RNG(seed * 911 + 3)
    edges, costo, componentes = costo_por_arista(adj, N_NODOS, V, flip_count, K_LOCAL, rng_costo)
    print(f"  costo calculado sobre {len(edges)} aristas ({time.time()-t0:.1f}s acumulado)", flush=True)

    variantes = {"sin_poda": (adj, 0)}
    for P in PERCENTILES:
        na, n_pod = podar_por_costo(adj, N_NODOS, edges, costo, P)
        variantes[f"costo_P{P}"] = (na, n_pod)
        rng_rand = RNG(seed * 733 + P)
        na_r = podar_aleatorio(adj, N_NODOS, edges, n_pod, rng_rand)
        variantes[f"azar_P{P}"] = (na_r, n_pod)

    for nombre, (adj_v, n_podadas) in variantes.items():
        tb = time.time()
        rng_m = RNG(seed * 991 + hash(nombre) % 7919)
        m1 = C80.metricas_escala(adj_v, N_NODOS, V, rng_m)
        m1.update(dict(seed=seed, variante=nombre, n_podadas=n_podadas, b=1, n_cajas=N_NODOS))
        filas.append(m1)
        for b in ESCALAS_B:
            rng_b = RNG(seed * 733 + b * 31 + hash(nombre) % 4999)
            asign, n_cajas = C80.cajas_bfs(adj_v, N_NODOS, b, rng_b)
            adj_g = C80.grafo_grueso(adj_v, N_NODOS, asign, n_cajas)
            Vg = C80.propagar_spins(V, asign, n_cajas, V.shape[1])
            rng_m2 = RNG(seed * 1291 + b * 53 + hash(nombre) % 3571)
            mb = C80.metricas_escala(adj_g, n_cajas, Vg, rng_m2)
            mb.update(dict(seed=seed, variante=nombre, n_podadas=n_podadas, b=b, n_cajas=n_cajas))
            filas.append(mb)
        print(f"  [{nombre:<10}] podadas={n_podadas:<6} listo ({time.time()-tb:.1f}s)", flush=True)
    return filas


def _campos():
    return ["seed", "variante", "n_podadas", "b", "n_cajas", "N_b", "diam", "giant", "d_s", "holonomia", "n_ciclos"]


def main():
    print("=" * 100, flush=True)
    print("CS081 — Experimento 2 de Fase III: PODA DINÁMICA POR COSTO DE ENLACE (protocolo pre-registrado)", flush=True)
    print(f"N={N_NODOS}  k_local={K_LOCAL}  steps={STEPS}  escalas b={ESCALAS_B}  semillas={SEEDS}  "
          f"percentiles P={PERCENTILES}", flush=True)
    print("variantes por semilla: sin_poda | costo_P{50,70,90} | azar_P{50,70,90} (mismo nº de aristas que costo_P)",
          flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    fout = open(OUT, "w", newline="")
    wr = csv.DictWriter(fout, fieldnames=_campos())
    wr.writeheader()
    for seed in SEEDS:
        print(f"\n--- semilla {seed} ---", flush=True)
        filas = corre_semilla(seed)
        for f in filas:
            wr.writerow({k: f[k] for k in _campos()})
        fout.flush()
        print(f"  (acumulado {(time.time()-t0)/60:.1f} min)", flush=True)
    fout.close()
    print(f"\nCOMPLETO en {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
