"""
CS071 — Histéresis / memoria-de-enlace: ¿la asimetría que fabrica el PROCESO auto-organiza métrica?
==============================================================================
Diseño/ruling: CS (DISENO_CS071_histeresis_memoria_enlace_CS.md, 17-jul-2026). Batería topológica Gemini,
Test 2.2. El arco (CS066-070) inyectó asimetría DESDE FUERA (estructura, fase, semilla) y el mundo-pequeño
la lavó. CS071 prueba la única fuente que las anteriores no tocaron: la que fabrica el PROPIO proceso
dinámico -- no hay semilla ni estructura privilegiada al inicio (todo enlace vale igual); si hay asimetría,
nace de que TRANSITAR un enlace lo refuerza y NO-usarlo lo hace decaer.

MECANISMO (homeostático -- escalado sináptico, presupuesto por nodo, ciego a geometría):
- Sustrato: Watts-Strogatz (k=6, p=0.1, el mismo usado en las auditorías previas del arco), N∈{400,900,1600}.
  Enlaces uniformes (w_ij=1.0 al inicio).
- Caminantes (N_WALKERS=N, PASOS_POR_CAMINANTE=5 por macro-paso): en cada macro-paso, cada caminante da 5
  pasos, en cada uno elige el SIGUIENTE nodo entre los vecinos ACTUALES con probabilidad ∝ w_ij (el
  reforzado se prefiere). Cada enlace transitado se refuerza: w_ij *= (1+REFUERZO).
- Decaimiento: TODOS los enlaces decaen w_ij *= DECAY cada macro-paso (lo no usado se diluye relativo a lo
  usado).
- Homeostasis (escalado sináptico): cada nodo i tiene un presupuesto FIJO = su grado ORIGINAL deg0_i; tras
  refuerzo+decay, los pesos de sus enlaces se reescalan para que Σ_j w_ij vuelva a deg0_i. Como w_ij es
  compartido por sus dos extremos, se aplica el factor GEOMÉTRICO de ambos lados (normalización simétrica).
- Poda: un enlace se elimina si, tras homeostasis, w_ij < PRUNE_FRAC × (deg0_i/actual_deg_i) en CUALQUIERA
  de sus dos extremos (cayó muy por debajo de su "parte justa" en ese nodo).

G-PASEO-CIEGO: la función de transición del caminante usa SOLO w_ij actual (y por tanto los grados
derivados) -- NUNCA coordenada, distancia de anillo, ni ninguna etiqueta que codifique la geometría-objetivo.
Auditable: `_elige_vecino` no recibe posición.

Codea/ejecuta: CC. Diseño/ruling: CS.
"""
from __future__ import annotations
import os, sys, time, math
from collections import deque
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs068_paso1_sintetico as S1  # retícula 2D limpia (4-vecinos), reusada para HISTERESIS_SOBRE_RETICULA

RNG = np.random.default_rng

# ------------ cronograma FIJO, decidido antes de ver diámetros (G-NO-AJUSTAR-CRONOGRAMA) ------------
WS_K, WS_P = 6, 0.1
PASOS_PROCESO = 30
PASOS_POR_CAMINANTE = 5
# REFUERZO/DECAY/PRUNE_FRAC: corrida EXPLORATORIA declarada (CC, antes de la tanda de veredicto, G-NO-
# AJUSTAR-CRONOGRAMA respetado -- el criterio de selección fue evitar DEGENERACIÓN de conectividad, nunca
# "acercarse a √N"). Los valores iniciales (0.15/0.95/0.30, calcados de la prosa del diseño) colapsaban el
# grafo casi por completo (frac_gigante~0.01, retroalimentación fuera de control: ver _homeostasis_y_poda).
# Barrido sobre {REFUERZO, DECAY, PRUNE_FRAC} buscando un régimen SIN colapso catastrófico en NINGUNO de
# los dos sustratos (WS denso grado~6, retícula grado~4 -- la retícula, con menos redundancia, resultó
# mucho más frágil al mismo refuerzo). REFUERZO=0.04/DECAY=0.99/PRUNE_FRAC=0.15 es el punto encontrado:
# ambos sustratos quedan mayormente conectados (frac_gigante 0.80-1.00) y AMBOS muestran poda real (no es
# un régimen inerte que reduzca trivialmente a SIN_PROCESO).
REFUERZO = 0.04           # w_ij *= (1+REFUERZO) por tránsito
DECAY = 0.99               # w_ij *= DECAY cada macro-paso, para TODOS los enlaces
PRUNE_FRAC = 0.15          # poda si w_ij (tras homeostasis) < PRUNE_FRAC (fijo, relativo al peso original 1.0)


# ============================ sustratos ============================
def _ws_graph(N, k, p, rng):
    """Watts-Strogatz clásico: anillo con k vecinos más cercanos (k par), luego cada enlace se re-cablea
    con probabilidad p a un destino al azar (sin self-loop ni duplicado)."""
    adj = [set() for _ in range(N)]
    for i in range(N):
        for d in range(1, k // 2 + 1):
            j = (i + d) % N
            adj[i].add(j); adj[j].add(i)
    edges = [(i, j) for i in range(N) for j in adj[i] if i < j]
    for (i, j) in edges:
        if rng.random() < p:
            tentativas = 0
            while tentativas < 20:
                tentativas += 1
                nuevo = int(rng.integers(0, N))
                if nuevo != i and nuevo not in adj[i]:
                    adj[i].discard(j); adj[j].discard(i)
                    adj[i].add(nuevo); adj[nuevo].add(i)
                    break
    return adj


def _reticula_control(N_objetivo):
    side = int(round(N_objetivo ** 0.5))
    adj, N = S1._reticula_2d(side)
    return adj, N


# ============================ el proceso de histéresis (homeostático) ============================
def _pesos_iniciales(adj, N):
    w = {}
    for i in range(N):
        for j in adj[i]:
            if i < j:
                w[(i, j)] = 1.0
    return w


def _vecinos_con_peso(adj, w, i):
    """Lista (vecino, peso) actual de i -- SOLO topología+peso, nunca coordenada (G-PASEO-CIEGO)."""
    out = []
    for j in adj[i]:
        e = (i, j) if i < j else (j, i)
        out.append((j, w.get(e, 0.0)))
    return out


def _elige_vecino(adj, w, i, rng):
    """Probabilidad ∝ w_ij actual. G-PASEO-CIEGO: no recibe posición ni distancia de anillo."""
    vs = _vecinos_con_peso(adj, w, i)
    if not vs:
        return None
    pesos = np.array([p for _, p in vs], float)
    tot = pesos.sum()
    if tot <= 0:
        return vs[int(rng.integers(0, len(vs)))][0]
    r = rng.random() * tot
    acc = 0.0
    for (j, p) in vs:
        acc += p
        if acc >= r:
            return j
    return vs[-1][0]


def _homeostasis_y_poda(adj, w, N, deg0):
    """Escalado sináptico: cada nodo i reescala sus enlaces para que Σ_j w_ij == deg0_i (su presupuesto
    ORIGINAL, fijo). w_ij es compartido -> se aplica el factor GEOMÉTRICO de ambos extremos (normalización
    simétrica, no privilegia ningún lado).

    CORRECCIÓN encontrada al smoke-testear (no estaba en el diseño de CS, es un detalle de implementación):
    comparar contra deg0_i/deg_actual_i (el "justo" recalculado sobre el grado YA reducido) crea una
    retroalimentación fuera de control -- cada poda sube el umbral para los enlaces que quedan en ese nodo
    (mismo presupuesto FIJO repartido entre menos enlaces => "justo" más alto => más caen debajo => más
    podas), y en pocos macro-pasos colapsa casi todo el grafo (grado medio de 6 a <1 en el smoke). El umbral
    correcto es FIJO respecto al peso ORIGINAL (1.0, con el que every enlace arrancó): poda si w_ij (tras
    homeostasis) < PRUNE_FRAC. Sin cascada -- el umbral no se mueve porque el grafo se adelgace."""
    suma = np.zeros(N)
    for (i, j), wij in w.items():
        suma[i] += wij; suma[j] += wij
    factor = np.ones(N)
    for i in range(N):
        if suma[i] > 1e-9:
            factor[i] = deg0[i] / suma[i]
    w2 = {}
    a_podar = []
    for (i, j), wij in w.items():
        wij2 = wij * math.sqrt(max(factor[i], 1e-9) * max(factor[j], 1e-9))
        if wij2 < PRUNE_FRAC:
            a_podar.append((i, j))
        else:
            w2[(i, j)] = wij2
    for (i, j) in a_podar:
        adj[i].discard(j); adj[j].discard(i)
    return w2


def _proceso_histeresis(adj0, N, rng, barajado=False, pasos=PASOS_PROCESO):
    """HISTERESIS real (barajado=False): N caminantes dan PASOS_POR_CAMINANTE pasos por macro-paso,
    prefiriendo enlaces de mayor peso (G-PASEO-CIEGO). NULL_BARAJADO (barajado=True): el MISMO número de
    'toques' de enlace por macro-paso (N*PASOS_POR_CAMINANTE), pero elegidos AL AZAR entre los enlaces
    VIVOS -- misma magnitud de refuerzo, sin correlación con el tráfico real (G-NULL-MISMA-MAGNITUD).
    Decay+homeostasis+poda son IDÉNTICOS en ambos brazos (G-NO-AJUSTAR-CRONOGRAMA)."""
    adj = [set(a) for a in adj0]
    deg0 = np.array([len(a) for a in adj], float)
    w = _pesos_iniciales(adj, N)
    n_toques = N * PASOS_POR_CAMINANTE
    for _ in range(pasos):
        if not barajado:
            pos = rng.integers(0, N, size=N)
            for _paso_c in range(PASOS_POR_CAMINANTE):
                nuevas_pos = pos.copy()
                for idx in range(N):
                    i = int(pos[idx])
                    if not adj[i]:
                        continue
                    j = _elige_vecino(adj, w, i, rng)
                    if j is None:
                        continue
                    e = (i, j) if i < j else (j, i)
                    if e in w:
                        w[e] *= (1 + REFUERZO)
                    nuevas_pos[idx] = j
                pos = nuevas_pos
        else:
            edges_vivos = list(w.keys())
            if edges_vivos:
                idxs = rng.integers(0, len(edges_vivos), size=n_toques)
                for t in idxs:
                    e = edges_vivos[int(t)]
                    if e in w:
                        w[e] *= (1 + REFUERZO)
        for e in list(w.keys()):
            w[e] *= DECAY
        w = _homeostasis_y_poda(adj, w, N, deg0)
    return adj, w


# ============================ jueces ============================
def _diam_robusto(adj, N, rng, n_src=8):
    eccs = []
    for s in rng.integers(0, N, size=min(n_src, N)):
        s = int(s)
        if not adj[s]:
            continue
        dist = {s: 0}; q = deque([s]); far = 0
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1; far = max(far, dist[v]); q.append(v)
        if len(dist) > 0.3 * N:
            eccs.append(far)
    return float(np.median(eccs)) if eccs else float("nan")


def _frac_gigante(adj, N, rng, n_src=8):
    mejor = 0
    for s in rng.integers(0, N, size=min(n_src, N)):
        s = int(s)
        if not adj[s]:
            continue
        dist = {s: 0}; q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1; q.append(v)
        mejor = max(mejor, len(dist))
    return mejor / N


def _bfs_completo(adj, N, fuente):
    dist = {fuente: 0}; q = deque([fuente])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1; q.append(v)
    return dist


def _delta_gromov(adj, N, rng, n_landmarks=50, n_quad=300):
    """δ-Gromov muestreado sobre un conjunto de landmarks (BFS desde cada uno, matriz de distancias entre
    landmarks), para no pagar BFS todos-contra-todos. δ(a,b,c,d) = (S1-S2)/2 sobre las tres sumas de pares
    opuestos, S1>=S2>=S3. Devuelve la MEDIANA sobre n_quad cuádruplas muestreadas de los landmarks."""
    cand = [i for i in range(N) if adj[i]]
    if len(cand) < n_landmarks:
        n_landmarks = len(cand)
    if n_landmarks < 4:
        return float("nan")
    landmarks = rng.choice(cand, size=n_landmarks, replace=False)
    D = np.full((n_landmarks, n_landmarks), np.nan)
    idx_de = {int(l): k for k, l in enumerate(landmarks)}
    for k, l in enumerate(landmarks):
        dist = _bfs_completo(adj, N, int(l))
        for k2, l2 in enumerate(landmarks):
            if int(l2) in dist:
                D[k, k2] = dist[int(l2)]
    deltas = []
    for _ in range(n_quad):
        a, b, c, d = rng.choice(n_landmarks, size=4, replace=False)
        dab, dcd = D[a, b], D[c, d]
        dac, dbd = D[a, c], D[b, d]
        dad, dbc = D[a, d], D[b, c]
        if not (np.isfinite(dab) and np.isfinite(dcd) and np.isfinite(dac) and np.isfinite(dbd)
                and np.isfinite(dad) and np.isfinite(dbc)):
            continue
        s1 = dab + dcd; s2 = dac + dbd; s3 = dad + dbc
        s = sorted([s1, s2, s3], reverse=True)
        deltas.append((s[0] - s[1]) / 2.0)
    return float(np.median(deltas)) if deltas else float("nan")


def _grado_max_y_medio(adj, N):
    grados = [len(a) for a in adj]
    return max(grados) if grados else 0, float(np.mean(grados)) if grados else 0.0


# ============================ LOS 4 BRAZOS ============================
def brazo_histeresis(N, seed):
    rng = RNG(seed)
    adj0 = _ws_graph(N, WS_K, WS_P, RNG(seed + 1))
    adj, w = _proceso_histeresis(adj0, N, RNG(seed + 2), barajado=False)
    return adj


def brazo_null_barajado(N, seed):
    rng = RNG(seed)
    adj0 = _ws_graph(N, WS_K, WS_P, RNG(seed + 1))
    adj, w = _proceso_histeresis(adj0, N, RNG(seed + 2), barajado=True)
    return adj


def brazo_sin_proceso(N, seed):
    return _ws_graph(N, WS_K, WS_P, RNG(seed + 1))


def brazo_histeresis_sobre_reticula(N, seed):
    adj0, N2 = _reticula_control(N)
    assert N2 == N, f"N={N} no es cuadrado perfecto exacto (side²={N2}) -- usar solo N∈{{400,900,1600}}"
    adj, w = _proceso_histeresis(adj0, N2, RNG(seed + 2), barajado=False)
    return adj


BRAZOS = dict(histeresis=brazo_histeresis, null_barajado=brazo_null_barajado,
              sin_proceso=brazo_sin_proceso, histeresis_sobre_reticula=brazo_histeresis_sobre_reticula)


def evalua(adj, N, seed, deg0_inicial_medio=None):
    rng = RNG(seed + 3)
    diam = _diam_robusto(adj, N, rng)
    frac_gig = _frac_gigante(adj, N, RNG(seed + 4))
    delta_g = _delta_gromov(adj, N, RNG(seed + 5))
    gmax, gmedio = _grado_max_y_medio(adj, N)
    return dict(diam=diam, frac_gigante=frac_gig, delta_gromov=delta_g, grado_max=gmax, grado_medio=gmedio)


if __name__ == "__main__":
    print("cs071_histeresis.py -- módulo de mecanismo. Correr cs071_smoke.py o cs071_tanda.py.")
