"""
CG004-f ETAPA 2 — BARRIDO DE CURVATURA: cortar costura + re-pegar por holonomía afín de lazo
=============================================================================================
Etapa 1 (cg004f) validó la familia de sustratos {3,q}: q=6 plano (defic=0), q=7/8 hiperbólicos
(defic=π/3, 2π/3), métrica plano→hiperbólico, %gig=100. Derecho a Etapa 2 concedido.

Aquí ejecuto el test que adjudicó CS (adjudicacion_cg004e_CS.md §4): en cada κ (=q) cortar una
costura y re-pegar
   REGLA   = pega donde la HOLONOMÍA AFÍN DE LAZO ≈ 0 (criterio honesto, path-dependent, NO offset
             absoluto): desarrollo el grafo cortado sobre un árbol (giro-por-camino equilátero) y
             pego el par (a,b) cuyas posiciones DESARROLLADAS quedan adyacentes (el lazo a→bisagra→b
             cierra afínmente). En κ=0 esto recupera la retícula (=cg004e). En κ≠0 el lazo
             costura↔bisagra ENCIERRA curvatura → el desarrollo de b (por el camino largo) cae
             ROTADO/lejos → ni el par verdadero cierra → REGLA no recupera.
   CONTROL = pega la misma cantidad al azar.
Métrica CUANTITATIVA, no binaria: ¿a qué κ deja REGLA de restaurar la métrica original del sustrato?

PRE-VUELO (cuerda: el desarrollo por-camino es donde se escondía el bug de borde). Antes del
barrido AUTO-TESTEO el desarrollo sobre el sustrato PLANO intacto: debe cerrar con defdev≈0 en TODAS
las aristas (interior Y borde). Si no, la construcción está mal y NO se corre el barrido.

Cuerdas de CS: defdev≠0 es la señal (ya validado por q en Etapa 1); criterio = holonomía de LAZO;
guardar %gig (no confundir "no re-pega" con "sustrato fragmentado").

numpy-only. Reutiliza builders de cg004f.py y medición de cg004_attach.py.
"""
from __future__ import annotations

import os
import time
import math
import cmath
from collections import deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# --- builders de sustrato (Etapa 1) ---
_srcF = open(os.path.join(_HERE, "cg004f_barrido_curvatura.py")).read().replace("\nmain()\n", "\n")
_F = {"__file__": os.path.join(_HERE, "cg004f_barrido_curvatura.py")}
exec(compile(_srcF, "cg004f_barrido_curvatura.py", "exec"), _F)
construir = _F["construir"]; sphere_turnover = _F["sphere_turnover"]; _fin = _F["_fin"]
curvatura_discreta = _F["curvatura_discreta"]
su_inv = _F["su_inv"]; su_apply = _F["su_apply"]
# --- medición calibrada ---
diametro = _F["diametro"]; dimension_crecimiento = _F["dimension_crecimiento"]; diagnos = _F["diagnos"]


# ============================ CONFIG ============================
QS      = [6, 7, 8]        # knob de curvatura (6=plano, ≥7=hiperbólico)
TARGET_N = 3000
K        = 100
NEAR_TOL = 1.6            # REGLA pega si la posición desarrollada de b cae a <NEAR_TOL de a (adyacente)
SEEDS   = [1, 2, 3, 4]
# ===============================================================


def _centroid(pos):
    return sum(pos) / len(pos)


def _es_interior(adj, u):
    """Fan completo: cada par cíclico consecutivo es triángulo => n_triángulos == grado."""
    S = list(adj[u]); d = len(S)
    if d < 3:
        return False
    ntri = 0
    for i in range(d):
        ai = adj[S[i]]
        for j in range(i + 1, d):
            if S[j] in ai:
                ntri += 1
    return ntri == d


def _seed_interior(adj, pos, N):
    """Semilla del desarrollo = vértice INTERIOR (fan completo) más cercano al centroide.
    La raíz debe ser interior para que el marco inicial k·(π/3) sea geométricamente válido."""
    c = _centroid(pos)
    cand = [u for u in range(N) if _es_interior(adj, u)]
    if not cand:
        return max(range(N), key=lambda u: len(adj[u]))
    return min(cand, key=lambda u: abs(pos[u] - c))


def cortar_costura(adj, pos, N):
    """Costura vertical por el centroide, con DOS REMACHES en los EXTREMOS (adjudicación CS): dos
    puentes separados al máximo fijan posición + rotación del semiplano derecho (matan el modo cero
    de giro de la bisagra única). Corta el resto de los cruces.
    Devuelve (adj_cortado, cut_pairs, La, Ra, side, riv_bottom, riv_top, seed_dev)."""
    c = _centroid(pos)
    side = {u: (1 if (pos[u] - c).real >= 0 else -1) for u in range(N)}
    cruces = [(u, w) for u in range(N) for w in adj[u]
              if u < w and side[u] != side[w]]
    if len(cruces) < 3:
        return None
    # remaches = los dos cruces MÁS SEPARADOS a lo largo de la costura (menor y mayor Im)
    def imcross(e):
        return 0.5 * ((pos[e[0]] - c).imag + (pos[e[1]] - c).imag)
    riv_bottom = min(cruces, key=imcross)
    riv_top = max(cruces, key=imcross)
    remaches = {riv_bottom, riv_top}
    adjc = [set(s) for s in adj]
    cut_pairs = []; La = set(); Ra = set()
    for (u, w) in cruces:
        a, b = (u, w) if side[u] < 0 else (w, u)     # a=izq, b=der
        La.add(a); Ra.add(b)
        if (u, w) in remaches:
            continue                                  # remache: se conserva (puente)
        adjc[u].discard(w); adjc[w].discard(u)
        cut_pairs.append((a, b))
    def izqder(e):
        return (e[0], e[1]) if side[e[0]] < 0 else (e[1], e[0])
    seed_dev = _seed_interior(adjc, pos, N)
    return (adjc, cut_pairs, sorted(La), sorted(Ra), side,
            izqder(riv_bottom), izqder(riv_top), seed_dev)


def _chart_angle(u, w, pos, giso):
    """Ángulo del vecino w en la CARTA de u (u llevado al centro). En el disco de Poincaré
    (conforme) da el ángulo geodésico real en u; en el plano, el ángulo euclídeo."""
    if giso is None:
        z = pos[w] - pos[u]
    else:
        z = su_apply(su_inv(giso[u]), pos[w])
    return math.atan2(z.imag, z.real)


def _turn(orden_u, idx_u, adj, p, w):
    """Giro con signo de la arista (u→p) a la (u→w) en el desarrollo equilátero: CUENTA TRIÁNGULOS
    de p a w por el lado INTERIOR (cada par cíclico consecutivo que es triángulo = π/3), cortando al
    llegar al hueco exterior (par sin triángulo). Prueba CCW (+) y CW (−); usa el lado que alcanza w
    sin cruzar hueco. Geométricamente correcto en interior y borde; sin umbrales. None si no alcanza."""
    d = len(orden_u)
    if p not in idx_u or w not in idx_u or d == 0:
        return None
    ip = idx_u[p]
    for direction in (1, -1):
        ang = 0.0; i = ip
        for _ in range(d):
            ni = orden_u[i]; nj = orden_u[(i + direction) % d]
            if nj not in adj[ni]:                  # par sin triángulo = hueco exterior: corta
                break
            ang += direction * (math.pi / 3.0); i = (i + direction) % d
            if orden_u[i] == w:
                return ang
    return None


def desarrollar_arbol(adj, pos, giso, N, seed):
    """Desarrollo afín equilátero por giro-por-camino (hueco-exterior) sobre árbol BFS desde 'seed'
    (interior). Devuelve dev (N,2) y defdev (max cierre; ~0 si sustrato plano). Cruza la bisagra
    porque el giro de borde ya tiene signo correcto."""
    orden = [None] * N; ang = [None] * N; idx = [None] * N
    for u in range(N):
        if not adj[u]:
            orden[u] = []; ang[u] = {}; idx[u] = {}; continue
        a = {w: _chart_angle(u, w, pos, giso) for w in adj[u]}
        o = sorted(a, key=lambda w: a[w])
        orden[u] = o; ang[u] = a; idx[u] = {w: k for k, w in enumerate(o)}
    dev = np.full((N, 2), np.nan); dir_edge = {}
    dev[seed] = (0.0, 0.0)
    for k, w in enumerate(orden[seed]):            # raíz interior: marco k·60°
        dir_edge[(seed, w)] = k * (math.pi / 3.0)
    parent = {seed: None}; vis = {seed}; q = deque([seed])
    while q:
        u = q.popleft(); p = parent[u]
        for w in orden[u]:
            if w == p:                                   # arista de VUELTA = inversa exacta
                dir_edge[(u, w)] = dir_edge[(p, u)] + math.pi
                continue
            if (u, w) not in dir_edge:
                if p is None:
                    continue
                t = _turn(orden[u], idx[u], adj, p, w)
                if t is None:
                    continue
                dir_edge[(u, w)] = (dir_edge[(p, u)] + math.pi) + t
            if w not in vis and (u, w) in dir_edge:
                duw = dir_edge[(u, w)]
                dev[w] = (dev[u][0] + math.cos(duw), dev[u][1] + math.sin(duw))
                vis.add(w); parent[w] = u; q.append(int(w))
    defmax = 0.0
    for (u, w), duw in dir_edge.items():
        if not math.isnan(dev[u][0]) and not math.isnan(dev[w][0]):
            ex = dev[u][0] + math.cos(duw); ey = dev[u][1] + math.sin(duw)
            defmax = max(defmax, math.hypot(ex - dev[w][0], ey - dev[w][1]))
    return dev, defmax


def _sistema_rot(adj, pos, giso, N):
    """orden CCW, ángulos de carta e índice cíclico por vértice (sistema de rotación)."""
    orden = [None] * N; ang = [None] * N; idx = [None] * N
    for u in range(N):
        if not adj[u]:
            orden[u] = []; ang[u] = {}; idx[u] = {}; continue
        a = {w: _chart_angle(u, w, pos, giso) for w in adj[u]}
        o = sorted(a, key=lambda w: a[w])
        orden[u] = o; ang[u] = a; idx[u] = {w: k for k, w in enumerate(o)}
    return orden, ang, idx


def _camino_lado(adjc, side, s, t):
    """Camino más corto s→t RESTRINGIDO al lado de s (no cruza la costura)."""
    sd = side[s]
    prev = {s: None}; q = deque([s])
    while q:
        u = q.popleft()
        if u == t:
            break
        for w in adjc[u]:
            if w not in prev and side.get(w) == sd:
                prev[w] = u; q.append(int(w))
    if t not in prev:
        return None
    path = []; x = t
    while x is not None:
        path.append(x); x = prev[x]
    return path[::-1]


def burgers_franja(adjc, orden, ang, idx, riv_bottom, riv_top, side):
    """Vector de BURGERS de la franja entre los dos remaches (adjudicación CS: el estadístico es el
    cierre afín del 2º puente). Transporta un marco alrededor del lazo cerrado
        a1 →[orilla izq]→ a2 →(remache top)→ b2 →[orilla der]→ b1 →(remache bottom)→ a1
    y suma los vectores de arista desarrollados (equilátero, giro por hueco-exterior). En el PLANO
    el lazo cierra (Σ=0, burgers≈0); en curvatura la franja encierra déficit → Σ≠0 (burgers>0).
    Devuelve (burgers_traslacional, holonomía_rotacional, nº_aristas_lazo)."""
    a1, b1 = riv_bottom; a2, b2 = riv_top
    P1 = _camino_lado(adjc, side, a1, a2)          # orilla izquierda (lado<0)
    P2 = _camino_lado(adjc, side, b2, b1)          # orilla derecha (lado>0)
    if P1 is None or P2 is None:
        return None
    L = P1 + P2                                     # lazo cerrado (a2→b2 y b1→a1 son los remaches)
    n = len(L)
    if n < 4:
        return None
    d = 0.0; sx = math.cos(d); sy = math.sin(d)     # 1ª arista a2... marco global arbitrario
    for i in range(1, n):
        u = L[i]; prev = L[i - 1]; nxt = L[(i + 1) % n]
        t = _turn(orden[u], idx[u], adjc, prev, nxt)
        if t is None:
            return None
        d = (d + math.pi) + t                       # dir(u→nxt)
        sx += math.cos(d); sy += math.sin(d)
    burg = math.hypot(sx, sy)
    # holonomía rotacional (cierre de marco) = déficit encerrado, como cruce
    t0 = _turn(orden[L[0]], idx[L[0]], adjc, L[-1], L[1])
    rot = ((d + math.pi) + t0) if t0 is not None else float("nan")
    rot = abs((rot + math.pi) % (2 * math.pi) - math.pi)
    return burg, rot, n


def repegar(adjc, dev, La, Ra, cut_pairs, modo, rng):
    """REGLA: pega a∈La con el b∈Ra cuya posición DESARROLLADA cae más cerca de a (<NEAR_TOL) =
    holonomía afín de lazo ≈0. CONTROL: G pares al azar (mismos pools). Devuelve (adj, ng)."""
    adj = [set(s) for s in adjc]
    G = len(cut_pairs); ng = 0
    Ra_arr = np.array(Ra)
    if modo == "REGLA":
        devR = dev[Ra_arr]
        for a in La:
            if math.isnan(dev[a][0]):
                continue
            off = devR - dev[a]
            dist = np.hypot(off[:, 0], off[:, 1])
            dist[~np.isfinite(dist)] = 1e18
            k = int(np.argmin(dist))
            b = int(Ra_arr[k])
            if dist[k] < NEAR_TOL and b not in adj[a] and a != b:
                adj[a].add(b); adj[b].add(a); ng += 1
    else:
        aa = list(rng.permutation(La)); bb = list(rng.permutation(Ra))
        for a, b in zip(aa[:G], bb[:G]):
            a = int(a); b = int(b)
            if a != b and b not in adj[a]:
                adj[a].add(b); adj[b].add(a); ng += 1
    return adj, ng


def _medir(adj, N, sd):
    adjF = _fin(adj)
    dia = diametro(adjF, N, seed=sd)
    g = dimension_crecimiento(adjF, N, seed=sd)
    r = diagnos(adjF, N, K, seed=sd + 11)
    turn = sphere_turnover(adjF, N, seed=sd + 5)
    return dia, g["d"], r["dmean"], r["fg"], turn


def main():
    t0 = time.time()
    print("CG004-f ETAPA 2 — BARRIDO κ: DOS REMACHES + cierre afín del 2º puente (vector de Burgers)")
    print("=" * 100)
    print(f"q∈{QS} (6=plano,≥7=hiperb) · N≈{TARGET_N} · estadístico = BURGERS de la franja entre remaches")
    print("adjudicación CS: objeto = holonomía TRASLACIONAL (Burgers), no rotacional; 2 remaches matan el")
    print("modo cero de giro; en κ=0 el lazo cierra (burgers≈0), en κ≠0 la franja encierra déficit (burgers>0)")

    res = {}
    print(f"\n  {'q':>2} {'N':>6} {'defic':>6} {'lazo':>5} {'BURGERS':>9} {'rot_holon':>10} {'esperado':>9}")
    print("  " + "-" * 60)
    for q in QS:
        adj, pos, N, orden, giso = construir(q, TARGET_N)
        corte = cortar_costura(adj, pos, N)
        if corte is None:
            print(f"  q={q}: <3 cruces de costura — salto"); continue
        adjc, cut_pairs, La, Ra, side, riv_b, riv_t, seed_dev = corte
        oo, aa, ii = _sistema_rot(adjc, pos, giso, N)
        out = burgers_franja(adjc, oo, aa, ii, riv_b, riv_t, side)
        if out is None:
            print(f"  q={q}: lazo de franja no bien formado (orilla desconexa) — revisar"); continue
        burg, rot, nloop = out
        defic = abs(6 - q) * math.pi / 3.0
        res[q] = (burg, rot, nloop, N, defic)
        print(f"  {q:>2} {N:>6} {defic:>6.2f} {nloop:>5} {burg:>9.3f} {rot:>10.3f} {defic:>9.2f}", flush=True)

    # ---------- GUARDIÁN + FRONTERA ----------
    print("\n" + "=" * 100)
    if 6 not in res:
        print("✗ No se pudo medir q=6 (plano) — no hay guardián. Abortado."); return
    burg6 = res[6][0]
    guard_ok = burg6 < 0.5
    print(f"GUARDIÁN (plano q=6): burgers={burg6:.3f}  ->  "
          f"{'OK (el lazo cierra en el plano; 2 remaches fijan el giro)' if guard_ok else '¡NO cierra! aún hay modo cero suelto — no leer frontera'}")
    if not guard_ok:
        print("  El plano no da burgers≈0: la construcción de los remaches/orillas sigue mal. Abortado.")
        print(f"\nTiempo: {(time.time()-t0)/60:.1f} min"); return

    print("\nFRONTERA — BURGERS de la franja vs curvatura (κ = defic):")
    frontera = None
    for q in QS:
        if q not in res:
            continue
        burg, rot, nloop, N, defic = res[q]
        abre = burg > 0.5
        estado = "CIERRA (preserva)" if not abre else "ABRE (frontera)"
        if frontera is None and abre and q > 6:
            frontera = q
        print(f"  q={q} (defic={defic:.2f}): burgers={burg:.3f}  [{estado}]  · rot_holon={rot:.3f} (≈déficit encerrado)")
    print("\nLECTURA (pre-registrada, adjudicación CS):")
    print("  · burgers≈0 en el plano y >0 apenas κ≠0 → el pegado-por-desarrollo NO tolera curvatura →")
    print("    PRESERVA pero NO GENERA planitud. Lever RELOCALIZADO a 'generar consistencia de marcos'.")
    print("  · burgers≈0 hasta cierto κ_c>0 → ventana donde el pegado ayuda (positivo no trivial).")
    if frontera is not None:
        print(f"\n  >>> FRONTERA: el 2º puente deja de cerrar (Burgers>0) en q={frontera} "
              f"(defic={abs(6-frontera)*math.pi/3:.2f}).")
    print(f"\nTiempo: {(time.time()-t0)/60:.1f} min")


main()
