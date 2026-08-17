"""
CS059 (R7) — EL ESPÍN COMO MARCO: ¿es el marco —no la fuerza— lo que selecciona una dimensión?
==============================================================================================
CS057 mostró que NINGUNA fuerza local selecciona el 3D-plano. El arco entero converge: todos los mecanismos
probados operan sobre la ADYACENCIA (quién-con-quién), ninguno sobre el MARCO (con-qué-orientación). CS059
mete el marco —vía el espín, la orientación intrínseca de las partículas— y pregunta si el MARCO selecciona
una dimensión de forma consistente y falsable. El éxito NO es "salió 3D" (G-NO-FORZAR-3D): es que el
acoplamiento de marcos SELECCIONE ALGUNA dimensión, robusta, que COLAPSE bajo NULL.

MECANISMO (dim-neutral, NO-gauge — la lección de CS052 respetada):
- ESPÍN = vector unitario intrínseco en un espacio interno S^{K-1} de dimensión K FIJA e igual para TODAS las
  semillas dimensionales (una rejilla 4D también recibe espines de K comp.). Se asigna al nacer, JAMÁS se
  reajusta mirando la geometría (G-ESPIN-INTRINSECO). Se BARRE K∈{2,3,4,5}: si la dim seleccionada cambia con
  K, es inyección (G-NO-INYECTAR lo caza); si es robusta a K, es genuina.
- TRANSPORTE por enlace = TRANSPORTE PARALELO (rotación mínima) entre los espines de los extremos. NO es una
  diferencia de valores de nodo (eso sería puro gauge, holonomía trivial siempre — CS052). El transporte
  paralelo alrededor de un ciclo da la FASE DE BERRY / ángulo sólido: para K≥3 NO telescopia (curvatura real
  de la esfera), para K=2 (círculo abeliano) SÍ telescopia → trivial. Ese contraste K=2 vs K≥3 es un chequeo.
- JUEZ = la HOLONOMÍA del marco (generalización del Burgers de CG004: holonomía de una conexión alrededor de
  un lazo cerrado; CG004 es la instancia traslacional-2D, CS059 la instancia del espín). Marco consistente
  (plano) ⟺ un vector de prueba transportado alrededor del ciclo VUELVE a sí mismo (holonomía≈0).
- Bajo EXPANSIÓN (se remueven enlaces) se mide cómo evoluciona la holonomía del marco por dimensión.
- NULL (G-NULL-MARCO): transportes al AZAR por enlace (rompe la regla física de transporte paralelo). La
  discriminación de dimensión debe COLAPSAR bajo NULL.

DESENLACES (los cuatro honestos): (A) el marco selecciona 3D-plano → la planitud vive en el marco (cierre
mayor). (B) selecciona OTRA dim → el marco selecciona pero no la nuestra (falsación nueva). (C) no selecciona
/ se sostiene bajo NULL → el espín tampoco basta (negativo fuerte). (D) la representación resultó no-neutral
(G-NO-INYECTAR) → rediseño.

numpy + multiprocessing. Reusa constructores de retículo de CS055/CS057 (semillas dimensionales).
"""
from __future__ import annotations
import os, sys, csv, math, time
import numpy as np
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
_s = open(os.path.join(_HERE, "cs055_proceso_acoplado.py")).read().replace("\nmain()\n", "\n")
_C5 = {"__file__": os.path.join(_HERE, "cs055_proceso_acoplado.py")}
exec(compile(_s, "cs055_proceso_acoplado.py", "exec"), _C5)
cadena = _C5["cadena"]; cuadrada2d = _C5["cuadrada2d"]; cubica3d = _C5["cubica3d"]; hipercubica4d = _C5["hipercubica4d"]
# curvo desde cg004f
_sf = open(os.path.join(_HERE, "cg004f_barrido_curvatura.py")).read().replace("\nmain()\n", "\n")
_F = {"__file__": os.path.join(_HERE, "cg004f_barrido_curvatura.py")}
exec(compile(_sf, "cg004f_barrido_curvatura.py", "exec"), _F)
tri_hiperbolica = _F["tri_hiperbolica"]

# ============================ CONFIG ============================
K_SWEEP  = [2, 3, 4, 5]                  # dimensión del espacio INTERNO del espín (barrido — G-NO-INYECTAR)
SEEDS    = int(os.environ.get("CS059_SEEDS", 8))
N_CICLOS = int(os.environ.get("CS059_CICLOS", 400))   # nº de ciclos muestreados por grafo
EXP_PASOS = [0, 1, 2, 3]                 # pasos de expansión donde se mide la holonomía
EXP_RATE = 0.10                          # fracción de enlaces removidos por paso
WORKERS  = int(os.environ.get("CS059_WORKERS", max(1, (os.cpu_count() or 4) - 2)))
OUT      = os.environ.get("CS059_OUT", os.path.join(_HERE, "cs059_marco.csv"))
SMOKE    = os.environ.get("CS059_SMOKE", "") != ""
# ===============================================================


def _ensemble():
    """Semillas dimensionales — d1..d4 plano + curvo. La MISMA representación de espín se ofrece a todas."""
    c = tri_hiperbolica(7, 360)
    return [
        ("d1",   cadena(256)),
        ("d2",   cuadrada2d(289)),
        ("d3",   cubica3d(343)),
        ("d4",   hipercubica4d(256)),
        ("curv", (list(c[0]), c[2])),
    ]


# ---------- ESPÍN y TRANSPORTE PARALELO (dim-neutral, no-gauge) ----------
def _spins(N, K, rng):
    v = rng.standard_normal((N, K))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _transporta(a, b, w, eps=1e-9):
    """Aplica al vector w la ROTACIÓN MÍNIMA que lleva a→b (transporte paralelo en la esfera S^{K-1}).
    Rota solo en el plano generado por a,b, por el ángulo entre ellos. Identidad si a≈b o a≈-b."""
    c = float(np.dot(a, b))
    c = max(-1.0, min(1.0, c))
    if c > 1 - eps:
        return w
    # base ortonormal del plano (a, b): e1=a, e2 = componente de b ortogonal a a
    e1 = a
    b_perp = b - c * a
    n = np.linalg.norm(b_perp)
    if n < eps:
        return w  # a y b antipodales: rotación mal definida → identidad (transporte trivial)
    e2 = b_perp / n
    theta = math.acos(c)
    w1 = float(np.dot(w, e1)); w2 = float(np.dot(w, e2))
    ct, st = math.cos(theta), math.sin(theta)
    # rota la componente (w1,w2) en el plano (e1,e2); el resto de w queda igual
    dw1 = (ct - 1) * w1 - st * w2
    dw2 = st * w1 + (ct - 1) * w2
    return w + dw1 * e1 + dw2 * e2


def _holonomia_ciclo(spins, ciclo, w0):
    """Transporta el vector de prueba w0 alrededor del ciclo (secuencia de nodos, cerrada) por transporte
    paralelo entre espines consecutivos. Devuelve el ÁNGULO entre w0 y el vector final (0 = marco consistente)."""
    w = w0.copy()
    L = len(ciclo)
    for k in range(L):
        a = spins[ciclo[k]]; b = spins[ciclo[(k + 1) % L]]
        w = _transporta(a, b, w)
    c = float(np.dot(w0, w)) / (np.linalg.norm(w0) * np.linalg.norm(w) + 1e-12)
    c = max(-1.0, min(1.0, c))
    return math.acos(c)   # ángulo de holonomía en [0, π]


def _ciclos_fundamentales(adj, N, nmax, rng):
    """Ciclos fundamentales: BFS árbol de expansión; cada arista NO-árbol + camino en el árbol = un ciclo.
    Muestra hasta nmax. Dim-neutral: solo usa adyacencia."""
    padre = {}; vis = np.zeros(N, bool); raiz = -1
    # componente gigante: empezar del nodo de mayor grado
    deg = np.array([len(a) for a in adj])
    if deg.sum() == 0:
        return []
    start = int(deg.argmax())
    padre[start] = -1; vis[start] = True; q = deque([start]); orden = []
    tree = set()
    while q:
        u = q.popleft(); orden.append(u)
        for v in adj[u]:
            if not vis[v]:
                vis[v] = True; padre[v] = u; tree.add((min(u, v), max(u, v))); q.append(int(v))
    # aristas no-árbol
    no_arbol = []
    for u in range(N):
        for v in adj[u]:
            if u < v and (u, v) not in tree:
                no_arbol.append((u, v))
    if len(no_arbol) > nmax:
        idx = rng.choice(len(no_arbol), size=nmax, replace=False)
        no_arbol = [no_arbol[i] for i in idx]

    def camino_raiz(x):
        p = []
        while x != -1 and x in padre:
            p.append(x); x = padre[x]
        return p
    ciclos = []
    for (u, v) in no_arbol:
        pu = camino_raiz(u); pv = camino_raiz(v)
        su = set(pu); lca = next((x for x in pv if x in su), None)
        if lca is None:
            continue
        cu = pu[:pu.index(lca) + 1]; cv = pv[:pv.index(lca)]
        ciclo = cu + cv[::-1]
        if len(ciclo) >= 3:
            ciclos.append(ciclo)
    return ciclos


def _frame_burgers(adj, N, spins, K, rng, null=False):
    """Holonomía media del marco sobre ciclos fundamentales. null=True → transportes al AZAR (rompe la regla
    física de transporte paralelo): cada paso rota w por un ángulo aleatorio en un plano aleatorio."""
    ciclos = _ciclos_fundamentales(adj, N, N_CICLOS, rng)
    if not ciclos:
        return float("nan"), 0, 0.0
    w0 = np.zeros(K); w0[0] = 1.0
    hol = []
    longs = []
    if null:
        # NULL: transportes aleatorios independientes (no derivados de los espines)
        for ciclo in ciclos:
            w = w0.copy()
            for _ in range(len(ciclo)):
                a = rng.standard_normal(K); a /= np.linalg.norm(a)
                b = rng.standard_normal(K); b /= np.linalg.norm(b)
                w = _transporta(a, b, w)
            c = max(-1.0, min(1.0, float(np.dot(w0, w)) / (np.linalg.norm(w) + 1e-12)))
            hol.append(math.acos(c)); longs.append(len(ciclo))
    else:
        for ciclo in ciclos:
            hol.append(_holonomia_ciclo(spins, ciclo, w0)); longs.append(len(ciclo))
    return float(np.mean(hol)), len(ciclos), float(np.mean(longs))


def _expandir(adj, N, rng, rate):
    edges = [(i, j) for i in range(N) for j in adj[i] if i < j]
    rng.shuffle(edges)
    for (i, j) in edges[:int(rate * len(edges))]:
        adj[i].discard(j); adj[j].discard(i)


def _worker(arg):
    pid, nombre, K, seed = arg
    ens = dict(_ensemble())
    adj0, N = ens[nombre]
    adj = [set(a) for a in adj0]
    rng = np.random.default_rng(seed * 100003 + pid * 17 + K)
    spins = _spins(N, K, np.random.default_rng(seed * 991 + K * 7 + 1))
    filas = []
    for ep in EXP_PASOS:
        if ep > 0:
            _expandir(adj, N, rng, EXP_RATE)
        fb, nc, ml = _frame_burgers(adj, N, spins, K, np.random.default_rng(seed * 41 + ep * 13 + K), null=False)
        fbn, _, _ = _frame_burgers(adj, N, spins, K, np.random.default_rng(seed * 41 + ep * 13 + K), null=True)
        filas.append(dict(point_id=pid, dim=nombre, K=K, seed=seed, exp_paso=ep,
                          frame_burgers=round(fb, 5), frame_burgers_null=round(fbn, 5),
                          n_ciclos=nc, long_media=round(ml, 2), N=N))
    return filas


def main():
    print("CS059 (R7) — EL ESPÍN COMO MARCO: ¿selecciona el marco una dimensión? (juez=holonomía del marco)", flush=True)
    print("=" * 104, flush=True)
    print("G-NO-FORZAR-3D: éxito = seleccionar ALGUNA dim robusta a K que COLAPSE bajo NULL, NO 'salió 3D'.", flush=True)
    print("PREDICCIÓN CIEGA (pre-registrada): si el MARCO selecciona, una dimensión tendrá holonomía del marco", flush=True)
    print("  sistemáticamente MENOR (marcos más consistentes) que las otras, ROBUSTA a K∈{2,3,4,5}, y esa", flush=True)
    print("  separación COLAPSA bajo NULL (transportes al azar). K=2 (abeliano) debe dar holonomía ~trivial", flush=True)
    print("  (chequeo interno: el marco necesita K≥3 para ser no-trivial, como el espín real SU(2)/S²).", flush=True)
    print(f"K_sweep={K_SWEEP} · dims=d1/d2/d3/d4/curv · seeds={SEEDS} · exp_pasos={EXP_PASOS} · ciclos≤{N_CICLOS}", flush=True)

    dims = ["d1", "d2", "d3", "d4", "curv"]
    args = []
    pid = 0
    for nombre in dims:
        for K in K_SWEEP:
            for seed in range(SEEDS):
                args.append((pid, nombre, K, seed)); pid += 1
    if SMOKE:
        args = args[:12]
    print(f"corridas = {len(args)} · workers={WORKERS} · salida {OUT}", flush=True)

    hechos = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for row in csv.DictReader(f):
                hechos.add(int(row["point_id"]))
    args = [a for a in args if a[0] not in hechos]
    campos = ["point_id", "dim", "K", "seed", "exp_paso", "frame_burgers", "frame_burgers_null",
              "n_ciclos", "long_media", "N"]
    fout = open(OUT, "a", newline=""); wr = csv.DictWriter(fout, fieldnames=campos)
    if not hechos:
        wr.writeheader()
    t0 = time.time(); n = 0
    import multiprocessing as mp
    if WORKERS > 1 and not SMOKE:
        with mp.Pool(WORKERS) as pool:
            for filas in pool.imap_unordered(_worker, args, chunksize=1):
                for fila in filas: wr.writerow(fila)
                fout.flush(); n += 1
                if n % 20 == 0 or n == len(args):
                    dt = time.time() - t0; r = n / dt
                    print(f"  {n}/{len(args)} · {dt/60:.1f}min · ETA {(len(args)-n)/r/60:.1f}min", flush=True)
    else:
        for a in args:
            for fila in _worker(a): wr.writerow(fila)
            fout.flush(); n += 1
            print(f"  {n}/{len(args)} · {time.time()-t0:.1f}s", flush=True)
    fout.close()
    print(f"\nCOMPLETO: {n} corridas en {(time.time()-t0)/60:.1f} min → {OUT}", flush=True)


if __name__ == "__main__":
    main()
