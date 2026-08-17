"""
CS063 — EL VÉRTICE DE 3 CUERPOS GENUINO: mover los TRES marcos JUNTOS (no cada uno hacia la media de vecinos).
=============================================================================================================
CS061 midió un defecto de 3 cuerpos pero su UPDATE era campo medio PAREADO (cada nodo→media de vecinos). Por
eso no tuvo derecho a declarar "el 3-puntos no basta". CS063 hace lo que falta: un update de 3 cuerpos GENUINO,
donde los tres marcos de una tríada se mueven JUNTOS por un término IRREDUCIBLE, sin término pareado.

TÉRMINO DE 3 CUERPOS (irreducible): E = Σ_tríadas (s_i·(s_j×s_k))²  — el PRODUCTO TRIPLE ESCALAR al cuadrado.
Es cero ⟺ los tres marcos son COPLANARES (la tríada "cierra"); mide el volumen orientado del triple de marcos.
G-IRREDUCIBLE: el producto triple NO es suma de funciones de pares — ∂³E/∂s_i∂s_j∂s_k ∝ Levi-Civita ≠ 0
(verificado numéricamente antes de correr). El update: s_i -= lr·∂E/∂s_i con ∂E/∂s_i = 2(s_i·(s_j×s_k))(s_j×s_k),
que depende CONJUNTAMENTE de s_j×s_k — NO reducible a media-de-vecinos. Requiere K=3 (producto vectorial).

JUEZ: holonomía del marco (CS059) CON control de longitud de ciclo (el que cazó CS059 y CS061). BRAZOS:
3cuerpos (este) / 2cuerpos (=CS061, media de vecinos) / null_triada (tríadas al azar) / null_marco (espines
barajados). La selección real debe colapsar bajo NULL y sobrevivir al control de longitud. Éxito ≠ "salió 3D".

DESENLACES: (A) el 3-cuerpos GENUINO selecciona dim → el ingrediente que faltaba. (B) tampoco selecciona → NI
el vértice de 3 puntos basta (el negativo que CS061 no podía declarar; cierra el arco de eliminación → empuja
a la hipótesis de dimensión CONTINGENTE). (C) selecciona OTRA dim → falsación aguas arriba.
Reusa CS059. numpy + multiprocessing.
"""
from __future__ import annotations
import os, sys, csv, math, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_s = open(os.path.join(_HERE, "cs059_espin_como_marco.py")).read().replace('\nif __name__ == "__main__":\n    main()\n', "\n")
_C9 = {"__file__": os.path.join(_HERE, "cs059_espin_como_marco.py"), "__name__": "cs059_mod"}
exec(compile(_s, "cs059_espin_como_marco.py", "exec"), _C9)
_spins = _C9["_spins"]; _holonomia_ciclo = _C9["_holonomia_ciclo"]; _ciclos_fundamentales = _C9["_ciclos_fundamentales"]
_ensemble = _C9["_ensemble"]

# ============================ CONFIG ============================
K = 3                                    # producto triple → espacio interno S² (dim-neutral respecto a la semilla)
SEEDS    = int(os.environ.get("CS063_SEEDS", 10))
N_CICLOS = int(os.environ.get("CS063_CICLOS", 500))
PASOS    = 12
LR       = 0.15
MAXTRI   = 8                             # tríadas (pares de vecinos) muestreadas por nodo
WORKERS  = int(os.environ.get("CS063_WORKERS", max(1, (os.cpu_count() or 4) - 2)))
OUT      = os.environ.get("CS063_OUT", os.path.join(_HERE, "cs063_3cuerpos.csv"))
SMOKE    = os.environ.get("CS063_SMOKE", "") != ""
DIMS = ["d2", "d3", "d4", "curv"]
LBINS = [3, 4, 5, 6, 8]
# ===============================================================


def _triadas_por_nodo(adj, N, rng):
    """Para cada nodo i, una lista de pares de vecinos (j,k) — la tríada j-i-k (i central). Muestra si hay muchos."""
    tri = [[] for _ in range(N)]
    for i in range(N):
        vec = list(adj[i])
        if len(vec) < 2:
            continue
        if len(vec) <= 4:
            for a in range(len(vec)):
                for b in range(a + 1, len(vec)):
                    tri[i].append((vec[a], vec[b]))
        else:
            for _ in range(MAXTRI):
                a, b = rng.choice(len(vec), 2, replace=False)
                tri[i].append((vec[a], vec[b]))
    return tri


def _verifica_irreducible():
    """G-IRREDUCIBLE: ∂³E/∂s_i∂s_j∂s_k del producto triple ≠ 0 (numérico). Sin esto, no es 3-cuerpos."""
    rng = np.random.default_rng(0); h = 1e-4
    def E(si, sj, sk):
        return float(np.dot(si, np.cross(sj, sk))) ** 2
    si = rng.standard_normal(3); sj = rng.standard_normal(3); sk = rng.standard_normal(3)
    # diferencia cruzada mixta en una componente de cada uno
    def shift(v, comp, d):
        v2 = v.copy(); v2[comp] += d; return v2
    d3 = 0.0
    for a in (1, -1):
        for b in (1, -1):
            for c in (1, -1):
                d3 += a * b * c * E(shift(si, 0, a * h), shift(sj, 1, b * h), shift(sk, 2, c * h))
    d3 /= (2 * h) ** 3
    return abs(d3) > 1e-3      # ∂³E ≠ 0 → irreducible


def _update_3cuerpos(adj, N, spins, tri, lr, pasos):
    """Descenso sobre E=Σ(s_i·(s_j×s_k))². El grad de s_i depende CONJUNTAMENTE de s_j×s_k (3-cuerpos genuino,
    SIN término pareado). Mueve los tres marcos de cada tríada hacia coplanaridad (la tríada cierra)."""
    s = spins.copy()
    for _ in range(pasos):
        grad = np.zeros_like(s)
        for i in range(N):
            for (j, k) in tri[i]:
                cr = np.cross(s[j], s[k])           # s_j × s_k
                tp = float(np.dot(s[i], cr))         # producto triple
                grad[i] += 2.0 * tp * cr             # ∂E/∂s_i (conjunto en j,k)
                # contribuciones a j y k (la tríada mueve los TRES): ∂/∂s_j (s_i·(s_j×s_k)) = s_k×s_i
                grad[j] += 2.0 * tp * np.cross(s[k], s[i])
                grad[k] += 2.0 * tp * np.cross(s[i], s[j])
        s = s - lr * grad
        nrm = np.linalg.norm(s, axis=1, keepdims=True)
        s = s / np.maximum(nrm, 1e-12)
    return s


def _update_2cuerpos(adj, N, spins, pasos):
    """Control = CS061: campo medio PAREADO (cada nodo → media de vecinos). El update que CS063 reemplaza."""
    s = spins.copy()
    for _ in range(pasos):
        ns = s.copy()
        for i in range(N):
            if adj[i]:
                v = s[list(adj[i])].sum(axis=0) + s[i]
                n = np.linalg.norm(v)
                if n > 1e-9:
                    ns[i] = v / n
        s = ns
    return s


def _holonomia_binned(adj, N, spins, rng):
    ciclos = _ciclos_fundamentales(adj, N, N_CICLOS, rng)
    if not ciclos:
        return {}, float("nan")
    w0 = np.zeros(K); w0[0] = 1.0
    porbin = {L: [] for L in LBINS}; todo = []
    for c in ciclos:
        h = _holonomia_ciclo(spins, c, w0); todo.append(h)
        if len(c) in porbin:
            porbin[len(c)].append(h)
    return {L: (float(np.mean(v)) if len(v) >= 3 else None) for L, v in porbin.items()}, float(np.mean(todo))


def _worker(arg):
    pid, dim, brazo, seed = arg
    ens = dict(_ensemble()); adj0, N = ens[dim]
    adj = [set(a) for a in adj0]
    spins = _spins(N, K, np.random.default_rng(seed * 991 + 7))
    rng = np.random.default_rng(seed * 100003 + pid * 17)
    tri = _triadas_por_nodo(adj, N, rng)
    if brazo == "3cuerpos":
        s = _update_3cuerpos(adj, N, spins, tri, LR, PASOS)
    elif brazo == "2cuerpos":
        s = _update_2cuerpos(adj, N, spins, PASOS)
    elif brazo == "null_triada":
        tri2 = [[(int(rng.integers(N)), int(rng.integers(N))) for _ in t] for t in tri]  # tríadas al azar
        s = _update_3cuerpos(adj, N, spins, tri2, LR, PASOS)
    elif brazo == "null_marco":
        s = spins.copy(); rng.shuffle(s)
    binmean, fb = _holonomia_binned(adj, N, s, np.random.default_rng(seed * 41 + 3))
    fila = dict(point_id=pid, dim=dim, brazo=brazo, seed=seed, frame_burgers=round(fb, 5))
    for L in LBINS:
        v = binmean.get(L); fila[f"L{L}"] = round(v, 5) if v is not None else ""
    return [fila]


def main():
    print("CS063 — VÉRTICE DE 3 CUERPOS GENUINO: ¿el update de 3 cuerpos selecciona dim donde el pareado no?", flush=True)
    print("=" * 104, flush=True)
    irr = _verifica_irreducible()
    print(f"G-IRREDUCIBLE (∂³E≠0 del producto triple): {'PASA ✓' if irr else 'FALLA ✗ — NO correr'}", flush=True)
    if not irr:
        print("El término no es 3-cuerpos genuino. Abortando (sería CS061 con otro nombre).", flush=True); return
    print("PREDICCIÓN CIEGA: si el 3-cuerpos GENUINO es el ingrediente, seleccionará una dim a IGUAL longitud de", flush=True)
    print("  ciclo (donde el pareado no pudo), colapsando bajo NULL. Si no, NI el vértice de 3 puntos basta (B).", flush=True)
    brazos = ["3cuerpos", "2cuerpos", "null_triada", "null_marco"]
    args = []; pid = 0
    for dim in DIMS:
        for brazo in brazos:
            for seed in range(SEEDS):
                args.append((pid, dim, brazo, seed)); pid += 1
    if SMOKE:
        args = [a for a in args if a[3] < 1][:12]
    print(f"dims={DIMS} brazos={brazos} seeds={SEEDS} · corridas={len(args)} · workers={WORKERS} · K={K}", flush=True)

    hechos = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for row in csv.DictReader(f):
                hechos.add(int(row["point_id"]))
    args = [a for a in args if a[0] not in hechos]
    campos = ["point_id", "dim", "brazo", "seed", "frame_burgers"] + [f"L{L}" for L in LBINS]
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
