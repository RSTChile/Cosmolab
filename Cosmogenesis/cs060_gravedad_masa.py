"""
CS060 — Misión B: LA GRAVEDAD CON MASA REAL vs el PROXY DE GRADO (re-test del negativo central del arco)
========================================================================================================
Alexis cazó por lógica que la "gravedad" de CS054-057 se acopló a ρ=GRADO (densidad de vínculos), no a la
MASA — era enlace preferencial mal llamado gravedad (verificado en los 4 archivos: AUDITORIA_gravedad_sin_masa).
La masa que CS060 introduce es EXACTAMENTE el ingrediente que la gravedad necesitaba. Pregunta filosa:
  ¿cambia la geometría seleccionada cuando la gravedad se acopla a lo que le corresponde (MASA) en vez de a
  un proxy (grado)? Si la gravedad-real elige distinto que la gravedad-por-densidad, el negativo de CS057 se
  relee: probamos el proxy, no la gravedad.

BRAZOS de acoplamiento gravitatorio: 'masa' (∝ masa del nodo, INTRÍNSECA y fija) · 'grado' (∝ grado = CS057) ·
'null' (∝ masa BARAJADA). Todo lo demás igual. Se mide la selección de geometría (viable/expande por dim),
mismo criterio ciego de CS057.

GUARDIÁN CRÍTICO G-MASA-SEPARADA-DEL-GRADO: la masa se sortea INDEPENDIENTE del grado (log-uniforme en el
rango leptónico 1..3477). Assert de correlación masa↔grado ≈ 0 al inicio. Si se colapsa en ρ, es el mismo
error de CS054-057. La masa es intrínseca (fija), el grado es emergente (cambia) — como en la física.

Reusa el motor de CS057. numpy + multiprocessing.
"""
from __future__ import annotations
import os, sys, csv, math, time
import numpy as np
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
_s = open(os.path.join(_HERE, "cs057_paisaje_completo.py")).read()
_s = _s.replace('\nif __name__ == "__main__":\n    main()\n', "\n")
_C7 = {"__file__": os.path.join(_HERE, "cs057_paisaje_completo.py"), "__name__": "cs057_mod"}
exec(compile(_s, "cs057_paisaje_completo.py", "exec"), _C7)
_confin = _C7["_confin"]; _em = _C7["_em"]; _debil = _C7["_debil"]; _despliegue = _C7["_despliegue"]
_diam = _C7["_diam"]; _giant = _C7["_giant"]; _colores = _C7["_colores"]; _construye_ensemble = _C7["_construye_ensemble"]
_alc_a_dmax = _C7["_alc_a_dmax"]
R_STRONG = _C7["R_STRONG"]; R_EM = _C7["R_EM"]; R_WEAK = _C7["R_WEAK"]; R_EXP = _C7["R_EXP"]; R_GRAV = _C7["R_GRAV"]
T_HI = _C7["T_HI"]; T_LO = _C7["T_LO"]; T_CONF = _C7["T_CONF"]; ALPHA = _C7["ALPHA"]

# ============================ CONFIG ============================
STEPS   = 16
SEEDS   = int(os.environ.get("CS060_SEEDS", 10))
G_LEVELS = [0.3, 0.6, 1.0]              # fuerza gravitatoria (barrido)
WORKERS = int(os.environ.get("CS060_WORKERS", max(1, (os.cpu_count() or 4) - 2)))
OUT     = os.environ.get("CS060_OUT", os.path.join(_HERE, "cs060_gravmasa.csv"))
SMOKE   = os.environ.get("CS060_SMOKE", "") != ""
# régimen fijo donde la gravedad actúa (expansión moderada, confinamiento on, EM/débil bajos)
W_EXP = float(os.environ.get("CS060_WEXP", 0.5))
W_STRONG, W_EM, W_WEAK, W_COOL, ALC = 0.6, 0.2, 0.1, 0.5, 1.0
CL = ["d1", "d2", "d3", "d4", "curv"]
MASA_MIN, MASA_MAX = 1.0, 3477.0        # rango leptónico real (electrón..tauón)
# ===============================================================


def _masa(N, rng):
    """Masa por nodo: log-uniforme en el rango leptónico, INDEPENDIENTE del grado (G-MASA-SEPARADA-DEL-GRADO)."""
    return np.exp(rng.uniform(math.log(MASA_MIN), math.log(MASA_MAX), N))


def _T_paso(step, wcool):
    frac = step / max(STEPS - 1, 1); depth = 0.2 + 1.8 * wcool
    return T_HI * (T_LO / T_HI) ** min(1.0, frac * depth)


def _grav_w(adj, N, rng, rate, dmax, T, w):
    """Gravedad acoplada al VECTOR DE PESO w (masa o grado): fuente ∝ w, blanco ∝ w[j]/d^ALPHA por saltos.
    Idéntica a la de CS057 EXCEPTO que w es un parámetro (no forzosamente el grado)."""
    if rate <= 0:
        return
    E = sum(len(a) for a in adj) // 2
    if E < 1 or w.sum() <= 0:
        return
    nadd = int(rate * (0.5 + T) * E)
    if nadd <= 0:
        return
    srcs = rng.choice(N, size=nadd, p=w / w.sum())
    for i in srcs:
        i = int(i); dist = {i: 0}; q = deque([i])
        while q:
            u = q.popleft()
            if dist[u] >= dmax:
                continue
            for x in adj[u]:
                if x not in dist:
                    dist[x] = dist[u] + 1; q.append(int(x))
        cand = [(j, d) for j, d in dist.items() if d >= 2]
        if not cand:
            continue
        ww = np.array([w[j] / (d ** ALPHA) for j, d in cand])
        if ww.sum() <= 0:
            continue
        j = cand[int(rng.choice(len(cand), p=ww / ww.sum()))][0]
        adj[i].add(j); adj[j].add(i)


def proceso060(adj0, N, color0, carga0, masa, acople, g_level, dmax, rng):
    """Motor de CS057 (sync) con la GRAVEDAD acoplada según 'acople': 'masa' (w=masa fija), 'grado' (w=grado,
    recomputado), 'null' (w=masa barajada). Devuelve viable/expande de la geometría resultante."""
    adj = [set(a) for a in adj0]; col = color0.copy(); car = carga0.copy()
    deg0 = [len(a) for a in adj]; t = np.zeros(N, dtype=np.int32)
    if acople == "null":
        masa = masa.copy(); rng.shuffle(masa)
    CAP_E = 12 * N; D = []; G = []
    for step in range(STEPS):
        T = _T_paso(step, W_COOL)
        E = sum(len(a) for a in adj) // 2
        if E < 2 or E > CAP_E:
            D.append(_diam(adj, N)); G.append(_giant(adj, N)); break
        # gravedad con el acoplamiento elegido
        if acople == "grado":
            w = np.array([len(a) for a in adj], float)
        else:                                     # 'masa' o 'null' → masa (fija o barajada), INTRÍNSECA
            w = masa
        _grav_w(adj, N, rng, g_level * R_GRAV, dmax, T, w)
        if T < T_CONF:
            _confin(adj, N, col, t, rng, W_STRONG * R_STRONG)
        _em(adj, N, car, deg0, rng, W_EM * R_EM)
        _debil(N, col, car, rng, W_WEAK * R_WEAK)
        _despliegue(adj, N, rng, W_EXP * R_EXP)
        dd = _diam(adj, N); gg = _giant(adj, N); D.append(dd); G.append(gg)
        if gg >= 0.9 and dd <= 2:
            break
    d0 = D[0] if D else 0; d1 = D[-1] if D else 0
    expande = int(d1 > d0)
    estable = int(len(G) > 0 and G[-1] >= 0.45 and d1 >= 2 and min(G[len(G)//2:] or [0]) >= 0.35)
    gfin = G[-1] if G else 0.0
    return int(expande and estable), expande, estable, d1, round(gfin, 3)


def _worker(arg):
    pid, g_level, acople, seed = arg
    ens = _construye_ensemble()
    rng = np.random.default_rng(seed * 100003 + pid * 17)
    filas = []
    corr_ok = True
    for ci, (nom, (adj, N)) in enumerate(ens):
        masa = _masa(N, np.random.default_rng(seed * 991 + ci * 7 + 1))
        # G-MASA-SEPARADA-DEL-GRADO: correlación masa↔grado ≈ 0 al inicio
        deg = np.array([len(a) for a in adj], float)
        if deg.std() > 0 and masa.std() > 0:
            r = float(np.corrcoef(masa, deg)[0, 1])
            if abs(r) > 0.15:
                corr_ok = False
        col = _colores(N, np.random.default_rng(seed * 131 + ci * 17 + 1))
        car = (np.arange(N) % 2).astype(np.int8)
        np.random.default_rng(seed * 977 + ci * 29 + 7).shuffle(car)
        v, e, s, dfin, gfin = proceso060(adj, N, col, car, masa, acople, g_level, _alc_a_dmax(ALC),
                                         np.random.default_rng(seed * 31 + ci * 101 + 13))
        filas.append(dict(point_id=pid, g_level=g_level, acople=acople, seed=seed, dim=nom,
                          viable=v, expande=e, estable=s, diam_fin=dfin, gig_fin=gfin,
                          corr_masa_grado_ok=int(corr_ok)))
    return filas


def main():
    print("CS060-B — GRAVEDAD CON MASA REAL vs PROXY DE GRADO (¿el negativo de CS057 era del proxy?)", flush=True)
    print("=" * 100, flush=True)
    print("PREDICCIÓN CIEGA: si el negativo de CS057 (gravedad→2D/colapso) era por el PROXY (grado), la gravedad", flush=True)
    print("  ∝MASA (intrínseca, independiente del grado) debería seleccionar geometría DISTINTA. Si da lo mismo,", flush=True)
    print("  el negativo era de la gravedad, no del proxy. NULL (masa barajada) debe parecerse a... ver.", flush=True)
    print(f"G_LEVELS={G_LEVELS} · acoples=masa/grado/null · seeds={SEEDS} · masa log-uniforme [{MASA_MIN},{MASA_MAX}]", flush=True)

    args = []; pid = 0
    for g in G_LEVELS:
        for acople in ("masa", "grado", "null"):
            for seed in range(SEEDS):
                args.append((pid, g, acople, seed)); pid += 1
    if SMOKE:
        args = args[:9]
    print(f"corridas = {len(args)} (×{len(CL)} dims) · workers={WORKERS} · salida {OUT}", flush=True)

    hechos = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for row in csv.DictReader(f):
                hechos.add(int(row["point_id"]))
    args = [a for a in args if a[0] not in hechos]
    campos = ["point_id", "g_level", "acople", "seed", "dim", "viable", "expande", "estable",
              "diam_fin", "gig_fin", "corr_masa_grado_ok"]
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
