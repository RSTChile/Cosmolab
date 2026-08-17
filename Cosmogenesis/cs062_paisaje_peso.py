"""
CS062 — EL PAISAJE CON GRAVEDAD ∝ PESO-INTRÍNSECO (no grado): ¿el 3D emerge más en TODO el mapa con la
gravedad correcta? (releer el negativo central del arco — la grieta de CS060-B a escala completa)
=====================================================================================================
CS057 acopló la gravedad al GRADO (ρ=nº de vínculos), un proxy que se AUTO-AMPLIFICA (hubs atraen más →
colapso a dim baja → sesgo ACTIVO contra el 3D — mostrado en CS060-B). CS062 acopla la gravedad a un PESO
INTRÍNSECO FIJO m_i (la ley de Newton real m_i·m_j/d²), y re-corre el PAISAJE ENTERO de CS057 (Sobol de las 6
fuerzas + punto físico + vecindad densa). ¿El 3D-plano emerge más, o el negativo se sostiene con la gravedad
correcta?

CAMBIO QUIRÚRGICO sobre CS057: SOLO el acople gravitatorio. Todo lo demás idéntico (Sobol, ejes, criterio ciego
viable=estable∧expande por tipos, sector oscuro). BRAZOS de acople: peso (m_i·m_j/d²) / grado (=CS057, control)
/ null_peso (pesos BARAJADOS — la lección de CS060-B: separa "peso real" de "cualquier cosa que no sea grado").
GUARDIANES: G-PESO-INTRÍNSECO-FIJO (m_i nunca del grado), G-PESO-SEPARADO-DEL-GRADO (corr≈0), G-NULL-PESO.

Reusa cs057_paisaje_completo.py. numpy + scipy(Sobol) + multiprocessing. Checkpoint por fila (reanudable).
"""
from __future__ import annotations
import os, sys, csv, math, time
import numpy as np
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
_s = open(os.path.join(_HERE, "cs057_paisaje_completo.py")).read().replace('\nif __name__ == "__main__":\n    main()\n', "\n")
_C7 = {"__file__": os.path.join(_HERE, "cs057_paisaje_completo.py"), "__name__": "cs057_mod"}
exec(compile(_s, "cs057_paisaje_completo.py", "exec"), _C7)
_confin = _C7["_confin"]; _em = _C7["_em"]; _debil = _C7["_debil"]; _despliegue = _C7["_despliegue"]
_diam = _C7["_diam"]; _giant = _C7["_giant"]; _colores = _C7["_colores"]; _clasifica = _C7["_clasifica"]
_construye_ensemble = _C7["_construye_ensemble"]; _alc_a_dmax = _C7["_alc_a_dmax"]
_sobol_puntos = _C7["_sobol_puntos"]; _punto_fisico = _C7["_punto_fisico"]; _denso_fisico = _C7["_denso_fisico"]
_T_de_paso = _C7["_T_de_paso"]; CLASES = _C7["CLASES"]
R_GRAV = _C7["R_GRAV"]; R_STRONG = _C7["R_STRONG"]; R_EM = _C7["R_EM"]; R_WEAK = _C7["R_WEAK"]; R_EXP = _C7["R_EXP"]
ALPHA = _C7["ALPHA"]; T_CONF = _C7["T_CONF"]; STEPS = _C7["STEPS"]

# ============================ CONFIG ============================
N_POINTS = int(os.environ.get("CS062_POINTS", 2048))     # 2^11 (con 3 brazos, cabe overnight)
SEEDS    = int(os.environ.get("CS062_SEEDS", 8))
WORKERS  = int(os.environ.get("CS062_WORKERS", max(1, (os.cpu_count() or 4) - 2)))
OUT      = os.environ.get("CS062_OUT", os.path.join(_HERE, "cs062_paisaje_peso.csv"))
SMOKE    = os.environ.get("CS062_SMOKE", "") != ""
DIM_DENSO = int(os.environ.get("CS062_DENSO", 128))
MASA_MIN, MASA_MAX = 1.0, 3477.0                          # rango leptónico (log-uniforme)
ACOPLES = ["peso", "grado", "null_peso"]
# ===============================================================


def _masa(N, rng):
    return np.exp(rng.uniform(math.log(MASA_MIN), math.log(MASA_MAX), N))


def _grav_peso(adj, N, rng, rate, dmax, T, w):
    """Gravedad ∝ w_i·w_j/d^ALPHA (Newton): fuente ∝ w, blanco ∝ w[j]/d^ALPHA. w = peso intrínseco o grado."""
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


def proceso062(adj0, N, color0, carga0, masa, W, dmax_grav, acople, rng):
    """Motor de CS057 (sync) con la GRAVEDAD acoplada a PESO intrínseco (Newton) según 'acople'. Devuelve la
    trayectoria del diámetro (para clasificar viable/expande/acelera con el MISMO criterio ciego de CS057)."""
    wg, ws, wem, wwk, wexp, wcool = W
    adj = [set(a) for a in adj0]; col = color0.copy(); car = carga0.copy()
    deg0 = [len(a) for a in adj]; t = np.zeros(N, dtype=np.int32)
    if acople == "null_peso":
        masa = masa.copy(); rng.shuffle(masa)
    CAP_E = 12 * N; D = []; G = []
    fuerzas = ("grav", "strong", "em", "weak")
    for step in range(STEPS):
        T = _T_de_paso(step, wcool)
        E = sum(len(a) for a in adj) // 2
        if E < 2 or E > CAP_E:
            D.append(_diam(adj, N)); G.append(_giant(adj, N)); break
        # gravedad: peso intrínseco (peso/null_peso) o grado (control = CS057)
        w = np.array([len(a) for a in adj], float) if acople == "grado" else masa
        _grav_peso(adj, N, rng, wg * R_GRAV, dmax_grav, T, w)
        if T < T_CONF:
            _confin(adj, N, col, t, rng, ws * R_STRONG)
        _em(adj, N, car, deg0, rng, wem * R_EM)
        _debil(N, col, car, rng, wwk * R_WEAK)
        _despliegue(adj, N, rng, wexp * R_EXP)
        dd = _diam(adj, N); gg = _giant(adj, N); D.append(dd); G.append(gg)
        if gg >= 0.9 and dd <= 2:
            break
    return D, G


def _worker(arg):
    pid, W7, phys = arg
    W = tuple(float(x) for x in W7[:6]); dmax = _alc_a_dmax(float(W7[6]))
    ens = _construye_ensemble()
    filas = []
    corr_ok = 1
    for seed in range(SEEDS):
        for acople in ACOPLES:
            base = dict(point_id=pid, w_grav=W[0], w_strong=W[1], w_em=W[2], w_weak=W[3], w_exp=W[4],
                        w_cool=W[5], alc=float(W7[6]), dmax_grav=dmax, seed=seed, acople=acople, phys=phys)
            for ci, (nom, (adj, N)) in enumerate(ens):
                masa = _masa(N, np.random.default_rng(seed * 991 + ci * 7 + 1))
                deg = np.array([len(a) for a in adj], float)
                if deg.std() > 0 and masa.std() > 0 and abs(float(np.corrcoef(masa, deg)[0, 1])) > 0.15:
                    corr_ok = 0
                col = _colores(N, np.random.default_rng(seed * 131 + ci * 17 + 1))
                car = (np.arange(N) % 2).astype(np.int8)
                np.random.default_rng(seed * 977 + ci * 29 + 7).shuffle(car)
                D, G = proceso062(adj, N, col, car, masa, W, dmax, acople,
                                  np.random.default_rng(seed * 31 + ci * 101 + hash(acople) % 97 + 13))
                m = _clasifica(D, G)
                base[f"viable_{nom}"] = m["viable"]; base[f"acelera_{nom}"] = m["acelera"]
            base["corr_ok"] = corr_ok
            filas.append(base)
    return filas


def _campos():
    c = ["point_id", "w_grav", "w_strong", "w_em", "w_weak", "w_exp", "w_cool", "alc", "dmax_grav",
         "seed", "acople", "phys"]
    for cl in CLASES:
        c += [f"viable_{cl}", f"acelera_{cl}"]
    return c + ["corr_ok"]


def main():
    print("CS062 — PAISAJE con GRAVEDAD ∝ PESO-INTRÍNSECO (Newton m·m/d²) vs GRADO (=CS057) vs NULL-PESO", flush=True)
    print("=" * 100, flush=True)
    print("PREDICCIÓN CIEGA: si el proxy de grado sesgaba contra el 3D (CS060-B), el brazo PESO dará 3D/4D más", flush=True)
    print("  viable que GRADO en TODO el mapa; y NULL-PESO dirá si es la masa o solo la independencia-del-grado.", flush=True)
    npts = 8 if SMOKE else N_POINTS
    sob = _sobol_puntos(npts, seed=2062)
    puntos = [(i, sob[i], 0) for i in range(len(sob))]
    pid = len(puntos); puntos.append((pid, _punto_fisico(), 1)); pid += 1
    if not SMOKE:
        for p in _denso_fisico(DIM_DENSO):
            puntos.append((pid, p, 2)); pid += 1
    print(f"puntos={len(puntos)} · seeds={SEEDS} · acoples={ACOPLES} · corridas≈{len(puntos)*SEEDS*len(ACOPLES)} · workers={WORKERS}", flush=True)
    print(f"salida {OUT}", flush=True)

    hechos = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for row in csv.DictReader(f):
                hechos.add(int(row["point_id"]))
        print(f"REANUDANDO: {len(hechos)} puntos ya hechos", flush=True)
    pend = [p for p in puntos if p[0] not in hechos]
    campos = _campos()
    fout = open(OUT, "a", newline=""); wr = csv.DictWriter(fout, fieldnames=campos)
    if not hechos:
        wr.writeheader()
    t0 = time.time(); n = 0
    import multiprocessing as mp
    if WORKERS > 1 and not SMOKE:
        with mp.Pool(WORKERS) as pool:
            for filas in pool.imap_unordered(_worker, pend, chunksize=1):
                for fila in filas: wr.writerow(fila)
                fout.flush(); n += 1
                if n % 25 == 0 or n == len(pend):
                    dt = time.time() - t0; r = n / dt
                    print(f"  {n}/{len(pend)} · {dt/60:.1f}min · {r*60:.1f}pt/min · ETA {(len(pend)-n)/r/3600:.2f}h", flush=True)
    else:
        for p in pend:
            for fila in _worker(p): wr.writerow(fila)
            fout.flush(); n += 1
            print(f"  {n}/{len(pend)} · {time.time()-t0:.1f}s", flush=True)
    fout.close()
    print(f"\nCOMPLETO: {n} puntos en {(time.time()-t0)/60:.1f} min → {OUT}", flush=True)


if __name__ == "__main__":
    main()
