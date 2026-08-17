#!/usr/bin/env python3
"""
CG003-b — Exergia primero: el espacio como estructura del flujo de exergia
==========================================================================
Vision de Alexis (desde los 14, su respuesta a "que es Dios para mi"):
la energia total es EL TODO (S>0 en potencia, normalizado a 1); un contador
de resta la convierte de potencia en ACTO (exergia); cada acto ocupa un poco
del Todo, limitado por si mismo contra la nada; el Universo es esa conversion,
hasta que la exergia se agota (fin) y el Todo vuelve a potencia (ciclo:
clausura dinamica). Dios = Todo (fuera, trascendente, la potencia conservada);
Universo = expresion (dentro, inmanente, el acto). Potencia / acto.

CLAVE que aporta la vision — lo que a CG003 le faltaba: la SEGUNDA LEY.
CG003 se agrumaba porque solo tenia ATRACCION (lo parecido se junta -> puntos).
El calor inicial SE ESPARCE (disipacion, entropia sube): el esparcir es la
fuerza que crea EXTENSION. El Big Bang enfriandose ES el anti-grumo.

Cero coordenadas. Estado: exergia e_i por modo + acoplamiento W (conductancia).
  · potencia = 1 - sum(e)   (el Todo conservado, normalizado a 1)
  · inyeccion: potencia -> acto (fluctuaciones que expresan; C-N2.5.5)
  · difusion (2a ley): exergia fluye alto->bajo por W (ESPARCE, conservativa)
  · disipacion: exergia degrada a no-disponible (el contador; flecha del tiempo)
  · W adapta por FLUJO (donde fluyo exergia se refuerza la conductancia),
    presupuesto de conductancia por modo (economia), poda de lo debil.
El ESPACIO = geometria del flujo (dim espectral de W). La DISTANCIA = resistencia
al flujo de exergia entre dos asimetrias. El TIEMPO = exergia convertida/gastada.
Falsadores: shuffle de W (la geometria debe desaparecer), N-independencia.

USO: python3 cg003b_exergia_primero.py [--quick]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

PASOS = 200
D_DIF = 0.20        # coeficiente de difusion (esparcir)
GAMMA = 0.02        # disipacion (el contador de resta; exergia -> no disponible)
INJECT = 0.06       # fraccion de la potencia que se expresa por paso (potencia->acto)
BUDGET = 4.0        # presupuesto de conductancia por modo (economia -> localidad)
KDEG = 6            # economia de GRADO: cada modo conserva sus k canales mas fuertes
PRUNE = 0.02
HEBB = 0.15         # cuanto refuerza el flujo a su conductancia
INIT_P = 0.12       # densidad inicial de conductancias (simetrica, azarosa, SIN dimension)
EDGE_THRESH = 0.05


def init_W(N, rng, p=INIT_P):
    W = np.abs(rng.normal(0, 1, (N, N)))
    W = 0.5 * (W + W.T)
    mask = rng.random((N, N)) < p
    mask = mask | mask.T
    W = W * mask
    np.fill_diagonal(W, 0.0)
    return W


def paso(e, W, rng):
    N = len(e)
    potencia = max(0.0, 1.0 - float(e.sum()))
    # 1. inyeccion: potencia -> acto (fluctuaciones)
    if potencia > 1e-9:
        k = max(1, N // 20)
        idx = rng.choice(N, size=k, replace=False)
        e[idx] += INJECT * potencia / k
    # 2. difusion (2a ley): flujo conservativo alto->bajo por W
    diff = e[:, None] - e[None, :]              # e_i - e_j (antisimetrica)
    F = D_DIF * W * diff                         # flujo i->j (sale de i si >0)
    net = F.sum(axis=1)                          # neto que sale de i (sum global = 0)
    e = np.clip(e - net, 0.0, None)
    # 3. disipacion: el contador (exergia degrada, monotono; flecha del tiempo)
    e = e * (1.0 - GAMMA)
    # 4. W adapta por FLUJO: donde fluyo exergia, refuerza conductancia
    W = W + HEBB * np.abs(F)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    rs = W.sum(axis=1, keepdims=True)           # presupuesto de conductancia (economia)
    sc = np.where(rs > BUDGET, BUDGET / (rs + 1e-12), 1.0)
    W = W * np.minimum(sc, sc.T)
    W = np.where(W < PRUNE, 0.0, W)
    # economia de GRADO: cada modo conserva solo sus KDEG canales mas fuertes (localidad de numero)
    if N > KDEG:
        keep = np.zeros_like(W, dtype=bool)
        kth = np.argsort(-W, axis=1)[:, :KDEG]
        keep[np.repeat(np.arange(N), KDEG), kth.ravel()] = True
        keep = keep | keep.T
        W = np.where(keep, W, 0.0)
    return e, W


def correr(N, seed, pasos=PASOS):
    rng = np.random.default_rng(seed)
    W = init_W(N, rng)
    e = np.zeros(N)
    hist_ex = []
    for _ in range(pasos):
        e, W = paso(e, W, rng)
        hist_ex.append(float(e.sum()))
    return W, e, hist_ex


# ---------- geometria (coordinate-free) ----------
def _componentes(A):
    n = len(A); lab = -np.ones(n, int); c = 0
    for s in range(n):
        if lab[s] != -1:
            continue
        st = [s]; lab[s] = c
        while st:
            u = st.pop()
            for v in np.where((A[u] > 0) & (lab == -1))[0]:
                lab[v] = c; st.append(int(v))
        c += 1
    return lab, c


def _d_espectral(A):
    lab, ncomp = _componentes(A)
    g = int(np.argmax(np.bincount(lab))); gi = np.where(lab == g)[0]
    if len(gi) < 20:
        return np.nan, ncomp, len(gi), 0.0
    Ag = A[np.ix_(gi, gi)]; deg = Ag.sum(1); L = np.diag(deg) - Ag
    lam = np.sort(np.real(np.linalg.eigvalsh(L))); lam = lam[lam > 1e-9]
    if len(lam) < 10:
        return np.nan, ncomp, len(gi), float(deg.mean())
    ts = np.logspace(np.log10(2.0 / lam[-1]), np.log10(0.5 / lam[0]), 40)
    P = np.array([np.mean(np.exp(-lam * t)) for t in ts]); ok = (P > 1e-4) & (P < 0.9)
    d_s = -2.0 * np.polyfit(np.log(ts[ok]), np.log(P[ok]), 1)[0] if ok.sum() >= 6 else np.nan
    return float(d_s), ncomp, len(gi), float(deg.mean())


def _bfs_shells(A, src, rmax=40):
    """Distancia relacional (en pasos) desde src a todos, por BFS."""
    n = len(A); dist = -np.ones(n, int); dist[src] = 0; frontier = [src]; d = 0
    while frontier and d < rmax:
        d += 1; nxt = []
        for u in frontier:
            nb = np.where((A[u] > 0) & (dist < 0))[0]
            dist[nb] = d; nxt.extend(nb.tolist())
        frontier = nxt
    return dist


def dimension_crecimiento(A, n_src=40, seed=0):
    """
    Dimension de CRECIMIENTO: N(r) = #{nodos a distancia relacional <= r}.
    Enrejado d-dim -> N(r) ~ r^d (ley de potencias).  Azar/mundo-pequeño -> N(r) ~ e^{br} (exponencial).
    Ajusta AMBOS modelos y compara: eso separa geometria de azar (lo que la dim espectral saturada no podia).
    """
    lab, _ = _componentes(A); g = int(np.argmax(np.bincount(lab))); gi = np.where(lab == g)[0]
    if len(gi) < 40:
        return dict(d=np.nan, r2_pot=np.nan, r2_exp=np.nan, veredicto="componente chica", rango=0)
    Ag = A[np.ix_(gi, gi)]; n = len(gi)
    rng = np.random.default_rng(seed)
    srcs = rng.choice(n, size=min(n_src, n), replace=False)
    rmax = 40; acc = np.zeros(rmax + 1); cnt = 0
    for s in srcs:
        dist = _bfs_shells(Ag, int(s), rmax); reach = dist[dist >= 0]
        if len(reach) < 10:
            continue
        acc += np.array([(reach <= r).sum() for r in range(rmax + 1)], float); cnt += 1
    if cnt == 0:
        return dict(d=np.nan, r2_pot=np.nan, r2_exp=np.nan, veredicto="sin fuentes", rango=0)
    Nr = acc / cnt
    # regimen de crecimiento: r>=1 hasta saturar (~mitad de la gigante) o dejar de crecer
    half = 0.5 * n; rs, ys = [], []
    for r in range(1, rmax + 1):
        if Nr[r] <= Nr[r - 1] + 1e-9:
            break
        rs.append(r); ys.append(Nr[r])
        if Nr[r] >= half:
            break
    if len(rs) < 4:
        return dict(d=np.nan, r2_pot=np.nan, r2_exp=np.nan, veredicto="poco rango", rango=len(rs))
    rs = np.array(rs, float); ys = np.log(np.array(ys, float)); ymean = ys.mean()
    # potencias: log N ~ d * log r
    cp = np.polyfit(np.log(rs), ys, 1); d = cp[0]
    r2p = 1 - np.sum((ys - np.polyval(cp, np.log(rs))) ** 2) / (np.sum((ys - ymean) ** 2) + 1e-12)
    # exponencial: log N ~ b * r
    ce = np.polyfit(rs, ys, 1)
    r2e = 1 - np.sum((ys - np.polyval(ce, rs)) ** 2) / (np.sum((ys - ymean) ** 2) + 1e-12)
    ver = "GEOMETRIA (potencias)" if r2p >= r2e else "AZAR (exponencial)"
    return dict(d=float(d), r2_pot=float(r2p), r2_exp=float(r2e), veredicto=ver, rango=len(rs))


def geometria(W, thresh=EDGE_THRESH):
    A = (W > thresh).astype(float); np.fill_diagonal(A, 0.0)
    d_s, ncomp, giant, meandeg = _d_espectral(A)
    gc = dimension_crecimiento(A)
    return dict(d_s=d_s, ncomp=ncomp, giant=giant, meandeg=meandeg, n=len(W),
                d_g=gc["d"], r2p=gc["r2_pot"], r2e=gc["r2_exp"], ver=gc["veredicto"])


def shuffle_adj(W, rng, thresh=EDGE_THRESH):
    """Grafo azaroso con la MISMA cantidad de aristas (null: rompe la historia)."""
    A = (W > thresh).astype(float); np.fill_diagonal(A, 0.0)
    n = len(A); E = int(A.sum() / 2); R = np.zeros((n, n)); iu = np.triu_indices(n, 1)
    if E > 0:
        pk = rng.choice(len(iu[0]), size=min(E, len(iu[0])), replace=False)
        R[iu[0][pk], iu[1][pk]] = 1.0; R = R + R.T
    return R


def shuffle_null(W, rng, thresh=EDGE_THRESH):
    A = (W > thresh).astype(float); np.fill_diagonal(A, 0.0)
    n = len(A); E = int(A.sum() / 2)
    R = np.zeros((n, n)); iu = np.triu_indices(n, 1)
    if E > 0:
        pk = rng.choice(len(iu[0]), size=min(E, len(iu[0])), replace=False)
        R[iu[0][pk], iu[1][pk]] = 1.0; R = R + R.T
    d_s, _, _, _ = _d_espectral(R)
    return d_s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    Ns = [300, 500] if args.quick else [300, 600, 1000]
    seeds = range(1, 4) if args.quick else range(1, 7)
    pasos = 150 if args.quick else PASOS

    t0 = time.monotonic()
    print("CG003-b — exergia primero · REGLA AFILADA (dimension de crecimiento)\n" + "=" * 74)
    print("real: N(r)~r^d = geometria ; N(r)~e^{br} = azar.  shuffle debe dar AZAR.")
    print(f"\n{'N':>5} {'seed':>4} {'giant':>6} {'deg':>5} | {'d_growth':>8} {'R2_pot':>7} {'R2_exp':>7} {'veredicto':>22} | {'shuf_d':>7} {'shuf_ver':>12}")
    rows = []
    for N in Ns:
        for sd in seeds:
            W, e, hist = correr(N, sd, pasos)
            g = geometria(W)
            R = shuffle_adj(W, np.random.default_rng(sd + 999))
            gs = dimension_crecimiento(R, seed=sd + 111)
            rows.append((N, sd, g, gs))
            print(f"{N:>5} {sd:>4} {g['giant']:>6} {g['meandeg']:>5.1f} | "
                  f"{g['d_g']:>8.2f} {g['r2p']:>7.3f} {g['r2e']:>7.3f} {g['ver']:>22} | "
                  f"{gs['d']:>7.2f} {('AZAR' if gs['veredicto'].startswith('AZAR') else 'geom?'):>12}", flush=True)

    print("\n" + "=" * 74)
    for N in Ns:
        dg = [g['d_g'] for (n, s, g, gs) in rows if n == N and g['d_g'] == g['d_g']]
        print(f"  N={N:>4}: d_growth = {np.mean(dg):.2f} ± {np.std(dg):.2f}" if dg else f"  N={N}: sin d")
    real_geo = sum(1 for (n, s, g, gs) in rows if g['ver'].startswith('GEOMETRIA'))
    shuf_azar = sum(1 for (n, s, g, gs) in rows if gs['veredicto'].startswith('AZAR'))
    tot = len(rows)
    print(f"\nVEREDICTO:")
    print(f"  trama real   : {real_geo}/{tot} dan GEOMETRIA (ley de potencias)")
    print(f"  shuffle null : {shuf_azar}/{tot} dan AZAR (exponencial)  [debe ser alto]")
    dg_all = [g['d_g'] for (n, s, g, gs) in rows if g['d_g'] == g['d_g']]
    stable = (np.std(dg_all) < 0.4) if dg_all else False
    if real_geo >= 0.7 * tot and shuf_azar >= 0.7 * tot and stable:
        print(f"  => EL ESPACIO EMERGE: dimension ~{np.mean(dg_all):.2f}, estable, real=potencias / shuffle=azar.")
    elif real_geo >= 0.7 * tot:
        print("  => real parece geometria, pero revisar shuffle / estabilidad antes de afirmar.")
    else:
        print("  => NO: la trama real tambien crece exponencial (mundo-pequeño). No hay geometria aun.")
    print(f"Tiempo: {time.monotonic()-t0:.1f}s")


if __name__ == "__main__":
    main()
