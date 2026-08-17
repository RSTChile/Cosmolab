#!/usr/bin/env python3
"""
CG003 — Espacio relacional: ¿emerge el espacio de S>0 SIN dibujarlo?
====================================================================
Pregunta (Alexis, 1-jul-2026): ¿puede la geometria —distancia, dimension—
SALIR de S>0 como consecuencia de la estructura relacional del acoplamiento,
sin que le asignemos coordenadas?

REGLA DE ORO: cero coordenadas. No hay px,py,pz, ni grilla, ni caja.
El UNICO objeto es la matriz de acoplamiento C_ij (quien se relaciona con quien).
  · El espacio ES la trama de C (el "con-quien", C-N2).
  · La distancia = distancia relacional en el grafo (pasos de acoplamiento).
  · La dimension = se MIDE (dimension espectral), NUNCA se asigna.
Si una coordenada aparece en el setup, hicimos trampa. No aparece.

Ingredientes (cada uno = un nodo de la Teoria, ninguno inyecta dimension):
  · Identidad relacional: la identidad de i ES su fila C_i (su perfil de con-quien).
  · Acoplamiento por perfil compartido: C_ij crece si i,j tienen perfiles ALINEADOS
    (relacionarse parecido con el mundo -> relacionarse entre si). C-N2, cierre triadico.
  · Antipodal: perfiles OPUESTOS se cancelan (C-N2.5.10).
  · Economia: presupuesto FINITO de acoplamiento por diferencia (bioenergetica,
    "no gastar lo no usado") -> la trama se RALA -> LOCALIDAD (necesaria para dim finita).
  · Extincion: acoplamientos debiles mueren; S_i < kappa_s muere. Filtro C-N1.1 / C-N1.3.
Esta es la MISMA arquitectura que le dio SI al experimento de la triada en ANIMA
(iteracion + filtro economico sobre relaciones -> estructura desde inicio simetrico,
falsada al barajar la historia). Aca la estructura emergente es la DIMENSION.

Diagnosticos que NO se pueden falsear (las lecciones del arco):
  · N-independencia: una dimension REAL no depende de N (como el grano κ_Δ era N-indep).
    Si d_s sigue a N -> artefacto. Si converge -> seleccionada de verdad.
  · Multi-semilla: ¿d_s constante (LEY) o varia entre semillas (HISTORIA = multiverso
    de dimensiones, cada cosmos elige la suya)? Igual que direccion-vs-constantes del arco.
  · Shuffle null: barajar la trama superviviente (misma densidad, aristas al azar) ->
    la geometria debe DESAPARECER (d_s -> alta/inf, como un grafo azaroso). Si no cambia,
    la "geometria" era artefacto.

Resultado posible y HONESTO de antemano:
  · d_s finita + N-independiente + destruida por shuffle  -> EL ESPACIO EMERGE de S>0,
    con dimension SELECCIONADA (no diseñada).
  · d_s -> alta / crece con N / sobrevive al shuffle      -> NO hay geometria: negativo
    honesto (la dinamica actual no teje espacio; sabremos que ingrediente falta).

USO:
  python3 cg003_espacio_relacional.py --quick
  python3 cg003_espacio_relacional.py            # barrido completo
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).parent
SWEEP_CSV = OUT_DIR / "cg003_espacio_relacional.csv"

# --- parametros de la dinamica (baseline; ninguno es una dimension) ---
ETA, MU, KAPPA_S, S0, S_BAND = 0.05, 0.01, 1e-6, 1.0, 8.0
ALPHA = 1.0
BUDGET = 6.0        # presupuesto de acoplamiento positivo por diferencia (economia -> localidad)
KDEG = 8            # economia de GRADO: cada diferencia conserva a lo sumo k vecinos fuertes
PRUNE = 0.02        # umbral de extincion de acoplamientos debiles
INIT_P = 0.15       # densidad de la trama inicial (simetrica, azarosa, SIN dimension)
PASOS = 160
EDGE_THRESH = 0.02  # una arista del ESPACIO = acoplamiento positivo por encima de esto


def sat(S):
    return S / (1.0 + S / S_BAND)


def init_web(N, rng, p=INIT_P):
    """Arranque simetrico, azaroso, SIN dimension: acoplamientos ralos aleatorios."""
    C = rng.normal(0.0, 1.0, (N, N))
    C = 0.5 * (C + C.T)
    mask = rng.random((N, N)) < p
    mask = mask | mask.T
    C = C * mask
    np.fill_diagonal(C, 0.0)
    return C


def paso(C, S, rng, *, eta=ETA, mu=MU, kappa_s=KAPPA_S, budget=BUDGET, prune=PRUNE, alpha=ALPHA):
    """Un paso: la trama define identidades, las identidades reescriben la trama; economia + extincion."""
    alive = S > kappa_s
    m = np.where(alive, np.sqrt(sat(S)), 0.0)

    # identidad relacional = perfil de con-quien (fila normalizada)
    norm = np.linalg.norm(C, axis=1, keepdims=True)
    P = C / np.where(norm > 1e-12, norm, 1.0)
    Sim = P @ P.T                      # cos de perfiles en [-1,1]: alineados vs opuestos
    np.fill_diagonal(Sim, 0.0)

    # flujo de acoplamiento: perfiles alineados se refuerzan; opuestos se cancelan (antipodal)
    C = C + eta * np.outer(m, m) * Sim
    np.fill_diagonal(C, 0.0)
    C = 0.5 * (C + C.T)                # simetrica

    # economia: presupuesto de acoplamiento POSITIVO por diferencia -> ralea la trama
    rowpos = np.maximum(C, 0.0).sum(axis=1, keepdims=True)
    scale = np.where(rowpos > budget, budget / (rowpos + 1e-12), 1.0)
    C = C * np.minimum(scale, scale.T)  # escalado simetrico

    # extincion de acoplamientos debiles
    C = np.where(np.abs(C) < prune, 0.0, C)

    # economia de GRADO: cada diferencia conserva solo sus KDEG acoplamientos positivos mas fuertes
    # (fuerza localidad de NUMERO -> sin esto la refuerzo forma cliques densos, no espacio)
    pos = np.maximum(C, 0.0)
    if pos.shape[0] > KDEG:
        keep = np.zeros_like(C, dtype=bool)
        kth = np.argsort(-pos, axis=1)[:, :KDEG]
        rows_ix = np.repeat(np.arange(C.shape[0]), KDEG)
        keep[rows_ix, kth.ravel()] = True
        keep = keep | keep.T                # una arista sobrevive si cualquiera de los dos la eligio
        C = np.where(keep, C, 0.0)

    # persistencia: crece quien pertenece a un vecindario coherente (coop positiva)
    coop = np.maximum(C, 0.0) @ m
    dS = alpha * eta * m * coop - alpha * eta * m * m
    S = S * (1 - mu) + dS
    S = np.clip(S, 0.0, 1e6)
    S = np.where(S > kappa_s, S, 0.0)   # extincion de diferencias
    return C, S


def correr(N, seed, pasos=PASOS):
    rng = np.random.default_rng(seed)
    C = init_web(N, rng)
    S = np.full(N, S0)
    for _ in range(pasos):
        C, S = paso(C, S, rng)
    alive = S > KAPPA_S
    return C, alive


# ---------- medicion de la geometria (coordinate-free) ----------
def _componentes(A):
    """Componentes conexas de una adyacencia 0/1 (BFS). Devuelve etiquetas por nodo."""
    n = len(A)
    lab = -np.ones(n, dtype=int)
    c = 0
    for s in range(n):
        if lab[s] != -1:
            continue
        stack = [s]
        lab[s] = c
        while stack:
            u = stack.pop()
            vs = np.where((A[u] > 0) & (lab == -1))[0]
            for v in vs:
                lab[v] = c
                stack.append(int(v))
        c += 1
    return lab, c


def dimension_espectral(C, alive, thresh=EDGE_THRESH):
    """
    Dimension espectral de la trama de acoplamientos POSITIVOS (el espacio).
    P_ret(t) = <exp(-λ t)> ~ t^{-d_s/2}  ->  d_s = -2 * pendiente(log P vs log t).
    Todo se lee del grafo: cero coordenadas.
    """
    idx = np.where(alive)[0]
    if len(idx) < 20:
        return dict(d_s=np.nan, d_s_dos=np.nan, ncomp=0, giant=0, meandeg=0.0, n_surv=len(idx))
    Cs = C[np.ix_(idx, idx)]
    A = ((Cs > thresh) | (Cs.T > thresh)).astype(float)
    np.fill_diagonal(A, 0.0)

    lab, ncomp = _componentes(A)
    # componente gigante
    sizes = np.bincount(lab)
    g = int(np.argmax(sizes))
    gi = np.where(lab == g)[0]
    giant = len(gi)
    if giant < 20:
        return dict(d_s=np.nan, d_s_dos=np.nan, ncomp=ncomp, giant=giant, meandeg=float(A.sum() / max(len(idx), 1)), n_surv=len(idx))
    Ag = A[np.ix_(gi, gi)]
    deg = Ag.sum(1)
    meandeg = float(deg.mean())
    L = np.diag(deg) - Ag
    lam = np.linalg.eigvalsh(L)
    lam = np.sort(np.real(lam))
    lam = lam[lam > 1e-9]              # descartar modo cero
    if len(lam) < 10:
        return dict(d_s=np.nan, d_s_dos=np.nan, ncomp=ncomp, giant=giant, meandeg=meandeg, n_surv=len(idx))

    # (1) probabilidad de retorno
    tmin, tmax = 1.0 / lam[-1], 1.0 / lam[0]
    ts = np.logspace(np.log10(tmin * 2), np.log10(tmax / 2), 40)
    P = np.array([np.mean(np.exp(-lam * t)) for t in ts])
    ok = (P > 1e-4) & (P < 0.9)
    d_s = np.nan
    if ok.sum() >= 6:
        lx, ly = np.log(ts[ok]), np.log(P[ok])
        sl = np.polyfit(lx, ly, 1)[0]
        d_s = -2.0 * sl
    # (2) conteo de autovalores: N(λ) ~ λ^{d/2} para λ chico (chequeo cruzado)
    k = max(10, len(lam) // 3)
    lam_lo = lam[:k]
    rank = np.arange(1, k + 1) / len(lam)
    d_s_dos = 2.0 * np.polyfit(np.log(lam_lo), np.log(rank), 1)[0]
    return dict(d_s=float(d_s), d_s_dos=float(d_s_dos), ncomp=ncomp, giant=giant, meandeg=meandeg, n_surv=len(idx))


def shuffle_null(C, alive, rng, thresh=EDGE_THRESH):
    """Null: misma densidad de aristas pero colocadas AL AZAR (rompe la historia relacional)."""
    idx = np.where(alive)[0]
    if len(idx) < 20:
        return dict(d_s=np.nan, giant=0)
    Cs = C[np.ix_(idx, idx)]
    A = ((Cs > thresh) | (Cs.T > thresh)).astype(float)
    np.fill_diagonal(A, 0.0)
    n = len(idx)
    E = int(A.sum() / 2)
    # grafo azaroso con el mismo numero de aristas
    R = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    pick = rng.choice(len(iu[0]), size=min(E, len(iu[0])), replace=False)
    R[iu[0][pick], iu[1][pick]] = 1.0
    R = R + R.T
    lab, ncomp = _componentes(R)
    sizes = np.bincount(lab); g = int(np.argmax(sizes)); gi = np.where(lab == g)[0]
    if len(gi) < 20:
        return dict(d_s=np.nan, giant=len(gi))
    Ag = R[np.ix_(gi, gi)]; deg = Ag.sum(1); L = np.diag(deg) - Ag
    lam = np.sort(np.real(np.linalg.eigvalsh(L))); lam = lam[lam > 1e-9]
    if len(lam) < 10:
        return dict(d_s=np.nan, giant=len(gi))
    tmin, tmax = 1.0 / lam[-1], 1.0 / lam[0]
    ts = np.logspace(np.log10(tmin * 2), np.log10(tmax / 2), 40)
    P = np.array([np.mean(np.exp(-lam * t)) for t in ts]); ok = (P > 1e-4) & (P < 0.9)
    d_s = np.nan
    if ok.sum() >= 6:
        d_s = -2.0 * np.polyfit(np.log(ts[ok]), np.log(P[ok]), 1)[0]
    return dict(d_s=float(d_s), giant=len(gi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    Ns = [300, 500] if args.quick else [300, 600, 1000]
    seeds = range(1, 4) if args.quick else range(1, 9)
    pasos = 120 if args.quick else PASOS

    t0 = time.monotonic()
    rows = []
    print("CG003 — espacio relacional (cero coordenadas)\n" + "=" * 60)
    print(f"{'N':>5} {'seed':>4} {'n_surv':>7} {'giant':>6} {'ncomp':>6} {'meandeg':>8} {'d_s':>7} {'d_s(2)':>7} {'d_s_shuf':>9}")
    for N in Ns:
        for seed in seeds:
            C, alive = correr(N, seed, pasos)
            g = dimension_espectral(C, alive)
            sh = shuffle_null(C, alive, np.random.default_rng(seed + 777))
            rows.append(dict(N=N, seed=seed, **g, d_s_shuffle=sh["d_s"]))
            print(f"{N:>5} {seed:>4} {g['n_surv']:>7} {g['giant']:>6} {g['ncomp']:>6} "
                  f"{g['meandeg']:>8.2f} {g['d_s']:>7.2f} {g['d_s_dos']:>7.2f} {sh['d_s']:>9.2f}", flush=True)

    import csv
    with open(SWEEP_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # sintesis
    print("\n" + "=" * 60)
    def med(key, N=None):
        v = [r[key] for r in rows if (N is None or r["N"] == N) and r[key] == r[key]]
        return (np.mean(v), np.std(v)) if v else (np.nan, np.nan)
    print("Dimension espectral d_s por N (media ± std):")
    for N in Ns:
        mu_, sd_ = med("d_s", N)
        print(f"  N={N:>4}: d_s = {mu_:.2f} ± {sd_:.2f}")
    mall, sall = med("d_s")
    msh, _ = med("d_s_shuffle")
    print(f"\nN-independencia: {'SI (converge)' if sall < 0.5*max(mall,1e-9)+0.6 else 'revisar (varia con N)'}")
    print(f"Shuffle: d_s real medio = {mall:.2f}  vs  shuffle = {msh:.2f}  "
          f"({'SEPARA (geometria real)' if (msh!=msh) or msh > mall + 1.0 else 'NO separa (revisar)'})")
    print(f"\nCSV: {SWEEP_CSV}   Tiempo: {time.monotonic()-t0:.1f}s")


if __name__ == "__main__":
    main()
