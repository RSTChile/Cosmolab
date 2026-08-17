#!/usr/bin/env python3
"""
CG003-f — Planitud EMERGENTE: costo de holonomía bajo la 2ª ley
================================================================
Diagnóstico cerrado (CC + Claude Science, 2-jul-2026): cg003d es HIPERBÓLICO
(δ=0.00, diam~log N) y fragmentado (58–77% componente gigante). El ángulo local
(libre/cos_min) no basta; falta CONEXIÓN de curvatura controlada.

NO imponemos holonomía≈0 (eso sería cg003e = dibujar el plano). Seleccionamos:
  · Cada enlace cuesta exergía base C_LINK.
  · Cerrar un triángulo u–v–w con frustración de holonomía H cuesta extra
    LAMBDA_H · H² (curvatura cara, plano barato).
  · La exergía se inyecta en el frente (potencia→acto, cg003b) y SE ESPARCE por
    los enlaces existentes (2ª ley: difusión conservativa alto→bajo).
  · Solo se aceptan cross-links que el tejido puede PAGAR → la dinámica favorece
    lazos de baja holonomía sin coordenadas globales.

Mecánica heredada de cg003d: adj, dirs, libre(), enlazar(), cross-link por
dv = dirs[u][w] - d_new — pero el cross-link ahora compite por exergía.

Litmus (Claude Science): con LAMBDA_H=0 debe volver al régimen hiperbólico;
con LAMBDA_H alto y exergía activa, δ debe CRECER y diam-pend acercarse a ~0.5
(solo en ese régimen — no siempre plano).

USO: python3 cg003f_planitud_exergia.py [--quick]
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np

from cg003d_campo_angular import (
    crecer_campo,
    diametro,
    dimension_crecimiento,
    shuffle_adj,
)
from cg003_diagnostico_gromov import diagnos


# --- parámetros exergía (misma familia que cg003b) ---
C_LINK = 0.04          # costo base por enlace nuevo
INJECT = 0.08          # exergía al nacer un nodo (potencia→acto)
D_DIFF = 0.18          # difusión conservativa
GAMMA = 0.008          # disipación lenta (contador)
DIFFUSE_EVERY = 40     # cada cuántos nodos nuevos difundir


def _rand_unit(rng, Dtan):
    v = rng.normal(0.0, 1.0, Dtan)
    return v / (np.linalg.norm(v) + 1e-12)


def _ang(a, b):
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def _psi(d):
    """Ángulo del vector unitario en R^2 (Dtan=2); generaliza por atan2."""
    return float(np.arctan2(d[1], d[0]))


def cerrar_plano(duv, duw):
    dv = duw - duv
    nv = float(np.linalg.norm(dv))
    return dv / nv if nv > 1e-9 else None


def frustracion_transporte(phi, v, w, dv):
    """
    [DEGENERADA — conservada como registro del hallazgo de Msg 50, NO usar.]
    Holonomía de transporte por UNA sola arista. En R² es trivial por geometría:
    ψ(−dv) = ψ(dv) + π exacto, así que diff = wrap(φ[w] − φ[v]) y el término dv
    se CANCELA por completo → seleccionar entre 24 direcciones era un empate total
    (rango ~1e-16). Verificado por CC (álgebra) y por Claude Science (medición).
    La holonomía sólo puede vivir en un LAZO QUE ENCIERRA ÁREA → ver frustracion_ciclo.
    """
    theta_vw = phi[v] + _psi(dv)
    theta_wv = phi[w] + _psi(-dv)
    diff = (theta_wv - (theta_vw + np.pi) + np.pi) % (2 * np.pi) - np.pi
    return abs(float(diff))


def frustracion_ciclo(d_uv, d_uw, dv):
    """
    v1 — Holonomía del LAZO CERRADO u→v→w→u: defecto angular del triángulo,
    leído SÓLO de las direcciones reales de las aristas (sin coordenadas, sin φ).

    En R² plano la suma de los ángulos interiores de un triángulo es π; el defecto
    |Σángulos − π| es la curvatura ENCERRADA por el lazo (Gauss-Bonnet discreto).
    Ángulos interiores (entre las dos aristas que salen de cada vértice):
      · en u: entre u→v (d_uv) y u→w (d_uw)
      · en v: entre v→u (−d_uv) y v→w (dv)
      · en w: entre w→u (−d_uw) y w→v (−dv) = ∠(d_uw, dv)
    A diferencia del transporte por una arista, ESTO SÍ depende de dv (lo verifica
    el control _check_no_degenerada). Usa sólo ∠(·,·)=arccos(dot) → vale en cualquier Dtan.
    """
    ang_u = _ang(d_uv, d_uw)
    ang_v = _ang(-d_uv, dv)
    ang_w = _ang(d_uw, dv)
    return abs((ang_u + ang_v + ang_w) - np.pi)


def _check_no_degenerada(seed=0, Dtan=2, n_dir=24):
    """
    Control de Claude Science: la métrica de selección NO debe ser un empate.
    Barre n_dir direcciones candidatas sobre un triángulo fijo y exige que la
    holonomía VARÍE (std ≫ ruido de punto flotante). Si std ~1e-16, la selección
    volvería a ser un empate silencioso (el bug de v0 disfrazado).
    Devuelve (std, ok). Barato; se corre una vez antes de confiar en la selección.
    """
    rng = np.random.default_rng(seed)
    d_uv = _rand_unit(rng, Dtan)
    d_uw = _rand_unit(rng, Dtan)
    vals = [frustracion_ciclo(d_uv, d_uw, _rand_unit(rng, Dtan)) for _ in range(n_dir)]
    sd = float(np.std(vals))
    return sd, sd > 1e-6


def _pagar(e, nodes, cost):
    """Descuenta cost de exergía repartido entre nodos; False si no alcanza."""
    if cost <= 0:
        return True
    tot = sum(float(e[i]) for i in nodes)
    if tot < cost - 1e-12:
        return False
    for i in nodes:
        share = cost * float(e[i]) / (tot + 1e-12)
        e[i] = max(0.0, float(e[i]) - share)
    return True


def _difundir(e, adj, n_act):
    """2ª ley: exergía fluye por enlaces existentes (conservativo)."""
    flow = np.zeros(n_act, dtype=np.float64)
    for u in range(n_act):
        eu = float(e[u])
        for w in adj[u]:
            if w <= u or w >= n_act:
                continue
            ew = float(e[w])
            f = D_DIFF * (eu - ew)
            flow[u] -= f
            flow[w] += f
    e[:n_act] = np.clip(e[:n_act] + flow, 0.0, None)
    e[:n_act] = e[:n_act] * (1.0 - GAMMA)


def _frac_gigante(adj, N):
    best = 0
    for s in range(min(N, 32)):
        dist = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    q.append(int(w))
        best = max(best, len(dist))
    return best / N


def crecer_campo_exergia(
    N,
    Dtan=2,
    kdeg=8,
    cos_min=0.5,
    m_cross=2,
    lambda_H=1.0,
    seed=0,
):
    """
    cg003d + costo de holonomía pagado en exergía + difusión.
    lambda_H=0 → cross-links como cg003d (sin filtro exergético).
    """
    rng = np.random.default_rng(seed)
    adj = [set() for _ in range(N)]
    dirs = [dict() for _ in range(N)]
    e = np.zeros(N, dtype=np.float64)
    phi = np.zeros(N, dtype=np.float64)  # orientación de marco tangente (conexión)

    def libre(i, d, exclude=None):
        if len(adj[i]) >= kdeg:
            return False
        for w, dw in dirs[i].items():
            if w == exclude:
                continue
            if float(np.dot(d, dw)) > cos_min:
                return False
        return True

    def enlazar(i, j, d, cost_nodes, cost, set_phi_j=None):
        if not _pagar(e, cost_nodes, cost):
            return False
        adj[i].add(j)
        adj[j].add(i)
        dirs[i][j] = d
        dirs[j][i] = -d
        if set_phi_j is not None:
            phi[j] = set_phi_j % (2 * np.pi)
        return True

    # semilla plana local: simplex con direcciones desde pivote (sin pseudo-posiciones)
    s0 = Dtan + 1
    base = _rand_unit(rng, Dtan)
    perp = _rand_unit(rng, Dtan)
    perp = perp - np.dot(perp, base) * base
    perp = perp / (np.linalg.norm(perp) + 1e-12)
    seed_dirs = []
    for k in range(s0):
        th = 2.0 * np.pi * k / s0
        seed_dirs.append(np.cos(th) * base + np.sin(th) * perp)
    for i in range(s0):
        e[i] = INJECT
        for j in range(i + 1, s0):
            d = seed_dirs[j] - seed_dirs[i]
            d = d / (np.linalg.norm(d) + 1e-12)
            enlazar(i, j, d, [i, j], C_LINK, set_phi_j=(phi[i] + _psi(d)) % (2 * np.pi))

    frontier = [i for i in range(s0) if len(adj[i]) < kdeg]

    def pick():
        # favorecer nodos con más exergía (frente activo) para cohesionar el tejido
        while frontier:
            idx = int(rng.integers(len(frontier)))
            u = frontier[idx]
            if len(adj[u]) < kdeg:
                return u
            frontier[idx] = frontier[-1]
            frontier.pop()
        return None

    for v in range(s0, N):
        u = pick()
        if u is None:
            break

        e[v] = INJECT

        d_new = None
        for _ in range(16):
            cand = _rand_unit(rng, Dtan)
            if libre(u, cand):
                d_new = cand
                break
        if d_new is None:
            continue

        if not enlazar(
            u, v, d_new, [u, v], C_LINK,
            set_phi_j=(phi[u] + _psi(d_new)) % (2 * np.pi),
        ):
            continue

        cross = []
        for w, dw in dirs[u].items():
            if w == v:
                continue
            c = float(np.dot(dw, d_new))
            if c > 0.0:
                cross.append((c, w))
        cross.sort(reverse=True)

        added = 0
        for _, w in cross:
            if added >= m_cross:
                break
            if lambda_H <= 0:
                # baseline cg003d: cierre en marco de u
                dv = cerrar_plano(d_new, dirs[u][w])
                if dv is None or not libre(v, dv) or not libre(w, -dv):
                    continue
                if enlazar(v, w, dv, [u, v, w], C_LINK):
                    added += 1
                continue

            # λ_H>0: SELECCIONAR dirección v→w con mínima frustración de transporte
            # (no imponer cerrar_plano — competir entre candidatos libres)
            # v1: holonomía sobre el CICLO CERRADO u-v-w (depende de dv, no-degenerada)
            d_uw = dirs[u][w]
            cand = []
            dv0 = cerrar_plano(d_new, d_uw)
            if dv0 is not None and libre(v, dv0) and libre(w, -dv0):
                cand.append((frustracion_ciclo(d_new, d_uw, dv0), dv0))
            for _ in range(24):
                dtry = _rand_unit(rng, Dtan)
                if not libre(v, dtry) or not libre(w, -dtry):
                    continue
                cand.append((frustracion_ciclo(d_new, d_uw, dtry), dtry))
            if not cand:
                continue
            cand.sort(key=lambda x: x[0])
            H, dv = cand[0]
            cost = C_LINK + lambda_H * (H ** 2)
            if sum(float(e[i]) for i in (u, v, w)) < cost:
                continue
            if enlazar(v, w, dv, [u, v, w], cost):
                added += 1

        if len(adj[v]) < kdeg:
            frontier.append(v)
        if len(adj[u]) < kdeg:
            frontier.append(u)

        if v % DIFFUSE_EVERY == 0:
            _difundir(e, adj, v + 1)  # solo nodos ya creados

    adj_out = [np.fromiter(s, dtype=np.int32) for s in adj]
    return adj_out, float(e.sum()), _frac_gigante(adj, N)


def _pend_diam(Ns, dias):
    if len(Ns) < 2:
        return np.nan
    xs = np.log(np.array(Ns, float))
    ys = np.log(np.maximum(np.array(dias, float), 1.0))
    return float(np.polyfit(xs, ys, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    Ns = [1024, 4096] if args.quick else [1024, 4096, 16384]
    seeds = [1, 2] if args.quick else [1, 2, 3]
    lambdas = [0.0, 1.0, 4.0] if args.quick else [0.0, 0.5, 1.0, 2.0, 4.0]
    Dtans = [2, 3]
    K = 80 if args.quick else 120

    t0 = time.monotonic()
    print("CG003-f — planitud EMERGENTE (costo holonomía + 2ª ley)")
    print("=" * 96)

    # Control de no-degeneración (Claude Science): la métrica de selección debe VARIAR.
    print("Pre-vuelo · control de no-degeneración de la métrica (ciclo cerrado):")
    for Dt in (2, 3):
        sd, ok = _check_no_degenerada(seed=0, Dtan=Dt)
        print(f"  Dtan={Dt}: std(24 dir)={sd:.3e}  ->  {'OK (orienta)' if ok else 'DEGENERADA (empate)'}")
        assert ok, f"MÉTRICA DEGENERADA en Dtan={Dt}: la selección sería un empate silencioso"
    print()
    print("λ_H=0 → baseline hiperbólico.  λ_H>0 → selección de lazos planos baratos.\n")
    hdr = f"{'λ_H':>5} {'Dt':>2} {'N':>6} {'sd':>2} {'%gig':>5} {'diam':>5} {'δ_med':>7} {'d_grow':>6} {'ver':>10}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for lam in lambdas:
        for Dt in Dtans:
            for N in Ns:
                dias, deltas, fracs = [], [], []
                for sd in seeds:
                    adj, ex_tot, fg = crecer_campo_exergia(
                        N, Dtan=Dt, kdeg=2 * Dt + 4, lambda_H=lam, seed=sd
                    )
                    dia = diametro(adj, N, seed=sd)
                    g = dimension_crecimiento(adj, N, seed=sd)
                    r = diagnos(f"cg003f_l{lam}_Dt{Dt}", adj, N, K, seed=sd + 11)
                    dmean = r["dmean"] if r["dmean"] == r["dmean"] else np.nan
                    rows.append((lam, Dt, N, sd, fg, dia, dmean, g, ex_tot))
                    dias.append(dia)
                    deltas.append(dmean)
                    fracs.append(fg)
                    print(
                        f"{lam:5.1f} {Dt:2d} {N:6d} {sd:2d} {fg*100:5.0f} {dia:5d} "
                        f"{dmean:7.2f} {g['d']:6.2f} {g['ver']:>10}",
                        flush=True,
                    )
                pend = _pend_diam(Ns[: len(dias)], dias) if len(dias) >= 2 else np.nan
                ddel = np.nanmean(deltas) if deltas else np.nan
                print(
                    f"  -> λ_H={lam} Dt={Dt}: diam-pend={pend:.2f}  δ_media~{ddel:.2f}  "
                    f"%gig~{100*np.mean(fracs):.0f}%"
                )

    print("\n" + "=" * 96)
    print("Baseline cg003d (λ_H=0 debe ≈ hiperbólico):")
    for Dt in Dtans:
        for N in Ns[:2]:
            adj = crecer_campo(N, Dtan=Dt, kdeg=2 * Dt + 4, seed=1)
            r = diagnos(f"cg003d_Dt{Dt}", adj, N, K, seed=7)
            print(
                f"  cg003d Dt={Dt} N={N}: %gig={r['fg']*100:.0f}% diam={r['diam']} "
                f"δ={r['dmean']:.2f}"
            )

    print("\nCriterio de ÉXITO (emergencia, no impuesto):")
    print("  · λ_H=0: δ≈0, diam-pend~0.1 (hiperbólico — control)")
    print("  · λ_H>0 en régimen: δ CRECE, diam-pend→0.5, %gig↑, shuffle destruye")
    print("  · Litmus: λ_H distintos dan lecturas DISTINTAS (no siempre plano)")
    print(f"\nTiempo: {time.monotonic()-t0:.1f}s")


if __name__ == "__main__":
    main()