"""
CS092 — GUARDAS del control positivo (se reportan las cuatro, salgan como salgan)
================================================================================================
G1  IDENTIDAD ALGEBRAICA de J-A: pi_local(r) = |S(r)|/(2r). En una retícula d-dimensional
    |S(r)| ~ r^(d-1), luego pi(r) ~ r^(d-2): CONSTANTE si y solo si d=2. Se mide el perfil real
    en retículas 1D/2D/3D/4D y se ajusta el exponente. Si el exponente sale ~ d-2, J-A no mide
    "planitud/orden global": mide BIDIMENSIONALIDAD.
G1b HORIZONTE de la via Q: L=8 en Q._K_y_Dq. Se mide, por sustrato, que fraccion de pares queda
    alcanzable dentro de L pasos. Si un sustrato metrico deja <30% alcanzable, diam_q_robusto
    devuelve NaN POR CONSTRUCCION -- el juez no puede dar un numero distinto.
G2  ¿EL NUMERO PODIA SALIR DISTINTO? Para J-B: pendiente esperada 1/d en una retícula d-dim; se
    tabula contra el umbral 0.3. Para J-C: se construye un caso PLANTADO (nodos sobre k ejes
    ortogonales exactos) y se verifica que cuenta_ejes_gap+picado_por_nodo SI pueden certificar.
G3  BUG _diam: se comparan H._diam_robusto y C90.diam_gigante en todos los sustratos.
G4  El BARAJADO debe destruir algo medible y no ser isomorfo al real: clustering, triangulos,
    solapamiento de aristas, secuencia de grados.
"""
from __future__ import annotations
import sys, csv, json, math, itertools
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs092_control_positivo_orden_global as C91
import cs069_quantum_graph as Q
import cs067_habitacion_completa as H
import cs068_paso2_mundo_ab as P2

RNG = np.random.default_rng


def g1_identidad_pi():
    print("\n" + "=" * 100)
    print("G1 — IDENTIDAD ALGEBRAICA de J-A:  pi(r)=|S(r)|/(2r) ~ r^(d-2)  =>  constante <=> d=2")
    print("=" * 100)
    casos = {1: C91.sub_anillo1d(2500, 1), 2: C91.sub_reticula2d(2500, 1),
             3: C91.sub_reticula3d(2197, 1), 4: C91.sub_reticula4d(2401, 1)}
    filas = []
    print(f"{'d':>2} {'N':>6} {'exp.ajustado(pi~r^a)':>22} {'d-2 (predicho)':>16} {'pi_cv (J-A)':>12} {'J-A detecta?':>13}")
    for d, (adj, N) in casos.items():
        D = C91._dist_bfs(adj, N)
        rng = RNG(1234)
        fuentes = [int(s) for s in rng.integers(0, N, size=min(8, N))]
        maxr = int(np.nanmax(D))
        rs, pis, cas = [], [], []
        for r in range(1, maxr + 1):
            cnt = float(np.mean([np.sum(D[s] == r) for s in fuentes]))
            if cnt > 0:
                rs.append(r); cas.append(cnt); pis.append(cnt / (2.0 * r))
                filas.append(dict(d=d, N=N, r=r, cascaron=cnt, pi=cnt / (2.0 * r)))
        # ajuste log-log sobre el tramo intermedio (evita r muy chico y el borde/periodicidad)
        rs_a = np.array(rs, float); pis_a = np.array(pis, float)
        lo, hi = max(1, int(0.15 * len(rs))), int(0.55 * len(rs))
        if hi - lo < 3:
            lo, hi = 0, len(rs)
        x = np.log(rs_a[lo:hi]); y = np.log(pis_a[lo:hi])
        a = float(np.polyfit(x, y, 1)[0])
        _, cv = Q.cedazo_pi(D, N, RNG(3))
        det = "SI" if cv < C91.UMBRAL_PI_CV else "no"
        print(f"{d:>2} {N:>6} {a:>22.3f} {d-2:>16d} {cv:>12.3f} {det:>13}")
    with open(f"{_HERE}/cs092_guarda1_perfil_pi.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["d", "N", "r", "cascaron", "pi"]); w.writeheader(); w.writerows(filas)
    print("perfil completo -> cs092_guarda1_perfil_pi.csv")


def g1b_horizonte_L():
    print("\n" + "=" * 100)
    print("G1b — HORIZONTE de la via Q: L=8 en Q._K_y_Dq. ¿que fraccion de pares alcanza la integral de camino?")
    print("     (diam_q_robusto exige >30% de la fila finita; por debajo devuelve NaN POR CONSTRUCCION)")
    print("=" * 100)
    print(f"{'sustrato':>15} {'N':>6} {'diam_BFS':>9} {'frac_pares_alcanzados(L=8)':>27} {'diam_q':>9} {'medible?':>9}")
    filas = []
    for nom in ["reticula2d", "reticula3d", "anillo1d", "aniso2d", "flujo_capas", "er", "real"]:
        fn, esc, _ = C91.SUSTRATOS[nom]
        adj, N = fn(esc[-1], 91001)
        D = C91._dist_bfs(adj, N)
        diam_bfs = float(np.nanmax(D))
        Dq = Q.brazo_completo(adj, N, RNG(91002))
        frac = float(np.isfinite(Dq).sum() / (N * N))
        dq = Q.diam_q_robusto(Dq, N, RNG(3))
        print(f"{nom:>15} {N:>6} {diam_bfs:>9.0f} {frac:>27.3f} {dq:>9.2f} {'si' if np.isfinite(dq) else 'NO':>9}")
        filas.append(dict(sustrato=nom, N=N, diam_bfs=diam_bfs, frac_alcanzada_L8=frac,
                          diam_q=None if not np.isfinite(dq) else dq))
        del Dq, D
    json.dump(filas, open(f"{_HERE}/cs092_guarda1b_horizonte.json", "w"), indent=1)


def g2_podia_salir_distinto():
    print("\n" + "=" * 100)
    print("G2 — ¿EL NUMERO PODIA SALIR DISTINTO?")
    print("=" * 100)
    print("J-B: en una retícula d-dim, diam ~ N^(1/d) => pendiente log-log = 1/d. Umbral pre-inscrito 0.3:")
    for d in range(1, 7):
        p = 1.0 / d
        print(f"   d={d}: pendiente esperada={p:.3f}  ->  {'PASA' if p > 0.3 else 'REPRUEBA (orden metrico real declarado ausente)'}")
    print("\nJ-C: control PLANTADO -- nodos exactamente sobre k ejes ortogonales (dominios discretos reales).")
    for k in [2, 3, 5]:
        N = 1200
        rng = RNG(7)
        V = np.zeros((N, 8))
        lab = rng.integers(0, k, size=N)
        for i in range(N):
            V[i, lab[i]] = 1.0 + 0.02 * rng.normal()
        Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        T = (Vn.T @ Vn) / N
        ev = np.linalg.eigvalsh(T)[::-1]
        n_ejes, PR, gap, r_thr = H.cuenta_ejes_gap(ev)
        pico, frac = H.picado_por_nodo(V)
        print(f"   k={k} ejes plantados -> n_ejes={n_ejes} PR={PR:.2f} gap={gap:.1f} pico_medio={pico:.3f} "
              f"certificado={pico > 0.85}")
    print("   (si el plantado certifica, J-C SI puede dar un numero distinto: el criterio es alcanzable)")


def g3_diam():
    print("\n" + "=" * 100)
    print("G3 — BUG _diam: H._diam_robusto vs C90.diam_gigante (componente gigante) en todos los sustratos")
    print("=" * 100)
    import cs090_diam_corregido as C90
    print(f"{'sustrato':>15} {'N':>6} {'diam_robusto':>13} {'diam_gigante':>13} {'coincide?':>10}")
    for nom, (fn, esc, _) in C91.SUSTRATOS.items():
        adj, N = fn(esc[-1], 91001)
        dr = H._diam_robusto(adj, N, RNG(4))
        dg = float(C90.diam_gigante(adj, N))
        print(f"{nom:>15} {N:>6} {dr:>13.1f} {dg:>13.1f} {('si' if abs(dr-dg) < 1e-9 else 'NO'):>10}")


def g4_barajado():
    print("\n" + "=" * 100)
    print("G4 — el BARAJADO debe destruir algo medible y NO ser isomorfo al real")
    print("=" * 100)
    for N in [900, 2500]:
        adj, _ = C91.sub_real(N, 91001)
        adj2, _ = C91.sub_real_barajado(N, 91001)
        e1 = {(i, j) for i in range(N) for j in adj[i] if i < j}
        e2 = {(i, j) for i in range(N) for j in adj2[i] if i < j}
        g1 = sorted(len(a) for a in adj); g2 = sorted(len(a) for a in adj2)
        c1, c2 = C91._clustering_global(adj, N), C91._clustering_global(adj2, N)
        t1, t2 = C91._n_triangulos(adj, N), C91._n_triangulos(adj2, N)
        print(f"  N={N}:  aristas real={len(e1)} barajado={len(e2)}  solapamiento="
              f"{len(e1 & e2)}/{len(e1)} = {100.0*len(e1 & e2)/max(len(e1),1):.1f}%")
        print(f"          secuencia de grados identica: {g1 == g2}   (config-model: DEBE ser True)")
        print(f"          clustering  real={c1:.4f} -> barajado={c2:.4f}   (caida {100*(1-c2/max(c1,1e-9)):.1f}%)")
        print(f"          triangulos  real={t1} -> barajado={t2}          (caida {100*(1-t2/max(t1,1)):.1f}%)")


if __name__ == "__main__":
    g1_identidad_pi()
    g2_podia_salir_distinto()
    g3_diam()
    g4_barajado()
    g1b_horizonte_L()
