"""
CG004-c — ROBUSTIFICACIÓN del negativo (disciplina B-antes-de-A, pedido por CS)
==============================================================================
Antes de fijar "la obstrucción es global" o construir el mecanismo global encima, confirmar que
el negativo aguanta fuera del alcance débil (2 semillas, Dt=2, quick). CS pidió: Dt=3 y ≥8 semillas,
mismo arnés, y verificar que diam-pend y la convergencia de dimensión SIGUEN SIN separarse del control.

Enunciado a robustecer (redacción corregida por CS, sin "pared demostrada"):
  "El espacio plano NO emerge de crecimiento relacional local en la familia probada; la obstrucción
   aparece como GLOBAL, no local."  -> aunque haya triángulos abundantes (clu alto), sigue hiperbólico.

BRAZOS (los dos extremos que deciden):
  · ARBOL  = crecer(λ_H=2.0, cos_min=0.5)  -> árbol puro (clu~0), baseline hiperbólico
  · CICLOS = crecer(λ_H=0.0, cos_min=0.6, m_cross=8)  -> triángulos abundantes (clu~0.4), gate relajado
  · AZAR   = shuffle(CICLOS)  -> null
ANCLAS: lattice2D (plano: δ CRECE) y árbol_b3 (δ=0), para escala.

CRITERIO PRE-REGISTRADO: CICLOS "se separa hacia lo plano" SOLO si, robusto en 8 semillas y Dt∈{2,3}:
  δ_med CRECE con N · diam-pend → 1/Dt (0.5 en Dt2, 0.33 en Dt3) · dim CONVERGE · %gig alto · shuffle destruye.
Se reporta diam-pend por-semilla (media±std) y d_grow(N) media±std -> robustez explícita.

Reusa el arnés de cg004_attach.py. Se reanuda por CSV.
"""
from __future__ import annotations

import csv
import importlib.util
import os
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("cg004_attach", os.path.join(_HERE, "cg004_attach.py"))
_M = importlib.util.module_from_spec(_spec)
# evitar que corra su main() al importar
_src = open(os.path.join(_HERE, "cg004_attach.py")).read().replace("\nmain()\n", "\n")
exec(compile(_src, "cg004_attach.py", "exec"), _M.__dict__)

crecer = _M.crecer
diametro = _M.diametro; dimension_crecimiento = _M.dimension_crecimiento; diagnos = _M.diagnos
shuffle_adj = _M.shuffle_adj; lattice2d = _M.lattice2d; arbol = _M.arbol
_check_no_degenerada = _M._check_no_degenerada


# ============================ CONFIG ============================
LOG    = "cg004c_robusto"
SEEDS  = [1, 2, 3, 4, 5, 6, 7, 8]     # >=8 (pedido de CS)
NS     = [1024, 4096, 16384]
DTANS  = [2, 3]
K      = 120
# CICLOS (gate relajado, ciclos abundantes)
COSM_CICLOS = 0.6
MCROSS_CICLOS = 8
# ===============================================================


def clustering(adj, N, n=400, seed=0):
    S = [set(int(x) for x in a) for a in adj]
    active = [i for i in range(N) if S[i]]
    if not active:
        return 0.0
    rng = np.random.default_rng(seed)
    samp = rng.choice(active, size=min(n, len(active)), replace=False)
    cs = []
    for u in samp:
        nb = list(S[u])
        if len(nb) < 2:
            continue
        links = 0; tot = 0
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                tot += 1
                if nb[j] in S[int(nb[i])]:
                    links += 1
        if tot:
            cs.append(links / tot)
    return float(np.mean(cs)) if cs else 0.0


def _ev(adj_sets, N):
    act = sum(1 for s in adj_sets if s)
    E = sum(len(s) for s in adj_sets) // 2
    return (E / act) if act else 0.0


def _finalizar(adj_sets):
    return [np.fromiter(s, dtype=np.int32) for s in adj_sets]


def _medir(adj, N, sd):
    dia = diametro(adj, N, seed=sd)
    g = dimension_crecimiento(adj, N, seed=sd)
    r = diagnos(adj, N, K, seed=sd + 11)
    clu = clustering(adj, N, seed=sd + 3)
    return dia, g, r, clu


def _slope(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(xs[m]), np.log(np.maximum(ys[m], 1.0)), 1)[0])


def main():
    csv_path = f"{LOG}.csv"
    cols = ["brazo", "Dt", "N", "seed", "fg", "clu", "ev", "diam", "dmean", "d95", "dgrow", "ver"]
    t0 = time.time()

    print("CG004-c — ROBUSTIFICACIÓN (Dt∈{2,3}, 8 semillas) — ¿aguanta el negativo global?")
    print("=" * 100)
    print(f"CICLOS: λ_H=0 · cos_min={COSM_CICLOS} · m_cross={MCROSS_CICLOS} (gate relajado, triángulos abundantes)")

    print("\nPre-vuelo · no-degeneración de la métrica:")
    for Dt in DTANS:
        sd, ok = _check_no_degenerada(seed=0, Dtan=Dt)
        print(f"  Dtan={Dt}: std(24 dir)={sd:.3e} -> {'OK' if ok else 'DEGENERADA'}")
        assert ok

    print("\nAnclas (escala): lattice2D plano (δ CRECE) · árbol_b3 (δ=0):")
    print(f"  {'ancla':>10} {'N':>6} {'clu':>5} {'%gig':>5} {'diam':>5} {'δ_med':>7} {'d_grow':>6} {'ver':>10}")
    for Nanc in (NS[0], NS[-1]):
        for nombre, mk in (("lattice2D", lambda n: lattice2d(n)), ("arbol_b3", lambda n: arbol(n, 3))):
            adj, Nr = mk(Nanc)
            dia, g, r, clu = _medir(adj, Nr, 7)
            print(f"  {nombre:>10} {Nr:>6} {clu:>5.2f} {r['fg']*100:>4.0f} {dia:>5} {r['dmean']:>7.2f} "
                  f"{g['d']:>6.2f} {g['ver']:>10}", flush=True)

    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["brazo"], int(row["Dt"]), int(row["N"]), int(row["seed"])))

    todos = ["ARBOL", "CICLOS", "AZAR"]
    unidades = [(Dt, N, sd) for Dt in DTANS for N in NS for sd in SEEDS]
    total = len(unidades) * len(todos)
    print(f"\nfilas totales={total}  ya hechas={len(done)}  faltan={total-len(done)}   LOG={csv_path}\n")
    hdr = (f"{'brazo':>7} {'Dt':>2} {'N':>6} {'sd':>2} {'clu':>5} {'ev':>5} {'%gig':>5} {'diam':>6} "
           f"{'δ_med':>7} {'d_grow':>6} {'ver':>10}")
    print(hdr); print("-" * len(hdr))

    nuevo = not os.path.exists(csv_path)
    fcsv = open(csv_path, "a", newline=""); w = csv.writer(fcsv)
    if nuevo:
        w.writerow(cols); fcsv.flush()

    def emit(br, Dt, N, sd, dia, g, r, clu, ev):
        w.writerow([br, Dt, N, sd, r["fg"], clu, ev, dia, r["dmean"], r["d95"], g["d"], g["ver"]])
        fcsv.flush()
        print(f"{br:>7} {Dt:>2} {N:>6} {sd:>2} {clu:>5.2f} {ev:>5.2f} {r['fg']*100:>4.0f} {dia:>6} "
              f"{r['dmean']:>7.2f} {g['d']:>6.2f} {g['ver']:>10}", flush=True)

    for (Dt, N, sd) in unidades:
        faltan = [b for b in todos if (b, Dt, N, sd) not in done]
        if not faltan:
            continue
        kdeg = 2 * Dt + 4
        adjCIC_sets = None
        if "ARBOL" in faltan:
            a_sets, _ = crecer(N, Dtan=Dt, kdeg=kdeg, cos_min=0.5, m_cross=2, lambda_H=2.0, seed=sd)
            dia, g, r, clu = _medir(_finalizar(a_sets), N, sd)
            emit("ARBOL", Dt, N, sd, dia, g, r, clu, _ev(a_sets, N))
        if "CICLOS" in faltan or "AZAR" in faltan:
            adjCIC_sets, _ = crecer(N, Dtan=Dt, kdeg=kdeg, cos_min=COSM_CICLOS,
                                    m_cross=MCROSS_CICLOS, lambda_H=0.0, seed=sd)
        if "CICLOS" in faltan:
            dia, g, r, clu = _medir(_finalizar(adjCIC_sets), N, sd)
            emit("CICLOS", Dt, N, sd, dia, g, r, clu, _ev(adjCIC_sets, N))
        if "AZAR" in faltan and adjCIC_sets is not None:
            adjZ = shuffle_adj(_finalizar(adjCIC_sets), N, seed=sd + 7)
            dia, g, r, clu = _medir(adjZ, N, sd)
            emit("AZAR", Dt, N, sd, dia, g, r, clu, _ev(adjCIC_sets, N))
    fcsv.close()

    # ------------------- RESUMEN ROBUSTO (media ± std sobre semillas) -------------------
    rows = list(csv.DictReader(open(csv_path, newline="")))
    def fnum(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    def per_seed(br, Dt, N, campo):
        return {int(r["seed"]): fnum(r[campo]) for r in rows
                if r["brazo"] == br and int(r["Dt"]) == Dt and int(r["N"]) == N}

    print("\n" + "=" * 100)
    print("RESUMEN ROBUSTO (media ± std sobre semillas) — objetivo plano: diam-pend→1/Dt, δ CRECE, dim CONVERGE")
    for Dt in DTANS:
        objetivo = 1.0 / Dt
        print(f"\n  ── Dtan={Dt}  (objetivo diam-pend={objetivo:.2f}) ──")
        for br in todos:
            # diam-pend POR SEMILLA -> media±std (robustez)
            pends = []
            for sd in SEEDS:
                dias = [per_seed(br, Dt, N, "diam").get(sd, np.nan) for N in NS]
                pends.append(_slope(NS, dias))
            pends = [p for p in pends if p == p]
            pend_m = float(np.mean(pends)) if pends else float("nan")
            pend_s = float(np.std(pends)) if pends else float("nan")
            # d_grow(N) media±std, y δ_med(N) media
            dg = [(np.nanmean(list(per_seed(br, Dt, N, "dgrow").values())),
                   np.nanstd(list(per_seed(br, Dt, N, "dgrow").values()))) for N in NS]
            dme = [np.nanmean(list(per_seed(br, Dt, N, "dmean").values())) for N in NS]
            dtrend = "CRECE(plano)" if dme[-1] > dme[0] + 0.5 else "ACOTADA(hiperb)"
            clu = np.nanmean(list(per_seed(br, Dt, NS[-1], "clu").values()))
            gig = np.nanmean(list(per_seed(br, Dt, NS[-1], "fg").values())) * 100
            dstr = "  ".join(f"N={N}:{m:.2f}±{s:.2f}" for N, (m, s) in zip(NS, dg))
            print(f"    {br:>7}: clu={clu:.2f} %gig={gig:3.0f}  diam-pend={pend_m:5.2f}±{pend_s:.2f}  "
                  f"δ→{dtrend:>16}")
            print(f"             d_grow(N): {dstr}   δ_med(N): {'  '.join(f'{x:.2f}' for x in dme)}")
    print("\nLECTURA (pre-registrada, redacción CS):")
    print("  · Si CICLOS NO se separa de ARBOL hacia lo plano en Dt∈{2,3}×8 semillas (diam-pend<<1/Dt,")
    print("    δ acotada, dim trepa) => negativo ROBUSTO: 'el espacio plano no emerge de crecimiento")
    print("    relacional local en la familia probada; la obstrucción aparece GLOBAL, no local'.")
    print("  · Si CICLOS SÍ separa en algún régimen => el negativo NO era robusto: revisar antes de fijar.")
    print(f"\nTiempo: {(time.time()-t0)/60:.1f} min · CSV: {csv_path}")


main()
