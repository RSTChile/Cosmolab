"""
CS072 v7 -- refinamiento de resolución en la ventana crítica [0.04, 0.13] hallada por
cs072_v7_banda_persistencia.py (salto abrupto: poda=0.08 aún sostiene, poda=0.12 colapso total).
Mismo motor, mismo protocolo (exploratoria, sin veredicto). Codea/ejecuta: CC. Diseño/ruling: CS.
"""
from __future__ import annotations
import sys, time, json
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs072_v6_nucleo as V6
import cs071_histeresis as S71
import cs071_tanda as S71T
import cs064_smoke as SM

RNG = np.random.default_rng
DELTA = 1e-4
PASOS = 80
N_PRIMARIO = 400
NS_BETA = [400, 900, 1600]
SEED_BASE = 90172

PODA_TASAS = [0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075,
              0.080, 0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115, 0.120]
FOCOS = [1, 5, 20]


def _judges(adj, N, rng):
    diam = S71._diam_robusto(adj, N, rng)
    delta_g = S71._delta_gromov(adj, N, rng)
    frac_gig = S71._frac_gigante(adj, N, rng)
    ds = SM.dim_volumen(adj, N, rng=rng)
    grado_max = max((len(a) for a in adj), default=0)
    return dict(diam=diam, delta_gromov=delta_g, frac_gigante=frac_gig, d_s=ds, grado_max=grado_max)


def _corre_punto(N, n_focos, poda_tasa, seed):
    r = V6.corre_nucleo_v6(N=N, n_focos=n_focos, delta=DELTA, pasos=PASOS, seed=seed, poda_tasa=poda_tasa)
    j = _judges(r["adj"], N, RNG(seed + 1))
    j["cv_final"] = float(r["cvs"][-1])
    return j


def main():
    t0 = time.time()
    print("=" * 100, flush=True)
    print("CS072 v7 -- resolución FINA en la ventana crítica [0.04,0.12] (N=400)", flush=True)
    print("=" * 100, flush=True)

    resultados = []
    for n_focos in FOCOS:
        print(f"\n--- n_focos={n_focos} ---", flush=True)
        for poda in PODA_TASAS:
            seed = SEED_BASE + n_focos * 1000 + int(round(poda * 100000))
            j = _corre_punto(N_PRIMARIO, n_focos, poda, seed)
            j.update(n_focos=n_focos, poda_tasa=poda, N=N_PRIMARIO)
            resultados.append(j)
            print(f"  poda={poda:.3f}  grado_max={j['grado_max']:4d}  frac_conectada={j['frac_gigante']:.3f}  "
                  f"delta_gromov={j['delta_gromov']:.2f}  d_s={j['d_s']:.2f}  diam={j['diam']:.2f}  "
                  f"CV={j['cv_final']:.3f}  (t={(time.time()-t0)/60:.1f}min)", flush=True)

    with open("cs072_v7_banda_persistencia_fino.json", "w") as f:
        json.dump(resultados, f, indent=2)

    # buscar la sub-ventana donde frac_gigante>=0.9 Y grado_max plano (<20), por n_focos
    print("\n" + "=" * 100, flush=True)
    print("SUB-VENTANA candidata a BANDA (frac_conectada>=0.9 Y grado_max<20), por n_focos", flush=True)
    print("=" * 100, flush=True)
    candidatos_beta = set()
    for n_focos in FOCOS:
        fila = sorted([r for r in resultados if r["n_focos"] == n_focos], key=lambda r: r["poda_tasa"])
        buenos = [r["poda_tasa"] for r in fila if r["frac_gigante"] >= 0.9 and 0 < r["grado_max"] < 20]
        # última tasa con frac>0.5 (borde justo antes del colapso total)
        vivos = [r["poda_tasa"] for r in fila if r["frac_gigante"] > 0.3]
        borde_colapso = max(vivos) if vivos else None
        primero_colapso = min([r["poda_tasa"] for r in fila if r["frac_gigante"] <= 0.05], default=None)
        print(f"n_focos={n_focos}: sub-ventana(frac>=0.9,grado<20)={buenos}  "
              f"ultimo_vivo(frac>0.3)~{borde_colapso}  primer_colapso_total(frac<=0.05)~{primero_colapso}",
              flush=True)
        candidatos_beta.update(buenos)
        if borde_colapso is not None:
            candidatos_beta.add(borde_colapso)

    print("\n" + "=" * 100, flush=True)
    print(f"beta en candidatos de la ventana fina, n_focos=5: {sorted(candidatos_beta)}", flush=True)
    print("=" * 100, flush=True)
    resumen_beta = []
    for poda in sorted(candidatos_beta):
        diams, fracs, gmax, dgs = [], [], [], []
        for N in NS_BETA:
            seed = SEED_BASE + 7000 + int(round(poda * 100000)) + N
            j = _corre_punto(N, 5, poda, seed)
            diams.append(j["diam"]); fracs.append(j["frac_gigante"])
            gmax.append(j["grado_max"]); dgs.append(j["delta_gromov"])
            print(f"  poda={poda:.3f} N={N}: diam={j['diam']:.2f} grado_max={j['grado_max']} "
                  f"frac_gigante={j['frac_gigante']:.3f} delta_g={j['delta_gromov']:.2f} "
                  f"(t={(time.time()-t0)/60:.1f}min)", flush=True)
        Ns_validos = [N for N, d in zip(NS_BETA, diams) if np.isfinite(d) and d > 0]
        diams_validos = [d for d in diams if np.isfinite(d) and d > 0]
        if len(Ns_validos) >= 2:
            beta, _ = S71T._pendiente_loglog(Ns_validos, diams_validos)
        else:
            beta = float("nan")
        fila = dict(poda_tasa=poda, beta=beta, grado_max_N1600=gmax[-1] if gmax else None,
                    frac_gigante_N1600=fracs[-1] if fracs else None,
                    delta_gromov_N1600=dgs[-1] if dgs else None)
        resumen_beta.append(fila)
        print(f"  >>> poda={poda:.3f}: beta={beta:.3f}  grado_max(N=1600)={fila['grado_max_N1600']}  "
              f"frac_gigante(N=1600)={fila['frac_gigante_N1600']}  delta_g(N=1600)={fila['delta_gromov_N1600']}",
              flush=True)

    with open("cs072_v7_banda_persistencia_fino_beta.json", "w") as f:
        json.dump(resumen_beta, f, indent=2)

    print(f"\ntiempo total: {(time.time()-t0)/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
