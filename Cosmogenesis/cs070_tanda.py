"""
CS070 — TANDA BLINDADA: 4 brazos, N∈{900,1500,2500}, ≥8 semillas/brazo
==============================================================================
Autorizada tras 3/3 anclas de smoke pasando (cs070_smoke.py). Cuerda decisiva: SEMILLA_COHERENTE vs
SEMILLA_BARAJADA en direccion_real (n_ejes>1 Y certificado, G-JUEZ-NO-COHERENCIA -- nunca coherencia sola).

Lectura pre-inscrita (DISENO_CS070_semilla_amplificacion_CS.md):
(A) SEMILLA_COHERENTE > BARAJADA en n_ejes múltiples sobre el sustrato relacional -> la semilla amplifica:
    primer "sí" direccional del arco, confirma C-N2.5.5 con dato.
(B) COHERENTE ≈ BARAJADA ≈ SIN_SEMILLA (todos sin dirección real) -> el mundo-pequeño lava la semilla
    también. El sustrato es el muro, no la falta de semilla.
(C) La semilla SOLO prende en semilla_sustrato_local -> la dirección necesita semilla Y métrica juntas
    (re-ata con CS068 Mundo B: la métrica no emerge sola).

Codea/ejecuta: CC. Diseño/ruling: CS.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs070_semilla as S
from cs070_smoke import direccion_real

RNG = np.random.default_rng
NS = [900, 1500, 2500]
N_SEEDS = 8
ARMS = ["semilla_coherente", "semilla_barajada", "sin_semilla", "semilla_sustrato_local"]


def _ic95(vals):
    x = np.array(vals, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(x.mean())
    sem = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return m, m - 1.96 * sem, m + 1.96 * sem


def main():
    t0 = time.time()
    print("=" * 108, flush=True)
    print("CS070 TANDA BLINDADA", flush=True)
    print(f"brazos={ARMS}  N={NS}  semillas/brazo={N_SEEDS}", flush=True)
    print("=" * 108, flush=True)
    resultados = []
    for arm in ARMS:
        fn = S.BRAZOS[arm]
        for i, N in enumerate(NS):
            for s in range(N_SEEDS):
                seed = 70600 + 1000 * ARMS.index(arm) + 97 * i + 13 * s
                r = fn(N, seed)
                r["arm"] = arm; r["N"] = N; r["seed"] = seed; r["direccion_real"] = direccion_real(r)
                resultados.append(r)
                print(f"  [{arm:22s} N={N:5d} s={s}] n_ejes={r['n_ejes']} pico_medio={r['pico_medio']:.3f} "
                      f"certificado={r['certificado']} direccion_real={r['direccion_real']} "
                      f"peso_semilla={r['peso_semilla']:.3f} (t={(time.time()-t0)/60:.1f}min)", flush=True)
    with open(os.path.join(_HERE, "cs070_tanda_resultados.json"), "w") as f:
        json.dump(resultados, f, indent=1, default=float)
    print(f"\ntiempo total: {(time.time()-t0)/60:.2f} min -- resultados en cs070_tanda_resultados.json", flush=True)

    print("\n" + "=" * 108, flush=True)
    print("AGREGADO por brazo (blindaje de semillas, IC95%)", flush=True)
    print("=" * 108, flush=True)
    resumen = {}
    for arm in ARMS:
        rs = [r for r in resultados if r["arm"] == arm]
        dr = [float(r["direccion_real"]) for r in rs]
        cert = [float(r["certificado"]) for r in rs]
        nejes = [r["n_ejes"] for r in rs]
        m_dr, lo_dr, hi_dr = _ic95(dr)
        resumen[arm] = dict(frac_direccion_real=m_dr, ic95=(lo_dr, hi_dr),
                             frac_certificado=float(np.mean(cert)), n_ejes_medio=float(np.mean(nejes)))
        print(f"  {arm:22s}: frac_direccion_real={m_dr:.3f} IC95%=[{lo_dr:.3f},{hi_dr:.3f}]  "
              f"frac_certificado={np.mean(cert):.3f}  n_ejes_medio={np.mean(nejes):.2f}", flush=True)

    print("\n" + "=" * 108, flush=True)
    print("VEREDICTO -- cuerda decisiva: SEMILLA_COHERENTE vs SEMILLA_BARAJADA en direccion_real", flush=True)
    print("=" * 108, flush=True)
    rc = resumen["semilla_coherente"]; rb = resumen["semilla_barajada"]; rs_ = resumen["sin_semilla"]
    rl = resumen["semilla_sustrato_local"]
    gana_coherente = rc["ic95"][0] > rb["ic95"][1]  # IC95% inf de coherente > IC95% sup de barajada, sin solape
    print(f"  coherente frac_direccion_real={rc['frac_direccion_real']:.3f} IC95%={rc['ic95']}", flush=True)
    print(f"  barajada  frac_direccion_real={rb['frac_direccion_real']:.3f} IC95%={rb['ic95']}", flush=True)
    print(f"  sin_semilla frac_direccion_real={rs_['frac_direccion_real']:.3f}", flush=True)
    print(f"  semilla_sustrato_local frac_direccion_real={rl['frac_direccion_real']:.3f} IC95%={rl['ic95']}",
          flush=True)

    if gana_coherente:
        print("\n(A) SEMILLA_COHERENTE gana a BARAJADA en direccion_real, sin solape de IC95% -> LA SEMILLA", flush=True)
        print("AMPLIFICA. Primer 'sí' direccional del arco. Confirma C-N2.5.5 con dato.", flush=True)
    elif rl["ic95"][0] > 0.0 and rl["frac_direccion_real"] > rc["frac_direccion_real"] + 0.2:
        print("\n(C) La semilla SOLO prende en semilla_sustrato_local -> la dirección necesita SEMILLA Y", flush=True)
        print("MÉTRICA juntas. Re-ata con CS068 Mundo B (la métrica no emerge sola).", flush=True)
    else:
        print("\n(B) COHERENTE ≈ BARAJADA ≈ SIN_SEMILLA -> el mundo-pequeño lava la semilla también. El", flush=True)
        print("sustrato es el muro, no la falta de semilla. Profundiza el veredicto del arco.", flush=True)


if __name__ == "__main__":
    main()
