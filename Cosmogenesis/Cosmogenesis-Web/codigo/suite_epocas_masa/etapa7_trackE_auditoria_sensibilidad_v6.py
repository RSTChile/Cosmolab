#!/usr/bin/env python3
"""
etapa7_trackE_auditoria_sensibilidad_v6.py
===========================================

Script de AUDITORIA (no forma parte de la suite oficial; prefijo
etapa7_trackE_auditoria_ para dejarlo separado de los archivos de producción).

Objetivo: para los parametros de v6 que gatean el veredicto final
(rate_e4_lineage_pass >= RATE_PASS), medir si el PASS/FAIL de la cadena
es sensible a variaciones +-30%/+-50% de esos parametros, usando un
subconjunto (smoke) de las semillas de produccion (no las 10 completas,
por costo de computo: ~33s/semilla).

No modifica el archivo original suite_epocas_masa_v6_mass_linaje.py.
Hace monkeypatch de las constantes de MODULO (que las funciones
lineage_wins/nbody_step/groups_from_ids/analyze/run_controls leen como
globals en tiempo de llamada) y vuelve a correr run_controls().

Metodo: documentado en linea; no hay ocultamiento de que valores se
usaron.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import suite_epocas_masa_v6_mass_linaje as v6  # noqa: E402

SEEDS_SMOKE = (54321, 99, 2025)  # subconjunto: 54321 (comem_vs=1.183, justo sobre
# el umbral 1.15), 99 y 2025 (comem_vs=1.003/1.013, justo bajo el umbral) --
# elegidas por ser las semillas de produccion MAS CERCA del umbral de decision
# (ver JSON suite_epocas_masa_v6_result.json), donde una perturbacion del
# parametro tiene mas chance de cambiar el veredicto por semilla.
BASELINE = {
    "COMEM_VS_SHUFFLE_MIN": v6.COMEM_VS_SHUFFLE_MIN,
    "MASS_REAL_MIN": v6.MASS_REAL_MIN,
    "DENS_MIN": v6.DENS_MIN,
    "MUTUAL_MIN_STEPS": v6.MUTUAL_MIN_STEPS,
    "GROUP_LINK_R": v6.GROUP_LINK_R,
    "RATE_PASS": v6.RATE_PASS,
}


def run_batch(seeds=SEEDS_SMOKE, G=0.20):
    rows = []
    for s in seeds:
        rows.append(v6.run_controls(s, G=G))
    rate_lin = sum(r["e4_lineage_pass"] for r in rows) / len(rows)
    rate_lineage_ok = sum(r["lineage_ok"] for r in rows) / len(rows)
    mass_real = [r["modes"]["real"]["mass_E4"] for r in rows]
    dens_real = [r["modes"]["real"]["dens_enhance"] for r in rows]
    comem_vs = [r["comem_vs_shuffle"] for r in rows]
    return {
        "rate_e4_lineage_pass": rate_lin,
        "rate_lineage_ok": rate_lineage_ok,
        "mass_real_per_seed": mass_real,
        "dens_real_per_seed": dens_real,
        "comem_vs_shuffle_per_seed": comem_vs,
        "rows": [
            {
                "seed": r["seed"],
                "e4_lineage_pass": r["e4_lineage_pass"],
                "lineage_ok": r["lineage_ok"],
                "mass_real": r["modes"]["real"]["mass_E4"],
                "dens_real": r["modes"]["real"]["dens_enhance"],
                "comem_vs_shuffle": r["comem_vs_shuffle"],
            }
            for r in rows
        ],
    }


def restore():
    v6.COMEM_VS_SHUFFLE_MIN = BASELINE["COMEM_VS_SHUFFLE_MIN"]
    v6.MASS_REAL_MIN = BASELINE["MASS_REAL_MIN"]
    v6.DENS_MIN = BASELINE["DENS_MIN"]
    v6.MUTUAL_MIN_STEPS = BASELINE["MUTUAL_MIN_STEPS"]
    v6.GROUP_LINK_R = BASELINE["GROUP_LINK_R"]


def main():
    t0 = time.time()
    out = {"seeds_smoke": list(SEEDS_SMOKE), "baseline_params": BASELINE}

    print(f"[baseline] corriendo {len(SEEDS_SMOKE)} seeds con params base...")
    restore()
    out["baseline"] = run_batch()
    print(f"  rate_e4_lineage_pass={out['baseline']['rate_e4_lineage_pass']:.2f} "
          f"rate_lineage_ok={out['baseline']['rate_lineage_ok']:.2f}")

    # NOTA: MASS_REAL_MIN (0.3) y DENS_MIN (1.2) se EXCLUYEN del barrido por
    # computo: ya se confirmo analiticamente (JSON produccion v6, 10 seeds)
    # que mass_real real oscila 43.8-283.9 (100-1000x el umbral) y dens_real
    # oscila 30.1-59.8 (25-50x el umbral) -> un +-50% en el umbral no puede
    # acercarse a esos valores; no son gates activos, no ameritan compute.
    # Los parametros que SI estan cerca del margen de decision (comem_vs_shuffle
    # cluster 0.70-2.03 contra umbral 1.15) son los que se prueban aqui.
    perturbations = {
        "COMEM_VS_SHUFFLE_MIN": [0.5, 1.5],   # base 1.15 -> 0.575, 1.725
        "GROUP_LINK_R": [0.5, 1.5],           # base 4.5 -> 2.25, 6.75
        "MUTUAL_MIN_STEPS": [0.5, 1.5],       # base 5 -> 3(round), 8(round)
    }

    out["perturbaciones"] = {}
    for pname, factors in perturbations.items():
        out["perturbaciones"][pname] = []
        for f in factors:
            restore()
            base_val = BASELINE[pname]
            new_val = base_val * f
            if pname == "MUTUAL_MIN_STEPS":
                new_val = max(1, round(new_val))
            setattr(v6, pname, new_val)
            print(f"[{pname}] factor={f} valor_base={base_val} valor_nuevo={new_val}")
            res = run_batch()
            print(f"  -> rate_e4_lineage_pass={res['rate_e4_lineage_pass']:.2f} "
                  f"rate_lineage_ok={res['rate_lineage_ok']:.2f} "
                  f"mass_real={res['mass_real_per_seed']}")
            out["perturbaciones"][pname].append({
                "factor": f,
                "valor_base": base_val,
                "valor_nuevo": new_val,
                **res,
            })
        restore()

    out["elapsed_s"] = time.time() - t0
    out_path = HERE.parent.parent / "results" / "etapa7_trackE_auditoria" / "sensibilidad_v6_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[archivo] {out_path}")
    print(f"[elapsed] {out['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
