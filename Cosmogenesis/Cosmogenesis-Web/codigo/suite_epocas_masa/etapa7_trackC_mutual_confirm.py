#!/usr/bin/env python3
"""
etapa7_trackC_mutual_confirm.py — confirmación fuera de muestra de la
variante `mean_instant` (H4), según
PROTOCOLO_TRACKC_MUTUAL_CONFIRMACION_ADENDO.md (pre-registrado ANTES
de esta corrida). Definición congelada, sin más ajustes; semillas
nuevas nunca usadas en el proyecto ni en el set estándar de 10.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etapa7_trackC_mutual_engine import P, simulate_diag, stats_from_sim, rs  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "etapa7_trackC_mutual"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS_CONFIRM = (111, 222, 333, 444, 555, 666, 777777, 13, 31, 271828)
G = 0.20
DECISION_RS = 1.15
DECISION_RATE = 0.60


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=== CONFIRMACION fuera de muestra: mean_instant, 10 semillas nuevas ===")
    rows = []
    for seed in SEEDS_CONFIRM:
        t0 = time.time()
        p_r = P(seed=seed, G_GRAV=G, grav_mode="real")
        p_s = P(seed=seed, G_GRAV=G, grav_mode="shuffle")
        sim_r = simulate_diag(p_r)
        sim_s = simulate_diag(p_s)
        st_r = stats_from_sim(sim_r)
        st_s = stats_from_sim(sim_s)
        r_val = st_r["mean_instant"]
        s_val = st_s["mean_instant"]
        row = {"seed": seed, "real": r_val, "shuffle": s_val, "RS": rs(r_val, s_val)}
        rows.append(row)
        log(f"  seed={seed:7d} real={r_val:.3f} shuffle={s_val:.3f} RS={row['RS']} ({time.time()-t0:.1f}s)")

    finite = [row["RS"] for row in rows if row["RS"] == row["RS"] and row["RS"] != float("inf")]
    n_gt1 = sum(1 for row in rows if row["RS"] == row["RS"] and row["RS"] > 1.0)
    summary = {
        "n": len(rows),
        "mean_RS_finite": (sum(finite) / len(finite)) if finite else None,
        "n_finite": len(finite),
        "rate_RS_gt_1": n_gt1 / len(rows) if rows else 0.0,
        "n_RS_gt_1": n_gt1,
    }
    if summary["mean_RS_finite"] is not None and summary["mean_RS_finite"] >= DECISION_RS and summary["rate_RS_gt_1"] >= DECISION_RATE:
        decision = "RESUELVE (confirmado fuera de muestra)"
    else:
        decision = "NO REPLICA / candidato no confirmado (posible artefacto de seleccion de estadistico)"

    log(f"summary: {summary} -> {decision}")
    out = {"seeds": list(SEEDS_CONFIRM), "rows": rows, "summary": summary, "decision": decision}
    (OUT / "trackC_H4_mean_instant_confirmacion.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"JSON -> {OUT / 'trackC_H4_mean_instant_confirmacion.json'}")


if __name__ == "__main__":
    main()
