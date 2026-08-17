#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CG001 — Experimento de control A (ε=0) vs B (ε>0).
Corre ambos universos con la misma semilla, registra series temporales y resume divergencia.
Uso:
  PYTHONPATH=/app python CG001/experiments/CG001_control_ab.py
  CG_STEPS=8000 CG_LOG_EVERY=25 CG_SEED=42
"""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from CG001.core.universe import Universe  # noqa: E402

STEPS = int(os.environ.get("CG_STEPS", "5000"))
LOG_EVERY = int(os.environ.get("CG_LOG_EVERY", "25"))
SEED = int(os.environ.get("CG_SEED", "42"))
EPS_B = float(os.environ.get("CG_EPSILON_B", "0.00001"))
N_ENTITIES = int(os.environ.get("CG_N_ENTITIES", "1000"))
OUT_DIR = Path(os.environ.get("CG_OUT_DIR", ROOT / "CG001" / "logs"))
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "CG001_default.yaml"


def _base_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["universe"]["n_entities"] = N_ENTITIES
    return cfg


def _run_arm(experiment_id: str, epsilon: float) -> tuple[list[dict], list[dict], dict]:
    os.environ["CG_SEED"] = str(SEED)
    os.environ["CG_EPSILON"] = str(epsilon)
    os.environ["CG_EXPERIMENT_ID"] = experiment_id
    os.environ["CG_QUIET_EVENTS"] = "1"
    os.environ["CG_FAST_METRICS"] = "1"
    os.environ["CG_MAX_PAIRS"] = os.environ.get("CG_MAX_PAIRS", "4000")

    u = Universe(config=_base_config())
    series: list[dict] = []
    events_all: list[dict] = []

    for t in range(1, STEPS + 1):
        snap = u.step()
        if t % 500 == 0:
            print(f"    [{experiment_id}] t={t} N={snap['N']} IPD={snap['metrics']['IPD']:.3f}", flush=True)
        if t == 1 or t % LOG_EVERY == 0 or t == STEPS:
            m = snap["metrics"]
            series.append({
                "t_sim": t,
                "experiment_id": experiment_id,
                "epsilon": epsilon,
                "N": snap["N"],
                "N_frac": round(snap["N"] / snap["N0"], 6),
                "R": snap["R"],
                "IPD": m["IPD"],
                "IH": m["IH"],
                "IN": m["IN"],
                "IPA": m["IPA"],
                "ICG0": m["ICG0"],
                "S_max": m["S_max"],
                "S_mean": m["S_mean"],
                "delta_mean": m["delta_mean"],
                "H_delta": m["H_delta"],
                "niches_grid": snap["niches"],
                "env_H": snap["env_H"],
                "S_max_entity": snap.get("S_max_entity"),
            })
        if snap.get("events_recent"):
            for ev in snap["events_recent"]:
                if ev not in events_all[-20:]:
                    events_all.append({**ev, "experiment_id": experiment_id})

    final = u.snapshot()
    summary = {
        "experiment_id": experiment_id,
        "epsilon": epsilon,
        "seed": SEED,
        "steps": STEPS,
        "N_final": final["N"],
        "N0": final["N0"],
        "survival_frac": round(final["N"] / final["N0"], 6),
        "IPD_final": final["metrics"]["IPD"],
        "IH_final": final["metrics"]["IH"],
        "IN_final": final["metrics"]["IN"],
        "S_max_final": final["metrics"]["S_max"],
        "filter_pct_lost": round(100 * (1 - final["N"] / final["N0"]), 2),
        "events_total": len(u.events),
    }
    return series, events_all[-500:], summary


def _first_cross(series: list[dict], key: str, thr: float) -> int | None:
    for row in series:
        if row[key] >= thr:
            return row["t_sim"]
    return None


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_DIR / f"CG001_AB_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CG001] Experimento control A/B — seed={SEED} steps={STEPS} log_every={LOG_EVERY}")
    series_a, events_a, sum_a = _run_arm("CG001-A", 0.0)
    print(f"  A terminado: N={sum_a['N_final']} IPD={sum_a['IPD_final']} IH={sum_a['IH_final']}")
    series_b, events_b, sum_b = _run_arm("CG001-B", EPS_B)
    print(f"  B terminado: N={sum_b['N_final']} IPD={sum_b['IPD_final']} IH={sum_b['IH_final']}")

    csv_path = run_dir / "cg001_ab_series.csv"
    fields = list(series_a[0].keys()) if series_a else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(series_a)
        w.writerows(series_b)

    divergencia = {
        "d_N": sum_b["N_final"] - sum_a["N_final"],
        "d_survival_frac": round(sum_b["survival_frac"] - sum_a["survival_frac"], 6),
        "d_IPD_final": round(sum_b["IPD_final"] - sum_a["IPD_final"], 6),
        "d_IH_final": round(sum_b["IH_final"] - sum_a["IH_final"], 6),
        "d_IN_final": sum_b["IN_final"] - sum_a["IN_final"],
        "d_S_max_final": round(sum_b["S_max_final"] - sum_a["S_max_final"], 6),
        "ratio_IPD_B_over_A": round(sum_b["IPD_final"] / max(sum_a["IPD_final"], 1e-9), 6),
        "ratio_IH_B_over_A": round(sum_b["IH_final"] / max(sum_a["IH_final"], 1e-9), 6),
    }

    milestones = {
        "A_first_IPD_1.5": _first_cross(series_a, "IPD", 1.5),
        "B_first_IPD_1.5": _first_cross(series_b, "IPD", 1.5),
        "A_first_IH_100": _first_cross(series_a, "IH", 100),
        "B_first_IH_100": _first_cross(series_b, "IH", 100),
    }

    prediccion_cumple = (
        divergencia["d_IPD_final"] > 0
        and divergencia["d_IH_final"] > 0
    )

    resultado = {
        "protocolo": "CG001 control experimental §59-60",
        "epsilon_fix": "v2 — ε solo en condición inicial id=0 (revisión Claude §3.4)",
        "timestamp_utc": stamp,
        "parametros": {
            "seed": SEED,
            "steps": STEPS,
            "log_every": LOG_EVERY,
            "epsilon_B": EPS_B,
            "n_entities": N_ENTITIES,
            "max_pairs": int(os.environ.get("CG_MAX_PAIRS", "4000")),
            "fast_metrics": True,
        },
        "CG001-A": sum_a,
        "CG001-B": sum_b,
        "divergencia": divergencia,
        "milestones": milestones,
        "prediccion_cosmosemiotica_B_supera_A": prediccion_cumple,
        "archivos": {"series_csv": str(csv_path.name)},
    }

    json_path = run_dir / "cg001_ab_resultado.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    events_path = run_dir / "cg001_ab_eventos.jsonl"
    with open(events_path, "w", encoding="utf-8") as f:
        for ev in events_a + events_b:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    txt_lines = [
        "CG001 — RESULTADO EXPERIMENTO CONTROL A vs B",
        f"UTC: {stamp} | semilla: {SEED} | pasos: {STEPS}",
        "",
        "── CG001-A (ε=0) ──",
        f"  Supervivientes: {sum_a['N_final']} / {sum_a['N0']} ({sum_a['survival_frac']*100:.2f}%)",
        f"  Filtro (pérdida): {sum_a['filter_pct_lost']:.1f}%",
        f"  IPD final: {sum_a['IPD_final']}",
        f"  IH final:  {sum_a['IH_final']:.2f}",
        f"  IN final:  {sum_a['IN_final']}",
        f"  S_max:     {sum_a['S_max_final']}",
        "",
        "── CG001-B (ε>0) ──",
        f"  Supervivientes: {sum_b['N_final']} / {sum_b['N0']} ({sum_b['survival_frac']*100:.2f}%)",
        f"  Filtro (pérdida): {sum_b['filter_pct_lost']:.1f}%",
        f"  IPD final: {sum_b['IPD_final']}",
        f"  IH final:  {sum_b['IH_final']:.2f}",
        f"  IN final:  {sum_b['IN_final']}",
        f"  S_max:     {sum_b['S_max_final']}",
        "",
        "── Divergencia (B − A) ──",
        f"  ΔN:   {divergencia['d_N']}",
        f"  ΔIPD: {divergencia['d_IPD_final']}",
        f"  ΔIH:  {divergencia['d_IH_final']}",
        f"  ΔIN:  {divergencia['d_IN_final']}",
        f"  ratio IPD B/A: {divergencia['ratio_IPD_B_over_A']}",
        f"  ratio IH  B/A: {divergencia['ratio_IH_B_over_A']}",
        "",
        "── Hitos ──",
        f"  Primer IPD≥1.5 — A: t={milestones['A_first_IPD_1.5']} | B: t={milestones['B_first_IPD_1.5']}",
        f"  Primer IH≥100  — A: t={milestones['A_first_IH_100']} | B: t={milestones['B_first_IH_100']}",
        "",
        f"Predicción §60 (B > A en IPD e IH): {'SÍ' if prediccion_cumple else 'NO'}",
        "",
        f"CSV: {csv_path}",
        f"JSON: {json_path}",
    ]
    txt_path = run_dir / "cg001_ab_resultado.txt"
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    print("\n" + "\n".join(txt_lines[-12:]))
    print(f"\n[CG001] Resultados en {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())