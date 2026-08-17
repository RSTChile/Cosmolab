#!/usr/bin/env python3
"""
etapa7_trackC_mutual_run.py — corredor de Track C (H1-H4), según
PROTOCOLO_TRACKC_MUTUAL_PREREGISTRO.md (pre-registrado ANTES de esta
corrida). Usa etapa7_trackC_mutual_engine.py (motor instrumentado,
copia paramétrica fiel de v6 — v6.py no se toca).

Salidas (JSON crudos, uno por sección):
  results/etapa7_trackC_mutual/trackC_baseline.json
  results/etapa7_trackC_mutual/trackC_H1_window.json
  results/etapa7_trackC_mutual/trackC_H2_cutoff.json
  results/etapa7_trackC_mutual/trackC_H3_posrandom.json
  results/etapa7_trackC_mutual/trackC_H4_selection.json (derivado de baseline, sin correr sim nuevas)
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

SEEDS_ALL = (7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321)
SEEDS_SMOKE = (7, 42, 99, 777)
G = 0.20
DECISION_RS = 1.15
DECISION_RATE = 0.60
SMOKE_RS = 1.05


def run_point(seed, mode, **cfg_kwargs):
    p_kwargs = {}
    for key in ("GRAV_START_FRAC", "pasos"):
        if key in cfg_kwargs:
            p_kwargs[key] = cfg_kwargs.pop(key)
    p = P(seed=seed, G_GRAV=G, grav_mode=mode, **p_kwargs)
    sim = simulate_diag(p, **cfg_kwargs)
    return stats_from_sim(sim)


def rs_row(seed, cfg_kwargs, variant="min_gated"):
    r = run_point(seed, "real", **cfg_kwargs)
    s = run_point(seed, "shuffle", **cfg_kwargs)
    return {
        "seed": seed,
        "real": r[variant],
        "shuffle": s[variant],
        "RS": rs(r[variant], s[variant]),
        "real_full": r,
        "shuffle_full": s,
    }


def summarize(rows, variant_key="RS"):
    finite = [row[variant_key] for row in rows if row[variant_key] == row[variant_key] and row[variant_key] != float("inf")]
    n_gt1 = sum(1 for row in rows if row[variant_key] == row[variant_key] and row[variant_key] > 1.0)
    return {
        "n": len(rows),
        "mean_RS_finite": (sum(finite) / len(finite)) if finite else None,
        "n_finite": len(finite),
        "rate_RS_gt_1": n_gt1 / len(rows) if rows else 0.0,
        "n_RS_gt_1": n_gt1,
    }


def decide(summary):
    if summary["mean_RS_finite"] is None:
        return "NO_RESUELVE (sin datos finitos)"
    if summary["mean_RS_finite"] >= DECISION_RS and summary["rate_RS_gt_1"] >= DECISION_RATE:
        return "RESUELVE"
    if summary["mean_RS_finite"] >= 1.0 or summary["rate_RS_gt_1"] >= 0.5:
        return "PARCIAL"
    return "NO_RESUELVE"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------
def run_baseline():
    log("=== BASELINE (10 semillas, params v6 default) ===")
    rows = []
    for seed in SEEDS_ALL:
        t0 = time.time()
        row = rs_row(seed, {})
        rows.append(row)
        log(f"  seed={seed:5d} min_gated R={row['real']:.3f} S={row['shuffle']:.3f} RS={row['RS']:.3f}  ({time.time()-t0:.1f}s)")
    summary = summarize(rows)
    out = {"rows": rows, "summary": summary, "seeds": list(SEEDS_ALL)}
    (OUT / "trackC_baseline.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"baseline summary: {summary}")
    return out


def run_h1(baseline):
    log("=== H1: ventana de gravedad (smoke primero) ===")
    baseline_by_seed = {r["seed"]: r for r in baseline["rows"]}
    grid = [
        {"GRAV_START_FRAC": 0.35, "pasos": 400},
        {"GRAV_START_FRAC": 0.50, "pasos": 400},
        {"GRAV_START_FRAC": 0.80, "pasos": 400},
        {"GRAV_START_FRAC": 0.65, "pasos": 800},
        {"GRAV_START_FRAC": 0.35, "pasos": 800},
    ]
    result = {"baseline_point": {"GRAV_START_FRAC": 0.65, "pasos": 400, "summary": baseline["summary"]}, "grid": []}
    for cfg in grid:
        log(f"-- smoke config {cfg} --")
        rows = []
        for seed in SEEDS_SMOKE:
            t0 = time.time()
            row = rs_row(seed, dict(cfg))
            rows.append(row)
            log(f"   seed={seed:5d} R={row['real']:.3f} S={row['shuffle']:.3f} RS={row['RS']:.3f} ({time.time()-t0:.1f}s)")
        smoke_summary = summarize(rows)
        log(f"   smoke summary {cfg}: {smoke_summary}")
        entry = {"cfg": cfg, "smoke_rows": rows, "smoke_summary": smoke_summary, "escalated": False}
        promising = (
            smoke_summary["mean_RS_finite"] is not None
            and smoke_summary["mean_RS_finite"] >= SMOKE_RS
        ) or smoke_summary["rate_RS_gt_1"] >= 0.75
        if promising:
            log(f"   -> señal en smoke, ESCALANDO a 10 semillas: {cfg}")
            full_rows = list(rows)  # ya tenemos las 4 smoke
            for seed in SEEDS_ALL:
                if seed in SEEDS_SMOKE:
                    continue
                t0 = time.time()
                row = rs_row(seed, dict(cfg))
                full_rows.append(row)
                log(f"   [full] seed={seed:5d} R={row['real']:.3f} S={row['shuffle']:.3f} RS={row['RS']:.3f} ({time.time()-t0:.1f}s)")
            full_summary = summarize(full_rows)
            entry["escalated"] = True
            entry["full_rows"] = full_rows
            entry["full_summary"] = full_summary
            entry["decision"] = decide(full_summary)
            log(f"   full(10) summary {cfg}: {full_summary} -> {entry['decision']}")
        else:
            entry["decision"] = "NO_RESUELVE (sin señal en smoke, no escalado)"
        result["grid"].append(entry)
        (OUT / "trackC_H1_window.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_h2(baseline):
    log("=== H2: cutoff/softening (smoke primero) ===")
    grid = [
        {"force_cutoff": 3.0, "softening": 1.2},
        {"force_cutoff": 5.0, "softening": 1.2},
        {"force_cutoff": 12.0, "softening": 1.2},
        {"force_cutoff": 8.0, "softening": 0.4},
        {"force_cutoff": 8.0, "softening": 3.0},
    ]
    result = {"baseline_point": {"force_cutoff": 8.0, "softening": 1.2, "summary": baseline["summary"]}, "grid": []}
    for cfg in grid:
        log(f"-- smoke config {cfg} --")
        rows = []
        for seed in SEEDS_SMOKE:
            t0 = time.time()
            row = rs_row(seed, dict(cfg))
            rows.append(row)
            log(f"   seed={seed:5d} R={row['real']:.3f} S={row['shuffle']:.3f} RS={row['RS']:.3f} ({time.time()-t0:.1f}s)")
        smoke_summary = summarize(rows)
        log(f"   smoke summary {cfg}: {smoke_summary}")
        entry = {"cfg": cfg, "smoke_rows": rows, "smoke_summary": smoke_summary, "escalated": False}
        promising = (
            smoke_summary["mean_RS_finite"] is not None
            and smoke_summary["mean_RS_finite"] >= SMOKE_RS
        ) or smoke_summary["rate_RS_gt_1"] >= 0.75
        if promising:
            log(f"   -> señal en smoke, ESCALANDO a 10 semillas: {cfg}")
            full_rows = list(rows)
            for seed in SEEDS_ALL:
                if seed in SEEDS_SMOKE:
                    continue
                t0 = time.time()
                row = rs_row(seed, dict(cfg))
                full_rows.append(row)
                log(f"   [full] seed={seed:5d} R={row['real']:.3f} S={row['shuffle']:.3f} RS={row['RS']:.3f} ({time.time()-t0:.1f}s)")
            full_summary = summarize(full_rows)
            entry["escalated"] = True
            entry["full_rows"] = full_rows
            entry["full_summary"] = full_summary
            entry["decision"] = decide(full_summary)
            log(f"   full(10) summary {cfg}: {full_summary} -> {entry['decision']}")
        else:
            entry["decision"] = "NO_RESUELVE (sin señal en smoke, no escalado)"
        result["grid"].append(entry)
        (OUT / "trackC_H2_cutoff.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_h3():
    log("=== H3: posiciones aleatorizadas al entrar a E4 (10 semillas directo) ===")
    rows = []
    for seed in SEEDS_ALL:
        t0 = time.time()
        row = rs_row(seed, {"posrandom_e4_entry": True})
        rows.append(row)
        log(f"  seed={seed:5d} R={row['real']:.3f} S={row['shuffle']:.3f} RS={row['RS']:.3f} ({time.time()-t0:.1f}s)")
    summary = summarize(rows)
    decision = decide(summary)
    out = {"rows": rows, "summary": summary, "decision": decision, "seeds": list(SEEDS_ALL)}
    (OUT / "trackC_H3_posrandom.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"H3 summary: {summary} -> {decision}")
    return out


def run_h4(baseline):
    log("=== H4: variantes de estadístico (derivado de baseline, sin sims nuevas) ===")
    variants = [
        "min_gated", "mean_gated", "min_instant", "mean_instant",
        "per_pair_mean_gated", "per_pair_mean_instant",
    ]
    out = {"variants": {}}
    for var in variants:
        rows = []
        for row in baseline["rows"]:
            r_val = row["real_full"][var]
            s_val = row["shuffle_full"][var]
            rows.append({"seed": row["seed"], "real": r_val, "shuffle": s_val, "RS": rs(r_val, s_val)})
        summary = summarize(rows)
        decision = decide(summary)
        out["variants"][var] = {"rows": rows, "summary": summary, "decision": decision}
        log(f"  variant={var:22s} mean_RS_finite={summary['mean_RS_finite']} rate_gt1={summary['rate_RS_gt_1']:.2f} -> {decision}")
    (OUT / "trackC_H4_selection.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    t0 = time.time()
    baseline = run_baseline()
    h4 = run_h4(baseline)
    h3 = run_h3()
    h1 = run_h1(baseline)
    h2 = run_h2(baseline)
    log(f"=== TOTAL elapsed {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
