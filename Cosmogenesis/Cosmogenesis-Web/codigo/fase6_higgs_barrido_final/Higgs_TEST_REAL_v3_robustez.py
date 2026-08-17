"""
Higgs_TEST_REAL_v3 — robustez (multi-seed + L mayor)

Sello de física = el de v3 (no se retoca hacia 1/1836).
Solo se varían seed y tamaño de grilla L.

Batería pre-registrada:
  A) L=30 × 10 seeds (incl. 2025 réplica)
  B) L=45 × 5 seeds
  C) L=60 × 3 seeds

Criterio de robustez (pre-registrado):
  - tasa signal_ok >= 0.7 en L=30
  - mediana |Rm-NULL| > SEP_THR
  - NULL mediana cerca de 1/3 (control)
  - sin exigir hierarchy ni 1/1836
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# import del test v3 (mismo directorio)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Higgs_TEST_REAL_v3_test import SEP_THR, run  # noqa: E402

SEEDS_L30 = [2025, 7, 42, 99, 123, 777, 1024, 3141, 8191, 99991]
SEEDS_L45 = [2025, 42, 777, 3141, 99991]
SEEDS_L60 = [2025, 42, 777]


def slim(out: dict) -> dict:
    return {
        "L": out["constants"]["L"],
        "seed": out["constants"]["SEED"],
        "verdict": out["verdict"],
        "Phi_abs_mean_SSB": out["Phi_abs_mean_SSB"],
        "wall_frac_SSB": out["wall_frac_SSB"],
        "v_k1_SSB": out["v_k1_SSB"],
        "v_k3_SSB": out["v_k3_SSB"],
        "contrast_v_k1_k3": out["contrast_v_k1_k3"],
        "Rm_mean_SSB": out["Rm_mean_SSB"],
        "NULL_mean_SSB": out["NULL_mean_SSB"],
        "separation_Rm_NULL": out["separation_Rm_NULL"],
        "flags": out["flags"],
    }


def summarize(rows: list[dict], label: str) -> dict:
    n = len(rows)
    verdicts = Counter(r["verdict"] for r in rows)
    seps = [r["separation_Rm_NULL"] for r in rows if r["separation_Rm_NULL"] is not None]
    rms = [r["Rm_mean_SSB"] for r in rows if r["Rm_mean_SSB"] is not None]
    nulls = [r["NULL_mean_SSB"] for r in rows if r["NULL_mean_SSB"] is not None]
    contrasts = [r["contrast_v_k1_k3"] for r in rows]
    signal_n = sum(1 for r in rows if r["flags"].get("signal_ok"))
    inherit_n = sum(1 for r in rows if r["flags"].get("inherit_ok"))
    vev_n = sum(1 for r in rows if r["flags"].get("vev_ok"))
    hier_n = sum(1 for r in rows if r["flags"].get("hierarchy_Rm_lt_0.1"))
    partial_or_pass = sum(
        1
        for r in rows
        if r["verdict"]
        in ("TEST_PARTIAL_medium_coupling", "TEST_PASS_higgs_like")
    )
    return {
        "label": label,
        "n": n,
        "verdicts": dict(verdicts),
        "rate_vev_ok": vev_n / n if n else 0.0,
        "rate_inherit_ok": inherit_n / n if n else 0.0,
        "rate_signal_ok": signal_n / n if n else 0.0,
        "rate_partial_or_pass": partial_or_pass / n if n else 0.0,
        "rate_hierarchy": hier_n / n if n else 0.0,
        "sep_median": float(np.median(seps)) if seps else None,
        "sep_mean": float(np.mean(seps)) if seps else None,
        "sep_min": float(np.min(seps)) if seps else None,
        "sep_max": float(np.max(seps)) if seps else None,
        "Rm_median": float(np.median(rms)) if rms else None,
        "NULL_median": float(np.median(nulls)) if nulls else None,
        "contrast_median": float(np.median(contrasts)) if contrasts else None,
        "robust_signal": (signal_n / n >= 0.7) if n else False,
        "robust_sep_median_above_thr": (
            float(np.median(seps)) > SEP_THR if seps else False
        ),
    }


def battery():
    plan = [
        ("L30_multiseed", 30, SEEDS_L30),
        ("L45_multiseed", 45, SEEDS_L45),
        ("L60_multiseed", 60, SEEDS_L60),
    ]
    all_rows = []
    summaries = []

    print("=== Higgs_TEST_REAL_v3 ROBUSTEZ ===")
    print(f"SEP_THR={SEP_THR}  (sin retocar física hacia 1/1836)")
    print(f"plan: {[(lab, L, len(seeds)) for lab, L, seeds in plan]}")

    for label, L_run, seeds in plan:
        print(f"\n--- {label} L={L_run} n={len(seeds)} ---")
        rows = []
        for seed in seeds:
            out = run(L_run=L_run, seed=seed, verbose=False)
            row = slim(out)
            rows.append(row)
            all_rows.append(row)
            sep = row["separation_Rm_NULL"]
            sep_s = f"{sep:.4f}" if sep is not None else "None"
            print(
                f"  seed={seed:6d}  {row['verdict']:36s}  "
                f"Rm={row['Rm_mean_SSB']:.4f} NULL={row['NULL_mean_SSB']:.4f} "
                f"|Δ|={sep_s}  v1/v3={row['v_k1_SSB']:.3f}/{row['v_k3_SSB']:.3f} "
                f"sig={int(row['flags']['signal_ok'])}"
            )
        sm = summarize(rows, label)
        summaries.append(sm)
        print(
            f"  >> rate_signal={sm['rate_signal_ok']:.2f}  "
            f"partial|pass={sm['rate_partial_or_pass']:.2f}  "
            f"sep_med={sm['sep_median']}  NULL_med={sm['NULL_median']}  "
            f"robust_signal={sm['robust_signal']}"
        )

    # veredicto global de robustez
    s30 = next(s for s in summaries if s["label"] == "L30_multiseed")
    if s30["robust_signal"] and s30["robust_sep_median_above_thr"]:
        global_verdict = "ROBUST_PARTIAL_medium_coupling"
        note = (
            "La señal v3 (Rm vs NULL) se reproduce en multi-seed L=30 "
            f"(rate_signal={s30['rate_signal_ok']:.2f}, sep_med={s30['sep_median']:.4f}). "
            "Germen de medio estable; no es jerarquía 1/1836."
        )
    elif s30["rate_signal_ok"] >= 0.4:
        global_verdict = "FRAGILE_PARTIAL_medium_coupling"
        note = (
            "Señal presente en una fracción de seeds pero no alcanza tasa 0.7. "
            "Germen real pero sensible a condiciones iniciales."
        )
    else:
        global_verdict = "NOT_ROBUST_single_seed_fluke"
        note = (
            "La corrida original no se generaliza: posible fluke de seed 2025."
        )

    # tendencia con L
    seps_by_L = {s["label"]: s["sep_median"] for s in summaries}
    rates_by_L = {s["label"]: s["rate_signal_ok"] for s in summaries}

    result = {
        "battery": "v3_robustez_multiseed_L",
        "global_verdict": global_verdict,
        "note": note,
        "criteria": {
            "rate_signal_ok_ge": 0.7,
            "sep_median_gt": SEP_THR,
            "no_1836_tuning": True,
        },
        "summaries": summaries,
        "sep_median_by_block": seps_by_L,
        "rate_signal_by_block": rates_by_L,
        "rows": all_rows,
    }
    return result


def main():
    out = battery()
    print("\n=== VEREDICTO GLOBAL ROBUSTEZ ===")
    print(out["global_verdict"])
    print(out["note"])
    for s in out["summaries"]:
        print(
            f"  {s['label']}: signal={s['rate_signal_ok']:.2f} "
            f"partial|pass={s['rate_partial_or_pass']:.2f} "
            f"sep_med={s['sep_median']} NULL_med={s['NULL_median']} "
            f"verdicts={s['verdicts']}"
        )
    out_path = (
        Path(__file__).resolve().parents[2]
        / "results"
        / "fase6_higgs_barrido_final"
        / "Higgs_TEST_REAL_v3_robustez_result.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
