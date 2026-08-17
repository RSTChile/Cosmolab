#!/usr/bin/env python3
"""
CG001 — Test de causalidad de ε (mover arruga, línea base caos).

Observable: argmax(|m_B − m_A|) vs posición de ε.
Control: |m_A1 − m_A2| semillas distintas.
Soporta barrido de RUIDO y de ε. Solo reporta datos — sin veredicto.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from cg001_field import FieldConfig, correr

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"

UMBRAL_SIGNO = 0.83
RADIO_SEGUIMIENTO = 5
SEMILLAS_DEFAULT = list(range(1, 31))
SEED_OFFSET_BASELINE = 10_000


def _dist(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return float(np.linalg.norm(np.subtract(a, b)))


def _argmax_3d(f: np.ndarray) -> tuple[int, int, int]:
    flat = int(np.argmax(f))
    return np.unravel_index(flat, f.shape)


def posiciones_eps(L: int) -> list[dict]:
    c = L // 2
    rng = np.random.default_rng(43)
    aleatorias = [tuple(int(x) for x in p) for p in rng.integers(4, L - 4, size=(3, 3))]
    return [
        {"id": "centro", "pos": (c, c, c)},
        {"id": "esquina", "pos": (0, 0, 0)},
        {"id": "aleatoria_1", "pos": aleatorias[0]},
        {"id": "aleatoria_2", "pos": aleatorias[1]},
        {"id": "aleatoria_3", "pos": aleatorias[2]},
    ]


def metricas_diferencia(m_ref: np.ndarray, m_cmp: np.ndarray) -> dict:
    diff = np.abs(m_cmp - m_ref)
    idx = _argmax_3d(diff)
    return {
        "argmax_diff": idx,
        "max_diff": float(diff.max()),
        "mean_diff": float(diff.mean()),
        "l2_diff": float(np.sqrt((diff * diff).sum())),
    }


def linea_base_caos(
    semillas: list[int],
    cfg: FieldConfig,
    ruido: float,
) -> tuple[list[dict], dict]:
    filas = []
    peaks = []
    for s in semillas:
        s2 = s + SEED_OFFSET_BASELINE
        m1 = correr(False, seed=s, cfg=cfg, ruido=ruido, retornar_campos=True)["m"]
        m2 = correr(False, seed=s2, cfg=cfg, ruido=ruido, retornar_campos=True)["m"]
        d = metricas_diferencia(m1, m2)
        peaks.append(d["max_diff"])
        filas.append({
            "seed": s,
            "seed_par": s2,
            "max_diff_aa": d["max_diff"],
            "mean_diff_aa": d["mean_diff"],
        })
    arr = np.asarray(peaks, dtype=np.float64)
    stats = {
        "mediana_max_diff": float(np.median(arr)),
        "media_max_diff": float(arr.mean()),
        "p90_max_diff": float(np.quantile(arr, 0.90)),
        "p95_max_diff": float(np.quantile(arr, 0.95)),
        "max_max_diff": float(arr.max()),
    }
    return filas, stats


def evaluar_semillas(
    pos: tuple[int, int, int],
    semillas: list[int],
    cfg: FieldConfig,
    ruido: float,
    baseline_stats: dict,
    radio: int,
) -> list[dict]:
    filas = []
    for s in semillas:
        ra = correr(False, seed=s, cfg=cfg, ruido=ruido, retornar_campos=True)
        rb = correr(True, seed=s, cfg=cfg, ruido=ruido, eps_pos=pos, retornar_campos=True)
        dab = metricas_diferencia(ra["m"], rb["m"])
        dist_eps = _dist(dab["argmax_diff"], pos)
        sigue = dist_eps <= radio
        supera_base = dab["max_diff"] > baseline_stats["p90_max_diff"]
        ratio_base = dab["max_diff"] / (baseline_stats["mediana_max_diff"] + 1e-12)
        eps_agrega = supera_base and ratio_base > 1.5
        filas.append({
            "seed": s,
            "dist_pico_a_eps": dist_eps,
            "sigue_eps": sigue,
            "max_diff_ab": dab["max_diff"],
            "supera_baseline_p90": supera_base,
            "ratio_vs_baseline": ratio_base,
            "eps_agrega": eps_agrega,
            "causal": sigue and eps_agrega,
            "argmax_diff": dab["argmax_diff"],
        })
    return filas


def agregar_posicion(filas: list[dict], pid: str, pos: tuple[int, int, int]) -> dict:
    dists = [f["dist_pico_a_eps"] for f in filas]
    fr_sigue = float(np.mean([f["sigue_eps"] for f in filas]))
    fr_agrega = float(np.mean([f["eps_agrega"] for f in filas]))
    fr_causal = float(np.mean([f["causal"] for f in filas]))
    return {
        "id": pid,
        "eps_pos": pos,
        "n_semillas": len(filas),
        "frac_sigue_eps": fr_sigue,
        "frac_eps_agrega": fr_agrega,
        "frac_causal": fr_causal,
        "certifica_sigue": fr_sigue >= UMBRAL_SIGNO,
        "certifica_agrega": fr_agrega >= UMBRAL_SIGNO,
        "certifica_causal": fr_causal >= UMBRAL_SIGNO,
        "dist_pico_media": float(np.mean(dists)),
        "dist_pico_mediana": float(np.median(dists)),
        "max_diff_ab_media": float(np.mean([f["max_diff_ab"] for f in filas])),
    }


def correr_bloque(
    ruido: float,
    eps: float,
    cfg: FieldConfig,
    semillas: list[int],
    radio: int,
    pos_list: list[dict],
    verbose: bool = True,
) -> dict:
    cfg_r = replace(cfg, eps=eps, ruido=ruido)
    if verbose:
        print(f"\n{'='*60}")
        print(f"RUIDO={ruido} eps={eps} L={cfg_r.L} pasos={cfg_r.pasos}")
        print(f"{'='*60}")

    base_filas, base_stats = linea_base_caos(semillas, cfg_r, ruido)
    if verbose:
        print(
            f"baseline |m_A1-m_A2|: med={base_stats['mediana_max_diff']:.4f} "
            f"p90={base_stats['p90_max_diff']:.4f}"
        )

    resultados_pos = []
    todas = []
    for pinfo in pos_list:
        pos = pinfo["pos"]
        filas = evaluar_semillas(pos, semillas, cfg_r, ruido, base_stats, radio)
        agg = agregar_posicion(filas, pinfo["id"], pos)
        resultados_pos.append(agg)
        for f in filas:
            f["pos_id"] = pinfo["id"]
            f["ruido"] = ruido
            f["eps"] = eps
            todas.append(f)
        if verbose:
            print(
                f"  {pinfo['id']:<14} sigue={agg['frac_sigue_eps']:.2f} "
                f"agrega={agg['frac_eps_agrega']:.2f} causal={agg['frac_causal']:.2f} "
                f"dist_med={agg['dist_pico_mediana']:.1f}"
            )

    return {
        "ruido": ruido,
        "eps": eps,
        "baseline": base_stats,
        "baseline_filas": base_filas,
        "posiciones": resultados_pos,
        "corridas": todas,
    }


def parse_semillas(args: argparse.Namespace) -> list[int]:
    if args.quick:
        return list(range(1, 7))
    if args.semillas:
        if "-" in args.semillas and "," not in args.semillas:
            a, b = args.semillas.split("-", 1)
            return list(range(int(a), int(b) + 1))
        return [int(x.strip()) for x in args.semillas.split(",")]
    return SEMILLAS_DEFAULT


def main() -> None:
    parser = argparse.ArgumentParser(description="CG001 test causalidad ε")
    parser.add_argument("--semillas", type=str, default="")
    parser.add_argument("--L", type=int, default=48)
    parser.add_argument("--pasos", type=int, default=300)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--radio", type=int, default=RADIO_SEGUIMIENTO)
    parser.add_argument("--ruido", type=float, default=None)
    parser.add_argument("--ruidos", type=str, default="", help="0.074,0.02,0.007,...")
    parser.add_argument("--eps-list", type=str, default="", help="0.05,0.5,5.0")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    from cg001_field import PRODUCTION

    if args.production:
        cfg = PRODUCTION
    else:
        cfg = FieldConfig(L=args.L, pasos=args.pasos)

    semillas = parse_semillas(args)
    pos_list = posiciones_eps(cfg.L)

    if args.ruidos:
        ruidos = [float(x.strip()) for x in args.ruidos.split(",")]
    elif args.ruido is not None:
        ruidos = [args.ruido]
    else:
        ruidos = [1.0]

    if args.eps_list:
        epsilons = [float(x.strip()) for x in args.eps_list.split(",")]
    else:
        epsilons = [cfg.eps]

    print("=== CG001 — TEST CAUSALIDAD ε ===")
    print(f"ruidos={ruidos} epsilons={epsilons} semillas={len(semillas)} radio={args.radio}")

    bloques = []
    resumen_filas = []
    for ruido in ruidos:
        for eps in epsilons:
            bloque = correr_bloque(ruido, eps, cfg, semillas, args.radio, pos_list)
            bloques.append(bloque)
            for p in bloque["posiciones"]:
                resumen_filas.append({
                    "ruido": ruido,
                    "eps": eps,
                    **{k: v for k, v in p.items() if k != "eps_pos"},
                    "eps_pos": str(p["eps_pos"]),
                    "baseline_med": bloque["baseline"]["mediana_max_diff"],
                    "baseline_p90": bloque["baseline"]["p90_max_diff"],
                })

    print("\n=== RESUMEN (datos) ===")
    print(f"{'RUIDO':>8} {'eps':>6} {'pos':<14} {'sigue':>5} {'agrega':>6} {'causal':>6} {'base_p90':>9}")
    print("-" * 72)
    for row in resumen_filas:
        print(
            f"{row['ruido']:>8.4f} {row['eps']:>6.3f} {row['id']:<14} "
            f"{row['frac_sigue_eps']:>5.2f} {row['frac_eps_agrega']:>6.2f} "
            f"{row['frac_causal']:>6.2f} {row['baseline_p90']:>9.2f}"
        )

    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out_dir = LOGS / f"causalidad{tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tipo": "test_causalidad_eps",
        "cfg": cfg.__dict__,
        "ruidos": ruidos,
        "epsilons": epsilons,
        "radio_seguimiento": args.radio,
        "umbral_signo": UMBRAL_SIGNO,
        "semillas": semillas,
        "bloques": bloques,
        "resumen": resumen_filas,
    }
    json_path = out_dir / "resultado.json"
    csv_path = out_dir / "resumen.csv"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resumen_filas[0].keys()))
        w.writeheader()
        w.writerows(resumen_filas)
    print(f"\nGuardado: {json_path}\n         {csv_path}")


if __name__ == "__main__":
    main()