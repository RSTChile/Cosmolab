#!/usr/bin/env python3
"""
CG001 — Compuerta paso 2: tres criterios en paralelo (A, B, C)

A — Localización en arruga: max(m)_B en centro ± radio; memoria nuclear B >> A.
B — Perfil espacial distinto: corr(m_A, m_B) baja; B ≠ A en forma (sin exigir centro).
C — Magnitud absoluta: max(m) y sum(núcleo) B >> A; concentración NO entra al veredicto.

Multi-semilla por punto RUIDO. Certificación: fracción de semillas que pasan (umbral 0.83).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from cg001_field import FieldConfig, PRODUCTION, correr, signo_estable

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"

RADIO_ARRUGA = 2
UMBRAL_SIGNO = 0.83

# A
UMBRAL_RATIO_NUCLEO_A = 2.0
UMBRAL_MAX_ABS_A = 1.5

# B
UMBRAL_CORR_B = 0.98  # por debajo = perfiles distinguibles
UMBRAL_MAX_ABS_B = 1.1
UMBRAL_ARGMAX_SEP_B = 3.0  # celdas entre picos A y B

# C (solo magnitudes absolutas / ratios directos, sin concentracion)
UMBRAL_MAX_ABS_C = 1.5
UMBRAL_SUM_NUCLEO_C = 2.0
UMBRAL_SUM_TOTAL_C = 1.05

BANDA_DEFAULT = [0.074, 0.02, 0.007, 0.003, 0.001]
SEMILLAS_DEFAULT = list(range(1, 7))


def _indice_centro(L: int) -> tuple[int, int, int]:
    c = L // 2
    return c, c, c


def _distancia(idx: tuple[int, int, int], centro: tuple[int, int, int]) -> float:
    return float(np.linalg.norm(np.subtract(idx, centro)))


def _argmax_3d(m: np.ndarray) -> tuple[int, int, int]:
    flat = int(np.argmax(m))
    return np.unravel_index(flat, m.shape)


def _mascara_nucleo(L: int, radio: int) -> np.ndarray:
    c = L // 2
    z, y, x = np.ogrid[:L, :L, :L]
    return (np.abs(z - c) <= radio) & (np.abs(y - c) <= radio) & (np.abs(x - c) <= radio)


def analizar_campo(m: np.ndarray, cfg: FieldConfig, *, radio: int = RADIO_ARRUGA) -> dict:
    L = cfg.L
    centro = _indice_centro(L)
    idx_max = _argmax_3d(m)
    mask = _mascara_nucleo(L, radio)
    m_nuc = float(m[mask].sum())
    m_tot = float(m.sum())
    return {
        "argmax": idx_max,
        "centro": centro,
        "dist_max_al_centro": _distancia(idx_max, centro),
        "max_m": float(m.max()),
        "mean_m": float(m.mean()),
        "sum_m": m_tot,
        "sum_nucleo": m_nuc,
        "frac_nucleo": m_nuc / (m_tot + 1e-12),
        "en_arruga": _distancia(idx_max, centro) <= radio + 0.5,
    }


def perfil_ab(m_a: np.ndarray, m_b: np.ndarray) -> dict:
    a = m_a.ravel().astype(np.float64)
    b = m_b.ravel().astype(np.float64)
    sa, sb = a.std(), b.std()
    if sa < 1e-15 and sb < 1e-15:
        corr = 1.0
    elif sa < 1e-15 or sb < 1e-15:
        corr = 0.0
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    ia, ib = _argmax_3d(m_a), _argmax_3d(m_b)
    return {
        "corr_m": corr,
        "argmax_sep": _distancia(ia, ib),
        "argmax_A": ia,
        "argmax_B": ib,
    }


def criterio_a(la: dict, lb: dict) -> dict:
    checks = {
        "b_max_en_arruga": lb["en_arruga"],
        "b_sum_nucleo_ratio": (
            lb["sum_nucleo"] > la["sum_nucleo"] * UMBRAL_RATIO_NUCLEO_A
            if la["sum_nucleo"] > 1e-9
            else lb["sum_nucleo"] > 1e-6
        ),
        "b_max_abs": lb["max_m"] > la["max_m"] * UMBRAL_MAX_ABS_A,
    }
    pasa = all(checks.values())
    return {"id": "A", "nombre": "localizacion_arruga", "pasa": pasa, "checks": checks}


def criterio_b(la: dict, lb: dict, perfil: dict) -> dict:
    checks = {
        "perfil_distinto": abs(perfil["corr_m"]) < UMBRAL_CORR_B,
        "argmax_separados": perfil["argmax_sep"] >= UMBRAL_ARGMAX_SEP_B,
        "b_max_mayor": lb["max_m"] > la["max_m"] * UMBRAL_MAX_ABS_B,
    }
    # B pasa si perfil distinto (corr O separación de picos) Y B tiene más pico
    pasa = checks["b_max_mayor"] and (checks["perfil_distinto"] or checks["argmax_separados"])
    return {
        "id": "B",
        "nombre": "perfil_espacial_distinto",
        "pasa": pasa,
        "checks": checks,
        "corr_m": perfil["corr_m"],
        "argmax_sep": perfil["argmax_sep"],
    }


def criterio_c(la: dict, lb: dict) -> dict:
    checks = {
        "b_max_abs": lb["max_m"] > la["max_m"] * UMBRAL_MAX_ABS_C,
        "b_sum_nucleo": (
            lb["sum_nucleo"] > la["sum_nucleo"] * UMBRAL_SUM_NUCLEO_C
            if la["sum_nucleo"] > 1e-9
            else lb["sum_nucleo"] > 1e-6
        ),
        "b_sum_total": lb["sum_m"] > la["sum_m"] * UMBRAL_SUM_TOTAL_C,
    }
    pasa = checks["b_max_abs"] and (checks["b_sum_nucleo"] or checks["b_sum_total"])
    artefacto_solo_ratio = (
        la["concentracion"] > 1e-6
        and lb["concentracion"] > la["concentracion"] * 3
        and not checks["b_max_abs"]
    )
    return {
        "id": "C",
        "nombre": "magnitud_absoluta",
        "pasa": pasa,
        "checks": checks,
        "artefacto_solo_ratio": artefacto_solo_ratio,
    }


def corrida_par(seed: int, ruido: float, cfg: FieldConfig) -> dict:
    ra = correr(False, seed=seed, cfg=cfg, ruido=ruido, retornar_campos=True)
    rb = correr(True, seed=seed, cfg=cfg, ruido=ruido, retornar_campos=True)
    la = analizar_campo(ra["m"], cfg)
    lb = analizar_campo(rb["m"], cfg)
    la["concentracion"] = ra["concentracion"]
    lb["concentracion"] = rb["concentracion"]
    perfil = perfil_ab(ra["m"], rb["m"])
    ca = criterio_a(la, lb)
    cb = criterio_b(la, lb, perfil)
    cc = criterio_c(la, lb)
    return {
        "seed": seed,
        "ruido": ruido,
        "A": la,
        "B": lb,
        "perfil": perfil,
        "dif_concentracion": lb["concentracion"] - la["concentracion"],
        "dif_max_m": lb["max_m"] - la["max_m"],
        "criterios": {"A": ca, "B": cb, "C": cc},
    }


def agregar_por_ruido(corridas: list[dict]) -> dict:
    por_ruido: dict[float, list[dict]] = {}
    for c in corridas:
        por_ruido.setdefault(c["ruido"], []).append(c)

    filas = []
    for ruido in sorted(por_ruido.keys(), reverse=True):
        grupo = por_ruido[ruido]
        n = len(grupo)
        fila: dict = {"ruido": ruido, "n_semillas": n}
        for cid in ("A", "B", "C"):
            pasan = [1.0 if c["criterios"][cid]["pasa"] else 0.0 for c in grupo]
            fraccion = float(np.mean(pasan))
            fila[f"{cid}_frac_pasa"] = fraccion
            fila[f"{cid}_certifica"] = fraccion >= UMBRAL_SIGNO
            fila[f"{cid}_n_pasa"] = int(sum(pasan))
        difs_conc = [c["dif_concentracion"] for c in grupo]
        mu, sg = signo_estable(difs_conc)
        fila["dif_conc_mean"] = mu
        fila["dif_conc_signo"] = sg
        filas.append(fila)
    return {"filas": filas, "por_ruido": por_ruido}


def imprimir_resumen(agg: dict) -> None:
    print(f"\n{'RUIDO':>8} | {'A':>5} {'B':>5} {'C':>5} | {'Δconc':>8} {'sgn':>5} | certifica (>=0.83)")
    print("-" * 62)
    for f in agg["filas"]:
        a = f["A_frac_pasa"]
        b = f["B_frac_pasa"]
        c = f["C_frac_pasa"]
        cert = []
        for k in ("A", "B", "C"):
            if f[f"{k}_certifica"]:
                cert.append(k)
        marca = ",".join(cert) if cert else "—"
        print(
            f"{f['ruido']:>8.4f} | {a:>5.2f} {b:>5.2f} {c:>5.2f} | "
            f"{f['dif_conc_mean']:>+8.3f} {f['dif_conc_signo']:>5.2f} | {marca}"
        )
    print("-" * 62)
    print("A=arruga  B=perfil distinto  C=magnitud absoluta (sin concentración)")


def main() -> None:
    parser = argparse.ArgumentParser(description="CG001 compuerta: criterios A, B, C")
    parser.add_argument("--ruidos", type=str, default="", help="Lista coma: 0.074,0.02,...")
    parser.add_argument("--semillas", type=str, default="", help="Lista coma o rango 1-6")
    parser.add_argument("--L", type=int, default=48)
    parser.add_argument("--pasos", type=int, default=300)
    parser.add_argument("--production", action="store_true", help="L=64 pasos=400")
    parser.add_argument("--compare", action="store_true", help="Banda default 5 puntos")
    parser.add_argument("--ruido", type=float, default=None, help="Un solo punto")
    args = parser.parse_args()

    cfg = PRODUCTION if args.production else FieldConfig(L=args.L, pasos=args.pasos)

    if args.ruidos:
        ruidos = [float(x.strip()) for x in args.ruidos.split(",")]
    elif args.compare or args.ruido is None:
        ruidos = list(BANDA_DEFAULT)
    else:
        ruidos = [args.ruido]

    if args.semillas:
        if "-" in args.semillas and "," not in args.semillas:
            a, b = args.semillas.split("-", 1)
            semillas = list(range(int(a), int(b) + 1))
        else:
            semillas = [int(x.strip()) for x in args.semillas.split(",")]
    else:
        semillas = SEMILLAS_DEFAULT

    n = len(ruidos) * len(semillas) * 2
    print("=== CG001 — COMPUERTA A + B + C (multi-semilla) ===")
    print(f"L={cfg.L} pasos={cfg.pasos} semillas={semillas}")
    print(f"ruidos={ruidos}")
    print(f"corridas={n}\n")

    corridas = []
    for ri, ruido in enumerate(ruidos):
        for seed in semillas:
            r = corrida_par(seed, ruido, cfg)
            corridas.append(r)
            ca, cb, cc = r["criterios"]["A"], r["criterios"]["B"], r["criterios"]["C"]
            print(
                f"RUIDO={ruido:.4f} s={seed:2d} | "
                f"A:{'✓' if ca['pasa'] else '✗'} "
                f"B:{'✓' if cb['pasa'] else '✗'} "
                f"C:{'✓' if cc['pasa'] else '✗'} | "
                f"Δconc={r['dif_concentracion']:+.2f} corr={r['perfil']['corr_m']:.3f}"
            )
        print(f"  ... punto {ri+1}/{len(ruidos)}")
        print()

    agg = agregar_por_ruido(corridas)
    imprimir_resumen(agg)

    banda = {}
    for cid in ("A", "B", "C"):
        pts = [f["ruido"] for f in agg["filas"] if f[f"{cid}_certifica"]]
        banda[cid] = pts
        if pts:
            print(f"BANDA {cid}: RUIDO en [{min(pts):.6f}, {max(pts):.6f}] ({len(pts)} puntos)")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = LOGS / f"compuerta_abc_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "tipo": "compuerta_abc",
        "cfg": cfg.__dict__,
        "umbrales": {
            "signo": UMBRAL_SIGNO,
            "A": {"radio": RADIO_ARRUGA, "ratio_nucleo": UMBRAL_RATIO_NUCLEO_A, "max_abs": UMBRAL_MAX_ABS_A},
            "B": {"corr_max": UMBRAL_CORR_B, "argmax_sep": UMBRAL_ARGMAX_SEP_B, "max_abs": UMBRAL_MAX_ABS_B},
            "C": {"max_abs": UMBRAL_MAX_ABS_C, "sum_nucleo": UMBRAL_SUM_NUCLEO_C, "sum_total": UMBRAL_SUM_TOTAL_C},
        },
        "ruidos": ruidos,
        "semillas": semillas,
        "agregado": agg["filas"],
        "banda_por_criterio": banda,
        "corridas": corridas,
    }

    json_path = out_dir / "resultado.json"
    csv_path = out_dir / "resumen.csv"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(agg["filas"][0].keys()))
        w.writeheader()
        w.writerows(agg["filas"])
    print(f"\nGuardado: {json_path}\n         {csv_path}")


if __name__ == "__main__":
    main()