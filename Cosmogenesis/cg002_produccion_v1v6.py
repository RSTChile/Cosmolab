"""
CG002 — PRODUCCIÓN de los veredictos V1, V2, V3, V6 (13-ago-2026)
=================================================================

Qué hace este archivo
---------------------
El protocolo `PROTOCOLO_CG002_ACOPLAMIENTO_ORIGINARIO.md` (29-jun-2026) pre-registró
seis veredictos. En junio sólo se corrió un smoke (N=2, 3 semillas) que dejó V4 y V5
en PASS y el control G limpio. V1, V2, V3 y V6 quedaron **pendientes de producción**
(el informe de avance pide N≥4 y S_max≥30 semillas).

Este script NO reescribe el motor: importa `cg002_acoplamiento.correr` tal cual y sólo
orquesta las corridas, arma los CSV y aplica los criterios **textuales** del protocolo.

Los tres brazos
---------------
  A) REAL      : α=1, firmas ω fijas por corrida  (condición principal §7.1)
  B) BARAJADO  : α=1, pero la asignación firma→nodo se re-permuta en CADA micro-paso.
                 HAY acoplamiento (misma α, mismo multiset de compatibilidades) pero la
                 relación concreta NO persiste. Separa "hay acoplamiento" de
                 "hay ESTE acoplamiento". (Brazo añadido, autorizado por el director.)
  C) ALFA_0    : α=0, control G del protocolo (se conserva para no romper el pre-registro).

Guarda de no-isomorfía (§ del informe): barajar UNA sola vez al inicio es un simple
renombre de nodos y da observables idénticos hasta la última cifra — por eso se baraja
por paso. La verificación numérica está en `verificar_no_isomorfia()`.

Salidas
-------
  cg002_produccion_series.csv        (bloque θ_CP=0: V1, V2, V6)
  cg002_produccion_theta_series.csv  (bloque θ_CP≠0: V3)
  cg002_produccion_resumen.json      (todas las fracciones crudas + tests)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr

from cg002_acoplamiento import (
    KAPPA_S,
    K_MAX_MIN,
    TAU_CAP,
    CG002Config,
    correr,
    matriz_compat_circulo,
)

RAIZ = Path(__file__).resolve().parent

# --- Diseño de producción -------------------------------------------------
N_LIST = [4, 6, 8]          # informe: N≥4
SEEDS = list(range(1, 31))  # informe: S_max≥30 semillas (el director pedía ≥12)
THETAS = [0.1, 0.3, 0.5]    # los mismos θ_CP del smoke pre-registrado (§11)
N_LIST_THETA = [4, 8]
K_MATCH = 500               # presupuesto de micro-pasos igualado entre brazos

BRAZOS = {
    # nombre: (alpha, barajar_por_paso)
    "REAL": (1.0, False),
    "BARAJADO": (1.0, True),
    "ALFA_0": (0.0, False),
}


def _cfg(brazo: str, n: int, seed: int, theta: float = 0.0, k_max: int = K_MAX_MIN,
         tau_cap: int = TAU_CAP) -> CG002Config:
    alpha, baraja = BRAZOS[brazo]
    return CG002Config(N=n, alpha=alpha, seed=seed, k_max=k_max, tau_cap=tau_cap,
                       theta_cp=theta, barajar_por_paso=baraja)


def _coop_input(omega: list[int], k: int = 8) -> tuple[float, float]:
    """Cuánta cooperación es LEÍBLE en el input ω (para el test 'ρ no es input')."""
    c = matriz_compat_circulo(np.asarray(omega), k)
    n = len(omega)
    iu = np.triu_indices(n, 1)
    vals = c[iu]
    if vals.size == 0:
        return 0.0, 0.0
    return float(np.mean(vals > 0)), float(np.mean(vals))


# =========================================================================
# GUARDA: el barajado no puede ser isomorfo al real
# =========================================================================
def verificar_no_isomorfia() -> dict[str, Any]:
    """Barajar una vez = renombrar nodos (isomorfo). Barajar por paso = no isomorfo."""
    out: dict[str, Any] = {}
    difs = []
    for seed in SEEDS[:10]:
        a = correr(_cfg("REAL", 8, seed))
        b = correr(_cfg("BARAJADO", 8, seed))
        difs.append({
            "seed": seed,
            "tau_real": a["tau"], "tau_baraj": b["tau"],
            "vivos_real": a["n_vivos"], "vivos_baraj": b["n_vivos"],
            "arist_real": a["n_aristas"], "arist_baraj": b["n_aristas"],
            "rho_real": a["rho"], "rho_baraj": b["rho"],
            "S_ordenado_identico": bool(np.allclose(sorted(a["S_final"]), sorted(b["S_final"]))),
        })
    out["por_paso"] = difs
    out["frac_semillas_identicas"] = sum(1 for d in difs if d["S_ordenado_identico"]) / len(difs)
    return out


# =========================================================================
# Producción
# =========================================================================
def bloque_principal() -> list[dict[str, Any]]:
    """θ_CP = 0. Alimenta V1, V2, V6 (y de paso V4/V5 con más semillas)."""
    filas: list[dict[str, Any]] = []
    for n in N_LIST:
        for brazo in BRAZOS:
            for seed in SEEDS:
                r = correr(_cfg(brazo, n, seed))
                # sonda con presupuesto de micro-pasos IGUALADO entre brazos
                rk = correr(_cfg(brazo, n, seed, k_max=K_MATCH))
                frac_c_pos, media_c = _coop_input(r["omega"])
                s_fin = np.asarray(r["S_final"], dtype=float)
                filas.append({
                    "brazo": brazo, "N": n, "semilla": seed, "alpha": r["alpha"],
                    "theta_cp": 0.0,
                    "tau_final": r["tau"], "k_micro": r["k_micro"],
                    "alcanzo_tau_cap": int(r["tau"] >= TAU_CAP),
                    "n_vivos_fin": r["n_vivos"],
                    "vivos_ge2_fin": int(r["n_vivos"] >= 2),
                    "n_vivos_kmatch": rk["n_vivos"],
                    "vivos_ge2_kmatch": int(rk["n_vivos"] >= 2),
                    "vivos_ge2_en_taucap": int(r["tau"] >= TAU_CAP and r["n_vivos"] >= 2),
                    "n_aristas": r["n_aristas"], "diametro": r["diametro"],
                    "componentes": r["componentes"],
                    "delta_struct": r["delta_struct"],
                    "rho": r["rho"], "phi_frustracion": r["phi_frustracion"],
                    "ds_max": r["ds_max"], "cn4_positivo": int(bool(r["cn4_positivo"])),
                    "S_final_max": float(s_fin.max()) if s_fin.size else 0.0,
                    "S_final_suma": float(s_fin.sum()),
                    "n_con_borde": int(np.sum(np.asarray(r["S_final"]) > KAPPA_S)),
                    "frac_c_positiva_input": frac_c_pos, "media_c_input": media_c,
                    "competencia_pasos": r["eventos"]["competencia_pasos"],
                    "fusion_aristas_max": r["eventos"]["fusion_aristas_max"],
                    "aniquilacion": int(r["eventos"]["aniquilacion"]),
                    "omega": " ".join(str(x) for x in r["omega"]),
                })
    return filas


def bloque_theta() -> list[dict[str, Any]]:
    """θ_CP ≠ 0. Único régimen donde V3 (dirección) es evaluable."""
    filas: list[dict[str, Any]] = []
    for n in N_LIST_THETA:
        for theta in THETAS:
            for brazo in BRAZOS:
                for seed in SEEDS:
                    r = correr(_cfg(brazo, n, seed, theta=theta))
                    fn = np.asarray(r["flujo_neto"], dtype=float) if r["flujo_neto"] else np.zeros(n)
                    filas.append({
                        "brazo": brazo, "N": n, "semilla": seed, "alpha": r["alpha"],
                        "theta_cp": theta,
                        "tau_final": r["tau"], "n_vivos_fin": r["n_vivos"],
                        "n_aristas": r["n_aristas"],
                        "t_hat_mod": r["t_hat_mod"], "asimetria": r["asimetria"],
                        "rho": r["rho"], "rho_menos_medio": r["rho"] - 0.5,
                        "tracking_score": r["tracking_score"] if r["tracking_score"] is not None else "",
                        "degenerada": int(bool(r["degenerada"])),
                        "tracking_limpio": int(bool(r["tracking_limpio"])),
                        "flujo_neto_suma": float(fn.sum()),
                        "flujo_neto_medio": float(fn.mean()) if fn.size else 0.0,
                        "criterio_B": r["criterio_B"],
                        "ds_max": r["ds_max"],
                    })
    return filas


# =========================================================================
# Veredictos con los criterios TEXTUALES del protocolo §9.1
# =========================================================================
def _frac(rows: list[dict], campo: str) -> float:
    return sum(r[campo] for r in rows) / len(rows) if rows else float("nan")


def veredictos(filas: list[dict], filas_theta: list[dict]) -> dict[str, Any]:
    V: dict[str, Any] = {}

    # ---- V1: fracción de semillas con ≥2 nodos vivos a τ_cap, α=1 vs α=0 ----
    v1: dict[str, Any] = {}
    for n in N_LIST:
        sub = {b: [r for r in filas if r["N"] == n and r["brazo"] == b] for b in BRAZOS}
        d: dict[str, Any] = {}
        for lect, campo in [("a_final_corrida", "vivos_ge2_fin"),
                            ("en_tau_cap", "vivos_ge2_en_taucap"),
                            ("k_igualado_500", "vivos_ge2_kmatch")]:
            d[lect] = {b: _frac(sub[b], campo) for b in BRAZOS}
        # test contra cada control, lectura principal = a_final_corrida
        for ctrl in ("BARAJADO", "ALFA_0"):
            a = sum(r["vivos_ge2_fin"] for r in sub["REAL"])
            b = sum(r["vivos_ge2_fin"] for r in sub[ctrl])
            tabla = [[a, len(sub["REAL"]) - a], [b, len(sub[ctrl]) - b]]
            d[f"fisher_vs_{ctrl}"] = float(fisher_exact(tabla, alternative="greater")[1])
        d["positivo_vs_ALFA_0"] = d["a_final_corrida"]["REAL"] > d["a_final_corrida"]["ALFA_0"]
        d["positivo_vs_BARAJADO"] = d["a_final_corrida"]["REAL"] > d["a_final_corrida"]["BARAJADO"]
        v1[f"N{n}"] = d
    V["V1"] = v1

    # ---- V2: τ_final > 0 en ≥83% con α=1; τ ≈ 0 con α=0 ----
    v2: dict[str, Any] = {}
    for b in BRAZOS:
        sub = [r for r in filas if r["brazo"] == b]
        v2[b] = {
            "frac_tau_pos_global": sum(1 for r in sub if r["tau_final"] > 0) / len(sub),
            "por_N": {f"N{n}": sum(1 for r in sub if r["N"] == n and r["tau_final"] > 0)
                              / max(1, len([r for r in sub if r["N"] == n])) for n in N_LIST},
            "tau_mediana": float(np.median([r["tau_final"] for r in sub])),
        }
    v2["criterio"] = "α=1: frac(τ>0) ≥ 0.83 ; α=0: τ ≈ 0"
    v2["positivo_REAL"] = v2["REAL"]["frac_tau_pos_global"] >= 0.83
    v2["positivo_control_G"] = v2["ALFA_0"]["frac_tau_pos_global"] < 0.17
    v2["positivo_BARAJADO"] = v2["BARAJADO"]["frac_tau_pos_global"] >= 0.83
    V["V2"] = v2

    # ---- V3: |T̂| y ρ con signo estable ≥83% entre semillas del mismo N ----
    v3: dict[str, Any] = {}
    for n in N_LIST_THETA:
        for theta in THETAS:
            for b in ("REAL", "BARAJADO"):
                sub = [r for r in filas_theta if r["N"] == n and r["theta_cp"] == theta
                       and r["brazo"] == b]
                if not sub:
                    continue
                that = np.array([r["t_hat_mod"] for r in sub])
                rho = np.array([r["rho"] for r in sub])
                # lectura LITERAL (|T̂| es un módulo, ρ∈[0,1]: el signo no puede ser otro)
                lit_that = float(np.mean(that >= 0))
                lit_rho = float(np.mean(rho >= 0))
                # lectura no trivial: ¿el sistema cae del mismo lado?
                nz = that[that > 1e-12]
                trk = np.array([float(r["tracking_score"]) for r in sub
                                if r["tracking_score"] != "" and not r["degenerada"]])
                sg_trk = float(np.mean(np.sign(trk) == np.sign(np.mean(trk)))) if trk.size else float("nan")
                rho_c = rho - 0.5
                sg_rho = float(np.mean(np.sign(rho_c) == np.sign(np.mean(rho_c)))) if rho_c.size else float("nan")
                v3[f"N{n}_theta{theta}_{b}"] = {
                    "n": len(sub),
                    "literal_frac_signo_T_hat": lit_that,
                    "literal_frac_signo_rho": lit_rho,
                    "frac_T_hat_no_nulo": float(nz.size / len(sub)),
                    "media_T_hat": float(that.mean()),
                    "media_asimetria": float(np.mean([r["asimetria"] for r in sub])),
                    "signo_estable_tracking": sg_trk,
                    "frac_degenerada": float(np.mean([r["degenerada"] for r in sub])),
                    "signo_estable_rho_menos_0.5": sg_rho,
                    "media_rho": float(rho.mean()),
                }
    # ¿ρ es input? correlación con la cooperación legible en ω
    sub1 = [r for r in filas if r["brazo"] == "REAL"]
    rho_v = [r["rho"] for r in sub1]
    v3["rho_es_input"] = {
        "spearman_rho_vs_frac_c_positiva_input": list(map(float, spearmanr(rho_v, [r["frac_c_positiva_input"] for r in sub1]))),
        "spearman_rho_vs_media_c_input": list(map(float, spearmanr(rho_v, [r["media_c_input"] for r in sub1]))),
    }
    V["V3"] = v3

    # ---- V6: ∂̂S > ∂_crit(2.0) en ≥1 nodo en ≥50% semillas (α=1, N≥4) ----
    v6: dict[str, Any] = {}
    for b in BRAZOS:
        sub = [r for r in filas if r["brazo"] == b and r["N"] >= 4]
        v6[b] = {
            "frac_cn4": _frac(sub, "cn4_positivo"),
            "ds_max_mediana": float(np.median([r["ds_max"] for r in sub])),
            "ds_max_p90": float(np.percentile([r["ds_max"] for r in sub], 90)),
            "ds_max_maximo": float(np.max([r["ds_max"] for r in sub])),
            "por_N": {f"N{n}": _frac([r for r in sub if r["N"] == n], "cn4_positivo") for n in N_LIST},
        }
    v6["criterio"] = "frac(∂̂S_max > 2.0) ≥ 0.50"
    v6["positivo_REAL"] = v6["REAL"]["frac_cn4"] >= 0.5
    v6["positivo_BARAJADO"] = v6["BARAJADO"]["frac_cn4"] >= 0.5
    # cota estructural: ∂̂S_max ≤ nº de nodos con B>0 ⇒ hacen falta ≥3 bordes activos
    v6["cota_estructural"] = ("∂̂S_i = B_i/mean(B|B>0) ⇒ ∂̂S_max ≤ m (m = nodos con borde). "
                              "Con m=1 vale exactamente 1; con m=2, <2. Superar 2.0 exige m≥3.")
    for b in BRAZOS:
        sub = [r for r in filas if r["brazo"] == b and r["N"] >= 4]
        v6[b]["frac_m_ge3"] = float(np.mean([1 if r["n_aristas"] >= 2 else 0 for r in sub]))
    V["V6"] = v6

    # ---- identidades algebraicas sospechosas ----
    sub = [r for r in filas if r["brazo"] == "REAL"]
    ident = {}
    pares = [("ds_max", "n_aristas"), ("ds_max", "n_vivos_fin"), ("ds_max", "S_final_max"),
             ("delta_struct", "tau_final"), ("rho", "n_vivos_fin"), ("tau_final", "n_vivos_fin")]
    for a, b in pares:
        r_, p_ = spearmanr([x[a] for x in sub], [x[b] for x in sub])
        ident[f"spearman_{a}_vs_{b}"] = [float(r_), float(p_)]
    V["identidades"] = ident

    # ---- piso de ruido ----
    a = len([r for r in filas if r["brazo"] == "REAL" and r["N"] == 8])
    V["piso_de_ruido"] = {
        "n_por_brazo_por_N": len(SEEDS),
        "fisher_2x2_30vs30_separacion_total_p": float(fisher_exact([[30, 0], [0, 30]], alternative="greater")[1]),
        "nota": ("Con 30 semillas por brazo y por N, el p más chico alcanzable en una tabla 2x2 "
                 "con separación total es el de arriba; ninguna fracción puede dar un p menor. "
                 "Las fracciones se reportan crudas, sin binarizar por umbrales nuevos."),
    }
    return V


def main() -> None:
    print("== CG002 producción V1/V2/V3/V6 ==")
    print("[1/4] guarda de no-isomorfía del brazo barajado...")
    iso = verificar_no_isomorfia()
    print("     semillas con S_final idéntico (debe ser 0.0):", iso["frac_semillas_identicas"])

    print("[2/4] bloque principal θ_CP=0 ...")
    filas = bloque_principal()
    cols = list(filas[0].keys())
    with (RAIZ / "cg002_produccion_series.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(filas)
    print(f"     {len(filas)} corridas -> cg002_produccion_series.csv")

    print("[3/4] bloque θ_CP≠0 (V3) ...")
    filas_t = bloque_theta()
    with (RAIZ / "cg002_produccion_theta_series.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_t[0].keys()))
        w.writeheader()
        w.writerows(filas_t)
    print(f"     {len(filas_t)} corridas -> cg002_produccion_theta_series.csv")

    print("[4/4] veredictos ...")
    V = veredictos(filas, filas_t)
    salida = {
        "protocolo": "PROTOCOLO_CG002_ACOPLAMIENTO_ORIGINARIO.md (29-jun-2026)",
        "diseno": {"N": N_LIST, "semillas": len(SEEDS), "brazos": list(BRAZOS),
                   "thetas": THETAS, "k_max": K_MAX_MIN, "tau_cap": TAU_CAP,
                   "k_igualado": K_MATCH},
        "guarda_no_isomorfia": iso,
        "veredictos": V,
    }
    with (RAIZ / "cg002_produccion_resumen.json").open("w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(json.dumps(V, indent=2, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
