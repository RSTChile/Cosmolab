#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F4_2_motor.py — BATERÍA FUNDAMENTOS, Enfoque 4, experimento F4-2 ★
"Densidad vs expansión DESACOPLADAS: ¿efecto causal propio de ρ?"

Criterio congelado en PROTOCOLO_F4-2_PREREGISTRO.md (leer ANTES de este
archivo; este script tiene mtime posterior a ese protocolo, T3). No se
edita este motor para favorecer un resultado tras verlo.

No importa ni edita cs074_rcruz.py ni CF2_estiramiento_motor.py. Reescribe,
sin modificar el álgebra, las piezas de cs074_rcruz.py que necesita
(campo_inicial, difusión por aristas activas, medición de D, persistencia)
porque F4-2 necesita inyectar DOS cosas que esas funciones originales no
aceptan como parámetro: (1) un forzamiento dinámico por paso (T7), y (2) un
"gate" de densidad continuo sobre la fuerza de la difusión (§3 del
protocolo). Con density_gate=1.0 fijo, `paso_difusion_gated` reduce
ALGEBRAICAMENTE a `paso_difusion` de cs074_rcruz (verificado).

DESACOPLE (el corazón del experimento, ver protocolo §3):
  H = min(r_target * D0, 1)   — D0 medido, no puesto a mano (T1)
  EXPANSIÓN (topológica) ON  → cortar aristas activas ~ Bernoulli(H)/paso
                          OFF → topología 100% conectada todo el run
  DILUCIÓN  (cinética)   ON  → gate de densidad q(t) = min(exp(-3·H·t), 1)
                                sobre la FUERZA de la difusión (ρ(t)/ρ0=a(t)^-3,
                                a(t)=exp(H·t) — misma ley que CF2, n=3 fijo)
                          OFF → q(t) ≡ 1 (ρ fija, "compensada")

  rama 00 = OFF/OFF (control: nada pasa)
  rama a  = ON/OFF  (expandir con ρ fija)             [protocolo (a)]
  rama b  = OFF/ON  (diluir ρ sin expandir, a fijo)   [protocolo (b)]
  rama c  = ON/ON   (ambas juntas, natural ρ∝a⁻³)     [protocolo (c)]

Nota de consistencia (auto-chequeo, no es el resultado): con dilución OFF,
la rama `a` es matemáticamente IDÉNTICA al motor original de
cs074_rcruz.py a esa H — sirve de verificación cruzada contra el
experimento padre.

Segundo observable independiente (regla de verificación múltiple):
información mutua espacial antipodal (histograma conjunto discretizado),
método distinto de P (que es correlación×varianza).

Entrega crudo (φ final REAL y NULL de cada corrida) para auditoría en
disco por quien no escribió este código.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
WEB_ROOT = CODE_DIR.parents[2]  # .../Cosmogenesis-Web
OUT_DIR = WEB_ROOT / "results" / "BATERIA_FUNDAMENTOS" / "F4_2_densidad_expansion_desacopladas"

# ============================================================
# Congelado en el protocolo, §6 (barrido) y §3 (desacople)
# ============================================================
EPS_CANONICO = 1e-2
R_TARGETS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]
SEEDS_16 = [7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321,
            271828, 161803, 500500, 31415, 27182, 141421]
SIGMA_DYN = 1e-3      # amplitud fija del forzamiento dinámico (T7), §4
N_EXPONENTE_DILUCION = 3.0   # rho ~ a^-3, fijo aquí (barrer n es tarea de F3-3/F4-6)

P_LAVADO_THR = 0.05
MARGEN_LAVADO = 1.15
MAX_STEPS_CAL = 20000
CHECK_EVERY_CAL = 25

MI_BINS = 10

# ============================================================
# Criterio de decisión pre-registrado, §8
# ============================================================
Z_THR = 2.0
DELTA_MIN = 0.05
RATIO_CAUSAL_PROPIA = 0.5
RATIO_PROXY = 0.8
BANDA_R_DECISION = [5.0, 10.0, 30.0, 100.0]

RAMAS = {
    "00": {"expansion": False, "dilucion": False},
    "a":  {"expansion": True,  "dilucion": False},
    "b":  {"expansion": False, "dilucion": True},
    "c":  {"expansion": True,  "dilucion": True},
}


# ============================================================
# Motor de campo (reescritura fiel de cs074_rcruz.py + gate de densidad
# + forzamiento dinámico)
# ============================================================
def campo_inicial(N: int, eps: float, rng: np.random.Generator):
    """Idéntico a cs074_rcruz.campo_inicial (sin modificar el álgebra)."""
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    fondo = np.ones(N, dtype=float)
    if eps <= 0.0:
        return fondo, x
    pert = np.zeros(N, dtype=float)
    for m in range(1, 6):
        fase = rng.uniform(0, 2 * np.pi)
        pert += np.sin(2 * np.pi * m * x + fase) / m
    pert -= pert.mean()
    if pert.std() > 0:
        pert = pert / pert.std()
    return fondo + eps * pert, x


def paso_difusion_gated(phi: np.ndarray, activo: np.ndarray, q: float,
                         rng: np.random.Generator) -> np.ndarray:
    """
    Difusión por aristas activas, con gate de densidad `q` (probabilidad de
    que una arista topológicamente activa participe ESTE paso).

    q=1.0 -> ALGEBRAICAMENTE IDÉNTICO a cs074_rcruz.paso_difusion (mismo
    `nuevo = phi + 0.5*(media-phi)`, misma máscara n_nb>0). Verificado.
    """
    if q >= 1.0:
        eff = activo
    elif q <= 0.0:
        eff = np.zeros_like(activo)
    else:
        u = rng.random(activo.shape)
        eff = activo & (u < q)

    left = np.roll(phi, 1)
    right = np.roll(phi, -1)
    e_left = np.roll(eff, 1)
    e_right = eff
    n_nb = e_left.astype(np.float64) + e_right.astype(np.float64)
    s = e_left * left + e_right * right
    media = np.divide(s, n_nb, out=phi.copy(), where=n_nb > 0)
    nuevo = phi + 0.5 * (media - phi)
    return np.where(n_nb > 0, nuevo, phi)


def paso_expansion(activo: np.ndarray, H: float, rng: np.random.Generator) -> np.ndarray:
    """Idéntico a cs074_rcruz.paso_expansion (corte topológico permanente)."""
    if H <= 0.0:
        return activo
    activo = activo.copy()
    if H >= 1.0:
        activo[:] = False
        return activo
    u = rng.random(activo.shape)
    cortar = activo & (u < H)
    activo[cortar] = False
    return activo


def medir_D(N: int, eps: float, seed: int) -> float:
    """
    D0 de referencia: fracción de contraste borrada en UN paso de difusión
    pura (q=1, todas las aristas activas, sin ruido dinámico). Idéntico en
    espíritu a cs074_rcruz.medir_D.
    """
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    c0 = phi.std()
    if c0 <= 0:
        return 0.0
    phi1 = paso_difusion_gated(phi, activo, 1.0, rng)
    c1 = phi1.std()
    return max(0.0, float((c0 - c1) / c0))


def persistencia(phi: np.ndarray, contraste0: float) -> float:
    """Idéntico a cs074_rcruz.persistencia."""
    if contraste0 <= 0 or phi.std() <= 1e-12:
        return 0.0
    c = np.corrcoef(phi, np.roll(phi, 1))[0, 1]
    if not np.isfinite(c):
        c = 0.0
    c = max(0.0, float(c))
    v = float(phi.var() / (contraste0 ** 2))
    return float(c * v)


def mutual_info_antipodal(phi: np.ndarray, bins: int = MI_BINS) -> float:
    """
    Segundo observable independiente (protocolo §7): información mutua
    discreta entre la celda i y la celda i+N/2 (emparejamiento antipodal
    en el anillo), método distinto de la autocorrelación de P.
    """
    N = phi.size
    half = N // 2
    a = phi[:half]
    b = phi[half:half + half]
    joint, _, _ = np.histogram2d(a, b, bins=bins, range=[[0.0, 1.0], [0.0, 1.0]])
    total = joint.sum()
    if total <= 0:
        return 0.0
    joint = joint / total
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(joint > 0, joint / (pa * pb + 1e-300), 1.0)
        terms = np.where(joint > 0, joint * np.log(ratio + 1e-300), 0.0)
    return float(max(terms.sum(), 0.0))


def evolucionar_rama(N: int, eps: float, H: float, pasos: int, seed: int,
                      expansion_on: bool, dilucion_on: bool,
                      sigma_dyn: float = SIGMA_DYN, null: bool = False) -> dict:
    """
    Corre la dinámica desacoplada `pasos` pasos y devuelve φ final +
    diagnósticos. `null=True` baraja φ al final ("barajado del acople").
    """
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    c0 = float(phi.std())

    for t in range(pasos):
        if dilucion_on:
            q = float(min(np.exp(-N_EXPONENTE_DILUCION * H * t), 1.0))
        else:
            q = 1.0
        phi = paso_difusion_gated(phi, activo, q, rng)
        if sigma_dyn > 0:
            phi = phi + sigma_dyn * rng.normal(size=phi.shape)
            phi = np.clip(phi, 0.0, 1.0)
        if expansion_on:
            activo = paso_expansion(activo, H, rng)

    frac_exp = 1.0 - float(activo.mean())

    if null:
        phi = rng.permutation(phi)

    P = persistencia(phi, c0)
    MI = mutual_info_antipodal(phi)
    return {
        "P": P,
        "MI": MI,
        "frac_exp": frac_exp,
        "contraste0": c0,
        "phi_final": [round(float(v), 6) for v in phi],
    }


def medir_pasos_lavado_00(N: int, eps: float, seeds: list[int], sigma_dyn: float,
                           P_thr: float = P_LAVADO_THR, max_steps: int = MAX_STEPS_CAL,
                           check_every: int = CHECK_EVERY_CAL) -> dict:
    """
    Calibra pasos_fijo en la rama 00 (OFF/OFF: sin corte, sin dilución,
    CON ruido dinámico) — la más rápida en lavar. Mismo criterio que
    cs074_rcruz.medir_pasos_lavado (mediana de N semillas * MARGEN_LAVADO).
    """
    tiempos = []
    for s in seeds:
        rng = np.random.default_rng(s)
        phi, _ = campo_inicial(N, eps, rng)
        activo = np.ones(N, dtype=bool)
        c0 = float(phi.std())
        if c0 <= 0:
            tiempos.append(0)
            continue
        t_hit = None
        for t in range(1, max_steps + 1):
            phi = paso_difusion_gated(phi, activo, 1.0, rng)
            if sigma_dyn > 0:
                phi = phi + sigma_dyn * rng.normal(size=phi.shape)
                phi = np.clip(phi, 0.0, 1.0)
            if t % check_every == 0:
                if persistencia(phi, c0) < P_thr:
                    t_hit = t
                    break
        if t_hit is None:
            t_hit = max_steps
        tiempos.append(t_hit)
    med = int(np.median(tiempos))
    pasos = int(np.ceil(med * MARGEN_LAVADO))
    return {
        "tiempos": tiempos,
        "mediana": med,
        "pasos": pasos,
        "P_thr": P_thr,
        "lavo_todas": all(t < max_steps for t in tiempos),
    }


# ============================================================
# Barrido de producción
# ============================================================
def correr_grid(N: int, eps: float, seeds: list[int], r_targets: list[float],
                 D0: float, pasos_fijo: int, sigma_dyn: float = SIGMA_DYN,
                 guardar_phi: bool = True) -> dict:
    filas = []
    for r_tgt in r_targets:
        H = float(min(r_tgt * D0, 1.0)) if D0 > 0 else (1.0 if r_tgt > 0 else 0.0)
        r_eff = (H / D0) if D0 > 0 else float("inf")

        for rama, flags in RAMAS.items():
            Preal, Pnull, MIreal, MInull = [], [], [], []
            frac_exps = []
            per_seed_rows = []
            for seed in seeds:
                real = evolucionar_rama(N, eps, H, pasos_fijo, seed,
                                         flags["expansion"], flags["dilucion"],
                                         sigma_dyn=sigma_dyn, null=False)
                null = evolucionar_rama(N, eps, H, pasos_fijo, seed,
                                         flags["expansion"], flags["dilucion"],
                                         sigma_dyn=sigma_dyn, null=True)
                Preal.append(real["P"])
                Pnull.append(null["P"])
                MIreal.append(real["MI"])
                MInull.append(null["MI"])
                frac_exps.append(real["frac_exp"])
                row = {
                    "seed": seed,
                    "P_real": real["P"], "P_null": null["P"],
                    "MI_real": real["MI"], "MI_null": null["MI"],
                    "frac_exp": real["frac_exp"],
                }
                if guardar_phi:
                    row["phi_final_real"] = real["phi_final"]
                    row["phi_final_null"] = null["phi_final"]
                per_seed_rows.append(row)

            Preal_a = np.array(Preal)
            Pnull_a = np.array(Pnull)
            MIreal_a = np.array(MIreal)
            MInull_a = np.array(MInull)

            sd_p = np.sqrt((Preal_a.var() + Pnull_a.var()) / 2.0)
            sd_p = max(sd_p, 1.0 / max(len(Preal_a), 1) ** 0.5)
            z_p = float((Preal_a.mean() - Pnull_a.mean()) / sd_p)

            sd_mi = np.sqrt((MIreal_a.var() + MInull_a.var()) / 2.0)
            sd_mi = max(sd_mi, 1.0 / max(len(MIreal_a), 1) ** 0.5)
            z_mi = float((MIreal_a.mean() - MInull_a.mean()) / sd_mi)

            filas.append({
                "rama": rama,
                "expansion_on": flags["expansion"],
                "dilucion_on": flags["dilucion"],
                "r_target": r_tgt,
                "H": H,
                "r_eff": r_eff,
                "D0": D0,
                "P_real_mean": float(Preal_a.mean()),
                "P_real_std": float(Preal_a.std()),
                "P_null_mean": float(Pnull_a.mean()),
                "P_null_std": float(Pnull_a.std()),
                "z_P": z_p,
                "MI_real_mean": float(MIreal_a.mean()),
                "MI_real_std": float(MIreal_a.std()),
                "MI_null_mean": float(MInull_a.mean()),
                "MI_null_std": float(MInull_a.std()),
                "z_MI": z_mi,
                "frac_exp_mean": float(np.mean(frac_exps)),
                "n_seeds": len(seeds),
                "por_semilla": per_seed_rows,
            })
    return {"filas": filas}


def decidir_lectura(filas: list[dict], seeds: list[int]) -> dict:
    """Aplica el criterio pre-registrado del protocolo §8 sobre la banda de r."""
    def get(rama, r_tgt):
        for f in filas:
            if f["rama"] == rama and abs(f["r_target"] - r_tgt) < 1e-9:
                return f
        return None

    banda = BANDA_R_DECISION
    delta_a_list, delta_b_list, delta_c_list = [], [], []
    z_a_list, z_b_list, z_c_list = [], [], []
    detalle_banda = []
    for r_tgt in banda:
        f00 = get("00", r_tgt)
        fa = get("a", r_tgt)
        fb = get("b", r_tgt)
        fc = get("c", r_tgt)
        if not (f00 and fa and fb and fc):
            continue
        d_a = fa["P_real_mean"] - f00["P_real_mean"]
        d_b = fb["P_real_mean"] - f00["P_real_mean"]
        d_c = fc["P_real_mean"] - f00["P_real_mean"]
        delta_a_list.append(d_a)
        delta_b_list.append(d_b)
        delta_c_list.append(d_c)
        z_a_list.append(fa["z_P"])
        z_b_list.append(fb["z_P"])
        z_c_list.append(fc["z_P"])
        detalle_banda.append({
            "r_target": r_tgt,
            "P_00": f00["P_real_mean"], "P_a": fa["P_real_mean"],
            "P_b": fb["P_real_mean"], "P_c": fc["P_real_mean"],
            "delta_a": d_a, "delta_b": d_b, "delta_c": d_c,
            "z_a": fa["z_P"], "z_b": fb["z_P"], "z_c": fc["z_P"],
        })

    delta_a = float(np.mean(delta_a_list)) if delta_a_list else 0.0
    delta_b = float(np.mean(delta_b_list)) if delta_b_list else 0.0
    delta_c = float(np.mean(delta_c_list)) if delta_c_list else 0.0
    z_a = float(np.mean(z_a_list)) if z_a_list else 0.0
    z_b = float(np.mean(z_b_list)) if z_b_list else 0.0
    z_c = float(np.mean(z_c_list)) if z_c_list else 0.0

    sig_a = (z_a >= Z_THR) and (delta_a >= DELTA_MIN)
    sig_b = (z_b >= Z_THR) and (delta_b >= DELTA_MIN)
    sig_c = (z_c >= Z_THR) and (delta_c >= DELTA_MIN)

    ratio_bc = (delta_b / delta_c) if abs(delta_c) > 1e-12 else float("nan")
    ratio_ac = (delta_a / delta_c) if abs(delta_c) > 1e-12 else float("nan")

    lectura_1_densidad_causal_propia = bool(sig_b and (ratio_bc == ratio_bc) and ratio_bc >= RATIO_CAUSAL_PROPIA)
    lectura_2_densidad_proxy = bool((not sig_b) and sig_a and (ratio_ac == ratio_ac) and ratio_ac >= RATIO_PROXY)
    lectura_3_interaccion = bool(not lectura_1_densidad_causal_propia and not lectura_2_densidad_proxy)

    if lectura_1_densidad_causal_propia:
        etiqueta = "DENSIDAD_CAUSAL_PROPIA"
    elif lectura_2_densidad_proxy:
        etiqueta = "DENSIDAD_PROXY_DE_EXPANSION"
    else:
        etiqueta = "INTERACCION_DE_AMBAS"

    return {
        "banda_r": banda,
        "detalle_banda": detalle_banda,
        "delta_a_mean_banda": delta_a,
        "delta_b_mean_banda": delta_b,
        "delta_c_mean_banda": delta_c,
        "z_a_mean_banda": z_a,
        "z_b_mean_banda": z_b,
        "z_c_mean_banda": z_c,
        "sig_a": sig_a,
        "sig_b": sig_b,
        "sig_c": sig_c,
        "ratio_bc_dilucion_sobre_natural": ratio_bc,
        "ratio_ac_expansion_sobre_natural": ratio_ac,
        "umbrales": {
            "Z_THR": Z_THR, "DELTA_MIN": DELTA_MIN,
            "RATIO_CAUSAL_PROPIA": RATIO_CAUSAL_PROPIA, "RATIO_PROXY": RATIO_PROXY,
        },
        "lectura_1_densidad_causal_propia": lectura_1_densidad_causal_propia,
        "lectura_2_densidad_proxy_de_expansion": lectura_2_densidad_proxy,
        "lectura_3_interaccion_de_ambas": lectura_3_interaccion,
        "etiqueta": etiqueta,
    }


def gate_eps0(N: int, r_targets: list[float], seeds: list[int], D0: float, pasos_fijo: int) -> dict:
    """Gate barato pre-registrado: eps=0 -> P=0 en las 4 ramas, todo r (subconjunto de r y semillas)."""
    resultados = []
    r_sample = [r_targets[0], r_targets[len(r_targets) // 2], r_targets[-1]]
    seed_sample = seeds[:4]
    ok = True
    for r_tgt in r_sample:
        H = float(min(r_tgt * D0, 1.0)) if D0 > 0 else 0.0
        for rama, flags in RAMAS.items():
            for seed in seed_sample:
                real = evolucionar_rama(N, 0.0, H, pasos_fijo, seed,
                                         flags["expansion"], flags["dilucion"],
                                         sigma_dyn=SIGMA_DYN, null=False)
                if real["P"] > 1e-9:
                    ok = False
                resultados.append({"rama": rama, "r_target": r_tgt, "seed": seed, "P": real["P"]})
    return {"ok": ok, "n_checks": len(resultados), "detalle": resultados}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "produccion", "n400"], nargs="?", default="produccion")
    parser.add_argument("--no-phi", action="store_true", help="no guardar phi_final (JSON mas chico)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.mode == "smoke":
        N = 200
        seeds = SEEDS_16[:3]
        r_targets = [0.0, 1.0, 10.0]
        tag = "smoke"
    elif args.mode == "n400":
        N = 400
        seeds = SEEDS_16
        r_targets = R_TARGETS
        tag = "n400_confirmatorio"
    else:
        N = 200
        seeds = SEEDS_16
        r_targets = R_TARGETS
        tag = "produccion"

    guardar_phi = not args.no_phi

    print(f"=== F4-2 densidad/expansion desacopladas — modo={args.mode} N={N} n_seeds={len(seeds)} ===", flush=True)

    print("[1/4] midiendo D0 (referencia, eps canonico, sin ruido dinamico)...", flush=True)
    D0 = float(np.mean([medir_D(N, EPS_CANONICO, s) for s in seeds]))
    print(f"      D0 = {D0:.6f}", flush=True)

    print("[2/4] calibrando pasos_fijo en rama 00 (lavado con ruido dinamico)...", flush=True)
    cal = medir_pasos_lavado_00(N, EPS_CANONICO, seeds, SIGMA_DYN,
                                 max_steps=(MAX_STEPS_CAL if args.mode != "smoke" else 3000),
                                 check_every=(CHECK_EVERY_CAL if args.mode != "smoke" else 10))
    pasos_fijo = cal["pasos"]
    print(f"      mediana_lavado={cal['mediana']} pasos_fijo={pasos_fijo} lavo_todas={cal['lavo_todas']}", flush=True)
    print(f"      tiempos={cal['tiempos']}", flush=True)

    print("[3/4] gate eps=0 -> P=0 (subconjunto barato)...", flush=True)
    gate = gate_eps0(N, r_targets, seeds, D0, min(pasos_fijo, 2000))
    print(f"      gate_eps0_ok={gate['ok']} (n_checks={gate['n_checks']})", flush=True)

    print(f"[4/4] barrido principal: 4 ramas x {len(r_targets)} r x {len(seeds)} semillas x2(real,null)...", flush=True)
    grid = correr_grid(N, EPS_CANONICO, seeds, r_targets, D0, pasos_fijo,
                        sigma_dyn=SIGMA_DYN, guardar_phi=guardar_phi)
    elapsed = time.time() - t0
    print(f"      listo en {elapsed:.1f}s", flush=True)

    lectura = decidir_lectura(grid["filas"], seeds)
    print("\n=== LECTURA (numeros crudos, sin adjudicar en prosa) ===")
    print(json.dumps({k: v for k, v in lectura.items() if k != "detalle_banda"}, indent=2, ensure_ascii=False))

    payload = {
        "experimento": "F4-2 densidad vs expansion desacopladas",
        "tag": tag,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocolo": "PROTOCOLO_F4-2_PREREGISTRO.md",
        "parametros": {
            "N": N, "EPS_CANONICO": EPS_CANONICO, "SIGMA_DYN": SIGMA_DYN,
            "N_EXPONENTE_DILUCION": N_EXPONENTE_DILUCION,
            "D0": D0, "pasos_fijo": pasos_fijo,
            "seeds": seeds, "r_targets": r_targets,
            "ramas": RAMAS,
        },
        "calibracion_pasos": cal,
        "gate_eps0": gate,
        "grid": grid["filas"],
        "lectura_preregistrada": lectura,
        "runtime_seconds": elapsed,
    }

    suffix = "_no_phi" if args.no_phi else ""
    out_json = OUT_DIR / f"F4_2_{args.mode}_result{suffix}.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON -> {out_json}")
    print(f"[elapsed_total] {elapsed:.1f}s")


if __name__ == "__main__":
    main()
