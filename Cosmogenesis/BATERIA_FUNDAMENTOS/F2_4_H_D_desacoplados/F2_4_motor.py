#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F2_4_motor.py — Expansión (H) y reabsorción (D) como ejes 2D INDEPENDIENTES
============================================================================

Pregunta (spec F2-4, BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md líneas 152-159):
¿de verdad manda solo la razón r = H/D, o H y D importan por separado? Barrer H y D
en un grid 2D independiente (no solo su cociente) y ver si los puntos colapsan sobre
una curva única P(r).

Protocolo fijado ANTES de correr este motor: F2_4_PROTOCOLO_PREREGISTRO.md (mismo
directorio). No se cambia ningún criterio después de ver resultados (T3).

Relación con el código base cs074_rcruz.py (leído, NO editado):
  - Misma física: dominio periódico 1D, campo inicial = 1 + eps·(5 modos Fourier +
    fase aleatoria), difusión SOLO por aristas vivas, expansión = corte Bernoulli de
    aristas por paso, observable de persistencia = autocorrelación×varianza, NULL =
    permutación final de phi.
  - ÚNICA generalización: en cs074_rcruz el coeficiente de mezcla de la difusión está
    fijo en la constante 0.5 dentro de paso_difusion (`nuevo = phi + 0.5*(media-phi)`).
    Ahí es donde vive físicamente "D" (fuerza de reabsorción por paso) en el modelo
    original, pero NUNCA se barre — está fijo, y D se mide luego indirectamente
    (fracción de contraste borrada en un paso) solo para DERIVAR H = r_target * D.
    Aquí ese 0.5 se convierte en el parámetro D que se barre exactamente igual que H
    (mismo tipo de cantidad: tasa por paso, D∈(0,1]). H conserva la fórmula original
    de paso_expansion sin cambios.
  - Esto hace a H y D dos palancas físicas verdaderamente independientes, en vez de
    una D medida (fija) y una H derivada de un r objetivo — que es exactamente lo que
    la spec de F2-4 pide (T1/T7: "dos ejes barridos, no un cociente asumido").

Diseño del grid: H_list = D_list = misma secuencia geométrica de 8 valores. Así cada
diagonal (i-j=k constante) del grid 8x8 comparte EXACTAMENTE el mismo r=H/D con
escalas absolutas de (H,D) distintas — la prueba más directa de "mismo r, distinta
física absoluta, ¿mismo P?".
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent

# ---- Parámetros fijados en el pre-registro (NO tocar tras ver resultados) ----
N = 200
EPS = 1e-2
SEMILLAS = 12
P_LAVADO = 0.05
MARGEN_LAVADO = 1.15
GRID_MIN = 0.080
GRID_MAX = 0.900
GRID_N = 8


def grid_geometrico(vmin, vmax, n):
    return [float(v) for v in np.geomspace(vmin, vmax, n)]


GRID_8 = grid_geometrico(GRID_MIN, GRID_MAX, GRID_N)
H_LIST = list(GRID_8)
D_LIST = list(GRID_8)


# ---------------------------------------------------------------------------
# Física: idéntica a cs074_rcruz.py salvo D paramétrico en paso_difusion
# (ver docstring del módulo — generalización única y declarada).
# ---------------------------------------------------------------------------

def campo_inicial(N, eps, rng):
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


def paso_difusion(phi, activo, D):
    """
    Difusión solo por aristas vivas. IDÉNTICA a cs074_rcruz.paso_difusion salvo que
    el coeficiente de mezcla (0.5 en el original) es aquí el parámetro D barrido.
    """
    left = np.roll(phi, 1)
    right = np.roll(phi, -1)
    e_left = np.roll(activo, 1)
    e_right = activo
    n_nb = e_left.astype(np.float64) + e_right.astype(np.float64)
    s = e_left * left + e_right * right
    media = np.divide(s, n_nb, out=phi.copy(), where=n_nb > 0)
    nuevo = phi + D * (media - phi)
    return np.where(n_nb > 0, nuevo, phi)


def paso_expansion(activo, H, rng):
    """Corte de aristas Bernoulli(H) por paso. IDÉNTICA a cs074_rcruz.paso_expansion."""
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


def evolucionar(phi, activo, H, D, pasos, rng, null=False):
    contraste0 = float(phi.std())
    for _ in range(pasos):
        phi = paso_difusion(phi, activo, D)
        activo = paso_expansion(activo, H, rng)
    if null:
        phi = rng.permutation(phi)
    return phi, activo, contraste0


def persistencia(phi, contraste0):
    if contraste0 <= 0 or phi.std() <= 1e-12:
        return 0.0
    c = np.corrcoef(phi, np.roll(phi, 1))[0, 1]
    if not np.isfinite(c):
        c = 0.0
    c = max(0.0, float(c))
    v = float(phi.var() / (contraste0 ** 2))
    return float(c * v)


def medir_D_emp(N, eps, D, seed):
    """Fracción de contraste borrada en UN paso de difusión pura (H=0), a D dado.
    Diagnóstico/cross-check, NO es el D barrido (ver protocolo §1)."""
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    c0 = phi.std()
    if c0 <= 0:
        return 0.0
    phi1 = paso_difusion(phi, activo, D)
    c1 = phi1.std()
    return max(0.0, float((c0 - c1) / c0))


def medir_pasos_lavado(N, eps, D, semillas, P_thr=P_LAVADO, max_steps=300000, check_every=200):
    tiempos = []
    for s in range(semillas):
        rng = np.random.default_rng(20_000 + s)
        phi, _ = campo_inicial(N, eps, rng)
        activo = np.ones(N, dtype=bool)
        c0 = float(phi.std())
        if c0 <= 0:
            tiempos.append(0)
            continue
        t_hit = None
        for t in range(1, max_steps + 1):
            phi = paso_difusion(phi, activo, D)
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
        "D_calibracion": D,
    }


def corrida(N, eps, H, D, pasos, seed, null=False):
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    phi, activo, c0 = evolucionar(phi, activo, H, D, pasos, rng, null=null)
    P = persistencia(phi, c0)
    frac_exp = 1.0 - float(activo.mean())
    return {"P": P, "frac_exp": frac_exp, "std_ratio": float(phi.std() / c0) if c0 > 0 else 0.0}


def barrido_grid(N, eps, H_list, D_list, semillas, pasos_fijo):
    filas = []
    for i, H in enumerate(H_list):
        for j, D in enumerate(D_list):
            r = H / D
            D_emp = float(np.mean([medir_D_emp(N, eps, D, 5000 + s) for s in range(semillas)]))
            Preal, Pnull = [], []
            for s in range(semillas):
                seed = 1000 + s
                rr = corrida(N, eps, H, D, pasos_fijo, seed=seed, null=False)
                nn = corrida(N, eps, H, D, pasos_fijo, seed=seed, null=True)
                Preal.append(rr["P"])
                Pnull.append(nn["P"])
            Preal = np.array(Preal)
            Pnull = np.array(Pnull)
            sd = np.sqrt((Preal.var() + Pnull.var()) / 2.0)
            sd = max(sd, 1.0 / max(len(Preal), 1))
            z = float((Preal.mean() - Pnull.mean()) / sd)
            filas.append(
                {
                    "i": i,
                    "j": j,
                    "k_diag": i - j,
                    "H": H,
                    "D": D,
                    "r": r,
                    "D_emp": D_emp,
                    "P_real": float(Preal.mean()),
                    "P_null": float(Pnull.mean()),
                    "P_real_std": float(Preal.std()),
                    "P_null_std": float(Pnull.std()),
                    "P_real_sem": float(Preal.std() / np.sqrt(semillas)),
                    "z": z,
                }
            )
            print(
                f"[{i},{j}] H={H:.4f} D={D:.4f} r={r:.4f} P_real={Preal.mean():.4f} "
                f"P_null={Pnull.mean():.4f} z={z:.2f}",
                file=sys.stderr,
                flush=True,
            )
    return filas


def analisis_diagonales(filas, sigma_semilla):
    from collections import defaultdict

    por_k = defaultdict(list)
    for f in filas:
        por_k[f["k_diag"]].append(f)
    tol = max(3 * sigma_semilla, 0.03)
    resultado = []
    for k in sorted(por_k.keys()):
        grupo = por_k[k]
        if len(grupo) < 2:
            continue
        Ps = np.array([g["P_real"] for g in grupo])
        spread = float(Ps.std())
        resultado.append(
            {
                "k_diag": k,
                "r": grupo[0]["r"],
                "n_puntos": len(grupo),
                "P_real_values": [round(float(p), 6) for p in Ps],
                "spread_std": round(spread, 6),
                "tol": round(tol, 6),
                "colapsa": bool(spread <= tol),
                "H_D_pairs": [(round(g["H"], 4), round(g["D"], 4)) for g in grupo],
            }
        )
    return resultado, tol


def regresion_bilineal(filas):
    """P_real ~ a*log10(H) + b*log10(D) + c, ponderada por 1/SEM^2.
    Devuelve a, b, a+b, SE(a+b) vía mínimos cuadrados ponderados con matriz de
    covarianza."""
    X = []
    y = []
    w = []
    for f in filas:
        X.append([np.log10(f["H"]), np.log10(f["D"]), 1.0])
        y.append(f["P_real"])
        sem = f["P_real_sem"] if f["P_real_sem"] > 1e-9 else 1e-9
        w.append(1.0 / (sem ** 2))
    X = np.array(X)
    y = np.array(y)
    w = np.array(w)
    W = np.diag(w)
    XtW = X.T @ W
    cov_beta = np.linalg.inv(XtW @ X)
    beta = cov_beta @ (XtW @ y)
    a, b, c = beta
    var_a_plus_b = cov_beta[0, 0] + cov_beta[1, 1] + 2 * cov_beta[0, 1]
    se_a_plus_b = float(np.sqrt(max(var_a_plus_b, 0.0)))
    return {
        "a_log10H": float(a),
        "b_log10D": float(b),
        "c": float(c),
        "desbalance_a_mas_b": float(a + b),
        "SE_a_mas_b": se_a_plus_b,
        "pasa_3SE": bool(abs(a + b) <= 3 * se_a_plus_b),
    }


def main():
    t0 = time.time()
    ts_inicio = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[inicio] {ts_inicio}", file=sys.stderr, flush=True)
    print(f"[grid] H_list=D_list={GRID_8}", file=sys.stderr, flush=True)

    D_min_cal = min(D_LIST)
    cal = medir_pasos_lavado(N, EPS, D_min_cal, SEMILLAS)
    pasos_fijo = cal["pasos"]
    print(
        f"[calibracion] D_min={D_min_cal} mediana_lavado={cal['mediana']} "
        f"pasos_fijo={pasos_fijo} lavo_todas={cal['lavo_todas']} tiempos={cal['tiempos']}",
        file=sys.stderr,
        flush=True,
    )

    filas = barrido_grid(N, EPS, H_LIST, D_LIST, SEMILLAS, pasos_fijo)

    sigma_semilla = float(np.mean([f["P_real_sem"] for f in filas]))
    diagonales, tol = analisis_diagonales(filas, sigma_semilla)
    bilineal = regresion_bilineal(filas)

    n_diag_total = len(diagonales)
    n_diag_pasa = sum(1 for d in diagonales if d["colapsa"])

    z_vals = np.array([f["z"] for f in filas])
    r_vals = np.array([f["r"] for f in filas])
    mask_r_gt1 = r_vals > 1.5
    frac_z_null_muerde = float(np.mean(z_vals[mask_r_gt1] > 1.0)) if mask_r_gt1.any() else None

    ts_fin = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    elapsed = time.time() - t0

    result = {
        "experimento": "F2_4_H_D_desacoplados",
        "protocolo": "F2_4_PROTOCOLO_PREREGISTRO.md",
        "ts_inicio_utc": ts_inicio,
        "ts_fin_utc": ts_fin,
        "elapsed_s": elapsed,
        "N": N,
        "eps": EPS,
        "semillas": SEMILLAS,
        "grid_8": GRID_8,
        "H_list": H_LIST,
        "D_list": D_LIST,
        "calibracion_pasos": cal,
        "pasos_fijo": pasos_fijo,
        "filas": filas,
        "sigma_semilla_pooled": sigma_semilla,
        "diagonales": diagonales,
        "tol_colapso": tol,
        "n_diagonales_total": n_diag_total,
        "n_diagonales_colapsan": n_diag_pasa,
        "frac_diagonales_colapsan": n_diag_pasa / n_diag_total if n_diag_total else None,
        "regresion_bilineal": bilineal,
        "gate_null_muerde_r_gt_1.5_frac_z_gt_1": frac_z_null_muerde,
    }

    out_json = OUT / "F2_4_resultado.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[archivo] {out_json}", file=sys.stderr)
    print(f"[diagonales] {n_diag_pasa}/{n_diag_total} colapsan (tol={tol:.4f})", file=sys.stderr)
    print(f"[bilineal] a={bilineal['a_log10H']:.4f} b={bilineal['b_log10D']:.4f} "
          f"a+b={bilineal['desbalance_a_mas_b']:.4f} SE={bilineal['SE_a_mas_b']:.4f} "
          f"pasa_3SE={bilineal['pasa_3SE']}", file=sys.stderr)
    print(f"[elapsed] {elapsed:.1f}s ({elapsed/60:.1f} min)", file=sys.stderr)


if __name__ == "__main__":
    main()
