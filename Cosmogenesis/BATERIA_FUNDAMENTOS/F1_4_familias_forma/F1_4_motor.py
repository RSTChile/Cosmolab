#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1-4 — Independencia de la forma inicial: barrido de familias de perturbación
===============================================================================

Pregunta: ¿el mapa de persistencia P(eps,r) de cs074_rcruz.py depende de la
forma arbitraria con que se siembra la mancha inicial (multi-modo de Fourier,
la única familia usada hasta ahora), o es invariante a la familia elegida?

Protocolo congelado ANTES de esta corrida:
  PROTOCOLO_F1-4_PREREGISTRO.md (mismo directorio)

Física reutilizada SIN MODIFICAR de cs074_rcruz.py (import directo, no copia
a mano): paso_difusion, paso_expansion, persistencia, detectar_cuantizacion,
temperatura_fisica, reloj_fisico. Lo único que se generaliza aquí es la
función que siembra el campo inicial — cs074_rcruz.campo_inicial() solo sabe
sembrar la familia "multi_modo"; este motor define un generador por familia
y reusa el resto de la dinámica intacta.

6 familias (definidas en el pre-registro, ninguna elegida para favorecer el
resultado — T0/T1):
  1. multi_modo       — baseline, idéntico a cs074_rcruz.campo_inicial()
  2. modo_unico        — un solo modo de Fourier (m aleatorio por semilla)
  3. bulto_gaussiano    — bulto localizado (centro y ancho aleatorios)
  4. ruido_blanco       — espectro de potencia plano
  5. ruido_rojo         — espectro de potencia ~ 1/k^2 (domina escala grande)
  6. ruido_azul         — espectro de potencia ~ k^2 (domina escala chica)

Normalización común (para que eps sea comparable entre familias): pert -= mean,
pert /= std (si std>0), phi = 1 + eps*pert.

NULL: permutación del campo final (rng.permutation), por familia — igual
receta que cs074_rcruz, aplicada de forma independiente a cada familia.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # Cosmogenesis/
OUT_RES = HERE / "resultados"

# --- Import SIN MODIFICAR del código base (T-regla: no se copia a mano) ---
_spec = importlib.util.spec_from_file_location("cs074_rcruz", ROOT / "cs074_rcruz.py")
cs074 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs074)

paso_difusion = cs074.paso_difusion
paso_expansion = cs074.paso_expansion
persistencia = cs074.persistencia
detectar_cuantizacion = cs074.detectar_cuantizacion
temperatura_fisica = cs074.temperatura_fisica
T_SING = cs074.T_SING

P_LAVADO = 0.05
MARGEN_LAVADO = 1.15
R_TARGETS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]  # idéntico a cs074_rcruz
EPS_REPRESENTATIVO_CALIBRACION = 1e-2  # para calibrar pasos_fijo por familia (ver protocolo §5)

FAMILIAS = [
    "multi_modo",
    "modo_unico",
    "bulto_gaussiano",
    "ruido_blanco",
    "ruido_rojo",
    "ruido_azul",
]


def _normalizar(pert: np.ndarray) -> np.ndarray:
    pert = pert - pert.mean()
    s = pert.std()
    if s > 0:
        pert = pert / s
    return pert


def _dist_circular(x, x0):
    d = np.abs(x - x0)
    return np.minimum(d, 1.0 - d)


def _ruido_espectral(N, rng, alpha):
    """Ruido con espectro de amplitud |A(k)| ~ k^alpha, fase aleatoria, síntesis
    por FFT real inversa. alpha=0 blanco, alpha=-1 rojo (domina escala grande),
    alpha=+1 azul (domina escala chica). k=0 (DC) siempre amplitud 0 (se resta
    la media de todas formas)."""
    nk = N // 2 + 1
    k = np.arange(nk, dtype=float)
    amp = np.zeros(nk, dtype=float)
    amp[1:] = k[1:] ** alpha
    fase = rng.uniform(0, 2 * np.pi, size=nk)
    espectro = amp * np.exp(1j * fase)
    señal = np.fft.irfft(espectro, n=N)
    return señal


def campo_familia(N, eps, rng, familia):
    """Devuelve (phi, x) para la familia dada. Generaliza cs074_rcruz.campo_inicial()
    (que solo cubre 'multi_modo') a las 6 familias del protocolo F1-4."""
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    fondo = np.ones(N, dtype=float)
    if eps <= 0.0:
        return fondo, x

    if familia == "multi_modo":
        # idéntico a cs074_rcruz.campo_inicial(): suma de modos m=1..5, fase aleatoria
        pert = np.zeros(N, dtype=float)
        for m in range(1, 6):
            fase = rng.uniform(0, 2 * np.pi)
            pert += np.sin(2 * np.pi * m * x + fase) / m
    elif familia == "modo_unico":
        m = int(rng.integers(1, 9))  # 1..8, no se fija "el" modo
        fase = rng.uniform(0, 2 * np.pi)
        pert = np.sin(2 * np.pi * m * x + fase)
    elif familia == "bulto_gaussiano":
        x0 = rng.uniform(0.0, 1.0)
        sigma = rng.uniform(0.02, 0.08)
        d = _dist_circular(x, x0)
        pert = np.exp(-(d ** 2) / (2 * sigma ** 2))
    elif familia == "ruido_blanco":
        pert = _ruido_espectral(N, rng, alpha=0.0)
    elif familia == "ruido_rojo":
        pert = _ruido_espectral(N, rng, alpha=-1.0)
    elif familia == "ruido_azul":
        pert = _ruido_espectral(N, rng, alpha=1.0)
    else:
        raise ValueError(f"familia desconocida: {familia}")

    pert = _normalizar(pert)
    return fondo + eps * pert, x


def evolucionar(phi, activo, H, pasos, rng, null=False):
    contraste0 = float(phi.std())
    for _ in range(pasos):
        phi = paso_difusion(phi, activo)
        activo = paso_expansion(activo, H, rng)
    if null:
        phi = rng.permutation(phi)
    return phi, activo, contraste0


def medir_D(N, eps, seed, familia):
    rng = np.random.default_rng(seed)
    phi, _ = campo_familia(N, eps, rng, familia)
    activo = np.ones(N, dtype=bool)
    c0 = phi.std()
    if c0 <= 0:
        return 0.0
    phi1 = paso_difusion(phi, activo)
    c1 = phi1.std()
    return max(0.0, float((c0 - c1) / c0))


def medir_pasos_lavado(N, eps, semillas, familia, P_thr=P_LAVADO, max_steps=50000, check_every=50):
    tiempos = []
    for s in range(semillas):
        rng = np.random.default_rng(20_000 + s)
        phi, _ = campo_familia(N, eps, rng, familia)
        activo = np.ones(N, dtype=bool)
        c0 = float(phi.std())
        if c0 <= 0:
            tiempos.append(0)
            continue
        t_hit = None
        for t in range(1, max_steps + 1):
            phi = paso_difusion(phi, activo)
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
        "max_steps": max_steps,
        "lavo_todas": all(t < max_steps for t in tiempos),
    }


def corrida(N, eps, H, pasos, seed, familia, null=False):
    rng = np.random.default_rng(seed)
    phi, _ = campo_familia(N, eps, rng, familia)
    activo = np.ones(N, dtype=bool)
    phi, activo, c0 = evolucionar(phi, activo, H, pasos, rng, null=null)
    P = persistencia(phi, c0)
    cuantos = detectar_cuantizacion(phi, activo)
    frac_exp = 1.0 - float(activo.mean())
    T_fin = temperatura_fisica(frac_exp)
    return {
        "P": P,
        "cuantos": cuantos,
        "frac_exp": frac_exp,
        "T_fin_K": T_fin,
        "std_ratio": float(phi.std() / c0) if c0 > 0 else 0.0,
    }


def barrido_familia(N, eps_list, r_targets, semillas, familia, pasos_fijo, log):
    filas = []
    meta_por_eps = []
    for eps in eps_list:
        if eps <= 0:
            D = 0.0
            pasos = pasos_fijo
        else:
            D = float(np.mean([medir_D(N, eps, s, familia) for s in range(semillas)]))
            pasos = pasos_fijo
        meta_por_eps.append({"eps": eps, "D": D, "pasos": pasos})

        for r_tgt in r_targets:
            if D > 0:
                H = float(min(r_tgt * D, 1.0))
                r_eff = H / D if D > 0 else float("inf")
            else:
                H = 0.0 if r_tgt == 0 else 1.0
                r_eff = float("inf") if D <= 0 and r_tgt > 0 else 0.0

            Preal, Pnull, Tfin, srr, srn, fracs = [], [], [], [], [], []
            hist_real = {}
            for s in range(semillas):
                rr = corrida(N, eps, H, pasos, seed=1000 + s, familia=familia, null=False)
                nn = corrida(N, eps, H, pasos, seed=1000 + s, familia=familia, null=True)
                Preal.append(rr["P"])
                Pnull.append(nn["P"])
                Tfin.append(rr["T_fin_K"])
                srr.append(rr["std_ratio"])
                srn.append(nn["std_ratio"])
                fracs.append(rr["frac_exp"])
                for k, c in rr["cuantos"].items():
                    hist_real[k] = hist_real.get(k, 0) + c
            Preal = np.array(Preal)
            Pnull = np.array(Pnull)
            sd = np.sqrt((Preal.var() + Pnull.var()) / 2.0)
            sd = max(sd, 1.0 / max(len(Preal), 1))
            z = float((Preal.mean() - Pnull.mean()) / sd)
            filas.append(
                {
                    "familia": familia,
                    "eps": eps,
                    "r_target": r_tgt,
                    "H": H,
                    "D": D,
                    "r": r_eff,
                    "pasos": pasos,
                    "P_real": float(Preal.mean()),
                    "P_null": float(Pnull.mean()),
                    "P_real_std": float(Preal.std()),
                    "P_null_std": float(Pnull.std()),
                    "z": z,
                    "std_ratio_real": float(np.mean(srr)),
                    "std_ratio_null": float(np.mean(srn)),
                    "frac_exp_mean": float(np.mean(fracs)),
                    "T_fin_K": float(np.mean(Tfin)),
                    "cuantos_k": {int(k): int(v) for k, v in sorted(hist_real.items())},
                }
            )
        log(f"  [familia={familia}] eps={eps:.3e} D={D:.5f} pasos={pasos} -> {len(r_targets)} puntos r listos")
    return filas, meta_por_eps


def control_r0_ok(filas, P_max=0.15):
    rows = [f for f in filas if f["r_target"] == 0.0 and f["eps"] > 0]
    if not rows:
        return False, {}
    mean_P = float(np.mean([f["P_real"] for f in rows]))
    return mean_P < P_max, {"mean_P_r0_eps_gt0": mean_P, "n": len(rows), "P_max": P_max}


def control_eps0_ok(filas, P_max=0.05, frac_min=0.95):
    rows = [f for f in filas if f["eps"] == 0.0]
    if not rows:
        return False, {}
    ok = [f["P_real"] < P_max for f in rows]
    frac_ok = float(np.mean(ok))
    return frac_ok >= frac_min, {"frac_ok": frac_ok, "n": len(rows), "P_max": P_max}


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    t0 = time.time()
    OUT_RES.mkdir(parents=True, exist_ok=True)

    log_path = OUT_RES / "F1_4_log_ejecucion.txt"
    log_f = open(log_path, "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, file=sys.stderr, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    log(f"=== F1-4 motor arrancando modo={modo} ===")

    if modo == "smoke":
        N = 100
        semillas = 4
        eps_list = [0.0] + list(np.round(np.logspace(-3, 0, 4), 8))
        r_targets = [0.0, 1.0, 10.0, 100.0]
        familias = FAMILIAS
        max_steps_cal = 20000
    elif modo == "produccion":
        N = 200
        semillas = 12
        eps_list = [0.0] + list(np.round(np.logspace(-4, 0, 8), 10))
        r_targets = R_TARGETS
        familias = FAMILIAS
        max_steps_cal = 50000
    else:
        raise SystemExit(f"modo desconocido: {modo} (usa smoke|produccion)")

    log(f"N={N} semillas={semillas} eps_list={eps_list} r_targets={r_targets} familias={familias}")

    resultado_por_familia = {}
    calibraciones = {}
    for familia in familias:
        t_fam = time.time()
        log(f"[familia={familia}] calibrando D/pasos en eps={EPS_REPRESENTATIVO_CALIBRACION} (repr.)")
        cal = medir_pasos_lavado(
            N, EPS_REPRESENTATIVO_CALIBRACION, min(semillas, 8), familia,
            max_steps=max_steps_cal,
        )
        pasos_fijo = cal["pasos"]
        calibraciones[familia] = cal
        log(f"[familia={familia}] pasos_fijo={pasos_fijo} lavo_todas={cal['lavo_todas']} mediana={cal['mediana']}")

        filas, meta = barrido_familia(N, eps_list, r_targets, semillas, familia, pasos_fijo, log)
        ok_r0, det_r0 = control_r0_ok(filas)
        ok_eps0, det_eps0 = control_eps0_ok(filas)
        resultado_por_familia[familia] = {
            "familia": familia,
            "calibracion": cal,
            "pasos_fijo": pasos_fijo,
            "meta_por_eps": meta,
            "filas": filas,
            "control_r0_lava": ok_r0,
            "control_r0_detail": det_r0,
            "control_eps0_ok": ok_eps0,
            "control_eps0_detail": det_eps0,
            "elapsed_s": time.time() - t_fam,
        }
        log(f"[familia={familia}] listo en {time.time()-t_fam:.1f}s "
            f"control_r0_lava={ok_r0} control_eps0_ok={ok_eps0}")

    def rnd(f):
        out = dict(f)
        for k in ("D", "H", "r", "P_real", "P_null", "P_real_std", "P_null_std",
                   "std_ratio_real", "std_ratio_null"):
            if k in out and isinstance(out[k], float):
                out[k] = round(out[k], 6)
        out["z"] = round(out["z"], 3)
        return out

    for familia in familias:
        resultado_por_familia[familia]["filas"] = [rnd(f) for f in resultado_por_familia[familia]["filas"]]

    result = {
        "experimento": "F1-4",
        "modo": modo,
        "N": N,
        "semillas": semillas,
        "eps_list": eps_list,
        "r_targets": r_targets,
        "familias": familias,
        "eps_representativo_calibracion": EPS_REPRESENTATIVO_CALIBRACION,
        "por_familia": resultado_por_familia,
        "elapsed_s": time.time() - t0,
        "timestamp_fin": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_json = OUT_RES / f"F1_4_{modo}_resultado.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[archivo] {out_json}")
    log(f"[elapsed] {result['elapsed_s']:.1f}s")
    log("=== F1-4 motor terminado ===")
    log_f.close()


if __name__ == "__main__":
    main()
