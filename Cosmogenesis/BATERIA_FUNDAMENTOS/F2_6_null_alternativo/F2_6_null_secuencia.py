#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F2-6 — NULL alternativo: barajar la historia de cortes, no el acople final
============================================================================

Motor de producción. Ver PROTOCOLO_F2-6_PREREGISTRO.md (congelado antes de
este código) para la hipótesis, el observable, los dos nulos y el criterio de
PASS exactos.

Este script NO edita `cs074_rcruz.py` (raíz de Cosmogenesis/): lo IMPORTA y
reutiliza sus funciones de física (campo_inicial, paso_difusion,
paso_expansion, medir_D, medir_pasos_lavado, persistencia, R_TARGETS) sin
copiarlas a mano, para que la dinámica y el observable sean byte-idénticos al
resto del Enfoque 2.

Qué agrega este script (lo que NO está en cs074_rcruz.py):
  - `evolucionar_con_historia()`: corre la dinámica REAL grabando, por
    arista, el paso exacto en que fue cortada (`cut_step`). El código base
    solo guarda el estado final de `activo`, no CUÁNDO se cortó cada arista.
  - `null_secuencia()`: baraja el vector de tiempos de corte ENTRE las
    aristas que se cortaron (mismo conjunto final de aristas cortadas, mismo
    histograma de cortes-por-paso) y re-corre la difusión desde el mismo φ
    inicial bajo ese calendario alternativo. Mide P sobre el resultado — sin
    permutar el campo final. Éste es el segundo NULL que pide F2-6.
  - `null_clasico()`: el NULL ya existente (permutar el campo final), aplicado
    aquí sobre el mismo φ_real_final para que ambos nulos compartan
    exactamente la misma corrida REAL de base ("sobre los mismos casos").
  - Auto-chequeo `grafo_final_identico`: verifica que el NULL-secuencia
    preserva el grafo final exacto (T6 — chequeo que puede fallar).

Anti-Shannon: nada de esto fija ε, r o N a mano fuera de los rangos
pre-registrados; D y pasos de lavado se miden, no se imponen (igual que
cs074_rcruz.py).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# --- importar el código base SIN editarlo ---
_THIS = Path(__file__).resolve()
_COSMOGENESIS_DIR = _THIS.parents[2]  # .../F2_6_null_alternativo -> BATERIA_FUNDAMENTOS -> Cosmogenesis
sys.path.insert(0, str(_COSMOGENESIS_DIR))
import cs074_rcruz as base  # noqa: E402

OUT = _THIS.parent / "resultados"
OUT.mkdir(exist_ok=True)

# Offsets deterministas para separar los streams de RNG del barajado de cada
# NULL del stream de la dinámica REAL (evita correlacionar el azar del NULL
# con el azar de la física).
OFFSET_NULL_CLASICO = 500_000
OFFSET_NULL_SEQ = 900_000

Z_PASS = 3.0
BANDA_CONGELAMIENTO_R = 10.0
EPS_MIN_BANDA = 1e-6


def evolucionar_con_historia(phi, activo, H, pasos, rng):
    """
    Idéntico en física a base.evolucionar(..., null=False) pero grabando,
    para cada arista, el paso exacto (1-indexado) en que se corta.

    cut_step[e] == -1  -> la arista e nunca se cortó en el horizonte de pasos.
    cut_step[e] == t    -> la arista e se cortó DURANTE el paso t (queda
                            inactiva para la difusión del paso t+1 en adelante).

    Reutiliza base.paso_difusion y base.paso_expansion sin modificarlos, en
    el MISMO orden que base.evolucionar (difusión, luego expansión), para que
    el consumo de rng —y por tanto phi_final y activo_final— sea bit-idéntico
    al que produciría cs074_rcruz.py con el mismo seed/H/pasos.
    """
    N = activo.size
    contraste0 = float(phi.std())
    cut_step = np.full(N, -1, dtype=np.int64)
    for t in range(1, pasos + 1):
        phi = base.paso_difusion(phi, activo)
        antes = activo
        activo = base.paso_expansion(activo, H, rng)
        recien_cortadas = antes & (~activo)
        if recien_cortadas.any():
            cut_step[recien_cortadas] = t
    return phi, activo, contraste0, cut_step


def null_clasico(phi_real_final, seed):
    """NULL ya usado en cs074_rcruz.py: permuta el campo final. Independiente
    del NULL-secuencia; comparten el mismo phi_real_final (misma corrida
    REAL de base)."""
    rng_c = np.random.default_rng(seed + OFFSET_NULL_CLASICO)
    return rng_c.permutation(phi_real_final)


def null_secuencia(phi0, N, pasos, cut_step, seed):
    """
    Baraja la SECUENCIA temporal de cortes (no el grafo final):
    - toma las aristas que SÍ se cortaron en REAL y sus tiempos de corte,
    - permuta esos tiempos ENTRE esas mismas aristas (mismo conjunto final
      de aristas cortadas, mismo histograma de cortes-por-paso),
    - reconstruye activo(t) para t=1..pasos a partir del cut_step barajado,
    - re-corre la difusión desde el MISMO campo inicial phi0 bajo ese
      calendario alternativo. No se permuta el campo resultante.

    Devuelve (phi_final_shuffled, activo_final_shuffled, cut_step_shuffled).
    """
    rng_s = np.random.default_rng(seed + OFFSET_NULL_SEQ)
    cortadas_mask = cut_step >= 0
    idx_cortadas = np.where(cortadas_mask)[0]
    cut_step_shuf = cut_step.copy()
    if idx_cortadas.size > 0:
        tiempos = cut_step[idx_cortadas].copy()
        tiempos_perm = rng_s.permutation(tiempos)
        cut_step_shuf[idx_cortadas] = tiempos_perm

    phi = phi0.copy()
    activo = np.ones(N, dtype=bool)
    for t in range(1, pasos + 1):
        phi = base.paso_difusion(phi, activo)
        # arista activa en el paso t+1 en adelante ssi no fue "cortada" en
        # un paso <= t bajo el calendario barajado
        activo = (cut_step_shuf < 0) | (cut_step_shuf > t)
    return phi, activo, cut_step_shuf


def z_score(preal, pnull):
    preal = np.asarray(preal, dtype=float)
    pnull = np.asarray(pnull, dtype=float)
    sd = np.sqrt((preal.var() + pnull.var()) / 2.0)
    sd = max(sd, 1.0 / max(len(preal), 1))
    return float((preal.mean() - pnull.mean()) / sd)


def corrida_f26(N, eps, H, pasos, seed):
    """Una combinación (eps,r,seed): 1 corrida REAL con historia + los dos
    nulos derivados de ella. Devuelve dict con las tres P y el auto-chequeo."""
    rng = np.random.default_rng(seed)
    phi0, _ = base.campo_inicial(N, eps, rng)
    activo0 = np.ones(N, dtype=bool)

    phi_r, activo_r, c0, cut_step = evolucionar_con_historia(
        phi0.copy(), activo0.copy(), H, pasos, rng
    )
    P_real = base.persistencia(phi_r, c0)

    phi_nc = null_clasico(phi_r, seed)
    P_null_clasico = base.persistencia(phi_nc, c0)

    phi_ns, activo_ns, cut_step_shuf = null_secuencia(phi0, N, pasos, cut_step, seed)
    P_null_seq = base.persistencia(phi_ns, c0)

    grafo_final_identico = bool(np.array_equal(activo_r, activo_ns))
    n_cortadas = int((cut_step >= 0).sum())

    return {
        "P_real": P_real,
        "P_null_clasico": P_null_clasico,
        "P_null_seq": P_null_seq,
        "grafo_final_identico": grafo_final_identico,
        "n_cortadas": n_cortadas,
    }


def barrido_f26(N, eps_list, r_targets, semillas_ids, pasos_fijo):
    filas = []
    meta_por_eps = []
    for eps in eps_list:
        D = float(np.mean([base.medir_D(N, eps, s) for s in semillas_ids]))
        meta_por_eps.append({"eps": eps, "D": D, "pasos": pasos_fijo})

        for r_tgt in r_targets:
            if D > 0:
                H = float(min(r_tgt * D, 1.0))
                r_eff = H / D
            else:
                H = 0.0 if r_tgt == 0 else 1.0
                r_eff = 0.0 if r_tgt == 0 else float("inf")

            Preal, Pnc, Pns, ok_flags, ncorts = [], [], [], [], []
            for s in semillas_ids:
                res = corrida_f26(N, eps, H, pasos_fijo, seed=s)
                Preal.append(res["P_real"])
                Pnc.append(res["P_null_clasico"])
                Pns.append(res["P_null_seq"])
                ok_flags.append(res["grafo_final_identico"])
                ncorts.append(res["n_cortadas"])

            z_c = z_score(Preal, Pnc)
            z_s = z_score(Preal, Pns)
            filas.append(
                {
                    "eps": eps,
                    "r_target": r_tgt,
                    "H": H,
                    "D": D,
                    "r": r_eff,
                    "pasos": pasos_fijo,
                    "n_semillas": len(semillas_ids),
                    "P_real_mean": float(np.mean(Preal)),
                    "P_real_std": float(np.std(Preal)),
                    "P_null_clasico_mean": float(np.mean(Pnc)),
                    "P_null_clasico_std": float(np.std(Pnc)),
                    "P_null_seq_mean": float(np.mean(Pns)),
                    "P_null_seq_std": float(np.std(Pns)),
                    "z_clasico": round(z_c, 4),
                    "z_seq": round(z_s, 4),
                    "gana_clasico": bool(z_c >= Z_PASS),
                    "gana_seq": bool(z_s >= Z_PASS),
                    "robusto": bool(z_c >= Z_PASS and z_s >= Z_PASS),
                    "fragil": bool((z_c >= Z_PASS) != (z_s >= Z_PASS)),
                    "grafo_final_identico_todas": all(ok_flags),
                    "n_cortadas_medio": float(np.mean(ncorts)),
                    "P_real_por_semilla": [float(v) for v in Preal],
                    "P_null_clasico_por_semilla": [float(v) for v in Pnc],
                    "P_null_seq_por_semilla": [float(v) for v in Pns],
                }
            )
    return filas, meta_por_eps


def evaluar_pass(filas):
    """Criterio de PASS pre-registrado (§8): banda r>=10, eps>1e-6."""
    banda = [
        f for f in filas
        if f["r_target"] >= BANDA_CONGELAMIENTO_R and f["eps"] > EPS_MIN_BANDA
    ]
    if not banda:
        return {"n_puntos_banda": 0, "veredicto": "SIN_PUNTOS_EN_BANDA"}
    frac_gana_clasico = float(np.mean([f["gana_clasico"] for f in banda]))
    frac_gana_seq = float(np.mean([f["gana_seq"] for f in banda]))
    frac_robusto = float(np.mean([f["robusto"] for f in banda]))
    frac_fragil = float(np.mean([f["fragil"] for f in banda]))
    pass_clasico = frac_gana_clasico >= 0.5
    pass_seq = frac_gana_seq >= 0.5
    veredicto = (
        "ROBUSTO_REAL_GANA_AMBOS_NULOS" if (pass_clasico and pass_seq)
        else "FRAGIL_GANA_SOLO_UN_NULO" if (pass_clasico or pass_seq)
        else "NEGATIVO_NO_GANA_NINGUN_NULO"
    )
    return {
        "n_puntos_banda": len(banda),
        "frac_puntos_gana_null_clasico": round(frac_gana_clasico, 4),
        "frac_puntos_gana_null_seq": round(frac_gana_seq, 4),
        "frac_puntos_robustos": round(frac_robusto, 4),
        "frac_puntos_fragiles": round(frac_fragil, 4),
        "pass_null_clasico_ge_50pct": pass_clasico,
        "pass_null_seq_ge_50pct": pass_seq,
        "veredicto": veredicto,
    }


def evaluar_controles(filas):
    """Controles eps=0 (P debe ser 0 en las 3 ramas) y r=0 (nulos deben
    colapsar sobre REAL: sin cortes, nada que barajar)."""
    eps0 = [f for f in filas if f["eps"] == 0.0]
    ok_eps0 = all(
        f["P_real_mean"] == 0.0 and f["P_null_clasico_mean"] == 0.0 and f["P_null_seq_mean"] == 0.0
        for f in eps0
    ) if eps0 else None

    r0 = [f for f in filas if f["r_target"] == 0.0 and f["eps"] > 0]
    detalle_r0 = []
    for f in r0:
        detalle_r0.append(
            {
                "eps": f["eps"],
                "n_cortadas_medio": f["n_cortadas_medio"],
                "P_real_mean": f["P_real_mean"],
                "P_null_seq_mean": f["P_null_seq_mean"],
                "diff_real_vs_seq": round(f["P_real_mean"] - f["P_null_seq_mean"], 8),
            }
        )
    ok_r0 = all(d["n_cortadas_medio"] == 0.0 for d in detalle_r0) if detalle_r0 else None

    return {
        "control_eps0_ok": ok_eps0,
        "control_r0_null_seq_colapsa_ok": ok_r0,
        "detalle_r0": detalle_r0,
    }


def run(modo, log_lines):
    def log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, file=sys.stderr, flush=True)
        log_lines.append(line)

    t0 = time.time()

    if modo == "smoke":
        N = 60
        eps_list = [0.0, 1e-3, 0.5]
        semillas_ids = list(range(1000, 1004))  # 4
        r_targets = [0.0, 0.5, 1.0, 10.0]
    elif modo == "produccion":
        N = 200
        eps_list = [0.0, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.5, 1.0]
        semillas_ids = list(range(1000, 1016))  # 16
        r_targets = list(base.R_TARGETS)
    else:
        raise SystemExit(f"modo desconocido: {modo} (usa smoke|produccion)")

    log(f"modo={modo} N={N} eps_list={eps_list} r_targets={r_targets} n_semillas={len(semillas_ids)}")

    log(f"calibrando pasos de lavado (N={N}, eps=1e-3, {len(semillas_ids)} semillas)...")
    cal_ref = base.medir_pasos_lavado(N, 1e-3, max(len(semillas_ids), 4))
    pasos_fijo = cal_ref["pasos"]
    log(f"pasos_fijo={pasos_fijo} mediana_lavado={cal_ref['mediana']} lavo_todas={cal_ref['lavo_todas']}")

    log("iniciando barrido F2-6 (REAL con historia + NULL-clasico + NULL-secuencia)...")
    filas, meta = barrido_f26(N, eps_list, r_targets, semillas_ids, pasos_fijo)
    log(f"barrido completo: {len(filas)} puntos (eps x r)")

    pass_eval = evaluar_pass(filas)
    controles = evaluar_controles(filas)
    log(f"evaluacion PASS (banda r>=10, eps>1e-6): {pass_eval}")
    log(f"controles: eps0_ok={controles['control_eps0_ok']} r0_null_seq_colapsa_ok={controles['control_r0_null_seq_colapsa_ok']}")

    grafo_ok_global = all(f["grafo_final_identico_todas"] for f in filas)
    log(f"auto-chequeo grafo_final_identico en TODAS las combinaciones: {grafo_ok_global}")

    elapsed = time.time() - t0
    log(f"elapsed={elapsed:.1f}s")

    result = {
        "experimento": "F2-6",
        "titulo": "NULL alternativo: barajar la historia de cortes, no el acople final",
        "modo": modo,
        "N": N,
        "eps_list": eps_list,
        "r_targets": r_targets,
        "semillas_ids": semillas_ids,
        "n_semillas": len(semillas_ids),
        "pasos_fijo": pasos_fijo,
        "calibracion_ref": cal_ref,
        "meta_por_eps": meta,
        "grafo_final_identico_global": grafo_ok_global,
        "evaluacion_pass": pass_eval,
        "controles": controles,
        "filas": filas,
        "elapsed_s": elapsed,
        "timestamp_fin": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pre_registrado": {
            "criterio_pass": "z_clasico>=3 y z_seq>=3 simultaneamente en >=50% de los puntos de la banda r>=10, eps>1e-6",
            "veredicto_fragil_si": "gana a un NULL y no al otro en el mismo punto",
            "banda_congelamiento_r": BANDA_CONGELAMIENTO_R,
            "eps_min_banda": EPS_MIN_BANDA,
            "z_pass": Z_PASS,
        },
    }

    out_json = OUT / f"F2_6_{modo}_resultado.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"escrito: {out_json}")
    return result, out_json


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else "produccion"
    log_lines = []
    result, out_json = run(modo, log_lines)
    log_path = OUT / f"F2_6_log_ejecucion_{modo}.txt"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(json.dumps(result["evaluacion_pass"], ensure_ascii=False))
    print(f"\n[archivo] {out_json}", file=sys.stderr)
    print(f"[log] {log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
