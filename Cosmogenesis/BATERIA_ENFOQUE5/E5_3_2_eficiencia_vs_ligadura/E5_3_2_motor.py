#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5.3-2 — Eficiencia vs intensidad de ligadura, rango nula-a-total
===================================================================

Motor propio del agente E5.3-2 (batería Enfoque 5, 30 experimentos en paralelo).
Pre-registro congelado ANTES de este archivo:
  PROTOCOLO_E5.3-2_PREREGISTRO.md (mismo directorio)

Reutiliza SIN EDITAR las funciones físicas de
  /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py
  (campo_inicial, paso_difusion, paso_expansion, medir_D, medir_pasos_lavado)

Novedad de este experimento (no existe en el código base):
  intensidad_ligadura L modula hacia abajo la tasa de corte de aristas:
      H_eff(L) = H0 / (1+L),   H0 = D medido (ancla r=1 identificada por cs074_rcruz)
  L barre [1e-3 .. 1e2] (5 décadas): L chico = ligadura nula (se rompe al ritmo natural
  H0); L grande = ligadura casi total (apenas se rompe nada).

Observable (SALIDA, nunca ajustada): eficiencia = E_ligada / E_total, donde
  E_total  = Σ(φ0 - mean(φ0))^2               medido UNA vez al inicio, fijo
  segmentos = tramos contiguos del anillo separados por aristas cortadas (activo final)
  E_ligada = Σ_k n_k (μ_k - mean_global(φ))^2  (varianza ENTRE segmentos, ANOVA)
  E_dentro = Σ_k Σ_{i∈k} (φ_i - μ_k)^2          (varianza DENTRO de cada segmento)
  Identidad auditada cada corrida: E_ligada + E_dentro == E_final (tolerancia 1e-9)

NULL: se permutan los VALORES de φ manteniendo la MISMA partición (activo real) —
aísla si la eficiencia depende de la estructura espacial real o de solo los tamaños
de segmento.

Ningún valor de 4.9%/31.5% entra en este archivo en ningún punto del cómputo.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE_CODE = HERE.parent.parent / "cs074_rcruz.py"

# --- Importar funciones del código base SIN editarlo ---
spec = importlib.util.spec_from_file_location("cs074_rcruz", str(BASE_CODE))
cs074 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs074)  # type: ignore

campo_inicial = cs074.campo_inicial
paso_difusion = cs074.paso_difusion
paso_expansion = cs074.paso_expansion
medir_D = cs074.medir_D
medir_pasos_lavado = cs074.medir_pasos_lavado

# ---------------- Constantes declaradas ANTES de correr (T1) ----------------
N = 200                       # igual que modo "produccion" de la base
SEMILLAS = 12                 # mínimo exigido por el pre-registro
NOISE_REL = 0.02              # idéntica a E5.1-1 (consistencia entre agentes de la batería)
EPS_LIST = [0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.3, 1.0]  # idéntica a E5.1-1
L_LIST = list(np.logspace(-3, 2, 10))  # 5 décadas exactas, 10 puntos
P_LAVADO_EPS_REF = 1e-3
ANOVA_TOL = 1e-6


def energia_total_inicial(phi0):
    m = phi0.mean()
    return float(np.sum((phi0 - m) ** 2))


def segmentos_desde_activo(activo):
    """
    Tramos contiguos del anillo separados por aristas cortadas.
    activo[i] = arista viva entre nodo i y nodo (i+1) mod N.
    Devuelve lista de arrays de índices (uno por segmento).
    Si todas las aristas están vivas -> un solo segmento = todo el anillo.
    """
    N_ = activo.size
    if activo.all():
        return [np.arange(N_)]
    cortes = np.where(~activo)[0]  # arista i--i+1 cortada -> límite de segmento tras i
    segmentos = []
    start = 0
    prev_cut = None
    cortes_sorted = np.sort(cortes)
    idx = 0
    # Reconstrucción lineal: recorremos desde el primer corte para evitar el wrap-around.
    if cortes_sorted.size == 0:
        return [np.arange(N_)]
    first_cut = cortes_sorted[0]
    order = np.concatenate([np.arange(first_cut + 1, N_), np.arange(0, first_cut + 1)])
    boundaries = []
    for c in cortes_sorted:
        boundaries.append(c)
    # recorrer 'order' (empieza justo despues del primer corte, wrap, termina en first_cut)
    seg = []
    seg_start_pos = 0
    cortes_set = set(cortes_sorted.tolist())
    cur = []
    for node in order:
        cur.append(node)
        # si la arista node->node+1 (mod N) esta cortada, cierra segmento
        if node in cortes_set:
            segmentos.append(np.array(cur, dtype=int))
            cur = []
    if cur:
        segmentos.append(np.array(cur, dtype=int))
    return segmentos


def descomponer_energia(phi, segmentos):
    mean_global = phi.mean()
    e_ligada = 0.0
    e_dentro = 0.0
    for seg in segmentos:
        vals = phi[seg]
        mu_k = vals.mean()
        n_k = vals.size
        e_ligada += n_k * (mu_k - mean_global) ** 2
        e_dentro += float(np.sum((vals - mu_k) ** 2))
    e_final = float(np.sum((phi - mean_global) ** 2))
    return float(e_ligada), float(e_dentro), e_final


def evolucionar_ligadura(phi0, H_eff, pasos, rng, eps):
    """
    Difusión (paso_difusion, sin editar) + expansión con H_eff (paso_expansion, sin
    editar) + ruido dinámico (T7) NOISE_REL*eps por paso, sumado ANTES de difundir.
    Devuelve phi_final, activo_final.
    """
    phi = phi0.copy()
    activo = np.ones(phi.size, dtype=bool)
    amp_ruido = NOISE_REL * eps
    for _ in range(pasos):
        if amp_ruido > 0:
            phi = phi + rng.normal(0.0, amp_ruido, size=phi.shape)
        phi = paso_difusion(phi, activo)
        activo = paso_expansion(activo, H_eff, rng)
    return phi, activo


def corrida_una(N_, eps, L, H0, pasos, seed):
    rng = np.random.default_rng(seed)
    phi0, _ = campo_inicial(N_, eps, rng)
    E_total = energia_total_inicial(phi0)

    H_eff = H0 / (1.0 + L)
    phi_f, activo_f = evolucionar_ligadura(phi0, H_eff, pasos, rng, eps)
    segmentos = segmentos_desde_activo(activo_f)
    e_ligada, e_dentro, e_final = descomponer_energia(phi_f, segmentos)

    anova_ok = abs((e_ligada + e_dentro) - e_final) <= max(ANOVA_TOL, 1e-9 * max(e_final, 1.0))
    eficiencia_real = 0.0 if E_total <= 0 else e_ligada / E_total

    # NULL: permutar VALORES de phi_f manteniendo la MISMA partición (segmentos reales)
    phi_null = rng.permutation(phi_f)
    e_ligada_n, e_dentro_n, e_final_n = descomponer_energia(phi_null, segmentos)
    eficiencia_null = 0.0 if E_total <= 0 else e_ligada_n / E_total

    return {
        "E_total": E_total,
        "H_eff": H_eff,
        "n_segmentos": len(segmentos),
        "e_ligada": e_ligada,
        "e_dentro": e_dentro,
        "e_final": e_final,
        "anova_ok": bool(anova_ok),
        "eficiencia_real": eficiencia_real,
        "eficiencia_null": eficiencia_null,
        "E_final_sobre_E_total": (e_final / E_total) if E_total > 0 else 0.0,
    }


def barrido():
    t0 = time.time()
    # Calibración de pasos: igual método que la base (lavado a P<0.05, eps=1e-3, H=0)
    cal = medir_pasos_lavado(N, P_LAVADO_EPS_REF, SEMILLAS)
    pasos = cal["pasos"]
    print(
        f"[calibracion] N={N} eps={P_LAVADO_EPS_REF} mediana_lavado={cal['mediana']} "
        f"pasos={pasos} lavo_todas={cal['lavo_todas']} tiempos={cal['tiempos']}",
        file=sys.stderr, flush=True,
    )

    filas = []
    meta_D = {}
    for eps in EPS_LIST:
        Ds = [medir_D(N, eps, s) for s in range(SEMILLAS)]
        D = float(np.mean(Ds))
        meta_D[eps] = {"D_medio": D, "D_por_semilla": Ds}
        H0 = D  # ancla r=1: la propia base identifica r=H/D≈1 como transición natural
        for L in L_LIST:
            rows_real, rows_null, n_segs, anova_flags, Efrac = [], [], [], [], []
            for s in range(SEMILLAS):
                seed = 5000 + s
                r = corrida_una(N, eps, L, H0, pasos, seed)
                rows_real.append(r["eficiencia_real"])
                rows_null.append(r["eficiencia_null"])
                n_segs.append(r["n_segmentos"])
                anova_flags.append(r["anova_ok"])
                Efrac.append(r["E_final_sobre_E_total"])
            rr = np.array(rows_real)
            nn = np.array(rows_null)
            sd = np.sqrt((rr.var() + nn.var()) / 2.0)
            sd = max(sd, 1e-9)
            z = float((rr.mean() - nn.mean()) / sd)
            filas.append({
                "eps": eps,
                "L": float(L),
                "H0": H0,
                "D": D,
                "pasos": pasos,
                "eficiencia_real_media": float(rr.mean()),
                "eficiencia_real_std": float(rr.std()),
                "eficiencia_real_por_semilla": rows_real,
                "eficiencia_null_media": float(nn.mean()),
                "eficiencia_null_std": float(nn.std()),
                "eficiencia_null_por_semilla": rows_null,
                "z": z,
                "n_segmentos_medio": float(np.mean(n_segs)),
                "anova_ok_todas": bool(all(anova_flags)),
                "E_final_sobre_E_total_medio": float(np.mean(Efrac)),
            })
        print(f"[eps={eps:g}] D={D:.6g} H0={H0:.6g}  ({len(L_LIST)} L-puntos x {SEMILLAS} semillas listos)",
              file=sys.stderr, flush=True)

    elapsed = time.time() - t0
    resultado = {
        "experimento": "E5_3_2_eficiencia_vs_ligadura",
        "protocolo": "PROTOCOLO_E5.3-2_PREREGISTRO.md",
        "N": N,
        "semillas": SEMILLAS,
        "eps_list": EPS_LIST,
        "L_list": [float(x) for x in L_LIST],
        "noise_rel": NOISE_REL,
        "pasos": pasos,
        "calibracion_lavado": cal,
        "meta_D_por_eps": meta_D,
        "filas": filas,
        "elapsed_s": elapsed,
        "definicion_eficiencia": (
            "eficiencia = E_ligada/E_total; E_ligada = varianza ENTRE segmentos "
            "(ANOVA, segmentos = tramos vivos tras cortes de expansion); E_total = "
            "energia de desviacion inicial Sum((phi0-mean)^2), fija por (eps,semilla)."
        ),
        "definicion_ligadura": "H_eff(L) = H0/(1+L); H0 = D medido (ancla r=1 de cs074_rcruz.py)",
        "advertencia_definicion_propia": (
            "E5_3_1_eficiencia_12decadas/ estaba VACIA (sin protocolo) en el momento de "
            "este pre-registro y motor -> esta es una definicion PROPIA, no heredada."
        ),
    }
    return resultado


def main():
    resultado = barrido()
    out_json = HERE / "E5_3_2_resultado_crudo.json"
    out_json.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[archivo] {out_json}", file=sys.stderr)
    print(f"[elapsed] {resultado['elapsed_s']:.1f}s", file=sys.stderr)
    # resumen corto a stdout
    for f in resultado["filas"]:
        print(
            f"eps={f['eps']:.3g} L={f['L']:.4g} eff_real={f['eficiencia_real_media']:.4f}"
            f"(±{f['eficiencia_real_std']:.4f}) eff_null={f['eficiencia_null_media']:.4f} "
            f"z={f['z']:.2f} n_seg={f['n_segmentos_medio']:.1f} anova_ok={f['anova_ok_todas']}"
        )


if __name__ == "__main__":
    main()
