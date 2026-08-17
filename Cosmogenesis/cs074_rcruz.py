#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS074-rcruz — Persistencia en campo continuo con CRUCE REAL de r≈1
==================================================================

Sucesor de cs074_persistencia_campo.py (producción original NO se modifica).

Diagnóstico del run original (cs074_produccion_resultado_crudo.json):
  - D medido ~ 5e-5 (N=800) → r = H/D ≥ 0.05/D ~ 1000 para todo H>0
  - pasos=120 << 1/D → a H=0 la difusión NO lava (P~0.99) → control roto
  - El barrido nunca cruzó r≈1: todo el grid vivía en r≫1

Este script reabre el experimento en el régimen que la pre-inscripción pedía:

  1) Mide D del propio campo (H=0, un paso) — igual que CS074.
  2) Calibra pasos_lavado: tiempo (en pasos) para que a H=0, P caiga < P_LAVADO
     (medido, no impuesto como tasa). Así el control r=0 es falsable.
  3) Eje primario = r pre-registrado que CRUZA 1:
        r ∈ {0, 0.1, 0.3, 0.5, 1, 2, 5, 10, 30, 100}
     H(r) = min(r · D, 1.0).  D medido → H emerge; no se elige H a mano.
  4) Misma física CS074: campo continuo + mancha ε; difusión SOLO por aristas
     vivas; expansión corta aristas; NULL = permuta phi al final; P = forma×magnitud.
  5) Difusión vectorizada (numpy) numéricamente idéntica al loop original.
  6) Expansión: Bernoulli por arista (P_corte=H). El original usaba round(H·N),
     que con H·N≪1 da 0 cortes → en r-cruz la expansión no actuaba. Corregido
     para que la esperanza de fracción cortada/paso sea H también a H pequeño.

Lectura pre-inscrita (antes de correr):
  - eps=0 → P=0 a todo r
  - r=0 (H=0) → P_real ≈ 0 (difusión lava); si no, el lavado falló → reportar y no
    interpretar el resto como cruce
  - r≪1 → persistencia baja (se reabsorbe)
  - r≈1 → transición (si el mecanismo es real)
  - r≫1 → persistencia alta vs NULL (congelado por aislamiento)
  - Nulo = hallazgo si nunca se separa de NULL

Anti-Shannon: sin target η/7:1/GeV; T(K) y t(s) solo mapeo/reporte; D y pasos_lavado
MEDIDOS; r es razón interna.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent

# Escalas de reporte (NO dinámica) — mismas anclas que CS074
T_SING = 1e20
T_FIN = 1e10
t_INI, t_FIN = 1e-20, 1e-4

# Umbral de lavado para calibrar pasos (observable del propio campo)
P_LAVADO = 0.05
# Margen sobre el tiempo medido de lavado (para no cortar en el borde)
MARGEN_LAVADO = 1.15
# r pre-registrados que cruzan 1 (eje primario)
R_TARGETS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]


def reloj_fisico(paso, pasos):
    a, b = np.log10(t_INI), np.log10(t_FIN)
    return 10.0 ** (a + (b - a) * (paso / max(pasos, 1)))


def temperatura_fisica(frac_expandida):
    a, b = np.log10(T_SING), np.log10(T_FIN)
    return 10.0 ** (a + (b - a) * min(max(frac_expandida, 0.0), 1.0))


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


def paso_difusion(phi, activo):
    """
    Difusión solo por aristas vivas. Vectorizada; idéntica al loop de CS074
    (verificado: max|Δ|=0 frente a la implementación por índices).
    activo[i] = arista entre i y (i+1) mod N.
    """
    left = np.roll(phi, 1)
    right = np.roll(phi, -1)
    e_left = np.roll(activo, 1)  # arista (i-1)--i
    e_right = activo             # arista i--(i+1)
    n_nb = e_left.astype(np.float64) + e_right.astype(np.float64)
    s = e_left * left + e_right * right
    media = np.divide(s, n_nb, out=phi.copy(), where=n_nb > 0)
    nuevo = phi + 0.5 * (media - phi)
    return np.where(n_nb > 0, nuevo, phi)


def paso_expansion(activo, H, rng):
    """
    Expansión = cortar aristas vivas.

    CS074 original usaba ncorte = round(H * n_vivas). Con H·N ≪ 1 eso redondea a 0
    y la expansión NUNCA actúa — fatal en el régimen r-cruz (H = r·D ~ 1e-4).
    Aquí: cada arista viva se corta con probabilidad H (Bernoulli). Esperanza de
    fracción cortada por paso = H, válido también para H·N < 1. H≥1 corta todas.
    """
    if H <= 0.0:
        return activo
    activo = activo.copy()
    if H >= 1.0:
        activo[:] = False
        return activo
    # Bernoulli independiente por arista viva
    u = rng.random(activo.shape)
    cortar = activo & (u < H)
    activo[cortar] = False
    return activo


def evolucionar(phi, activo, H, pasos, rng, null=False):
    contraste0 = float(phi.std())
    for _ in range(pasos):
        phi = paso_difusion(phi, activo)
        activo = paso_expansion(activo, H, rng)
    if null:
        phi = rng.permutation(phi)
    return phi, activo, contraste0


def medir_D(N, eps, seed):
    """Fracción de contraste borrada en UN paso de difusión pura (H=0)."""
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    c0 = phi.std()
    if c0 <= 0:
        return 0.0
    phi1 = paso_difusion(phi, activo)
    c1 = phi1.std()
    return max(0.0, float((c0 - c1) / c0))


def persistencia(phi, contraste0):
    if contraste0 <= 0 or phi.std() <= 1e-12:
        return 0.0
    c = np.corrcoef(phi, np.roll(phi, 1))[0, 1]
    if not np.isfinite(c):
        c = 0.0
    c = max(0.0, float(c))
    v = float(phi.var() / (contraste0 ** 2))
    return float(c * v)


def detectar_cuantizacion(phi, activo):
    N = phi.size
    media = phi.mean()
    signo = phi >= media
    hist = {}
    visto = np.zeros(N, dtype=bool)
    for start in range(N):
        if visto[start]:
            continue
        comp = [start]
        visto[start] = True
        pila = [start]
        while pila:
            u = pila.pop()
            for v in ((u - 1) % N, (u + 1) % N):
                arista_viva = activo[u] if v == (u + 1) % N else activo[(u - 1) % N]
                if (not visto[v]) and arista_viva and (signo[v] == signo[start]):
                    visto[v] = True
                    comp.append(v)
                    pila.append(v)
        k = len(comp)
        hist[k] = hist.get(k, 0) + 1
    return hist


def medir_pasos_lavado(N, eps, semillas, P_thr=P_LAVADO, max_steps=200000, check_every=50):
    """
    Tiempo medido (pasos) a H=0 para que P < P_thr.
    Sale del propio campo; no es un número puesto a mano.
    """
    tiempos = []
    for s in range(semillas):
        rng = np.random.default_rng(10_000 + s)
        phi, _ = campo_inicial(N, eps, rng)
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
            t_hit = max_steps  # no lavó a tiempo — se reporta
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


def corrida(N, eps, H, pasos, seed, null=False):
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
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


def barrido_rcruz(N, eps_list, r_targets, semillas, pasos_fijo=None):
    """
    Para cada eps:
      - mide D
      - calibra pasos (si pasos_fijo is None: por eps; si no, usa fijo)
      - barre r → H = min(r*D, 1)
    """
    filas = []
    meta_por_eps = []
    for eps in eps_list:
        D = float(np.mean([medir_D(N, eps, s) for s in range(semillas)]))
        if eps <= 0:
            # sin diferencia no hay lavado que medir; usa pasos del primer eps>0 o default
            cal = {"tiempos": [], "mediana": 0, "pasos": pasos_fijo or 100, "P_thr": P_LAVADO, "lavo_todas": True}
            pasos = pasos_fijo or 100
        else:
            if pasos_fijo is not None:
                cal = {"tiempos": [], "mediana": pasos_fijo, "pasos": pasos_fijo, "P_thr": P_LAVADO, "lavo_todas": True, "fijo": True}
                pasos = pasos_fijo
            else:
                cal = medir_pasos_lavado(N, eps, max(semillas, 4))
                pasos = cal["pasos"]
        meta_por_eps.append({"eps": eps, "D": D, "calibracion_lavado": cal, "pasos": pasos})

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
                rr = corrida(N, eps, H, pasos, seed=1000 + s, null=False)
                nn = corrida(N, eps, H, pasos, seed=1000 + s, null=True)
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
                    "T_ini_K": T_SING,
                    "cuantos_k": {int(k): int(v) for k, v in sorted(hist_real.items())},
                }
            )
    return filas, meta_por_eps


def control_r0_ok(filas, P_max=0.15):
    """¿A r=0 y eps>0 la difusión lava? (gate de validez del cruce)."""
    rows = [f for f in filas if f["r_target"] == 0.0 and f["eps"] > 0]
    if not rows:
        return False, {}
    mean_P = float(np.mean([f["P_real"] for f in rows]))
    return mean_P < P_max, {"mean_P_r0_eps_gt0": mean_P, "n": len(rows), "P_max": P_max}


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else "produccion"
    t0 = time.time()

    if modo == "produccion":
        # N=200: D mayor que N=800 → lavado en ~5k pasos (medido); aún campo continuo
        N = 200
        semillas = 8
        eps_list = [0.0, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.5, 1.0]
        cal_ref = medir_pasos_lavado(N, 1e-3, semillas)
        pasos_fijo = cal_ref["pasos"]
        print(
            f"[calibracion] N={N} eps=1e-3 mediana_lavado={cal_ref['mediana']} "
            f"pasos={pasos_fijo} lavo_todas={cal_ref['lavo_todas']} tiempos={cal_ref['tiempos']}",
            file=sys.stderr,
            flush=True,
        )
    elif modo == "robustez400":
        # Pedido CS: ¿el umbral en r se mueve con N? (análogo "R es de T, no de N")
        # Mismo protocolo que produccion; solo cambia N → D y pasos se re-miden.
        N = 400
        semillas = 8
        eps_list = [0.0, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.5, 1.0]
        cal_ref = medir_pasos_lavado(N, 1e-3, semillas)
        pasos_fijo = cal_ref["pasos"]
        print(
            f"[calibracion robustez400] N={N} eps=1e-3 mediana_lavado={cal_ref['mediana']} "
            f"pasos={pasos_fijo} lavo_todas={cal_ref['lavo_todas']} tiempos={cal_ref['tiempos']}",
            file=sys.stderr,
            flush=True,
        )
    elif modo == "chico":
        N = 100
        semillas = 4
        eps_list = [0.0, 1e-3, 0.1, 1.0]
        cal_ref = medir_pasos_lavado(N, 1e-3, semillas)
        pasos_fijo = cal_ref["pasos"]
        print(
            f"[calibracion chico] pasos={pasos_fijo} tiempos={cal_ref['tiempos']}",
            file=sys.stderr,
            flush=True,
        )
    else:
        raise SystemExit(
            f"modo desconocido: {modo} (usa produccion|robustez400|chico)"
        )

    r_targets = list(R_TARGETS)
    filas, meta = barrido_rcruz(N, eps_list, r_targets, semillas, pasos_fijo=pasos_fijo)
    ok, ctrl = control_r0_ok(filas)

    # redondeo legible para JSON
    def rnd(f):
        out = dict(f)
        for k in ("D", "H", "r", "P_real", "P_null", "P_real_std", "P_null_std", "std_ratio_real", "std_ratio_null"):
            if k in out and isinstance(out[k], float):
                out[k] = round(out[k], 6)
        out["z"] = round(out["z"], 3)
        return out

    result = {
        "experimento": "CS074-rcruz",
        "modo": modo,
        "N": N,
        "semillas": semillas,
        "r_targets": r_targets,
        "pasos_fijo": pasos_fijo,
        "calibracion_ref": cal_ref,
        "meta_por_eps": meta,
        "control_r0_lava": ok,
        "control_r0_detail": ctrl,
        "filas": [rnd(f) for f in filas],
        "elapsed_s": time.time() - t0,
        "pre_inscrito": {
            "r0": "P_real debe ser bajo (difusión lava)",
            "r_ll_1": "persistencia baja",
            "r_approx_1": "transición si el mecanismo es real",
            "r_gg_1": "persistencia alta vs NULL",
            "eps0": "P=0 a todo r",
        },
    }

    out_json = OUT / f"cs074_rcruz_{modo}_resultado.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    print(f"\n[archivo] {out_json}", file=sys.stderr)
    print(f"[control_r0_lava] {ok} {ctrl}", file=sys.stderr)
    print(f"[elapsed] {result['elapsed_s']:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
