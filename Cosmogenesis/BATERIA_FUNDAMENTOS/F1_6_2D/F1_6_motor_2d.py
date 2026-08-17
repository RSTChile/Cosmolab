#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1-6 — Persistencia en 2D: ¿es un artefacto del anillo 1D?
============================================================

Ejecutor: CC. Diseño: CS (BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md, sección F1-6).
Pre-registro: PROTOCOLO_F1-6_PREREGISTRO.md (leer antes que este archivo — este
motor implementa exactamente lo pre-registrado ahí, nada más).

Qué es esto: la MISMA física y el MISMO observable de `cs074_rcruz.py` (persistencia
por autocorrelación de forma × magnitud contra NULL barajado, con expansión que corta
aristas y compite con difusión que las repara) pero en una malla L×L toroidal en vez
de un anillo 1D. Adaptación de estilo `np.roll` 2D tomada como referencia de
`suite_epocas_masa_v6_mass_linaje.py` (SOLO el patrón de aristas horizontales/
verticales con np.roll en 2 ejes — no se copia física de masa/átomos de ese archivo).

NO se edita `cs074_rcruz.py` ni el archivo de referencia. Este es un archivo nuevo,
prefijo F1_6_, en su propia carpeta F1_6_2D/.

Observable (idéntico en espíritu a CS074, generalizado isotrópicamente a 2D):
  P = c_isotropo * v
  c_isotropo = max(0, 0.5*(corr(phi, roll(phi,1,eje=1)) + corr(phi, roll(phi,1,eje=0))))
  v = var(phi_final) / var(phi_inicial)

NULL: permutación 2D de phi al final (aplanar -> permutar -> reformar), igual criterio
que CS074 (destruye forma, conserva histograma).

Expansión: cada arista (horizontal ar, vertical ad) viva se corta Bernoulli(H) por
paso, independiente — misma corrección de CS074 frente al round(H*N) roto (válido
también cuando H*L^2 << 1).

D, pasos_lavado: MEDIDOS del propio campo, no impuestos (T1). r = H/D barre el grid
pre-registrado que cruza r=1 (idéntico a cs074_rcruz.R_TARGETS).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent

# Umbral de lavado para calibrar pasos (observable del propio campo) — igual CS074
P_LAVADO = 0.05
MARGEN_LAVADO = 1.15
# r pre-registrados que cruzan 1 (eje primario) — idéntico a cs074_rcruz.R_TARGETS
R_TARGETS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]
# eps pre-registrado (mismo grid que cs074_rcruz modo=produccion, comparabilidad 1D<->2D)
EPS_LIST = [0.0, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.5, 1.0]


# ----------------------------------------------------------------------------
# Sustrato 2D toroidal
# ----------------------------------------------------------------------------

def campo_inicial_2d(L, eps, rng):
    """
    Fondo uniforme + perturbación multi-modo 2D (5 modos, numeros de onda enteros
    aleatorios (kx,ky), fase aleatoria), normalizada a std=1 antes de escalar por eps.
    Generalizacion directa de campo_inicial() 1D de cs074_rcruz.py.
    """
    x = np.linspace(0.0, 1.0, L, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    fondo = np.ones((L, L), dtype=float)
    if eps <= 0.0:
        return fondo
    pert = np.zeros((L, L), dtype=float)
    for m in range(1, 6):
        kx = int(rng.integers(1, 4))
        ky = int(rng.integers(1, 4))
        fase = rng.uniform(0, 2 * np.pi)
        pert += np.sin(2 * np.pi * (kx * X + ky * Y) + fase) / m
    pert -= pert.mean()
    if pert.std() > 0:
        pert = pert / pert.std()
    return fondo + eps * pert


def paso_difusion_2d(phi, ar, ad):
    """
    Difusion solo por aristas vivas, malla 4-conexa toroidal.
    ar[i,j] = arista horizontal entre (i,j) y (i,j+1 mod L)
    ad[i,j] = arista vertical entre (i,j) y (i+1 mod L, j)
    Vectorizada; generaliza paso_difusion() 1D de cs074_rcruz.py a 2 ejes.
    """
    right_active = ar
    left_active = np.roll(ar, 1, axis=1)
    down_active = ad
    up_active = np.roll(ad, 1, axis=0)

    right_val = np.roll(phi, -1, axis=1)
    left_val = np.roll(phi, 1, axis=1)
    down_val = np.roll(phi, -1, axis=0)
    up_val = np.roll(phi, 1, axis=0)

    cnt = (
        right_active.astype(np.float64)
        + left_active.astype(np.float64)
        + down_active.astype(np.float64)
        + up_active.astype(np.float64)
    )
    s = (
        np.where(right_active, right_val, 0.0)
        + np.where(left_active, left_val, 0.0)
        + np.where(down_active, down_val, 0.0)
        + np.where(up_active, up_val, 0.0)
    )
    media = np.divide(s, cnt, out=phi.copy(), where=cnt > 0)
    nuevo = phi + 0.5 * (media - phi)
    return np.where(cnt > 0, nuevo, phi)


def paso_expansion_2d(ar, ad, H, rng):
    """
    Expansion = cortar aristas vivas (horizontales y verticales), Bernoulli(H)
    independiente por arista. Igual correccion que cs074_rcruz.paso_expansion:
    esperanza de fraccion cortada/paso = H, valido tambien para H*L^2 << 1.
    """
    if H <= 0.0:
        return ar, ad
    ar = ar.copy()
    ad = ad.copy()
    if H >= 1.0:
        ar[:] = False
        ad[:] = False
        return ar, ad
    u_ar = rng.random(ar.shape)
    u_ad = rng.random(ad.shape)
    ar[ar & (u_ar < H)] = False
    ad[ad & (u_ad < H)] = False
    return ar, ad


def evolucionar_2d(phi, ar, ad, H, pasos, rng, null=False):
    contraste0 = float(phi.std())
    for _ in range(pasos):
        phi = paso_difusion_2d(phi, ar, ad)
        ar, ad = paso_expansion_2d(ar, ad, H, rng)
    if null:
        flat = phi.reshape(-1)
        flat = rng.permutation(flat)
        phi = flat.reshape(phi.shape)
    return phi, ar, ad, contraste0


def medir_D_2d(L, eps, seed):
    """Fraccion de contraste borrada en UN paso de difusion pura (H=0)."""
    rng = np.random.default_rng(seed)
    phi = campo_inicial_2d(L, eps, rng)
    ar = np.ones((L, L), dtype=bool)
    ad = np.ones((L, L), dtype=bool)
    c0 = phi.std()
    if c0 <= 0:
        return 0.0
    phi1 = paso_difusion_2d(phi, ar, ad)
    c1 = phi1.std()
    return max(0.0, float((c0 - c1) / c0))


def persistencia_2d(phi, contraste0):
    """
    P = c_isotropo * v.
    c_isotropo = max(0, promedio de la autocorrelacion a primer vecino en los
    2 ejes) -- generalizacion isotropica de la autocorrelacion 1D de CS074.
    v = var(phi_final)/contraste0^2.
    """
    if contraste0 <= 0 or phi.std() <= 1e-12:
        return 0.0
    flat = phi.reshape(-1)
    flat_h = np.roll(phi, 1, axis=1).reshape(-1)
    flat_v = np.roll(phi, 1, axis=0).reshape(-1)
    c_h = np.corrcoef(flat, flat_h)[0, 1]
    c_v = np.corrcoef(flat, flat_v)[0, 1]
    if not np.isfinite(c_h):
        c_h = 0.0
    if not np.isfinite(c_v):
        c_v = 0.0
    c = max(0.0, float(0.5 * (c_h + c_v)))
    v = float(phi.var() / (contraste0 ** 2))
    return float(c * v)


def medir_pasos_lavado_2d(L, eps, semillas, P_thr=P_LAVADO, max_steps=20000, check_every=25):
    """
    Tiempo medido (pasos) a H=0 para que P < P_thr. Sale del propio campo (T1).
    max_steps mas chico que en 1D (20000 vs 200000 de CS074) porque en 2D cada
    celda tiene 4 vecinos (vs 2 en 1D) -> difusion decorrela mas rapido; se valida
    empiricamente con 'lavo_todas' en el JSON de salida.
    """
    tiempos = []
    for s in range(semillas):
        rng = np.random.default_rng(20_000 + s)
        phi = campo_inicial_2d(L, eps, rng)
        ar = np.ones((L, L), dtype=bool)
        ad = np.ones((L, L), dtype=bool)
        c0 = float(phi.std())
        if c0 <= 0:
            tiempos.append(0)
            continue
        t_hit = None
        for t in range(1, max_steps + 1):
            phi = paso_difusion_2d(phi, ar, ad)
            if t % check_every == 0:
                if persistencia_2d(phi, c0) < P_thr:
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


def corrida_2d(L, eps, H, pasos, seed, null=False):
    rng = np.random.default_rng(seed)
    phi = campo_inicial_2d(L, eps, rng)
    ar = np.ones((L, L), dtype=bool)
    ad = np.ones((L, L), dtype=bool)
    phi, ar, ad, c0 = evolucionar_2d(phi, ar, ad, H, pasos, rng, null=null)
    P = persistencia_2d(phi, c0)
    n_edges = 2 * L * L
    frac_exp = 1.0 - float((ar.sum() + ad.sum()) / n_edges)
    return {
        "P": P,
        "frac_exp": frac_exp,
        "std_ratio": float(phi.std() / c0) if c0 > 0 else 0.0,
    }


def barrido_rcruz_2d(L, eps_list, r_targets, semillas, pasos_fijo=None):
    filas = []
    meta_por_eps = []
    for eps in eps_list:
        D = float(np.mean([medir_D_2d(L, eps, s) for s in range(semillas)]))
        if eps <= 0:
            cal = {"tiempos": [], "mediana": 0, "pasos": pasos_fijo or 50, "P_thr": P_LAVADO, "lavo_todas": True}
            pasos = pasos_fijo or 50
        else:
            if pasos_fijo is not None:
                cal = {"tiempos": [], "mediana": pasos_fijo, "pasos": pasos_fijo, "P_thr": P_LAVADO, "lavo_todas": True, "fijo": True}
                pasos = pasos_fijo
            else:
                cal = medir_pasos_lavado_2d(L, eps, max(semillas, 4))
                pasos = cal["pasos"]
        meta_por_eps.append({"eps": eps, "D": D, "calibracion_lavado": cal, "pasos": pasos})

        for r_tgt in r_targets:
            if D > 0:
                H = float(min(r_tgt * D, 1.0))
                r_eff = H / D if D > 0 else float("inf")
            else:
                H = 0.0 if r_tgt == 0 else 1.0
                r_eff = float("inf") if D <= 0 and r_tgt > 0 else 0.0

            Preal, Pnull, srr, srn, fracs = [], [], [], [], []
            for s in range(semillas):
                rr = corrida_2d(L, eps, H, pasos, seed=2000 + s, null=False)
                nn = corrida_2d(L, eps, H, pasos, seed=2000 + s, null=True)
                Preal.append(rr["P"])
                Pnull.append(nn["P"])
                srr.append(rr["std_ratio"])
                srn.append(nn["std_ratio"])
                fracs.append(rr["frac_exp"])
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
                }
            )
    return filas, meta_por_eps


def control_r0_ok(filas, P_max=0.15):
    rows = [f for f in filas if f["r_target"] == 0.0 and f["eps"] > 0]
    if not rows:
        return False, {}
    mean_P = float(np.mean([f["P_real"] for f in rows]))
    return mean_P < P_max, {"mean_P_r0_eps_gt0": mean_P, "n": len(rows), "P_max": P_max}


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else "smoke_L32"
    t0 = time.time()

    if modo == "smoke_L32":
        # Smoke chico: valida el motor antes de escalar. Grid reducido pero con
        # el mismo mecanismo/observable/NULL que produccion.
        L = 32
        semillas = 4
        eps_list = [0.0, 1e-3, 0.1, 1.0]
        r_targets = [0.0, 0.1, 1.0, 10.0, 100.0]
        cal_ref = medir_pasos_lavado_2d(L, 1e-3, semillas)
        pasos_fijo = cal_ref["pasos"]
    elif modo in ("prod_L32", "prod_L64", "prod_L128"):
        L = {"prod_L32": 32, "prod_L64": 64, "prod_L128": 128}[modo]
        semillas = 8
        eps_list = list(EPS_LIST)
        r_targets = list(R_TARGETS)
        cal_ref = medir_pasos_lavado_2d(L, 1e-3, semillas)
        pasos_fijo = cal_ref["pasos"]
    else:
        raise SystemExit(f"modo desconocido: {modo} (usa smoke_L32|prod_L32|prod_L64|prod_L128)")

    print(
        f"[calibracion] modo={modo} L={L} eps=1e-3 mediana_lavado={cal_ref['mediana']} "
        f"pasos={pasos_fijo} lavo_todas={cal_ref['lavo_todas']} tiempos={cal_ref['tiempos']}",
        file=sys.stderr, flush=True,
    )

    filas, meta = barrido_rcruz_2d(L, eps_list, r_targets, semillas, pasos_fijo=pasos_fijo)
    ok, ctrl = control_r0_ok(filas)

    def rnd(f):
        out = dict(f)
        for k in ("D", "H", "r", "P_real", "P_null", "P_real_std", "P_null_std", "std_ratio_real", "std_ratio_null"):
            if k in out and isinstance(out[k], float):
                out[k] = round(out[k], 6)
        out["z"] = round(out["z"], 3)
        return out

    result = {
        "experimento": "F1-6-2D",
        "modo": modo,
        "L": L,
        "semillas": semillas,
        "eps_list": eps_list,
        "r_targets": r_targets,
        "pasos_fijo": pasos_fijo,
        "calibracion_ref": cal_ref,
        "meta_por_eps": meta,
        "control_r0_lava": ok,
        "control_r0_detail": ctrl,
        "filas": [rnd(f) for f in filas],
        "elapsed_s": time.time() - t0,
        "pre_inscrito": {
            "r0": "P_real debe ser bajo (difusion lava)",
            "r_ll_1": "persistencia baja",
            "r_approx_1": "transicion si el mecanismo es real (comparar con r*~0.1 conocido de 1D)",
            "r_gg_1": "persistencia alta vs NULL",
            "eps0": "P=0 a todo r",
        },
    }

    out_json = OUT / f"F1_6_resultado_{modo}.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "filas"}, ensure_ascii=False))
    print(f"\n[archivo] {out_json}", file=sys.stderr)
    print(f"[control_r0_lava] {ok} {ctrl}", file=sys.stderr)
    print(f"[elapsed] {result['elapsed_s']:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
