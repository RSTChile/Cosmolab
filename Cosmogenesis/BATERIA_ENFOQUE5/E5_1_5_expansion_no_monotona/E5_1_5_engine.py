#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5_1_5_engine.py — Persistencia de exergía bajo expansión no monótona (historias H(t))
========================================================================================

Implementa el protocolo pre-registrado en PROTOCOLO_E5.1-5_PREREGISTRO.md (léelo primero;
este motor NO se corre sin ese archivo fechado y congelado).

Pregunta (ficha E5.1-5, TEMA 1 de BATERIA_ENFOQUE5): si la expansión acelera y frena
(H(t) no constante, distintas historias), ¿la exergía (capacidad de hacer trabajo,
medida como desviación del equilibrio uniforme) aguanta? ¿El resultado depende del
r_efectivo INTEGRADO, o del perfil H(t) específico?

Reusa la física de cs074_rcruz.py (CS074-rcruz) SIN EDITARLO: se importa el módulo por
ruta y se llaman sus funciones (campo_inicial, paso_difusion, paso_expansion,
persistencia, medir_D, medir_pasos_lavado). El mecanismo de perfiles H(t) extiende el
método de F2_5_engine.py (Enfoque 2, no editado, solo leído como referencia de diseño) a
6 perfiles y a un observable nuevo: X_final (exergía), no P.

Definiciones (ver protocolo §2-3):
  - E_total(t) := sum(phi_t) — candidato a "energía total" (axioma E1, declarado). Se
    AUDITA (no se asume) su conservación en cada corrida: E_drift_rel_max muestreado en
    5 puntos de la trayectoria. Hallazgo lateral pre-registrado: la física reusada NO
    conserva sum(phi) exactamente en fronteras de segmentos aislados (deriva pequeña,
    ~1e-4..1e-6 relativo en las pruebas previas) — se reporta, no se esconde (T1), y no
    se edita cs074_rcruz.py para "arreglarlo" (eso es alcance de TEMA 2).
  - X_final := std(phi_final) / std(phi_inicial)  (= std_ratio de CS074-rcruz). Fracción
    de la exergía (dispersión respecto del equilibrio uniforme) que sobrevive.
  - P (persistencia, secundario/cross-check) := idéntico a CS074-rcruz.
  - r_efectivo_realizado: derivado de frac_exp, misma fórmula que F2_5_engine.py.

NULL: barajado del campo al final (idéntico a CS074-rcruz/F2-5) — misma historia de
cortes que su pareja REAL (misma semilla), solo se destruye el orden espacial al final.

No se toca cs074_rcruz.py ni F2_5_engine.py. No se cierra el experimento aquí — solo se
reporta crudo (T3: no se ajusta nada tras ver resultados).

*** MODIFICADO 2026-07-25 (ARREGLO 3, definición común de exergía) ***
Ver ADENDA en PROTOCOLO_E5.1-5_PREREGISTRO.md. Se agrega, EN PARALELO a X_final
(std_ratio, vieja, sin tocar), Xh_final = exergia_X(phi) importada verbatim de
BATERIA_ENFOQUE5/_observables_homologadas.py (Xh = (1/N)·Σ(φ_i−1)², absoluta,
phi_eq=1). Se recalcula sobre el MISMO phi final de cada corrida (real y null), se
propaga z_Xh y una segunda pasada de analisis_robustez() sobre Xh, y se guarda el
phi crudo (inicial y final, real y null) por semilla en cada fila del barrido
principal y del mini-estudio de jitter, para poder auditar/recomputar sin re-correr.
No se toca el barrido (mismos perfiles, r_medio, eps, semillas, NULL). Se verificó
que el mecanismo estocástico dinámico de este motor (jitter multiplicativo sobre
H(t), §6.1 del protocolo) NO es el bug de ARREGLO 2 (ruido aditivo constante
acumulado sobre φ): el jitter perturba la TASA de expansión H(t), no φ directamente,
así que ARREGLO 2 no aplica aquí.
"""
from __future__ import annotations

import json
import sys
import time
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE_FILE = HERE.parent.parent / "cs074_rcruz.py"
BASE_DIR = HERE.parent.parent  # Cosmogenesis/

# --- import de cs074_rcruz.py SIN editarlo (import por ruta explícita) ---
spec = importlib.util.spec_from_file_location("cs074_rcruz_base_e515", BASE_FILE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

# --- ARREGLO 3 (2026-07-25): definición común de exergía, ver ADENDA en
# PROTOCOLO_E5.1-5_PREREGISTRO.md. Se importa de _observables_homologadas.py,
# NO se reimplementa la fórmula (mismo patrón que E5_5_3_engine.py). ---
sys.path.insert(0, str(BASE_DIR))
from BATERIA_ENFOQUE5._observables_homologadas import exergia_X  # noqa: E402

OUT = HERE

# ---------------------------------------------------------------------------
# Parámetros pre-registrados (ver PROTOCOLO_E5.1-5_PREREGISTRO.md, secciones 4-6)
# ---------------------------------------------------------------------------
SEED_OFFSET = 3000  # espacio propio: CS074-rcruz usa 1000+, F2-5 usa 2000+
N = 200
EPS_LIST = [0.0, 1e-3]
R_MEDIO_LIST = [1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0]
SEMILLAS = 16
PERFILES = [
    "constante",
    "acelerante",
    "frenante",
    "rafaga_temprana",
    "rafaga_tardia",
    "rafagas_multiples",
]
F_RAFAGA = 0.1     # fracción total de tiempo en ráfaga (perfiles de una sola ráfaga)
W_BURST = 5.0      # amplitud de la ráfaga
K_MULTI = 5        # número de sub-ráfagas en rafagas_multiples (mismo presupuesto total)

# mini-estudio de jitter dinámico (protocolo §6.1)
JITTER_DELTAS = [0.0, 0.3]
JITTER_PERFILES = ["constante", "rafagas_multiples"]
JITTER_R_MEDIO = 1.0
JITTER_EPS = 1e-3

# puntos de muestreo de E_total a lo largo de la trayectoria (fracciones del total de pasos)
E_CHECK_FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]


def forma_w(perfil: str, pasos: int) -> np.ndarray:
    """Forma adimensional w(t), media temporal discreta EXACTA = 1 por construcción."""
    t = np.arange(pasos, dtype=float)
    if perfil == "constante":
        w = np.ones(pasos, dtype=float)
    elif perfil == "acelerante":
        w = 2.0 * (t + 0.5) / pasos
    elif perfil == "frenante":
        w = 2.0 * (pasos - t - 0.5) / pasos
    elif perfil == "rafaga_temprana":
        n_burst = max(1, int(round(F_RAFAGA * pasos)))
        w_slow = (1.0 - F_RAFAGA * W_BURST) / (1.0 - F_RAFAGA)
        w = np.full(pasos, w_slow, dtype=float)
        w[:n_burst] = W_BURST
    elif perfil == "rafaga_tardia":
        n_burst = max(1, int(round(F_RAFAGA * pasos)))
        w_slow = (1.0 - F_RAFAGA * W_BURST) / (1.0 - F_RAFAGA)
        w = np.full(pasos, w_slow, dtype=float)
        w[-n_burst:] = W_BURST
    elif perfil == "rafagas_multiples":
        # K_MULTI ráfagas iguales, equiespaciadas, mismo presupuesto total F_RAFAGA
        n_burst_total = max(K_MULTI, int(round(F_RAFAGA * pasos)))
        n_burst_each = max(1, n_burst_total // K_MULTI)
        w_slow = (1.0 - F_RAFAGA * W_BURST) / (1.0 - F_RAFAGA)
        w = np.full(pasos, w_slow, dtype=float)
        # centros equiespaciados dentro de [0, pasos)
        centros = np.linspace(pasos / (2 * K_MULTI), pasos - pasos / (2 * K_MULTI), K_MULTI)
        for c in centros:
            i0 = int(round(c - n_burst_each / 2))
            i1 = i0 + n_burst_each
            i0 = max(0, i0)
            i1 = min(pasos, i1)
            w[i0:i1] = W_BURST
    else:
        raise ValueError(f"perfil desconocido: {perfil}")
    return w


def construir_H(perfil: str, pasos: int, H_bar: float):
    w = forma_w(perfil, pasos)
    w_mean_exacto = float(w.mean())
    H_raw = H_bar * w
    H = np.clip(H_raw, 0.0, 1.0)
    H_bar_realizado = float(H.mean())
    clip_afectado = bool(np.any(H_raw > 1.0))
    return H, w_mean_exacto, H_bar_realizado, clip_afectado


def evolucionar_perfil(phi, activo, H_arr, rng, jitter_delta=0.0, null=False):
    """Difusión, luego expansión, por paso, con H(t) del perfil. Audita E_total=sum(phi)
    en los puntos de E_CHECK_FRACS (T6: conservación verificada, no asumida)."""
    pasos = len(H_arr)
    contraste0 = float(phi.std())
    E0 = float(phi.sum())
    check_steps = sorted(set(int(round(f * (pasos - 1))) for f in E_CHECK_FRACS)) if pasos > 0 else []
    E_checks = {}
    if 0 in check_steps or pasos == 0:
        E_checks[0] = E0
    for t, Ht in enumerate(H_arr):
        phi = base.paso_difusion(phi, activo)
        Ht_ef = float(Ht)
        if jitter_delta > 0.0:
            xi = rng.uniform(1.0 - jitter_delta, 1.0 + jitter_delta)
            Ht_ef = float(np.clip(Ht * xi, 0.0, 1.0))
        activo = base.paso_expansion(activo, Ht_ef, rng)
        if (t + 1) in check_steps:
            E_checks[t + 1] = float(phi.sum())
    if null:
        phi = rng.permutation(phi)
    E_final = float(phi.sum()) if not null else E_checks.get(pasos - 1 if pasos > 0 else 0, E0)
    if E0 != 0:
        drifts = [abs(v - E0) / abs(E0) for v in E_checks.values()]
    else:
        drifts = [abs(v - E0) for v in E_checks.values()]
    E_drift_rel_max = float(max(drifts)) if drifts else 0.0
    return phi, activo, contraste0, E0, E_drift_rel_max


def corrida_perfil(N, eps, perfil, H_bar, pasos, seed, jitter_delta=0.0, null=False):
    rng = np.random.default_rng(seed)
    phi, _ = base.campo_inicial(N, eps, rng)
    phi_inicial = phi.copy()  # crudo, ANTES de evolucionar (Arreglo 3: guardar detalle)
    activo = np.ones(N, dtype=bool)
    H_arr, w_mean, H_bar_real, clip_af = construir_H(perfil, pasos, H_bar)
    phi, activo, c0, E0, E_drift = evolucionar_perfil(
        phi, activo, H_arr, rng, jitter_delta=jitter_delta, null=null
    )
    P = base.persistencia(phi, c0)
    X_final = float(phi.std() / c0) if c0 > 0 else 0.0
    # ARREGLO 3 (2026-07-25): exergía canónica homologada, EN PARALELO a X_final
    # (std_ratio, vieja), sobre el MISMO phi final de esta corrida (real o null).
    Xh_final = exergia_X(phi)
    frac_exp = 1.0 - float(activo.mean())
    return {
        "P": P,
        "X_final": X_final,
        "Xh_final": Xh_final,
        "frac_exp": frac_exp,
        "w_mean_exacto": w_mean,
        "H_bar_realizado": H_bar_real,
        "clip_afectado": clip_af,
        "E_drift_rel_max": E_drift,
        "phi_final": phi,          # array crudo (np.ndarray), el caller decide si serializa
        "phi_inicial": phi_inicial,  # array crudo, idéntico entre rama real/null (misma seed)
    }


def r_efectivo_realizado(frac_exp, pasos, D):
    """r equivalente-constante que habría dado la misma supervivencia de aristas en
    `pasos` pasos (idéntica fórmula/razonamiento que F2_5_engine.py). Honestidad del
    instrumento: si frac_exp==1.0 exacto (fragmentación total), ya no hay información
    para distinguir r_medio grandes entre sí -> se devuelve inf y se marca saturado."""
    if D <= 0:
        return float("inf") if frac_exp > 0 else 0.0, False
    if frac_exp >= 1.0:
        return float("inf"), True
    surv = 1.0 - frac_exp
    H_const = 1.0 - surv ** (1.0 / pasos)
    return float(H_const / D), False


def barrido_principal(N, eps_list, r_medio_list, perfiles, semillas, D, pasos):
    filas = []
    for eps in eps_list:
        for perfil in perfiles:
            for r_medio in r_medio_list:
                H_bar = min(r_medio * D, 1.0) if D > 0 else (0.0 if r_medio == 0 else 1.0)
                Xreal, Xnull, Preal, Pnull = [], [], [], []
                Xhreal, Xhnull = [], []
                frac_real, frac_null = [], []
                w_means, Hbar_reals, clip_flags, Edrifts = [], [], [], []
                phi_final_real_list, phi_final_null_list, phi_inicial_list = [], [], []
                for i in range(semillas):
                    seed = SEED_OFFSET + i
                    rr = corrida_perfil(N, eps, perfil, H_bar, pasos, seed, null=False)
                    nn = corrida_perfil(N, eps, perfil, H_bar, pasos, seed, null=True)
                    Xreal.append(rr["X_final"])
                    Xnull.append(nn["X_final"])
                    Xhreal.append(rr["Xh_final"])
                    Xhnull.append(nn["Xh_final"])
                    Preal.append(rr["P"])
                    Pnull.append(nn["P"])
                    frac_real.append(rr["frac_exp"])
                    frac_null.append(nn["frac_exp"])
                    w_means.append(rr["w_mean_exacto"])
                    Hbar_reals.append(rr["H_bar_realizado"])
                    clip_flags.append(rr["clip_afectado"])
                    Edrifts.append(rr["E_drift_rel_max"])
                    # ARREGLO 3: guardar detalle crudo (phi_final real+null, phi_inicial
                    # -- idéntico entre real/null porque comparten semilla) para las 16
                    # semillas de esta fila; N=200 es chico, es barato guardarlo todo.
                    phi_final_real_list.append([round(float(x), 6) for x in rr["phi_final"]])
                    phi_final_null_list.append([round(float(x), 6) for x in nn["phi_final"]])
                    phi_inicial_list.append([round(float(x), 6) for x in rr["phi_inicial"]])
                Xreal = np.array(Xreal); Xnull = np.array(Xnull)
                Xhreal = np.array(Xhreal); Xhnull = np.array(Xhnull)
                Preal = np.array(Preal); Pnull = np.array(Pnull)
                sdX = np.sqrt((Xreal.var() + Xnull.var()) / 2.0)
                sdX = max(sdX, 1.0 / max(len(Xreal), 1))
                zX = float((Xreal.mean() - Xnull.mean()) / sdX)
                sdXh = np.sqrt((Xhreal.var() + Xhnull.var()) / 2.0)
                sdXh = max(sdXh, 1.0 / max(len(Xhreal), 1))
                zXh = float((Xhreal.mean() - Xhnull.mean()) / sdXh)
                sdP = np.sqrt((Preal.var() + Pnull.var()) / 2.0)
                sdP = max(sdP, 1.0 / max(len(Preal), 1))
                zP = float((Preal.mean() - Pnull.mean()) / sdP)
                frac_real_mean = float(np.mean(frac_real))
                r_eff, r_eff_sat = r_efectivo_realizado(frac_real_mean, pasos, D)
                filas.append({
                    "eps": eps,
                    "perfil": perfil,
                    "r_medio": r_medio,
                    "H_bar_objetivo": H_bar,
                    "D": D,
                    "pasos": pasos,
                    "X_final_real_mean": float(Xreal.mean()),
                    "X_final_real_std": float(Xreal.std()),
                    "X_final_null_mean": float(Xnull.mean()),
                    "X_final_null_std": float(Xnull.std()),
                    "z_X": round(zX, 4),
                    "Xh_final_real_mean": float(Xhreal.mean()),
                    "Xh_final_real_std": float(Xhreal.std()),
                    "Xh_final_null_mean": float(Xhnull.mean()),
                    "Xh_final_null_std": float(Xhnull.std()),
                    "z_Xh": round(zXh, 4),
                    "P_real_mean": float(Preal.mean()),
                    "P_real_std": float(Preal.std()),
                    "P_null_mean": float(Pnull.mean()),
                    "P_null_std": float(Pnull.std()),
                    "z_P": round(zP, 4),
                    "frac_exp_real_mean": frac_real_mean,
                    "frac_exp_null_mean": float(np.mean(frac_null)),
                    "r_efectivo_realizado": r_eff,
                    "r_efectivo_saturado": r_eff_sat,
                    "w_mean_exacto_mean": float(np.mean(w_means)),
                    "H_bar_realizado_mean": float(np.mean(Hbar_reals)),
                    "clip_afectado_alguna_semilla": bool(any(clip_flags)),
                    "E_drift_rel_max": float(np.max(Edrifts)),
                    "E_drift_rel_mean": float(np.mean(Edrifts)),
                    "n_semillas": semillas,
                    "X_final_real_por_semilla": [float(x) for x in Xreal],
                    "X_final_null_por_semilla": [float(x) for x in Xnull],
                    "Xh_final_real_por_semilla": [float(x) for x in Xhreal],
                    "Xh_final_null_por_semilla": [float(x) for x in Xhnull],
                    "phi_final_real_por_semilla": phi_final_real_list,
                    "phi_final_null_por_semilla": phi_final_null_list,
                    "phi_inicial_por_semilla": phi_inicial_list,
                })
    return filas


def mini_estudio_jitter(N, eps, r_medio, perfiles, deltas, semillas, D, pasos):
    filas = []
    H_bar = min(r_medio * D, 1.0) if D > 0 else 0.0
    for perfil in perfiles:
        for delta in deltas:
            Xreal, Xnull, frac_real = [], [], []
            Xhreal, Xhnull = [], []
            phi_final_real_list, phi_final_null_list, phi_inicial_list = [], [], []
            for i in range(semillas):
                seed = SEED_OFFSET + i
                rr = corrida_perfil(N, eps, perfil, H_bar, pasos, seed, jitter_delta=delta, null=False)
                nn = corrida_perfil(N, eps, perfil, H_bar, pasos, seed, jitter_delta=delta, null=True)
                Xreal.append(rr["X_final"])
                Xnull.append(nn["X_final"])
                Xhreal.append(rr["Xh_final"])
                Xhnull.append(nn["Xh_final"])
                frac_real.append(rr["frac_exp"])
                phi_final_real_list.append([round(float(x), 6) for x in rr["phi_final"]])
                phi_final_null_list.append([round(float(x), 6) for x in nn["phi_final"]])
                phi_inicial_list.append([round(float(x), 6) for x in rr["phi_inicial"]])
            Xreal = np.array(Xreal); Xnull = np.array(Xnull)
            Xhreal = np.array(Xhreal); Xhnull = np.array(Xhnull)
            sd = np.sqrt((Xreal.var() + Xnull.var()) / 2.0)
            sd = max(sd, 1.0 / max(len(Xreal), 1))
            z = float((Xreal.mean() - Xnull.mean()) / sd)
            sdh = np.sqrt((Xhreal.var() + Xhnull.var()) / 2.0)
            sdh = max(sdh, 1.0 / max(len(Xhreal), 1))
            zh = float((Xhreal.mean() - Xhnull.mean()) / sdh)
            frac_real_mean = float(np.mean(frac_real))
            r_eff, r_eff_sat = r_efectivo_realizado(frac_real_mean, pasos, D)
            filas.append({
                "perfil": perfil,
                "jitter_delta": delta,
                "r_medio": r_medio,
                "eps": eps,
                "X_final_real_mean": float(Xreal.mean()),
                "X_final_real_std": float(Xreal.std()),
                "X_final_null_mean": float(Xnull.mean()),
                "z_X": round(z, 4),
                "Xh_final_real_mean": float(Xhreal.mean()),
                "Xh_final_real_std": float(Xhreal.std()),
                "Xh_final_null_mean": float(Xhnull.mean()),
                "z_Xh": round(zh, 4),
                "frac_exp_real_mean": frac_real_mean,
                "r_efectivo_realizado": r_eff,
                "r_efectivo_saturado": r_eff_sat,
                "n_semillas": semillas,
                "phi_final_real_por_semilla": phi_final_real_list,
                "phi_final_null_por_semilla": phi_final_null_list,
                "phi_inicial_por_semilla": phi_inicial_list,
            })
    return filas


def analisis_robustez(filas, eps_signal, campo_mean="X_final_real_mean", campo_std="X_final_real_std"):
    """Bin por r_efectivo (log-ancho); para cada bin compara dispersion ENTRE PERFILES
    (a r_efectivo fijo) vs dispersion ENTRE SEMILLAS (dentro de un mismo perfil).
    Criterio protocolo §8(c): razón <= 2x -> robusto; > 2x -> ruptura reportada.

    ARREGLO 3 (2026-07-25): parametrizado por `campo_mean`/`campo_std` para poder
    correr EXACTAMENTE el mismo análisis (mismos bins, mismo criterio ≤2x) sobre
    X (std_ratio, vieja) o Xh (exergía canónica homologada), sin duplicar lógica."""
    filas_signal = [f for f in filas if f["eps"] == eps_signal and np.isfinite(f["r_efectivo_realizado"])]
    if not filas_signal:
        return {"bins": [], "nota": "sin filas finitas para eps_signal", "campo_mean": campo_mean, "campo_std": campo_std}
    r_effs = np.array([f["r_efectivo_realizado"] for f in filas_signal])
    r_effs_pos = r_effs[r_effs > 0]
    if len(r_effs_pos) == 0:
        return {"bins": [], "nota": "todos r_efectivo <= 0", "campo_mean": campo_mean, "campo_std": campo_std}
    lo, hi = np.log10(r_effs_pos.min()), np.log10(max(r_effs.max(), r_effs_pos.min() * 10))
    n_bins = 8
    edges = np.linspace(lo, hi + 1e-9, n_bins + 1)
    bins_out = []
    for b in range(n_bins):
        lo_e, hi_e = 10 ** edges[b], 10 ** edges[b + 1]
        en_bin = [f for f in filas_signal if lo_e <= f["r_efectivo_realizado"] < hi_e]
        if len(en_bin) < 2:
            continue
        x_por_perfil = {}
        for f in en_bin:
            x_por_perfil.setdefault(f["perfil"], []).append(f[campo_mean])
        medias_perfil = [np.mean(v) for v in x_por_perfil.values()]
        disp_entre_perfiles = float(np.std(medias_perfil)) if len(medias_perfil) > 1 else 0.0
        disp_entre_semillas = float(np.mean([f[campo_std] for f in en_bin]))
        disp_entre_semillas = max(disp_entre_semillas, 1e-6)
        razon = disp_entre_perfiles / disp_entre_semillas
        bins_out.append({
            "r_eff_lo": float(lo_e),
            "r_eff_hi": float(hi_e),
            "n_perfiles_en_bin": len(x_por_perfil),
            "perfiles": list(x_por_perfil.keys()),
            "X_mean_por_perfil": {k: float(np.mean(v)) for k, v in x_por_perfil.items()},
            "disp_entre_perfiles": disp_entre_perfiles,
            "disp_entre_semillas_media": disp_entre_semillas,
            "razon_perfil_vs_semilla": razon,
            "robusto": bool(razon <= 2.0),
        })
    return {"bins": bins_out, "n_bins_evaluados": len(bins_out), "campo_mean": campo_mean, "campo_std": campo_std}


def main():
    t0 = time.time()
    ts_inicio = datetime.now(timezone.utc).isoformat()

    eps_cal = 1e-3
    D = float(np.mean([base.medir_D(N, eps_cal, s) for s in range(SEMILLAS)]))
    cal = base.medir_pasos_lavado(N, eps_cal, SEMILLAS)
    pasos = cal["pasos"]
    print(
        f"[calibracion] N={N} eps={eps_cal} D={D:.8g} mediana_lavado={cal['mediana']} "
        f"pasos={pasos} lavo_todas={cal['lavo_todas']}",
        file=sys.stderr, flush=True,
    )

    filas = barrido_principal(N, EPS_LIST, R_MEDIO_LIST, PERFILES, SEMILLAS, D, pasos)
    print(f"[barrido_principal] {len(filas)} filas listas t={time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    filas_jitter = mini_estudio_jitter(
        N, JITTER_EPS, JITTER_R_MEDIO, JITTER_PERFILES, JITTER_DELTAS, SEMILLAS, D, pasos
    )
    print(f"[mini_jitter] {len(filas_jitter)} filas listas t={time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    robustez = analisis_robustez(filas, eps_signal=1e-3)
    print(f"[robustez X] {robustez['n_bins_evaluados']} bins evaluados t={time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    # ARREGLO 3 (2026-07-25): MISMO análisis de robustez, sobre Xh (canónica homologada).
    robustez_Xh = analisis_robustez(
        filas, eps_signal=1e-3, campo_mean="Xh_final_real_mean", campo_std="Xh_final_real_std"
    )
    print(f"[robustez Xh] {robustez_Xh['n_bins_evaluados']} bins evaluados t={time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    null_X = [f["X_final_null_mean"] for f in filas]
    null_X_max = max(null_X) if null_X else None
    null_Xh = [f["Xh_final_null_mean"] for f in filas]
    null_Xh_max = max(null_Xh) if null_Xh else None
    null_P = [f["P_null_mean"] for f in filas]
    null_P_max = max(null_P) if null_P else None
    eps0_X = [f["X_final_real_mean"] for f in filas if f["eps"] == 0.0]
    eps0_X_max = max(eps0_X) if eps0_X else None
    eps0_Xh = [f["Xh_final_real_mean"] for f in filas if f["eps"] == 0.0]
    eps0_Xh_max = max(eps0_Xh) if eps0_Xh else None
    eps0_P = [f["P_real_mean"] for f in filas if f["eps"] == 0.0]
    eps0_P_max = max(eps0_P) if eps0_P else None
    E_drift_global_max = max(f["E_drift_rel_max"] for f in filas) if filas else None

    t1 = time.time()
    ts_fin = datetime.now(timezone.utc).isoformat()

    result = {
        "experimento": "E5.1-5_persistencia_exergia_expansion_no_monotona",
        "protocolo": "PROTOCOLO_E5.1-5_PREREGISTRO.md",
        "base_fisica_reusada": str(BASE_FILE),
        "referencia_diseno_perfiles": str(HERE.parent.parent / "BATERIA_FUNDAMENTOS" / "F2_5_expansion_no_uniforme" / "F2_5_engine.py"),
        "arreglo_3_definicion_comun": {
            "aplicado": True,
            "fecha": "2026-07-25",
            "modulo": str(HERE.parent / "_observables_homologadas.py"),
            "formula_Xh": "Xh = (1/N) * sum_i (phi_i - 1)^2  (exergia canonica, absoluta, phi_eq=1)",
            "formula_X_vieja": "X = std(phi_final) / std(phi_inicial)  (std_ratio, sin factor de autocorrelacion)",
            "nota": "X (vieja, std_ratio) se conserva sin tocar para comparacion lado a lado; "
                    "Xh (canonica) se calcula EN PARALELO sobre el MISMO phi final de cada corrida "
                    "(real y null). Ver ADENDA 2026-07-25 en PROTOCOLO_E5.1-5_PREREGISTRO.md.",
        },
        "timestamp_inicio_utc": ts_inicio,
        "timestamp_fin_utc": ts_fin,
        "elapsed_s": t1 - t0,
        "N": N,
        "eps_list": EPS_LIST,
        "r_medio_list": R_MEDIO_LIST,
        "perfiles": PERFILES,
        "f_rafaga": F_RAFAGA,
        "w_burst": W_BURST,
        "k_multi": K_MULTI,
        "semillas": SEMILLAS,
        "seed_offset": SEED_OFFSET,
        "D_medido": D,
        "pasos": pasos,
        "calibracion_lavado": cal,
        "controles": {
            "X_final_null_max_observado": null_X_max,
            "Xh_final_null_max_observado": null_Xh_max,
            "P_null_max_observado": null_P_max,
            "X_final_eps0_max_observado": eps0_X_max,
            "Xh_final_eps0_max_observado": eps0_Xh_max,
            "P_eps0_max_observado": eps0_P_max,
            "E_drift_rel_max_global": E_drift_global_max,
        },
        "filas": filas,
        "analisis_robustez_perfil_vs_r_efectivo": robustez,
        "analisis_robustez_perfil_vs_r_efectivo_Xh": robustez_Xh,
        "mini_estudio_jitter": {
            "deltas": JITTER_DELTAS,
            "perfiles": JITTER_PERFILES,
            "r_medio": JITTER_R_MEDIO,
            "eps": JITTER_EPS,
            "filas": filas_jitter,
        },
    }

    out_json = OUT / "E5_1_5_resultado_crudo.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[archivo] {out_json}", file=sys.stderr)
    print(f"[controles] X_null_max={null_X_max} Xh_null_max={null_Xh_max} P_null_max={null_P_max} "
          f"X_eps0_max={eps0_X_max} Xh_eps0_max={eps0_Xh_max} P_eps0_max={eps0_P_max} "
          f"E_drift_max={E_drift_global_max}",
          file=sys.stderr)
    print(f"[elapsed] {result['elapsed_s']:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
