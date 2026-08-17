#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F2_5_engine.py — Congelamiento bajo expansión no uniforme (historia temporal variable)
========================================================================================

Implementa el protocolo pre-registrado en PROTOCOLO_F2-5_PREREGISTRO.md (léelo primero;
este motor NO se corre sin ese archivo fechado y congelado).

Pregunta: la expansión real no es a tasa constante — ¿el congelamiento de la diferencia
(persistencia P) aguanta si la tasa H cambia en el tiempo, manteniendo la misma expansión
TOTAL integrada? ¿El veredicto depende del PERFIL H(t) específico o solo del r efectivo
integrado?

Reusa la física de cs074_rcruz.py (CS074-rcruz) SIN EDITARLO: se importa el módulo y se
llaman sus funciones (campo_inicial, paso_difusion, paso_expansion, persistencia, medir_D,
medir_pasos_lavado). Lo único nuevo aquí es la construcción de H(t) NO constante y el
bucle de evolución que la usa paso a paso.

5 perfiles de forma w(t) (media temporal discreta EXACTA = 1 por construcción):
  - constante:      w(t) = 1
  - acelerando:      w(t) = 2*(t+0.5)/pasos            (rampa 0->2, empieza lento)
  - desacelerando:   w(t) = 2*(pasos-t-0.5)/pasos       (espejo: empieza fuerte)
  - rafaga_lento:    primeros 10% del tiempo w=5.0, resto w≈0.556 (ráfaga al inicio)
  - lento_rafaga:    espejo temporal de rafaga_lento (ráfaga al final)

H(t) = clip(H_bar * w(t), 0, 1), con H_bar = r_medio * D (D medido, no impuesto).

Verificación cruzada central (T1/T7): tras cada corrida se mide la fracción de aristas
realmente cortada (frac_exp) y se calcula el r_efectivo REALIZADO (equivalente-constante
que habría dado la misma supervivencia de aristas en `pasos` pasos). Los perfiles se
comparan en esa vara común, no en el r_medio nominal.

NULL: barajado del campo al final (idéntico a CS074-rcruz) — misma historia de cortes
que su pareja REAL (misma semilla), solo se destruye el orden espacial al final.

Segundo observable (independiente de la autocorrelación): std_ratio = phi.std()/contraste0.

No se toca cs074_rcruz.py. No se cierra el experimento aquí — solo se reporta crudo.
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

# --- import de cs074_rcruz.py SIN editarlo (import por ruta explícita) ---
spec = importlib.util.spec_from_file_location("cs074_rcruz_base", BASE_FILE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

OUT = HERE

# ---------------------------------------------------------------------------
# Parámetros pre-registrados (ver PROTOCOLO_F2-5_PREREGISTRO.md, secciones 2-3)
# ---------------------------------------------------------------------------
SEED_OFFSET = 2000
N = 200
EPS_LIST = [0.0, 1e-3]
R_MEDIO_LIST = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
SEMILLAS = 16
PERFILES = ["constante", "acelerando", "desacelerando", "rafaga_lento", "lento_rafaga"]
F_RAFAGA = 0.1
W_BURST = 5.0

# mini-estudio de jitter dinámico (sección 3b)
JITTER_DELTAS = [0.0, 0.3]
JITTER_PERFILES = ["constante", "rafaga_lento"]
JITTER_R_MEDIO = 1.0
JITTER_EPS = 1e-3


def forma_w(perfil: str, pasos: int) -> np.ndarray:
    t = np.arange(pasos, dtype=float)
    if perfil == "constante":
        w = np.ones(pasos, dtype=float)
    elif perfil == "acelerando":
        w = 2.0 * (t + 0.5) / pasos
    elif perfil == "desacelerando":
        w = 2.0 * (pasos - t - 0.5) / pasos
    elif perfil == "rafaga_lento":
        n_burst = max(1, int(round(F_RAFAGA * pasos)))
        w_slow = (1.0 - F_RAFAGA * W_BURST) / (1.0 - F_RAFAGA)
        w = np.full(pasos, w_slow, dtype=float)
        w[:n_burst] = W_BURST
    elif perfil == "lento_rafaga":
        n_burst = max(1, int(round(F_RAFAGA * pasos)))
        w_slow = (1.0 - F_RAFAGA * W_BURST) / (1.0 - F_RAFAGA)
        w = np.full(pasos, w_slow, dtype=float)
        w[-n_burst:] = W_BURST
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
    """Mismo orden por paso que base.evolucionar: difusión, luego expansión.
    H_arr trae un valor de H por paso (ya con la forma del perfil aplicada).
    Si jitter_delta>0, se multiplica H(t) por xi~U(1-d,1+d) ANTES de cortar,
    usando la MISMA rng (reproducible por semilla, comparte draws con REAL/NULL
    hasta el shuffle final)."""
    contraste0 = float(phi.std())
    for Ht in H_arr:
        phi = base.paso_difusion(phi, activo)
        Ht_ef = float(Ht)
        if jitter_delta > 0.0:
            xi = rng.uniform(1.0 - jitter_delta, 1.0 + jitter_delta)
            Ht_ef = float(np.clip(Ht * xi, 0.0, 1.0))
        activo = base.paso_expansion(activo, Ht_ef, rng)
    if null:
        phi = rng.permutation(phi)
    return phi, activo, contraste0


def corrida_perfil(N, eps, perfil, H_bar, pasos, seed, jitter_delta=0.0, null=False):
    rng = np.random.default_rng(seed)
    phi, _ = base.campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    H_arr, w_mean, H_bar_real, clip_af = construir_H(perfil, pasos, H_bar)
    phi, activo, c0 = evolucionar_perfil(
        phi, activo, H_arr, rng, jitter_delta=jitter_delta, null=null
    )
    P = base.persistencia(phi, c0)
    std_ratio = float(phi.std() / c0) if c0 > 0 else 0.0
    frac_exp = 1.0 - float(activo.mean())
    return {
        "P": P,
        "std_ratio": std_ratio,
        "frac_exp": frac_exp,
        "w_mean_exacto": w_mean,
        "H_bar_realizado": H_bar_real,
        "clip_afectado": clip_af,
    }


def r_efectivo_realizado(frac_exp, pasos, D):
    """r equivalente-constante que habría dado la misma supervivencia de aristas
    en `pasos` pasos: (1-H_const)^pasos = 1-frac_exp  =>  H_const = 1-(1-frac_exp)^(1/pasos)
    r_efectivo = H_const / D.

    HONESTIDAD DEL INSTRUMENTO (T5): si frac_exp==1.0 exacto (fragmentación total
    del anillo, todas las aristas cortadas antes de terminar la corrida — ocurre en
    float64 cuando la probabilidad de supervivencia de UNA arista cae bajo ~1e-16,
    caso real y no numérico aquí porque además hay solo N aristas y su número
    esperado de sobrevivientes ya es <1), el estado final ya NO contiene información
    para distinguir r_medio grandes entre sí (r=10 y r=30 lucen idénticos: 0
    aristas vivas). En vez de inventar un número finito con un clamp arbitrario,
    se devuelve inf y se marca `saturado=True` para que el análisis no confunda
    "no medible" con "medido"."""
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
                Preal, Pnull = [], []
                sr_real, sr_null = [], []
                frac_real, frac_null = [], []
                w_means, Hbar_reals, clip_flags = [], [], []
                for i in range(semillas):
                    seed = SEED_OFFSET + i
                    rr = corrida_perfil(N, eps, perfil, H_bar, pasos, seed, null=False)
                    nn = corrida_perfil(N, eps, perfil, H_bar, pasos, seed, null=True)
                    Preal.append(rr["P"])
                    Pnull.append(nn["P"])
                    sr_real.append(rr["std_ratio"])
                    sr_null.append(nn["std_ratio"])
                    frac_real.append(rr["frac_exp"])
                    frac_null.append(nn["frac_exp"])
                    w_means.append(rr["w_mean_exacto"])
                    Hbar_reals.append(rr["H_bar_realizado"])
                    clip_flags.append(rr["clip_afectado"])
                Preal = np.array(Preal)
                Pnull = np.array(Pnull)
                sd = np.sqrt((Preal.var() + Pnull.var()) / 2.0)
                sd = max(sd, 1.0 / max(len(Preal), 1))
                z = float((Preal.mean() - Pnull.mean()) / sd)
                frac_real_mean = float(np.mean(frac_real))
                r_eff, r_eff_saturado = r_efectivo_realizado(frac_real_mean, pasos, D)
                filas.append(
                    {
                        "eps": eps,
                        "perfil": perfil,
                        "r_medio": r_medio,
                        "H_bar_objetivo": H_bar,
                        "D": D,
                        "pasos": pasos,
                        "P_real_mean": float(Preal.mean()),
                        "P_real_std": float(Preal.std()),
                        "P_null_mean": float(Pnull.mean()),
                        "P_null_std": float(Pnull.std()),
                        "z": round(z, 4),
                        "std_ratio_real_mean": float(np.mean(sr_real)),
                        "std_ratio_null_mean": float(np.mean(sr_null)),
                        "frac_exp_real_mean": frac_real_mean,
                        "frac_exp_null_mean": float(np.mean(frac_null)),
                        "r_efectivo_realizado": r_eff,
                        "r_efectivo_saturado": r_eff_saturado,
                        "w_mean_exacto_mean": float(np.mean(w_means)),
                        "H_bar_realizado_mean": float(np.mean(Hbar_reals)),
                        "clip_afectado_alguna_semilla": bool(any(clip_flags)),
                        "n_semillas": semillas,
                        "P_real_por_semilla": [float(x) for x in Preal],
                        "P_null_por_semilla": [float(x) for x in Pnull],
                    }
                )
    return filas


def mini_estudio_jitter(N, eps, r_medio, perfiles, deltas, semillas, D, pasos):
    filas = []
    H_bar = min(r_medio * D, 1.0) if D > 0 else 0.0
    for perfil in perfiles:
        for delta in deltas:
            Preal, Pnull, frac_real = [], [], []
            for i in range(semillas):
                seed = SEED_OFFSET + i
                rr = corrida_perfil(N, eps, perfil, H_bar, pasos, seed, jitter_delta=delta, null=False)
                nn = corrida_perfil(N, eps, perfil, H_bar, pasos, seed, jitter_delta=delta, null=True)
                Preal.append(rr["P"])
                Pnull.append(nn["P"])
                frac_real.append(rr["frac_exp"])
            Preal = np.array(Preal)
            Pnull = np.array(Pnull)
            sd = np.sqrt((Preal.var() + Pnull.var()) / 2.0)
            sd = max(sd, 1.0 / max(len(Preal), 1))
            z = float((Preal.mean() - Pnull.mean()) / sd)
            frac_real_mean = float(np.mean(frac_real))
            r_eff, r_eff_saturado = r_efectivo_realizado(frac_real_mean, pasos, D)
            filas.append(
                {
                    "perfil": perfil,
                    "jitter_delta": delta,
                    "r_medio": r_medio,
                    "eps": eps,
                    "P_real_mean": float(Preal.mean()),
                    "P_real_std": float(Preal.std()),
                    "P_null_mean": float(Pnull.mean()),
                    "z": round(z, 4),
                    "frac_exp_real_mean": frac_real_mean,
                    "r_efectivo_realizado": r_eff,
                    "r_efectivo_saturado": r_eff_saturado,
                    "n_semillas": semillas,
                }
            )
    return filas


def main():
    t0 = time.time()
    ts_inicio = datetime.now(timezone.utc).isoformat()

    # --- calibración (misma receta que CS074-rcruz modo `produccion`) ---
    eps_cal = 1e-3
    D = float(np.mean([base.medir_D(N, eps_cal, s) for s in range(SEMILLAS)]))
    cal = base.medir_pasos_lavado(N, eps_cal, SEMILLAS)
    pasos = cal["pasos"]
    print(
        f"[calibracion] N={N} eps={eps_cal} D={D:.8g} mediana_lavado={cal['mediana']} "
        f"pasos={pasos} lavo_todas={cal['lavo_todas']}",
        file=sys.stderr,
        flush=True,
    )

    filas = barrido_principal(N, EPS_LIST, R_MEDIO_LIST, PERFILES, SEMILLAS, D, pasos)
    print(f"[barrido_principal] {len(filas)} filas listas t={time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    filas_jitter = mini_estudio_jitter(
        N, JITTER_EPS, JITTER_R_MEDIO, JITTER_PERFILES, JITTER_DELTAS, SEMILLAS, D, pasos
    )
    print(f"[mini_jitter] {len(filas_jitter)} filas listas t={time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    # --- controles automáticos (no adjudican el veredicto físico, solo validez) ---
    null_filas = [f["P_null_mean"] for f in filas]
    null_max = max(null_filas) if null_filas else None
    eps0_filas = [f["P_real_mean"] for f in filas if f["eps"] == 0.0]
    eps0_max = max(eps0_filas) if eps0_filas else None

    t1 = time.time()
    ts_fin = datetime.now(timezone.utc).isoformat()

    result = {
        "experimento": "F2-5_expansion_no_uniforme",
        "protocolo": "PROTOCOLO_F2-5_PREREGISTRO.md",
        "base_fisica_reusada": str(BASE_FILE),
        "timestamp_inicio_utc": ts_inicio,
        "timestamp_fin_utc": ts_fin,
        "elapsed_s": t1 - t0,
        "N": N,
        "eps_list": EPS_LIST,
        "r_medio_list": R_MEDIO_LIST,
        "perfiles": PERFILES,
        "f_rafaga": F_RAFAGA,
        "w_burst": W_BURST,
        "semillas": SEMILLAS,
        "seed_offset": SEED_OFFSET,
        "D_medido": D,
        "pasos": pasos,
        "calibracion_lavado": cal,
        "controles": {
            "null_P_max_observado": null_max,
            "eps0_P_max_observado": eps0_max,
        },
        "filas": filas,
        "mini_estudio_jitter": {
            "deltas": JITTER_DELTAS,
            "perfiles": JITTER_PERFILES,
            "r_medio": JITTER_R_MEDIO,
            "eps": JITTER_EPS,
            "filas": filas_jitter,
        },
    }

    out_json = OUT / "F2_5_resultado_crudo.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[archivo] {out_json}", file=sys.stderr)
    print(f"[controles] null_P_max={null_max} eps0_P_max={eps0_max}", file=sys.stderr)
    print(f"[elapsed] {result['elapsed_s']:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
