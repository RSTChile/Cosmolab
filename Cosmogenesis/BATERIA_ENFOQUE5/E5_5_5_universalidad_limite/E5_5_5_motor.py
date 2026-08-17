#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5.5-5 — Universalidad del límite: ¿todos los ε→0 llegan al mismo estado?
===========================================================================

Pregunta (Tema 5, Enfoque 5): distintas formas iniciales de diferencia (6+
familias de perturbación), dejadas evolucionar hasta el estado de muerte
térmica (r=0, pasos calibrados con margen ×5 sobre el lavado medido) —
¿mueren todas en el mismo (E, X, S_ent), o hay dependencia de la forma
inicial?

Protocolo congelado ANTES de este motor:
  PROTOCOLO_E5.5-5_PREREGISTRO.md (mismo directorio)

Física reutilizada SIN MODIFICAR de cs074_rcruz.py (import por importlib,
igual mecanismo que F1_4_motor.py, no se copia a mano):
  paso_difusion, paso_expansion, persistencia, detectar_cuantizacion,
  temperatura_fisica, T_SING.

Las 6 familias de forma inicial son una reimplementación LITERAL (fórmula
por fórmula, verificada) de las definidas y congeladas en
BATERIA_FUNDAMENTOS/F1_4_familias_forma/F1_4_motor.py::campo_familia — no
se importan por path relativo entre baterías (carpetas hermanas
independientes) pero el código es idéntico, para comparabilidad directa.

Observables (definidos y justificados en el pre-registro §4):
  X (exergía)    = persistencia() de cs074_rcruz, sin modificar.
  E (energía)    = mean(phi_final)/mean(phi_inicial) — auditoría de E1.
  S_ent (entropía) = Gibbs sobre p_i=phi_i/sum(phi) (piso 1e-9), normalizada
                     por log(N) — máxima en el estado uniforme (muerte
                     térmica), NO un histograma de valores (ver justificación
                     en el pre-registro: histograma iría en la dirección
                     opuesta).

NULL: permutación del campo final (rng.permutation), aplica solo a X (E y
S_ent son invariantes a permutación espacial por construcción — declarado en
el pre-registro §4, no es una omisión).
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

# --- Import SIN MODIFICAR del código base (no se copia a mano) ---
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
MARGEN_MUERTE = 5.0  # margen sobre lavado medido (mucho mayor que el 1.15 del código base; ver §6 protocolo)
FLOOR_ENTROPIA = 1e-9

FAMILIAS = [
    "multi_modo",
    "modo_unico",
    "bulto_gaussiano",
    "ruido_blanco",
    "ruido_rojo",
    "ruido_azul",
]

EPS_LIST = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1]
EPS_REPRESENTATIVO_CALIBRACION = 1e-2


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
    nk = N // 2 + 1
    k = np.arange(nk, dtype=float)
    amp = np.zeros(nk, dtype=float)
    amp[1:] = k[1:] ** alpha
    fase = rng.uniform(0, 2 * np.pi, size=nk)
    espectro = amp * np.exp(1j * fase)
    señal = np.fft.irfft(espectro, n=N)
    return señal


def campo_familia(N, eps, rng, familia, m_fijo=None, sigma_fijo=None):
    """Reimplementación literal de F1_4_motor.py::campo_familia (verificada
    fórmula por fórmula contra el original congelado en F1-4).

    m_fijo/sigma_fijo: SOLO usados por la calibración de peor-caso (§ addendum
    post-smoke del pre-registro) para medir determinísticamente el caso de
    decaimiento más lento conocido de una familia paramétrica (m=1 para
    modo_unico, sigma=0.08 para bulto_gaussiano). Los barridos reales
    (grid de producción) NUNCA pasan estos argumentos — siguen usando el
    sorteo aleatorio por semilla tal como se congeló en el pre-registro §3.
    """
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    fondo = np.ones(N, dtype=float)
    if eps <= 0.0:
        return fondo, x

    if familia == "multi_modo":
        pert = np.zeros(N, dtype=float)
        for m in range(1, 6):
            fase = rng.uniform(0, 2 * np.pi)
            pert += np.sin(2 * np.pi * m * x + fase) / m
    elif familia == "modo_unico":
        m = int(m_fijo) if m_fijo is not None else int(rng.integers(1, 9))
        fase = rng.uniform(0, 2 * np.pi)
        pert = np.sin(2 * np.pi * m * x + fase)
    elif familia == "bulto_gaussiano":
        x0 = rng.uniform(0.0, 1.0)
        sigma = float(sigma_fijo) if sigma_fijo is not None else rng.uniform(0.02, 0.08)
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


def entropia_gibbs(phi):
    """S_ent = H(p)/log(N), p_i = max(phi_i,piso)/sum(...). Ver pre-registro §4."""
    N = phi.size
    clipped = np.clip(phi, FLOOR_ENTROPIA, None)
    frac_clip = float(np.mean(phi < FLOOR_ENTROPIA))
    s = clipped.sum()
    if s <= 0:
        return 0.0, frac_clip
    p = clipped / s
    H = float(-np.sum(p * np.log(p)))
    Hmax = float(np.log(N))
    return (H / Hmax if Hmax > 0 else 0.0), frac_clip


def energia_frac(phi_final, phi_inicial):
    m0 = float(phi_inicial.mean())
    if abs(m0) < 1e-15:
        return float("nan")
    return float(phi_final.mean() / m0)


def evolucionar(phi, activo, pasos, rng, null=False):
    contraste0 = float(phi.std())
    for _ in range(pasos):
        phi = paso_difusion(phi, activo)
        # r=0 fijo (H=0): no se llama paso_expansion (equivalente a H=0, pero
        # se evita el costo de generar aleatorios de corte innecesariamente)
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


def medir_pasos_lavado(N, eps, semillas, familia, P_thr=P_LAVADO, max_steps=50000, check_every=50,
                       m_fijo=None, sigma_fijo=None):
    tiempos = []
    for s in range(semillas):
        rng = np.random.default_rng(30_000 + s)
        phi, _ = campo_familia(N, eps, rng, familia, m_fijo=m_fijo, sigma_fijo=sigma_fijo)
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
    return {
        "tiempos": tiempos,
        "mediana": med,
        "P_thr": P_thr,
        "max_steps": max_steps,
        "lavo_todas": all(t < max_steps for t in tiempos),
    }


def corrida(N, eps, pasos, seed, familia, null=False, m_fijo=None, sigma_fijo=None):
    rng = np.random.default_rng(seed)
    phi0, _ = campo_familia(N, eps, rng, familia, m_fijo=m_fijo, sigma_fijo=sigma_fijo)
    activo = np.ones(N, dtype=bool)
    phi_f, activo, c0 = evolucionar(phi0.copy(), activo, pasos, rng, null=null)
    X = persistencia(phi_f, c0)
    S_ent, frac_clip = entropia_gibbs(phi_f)
    E_frac = energia_frac(phi_f, phi0)
    S_ent_ini, _ = entropia_gibbs(phi0)
    return {
        "X": X,
        "S_ent": S_ent,
        "S_ent_ini": S_ent_ini,
        "E_frac": E_frac,
        "frac_clip_entropia": frac_clip,
        "std_ratio": float(phi_f.std() / c0) if c0 > 0 else 0.0,
    }


def barrido_familia(N, eps_list, semillas, familia, pasos_muerte, log):
    filas = []
    for eps in eps_list:
        Xr, Xn, Sr, Er, srr, srn, fclip = [], [], [], [], [], [], []
        for s in range(semillas):
            seed = 1000 + s
            rr = corrida(N, eps, pasos_muerte, seed=seed, familia=familia, null=False)
            nn = corrida(N, eps, pasos_muerte, seed=seed, familia=familia, null=True)
            Xr.append(rr["X"])
            Xn.append(nn["X"])
            Sr.append(rr["S_ent"])
            Er.append(rr["E_frac"])
            srr.append(rr["std_ratio"])
            srn.append(nn["std_ratio"])
            fclip.append(rr["frac_clip_entropia"])
        Xr = np.array(Xr)
        Xn = np.array(Xn)
        Sr = np.array(Sr)
        Er = np.array(Er)
        sd = np.sqrt((Xr.var() + Xn.var()) / 2.0)
        sd = max(sd, 1.0 / max(len(Xr), 1))
        z = float((Xr.mean() - Xn.mean()) / sd)
        filas.append(
            {
                "familia": familia,
                "eps": eps,
                "pasos": pasos_muerte,
                "X_real_mean": float(Xr.mean()),
                "X_real_std": float(Xr.std()),
                "X_real_semillas": [round(float(v), 8) for v in Xr],
                "X_null_mean": float(Xn.mean()),
                "X_null_std": float(Xn.std()),
                "z_X": round(z, 3),
                "S_ent_mean": float(Sr.mean()),
                "S_ent_std": float(Sr.std()),
                "S_ent_semillas": [round(float(v), 8) for v in Sr],
                "E_frac_mean": float(Er.mean()),
                "E_frac_std": float(Er.std()),
                "E_frac_semillas": [round(float(v), 8) for v in Er],
                "E_frac_deriva_abs_max": float(np.max(np.abs(Er - 1.0))),
                "std_ratio_real_mean": float(np.mean(srr)),
                "std_ratio_null_mean": float(np.mean(srn)),
                "frac_clip_entropia_max": float(np.max(fclip)),
            }
        )
        log(f"  [familia={familia}] eps={eps:.3e} X_real={Xr.mean():.6f} "
            f"S_ent={Sr.mean():.6f} E_frac={Er.mean():.6f} z_X={z:.2f}")
    return filas


def chequeo_plateau(N, familia, pasos_muerte, log, semillas_chk=3, eps_chk=EPS_REPRESENTATIVO_CALIBRACION):
    """Verificación INDEPENDIENTE final (con semillas del GRID, 1000+s) de que
    pasos_muerte (ya elegido por calibrar_pasos_muerte_adaptativo) es estable:
    compara pasos_muerte vs 2x pasos_muerte (pre-registro §7)."""
    filas = []
    for s in range(semillas_chk):
        seed = 1000 + s
        r1 = corrida(N, eps_chk, pasos_muerte, seed=seed, familia=familia, null=False)
        r2 = corrida(N, eps_chk, 2 * pasos_muerte, seed=seed, familia=familia, null=False)
        filas.append(
            {
                "seed": seed,
                "X_1x": r1["X"], "X_2x": r2["X"],
                "S_ent_1x": r1["S_ent"], "S_ent_2x": r2["S_ent"],
                "E_frac_1x": r1["E_frac"], "E_frac_2x": r2["E_frac"],
            }
        )
    dX = np.mean([abs(f["X_2x"] - f["X_1x"]) for f in filas])
    dS = np.mean([abs(f["S_ent_2x"] - f["S_ent_1x"]) for f in filas])
    log(f"  [plateau-final {familia}] eps={eps_chk} dX_medio={dX:.6f} dS_ent_medio={dS:.6f}")
    return {"eps_chk": eps_chk, "pasos_muerte": pasos_muerte, "filas": filas,
            "dX_medio_abs": float(dX), "dS_ent_medio_abs": float(dS)}


# Addendum #2 post-smoke (declarado, PRE-producción): reemplaza la calibración
# por umbral-de-lavado+margen-fijo por una calibración ADAPTATIVA de plateau,
# aplicada UNIFORMEMENTE a las 6 familias (sin excepciones por familia). Nace
# de una segunda observación del smoke test: ruido_blanco (y por extensión
# ruido_azul) también mostraban un residuo de decaimiento lento entre
# pasos_muerte y 2×pasos_muerte (dX_medio=0.0075 sobre X=0.0154, ~50%
# relativo) pese a que su umbral P<0.05 se cruzaba rápido — el motivo físico
# es que TODO sorteo de ruido_blanco/azul contiene el modo k=1 (misma
# amplitud, solo fase aleatoria en blanco; amplitud reducida pero presente en
# azul), que decae lento igual que el m=1 de modo_unico — pero a diferencia
# de modo_unico esto NO es un problema de "calibración con mala suerte de
# semilla" (la amplitud de k=1 no es aleatoria, solo la fase, así que
# cualquier conjunto de semillas de calibración ya lo captura) — es que el
# criterio "cruzar P<0.05" simplemente no mide profundidad de plateau.
# Se sustituye entonces el criterio por: duplicar pasos hasta que el cambio
# absoluto en X y en S_ent entre P y 2P sea < TOL_PLATEAU, usando semillas de
# calibración dedicadas MÁS (para las familias paramétricas) el caso peor
# determinista ya declarado en el Addendum #1 — así ninguna familia se calibra
# con menos rigor que otra. No es un ajuste hacia un valor esperado (T1): el
# criterio de parada es el mismo número (TOL_PLATEAU) para las 6 familias.
TOL_PLATEAU = 0.01


def calibrar_pasos_muerte_adaptativo(N, familia, log, eps=EPS_REPRESENTATIVO_CALIBRACION,
                                      semillas_chk=4, pasos_inicial=200, max_pasos=200000,
                                      tol=TOL_PLATEAU, peor_caso_kwargs=None):
    """Duplica `pasos` hasta que |X(2P)-X(P)| y |S_ent(2P)-S_ent(P)| < tol,
    tomando el MÁXIMO (peor caso) entre varias semillas de calibración
    aleatorias MÁS (si aplica) una réplica determinista del peor caso físico
    conocido de la familia (m_fijo=1 / sigma_fijo=0.08). Devuelve
    (pasos_muerte=2*P_convergido, historial completo para auditoría)."""
    seeds_calib = [40_000 + i for i in range(semillas_chk)]
    reps = [{"seed": s, "kwargs": {}} for s in seeds_calib]
    if peor_caso_kwargs:
        reps.append({"seed": 41_000, "kwargs": peor_caso_kwargs})

    pasos = pasos_inicial
    historia = []
    while True:
        dXs, dSs = [], []
        for rep in reps:
            r1 = corrida(N, eps, pasos, seed=rep["seed"], familia=familia, null=False,
                         **rep["kwargs"])
            r2 = corrida(N, eps, 2 * pasos, seed=rep["seed"], familia=familia, null=False,
                         **rep["kwargs"])
            dXs.append(abs(r2["X"] - r1["X"]))
            dSs.append(abs(r2["S_ent"] - r1["S_ent"]))
        dX_peor = float(np.max(dXs))
        dS_peor = float(np.max(dSs))
        historia.append({"pasos": pasos, "dX_peor": dX_peor, "dS_ent_peor": dS_peor})
        log(f"    [calib-adapt {familia}] pasos={pasos} dX_peor={dX_peor:.6f} dS_ent_peor={dS_peor:.6f}")
        if dX_peor < tol and dS_peor < tol:
            return 2 * pasos, historia
        pasos *= 2
        if pasos > max_pasos:
            log(f"    [calib-adapt {familia}] TOPE max_pasos={max_pasos} sin plateau claro — se usa igual, reportado")
            return pasos, historia


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    t0 = time.time()
    OUT_RES.mkdir(parents=True, exist_ok=True)

    log_path = OUT_RES / "E5_5_5_log_ejecucion.txt"
    log_f = open(log_path, "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, file=sys.stderr, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    log(f"=== E5_5_5 motor arrancando modo={modo} ===")

    if modo == "smoke":
        N = 100
        semillas = 4
        eps_list = [0.0, 1e-4, 1e-2, 3e-1]
        familias = FAMILIAS
        max_steps_cal = 20000
    elif modo == "produccion":
        N = 200
        semillas = 12
        eps_list = EPS_LIST
        familias = FAMILIAS
        max_steps_cal = 50000
    else:
        raise SystemExit(f"modo desconocido: {modo} (usa smoke|produccion)")

    log(f"N={N} semillas={semillas} eps_list={eps_list} familias={familias} r=0(fijo)")

    # Peor caso físico conocido y determinista por familia paramétrica
    # (Addendum #1) — se pasa como réplica extra a la calibración adaptativa
    # (Addendum #2, ver docstring de calibrar_pasos_muerte_adaptativo).
    PEOR_CASO = {
        "modo_unico": {"m_fijo": 1},
        "bulto_gaussiano": {"sigma_fijo": 0.08},
    }

    resultado_por_familia = {}
    calibraciones = {}
    for familia in familias:
        t_fam = time.time()
        log(f"[familia={familia}] calibracion adaptativa de plateau (tol={TOL_PLATEAU}) en eps={EPS_REPRESENTATIVO_CALIBRACION}")
        pasos_muerte, historia_calib = calibrar_pasos_muerte_adaptativo(
            N, familia, log, eps=EPS_REPRESENTATIVO_CALIBRACION,
            max_pasos=max_steps_cal, peor_caso_kwargs=PEOR_CASO.get(familia),
        )
        # Diagnóstico descriptivo adicional (no determina pasos_muerte): tiempo
        # de cruce de P<0.05, útil para comparar con el resto de la batería.
        cal_lavado = medir_pasos_lavado(
            N, EPS_REPRESENTATIVO_CALIBRACION, min(semillas, 8), familia,
            max_steps=max_steps_cal,
        )
        calibraciones[familia] = {
            "historia_calibracion_adaptativa": historia_calib,
            "pasos_muerte": pasos_muerte,
            "tol_plateau": TOL_PLATEAU,
            "diagnostico_lavado_P005": cal_lavado,
        }
        log(f"[familia={familia}] -> pasos_muerte={pasos_muerte} (adaptativo, tol={TOL_PLATEAU}; "
            f"diagnóstico lavado P<0.05 mediana={cal_lavado['mediana']})")

        filas = barrido_familia(N, eps_list, semillas, familia, pasos_muerte, log)
        plateau = chequeo_plateau(N, familia, pasos_muerte, log)

        resultado_por_familia[familia] = {
            "familia": familia,
            "calibracion_lavado": calibraciones[familia],
            "pasos_muerte": pasos_muerte,
            "filas": filas,
            "chequeo_plateau": plateau,
            "elapsed_s": time.time() - t_fam,
        }
        log(f"[familia={familia}] listo en {time.time()-t_fam:.1f}s")

    result = {
        "experimento": "E5.5-5",
        "modo": modo,
        "N": N,
        "semillas": semillas,
        "eps_list": eps_list,
        "familias": familias,
        "r_fijo": 0.0,
        "margen_muerte": MARGEN_MUERTE,
        "eps_representativo_calibracion": EPS_REPRESENTATIVO_CALIBRACION,
        "por_familia": resultado_por_familia,
        "elapsed_s": time.time() - t0,
        "timestamp_inicio": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
        "timestamp_fin": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_json = OUT_RES / f"E5_5_5_{modo}_resultado.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[archivo] {out_json}")
    log(f"[elapsed] {result['elapsed_s']:.1f}s")
    log("=== E5_5_5 motor terminado ===")
    log_f.close()


if __name__ == "__main__":
    main()
