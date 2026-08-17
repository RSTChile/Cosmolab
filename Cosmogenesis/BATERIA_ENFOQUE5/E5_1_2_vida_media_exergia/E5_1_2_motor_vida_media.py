#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5_1_2_motor_vida_media.py — "Vida media de la exergía: ¿cuántos pasos tarda X en
decaer sin expansión?" (Enfoque 5, TEMA 1, experimento E5.1-2)

QUIÉN SOY: motor de producción de UN experimento de la batería de 30. Mide τ(D) — pasos
hasta que la exergía (proxy: X = var(φ_t)/var(φ_0)) cae a la mitad — bajo difusión PURA
(sin expansión, H=0 siempre) del campo φ de cs074_rcruz.py.

QUÉ HAGO:
  - Importo (NO edito) cs074_rcruz.py: reuso paso_difusion, campo_inicial, medir_D tal
    cual están escritos. La física es 100% la del código base; este archivo solo la
    orquesta y mide τ.
  - D no es un parámetro directo del código base (solo N lo es; D se MIDE). Por eso barro
    N (log-espaciado) y reporto el D medido resultante — nunca invento D a mano.
  - Tier 1: curva primaria τ(D), eps fijo, 16 semillas (cumple el piso del documento madre).
  - Tier 2: invariancia a ε en 5 anclas de N, 8 semillas.
  - Tier 3: extensión de D hacia abajo (N grande, caro), 6 semillas.
  - Tier 4: perturbación dinámica (ruido gaussiano aditivo por paso), requerida por T7.
  - Todo corre con semilla de numpy propia (no comparte RNG global); guarda JSON incremental
    por tier para que un corte de cómputo no pierda todo el trabajo previo.

QUÉ NO HAGO: no edito cs074_rcruz.py, no impongo NULL (el documento madre dice NULL:"—" para
este experimento — es caracterización pura), no fuerzo τ=k/D si los datos no lo sostienen.

Ver PROTOCOLO_E5.1-2_PREREGISTRO.md (mismo directorio) para el diseño completo, congelado
ANTES de correr este motor.

--- ADENDA 2026-07-25 (re-corrida, ver PROTOCOLO §"ADENDA") ---
Además del tau histórico (var_ratio), este motor mide EN PARALELO (misma trayectoria, sin
duplicar la simulación) tau_canonico sobre Xh(t)=exergia_X(phi_t)/exergia_X(phi_0), con
exergia_X importada tal cual de BATERIA_ENFOQUE5/_observables_homologadas.py (Arreglo 3,
definición común de la batería). Tier 4 usa ruido_por_paso() de
BATERIA_ENFOQUE5/_ruido_calibrado.py en vez de sigma_ruido=frac*eps constante (Arreglo 2).
Tier 2 y Tier 3 reproducen EXACTAMENTE las combinaciones que terminaron guardadas en la
corrida histórica del 2026-07-24 (truncada por SIGTERM/costo, ver
E5_1_2_motor_extension.py), no el diseño preregistrado completo nunca terminado — así la
comparación lado-a-lado es honesta (mismo número de combinaciones). Detalle completo en la
ADENDA del protocolo.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent.parent  # Cosmogenesis/
sys.path.insert(0, str(BASE_DIR))

from cs074_rcruz import paso_difusion, campo_inicial, medir_D  # noqa: E402  (código base, NO editado)
from BATERIA_ENFOQUE5._observables_homologadas import exergia_X  # noqa: E402  (Arreglo 3, definición común)
from BATERIA_ENFOQUE5._ruido_calibrado import ruido_por_paso  # noqa: E402  (Arreglo 2, ruido calibrado)

OUT_DIR = HERE
LOG = OUT_DIR / "E5_1_2_log_ejecucion.txt"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def var_ratio(phi, var0):
    if var0 <= 0:
        return 0.0
    return float(phi.var() / var0)


def std_corr(phi):
    """Segundo observable (verificación cruzada, regla 4): correlación espacial c,
    el mismo componente que usa persistencia() en cs074_rcruz.py."""
    if phi.std() <= 1e-12:
        return 0.0
    c = np.corrcoef(phi, np.roll(phi, 1))[0, 1]
    if not np.isfinite(c):
        c = 0.0
    return max(0.0, float(c))


def correr_tau(N, eps, seed, max_steps, check_every, sigma_ruido=0.0):
    """
    Difusión pura (sin expansión) hasta max_steps, midiendo X(t)=var(t)/var(0) cada
    check_every pasos. sigma_ruido>0 agrega ruido gaussiano dinámico aditivo por paso
    (Tier 4, requerido por T7 — perturbación dinámica además de semilla).
    Devuelve dict con tau (censurado o no), curva muestreada X(t), y el 2do observable c(t).

    ADENDA 2026-07-25 (Arreglo 3): EN PARALELO, sobre la MISMA trayectoria (no se duplica
    la simulación), se mide tau_canonico sobre Xh(t)=exergia_X(phi_t)/exergia_X(phi_0), con
    exergia_X de _observables_homologadas.py (definición común de la batería). Mismo
    criterio de cruce (primer t con Xh<=0.5), mismo max_steps/censura.
    """
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)  # H=0 permanente: nunca se cortan aristas (sin expansión)
    var0 = float(phi.var())
    Xh0 = exergia_X(phi)
    if var0 <= 0:
        return {
            "tau": 0, "censurado": False, "var0": 0.0,
            "curva_t": [0], "curva_X": [0.0], "curva_c": [0.0],
            "tau_canonico": 0, "censurado_canonico": False, "Xh0": Xh0,
            "curva_Xh": [0.0],
        }
    curva_t, curva_X, curva_c = [0], [1.0], [std_corr(phi)]
    curva_Xh = [1.0 if Xh0 > 0 else 0.0]
    t = 0
    tau = None
    tau_canonico = None
    while t < max_steps:
        n = min(check_every, max_steps - t)
        for _ in range(n):
            phi = paso_difusion(phi, activo)
            if sigma_ruido > 0.0:
                phi = phi + rng.normal(0.0, sigma_ruido, size=N)
        t += n
        X = var_ratio(phi, var0)
        curva_t.append(t)
        curva_X.append(X)
        curva_c.append(std_corr(phi))
        Xh_ratio = float(exergia_X(phi) / Xh0) if Xh0 > 0 else 0.0
        curva_Xh.append(Xh_ratio)
        if tau is None and X <= 0.5:
            tau = t
        if tau_canonico is None and Xh_ratio <= 0.5:
            tau_canonico = t
        if tau is not None and tau_canonico is not None:
            break
    censurado = tau is None
    censurado_canonico = tau_canonico is None
    if censurado:
        tau = max_steps
    if censurado_canonico:
        tau_canonico = max_steps
    return {
        "tau": int(tau), "censurado": bool(censurado), "var0": var0,
        "tau_canonico": int(tau_canonico), "censurado_canonico": bool(censurado_canonico),
        "Xh0": Xh0,
        "curva_t": curva_t, "curva_X": curva_X, "curva_c": curva_c, "curva_Xh": curva_Xh,
    }


def estimar_presupuesto(D, margen=3.0):
    """Usa la ley piloto tau*D~0.668 SOLO para dimensionar max_steps/check_every,
    nunca para reportar tau (eso siempre sale de correr_tau)."""
    tau_est = 0.668 / D if D > 0 else 1.0
    max_steps = int(np.ceil(margen * tau_est)) + 10
    check_every = max(1, min(50, max_steps // 60))
    return max_steps, check_every


def medir_D_prom(N, eps, semillas=4):
    return float(np.mean([medir_D(N, eps, s) for s in range(semillas)]))


def tier1_curva_primaria():
    log("=== TIER 1: curva primaria tau(D) — eps=1e-3, 16 semillas ===")
    N_grid = [16, 23, 32, 46, 65, 92, 130, 184, 261, 370, 524, 743, 1053, 1493, 2116, 3000]
    eps = 1e-3
    semillas = 16
    filas = []
    for N in N_grid:
        D = medir_D_prom(N, eps, semillas=4)
        max_steps, check_every = estimar_presupuesto(D)
        t0 = time.time()
        taus, censuras, ultimas_X = [], [], []
        taus_canonico, censuras_canonico, ultimas_Xh = [], [], []
        curvas_todas_semillas = []
        for s in range(semillas):
            seed_id = 20_000 + s
            r = correr_tau(N, eps, seed=seed_id, max_steps=max_steps, check_every=check_every)
            taus.append(r["tau"])
            censuras.append(r["censurado"])
            ultimas_X.append(r["curva_X"][-1])
            taus_canonico.append(r["tau_canonico"])
            censuras_canonico.append(r["censurado_canonico"])
            ultimas_Xh.append(r["curva_Xh"][-1])
            # ADENDA 2026-07-25: curva completa X(t) y Xh(t) para TODAS las semillas
            # (antes solo se guardaba curva_ejemplo_seed0).
            curvas_todas_semillas.append({
                "seed": seed_id, "t": r["curva_t"], "X": r["curva_X"],
                "Xh": r["curva_Xh"], "c": r["curva_c"],
            })
        dt = time.time() - t0
        fila = {
            "N": N, "D": D, "eps": eps, "semillas": semillas,
            "max_steps": max_steps, "check_every": check_every,
            "tau_media": float(np.mean(taus)), "tau_mediana": float(np.median(taus)),
            "tau_std": float(np.std(taus)), "tau_min": int(np.min(taus)), "tau_max": int(np.max(taus)),
            "taus_todas": [int(x) for x in taus],
            "n_censurados": int(sum(censuras)),
            "X_final_media": float(np.mean(ultimas_X)),
            "tau_canonico_media": float(np.mean(taus_canonico)),
            "tau_canonico_mediana": float(np.median(taus_canonico)),
            "tau_canonico_std": float(np.std(taus_canonico)),
            "tau_canonico_min": int(np.min(taus_canonico)), "tau_canonico_max": int(np.max(taus_canonico)),
            "taus_canonico_todas": [int(x) for x in taus_canonico],
            "n_censurados_canonico": int(sum(censuras_canonico)),
            "Xh_final_media": float(np.mean(ultimas_Xh)),
            "curvas_todas_semillas": curvas_todas_semillas,
            "wall_s": dt,
        }
        filas.append(fila)
        log(f"  N={N:>6d} D={D:.4e} tau_med={fila['tau_media']:.1f}±{fila['tau_std']:.1f} "
            f"tau_canon_med={fila['tau_canonico_media']:.1f}±{fila['tau_canonico_std']:.1f} "
            f"censurados={fila['n_censurados']}/{semillas} wall={dt:.1f}s")
        _guardar_parcial("tier1", filas)
    return filas


def tier2_invariancia_eps():
    """ADENDA 2026-07-25: reproduce EXACTAMENTE las combinaciones que quedaron guardadas en
    la corrida histórica del 2026-07-24 (truncada por SIGTERM/costo en N=1493/eps=1e-9,
    ANTES de llegar a N=3000) — no el diseño preregistrado completo, nunca terminado. Ver
    PROTOCOLO §ADENDA. N=16,130,524: 8 eps completos. N=1493: solo {0.0, 1e-9} (los 2 puntos
    que alcanzaron a correr). N=3000: ausente (igual que en el histórico)."""
    log("=== TIER 2: invariancia a eps — replica exacta de combos historicos (SIGTERM 2026-07-24) ===")
    eps_list_completo = [0.0, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.5, 1.0]
    combos_por_N = {
        16: eps_list_completo,
        130: eps_list_completo,
        524: eps_list_completo,
        1493: [0.0, 1e-9],  # solo lo que terminó antes del SIGTERM histórico
        # N=3000: ausente, igual que en el histórico (nunca llegó a correr)
    }
    semillas = 8
    filas = []
    for N, eps_list in combos_por_N.items():
        for eps in eps_list:
            if eps <= 0.0:
                filas.append({
                    "N": N, "eps": eps, "D": None, "semillas": semillas,
                    "tau_media": 0.0, "tau_std": 0.0, "nota": "eps=0: var0=0, X trivial (sin exergia que decaiga)",
                })
                continue
            D = medir_D_prom(N, eps, semillas=4)
            max_steps, check_every = estimar_presupuesto(D)
            t0 = time.time()
            taus, censuras, taus_canonico, censuras_canonico = [], [], [], []
            for s in range(semillas):
                r = correr_tau(N, eps, seed=30_000 + s, max_steps=max_steps, check_every=check_every)
                taus.append(r["tau"])
                censuras.append(r["censurado"])
                taus_canonico.append(r["tau_canonico"])
                censuras_canonico.append(r["censurado_canonico"])
            dt = time.time() - t0
            fila = {
                "N": N, "eps": eps, "D": D, "semillas": semillas,
                "max_steps": max_steps, "check_every": check_every,
                "tau_media": float(np.mean(taus)), "tau_std": float(np.std(taus)),
                "taus_todas": [int(x) for x in taus],
                "n_censurados": int(sum(censuras)),
                "tau_canonico_media": float(np.mean(taus_canonico)),
                "tau_canonico_std": float(np.std(taus_canonico)),
                "taus_canonico_todas": [int(x) for x in taus_canonico],
                "n_censurados_canonico": int(sum(censuras_canonico)),
                "wall_s": dt,
            }
            filas.append(fila)
            log(f"  N={N:>6d} eps={eps:<8g} D={D:.4e} tau_med={fila['tau_media']:.1f}±{fila['tau_std']:.1f} "
                f"tau_canon_med={fila['tau_canonico_media']:.1f}±{fila['tau_canonico_std']:.1f} wall={dt:.1f}s")
            _guardar_parcial("tier2", filas)
    return filas


def tier3_extension_D_bajo():
    """ADENDA 2026-07-25: reproduce el TIER 3 RECORTADO que efectivamente corrió en la
    historia (E5_1_2_motor_extension.py::tier3_recortado, N=4000×4 semillas) — el diseño
    preregistrado original (N∈{4500,6000}×6 semillas) nunca se terminó de correr (costo).
    Ver PROTOCOLO §ADENDA."""
    log("=== TIER 3 (recortado, replica del historico): N=4000, 4 semillas, eps=1e-3 ===")
    N_grid = [4000]
    eps = 1e-3
    semillas = 4
    filas = []
    for N in N_grid:
        D = medir_D_prom(N, eps, semillas=4)
        max_steps, check_every = estimar_presupuesto(D)
        t0 = time.time()
        taus, censuras, taus_canonico, censuras_canonico = [], [], [], []
        for s in range(semillas):
            r = correr_tau(N, eps, seed=40_000 + s, max_steps=max_steps, check_every=check_every)
            taus.append(r["tau"])
            censuras.append(r["censurado"])
            taus_canonico.append(r["tau_canonico"])
            censuras_canonico.append(r["censurado_canonico"])
            log(f"    N={N} semilla={s} tau={r['tau']} tau_canonico={r['tau_canonico']} "
                f"censurado={r['censurado']} (parcial, {time.time()-t0:.0f}s acum)")
        dt = time.time() - t0
        fila = {
            "N": N, "D": D, "eps": eps, "semillas": semillas,
            "max_steps": max_steps, "check_every": check_every,
            "tau_media": float(np.mean(taus)), "tau_std": float(np.std(taus)),
            "taus_todas": [int(x) for x in taus],
            "n_censurados": int(sum(censuras)),
            "tau_canonico_media": float(np.mean(taus_canonico)),
            "tau_canonico_std": float(np.std(taus_canonico)),
            "taus_canonico_todas": [int(x) for x in taus_canonico],
            "n_censurados_canonico": int(sum(censuras_canonico)),
            "wall_s": dt,
            "nota": "TIER3 recortado (replica del historico); N=4500 y N=6000 originalmente disenados NO se corrieron (ni antes ni ahora)",
        }
        filas.append(fila)
        log(f"  N={N:>6d} D={D:.4e} tau_med={fila['tau_media']:.1f}±{fila['tau_std']:.1f} "
            f"tau_canon_med={fila['tau_canonico_media']:.1f}±{fila['tau_canonico_std']:.1f} "
            f"censurados={fila['n_censurados']}/{semillas} wall={dt:.1f}s")
        _guardar_parcial("tier3", filas)
    return filas


def tier4_perturbacion_dinamica():
    """ADENDA 2026-07-25: (a) reproduce las anclas N∈{16,130,524} que efectivamente corrieron
    en la historia (E5_1_2_motor_extension.py::tier4_completo excluyó N=3000 por costo — el
    diseño preregistrado original tenía N∈{16,524,3000}, nunca se corrió con N=3000). (b)
    Aplica el Arreglo 2 (ruido calibrado): sigma_ruido = ruido_por_paso(frac, eps, max_steps)
    en vez de frac*eps constante — evita que la varianza acumulada del ruido crezca sin tope
    con max_steps (que aquí varía ~3600x entre N=16 y N=524). Ver PROTOCOLO §ADENDA."""
    log("=== TIER 4: perturbacion dinamica (T7) — replica de anclas historicas N={16,130,524}, "
        "2 niveles ruido, 8 semillas, ruido calibrado (Arreglo 2) ===")
    N_anclas = [16, 130, 524]
    eps = 1e-3
    fracs = [0.01, 0.1]
    semillas = 8
    filas = []
    for N in N_anclas:
        D = medir_D_prom(N, eps, semillas=4)
        max_steps, check_every = estimar_presupuesto(D, margen=4.0)  # margen extra: el ruido puede retrasar el cruce
        for frac in fracs:
            # Arreglo 2: amplitud de ruido POR PASO calibrada para que la varianza acumulada
            # total sobre max_steps pasos sea ~(frac*eps)^2, independiente de N/max_steps.
            # frac (0.01, 0.1) NO cambia de valor ni de significado -- solo como se reparte
            # en el tiempo. Antes: sigma = frac*eps (constante, sin tope con N).
            sigma = ruido_por_paso(frac, eps, max_steps)
            t0 = time.time()
            taus, censuras, taus_canonico, censuras_canonico = [], [], [], []
            for s in range(semillas):
                r = correr_tau(N, eps, seed=50_000 + s, max_steps=max_steps, check_every=check_every, sigma_ruido=sigma)
                taus.append(r["tau"])
                censuras.append(r["censurado"])
                taus_canonico.append(r["tau_canonico"])
                censuras_canonico.append(r["censurado_canonico"])
            dt = time.time() - t0
            fila = {
                "N": N, "D": D, "eps": eps, "sigma_ruido": sigma, "frac_ruido": frac,
                "semillas": semillas, "max_steps": max_steps, "check_every": check_every,
                "tau_media": float(np.mean(taus)), "tau_std": float(np.std(taus)),
                "taus_todas": [int(x) for x in taus],
                "n_censurados": int(sum(censuras)),
                "tau_canonico_media": float(np.mean(taus_canonico)),
                "tau_canonico_std": float(np.std(taus_canonico)),
                "taus_canonico_todas": [int(x) for x in taus_canonico],
                "n_censurados_canonico": int(sum(censuras_canonico)),
                "wall_s": dt,
            }
            filas.append(fila)
            log(f"  N={N:>6d} D={D:.4e} frac_ruido={frac} sigma={sigma:.3e} tau_med={fila['tau_media']:.1f}±{fila['tau_std']:.1f} "
                f"tau_canon_med={fila['tau_canonico_media']:.1f}±{fila['tau_canonico_std']:.1f} wall={dt:.1f}s")
            _guardar_parcial("tier4", filas)
    return filas


def _guardar_parcial(nombre, filas):
    path = OUT_DIR / f"E5_1_2_resultado_{nombre}_PARCIAL.json"
    path.write_text(json.dumps(filas, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    t_inicio_total = time.time()
    log("###### INICIO E5.1-2 vida media de la exergia ######")

    resultado = {
        "experimento": "E5.1-2",
        "nombre": "Vida media de la exergia: cuantos pasos tarda X en decaer sin expansion",
        "base_code": "cs074_rcruz.py (importado, no editado)",
        "timestamp_inicio": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    resultado["tier1_curva_primaria"] = tier1_curva_primaria()
    resultado["tier2_invariancia_eps"] = tier2_invariancia_eps()
    resultado["tier3_extension_D_bajo"] = tier3_extension_D_bajo()
    resultado["tier4_perturbacion_dinamica"] = tier4_perturbacion_dinamica()

    resultado["timestamp_fin"] = time.strftime("%Y-%m-%d %H:%M:%S")
    resultado["elapsed_s_total"] = time.time() - t_inicio_total

    out_path = OUT_DIR / "E5_1_2_resultado_FINAL.json"
    out_path.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"###### FIN. elapsed={resultado['elapsed_s_total']:.1f}s archivo={out_path} ######")


if __name__ == "__main__":
    main()
