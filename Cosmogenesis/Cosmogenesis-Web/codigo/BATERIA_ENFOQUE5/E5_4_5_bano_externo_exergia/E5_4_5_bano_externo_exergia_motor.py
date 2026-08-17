#!/usr/bin/env python3
"""
E5_4_5_bano_externo_exergia_motor.py — BATERIA_ENFOQUE5, TEMA 4, experimento E5.4-5

"Control negativo: enfriamiento con baño externo (lo prohibido) vs adiabático"

Todo el TEMA 4 (E5.4-1..E5.4-4) afirma que la exergía X aparece por enfriamiento
ADIABÁTICO vía expansión — solo por re-escalamiento geométrico del campo por `a`
y por la dilución de la difusividad (D=D0/a³), SIN ningún término de
enfriamiento impuesto a mano. Esa afirmación solo es creíble si el arnés
experimental sabe distinguir ese mecanismo de la alternativa prohibida: que el
modelo estuviera, sin darnos cuenta, simulando un baño térmico externo
(acoplamiento a un reservorio a temperatura fija).

Este motor mete ESE baño A PROPÓSITO, como control negativo: añade un término
de relajación tipo Newton hacia una temperatura de reservorio fija T_BANO, con
intensidad kappa barrida desde 0 (= adiabático puro, reduce EXACTAMENTE al caso
CF-2/F3-6) hasta un valor "fuerte" (kappa_max = 1/dt_sub, acople perfecto en un
subpaso). Predicción pre-registrada (PROTOCOLO_E5.4-5_PREREGISTRO.md):

    kappa=0  (adiabático) : T_fis(a) ~ a^(-n)   (ley de potencia, n~1)
                             X_fis(a) ~ a^(-m)   (ley de potencia, m~2, "freeze-out")
    kappa>0  (con baño)   : T_fis(a) -> T_BANO   (se APLANA, NO ley de potencia)
                             X_fis(a) -> colapsa MÁS RÁPIDO que en el adiabático
                             (el baño no solo mueve el nivel medio, BORRA la
                             estructura espacial que sostiene la capacidad de
                             hacer trabajo)

Firma cualitativa DISTINTA por observable: T se aplana, X se derrumba. Si ambos
casos NO difieren claramente en T Y en X, es una alerta seria sobre el arnés de
todo el TEMA 4 (se reporta como hallazgo, no se esconde).

Metodología reutilizada de (ambos leídos completos, NO editados):
  - Cosmogenesis-Web/codigo/CF2_estiramiento/CF2_estiramiento_motor.py
    (sello físico: L, H_EXP, RHO0, D0, W0, DT, N_SUB, reloj genético,
    laplaciano de 5 puntos, salto tanh, banda central anti-wrap,
    observable de gradiente grad_fis=grad_comov/a)
  - Cosmogenesis-Web/codigo/BATERIA_FUNDAMENTOS/F3_6_bano_externo/F3_6_bano_externo_motor.py
    (término de baño: relajación tipo Newton hacia objetivo comóvil a*T_BANO,
    T_fis(a)=mean(T_comov)/a, sin clip[0,1] en ningún brazo)

Ingrediente NUEVO propio de E5.4-5 (no está en F3-6): el observable de
EXERGÍA, X_fis(a) = Var_espacial(T_comov)/a² — ver justificación completa en
el pre-registro §4 (energía libre de fluctuación ~ (ΔT)², convención
físico=comóvil/a ya usada por CF2/F3-6, aplicada a varianza -> /a²). También se
agrega un tercer observable informativo, E_comov_sum(a)=sum(T_comov), para
verificar el axioma E1 de la batería (conservación): se espera drift≈0 en
kappa=0 y drift creciente con kappa (firma física de "baño=violación de E1").

No se auto-adjudica el hallazgo más amplio del TEMA 4. Entrega números crudos;
la adjudicación es de CS. No topología. No commits.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ============================================================
# Sello físico heredado de CF2_estiramiento_motor.py / F3_6_bano_externo_motor.py,
# IDÉNTICO (T1: no se retoca nada del núcleo para favorecer un resultado)
# ============================================================
L = 64
H_EXP = 6.0
RHO0 = 1.0
D0 = 0.12
W0 = 1.2
DT = 0.25
N_SUB = 2
ORIGINAL_STEPS_PER_TG = 399

# ============================================================
# Ingrediente heredado de F3-6: baño térmico externo (PROTOCOLO §3)
# ============================================================
T_BANO = 0.5  # temperatura física fija del reservorio (idéntico a F3-6, T1: reutilizado)

# ============================================================
# Barrido pre-registrado (PROTOCOLO_E5.4-5_PREREGISTRO.md §5), sobredimensionado
# respecto a F3-6: kappa_max 8x mayor (acople perfecto en 1 subpaso), a hasta 1e4
# (4 décadas, vs 3 de F3-6), en línea con el rango de E5.4-1 de esta misma spec.
# ============================================================
DT_SUB = DT / N_SUB  # 0.125
KAPPA_MAX = 1.0 / DT_SUB  # 8.0 -> kappa*dt_sub=1.0 -> factor de contracción = 0 (salto exacto)
A_GRID = np.geomspace(1.0, 1.0e4, 9)  # 9 puntos, 4 décadas
KAPPA_GRID = np.concatenate([[0.0], np.geomspace(1.0e-3, KAPPA_MAX, 7)])  # 8 puntos, incluye 0
SEEDS_STANDARD_PROJECT = [7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321]
SEEDS_NUEVAS_E5_4_5 = [4242, 161803]
SEEDS = SEEDS_STANDARD_PROJECT + SEEDS_NUEVAS_E5_4_5  # 12 semillas (>=12 exigidas)

# ============================================================
# Criterio de PASS pre-registrado (PROTOCOLO_E5.4-5_PREREGISTRO.md §6)
# ============================================================
SLOPE_T_ADIABATICO_MAX = -0.5    # slope_T(kappa=0) debe ser <= esto
SLOPE_T_FLATTENING_MIN = 0.3     # slope_T(kappa_max) - slope_T(kappa=0) >= esto (T se aplana)
SLOPE_X_ADIABATICO_MAX = -1.0    # slope_X(kappa=0) debe ser <= esto (freeze-out ~a^-2 esperado)
SLOPE_X_STEEPENING_MIN = 0.3     # slope_X(kappa=0) - slope_X(kappa_max) >= esto (X se derrumba)
X_RATIO_BANO_ADIABATICO_MAX = 0.5  # valor_final_X(kappa_max) <= esto * valor_final_X(kappa=0)
PASS_RATE_MIN = 0.55             # idéntico a CF-2/F3-1/F3-6, no re-elegido (T1)

CODE_DIR = Path(__file__).resolve().parent
WEB_ROOT = CODE_DIR.parents[2]  # .../Cosmogenesis-Web
OUT_DIR = WEB_ROOT / "results" / "BATERIA_ENFOQUE5" / "E5_4_5_bano_externo_exergia"


def initial_T(L: int, w0: float) -> np.ndarray:
    """Salto abrupto vertical: T≈1 a la izquierda, T≈0 a la derecha (frente plano en y).
    Idéntico a CF2_estiramiento_motor.py / F3_6_bano_externo_motor.py."""
    x = np.arange(L) - (L - 1) / 2.0
    profile = 0.5 * (1.0 - np.tanh(x / w0))
    return np.tile(profile, (L, 1))


def grad_metrics(T: np.ndarray, a: float) -> dict:
    """Abruptancia comóvil y física, banda central (evita wrap-around periódico).
    Idéntico a CF2/F3-6 (observable secundario de E5.4-5, PROTOCOLO §4)."""
    dTx = 0.5 * (np.roll(T, -1, axis=1) - np.roll(T, 1, axis=1))
    n = T.shape[1]
    band = slice(n // 8, 7 * n // 8)
    g = np.abs(dTx[:, band])
    A_comov = float(g.max()) if g.size else 0.0
    A_phys = A_comov / max(a, 1e-12)
    return {"A_comov": A_comov, "A_phys": A_phys}


def field_temperature(T: np.ndarray, a: float) -> dict:
    """Observable heredado de F3-6 (PROTOCOLO §4): temperatura física del campo,
    T_fis(a) = mean(T_comov) / a — media espacial GLOBAL, análoga a la dilución
    cosmológica T∝1/a de un gas sin interacción."""
    T_mean_comov = float(np.mean(T))
    T_fis = T_mean_comov / max(a, 1e-12)
    return {"T_mean_comov": T_mean_comov, "T_fis": T_fis}


def field_exergy(T: np.ndarray, a: float) -> dict:
    """Observable NUEVO de E5.4-5 (PROTOCOLO §4): exergía física del campo,
    X_fis(a) = Var_espacial(T_comov) / a² — capacidad de hacer trabajo, medida
    como desviación cuadrática del equilibrio uniforme (energía libre de
    fluctuación ~ (ΔT)², resultado estándar de termodinámica lineal). Campo
    perfectamente uniforme (Var=0) -> X=0 -> cero exergía. Global (L×L, sin
    banda): la varianza no sufre wrap-around, a diferencia del máximo de
    gradiente."""
    var_comov = float(np.var(T))
    X_fis = var_comov / max(a * a, 1e-24)
    return {"Var_comov": var_comov, "X_fis": X_fis}


def field_energy_sum(T: np.ndarray) -> float:
    """Verificación adicional (PROTOCOLO §4c, axioma E1, informativa, no gatea):
    suma total del campo comóvil. Bajo difusión pura de 5 puntos en malla
    periódica esta cantidad se conserva EXACTAMENTE (el laplaciano de roll
    suma cero). El baño (kappa>0) la modifica activamente -> firma física de
    violación de E1."""
    return float(np.sum(T))


def diffuse_con_bano(T: np.ndarray, D: float, a: float, kappa: float,
                      dt: float, n_sub: int) -> np.ndarray:
    """Difusión con laplaciano de 5 puntos (idéntico a CF2/F3-6) + término de
    baño externo (PROTOCOLO §3): en cada subpaso,

        T_c <- T_c + dt_sub*D*lap(T_c) - dt_sub*kappa*(T_c - a*T_BANO)

    Con kappa=0 el segundo término se anula EXACTAMENTE y esto reduce
    bit-a-bit a la función `diffuse` de CF2/`diffuse_con_bano` de F3-6 en
    kappa=0. Sin clip[0,1] en ningún brazo (misma razón declarada en F3-6
    §3, heredada aquí: el objetivo comóvil a*T_BANO supera 1 al crecer a)."""
    out = T
    dt_sub = dt / n_sub
    objetivo_comov = a * T_BANO
    for _ in range(n_sub):
        if D > 0:
            lap = (
                np.roll(out, -1, 1)
                + np.roll(out, 1, 1)
                + np.roll(out, -1, 0)
                + np.roll(out, 1, 0)
                - 4.0 * out
            )
            out = out + dt_sub * D * lap
        if kappa > 0:
            out = out - dt_sub * kappa * (out - objetivo_comov)
    return out


def run_sweep(seed: int, a_grid: np.ndarray, kappa: float) -> dict:
    """Integra desde t_g=0, muestreando los observables en los checkpoints
    t_g(a)=ln(a)/H_EXP. Mismo método de checkpointing markoviano que CF2/F3-6
    (una sola trayectoria por (kappa, semilla), muestreada en los `a`
    objetivo). El brazo adiabático (kappa=0) usa D(a)=D0/a³ como en CF2/F3-6 —
    ese ingrediente NUNCA se apaga (T1), solo se le suma o no el término de
    baño."""
    rng = np.random.default_rng(seed)
    T = initial_T(L, W0)
    T = T + 1e-4 * rng.normal(size=T.shape)  # mismo ruido de condición inicial que CF2/F3-6

    dtg = 1.0 / ORIGINAL_STEPS_PER_TG
    tg_targets = np.log(a_grid) / H_EXP
    tg_max = float(tg_targets[-1])
    n_steps = max(int(np.ceil(tg_max / dtg)), 1)

    checkpoints = []
    next_ckpt_idx = 0

    def record(tg_now, a_now):
        gm = grad_metrics(T, a_now)
        ft = field_temperature(T, a_now)
        ex = field_exergy(T, a_now)
        e_sum = field_energy_sum(T)
        checkpoints.append(
            {
                "a": float(a_now),
                "tg": float(tg_now),
                "A_comov": gm["A_comov"],
                "grad_fis": gm["A_phys"],
                "T_mean_comov": ft["T_mean_comov"],
                "T_fis": ft["T_fis"],
                "Var_comov": ex["Var_comov"],
                "X_fis": ex["X_fis"],
                "E_comov_sum": e_sum,
                "T_fuera_de_0_1": bool(np.max(T) > 1.0 + 1e-6 or np.min(T) < -1e-6),
            }
        )

    if tg_targets[0] <= 1e-15:
        record(0.0, float(a_grid[0]))
        next_ckpt_idx = 1

    for step in range(1, n_steps + 1):
        tg = step * dtg
        a = float(np.exp(H_EXP * tg))

        # brazo adiabático SIEMPRE presente: rho=rho0/a^3, D=D0*rho/rho0=D0/a^3
        # (idéntico a CF2/F3-6 REAL)
        rho = RHO0 / (a**3)
        D = D0 * (rho / RHO0)

        T = diffuse_con_bano(T, D, a, kappa, DT, N_SUB)

        while next_ckpt_idx < len(tg_targets) and tg >= tg_targets[next_ckpt_idx] - 1e-9:
            record(tg, float(np.exp(H_EXP * tg_targets[next_ckpt_idx])))
            next_ckpt_idx += 1

    while next_ckpt_idx < len(tg_targets):
        a_last = float(a_grid[next_ckpt_idx])
        record(tg_targets[next_ckpt_idx], a_last)
        next_ckpt_idx += 1

    a_vals = np.array([c["a"] for c in checkpoints])
    T_fis = np.array([c["T_fis"] for c in checkpoints])
    X_fis = np.array([c["X_fis"] for c in checkpoints])
    grad_fis = np.array([c["grad_fis"] for c in checkpoints])
    e_sum_vals = np.array([c["E_comov_sum"] for c in checkpoints])
    e_sum0 = e_sum_vals[0] if len(e_sum_vals) else 0.0
    e_sum_drift = np.abs(e_sum_vals - e_sum0) / max(abs(e_sum0), 1e-12)

    return {
        "seed": seed,
        "kappa": kappa,
        "a_grid": a_vals.tolist(),
        "T_fis": T_fis.tolist(),
        "X_fis": X_fis.tolist(),
        "grad_fis": grad_fis.tolist(),
        "T_mean_comov": [c["T_mean_comov"] for c in checkpoints],
        "Var_comov": [c["Var_comov"] for c in checkpoints],
        "E_comov_sum": e_sum_vals.tolist(),
        "E_comov_sum_drift_rel": e_sum_drift.tolist(),
        "fuera_de_0_1_en_algun_checkpoint": bool(any(c["T_fuera_de_0_1"] for c in checkpoints)),
    }


def loglog_slope(a_vals: np.ndarray, vals: np.ndarray) -> float:
    x = np.log(a_vals)
    y = np.log(np.clip(np.abs(vals), 1e-300, None))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def loglog_slope_robusto(a_vals: np.ndarray, vals: np.ndarray, floor_ratio: float = 1e-18):
    """Ajuste de pendiente log-log robusto a COLAPSO NUMÉRICO TOTAL.

    Hallazgo de calibración (smoke test, ANTES de producción, documentado en
    PROTOCOLO_E5.4-5_PREREGISTRO.md §6-bis): en el extremo kappa_max=8.0
    (kappa*dt_sub=1.0, factor de contracción EXACTAMENTE 0), el campo salta
    bit-a-bit al valor uniforme del objetivo del baño en un solo subpaso, así
    que Var_comov (y por tanto X_fis) y grad_fis caen a EXACTAMENTE 0.0 en
    punto flotante (no "muy chico": cero exacto). `loglog_slope` (arriba)
    recorta esos ceros a 1e-300 antes del log, lo que introduce un valor de
    log(1e-300)=-690 como outlier de leverage extremo en un ajuste de solo
    8-9 puntos, produciendo pendientes sin sentido físico (se observó
    slope=+26 en la calibración). Eso es un defecto del AJUSTE, no de la
    física: el colapso exacto a cero ES la señal más fuerte posible de
    "la exergía se derrumbó" — no hay pendiente de ley de potencia que
    ajustar cuando la cantidad es idénticamente cero.

    Esta versión filtra, ANTES de ajustar, los puntos cuyo valor absoluto cae
    por debajo de `floor_ratio` del máximo de la propia curva (ese piso
    separa señal física de ruido de redondeo tras un colapso exacto, que
    aparece en el rango ~1e-16..1e-32 relativo, muy por debajo de cualquier
    decaimiento físico real observado en este barrido). Si sobreviven menos
    de 2 puntos, se reporta colapso_total=True (bandera explícita, no una
    pendiente inventada); el criterio de PASS trata colapso_total como
    "más empinado que cualquier caso no colapsado" (ver evaluate_seed)."""
    vals_abs = np.abs(np.asarray(vals, dtype=float))
    vmax = float(vals_abs.max()) if vals_abs.size else 0.0
    if vmax <= 0.0:
        return None, True
    mask = vals_abs > (floor_ratio * vmax)
    if int(mask.sum()) < 2:
        return None, True
    x = np.log(np.asarray(a_vals, dtype=float)[mask])
    y = np.log(vals_abs[mask])
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), False


def evaluate_seed(runs_by_kappa: dict, kappa_grid: np.ndarray) -> dict:
    """Para una semilla: calcula slope(kappa) y valor_final(kappa) de AMBOS
    observables primarios (T_fis, X_fis), para TODOS los kappa (curva entera,
    T5), y aplica el criterio de PASS pre-registrado (PROTOCOLO §6)."""
    kappa0 = float(kappa_grid[0])
    kappa_max = float(kappa_grid[-1])

    slope_T = {}
    final_T = {}
    slope_X = {}
    colapso_total_X = {}
    final_X = {}
    for k in kappa_grid:
        k_key = f"{k:.6e}"
        run = runs_by_kappa[k_key]
        Tf = np.array(run["T_fis"])
        Xf = np.array(run["X_fis"])
        slope_T[k_key] = loglog_slope(np.array(run["a_grid"]), Tf)
        final_T[k_key] = float(Tf[-1])
        sX, colapso = loglog_slope_robusto(np.array(run["a_grid"]), Xf)
        slope_X[k_key] = sX  # puede ser None si colapso_total=True
        colapso_total_X[k_key] = colapso
        final_X[k_key] = float(Xf[-1])

    k0_key = f"{kappa0:.6e}"
    kmax_key = f"{kappa_max:.6e}"

    slope_T_k0, slope_T_kmax = slope_T[k0_key], slope_T[kmax_key]
    final_T_k0, final_T_kmax = final_T[k0_key], final_T[kmax_key]
    slope_X_k0 = slope_X[k0_key]
    slope_X_kmax = slope_X[kmax_key]
    final_X_k0, final_X_kmax = final_X[k0_key], final_X[kmax_key]

    cond_adiabatico_T = bool(slope_T_k0 <= SLOPE_T_ADIABATICO_MAX)
    cond_bano_aplana_T = bool((slope_T_kmax - slope_T_k0) >= SLOPE_T_FLATTENING_MIN)
    cond_convergencia_T = bool(abs(final_T_kmax - T_BANO) < abs(final_T_k0 - T_BANO))

    # kappa=0 (adiabático) nunca colapsa exactamente (T1: el campo real difunde,
    # no se anula bit-a-bit); si lo hiciera sería en sí una alerta, se reporta
    # como slope=None -> cond_adiabatico_X=False explícito (no se disfraza).
    cond_adiabatico_X = bool(slope_X_k0 is not None and slope_X_k0 <= SLOPE_X_ADIABATICO_MAX)
    # colapso_total_X[kmax]=True (Var/X cayó a exactamente 0.0, el caso del
    # bano mas fuerte) es la senial de derrumbe MAS extrema posible -> satisface
    # trivialmente "mas empinado que el adiabatico" (documentado en PROTOCOLO §6-bis).
    if colapso_total_X[kmax_key]:
        cond_bano_derrumba_X = True
    elif slope_X_k0 is not None and slope_X_kmax is not None:
        cond_bano_derrumba_X = bool((slope_X_k0 - slope_X_kmax) >= SLOPE_X_STEEPENING_MIN)
    else:
        cond_bano_derrumba_X = False
    cond_bano_agota_X = bool(final_X_kmax <= X_RATIO_BANO_ADIABATICO_MAX * final_X_k0)

    seed_pass = bool(
        cond_adiabatico_T and cond_bano_aplana_T and cond_convergencia_T
        and cond_adiabatico_X and cond_bano_derrumba_X and cond_bano_agota_X
    )

    return {
        "slope_T_fis_por_kappa": slope_T,
        "valor_final_T_fis_por_kappa": final_T,
        "slope_X_fis_por_kappa": slope_X,
        "colapso_total_X_por_kappa": colapso_total_X,
        "valor_final_X_fis_por_kappa": final_X,
        "slope_T_kappa0": slope_T_k0,
        "slope_T_kappa_max": slope_T_kmax,
        "valor_final_T_kappa0": final_T_k0,
        "valor_final_T_kappa_max": final_T_kmax,
        "slope_X_kappa0": slope_X_k0,
        "slope_X_kappa_max": slope_X_kmax,
        "valor_final_X_kappa0": final_X_k0,
        "valor_final_X_kappa_max": final_X_kmax,
        "cond_adiabatico_T": cond_adiabatico_T,
        "cond_bano_aplana_T": cond_bano_aplana_T,
        "cond_convergencia_T": cond_convergencia_T,
        "cond_adiabatico_X": cond_adiabatico_X,
        "cond_bano_derrumba_X": cond_bano_derrumba_X,
        "cond_bano_agota_X": cond_bano_agota_X,
        "seed_pass": seed_pass,
    }


def evaluate_seed_grad(runs_by_kappa: dict, kappa_grid: np.ndarray) -> dict:
    """Mismo análisis de slope/valor_final pero para el observable SECUNDARIO
    grad_fis (verificación cruzada independiente, PROTOCOLO §7b). No gatea el
    veredicto principal. Usa el ajuste robusto (loglog_slope_robusto) por la
    misma razón que X_fis: grad_fis también colapsa a exactamente 0.0 en el
    baño más fuerte (campo bit-a-bit uniforme -> gradiente exactamente nulo)."""
    kappa0 = float(kappa_grid[0])
    kappa_max = float(kappa_grid[-1])
    slope_por_kappa = {}
    colapso_total_por_kappa = {}
    final_por_kappa = {}
    for k in kappa_grid:
        k_key = f"{k:.6e}"
        run = runs_by_kappa[k_key]
        gf = np.array(run["grad_fis"])
        s, colapso = loglog_slope_robusto(np.array(run["a_grid"]), gf)
        slope_por_kappa[k_key] = s
        colapso_total_por_kappa[k_key] = colapso
        final_por_kappa[k_key] = float(gf[-1])
    return {
        "slope_por_kappa": slope_por_kappa,
        "colapso_total_por_kappa": colapso_total_por_kappa,
        "valor_final_por_kappa": final_por_kappa,
        "slope_kappa0": slope_por_kappa[f"{kappa0:.6e}"],
        "slope_kappa_max": slope_por_kappa[f"{kappa_max:.6e}"],
    }


def evaluate_seed_e1(runs_by_kappa: dict, kappa_grid: np.ndarray) -> dict:
    """Verificación adicional del axioma E1 (PROTOCOLO §4c/§7c, informativa,
    no gatea): deriva relativa final de E_comov_sum por kappa. Se espera
    ≈0 en kappa=0 (difusión pura conserva la suma) y creciente con kappa
    (el baño bombea/drena el campo)."""
    drift_final_por_kappa = {}
    for k in kappa_grid:
        k_key = f"{k:.6e}"
        run = runs_by_kappa[k_key]
        drift = run["E_comov_sum_drift_rel"]
        drift_final_por_kappa[k_key] = float(drift[-1]) if drift else 0.0
    return {"drift_final_E_comov_sum_por_kappa": drift_final_por_kappa}


def run_production(seeds: list[int], a_grid: np.ndarray, kappa_grid: np.ndarray, tag: str) -> dict:
    t0 = time.time()

    resultados_crudos: dict = {}
    evaluaciones: dict = {}
    evaluaciones_grad: dict = {}
    evaluaciones_e1: dict = {}

    n_pass = 0
    fuera_de_rango_alerta = False

    for seed in seeds:
        runs_by_kappa = {}
        for k in kappa_grid:
            k_key = f"{float(k):.6e}"
            run = run_sweep(seed, a_grid, float(k))
            runs_by_kappa[k_key] = run
            if run["fuera_de_0_1_en_algun_checkpoint"] and float(k) == 0.0:
                fuera_de_rango_alerta = True
        resultados_crudos[str(seed)] = runs_by_kappa

        ev = evaluate_seed(runs_by_kappa, kappa_grid)
        evaluaciones[str(seed)] = ev
        if ev["seed_pass"]:
            n_pass += 1

        evg = evaluate_seed_grad(runs_by_kappa, kappa_grid)
        evaluaciones_grad[str(seed)] = evg

        eve1 = evaluate_seed_e1(runs_by_kappa, kappa_grid)
        evaluaciones_e1[str(seed)] = eve1

    n_total = len(seeds)
    rate = n_pass / n_total if n_total else 0.0
    verdict_label = "E5_4_5_PASS" if rate >= PASS_RATE_MIN else "E5_4_5_FAIL"

    def _mean_std_filtrando_none(vals):
        """Promedio/std ignorando None (colapso_total: sin pendiente definida).
        Reporta también cuántas de las N semillas colapsaron totalmente en ese
        kappa (informativo, T5: no se esconde el colapso, se cuenta)."""
        limpios = [v for v in vals if v is not None]
        n_colapso = len(vals) - len(limpios)
        if limpios:
            return float(np.mean(limpios)), float(np.std(limpios)), n_colapso
        return None, None, n_colapso

    # curva promedio sobre semillas, por kappa (reporte descriptivo, T5)
    curva_kappa = {}
    for k in kappa_grid:
        k_key = f"{float(k):.6e}"
        slopes_T = [evaluaciones[str(s)]["slope_T_fis_por_kappa"][k_key] for s in seeds]
        finals_T = [evaluaciones[str(s)]["valor_final_T_fis_por_kappa"][k_key] for s in seeds]
        slopes_X = [evaluaciones[str(s)]["slope_X_fis_por_kappa"][k_key] for s in seeds]
        finals_X = [evaluaciones[str(s)]["valor_final_X_fis_por_kappa"][k_key] for s in seeds]
        n_colapso_X = sum(1 for s in seeds if evaluaciones[str(s)]["colapso_total_X_por_kappa"][k_key])
        slopes_grad = [evaluaciones_grad[str(s)]["slope_por_kappa"][k_key] for s in seeds]
        drifts_e1 = [evaluaciones_e1[str(s)]["drift_final_E_comov_sum_por_kappa"][k_key] for s in seeds]

        slope_X_mean, slope_X_std, _ = _mean_std_filtrando_none(slopes_X)
        slope_grad_mean, slope_grad_std, n_colapso_grad = _mean_std_filtrando_none(slopes_grad)

        curva_kappa[k_key] = {
            "kappa": float(k),
            "slope_T_fis_mean": float(np.mean(slopes_T)),
            "slope_T_fis_std": float(np.std(slopes_T)),
            "valor_final_T_fis_mean": float(np.mean(finals_T)),
            "valor_final_T_fis_std": float(np.std(finals_T)),
            "slope_X_fis_mean": slope_X_mean,
            "slope_X_fis_std": slope_X_std,
            "n_semillas_colapso_total_X": n_colapso_X,
            "valor_final_X_fis_mean": float(np.mean(finals_X)),
            "valor_final_X_fis_std": float(np.std(finals_X)),
            "slope_grad_fis_mean": slope_grad_mean,
            "slope_grad_fis_std": slope_grad_std,
            "n_semillas_colapso_total_grad": n_colapso_grad,
            "drift_final_E_comov_sum_mean": float(np.mean(drifts_e1)),
            "drift_final_E_comov_sum_std": float(np.std(drifts_e1)),
        }

    payload = {
        "experimento": "E5.4-5 control negativo: enfriamiento CON baño externo (lo prohibido) vs adiabático",
        "tema": "TEMA 4 — Exergía y enfriamiento adiabático",
        "enfoque": "ENFOQUE 5 — Energía, Exergía, Entropía (S=I*E)",
        "tag": tag,
        "sello_fisico_heredado_de_CF2_y_F3_6": {
            "L": L, "H_EXP": H_EXP, "RHO0": RHO0, "D0": D0, "W0": W0,
            "DT": DT, "N_SUB": N_SUB, "ORIGINAL_STEPS_PER_TG": ORIGINAL_STEPS_PER_TG,
        },
        "ingrediente_bano_heredado_de_F3_6": {
            "T_BANO": T_BANO,
            "descripcion": (
                "termino de relajacion tipo Newton hacia objetivo comovil a*T_BANO, "
                "aplicado en cada subpaso: T_c <- T_c + dt_sub*D*lap(T_c) "
                "- dt_sub*kappa*(T_c - a*T_BANO). kappa=0 anula el termino exactamente."
            ),
            "clip_0_1_omitido": True,
            "kappa_max": float(KAPPA_MAX),
            "razon_kappa_max": (
                "kappa_max=1/dt_sub -> kappa*dt_sub=1.0 -> factor de contraccion "
                "(1-kappa*dt_sub)=0: salto exacto al objetivo del bano en un subpaso "
                "(acople perfecto, el caso mas extremo sin cruzar a oscilacion de "
                "signo que empezaria en kappa*dt_sub>1). 8x mayor que F3-6 (kappa_max=1.0), "
                "sobredimensionado segun regla de oro de ENFOQUE 5."
            ),
        },
        "observable_nuevo_exergia_E5_4_5": {
            "formula": "X_fis(a) = Var_espacial(T_comov) / a^2",
            "justificacion": (
                "energia libre disponible de un campo de temperatura escala a segundo "
                "orden con la varianza espacial respecto a su media (termodinamica de "
                "fluctuaciones, ~(deltaT)^2); campo uniforme (Var=0) = cero exergia; "
                "convencion fisico=comovil/a de CF2/F3-6, elevada al cuadrado por "
                "unidades cuadraticas de la varianza; calculada sobre el campo global "
                "(sin banda, la varianza no sufre wrap-around)."
            ),
        },
        "observable_adicional_axioma_E1": {
            "formula": "E_comov_sum(a) = sum(T_comov); drift_rel = |E_sum(a)-E_sum(a=1)|/|E_sum(a=1)|",
            "descripcion": (
                "informativo, no gatea el veredicto. Bajo difusion pura periodica la "
                "suma se conserva exactamente; el bano la modifica activamente -> "
                "firma fisica de violacion del axioma E1 declarado por la bateria."
            ),
        },
        "barrido": {
            "a_grid": a_grid.tolist(),
            "n_a": len(a_grid),
            "kappa_grid": kappa_grid.tolist(),
            "n_kappa": len(kappa_grid),
            "seeds": seeds,
            "n_seeds": len(seeds),
        },
        "criterio_preregistrado": {
            "SLOPE_T_ADIABATICO_MAX": SLOPE_T_ADIABATICO_MAX,
            "SLOPE_T_FLATTENING_MIN": SLOPE_T_FLATTENING_MIN,
            "SLOPE_X_ADIABATICO_MAX": SLOPE_X_ADIABATICO_MAX,
            "SLOPE_X_STEEPENING_MIN": SLOPE_X_STEEPENING_MIN,
            "X_RATIO_BANO_ADIABATICO_MAX": X_RATIO_BANO_ADIABATICO_MAX,
            "PASS_RATE_MIN": PASS_RATE_MIN,
            "descripcion": (
                "seed_pass = cond_adiabatico_T AND cond_bano_aplana_T AND cond_convergencia_T "
                "AND cond_adiabatico_X AND cond_bano_derrumba_X AND cond_bano_agota_X. "
                "T se APLANA con bano (slope sube, converge a T_BANO); X se DERRUMBA con "
                "bano (slope baja mas, valor final cae a <=50% del adiabatico)."
            ),
        },
        "curva_por_kappa_promedio_semillas": curva_kappa,
        "evaluacion_por_semilla": evaluaciones,
        "evaluacion_grad_fis_por_semilla_verificacion_cruzada": evaluaciones_grad,
        "evaluacion_axioma_E1_por_semilla_informativa": evaluaciones_e1,
        "resultados_crudos_por_semilla_y_kappa": resultados_crudos,
        "alerta_metodo": {
            "campo_fuera_de_0_1_en_kappa_0_alguna_semilla": fuera_de_rango_alerta,
            "nota": (
                "si True: omitir el clip cambio apreciablemente el brazo adiabatico "
                "respecto a CF2/F3-6 original; revisar antes de confiar en el veredicto."
            ),
        },
        "veredicto": {
            "n_seeds_pass": n_pass,
            "n_seeds_total": n_total,
            "rate": rate,
            "verdict": verdict_label,
        },
        "runtime_seconds": time.time() - t0,
        "generated_at_unix": time.time(),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "produccion"], default="produccion", nargs="?")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "smoke":
        seeds = SEEDS[:3]
        a_grid = np.geomspace(1.0, 100.0, 4)
        kappa_grid = np.concatenate([[0.0], np.geomspace(1.0e-3, KAPPA_MAX, 3)])
        tag = "smoke"
    else:
        seeds = SEEDS
        a_grid = A_GRID
        kappa_grid = KAPPA_GRID
        tag = "produccion"

    print(f"=== E5.4-5 bano externo vs exergia (control negativo) — modo={args.mode} ===")
    print(f"seeds({len(seeds)})={seeds}")
    print(f"a_grid({len(a_grid)})={a_grid.tolist()}")
    print(f"kappa_grid({len(kappa_grid)})={kappa_grid.tolist()}")
    print(f"T_BANO={T_BANO}  KAPPA_MAX={KAPPA_MAX}")

    payload = run_production(seeds, a_grid, kappa_grid, tag)

    print("\n=== CURVA slope(kappa) / valor_final(kappa) — promedio sobre semillas ===")
    for k_key, pt in payload["curva_por_kappa_promedio_semillas"].items():
        slope_X_str = (
            f"{pt['slope_X_fis_mean']:+.4f}±{pt['slope_X_fis_std']:.4f}"
            if pt["slope_X_fis_mean"] is not None
            else "COLAPSO_TOTAL"
        )
        print(
            f"  kappa={pt['kappa']:.4e}  "
            f"slope_T={pt['slope_T_fis_mean']:+.4f}±{pt['slope_T_fis_std']:.4f}  "
            f"final_T={pt['valor_final_T_fis_mean']:.4f}±{pt['valor_final_T_fis_std']:.4f}  "
            f"slope_X={slope_X_str} (colapso_total en {pt['n_semillas_colapso_total_X']} semillas)  "
            f"final_X={pt['valor_final_X_fis_mean']:.4e}±{pt['valor_final_X_fis_std']:.4e}  "
            f"drift_E1={pt['drift_final_E_comov_sum_mean']:.4e}"
        )

    v = payload["veredicto"]
    print(f"\nrate={v['rate']:.3f}  ({v['n_seeds_pass']}/{v['n_seeds_total']} semillas)  VERDICT={v['verdict']}")
    print(f"(umbral pre-registrado PASS_RATE_MIN={PASS_RATE_MIN})")

    alerta = payload["alerta_metodo"]
    if alerta["campo_fuera_de_0_1_en_kappa_0_alguna_semilla"]:
        print("\n*** ALERTA: el campo salio de [0,1] en el brazo adiabatico (kappa=0) en "
              "alguna semilla. Revisar antes de comparar con CF2/F3-6 original. ***")

    out_json = OUT_DIR / f"E5_4_5_bano_externo_exergia_{args.mode}_result.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON -> {out_json}")


if __name__ == "__main__":
    main()
