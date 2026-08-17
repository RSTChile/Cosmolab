#!/usr/bin/env python3
"""
F3_6_bano_externo_motor.py — BATERÍA_FUNDAMENTOS, Enfoque 3, experimento F3-6

"Control negativo: enfriamiento CON baño externo (lo prohibido)"

Todo el Enfoque 3 (F3-1..F3-5) afirma que el campo se enfría SIN ningún término
de enfriamiento impuesto — solo por re-escalamiento geométrico del gradiente
(∇_fis=∇_comov/a) y por la dilución de la difusividad (D=D0/a³). Esa
afirmación solo es creíble si el arnés experimental sabe distinguir ese
mecanismo de la alternativa prohibida: que el modelo estuviera, sin darnos
cuenta, simulando un baño térmico externo (acoplamiento a un reservorio a
temperatura fija).

Este motor mete ESE baño A PROPÓSITO, como control negativo: añade un término
de relajación tipo Newton hacia una temperatura de reservorio fija T_BANO,
con intensidad kappa barrida desde 0 (= adiabático puro, reduce EXACTAMENTE
al caso CF-2/F3-1) hasta un valor "fuerte". Predicción pre-registrada
(PROTOCOLO_F3-6_PREREGISTRO.md):

    kappa=0  (adiabático) : T_fis(a) ~ a^(-n)   (ley de potencia)
    kappa>0  (con baño)   : T_fis(a) -> T_BANO  (NO ley de potencia)

Si ambos casos NO difieren claramente, es una alerta seria sobre el arnés de
todo el Enfoque 3 (se reporta como hallazgo, no se esconde).

No edita CF2_estiramiento_motor.py. Reutiliza su mismo sello físico (L, H_EXP,
RHO0, D0, W0, DT, N_SUB, reloj genético, laplaciano de 5 puntos, salto tanh,
banda central anti-wrap, observable de gradiente ∇_fis=∇_comov/a) y añade
EXACTAMENTE lo que pide F3-6: el término de baño explícito, con su propia
intensidad barrida (kappa), y un observable primario nuevo — T_fis(a) =
mean(T_comov)/a, "temperatura física del campo" (análogo a la dilución
cosmológica T∝1/a de un gas sin interacción) — que es la cantidad que la
spec pide comparar contra T_BANO.

Diferencia declarada respecto a CF2 (documentada en el pre-registro §3): se
omite el np.clip(T,0,1) que CF2 aplica, en AMBOS brazos (kappa=0 y kappa>0),
porque para kappa>0 el objetivo comóvil del baño (a*T_BANO) supera 1 a
medida que crece `a` por diseño (es la representación comóvil correcta de un
baño físico a temperatura fija); recortar destruiría la física a probar. Se
verifica en el JSON que para kappa=0 esto no cambia apreciablemente el
resultado frente al CF2 original (el campo no se sale de [0,1] bajo difusión
pura con esta amplitud de ruido).

No se auto-adjudica el hallazgo más amplio del Enfoque 3. Entrega números
crudos; la adjudicación es de CS. No topología. No commits.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ============================================================
# Sello físico heredado de CF2_estiramiento_motor.py, IDÉNTICO
# (T1: no se retoca nada del núcleo para favorecer un resultado)
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
# Ingrediente nuevo de F3-6: baño térmico externo (PROTOCOLO §3)
# ============================================================
T_BANO = 0.5  # temperatura física fija del reservorio (punto medio de [0,1], fijada una vez)

# ============================================================
# Barrido pre-registrado (PROTOCOLO_F3-6_PREREGISTRO.md §5)
# ============================================================
A_GRID = np.geomspace(1.0, 1000.0, 7)  # idéntico a A_GRID de CF2 (T1: reutilizado, no re-elegido)
KAPPA_GRID = np.concatenate([[0.0], np.geomspace(1.0e-3, 1.0, 7)])  # 8 puntos, incluye 0 exacto
SEEDS_STANDARD_PROJECT = [7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321]
SEEDS_NUEVAS_F3_6 = [13, 271828]
SEEDS = SEEDS_STANDARD_PROJECT + SEEDS_NUEVAS_F3_6  # 12 semillas (>=12 exigidas)

# ============================================================
# Criterio de PASS pre-registrado (PROTOCOLO_F3-6_PREREGISTRO.md §6)
# ============================================================
SLOPE_ADIABATICO_MAX = -0.5     # slope(kappa=0) debe ser <= esto (caída clara en ley de potencia)
SLOPE_FLATTENING_MIN = 0.3      # slope(kappa_max) - slope(kappa=0) >= esto (aplanamiento claro)
PASS_RATE_MIN = 0.55            # idéntico a CF-2/F3-1, no re-elegido (T1)

CODE_DIR = Path(__file__).resolve().parent
WEB_ROOT = CODE_DIR.parents[2]  # .../Cosmogenesis-Web
OUT_DIR = WEB_ROOT / "results" / "BATERIA_FUNDAMENTOS" / "F3_6_bano_externo"


def initial_T(L: int, w0: float) -> np.ndarray:
    """Salto abrupto vertical: T≈1 a la izquierda, T≈0 a la derecha (frente plano en y).
    Idéntico a CF2_estiramiento_motor.py."""
    x = np.arange(L) - (L - 1) / 2.0
    profile = 0.5 * (1.0 - np.tanh(x / w0))
    return np.tile(profile, (L, 1))


def grad_metrics(T: np.ndarray, a: float) -> dict:
    """Abruptancia comóvil y física, banda central (evita wrap-around periódico).
    Idéntico a CF2_estiramiento_motor.py (observable secundario de F3-6, §4)."""
    dTx = 0.5 * (np.roll(T, -1, axis=1) - np.roll(T, 1, axis=1))
    n = T.shape[1]
    band = slice(n // 8, 7 * n // 8)
    g = np.abs(dTx[:, band])
    A_comov = float(g.max()) if g.size else 0.0
    A_phys = A_comov / max(a, 1e-12)
    return {"A_comov": A_comov, "A_phys": A_phys}


def field_temperature(T: np.ndarray, a: float) -> dict:
    """Observable PRIMARIO de F3-6 (PROTOCOLO §4): temperatura física del campo,
    T_fis(a) = mean(T_comov) / a — media espacial GLOBAL (todo L×L, no banda),
    análoga a la dilución cosmológica T∝1/a de un gas sin interacción."""
    T_mean_comov = float(np.mean(T))
    T_fis = T_mean_comov / max(a, 1e-12)
    return {"T_mean_comov": T_mean_comov, "T_fis": T_fis}


def diffuse_con_bano(T: np.ndarray, D: float, a: float, kappa: float,
                      dt: float, n_sub: int) -> np.ndarray:
    """Difusión con laplaciano de 5 puntos (idéntico a CF2) + término de baño
    externo (PROTOCOLO §3): en cada subpaso,

        T_c <- T_c + dt_sub*D*lap(T_c) - dt_sub*kappa*(T_c - a*T_BANO)

    Con kappa=0 el segundo término se anula EXACTAMENTE y esto reduce
    bit-a-bit a la función `diffuse` de CF2_estiramiento_motor.py, salvo por
    el clip (ver nota de deviación declarada en el pre-registro §3: aquí NO
    se recorta a [0,1] en ningún brazo, para no introducir una discontinuidad
    de comportamiento justo en kappa=0)."""
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
    t_g(a)=ln(a)/H_EXP. Mismo método de checkpointing markoviano que CF2/F3-1
    (una sola trayectoria por (kappa, semilla), muestreada en los `a`
    objetivo). El brazo adiabático (kappa=0) usa D(a)=D0/a³ como en CF2 —
    ese ingrediente NUNCA se apaga (T1: no se quita física ya validada), solo
    se le suma o no el término de baño."""
    rng = np.random.default_rng(seed)
    T = initial_T(L, W0)
    T = T + 1e-4 * rng.normal(size=T.shape)  # mismo ruido de condición inicial que CF2 (sin clip, ver §3)

    dtg = 1.0 / ORIGINAL_STEPS_PER_TG
    tg_targets = np.log(a_grid) / H_EXP
    tg_max = float(tg_targets[-1])
    n_steps = max(int(np.ceil(tg_max / dtg)), 1)

    checkpoints = []
    next_ckpt_idx = 0

    def record(tg_now, a_now):
        gm = grad_metrics(T, a_now)
        ft = field_temperature(T, a_now)
        checkpoints.append(
            {
                "a": float(a_now),
                "tg": float(tg_now),
                "A_comov": gm["A_comov"],
                "grad_fis": gm["A_phys"],
                "T_mean_comov": ft["T_mean_comov"],
                "T_fis": ft["T_fis"],
                "T_fuera_de_0_1": bool(np.max(T) > 1.0 + 1e-6 or np.min(T) < -1e-6),
            }
        )

    if tg_targets[0] <= 1e-15:
        record(0.0, float(a_grid[0]))
        next_ckpt_idx = 1

    for step in range(1, n_steps + 1):
        tg = step * dtg
        a = float(np.exp(H_EXP * tg))

        # brazo adiabático SIEMPRE presente: ρ=ρ0/a³, D=D0*ρ/ρ0=D0/a³ (idéntico a CF2 REAL)
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
    grad_fis = np.array([c["grad_fis"] for c in checkpoints])

    return {
        "seed": seed,
        "kappa": kappa,
        "a_grid": a_vals.tolist(),
        "T_fis": T_fis.tolist(),
        "grad_fis": grad_fis.tolist(),
        "T_mean_comov": [c["T_mean_comov"] for c in checkpoints],
        "fuera_de_0_1_en_algun_checkpoint": bool(any(c["T_fuera_de_0_1"] for c in checkpoints)),
    }


def loglog_slope(a_vals: np.ndarray, vals: np.ndarray) -> float:
    x = np.log(a_vals)
    y = np.log(np.clip(np.abs(vals), 1e-300, None))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def evaluate_seed(runs_by_kappa: dict, kappa_grid: np.ndarray) -> dict:
    """Para una semilla: calcula slope(kappa) y valor_final(kappa) del
    observable primario T_fis, para TODOS los kappa (curva entera, T5), y
    aplica el criterio de PASS pre-registrado (§6) comparando kappa=0 vs
    kappa=kappa_max."""
    kappa0 = float(kappa_grid[0])
    kappa_max = float(kappa_grid[-1])

    slope_por_kappa = {}
    final_por_kappa = {}
    for k in kappa_grid:
        k_key = f"{k:.6e}"
        run = runs_by_kappa[k_key]
        Tf = np.array(run["T_fis"])
        slope_por_kappa[k_key] = loglog_slope(np.array(run["a_grid"]), Tf)
        final_por_kappa[k_key] = float(Tf[-1])

    slope_k0 = slope_por_kappa[f"{kappa0:.6e}"]
    slope_kmax = slope_por_kappa[f"{kappa_max:.6e}"]
    final_k0 = final_por_kappa[f"{kappa0:.6e}"]
    final_kmax = final_por_kappa[f"{kappa_max:.6e}"]

    cond_adiabatico = bool(slope_k0 <= SLOPE_ADIABATICO_MAX)
    cond_bano_aplana = bool((slope_kmax - slope_k0) >= SLOPE_FLATTENING_MIN)
    cond_convergencia = bool(abs(final_kmax - T_BANO) < abs(final_k0 - T_BANO))

    seed_pass = bool(cond_adiabatico and cond_bano_aplana and cond_convergencia)

    return {
        "slope_por_kappa": slope_por_kappa,
        "valor_final_por_kappa": final_por_kappa,
        "slope_kappa0": slope_k0,
        "slope_kappa_max": slope_kmax,
        "valor_final_kappa0": final_k0,
        "valor_final_kappa_max": final_kmax,
        "cond_adiabatico": cond_adiabatico,
        "cond_bano_aplana": cond_bano_aplana,
        "cond_convergencia": cond_convergencia,
        "seed_pass": seed_pass,
    }


def evaluate_seed_grad(runs_by_kappa: dict, kappa_grid: np.ndarray) -> dict:
    """Mismo análisis de slope/valor_final pero para el observable SECUNDARIO
    grad_fis (verificación cruzada independiente, PROTOCOLO §7b). No gatea
    el veredicto principal."""
    kappa0 = float(kappa_grid[0])
    kappa_max = float(kappa_grid[-1])
    slope_por_kappa = {}
    final_por_kappa = {}
    for k in kappa_grid:
        k_key = f"{k:.6e}"
        run = runs_by_kappa[k_key]
        gf = np.array(run["grad_fis"])
        slope_por_kappa[k_key] = loglog_slope(np.array(run["a_grid"]), gf)
        final_por_kappa[k_key] = float(gf[-1])
    return {
        "slope_por_kappa": slope_por_kappa,
        "valor_final_por_kappa": final_por_kappa,
        "slope_kappa0": slope_por_kappa[f"{kappa0:.6e}"],
        "slope_kappa_max": slope_por_kappa[f"{kappa_max:.6e}"],
    }


def run_production(seeds: list[int], a_grid: np.ndarray, kappa_grid: np.ndarray, tag: str) -> dict:
    t0 = time.time()

    # resultados_crudos[str(seed)][kappa_key] = run dict
    resultados_crudos: dict = {}
    evaluaciones: dict = {}
    evaluaciones_grad: dict = {}

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

    n_total = len(seeds)
    rate = n_pass / n_total if n_total else 0.0
    verdict_label = "F3_6_PASS" if rate >= PASS_RATE_MIN else "F3_6_FAIL"

    # curva promedio sobre semillas, por kappa (reporte descriptivo, T5)
    curva_kappa = {}
    for k in kappa_grid:
        k_key = f"{float(k):.6e}"
        slopes = [evaluaciones[str(s)]["slope_por_kappa"][k_key] for s in seeds]
        finals = [evaluaciones[str(s)]["valor_final_por_kappa"][k_key] for s in seeds]
        slopes_grad = [evaluaciones_grad[str(s)]["slope_por_kappa"][k_key] for s in seeds]
        curva_kappa[k_key] = {
            "kappa": float(k),
            "slope_T_fis_mean": float(np.mean(slopes)),
            "slope_T_fis_std": float(np.std(slopes)),
            "valor_final_T_fis_mean": float(np.mean(finals)),
            "valor_final_T_fis_std": float(np.std(finals)),
            "slope_grad_fis_mean": float(np.mean(slopes_grad)),
            "slope_grad_fis_std": float(np.std(slopes_grad)),
        }

    payload = {
        "experimento": "F3-6 control negativo: enfriamiento CON baño externo (lo prohibido)",
        "enfoque": "ENFOQUE 3 — ¿enfriar es expandir?",
        "tag": tag,
        "sello_fisico_heredado_de_CF2": {
            "L": L, "H_EXP": H_EXP, "RHO0": RHO0, "D0": D0, "W0": W0,
            "DT": DT, "N_SUB": N_SUB, "ORIGINAL_STEPS_PER_TG": ORIGINAL_STEPS_PER_TG,
        },
        "ingrediente_nuevo_F3_6": {
            "T_BANO": T_BANO,
            "descripcion": (
                "termino de relajacion tipo Newton hacia objetivo comovil a*T_BANO, "
                "aplicado en cada subpaso: T_c <- T_c + dt_sub*D*lap(T_c) "
                "- dt_sub*kappa*(T_c - a*T_BANO). kappa=0 anula el termino exactamente."
            ),
            "clip_0_1_omitido": True,
            "razon_clip_omitido": (
                "el objetivo comovil a*T_BANO supera 1 al crecer a; recortar destruiria "
                "la fisica del bano. Ver PROTOCOLO_F3-6_PREREGISTRO.md #3."
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
            "SLOPE_ADIABATICO_MAX": SLOPE_ADIABATICO_MAX,
            "SLOPE_FLATTENING_MIN": SLOPE_FLATTENING_MIN,
            "PASS_RATE_MIN": PASS_RATE_MIN,
            "descripcion": (
                "seed_pass = (slope(kappa=0) <= SLOPE_ADIABATICO_MAX) AND "
                "((slope(kappa_max) - slope(kappa=0)) >= SLOPE_FLATTENING_MIN) AND "
                "(|valor_final(kappa_max) - T_BANO| < |valor_final(kappa=0) - T_BANO|)"
            ),
        },
        "curva_por_kappa_promedio_semillas": curva_kappa,
        "evaluacion_T_fis_por_semilla": evaluaciones,
        "evaluacion_grad_fis_por_semilla_verificacion_cruzada": evaluaciones_grad,
        "resultados_crudos_por_semilla_y_kappa": resultados_crudos,
        "alerta_metodo": {
            "campo_fuera_de_0_1_en_kappa_0_alguna_semilla": fuera_de_rango_alerta,
            "nota": (
                "si True: omitir el clip cambio apreciablemente el brazo adiabatico "
                "respecto a CF2 original; revisar antes de confiar en el veredicto."
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
        kappa_grid = np.concatenate([[0.0], np.geomspace(1.0e-3, 1.0, 3)])
        tag = "smoke"
    else:
        seeds = SEEDS
        a_grid = A_GRID
        kappa_grid = KAPPA_GRID
        tag = "produccion"

    print(f"=== F3-6 bano externo (control negativo) — modo={args.mode} ===")
    print(f"seeds({len(seeds)})={seeds}")
    print(f"a_grid({len(a_grid)})={a_grid.tolist()}")
    print(f"kappa_grid({len(kappa_grid)})={kappa_grid.tolist()}")
    print(f"T_BANO={T_BANO}")

    payload = run_production(seeds, a_grid, kappa_grid, tag)

    print("\n=== CURVA slope(kappa) / valor_final(kappa) — promedio sobre semillas ===")
    for k_key, pt in payload["curva_por_kappa_promedio_semillas"].items():
        print(
            f"  kappa={pt['kappa']:.4e}  "
            f"slope_T_fis={pt['slope_T_fis_mean']:+.4f}±{pt['slope_T_fis_std']:.4f}  "
            f"valor_final_T_fis={pt['valor_final_T_fis_mean']:.4f}±{pt['valor_final_T_fis_std']:.4f}  "
            f"slope_grad_fis={pt['slope_grad_fis_mean']:+.4f}±{pt['slope_grad_fis_std']:.4f}"
        )

    v = payload["veredicto"]
    print(f"\nrate={v['rate']:.3f}  ({v['n_seeds_pass']}/{v['n_seeds_total']} semillas)  VERDICT={v['verdict']}")
    print(f"(umbral pre-registrado PASS_RATE_MIN={PASS_RATE_MIN})")

    alerta = payload["alerta_metodo"]
    if alerta["campo_fuera_de_0_1_en_kappa_0_alguna_semilla"]:
        print("\n*** ALERTA: el campo salio de [0,1] en el brazo adiabatico (kappa=0) en "
              "alguna semilla. Revisar antes de comparar con CF2 original. ***")

    out_json = OUT_DIR / f"F3_6_bano_externo_{args.mode}_result.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON -> {out_json}")


if __name__ == "__main__":
    main()
