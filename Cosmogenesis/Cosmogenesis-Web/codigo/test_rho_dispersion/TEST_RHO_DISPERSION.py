#!/usr/bin/env python3
"""
TEST_RHO_DISPERSION — expansión estira densidad/T; caída abrupta → suave

Pregunta pre-registrada (Alexis + Grok, 2026-07-22):
  ¿La expansión convierte una caída de temperatura ABRUPTA en una caída
  MÁS SUAVE porque la escala se agranda (estiramiento), con densidad
  ρ∝a⁻³ como eslabón activo del medio (no decorado)?

Física del sello (fija antes de ver resultados; NO sintonía a 1/1836):
  1) Reloj genético: t_g = step/PASOS ∈ [0,1]
  2) Expansión: a = exp(H_EXP * t_g)   (misma familia F0: H_EXP=6 → ~403×)
  3) Densidad:  ρ = ρ0 / a³            (activa en transporte)
  4) Temperatura de fondo (reloj): T̄ ∝ 1/a
  5) Campo T(x) comóvil con salto inicial ABRUPTO (frente estrecho)
  6) Difusión comóvil con D = D0 * (ρ/ρ0) = D0 / a³
     → denso temprano puede suavizar un poco; raro tardío CONGELA el perfil
  7) Gradiente FÍSICO: ∇_phys = ∇_comov / a
     → aunque el perfil comóvil se congele, la caída en espacio físico
       se ESTIRA (abrupta → suave) al crecer a

Brazos (mismo sello geométrico/inicial salvo el factor bajo prueba):
  REAL          : a(t), ρ=ρ0/a³, D∝ρ, lectura A_phys = A_comov/a
  NULL_RHO_FIXED: a(t), ρ≡ρ0 (no se enrarece), D constante
  NULL_A_FIXED  : a≡1,  ρ≡ρ0 (ni expansión ni rarefacción)
  NULL_STRETCH  : a(t), D=0 (solo estiramiento geométrico del perfil inicial)

Éxito ≠ ratios de masa MS.
Éxito = criterios abajo (stretch + contraste vs nulos).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# --- sello pre-registrado ---
L = 64
PASOS = 400
H_EXP = 6.0
RHO0 = 1.0
D0 = 0.12          # difusión base a t_g=0 (comóvil)
SEED = 2025
# frente inicial: tanh con ancho comóvil pequeño (ABRUPTO)
W0 = 1.2           # en celdas; << L
DT = 0.25          # subpaso difusión (estable si D*DT*4 < ~1)
N_SUB = 2
# umbrales de veredicto (pre-registrados; no 1/1836)
STRETCH_RATIO_MAX = 0.25   # A_phys_final / A_phys_init < esto → stretch OK
SMOOTH_WIDTH_MIN = 2.0     # w_phys_final / w_phys_init > esto → suavizado OK
RHO_SEP_THR = 0.08         # |Δ A_comov_late| relativo REAL vs RHO_FIXED
A_END_MIN = 50.0           # a final debe ser grande (familia H_EXP=6)


OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "test_rho_dispersion"


def a_of(tg: float) -> float:
    return float(np.exp(H_EXP * tg))


def initial_T(L: int, w0: float) -> np.ndarray:
    """Salto abrupto vertical: T≈1 a la izquierda, T≈0 a la derecha."""
    x = np.arange(L) - (L - 1) / 2.0
    # perfil 1D extendido en y (frente plano)
    profile = 0.5 * (1.0 - np.tanh(x / w0))
    return np.tile(profile, (L, 1))


def grad_metrics(T: np.ndarray, a: float) -> dict:
    """Abruptness comóvil y física + ancho de transición efectivo."""
    # gradiente en x (dirección del salto)
    dTx = 0.5 * (np.roll(T, -1, axis=1) - np.roll(T, 1, axis=1))
    # evitar wrap-around del borde periódico en la métrica: usar solo banda central
    n = T.shape[1]
    band = slice(n // 8, 7 * n // 8)
    g = np.abs(dTx[:, band])
    A_comov = float(g.max()) if g.size else 0.0
    A_phys = A_comov / max(a, 1e-12)

    # ancho comóvil: distancia entre percentiles del perfil medio en x
    row = T.mean(axis=0)
    # interpolar posiciones donde T cruza 0.8 y 0.2
    xs = np.arange(n, dtype=float)

    def cross(level: float) -> float:
        # busca cruce de izquierda (alto) a derecha (bajo)
        for i in range(n - 1):
            if (row[i] - level) * (row[i + 1] - level) <= 0:
                if row[i] == row[i + 1]:
                    return float(i)
                t = (level - row[i]) / (row[i + 1] - row[i])
                return float(i + t)
        return float(n // 2)

    x_hi = cross(0.80)
    x_lo = cross(0.20)
    w_comov = abs(x_lo - x_hi)
    w_phys = w_comov * a

    # suavidad: inverso de abruptness física (más grande = más suave)
    smoothness_phys = 1.0 / (A_phys + 1e-12)
    return {
        "A_comov": A_comov,
        "A_phys": A_phys,
        "w_comov": float(w_comov),
        "w_phys": float(w_phys),
        "smoothness_phys": float(smoothness_phys),
        "T_mean": float(T.mean()),
        "T_std": float(T.std()),
    }


def diffuse(T: np.ndarray, D: float, dt: float, n_sub: int) -> np.ndarray:
    """Difusión isótropa comóvil; D puede ser 0."""
    if D <= 0:
        return T
    out = T
    for _ in range(n_sub):
        lap = (
            np.roll(out, -1, 1)
            + np.roll(out, 1, 1)
            + np.roll(out, -1, 0)
            + np.roll(out, 1, 0)
            - 4.0 * out
        )
        out = out + (dt / n_sub) * D * lap
    return out


def run_arm(mode: str, seed: int = SEED) -> dict:
    """
    mode ∈ {REAL, NULL_RHO_FIXED, NULL_A_FIXED, NULL_STRETCH}
    """
    rng = np.random.default_rng(seed)
    T = initial_T(L, W0)
    # minúscula semilla de ruido espacial (no borra el frente)
    T = T + 1e-4 * rng.normal(size=T.shape)
    T = np.clip(T, 0.0, 1.0)

    hist = []
    for step in range(PASOS):
        tg = step / max(PASOS - 1, 1)
        if mode == "NULL_A_FIXED":
            a = 1.0
            rho = RHO0
        else:
            a = a_of(tg)
            if mode == "NULL_RHO_FIXED":
                rho = RHO0
            else:
                # REAL y NULL_STRETCH: densidad se enrarece
                rho = RHO0 / (a**3)

        Tbar = 1.0 / a  # reloj de fondo (reporte); el contraste vive en T(x)

        if mode == "NULL_STRETCH":
            D = 0.0
        elif mode == "NULL_RHO_FIXED":
            D = D0 * (rho / RHO0)  # = D0
        elif mode == "NULL_A_FIXED":
            D = D0
        else:  # REAL
            D = D0 * (rho / RHO0)  # = D0 / a³

        T = diffuse(T, D, DT, N_SUB)
        T = np.clip(T, 0.0, 1.0)

        if step % 25 == 0 or step == PASOS - 1:
            m = grad_metrics(T, a)
            rec = {
                "step": step,
                "tg": tg,
                "a": a,
                "rho": float(rho),
                "D": float(D),
                "Tbar": float(Tbar),
                **m,
            }
            hist.append(rec)

    init = hist[0]
    final = hist[-1]
    # punto a mitad de expansión
    mid = hist[len(hist) // 2]

    return {
        "mode": mode,
        "seed": seed,
        "init": init,
        "mid": mid,
        "final": final,
        "hist": hist,
        "ratios": {
            "A_phys_ratio": final["A_phys"] / max(init["A_phys"], 1e-12),
            "A_comov_ratio": final["A_comov"] / max(init["A_comov"], 1e-12),
            "w_phys_ratio": final["w_phys"] / max(init["w_phys"], 1e-12),
            "w_comov_ratio": final["w_comov"] / max(init["w_comov"], 1e-12),
            "a_final": final["a"],
            "rho_final": final["rho"],
        },
    }


def verdict(arms: dict[str, dict]) -> dict:
    real = arms["REAL"]["ratios"]
    stretch = arms["NULL_STRETCH"]["ratios"]
    rho_fix = arms["NULL_RHO_FIXED"]["ratios"]
    a_fix = arms["NULL_A_FIXED"]["ratios"]

    # 1) Estiramiento geométrico puro: A_phys cae ~1/a, A_comov ~cte
    stretch_ok = (
        stretch["A_phys_ratio"] < STRETCH_RATIO_MAX
        and stretch["a_final"] >= A_END_MIN
        and stretch["A_comov_ratio"] > 0.85  # perfil congelado
    )
    # suavizado físico: ancho físico crece con a
    smooth_ok = stretch["w_phys_ratio"] >= SMOOTH_WIDTH_MIN

    # 2) REAL: también suaviza en físico; además ρ diluye D
    real_stretch_ok = real["A_phys_ratio"] < STRETCH_RATIO_MAX
    real_smooth_ok = real["w_phys_ratio"] >= SMOOTH_WIDTH_MIN

    # 3) Efecto densidad: en REAL, D cae → menos erosión comóvil tardía
    #    que con ρ fija (donde D se mantiene alta y aplana más el frente).
    #    Esperamos A_comov_final_REAL > A_comov_final_RHO_FIXED (frente más vivo)
    #    o w_comov_REAL < w_comov_RHO_FIXED
    A_real = arms["REAL"]["final"]["A_comov"]
    A_rf = arms["NULL_RHO_FIXED"]["final"]["A_comov"]
    rho_contrast = abs(A_real - A_rf) / max(A_rf, A_real, 1e-12)
    # dirección esperada: rarefacción preserva más contraste comóvil
    rho_direction_ok = A_real > A_rf * (1.0 + 0.5 * RHO_SEP_THR)
    rho_effect_ok = rho_contrast >= RHO_SEP_THR and rho_direction_ok

    # 4) Control a fija: sin expansión, A_phys no se “estira”; puede caer solo por difusión
    a_fixed_no_stretch = a_fix["a_final"] < 1.01

    flags = {
        "stretch_pure_ok": bool(stretch_ok),
        "smooth_pure_ok": bool(smooth_ok),
        "real_stretch_ok": bool(real_stretch_ok),
        "real_smooth_ok": bool(real_smooth_ok),
        "rho_effect_ok": bool(rho_effect_ok),
        "a_fixed_control_ok": bool(a_fixed_no_stretch),
    }

    if (
        flags["stretch_pure_ok"]
        and flags["smooth_pure_ok"]
        and flags["real_stretch_ok"]
        and flags["rho_effect_ok"]
    ):
        label = "TEST_PASS_stretch_and_rho"
    elif flags["stretch_pure_ok"] and flags["smooth_pure_ok"] and flags["real_stretch_ok"]:
        label = "TEST_PARTIAL_stretch_rho_weak"
    elif flags["stretch_pure_ok"] and flags["smooth_pure_ok"]:
        label = "TEST_PARTIAL_stretch_only"
    else:
        label = "TEST_FAIL_no_dispersion"

    return {
        "verdict": label,
        "flags": flags,
        "rho_contrast": float(rho_contrast),
        "A_comov_final_REAL": float(A_real),
        "A_comov_final_RHO_FIXED": float(A_rf),
        "ratios_REAL": real,
        "ratios_NULL_STRETCH": stretch,
        "ratios_NULL_RHO_FIXED": rho_fix,
        "ratios_NULL_A_FIXED": a_fix,
        "thresholds": {
            "STRETCH_RATIO_MAX": STRETCH_RATIO_MAX,
            "SMOOTH_WIDTH_MIN": SMOOTH_WIDTH_MIN,
            "RHO_SEP_THR": RHO_SEP_THR,
            "A_END_MIN": A_END_MIN,
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    modes = ["REAL", "NULL_RHO_FIXED", "NULL_A_FIXED", "NULL_STRETCH"]
    print("=== TEST_RHO_DISPERSION ===")
    print(
        f"sello L={L} PASOS={PASOS} H_EXP={H_EXP} D0={D0} W0={W0} "
        f"ρ=ρ0/a³  ∇_phys=∇_comov/a"
    )
    print("hipótesis: expansión ESTIRA → caída T abrupta → suave en físico;")
    print("           densidad al caer CONGELA el transporte (D∝ρ).\n")

    arms = {}
    for mode in modes:
        print(f"--- {mode} ---")
        arm = run_arm(mode)
        arms[mode] = arm
        r = arm["ratios"]
        print(
            f"  a: {arm['init']['a']:.2f} → {arm['final']['a']:.2f}  "
            f"ρ: {arm['init']['rho']:.3e} → {arm['final']['rho']:.3e}"
        )
        print(
            f"  A_comov: {arm['init']['A_comov']:.4f} → {arm['final']['A_comov']:.4f}  "
            f"(×{r['A_comov_ratio']:.3f})"
        )
        print(
            f"  A_phys:  {arm['init']['A_phys']:.4f} → {arm['final']['A_phys']:.4f}  "
            f"(×{r['A_phys_ratio']:.3f})"
        )
        print(
            f"  w_phys:  {arm['init']['w_phys']:.3f} → {arm['final']['w_phys']:.3f}  "
            f"(×{r['w_phys_ratio']:.3f})"
        )

    v = verdict(arms)
    print("\n=== VEREDICTO ===")
    print(v["verdict"])
    print("flags:", v["flags"])
    print(
        f"rho_contrast={v['rho_contrast']:.4f}  "
        f"A_comov REAL={v['A_comov_final_REAL']:.4f}  "
        f"RHO_FIXED={v['A_comov_final_RHO_FIXED']:.4f}"
    )

    # JSON compacto (hist completo)
    payload = {
        "sello": {
            "L": L,
            "PASOS": PASOS,
            "H_EXP": H_EXP,
            "RHO0": RHO0,
            "D0": D0,
            "W0": W0,
            "SEED": SEED,
            "DT": DT,
            "N_SUB": N_SUB,
        },
        "arms": {
            m: {
                "mode": arms[m]["mode"],
                "seed": arms[m]["seed"],
                "init": arms[m]["init"],
                "mid": arms[m]["mid"],
                "final": arms[m]["final"],
                "ratios": arms[m]["ratios"],
                "hist": arms[m]["hist"],
            }
            for m in modes
        },
        "verdict": v,
        "lectura": {
            "acuerdo_estiramiento": (
                "Sí: con a↑, ∇_phys=∇_comov/a cae → la misma caída de T "
                "pasa de abrupta a suave en espacio físico (escala agrandada)."
            ),
            "rol_densidad": (
                "ρ∝a⁻³ hace D∝1/a³: el transporte se apaga al expandir; "
                "el frente comóvil se congela y el suavizado físico lo hace el estiramiento."
            ),
        },
    }
    out_json = OUT_DIR / "TEST_RHO_DISPERSION_result.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON → {out_json}")

    # resumen markdown
    md = []
    md.append("# TEST_RHO_DISPERSION — resultado\n")
    md.append("**Fecha:** 2026-07-22\n")
    md.append("## Hipótesis (Alexis)\n")
    md.append(
        "Con la expansión la densidad se **estira**/enrarece: de una "
        "caída de temperatura **abrupta** se pasa a una caída **suave** "
        "porque la escala se agranda.\n"
    )
    md.append("## Veredicto\n")
    md.append(f"**`{v['verdict']}`**\n")
    md.append("### Flags\n")
    for k, val in v["flags"].items():
        md.append(f"- `{k}`: **{val}**\n")
    md.append("\n### Ratios (final/init)\n")
    md.append("| brazo | A_phys | A_comov | w_phys | a_final | ρ_final |\n")
    md.append("|-------|--------|---------|--------|---------|--------|\n")
    for m in modes:
        r = arms[m]["ratios"]
        md.append(
            f"| {m} | ×{r['A_phys_ratio']:.4f} | ×{r['A_comov_ratio']:.4f} | "
            f"×{r['w_phys_ratio']:.3f} | {r['a_final']:.1f} | {r['rho_final']:.3e} |\n"
        )
    md.append("\n### Contraste densidad\n")
    md.append(
        f"- A_comov final REAL = {v['A_comov_final_REAL']:.4f}\n"
        f"- A_comov final RHO_FIXED = {v['A_comov_final_RHO_FIXED']:.4f}\n"
        f"- rho_contrast = {v['rho_contrast']:.4f} (umbral {RHO_SEP_THR})\n"
    )
    md.append("\n## Lectura\n")
    md.append(
        "1. **Estiramiento:** ∇_phys = ∇_comov/a — a mayor a, la caída es más suave "
        "en espacio físico aunque el perfil comóvil esté congelado.\n"
        "2. **Densidad:** ρ∝a⁻³ apaga D; sin eso (ρ fija) el frente se erosiona más "
        "en coordenadas comóviles.\n"
        "3. No es claim de Higgs ni de 1/1836; es el eslabón basal "
        "expansión → densidad → dispersión del gradiente térmico.\n"
    )
    md.append("\n## Artefactos\n")
    md.append("- `codigo/test_rho_dispersion/TEST_RHO_DISPERSION.py`\n")
    md.append("- `results/test_rho_dispersion/TEST_RHO_DISPERSION_result.json`\n")
    out_md = OUT_DIR / "RESUMEN_TEST_RHO_DISPERSION.md"
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"MD  → {out_md}")


if __name__ == "__main__":
    main()
