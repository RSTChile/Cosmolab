"""
kappas_mortalidad_barrido.py — κ_P y κ_Δ remedidos en CG002 (instrumento CON mortalidad)
========================================================================================

QUÉ HACE ESTE ARCHIVO
---------------------
Busca si existe un PISO de persistencia (κ_P) y un PISO de diferencia operable (κ_Δ) en
CG002, el único instrumento del proyecto donde los nodos SÍ pueden morir
(`_vivo(s) = s > KAPPA_S`).  Los κ medidos sobre Phantom se cayeron porque allí la
mortalidad era cero (280/280 sumideros llegan al final) — ver
`VALIDACION_kappaP_kappaDelta_controles_con_malla_CS.md`.

El pre-registro está en `KAPPAS_remedidos_instrumento_con_mortalidad_CS.md` §0 y se
escribió ANTES que este archivo.

NO REESCRIBE EL MOTOR
---------------------
`cg002_acoplamiento.correr` se importa tal cual y se usa como REFERENCIA de verificación.
Lo que sí hace falta es un núcleo parametrizado por PRECISIÓN NUMÉRICA (float64 / float80 /
mpmath), porque la guarda crítica del encargo —la lección F1-3— exige correr el barrido en
dos precisiones y ver si el piso se mueve. El motor original está clavado en float64.
`verificar_replica()` comprueba que el núcleo reproduce al original antes de usarlo.

ESTRUCTURA
----------
  §1  núcleo del micro-paso, parametrizado por dtype (vectorizado)
  §2  núcleo en mpmath (precisión arbitraria, escalar, lento — sólo para la guarda)
  §3  verificación del núcleo contra cg002_acoplamiento.correr
  §4  BARRIDO A/A'  — κ_P: α y MU en muchas décadas, 3 brazos, 2 precisiones
  §5  BARRIDO C     — κ_P: S0 en 15 décadas (test de tautología)
  §6  BARRIDO B/B'  — κ_Δ: diferencia de fase δ en 20 décadas, 3 precisiones
  §7  guardas: identidades algebraicas, piso de ruido, no-isomorfía
  §8  figura y salidas
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

RAIZ = Path(__file__).resolve().parent

# --- Constantes del protocolo CG002 v0.1c (copiadas, no reinterpretadas) ---
K_FIRMA = 8
KAPPA_S_0 = 1e-6
S0_0 = 1.0
ETA_0 = 0.05
MU_0 = 0.01
EPS_TAU_0 = 1e-4
W_MIN_0 = 0.1 * S0_0
TAU_CAP_0 = 500
K_MAX_0 = 1500

SEMILLAS = list(range(1, 11))   # 10 semillas por brazo y por punto del barrido
SEED_BARAJA = 90000


# =========================================================================
# §1 · NÚCLEO parametrizado por precisión (vectorizado)
# =========================================================================
# π con todos los dígitos que la precisión admita: si se usara np.longdouble(np.pi) el
# brazo de float80 arrastraría el redondeo de float64 y la guarda de precisión sería falsa.
_PI = {np.float64: np.float64("3.14159265358979311600"),
       np.longdouble: np.longdouble("3.14159265358979323846264338327950288")}
# nombre estable de la precisión (np.dtype(np.longdouble).name da 'float128' aunque el
# formato real en x86 sea el extendido de 80 bits: 64 bits de mantisa, eps=1.08e-19)
PREC = {np.float64: "float64", np.longdouble: "float80"}


def _matriz_g(omega: np.ndarray, K: int, theta: float, dt) -> np.ndarray:
    """g_ij = cos(2π(ω_i−ω_j)/K + θ). Diagonal a 0 (un nodo no se acopla consigo mismo).

    Es la MISMA fórmula de `cg002_observables_v02.g_dirigido`, sólo que evaluada de una
    vez para todos los pares en lugar de par por par."""
    om = np.asarray(omega, dtype=dt)
    a = (dt(2) * _PI[dt]) * (om[:, None] - om[None, :]) / dt(K)
    G = np.cos(a + dt(theta))
    np.fill_diagonal(G, dt(0))
    return G


def motor(
    *,
    N: int = 8,
    alpha: float = 1.0,
    seed: int = 1,
    K: int = K_FIRMA,
    theta: float = 0.0,
    MU: float = MU_0,
    ETA: float = ETA_0,
    S0: float = S0_0,
    KAPPA_S: float = KAPPA_S_0,
    W_MIN: float | None = None,
    EPS_TAU: float = EPS_TAU_0,
    k_max: int = K_MAX_0,
    tau_cap: int = TAU_CAP_0,
    barajar: bool = False,
    dtype=np.float64,
    omega_override: np.ndarray | None = None,
    renormalizar: bool = False,
    mortalidad: bool = True,
    guardar_S: bool = False,
) -> dict[str, Any]:
    """Un run de CG002 con la aritmética en `dtype`.

    Réplica literal del bucle de `cg002_acoplamiento.correr` (líneas 327-407), con dos
    modos añadidos que NO cambian el modelo sino la forma de leerlo:

    · `renormalizar=True`: cada paso se divide S por su suma y se acumula el logaritmo.
      Como el mapa es homogéneo de grado 1 (√(s_i s_j) escala como s), esto es una
      reescritura EXACTA de la misma trayectoria en coordenadas (dirección, escala).
      Sirve para medir la tasa de crecimiento asintótica λ sin desbordar el flotante.
    · `mortalidad=False`: apaga el umbral de muerte KAPPA_S. Deja el operador puro, sin
      pisos puestos a mano. Es la condición donde un umbral, si aparece, es ESTRUCTURAL.
    """
    dt = dtype
    W_MIN = 0.1 * S0 if W_MIN is None else W_MIN

    rng = np.random.default_rng(seed)
    omega = rng.integers(0, K, size=N) if omega_override is None else np.asarray(omega_override)
    rng_baraja = np.random.default_rng(SEED_BARAJA + seed)

    s = np.full(N, dt(S0), dtype=dt)
    w_link = np.zeros((N, N), dtype=dt)
    G_fija = _matriz_g(omega, K, theta, dt)

    uno = dt(1.0)
    mu_d, eta_d, alpha_d = dt(MU), dt(ETA), dt(alpha)
    kappa_d, wmin_d, eps_d = dt(KAPPA_S), dt(W_MIN), dt(EPS_TAU)

    tau = 0
    delta_struct = dt(0.0)
    k_micro = 0
    prev_edges: frozenset = frozenset()
    curr_edges: frozenset = frozenset()
    log_escala = 0.0            # log del factor acumulado sacado por la renormalización
    log_ratios: list[float] = []  # log(S_tot(k+1)/S_tot(k)) paso a paso
    S_hist: list[float] = []
    desbordo = False

    while k_micro < k_max and tau < tau_cap:
        vivos = (s > kappa_d) if mortalidad else np.ones(N, dtype=bool)
        if mortalidad and not vivos.any():
            break

        k_micro += 1
        S_ant = float(s.sum())

        s = (uno - mu_d) * s

        if barajar:
            om_eff = rng_baraja.permutation(omega)
            G = _matriz_g(om_eff, K, theta, dt)
        else:
            G = G_fija

        # --- acoplamiento: Δs_i = ETA·α·Σ_j g_ij·√(s_i s_j) ---
        x = np.sqrt(np.maximum(s, dt(0)))
        x = np.where(vivos, x, dt(0))
        Gm = G * np.outer(vivos, vivos)
        if alpha_d == dt(0):
            df = np.zeros((N, N), dtype=dt)
            delta_s = np.zeros(N, dtype=dt)
            sum_abs_f = dt(0)
        else:
            df = alpha_d * Gm * np.outer(x, x)        # df[i,j] = f que i recibe de j
            delta_s = eta_d * df.sum(axis=1)
            sum_abs_f = np.abs(df).sum()

        # --- w_link acumula sólo la parte cooperativa (igual que el original) ---
        coop = (np.maximum(df, dt(0)) + np.maximum(df.T, dt(0))) * dt(0.5)
        mask = (df != dt(0)) | (df.T != dt(0))
        w_link = w_link + np.where(mask & (coop > dt(0)), coop, dt(0))

        s = s + delta_s
        s = np.maximum(s, dt(0))

        vivos_post = (s > kappa_d) if mortalidad else np.ones(N, dtype=bool)
        act = (w_link > wmin_d) & np.outer(vivos_post, vivos_post)
        iu = np.triu_indices(N, 1)
        curr_edges = frozenset(zip(*[a[act[iu]] for a in iu]))
        delta_topo = dt(1.0) if curr_edges != prev_edges else dt(0.0)
        prev_edges = curr_edges

        delta_step = sum_abs_f + np.abs(delta_s).sum() + delta_topo
        delta_struct = delta_struct + delta_step
        if delta_step > eps_d:
            tau += 1

        S_new = float(s.sum())
        if S_ant > 0 and S_new > 0 and math.isfinite(S_new):
            log_ratios.append(math.log(S_new / S_ant))
        if guardar_S:
            S_hist.append(math.log(S_new) + log_escala if S_new > 0 else -math.inf)

        if not np.all(np.isfinite(s)):
            desbordo = True
            break
        if renormalizar:
            tot = s.sum()
            if float(tot) > 0:
                log_escala += math.log(float(tot) / S0)
                s = s * (dt(S0) / tot)
        elif S_new > 1e290:
            desbordo = True
            break

    vivos_fin = (s > kappa_d) if mortalidad else np.ones(N, dtype=bool)
    adj = (w_link > wmin_d) & np.outer(vivos_fin, vivos_fin)
    np.fill_diagonal(adj, False)

    # tasa asintótica: media de los log-cocientes en la última ventana disponible
    vent = log_ratios[-400:] if len(log_ratios) >= 40 else log_ratios
    lam = float(np.mean(vent)) if vent else float("nan")

    return {
        "N": N, "alpha": alpha, "seed": seed, "omega": omega.tolist(),
        "k_micro": k_micro, "tau": tau,
        "delta_struct": float(delta_struct),
        "n_vivos": int(vivos_fin.sum()),
        "n_aristas": int(adj.sum() // 2),
        "S_final_max": float(s.max()),
        "S_final_suma": float(s.sum()),
        "log_S_total": float(math.log(max(float(s.sum()), 1e-320)) + log_escala),
        "lambda": lam,
        "desbordo": desbordo,
        "S_final": np.asarray(s, dtype=np.float64).tolist(),
        "aristas": sorted(curr_edges) if k_micro else [],
        "S_hist": S_hist,
    }


def lambda_max_G(omega, K=K_FIRMA, theta=0.0) -> float:
    """λ_max de la matriz de compatibilidad con diagonal nula. Predice α_c (§0.2)."""
    G = _matriz_g(np.asarray(omega), K, theta, np.float64)
    return float(np.linalg.eigvalsh((G + G.T) / 2)[-1])


def alpha_c_teorico(omega, MU=MU_0, ETA=ETA_0, K=K_FIRMA) -> float:
    lm = lambda_max_G(omega, K)
    if lm <= 0:
        return float("inf")
    return (MU / (1.0 - MU)) / (ETA * lm)


# =========================================================================
# §2 · NÚCLEO en mpmath (precisión arbitraria) — sólo para la guarda F1-3
# =========================================================================
def motor_mp(*, N=8, alpha=1.0, seed=1, K=K_FIRMA, MU=MU_0, ETA=ETA_0, S0=S0_0,
             k_max=400, dps=50, omega_override=None, mortalidad=False,
             KAPPA_S=KAPPA_S_0) -> dict[str, Any]:
    """Mismo micro-paso, en aritmética de `dps` dígitos decimales, con renormalización.

    Lento (escalar), así que se usa sólo en los puntos donde hace falta decidir si un
    umbral es físico o de la máquina."""
    from mpmath import mp, mpf, sqrt as mpsqrt, cos as mpcos, log as mplog, pi as mppi
    mp.dps = dps

    rng = np.random.default_rng(seed)
    omega = rng.integers(0, K, size=N) if omega_override is None else np.asarray(omega_override)
    om = [mpf(str(float(o))) for o in omega]

    G = [[mpcos(2 * mppi * (om[i] - om[j]) / mpf(K)) if i != j else mpf(0)
          for j in range(N)] for i in range(N)]

    s = [mpf(str(S0)) for _ in range(N)]
    mu, eta, al = mpf(str(MU)), mpf(str(ETA)), mpf(str(alpha))
    ks = mpf(str(KAPPA_S))
    log_esc = mpf(0)
    ratios = []

    for _ in range(k_max):
        vivos = [(si > ks) if mortalidad else True for si in s]
        if mortalidad and not any(vivos):
            break
        S_ant = sum(s)
        s = [(1 - mu) * si for si in s]
        x = [mpsqrt(si) if v else mpf(0) for si, v in zip(s, vivos)]
        ds = [mpf(0)] * N
        for i in range(N):
            if not vivos[i]:
                continue
            acc = mpf(0)
            for j in range(N):
                if i != j and vivos[j]:
                    acc += G[i][j] * x[i] * x[j]
            ds[i] = eta * al * acc
        s = [si + di for si, di in zip(s, ds)]
        s = [si if si > 0 else mpf(0) for si in s]
        S_new = sum(s)
        if S_ant > 0 and S_new > 0:
            ratios.append(mplog(S_new / S_ant))
        tot = sum(s)
        if tot > 0:
            log_esc += mplog(tot / mpf(str(S0)))
            s = [si * mpf(str(S0)) / tot for si in s]

    vent = ratios[-200:] if len(ratios) >= 20 else ratios
    lam = float(sum(vent) / len(vent)) if vent else float("nan")
    return {"lambda": lam, "log_S_total": float(log_esc), "dps": dps,
            "S_final_norm": [float(v) for v in s]}


# =========================================================================
# §3 · VERIFICACIÓN del núcleo contra el motor original
# =========================================================================
def verificar_replica() -> dict[str, Any]:
    """El núcleo vectorizado debe reproducir `cg002_acoplamiento.correr`.

    Se compara a HORIZONTE CORTO (k_max pequeño) porque la dinámica es exponencialmente
    inestable: por encima del umbral, dos sumas de los mismos términos en distinto orden
    divergen a horizonte largo aunque el modelo sea el mismo. Ese hecho es, en sí, uno de
    los resultados (ver §RESULTADOS del informe)."""
    from cg002_acoplamiento import CG002Config, correr

    filas = []
    for N in (4, 6, 8):
        for seed in (1, 2, 3, 4, 5):
            for k in (25, 60):
                ref = correr(CG002Config(N=N, alpha=1.0, seed=seed, k_max=k, tau_cap=10_000))
                mio = motor(N=N, alpha=1.0, seed=seed, k_max=k, tau_cap=10_000)
                a = np.asarray(ref["S_final"], float)
                b = np.asarray(mio["S_final"], float)
                den = max(float(np.abs(a).max()), 1e-300)
                filas.append({
                    "N": N, "seed": seed, "k_max": k,
                    "err_rel_S": float(np.abs(a - b).max() / den),
                    "tau_igual": int(ref["tau"] == mio["tau"]),
                    "vivos_igual": int(ref["n_vivos"] == mio["n_vivos"]),
                    "aristas_igual": int(ref["n_aristas"] == mio["n_aristas"]),
                    "err_rel_delta_struct": abs(ref["delta_struct"] - mio["delta_struct"])
                                            / max(abs(ref["delta_struct"]), 1e-300),
                })
    return {
        "n_comparaciones": len(filas),
        "err_rel_S_max": max(f["err_rel_S"] for f in filas),
        "err_rel_delta_struct_max": max(f["err_rel_delta_struct"] for f in filas),
        "frac_tau_igual": sum(f["tau_igual"] for f in filas) / len(filas),
        "frac_vivos_igual": sum(f["vivos_igual"] for f in filas) / len(filas),
        "frac_aristas_igual": sum(f["aristas_igual"] for f in filas) / len(filas),
        "detalle": filas,
    }


def verificar_no_isomorfia() -> dict[str, Any]:
    """Barajar UNA vez = renombrar nodos (isomorfo). Barajar POR PASO = no isomorfo.
    Se repite la guarda del 13-ago sobre el núcleo nuevo."""
    difs = []
    for seed in SEMILLAS:
        a = motor(N=8, alpha=1.0, seed=seed, barajar=False, k_max=300, tau_cap=10_000)
        b = motor(N=8, alpha=1.0, seed=seed, barajar=True, k_max=300, tau_cap=10_000)
        difs.append({
            "seed": seed,
            "S_ordenado_identico": bool(np.allclose(sorted(a["S_final"]), sorted(b["S_final"]))),
            "lambda_real": a["lambda"], "lambda_baraj": b["lambda"],
        })
    return {"frac_identicas": sum(d["S_ordenado_identico"] for d in difs) / len(difs),
            "detalle": difs}


# =========================================================================
# §4 · BARRIDO A / A' — κ_P
# =========================================================================
BRAZOS = {"REAL": (False,), "BARAJADO": (True,)}


def barrido_alpha(alphas: np.ndarray, dtype=np.float64, N=8, k_max=800,
                  semillas=None) -> list[dict]:
    """Para cada α y cada brazo: tasa asintótica λ (modo puro, sin pisos) y
    supervivencia a horizonte finito (modo original, con KAPPA_S y k_max)."""
    semillas = SEMILLAS if semillas is None else semillas
    filas = []
    for alpha in alphas:
        for brazo, (baraj,) in BRAZOS.items():
            for seed in semillas:
                puro = motor(N=N, alpha=float(alpha), seed=seed, barajar=baraj,
                             dtype=dtype, k_max=k_max, tau_cap=10**9,
                             renormalizar=True, mortalidad=False)
                orig = motor(N=N, alpha=float(alpha), seed=seed, barajar=baraj,
                             dtype=dtype, k_max=K_MAX_0, tau_cap=TAU_CAP_0)
                filas.append({
                    "barrido": "A_alpha", "precision": PREC[dtype],
                    "brazo": brazo, "N": N, "semilla": seed,
                    "alpha": float(alpha), "MU": MU_0, "S0": S0_0,
                    "lambda_puro": puro["lambda"],
                    "persiste_puro": int(puro["lambda"] > 0),
                    "n_vivos_fin": orig["n_vivos"], "n_aristas": orig["n_aristas"],
                    "tau_final": orig["tau"], "k_micro": orig["k_micro"],
                    "delta_struct": orig["delta_struct"],
                    "S_final_max": orig["S_final_max"],
                    "sobrevive_finito": int(orig["n_vivos"] >= 1),
                    "sobrevive_ge2": int(orig["n_vivos"] >= 2),
                    "alpha_c_teorico": alpha_c_teorico(orig["omega"]),
                    "lambda_max_G": lambda_max_G(orig["omega"]),
                })
    # brazo ALFA_0 (control G del protocolo): un solo punto, α=0 por definición
    for seed in semillas:
        puro = motor(N=N, alpha=0.0, seed=seed, dtype=dtype, k_max=k_max,
                     tau_cap=10**9, renormalizar=True, mortalidad=False)
        orig = motor(N=N, alpha=0.0, seed=seed, dtype=dtype, k_max=K_MAX_0, tau_cap=TAU_CAP_0)
        filas.append({
            "barrido": "A_alpha", "precision": PREC[dtype], "brazo": "ALFA_0",
            "N": N, "semilla": seed, "alpha": 0.0, "MU": MU_0, "S0": S0_0,
            "lambda_puro": puro["lambda"], "persiste_puro": int(puro["lambda"] > 0),
            "n_vivos_fin": orig["n_vivos"], "n_aristas": orig["n_aristas"],
            "tau_final": orig["tau"], "k_micro": orig["k_micro"],
            "delta_struct": orig["delta_struct"], "S_final_max": orig["S_final_max"],
            "sobrevive_finito": int(orig["n_vivos"] >= 1),
            "sobrevive_ge2": int(orig["n_vivos"] >= 2),
            "alpha_c_teorico": alpha_c_teorico(orig["omega"]),
            "lambda_max_G": lambda_max_G(orig["omega"]),
        })
    return filas


def barrido_mu(mus: np.ndarray, dtype=np.float64, N=8, k_puro=800) -> list[dict]:
    """Si el umbral está en el COCIENTE acoplamiento/decaimiento, mover MU debe moverlo
    de forma exactamente recíproca. Con α fijo=0,05 el umbral esperado es
    MU_c/(1−MU_c) = ETA·α·λ_max(G)."""
    filas = []
    alpha_fijo = 0.05
    for mu in mus:
        for brazo, (baraj,) in BRAZOS.items():
            for seed in SEMILLAS:
                puro = motor(N=N, alpha=alpha_fijo, MU=float(mu), seed=seed, barajar=baraj,
                             dtype=dtype, k_max=k_puro, tau_cap=10**9,
                             renormalizar=True, mortalidad=False)
                filas.append({
                    "barrido": "Ap_mu", "precision": PREC[dtype], "brazo": brazo,
                    "N": N, "semilla": seed, "alpha": alpha_fijo, "MU": float(mu), "S0": S0_0,
                    "lambda_puro": puro["lambda"], "persiste_puro": int(puro["lambda"] > 0),
                    "n_vivos_fin": "", "n_aristas": "", "tau_final": "", "k_micro": "",
                    "delta_struct": "", "S_final_max": "",
                    "sobrevive_finito": "", "sobrevive_ge2": "",
                    "alpha_c_teorico": "", "lambda_max_G": lambda_max_G(puro["omega"]),
                })
    return filas


def umbral_por_biseccion(seed: int, brazo="REAL", dtype=np.float64, k_max=800,
                         lo=1e-4, hi=10.0, iters=45) -> float:
    """α_c empírico: bisección sobre el signo de λ(α). Sin binarizar por ningún umbral
    nuevo — λ=0 es la frontera natural entre crecer y decaer."""
    baraj = BRAZOS[brazo][0]

    def lam(a):
        return motor(N=8, alpha=a, seed=seed, barajar=baraj, dtype=dtype, k_max=k_max,
                     tau_cap=10**9, renormalizar=True, mortalidad=False)["lambda"]

    if lam(lo) > 0 or lam(hi) < 0:
        return float("nan")
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if lam(mid) > 0:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


def umbral_por_biseccion_mp(seed: int, dps=50, k_max=400, lo=1e-4, hi=10.0, iters=30) -> float:
    def lam(a):
        return motor_mp(N=8, alpha=a, seed=seed, k_max=k_max, dps=dps)["lambda"]
    if lam(lo) > 0 or lam(hi) < 0:
        return float("nan")
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if lam(mid) > 0:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


# =========================================================================
# §5 · BARRIDO C — κ_P: ¿hay un piso EN S? (test de tautología)
# =========================================================================
def barrido_S0(s0s: np.ndarray, dtype=np.float64, N=8) -> list[dict]:
    """Dos modos:
      · 'absoluto' : KAPPA_S y W_MIN quedan en su valor original (1e−6 y 0,1).
      · 'escalado' : KAPPA_S y W_MIN se escalan con S0 (misma física, otras unidades).
    Si en modo escalado el resultado es EXACTAMENTE invariante en 15 décadas, entonces
    la escala de S no fija ningún piso y κ_P = inf(S_viable) es 0 (cota trivial)."""
    filas = []
    for s0 in s0s:
        for modo in ("absoluto", "escalado"):
            for brazo, (baraj,) in BRAZOS.items():
                for seed in SEMILLAS:
                    if modo == "absoluto":
                        ks, wm = KAPPA_S_0, W_MIN_0
                    else:
                        ks, wm = KAPPA_S_0 * float(s0), W_MIN_0 * float(s0)
                    r = motor(N=N, alpha=1.0, seed=seed, barajar=baraj, dtype=dtype,
                              S0=float(s0), KAPPA_S=ks, W_MIN=wm,
                              EPS_TAU=EPS_TAU_0 * (float(s0) if modo == "escalado" else 1.0),
                              k_max=K_MAX_0, tau_cap=TAU_CAP_0)
                    filas.append({
                        "barrido": "C_S0", "modo": modo, "precision": PREC[dtype],
                        "brazo": brazo, "semilla": seed, "S0": float(s0),
                        "n_vivos_fin": r["n_vivos"], "n_aristas": r["n_aristas"],
                        "tau_final": r["tau"], "k_micro": r["k_micro"],
                        "sobrevive_finito": int(r["n_vivos"] >= 1),
                        "log_S_total": r["log_S_total"],
                    })
    return filas


# =========================================================================
# §6 · BARRIDO B / B' — κ_Δ: ¿cuál es la diferencia mínima OPERABLE?
# =========================================================================
def barrido_delta(deltas: np.ndarray, dtype=np.float64, N=8, k_max=250,
                  eps_tau=EPS_TAU_0) -> list[dict]:
    """Se toma la configuración ω de una semilla, se le suma δ a la fase de UN nodo y se
    compara con la corrida sin perturbar.

    Dos lecturas, distintas a propósito:
      · 'distinguible' : algún observable continuo cambia (‖ΔS‖/‖S‖ > 0 exactamente).
      · 'operable'     : cambia el estado OPERABLE — conjunto de aristas, nº de vivos o τ.
    El canon pide las dos: "sin el cual nada puede distinguirse NI OPERAR"."""
    filas = []
    for seed in SEMILLAS[:5]:
        rng = np.random.default_rng(seed)
        om0 = rng.integers(0, K_FIRMA, size=N).astype(np.float64)
        base = motor(N=N, alpha=1.0, seed=seed, dtype=dtype, omega_override=om0,
                     k_max=k_max, tau_cap=10**9, EPS_TAU=eps_tau)
        Sb = np.asarray(base["S_final"], float)
        for d in deltas:
            om = om0.copy()
            om[0] = om[0] + float(d)
            r = motor(N=N, alpha=1.0, seed=seed, dtype=dtype, omega_override=om,
                      k_max=k_max, tau_cap=10**9, EPS_TAU=eps_tau)
            Sr = np.asarray(r["S_final"], float)
            den = max(float(np.abs(Sb).max()), 1e-320)
            div = float(np.abs(Sb - Sr).max() / den)
            operable = int(
                (set(map(tuple, r["aristas"])) != set(map(tuple, base["aristas"])))
                or (r["n_vivos"] != base["n_vivos"])
                or (r["tau"] != base["tau"])
            )
            filas.append({
                "barrido": "B_delta", "precision": PREC[dtype],
                "eps_tau": eps_tau, "semilla": seed, "delta": float(d),
                "div_rel_S": div, "distinguible": int(div > 0.0),
                "operable": operable,
                "d_tau": r["tau"] - base["tau"],
                "d_vivos": r["n_vivos"] - base["n_vivos"],
                "d_aristas": r["n_aristas"] - base["n_aristas"],
            })
    return filas


def delta_min(filas: list[dict], campo: str, precision: str, eps_tau=EPS_TAU_0) -> float:
    """δ más chico en el que ≥50% de las semillas ya muestran el efecto."""
    ds = sorted({f["delta"] for f in filas
                 if f["precision"] == precision and f["eps_tau"] == eps_tau})
    for d in ds:
        sub = [f for f in filas if f["delta"] == d and f["precision"] == precision
               and f["eps_tau"] == eps_tau]
        if sub and sum(f[campo] for f in sub) / len(sub) >= 0.5:
            return d
    return float("nan")


def barrido_delta_mp(deltas, dps=40, N=8, k_max=250) -> list[dict]:
    """Misma pregunta en mpmath: si el δ mínimo baja al subir los dígitos, es la máquina."""
    filas = []
    for seed in SEMILLAS[:3]:
        rng = np.random.default_rng(seed)
        om0 = rng.integers(0, K_FIRMA, size=N).astype(np.float64)
        base = motor_mp(N=N, alpha=1.0, seed=seed, k_max=k_max, dps=dps, omega_override=om0)
        Sb = np.asarray(base["S_final_norm"], float)
        for d in deltas:
            om = om0.copy()
            om[0] = om[0] + float(d)
            r = motor_mp(N=N, alpha=1.0, seed=seed, k_max=k_max, dps=dps, omega_override=om)
            Sr = np.asarray(r["S_final_norm"], float)
            div = float(np.abs(Sb - Sr).max() / max(float(np.abs(Sb).max()), 1e-320))
            filas.append({"barrido": "B_delta_mp", "precision": f"mpmath_dps{dps}",
                          "eps_tau": EPS_TAU_0, "semilla": seed, "delta": float(d),
                          "div_rel_S": div, "distinguible": int(div > 0.0),
                          "operable": "", "d_tau": "", "d_vivos": "", "d_aristas": ""})
    return filas


# =========================================================================
# §7 · MAIN
# =========================================================================
def main() -> None:
    t0 = time.time()
    salida: dict[str, Any] = {"guardas": {}}

    print("[0/7] verificando que el núcleo replica el motor original...")
    rep = verificar_replica()
    salida["guardas"]["replica_motor"] = {k: v for k, v in rep.items() if k != "detalle"}
    print("      err_rel_S_max =", rep["err_rel_S_max"],
          "| discretos iguales:", rep["frac_tau_igual"], rep["frac_vivos_igual"],
          rep["frac_aristas_igual"])

    print("[1/7] guarda de no-isomorfía del barajado...")
    iso = verificar_no_isomorfia()
    salida["guardas"]["no_isomorfia"] = {"frac_identicas": iso["frac_identicas"]}
    print("      frac semillas idénticas (debe ser 0.0):", iso["frac_identicas"])

    # ---------- BARRIDO A: α en 7 décadas ----------
    print("[2/7] BARRIDO A — α en 7 décadas x 2 precisiones ...")
    alphas = np.unique(np.concatenate([
        np.logspace(-6, 1, 29),                          # rejilla gruesa, 7 décadas
        np.logspace(np.log10(0.015), np.log10(0.25), 17),  # rejilla fina en la zona predicha
    ]))
    filas_A = barrido_alpha(alphas, dtype=np.float64)
    # float80 sobre la mitad de los puntos y 5 semillas: la curva en la segunda precisión
    # es una GUARDA de forma; la posición exacta del umbral se decide por bisección (§4).
    filas_A += barrido_alpha(alphas[::2], dtype=np.longdouble, semillas=SEMILLAS[:5])
    print(f"      {len(filas_A)} corridas  ({time.time()-t0:.0f}s)")

    # ---------- BARRIDO A': MU en 4 décadas ----------
    print("[3/7] BARRIDO A' — MU en 4 décadas ...")
    mus = np.logspace(-5, -1, 21)
    filas_Ap = barrido_mu(mus, dtype=np.float64)

    # ---------- umbrales por bisección en 3 precisiones ----------
    print("[4/7] umbral α_c por bisección: float64 / float80 / mpmath50 ...")
    umbrales = []
    for seed in SEMILLAS[:5]:
        om = np.random.default_rng(seed).integers(0, K_FIRMA, size=8)
        fila = {
            "semilla": seed,
            "lambda_max_G": lambda_max_G(om),
            "alpha_c_teorico": alpha_c_teorico(om),
            "alpha_c_float64": umbral_por_biseccion(seed, "REAL", np.float64),
            "alpha_c_float80": umbral_por_biseccion(seed, "REAL", np.longdouble),
            "alpha_c_REAL_k400": umbral_por_biseccion(seed, "REAL", np.float64, k_max=400),
            "alpha_c_BARAJADO_float64": umbral_por_biseccion(seed, "BARAJADO", np.float64),
            "alpha_c_BARAJADO_float80": umbral_por_biseccion(seed, "BARAJADO", np.longdouble),
        }
        fila["alpha_c_mpmath50"] = umbral_por_biseccion_mp(seed, dps=50)
        umbrales.append(fila)
        print(f"      semilla {seed}: teor={fila['alpha_c_teorico']:.6g} "
              f"f64={fila['alpha_c_float64']:.6g} f80={fila['alpha_c_float80']:.6g} "
              f"mp50={fila['alpha_c_mpmath50']:.6g} | baraj f64={fila['alpha_c_BARAJADO_float64']:.6g}")
    salida["umbrales_alpha_c"] = umbrales

    # ---------- BARRIDO C: S0 en 15 décadas ----------
    print(f"[5/7] BARRIDO C — S0 en 15 décadas ...  ({time.time()-t0:.0f}s)")
    s0s = np.logspace(-12, 3, 16)
    filas_C = barrido_S0(s0s, dtype=np.float64)

    # ---------- BARRIDO B: δ en 20 décadas ----------
    print(f"[6/7] BARRIDO B — δ en 20 décadas x 3 precisiones ...  ({time.time()-t0:.0f}s)")
    deltas = np.logspace(-20, 0, 41)
    filas_B = barrido_delta(deltas, dtype=np.float64)
    filas_B += barrido_delta(deltas, dtype=np.longdouble)
    filas_B += barrido_delta(deltas, dtype=np.float64, eps_tau=1e-12)
    filas_B += barrido_delta(deltas, dtype=np.float64, eps_tau=1e-1)
    filas_B += barrido_delta_mp(np.logspace(-20, 0, 21), dps=40)

    salida["kappa_delta"] = {
        "delta_min_distinguible_float64": delta_min(filas_B, "distinguible", "float64"),
        "delta_min_distinguible_float80": delta_min(filas_B, "distinguible", "float80"),
        "delta_min_distinguible_mpmath40": delta_min(filas_B, "distinguible", "mpmath_dps40"),
        "delta_min_operable_float64": delta_min(filas_B, "operable", "float64"),
        "delta_min_operable_float80": delta_min(filas_B, "operable", "float80"),
        "delta_min_operable_epsTau_1e-12": delta_min(filas_B, "operable", "float64", 1e-12),
        "delta_min_operable_epsTau_1e-1": delta_min(filas_B, "operable", "float64", 1e-1),
        "eps_maquina": {"float64": float(np.finfo(np.float64).eps),
                        "float80": float(np.finfo(np.longdouble).eps),
                        "mpmath_dps40": 1e-40},
        "kappa_delta_del_alfabeto_K8": float(1 - math.cos(2 * math.pi / K_FIRMA)),
    }

    # ---------- §7 guardas cuantitativas ----------
    print(f"[7/7] guardas: identidades algebraicas y piso de ruido ...  ({time.time()-t0:.0f}s)")
    from scipy.stats import spearmanr, pearsonr, fisher_exact
    sub = [f for f in filas_A if f["precision"] == "float64" and f["brazo"] == "REAL"]
    ident = {}
    for var in ("n_vivos_fin", "S_final_max", "delta_struct", "n_aristas", "tau_final",
                "k_micro", "alpha"):
        xs = [f["lambda_puro"] for f in sub]
        ys = [float(f[var]) for f in sub]
        ok = [(a, b) for a, b in zip(xs, ys) if math.isfinite(a) and math.isfinite(b)]
        if len(ok) > 3:
            a, b = zip(*ok)
            ident[f"lambda_vs_{var}"] = {"spearman": float(spearmanr(a, b)[0]),
                                         "pearson": float(pearsonr(a, b)[0])}
    # α_c medido vs las mismas variables (a nivel de semilla)
    ac = [u["alpha_c_float64"] for u in umbrales]
    lm = [u["lambda_max_G"] for u in umbrales]
    ident["alpha_c_vs_lambda_max_G"] = {"spearman": float(spearmanr(ac, lm)[0]),
                                        "pearson": float(pearsonr(ac, lm)[0])}
    salida["guardas"]["identidades"] = ident
    salida["guardas"]["piso_de_ruido"] = {
        "n_semillas_por_brazo_por_punto": len(SEMILLAS),
        "p_minimo_fisher_2x2_10vs10": float(fisher_exact([[10, 0], [0, 10]],
                                                         alternative="greater")[1]),
        "nota": ("Con 10 semillas por brazo y por punto del barrido, ningún test 2x2 puede "
                 "dar un p menor que éste, ni con separación total. Las curvas se reportan "
                 "crudas; los umbrales se obtienen por bisección del signo de λ, sin "
                 "binarizar por ningún umbral nuevo."),
    }

    # ---------- salidas ----------
    def _dump(nombre, filas):
        cols = sorted({k for f in filas for k in f})
        with (RAIZ / nombre).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for f in filas:
                w.writerow({c: f.get(c, "") for c in cols})
        print("      ->", nombre, len(filas), "filas")

    _dump("kappas_mortalidad_curva_alpha.csv", filas_A + filas_Ap)
    _dump("kappas_mortalidad_curva_S0.csv", filas_C)
    _dump("kappas_mortalidad_curva_delta.csv", filas_B)

    with (RAIZ / "kappas_mortalidad_resumen.json").open("w", encoding="utf-8") as fh:
        json.dump(salida, fh, indent=2, ensure_ascii=False, default=str)

    _figura(filas_A, filas_Ap, filas_C, filas_B, umbrales)
    print(f"LISTO en {time.time()-t0:.0f}s")


def _figura(filas_A, filas_Ap, filas_C, filas_B, umbrales) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COL = {"REAL": "#2563eb", "BARAJADO": "#d97706", "ALFA_0": "#6b7280"}
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) λ(α) — la curva completa, 7 décadas
    a0 = ax[0, 0]
    for brazo in ("REAL", "BARAJADO"):
        for prec, ls in (("float64", "-"), ("float80", "--")):
            sub = [f for f in filas_A if f["brazo"] == brazo and f["precision"] == prec
                   and f["barrido"] == "A_alpha"]
            xs = sorted({f["alpha"] for f in sub})
            ys = [np.mean([f["lambda_puro"] for f in sub if f["alpha"] == x]) for x in xs]
            a0.plot(xs, ys, ls, color=COL[brazo], lw=1.6,
                    label=f"{brazo} · {'f64' if prec=='float64' else 'f80'}")
    a0.axhline(0, color="k", lw=0.8)
    a0.set_xscale("log")
    a0.set_xlabel("α (acoplamiento)")
    a0.set_ylabel("λ = tasa de crecimiento por paso")
    a0.set_title("(a) κ_P — curva completa de λ(α), 7 décadas, 2 precisiones")
    a0.legend(fontsize=7)
    a0.grid(alpha=0.25)

    # (b) supervivencia a horizonte finito vs criterio asintótico
    a1 = ax[0, 1]
    for brazo in ("REAL", "BARAJADO"):
        sub = [f for f in filas_A if f["brazo"] == brazo and f["precision"] == "float64"
               and f["barrido"] == "A_alpha"]
        xs = sorted({f["alpha"] for f in sub})
        a1.plot(xs, [np.mean([f["sobrevive_finito"] for f in sub if f["alpha"] == x]) for x in xs],
                "-", color=COL[brazo], lw=1.8, label=f"{brazo}: vivo a k=1500 (reloj)")
        a1.plot(xs, [np.mean([f["persiste_puro"] for f in sub if f["alpha"] == x]) for x in xs],
                ":", color=COL[brazo], lw=1.8, label=f"{brazo}: λ>0 (asintótico)")
    a1.set_xscale("log")
    a1.set_xlabel("α")
    a1.set_ylabel("fracción de semillas")
    a1.set_title("(b) el criterio de horizonte finito NO es el piso")
    a1.legend(fontsize=7)
    a1.grid(alpha=0.25)

    # (c) κ_P no está en S: S0 en 15 décadas
    a2 = ax[1, 0]
    for modo, mk in (("absoluto", "o-"), ("escalado", "s--")):
        sub = [f for f in filas_C if f["modo"] == modo and f["brazo"] == "REAL"]
        xs = sorted({f["S0"] for f in sub})
        a2.plot(xs, [np.mean([f["sobrevive_finito"] for f in sub if f["S0"] == x]) for x in xs],
                mk, color="#2563eb" if modo == "escalado" else "#dc2626", lw=1.6,
                ms=4, label=f"REAL · pisos {modo}")
    a2.set_xscale("log")
    a2.set_xlabel("S0 (persistencia inicial)")
    a2.set_ylabel("fracción con ≥1 nodo vivo")
    a2.set_title("(c) test de tautología: ¿hay piso EN S?")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.25)

    # (d) κ_Δ: δ mínimo distinguible / operable vs precisión
    a3 = ax[1, 1]
    for prec, col in (("float64", "#dc2626"), ("float80", "#2563eb"),
                      ("mpmath_dps40", "#059669")):
        sub = [f for f in filas_B if f["precision"] == prec and f["eps_tau"] == EPS_TAU_0]
        if not sub:
            continue
        xs = sorted({f["delta"] for f in sub})
        a3.plot(xs, [np.mean([f["distinguible"] for f in sub if f["delta"] == x]) for x in xs],
                "-", color=col, lw=1.6, label=f"{prec}: distinguible")
        if sub[0]["operable"] != "":
            a3.plot(xs, [np.mean([f["operable"] for f in sub if f["delta"] == x]) for x in xs],
                    "--", color=col, lw=1.4, label=f"{prec}: operable")
    for e, c in ((np.finfo(np.float64).eps, "#dc2626"),
                 (float(np.finfo(np.longdouble).eps), "#2563eb"), (1e-40, "#059669")):
        a3.axvline(e, color=c, lw=0.7, alpha=0.5)
    a3.set_xscale("log")
    a3.set_xlabel("δ (diferencia de fase inyectada)")
    a3.set_ylabel("fracción de semillas que la ven")
    a3.set_title("(d) κ_Δ — las verticales son los ε de máquina")
    a3.legend(fontsize=7)
    a3.grid(alpha=0.25)

    fig.suptitle("κ_P y κ_Δ en CG002 (instrumento con mortalidad) — 13-ago-2026 · NO ES UN CIERRE",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RAIZ / "kappas_mortalidad_curvas.png", dpi=140)
    print("      -> kappas_mortalidad_curvas.png")


if __name__ == "__main__":
    main()
