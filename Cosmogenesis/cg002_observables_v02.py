"""
cg002_observables_v02.py
========================
Observables y mecanismo de ruptura de simetría para CG002 v0.2.

Propuesta de integración (Claude Web -> Grok/Diotallevi), 2026-06-29.
Deriva del diagnóstico verificado en sesión: el grafo de v0.1 es SIMÉTRICO
(Fij = Fji) y la firma ℤ_K es de RANGO 2 (coseno de una sola fase). Eso
implica, por construcción y antes de cualquier bug:

  · no hay T+/T- que medir (grafo no dirigido)            -> dirección muerta
  · el embedding emergente está topeado a 2D              -> C-N2.6 capada
  · la frustración triádica no existe en K=8 (sí en N>=4) -> C-N2.5.10 escondida

Este módulo NO reescribe el motor. Aporta:
  1. Observable de FRUSTRACIÓN (sustrato de C-N2.5.10).
  2. Observable de EMBEDDING / dimensión efectiva (sustrato de C-N2.6).
  3. Acoplamiento DIRIGIDO theta_CP: la 'violación CP' cosmosemiótica
     (C-N2.5.5), única fuente de ruptura de simetría que da sustrato real
     a la dirección T+/T- (C-N2.5.7) y a la cancelación (C-N2.5.10).
  4. Firma multicomponente (esfera S^{d-1}) que sube el techo de rango.

theta_CP = 0  reproduce v0.1 exactamente -> control nulo de la dirección.
Todas las funciones son puras (leen estado, no lo escriben).
"""

import numpy as np
from itertools import combinations


# =====================================================================
# 1. FRUSTRACIÓN  ->  sustrato de C-N2.5.10 (cancelación de orientaciones)
# =====================================================================
def laplaciano_signado_lambda_min(W_signed):
    """
    W_signed: matriz de adyacencia SIGNADA y simétrica.
              Wij = flujo de acoplamiento acumulado y SIGNADO entre i,j
              (cooperación > 0, competencia < 0). DEBE conservar el signo:
              si se usa |F| el observable queda ciego (todo 'balanceado').

    Devuelve lambda_min(L) >= 0 del Laplaciano signado L = Dbar - W,
    con Dbar_ii = sum_j |Wij|.

    Teorema de balance (Heider/Kunegis):
        lambda_min = 0  <=>  el grafo es BALANCEADO (frustración nula)
        lambda_min > 0  <=>  hay FRUSTRACIÓN (tensión no satisfacible)

    Interpretación cosmosemiótica: una configuración frustrada no puede
    estabilizarse cooperando -> algún nodo cae a kappa_s. C-N2.5.10 predice
    que la aniquilación ES la frustración resolviéndose.
    """
    W = (np.asarray(W_signed, float) + np.asarray(W_signed, float).T) / 2.0
    Dbar = np.diag(np.abs(W).sum(axis=1))
    L = Dbar - W
    return float(np.linalg.eigvalsh(L)[0])


def es_balanceado(W_signed, tol=1e-8):
    return laplaciano_signado_lambda_min(W_signed) < tol


# =====================================================================
# 2. EMBEDDING ESPECTRAL  ->  sustrato de C-N2.6 (curvatura sin coordenadas)
# =====================================================================
def embedding_espectral(W_abs, n_dim=3):
    """
    W_abs: adyacencia de PESOS (>=0, p.ej. |F| acumulado o w_link).
    Devuelve posiciones emergentes de los nodos como autovectores bajos
    (>0) del Laplaciano de grafo NO signado L = D - W_abs. Geometría que
    EMERGE del enredo, no coordenada importada (C-N2.6).
    """
    W = np.asarray(W_abs, float)
    D = np.diag(W.sum(axis=1))
    L = D - W
    evals, evecs = np.linalg.eigh((L + L.T) / 2)
    idx = np.argsort(evals)
    pos = evecs[:, idx[1:1 + n_dim]]  # saltar el modo trivial lambda_0=0
    return pos


def rango_estructural(C, tol=1e-9):
    """Techo DURO de la dimensión del embedding: rango de la matriz de
    compatibilidad C. Firma de 1 fase (ℤ_K) -> 2 para todo N,K."""
    ev = np.linalg.eigvalsh((C + C.T) / 2)
    return int(np.sum(np.abs(ev) > tol))


def dimension_efectiva(C):
    """Dimensión que la estructura USA (participation ratio del |espectro|).
    <= rango_estructural; depende de la muestra. Reportar junto al rango."""
    ev = np.abs(np.linalg.eigvalsh((C + C.T) / 2))
    ev = ev[ev > 1e-9]
    if ev.size == 0:
        return 0.0
    return float((ev.sum() ** 2) / (np.sum(ev ** 2)))


# =====================================================================
# 3. ACOPLAMIENTO DIRIGIDO theta_CP  ->  'violación CP' cosmosemiótica
#    C-N2.5.5 (asimetría primordial) -> sustrato real de C-N2.5.7 / C-N2.5.10
# =====================================================================
def g_dirigido(wi, wj, K, theta_cp):
    """
    Acoplamiento que i recibe de j, y que j recibe de i, por separado.
        g_{i<-j} = cos(2*pi*(wi-wj)/K + theta_cp)
        g_{j<-i} = cos(2*pi*(wj-wi)/K + theta_cp)
    theta_cp = 0  =>  g_{i<-j} = g_{j<-i}  =>  v0.1 EXACTO (simétrico).
    theta_cp != 0 =>  Fij != Fji  =>  grafo orientado genuino.

    theta_cp es el ÚNICO parámetro nuevo. Es la fuente de ruptura de
    simetría. Su control nulo es theta_cp = 0 (G').
    """
    a = 2 * np.pi * (wi - wj) / K
    return np.cos(a + theta_cp), np.cos(-a + theta_cp)


def paso_acoplamiento_dirigido(S, omegas, K, eta, theta_cp, kappa_s=1e-6):
    """
    Un micro-paso del acoplamiento dirigido (drop-in para §5.1 con dirección).
    Reemplaza el update simétrico por uno donde cada nodo recibe su propio
    flujo. theta_cp=0 -> idéntico a v0.1. Devuelve (S_nuevo, W_orient).

    W_orient[i,j] = flujo SIGNADO que i recibe de j en este paso
                    (no simétrico si theta_cp != 0).
    """
    S = np.asarray(S, float).copy()
    N = len(S)
    W_orient = np.zeros((N, N))
    dS = np.zeros(N)
    vivos = S > kappa_s
    for i, j in combinations(range(N), 2):
        if not (vivos[i] and vivos[j]):
            continue
        gi, gj = g_dirigido(omegas[i], omegas[j], K, theta_cp)
        mag = np.sqrt(S[i] * S[j])
        f_i = eta * gi * mag
        f_j = eta * gj * mag
        dS[i] += f_i
        dS[j] += f_j
        W_orient[i, j] = f_i
        W_orient[j, i] = f_j
    return S + dS, W_orient


def vector_predictor(omegas, K, theta_cp, eps_deg=1e-3):
    """
    Predictor de eje orientacional (solo ω, K, θ_CP).
    θ va FUERA del seno: v_pred_i = −sign(θ_CP) · Σ_j sin(2π(ωᵢ−ωⱼ)/K).
    (CC Mensaje 05 — coherente con flujo_neto del motor g_dirigido.)
    """
    if abs(theta_cp) < eps_deg:
        return np.zeros(len(omegas), dtype=np.float64), 0.0
    omg = np.asarray(omegas, dtype=np.float64)
    n = len(omg)
    v = np.zeros(n, dtype=np.float64)
    sgn = -1.0 if theta_cp > 0 else 1.0
    for i in range(n):
        for j in range(n):
            if i != j:
                v[i] += sgn * np.sin(2 * np.pi * (omg[i] - omg[j]) / K)
    return v, float(np.linalg.norm(v))


def tracking_score(flujo_neto, v_pred, eps_deg=1e-3, eps_norm=1e-12):
    """
    Proyección predict-then-verify (v0.1c).
    Retorna (score, degenerada). score=None si degenerada.
    """
    v_real = np.asarray(flujo_neto, dtype=np.float64)
    v_pred = np.asarray(v_pred, dtype=np.float64)
    n_pred = float(np.linalg.norm(v_pred))
    if n_pred < eps_deg:
        return None, True
    n_real = float(np.linalg.norm(v_real))
    if n_real < eps_norm:
        return 0.0, False
    return float(np.dot(v_real, v_pred) / (n_real * n_pred)), False


def imbalance_orientado(W_orient, eps=1e-12):
    """
    rho y |T_hat| sobre el grafo ORIENTADO (W_orient no simétrico).
    Mide la asimetría real de flujo, no una convención de almacenamiento.

    rho = fracción cooperativa neta en [0,1] (escalar, NO dirección).
    flujo_neto_i = (entrante - saliente) por nodo: vector de orientación.
    asym = ‖W - Wᵀ‖/‖W‖ : 0 si el grafo es simétrico (sin dirección que medir).
    """
    W = np.asarray(W_orient, float)
    asym = np.linalg.norm(W - W.T) / (np.linalg.norm(W) + eps)
    rho = W[W > 0].sum() / (np.abs(W).sum() + eps)
    flujo_neto = W.sum(axis=1) - W.sum(axis=0)
    T_hat_mod = float(np.linalg.norm(flujo_neto))
    return {
        "asimetria": float(asym),       # 0 => no hay dirección (caso v0.1)
        "rho": float(rho),
        "T_hat_mod": T_hat_mod,
        "flujo_neto": flujo_neto,
    }


# =====================================================================
# 4. FIRMAS  ->  techo de rango (decoupla 'riqueza' de 'dirección')
# =====================================================================
def matriz_compat_circulo(omegas, K):
    """v0.1: 1 fase en ℤ_K. C = <u_i,u_j>, u en círculo R^2. RANGO 2 SIEMPRE."""
    th = 2 * np.pi * np.asarray(omegas) / K
    return np.cos(th[:, None] - th[None, :])


def matriz_compat_esfera(U):
    """v0.2 opcional: firma multicomponente. U:(N,d) -> S^{d-1}. RANGO hasta d.
    Sube el techo de embedding (C-N2.6) y habilita frustración más rica.
    Ortogonal a theta_CP: 'cuánta estructura' (d) vs '¿hay flecha?' (theta_CP)."""
    U = np.asarray(U, float)
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    return U @ U.T


# =====================================================================
# Autotest
# =====================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    W_bal = np.array([[0, .71, -1], [.71, 0, -.71], [-1, -.71, 0]])
    W_fru = np.array([[0, 1, 1], [1, 0, -1], [1, -1, 0]])
    print("[1] frustración: balanceada=%.3f  frustrada=%.3f"
          % (laplaciano_signado_lambda_min(W_bal),
             laplaciano_signado_lambda_min(W_fru)))

    omg = rng.integers(0, 8, size=12)
    C = matriz_compat_circulo(omg, 8)
    print("[2] círculo ℤ_8: rango=%d dim_efectiva=%.2f  | esfera S^2: rango=%d"
          % (rango_estructural(C), dimension_efectiva(C),
             rango_estructural(matriz_compat_esfera(rng.standard_normal((12, 3))))))

    S = np.ones(8)
    for th in (0.0, 0.5):
        _, W = paso_acoplamiento_dirigido(S, omg, 8, 0.05, th)
        m = imbalance_orientado(W)
        print("[3] theta_CP=%.1f: asimetria=%.3f T_hat=%.3f"
              % (th, m["asimetria"], m["T_hat_mod"]))
    print("OK")
