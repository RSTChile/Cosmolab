"""
CG002 v0.2 — FIRMA MULTICOMPONENTE S^{d-1}  (Claude CC, 29-jun-2026)
====================================================================
Sube el techo de RANGO RELACIONAL. En v0.1c la firma es 1 fase (ω∈ℤ_K) →
C = cos(...) tiene RANGO 2 SIEMPRE → la dimensión relacional emergente está
topada en 2 (el "3D" del visor génesis era solo layout). Con firma U∈S^{d-1}
(d componentes), C = U·Uᵀ tiene rango hasta d → la dimensión 3D puede ser
INTRÍNSECA (output medido), no pintada.

NO modifica el motor verificado v0.1c (cg002_acoplamiento.py, B PASS cualificado).
Reutiliza la MISMA dinámica de persistencia (η, μ, κ_s, banda C-N5.1).

Decisión de diseño (registrada en PROTOCOLO_CG002_v02_ADDENDUM.md):
 - Firma: U_i ∈ S^{d-1} (vector unitario en R^d), sorteado por semilla.
 - Compatibilidad base (simétrica): c_ij = U_i · U_j ∈ [−1,1].
 - Acoplamiento DIRIGIDO (θ_CP): rotación R(θ_CP) en el plano (e0,e1):
       g_{i←j} = U_i · R(θ_CP) · U_j ;  g_{j←i} = U_i · R(−θ_CP) · U_j
   θ_CP=0 ⇒ R=I ⇒ simétrico (control G′ exacto). Recupera ℤ₈ con d=2.
   Asimetría = g_{i←j}−g_{j←i} = 2·sinθ·(u_i0·u_j1 − u_i1·u_j0)  (antisimétrica).
 - d (riqueza estructural) y θ_CP (flecha) son ORTOGONALES (propuesta §2.5).
"""
from __future__ import annotations
import numpy as np

ETA, MU, KAPPA_S, S0, S_BAND = 0.05, 0.01, 1e-6, 1.0, 8.0

def _rot(d, theta):
    R = np.eye(d)
    c, s = np.cos(theta), np.sin(theta)
    R[0,0]=c; R[0,1]=-s; R[1,0]=s; R[1,1]=c
    return R

def _sat(S): return S/(1.0+S/S_BAND)

def rango_estructural(C, tol=1e-9):
    ev = np.linalg.eigvalsh((C+C.T)/2)
    return int(np.sum(np.abs(ev) > tol))

def dimension_efectiva(C):
    ev = np.abs(np.linalg.eigvalsh((C+C.T)/2)); ev = ev[ev>1e-9]
    return float(ev.sum()**2/np.sum(ev**2)) if ev.size else 0.0

def correr(N=300, d=3, theta_cp=0.3, seed=1, pasos=160):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((N, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    R = _rot(d, theta_cp)
    S = np.full(N, S0); alive = np.ones(N, bool)
    W = np.zeros((N, N))
    G = U @ (R @ U.T)                       # G[i,j] = g_{i←j} = U_i · R · U_j
    np.fill_diagonal(G, 0.0)
    asym = float(np.linalg.norm(G - G.T) / (np.linalg.norm(G) + 1e-12))
    for _ in range(pasos):
        S = np.where(alive, S*(1-MU), S)
        ss = np.where(alive, _sat(S), 0.0)
        M = np.sqrt(np.outer(ss, ss))       # magnitud √(sat Si · sat Sj)
        F = ETA * G * M                      # F[i,j] = flujo que i recibe de j
        dS = F.sum(axis=1)
        S = S + dS
        S = np.where(S < 0, 0.0, S)
        W += F
        alive = S > KAPPA_S
        S = np.where(alive, S, 0.0)
    idx = np.where(alive)[0]
    n = len(idx)
    if n < 3:
        return dict(d=d, theta=theta_cp, n=n, rango=0, dim=0.0, asym=asym)
    Calive = (U[idx] @ U[idx].T)
    return dict(d=d, theta=theta_cp, n=n,
                rango=rango_estructural(Calive),
                dim=round(dimension_efectiva(Calive), 2),
                asym=round(asym, 3))

if __name__ == "__main__":
    print("=== CG002 v0.2 — rango relacional vs dimensión de firma d (N=300, θ_CP=0.3, prom seeds 1-3) ===")
    print("d\tn_vivos\trango(C)\tdim_efectiva\tasym(θ=0.3)")
    for d in (2, 3, 4, 5):
        rs, ds, ns, ay = [], [], [], []
        for sd in (1, 2, 3):
            r = correr(N=300, d=d, theta_cp=0.3, seed=sd)
            rs.append(r["rango"]); ds.append(r["dim"]); ns.append(r["n"]); ay.append(r["asym"])
        print(f"{d}\t{round(np.mean(ns))}\t{round(np.mean(rs),1)}\t\t{round(np.mean(ds),2)}\t\t{round(np.mean(ay),3)}")
    print("\n=== Control G' — θ_CP=0 debe dar asimetría = 0 (sin flecha), rango = d ===")
    for d in (3, 4):
        r = correr(N=300, d=d, theta_cp=0.0, seed=1)
        print(f"d={d}: asym={r['asym']:.3e}  rango={r['rango']}  dim_efectiva={r['dim']}  vivos={r['n']}")
