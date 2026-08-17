"""
CG002 — ¿Emerge baryogénesis? (Claude CC, 29-jun-2026)
NO programamos aniquilación. Damos la posibilidad: firmas U∈S^{d-1} pueden caer
en cualquier dirección, incluidas antípodas (U vs −U = competencia máxima -1).
La aniquilación EMERGE del acoplamiento (anti-alineados se matan). Vemos qué pasa.

Predicción C-N2.5.10: simetría perfecta (cada U con su −U) → campo medio v=0 →
universo vacío. Asimetría (aunque estadística) → v≠0 → exceso sobrevive (materia).
Física = misma regla validada; cálculo O(N) por factorización de campo medio.
"""
import numpy as np
ETA,MU,KAPPA_S,S0,S_BAND=0.05,0.01,1e-6,1.0,8.0
def sat(S): return S/(1.0+S/S_BAND)
def rot(d,th):
    R=np.eye(d); c,s=np.cos(th),np.sin(th); R[0,0]=c;R[0,1]=-s;R[1,0]=s;R[1,1]=c; return R
def correr(U,theta,pasos=240):
    N,d=U.shape; R=rot(d,theta); S=np.full(N,S0); alive=np.ones(N,bool)
    UR=U@R                                   # fila i = U_i·R
    selfdrag=np.einsum('ij,ij->i',UR,U)      # U_i·R·U_i
    for _ in range(pasos):
        S=np.where(alive,S*(1-MU),S)
        m=np.where(alive,np.sqrt(sat(S)),0.0)
        v=(m[:,None]*U).sum(0)               # campo medio Σ m_j U_j  (d,)
        dS=ETA*m*(UR@v) - ETA*m*m*selfdrag   # mean-field menos auto-término
        S=S+dS; S=np.where(S<0,0.0,S); alive=S>KAPPA_S; S=np.where(alive,S,0.0)
    return S,alive
def medir(U,S,alive):
    N=len(S); idx=np.where(alive)[0]; n=len(idx)
    a0=np.linalg.norm(U.mean(0))             # asimetría primordial (sorteo)
    a1=np.linalg.norm(U[idx].mean(0)) if n else 0.0  # orientación de los sobrevivientes
    return n/N, a0, a1

print("=== Damos la posibilidad (firmas aleatorias en S^2) — vemos qué sobrevive ===")
print("d=3, N=2000, 240 pasos. a0=asimetría inicial (sorteo), a1=orientación de sobrevivientes")
for th in (0.0,0.3):
    print(f"\n--- θ_CP={th} ---")
    print("seed\tfrac_viva\ta0(inicial)\ta1(sobrevivientes)\tamplificación a1/a0")
    for sd in (1,2,3):
        rng=np.random.default_rng(sd); U=rng.standard_normal((2000,3)); U/=np.linalg.norm(U,axis=1,keepdims=True)
        S,al=correr(U,th); fv,a0,a1=medir(U,S,al)
        print(f"{sd}\t{fv*100:.1f}%\t\t{a0:.4f}\t\t{a1:.4f}\t\t{a1/(a0+1e-9):.1f}×")

print("\n=== CONTROL: simetría perfecta (cada U con su −U) → ¿universo vacío? ===")
print("seed\tfrac_viva\ta0(inicial)\ta1")
for sd in (1,2,3):
    rng=np.random.default_rng(sd); H=rng.standard_normal((1000,3)); H/=np.linalg.norm(H,axis=1,keepdims=True)
    U=np.vstack([H,-H])                       # 1000 pares antípoda exactos -> a0=0
    S,al=correr(U,0.3); fv,a0,a1=medir(U,S,al)
    print(f"{sd}\t{fv*100:.1f}%\t\t{a0:.2e}\t{a1:.4f}")
