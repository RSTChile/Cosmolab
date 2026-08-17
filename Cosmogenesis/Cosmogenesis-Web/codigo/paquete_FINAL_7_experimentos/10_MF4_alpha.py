"""
MF-4 alpha_EM - alpha^-1 ~ K^-1 * a^beta * (T0/T)
Contrato: a=exp(6tg) T=T0/a K=exp(-1/a)
Objetivo: ver corrida alpha 137 con expansion, formula unica sin if k
"""
import numpy as np
T0=1e15; rho0=1e30; L=30; pasos=500
rng=np.random.default_rng(2025)
phi=np.ones((L,L))+0.5*rng.normal(size=(L,L))
ar=np.ones((L,L),bool); ad=np.ones((L,L),bool)
alpha0=1/137.035999
print("MF-4 alpha_EM corrida")
for step in range(pasos):
    tg=step/pasos; a=np.exp(6*tg); T=T0/a
    K=np.exp(-1/(a+1e-10))
    # modelo: alpha = alpha0 * K * (T/T0)^0.5 * a^0.1  -> unica
    beta=0.1
    alpha = alpha0 * K * np.sqrt(T/T0) * (a**beta)
    alpha_inv = 1/alpha if alpha>0 else 0
    if step%50==0:
        print(f"step {step:3d} a={a:6.1f} T={T:.2e} K={K:.3f} alpha={alpha:.3e} alpha^-1={alpha_inv:.1f} objetivo ~137")
print("FIN MF-4: alpha^-1 corre 137*? K domina 0.368->1, sqrt(T/T0)=a^-0.5 decrece")
