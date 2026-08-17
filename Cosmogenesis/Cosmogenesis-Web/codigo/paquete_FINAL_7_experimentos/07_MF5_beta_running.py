"""
07_MF5_beta_running - acoplamiento g(a) = g0 / log(a/a0) , K = exp(-1/(g^2 a))
Tension sigma = Lambda^2 * g^2, Lambda = rho0^{1/4}/a
"""
import numpy as np
T0=1e15; rho0=1e30; H_topo=0.01; L=30; pasos=500
rng=np.random.default_rng(2025)
phi=np.ones((L,L))+0.5*rng.normal(size=(L,L))
ar=np.ones((L,L),bool); ad=np.ones((L,L),bool)
print("07 MF5 beta running g(a)=g0/log(a) K=exp(-1/g^2 a) sigma=Lambda^2 g^2")
g0=0.5
for step in range(pasos):
    tg=step/pasos; a=np.exp(6*tg); T=T0/a
    g = g0/np.log(a+2)  # running, decrece con a
    K = np.exp(-1.0/(g**2 * a + 1e-10))
    rho=rho0/(a**3); Lambda = (rho)**0.25
    sigma = Lambda**2 * g**2
    if step%100==0:
        print(f"step {step:3d} a={a:5.1f} g={g:.3f} K={K:.3f} Lambda={Lambda:.2e} sigma={sigma:.2e} T={T:.2e}")
print("FIN 07: g decrece, K crece 0->1, sigma decrece? Lambda^2~a^{-1.5} g^2~log^-2 => sigma decrece, necesita Lambda creciente (confinamiento)")
