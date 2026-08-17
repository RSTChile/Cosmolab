"""
MF-5 entropia S = a^3 * (rho+P)/T * V_cut
V_cut = 1/H_fis^3, H_fis = H_topo * sqrt(T/T0)
rho = rho0/a^3, P = w rho, w=1/3
S debe crecer con a (2do principio) S ~ a^3 * rho/T * 1/H^3
"""
import numpy as np
T0=1e15; rho0=1e30; H_topo=0.01; L=30; pasos=500; w=1/3
print("MF-5 entropia S = a^3 (rho+P)/T * 1/H_fis^3")
for step in range(pasos):
    tg=step/pasos; a=np.exp(6*tg); T=T0/a; rho=rho0/(a**3); P=w*rho
    H_fis=H_topo*np.sqrt(T/T0)
    V_cut = 1/(H_fis**3+1e-30)
    S = (a**3)*(rho+P)/ (T+1e-30) * V_cut
    # SMOKE: S*a? debe crecer
    if step%50==0:
        print(f"step {step:3d} a={a:6.1f} T={T:.2e} rho={rho:.2e} H_fis={H_fis:.4e} V_cut={V_cut:.2e} S={S:.2e}")
print("FIN MF-5: S ~ a^3 * a^-3 / a^-1 * 1/(a^-1.5) = a * a^{1.5}=a^{2.5} crece => 2do principio OK")
