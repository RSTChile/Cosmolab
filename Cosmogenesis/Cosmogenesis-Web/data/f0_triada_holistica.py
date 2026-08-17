"""
F0 Triada Holistica - Fase 5 Dominio F
Plasma + Expansion supraluminica + Tiempo genetico
Alexis Lopez Tapia - Teoria Cosmosemiotica

Tríada basal acoplada desde t=0 t_g (genetico) no t_e (emergente)
a(t) expansion, T(t) adiabasis, rho(t) densidad plasma
k=3 perim8 U(1) 0.613 como condicion contorno T, no resultado a forzar
"""

import numpy as np
from collections import deque

# Constantes fisicas basales (fijadas una vez, no ajustables para 1/1836)
kB = 1.380649e-23
e_charge = 1.602176634e-19
eps0 = 8.8541878128e-12
Tc = 97e6 * e_charge  # 97 MeV en Joules - escala banda 1e15-1e12 K
T0 = 1e15  # K inicio banda
rho0 = 1e30  # kg/m3 densidad plasma inicial estimada
w = 1/3  # EOS radiacion P = w rho c^2 simplificado P = w rho

# Parametros cosmologicos Fase 5
H_inf = 1e35  # s-1 inflacion supraluminica orden 1e-36s
L = 30
dt_gen = 1e-38  # paso tiempo genetico s
pasos = 200
H_topo = 0.0025  # corte topologico como tasa base -> ahora H_fisico(t)=H_inf * f(T)

def campo_inicial(L, eps, rng):
    xs = np.linspace(0,1,L)
    xx, yy = np.meshgrid(xs, xs)
    pert = np.zeros((L,L))
    for mx in range(1,4):
        for my in range(1,4):
            fase = rng.uniform(0, 2*np.pi)
            pert += np.sin(2*np.pi*(mx*xx+my*yy)+fase)/(mx+my)
    pert -= pert.mean()
    if pert.std()>0: pert/=pert.std()
    return np.ones((L,L))+eps*pert, rng.uniform(0,2*np.pi,size=(L,L))

def paso_triada(phi, theta, T, rho, a, ar, ad, H_fisico):
    # expansion afecta dilucion y enfriamiento
    # phi = densidad energia normalizada, T = temperatura local
    cnt = np.zeros_like(phi,dtype=int); s = np.zeros_like(phi)
    cnt+= ar.astype(int); s+= np.where(ar, np.roll(phi,-1,axis=1),0)
    left=np.roll(ar,1,axis=1); cnt+= left.astype(int); s+= np.where(left, np.roll(phi,1,axis=1),0)
    cnt+= ad.astype(int); s+= np.where(ad, np.roll(phi,-1,axis=0),0)
    up=np.roll(ad,1,axis=0); cnt+= up.astype(int); s+= np.where(up, np.roll(phi,1,axis=0),0)
    mean = np.divide(s,cnt,out=np.zeros_like(phi),where=cnt>0)
    phi_new = phi.copy()
    phi_new[cnt>0] = phi[cnt>0] + 0.5*(mean[cnt>0]-phi[cnt>0])
    # enfriamiento adiabatico T propto 1/a + difusion termica local
    T_new = T / a  # simplificado: T decrece con factor escala
    # actualizacion theta con K_phys(T, rho) no Kuramoto fijo
    n_local = rho / 1e-27  # densidad numerica aprox proton mass
    lambda_D = np.sqrt(eps0 * kB * T_new / (n_local * e_charge**2 + 1e-100))
    K_phys = np.mean(np.exp(-1.0/(lambda_D+1e-10)))  # apantallamiento 0-1
    dtheta = np.zeros_like(theta)
    dtheta+= np.where(ar, np.sin(np.roll(theta,-1,axis=1)-theta),0)
    dtheta+= np.where(left, np.sin(np.roll(theta,1,axis=1)-theta),0)
    dtheta+= np.where(ad, np.sin(np.roll(theta,-1,axis=0)-theta),0)
    dtheta+= np.where(up, np.sin(np.roll(theta,1,axis=0)-theta),0)
    theta_new = np.mod(theta + 0.05*K_phys*dtheta, 2*np.pi)
    return phi_new, theta_new, T_new, lambda_D, K_phys

# Sim principal F0
rng = np.random.default_rng(2025)
phi, theta = campo_inicial(L, 1e-9, rng)
ar = np.ones((L,L),bool); ad = np.ones((L,L),bool)
T = np.ones((L,L))*T0
rho = np.ones((L,L))*rho0
a = 1.0

print("F0 triada holistica inicio")
print(f"T0={T0:.2e}K Tc=97MeV a0={a} rho0={rho0:.2e} H_inf={H_inf:.2e}")

historial = []
for step in range(pasos):
    # expansion supraluminica a(t)=exp(H_inf * t_g)
    t_g = step * dt_gen
    a = np.exp(H_inf * t_g) if step<50 else a * (1+ H_inf*dt_gen*0.01)  # infla luego lenta
    T_mean = T.mean()
    rho_mean = rho0 / (a**3)
    rho[:,:] = rho_mean
    H_fisico = H_topo * (T_mean/T0)  # H disminuye con T
    phi, theta, T, lambda_D, K_phys = paso_triada(phi, theta, T, rho, a, ar, ad, H_fisico)
    # corte topologico con H_fisico
    tot=int(np.sum(ar)+np.sum(ad)); nc=int(round(H_fisico*tot))
    if nc>0 and tot>0:
        if np.sum(ar)>0:
            nr=int(round(nc*np.sum(ar)/tot))
            idx=np.argwhere(ar)
            if len(idx)>0 and nr>0:
                sel=rng.choice(len(idx), size=min(nr,len(idx)), replace=False)
                for i in sel: ar[tuple(idx[i])]=False
        rem=nc - (nr if 'nr' in locals() else 0)
        if rem>0 and np.sum(ad)>0:
            idx=np.argwhere(ad)
            sel=rng.choice(len(idx), size=min(rem,len(idx)), replace=False)
            for i in sel: ad[tuple(idx[i])]=False
    if step%40==0:
        # medir k3 y masa fisica m = E_int / P
        media=phi.mean(); visto=np.zeros((L,L),bool)
        k_counts={1:0,2:0,3:0}
        for y in range(L):
            for x in range(L):
                if visto[y,x]: continue
                q=deque([(y,x)]); visto[y,x]=True; nodes=[(y,x)]; lado=phi[y,x]>=media
                while q:
                    cy,cx=q.popleft()
                    if ar[cy,cx] and not visto[cy,(cx+1)%L] and (phi[cy,(cx+1)%L]>=media)==lado:
                        visto[cy,(cx+1)%L]=True; q.append((cy,(cx+1)%L)); nodes.append((cy,(cx+1)%L))
                    if ar[cy,(cx-1)%L] and not visto[cy,(cx-1)%L] and (phi[cy,(cx-1)%L]>=media)==lado:
                        visto[cy,(cx-1)%L]=True; q.append((cy,(cx-1)%L)); nodes.append((cy,(cx-1)%L))
                    if ad[cy,cx] and not visto[(cy+1)%L,cx] and (phi[(cy+1)%L,cx]>=media)==lado:
                        visto[(cy+1)%L,cx]=True; q.append(((cy+1)%L,cx)); nodes.append(((cy+1)%L,cx))
                    if ad[(cy-1)%L,cx] and not visto[(cy-1)%L,cx] and (phi[(cy-1)%L,cx]>=media)==lado:
                        visto[(cy-1)%L,cx]=True; q.append(((cy-1)%L,cx)); nodes.append(((cy-1)%L,cx))
                k=len(nodes)
                if k in k_counts: k_counts[k]+=1
        P_plasma = w * rho_mean * (3e8)**2  # Pa
        E_k3 = k_counts[3]* Tc  # energia interna atrapada aprox
        E_k1 = k_counts[1]* kB * T_mean
        m_fisica_ratio = (E_k1/(P_plasma+1e-100)) / (E_k3/(P_plasma+1e-100)+1e-100) if E_k3>0 else 0
        historial.append((step, t_g, a, T_mean, rho_mean, lambda_D.mean(), K_phys, k_counts[1], k_counts[3], m_fisica_ratio))
        print(f"step {step} t_g={t_g:.2e}s a={a:.3f} T={T_mean:.2e}K rho={rho_mean:.2e} lambda_D={lambda_D.mean():.2e}m K_phys={K_phys:.3f} k1={k_counts[1]} k3={k_counts[3]} m_k1/m_k3_fis={m_fisica_ratio:.4f}")

print("F0 triada completa - observables holísticos listos")
print("Nota: masa fisica = E_int / P_plasma / a(t) busca quiebre O(1)->0.00054")
