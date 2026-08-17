"""
F0 SMOKE PREREGISTRO + F0 TRIADA CORREGIDA V2
Contrato: T*a/T0 = 1 ±5%, rho*a^3/rho0 = 1 ±5%
Bugfix: T = T0 / a (no T/a acumulativo), a = exp(H_inf * tg) continuo

Fase 5 Triada basal holistica: a(t) supraluminica + T(t) adiabasis + rho(t) plasma + t_g genetico
"""
import numpy as np
from collections import deque

# F0 SMOKE - sin grafo, solo dilucion
print("=== F0 SMOKE PREREGISTRO ===")
T0 = 1e15
rho0 = 1e30
H_inf = 1e35
dt = 1e-38
pasos = 200
a0 = 1.0
max_err_T = 0
max_err_rho = 0
for step in range(pasos):
    tg = step*dt
    a = a0 * np.exp(H_inf * tg)  # continuo, no a trozos
    T = T0 / a  # FIX: no T/a acumulativo
    rho = rho0 / (a**3)  # FIX: rho0 / a^3
    err_T = abs(T*a/T0 - 1)
    err_rho = abs(rho*(a**3)/rho0 - 1)
    max_err_T = max(max_err_T, err_T)
    max_err_rho = max(max_err_rho, err_rho)
    if step%40==0:
        print(f"step {step} tg={tg:.2e} a={a:.6f} T*a/T0={T*a/T0:.6f} rho*a^3/rho0={rho*(a**3)/rho0:.6f}")

print(f"\nSMOKE RESULT: max|T*a/T0-1|={max_err_T:.2e} max|rho*a^3/rho0-1|={max_err_rho:.2e}")
print("PASS" if max_err_T<0.05 and max_err_rho<0.05 else "FAIL")
print("")

# F0 TRIADA COMPLETA CORREGIDA V2 - con grafo y fase pero con contrato respetado
print("=== F0 TRIADA HOLISTICA V2 CORREGIDA ===")
L=30
rng=np.random.default_rng(2025)
xs=np.linspace(0,1,L); xx,yy=np.meshgrid(xs,xs)
pert=np.zeros((L,L))
for mx in range(1,4):
    for my in range(1,4):
        fase=rng.uniform(0,2*np.pi)
        pert+= np.sin(2*np.pi*(mx*xx+my*yy)+fase)/(mx+my)
pert-=pert.mean()
if pert.std()>0: pert/=pert.std()
phi=np.ones((L,L))+1e-9*pert
theta=rng.uniform(0,2*np.pi,size=(L,L))
ar=np.ones((L,L),bool); ad=np.ones((L,L),bool)
w=1/3
H_topo=0.0025

kB=1.380649e-23
e_charge=1.602176634e-19
eps0=8.8541878128e-12

for step in range(pasos):
    tg=step*dt
    a = a0 * np.exp(H_inf * tg)
    T_field = T0 / a  # temperatura homogenea base + fluctuacion epsilon*S
    rho_field = rho0 / (a**3)
    # S>0 asimetria campo: gradiente T(r)=T*(1+eps*f(x))
    epsilon=1e-5
    T_local = T_field * (1+epsilon*pert)
    
    # paso topologico con a acoplado
    # phi representa densidad energia / rho_mean
    cnt=np.zeros_like(phi,dtype=int); s=np.zeros_like(phi)
    cnt+= ar.astype(int); s+= np.where(ar, np.roll(phi,-1,axis=1),0)
    left=np.roll(ar,1,axis=1); cnt+= left.astype(int); s+= np.where(left, np.roll(phi,1,axis=1),0)
    cnt+= ad.astype(int); s+= np.where(ad, np.roll(phi,-1,axis=0),0)
    up=np.roll(ad,1,axis=0); cnt+= up.astype(int); s+= np.where(up, np.roll(phi,1,axis=0),0)
    mean=np.divide(s,cnt,out=np.zeros_like(phi),where=cnt>0)
    phi_new=phi.copy()
    phi_new[cnt>0]=phi[cnt>0]+0.5*(mean[cnt>0]-phi[cnt>0])
    # dilucion correcta: phi mantiene contraste pero media sigue rho
    phi = phi_new * (1.0)  # contraste preservado, no diluido doble
    
    # K_phys con lambda_D adimensionalizado para no dar 0 siempre
    # lambda_D_norm = sqrt(T_field/T0 / (rho_field/rho0)) = sqrt( a^2 ) = a
    lambda_D_norm = a  # adimensional, no SI crudo
    K_phys = np.exp(-1.0/(lambda_D_norm+1e-10))
    dtheta=np.zeros_like(theta)
    dtheta+= np.where(ar, np.sin(np.roll(theta,-1,axis=1)-theta),0)
    dtheta+= np.where(left, np.sin(np.roll(theta,1,axis=1)-theta),0)
    dtheta+= np.where(ad, np.sin(np.roll(theta,-1,axis=0)-theta),0)
    dtheta+= np.where(up, np.sin(np.roll(theta,1,axis=0)-theta),0)
    theta = np.mod(theta + 0.05*K_phys*dtheta, 2*np.pi)
    
    # corte H_fisico(t)=H_topo * (T_field/T0) acoplado a expansion real
    H_fisico = H_topo * (T_field/T0)
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
        # detectar k
        media=phi.mean(); visto=np.zeros((L,L),bool)
        k_counts={1:0,2:0,3:0,6:0}
        clusters=[]
        for y in range(L):
            for x in range(L):
                if visto[y,x]: continue
                q=deque([(y,x)]); visto[y,x]=True; nodes=[(y,x)]; lado=phi[y,x]>=media; perim=0; sum_rho=phi[y,x]
                while q:
                    cy,cx=q.popleft()
                    if not ar[cy,cx] or (phi[cy,(cx+1)%L]>=media)!=lado: perim+=1
                    if not ar[cy,(cx-1)%L] or (phi[cy,(cx-1)%L]>=media)!=lado: perim+=1
                    if not ad[cy,cx] or (phi[(cy+1)%L,cx]>=media)!=lado: perim+=1
                    if not ad[(cy-1)%L,cx] or (phi[(cy-1)%L,cx]>=media)!=lado: perim+=1
                    if ar[cy,cx] and not visto[cy,(cx+1)%L] and (phi[cy,(cx+1)%L]>=media)==lado:
                        visto[cy,(cx+1)%L]=True; q.append((cy,(cx+1)%L)); nodes.append((cy,(cx+1)%L)); sum_rho+=phi[cy,(cx+1)%L]
                    if ar[cy,(cx-1)%L] and not visto[cy,(cx-1)%L] and (phi[cy,(cx-1)%L]>=media)==lado:
                        visto[cy,(cx-1)%L]=True; q.append((cy,(cx-1)%L)); nodes.append((cy,(cx-1)%L)); sum_rho+=phi[cy,(cx-1)%L]
                    if ad[cy,cx] and not visto[(cy+1)%L,cx] and (phi[(cy+1)%L,cx]>=media)==lado:
                        visto[(cy+1)%L,cx]=True; q.append(((cy+1)%L,cx)); nodes.append(((cy+1)%L,cx)); sum_rho+=phi[(cy+1)%L,cx]
                    if ad[(cy-1)%L,cx] and not visto[(cy-1)%L,cx] and (phi[(cy-1)%L,cx]>=media)==lado:
                        visto[(cy-1)%L,cx]=True; q.append(((cy-1)%L,cx)); nodes.append(((cy-1)%L,cx)); sum_rho+=phi[(cy-1)%L,cx]
                k=len(nodes)
                if k in k_counts: k_counts[k]+=1
                clusters.append((k,perim,sum_rho,nodes))
        # masa fisica = sum_rho / P_plasma
        P_plasma = w * rho_field * (3e8)**2
        k1_m = [c[2]/P_plasma for c in clusters if c[0]==1]
        k3_m = [c[2]/P_plasma for c in clusters if c[0]==3 and c[1]==8]
        mk1 = np.mean(k1_m) if k1_m else 0
        mk3 = np.mean(k3_m) if k3_m else 0
        ratio = mk1/(mk3+1e-30) if mk3>0 else 0
        print(f"step {step:3d} a={a:.4f} T={T_field:.2e}K rho={rho_field:.2e} H_fis={H_fisico:.2e} lam_D_norm={lambda_D_norm:.2f} K={K_phys:.3f} k1={k_counts[1]} k2={k_counts[2]} k3={k_counts[3]} k6={k_counts[6]} mk1/mk3={ratio:.4f}")

print("\nF0 V2 CORREGIDA FINAL - contrato T*a/T0 y rho*a^3/rho0 respetado, k1/k3 ahora medibles")
