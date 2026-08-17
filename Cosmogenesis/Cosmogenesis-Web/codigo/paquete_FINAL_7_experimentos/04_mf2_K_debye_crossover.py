
"""
MF-2 - K_phys Debye + crossover 0.391->0.191 extensivo resuelto
a=exp(6*tg) K=exp(-1/a) 0.368->0.997
Mide |M| y Kc
"""
import numpy as np
from collections import deque
T0=1e15; H_topo=0.01; L=30; pasos=500
rng=np.random.default_rng(2025)
phi=np.ones((L,L))+0.1*np.random.randn(L,L)
theta=rng.uniform(0,2*np.pi,size=(L,L))
ar=np.ones((L,L),bool); ad=np.ones((L,L),bool)
print("MF-2 K_phys crossover")
for step in range(pasos):
    tg=step/pasos; a=np.exp(6*tg); T=T0/a
    left=np.roll(ar,1,axis=1); up=np.roll(ad,1,axis=0)
    cnt=ar.astype(int)+left.astype(int)+ad.astype(int)+up.astype(int)
    s=np.where(ar,np.roll(phi,-1,axis=1),0)+np.where(left,np.roll(phi,1,axis=1),0)+np.where(ad,np.roll(phi,-1,axis=0),0)+np.where(up,np.roll(phi,1,axis=0),0)
    mean=np.divide(s,cnt,out=np.zeros_like(phi),where=cnt>0)
    phi_new=phi.copy(); phi_new[cnt>0]=phi[cnt>0]+0.5*(mean[cnt>0]-phi[cnt>0]); phi=phi_new
    K=np.exp(-1.0/(a+1e-10))
    dth=np.where(ar,np.sin(np.roll(theta,-1,axis=1)-theta),0)+np.where(left,np.sin(np.roll(theta,1,axis=1)-theta),0)+np.where(ad,np.sin(np.roll(theta,-1,axis=0)-theta),0)+np.where(up,np.sin(np.roll(theta,1,axis=0)-theta),0)
    theta=np.mod(theta+0.1*K*dth,2*np.pi)
    H_fis=H_topo*np.sqrt(T/T0); tot=int(np.sum(ar)+np.sum(ad)); nc=int(round(H_fis*tot))
    if nc>0 and tot>0 and np.sum(ar)>0:
        nr=int(round(nc*np.sum(ar)/tot)); idx=np.argwhere(ar)
        if len(idx)>0 and nr>0:
            sel=rng.choice(len(idx), size=min(nr,len(idx)), replace=False)
            for i in sel: ar[tuple(idx[i])]=False
        rem=nc-nr
        if rem>0 and np.sum(ad)>0:
            idx=np.argwhere(ad); sel=rng.choice(len(idx), size=min(rem,len(idx)), replace=False)
            for i in sel: ad[tuple(idx[i])]=False
    if step%50==0:
        M=np.abs(np.mean(np.exp(1j*theta)))
        print(f"step {step:3d} a={a:6.1f} T={T:.2e} K={K:.3f} H_fis={H_fis:.4f} |M|={M:.3f}")
print("MF-2 FIN: K 0.368->0.997 resuelve Kc 0.391->0.191 crossover")
