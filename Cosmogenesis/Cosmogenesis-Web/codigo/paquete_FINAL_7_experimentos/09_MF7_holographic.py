"""
09_MF7_holographic - S = A/4G, rho = rho0/a^3 * (1 + L^2/a^2) correccion holografica
m_fis = S * T_norm , S = area perim / 4
Formula unica m = perim * log(a) * T_norm
"""
import numpy as np
from collections import deque
T0=1e15; rho0=1e30; H_topo=0.01; L=30; pasos=500
rng=np.random.default_rng(2025)
phi=np.ones((L,L))+0.8*np.sin(np.linspace(0,4*np.pi,L))[:,None]*np.cos(np.linspace(0,4*np.pi,L))[None,:]
ar=np.ones((L,L),bool); ad=np.ones((L,L),bool)
print("09 MF7 holografico S=A/4G perim")
for step in range(pasos):
    tg=step/pasos; a=np.exp(6*tg); T=T0/a
    left=np.roll(ar,1,axis=1); up=np.roll(ad,1,axis=0)
    cnt=ar.astype(int)+left.astype(int)+ad.astype(int)+up.astype(int)
    s=np.where(ar,np.roll(phi,-1,axis=1),0)+np.where(left,np.roll(phi,1,axis=1),0)+np.where(ad,np.roll(phi,-1,axis=0),0)+np.where(up,np.roll(phi,1,axis=0),0)
    mean=np.divide(s,cnt,out=np.zeros_like(phi),where=cnt>0)
    phi_new=phi.copy(); phi_new[cnt>0]=phi[cnt>0]+0.3*(mean[cnt>0]-phi[cnt>0]); phi=phi_new
    H_fis=H_topo*np.sqrt(T/T0); tot=int(np.sum(ar)+np.sum(ad)); nc=int(round(H_fis*tot))
    if nc>0 and tot>0:
        idx=np.argwhere(ar)
        if len(idx)>0:
            nr=int(round(nc*np.sum(ar)/tot)); sel=rng.choice(len(idx), size=min(nr,len(idx)), replace=False)
            for i in sel: ar[tuple(idx[i])]=False
    if step%100==0:
        media=phi.mean(); visto=np.zeros((L,L),bool); clusters=[]
        for y in range(L):
            for x in range(L):
                if visto[y,x]: continue
                q=deque([(y,x)]); visto[y,x]=True; nodes=[(y,x)]; lado=phi[y,x]>=media; perim=0
                while q:
                    cy,cx=q.popleft()
                    if not ar[cy,cx] or (phi[cy,(cx+1)%L]>=media)!=lado: perim+=1
                    if not ar[cy,(cx-1)%L] or (phi[cy,(cx-1)%L]>=media)!=lado: perim+=1
                    if not ad[cy,cx] or (phi[(cy+1)%L,cx]>=media)!=lado: perim+=1
                    if not ad[(cy-1)%L,cx] or (phi[(cy-1)%L,cx]>=media)!=lado: perim+=1
                    if ar[cy,cx] and not visto[cy,(cx+1)%L] and (phi[cy,(cx+1)%L]>=media)==lado:
                        visto[cy,(cx+1)%L]=True; q.append((cy,(cx+1)%L)); nodes.append((cy,(cx+1)%L))
                    if ar[cy,(cx-1)%L] and not visto[cy,(cx-1)%L] and (phi[cy,(cx-1)%L]>=media)==lado:
                        visto[cy,(cx-1)%L]=True; q.append((cy,(cx-1)%L)); nodes.append((cy,(cx-1)%L))
                    if ad[cy,cx] and not visto[(cy+1)%L,cx] and (phi[(cy+1)%L,cx]>=media)==lado:
                        visto[(cy+1)%L,cx]=True; q.append(((cy+1)%L,cx)); nodes.append(((cy+1)%L,cx))
                    if ad[(cy-1)%L,cx] and not visto[(cy-1)%L,cx] and (phi[(cy-1)%L,cx]>=media)==lado:
                        visto[(cy-1)%L,cx]=True; q.append(((cy-1)%L,cx)); nodes.append(((cy-1)%L,cx))
                clusters.append((len(nodes),perim))
        # m = perim * log(a) / T_norm? unica
        T_norm=T/T0+1e-6
        def m_holo(c): k,perim=c; return perim * np.log(a+1) * T_norm  # perim k3=8, k1=4 => ratio 4/8=0.5
        k1_m=[m_holo(c) for c in clusters if c[0]==1]; k3_m=[m_holo(c) for c in clusters if c[0]==3 and c[1]==8]
        ratio=np.mean(k1_m)/(np.mean(k3_m)+1e-30) if k3_m else 0
        print(f"step {step:3d} a={a:5.1f} T={T:.2e} k1={sum(1 for c in clusters if c[0]==1)} k3={sum(1 for c in clusters if c[0]==3 and c[1]==8)} perim_ratio={ratio:.3f}")
print("FIN 09: perim k1=4 k3=8 => ratio 0.5 O(1) no 0.00054, holografia sola no rompe")
