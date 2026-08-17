"""
HIGGS B CORREGIDO L=30 - orden phi con lambda(rho) m=y0*|phi_avg|*sum
NULL lambda=0
"""
import numpy as np
from collections import deque
T0=1e15; rho0=1e30; H_topo=0.01; L=30; pasos=500; y0=0.3; lam0=0.1; D=0.1
rng=np.random.default_rng(2025)
phi=np.ones((L,L))+0.5*rng.normal(size=(L,L))
ar=np.ones((L,L),bool); ad=np.ones((L,L),bool)
print("HIGGS B L=30 lambda(rho)")
for step in range(pasos):
    tg=step/pasos; a=np.exp(6*tg); T=T0/a; rho=rho0/(a**3)
    lam = lam0 * (rho/rho0) * 1e6  # escala para que sea O(1) temprano
    lap = (np.roll(phi,-1,axis=0)+np.roll(phi,1,axis=0)+np.roll(phi,-1,axis=1)+np.roll(phi,1,axis=1)-4*phi)
    phi += D*lap - lam*(phi**2 -1)*phi *0.01
    left=np.roll(ar,1,axis=1); up=np.roll(ad,1,axis=0)
    cnt=ar.astype(int)+left.astype(int)+ad.astype(int)+up.astype(int)
    s=np.where(ar,np.roll(phi,-1,axis=1),0)+np.where(left,np.roll(phi,1,axis=1),0)+np.where(ad,np.roll(phi,-1,axis=0),0)+np.where(up,np.roll(phi,1,axis=0),0)
    mean=np.divide(s,cnt,out=np.zeros_like(phi),where=cnt>0)
    phi_new=phi.copy(); phi_new[cnt>0]=phi[cnt>0]+0.1*(mean[cnt>0]-phi[cnt>0]); phi=phi_new
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
        media=phi.mean(); visto=np.zeros((L,L),bool); clusters=[]
        for y in range(L):
            for x in range(L):
                if visto[y,x]: continue
                q=deque([(y,x)]); visto[y,x]=True; nodes=[(y,x)]; lado=phi[y,x]>=media; perim=0; sum_rho=1; sum_phi=abs(phi[y,x])
                while q:
                    cy,cx=q.popleft()
                    if not ar[cy,cx] or (phi[cy,(cx+1)%L]>=media)!=lado: perim+=1
                    if not ar[cy,(cx-1)%L] or (phi[cy,(cx-1)%L]>=media)!=lado: perim+=1
                    if not ad[cy,cx] or (phi[(cy+1)%L,cx]>=media)!=lado: perim+=1
                    if not ad[(cy-1)%L,cx] or (phi[(cy-1)%L,cx]>=media)!=lado: perim+=1
                    if ar[cy,cx] and not visto[cy,(cx+1)%L] and (phi[cy,(cx+1)%L]>=media)==lado:
                        visto[cy,(cx+1)%L]=True; q.append((cy,(cx+1)%L)); nodes.append((cy,(cx+1)%L)); sum_rho+=1; sum_phi+=abs(phi[cy,(cx+1)%L])
                    if ar[cy,(cx-1)%L] and not visto[cy,(cx-1)%L] and (phi[cy,(cx-1)%L]>=media)==lado:
                        visto[cy,(cx-1)%L]=True; q.append((cy,(cx-1)%L)); nodes.append((cy,(cx-1)%L)); sum_rho+=1; sum_phi+=abs(phi[cy,(cx-1)%L])
                    if ad[cy,cx] and not visto[(cy+1)%L,cx] and (phi[(cy+1)%L,cx]>=media)==lado:
                        visto[(cy+1)%L,cx]=True; q.append(((cy+1)%L,cx)); nodes.append(((cy+1)%L,cx)); sum_rho+=1; sum_phi+=abs(phi[(cy+1)%L,cx])
                    if ad[(cy-1)%L,cx] and not visto[(cy-1)%L,cx] and (phi[(cy-1)%L,cx]>=media)==lado:
                        visto[(cy-1)%L,cx]=True; q.append(((cy-1)%L,cx)); nodes.append(((cy-1)%L,cx)); sum_rho+=1; sum_phi+=abs(phi[(cy-1)%L,cx])
                k=len(nodes); avg=sum_phi/k
                m=y0*avg*sum_rho
                clusters.append((k,perim,m,avg))
        k1_m=[c[2] for c in clusters if c[0]==1]; k3_m=[c[2] for c in clusters if c[0]==3 and c[1]==8]
        ratio=np.mean(k1_m)/(np.mean(k3_m)+1e-30) if k3_m and k1_m else 0
        print(f"step {step:3d} a={a:6.1f} lam={lam:.3e} <phi>={np.mean(np.abs(phi)):.3f} k1={len(k1_m)} k3={len(k3_m)} ratio={ratio:.3f}")
print("FIN B")
