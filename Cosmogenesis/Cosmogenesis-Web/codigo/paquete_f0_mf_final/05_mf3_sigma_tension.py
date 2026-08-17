
"""
MF-3 - Tension cuerda sigma = P_plasma * a
P = w*rho*c^2, sigma = P*a  -> confinamiento V(r)~sigma*r
Mide clusters k3 energia atrapada
"""
import numpy as np
from collections import deque
T0=1e15; rho0=1e30; H_topo=0.01; L=30; pasos=500; w=1/3; c=3e8
rng=np.random.default_rng(2025)
phi=np.ones((L,L))+0.5*np.random.randn(L,L)
ar=np.ones((L,L),bool); ad=np.ones((L,L),bool)
print("MF-3 sigma = P*a tension cuerda")
for step in range(pasos):
    tg=step/pasos; a=np.exp(6*tg); T=T0/a; rho=rho0/(a**3); P=w*rho*c**2; sigma=P*a
    left=np.roll(ar,1,axis=1); up=np.roll(ad,1,axis=0)
    cnt=ar.astype(int)+left.astype(int)+ad.astype(int)+up.astype(int)
    s=np.where(ar,np.roll(phi,-1,axis=1),0)+np.where(left,np.roll(phi,1,axis=1),0)+np.where(ad,np.roll(phi,-1,axis=0),0)+np.where(up,np.roll(phi,1,axis=0),0)
    mean=np.divide(s,cnt,out=np.zeros_like(phi),where=cnt>0)
    phi_new=phi.copy(); phi_new[cnt>0]=phi[cnt>0]+0.5*(mean[cnt>0]-phi[cnt>0]); phi=phi_new
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
        media=phi.mean(); visto=np.zeros((L,L),bool); k3_E=[]
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
                if len(nodes)==3 and perim==8:
                    k3_E.append(sum_rho*P)
        E_k3=np.mean(k3_E) if k3_E else 0
        print(f"step {step:3d} a={a:6.1f} T={T:.2e} rho={rho:.2e} P={P:.2e} sigma=P*a={sigma:.2e} E_k3={E_k3:.2e}")
print("MF-3 FIN: sigma = P*a crece con a? No, P~1/a^3 => sigma~1/a^2 decrece -> desconfinamiento, invierte a sigma~a si w crece")
