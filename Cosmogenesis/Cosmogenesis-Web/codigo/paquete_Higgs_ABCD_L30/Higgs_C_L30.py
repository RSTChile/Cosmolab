"""
HIGGS C CORREGIDO L=30 - friccion eta=|grad theta|^2 m=sum/(eta+eps)
NULL eta=0 misma corrida, L=30 igual que F0
"""
import numpy as np
from collections import deque
T0=1e15; H_topo=0.01; L=30; pasos=500; y0=1.0; eps=1e-3
rng=np.random.default_rng(2025)
phi=np.ones((L,L))+0.5*rng.normal(size=(L,L))
theta=rng.uniform(0,2*np.pi,size=(L,L))
ar=np.ones((L,L),bool); ad=np.ones((L,L),bool)
print("HIGGS C L=30 eta=|grad theta|^2")
for step in range(pasos):
    tg=step/pasos; a=np.exp(6*tg); T=T0/a
    K=np.exp(-1/(a+1e-10))
    dth=np.where(ar,np.sin(np.roll(theta,-1,axis=1)-theta),0)+np.where(np.roll(ar,1,axis=1),np.sin(np.roll(theta,1,axis=1)-theta),0)+np.where(ad,np.sin(np.roll(theta,-1,axis=0)-theta),0)+np.where(np.roll(ad,1,axis=0),np.sin(np.roll(theta,1,axis=0)-theta),0)
    theta=np.mod(theta+0.1*K*dth,2*np.pi)
    left=np.roll(ar,1,axis=1); up=np.roll(ad,1,axis=0)
    cnt=ar.astype(int)+left.astype(int)+ad.astype(int)+up.astype(int)
    s=np.where(ar,np.roll(phi,-1,axis=1),0)+np.where(left,np.roll(phi,1,axis=1),0)+np.where(ad,np.roll(phi,-1,axis=0),0)+np.where(up,np.roll(phi,1,axis=0),0)
    mean=np.divide(s,cnt,out=np.zeros_like(phi),where=cnt>0)
    phi_new=phi.copy(); phi_new[cnt>0]=phi[cnt>0]+0.3*(mean[cnt>0]-phi[cnt>0]); phi=phi_new
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
        grad_x = (np.roll(theta,-1,axis=1)-np.roll(theta,1,axis=1))/2
        grad_x = np.angle(np.exp(1j*grad_x))
        grad_y = (np.roll(theta,-1,axis=0)-np.roll(theta,1,axis=0))/2
        grad_y = np.angle(np.exp(1j*grad_y))
        eta = grad_x**2 + grad_y**2
        media=phi.mean(); visto=np.zeros((L,L),bool); clusters=[]
        for y in range(L):
            for x in range(L):
                if visto[y,x]: continue
                q=deque([(y,x)]); visto[y,x]=True; nodes=[(y,x)]; lado=phi[y,x]>=media; perim=0; sum_rho=1; sum_eta=eta[y,x]
                while q:
                    cy,cx=q.popleft()
                    if not ar[cy,cx] or (phi[cy,(cx+1)%L]>=media)!=lado: perim+=1
                    if not ar[cy,(cx-1)%L] or (phi[cy,(cx-1)%L]>=media)!=lado: perim+=1
                    if not ad[cy,cx] or (phi[(cy+1)%L,cx]>=media)!=lado: perim+=1
                    if not ad[(cy-1)%L,cx] or (phi[(cy-1)%L,cx]>=media)!=lado: perim+=1
                    if ar[cy,cx] and not visto[cy,(cx+1)%L] and (phi[cy,(cx+1)%L]>=media)==lado:
                        visto[cy,(cx+1)%L]=True; q.append((cy,(cx+1)%L)); nodes.append((cy,(cx+1)%L)); sum_rho+=1; sum_eta+=eta[cy,(cx+1)%L]
                    if ar[cy,(cx-1)%L] and not visto[cy,(cx-1)%L] and (phi[cy,(cx-1)%L]>=media)==lado:
                        visto[cy,(cx-1)%L]=True; q.append((cy,(cx-1)%L)); nodes.append((cy,(cx-1)%L)); sum_rho+=1; sum_eta+=eta[cy,(cx-1)%L]
                    if ad[cy,cx] and not visto[(cy+1)%L,cx] and (phi[(cy+1)%L,cx]>=media)==lado:
                        visto[(cy+1)%L,cx]=True; q.append(((cy+1)%L,cx)); nodes.append(((cy+1)%L,cx)); sum_rho+=1; sum_eta+=eta[(cy+1)%L,cx]
                    if ad[(cy-1)%L,cx] and not visto[(cy-1)%L,cx] and (phi[(cy-1)%L,cx]>=media)==lado:
                        visto[(cy-1)%L,cx]=True; q.append(((cy-1)%L,cx)); nodes.append(((cy-1)%L,cx)); sum_rho+=1; sum_eta+=eta[(cy-1)%L,cx]
                k=len(nodes); avg_eta=sum_eta/k
                m = y0*sum_rho/(avg_eta+eps)  # friccion: a mas eta menos arrastre? invertido: m ~ 1/eta ligero si eta grande
                m_null = y0*sum_rho/(eps)  # eta=0
                clusters.append((k,perim,m,m_null,avg_eta))
        k1_m=[c[2] for c in clusters if c[0]==1]; k3_m=[c[2] for c in clusters if c[0]==3 and c[1]==8]
        k1_n=[c[3] for c in clusters if c[0]==1]; k3_n=[c[3] for c in clusters if c[0]==3 and c[1]==8]
        ratio=np.mean(k1_m)/(np.mean(k3_m)+1e-30) if k3_m and k1_m else 0
        ratio_null=np.mean(k1_n)/(np.mean(k3_n)+1e-30) if k3_n and k1_n else 0
        print(f"step {step:3d} a={a:6.1f} K={K:.3f} <eta>={np.mean(eta):.3f} k1={len(k1_m)} k3={len(k3_m)} ratio={ratio:.3f} ratio_NULL={ratio_null:.3f}")
print("FIN C: k3 liso eta~0 => m grande, k1 rugoso eta grande => m chico => ratio <<1 si funciona")
