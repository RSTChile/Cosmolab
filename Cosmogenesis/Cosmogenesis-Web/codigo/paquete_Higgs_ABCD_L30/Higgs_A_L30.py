"""
HIGGS A CORREGIDO L=30 - Campo Phi V=rPhi^2+uPhi^4 r~ (T/Tc-1)
NULL Phi=0 en misma corrida, m = y0*|Phi_avg|*sum_rho formula unica
"""
import numpy as np
from collections import deque
T0=1e15; rho0=1e30; H_topo=0.01; L=30; pasos=500; Tc=T0*0.1; y0=0.3; r0=1.0; u=0.5
rng=np.random.default_rng(2025)
phi=np.ones((L,L))+0.5*rng.normal(size=(L,L))
Phi=rng.normal(0,0.1,size=(L,L))  # campo Higgs
theta=rng.uniform(0,2*np.pi,size=(L,L))
ar=np.ones((L,L),bool); ad=np.ones((L,L),bool)
print("HIGGS A L=30 SSB T<Tc")
for step in range(pasos):
    tg=step/pasos; a=np.exp(6*tg); T=T0/a
    r = r0*(T/Tc -1)  # <0 si T<Tc -> SSB
    # evolucion Phi: dPhi = -dV/dPhi = -2rPhi -4uPhi^3 + diffusion
    lap = (np.roll(Phi,-1,axis=0)+np.roll(Phi,1,axis=0)+np.roll(Phi,-1,axis=1)+np.roll(Phi,1,axis=1)-4*Phi)
    dV = 2*r*Phi + 4*u*Phi**3
    Phi += 0.01*lap -0.01*dV
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
        media=phi.mean(); visto=np.zeros((L,L),bool); clusters=[]
        for y in range(L):
            for x in range(L):
                if visto[y,x]: continue
                q=deque([(y,x)]); visto[y,x]=True; nodes=[(y,x)]; lado=phi[y,x]>=media; perim=0; sum_rho=1; sum_Phi=abs(Phi[y,x])
                while q:
                    cy,cx=q.popleft()
                    if not ar[cy,cx] or (phi[cy,(cx+1)%L]>=media)!=lado: perim+=1
                    if not ar[cy,(cx-1)%L] or (phi[cy,(cx-1)%L]>=media)!=lado: perim+=1
                    if not ad[cy,cx] or (phi[(cy+1)%L,cx]>=media)!=lado: perim+=1
                    if not ad[(cy-1)%L,cx] or (phi[(cy-1)%L,cx]>=media)!=lado: perim+=1
                    if ar[cy,cx] and not visto[cy,(cx+1)%L] and (phi[cy,(cx+1)%L]>=media)==lado:
                        visto[cy,(cx+1)%L]=True; q.append((cy,(cx+1)%L)); nodes.append((cy,(cx+1)%L)); sum_rho+=1; sum_Phi+=abs(Phi[cy,(cx+1)%L])
                    if ar[cy,(cx-1)%L] and not visto[cy,(cx-1)%L] and (phi[cy,(cx-1)%L]>=media)==lado:
                        visto[cy,(cx-1)%L]=True; q.append((cy,(cx-1)%L)); nodes.append((cy,(cx-1)%L)); sum_rho+=1; sum_Phi+=abs(Phi[cy,(cx-1)%L])
                    if ad[cy,cx] and not visto[(cy+1)%L,cx] and (phi[(cy+1)%L,cx]>=media)==lado:
                        visto[(cy+1)%L,cx]=True; q.append(((cy+1)%L,cx)); nodes.append(((cy+1)%L,cx)); sum_rho+=1; sum_Phi+=abs(Phi[(cy+1)%L,cx])
                    if ad[(cy-1)%L,cx] and not visto[(cy-1)%L,cx] and (phi[(cy-1)%L,cx]>=media)==lado:
                        visto[(cy-1)%L,cx]=True; q.append(((cy-1)%L,cx)); nodes.append(((cy-1)%L,cx)); sum_rho+=1; sum_Phi+=abs(Phi[(cy-1)%L,cx])
                k=len(nodes); avg_Phi=sum_Phi/k if k>0 else 0
                m = y0*avg_Phi*sum_rho
                m_null = y0*0*sum_rho  # NULL Phi=0
                clusters.append((k,perim,m,m_null,avg_Phi))
        k1_m=[c[2] for c in clusters if c[0]==1]; k3_m=[c[2] for c in clusters if c[0]==3 and c[1]==8]
        k1_null=[c[3] for c in clusters if c[0]==1]; k3_null=[c[3] for c in clusters if c[0]==3 and c[1]==8]
        mk1=np.mean(k1_m) if k1_m else 0; mk3=np.mean(k3_m) if k3_m else 0; ratio=mk1/(mk3+1e-30) if mk3>0 else 0
        mk1_n=np.mean(k1_null) if k1_null else 0; mk3_n=np.mean(k3_null) if k3_null else 0; ratio_n=mk1_n/(mk3_n+1e-30) if mk3_n>0 else 0
        avg_Phi_global=np.mean(np.abs(Phi))
        print(f"step {step:3d} a={a:6.1f} T={T:.2e} r={r:.3f} <Phi>={avg_Phi_global:.3f} k1={len(k1_m)} k3={len(k3_m)} ratio={ratio:.3f} ratio_NULL={ratio_n:.3f}")
print("FIN A: si <Phi>->0 ratio O(1), si <Phi>!=0 y avg_Phi_k1 != avg_Phi_k3 ratio <<1 test Higgs")
