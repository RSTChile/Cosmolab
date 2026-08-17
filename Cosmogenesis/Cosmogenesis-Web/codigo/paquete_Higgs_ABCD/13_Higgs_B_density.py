"""
13_Higgs_B_density_contrast - phi dinamico con lambda(rho)*(phi^2-1)phi
lambda = lambda0 * rho/rho0
m = y0 * <phi>_k * sum_rho formula unica
"""
import numpy as np
from collections import deque
T0=1e15; rho0=1e30; H_topo=0.01; L=24; pasos=400; lambda0=1.0; y0=0.3; D=0.1
rng=np.random.default_rng(2025)
phi=np.ones((L,L))+0.5*rng.normal(size=(L,L))
ar=np.ones((L,L),bool); ad=np.ones((L,L),bool)
print("13 Higgs B densidad phi dinamico lambda(rho)")
for step in range(pasos):
    tg=step/pasos; a=np.exp(6*tg); T=T0/a; rho=rho0/(a**3)
    lam = lambda0 * (rho/rho0 + 0.01)  # crece con rho
    lap = (np.roll(phi,-1,axis=0)+np.roll(phi,1,axis=0)+np.roll(phi,-1,axis=1)+np.roll(phi,1,axis=1)-4*phi)
    phi = phi + D*lap - lam*(phi**2-1)*phi*0.1 + 0.01*rng.normal(size=(L,L))
    left=np.roll(ar,1,axis=1); up=np.roll(ad,1,axis=0)
    cnt=ar.astype(int)+left.astype(int)+ad.astype(int)+up.astype(int)
    s=np.where(ar,np.roll(phi,-1,axis=1),0)+np.where(left,np.roll(phi,1,axis=1),0)+np.where(ad,np.roll(phi,-1,axis=0),0)+np.where(up,np.roll(phi,1,axis=0),0)
    mean=np.divide(s,cnt,out=np.zeros_like(phi),where=cnt>0)
    phi_new=phi.copy(); phi_new[cnt>0]=phi[cnt>0]+0.1*(mean[cnt>0]-phi[cnt>0]); phi=phi_new
    H_fis=H_topo*np.sqrt(T/T0); tot=int(np.sum(ar)+np.sum(ad)); nc=int(round(H_fis*tot))
    if nc>0 and tot>0:
        idx=np.argwhere(ar)
        if len(idx)>0:
            nr=int(round(nc*np.sum(ar)/tot)); sel=rng.choice(len(idx), size=min(nr,len(idx)), replace=False)
            for i in sel: ar[tuple(idx[i])]=False
    if step%100==0:
        media=0.0 # phi tiende a +-1
        visto=np.zeros((L,L),bool); clusters=[]
        for y in range(L):
            for x in range(L):
                if visto[y,x]: continue
                q=deque([(y,x)]); visto[y,x]=True; nodes=[(y,x)]; lado=phi[y,x]>=0; perim=0; sum_phi=phi[y,x]; sum_rho=abs(phi[y,x])
                while q:
                    cy,cx=q.popleft()
                    if not ar[cy,cx] or (phi[cy,(cx+1)%L]>=0)!=lado: perim+=1
                    if not ar[cy,(cx-1)%L] or (phi[cy,(cx-1)%L]>=0)!=lado: perim+=1
                    if not ad[cy,cx] or (phi[(cy+1)%L,cx]>=0)!=lado: perim+=1
                    if not ad[(cy-1)%L,cx] or (phi[(cy-1)%L,cx]>=0)!=lado: perim+=1
                    for ny,nx,cond in [(cy,(cx+1)%L, ar[cy,cx]), (cy,(cx-1)%L, ar[cy,(cx-1)%L]), ((cy+1)%L,cx, ad[cy,cx]), ((cy-1)%L,cx, ad[(cy-1)%L,cx])]:
                        if cond and not visto[ny,nx] and (phi[ny,nx]>=0)==lado:
                            visto[ny,nx]=True; q.append((ny,nx)); nodes.append((ny,nx)); sum_phi+=phi[ny,nx]; sum_rho+=abs(phi[ny,nx])
                k=len(nodes); phi_avg=sum_phi/k if k>0 else 0
                m=y0*abs(phi_avg)*sum_rho
                clusters.append((k,perim,m,phi_avg))
        k1_m=[c[2] for c in clusters if c[0]==1]; k3_m=[c[2] for c in clusters if c[0]==3 and c[1]==8]
        mk1=np.mean(k1_m) if k1_m else 0; mk3=np.mean(k3_m) if k3_m else 0; ratio=mk1/(mk3+1e-30) if mk3>0 else 0
        print(f"step {step:3d} a={a:5.1f} rho={rho:.2e} lam={lam:.3e} <phi>={phi.mean():.3f} k1={len(k1_m)} k3={len(k3_m)} ratio={ratio:.5f}")
print("FIN B")
