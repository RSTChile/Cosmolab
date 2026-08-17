"""
15_Higgs_D_effective_potential - masa = curvatura d2E/dx2 del potencial efectivo
E_k = perim * T_norm + sum_rho * var_in
d2E medido como var_in / T_norm
m = y0 * d2E formula unica
"""
import numpy as np
from collections import deque
T0=1e15; H_topo=0.01; L=24; pasos=400; y0=0.3
rng=np.random.default_rng(2025)
phi=np.ones((L,L))+0.5*rng.normal(size=(L,L))
ar=np.ones((L,L),bool); ad=np.ones((L,L),bool)
print("15 Higgs D potencial efectivo m = y0 * d2E/dx2")
for step in range(pasos):
    tg=step/pasos; a=np.exp(6*tg); T=T0/a; T_norm=T/T0+1e-6
    left=np.roll(ar,1,axis=1); up=np.roll(ad,1,axis=0)
    cnt=ar.astype(int)+left.astype(int)+ad.astype(int)+up.astype(int)
    s=np.where(ar,np.roll(phi,-1,axis=1),0)+np.where(left,np.roll(phi,1,axis=1),0)+np.where(ad,np.roll(phi,-1,axis=0),0)+np.where(up,np.roll(phi,1,axis=0),0)
    mean=np.divide(s,cnt,out=np.zeros_like(phi),where=cnt>0)
    phi_new=phi.copy(); phi_new[cnt>0]=phi[cnt>0]+0.2*(mean[cnt>0]-phi[cnt>0]); phi=phi_new
    # var 3x3
    var3=np.zeros_like(phi)
    for y in range(L):
        for x in range(L):
            vals=[phi[(y+dy)%L,(x+dx)%L] for dy in [-1,0,1] for dx in [-1,0,1]]
            var3[y,x]=np.var(vals)
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
                q=deque([(y,x)]); visto[y,x]=True; nodes=[(y,x)]; lado=phi[y,x]>=media; perim=0; sum_rho=phi[y,x]; sum_var=var3[y,x]
                while q:
                    cy,cx=q.popleft()
                    if not ar[cy,cx] or (phi[cy,(cx+1)%L]>=media)!=lado: perim+=1
                    if not ar[cy,(cx-1)%L] or (phi[cy,(cx-1)%L]>=media)!=lado: perim+=1
                    if not ad[cy,cx] or (phi[(cy+1)%L,cx]>=media)!=lado: perim+=1
                    if not ad[(cy-1)%L,cx] or (phi[(cy-1)%L,cx]>=media)!=lado: perim+=1
                    for ny,nx,cond in [(cy,(cx+1)%L, ar[cy,cx]), (cy,(cx-1)%L, ar[cy,(cx-1)%L]), ((cy+1)%L,cx, ad[cy,cx]), ((cy-1)%L,cx, ad[(cy-1)%L,cx])]:
                        if cond and not visto[ny,nx] and (phi[ny,nx]>=media)==lado:
                            visto[ny,nx]=True; q.append((ny,nx)); nodes.append((ny,nx)); sum_rho+=phi[ny,nx]; sum_var+=var3[ny,nx]
                k=len(nodes); var_avg=sum_var/k if k>0 else 0
                # curvatura = perim/T_norm + var/T_norm
                d2E = perim/T_norm + var_avg/T_norm
                m = y0*d2E
                clusters.append((k,perim,m,var_avg))
        k1_m=[c[2] for c in clusters if c[0]==1]; k3_m=[c[2] for c in clusters if c[0]==3 and c[1]==8]
        mk1=np.mean(k1_m) if k1_m else 0; mk3=np.mean(k3_m) if k3_m else 0; ratio=mk1/(mk3+1e-30) if mk3>0 else 0
        print(f"step {step:3d} a={a:5.1f} T_norm={T_norm:.2e} k1={len(k1_m)} k3={len(k3_m)} ratio={ratio:.5f}")
print("FIN D")
