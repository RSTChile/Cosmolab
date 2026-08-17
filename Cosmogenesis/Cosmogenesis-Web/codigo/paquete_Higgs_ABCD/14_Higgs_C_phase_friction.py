"""
14_Higgs_C_phase_friction - eta_k = int |grad theta|^2
m_eff = y0 * eta_k * sum_rho ? invertido: eta bajo = arrastra mucho = pesado
Formula unica: m = y0 * sum_rho / (eta+eps)  -> coherente pesado
"""
import numpy as np
from collections import deque
T0=1e15; H_topo=0.01; L=24; pasos=400; y0=0.3
rng=np.random.default_rng(2025)
phi=np.ones((L,L))+0.3*rng.normal(size=(L,L))
theta=rng.uniform(0,2*np.pi,size=(L,L))
ar=np.ones((L,L),bool); ad=np.ones((L,L),bool)
print("14 Higgs C friccion fase eta=|grad theta|^2 m=sum/(eta+eps)")
for step in range(pasos):
    tg=step/pasos; a=np.exp(6*tg); T=T0/a; K=np.exp(-1/(a+1e-10))
    dth=np.where(ar,np.sin(np.roll(theta,-1,axis=1)-theta),0)+np.where(np.roll(ar,1,axis=1),np.sin(np.roll(theta,1,axis=1)-theta),0)+np.where(ad,np.sin(np.roll(theta,-1,axis=0)-theta),0)+np.where(np.roll(ad,1,axis=0),np.sin(np.roll(theta,1,axis=0)-theta),0)
    theta=np.mod(theta+0.1*K*dth,2*np.pi)
    grad_x = np.angle(np.exp(1j*(np.roll(theta,-1,axis=1)-theta)))
    grad_y = np.angle(np.exp(1j*(np.roll(theta,-1,axis=0)-theta)))
    grad2 = grad_x**2+grad_y**2
    left=np.roll(ar,1,axis=1); up=np.roll(ad,1,axis=0)
    cnt=ar.astype(int)+left.astype(int)+ad.astype(int)+up.astype(int)
    s=np.where(ar,np.roll(phi,-1,axis=1),0)+np.where(left,np.roll(phi,1,axis=1),0)+np.where(ad,np.roll(phi,-1,axis=0),0)+np.where(up,np.roll(phi,1,axis=0),0)
    mean=np.divide(s,cnt,out=np.zeros_like(phi),where=cnt>0)
    phi_new=phi.copy(); phi_new[cnt>0]=phi[cnt>0]+0.2*(mean[cnt>0]-phi[cnt>0]); phi=phi_new
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
                q=deque([(y,x)]); visto[y,x]=True; nodes=[(y,x)]; lado=phi[y,x]>=media; perim=0; sum_rho=phi[y,x]; sum_eta=grad2[y,x]
                while q:
                    cy,cx=q.popleft()
                    if not ar[cy,cx] or (phi[cy,(cx+1)%L]>=media)!=lado: perim+=1
                    if not ar[cy,(cx-1)%L] or (phi[cy,(cx-1)%L]>=media)!=lado: perim+=1
                    if not ad[cy,cx] or (phi[(cy+1)%L,cx]>=media)!=lado: perim+=1
                    if not ad[(cy-1)%L,cx] or (phi[(cy-1)%L,cx]>=media)!=lado: perim+=1
                    for ny,nx,cond in [(cy,(cx+1)%L, ar[cy,cx]), (cy,(cx-1)%L, ar[cy,(cx-1)%L]), ((cy+1)%L,cx, ad[cy,cx]), ((cy-1)%L,cx, ad[(cy-1)%L,cx])]:
                        if cond and not visto[ny,nx] and (phi[ny,nx]>=media)==lado:
                            visto[ny,nx]=True; q.append((ny,nx)); nodes.append((ny,nx)); sum_rho+=phi[ny,nx]; sum_eta+=grad2[ny,nx]
                k=len(nodes); eta_avg=sum_eta/k if k>0 else 0
                # coherente eta~0 -> pesado: m = sum / (eta+eps)
                m = y0*sum_rho/(eta_avg+0.01)
                clusters.append((k,perim,m,eta_avg))
        k1_m=[c[2] for c in clusters if c[0]==1]; k3_m=[c[2] for c in clusters if c[0]==3 and c[1]==8]
        mk1=np.mean(k1_m) if k1_m else 0; mk3=np.mean(k3_m) if k3_m else 0; ratio=mk1/(mk3+1e-30) if mk3>0 else 0
        print(f"step {step:3d} a={a:5.1f} K={K:.3f} <eta>={grad2.mean():.3e} k1={len(k1_m)} k3={len(k3_m)} ratio={ratio:.5f}")
print("FIN C: NULL eta=0 -> ratio 0.3333, si eta discrimina ratio <0.3333")
