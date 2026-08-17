"""
FASE6-HIGGS-B V3 FINAL - friccion fase barrido VIVO H=0.01 L=30
eta_k = int |grad theta|^2, m = sum * exp(-alpha*eta) unica
Barrido D 0.01-0.1, K0 1-3, alpha 0.5-2
Objetivo ventana donde eta_k1 >> eta_k3 sin apagar clusters
"""
import numpy as np
from collections import deque
import itertools

L=30; pasos=350
T0=1e15; H_topo=0.01
rng=np.random.default_rng(2025)

D_list=[0.01,0.05,0.1]
K0_list=[1.0,2.0,3.0]
alpha_list=[0.5,1.0,2.0]

def run_once(D_th,K0,alpha):
    phi=np.ones((L,L))+0.3*rng.normal(size=(L,L))
    theta=rng.uniform(0,2*np.pi,size=(L,L))
    ar=np.ones((L,L),bool); ad=np.ones((L,L),bool)
    for step in range(pasos):
        tg=step/pasos; a=np.exp(6*tg); Tnorm=np.exp(-6*tg)
        K=K0*np.exp(-1/(a+1e-10))
        left=np.roll(ar,1,axis=1); up=np.roll(ad,1,axis=0)
        dth=np.where(ar,np.sin(np.roll(theta,-1,axis=1)-theta),0)+np.where(left,np.sin(np.roll(theta,1,axis=1)-theta),0)+np.where(ad,np.sin(np.roll(theta,-1,axis=0)-theta),0)+np.where(up,np.sin(np.roll(theta,1,axis=0)-theta),0)
        lap_th=(np.roll(theta,-1,axis=1)+np.roll(theta,1,axis=1)+np.roll(theta,-1,axis=0)+np.roll(theta,1,axis=0)-4*theta)
        theta=np.mod(theta+0.1*K*dth+ D_th*0.02*lap_th, 2*np.pi)
        cnt=ar.astype(int)+left.astype(int)+ad.astype(int)+up.astype(int)
        s=np.where(ar,np.roll(phi,-1,axis=1),0)+np.where(left,np.roll(phi,1,axis=1),0)+np.where(ad,np.roll(phi,-1,axis=0),0)+np.where(up,np.roll(phi,1,axis=0),0)
        mean=np.divide(s,cnt,out=np.zeros_like(phi),where=cnt>0)
        phi_new=phi.copy(); phi_new[cnt>0]=phi[cnt>0]+0.3*(mean[cnt>0]-phi[cnt>0]); phi=phi_new
        H_fis=H_topo*np.sqrt(Tnorm+1e-12); tot=int(np.sum(ar)+np.sum(ad)); nc=int(round(H_fis*tot))
        if nc>0 and tot>0:
            idx=np.argwhere(ar)
            if len(idx)>0:
                nr=int(round(nc*np.sum(ar)/tot))
                if nr>0:
                    sel=rng.choice(len(idx), size=min(nr,len(idx)), replace=False)
                    for i in sel: ar[tuple(idx[i])]=False
            rem=nc-nr
            if rem>0:
                idx=np.argwhere(ad)
                if len(idx)>0:
                    sel=rng.choice(len(idx), size=min(rem,len(idx)), replace=False)
                    for i in sel: ad[tuple(idx[i])]=False
    gx=(np.roll(theta,-1,axis=1)-np.roll(theta,1,axis=1))/2; gy=(np.roll(theta,-1,axis=0)-np.roll(theta,1,axis=0))/2
    gx=np.angle(np.exp(1j*gx)); gy=np.angle(np.exp(1j*gy)); grad2=gx**2+gy**2
    media=phi.mean(); visto=np.zeros((L,L),bool); clusters=[]
    for y in range(L):
        for x in range(L):
            if visto[y,x]: continue
            q=deque([(y,x)]); visto[y,x]=True; nodes=[(y,x)]; lado=phi[y,x]>=media; sum_rho=phi[y,x]; sum_grad=grad2[y,x]; perim=0
            while q:
                cy,cx=q.popleft()
                if not ar[cy,cx] or (phi[cy,(cx+1)%L]>=media)!=lado: perim+=1
                if not ar[cy,(cx-1)%L] or (phi[cy,(cx-1)%L]>=media)!=lado: perim+=1
                if not ad[cy,cx] or (phi[(cy+1)%L,cx]>=media)!=lado: perim+=1
                if not ad[(cy-1)%L,cx] or (phi[(cy-1)%L,cx]>=media)!=lado: perim+=1
                for ny,nx,cond in [(cy,(cx+1)%L,ar[cy,cx]),(cy,(cx-1)%L,ar[cy,(cx-1)%L]),((cy+1)%L,cx,ad[cy,cx]),((cy-1)%L,cx,ad[(cy-1)%L,cx])]:
                    if cond and not visto[ny,nx] and (phi[ny,nx]>=media)==lado:
                        visto[ny,nx]=True; q.append((ny,nx)); nodes.append((ny,nx)); sum_rho+=phi[ny,nx]; sum_grad+=grad2[ny,nx]
            k=len(nodes); eta=sum_grad; m=sum_rho*np.exp(-alpha*eta)
            clusters.append((k,perim,eta/k if k>0 else 0,m))
    k1=[c for c in clusters if c[0]==1]; k3=[c for c in clusters if c[0]==3 and c[1]==8]
    if not k1 or not k3: return 0,0,0,0,0,0
    ratio=np.mean([c[3] for c in k1])/(np.mean([c[3] for c in k3])+1e-30)
    ek1=np.mean([c[2] for c in k1]); ek3=np.mean([c[2] for c in k3])
    return ratio,ek1,ek3,len(k1),len(k3),np.mean(grad2)

print("FASE6-HIGGS-B V3 barrido VIVO H=0.01")
for D_th,K0,alpha in itertools.product(D_list,K0_list,alpha_list):
    ratio,ek1,ek3,k1n,k3n,g2 = run_once(D_th,K0,alpha)
    flag="VIVO" if ratio>0 and ratio<0.9 and ek1>ek3*1.1 else "GEOM/APAG"
    print(f"D={D_th:.2f} K0={K0:.1f} a={alpha:.1f} ratio={ratio:.5f} eta_k1={ek1:.4f} eta_k3={ek3:.4f} k1={k1n} k3={k3n} g2={g2:.4f} {flag}")
