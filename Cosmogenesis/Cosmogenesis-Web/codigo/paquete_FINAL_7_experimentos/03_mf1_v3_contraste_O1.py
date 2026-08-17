
"""
MF-1 V3 - Contraste O(1) real phi = 1 + 0.5*pert (no 1e-9)
Formula UNICA: m = sum_rho * exp(-var_3x3 / T_norm)
Objetivo honesto ver si var discrimina k1 vs k3
"""
import numpy as np
from collections import deque
T0=1e15; H_topo=0.01; L=30; pasos=500
rng=np.random.default_rng(2025)
xs=np.linspace(0,1,L); xx,yy=np.meshgrid(xs,xs)
pert=np.zeros((L,L))
for mx in range(1,4):
    for my in range(1,4):
        pert+= np.sin(2*np.pi*(mx*xx+my*yy)+rng.uniform(0,2*np.pi))/(mx+my)
pert-=pert.mean(); pert/=pert.std() if pert.std()>0 else 1
phi=np.ones((L,L))+0.5*pert  # O(1) contraste
theta=rng.uniform(0,2*np.pi,size=(L,L))
ar=np.ones((L,L),bool); ad=np.ones((L,L),bool)

def var3x3(phi):
    vm=np.zeros_like(phi)
    for y in range(L):
        for x in range(L):
            v=[phi[(y+dy)%L,(x+dx)%L] for dy in [-1,0,1] for dx in [-1,0,1]]
            vm[y,x]=np.var(v)
    return vm

print("MF-1 V3 contraste O(1) phi=1+0.5*pert formula unica")
for step in range(pasos):
    tg=step/pasos; a=np.exp(6*tg); T=T0/a
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
        media=phi.mean(); visto=np.zeros((L,L),bool); clusters=[]; k_cnt={1:0,3:0}; perim8=0
        vm=var3x3(phi)
        for y in range(L):
            for x in range(L):
                if visto[y,x]: continue
                q=deque([(y,x)]); visto[y,x]=True; nodes=[(y,x)]; lado=phi[y,x]>=media; perim=0; vals=[phi[y,x]]; vvals=[vm[y,x]]
                while q:
                    cy,cx=q.popleft()
                    if not ar[cy,cx] or (phi[cy,(cx+1)%L]>=media)!=lado: perim+=1
                    if not ar[cy,(cx-1)%L] or (phi[cy,(cx-1)%L]>=media)!=lado: perim+=1
                    if not ad[cy,cx] or (phi[(cy+1)%L,cx]>=media)!=lado: perim+=1
                    if not ad[(cy-1)%L,cx] or (phi[(cy-1)%L,cx]>=media)!=lado: perim+=1
                    if ar[cy,cx] and not visto[cy,(cx+1)%L] and (phi[cy,(cx+1)%L]>=media)==lado:
                        visto[cy,(cx+1)%L]=True; q.append((cy,(cx+1)%L)); nodes.append((cy,(cx+1)%L)); vals.append(phi[cy,(cx+1)%L]); vvals.append(vm[cy,(cx+1)%L])
                    if ar[cy,(cx-1)%L] and not visto[cy,(cx-1)%L] and (phi[cy,(cx-1)%L]>=media)==lado:
                        visto[cy,(cx-1)%L]=True; q.append((cy,(cx-1)%L)); nodes.append((cy,(cx-1)%L)); vals.append(phi[cy,(cx-1)%L]); vvals.append(vm[cy,(cx-1)%L])
                    if ad[cy,cx] and not visto[(cy+1)%L,cx] and (phi[(cy+1)%L,cx]>=media)==lado:
                        visto[(cy+1)%L,cx]=True; q.append(((cy+1)%L,cx)); nodes.append(((cy+1)%L,cx)); vals.append(phi[(cy+1)%L,cx]); vvals.append(vm[(cy+1)%L,cx])
                    if ad[(cy-1)%L,cx] and not visto[(cy-1)%L,cx] and (phi[(cy-1)%L,cx]>=media)==lado:
                        visto[(cy-1)%L,cx]=True; q.append(((cy-1)%L,cx)); nodes.append(((cy-1)%L,cx)); vals.append(phi[(cy-1)%L,cx]); vvals.append(vm[(cy-1)%L,cx])
                k=len(nodes)
                if k in k_cnt: k_cnt[k]+=1
                if k==3 and perim==8: perim8+=1
                clusters.append((k,perim,np.sum(vals),np.mean(vvals),len(vals)))
        T_norm=T/T0+1e-6
        def m_unica(c): return c[2]*np.exp(-c[3]/(T_norm))
        k1_m=[m_unica(c) for c in clusters if c[0]==1]; k3_m=[m_unica(c) for c in clusters if c[0]==3 and c[1]==8]
        mk1=np.mean(k1_m) if k1_m else 0; mk3=np.mean(k3_m) if k3_m else 0; ratio=mk1/(mk3+1e-30) if mk3>0 else 0
        var_k1=np.mean([c[3] for c in clusters if c[0]==1]) if k_cnt[1]>0 else 0; var_k3=np.mean([c[3] for c in clusters if c[0]==3 and c[1]==8]) if perim8>0 else 0
        print(f"step {step:3d} a={a:5.1f} T={T:.2e} k1={k_cnt[1]:3d} k3={k_cnt[3]:3d} perim8={perim8:2d} var_k1={var_k1:.4e} var_k3={var_k3:.4e} ratio={ratio:.4f}")
print("FIN MF-1 V3")
