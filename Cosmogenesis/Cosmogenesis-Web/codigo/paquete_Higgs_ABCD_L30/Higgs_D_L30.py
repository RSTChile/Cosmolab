"""
HIGGS D CORREGIDO L=30 - curvatura efectiva d2E = perim * (1+var)/T_norm
m = y0 * d2E , NULL var=0 perim solo
"""
import numpy as np
from collections import deque
T0=1e15; H_topo=0.01; L=30; pasos=500; y0=1.0
rng=np.random.default_rng(2025)
phi=np.ones((L,L))+0.8*np.sin(np.linspace(0,4*np.pi,L))[:,None]*np.cos(np.linspace(0,4*np.pi,L))[None,:]*0.5
ar=np.ones((L,L),bool); ad=np.ones((L,L),bool)
print("HIGGS D L=30 curvatura d2E")
for step in range(pasos):
    tg=step/pasos; a=np.exp(6*tg); T=T0/a
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
        # var local 3x3
        var3=np.zeros_like(phi)
        for y in range(L):
            for x in range(L):
                v=[phi[(y+dy)%L,(x+dx)%L] for dy in [-1,0,1] for dx in [-1,0,1]]
                var3[y,x]=np.var(v)
        media=phi.mean(); visto=np.zeros((L,L),bool); clusters=[]
        for y in range(L):
            for x in range(L):
                if visto[y,x]: continue
                q=deque([(y,x)]); visto[y,x]=True; nodes=[(y,x)]; lado=phi[y,x]>=media; perim=0; sum_var=var3[y,x]
                while q:
                    cy,cx=q.popleft()
                    if not ar[cy,cx] or (phi[cy,(cx+1)%L]>=media)!=lado: perim+=1
                    if not ar[cy,(cx-1)%L] or (phi[cy,(cx-1)%L]>=media)!=lado: perim+=1
                    if not ad[cy,cx] or (phi[(cy+1)%L,cx]>=media)!=lado: perim+=1
                    if not ad[(cy-1)%L,cx] or (phi[(cy-1)%L,cx]>=media)!=lado: perim+=1
                    if ar[cy,cx] and not visto[cy,(cx+1)%L] and (phi[cy,(cx+1)%L]>=media)==lado:
                        visto[cy,(cx+1)%L]=True; q.append((cy,(cx+1)%L)); nodes.append((cy,(cx+1)%L)); sum_var+=var3[cy,(cx+1)%L]
                    if ar[cy,(cx-1)%L] and not visto[cy,(cx-1)%L] and (phi[cy,(cx-1)%L]>=media)==lado:
                        visto[cy,(cx-1)%L]=True; q.append((cy,(cx-1)%L)); nodes.append((cy,(cx-1)%L)); sum_var+=var3[cy,(cx-1)%L]
                    if ad[cy,cx] and not visto[(cy+1)%L,cx] and (phi[(cy+1)%L,cx]>=media)==lado:
                        visto[(cy+1)%L,cx]=True; q.append(((cy+1)%L,cx)); nodes.append(((cy+1)%L,cx)); sum_var+=var3[(cy+1)%L,cx]
                    if ad[(cy-1)%L,cx] and not visto[(cy-1)%L,cx] and (phi[(cy-1)%L,cx]>=media)==lado:
                        visto[(cy-1)%L,cx]=True; q.append(((cy-1)%L,cx)); nodes.append(((cy-1)%L,cx)); sum_var+=var3[(cy-1)%L,cx]
                k=len(nodes); avg_var=sum_var/k
                T_norm=T/T0+1e-6
                d2E = perim * (1+avg_var) / T_norm
                d2E_null = perim / T_norm  # var=0
                clusters.append((k,perim,d2E,d2E_null,avg_var))
        k1=[c[2] for c in clusters if c[0]==1]; k3=[c[2] for c in clusters if c[0]==3 and c[1]==8]
        k1_n=[c[3] for c in clusters if c[0]==1]; k3_n=[c[3] for c in clusters if c[0]==3 and c[1]==8]
        ratio=np.mean(k1)/(np.mean(k3)+1e-30) if k3 and k1 else 0
        ratio_null=np.mean(k1_n)/(np.mean(k3_n)+1e-30) if k3_n and k1_n else 0
        print(f"step {step:3d} a={a:6.1f} k1={len(k1)} k3={len(k3)} ratio={ratio:.3f} ratio_NULL={ratio_null:.3f} <var>={np.mean(var3):.2e}")
print("FIN D")
