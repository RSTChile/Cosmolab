"""
08_MF6_topological_charge - Q = (1/2pi) sum theta winding per cluster
m = |Q| * rho0/a^3 * exp(-var/T)
k3 con Q=1 retiene, k1 Q=0 disipa
"""
import numpy as np
from collections import deque
T0=1e15; rho0=1e30; H_topo=0.01; L=30; pasos=500
rng=np.random.default_rng(2025)
phi=np.ones((L,L))+0.5*rng.normal(size=(L,L))
theta=rng.uniform(0,2*np.pi,size=(L,L))
ar=np.ones((L,L),bool); ad=np.ones((L,L),bool)
print("08 MF6 Q topologico")
for step in range(pasos):
    tg=step/pasos; a=np.exp(6*tg); T=T0/a
    left=np.roll(ar,1,axis=1); up=np.roll(ad,1,axis=0)
    cnt=ar.astype(int)+left.astype(int)+ad.astype(int)+up.astype(int)
    s=np.where(ar,np.roll(phi,-1,axis=1),0)+np.where(left,np.roll(phi,1,axis=1),0)+np.where(ad,np.roll(phi,-1,axis=0),0)+np.where(up,np.roll(phi,1,axis=0),0)
    mean=np.divide(s,cnt,out=np.zeros_like(phi),where=cnt>0)
    phi_new=phi.copy(); phi_new[cnt>0]=phi[cnt>0]+0.3*(mean[cnt>0]-phi[cnt>0]); phi=phi_new
    K=np.exp(-1/(a+1e-10))
    dth=np.where(ar,np.sin(np.roll(theta,-1,axis=1)-theta),0)+np.where(left,np.sin(np.roll(theta,1,axis=1)-theta),0)+np.where(ad,np.sin(np.roll(theta,-1,axis=0)-theta),0)+np.where(up,np.sin(np.roll(theta,1,axis=0)-theta),0)
    theta=np.mod(theta+0.1*K*dth,2*np.pi)
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
                q=deque([(y,x)]); visto[y,x]=True; nodes=[(y,x)]; lado=phi[y,x]>=media; perim=0
                while q:
                    cy,cx=q.popleft()
                    if not ar[cy,cx] or (phi[cy,(cx+1)%L]>=media)!=lado: perim+=1
                    if not ar[cy,(cx-1)%L] or (phi[cy,(cx-1)%L]>=media)!=lado: perim+=1
                    if not ad[cy,cx] or (phi[(cy+1)%L,cx]>=media)!=lado: perim+=1
                    if not ad[(cy-1)%L,cx] or (phi[(cy-1)%L,cx]>=media)!=lado: perim+=1
                    if ar[cy,cx] and not visto[cy,(cx+1)%L] and (phi[cy,(cx+1)%L]>=media)==lado:
                        visto[cy,(cx+1)%L]=True; q.append((cy,(cx+1)%L)); nodes.append((cy,(cx+1)%L))
                    if ar[cy,(cx-1)%L] and not visto[cy,(cx-1)%L] and (phi[cy,(cx-1)%L]>=media)==lado:
                        visto[cy,(cx-1)%L]=True; q.append((cy,(cx-1)%L)); nodes.append((cy,(cx-1)%L))
                    if ad[cy,cx] and not visto[(cy+1)%L,cx] and (phi[(cy+1)%L,cx]>=media)==lado:
                        visto[(cy+1)%L,cx]=True; q.append(((cy+1)%L,cx)); nodes.append(((cy+1)%L,cx))
                    if ad[(cy-1)%L,cx] and not visto[(cy-1)%L,cx] and (phi[(cy-1)%L,cx]>=media)==lado:
                        visto[(cy-1)%L,cx]=True; q.append(((cy-1)%L,cx)); nodes.append(((cy-1)%L,cx))
                k=len(nodes)
                # Q = winding theta
                wind=0
                for (yy,xx) in nodes:
                    for dy,dx in [(0,1),(1,0)]:
                        ny,nx=(yy+dy)%L,(xx+dx)%L
                        if (ny,nx) in nodes:
                            d= np.angle(np.exp(1j*(theta[ny,nx]-theta[yy,xx])))
                            wind+=d
                Q=wind/(2*np.pi)
                clusters.append((k,perim,abs(Q),len(nodes)))
        k1_Q=np.mean([c[2] for c in clusters if c[0]==1]) if any(c[0]==1 for c in clusters) else 0
        k3_Q=np.mean([c[2] for c in clusters if c[0]==3 and c[1]==8]) if any(c[0]==3 and c[1]==8 for c in clusters) else 0
        print(f"step {step:3d} a={a:5.1f} K={K:.3f} Q_k1={k1_Q:.3f} Q_k3_perim8={k3_Q:.3f} k1={sum(1 for c in clusters if c[0]==1)} k3={sum(1 for c in clusters if c[0]==3 and c[1]==8)}")
print("FIN 08: Q topologico discrimina k3 coherente vs k1 trivial?")
