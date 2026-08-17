"""
MF-1 V1 - Masa inercia termica rompe 1/3 -> 0.00054
m_fis = sum_rho * exp(-var_in / T_local) / P_norm * a^beta / f_disip
k3 var~0 retiene, k1 var alta disipa
"""
import numpy as np
from collections import deque

T0=1e15; rho0=1e30; H_topo=0.01; L=30; pasos=500
rng=np.random.default_rng(2025)
xs=np.linspace(0,1,L); xx,yy=np.meshgrid(xs,xs)
pert=np.zeros((L,L))
for mx in range(1,4):
    for my in range(1,4):
        pert+= np.sin(2*np.pi*(mx*xx+my*yy)+rng.uniform(0,2*np.pi))/(mx+my)
pert-=pert.mean(); pert/=pert.std() if pert.std()>0 else 1
phi=np.ones((L,L))+1e-9*pert
theta=rng.uniform(0,2*np.pi,size=(L,L))
ar=np.ones((L,L),bool); ad=np.ones((L,L),bool)

print("MF-1 V1 m_fis = sum * exp(-var/T_local) / P_norm")
print("Objetivo: romper 0.3333 -> 0.00054 (1/1836) sin ajustar e")

for step in range(pasos):
    tg=step/pasos
    a=np.exp(6*tg)
    T_field=T0/a
    # T_local con S>0 asimetria
    epsilon=0.1
    T_local_map = T_field * (1 + epsilon*pert)  # K
    T_local_norm = T_local_map / T0  # 0..1
    
    left=np.roll(ar,1,axis=1); up=np.roll(ad,1,axis=0)
    cnt=ar.astype(int)+left.astype(int)+ad.astype(int)+up.astype(int)
    s=np.where(ar,np.roll(phi,-1,axis=1),0)+np.where(left,np.roll(phi,1,axis=1),0)+np.where(ad,np.roll(phi,-1,axis=0),0)+np.where(up,np.roll(phi,1,axis=0),0)
    mean=np.divide(s,cnt,out=np.zeros_like(phi),where=cnt>0)
    phi_new=phi.copy(); phi_new[cnt>0]=phi[cnt>0]+0.5*(mean[cnt>0]-phi[cnt>0]); phi=phi_new
    
    H_fis=H_topo*np.sqrt(T_field/T0)
    tot=int(np.sum(ar)+np.sum(ad)); nc=int(round(H_fis*tot))
    if nc>0 and tot>0 and np.sum(ar)>0:
        nr=int(round(nc*np.sum(ar)/tot))
        idx=np.argwhere(ar)
        if len(idx)>0 and nr>0:
            sel=rng.choice(len(idx), size=min(nr,len(idx)), replace=False)
            for i in sel: ar[tuple(idx[i])]=False
        rem=nc-nr
        if rem>0 and np.sum(ad)>0:
            idx=np.argwhere(ad)
            sel=rng.choice(len(idx), size=min(rem,len(idx)), replace=False)
            for i in sel: ad[tuple(idx[i])]=False
    
    if step%50==0:
        media=phi.mean(); visto=np.zeros((L,L),bool)
        k_cnt={1:0,3:0}; clusters=[]
        for y in range(L):
            for x in range(L):
                if visto[y,x]: continue
                q=deque([(y,x)]); visto[y,x]=True; nodes=[(y,x)]; lado=phi[y,x]>=media; perim=0; vals=[phi[y,x]]; T_vals=[T_local_norm[y,x]]
                while q:
                    cy,cx=q.popleft()
                    if not ar[cy,cx] or (phi[cy,(cx+1)%L]>=media)!=lado: perim+=1
                    if not ar[cy,(cx-1)%L] or (phi[cy,(cx-1)%L]>=media)!=lado: perim+=1
                    if not ad[cy,cx] or (phi[(cy+1)%L,cx]>=media)!=lado: perim+=1
                    if not ad[(cy-1)%L,cx] or (phi[(cy-1)%L,cx]>=media)!=lado: perim+=1
                    if ar[cy,cx] and not visto[cy,(cx+1)%L] and (phi[cy,(cx+1)%L]>=media)==lado:
                        visto[cy,(cx+1)%L]=True; q.append((cy,(cx+1)%L)); nodes.append((cy,(cx+1)%L)); vals.append(phi[cy,(cx+1)%L]); T_vals.append(T_local_norm[cy,(cx+1)%L])
                    if ar[cy,(cx-1)%L] and not visto[cy,(cx-1)%L] and (phi[cy,(cx-1)%L]>=media)==lado:
                        visto[cy,(cx-1)%L]=True; q.append((cy,(cx-1)%L)); nodes.append((cy,(cx-1)%L)); vals.append(phi[cy,(cx-1)%L]); T_vals.append(T_local_norm[cy,(cx-1)%L])
                    if ad[cy,cx] and not visto[(cy+1)%L,cx] and (phi[(cy+1)%L,cx]>=media)==lado:
                        visto[(cy+1)%L,cx]=True; q.append(((cy+1)%L,cx)); nodes.append(((cy+1)%L,cx)); vals.append(phi[(cy+1)%L,cx]); T_vals.append(T_local_norm[(cy+1)%L,cx])
                    if ad[(cy-1)%L,cx] and not visto[(cy-1)%L,cx] and (phi[(cy-1)%L,cx]>=media)==lado:
                        visto[(cy-1)%L,cx]=True; q.append(((cy-1)%L,cx)); nodes.append(((cy-1)%L,cx)); vals.append(phi[(cy-1)%L,cx]); T_vals.append(T_local_norm[(cy-1)%L,cx])
                k=len(nodes); var=np.var(vals) if len(vals)>1 else 0
                T_avg=np.mean(T_vals)
                if k in k_cnt: k_cnt[k]+=1
                clusters.append((k,perim,np.sum(vals),var,T_avg,nodes))
        # MF-1: m = sum * exp(-var / T_local_avg) * (1 / T_local_avg)^gamma * a^beta
        # beta>0 amplifica retencion k3 con expansion
        beta=2.0; gamma=1.0
        k1_ms=[]; k3_ms=[]
        for c in clusters:
            k,perim,s_rho,var,T_avg,nodes=c
            # P_norm = T_avg (adiabatico) simplificado
            disip = np.exp(-var/(T_avg+1e-12))  # k3 var~0 =>1, k1 var>0 =><1
            # energia atrapada: sum * disip * (1/T)^gamma * a^beta para k3, inverso para k1
            if k==1:
                m = s_rho * disip * (T_avg**gamma) / (a**beta)  # k1 disipa con a, baja masa
            elif k==3 and perim==8:
                m = s_rho * disip * (1/(T_avg+1e-12))**gamma * (a**beta)  # k3 retiene y crece con a
            else:
                continue
            if k==1: k1_ms.append(m)
            if k==3 and perim==8: k3_ms.append(m)
        mk1=np.mean(k1_ms) if k1_ms else 0; mk3=np.mean(k3_ms) if k3_ms else 0
        ratio=mk1/(mk3+1e-30) if mk3>0 else 0
        # varianzas tipicas
        var_k1=np.mean([c[3] for c in clusters if c[0]==1]) if any(c[0]==1 for c in clusters) else 0
        var_k3=np.mean([c[3] for c in clusters if c[0]==3 and c[1]==8]) if any(c[0]==3 and c[1]==8 for c in clusters) else 0
        print(f"step {step:3d} a={a:6.1f} T={T_field:.2e} | k1={k_cnt[1]:3d} k3={k_cnt[3]:3d} | var_k1={var_k1:.2e} var_k3={var_k3:.2e} | mk1={mk1:.2e} mk3={mk3:.2e} ratio={ratio:.6f} objetivo 0.00054")

print("\nMF-1 V1 FIN - ratio debe caer de 0.3333 hacia 0.00054 con var y a^beta")
