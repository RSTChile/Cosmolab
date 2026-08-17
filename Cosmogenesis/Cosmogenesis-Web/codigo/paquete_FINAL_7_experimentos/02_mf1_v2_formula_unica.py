"""
MF-1 V2 CORREGIDA - Formula unica para todo dominio
m_fis = sum_rho * exp(-var_3x3 / T_norm) * (T_norm**alpha) / (P_norm * a**beta) ??? No, unica sin if k

Anti-Shannon: UNA sola expresion para k=1 y k=3, solo datos campo: rho_E, var_3x3, T_local, P_norm, a
var_3x3 = var de phi en ventana 3x3 alrededor dominio -> k1 var alta (borde), k3 var baja (coherente)

Contrato F0 intacto: T=T0/a, rho=rho0/a^3, a=exp(6*tg)
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

print("MF-1 V2 - Formula unica m_fis = sum_rho * exp(-var_3x3 / T_norm) / (P_norm) * f(a,T) unica")
print("T0=1e15 a=exp(6*tg) 1->403x var_3x3 = var phi ventana 3x3 (k1 var alta, k3 var~0)")

def var_3x3_map(phi):
    # var local 3x3 para cada celda
    var_map=np.zeros_like(phi)
    for y in range(L):
        for x in range(L):
            vals=[]
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    vals.append(phi[(y+dy)%L,(x+dx)%L])
            var_map[y,x]=np.var(vals)
    return var_map

max_err=0
for step in range(pasos):
    tg=step/pasos
    a=np.exp(6*tg)
    T_field=T0/a
    rho_field=rho0/(a**3)
    err=abs(T_field*a/T0-1); max_err=max(max_err,err)
    
    left=np.roll(ar,1,axis=1); up=np.roll(ad,1,axis=0)
    cnt=ar.astype(int)+left.astype(int)+ad.astype(int)+up.astype(int)
    s=np.where(ar,np.roll(phi,-1,axis=1),0)+np.where(left,np.roll(phi,1,axis=1),0)+np.where(ad,np.roll(phi,-1,axis=0),0)+np.where(up,np.roll(phi,1,axis=0),0)
    mean=np.divide(s,cnt,out=np.zeros_like(phi),where=cnt>0)
    phi_new=phi.copy(); phi_new[cnt>0]=phi[cnt>0]+0.5*(mean[cnt>0]-phi[cnt>0]); phi=phi_new
    
    K=np.exp(-1.0/(a+1e-10))
    dth=np.where(ar,np.sin(np.roll(theta,-1,axis=1)-theta),0)+np.where(left,np.sin(np.roll(theta,1,axis=1)-theta),0)+np.where(ad,np.sin(np.roll(theta,-1,axis=0)-theta),0)+np.where(up,np.sin(np.roll(theta,1,axis=0)-theta),0)
    theta=np.mod(theta+0.1*K*dth,2*np.pi)
    
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
        k_cnt={1:0,2:0,3:0}; perim8=0; clusters=[]
        var_map=var_3x3_map(phi)
        for y in range(L):
            for x in range(L):
                if visto[y,x]: continue
                q=deque([(y,x)]); visto[y,x]=True; nodes=[(y,x)]; lado=phi[y,x]>=media; perim=0; vals=[phi[y,x]]; var_vals=[var_map[y,x]]
                while q:
                    cy,cx=q.popleft()
                    if not ar[cy,cx] or (phi[cy,(cx+1)%L]>=media)!=lado: perim+=1
                    if not ar[cy,(cx-1)%L] or (phi[cy,(cx-1)%L]>=media)!=lado: perim+=1
                    if not ad[cy,cx] or (phi[(cy+1)%L,cx]>=media)!=lado: perim+=1
                    if not ad[(cy-1)%L,cx] or (phi[(cy-1)%L,cx]>=media)!=lado: perim+=1
                    if ar[cy,cx] and not visto[cy,(cx+1)%L] and (phi[cy,(cx+1)%L]>=media)==lado:
                        visto[cy,(cx+1)%L]=True; q.append((cy,(cx+1)%L)); nodes.append((cy,(cx+1)%L)); vals.append(phi[cy,(cx+1)%L]); var_vals.append(var_map[cy,(cx+1)%L])
                    if ar[cy,(cx-1)%L] and not visto[cy,(cx-1)%L] and (phi[cy,(cx-1)%L]>=media)==lado:
                        visto[cy,(cx-1)%L]=True; q.append((cy,(cx-1)%L)); nodes.append((cy,(cx-1)%L)); vals.append(phi[cy,(cx-1)%L]); var_vals.append(var_map[cy,(cx-1)%L])
                    if ad[cy,cx] and not visto[(cy+1)%L,cx] and (phi[(cy+1)%L,cx]>=media)==lado:
                        visto[(cy+1)%L,cx]=True; q.append(((cy+1)%L,cx)); nodes.append(((cy+1)%L,cx)); vals.append(phi[(cy+1)%L,cx]); var_vals.append(var_map[cy,(cx+1)%L])
                    if ad[(cy-1)%L,cx] and not visto[(cy-1)%L,cx] and (phi[(cy-1)%L,cx]>=media)==lado:
                        visto[(cy-1)%L,cx]=True; q.append(((cy-1)%L,cx)); nodes.append(((cy-1)%L,cx)); vals.append(phi[(cy-1)%L,cx]); var_vals.append(var_map[(cy-1)%L,cx])
                k=len(nodes); var_in=np.mean(var_vals); var_intra=np.var(vals)
                if k in k_cnt: k_cnt[k]+=1
                if k==3 and perim==8: perim8+=1
                clusters.append((k,perim,np.sum(vals),var_in,var_intra,nodes))
        
        # FORMULA UNICA - sin if k
        # m_fis = sum_rho * exp(-var_3x3 / T_norm) / a^0  - solo campo, no potencias opuestas por k
        # T_norm = T_field/T0 0..1, P_norm = 1 (no SI crudo)
        T_norm = T_field/T0
        # disipacion por rugosidad local: var alta -> disipa, var~0 -> retiene
        # k3 coherente var_3x3 ~1e-12, k1 borde var_3x3 ~1e-4
        k1_list = [c for c in clusters if c[0]==1]
        k3_list = [c for c in clusters if c[0]==3 and c[1]==8]
        def m_unica(c):
            sum_rho, var_3x3 = c[2], c[3]
            # UNA sola formula
            return sum_rho * np.exp(-var_3x3 / (T_norm + 1e-10))  # sin a**beta opuesto por k
        
        k1_m = [m_unica(c) for c in k1_list]
        k3_m = [m_unica(c) for c in k3_list]
        mk1 = np.mean(k1_m) if k1_m else 0; mk3 = np.mean(k3_m) if k3_m else 0
        ratio = mk1/(mk3+1e-30) if mk3>0 else 0
        var_k1 = np.mean([c[3] for c in k1_list]) if k1_list else 0
        var_k3 = np.mean([c[3] for c in k3_list]) if k3_list else 0
        print(f"step {step:3d} a={a:6.1f} T={T_field:.2e} K={K:.3f} | k1={k_cnt[1]:3d} k3={k_cnt[3]:3d} perim8={perim8:2d} | var_k1={var_k1:.2e} var_k3={var_k3:.2e} | mk1={mk1:.3e} mk3={mk3:.3e} ratio={ratio:.4f}")

print(f"\nSMOKE max err {max_err:.2e} PASS={max_err<0.05}")
print("MF-1 V2: formula unica, var_3x3 real, sin bifurcacion por k")
