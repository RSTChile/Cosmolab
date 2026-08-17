"""
06_MF4_rhoE_real - rho_E = rho0/a^3 * phi^2 * (1 + (grad phi)^2)
Formula unica m = rho_E * exp(-var_grad / T_norm) / T_norm
phi O(1) con energia real, no 1+1e-9
"""
import numpy as np
from collections import deque
T0=1e15; rho0=1e30; H_topo=0.01; L=30; pasos=500
rng=np.random.default_rng(2025)
pert=np.zeros((L,L))
xs=np.linspace(0,1,L); xx,yy=np.meshgrid(xs,xs)
for mx in range(1,4):
    for my in range(1,4):
        pert+= np.sin(2*np.pi*(mx*xx+my*yy)+rng.uniform(0,2*np.pi))/(mx+my)
pert/=pert.std()
phi=1.0+0.8*pert  # O(1)
ar=np.ones((L,L),bool); ad=np.ones((L,L),bool)
print("06 MF4 rho_E real = rho0/a^3 * phi^2 * (1+|grad|^2)")
for step in range(pasos):
    tg=step/pasos; a=np.exp(6*tg); T=T0/a; rho=rho0/(a**3)
    grad_x=(np.roll(phi,-1,axis=1)-np.roll(phi,1,axis=1))/2
    grad_y=(np.roll(phi,-1,axis=0)-np.roll(phi,1,axis=0))/2
    grad2=grad_x**2+grad_y**2
    rho_E = rho * (phi**2) * (1+grad2)
    left=np.roll(ar,1,axis=1); up=np.roll(ad,1,axis=0)
    cnt=ar.astype(int)+left.astype(int)+ad.astype(int)+up.astype(int)
    s=np.where(ar,np.roll(phi,-1,axis=1),0)+np.where(left,np.roll(phi,1,axis=1),0)+np.where(ad,np.roll(phi,-1,axis=0),0)+np.where(up,np.roll(phi,1,axis=0),0)
    mean=np.divide(s,cnt,out=np.zeros_like(phi),where=cnt>0)
    phi_new=phi.copy(); phi_new[cnt>0]=phi[cnt>0]+0.3*(mean[cnt>0]-phi[cnt>0]); phi=phi_new
    H_fis=H_topo*np.sqrt(T/T0); tot=int(np.sum(ar)+np.sum(ad)); nc=int(round(H_fis*tot))
    if nc>0 and tot>0:
        idx=np.argwhere(ar)
        if len(idx)>0:
            nr=int(round(nc*np.sum(ar)/tot)); sel=rng.choice(len(idx), size=min(nr,len(idx)), replace=False)
            for i in sel: ar[tuple(idx[i])]=False
        rem=nc-(nr if 'nr' in locals() else 0)
        if rem>0:
            idx=np.argwhere(ad)
            if len(idx)>0:
                sel=rng.choice(len(idx), size=min(rem,len(idx)), replace=False)
                for i in sel: ad[tuple(idx[i])]=False
    if step%100==0:
        media=phi.mean(); visto=np.zeros((L,L),bool); clusters=[]
        for y in range(L):
            for x in range(L):
                if visto[y,x]: continue
                q=deque([(y,x)]); visto[y,x]=True; nodes=[(y,x)]; lado=phi[y,x]>=media; perim=0; sum_rho=rho_E[y,x]; var_g=grad2[y,x]
                while q:
                    cy,cx=q.popleft()
                    if not ar[cy,cx] or (phi[cy,(cx+1)%L]>=media)!=lado: perim+=1
                    if not ar[cy,(cx-1)%L] or (phi[cy,(cx-1)%L]>=media)!=lado: perim+=1
                    if not ad[cy,cx] or (phi[(cy+1)%L,cx]>=media)!=lado: perim+=1
                    if not ad[(cy-1)%L,cx] or (phi[(cy-1)%L,cx]>=media)!=lado: perim+=1
                    for ny,nx in [(cy,(cx+1)%L),(cy,(cx-1)%L),((cy+1)%L,cx),((cy-1)%L,cx)]:
                        if not visto[ny,nx]:
                            cond = (ar[cy,cx] if nx!=cx else ad[cy,cx] if ny>cy else ad[(cy-1)%L,cx] if ny<cy else ar[cy,(cx-1)%L]) if False else True
                            # simplificado
                            if (phi[ny,nx]>=media)==lado and (ar[cy,cx] if nx!=cx else ad[cy,cx] if ny== (cy+1)%L else ad[(cy-1)%L,cx] or True):
                                if ( (nx==(cx+1)%L and ar[cy,cx]) or (nx==(cx-1)%L and ar[cy,(cx-1)%L]) or (ny==(cy+1)%L and ad[cy,cx]) or (ny==(cy-1)%L and ad[(cy-1)%L,cx]) ):
                                    visto[ny,nx]=True; q.append((ny,nx)); nodes.append((ny,nx)); sum_rho+=rho_E[ny,nx]; var_g+=grad2[ny,nx]
                k=len(nodes); clusters.append((k,perim,sum_rho,var_g/k if k>0 else 0))
        T_norm=T/T0+1e-6
        def m_unica(c): return c[2]*np.exp(-c[3]/(T_norm+1e-6))
        k1=[m_unica(c) for c in clusters if c[0]==1]; k3=[m_unica(c) for c in clusters if c[0]==3 and c[1]==8]
        mk1=np.mean(k1) if k1 else 0; mk3=np.mean(k3) if k3 else 0; ratio=mk1/(mk3+1e-30) if mk3>0 else 0
        print(f"step {step:3d} a={a:5.1f} rho_E_avg={rho_E.mean():.2e} k1={len(k1)} k3={len(k3)} ratio={ratio:.5f}")
print("FIN 06")
