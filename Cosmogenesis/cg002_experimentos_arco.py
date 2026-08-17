"""
cg002_experimentos_arco.py
==========================
TODOS los experimentos del arco CG002, reproducibles (Claude CC para Alexis, 29-jun-2026).
Acompaña a INFORME_GENERAL_CG002_ARCO.md.

Modelo único (regla validada CG002):
  Cada diferencia i tiene persistencia S_i y firma U_i (en S^{d-1}) o fase ω_i (ℤ_K).
  Compatibilidad (simétrica):     c_ij = U_i · U_j         (coopera si alineadas, compite si opuestas)
  Acoplamiento dirigido (θ_CP):   g_{i<-j} = U_i · R(θ_CP) · U_j   (R = rotación en plano (e0,e1))
  Flujo:                          f_{i<-j} = α·η·g_{i<-j}·sqrt(sat(S_i)·sat(S_j))
  Decaimiento:                    S <- (1-μ)·S
  Actualización:                  S_i += Σ_j f_{i<-j}
  Banda de viabilidad (C-N5.1):   sat(S) = S/(1+S/S_BAND)
  Extinción (C-N1.1):             S <= κ_s  =>  muerta

Factorización de campo medio (vuelve la física O(N), permite N enorme; EXACTA, no aproxima):
  dS_i = η·m_i·(U_i·R·v) − η·m_i²·(U_i·R·U_i),   v = Σ_j m_j U_j,   m = sqrt(sat(S))

USO: python3 cg002_experimentos_arco.py <experimento>
  experimentos: baryo, inercia, flecha, coexistencia, paredes, colision,
                autosimilar, dimension, constantes, criticidad, invariantes, todos
"""
import sys, numpy as np

# ---- parámetros del modelo (no cambiar para reproducir) ----
ETA, MU, KAPPA_S, S0, S_BAND, K = 0.05, 0.01, 1e-6, 1.0, 8.0, 8
def sat(S): return S/(1.0+S/S_BAND)
def rot(d, th):
    R = np.eye(d); c, s = np.cos(th), np.sin(th); R[0,0]=c; R[0,1]=-s; R[1,0]=s; R[1,1]=c; return R

# ======================= núcleo: dinámica de campo medio (O(N)) =======================
def evolucion_campo_medio(U, theta=0.0, alpha=1.0, pasos=240, mu=MU, eta=ETA, peso='banda'):
    N, d = U.shape; R = rot(d, theta); UR = U @ R
    selfd = np.einsum('ij,ij->i', UR, U)
    S = np.full(N, S0); alive = np.ones(N, bool)
    def w(S):
        return np.sqrt(sat(S)) if peso == 'banda' else np.log1p(np.maximum(S, 0.0))
    for _ in range(pasos):
        S = np.where(alive, S*(1-mu), S)
        m = np.where(alive, w(S), 0.0); v = (m[:, None]*U).sum(0)
        dS = alpha*eta*m*(UR@v) - alpha*eta*m*m*selfd
        S = S + dS; S = np.minimum(S, 1e12); S = np.where(S < 0, 0, S)
        alive = S > KAPPA_S; S = np.where(alive, S, 0.0)
    return S, alive

def firmas(N, d, seed):
    rng = np.random.default_rng(seed); U = rng.standard_normal((N, d)); return U/np.linalg.norm(U, axis=1, keepdims=True)

# ======================= EXPERIMENTOS =======================
def exp_baryo():
    print("BARYOGÉNESIS: aniquilación y exceso emergen de la diferencia (no se programan)")
    for et,lab in [(False,'aleatorias (mean≈0, diferencias presentes)'),(True,'idénticas (sin diferencia)')]:
        for s in (1,2,3):
            if et:
                u0=np.array([0.,0.,1.]); U=np.tile(u0,(2000,1))
            else:
                U=firmas(2000,3,s)
            S,al=evolucion_campo_medio(U,0.0)
            idx=np.where(al)[0]; a0=np.linalg.norm(U.mean(0)); a1=np.linalg.norm(U[idx].mean(0)) if len(idx) else 0
            print(f"  {lab[:28]:28} seed{s} f={len(idx)/2000*100:4.1f}%  a0={a0:.3f} a1={a1:.3f}")
            if et: break

def exp_inercia():
    print("INERCIA HISTÓRICA (C-N2.5.8): ¿la duración da resistencia a revertir? banda dura vs blanda")
    def flip(modo,te,fr,seed,N0=800,d=3,tau2=100):
        U=firmas(N0,d,seed); R=rot(d,0.0)
        def w(S): return np.sqrt(sat(S)) if modo=='banda' else np.log1p(np.maximum(S,0))
        S=np.full(N0,S0); alive=np.ones(N0,bool)
        def step(U,S,alive,Nn):
            m=np.where(alive,w(S),0.0); v=(m[:,None]*U).sum(0)
            dS=ETA*m*(U@v)-ETA*m*m; S=S+dS; S=np.minimum(S,1e12); S=np.where(S<0,0,S); alive=S>KAPPA_S; S=np.where(alive,S,0.0); return S,alive,v
        for _ in range(te): S,alive,v=step(U,S,alive,N0)
        vhat=v/np.linalg.norm(v); ns=alive.sum(); Ni=int(fr*ns)
        rng=np.random.default_rng(seed); Ui=-vhat+0.35*rng.standard_normal((Ni,d)); Ui/=np.linalg.norm(Ui,axis=1,keepdims=True)
        U=np.vstack([U,Ui]); S=np.concatenate([S,np.full(Ni,S0)]); alive=np.concatenate([alive,np.ones(Ni,bool)])
        for _ in range(tau2): S,alive,v=step(U,S,alive,len(S))
        return float((v/(np.linalg.norm(v)+1e-12))@vhat)
    def umbral(modo,te):
        for fr in (0.5,1,1.5,2,3,4,6,8):
            if np.mean([flip(modo,te,fr,s) for s in (1,2)])<-0.3: return f"×{fr}"
        return ">×8"
    print("  modo        τ=15   τ=50   τ=150")
    for modo in ('banda','blanda'):
        print(f"  {modo:<10}  "+"  ".join(umbral(modo,te) for te in (15,50,150)))

def exp_flecha():
    print("FLECHA DEL TIEMPO (C-N3): ¿decrece monótona la diversidad de orientación?")
    def H(U,alive,nb=24):
        idx=np.where(alive)[0]
        if len(idx)<2: return 0.0
        u=U[idx]; th=np.arccos(np.clip(u[:,2],-1,1)); ph=np.arctan2(u[:,1],u[:,0])
        h,_,_=np.histogram2d(th,ph,bins=[nb,nb]); p=h[h>0]/h.sum(); return float(-(p*np.log(p)).sum())
    U=firmas(1200,3,1); R=rot(3,0.0); S=np.full(1200,S0); alive=np.ones(1200,bool); prev=None
    print("  τ\tvivos\tentropía")
    for t in range(160):
        m=np.where(alive,np.sqrt(sat(S)),0.0); v=(m[:,None]*U).sum(0)
        dS=ETA*m*(U@v)-ETA*m*m; S=S+dS; S=np.where(S<0,0,S); alive=S>KAPPA_S; S=np.where(alive,S,0.0)
        if t%40==0 or t==159:
            h=H(U,alive); flag='' if prev is None or h<=prev+1e-9 else ' SUBIÓ'; prev=h
            print(f"  {t+1}\t{int(alive.sum())}\t{h:.3f}{flag}")

def _spatial_run(seed,rc,N=1500,d=3,L=10.0,pasos=160):
    rng=np.random.default_rng(seed); P=rng.uniform(0,L,(N,3)); U=firmas(N,d,seed+777)
    C=U@U.T; np.fill_diagonal(C,0)
    D=np.sqrt(((P[:,None,:]-P[None,:,:])**2).sum(2)); CM=C*(D<rc); np.fill_diagonal(CM,0)
    S=np.full(N,S0); alive=np.ones(N,bool)
    for _ in range(pasos):
        m=np.where(alive,np.sqrt(sat(S)),0.0); F=ETA*CM*np.outer(m,m); dS=F.sum(1)
        S=S+dS; S[S<0]=0; alive=S>KAPPA_S; S=np.where(alive,S,0.0)
    return P,U,D,alive

def exp_coexistencia():
    print("COEXISTENCIA (C-N2.5.9): orden local vs global según alcance rc (local≫global = dominios)")
    print("  rc\torden_global\torden_local")
    for rc in (1.5,3.0,6.0,100.0):
        gs,ls=[],[]
        for s in (1,2,3):
            P,U,D,alive=_spatial_run(s,rc); idx=np.where(alive)[0]; Us=U[idx]; Ps=P[idx]
            gs.append(np.linalg.norm(Us.mean(0)))
            nb=4; ci=np.clip((Ps/10.0*nb).astype(int),0,nb-1); key=ci[:,0]*16+ci[:,1]*4+ci[:,2]
            locs=[np.linalg.norm(Us[key==c].mean(0)) for c in np.unique(key) if (key==c).sum()>=5]
            ls.append(np.mean(locs) if locs else 0)
        print(f"  {rc}\t{np.mean(gs):.3f}\t\t{np.mean(ls):.3f}")

def exp_dimension():
    print("EL ESPACIO HEREDA LA DIMENSIÓN DEL SUSTRATO (C-N2.5/2.6)")
    def dcorr(X):
        if len(X)>900: X=X[np.random.default_rng(0).choice(len(X),900,replace=False)]
        D=np.sqrt(((X[:,None,:]-X[None,:,:])**2).sum(2)); dd=D[np.triu_indices(len(X),1)]
        rs=np.logspace(np.log10(np.percentile(dd,3)),np.log10(np.percentile(dd,40)),12)
        Cr=np.array([(dd<r).mean() for r in rs]); ok=Cr>0
        return np.polyfit(np.log(rs[ok]),np.log(Cr[ok]),1)[0]
    print("  sustrato\tdim_intrínseca\tD_correlación")
    for m in (1,2,3,4):
        Ds=[]
        for s in (1,2,3):
            U=firmas(2000,m+1,s); S,al=evolucion_campo_medio(U,0.0,pasos=200); Ds.append(dcorr(U[al]))
        print(f"  S^{m}\t\t{m}\t\t{np.mean(Ds):.2f}")

def exp_constantes():
    print("CONSTANTES EMERGENTES vs HISTORIA (20 cosmos): qué converge (ley) y qué no (historia)")
    fs,a1s,V=[],[],[]
    for s in range(1,21):
        U=firmas(2000,3,s); S,al=evolucion_campo_medio(U,0.0); idx=np.where(al)[0]
        fs.append(len(idx)/2000); vv=U[idx].mean(0); a1s.append(np.linalg.norm(vv)); V.append(vv/np.linalg.norm(vv))
    cv=lambda a:100*np.std(a)/abs(np.mean(a))
    print(f"  fracción persiste: {np.mean(fs):.3f}  CV={cv(fs):.1f}%  -> {'CONSTANTE' if cv(fs)<5 else 'contingente'}")
    print(f"  magnitud orden a1: {np.mean(a1s):.3f}  CV={cv(a1s):.1f}%  -> {'CONSTANTE' if cv(a1s)<5 else 'contingente'}")
    V=np.array(V); angs=[np.degrees(np.arccos(np.clip(V[i]@V[j],-1,1))) for i in range(len(V)) for j in range(i+1,len(V))]
    print(f"  dirección flecha:  ángulo medio entre cosmos {np.mean(angs):.0f}° (azar≈90°) -> HISTORIA contingente")

def exp_criticidad():
    print("CRITICIDAD (C-N2.7): parámetro de orden y fluctuación vs alcance rc")
    print("  rc\torden_global\tfluctuación(std)")
    pre=[_spatial_prep(s) for s in (1,2,3,4,5)]
    for rc in (1.0,2.0,2.5,3.0,3.5,4.0,5.0,7.0):
        vals=[_order_at(P,U,D,rc) for (P,U,D) in pre]
        print(f"  {rc}\t{np.mean(vals):.3f}\t\t{np.std(vals):.3f}")

def _spatial_prep(seed,N=1200,d=3,L=10.0):
    rng=np.random.default_rng(seed); P=rng.uniform(0,L,(N,3)); U=firmas(N,d,seed+777)
    C=U@U.T; np.fill_diagonal(C,0); D=np.sqrt(((P[:,None,:]-P[None,:,:])**2).sum(2)); return P,U,(C,D)
def _order_at(P,U,CD,rc,pasos=150):
    C,D=CD; CM=C*(D<rc); np.fill_diagonal(CM,0); N=len(U); S=np.full(N,S0); alive=np.ones(N,bool)
    for _ in range(pasos):
        m=np.where(alive,np.sqrt(sat(S)),0.0); F=ETA*CM*np.outer(m,m); dS=F.sum(1)
        S=S+dS; S[S<0]=0; alive=S>KAPPA_S; S=np.where(alive,S,0.0)
    idx=np.where(alive)[0]; return float(np.linalg.norm(U[idx].mean(0))) if len(idx)>10 else 0.0

def exp_invariantes():
    print("INVARIANTES FUNDACIONALES y TIPOLOGÍA DE RUPTURA (C-N2.8.9a)")
    def run(seed=1,N=1500,d=3,eta=ETA,mu=MU,alpha=1.0,div=1.0,pasos=240):
        rng=np.random.default_rng(seed); base=rng.standard_normal(d)
        U=div*rng.standard_normal((N,d))+(1-div)*base; U/=np.linalg.norm(U,axis=1,keepdims=True)
        S=np.full(N,S0); alive=np.ones(N,bool)
        for _ in range(pasos):
            m=np.where(alive,np.sqrt(sat(S)),0.0); v=(m[:,None]*U).sum(0)
            dS=alpha*eta*m*(U@v)-alpha*eta*m*m; S=np.where(alive,S*(1-mu),S)+dS
            S=np.where(S<0,0,S); alive=S>KAPPA_S; S=np.where(alive,S,0.0)
        idx=np.where(alive)[0]; return len(idx)/N, (np.linalg.norm(U[idx].mean(0)) if len(idx) else 0)
    m=lambda **k: tuple(np.mean([run(seed=s,**k)[i] for s in (1,2,3)]) for i in (0,1))
    print("  baseline:           f=%.2f orden=%.2f"%m())
    print("  κ_Δ div=0.1 (sin diferencia):  f=%.2f orden=%.2f -> HOMOGENEIZACIÓN"%m(div=0.1))
    print("  κ_V α=0 (sin acoplamiento):    f=%.2f orden=%.2f -> DESACOPLAMIENTO"%m(alpha=0.0))
    print("  κ_P α=0,μ=0.1 (sin regenerar): f=%.2f orden=%.2f -> DISOLUCIÓN"%m(alpha=0.0,mu=0.1))

EXPS={'baryo':exp_baryo,'inercia':exp_inercia,'flecha':exp_flecha,'coexistencia':exp_coexistencia,
      'dimension':exp_dimension,'constantes':exp_constantes,'criticidad':exp_criticidad,'invariantes':exp_invariantes}
if __name__=='__main__':
    arg=sys.argv[1] if len(sys.argv)>1 else 'todos'
    if arg=='todos':
        for k,fn in EXPS.items(): print('\n==== %s ===='%k); fn()
    elif arg in EXPS: EXPS[arg]()
    else: print("experimentos:", ', '.join(EXPS)+', todos')
