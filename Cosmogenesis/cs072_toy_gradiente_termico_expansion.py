# CS072 TOY — ¿la asimetría térmica + expansión rompe los empates por FÍSICA, no por índice?
# CS, 18-jul. Toy de verificación (NO el fold). Prueba la cadena del director:
#   temperatura inicial desigual -> expansión amplifica -> historias W divergen -> empates rotos sin índice.
# Cinco verificaciones de Codex incluidas. CERO azar, CERO parámetro que dibuje la forma.
import numpy as np, json

def evoluciona(T0, expansion=True, pasos=40):
    N=len(T0); T=T0.copy(); W=np.zeros((N,N))
    for t in range(pasos):
        dT=np.abs(T[:,None]-T[None,:]); aff=np.exp(-dT/(T.mean()+1e-9)); np.fill_diagonal(aff,0)
        W=0.9*W+0.1*aff                                   # memoria: acumula historia térmica
        if expansion: T=T*(1-0.02*(T.max()-T)/(T.max()+1e-9))  # expansión: tasa GLOBAL, amplifica contraste
    return W
def firmas_raw(W): return np.array([np.sort(r) for r in W])   # firma = fila ordenada (invariante a etiqueta)
def n_firmas(W, tol=5): return len(np.unique(np.round(firmas_raw(W),tol),axis=0))

if __name__=="__main__":
    N=8; Tmed=1.0; d=np.linspace(-0.1,0.1,N); d-=d.mean()   # gradiente suma-cero: misma media y energía total
    T_hom=np.full(N,Tmed); T_grad=Tmed+d
    # V1 cuatro brazos
    b=[n_firmas(evoluciona(T_hom,False)), n_firmas(evoluciona(T_hom,True)),
       n_firmas(evoluciona(T_grad,False)), n_firmas(evoluciona(T_grad,True))]
    # V4 invariancia dura
    perm=np.array([5,2,7,0,3,1,6,4]); inv=np.argsort(perm)
    Wge=evoluciona(T_grad,True); Wge_p=evoluciona(T_grad[perm],True)
    dura=np.allclose(Wge, Wge_p[np.ix_(inv,inv)], atol=1e-9)
    res=dict(patron=b, invariancia_dura=bool(dura),
             tol_sweep={f'1e-{t}':n_firmas(Wge,t) for t in [2,3,5,8,10]})
    print(json.dumps(res,indent=2))
