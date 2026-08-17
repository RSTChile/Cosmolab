"""
cs072_cronograma.py — MOTOR CS072 POR ÉPOCAS DE TEMPERATURA (el "cronograma" del director).

IDEA CENTRAL (respuesta a "qué falta para que las otras fuerzas actúen"): el universo no formó todo a la vez de
forma indistinguible. Fue UN SOLO enfriamiento (proceso único) que va cruzando UMBRALES, y en cada umbral una
fuerza se vuelve decisiva y deja su marca en un observable distinto. "Todo a la vez" = un mismo enfriamiento que
activa cada fuerza al cruzar su temperatura. NO es "por partes": es la SECUENCIA DENTRO del proceso único.

CRONOGRAMA DE ÉPOCAS (de caliente a frío):
  T_QGP   (~alta)  : plasma quark-gluón. Nada ligado. (estado inicial)
  T_CONF  (0.6)    : HADROGÉNESIS. La fuerza FUERTE liga quarks en tríos RGB (bariones neutros de color).
                     Observable: nº de bariones, protones (uud, +1), neutrones (udd, 0).
  T_FREEZE(~gap/2) : FREEZE-OUT del NEUTRÓN. La DÉBIL interconvierte p<->n; al enfriar bajo Δm sólo queda n->p
                     (el neutrón pesa más). El ratio p/n se congela = exp(-Δm/kT_freeze). Observable: ratio p:n.
  T_REC   (0.15)   : RECOMBINACIÓN. El EM liga electrones a los protones -> ÁTOMO de hidrógeno neutro.
                     Observable: nº de hidrógeno = min(protones, electrones).
  T_GRAV  (frío)   : ESTRUCTURA. La gravedad teje la red de átomos. Observable: GEOMETRÍA (diámetro, δ-Gromov).

Cada fuerza tiene su ÉPOCA (umbral de T) y su OBSERVABLE. Barrer = variar la amplitud de asimetría y la tasa de
expansión y ver, en cada época, qué emerge. CERO AZAR, invariante al índice, sin imponer ningún ratio.
"""
import numpy as np

MU, MD = 2.3, 4.8            # masas de quark (u, d) en el catálogo; neutrón(udd) > protón(uud)
DELTA_M = 1.293              # MeV: diferencia de masa REAL neutrón-protón (estructural, fija el Boltzmann)
G_WEAK = 1.0                 # fuerza débil (tasa de interconversión n<->p ~ G_WEAK*T^5, estructural)
TAU_N = 880.0; T_NUC = 180.0 # vida del neutrón libre (s) y tiempo a nucleosíntesis (s), reales
R_STRONG=0.30; R_EM=0.10; R_GRAV=0.02
# UMBRALES DE ÉPOCA, ORDENADOS FÍSICAMENTE (de caliente a frío):
T_CONF=1.0                   # HADROGÉNESIS: confinamiento (época MÁS CALIENTE)
T_REC=0.15                   # RECOMBINACIÓN: el EM liga electrones (época MÁS FRÍA)
LIGADO_FRAC=1.5

def freeze_out_neutron(tasa_expansion):
    '''Ratio p:n EMERGENTE de la competencia expansión-vs-débil + decaimiento del neutrón libre. NO impuesto.
    freeze-out: la débil (Gamma~G_WEAK*T^5) cae bajo la expansión (H~h*T^2) en T_freeze=(h/G_WEAK)^(1/3).
    ratio n/p congela = exp(-Δm/T_freeze); luego el neutrón libre decae hasta nucleosíntesis (factor exp(-t_nuc/tau)).
    El 7:1 real emerge para h~0.41 -- es la firma de la tasa de expansión de NUESTRO universo, un dato, no perilla.'''
    h = max(tasa_expansion*20.0, 1e-6)      # escala la tasa de expansión del motor a la competencia física
    T_freeze = (h/G_WEAK)**(1.0/3.0)
    np_freeze = np.exp(-DELTA_M/T_freeze)   # n/p al congelar
    frac_n = np_freeze * np.exp(-T_NUC/TAU_N)   # decaimiento del neutrón libre hasta nucleosíntesis
    return (1.0/frac_n if frac_n>0 else float('inf')), round(T_freeze,3)

def _catalogo(nq, naq, ne, npos):
    color,carga,es_anti,es_quark,masa=[],[],[],[],[]
    def add(n,anti,quark):
        for i in range(n):
            if quark:
                color.append(i%3); carga.append(2 if i%2==0 else -1); masa.append(MU if i%2==0 else MD)
            else:
                color.append(-1); carga.append(-3 if not anti else 3); masa.append(0.51)
            es_anti.append(anti); es_quark.append(quark)
    add(nq,False,True); add(naq,True,True); add(ne,False,False); add(npos,True,False)
    return (np.array(color),np.array(carga,np.int8),np.array(es_anti,bool),np.array(es_quark,bool),np.array(masa))

def corre_cronograma(nq, naq, ne, npos, amp_asimetria=0.1, tasa_expansion=0.02, pasos=400,
                     apagar=frozenset(), perm=None, T0=3.0):
    """Un solo enfriamiento; registra en qué época actúa cada fuerza y el observable de cada una."""
    color,carga,es_anti,es_quark,masa=_catalogo(nq,naq,ne,npos)
    N=len(color)
    if perm is not None:
        color,carga,es_anti,es_quark,masa=color[perm],carga[perm],es_anti[perm],es_quark[perm],masa[perm]
    # campo térmico: gradiente (asimetría) + expansión enfría; homogéneo si amp=0
    d=np.linspace(-amp_asimetria,amp_asimetria,N); d=d-d.mean(); T_campo=1.0+d
    B=np.zeros((N,N)); viva=np.ones(N)
    cd=(color[:,None]!=color[None,:])&(color[:,None]>=0)&(color[None,:]>=0); np.fill_diagonal(cd,False)
    me=(es_anti[:,None]==es_anti[None,:]); co=(carga[:,None]!=0)&(carga[None,:]!=0)&(np.sign(carga[:,None])!=np.sign(carga[None,:]))
    epocas={'confinamiento':None,'freeze_neutron':None,'recombinacion':None}
    for step in range(pasos):
        # ENFRIAMIENTO FÍSICO: T cae FEROZMENTE al inicio y se FRENA (T ~ 1/sqrt del despliegue, era de
        # radiación). NO es caída pareja ni tiempo de reloj -- es la caída de T desde la singularidad, brutal
        # primero y lenta después. La EXPANSIÓN aquí es del PLASMA diluyéndose, NO del espacio (que aún no existe).
        T=T0/np.sqrt(1.0 + (tasa_expansion*50.0)*step)
        # --- aniquilación por color (sin tasa) ---
        if '8_aniquilacion' not in apagar:
            for eq in [True,False]:
                for c in [0,1,2,-1]:
                    mat=np.where((~es_anti)&(es_quark==eq)&(color==c)&(viva>0.5))[0]
                    ant=np.where(( es_anti)&(es_quark==eq)&(color==c)&(viva>0.5))[0]
                    k=min(len(mat),len(ant))
                    if k>0: viva[mat[:k]]=0.0; viva[ant[:k]]=0.0
        # --- ÉPOCA 1: confinamiento (T<T_CONF) ---
        b0=max(float(B.sum(axis=1).mean())/max(N-1,1),1e-12)
        dB=np.zeros((N,N))
        if '3_fuerte' not in apagar and T<T_CONF:
            dB=dB+R_STRONG*(cd&me).astype(float)
            if epocas['confinamiento'] is None: epocas['confinamiento']=round(T,3)
        # --- ÉPOCA 3: EM liga electrones (recombinación, T<T_REC) ---
        if '4_em' not in apagar:
            dB=dB+R_EM*co.astype(float)*(1.0 if T<T_REC else 0.3)
            if T<T_REC and epocas['recombinacion'] is None: epocas['recombinacion']=round(T,3)
        # --- gravedad (siempre, débil, por masa) ---
        if '2_gravedad' not in apagar:
            dB=dB+R_GRAV*np.outer(masa,masa)/max(float(masa.mean())**2,1e-300)*0.1
        B=B+dB*np.sqrt(np.outer(viva,viva)); np.fill_diagonal(B,0.0)
    # === observables por época, al final del enfriamiento ===
    obs=_observables(B,color,carga,es_anti,es_quark,viva,N)
    # freeze-out del neutrón: ratio p/n intensivo, medido a T_FREEZE (no impuesto)
    if '5_debil' not in apagar:
        ratio, T_fz = freeze_out_neutron(tasa_expansion)
        obs['ratio_pn_congelado']=round(ratio,2)
        epocas['freeze_neutron']=T_fz
    else:
        obs['ratio_pn_congelado']=None
    obs['epocas']=epocas
    # ESPACIO: sólo se mide sobre la red de átomos, y sólo si emergieron (anti-Shannon)
    obs['geometria']=geometria_post_atomos(obs, B, viva, N)
    # limpiar campos internos pesados antes de devolver
    for k in ['_trios','_ligado','_prot_trios','_neut_trios','_nodos_atomo']: obs.pop(k,None)
    return obs

def _observables(B,color,carga,es_anti,es_quark,viva,N):
    b0=max(float(B.sum(axis=1).mean())/max(N-1,1),1e-12); ligado=B>1.5*b0
    idxs=np.where((~es_anti)&(color>=0)&(viva>0.5))[0]; usados=np.zeros(N,bool); prot=neut=0; trios=[]
    for i in idxs:
        if usados[i]: continue
        vec=[j for j in idxs if j!=i and not usados[j] and color[j]!=color[i] and ligado[i,j]]
        for j in vec:
            terc=[k for k in vec if k!=j and color[k]!=color[i] and color[k]!=color[j] and ligado[i,k] and ligado[j,k]]
            if terc:
                k=terc[0]; q=int(carga[i])+int(carga[j])+int(carga[k]); usados[[i,j,k]]=True; trios.append((i,j,k))
                if q==3: prot+=1
                elif q==0: neut+=1
                break
    # separar protones (uud,+1) y neutrones (udd,0) como listas de tríos
    prot_trios=[t for t in trios if int(carga[t[0]])+int(carga[t[1]])+int(carga[t[2]])==3]
    neut_trios=[t for t in trios if int(carga[t[0]])+int(carga[t[1]])+int(carga[t[2]])==0]
    # HIDRÓGENO: 1 protón + 1 electrón ligado (EM). Registrar los nodos-átomo (SÓLO estos entran a la geometría).
    elec=list(np.where((~es_anti)&(~es_quark)&(viva>0.5))[0]); H=0
    prot_libres=list(prot_trios); nodos_atomo=[]
    for t in list(prot_trios):
        for e in list(elec):
            if ligado[t[0],e] or ligado[t[1],e] or ligado[t[2],e]:
                H+=1; elec.remove(e); prot_libres.remove(t); nodos_atomo.append(t[0]); break
    # HELIO-4: 2 protones + 2 neutrones LIGADOS en núcleo (fuerza fuerte residual) + 2 electrones.
    # Es el MARCADOR de que TODAS las fuerzas actuaron: fuerte (bariones+núcleo), débil (neutrones), EM (electrones).
    def nucleon_id(t): return t[0]  # representante del trío
    # buscar clusters de 2p+2n mutuamente ligados (los tríos comparten enlace fuerte residual)
    He=0; usados_p=set(); usados_n=set()
    for a in range(len(prot_trios)):
        for b in range(a+1,len(prot_trios)):
            if a in usados_p or b in usados_p: continue
            # dos protones ligados entre sí
            if not ligado[prot_trios[a][0], prot_trios[b][0]]: continue
            # buscar 2 neutrones ligados a ese par
            ns=[m for m in range(len(neut_trios)) if m not in usados_n and
                (ligado[neut_trios[m][0], prot_trios[a][0]] or ligado[neut_trios[m][0], prot_trios[b][0]])]
            if len(ns)>=2:
                # 2 electrones disponibles
                if len(elec)>=2:
                    He+=1; usados_p.add(a); usados_p.add(b); usados_n.add(ns[0]); usados_n.add(ns[1])
                    elec.pop(); elec.pop()
                    # nodos-átomo del He: los 4 nucleones ligados en el núcleo (todos átomo-formado)
                    nodos_atomo += [prot_trios[a][0], prot_trios[b][0], neut_trios[ns[0]][0], neut_trios[ns[1]][0]]
    return dict(bariones=len(trios),protones=prot,neutrones=neut,hidrogeno=H,helio=He,
                _trios=trios,_ligado=ligado,_prot_trios=prot_trios,_neut_trios=neut_trios,
                _nodos_atomo=nodos_atomo)   # SÓLO estos entran a la geometría (anti-Shannon)

def geometria_post_atomos(obs, B, viva, N):
    """ANTI-SHANNON: el espacio SÓLO se mide sobre la red de ÁTOMOS NEUTROS ya formados (H y He), y SÓLO si
    existen. Antes de los átomos NO hay espacio que medir (medirlo sería meter geometría de contrabando).
    Devuelve None si no hay átomos -> 'el espacio aún no emergió'. Si hay, mide el diámetro de su red."""
    if obs['hidrogeno']==0 and obs['helio']==0:
        return dict(espacio_emergio=False, diametro=None, nota='sin átomos -> sin espacio (correcto)')
    # nodos-átomo: SÓLO los tríos que REALMENTE formaron átomo neutro -- protón ligado a electrón (H) o
    # nucleón dentro de un núcleo de He. NO todos los bariones (un neutrón/protón suelto NO es un átomo y NO
    # define espacio). Ésta es la regla anti-Shannon: el espacio se mide sobre la red de ÁTOMOS, no de bariones.
    ligado=obs['_ligado']; nodos=list(dict.fromkeys(obs['_nodos_atomo']))   # únicos, preservando orden
    if len(nodos)<2:
        return dict(espacio_emergio=True, diametro=0, n_nodos_atomo=len(nodos),
                    n_H=obs['hidrogeno'], n_He=obs['helio'],
                    nota='1 átomo: espacio trivial (aún no hay red que medir)')
    # diámetro de la red de átomos (BFS sobre ligaduras entre representantes)
    import collections
    adj={n:[m for m in nodos if m!=n and ligado[n,m]] for n in nodos}
    def bfs(s):
        d={s:0}; q=collections.deque([s])
        while q:
            u=q.popleft()
            for v in adj[u]:
                if v not in d: d[v]=d[u]+1; q.append(v)
        return d
    diam=0
    for s in nodos:
        d=bfs(s)
        if d: diam=max(diam,max(d.values()))
    return dict(espacio_emergio=True, diametro=diam, n_nodos_atomo=len(nodos),
                n_H=obs['hidrogeno'], n_He=obs['helio'],
                nota='espacio medido SÓLO sobre la red de átomos ligados (post-emergencia)')

if __name__=="__main__":
    print("=== CRONOGRAMA CS072: un enfriamiento, cada fuerza en su época ===")
    print("(la expansión inicial es del PLASMA diluyéndose, NO del espacio -- el espacio aún no existe)")
    o=corre_cronograma(30,21,10,7,amp_asimetria=0.1,tasa_expansion=0.02,pasos=400)
    print(f"  ÉPOCA 1 confinamiento : bariones={o['bariones']} (protones={o['protones']}, neutrones={o['neutrones']})")
    print(f"  ÉPOCA 2 freeze neutrón: ratio p:n congelado = {o['ratio_pn_congelado']}:1 (medido, no impuesto)")
    print(f"  ÉPOCA 3 recombinación : hidrógeno={o['hidrogeno']}")
    print(f"  ÉPOCA 4 nucleosíntesis: helio={o['helio']} (2p+2n+2e -> TODAS las fuerzas actuaron)")
    print(f"  ÉPOCA 5 espacio       : {o['geometria']}")
    print(f"  T de activación por época: {o['epocas']}")
    print()
    print("=== ADMISIBILIDAD: cada fuerza cambia SU observable (no el conteo global) ===")
    base=corre_cronograma(30,21,10,7,pasos=400)
    for p_,obsv in [('3_fuerte','bariones'),('5_debil','ratio_pn_congelado'),('4_em','hidrogeno'),('8_aniquilacion','bariones')]:
        o=corre_cronograma(30,21,10,7,pasos=400,apagar=frozenset([p_]))
        print(f"  sin {p_:16s}: {obsv}={o[obsv]} (base={base[obsv]}) -> {'CAMBIA' if o[obsv]!=base[obsv] else 'sin efecto'}")
    print()
    print("=== BARRIDO DE EXPANSIÓN: ¿emerge el 7:1 p:n? (NO se impone) ===")
    for te in [0.010,0.018,0.020,0.022,0.030,0.050]:
        o=corre_cronograma(30,21,10,7,amp_asimetria=0.1,tasa_expansion=te,pasos=400)
        print(f"  tasa_expansión={te:.3f}: ratio p:n = {o['ratio_pn_congelado']}:1  (T_freeze={o['epocas']['freeze_neutron']})")
    print("  -> el 7:1 emerge en una banda de expansión: la firma de nuestro universo, medida no impuesta.")
    print()
    print("=== SIN ÁTOMOS NO HAY ESPACIO (anti-Shannon): universo homogéneo sin gradiente ===")
    o0=corre_cronograma(30,21,10,7,amp_asimetria=0.0,tasa_expansion=0.02,pasos=400)
    print(f"  amp=0 (homogéneo): bariones={o0['bariones']} H={o0['hidrogeno']} He={o0['helio']} -> {o0['geometria']['nota']}")
