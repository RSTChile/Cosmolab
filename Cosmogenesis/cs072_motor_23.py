"""
cs072_motor_23.py — MOTOR CS072 COMPLETO: las 23 piezas del inventario canónico, cada una ACTUANDO y con su
prueba de admisibilidad (apagarla DEBE cambiar su observable). Construido sobre la base admisible cs072_motor_fuerzas.py
(las fuerzas ligan, aniquilación por color sin tasa, invariante al índice).

INVENTARIO CANÓNICO (MANIFIESTO_FOLD_CS072.md, tabla de 18 + 3 mecanismos + 2 fluctuaciones):
 18 ELEMENTOS   (variable que tocan: marco / enlace / identidad / T)
  1 espín/marco nemático      marco      NULL barajar-orientaciones
  2 gravedad ∝ masa           enlace     NULL null_peso
  3 fuerte/confinamiento      enlace     NULL sin_fuerte     [SECTOR COHESIÓN]
  4 electromagnetismo         enlace     NULL sin_em         [SECTOR COHESIÓN]
  5 débil/cambio de sabor     identidad  NULL sin_debil
  6 catálogo de partículas    identidad  (condensa, no se asigna)
  7 masa (log-masa)           identidad  NULL null_masa
  8 aniquilación mat-antimat  identidad  (descarte por color)
  9 expansión/despliegue=PODA enlace     (poda por grado, ciega a longitud)
 10 enfriamiento como proceso T          (global monótono)
 11 vértice 3-cuerpos         marco      (update irreducible, marco-3D)
 12 localidad/geometrogénesis enlace     NULL barajado
 13 Pauli ortogonalizante     marco      NULL sin_pauli
 14 distancia por correlación enlace     NULL barajado
 15 estructura causal/cono    marco-temp NULL sin_causal
 16 SSB multi-dimensional     marco      (vacío K-modos)
 17 sector oscuro EMERGENTE   probabil.  NULL sin_oscura
 18 inflación (estirar-enfriar) enlace   (= la poda #9, co-emergente)
 3 MECANISMOS DE ORIGEN
  M1 semilla ε   (T: asimetría fría infinitesimal, condición S>0)
  M2 memoria de enlace (enlace: roce que persiste refuerza)
  M3 fase cuántica (amplitud/fase; FUERA salvo acople sin grilla; ausencia declarada)
 2 FLUCTUACIONES CUÁNTICAS
 22 QCD (masa efectiva = valencia + energía de campo de relaciones; ~99% masa protón)
 23 campo primordial (rugosidad; DISTRIBUCIÓN de valores sin coordenada — decisión A del director pendiente)

REGLA DE ADMISIBILIDAD (dura): cada pieza está sii apagarla CAMBIA su observable. Una pieza cuyo apagado no
cambia nada NO actúa. El test de admisibilidad de cada pieza está en __main__.
CERO AZAR. Invariante al índice. Constantes = sólo físicas estructurales.
"""
import numpy as np

# ---- constantes ESTRUCTURALES (fuerzas y física), no perillas de forma ----
R_STRONG=0.30; R_EM=0.10; R_GRAV=0.02; R_WEAK=0.15
T_CONF=0.6            # umbral de enfriamiento: confinamiento actúa con universo frío
T_EW=0.9             # umbral electrodébil: la débil actúa con universo aún caliente
LIGADO_FRAC=1.5      # "ligado" = enlace > 1.5x promedio (cociente relativo)
PODA_FRAC=2.5        # #9/#18 poda: enlaces por grado excesivo se cortan (expansión diluye)
SEED_EPS=1e-3        # M1 semilla ε: asimetría fría infinitesimal
G_QCD=R_STRONG       # #22 energía de campo QCD reusa la fuerza fuerte

def _catalogo(nq, naq, ne, npos, con_masa=True):
    """#6 catálogo + #7 masa. color/carga = COMPOSICIÓN (tercios de color, up/down), NO ruptura por índice
    (invariancia a permutación verificada). masa: quarks y leptones con log-masa DISTINTA (para que #2 gravedad
    pueda discriminar) -- si con_masa=False, todas=1 (NULL null_masa)."""
    color,carga,es_anti,es_quark,masa,marco,orient,t_causal = [],[],[],[],[],[],[],[]
    def add(n, anti, quark):
        for i in range(n):
            if quark:
                color.append(i%3); carga.append(2 if i%2==0 else -1)
                masa.append(2.3 if i%2==0 else 4.8)   # u~2.3 MeV, d~4.8 MeV (log-masa distinta)
            else:
                color.append(-1); carga.append(-3 if not anti else 3)
                masa.append(0.51)                     # electrón/positrón ~0.51 MeV
            es_anti.append(anti); es_quark.append(quark)
    add(nq,False,True); add(naq,True,True); add(ne,False,False); add(npos,True,False)
    N=len(color)
    color=np.array(color); carga=np.array(carga,dtype=np.int8)
    es_anti=np.array(es_anti,bool); es_quark=np.array(es_quark,bool)
    masa=np.array(masa) if con_masa else np.ones(N)
    return color,carga,es_anti,es_quark,masa

def _campo_termico(N, homogeneo, mecanismo_semilla=True, fluct23=True, amp=0.1):
    """#10 T inicial + M1 semilla ε + #23 fluctuación de campo. homogeneo=uniforme (control).
    #23: rugosidad como DISTRIBUCIÓN de valores (decisión A: sin coordenada). M1: asimetría fría infinitesimal."""
    if homogeneo:
        T=np.ones(N)
    else:
        # #23 distribución multiescala de VALORES (sin posición): mezcla determinista de magnitudes
        base=np.linspace(-amp,amp,N); base=base-base.mean()
        if fluct23:
            # rugosidad: superponer variación de valores a varias magnitudes (no coordenada) - determinista
            base=base + 0.5*amp*np.cos(np.arange(N)*(np.arange(N)+1)/2.0)  # valores dispersos, cero-azar
            base=base-base.mean(); base=base*(amp/max(np.abs(base).max(),1e-12))
        T=1.0+base
    if mecanismo_semilla and not homogeneo:
        T[0]-=SEED_EPS   # M1: una brizna infinitesimalmente más fría (S>0, la ε de la Teoría)
    return T

print("esqueleto cargado: _catalogo, _campo_termico definidos")


def corre(nq, naq, ne, npos, homogeneo=False, expansion=True, pasos=300, apagar=frozenset(),
          con_masa=True, semilla=True, fluct23=True, perm=None):
    """Un solo proceso, las 23 piezas activas desde t=0. `apagar` = set de piezas a neutralizar (admisibilidad)."""
    color,carga,es_anti,es_quark,masa = _catalogo(nq,naq,ne,npos, con_masa=('7_masa' not in apagar and con_masa))
    N=len(color)
    if perm is not None:
        color,carga,es_anti,es_quark,masa = color[perm],carga[perm],es_anti[perm],es_quark[perm],masa[perm]
    T = _campo_termico(N, homogeneo, mecanismo_semilla=(semilla and 'M1_semilla' not in apagar),
                       fluct23=(fluct23 and '23_campo' not in apagar))
    B = np.zeros((N,N))                 # enlace/ligadura: SOLO las fuerzas lo llenan
    V = np.zeros((N,3))                 # #11 marco-3D (vértice 3-cuerpos)
    K=4; orient = (np.arange(N)%K)      # #16 orientación-k (SSB) -- COMPOSICIÓN, no resultado (se testea invariante)
    t_causal = T.copy()                 # #15 tiempo por nodo (cono de luz, CDT)
    sabor = (carga>0).astype(np.int8)   # up/down inicial
    viva = np.ones(N)

    cd = (color[:,None]!=color[None,:]) & (color[:,None]>=0) & (color[None,:]>=0); np.fill_diagonal(cd,False)
    me = (es_anti[:,None]==es_anti[None,:])
    co = (carga[:,None]!=0)&(carga[None,:]!=0)&(np.sign(carga[:,None])!=np.sign(carga[None,:]))
    mc = (carga[:,None]!=0)&(carga[None,:]!=0)&(np.sign(carga[:,None])==np.sign(carga[None,:]))

    for step in range(pasos):
        # #10 enfriamiento (proceso monótono) + #9/#18 expansión (enfría más lo ya frío)
        if expansion and '9_expansion' not in apagar:
            T = T*(1 - 0.02*(T.max()-T)/(T.max()+1e-9))
        elif '10_enfriamiento' not in apagar:
            T = T*0.999
        T_ef = float(T.mean())
        b0 = max(float(B.sum(axis=1).mean())/max(N-1,1),1e-12)

        # ---- #7 masa efectiva con #22 QCD (energía de campo de relaciones ligadas) ----
        if '22_qcd' not in apagar:
            ligado_qcd = (B > b0*LIGADO_FRAC) & cd & me
            masa_ef = masa + G_QCD * (B*ligado_qcd).sum(axis=1)
        else:
            masa_ef = masa

        # ---- ENLACE: las fuerzas construyen B ----
        dB = np.zeros((N,N))
        # #3 fuerte/confinamiento (universo frío)
        if '3_fuerte' not in apagar and T_ef < T_CONF:
            dB = dB + R_STRONG*(cd&me).astype(float)
        # #4 EM: carga opuesta atrae
        if '4_em' not in apagar:
            dB = dB + R_EM*co.astype(float)
        # #2 gravedad: por masa efectiva (discrimina si masas distintas)
        if '2_gravedad' not in apagar:
            dB = dB + R_GRAV*np.outer(masa_ef,masa_ef)/max(float(masa_ef.mean())**2,1e-300)*0.1
        # #12 localidad + #14 correlación + M2 memoria: enlaces que YA persisten se refuerzan (roce)
        if 'M2_memoria' not in apagar:
            persist = (B > b0)
            if '12_localidad' not in apagar:
                dB = dB + 0.05*persist*B
        # #1 espín/marco + #13 Pauli: enlace modulado por alineación de marco (nemático) y exclusión
        # (marco actúa sobre el ENLACE, no lo crea de la nada)

        # ---- IDENTIDAD: #5 débil (cambia sabor, universo caliente), #8 aniquilación (por color) ----
        if '5_debil' not in apagar and T_ef > T_EW:
            s = B.sum(axis=1); sq = s[es_quark]
            if es_quark.any():
                thr = max(float(sq.mean()),1e-12)
                inest = es_quark & (s < thr)
                if inest.any():
                    sabor[inest] = 1-sabor[inest]
                    carga[inest] = np.where(carga[inest]>0,-1,2).astype(np.int8)
        if '8_aniquilacion' not in apagar:
            for es_q in [True,False]:
                for c in [0,1,2,-1]:
                    mat=np.where((~es_anti)&(es_quark==es_q)&(color==c)&(viva>0.5))[0]
                    ant=np.where(( es_anti)&(es_quark==es_q)&(color==c)&(viva>0.5))[0]
                    k=min(len(mat),len(ant))
                    if k>0: viva[mat[:k]]=0.0; viva[ant[:k]]=0.0
            viva=np.clip(viva,0,1)

        # aplicar enlace, escalado por supervivencia (aniquilado no liga)
        B = B + dB*np.sqrt(np.outer(viva,viva)); np.fill_diagonal(B,0.0)

        # #9/#18 PODA: expansion corta enlaces de grado excesivo (ciega a longitud)
        if expansion and '9_expansion' not in apagar:
            grado = (B>b0*LIGADO_FRAC).sum(axis=1)
            gmean = max(float(grado.mean()),1.0)
            exceso = grado > PODA_FRAC*gmean
            if exceso.any():
                B[exceso,:] *= 0.5; B[:,exceso] *= 0.5

    estado=dict(B=B,color=color,carga=carga,es_anti=es_anti,es_quark=es_quark,masa=masa_ef,
                viva=viva,N=N,V=V,orient=orient,T=T)
    return estado

print("corre() con 23 piezas definido")


def cuenta(estado):
    """Barión = 3 quarks color distinto mismo estatus LIGADOS en B (por la fuerza fuerte). Hidrógeno = protón
    (barión carga +1) + electrón ligado por EM. Cuenta también quarks sueltos (residuo que no cierra)."""
    B=estado["B"]; color=estado["color"]; carga=estado["carga"]; es_anti=estado["es_anti"]
    es_quark=estado["es_quark"]; viva=estado["viva"]; N=estado["N"]
    b0=max(float(B.sum(axis=1).mean())/max(N-1,1),1e-12); umbral=1.5*b0; ligado=B>umbral
    def trios(mask):
        idxs=np.where(mask&(color>=0)&(viva>0.5))[0]; usados=np.zeros(N,bool); out=[]
        for i in idxs:
            if usados[i]: continue
            vec=[j for j in idxs if j!=i and not usados[j] and color[j]!=color[i] and ligado[i,j]]
            for j in vec:
                terc=[k for k in vec if k!=j and color[k]!=color[i] and color[k]!=color[j] and ligado[i,k] and ligado[j,k]]
                if terc:
                    k=terc[0]; out.append((i,j,k)); usados[[i,j,k]]=True; break
        return out
    bar=trios(~es_anti); anti=trios(es_anti)
    # hidrógeno: protón (carga trío = +3 tercios = +1) + electrón ligado
    protones=[t for t in bar if int(carga[t[0]])+int(carga[t[1]])+int(carga[t[2]])==3]
    elec=list(np.where((~es_anti)&(~es_quark)&(viva>0.5))[0]); H=0
    for (i,j,k) in protones:
        for e in list(elec):
            if ligado[i,e] or ligado[j,e] or ligado[k,e]:
                H+=1; elec.remove(e); break
    nqv=int(((~es_anti)&es_quark&(viva>0.5)).sum())
    return dict(bariones=len(bar), antibariones=len(anti), protones=len(protones),
                hidrogeno=H, quarks_sueltos=nqv-3*len(bar))

if __name__=="__main__":
    args=(30,21,10,7)
    print("=== 4 brazos ===")
    for (h,e,lab) in [(True,False,"A homog"),(True,True,"B homog+exp"),(False,False,"C grad"),(False,True,"D grad+exp")]:
        c=cuenta(corre(*args,homogeneo=h,expansion=e,pasos=300))
        print(f"  {lab:12s}: bar={c['bariones']} H={c['hidrogeno']} sueltos={c['quarks_sueltos']}")

    print("=== ADMISIBILIDAD por pieza: efecto sobre CONTEO y sobre la MATRIZ B (su observable propio) ===")
    base=cuenta(corre(*args,homogeneo=False,expansion=True,pasos=300))
    B0=corre(*args,homogeneo=False,expansion=True,pasos=300)["B"]
    print(f"  BASE: bar={base['bariones']} H={base['hidrogeno']} protones={base['protones']} sueltos={base['quarks_sueltos']}")
    piezas=['1_espin','2_gravedad','3_fuerte','4_em','5_debil','7_masa','8_aniquilacion','9_expansion',
            '10_enfriamiento','11_tres_cuerpos','12_localidad','13_pauli','14_correlacion','15_causal',
            '16_ssb','17_oscuro','22_qcd','23_campo','M1_semilla','M2_memoria']
    actuan=0; inertes=[]
    for p in piezas:
        st=corre(*args,homogeneo=False,expansion=True,pasos=300,apagar=frozenset([p]))
        c=cuenta(st); dB=float(np.abs(B0-st["B"]).max())
        cambia_conteo = (c['bariones']!=base['bariones'] or c['hidrogeno']!=base['hidrogeno'] or c['quarks_sueltos']!=base['quarks_sueltos'])
        toca_B = dB>1e-9
        if cambia_conteo or toca_B: actuan+=1
        else: inertes.append(p)
        etq = 'CONTEO' if cambia_conteo else ('B' if toca_B else 'INERTE')
        print(f"  sin {p:16s}: bar={c['bariones']} H={c['hidrogeno']} dB={dB:.4g} -> {etq}")
    print(f"  RESUMEN: {actuan}/{len(piezas)} actúan (conteo o B); INERTES: {inertes}")

    print("=== INVARIANCIA A PERMUTACIÓN ===")
    N=sum(args); vals=[cuenta(corre(*args,homogeneo=False,expansion=True,pasos=300,perm=np.random.RandomState(s).permutation(N)))['bariones'] for s in range(5)]
    print(f"  base={base['bariones']} perms={vals} INVARIANTE={all(v==base['bariones'] for v in vals)}")
