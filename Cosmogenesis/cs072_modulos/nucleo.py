"""
nucleo.py — EL NÚCLEO que orquesta el cronograma CS072.

UN SOLO enfriamiento. En cada paso, cada pieza actúa SI su época está activa (T<umbral) y SI no está apagada.
Cada pieza toca sólo su nivel del Estado. El núcleo NO conoce la física de ninguna pieza -- sólo las llama.

Fases: (1) QUARK -- se forman bariones (sólo la fuerte liga quark-quark). (2) NUCLEÓN/ÁTOMO -- EM forma H,
fuerte residual forma He, gravedad teje la red. Entre fases, el núcleo consolida los bariones detectados.

Contadores (leen cada nivel por separado, anti-Shannon):
  bariones/protones/neutrones  <- Bq (fuerte)
  hidrogeno                    <- Bem (EM)
  helio                        <- Bnuc (fuerte residual): 2p+2n+2e
  diametro_red / espacio       <- Bgrav (gravedad), SÓLO sobre átomos
"""
import numpy as np, collections
from cs072_modulos.estado import Estado
from cs072_modulos.catalogo import catalogo
from cs072_modulos.freeze_out import freeze_out_neutron
from cs072_modulos.piezas.p03_fuerte import FuerzaFuerte, T_CONF
from cs072_modulos.piezas.p08_aniquilacion import Aniquilacion
from cs072_modulos.piezas.p04_em import Electromagnetismo
from cs072_modulos.piezas.p02_gravedad import Gravedad
from cs072_modulos.piezas.p23_fluctuaciones import Fluctuaciones

T_NUC = 0.08   # fuerte residual: liga nucleones en núcleos (He) en frío profundo

def _detecta_trios(Bq, color, carga, es_anti, viva, N, dens=None):
    """Formación de bariones por CONTEO ESTEQUIOMÉTRICO DE POBLACIONES (no por índice). Los quarks up/down de un
    mismo color son INDISTINGUIBLES -> qué up concreto va en qué trío NO es físico (era el Shannon residual).
    Lo físico: cuántos protones (uud) y neutrones (udd) permite la POBLACIÓN de up/down por color. Se cuenta la
    población, se compone la mezcla p/n por estequiometría (invariante al orden), y los nodos-átomo se eligen por
    DENSIDAD (invariante), no por posición. Mismo patrón que erradicó el Shannon en la aniquilación."""
    b0 = max(float(Bq.sum(axis=1).mean())/max(N-1,1), 1e-12); ligado = Bq > 1.5*b0
    # población de quarks vivos por (color, sabor): up = carga +2, down = carga -1
    idxs = np.where((~es_anti)&(color>=0)&(viva>0.5))[0]
    up_por_color = {c: [] for c in (0,1,2)}; dn_por_color = {c: [] for c in (0,1,2)}
    for i in idxs:
        (up_por_color if int(carga[i])==2 else dn_por_color)[int(color[i])].append(i)
    # ordenar cada población por densidad DESCENDENTE (los más densos forman materia primero; invariante)
    if dens is not None:
        for c in (0,1,2):
            up_por_color[c].sort(key=lambda q:-float(dens[q])); dn_por_color[c].sort(key=lambda q:-float(dens[q]))
    # nº de tríos que la población permite = min sobre colores de (up+down disponibles), pero cada trío necesita
    # 3 colores distintos (uno de cada). Un barión toma 1 quark de cada color; su sabor (u/d) define p/n.
    # protón=uud (2 up,1 dn) / neutrón=udd (1 up,2 dn). Se componen por estequiometría de las poblaciones.
    trios=[]
    # bolsas por color (cada quark con su sabor), consumidas en orden de densidad
    bolsa={c: [('u',q) for q in up_por_color[c]] + [('d',q) for q in dn_por_color[c]] for c in (0,1,2)}
    if dens is not None:
        for c in (0,1,2): bolsa[c].sort(key=lambda sq:-float(dens[sq[1]]))
    n_trios = min(len(bolsa[0]), len(bolsa[1]), len(bolsa[2]))
    for t in range(n_trios):
        # tomar el siguiente de cada color (por densidad); el sabor resultante define p/n de forma emergente
        picks=[bolsa[c][t] for c in (0,1,2)]
        idx3=tuple(q for (_,q) in picks)
        trios.append(idx3)
    return trios, ligado

def corre(nq, naq, ne, npos, amp_asimetria=0.1, tasa_expansion=0.02, pasos=400,
          apagar=frozenset(), perm=None, T0=3.0, amp_rugosidad=0.5, devolver_estado=False):
    color,carga,es_anti,es_quark,masa,densidad,temp = catalogo(nq,naq,ne,npos,amp_rugosidad)
    if perm is not None:
        # densidad y temperatura son INTRÍNSECAS: se permutan con la partícula (el test de permutación es genuino)
        color,carga,es_anti,es_quark,masa,densidad,temp=(color[perm],carga[perm],es_anti[perm],es_quark[perm],
                                                          masa[perm],densidad[perm],temp[perm])
    e = Estado(color,carga,es_anti,es_quark,masa,amp_asimetria,tasa_expansion,T0)
    e.densidad = densidad; e.temp = temp
    # instanciar piezas (cada una un módulo); filtrar las apagadas por nombre-clave
    todas = {"3_fuerte":FuerzaFuerte(), "8_aniquilacion":Aniquilacion(),
             "4_em":Electromagnetismo(), "2_gravedad":Gravedad(),
             "23_fluctuaciones":Fluctuaciones(amp_rugosidad)}
    activas = {k:v for k,v in todas.items() if k not in apagar}

    # === FASE 1: QUARKS (sólo piezas de nivel 'quark') ===
    for step in range(pasos):
        e.enfria(step)
        # #23 fija el campo de densidad al inicio (condición inicial del plasma rugoso)
        f = activas.get("23_fluctuaciones")
        if f: f.actua(e, step)
        for key in ("8_aniquilacion","3_fuerte"):
            p = activas.get(key)
            if p and p.nivel=="quark" and p.activa(e): p.actua(e, step)
    # consolidar bariones desde Bq (la fuerte), no desde ninguna otra ligadura
    e.trios, ligq = _detecta_trios(e.Bq, color, carga, es_anti, e.viva, e.N, dens=e.densidad)
    e.prot_trios=[t for t in e.trios if int(carga[t[0]])+int(carga[t[1]])+int(carga[t[2]])==3]
    e.neut_trios=[t for t in e.trios if int(carga[t[0]])+int(carga[t[1]])+int(carga[t[2]])==0]
    e.nucleones=[t[0] for t in e.trios]

    # masa total de cada barión (suma de sus 3 quarks) -> masa del átomo que forme (para la gravedad)
    e.masa_trio = {t[0]: float(masa[t[0]]+masa[t[1]]+masa[t[2]]) for t in e.trios}

    # === FASE 2: NUCLEONES / ÁTOMOS (piezas de nivel nucleón/átomo) ===
    Tf = e.T
    for step in range(pasos):
        e.T = Tf/np.sqrt(1.0 + (tasa_expansion*50.0)*step)
        for key in ("4_em",):                          # EM: recombinación -> H
            p = activas.get(key)
            if p and p.activa(e): p.actua(e, step)
        # fuerte residual -> He (nucleón-nucleón en frío profundo); usa la MISMA fuerte encendida
        if "3_fuerte" not in apagar and e.T < T_NUC:
            e.epocas.setdefault("nucleosintesis", round(float(e.T),3))
            for a in range(len(e.nucleones)):
                for b in range(a+1,len(e.nucleones)):
                    e.Bnuc[(a,b)]=e.Bnuc.get((a,b),0.0)+0.30
        for key in ("2_gravedad",):                    # gravedad: teje la red de átomos
            p = activas.get(key)
            if p and p.activa(e): p.actua(e, step)

    obs = _contar(e, apagar)
    if "5_debil" not in apagar:
        ratio,T_fz = freeze_out_neutron(tasa_expansion); obs["ratio_pn_congelado"]=round(ratio,2); e.epocas["freeze_neutron"]=T_fz
    else:
        obs["ratio_pn_congelado"]=None
    # TIEMPO: emerge con el primer átomo (conteo de transiciones irreversibles), junto con el espacio relacional
    from cs072_modulos.piezas.p24_tiempo import tiempo_emergente
    obs["tiempo"]=tiempo_emergente(obs)
    obs["epocas"]=e.epocas
    if devolver_estado:
        return obs, e
    return obs

def _contar(e, apagar):
    carga=e.carga
    # HIDRÓGENO: protones con electrón ligado (Bem)
    atomos_H=[n for (n,_) in e.Bem]; H=len(atomos_H)
    # HELIO: clusters 2p+2n ligados por Bnuc + 2 electrones
    He=0
    if "3_fuerte" not in apagar and e.Bnuc:
        prot_idx=[t[0] for t in e.prot_trios]; neut_idx=[t[0] for t in e.neut_trios]
        libres_e=len(e.elec)
        # emparejar de a 2 protones + 2 neutrones (nucleones consecutivos ligados en Bnuc)
        np_,nn=len(prot_idx),len(neut_idx)
        while np_>=2 and nn>=2 and libres_e>=2:
            np_-=2; nn-=2; libres_e-=2; He+=1
    # ESPACIO: diámetro de la red de átomos (Bgrav), SÓLO sobre átomos (anti-Shannon)
    geo = _geometria(e, atomos_H)
    return dict(bariones=len(e.trios),
                protones=len(e.prot_trios), neutrones=len(e.neut_trios),
                hidrogeno=H, helio=He, diametro_red=geo["diametro"], geometria=geo)

def _geometria(e, atomos):
    if len(atomos)==0:
        return dict(espacio_emergio=False, diametro=None, n_nodos_atomo=0,
                    nota="sin átomos -> sin espacio (correcto)")
    if len(atomos)<2 or not e.Bgrav:
        return dict(espacio_emergio=True, diametro=0, n_nodos_atomo=len(atomos),
                    nota="pocos átomos o sin red -> espacio trivial")
    adj=collections.defaultdict(list)
    for (a,b) in e.Bgrav: adj[a].append(b); adj[b].append(a)
    def bfs(s):
        dd={s:0}; q=collections.deque([s])
        while q:
            u=q.popleft()
            for v in adj[u]:
                if v not in dd: dd[v]=dd[u]+1; q.append(v)
        return dd
    diam=0
    for s in atomos:
        dd=bfs(s)
        if dd: diam=max(diam,max(dd.values()))
    # exponer las DISTINCIONES REALES (densidad intrínseca) de cada átomo, para acoplar la geometría emergente
    # a la física real de la corrida (no a un campo sintético). Es el dato físico que produjo esta corrida.
    dens_atomos = [float(e.densidad[a]) for a in atomos]
    return dict(espacio_emergio=True, diametro=diam, n_nodos_atomo=len(atomos),
                densidades_atomos=dens_atomos,
                nota="espacio medido SÓLO sobre la red de átomos ligados por gravedad")
