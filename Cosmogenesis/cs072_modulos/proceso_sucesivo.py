"""
CS072 — PROCESO SUCESIVO INTEGRADO (un solo motor, épocas por temperatura descendente).
Une TODO lo hallado: la física del Modelo Estándar (nucleo.corre) + las dos fases de expansión
(pre-luz corta largo alcance ilimitado / post-luz horizonte causal) que hacen EMERGER la dimensión,
+ el control de materia oscura como condición de contorno del "lejos".

NO cerrar ningún experimento hasta que Alexis diga que terminó (ver NOTA_PERMANENTE_CS.md).

Épocas (una sola corrida, T cae log-inversa):
  1. Plasma rugoso (asimetría de distribución, #23)          -> catalogo
  2. Confinamiento (fuerte liga tríos RGB)                    -> nucleo FASE 1
  3. Aniquilación por (color,sabor), sin tasa, invariante     -> nucleo FASE 1
  4. Freeze-out débil (ratio p:n ~7:1 emergente)              -> freeze_out
  5. Recombinación EM (electrón+protón -> H), nace la luz     -> nucleo FASE 2
  6. FASE PRE-LUZ (inflación): corte de largo alcance ILIMITADO (rompe mundo pequeño)
  7. FASE POST-LUZ (causal): enlaces sólo en horizonte causal sobre D distinciones independientes
  8. Lectores: distancia (diámetro), DIMENSIÓN (pendiente log-log Hausdorff), tiempo, materia oscura.
"""
import numpy as np, collections
from cs072_modulos.nucleo import corre
from cs072_modulos.catalogo import densidad_intrinseca

try:
    from scipy.spatial import cKDTree
    _HAY_KD = True
except Exception:
    _HAY_KD = False

# --- lectores geométricos (mismos que validamos) ---
def _diametro(adj, n, arranque=0):
    """Estimador double-sweep del diámetro. arranque = nodo físico de partida (NO el índice 0, que depende del
    orden del array). El caller pasa un nodo elegido por criterio físico -> estimador invariante a permutación.
    Desempate de 'más lejano' por (distancia, grado) para no depender del orden de iteración."""
    def bfs(s):
        d={s:0}; q=collections.deque([s])
        while q:
            u=q.popleft()
            for v in sorted(adj[u]):
                if v not in d: d[v]=d[u]+1; q.append(v)
        return d
    if n<2: return 0
    c=bfs(arranque)
    if len(c)<2: return 0
    # 'más lejano' con desempate determinista por grado (no por índice de iteración)
    f=max(c, key=lambda u:(c[u], len(adj[u])))
    d2=bfs(f)
    return max(d2.values())

def _malla_causal(V, k=4):
    """Horizonte causal: cada nodo se liga a sus k vecinos más cercanos en el espacio de D distinciones.
    DESEMPATE FÍSICO (anti-Shannon): cuando dos candidatos están casi-equidistantes, el empate NO se rompe por el
    índice del array (Shannon) sino por una MAGNITUD FÍSICA del nodo: la norma de su vector de distinciones
    (potencial térmico acumulado). Determinista, invariante a permutación."""
    m=len(V)
    adj=collections.defaultdict(set)
    # magnitud física por nodo (no el índice): norma del vector de distinciones = 'energía' relacional del nodo
    fis=np.sum(V*V, axis=1)
    arranque_fisico=int(np.argmax(fis))   # nodo de mayor magnitud: el MISMO bajo cualquier permutación
    if _HAY_KD:
        tree=cKDTree(V)
        # pedir un margen extra (2k+2) para tener los casi-empates y resolverlos por física
        kk=min(2*k+2, m)
        dist,idx=tree.query(V, k=kk)
        for i in range(m):
            cand=idx[i][1:]; cd=dist[i][1:]
            # ordenar candidatos por (distancia redondeada a tolerancia, magnitud física) -> desempate NO por índice
            tol=1e-9+1e-6*max(cd.max(),1e-12)
            orden=sorted(range(len(cand)), key=lambda t:(round(cd[t]/tol), fis[cand[t]], cd[t]))
            for t in orden[:k]:
                j=int(cand[t]); adj[i].add(j); adj[j].add(i)
    else:
        for i in range(m):
            d2=np.sum((V-V[i])**2, axis=1); d2[i]=np.inf
            cand=np.argsort(d2)[:2*k+2]
            tol=1e-9+1e-6*max(float(d2[cand].max()),1e-12)
            orden=sorted(cand, key=lambda j:(round(float(d2[j])/tol), fis[j], float(d2[j])))
            for j in orden[:k]: adj[i].add(int(j)); adj[int(j)].add(i)
    return adj, m, arranque_fisico

def _ejes_independientes(n, D, s0=1000):
    """D distinciones térmicas INDEPENDIENTES (mismo campo, barajado determinista distinto por eje; corr~0)."""
    base=densidad_intrinseca(n, 1.5)
    return np.column_stack([base[np.random.default_rng(s0+e).permutation(n)] for e in range(D)])

def dimension_acoplada(nq, naq, ne, npos, D, k=4, escalas=(1,2,3,4), tasa_expansion=0.02, pasos=60, amp_rugosidad=1.5, apagar=frozenset()):
    """LA DIMENSIÓN DE ESTE UNIVERSO (Nivel 2 -- fosilizada con el primer átomo). ACOPLAMIENTO GENUINO: la
    dimensión emerge de los átomos REALES que la física produjo (sus densidades intrínsecas), NO de un campo
    sintético. Escala el nº de partículas de entrada -> más átomos reales -> más puntos para medir el crecimiento
    del diámetro. Cada eje = una distinción real derivada de la densidad de los átomos. Si apagas una fuerza y
    deja de haber átomos, la dimensión CAE -> acoplamiento verdadero (proceso, no sucesión). Por eso cae a None si
    hay <8 átomos reales por escala (o <3 escalas válidas): sin átomo no hay espacio, y sin espacio no hay
    dimensión que medir (mismo guardián que _geometria/distancia y tiempo_emergente). Ésta es la ÚNICA medida
    de dimensión que se adjudica como "la dimensión de este universo"; ver nota en dimension_emergente()."""
    dd=[]; nn=[]
    for f in escalas:
        obs = corre(nq*f, naq*f, ne*f, npos*f, tasa_expansion=tasa_expansion, pasos=pasos, amp_rugosidad=amp_rugosidad, apagar=apagar)
        geo = obs.get("geometria") or {}
        dens = geo.get("densidades_atomos") or []
        if len(dens) < 8:   # sin átomos suficientes reales -> no hay geometría que medir
            continue
        dens=np.array(dens, float)
        # D distinciones reales: la densidad real barajada de forma determinista distinta por eje (corr~0)
        V=np.column_stack([dens[np.random.default_rng(2000+e).permutation(len(dens))] for e in range(D)])
        adj,nn_,arr=_malla_causal(V, min(k, len(dens)-1)); dd.append(_diametro(adj,nn_,arr)); nn.append(nn_)
    if len(dd)<3 or min(dd)<1:
        return dict(pendiente=None, dim_efectiva=None, n_puntos=len(dd), nota="pocos átomos reales para medir dim")
    pend=float(np.polyfit(np.log(nn), np.log(dd), 1)[0])
    return dict(pendiente=round(pend,3), dim_efectiva=round(1.0/pend,2) if pend>0.05 else None,
                n_atomos_por_escala=nn, diametros=dd, acoplada=True)

def dimension_emergente(n_atomos, D, k=4, Ns=None):
    """LEY DEL RÉGIMEN DE MALLADO (Nivel 1) -- NO ES LA DIMENSIÓN DE ESTE UNIVERSO. Mide la dimensión efectiva =
    1/pendiente(log diámetro vs log N) del horizonte causal sobre D distinciones. Es la dimensión de Hausdorff
    (crecimiento del diámetro), el mismo estimador que usa CDT.
    ROBUSTEZ: la dimensión es una propiedad del RÉGIMEN, no del conteo puntual de átomos -> se mide SIEMPRE sobre
    un ENSEMBLE DEDICADO de N grande fijo (como CDT, que mide en su ensemble, no en una sola triangulación).
    n_atomos se ignora para la medición dimensional (queda como dato del Modelo Estándar aparte).
    ADVERTENCIA DE ENMARCADO (Lectura A, ver RESPUESTA_CC_dim_ensemble_y_direccion_CS.md): este V es un campo
    SINTÉTICO -- barajados del mismo escalar de densidad del catálogo, que existe DESDE ANTES del confinamiento,
    antes de cualquier átomo. Esta función corre siempre, condense o no un átomo esta corrida: mide en el régimen
    PRE-atómico, con un marco que preexiste al átomo. El número es correcto (ley del mallado, análoga a Hubble:
    propiedad del todo/régimen, no de un componente condensado), pero NO se debe citar como "la dimensión de este
    universo" -- ésa es dimension_acoplada(), la única fosilizada con el primer átomo real."""
    if Ns is None:
        Ns=[1000,2000,4000,8000,16000,32000]   # ensemble fijo grande -> pendiente estable, invariante
    if len(Ns)<3: return None
    dd=[]; nn=[]
    for m in Ns:
        V=_ejes_independientes(m, D); adj,nn_,arr=_malla_causal(V,k); dd.append(_diametro(adj,nn_,arr)); nn.append(nn_)
    if min(dd)<1: return None
    pend=float(np.polyfit(np.log(nn), np.log(dd), 1)[0])
    return dict(pendiente=round(pend,3), dim_efectiva=round(1.0/pend,2) if pend>0.05 else None,
                Ns=Ns, diametros=dd)

def control_materia_oscura(n_atomos, D=3, k=4):
    """CON oscura = horizonte causal (poda de largo alcance). SIN oscura = gravedad pura (grafo completo)."""
    V=_ejes_independientes(n_atomos, D)
    adj_con,_,arr=_malla_causal(V,k); diam_con=_diametro(adj_con,len(V),arr)
    # SIN oscura: gravedad bariónica sin poda liga a TODOS con TODOS -> grafo completo. Su diámetro es 1 por
    # definición (todo par es adyacente) para N>=2; no se materializa el grafo O(N^2) (evita colgar a N grande).
    diam_sin = 1 if len(V) >= 2 else 0
    return dict(diam_con_oscura=diam_con, diam_sin_oscura=diam_sin,
                oscura_necesaria=(diam_con>diam_sin))

def invariancia_dimension(n_atomos, D=3, k=4, n_perm=4):
    """La dimensión NO debe depender del orden del array (anti-Shannon)."""
    V=_ejes_independientes(n_atomos, D)
    base=_diametro(*_malla_causal(V,k))
    perms=[_diametro(*_malla_causal(V[np.random.default_rng(s).permutation(len(V))],k)) for s in range(n_perm)]
    return dict(base=base, perms=perms, invariante=(len(set([base]+perms))==1))

def proceso_sucesivo(nq, naq, ne, npos, D_distinciones=3, amp_rugosidad=1.5,
                     tasa_expansion=0.02, pasos=200, apagar=frozenset(), medir_acoplada=False):
    """UN SOLO PROCESO: física del Modelo Estándar -> materia -> átomos -> emergencia de dimensión.
    Devuelve el arco completo en un dict."""
    # --- épocas 1-5: Modelo Estándar (materia, freeze-out, átomos, tiempo) ---
    obs = corre(nq, naq, ne, npos, tasa_expansion=tasa_expansion, pasos=pasos,
                apagar=apagar, amp_rugosidad=amp_rugosidad)
    geo = obs.get("geometria") or {}
    n_at = geo.get("n_nodos_atomo", 0) or 0

    # --- épocas 6-7: dos fases de expansión -> emergencia de dimensión sobre D distinciones ---
    # (se corre sobre una población de átomos a escala; si hay muy pocos átomos reales, se reporta n_at real
    #  pero la ley dimensional se mide en su barrido propio, como en CDT que mide en su ensemble)
    # DOS medidas de dimensión, NO complementarias-al-mismo-nivel -- niveles distintos (Lectura A, adjudicado
    # 19-jul-2026, ver RESPUESTA_CC_dim_ensemble_y_direccion_CS.md). Las claves NO cambian (compatibilidad con
    # verificar_cs072.py y toda comprobación paralela ya corrida); lo que cambia es el enmarcado del significado:
    #  - dimension_acoplada = LA DIMENSIÓN DE ESTE UNIVERSO (Nivel 2). Fosilizada con el primer átomo real: emerge
    #    de los átomos REALES de esta corrida; si apagas una fuerza y no hay átomos, CAE (None). Es la ÚNICA que
    #    se adjudica como "dimensión del universo". Baja resolución (pocos átomos), cara (opt-in medir_acoplada=True).
    #  - dimension = LEY DEL RÉGIMEN DE MALLADO (Nivel 1). Mide sobre un ensemble sintético dedicado de N grande
    #    (como CDT), CIEGO a si esta corrida condensó o no un átomo -- corre igual, dé lo que dé la física. Alta
    #    resolución, estable, pero es un marco PRE-atómico (mismo error de categoría que medir dirección absoluta
    #    en el sustrato). El número es correcto y se conserva; NUNCA se cita como "la dimensión de este universo".
    dim_acoplada = dimension_acoplada(nq, naq, ne, npos, D_distinciones, apagar=apagar) if medir_acoplada else None
    dim = dimension_emergente(n_at, D_distinciones)   # ensemble dedicado (n_at se ignora dentro) -- Nivel 1, ley del régimen
    osc = control_materia_oscura(8000, D_distinciones)
    inv = invariancia_dimension(8000, D_distinciones)

    nota_dimension = ('dimension = ley del régimen de mallado (Nivel 1, marco PRE-atómico; NO es la dimensión de '
                       'este universo); dimension_acoplada = dimensión DE ESTE UNIVERSO (Nivel 2, fosilizada con '
                       'el primer átomo real; None si <8 átomos reales).')

    return dict(
        # Modelo Estándar
        bariones=obs.get("bariones"), protones=obs.get("protones"), neutrones=obs.get("neutrones"),
        ratio_pn_congelado=obs.get("ratio_pn_congelado"),
        n_atomos=n_at, hidrogeno=obs.get("hidrogeno"), helio=obs.get("helio"),
        diametro_red=obs.get("diametro_red"), tiempo=obs.get("tiempo"),
        # geometría emergente (dos medidas, dos niveles -- ver nota_dimension)
        dimension=dim, dimension_acoplada=dim_acoplada, nota_dimension=nota_dimension,
        materia_oscura=osc, invariancia=inv,
        epocas=obs.get("epocas"),
    )
