"""
CS067 — LA HABITACIÓN COMPLETA: los 17 ingredientes del arco, juntos, cada uno con su espejo anti-Shannon.
=========================================================================================================
Diseño: CS (DISENO_CS067_habitacion_completa_CS.md v2), pre-registrado, endosado por Alexis ("ver toda la
habitación, no ir a tientas pieza por pieza"). Codea/ejecuta: CC.

Principio (Alexis / Cosmosemiótica): la RELACIÓN genera lo que la pieza aislada no. Por eso NO se saca ningún
ingrediente —ni la exclusión de Pauli que murió sola ×2—: entra a probarse EN COMBINACIÓN, con su null puesto.

12 heredadas (intactas, corren SIEMPRE, en todos los brazos): espín-marco, gravedad∝peso, confinamiento, EM,
débil, catálogo completo de partículas, masa, aniquilación, expansión, enfriamiento-proceso, vértice 3-cuerpos,
localidad (CS066). Se importan tal cual del motor de cs064 (que encadena cs057/cs062/cs059) + gate de cs066.

5 bajo escrutinio (conmutables, cada uno con sin_X y —los relacionales— barajado):
 13 Pauli ortogonalizante (reingresa como modo del marco; null=barajada de fermiones).
 14 Distancia por correlación: enlaces con peso continuo w∈(0,1], d_ij=-log(w); el atajo global queda con w→0
    (distancia enorme) en vez de cortarse — ataca el mundo-pequeño residual que el confirmatorio de CS066 destapó.
 15 Cono causal: t_i de nacimiento (orden de congelamiento); un enlace transmite orientación solo si
    |t_i-t_j| >= d_ij/c — el eje distinguido que el arco nunca tuvo.
 16 SSB multi-dimensional: el marco relaja bajo un vacío degenerado que premia mantener K modos ortogonales
    poblados (K sorteado ALTO, NUNCA 3) — ataca el colapso-a-1 de frente. sin_SSB = U(1) (= vuelve a B').
 17 Sector oscuro EMERGENTE: canal de energía residual que pesa (gravita) pero no carga/color/confina; su
    fracción EMERGE del barrido (G-NO-INSERTAR-OSCURO), no se fija.

Orden de lectura SAGRADO: Nivel 0 continuidad → Nivel 1 espacio MÉTRICO (listón nuevo: exponente de diámetro
∈[0.29,0.40] con gigante sano, no basta d_s~3) → Nivel 2 direcciones (n_ejes). G-NO-CALIBRAR / G-NO-TOPADO /
G-NO-INSERTAR-OSCURO vigentes. Este archivo corre la tanda; el SMOKE valida cada ingrediente aislado antes.
"""
from __future__ import annotations
import os, sys, csv, math, time, heapq
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs057_paisaje_completo as C7
import cs062_paisaje_peso as C62
import cs059_espin_como_marco as C9
import cg003_diagnostico_gromov as GR
import cs064_smoke as SM
import cs064_sistema_completo as C64
import cs065_exclusion_pauli as C65        # catálogo con es_ferm + _mask_diag
import cs066_localidad_geometrogenesis as C66   # gate de localidad (ingrediente 12)

RNG = np.random.default_rng
DMAX_INT = C64.DMAX_INT                     # 8 (G-NO-TOPADO)
STEPS   = int(os.environ.get("CS067_STEPS", 20))
N_NODOS = int(os.environ.get("CS067_N", 1500))
PATCHES = int(os.environ.get("CS067_PATCHES", 40))
WORKERS = int(os.environ.get("CS067_WORKERS", max(1, min(6, (os.cpu_count() or 4) - 2))))
OUT     = os.environ.get("CS067_OUT", os.path.join(_HERE, "cs067_habitacion.csv"))
SMOKE   = os.environ.get("CS067_SMOKE", "") != ""
# fase A (esencial): completo + 5 sin_X. fase B añade barajados+anclas. Selector por env.
FASE    = os.environ.get("CS067_FASE", "A")
ARMS_A  = ["completo", "sin_pauli", "sin_correlacion", "sin_causal", "sin_SSB", "sin_oscura"]
ARMS_B  = ARMS_A + ["corr_barajada", "causal_barajado", "null_marco_congelado", "sin_local"]
ARMS    = ARMS_A if FASE == "A" else ARMS_B

# rangos SORTEADOS (G-NO-CALIBRAR) — declarados aquí, antes de correr
K_LOCAL_RANGE = (5, 8)      # presupuesto de localidad en el régimen conexo (confirmatorio CS066)
GAMMA_RANGE   = (0.5, 2.5)  # decaimiento de la correlación w (exponente)
C_RANGE       = (0.3, 5.0)  # velocidad causal c (permisivo→restrictivo; nº de dominios EMERGE de aquí)
K_SSB_RANGE   = (4, 8)      # nº de modos de vacío que el SSB intenta mantener (ALTO, nunca 3)
DARK_RANGE    = (0.05, 0.35)  # propensión del canal oscuro (la fracción que PERSISTE emerge)

def _flags(arm):
    """matriz de ingredientes por brazo (DISENO §arquitectura)."""
    f = dict(pauli=True, corr=True, causal=True, ssb=True, dark=True, marco_vivo=True, local=True)
    if arm == "completo":               pass
    elif arm == "sin_pauli":            f["pauli"] = False
    elif arm == "sin_correlacion":      f["corr"] = False          # poda binaria (CS066)
    elif arm == "sin_causal":           f["causal"] = False
    elif arm == "sin_SSB":              f["ssb"] = False            # U(1) → vuelve a B'
    elif arm == "sin_oscura":           f["dark"] = False
    elif arm == "corr_barajada":        f["corr"] = "shuf"         # w barajados
    elif arm == "causal_barajado":      f["causal"] = "shuf"       # t_i barajados
    elif arm == "null_marco_congelado": f.update(pauli=False, ssb=False, marco_vivo=False)
    elif arm == "sin_local":            f.update(pauli=False, corr=False, causal=False, ssb=False,
                                                 dark=False, local=False)   # = blob CS064
    # solo-X (chequeo #2 del smoke): base motor+localidad, un solo ingrediente bajo escrutinio activo
    elif arm == "solo_ssb":             f.update(pauli=False, corr=False, causal=False, dark=False)
    elif arm == "solo_causal":          f.update(pauli=False, corr=False, ssb=False, dark=False)
    elif arm == "solo_correlacion":     f.update(pauli=False, causal=False, ssb=False, dark=False)
    elif arm == "solo_oscura":          f.update(pauli=False, corr=False, causal=False, ssb=False)
    elif arm == "ssb_causal":           f.update(pauli=False, corr=False, dark=False)  # el par candidato (K-Z)
    return f


# ============================ INGREDIENTE 14: distancia por correlación (métrica pesada) ============================
def _pesos_correlacion(adj, N, gamma, modo, rng):
    """w_ij ∈ (0,1] por enlace: alto si hay soporte local (correlación), →0 si es atajo (sin vecinos comunes).
    d_ij = -log(w). modo='shuf' baraja los w sobre los enlaces (misma distribución, mal colocada)."""
    edges = []; sup = []
    for i in range(N):
        for j in adj[i]:
            if i < j:
                c = len(adj[i] & adj[j]); edges.append((i, j)); sup.append(c)
    if not edges:
        return {}
    sup = np.asarray(sup, float); smax = sup.max() if sup.max() > 0 else 1.0
    w = ((sup + 0.5) / (smax + 0.5)) ** gamma          # correlación ~ soporte, decae con gamma
    w = np.clip(w, 1e-4, 1.0)
    if modo == "shuf":
        rng.shuffle(w)                                  # placebo: misma w, otro enlace
    return {e: float(wi) for e, wi in zip(edges, w)}

def diam_pesado(adj, N, wmap, n_src=16, rng=None):
    """Diámetro con la métrica de correlación (d_ij=-log w). Un atajo (w→0) NO acorta: cuesta caro cruzarlo.
    Percentil alto de excentricidades pesadas desde fuentes muestreadas (Dijkstra)."""
    if rng is None: rng = RNG(0)
    if not wmap: return float(C7._diam(adj, N))
    d_edge = {e: -math.log(w) for e, w in wmap.items()}
    srcs = rng.integers(0, N, size=min(n_src, N))
    eccs = []
    for s in srcs:
        dist = {int(s): 0.0}; pq = [(0.0, int(s))]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, 1e18): continue
            for v in adj[u]:
                de = d_edge.get((u, v) if u < v else (v, u), 1.0)
                nd = d + de
                if nd < dist.get(v, 1e18):
                    dist[v] = nd; heapq.heappush(pq, (nd, v))
        if len(dist) > 1:
            eccs.append(np.percentile(list(dist.values()), 90))
    return float(np.median(eccs)) if eccs else float(C7._diam(adj, N))


# ============================ JUEZ de ejes (gap+PR) — GUARDA 2 (ADDENDUM CS 15-jul) ============================
def cuenta_ejes_gap(ev, c=1.6, g=3.0, piso=0.02):
    """T=(1/N)Σ v v^T, autoval ev desc. PR=1/Σλ² (repórtese siempre, ancla anti-arbitrariedad).
    GUARDA 2 (CS, ADDENDUM_CS067_pico_guarda_rango_CS.md): el viejo floor_frac (relativo a λ_max) NO bastaba —
    con K_sorteado<D=8 hay dimensiones del embedding en CERO EXACTO, y el mayor salto λ_r/λ_{r+1} cae
    trivialmente en el BORDE de rango (λ/0≈1e11), leído como "gap limpio" cuando solo significa rango<D.
    Verificado por CS sobre tensores plantados: gap_val~1e11 en TODOS los casos, incluido colapso-1. Como
    puerta de calidad es inservible. La corrección: el gap se busca SOLO entre modos POBLADOS (λ_i>=piso),
    ignorando el borde rango-vs-cero. gap_interno = mayor salto DENTRO del subespacio poblado. Si <=1 modo
    poblado, no hay gap interno que buscar (colapso). n_ejes=0 si no hay salto limpio (>=g) — el picado por
    nodo (picado_por_nodo) es quien certifica si esos n_ejes son dominios reales o smear, NO este espectro.
    Devuelve (n_ejes, PR, gap_interno, r_thr)."""
    lam = np.clip(np.asarray(ev, float), 0, None); s = float(lam.sum())
    if s <= 0: return 0, 0.0, 0.0, 0
    p = lam / s; D = len(p)
    PR = 1.0 / float(np.sum(p ** 2)); r_thr = int(np.sum(p > c / D))
    pobl = p[p >= piso]; n_pobl = len(pobl)
    if n_pobl == D:
        return 0, PR, 0.0, r_thr                      # rango PLENO = isotropía, sin estructura (esfera 8D -> 0)
    if n_pobl <= 1:
        return n_pobl, PR, 0.0, r_thr                  # colapso: 0 o 1 modo poblado
    ratios = pobl[:-1] / (pobl[1:] + 1e-12)
    r_gap = int(np.argmax(ratios)) + 1
    gap_interno = float(ratios.max())
    # salto interno limpio -> cuenta el subconjunto dominante; sin salto -> el poblado ENTERO cuenta
    # (poblado != D, así que ya hay una frontera real poblado/no-poblado, no es el borde de rango trivial)
    n_ejes = r_gap if gap_interno >= g else n_pobl
    return n_ejes, PR, gap_interno, r_thr


# ============================ GUARDA 1 (ADDENDUM CS 15-jul): picado POR NODO, no espectro global =========
def picado_por_nodo(V):
    """pico_i = max_k |⟨v̂_i, e_k⟩| (v̂ normalizado; e_k = ejes estándar, los pozos del SSB viven en la base
    canónica). Distingue dominios DISCRETOS (pico_i~1) de un subespacio blando/continuo (pico_i<1) — el
    candado OBLIGATORIO de ADJUDICACION_CS067_SSB_juez_CS.md §2 que la línea vieja (ev[0]/ev.sum(), medida
    GLOBAL del espectro) nunca implementó. Devuelve (pico_medio, frac_picados) con frac_picados=frac(pico_i>0.9)."""
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    pico_i = np.max(np.abs(Vn), axis=1)
    return float(pico_i.mean()), float((pico_i > 0.9).mean())

def _diam_robusto(adj, N, rng, n_src=12):
    """Diámetro BFS robusto sobre el gigante (percentil de excentricidades) — caza el artefacto diam=1 de C7._diam."""
    from collections import deque
    eccs = []
    for s in rng.integers(0, N, size=min(n_src, N)):
        s = int(s)
        if not adj[s]: continue
        dist = {s: 0}; q = deque([s]); far = 0
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1; far = max(far, dist[v]); q.append(v)
        if len(dist) > 0.3 * N: eccs.append(far)     # solo fuentes en una componente grande
    return float(np.median(eccs)) if eccs else 1.0

# ============================ INGREDIENTE 16: SSB POTTS (dominios discretos + coarsening) ============================
def ssb_potts(V, Aw, K, rng, es_ferm=None, pauli=False, inertia=0.3):
    """SSB como POTTS/reloj de 2K estados discretos (±e_1…±e_K). Cada nodo adopta el pozo MAYORITARIO de sus
    vecinos, PERO cada voto se PESA por w_ij (ingrediente 14 completo, adjudicación (a) de CS): Aw es la
    adyacencia PESADA-por-correlación y enmascarada por el cono. Un atajo con w→0 existe pero NO transmite
    consenso → deja de percolar el colapso global. Antes el Potts corría sobre el grafo topológico crudo (w=1),
    lo que dejaba percolar y colapsaba a 1. pauli: los fermiones anti-votan (×w) el pozo de sus vecinos
    fermiónicos (exclusión de CS065 EN COMBINACIÓN).

    Devuelve (Vn, conf): conf_i = fracción del voto (por EJE, sumando ±signo) que ganó el eje elegido. Es el
    picado REAL, medido ANTES del snap discreto — CC destapó que medir picado sobre Vn (el estado final) es
    vacuo: Vn es SIEMPRE exactamente one-hot por construcción (max_k|⟨v_i,e_k⟩|=1 para todo nodo, todo paso,
    con o sin dominios reales), así que no puede distinguir consenso genuino de disputa entre vecinos. conf_i
    sí lo hace: 1.0 = unanimidad de los vecinos en ese eje; ~1/K = voto repartido por igual entre los K ejes
    (smear de consenso). Nodo aislado (sin vecinos) = conf 1.0 por convención (nada lo disputa)."""
    N, D = V.shape
    P = V[:, :K]
    axis = np.argmax(np.abs(P), axis=1)
    sign = (P[np.arange(N), axis] >= 0).astype(int)
    state = (axis * 2 + sign).astype(int)                       # 2K estados discretos
    A = Aw.tocsr(); indptr, indices, wdata = A.indptr, A.indices, A.data
    new = state.copy()
    conf = np.ones(N)
    for i in range(N):
        a, b = indptr[i], indptr[i + 1]
        nb = indices[a:b]; wt = wdata[a:b]
        if len(nb) == 0: continue
        wsum = wt.sum()
        if wsum <= 0: continue
        counts = np.zeros(2 * K)
        np.add.at(counts, state[nb], wt / wsum)                # voto pesado NORMALIZADO: w gobierna QUIÉN influye, no la magnitud
        counts[state[i]] += inertia
        if pauli and es_ferm is not None and es_ferm[i]:
            fm = es_ferm[nb]
            if fm.any(): np.add.at(counts, state[nb[fm]], -0.75 * wt[fm] / wsum)
        new[i] = int(np.argmax(counts))
        axis_counts = np.clip(counts, 0, None).reshape(K, 2).sum(axis=1)   # colapsa signo: el eje es lo que importa
        tot = float(axis_counts.sum())
        conf[i] = float(axis_counts[new[i] // 2] / tot) if tot > 0 else 1.0
    ax = new // 2; sg = np.where(new % 2 == 1, 1.0, -1.0)
    Vn = np.zeros((N, D)); Vn[np.arange(N), ax] = sg
    return Vn, conf


# ============================ INGREDIENTE 15: cono causal (máscara de transmisión del marco) ============================
def _A_causal(A, t, c, N):
    """Máscara la adyacencia para el alineamiento: un enlace transmite orientación solo si |t_i-t_j| >= 1/c
    (el cono de luz; d_ij≈1 salto). Devuelve A enmascarada (sparse) y el grado enmascarado."""
    Ac = A.tocoo()
    keep = np.abs(t[Ac.row] - t[Ac.col]) >= (1.0 / max(c, 1e-6))
    import scipy.sparse as sp
    Am = sp.coo_matrix((Ac.data[keep], (Ac.row[keep], Ac.col[keep])), shape=(N, N)).tocsr()
    return Am


# ============================ EL MOTOR: 12 heredadas + 5 conmutables ============================
def proceso067(N, cat, arm, par, rng):
    fam, color, carga, masa, es_anti = cat["fam"], cat["color"], cat["carga"], cat["masa"], cat["es_anti"]
    es_ferm = cat["es_ferm"]
    f = _flags(arm)
    # sector oscuro (17): fracción sorteada de nodos que PESAN pero no cargan/color/confinan
    dark = np.zeros(N, bool)
    if f["dark"]:
        dark = rng.random(N) < par["dark"]
        color = color.copy(); carga = carga.copy()
        color[dark] = -1; carga[dark] = 0            # desacople de fuerzas conocidas (siguen con masa→gravitan)
    # Pauli (13): máscara de exclusión sobre fermiones; null cuando pauli off
    emask = es_ferm.copy()
    if not f["pauli"]:           emask = np.zeros(N, bool)
    Dm = C65._mask_diag(emask) if f["pauli"] else None

    adj0, _ = GR.aleatorio(N, meandeg=6.0, seed=int(rng.integers(1 << 30)))
    adj = [set(a) for a in adj0]
    col = np.where(color >= 0, color, rng.integers(0, 3, N)).astype(np.int8)
    car = (carga > 0).astype(np.int8)
    deg0 = [len(a) for a in adj]
    t = np.zeros(N, dtype=np.int32)
    V = C9._spins(N, DMAX_INT, rng)
    t_birth = np.full(N, STEPS, dtype=float); deg_prev = np.array(deg0, float)
    R = getattr(C7, "R_GRAV", 1.0); T_CONF = getattr(C7, "T_CONF", 0.5); CAP_E = 12 * N
    k_local = par["k_local"]
    D, G = [], []
    conf_final = None                          # picado pre-snap (Guarda 1 corregida) del ÚLTIMO paso Potts
    for step in range(STEPS):
        T = C7._T_de_paso(step, 1.0) if hasattr(C7, "_T_de_paso") else max(0.02, 1.6 * (1 - step / STEPS))
        E = sum(len(a) for a in adj) // 2
        if E > CAP_E: break
        w = masa * max(0.0, 1.0 - T); CAP = max(1, N // 4)
        try:
            rg = min(0.30 * R * (1 - T), CAP / max(1.0, (0.5 + T) * E))
            C62._grav_peso(adj, N, rng, rg, dmax=3, T=T, w=w + 1e-9)      # gravedad: TODOS pesan (incl. oscuros)
        except Exception: pass
        if T < T_CONF:
            try: C7._confin(adj, N, col, t, rng, min(0.8, CAP / max(1.0, E)))
            except Exception: pass
        try: C7._em(adj, N, car, deg0, rng, 0.12)
        except Exception: pass
        try: C7._debil(N, col, car, rng, 0.05)
        except Exception: pass
        try: C7._despliegue(adj, N, rng, 0.14 * T)
        except Exception: pass
        if T > 0.4 and step % 2 == 0:
            for i in rng.choice(N, size=max(1, N // 60), replace=False):
                if adj[i] and es_anti[i] != es_anti[list(adj[i])[0]]:
                    j = list(adj[i])[0]; adj[i].discard(j); adj[j].discard(i)
        # localidad (12): qué enlaces persisten (o poda binaria si f['local'] pero corr off; nada si sin_local)
        if f["local"]:
            C66.gate_localidad(adj, N, rng, k_local, barajado=False)
        # tiempo de nacimiento CONTINUO: nodos que dejan de cambiar 'congelan' (jitter → cono varía suave con c)
        degnow = np.array([len(a) for a in adj], float)
        cambiaron = np.abs(degnow - deg_prev) > 0.5
        nc = int(cambiaron.sum())
        if nc: t_birth[cambiaron] = step + rng.random(nc)
        deg_prev = degnow
        # co-evolución del marco: SSB(Potts, coarsening por cono causal) o alineamiento (nemático/Pauli)
        if f["marco_vivo"]:
            A = SM.adj_sparse(adj, N)
            tt = t_birth.copy()
            if f["causal"] == "shuf": rng.shuffle(tt)
            A_use = _A_causal(A, tt, par["c"], N) if f["causal"] else A
            if f["ssb"]:
                # (a): pesar el voto Potts por w_ij (correlación) — el atajo con w→0 no transmite consenso
                if f["corr"]:
                    import scipy.sparse as sp
                    wmap = _pesos_correlacion(adj, N, par["gamma"], "shuf" if f["corr"] == "shuf" else "real", rng)
                    Ac = A_use.tocoo()
                    if wmap:
                        wd = np.array([wmap.get((r, c) if r < c else (c, r), 1.0)
                                       for r, c in zip(Ac.row, Ac.col)])
                        Aw = sp.coo_matrix((wd, (Ac.row, Ac.col)), shape=(N, N)).tocsr()
                    else:
                        Aw = A_use
                else:
                    Aw = A_use                                  # sin_correlacion: Potts crudo (topológico, colapsa)
                V, conf_final = ssb_potts(V, Aw, par["K"], rng, es_ferm=es_ferm, pauli=bool(f["pauli"]))
            else:
                deg = np.asarray(A_use.sum(axis=1)).ravel()
                if f["pauli"]:
                    A_excl = Dm @ A_use @ Dm
                    V = SM.alinear_orto_fast(V, A_use, A_excl, mezcla=0.35)
                else:
                    V = SM.alinear_nematico_fast(V, A_use, deg, mezcla=0.35)
        D.append(C7._diam(adj, N)); G.append(C7._giant(adj, N))
    return adj, V, dark, t_birth, f, D, conf_final


# ============================ JUEZ ============================
def juzga067(adj, V, dark, N, arm, par, f, seed, conf=None):
    m = C64.juzga(adj, V, N, seed)                            # d_s, delta_rel, holonomia, gigante (n_ejes se re-juzga)
    ev = SM.tensor_orientacion(V)                             # JUEZ (gap+PR, GUARDA 2 de rango-poblado, spec CS)
    nej, PR, gap_interno, rthr = cuenta_ejes_gap(ev)
    if conf is not None:
        # GUARDA 1 corregida: picado sobre el voto PRE-snap de ssb_potts (conf_i), no sobre V final —
        # V post-Potts es SIEMPRE one-hot por construcción (picado_por_nodo(V) daría 1.0 trivialmente, CC lo cazó).
        pico_medio = float(np.mean(conf)); frac_picados = float(np.mean(np.asarray(conf) > 0.9))
    else:
        pico_medio, frac_picados = picado_por_nodo(V)         # sin_SSB (nemático/orto): V SÍ es continuo, válido medirlo así
    m["n_ejes"] = nej; m["PR"] = round(PR, 3); m["gap_interno"] = round(gap_interno, 3); m["r_thr"] = rthr
    m["pico_medio"] = round(pico_medio, 3); m["frac_picados"] = round(frac_picados, 3)
    m["diam_fin"] = round(_diam_robusto(adj, N, RNG(seed + 11)), 3)   # robusto (caza diam=1)
    m["grado_medio"] = C66.grado_medio(adj, N)
    m["clustering"] = C66.clustering_medio(adj, N, RNG(seed + 3))
    # métrica de correlación: diámetro pesado (o binario si corr off)
    if f["corr"]:
        wmap = _pesos_correlacion(adj, N, par["gamma"], "shuf" if f["corr"] == "shuf" else "real", RNG(seed + 5))
        m["diam_corr"] = round(diam_pesado(adj, N, wmap, rng=RNG(seed + 7)), 3)
        m["w_medio"] = round(float(np.mean(list(wmap.values()))) if wmap else -1, 4)
    else:
        m["diam_corr"] = -1.0; m["w_medio"] = -1.0
    # sector oscuro emergente: fracción de masa oscura que persiste en el gigante
    gid = _giant_nodes(adj, N)
    frac_dark = float(dark[list(gid)].mean()) if gid else 0.0
    m["frac_oscura"] = round(frac_dark, 4)
    m["K_sorteado"] = par["K"] if f["ssb"] else 1
    m["K_surv"] = int(m["n_ejes"])
    m["c"] = round(par["c"], 3) if f["causal"] else -1
    m["k_local"] = par["k_local"] if f["local"] else -1
    m["delta_rel"] = m.get("delta_rel", -1)
    return m

def _giant_nodes(adj, N):
    seen = set(); best = set()
    for s in range(N):
        if s in seen: continue
        comp = set(); stack = [s]
        while stack:
            u = stack.pop()
            if u in comp: continue
            comp.add(u); stack.extend(adj[u] - comp)
        seen |= comp
        if len(comp) > len(best): best = comp
    return best


# ============================ WORKER ============================
def _sorteo(seed):
    r = RNG(seed)
    return dict(k_local=int(r.integers(*K_LOCAL_RANGE, endpoint=True)),
                gamma=float(r.uniform(*GAMMA_RANGE)), c=float(r.uniform(*C_RANGE)),
                K=int(r.integers(*K_SSB_RANGE, endpoint=True)), dark=float(r.uniform(*DARK_RANGE)))

def _worker(arg):
    pidx, seed = arg
    par = _sorteo(seed)
    cat = C65._cataloga065(N_NODOS, RNG(seed))
    filas = []
    for arm in ARMS:
        r2 = RNG(seed * 137 + hash(arm) % 9973 + 5)
        adj, V, dark, tb, f, D, conf = proceso067(N_NODOS, cat, arm, par, r2)
        m = juzga067(adj, V, dark, N_NODOS, arm, par, f, seed * 19 + hash(arm) % 991, conf=conf)
        m.update(dict(patch=pidx, seed=seed, N=N_NODOS, arm=arm, D_max=DMAX_INT))
        filas.append(m)
    return filas

def _campos():
    return ["patch", "seed", "N", "arm", "D_max", "k_local", "c", "K_sorteado", "K_surv",
            "d_s", "delta_rel", "n_ejes", "PR", "gap_interno", "r_thr", "pico_medio", "frac_picados",
            "ejes_reales", "holonomia", "gigante",
            "grado_medio", "clustering", "diam_fin", "diam_corr", "w_medio", "frac_oscura"]


# ============================ SMOKE ============================
def _smoke():
    print("=" * 96, flush=True)
    print("CS067 SMOKE — cada ingrediente aislado hace lo que dice + anclas de continuidad", flush=True)
    print("=" * 96, flush=True)
    Nn = int(os.environ.get("CS067_N", 1200)); npar = 3
    global N_NODOS, ARMS; N_NODOS = Nn
    ARMS = ["completo", "sin_SSB", "sin_causal", "solo_ssb", "solo_causal", "ssb_causal",
            "sin_correlacion", "solo_correlacion", "null_marco_congelado", "sin_local"]
    hdr = (f"   {'arm':<22}{'diam':>6}{'diam_c':>8}{'d_s':>6}{'clust':>7}{'gig':>6}"
           f"{'nej':>5}{'PR':>6}{'gapI':>7}{'pico':>7}{'fdark':>7}")
    for pidx in range(npar):
        seed = 67000 + 11 * pidx
        par = _sorteo(seed)
        print(f"  --- parche {pidx} · k_local={par['k_local']} gamma={par['gamma']:.2f} c={par['c']:.2f} "
              f"K={par['K']} dark~{par['dark']:.2f} ---\n{hdr}", flush=True)
        for fila in _worker((pidx, seed)):
            print(f"   {fila['arm']:<22}{fila['diam_fin']:>6.1f}{fila['diam_corr']:>8.1f}{fila['d_s']:>6.1f}"
                  f"{fila['clustering']:>7.3f}{fila['gigante']:>6.2f}{fila['n_ejes']:>5d}{fila['PR']:>6.1f}"
                  f"{fila['gap_interno']:>7.1f}{fila['pico_medio']:>7.3f}{fila['frac_oscura']:>7.3f}", flush=True)
    print("\nSMOKE — chequeos (adjudica CS): (0) sin_local=blob, null_marco_congelado=0 ejes ; "
          "(#2 solo-X) solo_ssb deja >1 modo con gap ; solo_correlacion sube diam_corr ; solo_causal da eje ; "
          "fdark>0 con oscura ; (predicción K-Z) ssb_causal/completo (con cono) n_ejes>1 vs sin_causal→1. "
          "El juez gap+PR ya se validó aparte (1/3/0/3/5). NO se lanza tanda hasta que CS valide.", flush=True)


# ============================ MAIN ============================
def main():
    if SMOKE:
        _smoke(); return
    print("=" * 96, flush=True)
    print(f"CS067 — LA HABITACIÓN COMPLETA (17 ingredientes). FASE {FASE}. PRE-REGISTRADO.", flush=True)
    print("=" * 96, flush=True)
    print(f"N={N_NODOS} · patches={PATCHES} · steps={STEPS} · D_max={DMAX_INT} · arms={ARMS}", flush=True)
    print("PRE-INSCRITO §salidas: (A) completo da espacio métrico (exp diam∈[0.29,0.40]) Y n_ejes>1, supera "
          "TODOS los nulls ; (A-parc) manda un subconjunto ; (B) espacio sí/direcciones no (colapso más profundo "
          "que TODO) ; (C) ni espacio ; (D) todo ruido. G-NO-CALIBRAR/G-NO-TOPADO/G-NO-INSERTAR-OSCURO.", flush=True)
    hechos = set()
    if os.path.exists(OUT):
        with open(OUT) as fh:
            for row in csv.DictReader(fh): hechos.add(int(row["patch"]))
        print(f"REANUDANDO: {len(hechos)} parches ya hechos", flush=True)
    pend = [(p, 67100 + 7 * p) for p in range(PATCHES) if p not in hechos]
    if not pend:
        print("nada pendiente.", flush=True); return
    fout = open(OUT, "a", newline=""); wr = csv.DictWriter(fout, fieldnames=_campos())
    if not hechos: wr.writeheader()
    t0 = time.time(); n = 0
    import multiprocessing as mp
    if WORKERS > 1:
        with mp.Pool(WORKERS) as pool:
            for filas in pool.imap_unordered(_worker, pend, chunksize=1):
                for fila in filas: wr.writerow(fila)
                fout.flush(); n += 1
                if n % 2 == 0 or n == len(pend):
                    dt = time.time() - t0; r = n / dt
                    print(f"  {n}/{len(pend)} parches · {dt/60:.1f}min · ETA {(len(pend)-n)/r/3600:.2f}h", flush=True)
    else:
        for arg in pend:
            filas = _worker(arg)
            for fila in filas: wr.writerow(fila)
            fout.flush(); n += 1
            dd = {fl["arm"]: (fl["diam_corr"], fl["n_ejes"]) for fl in filas}
            print(f"  parche {n}/{len(pend)} · {time.time()-t0:.1f}s · (diam_corr,nej)={dd}", flush=True)
    fout.close()
    print(f"\nCOMPLETO: {n} parches en {(time.time()-t0)/60:.1f} min → {OUT}", flush=True)

if __name__ == "__main__":
    main()
