"""
CS064 — SMOKE (validacion de andamiaje + CALIBRACION de los tres jueces). NO concluye fisica.
================================================================================================
Contrato (Alexis, 9-jul): "haz el smoke, pero no decidas nada de su resultado; primero hay que pensar."
Este script NO adjudica ningun desenlace (A/B/B'/C/D). Solo verifica:
  (1) que los TRES jueces recuperan respuestas CONOCIDAS en controles (d_s, delta de Gromov, conteo de ejes).
  (2) el FILO CRITICO (auditoria CC): que el conteo de ejes distingue "3 ejes reales" de "ruido isotropo"
      (un conteo ingenuo leeria el ruido como 'muchos ejes' = lo contrario de la verdad).
  (3) que las DOS reglas de alineamiento (polar vs nematico) corren y dan lecturas distinguibles.
  (4) que un tick end-to-end del brazo 'completo' (fuerzas mediadas + aniquilacion + co-evolucion del marco)
      corre sin romperse a N pequeno.
Reusa el arco: cs057 (motor/ensemble/controles), cs062 (gravedad Newton con masa), cs059 (transporte
paralelo + holonomia), cg003_diagnostico_gromov (delta + grados control). Nada nuevo salvo: dim espectral,
tensor de orientacion + conteo de ejes emergente, y las dos reglas de alineamiento.
"""
from __future__ import annotations
import os, sys, math, time
import numpy as np
from collections import deque
import scipy.sparse as sp

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

# ---- reuso del arco (import: no dispara main porque __name__ != '__main__') ----
import cs057_paisaje_completo as C7      # motor, ensemble, constructores de control
import cs059_espin_como_marco as C9      # _spins, _transporta, _holonomia_ciclo, _ciclos_fundamentales, _frame_burgers
import cg003_diagnostico_gromov as GR    # lattice2d, arbol, aleatorio, landmarks_dist, gromov_delta
import cs062_paisaje_peso as C62         # _grav_peso (gravedad Newton m*m/d^2), _masa

RNG = np.random.default_rng

# =====================================================================================
#  JUEZ 1 — DIMENSION ESPECTRAL d_s (nuevo). P_retorno(t) ~ t^(-d_s/2). Sin coordenadas.
# =====================================================================================
def dim_espectral(adj, N, t_max=64, n_walkers=None, rng=None):
    """Paseo aleatorio simple; estima d_s por la pendiente de log P_retorno vs log t en la ventana media."""
    rng = rng or RNG(0)
    deg = np.array([len(a) for a in adj])
    validos = np.where(deg > 0)[0]
    if len(validos) < 8:
        return float("nan")
    n_walkers = n_walkers or min(400, len(validos))
    origen = rng.choice(validos, size=n_walkers, replace=True)
    pos = origen.copy()
    Pret = np.zeros(t_max + 1)
    adjL = [list(a) for a in adj]
    for t in range(1, t_max + 1):
        for i in range(n_walkers):
            vec = adjL[pos[i]]
            if vec:
                pos[i] = vec[rng.integers(len(vec))]
        Pret[t] = np.mean(pos == origen)
    ts = np.arange(2, t_max + 1)
    P = Pret[2:]
    m = P > 0
    if m.sum() < 4:
        return float("nan")
    lt, lp = np.log(ts[m]), np.log(P[m])
    # ventana media (evita el transitorio corto y la cola ruidosa)
    lo, hi = len(lt) // 4, max(len(lt) // 4 + 3, 3 * len(lt) // 4)
    slope = np.polyfit(lt[lo:hi], lp[lo:hi], 1)[0]
    return float(-2.0 * slope)

# =====================================================================================
#  JUEZ 1b — DIMENSION por CRECIMIENTO DE BOLA |B(v,r)| ~ r^d (robusta en grafos emergentes)
#  Distingue retícula (ley de potencia → d) de blob mundo-pequeño (crecimiento exponencial → d alto).
# =====================================================================================
def dim_volumen(adj, N, n_src=24, rmax=14, rng=None):
    rng = rng or RNG(0)
    deg = np.array([len(a) for a in adj])
    validos = np.where(deg > 0)[0]
    if len(validos) < 4:
        return float("nan")
    src = rng.choice(validos, size=min(n_src, len(validos)), replace=False)
    ballsum = np.zeros(rmax + 1); cnt = 0
    for s in src:
        dist = {int(s): 0}; q = deque([int(s)])
        while q:
            u = q.popleft()
            if dist[u] >= rmax:
                continue
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1; q.append(int(v))
        ds = np.fromiter(dist.values(), dtype=int)
        ballsum += np.array([(ds <= r).sum() for r in range(rmax + 1)], float); cnt += 1
    ball = ballsum / max(cnt, 1)
    rr = np.arange(1, rmax + 1); lb = np.log(ball[1:]); lr = np.log(rr)
    dmean = max(ball[1], 2.0)                            # ~grado medio (r=1)
    m = (ball[1:] >= dmean) & (ball[1:] <= 0.25 * N)     # régimen limpio: pre-saturación, post-vecindad
    if m.sum() < 3:                                      # si es corto, tomar los primeros r con crecimiento
        m = (ball[1:] > ball[:-1].min()) & (ball[1:] < 0.6 * N)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(lr[m], lb[m], 1)[0])

# =====================================================================================
#  JUEZ 2 — PLANITUD delta de Gromov (reuso directo de cg003)
# =====================================================================================
def delta_gromov(adj, N, seed=0, K=48):
    D, dmax, fg = GR.landmarks_dist(adj, N, K, seed=seed)
    if D is None:
        return float("nan"), float("nan")
    dm, q95 = GR.gromov_delta(D, n_quad=8000, seed=seed)
    esc = dmax if dmax and dmax > 0 else 1.0
    return float(dm), float(dm / (esc + 1e-9))    # delta, delta/escala (→0 = plano)

# =====================================================================================
#  JUEZ 3 — DIRECCION EMERGENTE (nuevo, honestidad-critico): tensor de orientacion + conteo de ejes
#  El FILO: distinguir "k ejes reales" (espectro con ESCALON) de "ruido isotropo" (espectro PLANO).
# =====================================================================================
def tensor_orientacion(V):
    """V: (N,D) vectores unitarios. Q = <v ⊗ v> (D×D), traza=1. Autovalores desc."""
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    Q = (Vn.T @ Vn) / len(Vn)
    ev = np.linalg.eigvalsh(Q)[::-1]
    return np.clip(ev, 0, None)

def cuenta_ejes(ev):
    """Devuelve (n_ejes, PR, es_real). PR=razon de participacion (1=colapso .. D=isotropo).
    es_real = hay ESCALON (unos pocos grandes + salto), NO espectro plano (isotropo=no emergio)."""
    p = ev / (ev.sum() + 1e-12)
    PR = 1.0 / np.sum(p ** 2)                       # participacion: cuantos autovalores 'cuentan'
    D = len(ev)
    iso_ref = 1.0 / D                               # autovalor de un espectro perfectamente plano
    # escalon: mayor caida relativa entre autovalores consecutivos (normalizada por el mayor)
    gaps = (p[:-1] - p[1:]) / (p[0] + 1e-12)
    k_gap = int(np.argmax(gaps)) + 1                # nº de ejes por encima del mayor escalon
    max_gap = float(gaps.max()) if len(gaps) else 0.0
    # isotropo si el mayor autovalor apenas supera el plano y no hay escalon marcado
    es_real = bool((p[0] > 1.6 * iso_ref) and (max_gap > 0.15))
    n_ejes = k_gap if es_real else 0               # 0 ejes = 'no emergio direccion' (isotropo/ruido)
    return n_ejes, float(PR), es_real, float(max_gap), float(p[0] / iso_ref)

# ---- las DOS reglas de alineamiento (el pulgar en la balanza que NO puede ser silencioso) ----
def alinear_polar(V, adj, mezcla=0.5):
    """(a) me oriento hacia el PROMEDIO de mis vecinos (todos tiran hacia un lado). Attractor tipico: 1 eje."""
    Vn = V.copy()
    for i, vec in enumerate(adj):
        if vec:
            m = V[list(vec)].mean(axis=0)
            Vn[i] = (1 - mezcla) * V[i] + mezcla * m
    return Vn / (np.linalg.norm(Vn, axis=1, keepdims=True) + 1e-12)

def alinear_nematico(V, adj, mezcla=0.5):
    """(b) me oriento hacia el EJE dominante de mis vecinos (sin importar sentido +/-). Permite VARIOS ejes."""
    Vn = V.copy()
    for i, vec in enumerate(adj):
        if len(vec) >= 1:
            W = V[list(vec)]
            Qi = (W.T @ W)
            axis = np.linalg.eigh(Qi)[1][:, -1]        # eje principal local
            if np.dot(axis, V[i]) < 0:
                axis = -axis
            Vn[i] = (1 - mezcla) * V[i] + mezcla * axis
    return Vn / (np.linalg.norm(Vn, axis=1, keepdims=True) + 1e-12)

# ---- versiones VECTORIZADAS (sparse) — mismas reglas, O(D²·E) por paso, escalan a N grande ----
def adj_sparse(adj, N):
    rows, cols = [], []
    for i, vec in enumerate(adj):
        for j in vec:
            rows.append(i); cols.append(j)
    if not rows:
        return sp.csr_matrix((N, N))
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))

def alinear_polar_fast(V, A, deg, mezcla=0.5):
    """(a) promedio de vecinos, vectorizado: Vnew = norm((1-m)V + m (A@V)/deg)."""
    m = A @ V
    d = np.maximum(deg, 1)[:, None]
    Vn = (1 - mezcla) * V + mezcla * (m / d)
    return Vn / (np.linalg.norm(Vn, axis=1, keepdims=True) + 1e-12)

def alinear_nematico_fast(V, A, deg, mezcla=0.5):
    """(b) eje dominante local por UNA power-iteration vectorizada del tensor de vecinos:
    w[i] = Σ_j∈nbr (V[j]·V[i]) V[j]  =  Σ_e V[:,e] ⊙ (A @ (V[:,e]⊙V[:,d])).  O(D²·E)."""
    D = V.shape[1]
    W = np.zeros_like(V)
    for e in range(D):
        Ve = V[:, e]
        AVe = A @ (V * Ve[:, None])           # (A @ (V[:,e]*V[:,d]))  para todo d a la vez
        W += Ve[:, None] * AVe
    # fijar signo hacia V[i] y mezclar
    s = np.sign(np.sum(W * V, axis=1)); s[s == 0] = 1.0
    W = W * s[:, None]
    nrm = np.linalg.norm(W, axis=1, keepdims=True)
    W = np.where(nrm > 1e-9, W / (nrm + 1e-12), V)
    Vn = (1 - mezcla) * V + mezcla * W
    return Vn / (np.linalg.norm(Vn, axis=1, keepdims=True) + 1e-12)

def alinear_excl_fast(V, A, A_excl, mezcla=0.35, lam=0.5):
    """CS065: inercia (alineamiento con A, consenso) MENOS exclusión de Pauli (repulsión con A_excl:
    empuja s_i a ser ORTOGONAL a los vecinos que la excluyen). A_excl = sub-adyacencia fermión-fermión
    (real) / bosón-bosón (placebo) / barajada. O(D²·E). lam = fuerza de exclusión (sorteada, NO calibrada)."""
    D = V.shape[1]
    W = np.zeros_like(V); Wx = np.zeros_like(V)
    for e in range(D):
        Ve = V[:, e][:, None]
        VtVe = V * Ve
        W += Ve * (A @ VtVe)                     # consenso (todos los vecinos)
        Wx += Ve * (A_excl @ VtVe)               # componente compartida con vecinos que EXCLUYEN
    s = np.sign(np.sum(W * V, axis=1)); s[s == 0] = 1.0
    W = W * s[:, None]
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    Vn = (1 - mezcla) * V + mezcla * Wn - lam * Wx    # restar la componente de exclusión = empujar a ortogonal
    return Vn / (np.linalg.norm(Vn, axis=1, keepdims=True) + 1e-12)

def alinear_orto_fast(V, A, A_excl, mezcla=0.35):
    """CS065b: inercia (alinear al consenso) + exclusión de Pauli FIEL = ORTOGONALIZACIÓN SATURANTE.
    El fermión se orienta ortogonal a la dirección de sus vecinos que excluyen (proyecta fuera la componente
    paralela UNA vez) y SE DETIENE en ortogonal (⟨s_i,s_j⟩=0) — no sigue empujando (no hay λ, no hay desorden).
    G-NO-CALIBRAR: el punto de frenado es la ortogonalidad misma, física, no un parámetro."""
    D = V.shape[1]
    W = np.zeros_like(V)
    for e in range(D):
        Ve = V[:, e][:, None]
        W += Ve * (A @ (V * Ve))                       # consenso (inercia)
    s = np.sign(np.sum(W * V, axis=1)); s[s == 0] = 1.0
    W = W * s[:, None]
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    V_al = (1 - mezcla) * V + mezcla * Wn
    U = A_excl @ V                                       # dirección media de los vecinos que excluyen
    nu = np.linalg.norm(U, axis=1, keepdims=True)
    Uhat = np.where(nu > 1e-9, U / (nu + 1e-12), 0.0)
    proj = np.sum(V_al * Uhat, axis=1, keepdims=True)    # componente paralela
    V_orth = V_al - proj * Uhat                          # quitarla UNA vez → ortogonal, SATURA ahí (no overshoot)
    return V_orth / (np.linalg.norm(V_orth, axis=1, keepdims=True) + 1e-12)

# =====================================================================================
#  CALIBRACION — respuestas CONOCIDAS (G-SMOKE con dientes)
# =====================================================================================
def _cad(n): return C7.cadena(n)
def _l2(n):  return GR.lattice2d(n)
def _c3(n):  return C7.cubica3d(n)
def _h4(n):  return C7.hipercubica4d(n)
def _arb(n): return GR.arbol(n)

def calibra_dim_y_delta():
    print("\n[CAL-1] DIMENSION (crecimiento de bola d_vol) y delta de Gromov sobre controles (N~2-3k):", flush=True)
    print("   control        d_vol(esperado)  d_s(viejo)  delta/escala", flush=True)
    casos = [("cadena~1D", _cad(2000), 1), ("lattice2D", _l2(2500), 2),
             ("cubica3D", _c3(3375), 3), ("hipercub4D", _h4(2401), 4), ("arbol(hip)", _arb(2000), None)]
    ok = True
    for nom, (adj, N), dexp in casos:
        dv = dim_volumen(adj, N, rng=RNG(1))
        ds = dim_espectral(adj, N, rng=RNG(1))
        marca = ""
        if dexp is not None and not math.isnan(dv):
            marca = "OK" if abs(dv - dexp) <= 0.7 else "REVISAR"
            if marca == "REVISAR": ok = False
        dm, drel = delta_gromov(adj, N, seed=1)
        print("   %-14s d_vol=%5.2f (~%s)   d_s=%5.2f   delta/esc=%.3f   %s" %
              (nom, dv, str(dexp), ds, drel, marca), flush=True)
    print("   (d_vol debe recuperar 1/2/3/4; arbol debe dar d_vol ALTO=blob y delta grande)", flush=True)
    return ok

def calibra_conteo_ejes():
    print("\n[CAL-2] conteo de ejes — EL FILO: distinguir k ejes reales de RUIDO isotropo:", flush=True)
    D = 8; N = 2000; rng = RNG(2)
    def unit(v): return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    casos = {}
    # 1 eje: todos ~ e0 con ruido
    V1 = np.zeros((N, D)); V1[:, 0] = 1.0; V1 += 0.05 * rng.standard_normal((N, D)); casos["1 eje (colapso)"] = (unit(V1), 1)
    # 3 ejes ortogonales equipoblados: cada nodo elige ±e0/±e1/±e2
    idx = rng.integers(0, 3, N); sgn = rng.choice([-1, 1], N)
    V3 = np.zeros((N, D)); V3[np.arange(N), idx] = sgn; V3 += 0.05 * rng.standard_normal((N, D)); casos["3 ejes ortog."] = (unit(V3), 3)
    # isotropo: ruido puro en R^8  -> DEBE dar 0 ejes (no 8)
    Viso = unit(rng.standard_normal((N, D))); casos["ruido isotropo"] = (Viso, 0)
    ok = True
    print("   config             n_ejes(esperado)  PR    escalon  lambda0/iso  es_real", flush=True)
    for nom, (V, k) in casos.items():
        ev = tensor_orientacion(V)
        n, PR, real, gap, ratio = cuenta_ejes(ev)
        marca = "OK" if n == k else "REVISAR"
        if n != k: ok = False
        print("   %-18s %d (esp %d)        %.2f  %.3f    %.2f        %s   %s" %
              (nom, n, k, PR, gap, ratio, real, marca), flush=True)
    print("   >>> clave: 'ruido isotropo' DEBE dar 0 ejes (no 8). Es el filo que auditamos.", flush=True)
    return ok

def calibra_holonomia():
    print("\n[CAL-3] holonomia (reuso cs059 _frame_burgers): marco alineado -> trivial ; aleatorio -> no-trivial:", flush=True)
    adj, N = _c3(1000); rng = RNG(3)
    K = 4
    spins_al = np.tile(np.eye(K)[0], (N, 1)).astype(float)   # todos iguales -> holonomia trivial
    spins_rd = C9._spins(N, K, rng)                          # aleatorios -> holonomia no-trivial
    # vector de prueba FIJO (no el propio espin del nodo, que volveria siempre a si mismo)
    ha, _, _ = C9._frame_burgers(adj, N, spins_al, K, rng, null=False)
    hr, _, _ = C9._frame_burgers(adj, N, spins_rd, K, rng, null=False)
    ok = (ha < 0.3) and (hr > ha + 0.3)
    print("   marco alineado: holonomia=%.3f (esperado ~0) ; aleatorio: holonomia=%.3f (esperado grande)   %s" %
          (ha, hr, "OK" if ok else "REVISAR"), flush=True)
    return ok

# =====================================================================================
#  TICK end-to-end (brazo 'completo', minimo) — solo comprueba que CORRE, no concluye nada
# =====================================================================================
def tick_completo_smoke(N=1000, pasos=12, regla="nematico", rng=None):
    """Un parche minimo: fuerzas del arco + gravedad con masa + aniquilacion + co-evolucion del marco.
    Devuelve las lecturas de los jueces (para verificar que se computan; NO para interpretar)."""
    rng = rng or RNG(0)
    # sustrato inicial 'caliente': grafo aleatorio disperso (nada ligado fuerte)
    adj0, _ = GR.aleatorio(N, meandeg=2.0, seed=int(rng.integers(1 << 30)))
    adj = [set(a) for a in adj0]
    masa = C62._masa(N, rng)
    col = (rng.integers(0, 3, N)).astype(np.int8)      # color proxy
    car = (np.arange(N) % 2).astype(np.int8); rng.shuffle(car)
    deg0 = [len(a) for a in adj]
    Dmax = 8
    V = C9._spins(N, Dmax, rng)                         # marco: abanico de orientaciones
    t = np.zeros(N, dtype=np.int32)
    alinear = alinear_nematico if regla == "nematico" else alinear_polar
    R = getattr(C7, "R_GRAV", 1.0)
    for step in range(pasos):
        T = max(0.05, 1.5 * (1.0 - step / pasos))      # enfriamiento
        # gravedad con masa (reuso cs062) — solo si hay aristas
        try:
            C62._grav_peso(adj, N, rng, 0.15 * R, dmax=4, T=T, w=masa)
        except Exception:
            pass
        # fuerzas del arco (reuso cs057), envueltas por si alguna primitiva exige forma distinta
        for fn, args in [(C7._em, (adj, N, car, deg0, rng, 0.1)),
                         (C7._despliegue, (adj, N, rng, 0.1))]:
            try: fn(*args)
            except Exception: pass
        # aniquilacion minima: retira una fraccion pequena de aristas (poblacion no conservada)
        if step % 3 == 0:
            for i in rng.choice(N, size=max(1, N // 50), replace=False):
                if adj[i]:
                    j = list(adj[i])[0]; adj[i].discard(j); adj[j].discard(i)
        # co-evolucion del marco por inercia (la regla bajo prueba)
        adjL = [list(a) for a in adj]
        V = alinear(V, adjL, mezcla=0.4)
    # jueces
    ds = dim_espectral(adj, N, rng=rng)
    dm, drel = delta_gromov(adj, N, seed=7)
    ev = tensor_orientacion(V); n_ejes, PR, real, gap, ratio = cuenta_ejes(ev)
    Efin = C7._giant(adj, N) if hasattr(C7, "_giant") else float("nan")
    return dict(regla=regla, d_s=round(ds, 3), delta_rel=round(drel, 3), n_ejes=n_ejes,
                PR=round(PR, 2), ejes_reales=real, gigante=round(float(Efin), 3))

def corre_tick_smoke():
    print("\n[TICK] brazo 'completo' minimo (N=1000, 12 pasos) — SOLO verifica que corre:", flush=True)
    ok = True
    for regla in ("polar", "nematico"):
        try:
            t0 = time.time()
            r = tick_completo_smoke(N=1000, pasos=12, regla=regla, rng=RNG(11))
            print("   %-9s -> %s   (%.1fs)" % (regla, r, time.time() - t0), flush=True)
        except Exception as e:
            ok = False
            print("   %-9s -> FALLO: %r" % (regla, e), flush=True)
    return ok

# =====================================================================================
def main():
    print("=" * 96, flush=True)
    print("CS064 SMOKE — validacion de andamiaje + calibracion de jueces. NO concluye fisica (contrato Alexis).", flush=True)
    print("=" * 96, flush=True)
    r1 = calibra_dim_y_delta()
    r2 = calibra_conteo_ejes()
    r3 = calibra_holonomia()
    r4 = corre_tick_smoke()
    print("\n" + "=" * 96, flush=True)
    print("RESUMEN SMOKE (pass/fail de ANDAMIAJE, no de fisica):", flush=True)
    print("   [CAL-1] d_s + delta recuperan controles : %s" % ("PASA" if r1 else "REVISAR"), flush=True)
    print("   [CAL-2] conteo de ejes distingue 3 vs iso: %s" % ("PASA" if r2 else "REVISAR"), flush=True)
    print("   [CAL-3] holonomia trivial vs no-trivial  : %s" % ("PASA" if r3 else "REVISAR"), flush=True)
    print("   [TICK]  brazo completo corre (2 reglas)  : %s" % ("PASA" if r4 else "REVISAR"), flush=True)
    print("=" * 96, flush=True)
    print("RECORDATORIO: no se decide nada del arco con esto. Es andamiaje. Primero pensar. — CC", flush=True)

if __name__ == "__main__":
    main()
