"""
CS057 — EL PAISAJE COMPLETO: barrido de TODAS las fuerzas (0→1) + sector oscuro EMERGENTE + sync/async
=====================================================================================================
Planteo de Alexis: fuera el Big Bang con 2 variables y tres copas de vino. Probar MILES de variantes de
inicio y ver QUÉ combinaciones de fuerzas ESTABILIZAN un universo persistente EN EXPANSIÓN — de la dimensión
que sea (3D, 4D, lo que salga). Nuestro universo (3D-plano-expansión) es UN punto del mapa, no el objetivo.
Incluir materia/energía oscuras NO como algo dado, sino como PROBABILIDAD de algo emergente cuando todas las
fuerzas varían juntas. Probar sincrónico (default físico, un tiempo) vs asincrónico (por turnos) para FALSAR
la tesis "es un proceso, no una sucesión". La distancia modula cada fuerza según su alcance.

DISEÑO CS057 de Claude Science (DISENO_CS057_paisaje_completo.md). Esta es la implementación de CC.

QUÉ SE BARRE (7 ejes, muestreo SOBOL de baja discrepancia sobre [0,1]^7):
  w_grav · w_strong(confinamiento) · w_em · w_weak · w_exp(despliegue) · w_cool(enfriamiento) · alcance_grav
Cada peso escala CUÁNTO actúa esa fuerza por paso. El enfriamiento enciende el confinamiento (T<umbral).

LA DISTANCIA MODULA EL ALCANCE (subsume CS056-v2): gravedad LARGA (se acumula), EM CORTA (se cancela),
fuerte/débil ULTRACORTA (vecindad). Toda distancia por SALTOS DE GRAFO (BFS), jamás coordenada.

CRITERIO CIEGO (la trampa cerrada): ESTABLE = componente gigante persistente con geometría medible (un blob
colapsado NO; un gas fragmentado NO). EXPANDE = el diámetro (saltos) CRECE. VIABLE = estable Y expande.
Medido por TIPOS de retículo, ciego a los pesos. SECTOR OSCURO = SALIDA emergente: energía oscura = expansión
que se ACELERA sola (2ª diferencia del diámetro > 0, sin insertar ningún término); materia oscura = proxy de
gravitación/contracción extra. G-NO-INSERTAR-OSCURO: ningún término se llama "oscuro" ni se ajusta.

DOS BRAZOS: sync (las 4 fuerzas cada paso, juntas) vs async (por turnos, una fuerza por fase). Si dan el
MISMO paisaje → "es un proceso" FALSADO. Si sync estabiliza universos que async NO → tesis PROBADA.

ESCALA (no negociable, §7-bis): Sobol 4096 pts × 8 semillas × 2 brazos ≈ 65k corridas + punto físico denso.
Paralelizado por puntos (independientes), CHECKPOINT incremental por fila (reanuda sin recomenzar). Config por
env: CS057_POINTS, CS057_SEEDS, CS057_STEPS, CS057_WORKERS, CS057_OUT, CS057_SMOKE.

numpy + scipy(qmc.Sobol) + multiprocessing. Autodescriptivo por diseño (preferencia de Alexis).
"""
from __future__ import annotations

import os
import sys
import csv
import math
import time
import numpy as np
from collections import deque

# ------- reusar constructores del ensemble y medidores de CS055 (exec-strip-main) -------
_HERE = os.path.dirname(os.path.abspath(__file__))
_s = open(os.path.join(_HERE, "cs055_proceso_acoplado.py")).read().replace("\nmain()\n", "\n")
_C5 = {"__file__": os.path.join(_HERE, "cs055_proceso_acoplado.py")}
exec(compile(_s, "cs055_proceso_acoplado.py", "exec"), _C5)
_giant = _C5["_giant"]; _diam = _C5["_diam"]; _colores = _C5["_colores"]
cadena = _C5["cadena"]; cuadrada2d = _C5["cuadrada2d"]; cubica3d = _C5["cubica3d"]
hipercubica4d = _C5["hipercubica4d"]
# tri_hiperbolica (curvo/hiperbólico) desde cg004f, para construir el curvo a tamaño MODERADO (no ~1000)
_sf = open(os.path.join(_HERE, "cg004f_barrido_curvatura.py")).read().replace("\nmain()\n", "\n")
_F = {"__file__": os.path.join(_HERE, "cg004f_barrido_curvatura.py")}
exec(compile(_sf, "cg004f_barrido_curvatura.py", "exec"), _F)
tri_hiperbolica = _F["tri_hiperbolica"]


# ============================ CONFIG (por env, para smoke-test o tanda completa) ============================
N_POINTS  = int(os.environ.get("CS057_POINTS", 4096))      # puntos Sobol del hipercubo [0,1]^7
SEEDS     = int(os.environ.get("CS057_SEEDS", 8))          # réplicas por punto (robustez)
STEPS     = int(os.environ.get("CS057_STEPS", 16))         # pasos (divisible por 4 = las 4 fuerzas → turnos parejos en async)
WORKERS   = int(os.environ.get("CS057_WORKERS", max(1, (os.cpu_count() or 4) - 2)))
OUT       = os.environ.get("CS057_OUT", os.path.join(_HERE, "cs057_paisaje.csv"))
SMOKE     = os.environ.get("CS057_SMOKE", "") != ""        # si set: corre mini para validar+cronometrar
DIM_DENSO = int(os.environ.get("CS057_DENSO", 256))        # sub-barrido denso alrededor del punto físico
NDIM      = 7                                              # ejes del hipercubo
# tasas base (el valor cuando el peso = 1); alcances fijados por FÍSICA (G-ALCANCE-FISICO, no se afinan)
R_GRAV, R_STRONG, R_EM, R_WEAK, R_EXP = 0.10, 0.10, 0.10, 0.02, 0.12
DMAX_EM = 2                                                # EM alcance CORTO (se cancela por neutralidad)
ALPHA   = 2                                               # caída 1/d² (cuadrado inverso emergente)
SAT     = 6
T_HI, T_LO, T_CONF = 3.0, 0.04, 1.0
# ==========================================================================================================


# ------------------------------ FUERZAS PARAMETRIZADAS (peso + alcance explícitos) ------------------------------
def _grav(adj, N, rng, rate, dmax, T):
    """Gravedad: contrae ∝ densidad (grado), cae 1/d^ALPHA por SALTOS, alcance dmax. Solo atrae. (CS054-v2)."""
    if rate <= 0:
        return
    E = sum(len(a) for a in adj) // 2
    rho = np.array([len(a) for a in adj], float)
    if rho.sum() <= 0 or E < 1:
        return
    nadd = int(rate * (0.5 + T) * E)
    if nadd <= 0:
        return
    srcs = rng.choice(N, size=nadd, p=rho / rho.sum())
    for i in srcs:
        i = int(i); dist = {i: 0}; q = deque([i])
        while q:
            u = q.popleft()
            if dist[u] >= dmax:
                continue
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1; q.append(int(w))
        cand = [(j, d) for j, d in dist.items() if d >= 2]
        if not cand:
            continue
        w = np.array([rho[j] / (d ** ALPHA) for j, d in cand])
        if w.sum() <= 0:
            continue
        j = cand[int(rng.choice(len(cand), p=w / w.sum()))][0]
        adj[i].add(j); adj[j].add(i)


def _confin(adj, N, col, t, rng, rate):
    """Fuerte/confinamiento: tríos NEUTROS R+V+A (ULTRACORTO, vecindad 2 saltos). Solo ve COLOR. (CS055)."""
    if rate <= 0:
        return
    E = sum(len(a) for a in adj) // 2
    nc = int(rate * E)
    for _ in range(nc):
        i = int(rng.integers(N))
        if t[i] >= SAT:
            continue
        ci = col[i]
        vecinos2 = set()
        for u in adj[i]:
            vecinos2.add(u)
            for w in adj[u]:
                vecinos2.add(w)
        vecinos2.discard(i)
        otros = [x for x in vecinos2 if col[x] != ci]
        hecho = False
        for a in otros:
            if hecho:
                break
            for b in adj[a]:
                if b != i and col[b] != ci and col[b] != col[a] and b in vecinos2:
                    adj[i].add(a); adj[a].add(i); adj[i].add(b); adj[b].add(i)
                    t[i] += 1
                    hecho = True
                    break


def _em(adj, N, carga, deg0, rng, rate):
    """EM (CORTO): atrae opuestos (1/d², dmax corto), repele iguales SOLO donde la gravedad comprimió
    (grado sobre el basal) — modelo JUSTO de CS056: la repulsión frena colapso, no erosiona retículo sano."""
    if rate <= 0:
        return
    E = sum(len(a) for a in adj) // 2
    if E < 2:
        return
    mismas = [(i, j) for i in range(N) for j in adj[i]
              if i < j and carga[i] == carga[j] and len(adj[i]) > deg0[i] and len(adj[j]) > deg0[j]]
    if mismas:
        rng.shuffle(mismas)
        for (i, j) in mismas[:int(rate * len(mismas))]:
            adj[i].discard(j); adj[j].discard(i)
    nadd = int(rate * E)
    for _ in range(nadd):
        i = int(rng.integers(N))
        if not adj[i]:
            continue
        dist = {i: 0}; q = deque([i])
        while q:
            u = q.popleft()
            if dist[u] >= DMAX_EM:
                continue
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1; q.append(int(w))
        cand = [(j, d) for j, d in dist.items() if d >= 2 and carga[j] != carga[i]]
        if not cand:
            continue
        w = np.array([1.0 / (d ** 2) for j, d in cand])
        j = cand[int(rng.choice(len(cand), p=w / w.sum()))][0]
        adj[i].add(j); adj[j].add(i)


def _debil(N, col, carga, rng, prob):
    """Débil (ULTRACORTO): transmuta el TIPO (color/carga) de nodos al azar, prob baja. Deja escapar metaestables."""
    if prob <= 0:
        return
    flip = np.where(rng.random(N) < prob)[0]
    for i in flip:
        if rng.random() < 0.5:
            col[i] = np.int8(rng.integers(3))
        else:
            carga[i] = np.int8(1 if carga[i] == 0 else 0)


def _despliegue(adj, N, rng, rate):
    """Despliegue/expansión: remueve una fracción de vínculos (estira, sube el diámetro)."""
    if rate <= 0:
        return
    edges = [(i, j) for i in range(N) for j in adj[i] if i < j]
    if edges:
        rng.shuffle(edges)
        for (i, j) in edges[:int(rate * len(edges))]:
            adj[i].discard(j); adj[j].discard(i)


def _T_de_paso(step, w_cool):
    """Enfriamiento: T baja geométricamente; w_cool controla CUÁNTO baja (bajo=queda tibio, confin no prende)."""
    frac = step / max(STEPS - 1, 1)
    depth = 0.2 + 1.8 * w_cool
    return T_HI * (T_LO / T_HI) ** min(1.0, frac * depth)


# ------------------------------ EL PROCESO (una corrida sobre UN retículo) ------------------------------
def proceso057(adj0, N, color0, carga0, W, dmax_grav, arm, rng):
    """UN bucle temporal con las fuerzas moduladas por sus pesos W=(g,s,em,wk,exp,cool). Devuelve trayectoria
    (diámetro y %gigante por paso) para leer estable/expande/acelera CIEGO. arm='sync' (todas cada paso) o
    'async' (las 4 fuerzas por TURNOS — una por fase; cooling y expansión son el trasfondo)."""
    wg, ws, wem, wwk, wexp, wcool = W
    adj = [set(a) for a in adj0]; col = color0.copy(); car = carga0.copy()
    deg0 = [len(a) for a in adj]
    t = np.zeros(N, dtype=np.int32)
    D = []; G = []
    fuerzas = ("grav", "strong", "em", "weak")            # las 4 que se sincronizan o no
    NF = len(fuerzas)
    CAP_E = 12 * N                                         # tope de aristas (grado medio ~24): por encima ya es BLOB (no viable)

    def _aplica(f, T):                                     # aplica UNA fuerza a tasa 1× (misma dosis en ambos brazos)
        if f == "grav":
            _grav(adj, N, rng, wg * R_GRAV, dmax_grav, T)
        elif f == "strong":
            if T < T_CONF:                                # el enfriamiento enciende el confinamiento
                _confin(adj, N, col, t, rng, ws * R_STRONG)
        elif f == "em":
            _em(adj, N, car, deg0, rng, wem * R_EM)
        elif f == "weak":
            _debil(N, col, car, rng, wwk * R_WEAK)

    def _mide_y_corta():                                  # medición + cortes neutrales (gas/blob → no viable)
        dd = _diam(adj, N); gg = _giant(adj, N)
        D.append(dd); G.append(gg)
        return gg >= 0.9 and dd <= 2

    # DOS BRAZOS, dosis por fuerza y expansión TOTAL idénticas — lo único que difiere es la SIMULTANEIDAD:
    #  sync : cada macropaso las 4 fuerzas CO-OCURREN (se apilan) y LUEGO el trasfondo (expansión) estira una vez.
    #  async: NF micropasos por macropaso, UNA fuerza cada uno, con 1/NF de la expansión entre cada fuerza → las
    #         fuerzas actúan por turnos (B ve el grafo ya relajado por la expansión tras A), NUNCA simultáneas.
    for step in range(STEPS):
        T = _T_de_paso(step, wcool)
        E_now = sum(len(a) for a in adj) // 2
        if E_now < 2 or E_now > CAP_E:                    # gas disuelto o BLOB denso → CORTE NEUTRAL (no viable)
            D.append(_diam(adj, N)); G.append(_giant(adj, N))
            break
        if arm == "sync":
            for f in fuerzas:                             # las 4 juntas (co-ocurren)...
                _aplica(f, T)
            _despliegue(adj, N, rng, wexp * R_EXP)        # ...y después el trasfondo estira una vez
        else:
            for k in range(NF):                           # una fuerza por micropaso, expansión repartida entre ellas
                _aplica(fuerzas[(step + k) % NF], T)      # (rota el orden por macropaso: sin sesgo de orden fijo)
                _despliegue(adj, N, rng, wexp * R_EXP / NF)
        if _mide_y_corta():
            break
    return D, G


def _clasifica(D, G):
    """Métricas CIEGAS de la trayectoria (ninguna mira los pesos ni la dimensión objetivo):
    estable = gigante persistente + geometría medible (no punto, no gas); expande = diámetro crece;
    acelera = 2ª diferencia del diámetro > 0 en la mitad tardía (candidato ENERGÍA OSCURA, emergente)."""
    if not D:
        return dict(estable=0, expande=0, acelera=0, viable=0, d0=0, d1=0, g1=0.0)
    d0, d1 = D[0], D[-1]
    g1 = G[-1]
    half = len(G) // 2
    persist = min(G[half:]) if len(G) > half else g1
    estable = int(g1 >= 0.45 and d1 >= 2 and persist >= 0.35)
    expande = int(d1 > d0)
    # aceleración: 2ª diferencia del diámetro en la mitad tardía
    acelera = 0
    if len(D) >= 5 and expande:
        seg = D[half - 1:] if half >= 1 else D
        diff2 = [seg[k + 1] - 2 * seg[k] + seg[k - 1] for k in range(1, len(seg) - 1)]
        if diff2 and (sum(diff2) / len(diff2)) > 0:
            acelera = 1
    viable = int(estable and expande)
    return dict(estable=estable, expande=expande, acelera=acelera, viable=viable, d0=d0, d1=d1, g1=round(g1, 3))


# ------------------------------ ENSEMBLE (dims simétricas d≈1..4 + curvo; ninguna privilegiada) ------------------------------
def _construye_ensemble():
    """d≈1..4 plano + hiperbólico(curvo). Tamaños moderados (corrida barata para escalar a decenas de miles)."""
    _c = tri_hiperbolica(7, 360)           # hiperbólico {3,7} a ~360 nodos (moderado, como los demás)
    ens = [
        ("d1",   cadena(256)),
        ("d2",   cuadrada2d(289)),         # L=17 → 289
        ("d3",   cubica3d(343)),           # L=7  → 343
        ("d4",   hipercubica4d(256)),      # L=4  → 256
        ("curv", (list(_c[0]), _c[2])),    # hiperbólico {3,7}, N≈360
    ]
    return ens


_ENSEMBLE = None
def _ensemble():
    global _ENSEMBLE
    if _ENSEMBLE is None:
        _ENSEMBLE = _construye_ensemble()
    return _ENSEMBLE


CLASES = ["d1", "d2", "d3", "d4", "curv"]


def corrida(W, dmax_grav, seed, arm):
    """Una corrida = evaluar UN punto de fuerzas (W,alcance) con UNA semilla sobre TODO el ensemble, en un brazo.
    Devuelve un dict de métricas agregadas por clase de dimensión (ciego a los pesos)."""
    rng = np.random.default_rng(seed)
    fila = {}
    for ci, (nombre, (adj, N)) in enumerate(_ensemble()):
        col = _colores(N, np.random.default_rng(seed * 131 + ci * 17 + 1))
        car = (np.arange(N) % 2).astype(np.int8)
        np.random.default_rng(seed * 977 + ci * 29 + 7).shuffle(car)
        D, G = proceso057(adj, N, col, car, W, dmax_grav, arm, np.random.default_rng(seed * 31 + ci * 101 + 13))
        m = _clasifica(D, G)
        fila[f"viable_{nombre}"] = m["viable"]
        fila[f"estable_{nombre}"] = m["estable"]
        fila[f"expande_{nombre}"] = m["expande"]
        fila[f"acelera_{nombre}"] = m["acelera"]
        fila[f"gfin_{nombre}"] = m["g1"]
    return fila


# ------------------------------ MUESTREO DEL HIPERCUBO ------------------------------
def _sobol_puntos(n, seed=12345):
    from scipy.stats import qmc
    m = max(1, int(math.ceil(math.log2(max(n, 2)))))
    sob = qmc.Sobol(d=NDIM, scramble=True, seed=seed)
    pts = sob.random_base2(m=m)[:n]                        # [0,1]^NDIM, baja discrepancia
    return pts


def _punto_fisico():
    """El punto FÍSICO real (constantes del mundo): fuerte 1 · EM 1/137 · débil 1e-6 · gravedad ~0 (1e-38).
    Expansión/enfriamiento presentes (~0.5). Gravedad alcance LARGO (alc=1). Marcado, no objetivo."""
    return np.array([1e-38, 1.0, 1.0 / 137.0, 1e-6, 0.5, 0.5, 1.0])


def _denso_fisico(n):
    """Sub-barrido DENSO alrededor del punto físico: varía los ejes NO despreciables (exp, cool, alcance);
    los otros quedan en su valor físico (grav~0, strong=1, em=1/137, weak~0). Resuelve la vecindad del real."""
    base = _punto_fisico()
    rng = np.random.default_rng(999)
    out = []
    k = int(math.ceil(n ** (1 / 3)))
    for a in np.linspace(0.05, 0.95, k):          # w_exp
        for b in np.linspace(0.05, 0.95, k):      # w_cool
            for c in np.linspace(0.0, 1.0, k):    # alcance
                p = base.copy(); p[4] = a; p[5] = b; p[6] = c
                out.append(p)
    return np.array(out[:n])


def _alc_a_dmax(alc):
    return 2 + int(round(alc * 4))                        # alcance gravedad: 2..6 saltos (eje libre a mapear)


# ------------------------------ WORKER (un punto → filas de todas las semillas × brazos) ------------------------------
def _worker(arg):
    pid, W7, phys = arg
    W = tuple(float(x) for x in W7[:6])
    dmax_grav = _alc_a_dmax(float(W7[6]))
    filas = []
    for seed in range(SEEDS):
        for arm in ("sync", "async"):
            base = dict(point_id=pid, w_grav=W[0], w_strong=W[1], w_em=W[2], w_weak=W[3],
                        w_exp=W[4], w_cool=W[5], alc=float(W7[6]), dmax_grav=dmax_grav,
                        seed=seed, arm=arm, phys=phys)
            base.update(corrida(W, dmax_grav, seed * 100003 + pid, arm))
            filas.append(base)
    return filas


def _campos():
    campos = ["point_id", "w_grav", "w_strong", "w_em", "w_weak", "w_exp", "w_cool", "alc", "dmax_grav",
              "seed", "arm", "phys"]
    for c in CLASES:
        campos += [f"viable_{c}", f"estable_{c}", f"expande_{c}", f"acelera_{c}", f"gfin_{c}"]
    return campos


def main():
    print("CS057 — PAISAJE COMPLETO: barrido Sobol de las 6 fuerzas (0→1) + alcance + sync/async", flush=True)
    print("=" * 108, flush=True)
    if SMOKE:
        npts = 8
        print(f"[SMOKE] {npts} puntos × {SEEDS} semillas × 2 brazos × {len(CLASES)} clases — validación+cronometraje", flush=True)
    else:
        npts = N_POINTS
    print(f"puntos={npts} semillas={SEEDS} pasos={STEPS} workers={WORKERS} ensemble={CLASES}", flush=True)
    print(f"tasas base g={R_GRAV} s={R_STRONG} em={R_EM} wk={R_WEAK} exp={R_EXP} · alcances FIJOS (grav 2-6/EM {DMAX_EM}/fuerte-débil vecindad)", flush=True)
    print(f"salida CHECKPOINT: {OUT}", flush=True)

    # ---- construir la lista de puntos: Sobol + punto físico + denso alrededor del físico ----
    sob = _sobol_puntos(npts)
    puntos = [(i, sob[i], 0) for i in range(len(sob))]
    pid = len(puntos)
    puntos.append((pid, _punto_fisico(), 1)); pid += 1
    if not SMOKE:
        for p in _denso_fisico(DIM_DENSO):
            puntos.append((pid, p, 2)); pid += 1                 # phys=2 = vecindad densa del físico
    print(f"total puntos (Sobol + físico + denso) = {len(puntos)}  → corridas ≈ {len(puntos) * SEEDS * 2}", flush=True)

    # ---- reanudar: saltar puntos ya escritos ----
    hechos = set()
    campos = _campos()
    existe = os.path.exists(OUT)
    if existe:
        try:
            with open(OUT) as f:
                for row in csv.DictReader(f):
                    hechos.add(int(row["point_id"]))
            print(f"REANUDANDO: {len(hechos)} puntos ya en el CSV, se saltan.", flush=True)
        except Exception as e:
            print(f"(no se pudo leer checkpoint, empiezo de cero: {e})", flush=True)
    pend = [p for p in puntos if p[0] not in hechos]
    if not pend:
        print("Nada pendiente — el paisaje ya está completo.", flush=True)
        return

    fout = open(OUT, "a", newline="")
    wr = csv.DictWriter(fout, fieldnames=campos)
    if not existe or os.path.getsize(OUT) == 0:
        wr.writeheader(); fout.flush()

    t0 = time.time(); hechos_n = 0
    import multiprocessing as mp
    if WORKERS > 1 and not SMOKE:
        with mp.Pool(WORKERS) as pool:
            for filas in pool.imap_unordered(_worker, pend, chunksize=1):
                for fila in filas:
                    wr.writerow(fila)
                fout.flush()
                hechos_n += 1
                if hechos_n % 25 == 0 or hechos_n == len(pend):
                    dt = time.time() - t0
                    rate = hechos_n / dt
                    eta = (len(pend) - hechos_n) / rate if rate > 0 else 0
                    print(f"  {hechos_n}/{len(pend)} puntos · {dt/60:.1f} min · {rate*60:.1f} pt/min · ETA {eta/3600:.2f} h", flush=True)
    else:
        for p in pend:
            filas = _worker(p)
            for fila in filas:
                wr.writerow(fila)
            fout.flush()
            hechos_n += 1
            dt = time.time() - t0
            print(f"  {hechos_n}/{len(pend)} · {dt:.1f}s · {dt/hechos_n:.2f}s/pt", flush=True)
    fout.close()
    dt = time.time() - t0
    print(f"\nCOMPLETO: {hechos_n} puntos en {dt/60:.1f} min → {OUT}", flush=True)
    if SMOKE:
        cps = (len(pend) * SEEDS * 2)
        print(f"[SMOKE] cronometraje: {dt/max(cps,1):.3f} s/corrida. Para la tanda completa (4096 pts):", flush=True)
        full = 4096 * SEEDS * 2 * (dt / max(cps, 1)) / WORKERS
        print(f"        ~{full/3600:.1f} h con {WORKERS} workers (paralelizado por puntos).", flush=True)


if __name__ == "__main__":
    main()
