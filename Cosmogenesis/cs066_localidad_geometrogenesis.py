"""
CS066 — LA LOCALIDAD PRIMERO: geometrogénesis. ¿Emerge un espacio con "lejos" — y recién ahí, direcciones?
==========================================================================================================
Diseño: CS (DISENO_CS066_localidad_geometrogenesis.md), pre-registrado, endosado por Alexis ("el tejido primero").
Codea/ejecuta: CC. Funda: auditoría de CS sobre los CSV de CS064 — el brazo `completo` es un blob ULTRA-mundo-
pequeño (diam≈3.9 que NO crece con N; d_s se infla 4.8→5.6). Todo está a 3-4 saltos de todo: no hay "lejos".

EL GIRO: CS065/065b buscaron el ingrediente en el nivel de las DIRECCIONES (exclusión) y murieron dos veces.
En un blob sin "lejos", todas las direcciones son la misma — el colapso-a-1 es SÍNTOMA de que no hay espacio
local, no falla de la orientación. Lo que faltó no es un ingrediente que ELIJA direcciones (ya se falsificó),
es la LOCALIDAD: la condición que hace que la DISTANCIA signifique algo. Primero el tejido; después, lo que vive.

EL ACTOR NUEVO (uno solo; la exclusión se retira): COSTO DE NO-LOCALIDAD, EN LA FORMACIÓN (Quantum Graphity).
Adjudicado en el smoke (10-jul): una poda EXTERNA posterior a la formación NO tiene punto fijo de tejido — las
fuerzas rellenan el blob (poda débil) o se pasa a gas (poda fuerte); el tejido es un transitorio inestable. Por
eso la localidad gobierna la PERSISTENCIA del enlace, JUNTO a las fuerzas, no peleando contra ellas después:
cada nodo conserva sus k_local enlaces MÁS locales (mayor soporte = vecinos comunes); los atajos de largo
alcance no persisten. Esto acota la densidad (ni blob ni gas: punto fijo estable) y deja crecer solo el tejido
local. NO se calibra (G-NO-CALIBRAR): el presupuesto k_local se SORTEA en rango amplio por parche; se busca si
EXISTE una transición blob→tejido, no un valor. La orientación/marco co-evoluciona como en CS064 (nemático, SIN
exclusión). k_local BAJO = localidad FUERTE (geometrogénesis marcada); k_local alto → tiende a sin_local (blob).

Reusa: cs064 (motor + jueces), cs057 (_diam/_giant), cg003 (grafo aleatorio caliente). El SMOKE valida
andamiaje (sin_local=blob; el gate da tejido sin gas; diam-vs-N discrimina en calibradores). No se acomoda
ningún desenlace: se leen contra §5. G-TEJIDO-ANTES-QUE-EJES: el Nivel 1 (¿hay espacio?) se adjudica ANTES del
Nivel 2 (¿cuántos ejes?). Contar ejes en un brazo sin tejido local sería contar en un blob.
"""
from __future__ import annotations
import os, sys, csv, math, time
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs057_paisaje_completo as C7
import cs062_paisaje_peso as C62
import cs059_espin_como_marco as C9
import cg003_diagnostico_gromov as GR
import cs064_smoke as SM
import cs064_sistema_completo as C64      # motor de referencia (sin_local == CS064), jueces y catálogo

RNG = np.random.default_rng
DMAX_INT = C64.DMAX_INT                    # 8 (holgado, G-NO-TOPADO)
STEPS   = int(os.environ.get("CS066_STEPS", 20))
N_NODOS = int(os.environ.get("CS066_N", 2500))
PATCHES = int(os.environ.get("CS066_PATCHES", 100))
WORKERS = int(os.environ.get("CS066_WORKERS", max(1, min(6, (os.cpu_count() or 4) - 2))))
OUT     = os.environ.get("CS066_OUT", os.path.join(_HERE, "cs066_localidad.csv"))
SMOKE   = os.environ.get("CS066_SMOKE", "") != ""
KLOC_LO = int(os.environ.get("CS066_KLO", 5))          # presupuesto de grado local, rango AMPLIO (G-NO-CALIBRAR)
KLOC_HI = int(os.environ.get("CS066_KHI", 14))
KFIX    = os.environ.get("CS066_KFIX", "")             # confirmatorio: k_local FIJO (malla), no sorteado
ARMS    = ["local", "sin_local", "local_barajado", "local_marco_congelado"]

def _sample_k(seed):
    """k_local por parche: FIJO si CS066_KFIX (confirmatorio, malla declarada) — si no, SORTEADO (G-NO-CALIBRAR)."""
    return int(KFIX) if KFIX != "" else int(RNG(seed).integers(KLOC_LO, KLOC_HI + 1))


# ============================ EL ACTOR NUEVO: localidad en la formación ============================
def gate_localidad(adj, N, rng, k_local, barajado=False):
    """Geometrogénesis por PERSISTENCIA: cada nodo conserva sus k_local enlaces MÁS locales (mayor soporte =
    nº de vecinos comunes); los atajos de largo alcance (poco soporte) no persisten. Un enlace sobrevive si
    ALGUNO de sus extremos lo conserva (unión → nunca aísla un nodo: anti-gas estructural). Densidad acotada
    ⇒ ni blob ni gas: punto fijo estable de tejido local.
    barajado (placebo): cada nodo conserva k_local enlaces AL AZAR — MISMO tope de grado, pero SIN elegir por
    localidad. Aísla si importa 'quedarse con los locales' (tejido) o solo 'acotar el grado' (podría seguir
    mundo-pequeño). Devuelve nº de enlaces podados."""
    if k_local <= 0:
        return 0
    keep = set()
    for i in range(N):
        nb = list(adj[i])
        if len(nb) <= k_local:                          # bajo presupuesto: se conserva entero (no se toca)
            for j in nb:
                keep.add((i, j) if i < j else (j, i))
            continue
        if barajado:
            sel = rng.choice(len(nb), size=k_local, replace=False)
            chosen = [nb[t] for t in sel]
        else:
            sup = sorted(((len(adj[i] & adj[j]), j) for j in nb), reverse=True)   # soporte local desc.
            chosen = [j for _, j in sup[:k_local]]
        for j in chosen:
            keep.add((i, j) if i < j else (j, i))
    culled = sum(len(a) for a in adj) // 2 - len(keep)
    na = [set() for _ in range(N)]
    for (i, j) in keep:
        na[i].add(j); na[j].add(i)
    for i in range(N):
        adj[i].clear(); adj[i].update(na[i])
    return max(0, culled)


# ============================ EL MOTOR: CS064 + localidad en la formación (exclusión retirada) ============================
def proceso066(N, cat, arm, k_local, rng):
    """Motor de CS064 (4 fuerzas mediadas + aniquilación + co-evolución del marco NEMÁTICO, SIN exclusión)
    + localidad en la formación. arm ∈ {local, sin_local, local_barajado, local_marco_congelado}."""
    fam, color, carga, masa, es_anti = cat["fam"], cat["color"], cat["carga"], cat["masa"], cat["es_anti"]
    gate_on   = arm in ("local", "local_barajado", "local_marco_congelado")
    barajado  = (arm == "local_barajado")
    marco_vivo = (arm != "local_marco_congelado")
    # sin_local == CS064 (sin gate) → G-CONTINUIDAD

    adj0, _ = GR.aleatorio(N, meandeg=6.0, seed=int(rng.integers(1 << 30)))
    adj = [set(a) for a in adj0]
    col = np.where(color >= 0, color, rng.integers(0, 3, N)).astype(np.int8)
    car = (carga > 0).astype(np.int8)
    deg0 = [len(a) for a in adj]
    t = np.zeros(N, dtype=np.int32)
    V = C9._spins(N, DMAX_INT, rng)
    R = getattr(C7, "R_GRAV", 1.0); T_CONF = getattr(C7, "T_CONF", 0.5); CAP_E = 12 * N
    D, G = [], []
    for step in range(STEPS):
        T = C7._T_de_paso(step, 1.0) if hasattr(C7, "_T_de_paso") else max(0.02, 1.6 * (1 - step / STEPS))
        E = sum(len(a) for a in adj) // 2
        if E > CAP_E:
            break
        w = masa * max(0.0, 1.0 - T); CAP = max(1, N // 4)
        try:
            rg = min(0.30 * R * (1 - T), CAP / max(1.0, (0.5 + T) * E))
            C62._grav_peso(adj, N, rng, rg, dmax=3, T=T, w=w + 1e-9)
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
        # >>> EL ACTOR NUEVO: geometrogénesis — la localidad gobierna qué enlaces PERSISTEN (no un podador aparte) <<<
        if gate_on:
            gate_localidad(adj, N, rng, k_local, barajado=barajado)
        # co-evolución del marco NEMÁTICA (como CS064, SIN exclusión) — o congelado
        if marco_vivo:
            A = SM.adj_sparse(adj, N); deg = np.asarray(A.sum(axis=1)).ravel()
            V = SM.alinear_nematico_fast(V, A, deg, mezcla=0.35)
        D.append(C7._diam(adj, N)); G.append(C7._giant(adj, N))
    return adj, V, D, G


# ============================ MÉTRICAS DE NIVEL 1 (¿hay tejido local?) ============================
def grado_medio(adj, N):
    return round(2.0 * sum(len(a) for a in adj) / max(1, N), 3)

def clustering_medio(adj, N, rng, sample=400):
    """Clustering local medio sobre una muestra: alto = tejido con triángulos; bajo = maraña con atajos."""
    idx = rng.integers(0, N, size=min(sample, N))
    vals = []
    for i in idx:
        nb = list(adj[i]); k = len(nb)
        if k < 2:
            vals.append(0.0); continue
        links = 0
        for a_pos in range(k):
            Aa = adj[nb[a_pos]]
            for b_pos in range(a_pos + 1, k):
                if nb[b_pos] in Aa:
                    links += 1
        vals.append(2.0 * links / (k * (k - 1)))
    return round(float(np.mean(vals)) if vals else 0.0, 4)


# ============================ WORKER ============================
def _worker(arg):
    pidx, seed, k_local = arg
    rng = RNG(seed)
    cat = C64._cataloga(N_NODOS, rng)
    filas = []
    for arm in ARMS:
        r2 = RNG(seed * 137 + hash(arm) % 9973 + 5)
        adj, V, D, G = proceso066(N_NODOS, cat, arm, k_local, r2)
        m = C64.juzga(adj, V, N_NODOS, seed * 19 + hash(arm) % 991)
        m.update(dict(patch=pidx, seed=seed, N=N_NODOS, arm=arm,
                      k_local=(k_local if arm != "sin_local" else -1), D_max=DMAX_INT,
                      grado_medio=grado_medio(adj, N_NODOS),
                      clustering=clustering_medio(adj, N_NODOS, RNG(seed * 7 + 3)),
                      diam_fin=(D[-1] if D else -1), pasos=len(D)))
        filas.append(m)
    return filas

def _campos():
    return ["patch", "seed", "N", "arm", "k_local", "D_max", "d_s", "delta_rel", "n_ejes", "PR",
            "ejes_reales", "holonomia", "gigante", "grado_medio", "clustering", "diam_fin", "pasos"]


# ============================ SMOKE: andamiaje antes de la tanda (G-SMOKE-ANTES) ============================
def _lattice3d(N):
    """Calibrador de TEJIDO: retícula cúbica ~L³. diam ~ N^(1/3) (crece). Para validar la medida decisiva."""
    L = max(2, round(N ** (1.0 / 3.0)))
    idx = {}
    for x in range(L):
        for y in range(L):
            for z in range(L):
                idx[(x, y, z)] = len(idx)
    adj = [set() for _ in range(len(idx))]
    for (x, y, z), i in idx.items():
        for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            nb = (x + dx, y + dy, z + dz)
            if nb in idx:
                j = idx[nb]; adj[i].add(j); adj[j].add(i)
    return adj, len(idx)

def _smoke():
    rng = RNG(66066)
    print("=" * 96, flush=True)
    print("CS066 SMOKE — andamiaje. (i) sin_local=blob  (ii) el gate da tejido sin gas  (iii) diam-vs-N discrimina", flush=True)
    print("=" * 96, flush=True)
    print("\n[iii] CALIBRADORES del juez diam (¿discrimina tejido de blob?):", flush=True)
    for Nc in (1000, 3000):
        aL, nL = _lattice3d(Nc)
        dL = C7._diam(aL, nL)
        aR = [set(a) for a in GR.aleatorio(Nc, meandeg=6.0, seed=int(rng.integers(1 << 30)))[0]]
        dR = C7._diam(aR, Nc)
        print(f"   N≈{Nc:>4}: retícula-3D diam={dL:.2f} (esperado ~N^(1/3)≈{Nc**(1/3):.1f})  |  "
              f"aleatorio diam={dR:.2f} (esperado ~log N≈{math.log(Nc):.1f})", flush=True)
    Nn = int(os.environ.get("CS066_N", 1000))
    npar = 4
    print(f"\n[i,ii] {npar} parches × 4 brazos a N={Nn} (k_local sorteado por parche):", flush=True)
    hdr = f"   {'arm':<24}{'diam':>7}{'d_s':>7}{'grado':>7}{'clust':>8}{'gigante':>9}{'n_ejes':>7}"
    for pidx in range(npar):
        seed = 66000 + 7 * pidx
        k_local = int(RNG(seed).integers(KLOC_LO, KLOC_HI + 1))
        print(f"  --- parche {pidx} · k_local={k_local} ---\n{hdr}", flush=True)
        for f in _worker_smoke(pidx, seed, k_local, Nn):
            print(f"   {f['arm']:<24}{f['diam_fin']:>7.2f}{f['d_s']:>7.2f}{f['grado_medio']:>7.2f}"
                  f"{f['clustering']:>8.3f}{f['gigante']:>9.3f}{f['n_ejes']:>7d}", flush=True)
    print("\nSMOKE listo. Adjudica CS antes de la tanda: (i) sin_local≈blob (diam~3-4, grado alto, ~1.2 ejes) ; "
          "(ii) local con k_local bajo debe dar diam MAYOR y gigante SANO (no gas) ; comparar local vs barajado. "
          "NO correr el barrido hasta el visto bueno. — CC", flush=True)

def _worker_smoke(pidx, seed, k_local, Nn):
    global N_NODOS
    N_NODOS = Nn
    return _worker((pidx, seed, k_local))


# ============================ MAIN ============================
def main():
    if SMOKE:
        _smoke(); return
    npatch = PATCHES
    print("=" * 96, flush=True)
    print("CS066 — LOCALIDAD PRIMERO (geometrogénesis en la formación). PRE-REGISTRADO. ¿emerge tejido con 'lejos'?", flush=True)
    print("=" * 96, flush=True)
    kdesc = f"k_local={KFIX} FIJO (confirmatorio/malla)" if KFIX != "" else f"k_local∈[{KLOC_LO},{KLOC_HI}] SORTEADO (G-NO-CALIBRAR)"
    print(f"N={N_NODOS} · patches={npatch} · steps={STEPS} · D_max={DMAX_INT} (G-NO-TOPADO) · "
          f"{kdesc} · arms={ARMS}", flush=True)
    print("PRE-INSCRITO §5 (se lee contra esto, NO se acomoda): "
          "(A) diam CRECE~N^(1/d) Y d_s se estabiliza Y n_ejes≥2 Y local>local_barajado → LOCALIDAD era lo que faltaba ; "
          "(B) diam crece PERO sigue colapsando a 1 eje → espacio y direcciones son problemas SEPARADOS ; "
          "(C) diam sigue plano → no hay geometrogénesis en este sustrato ; "
          "(D) placebo: local≈local_barajado → solo acotar grado ; (E) depende de N. "
          "NIVEL 1 (¿hay tejido? diam-vs-N) SE ADJUDICA ANTES QUE NIVEL 2 (¿cuántos ejes?). G-TEJIDO-ANTES-QUE-EJES.", flush=True)
    hechos = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for row in csv.DictReader(f): hechos.add(int(row["patch"]))
        print(f"REANUDANDO: {len(hechos)} parches ya hechos", flush=True)
    pend = [(p, 66100 + 3 * p, _sample_k(66100 + 3 * p)) for p in range(npatch) if p not in hechos]
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
                if n % 5 == 0 or n == len(pend):
                    dt = time.time() - t0; r = n / dt
                    print(f"  {n}/{len(pend)} parches · {dt/60:.1f}min · ETA {(len(pend)-n)/r/3600:.2f}h", flush=True)
    else:
        for arg in pend:
            filas = _worker(arg)
            for fila in filas: wr.writerow(fila)
            fout.flush(); n += 1
            dd = {f["arm"]: (f["diam_fin"], f["n_ejes"]) for f in filas}
            print(f"  parche {n}/{len(pend)} · {time.time()-t0:.1f}s · k_local={arg[2]} · (diam,n_ejes)={dd}", flush=True)
    fout.close()
    print(f"\nCOMPLETO: {n} parches en {(time.time()-t0)/60:.1f} min → {OUT}", flush=True)
    print("RECORDATORIO: NIVEL 1 primero (¿diam crece con N?). No leer ejes de un brazo sin tejido. — CC", flush=True)

if __name__ == "__main__":
    main()
