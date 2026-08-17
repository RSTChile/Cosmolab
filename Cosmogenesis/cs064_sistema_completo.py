"""
CS064 — EL SISTEMA COMPLETO A LA VEZ: ¿emerge la geometría (y la DIRECCIÓN) de la relación plena?
==================================================================================================
Diseño: CS (DISENO_CS064_sistema_completo_emergencia.md v2). Codea/ejecuta: CC. Hipótesis: Alexis.
Reencuadre 9-jul (Alexis): EL ESPACIO ES UNA EXAPTACIÓN. El negativo del arco (nada SELECCIONA la dimensión)
es la FIRMA, no el fracaso: una exaptación no se selecciona para su función nueva. El brazo `null_marco` es el
TEST (O-N8.3): si congelar la orientación MATA la dirección, la dirección era REÚSO de algo que persistía por
otra razón (la inercia) → exaptación, con sus partes nombrables. Si aparece igual congelada, era primitivo.

Arranca desde una SOPA CALIENTE sin estructura (no una retícula — eso inyectaría el espacio). Cataloga las
partículas del Modelo Estándar con propiedades intrínsecas fijas (G-INTRÍNSECO), enfría+expande, deja que las
cuatro fuerzas (mediadas) liguen y que la aniquilación/decaimiento desarme, y MIDE sin coordenadas
(G-SIN-COORDENADAS): dimensión espectral d_s, planitud δ de Gromov, y DIRECCIÓN EMERGENTE (nº de ejes que la
inercia deja en pie + holonomía). Cinco brazos NULL. Reanudable por parche (checkpoint como CS062).

Reusa: cs057 (motor de fuerzas + constantes), cs062 (gravedad Newton con masa), cs059 (transporte/holonomía),
cs064_smoke (jueces ya calibrados: d_s, tensor de orientación, conteo de ejes, reglas de alineamiento).
Contrato: el SMOKE valida andamiaje; esto corre la tanda. No se acomoda ningún desenlace: se leen contra §8.
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
import cs064_smoke as SM        # jueces calibrados + reglas de alineamiento

RNG = np.random.default_rng

# ============================ CONFIG (todo sorteable; nada fija 3D) ============================
N_NODOS  = int(os.environ.get("CS064_N", 10000))          # nodos por parche (barrer para (D): emergencia colectiva)
PATCHES  = int(os.environ.get("CS064_PATCHES", 200))       # 'números enormes' = muchos intentos independientes
STEPS    = int(os.environ.get("CS064_STEPS", 20))
DMAX_INT = int(os.environ.get("CS064_DMAX", 8))            # techo de componentes del marco (NO el nº de ejes)
REGLA    = os.environ.get("CS064_REGLA", "nematico")       # nematico (permite varios ejes) | polar | both
WORKERS  = int(os.environ.get("CS064_WORKERS", max(1, min(8, (os.cpu_count() or 4) - 2))))  # conservador (no tumbar Docker)
OUT      = os.environ.get("CS064_OUT", os.path.join(_HERE, "cs064_sistema_completo.csv"))
SMOKE    = os.environ.get("CS064_SMOKE", "") != ""
ARMS     = ["completo", "null_tipos", "null_marco", "null_mediado", "subconjunto"]

# catálogo del Modelo Estándar (proporciones ~ realistas, sorteadas en rango, no calibradas)
# familia: 0=quark 1=leptón 2=neutrino 3=mediador ; se porta color, carga, masa (log), es_anti
def _cataloga(N, rng):
    fam = rng.choice([0, 1, 2, 3], size=N, p=[0.55, 0.18, 0.20, 0.07])
    color = np.where(fam == 0, rng.integers(0, 3, N), -1).astype(np.int8)      # solo quarks
    carga = np.zeros(N, np.int8)
    carga[fam == 0] = rng.choice([-1, 1, 2, -2], size=(fam == 0).sum())         # ±1/3,±2/3 (×3 para enteros)
    carga[fam == 1] = rng.choice([-3, 3], size=(fam == 1).sum())                # ±1
    # neutrinos y mediadores: carga 0 (fotón/Z/Higgs) — W lo tratamos como transmutador en débil
    masa = np.exp(rng.uniform(math.log(1.0), math.log(3477.0), N))              # rango leptónico log
    masa[fam == 2] *= 1e-6                                                      # neutrinos ~ 0
    es_anti = (rng.random(N) < 0.5).astype(np.int8)
    # asimetría materia-antimateria minúscula SORTEADA (no elegida): baryogénesis ~1 en 10^9 → aquí pequeña
    asim = rng.uniform(0.0, 0.02)
    es_anti[(rng.random(N) < asim)] = 0
    return dict(fam=fam, color=color, carga=carga, masa=masa, es_anti=es_anti)

# ============================ EL MOTOR — un tick, todos sobre todos a la vez ============================
def proceso064(N, cat, arm, regla, rng):
    """Sopa caliente → enfría+expande → 4 fuerzas mediadas + aniquilación + co-evolución del marco.
    arm ∈ {completo, null_tipos, null_marco, null_mediado, subconjunto}. Devuelve trayectoria + estado final."""
    fam, color, carga, masa, es_anti = (cat["fam"].copy(), cat["color"].copy(), cat["carga"].copy(),
                                        cat["masa"].copy(), cat["es_anti"].copy())
    if arm == "null_tipos":                       # rompe la ESTRUCTURA del catálogo, deja la estadística
        for a in (fam, color, carga, es_anti):
            rng.shuffle(a)
        rng.shuffle(masa)
    mediado = (arm != "null_mediado")             # fuerzas como relación mediada (dist>=2) vs instantánea
    marco_vivo = (arm == "completo" or arm == "null_tipos" or arm == "null_mediado")  # co-evoluciona salvo null_marco/subconjunto
    subconj = (arm == "subconjunto")              # ≈ el arco: solo 4 fuerzas, sin anti/mediador/marco

    # sopa inicial: DENSA y caliente (muchas colisiones; nada cuaja aún). Aleatoria = sin geometría inyectada.
    adj0, _ = GR.aleatorio(N, meandeg=6.0, seed=int(rng.integers(1 << 30)))
    adj = [set(a) for a in adj0]
    col = np.where(color >= 0, color, rng.integers(0, 3, N)).astype(np.int8)
    car = ((carga > 0).astype(np.int8))
    deg0 = [len(a) for a in adj]
    t = np.zeros(N, dtype=np.int32)
    V = C9._spins(N, DMAX_INT, rng)               # marco: abanico de TODAS las orientaciones posibles
    alinear_fast = SM.alinear_nematico_fast if regla == "nematico" else SM.alinear_polar_fast
    R = getattr(C7, "R_GRAV", 1.0)
    T_CONF = getattr(C7, "T_CONF", 0.5)
    CAP_E = 12 * N
    D, G = [], []
    for step in range(STEPS):
        T = C7._T_de_paso(step, 1.0) if hasattr(C7, "_T_de_paso") else max(0.02, 1.6 * (1 - step / STEPS))
        E = sum(len(a) for a in adj) // 2
        if E > CAP_E:                             # BLOB denso → corte neutral
            break
        # masa efectiva: a T alta el Higgs no cuajó → masa≈0; al enfriar aparece
        w = masa * max(0.0, 1.0 - T)              # masa efectiva ∝ (1-T)
        dmax_g = 3 if mediado else 1              # mediado: liga a dist>=2 (por un 'entre'); instantáneo: directo
        CAP = max(1, N // 4)                       # tope de EVENTOS por paso (acota costo a ~O(N); no hay ligados ilimitados/tick)
        try:
            rg = min(0.30 * R * (1 - T), CAP / max(1.0, (0.5 + T) * E))     # gravedad: nadd ≤ CAP
            C62._grav_peso(adj, N, rng, rg, dmax=max(2, dmax_g), T=T, w=w + 1e-9)
        except Exception:
            pass
        if T < T_CONF:                            # confinamiento: EMERGE del enfriado (no se fija cuándo)
            try: C7._confin(adj, N, col, t, rng, min(0.8, CAP / max(1.0, E)))   # nc ≤ CAP
            except Exception: pass
        try: C7._em(adj, N, car, deg0, rng, 0.12)               # EM por carga
        except Exception: pass
        if not subconj:
            try: C7._debil(N, col, car, rng, 0.05)              # débil: W transmuta sabor (color/carga)
            except Exception: pass
        try: C7._despliegue(adj, N, rng, 0.14 * T)              # expansión: FUERTE en caliente, →0 al enfriar
        except Exception: pass
        # aniquilación/decaimiento: población NO conservada; fuerte solo en caliente (salvo subconjunto ≈ arco)
        if not subconj and T > 0.4 and step % 2 == 0:
            npar = max(1, N // 60)
            cand = rng.choice(N, size=npar, replace=False)
            for i in cand:
                if adj[i] and es_anti[i] != es_anti[list(adj[i])[0]]:   # par partícula-antipartícula → aniquila
                    j = list(adj[i])[0]; adj[i].discard(j); adj[j].discard(i)
        # co-evolución del marco por inercia (completo) — o congelado (null_marco/subconjunto). VECTORIZADO.
        if marco_vivo:
            A = SM.adj_sparse(adj, N); deg = np.asarray(A.sum(axis=1)).ravel()
            V = alinear_fast(V, A, deg, mezcla=0.35)
        D.append(C7._diam(adj, N)); G.append(C7._giant(adj, N))
    return adj, V, D, G

# ============================ JUEZ (los tres, sobre el estado final) ============================
def juzga(adj, V, N, seed):
    ds = SM.dim_volumen(adj, N, rng=RNG(seed + 1))     # dimensión por crecimiento de bola (robusta en emergentes)
    dm, drel = SM.delta_gromov(adj, N, seed=seed + 2)
    ev = SM.tensor_orientacion(V)
    n_ejes, PR, real, gap, ratio = SM.cuenta_ejes(ev)
    # holonomía del marco final (¿consistencia global? reuso cs059)
    try:
        # usar las 3-4 componentes dominantes como 'spins' efectivos para la holonomía
        K = min(4, DMAX_INT)
        spins = V[:, :K] / (np.linalg.norm(V[:, :K], axis=1, keepdims=True) + 1e-12)
        hol, ncic, _ = C9._frame_burgers(adj, N, spins, K, RNG(seed + 3), null=False)
    except Exception:
        hol, ncic = float("nan"), 0
    gig = float(C7._giant(adj, N))
    return dict(d_s=round(ds, 3), delta_rel=round(float(drel), 3), n_ejes=int(n_ejes),
                PR=round(float(PR), 3), ejes_reales=int(real), holonomia=round(float(hol), 3),
                gigante=round(gig, 3))

# ============================ WORKER: un parche → filas (una por brazo × regla) ============================
def _worker(arg):
    pidx, seed, reglas = arg
    rng = RNG(seed)
    cat = _cataloga(N_NODOS, rng)
    filas = []
    for arm in ARMS:
        for regla in reglas:
            r2 = RNG(seed * 131 + hash(arm + regla) % 9973 + 7)
            adj, V, D, G = proceso064(N_NODOS, cat, arm, regla, r2)
            m = juzga(adj, V, N_NODOS, seed * 17 + hash(arm + regla) % 991)
            m.update(dict(patch=pidx, seed=seed, N=N_NODOS, arm=arm, regla=regla,
                          diam_fin=(D[-1] if D else -1), pasos=len(D)))
            filas.append(m)
    return filas

def _campos():
    return ["patch", "seed", "N", "arm", "regla", "d_s", "delta_rel", "n_ejes", "PR",
            "ejes_reales", "holonomia", "gigante", "diam_fin", "pasos"]

def main():
    reglas = ["polar", "nematico"] if REGLA == "both" else [REGLA]
    npatch = 8 if SMOKE else PATCHES
    print("=" * 96, flush=True)
    print("CS064 — SISTEMA COMPLETO. ¿emerge geometría+dirección de la relación plena? (el espacio como exaptación)", flush=True)
    print("=" * 96, flush=True)
    print(f"N={N_NODOS} · patches={npatch} · steps={STEPS} · arms={ARMS} · reglas={reglas} · workers={WORKERS}", flush=True)
    print(f"salida {OUT}", flush=True)
    print("PRE-INSCRITO (§8): (A) 3 ejes+d_s≈3+δ≈0 y completo>NULL>subconjunto ⇒ 3D-plano EMERGE ; "
          "(B) dirección sí, 3D no ; (B') colapso a 1 eje ; (C) completo≈subconjunto≈NULL (negativo se sostiene) ; "
          "(D) depende de N. null_marco = test de EXAPTACIÓN (O-N8.3).", flush=True)

    hechos = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for row in csv.DictReader(f):
                hechos.add(int(row["patch"]))
        print(f"REANUDANDO: {len(hechos)} parches ya hechos", flush=True)
    pend = [(p, 2064 * p + 13, reglas) for p in range(npatch) if p not in hechos]
    if not pend:
        print("nada pendiente.", flush=True); return
    campos = _campos()
    fout = open(OUT, "a", newline=""); wr = csv.DictWriter(fout, fieldnames=campos)
    if not hechos:
        wr.writeheader()
    t0 = time.time(); n = 0
    import multiprocessing as mp
    if WORKERS > 1 and not SMOKE:
        with mp.Pool(WORKERS) as pool:
            for filas in pool.imap_unordered(_worker, pend, chunksize=1):
                for fila in filas: wr.writerow(fila)
                fout.flush(); n += 1
                if n % 5 == 0 or n == len(pend):
                    dt = time.time() - t0; r = n / dt
                    print(f"  {n}/{len(pend)} parches · {dt/60:.1f}min · {r*60:.1f}/min · ETA {(len(pend)-n)/r/3600:.2f}h", flush=True)
    else:
        for arg in pend:
            filas = _worker(arg)
            for fila in filas: wr.writerow(fila)
            fout.flush(); n += 1
            ej = {f["arm"]: (f["n_ejes"], f["gigante"], f["d_s"]) for f in filas if f["regla"] == reglas[0]}
            print(f"  parche {n}/{len(pend)} · {time.time()-t0:.1f}s · (n_ejes,gig,d_s) por brazo: {ej}", flush=True)
    fout.close()
    print(f"\nCOMPLETO: {n} parches en {(time.time()-t0)/60:.1f} min → {OUT}", flush=True)
    print("RECORDATORIO: leer contra §8; no acomodar. El null_marco dice si el espacio fue exaptación. — CC", flush=True)

if __name__ == "__main__":
    main()
