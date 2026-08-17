"""
CS065 — EL INGREDIENTE ANTI-COLAPSO REAL: exclusión de Pauli. ¿Sostiene varios ejes sin fundirlos?
====================================================================================================
Diseño: CS (DISENO_CS065_exclusion_pauli_anticolapso.md). Codea/ejecuta: CC. Endosa: Alexis.
CS064 dio B': la inercia-de-la-mayoría colapsa a 1 eje (73-81% de parches). El experimento nombró su
faltante: algo que impida que todo apunte igual. La ÚNICA cosa que cumple la regla de oro (real + omitida +
no-calibrada) es la EXCLUSIÓN DE PAULI: dos fermiones no pueden ocupar el mismo estado. Quarks y leptones son
fermiones; la omitimos. Se añade UN término a la co-evolución del marco: repulsión de orientación entre
fermiones vecinos (empuja a ortogonal). Los bosones NO la sienten (discriminante interno).

REGLA DE ORO (G-NO-CALIBRAR): la fuerza de la exclusión NO se fija mirando el resultado — se SORTEA en rango
amplio por parche. El éxito NO es que salga 3; es que el test DISCRIMINE (excl vs sus controles).
Reusa el motor de CS064 SIN tocarlo (mismas 4 fuerzas mediadas, expansión, enfriamiento, jueces).
"""
from __future__ import annotations
import os, sys, csv, math, time
import numpy as np
import scipy.sparse as sp

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs057_paisaje_completo as C7
import cs062_paisaje_peso as C62
import cs059_espin_como_marco as C9
import cg003_diagnostico_gromov as GR
import cs064_smoke as SM
import cs064_sistema_completo as C64          # reuso juzga (mismos jueces) y constantes

RNG = np.random.default_rng
DMAX_INT = C64.DMAX_INT
STEPS   = int(os.environ.get("CS065_STEPS", 20))
N_NODOS = int(os.environ.get("CS065_N", 2500))
PATCHES = int(os.environ.get("CS065_PATCHES", 100))
WORKERS = int(os.environ.get("CS065_WORKERS", max(1, min(6, (os.cpu_count() or 4) - 2))))
OUT     = os.environ.get("CS065_OUT", os.path.join(_HERE, "cs065_exclusion.csv"))
SMOKE   = os.environ.get("CS065_SMOKE", "") != ""
ARMS    = ["excl", "sin_excl", "excl_barajada", "excl_bosones", "marco_congelado"]

def _cataloga065(N, rng):
    """Catálogo con FRACCIÓN DE FERMIONES SORTEADA por parche (para leer nº-ejes ↔ fracción-fermiones)."""
    frac_ferm = rng.uniform(0.5, 0.95)
    es_ferm = rng.random(N) < frac_ferm
    fam = np.where(es_ferm, rng.integers(0, 3, N), 3).astype(np.int8)   # 0/1/2=fermión, 3=bosón(mediador)
    color = np.where(fam == 0, rng.integers(0, 3, N), -1).astype(np.int8)
    carga = np.zeros(N, np.int8)
    carga[fam == 0] = rng.choice([-1, 1, 2, -2], size=int((fam == 0).sum()))
    carga[fam == 1] = rng.choice([-3, 3], size=int((fam == 1).sum()))
    masa = np.exp(rng.uniform(math.log(1.0), math.log(3477.0), N)); masa[fam == 2] *= 1e-6
    es_anti = (rng.random(N) < 0.5).astype(np.int8)
    return dict(fam=fam, color=color, carga=carga, masa=masa, es_anti=es_anti,
                es_ferm=es_ferm, frac_ferm=frac_ferm)

def _mask_diag(mask):
    return sp.diags(mask.astype(float))

def proceso065(N, cat, arm, lam, rng):
    """Motor COMPLETO de CS064 (todos los brazos) + exclusión en la co-evolución del marco según 'arm'."""
    fam, color, carga, masa, es_anti = cat["fam"], cat["color"], cat["carga"], cat["masa"], cat["es_anti"]
    es_ferm = cat["es_ferm"]
    marco_vivo = (arm != "marco_congelado")
    excl_on = arm in ("excl", "excl_barajada", "excl_bosones")
    # máscara de quién ejerce/siente exclusión
    if arm == "excl":
        emask = es_ferm.copy()
    elif arm == "excl_bosones":
        emask = (fam == 3)                                 # placebo físicamente falso
    elif arm == "excl_barajada":
        emask = es_ferm.copy(); rng.shuffle(emask)         # misma cantidad, ubicación al azar
    else:
        emask = np.zeros(N, bool)
    Dm = _mask_diag(emask) if excl_on else None

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
        w = masa * max(0.0, 1.0 - T)
        CAP = max(1, N // 4)
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
        # co-evolución del marco: inercia (sin_excl) o inercia+exclusión (excl*)
        if marco_vivo:
            A = SM.adj_sparse(adj, N)
            if excl_on:
                A_excl = Dm @ A @ Dm                       # sub-adyacencia entre los que excluyen
                V = SM.alinear_excl_fast(V, A, A_excl, mezcla=0.35, lam=lam)
            else:
                deg = np.asarray(A.sum(axis=1)).ravel()
                V = SM.alinear_nematico_fast(V, A, deg, mezcla=0.35)
        D.append(C7._diam(adj, N)); G.append(C7._giant(adj, N))
    return adj, V, D, G

def _worker(arg):
    pidx, seed = arg
    rng = RNG(seed)
    cat = _cataloga065(N_NODOS, rng)
    lam = float(np.exp(rng.uniform(math.log(0.15), math.log(1.2))))   # SORTEADA en rango amplio (G-NO-CALIBRAR)
    filas = []
    for arm in ARMS:
        r2 = RNG(seed * 131 + hash(arm) % 9973 + 7)
        adj, V, D, G = proceso065(N_NODOS, cat, arm, lam, r2)
        m = C64.juzga(adj, V, N_NODOS, seed * 17 + hash(arm) % 991)
        m.update(dict(patch=pidx, seed=seed, N=N_NODOS, arm=arm, lam=round(lam, 3),
                      frac_ferm=round(cat["frac_ferm"], 3), diam_fin=(D[-1] if D else -1), pasos=len(D)))
        filas.append(m)
    return filas

def _campos():
    return ["patch", "seed", "N", "arm", "lam", "frac_ferm", "d_s", "delta_rel", "n_ejes", "PR",
            "ejes_reales", "holonomia", "gigante", "diam_fin", "pasos"]

def main():
    npatch = 8 if SMOKE else PATCHES
    print("=" * 96, flush=True)
    print("CS065 — EXCLUSIÓN DE PAULI como anti-colapso. ¿rompe el B' de CS064 y sostiene varios ejes?", flush=True)
    print("=" * 96, flush=True)
    print(f"N={N_NODOS} · patches={npatch} · steps={STEPS} · arms={ARMS} · workers={WORKERS}", flush=True)
    print(f"salida {OUT} · lam SORTEADA en [0.15,1.2] por parche (G-NO-CALIBRAR)", flush=True)
    print("PRE-INSCRITO §6: (A) excl>>sin_excl y correlaciona con frac_ferm ; (B) abre ejes pero no 3 ; "
          "(C) excl≈sin_excl (colapso definitivo) ; (D) placebo excl≈excl_bosones ; (E) depende de N. "
          "ÉXITO = que DISCRIMINE, no que salga 3. G-CONTINUIDAD: sin_excl≈CS064 (~1.2 ejes); congelado→0.", flush=True)
    hechos = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for row in csv.DictReader(f): hechos.add(int(row["patch"]))
        print(f"REANUDANDO: {len(hechos)} parches ya hechos", flush=True)
    pend = [(p, 2065 * p + 11) for p in range(npatch) if p not in hechos]
    if not pend:
        print("nada pendiente.", flush=True); return
    fout = open(OUT, "a", newline=""); wr = csv.DictWriter(fout, fieldnames=_campos())
    if not hechos: wr.writeheader()
    t0 = time.time(); n = 0
    import multiprocessing as mp
    if WORKERS > 1 and not SMOKE:
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
            ej = {f["arm"]: f["n_ejes"] for f in filas}
            print(f"  parche {n}/{len(pend)} · {time.time()-t0:.1f}s · lam={filas[0]['lam']} fferm={filas[0]['frac_ferm']} · n_ejes={ej}", flush=True)
    fout.close()
    print(f"\nCOMPLETO: {n} parches en {(time.time()-t0)/60:.1f} min → {OUT}", flush=True)

if __name__ == "__main__":
    main()
