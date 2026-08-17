"""
CS065b — EXCLUSIÓN ORTOGONALIZANTE (Pauli FIEL). ¿sostiene varios ejes, o el negativo se cierra?
=================================================================================================
Diseño: CS (DISENO_CS065b_exclusion_ortogonalizante.md), pre-registrado, endosado por Alexis (10-jul-2026).
CS065 (repulsión lineal SIN freno) dio negativo: destruye dirección, igual sobre fermiones que sobre pares al
azar (excl≈barajada) → empuja a ISOTROPÍA, no a ortogonalidad. Diagnóstico (antes del veredicto completo): la
traducción FIEL de Pauli es ortogonalización SATURANTE (Gram-Schmidt: hazlo ortogonal y PARA), no resta ilimitada.
CS065b mete esa corrección de fidelidad. Mismos 5 brazos, mismo motor, mismo N. La cuerda decisiva:
excl_orto vs excl_orto_barajada — si NO se separan, la exclusión muere limpia (no hay CS065c).
G-NO-TOPADO: D_max=8 holgado; hay que verificar que n_ejes < D_max (si pega el techo, es artefacto, no física).
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
import cs064_sistema_completo as C64
import cs065_exclusion_pauli as C65      # reuso catálogo (fracción de fermiones sorteada) y estructura

RNG = np.random.default_rng
DMAX_INT = C64.DMAX_INT                   # 8 (holgado, G-NO-TOPADO)
STEPS   = int(os.environ.get("CS065B_STEPS", 20))
N_NODOS = int(os.environ.get("CS065B_N", 2500))
PATCHES = int(os.environ.get("CS065B_PATCHES", 100))
WORKERS = int(os.environ.get("CS065B_WORKERS", max(1, min(6, (os.cpu_count() or 4) - 2))))
OUT     = os.environ.get("CS065B_OUT", os.path.join(_HERE, "cs065b_exclusion_orto.csv"))
SMOKE   = os.environ.get("CS065B_SMOKE", "") != ""
ARMS    = ["excl_orto", "sin_excl", "excl_orto_barajada", "excl_orto_bosones", "marco_congelado"]

def proceso065b(N, cat, arm, rng):
    """Motor de CS064/CS065 + exclusión ORTOGONALIZANTE (saturante) en la co-evolución del marco."""
    fam, color, carga, masa, es_anti = cat["fam"], cat["color"], cat["carga"], cat["masa"], cat["es_anti"]
    es_ferm = cat["es_ferm"]
    marco_vivo = (arm != "marco_congelado")
    excl_on = arm in ("excl_orto", "excl_orto_barajada", "excl_orto_bosones")
    if arm == "excl_orto":
        emask = es_ferm.copy()
    elif arm == "excl_orto_bosones":
        emask = (fam == 3)
    elif arm == "excl_orto_barajada":
        emask = es_ferm.copy(); rng.shuffle(emask)
    else:
        emask = np.zeros(N, bool)
    Dm = C65._mask_diag(emask) if excl_on else None

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
        if marco_vivo:
            A = SM.adj_sparse(adj, N)
            if excl_on:
                A_excl = Dm @ A @ Dm
                V = SM.alinear_orto_fast(V, A, A_excl, mezcla=0.35)     # <<< ORTOGONALIZANTE (la corrección)
            else:
                deg = np.asarray(A.sum(axis=1)).ravel()
                V = SM.alinear_nematico_fast(V, A, deg, mezcla=0.35)
        D.append(C7._diam(adj, N)); G.append(C7._giant(adj, N))
    return adj, V, D, G

def _worker(arg):
    pidx, seed = arg
    rng = RNG(seed)
    cat = C65._cataloga065(N_NODOS, rng)
    filas = []
    for arm in ARMS:
        r2 = RNG(seed * 137 + hash(arm) % 9973 + 5)
        adj, V, D, G = proceso065b(N_NODOS, cat, arm, r2)
        m = C64.juzga(adj, V, N_NODOS, seed * 19 + hash(arm) % 991)
        m.update(dict(patch=pidx, seed=seed, N=N_NODOS, arm=arm, D_max=DMAX_INT,
                      frac_ferm=round(cat["frac_ferm"], 3), diam_fin=(D[-1] if D else -1), pasos=len(D)))
        filas.append(m)
    return filas

def _campos():
    return ["patch", "seed", "N", "arm", "D_max", "frac_ferm", "d_s", "delta_rel", "n_ejes", "PR",
            "ejes_reales", "holonomia", "gigante", "diam_fin", "pasos"]

def main():
    npatch = 8 if SMOKE else PATCHES
    print("=" * 96, flush=True)
    print("CS065b — EXCLUSIÓN ORTOGONALIZANTE (Pauli fiel). PRE-REGISTRADO. ¿rompe B' con especificidad?", flush=True)
    print("=" * 96, flush=True)
    print(f"N={N_NODOS} · patches={npatch} · steps={STEPS} · D_max={DMAX_INT} (G-NO-TOPADO) · arms={ARMS}", flush=True)
    print("PRE-INSCRITO §5: (A) excl_orto>>sin_excl Y >barajada Y bosones≈sin_excl Y corr(frac_ferm)>0 ; "
          "(B) abre ejes pero no 3 ; (C) excl_orto≈sin_excl O ≈barajada → EXCLUSIÓN MUERE (sin duelo) ; (D) depende de N. "
          "DECISIVO: excl_orto vs barajada. ÉXITO=discriminar, no que salga 3. Verificar n_ejes < D_max.", flush=True)
    hechos = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for row in csv.DictReader(f): hechos.add(int(row["patch"]))
        print(f"REANUDANDO: {len(hechos)} parches ya hechos", flush=True)
    pend = [(p, 20650 + 3 * p) for p in range(npatch) if p not in hechos]
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
            print(f"  parche {n}/{len(pend)} · {time.time()-t0:.1f}s · fferm={filas[0]['frac_ferm']} · n_ejes={ej}", flush=True)
    fout.close()
    print(f"\nCOMPLETO: {n} parches en {(time.time()-t0)/60:.1f} min → {OUT}", flush=True)

if __name__ == "__main__":
    main()
