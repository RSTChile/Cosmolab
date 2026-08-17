"""
CS060 — Misión A: LOS TRES LEPTONES = el marco SIN ligadura, a tres inercias reales (e/μ/τ). ¿La MASA
(inercia+persistencia del marco) cambia qué geometría se selecciona?
=====================================================================================================
Leptón = marco (espín, CS059) SIN color (no se liga por confinamiento — G-LEPTÓN-SIN-COLOR). La masa entra por
lo que HACE físicamente (no como número mágico): dos ejes contrastables (§2 del diseño):
  · INERCIA de orientación (↑ con masa): resistencia a reorientar el marco. Electrón cede, tauón casi no.
  · PERSISTENCIA temporal (↓ con masa): vida antes de decaer. Electrón persiste, tauón decae casi instantáneo.
Razones reales FIJAS 1:207:3477 (G-MASA-FÍSICA-FIJA). Juez = holonomía del marco (CS059) CON EL CONTROL DE
LONGITUD DE CICLO (la lección de CS059: sin ese control, el confound de long. de ciclo simula selección).

BRAZOS: adyacencia(sin marco) · marco(CS059, espín aleatorio fijo) · electron/muon/tauon(marco+inercia+
persistencia) · alineado(inercia→0, control) · null(masas barajadas). DIMS: d2/d3/d4/curv.
PREGUNTA (G2 vs G1): ¿la escala de masa cambia qué dim tiene marcos más consistentes, CONTROLADO por long. de
ciclo? Éxito ≠ "salió 3D": selección consistente que colapsa bajo NULL. Reusa CS059. numpy + multiprocessing.
"""
from __future__ import annotations
import os, sys, csv, math, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_s = open(os.path.join(_HERE, "cs059_espin_como_marco.py")).read().replace('\nif __name__ == "__main__":\n    main()\n', "\n")
_C9 = {"__file__": os.path.join(_HERE, "cs059_espin_como_marco.py"), "__name__": "cs059_mod"}
exec(compile(_s, "cs059_espin_como_marco.py", "exec"), _C9)
_spins = _C9["_spins"]; _holonomia_ciclo = _C9["_holonomia_ciclo"]; _ciclos_fundamentales = _C9["_ciclos_fundamentales"]
_ensemble = _C9["_ensemble"]

# ============================ CONFIG ============================
K_SWEEP  = [3, 4]                        # K=2 es trivial (CS059); se omite
SEEDS    = int(os.environ.get("CS060A_SEEDS", 10))
N_CICLOS = int(os.environ.get("CS060A_CICLOS", 500))
PASOS    = 10                            # pasos de relajación del marco
WORKERS  = int(os.environ.get("CS060A_WORKERS", max(1, (os.cpu_count() or 4) - 2)))
OUT      = os.environ.get("CS060A_OUT", os.path.join(_HERE, "cs060_leptones.csv"))
SMOKE    = os.environ.get("CS060A_SMOKE", "") != ""
DIMS = ["d2", "d3", "d4", "curv"]        # d1=cadena sin ciclos
# generaciones: masa real → (align_rate ↓ con masa, decay ↑ con masa)
MASAS = {"electron": 1.0, "muon": 207.0, "tauon": 3477.0}
LBINS = [3, 4, 5, 6, 8]                  # bins de longitud de ciclo para el CONTROL del confound
# ===============================================================


def _align_decay(masa):
    return 1.0 / (1.0 + masa / 8.0), masa / (masa + 500.0)   # (align_rate, decay_prob)


def _relaja(adj, N, spins, align, decay, pasos, rng):
    """Relaja el marco hacia el promedio de vecinos con tasa 'align' (inercia = 1-align) y decaimiento 'decay'
    (persistencia = 1-decay): con prob decay un nodo re-randomiza su espín (pierde coherencia)."""
    s = spins.copy(); K = s.shape[1]
    for _ in range(pasos):
        ns = s.copy()
        for i in range(N):
            if adj[i]:
                v = s[list(adj[i])].sum(axis=0) + s[i]
                n = np.linalg.norm(v)
                if n > 1e-9:
                    tgt = v / n
                    w = s[i] + align * (tgt - s[i])
                    nw = np.linalg.norm(w)
                    if nw > 1e-9:
                        ns[i] = w / nw
        if decay > 0:
            dec = np.where(rng.random(N) < decay)[0]
            for i in dec:
                r = rng.standard_normal(K); ns[i] = r / np.linalg.norm(r)
        s = ns
    return s


def _holonomia_binned(adj, N, spins, K, rng):
    """Holonomía media POR BIN de longitud de ciclo (para controlar el confound de CS059) + media global."""
    ciclos = _ciclos_fundamentales(adj, N, N_CICLOS, rng)
    if not ciclos:
        return {}, float("nan"), 0.0
    w0 = np.zeros(K); w0[0] = 1.0
    porbin = {L: [] for L in LBINS}
    todo = []; longs = []
    for c in ciclos:
        h = _holonomia_ciclo(spins, c, w0); L = len(c)
        todo.append(h); longs.append(L)
        b = min(LBINS, key=lambda x: abs(x - L)) if L <= LBINS[-1] else LBINS[-1]
        if L in porbin:
            porbin[L].append(h)
    binmean = {L: (float(np.mean(v)) if len(v) >= 3 else None) for L, v in porbin.items()}
    return binmean, float(np.mean(todo)), float(np.mean(longs))


def _worker(arg):
    pid, dim, K, brazo, seed = arg
    ens = dict(_ensemble()); adj0, N = ens[dim]
    adj = [set(a) for a in adj0]
    spins = _spins(N, K, np.random.default_rng(seed * 991 + K * 7 + 1))
    rng = np.random.default_rng(seed * 100003 + pid * 17 + K)
    # aplicar el brazo
    if brazo == "adyacencia":
        # sin marco: holonomía indefinida → se marca aparte (no aplica). Devuelvo NaN.
        return [dict(point_id=pid, dim=dim, K=K, brazo=brazo, seed=seed, frame_burgers=float("nan"),
                     long_media=0.0, **{f"L{L}": "" for L in LBINS})]
    if brazo == "marco":
        s = spins                                    # CS059: aleatorio fijo
    elif brazo == "alineado":
        s = _relaja(adj, N, spins, 1.0, 0.0, PASOS, rng)
    elif brazo == "nulo":
        s = spins.copy(); rng.shuffle(s)             # espines barajados
    else:                                            # electron/muon/tauon
        align, decay = _align_decay(MASAS[brazo])
        s = _relaja(adj, N, spins, align, decay, PASOS, rng)
    binmean, fb, ml = _holonomia_binned(adj, N, s, K, np.random.default_rng(seed * 41 + K))
    fila = dict(point_id=pid, dim=dim, K=K, brazo=brazo, seed=seed,
                frame_burgers=round(fb, 5), long_media=round(ml, 2))
    for L in LBINS:
        v = binmean.get(L)
        fila[f"L{L}"] = round(v, 5) if v is not None else ""
    return [fila]


def main():
    print("CS060-A — LOS TRES LEPTONES: ¿la MASA (inercia+persistencia del marco) cambia la selección de dim?", flush=True)
    print("=" * 104, flush=True)
    print("PREDICCIÓN CIEGA: si la masa toca la geometría (G1), la dim con marcos más consistentes cambiará", flush=True)
    print("  entre electron/muon/tauon, CONTROLADO por long. de ciclo, y colapsará bajo NULL. Si no (G2), la", flush=True)
    print("  masa no toca la geometría: los tres seleccionan igual (o nada, controlando el confound de CS059).", flush=True)
    brazos = ["marco", "electron", "muon", "tauon", "alineado", "nulo"]
    args = []; pid = 0
    for dim in DIMS:
        for K in K_SWEEP:
            for brazo in brazos:
                for seed in range(SEEDS):
                    args.append((pid, dim, K, brazo, seed)); pid += 1
    if SMOKE:
        args = [a for a in args if a[4] < 1][:12]
    print(f"dims={DIMS} K={K_SWEEP} brazos={brazos} seeds={SEEDS} · corridas={len(args)} · workers={WORKERS}", flush=True)
    print(f"CONTROL de long. de ciclo en bins {LBINS} (la lección de CS059). Salida {OUT}", flush=True)

    hechos = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for row in csv.DictReader(f):
                hechos.add(int(row["point_id"]))
    args = [a for a in args if a[0] not in hechos]
    campos = ["point_id", "dim", "K", "brazo", "seed", "frame_burgers", "long_media"] + [f"L{L}" for L in LBINS]
    fout = open(OUT, "a", newline=""); wr = csv.DictWriter(fout, fieldnames=campos)
    if not hechos:
        wr.writeheader()
    t0 = time.time(); n = 0
    import multiprocessing as mp
    if WORKERS > 1 and not SMOKE:
        with mp.Pool(WORKERS) as pool:
            for filas in pool.imap_unordered(_worker, args, chunksize=1):
                for fila in filas: wr.writerow(fila)
                fout.flush(); n += 1
                if n % 25 == 0 or n == len(args):
                    dt = time.time() - t0; r = n / dt
                    print(f"  {n}/{len(args)} · {dt/60:.1f}min · ETA {(len(args)-n)/r/60:.1f}min", flush=True)
    else:
        for a in args:
            for fila in _worker(a): wr.writerow(fila)
            fout.flush(); n += 1
            print(f"  {n}/{len(args)} · {time.time()-t0:.1f}s", flush=True)
    fout.close()
    print(f"\nCOMPLETO: {n} corridas en {(time.time()-t0)/60:.1f} min → {OUT}", flush=True)


if __name__ == "__main__":
    main()
