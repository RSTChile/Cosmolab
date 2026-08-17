"""
CS061 — LA MASA QUE EMERGE: el vértice de 3 puntos (tipo Higgs) donde la inercia NACE, no se asigna.
====================================================================================================
Convergencia del arco (CS): el vértice de 3 puntos que CS059 pidió = el del Higgs = el origen de la masa.
CS061 los une: un campo de fondo uniforme (Higgs exaptado) + un vértice de 3 puntos (tríada local) del que la
INERCIA EMERGE (no se asigna), y se pregunta si el 3-puntos selecciona una dimensión donde el 2-puntos (CS059)
no pudo — y si emerge un espectro de masas no trivial.

MECANISMO (dim-neutral, 3-cuerpos genuino, SIN confound de longitud de ciclo — la lección de CS059/CS060):
- CAMPO φ UNIFORME (mismo valor en todo el sustrato — G-CAMPO-UNIFORME). φ=1.
- VÉRTICE DE 3 PUNTOS = DEFECTO DE CIERRE DE TRÍADAS LOCALES: para cada nodo i y cada par de vecinos (j,k),
  el defecto = ángulo entre [transportar w por j→i→k] y [transportar w por j→k directo]. Es de 3 cuerpos
  (j,i,k), TAMAÑO FIJO (3), definido en cualquier grafo (no necesita triángulos cerrados), dim-neutral. NO
  telescopia con longitud de ciclo (a diferencia del 2-puntos de CS059).
- INERCIA EMERGENTE (G-MASA-EMERGE-NO-ASIGNADA): m_i = φ · frustración_local(i), donde frustración_local(i) =
  media del defecto de tríada sobre los pares de vecinos de i. Un nodo cuyos marcos locales NO cierran adquiere
  masa (análogo al Higgs por ruptura). Se SORTEA nada; la masa se MIDE. Espectro comparado con 1:207:3477 SOLO
  a posteriori.
- La inercia RETROACTÚA: nodos pesados resisten reorientar su marco (como CS060-A).
- JUEZ DOBLE: (a) holonomía del marco por dim, CON control de longitud de ciclo — ¿el 3-puntos (con campo+
  inercia) selecciona dim donde el 2-puntos no?; (b) el espectro de inercias emergentes.
- BRAZOS: 3punto (campo+inercia+vértice) · 2punto (=CS059, sin campo) · null_campo (inercia barajada) ·
  null_vertice (frustración de espines barajados). La selección real debe COLAPSAR bajo los NULL.

Éxito ≠ "salió 3D" (G-NO-FORZAR-3D): selección consistente que colapsa bajo NULL, o espectro no-trivial.
Reusa CS059. numpy + multiprocessing.
"""
from __future__ import annotations
import os, sys, csv, math, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_s = open(os.path.join(_HERE, "cs059_espin_como_marco.py")).read().replace('\nif __name__ == "__main__":\n    main()\n', "\n")
_C9 = {"__file__": os.path.join(_HERE, "cs059_espin_como_marco.py"), "__name__": "cs059_mod"}
exec(compile(_s, "cs059_espin_como_marco.py", "exec"), _C9)
_spins = _C9["_spins"]; _transporta = _C9["_transporta"]; _holonomia_ciclo = _C9["_holonomia_ciclo"]
_ciclos_fundamentales = _C9["_ciclos_fundamentales"]; _ensemble = _C9["_ensemble"]

# ============================ CONFIG ============================
K_SWEEP  = [3, 4]
SEEDS    = int(os.environ.get("CS061_SEEDS", 10))
N_CICLOS = int(os.environ.get("CS061_CICLOS", 500))
PASOS    = 10
PHI      = 1.0                           # campo de fondo UNIFORME (Higgs)
MAXPARES = 12                            # pares de vecinos muestreados por nodo (coste)
WORKERS  = int(os.environ.get("CS061_WORKERS", max(1, (os.cpu_count() or 4) - 2)))
OUT      = os.environ.get("CS061_OUT", os.path.join(_HERE, "cs061_masa.csv"))
SMOKE    = os.environ.get("CS061_SMOKE", "") != ""
DIMS = ["d2", "d3", "d4", "curv"]
LBINS = [3, 4, 5, 6, 8]
# ===============================================================


def _defecto_triada(spins, j, i, k, w0):
    """Defecto de cierre del vértice de 3 puntos (j,i,k): ángulo entre transportar w0 por j→i→k y por j→k."""
    a = _transporta(spins[j], spins[i], w0)
    a = _transporta(spins[i], spins[k], a)          # camino j→i→k
    b = _transporta(spins[j], spins[k], w0)          # directo j→k
    c = float(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return math.acos(max(-1.0, min(1.0, c)))


def _frustracion(adj, N, spins, K, rng):
    """Inercia EMERGENTE por nodo: m_i = φ · media del defecto de tríada sobre pares de vecinos de i.
    3-cuerpos, local, tamaño fijo → sin confound de longitud de ciclo. Devuelve vector m (N,)."""
    w0 = np.zeros(K); w0[0] = 1.0
    m = np.zeros(N)
    for i in range(N):
        vec = list(adj[i])
        if len(vec) < 2:
            continue
        pares = []
        if len(vec) <= 5:
            for a in range(len(vec)):
                for b in range(a + 1, len(vec)):
                    pares.append((vec[a], vec[b]))
        else:
            for _ in range(MAXPARES):
                a, b = rng.choice(len(vec), 2, replace=False)
                pares.append((vec[a], vec[b]))
        d = [_defecto_triada(spins, j, i, k, w0) for (j, k) in pares]
        m[i] = PHI * float(np.mean(d)) if d else 0.0
    return m


def _relaja_inercia(adj, N, spins, m, pasos, rng):
    """Relaja el marco; cada nodo resiste según su inercia EMERGENTE m_i (align = 1/(1+m·escala))."""
    s = spins.copy(); K = s.shape[1]
    esc = 3.0 / (m.mean() + 1e-9)                    # escala para que la inercia media dé align~0.25
    align = 1.0 / (1.0 + m * esc)
    for _ in range(pasos):
        ns = s.copy()
        for i in range(N):
            if adj[i]:
                v = s[list(adj[i])].sum(axis=0) + s[i]
                n = np.linalg.norm(v)
                if n > 1e-9:
                    w = s[i] + align[i] * (v / n - s[i])
                    nw = np.linalg.norm(w)
                    if nw > 1e-9:
                        ns[i] = w / nw
        s = ns
    return s


def _holonomia_binned(adj, N, spins, K, rng):
    ciclos = _ciclos_fundamentales(adj, N, N_CICLOS, rng)
    if not ciclos:
        return {}, float("nan")
    w0 = np.zeros(K); w0[0] = 1.0
    porbin = {L: [] for L in LBINS}; todo = []
    for c in ciclos:
        h = _holonomia_ciclo(spins, c, w0); todo.append(h)
        if len(c) in porbin:
            porbin[len(c)].append(h)
    binmean = {L: (float(np.mean(v)) if len(v) >= 3 else None) for L, v in porbin.items()}
    return binmean, float(np.mean(todo))


def _worker(arg):
    pid, dim, K, brazo, seed = arg
    ens = dict(_ensemble()); adj0, N = ens[dim]
    adj = [set(a) for a in adj0]
    spins = _spins(N, K, np.random.default_rng(seed * 991 + K * 7 + 1))
    rng = np.random.default_rng(seed * 100003 + pid * 17 + K)
    # inercia emergente (vértice de 3 puntos con el campo)
    m = _frustracion(adj, N, spins, K, rng)
    if brazo == "2punto":
        s = spins                                    # CS059: sin campo, sin inercia (aleatorio fijo)
        mesp = m                                      # (se reporta el espectro igual, como referencia)
    elif brazo == "3punto":
        s = _relaja_inercia(adj, N, spins, m, PASOS, rng)   # campo+inercia EMERGENTE retroactúa
        mesp = m
    elif brazo == "null_campo":
        mm = m.copy(); rng.shuffle(mm)               # inercia barajada (rompe la estructura del campo)
        s = _relaja_inercia(adj, N, spins, mm, PASOS, rng); mesp = mm
    elif brazo == "null_vertice":
        sp2 = spins.copy(); rng.shuffle(sp2)          # frustración de espines barajados (vértice al azar)
        m2 = _frustracion(adj, N, sp2, K, rng)
        s = _relaja_inercia(adj, N, spins, m2, PASOS, rng); mesp = m2
    binmean, fb = _holonomia_binned(adj, N, s, K, np.random.default_rng(seed * 41 + K))
    # espectro de inercias emergentes: cuantiles (para comparar forma con 1:207:3477 a POSTERIORI)
    mp = mesp[mesp > 1e-6]
    if len(mp) > 5:
        q = np.quantile(mp, [0.5, 0.9, 0.99])
        r_lo_hi = float(q[2] / (q[0] + 1e-9))         # razón alto/bajo del espectro
        frac0 = float(np.mean(mesp < 0.05 * (mesp.max() + 1e-9)))  # fracción ~sin masa (tipo fotón)
    else:
        r_lo_hi = float("nan"); frac0 = float("nan")
    fila = dict(point_id=pid, dim=dim, K=K, brazo=brazo, seed=seed, frame_burgers=round(fb, 5),
                m_media=round(float(mesp.mean()), 4), m_max=round(float(mesp.max()), 4),
                espectro_razon=round(r_lo_hi, 3) if r_lo_hi == r_lo_hi else "",
                frac_sin_masa=round(frac0, 3) if frac0 == frac0 else "")
    for L in LBINS:
        v = binmean.get(L)
        fila[f"L{L}"] = round(v, 5) if v is not None else ""
    return [fila]


def main():
    print("CS061 — LA MASA QUE EMERGE (vértice de 3 puntos tipo Higgs): ¿selecciona dim donde el 2-puntos no?", flush=True)
    print("=" * 106, flush=True)
    print("PREDICCIÓN CIEGA: (a) si el 3-puntos es el ingrediente, su holonomía por dim, CONTROLADA por longitud", flush=True)
    print("  de ciclo, seleccionará una dim que el 2-puntos (CS059) no pudo, y colapsará bajo null_campo/vertice.", flush=True)
    print("  (b) la inercia emergente dará un ESPECTRO no-trivial (algunos ~0 tipo fotón, algunos altos) sin", flush=True)
    print("  asignarlo. Comparación con 1:207:3477 SOLO a posteriori. Éxito NUNCA = 'salió 3D'.", flush=True)
    brazos = ["2punto", "3punto", "null_campo", "null_vertice"]
    args = []; pid = 0
    for dim in DIMS:
        for K in K_SWEEP:
            for brazo in brazos:
                for seed in range(SEEDS):
                    args.append((pid, dim, K, brazo, seed)); pid += 1
    if SMOKE:
        args = [a for a in args if a[4] < 1][:16]
    print(f"dims={DIMS} K={K_SWEEP} brazos={brazos} seeds={SEEDS} · corridas={len(args)} · workers={WORKERS}", flush=True)

    hechos = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for row in csv.DictReader(f):
                hechos.add(int(row["point_id"]))
    args = [a for a in args if a[0] not in hechos]
    campos = ["point_id", "dim", "K", "brazo", "seed", "frame_burgers", "m_media", "m_max",
              "espectro_razon", "frac_sin_masa"] + [f"L{L}" for L in LBINS]
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
                if n % 20 == 0 or n == len(args):
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
