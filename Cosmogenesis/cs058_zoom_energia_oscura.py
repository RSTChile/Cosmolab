"""
CS058 — ZOOM DENSO al candidato de ENERGÍA OSCURA: ¿la aceleración emergente de CS057 es REAL o ARTEFACTO?
=========================================================================================================
CS057 vio que "algo que acelera solo" (2ª diferencia del diámetro > 0, SIN insertar término) es 2.4× más
común cerca del punto físico. CS058 lo CARACTERIZA o lo MATA, con tres preguntas y sus falsaciones:
  1. ¿SOBREVIVE A LA RESOLUCIÓN? — misma región a ×1/×2/×4 pasos temporales. Si es real, la fracción acelera
     se ESTABILIZA/AFILA; si es submuestreo, DECAE. (G-FALSABLE-POR-RESOLUCION)
  2. ¿TIENE REGIÓN PROPIA? — mapa local de acelera; si es real, región CONTIGUA, no puntos dispersos.
  3. ¿ES FRONTERA CURVA? — cruzar acelera con viable_curv vs viable_d3/d4; si conecta con R7, vive donde la
     geometría viable es CURVA, no en el interior 3D-plano.
Más el brazo NULL: acelera medida sobre la trayectoria BARAJADA temporalmente — debe COLAPSAR (G-CONTROL-NULL).

REGIÓN LEÍDA DEL DATO (G-REGION-DEL-DATO), no elegida a mano: de cs057_paisaje.csv, la aceleración se
concentra en w_exp ALTO (0.79), w_grav/w_strong/w_em BAJOS (~0.3); w_weak/w_cool/alcance inertes. Ejes
activos = {w_grav, w_strong, w_em, w_exp}. Se densifica ESE box + la vecindad del punto físico.

PREDICCIONES CIEGAS (pre-registradas, ANTES de correr) — ver banner al ejecutar.
Reusa el motor de CS057 (mismo criterio ciego, misma def de acelera — NO se toca). numpy + multiprocessing.
"""
from __future__ import annotations
import os, sys, csv, math, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# reusar el motor de CS057 (exec-strip-main)
_s = open(os.path.join(_HERE, "cs057_paisaje_completo.py")).read()
_s = _s.replace('\nif __name__ == "__main__":\n    main()\n', "\n")
_C7 = {"__file__": os.path.join(_HERE, "cs057_paisaje_completo.py"), "__name__": "cs057_mod"}
exec(compile(_s, "cs057_paisaje_completo.py", "exec"), _C7)
_grav = _C7["_grav"]; _confin = _C7["_confin"]; _em = _C7["_em"]; _debil = _C7["_debil"]
_despliegue = _C7["_despliegue"]; _diam = _C7["_diam"]; _giant = _C7["_giant"]; _colores = _C7["_colores"]
_construye_ensemble = _C7["_construye_ensemble"]; _alc_a_dmax = _C7["_alc_a_dmax"]
R_GRAV = _C7["R_GRAV"]; R_STRONG = _C7["R_STRONG"]; R_EM = _C7["R_EM"]; R_WEAK = _C7["R_WEAK"]; R_EXP = _C7["R_EXP"]
T_HI = _C7["T_HI"]; T_LO = _C7["T_LO"]; T_CONF = _C7["T_CONF"]

# ============================ CONFIG ============================
BASE_STEPS = 16                         # ×1; el brazo de resolución multiplica esto
RES_MULT   = [1, 2, 4]                  # ×1/×2/×4 pasos — la FALSACIÓN por resolución
SEEDS      = int(os.environ.get("CS058_SEEDS", 6))
NIV        = int(os.environ.get("CS058_NIV", 6))   # niveles por eje activo (malla local densa)
WORKERS    = int(os.environ.get("CS058_WORKERS", max(1, (os.cpu_count() or 4) - 2)))
OUT        = os.environ.get("CS058_OUT", os.path.join(_HERE, "cs058_zoom.csv"))
SMOKE      = os.environ.get("CS058_SMOKE", "") != ""
CL = ["d1", "d2", "d3", "d4", "curv"]
# ===============================================================


def _T_paso(step, steps, wcool):
    frac = step / max(steps - 1, 1)
    depth = 0.2 + 1.8 * wcool
    return T_HI * (T_LO / T_HI) ** min(1.0, frac * depth)


def _acelera(D, expande):
    """MISMA definición que CS057: 2ª diferencia del diámetro > 0 en la mitad tardía, solo si expande."""
    if len(D) < 5 or not expande:
        return 0
    half = len(D) // 2
    seg = D[half - 1:] if half >= 1 else D
    diff2 = [seg[k + 1] - 2 * seg[k] + seg[k - 1] for k in range(1, len(seg) - 1)]
    return int(bool(diff2) and (sum(diff2) / len(diff2)) > 0)


def _acelera_null(D, expande, rng):
    """NULL de la métrica de CS057: acelera sobre la trayectoria BARAJADA. (La de CS057 telescopia → NULL débil;
    por eso se añade la métrica ROBUSTA abajo, que es la falsación de verdad.)"""
    if len(D) < 5 or not expande:
        return 0
    Dp = list(D); rng.shuffle(Dp)
    half = len(Dp) // 2
    seg = Dp[half - 1:]
    diff2 = [seg[k + 1] - 2 * seg[k] + seg[k - 1] for k in range(1, len(seg) - 1)]
    return int(bool(diff2) and (sum(diff2) / len(diff2)) > 0)


def _accel_robusto(D, expande):
    """Métrica ROBUSTA de aceleración: PENDIENTE de la velocidad (1as diferencias del diámetro) vs tiempo por
    regresión — usa TODO el orden temporal (no telescopia). >0 = la expansión se acelera de verdad."""
    if len(D) < 5 or not expande:
        return 0
    v = np.diff(np.asarray(D, float))
    if len(v) < 3:
        return 0
    tt = np.arange(len(v))
    return int(np.polyfit(tt, v, 1)[0] > 1e-9)


def _accel_robusto_null(D, expande, rng):
    """NULL de la métrica robusta: barajar la trayectoria destruye el orden temporal → la pendiente de la
    velocidad debe colapsar a ~0 (sign aleatorio). Si acc_robusto >> este NULL, la aceleración es REAL."""
    if len(D) < 5 or not expande:
        return 0
    Dp = np.asarray(D, float).copy(); rng.shuffle(Dp)
    v = np.diff(Dp)
    if len(v) < 3:
        return 0
    tt = np.arange(len(v))
    return int(np.polyfit(tt, v, 1)[0] > 1e-9)


def proceso058(adj0, N, color0, carga0, W, dmax_grav, steps, rng):
    """Motor de CS057 (brazo SYNC) con nº de pasos EXPLÍCITO (para el barrido de resolución). Devuelve la
    trayectoria del diámetro D y si expandió — de ahí se leen acelera (real) y acelera_null (barajado)."""
    wg, ws, wem, wwk, wexp, wcool = W
    adj = [set(a) for a in adj0]; col = color0.copy(); car = carga0.copy()
    deg0 = [len(a) for a in adj]; t = np.zeros(N, dtype=np.int32)
    fuerzas = ("grav", "strong", "em", "weak"); CAP_E = 12 * N
    D = []; G = []

    def aplica(f, T):
        if f == "grav": _grav(adj, N, rng, wg * R_GRAV, dmax_grav, T)
        elif f == "strong":
            if T < T_CONF: _confin(adj, N, col, t, rng, ws * R_STRONG)
        elif f == "em": _em(adj, N, car, deg0, rng, wem * R_EM)
        elif f == "weak": _debil(N, col, car, rng, wwk * R_WEAK)

    for step in range(steps):
        T = _T_paso(step, steps, wcool)
        E = sum(len(a) for a in adj) // 2
        if E < 2 or E > CAP_E:
            D.append(_diam(adj, N)); G.append(_giant(adj, N)); break
        for f in fuerzas:
            aplica(f, T)
        _despliegue(adj, N, rng, wexp * R_EXP)
        dd = _diam(adj, N); gg = _giant(adj, N); D.append(dd); G.append(gg)
        if gg >= 0.9 and dd <= 2: break
    d0 = D[0] if D else 0; d1 = D[-1] if D else 0
    expande = d1 > d0
    estable = int(len(G) > 0 and G[-1] >= 0.45 and d1 >= 2 and min(G[len(G)//2:] or [0]) >= 0.35)
    return D, int(expande), estable


# ---- región LEÍDA DEL DATO (G-REGION-DEL-DATO) ----
def _centroide_del_csv():
    import pandas as pd
    df = pd.read_csv(os.path.join(_HERE, "cs057_paisaje.csv"))
    df["acc"] = df[[f"acelera_{c}" for c in CL]].sum(axis=1)
    acc = df[(df.phys == 0) & (df.acc > 0)]
    W = ["w_grav", "w_strong", "w_em", "w_weak", "w_exp", "w_cool", "alc"]
    return {w: float(acc[w].mean()) for w in W}


def _malla_local():
    """Box denso alrededor del centroide de alta-aceleración (ejes activos) + vecindad del punto físico."""
    c = _centroide_del_csv()
    niv = NIV
    # ejes activos y sus rangos locales (centrados en el dato, cubriendo su vecindad)
    ejes = {
        "w_grav":   np.linspace(0.0, 0.5, niv),
        "w_strong": np.linspace(0.0, 0.6, niv),
        "w_em":     np.linspace(0.0, 0.6, niv),
        "w_exp":    np.linspace(0.5, 1.0, niv),
    }
    fijos = {"w_weak": 0.5, "w_cool": 0.5, "alc": 0.5}   # inertes → fijos al centro
    puntos = []
    for g in ejes["w_grav"]:
        for s in ejes["w_strong"]:
            for e in ejes["w_em"]:
                for x in ejes["w_exp"]:
                    puntos.append((g, s, e, fijos["w_weak"], x, fijos["w_cool"], fijos["alc"], 0))  # region=0 dato
    # vecindad del punto físico (strong=1, em=1/137, grav~0; variar exp/cool/alc)
    for x in np.linspace(0.2, 0.9, niv):
        for cool in np.linspace(0.2, 0.9, niv):
            for al in np.linspace(0.0, 1.0, max(3, niv // 2)):
                puntos.append((1e-38, 1.0, 1/137.0, 1e-6, x, cool, al, 1))  # region=1 físico
    return puntos, c


def _campos():
    return (["point_id", "region", "res_mult", "steps", "seed",
             "w_grav", "w_strong", "w_em", "w_weak", "w_exp", "w_cool", "alc", "dmax_grav"]
            + [f"{m}_{c}" for c in CL for m in ("acc", "accnull", "accrob", "accrobnull", "viable", "curv")])


def _worker(arg):
    pid, W7, region = arg
    W = tuple(float(x) for x in W7[:6]); dmax = _alc_a_dmax(float(W7[6]))
    ens = _construye_ensemble()
    filas = []
    for res in RES_MULT:
        steps = BASE_STEPS * res
        for seed in range(SEEDS):
            fila = dict(point_id=pid, region=region, res_mult=res, steps=steps, seed=seed,
                        w_grav=W[0], w_strong=W[1], w_em=W[2], w_weak=W[3], w_exp=W[4], w_cool=W[5],
                        alc=float(W7[6]), dmax_grav=dmax)
            rng = np.random.default_rng(seed * 100003 + pid * 17 + res)
            for ci, (nom, (adj, N)) in enumerate(ens):
                col = _colores(N, np.random.default_rng(seed * 131 + ci * 17 + 1))
                car = (np.arange(N) % 2).astype(np.int8)
                np.random.default_rng(seed * 977 + ci * 29 + 7).shuffle(car)
                D, exp, est = proceso058(adj, N, col, car, W, dmax, steps,
                                         np.random.default_rng(seed * 31 + ci * 101 + res * 7 + 13))
                fila[f"acc_{nom}"] = _acelera(D, exp)
                fila[f"accnull_{nom}"] = _acelera_null(D, exp, np.random.default_rng(seed * 53 + ci * 13 + res))
                fila[f"accrob_{nom}"] = _accel_robusto(D, exp)
                fila[f"accrobnull_{nom}"] = _accel_robusto_null(D, exp, np.random.default_rng(seed * 71 + ci * 19 + res))
                fila[f"viable_{nom}"] = int(exp and est)
                fila[f"curv_{nom}"] = 1 if nom == "curv" else 0
            filas.append(fila)
    return filas


def main():
    print("CS058 — ZOOM DENSO al candidato de ENERGÍA OSCURA (¿real o artefacto?)", flush=True)
    print("=" * 100, flush=True)
    print("PREDICCIONES CIEGAS (pre-registradas):", flush=True)
    print("  1. RESOLUCIÓN: si la aceleración es REAL, su fracción se ESTABILIZA/AFILA de ×1→×2→×4 pasos;", flush=True)
    print("     si es artefacto de submuestreo, DECAE hacia 0. (falsación directa)", flush=True)
    print("  2. NULL: la aceleración real COLAPSA al barajar la trayectoria (accnull << acc).", flush=True)
    print("  3. FRONTERA CURVA: la aceleración se concentra donde la geometría viable es CURVA, no en 3D-plano.", flush=True)
    puntos, cent = _malla_local()
    if SMOKE:
        puntos = puntos[:6] + [p for p in puntos if p[7] == 1][:4]
    print(f"\ncentroide alta-acc del dato: { {k: round(v,2) for k,v in cent.items()} }", flush=True)
    print(f"puntos={len(puntos)} · res={RES_MULT} · seeds={SEEDS} · niv={NIV} · workers={WORKERS}", flush=True)
    print(f"corridas ≈ {len(puntos)*len(RES_MULT)*SEEDS} · salida {OUT}", flush=True)

    args = [(i, p[:7], p[7]) for i, p in enumerate(puntos)]
    hechos = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for row in csv.DictReader(f):
                hechos.add(int(row["point_id"]))
    args = [a for a in args if a[0] not in hechos]
    campos = _campos()
    fout = open(OUT, "a", newline=""); wr = csv.DictWriter(fout, fieldnames=campos)
    if os.path.getsize(OUT) if os.path.exists(OUT) else 0:
        pass
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
                    print(f"  {n}/{len(args)} · {dt/60:.1f}min · ETA {(len(args)-n)/r/3600:.2f}h", flush=True)
    else:
        for a in args:
            for fila in _worker(a): wr.writerow(fila)
            fout.flush(); n += 1
            print(f"  {n}/{len(args)} · {time.time()-t0:.1f}s", flush=True)
    fout.close()
    print(f"\nCOMPLETO: {n} puntos en {(time.time()-t0)/60:.1f} min → {OUT}", flush=True)


if __name__ == "__main__":
    main()
