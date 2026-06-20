#!/usr/bin/env python3
"""
V182C — SENTIDO COMPARTIDO: ¿EMERGE UNA CONVENCION, Y QUE INGREDIENTE LA PRODUCE?
================================================================================
Sigue de V182A.5. En el informe, el nodo C-N9 (sentido compartido, R₁ ↔ R₂ ⇒
S_shared) quedo como APROXIMADO: se transfiere y acumula competencia, pero nadie
habia mostrado que A y B INVENTEN una convencion que ninguno tenia solo, ni QUE
ingrediente la hace posible.

BITACORA METODOLOGICA (honestidad — Libertad Funcional). Llegar al diseño correcto
costo dos falsaciones, ambas informativas:
  (1) Ablacion memoria-ON / peso-fijo-OFF con premio de coordinacion: AMBOS
      convergieron. Un empujon reactivo hacia el otro ya basta; la acumulacion no
      era decisiva.
  (2) Tres brazos con premio de coordinacion: hasta el brazo SIN comunicacion
      convergio. El premio refuerza la MISMA banda en ambos a la vez -> es
      coordinacion implicita. El premio era el confound.
  (3) Control a igual coeficiente: la supuesta ventaja de la memoria era artefacto
      de magnitud (0.60 vs 0.10). A igual acoplamiento, memoria y reaccion
      convergen igual; la memoria no es lo que produce la convencion.
Resultado: la convencion se modela como ALINEAMIENTO MUTUO sin premio (clase "juego
de nombres"): la convencion es una regularidad autosostenida por expectativa mutua,
no un pago externo. Y los brazos con comunicacion usan EL MISMO coeficiente, para
que cualquier diferencia sea memoria, no magnitud.

PREGUNTA (C-N9): ante un dilema simetrico (-60°/+60°, ninguna correcta), ¿emerge un
PUNTO FOCAL estable y EMERGENTE —no programado—, y que se necesita para que emerja?

TRES BRAZOS (sin premio; alineamiento por imitacion):
  1. AISLADA  (sin comunicacion): cada uno hace softmax sobre su valencia y decae;
              NO observa al otro. No hay canal de alineamiento.
  2. COMUNIC  (comunicacion, reactiva): observa la eleccion ACTUAL del otro y la
              incorpora con peso COEF. Sin memoria.
  3. MEMORIA  (comunicacion + memoria): observa y ACUMULA la preferencia del otro
              (histeresis) con el MISMO COEF de techo. Brazo de control de equidad.

HIPOTESIS:
  H_com (principal): la comunicacion es necesaria para la convencion.
        -> AISLADA fracasa; COMUNIC y MEMORIA forman convencion path-dependent.
  H_mem (control): a igual coeficiente, la memoria NO añade convencion.
        -> COMUNIC ≈ MEMORIA. (La memoria servia para ACUMULAR competencia, A.5,
           que es otra pregunta.)

FALSADORES: si AISLADA converge -> H_com refutada (la convencion no requiere
  comunicacion). Si MEMORIA supera claramente a COMUNIC a igual coeficiente ->
  la memoria SI aporta a la convencion (se reporta el dato).

DECISIONES DE DISEÑO (declaradas):
  - Dinamica a nivel de valencia/preferencia; el cuerpo V180 se importa verbatim por
    continuidad (la valencia vive en su ValenciaLocal), su orientacion no es el locus.
  - 8 pares de semillas: muestran que el foco es EMERGENTE (path-dependent), no
    programado. Replicacion honesta sin inventar estadistica de una sola corrida.

MARCADORES: ✅ convencion estable · ❌ sin convencion · ⊘ degenerada · · n/a
================================================================================
"""
import os, json, time, math
import numpy as np
import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("V180", os.path.join(_here, "V180.py"))
V180 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V180)

OPCIONES = [-60.0, 60.0]
N_EPISODES   = 150     # episodios del dilema repetido
VENTANA      = 20      # ventana final para medir convencion
N_PARES      = 8       # pares de semillas por brazo (muestra la emergencia)
TEMP         = 0.90    # temperatura del softmax (exploracion)
COEF         = 0.20    # MISMO coeficiente de acoplamiento en COMUNIC y MEMORIA (equidad)
LR_M         = 0.15    # tasa de acumulacion de la preferencia del otro (MEMORIA)
DECAY        = 0.03    # decaimiento de valencia por episodio
RUIDO_INI    = 0.01    # ruido inicial que rompe la simetria
UMBRAL_COORD = 0.80    # coordinacion minima para "convencion formada"
UMBRAL_CONV  = 0.80    # dominancia minima de una opcion para "estable"
UMBRAL_BLOQ  = 0.90    # coordinacion en ventana movil para considerar "bloqueado"
TS = time.strftime("%Y%m%d_%H%M%S")

BRAZOS = [
    ("aislada", "AISLADA (sin comunicacion)"),
    ("comunic", "COMUNIC (comunicacion reactiva)"),
    ("memoria", "MEMORIA (comunicacion + memoria, mismo coef)"),
]


def _key(b): return round(b/5)*5 if b != 0 else 0
def getv(v, b): return v.valencia.get(_key(b), 0.0)
def setv(v, b, x): v.valencia[_key(b)] = float(np.clip(x, -100, 100))
def otra(b): return 60.0 if b == -60.0 else -60.0


def nueva_valencia(seed):
    v = V180.ValenciaLocal()
    rng = np.random.default_rng(seed)
    for b in OPCIONES:
        v.valencia[_key(b)] = float(rng.normal(0.0, RUIDO_INI))
    return v


def elegir(v, rng):
    a = getv(v, -60.0); b = getv(v, 60.0)
    m = max(a, b)
    ea = math.exp((a - m) / TEMP); eb = math.exp((b - m) / TEMP)
    p_neg = ea / (ea + eb)
    return -60.0 if rng.random() < p_neg else 60.0


def correr_par(par, brazo):
    vA = nueva_valencia(par * 2 + 1); vB = nueva_valencia(par * 2 + 2)
    mA = {_key(b): 0.0 for b in OPCIONES}; mB = {_key(b): 0.0 for b in OPCIONES}
    rngA = np.random.default_rng(1000 + par); rngB = np.random.default_rng(2000 + par)
    hist = []
    for _ in range(N_EPISODES):
        cA = elegir(vA, rngA); cB = elegir(vB, rngB)
        if brazo == "memoria":                 # observa + acumula (histeresis), techo COEF
            mA[_key(cB)] = (1 - LR_M) * mA[_key(cB)] + LR_M
            mA[_key(otra(cB))] = (1 - LR_M) * mA[_key(otra(cB))]
            mB[_key(cA)] = (1 - LR_M) * mB[_key(cA)] + LR_M
            mB[_key(otra(cA))] = (1 - LR_M) * mB[_key(otra(cA))]
            setv(vA, cB, getv(vA, cB) + COEF * mA[_key(cB)])
            setv(vB, cA, getv(vB, cA) + COEF * mB[_key(cA)])
        elif brazo == "comunic":               # observa la eleccion actual, peso fijo COEF
            setv(vA, cB, getv(vA, cB) + COEF)
            setv(vB, cA, getv(vB, cA) + COEF)
        # brazo == "aislada": no observa al otro; sin canal de alineamiento
        for v in (vA, vB):
            for b in OPCIONES:
                setv(v, b, getv(v, b) * (1 - DECAY))
        hist.append((cA, cB, (cA == cB)))
    return hist


def episodio_bloqueo(hist):
    n = len(hist)
    for e in range(0, n - VENTANA + 1):
        if all(sum(1 for (_, _, x) in hist[f:f + VENTANA] if x) / VENTANA >= UMBRAL_BLOQ
               for f in range(e, n - VENTANA + 1)):
            return e
    return None


def analizar(hist):
    ult = hist[-VENTANA:]
    coord = sum(1 for (_, _, c) in ult if c) / len(ult)
    elec = [cA for (cA, _, _) in ult] + [cB for (_, cB, _) in ult]
    n_neg = sum(1 for e in elec if e == -60.0); n_pos = len(elec) - n_neg
    dom = -60.0 if n_neg >= n_pos else 60.0
    fuerza = max(n_neg, n_pos) / len(elec)
    conv = (coord >= UMBRAL_COORD) and (fuerza >= UMBRAL_CONV)
    bloq = episodio_bloqueo(hist) if conv else None
    return coord, dom, fuerza, conv, bloq


def correr_brazo(brazo, etiqueta):
    print(f"\n{'='*94}\n{etiqueta}\n{'='*94}")
    print(f"  {'par':>3} | coord_ult20 | foco | fuerza | bloqueo(ep) | convencion")
    print(f"  ----+-------------+------+--------+-------------+-----------")
    filas = []
    for par in range(N_PARES):
        hist = correr_par(par, brazo)
        coord, dom, fuerza, conv, bloq = analizar(hist)
        mk = '✅' if conv else '❌'
        foco = f"{dom:+.0f}°" if conv else "—"
        bloq_txt = f"{bloq}" if bloq is not None else "—"
        print(f"  {par:>3} | {coord*100:>9.0f}% | {foco:>4} | {fuerza*100:>5.0f}% | {bloq_txt:>11} | {mk}")
        filas.append({'par': par, 'coord': coord, 'dom': dom if conv else None,
                      'fuerza': fuerza, 'conv': bool(conv), 'bloqueo': bloq})
    n_conv = sum(1 for f in filas if f['conv'])
    doms = [f['dom'] for f in filas if f['conv']]
    n_neg = sum(1 for d in doms if d == -60.0); n_pos = sum(1 for d in doms if d == 60.0)
    bloqs = [f['bloqueo'] for f in filas if f['conv'] and f['bloqueo'] is not None]
    bloq_med = (sum(bloqs) / len(bloqs)) if bloqs else None
    print(f"\n  Convenciones estables: {n_conv}/{N_PARES}"
          + (f" | foco -60° en {n_neg}, +60° en {n_pos}" if n_conv else "")
          + (f" | bloqueo medio {bloq_med:.0f} ep" if bloq_med is not None else ""))
    return {'brazo': brazo, 'filas': filas, 'n_conv': n_conv,
            'n_neg': n_neg, 'n_pos': n_pos, 'bloqueo_medio': bloq_med}


def main():
    print("=" * 94)
    print("V182C — SENTIDO COMPARTIDO: ¿EMERGE UNA CONVENCION, Y QUE LA PRODUCE?")
    print("=" * 94)
    print("  Dilema de Schelling repetido, -60°/+60° simetricas, convencion por alineamiento mutuo.")
    print(f"  {N_EPISODES} episodios, {N_PARES} pares de semillas, ventana {VENTANA}, coef={COEF} (igual en com/mem).")
    print("=" * 94)
    t0 = time.time()
    res = {}
    for brazo, et in BRAZOS:
        res[brazo] = correr_brazo(brazo, et)

    a, c, m = res['aislada'], res['comunic'], res['memoria']
    print(f"\n{'#'*94}\n#  VEREDICTO\n{'#'*94}")
    print(f"  AISLADA (sin comunicacion)      : {a['n_conv']}/{N_PARES}")
    print(f"  COMUNIC (comunicacion)          : {c['n_conv']}/{N_PARES}"
          + (f" | foco -60°×{c['n_neg']} +60°×{c['n_pos']}" if c['n_conv'] else "")
          + (f" | bloqueo {c['bloqueo_medio']:.0f} ep" if c['bloqueo_medio'] else ""))
    print(f"  MEMORIA (comunicacion + memoria): {m['n_conv']}/{N_PARES}"
          + (f" | bloqueo {m['bloqueo_medio']:.0f} ep" if m['bloqueo_medio'] else ""))

    com_necesaria = (a['n_conv'] <= 0.25 * N_PARES
                     and c['n_conv'] >= 0.75 * N_PARES and m['n_conv'] >= 0.75 * N_PARES)
    emergente = (c['n_neg'] > 0 and c['n_pos'] > 0) or (m['n_neg'] > 0 and m['n_pos'] > 0)
    mem_no_aporta = abs(c['n_conv'] - m['n_conv']) <= 0.25 * N_PARES
    print()
    if a['n_conv'] > 0.5 * N_PARES:
        print("  ❌ Falsador de H_com: AISLADA converge. La convencion no requiere comunicacion. Se lee.")
    elif com_necesaria and emergente:
        print("  ✅ H_com confirmada: la COMUNICACION (observar al otro, R₁ ↔ R₂) es necesaria y")
        print("     suficiente para la convencion. Sin ella no emerge; con ella si, con punto focal")
        print("     path-dependent (no programado). Eso mueve C-N9 de aproximado hacia operativo.")
        if mem_no_aporta:
            print("  ✅ H_mem (control): a igual coeficiente, la memoria NO añade convencion")
            print(f"     (COMUNIC {c['n_conv']}/{N_PARES} ≈ MEMORIA {m['n_conv']}/{N_PARES}). La memoria producia")
            print("     ACUMULACION de competencia (A.5), no la convencion. Dos roles distintos.")
        else:
            print(f"  ⊘ La memoria SI separa a igual coeficiente (COMUNIC {c['n_conv']} vs MEMORIA {m['n_conv']}). Se reporta.")
    else:
        print("  ⊘ Separacion parcial / calibracion no concluyente. Se reporta y se lee.")
    print(f"\n  tiempo {time.time()-t0:.2f}s")
    os.makedirs("V182_logs", exist_ok=True)
    with open(f"V182_logs/v182c_sentido_compartido_{TS}.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
