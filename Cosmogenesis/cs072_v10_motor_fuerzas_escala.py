"""
CS072 -- TAREAS 1/2/3 de INSTRUCCION_CS072_motor_rediseñado_fuerzas_PARA_CC.md, sobre cs072_motor_fuerzas.py
(motor rediseñado por CS: B de ligadura empieza en CERO, SOLO las fuerzas la construyen -- el campo térmico T
es sólo condición inicial y NUNCA aporta a B; el contador lee B, no T).

TAREA 1: confirmar que apagar confinamiento lleva bariones a 0 en TODAS las escalas N=68/136/272/544, a pasos
  equilibrados (300+; sube si el conteo aún cambia entre pasos y pasos+100).
TAREA 2: con confinamiento puesto, ¿los bariones crecen ~ n_quarks/3 (física de estequiometría, 3 quarks/barión)?
TAREA 3: auditoría de apagado de las 4 piezas que tiene ESTE motor (confinamiento, em, gravedad, aniquilacion) --
  cuáles cambian el conteo (actúan) y cuáles no. NOTA DE ALCANCE: este motor reescrito sólo implementa estas 4
  piezas (no las 23 del inventario viejo) -- es el rediseño mínimo para probar que las FUERZAS deciden, no el
  campo térmico. Se reporta así, sin inflar el alcance.
CERO AZAR (no se toca el motor, sólo se llaman sus funciones).
"""
import sys, time, json
sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
import cs072_motor_fuerzas as m

t0 = time.time()
log_lines = []


def p(s):
    print(s, flush=True)
    log_lines.append(s)


BASE = (30, 21, 10, 7)  # n_quarks, n_antiquarks, n_electrones, n_positrones -- N=68


def poblacion(scale):
    return tuple(int(round(x * scale)) for x in BASE)


def bariones(nq, naq, ne, npz, pasos, apagar=frozenset()):
    r = m.corre(nq, naq, ne, npz, homogeneo=False, expansion=True, pasos=pasos, apagar=apagar)
    return m.cuenta_bariones(r)["bariones"]


def pasos_equilibrados(nq, naq, ne, npz, candidatos):
    conteos = []
    for pasos in candidatos:
        b = bariones(nq, naq, ne, npz, pasos)
        p(f"    [equilibrio] pasos={pasos} bariones(con confin)={b}")
        conteos.append((pasos, b))
        if len(conteos) >= 2 and conteos[-1][1] == conteos[-2][1]:
            return conteos[-2][0], True
    return candidatos[-1], False


ESCALAS_N = [68, 136, 272, 544]
CANDIDATOS_PASOS = {
    68: [300, 400, 500],
    136: [300, 400, 500],
    272: [300, 400, 500],
    544: [300, 400, 500],
}

resultados = {"punto0_pasos_equilibrados": {}, "tarea1_confinamiento_apagado": {},
              "tarea2_escala_n_quarks_tercio": {}, "tarea3_auditoria": {}}

p("=" * 100)
p("PUNTO 0 -- pasos de equilibrio por N (brazo D, con confinamiento)")
p("=" * 100)
pasos_por_N = {}
for N in ESCALAS_N:
    nq, naq, ne, npz = poblacion(N / 68)
    p(f"  N={N} (quarks={nq}, antiquarks={naq}, electrones={ne}, positrones={npz})")
    pasos_eq, estable = pasos_equilibrados(nq, naq, ne, npz, CANDIDATOS_PASOS[N])
    pasos_por_N[N] = pasos_eq
    resultados["punto0_pasos_equilibrados"][N] = {"pasos": pasos_eq, "estabilizo": estable}
    p(f"  -> pasos equilibrados N={N}: {pasos_eq} (estabilizó: {estable})")
p(f"(t={(time.time()-t0)/60:.2f} min)")

p("")
p("=" * 100)
p("TAREA 1 -- apagar confinamiento debe dar 0 bariones a TODA escala")
p("=" * 100)
todas_dan_cero = True
for N in ESCALAS_N:
    nq, naq, ne, npz = poblacion(N / 68)
    pasos = pasos_por_N[N]
    b_con = bariones(nq, naq, ne, npz, pasos)
    b_sin = bariones(nq, naq, ne, npz, pasos, apagar={"confinamiento"})
    admisible = (b_sin == 0)
    todas_dan_cero = todas_dan_cero and admisible
    resultados["tarea1_confinamiento_apagado"][N] = {"pasos": pasos, "con_confinamiento": b_con,
                                                       "sin_confinamiento": b_sin, "admisible": admisible}
    p(f"  N={N:>4} (pasos={pasos}): con_confinamiento={b_con:>3}  sin_confinamiento={b_sin:>3}  "
      f"{'ADMISIBLE' if admisible else '*** NO ADMISIBLE -- ARTEFACTO REAPARECIÓ ***'}")
p(f"  VEREDICTO TAREA 1: {'TODAS LAS ESCALAS DAN 0 SIN CONFINAMIENTO -- ADMISIBLE' if todas_dan_cero else '*** FALLA EN ALGUNA ESCALA -- NO ADMISIBLE ***'}")
p(f"(t={(time.time()-t0)/60:.2f} min)")

p("")
p("=" * 100)
p("TAREA 2 -- escala: ¿bariones ~ n_quarks/3?")
p("=" * 100)
for N in ESCALAS_N:
    nq, naq, ne, npz = poblacion(N / 68)
    pasos = pasos_por_N[N]
    b_con = resultados["tarea1_confinamiento_apagado"][N]["con_confinamiento"]
    esperado = nq / 3.0
    ratio = b_con / esperado if esperado > 0 else None
    resultados["tarea2_escala_n_quarks_tercio"][N] = {"n_quarks": nq, "bariones": b_con,
                                                        "n_quarks_div3": esperado, "ratio": ratio}
    p(f"  N={N:>4}: n_quarks={nq:>3}  bariones={b_con:>3}  n_quarks/3={esperado:>6.2f}  ratio={ratio:.3f}")
p(f"(t={(time.time()-t0)/60:.2f} min)")

p("")
p("=" * 100)
p("TAREA 3 -- auditoría de apagado (confinamiento, em, gravedad, aniquilacion) -- N=68 y N=544")
p("=" * 100)
PIEZAS = ["confinamiento", "em", "gravedad", "aniquilacion"]
for N in (68, 544):
    nq, naq, ne, npz = poblacion(N / 68)
    pasos = pasos_por_N[N]
    base = bariones(nq, naq, ne, npz, pasos)
    p(f"  N={N} (pasos={pasos}) -- base (todas activas): {base} bariones")
    fila = {"base": base, "piezas": {}}
    for pieza in PIEZAS:
        b = bariones(nq, naq, ne, npz, pasos, apagar={pieza})
        actua = (b != base)
        fila["piezas"][pieza] = {"bariones_sin": b, "actua": actua}
        p(f"    sin {pieza:15s}: {b:>3} bariones  {'ACTÚA' if actua else 'NO ACTÚA (apagarla no cambió nada)'}")
    resultados["tarea3_auditoria"][N] = fila
p(f"(t={(time.time()-t0)/60:.2f} min)")

p("")
p("=" * 100)
p(f"TIEMPO TOTAL: {(time.time()-t0)/60:.2f} min")
p("=" * 100)

with open("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs072_v10_motor_fuerzas_escala_resultados.json", "w") as f:
    json.dump(resultados, f, indent=2, default=str)

with open("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs072_v10_motor_fuerzas_escala_log.txt", "w") as f:
    f.write("\n".join(log_lines))
