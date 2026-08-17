"""
CS072 -- TAREA 1 y 2 de INSTRUCCION_CS072_cerrar_reservaB_umbral_escala_PARA_CC.md
TAREA 1: barrido fino de amplitud de gradiente por N in {68,136,272,544}, hallar amplitud_critica(N)
  (primer valor con bariones>0 SOSTENIDO -- el siguiente punto también >0, para no confundir ruido con umbral),
  y probar si amplitud_critica(N) * N ~= constante (tesis #2 en su forma "cantidad total de diferencias").
TAREA 2: caracterizar la banda de memoria (MEMORIA_ALPHA=0.5..0.99) -- dónde demasiada memoria apaga la materia.
PUNTO 0 (guardián de CS): hallar pasos de equilibrio POR CADA N antes de barrer -- no adjudicar sobre corridas
  no equilibradas. Se mide en el régimen SATURADO (amplitud=1.0, donde ya sabemos que hay bariones) subiendo
  pasos hasta que el conteo deje de cambiar entre dos candidatos consecutivos.
Todo determinista (G-CERO-AZAR intacto -- no se toca el motor, sólo se llaman sus funciones existentes).
"""
import sys, time, json
sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
import cs072_fold_completo as m

t0 = time.time()
log_lines = []


def p(s):
    print(s, flush=True)
    log_lines.append(s)


BASE = (30, 21, 10, 7)  # n_quarks, n_antiquarks, n_electrones, n_positrones -- suma N=68


def poblacion(scale):
    return tuple(int(round(x * scale)) for x in BASE)


def bariones_D(n_quarks, n_aq, n_e, n_p, pasos, amplitud=None, alpha=None, tasa_exp=None):
    """Corre el brazo D (homogeneo=False, expansion=True -- la cadena del director) con los globales
    del motor temporalmente sobreescritos, y devuelve el conteo de bariones medidos al final."""
    orig = (m.MEMORIA_ALPHA, m.TASA_EXPANSION_GLOBAL, m.GRADIENTE_TERMICO_AMPLITUD)
    if amplitud is not None:
        m.GRADIENTE_TERMICO_AMPLITUD = amplitud
    if alpha is not None:
        m.MEMORIA_ALPHA = alpha
    if tasa_exp is not None:
        m.TASA_EXPANSION_GLOBAL = tasa_exp
    try:
        r = m.corre_proceso_unico(n_quarks, n_aq, n_e, n_p, pasos=pasos, homogeneo=False, expansion=True)
        at = m.cuenta_bariones_e_hidrogeno(r)
        return at["bariones_medidos"]
    finally:
        m.MEMORIA_ALPHA, m.TASA_EXPANSION_GLOBAL, m.GRADIENTE_TERMICO_AMPLITUD = orig


def pasos_equilibrados(n_quarks, n_aq, n_e, n_p, candidatos):
    """Régimen saturado (amplitud=1.0) en pasos crecientes; el primer candidato cuyo conteo COINCIDE con el
    siguiente se declara estable (devuelve ese pasos, el menor de los dos que ya coinciden). Si nunca
    coincide, devuelve el último candidato probado y lo DECLARA no-estabilizado (no lo esconde)."""
    conteos = []
    for pasos in candidatos:
        b = bariones_D(n_quarks, n_aq, n_e, n_p, pasos, amplitud=1.0)
        p(f"    [equilibrio] pasos={pasos} bariones(amp=1.0,saturado)={b}")
        conteos.append((pasos, b))
        if len(conteos) >= 2 and conteos[-1][1] == conteos[-2][1]:
            return conteos[-2][0], True
    return candidatos[-1], False


ESCALAS_N = [68, 136, 272, 544]
CANDIDATOS_PASOS = {
    68: [300, 400, 600],
    136: [400, 600, 800],
    272: [400, 600, 800, 1000],
    544: [400, 600, 800, 1000],
}
AMPLITUDES = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
ALPHAS_MEMORIA = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

resultados = {"punto0_pasos_equilibrados": {}, "tarea1_amplitud_critica": {}, "tarea2_banda_memoria": {}}

p("=" * 100)
p("PUNTO 0 -- HALLANDO PASOS DE EQUILIBRIO POR N (régimen saturado, amplitud=1.0)")
p("=" * 100)
pasos_por_N = {}
for N in ESCALAS_N:
    nq, naq, ne, npz = poblacion(N / 68)
    p(f"  N={N} (quarks={nq}, antiquarks={naq}, electrones={ne}, positrones={npz})")
    pasos_eq, estable = pasos_equilibrados(nq, naq, ne, npz, CANDIDATOS_PASOS[N])
    pasos_por_N[N] = pasos_eq
    resultados["punto0_pasos_equilibrados"][N] = {"pasos": pasos_eq, "estabilizo": estable}
    p(f"  -> pasos equilibrados para N={N}: {pasos_eq} (estabilizó: {estable})")
p(f"(t={(time.time()-t0)/60:.2f} min)")

p("")
p("=" * 100)
p("TAREA 1 -- amplitud_critica(N) y producto amplitud_critica x N")
p("=" * 100)
tabla_amplitud = {}
for N in ESCALAS_N:
    nq, naq, ne, npz = poblacion(N / 68)
    pasos = pasos_por_N[N]
    p(f"  N={N} (pasos={pasos}):")
    fila = []
    for amp in AMPLITUDES:
        b = bariones_D(nq, naq, ne, npz, pasos, amplitud=amp)
        fila.append((amp, b))
        p(f"    amplitud={amp:.2f} -> bariones={b}")
    resultados["tarea1_amplitud_critica"][N] = {"pasos": pasos, "barrido": fila}
    # amplitud_critica = primer valor con bariones>0 Y el siguiente valor de la lista TAMBIÉN >0 (sostenido)
    amp_critica = None
    for idx in range(len(fila) - 1):
        if fila[idx][1] > 0 and fila[idx + 1][1] > 0:
            amp_critica = fila[idx][0]
            break
    if amp_critica is None and fila[-1][1] > 0:
        amp_critica = fila[-1][0]  # sólo el último punto enciende -- se declara igual, sin "sostenido" a confirmar
    tabla_amplitud[N] = amp_critica
    prod = None if amp_critica is None else amp_critica * N
    p(f"  -> amplitud_critica(N={N}) = {amp_critica} | producto amplitud_critica*N = {prod}")
p(f"(t={(time.time()-t0)/60:.2f} min)")

p("")
p("  TABLA RESUMEN amplitud_critica(N) x N:")
p("  " + "-" * 60)
for N in ESCALAS_N:
    ac = tabla_amplitud[N]
    prod = None if ac is None else round(ac * N, 3)
    p(f"  N={N:>4}  amplitud_critica={ac}  producto={prod}")
resultados["tarea1_tabla_resumen"] = {str(N): {"amplitud_critica": tabla_amplitud[N],
                                                "producto": (None if tabla_amplitud[N] is None
                                                             else tabla_amplitud[N] * N)}
                                       for N in ESCALAS_N}

p("")
p("=" * 100)
p("TAREA 2 -- banda de memoria (N=68, pasos equilibrados, amplitud=1.0 saturada)")
p("=" * 100)
nq, naq, ne, npz = poblacion(1.0)
pasos68 = pasos_por_N[68]
fila_mem = []
for alpha in ALPHAS_MEMORIA:
    b = bariones_D(nq, naq, ne, npz, pasos68, amplitud=1.0, alpha=alpha)
    fila_mem.append((alpha, b))
    p(f"  alpha={alpha:.2f} -> bariones={b}")
resultados["tarea2_banda_memoria"] = {"N": 68, "pasos": pasos68, "barrido": fila_mem}
p(f"(t={(time.time()-t0)/60:.2f} min)")

p("")
p("=" * 100)
p(f"TIEMPO TOTAL: {(time.time()-t0)/60:.2f} min")
p("=" * 100)

with open("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs072_v9_umbral_escala_resultados.json", "w") as f:
    json.dump(resultados, f, indent=2, default=str)

with open("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs072_v9_umbral_escala_log.txt", "w") as f:
    f.write("\n".join(log_lines))
