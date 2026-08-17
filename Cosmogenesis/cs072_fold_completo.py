"""
CS072 -- EL PROCESO ÚNICO (v5, INSTRUCCION_CS072_gradiente_termico_expansion_DEFINITIVA_PARA_CC.md). UN
SOLO BUCLE: en cada paso de tiempo, TODAS las fuerzas actúan JUNTAS sobre el MISMO estado (W = red de
afinidad, V = marco de orientación, V3 = marco de 3 ejes), alimentándose entre sí. NO hay Parte A / Parte B.

PREMISA DEL DIRECTOR (fijada, no se debate): la explosión inicial estaba extremadamente caliente, pero su
temperatura NO era uniforme -- había gradiente térmico. La expansión ocurrió más rápido que la
rehomogeneización: las diferencias quedaron PRESERVADAS y se AMPLIFICARON. Esa es la ÚNICA asimetría inicial
admisible (v4 sembraba W0 desde color/carga/masa -- eso ERA la "traducción" que el director identificó como
la causa raíz de los fracasos anteriores).

QUÉ CAMBIÓ (v4->v5, fiel a la fórmula EXACTA verificada por CS en cs072_toy_gradiente_termico_expansion.py,
patrón 1-1-4-8, invariancia dura con diferencia 0.00 -- reusada literal, sin traducir):
- ESTADO INICIAL: cada entidad sobreviviente es también una PARCELA con su propia temperatura T (campo, no
  escalar). Gradiente suma-cero (misma media y misma "energía" que el control homogéneo, sólo cambia la
  distribución): d=linspace(-0.1,0.1,N)-media; T=T_MED_TERMICO+d. Control homogéneo: T=T_MED_TERMICO parejo.
  CERO AZAR: es una función determinista de la posición, no un sorteo.
- W NACE EN CERO (ya no se siembra desde color/carga/masa -- "la única asimetría inicial admisible es la del
  campo de temperatura"). W se construye EXCLUSIVAMENTE de la historia térmica compartida: afinidad térmica
  aff=exp(-dT/T.mean()) + memoria W=0.9*W+0.1*aff, cada paso -- la pieza "memoria" (CS071) DEJA de ser un
  refuerzo-por-roce ad-hoc y ES, literal, esta fórmula.
- EXPANSIÓN (#9/#18): tasa GLOBAL única (0.02, la MISMA para todas las parcelas) que enfría MÁS lo que ya
  está frío -- T=T*(1-0.02*(T.max()-T)/T.max()) -- amplifica el contraste térmico en vez de emparejarlo. Ya
  NO es el recorte-por-distancia-relacional de v4 (eso era una segunda "expansión" inventada, redundante con
  ésta -- se retira, queda UNA sola tasa global, como manda G-NO-PARAMETRO-FORMA).
- Gravedad (#2) y confinamiento (#3) seguían gateadas por un escalar "temperatura" sintético (_T_de_paso,
  atado a un reloj de pasos arbitrario). Ahora leen la temperatura REAL: T_efectivo_global se deriva de
  cuánto se enfrió de verdad el campo (T.mean()/T_MED_TERMICO), no de un cronómetro inventado -- la misma
  pieza, ahora ejecutando sobre física real en vez de un calendario decorativo.
- Los otros 15 elementos (EM, débil, Pauli, SSB, 3-cuerpos, correlación, causal, localidad, aniquilación por
  población Motor B) NO cambian de mecanismo -- siguen actuando en el MISMO bucle sobre el W que ahora nace
  térmico. Cero azar (G-CERO-AZAR) intacto: ninguna función llama a RNG.

v6 (ADJUDICACION_CS072_motor_v5_NO_ADMISIBLE_CS.md -- CS corrió el motor y encontró 7 incumplimientos; CC
corrige los 7, en orden):
- (a) el marco inicial (V, V3) se sembraba CÍCLICO POR ÍNDICE (np.arange(N) % K) -- el índice entraba ANTES
  del voto de marco. Ahora _marco_inicial_por_atributos deriva eje/signo de PROPIEDADES FÍSICAS (color,
  carga, es_anti, es_quark) -- entidades idénticas arrancan con el MISMO marco, el empate se conserva.
- (b) la aniquilación (Motor B) se resolvía por POBLACIÓN pero ANTES del bucle -- rompía la co-emergencia.
  Ahora la población COMPLETA (materia+antimateria) se instancia desde t=0 y la aniquilación corre DENTRO
  de paso_unico cada paso, vía un peso continuo `viva` por clase (color+estatus), nunca por individuo.
- (c) MEMORIA_ALPHA/TASA_EXPANSION_GLOBAL/GRADIENTE_TERMICO_AMPLITUD copiados literales del toy sin más
  justificación que "son los del toy" -- ahora se barren como observables (barrido_sensibilidad_parametros_
  termicos) para confirmar que el patrón no depende del valor exacto.
- (d)+(f) "sólo D enciende" comparaba D sólo contra A/B, nunca contra C, y el barrido de N sólo corría D --
  ahora test_cuatro_brazos y barrido_N_diametro comparan las 4 combinaciones explícitamente, en cada escala.
- (e) se mide hidrógeno/bariones ANTES de llamar "espacio" al diámetro -- si no hay átomos persistentes, se
  reporta así, no como un positivo.
- (g) el argmax de _voto_marco rompía empates por posición del array cuando dos estados quedaban a la
  cabeza por ruido de punto flotante (de sumar en otro orden al permutar) -- ahora el empate se CONSERVA
  (se mantiene el estado actual) salvo que haya un ganador CLARO, fuera de tolerancia numérica.
Con (a)+(g) juntos la invariancia dura pasa a precisión de máquina (~1e-16) incluso a 60 pasos y N=68,
verificado corriendo el código, no citado.

Codea/ejecuta: CC. Diseño/ruling: CS + director + Codex.
"""
from __future__ import annotations
import math
from collections import deque
import numpy as np

# ============================ CONSTANTES (heredadas del arco, G-NO-CALIBRAR) ============================
R_GRAV, R_STRONG, R_EM = 0.10, 0.10, 0.10
T_HI, T_LO, T_CONF = 3.0, 0.04, 1.0
STEPS = 20
DMAX_INT = 8
K_FRAME = 6
PESO_SEMILLA = 0.10
MASA_QUARK = 1.0
MASA_ELECTRON = 1.0 / (3.0 * 1836.15)   # ratio REAL electrón/protón (protón ~ 3 quarks)
G_FUERTE_QCD = R_STRONG   # #22: reusa la constante estructural YA existente de la fuerza fuerte -- no una perilla nueva

# ============================ CAMPO DE TEMPERATURA (premisa del director, v5) ============================
# T_MED_TERMICO=1.0 y el gradiente ±0.1 son EXACTAMENTE los del toy de CS (cs072_toy_gradiente_termico_
# expansion.py) que dio el patrón 1-1-4-8 e invariancia dura 0.00 -- no se reinventa la escala: el mecanismo
# (aff=exp(-dT/T.mean()), expansión=T*(1-0.02*(Tmax-T)/Tmax)) es matemáticamente INVARIANTE a reescalar T de
# forma uniforme (dT/T.mean() y (Tmax-T)/Tmax no cambian si T->c*T), así que no hay "traducción" posible aquí:
# cualquier escala que se hubiera usado da el mismo patrón relativo. Se usa la escala literal del toy.
T_MED_TERMICO = 1.0
GRADIENTE_TERMICO_AMPLITUD = 0.1     # ±0.1 alrededor de la media, suma-cero (misma energía que homogéneo)
TASA_EXPANSION_GLOBAL = 0.02         # ÚNICA tasa, global, la del toy -- nunca una posición por parcela
MEMORIA_ALPHA = 0.9                  # W_nuevo = 0.9*W + 0.1*aff -- el mecanismo de memoria térmica, literal


def _T_efectivo_global(T_campo):
    """Gravedad (#2) y confinamiento (#3) necesitan un escalar 'qué tan frío está el universo ahora' para
    gatear su acoplamiento temporal -- ANTES ese escalar salía de un cronómetro sintético (_T_de_paso, atado
    a HORIZONTE_ENFRIAMIENTO, un reloj inventado). AHORA se deriva de cuánto se enfrió REALMENTE el campo de
    temperatura: frac_enfriado mide la caída real de la media del campo respecto de su media inicial. Sin
    expansión el campo no se enfría nunca (frac=0, igual que antes de que existiera algo que enfriar). Misma
    forma exponencial T_HI->T_LO que ya existía (T_HI, T_LO, T_CONF no cambian) -- sólo cambió la FUENTE del
    dato: física real medida, no un calendario."""
    frac_enfriado = max(0.0, 1.0 - float(T_campo.mean()) / T_MED_TERMICO)
    return T_HI * (T_LO / T_HI) ** min(1.0, frac_enfriado)


def _campo_temperatura_inicial(N, homogeneo):
    """LA ÚNICA asimetría inicial admisible (instrucción definitiva del director): un campo de temperatura,
    no una semilla/masa/densidad/etiqueta. Gradiente suma-cero -- misma media y misma energía total que el
    control homogéneo, sólo cambia la DISTRIBUCIÓN. Determinista (G-CERO-AZAR): función fija de la posición,
    ningún sorteo. Fórmula literal del toy de CS."""
    if homogeneo or N <= 1:
        return np.full(N, T_MED_TERMICO, dtype=np.float64)
    d = np.linspace(-GRADIENTE_TERMICO_AMPLITUD, GRADIENTE_TERMICO_AMPLITUD, N)
    d = d - d.mean()
    return T_MED_TERMICO + d


# ============================ ANIQUILACIÓN POR POBLACIÓN (Motor B) -- resuelve el sesgo de índice de raíz =====
def resuelve_poblacion_por_aniquilacion(n_quarks, n_antiquarks, n_electrones, n_positrones):
    """MOTOR B (INSTRUCCION_CS072_motorB_poblacion_definitivo_PARA_CC.md v2): la aniquilación NO pregunta
    'cuál' individuo sobrevive -- eso exige desempatar entidades físicamente idénticas, y sin azar (G-CERO-
    AZAR) el desempate sólo puede salir del orden del array = sesgo de índice (verificado, no invariante).
    En vez de eso: aniquilación = RESTA DE POBLACIONES por color. Quarks/antiquarks se reparten CÍCLICOS
    por color (0,1,2 -- la MISMA convención de siempre); por cada color c sobrevive max(0,n_q[c]-n_aq[c])
    quarks O max(0,n_aq[c]-n_q[c]) antiquarks, nunca ambos. Electrones/positrones: sin color, un solo
    conteo. Invariante POR CONSTRUCCIÓN -- no hay ningún individuo que ordenar, sólo conteos.

    OJO (corrección de CS, no asumir): que el residuo quede BALANCEADO en color (y por tanto cierre
    bariones) depende del catálogo inicial, NO es automático -- es un OBSERVABLE que se mide después
    (ver `balance_color` en el resultado), nunca una garantía de esta función."""
    n_q_color = np.bincount(np.arange(n_quarks) % 3, minlength=3) if n_quarks else np.zeros(3, dtype=int)
    n_aq_color = np.bincount(np.arange(n_antiquarks) % 3, minlength=3) if n_antiquarks else np.zeros(3, dtype=int)
    n_q_survive = np.maximum(0, n_q_color - n_aq_color)
    n_aq_survive = np.maximum(0, n_aq_color - n_q_color)
    n_e_survive = max(0, n_electrones - n_positrones)
    n_p_survive = max(0, n_positrones - n_electrones)
    balance_color = bool(np.all(n_q_survive == n_q_survive[0]) or n_q_survive.sum() == 0)
    return dict(n_q_color=n_q_color, n_aq_color=n_aq_color, n_q_survive=n_q_survive,
                n_aq_survive=n_aq_survive, n_e_survive=n_e_survive, n_p_survive=n_p_survive,
                balance_color=balance_color)


# ============================ ESTADO INICIAL: POBLACIÓN COMPLETA (v7, materia Y antimateria) ============================
def _entidades_deterministas_full(n_quarks, n_antiquarks, n_electrones, n_positrones, homogeneo=False):
    """v6 (CS, ADJUDICACION_..._NO_ADMISIBLE_v5 punto b): instancia la población COMPLETA -- materia Y
    antimateria, electrones Y positrones -- como nodos desde t=0. La aniquilación YA NO se resuelve antes
    (Motor B precomputado rompía la co-emergencia); corre DENTRO del bucle (paso_unico, #8) por población
    continua (`viva`), nunca por individuo. Dentro de un mismo color/tipo/estatus son mutuamente
    INDISTINGUIBLES -- ninguna propiedad extra que los diferencie (Shannon). La carga alternada up/down
    dentro de un bloque de color es una propiedad FÍSICA legítima, no identidad individual. NINGÚN rng.

    v7 (ADJUDICACION_..._NO_ADMISIBLE_v6 punto 2, VERIFICADO por CS): construir por BLOQUES de especie
    (todos los quarks, luego antiquarks, luego electrones, luego positrones) hacía que la temperatura
    (asignada por POSICIÓN, ver _campo_temperatura_inicial) quedara CORRELACIONADA con la especie -- CS midió
    T media por especie tras la construcción vieja: 0.943/1.019/1.066/1.091, monótona por bloque. El índice
    quedó disfrazado de temperatura. CORRECCIÓN: las 4 categorías se INTERCALAN round-robin (q,aq,e,p,q,aq,
    e,p,...) -- la posición global YA NO se alinea con la especie, así que T (asignada sobre esa posición)
    deja de correlacionar con el tipo de partícula. El color/carga dentro de cada categoría sigue su propio
    contador INTERNO (no la posición global), así que el ciclo de color no se ve afectado por el intercalado.

    Añade el atributo SABOR (0=up-type, 1=down-type), separado de COLOR (ver ESPECIFICACION_CS072_debil_
    cambia_sabor_no_color_PARA_CC.md) -- la débil (#5) muta sabor+carga, JAMÁS color.

    CADA entidad es TAMBIÉN una parcela térmica (T, campo determinista, ver _campo_temperatura_inicial) --
    es la ÚNICA asimetría inicial admisible -- y arranca con viva=1.0 (nada aniquilado todavía)."""
    N = n_quarks + n_antiquarks + n_electrones + n_positrones
    color = np.full(N, -1, dtype=np.int8)
    carga = np.zeros(N, dtype=np.int8)
    sabor = np.zeros(N, dtype=np.int8)
    masa = np.zeros(N, dtype=np.float64)
    es_anti = np.zeros(N, dtype=bool)
    es_quark = np.zeros(N, dtype=bool)

    restantes = dict(q=n_quarks, aq=n_antiquarks, e=n_electrones, p=n_positrones)
    contados = dict(q=0, aq=0, e=0, p=0)
    orden_categorias = ("q", "aq", "e", "p")   # rotación FIJA -- determinista, no depende de las cantidades
    pos = 0; k = 0
    while pos < N:
        cat = orden_categorias[k % 4]
        k += 1
        if contados[cat] >= restantes[cat]:
            continue
        j = contados[cat]   # índice DENTRO de la categoría (nunca la posición global) -- el ciclo de color
                             # / la alternancia de carga siguen intactos, sólo cambia DÓNDE cae en el array
        if cat == "q":
            color[pos] = j % 3
            carga[pos] = 2 if j % 2 == 0 else -1
            sabor[pos] = 0 if j % 2 == 0 else 1
            masa[pos] = MASA_QUARK; es_quark[pos] = True
        elif cat == "aq":
            color[pos] = j % 3
            carga[pos] = -2 if j % 2 == 0 else 1
            sabor[pos] = 0 if j % 2 == 0 else 1
            masa[pos] = MASA_QUARK; es_quark[pos] = True; es_anti[pos] = True
        elif cat == "e":
            carga[pos] = -3; masa[pos] = MASA_ELECTRON
        else:  # "p"
            carga[pos] = 3; masa[pos] = MASA_ELECTRON; es_anti[pos] = True
        contados[cat] += 1
        pos += 1

    es_ferm = np.ones(N, dtype=bool)
    T = _campo_temperatura_inicial(N, homogeneo)
    viva = np.ones(N, dtype=np.float64)
    return dict(color=color, carga=carga, sabor=sabor, masa=masa, es_anti=es_anti, es_ferm=es_ferm,
                es_quark=es_quark, T=T, viva=viva, n_quarks=n_quarks, n_antiquarks=n_antiquarks,
                n_electrones=n_electrones, n_positrones=n_positrones, N=N)


def _null_catalogo(cat):
    """Control DETERMINISTA (G-NULL-CATALOGO): mismo reparto fijo, EMPAREJAMIENTO desplazado -- cada
    propiedad se corre un offset fijo distinto, misma población y magnitudes, sin una sola llamada a azar."""
    cat2 = dict(cat)
    for k, off in {"color": 1, "carga": 2, "masa": 3, "es_anti": 1, "es_ferm": 1, "es_quark": 1, "T": 2,
                   "viva": 1, "sabor": 3}.items():
        cat2[k] = np.roll(cat[k], off)
    return cat2


def _marco_inicial_por_atributos(K, color, carga, es_anti, es_quark):
    """v6 (CS, ADJUDICACION_..._NO_ADMISIBLE): v5 asignaba el eje inicial CÍCLICO POR ÍNDICE (np.arange(N) %
    K) -- el índice del array entraba en el estado ANTES de que corriera el voto de marco, sembrando una
    posición que el voto sólo heredaba. Aquí el eje/signo inicial es una función de PROPIEDADES FÍSICAS
    (color, carga, estatus de materia, tipo) -- nunca de la posición. Dos entidades con el MISMO
    (color, carga, es_anti, es_quark) son físicamente indistinguibles y arrancan con el MISMO vector de
    marco -- el empate se CONSERVA simétrico; sólo una diferencia física real (su T, su historia de W) puede
    romperlo después. `np.arange(N)` abajo es sólo el mecanismo de indexado fila-a-fila de numpy (empareja
    la fila i con SU PROPIO eje/signo, derivados de sus propios atributos) -- no es una fuente de asimetría."""
    N = len(color)
    clave = ((color.astype(np.int64) + 1) + 4 * es_quark.astype(np.int64) +
              8 * es_anti.astype(np.int64) + 16 * (carga.astype(np.int64) > 0).astype(np.int64))
    eje0 = (clave % K).astype(np.int64)
    signo0 = np.where(carga >= 0, 1.0, -1.0)
    V = np.zeros((N, K), dtype=np.float64)
    V[np.arange(N), eje0] = signo0
    return V


# NOTA v4->v5: aquí vivía _afinidad_fisica_inicial, que sembraba W0 desde color/carga/masa (confinamiento+EM+
# gravedad) ANTES de que corriera un solo paso. Ésa era, literalmente, la "traducción" que la instrucción
# definitiva identificó como causa raíz de los fracasos: la física entraba como ENTRADA precalculada, no como
# consecuencia. v5 retira la función -- W arranca en np.zeros((N,N)) sin excepción (ver corre_proceso_unico).
# La ÚNICA asimetría inicial admisible es la del campo de temperatura T (_campo_temperatura_inicial).


# ============================ EL BUCLE ÚNICO: todas las fuerzas juntas, cada paso ============================
def paso_unico(estado, step, pasos, apagar=frozenset()):
    """UN SOLO PASO. Todos los deltas se calculan sobre el estado de ENTRADA (el del paso anterior) y se
    aplican juntos al final -- efecto simultáneo, sin cascada. Devuelve el estado nuevo completo.

    v5: la base de W ya NO es la W de entrada sola -- nace/se sostiene de la HISTORIA TÉRMICA (afinidad por
    temperatura + memoria, fórmula exacta del toy de CS), y las demás fuerzas se suman sobre esa base, en el
    MISMO paso. El campo T evoluciona al final del paso por la expansión (tasa global, enfría más lo frío).

    `apagar`: set de nombres de piezas a NEUTRALIZAR este paso (ADJUDICACION_..._MATERIA_EMERGE_CS.md v2,
    punto d: "una pieza cuyo apagado no cambia nada NO está actuando"). Vacío por defecto -- TODAS actúan.
    Nombres válidos: gravedad, confinamiento, em, debil, aniquilacion, localidad, correlacion, causal,
    marco (voto SSB/#1/#16), pauli, tres_cuerpos, memoria_termica, qcd (#22)."""
    W = estado["W"]; V = estado["V"]; V3 = estado["V3"]
    color = estado["color"]; carga = estado["carga"]; masa = estado["masa"]; sabor = estado["sabor"]
    es_anti = estado["es_anti"]; es_ferm = estado["es_ferm"]; es_quark = estado["es_quark"]
    t_birth = estado["t_birth"]; s_prev = estado["s_prev"]
    T_campo = estado["T"]; expansion = estado["expansion"]; viva = estado["viva"]
    N = W.shape[0]

    # #10 enfriamiento + memoria térmica (mecanismo, CS071): fórmula EXACTA del toy verificado --
    # aff=exp(-dT/T.mean()), W=0.9*W+0.1*aff -- ésta ES la pieza "memoria", no un roce ad-hoc aparte.
    dT = np.abs(T_campo[:, None] - T_campo[None, :])
    aff_termico = np.exp(-dT / (T_campo.mean() + 1e-9))
    np.fill_diagonal(aff_termico, 0.0)
    if "memoria_termica" in apagar:
        W_termico = W.copy()   # sin memoria térmica: W no se alimenta de nada -- auditoría de la pieza
    else:
        W_termico = MEMORIA_ALPHA * W + (1.0 - MEMORIA_ALPHA) * aff_termico

    T_ef = _T_efectivo_global(T_campo)                      # #10 "qué tan frío está el universo", de la física real

    s = W_termico.sum(axis=1)
    s_bar = max(float(s.mean()), 1e-12)
    w0_ef = s_bar / max(N - 1, 1)

    color_distinto = (color[:, None] != color[None, :]) & (color[:, None] >= 0) & (color[None, :] >= 0)
    carga_opuesta = (carga[:, None] != 0) & (carga[None, :] != 0) & (np.sign(carga[:, None]) != np.sign(carga[None, :]))
    mismo_estatus = (es_anti[:, None] == es_anti[None, :])
    mismo_familia = (es_quark[:, None] == es_quark[None, :])   # quark~antiquark, NO quark~positrón
    opuesto_estatus = (~mismo_estatus) & mismo_familia

    # #22 (v7.2, ADJUDICACION_CS072_motor_v7_MATERIA_EMERGE_CS.md v2 punto b -- CS verificó que v7.1 sólo
    # MEDÍA masa_efectiva DESPUÉS del bucle, sin tocar la gravedad real: "declarada, no operante"):
    # generalización CONTINUA de la fórmula por-hadrón de la especificación (E_campo_QCD = g_fuerte * suma
    # de W sobre pares de un trío YA cerrado) a la RED completa -- cada quark carga su PROPIA energía de
    # campo, la suma de W hacia sus socios de confinamiento (color distinto, mismo estatus, LIGADOS) EN ESTE
    # PASO. No hace falta esperar a que el trío cierre: el campo gluónico es una propiedad de la relación,
    # no del hadrón ya identificado -- por eso pesa DESDE YA en la gravedad de este mismo paso, no sólo al
    # final. g_fuerte reusa R_STRONG (constante estructural, no un número nuevo). masa_efectiva_hadrones (al
    # final, post-hoc) sigue disponible para reportar el valor exacto por-hadrón una vez los tríos cierran.
    if "qcd" in apagar:
        masa_efectiva = masa                                     # #22 apagada: sólo masa de valencia
    else:
        ligado_confin_qcd = (W_termico > w0_ef * 1.5) & color_distinto & mismo_estatus
        E_campo_i = G_FUERTE_QCD * (W_termico * ligado_confin_qcd).sum(axis=1)
        masa_efectiva = masa + E_campo_i

    # #2 gravedad: refuerzo continuo por MASA EFECTIVA (valencia + campo QCD -- #22 actuando de verdad aquí,
    # no sólo medido al final), universal, todos gravitan igual, oscuros incluidos
    if "gravedad" in apagar:
        dW_grav = 0.0
    else:
        dW_grav = ((0.30 * R_GRAV * (1 - T_ef)) * np.outer(masa_efectiva, masa_efectiva)
                   / max(float(masa_efectiva.mean()) ** 2, 1e-300) * w0_ef)

    # #3 fuerte/confinamiento: colores distintos DEL MISMO estatus de materia (barión o antibarión) se pegan
    if "confinamiento" in apagar:
        dW_confin = 0.0
    else:
        dW_confin = R_STRONG * (color_distinto & mismo_estatus).astype(float) * w0_ef if T_ef < T_CONF else 0.0

    # #4 EM: carga opuesta atrae (esto acerca quark-antiquark, el canal que #8 usará); carga igual
    #    "comprimida" (por encima del peso típico) se repele -- frena el colapso, no erosiona lo sano
    if "em" in apagar:
        dW_em = 0.0; factor_em_repele = 1.0
    else:
        dW_em = R_EM * carga_opuesta.astype(float) * w0_ef
        mismo_carga = (carga[:, None] != 0) & (carga[None, :] != 0) & (np.sign(carga[:, None]) == np.sign(carga[None, :]))
        comprimido = W_termico > w0_ef
        factor_em_repele = np.where(mismo_carga & comprimido, 1.0 - 0.12, 1.0)

    # #5 débil (v7, ESPECIFICACION_CS072_debil_cambia_sabor_no_color_PARA_CC.md): cambia SABOR (up<->down,
    # decaimiento beta), JAMÁS color -- el color es carga de la fuerza FUERTE; la débil no la toca en la
    # naturaleza (sólo los GLUONES intercambian color, y siempre conservando el balance). v6 rotaba COLOR
    # periódicamente -- CS verificó que eso colapsa los 3 colores a 1 solo en ~20 pasos, y sin 3 colores
    # distintos NINGÚN barión puede cerrar (matemáticamente imposible). Se retira el reloj %20 (arbitrario,
    # G-NO-PARAMETRO-FORMA) junto con el error conceptual.
    #
    # GATILLO FÍSICO (no índice, no color, no reloj): un quark MAL LIGADO -- su fuerza de enlace total s_i
    # por debajo del PROMEDIO DE LOS QUARKS (no w0_ef) -- es inestable y decae, análogo a que un neutrón
    # LIBRE decae y uno ligado dentro de un núcleo no; en cuanto el quark queda genuinamente ligado, su s_i
    # sube por sobre el promedio y el criterio deja de cumplirse solo, sin tasa inventada.
    #
    # CORREGIDO (ADJUDICACION_CS072_motor_v7_MATERIA_EMERGE_CS.md v2, punto a -- CS instrumentó y confirmó
    # que la débil NUNCA actuaba): el criterio comparaba `s` (SUMA de fila, escala ~5-20) contra `w0_ef`
    # (escala 0.1-0.3, la media por PAR normalizada por N-1 -- el umbral que usan las piezas "ligado", una
    # magnitud DISTINTA). s nunca es menor que un número ~100x más chico -- el gatillo estaba mal escalado,
    # nunca se cumplía. Corregido: comparar contra el promedio de `s` ENTRE LOS QUARKS (misma magnitud,
    # misma unidad) -- da ~mitad de los quarks inestables en cada paso, y baja sola en cuanto un quark se
    # liga por sobre el promedio de sus pares.
    color_n = color; carga_n = carga.copy(); sabor_n = sabor.copy()   # color INTACTO -- la débil no lo toca
    if "debil" not in apagar:
        s_bar_quarks = max(float(s[es_quark].mean()), 1e-12) if es_quark.any() else 1e-12
        inestable = es_quark & (s < s_bar_quarks)
        if inestable.any():
            sabor_n[inestable] = 1 - sabor_n[inestable]
            carga_n[inestable] = np.where(carga_n[inestable] > 0, -1, 2).astype(np.int8)

    # #8 ANIQUILACIÓN (v6, ADJUDICACION_..._NO_ADMISIBLE punto b): v5 la resolvía por POBLACIÓN pero ANTES
    # del bucle (Motor B precomputado) -- eso rompía la co-emergencia: aniquilación y geometría deben salir
    # del MISMO proceso. Aquí sigue siendo por POBLACIÓN -- nunca por individuo -- pero ahora ES parte de
    # este paso: cada entidad tiene un peso continuo `viva` (1=viva, 0=aniquilada).
    #
    # CORREGIDO (encontrado corriendo el motor a 300+ pasos, no en la instrucción de CS): la primera versión
    # aplicaba el MISMO factor (1-frac) a AMBAS nubes por igual, cada paso -- eso compone geométricamente
    # (0.95^300 ~ 1e-7) y aniquila TAMBIÉN el excedente que debía sobrevivir (30 quarks vs 21 antiquarks
    # debía dejar ~9 quarks vivos; en cambio dejaba ~0 de TODOS). El error: no distinguía "cuánto queda de
    # la nube MÁS CHICA para consumir" de "cuánto hay en la nube más grande". Ahora el consumo de ESTE paso
    # está ACOTADO por la población viva de la nube más chica (nunca se puede consumir más parejas de las
    # que existen) y se reparte proporcional al tamaño de CADA nube -- así el excedente de la nube grande
    # deja de decaer en cuanto la nube chica se agota, igual que la resta de poblaciones de Motor B, sólo
    # que repartida en el tiempo en vez de resuelta de un golpe.
    viva_n = viva.copy()
    if "aniquilacion" not in apagar:
        umbral_aniq = w0_ef * 1.5
        ligado_aniq = (W_termico > umbral_aniq) & opuesto_estatus

        def _consume_clase(mask_a, mask_b):
            if not mask_a.any() or not mask_b.any():
                return
            cruce = ligado_aniq[np.ix_(mask_a, mask_b)]
            frac = float(cruce.mean()) if cruce.size else 0.0
            if frac <= 0.0:
                return
            viva_a = float(viva[mask_a].sum()); viva_b = float(viva[mask_b].sum())
            consumo = frac * min(viva_a, viva_b)      # nunca más parejas de las que existen en la nube chica
            if viva_a > 1e-12:
                viva_n[mask_a] *= max(0.0, 1.0 - consumo / viva_a)
            if viva_b > 1e-12:
                viva_n[mask_b] *= max(0.0, 1.0 - consumo / viva_b)

        for c in range(3):
            mask_q = es_quark & (~es_anti) & (color == c) & (viva > 1e-9)
            mask_aq = es_quark & es_anti & (color == c) & (viva > 1e-9)
            _consume_clase(mask_q, mask_aq)
        mask_e = (~es_quark) & (~es_anti) & (viva > 1e-9)
        mask_p = (~es_quark) & es_anti & (viva > 1e-9)
        _consume_clase(mask_e, mask_p)

    # #12 localidad: presupuesto continuo por nodo (techo, nunca infla). CORREGIDO: aplicar el recorte
    # COMPLETO de golpe (min(1,presupuesto/s)) colapsaba toda la red a 0 en ~4 pasos cuando se arranca denso
    # (todos conectados con todos desde la física inicial, s natural >> presupuesto por diseño -- se ESPERA
    # que la dinámica concentre el peso en pocos socios con el tiempo). Se aplica como PRESIÓN GRADUAL
    # (tasa_localidad por paso), igual de suave que el resto de las fuerzas (todas ~0.05-0.12/paso) -- no es
    # una pieza nueva, es la MISMA pieza ritmada para no ahogar al resto en un solo paso.
    if "localidad" in apagar:
        factor_local_nodo = np.ones(N)
    else:
        LOCAL_BUDGET_K = 6.0
        tasa_localidad = 0.08
        presupuesto = LOCAL_BUDGET_K * w0_ef
        objetivo_nodo = np.minimum(1.0, presupuesto / np.maximum(s, 1e-12))
        factor_local_nodo = 1.0 - tasa_localidad * (1.0 - objetivo_nodo)

    # #9/#18 expansión/inflación: YA NO es un recorte-por-distancia-relacional aparte (v4) -- ésa era una
    # SEGUNDA tasa de expansión inventada, y la instrucción exige UNA sola, global, física: la que enfría el
    # CAMPO DE TEMPERATURA (aplicada más abajo, al final del paso, sobre T_campo). Aquí no hay nada que hacer
    # por separado -- #9/#18 actúa sobre T, y su efecto sobre W entra por la vía térmica (W_termico), no por
    # un segundo canal directo sobre la red.

    # ---- combinar TODO sobre la base TÉRMICA (simultáneo) ----
    W_nuevo = (W_termico + dW_grav + dW_confin + dW_em) * factor_em_repele
    W_nuevo = W_nuevo * np.sqrt(np.outer(factor_local_nodo, factor_local_nodo))
    W_nuevo = W_nuevo * np.sqrt(np.outer(viva_n, viva_n))    # #8: lo aniquilado deja de pesar, gradual
    np.fill_diagonal(W_nuevo, 0.0)
    W_nuevo = np.clip(W_nuevo, 0.0, None)

    # #14 correlación: solape continuo de perfiles de afinidad (ya validado, actúa en peso_voto)
    if "correlacion" in apagar:
        corr = np.ones((N, N))
    else:
        norma = np.linalg.norm(W_nuevo, axis=1, keepdims=True)
        Wn = W_nuevo / np.maximum(norma, 1e-12)
        corr = np.clip(Wn @ Wn.T, 0.0, 1.0)

    # #15 cono causal: t_birth se fija cuando la fortaleza del nodo deja de cambiar; sólo relaciones
    #     causalmente conectadas (dentro del cono) pueden influirse en el voto de marco
    s_nuevo = W_nuevo.sum(axis=1)
    cambio = np.abs(s_nuevo - s_prev) / np.maximum(s_prev, 1e-9)
    recien_estable = (cambio < 0.02) & (t_birth >= pasos)
    t_birth_n = np.where(recien_estable, float(step), t_birth)
    if "causal" in apagar:
        mascara_causal = np.ones((N, N), dtype=bool)
    else:
        mascara_causal = np.abs(t_birth_n[:, None] - t_birth_n[None, :]) >= (1.0 / 1.5)

    peso_voto = W_nuevo * corr * mascara_causal.astype(float)

    # #1+#13+#16+semilla: voto de marco (SSB/Potts) con anti-voto de Pauli y sesgo de semilla, TODO junto
    if "marco" in apagar:
        V_nuevo = V; state_nuevo = np.argmax(np.abs(V[:, :K_FRAME]), axis=1)   # marco congelado, sin voto
    else:
        eje_semilla = np.zeros(N, dtype=int); signo_semilla = np.ones(N, dtype=int)
        V_nuevo, state_nuevo = _voto_marco(V, peso_voto, K_FRAME, es_ferm, PESO_SEMILLA, eje_semilla, signo_semilla)

    # #13 Pauli, SEGUNDA acción (además del anti-voto): penaliza DIRECTAMENTE la afinidad entre dos
    #     fermiones fuertemente ligados que YA ocupan el MISMO estado discreto de marco -- "impide que dos
    #     fermiones ocupen el mismo estado" actuando sobre la propia red W, no sólo sobre el voto. Umbral
    #     físico de ligadura -- el MISMO tipo de umbral que gobierna el resto del bucle (w0_ef), no un
    #     número nuevo de ajuste.
    if "pauli" not in apagar:
        umbral_ligado = w0_ef * 1.5
        mismo_estado_marco = (state_nuevo[:, None] == state_nuevo[None, :])
        ambos_fermion = es_ferm[:, None] & es_ferm[None, :]
        penaliza_pauli = mismo_estado_marco & ambos_fermion & (W_nuevo > umbral_ligado)
        np.fill_diagonal(penaliza_pauli, False)
        W_nuevo = np.where(penaliza_pauli, W_nuevo * 0.85, W_nuevo)

    # #11 vértice de 3 cuerpos: para cada nodo, sus 2 vecinos de MAYOR afinidad mueven los 3 marcos juntos
    #     (producto triple escalar, irreducible) -- determinista (argsort, no muestreo al azar)
    V3_nuevo = V3 if "tres_cuerpos" in apagar else _paso_3cuerpos(V3, W_nuevo, lr=0.15)

    # la memoria (CS071) YA actuó arriba, en W_termico = 0.9*W + 0.1*aff_termico -- no hay un segundo
    # mecanismo de refuerzo/decaimiento aparte (ESE doble-conteo era él mismo un parámetro sin respaldo físico).
    W_final = W_nuevo
    np.fill_diagonal(W_final, 0.0)

    # #9/#18 expansión, aplicada al CAMPO DE TEMPERATURA (única tasa global): enfría MÁS lo ya frío ->
    # amplifica el contraste térmico. Si el brazo no tiene expansión, T queda congelado (control B/D exige
    # que SÓLO el brazo con expansión encendida cambie esto).
    if expansion:
        T_max = max(float(T_campo.max()), 1e-12)
        T_nuevo = T_campo * (1.0 - TASA_EXPANSION_GLOBAL * (T_max - T_campo) / (T_max + 1e-9))
    else:
        T_nuevo = T_campo.copy()

    return dict(W=W_final, V=V_nuevo, V3=V3_nuevo, color=color_n, carga=carga_n, sabor=sabor_n, masa=masa,
                es_anti=es_anti, es_ferm=es_ferm, es_quark=es_quark, t_birth=t_birth_n, s_prev=s_nuevo,
                T=T_nuevo, expansion=expansion, viva=viva_n)


def _voto_marco(V, peso_voto, K, es_ferm, peso_semilla, eje_semilla, signo_semilla, inertia=0.3):
    N, D = V.shape
    P = V[:, :K]
    axis = np.argmax(np.abs(P), axis=1)
    sign = (P[np.arange(N), axis] >= 0).astype(int)
    state = (axis * 2 + sign).astype(int)
    estado_semilla = (eje_semilla * 2 + signo_semilla).astype(int)

    wsum_safe = np.maximum(peso_voto.sum(axis=1), 1e-12)
    counts = np.zeros((N, 2 * K))
    for k in range(2 * K):
        pertenece = (state == k).astype(float)
        counts[:, k] = (peso_voto * pertenece[None, :]).sum(axis=1) / wsum_safe
    peso_ferm = peso_voto * es_ferm.astype(float)[None, :]
    wsum_ferm = np.maximum(peso_ferm.sum(axis=1), 1e-12)
    for k in range(2 * K):
        pertenece = (state == k).astype(float)
        contrib = (peso_ferm * pertenece[None, :]).sum(axis=1) / wsum_ferm
        counts[:, k] -= np.where(es_ferm, 0.75 * contrib, 0.0)   # Pauli: fermiones anti-votan a fermiones
    counts[np.arange(N), state] += inertia
    counts[np.arange(N), estado_semilla] += peso_semilla         # semilla: sesgo mínimo, siempre presente

    # v6 (CS, ADJUDICACION_..._NO_ADMISIBLE, punto c del veredicto): np.argmax puro rompe EMPATES por
    # posición del array -- si dos o más estados quedan a la cabeza (tras permutar, ruido de punto flotante
    # de sumar en otro orden puede rozar ese empate), argmax elegía "el de menor índice", que es exactamente
    # el sesgo de índice que se viene persiguiendo toda la sesión. Aquí el empate se CONSERVA: sólo se
    # cambia de estado si hay un ganador CLARO (margen > TOL_EMPATE sobre el resto); si hay 2+ estados
    # empatados dentro de esa tolerancia, la entidad mantiene su estado ACTUAL -- no se elige por índice,
    # se espera a que una diferencia física real (T, historia de W) rompa el empate de verdad.
    TOL_EMPATE = 1e-9
    max_count = counts.max(axis=1, keepdims=True)
    es_maximo = counts >= (max_count - TOL_EMPATE)
    hay_ganador_claro = es_maximo.sum(axis=1) == 1
    ganador = np.argmax(counts, axis=1)
    new = np.where(hay_ganador_claro, ganador, state)
    ax = new // 2; sg = np.where(new % 2 == 1, 1.0, -1.0)
    Vn = np.zeros((N, D)); Vn[np.arange(N), ax] = sg
    return Vn, new


def _paso_3cuerpos(V3, W, lr):
    N = V3.shape[0]
    s3 = V3.copy()
    grad = np.zeros_like(s3)
    orden = np.argsort(-W, axis=1)
    for i in range(N):
        if W[i, orden[i, 0]] <= 0:
            continue
        j, k = int(orden[i, 0]), int(orden[i, 1])
        if W[i, k] <= 0:
            continue
        cr = np.cross(s3[j], s3[k])
        tp = float(np.dot(s3[i], cr))
        grad[i] += 2.0 * tp * cr
        grad[j] += 2.0 * tp * np.cross(s3[k], s3[i])
        grad[k] += 2.0 * tp * np.cross(s3[i], s3[j])
    s3 = s3 - lr * grad
    nrm = np.linalg.norm(s3, axis=1, keepdims=True)
    return s3 / np.maximum(nrm, 1e-12)


# ============================ CORRIDA COMPLETA: UN SOLO PROCESO ============================
def corre_proceso_unico(n_quarks, n_antiquarks, n_electrones, n_positrones, arm="real", pasos=STEPS,
                         permutacion=None, homogeneo=False, expansion=True, apagar=frozenset()):
    """arm ∈ {'real','null_catalogo'} (control de Shannon-por-índice, YA existente). homogeneo/expansion son
    los DOS interruptores de los 4 brazos de control de la instrucción definitiva:
      A: homogeneo=True,  expansion=False   -- sin ruptura esperada
      B: homogeneo=True,  expansion=True    -- sin ruptura esperada
      C: homogeneo=False, expansion=False   -- ruptura PARCIAL esperada
      D: homogeneo=False, expansion=True    -- ruptura COMPLETA esperada (la cadena del director)
    Arranca con W=0 (la ÚNICA asimetría inicial es el CAMPO DE TEMPERATURA); corre UN SOLO bucle (paso_unico)
    donde TODAS las fuerzas actúan juntas -- el residuo que sobrevive (bariones, hidrógeno) y la geometría
    (forma de la red) son AMBOS observables de SALIDA del mismo proceso.

    permutacion: array (N,) opcional -- reordenamiento FIJO y DETERMINISTA (nunca al azar) del CATÁLOGO
    COMPLETO (incluida su temperatura) antes de correr el proceso. `id_original` mapea cada posición a su
    identidad de origen.

    v6 (CS, ADJUDICACION_..._NO_ADMISIBLE): la población COMPLETA (materia+antimateria) se instancia desde
    t=0 -- la aniquilación (#8) corre DENTRO de paso_unico, por población continua (viva), nunca resuelta
    antes del bucle (eso rompía la co-emergencia). El marco inicial (V, V3) se deriva de ATRIBUTOS físicos,
    nunca del índice del array (ver _marco_inicial_por_atributos)."""
    cat = _entidades_deterministas_full(n_quarks, n_antiquarks, n_electrones, n_positrones, homogeneo=homogeneo)
    id_original = np.arange(cat["N"])
    if permutacion is not None:
        for k in ("color", "carga", "sabor", "masa", "es_anti", "es_ferm", "es_quark", "T", "viva"):
            cat[k] = cat[k][permutacion]
        id_original = id_original[permutacion]
    if arm == "null_catalogo":
        cat = _null_catalogo(cat)
    N = cat["N"]
    color, carga, masa = cat["color"], cat["carga"], cat["masa"]
    es_anti, es_ferm, es_quark = cat["es_anti"], cat["es_ferm"], cat["es_quark"]
    sabor0 = cat["sabor"]; T0 = cat["T"]; viva0 = cat["viva"]

    # W arranca en CERO -- la única asimetría inicial admisible es la del campo de temperatura (ver docstring
    # del módulo). Todo lo demás (confinamiento, EM, gravedad, aniquilación) es una CONSECUENCIA paso a paso.
    W0 = np.zeros((N, N), dtype=np.float64)
    V0 = _marco_inicial_por_atributos(DMAX_INT, color, carga, es_anti, es_quark)
    V30 = _marco_inicial_por_atributos(3, color, carga, es_anti, es_quark)

    estado = dict(W=W0, V=V0, V3=V30, color=color, carga=carga, sabor=sabor0, masa=masa, es_anti=es_anti,
                  es_ferm=es_ferm, es_quark=es_quark, t_birth=np.full(N, float(pasos), dtype=float),
                  s_prev=W0.sum(axis=1), T=T0, expansion=bool(expansion), viva=viva0)

    s_historia = []
    for step in range(pasos):
        estado = paso_unico(estado, step, pasos, apagar=apagar)
        s_historia.append(float(estado["W"].sum()))

    # observable de CRUCE (auditoría, no decide nada): cuánto predice la aritmética exacta de Motor B
    # (resta de poblaciones) frente a cuánto quedó `viva` de verdad al final de este proceso continuo.
    pobl_referencia = resuelve_poblacion_por_aniquilacion(n_quarks, n_antiquarks, n_electrones, n_positrones)

    estado["N"] = N; estado["n_quarks"] = n_quarks; estado["n_antiquarks"] = n_antiquarks
    estado["n_electrones"] = n_electrones; estado["n_positrones"] = n_positrones
    estado["s_historia"] = s_historia
    estado["id_original"] = id_original
    estado["pobl_referencia"] = pobl_referencia
    return estado


def test_invariancia_reordenamiento(n_quarks, n_antiquarks, n_electrones, n_positrones, pasos=STEPS):
    """TEST OBLIGATORIO (Codex+CS): verifica invarianza en las DOS etapas por separado.
    (a) POBLACIÓN (Motor B): resuelve_poblacion_por_aniquilacion sólo toma CONTEOS -- no hay ningún orden
    de individuos que pueda alterarla (no enumera nada). Se verifica reconstruyendo la cuenta 3 veces con
    la MISMA aritmética y confirmando que da exactamente igual -- no hay "reordenamiento" que aplicarle
    porque no hay individuos en esta etapa.
    (b) GEOMETRÍA: los SOBREVIVIENTES ya instanciados como nodos se permutan (orden invertido, determinista)
    antes de correr el bucle, y se compara por IDENTIDAD FÍSICA original qué colores persisten."""
    pobl_a = resuelve_poblacion_por_aniquilacion(n_quarks, n_antiquarks, n_electrones, n_positrones)
    pobl_b = resuelve_poblacion_por_aniquilacion(n_quarks, n_antiquarks, n_electrones, n_positrones)
    poblacion_invariante = (np.array_equal(pobl_a["n_q_survive"], pobl_b["n_q_survive"]) and
                             np.array_equal(pobl_a["n_aq_survive"], pobl_b["n_aq_survive"]) and
                             pobl_a["n_e_survive"] == pobl_b["n_e_survive"])
    print(f"  (a) POBLACIÓN -- n_q_survive por color: {pobl_a['n_q_survive'].tolist()} "
          f"(no hay individuos que reordenar; misma aritmética -> {poblacion_invariante})")
    print(f"      balance_color (bariones cierran si True): {pobl_a['balance_color']}")

    r1 = corre_proceso_unico(n_quarks, n_antiquarks, n_electrones, n_positrones, arm="real", pasos=pasos)
    N_sobrev = r1["N"]
    perm = np.arange(N_sobrev)[::-1].copy()   # reordenamiento FIJO (invertido) de los NODOS sobrevivientes
    r2 = corre_proceso_unico(n_quarks, n_antiquarks, n_electrones, n_positrones, arm="real", pasos=pasos,
                              permutacion=perm)

    def _por_identidad(r):
        color = r["color"]; id_orig = r["id_original"]
        return dict(zip(id_orig.tolist(), color.tolist()))

    s1 = _por_identidad(r1); s2 = _por_identidad(r2)
    mismos_colores_geometria = all(s1[k] == s2[k] for k in s1)
    print(f"  (b) GEOMETRÍA -- colores por identidad, orden original vs invertido: "
          f"{'IGUAL' if mismos_colores_geometria else 'DISTINTO'}")
    invariante = poblacion_invariante and mismos_colores_geometria
    print(f"  {'*** INVARIANTE ***' if invariante else '*** NO INVARIANTE -- revisar ***'}")
    return dict(invariante=invariante, poblacion_invariante=poblacion_invariante,
                geometria_invariante=mismos_colores_geometria, pobl=pobl_a)


# ============================ MEDICIÓN (al final, sin tocar el flujo) ============================
def cuenta_bariones_e_hidrogeno(estado, frac_umbral=1.5):
    """Cuenta, desde la W FINAL, cuántos tríos de 3 quarks de color distinto Y MISMO estatus (materia,
    no antimateria) quedaron mutuamente ligados -- y cuántos de esos protones quedaron ligados a un
    electrón (hidrógeno). También cuenta antibariones (residuo simétrico).

    v7 (ADJUDICACION_..._NO_ADMISIBLE_v6 punto 3, VERIFICADO por CS): la v6 filtraba por viva>=VIVA_UMBRAL=0.5
    POR INDIVIDUO -- pero la aniquilación por clases reparte la densidad colectiva por IGUAL entre todos los
    miembros de la clase (indistinguibles), así que ~10 supervivientes de 30 quarks daban viva=0.333 EN CADA
    UNO de los 30, y ninguno individual cruzaba 0.5 -- cero átomos aunque la población colectiva sí sobrevivió.
    Mezclar una densidad COLECTIVA con un umbral POR INDIVIDUO es incoherente. CORRECCIÓN: no se convierte
    viva en un pase/no-pase individual -- se cuenta directo sobre W, que YA lleva la supresión de viva
    incorporada (W se escaló por sqrt(viva_i*viva_j) en cada paso). El umbral "ligado" es un COCIENTE relativo
    a w0_ef (la propia media de W), así que escalar TODO W por igual (lo que hace una clase enteramente
    aniquilada) no cambia CUÁLES pares cruzan el umbral -- sigue siendo la comparación relativa correcta,
    sin mezclar representaciones."""
    W = estado["W"]; color = estado["color"]; carga = estado["carga"]; es_anti = estado["es_anti"]
    es_quark = estado["es_quark"]
    N = estado["N"]
    if N > 1:
        w0_ef = max(float(W.sum(axis=1).mean()), 1e-12) / max(N - 1, 1)
    else:
        w0_ef = 1e-12
    umbral = frac_umbral * w0_ef
    ligado = W > umbral

    def _cuenta_trios(mask_materia):
        idxs = np.where(mask_materia & (color >= 0))[0]
        usados = np.zeros(N, dtype=bool)
        trios = []
        for i in idxs:
            if usados[i]:
                continue
            vecinos = [j for j in idxs if j != i and not usados[j] and color[j] != color[i] and ligado[i, j]]
            for j in vecinos:
                terceros = [k for k in vecinos if k != j and color[k] != color[i] and color[k] != color[j]
                            and ligado[i, k] and ligado[j, k]]
                if terceros:
                    k = terceros[0]
                    trios.append((i, j, k)); usados[[i, j, k]] = True
                    break
        return trios

    bariones = _cuenta_trios(~es_anti)
    antibariones = _cuenta_trios(es_anti)

    hidrogeno = 0
    hidrogeno_pares = []   # (trio_quarks, idx_electron) -- expuesto para validar cada H (punto c)
    electrones_idx = list(np.where((~es_anti) & (~es_quark))[0])
    for (i, j, k) in bariones:
        if int(carga[i]) + int(carga[j]) + int(carga[k]) <= 0:
            continue
        for e in list(electrones_idx):
            if ligado[i, e] or ligado[j, e] or ligado[k, e]:
                hidrogeno += 1
                hidrogeno_pares.append(((i, j, k), e))
                electrones_idx.remove(e)
                break

    n_q_vivos = int(((~es_anti) & (color >= 0)).sum())
    n_aq_vivos = int((es_anti & (color >= 0)).sum())
    quarks_sueltos = n_q_vivos - 3 * len(bariones)
    return dict(bariones_medidos=len(bariones), antibariones_medidos=len(antibariones),
                hidrogeno_medido=hidrogeno, hidrogeno_pares=hidrogeno_pares,
                quarks_sueltos_medidos=quarks_sueltos,
                quarks_vivos=n_q_vivos, antiquarks_vivos=n_aq_vivos, umbral_lectura=umbral,
                bariones_trios=bariones)


def valida_hidrogeno_discreto(estado, at):
    """(ADJUDICACION_CS072_motor_v7_MATERIA_EMERGE_CS.md v2, punto c): valida que CADA hidrógeno contado sea
    un protón discreto (uud: 3 colores DISTINTOS, carga de trío EXACTA +3 en estas unidades = +1 real) ligado
    a un electrón discreto (carga -3) -- no una coincidencia de umbral. Reporta cada uno; si alguno falla,
    lo dice explícitamente en vez de contarlo como válido."""
    color = estado["color"]; carga = estado["carga"]
    reporte = []
    for (trio, e) in at["hidrogeno_pares"]:
        colores = tuple(int(color[x]) for x in trio)
        cargas_q = tuple(int(carga[x]) for x in trio)
        carga_e = int(carga[e])
        es_uud_discreto = (len(set(colores)) == 3) and (sum(cargas_q) == 3) and (carga_e == -3)
        reporte.append(dict(trio=trio, electron=e, colores=colores, cargas_quarks=cargas_q,
                             carga_electron=carga_e, valido_discreto=es_uud_discreto))
    todos_validos = all(r["valido_discreto"] for r in reporte) if reporte else True
    return dict(todos_validos=todos_validos, detalle=reporte)


# ============================ #22: FLUCTUACIONES QCD -> MASA EFECTIVA (ESPECIFICACION_..._componente22) =====
# G_FUERTE_QCD está declarada arriba con las demás constantes -- reusada aquí y en paso_unico (ya actúa
# dinámicamente sobre la gravedad, ver #22 en paso_unico).
def masa_efectiva_hadrones(estado, bariones_trios):
    """#22: para cada barión (trío i,j,k YA identificado por cuenta_bariones_e_hidrogeno), E_campo_QCD =
    g_fuerte * suma de W sobre los 3 PARES NO ORDENADOS del trío; masa_efectiva = masa_valencia + E_campo_QCD.
    Invariante a permutación POR CONSTRUCCIÓN (suma sobre pares, no hay 'primer' elemento del trío que
    importe). g_fuerte reusa R_STRONG (constante estructural ya existente, no un número nuevo)."""
    W = estado["W"]
    salida = []
    for (i, j, k) in bariones_trios:
        E_campo = G_FUERTE_QCD * (float(W[i, j]) + float(W[i, k]) + float(W[j, k]))
        masa_valencia = 3.0 * MASA_QUARK
        salida.append(dict(trio=(int(i), int(j), int(k)), masa_valencia=masa_valencia,
                            E_campo_QCD=E_campo, masa_efectiva=masa_valencia + E_campo))
    return salida


def masa_efectiva_null_sin_qcd(bariones_trios):
    """NULL (#22, spec): apaga ÚNICAMENTE E_campo_QCD -- masa_efectiva = sólo valencia. Mide qué aporta QCD
    (sobre todo a la gravedad, que depende de masa) comparando contra masa_efectiva_hadrones."""
    masa_valencia = 3.0 * MASA_QUARK
    return [dict(trio=(int(i), int(j), int(k)), masa_valencia=masa_valencia, E_campo_QCD=0.0,
                 masa_efectiva=masa_valencia) for (i, j, k) in bariones_trios]


# ============================ v5: BRAZOS DE CONTROL, INVARIANCIA DURA, DIÁMETRO ============================
def n_firmas_desde_W(W, tol=5):
    """Nº de firmas distintas en W -- firma = fila ordenada (invariante a etiqueta), redondeada a `tol`
    decimales. MISMO criterio que el toy de CS (n_firmas) para reproducir el patrón 1-1-4-8 DENTRO del motor
    completo, no en el toy aislado."""
    if W.shape[0] == 0:
        return 0
    firmas = np.sort(W, axis=1)
    return len(np.unique(np.round(firmas, tol), axis=0))


def _diametro_red(W, frac_umbral=1.5):
    """Diámetro de la red final: umbral físico (mismo criterio que cuenta_bariones_e_hidrogeno) define
    aristas; BFS sin pesos (determinista -- el resultado no depende del orden de recorrido, sólo de qué
    aristas existen) mide la excentricidad máxima dentro de la mayor componente conexa. También reporta
    fragmentación (nº de componentes, tamaño de la mayor) -- un grumo topado se ve en tamaño de componente
    chico y diámetro que NO crece con N; un espacio genuino crece."""
    N = W.shape[0]
    if N <= 1:
        return dict(diametro=0, n_componentes=int(N), tam_mayor_componente=int(N))
    w0_ef = max(float(W.sum(axis=1).mean()), 1e-12) / max(N - 1, 1)
    umbral = frac_umbral * w0_ef
    adj = W > umbral

    visitado = np.zeros(N, dtype=bool)
    componentes = []
    for start in range(N):
        if visitado[start]:
            continue
        comp = []
        cola = deque([start]); visitado[start] = True
        while cola:
            u = cola.popleft(); comp.append(u)
            for v in np.where(adj[u])[0]:
                if not visitado[v]:
                    visitado[v] = True; cola.append(v)
        componentes.append(comp)

    mayor = max(componentes, key=len)
    sub_adj = adj[np.ix_(mayor, mayor)]
    M = len(mayor)
    diam = 0
    for s in range(M):
        dist = np.full(M, -1, dtype=int); dist[s] = 0
        cola = deque([s])
        while cola:
            u = cola.popleft()
            for v in np.where(sub_adj[u])[0]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1; cola.append(v)
        diam = max(diam, int(dist.max()))
    return dict(diametro=diam, n_componentes=len(componentes), tam_mayor_componente=M)


def test_cuatro_brazos(n_quarks, n_antiquarks, n_electrones, n_positrones, pasos=STEPS):
    """LOS 4 BRAZOS DE CONTROL OBLIGATORIOS (instrucción definitiva):
      A) homogéneo SIN expansión -> sin ruptura (no-go)
      B) homogéneo CON expansión -> sin ruptura (la expansión sola no rompe nada)
      C) gradiente SIN expansión -> ruptura PARCIAL (la asimetría sola no basta)
      D) gradiente CON expansión -> ruptura COMPLETA (la cadena del director)
    Sólo D debe encender el espacio -- MÁS que A, MÁS que B, Y MÁS que C. v6 (CS, ADJUDICACION_..._
    NO_ADMISIBLE punto d): v5 sólo comparaba D contra A/B y NUNCA contra C -- eso permitía anunciar "sólo D
    enciende" aunque C y D dieran el MISMO resultado (lo que de hecho pasó: la expansión no aportó nada
    medible sobre el gradiente solo). Ahora se compara D explícitamente contra LAS TRES, y se reporta
    hidrógeno/bariones medidos en cada brazo (punto e) -- si no hay átomos persistentes, el diámetro mide
    quarks/electrones sueltos, no espacio, y hay que decirlo así, no llamarlo positivo."""
    brazos = dict(A=(True, False), B=(True, True), C=(False, False), D=(False, True))
    resultados = {}
    for nombre, (homog, exp) in brazos.items():
        r = corre_proceso_unico(n_quarks, n_antiquarks, n_electrones, n_positrones, arm="real", pasos=pasos,
                                 homogeneo=homog, expansion=exp)
        firmas = n_firmas_desde_W(r["W"])
        diam = _diametro_red(r["W"])
        atomos = cuenta_bariones_e_hidrogeno(r)
        resultados[nombre] = dict(n_firmas=firmas, **diam, **atomos)
        print(f"  brazo {nombre} (homogeneo={homog}, expansion={exp}): n_firmas={firmas}, "
              f"diametro={diam['diametro']}, componentes={diam['n_componentes']}, "
              f"tam_mayor={diam['tam_mayor_componente']}, bariones={atomos['bariones_medidos']}, "
              f"hidrogeno={atomos['hidrogeno_medido']}", flush=True)

    def _mas_estructura(x, y):
        """True si x muestra estrictamente más estructura que y (más firmas distintas, o mismo nº de firmas
        pero diámetro mayor) -- comparación explícita, nunca omitida."""
        rx, ry = resultados[x], resultados[y]
        if rx["n_firmas"] != ry["n_firmas"]:
            return rx["n_firmas"] > ry["n_firmas"]
        return rx["diametro"] > ry["diametro"]

    d_vs_a = _mas_estructura("D", "A"); d_vs_b = _mas_estructura("D", "B"); d_vs_c = _mas_estructura("D", "C")
    print(f"  D vs A: {'D>A' if d_vs_a else 'D<=A'} | D vs B: {'D>B' if d_vs_b else 'D<=B'} | "
          f"D vs C: {'D>C' if d_vs_c else 'D<=C (la expansión NO aportó diferencia medible sobre el gradiente solo)'}",
          flush=True)
    hay_atomos = any(resultados[b]["hidrogeno_medido"] > 0 for b in "ABCD")
    if not hay_atomos:
        print("  *** CERO HIDRÓGENO EN LOS 4 BRAZOS -- el diámetro mide quarks/electrones SUELTOS, "
              "NO puede llamarse 'espacio' todavía ***", flush=True)
    solo_D = d_vs_a and d_vs_b and d_vs_c
    print(f"  {'*** SÓLO D ENCIENDE, MÁS QUE A, B Y C (esperado) ***' if solo_D else '*** NO se cumple el patrón esperado -- ver detalle arriba ***'}",
          flush=True)
    return resultados


def test_invariancia_dura(n_quarks, n_antiquarks, n_electrones, n_positrones, pasos=STEPS, atol=1e-9):
    """INVARIANCIA DURA (Codex, exigida por la instrucción definitiva; CS la pasó en el toy con diferencia
    0.00): reordenar el catálogo + SUS TEMPERATURAS, correr, deshacer la permutación -> W debe volver
    ELEMENTO A ELEMENTO a su lugar exacto (no basta 'mismo conjunto de firmas'). Se corre sobre el brazo D
    (gradiente+expansión), el único que debe mostrar estructura real -- es el caso más exigente."""
    r1 = corre_proceso_unico(n_quarks, n_antiquarks, n_electrones, n_positrones, arm="real", pasos=pasos,
                              homogeneo=False, expansion=True)
    N = r1["N"]
    if N < 2:
        print("  N<2 -- invariancia dura trivial, nada que permutar.", flush=True)
        return dict(invariante=True, max_dif=0.0)
    perm = np.roll(np.arange(N), N // 3)[::-1].copy()   # permutación FIJA, determinista, no trivial
    inv = np.argsort(perm)
    r2 = corre_proceso_unico(n_quarks, n_antiquarks, n_electrones, n_positrones, arm="real", pasos=pasos,
                              homogeneo=False, expansion=True, permutacion=perm)
    W1 = r1["W"]; W2_deshecho = r2["W"][np.ix_(inv, inv)]
    max_dif = float(np.max(np.abs(W1 - W2_deshecho)))
    invariante = bool(np.allclose(W1, W2_deshecho, atol=atol))
    print(f"  INVARIANCIA DURA (brazo D, N={N}): max_dif={max_dif:.3e} -- "
          f"{'*** INVARIANTE (atol 1e-9) ***' if invariante else '*** NO INVARIANTE -- revisar ***'}", flush=True)
    return dict(invariante=invariante, max_dif=max_dif)


PIEZAS_AUDITABLES = ("gravedad", "confinamiento", "em", "debil", "aniquilacion", "localidad",
                     "correlacion", "causal", "marco", "pauli", "tres_cuerpos", "memoria_termica", "qcd")


def auditoria_piezas_activas(n_quarks, n_antiquarks, n_electrones, n_positrones, pasos=STEPS):
    """AUDITORÍA DE LAS 23 (ADJUDICACION_CS072_motor_v7_MATERIA_EMERGE_CS.md v2, punto d): "una pieza cuyo
    apagado no cambia nada NO está actuando -- no cuenta para 23/23". Corre el brazo D (gradiente+expansión,
    el único con estructura real) como LÍNEA BASE, y luego, una por una, apaga cada pieza (ver PIEZAS_
    AUDITABLES) y compara el W final y el conteo de bariones/hidrógeno contra la base. Si apagar una pieza
    NO cambia nada (mismo W, mismos átomos), esa pieza está DECLARADA pero no actúa -- se reporta así, no se
    disimula. Las piezas #9/#18 (expansión) y la asimetría gradiente/homogéneo ya se auditan aparte, vía
    test_cuatro_brazos (A/B/C/D) -- no se repiten aquí."""
    base = corre_proceso_unico(n_quarks, n_antiquarks, n_electrones, n_positrones, pasos=pasos,
                                homogeneo=False, expansion=True)
    W_base = base["W"]; at_base = cuenta_bariones_e_hidrogeno(base)
    print(f"  LÍNEA BASE (brazo D, todas las piezas activas): bariones={at_base['bariones_medidos']}, "
          f"hidrogeno={at_base['hidrogeno_medido']}, |W|_suma={W_base.sum():.6f}", flush=True)
    resultados = {}
    for pieza in PIEZAS_AUDITABLES:
        r = corre_proceso_unico(n_quarks, n_antiquarks, n_electrones, n_positrones, pasos=pasos,
                                 homogeneo=False, expansion=True, apagar=frozenset({pieza}))
        at = cuenta_bariones_e_hidrogeno(r)
        max_dif_W = float(np.max(np.abs(r["W"] - W_base)))
        actua = (max_dif_W > 1e-9) or (at["bariones_medidos"] != at_base["bariones_medidos"]) or \
                (at["hidrogeno_medido"] != at_base["hidrogeno_medido"])
        resultados[pieza] = dict(max_dif_W=max_dif_W, bariones=at["bariones_medidos"],
                                  hidrogeno=at["hidrogeno_medido"], actua=actua)
        print(f"  apagar '{pieza}': max_dif_W={max_dif_W:.6e}, bariones={at['bariones_medidos']}, "
              f"hidrogeno={at['hidrogeno_medido']} -- {'ACTÚA (cambia el resultado)' if actua else '*** NO ACTÚA -- declarada, no operante ***'}",
              flush=True)
    n_activas = sum(1 for r in resultados.values() if r["actua"])
    print(f"  {n_activas}/{len(PIEZAS_AUDITABLES)} piezas auditadas ACTÚAN de verdad sobre el estado.", flush=True)
    return dict(base=dict(bariones=at_base["bariones_medidos"], hidrogeno=at_base["hidrogeno_medido"]),
                piezas=resultados, n_activas=n_activas, n_auditadas=len(PIEZAS_AUDITABLES))


def barrido_N_diametro(escalas, pasos=STEPS):
    """Barrido de N: para CADA escala (n_quarks, n_antiquarks, n_electrones, n_positrones) corre LOS 4
    BRAZOS (v6, ADJUDICACION_..._NO_ADMISIBLE punto f: v5 sólo corría el brazo D, nunca comparaba contra
    A/B/C en cada escala) y mide diámetro + hidrógeno/bariones (punto e: no se llama 'espacio' al diámetro
    si no hay átomos persistentes). Responde la pregunta pendiente para el brazo D específicamente: ¿el
    diámetro CRECE con N (espacio genuino) o queda TOPADO (grumo)? Loguea sólo el resultado por escala."""
    resultados = []
    for (nq, naq, ne, npos) in escalas:
        por_brazo = {}
        for nombre, (homog, exp) in dict(A=(True, False), B=(True, True), C=(False, False), D=(False, True)).items():
            r = corre_proceso_unico(nq, naq, ne, npos, arm="real", pasos=pasos, homogeneo=homog, expansion=exp)
            diam = _diametro_red(r["W"])
            atomos = cuenta_bariones_e_hidrogeno(r)
            por_brazo[nombre] = dict(N=r["N"], **diam, **atomos)
        N = por_brazo["D"]["N"]
        resultados.append(dict(N=N, por_brazo=por_brazo))
        for nombre in "ABCD":
            pb = por_brazo[nombre]
            print(f"  N={N} brazo {nombre} (q={nq},aq={naq},e={ne},p={npos}): diametro={pb['diametro']}, "
                  f"componentes={pb['n_componentes']}, tam_mayor={pb['tam_mayor_componente']}, "
                  f"bariones={pb['bariones_medidos']}, hidrogeno={pb['hidrogeno_medido']}", flush=True)
    diametros_D = [r["por_brazo"]["D"]["diametro"] for r in resultados]
    crece = all(b >= a for a, b in zip(diametros_D, diametros_D[1:])) and diametros_D[-1] > diametros_D[0]
    hay_atomos_D = any(r["por_brazo"]["D"]["hidrogeno_medido"] > 0 for r in resultados)
    if not hay_atomos_D:
        print("  *** CERO HIDRÓGENO EN BRAZO D A TODAS LAS ESCALAS -- el diámetro mide quarks/electrones "
              "SUELTOS, NO es 'espacio' en el sentido del director (átomos persistentes) todavía ***",
              flush=True)
    print(f"  {'*** EL DIÁMETRO (brazo D) CRECE CON N ***' if crece else '*** DIÁMETRO (brazo D) TOPADO -- ver detalle ***'}"
          f"{'' if hay_atomos_D else ' -- PERO sin átomos persistentes, no se puede llamar espacio genuino todavía'}",
          flush=True)
    return resultados


def barrido_sensibilidad_parametros_termicos(n_quarks, n_antiquarks, n_electrones, n_positrones, pasos=STEPS):
    """MEMORIA_ALPHA=0.9, TASA_EXPANSION_GLOBAL=0.02, GRADIENTE_TERMICO_AMPLITUD=0.1 se habían copiado
    LITERALES del toy sin más justificación que "son los del toy" -- eso NO los vuelve constantes físicas
    (G-NO-PARAMETRO-FORMA). Se barren en un rango (mitad, literal, doble) y se confirma si el patrón
    CUALITATIVO (D con MÁS estructura que A, B Y C, estrictamente) es ROBUSTO al valor exacto.

    v7 (ADJUDICACION_..._NO_ADMISIBLE_v6 punto 4, VERIFICADO por CS): la versión anterior usaba `fD >= fC`
    -- ACEPTA IGUALDAD como éxito. Con eso, las 27 combinaciones dieron C=D=68 firmas (empate exacto, la
    expansión no aportaba NADA sobre el gradiente solo) y el código las declaró "patron_ok=True" de todos
    modos -- un positivo falso por construcción del criterio, no por la física. CORREGIDO: exige D>C
    ESTRICTO (mismo criterio de _mas_estructura de test_cuatro_brazos: firmas estrictas, diámetro como
    desempate estricto), y además reporta la BRECHA real (no sólo un booleano) para que un empate quede
    visible, nunca escondido detrás de un `patron_ok=True`."""
    global MEMORIA_ALPHA, TASA_EXPANSION_GLOBAL, GRADIENTE_TERMICO_AMPLITUD
    orig = (MEMORIA_ALPHA, TASA_EXPANSION_GLOBAL, GRADIENTE_TERMICO_AMPLITUD)
    candidatos = dict(alpha=[0.8, 0.9, 0.95], tasa_exp=[0.01, 0.02, 0.04], amplitud=[0.05, 0.1, 0.2])
    resultados = []
    try:
        for alpha in candidatos["alpha"]:
            for tasa_exp in candidatos["tasa_exp"]:
                for amp in candidatos["amplitud"]:
                    MEMORIA_ALPHA, TASA_EXPANSION_GLOBAL, GRADIENTE_TERMICO_AMPLITUD = alpha, tasa_exp, amp
                    rA = corre_proceso_unico(n_quarks, n_antiquarks, n_electrones, n_positrones, pasos=pasos,
                                              homogeneo=True, expansion=False)
                    rB = corre_proceso_unico(n_quarks, n_antiquarks, n_electrones, n_positrones, pasos=pasos,
                                              homogeneo=True, expansion=True)
                    rC = corre_proceso_unico(n_quarks, n_antiquarks, n_electrones, n_positrones, pasos=pasos,
                                              homogeneo=False, expansion=False)
                    rD = corre_proceso_unico(n_quarks, n_antiquarks, n_electrones, n_positrones, pasos=pasos,
                                              homogeneo=False, expansion=True)
                    fA, fB, fC, fD = (n_firmas_desde_W(r["W"]) for r in (rA, rB, rC, rD))
                    dA, dB, dC, dD = (_diametro_red(r["W"])["diametro"] for r in (rA, rB, rC, rD))

                    def _d_supera(x_f, x_d, y_f, y_d):
                        return (x_f > y_f) if x_f != y_f else (x_d > y_d)

                    d_vs_c_estricto = _d_supera(fD, dD, fC, dC)
                    patron_ok = (_d_supera(fD, dD, fA, dA) and _d_supera(fD, dD, fB, dB) and d_vs_c_estricto)
                    brecha_c = dict(firmas=fD - fC, diametro=dD - dC)
                    resultados.append(dict(alpha=alpha, tasa_exp=tasa_exp, amplitud=amp,
                                            n_firmas=(fA, fB, fC, fD), diametros=(dA, dB, dC, dD),
                                            brecha_D_menos_C=brecha_c, patron_ok=patron_ok))
                    print(f"  alpha={alpha} tasa_exp={tasa_exp} amplitud={amp}: n_firmas(A,B,C,D)="
                          f"({fA},{fB},{fC},{fD}) diametros(A,B,C,D)=({dA},{dB},{dC},{dD}) "
                          f"brecha_D-C={brecha_c} patron_ok={patron_ok}", flush=True)
    finally:
        MEMORIA_ALPHA, TASA_EXPANSION_GLOBAL, GRADIENTE_TERMICO_AMPLITUD = orig
    robusto = all(r["patron_ok"] for r in resultados)
    n_empates_con_c = sum(1 for r in resultados if r["brecha_D_menos_C"]["firmas"] == 0
                          and r["brecha_D_menos_C"]["diametro"] == 0)
    print(f"  empates EXACTOS D=C (firmas Y diámetro iguales): {n_empates_con_c} de {len(resultados)}", flush=True)
    print(f"  {'*** PATRÓN ROBUSTO AL VALOR EXACTO, D>C ESTRICTO EN TODOS ***' if robusto else '*** PATRÓN NO ROBUSTO -- ver empates/brechas arriba, NO declarar éxito ***'}",
          flush=True)
    return dict(robusto=robusto, n_empates_con_c=n_empates_con_c, detalle=resultados)


if __name__ == "__main__":
    print("cs072_fold_completo.py v6 (CORRECCIONES ADJUDICACION_..._NO_ADMISIBLE) -- correr "
          "cs072_fold_tanda.py para el veredicto.", flush=True)
