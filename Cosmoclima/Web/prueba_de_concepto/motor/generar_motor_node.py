#!/usr/bin/env python3
# Genera motor_fisico.generado.js extrayendo la física REAL del instrumento
# (sim-cosmoclima.html) sin editarlo ni reescribirla
# a mano -- copia literal de cada pieza, por eso el archivo de salida lleva
# ".generado." en el nombre: se vuelve a generar corriendo este script de
# nuevo, nunca se toca a mano. Fase A del plan de granularidad (08-ago-2026).
#
# Único cambio deliberado (declarado, no oculto): KAPPA_V/O/LF/DELTA pasan de
# `const` a `let` para que el runner de baterías (runner_bateria.js) pueda
# pisarlos entre corridas -- es la única transformación de texto que este
# script hace además de copiar-pegar.
#
# Alcance v1: soporta pasoFisica(false) (el modo "sin registrar historial"
# que ya usa calcularDistribucionRegimen() en el HTML) -- pushHistory(),
# registrarDiaSiCambio(), historialDiario, HIST_MAX NO se portan porque
# pasoFisica() nunca las toca cuando registrar=false. Si en el futuro hace
# falta registrar=true en Node, agregar esos símbolos a la lista de abajo.

import re
import sys
import os

HTML_PATH = os.path.join(os.path.dirname(__file__), '..', 'sim-cosmoclima.html')
OUT_PATH = os.path.join(os.path.dirname(__file__), 'motor_fisico.generado.js')


def find_balanced(text, start_idx, open_ch, close_ch):
    """start_idx debe apuntar exactamente a open_ch. Devuelve el índice
    INMEDIATAMENTE DESPUÉS del close_ch que lo balancea, ignorando llaves/
    corchetes dentro de strings/template literals."""
    assert text[start_idx] == open_ch, f"esperaba '{open_ch}' en {start_idx}, encontré {text[start_idx]!r}"
    depth = 0
    i = start_idx
    in_str = None
    n = len(text)
    while i < n:
        c = text[i]
        if in_str:
            if c == '\\':
                i += 2
                continue
            if c == in_str:
                in_str = None
            i += 1
            continue
        if c in ('"', "'", '`'):
            in_str = c
            i += 1
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError(f"sin balancear desde {start_idx}")


def extract_function(html, name):
    """Extrae 'function name(...){...}' o 'async function name(...){...}' completa."""
    m = re.search(r'(async\s+)?function\s+' + re.escape(name) + r'\s*\(', html)
    if not m:
        raise ValueError(f"función {name!r} no encontrada")
    start = m.start()
    brace_start = html.index('{', m.end())
    end = find_balanced(html, brace_start, '{', '}')
    return start, html[start:end]


def extract_statement(html, anchor):
    """Extrae la sentencia const/let/var completa que EMPIEZA con `anchor`
    (texto exacto verificado con grep), hasta su ';' de nivel superior,
    respetando llaves/corchetes/paréntesis anidados y strings."""
    start = html.index(anchor)
    i = start
    n = len(html)
    depth = 0
    in_str = None
    while i < n:
        c = html[i]
        if in_str:
            if c == '\\':
                i += 2
                continue
            if c == in_str:
                in_str = None
            i += 1
            continue
        if c in ('"', "'", '`'):
            in_str = c
            i += 1
            continue
        if c in '{[(':
            depth += 1
        elif c in '}])':
            depth -= 1
        elif c == ';' and depth == 0:
            return start, html[start:i + 1]
        i += 1
    raise ValueError(f"sin ';' de cierre para anchor {anchor!r}")


# Orden de aparición ORIGINAL en el archivo (no orden "lógico") -- así se
# preservan automáticamente todas las dependencias de TDZ entre const/let
# (ej. TOPE_DIA_CALENDARIO necesita ANIOS_TOTALES_CAL ya declarado antes)
# sin tener que razonarlas a mano una por una.
FUNCIONES = [
    'mulberry32', 'claveSemilla', 'shannonEntropy', 'entropyLocalAbs',
    'entropyAtWidth', 'entropyRel', 'passiveNoiseSample',
    'updateBehavioralEntropy', 'temperaturaPlanetariaReal', 'abioticTf', 'resetField',
    'pseudoNoise', 'evolveField', 'computeDeltaStruct', 'computeCoupling', 'classifyRegime',
    'clasificarCierre', 'diaCalendarioActual', 'fechaDesdeDiaCalendario',
    'formatearFechaCalendario', 'actualizarLluviaDesdeCalendario',
    # Fase B (08-ago-2026): lluvia diaria real, aplicada al HTML para
    # probar -- ver claveFechaDiaria/actualizarLluviaDesdeCalendarioDiario/
    # enTramoLluviaDiaria/refrescarLluviaSiCambio (nuevo, reemplaza el
    # chequeo inline por año que pasoFisica tenía antes).
    'claveFechaDiaria', 'actualizarLluviaDesdeCalendarioDiario',
    'enTramoLluviaDiaria', 'refrescarLluviaSiCambio', 'diaDesdeAnio',
    'actualizarLluviaDesdeCalendarioMensual',
    'objetivoFloracionEmpirico', 'computeFloracion', 'getDayNightTemp',
    'seasonFactor', 'seasonLabel', 'applyDayNightCycle',
    'registrarSaturacion', 'ptcResponse', 'computeLFandErr', 'pasoFisica',
    'valorMesReal', 'picoMensualAnio',
    # Viabilidad mensual real (10-ago-2026) -- faltaba del todo en el puerto
    # (generado 08-ago, antes de que existiera en el HTML).
    'estacionAustralReal', 'actualizarViabilidadSemanal',
    # Activación semanal real, Nivel 3 (10-ago-2026, ronda 4) -- reemplaza al
    # Nivel 1/2 (mensual) para LF/Δ_struct.
    'semanaDelAnioReal', 'actualizarActivacionSemanal',
]

# Anchors exactos (verificados con grep contra el HTML) para cada sentencia
# const/let que hace falta, incluyendo declaraciones compuestas (una sola
# sentencia que define varios nombres, ej. clamp/lerp/gauss o los 4 KAPPA).
SENTENCIAS = [
    'const DAISY_S = 9.17e5,',
    'const clamp=(v,min,max)=>',
    'const gridSize=64;',
    'let field=Array.from(',
    'const HCFG = {win:120',
    'let rngTf  = mulberry32(',
    'let rngEco = mulberry32(',
    'let aBuf = [];',
    'let noiseEchoBuf = [];',
    'let satBuf=[];',
    'const SAT_VENTANA=120, SAT_UMBRAL=0.10;',
    'const TICKS_POR_DIA = 60;',
    'const ANIO_CERO = 1966;',
    'const DIAS_POR_ANIO_CAL = 365;',
    'const ANIOS_TOTALES_CAL = 62;',
    'const TOPE_DIA_CALENDARIO = ANIOS_TOTALES_CAL',
    'const DIAS_POR_MES_CAL = [31,28,31,30,31,30,31,31,30,31,30,31];',
    "const MES_ABREV_CAL = ['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic'];",
    'const DURACION_FLORACION_DIAS = 166;',
    'const EMP_B0 = -1.2123, EMP_B1 = 0.0185;',
    # Regulación interglacial (10-ago-2026, ronda 10).
    'const KAPPA_REGULACION_INTERGLACIAL = 9.4;',
    # Reemplazo de PTC por LF cosmosemiótica, ronda 12 (11-ago-2026) --
    # PTC_REEMPLAZO_MODO se muta a `let` más abajo (mismo mecanismo que
    # KAPPA_*) para poder alternar 'A'/'B' entre corridas sin regenerar.
    "const PTC_REEMPLAZO_MODO = 'B';",
    'const VELOCIDAD_FLORACION_MAX_TEORICA = 0.9/(90*TICKS_POR_DIA);',
    'const DAY_NIGHT_WIDTH = 6;',
    'let _ultimoAnioLluvia=-1;',
    'let _ultimoDiaLluvia=-1;',
    'let _ultimoMesLluvia=null;',
    'const PLUVIO_ANIO_INICIO = ANIO_CERO;',
    'const PLUVIO_ANIO_FIN = ANIO_CERO + ANIOS_TOTALES_CAL',
    'const PLUVIOSIDAD_MENSUAL = ',
    'const LLUVIA_DIARIA_1966_2017 = ',
    # Ronda 6 (10-ago-2026): temperatura real ZHCS/NASA POWER reemplaza el
    # vaivén sintético en applyDayNightCycle() -- antes solo se usaba para
    # un gráfico de comparación, ahora también alimenta la física.
    'const TEMPERATURA_DIARIA_ZHCS = ',
    'const LLUVIA_DIARIA_FIN_DIA = (function(){',
    'const KAPPA_V = 0.9246, KAPPA_O = 0.0408, KAPPA_LF = 0.0652, KAPPA_DELTA = 0.5146;',
    'const state={running:true,',
    # Viabilidad semanal real (10-ago-2026, ronda 5) -- reemplaza a la mensual (ronda 1).
    'let _ultimaSemanaViabilidad=null, _sumCouplingSemana=0, _nTicksSemanaViab=0, _A_sys_env_semanaAnterior=null;',
    # Activación semanal real, Nivel 3 (10-ago-2026, ronda 4) -- reemplaza al
    # Nivel 1/2 (mensual).
    'let _ultimaSemanaActivacion=null, _sumLFSemana=0, _sumDeltaSemana=0, _nTicksSemana=0, _LF_semanaAnterior=null, _delta_semanaAnterior=null;',
    'let KAPPA_LF_POR_SEMANA = [0.4898,0.4801,0.4634,0.4471,0.4312,0.4158,0.4008,0.3863,0.3722,0.3585,0.3453,0.3325,0.3201,0.3081,0.2966,0.2881,0.2788,0.2682,0.2636,0.2564,0.2508,0.2598,0.2844,0.3211,0.3726,0.4136,0.438,0.4704,0.5016,0.5454,0.612,0.6245,0.6712,0.6696,0.6729,0.694,0.6984,0.696,0.6798,0.6764,0.6558,0.6355,0.6355,0.636,0.6348,0.6148,0.6082,0.5887,0.573,0.5541,0.5355,0.5174,0.4998];',
    'let KAPPA_DELTA_POR_SEMANA = [0.7419,0.7594,0.744,0.7431,0.7368,0.7368,0.7359,0.7289,0.72,0.7265,0.7082,0.7221,0.7285,0.7175,0.7247,0.7028,0.7033,0.698,0.6862,0.6801,0.6721,0.6382,0.6553,0.6682,0.6557,0.6487,0.6754,0.6804,0.6822,0.677,0.6913,0.6948,0.7204,0.7059,0.7212,0.7236,0.7207,0.7386,0.759,0.7591,0.7525,0.754,0.7723,0.76,0.7699,0.7821,0.7774,0.7726,0.7669,0.7673,0.7649,0.7645,0.7489];',
    # κ_V/κ_O por semana (10-ago-2026, ronda 5) -- recalibrado con percentiles reales.
    'let KAPPA_V_POR_SEMANA = [0.7687,0.7623,0.7641,0.7765,0.7844,0.801,0.8253,0.8446,0.8653,0.8785,0.8893,0.8914,0.8915,0.8829,0.8672,0.8307,0.7937,0.7541,0.7277,0.6854,0.6528,0.6167,0.606,0.5784,0.5521,0.5517,0.5425,0.5256,0.5341,0.5437,0.5533,0.5774,0.6011,0.6249,0.6556,0.6849,0.7193,0.7517,0.7912,0.828,0.8466,0.8674,0.8762,0.8849,0.8831,0.8748,0.8632,0.852,0.8357,0.8207,0.8052,0.783,0.7686];',
    # Ronda 13 (11-ago-2026): las dos correcciones de fondo, tras banderas
    # independientes. LF_SIN_V/KAPPA_CANONICOS/KAPPA_LF_INFIMO/
    # KAPPA_DELTA_INFIMO se mutan a `let` más abajo (mismo mecanismo que
    # KAPPA_*/PTC_REEMPLAZO_MODO) para poder correr C1/C2/C3 sin regenerar.
    'const LF_SIN_V = true;',
    'const KAPPA_CANONICOS = true;',
    'const KAPPA_V_CANONICO = 0.70;',
    'const KAPPA_O_CANONICO = 0.20;',
    'let KAPPA_LF_INFIMO = 0.35;',
    'let KAPPA_DELTA_INFIMO = 0.5102;',
    # Ronda 16 (13-ago-2026): las dos cotas de C-N2.8.5 sobre e_R y la condición
    # canónica de κ_Δ. Mismo mecanismo de bandera: const -> let más abajo, para
    # poder correr las variantes M0/M1/M2/M3 sin regenerar el motor.
    'const ERR_ABSOLUTO = true;',
    'const ERR_COTA_INFERIOR = true;',
    'const KAPPA_DELTA_CANONICO = true;',
    'let KAPPA_O_POR_SEMANA = [0.0209,0.0247,0.0146,0.0129,0.011,0.0005,0.0005,0.0005,0.0005,0.0005,0.0012,0.009,0.0137,0.0204,0.0364,0.062,0.0726,0.0811,0.0816,0.0811,0.0729,0.0755,0.0781,0.0561,0.0834,0.0764,0.0563,0.0468,0.0467,0.0553,0.0521,0.0199,0.0293,0.0358,0.0209,0.0327,0.0399,0.0277,0.006,0.0028,0.0095,0.0051,0.0066,0.0084,0.0147,0.0188,0.0294,0.0321,0.0314,0.0402,0.0393,0.04,0.0366];',
]


def main():
    with open(HTML_PATH, 'r', encoding='utf-8') as f:
        html = f.read()

    piezas = []  # (start_idx, texto)

    for nombre in FUNCIONES:
        start, texto = extract_function(html, nombre)
        piezas.append((start, f'// --- función: {nombre} ---\n{texto}\n'))

    for anchor in SENTENCIAS:
        start, texto = extract_statement(html, anchor)
        etiqueta = anchor[:40].replace('\n', ' ')
        piezas.append((start, f'// --- sentencia: {etiqueta}... ---\n{texto}\n'))

    piezas.sort(key=lambda p: p[0])

    cuerpo = '\n'.join(texto for _, texto in piezas)

    # Único cambio de texto deliberado: KAPPA_* de const a let, para poder
    # pisarlos entre corridas de una batería. Declarado en el comentario de
    # cabecera de este script y verificado abajo (falla fuerte si no matchea).
    marca_kappa = 'const KAPPA_V = 0.9246, KAPPA_O = 0.0408, KAPPA_LF = 0.0652, KAPPA_DELTA = 0.5146;'
    if marca_kappa not in cuerpo:
        print('ERROR: no se encontró la línea exacta de KAPPA_* a mutar a let -- revisar generar_motor_node.py', file=sys.stderr)
        sys.exit(1)
    cuerpo = cuerpo.replace(marca_kappa, marca_kappa.replace('const KAPPA_V', 'let KAPPA_V'))

    marca_modo = "const PTC_REEMPLAZO_MODO = 'B';"
    if marca_modo not in cuerpo:
        print('ERROR: no se encontró la línea exacta de PTC_REEMPLAZO_MODO a mutar a let -- revisar generar_motor_node.py', file=sys.stderr)
        sys.exit(1)
    cuerpo = cuerpo.replace(marca_modo, marca_modo.replace('const PTC_REEMPLAZO_MODO', 'let PTC_REEMPLAZO_MODO'))

    # Ronda 13: banderas de las dos correcciones, const -> let (mismo motivo).
    for nombre in ('LF_SIN_V', 'KAPPA_CANONICOS',
                   'ERR_ABSOLUTO', 'ERR_COTA_INFERIOR', 'KAPPA_DELTA_CANONICO'):
        marca = f'const {nombre} = '
        if marca not in cuerpo:
            print(f'ERROR: no se encontró {marca!r} a mutar a let -- revisar generar_motor_node.py', file=sys.stderr)
            sys.exit(1)
        cuerpo = cuerpo.replace(marca, f'let {nombre} = ')

    prologo = '''\
// ============================================================
// motor_fisico.generado.js -- NO EDITAR A MANO.
// Generado automáticamente por generar_motor_node.py a partir de
// sim-cosmoclima.html (extracción literal, no
// reescritura). Para actualizar tras un cambio de física en el HTML:
//   python3 generar_motor_node.py
// Verificar siempre después con verificar_contra_navegador.js /
// verificar_motor_node.js antes de confiar en cualquier corrida.
//
// Alcance: soporta pasoFisica(false) -- el modo "sin historial" que ya usa
// calcularDistribucionRegimen() en el HTML. pushHistory/registrarDiaSiCambio/
// historialDiario NO están portados (pasoFisica nunca los toca cuando
// registrar=false).
// ============================================================
'use strict';

// Stubs mínimos, mecánicos, sin lógica inventada -- el HTML original ya
// tiene guards seguros (`if(els.lluvia)`, `if(badge)`) alrededor de todo
// uso de DOM, así que no-ops bastan.
const els = {};
function $(id) { return null; }

'''

    epilogo = '''

module.exports = {
  state, pasoFisica, clasificarCierre, classifyRegime,
  // field se REASIGNA dentro de evolveField()/resetField() (no se muta en
  // el mismo array) -- exportarlo como valor directo quedaría obsoleto tras
  // el primer tick. Getter para siempre devolver la referencia viva.
  get field() { return field; },
  diaCalendarioActual, fechaDesdeDiaCalendario, formatearFechaCalendario,
  computeFloracion, resetField, mulberry32, claveSemilla,
  get KAPPA_V() { return KAPPA_V; }, set KAPPA_V(v) { KAPPA_V = v; },
  get KAPPA_O() { return KAPPA_O; }, set KAPPA_O(v) { KAPPA_O = v; },
  get KAPPA_LF() { return KAPPA_LF; }, set KAPPA_LF(v) { KAPPA_LF = v; },
  get KAPPA_DELTA() { return KAPPA_DELTA; }, set KAPPA_DELTA(v) { KAPPA_DELTA = v; },
  get KAPPA_LF_POR_SEMANA() { return KAPPA_LF_POR_SEMANA; }, set KAPPA_LF_POR_SEMANA(v) { KAPPA_LF_POR_SEMANA = v; },
  get KAPPA_DELTA_POR_SEMANA() { return KAPPA_DELTA_POR_SEMANA; }, set KAPPA_DELTA_POR_SEMANA(v) { KAPPA_DELTA_POR_SEMANA = v; },
  get KAPPA_V_POR_SEMANA() { return KAPPA_V_POR_SEMANA; }, set KAPPA_V_POR_SEMANA(v) { KAPPA_V_POR_SEMANA = v; },
  get KAPPA_O_POR_SEMANA() { return KAPPA_O_POR_SEMANA; }, set KAPPA_O_POR_SEMANA(v) { KAPPA_O_POR_SEMANA = v; },
  get PTC_REEMPLAZO_MODO() { return PTC_REEMPLAZO_MODO; }, set PTC_REEMPLAZO_MODO(v) { PTC_REEMPLAZO_MODO = v; },
  get LF_SIN_V() { return LF_SIN_V; }, set LF_SIN_V(v) { LF_SIN_V = v; },
  get KAPPA_CANONICOS() { return KAPPA_CANONICOS; }, set KAPPA_CANONICOS(v) { KAPPA_CANONICOS = v; },
  get KAPPA_LF_INFIMO() { return KAPPA_LF_INFIMO; }, set KAPPA_LF_INFIMO(v) { KAPPA_LF_INFIMO = v; },
  get KAPPA_DELTA_INFIMO() { return KAPPA_DELTA_INFIMO; }, set KAPPA_DELTA_INFIMO(v) { KAPPA_DELTA_INFIMO = v; },
  get ERR_ABSOLUTO() { return ERR_ABSOLUTO; }, set ERR_ABSOLUTO(v) { ERR_ABSOLUTO = v; },
  get ERR_COTA_INFERIOR() { return ERR_COTA_INFERIOR; }, set ERR_COTA_INFERIOR(v) { ERR_COTA_INFERIOR = v; },
  get KAPPA_DELTA_CANONICO() { return KAPPA_DELTA_CANONICO; }, set KAPPA_DELTA_CANONICO(v) { KAPPA_DELTA_CANONICO = v; },
  TICKS_POR_DIA, ANIO_CERO, DIAS_POR_ANIO_CAL, ANIOS_TOTALES_CAL,
  TOPE_DIA_CALENDARIO, PLUVIOSIDAD_MENSUAL,
  set rngTf(fn) { rngTf = fn; }, get rngTf() { return rngTf; },
  set rngEco(fn) { rngEco = fn; }, get rngEco() { return rngEco; },
  resetBuffers() { aBuf = []; noiseEchoBuf = []; satBuf = []; },
};
'''

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write(prologo + cuerpo + epilogo)

    print(f'Generado: {OUT_PATH} ({len(piezas)} piezas, {len(cuerpo)} caracteres)')


if __name__ == '__main__':
    main()
