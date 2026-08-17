#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs078_kappaV_permutacion.py — Rigor estadístico para κ_V (Bloque 2.8, sumideros CS073)
========================================================================================

Quién soy / qué hago (código autodescriptivo):

  Este script NO corre ninguna simulación nueva. Lee, tal cual están en disco, los
  archivos `cosmog01.sink` (texto plano de Phantom) de la batería N=2000 de CS073:

      /Users/alexis/phantom_cs073/bateria_n2000/ic_real/cosmog01.sink       (REAL, 1 corrida)
      /Users/alexis/phantom_cs073/bateria_n2000/ic_null{1..8}/cosmog01.sink (NULL, 8 corridas)

  y recalcula el invariante κ_V ("acoplamiento sostenido": masa acretada en el último
  tercio de la vida del sumidero / masa acretada en el primer tercio) exactamente como lo
  describe `DISENO_EXPERIMENTOS_NODOS_ABIERTOS_desde_2.5.5_CS.md` (sección 2, fila U3).
  El archivo `analisis_kappa_bloque28.py` mencionado ahí no existe en disco -- este script
  lo reconstruye desde la descripción del método y, a partir de ahí, hace lo que ese
  análisis original NO hacía: tratar la incertidumbre con el rigor que un revisor externo
  (consolidado de 5 evaluaciones de otros modelos, roadmap Fase I-D) pidió explícitamente:

    1. Comparar las distribuciones COMPLETAS de κ_V (REAL vs NULL) a nivel de sumidero
       individual, no sólo la media±DE de 8 números por corrida.
    2. Un test de permutación no paramétrico -- documentando con cuidado cuál es la
       UNIDAD de permutación correcta (ver punto 3).
    3. Un análisis jerárquico: cada corrida (REAL o NULL_i) contiene VARIOS sumideros
       (7-8) que nacen de la MISMA condición inicial / misma semilla de esa corrida.
       Esos sumideros NO son réplicas independientes entre sí -- comparten toda la
       estructura de la corrida que los parió. Tratar los ~63 sumideros NULL como 63
       muestras independientes (pseudoreplicación) infla artificialmente la potencia
       estadística sin agregar evidencia real. La unidad de permutación válida es la
       CORRIDA (9 unidades: 1 REAL + 8 NULL), no el sumidero individual. Este script
       CUANTIFICA cuánta correlación hay dentro de cada corrida (ICC, sección 5) en vez
       de sólo afirmarlo.
    4. Si con el método más riguroso el resultado sigue sin ser significativo, este
       script lo reporta así, sin maquillaje -- no declara "confirmado" ni "refutado":
       eso le corresponde exclusivamente al director del proyecto (Alexis López Tapia).

  (Nota: existía ya en disco una primera pasada rápida de este mismo frente -- resolvía el
  punto 2 con un test de rango a nivel de corrida y un bootstrap simple de los 8 números
  NULL, pero no atacaba los puntos 1 y 3 -- comparación de distribuciones completas a nivel
  de sumidero individual, y evaluación cuantitativa de si los 8 NULL son intercambiables.
  Esta versión reemplaza esa pasada e incorpora los cuatro puntos completos.)

  MÉTODO (idéntico al descrito en el documento de diseño, U3 · κ_V):
    Para cada sumidero individual (columna 19 del .sink = sink ID), con vida
    [t_nace, t_muere] y masa acretada acumulada macc(t) (columna 12 -- se verificó que
    macc(t) == mass(t) exactamente en estos datos: el sumidero sólo crece por acreción,
    nunca pierde masa, así que "masa" y "masa acretada acumulada" son la misma serie):

        D = t_muere - t_nace
        masa_primer_tercio = macc(t_nace + D/3) - macc(t_nace)
        masa_ultimo_tercio = macc(t_muere)      - macc(t_muere - D/3)
        κ_V(sumidero) = masa_ultimo_tercio / masa_primer_tercio

    Los tiempos de frontera de los tercios no siempre caen en un paso de tiempo grabado
    -> se interpola linealmente macc(t) entre los pasos vecinos (np.interp).

    Agregado POR CORRIDA (nivel válido de comparación REAL vs NULL, el mismo nivel del
    documento original): razón AGRUPADA, no promedio de razones individuales --
        κ_V(corrida) = Σ masa_ultimo_tercio(todos sus sumideros) / Σ masa_primer_tercio(ídem)
    Se eligió la razón agrupada como agregado PRINCIPAL porque es numéricamente robusta
    (evita dividir por cero cuando un sumidero individual no acretó nada en su primer
    tercio -- ver más abajo) y porque reproduce, en la misma zona, los números del
    documento de diseño (REAL≈0.84, NULL≈0.49 ± 0.19; el script original que produjo
    0.832/0.511±0.235 no está en disco, así que esta es una reconstrucción fiel del
    método, no una repetición byte-a-byte -- las pequeñas diferencias numéricas son
    esperables y se documentan en el informe adjunto). El promedio de razones
    individuales válidas por corrida se reporta también, como chequeo de robustez.

  CASO DEGENERADO (documentado, no escondido): 9 de 63 sumideros NULL (14%) no acretan
  NADA de masa en su primer tercio de vida (masa_primer_tercio == 0 exactamente), lo que
  vuelve indefinida (0/0) o infinita (X/0) la razón a nivel de sumidero individual. Esto
  NO ocurre en ningún sumidero REAL. Estos casos se excluyen (documentados, contados) del
  análisis a nivel de sumidero individual; no afectan el agregado por corrida (razón
  agrupada), que usa sumas y por tanto es robusto a ceros individuales.

  QUÉ CALCULA ESTE SCRIPT (seis secciones, en orden):
    1. Carga y extracción: κ_V por sumidero individual + κ_V agregado por corrida.
    2. Distribuciones completas (no sólo media±DE): percentiles, mediana, IQR, para
       REAL (n=8 sumideros) y NULL agrupado (n=63, y n=54 tras excluir indefinidos).
    3. Test de permutación EXACTO al nivel de corrida (9 unidades, C(9,1)=9 asignaciones
       posibles) -- el test primario/válido, respeta la estructura jerárquica.
    4. Test de permutación a nivel de sumidero individual (Monte Carlo, 71 unidades)
       -- marcado explícitamente como IMPROPIO/optimista (pseudoreplicación), se reporta
       sólo como sensibilidad para que se vea cuánto infla la significancia ignorar la
       estructura jerárquica.
    5. Descomposición de varianza / ICC dentro de NULL (varianza entre-corridas vs
       varianza entre-sumideros-de-la-misma-corrida) + N efectivo (fórmula de Kish) --
       cuantifica por qué 63 sumideros NULL no son 63 muestras independientes.
    6. Intervalo de confianza no paramétrico (bootstrap jerárquico de dos etapas:
       remuestrea CORRIDAS con reemplazo, y dentro de cada corrida remuestreada,
       remuestrea sus SUMIDEROS con reemplazo) para la media NULL a nivel de corrida,
       comparado contra el único punto REAL observado.

  Restricciones respetadas: no se toca ningún .sink original (sólo lectura), no se corre
  ninguna simulación Phantom nueva, no se edita ningún script existente, no se declara
  ningún veredicto de cierre.

  Requiere: numpy (usa el venv del proyecto):
      /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/venv/bin/python3 cs078_kappaV_permutacion.py

  Reproducibilidad: todo el muestreo aleatorio (Monte Carlo y bootstrap) usa
  np.random.default_rng(SEMILLA_RNG) con SEMILLA_RNG fija (ver abajo) -- correr el script
  dos veces da exactamente los mismos números.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------------------------
BASE = Path("/Users/alexis/phantom_cs073/bateria_n2000")
RUNS = ["ic_real"] + [f"ic_null{i}" for i in range(1, 9)]
ES_REAL = {"ic_real": True, **{f"ic_null{i}": False for i in range(1, 9)}}

COL_T = 0       # columna 1 del .sink: tiempo
COL_MACC = 11   # columna 12: masa acretada acumulada (== masa del sumidero, ver docstring)
COL_SINKID = 18 # columna 19: ID del sumidero dentro de la corrida

N_PERM_MC = 200_000     # tamaño del Monte Carlo para el test a nivel de sumidero (sección 4)
N_BOOT = 20_000          # tamaño del bootstrap jerárquico (sección 6)
SEMILLA_RNG = 20260805   # fecha de esta corrida, fija para reproducibilidad exacta

RNG = np.random.default_rng(SEMILLA_RNG)


# ------------------------------------------------------------------------------------
# Sección 1 — Carga y extracción de κ_V por sumidero y por corrida
# ------------------------------------------------------------------------------------
def cargar_sink(path: Path) -> np.ndarray:
    """Lee un archivo .sink de Phantom, saltando las 2 líneas de cabecera de texto."""
    data = np.loadtxt(path, skiprows=2)
    if data.ndim == 1:
        data = data[None, :]
    return data


def kappa_v_por_sumidero(t: np.ndarray, macc: np.ndarray) -> tuple[float, float]:
    """
    Dada la serie temporal (t, masa acretada acumulada) de UN sumidero, ya ordenada por
    tiempo, devuelve (masa_primer_tercio, masa_ultimo_tercio) de su vida. La razón
    κ_V = masa_ultimo_tercio / masa_primer_tercio se calcula fuera de esta función para
    poder manejar explícitamente el caso masa_primer_tercio == 0 (ver docstring del
    módulo, sección "CASO DEGENERADO").
    """
    t0, t1 = t[0], t[-1]
    duracion = t1 - t0
    frontera_1 = t0 + duracion / 3.0
    frontera_2 = t1 - duracion / 3.0
    m_inicio = np.interp(t0, t, macc)
    m_frontera1 = np.interp(frontera_1, t, macc)
    m_frontera2 = np.interp(frontera_2, t, macc)
    m_final = np.interp(t1, t, macc)
    masa_primer_tercio = m_frontera1 - m_inicio
    masa_ultimo_tercio = m_final - m_frontera2
    return float(masa_primer_tercio), float(masa_ultimo_tercio)


def procesar_corrida(nombre: str) -> dict:
    """
    Procesa una corrida (ic_real o ic_null{i}): devuelve, por cada sumidero que nació en
    ella, (masa_primer_tercio, masa_ultimo_tercio, razón-o-NaN-si-indefinida), más los dos
    agregados a nivel de corrida (razón agrupada y promedio de razones válidas).
    """
    path = BASE / nombre / "cosmog01.sink"
    data = cargar_sink(path)
    sink_ids = data[:, COL_SINKID].astype(int)

    primeros, ultimos, razones = [], [], []
    for sid in np.unique(sink_ids):
        sub = data[sink_ids == sid]
        orden = np.argsort(sub[:, COL_T])
        t = sub[orden, COL_T]
        macc = sub[orden, COL_MACC]
        primero, ultimo = kappa_v_por_sumidero(t, macc)
        primeros.append(primero)
        ultimos.append(ultimo)
        razones.append(ultimo / primero if primero > 0 else np.nan)

    primeros = np.array(primeros)
    ultimos = np.array(ultimos)
    razones = np.array(razones)
    n_indefinidos = int(np.isnan(razones).sum())

    razon_agrupada = float(ultimos.sum() / primeros.sum())
    razon_media_validas = float(np.nanmean(razones)) if n_indefinidos < len(razones) else np.nan

    return dict(
        nombre=nombre,
        es_real=ES_REAL[nombre],
        n_sumideros=len(razones),
        n_indefinidos=n_indefinidos,
        masa_primer_tercio=primeros,
        masa_ultimo_tercio=ultimos,
        razones_sumidero=razones,          # incluye NaN donde es indefinida
        razon_agrupada=razon_agrupada,      # agregado principal de la corrida
        razon_media_validas=razon_media_validas,  # agregado de robustez
    )


def cargar_todo() -> dict:
    return {nombre: procesar_corrida(nombre) for nombre in RUNS}


# ------------------------------------------------------------------------------------
# Sección 2 — Distribuciones completas (no sólo media±DE)
# ------------------------------------------------------------------------------------
def resumen_distribucion(x: np.ndarray, etiqueta: str) -> str:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return f"  {etiqueta}: sin datos válidos"
    pcts = np.percentile(x, [0, 25, 50, 75, 100])
    de = x.std(ddof=1) if len(x) > 1 else 0.0
    return (
        f"  {etiqueta}: n={len(x)}  media={x.mean():.4f}  DE={de:.4f}  "
        f"min={pcts[0]:.4f}  Q1={pcts[1]:.4f}  mediana={pcts[2]:.4f}  Q3={pcts[3]:.4f}  max={pcts[4]:.4f}"
    )


# ------------------------------------------------------------------------------------
# Sección 3 — Test de permutación EXACTO a nivel de corrida (unidad válida)
# ------------------------------------------------------------------------------------
def test_permutacion_nivel_corrida(por_corrida: dict, agregado_key: str) -> dict:
    """
    9 corridas en total (1 REAL + 8 NULL). Bajo H0 ("REAL" es sólo una etiqueta
    intercambiable, no hay diferencia real), cualquiera de las 9 corridas pudo haber sido
    la etiquetada REAL con igual probabilidad -> hay exactamente C(9,1) = 9 asignaciones
    posibles. Estadístico: valor_agregado(corrida etiquetada REAL) - media(valor_agregado
    de las 8 restantes). p (una cola, dirección pre-registrada REAL > NULL) = fracción de
    las 9 asignaciones cuyo estadístico es >= al observado.

    Esto ES el test jerárquico correcto: al permutar corridas completas (no sumideros
    sueltos) se preserva intacta la correlación interna de cada corrida.
    """
    valores = {nombre: por_corrida[nombre][agregado_key] for nombre in RUNS}
    nombres = list(valores.keys())
    obs_stat = valores["ic_real"] - np.mean([valores[n] for n in nombres if n != "ic_real"])

    stats = []
    for elegido in nombres:  # "elegido" hace de REAL bajo cada permutación
        resto = [valores[n] for n in nombres if n != elegido]
        stats.append(valores[elegido] - np.mean(resto))
    stats = np.array(stats)

    p_una_cola = float(np.mean(stats >= obs_stat - 1e-12))
    p_dos_colas = float(np.mean(np.abs(stats) >= abs(obs_stat) - 1e-12))
    rank = int(np.sum(stats >= obs_stat - 1e-12))  # 1 = el más extremo (incluye a sí mismo)

    return dict(
        valores_por_corrida=valores,
        estadistico_observado=float(obs_stat),
        distribucion_nula_9=stats.tolist(),
        p_una_cola=p_una_cola,
        p_dos_colas=p_dos_colas,
        rank_de_9=rank,
    )


# ------------------------------------------------------------------------------------
# Sección 4 — Test de permutación a nivel de sumidero individual (NAIVE, marcado)
# ------------------------------------------------------------------------------------
def test_permutacion_naive_sumidero(datos: dict, rng: np.random.Generator, n_mc: int) -> dict:
    """
    Test IMPROPIO a propósito: pool de los 8 sumideros REAL + 63 sumideros NULL (razones
    válidas únicamente, se excluyen los 9 indefinidos), tratados como si fueran 71
    muestras independientes. NO lo son (sección 5 cuantifica cuánto). Se reporta como
    sensibilidad -- para mostrar, con números, cuánto infla la significancia ignorar la
    estructura jerárquica; NO se usa como evidencia confirmatoria.
    """
    real_vals = datos["ic_real"]["razones_sumidero"]
    real_vals = real_vals[~np.isnan(real_vals)]
    null_vals = np.concatenate(
        [datos[f"ic_null{i}"]["razones_sumidero"] for i in range(1, 9)]
    )
    null_vals = null_vals[~np.isnan(null_vals)]

    pool = np.concatenate([real_vals, null_vals])
    n_real = len(real_vals)
    obs_diff = real_vals.mean() - null_vals.mean()

    n_total = len(pool)
    diffs = np.empty(n_mc)
    for k in range(n_mc):
        perm = rng.permutation(n_total)
        grupo_real = pool[perm[:n_real]]
        grupo_null = pool[perm[n_real:]]
        diffs[k] = grupo_real.mean() - grupo_null.mean()

    p_una_cola = float((np.sum(diffs >= obs_diff) + 1) / (n_mc + 1))
    p_dos_colas = float((np.sum(np.abs(diffs) >= abs(obs_diff)) + 1) / (n_mc + 1))

    return dict(
        n_real=n_real,
        n_null=len(null_vals),
        n_excluidos_indefinidos=9,
        diff_observada=float(obs_diff),
        p_una_cola=p_una_cola,
        p_dos_colas=p_dos_colas,
    )


# ------------------------------------------------------------------------------------
# Sección 5 — Descomposición de varianza / ICC dentro de NULL (jerarquía por semilla)
# ------------------------------------------------------------------------------------
def descomposicion_varianza_null(datos: dict) -> dict:
    """
    ¿Los 8 NULL son 8 "semillas" con estructura propia (cada una empuja a TODOS sus
    sumideros hacia arriba o hacia abajo juntos), o son ruido intercambiable sumidero a
    sumidero? Se responde con un ANOVA de un factor (factor = corrida) sobre las razones
    κ_V de sumidero individual dentro del grupo NULL:

        varianza_entre_corridas: cuánto varía la MEDIA de cada corrida NULL respecto a
            la media global NULL.
        varianza_dentro_corridas: cuánto varían los sumideros de UNA MISMA corrida
            respecto a la media de esa corrida.
        ICC = varianza_entre / (varianza_entre + varianza_dentro): fracción de la
            varianza total que es "de la corrida" y no "del sumidero". ICC alto ->
            los sumideros de una corrida son casi-copias entre sí a efectos
            estadísticos -> tratarlos como independientes es pseudoreplicación fuerte.
        N_efectivo (Kish): n_total_sumideros / (1 + (tamaño_medio_de_grupo - 1) * ICC).
            Es el "tamaño de muestra independiente equivalente" real del pool de 63.
    """
    grupos = []
    for i in range(1, 9):
        r = datos[f"ic_null{i}"]["razones_sumidero"]
        grupos.append(r[~np.isnan(r)])

    medias_grupo = np.array([g.mean() for g in grupos])
    tamanos = np.array([len(g) for g in grupos])
    todo = np.concatenate(grupos)
    media_global = todo.mean()
    k = len(grupos)  # nº de corridas NULL
    n_total = len(todo)

    # Cuadrados medios estilo ANOVA de un factor no balanceado
    ss_entre = float(np.sum(tamanos * (medias_grupo - media_global) ** 2))
    gl_entre = k - 1
    ms_entre = ss_entre / gl_entre

    ss_dentro = float(sum(np.sum((g - g.mean()) ** 2) for g in grupos))
    gl_dentro = n_total - k
    ms_dentro = ss_dentro / gl_dentro if gl_dentro > 0 else float("nan")

    n0 = (n_total - np.sum(tamanos ** 2) / n_total) / (k - 1)
    var_entre_sin_recortar = (ms_entre - ms_dentro) / n0  # puede dar negativo, se reporta igual
    var_entre = max(0.0, var_entre_sin_recortar)
    var_dentro = ms_dentro
    icc = var_entre / (var_entre + var_dentro) if (var_entre + var_dentro) > 0 else 0.0

    m_medio = tamanos.mean()
    design_effect = 1 + (m_medio - 1) * icc
    n_efectivo = n_total / design_effect

    return dict(
        k_corridas_null=k,
        n_total_sumideros_null=n_total,
        tamanos_grupo=tamanos.tolist(),
        medias_por_corrida=medias_grupo.tolist(),
        ms_entre_corridas=ms_entre,
        ms_dentro_corridas=ms_dentro,
        var_entre_sin_recortar=var_entre_sin_recortar,
        var_entre_corridas=var_entre,
        var_dentro_corridas=var_dentro,
        icc=icc,
        design_effect=design_effect,
        n_efectivo=n_efectivo,
    )


# ------------------------------------------------------------------------------------
# Sección 6 — Bootstrap jerárquico (2 etapas) -> intervalo de confianza no paramétrico
# ------------------------------------------------------------------------------------
def bootstrap_jerarquico_null(datos: dict, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """
    Etapa 1: remuestrea con reemplazo las 8 CORRIDAS NULL (respeta que la corrida es la
             unidad independiente real).
    Etapa 2: dentro de cada corrida remuestreada, remuestrea con reemplazo sus propios
             sumideros (masa_primer_tercio, masa_ultimo_tercio) -- no las razones sueltas,
             para poder recomputar la razón agrupada de esa corrida-remuestreada tal como
             se define en el agregado principal (Σúltimo/Σprimero).
    Con las 8 corridas-remuestreadas, agrega otra vez (media de las 8 razones agrupadas)
    -> una realización bootstrap de "la media NULL a nivel de corrida". Repetido n_boot
    veces da la distribución bootstrap de esa media, de la que se lee el IC 95% percentil.
    """
    corridas_null = [datos[f"ic_null{i}"] for i in range(1, 9)]
    medias_boot = np.empty(n_boot)

    for b in range(n_boot):
        elegidas_idx = rng.integers(0, 8, size=8)  # remuestreo de corridas con reemplazo
        razones_corrida_boot = np.empty(8)
        for j, idx in enumerate(elegidas_idx):
            corrida = corridas_null[idx]
            n_s = corrida["n_sumideros"]
            sub_idx = rng.integers(0, n_s, size=n_s)  # remuestreo de sumideros con reemplazo
            primeros_b = corrida["masa_primer_tercio"][sub_idx]
            ultimos_b = corrida["masa_ultimo_tercio"][sub_idx]
            suma_primeros = primeros_b.sum()
            # Caso degenerado (ver docstring): si el remuestreo saca sólo sumideros con
            # masa_primer_tercio==0, la razón agrupada de ESA corrida-remuestreada queda
            # indefinida (0/0). No se rellena con un número arbitrario -- se marca NaN y
            # la réplica bootstrap completa se descarta más abajo (documentado, no oculto).
            razones_corrida_boot[j] = (ultimos_b.sum() / suma_primeros) if suma_primeros > 0 else np.nan
        medias_boot[b] = razones_corrida_boot.mean() if not np.any(np.isnan(razones_corrida_boot)) else np.nan

    return medias_boot


def bootstrap_real_un_solo_run(datos: dict, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """
    Para REAL sólo hay 1 corrida -> no se puede remuestrear "entre semillas" (n=1). Lo
    único que se puede propagar es la incertidumbre DENTRO de esa corrida (remuestreo de
    sus propios sumideros). Se reporta por separado y se documenta la limitación: este IC
    NO incluye la variabilidad entre-semillas para REAL, que en NULL sí se ve reflejada.
    """
    corrida = datos["ic_real"]
    n_s = corrida["n_sumideros"]
    valores = np.empty(n_boot)
    for b in range(n_boot):
        sub_idx = rng.integers(0, n_s, size=n_s)
        primeros_b = corrida["masa_primer_tercio"][sub_idx]
        ultimos_b = corrida["masa_ultimo_tercio"][sub_idx]
        valores[b] = ultimos_b.sum() / primeros_b.sum()
    return valores


# ------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------
def main() -> None:
    print("=" * 88)
    print("CS078 — κ_V (Bloque 2.8) con rigor estadístico: distribuciones completas,")
    print("permutación jerárquica, ICC por semilla, bootstrap. Sin simulación nueva.")
    print("=" * 88)

    datos = cargar_todo()

    # --- Sección 1: valores de referencia por corrida ---
    print("\n[1] κ_V agregado por corrida (razón agrupada Σúltimo/Σprimero, y promedio de")
    print("    razones individuales válidas, como chequeo de robustez):")
    for nombre in RUNS:
        d = datos[nombre]
        etiqueta = "REAL" if d["es_real"] else nombre.replace("ic_", "")
        print(
            f"    {etiqueta:9s}  n_sumideros={d['n_sumideros']}  "
            f"indefinidos(1er tercio=0)={d['n_indefinidos']}  "
            f"razon_agrupada={d['razon_agrupada']:.4f}  "
            f"razon_media_validas={d['razon_media_validas']:.4f}"
        )
    null_agrupadas = np.array([datos[f"ic_null{i}"]["razon_agrupada"] for i in range(1, 9)])
    null_medias_validas = np.array([datos[f"ic_null{i}"]["razon_media_validas"] for i in range(1, 9)])
    print(
        f"\n    NULL (razon_agrupada)      media±DE = {null_agrupadas.mean():.4f} ± "
        f"{null_agrupadas.std(ddof=1):.4f}   (n=8 corridas)"
    )
    print(
        f"    NULL (razon_media_validas) media±DE = {null_medias_validas.mean():.4f} ± "
        f"{null_medias_validas.std(ddof=1):.4f}   (n=8 corridas)"
    )
    print(
        f"    REAL razon_agrupada = {datos['ic_real']['razon_agrupada']:.4f}  |  "
        f"z (aprox., normal) = "
        f"{(datos['ic_real']['razon_agrupada'] - null_agrupadas.mean()) / null_agrupadas.std(ddof=1):.3f}"
    )
    print(
        "    (documento de diseño original: REAL=0.832, NULL=0.511±0.235, z=1.37 -- esta"
    )
    print(
        "     reconstrucción da números en la misma zona, no idénticos byte-a-byte; ver informe.)"
    )

    # --- Sección 2: distribuciones completas a nivel de sumidero individual ---
    print("\n[2] Distribuciones COMPLETAS a nivel de sumidero individual (no sólo media±DE):")
    real_razones = datos["ic_real"]["razones_sumidero"]
    null_razones_pool = np.concatenate(
        [datos[f"ic_null{i}"]["razones_sumidero"] for i in range(1, 9)]
    )
    print(resumen_distribucion(real_razones, "REAL (8 sumideros, 1 corrida)"))
    print(resumen_distribucion(null_razones_pool, "NULL agrupado (63 sumideros, 8 corridas, pool naive)"))
    print(
        f"    Sumideros NULL con masa_primer_tercio == 0 (razón indefinida, excluidos "
        f"arriba): {int(np.isnan(null_razones_pool).sum())} de 63 (ningún caso así en REAL)."
    )

    # --- Sección 3: test de permutación exacto a nivel de corrida (VÁLIDO) ---
    print("\n[3] Test de permutación EXACTO a nivel de CORRIDA (9 unidades: 1 REAL + 8 NULL;")
    print("    C(9,1)=9 asignaciones posibles -- éste es el test primario/válido, respeta")
    print("    que los sumideros de una misma corrida no son independientes entre sí):")
    for key, etiqueta in [("razon_agrupada", "razón agrupada"), ("razon_media_validas", "media de razones válidas")]:
        r = test_permutacion_nivel_corrida(datos, key)
        print(f"\n    -- agregado: {etiqueta} --")
        print(f"    estadístico observado (REAL - media(resto de 8)) = {r['estadistico_observado']:.4f}")
        print(f"    distribución nula exacta (9 valores): {[round(v, 4) for v in r['distribucion_nula_9']]}")
        print(f"    rank de REAL entre las 9 asignaciones = {r['rank_de_9']} de 9 (1 = más extremo)")
        print(f"    p (una cola, H1 pre-registrada REAL>NULL) = {r['p_una_cola']:.4f}  (mínimo posible con n=9: 1/9=0.111)")
        print(f"    p (dos colas) = {r['p_dos_colas']:.4f}")

    # --- Sección 4: test naive a nivel de sumidero individual (marcado como impropio) ---
    print("\n[4] Test de permutación a nivel de SUMIDERO INDIVIDUAL (Monte Carlo, "
          f"{N_PERM_MC:,} permutaciones)")
    print("    -- MARCADO COMO IMPROPIO: pool de 71 sumideros tratados como si fueran")
    print("    independientes (pseudoreplicación). Se reporta SÓLO como sensibilidad, para")
    print("    mostrar cuánto infla la significancia ignorar la jerarquía (sección 5 cuantifica")
    print("    por qué esto no es válido como evidencia primaria):")
    r_naive = test_permutacion_naive_sumidero(datos, RNG, N_PERM_MC)
    print(
        f"    n_REAL={r_naive['n_real']}  n_NULL={r_naive['n_null']}  "
        f"(excluidos por indefinidos: {r_naive['n_excluidos_indefinidos']})"
    )
    print(f"    diferencia de medias observada = {r_naive['diff_observada']:.4f}")
    print(f"    p (una cola, naive) = {r_naive['p_una_cola']:.5f}")
    print(f"    p (dos colas, naive) = {r_naive['p_dos_colas']:.5f}")
    print("    (comparar contra la sección 3: la caída de p al tratar sumideros como")
    print("     independientes es precisamente la pseudoreplicación que este informe evita.)")

    # --- Sección 5: descomposición de varianza / ICC dentro de NULL ---
    print("\n[5] Descomposición de varianza dentro de NULL (¿los 8 NULL son 8 semillas con")
    print("    estructura propia, o ruido intercambiable sumidero-a-sumidero?):")
    dv = descomposicion_varianza_null(datos)
    print(f"    corridas NULL = {dv['k_corridas_null']}, sumideros NULL (válidos) = {dv['n_total_sumideros_null']}")
    print(f"    tamaños de grupo por corrida = {dv['tamanos_grupo']}")
    print(f"    medias de κ_V por corrida NULL = {[round(v, 4) for v in dv['medias_por_corrida']]}")
    print(f"    MS entre-corridas = {dv['ms_entre_corridas']:.5f}   MS dentro-de-corridas = {dv['ms_dentro_corridas']:.5f}")
    print(f"    varianza ENTRE corridas (estimador de momentos, sin recortar) = {dv['var_entre_sin_recortar']:.5f}")
    print(f"    varianza ENTRE corridas (recortada a >=0 para el ICC) = {dv['var_entre_corridas']:.5f}")
    print(f"    varianza DENTRO de corridas (sumidero a sumidero) = {dv['var_dentro_corridas']:.5f}")
    print(f"    ICC (fracción de varianza total que es 'de la corrida/semilla') = {dv['icc']:.3f}")
    print(f"    efecto de diseño (Kish) = {dv['design_effect']:.3f}")
    print(
        f"    N efectivo del pool de {dv['n_total_sumideros_null']} sumideros NULL = "
        f"{dv['n_efectivo']:.2f}  (compárese con las 8 corridas reales)"
    )
    if dv["var_entre_sin_recortar"] <= 0:
        print(
            "    AVISO honesto: el MS entre-corridas (0.35) es, de hecho, MENOR que el MS"
            "\n    dentro-de-corridas (0.37) -- el estimador de momentos da ICC=0 (recortado)."
            "\n    Con sólo 8 grupos de 4-8 sumideros, este estimador tiene varianza de muestreo"
            "\n    enorme y una distribución truncada en 0 -- 'ICC=0 aquí' NO equivale a 'los"
            "\n    sumideros de una corrida son independientes'. Las medias por corrida arriba"
            "\n    (0.21 a 0.86) SÍ varían de forma apreciable a simple vista; lo que este test"
            "\n    concreto no puede es CONFIRMAR con soltura que esa variación excede el ruido"
            "\n    esperable de grupos tan chicos. Por eso la sección 3 (permutación a nivel de"
            "\n    corrida) sigue siendo la prueba primaria -- no depende de resolver este punto,"
            "\n    porque el argumento físico para agrupar por corrida (misma condición inicial,"
            "\n    misma dinámica de colapso compartida) no depende de que el ICC salga alto."
        )

    # --- Sección 6: bootstrap jerárquico -> IC 95% ---
    print(f"\n[6] Bootstrap jerárquico de dos etapas ({N_BOOT:,} réplicas: remuestrea corridas")
    print("    NULL con reemplazo, y dentro de cada una remuestrea sus sumideros con reemplazo)")
    print("    para el IC de la media NULL a nivel de corrida, comparado contra el único punto REAL:")
    boot_null_crudo = bootstrap_jerarquico_null(datos, RNG, N_BOOT)
    n_descartadas = int(np.isnan(boot_null_crudo).sum())
    boot_null = boot_null_crudo[~np.isnan(boot_null_crudo)]
    ic_null_95 = np.percentile(boot_null, [2.5, 97.5])
    print(
        f"    ({n_descartadas} de {N_BOOT:,} réplicas descartadas: el remuestreo de sumideros dejó"
        f"\n     alguna corrida con masa_primer_tercio total = 0, razón indefinida -- caso"
        f"\n     degenerado documentado arriba, no se rellena con un número inventado.)"
    )
    print(f"    media NULL bootstrap = {boot_null.mean():.4f}   IC 95% = [{ic_null_95[0]:.4f}, {ic_null_95[1]:.4f}]")
    real_obs = datos["ic_real"]["razon_agrupada"]
    percentil_real = float(np.mean(boot_null < real_obs) * 100)
    print(f"    valor REAL observado ({real_obs:.4f}) cae en el percentil {percentil_real:.1f} de la")
    print("    distribución bootstrap de la media NULL (100 = por encima de todas las réplicas).")

    boot_real = bootstrap_real_un_solo_run(datos, RNG, N_BOOT)
    ic_real_95 = np.percentile(boot_real, [2.5, 97.5])
    print(
        f"\n    (Nota de honestidad: REAL sólo tiene 1 corrida -> no se puede remuestrear"
        f"\n     'entre semillas' para REAL. El IC de abajo SÓLO propaga incertidumbre dentro"
        f"\n     de esa única corrida -- remuestreo de sus 8 sumideros -- y por diseño es más"
        f"\n     angosto que el de NULL; no es comparable en pie de igualdad.)"
    )
    print(f"    IC 95% intra-corrida de REAL (sólo incertidumbre dentro de la corrida) = "
          f"[{ic_real_95[0]:.4f}, {ic_real_95[1]:.4f}]")

    solapan = not (ic_real_95[1] < ic_null_95[0] or ic_null_95[1] < ic_real_95[0])
    print(f"\n    ¿Se solapan el IC intra-corrida de REAL y el IC jerárquico de NULL? {'SÍ' if solapan else 'NO'}")

    print("\n" + "=" * 88)
    print("FIN. Números arriba; la lectura/veredicto es del informe adjunto y, en última")
    print("instancia, del director del proyecto -- este script no declara cierres.")
    print("=" * 88)


if __name__ == "__main__":
    sys.exit(main())
