"""
CS090 — FASE VI / O2-D: SUSTRATO A0 GENUINAMENTE 2D (y 3D) — CAMPO CONTINUO SIN GRAFO
====================================================================================================
QUIÉN SOY
---------
Script NUEVO (no toca ni edita ningún congelado; sólo importa `cs090_fase5_generador` para reusar la
lógica de dos de los chequeos del filtro P1-P5). Implementa y mide un sustrato A0 en 2 dimensiones
(opcionalmente 3), con dinámica local de reacción-difusión, y lo mide con instrumentos apropiados a
2D/3D **sin construir NINGÚN grafo derivado**.

POR QUÉ EXISTE (la limitación que motiva la tarea)
--------------------------------------------------
La representación A0 del motor de Fase V (`cs090_fase5_motor.construir_A0`) es **un anillo 1D**: N
sitios en collar, fase en Z_K, difusión a los vecinos izquierda/derecha. En 1D varias métricas
geométricas son degeneradas — el propio informe `FASE5_A0_metricas_nativas_CS.md` descartó box-counting
porque "en 1D casi siempre da dimensión ≈1 trivialmente". Y el cierre `FASE6_O1C_cierre_A0_CS.md`
mostró que la única forma que tenía el pipeline de "ver estructura" en A0 era **derivar un grafo de
similitud por muestreo de candidatos al azar**, receta que fabrica mundo-pequeño por construcción
(test-retest: la etiqueta Clase II sólo se reproduce el 45.6% de las veces sobre el MISMO campo).

La pregunta de fondo (propuesta A4 del 3er analista): **¿S>0 puede operar en un campo continuo sin
grafo, o el grafo es condición NECESARIA (no sólo suficiente) para el régimen del modelo?**

QUÉ HACE ESTE ARCHIVO
---------------------
1) DOS FAMILIAS de dinámica local en grilla toroidal L^d (d=2 o 3), ambas pasando el filtro P1-P5
   verificado explícitamente (no asumido):
     · Familia FASE — generalización DIRECTA del A0 1D: fase en Z_K por sitio, difusión circular a los
       vecinos de la grilla (vecindad 4 u 8 en 2D; 6 en 3D) + ruido. Es literalmente el mismo motor
       que el anillo, con d ejes en vez de 1.
     · Familia RD — reacción-difusión tipo Gray-Scott: dos especies U,V acopladas (U+2V→3V), difusión
       local por laplaciano de 5 puntos, alimentación F y remoción k. Es un campo continuo genuino con
       formación de patrón, NO una variante estadística de un grafo.
2) MEDICIÓN sin grafo derivado, toda dimensión-genérica (funciona en 2D y 3D con el mismo código):
     · espectro de potencia radial P(k) — pico, prominencia, nitidez (ancho relativo), entropía
       espectral, exponente de ley de potencias
     · espectro ANGULAR — ¿hay dirección privilegiada? (anisotropía en el anillo del pico)
     · dimensión por BOX-COUNTING del conjunto de nivel (en 2D/3D sí es informativa)
     · ENTROPÍA de configuración espacial (entropía de bloques 2^d binarizados) — y su evolución
       temporal, para ver si el campo se estructura o se homogeniza
     · PERCOLACIÓN de conjuntos de nivel con BARRIDO de umbral (19 cuantiles, no un umbral elegido:
       ésa es la lección de O1-C — un umbral fijo puede fabricar el resultado)
     · ESCALAMIENTO MASA-RADIO desde los máximos locales del campo
     · longitud de correlación ξ por autocorrelación radial (vía FFT)
3) CONTROL NULL obligatorio: **el mismo campo con las posiciones barajadas** (destruye la estructura
   espacial, conserva exactamente la distribución de valores). Toda métrica se reporta REAL y NULL.
4) BARRIDO: 20 configuraciones de parámetros (10 FASE + 10 RD) × 4 semillas = 80 corridas en 2D,
   más un bloque 3D reducido si el presupuesto alcanza.

POR QUÉ ESTAS MÉTRICAS Y NO UN GRAFO
-------------------------------------
Todas las métricas de acá leen el array del campo directamente. La única "conectividad" que se usa es
la **adyacencia de la propia grilla** (los mismos vecinos que usa la dinámica: `np.roll` por eje), en
la etiquetación de componentes de la percolación. Eso NO es un grafo derivado: no se inventa ninguna
arista nueva, no hay muestreo de candidatos al azar, no hay saltos de largo alcance. Es exactamente el
contraste con `_grafo_medicion_A0` del pipeline viejo.

NOTA HONESTA SOBRE COORDENADAS (P3): la DINÁMICA nunca lee (x,y,z) — los vecinos son desplazamientos
fijos declarados una vez, igual que `left/right` en el anillo 1D que el generador congelado ya acepta
como P3-compatible. Dos MÉTRICAS (box-counting y masa-radio) sí usan la geometría de la grilla, pero
eso es del lado del instrumento, no de la regla: la distancia que usan es la distancia de la propia
adyacencia toroidal, no una coordenada horneada en la ley de actualización. Queda documentado, no
escondido.

Código autodescriptivo. No declara cierre ni veredicto — reporta números. No corre Phantom.
"""
from __future__ import annotations
import sys, time, csv, json, inspect
import numpy as np
from scipy import ndimage

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

import cs090_fase5_generador as GEN   # SÓLO IMPORT (patrones P3 y nombres prohibidos P5) — no se edita


# ====================================================================================================
# 1) SUSTRATOS — grilla toroidal L^d. Vecinos = desplazamientos fijos por eje (np.roll), nunca (x,y,z).
# ====================================================================================================
def desplazamientos_grilla(ndim, vecindad):
    """Devuelve la lista de desplazamientos que definen la vecindad relacional de la grilla.
    2D vecindad=4 → (±1,0),(0,±1); 2D vecindad=8 → agrega las 4 diagonales; 3D vecindad=6 → (±1,0,0)...
    Estos desplazamientos se declaran UNA vez y no dependen de la posición del sitio: son una relación
    fija (como `left`/`right` del anillo 1D), no una lectura de coordenada global."""
    if ndim == 2:
        base = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if vecindad == 8:
            base += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        return base
    if ndim == 3:
        return [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    raise ValueError("ndim debe ser 2 o 3")


def _desplazar(A, despl):
    """Traslada el array A por el desplazamiento `despl` (un vecino de la grilla), con envoltura
    toroidal. Rueda SÓLO los ejes con desplazamiento distinto de cero — `np.roll` con shift=0 igual
    copia el array entero, así que hacerlo por tupla completa duplicaba el costo de cada paso (medido:
    2.3 ms/paso a L=96 con la versión ingenua)."""
    ejes = [i for i, d in enumerate(despl) if d != 0]
    if not ejes:
        return A
    return np.roll(A, shift=tuple(despl[i] for i in ejes), axis=tuple(ejes))


def construir_FASE(L, ndim, rng, p):
    """Campo de FASE inicial: cada sitio de la grilla L^d arranca con una fase uniforme en [0,K).
    Estado desordenado puro — ninguna estructura puesta a mano."""
    forma = (L,) * ndim
    return {"tipo": "FASE", "S": rng.uniform(0, p["K"], forma), "L": L, "ndim": ndim}


def paso_FASE(sustrato, p, rng):
    """UN sweep de la dinámica de fase: media circular con los vecinos de la grilla + ruido.
    Es la MISMA regla que usa A0 en el anillo 1D (`(1-J)*S + J*<vecinos>` en el círculo, más ruido
    gaussiano en unidades de K), generalizada a d ejes: el promedio de vecinos usa el vector unitario
    exp(iθ) para que el promedio sea circular y no salte en el corte 0/K."""
    S = sustrato["S"]
    z = np.exp(1j * 2 * np.pi * S / p["K"])
    acum = np.zeros_like(z)
    for despl in sustrato["despl"]:
        acum = acum + _desplazar(z, despl)
    z_vec = acum / len(sustrato["despl"])
    ang_vec = np.angle(z_vec) / (2 * np.pi) * p["K"]          # fase media de los vecinos, en unidades de K
    S_new = ((1 - p["J"]) * S + p["J"] * ang_vec) % p["K"]
    S_new = (S_new + rng.normal(0, p["noise"], S.shape)) % p["K"]
    sustrato["S"] = S_new
    return sustrato


def construir_RD(L, ndim, rng, p):
    """Campo de REACCIÓN-DIFUSIÓN (Gray-Scott) inicial: U≈1, V≈0, con una siembra ALEATORIA e
    INDEPENDIENTE por sitio (fracción `siembra` de sitios sorteados al azar reciben V=`amp_siembra`).
    Nada de una 'burbuja en el centro': el estado inicial no tiene NINGUNA estructura espacial — es
    i.i.d. sitio a sitio. Toda la estructura que aparezca después la produce la dinámica.

    Nota de calibración (verificada empíricamente, no supuesta): con siembra muy rala (2% de sitios
    aislados) la autocatálisis de V no arranca — un sitio suelto difunde su V antes de poder
    reproducirse y el patrón muere (V→0 exacto en las 5 configuraciones probadas). Con siembra ~0.3
    hay suficientes pares/tríos de sitios adyacentes por puro azar para que la reacción nuclee. Se usa
    entonces siembra=0.3 por defecto, y una configuración del barrido varía la siembra a propósito
    para dejar el efecto documentado."""
    forma = (L,) * ndim
    U = np.ones(forma)
    V = np.zeros(forma)
    mascara = rng.random(forma) < p["siembra"]
    amp = p.get("amp_siembra", 0.25)
    V[mascara] = amp
    U[mascara] = 1.0 - amp
    U += 0.01 * rng.normal(size=forma)
    V += 0.01 * rng.normal(size=forma)
    return {"tipo": "RD", "U": np.clip(U, 0, 1), "V": np.clip(V, 0, 1), "L": L, "ndim": ndim}


def _laplaciano(A, despl):
    """Laplaciano discreto por vecindad de grilla: suma de vecinos menos n_vecinos·centro.
    Sólo usa los desplazamientos fijos (np.roll) — misma relación local que la dinámica de FASE."""
    acum = np.zeros_like(A)
    for d in despl:
        acum = acum + _desplazar(A, d)
    return acum - len(despl) * A


def paso_RD(sustrato, p, rng):
    """UN paso de Gray-Scott: U + 2V → 3V (autocatálisis de V consumiendo U), alimentación de U a tasa
    F(1-U) y remoción de V a tasa (F+k)V. Difusión local de cada especie por laplaciano de grilla.
    Reciprocidad (P4): el término UV² aparece con signo opuesto en las dos ecuaciones — U afecta a V y
    V afecta a U, en el mismo paso. `ruido` es opcional y muy chico (mantiene la dinámica estocástica
    sin destruir el patrón)."""
    U, V = sustrato["U"], sustrato["V"]
    despl = sustrato["despl"]
    reaccion = U * V * V
    U_new = U + p["dt"] * (p["Du"] * _laplaciano(U, despl) - reaccion + p["F"] * (1.0 - U))
    V_new = V + p["dt"] * (p["Dv"] * _laplaciano(V, despl) + reaccion - (p["F"] + p["kmata"]) * V)
    if p["noise"] > 0:
        V_new = V_new + rng.normal(0, p["noise"], V.shape)
    sustrato["U"] = np.clip(U_new, 0.0, 1.5)
    sustrato["V"] = np.clip(V_new, 0.0, 1.5)
    return sustrato


def construir(p, L, ndim, rng):
    """Fábrica: arma el sustrato de la familia pedida y le adjunta los desplazamientos de vecindad."""
    despl = desplazamientos_grilla(ndim, p.get("vecindad", 4 if ndim == 2 else 6))
    s = construir_FASE(L, ndim, rng, p) if p["familia"] == "FASE" else construir_RD(L, ndim, rng, p)
    s["despl"] = despl
    return s


def evolucionar(sustrato, p, rng, n_pasos, snapshots=()):
    """Corre n_pasos de la dinámica de la familia. Si se piden `snapshots` (lista de números de paso),
    devuelve además el observable escalar en esos instantes — para poder ver si el campo se ESTRUCTURA
    o se HOMOGENIZA con el tiempo (persistencia), en vez de mirar sólo el estado final."""
    paso = paso_FASE if sustrato["tipo"] == "FASE" else paso_RD
    guardados = {}
    for t in range(1, n_pasos + 1):
        sustrato = paso(sustrato, p, rng)
        if t in snapshots:
            guardados[t] = observable_escalar(sustrato, p).copy()
    return sustrato, guardados


def observable_escalar(sustrato, p):
    """Observable escalar real sobre el que se calculan TODAS las métricas, uniforme entre familias:
      · FASE → cos(2πS/K), la proyección del vector unitario de fase (misma cantidad que usa la propia
        dinámica al promediar), un número real por sitio.
      · RD   → V, la concentración de la especie autocatalítica (la que forma el patrón).
    Tener un único observable escalar permite que REAL y NULL, 2D y 3D, y las dos familias, pasen por
    exactamente el mismo instrumento."""
    if sustrato["tipo"] == "FASE":
        return np.cos(2 * np.pi * sustrato["S"] / p["K"])
    return sustrato["V"].copy()


# ====================================================================================================
# 2) FILTRO DE ADMISIÓN P1-P5 — verificado explícitamente sobre las familias 2D/3D, no asumido.
#    P1 y P4 son empíricos (se corre la dinámica chica de verdad); P3 y P5 reusan la lógica del
#    generador congelado (mismos patrones de coordenadas prohibidos, mismos nombres prohibidos).
# ====================================================================================================
_L_CHEQUEO = 32        # grilla chica sólo para el filtro (NO es la L del barrido)
_PASOS_CHEQUEO = 8


def _burnin_chequeo(p):
    """Pasos de asentamiento antes de medir P1/P4. Sin esto se estaría midiendo la memoria de un campo
    recién sorteado al azar (donde ninguna dinámica ha tenido tiempo de imprimir nada) en vez de la
    memoria del régimen en el que el campo realmente vive. Es 1/10 del presupuesto de pasos de la
    propia configuración, acotado para que el filtro siga siendo barato."""
    return int(min(600, max(20, p["n_pasos"] // 10)))


def chequear_P1_persistencia_2d(p, ndim, rng):
    """P1 — persistencia mínima: la dinámica REAL encadenada (cada paso usa el estado del anterior)
    predice el estado siguiente MEJOR que un control SIN MEMORIA (mismo estado con los valores
    re-barajados al azar, es decir: la misma distribución de valores pero sin ninguna relación con
    quién era quién en el paso anterior). Mismo criterio y mismo margen (>0.02) que
    `GEN.chequear_P1_persistencia` usa en 1D, con un asentamiento previo (ver `_burnin_chequeo`)."""
    s = construir(p, _L_CHEQUEO, ndim, rng)
    s, _ = evolucionar(s, p, rng, _burnin_chequeo(p))
    corte = float(np.median(observable_escalar(s, p)))
    aciertos_real = aciertos_sinmem = total = 0
    for _ in range(_PASOS_CHEQUEO):
        alto_antes = observable_escalar(s, p) > corte
        s, _g = evolucionar(s, p, rng, 1)
        u_desp = observable_escalar(s, p)
        aciertos_real += int(np.sum(alto_antes == (u_desp > corte)))
        sin_memoria = rng.permutation(u_desp.ravel()).reshape(alto_antes.shape) > corte
        aciertos_sinmem += int(np.sum(alto_antes == sin_memoria))
        total += alto_antes.size
    tasa_real, tasa_sinmem = aciertos_real / total, aciertos_sinmem / total
    return (tasa_real > tasa_sinmem + 0.02,
            f"persistencia_real={tasa_real:.3f} vs sin_memoria={tasa_sinmem:.3f} "
            f"(burn-in={_burnin_chequeo(p)} pasos)")


def chequear_P2_diferencia_2d(p):
    """P2 — diferencia operable: el estado no es un escalar homogéneo.
      · FASE: fase en Z_K con K>1 → dimensión interna de diferencia (orientación), acoplable.
      · RD  : DOS especies U,V distinguibles y acopladas → el 'tipo' de sustancia es la diferencia."""
    if p["familia"] == "FASE":
        return p["K"] > 1, f"K={p['K']} (alfabeto de fase = dimensión interna de diferencia)"
    return True, "dos especies U,V acopladas (tipo de sustancia = dimensión interna de diferencia)"


def chequear_P3_localidad_2d():
    """P3 — localidad relacional sin coordenadas: se inspecciona el CÓDIGO de las funciones que
    construyen el sustrato y aplican la actualización, buscando los MISMOS patrones de coordenadas que
    usa el generador congelado (`GEN.chequear_P3_localidad`). Los vecinos salen siempre de
    desplazamientos fijos por eje (np.roll), nunca de índice global ni de (x,y,z).
    Se inspeccionan SÓLO las funciones de sustrato/dinámica — las funciones de MEDICIÓN quedan fuera a
    propósito y eso se declara en el informe: box-counting y masa-radio sí usan geometría de grilla,
    pero del lado del instrumento, no de la regla."""
    funciones = [desplazamientos_grilla, construir_FASE, paso_FASE, construir_RD, _laplaciano,
                 paso_RD, construir, evolucionar, observable_escalar]
    fuente = "\n".join(_codigo_sin_docstrings(f) for f in funciones)
    patrones = ["pos[", "posiciones[", "coord", "xy=", "(x, y)", "embedding"]
    hallados = [pat for pat in patrones if pat in fuente]
    return (len(hallados) == 0,
            "sin patrones de coordenadas en el CÓDIGO de sustrato/dinámica (docstrings excluidas)"
            if not hallados else f"sospechosos: {hallados}")


def _codigo_sin_docstrings(f):
    """Devuelve el código ejecutable de una función, SIN sus docstrings ni comentarios. Hace falta
    porque los propios comentarios de este archivo hablan de 'coordenadas' para explicar que NO se
    usan, y eso daría un falso positivo del chequeo P3 consigo mismo (el generador congelado resuelve
    el mismo problema excluyendo su propia función de chequeo de la inspección). Lo que P3 exige es
    que la LEY DE ACTUALIZACIÓN no lea coordenadas — eso vive en el código, no en la prosa."""
    lineas = []
    for linea in inspect.getsource(f).splitlines():
        sin_com = linea.split("#", 1)[0]
        lineas.append(sin_com)
    codigo = "\n".join(lineas)
    for comilla in ('"""', "'''"):                       # elimina bloques de docstring completos
        partes = codigo.split(comilla)
        codigo = "".join(partes[::2]) if len(partes) > 1 else codigo
    return codigo


def chequear_P4_reciprocidad_2d(p, ndim, rng):
    """P4 — interacción recíproca, verificada EMPÍRICAMENTE en dos niveles:
      (a) espacial: el conjunto de desplazamientos es cerrado bajo negación (si i lee a j, j lee a i);
      (b) interno (sólo RD): se perturba U en un sitio y se verifica que V cambia en el paso siguiente,
          y viceversa — que el acople de especies va en los dos sentidos, no es campo externo.
          La perturbación se aplica en el sitio con MÁS V tras el asentamiento: el acople U↔V es el
          término U·V², así que en un sitio donde V=0 el acople está apagado por construcción y el test
          daría un falso negativo (medir la reciprocidad donde la reacción ni siquiera ocurre)."""
    despl = desplazamientos_grilla(ndim, p.get("vecindad", 4 if ndim == 2 else 6))
    simetrico = all(tuple(-np.array(d)) in set(despl) for d in despl)
    detalle = f"desplazamientos cerrados bajo negación={simetrico} ({len(despl)} vecinos)"
    if p["familia"] == "RD":
        s0 = construir(p, _L_CHEQUEO, ndim, np.random.default_rng(7))
        s0, _ = evolucionar(s0, p, np.random.default_rng(9), _burnin_chequeo(p))
        sitio = np.unravel_index(int(np.argmax(s0["V"])), s0["V"].shape)
        def _clon(delta_U=0.0, delta_V=0.0):
            c = dict(s0); c["U"] = s0["U"].copy(); c["V"] = s0["V"].copy()
            c["U"][sitio] += delta_U; c["V"][sitio] += delta_V
            return evolucionar(c, p, np.random.default_rng(11), 1)[0]
        base = _clon()
        traU = _clon(delta_U=0.05)
        traV = _clon(delta_V=0.05)
        u_afecta_v = float(np.max(np.abs(traU["V"] - base["V"])))
        v_afecta_u = float(np.max(np.abs(traV["U"] - base["U"])))
        reciproco = (u_afecta_v > 1e-12) and (v_afecta_u > 1e-12)
        detalle += (f"; en el sitio de V máximo (V={float(s0['V'][sitio]):.3f}): "
                    f"U→V={u_afecta_v:.2e}, V→U={v_afecta_u:.2e} -> "
                    + ("recíproco (ambos >0)" if reciproco else
                       "NO recíproco: el patrón murió (V≈0 en todo el dominio), el acople U·V² queda "
                       "apagado y la influencia deja de ir en los dos sentidos"))
        return (simetrico and reciproco), detalle
    return simetrico, detalle


def chequear_P5_sin_hornear_2d(p):
    """P5 — sin valores físicos horneados: se reusa el MISMO conjunto de nombres prohibidos del
    generador congelado (`GEN._PROHIBIDOS_P5`), y se exige que la descripción de la regla tenga
    <500 caracteres (complejidad algorítmica acotada). Los parámetros de acá (J, ruido, Du, Dv, F, k,
    dt) son constantes adimensionales del propio motor: no hay G, ℏ, c, masa ni escala de longitud —
    la unidad de longitud ES el paso de grilla, que es la relación de vecindad, no una escala física."""
    hallados = set(p.keys()) & GEN._PROHIBIDOS_P5
    ok_long = len(p["descripcion"]) < 500
    return (len(hallados) == 0 and ok_long,
            f"nombres prohibidos={sorted(hallados) or 'ninguno'}; len(desc)={len(p['descripcion'])}<500={ok_long}")


def aplicar_filtro(p, ndim, seed_chequeo):
    """Corre los 5 chequeos y anota resultado + detalle por P en el propio dict de la configuración."""
    rng = np.random.default_rng(seed_chequeo)
    res = {
        "P1": chequear_P1_persistencia_2d(p, ndim, rng),
        "P2": chequear_P2_diferencia_2d(p),
        "P3": chequear_P3_localidad_2d(),
        "P4": chequear_P4_reciprocidad_2d(p, ndim, rng),
        "P5": chequear_P5_sin_hornear_2d(p),
    }
    p["filtro"] = {k: bool(v[0]) for k, v in res.items()}
    p["filtro_detalle"] = {k: v[1] for k, v in res.items()}
    p["admitida"] = all(v[0] for v in res.values())
    p["motivo_descarte"] = None if p["admitida"] else \
        "; ".join(f"{k}={res[k][1]}" for k, v in res.items() if not v[0])
    return p


# ====================================================================================================
# 3) INSTRUMENTOS — todos dimensión-genéricos (2D y 3D), todos sobre el array del campo, CERO grafo
#    derivado. Cada uno se aplica idéntico al campo REAL y al campo BARAJADO (NULL).
# ====================================================================================================
def espectro_potencia_radial(u, n_bins=None):
    """P(k): espectro de potencia promediado en cáscaras de |k| constante.
    Se quita la media (la componente DC no informa sobre estructura) y se normaliza por el total, así
    P(k) es comparable entre campos con distinta amplitud. Devuelve (k, P(k)).
    Cómo leerlo: un PICO nítido a k>0 = hay una escala característica (patrón con longitud de onda
    propia); un espectro plano = ruido blanco (sin escala); una recta en log-log = ley de potencias
    (estructura en todas las escalas, sin una preferida)."""
    v = u - np.mean(u)
    F = np.fft.fftn(v)
    P = np.abs(F) ** 2
    L = u.shape[0]
    ejes = [np.fft.fftfreq(n, d=1.0 / n) for n in u.shape]     # frecuencias en unidades de "modos"
    malla = np.meshgrid(*ejes, indexing="ij")
    kmod = np.sqrt(sum(m ** 2 for m in malla))
    kmax = L // 2
    n_bins = n_bins or kmax
    idx = np.clip(np.round(kmod).astype(int), 0, n_bins)
    suma = np.bincount(idx.ravel(), weights=P.ravel(), minlength=n_bins + 1)
    cuenta = np.bincount(idx.ravel(), minlength=n_bins + 1).astype(float)
    Pk = suma[1:kmax + 1] / np.maximum(cuenta[1:kmax + 1], 1)   # se descarta k=0 (DC)
    ks = np.arange(1, kmax + 1, dtype=float)
    total = Pk.sum()
    return ks, (Pk / total if total > 0 else Pk)


def descriptores_espectro(ks, Pk):
    """Resume P(k) en números comparables:
      · k_pico y su prominencia = P(k_pico)/mediana(P) — cuán por encima del fondo está el pico
      · nitidez = ancho a media altura del pico, dividido por k_pico (chico = pico afilado)
      · entropia_espectral normalizada en [0,1]: 1 = potencia repartida por igual entre todos los k
        (ruido blanco, sin escala); cerca de 0 = toda la potencia en unos pocos modos (patrón nítido)
      · exponente de ley de potencias: pendiente de log P vs log k en el tramo k∈[2, kmax/2]"""
    i_pico = int(np.argmax(Pk))
    k_pico = float(ks[i_pico])
    med = float(np.median(Pk))
    prominencia = float(Pk[i_pico] / med) if med > 0 else float("nan")
    mitad = Pk[i_pico] / 2.0
    izq = i_pico
    while izq > 0 and Pk[izq] > mitad:
        izq -= 1
    der = i_pico
    while der < len(Pk) - 1 and Pk[der] > mitad:
        der += 1
    ancho = float(ks[der] - ks[izq])
    nitidez = ancho / k_pico if k_pico > 0 else float("nan")
    p = Pk / Pk.sum() if Pk.sum() > 0 else Pk
    p_pos = p[p > 0]
    H = float(-np.sum(p_pos * np.log(p_pos)) / np.log(len(Pk)))
    lo, hi = 1, max(3, len(ks) // 2)
    x, y = np.log(ks[lo:hi]), np.log(np.maximum(Pk[lo:hi], 1e-300))
    exponente = float(np.polyfit(x, y, 1)[0])
    return dict(k_pico=k_pico, prominencia_pico=prominencia, nitidez_pico=nitidez,
                entropia_espectral=H, exponente_ley_potencias=exponente)


def anisotropia_angular(u, n_ang=36):
    """Espectro ANGULAR: ¿hay una dirección privilegiada? Sólo tiene sentido en 2D (en 3D se omite).
    Se reparte la potencia del anillo k∈[2, L/4] en `n_ang` sectores de ángulo (mod π, porque el
    espectro de un campo real es simétrico) y se mide cuán desparejo queda:
      · anisotropia = desvío estándar / media del perfil angular (coeficiente de variación). Se elige
        ésta como medida principal porque usa TODOS los sectores; el (max-min)/(max+min) depende de
        dos sectores extremos y con pocos píxeles por sector infla el valor del ruido puro (en el
        smoke test daba 0.59 para un campo barajado, que es isótropo por construcción).
      · aniso_maxmin = (max-min)/(max+min), se conserva para comparación.
    En un campo barajado el CV esperado es ≈1/√(píxeles por sector); lo que importa es la diferencia
    REAL vs NULL, no el valor absoluto."""
    if u.ndim != 2:
        return dict(anisotropia=float("nan"), aniso_maxmin=float("nan"), ang_pico=float("nan"))
    v = u - np.mean(u)
    P = np.abs(np.fft.fftn(v)) ** 2
    L = u.shape[0]
    kx = np.fft.fftfreq(L, d=1.0 / L)[:, None] * np.ones((1, L))
    ky = np.ones((L, 1)) * np.fft.fftfreq(L, d=1.0 / L)[None, :]
    kmod = np.sqrt(kx ** 2 + ky ** 2)
    anillo = (kmod >= 2) & (kmod <= L / 4)
    if anillo.sum() == 0:
        return dict(anisotropia=float("nan"), aniso_maxmin=float("nan"), ang_pico=float("nan"))
    ang = np.mod(np.arctan2(ky, kx), np.pi)
    sector = np.clip((ang / np.pi * n_ang).astype(int), 0, n_ang - 1)
    suma = np.bincount(sector[anillo].ravel(), weights=P[anillo].ravel(), minlength=n_ang)
    cuenta = np.bincount(sector[anillo].ravel(), minlength=n_ang).astype(float)
    perfil = suma / np.maximum(cuenta, 1)
    mx, mn, mu = float(perfil.max()), float(perfil.min()), float(perfil.mean())
    return dict(anisotropia=float(perfil.std() / mu) if mu > 0 else float("nan"),
                aniso_maxmin=(mx - mn) / (mx + mn) if (mx + mn) > 0 else float("nan"),
                ang_pico=float(np.argmax(perfil) / n_ang * 180.0))


def dimension_box_counting(u, ocupacion, cajas=(1, 2, 4, 8, 16, 32)):
    """Dimensión de box-counting del conjunto de nivel {sitio : u > cuantil(1-ocupacion)}.
    Se cuenta, para cada tamaño de caja b, cuántas cajas contienen al menos un sitio del conjunto, y se
    ajusta N(b) ~ b^(-D). Interpretación: D≈d (2 en 2D) = el conjunto está esparcido por todas partes
    (típico de ruido); D<d = el conjunto está concentrado en objetos (blobs, filamentos) y muchas
    cajas quedan vacías. En 1D esto era trivial; en 2D/3D sí discrimina."""
    corte = np.quantile(u, 1.0 - ocupacion)
    binario = u > corte
    L = u.shape[0]
    ns, bs = [], []
    for b in cajas:
        if L % b != 0:
            continue
        nueva = binario.reshape(sum([[L // b, b] for _ in range(u.ndim)], []))
        ejes_internos = tuple(range(1, 2 * u.ndim, 2))
        ocupadas = int(np.any(nueva, axis=ejes_internos).sum())
        if ocupadas > 0:
            ns.append(ocupadas); bs.append(b)
    if len(bs) < 3:
        return float("nan")
    return float(-np.polyfit(np.log(bs), np.log(ns), 1)[0])


def entropia_bloques(u, lado=2):
    """Entropía de configuración ESPACIAL: se binariza el campo por su mediana y se cuenta la
    frecuencia de cada patrón posible de bloque lado^d (en 2D con lado=2 hay 16 patrones). La entropía
    de esa distribución, normalizada a [0,1], dice si el campo está ESTRUCTURADO o HOMOGENEIZADO:
      1.0 = todos los patrones igual de probables (los vecinos no dicen nada del sitio → desorden)
      →0  = pocos patrones dominan (bloques todos-1 y todos-0 → dominios, estructura)
    Es la métrica clave para 'el campo se estructura o se homogeniza con el tiempo'. Ojo: la entropía
    de los VALORES (histograma) NO sirve acá, porque el NULL barajado la conserva exacta por
    construcción — por eso se mide la de BLOQUES, que sí ve el orden espacial."""
    b = (u > np.median(u)).astype(np.uint8)
    L = u.shape[0]
    if L % lado != 0:
        return float("nan")
    nueva = b.reshape(sum([[L // lado, lado] for _ in range(u.ndim)], []))
    orden = tuple(range(0, 2 * u.ndim, 2)) + tuple(range(1, 2 * u.ndim, 2))
    bloques = np.transpose(nueva, orden).reshape(-1, lado ** u.ndim)
    pesos = 2 ** np.arange(lado ** u.ndim)
    codigos = bloques @ pesos
    cuentas = np.bincount(codigos, minlength=2 ** (lado ** u.ndim)).astype(float)
    p = cuentas / cuentas.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(2 ** (lado ** u.ndim)))


def barrido_percolacion(u, cuantiles=None):
    """PERCOLACIÓN de conjuntos de nivel, con BARRIDO de umbral (nunca uno solo — lección de O1-C).
    Para cada ocupación q (fracción de sitios encendidos), se toma {u > cuantil(1-q)} y se etiquetan sus
    componentes conexas usando la MISMA adyacencia de la grilla que usa la dinámica (4-conexo en 2D,
    6-conexo en 3D). Se reporta, por q:
      · frac_gigante_del_conjunto = tamaño de la componente mayor / nº de sitios encendidos
      · n_componentes
    Lectura: en un campo barajado (sitios independientes) la componente gigante aparece recién cerca de
    la ocupación crítica de percolación de sitios (≈0.593 en la red cuadrada 2D, ≈0.312 en la cúbica
    3D). Si el campo REAL percola MUCHO ANTES, es porque los sitios altos están agrupados: eso es
    estructura espacial, medida sin ningún grafo derivado."""
    cuantiles = cuantiles or np.round(np.arange(0.05, 0.96, 0.05), 2)
    estructura = ndimage.generate_binary_structure(u.ndim, 1)   # conectividad de cara = vecindad de la dinámica
    filas = []
    for q in cuantiles:
        corte = np.quantile(u, 1.0 - q)
        binario = u > corte
        n_on = int(binario.sum())
        if n_on == 0:
            filas.append((float(q), 0.0, 0)); continue
        etiquetas, n_comp = ndimage.label(binario, structure=estructura)
        tam = np.bincount(etiquetas.ravel())[1:]
        filas.append((float(q), float(tam.max() / n_on), int(n_comp)))
    return filas


def resumen_percolacion(filas_real, filas_null):
    """Resume el barrido en tres números:
      · q_c = primera ocupación donde la componente mayor se come >50% del conjunto encendido
      · área bajo la curva de fracción-gigante (integral sobre el barrido) — resistente a elegir un q
      · máxima separación REAL-NULL a lo largo del barrido, y en qué q ocurre"""
    def _qc(filas):
        for q, g, _ in filas:
            if g > 0.5:
                return q
        return float("nan")
    g_real = np.array([f[1] for f in filas_real])
    g_null = np.array([f[1] for f in filas_null])
    qs = np.array([f[0] for f in filas_real])
    dif = g_real - g_null
    i = int(np.argmax(np.abs(dif)))
    return dict(qc_real=_qc(filas_real), qc_null=_qc(filas_null),
                auc_gigante_real=float(np.trapezoid(g_real, qs)),
                auc_gigante_null=float(np.trapezoid(g_null, qs)),
                max_dif_gigante=float(dif[i]), q_max_dif=float(qs[i]))


def escalamiento_masa_radio(u, n_centros=8, r_max=None, rng=None):
    """MASA-RADIO: cuánta 'masa' del campo hay dentro de un radio r de un centro, en función de r.
    Centros = los `n_centros` máximos locales más altos del campo (no puntos al azar: los máximos son
    los candidatos naturales a 'núcleo de estructura'). La masa usa el campo desplazado a mínimo 0 para
    que sea no-negativa. Se ajusta M(r) ~ r^alfa en el tramo r∈[2, r_max].
    Lectura: alfa ≈ d (2 en 2D) = la masa crece como el volumen → campo homogéneo, sin agrupamiento;
    alfa < d = la masa se concentra cerca del centro → agrupamiento; alfa > d sólo puede pasar si el
    centro está en un hueco. Las distancias son toroidales (envuelven el borde), igual que la
    adyacencia de la dinámica."""
    L = u.shape[0]
    r_max = r_max or L // 4
    campo = u - u.min()
    maxlocal = (campo == ndimage.maximum_filter(campo, size=3))
    idx = np.argwhere(maxlocal)
    if len(idx) == 0:
        return float("nan")
    valores = campo[tuple(idx.T)]
    orden = np.argsort(-valores)[:n_centros]
    centros = idx[orden]
    ejes = [np.arange(L) for _ in range(u.ndim)]
    alfas = []
    for c in centros:
        d2 = 0
        for eje in range(u.ndim):
            delta = np.abs(ejes[eje] - c[eje])
            delta = np.minimum(delta, L - delta)                        # distancia toroidal
            forma = [1] * u.ndim; forma[eje] = L
            d2 = d2 + (delta.reshape(forma)) ** 2
        dist = np.sqrt(d2)
        rs = np.arange(2, r_max + 1)
        masas = np.array([campo[dist <= r].sum() for r in rs])
        buenos = masas > 0
        if buenos.sum() >= 3:
            alfas.append(float(np.polyfit(np.log(rs[buenos]), np.log(masas[buenos]), 1)[0]))
    return float(np.mean(alfas)) if alfas else float("nan")


def longitud_correlacion(u):
    """ξ: longitud de correlación espacial, calculada por autocorrelación vía FFT (teorema de
    Wiener-Khinchin) y promediada en cáscaras radiales. Se devuelve el primer radio donde la
    autocorrelación normalizada cae bajo 1/e. Si nunca cae dentro del rango medido, se devuelve el
    radio máximo y se marca como saturada (coherencia sostenida hasta la escala más grande medida)."""
    v = u - np.mean(u)
    F = np.fft.fftn(v)
    ac = np.real(np.fft.ifftn(np.abs(F) ** 2))
    ac = ac / ac.ravel()[0] if ac.ravel()[0] != 0 else ac
    L = u.shape[0]
    ejes = [np.minimum(np.arange(L), L - np.arange(L)) for _ in range(u.ndim)]
    malla = np.meshgrid(*ejes, indexing="ij")
    r = np.sqrt(sum(m.astype(float) ** 2 for m in malla))
    rmax = L // 2
    idx = np.clip(np.round(r).astype(int), 0, rmax)
    suma = np.bincount(idx.ravel(), weights=ac.ravel(), minlength=rmax + 1)
    cuenta = np.bincount(idx.ravel(), minlength=rmax + 1).astype(float)
    perfil = suma / np.maximum(cuenta, 1)
    bajo = np.where(perfil < 1.0 / np.e)[0]
    if len(bajo) == 0:
        return float(rmax), True
    return float(bajo[0]), False


def medir_todo(u, etiqueta, rng):
    """Aplica TODOS los instrumentos a un campo escalar y devuelve un dict plano con sufijo de etiqueta
    ('real' o 'null'), para que la fila del CSV tenga las dos mediciones lado a lado."""
    ks, Pk = espectro_potencia_radial(u)
    d = {}
    d.update({f"{k}_{etiqueta}": v for k, v in descriptores_espectro(ks, Pk).items()})
    d.update({f"{k}_{etiqueta}": v for k, v in anisotropia_angular(u).items()})
    for occ in (0.05, 0.10, 0.25, 0.50):
        d[f"boxdim_occ{int(occ*100)}_{etiqueta}"] = dimension_box_counting(u, occ)
    d[f"entropia_bloques_{etiqueta}"] = entropia_bloques(u)
    d[f"masa_radio_alfa_{etiqueta}"] = escalamiento_masa_radio(u, rng=rng)
    xi, sat = longitud_correlacion(u)
    d[f"xi_{etiqueta}"] = xi
    d[f"xi_saturada_{etiqueta}"] = sat
    d[f"std_valores_{etiqueta}"] = float(np.std(u))
    return d, (ks, Pk)


# ====================================================================================================
# 4) CONFIGURACIONES DEL BARRIDO — 10 FASE + 10 RD. Se listan explícitas (no al azar) para que el
#    barrido cubra a propósito los regímenes conocidos de cada familia y no quede a merced del sorteo.
# ====================================================================================================
def configuraciones_2D():
    """20 configuraciones. FASE barre acople J y ruido (de casi-determinista a casi-puro-ruido) más
    vecindad 4 vs 8. RD barre el plano (F, k) de Gray-Scott, que es donde viven los distintos regímenes
    de patrón (manchas, rayas, laberintos, ondas, y también la muerte del patrón)."""
    cfgs = []
    fase_params = [
        # (K, J, ruido, vecindad, sweeps) — de fuerte acople/poco ruido a débil acople/mucho ruido
        (6, 0.80, 0.05, 4, 400), (6, 0.80, 0.20, 4, 400), (6, 0.50, 0.10, 4, 400),
        (6, 0.50, 0.35, 4, 400), (6, 0.30, 0.10, 4, 400), (6, 0.30, 0.35, 4, 400),
        (4, 0.70, 0.15, 8, 400), (8, 0.70, 0.15, 8, 400), (6, 0.90, 0.02, 8, 400),
        (6, 0.20, 0.50, 4, 400),
    ]
    for i, (K, J, ru, vec, sw) in enumerate(fase_params):
        cfgs.append(dict(
            cfg_id=f"FASE{i:02d}", familia="FASE", K=K, J=J, noise=ru, vecindad=vec, n_pasos=sw,
            descripcion=(f"Campo de fase en Z_{K} sobre grilla toroidal 2D; actualizacion = "
                         f"(1-J)*fase + J*media_circular(vecinos de grilla, vecindad={vec}) + ruido "
                         f"gaussiano; J={J}, ruido={ru}. Sin coordenadas: vecinos por desplazamiento fijo."),
        ))
    rd_params = [
        # (F, k, Du, Dv, dt, pasos, ruido, siembra) — regímenes del plano de Gray-Scott. Los tres
        # últimos son CONTROLES a propósito (siembra rala, alimentación alta, remoción alta): si el
        # patrón muere, el campo queda constante y P1 lo descalifica solo — el filtro haciendo su
        # trabajo, en vez de que lo decida quien escribe la lista.
        (0.0350, 0.0650, 0.16, 0.08, 1.0, 4000, 0.0,   0.30),   # manchas/coral (verificado vivo)
        (0.0300, 0.0620, 0.16, 0.08, 1.0, 4000, 0.0,   0.30),   # laberinto (verificado vivo)
        (0.0400, 0.0600, 0.16, 0.08, 1.0, 4000, 0.0,   0.30),   # solitones/manchas densas (verificado vivo)
        (0.0300, 0.0570, 0.16, 0.08, 1.0, 4000, 0.0,   0.30),   # rayas
        (0.0450, 0.0630, 0.16, 0.08, 1.0, 4000, 0.0,   0.30),   # manchas dispersas
        (0.0250, 0.0600, 0.16, 0.08, 1.0, 4000, 0.0,   0.30),   # cerca del borde del régimen
        (0.0350, 0.0650, 0.16, 0.08, 1.0, 4000, 1e-3,  0.30),   # mismo régimen, dinámica estocástica
        (0.0350, 0.0650, 0.16, 0.08, 1.0, 4000, 0.0,   0.02),   # CONTROL: siembra rala (el patrón no nuclea)
        (0.0900, 0.0600, 0.16, 0.08, 1.0, 4000, 0.0,   0.30),   # CONTROL: alimentación alta
        (0.0300, 0.0850, 0.16, 0.08, 1.0, 4000, 0.0,   0.30),   # CONTROL: remoción alta
    ]
    for i, (F, kk, Du, Dv, dt, pasos, ru, siembra) in enumerate(rd_params):
        cfgs.append(dict(
            cfg_id=f"RD{i:02d}", familia="RD", F=F, kmata=kk, Du=Du, Dv=Dv, dt=dt, n_pasos=pasos,
            noise=ru, siembra=siembra, vecindad=4,
            descripcion=(f"Reaccion-difusion de dos especies U,V en grilla toroidal 2D: U+2V->3V; "
                         f"dU=Du*lap(U)-U*V^2+F*(1-U); dV=Dv*lap(V)+U*V^2-(F+k)*V; "
                         f"F={F}, k={kk}, Du={Du}, Dv={Dv}, dt={dt}, ruido={ru}. Laplaciano por "
                         f"desplazamientos fijos de grilla; sin coordenadas ni escala fisica."),
        ))
    return cfgs


def configuraciones_3D():
    """Bloque 3D reducido (6 configuraciones): 3 de FASE y 3 de RD, elegidas entre los regímenes que en
    2D resultaron más informativos. Grilla más chica porque el costo crece como L^3."""
    cfgs = []
    for i, (K, J, ru, sw) in enumerate([(6, 0.80, 0.05, 250), (6, 0.50, 0.20, 250), (6, 0.30, 0.35, 250)]):
        cfgs.append(dict(cfg_id=f"F3D{i:02d}", familia="FASE", K=K, J=J, noise=ru, vecindad=6, n_pasos=sw,
                         descripcion=(f"Campo de fase en Z_{K} sobre grilla toroidal 3D (vecindad 6); "
                                      f"J={J}, ruido={ru}; vecinos por desplazamiento fijo, sin coordenadas.")))
    for i, (F, kk) in enumerate([(0.0350, 0.0650), (0.0300, 0.0620), (0.0400, 0.0600)]):
        cfgs.append(dict(cfg_id=f"R3D{i:02d}", familia="RD", F=F, kmata=kk, Du=0.16, Dv=0.08, dt=1.0,
                         n_pasos=4000, noise=0.0, siembra=0.30, vecindad=6,
                         descripcion=(f"Reaccion-difusion U,V en grilla toroidal 3D (laplaciano de 6 vecinos): "
                                      f"U+2V->3V, F={F}, k={kk}, Du=0.16, Dv=0.08. Sin coordenadas.")))
    return cfgs


# ====================================================================================================
# 5) UNA CORRIDA = una configuración + una semilla: evoluciona, mide REAL, baraja, mide NULL.
# ====================================================================================================
def correr_una(p, ndim, L, seed, guardar_campo=False):
    rng = np.random.default_rng(seed)
    s = construir(p, L, ndim, rng)
    # snapshots para la pregunta 'se estructura o se homogeniza': 5 instantes repartidos en el tiempo
    pasos = p["n_pasos"]
    instantes = sorted({max(1, int(pasos * f)) for f in (0.02, 0.10, 0.25, 0.50, 1.00)})
    s, guardados = evolucionar(s, p, rng, pasos, snapshots=set(instantes))
    u_real = observable_escalar(s, p)

    # ---- NULL: el MISMO campo con las posiciones barajadas (conserva la distribución de valores exacta) ----
    rng_null = np.random.default_rng(seed + 777_000)
    u_null = rng_null.permutation(u_real.ravel()).reshape(u_real.shape)

    fila = dict(cfg_id=p["cfg_id"], familia=p["familia"], ndim=ndim, L=L, seed=seed, n_pasos=pasos)
    for clave in ("K", "J", "noise", "vecindad", "F", "kmata", "Du", "Dv", "dt", "siembra"):
        fila[clave] = p.get(clave, "")

    d_real, (ks, Pk_real) = medir_todo(u_real, "real", rng)
    d_null, (_, Pk_null) = medir_todo(u_null, "null", rng)
    fila.update(d_real); fila.update(d_null)

    perc_real = barrido_percolacion(u_real)
    perc_null = barrido_percolacion(u_null)
    fila.update(resumen_percolacion(perc_real, perc_null))

    # ---- trayectoria de la entropía de bloques: ¿se estructura (baja) o se homogeniza (sube)? ----
    traj = {t: entropia_bloques(g) for t, g in guardados.items()}
    fila["entropia_traj"] = json.dumps({str(t): round(v, 5) for t, v in sorted(traj.items())})
    ts = sorted(traj)
    fila["entropia_inicio"] = round(traj[ts[0]], 5)
    fila["entropia_final"] = round(traj[ts[-1]], 5)
    fila["entropia_delta"] = round(traj[ts[-1]] - traj[ts[0]], 5)

    extras = dict(perc_real=perc_real, perc_null=perc_null, ks=ks.tolist(),
                  Pk_real=Pk_real.tolist(), Pk_null=Pk_null.tolist())
    if guardar_campo:
        extras["campo"] = u_real
        extras["campo_null"] = u_null
    return fila, extras


# ====================================================================================================
# 6) BARRIDO COMPLETO
# ====================================================================================================
def barrido(ndim=2, L=128, semillas=(1, 2, 3, 4), prefijo="cs090_fase6_o2d_2d"):
    t0 = time.time()
    cfgs = configuraciones_2D() if ndim == 2 else configuraciones_3D()

    # ---------------- filtro P1-P5, verificado de verdad, ANTES de correr nada ----------------
    print(f"\n{'='*100}\nFILTRO DE ADMISIÓN P1-P5 (ndim={ndim}) — verificado, no asumido\n{'='*100}")
    filas_filtro, admitidas = [], []
    for c in cfgs:
        c = aplicar_filtro(dict(c), ndim, seed_chequeo=hash(c["cfg_id"]) % (1 << 30))
        filas_filtro.append(dict(cfg_id=c["cfg_id"], familia=c["familia"], admitida=c["admitida"],
                                 **{f"{k}_ok": v for k, v in c["filtro"].items()},
                                 **{f"{k}_detalle": v for k, v in c["filtro_detalle"].items()},
                                 motivo_descarte=c["motivo_descarte"]))
        estado = "ADMITIDA" if c["admitida"] else f"DESCARTADA ({c['motivo_descarte']})"
        print(f"  {c['cfg_id']:<8} {estado}")
        print(f"           P1 {c['filtro_detalle']['P1']}")
        print(f"           P4 {c['filtro_detalle']['P4']}")
        if c["admitida"]:
            admitidas.append(c)
    print(f"\n  admitidas={len(admitidas)}/{len(cfgs)}")

    # ---------------- barrido ----------------
    print(f"\n{'='*100}\nBARRIDO ndim={ndim} L={L}: {len(admitidas)} configuraciones × {len(semillas)} semillas\n{'='*100}")
    filas, guardados_extra = [], {}
    for c in admitidas:
        for j, sd in enumerate(semillas):
            t1 = time.time()
            guardar = (j == 0)     # se guarda el campo de la primera semilla de cada config, para los PNG
            fila, extras = correr_una(c, ndim, L, seed=sd * 1000 + hash(c["cfg_id"]) % 997, guardar_campo=guardar)
            filas.append(fila)
            if guardar:
                guardados_extra[c["cfg_id"]] = extras
            print(f"  {c['cfg_id']:<8} s{sd}  k_pico={fila['k_pico_real']:>5.1f} "
                  f"prom={fila['prominencia_pico_real']:>7.2f} (null {fila['prominencia_pico_null']:>5.2f})  "
                  f"H_esp={fila['entropia_espectral_real']:.3f} (null {fila['entropia_espectral_null']:.3f})  "
                  f"boxD25={fila['boxdim_occ25_real']:.3f} (null {fila['boxdim_occ25_null']:.3f})  "
                  f"H_bloq={fila['entropia_bloques_real']:.3f} (null {fila['entropia_bloques_null']:.3f})  "
                  f"qc={fila['qc_real']} (null {fila['qc_null']})  xi={fila['xi_real']:.0f}  "
                  f"({time.time()-t1:.1f}s)")

    # ---------------- CSV crudo ----------------
    campos = list(filas[0].keys())
    ruta = f"{_HERE}/{prefijo}_raw.csv"
    with open(ruta, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos); w.writeheader(); w.writerows(filas)
    print(f"\n  -> {prefijo}_raw.csv ({len(filas)} filas)")

    ruta_f = f"{_HERE}/{prefijo}_filtro_P1P5.csv"
    with open(ruta_f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas_filtro[0].keys())); w.writeheader(); w.writerows(filas_filtro)
    print(f"  -> {prefijo}_filtro_P1P5.csv ({len(filas_filtro)} filas)")

    # ---------------- curvas de percolación y P(k) crudos (una por config, semilla 1) ----------------
    filas_perc, filas_pk = [], []
    for cid, ex in guardados_extra.items():
        for (q, g, nc), (q2, g2, nc2) in zip(ex["perc_real"], ex["perc_null"]):
            filas_perc.append(dict(cfg_id=cid, q=q, gigante_real=g, ncomp_real=nc,
                                   gigante_null=g2, ncomp_null=nc2))
        for k, pr, pn in zip(ex["ks"], ex["Pk_real"], ex["Pk_null"]):
            filas_pk.append(dict(cfg_id=cid, k=k, Pk_real=pr, Pk_null=pn))
    for nombre, fs in ((f"{prefijo}_percolacion.csv", filas_perc), (f"{prefijo}_espectros.csv", filas_pk)):
        with open(f"{_HERE}/{nombre}", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(fs[0].keys())); w.writeheader(); w.writerows(fs)
        print(f"  -> {nombre} ({len(fs)} filas)")

    print(f"\nTOTAL ndim={ndim}: {len(filas)} corridas en {time.time()-t0:.1f}s")
    return filas, guardados_extra


if __name__ == "__main__":
    ndim = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    if ndim == 2:
        filas, extras = barrido(ndim=2, L=128, semillas=(1, 2, 3, 4), prefijo="cs090_fase6_o2d_2d")
        np.save(f"{_HERE}/cs090_fase6_o2d_2d_campos.npy",
                {cid: {"campo": ex["campo"], "campo_null": ex["campo_null"]} for cid, ex in extras.items()},
                allow_pickle=True)
        print("  -> cs090_fase6_o2d_2d_campos.npy (campos de la semilla 1 por config, para los PNG)")
    else:
        filas, extras = barrido(ndim=3, L=48, semillas=(1, 2), prefijo="cs090_fase6_o2d_3d")
