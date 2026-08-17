#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS074 — Persistencia de una diferencia ínfima en un CAMPO CONTINUO bajo expansión
=================================================================================

############################################################################
## LA OBJECIÓN DEL DIRECTOR (Alexis) — POR QUÉ SE REESCRIBIÓ ESTE CÓDIGO   ##
############################################################################
##
## La versión anterior (cs074_ARCHIVO_version_discreta_INCORRECTA.py) estaba
## MAL DE RAÍZ. Arrancaba con N "puntitos", cada uno una ENTIDAD DISCRETA con
## identidad y etiqueta (+/-) desde el paso cero. Eso presupone que la realidad
## VIENE EN PEDAZOS — que hay "cosas" contables antes de que exista nada.
## Es exactamente el contrabando que este experimento debe evitar: mete en la
## PREMISA (la realidad es discreta) lo único que debía EMERGER como RESULTADO.
##
## LA CORRECCIÓN — LA ANALOGÍA DE LA SUPERFICIE SOLAR:
## Una MANCHA SOLAR no es una "cosa" dentro del Sol. No está hecha de partículas
## distintas, no tiene borde, no es un objeto que se pueda contar. Es EL MISMO
## PLASMA a una TEMPERATURA DIFERENTE del resto — una REGIÓN del campo de
## temperatura de la estrella con un valor distinto. La mancha ES un gradiente
## dentro de un campo continuo, NO un conjunto de cosas.
##
## Así es la diferencia primordial: la singularidad es un CAMPO (una magnitud
## continua, "una fluctuación sobre un fondo neutro"). La diferencia inicial NO
## es "más puntitos de un tipo"; es que el campo NO ES PERFECTAMENTE UNIFORME —
## una región vale un poco distinto que el fondo, con contraste ε. Como la mancha.
##
## CÓMO SE EXPRESA ESTO EN EL CÓDIGO (línea por línea, la diferencia clave):
##  - NO hay array de "entidades" con tag. Hay UN CAMPO ESCALAR phi: una magnitud
##    continua muestreada en puntos de una malla. Los puntos de la malla NO son
##    "cosas" — son como los píxeles de una foto del Sol: el MISMO campo leído en
##    muchos lugares. Un píxel no es una partícula.
##  - La diferencia = una VARIACIÓN DE AMPLITUD ε del campo sobre su fondo plano
##    (una "mancha": región a distinto valor). ε=0 -> campo plano = la Nada.
##  - La reabsorción = DIFUSIÓN: el campo tiende a re-aplanarse (la mancha
##    contagia su valor al entorno y el gradiente se borra). Mismo plasma
##    uniformándose. NO es "aniquilación de pares de cosas".
##  - La expansión = ESTIRAMIENTO del dominio: separa las regiones; si estira más
##    rápido de lo que la difusión re-aplana, el gradiente queda CONGELADO (no
##    por enfriarse, sino porque las regiones dejan de poder intercambiar).
##  - LA DISCRETIZACIÓN ES SALIDA, NO ENTRADA. La pregunta medida: ¿el campo
##    continuo SE CUANTIZA SOLO —se rompe en regiones discretas y estables que
##    persisten (ESO serían los "cuantos"/cierres)— o se disuelve a plano?
##    Que aparezcan pedazos, si aparecen, es EL HALLAZGO; jamás el punto de partida.
##
## Ninguna 'cosa' entra al modelo. Solo un campo continuo, su variación, y la
## competencia estiramiento-vs-difusión. Los cuantos, si existen, EMERGEN y SE MIDEN.
############################################################################

REGLA DEL DIRECTOR (anti-Shannon):
 - Único parámetro de ENTRADA: ε (amplitud de la variación del campo) y H (expansión).
 - CERO unidades de este universo: todo es adimensional (razones internas del campo).
 - La difusión (reabsorción) se MIDE del propio campo (H=0), no se impone.
 - Los cuantos/cierres EMERGEN y se MIDEN a la salida; NUNCA se imponen ni se cuentan
   como primitivos. "Cuanto" = unidad discreta en que el campo se cuantiza SOLO.
 - Persistencia SIEMPRE contra su NULL (misma energía, estructura del gradiente rota).
 - NO target-matching: prohibido validar por η, 7:1, Y_He ni número conocido.
 - Nulo = hallazgo. La curva entera se reporta, no se recorta.
 - El barrido (ya definido por el director) hace tres cosas a la vez: (1) barre
   expansión/enfriamiento vía r=H/D; (2) los cuantos que emergen se miden; (3) los
   cierres de esos cuantos (k=2,3,4,5,6,...) se miden a la salida, k libre, no impuesto.

############################################################################
## COHERENCIA FÍSICA — EL ENFRIAMIENTO ES LA CARA DE LA EXPANSIÓN          ##
############################################################################
## El universo no se enfría transfiriendo calor a un AFUERA (no hay afuera):
## se enfría por su propia EXPANSIÓN (enfriamiento adiabático). Enfriar y
## expandir NO son dos procesos: son el mismo. Por eso este código NO tiene
## módulo de "enfriamiento" separado — la expansión (cortar acoplamientos,
## estirar el dominio) es lo único que actúa; el enfriamiento se LEE de ella.
## La difusión de este modelo NO viola ese "no hay transferencia": es reversión
## INTERNA (el campo re-homogeneizándose consigo mismo), el canal por el que la
## diferencia PODRÍA borrarse — y es justo lo que la expansión debe ganarle.
## Ninguna transfiere a un exterior.
##
## LAS DOS ESCALAS DE MAGNITUD QUE EL BARRIDO ATRAVIESA (director):
##  (T) TEMPERATURA: de la singularidad ~10^20 K bajando a ~10^10 K.
##      ENVUELVE (no apunta a) la banda física real 10^15..10^12 K.
##  (t) TIEMPO: ventana 10^-20 .. 10^-4 s. ENVUELVE la banda física 10^-12..10^-6 s
##      que la física da para este proceso. Como es imposible simular a esa
##      velocidad, se escala el EXPONENTE del tiempo a pasos discretos: ese reloj
##      discreto ES la velocidad de expansión (se cambia la DISTANCIA por la TASA
##      DE ENFRIAMIENTO).
##
## !! GUARDIÁN G-ESCALAS-SON-MAPEO-NO-DINÁMICA !!
##  T(K) y t(s) son un RE-ETIQUETADO MONÓTONO de los ejes adimensionales, para
##  REPORTE y para mostrar que el barrido ENVUELVE la física — NO son motores.
##  NINGUNA regla dinámica lee un valor en Kelvin o en segundos: la dinámica sigue
##  gobernada SOLO por ε (amplitud) y r=H/D (expansión vs difusión). Que las
##  escalas sean MÁS ANCHAS que la física (10^20-10^10 ⊃ 10^15-10^12; 10^-20-10^-4
##  ⊃ 10^-12-10^-6) es el candado anti-target-matching: los valores reales son
##  puntos INTERIORES que la curva cruza, nunca blancos a reproducir.
############################################################################

============================================================================
!! DECISIONES DE MODELADO QUE EL EQUIPO DEBE AUDITAR (posible Shannon) !!
============================================================================
 (D1) La forma inicial de la variación del campo (la "mancha"). Aquí: una
      perturbación suave de amplitud ε sobre fondo plano, SIN forma privilegiada
      (se usa una función suave genérica). ¿Es neutral, o su forma sesga el
      resultado? El NULL debe absorber esto.
 (D2) Qué cuenta como "región discreta" al medir la cuantización de salida
      (umbral de segmentación del campo). Debe emerger del propio campo (p.ej.
      cruces por la media), NO de un valor puesto a mano.
 (D3) La medida de persistencia. CORREGIDA tras revisión del equipo: era
      std(final)/std(inicial) — INVARIANTE bajo permutación, así que el NULL
      (que baraja phi) no mordía y daba z=0 SIEMPRE (defecto 4). Ahora es la
      AUTOCORRELACIÓN ESPACIAL del campo final (corr a primer vecino): mide FORMA
      (estructura suave), no MAGNITUD. El NULL destruye la forma -> corr cae -> z
      discrimina. Se usa autocorr del final (no correlación con el inicial) para no
      atar el resultado a la mancha sembrada (esa forma es arbitraria). Va contra NULL.
============================================================================
"""
import numpy as np
import json, sys

# ----------------------------------------------------------------------------
# 0. ESCALAS DE MAGNITUD (MAPEO/REPORTE, NO DINÁMICA — ver G-ESCALAS)
# ----------------------------------------------------------------------------
# Estas constantes NO entran en ninguna regla de evolución. Solo re-etiquetan los
# ejes adimensionales a unidades físicas, para reportar y mostrar que el barrido
# ENVUELVE la banda física (los valores reales son interiores, no blancos).
T_SING = 1e20        # temperatura de la singularidad (K) — elegida MÁS alta que la física
T_FIN  = 1e10        # temperatura al final del barrido (K)
T_FIS_HI, T_FIS_LO = 1e15, 1e12    # banda física real de la era de quarks (referencia interior)
t_INI, t_FIN = 1e-20, 1e-4          # ventana temporal (s) — MÁS ancha que la física
t_FIS_HI, t_FIS_LO = 1e-12, 1e-6    # banda física real del proceso (referencia interior)

def reloj_fisico(paso, pasos):
    """Índice de paso discreto -> tiempo físico log-espaciado en [t_INI, t_FIN].
    El proceso real dura ~1e-12..1e-6 s, imposible de simular a esa velocidad; se escala
    el EXPONENTE del tiempo a pasos discretos. Ese reloj discreto es la velocidad de
    expansión. SOLO reporte: ninguna regla dinámica lo lee."""
    a, b = np.log10(t_INI), np.log10(t_FIN)
    return 10.0 ** (a + (b - a) * (paso / max(pasos, 1)))

def temperatura_fisica(frac_expandida):
    """Avance ADIMENSIONAL de la expansión (fracción de acoplamientos cortados, 0..1)
    -> temperatura log en [T_FIN, T_SING]. frac=0 -> 10^20 K (singularidad, sin expandir);
    frac=1 -> 10^10 K (todo expandido). La EXPANSIÓN ES el enfriamiento (adiabático):
    'cambiamos la distancia por la tasa de enfriamiento'. SOLO reporte: la dinámica no
    lee este valor; se calcula DESDE el estado (cuántos acoplamientos cortó la expansión)."""
    a, b = np.log10(T_SING), np.log10(T_FIN)
    return 10.0 ** (a + (b - a) * min(max(frac_expandida, 0.0), 1.0))

# ----------------------------------------------------------------------------
# 1. CAMPO INICIAL — continuo, con UNA variación de amplitud ε (la "mancha")
# ----------------------------------------------------------------------------
def campo_inicial(N, eps, rng):
    """
    El Todo = UN CAMPO ESCALAR continuo phi, muestreado en N puntos de malla.
    Los N puntos NO son entidades: son el MISMO campo leído en N lugares (píxeles
    de la foto del Sol). No hay identidades, no hay 'cosas', no hay tags.

    Fondo uniforme = 1 (todo el campo vale 1: 'toda la energía, cero diferencia').
    La ÚNICA diferencia = una VARIACIÓN de amplitud ε sobre ese fondo: una región
    del campo a distinto valor (la mancha solar). Con eps=0 -> campo plano = la Nada.

    La forma de la variación NO se privilegia (D1): se genera una perturbación
    suave (superposición de modos de fase aleatoria) y se re-escala para que su
    amplitud (contraste) sea exactamente eps. 'rng' solo fija las fases de esa
    rugosidad — no fabrica la diferencia (la magnitud la fija eps).
    """
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    fondo = np.ones(N, dtype=float)
    if eps <= 0.0:
        return fondo, x
    # rugosidad suave: suma de pocos modos de Fourier con fase aleatoria
    pert = np.zeros(N, dtype=float)
    for m in range(1, 6):                       # modos bajos = variación suave
        fase = rng.uniform(0, 2*np.pi)
        pert += np.sin(2*np.pi*m*x + fase) / m
    pert -= pert.mean()                          # media cero (no cambia el fondo)
    if pert.std() > 0:
        pert = pert / pert.std()                 # normaliza forma
    phi = fondo + eps * pert                     # amplitud de la variación = eps
    return phi, x

# ----------------------------------------------------------------------------
# 2. DIFUSIÓN — la reabsorción: el campo se re-aplana (borra el gradiente)
# ----------------------------------------------------------------------------
def paso_difusion(phi, activo):
    """
    Reabsorción = DIFUSIÓN sobre el campo continuo. La 'mancha' contagia su valor
    al entorno y el gradiente tiende a borrarse (el campo vuelve a plano = a la Nada).
    Es el mismo plasma uniformándose. Laplaciano discreto con conexiones VIVAS:
    solo difunde entre puntos aún acoplados (la expansión corta acoplamientos).
    'activo' = máscara booleana de aristas vivas entre vecinos (i, i+1) en anillo.
    NO hay tasa impuesta: el coeficiente es el natural del laplaciano (0.5·vecindad);
    la RAPIDEZ efectiva D emerge de cuántas conexiones siguen vivas (ver medir_D).
    """
    N = phi.size
    nuevo = phi.copy()
    for i in range(N):
        izq = (i - 1) % N
        der = (i + 1) % N
        vecinos = []
        if activo[izq]:      # arista (izq, i) viva
            vecinos.append(phi[izq])
        if activo[i]:        # arista (i, der) viva
            vecinos.append(phi[der])
        if vecinos:
            media_vec = np.mean(vecinos)
            nuevo[i] = phi[i] + 0.5 * (media_vec - phi[i])   # relajación difusiva
    return nuevo

# ----------------------------------------------------------------------------
# 3. EXPANSIÓN — estira el dominio: corta acoplamientos (irreversibles)
# ----------------------------------------------------------------------------
def paso_expansion(activo, H, rng):
    """
    Expansión = SEPARACIÓN del dominio: con intensidad H se cortan acoplamientos
    entre regiones vecinas (fracción H de las aristas vivas por paso). Una vez
    cortada, NO vuelve: las dos regiones ya no pueden intercambiar -> el gradiente
    entre ellas queda CONGELADO. H adimensional (fracción de aristas vivas/paso).
    """
    viv = np.where(activo)[0]
    if viv.size == 0:
        return activo
    ncorte = int(round(min(H, 1.0) * viv.size))
    if ncorte <= 0:
        return activo
    sel = rng.choice(viv, size=ncorte, replace=False)
    activo[sel] = False
    return activo

def evolucionar(phi, activo, H, pasos, rng, null=False):
    """
    Un proceso, no una sucesión: cada paso DIFUNDE (re-aplana) y EXPANDE (estira)
    a la vez. Es la carrera: ¿la difusión borra el gradiente antes de que la
    expansión lo congele aislando regiones?

    NULL LEGÍTIMO: la dinámica real corre COMPLETA e IDÉNTICA. SOLO AL FINAL, en el
    brazo null, se BARAJA el campo (se permuta phi entre las posiciones) UNA vez:
    destruye la ESTRUCTURA ESPACIAL del gradiente conservando EXACTAMENTE la misma
    energía y el mismo histograma de valores. Responde: ¿lo que persiste es la
    FORMA del gradiente, o sólo su magnitud/valores? (No baraja cada paso: eso
    cambiaría la dinámica y fabricaría persistencia — defecto ya identificado.)
    """
    contraste0 = phi.std()                        # amplitud inicial de la variación
    for _ in range(pasos):
        phi = paso_difusion(phi, activo)
        activo = paso_expansion(activo, H, rng)
    if null:
        phi = rng.permutation(phi)                # rompe la forma, conserva valores
    return phi, activo, contraste0

# ----------------------------------------------------------------------------
# 4. D EMERGENTE — la difusividad se MIDE del propio campo (H=0)
# ----------------------------------------------------------------------------
def medir_D(N, eps, seed):
    """
    Rapidez intrínseca de re-aplanamiento del campo: con H=0 (sin expansión),
    ¿cuánto cae el contraste (std de phi) en un paso de difusión pura?
    D = fracción de contraste borrada por paso. NO es un número puesto; sale del
    propio campo. r = H / D será el eje adimensional del barrido.
    """
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    c0 = phi.std()
    if c0 <= 0:
        return 0.0
    phi1 = paso_difusion(phi, activo)
    c1 = phi1.std()
    return max(0.0, (c0 - c1) / c0)

# ----------------------------------------------------------------------------
# 5. PERSISTENCIA y CUANTIZACIÓN (medidas a la salida, NO impuestas)
# ----------------------------------------------------------------------------
def persistencia(phi, contraste0):
    """
    P = FORMA x MAGNITUD = autocorr_espacial_primer_vecino(phi) * (var(phi)/contraste0^2).
    Mide si sobrevivió una DIFERENCIA ESTRUCTURADA: un gradiente suave (forma) que
    además conserva amplitud (magnitud). Distingue TRES casos:
      - Expansión gana (gradiente CONGELADO): suave (autocorr≈1) x amplitud viva -> P alto.
      - Difusión gana (campo APLANADO a casi-constante): suave (autocorr≈1) pero
        amplitud≈0 -> P≈0. (Lo mata el factor de magnitud.)
      - NULL (phi permutado): amplitud intacta pero forma destruida (autocorr≈0)
        -> P≈0. (Lo mata el factor de forma.)

    POR QUÉ ESTOS DOS FACTORES (historia de los defectos, para el equipo):
      * std solo (v1) -> INVARIANTE a permutación -> NULL no mordía -> z=0 SIEMPRE (Defecto 4).
      * autocorr-primer-vecino solo (v2) -> NO distingue congelado de aplanado: un campo
        casi-constante (difusión ganó) también es suave -> autocorr≈1 -> P alto falso.
        (Defecto 5, cazado por la revisión: "¿el campo es suave?" no es "¿persistió la
        diferencia?".)
      * autocorr-larga-distancia (lag=N/2) x var (remedio propuesto) -> el lag=N/2
        ANTI-correlaciona para modos suaves de baja frecuencia (p.ej. m=1: phi y
        phi(x+N/2) opuestos) -> P=0 en el propio caso de éxito. Descartado por paridad.
      * FIX ADOPTADO: autocorr a PRIMER vecino (robusta, sin paridad; el NULL la derrumba)
        MULTIPLICADA por la varianza normalizada (mata el aplanado). El NULL sigue
        mordiendo vía el factor de forma. No es el std invariante: la magnitud va
        multiplicada por una forma que la permutación destruye.

    Anti-Shannon: ambos factores son razones MEDIDAS del propio campo; lag=1 es la
    escala mínima natural (vecino inmediato), no un parámetro ajustable. Sin números a mano.
    Se usa autocorr del campo FINAL (no correlación con el inicial) para no atar el
    resultado a la mancha sembrada (forma arbitraria) = target-matching.

    Con eps=0 no hay variación: campo plano -> std 0 -> P=0.
    """
    if contraste0 <= 0 or phi.std() <= 1e-12:
        return 0.0
    c = np.corrcoef(phi, np.roll(phi, 1))[0, 1]        # FORMA: suavidad a primer vecino
    if not np.isfinite(c):
        c = 0.0
    c = max(0.0, c)
    v = float(phi.var() / (contraste0 ** 2))            # MAGNITUD: amplitud sobreviviente
    return float(c * v)

def detectar_cuantizacion(phi, activo):
    """
    CUANTIZACIÓN EMERGENTE (salida, no entrada): ¿el campo continuo se rompió en
    REGIONES DISCRETAS estables? Una 'región' = tramo conexo (por aristas vivas)
    cuyo valor está del mismo lado de la media global (por encima / por debajo).
    El umbral es la MEDIA del propio campo (emerge, D2), no un número a mano.
    Devuelve histograma {k: conteo} de tamaños de región k (en nº de puntos de malla).
    k grande = el campo sigue continuo (una región enorme); muchos k chicos =
    se cuantizó en pedazos. Que aparezcan pedazos discretos ES el hallazgo.
    """
    N = phi.size
    media = phi.mean()
    signo = phi >= media
    hist = {}
    visto = np.zeros(N, dtype=bool)
    for start in range(N):
        if visto[start]:
            continue
        # crecer región conexa por aristas vivas con mismo signo
        comp = [start]; visto[start] = True; pila = [start]
        while pila:
            u = pila.pop()
            for v in ((u-1) % N, (u+1) % N):
                arista_viva = activo[u] if v == (u+1) % N else activo[(u-1) % N]
                if (not visto[v]) and arista_viva and (signo[v] == signo[start]):
                    visto[v] = True; comp.append(v); pila.append(v)
        k = len(comp)
        hist[k] = hist.get(k, 0) + 1
    return hist

# ----------------------------------------------------------------------------
# 6. CORRIDA y NULL
# ----------------------------------------------------------------------------
def corrida(N, eps, H, pasos, seed, null=False):
    rng = np.random.default_rng(seed)
    phi, x = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    phi, activo, c0 = evolucionar(phi, activo, H, pasos, rng, null=null)
    P = persistencia(phi, c0)
    cuantos = detectar_cuantizacion(phi, activo)
    # Reporte físico (NO dinámica): la temperatura se LEE del estado — fracción de
    # acoplamientos que la expansión cortó (frac=0 sin expandir -> T_SING; frac=1 -> T_FIN).
    frac_exp = 1.0 - float(activo.mean())
    T_fin = temperatura_fisica(frac_exp)
    return {"P": P, "cuantos": cuantos, "frac_exp": frac_exp, "T_fin_K": T_fin}

# ----------------------------------------------------------------------------
# 7. BARRIDO — único eje de entrada: eps y H (H se reexpresa como r=H/D)
# ----------------------------------------------------------------------------
def barrido(N, eps_list, H_list, pasos, semillas):
    filas = []
    for eps in eps_list:
        D = np.mean([medir_D(N, eps, s) for s in range(semillas)])   # difusividad emergente
        for H in H_list:
            r = H / D if D > 0 else np.inf
            Preal, Pnull, Tfin = [], [], []
            hist_real = {}
            for s in range(semillas):
                rr = corrida(N, eps, H, pasos, seed=1000 + s, null=False)
                nn = corrida(N, eps, H, pasos, seed=1000 + s, null=True)
                Preal.append(rr["P"]); Pnull.append(nn["P"]); Tfin.append(rr["T_fin_K"])
                for k, c in rr["cuantos"].items():
                    hist_real[k] = hist_real.get(k, 0) + c
            Preal = np.array(Preal); Pnull = np.array(Pnull)
            sd = np.sqrt((Preal.var() + Pnull.var()) / 2.0)
            sd = max(sd, 1.0 / max(len(Preal), 1))       # piso sensato (no 1e-9)
            z = (Preal.mean() - Pnull.mean()) / sd
            filas.append({
                "eps": eps, "H": H, "D": round(float(D), 4), "r": round(float(r), 3),
                "P_real": round(float(Preal.mean()), 4),
                "P_null": round(float(Pnull.mean()), 4),
                "z": round(float(z), 2),
                "cuantos_k": {int(k): int(v) for k, v in sorted(hist_real.items())},
                # --- reporte físico (mapeo, NO dinámica) ---
                "T_fin_K": float(np.mean(Tfin)),          # temperatura final leída del estado
                "T_ini_K": T_SING,                        # singularidad (envuelve 1e15 real)
                "t_span_s": [t_INI, reloj_fisico(pasos, pasos)],  # ventana temporal (envuelve 1e-12..1e-6)
            })
    return filas

# ----------------------------------------------------------------------------
# 8. Configuración de corrida (la ejecuta el equipo tras revisar; SIN smoke previo)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "produccion"
    if modo == "produccion":
        N = 800; pasos = 120; semillas = 12
        eps_list = [0.0, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.5, 1.0]
        H_list = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
    else:  # 'chico' — solo si el equipo quiere una pasada rápida de verificación
        N = 200; pasos = 40; semillas = 4
        eps_list = [0.0, 1e-3, 1e-1, 0.5]
        H_list = [0.0, 0.1, 0.5, 0.9]
    filas = barrido(N, eps_list, H_list, pasos, semillas)
    print(json.dumps({"modo": modo, "N": N, "pasos": pasos, "semillas": semillas,
                      "filas": filas}, ensure_ascii=False))
