# INSTRUCCIÓN CS075 v3 — EL BARRIDO. ¿Dónde nace el universo completo?

**Encarga:** Alexis López Tapia (director) · **Diseña:** Claude Science · **Ejecuta:** CC
**Fecha:** 30-jul-2026 · **Sucede a:** v2, cerrada por CC (los tres pasos hechos y verificados)

---

## 1. La pregunta

El smoke encontró algo que ninguna corrida anterior había visto: con `amp_asimetria=2,0` **los
23 agentes despertaron**, incluido `hay_red` con 15 regiones conexas. Con 0,5, sólo 17. Con 0,1 y
0,01, sólo 15.

**Entre 0,5 y 2,0 el universo pasa de incompleto a completo.** El barrido contesta tres cosas:

1. **¿Dónde está el borde?** El smoke tiene un punto cada factor 4 — demasiado grueso para
   localizarlo.
2. **¿Es un borde o es una banda?** ¿El inventario completo se sostiene por encima de 2,0, o hay
   un techo donde el colapso vuelve a apagar agentes?
3. **¿Es real o es la semilla 12345?** Todo el experimento corrió con **una sola semilla**. Éste
   es el hueco más serio que queda abierto.

---

## 2. Presupuesto: minutos, y por eso el barrido puede ser generoso

Calculado sobre lo que CC **midió** (ADENDA 4: 1,31 ms/paso con pocos agentes, 4,28 ms/paso con
los 23; las 4 configuraciones en 46,9 s ⟹ 11,72 s/configuración de promedio real):

| barrido | configuraciones | en serie | con 8 núcleos |
|---|---|---|---|
| 20 amplitudes × 5 semillas | 100 | 19,5 min | 2,4 min |
| **20 amplitudes × 8 semillas** | **160** | **31,3 min** | **3,9 min** |
| 30 amplitudes × 10 semillas | 300 | 58,6 min | 7,3 min |

**Se corre el de 160.** No hay razón para escatimar a este costo, y 8 semillas dan estadística
suficiente para separar señal de semilla.

**Margen de pasos:** el confinamiento cruza en el paso 36; `T_TOTAL=5,0` son 5.000 pasos, 139×
margen. Se mantiene tal cual — CC ya lo validó y bajarlo ahorraría segundos a cambio de arriesgar
los hitos altos, que necesitan `MIN_PERSISTENCIA=5` y el factor de átomos acumulándose después
del cruce.

---

## 3. La grilla de amplitud, anclada en lo ya medido

20 valores log-espaciados de **0,05 a 6,0** (razón 1,287 entre vecinos). Los bordes no son
arbitrarios: cubren los tres regímenes que cs074A ya midió y extienden un poco cada lado.

| tramo | puntos | régimen de cs074A | qué se espera |
|---|---|---|---|
| 0,050 – 0,483 | 10 | meseta (ε<0,5) | inventario incompleto (el smoke dio 15-17/23) |
| 0,621 – 2,190 | 6 | fragmentación (0,9–2,3) | **aquí está la transición** |
| 2,817 – 3,625 | 2 | entre fragmentación y colapso | ¿se sostiene el 23/23? |
| 4,664 – 6,000 | 2 | colapso (>3,8) | ¿vuelve a caer? |

**Un dato que el barrido tiene que resolver, no asumir:** la transición del smoke (entre 0,5 y
2,0) cae **dentro** de la banda de fragmentación que cs074A midió de forma independiente
(0,9–2,3). Puede ser que el inventario se complete justamente cuando la estructura empieza a
fragmentarse — o puede ser coincidencia de dos escalas distintas. Con 6 puntos en esa banda y 8
semillas se distingue.

**Semillas:** 8 valores fijos y declarados en el código — `12345` (la de todo el experimento
anterior, para poder comparar) más 7 más. No generadas al azar en tiempo de corrida: escritas.

---

## 4. Los observables

Por cada una de las 160 configuraciones, del registro que el motor ya produce:

1. **`n_despiertos`** (0–23) — el observable principal.
2. **`nivel_maximo_alcanzado`** (0–6).
3. **Por agente:** `paso_despertar`, o el hito que le faltó.
4. **Los 7 hitos:** en qué paso se cumplió cada uno, o `None`.
5. **Conteos de estructura:** celdas sobredensas, núcleos, átomos, regiones conexas.
6. **Estado físico final:** T, ρ, X, S, gradientes, factor de escala.
7. **Costo:** ms/paso y segundos por configuración.

**Criterio de "universo completo": `n_despiertos == 23`.** Es binario y no necesita umbral
estadístico inventado — un agente despertó o no despertó.

---

## 5. Los tres análisis, sobre los mismos datos

### 5.1 El borde

Para cada amplitud, la **fracción de las 8 semillas que alcanzan 23/23**. Eso da una curva de 0 a
1 sobre la grilla. Reportar dónde cruza 0,5 y qué tan abrupta es la subida — si pasa de 0 a 1 en
un solo escalón de la grilla, es un borde; si tarda cuatro, es una rampa.

### 5.2 La banda

¿Hay un techo? Si a 4,664 o 6,000 el conteo vuelve a bajar, el inventario completo vive en una
**banda** y no en una semirrecta. Reportar el conteo por amplitud en toda la grilla, sin recortar.

### 5.3 La semilla

Por cada amplitud, la **dispersión entre las 8 semillas** de `n_despiertos`. Dos lecturas
posibles y ambas informativas:

- **dispersión baja** en casi toda la grilla → el resultado es del modelo, no de la semilla. Es
  lo que hace falta para que el 23/23 signifique algo.
- **dispersión alta** cerca del borde y baja lejos → normal en una transición, y ubica el borde
  con más precisión que la curva media.

Si la dispersión es alta **en todas partes**, el 23/23 del smoke fue suerte de la semilla 12345 y
**hay que reportarlo como tal.** Ese resultado sería negativo y hay que decirlo con la misma
claridad que el positivo.

---

## 6. Qué entregar

- `cs075_barrido_v3.py` — el script del barrido
- `cs075_resultado_barrido_v3.json` — registro completo de las 160 configuraciones
- Una tabla de 20 filas: amplitud, media de `n_despiertos`, mínimo, máximo, cuántas de las 8
  semillas dieron 23/23, nivel máximo alcanzado
- El costo real medido

Con eso alcanza para adjudicar. **Nada de barrer más fino sin autorización**, aunque el borde
quede entre dos puntos de la grilla — reportarlo y esperar.

---

## 7. Reglas que siguen vigentes

- **No se toca** `cs075_base_fisica.py`, `cs075_arquitectura_agentes.py` ni
  `cs075_23_sobre_fisica.py`. El barrido los importa y llama a `construir_23()`.
- **Ninguna constante nueva.** Si hace falta un valor, primero `grep` en `cs072_modulos/` y
  `cs072_motor_23.py` — el proyecto suele tenerlo ya fijado. Los umbrales `T_CONF=0,6` /
  `T_EW=0,9` de `cs072_motor_23.py` l.42-43 son el ejemplo: estaban escritos, y derivarlos por
  otro camino costó un factor 580.000 en el conteo de pasos.
- **No hay NULL en esta etapa.** Sigue siendo arquitectura.
- **Un desacuerdo es un dato:** si algo de esta instrucción choca con lo que el código hace, el
  código gana y CC reporta el choque.
- **El experimento no se cierra** sin autorización explícita del director.

---

## 8. Lo que este barrido NO contesta

Queda fuera de alcance, y conviene tenerlo escrito para no confundir el resultado con más de lo
que es:

- **No prueba que la estructura sea física** — prueba que el inventario se completa. Que los 23
  agentes despierten significa que cada uno encontró sus condiciones, no que lo que se formó sea
  materia en algún sentido fuerte.
- **No hay control NULL.** El director lo excluyó de esta etapa. Sin él, "los 23 despiertan" es un
  hecho de la arquitectura, no una afirmación sobre el universo.
- **Malla fija en 16³.** Si el borde depende del tamaño de malla, este barrido no lo vería. Una
  malla 24³ cuesta 3,4× y 32³ cuesta 8× — sigue siendo barato, pero es otra corrida y otra
  autorización.

---

*Verificá en disco, no de palabra: antes de escribir "verifiqué X", el valor de X tiene que estar
impreso en la salida que estás mirando.*
