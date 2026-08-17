# INSTRUCCIÓN CS075 v4 — LOS DOS CONTROLES. ¿El borde es del modelo o es mío?

**Encarga:** Alexis López Tapia (director) · **Diseña:** Claude Science · **Ejecuta:** CC
**Fecha:** 30-jul-2026 · **Sucede a:** v3, cerrada por CC (barrido de 160 configuraciones)

---

## 1. Por qué estos dos controles y no las otras dos preguntas

De las tres preguntas que quedaron abiertas, dos refinan un número del que no depende ninguna
conclusión: afinar la malla dice si el borde está en 1,05 o en 1,10, y extender más allá de 6,0
busca un techo que no cambia ninguna lectura. Son baratas pero no mueven lo que el experimento
significa.

**Lo que sí puede tumbar el resultado es que el borde no sea del modelo.** Hay dos maneras de que
no lo sea, y cada control ataca una:

- **N1 — que el borde sea de mis umbrales.** Todo el resultado descansa sobre `hay_atomos` y
  `hay_red`, y el propio informe de CC (ADENDA 1) los marca como *"declarados, no anclados"*.
  `FACTOR_ATOMOS = 2` lleva el comentario "DECLARADO, no anclado" en el código, línea 114. Si el
  borde se desplaza al mover esos valores, es un artefacto de dónde puse yo la raya.
- **N2 — que la cascada no venga de la física del inventario**, sino de que la temperatura baja
  monótonamente y los hitos se cumplen en secuencia fija sin importar quién dependa de qué.

El segundo es el que puede doler, y por eso va.

---

## 2. N1 — Sensibilidad de umbrales

**Qué se varía**, sólo estas dos constantes de `cs075_23_sobre_fisica.py` (l.113-114):

| constante | valor actual | valores a probar |
|---|---|---|
| `MIN_PERSISTENCIA` | 5 | 3, 5, 10 |
| `FACTOR_ATOMOS` | 2 | 1, 2, 4 |

9 combinaciones. Por cada una, correr **sólo el tramo del borde** — las 5 amplitudes de la grilla
v3 entre 0,621 y 2,190 (0,621 / 0,799 / 1,028 / 1,323 / 1,702 / 2,190, que son 6) × 8 semillas.
Son 9 × 6 × 8 = **432 configuraciones**. Al costo medido en v3 (29,1 s/config bajo carga, 8
procesos) son unos 26 minutos.

**Cómo pasarlas sin tocar el motor:** parametrizar por variable de entorno o por argumento del
script del control, leyendo el default del módulo si no se especifica. **No editar
`cs075_23_sobre_fisica.py`.** Si eso no se puede hacer limpio, **pará y reportá** antes de
modificar el motor.

**Qué se mide:** la amplitud donde la fracción de semillas en 23/23 cruza 0,5, para cada una de
las 9 combinaciones.

**Lectura pre-inscrita:**

- **El borde se mantiene dentro del mismo escalón de grilla** (entre 1,028 y 1,323) en la mayoría
  de las 9 → el borde es del modelo. Los umbrales fijan detalles, no el resultado.
- **El borde se desplaza más de un escalón** al mover los umbrales → **el borde es mío, no del
  modelo.** Hay que decirlo así, sin suavizar: el resultado de v3 pasaría a ser un artefacto de
  calibración y habría que anclar los hitos antes de afirmar nada.
- **El borde desaparece** (nunca se llega a 23/23, o se llega siempre) en varias combinaciones →
  los umbrales están en un punto muy particular y eso también hay que reportarlo.

---

## 3. N2 — Puertas permutadas (el control que ataca la tesis)

**La afirmación bajo prueba:** que los 23 agentes despierten en orden refleja la cadena causal del
inventario — que la gravedad necesita sobredensidad, que el EM necesita núcleos, que la poda
necesita red.

**La alternativa que hay que descartar:** que la cascada ordenada sea inevitable con *cualquier*
asignación de precondiciones, porque la temperatura baja monótonamente, los hitos se van
cumpliendo en secuencia, y entonces cualquier reparto produce un despertar escalonado.

**Cómo se construye.** La estructura real, verificada en el código:

- 6 agentes con `requiere = ()`: `23_campo`, `22_qcd`, `9_expansion`, `10_enfriamiento`,
  `M1_semilla`, `M3_fase_cuantica`
- 17 agentes con **exactamente un hito** cada uno, repartidos así: 4 en `T_bajo_electrodebil`,
  5 en `T_bajo_confinamiento`, 2 en `hay_sobredensidad`, 1 en `hay_nucleos`, 3 en `hay_atomos`,
  2 en `hay_red`

**El NULL permuta la asignación de esos 17 hitos entre esos 17 agentes**, conservando el
histograma exacto (4/5/2/1/3/2). Eso es clave: el NULL tiene la **misma dificultad** que el real
— la misma cantidad de agentes esperando cada hito. Lo único que se destruye es *qué* agente
espera *cuál*.

Los 6 sin precondición se quedan sin precondición. No se toca nada más.

**Cuántas permutaciones:** 20 permutaciones distintas, cada una con las 8 semillas, sobre 3
amplitudes: una por debajo del borde (0,799), la del borde (1,028) y una por encima (1,702). Son
20 × 8 × 3 = **480 configuraciones**, unos 29 minutos. Más el brazo REAL como referencia en esas
mismas 3 amplitudes (ya está en el JSON de v3 — se reusa, no se recorre).

**Las permutaciones se generan con semillas declaradas en el código**, no al azar en tiempo de
corrida.

**Qué se mide**, en las 3 amplitudes:

1. `n_despiertos` medio del brazo permutado contra el REAL.
2. **Cuántas permutaciones alcanzan 23/23** en amp=1,702, donde el REAL da 8/8.
3. **El orden de despertar**: ¿los niveles siguen apareciendo en secuencia, o se mezclan?

**Lectura pre-inscrita:**

- **Las permutaciones NO llegan a 23/23** donde el real sí, o llegan mucho menos seguido → la
  asignación de precondiciones importa. La cascada es del inventario, no de la monotonía del
  enfriamiento. **Es el resultado que sostiene la tesis.**
- **Las permutaciones llegan a 23/23 igual que el real** → la cascada ordenada es inevitable y no
  dice nada sobre la física del inventario. **La tesis no queda refutada, pero sí queda sin
  evidencia**: habría que buscar otro observable que distinga una asignación de otra.
- **Resultado intermedio** (algunas permutaciones sí, otras no) → informativo: hay que mirar qué
  tienen en común las que fallan. Probablemente sean las que ponen un agente de estructura
  temprano o un agente térmico tarde.

**Ojo con un caso degenerado:** algunas permutaciones pueden ser idénticas al real en la práctica
si intercambian dos agentes que comparten el mismo hito. Filtrarlas o contarlas aparte, no
mezclarlas con las genuinamente distintas.

---

## 4. Qué entregar

- `cs075_N1_sensibilidad_umbrales.py` + su JSON
- `cs075_N2_puertas_permutadas.py` + su JSON
- **Una tabla de 9 filas** (N1): `MIN_PERSISTENCIA`, `FACTOR_ATOMOS`, amplitud del borde
- **Una tabla de 20 filas** (N2): permutación, `n_despiertos` en cada una de las 3 amplitudes,
  cuántas semillas en 23/23
- Los dos veredictos según las lecturas pre-inscritas de §2 y §3

---

## 5. Reglas

- **No se toca** `cs075_base_fisica.py`, `cs075_arquitectura_agentes.py` ni
  `cs075_23_sobre_fisica.py`. Si un control no se puede montar sin editarlos, **pará y reportá**.
- **Ninguna constante nueva.** Los valores de N1 son múltiplos y divisores de los actuales, no
  números nuevos.
- **No ajustes nada para que el control pase.** Si N2 dice que la cascada es inevitable, ése es el
  resultado. Ya reportaste una vez un número que empeoraba tu propio informe anterior; esto es lo
  mismo.
- **El experimento no se cierra** sin autorización explícita del director.

---

## 6. Lo que estos controles no arreglan

- El NULL de la **física** sigue sin correrse. N2 prueba que la asignación de precondiciones
  importa; no prueba que los depósitos de cada agente sean físicamente correctos.
- **Malla fija en 16³** en los dos controles.
- Si N1 muestra que el borde depende de los umbrales, el arreglo **no es** elegir mejores
  umbrales: es anclarlos en algo del proyecto, o declarar que no se pueden anclar y que el borde
  es una propiedad de la calibración.

---

*Verificá en disco, no de palabra: antes de escribir "verifiqué X", el valor de X tiene que estar
impreso en la salida que estás mirando.*
