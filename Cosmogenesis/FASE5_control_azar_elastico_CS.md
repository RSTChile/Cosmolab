# El control de azar que faltaba en la rama elástica — cerrando el arco F5-C2-C → C5

**Fecha:** 10-ago-2026 · Ejecuta: CC (Claude) · Quinta y última tarea de la línea que abrió
`FASE5_presupuesto_emergente_CS.md` (F5-C2-C): `FASE5_presupuesto_soporte_local_CS.md` (F5-C2-C2),
`FASE5_mecanismo_aislado_CS.md` (F5-C2-C3), `FASE5_matriz_2x2_completa_CS.md` (F5-C2-C4) y esta
(F5-C2-C5). Alexis pidió explícitamente "vamos con la quinta celda" tras ver la matriz 2×2 cerrada en
F5-C2-C4.

Ningún script congelado ni los 4 archivos de las 4 tareas anteriores de esta línea fueron modificados
(`cs090_fase5_generador.py`, `cs090_fase5_motor.py`, `cs090_fase5_clasificador.py`,
`cs090_fase5_presupuesto_emergente.py`, `cs090_fase5_presupuesto_soporte.py`,
`cs090_fase5_mecanismo_aislado.py`, `cs090_fase5_presupuesto_variable.py` — verificable con `git diff`).
El único archivo de código nuevo es `cs090_fase5_control_azar_elastico.py`, que importa y reusa
`MA._cupo_variable` (F5-C2-C3), `PS._costos_relacionales_soporte` (F5-C2-C2) y
`PV.correr_regla_coarse_presupuesto_variable` (F5-C2-C4) tal cual, sin tocar una línea de ninguno. No se
corrió Phantom. No se hicieron commits de git.

## 0. La pregunta, en simple, cerrando la analogía del cupo de amigos

F5-C2-C4 dejó las 4 celdas de la matriz 2×2 medidas (mecanismo estricto/elástico × cupo uniforme/variable)
y, en su última nota honesta, señaló una pieza que faltaba: F5-C2-C3 ya había probado, DENTRO del
mecanismo estricto+variable, qué pasa si en vez de soltar la arista de "menos soporte local" se suelta
una AL AZAR — el resultado cayó de 35.0% a 5.0%, mostrando que el criterio (no sólo la rigidez del
número) importaba mucho ahí. Pero ese mismo experimento nunca se había hecho en la rama elástica+variable
(`C2-presupuesto-variable`, 10.0%/15.0%). La pregunta de esta tarea: **dentro del presupuesto elástico con
cupo variable, ¿importa CÓMO se elige qué arista comprar con el presupuesto, o el mecanismo elástico ya
"aplana" tanto el resultado que da igual si se elige por costo o al azar?**

En la analogía del cupo social: las 4 tareas anteriores ya probaron "cada quien gasta su presupuesto
social donde más rinde, y ese presupuesto ya varía según con cuánta gente nació conociendo" (10.0-15.0%).
Esta tarea prueba la variante: **"cada quien tiene el MISMO presupuesto de entrada (varía igual persona a
persona), pero ya no elige gastarlo en las amistades que más le convienen — lo gasta en el orden en que se
le van ocurriendo, sin mirar cuánto rinde cada una, hasta que se le acaba la plata."**

## 1. Cómo se construyó el control de azar

Archivo nuevo: **`cs090_fase5_control_azar_elastico.py`**. Se reusan, sin modificar una línea:

1. **`cs090_fase5_mecanismo_aislado._cupo_variable(grado_inicial, kcap_base)`** — la MISMA fórmula y el
   MISMO `budget_por_nodo` (`B_i`) que ya usó `C2-presupuesto-variable` en F5-C2-C4. La magnitud del
   presupuesto de entrada, nodo por nodo, es idéntica entre los dos brazos elásticos-variables — lo único
   que cambia entre ellos es cómo se decide en qué se gasta.
2. **`cs090_fase5_presupuesto_soporte._costos_relacionales_soporte(...)`** — el `c_ij` de 4 componentes
   (historia + holonomía + compatibilidad + soporte local) tal cual. Importante: el control de azar SIGUE
   necesitando el costo real de cada arista, porque el criterio de parada del knapsack sigue siendo "sumar
   costos hasta agotar `B_i`" — lo que se quita no es el costo, es que el costo decida el ORDEN en que se
   consideran las aristas.

**El único código genuinamente nuevo**, `_enforce_relacional_variable_azar`, es una copia de
`PV._enforce_relacional_variable` (F5-C2-C4) con un solo cambio real: en vez de ordenar las aristas vivas
de cada nodo por costo ascendente antes de acumular, se recorren en un **orden aleatorio**
(`rng.permutation`), acumulando el costo real de cada una y deteniéndose cuando el acumulado excedería
`B_i`. La magnitud del presupuesto (`B_i`, en las mismas unidades de costo que el brazo con criterio) es
idéntica; lo que cambia es exclusivamente qué aristas le tocó "aparecer primero" en el orden aleatorio
antes de que se agotara la plata de ese nodo.

**Honestidad sobre una diferencia real entre este control y el de la rama estricta (F5-C2-C3):** en
F5-C2-C3, `C2-random` mantenía el mismo NÚMERO EXACTO de aristas por nodo que `C2-hibrido` (`kcap_i`, un
conteo fijo) — sólo cambiaba cuáles se soltaban. Acá, como el mecanismo es un presupuesto en unidades de
costo (no un conteo), el número final de aristas por nodo del brazo azar puede diferir un poco del brazo
con criterio incluso con el mismo `B_i` de entrada — una arista cara que aparece temprano en el orden
aleatorio consume más presupuesto que si se hubiera dejado para el final, así que el brazo azar típicamente
termina con un poco MENOS aristas en promedio (grado medio 4.29 vs 4.51, ver §2) que el brazo con criterio.
Esto es intencional y es la analogía correcta de un presupuesto real (a diferencia de un conteo exacto): la
CANTIDAD de dinero es igual, lo que varía es cuánto rinde según el orden en que se gasta. No es un defecto
del control, es una consecuencia honesta de que el mecanismo elástico y el estricto no son directamente
equiparables en "cuántas aristas sobreviven", sólo en "cuánto presupuesto de entrada tenía cada nodo".

## 2. Los 5 brazos, mismo lote de reglas

1. **C2-hard** — `MOT._enforce_kcap`, sin cambios (control, ESTRICTO+UNIFORME+criterio-soporte).
2. **C2-hibrido** — reusa `MA.correr_regla_coarse_hibrido(p, modo="soporte")` tal cual, recalculado
   fresco (ESTRICTO+VARIABLE+criterio-soporte).
3. **C2-presupuesto-variable** — reusa `PV.correr_regla_coarse_presupuesto_variable(p)` tal cual,
   recalculado fresco (ELÁSTICO+VARIABLE+criterio-costo, F5-C2-C4).
4. **C2-presupuesto-variable-azar** — NUEVO, la celda que faltaba (ELÁSTICO+VARIABLE+SIN criterio).
5. **C0** — sin límite de escala, sin cambios.

**Control clave:** mismo `seed_base` (`SEED_BASE=90210` para piloto, `SEED_BASE+1` para completo) que las
4 tareas anteriores — las 20 reglas admitidas (A2-B0-C2, filtro P1-P5 real) son **idénticas** en
`K,J,noise,meandeg,kcap,seed` a las de esas 4 corridas. Confirmado en la corrida: `C2-hard` (45.0%) y
`C2-hibrido` (35.0%) dieron, hasta el decimal, los mismos números que en F5-C2-C3/C4 — el determinismo del
motor y el `seed_base` compartido garantizan comparabilidad directa entre las 5 tareas de esta línea.

## 3. Corrida

Piloto de 3 semillas × 5 brazos: **0.4 min**, sin fallos. Corrida completa de 20 semillas × 5 brazos = 100
reglas×brazo, N=2000, coarse-graining b=1/2/4/8/16: **2.7 minutos**, muy por debajo del presupuesto de
50-60 min. Sin "SALVAGUARDA DE TIEMPO" ni fallos de motor. Se usó `./venv/bin/python3` (venv del proyecto)
para que `scipy` esté disponible.

Salidas:
- `cs090_fase5_control_azar_elastico_resultados.csv` — 500 filas (20 reglas × 5 brazos × 5 escalas), dato crudo.
- `cs090_fase5_control_azar_elastico_resumen.csv` — 100 filas (una por regla×brazo), clase + observables + parámetros.
- `cs090_fase5_control_azar_elastico_piloto_raw.csv` / `_piloto_resumen.csv` — piloto de 3 semillas, conservado.

## 4. Resultado — fracción de Clase III y observables continuos

| brazo | n | I | II | III | IV | otro | **%Clase III** | %III+IV | grado medio (b=1) | n_aristas medio | diám medio | pendiente media | pendiente mediana |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **C2-hard**                      | 20 | 9  | 2 | **9** | 0 | 0 | **45.0%** | 45.0% | 3.62 | 3623.9 | 13.55 | 0.707 | 0.652 |
| **C2-hibrido**                    | 20 | 7  | 5 | **7** | 0 | 1 | **35.0%** | 35.0% | 4.31 | 4314.1 | 10.20 | 0.550 | 0.529 |
| **C2-presupuesto-variable**       | 20 | 12 | 5 | **2** | 1 | 0 | **10.0%** | 15.0% | 4.51 | 4513.3 | 11.10 | 0.555 | 0.526 |
| **C2-presupuesto-variable-azar**  | 20 | 14 | 3 | **2** | 0 | 1 | **10.0%** | 10.0% | 4.29 | 4289.6 | 11.00 | 0.537 | 0.538 |
| **C0**                            | 20 | 13 | 7 | **0** | 0 | 0 | **0.0%**  | 0.0%  | 6.22 | 6222.0 | 8.00  | 0.371 | 0.358 |

(La fila "otro" de `C2-hibrido` y de `C2-presupuesto-variable-azar` es, en los dos casos, la misma regla
`A2-B0-C2-r16` que ya era outlier en F5-C2-C3/C4 — "intermedio, sin clase clara". En `C2-hibrido` dio
pendiente=−1.137; en `C2-presupuesto-variable-azar` dio pendiente=0.229 — muy distinto entre sí, ver §6.)

**Comparaciones pareadas** (misma regla, mismo K/J/noise/meandeg/kcap/seed en los 5 brazos, n=20, sobre la
pendiente continua):

| comparación | dirección | media de la diferencia | mediana |
|---|---|---|---|
| hard vs hibrido | hard gana en **11/20** (casi moneda) | +0.157 | +0.023 |
| hard vs presupuesto-variable | hard gana en **18/20** | +0.153 | +0.156 |
| hard vs presupuesto-variable-azar | hard gana en **18/20** | +0.170 | +0.111 |
| presupuesto-variable vs presupuesto-variable-azar | **variable-azar gana en 12/20** | +0.017 (variable gana en media) | **−0.022** (azar gana en mediana) |
| hibrido vs presupuesto-variable-azar | hibrido gana en **11/20** (casi moneda) | +0.013 | +0.014 |
| presupuesto-variable-azar vs C0 | variable-azar gana en **17/20** | +0.166 | +0.140 |

**La comparación central de esta tarea** — `C2-presupuesto-variable` vs `C2-presupuesto-variable-azar` —
es prácticamente un empate: 8 reglas favorecen al brazo con criterio, 12 al azar, la diferencia de medias
(+0.017) y de medianas (−0.022) son ambas indistinguibles de cero en la escala en que las otras
comparaciones (p.ej. hard vs cualquier brazo elástico, ~+0.15-0.17) se mueven. La fracción de Clase III es
literalmente idéntica (10.0% en ambos); sólo cambia si se cuenta con o sin la única fila Clase IV
(15.0% con criterio vs 10.0% sin ella, porque el brazo azar no tuvo ninguna fila IV esta vez).

## 5. El cuadro completo — mecanismo × uniformidad × criterio, los 5 números de esta línea juntos

```
                              CON CRITERIO (soporte/costo)         SIN CRITERIO (azar)
  ESTRICTO + UNIFORME        C2-hard              45.0%             (no medido — no hay versión
    (mismo número, todos)    pendiente 0.707        azar de C2-hard en esta línea)

  ESTRICTO + VARIABLE        C2-hibrido            35.0%             C2-random         5.0%   [F5-C2-C3]
    (número por nodo)        pendiente 0.550        pendiente 0.494

  ELÁSTICO + UNIFORME        C2-budget-soporte     10.0% (15% c/IV)  (no medido — pendiente si se
    (mismo presupuesto)      pendiente 0.549        quisiera completar la 6ª celda)

  ELÁSTICO + VARIABLE        C2-presupuesto-var.   10.0% (15% c/IV)  C2-presupuesto-var-azar 10.0%
    (presupuesto por nodo)   pendiente 0.555        pendiente 0.537                     [NUEVO, F5-C2-C5]
```

**Los 4 pares comparables directamente (mismo mecanismo/uniformidad, con vs. sin criterio):**

| par | con criterio | sin criterio (azar) | caída |
|---|---|---|---|
| ESTRICTO + VARIABLE (F5-C2-C3) | 35.0% (hibrido) | 5.0% (random) | **−30 pp** |
| ELÁSTICO + VARIABLE (F5-C2-C5, esta tarea) | 10.0-15.0% (presupuesto-variable) | 10.0% (presupuesto-variable-azar) | **~0 pp** |

**Respuesta a la pregunta que abrió esta tarea:** dentro de la rama elástica+variable, el criterio (costo
de 4 componentes vs. orden aleatorio) **no movió la fracción de Clase III** — quedó pegada a 10.0% en
ambos casos, con una comparación pareada continua que es un empate estadístico (12-8, diferencias de
mediana/media cercanas a cero). Esto es el resultado opuesto al de la rama estricta, donde el mismo tipo de
control (mismo cupo, sin criterio) hundió el número 30 puntos porcentuales.

## 6. Lectura en simple, cerrando la analogía del cupo de amigos

Toda la línea de 5 tareas, junta, cuenta esta historia sobre "qué hace que una red social termine
extendida en vez de compacta":

1. Empezar con **"todos tienen exactamente 5 amigos, sin excepción, y se queda con los de más soporte
   mutuo"** da la red más extendida (45%).
2. Cambiar **sólo la uniformidad** del número (cada quien con su propia cifra, pero seguir eligiendo por
   soporte, sin excepción) apenas mueve la aguja: 45%→35%.
3. Cambiar **sólo el criterio** dentro de ese mismo cupo exacto y variable (dejar de elegir por soporte,
   elegir al azar) hunde mucho más: 35%→5%. **El criterio, dentro de un corte rígido, importa muchísimo.**
4. Cambiar **el mecanismo** (de corte exacto a "presupuesto que se gasta donde más rinde") ya hunde el
   número por sí solo, con o sin variar el cupo de entrada: 45%→10-15% (uniforme) o 35%→10-15% (variable).
5. Y la pieza que cerraba esta tarea: una vez que el mecanismo ya es un presupuesto elástico con cupo
   variable, **quitarle también el criterio (gastar al azar en vez de por costo) ya casi no cambia nada
   más**: 10-15%→10%. El presupuesto elástico "se come" casi todo el efecto del criterio — el efecto que el
   criterio tenía en la rama estricta (30 puntos) prácticamente desaparece una vez que el corte ya no es
   rígido.

**La imagen que arma el conjunto:** lo que más separa la red extensa (Clase III) de la compacta no es "con
cuánto cuidado se elige la amistad", sino **si en algún punto existe un límite duro y sin excepciones que
corta de un tajo, sin dejar comprar ni negociar**. El criterio de selección (soporte vs. costo vs. azar)
sólo importa MUCHO cuando ese límite duro ya está — ahí, cómo se decide a quién soltar es casi tan
determinante como el límite mismo (30 de los 40 puntos que separan hard de random se explican por el
criterio, no por la rigidez sola). Pero en cuanto el límite deja de ser duro (presupuesto elástico), el
criterio deja de importar casi del todo — ya sea que gaste bien o gaste al azar, el sistema converge al
mismo resultado bajo, ~10%.

## 7. Lecturas alternativas honestas (no se fuerza ninguna)

- **La comparación entre ramas no es perfectamente simétrica.** En la rama estricta, `C2-random` mantiene
  el mismo NÚMERO EXACTO de aristas por nodo que `C2-hibrido` (conteo fijo, `kcap_i`); en la rama elástica,
  `C2-presupuesto-variable-azar` mantiene el mismo PRESUPUESTO de entrada (`B_i`, en unidades de costo) que
  `C2-presupuesto-variable`, pero el número final de aristas por nodo puede diferir un poco (grado medio
  4.29 vs 4.51, una diferencia de ~5%, ver §1). Esto no invalida la lectura de §6 — la magnitud del cambio
  en grado medio entre los dos brazos elásticos (4.29 vs 4.51) es mucho más chica que la magnitud del
  cambio entre los dos brazos estrictos (4.67 vs 4.31, F5-C2-C3) — pero es una asimetría real del diseño
  que vale la pena tener presente si se quisiera afinar el control aún más.
- **Dos celdas del cuadro de §5 siguen sin medir**: "ESTRICTO+UNIFORME+azar" (¿qué pasaría si `C2-hard`
  soltara aristas al azar en vez de por soporte, con el mismo `kcap` fijo para todos?) y
  "ELÁSTICO+UNIFORME+azar" (equivalente para `C2-budget-soporte`). Ninguna de las 5 tareas de esta línea
  las corrió — quedan como pistas abiertas si se quisiera completar el cuadro 4×2 entero, no sólo las 4
  celdas con criterio + las 2 celdas variable-con-vs-sin-criterio que sí se midieron.
- **La misma regla `A2-B0-C2-r16` vuelve a ser un caso especial**, la quinta vez consecutiva en esta línea:
  pendiente negativa en `C2-hibrido` (−1.137), Clase IV en `C2-presupuesto-variable` (0.878), e "intermedio
  sin clase clara" tanto en `C2-hibrido` (−1.137) como ahora en `C2-presupuesto-variable-azar` (0.229) —
  pero estos dos últimos valores de pendiente son muy distintos entre sí (−1.137 vs 0.229) pese a ser la
  misma regla base, lo que sugiere que esta regla en particular es sensible al mecanismo/criterio de un
  modo que el resto del lote no lo es. Sigue sin investigarse a fondo — documentado como caso aislado,
  no una tendencia, en las 5 tareas de esta línea.
- **La fracción de Clase III (10.0% en ambos brazos elásticos) es más estable que la pendiente pareada
  continua** (empate 12-8, medias/medianas cerca de cero) — ambas lecturas apuntan en la misma dirección
  acá (a diferencia de F5-C2-C4, donde hibrido vs presupuesto-variable sí tenía una brecha notable entre
  fracción de clase y pendiente pareada). Esta vez las dos métricas cuentan la misma historia sin matices,
  lo cual hace la lectura de §6 más sólida que otras de esta línea.
- **El `%III+IV` difiere ligeramente entre los dos brazos elásticos-variables (15.0% vs 10.0%)** sólo
  porque `C2-presupuesto-variable` tuvo 1 fila Clase IV (la misma `r16`) y `C2-presupuesto-variable-azar`
  no tuvo ninguna (esa misma regla cayó en "intermedio" en vez de IV). Si se cuenta sólo Clase III pura,
  el empate es exacto (10.0%=10.0%); si se cuenta III+IV, hay una diferencia de 5pp que depende enteramente
  de cómo cayó una única regla ya identificada como atípica.

## 8. Archivos de esta tarea

- `cs090_fase5_control_azar_elastico.py` — script nuevo (único archivo de código; no toca ningún script
  congelado ni los 4 archivos de las 4 tareas anteriores, sólo los importa/reusa).
- `cs090_fase5_control_azar_elastico_resultados.csv` — 500 filas, dato crudo (20 reglas × 5 brazos × 5 escalas).
- `cs090_fase5_control_azar_elastico_resumen.csv` — 100 filas, una por regla×brazo (clase + observables + parámetros).
- `cs090_fase5_control_azar_elastico_piloto_raw.csv` / `_piloto_resumen.csv` — piloto de 3 semillas, conservado.
- Este informe.

Ningún script congelado ni las 4 tareas anteriores de esta línea fueron modificados. No se corrió Phantom.
No se hicieron commits de git. No se declara cierre ni veredicto sobre "qué combinación de factores explica
mejor la fuerza geométrica de C2-hard" — los números de las 5 tareas y las lecturas alternativas de §7
están arriba; la síntesis final, y qué hacer con ella, es de Alexis.
