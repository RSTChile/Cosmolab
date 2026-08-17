# NULL-5 (correspondencia nodo↔posición destruida) — Fase II CS073, escalón 6 de 6, ÚLTIMO de la jerarquía

**Encargo:** completar el último escalón de la jerarquía de 6 controles que blinda CS073. A diferencia de
todos los anteriores (que cambian topología final, orden de formación, etc.), NULL-5 conserva TANTO la
topología COMPLETA de REAL COMO el conjunto de posiciones finales de REAL (ambos intactos por separado) y
sólo pregunta: ¿importa que el nodo A (con sus vecinos causales específicos) haya terminado exactamente en
la posición X que le asignó la física real, o alcanza con que "algún nodo cualquiera" esté ahí? **No se
declara cierre ni veredicto sobre CS073 ni sobre la jerarquía completa — sólo se reportan números. La
lectura final es de Alexis.**

---

## Resultado de una línea

**NULL-5 colapsa trivial** — verificado tanto analíticamente (leyendo el pipeline) como empíricamente (en
el archivo de texto Y en el comportamiento real de Phantom): permutar qué nodo del grafo causal ocupa cuál
posición física no cambia NINGÚN valor físico de ninguna partícula en este pipeline. El archivo de
condición inicial resultante es, como conjunto de partículas, **idéntico a REAL**; las 2 corridas Phantom
de verificación reprodujeron **exactamente** el resultado de REAL (8 sumideros, masa total 2124.4, mismas
8 masas individuales, sin ninguna diferencia).

---

## Paso 0 — por qué NULL-5 es distinto de NULL-4 en operacionalizabilidad

NULL-4 (escalón anterior) SÍ resultó operacionalizable porque `layout_resortes(seed=...)` consume la
secuencia de números aleatorios de posición inicial EN EL ORDEN en que cada nodo aparece durante la
construcción de la malla de adyacencia — cambiar ese orden cambia qué número de la secuencia recibe cada
nodo, lo cual sí altera el resultado final (38% de la escala típica de coordenada, ya reportado en
`NULL4_resultado_CS.md`).

NULL-5 es un caso distinto: no se vuelve a correr `layout_resortes` — se parte del layout REAL YA
COMPUTADO (las posiciones finales, intactas) y sólo se pregunta si permutar qué nodo "es dueño" de cuál
posición cambia algo río abajo. Eso depende enteramente de si algún atributo físico que se escribe en el
archivo de condición inicial (IC) de Phantom **depende de la identidad del nodo** (sus vecinos causales)
en vez de depender sólo del **valor** de su posición final. Se verificó esto ANTES de construir ninguna
batería, con la misma disciplina que `null4_verificar_invarianza_orden.py` usó para NULL-4
(`null5_verificar_colapso.py`, nuevo, sólo lee/importa piezas congeladas):

| verificación | método | resultado |
|---|---|---|
| (1) Malla causal + `layout_resortes` reconstruidas ¿reproducen el archivo REAL en disco bit a bit? | reconstrucción con los mismos parámetros que toda la jerarquía (D=3,k=4,seed_ejes=2000,seed_layout=12345), comparación directa contra `bateria_n2000/ic_real/cosmogenesis_ic.txt` | **diff máxima = 0.0** — bit-idéntico |
| (2) ¿La velocidad depende de algo más que la posición final? | recomputar `campo_velocidad_turbulento` (Mach=3, seed=42, MISMO que toda la jerarquía) usando **sólo** las posiciones del archivo real (sin pasarle `adj` ni `dens_bar` — se le pasó `None` a propósito para forzar un error si de verdad los usara) y comparar contra la columna de velocidad ya escrita en el archivo real | **diff máxima = 0.0** — la velocidad es función PURA de la posición; el propio código de `campo_velocidad_turbulento.py` lo documenta ("adj/dens_bar se ignoran a propósito") y esto lo confirma empíricamente, no sólo por lectura |
| (3) ¿La masa y `h` son atributos por-partícula (podrían "pertenecer" a un nodo) o constantes globales? | inspección directa del archivo real: la cabecera trae un único `masa_particula=` global (no hay columna de masa por fila); `h` tiene **un solo valor único** (1.2) en las 2000 filas | uniforme — ninguno de los dos puede "viajar" con la identidad de un nodo porque ningún nodo tiene un valor distinto de otro |

**Conclusión de Paso 0:** en este pipeline, lo único que un nodo del grafo causal REAL "le aporta" a la
condición inicial de Phantom es su posición final (calculada una sola vez, vía `layout_resortes` sobre la
malla causal completa). Una vez calculada esa posición, absolutamente todo lo demás que se escribe para esa
partícula (masa, `h`, velocidad) es o bien una constante global (masa, `h`) o bien una función pura del
VALOR de esa posición (velocidad, que ni siquiera recibe `adj`/`dens_bar` como argumento útil). La
topología (`adj`) en sí misma nunca se vuelve a leer después de calcular el layout — el formato de IC que
lee `phantomsetup` no tiene ningún campo de identidad de nodo ni de adyacencia. Por lo tanto: **permutar
qué nodo ocupa cuál posición no puede cambiar ningún valor físico de ninguna partícula** — como mucho puede
reordenar las FILAS del archivo, que es un dato distinto (posición en el array, no en el espacio).

---

## Qué se construyó igual (`null5_generar_ic.py`, nuevo)

Reconstruye la malla causal REAL exacta y el layout físico final REAL exacto (idéntico método que Paso 0,
verificado bit a bit), aplica una permutación de fila `rng.permutation(n)` según `seed_permutacion`
(`pos_null5 = pos_real[permutacion]`), recalcula la velocidad a partir de la posición YA permutada (pura
función de posición, así que reproduce el mismo valor que tenía esa posición en REAL, sólo que ahora en
la fila de otro nodo), y antes de escribir el archivo compara el **multiset completo** de tuplas
`(posición, velocidad)` de NULL-5 contra el de REAL con un `assert` — deben coincidir exactamente como
conjunto, sólo el orden de fila puede diferir. Se generaron 2 condiciones (semillas de permutación 801,
802): el assert pasó en las 2 (ver log de generación abajo), confirmando por construcción, no sólo por el
Paso 0 analítico, que el contenido físico es idéntico a REAL.

```
[ic_null5_s801] n_aristas=4945 (assert multiset pos+vel==REAL paso OK) tiempo=51.7s
[ic_null5_s802] n_aristas=4945 (assert multiset pos+vel==REAL paso OK) tiempo=49.5s
```

---

## Corrida de verificación secundaria a través de Phantom real (no es "la batería" de NULL-5 propiamente dicha)

Dado que la hipótesis original de NULL-5 ("¿importa la identidad de nodo?") ya quedó resuelta por
construcción — colapsa trivial —, correr una batería completa de 8 semillas contra REAL habría sido, en
efecto, comparar REAL contra copias de sí mismo con las filas barajadas: no es una pregunta nueva sobre la
física del sistema, es una pregunta sobre si el ORDEN DE FILA en el archivo ASCII le importa a
`phantomsetup`/`phantom` por algún motivo de implementación (suma de fuerzas en punto flotante,
construcción de árbol, etc.) — una pregunta distinta, más débil, y barata de chequear. Por eso se corrieron
sólo **2** semillas de permutación (no 8, no una batería completa) como diligencia honesta, mismos
parámetros físicos que toda la jerarquía (`icreate_sinks=1`, `rho_crit_cgs=1000`, `r_crit=0.6`,
`h_acc=0.3`, `tmax=0.500`, `dtmax=0.001`), en
`/Users/alexis/phantom_cs073/bateria_null5_n2000/ic_null5_s{801,802}/`:

| corrida | exit setup | exit run | t_run | masa total sumideros | nº sumideros | masas individuales |
|---|---|---|---|---|---|---|
| REAL (`ic_real`, referencia directa, MISMA corrida ya en disco) | — | — | — | **2124.4** | **8** | 216.2, 225.6, 244.4, 263.2, 282.0, 282.0, 300.8, 310.2 |
| NULL-5 seed 801 (fila permutada) | 0 | 0 | 11.20s | **2124.4** | **8** | 216.2, 225.6, 244.4, 263.2, 282.0, 282.0, 300.8, 310.2 |
| NULL-5 seed 802 (fila permutada) | 0 | 0 | 11.49s | **2124.4** | **8** | 216.2, 225.6, 244.4, 263.2, 282.0, 282.0, 300.8, 310.2 |

**Las 3 filas de la tabla son idénticas** — masa total, número de sumideros y el conjunto completo de
masas individuales, sin ninguna diferencia, en ambas semillas de permutación. Esto confirma, ahora en el
comportamiento real del binario de Phantom (no sólo en el archivo de texto), que ni la identidad de nodo
ni siquiera el orden de fila en el ASCII tienen efecto alguno sobre el resultado en este pipeline: la
diferencia de punto flotante por reordenamiento de filas (si existe) queda muy por debajo de cualquier
cifra reportada por Phantom en el `.sink` (precisión a 1 decimal).

**Por qué esto NO se reporta como "NULL-5 (n=2) vs REAL (n=6), p=X" con el test de permutación de la
jerarquía:** las 2 corridas NULL-5 no son realizaciones físicas independientes de un proceso distinto —
son, por construcción demostrada arriba, la MISMA realización física de REAL con las filas del archivo
reordenadas. Tratarlas como una muestra estadística de n=2 comparable a NULL-1..4 (que sí son
realizaciones nuevas de un mecanismo distinto) inflaría artificialmente la aparente "supervivencia" de
NULL-5 frente a NULL-1/2 (masa 0) — no porque NULL-5 sea un mecanismo tan robusto como REAL, sino porque
NULL-5 **es** REAL. Se registra el hallazgo (colapso trivial, confirmado end-to-end) en vez de forzar un
test de permutación sin sentido estadístico.

---

## Tabla final — jerarquía completa de 6 controles (NULL-0 a NULL-5)

| escalón | qué preserva de REAL | qué destruye | resultado (masa en sumideros) | lectura sin cerrar nada |
|---|---|---|---|---|
| **NULL-0** (sanity check, `NULL0_masa_total_verificacion_CS.md`) | masa total del sistema | todo lo demás (posición, densidad, estructura) — chequeo trivial, no una simulación nueva | 9/9 corridas arrancan con masa total idéntica = 18800.0 | pasa — la batería es comparable, no aporta señal sobre estructura |
| **NULL-1** (ángulo isótropo aleatorio, mismo radio que REAL) | distribución RADIAL (perfil r) | ángulo/estructura angular — sin grafo/proceso relacional de fondo | **0** sumideros, masa 0 (n=8/8) | cero absoluto — perfil radial solo no basta |
| **NULL-2** (Zel'dovich, espectro de potencia parcial) | espectro de potencia P(k) (2 puntos) | fase/estructura de orden superior — sin grafo/proceso relacional de fondo | **0** sumideros, masa 0 (n=8/8) | cero absoluto — espectro de potencia solo no basta |
| **NULL-3** (double-edge-swap + filtro de longitud, ~87.5% de aristas preservadas) | grado + longitud típica de arista (grafo aproximado) | ~12.5% de las aristas específicas, algo de motivos/ciclos | media=2186.68, DE=53.16, rango 2068.0–2246.6, **8/8 sumideros** | REAL≈NULL-3, p=0.4212 (REAL vs NULL-3) — casi cualquier grafo relacional aproximado alcanza |
| **NULL-4** (topología 100% idéntica, orden de inserción/formación rebarajado) | topología COMPLETA (el 100% de las 4945 aristas) | el ORDEN en que esas aristas se insertaron antes del layout | media=2136.93, DE=33.01 (n=3), **3/3 sumideros** (8-9 cada una) | REAL≈NULL-4≈NULL-3, p=0.1786 (REAL vs NULL-4) — el orden de formación no bastó para romper nada, con las salvedades de n=3 y el caveat de mecanismo ya documentados en `NULL4_resultado_CS.md` |
| **NULL-5** (topología 100% + conjunto de posiciones 100% idénticos, correspondencia nodo↔posición permutada) | topología COMPLETA **y** el conjunto de posiciones finales COMPLETO (ambos intactos) | sólo la ETIQUETA — qué nodo específico ocupa cuál posición específica | **COLAPSA TRIVIAL** — 2/2 corridas de verificación reprodujeron a REAL EXACTO: masa 2124.4, 8 sumideros, mismas 8 masas individuales, sin ninguna diferencia | no es que "la identidad de nodo no importe estadísticamente" — es que, en este pipeline, ningún archivo escrito para Phantom PUEDE variar con la identidad de nodo: verificado por construcción (Paso 0) y confirmado en el binario real |

**Patrón consolidado de los 6 escalones:** el corte relevante en toda la jerarquía sigue siendo el mismo
que ya se veía desde NULL-1..4 — **preservar ALGO del grafo/proceso relacional de fondo** (aunque sea
aproximado, como NULL-3, o con orden de formación distinto, como NULL-4, o incluso reducido a "sólo qué
posiciones ocupó ese proceso relacional en conjunto", como NULL-5) reproduce formación de sumideros en el
mismo orden de magnitud que REAL; **no tener ningún grafo/proceso relacional de fondo** (NULL-1: sólo
perfil radial: NULL-2: sólo espectro de potencia) da cero absoluto. NULL-5 no agrega una separación nueva a
ese patrón porque, dado cómo está construido el pipeline actual, no puede aislar la variable que pretendía
aislar (identidad de nodo independiente de su posición) — la pregunta queda **respondida como "no
operacionalizable en este pipeline"**, no como "identidad de nodo = irrelevante para la física real", que
sería una lectura más fuerte de lo que estos números permiten.

---

## Nota honesta sobre el alcance de "colapsa trivial"

Esto NO dice que la pregunta original de Alexis ("¿importa la identidad del nodo, o alcanza con que
cualquiera esté en esa posición?") esté mal planteada — dice que, **en este pipeline concreto** (donde masa
y `h` son constantes globales, y la velocidad es una función pura de la posición final que ignora
`adj`/`dens_bar` a propósito, ver `campo_velocidad_turbulento.py`), no hay ningún canal por el que la
identidad de un nodo pueda todavía influir el resultado una vez que su posición ya fue calculada. Si en el
futuro algún generador de condición inicial hiciera depender la masa, la velocidad o `h` de un atributo
propio del nodo (por ejemplo, de `dens_bar[nodo]` directamente en vez de sólo a través de la posición, o
de alguna propiedad de sus vecinos causales), NULL-5 volvería a ser una pregunta operacionalizable y no
trivial — igual que NULL-4 lo fue gracias a la dependencia de orden en `layout_resortes(seed=...)`. Con el
pipeline actual, la respuesta honesta es: no hay tal canal.

---

## Tiempo de cómputo real vs. salvaguarda

Salvaguarda pedida: ~50-60 min totales.

| paso | tiempo |
|---|---|
| Lectura de antecedentes (NULL3/4, INFRA masa fija, `null4_generar_ic.py`, `p_semilla_causal.py`) | — (no cronometrado, parte de la preparación) |
| `null5_verificar_colapso.py` (Paso 0 — reconstrucción + verificación de velocidad/masa/h) | <5 s |
| Generación 2 IC NULL-5 (`null5_bateria_generar.py`) | 101.1 s |
| Batería Phantom 2 corridas (`null5_bateria_correr.py`) | 24.3 s |
| Comparación (lectura directa de `.sink`) | <1 s |
| **total cómputo de esta tarea** | **≈131 s ≈ 2.2 min** |

Muy por debajo de la salvaguarda — se priorizó verificar con rigor (bit a bit, con assert de multiset, y
con una corrida real de Phantom de confirmación) en vez de forzar una batería completa de 8 semillas que,
dado el colapso trivial ya demostrado, habría sido tiempo de cómputo sin pregunta nueva detrás.

---

## Entregables de esta tarea

- `null5_verificar_colapso.py` — verificación previa (Paso 0): reconstrucción bit-idéntica de malla+layout
  REAL, velocidad como función pura de posición (recomputada sin `adj`/`dens_bar`, comparada contra el
  archivo real), masa/`h` uniformes. Concluye colapso trivial ANTES de generar ninguna condición inicial.
- `null5_generar_ic.py` — módulo generador NULL-5: reconstruye malla+layout REAL exactos, permuta fila
  (`seed_permutacion`), recalcula velocidad desde la posición permutada, `assert` de multiset
  `(posición,velocidad)` NULL-5 == REAL antes de escribir. Reutiliza `p_semilla_causal.py`,
  `p_expansion.py`, `cs073_cierre_holistico.py`, `fase1_traducir_a_phantom.py` (sólo importados, no
  tocados).
- `null5_bateria_generar.py` / `null5_bateria_correr.py` — generación y corrida de 2 condiciones NULL-5
  (semillas de permutación 801, 802) en `/Users/alexis/phantom_cs073/bateria_null5_n2000/`.
- `/Users/alexis/phantom_cs073/bateria_null5_n2000/` — carpeta nueva con las 2 corridas de Phantom (IC,
  `cosmog.in`, `setup.log`, `run.log`, `.sink`, dumps). No se tocó ninguna carpeta de batería anterior
  (`bateria_n2000/`, `bateria_null1_n2000/`, `bateria_null2_n2000/`, `bateria_null3_n2000/`,
  `bateria_null4_n2000/`, `bateria_real_extra_n2000/`) ni ningún script congelado (`p_semilla_causal.py`,
  `grafo_random_layout_generar_ic_masa_fija.py`, `leer_volcado_phantom.py`, `null4_generar_ic.py`,
  `null4_verificar_invarianza_orden.py`) — sólo lectura/importación.
- Este informe, con la tabla final consolidando los 6 escalones (NULL-0 a NULL-5) de la jerarquía completa.

No se declara cierre ni veredicto sobre CS073 ni sobre esta jerarquía — los números de arriba son el
entregable; la síntesis final de los 6 escalones es de Alexis.
