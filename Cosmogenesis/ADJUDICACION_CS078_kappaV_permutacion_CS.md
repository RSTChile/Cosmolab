# ADJUDICACIÓN CS078 — κ_V (Bloque 2.8) con rigor estadístico completo

**Fecha:** 5-ago-2026 · **Frente:** Fase I-D, prioridad P0 (roadmap multi-IA, Bloque 2.8, invariante κ_V) · **Script:** `cs078_kappaV_permutacion.py`

**Nota de proceso:** hubo más de una pasada sobre este mismo frente en paralelo (nombres de
archivo coincidentes). Esta es la versión final: cubre las cuatro cosas que pidió el análisis
externo (roadmap Fase I-D) — distribuciones completas a nivel de sumidero individual, test de
permutación con la unidad documentada, evaluación cuantitativa de si los 8 NULL comparten
estructura, y bootstrap jerárquico sin el bug de NaN que traía una pasada anterior (algunas
réplicas bootstrap caían en el caso degenerado masa_primer_tercio=0 y contaminaban el promedio
con NaN; ahora se descartan explícitamente y se cuentan, no se ocultan). No se corrió ninguna
simulación Phantom nueva en ningún momento — todo se releyó de los `.sink` ya existentes en
`/Users/alexis/phantom_cs073/bateria_n2000/` (ic_real + ic_null1..8).

## Qué se midió y cómo

Por cada sumidero individual (8 en REAL; 7-8 por corrida NULL, 63 en total), κ_V = masa
acretada en el último tercio de su vida / masa acretada en el primer tercio, con los tiempos de
frontera de los tercios interpolados linealmente sobre la masa acretada acumulada (columna
`macc`, idéntica en estos datos a la masa del sumidero — no pierde masa, sólo acreta).

Agregado principal por corrida: razón AGRUPADA (Σ masa último tercio / Σ masa primer tercio de
todos los sumideros de esa corrida) — robusta a que un sumidero individual no haya acretado
nada en su primer tercio (9 de 63 sumideros NULL caen en ese caso degenerado; 0 de 8 REAL).

| Corrida | κ_V (razón agrupada) | n sumideros | indefinidos (1er tercio=0) |
|---|---|---|---|
| **REAL** | **0,8437** | 8 | 0 |
| null1 | 0,6000 | 8 | 1 |
| null2 | 0,4000 | 8 | 4 |
| null3 | 0,5526 | 8 | 2 |
| null4 | 0,8182 | 7 | 0 |
| null5 | 0,5833 | 8 | 0 |
| null6 | 0,2083 | 8 | 0 |
| null7 | 0,4118 | 8 | 0 |
| null8 | 0,3636 | 8 | 2 |

NULL media±DE = 0,4922 ± 0,1859 (n=8 corridas). z aproximado (asumiendo normalidad) = 1,89.

*(El documento de diseño original reportó REAL=0,832, NULL=0,511±0,235, z=1,37. Esta
reconstrucción da números en la misma zona pero no idénticos — el script original que produjo
esos números exactos no está en disco; esta es una reconstrucción fiel del método descrito, no
una repetición byte-a-byte. La diferencia no cambia la lectura cualitativa.)*

## 1. Distribuciones completas (no sólo media±DE)

A nivel de sumidero individual (no de corrida):

- **REAL** (n=8): media=0,818, DE=0,250, mediana=0,813, rango [0,50 – 1,22].
- **NULL agrupado** (n=54 válidos de 63; se excluyen 9 indefinidos): media=0,534, DE=0,609,
  mediana=0,417, rango [0,00 – 2,00].

Las distribuciones se solapan bastante (la de NULL es mucho más ancha y tiene masa grande en 0),
pero REAL está sistemáticamente desplazado hacia arriba y no tiene ningún caso de "cero
acreción temprana", cosa que sí ocurre en 9 de 63 sumideros NULL.

## 2. Test de permutación — unidad de permutación: la CORRIDA, no el sumidero

**Por qué la corrida es la unidad correcta:** los 7-8 sumideros de una misma corrida NULL nacen
de la MISMA condición inicial / mismo campo de turbulencia de esa corrida — no son réplicas
independientes entre sí, comparten toda la dinámica de colapso de su corrida. Tratar los 63
sumideros NULL como 63 muestras independientes es pseudoreplicación.

**Test exacto a nivel de corrida** (9 unidades: 1 REAL + 8 NULL; C(9,1)=9 asignaciones posibles
bajo H0 de intercambiabilidad total):

- Estadístico = κ_V(corrida etiquetada REAL) − media(las 8 restantes).
- Distribución nula exacta (9 valores): [0,3515, 0,0773, −0,1477, 0,0240, 0,3228, 0,0585,
  −0,3633, −0,1345, −0,1886].
- REAL es el **más extremo de las 9** (rank 1/9).
- **p (una cola, H1 pre-registrada REAL>NULL) = 0,1111** — el piso de resolución exacto con
  n=9 es 1/9=0,111, así que este es el p-valor MÍNIMO POSIBLE que este diseño puede producir,
  sea cual sea el tamaño real del efecto.
- p (dos colas) = 0,2222.
- Con el agregado de robustez (media de razones individuales válidas en vez de razón agrupada)
  el resultado es más débil: rank 2/9, p=0,2222 — null4 (0,857) queda por encima de REAL
  (0,818) con este agregado alternativo. **La conclusión depende algo de qué agregado se use**,
  y con n=9 ninguno de los dos alcanza significancia convencional.

**Sensibilidad, marcada como IMPROPIA:** si se ignora la jerarquía y se permutan los 71
sumideros individuales sueltos (Monte Carlo, 200.000 permutaciones), p (una cola) = 0,107 — casi
igual al test correcto en este caso particular, pero eso es coincidencia de estos datos, no una
justificación general para pseudoreplicar. Se reporta sólo para mostrar cuánto (o cuán poco, acá)
cambia ignorar la estructura, no para reemplazar el test de la sección anterior.

## 3. ¿Los 8 NULL son 8 semillas independientes o comparten estructura?

Se hizo una descomposición de varianza (ANOVA de un factor, factor=corrida) sobre los κ_V de
sumidero individual dentro de NULL: varianza ENTRE corridas vs varianza DENTRO de cada corrida.

- Medias de κ_V por corrida NULL: [0,64; 0,29; 0,49; 0,86; 0,63; 0,21; 0,70; 0,33] — a simple
  vista varían bastante (0,21 a 0,86).
- MS entre-corridas = 0,347; MS dentro-de-corridas = 0,375. El MS entre es, de hecho, **menor**
  que el MS dentro → el estimador de momentos del ICC da 0,000 (recortado; el valor sin recortar
  es levemente negativo, −0,004).
- **Lectura honesta, no maquillada:** con sólo 8 grupos de 4-8 sumideros cada uno, este
  estimador de ICC tiene varianza de muestreo enorme y su distribución está truncada en 0 — un
  "ICC=0" acá **no** demuestra que los sumideros de una corrida sean independientes entre sí,
  sólo que este test concreto, con esta cantidad de datos, no puede confirmar con soltura que la
  variación entre corridas exceda el ruido esperable de grupos tan chicos. El argumento físico
  para tratar la corrida como la unidad (misma condición inicial, misma dinámica de colapso
  compartida) sigue en pie independientemente de este resultado — por eso el test de la sección 2
  (permutación a nivel de corrida) es el que se usa como primario, no éste.
- N efectivo (Kish) del pool de 54 sumideros NULL válidos = 54,0 (design effect ≈ 1,0 con
  ICC=0) — con este estimador puntual no hay evidencia de que el pool esté inflado, pero dado lo
  anterior, **este número no debe leerse como luz verde para pseudoreplicar**: la razón para no
  hacerlo es el diseño del experimento (misma condición inicial por corrida), no sólo esta
  estimación estadística puntual, que con n=8 grupos tiene demasiado ruido para ser concluyente
  en cualquier dirección.

## 4. Intervalo de confianza no paramétrico (bootstrap jerárquico de dos etapas)

Bootstrap de 20.000 réplicas: remuestrea las 8 corridas NULL con reemplazo, y dentro de cada
corrida remuestreada, remuestrea sus propios sumideros con reemplazo (respeta la estructura de
dos niveles). 61 de 20.000 réplicas se descartaron por caer en el caso degenerado (masa primer
tercio total = 0 para alguna corrida remuestreada) — no se rellenaron con un número inventado ni
se dejó pasar el NaN silenciosamente.

- **Media NULL bootstrap = 0,518, IC 95% = [0,341, 0,733].**
- El valor REAL observado (0,844) cae en el **percentil 99,7** de esa distribución — por encima
  del límite superior del IC 95% de NULL.
- Para REAL sólo hay 1 corrida real (no se puede remuestrear "entre semillas" con n=1); el único
  IC que se puede construir para REAL remuestrea sus propios 8 sumideros (incertidumbre
  intra-corrida solamente): **IC 95% = [0,678, 1,000]**. Este intervalo SÍ se solapa con el IC de
  NULL — documentado explícitamente como una comparación en pie desigual (NULL incorpora
  variabilidad entre semillas; REAL, al tener n=1, no puede).

## Lectura honesta, sin cerrar el experimento

Con las cuatro mejoras pedidas (distribuciones completas, permutación con unidad documentada,
evaluación jerárquica por semilla, reporte honesto):

- La **dirección** sigue siendo la correcta en todos los métodos probados (REAL > NULL, en razón
  agrupada, en el test exacto de corrida, y en el percentil bootstrap).
- El **test de permutación exacto y válido** (nivel de corrida, la unidad correcta dado que los
  sumideros de una misma corrida comparten condición inicial) da **p=0,111 en el mejor caso**, y
  **p=0,222 con el agregado alternativo** — ninguno de los dos cruza el umbral convencional de
  p<0,05. Con sólo 9 corridas totales (1 REAL + 8 NULL), **el piso de resolución de cualquier
  test de permutación en este diseño es 1/9≈0,111** — este diseño no puede, por construcción,
  demostrar significancia convencional aunque el efecto real fuera exactamente el observado.
- El bootstrap jerárquico sitúa a REAL por encima del IC 95% de NULL a nivel de corrida — una
  señal algo más favorable que el test de permutación exacto — pero la comparación con el IC de
  REAL (que no puede incorporar variabilidad entre semillas, n=1) no es una comparación en pie
  de igualdad, y así se marca.
- La evaluación jerárquica (sección 3) no puede confirmar ni descartar con soltura que los 8
  NULL compartan estructura más allá del ruido esperable — que es, en sí, información real: hacen
  falta más semillas (NULL y, sobre todo, REAL, de la que hoy existe una sola) para que esta
  pregunta tenga poder estadístico de responderse.

**No se declara "confirmado" ni "refutado".** Con los datos ya existentes, κ_V queda exactamente
donde estaba en el documento de diseño: dirección correcta, sin fuerza estadística suficiente
para decidir con el rigor pedido. Una réplica genuina (más semillas NULL, y sobre todo más de una
semilla REAL, algo que hoy no existe en absoluto) requeriría nuevas corridas de Phantom — horas
de cómputo — que quedan, como siempre, pendientes de que el director del proyecto decida si vale
la pena autorizarlas. Ese cierre no es de este informe.
