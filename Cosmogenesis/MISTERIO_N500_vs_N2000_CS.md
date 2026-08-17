# MISTERIO N=500 vs N=2000 — ¿artefacto de resolución, falta de tiempo, u otra cosa?

**Fecha:** 6-ago-2026 · **Método:** sólo lectura (`leer_volcado_phantom.py`, sarracen), sobre datos ya
existentes del control grafo-random (`piloto_grafo_random/random_s1` N=500 vs
`bateria_grafo_random_n2000/ic_random_s701` N=2000). No se corrió Phantom nuevo (Pasos 1-3 cubrieron el
presupuesto de tiempo; Paso 4 opcional no se ejecutó — ver "Qué falta" al final). No se tocó ninguna
carpeta congelada, sólo se importó/leyó.

**Antecedente que se estaba testeando:** la hipótesis de que "cero sumideros a N=500" es un artefacto de
resolución numérica — que a esa escala ningún grumo alcanza los ~116 vecinos que pide el criterio de Bate
& Burkert 1997 (citado en `AUDITORIA_COMPLETA_COSMOGENESIS_2026.md`, sección "El freno"), y que por eso
el resultado es indistinguible de "estructura vs azar" — sería un artefacto numérico, no un resultado
sobre REAL/NULL/random.

---

## Hallazgo previo, no anticipado por la hipótesis: N=500 y N=2000 NO son el mismo sistema físico

El encargo pedía verificar, no asumir, que la masa total del sistema fuera ~18800 en ambos casos. **No lo
es.** Los datos muestran:

| | N=500 (`random_s1`, dump final t=0.5) | N=2000 (`ic_random_s701`, dump final t=0.5) |
|---|---|---|
| Partículas de gas | 500 | 1883 (117 ya acretadas en sumideros) |
| `massoftype` (masa/partícula) | **9.4** | **9.4** (idéntica, no ×4) |
| Masa total del sistema (gas+sumideros) | **4700** | **18800** (17700.2 gas + 1099.8 en 8 sumideros) |
| Razón de masa total N500/N2000 | **0.2655** (≈1/3.77, ≈ N500/N2000 en partículas) | — |
| Lado de la caja (`lado = n^(1/3)` × factor de expansión) | ≈ 61.5 unidades | ≈ 97.6 unidades (razón 1.587 ≈ 4^(1/3), exacta) |

La masa por partícula (9.4) es una constante fija del proyecto (ya verificada para N=2000 en
`NULL0_masa_total_verificacion_CS.md`, pero esa verificación nunca comparó **entre** escalas N distintas —
sólo entre REAL y NULL dentro de la misma batería N=2000). El generador congelado
(`grafo_random_layout_generar_ic.py`, línea 116: `lado = float(n) ** (1.0/3.0)`) hace que la caja crezca
con `n^(1/3)`: a más N, caja más grande, MISMA densidad numérica de partículas y MISMA densidad de masa,
pero **más masa total y más volumen absoluto**, no la misma masa repartida más fino.

**Consecuencia:** N=500 y N=2000 no son "el mismo sistema muestreado con más o menos partículas" — son dos
cajas de distinto tamaño con la misma densidad media, y por lo tanto con distinta masa de Jeans relativa a
la caja, distinto tiempo de caída libre, y distinta cantidad absoluta de masa autogravitante. Esto es un
tercer factor, no contemplado en la hipótesis original de "sólo resolución", que se mezcla con cualquier
lectura de "cero sumideros a N=500".

---

## Paso 1 — resolución directa (h, masa/partícula, vecinos)

- **h (longitud de suavizado):** N500 mediana=11.10, media=10.15, max=13.93 (partículas de gas difuso,
  nunca colapsaron). N2000 (dump final, ya post-formación de sumideros, gas difuso remanente):
  mediana=9.36, media=8.60, max=15.84. La h típica es algo *mayor* a N=500, consistente con menor densidad
  numérica de muestreo — pero la comparación en el dump final está sesgada porque a N=2000 el gas más
  denso ya fue devorado por los 8 sumideros; no es una comparación limpia de "el mismo tipo de región".
- **Masa por partícula:** idéntica (9.4) — ver hallazgo arriba, no es 4× como se esperaba.
- **Vecinos en la partícula de densidad máxima (radio = k×h_local):**

  | radio | N500 (dump final, nunca colapsa) | N2000 dump final (post-sumidero, gas difuso) | N2000 dump050, t=0.05 (pico real pre-sumidero, ρ_max=3093) |
  |---|---|---|---|
  | 1h | 6 | 6 | 5 |
  | 2h | 9 | 8 | 6 |
  | 3h | 11 | 15 | 7 |
  | 4h | 11 | 23 | 8 |
  | 5h | 14 | 26 | 8 |

  Con `hfact=1.2` (default de Phantom, cúbico), el kernel apunta a ~50-60 vecinos dentro del soporte
  compacto (~2h) en un medio bien resuelto. Ninguna de las tres columnas se acerca a eso — ni siquiera la
  columna del colapso real de N2000 justo antes de formar sumidero. Esto sugiere que la propia región de
  colapso, incluso cuando SÍ colapsa (N2000), está en el límite bajo de vecinos que el algoritmo de Phantom
  querría idealmente — es decir, el déficit de vecinos no es exclusivo de N=500 aunque N=500 no llega ni
  remotamente a formar el pico de densidad que sí forma N=2000.

## Paso 2 — trayectoria de densidad máxima en el tiempo (51 puntos, cada Δt=0.01, t=0..0.5=tmax)

- **N500:** ρ_max arranca en 0.742, baja levemente a ~0.732 hacia t=0.04, y desde ahí **sube de forma
  monótona y con aceleración suave** hasta 2.175 en t=0.5 (el final de la corrida). **Nunca se estanca** —
  la curva sigue subiendo en el último tramo (de t=0.4 a t=0.5 sube ×1.46). Nunca aparece un sumidero en
  las 501 salidas.
- **N2000:** ρ_max sube muy rápido desde 139 (t=0) hasta un pico agudo de 4384 en t=0.06 (ahí nacen los 2
  primeros sumideros), luego **se derrumba** a 2-3 hacia t=0.1 cuando los 8 sumideros (ya completos en
  t=0.1) devoran el núcleo colapsado, y el ρ_max remanente (gas difuso, ya no la estructura colapsada)
  oscila entre 0.8 y 1.4 el resto de la corrida.
- **Lectura del Paso 2:** a t=tmax=0.5, ρ_max de N500 (2.175) está ~460× por debajo de `rho_crit_cgs=1000`,
  pero la curva **no muestra signos de techo/plateau** — sigue subiendo. Una extrapolación ingenua del
  ritmo de crecimiento del último tramo (factor ×1.46 cada Δt=0.1) necesitaría del orden de 3-4× el tmax
  actual para cruzar ρ_crit — una extrapolación cruda, no una predicción firme, pero suficiente para decir
  que **los datos NO muestran el patrón esperado de un techo de resolución duro** (que sería una curva que
  sube y se aplana en una asíntota muy por debajo de 1000, cosa que no se observa aquí).

## Paso 3 — partículas en la región de mayor densidad, N=500, dump final, contra el criterio de ~116

| radio | vecinos (N500) |
|---|---|
| 1×h | 6 |
| 2×h | 9 |
| 3×h | 11 |
| 4×h | 11 |
| 5×h | 14 |

Todos los valores están **muy por debajo de 116** — incluso a 5×h (14 vecinos) se está casi un orden de
magnitud por debajo del umbral citado en la auditoría. Esto es evidencia directa, concreta, de que la
región más densa de N=500 nunca alcanza la cantidad de vecinos que el criterio de Bate & Burkert 1997 pide
para confiar en un colapso — consistente con la hipótesis de resolución insuficiente.

---

## Lectura honesta — ¿(a) artefacto de resolución, (b) falta de tiempo, o (c) algo distinto?

Los tres pasos, tomados juntos, **no dan un único ganador limpio entre (a) y (b)** — y revelan un
**candidato (c) no anticipado**:

- **A favor de (a) resolución insuficiente:** el conteo de vecinos en la región más densa de N=500 (Paso
  3) está muy por debajo de 116 en todas las escalas de radio probadas, y también por debajo de lo que el
  propio `hfact=1.2` de Phantom apunta a tener (~50-60). Esto es consistente con la hipótesis original.

- **En contra de (b) "es sólo falta de tiempo" como única explicación:** la curva de ρ_max de N500 (Paso 2)
  sigue subiendo, sin plateau, hasta el final de la corrida — así que no se puede descartar que con más
  tiempo llegara más lejos. Pero tampoco hay evidencia de que fuera a cruzar `rho_crit=1000` en un tiempo
  razonable (la extrapolación cruda pide 3-4× el tmax actual), y el crecimiento observado es mucho más
  lento y suave que el colapso real de N2000 (que en el mismo rango de tiempo pasó de ρ~140 a ρ~4384, casi
  30× en Δt=0.06 — un orden de magnitud más rápido que cualquier tramo de N500). Esta diferencia de
  *ritmo*, no sólo de tiempo disponible, apunta a que "sólo faltó tiempo" es insuficiente por sí sola.

- **(c) hallazgo no contemplado por la hipótesis original, y que la explica mejor que (a) o (b) solas:**
  N=500 y N=2000 **no son el mismo sistema a distinta resolución** — son cajas de tamaño distinto
  (`lado ∝ n^(1/3)`) con la misma densidad media pero **4× menos masa total absoluta** a N=500. Eso cambia
  la física real que se está comparando, no sólo cuántas partículas la resuelven: una caja más chica con
  menos masa total tiene, en general, menos modos de perturbación super-Jeans disponibles y una dinámica de
  colapso intrínsecamente distinta — independientemente de cuántos vecinos tenga cada partícula. El
  contraste "cero sumideros a N=500, sumideros consistentes a N=2000" observado en el patrón repetido
  (grafo random y también en los NULL originales) podría estar mezclando **dos efectos superpuestos**: (i)
  resolución SPH insuficiente (Paso 3, bien evidenciado) y (ii) una caja/masa total insuficiente para
  contener un modo inestable, que es un efecto físico real de la construcción de la condición inicial, no
  un artefacto numérico ni un problema de tiempo de integración.

**No se declara veredicto.** Los tres pasos dan evidencia a favor de "resolución insuficiente" (Paso 3) y
evidencia de que "sólo faltó tiempo" no alcanza a explicarlo solo (Paso 2), pero el hallazgo de masa
total/tamaño de caja escalando con N (no verificado antes de esta tarea) es un confound genuino que ninguna
de las dos lecturas originales contemplaba. Corresponde a Alexis decidir qué peso darle a cada lectura y si
amerita rehacer el control de grafo-random con masa total fija (en vez de masa por partícula fija) para
desacoplar "resolución" de "tamaño físico del sistema".

---

## Qué falta (Paso 4, no ejecutado por presupuesto de tiempo)

No se corrieron semillas nuevas a N intermedio (N=1000/1200). Dado el hallazgo de masa total escalando con
N, si se hace ese seguimiento convendría además correr un control con masa por partícula ajustada para
mantener la masa total fija (~18800) en vez de dejarla escalar — así se separaría limpiamente el efecto de
"más vecinos por partícula" del efecto de "más masa/caja más grande", que hoy están mezclados en la propia
construcción de `grafo_random_layout_generar_ic.py` (script congelado, no tocado).

## Archivos usados (sólo lectura)

- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/leer_volcado_phantom.py` (importado, no modificado)
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/grafo_random_layout_generar_ic.py`,
  `grafo_random_piloto_generar.py`, `grafo_random_bateria_generar.py` (leídos para entender la
  construcción de masa/caja, no modificados)
- `/Users/alexis/phantom_cs073/piloto_grafo_random/random_s1/` (N=500, dumps `cosmog_00000`..`cosmog_00500`)
- `/Users/alexis/phantom_cs073/bateria_grafo_random_n2000/ic_random_s701/` (N=2000, mismos dumps)
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/NULL0_masa_total_verificacion_CS.md` (contexto del
  chequeo de masa total previo, sólo dentro de N=2000)
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/AUDITORIA_COMPLETA_COSMOGENESIS_2026.md` (criterio de
  ~116 vecinos, sección "El freno")
- Script de análisis (scratchpad, no en el repo):
  `/private/tmp/claude-501/-Users-alexis-Desktop-RMD-Cosmolab-Cosmogenesis/cad269e0-d7b3-425d-8ec8-abc474b7f497/scratchpad/analisis_n500_vs_n2000.py`
