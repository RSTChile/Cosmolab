# FASE VII · F7-02 — Escalera de clustering: de "covaría" a "lo movimos y respondió"

**Fecha:** 12-ago-2026 · **Ejecuta:** CC (Claude) · **Tarea:** F7-02 de la Fase VII
**Antecedente directo:** `FASE6_O3B_control_rewiring_CS.md` §5.4 y `INFORME_EQUIPO_FASE6_11ago2026_CS.md` Parte 3.bis
**Phantom:** autorizado por Alexis para esta línea · **Diámetro:** medición oficial vigente
(`cs090_diam_corregido.diam_gigante`)

> No se declara cierre ni veredicto. Ningún script congelado fue modificado (todos sólo se importan).
> No se hicieron commits de git.

---

## 0. En simple, con analogía

En O3-B habíamos encontrado algo que apuntaba a un mecanismo: cuando a una maqueta de alambre se le
desatan y reatan todos los alambres cuidando que cada nudo conserve exactamente su número de alambres,
la maqueta original conserva una ventaja de ~5% en arena juntada, y **lo que predice cuánta ventaja
tiene cada maqueta es cuántos "triangulitos" tenía** (ρ=0.77), no lo estirada que estaba (ρ=0.04).

Pero eso era **mirar**, no **tocar**. Doce maquetas distintas, cada una con su propio número de
triangulitos, y una correlación entre dos cosas que podrían estar las dos causadas por una tercera.

Esta tarea toca. Agarra el número de triangulitos como si fuera una perilla y lo sube y lo baja **en la
misma maqueta**, dejando clavado todo lo demás: la misma cantidad de nudos, la misma cantidad total de
alambre, y —nudo por nudo— exactamente el mismo número de alambres en cada uno. Después se llenan todas
las variantes de arena igual y se las sacude igual.

**Si la arena sube cuando sube la perilla, dejamos de decir "observamos que van juntas" y empezamos a
decir "lo movimos y respondió".**

---

## 1. Qué se hizo, con qué archivos

| Archivo nuevo | Qué hace |
|---|---|
| `cs090_fase7_f702_escalera.py` | Selecciona los grafos base, construye la escalera de clustering con swaps dirigidos, verifica grados nodo por nodo, mide estructura y pendiente corregida, escribe las condiciones iniciales de Phantom |
| `cs090_fase7_f702_correr.py` | Corre Phantom (mismo protocolo exacto de toda la línea) |
| `cs090_fase7_f702_analizar.py` | Verificación cruzada contra `meta_regla.json`, extrae métricas, estadística pareada y PNG |

| CSV / PNG de salida | Contenido |
|---|---|
| `cs090_fase7_f702_estructura_shard{0..3}.csv` | estructura medida de cada escalón (crudo) |
| `cs090_fase7_f702_phantom_crudo.csv` | una fila por corrida de Phantom |
| `cs090_fase7_f702_por_grafo.csv` | una fila por grafo base: su escalera completa y su ρ intra-grafo |
| `cs090_fase7_f702_estadistica.csv` | todas las pruebas |
| `cs090_fase7_f702_escalera.png` | la escalera dibujada |

Baterías de Phantom en `/Users/alexis/phantom_cs073/bateria_fase7_f702_escalera/`, carpetas
`<rule_id>_s<seed>_f702_<escalon>` — **prefijo `f702` jamás usado antes en la línea**, y con el **seed
dentro del nombre** porque hay reglas distintas con el mismo `rule_id` en lotes distintos (bug
documentado en `FASE6_O3B` §2.1; sin el seed dos grafos distintos se pisarían la carpeta).

Scripts congelados sólo importados, nunca tocados: `cs090_fase6_o3b_rewiring.py` (funciones de medición),
`cs090_fase5b_phantom_adaptador.py`, `cs090_fase5b_analizar.py`, `cs090_fase5_generador/motor/clasificador.py`,
`cs090_diam_corregido.py`, `cs080_renormalizacion.py`, `cg003_diagnostico_gromov.py`,
`leer_volcado_phantom.py`. `null3_investigacion_preliminar.py` / `null3_generar_ic.py` se leyeron como
referencia del double-edge-swap; no se modificaron.

---

## 2. Cómo se mueve la perilla sin tocar nada más

La operación elemental es la misma de NULL-3 y de O3-B: el **double-edge-swap de Maslov-Sneppen**, que
toma dos aristas (a-b) y (c-d) y las reconecta como (a-d) y (c-b). Cada nodo pierde un vecino y gana
otro, así que **su grado nunca cambia**. La diferencia con O3-B es que allá los swaps eran **ciegos** (se
aceptaban todos) y acá son **dirigidos**:

- **bajar** el clustering: propuestas al azar, se aceptan sólo las que destruyen triángulos
  (Δtriángulos < 0), más las neutras con probabilidad 1/2 para que el grafo igual se mezcle;
- **subir** el clustering: propuesta **dirigida** — se elige un nodo u con ≥2 vecinos, dos vecinos suyos
  x,y que no estén conectados entre sí, y se hace el swap (x-p),(y-q) → (x-y),(p-q). La arista nueva x-y
  cierra el triángulo u-x-y. Se acepta sólo si el balance NETO de triángulos es positivo.

El cambio exacto de triángulos de cada swap se calcula localmente (`_delta_triangulos`): quitar la arista
a-b destruye exactamente |N(a)∩N(b)| triángulos y ningún triángulo puede contener dos aristas disjuntas,
así que no hay doble conteo. No se recuenta el grafo entero en cada intento.

**La escalera se arma en un solo recorrido desde un ancestro común**, para que todos los escalones sean
comparables entre sí:

```
   original --(swaps que destruyen triángulos, 10 x nº de aristas)--> E0 (piso, cero triángulos)
   E0 --(swaps dirigidos que crean triángulos, hasta 40 x nº de aristas)--> E1 -> E2 -> E3 -> E4 (techo)
```

Los escalones son las fotos del recorrido cuyo nº de triángulos cae más cerca del 0%, 6%, 18%, 45% y
100% del máximo **realmente alcanzado** (no de un número inventado de antemano).

### 2.1 — El techo tiene un freno: la componente gigante

En el piloto, dejando subir el clustering sin freno el grafo termina rompiéndose en camarillas: a
C=0.577 la componente gigante caía de 1926 a 1236 nodos. Eso metería un **segundo** cambio grande además
del clustering y arruinaría la intervención. Por eso el techo de la escalera se pone en la foto de más
triángulos que todavía conserva **≥97% de la componente gigante del original** (`FRAC_GIGANTE_MINIMA`).
Se reporta también dónde estaba el techo sin esa restricción.

### 2.2 — Por qué el piso E0 es también el control de "cantidad de barajado"

Una objeción natural sería: *los escalones altos recibieron más swaps que los bajos; ¿y si lo que importa
es la cantidad de recableo y no los triángulos?* El diseño ya contesta eso: **E0 es el escalón con MÁS
historia de recableo de todos** (10 × nº de aristas intentos de barajado) y es el que tiene **cero**
triángulos. Si "más recableo" fuera la causa, E0 debería ser el extremo de la respuesta, no el piso de la
escalera. Además, el solapamiento de aristas con el grafo original es **igual de bajo (~0.1–0.3%) en
todos los escalones**: ninguno está "más cerca del original" que otro.

---

## 3. Verificación de que lo único que cambió fue el clustering

Sobre cada escalón de cada grafo, en el código y con `assert` que aborta el grafo si falla:

| Chequeo | Cómo |
|---|---|
| Secuencia de grados idéntica **nodo por nodo** contra el original | `np.array_equal` sobre los 2000 grados |
| Nº de aristas idéntico | conteo explícito |
| Sin bucles i-i | recorrido explícito |
| N idéntico | 2000 en todos |

Además se **mide** (no se asume) en cada escalón: solapamiento de aristas con el original, clustering
local medio (Watts-Strogatz), transitividad global, nº de triángulos, componente gigante, nº de
componentes, **asortatividad de grados** (covariable que los swaps dirigidos podrían arrastrar sin
querer) y la **pendiente corregida** log(diám)–log(N_cajas) con la medición oficial de diámetro.

---

## 4. Phantom — protocolo y verificación cruzada

Idéntico al de toda la línea, sin un parámetro cambiado: N=2000, masa total fija=18800 (masa por
partícula 9.4), lado de caja fijo 2000^(1/3), `layout_resortes` con `seed_layout=12345`, dilatación
`Expansion` de 60 pasos, turbulencia Mach=3 seed=42, `icreate_sinks=1`, `rho_crit_cgs=1000`,
`r_crit=0.6`, `h_acc=0.3`, `tmax=0.500`, `dtmax=0.001`.

Verificaciones cruzadas hechas en el código antes de que ninguna corrida entre en la estadística:
la tarea declarada en el `meta_regla.json`, que el escalón y el `(rule_id, seed)` del meta coincidan con
el nombre de la carpeta, que la carpeta declarada dentro del meta sea la carpeta donde está el meta, que
el meta declare `grados_identicos_al_original = true`, que **todos los escalones de un mismo grafo
tengan el mismo nº de aristas y la misma `seed_layout`**, y que la corrida haya arrancado con las 2000
partículas de gas (chequeo anti-IC-truncado, porque generación y corrida se solapan en el tiempo).
**La unión con la estructura es por `(rule_id, seed, escalon)`**, nunca por `rule_id` solo.

---

## 5. La escalera que efectivamente se logró (no la pretendida)

**12 grafos base**, los mismos 12 de O3-B: las 3 de mayor pendiente corregida dentro de cada uno de los
4 lotes de `seed_base` (271828 / 371828 / 471828 / 571828). **5 escalones cada uno + el original sin
tocar como referencia = 72 corridas de Phantom**, todas con `exit_run=0`, dump final `cosmog_00500`,
501 volcados y 2000 partículas de gas iniciales.

| regla | lote | **C en e0** | **C en e1** | **C en e2** | **C en e3** | **C en e4** | C natural del original |
|---|---|---|---|---|---|---|---|
| `batch3-r0` | 471828 | 0.0000 | 0.0143 | 0.0456 | 0.1103 | 0.2354 | 0.0170 |
| `batch3-r111` | 471828 | 0.0000 | 0.0265 | 0.0829 | 0.1924 | 0.3789 | 0.0171 |
| `batch3-r60` | 471828 | 0.0000 | 0.0272 | 0.0718 | 0.1827 | 0.3653 | 0.0190 |
| `batch4-r10` | 571828 | 0.0000 | 0.0233 | 0.0730 | 0.1860 | 0.3865 | 0.0116 |
| `batch4-r36` | 571828 | 0.0000 | 0.0152 | 0.0442 | 0.1160 | 0.2543 | 0.0213 |
| `batch4-r62` | 571828 | 0.0000 | 0.0288 | 0.0923 | 0.2307 | 0.4392 | 0.0053 |
| `r14` | 271828 | 0.0000 | 0.0292 | 0.0933 | 0.2328 | 0.4535 | 0.0050 |
| `r17` | 271828 | 0.0000 | 0.0271 | 0.0752 | 0.1890 | 0.3832 | 0.0119 |
| `r19` | 271828 | 0.0000 | 0.0316 | 0.1006 | 0.2437 | 0.4819 | 0.0010 |
| `r20` | 371828 | 0.0000 | 0.0385 | 0.1195 | 0.2728 | 0.5162 | 0.0023 |
| `r28` | 371828 | 0.0000 | 0.0368 | 0.1183 | 0.2731 | 0.5203 | 0.0105 |
| `r39` | 371828 | 0.0000 | 0.0509 | 0.1429 | 0.3369 | 0.6050 | 0.0032 |

- **Rango logrado: C = 0.0000 → 0.235–0.605** (media del techo 0.418). El piso es **exactamente cero
  triángulos** en los 12; el techo es entre 11 y 480 veces el clustering natural del grafo del que salió.
- **Los grados son idénticos nodo por nodo en los 72 grafos** (`np.array_equal` sobre los 2000 grados,
  con `assert`): **0 nodos con grado distinto, 0 diferencias de nº de aristas, 0 bucles**.
- Solapamiento de aristas con el original: **0.1%–0.3% en todos los escalones por igual** — ningún
  escalón está "más cerca" del grafo original que otro.
- **El clustering natural de estos grafos (0.001–0.021) cae por debajo de e1 en 10 de 12 casos.** La
  escalera barre un rango mucho más ancho que el que la naturaleza de estas reglas produce sola: eso es
  bueno para detectar la respuesta, pero hay que tenerlo presente al leer el tamaño del efecto (§8).

### 5.1 — Lo que se movió de arrastre (y hay que descontar)

| covariable | e0 (media) | e4 (media) | comentario |
|---|---|---|---|
| componente gigante | 1884 | 1827 | baja ~3% (el techo está frenado a ≥97% de la del original, §2.1) |
| asortatividad de grados | −0.209 | −0.067 | sube con el clustering: es la covariable más fuerte |
| pendiente corregida | 0.725 | 0.838 | sube en promedio, pero sólo **1 de 12** grafos la tiene monótona |

Ninguna de las tres es inocua y las tres se descuentan explícitamente en §7.

---

## 6. Resultado central: la masa SÍ sigue a la perilla

**Fracción de masa en sumideros, un grafo por fila, escalones en orden de clustering creciente:**

| regla | e0 | e1 | e2 | e3 | e4 | ρ intra-grafo | monótona ↑ |
|---|---|---|---|---|---|---|---|
| `batch3-r0` | 0.1465 | 0.1485 | 0.1445 | 0.1490 | 0.1545 | +0.70 | no |
| `batch3-r111` | 0.1145 | 0.1155 | 0.1170 | 0.1255 | 0.1470 | +1.00 | **sí** |
| `batch3-r60` | 0.1155 | 0.1160 | 0.1180 | 0.1270 | 0.1485 | +1.00 | **sí** |
| `batch4-r10` | 0.1195 | 0.1180 | 0.1280 | 0.1315 | 0.1565 | +0.90 | no |
| `batch4-r36` | 0.1420 | 0.1465 | 0.1495 | 0.1510 | 0.1550 | +1.00 | **sí** |
| `batch4-r62` | 0.1015 | 0.1025 | 0.1055 | 0.1120 | 0.1415 | +1.00 | **sí** |
| `r14` | 0.0930 | 0.0980 | 0.1015 | 0.1165 | 0.1420 | +1.00 | **sí** |
| `r17` | 0.1080 | 0.1100 | 0.1130 | 0.1250 | 0.1430 | +1.00 | **sí** |
| `r19` | 0.0900 | 0.0900 | 0.0970 | 0.1085 | 0.1370 | +0.97 | **sí** |
| `r20` | 0.0850 | 0.0885 | 0.0925 | 0.1025 | 0.1380 | +1.00 | **sí** |
| `r28` | 0.0785 | 0.0835 | 0.0850 | 0.1050 | 0.1370 | +1.00 | **sí** |
| `r39` | 0.0765 | 0.0785 | 0.0815 | 0.1000 | 0.1430 | +1.00 | **sí** |
| **media** | **0.1059** | **0.1080** | **0.1111** | **0.1211** | **0.1452** | | |

| prueba | valor |
|---|---|
| grafos con ρ intra-grafo > 0 | **12 / 12** (media de los ρ = **+0.965**) |
| grafos con la masa **estrictamente monótona creciente** | **10 / 12** |
| Friedman (bloques = grafos, 5 escalones) | χ²=44.94, **p = 4.1e-09** |
| **Page L (tendencia creciente e0≤e1≤e2≤e3≤e4)** | L=655.5, z=6.67, **p = 1.3e-11** |
| signos e4 vs e0 | **12 / 12**, p = 4.9e-04 |
| Wilcoxon e4 vs e0 | p = 4.9e-04 |
| Spearman **global centrado por grafo** (pareado) | **ρ = +0.960**, p = 7.7e-34 (n=60) |
| Spearman global **crudo** (mezcla entre y dentro de grafos) | ρ = +0.381, p = 0.0027 (n=60) |

**La monotonía se cumple grafo por grafo, no sólo en promedio**: 10 de 12 grafos son estrictamente
crecientes en los 5 escalones y los 2 que no lo son (`batch3-r0`, `batch4-r10`) fallan por un solo
peldaño y por −0.0040 y −0.0015 respectivamente — del orden del piso de ruido de réplica medido en O3-B
(máximo 0.0035).

### 6.1 — La respuesta es escalonada y **convexa**: casi todo pasa arriba

| tramo | Δ masa medio | gana | Wilcoxon |
|---|---|---|---|
| e0 → e1 | +0.00208 | 10/12 | p = 0.0068 |
| e1 → e2 | +0.00312 | 11/12 | p = 0.0195 |
| e2 → e3 | +0.01004 | **12/12** | p = 4.9e-04 |
| e3 → e4 | +0.02412 | **12/12** | p = 4.9e-04 |
| **e0 → e4** | **+0.03937** (**+37.2%** relativo) | **12/12** | p = 4.9e-04 |

Cada peldaño da más que el anterior. En el tramo bajo —el único que roza el rango de clustering que
estas reglas producen espontáneamente— la respuesta existe pero es chica (+2% relativo, del mismo orden
que el +5% de O3-B). El +37% del extremo corresponde a un clustering 10–20 veces mayor que cualquiera
que aparezca solo.

### 6.2 — Observables secundarios

| observable | e0 | e1 | e2 | e3 | e4 | tendencia |
|---|---|---|---|---|---|---|
| **κ_V agregado** | 0.684 | 0.692 | 0.717 | 0.852 | 1.208 | Page z=5.54, **p=1.5e-08**; e4>e0 en **12/12** |
| nº de sumideros | 8.00 | 8.00 | 8.00 | 8.08 | 8.58 | prácticamente constante hasta e3 |
| t del primer sumidero | 0.0391 | 0.0357 | 0.0352 | 0.0339 | 0.0293 | **baja** en 12/12 (p=9.8e-04) |

**No se forman más grumos: cada grumo come más, y empieza antes.** Es el mismo patrón que O3-B (§5.1 de
aquel informe: "la diferencia no está en cuántos grumos se forman sino en cuánto come cada grumo"),
ahora con la perilla movida a mano.

---

## 7. ¿Por qué vía? Descontando las tres covariables

| relación (Spearman sobre valores **centrados por grafo**, n=60) | ρ | p |
|---|---|---|
| clustering ↔ masa (la relación de interés) | **+0.960** | 7.7e-34 |
| clustering ↔ **pendiente corregida** | +0.578 | 1.3e-06 |
| pendiente corregida ↔ masa | +0.545 | 6.8e-06 |
| **parcial clustering–masa descontando la pendiente** | **+0.943** | 2.0e-29 |
| clustering ↔ **asortatividad** | +0.801 | 1.5e-14 |
| asortatividad ↔ masa | +0.743 | 1.1e-11 |
| **parcial clustering–masa descontando la asortatividad** | **+0.911** | 5.0e-24 |
| clustering ↔ **componente gigante** | −0.661 | 9.4e-09 |
| componente gigante ↔ masa | −0.623 | 1.1e-07 |
| **parcial clustering–masa descontando la gigante** | **+0.934** | 1.1e-27 |

Las tres covariables se mueven y las tres correlacionan con la masa — pero **ninguna de las tres se
lleva la relación**: descontando cualquiera de ellas la asociación clustering–masa baja como mucho de
0.960 a 0.911. Y la pendiente, que es el observable con el que veníamos midiendo "geometría extensa",
**sube en promedio pero es monótona en sólo 1 de 12 grafos**, mientras que la masa es monótona en 10 de
12. Dicho en simple: **el clustering empuja un poco la geometría, pero no es por ahí que empuja la
masa.** Coincide con lo que ya decía O3-B (§3.3, §6): la pendiente no es un buen resumen de
"organización relacional" para esta familia de grafos.

---

## 8. El detalle incómodo: el original le gana a su propio escalón

El grafo original sin tocar corrió también, como referencia (no es un peldaño: tiene otra historia de
recableo). Comparado contra la escalera de su propio grafo:

| comparación | Δ masa medio | gana el original | Wilcoxon |
|---|---|---|---|
| original vs **e0** (cero triángulos) | **+0.00654** | **12 / 12** | p = 4.9e-04 |
| original vs **e1** (clustering ≥ el natural en 10/12) | **+0.00446** | 9 / 12 | p = 0.0137 |

La primera fila **replica O3-B con un contraste más duro y un resultado más limpio**: allá el original
le ganaba a su gemelo barajado a ciegas por +0.00533 en 9 de 12 (Wilcoxon p=0.0103); acá le gana a un
gemelo con **cero** triángulos por +0.00654 en **12 de 12**.

La segunda fila es la que hay que mirar con cuidado: **e1 tiene más clustering que el original en 10 de
los 12 grafos, y aun así el original acreta más.** Es decir: el clustering, por sí solo y fabricado a
mano, **no reproduce** lo que el grafo nativo tiene. Mover la perilla mueve la masa —eso quedó
mostrado— pero la perilla no es toda la historia: en el rango natural queda un margen de ~+0.004–0.007
que el número de triángulos no explica. En el gráfico (panel izquierdo del PNG) esto se ve como las
cruces (×) sentadas **por encima** de la curva de su propia escalera en la zona de clustering bajo.

---

## 9. Qué NO dice este resultado

- **No dice que la respuesta natural sea del 37%.** Ese número corresponde a un clustering de 0.24–0.60,
  entre 11 y 480 veces el que estos grafos tienen espontáneamente. En el tramo que sí roza el rango
  natural (e0→e1) la respuesta es **+0.002 (+2%)**, del orden del efecto de O3-B.
- **No dice que el clustering sea la única causa.** §8 muestra que a clustering igual o mayor el grafo
  nativo sigue ganando. Lo que quedó mostrado es que el clustering es una **palanca suficiente** para
  mover la masa, no que sea la palanca completa.
- **No aísla al clustering de todo lo demás.** Subir triángulos con los grados clavados arrastra
  necesariamente otras propiedades: asortatividad (ρ=+0.80 con el clustering), componente gigante
  (ρ=−0.66) y en menor medida la pendiente (ρ=+0.58). Las correlaciones parciales de §7 las descuentan
  de a una, no todas juntas, y con n=60 puntos que vienen de 12 grafos.
- **Los 12 grafos no son 12 muestras independientes en el sentido fuerte** — vienen de 4 `seed_base`
  (3 por lote), la limitación que ya señalaba `FASE6_O2F_N_efectivo_fase5b_CS.md`.
- **El techo de la escalera está puesto por una decisión de diseño** (conservar ≥97% de la componente
  gigante). Sin ese freno el clustering llega más alto pero el grafo se rompe en camarillas, y eso ya
  no sería "la misma maqueta con más triangulitos".
- **Un solo layout por grafo.** Cada escalón se corrió con `seed_layout=12345`; no hay repeticiones de
  layout que permitan separar el ruido del Fruchterman-Reingold dentro de un mismo escalón. El diseño
  pareado y la monotonía 10/12 lo compensan, pero no lo reemplazan.
- **No se declara cierre.** La lectura de hasta dónde llega esta evidencia es de Alexis.

---

## 10. Lo que quedó fabricado y disponible sin gastar cómputo nuevo

- **72 corridas de Phantom completas** en `/Users/alexis/phantom_cs073/bateria_fase7_f702_escalera/`
  (dumps, `.sink`, `run.log`, `meta_regla.json` de cada una).
- El script acepta `--sin-orig`, índices de grafo y `n_por_lote`, así que la escalera se puede extender
  a los 27 grafos Clase III restantes del universo de 40 sin tocar nada.
- `FRACCIONES_OBJETIVO` y `FRAC_GIGANTE_MINIMA` son parámetros: se puede densificar la escalera en el
  **tramo bajo** (que es donde vive el clustering natural y donde este experimento tiene menos
  resolución: un solo peldaño entre C=0 y C≈0.03) sin rehacer nada del resto.
- La escalera guarda internamente ~40 fotos por grafo sobre una rejilla fina de nº de triángulos; hoy se
  usan 5. Cambiar cuáles se usan no requiere volver a correr los swaps.

**Costo real medido:** construir la escalera de un grafo (bajar + subir + elegir escalones + medir los
5 escalones con pendiente corregida) ~15 s; generar cada condición inicial 51–124 s según la carga de la
máquina (4 shards en paralelo, con otros agentes corriendo Phantom al mismo tiempo) → ~35 min para las
72; Phantom 7–14 s por corrida → ~13 min; análisis <1 min.

### Nota operativa (por si le sirve a otra tarea de la línea)

Correr dos runners de Phantom en paralelo sobre la MISMA carpeta rompe la corrida: los dos ejecutan
`phantomsetup` y el segundo encuentra el `cosmog.in` ya editado. Pasó una vez acá
(`batch3-r111_..._e4`, quedó con 502 volcados en vez de 501); se detectó comparando el número de
volcados de las 72 carpetas, se borró y se rehizo esa corrida sola desde cero. **El chequeo "501
volcados en las 72 carpetas" conviene hacerlo siempre antes de analizar.**

