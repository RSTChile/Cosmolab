# CS086 — ¿SUENA distinto el tejido bajo renormalización o bajo poda? Espectro del laplaciano aplicado a Fase III

**Fecha:** 9-ago-2026 · **Codea/ejecuta:** CC (Claude) · **Script:** `cs086_espectro_renorm_poda.py`.
No se toca ningún script congelado (`cs080_renormalizacion.py`, `cs081_poda_dinamica.py`,
`cs084_espectro_laplaciano.py`, `cs066_localidad_geometrogenesis.py`) — todo es import. **No se declara
cierre ni veredicto — sólo números. La lectura final es de Alexis.**

## 0. La pregunta, en simple

CS084 ya mostró que se puede "escuchar" un grafo: el espectro del laplaciano (sus valores propios) es
como la lista de tonos de un tambor. Fase III (CS080/CS081) hizo dos cosas al tejido de CS066 —
alejarse la vista agrupando casas en manzanas (renormalización) y podar las conexiones más "caras"
(poda) — y las juzgó con **un solo número**: qué tan rápido crece el diámetro (la distancia máxima
entre dos puntos) cuando se agranda la escala. Ese único número no vio diferencia entre el tejido real y
sus controles bajo agrupamiento, y vio una diferencia sólo modesta entre podar-por-costo y podar-al-azar.

La pregunta de hoy: si en vez de mirar sólo el diámetro (un número) escuchamos el **tambor completo**
(todos los tonos), ¿aparece una diferencia que el diámetro no pudo ver?

## 1. Qué se corrió

- **N=8000, k_local=6** — exactamente los mismos parámetros de Fase III.
- **Mismas 3 semillas de Fase III** (80100, 80200, 80300) — para comparar manzana con manzana con los
  informes ya escritos.
- **Parte A (renormalización):** escalas b=1, 4, 16 (subconjunto de las b=1,2,4,8,16,32 de CS080, para
  acotar el costo de la diagonalización densa — ver nota de presupuesto abajo), brazos `local` (real),
  `local_barajado` (NULL 1: mismo tope de grado, sin criterio de localidad), `er_null` (NULL 2: piso
  Erdős-Rényi).
- **Parte B (poda):** tejido sin agrupar (b=1), variantes `sin_poda`, `costo_P50`, `azar_P50` (P50 es
  donde Fase III vio la brecha costo-vs-azar más grande en el diámetro).
- **Reducciones de presupuesto, documentadas:** 3 semillas (no 5, como en CS084) y sólo el percentil 50
  en la poda (no 50/70/90 como en CS081) — el costo dominante es siempre la escala b=1 (N=8000,
  diagonalización densa ~20-40s por matriz en esta corrida); b=4 y b=16 son mucho más baratos porque
  N_b baja a ~2900 y ~600-1500. Corrida completa: **32.6 minutos** (Parte A: 18.1 min, Parte B: 14.5 min).
- **Caveat heredado (no introducido acá):** `construir_sustrato` y `proceso066_instrumentado` (de
  CS080/CS081, sin tocar) usan `hash()` de Python para mezclar semillas, y ese hash no es
  reproducible bit-a-bit entre sesiones sin fijar `PYTHONHASHSEED` — esto ya era así en las corridas
  originales de Fase III y en CS084; significa que el tejido de esta corrida es "de la misma familia"
  que el de los informes anteriores (mismo motor, mismas semillas base) pero no un clon exacto,
  nodo-por-nodo, de esos grafos específicos.

Salidas: `cs086_espectro_renorm.csv` (27 filas: 3 arms × 3 semillas × 3 escalas) y
`cs086_espectro_poda.csv` (9 filas: 3 variantes × 3 semillas).

---

## 2. Parte A — espectro bajo renormalización

### Tabla (media de 3 semillas; entre corchetes, rango semilla-a-semilla)

| b | brazo | λ_max | dispersión (std eig) | λ₂ (conectividad algebraica) | d_s(t=1.0) | espaciado KS vs GOE |
|---|---|---|---|---|---|---|
| 1 | **local (real)** | 389.4 [282–530] | 11.28 [10.5–12.3] | 0.295 [0.27–0.34] | 0.649 | 0.013 |
| 1 | local_barajado (NULL 1) | 70.1 [55–83] | 4.38 [4.26–4.51] | 0.950 [0.85–1.0] | 0.537 | 0.013 |
| 1 | er_null (NULL 2) | 19.4 [18.7–20.5] | 3.45 [3.42–3.46] | 0.316 [0.24–0.40] | 3.590 | 0.014 |
| 4 | **local (real)** | 292.7 [271–311] | 12.81 | 1.07 | 0.138 | 0.021 |
| 4 | local_barajado | 81.6 [63–94] | 11.31 | 1.90 | 0.024 | 0.046 |
| 4 | er_null | 39.6 [39.2–40.0] | 8.82 | 0.88 | 2.637 | 0.072 |
| 16 | **local (real)** | 163.2 [135–184] | 15.77 [14.8–16.3] | 0.907 | 0.076 | 0.157 |
| 16 | local_barajado | 116.3 [110–128] | 31.62 [31.3–31.9] | 2.30* | 0.012 | 0.291 |
| 16 | er_null | 99.8 [97–101] | 30.84 [30.8–30.9] | 0.951 | 2.461 | 0.362 |

*local_barajado λ₂ en b=16 tiene una semilla atípica (0.99 vs 2.95/2.95 en las otras dos) — más ruido
semilla-a-semilla en esta celda que en el resto de la tabla.

### Lectura, número por número

**1. λ_max y la dispersión del espectro SÍ separan limpio real de los dos NULL, en las tres escalas,
sin solapamiento entre semillas.** En b=1, λ_max del tejido real (389) es ~5.5× el de local_barajado
(70) y ~20× el de er_null (19); el rango de semillas de cada brazo (282–530, 55–83, 18.7–20.5) no se
toca entre brazos — separación limpia, no un efecto de una semilla ruidosa. Esa brecha se achica al
agrupar pero el ORDEN se mantiene intacto hasta b=16 (163 > 116 > 100, otra vez sin solapamiento de
rangos). Esto es algo que el diámetro (Fase III Exp.1) **no vio en ninguna escala** — la pendiente
diam-vs-N_b fue indistinguible entre los tres brazos. λ_max está dominado por los nodos más conectados
del grafo (algo así como el pico más agudo del tambor, producido por los "hubs"); el tejido real de
CS066 (con su paso de gravedad) genera hubs mucho más extremos que sus controles, y esa asimetría
sobrevive al agrupamiento.

**2. La conectividad algebraica (λ₂) distingue algo distinto: local_barajado, no local, es el raro.**
En b=1, local (0.295) y er_null (0.316) están casi pegados, mientras que local_barajado (0.950) es
~3× más alto — es decir, barajar los enlaces (mismo tope de grado, sin criterio de localidad) produce un
grafo con MENOS cuellos de botella que el tejido real construido por localidad, y también menos que un
Erdős-Rényi puro. Es un hallazgo que el diámetro no aisló con esta claridad: la localidad, específicamente,
introduce cuellos de botella que ni el azar con el mismo grado ni el ruido puro producen.

**3. La dimensión espectral d_s(t=1.0) reagrupa los brazos de otra manera: real y barajado juntos,
Erdős-Rényi aparte.** En b=1, local (0.65) y local_barajado (0.54) están cerca entre sí y muy por
debajo de er_null (3.59) — a esa escala de difusión, ambos brazos "con estructura" (aunque uno tenga
localidad real y el otro no) se comportan casi-1D, mientras que el ruido puro se ve casi-4D. Esto
CONFIRMA, por un camino completamente independiente (difusión/núcleo de calor en vez de crecimiento de
bola), el hallazgo ya registrado en `FASE3_renormalizacion_resultado_CS.md`: la estructura se
"desarma" (fragmenta, colapsa en dimensión) más rápido que el ruido al agruparla — en b=16, d_s cae a
0.08 (local) y 0.01 (barajado), casi disconexo, mientras que er_null se mantiene en 2.46.

**4. El espaciado de niveles (Poisson vs GOE) NO distingue nada en b=1 — pero SÍ empieza a distinguir
en b=16.** En b=1, los tres brazos tienen un ajuste casi perfecto a GOE (KS≈0.013-0.014 en los tres,
prácticamente iguales) — la estadística fina del espaciado entre eigenvalues vecinos es "universal" a
esa escala, no importa si el grafo tiene localidad real o no. Pero en b=16, con menos nodos (~600-900
tras recortar), el ajuste a GOE se degrada de forma DIFERENTE por brazo: local se aleja menos de GOE
(KS=0.157) que local_barajado (0.291) y mucho menos que er_null (0.362) — es decir, a la escala más
agrupada, el tejido real conserva MÁS estructura de tipo "repulsión de niveles" (anti-Poisson, anti-
ruido) que sus dos controles. Es un hallazgo nuevo, en la dirección opuesta a lo que uno esperaría de
"el tejido real se desarma más rápido" (hallazgo #3) — el desarme en DIMENSIÓN y la preservación de
ESTRUCTURA DE NIVELES son diagnósticos distintos y no tienen por qué apuntar en la misma dirección.

**Analogía:** pensá el tejido real como un instrumento con algunas cuerdas mucho más gruesas que las
demás (los "hubs"), mientras que sus controles tienen cuerdas más parejas. Aun cuando alejás el oído
(agrupás en manzanas), esas cuerdas gruesas siguen sonando distinto — eso es λ_max. Pero si en cambio
medís "qué tan fácil es cortar el instrumento en dos mitades sin tocar casi ninguna cuerda" (λ₂), el
raro no es el instrumento real, es el que mezcló las cuerdas al azar. Y si medís el timbre fino
(espaciado entre notas vecinas), a corta distancia los tres instrumentos suenan igual de "afinados"
(estructura tipo GOE), pero al alejarte mucho el instrumento real se desafina MENOS que los controles.

---

## 3. Parte B — espectro bajo poda

### Tabla (media de 3 semillas; entre corchetes, rango semilla-a-semilla) — todo en b=1, N=8000

| variante | enlaces podados | λ₂ | componentes | giant | espaciado KS vs GOE | espaciado KS vs Poisson | d_s(t=1.0) |
|---|---|---|---|---|---|---|---|
| sin_poda | 0 | 0.295 [0.27–0.34] | 659 | 0.912 | 0.013 [0.011–0.017] | 0.211 | 0.649 |
| **costo_P50** | ~14 900 (62%) | **0.019** [0.016–0.024] | 819 | 0.890 | **0.055** [0.047–0.061] | 0.167 | 1.064 |
| azar_P50 | ~14 900 (62%) | 0.095 [0.079–0.118] | 734 | 0.904 | 0.023 [0.020–0.026] | 0.199 | 1.317 |

*(recordatorio de la métrica ya publicada, para contexto: pendiente diam-vs-N_b bajo el mismo
coarse-graining — sin_poda=0.421, azar_P50=0.655, costo_P50=0.786, `FASE3_poda_dinamica_resultado_CS.md`.)*

### Lectura, número por número

**1. λ₂ (conectividad algebraica) separa costo de azar MUCHO más nítido que el diámetro, y sin
solapamiento de semillas.** Podar por costo deja λ₂≈0.019, ~5× MÁS BAJO que podar al azar la misma
cantidad de enlaces (λ₂≈0.095) — y los rangos por semilla no se tocan (costo: 0.016–0.024; azar:
0.079–0.118). El diámetro ya había mostrado que costo > azar (pendiente 0.786 vs 0.655, una brecha de
~20%), pero λ₂ muestra una brecha de ~5× — el mismo fenómeno, visto con una lupa mucho más potente. λ₂
mide directamente "qué tan fácil es cortar el grafo en dos mitades grandes tocando pocos enlaces" (el
llamado valor de Fiedler); que la poda por costo lo hunda tanto más que la poda al azar dice que el
criterio de costo está encontrando, de forma consistente, los puentes/cuellos de botella reales del
tejido — no simplemente adelgazándolo de forma pareja.

**2. El número de componentes conexas confirma la misma historia por otro lado.** Podar por costo deja
819 componentes en promedio (más fragmentado) contra 734 de podar al azar, con sin_poda en 659 — cortar
por costo fragmenta más el tejido que cortar la misma cantidad al azar, consistente con que está
cortando específicamente los enlaces "puente" que mantenían unidas piezas distintas.

**3. El espaciado de niveles (Poisson vs GOE) también separa costo de azar, y en una dirección que
matiza la lectura de "costo descubre geometría real".** Podar por costo empuja el espectro MÁS LEJOS de
GOE (KS=0.055) que podar al azar (KS=0.023) — sin solapamiento entre semillas — y al mismo tiempo lo
acerca más a Poisson (KS_poisson baja de 0.211 en sin_poda a 0.167 en costo_P50, mientras que azar_P50
se queda cerca de sin_poda en 0.199). En el lenguaje anti-Shannon del proyecto (Poisson=ruido sin
correlación, GOE=estructura con correlación), esto dice que podar por costo no sólo "encuentra" más
distancia geométrica (λ₂, diámetro) — también degrada más la correlación fina del espectro hacia algo
más parecido a ruido puro, comparado con podar al azar la misma cantidad. Es un hallazgo con dos caras
que conviene reportar sin resolver la tensión: costo aísla mejor los cuellos de botella (λ₂, diámetro,
fragmentación) PERO también "desordena" más la estructura fina de niveles — no es evidencia limpia de
"más orden geométrico" en todos los diagnósticos a la vez.

**4. La dimensión espectral d_s(t=1.0) sube con cualquier poda, y sube MÁS con azar que con costo** —
0.65 (sin podar) → 1.06 (costo) → 1.32 (azar). Este diagnóstico, solo, apuntaría en la dirección
contraria a los otros tres (azar "más dimensional" que costo a esta escala de difusión) — otra señal de
que distintos diagnósticos espectrales del mismo grafo pueden no estar de acuerdo entre sí, y que "el
espectro" no es un solo veredicto sino varios ángulos que hay que mirar juntos.

**Analogía:** imaginá una ciudad con demasiados atajos, y dos formas de sacar la mitad de las calles.
Sacarlas al azar reduce un poco el efecto atajo, pero deja la ciudad todavía razonablemente fácil de
cruzar de punta a punta (λ₂ alto). Sacar las calles "sospechosas" por su historial (costo) deja la
ciudad mucho más fácil de partir en dos con pocos cortes (λ₂ mucho más bajo) — encontró los puentes
reales. Pero esa misma cirugía también deja el tránsito local más "errático" en su textura fina (más
parecido a ruido que a un patrón ordenado) que sacar calles al azar — como si, al enfocarse tan bien en
los puentes, también desordenara algo del pulso fino del tráfico que la poda al azar no toca tanto.

---

## 4. Resumen para Alexis (en simple, sin cerrar nada)

**Parte A (renormalización):** el espectro SÍ vio cosas que el diámetro no vio. λ_max separa limpio real
de sus dos controles en las tres escalas probadas (b=1,4,16), sin que se toquen los rangos entre
semillas — algo que la pendiente del diámetro no logró en ninguna escala. Pero distintos diagnósticos
espectrales apuntan en direcciones distintas: λ₂ dice que el "raro" es local_barajado (no el real); d_s
por difusión confirma (por un camino independiente) que la estructura real se desarma más rápido que el
ruido al agrupar; y el espaciado de niveles, que no distinguía nada en b=1, empieza a distinguir en
b=16 — con el tejido real preservando MÁS estructura fina que sus controles a esa escala agrupada.

**Parte B (poda):** el espectro separa costo de azar mucho más nítido que el diámetro — λ₂ cae ~5× con
poda-por-costo contra la poda-al-azar (el diámetro sólo había mostrado una brecha de pendiente de
~20%), sin solapamiento entre semillas, y el número de componentes/fragmentación va en la misma
dirección. Pero el espaciado de niveles cuenta una historia con matiz: podar por costo aleja MÁS el
espectro de la estructura tipo GOE (lo acerca más a ruido puro) que podar al azar — el criterio de
costo encuentra mejor los cuellos de botella geométricos, pero a la vez "desordena" más la textura fina
del espectro que una poda ciega de la misma magnitud.

Ningún diagnóstico acá se declara ganador ni cierre de arco — quedan los números completos en
`cs086_espectro_renorm.csv` y `cs086_espectro_poda.csv` para auditoría directa.

## 5. Archivos

- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs086_espectro_renorm_poda.py` — código del
  experimento (Parte A + Parte B), no toca ningún script congelado.
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs086_espectro_renorm.csv` — datos crudos Parte A
  (27 filas: 3 arms × 3 semillas × 3 escalas).
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs086_espectro_poda.csv` — datos crudos Parte B
  (9 filas: 3 variantes × 3 semillas).
