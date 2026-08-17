# NULL-4 (topología completa idéntica a REAL, orden temporal de formación destruido) — Fase II CS073, escalón 5 de 6

**Encargo:** completar el escalón NULL-4 de la jerarquía de 6 controles — a diferencia de NULL-1/2/3
(que cambian la TOPOLOGÍA final: ángulo, espectro de potencia, motivos/ciclos), NULL-4 conserva la
topología COMPLETA de la malla causal REAL (ningún enlace cambia) y en cambio rebaraja el ORDEN en que
esas aristas se insertaron antes de correr `layout_resortes` — la pregunta es si la HISTORIA de cómo se
construyó la malla (no sólo su forma final como conjunto) importa para que nazcan sumideros. **No se
declara cierre ni veredicto sobre CS073 ni sobre la jerarquía — sólo se reportan números. La lectura es
de Alexis.**

---

## Paso 0 — Verificación previa de que NULL-4 es operacionalizable (`null4_verificar_invarianza_orden.py`, ya en disco, no tocado en esta tarea)

Antes de gastar cómputo Phantom había que descartar que el pipeline colapsara NULL-4 a ser un no-op:
`layout_resortes` calcula fuerzas de forma vectorizada (numpy, suma conmutativa), así que en principio el
orden de inserción de las aristas en el `dict` de adyacencia podría no afectar nada más allá del ruido de
punto flotante. Se verificó empíricamente reconstruyendo la malla causal REAL (N=2000) y corriendo
`layout_resortes` (mismo `seed_layout=12345`) sobre 4 órdenes de inserción distintos del MISMO conjunto
de aristas (natural, invertido, 2 barajados al azar):

```
Peor diferencia absoluta entre órdenes: 2.413e+00 (razón sobre escala típica de coordenada: 0.383 — NO trivial)
```

**Resultado: las posiciones SÍ difieren de forma no trivial entre órdenes de inserción** — 38% de la
escala típica de coordenada en el peor caso. Esto confirma que NULL-4 es operacionalizable (produce una
condición inicial genuinamente distinta de REAL, no idéntica por construcción).

**Caveat de mecanismo — abierto, no resuelto aquí:** el motivo de por qué el orden importa NO está
caracterizado. Dos lecturas posibles, ambas compatibles con el número de arriba:
1. **Sensibilidad dinámica genuina** del proceso de relajación de Fruchterman-Reingold — apoyaría la
   idea de que la malla tiene algo parecido a "memoria de su propia historia de formación".
2. **Artefacto de implementación:** `layout_resortes(seed=...)` siembra las posiciones iniciales de cada
   nodo consumiendo una secuencia de números aleatorios en el orden en que ese nodo aparece durante la
   construcción de `adj`/`edges` — si el orden de inserción cambia, cambia qué número de la secuencia
   recibe cada nodo como semilla de posición inicial, sin que eso implique ninguna "dinámica" interesante
   más allá de "empezaste de un lugar distinto".

No se investigó cuál de las dos explica el 38% observado — **esta ambigüedad queda anotada
explícitamente para que el resultado de NULL-4 no se lea como más fuerte de lo que es** (ver "Lectura de
los números" al final).

---

## Método

- **NULL-4 (`null4_generar_ic.py`, módulo nuevo):** reconstruye la malla causal REAL exacta
  (`malla_causal_atomos`, D=3/k=4/`seed_ejes=2000`, MISMOS parámetros que `traducir_pool` — el mismo
  conjunto de 4945 aristas que ve REAL, ninguna cambia), extrae la lista canónica de aristas (reusa
  `adj_a_lista_aristas` de `null4_verificar_invarianza_orden.py`), la rebaraja con
  `np.random.default_rng(seed_reorden).permutation(...)` y reconstruye el `dict` de adyacencia en ese
  orden (reusa `construir_adj_en_orden`, misma función que ya validó Paso 0). Un `assert` compara la
  lista de aristas ordenada de NULL-4 contra la de REAL antes de continuar — topología idéntica
  verificada por conjunto, no sólo por conteo.
- **Layout físico:** `layout_resortes` (Fruchterman-Reingold), MISMA función/parámetros que toda la
  jerarquía (`iters=100`, `seed_layout=12345`), seguida de la MISMA dilatación isótropa estática
  (`Expansion`, `n_pasos_expansion=60`) y el MISMO campo de velocidad turbulento (Mach=3, seed=42) que
  REAL/NULL-1/NULL-2/NULL-3 — así NULL-4 sólo difiere de REAL en la única variable que se quiere aislar
  (orden de formación), ninguna otra.
- **Phantom:** binarios `_backup` (sin APR, misma build que toda la jerarquía). Configuración física
  copiada literal de `bateria_n2000/ic_real/cosmog.in` (`icreate_sinks=1`, `rho_crit_cgs=1000`,
  `r_crit=0.6`, `h_acc=0.3`, `tmax=0.500`, `dtmax=0.001`).
  N=2000 directo (sin piloto N=500 — ya sabido que falla por resolución, y Paso 0 ya cumplió el papel
  del piloto: confirmar que el mecanismo produce diferencia no trivial antes de gastar cómputo Phantom).
- **Observable:** masa total acumulada en sumideros al final — el mismo de toda la jerarquía CS073.
- **Estadístico:** test de permutación EXACTO de 2 muestras independientes (`test_permutacion_2_grupos`,
  importada tal cual de `null3_bateria_comparar.py`, no reimplementada), H1 de una cola pre-registrada,
  enumerando TODAS las C(n_a+n_b, n_a) asignaciones posibles bajo H0.

---

## Batería (N=2000, 3 semillas de reordenamiento 601–603, `null4_bateria_generar.py` + `_correr.py` + `_comparar.py`)

Generación de las 3 IC: 158.9 s (≈53 s/semilla, dominado por las 100 iteraciones O(n²) de
`layout_resortes` a N=2000 — mismo costo que REAL/NULL-1/2/3, ya que la topología/tamaño es idéntica).
Las 3 corridas Phantom: **40.7 s total, 3/3 exit code 0** en `setup` y en `phantom`, sin abortos de
conservación. Las 3 confirmaron 4945 aristas (idéntico a REAL) antes de correr.

| corrida | masa en sumideros | nº sumideros |
|---|---|---|
| NULL-4 seed 601 | 2105.6 | 8 |
| NULL-4 seed 602 | 2171.4 | 9 |
| NULL-4 seed 603 | 2133.8 | 8 |
| **media / DE** | **2136.93 / 33.01** | 3 (las 3) |

Comparación de referencia (todas ya en disco, sólo lectura):
- **REAL** (n=6, `bateria_n2000/`+`bateria_real_extra_n2000/`): media=2196.47, DE=95.98, rango
  2049.2–2293.6, 8 sumideros en las 6.
- **NULL-1** (ángulo isótropo aleatorio, n=8) y **NULL-2** (Zel'dovich, n=8): **0 sumideros, masa 0** en
  las 16 corridas combinadas.
- **NULL-3** (double-edge-swap + filtro de longitud, ~87.5% de aristas preservadas, n=8): media=2186.68,
  DE=53.16, rango 2068.0–2246.6, 8 sumideros en las 8.

**Las 3 corridas NULL-4 formaron sumideros en las 3, sin excepción**, con masas totales (2105.6–2171.4)
dentro del rango inferior de REAL (2049.2–2293.6) y de NULL-3 (2068.0–2246.6) — igual que NULL-3 y muy
distinto de NULL-1/NULL-2 (cero absoluto).

---

## Estadísticos de separación

### (a) REAL (n=6) vs NULL-4 (n=3) — test de permutación exacto
- estadístico observado (media_REAL − media_NULL4) = **59.53**
- C(9,6) = 84 asignaciones posibles bajo H0
- rank de la asignación observada = 15 de 84
- **p (una cola, REAL>NULL-4) = 0.1786** — p (dos colas) = 0.3214
- z-score = (media_REAL − media_NULL4) / DE_NULL4 = **1.803**

Con sólo n=3 en NULL-4 el piso teórico de p es 1/84≈0.012 — la resolución del test es más gruesa que en
NULL-1/2/3 (n=8). Aun así, no hay evidencia de separación fuerte: la diferencia de medias (59.5 sobre
~2190) es comparable en magnitud a la que hay entre REAL y NULL-3 (9.8) y menor que la dispersión interna
de REAL (DE=96.0).

### (b) NULL-4 (n=3) vs NULL-1 (n=8)
- estadístico observado = **2136.93**, C(11,3)=165, rank=1 de 165
- **p (una cola, NULL-4>NULL-1) = 1/165 ≈ 0.00606** — piso teórico exacto de este diseño.

### (c) NULL-4 (n=3) vs NULL-2 (n=8)
- estadístico observado = **2136.93** (idéntico a (b) porque NULL-2 también dio masa 0 exacta),
  C(11,3)=165, rank=1 de 165
- **p (una cola, NULL-4>NULL-2) = 1/165 ≈ 0.00606** — piso teórico exacto de este diseño.

### (d) NULL-4 (n=3) vs NULL-3 (n=8)
- estadístico observado (media_NULL4 − media_NULL3) = **−49.74**
- C(11,3)=165, rank=151 de 165
- **p (una cola, NULL-4>NULL-3) = 0.9152** — p (dos colas) = 0.1455

NULL-4 y NULL-3 no se distinguen entre sí con este diseño (si acaso, NULL-4 quedó levemente por debajo de
NULL-3 en esta muestra pequeña, sin significancia).

### (e) referencia: REAL (n=6) vs NULL-3 (n=8)
- estadístico observado = 9.79, p (una cola) = 0.4212 — repetido del informe NULL-3 para contexto lado a
  lado en este mismo documento.

---

## Lectura de los números (sin cerrar nada)

Con el observable y diseño de esta jerarquía: **NULL-4 (topología COMPLETA idéntica a REAL — 100% de las
aristas, no ~87.5% como NULL-3 —, sólo el orden de inserción rebarajado) formó sumideros en las 3
corridas, con masas (2105.6–2171.4) que caen en el mismo orden de magnitud que REAL (2049.2–2293.6) y
que NULL-3 (2068.0–2246.6), y ninguna comparación REAL-vs-NULL-4 ni NULL-4-vs-NULL-3 alcanzó
significancia (p=0.18 y p=0.92 respectivamente).** Dentro del panorama de la jerarquía hasta ahora, NULL-4
se ubica **junto a NULL-3 y REAL** (formación robusta de sumideros, masas indistinguibles entre sí dentro
del ruido de semilla), lejos del "cero absoluto" de NULL-1 y NULL-2. Es decir: en este diseño, rebarajar
sólo el ORDEN de formación (dejando la topología final intacta) no bastó para colapsar la formación de
sumideros — a diferencia de romper la topología misma (NULL-1: ángulo; NULL-2: espectro de potencia),
que sí la colapsa a cero.

Dos advertencias explícitas sobre el alcance de esta lectura:
1. **n=3 en NULL-4** (vs n=8 en NULL-1/2/3) — por la salvaguarda de tiempo del encargo se corrieron 3
   semillas de reordenamiento, no 8. La resolución estadística de (a) y (d) es más gruesa que la del
   resto de la jerarquía; no se puede descartar que una batería de 8 semillas revele una separación que
   3 no alcanzan a mostrar.
2. **El caveat de mecanismo de Paso 0 sigue sin resolver**: no se sabe si la dependencia de orden que
   hace operacionalizable a NULL-4 es sensibilidad dinámica genuina del proceso de relajación, o un
   artefacto de cómo `layout_resortes(seed=...)` reparte números aleatorios de posición inicial según el
   orden de inserción. Si es lo segundo, "el orden de formación no bastó para romper nada" es un
   resultado más débil de lo que suena — sería más parecido a "una realización distinta del mismo
   proceso estocástico" que a "una historia causal distinta". Esta tarea no intentó distinguir entre las
   dos lecturas.

Ningún veredicto sobre CS073 ni sobre esta jerarquía se declara aquí — los números de arriba son el
entregable; la interpretación final es de Alexis.

---

## Tiempo de cómputo real vs. salvaguarda

Salvaguarda pedida: ~45-55 min totales para esta tarea.

| paso | tiempo |
|---|---|
| Paso 0 (ya corrido antes de esta tarea, sólo lectura del resultado) | — |
| Generación de las 3 IC (N=2000) | 158.9 s |
| Batería: Phantom 3 corridas | 40.7 s |
| Comparación (`null4_bateria_comparar.py`) | <1 s |
| **total cómputo de esta tarea** | **≈200 s ≈ 3.3 min** |

Muy por debajo de la salvaguarda — quedó margen amplio; se priorizaron 3 semillas completas y bien
reportadas (según indicación explícita del encargo) en vez de forzar una cuarta apurada, dado que el
cómputo real resultó mucho más barato de lo previsto (Phantom corre en ~13 s/semilla a N=2000 con estos
parámetros, el costo dominante es `layout_resortes`, no la física).

---

## Entregables de esta tarea

- `null4_generar_ic.py` — módulo generador NULL-4 (malla causal REAL exacta → reordenamiento de
  inserción → layout de resortes → expansión estática → velocidad → escritura ASCII), reutiliza
  `adj_a_lista_aristas`/`construir_adj_en_orden` de `null4_verificar_invarianza_orden.py` (no reescritas)
  y las piezas congeladas de `p_semilla_causal.py`/`cs073_cierre_holistico.py`/
  `fase1_traducir_a_phantom.py` (sólo importadas).
- `null4_bateria_generar.py` / `null4_bateria_correr.py` / `null4_bateria_comparar.py` — generación,
  corrida y comparación de la batería completa N=2000, semillas de reordenamiento 601-603, en
  `/Users/alexis/phantom_cs073/bateria_null4_n2000/`. `_bateria_comparar.py` importa
  `masa_y_n_sumideros`/`test_permutacion_2_grupos` tal cual de `null3_bateria_comparar.py` (no
  reimplementadas).
- `/Users/alexis/phantom_cs073/bateria_null4_n2000/` — carpeta nueva con las 3 corridas de Phantom (IC,
  `cosmog.in`, `setup.log`, `run.log`, `.sink`, dumps). No se tocó ninguna carpeta de batería anterior
  (`bateria_n2000/`, `bateria_null1_n2000/`, `bateria_null2_n2000/`, `bateria_null3_n2000/`,
  `bateria_real_extra_n2000/`) ni ningún script congelado (`p_semilla_causal.py`,
  `grafo_random_layout_generar_ic_masa_fija.py`, `leer_volcado_phantom.py`,
  `null4_verificar_invarianza_orden.py`) — sólo lectura/importación.
- Nota al margen: se encontró una carpeta preexistente `bateria_n2000/ic_null4/` (fechada 2-ago, previa a
  esta tarea y a la jerarquía de 6 controles actual) — no se tocó ni se usó para nada de este informe; es
  de origen distinto y ajeno a este encargo.
- Este informe.
