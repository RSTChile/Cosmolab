# CS088 — espectro del laplaciano sobre grafos de PROXIMIDAD (k-NN, k=4): REAL vs NULL-1 vs NULL-2

**Fecha:** 9-ago-2026. **Codea/ejecuta:** CC (Claude). **Script:** `cs088_espectro_proximidad_null12.py`.
**No se declara cierre ni veredicto — sólo se reportan números. La lectura final es de Alexis.**

## La pregunta, en simple

CS085 ya había "escuchado" tres tejidos distintos con el mismo instrumento (el espectro completo del
laplaciano de un grafo — como escuchar las frecuencias de un tambor, según el problema clásico de Kac:
"¿se puede oír la forma de un tambor?"). Ahí el tambor era la **malla causal** (quién pudo influir sobre
quién). REAL sonaba muy distinto de sus controles: costaba mucho "romperlo en pedazos" (λ2 chico) y
vibraba en un rango angosto de frecuencias.

Pero NULL-1 (mismo radio, ángulo al azar) y NULL-2 (método Zel'dovich) nunca pasaron por la malla causal
— son puras posiciones, no tienen tambor que tocar. Y de ambos ya sabíamos algo importante por las
baterías con Phantom de verdad: **nunca colapsan, 0 de 8 corridas forman sumideros** (a diferencia de
REAL, que sí forma sumideros de forma robusta).

Este experimento les construye un tambor NUEVO — no la malla causal, sino un **grafo de vecindad
geométrica** (cada partícula atada a sus 4 vecinos más cercanos en el espacio, literal "quién está al
lado de quién", sin ninguna noción de causa) — y les aplica el mismo instrumento. La pregunta: ¿ese
tambor geométrico también suena distinto en REAL que en NULL-1/NULL-2? Si sí, la firma espectral de
CS085 podría estar oyendo algo de la posición/densidad en sí, no sólo de la malla causal específicamente.

## Método

1. Para cada variante (REAL, NULL-1, NULL-2) se leyó el volcado **inicial** (`cosmog_00000`, antes de que
   Phantom mueva una sola partícula por gravedad) de corridas YA existentes en disco, con
   `leer_volcado_phantom.py` (sólo lectura).
2. Sobre esas 2000 posiciones (x,y,z) se construyó un grafo k-NN con k=4 (cada partícula conectada a sus
   4 vecinos más cercanos en el espacio 3D, simetrizado por unión — ver docstring del script para el
   detalle de por qué el grado promedio queda algo por encima de 4).
3. Se corrió el mismo diagnóstico espectral de CS084/CS085 (reusado tal cual, sin reescribir la
   matemática): λ_max, λ2 (conectividad algebraica), dimensión espectral d_s(t) vía núcleo de calor, y
   estadística de espaciado de niveles (Poisson vs GOE).
4. Semillas: 5 corridas independientes ya en disco para NULL-1 (`s1..s5`) y NULL-2 (`s401..s405`); para
   REAL, la corrida canónica más 5 corridas REALES adicionales de `bateria_real_extra_n2000/`
   (`s301..s305` — se verificó que sus posiciones NO son un remuestreo de la canónica, son condiciones
   iniciales genuinamente distintas).

## Resultados — grafo de proximidad (este experimento)

| Grupo  | n semillas | n_aristas (rango) | λ2 (rango)         | λ_max (rango)   | d_s en t=1.0 (rango) | mean(s²) (rango)  |
|--------|-----------:|--------------------|---------------------|------------------|------------------------|---------------------|
| REAL   | 6          | 4523 – 4625         | **0.00275 – 0.00367** | 9.55 – 10.38     | **1.378 – 1.414**       | 1.653 – 1.769        |
| NULL-1 | 5          | 5037 – 5086         | 0.01398 – 0.01500    | 11.62 – 12.37    | 1.955 – 2.018           | 1.440 – 1.475         |
| NULL-2 | 5          | 5112 – 5172          | 0.01034 – 0.02112    | 11.28 – 12.23    | 1.953 – 2.010           | 1.375 – 1.514         |

(Tabla completa con las 3 estadísticas de espaciado de niveles, giant_frac, n_componentes, etc. en
`cs088_espectro_proximidad_null12.csv`.)

**Separación:** REAL queda claramente aislado — su rango de λ2 (0.00275–0.00367) NO se solapa ni un poco
con el de NULL-1 (0.01398–0.01500) ni con el de NULL-2 (0.01034–0.02112), en las 16 corridas (6+5+5). Lo
mismo con d_s(t=1.0): REAL 1.38–1.41 vs NULL-1/NULL-2 1.95–2.02, sin solape. NULL-1 y NULL-2 sí se
superponen bastante entre sí — el instrumento no los distingue con la misma limpieza con la que distingue
a REAL de ambos.

**Nº de aristas:** REAL genera un grafo con MENOS aristas (4523–4625) que NULL-1/NULL-2 (5037–5172) pese
a partir del mismo k=4 nominal. Como el grafo se simetriza por unión (una arista sale de DOS relaciones
"vecino más cercano" que coinciden, o de UNA sola que no coincide), menos aristas totales significa que en
REAL hay más pares de partículas que son "vecino más cercano" mutuo — es decir, REAL ya viene con más
agrupamiento/apareamiento local en sus posiciones iniciales que NULL-1/NULL-2, que están más
"parejamente" distribuidos.

## Comparación con CS085 (malla CAUSAL) — sugestiva, no directa

**Aviso de manzanas y naranjas:** la malla causal de CS085 se construye por un criterio de "quién pudo
influir sobre quién" en dos fases (no por cercanía euclídea), y opera sobre n=1599 nodos (un subconjunto
ya procesado, `dens_bar.npy`), mientras que el grafo de proximidad de este experimento usa las 2000
partículas de gas crudas. Los números NO son comparables en magnitud absoluta — pero el PATRÓN
(¿quién es más chico, quién es más grande?) sí se puede mirar en paralelo:

| Diagnóstico              | Malla CAUSAL (CS085): REAL vs NULL-3 vs RANDOM         | Grafo de PROXIMIDAD (CS088): REAL vs NULL-1/NULL-2 |
|---------------------------|----------------------------------------------------------|--------------------------------------------------------|
| λ2 (conectividad)         | REAL 0.020–0.022 < NULL-3 0.077–0.158 < RANDOM 0.234–0.314 | REAL 0.0028–0.0037 < NULL-1 0.014–0.015 ≈ NULL-2 0.010–0.021 |
| λ_max                     | REAL 11.31–11.58 < NULL-3 11.53–11.79 < RANDOM 15.55–17.99 | REAL 9.55–10.38 < NULL-1 11.62–12.37 ≈ NULL-2 11.28–12.23    |
| d_s en t=1.0               | REAL 1.96–2.05 < NULL-3 2.45–2.54 < RANDOM 2.73–2.83       | REAL 1.38–1.41 < NULL-1 1.96–2.02 ≈ NULL-2 1.95–2.01        |

En **ambas** familias de grafos, construidas con métodos totalmente distintos, REAL queda del lado de
"más difícil de partir, más angosto en frecuencias, dimensión espectral más chica" que sus controles —
la DIRECCIÓN de la separación se repite. Lo que NO se repite es la limpieza relativa entre los controles
entre sí: en la malla causal, NULL-3 y RANDOM SÍ se separan claramente entre ellos (RANDOM > NULL-3 en
todo); en el grafo de proximidad, NULL-1 y NULL-2 quedan mucho más pegados entre sí (rangos solapados en
λ2 y λ_max) — el instrumento distingue bien "REAL vs. el resto" pero no distingue tan bien "NULL-1 vs.
NULL-2" en este grafo geométrico.

## Pregunta central: ¿la firma espectral es sobre posición o sobre causalidad?

Con este resultado, un grafo de proximidad puramente geométrico —sin ninguna noción de causa, sólo
"quién está al lado de quién" en el espacio— YA separa a REAL de NULL-1/NULL-2 con la misma limpieza (sin
solape en ninguna de las 16 corridas) con la que la malla causal separaba a REAL de NULL-3/RANDOM. Eso
sugiere que **al menos parte** de lo que CS085 estaba "oyendo" en la malla causal no es exclusivo de la
construcción causal — ya está presente en la densidad/agrupamiento geométrico crudo de las posiciones
iniciales de REAL, antes de que exista cualquier noción de "quién influyó sobre quién". Dicho de otro
modo: NULL-1 y NULL-2 —que sabemos que nunca colapsan bajo gravedad real (0/8 sumideros)— YA se distinguen
de REAL desde el instante inicial, con un instrumento que ni siquiera mira la física, sólo la geometría.

No se puede concluir de esto que la malla causal "no aporta nada" — el grado de separación y la
consistencia entre semillas (REAL vs NULL-3/RANDOM en la malla causal) es al menos tan limpia como en el
grafo de proximidad, y las magnitudes absolutas de λ2/λ_max/d_s son bien distintas entre ambas familias
(no se puede decir que sean "el mismo número"). Lo que sí queda documentado: la firma espectral **no es
un artefacto exclusivo de construir un grafo causal** — el mismo tipo de separación aparece con un grafo
mucho más simple y agnóstico a la física, hecho sólo de distancias.

## Archivos

- Script nuevo: `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs088_espectro_proximidad_null12.py`
- Tabla completa: `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs088_espectro_proximidad_null12.csv`
- Referencia (malla causal, no tocada): `cs085_espectro_jerarquia_cs073.py` /
  `cs085_espectro_jerarquia_cs073.csv`
