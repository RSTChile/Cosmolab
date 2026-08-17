# CS084 — ¿Suena distinto el tejido real? Espectro del laplaciano de grafo

**Fecha:** 9-ago-2026 · **Origen:** intuición de Alexis a partir de un afiche sobre Laplace/Calor/Onda —
las tres comparten el operador ∇² y sólo difieren en la derivada temporal; la ecuación de onda predice
"frecuencias resonantes de un tambor", que son los valores propios del laplaciano del dominio (problema de
Kac, 1966: "¿se puede oír la forma de un tambor?"). **Escribe:** orquestador (el agente que corrió el cómputo
—real, a N=8000, mismo tejido y controles que Fase III— dejó los datos completos en disco pero no llegó a
escribir el informe). Análisis calculado directamente sobre `cs084_espectro_laplaciano.csv`, sin correr nada
nuevo. **No se declara cierre ni veredicto — sólo números. La lectura final es de Alexis.**

## La pregunta

Fase III (`FASE3_renormalizacion_resultado_CS.md`) midió el diámetro del tejido de CS066 bajo distintas
escalas de agrupamiento, y no logró distinguir el tejido real ("local") de sus controles NULL (mismo grado
pero barajado; Erdős-Rényi puro) — seguía pareciendo mundo-pequeño en todas las escalas. La pregunta de esta
tarea: **¿el espectro COMPLETO del laplaciano —que lleva mucha más información que un solo número como el
diámetro— distingue algo que el diámetro no vio?**

## Método

Mismo tejido, mismo motor (`proceso066` de CS066, k_local=6), mismos 2 controles, reusados tal cual de
`cs080_renormalizacion.py` (Fase III) — sin tocar ese archivo. N=8000 (la misma escala de Fase III, para
comparar manzana con manzana), 5 semillas por brazo, diagonalización DENSA completa del laplaciano L=D-A
(no truncada, porque el diagnóstico de espaciado de niveles necesita el espectro completo, no sólo los
extremos). ~65s por matriz, 15 matrices en total.

## Resultado 1 — la FORMA del espectro sí distingue, y de forma muy clara

| Estadístico | REAL (local) | NULL 1 (barajado) | NULL 2 (Erdős-Rényi) |
|---|---|---|---|
| λ_max (valor propio más grande) | **337 – 468** | 57 – 102 | 18 – 20 |
| Desviación estándar de los eigenvalues | **10.7 – 12.0** | 4.3 – 4.5 | 3.4 – 3.5 |
| Media de los eigenvalues | 7.81 – 7.86 | 7.47 – 7.52 | 6.00 (exacto) |

**El tejido real tiene un espectro MUCHO más ancho y con valores extremos mucho más altos que cualquiera de
los dos controles — sin ningún solapamiento entre las 5 semillas de cada grupo.** El mínimo de λ_max en real
(337) es más de 3 veces el máximo de λ_max en el control barajado (102), y casi 17 veces el máximo del control
Erdős-Rényi (20). Lo mismo con la dispersión (std_eig): real está siempre por encima de 10.7, barajado nunca
supera 4.5. **Esto es una separación que el diámetro nunca mostró — el tejido real "suena" con armónicos mucho
más agudos y extendidos que el ruido, aunque su diámetro sea indistinguible del de un grafo al azar.**

Lectura probable (no verificada en esta tarea): valores propios muy altos en un laplaciano suelen corresponder
a subestructuras localmente muy densas (motivos/hubs con muchos triángulos concentrados). El barajado preserva
el grado de cada nodo pero destruye esos motivos locales — consistente con lo que ya había encontrado NULL-3
de la jerarquía de CS073 (preservar grado+longitud, pero barajar aristas, destruye 28-35% de los motivos).

## Resultado 2 — dimensión espectral (núcleo de calor, la ecuación de calor del afiche en persona)

d_s(t) = -2·d(log Tr(e^{-tL}))/d(log t), en 4 tiempos de difusión:

| t | REAL | NULL barajado | NULL Erdős-Rényi |
|---|---|---|---|
| 0.05 | 0.65 | 0.69 | 0.57 |
| 0.2 | 1.72 – 1.74 | 1.78 – 1.82 | 1.71 – 1.72 |
| 1.0 | 0.63 – 0.68 | 0.51 – 0.53 | **3.5 – 3.7** |
| 5.0 | ~0.03 | ~0.00 | 1.1 – 1.9 |

Ninguna de las tres da algo remotamente parecido a una dimensión 3D (todas por debajo de 2 en casi todo el
rango) — este diagnóstico NO rescata la geometría que el diámetro tampoco encontró, consistente con Fase III.
El salto de Erdős-Rényi a t=1.0 (3.5-3.7) es probablemente un artefacto de que ese control es casi
completamente conexo (99.8% en un componente, contra 91% en real/barajado) — a t grande, la traza queda
dominada por cuántos componentes separados hay, no por geometría real; con 600-700 componentes chicos, real y
barajado se "estancan" antes.

## Resultado 3 — estadística de espaciado de niveles (Poisson vs. Wigner-Dyson/GOE): NO distingue

Esta era la pregunta más afilada ("¿suena a estructura o a puro ruido?"). **Resultado: los tres brazos —
incluido el control Erdős-Rényi puro— dan estadística de espaciado parecida a Wigner-Dyson/GOE (repulsión de
niveles), y NINGUNO se parece a Poisson puro.** Distancias KS a GOE: real 0.010-0.020 (p entre 0.02 y 0.31),
barajado 0.010-0.017 (p entre 0.06 y 0.60), Erdős-Rényi 0.008-0.014 (p entre 0.17 y 0.84) — los tres
estadísticamente compatibles con GOE en la mayoría de las semillas, sin separación clara entre grupos.

**Esto es un resultado negativo honesto, y vale la pena explicarlo:** incluso un grafo puramente al azar
(Erdős-Rényi) tiene, en el cuerpo de su espectro, una estadística de espaciado tipo GOE — es un fenómeno
conocido de matrices de grafos aleatorios, no exclusivo de estructura genuina. Este diagnóstico en particular
distingue "muy regular/tipo red cristalina" de "aleatorio", pero NO distingue "nuestro tejido específico" de
"un grafo al azar genérico" — la metáfora del tambor sonando distinto de la estática no se sostiene en ESTE
diagnóstico, aunque sí se sostuvo, y con fuerza, en el Resultado 1.

## Síntesis, en simple

La intuición del afiche tenía razón, pero no en el lugar exacto que sugería la analogía. **No es que el tejido
real "tenga una nota resonante distinta en un sentido simple" (el diagnóstico de espaciado de niveles no lo
mostró) — es que su espectro completo es mucho más ANCHO y EXTREMO, con armónicos que llegan mucho más alto
que los del ruido.** Es como comparar el sonido de una campana real (llena de armónicos agudos, algunos muy
por encima del tono fundamental) contra el de un trozo de metal genérico golpeado al azar (un timbre mucho más
apagado y estrecho) — ambos "tienen sonido", pero uno tiene una textura mucho más rica que el otro, aunque los
dos duren lo mismo en apagarse (el diámetro, que no distinguió nada).

## Archivos

- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs084_espectro_laplaciano.py` — código, no toca
  `cs066_localidad_geometrogenesis.py` ni `cs080_renormalizacion.py`.
- `cs084_espectro_laplaciano.csv` — datos crudos completos (15 filas: 3 brazos × 5 semillas).
- `cs084_espectros_crudos.npz`, `cs084_forma_espectral.png`, `cs084_dimension_espectral.png`,
  `cs084_espaciado_niveles.png` — datos/figuras de respaldo.
