# Estaciones del extremo norte: Tal Tal, Chañaral, y un bug real de atribución (06-ago-2026)

Alexis pidió analizar por qué angustus/camanchaca/curtisi/chango (clúster
Paposo-Pan de Azúcar) quedaban a 100-207 km de su estación real más cercana.
Resultado: 3 hallazgos encadenados, todos corregidos.

## 1. Taltal estaba excluida por error
`extraer_anuarios_historicos.py` tenía "tal tal" en la lista `EXCLUIR`
(pensada para nombres ambiguos como Mamiña/Toconao/Visviri, que sí quedan
fuera de la franja Gyriosomus). Taltal (-25.4°S, región de Antofagasta) SÍ
cae dentro de 24.8°S-34.2°S, a ~49 km de Paposo. Se sacó de `EXCLUIR` y se
sumó a `ESTACIONES_ZONA`.

## 2. Bug real en el detector de páginas riesgosas
Al ir a buscar los datos de Taltal se encontró que `pagina_es_riesgosa()`
consideraba una página COMPLETA "segura" con que UNA sola fila tuviera el
nombre pegado — pero hay páginas híbridas donde solo ALGUNAS estaciones
tienen el nombre pegado y otras no (verificado a mano en
`anuario-2015.pdf` página 83: 10 estaciones sin nombre pegado, con el orden
real del bloque de nombres sueltos NO coincidiendo con el orden del texto
crudo — la misma trampa que ya habíamos visto en 2024). El resultado
práctico: **Los Nichos, Hurtado, Coirón Retén, Combarbalá, Puerto Oscuro y
Chuchiñí de 2015/2016 nunca se habían extraído, y tampoco habían quedado en
la lista de pendientes** — desaparecidas en silencio, sin ningún aviso.

Se corrigió el detector (ahora CUALQUIER fila sin nombre pegado marca la
página entera como riesgosa) y se re-escaneraron los 30 años: de 7 páginas
riesgosas pasaron a 13. Las 6 nuevas se leyeron a mano con la página
renderizada como imagen (mismo método que las 7 anteriores) —
`agregar_paginas_verificadas_visualmente_2.py`, 184 filas nuevas: Tal Tal,
más los huecos reales de Salamanca, La Vega, El Cobre Santa María,
Catapilco, La Canela, Los Libertadores (2955 msnm, cordillera — se guarda
con su altura real, no es análoga al hábitat de Gyriosomus pese a caer en
la franja de latitud), Quilpué Esval, y relleno de Los Nichos/Hurtado/Coirón
Retén/Combarbalá/Puerto Oscuro/Chuchiñí para 2015-2016.

## 3. Chañaral — la corazonada de Alexis era correcta
Al pedir "poné todas las estaciones en el mapa para chequear visualmente",
apareció un hueco real en la costa entre Taltal y Copiapó — justo donde
Alexis esperaba una estación, al lado de Pan de Azúcar. Búsqueda directa en
los 35 PDFs: **Chañaral SÍ es una estación climatológica completa**
(-26.33°S, -70.62°W), presente en el catastro de 1979, 1980, 1982, 1983,
1986, 1991, 2002 y 2002, formato seguro ("NOMBRE ESTACIÓN:"). Se había
extraído automáticamente (14 filas, 1986/1991/2002) pero **se quedó sin
guardar en el CSV** — quedó solo impresa para inspección durante el
diagnóstico de Taltal y no se persistió. Corregido con `agregar_chanaral.py`.

## Resultado en el simulador
`ESTACION_MAS_CERCANA` recalculada (`generar_curvas_estaciones.py`):

| especie | antes | después |
|---|---|---|
| angustus | Inca de Oro, 205.7 km | Tal Tal, 47.3 km |
| camanchaca | Inca de Oro, 207.1 km | Tal Tal, 48.8 km |
| curtisi | Inca de Oro, 122.6 km | Chañaral, 49.9 km |
| chango | Inca de Oro, 102.6 km | **Chañaral, 24.4 km** |

Las 4 especies del extremo norte pasan de "referencial, no local" (>80 km)
a distancias genuinamente comparables a las del resto del set (mediana
original ~26 km).

## Capa nueva: "Estaciones meteorológicas (110→111)"
A pedido de Alexis, se agregó al mapa del instrumento ET3-Térmico una capa
togglable con **las 111 estaciones de la base**, color según cuántos meses
de dato real tiene cada una, popup con período cubierto — para poder
verificar visualmente en vez de confiar en listas de texto (fue este mismo
chequeo el que encontró el hueco de Chañaral). Generada por la misma
`generar_curvas_estaciones.py` (nuevo bloque `TODAS_LAS_ESTACIONES`).

## Estado final del CSV
`investigacion/fuentes/precipitacion_mensual_dmc_anuario2025.csv`:
**3.983 filas** (3.785 → 3.983), 111 estaciones con coordenada, sin
duplicados, sin filas fuera de la franja 24.5°S-34.3°S (verificado).

## Lección para la próxima vez que se sume un anuario nuevo
Un detector "seguro si ALGUNA fila tiene nombre pegado" no alcanza — hay que
exigirlo de CADA fila, incluso si eso genera más páginas para revisar a
mano. Y los resultados de una búsqueda de diagnóstico ("¿existe esta
estación?") no quedan guardados solos — hay que acordarse de correr el paso
de guardado, no solo el de detección (el hueco de Chañaral fue exactamente
eso: la búsqueda encontró el dato, pero nadie lo escribió al CSV).
