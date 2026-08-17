# 35 anuarios históricos DMC (1972-2025) — procesados (06-ago-2026)

Alexis pidió procesar `datos/anuarios meteorológicos/` (35 PDFs) e integrarlos a
la base general. Se hizo con 3 scripts en cadena, todos guardados como registro
de trazabilidad (mismo patrón que `agregar_dmc_a_consolidada.py`):

1. `extraer_anuarios_historicos.py` — extracción automática por texto.
2. `agregar_paginas_verificadas_visualmente.py` — las páginas que el script
   anterior marcó como "riesgosas" (no se pudo atribuir con confianza), leídas
   a mano con la página renderizada como imagen.
3. `completar_coordenadas_anuarios.py` — rellena lat/lon de las filas del paso 1
   cruzando contra el catastro de estaciones que traen los propios PDFs.

Todo quedó en `investigacion/fuentes/precipitacion_mensual_dmc_anuario2025.csv`
(el mismo archivo que ya tenía 2025) — **mensual, no diario** (ver más abajo
por qué).

## Resultado
- **3.785 filas nuevas** (estación × mes × año), **30 años** con dato real
  (1979-2025, con huecos reales en los años que la DMC no publicó o que no
  estaban en la carpeta: 1981, 1984-85, 1989-90, 1993, 1996, 1999-2001, 2008,
  2011-13, 2017, 2021-22), **101 estaciones distintas** de la franja
  Atacama-Coquimbo-N.Valparaíso.
- **5 años sin procesar**: 1972, 1973, 1976, 1977, 1978 — son PDFs escaneados
  sin capa de texto (confirmado, no es un problema de mi lectura). Quedan
  pendientes, requieren OCR aparte si se quieren en algún momento.
- **16 filas sin coordenada** (de 3.785) — nombres de estación mal partidos
  por el propio formato del PDF ("Fundo San Félix", "Illapel Sendos", "San
  Felipe D.O-S.", "Vicuña Subcomisaría") que no calzaron con ningún nombre del
  catastro con confianza suficiente. Quedan con el valor de lluvia pero sin
  lat/lon — no se adivinó la coordenada.

## Por qué costó más de un script (el riesgo real que apareció)
Los anuarios NO usan un solo formato en 46 años. Se encontraron 3:
1. **Bloque con nombre pegado** ("NOMBRE ESTACIÓN: X" + su propia tabla,
   ~1988-2010) — extracción directa, sin riesgo.
2. **Fila con nombre como prefijo** ("COMBARBALÁ Total mensual ...", 1979 y
   2010) — extracción directa, sin riesgo (el nombre a veces queda partido en
   2 líneas por el ancho de página; el script lo reconstruye).
3. **Grilla con nombres en bloque aparte** (2014, 2018, 2019, 2020, 2023,
   2024) — texto casi seguro de OCR sobre una tabla escaneada. **El orden en
   que el texto crudo lista los nombres NO es confiable**: se comprobó en la
   página de 2024 que el texto extraído traía "La Canela Fundo, Curimón,
   Marga Marga..." primero, cuando la tabla real (verificada visualmente)
   empezaba por "Visviri, Los Nichos, Combarbalá...". Si se hubiera asumido
   coincidencia de orden a ciegas, **cada valor de esa página habría quedado
   pegado a la estación equivocada**. El script detecta este patrón solo y no
   intenta adivinar — deja la página en una lista de pendientes
   (`anuarios_historicos_paginas_pendientes.csv`) en vez de fabricar la
   atribución. Las 7 páginas que cayeron ahí (2014, 2018×2, 2019, 2020, 2023,
   2024) se leyeron después a mano, viendo la imagen real de cada una.

## Otro error real que aparecí encontró y corregí
El filtro de "estaciones de la zona" usaba `"huaquen" in nombre` y `"san
felipe" in nombre` como substring — eso también capturó **"Huaquén Angol"**
(Araucanía, -38°S) y **"Colonia San Felipe"** (cerca de Angol también,
-37.5°S), que NO son de la zona Gyriosomus pese al nombre parecido. Se
detectaron con un chequeo final de rango de latitud (24.5°S-34.3°S) y se
purgaron 21 filas antes de cerrar el archivo. Lección: un nombre parecido no
basta, hay que verificar la coordenada real cuando esté disponible — ver
[[feedback-no-inventar-coordenadas-sin-verificar]].

## Hallazgo real, no error: julio de 1987
Varias estaciones de la zona (Coirón Retén, Salamanca, Las Vacas, La Canela
Fundo, Catapilco, etc.) muestran totales mensuales de **400 a 1.058 mm/año**
completo, con julio de 1987 solo aportando 400-600 mm de ese total — muy por
encima de lo normal para la zona (habitualmente 100-250 mm/año completo). Se
verificó contra la página original del PDF (no es error de lectura, el propio
documento dice eso). Coincide con el evento El Niño 1986-87/87-88 que ya
está en la serie ONI (`fuentes/oni_enso_2026_vs_historico.csv`, ver
[[cosmoclima-enso-desierto-florido-gyriosomus]]) — un evento real y extremo
en el registro de lluvia de la zona, vale la pena tenerlo presente si se
compara contra otros años "fuertes".

## La pregunta de fondo: ¿esto llega a resolución diaria?
**No.** Los 35 anuarios, sin excepción, publican precipitación **mensual**
("Total mensual" + "Máximo en 24 horas" de UN solo día destacado, no la serie
diaria completa). Es el formato de la publicación anual de la DMC, no cambió
en 46 años. Sumar más anuarios no cambia este techo. El camino a diario real
sigue siendo el mismo de antes: CR2 diario (1900-2018, ya integrado), la API
`getAguaCaidaDiaria` de la DMC (ya usada para Caldera/La Serena, se puede
pedir para más estaciones de la zona si se quiere), o reportes tipo el Excel
de estaciones julio-2026 que ya se sumó.

## Pendiente
- 1972-1978 (escaneados, sin texto) — requieren OCR si se quieren integrar.
- 16 filas sin coordenada por nombre mal partido — quedan documentadas, no
  se completaron.
- Los 4 nombres sin match (`Fundo San Félix`, `Illapel Sendos`, `San Felipe
  D.O-S.`, `Vicuña Subcomisaría`) podrían resolverse revisando a mano la
  página exacta si en algún momento importan para el análisis.
