# Listado Gyriosomus de Marcelo Guerrero (05-ago-2026)

Fecha: 05-ago-2026. Alexis pasó
`Database Gyriosomus phylogeny/Listado Gyriosomus Alexis_041512.xlsx`, enviado por
Marcelo Guerrero (coautor de Anguita-Salinas et al. 2026, contacto de Alexis — ver
`investigacion/00_indice.md`, octava ronda). Trae 2 hojas.

**Crédito**: todo dato de este archivo se marca en el mapa web
(`Web/prueba_de_concepto/`) con la fuente "Datos de la especie: Marcelo Guerrero" —
aparece en el popup de cada punto al pincharlo, como pidió Alexis.

## Hoja1 — checklist taxonómico, 44 especies del género
Lista completa de nombres válidos con autoría/año. Comparada contra las especies que
ya estaban en el mapa web (unión de 4 fuentes previas: Tabla S1 Anguita-Salinas 2026,
GBIF, papers 1987-2010, Guerrero y Aceituno 2020 = 42 especies reales + 1 placeholder
"sp"): **solo 2 son nuevas para el proyecto**:
- *Gyriosomus granocostatus* Fairmaire, 1886
- *Gyriosomus lucens* Kulzer, 1959

**Dos typos reales encontrados y corregidos** (`generar_mapa.py`, diccionario
`FIX_NOMBRES`): la Tabla S1 traía `hopei` (una p) y `peniciliger` (una l); el nombre
válido según este checklist es `hoppei` (Gray, 1832, dos p) y `penicilliger` (Gebien,
1944, dos l). Se corrigieron ahí para que el mapa no cuente la misma especie dos veces.

## Hoja2 — dos tablas

### 2a. Campañas de terreno 1997-2022: año, localidades, tipo ENSO de campo, densidad
Guardada en `fuentes/guerrero_campanas_nino_nina_1997_2022.csv`. **Esto es justamente
la "Tabla S5 campaña-por-campaña Niño/Niña" que había quedado pendiente** en
`00_indice.md` (octava ronda, contacto con Guerrero). Ojo: la columna
`tipo_enso_campo_original` es la clasificación de campo de Marcelo, no necesariamente
la clasificación oficial ONI/NOAA que ya está en
`fuentes/oni_enso_2026_vs_historico.csv` — quedan como dos fuentes separadas y
etiquetadas, no se fusionaron ni se corrigió una contra la otra.

### 2b. Especie → localidad (32 de las 44 especies)
**No trae fecha de colecta ni coordenadas** — solo nombre de especie y nombre de
localidad de texto libre. Se integró al mapa (`Web/prueba_de_concepto/datos_fuentes/
guerrero_listado_localidades_2026.csv`) resolviendo cada localidad a coordenadas,
**solo donde hubo confianza real** (coordenada ya verificada en otra parte del
proyecto, o ficha oficial DMC/Nominatim consultada en vivo) — no se inventó ninguna:

- **27 especies con coordenada añadida** al mapa (algunas comparten localidad, ej.
  Freirina sirve a atacamensis/parvus/planatus; Choros Bajos a barriai/elongatus).
- **4 especies con dato pendiente, no se agregó punto**: *freyi*, *leechi*,
  *multigranulosus* (única localidad dada: "Puente de Talca" o "Puente El Teniente" —
  nombres de puente, no pudieron ubicarse con confianza) y *kulzeri* ("Sur de Huasco",
  descripción direccional, no un nombre de lugar). Las 4 ya estaban en el mapa por
  otras fuentes — solo falta sumarles este punto adicional de Marcelo.
- **Las 2 especies nuevas (granocostatus, lucens) NO entraron al mapa todavía**:
  granocostatus solo tiene "Céspedes" como localidad (lugar real, documentado en
  literatura de Tenebrionidae de Choapa, pero no se encontró su coordenada exacta ni
  en fuentes del proyecto ni en Nominatim); lucens no tiene ninguna localidad en el
  archivo. Quedan en el checklist taxonómico pero sin punto en el mapa — no se
  fabricó una coordenada aproximada para no ensuciar un dataset científico.

## Pendiente
- Si Marcelo puede precisar la coordenada de "Céspedes" (Choapa) y/o una localidad
  para *G. lucens*, quedan listas para sumarse al mapa con el mismo mecanismo
  (agregar fila a `guerrero_listado_localidades_2026.csv`, correr `generar_mapa.py`).
- "Puente de Talca" y "Puente El Teniente" — si Marcelo puede confirmar sobre qué río
  quedan (candidatos: Limarí, Choapa, Huasco), se pueden geolocalizar y sumar el punto
  a freyi/leechi/multigranulosus.
