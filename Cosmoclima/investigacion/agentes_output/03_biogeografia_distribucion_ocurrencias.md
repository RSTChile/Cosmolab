# Gyriosomus — Biogeografía, distribución y registros de ocurrencia (nuevos hallazgos)

Búsqueda enfocada en literatura de biogeografía/endemismo y en registros de ocurrencia con coordenadas reales. Complementa lo ya extraído en sesiones anteriores (Pizarro-Araya & Jerez 2004; Guerrero & Aceituno 2020; fichas MMA de *G. angustus* y *G. granulipennis*; Tabla S1 de Anguita-Salinas et al. 2026 con 78 localidades; ~16 registros de papers de huevos/larvas y ecología trófica). **No se repite ese material aquí**, salvo cuando aporta un dato puntual nuevo (una coordenada, una cita completa).

---

## (a) Registros de ocurrencia — GBIF (taxonKey=4760162, género *Gyriosomus*)

Verificación previa: `species/match?name=Gyriosomus` → `usageKey=4760162`, ACCEPTED, GENUS, confianza 94%. Coincide con el `taxonKey` sugerido.

Consulta `occurrence/search?taxonKey=4760162&hasCoordinate=true` paginada (offset 0/300/600) → **624 registros con coordenadas decimales reales**, ninguno inventado ni aproximado (todos vienen directo del JSON de GBIF). El volcado íntegro (las 624 filas, con todas las columnas: especie, lat, lon, fecha, localidad, provincia, país, colector, institución, colección, catálogo, basisOfRecord, datasetKey, etc.) se guardó en:

**`/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/investigacion/agentes_output/03_gbif_ocurrencias.csv`**

### Resumen por fuente/institución (624 registros)

| Fuente (institutionCode / dataset GBIF) | N registros | Naturaleza | ¿Nuevo vs. lo ya extraído? |
|---|---|---|---|
| iNaturalist (Observations) | 170 | Observaciones humanas georreferenciadas, 1992–2025 (grueso 2015-2025), regiones Coquimbo(94)/Atacama(46)/Antofagasta(13)/Valparaíso(12)/RM(5) | **NUEVO** — ciudadano-científico, coordenadas GPS reales |
| INSDC Sequences (EMBL-EBI, vía ENA) | 156 | Especímenes con secuencia genética depositada, catalogNumber tipo `GyrB…`, `GyrG…`, `GyrSOCO…` etc. (79 pares de coordenadas únicos) | Probable **solapamiento** con Tabla S1 de Anguita-Salinas et al. 2026 (mismo grupo/estudio molecular) — se documenta como confirmación cruzada de coordenadas específicas por espécimen, no como localidad nueva |
| iBOL (International Barcode of Life) | 135 | Ídem, códigos de barras (72 pares de coordenadas únicos) | Igual consideración que INSDC — probable mismo pool de especímenes que Tabla S1 |
| OSUC — Triplehorn Insect Collection, Ohio State University | 76 | Especímenes preservados 1957–1961, colectores **L. E. Peña**, J. E. Barriga, P. Vidal | **NUEVO** — colección histórica no citada antes |
| CNX — dataset "KIM753_BBDD_INVERT" | 70 | Especímenes 2024 (colectores "Cristobal Tello", "AF-FG"), localidades Atacama/Coquimbo (Las Verbenas, Vallenar, Copiapó, Canela, Illapel, Combarbalá, Río Hurtado, La Higuera, La Serena, Ovalle, Vicuña, Pudahuel) | **NUEVO** — base de datos de biodiversidad chilena (posible línea base de EIA/proyecto minero "KIM753"), no vista en extracciones previas |
| MNHN — Coleoptera collection (EC), Muséum national d'Histoire naturelle, París | 12 | Especímenes históricos: 1957 (colector L.E. Peña) + varios **sin fecha, colector "Gay C."** (probablemente Claude Gay, expedición naturalista de Chile, 1830s–1840s) | **NUEVO** — colección europea, valor histórico/biogeográfico alto |
| Checklist Guerrero et al. 2023 (Zootaxa, vía Plazi — material citation) | 3 | *G. crispaticollis*, localidades Alcones/Ovalle/Villablanca-Miranda (Ovalle), 2015 y 2020 | **NUEVO** paper (ver sección b) |
| fcnym.unlp.edu.ar — Colección de Entomología, Museo de La Plata | 2 | 2 especímenes etiquetados "Huasco" | **NUEVO**, pero uno con coordenada claramente errónea (ver nota de calidad abajo) |

**Especies con determinación a nivel de especie** (el resto, 599/624, quedó solo a nivel de género en GBIF): *G. maculatus* (7), *G. crispaticollis* (6), *G. confusus* (4), *G. nigrociliatus* (3), *G. chango* (3), *G. resplendens* (2).

**basisOfRecord**: PRESERVED_SPECIMEN 246, HUMAN_OBSERVATION 240, MATERIAL_SAMPLE 135 (secuencias), MATERIAL_CITATION 3.

### Tabla de registros seleccionados (nuevos, no-genéticos, con mayor detalle geográfico/histórico)

| Especie | Localidad | Lat | Lon | Fecha | Colector | Institución/fuente |
|---|---|---|---|---|---|---|
| *Gyriosomus* sp. | Carrizalillo, Chañaral, Atacama | -26.345833 | -70.618333 | 1957-10-25 | Peña G.L.E. | MNHN Paris (EC34401) |
| *Gyriosomus* sp. | Puerto Oscuro, Choapa, Coquimbo | -31.413889 | -71.573611 | 1957-10/11 | Peña G.L.E. | MNHN Paris (EC34406) |
| *Gyriosomus* sp. | Coquimbo, Elqui | -29.953056 | -71.343333 | 1957-10/11 | Peña G.L.E. | MNHN Paris (EC34407) |
| *Gyriosomus* sp. | Los Vilos | -31.917 | -71.512 | s/f | s/d | MNHN Paris (EC8649/EC8646) |
| *Gyriosomus* sp. | Coquimbo | -29.953 | -71.343 | s/f | **Gay C.** (histórico, s. XIX) | MNHN Paris (EC8660/EC8659) |
| *Gyriosomus* sp. | Cobija, Antofagasta | -22.55 | -70.26 | s/f | s/d | MNHN Paris (EC8661) |
| *Gyriosomus* sp. | Huasco, Atacama | -28.45 | -71.216667 | s/f | s/d | MNHN Paris (EC34421) |
| *Gyriosomus* sp. | Barraza, Limarí, Coquimbo | -30.655556 | -71.482222 | s/f | s/d | MNHN Paris (EC34405) |
| *Gyriosomus* sp. | La Higuera, Elqui, Coquimbo | -29.5 | -71.266667 | s/f | s/d | MNHN Paris (EC34413) |
| *Gyriosomus* sp. | Caleta Los Hornos, Coquimbo | -29.619444 | -71.286667 | 1961-11-17/19 | Peña L.E. | OSUC (683120/683121) |
| *Gyriosomus* sp. | Quebrada Amolanas, costa, Coquimbo | -31.215833 | -71.641389 | 1960-07-24/31 | Peña L.E. | OSUC (683093/683095) |
| *Gyriosomus* sp. | Chañaral de Aceituno, Atacama | -29.044722 | -71.421667 | 1957-10-23/25 | Peña L.E. | OSUC (683067/683124/683125) |
| *Gyriosomus* sp. | Fray Jorge, Coquimbo | -30.681389 | -71.604444 | 1957-11-04/05 | Peña L.E. | OSUC (683060/683063) |
| *Gyriosomus* sp. | Socos (30°44'S 71°31'W), Huentelauquén, Limarí | -30.733333 | -71.516667 | s/f | Barriga, J.E. | OSUC (362430-433, x4) |
| *Gyriosomus* sp. | Choros Bajos, Coquimbo | -29.285556 | -71.323611 | s/f | Vidal, P. | OSUC (683073) |
| *Gyriosomus* sp. | Guampulla, Coquimbo | -30.428333 | -70.981667 | s/f | Peña, L.E. | OSUC (683077) |
| *G. crispaticollis* | Alcones, Ovalle | -30.79 | -71.53 | 2020-10-23 | M. Diéguez & M. Guerrero | Guerrero et al. 2023 (Zootaxa, material citation) |
| *G. crispaticollis* | Ovalle | -30.72 | -71.49 | 2015-11-15 | M. Guerrero & Y. Muñoz | Guerrero et al. 2023 (Zootaxa) |
| *G. crispaticollis* | Villablanca-Miranda, Ovalle | -30.58 | -71.51 | s/f | s/d | Guerrero et al. 2023 (Zootaxa) |
| *Gyriosomus* sp. | Huasco, Atacama | -28.4664 | -71.2192 | s/f | s/d | Museo de La Plata (ent-col 305) |
| *Gyriosomus* sp. (⚠ coord. sospechosa) | "Huasco" (etiqueta) | -31.1 | -64.316681 | s/f | s/d | Museo de La Plata (ent-col 306) — coordenada cae en Córdoba, Argentina; probable error de digitación en la base fuente, NO usar sin verificar |
| *G. nigrociliatus* | Combarbalá, Coquimbo | -30.97886 / -31.08068 | -71.18494 / -71.19598 | s/f | AF-FG | CNX (KIM753_BBDD_INVERT) |
| *Gyriosomus* sp. (14 pts.) | Las Verbenas, Vallenar, Copiapó, La Serena, Río Hurtado, Canela, Illapel, La Higuera, Vicuña, Ovalle, Combarbalá, Pudahuel, El Molle | ver CSV | ver CSV | 2024-03-18 (mayoría) | Cristobal Tello / AF-FG | CNX (KIM753_BBDD_INVERT), 70 registros en total, detalle completo en CSV |
| *G. confusus* | s/loc, Atacama | -28.968815 | -71.341293 | 2022-09-15 | Rodrigo_Barahona | iNaturalist |
| *G. maculatus* | s/loc, Atacama | -28.112619 / -28.125032 / -28.540262 / -29.021 / -27.223069 | -71.1157 / -71.032738 / -71.185251 / -70.690087 / -70.811702 | 2017-2022 (5 obs.) | varios (iNat) | iNaturalist |
| *G. chango* | s/loc, Atacama | -26.137613 | -70.648777 | 2010-11-02 | ANDREA UGARTE | iNaturalist |

*(La tabla completa de las 624 filas — incluyendo las 170 iNaturalist y las ~291 de INSDC/iBOL — está en el CSV adjunto; aquí sólo se destacan los registros de colección/museo y los determinados a especie, que son el aporte más nuevo y verificable.)*

### Nota de calidad de datos
- El registro `ent-col 306` (Museo de La Plata, etiqueta "Huasco, Atacama, Chile") tiene coordenada `-31.1, -64.316681`, que cae en la provincia de Córdoba (Argentina), no en Chile. Es casi con certeza un error de captura en la base de origen (posible transposición de longitud). Se reporta tal cual viene de GBIF pero **no debe usarse** para el simulador sin contactar a la colección.
- Los datasets INSDC/iBOL (291 registros) traen coordenadas con 2 decimales (~1 km de precisión) y sin nombre de localidad — son casi con certeza los mismos especímenes de la Tabla S1 de Anguita-Salinas et al. 2026 (mismo grupo de autores, mismos prefijos de catálogo tipo `GyrSOCO`, `GyrSORU`, `GyrTABA`, etc. que corresponden a acrónimos de localidad). Se documentan como confirmación, no como localidades adicionales independientes.

---

## (b) Papers de biogeografía / endemismo / composición faunística — NUEVOS

### 1. Alfaro, F.M., Pizarro-Araya, J. & Flores, G.E. (2016)
**"Composición y estructura del ensamble de tenebriónidos epigeos (Coleoptera: Tenebrionidae) de ecosistemas continentales e insulares del desierto costero transicional de Chile."** *Revista Mexicana de Biodiversidad* 87(4).
- Muestreo con trampas de intercepción en el borde continental de Punta de Choros y en el archipiélago Los Choros, 2005-2006.
- 982 individuos, 7 tribus, 9 géneros, 17 especies. *Praocis* y *Gyriosomus* = géneros más diversos.
- ANOSIM: diferencias significativas entre estaciones del año, NO entre hábitats (continental vs. insular), aunque con efecto bajo.
- Relevante para el proyecto: confirma que la partición continente/isla no es la que estructura el ensamble — es la estacionalidad (relevante para el motor Gyriosomus/El Niño de Cosmoclima).

### 2. Guerrero, M., Diéguez, V.M., Anguita-Salinas, S. & Zúñiga-Reinoso, Á. (2023)
**"The discarded cow from the flowered desert: revalidation of Gyriosomus crispaticollis Fairmaire, 1886 stat. rev. (Coleoptera: Tenebrionidae) in Southern Atacama, Chile."** *Zootaxa* 5319(2): 283-291. DOI: 10.11646/zootaxa.5319.2.9.
- Revalida *G. crispaticollis* Fairmaire 1886, que estaba sinonimizado bajo *G. luczotii* Guérin-Méneville 1831, con base en estudio de sintipos + morfología comparada + genética.
- Designa lectotipo; aporta preferencias de hábitat y observaciones de campo.
- Localidades citadas con coordenadas: Alcones (-30.79,-71.53), Ovalle (-30.72,-71.49), Villablanca-Miranda/Ovalle (-30.58,-71.51) — todas en el "Desierto Florido" de Atacama Sur/Coquimbo Norte, zona núcleo del simulador Cosmoclima.
- Título mismo ("the discarded cow") remite al apodo local "vaquita"/"cow" para *Gyriosomus* — dato etnobiológico útil para el proyecto.

### 3. Pizarro-Araya, J., Alfaro, F.M., Ojanguren-Affilastro, A.A. & Moreira-Muñoz, A. (2021)
**"A Fine-Scale Hotspot at the Edge: Epigean Arthropods from the Atacama Coast (Paposo-Taltal, Antofagasta Region, Chile)."** *Insects* 12(10): 916. DOI: 10.3390/insects12100916. (Open access, PMC8540830)
- 17 sitios entre el Monumento Natural Paposo Norte y el Sitio Prioritario Paposo (24.5°-25.5°S), 0-1007 m s.n.m.
- Documenta *Gyriosomus angustus* Philippi 1864 con distribución restringida en estos ambientes, estatus EN (En Peligro, MMA Chile) — confirma independientemente el área de endemismo de Paposo ya descrita por Pizarro-Araya & Jerez (2004).

### 4. Pizarro-Araya, J., Villalobos, E.V., Alfaro, F.M. & Moreira-Muñoz, A. (2023)
**"Conservation efforts in need of survey improvement in epigean beetles from the Atacama coast, Chile."** *Journal of Arid Environments* 214: 104995. DOI: 10.1016/j.jaridenv.2023.104995.
- 17 sitios, borde costero Atacama sur, 3237 especímenes, 26 familias, 97 especies. Tenebrionidae y Curculionidae = familias mejor representadas.
- Curvas de rarefacción → esfuerzo de muestreo bajo, sugiere diversidad oculta/subestimada.
- Conclusión de conservación: Paposo Norte y Sitio Prioritario Paposo sin protección legal formal pese a alta riqueza — relevante para el marco de conservación de *Gyriosomus* del proyecto.

### 5. Alfaro, F.M. & Pizarro-Araya, J. — "Estimación de la riqueza de coleópteros epigeos de la Reserva Nacional Pingüino de Humboldt (Regiones de Atacama y Coquimbo, Chile)" / "Estimation of the richness of epigean coleopterans..." *Gayana* o similar (SciELO Chile, pid S0717-65382017000200039).
- Curvas de acumulación + estimadores no paramétricos para islas Choros, Damas y Chañaral.
- **Hallazgo clave para endemismo**: *Gyriosomus granulipennis* (endémica de Isla Choros) **NO fue registrada en Isla Chañaral**, confirmando que su distribución está restringida a un área muy específica dentro de la propia reserva (no sólo "insular" en general, sino a UNA isla del archipiélago). Refuerza el estatus de endemismo puntual ya conocido, con un dato negativo nuevo.
- Relacionado: "Epigean Insects of Chañaral Island (Pingüino de Humboldt National Reserve, Atacama, Chile)" — primer inventario biológico de esa isla, 730 especímenes, 40 especies/34 géneros/17 familias; mismo hallazgo de ausencia de *G. granulipennis*.

### 6. Pizarro-Araya, J., Vergara, O.E. & Flores, G.E. (2012)
**"Gyriosomus granulipennis Pizarro-Araya & Flores 2004 (Coleóptera: Tenebrionidae): Un caso extremo a conservar."** *Revista Chilena de Historia Natural* 85(3): 345-349.
- Versión "paper" (no sólo ficha MMA) del caso de conservación de *G. granulipennis*. Coordenada de referencia de Isla Choros: 29°15'S, 71°32'W (≈ -29.25, -71.533).
- Evalúa vulnerabilidad con 3 métricas: Índice de Prioridad de Conservación (CPI) → En Peligro; MMA Chile → Vulnerable; IUCN Red List también citada.

### 7. Cepeda-Pizarro, J., Pizarro-Araya, J. & Vásquez, H.R. (2005)
**"Composición y abundancia de artrópodos epígeos del Parque Nacional Llanos de Challe: impactos del ENOS de 1997 y efectos del hábitat pedológico."** *Revista Chilena de Historia Natural* 78(4) (dic. 2005). SciELO: S0716-078X2005000400004.
- Trampas de caída, dunas costeras del PN Llanos de Challe (frontera sur del Desierto de Atacama, III Región).
- Tenebrionidae domina el ensamble epigeo de las dunas costeras; **Gyriosomus = género más diverso y abundante**.
- Examina explícitamente el efecto del ENOS 1997 (El Niño) sobre la densidad de actividad — **directamente pertinente al motor El Niño/Desierto Florido/Gyriosomus de Cosmoclima**: este es probablemente el paper empírico más cercano al fenómeno que el simulador busca representar.
- (Nota: especímenes de campaña de primavera 2002 en el mismo parque fueron la base de Pizarro-Araya, Jerez & Cepeda-Pizarro 2005 sobre huevos/larvas, ya extraído antes — mismo sitio de estudio, dos papers distintos.)

### 8. Nota taxonómica colateral (no biogeografía per se, pero afecta la composición del género)
Fernández & Roig-Juñent o similar — **"La posición sistemática de *Geoborus lineatus* comb. nov. (ex. *Gyriosomus*) (Coleoptera: Tenebrionidae)"**, SciELO Argentina (S0373-56802006000200015). Una especie fue removida del género *Gyriosomus* y reclasificada en *Geoborus*. Relevante como advertencia: no todo registro histórico bajo "Gyriosomus" en literatura antigua corresponde necesariamente a la circunscripción actual del género.

---

## (c) Bibliografía completa (papers nuevos citados en este documento)

1. Alfaro, F.M., Pizarro-Araya, J. & Flores, G.E. (2016). Composición y estructura del ensamble de tenebriónidos epigeos (Coleoptera: Tenebrionidae) de ecosistemas continentales e insulares del desierto costero transicional de Chile. *Revista Mexicana de Biodiversidad* 87(4).
2. Guerrero, M., Diéguez, V.M., Anguita-Salinas, S. & Zúñiga-Reinoso, Á. (2023). The discarded cow from the flowered desert: revalidation of *Gyriosomus crispaticollis* Fairmaire, 1886 stat. rev. (Coleoptera: Tenebrionidae) in Southern Atacama, Chile. *Zootaxa* 5319(2): 283-291. DOI: 10.11646/zootaxa.5319.2.9.
3. Pizarro-Araya, J., Alfaro, F.M., Ojanguren-Affilastro, A.A. & Moreira-Muñoz, A. (2021). A Fine-Scale Hotspot at the Edge: Epigean Arthropods from the Atacama Coast (Paposo-Taltal, Antofagasta Region, Chile). *Insects* 12(10): 916. DOI: 10.3390/insects12100916.
4. Pizarro-Araya, J., Villalobos, E.V., Alfaro, F.M. & Moreira-Muñoz, A. (2023). Conservation efforts in need of survey improvement in epigean beetles from the Atacama coast, Chile. *Journal of Arid Environments* 214: 104995. DOI: 10.1016/j.jaridenv.2023.104995.
5. Alfaro, F.M. & Pizarro-Araya, J. Estimación de la riqueza de coleópteros epigeos de la Reserva Nacional Pingüino de Humboldt (Regiones de Atacama y Coquimbo, Chile). SciELO Chile, pid S0717-65382017000200039.
6. Pizarro-Araya, J., Alfaro, F.M. et al. Epigean Insects of Chañaral Island (Pingüino de Humboldt National Reserve, Atacama, Chile).
7. Pizarro-Araya, J., Vergara, O.E. & Flores, G.E. (2012). *Gyriosomus granulipennis* Pizarro-Araya & Flores 2004 (Coleóptera: Tenebrionidae): Un caso extremo a conservar. *Revista Chilena de Historia Natural* 85(3): 345-349.
8. Cepeda-Pizarro, J., Pizarro-Araya, J. & Vásquez, H.R. (2005). Composición y abundancia de artrópodos epígeos del Parque Nacional Llanos de Challe: impactos del ENOS de 1997 y efectos del hábitat pedológico. *Revista Chilena de Historia Natural* 78(4). SciELO: S0716-078X2005000400004.
9. Nota taxonómica: La posición sistemática de *Geoborus lineatus* comb. nov. (ex. *Gyriosomus*) (Coleoptera: Tenebrionidae). SciELO Argentina: S0373-56802006000200015.

**Fuente de datos de ocurrencia**: GBIF.org, `taxonKey=4760162` (género *Gyriosomus* Guérin-Méneville), consultado 2026-08-01. Datasets subyacentes: iNaturalist Research-grade Observations; INSDC Sequences (EMBL-EBI); International Barcode of Life project (iBOL); Triplehorn Insect Collection (Ohio State University); "KIM753_BBDD_INVERT" (CNX); The Coleoptera collection (EC) of the MNHN Paris; Colección de Entomología - Coleoptera (Museo de La Plata); checklist Plazi de Guerrero et al. 2023 (Zootaxa).

**Archivo de datos crudo**: `03_gbif_ocurrencias.csv` (624 filas, mismo directorio).
