# *Gyriosomus* — Base de datos y antecedentes consolidados

Documento maestro, fecha 01-ago-2026. Fusiona: (a) las extracciones directas de 8 papers/fichas
leídos en esta sesión (Pizarro-Araya & Jerez 2004; Guerrero & Aceituno 2020; Pizarro-Araya 2010;
Cepeda-Pizarro 1987; Pizarro-Araya et al. 2005 y 2007; fichas de *G. angustus* y *G. granulipennis*),
y (b) los 4 reportes de agentes de investigación paralelos (taxonomía/sistemática; ecología/biología/
fenología; biogeografía/distribución/ocurrencias; literatura reciente/gris/conservación). Es el
documento de referencia único para el experimento y la simulación — está pensado para que Alexis y
Claude lo consulten directo sin tener que releer los ~15 documentos fuente por separado.

**Convención de honestidad**: todo dato marcado "⚠" es una inferencia o no pudo confirmarse en texto
completo (paywall/403). Nunca se inventaron coordenadas ni fechas — donde la fuente no las da, se
anota explícitamente.

---

## 1. Rango geográfico del género (confirmado por múltiples fuentes independientes)

**24°55'S (Reserva Nacional Paposo, Antofagasta) hasta 34°11'S (precordillera de Rancagua/Machalí,
O'Higgins)** — confirmado por Pizarro-Araya & Jerez (2004), reafirmado con casi el mismo valor por
Pizarro-Araya (2010, "25°05'S...hasta la precordillera de Rancagua, 34°11'S"), y consistente con el
hallazgo de campo de Alexis (*G. laevigatus* en Laguna Carén, Pudahuel, ~33.4°S). Corresponde a las
provincias biogeográficas de Coquimbo y Santiago de la subregión Chilena Central (*sensu* Morrone
2006). ~90% de las especies se concentran en las terrazas costeras y llanuras interiores de Coquimbo
y Atacama (Anguita-Salinas et al. 2026, abstract).

**Ausencia confirmada al norte de 24°S** — el límite norte real (no solo el mejor muestreado) parece
ser Taltal/Paposo; *G. angustus* y *G. curtisi* son las especies más septentrionales conocidas
(Cepeda-Pizarro et al. 2005b).

Dos **áreas de endemismo** (análisis PAE, Pizarro-Araya & Jerez 2004): **Paposo** (*G. curtisi*, *G.
angustus*) y **Carrizal Bajo** (*G. planatus*, *G. kingi*). Hotspots de riqueza específica: desierto
costero de Huasco (11-15 especies según distintos conteos) y matorral estepario costero (12-14
especies).

---

## 2. Historia taxonómica y conteo de especies — cronología reconciliada

| Año | Fuente | Conteo reportado | Qué cambia |
|---|---|---|---|
| 1834 | Guérin-Méneville | género fundado, ~6 especies (2 trasladadas de *Nyctelia* + 4 nuevas ⚠no confirmado cuáles) | — |
| 2004 | Pizarro-Araya & Jerez, Rev Chil Hist Nat 77 | 34 especies | PAE, dos áreas de endemismo |
| 2006 | Pizarro-Araya, tesis UdeC ⚠(no verificado en repositorio, solo snippets) | 38 especies | — |
| 2006 | Pizarro-Araya & Flores, Rev Soc Entomol Argent 65 | **-1** (*G. lineatus* sale del género → *Geoborus lineatus* comb. nov.) | única reducción confirmada del conteo |
| 2010 | Pizarro-Araya, IDESIA 28(3) (art15) | "37 especies descritas a la fecha" | discrepancia leve con la cifra de 2006 (38) — no resuelta |
| 2018 | Guerrero & Vidal, Rev Chil Entomol 44(1) | +1 (*G. confusus* sp. nov., Quebrada Chañaral, Coquimbo) | especie nueva |
| 2018 | Guerrero & Diéguez, Rev Chil Entomol 44(1) | +0 (redescubrimiento, no especie nueva) | *G. kulzeri* reencontrada tras 53 años sin registros (desde 1965), cerca de Huasco |
| 2019 | Predel/Zúñiga-Reinoso/Pinto (=Zúñiga-Reinoso et al.), Annales Zoologici 69(1) | +1 (*G. camanchaca* sp. nov., Paposo Norte, 16-X-2017, **colectada durante floración intensa**) | primera especie descrita con genitalia + COI combinados |
| 2020 | Guerrero & Aceituno, Rev Chil Entomol 46(2) | +3 especies nuevas (*G. resplendens*, *G. maculatus*, *G. nigrociliatus*) + 1 stat. nov. (*G. foveopunctatus laevis* Kulzer → *G. laevis* especie plena) | lleva el conteo nominal a ~43-45 según se cuente la elevación de subespecie |
| 2023 | Guerrero, Diéguez, Anguita-Salinas & Zúñiga-Reinoso, Zootaxa 5319(2) | revalidación de *G. crispaticollis* Fairmaire 1886 (rescatada de sinonimia bajo *G. luczotii*) | cambio de contabilidad interna, no necesariamente +1 al total si ya estaba contada como sinónimo |
| 2026 | Anguita-Salinas et al., Systematic Entomology 51(1) | abstract confirma explícitamente **"44 species have been classified"** desde 1834; primera revisión molecular integral: **9 clados, 21 especies válidas, 12 sinonimias, 12 candidatas, 3 no evaluadas** | **tensión activa: 44 nominal vs. 21 validadas molecularmente** |

**Lo que Marcelo Guerrero confirmó a Alexis (44-45 especies)** coincide casi exactamente con la cifra
44 que Anguita-Salinas et al. (2026) confirma por escrito en su abstract. El candidato más plausible
para la especie "45" o el redondeo verbal es *G. crispaticollis* (revalidada 2023) — pero **esto es
una inferencia razonada, no una cita textual encontrada que lo declare explícitamente**. Ningún
documento accedido dice literalmente "con esto llegamos a 45 especies".

**Advertencia importante para el modelo Cosmoclima**: existe una tensión taxonómica real y no
resuelta entre el conteo **nominal/histórico** (43-44-45 especies, el que maneja la comunidad de
especialistas chilenos) y el conteo **molecularmente validado** (21 especies "buenas", con 12
sinonimias y 12 candidatas sin resolver, según Anguita-Salinas et al. 2026). Si el simulador necesita
fijar "cuántas especies tiene el género" como parámetro, hay que decidir explícitamente cuál de los
dos números usar y por qué — no son intercambiables.

**Especie removida del género** (advertencia para no arrastrar registros antiguos mal ubicados):
*Gyriosomus lineatus* Guérin-Méneville, 1834 → *Geoborus lineatus* (Guérin-Méneville, 1834) comb.
nov. (Pizarro-Araya & Flores 2006), con *Geoborus costatus* Blanchard 1847 como sinónimo júnior.
Distribución de esta especie (ya fuera del género): Domeyko (28°58'S) a Pudahuel (33°23'S).

### Checklist de nombres de especie conocidos (compilación, no definitiva)
amabilis, angustus, atacamensis, barriai, batesi, bridgesi, camanchaca, chango, confusus, coriaceus,
crispaticollis, curtisi, elongatus, foveopunctatus, freyi, gebieni, granocostatus, granulipennis,
hoppei, impressus, kingi, kulzeri, laevigatus, laevis, leechi, luczotii, lucens, maculatus,
marmoratus, melcheri, modestus, multigranulosus, nigrociliatus, paulseni, parvus, penicilliger,
penai, planatus, planicollis, pumilus, reedi, resplendens, subrugatus, whitei. (*G. lineatus* excluido
— ver arriba, ya no pertenece al género.) ⚠ Wikispecies reporta 47 en su texto introductorio pero
solo lista 41 nombres — no usar como fuente de conteo, solo como índice cruzado.

---

## 3. Ecología, fenología y respuesta a El Niño/Desierto Florido (sección más relevante para Cosmoclima)

### 3.1 El vínculo cuantitativo más fuerte encontrado con ENSO
**Cepeda-Pizarro, Pizarro-Araya & Vásquez (2005a)** — Parque Nacional Llanos de Challe (28°13'S,
71°04'O), mismo protocolo de trampeo en tres años comparables:

| Año | Tipo | Precipitación anual | % del total de artrópodos epígeos capturados |
|---|---|---|---|
| 1989 | Seco (no-ENOS) | 22,5 mm (44% bajo el promedio histórico) | 4,9% |
| **1997** | **ENOS intenso** | **219,5 mm (443% sobre el promedio)** | **80,7%** |
| 2000 | Húmedo (no-ENOS) | 61,5 mm (52% sobre el promedio) | 14,4% |

En el año ENOS 1997, *Gyriosomus kingi* + *Gyriosomus planicollis* dominaron **96,3%** de todos los
tenebriónidos capturados en el hábitat dunario costero (61,8% en el hábitat pedregoso interior),
concentrados en octubre (mes de mayor contribución, ~39-42%).

### 3.2 Pico fenológico estacional (reproducible en 3 fuentes independientes)
**Septiembre-diciembre, con máximo en octubre-noviembre**:
- Cepeda-Pizarro (1989, Las Cardas, 30°13'S): *G. luczoti* dominó 62,5% de 4.556 tenebriónidos en 15
  meses de muestreo; **73% de toda la actividad se concentró en un único pulso de primavera (1978)**,
  vinculado a la lluvia invernal previa (95,8mm en 1978 vs 13,1mm en 1979 — el autor conecta
  explícitamente el pico de primavera con "la cantidad de agua caída durante el invierno previo").
- Cepeda-Pizarro et al. (2005a, Llanos de Challe): octubre = mes de mayor contribución numérica.
- Cepeda-Pizarro et al. (2005b, transecto 27°-30°S): modelo fenológico propuesto — lluvias >20mm a
  fines de invierno disparan germinación/floración de anuales con "máximos alrededor de
  octubre-noviembre y declinación abrupta hacia enero"; los tenebriónidos seguirían un ciclo
  univoltino con último estadio larvario en diapausa, pupación a fines de invierno/inicio de
  primavera, maduración rápida de ovarios en primavera y oviposición en primavera-verano.

### 3.3 Evidencia con fecha exacta de un evento de Desierto Florido real (2017)
**Zúñiga-Reinoso, Pinto & Predel (2019)** — describen *G. camanchaca* sp. nov. a partir de ejemplares
colectados el **16 de octubre de 2017** en Paposo Norte, Antofagasta, durante lo que el propio texto
describe como *"an intense flowering of the desert"*. Cita textual clave del paper: *"The number of
adults of Gyriosomus abruptly increases when the desert is flowering; a phenomenon that is mostly
associated with rainy years in El Niño periods."* — es la declaración más explícita y citable
encontrada que liga al género con El Niño.

### 3.4 Ecología trófica: omnivoría real (no solo fitofagia)
**Pizarro-Araya (2010, IDESIA 28(3):115-119)** — primer registro documentado de omnivoría en el
género (campo 2006-2008, Atacama/Coquimbo): herbivoría específica (*G. elongatus* come pétalos de
*Rhodophiala bagnoldii* en Punta de Choros, 29°21'S,71°10'O), detritivoría (*G. barriai*, Los Choros),
depredación intraespecífica (exoesqueletos, *G. barriai*), depredación interespecífica (*G. withei*
sobre larvas de Noctuidae, Chañaral de Aceituno) y un registro excepcional de *G. elongatus*
alimentándose del cadáver de un lagarto *Liolaemus silvai*. El autor conecta esto con la necesidad de
optimizar estrategia trófica durante "la fase húmeda (años húmedos no ENOS o años ENOS)".

### 3.5 Reproducción y desarrollo (*G. kingi*, Pizarro-Araya et al. 2007)
Parejas colectadas sept-2002, Llanos de Challe; criadas en laboratorio (17-24°C, 12L:12D).
- **Oviposición inicia a fines de septiembre.** Cópula inmediata tras emergencia de pupa.
- Huevos en grupos de 7-10, máximo 6 posturas por hembra, enterrados 10-20cm o en superficie,
  cubiertos por película de arena adherida vía mucílago.
- Huevo: 4.0mm × 1.1mm, blanco, corion liso sin aeropilas, celdas subhexagonales en el primer tercio.
- Larva de primer estadio: 5.2mm, oligópoda, patas protorácicas cavadoras, tipo "pedobionta".
- **No se observó canibalismo marcado** en condiciones de laboratorio (contrario a lo esperado por la
  literatura previa).
- **Pizarro-Araya, Jerez & Cepeda-Pizarro (2005, Gayana 69(2))** — huevo/larva de 7 especies más (*G.
  impressus*, *luczotii*, *elongatus*, *whitei*, *subrugatus*, *curtisi*, *batesi*): **sin diferencias
  interespecíficas en estructura corial** (no sirve para diagnóstico de especie), pero clípeo, labro
  y tarsungulus de la larva **sí tienen valor taxonómico**.

### 3.6 Actividad térmica y mecanismo El Niño/La Niña (síntesis de divulgación, ⚠no confirmado en paper primario)
Según Ladera Sur (Benčina Campos, 2025, citando directamente a Jaime Pizarro-Araya): rango térmico
óptimo de actividad **25-35°C**, actividad superficial se suspende sobre **40°C**. Mecanismo descrito:
**El Niño** (mar más cálido, más lluvia) → aparición masiva de adultos; **La Niña** (mar más frío,
menos lluvia) → **diapausa prolongada, potencialmente de varios años**. ⚠ Esta última afirmación
("diapausa de varios años") es consistente con la lógica de los datos duros de 3.1-3.2 pero **no se
encontró el paper primario que la sustente numéricamente** — tratar como hipótesis divulgativa hasta
verificar.

### 3.7 Mecanismo evolutivo propuesto (2026, molecular)
Anguita-Salinas et al. (2026) proponen que el **ciclo El Niño/La Niña**, junto con preferencia de
hábitat y **alopatría por barreras fluviales**, serían los motores ecológicos/evolutivos de la
diversificación del género — es decir, ENSO no solo dispara fenología anual sino que podría explicar
especiación a escala geológica. ⚠ Solo abstract accesible, no el mecanismo detallado.

---

## 4. Distribución y registros de ocurrencia — inventario de datos reales disponibles

### 4.1 Archivos de datos ya construidos en el proyecto (coordenadas reales, no inventadas)
| Archivo | Filas | Contenido |
|---|---|---|
| `Web/prueba_de_concepto/datos_fuentes/tabla_s1_anguita_salinas_2026.csv` | 78 localidades | Dataset genético/distribución de Anguita-Salinas et al. 2026 |
| `Web/prueba_de_concepto/datos_fuentes/guerrero_aceituno_2020.csv` | 2 | Paratipos con coordenadas explícitas de *G. maculatus* y *G. laevis* |
| `Web/prueba_de_concepto/datos_fuentes/gyriosomus_papers_1987_2005_2007_2010.csv` | 16 | Parejas colectadas (huevos/larvas 2005/2007), puntos GPS de fichas *angustus*/*granulipennis*, registros tróficos 2010 |
| `investigacion/agentes_output/03_gbif_ocurrencias.csv` | 624 | Volcado crudo de GBIF (`taxonKey=4760162`, `hasCoordinate=true`) tal como lo entregó la API — museos, iNaturalist, secuencias genéticas |
| `Web/prueba_de_concepto/datos_fuentes/gbif_ocurrencias_2026.csv` | 623 | El mismo volcado, convertido al esquema del pipeline del mapa (excluye 1 registro con coordenada errónea del Museo de La Plata, ver 4.3) |

**Actualización 01-ago-2026 — integración completada**: los 4 CSVs de `datos_fuentes/` (incluyendo el
nuevo `gbif_ocurrencias_2026.csv`) ya fueron corridos por `generar_mapa.py` y están reflejados en el
mapa del simulador (`prueba_de_concepto_mapa_capas.html`): **791 especímenes, 43 especies** (27 con
polígono de área, 16 solo como puntos por tener <3 registros). Antes de esta ronda de agentes había
~94 registros reales y ~13-15 especies con coordenada. Pendiente aparte (no tocado en esta ronda):
sumar estos mismos registros a la base SQLite de lluvia diaria si se decide usarlos para alimentar
floración por localidad en el simulador térmico (plan "línea de tiempo real" ya guardado).

### 4.2 Desglose de los 624 registros GBIF por fuente
| Fuente | N | Naturaleza |
|---|---|---|
| iNaturalist | 170 | Observaciones humanas georreferenciadas, 1992-2025 (grueso 2015-2025) |
| INSDC Sequences (EMBL-EBI) | 156 | Especímenes con secuencia genética — probable solapamiento con Tabla S1 2026 |
| iBOL (Barcode of Life) | 135 | Ídem, códigos de barra — probable mismo pool que Tabla S1 |
| OSUC (Ohio State Triplehorn Collection) | 76 | Especímenes 1957-1961, colector histórico **L.E. Peña** |
| CNX "KIM753_BBDD_INVERT" | 70 | Especímenes 2024, Atacama/Coquimbo — ⚠posible línea base de EIA/proyecto minero, no verificado |
| MNHN París | 12 | Colección histórica; incluye especímenes **sin fecha, colector "Gay C."** (⚠probable Claude Gay, expedición s.XIX — alto valor histórico) |
| Guerrero et al. 2023 (material citation) | 3 | *G. crispaticollis*, Alcones/Ovalle/Villablanca-Miranda |
| Museo de La Plata | 2 | Uno con coordenada errónea (ver 4.3) |

### 4.3 Advertencias de calidad de datos
- Registro Museo de La Plata etiquetado "Huasco" con coordenada `-31.1, -64.316681` **cae en
  Córdoba, Argentina** — casi con certeza error de digitación en la base de origen. **No usar sin
  verificar directamente con la colección.**
- **Registro OSUC 683074** (colector Peña), etiquetado localidad "Neotropics" (sin región/altitud/
  año), coordenada `7.833333, -77.75` — **cae en Panamá**, muy lejos del rango real del género
  (24°-34°S, Chile). Casi con certeza un error de georreferenciación genérica en la base de origen
  (posible placeholder "Neotropics" mal geocodificado). **Eliminado** de
  `gbif_ocurrencias_2026.csv` y del mapa del simulador (01-ago-2026, a pedido de Alexis); se
  mantiene intacto en el volcado crudo `03_gbif_ocurrencias.csv` como registro de auditoría, pero
  no debe usarse.
- **Registro MNHN EC8661**, localidad Cobija (-22.55, -70.26, Región de Antofagasta) — coordenada
  real y válida (de hecho el punto más al norte de todo el dataset), pero **sin determinación de
  especie** (solo género). Alexis decidió excluirlo del mapa (01-ago-2026): sin especie asociada,
  un punto extremo del rango no aporta información útil y puede distorsionar la lectura del límite
  norte real del género. **Eliminado** de `gbif_ocurrencias_2026.csv` y del mapa; se mantiene
  intacto en `03_gbif_ocurrencias.csv` como registro de auditoría.
- Los 291 registros INSDC/iBOL vienen con solo 2 decimales de precisión (~1km) y sin nombre de
  localidad — son casi con certeza los mismos especímenes de la Tabla S1 de Anguita-Salinas et al.
  2026 (mismos prefijos de catálogo `GyrSOCO`, `GyrSORU`, `GyrTABA`, etc.). Tratarlos como
  confirmación cruzada, no como localidades independientes nuevas.

### 4.4 Papers de biogeografía/composición faunística nuevos (no taxonómicos)
- **Alfaro, Pizarro-Araya & Flores (2016)**, Rev Mex Biodiv 87(4) — Punta de Choros continental vs.
  archipiélago Los Choros (2005-2006): la estacionalidad estructura el ensamble, NO la partición
  continente/isla.
- **Pizarro-Araya, Alfaro, Ojanguren-Affilastro & Moreira-Muñoz (2021)**, Insects 12(10):916 — hotspot
  Paposo-Taltal, 17 sitios, 24.5°-25.5°S, campañas oct-2015/ago-2017/dic-2019 (la de 2017 coincide con
  la ventana del Desierto Florido de ese año).
- **Pizarro-Araya, Villalobos, Alfaro & Moreira-Muñoz (2023)**, J Arid Environ 214:104995 — mismo
  hotspot, conservación insuficiente, curvas de rarefacción sugieren diversidad oculta.
- **Alfaro & Pizarro-Araya**, islas Choros/Damas/Chañaral — *G. granulipennis* confirmada **ausente**
  en Isla Chañaral, reforzando que su endemismo es a UNA sola isla del archipiélago, no "insular" en
  general.

---

## 5. Estado de conservación — 3 especies con proceso de clasificación (no 2 como se creía)

| Especie | Localidad única/restringida | Estatus | Amenazas principales | Fuente |
|---|---|---|---|---|
| ***G. angustus*** | Paposo (Antofagasta) — único registro histórico | **EN PELIGRO** (RCE 2012, B1ab(iii)+2ab(iii)) | pastoreo caprino, microbasurales, extracción de vegetación, posible efecto central termoeléctrica Taltal | Ficha MMA 2012; solo 8 individuos en 60 trampas (expedición 2011) |
| ***G. granulipennis*** | Isla Choros (Coquimbo) — único registro | Ficha 2012: **VULNERABLE** (VU D2) → **reclasificada EN PELIGRO** por el MMA (Pizarro-Araya et al. 2017) | conejo europeo introducido (competencia trófica) | N=15, 10,5% de fauna Tenebrionidae de la isla |
| ***G. camanchaca*** | Paposo Norte (Antofagasta) — población <10km² | **En evaluación**, 21º Proceso de Clasificación MMA (2025-2026), consulta ciudadana hasta 19-jun-2026 | ⚠asociada en divulgación (no confirmado en expediente oficial) al proyecto de hidrógeno verde INNA (AES Andes, Taltal) — proyecto **retirado 24-ene-2026**, pero la prensa general atribuye el retiro solo a controversia con observatorios astronómicos (Paranal/ELT/CTAO), sin mencionar fauna | Res. MMA N°9517/2025 |

**Tensión documentada**: los propios especialistas (Pizarro-Araya et al. 2017) recomiendan para *G.
granulipennis* categoría UICN de **Datos Insuficientes**, en desacuerdo con la reclasificación oficial
del MMA a "En Peligro" — vale la pena que el proyecto lo tenga presente si usa categorías de
conservación como variable.

**Dato simbólico**: *G. granulipennis* fue el primer insecto chileno con categoría oficial de
conservación (Vulnerable, 2011). *G. angustus* ("vaquita de Paposo") fue elegida embajadora de la
fauna chilena 2022 (Instituto Jane Goodall Chile, votación pública nov-2021).

---

## 6. Qué NO se pudo confirmar (honestidad explícita, acumulado de los 4 agentes)

1. Texto original de Guérin-Méneville (1834) no localizado — no se confirmaron las especies
   fundacionales del género más allá de *hopei* y *luczotii*.
2. Texto completo de Kulzer (1959, en alemán) no accesible.
3. **Texto completo de Anguita-Salinas et al. (2026)** bloqueado por paywall Wiley (403) — todo lo
   usado aquí viene del abstract público + una nota de divulgación (Ladera Sur), NO del artículo
   completo. No se pudo confirmar la lista de las 12 sinonimias ni las 12 candidatas.
4. **Texto completo de Guerrero & Aceituno (2020)** — SciELO devolvió 403; no se pudieron extraer
   coordenadas de localidad tipo de las 3 especies nuevas descritas ese año (ver documento separado
   `distribucion_gyriosomus_pizarro_jerez_guerrero_aceituno.md` para lo que sí se extrajo de ese paper
   en la lectura directa del PDF).
5. Coordenadas de localidad tipo no obtenidas para: *G. chango* (Mondaca 2004), *G. confusus*
   (Guerrero & Vidal 2018), *G. crispaticollis* (revalidación 2023), *G. kulzeri* (redescubrimiento
   2018) — el texto accesible solo daba nombres de lugar.
6. Discrepancia de autoría/año no resuelta para *G. luczotii*: aparece como "Guérin-Méneville, 1831"
   en un snippet y "Laporte, 1840" en otro.
7. Tesis de Pizarro-Araya (2006, UdeC) — repositorio devolvió 404, cita reconstruida solo de snippets.
8. No se encontró ningún estudio primario peer-reviewed que cuantifique densidad/abundancia de
   *Gyriosomus* específicamente durante los eventos de Desierto Florido más recientes (2015-16,
   2021-22, 2023-24, 2026) — el hueco fenológico más importante que queda abierto para H4.
9. No se pudo confirmar si *Gyriosomus camanchaca* fue formalmente parte del expediente ambiental
   (SEA) del proyecto INNA, o si es una asociación hecha solo por la nota de divulgación científica.
10. PDF de la resolución MMA N°9517/2025 (21º Proceso) no se pudo leer como texto — no se confirmó la
    categoría de conservación específica propuesta para *G. camanchaca*.

---

## 7. Bibliografía completa unificada (todas las fuentes, deduplicadas)

### Ya extraídas en detalle antes de esta ronda de agentes (documentos separados en `investigacion/`)
1. Pizarro-Araya, J. & Jerez, V. (2004). Distribución geográfica del género *Gyriosomus*
   Guérin-Méneville, 1834: una aproximación biogeográfica. *Rev. Chil. Hist. Nat.* 77: 491-500.
2. Guerrero, M. & Aceituno, G. (2020). Nuevas especies del género *Gyriosomus*... y nuevo estatus para
   *Gyriosomus foveopunctatus laevis* Kulzer. *Rev. Chil. Entomología* 46(2).
3. Pizarro-Araya, J. (2010). Hábitos alimenticios del género *Gyriosomus*... ¿qué comen las vaquitas
   del desierto costero? *IDESIA* (Chile) 28(3): 115-119.
4. Cepeda-Pizarro, J.G. (1987). Respuesta de los adultos de *Gyriosomus luczoti* a las trampas de
   intercepción en un ecosistema árido-costero del norte de Chile. *Folia Entomol. Mex.* 73: 89-99.
5. Pizarro-Araya, J., Jerez, V. & Cepeda-Pizarro, J. (2007). Reproducción y ultraestructura del huevo
   y larva de primer estadio de *Gyriosomus kingi*. *Rev. Biol. Trop.* 55(2): 637-644.
6. Pizarro-Araya, J., Jerez, V. & Cepeda-Pizarro, J. (2005). Descripción de huevos y larvas de primer
   estadio del género *Gyriosomus*. *Gayana* 69(2): 277-284.
7. Ficha de antecedentes de especie *Gyriosomus angustus* Philippi, 1864. Comité de Clasificación,
   MMA Chile (2012).
8. Ficha de especie clasificada *Gyriosomus granulipennis* Pizarro-Araya & Flores, 2004. MMA Chile.
9. Anguita-Salinas, S. et al. (2026). How many cows are in the desert?... *Systematic Entomology*
   51(1): e70011. DOI: 10.1111/syen.70011. [Tabla S1 ya extraída en detalle — 78 localidades]

### Nuevas — Taxonomía/sistemática (agente 1)
10. Guérin-Méneville, F.E. (1834). [Descripción original del género]. Fuente primaria no localizada.
11. Philippi, R.A. (1887). Catálogo de los Coleópteros de Chile. *An. Univ. Chile* 71: 1-190.
12. Kulzer, H. (1959). Neue Tenebrioniden aus Südamerika, 18. Beitrag... *Ent. Arb. Mus. G. Frey* 10:
    523-547.
13. Flores, G.E. (1997). Revision of the tribe Nycteliini. *Rev. Soc. Entomol. Argent.* 56: 1-19.
14. Flores, G.E. (2000). Cladistic analysis of the Neotropical tribe Nycteliini. *J. NY Entomol. Soc.*
    108(1): 13-25.
15. Pizarro-Araya, J. & Flores, G.E. (2004). Two new species of *Gyriosomus*... *J. NY Entomol. Soc.*
    112(2): 121-126. DOI: 10.1664/0028-7199(2004)112[0121:TNSOGG]2.0.CO;2
16. Mondaca, J. (2004). Nueva especie de *Gyriosomus*... del extremo norte de la Región de Atacama.
    *Rev. Chil. Entomología* 30: 21-26.
17. Pizarro-Araya, J. & Flores, G.E. (2006). La posición sistemática de *Geoborus lineatus* comb. nov.
    (ex. *Gyriosomus*). *Rev. Soc. Entomol. Argent.* 65(3-4).
18. Guerrero, M., Diéguez, V.M., Anguita-Salinas, S. & Zúñiga-Reinoso, Á. (2023). The discarded cow
    from the flowered desert: revalidation of *Gyriosomus crispaticollis* Fairmaire, 1886 stat. rev.
    *Zootaxa* 5319(2): 283-291. DOI: 10.11646/zootaxa.5319.2.9

### Nuevas — Ecología/biología/fenología (agente 2)
19. Cepeda-Pizarro, J.G. (1989). Actividad temporal de tenebriónidos epígeos y su relación con la
    vegetación arbustiva. *Rev. Chil. Hist. Nat.* 62: 115-125.
20. Cepeda-Pizarro, J., Pizarro-Araya, J. & Vásquez, H. (2005a). Composición y abundancia de
    artrópodos epígeos del PN Llanos de Challe: impactos del ENOS de 1997. *Rev. Chil. Hist. Nat.*
    78: 635-650.
21. Cepeda-Pizarro, J., Pizarro-Araya, J. & Vásquez, H. (2005b). Variación en la abundancia de
    Artropoda en un transecto latitudinal del desierto costero transicional de Chile. *Rev. Chil.
    Hist. Nat.* 78: 651-663.
22. Alfaro, F.M., Pizarro-Araya, J. & Flores, G.E. (2009). Epigean tenebrionids from the Choros
    Archipelago. *Entomological News* 120(2): 125-130.
23. Alfaro, F.M., Pizarro-Araya, J. & Flores, G.E. (2016). Composición y estructura del ensamble de
    tenebriónidos epigeos de ecosistemas continentales e insulares. *Rev. Mex. Biodiv.* 87(4).
24. Zúñiga-Reinoso, Á., Pinto, P. & Predel, R. (2019). A New Species of *Gyriosomus*... *Annales
    Zoologici* 69(1): 105-112. DOI: 10.3161/00034541ANZ2019.69.1.006
25. Guerrero, M. & Diéguez, V.M. (2018). Redescubrimiento de *Gyriosomus kulzeri* Peña. *Rev. Chil.
    Entomología* 44(1).
26. Pizarro-Araya, J., Vergara, O.E. & Flores, G.E. (2012). *Gyriosomus granulipennis*... un caso
    extremo a conservar. *Rev. Chil. Hist. Nat.* 85: 345-349.
27. Cepeda-Pizarro, J., Vásquez, H., Veas, H. & Colon, G.O. (1996). Relaciones entre tamaño corporal y
    biomasa en adultos de Tenebrionidae. *Rev. Chil. Hist. Nat.* 69: 67-76.
28. Pizarro-Araya, J. & Cepeda-Pizarro, J. (2013). Taxonomic composition and abundance of epigean
    tenebrionids in the Chilean Coastal Matorral. *IDESIA* 31(4): 111-118.
29. Pizarro-Araya, J., Alfaro, F.M., Ojanguren-Affilastro, A.A. & Moreira-Muñoz, A. (2021). A
    Fine-Scale Hotspot at the Edge... *Insects* 12(10): 916. DOI: 10.3390/insects12100916

### Nuevas — Biogeografía/distribución/ocurrencias (agente 3)
30. Pizarro-Araya, J., Villalobos, E.V., Alfaro, F.M. & Moreira-Muñoz, A. (2023). Conservation efforts
    in need of survey improvement in epigean beetles from the Atacama coast. *J. Arid Environ.* 214:
    104995. DOI: 10.1016/j.jaridenv.2023.104995
31. Alfaro, F.M. & Pizarro-Araya, J. Estimación de la riqueza de coleópteros epigeos de la RN
    Pingüino de Humboldt. SciELO Chile, pid S0717-65382017000200039.
32. Pizarro-Araya, J. et al. Epigean Insects of Chañaral Island (RN Pingüino de Humboldt).
33. GBIF.org — `taxonKey=4760162` (género *Gyriosomus*), consultado 2026-08-01. Datasets: iNaturalist
    Research-grade Observations; INSDC Sequences (EMBL-EBI); iBOL; Triplehorn Insect Collection (OSU);
    "KIM753_BBDD_INVERT" (CNX); Coleoptera collection (EC) MNHN París; Colección de Entomología, Museo
    de La Plata; checklist Plazi de Guerrero et al. 2023.

### Nuevas — Literatura reciente/gris/conservación (agente 4)
34. Guerrero, M. & Vidal, P. (2018). Nueva especie del género *Gyriosomus*... *Rev. Chil. Entomología*
    44(1).
35. Pizarro-Araya, J., Alfaro, F.M., Flores, G.E. & Letelier, L. (2017). Distribution and Conservation
    Status of *Gyriosomus granulipennis*... *The Coleopterists Bulletin* 71(4): 661-666. DOI:
    10.1649/0010-065X-71.4.661
36. Pizarro-Araya, J. (2006). Taxonomía, antecedentes bionómicos y distribución geográfica del género
    *Gyriosomus*. Tesis, Universidad de Concepción. [acceso a repositorio no confirmado]
37. Ministerio del Medio Ambiente de Chile (MMA). 21º Proceso de Clasificación de Especies
    (2025-2026). Resolución de inicio N°9517/2025 (*Gyriosomus camanchaca*).
38. Benčina Campos, J. (28-oct-2025). Vaquitas del desierto, las discretas y esenciales guardianas de
    la flora desértica. *Ladera Sur*.
39. Honour, R. (8-nov-2021). Vaquitas del desierto, *Gyriosomus* sp. *CodexVerde*.
40. El Mostrador (24-ene-2026). AES Andes desiste de polémico megaproyecto de hidrógeno verde INNA.
41. Prensa 2021 (embajadora fauna chilena 2022): emol.com, nostalgica.cl, taltalhoy.cl, fulloutdoor.cl,
    fciencias.userena.cl (Universidad de La Serena).

---

## 8. Cómo usar este documento

- Para preguntas de **taxonomía/conteo de especies**: sección 2 + `agentes_output/01_taxonomia_sistematica.md` (detalle completo con advertencias por fuente).
- Para preguntas de **fenología/ENSO/floración**: sección 3 + `agentes_output/02_ecologia_biologia_fenologia.md`.
- Para **coordenadas reales de localidades**: sección 4 + los 4 CSVs listados en 4.1 (en particular `03_gbif_ocurrencias.csv` para el volumen más grande).
- Para **estado de conservación**: sección 5 + `agentes_output/04_literatura_reciente_gris_conservacion.md`.
- Los documentos previos ya existentes (`distribucion_gyriosomus_pizarro_jerez_guerrero_aceituno.md`,
  `gyriosomus_ecologia_trofica_reproduccion_1987_2010.md`, `h4_rezago_gyriosomus.md`) siguen vigentes y
  contienen detalle adicional (tablas completas de formaciones vegetacionales, descripciones
  morfológicas completas) que no se repitió aquí para no duplicar.
