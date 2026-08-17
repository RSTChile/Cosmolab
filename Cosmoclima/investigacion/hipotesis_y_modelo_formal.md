# Planteamiento formal — El Niño / Desierto Florido / Gyriosomus

Fecha: 2026-07-31. Autor: Alexis López Tapia (dirección) + CS (redacción/formalización).
Este documento cierra la fase de Aprehensión y abre la fase de diseño. Fuentes
completas en `00_indice.md` y los archivos `.md`/`.csv` de esta carpeta.

## 0. Alcance (ya decidido con Alexis, no reabrir sin razón nueva)
No se busca demostrar el efecto Daisyworld/Gaia a escala planetaria. Se busca:

- **H-Lovelock**: validar, con un modelo cuantitativo (no solo cualitativo), que la
  floración del desierto reduce el albedo superficial local/regional.
- **H-Ciclo**: demostrar que la cadena El Niño/La Niña → Desierto Florido → Gyriosomus
  es un fenómeno natural, cíclico y recursivo de profundidad milenaria — respuesta
  puntual a la narrativa mediática de que el actual Niño "Godzilla" es "por el cambio
  climático" — **sin generalizar a una negación del cambio climático antropogénico
  como fenómeno**, que no es el objetivo ni algo que este dataset pueda sostener.

## 1. Pregunta de investigación
¿Un modelo basado en el **pulso de precipitación acumulada** (magnitud, ventana y
distribución temporal de la lluvia real) predice la ocurrencia, extensión y efecto
climático-ecológico del Desierto Florido y de la emergencia de *Gyriosomus* mejor que
un modelo basado solo en la **etiqueta binaria El Niño/La Niña** — y es esta cadena de
acoplamiento detectable ya en el registro de los últimos ~1000 años, mostrando que es
un ciclo profundo y no un artefacto del período industrial reciente?

## 2. Hipótesis formales, con predicción falsable y control

### H1 — Albedo (Lovelock)
**Afirma**: la floración reduce el albedo superficial y produce un enfriamiento neto
diurno medible, con reparto cuantificable entre albedo y evapotranspiración (siguiendo
el modelo de He et al. 2017, LST ~ ΔAlbedo + ΔET, R²=0.571 de día).
- **Predicción**: aplicando esos mismos coeficientes a series NDVI/albedo de la Zona
  de Alta Simpatría Cladística de *Gyriosomus* (30.5°S-31.5°S) durante años de
  floración confirmados (2014, 2015, 2017, 2021...), debe observarse caída de albedo y
  enfriamiento diurno neto del mismo signo y orden de magnitud que en la zona vecina
  medida por He et al. (17.5°S-29.7°S).
- **Falsación**: si no aparece caída de albedo/enfriamiento en la ZHCS, o el signo se
  invierte, H1 queda refutada específicamente ahí (no invalida lo medido donde sí se
  midió — sería un hallazgo real de heterogeneidad espacial, no un error).
- **Control NULL**: comparar contra años sin floración confirmada en la misma zona; el
  efecto debe desaparecer o reducirse sustancialmente.
- **Pendiente para poder correr esto**: serie NDVI/albedo real para 30.5-31.5°S (no
  existe todavía) + idealmente el piso duro de CEAZA.

### H2 — Pulso de lluvia > fase ENSO como predictor
**Afirma**: un modelo con "precipitación acumulada en ventana de 2-12 meses, con al
menos un evento ≥15 mm" (umbral de germinación de Chávez et al. 2019) ajusta mejor la
ocurrencia de floración/emergencia de *Gyriosomus* que un modelo con solo "año El
Niño sí/no".
- Ya hay evidencia cualitativa a favor: Anguita-Salinas et al. (2026) reportan que la
  mayoría de sus campañas fueron en años La Niña, y que 2011 y 2017 (La Niña)
  produjeron floraciones tan masivas como cualquier El Niño.
- **Predicción**: comparando ambos modelos (regresión logística o similar) por
  AIC/BIC sobre los eventos ya catalogados (13 de Chávez + los de Anguita-Salinas), el
  modelo de pulso de lluvia debe tener mejor ajuste.
- **Falsación**: si el modelo de fase ENSO ajusta igual o mejor, H2 cae — habría que
  aceptar que la simplificación "El Niño → floración" tiene más peso empírico real del
  que este proyecto ha venido asumiendo (aunque seguiría sin sostener la causa
  atribuida por los medios).

**Actualización 01-ago-2026 — primer test cuantitativo real a favor de H2**: Cepeda-Pizarro,
Pizarro-Araya & Vásquez (2005a, PN Llanos de Challe, mismo protocolo de trampeo en tres años
comparables) da el mejor cuasi-experimento encontrado hasta ahora:

| Año | Tipo | Precipitación anual | % del total de artrópodos epígeos capturados |
|---|---|---|---|
| 1989 | Seco, no-ENOS | 22,5 mm (44% bajo el promedio) | 4,9% |
| 2000 | Húmedo, no-ENOS | 61,5 mm (52% sobre el promedio) | 14,4% |
| 1997 | ENOS intenso | 219,5 mm (443% sobre el promedio) | **80,7%** |

Lo relevante para H2: **dentro del propio grupo "no-ENOS"**, el año más lluvioso (2000, 61,5mm)
produjo ~3× más denso-actividad que el año seco (1989, 22,5mm) — es decir, la respuesta ya escala
con la magnitud de la lluvia ANTES de llegar a un año ENOS. Esto es evidencia a favor de un modelo
de pulso continuo (H2) sobre un modelo de solo dos estados ENOS/no-ENOS (que predeciría 1989≈2000
por estar ambos en el mismo estado). **Limitación honesta**: con n=3 años y el pulso más grande
coincidiendo justo con el único año ENOS de la serie, no se puede separar completamente "magnitud
de lluvia" de "ser año ENOS" — sigue siendo el mejor dato disponible, no una prueba definitiva.
*Gyriosomus kingi* + *G. planicollis* dominaron 96,3% de los tenebriónidos del hábitat dunario
costero en el año ENOS 1997. Fuente: `investigacion/agentes_output/02_ecologia_biologia_fenologia.md`.

### H3 — Profundidad histórica del ciclo (sin ruptura estructural reciente)
**Afirma**: la cronología ENSO de Ortlieb (1536-1900) y las dos series de anillos de
ciprés (San Gabriel 1132-1975, El Asiento 1012-1972) deben mostrar una tasa de
recurrencia de años lluviosos/floración-proxy consistente entre siglos, sin una
ruptura estructural que coincida con el inicio del calentamiento antropogénico
(~1850-1950 en adelante).
- **Falsación, y esto hay que decirlo con toda honestidad de antemano**: si aparece un
  cambio estructural claro justo desde mediados del s. XX, no se puede forzar la
  conclusión de "ciclo puramente natural sin modulación antropogénica" — la propia
  literatura climática (IPCC) reconoce que el calentamiento puede modular intensidad/
  frecuencia de ENSO sin causar su existencia. Si el dato apunta ahí, se reporta así,
  matizado — no se descarta ni se fuerza.

### H4 — Rezago ecológico de Gyriosomus
**Afirma**: la densidad máxima de *Gyriosomus* ocurre DESPUÉS del pico de floración
(cuando las anuales se marchitan), no simultáneamente — ya sugerido por observación de
campo en Anguita-Salinas et al. 2026.
- **Predicción**: cruzando el mes-pico de ocurrencias GBIF (oct-nov) contra la fecha de
  pico NDVI de cada evento de floración documentado, debe aparecer un rezago positivo
  y consistente (no simultaneidad).
- Esto es el más barato de probar con lo que YA tenemos (facets GBIF + fechas Chávez).

**Actualización 01-ago-2026 — tres fuentes independientes nuevas, todas apuntan al mismo
patrón**: (detalle completo en `h4_rezago_gyriosomus.md` y
`investigacion/agentes_output/02_ecologia_biologia_fenologia.md`)
- **Cepeda-Pizarro (1989)**: *G. luczoti* concentró el 73% de toda su actividad anual en un único
  pulso de primavera (1978), explícitamente ligado por el autor a "la cantidad de agua caída
  durante el invierno previo" (95,8mm en 1978 vs 13,1mm en 1979).
- **Cepeda-Pizarro et al. (2005a/b)**: octubre es consistentemente el mes de mayor contribución
  numérica (ENOS 1997 y transecto 1989); proponen modelo fenológico explícito — lluvia >20mm a
  fines de invierno dispara germinación/floración con "máximos alrededor de octubre-noviembre y
  declinación abrupta hacia enero", ciclo univoltino con última larva en diapausa, pupación a
  fines de invierno/inicio de primavera, maduración de ovarios en primavera, oviposición
  primavera-verano.
- **Pizarro-Araya et al. (2007)**: dato directo de laboratorio — oviposición de *G. kingi* inicia
  a **fines de septiembre**.
- **Zúñiga-Reinoso, Pinto & Predel (2019)**: *G. camanchaca* colectada el **16-X-2017** durante
  floración intensa documentada en Paposo Norte; cita textual: *"the number of adults of
  Gyriosomus abruptly increases when the desert is flowering; a phenomenon that is mostly
  associated with rainy years in El Niño periods."*

Con estas 4 fuentes independientes (antes solo había 2 casos con mes preciso, 2014/2015, n
pequeño), la ventana **septiembre-diciembre con pico octubre-noviembre** queda mucho mejor
sostenida como patrón robusto, no anecdótico. Sigue sin cerrar H4 en sentido estricto (falta
todavía cruzar mes-pico de floración vs. mes-pico de escarabajos evento por evento con la Tabla 1
completa de Chávez et al. 2019), pero el estado pasa de "evidencia parcial, n=2" a "patrón
reproducido de forma independiente en al menos 4 fuentes distintas, distintas décadas y distintos
sitios" — un salto real en la robustez de la hipótesis.

### H5 — Proyección 2026-2027 (El Niño "Godzilla" en curso)
**Afirma**: aplicando el modelo de los cuatro módulos (H1-H4) a los datos de lluvia YA
medidos para la temporada 2026, se puede proyectar dónde y cuándo va a ocurrir la
próxima emergencia de Desierto Florido y de *Gyriosomus* — antes de que termine de
ocurrir, no después. Es la primera hipótesis de este proyecto que hace una predicción
hacia adelante en vez de explicar datos ya cerrados.

**Datos duros que la sostienen** (NASA POWER, `investigacion/fuentes/lluvia_2026_*.csv`,
consultado 31-jul-2026): las dos zonas candidatas ya superaron varias veces el umbral de
germinación de 15 mm/año (Chávez et al. 2019) **solo en julio 2026**:

| Zona candidata | Localidades reales (Tabla S1) | Lluvia jul-2026 | Lluvia ene-jul 2026 acumulada |
|---|---|---|---|
| **A — Corredor Huasco-Freirina-Carrizal Bajo** (28.0-28.6°S, región III) | Guacolda, Quebrada Freirina, Carrizalillo, Arenales de Huasco, Tres Playitas | **186,7 mm** | 209,1 mm |
| **B — Corredor Ovalle-Limarí / ZHCS** (30.5-31.5°S, región IV) | Ovalle, Socos, Desembocadura río Limarí, Cuesta Tulahuen/Porotitos/Buenos Aires | **408,0 mm** | 415,0 mm |

**Especies candidatas por zona** (con voucher genético real y localidad dentro de esa
zona exacta, Tabla S1 — no toda la lista de 41, solo las que ya tienen presencia
documentada ahí):
- **Zona A**: *G. kingi* (n=11, la más representada con diferencia), *G. kulzeri*,
  *G. parvus*, *G. penai*, *G. subrugatus*, *G. atacamensis*, *G. maculatus*, *G. whitei*,
  *G. gebieni*.
- **Zona B**: *G. reedi* (n=4), *G. luczotii*, *G. marmoratus*, *G. crispaticollis*,
  *G. leechi*, *G. multigranulosus*, *G. peniciliger*, *G. resplendens*,
  *G. foveopunctatus*.

**Actualización 01-ago-2026 — re-chequeo con los ~720 registros nuevos**: se cruzaron las mismas
bandas de latitud (Zona A: 28.0-28.6°S; Zona B: 30.5-31.5°S) contra los 3 CSV nuevos agregados al
proyecto (`guerrero_aceituno_2020.csv`, `gyriosomus_papers_1987_2005_2007_2010.csv`,
`gbif_ocurrencias_2026.csv`), además de re-verificar contra la Tabla S1 original.

- **Nueva especie candidata para Zona A**: ***G. batesi*** — real, con coordenada y fecha
  (Algarrobal, 28°07'S 70°46'O, 350msnm, 03-XI-2002, Pizarro-Araya et al. 2005 Gayana), cae dentro
  de la banda de latitud de la Zona A y no estaba en la lista original de especies candidatas.
- **Zona B**: no aparecieron especies adicionales dentro de la misma banda de latitud en los CSV
  nuevos — todo lo encontrado ahí ya estaba cubierto por la Tabla S1.
- **Advertencia metodológica**: este re-chequeo usó SOLO la banda de latitud (no longitud ni
  nombre de localidad específico), que es una aproximación más gruesa que la selección original de
  H5 (hecha por localidad real dentro de cada corredor). Por eso species como *G. whitei*, *G.
  gebieni*, *G. penai* (Zona A) o *G. luczotii*, *G. marmoratus* (Zona B) no reaparecen con este
  método aunque figuraban en la tabla original — no significa que se hayan "perdido", sino que el
  corte por sólo latitud es menos preciso que la selección original por localidad. No se
  eliminaron esas especies de la lista; solo se añadió lo nuevo confirmado.

**Predicción de fechas**: con el umbral de germinación superado ya en julio (invierno),
y usando el rezago vegetación-lluvia de ~3 meses medido por He et al. (2017), se espera
**pico de floración entre septiembre y octubre de 2026** en ambas zonas. Aplicando el
rezago positivo de H4 (~1 mes entre pico de floración y pico de escarabajos), se espera
**pico de emergencia de Gyriosomus entre octubre y diciembre de 2026**, con posible cola
de actividad hasta enero-febrero de 2027 (diapausa/emergencia escalonada, ya documentada
en la literatura).

- **Cierre temporal de la hipótesis: 28 de febrero de 2027.** En esa fecha se revisa qué
  de esto ocurrió realmente (NDVI/imágenes satelitales, registros GBIF/iNaturalist,
  prensa/CONAF, observación directa) y se reporta el resultado tal como salga —
  confirmado, parcial o refutado por zona/especie — sin forzar ni esperar más allá de esa
  fecha para no mover la meta después de ver el resultado.
- **Falsación**: si para esa fecha no hay evidencia de floración significativa en
  ninguna de las dos zonas (a pesar de la lluvia ya medida), o no hay aumento de
  registros de las especies candidatas en esas zonas y ventana de tiempo respecto a años
  base sin floración, H5 queda refutada — sería un hallazgo real de que el modelo actual
  no captura algo importante (falta considerar distribución temporal de la lluvia, no
  solo el acumulado; ver limitación abajo).
- **Limitación honesta ya conocida**: el dato de lluvia es de **NASA POWER**
  (reanálisis satelital, no estación en terreno) — el propio proyecto ya documentó que
  POWER tiende a *subestimar* la lluvia real en Chile central (hasta -274 mm/año en
  Quinta Normal), así que estos números son probablemente un piso, no un techo. Aun así,
  408 mm en un solo mes en una zona desértica es una señal demasiado grande para ser
  artefacto de sesgo. Falta cruzar con una estación real de CR2 o CEAZA-Met si aparece
  antes del cierre.

## 3. Arquitectura del modelo (cuatro módulos encadenados)

```
Módulo A (forzante)          Módulo B (floración)         Módulo C (albedo/temp)        Módulo D (Gyriosomus)
lluvia real CR2/POWER/  -->  umbral germinación 15mm  -->  ΔAlbedo/ΔLST vía         -->  fenología emergencia
Quinta Normal + ONI          + rezago 2-3 meses            regresión He et al.           (ventana sep-nov,
NOAA (clasificación           + duración ~166 días          (transferida, marcada         rezago post-pico,
ENSO, solo para                                             como extrapolación           densidad 4-12 ind/m²)
comparar con H2)                                            hasta validar in situ)
```

- **Módulo A**: ya tenemos todos los datos (CR2, POWER, Quinta Normal, script
  `datos_clima.py`).
- **Módulo B**: calibrado provisionalmente con los 3 eventos "importantes" de Chávez
  (1997-98, 2002-03, 2011); falta la Tabla 1 completa (los 13) y una serie NDVI propia
  para la ZHCS.
- **Módulo C**: usa los coeficientes de He et al. como hipótesis de transferencia
  (Tabla 2 del paper), explícitamente no validada en la ZHCS todavía — este es el
  módulo más débil del modelo hasta que haya piso duro o medición MODIS directa ahí.
- **Módulo D**: calibrado con `gyriosomus_gbif_facets.csv` + observaciones de campo del
  paper 2026 (densidad, rezago, diapausa).
- **Capa de validación histórica** (no es un módulo, es el control de profundidad
  temporal para H3): cronología Ortlieb 1536-1900 + anillos de árbol 1012-1975.

## 4. Huecos de datos que faltan para que el modelo deje de ser un boceto
1. Serie NDVI/albedo real 2000-2025 recortada a 30.5°S-31.5°S (Módulos B y C — el
   hueco más importante).
2. Tabla 1 completa de Chávez et al. 2019 (calibración fina del Módulo B).
3. Respuesta de CEAZA — piso duro del Módulo C (pendiente).
4. Tabla S5 (suplementaria) de Anguita-Salinas et al. 2026, con el detalle
   campaña-por-campaña de intensidad Niño/Niña — afinaría H2 y H4.

## 5. Límites explícitos del modelo (para no repetir el error de sobre-extender)
- No modela ni afirma retroalimentación planetaria (Daisyworld/Gaia global) — solo
  mecanismo local/regional medido.
- No hace una afirmación general sobre si el cambio climático antropogénico es
  riguroso o no como fenómeno — solo testea si ESTE ciclo específico muestra
  continuidad estructural pre/post-industrial (H3), con apertura explícita a que el
  resultado matice en vez de confirmar.
- No asume causalidad estricta floración→temperatura: el propio He et al. lo advierte
  — es correlación robusta con mecanismo físico plausible, no prueba causal cerrada.
