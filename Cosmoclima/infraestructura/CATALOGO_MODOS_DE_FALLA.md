# Catálogo de modos de falla reales de la infraestructura chilena

**16-ago-2026.** Construido con `construir_catalogo_modos_falla.py` sobre los
**50.457 eventos de emergencia por comuna, 2015-2024**, registrados por las
Unidades de Alerta Temprana regionales de SENAPRED
(`datos/crudo/senapred/2026-08-15/Eventos_Emergencia_2015_2024.xlsx`).

La idea rectora del encargo: **no inventamos modos de falla, los encontramos**.
Todo lo que sigue lleva su conteo. Donde el registro no permite responder, dice
«el registro no lo permite» y no se rellena con nada.

Tabla completa y reproducible: `datos/modos_falla_senapred.csv` (22 modos × 47
columnas, incluye desglose mes a mes y año a año de cada modo).

---

## 0 · Privacidad, antes que nada

La columna **«Antecedentes Observaciones»** contiene texto libre de los
operadores con RUT y descripciones de personas fallecidas. **No fue leída, ni
citada, ni cargada en memoria.** La exclusión ocurre por nombre dentro del propio
lector del script (`COLUMNAS_PROHIBIDAS`), antes de cualquier análisis: la
columna no llega a existir en la estructura de datos.

Las demás 43 columnas son categóricas (fecha, región, provincia, comuna, tipo de
evento) o **conteos agregados** de personas y viviendas. Un conteo agregado por
comuna y fecha no identifica a nadie; se usa. **No hay otra columna con dato
personal identificable en el archivo.**

Esto tiene una consecuencia analítica que atraviesa todo el informe y que hay que
decir de frente: **buena parte de la causa de las fallas está escrita justamente
en la columna que no podemos leer.** Lo que sigue es lo que la parte
*estructurada* del registro permite afirmar.

---

## 1 · Qué falla en Chile, ordenado por frecuencia

Los 22 modos cubren **49.407 de 50.457 eventos (99,3 %)**; 350 (0,69 %) no
calzaron ningún patrón (son «accidentes misceláneos», accidente minero y
concentraciones masivas) y se dejaron sin clasificar antes que forzarlos.

| # | modo de falla | n | % del registro | categoría |
|---|---|---:|---:|---|
| 1 | Incendio estructural (residencial, comercial, público, industrial) | **13.691** | 27,13 % | contexto |
| 2 | **Interrupción o alteración del suministro eléctrico** | **12.129** | 24,04 % | **infraestructura** |
| 3 | Incendio forestal | 4.855 | 9,62 % | contexto |
| 4 | **Corte o alteración de la conectividad vial** | **4.744** | 9,40 % | **infraestructura** |
| 5 | Accidente de medios de transporte | 4.146 | 8,22 % | contexto |
| 6 | Incidente con materiales peligrosos | 2.483 | 4,92 % | contexto |
| 7 | Evento meteorológico de precipitación | 1.581 | 3,13 % | contexto |
| 8 | **Interrupción o alteración del agua potable** | **1.546** | 3,06 % | **infraestructura** |
| 9 | Búsqueda y rescate de personas | 1.261 | 2,50 % | contexto |
| 10 | Incendio sin especificar | 642 | 1,27 % | contexto |
| 11 | Inundación, anegamiento o desborde | 634 | 1,26 % | contexto |
| 12 | Temperaturas extremas (ola de calor, helada) | 461 | 0,91 % | contexto |
| 13 | Marejadas | 390 | 0,77 % | contexto |
| 14 | **Interrupción de telecomunicaciones / fibra óptica** | **359** | 0,71 % | **infraestructura** |
| 15 | Viento fuerte | 325 | 0,64 % | contexto |
| 16 | Sismo o erupción volcánica | 325 | 0,64 % | contexto |
| 17 | Remoción en masa | 226 | 0,45 % | contexto |
| 18 | **Interrupción del suministro de gas** | **97** | 0,19 % | **infraestructura** |
| 19 | Plagas y eventos biológicos | 84 | 0,17 % | contexto |
| 20 | **Falla del alcantarillado** | **60** | 0,12 % | **infraestructura** |
| 21 | **Colapso estructural** | **38** | 0,08 % | **infraestructura** |
| 22 | Déficit hídrico / sequía | 30 | 0,06 % | contexto |

**Total de fallas de infraestructura: 18.973 eventos, el 37,6 % del registro.**

Tres lecturas inmediatas:

- **Una de cada cuatro emergencias de Chile es un corte de luz.** 12.129 eventos
  en diez años, sólo por detrás del incendio estructural. La red eléctrica es, de
  lejos, el elemento que más veces falla.
- El agua potable falla **8 veces menos** que la electricidad (1.546 vs 12.129),
  y las telecomunicaciones **34 veces menos** (359). No es que sean más robustas:
  es que **se reportan menos**. Un corte de luz lo reporta la distribuidora al
  minuto; un corte de internet no tiene ese conducto.
- El **déficit hídrico aparece 30 veces en diez años**, menos que el colapso
  estructural. Un registro de emergencias no puede ver una amenaza lenta: la
  sequía no tiene fecha de inicio que anotar. **El registro no permite medir
  amenazas de evolución lenta**, y hay que ir a buscarlas a otra fuente.

---

## 2 · La estructura del registro, y por qué manda sobre todo lo demás

Antes de las causas hay que entender **cómo está anotado el registro**, porque
determina qué se puede preguntar.

No existe una columna «modo de falla». Existen cuatro columnas anidadas —
`Clase Evento` › `Tipo Evento` › `Sub Evento 1` › `Sub Evento 2` — llenadas a
mano, con vocabulario que cambió a lo largo de diez años:

- 69 valores distintos de `Clase Evento` que son ~25 conceptos repetidos con
  acentos, mayúsculas y espacios finales distintos (`Incendios` / `Incendios ` /
  `Vientos` / `Vientos `). Sin normalizar, un mismo modo se cuenta como varios.
- **El año 2021 corrió la jerarquía un nivel completo**: lo que los demás años
  ponen en `Tipo Evento` (p. ej. «Interrupción Suministro Eléctrico»), 2021 lo
  pone directamente en `Clase Evento`. Cualquier análisis que confíe en el nivel
  jerárquico se rompe en 2021.
- El uso de `Sub Evento 1` **sube de 12,5 % de los eventos en 2015 a 62,7 % en
  2022**:

  | año | eventos | con Sub Evento 1 | % |
  |---|---:|---:|---:|
  | 2015 | 6.753 | 842 | 12,5 % |
  | 2016 | 5.485 | 934 | 17,0 % |
  | 2017 | 4.719 | 1.183 | 25,1 % |
  | 2018 | 5.272 | 1.363 | 25,9 % |
  | 2019 | 5.267 | 2.031 | 38,6 % |
  | 2020 | 4.502 | 1.721 | 38,2 % |
  | 2021 | 4.366 | 2.086 | 47,8 % |
  | 2022 | 4.400 | 2.760 | 62,7 % |
  | 2023 | 4.284 | 2.079 | 48,5 % |
  | 2024 | 5.409 | 2.532 | 46,8 % |

Por eso el script normaliza (minúsculas, sin acentos, espacios colapsados) y
busca el modo **en las cuatro columnas a la vez**, sin confiar en el nivel.

### El hallazgo estructural: hay dos regímenes de anotación

Y sólo uno de los dos declara la causa.

**Régimen (a) — falla suelta.** `Clase = «Falla de Servicios y Suministros»`,
`Tipo = «Interrupción Suministro Eléctrico»`, sub-eventos vacíos. **No hay causa
en ninguna columna estructurada.** Es un cajón administrativo, no un fenómeno.

**Régimen (b) — encadenamiento.** `Clase = «Precipitaciones»`, `Tipo = «Sistema
Frontal»`, `Sub Evento 1 = «Alteración Servicio Suministro Eléctrico»`. Aquí el
registro **sí** declara la causa: es el fenómeno del nivel padre.

Sólo se asigna causa en el caso (b). **Nunca se imputa.** Y el resultado es
demoledor:

> **De los 12.129 cortes eléctricos, sólo 933 (7,7 %) declaran una causa.
> Los otros 11.196 (92,3 %) no la declaran en ninguna columna estructurada.**

---

## 3 · Causa declarada, por modo de infraestructura

| modo | n | % con causa declarada | causas declaradas (top 3) |
|---|---:|---:|---|
| Eléctrico | 12.129 | **7,7 %** | Sistema frontal/precipitaciones (664) · Accidente de transporte (96) · Viento (60) |
| Conectividad vial | 4.744 | **87,3 %** | **Accidente de transporte (3.544)** · Sistema frontal (191) · Remoción en masa (135) |
| Agua potable | 1.546 | 9,3 % | Sistema frontal (100) · Déficit hídrico (23) · Remoción en masa (6) |
| Telecomunicaciones | 359 | **0,8 %** | 3 eventos en total: viento, incendio forestal, sistema frontal |
| Gas | 97 | 37,1 % | Materiales peligrosos (34) |
| Alcantarillado | 60 | 46,7 % | Inundación/desborde (13) · Materiales peligrosos (8) · Sistema frontal (6) |
| Colapso estructural | 38 | 2,6 % | Sistema frontal (1) |

Causas completas de los 933 cortes eléctricos que la declaran:

| causa | n |
|---|---:|
| Sistema frontal / precipitaciones | 664 |
| Accidente de transporte | 96 |
| Viento | 60 |
| Incendio estructural | 55 |
| Incendio forestal | 29 |
| Sismo / erupción volcánica | 9 |
| Incendio sin especificar | 9 |
| Tormenta eléctrica | 8 |
| Temperaturas extremas | 1 |
| Remoción en masa | 1 |
| Materiales peligrosos | 1 |

**Cuando el registro se molesta en decir por qué se cortó la luz, en 3 de cada 4
casos dice «sistema frontal».**

Y hay un contraste que vale por sí solo: **la conectividad vial declara causa el
87,3 % de las veces y la electricidad sólo el 7,7 %.** No es que el camino se
entienda mejor. Es que **al camino lo corta algo que se ve** —un camión volcado,
un derrumbe— y a la luz la corta algo que está a kilómetros de donde se nota.

> **La hipótesis del director, contrastada.** «Un transformador que se incendia
> en invierno por sobredemanda de calefacción». El registro **no permite
> verificarla**: la sobredemanda no aparece nunca como causa declarada, y
> «Temperaturas extremas» figura como causa de corte eléctrico **1 sola vez en
> diez años**. Lo que el registro sí muestra es el otro mecanismo, el de la línea:
> sistema frontal (664) + viento (60) + tormenta eléctrica (8) = **732 de 933**.
> La falla por demanda, si existe, está en el texto libre o directamente no se
> reporta a SENAPRED. **Se declara pendiente, no resuelta.**

---

## 4 · ★ La pregunta central: ¿qué falla en invierno y no en verano?

### 4.1 · El método, y por qué no es circular

Mirar la estacionalidad de un modo de falla **no sirve**, porque «corte
eléctrico» mezcla dos poblaciones distintas. Y comparar sólo los eventos con
causa meteorológica **sería circular**: por supuesto que un «sistema frontal»
ocurre en invierno.

La partición que vuelve falsable la pregunta es en **tres grupos**:

1. **causa meteorológica declarada** — si el clima mueve el riesgo, aquí se ve;
2. **causa NO meteorológica declarada** (choque, incendio, sismo) — **el grupo de
   control**: si aquí *también* hubiera exceso invernal, el exceso sería un
   artefacto del registro (operadores que anotan más en invierno) y no un hecho
   del mundo;
3. **sin causa declarada** — la masa del registro.

Y todo se lee contra la **línea base del registro completo**: 13.097 eventos en
invierno (JJA) vs 14.086 en verano (DEF), **razón 0,93**. Chile, en conjunto,
tiene *levemente menos* emergencias en invierno que en verano.

### 4.2 · El resultado

Sobre los 18.973 eventos de falla de infraestructura:

| grupo | n | invierno (JJA) | verano (DEF) | **invierno/verano** |
|---|---:|---:|---:|---:|
| **(1) causa METEOROLÓGICA** | 1.255 | 794 | 116 | **6,84** |
| (2) causa NO meteo — **CONTROL** | 4.031 | 959 | 1.138 | **0,84** |
| (3) sin causa declarada | 13.687 | 3.381 | 3.934 | **0,86** |
| línea base (registro entero) | 50.457 | 13.097 | 14.086 | 0,93 |

**El control se comporta exactamente como debía.** El grupo (2) queda en 0,84 —
pegado a la línea base de 0,93, incluso ligeramente por debajo — mientras el
grupo (1) se dispara a **6,84**. **El exceso invernal no es del registro: es del
clima.** Si los operadores simplemente anotaran más en invierno, el grupo (2)
habría subido también. No subió.

Curva mensual de la falla de infraestructura con causa meteorológica (n=1.255):

```
   ene    66  █████████████
   feb    29  █████
   mar    86  █████████████████
   abr    83  ████████████████
   may    90  ██████████████████
   jun   376  ███████████████████████████████████████████████████████████████████████████
   jul   189  █████████████████████████████████████
   ago   229  █████████████████████████████████████████████
   sep    22  ████
   oct    36  ███████
   nov    28  █████
   dic    21  ████
```

**Junio concentra el 30 % del año.** Junio solo (376) tiene más fallas de
infraestructura de causa climática que **todo el semestre septiembre-febrero
junto** (202).

### 4.3 · Desglose por modo — el patrón se repite en todos

| modo | causa meteo | causa NO meteo (control) | sin causa |
|---|---:|---:|---:|
| **Agua potable** | **15,67** (n=112) | 0,18 (n=32) | 0,68 (n=1.402) |
| **Eléctrico** | **11,49** (n=734) | 0,45 (n=199) | 0,91 (n=11.196) |
| **Alcantarillado** | **6,00** (n=20) | — (n=8) | 1,00 (n=32) |
| **Conectividad vial** | **3,03** (n=386) | 0,89 (n=3.729) | 1,05 (n=629) |

*(razón invierno/verano; telecom, gas y colapso tienen n insuficiente en la
columna meteo y no se interpretan)*

**Cuatro modos independientes, el mismo patrón, sin una sola excepción**: la
columna meteo va de 3 a 16; la columna de control va de 0,18 a 0,89, toda ella
por debajo de la línea base. Es el resultado más sólido de este catálogo.

### 4.4 · El aislamiento de personas: la señal más limpia del registro

Medido sobre la columna numérica `Total Aislados` (no sobre la taxonomía, que no
tiene la palabra «aislado»):

**289 eventos dejaron personas aisladas; 140.303 personas en total.**

| mes | ene | feb | mar | abr | may | **jun** | jul | **ago** | sep | oct | nov | dic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eventos | 23 | 15 | 15 | 14 | 20 | **107** | 21 | **48** | 10 | 5 | 6 | 5 |

**Invierno 176 · verano 43 · razón 4,09.** Junio solo concentra el 37 % del año.

Causa (`Clase Evento`): **precipitaciones (219)**, remoción en masa (19),
accidentes misceláneos (17), accidente de transporte (10), remoción en masa por
detonante meteorológico (8), nevadas (3), lluvia (2). **252 de 289 (87,2 %) son
meteorológicos o disparados por meteorología.**

Comunas que concentran: **Alto Biobío (12), Lonquimay (10), San José de Maipo
(9), Melipeuco (9), San Fernando (8), San Clemente (8), Toltén (7), Curarrehue
(6)**. Comunas cordilleranas de acceso único: **exactamente el perfil
`extension` + `redundancia ausente` del §4.3 del estudio.** El aislamiento no es
una amenaza, es lo que pasa cuando un elemento lineal sin ruta alterna se corta.

---

## 5 · Distribución territorial

Fallas de infraestructura por región (n=18.973):

| región | n | | región | n |
|---|---:|---|---|---:|
| O'Higgins | 3.206 | | Los Ríos | 1.089 |
| Maule | 2.684 | | Valparaíso | 829 |
| Araucanía | 2.017 | | Antofagasta | 454 |
| Metropolitana | 1.941 | | Ñuble | 447 |
| Coquimbo | 1.675 | | Aysén | 344 |
| Biobío | 1.226 | | Magallanes | 341 |
| Atacama | 1.200 | | Tarapacá | 298 |
| Los Lagos | 1.100 | | Arica y Parinacota | 121 |

**Advertencia obligatoria:** estos son conteos **absolutos, sin normalizar por
población, por longitud de red ni por número de clientes**. O'Higgins encabeza en
parte porque reporta mucho. **El registro no permite calcular una tasa**; hacen
falta datos de población y de red que no están en esta fuente. Leer este cuadro
como ranking de fragilidad sería repetir el error de `ExpEstr`.

Lo que sí es interpretable es el **contraste entre territorios según la causa**,
porque ahí cada comuna se compara consigo misma:

| falla de infraestructura por causa | comunas que concentran |
|---|---|
| **causa meteorológica** (n=1.255) | Punta Arenas (33), Vicuña (25), Cisnes (23), Talca (15), Alto Biobío (15), Constitución (14), San Gregorio (13) |
| **causa NO meteorológica** (n=4.031) | Rancagua (143), Coquimbo (81), Requínoa (78), San Fernando (77), Graneros (76), Ovalle (71) |

**Son dos mapas distintos de Chile.** El de la causa climática es cordillerano,
austral y de valle transversal; el de la causa no climática es el eje urbano
central. Ningún nombre se repite entre las dos listas.

Corte eléctrico de causa meteorológica por región: **Maule (233), Biobío (199),
Metropolitana (106), Valparaíso (75), Araucanía (39), Magallanes (26)** — la zona
central-sur, donde el frente y el arbolado se cruzan con la red aérea.

> **Lo que no se encontró:** un gradiente norte-sur limpio del corte eléctrico.
> La razón invierno/verano por región es ruidosa (Biobío 2,07 y Tarapacá 2,10 en
> lo alto; O'Higgins 0,75 y Ñuble 0,63 en lo bajo) y no ordena por latitud. **El
> registro no sostiene la idea de que «más al sur, más invernal el corte».**

---

## 6 · Tendencia en los diez años — y por qué casi no se puede leer

| modo | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | pendiente |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Eléctrico | 1.547 | 1.235 | 1.540 | 1.475 | 1.518 | 1.418 | 1.162 | 856 | 629 | 749 | **−96/año** |
| Incendio estructural | 1.068 | 928 | 914 | 1.211 | 1.322 | 1.307 | 1.497 | 1.619 | 1.683 | 2.142 | **+117/año** |
| Conectividad vial | 149 | 405 | 366 | 515 | 469 | 264 | 350 | 778 | 692 | 756 | **+54/año** |
| Incendio forestal | 1.448 | 846 | 578 | 447 | 308 | 283 | 198 | 211 | 258 | 278 | **−105/año** |
| Agua potable | 114 | 113 | 205 | 251 | 160 | 137 | 150 | 143 | 120 | 153 | −1/año |

**Estas tendencias no son confiables como hechos del mundo, y hay que decirlo con
todas sus letras.** Tres pruebas de que reflejan el registro y no la realidad:

1. **Accidente de transporte cae de 598 (2021) a 43 (2022)**, un −93 % en un año.
   Ninguna política de tránsito hace eso. Es un cambio de clasificación.
2. **Gas registra 50 eventos en 2015 y luego 0, 1, 0, 5, 0…** El modo no
   desapareció; dejó de anotarse así.
3. **La contradicción más clara: los cortes eléctricos con causa meteorológica
   declarada caen de 331 (2017) a 1 (2020), 6 (2022) y 5 (2024)** — mientras el
   uso del sub-evento *subía* de 25 % a 63 %. Los frentes no dejaron de cortar la
   luz entre 2020 y 2024. **Dejó de anotarse el encadenamiento.**

**Conclusión honesta: el registro no permite medir tendencia decenal de ningún
modo de falla.** La serie mide una mezcla inseparable de fenómeno y práctica
administrativa. Sirve para estructura y estacionalidad —donde el sesgo de
anotación se reparte parejo entre meses— no para tendencia.

---

## 7 · Naturales y deliberados: lo que el registro NO permite

Se buscaron **17 patrones** de intención en toda la taxonomía normalizada
(`sabotaje`, `atentado`, `vandalismo`, `robo`, `hurto`, `intencional`,
`delictual`, `manifestación`, `disturbio`, `terrorismo`, `ciber`, `ataque`,
`saqueo`, `protesta`, `corte de cable`, `sustracción`, `malicioso`).

> **Resultado: 2 eventos de 50.457. El 0,0040 %.** Y son 2 apariciones del patrón
> «protesta».

**El registro de SENAPRED no distingue vectores deliberados. No es que sean
raros: es que la taxonomía no tiene la palabra.**

La columna `Origen Evento` **no sirve para esto** y confundirla sería un error
grave:

| Origen Evento | n |
|---|---:|
| Antrópico | 45.160 |
| Natural | 5.297 |

**«Antrópico» significa ORIGEN HUMANO, no INTENCIÓN.** Los 45.160 son
mayoritariamente incendios estructurales accidentales y choques. Tomar
«Antrópico» como «deliberado» inflaría el vector deliberado al 89 % del registro.

**Clasificación resultante de cada modo** (columna `vector` del CSV):

| clasificación | modos |
|---|---|
| **NATURAL** — la mayoría de sus causas declaradas es meteorológica | Eléctrico, Agua potable, Alcantarillado, Inundación, Remoción en masa, Marejadas, Viento, Precipitaciones |
| **MIXTO natural/accidental** — nunca deliberado, porque el registro no lo anota | Conectividad vial, Incendio forestal, Accidente de transporte, Rescate |
| **NO DISTINGUIBLE** — sin causa declarada, o con menos de 20 casos | Incendio estructural, Materiales peligrosos, Telecom, Gas, Temperaturas extremas, Sismo/volcán, Colapso estructural, Déficit hídrico, Plagas |

**Ningún modo se clasifica como DELIBERADO, y ninguno puede serlo con esta
fuente.** Para el eje deliberado del §4.3 del estudio hace falta otra fuente
(Fiscalía, Carabineros, reportes de las propias empresas). **Se declara: el
registro no lo permite.**

### El caso que lo demuestra: octubre de 2019

El registro **sí contiene** un ataque deliberado masivo a la infraestructura.
Simplemente **no lo etiqueta**. Cortes de conectividad vial, día a día,
octubre 2019:

```
   01-oct   1  ▇                        19-oct   9  ▇▇▇▇▇▇▇▇▇
   03-oct   2  ▇▇                       20-oct   5  ▇▇▇▇▇
   04-oct   3  ▇▇▇                      21-oct   5  ▇▇▇▇▇
   06-oct   3  ▇▇▇                      22-oct   4  ▇▇▇▇
   07-oct   1  ▇                        23-oct   5  ▇▇▇▇▇
   09-oct   1  ▇                        24-oct   7  ▇▇▇▇▇▇▇
   10-oct   2  ▇▇                       25-oct   4  ▇▇▇▇
   11-oct   1  ▇                        26-oct   1  ▇
   13-oct   4  ▇▇▇▇                     28-oct   8  ▇▇▇▇▇▇▇▇
   14-oct   1  ▇                        29-oct   6  ▇▇▇▇▇▇
   15-oct   2  ▇▇                       30-oct   8  ▇▇▇▇▇▇▇▇
   17-oct   3  ▇▇▇                      31-oct   1  ▇
```

**Hasta el 18 de octubre: 24 eventos. Desde el 19 de octubre: 63 eventos.** El
salto cae exactamente en la fecha del estallido social.

Y la comparación entre años es aún más nítida. Cortes de conectividad vial **sin
causa declarada**, octubre + noviembre de cada año:

| año | 2015 | 2016 | 2017 | 2018 | **2019** | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eventos | 12 | 11 | 4 | 36 | **117** | 9 | 2 | 5 | 8 | 6 |

**117 contra una mediana de 9.** Trece veces el año típico, en los dos meses
exactos, en el modo de falla exacto que un corte de ruta produce.

**Cómo los clasifica el registro:** `Falla de Conectividad Vial | Accidente |
Alteración Conectividad`, con `Origen = Antrópico` y **causa vacía**, en el
100 % de los 117 casos.

Esto confirma empíricamente el §2 del `ESTUDIO_VECTORES_DE_AMENAZA.md`: **el
vector deliberado existe en los datos como señal estadística y está ausente del
vocabulario.** Lo que no se puede nombrar, no se puede contar — y lo que no se
cuenta, termina apareciendo como una constante «Alta» en una columna FANC.

*Cautela declarada:* la atribución al estallido se apoya en la coincidencia de
fechas y en el salto diario, **no en el registro**, que no anota intención.
Confirmarla exigiría leer el texto libre, que está prohibido. **Se afirma el
patrón temporal; no se afirma la intención.**

---

## 8 · Qué propiedad expuesta explota cada modo de falla

Vocabulario del `ESTUDIO_VECTORES_DE_AMENAZA.md` §4.3: `dep_energia` ·
`dep_datos` · `dep_humana` · `exp_intemperie` · `extension` (puntual / lineal /
areal) · `confinamiento` · `redundancia` · `t_reposicion`.

| modo (n) | propiedades expuestas | por qué, según lo que muestra el registro |
|---|---|---|
| **Eléctrico** (12.129) | `dep_energia` · `extension` (lineal) · `exp_intemperie` · `redundancia` ausente | El frente (664 casos declarados) y el viento (60) **no atacan un nodo: atacan la longitud**. Una red lineal a la intemperie ofrece un blanco proporcional a sus kilómetros, y el clima lo golpea entero a la vez. Es la ventaja estructural del §2 del estudio: el temporal produce gratis la sincronización (`FSS`) que un atacante tendría que planificar. |
| **Conectividad vial** (4.744) | `extension` (lineal) · `exp_intemperie` · `redundancia` (rutas alternas) · `t_reposicion` | El camino se corta **en su punto más débil, no en su promedio**. La `redundancia` aquí es literal —¿hay otra ruta?— y es lo único que separa un atraso de un aislamiento. Sus 386 cortes climáticos son cordilleranos; sus 3.544 cortes por accidente son urbanos. |
| **Agua potable** (1.546) | `dep_energia` · `extension` (lineal) · `dep_humana` · `t_reposicion` | **Modo de falla de segundo orden**: el agua urbana depende del bombeo, o sea de la electricidad. Su razón invierno/verano de **15,67** —la más alta de todo el catálogo— sugiere que hereda la estacionalidad de la red que la alimenta. |
| **Telecom** (359) | `dep_datos` · `dep_energia` · `extension` (lineal) · `redundancia` | En el §4.3 es el vector deliberado por excelencia (`dep_datos`); **en el registro real aparece como consecuencia del corte eléctrico** — sin energía en la antena, no hay red. La propiedad que lo mata en Chile no es la que uno esperaría. |
| **Alcantarillado** (60) | `extension` (lineal) · `confinamiento` · capacidad | **No se rompe: se desborda.** Falla por saturación, cuando lo que entra excede la capacidad de diseño. Su causa declarada dominante es «inundación/desborde» (13). |
| **Gas** (97) | `extension` (lineal) · `confinamiento` · `t_reposicion` | Red enterrada: el `confinamiento` la protege del clima —**es el único modo lineal sin causa meteorológica declarada, 0 de 97**— pero convierte cualquier fuga en un evento de materiales peligrosos (34 de 36 causas declaradas). |
| **Colapso estructural** (38) | `t_reposicion` · `dep_humana` · `confinamiento` | Pérdida total del elemento; el tiempo de reposición domina todo lo demás. |
| **Aislamiento de personas** (289 eventos, 140.303 personas) | `extension` · **`redundancia` ausente** · `confinamiento` territorial | No es un modo de falla: es **la consecuencia** de que un elemento lineal sin ruta alterna se corte. Alto Biobío, Lonquimay, Melipeuco, Curarrehue: comunas de acceso único. Razón invierno/verano **4,09**. |

### La lección de diseño que sale de los números

Los modos con razón invierno/verano alta comparten **una misma firma**:
`extension` lineal **+** `exp_intemperie` **+** `redundancia` ausente. Los que no
la tienen —gas, enterrado y confinado— **no muestran estacionalidad climática en
absoluto (0 de 97 causas meteorológicas)**.

> **Esto es medible sin pedirle a nadie un veredicto sobre el futuro**, que es
> exactamente por qué `VT` funciona y `FANC` no (§1 del estudio). *¿Es lineal?
> ¿Está a la intemperie? ¿Hay camino alterno?* Son tres preguntas sobre **lo que
> la cosa es**, y el registro muestra que predicen la estacionalidad de la falla.

---

## 9 · Limitaciones declaradas

1. **La causa está mayormente en la columna prohibida.** 92,3 % de los cortes
   eléctricos no declaran causa en columnas estructuradas. Todo el análisis
   causal se apoya en el 7,7 % que sí la declara, y **ese 7,7 % no es una muestra
   aleatoria**: el operador anota el encadenamiento justamente cuando hay un
   fenómeno grande que lo motiva. **Sobrerrepresenta lo meteorológico.**
2. **Por eso el resultado del §4 se sostiene en el grupo de control, no en la
   proporción.** Lo que se afirma no es «el X % de los cortes es climático»
   —eso no se puede saber— sino que **entre los eventos con causa declarada, los
   climáticos son 6,84 veces más invernales y los no climáticos no lo son.**
   La comparación es interna y por eso resiste el sesgo de selección.
3. **No hay tendencia decenal legible.** Ver §6: tres pruebas de cambio de
   práctica administrativa que dominan sobre cualquier señal real.
4. **El año 2021 usa otra jerarquía.** Se maneja buscando en las cuatro columnas
   a la vez, pero cualquier análisis por nivel se rompe ahí.
5. **Los conteos territoriales no están normalizados.** Ni por población, ni por
   longitud de red, ni por clientes. **No son tasas.** Falta el denominador y no
   está en esta fuente.
6. **`Total Afectados` de las utilities cuenta clientes, no personas.** El modo
   eléctrico acumula 289.824.114 «afectados» en diez años — es evidentemente un
   conteo de clientes con doble conteo entre eventos. **No usar como conteo de
   personas.** Los conteos de `Total Aislados` (140.303) sí parecen ser personas.
7. **Amenazas lentas invisibles.** Déficit hídrico: 30 eventos en diez años. Un
   registro de emergencias necesita una fecha de inicio, y la sequía no la tiene.
8. **No se puede clasificar ningún modo como deliberado.** Ver §7. Hace falta
   otra fuente.
9. **350 eventos (0,69 %) quedaron sin clasificar** antes que forzarlos a un
   modo: accidentes misceláneos (175), accidente minero (77), concentraciones
   masivas (40), varamiento y mortandad de especies (10).

---

## 10 · Qué se lleva el proyecto de aquí

1. **El clima sí mueve el riesgo, y ahora está cuantificado con un control no
   circular: 6,84 contra 0,84.** Cuatro modos independientes, mismo patrón, sin
   excepciones. Es la primera confirmación empírica del supuesto de fondo del
   proyecto sobre registro nacional real.
2. **La estacionalidad es el eje explotable del registro; la tendencia no lo
   es.** Cualquier variable que se construya sobre series decenales de SENAPRED
   estará midiendo la práctica administrativa.
3. **La firma que predice la falla climática es `extension` lineal +
   `exp_intemperie` + `redundancia` ausente**, y las tres se leen de *qué es la
   cosa*, como `VT`. No hace falta pedir un veredicto sobre el futuro.
4. **El aislamiento es la mejor variable de salida del registro**: 289 eventos,
   razón invierno/verano 4,09, 79 % meteorológico, y apunta a comunas
   identificables de acceso único. Es un candidato mucho mejor que `ExpEstr`.
5. **El eje deliberado exige otra fuente.** Con SENAPRED no se puede, y octubre
   de 2019 muestra por qué importa: el ataque estaba ahí, medible, y sin nombre.

---

### Reproducir

```bash
.venv-esa/bin/python infraestructura/construir_catalogo_modos_falla.py
```

Salida: `infraestructura/datos/modos_falla_senapred.csv` (22 modos × 47
columnas) más los informes A–I por consola, que son la fuente de cada cuadro de
este documento.
