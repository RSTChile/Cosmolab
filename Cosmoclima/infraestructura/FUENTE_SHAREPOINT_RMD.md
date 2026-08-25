# El sitio SharePoint del RMD — dónde están de verdad las submatrices

**21-ago-2026.** Alexis indicó el sitio `rstchilecom.sharepoint.com/RMD`.
Acceso por Microsoft Graph, autenticado como `alexis.lopez.tapia@rst-chile.com`.

**Respuesta corta a «¿dónde están las submatrices?»: en SharePoint, y el
proyecto no las estaba mirando.** Hay dos, y traen exactamente la columna que a
mis archivos les faltaba.

---

## 1 · Lo que hay en el sitio

26 listas. Las cuatro que importan para este proyecto:

| lista | nombre real | filas | qué es |
|---|---|---:|---|
| **`mic`** | Matriz de Infraestructura Crítica | **835** | la Matriz completa, con todas sus columnas |
| **`120`** | Subestaciones | **39** | **submatriz del ítem 120** — las mismas 39 del piloto |
| **`centrales`** | centrales | **147** | **submatriz multi-ítem**, 14 ítems de Energía |
| `variables` | Matriz de Variables y Métricas | 252 | 110 variables + 142 métricas del RMD |

Copiadas a local: `datos/micr_sharepoint.csv` ·
`datos/submatriz_sp_120_subestaciones.csv` · `datos/submatriz_sp_centrales.csv`
· `datos/variables_sharepoint.csv`.

---

## 2 · ★★★ El formato canónico de submatriz existe, y lo tenía delante

Las dos listas comparten esquema, y la columna que las hace submatrices es
**`Ítem`**: el número de la fila de la Matriz a la que pertenece cada activo.

```
Ítem · Sector · Elemento · <nombre> · Región · Provincia · Dirección
     · Latitud · Longitud · Responsable · Teléfono
```

Ejemplos reales:

```
Subestación Arica (Urbana) · Ítem 120 · Energía · Subestaciones Eléctricas
   Av. Capitán Ávalos, Arica · -18° 28' 40.80" S · -70° 17' 49.20" W
   CGE · +56 800 800 767

Ralco · Ítem 95 · Energía · Represas Hidroeléctricas
   -37° 59' 57.1" S · -71° 31' 10.6" W
```

**Dos cosas que corrigen lo que yo venía haciendo:**

1. **Una submatriz puede cubrir varios ítems.** `centrales` tiene 147 activos
   repartidos en **14 ítems distintos**: represas hidroeléctricas (10, ítem 95),
   centrales de pasada (50, ítem 97), carbón (26, ítem 89), gas natural (8),
   diésel (8), solares (20), eólicos (12), geotérmica (1), biomasa (2),
   cogeneración (4), baterías BESS (2), micro-hidro (2), biogás (1),
   concentración solar (1). O sea: **la unidad no es «un archivo por ítem» sino
   «cada fila declara su ítem»**, que es justo lo que yo propuse ayer sin saber
   que ya estaba decidido.
2. **Las coordenadas van en grados, minutos y segundos**, no en decimal. Mis
   77.300 activos están en decimal. Hay que decidir cuál manda antes de fusionar
   —y la conversión no es trivial de leer a ojo, así que conviene guardar las
   dos.

---

## 3 · ★★★ SharePoint y el Excel NO son la misma Matriz

Comparadas las 835 filas de `mic` contra
`Matriz_Infraestructura_Critica_Prioridad_Estrategica_HOMOLOGADA.xlsx`, columna
por columna:

| columna | filas distintas de 835 |
|---|---:|
| Elemento (el nombre) | **0** |
| FEN · FANC · IB · VT · FVT · PF · IRMD | **0 · 0 · 0 · 0 · 0 · 0 · 0** |
| Pev | 53 |
| Peh | 101 |
| **Pen** | **680** |

**Las siete columnas de cálculo coinciden exactamente.** Todo el trabajo hecho
sobre el FEN se sostiene sin tocar una coma.

**Las tres columnas de priorización no.** Y `Pen` —que es la que ordena qué se
atiende primero— difiere en **680 de 835 filas, el 81 %**. El Excel tiene un
`Pen` aplastado; SharePoint tiene uno repartido de verdad:

| `Pen` | Excel | **SharePoint** |
|---|---:|---:|
| Muy Alta | 112 | **100** |
| Alta | 647 | **67** |
| Media | 76 | **148** |
| Baja | — | **444** |
| Muy Baja | — | **76** |

★ **Corrección a lo que yo reporté ayer:** dije «los 112 ítems de `Pen = Muy
Alta`». Con la versión viva de SharePoint son **100**. El frente prioritario del
sub-proyecto de submatrices es de 100 ítems, no de 112.

**Y esto vuelve a confirmar la regla del proyecto**: la fuente institucional
manda sobre la copia local. Igual que el Word oficial mandó sobre el Excel
degradado en agosto, ahora SharePoint manda sobre el Excel homologado.

---

## 4 · ★★★ El hallazgo del FEN no sólo sobrevive: se vuelve más fuerte

Con el `Pen` bueno, el cruce queda perfecto:

| FEN | → Pen | filas |
|---|---|---:|
| **Baja** | **Muy Baja**, siempre | 76 de 76 |
| **Media** | Baja (444) o Media (148) | 592 |
| **Alta** | Alta (67) o **Muy Alta (100)** | 167 |

**El FEN parte el `Pen` en tres bandas que no se tocan.** Un elemento con
`FEN = Media` **no puede** llegar a `Pen = Alta`, nunca, haga lo que haga el
resto de la matriz. Y los **100 ítems del frente prioritario son 100 de 100
`FEN = Alta`**.

Con la matriz que yo tenía esto se veía como una correlación fuerte. Con la
matriz de verdad es una **partición**: el FEN no influye en la prioridad, la
determina. Lo que hace que recalcularlo con dato real —lo que se hizo ayer— sea
más importante, no menos.

---

## 5 · La lista `variables` es de otro proyecto

252 entradas (110 variables + 142 métricas), y su estructura de ficha es
exactamente la que el proyecto viene usando: Sigla · Nombre · Tipo · Categoría ·
Fórmula · Descripción Técnica · Descripción No Técnica · Tipo de Dato ·
Dependencias · Fuentes · Fundamentación Teórica · Metodología y Validez ·
Rango de Factores de Ajuste · Instrucciones de Uso · Sugerencias de
Visualización · Ejemplos Históricos · Relaciones con Otras Variables ·
Limitaciones Conocidas.

**Pero ninguna es de infraestructura.** Las categorías son METPOL (62), MACH
(34), Social (30), Cultural (25), Política (25), Económica (15), Tecnológica
(15), METCOMH (14), METCOL (10), METINTEL (8), METECO (7), METLID (7). Se
verificó sigla por sigla: **no existen `FEN`, `FANC`, `FVT`, `IB`, `VT`,
`ICSGS`, `MACC` ni ninguna de las nuevas** (`PelPre`, `CClimP`, `FENef`,
`PFef`).

O sea: las variables de la Matriz de Infraestructura Crítica **no están en esa
lista**; viven en el Word. Si las nuevas del proyecto tienen que quedar
publicadas, hay que decidir si van ahí —con esa misma estructura de ficha, que
ya cumplen— o en una lista aparte.

---

## 6 · Lo que esto cambia en el plan

1. **El mapeo `item_micr` que ayer propuse construir a mano ya está decidido por
   el canon**: cada fila declara su ítem, y una submatriz puede cubrir varios.
   Mis 36 archivos tienen que adoptar esa columna con el número, no con el
   nombre de categoría.
2. **`centrales` da 147 activos ya atados a 14 ítems**, con responsable y
   coordenada. Es un puente directo entre el inventario que bajé (1.208
   centrales del Coordinador Eléctrico) y la Matriz: se puede cruzar por nombre
   y heredar el ítem.
3. **El frente prioritario es de 100 ítems**, no 112.
4. **Hay que revisar el resto de mis documentos** donde escribí «112».

## 7 · Lo que NO hice

- **No escribí nada en SharePoint.** Todo fue lectura.
- **No convertí las coordenadas** de grados-minutos-segundos a decimal ni al
  revés. Antes de fusionar hay que decidir cuál manda.
- **No revisé las otras 22 listas** del sitio (Lecturas, EPP, ARAUCANIA,
  CONFLICTOS, METPOL, Taxonomía, Sitios, Capturas, Agentes…). Son de las otras
  matrices del RMD y no de infraestructura, pero conviene mirarlas antes de
  suponerlo.

Relacionado: `SUBMATRICES_Y_EL_FEN_CONSTANTE.md` · `FEN_MEDIDO.md` ·
`SUBPROYECTO_SUBMATRICES.md` · `INVENTARIO_GEORREFERENCIADO.md`
