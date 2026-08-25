# Infraestructura Crítica × Clima

Proyecto nuevo, autocontenido en esta carpeta. Separado de Cosmoclima: comparte
método y datos climáticos, pero es otro objeto de estudio.

**Iniciado:** 15-ago-2026 · **Director:** Alexis López Tapia

---

## La pregunta

Las lluvias de julio y agosto de 2026 mostraron cuál es el punto débil del país:
la infraestructura vial, y en general la infraestructura crítica.

La Matriz de Infraestructura Crítica del RMD 2.0 (835 elementos) sabe **qué**
importa. No sabe **cuál, dónde ni cuándo** está por fallar. Este proyecto busca
darle esas tres cosas usando datos climáticos reales.

## La idea, en una línea

`FEN` — *Fragilidad ante Eventos Naturales* — es el único eje climático de toda
la matriz, y hoy es una etiqueta estática de tres niveles: sin territorio y sin
tiempo. Una autopista tiene FEN=Alta en Arica y en Aysén, en año seco y en año
de temporal.

La propuesta:

```
FEN_efectivo(activo, lugar, mes) = FEN_base(tipo) × C_clim(lugar, mes)
```

donde `C_clim` es un coeficiente MACC (rango 0,8–1,6, ya normado en el canon)
alimentado por variables MACLIMA que se pueden calcular con datos reales:
`ANPrecip`, `InEvExtre` (subíndices EOP lluvia intensa y EAL aluvión) y
`EstHidric`.

Como **`Pen` depende de FEN en un 60 %** (0,5 directo más 0,2 vía FVT), la
prioridad ante desastres pasaría a moverse con el clima real del mes y del
lugar, en vez de ser una lista fija.

## Marco obligatorio (instrucción del 15-ago-2026)

El modelo **debe** poder trabajar en los cuatro niveles jurídico-administrativos:

```
COMUNAL (346) → PROVINCIAL (56) → REGIONAL (16) → NACIONAL (1)
```

No es un agregado posterior: es requisito de diseño. El canon ya tenía la mitad
hecha — el MCSGS **exige** declarar la `US` (Unidad de Sistema) antes de calcular
el ICSGS, con US-Nacional / US-Regional / US-Global. Lo que falta es extender esa
escalera hacia abajo: `US-Comunal` y `US-Provincial`. Es extensión del canon, no
invención.

Debe además acatar la **Ley 21.542** (reforma constitucional, art. 32 N°21 CPR,
publicada 3-feb-2023) y servir a los propósitos del **COGRID** y de **SENAPRED**
(16 Direcciones Regionales, coordina el SINAPRED).

### Dónde encaja exactamente este proyecto

La Ley 21.542 está construida para **ataques** — Fuerzas Armadas, peligro grave o
inminente. Cubre el lado `FANC → Pev / Peh` de la matriz. El lado **desastres
naturales → `FEN` → `Pen`** es territorio de COGRID/SENAPRED y **la ley no lo
cubre bien**. Este proyecto trabaja justo ese lado: el que quedó descubierto.

### El chequeo normativo dejó tres huecos

La ley define *criticidad* por cuatro sub-criterios e *impacto* por cinco. Al
cruzarlos contra el RMD:

| Hueco | Dónde falta | Quién lo tapa |
|---|---|---|
| **Resiliencia** | criterio legal de criticidad, ausente en la MICR | `FRC` del MCSGS, `IRL` #210 |
| **Interdependencia** | criterio legal de criticidad, ausente en la MICR | `FAS` y `FPI` del MCSGS |
| **Tiempo de recuperación** | criterio legal de impacto | **nadie — hay que crearlo** |

Dos de los tres los tapa el MCSGS. Eso convierte al módulo de colapso en
requisito normativo, no en lujo teórico: **es lo que vuelve la matriz compatible
con la ley.** El tercero es un hueco real (hallazgo H-11).

### Convergencia de tres líneas independientes

El paper de Castillo Jofré y Saldaña González (*Anuario de Derechos Humanos*,
UACh, 2024) critica el estatuto chileno por estar pensado para **infraestructura
de gran escala**, dejando fuera **la pequeña o aislada, que es más vulnerable**
—su caso son los APR rurales— y sitúa la falla en la **fase de normalidad**
(prevención y planificación), no en la de emergencia.

Es el mismo punto ciego al que ya habíamos llegado por otros dos caminos:

1. **Empírico:** «Carreteras Secundarias (Rutas Rurales)» queda última del sector
   Transporte con `PF = 0,41` — y es la que aísla pueblos.
2. **Teórico:** el `NGF-L` del MCSGS predice exactamente que el nodo secundario
   se vuelve crítico cuando cae el principal.
3. **Jurídico:** el estatuto real de Chile comete hoy ese error.

Y la fase donde el paper sitúa la falla —normalidad, prevención— es exactamente
donde opera una matriz predictiva movida por clima.

## Estado

| | |
|---|---|
| Levantamiento de variables y métricas | **hecho** — 17 hallazgos anotados |
| Marco normativo y niveles | **incorporado** |
| Catálogo de conceptos GRD | **hecho** — 677 conceptos |
| Catastro de fuentes nacionales | **hecho** — 6 dominios, todo verificado |
| Esquema, normalización y territorio | **hechos y probados** |
| Primera validación | **corrida — el ancla FALLÓ**, ver bitácora |
| Siguiente | Corregir rareza→peligro · resolver H-05 · snapshot diario |
| Modificaciones al RMD | **Ninguna.** Regla de Alexis (15-ago): sólo se modifica contra datos reales |

> ## ★ Correr todos los días
>
> ```
> .venv-esa/bin/python infraestructura/traer_capas_sernageomin.py
> ```
>
> La Minuta Técnica de SERNAGEOMIN **se sobrescribe a sí misma**: no guarda
> historia. Cada día sin snapshot es un día que no se puede recuperar, y sin
> historia no hay con qué validar el modelo contra lo que la fuente decía en su
> momento. Primera foto: `datos/crudo/sernageomin/2026-08-15/`.

## Archivos

```
Variables_y_Metricas_Infraestructura_Critica_Clima.xlsx   ← el entregable
construir_variables_y_metricas_proyecto.py                ← lo reconstruye desde cero
traer_clima_subestaciones.py                              ← baja el clima de los 39 puntos (reanudable)
leer_docx_con_formulas.py                                 ← lector de .docx con ecuaciones OMML
datos/clima_diario_subestaciones_erA5.csv                 ← 1990-2026 diario: lluvia, tmax, tmin
datos/subestaciones_puntos.csv                            ← los 39 puntos con coordenada decimal
fuentes/                                                  ← copias de los originales (no se tocan en su lugar)
```

El Excel tiene seis hojas: **LÉEME**, **Variables y Métricas** (30 del catálogo
de 318, con doble columna de fórmula), **MICR Columnas** (10), **MCSGS Colapso**
(10), **Niveles y Normativa** (4 niveles + 12 criterios de la Ley 21.542) y
**Hallazgos** (14).

## El piloto: 39 subestaciones

Elegidas porque son **lo único georreferenciado** que hay. Las 39 comparten hoy
exactamente la misma fila de la matriz —el ítem 120, `FEN=Alta`, `PF=0,75`,
`Pen=Muy Alta`— desde Arica hasta Punta Arenas, **35 grados de latitud**. Esa
uniformidad es el problema en estado puro, y por eso son un buen caso de prueba.

Los datos: Open-Meteo Archive (ERA5), anónimo, 1990-2026 diario, lluvia y
temperatura para cada punto. **Límite declarado:** en la ronda 17 de Cosmoclima
se comprobó que ERA5 exagera los años secos en la zona de Illapel y fabrica
meses de lluvia. Acá se usa igual porque no hay estaciones para 39 puntos en 35
grados de latitud — y en consecuencia **el piloto prueba si el método funciona,
no cuánto riesgo real corre cada subestación.**

### La prueba de fuego ya tiene fecha

El **aluvión de Copiapó del 24-25 de marzo de 2015** aparece limpio en el dato:
39,8 y 64,3 mm en dos días, sobre un lugar cuya lluvia anual normal ronda los
12 mm. Es un evento documentado y catastrófico, con daño real a infraestructura.
Si el método sirve, tiene que encenderse ahí. Si no se enciende, el método no
sirve — y eso es lo que hace que el piloto sea falsable.

### Sobre el lector de fórmulas

Las 328 fórmulas del Anexo A.5 están guardadas como ecuaciones OMML de Word, no
como texto. Cualquier extractor común (incluido `python-docx`) las salta en
silencio y deja los documentos aparentando no tener fórmulas. `leer_docx_con_formulas.py`
las recupera y las escribe en forma lineal legible.

## Precedencia de fuentes

Manda el Word `RMD_2_Variables_y_METRICAS_COMPLETAS-11-06-2026.docx`. El Excel
`Variables-y-Metricas-318-06-03-2026-Tabla.xlsx` es la versión operativa y es lo
práctico, pero donde discrepan gana el Word. Por eso cada fila del entregable
trae **las dos fórmulas en columnas separadas**: así la discrepancia queda a la
vista en vez de esconderse.

## Lo que hay que resolver antes de calcular nada

**H-05 es el bloqueador.** El Listado Blanco de MACC —la lista de variables a
las que el módulo climático puede aplicar su coeficiente— nombra 17 variables.
Sólo 3 existen en el catálogo con ese nombre (`ICS`, `ICR`, `IRDE`). Cinco no
existen (`IIEC`, `IVP`, `IDE`, `IOC`, `IPT`). Nueve existen con la misma sigla
pero significando otra cosa.

Como la Regla de Oro de MACC dice que **sólo ajusta variables que ya existen**,
hoy el módulo no tiene a qué aplicarse. Y el bloque peor afectado es justamente
el Territorial / Infraestructura (`IVT`, `IRT`, `IPT`), que es el nuestro.

Los otros nueve hallazgos están en la hoja **Hallazgos** del Excel, cada uno con
cómo se comprobó.

## Lo que aporta el MCSGS

El Módulo de Colapso Sistémico Global Sincronizado (sección 23 del Word, ausente
del catálogo de 318) trae el concepto que faltaba: **NGF-L**, nodo de flujo
*latente*. Un nodo que no es crítico en condiciones normales y se vuelve crítico
cuando cae el principal y el flujo se redirige hacia él.

Es exactamente la ruta rural secundaria: `PF = 0,41`, la última del sector
Transporte en la matriz — hasta que un aluvión corta la autopista y pasa a ser
el único acceso, y se satura. El RMD ya tenía el concepto; faltaba conectarlo
con el clima. Y su factor `FCN` se deriva de `PF + IRMD`, así que la matriz de
835 ítems alimenta el módulo de colapso sin intermediarios.

## Datos disponibles hoy

Heredados de Cosmoclima, con su límite: **calibrados para el Norte Chico**, no
para el país.

- Lluvia 1966-2026 sin reanálisis, validada contra estación medida
- Humedad de suelo satelital ESA CCI (1988-2024, máximo mensual)
- Bandas ONI El Niño / La Niña (criterio oficial NOAA)
- Temperatura ERA5 y NASA POWER
- Motor que corre 62 años día a día, con paridad navegador↔Node verificada

Georreferenciado sólo hay **39 subestaciones eléctricas** (16 regiones, con
coordenadas, operador y teléfono). Es muestra, no inventario — ver hallazgo
H-10. Para vialidad **no existe todavía** el inventario de rutas
georreferenciado; habría que traerlo del MOP / Vialidad.
