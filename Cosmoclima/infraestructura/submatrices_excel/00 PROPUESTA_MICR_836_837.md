# Las dos filas nuevas que la Matriz necesita — propuesta

**21-ago-2026.** Instrucción de Alexis: *«las cuatro familias que faltan estimo
que hay que crearlas»*.

**Al ir a crearlas resultó que sólo dos lo necesitan.** Las otras dos ya tenían
lugar en la Matriz y yo me había equivocado. El detalle está abajo, y las cuatro
sub-matrices ya están generadas.

---

## 1 · Dos de las cuatro NO eran huérfanas

### `Sedes universitarias` (400) — error mío

**Sí tiene fila: el ítem 443, «Universidades (Infraestructura)»**, sector
Educación, `Pen = Media`. Y hay más: el archivo trae cuatro tipos, y **dos ítems
distintos de la Matriz los cubren**:

| tipo declarado por la fuente | filas | ítem |
|---|---:|---|
| Universidad Privada · Universidad CRUCH | **184** | **443** Universidades (Infraestructura) |
| Instituto Profesional · Centro de Formación Técnica | **216** | **446** Centros de Formación Técnica |

El archivo `443 Educacion Superior.xlsx` reparte fila por fila, como hace
`centrales` con sus 14 ítems. **No hace falta ninguna fila nueva.**

### `Centros públicos` (1.194) — no son centros públicos, son supermercados

El catastro de SENAPRED que lleva ese nombre contiene UNIMARC, HIPERLIDER,
TOTTUS, EKONO, Santa Isabel y 53 malls. Corresponde al ítem **528, «Centros
Comerciales»**, que ya existe.

★ **Y resuelve un duplicado que yo no había visto:** comprobado que el archivo
`supermercados` (993) está **contenido íntegramente** en `centros_publicos`
(1.194) — cero filas exclusivas del primero. Es el mismo catastro, más completo.
Así que el ítem 528 se puebla con `centros_publicos` y el archivo de
supermercados se descarta por redundante.

---

## 2 · Las dos que sí necesitan fila nueva

Ninguna de las dos tiene equivalente en las 835 filas. Se buscó por
«adulto», «mayor», «anciano», «geriátrico», «asilo», «niñez», «infancia»,
«menores» y «hogar»: **cero resultados**.

| | 836 | 837 |
|---|---|---|
| **Elemento** | Establecimientos para Adultos Mayores (ELEAM y Centros Diurnos) | Establecimientos de Protección de la Infancia (Residencias y Programas Ambulatorios) |
| **Sector propuesto** | Protección Social | Protección Social |
| **Activos** | **881** — ELEAM 832 · Centros de Vida y Trabajo 49 | **1.533** — Residencias 254 · Líneas ambulatorias 1.279 |
| `FEN` | **Alta** | **Alta** |
| `FANC` | Media | Media |
| `IB` | 0,80 | 0,80 |
| `VTic` | 0,40 | 0,30 |
| `FVTic` **calculado** | **0,69** | **0,66** |
| `PF` = IB × FVT | 0,55 | 0,52 |
| `IRMD` | Alto | Alto |
| `Pev` | 0,7347 → **Media** | 0,7308 → **Media** |
| `Peh` | 0,6471 → **Baja** | 0,6286 → **Baja** |
| `Pen` | **0,9587 → Alta** | **0,9556 → Alta** |

### Por qué esos valores, y qué es juicio y qué no

**`FEN = Alta` es lo mejor fundado de los cuatro.** Son edificios que alojan
personas con movilidad reducida o bajo cuidado permanente: una evacuación ante
sismo, incendio o aluvión es estructuralmente lenta. Es la misma lógica por la
que 441 Escuelas Primarias y 267 Centros de Atención Primaria llevan `Alta`.

**`FANC = Media`** por analogía con 266 Clínicas Rurales y 445 Escuelas Rurales:
no son blanco deliberado típico.

**`IB = 0,80`** se ubica entre 348 Centros de Atención a Víctimas (0,80) y 267
Centros de Atención Primaria (0,70). Es **juicio declarado**, no medición.

**`VTic` 0,40 y 0,30**: baja dependencia tecnológica. Los ELEAM llevan algo más
por el equipamiento clínico de los residentes con dependencia severa.

★ **El `FVTic` va CALCULADO con la fórmula escrita, no asignado a criterio.** Es
una diferencia deliberada con el resto de la Matriz: se midió que en las 835
filas existentes el FVT no es función de sus entradas (reproduce 3 de 835). Estas
dos filas serían las primeras reproducibles. Queda declarado para que nadie
las mezcle con las otras sin saberlo.

---

## 3 · ★ Dos comprobaciones que había que hacer antes de proponerlas

### 3.1 · No disparan el problema de no estacionariedad

Añadir filas reescala las tres columnas de prioridad, porque el divisor es el
máximo observado — y ya está medido que **un ítem nuevo puede reclasificar hasta
el 50 % de la matriz**. Comprobado para éstas dos:

| | máximo actual | 836 | 837 |
|---|---:|---:|---:|
| `Pev` | 1,5490 | 1,1380 | 1,1320 |
| `Peh` | 1,9440 | 1,2580 | 1,2220 |
| `Pen` | 1,9590 | 1,8780 | 1,8720 |

**Ninguna supera ningún máximo.** Los divisores no cambian y **no se reclasifica
ni una sola de las 835 filas existentes.** Se pueden agregar sin efectos
colaterales.

### 3.2 · Pero sí rompen la rejilla, y hay que decidirlo

La Matriz es **una rejilla regular que nadie había hecho notar**: 19 sectores,
**exactamente 44 ítems cada uno**, en rangos contiguos y sin un solo hueco en la
numeración 1-835. La única excepción es Financiero, con 43.

```
Hídrico 1–44 · Represas 45–88 · Energía 89–132 · Nuclear 133–176
Telecomunicaciones 177–220 · Comunicaciones 221–264 · Salud 265–308
Servicios de Emergencia 309–352 · Seguridad 353–396 · Alimentario 397–440
Educación 441–484 · Financiero 485–527 · Comercial 528–571 · Gobierno 572–615
Transporte 616–659 · Químico 660–703 · Industrial 704–747
Industria de Defensa 748–791 · Tecnologías Informáticas 792–835
```

Agregar dos ítems la rompe de alguna forma. Las opciones:

| opción | qué conserva | qué rompe |
|---|---|---|
| **★ Sector nuevo «Protección Social», ítems 836-837** | los bloques siguen siendo contiguos; cada sector ocupa su rango | la regla de 44 por sector — que ya no era universal, Financiero tiene 43 |
| Meterlos en Salud (836-837) | el número de sectores | Salud pasa a 46 ítems **y** deja de ser contigua: 265-308 más 836-837 |
| Insertarlos dentro del bloque de Salud | la rejilla entera | obliga a renumerar 528 ítems y rompe toda referencia existente, incluida la lista `120` |

**Recomiendo la primera.** Es la única que conserva la propiedad estructural que
de verdad se usa —cada sector ocupa un rango contiguo— y sólo cede en una regla
que la propia Matriz ya incumple. Además es conceptualmente correcta: ni un ELEAM
ni una residencia de infancia son establecimientos de salud; son de protección
social, y el marco normativo chileno los trata así.

---

## 4 · Lo que NO se hizo

**No se escribió nada en la lista `mic` de SharePoint.** Las dos filas están
propuestas acá y en `00 PROPUESTA_MICR_836_837.csv`, listas para revisar y
agregar. Cuatro de sus columnas son juicio declarado y eso es decisión del
director.

**Las sub-matrices ya están generadas igual**, con los ítems 836 y 837 puestos,
para que no haya que rehacerlas después. Si los números o el sector cambian, se
regeneran los dos archivos en un minuto.

## 5 · Un hueco de contenido, declarado

**El archivo 837 mezcla dos cosas distintas.** De sus 1.533 filas, **254 son
residencias** —edificios donde duermen niños— y **1.279 son programas
ambulatorios**, que son oficinas. Ante un desastre no son lo mismo en absoluto:
una residencia hay que evacuarla de noche, una oficina no. La fuente sí trae el
campo (`LÍNEAS AMBULATORIAS` / `RESIDENCIAS`), así que **el reparto en dos ítems
se puede hacer cuando se decida**; por ahora el nombre del elemento dice
explícitamente que incluye las dos, para que nadie lo lea mal.
