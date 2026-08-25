# Las 28 sub-matrices, listas para subir a SharePoint

**21-ago-2026.** Generadas con `../generar_submatrices_excel.py` a partir del
inventario del proyecto. **93.408 activos, 89.466 con coordenada**, cada uno
atado a su fila de la Matriz de Infraestructura Crítica por la columna `Ítem`.

---

## 1 · El esquema, copiado de las listas que ya existen

Se leyó de `120` (Subestaciones) y `centrales`, que comparten estructura:

```
Título · Número · Ítem · Sector · Elemento · Región · Provincia ·
Dirección · Latitud · Longitud · Responsable · Teléfono · MICR
```

`Sector` y `Elemento` **no se escribieron a mano**: se copian de la fila de la
Matriz que corresponde al `Ítem`, así que no pueden contradecirla.

### Dos columnas añadidas, y por qué

**`Comuna`**, entre Provincia y Dirección. El esquema de SharePoint no la tiene,
y la comuna es el nivel al que trabaja el Comité para la Gestión del Riesgo de
Desastres. Perderla al subir sería perder el nivel administrativo más operativo.
Si se prefiere fidelidad estricta al estilo, se borra la columna.

**`Latitud decimal` y `Longitud decimal`**, al final. Las listas existentes
guardan sólo grados-minutos-segundos, que es formato de lectura; todo el cálculo
del proyecto se hace en decimal. Conservar las dos evita que la conversión de ida
y vuelta meta error en el registro oficial.

### La conversión de coordenadas está verificada contra el dato real

Las 39 subestaciones de la lista `120` ya están en grados-minutos-segundos. Se
convirtieron sus coordenadas decimales con este mismo código y se compararon:
**38 de 39 coinciden carácter por carácter.** La única que difiere lo hace en
0,01 segundos de arco —unos tres centímetros— por redondeo del último dígito.

---

## 2 · ★ Lo único que el Excel no puede traer: la columna `MICR`

`MICR` es una **columna de búsqueda** —descrita en SharePoint como «Vínculo a la
Matriz de Infraestructura Crítica»— que apunta a la columna Título de la lista
`mic`. **Una columna de búsqueda no se puede crear importando un Excel:**
SharePoint infiere columnas de texto y de número, nunca vínculos.

Hay dos caminos, y **el segundo es mejor**:

### Camino A · Importar el Excel y agregar el vínculo después
1. Nueva lista → **Desde Excel** → subir el archivo.
2. Ya creada, agregar una columna de tipo **Búsqueda** llamada `MICR`, que
   apunte a la lista `mic`, columna `Elemento`.
3. Llenarla. Como todas las filas del archivo comparten el mismo `Ítem`, se
   selecciona la columna completa en vista de cuadrícula y se pega un solo valor.

### Camino B · Crear la lista desde una existente y pegar ★ recomendado
1. Nueva lista → **Desde una lista existente** → elegir `120` o `centrales`.
   Así el esquema llega completo, **incluida la columna de búsqueda `MICR`**.
2. Renombrar la lista y su columna Título.
3. Abrir en **vista de cuadrícula**, copiar el bloque de datos del Excel y
   pegarlo.

El camino B evita tener que reconstruir el vínculo 28 veces.

---

## 3 · Siete listas pasan el umbral de 5.000 elementos

SharePoint limita las vistas a 5.000 elementos. Las listas siguen funcionando y
los datos están completos, pero **la vista por defecto se rompe** si no se
prepara. Afecta a:

| lista | filas |
|---|---:|
| 441 Establecimientos Educacionales | 16.768 |
| 183 Telecomunicaciones | 16.669 |
| 616 Red Vial | 14.039 |
| 33 Infraestructura Sanitaria | 8.463 |
| 618 Puentes de Carreteras | 6.742 |
| 265 Establecimientos de Salud | 5.717 |
| 16 Suministro Alternativo de Agua | 5.743 |

**La solución es una sola**: indexar la columna `Región` (Configuración de la
lista → Columnas indizadas) y dejar la vista por defecto **agrupada o filtrada
por región**. Ninguna región supera los 5.000 elementos en ninguna de las siete,
así que con eso queda resuelto.

---

## 4 · Las 28, con su ítem y su prioridad ante desastres

Ver `00 INDICE.csv` para la tabla completa con conteos.

**25 de las 28 apuntan a ítems con `Pen = Muy Alta`**, o sea al frente
prioritario. Las tres que no: `618 Puentes` (Alta), `624 Aeropuertos` (Media) y
`639 Comunicación Aérea` (Media).

Tres ítems reciben más de un archivo, a propósito, porque son familias distintas
con fuentes distintas: el **353** (Cuarteles Policiales) recibe Carabineros y la
PDI; el **572** (Edificios Gubernamentales) recibe gobierno provincial, regional
y edificios públicos; el **265** (Hospitales Generales) recibe los
establecimientos de salud y el Servicio Médico Legal; el **117** recibe las
líneas de transmisión y las derivaciones.

---

## 5 · Lo que quedó fuera, y por qué

**Respaldos que duplican el mismo activo desde otro organismo.** El mismo puerto,
el mismo hospital o el mismo colegio publicado por dos entidades entraría dos
veces con dos ítems distintos. Se subió la fuente autoritativa de cada uno y se
dejó la otra como respaldo en disco.

**Familias sin fila en la Matriz**: residencias de protección de la infancia
(1.533), establecimientos de adulto mayor (881), centros públicos (1.194) y
sedes universitarias (400). Existen como catastro de SENAPRED pero la Matriz no
tiene dónde ponerlas. Hay que decidir si se les crea fila o se declaran fuera de
alcance.

**Las dos que ya existen**: `120` Subestaciones y `centrales`. No se tocaron.

---

## 6 · Dos huecos declarados en el contenido

**`Teléfono` va vacío en las 28.** Ninguna de las fuentes públicas lo entrega.
Las listas existentes sí lo traen, porque se capturaron a mano.

**`441 Establecimientos Educacionales` no reparte entre Primarias (441) y
Secundarias (442).** El directorio del Ministerio de Educación no trae el nivel
de enseñanza en los campos bajados, así que las 16.768 filas van con `Ítem = 441`.
Es un reparto pendiente, no un dato perdido: el identificador RBD está en el
archivo y permite hacerlo después.

**`117 Derivaciones de Linea` tiene 277 filas y sólo 58 con coordenada.** Se sube
igual porque los nombres sirven para pedirle las coordenadas al Coordinador
Eléctrico, que es un trámite acotado.
