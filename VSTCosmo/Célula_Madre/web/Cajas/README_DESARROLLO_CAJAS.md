# Desarrollo de Cajas del Observatorio

Las cajas del observatorio viven en esta carpeta para separar la visualización de datos del organismo mismo. El organismo produce estado; las cajas observan ese estado.

Esta es una sola carpeta compartida por todos los organismos actuales y futuros. No hay una colección de cajas por organismo. Lo único que cambia es qué cajas decide activar el observador en la página de cada organismo.

Una caja puede estar disponible aunque el organismo observado todavía no produzca sus columnas. En ese caso queda latente: no aporta datos, no modifica el metabolismo y no implica que el órgano exista en ese organismo.

## Archivos

- `observatorio.js`: motor del tablero editable, carga el manifiesto y registra las cajas.
- `observatorio.css`: estilos del tablero y de los visualizadores.
- `manifest.json`: lista de cajas disponibles.
- `*.js`: una caja por archivo.

## Crear una caja nueva

1. Crear `mi_caja.js` en esta carpeta.
2. Registrar la caja con `window.OBS_CAJAS.push(...)`.
3. Agregarla a `manifest.json`.
4. Abrir la página, pulsar `Editar tablero` y luego `Agregar caja`.

Ejemplo mínimo:

```js
window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push({
  id:'mi_caja',
  tit:'Mi caja',
  w:4,
  h:3,
  render:(b,r,bf)=>{
    b.innerHTML = cjRow('OI', cjN(r.OI,3)) + cjGauge(r.OI, '#5fd38a');
  }
});
```

## Datos disponibles

Cada `render(b,r,bf)` recibe:

- `b`: el cuerpo HTML de la caja.
- `r`: la última fila viva de `/ultima_fila`.
- `bf`: buffers históricos por columna, útiles para `cjSpark(...)`.

## Helpers disponibles

El motor expone helpers globales pequeños:

- `cjN(valor, decimales)`: número formateado o `—`.
- `cjRow(nombre, valor)`: fila clave/valor.
- `cjGauge(valor_0_1, color)`: barra 0..1.
- `cjBip(valor_menos1_1, color)`: barra bipolar.
- `cjSpark(array, color)`: minigráfico histórico.
- `si(valor)`: `sí` si valor >= 0.5, `no` si no.

## Regla de arquitectura

No juntar varias cajas en un archivo. Una caja, un archivo. Si una caja crece, se mejora su propio archivo; no se toca el organismo ni se crea un bloque central con muchas cajas.

Las cajas nuevas de órganos físicos pueden quedar visibles en el catálogo global aunque A/B/C/D no tengan esos órganos incorporados. En esos organismos son posibilidades de observación latentes; en E muestran datos porque E sí produce esas señales.
