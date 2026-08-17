# Informe: Observatorio de Cajas Modulares

Fecha: 2026-07-03

## Resumen

Se separó la visualización de datos del organismo respecto de la página principal. El organismo sigue produciendo estado por `/ultima_fila`; el observatorio ahora carga visualizadores externos desde `web/Cajas`.

`web/Cajas` es un sistema modular único y compartido por todos los organismos actuales y futuros. No hay una carpeta de cajas por organismo. Cada observador decide qué cajas activar en la página del organismo que está mirando.

La intención arquitectónica es simple:

- El organismo no se modifica para cambiar una visualización.
- Cada caja vive en su propio archivo.
- El catálogo de cajas se declara en `web/Cajas/manifest.json`.
- La disposición del observador sigue guardándose en `localStorage`, por organismo.
- Una caja puede quedar latente si el organismo observado no produce las columnas que esa caja espera.

## Carpeta Nueva

Ruta:

```text
web/Cajas/
```

Archivos principales:

```text
manifest.json
observatorio.js
observatorio.css
README_DESARROLLO_CAJAS.md
```

`README_DESARROLLO_CAJAS.md` es la nota de desarrollo permanente para el equipo.

## Inventario de Cajas Migradas

Cajas originales extraídas desde el bloque embebido:

```text
metabolismo.js
memoria.js
alteridad.js
balbuceo.js
salud.js
campo.js
homeostasis.js
voz.js
agencia.js
vozeco.js
expectativa.js
expresion.js
aprendizaje.js
```

Cajas nuevas para los sentidos físicos de E, disponibles en el catálogo global:

```text
localizacion.js
cloroplasto_fisico.js
vision.js
radio_sdr.js
```

En A/B/C/D estas cajas quedan latentes: están disponibles para el observador, pero no incorporan órganos nuevos al metabolismo ni producen datos por sí mismas. En E muestran datos porque E sí tiene esos órganos físicos integrados.

## Cambios en la Página Principal

Archivos locales actualizados:

```text
web/VST_CelulaMadre_WebLive_A.py
web/VST_CelulaMadre_WebLive_B.py
web/VST_CelulaMadre_WebLive_C.py
web/VST_CelulaMadre_WebLive_D.py
```

La página principal ya no contiene el arreglo `CAJAS` embebido. En su lugar carga:

```html
<link rel="stylesheet" href="/Cajas/observatorio.css">
<script src="/Cajas/observatorio.js"></script>
```

El servidor Python ahora sirve archivos estáticos bajo:

```text
/Cajas/...
```

También se agregó:

```text
/cam/capture.jpg
```

Ese endpoint actúa como proxy hacia la ESP32-CAM, usando por defecto:

```text
http://192.168.86.25/capture
```

Puede cambiarse con:

```text
ANIMA_ESP32_CAM_URL
```

## Estado en Raspberry Pi

Se actualizó el organismo E en:

```text
/home/ubuntu/anima/celula_madre/web/
```

Se dejó respaldo remoto en:

```text
/home/ubuntu/anima/celula_madre/web/backups_cajas_20260703_125534/
```

Después del reinicio, `/ultima_fila` confirmó:

```text
atmega_vivo = 1
gps_fix = 1
gps_sats = 12
vis_vivo = 1.0
sdr_vivo = 1
radio_vivo = 1.0
```

## Validación

Validaciones realizadas:

```text
python -m py_compile web/VST_CelulaMadre_WebLive_A.py
python -m py_compile web/VST_CelulaMadre_WebLive_B.py
python -m py_compile web/VST_CelulaMadre_WebLive_C.py
python -m py_compile web/VST_CelulaMadre_WebLive_D.py
node --check web/Cajas/*.js
python -m json.tool web/Cajas/manifest.json
```

En la Pi también se validó `py_compile` y `manifest.json`.

Nota: la prueba con navegador headless local no se ejecutó porque Playwright estaba instalado como librería, pero sin binario Chromium descargado. La validación funcional principal se hizo en la Pi con el servidor real reiniciado y los endpoints vivos.

## Regla de Desarrollo

Una caja debe vivir en un solo archivo propio. No juntar varias cajas en un archivo común.

Para agregar una caja:

1. Crear `web/Cajas/nombre_caja.js`.
2. Registrar la caja con `window.OBS_CAJAS.push(...)`.
3. Agregarla a `web/Cajas/manifest.json`.
4. Abrir la página del organismo.
5. Entrar a `Editar tablero`.
6. Pulsar `Agregar caja`.

## Próximos Pasos

- Ajustar visualmente las cajas nuevas con observación real.
- Calibrar umbrales de día/noche en `cloroplasto_fisico.js`.
- Extender el firmware del ATmega si se quiere emitir fecha/hora UTC GPS, no solo PPS.
- Decidir si las capturas de visión se conservan solo en navegador o si se guardan en disco con política de retención.
