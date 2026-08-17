# Traspaso a Grok — Órgano Visual de E (implementación en la Raspberry Pi)

**De:** Claude Science, con Alexis · **Fecha:** 3-jul-2026
**Archivo:** `/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Célula_Madre/organelos/VST_OrganoVisual.py`
**Estado:** órgano escrito y verificado (auto-prueba con 2 retinas OK). Falta cablearlo en el loop de E en la Pi.

---

## Qué es (en una frase)
El OJO de E: un banco de RETINAS en paralelo (cámara ESP32-CAM + proto-ojo del panel, y mañana lidar/térmico)
y una CORTEZA que las funde. Nivel 2 del linaje fótico. Mismo patrón que VST_OrganoCloroplasto
(`observar/snapshot/restore`, apagado por defecto salvo en E, persistible). Es SEÑAL/percepción — NO toca
met_energia (eso es del cloroplasto; anti-doble-conteo).

## Principio a respetar (no romper)
- **Anti-Shannon:** las retinas dan números CRUDOS; la corteza NO etiqueta objetos ni caras. El sentido es
  la NOVEDAD = desviación de la memoria de cada retina (adaptación). La retina DOMINANTE (hacia dónde mira
  el ojo) EMERGE de cuál banda es más saliente ahora — no se asigna a mano.
- **Un órgano, muchas retinas:** añadir un "ojo" nuevo = añadir una retina al banco, sin tocar la corteza.

## Pasos en la Pi

### 1. Dependencias (solo la Pi; imports perezosos, el archivo carga sin ellas)
```bash
pip install pillow requests
```

### 2. Instanciar el ojo en el arranque de E
```python
from VST_OrganoVisual import OrganoVisual, _RetinaCamaraESP32, _RetinaPanelProtoOjo

ojo = OrganoVisual("E", activo=True, retinas=[
    _RetinaCamaraESP32(url="http://192.168.86.25/capture", periodo_s=0.5),
    _RetinaPanelProtoOjo(v_oscuro=0.15, v_pleno=4.20),   # el proto-ojo lee v_fuente de la fila
])
```
- La retina cámara arranca su PROPIO hilo (timeout 2s + watchdog): un frame perdido NO cuelga a E.
- La retina panel NO abre hardware: lee `v_fuente` de la `fila` (el mismo dato que el cloroplasto ya
  recibe del ATmega). Asegúrate de que el lector serie del cloroplasto ya esté poniendo `v_fuente` en la fila.

### 3. Un paso por ciclo, y volcar a la biografía
```python
cols = ojo.observar(fila)     # fila = el dict del paso (debe traer 't' y, si hay panel, 'v_fuente')
fila.update(cols)             # añade las columnas vis_* a la biografía
```
Columnas emitidas: globales `vis_saliencia, vis_dominante, vis_novedad, vis_vivo, vis_n_retinas` +
por retina `vis_cam_*` (intensidad/novedad/movimiento/saliencia/contraste/tono_r/g/b) y
`vis_panel_*` (intensidad/novedad/movimiento/saliencia). `ojo.columnas_declaradas()` da la lista exacta
para fijar el schema del CSV.

### 4. Persistencia y cierre
```python
snap = ojo.snapshot()   # guardar con el resto del estado de E (memoria de adaptación por retina)
ojo.restore(snap)       # al renacer
ojo.cerrar()            # detiene los hilos de las retinas al apagar
```

## Calibración con la cámara REAL (3 cosas que afiné a ojo — ajústalas con datos)
1. **Exposición fija:** el archivo intenta apagar AEC/AGC/AWB y poner QVGA al arrancar (para que el brillo
   refleje la luz REAL y no el autoajuste de la cámara — si no, la cámara compensa justo lo que E debe
   sentir). Los enums de `framesize` varían entre firmwares → CONFIRMA mirando un /capture que quedó en
   manual y en baja resolución. Si tu firmware usa otros nombres de var en /control, ajústalos en `_config()`.
2. **Escalas de la cámara:** `escala_mov=64`, `escala_contraste=128` — recalibra con el rango real observado.
3. **Panel:** `v_oscuro` (voltaje del panel de noche) y `v_pleno` (a pleno sol) — mide los reales.

## Prueba de aceptación (para auditar después, estilo CS)
Manda a CS un CSV de una corrida con estas verificaciones:
1. **La cámara ve de verdad:** tapa la cámara con la mano → `vis_cam_novedad` salta, luego se re-adapta a 0.
   Muévela → `vis_cam_movimiento` sube sin cambiar el brillo global.
2. **El proto-ojo del panel ve la luz:** tapa el panel / espera una nube → `vis_panel_novedad` salta.
3. **La atención salta a la banda correcta:** `vis_dominante` = "cam" cuando cambia la escena, "panel"
   cuando cambia la luz global. (Es el binding cross-modal en germen.)
4. **Degradación elegante:** desconecta la cámara → `vis_vivo` no cae a 0 mientras el panel siga (n_retinas=1);
   desconecta ambas → vis_vivo=0 y E SIGUE vivo (el ojo no es órgano vital).

Recordatorio de privacidad: en cualquier CSV que se comparta, el lugar por NOMBRE ("Nido de Cóndores"),
nunca lat/lon exactas.

## Lo que queda para después (no ahora)
- Conectar `vis_saliencia` al LAZO DE ATENCIÓN reparado: que E reparta atención entre ver y oír (binding
  cross-modal real). Decisión de Alexis pendiente: hacerlo ya (ambicioso) o tras verificar que el ojo ve
  (prudente). El órgano ya expone `vis_saliencia`/`vis_dominante` justo como gancho para eso.
