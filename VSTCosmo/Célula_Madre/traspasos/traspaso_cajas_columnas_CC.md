# Traspaso a CC — Cajas de sentidos: contrato de columnas (para la Pi)

**De:** Claude Science · **Fecha:** 3-jul-2026
**Qué es:** las 4 cajas de los sentidos físicos de E ya están registradas en `web/Cajas/manifest.json` y
verificadas (cero campos mudos: cada columna que la caja lee, algún órgano/lector la produce). Esta nota dice
QUÉ columnas debe traer `/ultima_fila` en la Pi para que cada caja pinte. Si una falta, esa fila queda en "—"
(la caja NO se rompe: degrada).

## Cambios hechos hoy (software, verificado)
- `localizacion.js` reescrita: arreglado el campo muerto (`gps_pps_age` → `gps_pps_count`, que es lo que el
  lector produce) y añadido el bloque **sentido** (loc_*) sobre el dato crudo (gps_*). Ahora muestra lo que
  hace de la localización un ÓRGANO y no un GPS: novedad de lugar, confianza, deriva del reloj.
- `vision.js` reescrita: arreglado el campo muerto (`vis_cam_intensidad`, que el órgano NO emite → ahora usa
  el promedio de `vis_cam_tono_r/g/b`) y añadido el bloque **sentido** (vis_saliencia, vis_novedad).
- `cloroplasto_fisico.js` y `radio_sdr.js`: ya calzaban, sin cambios.

## Contrato por caja (columnas que el lector/órgano debe poner en la fila)
**cloroplasto_fisico** ← OrganoCloroplasto + lector ATmega:
  foto_luz_norm, foto_v_panel, foto_v_lipo, foto_adc_a0, foto_adc_a1, foto_sensor_vivo

**localizacion** ← OrganoLocalizacion (sentido) + lector GPS (crudo):
  loc_desplazamiento, loc_novedad, loc_confianza, loc_altitud_rel, loc_pps_deriva, loc_vivo
  gps_fix, gps_lat, gps_lon, gps_sats, gps_hdop, gps_alt, gps_pps_count

**vision** ← OrganoVisual:
  vis_saliencia, vis_novedad, vis_dominante, vis_n_retinas, vis_vivo,
  vis_cam_tono_r, vis_cam_tono_g, vis_cam_tono_b, vis_cam_movimiento, vis_cam_contraste, vis_cam_novedad
  (+ endpoint `/cam/capture.jpg` para el thumbnail — ya existe como proxy a la ESP32-CAM)

**radio_sdr** ← OrganoRadio + lector SDR:
  radio_saliencia, radio_estructura, radio_novedad, radio_potencia_total, radio_n_bandas,
  radio_freq_dom_hz, radio_vivo, sdr_espectro, sdr_vivo

## Nota
En A/B/C/D estas cajas quedan latentes (no hay esos órganos) — es correcto, la caja muestra "—". En E (Pi)
deben pintar porque los órganos están integrados. En A, si se activa ANIMA_RADIO_A=1 y el lector SDR entrega
datos, solo la caja radio_sdr cobra vida.
