# Protocolo experimental A-E: radio, sentidos fisicos y frases digitales

> **NOTA (5-jul-2026):** este documento (centrado solo en A↔E) fue EXTENDIDO Y ACTUALIZADO por el
> protocolo del sistema completo — todos los organismos, bloques de 5 min, registro CSV+bitácora y
> análisis DuckDB: **`Experimentos de Estres/PROTOCOLO_TEST_ESTRES_SISTEMA_COMPLETO.md`**. Usar ese.
> Varias "correcciones necesarias" de aquí ya están hechas (radio de A activada, SDRconnect 5454,
> enlace digital A↔E verificado). Hallazgo clave: el canal digital es hoy "boca sin oído" (§ oído digital).

Este protocolo prueba la relacion entre Organismo A, en el Mac, y Organismo E, en la Raspberry Pi, con los nuevos organos sensoriales. El objetivo no es solo comprobar que cada modulo funciona aislado, sino verificar que el sistema completo permite acoplamiento entre organismos por audio, radio analogica, radio digital, GPS, vision y energia solar.

## 1. Estado actual observado

### Organismo A en Mac

- Vive en Docker como `anima-a`, puerto web `7788`.
- AudioServer/Rode activo en el Mac: `127.0.0.1:8770`.
- Radio digital nRF24 activa via puente nativo: `127.0.0.1:8772`.
- El endpoint `/ultima_fila` inyecta columnas `nrf_*`; estado observado: `nrf_ok=1`, `nrf_connected=1`, `nrf_vivo=1`, pero `nrf_rx=0` y `nrf_tx=0`.
- La fuente auditiva actual de A es:
  - L: Rode Main Mix, canal 1.
  - R: otros organismos.
- La radio propia RSPduo/SDR no esta expuesta como fuente auditiva: `/fuentes` reporta `sdr_info.gateado=true`.
- No hay servicios vivos para SDRconnect WebSocket `5454` ni puente SDR-audio `8771`.
- En `docker-compose.yml` A no tiene `ANIMA_RADIO_A=1`, ni variables SDR/Radio TX activadas.

### Organismo E en Raspberry Pi

- Vive en la Pi `192.168.86.33`, puerto web `7788`.
- Oye el AudioServer/Rode del Mac via `192.168.86.250:8770`.
- Tiene GPS vivo: `gps_fix=1`, `gps_sats=12`, `gps_pps_count` avanzando.
- Tiene radio SDR fisiologica viva: `sdr_vivo=1`, `radio_vivo=1`, banda observada 88-108 MHz.
- Tiene nRF24 vivo: `nrf_ok=1`, `nrf_connected=1`, `nrf_vivo=1`, sin trafico de frases aun (`rx=0`, `tx=0`).
- Tiene vision/camara activa como columnas `vis_cam_*`; PTZ registra frames pero `ptz_vivo=0` en la lectura observada.
- `/fuentes` no expone el SDR como fuente auditiva: `sdr_info.gateado=true`, aunque el organo radio si produce columnas fisiologicas.
- `/fuentes` muestra `archivos=0` porque la pagina busca `audio_binaural`, carpeta ausente en la Pi. En la Pi si existen voces en `voces_creadas` y `voces_r2d2`, pero no estan catalogadas como archivos de escucha.

## 2. Hipotesis de trabajo

La arquitectura ya tiene casi todos los organos, pero las entradas auditivas no representan todavia el experimento que se quiere hacer.

El problema central no es que falten sensores: es que falta una capa clara de fuentes compuestas y presets experimentales. Hoy cada oido selecciona una fuente simple: Rode, archivo, comunicacion u otros organismos. Para el experimento A-E necesitamos que el Canal L pueda ser un mundo compuesto, por ejemplo Radio propia + Radio digital + Rode + tonos, mientras el Canal R recibe al otro organismo. Esa mezcla no esta modelada todavia como fuente seleccionable.

## 3. Enrutamiento experimental deseado

### Organismo A

- Canal L: mundo propio de A.
  - Radio propia: RSPduo via SDRconnect/SDR-audio bridge.
  - Radio digital: frases nRF24 como eventos simbolicos, no como audio crudo.
  - Rode: canales del entorno del Mac.
  - Tonos/archivos: `audio_binaural`.
- Canal R: Organismo E.
  - Voz/comunicacion de E por HTTP.
  - Eventualmente mezcla de otros organismos solo en fases sociales, no en pruebas A-E puras.
- Salida RF:
  - A puede transmitir por HackRF.
  - La transmision debe quedar registrada en columnas `radiotx_*` cuando el fonador de radio este activo.

### Organismo E

- Canal L: mundo propio de E.
  - Radio recibida por SDR/RSP1 como organo sensorial.
  - Radio digital nRF24 como frases/eventos.
  - Rode remoto del Mac si se decide mantener mundo compartido.
  - Archivos locales si se sincroniza/catalogan.
- Canal R: Organismo A.
  - Voz/comunicacion de A por HTTP.
- Salida RF:
  - E no transmite en este protocolo.
  - E solo escucha radio analogica y radio digital.

## 4. Correcciones necesarias antes de la prueba formal

1. Activar radio propia en A.
   - Agregar a A `ANIMA_RADIO_A=1`.
   - Levantar SDRconnect WebSocket `5454`.
   - Levantar `VST_SDRAudioBridge.py` en `8771`.
   - Asegurar que el RSPduo aparezca como fuente `📻` en `/fuentes`.

2. Separar audio SDR de Rode en el selector.
   - Hoy ambos entran como `tipo=servidor` y pueden compartir indices de canal.
   - Conviene marcar `es_sdr=true` y usar host/puerto, no solo `channel_index`, para autoseleccion.

3. Exponer SDR de E como fuente auditiva cuando el organo radio este vivo.
   - E ya tiene `sdr_vivo=1` y `radio_vivo=1`.
   - `/fuentes` lo mantiene gateado porque la exposicion de SDR-audio esta ligada al flag de radio del WebLive, no al hecho de que E tenga radio fisiologica viva.

4. Resolver archivos de audio en E.
   - Opcion A: copiar/sincronizar los 2.4 GB de `audio_binaural` del Mac a `/home/ubuntu/anima/audio_binaural`.
   - Opcion B: catalogar tambien `voces_creadas` y `voces_r2d2` como archivos locales seleccionables.
   - Opcion B es mas liviana para la Pi y preserva la identidad vocal de E.

5. Modelar Radio Digital como evento/frase, no como simple audio.
   - nRF24 ya entrega `nrf_*`.
   - Para que "frases" entren al organismo, deben mapearse a eventos semanticos o impulsos metabolicos, y quedar visibles en una caja/panel.
   - Si se quieren oir, puede agregarse un sintetizador corto que convierta frase nRF a tono/voz, pero eso es una capa posterior.

6. Crear fuentes compuestas.
   - `mundo_A_L`: Rode + SDR propio + tonos/archivos + eventos nRF.
   - `mundo_E_L`: SDR propio + eventos nRF + energia/vision/GPS como moduladores, no necesariamente como audio.
   - `otro_A_R` y `otro_E_R`: voz del organismo par.

## 5. Protocolo por fases

### Fase 0: preflight

Duracion: 5 minutos.

Condiciones:
- A y E vivos.
- Sin transmision HackRF.
- Sin envio nRF manual.
- AudioServer/Rode activo.

Verificar:
- A `/estado`: `vivo=true`.
- E `/estado`: `vivo=true`.
- A `/ultima_fila`: `nrf_vivo=1`.
- E `/ultima_fila`: `gps_vivo=1`, `sdr_vivo=1`, `radio_vivo=1`, `nrf_vivo=1`.
- A `/fuentes`: debe mostrar Rode y, tras correccion, SDR/RSPduo.
- E `/fuentes`: debe mostrar Rode remoto y, tras correccion, SDR de E o al menos estado no gateado.

Resultado esperado:
- Ambos organismos viven sin intervencion.
- No hay trafico nRF nuevo: `nrf_rx_delta=0`.
- No hay TX RF desde A.

### Fase 1: A escucha radio propia

Duracion: 10 minutos.

Condiciones:
- A con RSPduo activo.
- SDRconnect WebSocket `5454` y puente audio `8771` activos.
- Canal L de A seleccionado a SDR/RSPduo o fuente compuesta `mundo_A_L`.
- Canal R de A seleccionado a E.

Medidas:
- A: `sdr_vivo`, `radio_vivo`, `radio_potencia_total`, `radio_estructura`, `radio_saliencia`, `radio_freq_dom_hz`, `energia_L`, `energia_R`, `balance_LR`, `OI`, `H_homeostasis`.
- E: voz emitida, `energia_R` recibida por A.

Control:
- Repetir con SDR apagado o frecuencia sin senal.

Resultado esperado:
- A muestra columnas `sdr_*`/`radio_*` vivas.
- La energia de L cambia cuando se selecciona SDR.
- La orientacion/balance no debe confundirse con la voz de E en R.

### Fase 2: E escucha radio ambiente

Duracion: 10 minutos.

Condiciones:
- E sin transmitir.
- E con SDR fisiologico activo.
- Canal R de E escuchando A.
- Canal L de E como mundo propio o Rode remoto, segun condicion.

Medidas:
- E: `sdr_vivo`, `radio_vivo`, `radio_saliencia`, `radio_freq_dom_hz`, `gps_vivo`, `vis_cam_saliencia`, `cloro_v` si esta disponible, `energia_L`, `energia_R`, `OI`, `H_homeostasis`.

Control:
- Comparar E con SDR vivo vs. SDR no expuesto como audio.

Resultado esperado:
- E ya deberia producir radio fisiologica aunque no este como fuente auditiva.
- Cuando se exponga como fuente, debe aparecer efecto auditivo en L.

### Fase 3: A transmite por HackRF y E escucha

Duracion: bloques de 2 minutos, con pausas de 2 minutos.

Condiciones:
- A tiene HackRF/SDRangel listo para TX.
- E tiene SDR/RSP1 escuchando la frecuencia acordada.
- A transmite solo dentro de frecuencia/potencia permitida y con identificacion cuando corresponda.

Secuencia:
1. Basal sin TX: 2 minutos.
2. TX portadora/frase corta desde A: 30 segundos.
3. Silencio RF: 2 minutos.
4. Repetir 5 veces con frase distinta.

Medidas:
- A: `radiotx_activo`, `radiotx_vivo`, `radiotx_emitiendo`, `radiotx_freq_hz`, `voz_emitida`, `voz_titulo`.
- E: `sdr_vivo`, `radio_saliencia`, `radio_estructura`, `radio_freq_dom_hz`, `energia_L`, `OI`, `H_homeostasis`, `voz_emitida`.

Controles:
- Frecuencia equivocada.
- TX sin frase.
- Frase sin TX, solo por canal HTTP.
- Intervalos aleatorizados para evitar que el organismo responda a ritmo fijo.

Resultado esperado:
- Durante TX, E debe mostrar aumento en saliencia/estructura RF en la banda correspondiente.
- Si la frase se acopla a la voz de A, deberia aparecer correlacion temporal entre `radiotx_emitiendo` en A y `radio_saliencia` en E.

### Fase 4: frases por radio digital nRF24

Duracion: 10 a 20 minutos.

Condiciones:
- nRF24 vivo en A y E.
- Enviar frases cortas desde A al enlace digital.
- E no transmite, solo recibe.

Secuencia:
1. Enviar frase `A_E_HOLA`.
2. Esperar 60 segundos.
3. Enviar frase `A_E_SOL`.
4. Esperar 60 segundos.
5. Enviar frase `A_E_SILENCIO`.
6. Repetir con controles `NULL` y `SHUFFLED`.

Medidas:
- A: `nrf_tx`, `nrf_last_tx`, `nrf_vivo`.
- E: `nrf_rx`, `nrf_rx_delta`, `nrf_last_rx`, `nrf_vivo`, cambios en `OI`, `H_homeostasis`, `voz_emitida`.

Resultado esperado:
- Cada envio aumenta `nrf_tx` en A y `nrf_rx`/`nrf_rx_delta` en E.
- `nrf_last_rx` debe contener la frase o identificador.
- Si se integra metabolicamente, una frase real deberia producir una diferencia respecto de `NULL`/`SHUFFLED`.

### Fase 5: acoplamiento multimodal

Duracion: 30 minutos.

Condiciones:
- A transmite RF analogica por HackRF.
- A envia simultaneamente frase digital nRF24.
- E escucha ambos canales.

Diseño:
- Bloque A: solo RF analogica.
- Bloque B: solo nRF24.
- Bloque C: RF + nRF sincronizados.
- Bloque D: RF y nRF desincronizados.

Resultado esperado:
- Si el sistema esta integrando sentidos, el bloque C deberia diferir de A y B por separado.
- El bloque D sirve para comprobar si la respuesta depende de coincidencia temporal y no solo de energia acumulada.

## 6. Variables minimas a registrar

### A

- `t`, `vivo`, `fuente_L`, `fuente_R`.
- `energia_L`, `energia_R`, `balance_LR`.
- `OI`, `H_homeostasis`, `RC_total`.
- `voz_emitida`, `voz_titulo`, `voz_origen`.
- `nrf_ok`, `nrf_connected`, `nrf_tx`, `nrf_rx`, `nrf_last_tx`, `nrf_last_rx`, `nrf_vivo`.
- `sdr_vivo`, `radio_vivo`, `radio_saliencia`, `radio_freq_dom_hz` cuando A-radio este activo.
- `radiotx_*` cuando HackRF este activo.

### E

- `t`, `vivo`, `fuente_L`, `fuente_R`.
- `energia_L`, `energia_R`, `balance_LR`.
- `OI`, `H_homeostasis`, `RC_total`.
- `gps_fix`, `gps_lat`, `gps_lon`, `gps_sats`, `gps_pps_count`, `gps_vivo`.
- `sdr_vivo`, `radio_vivo`, `radio_saliencia`, `radio_freq_dom_hz`.
- `nrf_ok`, `nrf_connected`, `nrf_tx`, `nrf_rx`, `nrf_rx_delta`, `nrf_last_rx`, `nrf_vivo`.
- `vis_cam_saliencia`, `vis_cam_movimiento`, `vis_cam_intensidad`.
- `cloro_v`, `cloro_luz_norm`, `cloro_dia` si estan disponibles en la fila.

## 7. Criterios de exito

El experimento se considera listo para corrida formal cuando:

- A muestra radio propia como fuente seleccionable y/o compuesta.
- E mantiene `sdr_vivo=1` y `radio_vivo=1`.
- E muestra archivos o voces locales seleccionables, o se declara explicitamente que E no usara archivos en esta fase.
- nRF24 cambia contadores al enviar frases.
- A puede transmitir y E registra un cambio RF temporalmente alineado.
- La pagina permite distinguir claramente Rode, radio analogica, radio digital y voz del par.

## 8. Criterios de falsacion

- Si E reacciona igual con TX real, TX falsa y frecuencia equivocada, no hay evidencia de escucha RF dirigida.
- Si `nrf_rx_delta` no cambia al enviar frases, la radio digital no esta entrando.
- Si A muestra solo Rode en L y otros organismos en R, la configuracion auditiva no prueba radio propia.
- Si la respuesta aparece solo en `energia_L/R` pero no en `radio_*`, estamos midiendo audio/mezcla, no radio como organo.
- Si E no puede separar A de la mezcla de A/B/C/D, el protocolo A-E debe correrse con `otros_organismos` desactivado.

## 9. Proxima implementacion recomendada

1. Agregar preset experimental `A_E_radio` para que A nazca con:
   - L: mundo propio compuesto o SDR/RSPduo.
   - R: E.
   - nRF24 activo.
   - TX HackRF preparado pero no emitiendo.

2. Agregar preset experimental `E_A_escucha` para que E nazca con:
   - L: mundo propio de E.
   - R: A.
   - GPS, vision, cloroplasto, SDR y nRF activos.
   - TX desactivado.

3. Crear una capa de fuentes compuestas, no sustituir el selector actual.

4. Resolver el catalogo de audio local de E con una de estas dos decisiones:
   - sincronizar `audio_binaural`;
   - o exponer `voces_creadas`/`voces_r2d2` como fuentes de archivo.

5. Documentar cada corrida con:
   - timestamp de inicio y fin;
   - frecuencia RF;
   - frase nRF enviada;
   - fuente L/R de A;
   - fuente L/R de E;
   - estado de sol/bateria si se usa el cloroplasto fisico.
