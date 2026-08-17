# Traspaso a Grok — ANIMA Pi: GPS + Radio Digital Arduino

Fecha: 2026-07-07 19:22 CLT  
Origen: Codex  
Destino sugerido: Grok / equipo VSTCosmo  
Sistema afectado: Pi ANIMA `192.168.86.22` y enlace nRF24 con Organismo E `192.168.86.33`

## Resumen ejecutivo

Quedo implementada y validada la instalacion limpia del organismo ANIMA en la Pi nueva, con organelo Arduino GPS + Radio Digital nRF24. La Pi ANIMA esta corriendo `anima-pi-runtime 0.2.10-dev`, lee el Arduino por serial, publica GPS/PPS en la pagina del organismo y puede enviar/recibir mensajes nRF24 con Organismo E.

El cambio importante de hoy fue cerrar el circuito de radio digital manual:

- ANIMA -> E: envio manual desde `/nrf/tx` funciona.
- E -> ANIMA: recepcion completa funciona; ya no se corta a la primera letra.
- La caja Radio Digital ya puede identificar backend `arduino` y `TX manual`.
- El GPS vuelve a aportar hora/fecha UTC real al metabolismo cuando tiene fix valido.

## Estado validado al cierre

Consulta realizada contra `http://192.168.86.22:7788/estado`:

```text
organismo='ANIMA'
vivo=True
atmega_vivo=1
nrf_ok=1
nrf_connected=1
nrf_backend='arduino'
nrf_tx_manual=1
nrf_vivo=1
nrf_rx=2
nrf_tx=508
nrf_last_rx='E2A2'
nrf_last_tx='DATOS_OK'
gps_fix=1
gps_sats=9
gps_lat=-32.897182
gps_lon=-70.808296
gps_time_utc='23:22:05'
gps_date_utc='2026-07-07'
gps_pps_count=217
```

Consulta realizada contra `http://192.168.86.33:7788/nrf`:

```text
nrf_ok=1
nrf_connected=1
nrf_rx=5294
nrf_tx=12
nrf_last_rx='DATOS_OK'
nrf_last_tx='ok'
```

Nota: en E, `nrf_last_tx='ok'` viene de su lector/firmware anterior. ANIMA ya fue corregido para mostrar el payload real.

## Pruebas cruzadas realizadas

### ANIMA -> E

POST:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"text":"A2E2"}' \
  http://192.168.86.22:7788/nrf/tx
```

Resultado:

```json
{"ok": true, "via": "arduino", "len": 4, "error": null}
```

Lectura rapida en E:

```text
nrf_last_rx='A2E2'
```

Despues de menos de un segundo, `nrf_last_rx` vuelve a ser `DATOS_OK` porque ANIMA transmite heartbeat automaticamente cada segundo.

### E -> ANIMA

POST:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"text":"E2A2"}' \
  http://192.168.86.33:7788/nrf/tx
```

Resultado:

```json
{"ok": true, "via": "LectorATmega", "len": 4}
```

Lectura en ANIMA:

```json
{
  "nrf_ok": 1,
  "nrf_connected": 1,
  "nrf_rx": 2,
  "nrf_tx": 59,
  "nrf_last_rx": "E2A2",
  "nrf_last_tx": "DATOS_OK",
  "nrf_tx_manual": 1,
  "nrf_backend": "arduino"
}
```

## Archivos modificados o creados

### Firmware Arduino ANIMA

Archivo:

```text
/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Célula_Madre/arduino/anima_gps_nrf24_organelo/anima_gps_nrf24_organelo.ino
```

Funcion:

- Arduino Uno con GPS NEO-M8N + nRF24L01.
- Serial hacia la Pi a `115200`.
- GPS por `SoftwareSerial`.
- PPS en pin digital.
- nRF24 en CE/CSN definidos abajo.
- Soporta transmision manual desde la Pi con comando serial `T<mensaje>\n`.
- Publica estado en lineas seriales parseables por `VST_LectorSensores.py`.

Pines del firmware:

```cpp
const uint8_t PIN_RF_CE = 9;
const uint8_t PIN_RF_CSN = 10;

const int PIN_GPS_TX = 4;
const int PIN_GPS_RX = 6;
const int PIN_GPS_PPS = 7;
```

Direcciones nRF24:

```cpp
const byte RF_ADDR_SELF[6] = "ANA01";  // ANIMA escucha aqui
const byte RF_ADDR_PEER[6] = "ANE01";  // Organismo E escucha aqui
```

Contrato serial principal:

```text
RF24_CONFIG,OK,TXMANUAL
RF24_STATUS,<connected>,GPS_LAT,<lat>,GPS_LNG,<lon>,GPS_SAT,<n>,GPS_TIME,<hh:mm:ss>,GPS_DATE,<yyyy-mm-dd>
RADIO_RX,<payload>
RFTX,ok|fail,<payload>
# PPS_PULSE_DETECTED_AT:<millis>
```

Correcciones incluidas:

- `RADIO_RX` ahora lee buffer de 32 bytes completo:

```cpp
char bufferRecibido[32] = {0};
radio.read(&bufferRecibido, sizeof(bufferRecibido));
```

- Antes el sketch de prueba leia un solo `char`, por eso E -> ANIMA llegaba como `"E"` en vez de `"E2AN"` / `"E2A2"`.
- `TXMANUAL` queda anunciado en `RF24_CONFIG`.
- `RFTX,ok,<payload>` se emite tras cada transmision.
- El firmware no publica fechas GPS invalidas tipo `2000-00-00`; solo imprime fecha si es coherente (`year >= 2020`, mes/dia validos).

### Lector Python de sensores

Archivo:

```text
/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Célula_Madre/organelos/VST_LectorSensores.py
```

Cambio importante:

- El parser de `RFTX` ahora interpreta `RFTX,ok|fail[,payload]`.
- Si llega `RFTX,ok,DATOS_OK`, `nrf_last_tx` queda como `DATOS_OK`, no como `ok`.
- Si estado es `ok`, incrementa `nrf_tx`, marca `nrf_connected=1`, `nrf_ok=1` y actualiza watchdog del lector.

Bloque conceptual:

```python
elif tag == "RFTX":  # RFTX,ok|fail[,payload]
    estado = ...
    payload = ...
    self._datos["nrf_last_tx"] = (payload or estado)[:32]
```

### Paquete instalable

Archivo generado:

```text
/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Célula_Madre/dist/anima-pi-runtime_0.2.10-dev_arm64.deb
```

Instalado en ANIMA:

```text
anima-pi-runtime 0.2.10-dev
```

El paquete incluye el firmware Arduino actualizado dentro de `Célula_Madre/arduino`.

## Comandos utiles para Grok

### Ver estado de ANIMA

```bash
curl -fsS http://192.168.86.22:7788/estado | python3 -m json.tool
curl -fsS http://192.168.86.22:7788/nrf | python3 -m json.tool
```

### Ver estado nRF de E

```bash
curl -fsS http://192.168.86.33:7788/nrf | python3 -m json.tool
```

### Enviar mensaje ANIMA -> E

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"text":"A2E2"}' \
  http://192.168.86.22:7788/nrf/tx
```

### Enviar mensaje E -> ANIMA

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"text":"E2A2"}' \
  http://192.168.86.33:7788/nrf/tx
```

### Reinstalar paquete en ANIMA si hiciera falta

Desde el Mac:

```bash
scp /Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Célula_Madre/dist/anima-pi-runtime_0.2.10-dev_arm64.deb ubuntu@192.168.86.22:/tmp/
ssh ubuntu@192.168.86.22 'sudo dpkg -i /tmp/anima-pi-runtime_0.2.10-dev_arm64.deb'
```

### Reprogramar Arduino ANIMA si hiciera falta

En ANIMA:

```bash
anima stop || true
systemctl --user stop anima-watchdog.service anima-organismo.service 2>/dev/null || true
arduino --upload --board arduino:avr:uno --port /dev/ttyACM0 /home/ubuntu/Arduino/anima_gps_nrf24_organelo/anima_gps_nrf24_organelo.ino
anima start || systemctl --user start anima-organismo.service anima-watchdog.service
```

Librerias Arduino ya instaladas en ANIMA:

```text
RF24
TinyGPSPlus
```

Puerto Arduino validado:

```text
/dev/ttyACM0
```

## Advertencias y pendientes

1. El heartbeat `DATOS_OK` de ANIMA se envia cada segundo. Esto es util como latido, pero sobreescribe muy rapido `nrf_last_rx` en E. Para observacion humana conviene agregar historial breve de ultimos mensajes o separar `heartbeat` de `mensaje_manual`.

2. E aun muestra `nrf_last_tx='ok'` en su endpoint `/nrf`; eso viene del lado E y no fue corregido en esta pasada. ANIMA si quedo corregido.

3. La fecha GPS puede tardar tras cada cold start del GPS. Al cierre estaba correcta: `2026-07-07`. Antes del fix aparecio una vez `2000-00-00`; el firmware nuevo evita publicar esa basura.

4. `gps_hdop` en ANIMA aun puede quedar en `99.0` aunque haya fix, porque el protocolo `RF24_STATUS` del sketch actual no imprime HDOP. Si se necesita calibracion fina del GPS, agregar `GPS_HDOP,<valor>` al firmware y parser.

5. El archivo `VST_LectorSensores.py` aparece como no trackeado en el `git status` local de este entorno, probablemente por la forma en que esta organizado el repo/carpeta. No revertir nada: el runtime instalado en ANIMA si tiene el cambio.

## Estado final para continuar

ANIMA puede considerarse instalado y operacional como organismo limpio con organelo GPS + Radio Digital:

- Pagina: `http://192.168.86.22:7788/`
- Estado: `http://192.168.86.22:7788/estado`
- Radio: `http://192.168.86.22:7788/nrf`
- Transmision: `POST http://192.168.86.22:7788/nrf/tx`

Recomendacion inmediata para Grok:

1. No tocar Organismo E salvo que se quiera limpiar `nrf_last_tx='ok'`.
2. En ANIMA, revisar visualmente la Caja Radio Digital y confirmar que muestre `Arduino` / `TX manual`.
3. Si el equipo quiere mostrar conversaciones, implementar historial de mensajes nRF24 separado del heartbeat.
4. Luego pasar al siguiente nivel: integrar radio digital como organelo de presencia/interorganismos, no solo como diagnostico tecnico.
