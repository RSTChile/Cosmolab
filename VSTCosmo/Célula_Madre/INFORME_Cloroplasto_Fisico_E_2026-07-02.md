# Informe de traspaso - Cloroplasto fisico de E

Fecha: 2026-07-02  
Equipo: Organismo E - Planta / Celula Madre  
Estado: hardware validado en banco y listo para integracion de software

## Resumen ejecutivo

El cloroplasto fisico de E ya no es solo un sensor de voltaje solar. El montaje validado contiene tres canales reales:

1. Energia solar/bateria por ATmega 2560: voltajes directos en ADC.
2. GPS/PPS por modulo u-blox NEO-M8N: posicion, calidad de fix, satelites y pulso temporal.
3. Vision por ESP32-CAM OV2640: captura JPEG por WiFi.

La frontera semantica debe mantenerse en la Raspberry Pi. El ATmega y la ESP32-CAM entregan senales fisicas; la Pi decide como esas senales modulan el metabolismo, orientacion, biografia y conducta de E.

## Arquitectura validada

```text
Subsistema solar/bateria/controlador
        |
        +--> ATmega A0/A1  -> USB serial -> Raspberry Pi
        |
        +--> GPS NEO-M8N   -> ATmega Serial2 + D2 PPS
        |
        +--> ESP32-CAM     -> WiFi HTTP -> Raspberry Pi
```

En montaje actual, la ESP32-CAM funciona alimentada internamente desde 5V/GND de la ATmega, sin la base MB. La base MB queda como herramienta de programacion/debug, no como parte necesaria del montaje final.

## Hardware: ATmega

Placa: ATmega 2560 Pro con CH340G.  
Firmware actual: `/Users/alexis/Downloads/cloroplasto_E/cloroplasto_E.ino`

Conexiones activas:

```text
A0       <- salida positiva del controlador/energia solar, directo, max 5V
A1       <- lectura de bateria/LiPo, directo, max 5V
GND      <- tierra comun del subsistema solar
Serial2  <- GPS NMEA
D2       <- GPS PPS
USB      <- enlace serial con la Raspberry Pi
```

GPS:

```text
GPS TX  -> RX2 del ATmega
GPS RX  -> TX2 del ATmega, opcional
GPS PPS -> D2
GPS GND -> GND comun
GPS VCC -> alimentacion adecuada del modulo
```

Validacion observada:

```text
GPS fix      = 1
satelites    = 10-12
HDOP         = 0.87-1.34
PPS          = cuenta incremental estable, ~1 pulso/s
nmea_seen    = incremental
ubicacion    = validada en Nido de Condores
```

## Protocolo serial ATmega -> Raspberry Pi

Puerto observado en Mac:

```text
/dev/cu.usbserial-1460
```

En Raspberry probablemente sera:

```text
/dev/ttyUSB0
```

Baudios:

```text
115200
```

El firmware emite una muestra por segundo, con dos tipos de linea:

```text
LUZ,<v_fuente_V>,<v_lipo_V>,<adc_A0>,<adc_A1>
GPS,<fix>,<sats>,<hdop>,<lat>,<lon>,<alt_m>,<speed_kn>,<course_deg>,<pps_count>,<pps_age_ms>,<nmea_seen>
```

Ejemplo real:

```text
LUZ,3.524,3.084,721,631
GPS,1,12,0.87,<lat>,<lon>,772.0,0.01,20726.00,241,399,483
```

Campos `LUZ`:

```text
v_fuente_V  voltaje directo en A0
v_lipo_V    voltaje directo en A1
adc_A0      lectura cruda ADC 0..1023
adc_A1      lectura cruda ADC 0..1023
```

Campos `GPS`:

```text
fix          1 si hay posicion valida, 0 si no
sats         numero de satelites usados/visibles segun NMEA
hdop         dilucion horizontal, menor es mejor
lat/lon      coordenadas decimales
alt_m        altitud en metros
speed_kn     velocidad en nudos
course_deg   rumbo en grados, ver nota de robustez
pps_count    contador de pulsos PPS recibidos en D2
pps_age_ms   edad del ultimo PPS en ms
nmea_seen    contador de frases NMEA recibidas
```

Nota de robustez: el parser actual usa tokenizacion C simple. En algunas frases NMEA con campos vacios puede desalinear `course_deg`, visto como `20726.00`. Para produccion, el equipo deberia corregir el parseo preservando campos vacios o ignorar `course_deg` hasta robustecerlo. Los campos criticos ya validados son `fix`, `sats`, `hdop`, `lat`, `lon`, `alt_m`, `pps_count` y `nmea_seen`.

## Hardware: ESP32-CAM

Placa: ESP32-CAM con OV2640, probada sin base MB.  
IP actual:

```text
http://192.168.86.25/
```

Endpoints validados:

```text
GET /status   -> JSON de configuracion
GET /capture  -> JPEG 320x240
```

`/stream` respondio 404 en el firmware actual, por lo que la integracion inicial debe usar `/capture` a intervalos.

Prueba validada:

```text
/capture genero JPEG valido 320x240
brillo promedio observado: ~130.5
RGB promedio observado: ~129.8, 131.4, 130.5
```

La ESP32-CAM puede entregar a la Pi un canal visual crudo:

```text
VISION,<brightness>,<mean_r>,<mean_g>,<mean_b>,<motion>,<contrast>
```

No se recomienda que la ESP32-CAM etiquete objetos o imponga significado. Para mantener el principio anti-Shannon, debe entregar magnitudes visuales basicas y dejar la semantica a E.

## Integracion recomendada en la Raspberry Pi

Crear o extender:

```text
VST_OrganoCloroplasto.py
VST_OrganoVision.py
```

### Lector ATmega

Responsabilidades:

```text
abrir puerto serial 115200
auto-detectar /dev/ttyUSB* o /dev/ttyACM*
reconectar si cae el dispositivo
parsear lineas LUZ y GPS
mantener ultimo estado con timestamp
exponer sensor_vivo por watchdog
```

Estado sugerido:

```python
cloroplasto = {
    "v_fuente": float,
    "v_lipo": float,
    "adc_A0": int,
    "adc_A1": int,
    "gps_fix": bool,
    "gps_sats": int,
    "gps_hdop": float,
    "gps_lat": float,
    "gps_lon": float,
    "gps_alt_m": float,
    "gps_speed_kn": float,
    "pps_count": int,
    "pps_age_ms": int,
    "nmea_seen": int,
    "serial_vivo": bool,
}
```

### Vision ESP32-CAM

Responsabilidades:

```text
leer http://192.168.86.25/capture
decodificar JPEG
calcular brillo promedio
calcular RGB promedio
calcular contraste simple
calcular movimiento por diferencia entre frames
exponer vision_viva por watchdog
```

Estado sugerido:

```python
vision = {
    "brightness": float,
    "mean_r": float,
    "mean_g": float,
    "mean_b": float,
    "contrast": float,
    "motion": float,
    "capture_viva": bool,
}
```

## Integracion metabolica de E

No hacer:

```python
met_energia = luz
```

Hacer:

```python
luz_norm = normalizar(v_fuente)
aporte_foto = k_foto * luz_norm * (1.0 - met_energia / E_MAX)
met_energia = min(E_MAX, met_energia + aporte_foto - gasto_basal)
```

La luz es una ingesta mas, no un override del metabolismo.

GPS/PPS no debe convertirse en "significado" fijo. Usos recomendados:

```text
gps_fix/sats/hdop     -> confianza espacial
lat/lon/alt           -> anclaje biografico del lugar
pps_count/pps_age     -> reloj externo / latido temporal
```

Vision no debe clasificar objetos inicialmente. Usos recomendados:

```text
brightness            -> intensidad visual
motion                -> perturbacion/cambio
mean RGB              -> tonalidad ambiental
contrast              -> textura visual
```

## Bitacora recomendada

Agregar columnas por paso:

```text
foto_v_fuente
foto_v_lipo
foto_adc_A0
foto_adc_A1
foto_luz_norm
foto_aporte
gps_fix
gps_sats
gps_hdop
gps_alt_m
gps_pps_count
gps_pps_age_ms
gps_nmea_seen
vision_brightness
vision_motion
vision_contrast
vision_mean_r
vision_mean_g
vision_mean_b
sensor_serial_vivo
sensor_vision_vivo
```

Evitar registrar coordenadas exactas en logs publicos o compartibles si no es necesario. Para reportes humanos usar el nombre del lugar: Nido de Condores.

## Criterios de exito para la integracion

1. Serial estable: `LUZ` y `GPS` llegan por al menos 30 minutos sin bloquear el loop de E.
2. Degradacion elegante: si se desconecta ATmega, E sigue vivo sin crash.
3. Fotosintesis real: `foto_aporte` aumenta con `v_fuente` o luz normalizada.
4. GPS validado: `gps_fix=1`, `gps_sats>0`, `nmea_seen` creciente, `pps_count` creciente.
5. Vision validada: `/capture` responde, `brightness` y `motion` cambian ante cambios reales.
6. Anti-Shannon: ningun sensor fija directamente estados internos de E; todos modulan flujos/dinamicas.

## Tareas inmediatas para el equipo

1. Implementar lector serial robusto para `LUZ` y `GPS`.
2. Implementar lector HTTP `/capture` para ESP32-CAM.
3. Agregar columnas de bitacora.
4. Integrar `aporte_foto` al metabolismo como ingesta aditiva.
5. Integrar vision como senal continua de brillo/movimiento/color.
6. Corregir parser GPS o ignorar `course_deg` hasta robustecer campos NMEA vacios.
7. Probar corrida de 10-30 minutos con el organismo activo en Nido de Condores.

