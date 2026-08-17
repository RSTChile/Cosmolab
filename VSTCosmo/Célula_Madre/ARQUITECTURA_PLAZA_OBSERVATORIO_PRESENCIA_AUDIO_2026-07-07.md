# Arquitectura Plaza, Observatorio, Presencia y Audio Local

Fecha: 2026-07-07  
Proyecto: VSTCosmo / ANIMA / Celula Madre  
Estado: propuesta de arquitectura para el equipo antes de implementar

## Resumen Ejecutivo

El sistema debe dejar de depender de una topologia fija de laboratorio, en la que los organismos conocen de antemano a sus pares y el audio del mundo proviene del Rode del Mac. Para poder instalar organismos en Raspberry Pi, Mac o PC de terceros, necesitamos separar cuatro planos:

1. **Organelo de Presencia**: expresion basica del organismo como entidad propia. No es una utilidad de red. Es el organelo mediante el cual cada organismo dice "soy yo", anuncia que esta vivo y percibe a otros organismos.
2. **Plaza**: registro/ecologia publica de organismos presentes. Puede vivir inicialmente en Geografia Sagrada.
3. **Observatorio**: ventana publica, solo lectura, alimentada por la Plaza. No controla organismos ni expone paneles privados.
4. **Audicion / entrada de mundo**: debe priorizar el audio real local de cada equipo y tratar servidores remotos como fuentes opcionales, no como dependencia universal.

La meta es pasar de una configuracion fija de 5 organismos a una ecologia dinamica: cualquier organismo instalado correctamente puede aparecer en la Plaza, ser observado y, si las reglas lo permiten, entrar en relacion con otros.

## Problema Actual

### Organismos conocidos por configuracion

Hoy los organismos conocen a otros por variables como `ANIMA_OTROS_URLS`. Eso funciona en el laboratorio, pero falla cuando:

- aparece un sexto organismo;
- un organismo vive en otra red;
- cambia la IP;
- el organismo esta apagado;
- un usuario externo instala su propio animalito;
- queremos que "Otros organismos" signifique presencia viva y no una lista estatica.

### Audio acoplado al laboratorio

En el Mac actual, el mundo sonoro entra por `VST_AudioServer.py` y el Rode en `8770`. Eso es correcto para nuestro laboratorio, pero una instalacion limpia en Pi/PC/Mac no puede asumir:

- que existe un Rode;
- que existe el Mac servidor;
- que `host.docker.internal:8770` es accesible;
- que el usuario desea escuchar nuestro laboratorio.

El organismo nuevo debe nacer escuchando su **mundo local**: microfono, entrada USB, Pulse/PipeWire/ALSA/CoreAudio, BlackHole/loopback, o silencio basal si no hay entrada disponible.

### Observatorio fijo

El Observatorio de conversacion actual (`http://localhost:9100/`) expresa bien el espiritu visual del sistema, pero esta pensado para un conjunto conocido de organismos. Debe transformarse en una vista dinamica de los organismos presentes en la Plaza.

## Decision Conceptual

El descubrimiento de organismos no debe ser tratado como una funcion utilitaria. Debe implementarse como un organelo nuevo:

## Organelo de Presencia

Nombre propuesto: `OrganeloPresencia`

Funcion biologica/sistemica:

- expresar identidad;
- anunciar presencia;
- percibir presencia ajena;
- detectar ausencia, retorno, silencio y continuidad;
- entregar al metabolismo una senal social;
- alimentar al organo de comunicacion con un roster vivo.

Variables/metabolitos propuestos:

- `presencia_vivo`: 0/1, estado propio anunciado;
- `presencia_vecinos_n`: numero de organismos detectados;
- `presencia_local_n`: vecinos en LAN;
- `presencia_global_n`: vecinos en Plaza remota;
- `presencia_confiable_n`: vecinos conocidos/verificados;
- `presencia_ultimo_contacto_s`: edad de la ultima senal recibida;
- `presencia_confianza`: confianza agregada del roster;
- `presencia_aislamiento`: senal creciente si no hay otros;
- `presencia_retorno`: pulso cuando reaparece un organismo conocido;
- `presencia_novedad`: pulso cuando aparece un organismo desconocido.

Este organelo no decide por si mismo a quien escuchar. Entrega presencia al soma y al organo de comunicacion. La relacion efectiva sigue siendo modulada por metabolismo, experimento y reglas de intimidad.

## Dos Escalas de Presencia

### 1. Presencia Local

Para organismos en la misma red:

- UDP broadcast o mDNS/Bonjour;
- deteccion rapida;
- sin servidor publico;
- util para cajas, laboratorio, sala de exposicion o varios dispositivos en una casa.

Propuesta inicial: partir con UDP broadcast por simplicidad y depuracion.

Mensaje local de ejemplo:

```json
{
  "tipo": "anima.presence.v1",
  "id": "ANIMA_E_PI",
  "nombre": "Nido de Condores",
  "host": "192.168.86.33",
  "puerto": 7788,
  "version": "2026.07",
  "capacidades": ["audio_local", "gps", "cloroplasto", "radio_digital", "camara"],
  "endpoints": {
    "estado": "/estado",
    "voz": "/comunicacion/bloque.wav",
    "cabeza": "/cabeza",
    "identidad": "/identidad"
  },
  "ts": "2026-07-07T12:40:00-04:00"
}
```

### 2. Presencia Global

Para organismos en otras redes, ciudades o paises:

- cada organismo publica un heartbeat hacia la Plaza;
- la Plaza mantiene un roster publico/semipublico;
- el Observatorio lee ese roster;
- la comunicacion viva puede usar canales directos o relay.

Geografia Sagrada puede servir como primera sede publica de la Plaza:

- sitio publico: `https://geografiasagrada.cl/`
- posible ruta del Observatorio: `/observatorio-anima/`
- posible API WordPress: `/wp-json/anima/v1/...`

Importante: WordPress no debe ser el cerebro del sistema ni un canal de streaming intensivo. Debe ser una plaza/directorio/observatorio. Para audio, video o conversacion de baja latencia se debe usar WebSocket, WebRTC, MQTT/NATS o un relay separado.

## Plaza

La Plaza es un registro vivo, no un controlador.

Responsabilidades:

- recibir anuncios de organismos;
- validar identidad basica;
- mantener presencia y ultima senal;
- publicar un roster filtrado;
- permitir que el Observatorio renderice una ecologia dinamica;
- no exponer controles privados;
- no iniciar/detener organismos;
- no almacenar secretos de usuario;
- no obligar a publicar ubicacion real.

Endpoints propuestos:

```text
POST /wp-json/anima/v1/heartbeat
GET  /wp-json/anima/v1/organismos
GET  /wp-json/anima/v1/organismos/{id}
POST /wp-json/anima/v1/evento
GET  /wp-json/anima/v1/observatorio/feed
```

Heartbeat global propuesto:

```json
{
  "schema": "anima.presence.v1",
  "id": "ANIMA_NIDO_CONDORES",
  "nombre": "Nido de Condores",
  "instalacion": "pi",
  "version": "2026.07",
  "estado": "vivo",
  "publico": true,
  "capacidades": ["audio_local", "gps", "cloroplasto", "wifi", "radio_digital"],
  "endpoints_publicos": {
    "estado_resumen": "relay/plaza",
    "voz_resumen": "relay/plaza"
  },
  "ubicacion": {
    "modo": "aproximada",
    "pais": "CL",
    "region": "Valparaiso"
  },
  "public_key": "base64...",
  "ts": "2026-07-07T12:40:00-04:00"
}
```

## Observatorio Publico

El Observatorio debe mantener el espiritu visual actual, pero cambiar su fuente de verdad.

Antes:

```text
Observatorio conoce A/B/C/D/E por configuracion.
```

Despues:

```text
Observatorio consulta la Plaza y renderiza organismos presentes.
```

Principios:

- solo lectura;
- sin botones de start/stop/configuracion;
- no publicar IPs privadas;
- no exponer paneles internos;
- mostrar "presente", "silencioso", "ausente reciente", "latente";
- mostrar capacidades y organelos activos/latentes;
- mostrar conversacion dinamica segun organismos conectados;
- aceptar que puede haber 0, 1, 5, 6 o 100 organismos.

Vista conceptual:

```text
Plaza Geografia Sagrada
  ├─ Organismo A / Mac / vivo
  ├─ Organismo B / Mac / vivo
  ├─ Organismo C / Mac / vivo
  ├─ Organismo D / Mac / vivo
  ├─ Organismo E / Pi organelo fisico / vivo
  └─ Organismo F / Pi nueva / vivo

Observatorio
  └─ renderiza roster + estados publicos + conversacion
```

## Audicion y Mundo Local

La entrada de audio debe dejar de ser "Rode por defecto" y convertirse en un sistema dinamico de fuentes.

Orden recomendado para una instalacion limpia:

1. **Otros organismos**: fuente social, construida desde el roster del Organelo de Presencia.
2. **Entrada local del equipo**: microfono, USB audio, Pulse/PipeWire/ALSA/CoreAudio, loopback.
3. **Servidor remoto opcional**: Rode del Mac, SDR bridge, AudioServer externo.
4. **Biblioteca experimental**: audios `.wav` categorizados.
5. **Demos internos**: diagnostico, no primera opcion visible.

Variables futuras:

```text
ANIMA_AUDIO_MODE=auto|local|server|silent
ANIMA_AUDIO_LOCAL_POLICY=prefer-system-input|prefer-usb|manual
ANIMA_AUDIO_SERVER_OPTIONAL=1
ANIMA_AUDIO_LIBRARY_PROFILE=minimal|full|custom
ANIMA_PRESENCE_MODE=local|plaza|both|off
ANIMA_PLAZA_URL=https://geografiasagrada.cl/wp-json/anima/v1
```

En Raspberry Pi limpia:

```text
ANIMA_AUDIO_MODE=local
VST_DISABLE_DIRECT_AUDIO=0
ANIMA_FUENTE_DEFECTO=auto:entrada_local
ANIMA_ESCUCHAR_TODOS=auto
ANIMA_MUNDO_CANAL=auto
```

En Mac laboratorio:

```text
ANIMA_AUDIO_MODE=server
VST_SERVIDOR_HOST=host.docker.internal
VST_SERVIDOR_PORT=8770
ANIMA_MUNDO_CANAL=0
```

## Biblioteca `audio_binaural`

La carpeta actual pesa aproximadamente 2.4 GB. No debe incluirse completa en un paquete base instalable.

Propuesta:

- paquete base: audios esenciales y livianos;
- paquete experimental opcional: biblioteca completa;
- manifiesto `audio_manifest.json` con nombre, categoria, duracion, tamano, etiquetas y lateralidad;
- UI por categorias y busqueda, no lista desplegable plana;
- demos internos ocultos bajo "Diagnostico / Pruebas".

Categorias iniciales:

- `otros_organismos`
- `entrada_local`
- `servidores_remotos`
- `voces`
- `ambientes`
- `tonos`
- `musica`
- `experimentos_grandes`
- `diagnostico`

## Seguridad e Intimidad

Reglas basicas:

- no guardar credenciales en el repo;
- usar tokens o application passwords limitados para publicar heartbeats;
- no exponer `/start`, `/stop`, `/config` ni paneles privados en la Plaza;
- no publicar IP privada salvo en presencia local;
- ubicacion GPS debe ser opcional y configurable como exacta, aproximada o desactivada;
- cada organismo debe tener una clave/identidad propia;
- el usuario decide si su organismo es publico, privado o solo experimental.

Modos de visibilidad:

```text
privado        = no publica en Plaza
local          = solo LAN
plaza_latente  = aparece vivo, sin datos sensibles
plaza_publico  = aparece con estado publico
experimento    = aparece y acepta relacion con organismos autorizados
```

## Plan de Implementacion Propuesto

### Fase 1: Documento y contrato

- fijar nombres: Plaza, Observatorio, OrganeloPresencia, Audicion;
- definir JSON de identidad, heartbeat y roster;
- definir que campos son publicos y cuales privados.

### Fase 2: Organelo de Presencia local

- crear `organelos/VST_OrganoPresencia.py`;
- endpoint `/identidad`;
- endpoint `/presencia`;
- UDP broadcast local;
- roster local con TTL;
- fuente "Otros organismos" construida desde roster.

### Fase 3: Observatorio dinamico local

- modificar Observatorio de conversacion para leer roster;
- eliminar supuesto fijo de 5 organismos;
- renderizar cualquier numero de organismos;
- mantener modo solo lectura.

### Fase 4: Audio local dinamico

- habilitar `tipo:"dispositivo"` como fuente primaria en Pi limpia;
- crear descubrimiento y seleccion automatica de entrada local;
- relegar demos a diagnostico;
- agrupar biblioteca de archivos por manifiesto.

### Fase 5: Plaza en Geografia Sagrada

- crear API WordPress minimal para heartbeat y roster;
- crear pagina publica de Observatorio;
- publicar solo resumen seguro;
- autenticar organismos con token limitado o firma.

### Fase 6: Comunicacion remota

- evaluar relay WebSocket para estado/voz simbolica;
- evaluar WebRTC para audio/video directo;
- mantener Plaza como registro, no como cerebro.

## Criterio Rector

Un organismo instalable no debe nacer preguntando "donde esta el Rode del laboratorio". Debe nacer preguntando:

1. quien soy;
2. que cuerpo tengo;
3. que mundo local puedo oir;
4. que otros organismos estan presentes;
5. que relaciones estan permitidas.

La Plaza permite que esa pregunta escale desde una caja local hasta una biosfera distribuida.

