# Capa Biologica vs Capa Tecnologica

Fecha: 2026-07-07  
Proyecto: VSTCosmo / ANIMA / Celula Madre  
Estado: glosario operativo para implementacion

## Proposito

Este documento separa el lenguaje biologico del organismo de los mecanismos tecnologicos que lo soportan. La separacion evita que una decision instrumental, por ejemplo `Rode`, `WordPress`, `UDP` o `Docker`, se confunda con un organo o una necesidad metabolica.

Regla rectora:

```text
La capa biologica nombra lo que el organismo es, percibe o metaboliza.
La capa tecnologica nombra como el sistema operativo, la red o el hardware lo hacen posible.
```

## Capa Biologica

La capa biologica debe aparecer en:

- nombres de organelos;
- variables metabolicas;
- eventos de vida;
- decisiones del soma;
- interfaz del Observatorio;
- documentos conceptuales del equipo.

### Conceptos Biologicos Aprobados

```text
OrganeloPresencia
OrganeloAudicion
OrganoComunicacion
Campo Social
Presencia
Ausencia
Retorno
Novedad
Aislamiento
Hambre social
Confianza
Vinculo
Afinidad
Mundo local
Silencio basal
Entrada local
Otros organismos
Plaza
Observatorio
Estado vital
Memoria episodica
```

### Estados Vitales

```text
latente
dormido
atento
activo
conversando
explorando
silencioso
ausente
```

### Eventos Biologicos

```text
nacio
desperto
durmio
descubrio_organismo
organismo_retorno
organismo_ausente
formo_vinculo
rompio_vinculo
cambio_estado_vital
cambio_visibilidad
plaza_conectada
plaza_desconectada
```

### Variables Metabolicas de Presencia

```text
presencia_vivo
presencia_vecinos_n
presencia_local_n
presencia_global_n
presencia_confiable_n
presencia_confianza
presencia_aislamiento
presencia_retorno
presencia_novedad
presencia_densidad
presencia_proximidad
hambre_social
comunicacion_foco
```

## Capa Tecnologica

La capa tecnologica debe permanecer en:

- adaptadores;
- drivers;
- scripts de despliegue;
- configuracion;
- logs tecnicos;
- implementacion de red;
- empaquetado.

### Conceptos Tecnologicos Permitidos

```text
mDNS
Zeroconf
UDP broadcast
HTTP
SSE
WebSocket
WebRTC
WordPress
FastAPI
Node
Redis
Docker
systemd
LaunchAgent
sounddevice
PortAudio
ALSA
PulseAudio
PipeWire
CoreAudio
WASAPI
Rode
BlackHole
USB audio
Ed25519
JSON Schema
nonce
ttl_s
installation_id
```

## Mapeo Correcto

| Capa biologica | Capa tecnologica posible |
|---|---|
| Presencia local | mDNS, Zeroconf, UDP broadcast |
| Plaza | Microservicio, WordPress, Redis, REST API |
| Observatorio | HTML, JS, SSE, WebSocket |
| Entrada local | sounddevice, PortAudio, ALSA, CoreAudio, WASAPI |
| Servidor remoto | VST_AudioServer, TCP, WebSocket relay |
| Confianza | Ed25519, public_key, signature, challenge-response |
| Silencio basal | buffer de ceros, ausencia de input, dispositivo no disponible |
| Campo Social | roster interno, TTL, cache de vecinos |

## Ejemplos de Uso

### Correcto

```text
El OrganeloPresencia percibe tres organismos en el Campo Social.
```

Implementacion:

```text
El adaptador mDNS encontro tres servicios _anima._tcp.local.
```

### Incorrecto

```text
El organo UDP encontro tres IPs.
```

Motivo: UDP no es organo; es mecanismo.

### Correcto

```text
El organismo escucha su mundo local.
```

Implementacion:

```text
sounddevice abrio el dispositivo USB audio con 2 canales a 48 kHz.
```

### Incorrecto

```text
El organismo necesita el Rode para vivir.
```

Motivo: Rode es un hardware posible, no una condicion biologica universal.

## Reglas para Codigo

1. Los modulos de organelos pueden usar nombres biologicos.
2. Los adaptadores deben aislar nombres tecnologicos.
3. Variables de entorno pueden nombrar tecnologia, pero deben traducirse a conceptos biologicos al entrar al soma.
4. La UI publica debe preferir lenguaje biologico, con detalles tecnicos solo en paneles diagnosticos.
5. Los logs pueden incluir ambas capas si queda claro que una es mecanismo y otra funcion.

## Regla de Revision

Antes de aceptar un cambio, preguntar:

```text
Esto nombra una funcion viva del organismo o un mecanismo que la implementa?
```

Si es funcion viva, va en capa biologica.  
Si es mecanismo, va en capa tecnologica/adaptador.

