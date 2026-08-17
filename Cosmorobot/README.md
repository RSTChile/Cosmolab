# CosmoRobot

Robot LEGO Mindstorms NXT programado bajo la Teoría Cosmosemiótica de
Alexis. Reutiliza la mente cosmosemiótica desarrollada y probada en **ANIMA**
(proyecto Célula_Madre), con un cuerpo nuevo: el NXT.

## Principio de organización: modular, "órganos"

Cada capacidad vive en **su propio archivo/carpeta**, con una interfaz
mínima y estable. Si algo cambia (un sensor, el motor de decisión, el
cableado físico), se cambia **solo ese archivo** — el resto no se entera.
Es la misma filosofía "genoma/organelo" de ANIMA: membrana clara, sin que
un módulo se meta en los internos de otro.

```
Cosmorobot/
├── main.py                       — ensambla el ciclo D→R→Acción→A. Sin lógica propia.
├── genoma/
│   ├── VST_Genoma.py              — motor cosmosemiótico genérico (vendorizado de ANIMA, sin modificar)
│   └── PROCEDENCIA.md             — de dónde viene y cómo mantenerlo
├── organelos/                     — cada capacidad, un archivo
│   ├── organo_ultrasonico.py      — sensor de distancia (puerto 1) — PRESENTE
│   ├── organo_eopd.py             — sensor EOPD (puerto 2) — PRESENTE
│   ├── organo_color.py            — sensor Color (puerto 3) — PRESENTE
│   ├── organo_smux.py             — multiplexor (puerto 4): touch/gyro/accel/compass — PRESENTE (protocolo implementado a mano)
│   ├── organo_cambio_total.py     — agrega la diferencia normalizada de TODOS los sensores — PRESENTE
│   ├── organo_motor.py            — motores A+C, Volante/steering (dirección diferencial manual) — PRESENTE
│   ├── organo_deliberacion.py     — LA MENTE: selección desde el pool — PRESENTE
│   └── organo_propiocepcion.py    — bienestar (adaptado de ANIMA) — PRESENTE
├── conexion_nxt/
│   ├── conexion.py                — única capa que sabe de nxt-python (USB o Bluetooth)
│   └── backend_bluetooth_nativo.py — Bluetooth sin PyBluez (socket.AF_BLUETOOTH nativo)
├── config/                        — SOLO datos, ningún organelo hardcodea nada de esto
│   ├── puertos.py                  — mapa físico del robot + MAC Bluetooth
│   ├── pool_acciones.py            — pool de Volante, slack, umbrales
│   └── escalas_sensores.py         — normalización de cada sensor para CambioTotal
├── datalog/
│   └── registrador.py             — registro de sesión a CSV
├── docs/                           — notas de diseño, referencia teórica
└── tests/
    ├── test_ultrasonico_standalone.py — prueba de hardware sin mover motores
    └── test_motor_primera_prueba.py   — prueba acotada (N ciclos, se detiene sola)
```

## Decisión de arquitectura clave: por qué Python, no NXT-G

Sesión 2026-07-09: se automatizó primero NXT-G por cursor (funcionó — ver
`Escritorio\CosmoRobot_UI_Auto\`), pero se descubrió que la mente
cosmosemiótica **ya existe, desarrollada y probada, en ANIMA** (Python). En
vez de reimplementarla en bloques gráficos, CosmoRobot reutiliza esa mente
vía `nxt-python`, que controla el mismo ladrillo NXT por USB o Bluetooth sin
reflashear firmware.

## Cómo funciona la mente (organo_deliberacion.py)

**No es un Random ponderado.** Es un argmax modulado por conflicto, con
memoria (valencia por opción) y veto episódico (si una opción llevó a
"trauma" — p.ej. gatilló la capa reactiva — su puntaje se hunde -100,
domine lo que domine la valencia). Portado línea a línea del algoritmo de
`MemoriaDeTrabajo.deliberar` en
`Célula_Madre/campo/VST_Celula_Madre_001.py`.

El conflicto que modula la exploración (`D_actual`) ahora lo alimenta
**CambioTotal** — la diferencia agregada y normalizada de TODOS los
sensores, no solo el ultrasónico (diseño original de la sesión con GPT,
corregido para normalizar cada sensor por su escala antes de sumar — ver
`config/escalas_sensores.py`).

## El pool no es una lista de acciones discretas

Es un muestreo de la variable continua **Volante** (steering, -100..100),
con la misma semántica que el bloque Mover de NXT-G: los extremos son
rotación pura, el centro es traslación recta, los intermedios son arcos.
Traslación/Rotación/Curva son la MISMA familia, no tres acciones distintas.
Implementado con dos motores independientes (ver nota de diseño en
`organo_motor.py` — se abandonó `SynchronizedMotors` de nxt-python por no
ser confiable en pruebas físicas).

## Capa reactiva (protección física, determinista)

Veto SIN pasar por la deliberación si: obstáculo frontal cerca
(ultrasónico < `UMBRAL_CRITICO_CM`) O colisión trasera (sensor de
contacto vía SMUX). Deja huella en memoria episódica (`resultado='trauma'`)
para que la capa deliberativa "sepa" que esa opción fue vetada.

## Conexión: Bluetooth por defecto (sin cable)

`nxt-python` trae un backend Bluetooth que necesita PyBluez — no compila
en este sistema. Se implementó un backend propio sobre
`socket.AF_BLUETOOTH` nativo de Python (sin dependencias extra), en
`conexion_nxt/backend_bluetooth_nativo.py`. `main()` usa Bluetooth por
defecto (`metodo="bluetooth"`); para depurar con cable, `metodo="usb"`
(requiere el driver WinUSB puesto con Zadig — ver `MEMORY.md`).

## Cómo correr

```
py -m pip install -r requirements.txt
py tests/test_ultrasonico_standalone.py   # primero: solo verifica el sensor (USB)
py main.py                                 # el robot completo, por Bluetooth, sin límite de ciclos
```

Para una prueba acotada (se detiene sola tras N ciclos):
```python
from main import main
main(max_ciclos=15)
```

## Estado (2026-07-10)

- **Programa completo**: los 6 sensores (ultrasónico, EOPD, color, touch,
  gyro, accel, compass — 7 en realidad), CambioTotal multi-sensor
  normalizado, capa reactiva con doble veto (frontal + trasero),
  deliberación con memoria y veto episódico, motor diferencial manual,
  conexión por Bluetooth sin cable.
- **Honestidad sobre confianza (LF)**: el protocolo del multiplexor SMUX
  se implementó a mano (nxt-python no lo trae) a partir del driver de
  kernel real de ev3dev — direcciones de registro con confianza alta,
  pero el formato exacto de los canales analógicos (Touch, Gyro) y la
  polaridad del Touch NO están verificados en campo. Por decisión de
  Alexis (2026-07-10): se construye completo y se corrige después con los
  datos del datalog, en vez de verificar cada pieza por separado antes de
  avanzar. Ver `MEMORY.md` para el detalle completo de qué está en cada
  nivel de confianza.

### SMUX: dos bugs reales encontrados y corregidos en campo (2026-07-10)

Primera corrida completa por Bluetooth (20 ciclos, `datalog/sesion_20260710_023505.csv`)
mostró el SMUX muerto (`smux_vivo=0`) todo el tiempo. Se descartó Bluetooth/cableado
(confirmado con USB + con el programa NXT-G del usuario corriendo datos reales en
paralelo) y se aisló a dos bugs de protocolo en `organelos/organo_smux.py`:

1. **Dirección I2C equivocada**: `BaseDigitalSensor.I2C_DEV` por defecto es `0x02`
   (correcto para Color/EOPD/etc.), pero el SMUX HiTechnic usa `0x10` — confirmado
   cruzando el driver de kernel ev3dev (`nxt_i2c_sensor_defs.c`,
   `HT_NXT_SENSOR_MUX`: dirección 7-bit `0x08` → 8-bit `0x10`) con el mismo patrón
   ya presente en `nxt-python` para otros sensores HiTechnic no estándar
   (`nxt/sensor/hitechnic.py`: `I2C_DEV = 0x10`, comentado "different from
   standard 0x02"). Con `0x02` TODA lectura daba timeout, incluso los registros
   de identificación — nunca fue un problema físico.
2. **Faltaba el comando DETECT** (`command=1`) antes de RUN. Sin él, los canales
   daban valores saturados/basura (`touch_raw=1023`, `gyro_raw=1023`,
   `accel=(-1,-1,-1)`, `compass_deg=765` — ¡fuera de rango 0-359!). Con DETECT
   agregado en `SMUX.configurar_canales()`, los valores pasaron a ser físicamente
   plausibles (`gyro_raw≈666` en reposo, `accel` con Z dominante por gravedad,
   `compass_deg` dentro de 0-359).
3. **Polaridad del Touch verificada en campo**: presionado → raw baja a ~302
   (`touch_presionado=1`); suelto → raw sube a ~558 (`touch_presionado=0`). La
   polaridad ya codificada (menor valor = presionado) y el umbral
   `TOUCH_UMBRAL_PRESIONADO=500` son correctos, sin cambios necesarios.

Aclaración del usuario: HiTechnic fabricó dos multiplexores — uno sin batería
externa (solo sensores de contacto/analógicos simples) y otro con batería externa
para sensores "lógicos" (I2C) — el robot usa este segundo tipo, consistente con el
mapa de registros ya implementado (config I2C en 0x22-0x31, datos I2C en 0x40+).

Pendiente aún: formato/endianness exacto de Gyro y Accel sin verificar contra
rotación/inclinación física real (solo se confirmó que dejaron de estar
saturados); escalas de `config/escalas_sensores.py` siguen sin calibrar.
