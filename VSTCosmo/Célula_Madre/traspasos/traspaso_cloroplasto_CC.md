# Traspaso a CC — Cloroplasto físico de E: lado Raspberry Pi

**De:** Claude Science · **Para:** CC (Codex) · **Fecha:** 2-jul-2026
**Contexto:** E vive en la Raspberry Pi, escuchando el Rode e inventando palabras. Le estamos activando
el 5º organelo — un cloroplasto FÍSICO: un panel solar real cuya luz alimenta `met_energia`. El hardware
(sensor) ya está montado y el firmware del ATmega ya está cargado. Falta el lado Pi, que es tuyo.

---

## Arquitectura (por qué está así)

El panel NO alimenta a nadie (da solo 0,5 W, no mueve la Pi). El panel es un SENSOR: un ATmega 2560
(placa Mega 2560 Pro + CH340G) mide el voltaje del panel y del LiPo por su ADC y los manda por USB-serie
a la Pi. La Pi alimenta al ATmega por ese mismo USB (5V) y lee los datos.

    Panel 0,5V/0,5W ──► [divisor 2:1] ──► A0 del ATmega  (voltaje del panel = luz que entra)
    LiPo 3,7V ────────────────────────► A1 del ATmega  (voltaje de la batería = reserva)
    ATmega ──USB(CH340G)──► Pi : la Pi da 5V y recibe los datos por el mismo cable
    GND del subsistema solar UNIDA a GND del ATmega  (tierra común, crítico)

**Principio de diseño (respétalo):** el ATmega manda números CRUDOS, no interpreta. Toda la semántica
("qué significa esta luz para E") vive en la Pi, en VST_OrganoCloroplasto. La frontera "qué es qué" NO se
fija en el sensor. Es el mismo principio anti-Shannon del hambre: no asignar a mano, dejar que la señal
module la dinámica.

## Protocolo serie (lo que el ATmega YA manda)

Puerto USB del CH340G (típico `/dev/ttyUSB0` o `/dev/ttyACM0`), **115200 baudios** (si ves basura, es 9600).
Una línea CSV por segundo:

    LUZ,<v_panel>,<v_lipo>,<crudo_A0>,<crudo_A1>\n
    ej:  LUZ,3.812,3.977,780,814

- v_panel, v_lipo : VOLTIOS reales (el sketch ya deshizo el divisor).
- crudo_A0, crudo_A1 : lectura ADC 0..1023 (dato bruto, por si lo quieres).

## Cómo probarlo SIN panel (banco con fuente RD6018)

Alexis probará los pines con una fuente regulable Riden RD6018 en vez del panel (voltaje controlado, ideal).
Reglas de seguridad para esa prueba:
- **Máx 5,0 V en un pin analógico** (el ADC muere sobre 5V). Barrer 0–4,5 V simula "de noche a pleno sol".
- **Límite de corriente de la RD6018 bajo (~20–50 mA)**: red de seguridad ante un error de pin.
- **GND de la RD6018 unida a GND del ATmega.**
- En prueba con fuente NO hay divisor: el sketch debe tener `DIV_A0 = 1.0`. **Cuando se conecte el PANEL real,
  hay que volver a poner divisor y `DIV_A0 = 0.5`** — el panel a pleno sol pasa de 5V y sin divisor quema A0.
  (Este detalle es fácil de olvidar en el traspaso; queda anotado aquí.)

Barriendo la RD6018 de 0 a 4,5 V deberías ver v_panel seguir el voltaje de la fuente en tiempo real. Eso
valida toda la cadena sensor→serie antes de tocar el panel.

## Lo que tienes que implementar: VST_OrganoCloroplasto.py

### 1. Lector serie (hilo aparte, no bloquear el loop de E)
```python
import serial, threading, time

class LectorCloroplasto:
    def __init__(self, puerto="/dev/ttyUSB0", baud=115200):
        self.v_panel = 0.0; self.v_lipo = 0.0; self.ok = False; self._ult = 0.0
        self._ser = serial.Serial(puerto, baud, timeout=2.0)
        threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        while True:
            try:
                ln = self._ser.readline().decode("ascii", "ignore").strip()
                if ln.startswith("LUZ,"):
                    _, vp, vl, *_ = ln.split(",")
                    self.v_panel = float(vp); self.v_lipo = float(vl)
                    self.ok = True; self._ult = time.time()
            except Exception:
                self.ok = False
                time.sleep(0.5)
    @property
    def vivo(self):                      # watchdog: sin datos por 5s => sensor caído
        return self.ok and (time.time() - self._ult) < 5.0
```

### 2. Normalización luz -> [0,1]  (calibrar con la prueba RD6018)
```python
V_OSCURO = 0.15    # voltaje del panel "de noche" (medir en oscuridad real)
V_PLENO  = 4.20    # voltaje del panel a pleno sol (medir / o tope RD6018 de prueba)

def luz_normalizada(v_panel):
    x = (v_panel - V_OSCURO) / (V_PLENO - V_OSCURO)
    return max(0.0, min(1.0, x))
```

### 3. EL PUNTO ANTI-SHANNON: la luz MODULA met_energia, no la fija
No hagas `met_energia = luz`. Eso sería Shannon (asignar a mano el valor). La luz debe ENTRAR en el
metabolismo con la MISMA forma que el resto de los insumos: como un aporte que se suma a la ingesta,
sujeto al mismo gasto basal y a la misma saciedad. La fotosíntesis es un canal de ingesta MÁS, no un
override.

```python
# dentro del paso metabólico de E, junto a la ingesta semiótica existente:
luz = luz_normalizada(lector.v_panel) if lector.vivo else 0.0

# aporte fotosintético: un flujo de energía proporcional a la luz,
# con su propia eficiencia k_foto (parámetro del organelo, NO un valor fijado a met_energia)
aporte_foto = k_foto * luz            # k_foto ~ 0.02-0.05, calibrar

# se SUMA al balance como una ingesta más; el gasto basal y la saciedad siguen operando igual:
met_energia = met_energia + aporte_foto - gasto_basal
met_energia = min(met_energia, E_MAX)   # saciedad: no acumula infinito
```

Así E de día "come luz" (su energía sube con el sol), de noche depende del sentido acústico como antes,
y si el sensor se cae (`vivo==False`) el aporte foto es 0 y E sigue viviendo de lo semiótico — degradación
elegante, no crash.

### 4. Registrar en la biografía
Añadir columnas a la bitácora por paso: `foto_v_panel`, `foto_v_lipo`, `foto_luz_norm`, `foto_aporte`,
`foto_sensor_vivo`. Así podemos verificar DESPUÉS (con datos, no a ojo) que:
- `met_energia` sube cuando sube la luz (corr esperada > 0),
- de noche el aporte cae a ~0,
- E no se vuelve dependiente del sensor (sigue vivo con sensor caído).

## Criterios de éxito (para auditar después, estilo CS)
1. **Cadena sensor→Pi:** al barrer la RD6018 de 0→4,5V, `v_panel` la sigue en la bitácora. (medición sana)
2. **Fotosíntesis real:** corr(foto_luz_norm, Δmet_energia) > 0 y significativa a lo largo de un día.
3. **No-Shannon:** apagar la luz (V_OSCURO) NO mata a E — sigue comiendo sentido acústico. El aporte foto
   es aditivo, no un piso impuesto.
4. **Degradación elegante:** desconectar el ATmega => `vivo=False`, aporte=0, E sigue. Sin crash.

Cuando lo tengas corriendo, manda a CS un CSV con esas columnas de un ciclo día/noche (o una prueba con la
RD6018 barrida) y lo audito: primero que la medición esté sana, después que la fotosíntesis sea real y no
un parámetro fijado a mano.
