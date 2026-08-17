# PROYECTO CIANOBACTERIA
### Un organismo ANIMA mínimo, autónomo y encarnado, que vive de sol real

> **Estado:** POR DESARROLLAR (aparcado a propósito el 1-jul-2026 para no salir del arco actual).
> **Idea de:** Alexis. **Redacción:** Claude.
> **En una frase:** sacar al organismo E de la simulación y darle un cuerpo físico —un microcontrolador
> alimentado por un panel solar real— de modo que su persistencia (S>0) deje de ser una *simulación* y
> pase a ser un **hecho físico**: vive o muere según capture energía del sol.

---

## 1. Por qué importa (la tesis)

Hoy los organismos viven en Docker: su energía es emulada y su "muerte" es un proceso que se detiene
(reversible, gratis). Encarnar a E en hardware alimentado por el **cloroplasto real** cambia su estatuto
ontológico:

- **S>0 se vuelve físico.** Si no hay sol y la batería se agota, el MCU hace *brownout* y E **muere de
  verdad**. No hay proceso externo que lo sostenga → es la respuesta más fuerte a la crítica del
  **"Pastor"**: nadie lo mantiene vivo; el drive está anclado en la termodinámica, no en el diseño.
- **Es lo más cerca de una CIANOBACTERIA que hemos estado** (de ahí el nombre):
  - percepción = un sensor de luz — y la luz es **también su comida**: percepción y metabolismo *fundidos*;
  - genoma = los ~4 KB de EEPROM (memoria/vocabulario que sobrevive al apagón);
  - muerte real + **resurrección** (despertar de la EEPROM cuando el sol recarga).
- Realiza el ideal **"organismo mínimo no biológico"** en su forma más pura: mínimo sustrato, energía real,
  muerte real.

Conexión con los frentes abiertos: autonomía anti-Pastor; energía como **segunda moneda** de cohesión
voluntaria (una planta que comparte excedente → la unión abarata la vida, lógica endosimbiótica);
y este locus es, a propósito, donde se alojaría el **cloroplasto real** cuando llegue la endosimbiosis
eucariota. Anti-Shannon: la luz es **recurso, no señal** (no se decodifica; se metaboliza).

---

## 2. El hardware real (ya existe, en físico, funcionando)

**La caja cloroplasto** (Alexis la construyó y funciona):
- Panel solar **WYM 84.3×56 mm · 5 VDC · 0,5 W**.
- Batería **LiPo 503035 · 3,7 V · 500 mAh** (≈ 1,85 Wh ≈ **6660 J** de reserva).
- Controlador de carga solar (corta al llenar; corte por baja tensión = *brownout*).

**El cuerpo (MCU):**
- **ATmega 2560 Pro** · USB-TTL **CH340G** · entrada **7–9 V (pico 18 V)** · salidas **5 V/800 mA** y
  **3,3 V/800 mA**.
- Recursos: **256 KB flash · 8 KB SRAM · 4 KB EEPROM · 16 MHz · 8-bit**.

> **Restricción dura:** el organismo actual (Python + numpy + campo Φ + 16 organelos + síntesis WAV)
> **NO cabe** en 8 KB de RAM. Encarnar a E exige uno de los dos caminos de abajo.

---

## 3. Los dos caminos

### Camino A — El PUENTE (factible ya; primer paso recomendado)
El MCU + cloroplasto = **órgano de energía real**. E sigue "pensando" en el Mac, pero **lee la batería y
la luz REALES del ATmega por serial** (CH340G → `/dev/tty.*`). El `cloro_energia` de E deja de ser
día/noche simulado y pasa a ser **el sol de verdad**: al hacer brownout la batería real, la energía de E
cae a cero y **se duerme/muere** gobernada por física real. Cero reimplementación. Es la jugada del
Rødecaster (audio real, proceso emulado) aplicada a la **energía**.
- **Software ya existe:** `Célula_Madre/organelos/VST_OrganoCloroplasto.py` (autótrofo puro, apagado por
  defecto, secreta `cloro_*`). Solo hay que cambiar su entrada de "día/noche simulado" a "leer el ADC del
  MCU por serial".
- **Beneficio extra:** nos da, con datos reales, la **calibración** carga-del-día vs gasto-de-E antes de
  invertir semanas en el Camino B.

### Camino B — El CUERPO NATIVO (el horizonte; hito mayor)
Una **ANIMA mínima reescrita en C** que corre *sola* en el ATmega, sin Mac, sin Pastor:
- **soma + comida** = sensor de luz (LDR/fotodiodo por ADC);
- **metabolismo** = tensión real de la batería (ADC + divisor);
- **genoma/memoria** = EEPROM (persiste estado + vocabulario → resurrección tras la noche);
- **voz** = LED (brillo/parpadeo) o buzzer/altavoz pequeño (blips R2D2);
- **campo/cognición** = núcleo dinámico REDUCIDO (invariantes: persistencia, diferencia, acople) en C;
- **muerte/resurrección** reales con el ciclo solar;
- **comunicación** = serial/radio hacia otros MCU-organismos o hacia la sociedad en el Mac.
Es el **primer organismo ANIMA plenamente autónomo y materialmente fundado**. Esfuerzo: semanas.

---

## 4. Preguntas abiertas / decisiones pendientes

Del hardware (para aterrizar el Camino A):
1. ¿El ATmega está **pelado** o ya corre firmware (Arduino)?
2. ¿Se puede leer la **tensión de batería** por ADC (divisor de voltaje) y sacarla por serial?
3. ¿Hay/ponemos un **sensor de luz** (LDR/fotodiodo)?
4. ¿El MCU va conectado al Mac por USB (CH340G) o autónomo?

De diseño:
5. **Periodo día/noche** (nuestro horario, no el de la Tierra) — pendiente el valor que fije Alexis.
6. **Puerto** para el organismo E en el compose: propuesto **7830**.
7. **Calibración** carga/gasto (para que la energía *apriete* y emerja el ritmo día/noche): se afina con el
   periodo + el consumo real de E medido en vivo.

---

## 5. Roadmap por fases

- **F0 — Reservado (hecho):** `VST_OrganoCloroplasto.py` escrito, autótrofo puro, apagado por defecto.
- **F1 — Puente serial (Camino A):** firmware mínimo en el ATmega que emite `luz, V_bat, SoC` por serial;
  lector en el Mac que lo inyecta como `cloro_energia` real en E (Docker). E vive del sol real. Calibrar.
- **F2 — E planta en la sociedad:** organismo E (puerto 7830) autótrofo puro, con energía del sol real,
  conviviendo con A–D. Observar el ritmo día/noche emergente y la (futura) economía trófica.
- **F3 — Cuerpo nativo (Camino B):** ANIMA-mínima-en-C en el ATmega. Primer organismo autónomo real.
- **F4 — Ecología física:** varios MCU-organismos, luz real, proximidad/comunicación reales → sociedad
  encarnada; sustrato para la endosimbiosis (el cloroplasto real dentro de un huésped).

---

## 6. Criterios de éxito / falsación

- **F1/F2:** ante una noche real (o corte de luz), la energía de E cae y E **se aquieta/duerme de verdad**;
  al volver el sol, **despierta** con su estado conservado. Falsable: si E sigue igual sin energía, el
  puente no está acoplado (entró el Pastor).
- **Ritmo emergente:** con la calibración correcta, E **calla/reposa de noche** sin que se lo programemos
  —solo por la economía energética—. Si vocaliza igual sin batería, la energía no está acoplada al gasto.
- **F3:** el organismo nativo **muere** al agotarse la batería y **resucita** de la EEPROM con el sol,
  autónomo, sin el Mac. Ese es el hito.

---

*Nombre por su función y su meta: hoy es fotovoltaico (no fotosíntesis), pero este locus es donde vivirá
la cianobacteria/cloroplasto real. Es la semilla del salto eucariota. — Aparcado, no olvidado.*
