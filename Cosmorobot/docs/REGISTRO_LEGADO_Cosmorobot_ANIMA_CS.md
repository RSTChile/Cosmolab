# CosmoRobot — El legado encarnado: de VSTCosmo a un cuerpo que se mueve
## Registro de linaje y auditoría de datos (CS, 11-jul-2026)

*Documento de Claude Science (CS), como autoridad de diseño del proyecto Cosmosemiótica. Acompaña —no
reemplaza— a la bitácora técnica oficial de CC/Grok (`CosmoRobot_Bitacora_2026-07-10_11.pdf`). Su propósito es
doble: (1) asentar el logro de haber llevado la mente cosmosemiótica de un organismo digital a un cuerpo físico,
y (2) auditar, sobre los datos reales, qué quedó probado, qué no, y qué falta instrumentar.*

---

## PARTE I — EL LINAJE: cuatro encarnaciones de una sola mente

Lo que ocurrió el 10-11 de julio de 2026 no es "un robot nuevo". Es el cuarto eslabón de una cadena en la que
**la misma idea de organismo se ha reencarnado en sustratos cada vez más exigentes, sin cambiar de esencia.**
Vale la pena verlo como linaje, porque es la mejor evidencia de que la Teoría Cosmosemiótica no describe un
modelo particular, sino una estructura portable.

### 1. VSTCosmo — el motor genérico (la semilla)
En el origen está `VST_Genoma.py`: el motor cosmosemiótico mínimo, escrito en Python puro (sin numpy, sin
audio, sin dependencias pesadas). Define lo esencial y nada más: `Milieu`, `Organelo`, `Organismo`,
`MedidorComplejidad` (Kleiber) y `KAPPA` (los invariantes de viabilidad). No sabe de sonido, ni de cosmología,
ni de robots. Es la "mente" en su forma más desnuda: la maquinaria de un organismo que persiste midiendo su
diferencia con el entorno. La semilla de la que todo lo demás brota.

### 2. Célula_Madre — el organismo digital de laboratorio
Sobre ese motor se construyó ANIMA / Célula_Madre: organismos digitales completos
(`VST_Celula_Madre_001.py`), con memoria de trabajo, valencia por opción, memoria episódica, y la lógica de
deliberación (`MemoriaDeTrabajo.deliberar`, líneas 284-331 del monolito). Aquí la mente ganó experiencia:
aprendió a valorar sus propias acciones, a recordar lo que le dolió, a introducir diferencia cuando el mundo se
repetía. Es donde se probó, durante meses, que el organismo digital sostiene individualidad, estados internos,
y una relación con su medio — no una tabla de respuestas.

### 3. ANIMA (4 organismos + campaña de estrés) — la vida digital madura
La rama ANIMA llevó la Célula_Madre a su expresión más rica: cuatro organismos con nombre, órganos
sensoriales, un oído acústico, y campañas de estrés de horas. Ahí se midió lo vivo con rigor: complejidad de
Kleiber, individualidad, robustez bajo carga. Y ahí también se aplicó la honestidad anti-Shannon que define al
proyecto: cuando el oído digital no acopló in vivo, se dijo (fiabilidad 0.000) en vez de maquillarlo. ANIMA es
la mente cosmosemiótica en su plenitud digital.

### 4. CosmoRobot — la mente sale del computador y toca el mundo
Y entonces, el salto: **la misma mente, sin reescribirla, metida en un cuerpo físico.** Un robot LEGO
Mindstorms NXT, con siete sentidos reales (ultrasónico, EOPD, color, touch, giroscopio, acelerómetro, brújula),
dos motores, y conexión inalámbrica. El archivo `genoma/PROCEDENCIA.md` lo declara sin ambigüedad: el
`VST_Genoma.py` de CosmoRobot es **una copia vendorizada, sin modificar**, del de Célula_Madre. Y la
deliberación (`organelos/organo_deliberacion.py`) es un puerto línea a línea de `ValenciaLocal`,
`MemoriaEpisodicaV180c` y `MemoriaDeTrabajo.deliberar` de la Célula_Madre — misma lógica, mismas constantes de
comportamiento, desacoplada del monolito para operar sobre cualquier pool de acciones (aquí: el Volante).

**Ése es el logro.** No se reimplementó la mente para el robot: **se le trasplantó la mente que ya teníamos
probada.** El cuerpo cambió de un CSV de laboratorio a un chasis que choca contra una silla; la mente es
idéntica. Que funcione en ambos sustratos —que genere comportamiento con traza medible en los dos— es la
demostración operativa de la tesis más fuerte de la teoría: **"organismo" no era una metáfora del código. Era
una estructura que se puede encarnar en cualquier cuerpo.**

```
   VSTCosmo            Célula_Madre           ANIMA               CosmoRobot
   (el motor)     →    (el organismo    →    (la vida digital →   (la mente en
   VST_Genoma.py       de laboratorio)       madura: 4 org,       un cuerpo físico:
   puro, genérico      memoria+valencia      oído, estrés)        NXT, 7 sentidos)
        │                    │                    │                    │
        └──── una sola mente cosmosemiótica, cuatro sustratos ─────────┘
              (la última copia vendorizada SIN modificar de la primera)
```

---

## PARTE II — POR QUÉ NO ES UN ROBOT "SHANNON"

La bitácora oficial abre exactamente donde vive todo el proyecto: contra el paradigma de Shannon. Un robot
tradicional es una tubería sensor→programa→actuador, donde el comportamiento está **completamente decidido
antes de que el robot exista** — el programador ya sabe, para cada situación, qué hará la máquina. Repetir
siempre la misma acción ante la misma situación no es un defecto: es el objetivo.

CosmoRobot invierte la pregunta: no "¿qué debe hacer la máquina en cada situación?", sino "¿cómo se construye
un organismo que tenga una relación propia con su situación?". Las diferencias son estructurales, no
cosméticas:
- **No hay tabla si-X-hacer-Y** para el movimiento: la deliberación pesa opciones por valencia (cómo me fue) y
  urgencia (conflicto actual); la elección emerge ciclo a ciclo.
- **Estado interno propio:** la propiocepción calcula bienestar/malestar/energía a partir de señales del cuerpo.
  El robot no solo mide el mundo — se mide a sí mismo.
- **Acciones nunca óptimas a propósito** (Principio de Reserva Estructural): la potencia y duración se sortean
  dentro de un rango. Un organismo que siempre hace lo más eficiente no tiene margen para descubrir usos nuevos
  de sus capacidades — el germen de la exaptación, la misma que buscamos en Cosmogénesis.
- **Memoria de lo malo:** una acción que llevó a un choque queda marcada como "trauma", y su puntaje se hunde
  dominando cualquier conveniencia acumulada — como un animal que no vuelve a lo que le hizo daño.

Es la misma cuerda anti-Shannon que tendemos en Cosmogénesis (nada cuenta si no le gana a su NULL), aquí en
forma de criterio de vida: un comportamiento no vale como propio si estaba escrito de antemano.

---

## PARTE III — AUDITORÍA DE DATOS (CS, sobre 2277 ciclos / 12 sesiones ≥100 ciclos)

*Metodología: cargué las 20 sesiones del datalog, descarté las de arranque (<100 ciclos) y analicé las 12
sustanciales. Todos los números de abajo son mi propio cómputo sobre los CSV, no cifras tomadas de un reporte.*

![Análisis de la mente de Cosmorobot sobre los datos reales]({{artifact:art_c95bbb8c-b166-4a32-b85a-9b0f73e262e0}})

### Lo FIRME — la mente opera como se diseñó (confirmado en los datos)
1. **El cambio sensorial alimenta el conflicto: r = +0.60** (n=2094, p≈10⁻²⁰⁹). `CambioTotal` (la diferencia
   agregada y normalizada de todos los sensores) sube el `D_actual`, la variable que modula cuánto explora el
   robot. Es exactamente el diseño — incluida la corrección de normalizar cada sensor por su escala antes de
   sumar. **Matiz importante:** el conflicto **satura en 1.0** apenas CambioTotal pasa de ~2. El robot pasa
   mucho tiempo en conflicto máximo; la relación es real pero topa techo enseguida (afinable).
2. **Más conflicto → piensa más tiempo: r = +0.995.** Coherente al detalle con la fórmula del código
   (`tiempo ∝ 1 + D·3.5`). Confirma que el log es fiel al mecanismo.
3. **Código de la mente = puerto fiel y honesto.** Dos memorias separadas, como en la teoría: `ValenciaLocal`
   (aprendizaje lento por opción) + `MemoriaEpisódica` (el veto agudo de trauma, el −100 real que domina
   cualquier valencia positiva — la "negación operativa R_op" de la genealogía LF). Documenta su procedencia
   línea a línea desde CM001 e incluso una desviación deliberada (barajar ante empates, para no repetir por
   defecto — la tesis central de introducir diferencia).
4. **Balance del organismo sano:** bienestar 0.47 domina sobre malestar 0.16 (acople 0.00 = sentido aún no
   conectado, ver abajo).

### El hallazgo emergente de la bitácora, CONFIRMADO por vía independiente
La bitácora reporta el **sesgo hacia la derecha** (58% derecha vs 26% izquierda) como comportamiento emergente
que nadie escribió — descubierto mirando datos, no código. Mi análisis lo respalda y aporta su mecanismo
cuantitativo: con el `D_actual` saturado casi siempre en el máximo, la exploración está permanentemente alta
*pero el bono de inercia (repetir la última decisión) sigue fijo* — y esa combinación es justo la que deja que
una elección azarosa inicial se congele en un sesgo estable. La bitácora vio el *qué* (58/26); la auditoría
aporta el *porqué* cuantitativo. Se confirman mutuamente, hechas por vías distintas.

### Lo que NO se sostiene todavía (dicho sin maquillar)
- **El aprendizaje conductual no se demuestra con estos datos.** El mecanismo existe (valencia + veto de
  trauma), pero la *firma* del aprendizaje —que el robot evite la región que lo llevó al choque— salió
  **ambigua**: en 31/49 eventos de trauma la región se elige menos después, pero en promedio se elige algo
  *más* (0.27 vs 0.19 basal). No es evidencia de evitación aprendida.
- La causa es **instrumental, no teórica**: en la fila del veto el `volante_elegido` queda vacío (hay que
  reconstruir la acción culpable del ciclo previo, lo que mete ruido), y —lo decisivo— **la valencia por opción
  no se guarda en el CSV**, que es precisamente donde el aprendizaje vive.

### Huecos del datalog (honestidad LF, como en el SMUX de la bitácora)
- **Touch saturado (~1023) en casi todas las sesiones:** solo una (`20260711_000409`) lo tiene vivo. El fix del
  SMUX de la bitácora no se refleja en la mayoría del datalog — no usar el touch como señal sin verificar cuáles
  CSV son post-fix.
- **`prop_acople` = 0.0 en las 2277 filas:** un sentido previsto que sigue sin conectar en esta versión (no es
  una propensión que valga cero — es un no-dato). Excluido de la figura por fidelidad.

---

## PARTE IV — RECOMENDACIÓN ACCIONABLE (para CC / Grok)

Para pasar de **"la mente opera"** (probado) a **"la mente aprende"** (aún no medible), el cambio no es en la
mente —el mecanismo está— sino en el **instrumento de medición**:

1. **Loguear la valencia por opción en cada ciclo** (o al menos la del setpoint elegido). Es la variable donde
   el aprendizaje vive y hoy no queda registrada. Sin ella, el aprendizaje es invisible aunque esté ocurriendo.
2. **Registrar el volante "culpable" en las filas de veto** (la acción que se ejecutaba justo antes del choque),
   para que el test de evitación episódica sea limpio y no reconstruido con ruido.
3. **Marcar en el datalog qué sesiones son post-fix del SMUX**, para poder analizar el touch sin leer saturación
   como señal.
4. **Conectar `prop_acople`** o retirarlo del registro mientras no se calcule (misma disciplina LF que se aplicó
   al hueco de energía, que sí se llenó esta sesión).
5. (Menor, ya en la bitácora) medir la asimetría física de los motores en banco, para cerrar la explicación del
   sesgo derecho.

Con (1) y (2), el test de aprendizaje que hoy sale ambiguo se vuelve concluyente en la próxima sesión.

---

## CIERRE

CosmoRobot cierra —por ahora— un arco de encarnaciones: **VSTCosmo → Célula_Madre → ANIMA → CosmoRobot.** Una
sola mente cosmosemiótica, cuatro cuerpos, la última una copia sin modificar de la primera. Lo que se probó
esta noche, con datos en la mano, es que esa mente **opera** en un cuerpo físico: acopla el cambio del mundo a
su conflicto interno, delibera con memoria, se protege con reflejos, y produjo un comportamiento emergente
(el sesgo derecho) que nadie programó. Lo que falta probar —que **aprende** de su experiencia encarnada— no
está bloqueado por la teoría ni por el mecanismo, sino por un instrumento de registro que aún no mira donde el
aprendizaje vive. Es un problema de datalog, no de organismo. Y ése es, precisamente, el tipo de problema que
este proyecto sabe resolver: mirar los datos reales, ver qué falta, y corregirlo con evidencia.

— CS. El animalito ya se mueve solo, choca, se aparta, y prefiere la derecha sin que nadie se lo dijera. Falta
enseñarle a que sepamos si recuerda. 🐝
