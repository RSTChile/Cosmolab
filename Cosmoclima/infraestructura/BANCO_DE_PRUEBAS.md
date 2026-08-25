# Banco de pruebas: someter la infraestructura a ataques reales, no imaginados

**16-ago-2026.** Idea del director: *«someter la lista de infraestructura a
ataques virtuales (simulados), de todo tipo, naturales y deliberados, con datos
o ejemplos reales, y así robustecer mucho la evaluación».*

Este documento fija **cómo** se hace para que sirva, y sobre todo **cómo no** se
hace.

---

## 1 · El giro que decide si esto funciona

Un agente que **imagina** ataques es un generador de ficción. Produce escenarios
plausibles, bien escritos, y sin ninguna forma de saber cuáles son reales. Si se
alimenta la matriz con eso, el resultado se ve sofisticado y no vale nada — peor
aún, se vuelve incontrastable, porque un escenario inventado no se puede refutar
con datos.

**Chile ya tiene el registro de lo que efectivamente falla:**

| registro | tamaño | estado |
|---|---|---|
| SENAPRED, emergencias por comuna 2015-2024 | **50.457 eventos** | ✅ bajado |
| — de ellas, fallas eléctricas | 10.150 | ✅ |
| — cortes de agua | 1.269 | ✅ |
| — conectividad vial | 311 | ✅ |
| — eventos con personas **aisladas** | 289 (140.303 personas) | ✅ |
| ReTeRM SERNAGEOMIN, remociones evaluadas en terreno | 376 (1996-2026) | ✅ bajado |
| MOP, emergencias viales | 6.141 (desde 2015) | 🟡 verificado, sin bajar |
| SEC, clientes sin luz por comuna **y por hora** | ≥6 años | 🟡 en curso |

**Casi todo lo que uno querría "simular" ya ocurrió y quedó escrito.** Entonces
el banco no inventa el ataque: lo **encuentra en el registro** y le pregunta a
nuestra estructura si lo habría visto venir.

La simulación entra sólo en un lugar legítimo: **recombinar condiciones que ya
se dieron por separado** para ver qué pasa si coinciden. Eso no es inventar — es
preguntar por la conjunción de hechos documentados.

---

## 2 · Las tres reglas que hacen válida una prueba

**Regla 1 · El criterio se escribe antes.** Enunciado, predicción, umbral y nulo
quedan por escrito antes de mirar el resultado. Si después falla, se reporta el
fallo. Esta regla ya nos salvó dos veces: el ancla de Copiapó en agosto y el
recálculo de hoy.

**Regla 2 · El examen no puede ser el que generó la hipótesis.** Si una idea
nació mirando Copiapó, no se valida con Copiapó. Se busca un caso que no
participó en el diseño.

**Regla 3 · Toda prueba lleva su nulo.** Sin un brazo nulo, cualquier señal
parece real. Los dos que este proyecto usa:
- **NULL-1** — barajar las fechas dentro del mismo punto: controla «este lugar
  es peligroso siempre».
- **NULL-2** — atribuir el evento a otro punto: controla «fue un mes malo a
  nivel nacional». Ya nos mostró que **la mitad de la señal de `PelPre` era
  eso**, no destreza del lugar.

---

## 3 · Los tres oficios del banco

| oficio | qué hace | de dónde saca la verdad |
|---|---|---|
| **Minero** | extrae del registro qué falló, por qué, dónde y cuándo | el registro, nada más |
| **Probador** | contrasta una hipótesis contra el dato, con criterio previo y nulo | el dato |
| **Refutador** | intenta tumbar el resultado del probador | busca el error, no la confirmación |

El tercero es el que falta en casi todos los sistemas de este tipo y es el que
más vale. Su trabajo **no** es revisar prolijidad: es preguntarse *«¿qué otra
cosa produciría este mismo resultado?»* — un artefacto de medición, un sesgo del
registro, una circularidad. Ya tenemos un caso vivo: los 329 eventos ReTeRM
tienen «lluvia» como detonante **asignado por SERNAGEOMIN**, así que «hay
deslizamientos en meses lluviosos» es en parte tautológico. Eso lo encuentra un
refutador, no un probador.

---

## 4 · Cómo se escribe una hipótesis en este banco

```
H-xx · Enunciado en una línea
  Predicción falsable : qué tendría que verse si es cierta
  Predicción contraria: qué tendría que verse si es falsa
  Dato                : con qué se prueba, y si está bajado o no
  Nulo                : contra qué se compara
  Criterio            : el umbral, fijado ANTES
  Estado              : propuesta / corriendo / PASA / NO PASA / sin poder
```

La casilla «predicción contraria» es obligatoria. Una hipótesis que no dice qué
la refutaría no es una hipótesis: es una opinión.

---

## 5 · Registro de hipótesis

### En curso

**H-A · El terreno encamina la amenaza.** Con lluvia parecida, suelo seco →
remoción; suelo saturado → desborde. La humedad entra con **signo opuesto**.
*Dato:* ReTeRM + inundaciones SENAPRED + humedad ESA CCI. *Nulo:* fechas
barajadas del mismo punto. *Criterio:* p < 0,01 **y** signo predicho.

**H-B · El invierno ataca por dos vías a la vez.** El temporal daña la red desde
afuera y la calefacción la estresa desde adentro; los cortes de invierno deben
tener firma distinta (más clientes por evento, otra hora del día, otra duración).
*Dato:* SEC por comuna y hora. *Criterio:* fijado antes de mirar.

**H-C · Los modos de falla se concentran en invierno.** Si el clima mueve el
riesgo, el registro tiene que mostrarlo. *Dato:* 50.457 eventos SENAPRED.

### Propuestas, con dato identificado

**H-D · La extensión predice el sabotaje.** Lo que no se puede vigilar entero se
corta más. Un data center tiene guardias; 40 km de línea, no. *Predicción:* las
fallas por intervención humana se concentran en elementos de gran `extension`, y
no en los puntuales. *Contraria:* se reparten igual. *Dato:* causa declarada en
SENAPRED + SEC.

**H-E · El NGF-L existe y se puede ver.** Cuando cae la ruta principal, la
secundaria se satura y pasa a ser crítica. *Predicción:* tras un corte vial
mayor, las emergencias en la ruta alternativa suben. *Dato:* 6.141 emergencias
viales del MOP + fechas de cortes mayores. Es la prueba empírica del concepto
central del MCSGS.

**H-F · El aislamiento lo produce el clima.** *Ya tiene respaldo:* los 289
eventos de aislamiento de SENAPRED (140.303 personas) tienen como causas
dominantes inundación, sistema frontal, lluvia, nevadas y remoción. Falta
cuantificarlo bien y ponerle su nulo.

**H-G · La matriz ordena por importancia, no por vulnerabilidad.** *Ya probada
el 16-ago:* `PF` correlaciona +0,935 con `IB` y +0,349 con `FEN`; `FANC` dice
«Alta» en el 96 % de las 835 filas. Queda como resultado establecido, y como
recordatorio de que el banco también se aplica a nosotros mismos.

### Pendientes de dato

**H-H · La reparación se degrada justo cuando más se necesita.** En temporal las
cuadrillas están ocupadas y los caminos cortados, así que el tiempo de
reposición debería alargarse. Es la variable `TRec` que la Ley 21.542 exige y
que nadie en el RMD tiene. *Dato:* duración de cortes SEC + emergencias viales
simultáneas.

---

## 6 · Lo que este banco NO hace

- **No genera métodos de ataque nuevos.** Trabaja sobre modos ya documentados
  públicamente y su lectura defensiva: qué priorizar, qué redundar, qué vigilar.
  El valor está en anticipar lo que ya pasa, no en ampliar el repertorio.
- **No emite ni simula alertas.** El proyecto entrega insumo al SAE; no alerta.
- **No cierra ninguna hipótesis sin el director.** Ningún veredicto es válido
  sin autorización explícita, ni siquiera con resultado perfecto.
