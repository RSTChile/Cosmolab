# De dónde sacar los objetivos, sin inventarlos

**21-ago-2026.** Instrucción de Alexis: *«Respecto a Pev y Peh se tienen que usar
proxies. Es decir, en caso de guerra convencional, el ataque a un aeropuerto es
habitual… ese sería un ejemplo.»*

La idea resuelve un problema que yo había dado por irresoluble. `Pev` (Prioridad
Estratégica para Conflicto Vertical y Regular) y `Peh` (Prioridad Estratégica
para Conflicto Horizontal e Irregular) no se pueden calibrar contra daño real,
porque **no existe registro de infraestructura chilena dañada por guerra
convencional ni por sabotaje sistemático, y no va a existir**. Un proxy —una
variable de resultado sustituta— rompe ese bloqueo.

**Pero un proxy inventado por nosotros sería exactamente el vicio que llevamos
tres días corrigiendo en la Matriz.** Por eso este documento propone **fuentes**,
no listas: cada tipo de elemento entra o no entra según lo que diga un texto
público, verificable y ajeno al proyecto.

---

## 1 · Para `Pev` (conflicto vertical y regular): el derecho internacional humanitario

El Protocolo I Adicional a los Convenios de Ginebra (1977) **nombra tipos de
infraestructura de forma explícita**, y Chile es parte. No es doctrina de un
ejército: es tratado ratificado, publicado y estable.

| artículo | qué define | tipos que nombra |
|---|---|---|
| **52** | qué es un objetivo militar legítimo | criterio general: contribución efectiva a la acción militar y ventaja militar definida |
| **54** | bienes indispensables para la supervivencia de la población civil | **instalaciones de agua potable, obras de riego, zonas agrícolas, cosechas, ganado, alimentos** |
| **56** | obras e instalaciones que contienen fuerzas peligrosas | **presas, diques y centrales nucleares de energía eléctrica** |

**Por qué sirve como proxy y no como opinión:** los artículos 54 y 56 existen
justamente porque esos tipos **son atacados**. La protección especial es la
huella de la costumbre bélica. Un tipo que el tratado protege expresamente es un
tipo que la doctrina considera objetivo.

**Cómo se usaría, en concreto:** cada uno de los 835 ítems de la Matriz recibe
una marca de tres valores según lo que digan los artículos —protección especial
(56), bien indispensable (54), o ninguna— y esa marca es la variable contra la
cual se ajustan los pesos de `Pev`. La marca la pone el texto, no nosotros.

**Complemento acotado, si hiciera falta más resolución:** la doctrina pública de
selección de objetivos de la OTAN y los Estados Unidos está publicada y describe
categorías de objetivos de primera ola. Es más granular pero también más discutible
—es doctrina de una parte, no derecho—, así que la usaría sólo como segundo nivel
y declarada como tal.

---

## 2 · Para `Peh` (conflicto horizontal e irregular): la ley chilena

Aquí hay algo mejor que la doctrina extranjera: **una lista chilena, vigente y
con rango de ley**.

La **Ley 21.663, Marco de Ciberseguridad** (marzo de 2024) define en sus
artículos 4 y 5 los **sectores de servicios esenciales**, y crea la figura del
**Operador de Importancia Vital**, designado por la Agencia Nacional de
Ciberseguridad.

Sectores que la ley nombra:

> generación, transmisión o distribución eléctrica · transporte, almacenamiento o
> distribución de combustibles · suministro de agua potable o saneamiento ·
> telecomunicaciones · infraestructura digital · servicios digitales y de
> tecnología de la información gestionados por terceros · transporte terrestre,
> aéreo, ferroviario o marítimo · banca, servicios financieros y medios de pago ·
> administración de prestaciones de seguridad social · servicios postales y de
> mensajería · prestación institucional de salud · producción e investigación de
> productos farmacéuticos

**Por qué es la fuente correcta para `Peh`:** `Peh` mide prioridad ante sabotaje,
ciberataque y acción de actores no estatales. La Ley 21.663 fue escrita
exactamente para eso, por el Estado de Chile, y **ya hizo el trabajo de decidir
qué es esencial** — con consulta pública, tramitación legislativa y sanciones
asociadas. Usar su lista no es tomar prestada una opinión: es alinear la Matriz
con el criterio legal vigente del país al que sirve.

**Ventaja adicional:** la designación de Operador de Importancia Vital es
**nominal y actualizable**. No sólo dice qué tipos importan: llegado el caso dice
qué organizaciones concretas. Eso permitiría en el futuro pasar del tipo al
activo, que es la dirección en la que va todo el proyecto.

---

## 3 · Lo que esto arregla y lo que no

**Arregla** la falta de variable de resultado: `Pev` y `Peh` pasan a tener contra
qué ajustarse, con fuentes ajenas y citables.

**No arregla el problema de fondo, que es el mismo de `Pen`.** Medido hoy sobre
los cinco tipos con falla documentada del invierno de 2026: la varianza de `FEN`
(Fragilidad ante Eventos Naturales) entre ellos es **exactamente cero**, porque
la Matriz le dice «Alta» a todos. **El peso de una variable sólo se puede estimar
si la variable varía.** Con `FANC` (Fragilidad ante Ataques No Convencionales)
pasa lo mismo y peor: dice «Alta» en 802 de 835 filas.

⇒ **Aunque tengamos el proxy, el peso 0,5 de `FANC` en `Peh` seguirá siendo
inestimable mientras `FANC` sea casi una constante.** El proxy sirve para dos
cosas distintas y las dos valen: **primero para RE-MEDIR `FANC`** —qué tipos
nombra la ley y cuáles no, que sí varía— y **sólo después para ajustar los
pesos**.

Ése es, me parece, el orden correcto: **el proxy no es el calibrador de los
pesos, es el medidor de la entrada que hoy no varía.**

---

## 4 · Lo que haría falta decidir antes de implementar

1. **Si la marca es binaria o graduada.** El artículo 56 protege tres tipos de
   forma absoluta; el 54 protege una familia más amplia. ¿Dos niveles o tres?
2. **Qué hacer con los 176 ítems que no son activos** —«Personal de
   Operaciones», «Infraestructura Vulnerable a Ransomware»—: la ley habla de
   servicios y organizaciones, no de propiedades.
3. **Si la Ley 21.663 manda sobre la Matriz o sólo la informa.** Es una decisión
   de canon, y es tuya: alinear la Matriz con la ley la vuelve más defendible
   ante SENAPRED y la Agencia Nacional de Ciberseguridad, pero le quita
   independencia.

**No implementé nada.** Esto es la propuesta de fuentes que dije que te traería
antes de armar ninguna lista.

Fuentes:
- [Protocolo I adicional a los Convenios de Ginebra de 1949 — Comité Internacional de la Cruz Roja](https://www.icrc.org/es/document/protocolo-i-adicional-convenios-ginebra-1949-proteccion-victimas-conflictos-armados-internacionales-1977)
- [Ley 21.663 — Biblioteca del Congreso Nacional de Chile](https://bcn.cl/leychile/Navegar?idNorma=1202434&idParte=10496166&idVersion=)
