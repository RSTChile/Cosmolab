# ADJUDICACIÓN CS062 — Paisaje con gravedad ∝ peso-intrínseco. VEREDICTO: NEGATIVO-MATIZADO. El proxy de grado SÍ inflaba el negativo de CS057 (grieta CS060-B confirmada), pero el negativo CENTRAL se sostiene y NO es la masa — es la independencia-del-grado. La contingencia queda firme, con una corrección honesta.

**Adjudica:** CS · **Fecha:** 9-jul-2026 · **Método:** auditoría de CÓDIGO (cs062_paisaje_peso.py) + DATOS (cs062_paisaje_peso.csv, 52.248 filas verificadas), no la prosa del informe.
**Ejecutó:** CC. **Cifras del veredicto:** recomputadas por CS sobre el CSV, no copiadas del informe.

---

## 0. QUÉ SE AUDITÓ EN EL CÓDIGO (antes de mirar un solo número)
- **`_grav_peso`**: gravedad Newton genuina — fuente ∝ w, blanco ∝ w[j]/d^ALPHA. Con `acople="grado"` usa
  w=grado (= CS057, control); con "peso"/"null_peso" usa la masa intrínseca. Correcto.
- **`_masa`**: masa log-uniforme por nodo, INDEPENDIENTE del grado. Correcto.
- **`null_peso`**: baraja las masas (mismos valores, reasignados al azar) — rompe la correlación masa↔posición.
  Es exactamente la cuerda de CS060-B: separa "el peso REAL importa" de "cualquier peso que no sea grado importa".
- **Guardián corr_ok**: verificado = 1 en las 52.248 filas (100%). El peso quedó genuinamente descorrelacionado
  del grado en toda la corrida. No es el grado disfrazado.

## 1. LAS TRES CIFRAS DEL VEREDICTO (recomputadas por CS)
Viabilidad 3D+4D (viable_d3 + viable_d4):

| comparación | GLOBAL | punto físico |
|---|---|---|
| **peso**       | 0.162 | 0.000 (estabiliza curv 0.625) |
| **grado** (=CS057) | 0.110 | 0.000 (curv 0.750) |
| **null_peso**  | 0.163 | 0.000 (curv 0.875) |

- **peso − grado = +0.052 global** (d3 solo: 0.072 vs 0.045, +60% relativo). El grado SÍ sesgaba contra 3D/4D.
- **peso − null_peso = −0.001 global** (ruido). El peso "correcto de Newton" NO tiene nada de especial.
- **punto físico: 3D+4D = 0.000 en los tres brazos; curv domina.** El muro no cae.

## 2. LECTURA CONTRA LAS DOS SALIDAS PRE-INSCRITAS
Antes de correr, se fijaron dos desenlaces para no acomodar el dato:
- **(1)** si el 3D/4D se hace más viable en todo el mapa → el negativo de CS057 era en parte artefacto del proxy,
  y la contingencia pasa a revisión.
- **(2)** si el negativo se sostiene con la gravedad correcta → contingencia firme, sin asterisco.

**El resultado no cae limpio en ninguna de las dos — cae en un híbrido honesto, y hay que decirlo así:**

- La salida **(1) se activa en su forma DÉBIL**: sí, el proxy de grado inflaba el negativo. El 3D/4D global sube
  de 11.0% (grado) a 16.2% (peso). Parte de lo "muy negativo" de CS057 ERA un artefacto de la forma
  auto-amplificante del acople de grado. Esto es real y hay que asentarlo: **CS057 medía el negativo con un
  proxy que lo empeoraba.**
- Pero la salida (1) **NO se activa en su forma FUERTE**: el 3D no "emerge en todo el mapa". Sube a ~7-16% y
  sigue siendo minoritario; d4 ≳ d3 (ninguna preferencia por el 3D); y en el punto físico el 3D+4D es 0% —
  estabiliza curvo, igual que CS057. El muro de las direcciones (1-jul) no se rompe corrigiendo la gravedad.
- La salida **(2) se cumple para el NÚCLEO**: con la gravedad correcta, ningún acople local hace del 3D-plano
  el resultado privilegiado. El negativo central del arco se sostiene.

## 3. EL HALLAZGO DECISIVO: NO ES LA MASA
`null_peso ≈ peso` (Δ ≈ 0) es el resultado más informativo de CS062. La masa intrínseca correcta de Newton NO
produce nada que no produzca cualquier peso fijo barajado. Lo que alivió el sesgo NO fue la identidad de la masa
— fue la FORMA del acople: no-auto-amplificar-por-grado. Una atracción que premia al ya-conectado colapsa la
diferencia; una con peso propio fijo la deja respirar — pero ni así basta para el espacio. Es exactamente lo que
`null_peso` fue diseñado para discriminar, y respondió. (Cuerda anti-Shannon: el candidato "masa" no le gana a su
propio NULL.)

## 4. VEREDICTO
**CS062 = NEGATIVO-MATIZADO, y CIERRA la grieta de CS060-B con derecho.** El sesgo del proxy de grado era real
(se confirma CS060-B), se corrige, y aun corregido ningún acople local selecciona la dimensión. La grieta no era
la puerta de salida del negativo: era un artefacto de la forma del acople, no la masa.

**Efecto sobre la conclusión del arco:** la contingencia de la dimensión queda **FIRME**, con una corrección de
honestidad que hay que dejar escrita: el negativo de CS057 estaba MEDIDO CON UN PROXY que lo exageraba. Corregido
el proxy, el negativo es algo menos extremo (3D/4D pasa de 11% a 16% global) pero SIGUE SIENDO NEGATIVO en lo que
importa — el punto físico estabiliza curvo, el 3D no se privilegia, el muro de las direcciones sigue en pie. El
asterisco no se elimina: se TRANSFORMA, de "¿todo el negativo era artefacto del proxy?" (respuesta: no) a "el
proxy inflaba el negativo, pero el fondo se sostiene con la gravedad correcta".

**Arco de eliminación local: COMPLETO.** CS057 (fuerzas) · CS059 (espín) · CS060 (masa/gravedad, con la grieta
del proxy ahora cerrada por CS062) · CS061 (masa emergente) · CS063 (3 cuerpos genuino) · **CS062 (gravedad ∝
peso, paisaje completo)** — ninguno selecciona la dimensión. La contingencia se gana el derecho a ser la
conclusión, sin salvedades vivas.

## 5. LO QUE QUEDA (no es salvedad del negativo — es la decisión de rumbo)
El negativo dice qué NO genera el espacio: ningún acople local sobre un sustrato relacional puro. La pregunta
abierta (1-jul, no un cabo del negativo sino la bifurcación de diseño): ¿se deja el sustrato relacional y se
construye uno CON dirección/ángulo desde el arranque (el campo continuo), o se consolida el arco de eliminación
como cerrado y la contingencia como resultado? Eso lo decide Alexis. CS062 no deja ninguna salvedad experimental
viva: el arco local está agotado.

— CS. Auditado sobre código y datos reales. Las cifras son de mi cómputo; el veredicto se leyó contra las dos
salidas pre-inscritas sin forzarlo a ninguna.
