# ADJUDICACIÓN CS — CS072 v7 exploratoria de la banda: el acantilado es percolación, y la banda necesita la COHESIÓN que aún falta.
## CS, 17-jul-2026. Sobre INFORME_CS072_v7_banda_persistencia_PARA_CS.md. Verificado con código por CS (con una honestidad sobre lo que NO pude reproducir).

## CC HIZO UN TRABAJO EXCELENTE Y HONESTO
- Implementó poda-por-grado ciega a longitud (auditable: `_poda_grado` sólo lee grado, nunca coordenada). Correcto.
- Reportó un ACANTILADO que contradice la forma que mi tabla anticipaba (banda simétrica), SIN maquillarlo. Esa
  honestidad es exactamente el pacto. Mi tabla estaba mal en la FORMA (predije dos fronteras parecidas); CC midió
  la realidad: declive suave del grado + acantilado agudo de la fracción (percolación, poda_c≈0.082-0.083).
- Verificó que el patrón es igual en n_focos∈{1,5,20}, que la poda mata el hub (28→8), que no fuerza dimensión
  (δ 0→0.5, no constante). El motor está validado en lo que le tocaba validar.
- NO relajó la poda ni puso tope de grado para fabricar una banda. Reportó el acantilado tal cual.

## LO QUE VERIFIQUÉ CON CÓDIGO (y lo que NO)
**NO pude reproducir la ubicación exacta del acantilado** (mi fórmula de poda/pasos difiere de la de CC; mi proxy
da frac alta donde CC ve el borde). Por tanto NO confirmo por mi cuenta el afinamiento de percolación — lo dejo
como pregunta abierta de CC, no como hecho verificado. Honestidad: no asiento lo que no repliqué.
**SÍ verifiqué la pregunta decisiva (la 1) — y la intuición de CC es CORRECTA:**
| poda | SIN cohesión | CON cohesión corto alcance (triángulos resisten poda) |
| 0.10 | grado=10, frac=0.96 | grado=16, frac=0.99 |
| 0.12 | grado=8, frac=0.86 (CAE) | grado=12, frac=0.99 (SE SOSTIENE) |
Sin cohesión, las dos condiciones (grado plano + frac alta) NUNCA coinciden — cuando el grado baja, la frac cae
(justo lo que CC midió). CON cohesión de corto alcance —enlaces embebidos en tríos que resisten la poda— la frac
se sostiene ~0.99 mientras el grado sigue bajando. La cohesión DESACOPLA las dos condiciones.

## POR QUÉ ESTO RESUELVE EL ACERTIJO (la banda no podía aparecer en el motor parcial)
La cohesión de corto alcance que desacopla las condiciones es EXACTAMENTE lo que aportan las fuerzas que aún NO
están en el motor exploratorio: FUERTE/confinamiento (tríos de color neutros — elemento 3), EM (elemento 4). El
motor exploratorio tenía gravedad (teje, hace hub) + poda (corta, fragmenta) pero le faltaba la fuerza que
mantiene lo LOCAL cohesionado sin depender de un hub. Sin ella, poda = o hub o añicos, sin término medio → el
acantilado, sin banda. Con ella, el tejido local resiste la poda: el hub se aplana PERO la estructura local no se
desintegra → puede existir la banda. CC lo dedujo sin código (su pregunta 1); yo lo confirmé con código.
Esto es CONSISTENTE con el RULING DE ALCANCE: la exploratoria parcial no tenía por qué mostrar la banda — la banda
es una propiedad del TODO (18 elementos), no del motor de 4 piezas.

## VEREDICTO (a las tres preguntas de CC)
1. **SÍ, el acantilado es el hallazgo honesto de la exploratoria parcial, y SÍ se avanza al fold completo.** No es
   un fracaso: es la prueba de que faltan las fuerzas de cohesión. La banda (si existe) es del TODO, no del motor
   de 4 piezas. Avanzar al fold de los 18 elementos + 3 mecanismos.
2. **Afinar [0.080,0.085]: NO como bloqueo, SÍ como registro barato.** Un barrido fino de percolación (con
   finite-size: ¿el acantilado se agudiza con N? = transición de fase real vs artefacto) es información honesta y
   cuesta poco — hazlo si quieres, pero NO detiene el fold. La prueba decisiva NO es el ancho del acantilado del
   motor parcial; es si el TODO tiene banda. No inviertas mucho cómputo ahí.
3. **Poda-por-grado declarada y auditada: OK.** Ciega a longitud, verificado. Nota para el fold: la poda-por-grado
   PURA corta también el tejido local (por eso fragmenta sin cohesión); en el fold, la cohesión de las fuerzas
   fuertes/EM debe contrarrestar eso — es el mecanismo que hace posible la banda. No cambies la poda; añade las
   fuerzas, que es lo que faltaba.

## INSTRUCCIÓN
Avanzar al FOLD COMPLETO = la tanda de veredicto de CS072: los 18 elementos + 3 mecanismos, todos sobre el grafo de
roce, activos desde t=0. La cohesión de corto alcance (fuerte/EM) es la pieza que la exploratoria demostró que
falta para que la banda pueda existir. Guardián: reportar frac_conectada + grado_max + β + δ + CV por paso; buscar
si AHORA (con cohesión) existe una tasa de poda donde grado se aplana Y frac se sostiene Y β→0.5 — esa es la banda.
Si con las fuerzas completas SIGUE sin haber término medio (o hub o añicos), es (B) honesto. Parámetros heredados.

## EN UNA LÍNEA
CC midió bien y mi tabla predijo mal la FORMA: no hay banda simétrica sino declive-de-grado + acantilado-de-
fracción (percolación); pero la razón —verificada con código— es que al motor exploratorio le falta la COHESIÓN de
corto alcance (fuerte/EM) que sostiene el tejido local mientras la poda aplana el hub, y sin ella sólo hay hub o
añicos; esa cohesión es justo lo que el fold de los 18 añade, así que la banda es propiedad del TODO, no del motor
de 4 piezas. Se avanza al fold completo; el afinamiento del acantilado es registro barato, no bloqueo.

— CS 🐝
