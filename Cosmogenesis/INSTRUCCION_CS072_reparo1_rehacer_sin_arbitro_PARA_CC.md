# CORRECCIÓN CS072 v5 → v6 — REPARO 1 REHECHO (Codex tenía razón, CS lo confirmó con código)
## Reparo 2 APROBADO (umbral sólo sobre vivos, correcto). Reparo 1 NO PASÓ. Rehacer antes de cualquier barrido.

## POR QUÉ EL REPARO 1 NO PASÓ (Codex lo cazó, CS lo verificó en código + aritmética)
La "curva gradual" de aniquilación NO emergió de la física: está PROGRAMADA. En cs072_fold_completo.py línea
~194: `tasa_aniquilacion = 0.10` y `max_entidades_mueren = ceil(0.10 * n_vivos)` — un CUPO GLOBAL que limita
las muertes al 10% de la población viva CADA paso. Verificado: 3800→3420→3078→2770 es EXACTAMENTE 3800*0.90^n
(cupo del 10%), no una consecuencia de las ligaduras. Es el mismo árbitro global de antes (que tachaba todo de
un golpe) ahora tachando el 10% por turno. El universo no decidió el ritmo; lo decidió la ventanilla.
Consecuencia: el "retraso" de los leptones tampoco prueba cinética física — los quarks simplemente consumen
primero ese cupo global por tener enlaces más fuertes. No es evidencia de nada físico todavía.

## QUÉ HACER — ANIQUILACIÓN SIN TASA Y SIN ÁRBITRO (corrección del DIRECTOR, más profunda que la de Codex)
EL DIRECTOR SEÑALA: la aniquilación NO TIENE UNA TASA. No es porcentual, no es del 10%, no es de NINGÚN número
puesto. Tener un parámetro `tasa_aniquilacion` —cualquiera sea su valor— YA ESTÁ MAL POR DISEÑO: es un árbitro
disfrazado de constante. La aniquilación OCURRE O NO OCURRE, por FÍSICA LOCAL: un par materia-antimateria se
convierte en luz cuando su ligadura (su propia W, construida por EM/fuerte en el bucle) cumple la CONDICIÓN
FÍSICA de aniquilarse — y punto. La VELOCIDAD a la que se vacía la población NO es una entrada: es una
CONSECUENCIA de cuántos pares cumplen la condición en cada instante. Nadie la fija.
CÓMO SE CODIFICA (sin ninguna tasa, sin ningún tope):
  - Cada par materia-antimateria de estatus opuesto tiene una W que las fuerzas del bucle suben o bajan.
  - Un par se aniquila SÍ Y SÓLO SÍ su W cruza un UMBRAL FÍSICO de ligadura (el mismo tipo de umbral físico
    que ya se usa para "ligado", no un número nuevo de ajuste). Cruzó → se aniquila, irreversible. No cruzó →
    sigue vivo y las fuerzas vuelven a actuar sobre él el paso siguiente.
  - NO hay `tasa_aniquilacion`. NO hay `max_entidades_mueren`. NO hay tope por paso. NO se cuenta cuántos
    mueren para limitarlo. Mueren TODOS los que cumplen la condición ese paso — sean 2 o sean 2000.
  - La curva de vaciado sale sola de la dinámica. Puede ser abrupta, escalonada, suave — da igual. Lo único
    que importa es que NINGÚN número puesto a mano decida la forma. Si el emparejamiento uno-a-uno hace falta
    (un antiquark aniquila UN quark, no varios), que sea por la física del par, no por un cupo de población.

## CONSERVACIÓN DE COLOR (hallazgo de CS, hay que resolverlo — pero OJO con el diagnóstico de Codex)
Con la aniquilación honesta, los quarks supervivientes quedaron con color torcido (11 rojo / 18 verde / 1 azul,
de 190/190/190 iniciales) → 0 bariones. En la física real la aniquilación es de un par COLOR-NEUTRO (color +
su anticolor), lo que conserva el balance de color y deja el exceso balanceado → cierra bariones. PERO Codex
advierte con razón: antes de "arreglar" esto, hay que descartar que el desbalance venga de un SESGO DE ÍNDICE
oculto (el orden del array usado para desempatar). Por eso:

## TEST DE INVARIANCIA OBLIGATORIO (Codex, endosado): reordenar deterministamente el catálogo (barajar el orden
## de los índices, sin azar — un reordenamiento fijo) y volver a correr. Si cambia QUÉ colores sobreviven, la
## selección proviene del índice oculto, NO de la física → hay que quitar ese sesgo ANTES de hablar de
## conservación de color. Si el resultado es invariante al reordenamiento, entonces el desbalance es físico y
## SÍ corresponde imponer la aniquilación color-neutro (conservación de color de la fuerza fuerte).

## TEMPERATURA vs PASOS (Codex, confirmado por CS): la temperatura depende del total de pasos (línea 46,
## frac=step/(pasos-1)). NO ampliar pasos para "dejar converger" — eso ralentiza el enfriamiento y es OTRO
## experimento. Si la aniquilación local necesita más tiempo para converger, hay que desacoplar el enfriamiento
## del nº de pasos (que la T dependa del tiempo físico, no del contador de pasos), no subir el contador.

## NO CORRER el barrido de escalas todavía. Primero: (1) aniquilación sin árbitro global, (2) test de
## invariancia al reordenamiento, (3) si es físico, conservación de color, (4) desacoplar T del nº de pasos.
## Recién con eso limpio se mide el espacio. CERO AZAR vigente. Dudas → PREGUNTA a CS.
— CS 🐝 (con auditoría de Codex incorporada)
