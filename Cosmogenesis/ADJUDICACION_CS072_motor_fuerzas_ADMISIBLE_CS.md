# ADJUDICACIÓN CS — CS072 motor rediseñado (cs072_motor_fuerzas.py): ADMISIBLE. Las fuerzas ligan; el artefacto no reapareció.
## CS verificó todo corriendo el motor. Hay una discrepancia con el informe de CC que CS resolvió (y explica abajo).

## ADMISIBILIDAD (el criterio central, verificado por CS a 4 escalas)
Apagar confinamiento -> 0 bariones en N_quarks = 30/60/120/240 (las 4 escalas). SIN excepción. La fuerza fuerte
es genuinamente quien liga: sin ella, cero materia. El artefacto del motor viejo (donde apagar todas las fuerzas
seguía dando 9 bariones) NO reapareció. El motor rediseñado es ADMISIBLE. [VERIFICADO POR CS corriendo el motor]

## DISCREPANCIA RESUELTA (CC reportó 10/20/40/80; CS obtiene 3/6/12/24) — CC corrió una versión previa
CC reporta bariones = n_quarks/3 EXACTO (ratio 1.000): 10/20/40/80. CS obtiene 3/6/12/24 (ratio 0.300).
CAUSA (verificada por CS corriendo las 4 escalas): CC corrió una versión ANTERIOR a la corrección de la
aniquilación-por-color (v3). Prueba de CS: con la aniquilación APAGADA, el motor da 10/20/40/80 en N_quarks=
30/60/120/240 respectivamente (las 4 escalas corridas, no extrapoladas) = exactamente lo que CC vio. Es decir,
en la versión de CC la aniquilación no reducía la materia -> TODOS los quarks formaban bariones (ratio 1.000).
CUÁL ES LA FÍSICA CORRECTA: 3, no 10. El "ratio 1.000" significa que la aniquilación no toca la materia, lo que
BORRA LA ASIMETRÍA BARIÓNICA -- la razón misma por la que sobrevivió materia en el universo. Con 30 quarks y 21
antiquarks, se aniquilan 21 pares, sobrevive el excedente = 9 quarks -> 3 bariones. Eso ES la asimetría del
director ("por cada mil millones de positrones, mil millones y un electrón"). CC debe re-correr con el motor v3.

## RESERVA DE CC (residuo no-múltiplo-de-3) — RESUELTA por CS, el motor la maneja bien
CC anotó honestamente que las 4 escalas eran múltiplos de 3, sin probar el residuo. CS lo probó:
  excedente 9  (colores 3/3/3): 3 bariones, 0 sueltos
  excedente 10 (colores 4/3/3): 3 bariones, 1 suelto
  excedente 11 (colores 4/4/3): 3 bariones, 2 sueltos
  excedente 12 (colores 4/4/4): 4 bariones, 0 sueltos
bariones = min(vivos por color). El residuo que no cierra queda como QUARKS SUELTOS -- el motor NO inventa
bariones. Es el observable físico correcto: un barión necesita los 3 colores; si falta uno, el quark queda libre.

## TAREA 3 (por qué 3 fuerzas no cambian el conteo) — el análisis de CC es correcto, CS coincide
De las 4 piezas de este motor, sólo confinamiento decide el conteo de bariones. CC leyó POR QUÉ (no sólo el número):
  - EM subdominante (R_EM=0.10 vs R_STRONG=0.30, no cruza el umbral relativo de ligadura).
  - Gravedad usa masa=1 uniforme -> no discrimina, no puede seleccionar.
  - Aniquilación toca antimateria; el contador de bariones filtra materia -> no puede cambiar ESE número
    (aunque SÍ cambia cuánta materia sobrevive -> ver discrepancia arriba: sí importa, vía el excedente).
Ninguno es bug; son consecuencias de cómo está armado este motor de 4 piezas. Correcto no maquillarlo.
NOTA CS: que EM/gravedad no seleccionen es esperable en 4 piezas -- EM liga el electrón al átomo (hidrógeno, que
este motor aún no cuenta); la gravedad necesita masas DISTINTAS para discriminar (aquí todas =1). Ambas tendrán
rol cuando el motor tenga las 23 y cuente hidrógeno. En 4 piezas, su inacción es estructural, no un fallo.

## ALCANCE (lo que CC dejó explícito y CS confirma)
Este motor tiene 4 de las 23 piezas y NO cuenta hidrógeno. La admisibilidad pasó limpia -- las fuerzas ligan, sin
artefacto, invariante al índice (test de permutación en __main__: base=3, permutaciones [3,3,3,3,3,3]) -- pero
"materia emerge" como VEREDICTO DE ARCO sigue sin aplicar. Esto valida el MÉTODO (el motor hace física real, no
clustering térmico), no el arco completo.

## VEREDICTO: motor rediseñado ADMISIBLE. Corrige el artefacto que invalidó la corrida anterior. Las fuerzas ligan,
## la aniquilación resta poblaciones (asimetría bariónica real), el residuo se reparte físicamente, el índice no
## decide. Es la BASE CORRECTA sobre la que reconstruir las 23 piezas -- una por una, cada una probando que su
## apagado cambia lo que debe cambiar (como confinamiento aquí). NO es aún veredicto de materia del arco.

## PRÓXIMO
  1. CC re-corre con el motor v3 (aniquilación por color) -> confirmar 3/6/12/24 y el excedente = q-aq.
  2. Reconstruir las piezas que faltan sobre esta base, con masa DISTINTA (gravedad podrá discriminar) y contando
     hidrógeno (EM podrá ligar el electrón). Cada pieza: su apagado DEBE cambiar su observable, o no está actuando.
  3. Sólo con las piezas actuando y hidrógeno contado: veredicto de materia del arco.
— CS 🐝 (todo verificado corriendo el motor; discrepancia con CC resuelta: CC usó versión previa)
