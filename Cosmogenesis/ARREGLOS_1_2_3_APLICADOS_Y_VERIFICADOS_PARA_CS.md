# Los 3 arreglos: aplicados y verificados

**Fecha:** 2026-07-25 · **Para:** Alexis · **De:** CC · **En simple, con analogías** (regla permanente)

Resumen de una línea: **los 3 arreglos están hechos y probados que funcionan.** Antes de correr
los 17 pendientes queda UNA decisión tuya (al final de este documento).

---

## ARREGLO 1 — E5.5-3 ya no "inventa" energía

**Qué se cambió:** antes, para "revivir" el campo muerto, le sumábamos un patrón nuevo hecho
desde cero — como intentar revivir una fogata apagada echándole leña nueva en vez de remover las
brasas que ya están ahí. Ahora el motor solo **reordena** los valores que YA existen en el campo
(como remover las brasas sin agregar nada) — matemáticamente esto conserva la energía EXACTO,
no aproximado, porque nunca toca nada de afuera.

**Verificación:** corrí el experimento de cero con el arreglo. Resultado, y es un resultado
limpio e interesante:

- Con poca "removida" (fracción chica), no pasa nada — el campo sigue igual de muerto.
- Con más "removida", el campo se pone **peor**, nunca mejor: la exergía (energía útil) BAJA a
  medida que reordenas más, nunca sube por encima de la línea base.
- En NINGÚN caso, para NINGUNA de las 13 fracciones probadas, la energía útil revivió por encima
  de lo que ya había.

**Lectura simple:** es como confirmar que barajar de nuevo una mano de cartas ya jugada no te
da una mano mejor — solo puede quedar igual o peor, nunca mejor, si no metes cartas nuevas al
mazo. Confirma limpiamente lo que Tema 1 ya venía sugiriendo: la única forma de "salvar" energía
útil es aislarla ANTES de que se mezcle (vía expansión/corte de conexiones), no revivirla después
metiendo (o ahora, reordenando) algo una vez que ya se mezcló del todo.

---

## ARREGLO 2 — el "ruido" ya no se desmadra en sistemas grandes

**Qué se cambió:** el ruido (pequeñas sacudidas al azar que exige la regla de "no solo probar
con una semilla") se repartía igual en cada paso sin importar cuántos pasos tomara la corrida.
En sistemas grandes, que necesitan MUCHOS más pasos para "lavarse", ese ruido se acumulaba sin
control — como echar la misma cantidad de sal en cada cucharada de una olla, sin importar si la
olla es chica o gigante: en la olla gigante terminas con toneladas de sal. Ahora la cantidad de
sal por cucharada se ajusta al tamaño de la olla, para que el total de sal servida sea siempre
el mismo.

**Verificación:** repetí (a escala reducida, hasta N=2048, el mismo tamaño donde el problema
original se detectó) la prueba que lo detectó, comparando ruido VIEJO vs ruido NUEVO lado a lado:

| N (tamaño del sistema) | Ruido VIEJO: ¿distingue señal de ruido? (z) | Ruido VIEJO: cuánta energía se "fuga" | Ruido NUEVO: ¿distingue? (z) | Ruido NUEVO: cuánta energía se fuga |
|---|---|---|---|---|
| 64 (chico) | z=14.4 (bien) | 8% | z=30.6 (bien) | 0.3% |
| 1024 | z=-0.2 (**ya no distingue nada**) | 19% | z=68.2 (bien) | 0.07% |
| 2048 (grande, donde se rompía) | **z=0.04 (roto — confirma el problema original)** | 44% | **z=98.5 (mejor que a N chico)** | 0.03% |

**Lectura simple:** con el ruido viejo, mientras más grande el sistema, más ciego se vuelve el
experimento (no logra distinguir nada real) y más se rompe la contabilidad de energía — se
reproduce exactamente el problema que detectó el experimento 12. Con el ruido nuevo, pasa lo
contrario: sigue viendo perfectamente bien (incluso mejor) y la fuga de energía se queda
chiquita y estable, sin importar el tamaño. **El arreglo funciona, confirmado con números, no
solo en teoría.**

---

## ARREGLO 3 — una sola regla para "energía útil" (parcial, con una decisión pendiente)

Se escribió el módulo con la definición única (la del experimento 3, E5.2-2, más su
complemento de energía total del experimento 11, E5.5-1). Pero al revisar los 13 ya corridos
para ver si se pueden "traducir" a esa regla sin recomputar, salió esto:

- **3 de 13 YA usan exactamente esa definición** (E5.2-2, E5.5-1, y E5.6-2 con una diferencia
  trivial que se corrige con una multiplicación, sin recorrer nada).
- **5 de 13 miden otra cosa por diseño**, no por accidente — como comparar el peso de una fruta
  con el sabor de otra: no es que alguien se equivocó, es que la pregunta que hacían era
  distinta. Forzarlos a la regla común no tendría sentido físico. (Son los 3 del Tema 4, que ni
  siquiera usan el mismo motor base, y E5.3-1/E5.3-5, que miden "cuánta energía está ligada",
  otra pregunta.)
- **5 de 13 usan una tercera definición** ("persistencia", basada en autocorrelación, heredada
  del experimento 1) que es genuinamente distinta —no una variante con factor de conversión— de
  la definición común. Y ninguno de los 13 guardó el campo completo paso a paso, solo los
  resultados ya resumidos — así que **no se puede "traducir" sin volver a correrlos.**

**La decisión que es tuya:** los 5 de la familia "persistencia" (E5.1-2, E5.1-5, E5.5-3,
E5.6-3, E5.6-4) ya dieron resultados honestos y correctos BAJO SU PROPIA definición — el
problema es solo que no se pueden comparar en el mismo gráfico con los que usan la definición
común. Dos caminos:

1. **Dejarlos como están** (con su definición propia, ya reportada) y que solo los 17
   pendientes que toquen esta física arranquen con la definición común desde el principio.
   Más barato, no pierde nada de lo ya hecho, pero esos 5 quedan "en su propio idioma".
2. **Volver a correrlos** con la definición común para que TODOS hablen el mismo idioma y se
   puedan comparar en un solo gráfico. Cuesta cómputo real (5 corridas más).

Mi recomendación es la opción 1 (más barata, nada se pierde, y cada experimento ya declaró su
propia definición honestamente desde el pre-registro) — pero es tu llamada, no la mía.

---

## Qué sigue una vez que decidas lo de arriba

Con los arreglos 1 y 2 verificados, ya se puede correr E5.1-1, E5.1-3, E5.1-4 (los que estaban
bloqueados esperando el arreglo del ruido) usando el módulo `_ruido_calibrado.py`. Los 17
pendientes en general ya pueden arrancar con las reglas de siempre (pre-registro, barridos
amplios, comparar contra control) más los 3 arreglos ya listos.
