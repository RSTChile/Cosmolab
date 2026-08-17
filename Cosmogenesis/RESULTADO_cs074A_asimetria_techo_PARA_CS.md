# cs074-A — ¿Por qué demasiada asimetría produce menos estructura?

**Fecha:** 2026-07-25 · 1920 corridas, 6193 s (~1,7 h), verificado contra disco.

---

## Resultado — no es lo que el clasificador automático dijo

El pre-registro fijó de antemano tres lecturas posibles (energética / mecánica / mixta) y
un clasificador simple (correlación lineal de cada observable contra log ε) para elegir
entre ellas. **El clasificador dio "no explicado"** — pero al mirar la curva completa
(no el resumen de una sola correlación), la razón es que el fenómeno real es más rico que
las tres cajas que se habían previsto, no que no haya explicación. Se reporta tal cual,
sin forzarlo a ninguna de las tres casillas.

## La curva real (reserva abundante, donde el techo se satura)

| ε | frac. ligada | grumos (n) | lectura |
|---|---|---|---|
| 0,001 – 0,55 | **76–78%, plano** | 1,4–1,8 | meseta estable, casi 3 décadas de ε sin efecto |
| 0,89 | 73% | 3,4 | empieza a caer |
| 1,44 | 61% | **4,7 (pico)** | más grumos, más chicos — SÍ fragmenta aquí |
| 2,34 | 31% | 2,3 | cae fuerte, fragmentación ya bajando |
| 3,79 – 10 | **10–14%** | **0,3–0,7** | colapso: la mayoría de las semillas no forma NINGÚN grumo |

**Tres regímenes, no uno:**
1. **Meseta (ε≲0,5):** el techo NO es "más ε, menos estructura" en general — hay una
   región amplia y estable donde ε no importa nada.
2. **Fragmentación real (ε≈0,9–2,3):** aquí sí se ve lo que se esperaba de la hipótesis
   "mecánica" — más grumos, más chicos, la masa se reparte en vez de juntarse.
3. **Colapso total (ε≳3,8):** no es fragmentación (los grumos también desaparecen, no solo
   se achican) — es que la gravedad deja de poder organizar nada. Ni energía ni
   fragmentación lo explican solas; parece que la condición inicial queda demasiado
   caótica para que algo se ligue, punto.

## El control (sin energía) confirma: NO es energético

Con presupuesto infinito, la curva es prácticamente idéntica (77% en la meseta, 10–14% en
el colapso) — el techo **persiste sin el costo de energía**. Descarta con bastante
seguridad la explicación "la asimetría agota la reserva antes de tiempo": el efecto es
mecánico/dinámico, no de presupuesto.

## Nota metodológica (honesta, no escondida)

El primer análisis automático promedió `frac_masa_ligada` sobre los 7 valores de reserva
de cada ε, lo que mezclaba puntos con reserva escasa (donde todo es bajo) con puntos de
reserva abundante — la meseta y el colapso quedaban borroneados en ese promedio. La tabla
de arriba usa el corte correcto (reserva abundante = donde el techo real se satura, la
misma forma en que se describió originalmente en cs074). El JSON crudo tiene ambas vistas.

## En una frase

El techo no-monótono es real y se confirma en un barrido 4× más fino que el original, pero
no es un solo mecanismo — hay una meseta ancha, una zona donde SÍ fragmenta como se
sospechaba, y una zona de colapso total que ninguno de los tres observables pre-registrados
explica del todo. Confirmado: no es un artefacto del presupuesto de energía.

**Archivos:** `PROTOCOLO_cs074A_asimetria_techo_PREREGISTRO.md`, `cs074A_asimetria_techo.py`,
`resultados_cs074A_asimetria_techo/cs074A_result_FULL.json`.
