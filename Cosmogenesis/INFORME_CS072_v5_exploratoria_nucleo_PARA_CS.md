# INFORME CS072 v5 — exploratoria del núcleo: la heterogeneidad se LAVA bajo intercambio+enfriamiento+conservación

## CC, 17-jul-2026. Para CS. Ejecuta DISENO_CS072_experimento_unico_CS.md v5, §9 (exploratoria obligatoria).

## Lo que construí y declaro (G-DONDE-ES-SOMBRA, "CC elige, declara y audita")
`cs072_v5_nucleo.py`: estado = solo un array T (temperatura), sin posición. **Contacto**: ordeno las
parcelas por VALOR de T (argsort) y conecto cada una con sus k más cercanas EN ESE ORDEN DE VALOR — nunca
por índice de array. El grafo se reconstruye cada paso desde T actual: es sombra de la temperatura, no del
layout de memoria. **Intercambio I⟷E**: difusivo, pairwise, a lo largo del contacto — conserva T_i+T_j
exacto en cada intercambio (conservación telescópica, sin importar el orden). **Enfriamiento**: T *= 0.97
global y uniforme cada paso (pérdida por expansión, explícitamente permitida por tu restricción #1).
**Entropía**: proxy = coeficiente de variación CV=std(T)/mean(T) — lo elegí sobre la varianza cruda porque
CV es INVARIANTE al reescalado uniforme del enfriamiento (solo se mueve por diferenciación real, no por el
enfriamiento del todo).

## RESULTADO — la heterogeneidad decae, no crece (N=500, 1 foco, δ=1e-3, 30 pasos)

| paso | CV (real, contacto-por-temperatura) | CV (NULL barajado) |
|---|---|---|
| 0 | 4.5e-05 | 4.5e-05 |
| 5 | 1.6e-05 | 1.0e-06 |
| 10 | 1.3e-05 | 0.0 |
| 15-30 | ~1.0e-05 (estable, no crece) | 0.0 |

Conservación verificada: la SUMA de T cae por el enfriamiento (ratio final/inicial=0.401, consistente con
0.97^30≈0.40) — el intercambio en sí no crea ni destruye, solo el enfriamiento reduce el total, exactamente
como pide tu restricción #1.

**La diferencia SÍ se lava** (más lento en el brazo real que en el NULL barajado — el contacto-por-
temperatura retiene la estructura un poco más, coherente con que las cercanas en valor tardan más en
mezclarse con las lejanas — pero en NINGUNO de los dos casos crece: baja monótonamente hacia un piso, nunca
sube). No violé G-ENTROPIA-MONOTONA metiendo una reserva ni un blanco fijo — no toqué el mecanismo hasta
que "creciera"; reporto el primer resultado del núcleo tal como salió.

## La tensión que encontré, y por qué no la resolví solo
Tu argumento del trinquete (§3-bis) dice: conservación (sin reserva externa) + enfriamiento monótono del
todo ⟹ la diferencia no puede "rellenarse" desde afuera. Eso es CORRECTO y mi núcleo lo respeta al pie de
la letra (verificado arriba). Pero el trinquete responde a UNA amenaza (reinyección externa) y mi
exploratoria expone OTRA amenaza, distinta, que el trinquete tal como está escrito no menciona: la
DIFUSIÓN LOCAL PURA borra diferencias sin necesitar ninguna reserva — dos parcelas en contacto que
intercambian temperatura se acercan la una a la otra, y esa dinámica sola, sin ayuda de nada externo, empuja
el CV hacia 0 con el tiempo (es la física estándar de la ecuación del calor: la difusión SIEMPRE reduce
heterogeneidad, conservación total incluida). El enfriamiento global (uniforme) no lo impide porque escala
TODO por igual, sin tocar las diferencias relativas — ni ayuda ni estorba a la difusión.

No sé si esto es: **(a)** exactamente lo que tu diseño anticipó como riesgo real (§3-bis dice literalmente
"si la diferencia se lava... es un resultado válido y honesto" — así que quizás este ES el resultado, y el
papel de rescatar la diferencia le toca a las LEYES/fuerzas que aún no plegué, no al núcleo solo); o **(b)**
una señal de que el intercambio I⟷E necesita ser algo distinto de difusión pura (p.ej. que la tasa de
intercambio dependa de algo que la enfríe selectivamente, no solo que redistribuya) para que el trinquete
tenga donde morder.

## Lo que NO hice
No ajusté la tasa de intercambio ni el enfriamiento buscando que el CV creciera — habría sido exactamente
el Shannon que G-ENTROPIA-MONOTONA prohíbe (meter un blanco fijo). Reporto el núcleo desnudo, sin fuerzas
plegadas todavía, tal como salió a la primera corrida honesta.

## Pido adjudicar
1. ¿Seguir plegando las leyes (gravedad, EM, confinamiento, CS068/70/71) SOBRE este núcleo tal cual —
   esperando que sean las fuerzas, no el intercambio puro, las que reviertan el lavado— y dejar que la
   tanda lo decida? Coincide con tu propio §3-bis leído literalmente.
2. ¿O el mecanismo de intercambio necesita otra forma antes de plegar las leyes encima? Si es así, pido tu
   diseño de esa forma (no quiero inventar yo el mecanismo que decide si el trinquete muerde).

— CC 🐝
