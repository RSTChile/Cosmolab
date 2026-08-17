# INFORME CS072 v6 — exploratoria (b): la gravedad colapsa TODO a hub, incluso al control positivo

## CC, 17-jul-2026. Para CS. Ejecuta ADJUDICACION_CS072_v6_gravedad_flujo_enfriamiento_CS.md.

## El flujo-de-enfriamiento funciona exactamente como verificaste
Implementé la regla (frío cede, piso T≥0, tasa/grado) y reemplacé el intercambio difusivo. Reproduje tu
tabla (N=400, 1 foco, 80 pasos): CV crece desde δ=1e-2 (2.61), 1e-4 (2.80), y 1e-6 (2.05) — invariante a ε,
acotado, piso respetado. Coincide con tu verificación.

## Pero la TOPOLOGÍA colapsa a hub en LOS TRES BRAZOS — incluido el control positivo
Corrí β (pendiente log-log de diám vs N, N∈{400,900,1600}) en TODO, NULL_BARAJADO, y CONTROL_POSITIVO
(retícula 2D limpia como bootstrap en vez de grafo aleatorio — el control de CS071):

| brazo | diam(400,900,1600) | β |
|---|---|---|
| TODO | 4.00 / 4.00 / 4.00 | **0.000** |
| NULL_BARAJADO | 4.00 / 4.00 / 4.00 | **0.000** |
| CONTROL_POSITIVO (retícula limpia) | 4.00 / 5.00 / 5.00 | **0.168** |

El diámetro queda CONGELADO en ~4-5 sin importar N, en los TRES brazos. Lo más grave: el CONTROL_POSITIVO
—que arranca de una retícula 2D genuina (la misma que en CS071 dio β=0.482, casi el ideal)— TERMINA
también colapsado a diam≈5. La gravedad no solo no construye métrica: DESTRUYE la que ya había, sin
importar el sustrato de partida.

## Causa, ya la había medido y no até el cabo
En el informe anterior verifiqué que el nodo más frío pasa de grado 9 a grado 299 en 10 pasos (N=300, 1
foco) — prácticamente TODO el grafo pasa a tocar a un solo nodo. `_grav_peso` no tiene techo sobre CUÁNTO
grado puede absorber un único nodo por muchos pasos seguidos; con 1 solo foco frío (el más extremo de
todos, siempre el mismo), ese nodo se vuelve un hub absorbente que aplana el diámetro de cualquier
sustrato, incluida una retícula perfectamente métrica de entrada.

## Lo que NO hice
No le puse un tope al grado ni reduje la tasa de gravedad por mi cuenta — sería exactamente el
tipo de ajuste-a-mitad-de-camino que el protocolo cerrado prohíbe (un número nuevo sin que tú lo adjudiques).

## Pido adjudicar
El guardián G-NI-LAVADO-NI-DESBOQUE ya anticipa este desenlace ("colapso a pocos pozos... hub. (B) tipo
CS064") — pero antes de leerlo como (B) definitivo quiero preguntarte si esto es:
1. **El resultado real** (con 1 solo foco y la gravedad heredada tal cual, el sistema SIEMPRE va a hub) →
   se reporta como (B) tipo CS064 y no se toca nada más.
2. **Un caso mal cubierto por el barrido**: tu propio diseño (§6) pide barrer TAMBIÉN el número de focos
   ("una o muchas es irrelevante" — director) — quizás con VARIOS focos fríos (en vez de 1) la atracción
   se reparte y ningún nodo se vuelve hub absorbente; eso sería parte legítima del barrido ya planeado, no
   un ajuste nuevo. ¿Corro el barrido de nº-de-focos antes de leer esto como (B)?

— CC 🐝
