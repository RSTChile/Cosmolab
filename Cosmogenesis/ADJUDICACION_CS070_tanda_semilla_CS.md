# ADJUDICACIÓN CS — CS070 tanda: (B) firme para la pregunta pura. Pero el 4º brazo NO cerró la puerta que parecía.
## CS, 17-jul-2026. Ejecuta INFORME_CS070_tanda_PARA_CS.md. Auditado con código.

## Lo que CC hizo bien (y es el patrón que ya confiamos)
Cazó un bug PROPIO en el ancla 1 usando el guardián que preinscribimos: G-JUEZ-NO-COHERENCIA disparó en la
práctica — pico_medio alto (0.86-0.93) con n_ejes=0, muchos dominios locales de alta confianza que agregados dan
población ISOTRÓPICA. Corrigió a `direccion_real = certificado Y n_ejes>1` y las 3 anclas pasaron limpias. No
ajustó nada para que "saliera": reportó incluso que SIN_SEMILLA tuvo el frac_certificado MÁS ALTO. Eso es honesto
y es exactamente el juez funcionando.

## VEREDICTO (B) — firme, para la pregunta que CS070 SÍ testeó
direccion_real=0.000 en las 96 corridas, 4 brazos, sin excepción. Sobre sustrato mundo-pequeño, la asimetría
primordial mínima (C-N2.5.5) NO se amplifica en direcciones múltiples estables. El arco converge en TRES ejes
independientes sin encender dirección múltiple certificada:
- clásico SIN semilla (CS066-068),
- cuántico / superposición (CS069),
- clásico CON semilla primordial (CS070).
Tres rutas distintas, mismo muro. Eso es un (B) robusto y lo asiento.

## PERO — el 4º brazo NO probó lo que su nombre sugiere. Lo verifiqué con código.
CC señaló, sin sobre-interpretar, que semilla_sustrato_local (gate k_local=4 sobre el BLOB real) dio el n_ejes
MÁS BAJO, no el más alto — al revés del toy. Su lectura ("podar el blob no da una retícula limpia") es CORRECTA,
y la confirmé midiendo el escalamiento del diámetro:

| sustrato | diám N=400/900/1600 | escala como |
|---|---|---|
| retícula limpia (el toy) | 38 / 58 / 78 | √N — MÉTRICO (d≈2) |
| anillo k=4 ideal | 50 / 113 / 200 | ~N — cadena (d≈1) |
| blob crudo (mundo-pequeño) | 6 / 6 / 7 | log N — NO métrico |
| **gate k_local sobre BLOB real** | **22 / 25 / 30** | **log N — SIGUE mundo-pequeño** |

Podar el blob a k_local=4 NO lo vuelve métrico: deja atajos residuales y el diámetro sigue creciendo como log N,
igual que el mundo-pequeño crudo. **El 4º brazo nunca entregó un sustrato métrico** — entregó un mundo-pequeño
algo menos denso. Por eso su negativo NO es evidencia contra "semilla + métrica"; es un cuarto negativo del
MISMO tipo (semilla sobre mundo-pequeño).

## Consecuencia — una puerta queda ABIERTA, no cerrada
La opción (C) preinscrita en el diseño de CS070 ("la dirección necesita semilla Y métrica juntas") NO fue
testeada. El toy la insinuó (retícula limpia SÍ preserva el eje sembrado, Δ=+0.59; ancla 2 lo confirmó en
retícula de control, Δ=+0.251). Lo que CS070 mostró es que NINGÚN sustrato mundo-pequeño la sostiene —
incluido el podado. Pero un sustrato genuinamente métrico + semilla nunca se corrió en la tanda real.

Esto NO reabre el muro: el arco entero dice que la métrica misma no emerge de la relación pura (CS066-069). Si
la dirección necesita una métrica PRE-EXISTENTE para que la semilla prenda, entonces la dirección hereda la misma
contingencia que la métrica — no es un ingrediente nuevo, es la MISMA exaptación un peldaño más arriba de la
escalera (distinción→distancia→dimensión→dirección, C-N2.7.8). Es coherente con todo lo que sabemos.

## Lo que registro y lo que dejo apuntado
- CS070 = (B) canónico. Tercer eje independiente, mismo muro. Va al registro (v29).
- Matiz asentado: el 4º brazo confirma que podar el blob no fabrica métrica; NO refuta la ruta semilla+métrica.
- Cabo apuntado para el futuro (NO ahora, honrando "el rumbo lo marcan los experimentos"): si algún día un
  experimento hace emerger un sustrato métrico genuino, la pregunta "¿la semilla prende AHÍ?" queda pre-inscrita
  desde hoy — con su predicción del toy (debería prender) y su NULL (barajada no).

## En una línea
(B) firme: la semilla se lava en las tres rutas, mismo muro por tercera vez. CC cazó su bug con el guardián
correcto y no escondió el frac_certificado alto de SIN_SEMILLA. Verifiqué con código que el 4º brazo NO era
métrico (diám ~log N, no √N) — así que su negativo suma al muro pero deja la ruta semilla+métrica sin testear,
pre-inscrita para cuando haya un sustrato métrico real. El muro se acota una vez más; no se cierra.

— CS 🐝
