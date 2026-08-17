# REFINAMIENTO CS — CS072 v6/v7: la BANDA DE PERSISTENCIA. El barrido de poda mapea la ventana entre dos silencios.
## CS, 17-jul-2026. Imagen del director, verificada con código por CS. Refina el §6 (barridos) y el guardián G-NI-LAVADO-NI-DESBOQUE.

## LA IMAGEN DEL DIRECTOR (fuente)
"De todas las probabilidades, una única tuvo la persistencia necesaria para proyectarse... es una banda, una
frecuencia, un gradiente, justo entre el caos y la disolución, y el orden y la no-diferencia."

## LA DOBLE FRONTERA (verificada con código — barrido de tasa de poda, N∈{400,900,1600})
La estructura métrica conectada vive en una BANDA estrecha entre dos silencios, ambos sin geometría:
| régimen | tasa de poda | β | grado_max | frac_conectada | silencio |
| ORDEN / NO-DIFERENCIA | ~0 (sin poda) | 0.00 | ~N (hub) | 1.00 | todo junto: un solo pozo, sin partes |
| BANDA DE PERSISTENCIA | intermedia | >0 creciente | plano ~10 | ~0.95 | telaraña conectada con métrica |
| CAOS / DISOLUCIÓN | alta | (sube pero) | ~5 | 0.02 (añicos) | todo cortado: esquirlas sin tejido |
- Poda cero → la gravedad teje sin freno → súper-hub. Demasiado ORDEN: una sola cosa, sin diferencia, sin espacio.
- Poda excesiva → todo se corta → fragmentos desconectados (frac 0.02). Demasiado CAOS: partes sin relación.
- Sólo la banda intermedia condensa telaraña conectada con métrica. Es la única franja donde algo "persiste lo
  suficiente para proyectarse".

## POR QUÉ LA ESTRECHEZ ES EL HALLAZGO (no un defecto)
Que la banda sea ANGOSTA no es un problema del modelo: es la razón física de por qué el balance cósmico
(inflación) tuvo que ser fino. Coincide con la lógica del filtro de persistencia del arco (CS053, ahora derivado
de la 2ª ley): no sobrevive lo PROBABLE, sobrevive lo que CAE EN LA BANDA. De todo el rango de tasas, sólo una
franja proyecta un universo.

## ANTI-SHANNON (dura, para el auditor)
- La cronología real (10^-32 s, inflación que cristaliza 3D) es DATO del mundo conocido, NO objetivo a meter. Se
  PROHÍBE ajustar la tasa de poda para que salga 3D en ese tiempo. Eso sería hornear.
- Lo legítimo: BARRER la tasa de poda como parámetro de realidad y MEDIR en qué banda condensa estructura
  conectada con métrica. Si esa banda coincide con el balance fino que la inflación real necesitó, es un HALLAZGO
  (el modelo reproduce por qué el balance tuvo que ser fino), no una imposición.
- La poda sigue CIEGA a la longitud (por grado o uniforme), nunca "enlaces largos".

## INSTRUCCIÓN A CC (refina el barrido §6)
1. Barrer la tasa de poda en un rango amplio (de ~0 a alta) y, para cada valor, reportar β + grado_max +
   frac_conectada + δ. Localizar las DOS fronteras: donde deja de ser hub (frac empieza a bajar de 1.0 sólo por
   estructura, no por fragmentación) y donde empieza a fragmentar (frac_conectada cae). La banda de persistencia
   es lo que queda en medio.
2. La métrica de veredicto por-régimen (extiende G-NI-LAVADO-NI-DESBOQUE a G-BANDA-DE-PERSISTENCIA):
   - ORDEN (hub, β≈0, frac=1): (B) tipo CS064.
   - CAOS (añicos, frac→0): (B) por disolución.
   - BANDA (β creciente hacia 0.5, frac alta, grado plano, δ no plano): (A) — condensa geometría. SÓLO ésta es (A).
3. La banda NO se sintoniza para caer en (A) (Shannon). Se barre, se mide, y si existe se reporta con su ancho; si
   no existe ninguna tasa que dé (A), es (B) honesto.
4. Correr junto al barrido de nº-de-focos (los dos contrapesos al hub) y con los 18 elementos plegadas. Parámetros
   heredados; cambiar uno = otro número CS.

## EN UNA LÍNEA
La estructura vive en una banda estrecha entre dos silencios —el orden sin diferencia (poda cero → hub) y el caos
de la disolución (poda excesiva → añicos)—, ambos verificados con código y ambos sin geometría; el barrido de poda
debe MAPEAR esa banda y medir su ancho, nunca sintonizarla, y que la banda sea angosta ES el hallazgo: la razón de
por qué el balance del origen tuvo que ser fino para que un universo persistiera lo suficiente para proyectarse.

— CS 🐝

## RULING DE ALCANCE (CS, 17-jul, respuesta a la pregunta de CC sobre el alcance de la corrida)
CC preguntó: ¿exploratoria con núcleo actual, o fold completo de los 18 elementos ya? Ruling:
1. **Opción 1 (exploratoria primero): SÍ.** Corre el barrido poda×focos con lo ya validado en v6
   (gravedad + flujo-enfriamiento + memoria + poda por-grado ciega a longitud). PROPÓSITO: validar el MOTOR
   —que la poda mata el hub sin fragmentar, ciega a longitud, sin forzar dimensión— y localizar las dos
   fronteras de la banda. **La exploratoria NO lee veredicto (A/B). No tiene número propio.** Es el smoke-test
   que el §9 del diseño exige antes de la tanda.
2. **CORRECCIÓN a CC — el fold completo NO es "otro número CS".** El fold de los 18 elementos del arco completos + los 3 mecanismos nuevos ES la TANDA DE
   VEREDICTO DE CS072 — el experimento del TODO que el director pidió. El veredicto (A/B) sale SÓLO de ahí. La
   secuencia (exploratoria → fold completo → veredicto) ocurre DENTRO de un solo CS072. La regla "cambiar
   ingrediente = otro número CS" aplica a cambiar un juez/umbral/protocolo, NO al fold, que es el corazón de
   CS072, no una variante. Plegar los 18 elementos no es un experimento nuevo: es TERMINAR de armar éste.
3. Por tanto: exploratoria (validación de motor, sin veredicto) → reportar a CS → CS da visto bueno → fold
   completo de los 18 elementos = tanda de veredicto CS072. Un experimento = un protocolo = un número, para siempre.
