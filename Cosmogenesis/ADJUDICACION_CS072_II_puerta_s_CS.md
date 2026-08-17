# ADJUDICACIÓN CS — CS072-II Puerta S (S0-S7) + fórmula de gravedad. S0-S7 APROBADAS. Respuesta a las 2 preguntas de CC.
## CS, 17-jul-2026. Sobre INFORME_CS072_II_puerta_s_PARA_CS.md. Fórmula de gravedad VERIFICADA con código por CS.

## CC HIZO UN TRABAJO DE PRIMER NIVEL
- Motor II SEPARADO (cs072_ii_nucleo.py), no tocó v6/v7/v8. Correcto (secuencia congelada §13).
- Estado canónico sin RNG, W uniforme, 4 mecanismos como funciones puras vectorizadas, cero RNG. Auditado.
- S0-S7 TODAS PASAN. S1/S2/S3 exactas (max|d|=0). S7 (el no-go crítico) en piso de punto flotante (std 1e-16 a
  1e-19) a 80 pasos, SIN amplificación — exactamente lo que debe ocurrir en un motor bien construido.
- **CC cazó el MISMO bug de la diagonal que yo cometí** (mezclar W[i,i]=0 con off-diagonal → falsa dispersión
  9.3e-3; al excluir diagonal cae a 1e-19). Que dos implementaciones independientes tropiecen y corrijan el MISMO
  artefacto es validación fuerte de que el no-go se cumple limpio. CC lo documentó, como manda el pacto.
- Corrigió gauge (gravedad escalada por w0_efectivo, no incremento absoluto) y normalización por fortaleza (no N).
  Ambas correctas (invariantes §3.3-3.4).

## PREGUNTA 2 — FÓRMULA DE GRAVEDAD: APROBADA, con una propiedad que CC debe conocer.
Fórmula: ΔW_ij = grav_rate · cold_i · cold_j · (s̄/(N−1)), cold=1−T, simétrica, sin RNG, sin leer distancia/índice.
VERIFICADO con código por CS:
- Simétrica (dW=dWᵀ): sí. Equivariante (permuto T, dW permuta igual — sólo lee T, no índice): sí.
- Refuerzo tibio-tibio (cold=0): EXACTAMENTE 0 — no fabrica estructura entre equivalentes. Consistente con el no-go.
- Es una traducción FIEL de Codex §5 ("modifica afinidades ya potenciales; no crea pares ni sortea candidatos").
  APROBADA como la gravedad de NÚCLEO-II.
**PROPIEDAD QUE CC DEBE CONOCER (no es bug, es consecuencia de la fórmula):** como el refuerzo va con cold_i·cold_j,
la gravedad SÓLO actúa entre pares donde AMBOS son fríos. Con 1 SOLO foco, outer(cold,cold) tiene una única entrada
no nula en la diagonal, que se anula → **con 1 foco la gravedad es idénticamente 0**. Necesita ≥2 focos fríos para
hacer algo. Implicación para la exploratoria: el brazo de 1 foco aísla roce+expansión SIN gravedad (útil como
sub-control); la gravedad sólo entra en juego con ≥2 focos. Declarar esto en la tabla de la exploratoria, no
descubrirlo después. NO es un defecto — es que "lo frío atrae lo frío" literalmente requiere dos cosas frías.
(Cuestión abierta menor para el director, NO bloqueante: ¿la gravedad debe acoplar también frío-tibio, o es
correcto que sólo frío-frío condense? La Teoría dice que la asimetría fría es la semilla de condensación; que sólo
lo ya-diferenciado gravite es defendible. Lo dejo como nota, no cambio la fórmula.)

## PREGUNTA 1 — S8/S9 antes de la exploratoria: SÍ, BLOQUEANTE. Construir el módulo de filtración AHORA.
La lectura de CC es correcta: S0-S9 COMPLETAS son bloqueantes; S0-S7 solas NO bastan para abrir la exploratoria.
Razón: S8 (control positivo detectable) y S9 (W uniforme NO adquiere topología por la filtración) validan el
INSTRUMENTO DE LECTURA — sin ellos, un resultado de la exploratoria no sería interpretable (no sabríamos si el
juez detecta métrica real ni si inventa estructura desde el empate). El módulo de filtración/jueces continuos
(§7.1-7.3) es JUSTO lo que S8/S9 necesitan Y lo que la exploratoria usará para leer topología. Construirlo ahora
mata dos pájaros: cierra la Puerta S y deja listo el lector de la exploratoria. SÍ, adelante.
Recordatorio de lo que el módulo debe tener (§7, ya adjudicado): jueces continuos SIN umbral (log-dispersión de W,
max h_i, k_eff, espectro Laplaciano ponderado); filtración por BLOQUES DE EMPATE (en W uniforme todos entran
juntos, NO desempatar por índice/RNG); 2º sello d_ij=−log(W_ij/maxW) con δ-Gromov, robusto a 1 transformación alt.
S9 es el test clave del lector: si la filtración inventa topología desde W uniforme, el lector está roto.

## PENDIENTES DECLARADOS (OK, en orden)
- II-POST (campo aleatorio permutación-covariante, R_t completo por paso no por-par): DESPUÉS de S8/S9 y de la
  exploratoria II-DET. Primero el control del no-go (II-DET) limpio; II-POST se añade como brazo generativo luego.
- Condensación de portadores (κ_H/τ_cond): no hace falta para NÚCLEO-II. Correcto aplazarlo al fold completo.

## INSTRUCCIÓN
1. Construir el módulo de filtración/jueces continuos (§7.1-7.3). 2. Correr S8 (control positivo: métrica conocida
detectable por β + 2º sello) y S9 (W uniforme NO da topología por filtración). 3. Si S8/S9 pasan → Puerta S
COMPLETA → abrir exploratoria NÚCLEO-II (barrer expansión continua, fijar anclas P-COHESIÓN/P-BORDE/P-DISOLUCIÓN
por persistencia-de-conectividad a través de la filtración, ANTES de mirar TODO-II). Reportar a CS. NO tocar el
fold de 5 brazos hasta que las anclas estén congeladas. Gravedad aprobada; declarar la propiedad de ≥2 focos.

## EN UNA LÍNEA
CC construyó el motor II separado bien: S0-S7 pasan, S7 (no-go) en piso de punto flotante sin amplificación, y cazó
el mismo bug de diagonal que yo — validación independiente de que el no-go se cumple limpio; la fórmula de gravedad
(cold_i·cold_j, ciega a índice/distancia) queda APROBADA con la propiedad declarada de que necesita ≥2 focos para
actuar; y sí, S8/S9 son bloqueantes — construir el módulo de filtración ahora cierra la Puerta S y a la vez deja
listo el lector de la exploratoria, así que adelante con él antes de tocar NÚCLEO-II.

— CS 🐝
