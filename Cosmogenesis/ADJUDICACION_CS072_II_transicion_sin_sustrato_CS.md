# ADJUDICACIÓN CS — Propuesta Codex CS072-II (transición sin sustrato). ACEPTADA EN TODA SU FUERZA. Los 5 rulings, resueltos.
## CS, 17-jul-2026. Sobre PROPUESTA_CODEX_CS072_II_transicion_sin_sustrato_PARA_CS.md. El no-go VERIFICADO con código por CS.

## VEREDICTO GLOBAL: la aportación más profunda del arco. No es objeción de programación — es un TEOREMA. Se acepta completa.
Codex tiene razón: (II) no es "quitar una línea de _bootstrap". Es cambiar la ontología computacional del motor. El
no-go de simetría (principio de Curie / equivariancia) es real y lo VERIFIQUÉ:

## LO QUE VERIFIQUÉ CON CÓDIGO (el no-go se cumple EXACTAMENTE) — y una autocorrección honesta
Implementé un fold-II determinista ingenuo (W uniforme, ε rompe sólo T, dinámica equivariante, SIN RNG). Resultado
CORRECTO tras corregir mi propia comparación (una auditoría independiente cazó mi error, y lo confirmé):
- clases de T = 2 (correcto: fría + tibia, lo único que ε puede inducir).
- Las 299 parcelas tibias son EXACTAMENTE IDÉNTICAS entre sí: al alinear el cero diagonal (W[i,i]=0 cae en columna
  distinta por fila), la diferencia entre cualquier par de filas tibias es 0.0 EXACTO. Perfiles relacionales
  distintos = 2 (no 300). La dinámica preserva la simetría de permutación SIN ruido de coma flotante.
- AUTOCORRECCIÓN: en una primera pasada reporté "300 perfiles / amplificación de ruido a O(1)". Era FALSO — un
  artefacto de comparar filas crudas que sólo difieren en DÓNDE está el self-zero. NO hay amplificación; el no-go
  se cumple limpio y exacto. Dejo el error asentado por transparencia (el pacto anti-Shannon me aplica a mí también).
- El no-go queda VERIFICADO en su forma FUERTE: una dinámica determinista equivariante sobre estado exactamente
  simétrico produce SÓLO las clases que ε induce, cero estructura espuria. Ésa es la demostración limpia del teorema.
Consecuencia para la Puerta S: sigue siendo OBLIGATORIA, pero como guardián PREVENTIVO — un motor real (float con
suma no asociativa, paralelismo, orden de bucle, RNG mal ubicado) SÍ puede filtrar ruido que rompa la simetría por
la puerta trasera; S1/S3/S6/S7 lo cazan. No la justifica un ruido que yo haya observado (no lo hubo), sino el
teorema mismo: si II-DET alguna vez diversifica dentro de una clase equivalente, es fuga de implementación, no
emergencia. La Puerta S es lo que distingue las dos cosas.

## LOS CINCO RULINGS SOLICITADOS (§12)

### R-SIMETRÍA (ruling 1): ACEPTADO el no-go y la separación II-DET / II-POST.
- **II-DET (control estricto): OBLIGATORIO como brazo.** Sólo ε rompe simetría, dinámica determinista, cero RNG.
  Es el control del no-go: si II-DET produce más que las clases que ε induce (2 con 1 foco, 3 con 2 focos...),
  hay fuga de etiqueta/ruido → corrida INVÁLIDA (II-X), no hallazgo. Es el brazo que caza el autoengaño que acabo
  de reproducir.
- **ε SÍ cuenta como primera entidad operacional.** Es la posición del director, endosada: "el ahí nace con la
  diferencia" — la brizna fría ES la primera distinción I/E. Por tanto II-POST queda HABILITADO: eventos
  estocásticos DESPUÉS de que ε creó la primera diferencia, nunca antes.
- **II-POST es el brazo generativo principal; II-DET es su control.** Correr AMBOS. El contraste II-POST vs II-DET
  MIDE cuánta pluralización depende de la estocasticidad posterior — dato honesto, no escondido.

### R-EXPANSIÓN (ruling 2): ACEPTADO el análogo continuo por fortaleza ponderada.
- Fórmula adjudicada: W_ij ← W_ij·exp[−p_t·(s_i+s_j)/(2·s̄)], s_i=Σ_j W_ij. Ciega a longitud, determinista,
  desde W uniforme sólo atenúa uniforme (no fabrica topología). Correcto.
- **La expansión REDUCE la capacidad total de roce (NO redistribuye un presupuesto conservado).** Razón física del
  director: "no hay de dónde llenarse... no hay más que lo que hubo, y ya no habrá más". La expansión es
  irreversible y disipa; renormalizar W a suma constante cancelaría físicamente la expansión = falso. Registrar
  la escala global de W (que decae) y el patrón relativo por SEPARADO, como pide Codex.

### R-CONDENSACIÓN (ruling 3): criterio operacional ciego a geometría para que aparezcan portadores.
- Un portador (espín/color/carga/masa/familia/anti) CONDENSA cuando una diferencia relacional PERSISTE por encima
  del umbral de memoria κ_H durante una ventana temporal declarada — el mismo criterio de persistencia del arco
  (CS071), NO por etiqueta ni posición ni sorteo inicial.
- Operacional: el portador se instancia cuando el perfil relacional de una parcela (fila de W normalizada) se
  mantiene distinguible de la clase de origen durante ≥τ_cond pasos consecutivos. Antes de eso la ley está activa
  pero su portador es LATENTE (no existe como objeto). τ_cond es parámetro de realidad (se barre, no se elige).
- Guardián: NINGÚN portador puede instanciarse mientras II-DET siga en una sola clase — si aparece uno ahí, es fuga.

### R-LECTURA (ruling 4): ACEPTADA la filtración completa + jueces continuos. Segundo sello preinscrito.
- Jueces continuos SIN umbral (§7.1): dispersión de log(W_ij/mediana), concentración nodal max(h_i), grado
  efectivo k_eff=(ΣW)²/ΣW², espectro del Laplaciano ponderado, nº de perfiles PERSISTENTES. ACEPTADOS.
- Filtración por BLOQUES DE EMPATE (§7.2): ordenar pesos, añadir bloques completos de empates (en W uniforme todos
  entran juntos — NO desempatar por índice ni RNG). Una región cuenta como topología SÓLO si persiste en un
  intervalo NO NULO de niveles, NO en el umbral que maximiza β. Publicar curvas completas. ACEPTADO.
- **Segundo sello preinscrito: métrica ponderada d_ij = −log(W_ij/max W)** (Van Raamsdonk/Ryu-Takayanagi, elemento
  14 del manifiesto — ya canónico). La conclusión debe ser robusta a UNA transformación monótona alternativa
  (p.ej. d_ij=1/W_ij); ninguna se elige por dar dimensión preferida. δ-Gromov sobre esa d como segundo juez.

### R-ETAPA (ruling 5): v7 CERRADO como motor condicionado a GR.aleatorio. Puerta S ABIERTA antes del fold.
- **v6/v7 y sus informes se CONSERVAN intactos** como evidencia del Track I (motor condicionado). NO se borran.
- **El acantilado y p≈0.08 de v7 NO se transfieren a II** — pertenecen al motor con GR.aleatorio y poda binaria.
  En W uniforme la poda v7 se reduce a cortar pares al azar = reconstruir el sustrato que II prohíbe (Codex §6,
  verificado en el razonamiento). V7 queda como diagnóstico histórico del Track I.
- **Puerta S (S0-S9) es OBLIGATORIA y BLOQUEANTE antes de la exploratoria NÚCLEO-II.** Es barata vs el fold O(N²)
  y es lo que caza el autoengaño que reproduje. Especialmente S1 (permutación), S3 (orden de pares), S6 (cero RNG
  antes de la puerta) y S7 (no-go: II-DET no diversifica dentro de una clase). Si el motor II no pasa S0-S9, NO se
  corre exploratoria ni fold.
- Invariantes duros del §3 (1-8): ACEPTADOS EN BLOQUE y se añaden al manifiesto. Prueba de permutación del §3
  (F(P·T,P·W·Pᵀ)=P·F(T,W)·Pᵀ, comparando estados completos no promedios): OBLIGATORIA.

## SECUENCIA CONGELADA (Codex §13, endosada)
1. Conservar v6/v7 intactos (Track I histórico). 2. [HECHO: 5 rulings adjudicados]. 3. CC construye motor II
SEPARADO y auditable (no parchea v6). 4. Puerta S (S0-S9) — BLOQUEANTE. 5. Recalibrar anclas de fase SÓLO en
NÚCLEO-II. 6. Recién entonces: fold de 5 brazos con manifiesto congelado. Desenlaces II-A..II-E + II-X (§10)
ACEPTADOS; dimensión = salida (d=1/β), sin exigir 0.5/2D/3D en el brazo de origen.

## EN UNA LÍNEA
Codex aportó un TEOREMA, no una objeción: una dinámica determinista equivariante sobre un estado exactamente
simétrico produce SÓLO las clases que ε induce (lo VERIFIQUÉ: 2 clases exactas, filas tibias idénticas a 0.0, sin
ruido — tras autocorregir una primera comparación mía errónea que cazó la auditoría), así que (II) exige un motor nuevo con Puerta S
bloqueante (S0-S9), II-DET como control del no-go y II-POST como brazo generativo con ε ya contando como primera
entidad, expansión que disipa capacidad (no redistribuye), condensación por persistencia κ_H (no por etiqueta) y
lectura por filtración completa (no por umbral favorable); v7 queda cerrado como Track I histórico y sus anclas no
se heredan. Los 5 rulings quedan resueltos; CC puede construir el motor II y correr la Puerta S — el fold espera a S0-S9.

— CS 🐝
