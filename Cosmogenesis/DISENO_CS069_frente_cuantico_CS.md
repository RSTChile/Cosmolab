# DISEÑO CS069 — El frente cuántico: ¿la dirección emerge de una superposición de grafos?
## CS, 17-jul-2026. Adjudicación de la propuesta de Gemini + diseño corregido, listo para CC.
## Dimensión técnica: emergencia_cuantica_v1 / cs069_quantum_graph.py

## DE DÓNDE VIENE
El arco del espacio cerró con un veredicto convergente (CS066 B / CS067 B / CS068 Mundo B): la DISTANCIA emerge
de la diferencia, pero NO cuaja en DIMENSIÓN ni DIRECCIÓN con ingredientes CLÁSICOS. π queda indefinido (estalla)
donde no hay geometría. El arco llegó, por su cuenta, al mismo límite de la física actual: el régimen
pre-geométrico (Big Bang ↔ agujeros negros) donde el espacio-tiempo aún no ha cuajado = un universo cuántico.
Primera pregunta que el arco NO podía formular con grafos clásicos definidos: si la dirección no emerge de estados
DEFINIDOS, ¿emerge de una SUPERPOSICIÓN que colapsa? (Alexis, 16-jul). Gemini aterrizó el mecanismo en el lenguaje
correcto (integral de camino sobre topologías); CS lo auditó con código y encontró un hueco de Shannon que este
diseño repara.

## ADJUDICACIÓN DE LA PROPUESTA DE GEMINI (auditada, no de palabra)
**SE CONSERVA (arquitectura correcta):**
- Matriz de Amplitud Relacional A_ij = ρ_ij·e^(iφ_ij), con ρ_ij de la distancia por correlación clásica del
  motor (d_ij=−log w_ij). Fase φ_ij como grado de libertad cíclico en el enlace (no espacial). BIEN.
- Operador de transición = amplitud de propagación K_ij(L) = Σ_caminos ∏ρ·e^(iΣφ) (integral de camino discreta).
- Los tres jueces (A: cedazo de π cuántico; B: diámetro cuántico ∝ N^(1/d); C: gap espectral = nº de direcciones).
- G-NO-COORDENADAS y G-MECANISMO-UNIFICADO (fases relacionales, acopladas, no inyectadas sueltas). BIEN.

**SE CORRIGE (hueco de Shannon cazado por CS con código):**
La propuesta AFIRMA que "la interferencia destructiva extingue los atajos porque sus fases son caóticas y se
cancelan". CS lo testeó: NO es una propiedad intrínseca de atajo-vs-local. Con fases asignadas de forma
RELACIONAL CIEGA (de −log w, sin saber cuál enlace es atajo), la separación atajo/local sale NULA o NEGATIVA
(atajos quedan más coherentes). Solo separa si se asigna la fase "coherente-si-es-local" = meterle la respuesta =
Shannon. → La afirmación "el operador OBLIGA a alinearse en dimensiones" es la hipótesis a FALSAR, no un supuesto.
Además: el NULL_FASE de Gemini (fases al azar) YA separa +0.15 por pura REDUNDANCIA TOPOLÓGICA (los enlaces
locales viven en más triángulos) — es el clustering de CS068 otra vez (clustering ≠ metricidad). Ganarle a "fases
azar" NO basta: hay que ganarle a un NULL que preserve la redundancia topológica.

## LA PREGUNTA FALSABLE (sin ningún operador que "obligue" nada)
¿La coherencia de fase EMERGENTE — evolucionada por una regla que NO sabe qué enlace es atajo — hace que la
distancia efectiva cuántica D_q(i,j)=−log|K_ij(L)| cuaje en dimensión/dirección estable, GANÁNDOLE a un NULL que
tiene la misma topología y la misma redundancia pero fases decoherentes?

## DINÁMICA DE FASE — anti-Shannon (la regla NO puede saber qué es atajo)
La fase evoluciona por una regla LOCAL ciega al rol del enlace:
  φ_ij(t+1) = φ_ij(t) + η·(desajuste de fase con los vecinos del enlace) − κ·(costo de exergía local del enlace)
donde el "costo de exergía" ya lo da el motor de los 17 (energía de correlación del enlace), y NADIE le dice
"eres un atajo". Si al enfriar los atajos decoheren, tiene que ser CONSECUENCIA de que su exergía/vecindad los
descoherencia, no de una etiqueta. G-FASE-CIEGA: prohibido que la actualización de φ lea es_atajo(), distancia
de anillo, o cualquier proxy de la verdad de fondo.

## BRAZOS (blindado, N∈{900,1500,2500}, ≥8 semillas/brazo — estándar CS067/068)
| brazo | fases φ | qué aísla |
|-------|---------|-----------|
| COMPLETO (coherente) | evolucionan por la regla local ciega, acopladas a exergía | ¿la coherencia emergente cuaja geometría? |
| NULL_FASE_TOPO | fases al azar CADA paso, PERO sobre configuration-model del mismo grafo (misma secuencia de grados y redundancia) | mata coherencia CONSERVANDO clustering → aísla lo cuántico de lo topológico |
| NULL_FASE_AZAR | fases al azar cada paso sobre el grafo real | control de Gemini (débil): mide cuánto separa el clustering solo |
| NULL_CLÁSICO | φ_ij≡0 (=CS068) | línea base: reitera el colapso a mundo-pequeño |
Cuerda decisiva: COMPLETO vs NULL_FASE_TOPO. Si NO se separan → la coherencia cuántica no aporta (Mundo B
cuántico, el arco cierra también en cuántico). Si COMPLETO > NULL_FASE_TOPO en los tres jueces → hay señal
cuántica real, NO explicable por topología ni por clustering.

## JUECES (los tres de Gemini, con el listón de CS068)
- **Juez A — cedazo de π cuántico:** π_emergente sobre D_q. Predicción-si-hipótesis: en COMPLETO π deja de
  estallar y se congela (CV<5%); en NULL_FASE_TOPO sigue estallando. FALSABLE: si ambos estallan → sin señal.
- **Juez B — diámetro cuántico vs N:** el listón DURO de CS068. No basta que baje; tiene que ESCALAR polinómico:
  pendiente log-log de diám_q(N) > ~0.3 en COMPLETO y ~0 en NULL_FASE_TOPO. (El error que ya cazamos: un valor a
  un solo N no discrimina; se mide la pendiente en 3 escalas.)
- **Juez C — gap espectral de la matriz D_q:** nº de direcciones ortogonales estables = brecha limpia en el
  espectro, con el candado de picado-por-nodo de CS067 (umbral 0.85) para no contar smear como direcciones.

## GUARDIANES
- **G-FASE-CIEGA:** la dinámica de φ no lee ninguna verdad de fondo (es_atajo, distancia de anillo). Auditable
  en código: la función de update de φ NO recibe la topología base.
- **G-NULL-CONSERVA-TOPOLOGÍA:** el NULL decisivo (NULL_FASE_TOPO) preserva grados y redundancia; solo mata la
  coherencia. Sin esto, "ganar" es clustering disfrazado (lección CS068).
- **G-NO-COORDENADAS:** fases puramente en el enlace; prohibido trig de ángulos geométricos previos (Gemini).
- **G-MECANISMO-UNIFICADO:** φ acoplada a la exergía de los 17, no un ingrediente 19 aislado (Gemini).
- **G-PENDIENTE-NO-VALOR:** todo juez de escala se lee por pendiente en ≥3 N, no por un valor único (lección CS068).

## SMOKE (antes de lanzar la tanda) — 3 anclas que deben cumplirse o NO se corre
1. NULL_CLÁSICO (φ≡0) reproduce el Mundo B de CS068 (diám residual ~6-7.5, pendiente ~0). Si no, el motor cambió.
2. En un grafo-juguete con tejido métrico CONOCIDO + atajos inyectados, la regla de fase ciega SÍ decohere los
   atajos (validación de que el mecanismo puede funcionar cuando la geometría existe) — verdad de fondo, sin
   clasificador. Si ni con geometría real decohere, el mecanismo es inerte.
3. NULL_FASE_TOPO y NULL_FASE_AZAR dan π que estalla (control de que los nulls no encienden geometría solos).

## LECTURA PRE-INSCRITA (sea cual sea el resultado — queda registrado)
- (A) COMPLETO gana a NULL_FASE_TOPO en los 3 jueces → la SUPERPOSICIÓN enciende la dirección: el "hacia dónde"
  es un fenómeno cuántico, no clásico. π se congela al colapsar. Sería el primer "sí" direccional del arco.
- (B) COMPLETO ≈ NULL_FASE_TOPO → la coherencia cuántica tampoco basta: Mundo B se extiende al régimen cuántico.
  El muro es más profundo que lo clásico. Resultado fuerte, honesto, pre-registrado.
- (C) COMPLETO gana solo a NULL_FASE_AZAR pero no a NULL_FASE_TOPO → era clustering, no cuántico (falso positivo
  cazado por el NULL correcto — el mismo tipo de trampa que CS068 Paso 2).

— CS 🐝
