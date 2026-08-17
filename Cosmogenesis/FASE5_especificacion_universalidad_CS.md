# Fase V — especificación final: universalidad de S>0 ⇒ relación

**Fecha:** 9-ago-2026 · **Fuente:** síntesis de 2 borradores del equipo (`FASE5_preguntas_teoricas_para_Alexis_CS.md`
+ respuestas del equipo) más las decisiones editoriales tomadas acá donde los borradores divergían, con
autorización de Alexis para avanzar. **Se pre-registra ANTES de correr nada — el criterio de éxito y las
clases de salida quedan fijos de antemano, para que el barrido no pueda fabricar su propio resultado.**

## 0. Qué pregunta responde

¿La fecundidad que vimos toda la semana (S>0 ⇒ persistencia ⇒ historia ⇒ estructura) pertenece al **principio
relacional en sí**, o sólo al conjunto particular de ecuaciones que usa Cosmogénesis? Se responde generando una
población de reglas relacionales distintas, con el espacio de variación fijado de antemano, y viendo en qué
clases de comportamiento caen.

## 1. Filtro de admisión — P1 a P5 (no negociable, se aplica ANTES de clasificar nada)

Una regla candidata cuenta como instancia de S>0 ⇒ relación si y sólo si cumple las 5 a la vez. Si falla una,
queda descalificada — no es una instancia del principio, es otra cosa, y no entra al barrido:

- **P1 — Persistencia mínima:** existe un estado S tal que S(t+Δt)=f(S(t)) con f no trivial, y S(t)>0 implica
  probabilidad de S(t+1)>0 mayor que azar. Sin memoria (dinámica markoviana sin estado), descalificada.
- **P2 — Diferencia operable:** S no es escalar homogéneo — tiene al menos una dimensión interna de diferencia
  (fase, orientación, carga, tipo) que puede acoplarse. Sin esto, no hay relación posible.
- **P3 — Localidad relacional sin coordenadas:** la regla de actualización de un nodo sólo puede leer vecinos
  definidos por relación previa, nunca por índice global ni por (x,y,z) horneado.
- **P4 — Interacción recíproca:** si i afecta a j, j afecta a i en ventana finita (directa o indirectamente).
  Sin reciprocidad es campo externo, no relación.
- **P5 — Ausencia de valores físicos horneados:** sin G, ℏ, c, masa o escala de longitud puesta a mano.
  Complejidad algorítmica acotada (regla describible en <500 caracteres).

*(Mapeo teórico, para trazabilidad: P1≈C-N1, P2+P3≈C-N2, P4≈C-N2 recíproco, P1 con pérdida≈C-N3, P5≈O-N17.
No se usa esta correspondencia para generar las reglas — sólo para documentar de qué nodo viene cada filtro.)*

## 2. Ejes a barrer — 3 ejes, 18 clases (decisión: se usa la estructura del Borrador A; el eje "historia
irreversible sí/no" del Borrador B queda absorbido en P1 como filtro de admisión, no como eje — evita la
contradicción de barrer "sin memoria" cuando P1 ya lo excluye)

**Eje A — Fondo relacional (3 niveles):**
- A0: sin grafo — campo/estado con ruido pero sin estructura de vecindad emergente (equivalente a NULL-1/2).
  **Salvaguarda obligatoria (del Borrador B, riesgo O-N18.2):** al menos 1/3 de las reglas A0 deben ser
  representaciones genuinamente NO-grafo (campo continuo, autómata celular, red booleana) — no sólo variantes
  estadísticas de un grafo disfrazado. Si la fecundidad sólo aparece en representaciones de grafo, eso mismo
  es un resultado importante (S>0 sería propiedad de la representación, no universal).
- A1: grafo fijo generado al azar (equivalente al Erdős-Rényi+layout, ~52% de estructura ya medido).
- A2: grafo dinámico co-emergente con los nodos (equivalente a la malla causal REAL/NULL-3/4).

**Eje B — Retroalimentación entre relaciones (2 niveles):**
- B0: sólo entidad-entidad (equivalente a Fase IV sustratos 1-2-3).
- B1: relación-sobre-relación activa (equivalente a Fase IV sustrato 4, el único que rompió la Pared R7).

**Eje C — Costo/localidad (3 niveles):**
- C0: costo cero, mundo-pequeño libre (equivalente a CS066 sin poda).
- C1: costo por inconsistencia histórica/holonomía (equivalente a la poda dinámica, dio 0.786 vs 0.655).
- C2: costo + límite de escala duro (no probado aún esta semana — candidato nuevo a resolver mundo-pequeño).

**3×2×3 = 18 clases.**

## 3. Tamaño y presupuesto — dos etapas, no una

**Fase V-A (liviana, esta se ejecuta ahora):** 18 clases × 10 reglas con parámetros aleatorios dentro de
P1-P5 = **180 reglas**, motor liviano tipo CS064-068 (grafos/campos puros, N=500-2000, minutos por regla).
Objetivo: mapa de clases de universalidad. Presupuesto estimado: horas, no días.

**Fase V-B (pesada, PENDIENTE de checkpoint — no se lanza sin avisar):** sólo las clases que caigan en III o
IV en V-A (estimado 3-5 de las 18) se validan en Phantom con el generador de masa fija, ≥5 semillas REAL cada
una — el mismo patrón que CS073. Esto son ~15-25 corridas Phantom, no 180. Se avisa antes de comprometer ese
cómputo, como el resto de la sesión.

## 4. Clases de salida — 4 clases, pre-registradas con los números YA medidos esta semana

- **Clase I — Disolución:** z<3 vs NULL, cero estructura persistente, indistinguible de ruido puro.
- **Clase II — Mundo-pequeño congelado:** forma estructura pero pendiente log(diámetro)-vs-log(N) entre
  ~0.35-0.45 (el rango ya medido: real=0.376, barajado=0.420, ER=0.406) — estructura sin extensión.
- **Clase III — Geometría extensa:** pendiente sostenida >0.7-0.8 tras poda (ya medido: costo_P50=0.786), o
  formación de estructura con separación estadística fuerte (z>3) contra NULL emparejado.
- **Clase IV — Retroalimentación cerrada:** además de III, holonomía ≥5× menor que NULL (ya medido en Fase
  IV) y evidencia de cierre relación-sobre-relación (no sólo consenso global — ver §5.1 más abajo).

## 5. Criterio de éxito — combinado, pre-registrado

**Primario (estructural, del Borrador B — el más exigente):** ¿existe una correlación NECESARIA entre clase
estructural y fecundidad? Es decir: ¿las reglas SIN retroalimentación (B0) nunca alcanzan clase IV? ¿las
reglas SIN fondo relacional genuino (A0) nunca alcanzan clase II o superior? Si esas correlaciones se
sostienen limpias, es evidencia de que el principio relacional impone condiciones necesarias reales, no sólo
correlación estadística.

**Secundario (existencia, del Borrador A — el criterio operativo/de corte):**
- **Débil:** existe al menos una clase amplia (>15% de las reglas muestreadas en esa combinación de ejes) que
  cae en clase II o III, DISTINTA de la implementación específica de Cosmogénesis.
- **Fuerte:** existe una combinación de ejes que cae en clase III o IV de forma reproducible en >10 reglas
  independientes, y sobrevive la validación Phantom (V-B).
- **Muy fuerte:** clase III/IV es mayoritaria sobre clase II específicamente cuando B1 (retroalimentación) y
  C1/C2 (costo) están presentes juntos — diría que geometría extensa es un atractor bajo esas condiciones, no
  un accidente.

**No se pide mayoría sobre las 180 reglas totales** — sería fabricar una expectativa que el propio diseño no
justifica. Se pide existencia robusta de al menos una clase fecunda, más la estructura de correlación
necesaria/suficiente.

## 5.1 Nota sobre Fase IV y el caveat de consenso global

Independiente de Fase V-A/B, si se quiere blindar más la Clase IV específicamente, el Borrador de GPT-5.6 Sol
propone 4 controles adicionales sobre el sustrato 4 de Fase IV (F4-A a F4-D, cada uno conservando una cosa y
destruyendo otra) para separar "consenso global" de "cierre relacional local específico" con más precisión
que el 92%/8% ya medido en `FASE4_robustecido_CS.md`. Esto es un refinamiento de Fase IV, no parte del
barrido de Fase V — queda anotado, se puede correr por separado si Alexis lo pide.

## 6. Relación con O-N7.7 — DECISIÓN: Cosmogénesis prueba sólo el antecedente

Se adopta la posición más conservadora del equipo (GPT-5.6 Sol): O-N7.7 completo
(|Ω_proc|↓ ∧ |Ω_op|↑ ∧ LF↑) pertenece al bloque de Libertad Funcional, y la Regla de Plano de la Teoría exige
no mezclar niveles sin una transición explícita. **Cosmogénesis, en Fase V, prueba únicamente el antecedente:
¿existe restricción histórica capaz de reducir el espacio interno de una regla SIN aniquilar su capacidad
futura?** — no se reclama LF ni exaptación genuina desde estos datos.

**Observable para V-B, si se llega a esa etapa (reemplaza masa-en-sumideros Y el η_LF naïve ya cuestionado):**
branching efectivo de futuros. Desde un snapshot del gas/estado NO colapsado en t, generar un pequeño ensamble
de continuaciones bajo perturbaciones mínimas controladas, y medir B_τ = número efectivo de futuros
distinguibles (vía distancias de estado, clusters dinámicos, o entropía del ensamble). Distingue: colapso
(B_τ→0), ruido informe (B_τ grande sin persistencia), restricción generativa genuina (Ω_proc↓ pero B_τ
viable/estructurado ↑ — la firma que O-N7.7(b) predeciría, sin todavía llamarla "LF").

**La prueba completa de O-N7.7 con LF genuina queda para Célula Madre/ANIMA — no se fuerza acá.**

## 7. Qué NO se hace en esta etapa

- No se corre Phantom (Fase V-B espera checkpoint explícito).
- No se declara cierre ni veredicto sobre S>0 ni sobre O-N7.7, sea cual sea el resultado del barrido.
- No se rediseña NULL-3/4 con el factorial q_E/q_T propuesto por GPT-5.6 Sol en esta tarea — es un
  refinamiento de CS073, tema aparte, se puede retomar después si Alexis lo pide.

---

**Con esta especificación fija, Fase V-A queda lista para ejecutarse.**
