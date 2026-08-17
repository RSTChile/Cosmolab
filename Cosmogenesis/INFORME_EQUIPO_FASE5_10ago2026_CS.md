# Informe para el equipo — Fase V y cierre del análisis de la batería (9-10 de agosto de 2026)

**Para:** Equipo Transinteligente · **Preparado por:** CC (Claude Code) · **Dirige:** Alexis López Tapia ·
**Continúa de:** `INFORME_EQUIPO_SESION_5-8ago2026_CS.md` (Fases I-IV, 5-8 ago) — este documento cubre lo que
pasó después: la recepción del análisis del equipo sobre esa batería, y la ejecución completa de Fase V.

## Cómo leer este informe

Mismos símbolos que el informe anterior: ✅ separación estadística sólida · ⊘ señal parcial o con caveat real ·
❌ no se separa del control · ⚠️ falla metodológica encontrada y corregida en el camino · 🔧 infraestructura.
Todo verificado contra archivos en disco, no contra lo que un agente narró haber hecho — varias notificaciones
de esta sesión resultaron prematuras o repetidas; los números de este informe salen de los `.md`/`.csv` que
efectivamente quedaron escritos.

---

## PARTE 1 — Qué aportó el análisis del equipo sobre la batería de Fases I-IV

Alexis trajo el análisis de GPT-5.6 Sol y de un segundo analista (marco "canónico", con nodos O-N/C-N
explícitos) sobre el informe anterior. Puntos que se incorporaron formalmente:

1. **CS073 se reformula en 2 componentes**, no "tiene grafo vs no tiene grafo": estructura ≈ componente
   genérico de relajación relacional (grafo cualquiera + `layout_resortes`, ~52% de REAL, ya medido con
   Erdős-Rényi) + componente específico de topología (malla causal cercana/idéntica a REAL, ~100%).
2. **Precisión de alcance sobre "6 semillas REAL":** el p≈1/3003 (Fase II) prueba robustez frente a distintos
   `seed_layout` de UNA topología — no que mallas causales generadas independientemente produzcan el fenómeno
   como clase. Corrección importante para no citar ese número más allá de lo que prueba.
3. **La grieta de la Pared R7 (Fase IV) se renombra: "recursividad relacional".** Hipergrafo = relación SOBRE
   entidades; el 2-complejo activo = relación que MODIFICA a otras relaciones. Repite, un nivel más arriba, el
   patrón A=0/B=0/C-discrimina que `cs052_v1_coemergencia.py` ya había encontrado hace meses.
4. **NULL-3 vs NULL-4 (CS073):** se propuso un diseño factorial (q_E=fracción de aristas conservadas,
   q_T=fracción de orden conservado) en vez de seguir sumando puntos sueltos — queda anotado, no ejecutado
   todavía, es un refinamiento de CS073 aparte de Fase V.
5. **Debate sobre O-N7.7:** GPT-5.6 Sol argumentó que el nodo completo (con Libertad Funcional genuina)
   pertenece al bloque de LF y a ANIMA/Célula Madre, no a Cosmogénesis — la Regla de Plano de la Teoría exige
   no mezclar niveles sin transición explícita. Cosmogénesis, como mucho, prueba el antecedente (restricción
   histórica que reduce sin destruir capacidad futura). **Esta posición se adoptó formalmente en la
   especificación de Fase V** (ver Parte 2, §6).
6. **Dos borradores completos de especificación de Fase V** llegaron ya trabajados por el equipo, con ejes
   distintos (3 vs 4) — se sintetizaron en la especificación final, con decisiones editoriales explícitas
   donde divergían.

---

## PARTE 2 — Fase V: especificación final pre-registrada

Documento completo: `FASE5_especificacion_universalidad_CS.md`. Resumen de las decisiones tomadas:

**Filtro de admisión P1-P5** (una regla debe cumplir las 5 para contar como instancia de S>0⇒relación):
persistencia mínima (memoria no trivial), diferencia operable (no homogénea), localidad relacional sin
coordenadas, interacción recíproca, ausencia de valores físicos horneados + complejidad acotada.

**3 ejes = 18 clases** (se descartó un 4º eje "historia irreversible sí/no" propuesto por un borrador porque
contradecía su propio filtro P1, que ya excluye reglas sin memoria):
- **Eje A — fondo relacional:** A0 sin grafo (con salvaguarda: ≥1/3 deben ser representaciones NO-grafo
  genuinas — campo continuo, no solo estadística disfrazada de grafo) · A1 grafo fijo aleatorio · A2 grafo
  dinámico co-emergente.
- **Eje B — retroalimentación entre relaciones:** B0 sólo entidad-entidad · B1 relación-sobre-relación activa.
- **Eje C — costo/localidad:** C0 costo cero · C1 costo por inconsistencia/holonomía · C2 costo + límite de
  escala duro (nunca probado limpio antes de esta semana).

**4 clases de salida**, con umbrales numéricos ya fijados de las mediciones reales de esta semana: Clase I
Disolución (z<3) · Clase II Mundo-pequeño congelado (pendiente 0.35-0.45) · Clase III Geometría extensa
(pendiente >0.7-0.8) · Clase IV Retroalimentación cerrada (además de III, holonomía ≥5× menor que NULL).

**Criterio de éxito combinado:** primario = correlación necesaria/suficiente entre clase estructural y
fecundidad (más exigente); secundario = existencia débil/fuerte/muy fuerte de al menos una clase fecunda (más
operativo). Se pre-registró ANTES de correr nada.

**O-N7.7:** Cosmogénesis prueba sólo el antecedente, no LF completo (decisión de Parte 1, punto 5). Si se llega
a Fase V-B, el observable es branching efectivo de futuros (B_τ), no masa en sumideros ni el η_LF ya
cuestionado.

---

## PARTE 3 — Ejecución de Fase V-A (barrido liviano)

### 3.1 Piloto y corrección metodológica (⚠️→🔧)

Piloto inicial (5 clases × 3 reglas): el generador y el filtro P1-P5 quedaron validados como genuinamente
discriminantes (una regla deliberadamente rota, con constante física puesta a mano, fue correctamente
rechazada). Pero **el 90% de las reglas cayó en Clase I por un bug de método**: la pendiente se medía sobre
grafos independientes a cada N, mientras que los umbrales de la especificación se habían calibrado
originalmente (Fase III) agrupando UN MISMO grafo por escalas (coarse-graining). Mismo sustrato: 0.11-0.36
(método viejo, espurio) vs. 0.459 (coarse-graining, coincide con las mediciones previas del proyecto).
**Corregido** — el motor ahora coarse-grana un solo grafo por regla, tal como se calibraron los umbrales.

### 3.2 Barrido completo — 180 reglas (150 ejecutadas, 3/18 combos no ejecutables)

**Bug real encontrado y documentado, no oculto:** las combinaciones A0+B1 (campo sin grafo + retroalimentación
activa) hacen crashear el motor — la función de retroalimentación siempre espera una adyacencia de grafo, que
A0 nunca tiene. Nunca se había probado esta combinación en el piloto chico. No se forzó ni se ocultó; se marcó
como "no ejecutable" y se siguió con las 15 combinaciones restantes.

**Mapa global (150 reglas):** Clase I 66% · Clase II 29% · Clase III 5% (8 reglas) · Clase IV 0%.

**Correlaciones necesarias (criterio primario):**
- "B0 nunca Clase IV" — se sostiene, pero trivialmente (nadie, ni B0 ni B1, llegó a Clase IV — no informativo).
- ❌ **"A0 nunca Clase II+" — FALSIFICADA.** 27% de las reglas A0 medibles llegaron a Clase II. Causa
  sospechada, no confirmada: incluso los sustratos sin grafo reciben un "grafo de medición" derivado para
  poder aplicarles coarse-graining — ese grafo derivado podría producir mundo-pequeño como artefacto de la
  MEDICIÓN, no del campo real. **Sin resolver para A0.**

**Criterio secundario:** Débil ✅ (14/15 combos superan 15% en clase II/III). Fuerte ⊘ (no certificable sin
Phantom). **Muy fuerte ❌ CONTRADICHO** — la predicción exigía Clase III/IV mayoritaria cuando
retroalimentación (B1) Y costo (C1/C2) están juntos; en cambio, los 4 combos B1+C1/C2 dieron CERO Clase III, y
la única señal vino de **B0+C2** (sin retroalimentación, sólo límite de escala duro). Revierte la expectativa
heredada de Fase IV (donde retroalimentación era la clave) — son preguntas relacionadas pero no idénticas (Pared
R7 = física de partículas de a tres cuerpos; esto = extensión geométrica macroscópica), no hay que fusionarlas
sin más.

### 3.3 A2-B0-C2 — el candidato, confirmado y caracterizado

Primera aparición en el piloto (2/3 Clase III), profundizado después a n=30 (10 originales + 20 nuevas):
**13 Clase I (43.3%), 15 Clase III (50.0%), 1 Clase II, 1 "intermedio"** (curva no-monótona anómala, dejada
sin forzar) — **patrón bimodal fuerte, 93.3% en los extremos, casi nada en el medio.**

✅ **Verificado por inspección de código que NO comparte la duda de A0:** A2 mide directamente sobre el mismo
grafo real que corrió la dinámica (incluida la poda por límite de escala) — nunca sobre un grafo derivado
artificial.

⊘ **Pista sobre qué separa Clase I de Clase III, no umbral limpio:** dos parámetros correlacionan
moderadamente — `kcap` (límite de escala, más estricto → más Clase III, r=-0.43) y `K` (tamaño del alfabeto de
fase, más grande → más Clase III, r=+0.45). Combinados: 83% Clase III vs 42% en el resto (n=18, sugestivo).

**A2-B0-C2 es el candidato recomendado por el pipeline para Fase V-B (validación en Phantom) — pendiente de
autorización de Alexis, no lanzado.**

---

## PARTE 4 — Síntesis para el equipo

**Lo que se sostiene:** el principio S>0⇒relación SÍ produce, en un espacio de reglas amplio y pre-registrado,
al menos una clase reproducible de geometría extensa (A2-B0-C2, bimodal y robusto a n=30) — apoyo real, aunque
acotado, al criterio "débil" de universalidad.

**Lo que se cae:** la hipótesis "muy fuerte" (retroalimentación+costo es la combinación clave) se contradice
con los propios datos de Fase V — el eje que realmente correlaciona con geometría extensa en este barrido es
el límite de escala duro (C2) solo, no la retroalimentación que fue protagonista en Fase IV. Dos preguntas
relacionadas, resultados que no se transfieren limpio de una a la otra.

**Lo que queda abierto:** el posible artefacto de medición en A0 (sin resolver, afecta la lectura de "A0 nunca
llega a Clase II"); si A2-B0-C2 pasa a Phantom o no (checkpoint pendiente); el diseño factorial q_E/q_T para
NULL-3 vs NULL-4 (anotado, no ejecutado); y el refinamiento del umbral kcap/K si se quiere perseguir la pista.

---

*Documentos fuente: `FASE5_especificacion_universalidad_CS.md`, `FASE5A_piloto_resultado_CS.md`,
`FASE5A_completo_resultado_CS.md`, `FASE5A_profundizar_A2B0C2_resultado_CS.md` — todos en este directorio, con
metodología completa y CSVs de datos crudos para auditoría directa.*
