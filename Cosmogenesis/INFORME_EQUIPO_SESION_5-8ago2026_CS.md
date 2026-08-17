# Informe para el equipo — sesión 5 al 8 de agosto de 2026

**Para:** Equipo Transinteligente (analistas de la Teoría Cosmosemiótica y colaboradores de Cosmogénesis) ·
**Preparado por:** CC (Claude Code), orquestando agentes dedicados por experimento · **Dirige:** Alexis López
Tapia · **Cubre:** ejecución completa del roadmap consolidado del 5-ago (Fase I a IV) + un hilo teórico nuevo
(O-N7.7) introducido a mitad de sesión.

## Cómo leer este informe

Cada resultado se reporta con número y método, nunca con veredicto cerrado — la regla de la casa de este
proyecto es que ningún cierre de experimento es válido sin autorización explícita del director, ni con
resultado perfecto. Los símbolos:

✅ = separación estadística sólida de un control nulo/barajado · ⊘ = señal parcial, débil, o con caveat
metodológico real · ❌ = no se separa del control (incluye "cero absoluto") · ⚠️ = resultado retractado o con
falla metodológica encontrada y corregida en el camino · 🔧 = infraestructura (no es un resultado de la Teoría
en sí, habilita otros experimentos.

Todo lo reportado acá está verificado contra archivos en disco (no contra lo que un agente *dijo* haber
hecho) — varias notificaciones de agentes a lo largo de la sesión resultaron prematuras o parcialmente
narradas; los números de este informe se tomaron de los `.md`/`.csv`/`.json` que efectivamente quedaron
escritos, releídos directamente.

---

## PARTE 1 — Fase I: cinco experimentos baratos (roadmap multi-IA, prioridad P0)

| # | Experimento | Nodo(s) de la Teoría | Resultado |
|---|---|---|---|
| I-A | Dirección temporal T⁺/T⁻ a nivel micro (`cs076`) | C-N2.5.6-10 | ❌ Zona gris — el estadístico decisivo (KL de balance detallado vs. NULL barajado) cae dentro del NULL (z=-0.02 con memoria, z=+1.63 sin memoria, umbral z≥3). No hay asimetría temporal micro distinta de la ya confirmada a nivel agregado (C-N3/CS009). |
| I-B | Gradiente vs. estabilización accidental (`cs077`) | C-N2.6.1-4 | ✅ REAL forma 7-18× más masa ligada que dirección-azar en TODOS los puntos de ε probados; CV entre semillas 12% (REAL) vs 75% (azar) — el criterio de falsación no se cumplió, el nodo se sostiene con esta evidencia. |
| Infra | Lector de volcados binarios Phantom en Python | 🔧 | ✅ Resuelto con `sarracen`. Desbloqueó C-N4, la jerarquía NULL de CS073, y todo lo que sigue en este informe. |
| I-D | κ_V con test de permutación riguroso (`cs078`) | Bloque 2.8 (κ_V) | ⊘ Dirección correcta en todo método probado, pero p=0.111 — el **piso matemático exacto** con n=9 (1 REAL + 8 NULL), no una falla de método. |
| I-C | Delimitación real C-N4 en Phantom (`cs079`) | C-N4 | ⊘ 3 de 4 métricas separan REAL/NULL limpio (sin overlap); la métrica más directamente ligada a la pregunta (profundidad del valle) no discrimina, y los 8 NULL muestran una forma bimodal sospechosamente idéntica entre sí — posible artefacto de configuración compartida, sin resolver. |

---

## PARTE 2 — Fase II: jerarquía NULL-0 a NULL-5 para blindar CS073

**Contexto:** CS073 (el resultado más sólido del proyecto hasta ahora) mostró que una malla causal genuina,
llevada a posiciones físicas por `layout_resortes` y corrida en Phantom (gravedad+SPH), forma sumideros
(proto-estrellas) mucho más que un control al azar. Esta parte diseñó y corrió 6 controles independientes para
saber EXACTAMENTE qué información específica es la responsable.

**Prerrequisito resuelto — el cuello de botella de 1 sola semilla REAL:** todos los tests de permutación
chocaban con un piso matemático (p≥1/9=0.111, imposible de bajar con 1 sola semilla REAL). Se identificó
`seed_layout` (semilla de inicialización del layout de resortes, manteniendo fija la topología de la malla
causal) como parámetro legítimo para generar réplicas REAL genuinas. **5 semillas REAL nuevas** (301-305):
masa 2124.4/2209.0/2209.0/2293.6/2049.2/2293.6 (media=2196.5, CV=4.4%, ninguna cerca del rango NULL). El piso
de p bajó de 0.111 a **1/3003≈0.000333**.

| Escalón | Preserva de REAL | Destruye | Resultado (masa sumideros, N=2000) | Veredicto |
|---|---|---|---|---|
| **NULL-0** | masa total | todo lo demás (chequeo trivial) | masa idéntica=18800.0 en 9/9 corridas | ✅ terreno parejo, sin señal de estructura por sí solo |
| **NULL-1** | perfil radial (ángulo al azar) | toda correspondencia relacional | **0/8 sumideros** | ❌ cero absoluto |
| **NULL-2** | espectro de potencia P(k), método Zel'dovich | fase / orden superior | **0/8 sumideros** | ❌ cero absoluto (mejorado de KS=0.495 a 0.220 con desplazamiento tipo Zel'dovich vs. muestreo por rechazo; no cambió el resultado) |
| **NULL-3** | grado + longitud de arista (~87.5% de aristas intactas) | ~12.5% de aristas específicas + ~28-35% de motivos/triángulos | 8/8, media=2186.7 | ✅≈REAL (p=0.42, sin separación) |
| **NULL-4** | topología 100% completa | orden de inserción/formación | 3/3, media=2136.9 | ✅≈REAL/NULL-3 (p=0.18) |
| **NULL-5** | topología 100% + posiciones 100% | sólo la etiqueta nodo↔posición | **colapso trivial, exacto a REAL** | 🔧 hallazgo estructural: el pipeline no tiene NINGÚN canal por el que la identidad de nodo pueda importar (masa y `h` son constantes globales, velocidad es función pura de posición) — verificado antes y después de Phantom |

**Patrón consolidado, con 6 controles independientes:** el corte no está en los detalles finos — está entre
"tiene algún grafo/proceso relacional de fondo" (NULL-3, NULL-4, REAL: todos ≈mismo resultado) y "no tiene
ninguno" (NULL-1, NULL-2: cero absoluto). Ni el perfil radial global ni el espectro de potencia de 2 puntos
alcanzan por sí solos. Casi cualquier aproximación del grafo relacional —aunque cambien qué aristas específicas
hay, o en qué orden se formaron— reproduce REAL.

**Hallazgo adicional — proceso físico vs. identidad del grafo:** un grafo Erdős-Rényi *completamente ajeno* a
la malla causal real (0.24% de overlap de aristas, ~0 motivos), pasado por el MISMO `layout_resortes`, formó
sumideros en las 8 semillas — pero a **≈52% de la masa de REAL** (media=1143.3, p=3.3e-4 vs REAL, p=7.8e-5 vs
NULL-1/2/3). Es una posición intermedia limpia, no ruido: el proceso físico de relajación aporta una base real
de estructura incluso sobre un grafo sin ninguna relación con REAL, pero esa base sola no llega a la mitad de
lo que aporta la identidad específica de la malla causal.

**Confound de infraestructura encontrado y corregido en el camino:** todos los pilotos a escala reducida
(N=500) daban sistemáticamente cero, no por la estructura probada sino porque el generador hacía crecer el
lado de la caja física con `n^(1/3)` — a menor N, menos masa total absoluta (4700 a N=500 vs 18800 a N=2000),
no sólo menos resolución. Se corrigió con un generador de masa fija (⚠️→🔧, `INFRA_masa_fija_generador_CS.md`).
**Hallazgo posterior a la corrección:** con masa ya fija, N=500 SIGUE dando cero — la resolución SPH pura
(vecinos por partícula, criterio de Bate & Burkert ~116) resultó ser el factor dominante, no la masa como se
sospechaba. El piso real de resolución está entre N=500 y N=1000.

---

## PARTE 3 — O-N7.7 (nodo teórico nuevo, no estaba en el roadmap original)

**Contexto:** Alexis introdujo el nodo O-N7.7 (Kimi/Moonshot AI, 7-ago-2026) — distingue **acumulación
adaptativa** (más recursos dentro de la misma arquitectura generativa, ΔLF≈0, saturación) de **condensación
exaptativa** (la historia reduce el espacio de procesamiento Ω_proc mientras el dominio operativo Ω_op y la
Libertad Funcional LF suben — "la historia comprime el mecanismo mientras la exaptación expande la
capacidad"). Métrica propuesta: η_LF = LF/|Ω_proc|.

**Intento 1 — η_LF sobre datos existentes:** usando triángulos del grafo como Ω_proc y masa en sumideros como
Ω_op, NULL-1/NULL-2 dieron η indefinido (0/0, consistente); NULL-3 dio η levemente mayor que REAL pese a menos
motivos (consistente con la predicción). **Pero el grafo random+layout dio el η MÁS ALTO de los 5 (54.4)** —
porque su Ω_proc≈0 viene de *ausencia* de estructura, no de *filtrado histórico*. El cociente simple no
distingue esas dos causas, que es justo lo que el nodo necesita separar. ⊘ **la métrica, tal como se
operacionalizó, no es suficiente.**

**Intento 2 — diseño y piloto de Sistema A (acumulación) vs. Sistema B (condensación):** primer piloto
(N=50-400) dio cero en las 9 corridas — heredó el mismo confound de caja/masa de la Parte 2. Corregido el
mecanismo de Sistema B (memoria genuina de trayectoria verificada: Jaccard cae monótono con más "historia" H,
de 1.0 a 0.56), y re-corrido con masa fija a escala N=2000-8000: **ambos criterios de falsación fueron a
contramano de la predicción.** Sistema A: ganancia marginal de N=2000→4000 fue **grande y positiva** (+0.35,
sin saturación). Sistema B: masa en sumideros **cayó monótono** con más historia H (1118→996→837→808→705).

**⚠️ Corrección crítica de Alexis, posterior a estos resultados:** el observable usado en todo el hilo O-N7.7
(masa en sumideros) es conceptualmente el equivocado para esta pregunta. Un sumidero, en el modelo de Phantom,
es un horizonte — la materia que cruza no tiene más historia posible, y el sumidero no emite nada de vuelta
(sin retroalimentación, sin radiación). Masa-en-sumideros mide el FIN de la historia de esa materia, casi lo
opuesto de "dominio operativo creciente" que pide O-N7.7. **Los resultados de Sistema A/B (y del cálculo de
η_LF) quedan en suspenso, no refutados ni confirmados — el instrumento de medición necesita rediseñarse antes
de que estos números signifiquen algo sobre la Teoría.** Candidato propuesto, no implementado: alguna medida
sobre el gas que NO cae (diversidad de configuraciones futuras aún posibles, estructura diferenciada
sobreviviente), no sobre lo que ya colapsó.

**Estado de O-N7.7 en Cosmogénesis: abierto, con una lección metodológica clara para quien lo retome.**

---

## PARTE 4 — Fase III: ¿se resuelve el "mundo pequeño"? (línea CS064-068, distinta de CS073)

**Experimento 1 — renormalización por coarse-graining:** sobre el mejor tejido disponible del arco CS064-068
(CS066, brazo `local`, k_local=6), agrupando por escalas b=2,4,8,16,32. **❌ Resultado B reforzado, no A:** la
pendiente log(diámetro)-vs-log(N) es indistinguible entre real (0.376), barajado (0.420) y Erdős-Rényi puro
(0.406) — sigue siendo mundo-pequeño a CUALQUIER escala de agrupamiento. Hallazgo lateral no buscado: la
estructura real se FRAGMENTA más rápido que el ruido puro al agrandar la escala (componente gigante 91%→44%,
mientras el ruido puro se mantiene estable 3.7-4.4).

**Experimento 2 — poda dinámica por costo de enlace:** en vez de podar "atajos" por un criterio externo (ya
fallido en CS068), se les asignó a los enlaces un costo (inconsistencia histórica + conflicto de holonomía +
bajo soporte local + baja reciprocidad) y se podó por percentil de costo, comparado contra podar al azar la
MISMA cantidad. **⊘ Podar por costo da sistemáticamente pendiente mayor que podar al azar**, en los 3 niveles
probados (P50/P70/P90), reproducible entre 3 semillas: costo_P50=0.786 vs azar_P50=0.655 vs sin_poda=0.421. El
criterio de costo captura algo real sobre cuáles enlaces son los "culpables" — pero el efecto es modesto frente
al de podar en sí (podar al azar ya mueve bastante la aguja), y ningún nivel se acerca a lo que sería una
geometría 3D genuina.

---

## PARTE 5 — Fase IV: relaciones de orden superior, ¿se supera la Pared R7?

**Antecedente:** en CG002 (arco fundacional, cerrado 30-jun-2026), todo lo "de a pares" (color, carga)
funcionaba; el gluón y el Higgs (ambos vértices de 3 cuerpos) quedaron bloqueados — la "Pared R7". Un intento
posterior (CS032) de agregar una fuerza de 3 cuerpos ENCIMA del mismo grafo pareado no la resolvió ("la pared
se movió, no cayó").

**4 sustratos construidos (N=110, mismo grafo base, presupuesto de cómputo igualado):** (1) grafo diádico
[línea base], (2) hipergrafo [triángulos como relación arity-3, SIN retroalimentación], (3) complejo
simplicial [cara que sólo MIDE holonomía, pasiva], (4) 2-complejo [la cara empuja activamente sobre sus 3
aristas de borde — retroalimentación relación-sobre-relación].

**✅ Sólo el sustrato 4 se separó de NULL con solidez** (holonomía ~5× menor que el control, varianza más chica
de los 4, reproducible en 5 semillas). Los sustratos 1, 2 y 3 — incluido el hipergrafo con aridad 3 pero SIN
retroalimentación — se comportaron estadísticamente como el grafo pareado. **La aridad relacional sola no
supera la Pared R7; hace falta que una relación pueda actuar sobre otras relaciones.** Esto reproduce, un nivel
de estructura más arriba, el patrón A=0/B=0/C-discrimina que `cs052_v1_coemergencia.py` ya había encontrado
años atrás (una entidad sola no alcanza, un vínculo suelto no alcanza, sólo un vínculo atado a ambos extremos
discrimina). ⊘ Caveat: un control barajado muestra que parte del efecto es consenso/distribución global, no
puro cierre topológico local — señal real pero más mixta de lo que sugiere REAL-vs-NULL solo.

---

## PARTE 6 — Tabla consolidada por nodo de la Teoría

| Nodo | Antes de esta sesión | Después de esta sesión |
|---|---|---|
| C-N2.5.6-10 (dirección temporal) | ⏳ sin experimento | ❌ zona gris, sin señal micro más allá de lo agregado |
| C-N2.6.1-4 (gradientes/atractores) | ⏳ sin experimento | ✅ sostenido, REAL 7-18× sobre control azar |
| C-N4 (delimitación) | ⏳ sin experimento | ⊘ 3/4 métricas separan, artefacto de NULL sin resolver |
| Bloque 2.8, κ_V | ⊘ z=1.37, débil | ⊘ p=0.111, piso matemático de n=9 (no de método) |
| C-N2.7 (régimen de acoplamiento, vía CS073) | ✅ z=48.69 con n=1 REAL | ✅✅ reforzado: p=0.000333 con n=6 REAL; blindado por 6 controles NULL independientes; se aisló que la información crítica es "tener algún grafo relacional de fondo", no perfil radial ni espectro de potencia solos |
| C-N2.7.7-12 (distancia→dimensión→dirección, π contingente) | ✅ vía CS066-069 | ❌ reforzado en la dirección negativa: renormalizar (Fase III) no resuelve el mundo-pequeño en ningún régimen probado |
| O-N7.7 (nuevo, 7-ago) | — | ⚠️ abierto — observable usado (masa en sumideros) identificado como conceptualmente inadecuado por el propio director; sin resultado válido todavía |
| Pared R7 (gluón/Higgs, de a 3) | ❌ bloqueada desde 30-jun-2026 | ⊘ primera grieta genuina en meses: retroalimentación relación-sobre-relación (no aridad sola) separa de NULL |

---

## PARTE 7 — Qué queda pendiente / preguntas para el equipo

1. **O-N7.7** necesita un observable rediseñado (algo sobre el gas que NO cae, no sobre masa acretada) antes
   de que los resultados de Sistema A/B signifiquen algo — ¿alguien del equipo tiene una propuesta concreta de
   qué medir?
2. **NULL-3 vs NULL-4**: ambos reproducen REAL, pero por vías distintas (identidad de aristas vs. orden de
   formación) — ¿hay una forma de aislar cuál de los dos efectos pesa más, si es que son separables?
3. **La grieta de Fase IV** (retroalimentación relación-sobre-relación) es preliminar (N=110, 5 semillas, con
   un control barajado que ya mostró efecto mixto) — ¿vale la pena escalarla antes de generalizar la lectura?
4. **Fase V (universalidad de S>0)** no se tocó — necesita que Alexis defina primero la forma funcional de "un
   solo principio relacional" antes de que se pueda codear sin fabricar el resultado. Queda para una sesión
   posterior.
5. Todos los números de este informe están verificados contra archivos en disco, pero varios experimentos
   (κ_V, C-N4, la grieta de Fase IV) tienen n chico o caveats metodológicos explícitos — ninguno debe leerse
   como cierre.

*Documentos fuente completos, con metodología detallada de cada experimento: `NULL0` a `NULL5_resultado_CS.md`,
`REAL_semillas_adicionales_CS.md`, `INFRA_masa_fija_generador_CS.md`, `TEST_layout_vs_identidad_grafo_CS.md`,
`ON77_*_CS.md` (5 documentos), `FASE3_renormalizacion_resultado_CS.md`, `FASE3_poda_dinamica_resultado_CS.md`,
`FASE4_orden_superior_resultado_CS.md`, `ADJUDICACION_CS076/077/078/079_*.md` — todos en este directorio.*
