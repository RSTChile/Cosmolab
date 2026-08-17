# Informe para el equipo — Fase VI completa (11 de agosto de 2026)

**Para:** Equipo Transinteligente · **Preparado por:** CC (Claude Code) · **Dirige:** Alexis López Tapia ·
**Continúa de:** `INFORME_EQUIPO_FASE5B_11ago2026_CS.md`

**Qué se hizo:** el equipo (GPT-5.6 Sol + 2 analistas) propuso ~18 experimentos para cerrar causalidad
sobre A2-B0-C2. Alexis pidió ejecutarlos **todos, automáticamente, sin analizar hasta terminar**. Se
ejecutaron en 4 olas con agentes dedicados: **17 experimentos, los 17 con informe completo.**
Todos los números de abajo están verificados contra archivos en disco.

> **⚠️ ACTUALIZACIÓN (tras completarse O3-A, la "obligatoria"):** el resultado de O3-A refuerza y precisa la
> tensión de la Parte 3 — **al descontar la densidad, la ventaja de clase es indistinguible de cero en
> AMBAS resoluciones (p=0.22 y p=0.41)**. Ver Parte 3.bis, que es ahora la conclusión operativa del informe.

---

## TL;DR — árbol de decisión

```
¿El resultado de Fase V-B (Clase III acreta más masa) sobrevive a los controles duros?
│
├─ ¿Es artefacto del emparejamiento? ──────────── NO. Sólo 0.05% de 10.000 re-emparejamientos
│                                                  al azar son tan extremos. (O2-F/reanálisis)
├─ ¿Es pseudorreplicación (pocas semillas base)? ─ NO se cae. N_eff real 22.6 (37 pares) / 28.4 (40);
│                                                  Wilcoxon deflactado p=5.11×10⁻⁴ (37) / 1.84×10⁻⁴ (40)
│                                                  — valores canónicos en
│                                                  VALORES_CANONICOS_FASE5B_ajustados_CS.md.
│                                                  Pero κ_V SÍ sale de significancia
│                                                  (0.017→0.061). (O2-F)
├─ ¿Es artefacto del código de Phantom? ────────── NO. Un integrador independiente, con otra
│                                                  física, coincide 5/5 en señal fuerte. (O4-A)
├─ ¿Depende de propiedades triviales (grado)? ──── NO del todo: con grados IDÉNTICOS y 99.7% de
│                                                  aristas cambiadas, el original gana igual
│                                                  (+5%, p=0.010). (O3-B)
│
└─ ENTONCES ¿está confirmado? ─────────────────── NO TAN RÁPIDO. Tres resultados independientes
                                                   dicen que la DENSIDAD explica la mayor parte:
                                                   · grafos al azar con las mismas aristas caen
                                                     sobre la misma recta (O3-D)
                                                   · masa vs nº aristas: ρ = −0.97 (O3-E)
                                                   · la geometría INICIAL, sin integrar nada,
                                                     ya predice con r = 0.98 (O4-A)

¿La cadena mecanismo → geometría → gravedad se sostiene? ── SÍ, con 75% de mediación (O3-C).
¿A2-B0-C2 es exaptativo (antecedente de O-N7.7)? ────────── NO, por dos vías distintas (O3-F, O3-E).
```

---

## PARTE 1 — Lo que cambió sobre el MÉTODO (y afecta a todo lo anterior)

### 1.1 Un bug de medición real, encontrado y acotado ⚠️→🔧

`_diam` (congelado desde cs055) arrancaba su recorrido en **el primer nodo por índice con aristas** — que en
grafos fragmentados puede caer en un par suelto de 2 nodos, midiendo el diámetro de *ese fragmento* (=1) en
vez del de la componente gigante (22-25). *Analogía: medir la altura de un edificio apoyando el metro en el
buzón de la vereda.*

- **Alcance medido:** 15 de 430 reglas (3.5%) — exactamente las de pendiente negativa; **0 de las 415** con
  pendiente ≥0. La categoría "intermedio (sin clase clara)" era **entera** un artefacto: desaparece.
- **Corregido** en módulo nuevo (`cs090_diam_corregido.py`); `cs055` queda intacto para reproducir historia.
- **Fase V-B sobrevive:** 29/37 pares válidos, p=0.00075 (publicado: 31/40, p=0.00068).
- **Auditoría de las fases restantes: cero descarrilamientos.** Fase IV **ni siquiera usa diámetro** (su
  observable es la holonomía, verificado por AST); de 15 scripts CS07x-CS08x sólo 2 lo usan, ambos limpios
  (0/54 y 0/126). (O1-B)
- **Fase V-A re-corrida de forma reproducible:** **0 de 150** reglas afectadas — la inferencia indirecta pasa
  a medición directa. El 100% de la diferencia contra lo publicado es muestreo (p=0.93). (O2-E)

### 1.2 El patrón que se repitió CUATRO veces: los "saltos" son umbrales, no física

| Dónde apareció el "salto" | Qué era en realidad |
|---|---|
| Bimodalidad Clase I/III | Umbral 0.7 cortando una pendiente continua (Fase VI, reanálisis) |
| `kcap` 5→6: 71.7%→4.9% Clase III | Rampa suave: pendiente = 2.457 − 1.036·log(kcap), R²=0.729 (O2-C) |
| "Clase II" en A0 | Umbral de similitud + redondeo entero del diámetro; **reproducibilidad 45.6%** (O1-C) |
| Clase III en el barrido de kcap | El NULL de cada regla cruza el MISMO umbral 0.7 (O3-D) |

**Cuantificado:** la recta continua sobre la pendiente explica **R²=0.663** de la masa; el escalón oficial,
**0.182**. Y **el 37% de las reglas están tan cerca de un umbral que cambiarían de clase con sólo
re-medir** (O1-A, O2-E).

> **Recomendación operativa para el equipo:** abandonar "% Clase III" como endpoint. Ya no es una preferencia
> estilística — hay cuatro mediciones independientes que muestran que fabrica discontinuidades inexistentes.

---

## PARTE 2 — Lo que SOBREVIVIÓ a los controles duros ✅

| Control | Resultado |
|---|---|
| **Parejas al azar** (10.000 permutaciones) | Sólo **0.05%** tan extremas como el real. El efecto depende de la clase, no del pool |
| **N efectivo / pseudorreplicación** | N_eff 19-28. p ajustado ~10⁻⁴ (baja 1-2 órdenes vs publicado). **Clave: `kcap` está balanceado dentro del par por diseño, así que la fuente dominante de agrupamiento NO contamina el contraste pareado** — medido, no asumido (O2-F) |
| **Rewiring con grados idénticos** | Grados iguales nodo por nodo, 99.7% de aristas nuevas, clustering ÷10: **el original gana igual** (9/12 masa p=0.010; 11/12 κ_V) (O3-B) |
| **Solver independiente** | Integrador propio, otra física, validado (energía a 10⁻¹⁵): **5/5 en señal fuerte**, 9/10 con vara común (O4-A) |
| **20 genealogías independientes** | Efecto **repartido**: 0/20 estériles, 17/20 ≥30%. No lo sostienen 2-3 familias fértiles (O2-B) |
| **Cadena de mediación** | Condición 1 (rígido+soporte) domina también en gravedad; **75% del efecto está mediado por la geometría**; partir por pendiente separa la masa (p=2.4e-05), partir por condición no (p=0.42) (O3-C) |

**Y dos misterios viejos, resueltos:**
- **El trío-equivocado** (Fase IV): se sostiene a n=30. **No es cobertura** — refutado por demostración (el
  *derangement* es una biyección, el reparto es idéntico, Gini 0.5445 en ambos). **Es la desalineación**
  entre de dónde sale el defecto y adónde va el empujón: con alineación el trío real ayuda, sin alineación
  **estorba** (interacción z=−6.36). Reparto: 57% desalineación, 47% coherencia, ~0% cobertura (O1-D).
- **El orden de formación** (q_T): con instrumento posicional deja de ser 0% exacto. Cambia **dónde quedó
  cada partícula** (p=10⁻¹⁸) pero **no la forma agregada** de la nube (<0.7%, indistinguible del ruido).
  **Huella de identidad, no de forma** (O2-A).

---

## PARTE 3 — La tensión central, planteada de frente

Tres resultados independientes dicen que **la densidad de aristas explica la mayor parte** de lo que
atribuíamos a organización relacional:

1. **O3-D:** la masa sigue a `kcap` monótonamente y sin solape (η²=0.949, más del doble entre extremos) —
   **pero 6 controles Erdős-Rényi *sin ninguna estructura*, emparejados en aristas, caen SOBRE la misma
   recta.** `kcap=4` no supera a su espejo al azar (0.1488 vs 0.1502). Y el NULL de cada regla tiene la misma
   pendiente que el REAL, con ningún z_agg llegando a 1.
2. **O3-E:** masa vs nº de aristas da **ρ = −0.971 (R²=0.915)**. La "memoria histórica" gana masa, pero
   **menos de lo que su propia poda extra ya explicaba** (residuo negativo, p=0.048).
3. **O4-A:** el observable medido sobre las **condiciones iniciales, sin integrar nada**, predice el
   resultado de Phantom con **r = +0.98**; controlando por eso, el motor independiente no aporta nada
   (p=0.80).

**Contra eso, un resultado en la dirección opuesta:** **O3-B** compara contra un gemelo de **grados idénticos**
(misma densidad exacta) y el original **sí** gana (+5%, p=0.010). Y lo que predice cuál gana **no es la
pendiente** (ρ=0.04, p=0.90) **sino cuánto clustering tenía el original para perder** (ρ=0.77, p=0.003).

### Lectura conjunta más defendible hoy (no un veredicto)

```
densidad de aristas ──────────────► efecto GRANDE (factor 2 entre kcap extremos)
   │
   └─ organización relacional ────► residual CHICO pero real (~5%), sólo visible con
      (clustering, no pendiente)     diseño pareado a grados idénticos
```

La cadena **mecanismo → geometría → gravedad** se sostiene (O3-C, 75% mediado), **pero "geometría" resulta
estar dominada por "cuántas aristas hay"**, y la pendiente —nuestro resumen oficial— es un intermediario
pobre: sobrevive al barajado y no predice qué se pierde al barajar.

*En simple:* creíamos comparar pueblos con distinto urbanismo; lo que más cambiaba era **cuántas calles hay
por casa**. Un pueblo trazado al azar con las mismas calles junta casi la misma arena. Pero con exactamente
las mismas calles y casas, reordenadas, el pueblo original todavía junta un 5% más — y eso depende de cuántas
esquinas cerradas tenía para perder.

---

## PARTE 3.bis — O3-A cierra la tensión: descontando densidad, el efecto de clase ≈ 0

La tarea que el equipo marcó como **obligatoria** se completó (13 pares a N=2000 y 13 a N=4000; N=8000
resultó impracticable: ~73 min por grafo, bloqueado por el O(N²) del layout, no por RAM).

**Respuesta literal a la pregunta del equipo:** el efecto **no** se desvanece al subir resolución. Δmasa
pasa de +0.0095 a +0.0175, **magnitud absoluta ×2.18** (Wilcoxon sobre |Δ|, p=0.022).

**Pero con tres matices que lo vacían de contenido como "efecto de clase":**

1. **El signo par-a-par sólo se conserva en 9/13**; el Spearman del ordenamiento es +0.38; el Wilcoxon sobre
   el Δ *con signo* da **p=0.376**. Los 4 que se dan vuelta son de la zona de empate (los 4 de |Δ|≥0.012
   conservan 4/4). **κ_V se invierte entero:** +0.063 (9/13) → −0.062 (2/13).
2. **La Clase III teje sistemáticamente más RALO** (grado medio 3.30 vs 3.53, estable entre resoluciones).
   **Al descontar esa densidad, la ventaja de clase queda en +0.004, con p=0.22 (N=2000) y p=0.41
   (N=4000) — indistinguible de cero en las dos.**
3. **Lo que escala no es la clase: es la pendiente del efecto de DENSIDAD** (−0.044 → −0.121, ×2.7). Y la
   dominancia de la densidad **se refuerza** con la resolución (r −0.63 → −0.79).

**Converge con O3-D por un diseño completamente independiente.** Y precisa a O4-A: la ventaja geométrica
*inicial* de la Clase III **baja** (+0.064→+0.044) mientras su poder predictivo **sube** (r +0.52→+0.79) —
**no escala la geometría inicial, escala la conversión de geometría en masa.**

### La síntesis que sostienen los 17 experimentos juntos

```
"Clase III"  ══es en gran medida══►  "teje más ralo"  ═══►  más masa acretada
                                      (ρ = −0.97)

Descontando densidad: efecto de clase ≈ 0  (p=0.22 / 0.41, dos resoluciones)
                                    ↓
        PERO con grados EXACTAMENTE idénticos (O3-B), el original todavía gana +5% (p=0.010),
        y lo que predice cuál gana es el CLUSTERING (ρ=0.77), no la pendiente (ρ=0.04).
```

**Lectura:** la etiqueta "Clase III" estaba capturando, sobre todo, densidad de aristas. Debajo de eso hay un
efecto relacional genuino —el residual de O3-B— pero es **un orden de magnitud menor** que el contraste
crudo de clase, y **no lo captura la pendiente**, que es el observable con el que veníamos trabajando.

---

## PARTE 4 — Lo que se CAYÓ ❌

- **κ_V como métrica puente:** descartado. Descontando `kcap`, su relación con la geometría se evapora
  (ρ parcial +0.057, p=0.63); es casi una copia de la masa (ρ=0.902); y **no puede ahorrar cómputo por
  construcción: se mide dentro de Phantom** (O1-A). Además sale de significancia al ajustar por agrupamiento.
- **A2-B0-C2 como nodo exaptativo (antecedente de O-N7.7):** negativo por **dos vías independientes**.
  - $B_\tau$: parecía funcionar, pero **el 97.6-99.9% venía del denominador** — congelando el numerador en
    una constante, el patrón se reproduce idéntico. La entropía sola no alcanza significancia a favor de III
    en **ninguna** de 72 celdas; las 11 significativas favorecen a I. El gas de III queda **más caliente y
    más agrupado, no más variado** (O3-F).
  - Memoria: el grafo con memoria es **subconjunto estricto** del sin memoria (15/15, cero aristas
    exclusivas) — no reordena nada, sólo corta ~80 aristas más; y su residuo descontando densidad es
    **negativo** (O3-E).
  - **Encuadre exacto:** esto **no refuta O-N7.7** — dice que C2 no es el nodo exaptativo, y sólo para la
    operacionalización *estática*. La versión dinámica que pedía la especificación no se hizo.
- **"kcap=6 es un número especial":** no. Es una rampa continua; el residuo de kcap=6 es el más chico de los
  cinco topes. Y **la variable de control real no es `kcap` sino el grado medio efectivamente alcanzado** —
  las 570 corridas de 3 diseños caen sobre una sola curva (R²=0.834) (O2-C).
- **"A0 llega a Clase II":** la etiqueta tiene **45.6% de reproducibilidad** al re-medir el mismo campo, con
  un confound de umbral que usan los dos métodos. Bonus: **dentro de A0 el Eje C no hace nada** — C0/C1/C2 son
  bit a bit idénticos (O1-C, confirmado en O2-E).

---

## PARTE 5 — Lo que queda abierto

**Un límite duro de infraestructura, medido:** **N=8000 es impracticable con el protocolo actual** —
`layout_resortes` escala con exponente 2.74 entre N=2000 y 4000 (656 s/grafo), lo que da **~73 min por
grafo**. Se verificó que la máquina tiene 64 GB (no 16, como se había supuesto): **el bloqueante es el O(N²)
del layout, no la RAM.** Cambiarlo obligaría a revalidar toda la serie.

**El experimento que más falta, y que esta tanda hace obvio:**
> **Barrido de número de aristas a `kcap` fijo** (o densidad igualada con distinto kcap). Es lo único que
> rompe la colinealidad `kcap`↔densidad (r=+0.984, VIF hasta 47.8) y separa "geometría extendida" de "red con
> menos aristas". Con O3-A, O3-D y O3-E convergiendo por tres diseños independientes, ya no es "el
> experimento que falta": es **el único que puede rescatar un efecto de clase independiente de la densidad,
> o confirmar que no lo hay.**

**Otros pendientes concretos:**
- Control de O3-B con grados **y triángulos** fijos (aislaría si el clustering es mecanismo o correlato).
- Control de O3-E: podar las mismas ~80 aristas **al azar** (separa "elegir bien" de "cortar cualquier cosa").
- $B_\tau$ **dinámico** (ensamble de continuaciones perturbadas) — la única versión que distinguiría proxy de
  fenómeno.
- Permutar los cupos de HET-grado conservando la distribución (aislaría si importa la *alineación* del cupo
  con la estructura preexistente).
- La bimodalidad de REAL entre semillas en Fase IV: 3 de 30 fallan en aplanar, sin explicación.

**Dos avisos de higiene de datos:**
- **Unir CSVs por `(rule_id, seed)`, nunca por `rule_id` solo** — los lotes v1/v2 comparten el patrón de
  nombre y hay reglas distintas con el mismo id (O3-B). O1-A y O2-F ya unían por `seed`; conviene revisar
  cualquier otro análisis.
- `sarracen` **sí** está instalada (1.3.1, en `venv/`). Regla: Phantom → `./venv/bin/python`; grafos →
  `python3.9`.

---

## Cierre

**No se declaró cierre ni veredicto en ningún experimento de esta tanda** — 17 informes individuales, todos
con sus números crudos y CSVs en este directorio para auditoría directa.

Lo más honesto que se puede decir hoy: **hay un efecto relacional genuino —sobrevivió al control más duro,
el gemelo de grados exactamente idénticos (+5%, p=0.010)— pero es un orden de magnitud menor que el
contraste de clase que veníamos reportando.** La mayor parte de ese contraste era **densidad de aristas**:
al descontarla, la ventaja de clase es indistinguible de cero en las dos resoluciones probadas (p=0.22 y
p=0.41), y tres diseños independientes convergen en lo mismo.

La cadena causal se sostiene, pero hay que reescribirla con precisión:

> **la regla relacional determina cuán ralo se teje el grafo → la densidad determina la geometría del
> layout → la geometría determina la masa acretada.** Encima de esa cadena hay un residual estructural real,
> ligado al *clustering* y no a la pendiente, del orden del 5%.

Eso no es el resultado que la teoría predecía, pero es un resultado — y es el que los datos sostienen.

*Fuentes: `FASE6_O1A/O1B/O1C/O1D/O2A/O2B/O2C/O2D/O2E/O2F/O3A/O3B/O3C/O3D/O3E/O3F/O4A_*.md`,
`FASE6_adopcion_diam_corregido_CS.md`, `FASE6_outliers_pendiente_negativa_CS.md`,
`FASE6_reanalisis_azar_continuo_CS.md`, `FASE6_PLAN_EJECUCION_COMPLETA_CS.md` — todos en este directorio.*
