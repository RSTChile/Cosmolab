# PROTOCOLO E5.6-5 · PRE-REGISTRO (antes del motor)

**Experimento:** E5.6-5 · "Exergía informacional y la I de S=I·E: ¿la relación es medible?"
**Tema:** 6 — Definición y verificación cruzada de la exergía
**Ejecuta:** agente E5.6-5 (batería Enfoque 5, 30 experimentos en paralelo)
**Fecha/hora de escritura de este pre-registro:** 2026-07-24 20:46 (hora local, -04)
**Código base reutilizado (NO editado):** `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py`
  (se importan sin modificar: `campo_inicial`, `paso_difusion`, `paso_expansion`,
  `medir_D`, `medir_pasos_lavado`)

**Estado de E5.6-1 al momento de escribir esto:** NO está en disco. Se buscó en todo el
repo (`E5_6_1`, `5.6-1`, `5_6_1`) y no existe ningún archivo. Por lo tanto **X_info se
define aquí de cero**, de forma explícita y verificable, sin depender de una definición
ajena inexistente. Si más tarde aparece E5.6-1 con otra definición de X_info, la
discrepancia entre ambas es información legítima (no invalida este experimento: la
pregunta de E5.6-5 es la relación X_info↔I bajo UNA definición explícita, pre-registrada
y fijada antes de correr — T3).

---

## 0. Objetivo

Medir la relación empírica entre dos cantidades:
- **X_info** — una exergía informacional (capacidad de estructura, medida por vía de
  entropía de Shannon sobre el campo).
- **I** — una medida INDEPENDIENTE de información/estructura (entropía estructural del
  campo), calculada por un método matemático DISTINTO al de X_info (anti-T2: la I no debe
  ser X_info renombrada ni una transformación monótona trivial de la misma fórmula).

**No se fuerza S=I·E.** Solo se mide y reporta qué relación empírica aparece entre X_info
e I a lo largo del barrido, contra NULL.

---

## 1. Definiciones EXACTAS (fijadas antes de correr, T3)

### Campo físico (reutilizado de cs074_rcruz.py, sin editar)
Anillo de N puntos φ. Condición inicial: fondo=1 + ε·(suma de 5 armónicos sinusoidales de
fase aleatoria, normalizada a std=1). Difusión solo por aristas activas (vectorizada,
idéntica a CS074). Expansión = corte Bernoulli de aristas activas con probabilidad H por
paso. H(r) = min(r·D, 1.0), con D medido del propio campo (fracción de contraste borrada
en un paso de difusión pura), igual que en cs074_rcruz.py. r = H/D es el eje sobredimensionado.

### X_info — exergía informacional (método A: histograma de medias de bloque, ESPACIAL)
1. Al estado final φ (tras `pasos` evolución) se lo divide en B=32 bloques contiguos de
   ancho w=N/B (N=256 → w=8), respetando la topología de anillo (adyacencia real).
2. Se calcula la media de cada bloque: m_1..m_B.
3. Se discretizan las B medias en nbins=8 bins de igual ancho, usando como rango
   [min(φ), max(φ)] DEL PROPIO φ CRUDO (todos los N puntos, NO las medias de bloque) de
   esa corrida — auto-normalización interna, misma regla para REAL y NULL de la misma
   corrida (φ_null es una permutación de φ_real → mismo multiset de valores → mismo
   rango/bordes de bin exactos en ambos casos; no es un blanco movido). Aclaración
   explícita porque es la parte más fácil de implementar mal: si el rango se tomara de
   las medias de bloque en vez del φ crudo, el binning se auto-reescalaría a como caigan
   esas medias y borraría la señal de contracción hacia la media global que produce el
   barajado — se detectó y corrigió este error ANTES de correr el barrido completo (ver
   nota en E5_6_5_motor.py).
4. Se computa la entropía de Shannon H_bloques del histograma resultante (base 2).
5. **X_info := H_bloques / log2(nbins)** ∈ [0,1].

Justificación: un campo homogéneo (equilibrio, φ≈cte) da medias de bloque casi idénticas
→ histograma concentrado en 1 bin → H_bloques≈0 → X_info≈0 (igual que la X termodinámica
en el límite de equilibrio). Un campo con estructura espacial real (variación suave de
larga longitud de onda que sobrevive) da medias de bloque dispersas a lo largo del rango
→ histograma más parejo → H_bloques alto → X_info alto. Al barajar (NULL), las medias de
bloque colapsan hacia la media global por el efecto de promediado (con w=8 elementos por
bloque, el barajado regresiona la dispersión inter-bloque) → se espera X_info_NULL <
X_info_REAL cuando hay estructura real que barajar.

### I — entropía estructural (método B: entropía de permutación ordinal, Bandt–Pompe — DISTINTO método)
1. Sobre el mismo φ final (misma corrida), se recorre el anillo con ventanas deslizantes
   de tamaño m=4 (patrón ordinal: para cada ventana de 4 puntos consecutivos, se registra
   el ORDEN relativo — cuál es el menor, segundo, tercero, cuarto — ignorando la magnitud
   absoluta y usando SOLO la posición relativa de los valores).
2. Empates (φ_i == φ_j exactos, relevante en ε=0) se resuelven por desempate estable por
   índice (orden lexicográfico de posición), consistente y determinista.
3. Hay 4!=24 patrones ordinales posibles. Se cuenta la frecuencia de cada patrón sobre las
   N ventanas (anillo cerrado → N ventanas).
4. Se computa la entropía de Shannon H_perm de esa distribución de 24 patrones (base 2).
5. **I := H_perm / log2(24)** ∈ [0,1].

Justificación: esta es la "entropía de complejidad ordinal" estándar (Bandt & Pompe 2002),
construida EXCLUSIVAMENTE a partir del orden local relativo de ventanas de 4 puntos
consecutivos — no usa bloques, no usa histograma de valores, no usa varianza ni
correlación (a diferencia de X_termo del script base Y de X_info arriba). Un campo con
variación suave (φ crece/decrece localmente de forma predecible) concentra la
probabilidad en pocos patrones (monótonos) → I bajo. Un campo sin ninguna correlación
espacial (barajado) visita los 24 patrones casi uniformemente → I≈1 (entropía máxima).

**Independencia metodológica (anti-T2):** X_info usa promedios de bloque + histograma de
VALORES; I usa el ORDEN RELATIVO local de ventanas de 4 puntos, sin promediar ni
histogramar valores. Son dos construcciones matemáticas distintas sobre el mismo φ. Se
espera, si acaso, una relación INVERSA entre ellas (X_info sube con estructura; I, al ser
literalmente una entropía, baja con estructura) — eso es una hipótesis a verificar, no un
resultado impuesto.

---

## 2. Barrido (sobredimensionado, regla del director)

- **N = 256** (fijo; el barrido de N es E5.6-3, fuera de alcance aquí).
- **ε** ∈ {0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1.0} — 9 valores, 6 décadas + los
  dos extremos (incluye ε=0 estricto como control).
- **r** (r_target, eje H(r)=min(r·D,1)) ∈ {0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30,
  100, 300, 1000} — 13 valores, 6 décadas, cruza r=1, más ancho que el barrido de
  cs074_rcruz.py (10 valores) — sobredimensionado explícito.
- **Semillas** ≥12 requeridas por spec → se usan **16** semillas por celda (margen).
- **pasos**: calibrados UNA vez (no a mano) igual que el modo "produccion" de
  cs074_rcruz.py — `medir_pasos_lavado(N, eps=1e-3, semillas=16)`, con margen ×1.15, y ese
  mismo `pasos_fijo` se aplica a toda la grilla (evita recalibrar 9×13 veces, que sería
  computo redundante; el propio script base usa este mismo patrón en su modo "produccion").
- **Total combinaciones:** 9 × 13 × 16 = 1872 corridas (REAL + NULL cada una).

## 3. NULL

Igual convención que cs074_rcruz.py: NULL = barajar (permutar) φ del estado FINAL de la
misma corrida (misma evolución física, mismo H, mismo ε, misma semilla — solo se destruye
el orden espacial antes de medir X_info e I). Esto es un NULL fuerte porque usa exactamente
la misma trayectoria dinámica; solo cambia si el observable puede "ver" la posición.

## 4. Observable y relación a medir

Por cada combinación (ε, r, semilla): X_info_real, I_real, X_info_null, I_null.

Relación empírica a reportar (T5: curva entera, no gate binario):
- Correlación de Pearson y de Spearman entre X_info e I, calculada:
  (a) agrupada (pooled) sobre TODAS las 1872 corridas REAL,
  (b) igual para NULL,
  (c) por celda (ε,r) agregando las 16 semillas, para ver si la relación es estable o
      depende del régimen.
- Dispersión entre semillas (std de X_info y de I dentro de cada celda) y dispersión de la
  correlación misma vía bootstrap sobre semillas.
- Curvas X_info(ε,r) e I(ε,r) completas (no solo el resumen de correlación).

## 5. PASS / criterios de lectura (fijados ANTES de correr, T3)

No hay un "PASS" binario de éxito/fracaso — el experimento es de caracterización. Se
reporta sin suavizar:
- Si |corr(X_info,I)|_REAL >> |corr(X_info,I)|_NULL (el NULL debe "morder": la relación
  real debe ser claramente distinguible del barajado) → relación real, se reporta signo y
  fuerza.
- Si REAL ≈ NULL en correlación → no hay relación medible por esta vía → se reporta como
  hallazgo negativo honesto (no se fuerza ni se re-diseña el observable después de ver el
  resultado).
- Se reporta explícitamente si la relación es monótona, si cambia de signo en algún
  régimen, y si depende de ε o de r por separado.
- **No se compara contra la forma S=I·E** — no se ajusta ninguna constante para que
  parezca multiplicativa. Solo se mide qué sale.

## 6. Trampas verificadas contra este diseño

- T0: nada discreto puesto a mano — B, nbins, m, N, ε-grid y r-grid están fijos aquí,
  antes de correr, y son producto del propio barrido sobredimensionado, no ajustados
  después de ver resultados.
- T1: ningún número apuntado al blanco (4.9%/31.5%) — no aplica a este tema, no se toca.
- T2: X_info e I están construidas por métodos matemáticos DISTINTOS (bloques+histograma
  de valores vs. patrones ordinales locales) — no son el mismo observable renombrado.
- T3: esta definición queda congelada en este archivo ANTES de escribir el motor.
- T4: el NULL (barajado) debe morder — se verifica explícitamente comparando REAL vs NULL
  para AMBOS observables, no solo para la correlación.
- T5: se entrega la curva completa (por ε, por r), no un solo número.
- T6: no aplica balance de energía aquí (eso es Tema 2); se nota como fuera de alcance.
- T7: barrido + semillas + la propia dinámica estocástica del corte de aristas
  (Bernoulli) ya es perturbación dinámica, no un solo punto.

## 7. Archivos que produce este experimento

- Este pre-registro: `E5_6_5_PROTOCOLO_PREREGISTRO.md`
- Motor: `E5_6_5_motor.py`
- Resultado crudo: `E5_6_5_resultado.json`
- (El reporte final se entrega como mensaje de texto al coordinador/CS, no como .md
  adicional, según instrucción de entorno.)

## 8. Nota de implementación (rendimiento, no cambia la definición)

El costo dominante es el loop de `pasos` (~1e4, medido por calibración) en Python puro.
Con 9×13×16=1872 corridas escalares el barrido tomaría ~3.3h. Para hacerlo viable sin
recortar la grilla sobredimensionada, se vectorizó la física (difusión + expansión +
X_info + I) con un eje extra de semillas (16 por celda ε,r), MISMA fórmula matemática que
la versión escalar. Antes de usarse en el barrido real se corrió un test de regresión
bit-a-bit: `paso_difusion_batch` == `paso_difusion` exacto, `x_info_bloques_batch` ==
`x_info_bloques` (<1e-12), `i_entropia_permutacion_batch` == `i_entropia_permutacion`
(<1e-12), incluyendo el caso degenerado ε=0 (empates). Con esto, el barrido completo baja
a ~22-23 min. Esta es una optimización de cómputo, no una redefinición de X_info o I.

**Firmado (pre-registro, antes de correr el motor):** agente E5.6-5, 2026-07-24 20:46 -04.
