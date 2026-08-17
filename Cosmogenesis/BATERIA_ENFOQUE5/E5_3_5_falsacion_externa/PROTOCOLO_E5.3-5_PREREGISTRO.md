# PROTOCOLO E5.3-5 — PRE-REGISTRO (firmado antes de correr el motor)

**Experimento:** E5.3-5 · "Test de falsación externo: distancia emergente al 4.9%/31.5%, sin ajuste"
**Tema:** 3 — Eficiencia de conversión emergente ★ (ancla contra 4.9%/31.5%, junto con E5.3-1)
**Agente:** E5.3-5 (batería Enfoque 5, 30 experimentos en paralelo)
**Timestamp de pre-registro (UTC):** 2026-07-25T00:45:02Z (America/Santiago, UTC-4: 2026-07-24 20:45:02)
**Documento autoritativo:** `BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md` (sección 0 +
REGLAS DE EJECUCIÓN + intro TEMA 3 + spec E5.3-5)
**Ancla externa (test de salida, nunca entrada):** `CONSOLIDADO_presupuesto_energia_multi_fuente.md`
— materia ordinaria 4.9%, materia total 31.5% (Planck 2018).

---

## 0. Verificación de reutilización (obligatoria, hecha DOS VECES)

Instrucción recibida: reutilizar el grid completo y la definición de eficiencia de E5.3-1 y
E5.3-2; si sus resultados JSON no están en disco cuando yo corro, correr mi propia versión del
mismo grid con la misma definición.

**Chequeo #1 — 2026-07-24 16:36 (America/Santiago):**
- `E5_3_1_eficiencia_12decadas/` — carpeta VACÍA.
- `E5_3_2_eficiencia_vs_ligadura/` — carpeta NO EXISTÍA aún.

**Chequeo #2 (tras retomar sesión) — 2026-07-24 20:41–20:45 (America/Santiago):**
- `E5_3_1_eficiencia_12decadas/` — contiene **solo** `PROTOCOLO_E5.3-1_PREREGISTRO.md`
  (pre-registro completo, timestamp 2026-07-25T00:41:03Z). **Sin motor, sin JSON de
  resultados.** El protocolo E5.3-1 se declara a sí mismo "definición CANÓNICA de E_ligada
  para todo el TEMA 3" e instruye a E5.3-2/3/4/5 a reutilizarla.
- `E5_3_2_eficiencia_vs_ligadura/` — contiene `PROTOCOLO_E5.3-2_PREREGISTRO.md` (completo) y
  `E5_3_2_motor.py` (completo, funcional). **Sin `E5_3_2_resultado_crudo.json` todavía** — en
  el momento de este chequeo hay un proceso de humo corriendo (`ps aux` confirma
  `E5_3_2_motor.py` con subconjunto SEMILLAS=3 de validación, PID activo). El motor E5.3-2 usa
  su PROPIA definición (ANOVA entre-segmentos), declarada explícitamente en su protocolo como
  "no heredada" porque cuando E5.3-2 se pre-registró (16:38) la carpeta E5.3-1 seguía vacía
  (verificado por ellos dos veces: 16:37 y 16:40).

**Decisión (declarada aquí, antes de ver un solo resultado):** dado que ninguno de los dos
JSON de resultados existe en disco al momento de escribir este pre-registro, y el documento
madre exige explícitamente "si no están disponibles cuando tú corras, corre tu propia versión
del mismo grid con la misma definición", procedo así:

1. **Fuente E5.3-1 (definición canónica de dominios/E_ligada):** su protocolo SÍ está completo
   y congelado en disco, pero su motor no. Implemento yo mismo, en mi propio archivo
   (`E5_3_5_motor.py`, prefijo propio, sin tocar la carpeta de E5.3-1), el barrido y la
   definición EXACTA descrita en `E5_3_1_eficiencia_12decadas/PROTOCOLO_E5.3-1_PREREGISTRO.md`
   §2 (transcrita en la sección 1 de este documento). Esto satisface "misma definición,
   corrida por mí" porque la definición viene de su texto congelado, no de una improvisación
   mía.
2. **Fuente E5.3-2 (definición ANOVA entre-segmentos, propia de ese agente):** su motor SÍ
   está completo, congelado y en disco (`E5_3_2_motor.py`), sin editar por mí. Lo importo como
   módulo (igual que ellos importan `cs074_rcruz.py` sin editarlo) y ejecuto su función
   `barrido()` con su grid de producción completo, tal como está escrito — **sin recomputar
   desde cero una reconstrucción propia**, reutilizando literalmente su código y su
   definición. Si `E5_3_2_resultado_crudo.json` aparece en disco ANTES de que yo dispare esa
   corrida, lo cargo directamente en su lugar (chequeo automático en el motor, ver §3).
3. Las dos fuentes usan definiciones de "eficiencia" DISTINTAS (dominios aislados vs. ANOVA
   entre-segmentos) — esto es correcto y deseado: son dos operacionalizaciones independientes
   del mismo concepto (E_ligada = estructura que la expansión congeló), lo cual funciona como
   la "segunda verificación por método independiente" que exige la regla de ejecución #4. Se
   reportan SEPARADAS por método, nunca mezcladas en un solo número, y también agregadas para
   la distribución conjunta de distancias que pide la spec de E5.3-5.

Si en algún momento posterior a esta firma aparecen `E5_3_1_resultado_crudo.json` o un motor
E5.3-1 completo, o un `E5_3_2_resultado_crudo.json` más reciente, y da tiempo, se documentará
la comparación en el reporte final — pero no se re-abre este pre-registro (T3).

---

## 1. Definición EXACTA reutilizada de E5.3-1 (transcrita, no reinterpretada)

- Anillo de N=200 sitios. Campo inicial φ_i(0) = 1 + ε·pert_i (pert = 5 modos seno,
  fase aleatoria, centrada, normalizada a std=1) — funciones `campo_inicial`, `paso_difusion`,
  `paso_expansion`, `medir_D` importadas SIN EDITAR desde `cs074_rcruz.py` (mismo patrón que
  usó E5.3-2; el protocolo E5.3-1 dice "reescrito", yo importo — diferencia de implementación
  declarada, sin efecto en el resultado numérico porque son las mismas fórmulas).
- ε barrido en 12 décadas: `np.logspace(-12, 0, 13)` (13 puntos) + punto de control ε=0
  (excluido del barrido principal, reportado aparte, eficiencia indefinida 0/0 esperada).
- r barrido en 6 décadas: `np.logspace(-3, 3, 13)` (13 puntos).
- H = min(r·D_ε, 1); D_ε medida (fracción de contraste borrada en un paso de difusión pura),
  promedio sobre semillas, igual método que `cs074_rcruz.py::medir_D`. D_ε=0 por convención en
  ε=0.
- Ruido dinámico (T7): tras difundir, ruido gaussiano aditivo de amplitud `0.01·sqrt(D_ε)` por
  sitio, cada paso — atado a D_ε medida, no a un número objetivo.
- pasos = `clip(ceil(5.0 / max(H, H_floor)), 100, 3000)`. **H_floor no estaba numéricamente
  fijado en el texto de E5.3-1** (solo mencionado como símbolo) — declaro aquí, ANTES de
  correr, `H_floor = 1e-6` (evita división por cero cuando H=0 exacto; en la práctica el tope
  superior de 3000 domina para H pequeño, así que este relleno no mueve el resultado hacia
  ningún blanco — es una guarda numérica, no un ajuste).
- Semillas: 20 (≥16 exigidas).
- E_total(ε,semilla) := Σ_i (φ_i(0) − 1)² (presupuesto declarado, axioma E1, fijo por corrida).
- Al final de `pasos` pasos: componentes conexas del grafo de aristas activas remanentes
  (BFS sobre el anillo). Dominio D_k es "ligado" ssi 1 ≤ |D_k| < N (quedó aislado del resto).
- E_ligada := Σ_{i ∈ dominios aislados} (φ_i(final) − 1)².
- **eficiencia := E_ligada / E_total** ∈ [0,1]. Si E_total=0 (solo ε=0): "indefinida", excluida
  de la distribución principal.
- NULL: se permutan los VALORES finales de φ (misma rng, tirada adicional) manteniendo la
  MISMA partición en dominios (no se re-simula la expansión) — aísla si los dominios
  concentran más estructura real que una asignación aleatoria de esos valores a esos dominios.

## 2. Definición EXACTA reutilizada de E5.3-2 (ejecutando su motor sin editar)

- Mismo campo base (N=200, `campo_inicial`, `paso_difusion`, `paso_expansion` de
  `cs074_rcruz.py`, sin editar).
- Intensidad de ligadura L en `np.logspace(-3, 2, 10)` (5 décadas, 10 puntos); modula
  `H_eff(L) = H0/(1+L)`, con `H0 = D` (ancla r=1, medida).
- ε ∈ {0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.3, 1.0} (9 puntos, idéntico a E5.1-1).
- Ruido dinámico NOISE_REL=0.02·ε cada paso (T7).
- pasos calibrados una vez vía `medir_pasos_lavado(N=200, eps=1e-3, semillas=12)` (lavado
  P<0.05, margen ×1.15) — igual método que el modo "producción" de `cs074_rcruz.py`.
- Semillas: 12 (≥12 exigidas por spec de E5.3-2; ligeramente menor al piso de 16 que pide
  E5.3-1/E5.3-5 en general — se declara esta asimetría, no se homogeniza post-hoc para no
  editar el motor ajeno).
- Segmentos = tramos contiguos vivos tras cortes de expansión (no requiere aislamiento total
  bilateral como E5.3-1 — basta una arista de frontera cortada en CUALQUIER punto del anillo
  para que existan ≥2 segmentos).
- E_total := Σ(φ₀ − mean(φ₀))² (fijo, medido al inicio de esa corrida).
- E_ligada := Σ_k n_k·(μ_k − mean_global(φ))² (varianza ENTRE segmentos, identidad ANOVA
  auditada cada corrida: E_ligada + E_dentro = E_final, tolerancia 1e-6).
- **eficiencia := E_ligada / E_total**.
- NULL: se permutan los VALORES de φ_final manteniendo la MISMA partición en segmentos.

**Corrección hecha ANTES de escribir el motor (T3 — se declara, no se esconde):** en un
anillo, "componentes conexas del grafo de aristas activas" (lenguaje de E5.3-1) y "tramos
contiguos separados por aristas cortadas" (lenguaje de E5.3-2, función
`segmentos_desde_activo`) son EXACTAMENTE el mismo objeto topológico — con 0 o 1 arista
cortada el anillo sigue siendo un único componente de tamaño N (no cuenta como ligado en
ninguna de las dos definiciones); hacen falta ≥2 cortes para que existan ≥2 piezas aisladas.
Por eso este motor reutiliza literalmente `segmentos_desde_activo` de `E5_3_2_motor.py` (sin
editar) también para la parte E5.3-1, evitando reimplementar la misma lógica dos veces con
riesgo de discrepancia. La diferencia REAL entre las dos definiciones no está en qué cuenta
como "dominio/segmento" sino en CÓMO se agrega la energía dentro de esos dominios: E5.3-1 suma
la energía estructural TOTAL (Σ(φ−1)², dentro + entre) de los dominios aislados; E5.3-2 suma
SOLO la varianza ENTRE segmentos (ANOVA), dejando la varianza dentro de cada segmento fuera de
"ligada". Esta diferencia de agregación ES el dato relevante para la Tercera Lectura de §4, no
la identificación de dominios (que es idéntica).

## 3. El agregado propio de E5.3-5 (lo que este experimento AÑADE, no reemplaza)

Para CADA fila de CADA uno de los dos grids anteriores (celda = combinación de parámetros,
agregada sobre semillas: media y std de la eficiencia real y la NULL), se calcula:

    dist_49  := |eficiencia_real_media − 0.049|
    dist_315 := |eficiencia_real_media − 0.315|

Se reporta la distribución COMPLETA de `dist_49` y `dist_315` sobre las 169 celdas de E5.3-1
(13×13, excluyendo el control ε=0) MÁS las 90 celdas de E5.3-2 (9×10) — 259 celdas en total,
cada una con su media±std sobre semillas. Se reportan también las mismas distancias calculadas
fila-por-fila (no por celda-agregada) sobre las 3380+1080=4460 corridas individuales
(semilla-a-semilla), para exponer la dispersión entre semillas sin promediar prematuramente
(regla de ejecución #9: "curvas completas + dispersión entre semillas").

**Chequeo automático de resultados en disco (implementado en el motor, no a mano):** antes de
ejecutar la parte E5.3-2, el motor verifica si
`E5_3_2_eficiencia_vs_ligadura/E5_3_2_resultado_crudo.json` existe; si existe, lo carga y usa
esas filas directamente (sin recomputar); si no existe, importa `E5_3_2_motor.py` sin editarlo
y ejecuta `barrido()` él mismo. Se declara en el JSON de salida cuál de las dos rutas se usó.

## 4. Lectura pre-registrada — LAS TRES POSIBLES (congeladas ANTES de ver resultados, T3)

Definiciones operacionales de "cerca" fijadas AQUÍ, no ajustadas después:

- **(a) Hallazgo fuerte:** existe una región no trivial del grid (no un único punto aislado
  por azar de 4460 corridas — se exige que sobreviva agregación por semillas, es decir a nivel
  de CELDA, no de corrida individual) donde `dist_49 < 0.02` o `dist_315 < 0.02` (2 puntos
  porcentuales, mismo umbral de reporte que usa E5.3-1 en su propio protocolo, heredado aquí
  por consistencia entre agentes del mismo Tema) Y esa celda no es indistinguible del NULL
  (z de la celda, |z|>2, reutilizando el z-score que ya calculan ambos motores fuente).
- **(b) Dato honesto (cae en otro valor estable):** la eficiencia converge a un valor o meseta
  reproducible entre semillas (std entre semillas pequeño relativo a la media, y estable en un
  rango de r/L/ε — "meseta" definida como ≥3 celdas contiguas del grid con eficiencia dentro de
  ±0.02 entre sí) que NO cae cerca de 4.9%/31.5% (dist_49≥0.02 y dist_315≥0.02 en esa meseta).
- **(c) Negativo (no converge a nada estable):** la eficiencia no se estabiliza — alta
  dispersión entre semillas en toda o casi toda la celda (std entre semillas comparable o
  mayor a la media), o no hay meseta identificable, o el NULL no se distingue del REAL en
  ninguna región apreciable (T4: si el NULL nunca muerde, el "hallazgo" es artefacto).

**Este agente NO adjudica cuál de las tres ganó en prosa — reporta los números exactos
(distancias mínimas, en qué celda ocurren, con qué z, con qué dispersión entre semillas) y dice
técnicamente cuál lectura describen esos números, dejando el cierre interpretativo final a CS**
(regla de ejecución #9 y nota explícita del encargo).

## 5. NULL, T0–T7 (checklist explícito)

- **NULL:** heredado íntegro de cada fuente (permutación de valores finales manteniendo
  partición/topología real) — nunca "barajado global" que destruya también la partición, para
  que el NULL sea una prueba real de si la estructura importa, no solo de si hay varianza.
- **T0/T1:** ningún número puesto a mano salvo `H_floor=1e-6` (guarda numérica declarada en
  §1, sin relación con 4.9%/31.5%) y los umbrales de lectura de §4 (0.02, fijados ANTES de
  correr, iguales para ambos blancos, no elegidos por blanco).
- **T2:** el observable (eficiencia) es una fórmula fija por fuente; el juez (tres lecturas de
  §4) es un criterio separado, congelado antes de correr.
- **T3:** este archivo se guarda ANTES de escribir `E5_3_5_motor.py`.
- **T4:** el NULL de cada fuente se reporta siempre junto al REAL, con z-score.
- **T5:** se entrega la curva/distribución completa (259 celdas, 4460 corridas), nunca un gate
  binario.
- **T6:** se audita, para la parte E5.3-1, que E_ligada ≤ E_total en cada corrida (si no,
  error y se para); para la parte E5.3-2 (motor ajeno), se confía en su propia auditoría ANOVA
  interna (`anova_ok`, ya implementada en su motor) y se reporta agregada.
- **T7:** ruido dinámico presente en ambas fuentes (ya descrito en §1 y §2), además de
  semillas.

## 6. Prohibiciones explícitas (regla de oro de esta batería)

- Prohibido mover H_floor, el umbral 0.02, N, semillas, o cualquier coeficiente de §1/§2 hacia
  4.9% o 31.5% después de ver un resultado parcial.
- Prohibido re-ejecutar con otro seed/rango tras ver que "casi" cae cerca, sin declarar
  explícitamente que se trata de una corrida adicional post-hoc (no se hará ninguna).
- Si se detecta un error ajeno (en `cs074_rcruz.py`, en `E5_3_2_motor.py`, o en el protocolo
  de E5.3-1) se PARA y se reporta a CS con línea exacta — no se corrige en silencio, no se
  edita el archivo ajeno.

## 7. Archivos que produce este experimento

- `PROTOCOLO_E5.3-5_PREREGISTRO.md` (este archivo).
- `E5_3_5_motor.py` (motor propio; importa sin editar `cs074_rcruz.py` y, si hace falta,
  `E5_3_2_motor.py`).
- `E5_3_5_resultado_e531def.json` (barrido propio con la definición de dominios de E5.3-1).
- `E5_3_5_resultado_e532.json` (filas de E5.3-2, cargadas de disco o recomputadas — se declara
  cuál).
- `E5_3_5_agregado_distancias.json` (las 259 celdas + 4460 corridas con dist_49/dist_315,
  estadística descriptiva completa de ambas distribuciones).
- Reporte final verbatim entregado en la respuesta del agente (crudo, sin adjudicar veredicto).

---

## ADENDO (post-humo, ANTES de la corrida de producción) — 2026-07-25T01:00:48Z

El "modo smoke" (subconjunto pequeño: 4 ε, 3 r, 3 semillas) de la parte A detectó una
**violación del guardián T6** (E_ligada ≤ E_total) para ε pequeño, con `eficiencia` llegando a
valores de 10⁶–10⁸ (ver `E5_3_5_resultado_e531def_smoke.json`). Diagnóstico exacto, hecho
ANTES de tocar ningún coeficiente: el ruido dinámico de la definición canónica de E5.3-1
(`PROTOCOLO_E5.3-1_PREREGISTRO.md` §2.1: amplitud `0.01·sqrt(D_ε)` por sitio y por paso) está
atado a `D_ε`, que es **casi constante entre epsilons** (≈8.4e-4, medido, igual que reporta la
propia tabla `meta_por_eps` de `cs074_rcruz.py`), mientras que `E_total` escala como `N·ε²`. El
cruce señal/ruido (`E_total ≈ N·ruido_amp²·pasos`) ocurre en **ε≈0.016** — es decir, de los 13
puntos de la rejilla de 12 décadas (`1e-12 … 1e0`), **solo los 2 puntos más altos (ε=0.1 y
ε=1.0) tienen presupuesto inicial mayor que el ruido acumulado**; los 11 restantes están
dominados por ruido, no por estructura. Esto es un defecto de la definición canónica
compartida (`PROTOCOLO_E5.3-1_PREREGISTRO.md`), NO de este motor ni de `cs074_rcruz.py` —
**no se edita el protocolo ajeno, se reporta a CS integro** (regla de ejecución #7, T6).

**Decisión (declarada ANTES de la corrida de producción, no después de ver el resultado
final):** se agrega al motor un **diagnóstico SNR puramente observacional** (no toca la
física, no cambia ninguna fórmula del campo/ruido/expansión ni del cómputo de eficiencia):
por celda, `ruido_acumulado_esperado = N·ruido_amp²·pasos` (varianza esperada de una caminata
aleatoria gaussiana independiente por sitio a lo largo de `pasos` pasos) contra
`E_total_medio` (promedio de semillas); `snr_saludable = E_total_medio > ruido_acumulado_esperado`.
Se corre la rejilla COMPLETA igual (T5: la curva entera, contaminada o no, se entrega — no se
recorta el barrido para ocultar el defecto), y se reportan **dos** resúmenes de distancia por
separado: uno con TODAS las celdas (crudo, tal como sale del protocolo canónico) y otro
filtrado a `snr_saludable` (diagnóstico honesto de qué sobrevive cuando el ruido no ahoga la
señal). El método E5.3-2 (ANOVA) liga su ruido a `NOISE_REL·ε` — proporcional a ε — y por eso
NO exhibe esta patología (verificado en el mismo humo: valores acotados en [0,1], sin viola
ción de conservación); se reporta igual, sin el filtro SNR (no aplica).

Ningún número de 4.9%/31.5% participó en esta decisión — el umbral SNR=1 es una comparación
señal-contra-ruido interna al motor, fijada ANTES de ver las distancias a los blancos.

---
*Firmado antes de escribir una sola línea de `E5_3_5_motor.py`. Cualquier desviación técnica
necesaria durante la implementación se declara explícitamente en el reporte final, nunca en
silencio.*
