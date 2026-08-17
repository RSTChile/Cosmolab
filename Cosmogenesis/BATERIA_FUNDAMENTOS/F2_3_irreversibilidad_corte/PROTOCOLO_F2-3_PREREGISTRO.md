# PROTOCOLO F2-3 — PRE-REGISTRO
## "Irreversibilidad del corte: ¿el congelamiento requiere cortes permanentes?"

**Fecha/hora de pre-registro:** 2026-07-24 05:33 (hora local, America/Santiago -04)
**Ejecutor:** CC (agente de batería, prefijo `F2_3_`)
**Base autorizada (NO editada):** `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py`
**Spec autoritativa:** `BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md`, sección "F2-3"

Este documento se escribe y se congela **ANTES** de correr el motor. No se edita
después de ver resultados (regla T3). Si el resultado no cumple el PASS
pre-registrado, se reporta el FAIL tal cual — no se cambia el juez.

---

## 1. PREGUNTA

CS074-rcruz corta aristas de forma **irreversible**: una vez cortada, una arista
nunca vuelve a conducir difusión. F2-3 pregunta si esa irreversibilidad es el
**mecanismo real** del congelamiento (persistencia P alta a r≫1), no solo una
propiedad incidental correlacionada con r. La forma de aislar el mecanismo es
introducir una probabilidad de **reconexión** de aristas cortadas y barrerla de
0 (modelo actual, irreversible) a 1 (totalmente reversible), a r fijo.

**Predicción fuerte pre-registrada:** si la irreversibilidad es el mecanismo,
P debe caer **monótonamente** al subir la probabilidad de reconexión, para
cualquier r con congelamiento (r≥1). En el límite de reconexión=1, P debe
acercarse al nivel de P_null (la persistencia se vuelve indistinguible de la
que se logra por puro barajado). Si P NO cae al subir la reconexión —o cae y
luego sube, o no se acerca a NULL en reconexión=1— el mecanismo propuesto
(irreversibilidad del corte) está mal, y así se reporta.

---

## 2. MODIFICACIÓN AL MOTOR (mínima, aditiva, no toca cs074_rcruz.py)

Se clona la física exacta de `cs074_rcruz.py` (campo inicial, `paso_difusion`,
`persistencia`, `medir_D`, `medir_pasos_lavado`, NULL por permutación) en un
motor nuevo (`F2_3_motor_reconexion.py`, en esta carpeta). La ÚNICA adición es
la función de expansión:

```
paso_expansion_reconexion(activo, H, p_recon, rng):
    activo = paso_expansion(activo, H, rng)      # corte, IDÉNTICO a CS074-rcruz
    if p_recon > 0:
        inactivas = ~activo
        si p_recon >= 1: reconectar TODAS las inactivas
        si no: Bernoulli(p_recon) independiente por arista inactiva
    devuelve activo
```

Orden por paso: corte (Bernoulli, prob H, igual que el original) → reconexión
(Bernoulli, prob `p_recon`, sobre las aristas que quedaron inactivas ese mismo
paso, incluidas las recién cortadas). Es la generalización mínima: en
`p_recon=0` el motor es **matemáticamente idéntico** a `cs074_rcruz.py` (mismo
generador de aleatorios, mismo orden de llamadas) — esto se usa como
verificación de identidad del mecanismo (§5, chequeo T2).

**Por qué este orden y no otro:** cada arista es una cadena de Markov de 2
estados por paso (viva↔muerta) con prob. de transición muerte=H,
reparación=p_recon. Esto hace que la fracción de aristas muertas en régimen
estacionario tienda a H/(H+p_recon) (aprox., para H,p_recon pequeños) — en
p_recon≫H casi ninguna arista queda muerta (reversible), en p_recon≪H casi
todas quedan muertas (irreversible), reproduciendo el modelo original en el
límite p_recon→0.

---

## 3. OBSERVABLE PRINCIPAL Y OBSERVABLE SECUNDARIO (independiente)

- **Observable principal (P):** persistencia = autocorrelación a primer vecino
  × razón de varianza, idéntica a `persistencia()` de `cs074_rcruz.py`
  (`c = corr(φ, roll(φ,1))`, `v = var(φ)/contraste0²`, `P = c·v`). Ya usada en
  F1-1/F2-1/F2-2 — mismo juez en toda la batería.
- **Observable secundario, método independiente (T2):** `std_ratio` =
  `φ.std()/contraste0` (retención de magnitud del contraste, SIN el término
  de forma/autocorrelación). Si el veredicto de P coincide con el de
  std_ratio, la caída no es artefacto del observable de forma.
- **Observable mecanístico literal (tercera lente, la que prueba el POR QUÉ,
  no solo el QUÉ — exigencia T2 explícita de F2-3):** `frac_exp` = fracción de
  aristas inactivas (cortadas) al final de la corrida. Esta es la cantidad que
  la reconexión manipula directamente. Se reporta frac_exp(p_recon) junto a
  P(p_recon): si P cae porque frac_exp cae (menos aristas muertas → más
  difusión activa → menos persistencia), el mecanismo queda probado, no solo
  correlacionado.

---

## 4. BARRIDO (congelado antes de correr)

| Eje | Valores | Justificación |
|---|---|---|
| `prob_reconexión` | {0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.6, 1.0} — 11 puntos | ≥10 exigido. Log-denso cerca de 0 porque H (la tasa de corte) vive en [8e-5…0.084] en nuestro grid de r — la transición reversible↔irreversible ocurre donde p_recon≈H, no a mitad del rango [0,1]. Un grid lineal grueso la saltaría (mismo error que motivó el r fino de F2-1). Termina exactamente en 0.0 (modelo actual) y 1.0 (reversible total). |
| `r` (= H/D, D medida) | {0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0} — 10 puntos | Mismo `R_TARGETS` que `cs074_rcruz.py` (comparabilidad directa con F2-1/F2-2). r=0 es control mecanístico: sin cortes, la reconexión no tiene nada que hacer (P debe ser plana en p_recon). |
| semillas | 12 (seeds 1000..1011) | ≥12 exigido (T7). |
| eps | fijo en 0.1 | `cs074_rcruz` produccion muestra P_real/P_null **invariante a eps>0** (la normalización de `persistencia()` por contraste0 la hace scale-free) — confirmado empíricamente en `cs074_rcruz_produccion_resultado.json` antes de este barrido (mismos P a eps∈{1e-3,1e-2,0.1,0.5}). Barrer eps aquí sería redundante con F1-3; F2-3 barre lo que le corresponde: prob_reconexión × r. |
| N, pasos | N=200, pasos_fijo calibrado UNA vez (medir_pasos_lavado con H=0, eps=0.1, 12 semillas), igual para toda la grilla | Fijar la ventana temporal evita que cambios en pasos confundan con el efecto de p_recon (mismo criterio de F2-1: D y pasos medidos, no puestos a mano). |

**Control adicional (sanity, no parte del veredicto principal):** eps=0.0
estricto, r∈{0.0,1.0,100.0} × p_recon∈{0.0,0.5,1.0} × 4 semillas → debe dar
P=0 en todos los casos (sin diferencia inicial no hay nada que persista,
independiente de la reconexión).

Total corridas grilla principal: 10(r) × 11(p_recon) × 12(semillas) × 2(real+null)
= 2640 corridas. Más 24 corridas del control eps=0.

---

## 5. NULL Y VERIFICACIÓN CRUZADA

- **NULL:** barajado del acople — se permuta φ al final de la corrida
  (idéntico a `null=True` de `cs074_rcruz.py`; misma semilla que su pareja
  REAL, para comparación pareada).
- **Verificación (a):** el NULL debe permanecer bajo (cerca de 0) en todo el
  grid de p_recon — si el NULL sube con p_recon, algo más que el barajado
  está cambiando y se reporta.
- **Verificación (b) — identidad del mecanismo en p_recon=0:** los valores de
  P_real, P_null, D, frac_exp en p_recon=0 deben **coincidir** (dentro de
  ruido de semilla) con los de `cs074_rcruz_produccion_resultado.json` para
  los mismos r. Si no coinciden, el motor F2-3 no es una extensión fiel del
  modelo base y se reporta el desvío exacto antes de interpretar nada más.
- **Verificación (c) — segundo observable:** std_ratio(p_recon) debe mostrar
  el mismo veredicto cualitativo (caída monótona) que P(p_recon).
- **Verificación (d) — mecanismo literal:** frac_exp(p_recon) debe caer
  monótonamente (menos aristas cortadas al reconectar más), y esa caída debe
  **preceder/acompañar** la caída de P — si frac_exp cae pero P no, la
  irreversibilidad de las aristas no es lo que sostenía P y se reporta así.
- **Auditoría en disco:** código fuente (`F2_3_motor_reconexion.py`) + JSON
  crudo con las 2640+24 filas, sin agregación previa, quedan en esta carpeta
  para quien audite sin haber escrito el código.

---

## 6. CRITERIO DE PASS (fijado antes de correr, no se toca después — T3)

Para cada r con r≥1 (régimen de congelamiento esperado, r∈{1.0,2.0,5.0,10.0,30.0,100.0}):

1. **PASS fuerte:** P_real(p_recon) es monótonamente NO-creciente dentro de la
   dispersión entre semillas (correlación de Spearman entre p_recon y
   P_real ≤ −0.8, p<0.01), Y P_real(p_recon=1.0) cae dentro de 2 desviaciones
   estándar (entre semillas) de P_null(p_recon=1.0).
2. **FAIL del mecanismo:** si P_real NO cae monótonamente (Spearman > −0.8 o
   signo positivo en algún tramo fuera del ruido), o si P_real(p_recon=1.0)
   permanece muy por encima de P_null(p_recon=1.0) (>2 std) — la
   irreversibilidad NO explica el congelamiento y se reporta como tal, sin
   suavizar.
3. Para r<1 (régimen ya sin congelamiento en el modelo base): se reporta la
   curva igual, pero no se exige el mismo umbral (P ya es bajo en p_recon=0,
   así que "caída monótona" puede ser plana/trivial — se dice explícitamente
   si es el caso, sin forzar interpretación).
4. **Veredicto global de F2-3:** PASS solo si el criterio 1 se cumple en LA
   MAYORÍA (≥4 de 6) de los r≥1 evaluados. Si falla en más de 2, el veredicto
   es FAIL DEL MECANISMO propuesto (no es un error del código — es un
   hallazgo: el mecanismo de irreversibilidad no sostiene el congelamiento
   observado).

Ningún ajuste de este criterio se hace después de ver la curva. Cualquier
resultado (PASS o FAIL) se reporta completo, curva entera, sin adjudicar
cierre — la adjudicación es de CS.

---

## 7. QUÉ EVITA (trampas)

- **T1:** ningún p_recon "elegido para dar el resultado" — se barre todo
  [0,1] y se reporta la curva completa, no un punto.
- **T2 (la exigida explícitamente por F2-3):** se mide frac_exp (la cantidad
  mecanística literal que la reconexión manipula) además de P — prueba el
  POR QUÉ, no solo que P cambia.
- **T3:** este documento se congela ANTES de correr; el criterio de §6 no se
  edita después.
- **T4:** el NULL se verifica que cae (no se asume).
- **T5:** curva completa de 11 puntos de p_recon reportada, no un umbral
  binario.
- **T7:** 12 semillas + barrido extenso en dos ejes (p_recon × r), no un
  punto ni solo semillas.

---

## 8. ARCHIVOS QUE PRODUCE ESTE EXPERIMENTO

- `F2_3_motor_reconexion.py` — motor (clon fiel de la física de cs074_rcruz.py
  + la función de reconexión aditiva).
- `F2_3_resultado_crudo.json` — filas crudas (grilla principal + control eps=0).
- `F2_3_analisis.py` — script de análisis (monotonicidad Spearman, comparación
  p_recon=0 vs 1 vs NULL, dispersión entre semillas). Solo lee el JSON, no
  re-corre el motor.
- (este archivo) `PROTOCOLO_F2-3_PREREGISTRO.md`.

Ningún archivo existente fuera de `BATERIA_FUNDAMENTOS/F2_3_irreversibilidad_corte/`
es editado.
