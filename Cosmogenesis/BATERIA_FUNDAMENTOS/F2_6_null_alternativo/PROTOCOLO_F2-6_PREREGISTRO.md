# PROTOCOLO F2-6 — PRE-REGISTRO (fechado, congelado antes de correr el motor)

**Experimento:** F2-6 · "NULL alternativo: barajar la historia de cortes, no el acople final"
**Enfoque:** 2 — ¿la expansión congela la diferencia? (competencia expansión vs reabsorción)
**Ejecutor:** CC (agente de esta corrida, prefijo `F2_6_`)
**Fecha/hora de congelamiento:** 2026-07-24 05:32 (hora local del sistema, America/Santiago -04)
**Documento fuente:** `BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md`, sección "F2-6"
**Código base (NO editado, solo importado/leído):** `cs074_rcruz.py` (raíz de `Cosmogenesis/`)

Este documento se escribe y se congela ANTES de ejecutar el motor de producción.
No se edita después de ver resultados (regla T3). Si algo falla, se reporta el
FAIL tal cual.

---

## 1. Hipótesis / pregunta

El NULL usado en toda la batería del Enfoque 2 (incl. cs074_rcruz.py y F2-1..F2-5)
es "barajar el campo φ al final" — destruye la forma espacial pero conserva
intacta la HISTORIA de qué aristas se cortaron y cuándo (el grafo final y su
construcción temporal quedan sin tocar; solo se revuelven los VALORES del campo).

Pregunta de F2-6: **¿el veredicto REAL≻NULL depende de CÓMO se construye el azar
de comparación?** Si el mecanismo real es "la expansión aísla regiones y eso
congela la diferencia", entonces un NULL que ataque un aspecto distinto —la
SECUENCIA TEMPORAL de qué arista se corta cuándo, dejando el GRAFO FINAL
intacto (mismo conjunto de aristas cortadas, mismo φ inicial, misma física de
difusión)— debería TAMBIÉN perder frente al REAL. Si REAL solo le gana al NULL
clásico pero no a este NULL de secuencia, el resultado es frágil (T4).

## 2. Observable exacto (congelado, idéntico al de la batería)

`P = corr(φ, roll(φ,1)) · [var(φ) / var(φ₀)]`

función `persistencia()` de `cs074_rcruz.py` (líneas 152-160), **importada sin
modificar** (`import cs074_rcruz as base`; se llama `base.persistencia(...)`).
No se copia a mano — así el observable es byte-idéntico al de todo el resto de
la batería (condición necesaria para que la comparación entre nulos sea justa).

El observable no es función de ε, r, N ni de qué NULL se usó — se mide sobre el
campo φ final que produzca cada rama (T2: no circular).

## 3. Física reutilizada sin cambios (todo importado de `cs074_rcruz.py`)

- `campo_inicial(N, eps, rng)` — fondo=1 + ε·perturbación multi-modo Fourier.
- `paso_difusion(phi, activo)` — difusión vectorizada por aristas vivas.
- `paso_expansion(activo, H, rng)` — corte Bernoulli por arista viva con
  probabilidad H por paso.
- `medir_D(N, eps, seed)` — D medido del propio campo (H=0, un paso).
- `medir_pasos_lavado(N, eps, semillas)` — pasos calibrados por lavado (P<0.05,
  margen 1.15×), **una vez por N, a eps=1e-3, aplicado fijo a todo el barrido**
  — misma convención que el modo "produccion" del código base.
- `R_TARGETS = [0, 0.1, 0.3, 0.5, 1, 2, 5, 10, 30, 100]` — mismo eje de r que el
  código base (cruza r=1). F2-6 no pide resolución fina de r* (eso es F2-1); pide
  la robustez del NULL en el mismo eje ya validado.

**Nada de esto se reimplementa a mano** — se importa el módulo para que la
dinámica de F2-6 sea, físicamente, la MISMA que corre el resto del Enfoque 2.

## 4. Los DOS nulos (el corazón de F2-6)

Ambos nulos se derivan de la **misma corrida REAL** (mismo seed, mismo H, mismo
`pasos`, mismo φ inicial, misma secuencia de cortes real) — condición explícita
del documento madre: "con ambos NULLs... sobre los mismos casos".

### 4.1 NULL-clásico (el ya usado en cs074_rcruz.py y el resto del Enfoque 2)

Se corre la dinámica REAL completa (difusión + expansión, `pasos` pasos) y al
final se permuta el campo: `phi_null = rng_c.permutation(phi_real_final)`. Esto
destruye la FORMA espacial pero conserva el histograma exacto de valores, el
grafo final de aristas activas y toda la historia temporal de cortes que
efectivamente ocurrió — el barajado ataca solo el VALOR, no la CONSTRUCCIÓN.

### 4.2 NULL-secuencia (nuevo, el que pide F2-6)

Durante la corrida REAL se graba, para cada arista, el paso exacto en que fue
cortada (`cut_step[e] ∈ {1..pasos}`, o `-1` si nunca se cortó en el horizonte).
Esto es la "historia de cortes".

El NULL-secuencia **baraja esa historia, no el grafo final**:

1. Se toma el subconjunto de aristas que SÍ se cortaron en la corrida REAL y el
   vector de sus tiempos de corte `{t_1, ..., t_k}` (k = # aristas cortadas).
2. Se permutan esos tiempos ENTRE esas mismas aristas (`rng_s.permutation`) —
   es decir, la arista que en REAL se cortó primero puede terminar cortándose
   última, y viceversa, pero el CONJUNTO de aristas cortadas al final del
   horizonte es exactamente el mismo (mismo grafo final — condición literal del
   nombre del experimento), y el histograma de "cuántos cortes hubo en cada
   paso" también es exactamente el mismo (es una permutación de la misma
   multiserie de tiempos).
3. Se reconstruye la secuencia `activo(t)` a partir de este `cut_step` barajado
   (arista activa en el paso t ⟺ `cut_step==-1` o `cut_step ≥ t`) y se re-corre
   la difusión desde el MISMO φ inicial bajo este calendario de cortes
   alternativo — **sin tocar el campo final, sin permutarlo**. `P_null_seq` se
   mide directamente sobre el resultado de esta re-simulación.
4. **Auto-chequeo obligatorio (T6, puede fallar):** el grafo final de la
   réplica (aristas activas tras `pasos` pasos) debe ser IDÉNTICO byte a byte
   al grafo final de la corrida REAL. Se verifica con `np.array_equal` en cada
   combinación y se registra `grafo_final_identico` en el JSON. Si alguna vez
   es `False`, es un bug de implementación y se reporta como FALLO, no se
   interpreta el resto de esa fila.

**Por qué esto es un NULL distinto en naturaleza (no una variante cosmética del
clásico):** el NULL-clásico revuelve QUÉ VALOR quedó en cada sitio; el
NULL-secuencia revuelve CUÁNDO se aisló cada sitio, dejando el valor final que
la física produjo y el grafo final intactos. Si el mecanismo causal es "aislar
temprano congela mejor que aislar tarde" (irreversibilidad + orden), el
NULL-secuencia lo ataca directamente; el NULL-clásico no.

## 5. Barrido (congelado)

| Eje | Rango | Puntos |
|---|---|---|
| r = H/D | `R_TARGETS` del código base: {0, 0.1, 0.3, 0.5, 1, 2, 5, 10, 30, 100} | 10 |
| ε | igual que modo "produccion" de cs074_rcruz: {0, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.5, 1.0} | 8 (incluye control ε=0) |
| N | 200 (fijo — el barrido de N es tarea de F2-1/robustez400, no de F2-6) | 1 |
| semillas | 16 (≥12 pre-registrado), seeds 1000..1015, independientes por combinación | 16 |

Total combinaciones (ε×r) = 8 × 10 = 80 puntos de grid, cada uno con 16
semillas × 1 corrida REAL (que genera P_real, P_null_clasico y P_null_seq de
forma acoplada, como exige "sobre los mismos casos") = 1280 corridas REAL con
historia grabada + 1280 réplicas de NULL-secuencia.

**Perturbación dinámica:** F2-6 no pide barrido de ruido dinámico adicional
(eso es F1-5/F3-1); su perturbación de robustez es la cobertura de 16 semillas
independientes por punto (r,ε) — más de las 12 mínimas pre-registradas.

## 6. NULLs (resumen, ya descritos en §4)

- NULL-clásico: permutación del campo final (`rng.permutation`), el de
  `cs074_rcruz.py`.
- NULL-secuencia: permutación de los tiempos de corte entre las aristas
  cortadas, replay de la difusión, sin tocar el campo final.

## 7. Controles (congelados)

- **Control ε=0:** sin perturbación sembrada, `contraste0=0` → `persistencia()`
  devuelve 0.0 por construcción (rama `contraste0<=0`) en las tres ramas
  (REAL, NULL-clásico, NULL-secuencia) — control trivial pero se reporta la
  fila completa igual, sin excluirla.
- **Control r=0 (H=0):** a ε>0 no hay cortes → `cut_step` queda todo en -1 →
  el NULL-secuencia es estructuralmente IDÉNTICO al REAL (nada que barajar).
  Es el punto donde ambos nulos deben colapsar sobre el REAL (ningún margen).
  Se reporta explícitamente, no se oculta como caso degenerado.

## 8. Criterio de PASS (congelado, tres lecturas — no se cambia tras ver datos)

Se define, por punto (ε,r) con ε>0 y r≥10 (banda de congelamiento, misma banda
que usa F1-1/cs074_rcruz), sobre las 16 semillas:

- `z_clasico = (P_real_mean − P_null_clasico_mean) / sd_combinada`
- `z_seq = (P_real_mean − P_null_seq_mean) / sd_combinada`
  (sd_combinada = sqrt((var_real+var_null)/2), con piso 1/n_semillas — misma
  fórmula que usa `barrido_rcruz` en `cs074_rcruz.py`).

1. **Gana al NULL-clásico:** `z_clasico ≥ 3` en ≥50% de los puntos (ε>1e-6,
   r≥10).
2. **Gana al NULL-secuencia:** `z_seq ≥ 3` en ≥50% de esos mismos puntos.
3. **Veredicto robusto (PASS de F2-6):** (1) y (2) se cumplen SIMULTÁNEAMENTE
   en los MISMOS puntos (no en puntos distintos). Si gana a uno y no al otro
   en el mismo punto → ese punto se marca **FRÁGIL**, no se promedia para
   ocultarlo.
4. **Control r=0:** ambos `z` deben ser ≈0 (o no significativos) — si no, hay
   un bug (el NULL-secuencia no puede diferir del REAL cuando no hubo cortes).
5. **Control ε=0:** las tres P deben ser exactamente 0.0.

**No se ajusta este criterio después de correr.** El veredicto de LECTURA para
la batería completa (qué significa que sea frágil o robusto) lo da CS con la
curva cruda; este documento fija el gate mecánico de PASS/FRÁGIL/FAIL de ESTE
experimento.

## 9. Verificación cruzada (tres vías, obligatorias)

(a) **Su(s) NULL(s)** — es el propio experimento: dos nulos independientes
    comparados en paralelo sobre los mismos casos (descrito arriba).
(b) **Segundo método/observable** — F2-6 no introduce un observable nuevo (no
    es su tarea, eso es F1-2/F2-2); su segunda vía de verificación es el
    auto-chequeo estructural `grafo_final_identico` (§4.2 punto 4): garantiza
    que el NULL-secuencia de verdad preserva el grafo final y solo cambia el
    ORDEN — si esto fallara, invalidaría la interpretación del resultado
    aunque las P dieran "bien".
(c) **Auditoría en disco** por quien no escribió el código: JSON crudo con
    las 1280 filas (P_real, P_null_clasico, P_null_seq, z_clasico, z_seq,
    grafo_final_identico, conteo de aristas cortadas) + este protocolo + log
    con timestamps, en `BATERIA_FUNDAMENTOS/F2_6_null_alternativo/resultados/`.

## 10. Qué puede fallar (T6 — todo gate debe poder fallar)

- REAL podría ganarle al NULL-clásico pero NO al NULL-secuencia (o viceversa)
  → veredicto FRÁGIL, se reporta como tal, no se suaviza.
- `grafo_final_identico` podría dar `False` → bug de reconstrucción, se PARA
  y se reporta a CS con la fila exacta, no se sigue interpretando esa rama.
- El NULL-secuencia podría no diferir del NULL-clásico en ninguna banda
  (mismo z aproximadamente) → hallazgo: la ORDEN temporal no aporta nada más
  allá del grafo final — es negativo legítimo, se reporta.
- z podría ser inestable / con dispersión grande entre semillas → se reporta
  la dispersión real (P_real_std, P_null_clasico_std, P_null_seq_std por
  punto), no se oculta con el promedio.

## 11. Archivos de salida

- `resultados/F2_6_smoke_resultado.json` — corrida pequeña de validación
  (N chico, pocos pasos/semillas) para verificar mecánica antes de producción.
- `resultados/F2_6_produccion_resultado.json` — barrido completo (80 puntos ×
  16 semillas).
- `resultados/F2_6_log_ejecucion.txt` — log con timestamps de inicio/fin de
  cada fase.

---
*Congelado. No editar después de este punto salvo para anotar FAIL explícito de
algún control (T3: no se cambia el juez tras el resultado).*
