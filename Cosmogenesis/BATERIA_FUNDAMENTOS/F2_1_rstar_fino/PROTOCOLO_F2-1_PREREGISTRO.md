# PROTOCOLO F2-1 — PRE-REGISTRO
## "Umbral de congelamiento: barrido fino de r cruzando 1, con r* resuelto"

**Fecha/hora de congelamiento (UTC):** 2026-07-24T09:33:28Z, ANTES de correr el motor.
**Ejecutor:** CC (agente paralelo, prefijo `F2_1_`), batería de 24 experimentos.
**Base:** `cs074_rcruz.py` (NO editado, solo importado). Antecedentes leídos: `ADJUDICACION_CF-1_sello_CS.md`,
`RESUMEN_CS074_rcruz_robustez_N_PARA_CS.md`. El run previo (producción/robustez400) usó
`R_TARGETS = [0, 0.1, 0.3, 0.5, 1, 2, 5, 10, 30, 100]` — un solo punto entre r=0 y r=0.1.
Este experimento resuelve ese hueco con grid fino.

---

## 1. Pregunta

¿Dónde exactamente enciende el congelamiento al subir r=H/D, resuelto con un grid MUY
fino (no un solo bin), y es ese r* estable al subir N, o depende de N?

## 2. Observable (idéntico al núcleo CS074, sin cambiarlo)

`persistencia(phi, contraste0)` = correlación a primer vecino (`corrcoef(phi, roll(phi,1))`,
clip a ≥0) × varianza normalizada (`var(phi)/contraste0²`). Mismo observable forma×magnitud
validado en CF-1 (hace morder al NULL — defecto de instrumento ya corregido en sesión previa).
**No se cambia el observable en este experimento** (T2: la cantidad medida ya fue fijada
fuera de este experimento).

## 3. Grid de r (FINO, cruzando la región de transición)

Rango pre-registrado: **r ∈ [0.01, 3]**, log-espaciado con densificación adicional entre
[0.6, 1.5] (justo cruzando r=1, tal como pide el enunciado), MÁS un punto de control r=0
(fuera del rango fino, heredado de CS074 como chequeo de lavado — no cuenta para el
mínimo de 25).

```python
r_fino = sorted(set(np.round(np.concatenate([
    np.geomspace(0.01, 3.0, 25),
    np.linspace(0.6, 1.5, 10),
]), 6)))
# 32 puntos únicos en [0.01, 3] + r=0 de control = 33 puntos totales
```

**Deviación de compute pre-declarada (antes de correr, no después):** el costo del motor
heredado escala ~N³ (pasos_lavado ~ 1/D ~ N², multiplicado por N barridos × r × semillas).
Benchmark empírico en esta máquina (medido antes de correr el experimento real, ver
`benchmark_log.txt`): ~85-130 µs/paso, casi independiente de N (dominado por overhead de
Python, no por tamaño de array). Con eso, N=1600 con el grid completo (33 pts × 16 semillas
× 2 [real+NULL]) se estima en ~12h *single-thread*, con 23 agentes hermanos compitiendo por
16 núcleos. Por eso, **para N=1600 se usa un sub-grid de 15 puntos** (mismo rango [0.01,3],
misma densificación relativa cerca de 1, construido determinísticamente tomando 1 de cada 2
puntos del grid fino de 33 + los extremos) — declarado AQUÍ, antes de ver un solo resultado.
N=200/400/800 corren el grid completo de 33 puntos.

## 4. Ejes del barrido

- **r:** grid fino de la sección 3 (33 pts para N∈{200,400,800}; 15 pts para N=1600).
- **N:** {200, 400, 800, 1600}.
- **semillas:** 16 (índices 0..15), cada semilla determina TANTO la condición inicial COMO
  el flujo de ruido dinámico (ver §5) — no son repeticiones triviales de una PDE casi
  determinista (lección CF-2/d622550b).
- **ε (amplitud de la mancha inicial):** fijo en **ε=1e-3** (mismo valor de calibración de
  referencia usado en CS074-rcruz producción/robustez400, régimen ya validado como
  persistente y no nulo). F2-1 NO barre ε — ésa es la responsabilidad de F1-3; aquí se
  mantiene fijo para no mezclar los dos ejes y poder invertir el compute en resolver r*.
  **Chequeo barato de robustez a ε (solo N=200, no cuenta para el veredicto principal):**
  se repite el grid fino completo también con ε=1e-2, para verificar que r* no se mueve
  groseramente por la elección de ε.
- **Perturbación dinámica (T7):** en cada paso de difusión se inyecta ruido aditivo
  gaussiano de amplitud relativa fija `sigma_rel=0.01` (1% de la std instantánea del
  campo), generado por el rng propio de cada semilla — el ruido actúa EN CADA PASO, no
  solo en la condición inicial. Esto es lo que CF-1/CF-2 no tenían: ahora las 16 semillas
  son trayectorias dinámicamente distintas, no la misma PDE casi-determinista con 16
  arranques distintos que convergen al mismo número.

## 5. Motor (nuevo, en este directorio, NO toca cs074_rcruz.py)

`F2_1_motor.py` importa `campo_inicial`, `paso_difusion`, `paso_expansion`, `persistencia`,
`medir_D`, `medir_pasos_lavado`, `detectar_cuantizacion`, `temperatura_fisica` de
`cs074_rcruz.py` sin modificarlos. Añade únicamente:
- `paso_ruido_dinamico(phi, sigma_rel, rng)`: `phi + rng.normal(0, sigma_rel*phi.std(), N)`,
  aplicado tras cada `paso_difusion` y antes de `paso_expansion`.
- `evolucionar_f21(...)`: mismo bucle que `evolucionar()` de CS074 pero con el paso de
  ruido intercalado; NULL = permutar `phi` al final (idéntico a CS074, "barajado del
  acople al final").

## 6. NULL

Idéntico al núcleo: `rng.permutation(phi)` una sola vez al final de la evolución (destruye
forma espacial, conserva histograma). Se corre pareado con REAL (misma semilla, mismo r,
mismo ruido dinámico hasta el punto de permutación).

## 7. Calibración de pasos

`pasos_lavado` se mide (no se impone) por N vía `medir_pasos_lavado` a ε=1e-3, igual que
CS074 (umbral P_LAVADO=0.05, margen ×1.15). Se usa el mismo `pasos_fijo` para todos los r
de ese N (igual que producción/robustez400 de CS074), de modo que el único eje que cambia
entre puntos del grid es r (vía H=min(r·D,1)).

## 8. Estimadores de r* (TODAS se reportan — no se elige una a posteriori, T3)

Para cada N, sobre la curva P_real(r) (media de 16 semillas):
1. `r_half_rise`: interpolación lineal en log(r) del cruce por el punto medio entre
   P(r→0⁺, primer punto del grid fino) y P(r_max del grid).
2. `r_P>0.2`, `r_P>0.5`, `r_P>0.8`: primer r del grid (interpolado) donde P_real cruza
   cada umbral.
3. Pendiente máxima `dP/d(log r)` y el r donde ocurre (punto de inflexión numérico).

Se reporta la ley r*(N) para **cada** una de estas métricas por separado — no se afirma
"invariante" citando solo la que más convenga (lección explícita de d622550b, citada en la
instrucción del director).

## 9. Verificación cruzada (obligatoria, T4/T7)

(a) **NULL cae:** en cada punto (N, r) se reporta z=(P_real−P_null)/sd_pooled; se verifica
que P_null se mantenga bajo (no crece con r) mientras P_real sube — si el NULL también
sube con r, es un FAIL de instrumento y se reporta como tal, sin editar el observable.
(b) **r* estable al subir N:** se reporta la curva r*(N) (todas las métricas de §8) — PASS
cualificado solo si al menos la métrica de encendido (half-rise o P>0.5) es estable dentro
de un factor ~2 en el rango N=200→1600; si se mueve más, se reporta como tal (no invariante).
(c) **Disco:** cada corrida escribe JSON crudo por N (`F2_1_N{N}_resultado.json`) más un
consolidado (`F2_1_consolidado.json`) con las 16 semillas × r × N sin promediar de antemano
(dispersión completa, no solo la media), auditable por quien no escribió este código.

## 10. Criterio de PASS (congelado, T3 — no se toca tras ver resultados)

- **PASS cualificado** si: (a) el NULL no reproduce la subida de P_real con r en ningún N;
  (b) al menos una métrica de "encendido" (half-rise o P>0.5) es estable (factor ≤2) en
  N∈{200,400,800}; y (c) se reporta explícitamente si N=1600 confirma o contradice esa
  estabilidad (con el sub-grid de 15 pts).
- **FAIL / hallazgo negativo** si el NULL también sube con r (T4 roto), o si r* se
  desplaza sistemáticamente (>factor 2) al subir N sin asíntota visible — se reporta como
  hallazgo, no se disfraza.
- Ninguna lectura se declara "cierre de arco": es insumo crudo para CS, sin autoadjudicación.

## 11. Trampas que evita explícitamente

- **T5/T7 (las centrales de este experimento):** grid fino (33/15 pts) reportado ENTERO,
  no un bin; perturbación dinámica real (ruido por paso, no solo semilla).
- **T1:** ε fijo declarado y justificado (no elegido para dar resultado — es el valor de
  referencia ya usado en el núcleo); r, N y semillas se barren.
- **T3:** este documento se congela antes de correr `F2_1_motor.py`. Si algo falla, se
  reporta el FAIL, no se edita este archivo.
- **T4:** el NULL se reporta en cada punto, no solo al final.

---

**Firmado (congelado) antes de ejecutar el motor.** Cualquier desviación de lo aquí escrito
se declara explícitamente en el reporte final, no se oculta.
