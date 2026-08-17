# PROTOCOLO F2-2 — PRE-REGISTRO
### "D emergente multi-paso: ¿el r* nominal es el crítico real?"

**Fecha/hora de este pre-registro (UTC):** 2026-07-24T09:30:48Z
**Ejecutor:** CC (agente F2-2, batería paralela de 24 experimentos)
**Prohibido tras esto:** cambiar el criterio de PASS, la fórmula de r*, o el grid, después
de ver resultados (T3). Si algo falla, se reporta el FAIL.

---

## 0. Pregunta

`cs074_rcruz.py` mide la reabsorción/difusión D con `medir_D()`: la fracción de contraste
(std) que se borra en **UN solo paso** de difusión pura (H=0), y define `r = H/D`. El campo
real sigue difundiéndose por muchos pasos, no uno — y el primer paso puede no representar
la tasa efectiva a largo plazo (el campo inicial mezcla varios modos de Fourier que decaen
a tasas distintas; tras muchos pasos dominan los modos lentos). Pregunta: **si se mide D en
escalas de 1, 2, 5, 10 y 50 pasos y se usa esa D para redefinir r, ¿el cruce r* (donde la
persistencia transiciona) se mueve hacia r≈1, o el "r=1" del código base no es el crítico
físico real?**

## 1. Física reusada (NO reimplementada)

Se importa directamente de `cs074_rcruz.py` (solo lectura, sin editar el archivo):
`campo_inicial`, `paso_difusion`, `paso_expansion`, `persistencia`, `medir_D`,
`medir_pasos_lavado`, `corrida`, `evolucionar`, `temperatura_fisica`, `T_SING`, `T_FIN`.
Esto garantiza que la dinámica acoplada (difusión + expansión) es EXACTAMENTE la misma
que usa/usará F2-1; solo se reinterpreta el eje r.

**Hecho clave que se explota (no es un atajo, es honesto):** la dinámica acoplada
(`paso_difusion` + `paso_expansion`) depende únicamente de **H**, nunca de D ni de r
directamente — r es un cociente diagnóstico, no un parámetro de la simulación. Por lo
tanto:
- El barrido caro (grid de H × semillas × N, con difusión+expansión corriendo miles de
  pasos) se corre **UNA sola vez** por combinación (N, eps).
- D_k (k∈{1,2,5,10,50}) se mide **aparte**, en corridas de difusión PURA (H=0) de k pasos,
  mucho más baratas.
- Redefinir r con D_k es **reetiquetar el eje x** de los mismos puntos (mismo P_real,
  mismo P_null, mismo H) — no correr la dinámica de nuevo. Esto es lo que pide el
  experimento ("redefinir r con la D multi-paso"), no un experimento físico distinto.

## 2. Definición de D_k (dos métodos, cross-check)

Para cada semilla, se genera `campo_inicial(N, eps, rng)` y se corre `paso_difusion` en
bucle puro (activo=todo vivo, H=0) grabando el contraste (std) en cada paso
`c_0, c_1, ..., c_50`.

- **Método A (compuesto):** tasa por-paso equivalente que compondría el borrado observado
  en k pasos: `D_k^A = 1 - (c_k / c_0)^(1/k)`. En k=1 esto es idéntico a `medir_D()` del
  código base (se verifica numéricamente: |D_1^A − medir_D| < 1e-9).
- **Método B (tasa local en k):** la tasa de UN paso medida arrancando desde el estado ya
  difundido en el paso k−1: `D_k^B = 1 - c_k / c_{k-1}`. Muestra si la tasa instantánea
  cambia con el tiempo (predicción física: debe CAER si los modos lentos dominan tras
  varios pasos).
- Se reportan ambos; si difieren cualitativamente, se dice explícitamente (no se oculta).

D_k reportada para la redefinición de r es **Método A** (la relevante para "cuánto se
borró en total hasta la escala k", que es lo que pide el experimento). Método B es el
cross-check físico de por qué D_k^A se mueve.

## 3. Grid y barrido

**Primario:** N=200, eps=1e-3 (misma referencia de calibración que usa
`cs074_rcruz.py` modo "produccion"), semillas=16 (seeds 0..15, RNG offset propio para no
colisionar con otros agentes: `seed_dyn = 7000+s`, `seed_D = 8000+s`).

- `pasos` (longitud de la corrida acoplada): calibrado con `medir_pasos_lavado(N, eps,
  semillas)` importado del código base (idéntico criterio P_LAVADO=0.05,
  MARGEN_LAVADO=1.15).
- Grid de r_nominal (eje base, usa D_1 igual que el código original):
  `r_nominal ∈ {0.0} ∪ logspace(-2, 2, 24)` → 25 puntos, cruzando r=1 con densidad fina
  (≥8 puntos entre 0.3 y 3).
- `H = min(r_nominal · D_1, 1.0)` (idéntico a `cs074_rcruz`). D_1 = media de `medir_D`
  sobre las 16 semillas.
- Por cada punto de H: `corrida(real)` y `corrida(null)` (permutación final, del código
  base) para cada semilla → P_real(H), P_null(H), dispersión entre semillas.

**Robustez de eps:** se repite el barrido completo con eps=0.1 (mismo N=200, mismas 16
semillas) — segundo valor para confirmar que el corrimiento de r* no es artefacto de un
eps particular.

**Robustez de N (cross-check T7):** grid reducido en N=400, eps=1e-3, semillas=12,
r_nominal = {0.0} ∪ logspace(-2,2,12) (12 puntos) — para ver si la dirección/magnitud del
corrimiento se repite a otro N.

**D_k:** medida para k∈{1,2,5,10,50} en cada combinación (N,eps) usada arriba, sobre las
mismas semillas que esa combinación.

## 4. Definición de r* (congelada, NO se toca tras ver datos)

1. `P_floor` = P_real en el punto r_nominal=0 (H=0; este punto es invariante: H/D_k=0
   para cualquier D_k>0, así que su posición en el eje es r=0 bajo cualquier definición).
2. `P_ceiling` = media de P_real en los 3 puntos de H más grande del grid (régimen
   congelado). Nota: el ORDEN de los puntos por H es el mismo bajo cualquier D_k (D_k>0
   siempre), así que "los 3 de H más grande" es el mismo trío de puntos físicos para
   cualquier definición de r — solo cambia dónde caen en el eje x.
3. `objetivo = (P_floor + P_ceiling) / 2` (punto de medio-ascenso, estándar para
   localizar una transición sigmoide).
4. Se recorren los puntos con r>0 ordenados ascendentemente por H (≡ por r_k, cualquier
   k). Se busca el primer par de puntos consecutivos donde P_real cruza `objetivo`.
   Interpolación LINEAL en `log10(r_k)` entre esos dos puntos → `r*_k`.
5. Si P_real nunca cruza `objetivo` (curva plana, o floor≈ceiling, o el NULL no muerde),
   se reporta `r*_k = NaN` con la razón exacta — **no se fuerza un número**.
6. Esto se repite para D_1 (nominal, nuevo apodo `r*_1`, equivalente al r* "de F2-1"
   metodológicamente — ver nota de limitación abajo) y para D_2, D_5, D_10, D_50.

## 5. NULL

Ninguno propio — este experimento es verificación de instrumento (medición de D), tal
como indica el documento autoritativo ("NULL: —"). Se reportan P_null(H) y z-score
(idénticos al código base) como contexto, no como criterio de PASS de F2-2.

## 6. Criterio de PASS pre-registrado (tres lecturas posibles, ninguna se privilegia)

- **A. El cruce se mueve hacia r≈1:** r*_k (k grande, p.ej. 50) cae más cerca de 1 que
  r*_1, de forma monótona o casi monótona con k.
- **B. El cruce NO se mueve (o se aleja):** r*_k ≈ r*_1 dentro de la dispersión entre
  semillas, o se mueve en dirección contraria a 1.
- **C. Sin cruce definible:** floor≈ceiling o el NULL no separa — se reporta como tal.
Se reporta la curva r*_k vs k tal como salga, sin forzar hacia A.

## 7. Limitación explícita (declarada ANTES de correr)

El documento pide cotejar contra "el r* de F2-1". F2-1 es ejecutado por otro agente en
paralelo; al momento de este pre-registro **no existe archivo de resultado de F2-1 en
disco** (`find` no encontró `F2_1_*` ni `F2-1_*` bajo `BATERIA_FUNDAMENTOS/`). Por lo
tanto la "r* nominal" contra la que se compara aquí es la propia (misma metodología: D de
un paso, grid fino cruzando 1, física idéntica de `cs074_rcruz.py`, N=200) — **no** el
número literal que F2-1 vaya a publicar. Si CS necesita la comparación literal contra el
archivo de F2-1, debe re-cotejarse después de que ambos existan. Esto se declara aquí para
que no se lea como que se evadió la verificación cruzada pedida.

## 8. Verificación en disco (para auditor que no escribió el código)

- Este protocolo (`PROTOCOLO_F2-2_PREREGISTRO.md`).
- Script único: `F2_2_D_multipaso_engine.py` (importa cs074_rcruz.py, no lo edita).
- JSON crudo por combinación: `F2_2_resultado_<tag>.json` (todas las filas P_real/P_null
  por H, D_k por escala y método, r*_k por escala, seeds, tiempos).
- Log de stderr con timestamps de inicio/fin de cada bloque.
