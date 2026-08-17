# NULL-3 (double-edge-swap con filtro geométrico de longitud) — Fase II CS073, escalón 3 de 6

**Encargo:** completar el escalón NULL-3 de la jerarquía de controles — un control que conserva "grado
de cada nodo y longitudes [de enlace]" y destruye "motivos, ciclos e historia", identificando el
"efecto de la topología de orden superior". Punto de partida: `NULL3_investigacion_preliminar_CS.md`
(diagnóstico SOLO de grafo, sin `layout_resortes` ni Phantom, KS(L,L_real)=0.0040). Esta tarea retoma
exactamente donde quedó ese informe: layout físico → verificación de perfil radial → piloto → batería
completa. **No se declara cierre ni veredicto sobre CS073 ni sobre la jerarquía — sólo se reportan
números. La lectura es de Alexis.**

---

## Método

- **NULL-3 (`barajar_aristas_preservando_longitud`, ya validada en el informe preliminar, importada tal
  cual):** double-edge-swap de Maslov-Sneppen sobre la malla causal REAL, que sólo acepta un intercambio
  si ambas aristas NUEVAS quedan dentro de `tol_relativa=0.2` (20%) de la longitud de las aristas VIEJAS
  que reemplazan (longitud medida sobre las posiciones REAL ya escritas en disco). Preserva el grado de
  cada nodo EXACTO por construcción.
- **Layout físico:** `layout_resortes` (Fruchterman-Reingold), la MISMA función que usa
  `fase1_traducir_a_phantom.traducir_pool` para REAL/NULL-1-grafo/NULL-2 — importada de
  `p_semilla_causal.py` sin modificarla, mismos parámetros (`iters=100`, `seed_layout=12345`), seguida
  de la MISMA dilatación isótropa estática (`Expansion`, `n_pasos_expansion=60`, `a_final=√60≈7.75`) que
  usa toda la jerarquía.
- **Módulo nuevo `null3_generar_ic.py`:** duplica la FORMA del pipeline de `traducir_pool` (malla →
  barajado → layout → expansión → velocidad → h uniforme → escritura ASCII), pero con una TERCERA
  operación de barajado (con filtro de longitud) que `traducir_pool` no tiene — no se editó ningún
  archivo congelado.
- **Phantom:** binarios `_backup` (sin APR, misma build que toda la jerarquía). Configuración física
  copiada literal de `bateria_n2000/ic_real/cosmog.in` (`icreate_sinks=1`, `rho_crit_cgs=1000`,
  `r_crit=0.6`, `h_acc=0.3`, `tmax=0.500`, `dtmax=0.001`).
- **Observable:** masa total acumulada en sumideros al final — el mismo de toda la jerarquía CS073.
- **Estadístico:** test de permutación EXACTO de 2 muestras independientes, H1 de una cola
  pre-registrada, enumerando TODAS las C(n_a+n_b, n_a) asignaciones posibles bajo H0.

---

## Paso 1 — Verificación del perfil radial post-layout (N=2000, seed=501, `null3_paso1_verificar_perfil_radial.py`)

| | r_mean | r_std | KS vs REAL | p |
|---|---|---|---|---|
| REAL (`bateria_n2000/ic_real`) | 72.780 | 8.205 | — | — |
| NULL-3 (grafo + layout, seed 501) | 73.262 | 7.808 | **0.0295** | 0.349 |
| *(referencia) NULL1-8 originales, swap SIN filtro de longitud* | ≈63.2–63.5 | ≈13.3–13.7 | <1e-113 (×8) | ≈0 |
| *(referencia) NULL-3-grafo puro, sólo distribución de longitudes de arista* | — | — | 0.0040 | 1.00 |

**El perfil radial post-`layout_resortes` de NULL-3 es prácticamente indistinguible de REAL** (KS=0.0295,
p=0.349 — no se rechaza la hipótesis nula de que provienen de la misma distribución; diff de r_mean =
+0.7%), en contraste marcado con los NULL1-8 originales (swap de aristas sin restricción de longitud),
que rompían el perfil radial completo (KS<1e-113 en las 8 comparaciones). Esto confirma, a nivel de
posiciones físicas ya relajadas — no sólo a nivel de grafo abstracto — la hipótesis de trabajo del
informe preliminar: restringir el double-edge-swap por longitud geométrica preserva la escala local de
conexión lo suficiente como para que 100 iteraciones de Fruchterman-Reingold reproduzcan una nube con la
misma forma global que REAL. El swap sí cambió topología real (346/49450 intentos aceptados, 0.7% de
tasa de aceptación, igual que en el informe preliminar) — no es un no-op.

---

## Paso 2 — Piloto (N=500, semillas 601–603, `null3_piloto_generar.py` + `null3_piloto_correr.py`)

Reutiliza el pool N=500 y la REAL de referencia ya en disco (`piloto_null1/real/`, sin re-correr).

| corrida | swap aceptado | r_mean | r_std | exit setup/run | masa en sumideros | nº sumideros |
|---|---|---|---|---|---|---|
| REAL (referencia, `piloto_null1/real`) | — | 45.498 | 5.703 | — | 282.0 | 4 |
| NULL-3 seed 601 | 87/12450 (0.7%) | 46.084 | 5.078 | 0/0 | 347.8 | 5 |
| NULL-3 seed 602 | 76/12450 (0.6%) | 45.805 | 5.449 | 0/0 | 235.0 | 3 |
| NULL-3 seed 603 | 77/12450 (0.6%) | 45.902 | 5.385 | 0/0 | 272.6 | 4 |

**Las 3 corridas terminaron limpias: exit code 0 en `phantomsetup` y en `phantom`, sin abortos de
conservación, sin necesidad de `I_WILL_NOT_PUBLISH_CRAP`.** Las 3 formaron sumideros (3, 4 y 5 —
comparable a los 4 de REAL a esta escala), con masas en el mismo orden de magnitud que REAL (235–348
vs 282). El piloto salió limpio en todos los criterios de la salvaguarda pedida (sin errores numéricos,
partículas físicamente sensatas) → se escaló directo a la batería completa, según autorización previa
de Alexis para este patrón.

---

## Paso 3 — Batería completa (N=2000, 8 semillas 501–508, `null3_bateria_generar.py` + `_correr.py` + `_comparar.py`)

Generación de las 8 IC: 284.5 s (≈35 s/semilla, dominado por las 100 iteraciones O(n²) de
`layout_resortes` a N=2000). Las 8 corridas Phantom: **55.4 s total, 8/8 exit code 0** en `setup` y en
`phantom`, sin abortos de conservación.

| corrida | masa en sumideros | nº sumideros |
|---|---|---|
| NULL-3 seed 501 | 2180.8 | 8 |
| NULL-3 seed 502 | 2190.2 | 8 |
| NULL-3 seed 503 | 2246.6 | 8 |
| NULL-3 seed 504 | 2199.6 | 8 |
| NULL-3 seed 505 | 2068.0 | 8 |
| NULL-3 seed 506 | 2227.8 | 8 |
| NULL-3 seed 507 | 2180.8 | 8 |
| NULL-3 seed 508 | 2199.6 | 8 |
| **media / DE** | **2186.68 / 53.16** | 8 (las 8) |

Comparación de referencia (REAL n=6, `bateria_n2000/`+`bateria_real_extra_n2000/`, ya en disco):
media=2196.47, DE=95.98, rango 2049.2–2293.6, 8 sumideros en las 6. NULL-1 (n=8) y NULL-2 (n=8), ambos
ya en disco: **0 sumideros, masa 0 en las 16 corridas combinadas** (`NULL1_bateria_completa_CS.md`,
`NULL2_bateria_completa_CS.md`).

**Las 8 corridas NULL-3 formaron sumideros en las 8, sin excepción, con masas totales (2068.0–2246.6)
que caen dentro del rango de REAL (2049.2–2293.6)** — a diferencia de NULL-1 y NULL-2, que no formaron
ni un solo sumidero en 16 corridas combinadas.

---

## Estadísticos de separación

### (a) REAL (n=6) vs NULL-3 (n=8) — test de permutación exacto
- estadístico observado (media_REAL − media_NULL3) = **9.79**
- C(14,6) = 3003 asignaciones posibles bajo H0
- rank de la asignación observada = 1265 de 3003
- **p (una cola, REAL>NULL-3) = 0.4212** — p (dos colas) = 0.8142
- z-score = (media_REAL − media_NULL3) / DE_NULL3 = **0.184**

No hay evidencia de separación entre REAL y NULL-3 con este observable y este diseño — la diferencia de
medias (9.79 sobre una escala de ~2190) es del orden de la variabilidad de semilla dentro de cada grupo.

### (b) NULL-3 (n=8) vs NULL-1 (n=8)
- estadístico observado (media_NULL3 − media_NULL1) = **2186.68**
- C(16,8) = 12870 asignaciones, rank = 1 de 12870
- **p (una cola, NULL-3>NULL-1) = 1/12870 ≈ 0.0000777** — el piso teórico exacto de este diseño.

### (c) NULL-3 (n=8) vs NULL-2 (n=8)
- estadístico observado (media_NULL3 − media_NULL2) = **2186.68** (idéntico al de (b), porque NULL-1 y
  NULL-2 dieron ambos masa 0 exacta)
- C(16,8) = 12870 asignaciones, rank = 1 de 12870
- **p (una cola, NULL-3>NULL-2) = 1/12870 ≈ 0.0000777** — el piso teórico exacto de este diseño.

---

## Lectura de los números (sin cerrar nada)

Con el observable y diseño de esta jerarquía: **NULL-3 (grado exacto + longitud de arista ≈ REAL,
motivos/ciclos/triángulos barajados) es indistinguible de REAL en formación de sumideros (p≈0.42,
z=0.18), mientras que NULL-1 (radio exacto, ángulo aleatorio) y NULL-2 (P(k) parcialmente preservado,
Zel'dovich) no formaron ni un solo sumidero en 16 corridas combinadas.** Dentro de esta jerarquía de 3
escalones ya corridos, el salto de "cero estructura" a "estructura ≈REAL" ocurre exactamente en el
escalón que preserva grado + longitud de arista (la escala LOCAL de conexión, "quién queda cerca de
quién a qué distancia"), no en los escalones que preservan sólo el perfil radial global (NULL-1) o sólo
el espectro de 2 puntos del campo (NULL-2). Esto es consistente con — pero no prueba por sí solo — que
la topología de orden superior específica que NULL-3 SÍ destruye (motivos/ciclos/triángulos, el
"quién-con-quién" preciso más allá de la escala local) no es lo que se necesita para formar sumideros a
esta escala/tiempo; lo que parece necesitarse es la escala local de conexión misma, que NULL-1 y NULL-2
destruyen y NULL-3 conserva. No se cuantificó en esta tarea un observable directo de motivos/ciclos (esa
tarea, ya señalada como pendiente en el informe preliminar, sigue pendiente) — la lectura de motivos
queda como inferencia indirecta del contraste de sumideros, no como medición directa.

Ningún veredicto sobre CS073 ni sobre esta jerarquía se declara aquí — los números de arriba son el
entregable; la interpretación final es de Alexis.

---

## Tiempo de cómputo real vs. salvaguarda

Salvaguarda pedida: ~50-60 min totales para esta tarea (Paso 1 + piloto + posible batería completa).

| paso | tiempo |
|---|---|
| Paso 1 (verificación perfil radial, N=2000, 1 semilla) | 34.0 s |
| Piloto: generación 3 IC (N=500) | 118.0 s (incluye 108.8 s de re-extracción determinista del pool) |
| Piloto: Phantom 3 corridas | 12.3 s |
| Batería: generación 8 IC (N=2000) | 284.5 s |
| Batería: Phantom 8 corridas | 55.4 s |
| **total cómputo** | **≈504 s ≈ 8.4 min** |

Muy por debajo de la salvaguarda de 50-60 min — no fue necesario detenerse en ningún punto.

---

## Entregables de esta tarea

- `null3_generar_ic.py` — módulo generador NULL-3 (malla causal → swap con filtro de longitud → layout
  de resortes → expansión estática → velocidad → escritura ASCII), reutiliza
  `barajar_aristas_preservando_longitud` de `null3_investigacion_preliminar.py` (no reescrita) y las
  piezas congeladas de `p_semilla_causal.py`/`cs073_cierre_holistico.py`/`fase1_traducir_a_phantom.py`
  (sólo importadas).
- `null3_paso1_verificar_perfil_radial.py` — Paso 1: perfil radial NULL-3 (N=2000, seed 501) vs REAL.
- `null3_piloto_generar.py` / `null3_piloto_correr.py` — Paso 2: piloto N=500, semillas 601-603, en
  `/Users/alexis/phantom_cs073/piloto_null3/`.
- `null3_bateria_generar.py` / `null3_bateria_correr.py` / `null3_bateria_comparar.py` — Paso 3: batería
  completa N=2000, semillas 501-508, en `/Users/alexis/phantom_cs073/bateria_null3_n2000/`.
- `/Users/alexis/phantom_cs073/piloto_null3/` y `/Users/alexis/phantom_cs073/bateria_null3_n2000/` —
  carpetas nuevas con las corridas de Phantom (IC, `cosmog.in`, `setup.log`, `run.log`, `.sink`, dumps).
  No se tocó ninguna carpeta de batería anterior (`bateria_n2000/`, `bateria_null1_n2000/`,
  `bateria_null2_n2000/`, `bateria_real_extra_n2000/`, `piloto_null1/`, `piloto_null2/`,
  `piloto_null2_zeldovich/`) ni ningún script congelado — sólo lectura/importación.
- Este informe.
