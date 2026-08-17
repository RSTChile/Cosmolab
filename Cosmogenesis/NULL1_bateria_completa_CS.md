# NULL-1 — batería completa (N=2000, 8 semillas), Fase II CS073, escalón 1 de 6

**Encargo:** escalar el piloto NULL-1 (N=500, 3 semillas, ver `NULL1_piloto_distribucion_radial_CS.md`)
al mismo diseño que usó CS073 originalmente: N=2000 partículas, 8 semillas NULL-1, comparado contra la
corrida `ic_real` ya existente de `bateria_n2000/` (z=48.69 en la batería original de CS073, contra los
NULL1-8 que la auditoría posterior encontró que NO son un control aislado válido — ver piloto).

No se declara cierre ni veredicto sobre CS073 ni sobre esta jerarquía de controles — sólo se reportan
números. La lectura es de Alexis.

---

## Método

Igual que el piloto, a escala completa:

- **REAL:** no se re-corrió — se usó tal cual la corrida ya existente
  `/Users/alexis/phantom_cs073/bateria_n2000/ic_real/` (N=2000, `seed_layout=12345`).
- **NULL-1 (×8):** mismo multiconjunto EXACTO de radios que REAL (r_i = |pos_i − COM|, leído directo
  de `bateria_n2000/ic_real/cosmogenesis_ic.txt`, sin volver a correr `traducir_pool`), ángulo de cada
  partícula reasignado a un vector aleatorio isótropo (Marsaglia), 8 semillas angulares nuevas
  **201–208** (distintas de las 101-103 del piloto). Mismo campo de velocidad turbulento que REAL
  (Mach=3, semilla=42). Generado por `null1_bateria_generar.py` (importa `leer_ic_txt`/`generar_null1`
  de `null1_generar_ic.py`, sin tocarlo).
- **Pool de bariones:** reutilizado de `bateria_n2000/masa_bar.npy` / `dens_bar.npy` (no se re-extrajo;
  de hecho ni siquiera hizo falta cargarlos: las posiciones REALES ya escritas en
  `ic_real/cosmogenesis_ic.txt` bastan para derivar los radios).
- **Phantom:** binario `phantom_cosmogenesis_backup` / `phantomsetup_cosmogenesis_backup` (build previa
  a APR, la misma que usó `bateria_n2000` — el binario `phantom`/`phantomsetup` actual añade
  refinamiento adaptativo por defecto, confound frente al método original). Configuración física
  copiada literal de `bateria_n2000/ic_real/cosmog.in`: `icreate_sinks=1`, `rho_crit_cgs=1000`,
  `r_crit=0.6`, `h_acc=0.3`, `h_soft_sinkgas=0`, `r_merge_uncond=0`, `r_merge_cond=0`, `tmax=0.500`,
  `dtmax=0.001` (el resto de `cosmog.in` queda en su valor por defecto de `phantomsetup`, idéntico entre
  las 8 corridas). Orquestado por `null1_bateria_correr.py`.
- **Observable:** masa total acumulada en sumideros al final de la corrida — el mismo observable de
  CS073 original (`RESULTADO_bateria_ignicion_sumideros_N2000_CS.md`).
- **Estadístico:** test de permutación EXACTO a nivel de CORRIDA (9 unidades: 1 REAL + 8 NULL-1,
  C(9,1)=9 asignaciones posibles bajo H0), método idéntico al validado en
  `cs078_kappaV_permutacion.py` (sección 3) — la unidad de permutación es la corrida completa, no el
  sumidero individual (evita pseudoreplicación). Reimplementado sobre el observable de masa en
  `null1_bateria_comparar.py`. También se reporta el z-score original de CS073, con el caveat explícito
  si la desviación estándar de NULL es cero.

---

## Resultado

| corrida | masa en sumideros | nº sumideros |
|---|---|---|
| **REAL** (bateria_n2000, ya existente) | **2124.4** | 8 (263.2, 310.2, 300.8, 282.0, 216.2, 244.4, 225.6, 282.0) |
| NULL-1 seed 201 | 0.0 | 0 |
| NULL-1 seed 202 | 0.0 | 0 |
| NULL-1 seed 203 | 0.0 | 0 |
| NULL-1 seed 204 | 0.0 | 0 |
| NULL-1 seed 205 | 0.0 | 0 |
| NULL-1 seed 206 | 0.0 | 0 |
| NULL-1 seed 207 | 0.0 | 0 |
| NULL-1 seed 208 | 0.0 | 0 |

**Las 8 corridas NULL-1 a escala completa (N=2000) formaron CERO sumideros — las 8, sin excepción.**
Densidad máxima alcanzada al final: 0.039–0.131 g/cm³ en las 8 corridas (rango completo), frente al
umbral de creación de sumideros `rho_crit_cgs=1000` — es decir, ~4 órdenes de magnitud por debajo del
umbral, y también ~4 órdenes de magnitud por debajo de la densidad máxima que alcanzó REAL durante su
evolución. Esto es consistente y aún más categórico que el piloto a N=500 (donde las 3 semillas NULL-1
tampoco formaron sumideros, densidad máx. 0.028-0.042 g/cm³).

Las 9 corridas (1 REAL + 8 NULL-1) terminaron completas a `tmax=0.5` con exit code 0, sin ningún error
de conservación de energía/momento — no se necesitó `I_WILL_NOT_PUBLISH_CRAP`.

### Estadístico de separación

- **z-score:** INDEFINIDO. Las 8 corridas NULL-1 dieron exactamente el mismo valor (0.0, DE=0) —
  no hay varianza NULL sobre la que normalizar. No se rellena con un número inventado; se deja
  constancia de que el resultado es categórico (0/8 formaron sumidero), no un efecto al límite de la
  distribución NULL.
- **Test de permutación exacto a nivel de corrida (9 unidades, C(9,1)=9):** estadístico observado
  (masa_REAL − media(8 NULL-1)) = **2124.4**. REAL ocupa el rank 1 de 9 en la distribución nula exacta
  (es la asignación más extrema posible). **p (una cola, H1 pre-registrada REAL>NULL) = 1/9 = 0.1111**
  — el mínimo valor de p alcanzable con este diseño de n=9 unidades (mismo piso que menciona
  `cs078_kappaV_permutacion.py` para su propio test de corrida). El efecto es tan categórico como se
  puede medir con este número de semillas — el límite es el tamaño de muestra (n_null=8), no la
  fuerza de la separación.

---

## Tiempo de cómputo real vs. estimado

El piloto estimó, extrapolando de forma conservadora desde N=500 (usando el costo de REAL, más caro
por formar sumideros, como cota superior también para NULL-1 a N=2000): **~4-6 minutos** para las 8
corridas de Phantom a escala completa.

**Tiempo real medido:**

| paso | tiempo |
|---|---|
| generación de las 8 condiciones iniciales NULL-1 (`null1_bateria_generar.py`) | 0.19 s |
| `phantomsetup` + `phantom` para las 8 corridas NULL-1 (`null1_bateria_correr.py`) | **47.5 s** (por semilla, setup+run: 5.25s, 5.34s, 6.51s, 6.10s, 5.97s, 6.18s, 6.04s, 6.08s) |
| **total (generación + 8 corridas Phantom)** | **~48 s** |

Fue considerablemente MÁS rápido que la estimación conservadora del piloto (48s vs. 4-6 min
estimados) — la razón física es clara en retrospectiva: el costo dominante de la corrida REAL a
N=2000 (31.45s en la batería original) es la formación y crecimiento de sumideros (el árbol
gravitatorio se vuelve más profundo con masa concentrada, y hay que resolver la acreción); las 8
NULL-1 nunca cruzan ese régimen — permanecen como gas difuso durante toda la corrida, así que el costo
por partícula se mantiene bajo y constante. No se necesitó activar la salvaguarda de tiempo (límite
puesto en 35 min); el cómputo total quedó muy por debajo.

---

## Entregables de esta tarea

- `null1_bateria_generar.py` — genera las 8 condiciones iniciales NULL-1 a N=2000 (semillas 201-208),
  leyendo directo las posiciones REALES ya existentes en `bateria_n2000/ic_real/cosmogenesis_ic.txt`
  (sin re-correr `traducir_pool`). Importa `null1_generar_ic.py` sin tocarlo.
- `null1_bateria_correr.py` — corre Phantom (binarios `_backup`) sobre las 8 carpetas, edita
  `cosmog.in` para igualar los parámetros físicos de `bateria_n2000/ic_real/cosmog.in`, con
  salvaguarda de tiempo (se detiene y reporta si el cómputo acumulado supera ~35 min).
- `null1_bateria_comparar.py` — lee los `.sink` (REAL de `bateria_n2000/`, NULL-1 de la carpeta nueva),
  calcula masa total en sumideros por corrida y el test de permutación exacto a nivel de corrida
  (mismo método que `cs078_kappaV_permutacion.py` sección 3).
- `/Users/alexis/phantom_cs073/bateria_null1_n2000/ic_null1_s{1..8}/` — las 8 corridas de Phantom
  (IC, `cosmog.in`, `setup.log`, `run.log`, dumps). `bateria_n2000/` no se tocó (sólo lectura).
- Este informe.
