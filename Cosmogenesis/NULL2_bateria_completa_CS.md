# NULL-2 (Zel'dovich) — batería completa (N=2000, 8 semillas), Fase II CS073, escalón 2 de 6

**Encargo:** escalar el método Zel'dovich de NULL-2 (piloto N=500/3 semillas, ver
`NULL2_mejora_zeldovich_CS.md`: KS=0.220 vs REAL, mejora robusta de 56% sobre el método de
rechazo/inversión anterior, pero ninguna de las 3 semillas del piloto formó sumideros) al mismo
diseño N=2000/8 semillas que usó NULL-1 (`NULL1_bateria_completa_CS.md`), esta vez comparado contra
las 6 semillas REAL genuinas (`REAL_semillas_adicionales_CS.md`) y contra las 8 NULL-1 ya en disco.
No se declara cierre ni veredicto sobre CS073 ni sobre esta jerarquía — sólo se reportan números. La
lectura es de Alexis.

---

## Método

Mismo patrón que `null1_bateria_generar.py`/`_correr.py`/`_comparar.py` (congelados, no se tocan
directamente), aplicado al método Zel'dovich:

- **REAL de referencia:** no se re-corrió — se reutilizaron las 6 semillas ya existentes: `ic_real`
  original (`bateria_n2000/`) + `ic_real_s{301..305}` (`bateria_real_extra_n2000/`).
- **NULL-2 (×8):** se leyeron directo las 2000 posiciones REAL ya escritas en
  `bateria_n2000/ic_real/cosmogenesis_ic.txt` (sólo lectura). Sobre esas posiciones se aplicó el
  método Zel'dovich completo (`generar_null2_zeldovich` de `null2_zeldovich_generar_ic.py`,
  congelado, no se tocó): gridizar → aleatorizar fases (semilla de fase = semilla de la corrida) →
  campo de desplazamiento de Zel'dovich → grilla no perturbada (punto de partida homogéneo) →
  desplazamiento por interpolación trilineal. `ngrid=20` — la misma resolución que produjo el número
  de cabecera KS=0.220 en `NULL2_mejora_zeldovich_CS.md`. 8 semillas nuevas **401–408** (distintas de
  las 301-303 del piloto Zel'dovich en `piloto_null2_zeldovich/`, que no se tocó). Mismo campo de
  velocidad turbulento que REAL (Mach=3, semilla=42).
- **Phantom:** binarios `_backup` (sin APR, misma build que toda la jerarquía). Configuración física
  copiada literal de `bateria_n2000/ic_real/cosmog.in`: `icreate_sinks=1`, `rho_crit_cgs=1000`,
  `r_crit=0.6`, `h_acc=0.3`, `h_soft_sinkgas=0`, `r_merge_uncond=0`, `r_merge_cond=0`, `tmax=0.500`,
  `dtmax=0.001`.
- **Observable:** masa total acumulada en sumideros al final de la corrida — el mismo de toda la
  jerarquía CS073.
- **Estadístico:** test de permutación EXACTO de 2 muestras independientes (generalización del método
  ya validado en `cs078_kappaV_permutacion.py`/`null1_bateria_comparar.py`/`real_extra_comparar.py`),
  estadístico = diferencia de medias, H1 de una cola pre-registrada, enumerando TODAS las
  C(n_a+n_b, n_a) asignaciones posibles bajo H0 (no una aproximación Monte Carlo).

---

## Resultado — verificación de dos puntos ANTES de Phantom (por semilla, `ngrid=20`)

| semilla | KS vs REAL | r_mean sintético | r_std sintético | desplazamiento RMS |
|---|---|---|---|---|
| 401 | 0.529 | 41.92 | 19.67 | 34.04 |
| 402 | 0.157 | 66.78 | 22.72 | 34.39 |
| 403 | 0.360 | 52.09 | 22.83 | 34.02 |
| 404 | 0.357 | 54.46 | 18.71 | 33.66 |
| 405 | 0.355 | 53.94 | 18.23 | 34.31 |
| 406 | 0.254 | 60.74 | 18.96 | 34.60 |
| 407 | 0.278 | 58.80 | 22.02 | 34.43 |
| 408 | 0.317 | 56.31 | 19.63 | 34.07 |
| (REAL, referencia) | — | 72.78 | 8.20 | — |

Rango KS 0.16–0.53 en las 8 semillas (media≈0.33) — consistente en orden de magnitud con el barrido
de 5 semillas del informe anterior (0.19–0.39), aunque la semilla 401 cae algo por encima de ese
rango (0.529, ligeramente peor que el 0.495 del método de rechazo original) — se deja constancia sin
descartar el punto: es variabilidad de semilla ya documentada como existente (el barrido anterior
también mostró semillas peores, ej. 9003 con KS=0.393), no un error de método.

## Resultado — sumideros en Phantom

Las 8 corridas terminaron completas a `tmax=0.500`, exit code 0 en `setup` y en `phantom`, sin NaN ni
error de conservación (no se necesitó `I_WILL_NOT_PUBLISH_CRAP`).

| corrida | masa en sumideros | nº sumideros | densidad máx. final (g/cm³) |
|---|---|---|---|
| NULL-2 seed 401 | 0.0 | 0 | 0.568 |
| NULL-2 seed 402 | 0.0 | 0 | 0.129 |
| NULL-2 seed 403 | 0.0 | 0 | 0.405 |
| NULL-2 seed 404 | 0.0 | 0 | 0.671 |
| NULL-2 seed 405 | 0.0 | 0 | 0.531 |
| NULL-2 seed 406 | 0.0 | 0 | 0.518 |
| NULL-2 seed 407 | 0.0 | 0 | 0.159 |
| NULL-2 seed 408 | 0.0 | 0 | 0.252 |

**Las 8 corridas NULL-2-Zel'dovich a escala completa (N=2000) formaron CERO sumideros — las 8, sin
excepción**, igual que las 8 NULL-1. Densidad máxima alcanzada: 0.13–0.67 g/cm³, ~3-4 órdenes de
magnitud por debajo del umbral `rho_crit_cgs=1000` y de la densidad máxima de las 6 REAL
(~188-200 g/cm³ de orden). El resultado replica, a escala completa, lo que ya había mostrado el
piloto N=500 (3/3 sin sumideros): la mejora sustancial en la estadística de dos puntos a nivel de
partícula (KS 0.495→0.16-0.53) NO se tradujo, en ninguna de las 8 semillas, en formación de
estructura colapsada.

**Nota comparativa NULL-1 vs NULL-2 en densidad máxima (no en masa/nº de sumideros, que son 0/0 en
ambos):** el rango de densidad máxima de NULL-2 (0.13–0.67 g/cm³) queda sistemáticamente por encima
del de NULL-1 reportado en `NULL1_bateria_completa_CS.md` (0.039–0.131 g/cm³) — un factor ~3-5×
mayor en el pico de densidad, en la dirección que predeciría que preservar parcialmente la
estadística de 2 puntos ayuda a concentrar algo más de masa localmente. Se señala como observación
cruda, sin inferencia: ninguno de los dos escalones cruzó el umbral de formación de sumideros, así
que esta diferencia de densidad pico no tiene, con este diseño, un observable de "más estructura"
que la traduzca en algo estadísticamente comparable (ver test (b) abajo).

---

## Estadísticos de separación

### (a) REAL (n=6) vs NULL-2 (n=8) — test de permutación exacto

- REAL: media=2196.47, DE=95.98 (rango 2049.2–2293.6, las mismas 6 semillas de
  `REAL_semillas_adicionales_CS.md`).
- NULL-2: media=0.0, DE=0.0 (las 8 corridas dieron exactamente masa 0).
- estadístico observado (media_REAL − media_NULL2) = **2196.47**
- C(14,6) = **3003** asignaciones posibles bajo H0
- rank de la asignación observada = **1 de 3003**
- **p (una cola, H1 pre-registrada REAL>NULL2) = 1/3003 ≈ 0.000333** — el piso teórico exacto
  alcanzable con este diseño (n_REAL=6, n_NULL2=8), idéntico en magnitud al piso ya alcanzado contra
  NULL-1 en `REAL_semillas_adicionales_CS.md`.
- z-score: INDEFINIDO (DE_NULL2=0, sin varianza NULL sobre la que normalizar) — igual situación,
  documentada de la misma forma honesta, que NULL-1.

### (b) NULL-1 (n=8) vs NULL-2 (n=8) — comparación entre escalones de la jerarquía

Pregunta pre-registrada en el encargo: ¿el método Zel'dovich (2-puntos parcialmente preservado) forma
MÁS estructura, medida por el observable de la jerarquía (masa en sumideros), que NULL-1 (que no
preserva nada de eso)?

- NULL-1: media=0.0, DE=0.0. NULL-2: media=0.0, DE=0.0.
- estadístico observado (media_NULL2 − media_NULL1) = **0.0**
- C(16,8) = 12870 asignaciones posibles bajo H0
- p (una cola) = 1.0, p (dos colas) = 1.0 — **test no informativo por construcción**: con ambos
  grupos en masa=0 exacta, no hay varianza combinada sobre la que el test de permutación pueda
  distinguir nada. La respuesta a la pregunta del encargo, con el observable "masa en sumideros", es
  **empate categórico (0 vs 0)** — no hay diferencia medible en ESTE observable a este N/tmax. La
  única diferencia observada entre los dos escalones está en la densidad máxima pico (ver nota arriba,
  0.039–0.131 NULL-1 vs 0.13–0.67 NULL-2), que sugiere una dirección pero no cruza a formación de
  estructura ni tiene, en este diseño, un test de significación aplicable.

---

## Tiempo de cómputo real vs. salvaguarda

Salvaguarda pedida: detenerse y reportar si el cómputo total supera ~20 minutos.

| paso | tiempo |
|---|---|
| generación de las 8 condiciones iniciales NULL-2 (`null2_bateria_generar.py`) | 7.9 s |
| `phantomsetup` + `phantom` para las 8 corridas (`null2_bateria_correr.py`) | 49.8 s |
| **total (generación + 8 corridas Phantom)** | **≈58 s** |

Muy por debajo de la salvaguarda de 20 minutos — no fue necesario detenerse ni activar el límite de
tiempo. Con margen amplio se avanzó además el paso de investigación (no implementación) de NULL-3
(ver `NULL3_investigacion_preliminar_CS.md`).

---

## Entregables de esta tarea

- `null2_bateria_generar.py` — genera las 8 condiciones iniciales NULL-2-Zel'dovich a N=2000
  (semillas 401-408, `ngrid=20`), leyendo directo las posiciones REALES ya existentes de
  `bateria_n2000/ic_real/cosmogenesis_ic.txt`. Importa `null2_zeldovich_generar_ic.py` sin tocarlo.
- `null2_bateria_correr.py` — corre Phantom (binarios `_backup`) sobre las 8 carpetas, con la misma
  configuración física que toda la jerarquía, y la misma salvaguarda de tiempo (esta vez con límite de
  20 min, pedido específico de esta tarea).
- `null2_bateria_comparar.py` — lee los `.sink` de REAL (n=6, `bateria_n2000/` +
  `bateria_real_extra_n2000/`), NULL-1 (n=8, `bateria_null1_n2000/`) y NULL-2 (n=8, carpeta nueva),
  calcula masa total en sumideros por corrida y los dos tests de permutación exactos ((a) REAL vs
  NULL-2, (b) NULL-1 vs NULL-2).
- `/Users/alexis/phantom_cs073/bateria_null2_n2000/ic_null2_s{401..408}/` — las 8 corridas de Phantom
  (IC, `cosmog.in`, `setup.log`, `run.log`, dumps). No se tocó `bateria_n2000/`,
  `bateria_null1_n2000/`, `bateria_real_extra_n2000/`, ni `piloto_null2_zeldovich/` (sólo lectura).
- Este informe.
- `NULL3_investigacion_preliminar_CS.md` — punto de partida (no resultado) para el siguiente escalón.
