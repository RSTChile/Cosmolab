# REAL: semillas adicionales — atacando el piso de p=1/9 (CS073)

**Estado: números reportados, sin veredicto.** Este documento no cierra ni confirma nada sobre CS073 —
esa lectura es del director del proyecto. Sólo describe qué se hizo y qué salió.

## El problema exacto que se atacó

Con **1 sola corrida REAL** y 8 NULL, el test de permutación exacto a nivel de corrida
(`cs078_kappaV_permutacion.py`, `null1_bateria_comparar.py`) sólo puede repartir la etiqueta "REAL"
entre 9 unidades → C(9,1)=9 asignaciones posibles bajo H0 → **p nunca puede bajar de 1/9≈0.1111**, sin
importar cuán grande sea la diferencia observada. No es un problema de potencia que "más NULL" resuelva:
depende estructuralmente de cuántas etiquetas REAL hay para repartir. La única salida es más REAL.

## Paso 1 — Investigación: qué parámetro de semilla es legítimo variar

Se leyeron `p_semilla_causal.py`, `cs073_cierre_holistico.py`, `fase1_traducir_a_phantom.py`,
`campo_velocidad_turbulento.py`, `scratch_generar_ic_velocidades.py`, `null1_generar_ic.py` y los
scripts de la batería NULL-1 (`null1_bateria_generar.py`, `null1_bateria_correr.py`,
`null1_bateria_comparar.py`). El pipeline REAL tiene **tres** semillas estocásticas distintas, con roles
muy diferentes:

| Semilla | Dónde | Qué controla | ¿Se varió aquí? |
|---|---|---|---|
| `seed_ejes` (default 2000, fijo) | `malla_causal_atomos` (`p_semilla_causal.py`) | Proyección de la densidad #23 en D ejes → determina la TOPOLOGÍA del grafo causal (quién quedó cerca de quién) | **NO** — se dejó fija; es "la única coherencia relacional que el motor realmente produce" y variarla cambiaría el objeto mismo que CS073 mide, no sólo su despliegue |
| `seed_null` (`None`=REAL, entero=NULL) | `traducir_pool` | Si se barajan las ARISTAS del grafo (double-edge-swap) | Fijo en `None` en las 6 corridas REAL — es el interruptor REAL/NULL, no una fuente de diversidad REAL |
| **`seed_layout`** (default 12345) | `layout_resortes` (Fruchterman-Reingold) | Posición inicial aleatoria + relajación física de 100 iteraciones **sobre la misma malla causal** | **SÍ — éste es el que se varió** |
| `TURB_SEED` (42, fijo en el proyecto) | `campo_velocidad_turbulento.factory` | Campo de velocidad turbulento inicial (Mach=3) | NO — se mantuvo igual a REAL/NULL/NULL-1 originales (verificado bit a bit, ver abajo) |

**`seed_layout` es el parámetro correcto**, y no es una elección nueva de esta tarea: el propio docstring
de `traducir_pool` (`fase1_traducir_a_phantom.py`, ~línea 75) ya lo señala explícitamente: *"seed_layout
varía la realización estocástica del layout de resortes (las '>=5 semillas' de la Fase 2)"*. Es decir,
el diseño congelado del arco ya prevía este mecanismo para exactamente este propósito.

**Por qué es una realización REAL genuina y no diversidad artificial:** con `seed_ejes=2000` y
`seed_null=None` fijos, la malla causal (el grafo de "quién nació relacionado con quién") es
**idéntica** entre las 6 semillas — no se toca. Lo único que cambia es la posición inicial aleatoria del
algoritmo de relajación de Fruchterman-Reingold sobre ESA misma malla: exactamente análogo a correr la
misma dinámica física desde una condición inicial de ruido térmico distinta — mismo mecanismo, mismo
grafo, distinta trayectoria concreta de relajación → distinta realización espacial de la misma
coherencia relacional. No hay barajado de aristas (eso es la operación NULL, nunca invocada aquí).

**Infraestructura ya existente revisada y descartada:** `run_vel_turb_N2000_real` /
`run_vel_hered_N2000_real` (en `/Users/alexis/phantom_cs073/`) sí existen, pero usan **el mismo
`seed_layout=12345`** que la corrida original — sólo cambian el *brazo* de campo de velocidad
(turbulencia importada vs. heredada de la malla), no la malla ni su layout. No son realizaciones REAL
independientes en el sentido que pide este ataque, así que no se reutilizaron; se generaron 5 IC nuevas.

**Verificación de consistencia con el pipeline original** (antes de generar nada): se confirmó que
`TURB_SEED=42` reproduce exactamente el v_rms ya escrito en `bateria_n2000/ic_real/cosmogenesis_ic.txt`
(1.6431676725… en ambos, 10 decimales) — la misma turbulencia que usaron tanto `ic_real` como los 8
`ic_null1..8` originales y los 8 NULL-1.

## Paso 2 — Generación de semillas REAL adicionales

Script: `real_extra_generar_ic.py`. Reutiliza el pool de bariones ya extraído y guardado en
`bateria_n2000/masa_bar.npy` / `dens_bar.npy` (sólo lectura — el motor basal es determinista y no
depende de `seed_null`/`seed_layout`, así que no hace falta re-correrlo). Genera 5 condiciones REAL
nuevas con `seed_layout ∈ {301, 302, 303, 304, 305}` (consecutivas a las convenciones ya usadas en el
arco: 101-103 piloto NULL-1, 201-208 NULL-1 batería), manteniendo `seed_null=None` y el mismo
`TURB_SEED=42`. Tiempo total de generación: **184.9 s** (≈37 s por semilla).

## Paso 3 — Corridas Phantom

Script: `real_extra_correr.py`. Misma configuración física exacta que `bateria_n2000/ic_real/cosmog.in`
(copiada de ahí, sólo lectura): `icreate_sinks=1`, `rho_crit_cgs=1000`, `r_crit=0.6`, `h_acc=0.3`,
`h_soft_sinkgas=0`, `r_merge_uncond=0`, `r_merge_cond=0`, `tmax=0.500`, `dtmax=0.001`. Binarios
`phantomsetup_cosmogenesis_backup` / `phantom_cosmogenesis_backup` (los mismos que usó NULL-1, sin APR,
para no introducir un confound frente a la metodología ya validada). Salidas en
`/Users/alexis/phantom_cs073/bateria_real_extra_n2000/ic_real_s{301..305}/` — `bateria_n2000/` no se
tocó.

Las 5 corridas terminaron **limpias, completas (500 dumps, hasta `cosmog_00500`), sin NaN/error/abort**,
y sorprendentemente rápido: **~8 s cada una** (frente a los ~11 minutos que tardó la corrida REAL
original) — muy por debajo de la salvaguarda de 40 minutos (tiempo total real usado: generación de IC +
Phantom ≈ 3.8 minutos). Las 8 corridas NULL-1 ya completadas también habían sido rápidas (~4 s, sin
cruzar a colapso) — el colapso de estas 5 semillas nuevas sí ocurrió y sí formó sumideros, sólo que más
rápido que la corrida original, sin que se identificara ninguna anomalía en los logs.

### Resultado de sumideros por semilla

| Corrida | `seed_layout` | masa total en sumideros | nº sumideros |
|---|---|---|---|
| REAL original (`bateria_n2000/ic_real`) | 12345 | 2124.4 | 8 |
| REAL_s301 | 301 | 2209.0 | 8 |
| REAL_s302 | 302 | 2209.0 | 8 |
| REAL_s303 | 303 | 2293.6 | 8 |
| REAL_s304 | 304 | 2049.2 | 8 |
| REAL_s305 | 305 | 2293.6 | 8 |

**Variabilidad entre semillas REAL (n=6):** media=2196.47, DE=95.98 (CV≈4.4%), min=2049.2, max=2293.6.
Las 6 corridas forman consistentemente 8 sumideros cada una y ocupan una banda angosta de masa total,
muy por encima de cualquier NULL — no se observó ninguna semilla REAL que se comportara como outlier o
que se acercara al rango NULL. Esto no se podía saber con n_REAL=1 (no había con qué comparar la corrida
original).

## Paso 4 — Test de permutación recalculado

Script: `real_extra_comparar.py`. Generaliza el mismo test exacto de
`cs078_kappaV_permutacion.py`/`null1_bateria_comparar.py` (estadístico = media(REAL) − media(NULL), H1
pre-registrada de una cola REAL>NULL) al caso n_REAL=6: ahora hay C(14,6)=**3003** asignaciones posibles
bajo H0 (no C(9,1)=9), así que el piso teórico de p baja de **1/9=0.1111 a 1/3003≈0.000333** — un factor
>300, sin haber corrido ni un solo NULL adicional.

**REAL (n=6) vs. 8 NULL originales (`bateria_n2000`):**
- media NULL = 720.27 (DE=28.84, rango 676.8–770.8)
- estadístico observado (media_REAL − media_NULL) = 1476.19
- rank de la asignación observada = **1 de 3003**
- **p (una cola) = 0.000333** (el piso exacto alcanzable con este n)

**REAL (n=6) vs. 8 NULL-1 (`bateria_null1_n2000`, mismo radio que REAL, ángulo isótropo aleatorio):**
- las 8 corridas NULL-1 formaron 0 sumideros cada una (masa=0.0, ya reportado en el blindaje previo)
- estadístico observado = 2196.47
- rank de la asignación observada = **1 de 3003**
- **p (una cola) = 0.000333** (mismo piso; NULL-1 es categóricamente cero en las 8, así que cualquier
  conjunto REAL con masa>0 alcanza el extremo)

## Resumen de lo que cambió y lo que no

- El piso matemático del test de permutación bajó de 0.1111 a 0.000333 (factor >300) únicamente por
  tener n_REAL=6 en vez de n_REAL=1 — el cuello de botella que identificó Alexis.
- Con las 5 semillas nuevas, REAL sigue separándose limpio de ambos NULL (original y NULL-1): las 6
  corridas REAL caen en 2049–2294 de masa en sumideros; NULL original en 677–771; NULL-1 en 0 siempre.
  No apareció ninguna semilla REAL "sorpresa" que se acercara al rango NULL.
- La variabilidad genuina entre semillas REAL (DE≈96, CV≈4.4%) es ahora visible por primera vez, y es
  pequeña comparada con la separación REAL–NULL (>15 DE de NULL original).
- Presupuesto de tiempo: muy por debajo del límite de 40 minutos (≈3.8 min de cómputo real en total,
  generación de IC + 5 corridas Phantom).

## Archivos generados

- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/real_extra_generar_ic.py` — genera las 5 IC REAL
  nuevas (seed_layout 301–305).
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/real_extra_correr.py` — corre Phantom sobre esas 5
  IC con la configuración física exacta de `bateria_n2000/ic_real/cosmog.in`.
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/real_extra_comparar.py` — test de permutación exacto
  generalizado a n_REAL>1, contra NULL original y contra NULL-1.
- `/Users/alexis/phantom_cs073/bateria_real_extra_n2000/ic_real_s{301,302,303,304,305}/` — salidas de
  Phantom (condición inicial, dumps, `.sink`, logs).
- `bateria_n2000/` y `bateria_null1_n2000/` no se modificaron (sólo lectura).
