# INFRA: generador de masa total fija (corrige el confound N500 vs N2000)

**Fecha:** 7-ago-2026 · **Método:** código nuevo (no toca ningún generador congelado, sólo
importa/lee), corridas nuevas de Phantom, lectura de volcados binarios con `leer_volcado_phantom.py`
(congelado, no modificado). Presupuesto ~40-50 min.

**Antecedente:** `MISTERIO_N500_vs_N2000_CS.md` encontró que el generador de condiciones iniciales
`grafo_random_layout_generar_ic.py` (y el mismo patrón en `null1/2/3_generar_ic.py`,
`real_extra_generar_ic.py`) hace crecer el lado de la caja física con `lado = n**(1/3)`, manteniendo
`masa_por_particula ≈ 9.4` fija — así que a menor N no sólo baja la resolución (menos partículas
resolviendo el mismo sistema): también baja la masa TOTAL absoluta (verificado: 4700 a N=500 vs 18800 a
N=2000, razón exacta 0.2655 ≈ N500/N2000). Eso mezcla dos efectos: "menos vecinos por partícula"
(resolución) y "menos masa/caja física" (sistema distinto).

---

## Qué se cambió

Dos archivos nuevos, ninguno de los congelados tocado:

- **`grafo_random_layout_generar_ic_masa_fija.py`** — variante de
  `grafo_random_layout_generar_ic.py` (sólo importa de él `generar_grafo_erdos_renyi`, ya validado).
  Define `generar_control_random_masa_fija(n, dens_bar, n_aristas, seed_random, ...)` con dos cambios
  respecto al original:
  1. `lado` deja de ser `n**(1/3)` — se congela en `LADO_FIJO = 2000**(1/3) ≈ 12.5992` (el mismo valor
     que el generador original calcula para N=2000), para **cualquier** N.
  2. La masa por partícula deja de heredarse del pool físico (`masa_bar.mean()`, ~9.4 fijo,
     independiente de N) — se fuerza `masa_particula = MASA_TOTAL_OBJETIVO / n`, con
     `MASA_TOTAL_OBJETIVO = 18800` (la masa total real de `bateria_n2000/ic_real`, la misma que usa
     toda la jerarquía CS073). Así `n * masa_particula == 18800` por construcción, para cualquier N.
- **`grafo_random_masa_fija_generar.py`** — orquestador que genera 8 condiciones iniciales (N ∈
  {200, 500, 1000, 2000} × 2 semillas) en
  `/Users/alexis/phantom_cs073/bateria_grafo_random_masa_fija/ic_masaFija_N{n}_s{seed}/`.
- **`grafo_random_masa_fija_correr.py`** — corre `phantomsetup`/`phantom` sobre esas 8 carpetas, mismos
  parámetros físicos que toda la jerarquía (`icreate_sinks=1`, `rho_crit_cgs=1000`, `r_crit=0.6`,
  `h_acc=0.3`, `tmax=0.500`, `dtmax=0.001`).
- **`grafo_random_masa_fija_verificar.py`** — lee los volcados binarios (`leer_volcado_phantom.py`,
  no modificado) y reporta masa total a t=0 y sumideros al final de cada corrida.

**Atajo documentado (por presupuesto de tiempo):** en vez de re-extraer un pool físico nuevo
(`_extraer_bariones`) para cada N — una prueba mostró que a N=500 eso tarda 145.7s, y repetirlo para
4 valores de N hubiera consumido casi todo el presupuesto — `dens_bar` para N<2000 se obtiene
**submuestreando de forma determinista** el pool YA extraído de N=2000
(`bateria_n2000/dens_bar.npy`). Esto es válido para esta validación porque `dens_bar` en el generador
de masa fija sólo alimenta el campo de velocidad turbulento (no determina masa ni caja, que es lo que
se corrigió) — pero es una simplificación real, no un pool físico independiente por N. Si en el futuro
importa la identidad exacta del pool a cada N, hay que re-extraer con `_extraer_bariones` como hace
`grafo_random_piloto_generar.py`.

Sobre `h` (longitud de suavizado): el generador NO fija `h` — sólo escribe una semilla inicial uniforme
(`hfact=1.2`) que el solver grad-h nativo de Phantom reemplaza con el `h` de equilibrio real. Con caja y
masa total ahora fijas, cualquier diferencia de `h` entre N distintos que reporte Phantom refleja
SÓLO resolución (menos partículas repartiendo la misma masa en el mismo volumen), ya no mezclada con
menos masa total — no se midió `h` post-corrida en esta pasada por presupuesto de tiempo (quedó
priorizado el barrido de sumideros, pedido explícitamente como más importante en la salvaguarda).

---

## Paso 2 — verificación de masa total fija (leyendo el volcado BINARIO, no el ASCII propio)

`leer_volcado_phantom.py` sobre el dump inicial `cosmog_00000` (ya procesado por
`phantomsetup_cosmogenesis_backup`, formato binario nativo de Phantom) de las 8 corridas:

| N | masa total (seed 1) | masa total (seed 2) |
|---|---|---|
| 200 | 18800.0 | 18800.0 |
| 500 | 18800.0 | 18800.0 |
| 1000 | 18800.0 | 18800.0 |
| 2000 | 18800.0 | 18800.0 |

**La masa total queda exactamente fija en 18800 para los 4 valores de N probados** (200, 500, 1000,
2000), leída del volcado binario real de Phantom, no sólo del ASCII que escribe el generador. Confirma
que la corrección funciona de punta a punta (generador → `phantomsetup` → dump binario).

---

## Paso 3 — barrido de formación de sumideros con masa fija

Mismos parámetros físicos que toda la jerarquía CS073 (`rho_crit_cgs=1000`, `tmax=0.5`, `dtmax=0.001`),
2 semillas por N:

| N | sumideros seed 1 | sumideros seed 2 | nota |
|---|---|---|---|
| 200 | 0 | 0 | corrida completa a tmax=0.5 |
| 500 | 0 | 0 | corrida completa a tmax=0.5 |
| 1000 | **3** (masa 470.0) | **1** (masa 112.8) | **corrida ABORTADA antes de tmax** (ver aviso abajo) |
| 2000 | 8 (masa 1165.6) | 8 (masa 1109.2) | corrida completa a tmax=0.5 (consistente con la batería original) |

**Aviso importante — las dos corridas a N=1000 no llegaron a tmax=0.5:** Phantom abortó con `ERROR!
evolve: Large error in angular momentum conservation` en t≈0.404 (seed 1, dump final `cosmog_00403`,
975/1000 partículas de gas restantes) y t≈0.245 (seed 2, dump final `cosmog_00244`, 994/1000
partículas). No se tocó ningún parámetro de conservación ni se usó la variable de entorno que permite
ignorar el error (`I_WILL_NOT_PUBLISH_CRAP`) — se reporta el comportamiento observado tal cual, tal
como pide la metodología del proyecto. Los sumideros en N=1000 ya se habían formado ANTES de que la
corrida abortara (a t=0.404 y t=0.245 respectivamente, ambos menores que 0.5), así que el hallazgo "sí
forma sumideros" es válido pese al corte anticipado — pero el número final de sumideros/masa a t=0.5
para N=1000 queda sin determinar (podría seguir creciendo si la corrida hubiera continuado).

**Lectura de los números (sin veredicto — corresponde a Alexis):**

- **N=2000 con masa fija reproduce el resultado ya conocido** (8 sumideros ambas semillas) — sirve como
  chequeo de sanidad de que el pipeline nuevo no rompió nada.
- **N=200 y N=500, incluso con la masa total corregida a 18800 (en vez de 800 y 4700
  respectivamente), siguen dando CERO sumideros.** Esto pesa en contra de la lectura "el confound de
  masa total explicaba por sí solo el cero sumideros a N chico" — con la física corregida, N=500 sigue
  sin formar estructura.
- **N=1000 SÍ forma sumideros** (aunque con la corrida cortada antes de tmax) — es el primer valor de N
  intermedio, en esta pasada, donde aparece formación de sumideros con masa fija. Esto es coherente con
  la lectura de `MISTERIO_N500_vs_N2000_CS.md` de que el factor dominante es **resolución** (vecinos
  por partícula insuficientes a N≤500), no la masa total absoluta — la corrección de masa total no
  "arregla" N=500, pero sí deja ver un piso de resolución en algún punto entre N=500 y N=1000, ahora sin
  el confound de masa mezclado.
- El error de conservación de momento angular en N=1000 (y su ausencia en N=200/500/2000) es en sí un
  dato interesante no buscado — posiblemente relacionado con la propia formación abrupta de sumideros
  en una malla de densidad numérica intermedia; no se investigó la causa raíz por presupuesto de tiempo.

---

## Archivos nuevos (ninguno congelado tocado)

- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/grafo_random_layout_generar_ic_masa_fija.py`
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/grafo_random_masa_fija_generar.py`
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/grafo_random_masa_fija_correr.py`
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/grafo_random_masa_fija_verificar.py`
- Datos generados: `/Users/alexis/phantom_cs073/bateria_grafo_random_masa_fija/ic_masaFija_N{200,500,1000,2000}_s{1,2}/`
  (IC, `cosmog.in`, dumps binarios, `setup.log`, `run.log`)

## Qué falta (no ejecutado por presupuesto de tiempo)

- No se re-corrieron los dos N=1000 con más tiempo/tolerancia para ver si llegan a tmax=0.5 con más
  sumideros — quedaría como paso natural siguiente si a Alexis le interesa el número final en ese N.
- No se investigó la causa del error de conservación de momento angular en N=1000 (ni si aparecería
  también en N=700-900 u otro punto intermedio).
- No se midió `h` de equilibrio post-corrida por N (quedó priorizado el barrido de sumideros, pedido
  como más importante en la salvaguarda de tiempo).
- El atajo de submuestreo de `dens_bar` (en vez de pool físico propio por N) es válido para esta
  validación de infraestructura pero no reemplaza una re-extracción real si algún experimento futuro
  necesita la identidad física exacta del pool a cada N.

**No se declara cierre ni veredicto sobre CS073 ni sobre el misterio N500 vs N2000** — se reportan
números; la lectura final corresponde a Alexis.
