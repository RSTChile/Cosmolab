# INFRA — Liberar disco para subir la resolución (poda de volcados intermedios de Phantom)

**Fecha:** 12-ago-2026 · **Tipo:** infraestructura, no experimento · **Director:** Alexis López Tapia

Esto NO es un experimento y no declara ningún cierre. Es trabajo de infraestructura: sacar del disco
archivos que ningún análisis lee, para que se pueda subir la resolución a N=8000.

---

## 0. Resumen en una página

| | |
|---|---|
| Disco antes | 98% lleno — 13 GB libres de 466 GB |
| `phantom_cs073/` antes | **18 GB** |
| `phantom_cs073/` después | **7.8 GB** |
| **Borrado** | **10.19 GB** — 209.503 volcados intermedios en 420 corridas |
| Corridas dejadas intactas por incompletas | 22 (documentadas una por una, §4) |
| ¿El análisis da los mismos números? | **Sí, idénticos hasta el último dígito** (§5) |
| `test_massiva*` | **NO tocadas** — 5.8 GB, candidatas, decisión de Alexis (§6) |
| Arreglo hacia adelante | **Implementado** — `podar_una_corrida()` (§7) |

**Advertencia importante que hay que leer (§8):** los 10.19 GB están borrados del árbol de archivos
(`du` lo confirma: 18 GB → 7.8 GB), pero macOS todavía **no los devuelve** porque nueve *snapshots
locales de Time Machine* tomados hoy (04:49 a 12:52) siguen referenciando esos bloques. `df` sigue
diciendo 13 GB libres. Se sueltan solos en ~24 h, o al instante con un comando que necesita `sudo`
y por eso no corrí yo.

---

## 1. El problema, medido

El protocolo validado de esta línea corre Phantom con `tmax=0.500` y `dtmax=0.001`. Phantom escribe
un volcado por cada `dtmax`, o sea **501 volcados por corrida** (`cosmog_00000` … `cosmog_00500`).

Corrección a la estimación de partida: no todos los volcados pesan lo mismo. Con `nfulldump=10`
Phantom escribe un volcado *completo* cada 10 y los otros nueve son *volcados chicos*. Medido en
`bateria_fase7_f704_cortar_bien/A2-B0-C2-batch3-r100_f704_c2`:

| Archivo | N=2000 | N=8000 (medido en `ON77_sistemaA_cierre/ic_N8000`) |
|---|---|---|
| volcado completo (1 de cada 10) | 154 KB | 610 KB |
| volcado chico (9 de cada 10) | 34 KB | 130 KB |
| **volcados de una corrida entera** | **~24 MB** | **~89 MB** |
| carpeta de corrida completa | 27 MB | ~92 MB |

Así que una batería de 40 corridas a N=8000 sin podar son **~3,6 GB**, no 12,6 GB. La cifra es más
benigna de lo estimado, pero el diagnóstico de fondo no cambia: con 13 GB libres y varias baterías
por delante, el disco es el cuello de botella, y **el 89% de lo que se escribe no lo lee nadie**.

## 2. Auditoría: qué lee realmente el análisis (verificado en el código, no supuesto)

Leí los analizadores antes de borrar. El contrato de lectura del proyecto es:

**El analizador central.** `cs090_fase5b_analizar.py::analizar_carpeta` es el que usa toda la línea
de Fase V-B en adelante — lo importan *tal cual* `cs090_fase6_o3b_analizar.py`,
`cs090_fase6_o3e_correr.py`, `cs090_fase7_f702_analizar.py`, `cs090_fase7_f704_analizar.py` y
`cs090_fase6_o3a_convergencia_resolucion.py`. Pide `listar_dumps(carpeta)` y de esa lista toca
**sólo dos elementos**:

```python
gas0, sinks0 = leer_dump(dumps[0])     # n_gas_inicial
dump_final   = dumps[-1]               # masa de gas, masa de sumideros, fracción de masa
```

Nunca recorre los del medio.

**Archivos imprescindibles por corrida** (la lista de Alexis, verificada y ampliada en dos ítems):

| Archivo | Quién lo lee |
|---|---|
| `cosmog_00000` | `analizar_carpeta` (`dumps[0]`); explícito en `cs088_espectro_proximidad_null12.py`, `null2_zeldovich_disenar_verificar.py`, `grafo_random_masa_fija_verificar.py`, `cs090_fase6_o3f_extraer_gas.py` |
| `cosmog_00500` | `analizar_carpeta` (`dumps[-1]`); explícito en `cn4_delimitacion_fof.py`, `cs079_delimitacion_cn4.py`, `cs090_fase6_o4a_observable_comun.py`, `cs090_fase6_o3f_extraer_gas.py`; y todos los `*_correr.py` lo usan como marca de "ya corrida, no recomputar" |
| `cosmog01.sink` | κ_V, n_sumideros, t_primer_sumidero, masa acretada — `cs078_kappaV_permutacion.py`, `null{1,2,3}_bateria_comparar.py`, `real_extra_comparar.py`, `grafo_random_bateria_comparar.py`, `ON77_sistemaA/B*.py`, `cs090_fase6_outliers_paso3_phantom.py` |
| `cosmogenesis_ic.txt` | `cs090_fase7_f705_geometria_ic_todas.py`, `cs090_fase6_o3a_geometria_ic.py`, `cs090_fase6_o4a_observable_comun.py` |
| `meta_regla.json` | todos los analizadores de Fase V-B en adelante |
| `cosmog.in`, `*.log` | configuración y bitácora; pesan poco, son la evidencia de qué se corrió |
| **`cosmog01.ev`** *(agregado a la lista)* | serie temporal de energías/momentos, 181 KB; barato y es la única traza temporal que queda tras podar |

**Búsqueda negativa — lo que autoriza a borrar.** Un `grep` sobre *todos* los `.py` del proyecto
buscando `for … in listar_dumps`, `dumps[1:-1]`, `dumps[1:]`, `len(dumps)` **no encuentra ningún
analizador que recorra los volcados intermedios**. Los dos únicos usos de `dumps[1:-1]` son podas ya
existentes: `cs090_fase6_o3d_barrido_kcap.py:356` y `cs090_fase7_f701_factorial.py:372` — o sea que
podar ya era práctica establecida (y por eso esas dos baterías aparecen con 0 MB liberables abajo:
ya estaban podadas).

**El único lector de un volcado intermedio** es el `_smoke_test()` de `leer_volcado_phantom.py`, que
toma `dumps[len(dumps)//2]`. Tras podar, esa lista tiene 2 elementos y `dumps[1]` es el volcado
final: el smoke test sigue pasando. **Verificado corriéndolo después de podar** (§5).

**Conclusión de la auditoría:** la lectura preliminar de Alexis era correcta. Sólo agrego `*.ev` a
la lista de conservados y aclaro que se conserva *todo* archivo que no sea un volcado binario.

## 3. Dry run (hecho ANTES de borrar) y resultado

Herramienta: `infra_podar_volcados.py`, **dry run por defecto** (hay que pasar `--ejecutar` para que
borre). El dry run recorrió las 554 carpetas de corrida y escribió el informe antes de tocar nada.

Detalle completo por corrida: **`infra_poda_detalle.csv`** (554 filas — carpeta, nº de volcados,
índice primero/último, tmax, dtmax, índice final esperado, MB, podable sí/no, motivo, verificación).
Resumen por batería: **`infra_poda_por_bateria.csv`**.

| batería | corridas | podadas | intactas | MB antes | **MB liberados** |
|---|---:|---:|---:|---:|---:|
| bateria_fase7_f702_escalera | 72 | 72 | 0 | 1737,3 | **1714,8** |
| bateria_fase7_f704_cortar_bien | 60 | 60 | 0 | 1447,0 | **1428,4** |
| bateria_fase6_o3c_mecanistico | 48 | 47 | 1 | 1137,0 | **1117,5** |
| bateria_fase5b_a2b0c2_escala_v4 | 40 | 40 | 0 | 964,9 | **952,4** |
| (corridas sueltas en la raíz `phantom_cs073/`) | 26 | 6 | 14 | 7403,6 | **898,4** |
| bateria_fase6_o3e_memoria | 30 | 30 | 0 | 723,4 | **714,0** |
| bateria_fase6_o3b_rewiring | 24 | 24 | 0 | 579,0 | **571,5** |
| bateria_fase5b_a2b0c2_escala_v3 | 23 | 23 | 0 | 554,8 | **547,6** |
| bateria_fase6_outliers_negativos | 11 | 11 | 0 | 265,5 | **262,1** |
| bateria_n2000 | 19 | 9 | 4 | 218,9 | **211,9** |
| bateria_null3_n2000 | 8 | 8 | 0 | 193,0 | **190,5** |
| bateria_fase5b_a2b0c2_escala_v2 | 8 | 8 | 0 | 192,9 | **190,4** |
| bateria_grafo_random_n2000 | 8 | 8 | 0 | 192,6 | **190,1** |
| bateria_null1_n2000 | 8 | 8 | 0 | 185,5 | **183,0** |
| bateria_null2_n2000 | 8 | 8 | 0 | 185,5 | **183,0** |
| bateria_fase5b_a2b0c2_piloto | 6 | 6 | 0 | 144,7 | **142,8** |
| bateria_real_extra_n2000 | 5 | 5 | 0 | 120,7 | **119,1** |
| ON77_sistemaB_cierre | 5 | 5 | 0 | 120,0 | **118,4** |
| bateria_null4_n2000 | 3 | 3 | 0 | 72,4 | **71,4** |
| ON77_sistemaA_cierre | 3 | 2 | 1 | 137,8 | **69,8** |
| bateria_grafo_random_masa_fija | 8 | 6 | 2 | 83,8 | **66,9** |
| bateria_null5_n2000 | 2 | 2 | 0 | 48,2 | **47,6** |
| test_turbulencia_r9 | 2 | 2 | 0 | 48,2 | **47,6** |
| piloto_null1 | 4 | 4 | 0 | 26,5 | **26,2** |
| piloto_null3 | 3 | 3 | 0 | 20,2 | **20,0** |
| piloto_grafo_random | 3 | 3 | 0 | 19,7 | **19,5** |
| piloto_null2 | 3 | 3 | 0 | 19,7 | **19,5** |
| piloto_null2_zeldovich | 3 | 3 | 0 | 19,7 | **19,5** |
| ON77_sistemaB_corregido | 5 | 5 | 0 | 16,2 | **16,1** |
| piloto_null3_dosis | 2 | 2 | 0 | 13,6 | **13,4** |
| ON77_sistemaA_corregido | 4 | 4 | 0 | 12,4 | **12,3** |
| N4000 | 26 | 0 | 0 | 16,2 | 0,0 |
| bateria_fase6_o3d_kcap | 38 | 0 | 0 | 11,8 | 0,0 *(ya podada)* |
| bateria_fase7_f701_kcapM | 36 | 0 | 0 | 11,2 | 0,0 *(ya podada)* |
| **TOTAL** | **554** | **420** | **22** | **16.944** | **10.186 MB = 10,19 GB** |

Ejecutado con `--ejecutar --verificar-sarracen`. Resultado real, idéntico al dry run:
**209.503 volcados borrados, 10,19 GB, en 420 corridas.** `du -sh phantom_cs073` pasó de **18 G a 7,8 G**.

Una corrida típica pasó de 27 MB a **3,0 MB**:

```
cosmog.in  cosmog01.ev  cosmog01.sink  cosmog_00000  cosmog_00500
cosmogenesis_ic.txt  meta_regla.json  run.log  setup.log
```

## 4. Verificación de integridad: qué NO se podó y por qué

Regla aplicada, sin excepciones: **una corrida se poda sólo si está completa Y su volcado final abre
bien.** Concretamente, la herramienta exige las cuatro cosas:

1. Al menos 3 volcados (si no, no hay intermedios que borrar).
2. `tmax` y `dtmax` legibles del propio `.in` de esa corrida → índice final esperado = `round(tmax/dtmax)`.
   No se asume 500: se calcula por corrida.
3. El índice del último volcado en disco **es exactamente** el esperado, y el primero es 0.
4. Ese volcado final se **abre con `sarracen`** (el mismo lector que usa el análisis) y tiene >0
   partículas. No basta con que el archivo exista.

Si algo falla, la carpeta queda **intacta** y se documenta. Motivo: en una corrida incompleta
`dumps[-1]` *es* un volcado intermedio, y borrarlo destruiría el único estado final que esa corrida
tiene. Las 22 corridas intactas (6,56 GB de volcados intermedios que **no** se tocaron):

| corrida | último volcado | esperado | MB no tocados |
|---|---:|---:|---:|
| `test_massiva` | 6011 | 6366 | 3186,3 |
| `test_massiva_hires` | 650 | 1592 | 2998,5 |
| `test_apr` | 252 | 1592 | 134,7 |
| `ON77_sistemaA_cierre/ic_N8000` | 361 | 500 | 66,3 |
| `test_apr_iso`, `test_apr_iso2` | 125 | 12732 | 60,0 c/u |
| `bateria_grafo_random_masa_fija/ic_masaFija_N1000_s1` | 403 | 500 | 9,9 |
| `bateria_grafo_random_masa_fija/ic_masaFija_N1000_s2` | 244 | 500 | ~6 |
| `bateria_fase6_o3c_mecanistico/A2-B0-C2-mec-r10__c1-rigido-soporte` | 100 | 500 | ~2,5 |
| `bateria_n2000/ic_real_apr_sink` | 458 | 1000 | ~11 |
| `bateria_n2000/pool_d_vel`, `pool_d`, `diag_layout222` | 38 / 2 / 2 | 500 | ~1 |
| `run_vel_{turb,hered}_N8550_{real,null}` (4) | 8–16 | 20 | ~9,8 el mayor |
| `run_control_N4000`, `run_sweep_N500`, `run_sweep_N1000`, `run_n250_smoke2`, `run_smoke` | 2–8 | 20–100 | <1 c/u |

Ninguna dio error de integridad: las 22 quedaron fuera por **incompletitud**, no por corrupción.
Ninguna de las 420 podadas falló la verificación con `sarracen`.

## 5. Verificación de que el análisis sigue dando los MISMOS números

Herramienta: `infra_verificar_poda_no_cambia_analisis.py`. Corre el analizador **real** del proyecto
(`cs090_fase5b_analizar.analizar_carpeta`, importado tal cual, sin modificarlo) sobre 5 corridas y
compara **12 métricas** contra los CSV que ya estaban guardados en disco *antes* de podar
(`cs090_fase7_f704_phantom_crudo.csv`, `cs090_fase7_f702_phantom_crudo.csv`). La comparación es de
igualdad exacta de la representación numérica (`%.17g`): no se tolera ni un dígito de deriva.

Por qué contra esos CSV y no contra una medición mía: esos CSV los escribió otro script en otra
sesión, con los 501 volcados en disco. Son el "antes" congelado, independiente de mí.

| | |
|---|---|
| **ANTES de podar** | 40 valores comparados, **0 diferencias** |
| **DESPUÉS de podar** | 40 valores comparados, **0 diferencias** |

Corridas verificadas: `A2-B0-C2-batch3-r100_f704_{anticosto,antisoporte,azar}`,
`A2-B0-C2-batch3-r0_s471829_f702_{e0,e1}`. Métricas: `n_gas_inicial`, `n_dump_final`,
`masa_gas_final`, `masa_sumideros_final`, `masa_total_final`, `fraccion_masa_en_sumideros`,
`n_sumideros`, `t_primer_sumidero`, `masa_acretada_total`, `kappa_v_agregado`,
`kappa_v_medio_valido`, `n_kappa_indefinidos`.

Ejemplo (idéntico antes y después): `f704_anticosto` → `dump_final=cosmog_00500`,
`frac_masa=0.15099999999999997`, `kappaV=1.3424657534246576`.

Dos controles extra, también post-poda:

- `./venv/bin/python leer_volcado_phantom.py` (el smoke test que lee un volcado *intermedio*):
  **pasa** — `ic_real: cosmog_00500 → 1774 partículas, 8 sumideros` / `ic_null1: 1922, 8`.
- Acceso explícito `cosmog_00000` + `cosmog_00500` (el camino de `cs088`, `o3f`, `cn4`, `cs079`,
  `o4a`) sobre `bateria_n2000/ic_real`: **funciona** — 2000 gas en t=0, 1774 gas + 8 sumideros en t=0,5.

**En simple:** la corrida era una película de 501 fotogramas de la que el análisis sólo miraba el
primero y el último. Tiramos los 499 del medio y nos quedamos con las dos fotos que se miraban, más
el cuaderno de bitácora (`.sink`, `.ev`, `.log`). Las cuentas dan exactamente lo mismo porque son
literalmente los mismos bytes de entrada.

## 6. Las dos carpetas `test_massiva*` — evaluadas, NO borradas

Como se pidió: evaluadas y **no tocadas**.

| carpeta | tamaño | estado |
|---|---:|---|
| `test_massiva/` | 3,0 GB | incompleta (último volcado 6011 de 6366 esperados), 6021 archivos |
| `test_massiva_hires/` | 2,8 GB | incompleta (último 650 de 1592), 660 archivos |

**Quién las referencia:** una búsqueda de la cadena `test_massiva` en todo `Cosmolab/` devuelve
exactamente **dos apariciones, ambas de mención pasajera y ninguna funcional**:

- `leer_volcado_phantom.py`, en el docstring, sección "qué falta para escalar esto": *"Para corridas
  grandes (N8550, test_massiva) verificar tiempo de lectura por dump…"*. Es una nota al futuro, no
  código: no hay ninguna ruta a `test_massiva` en ninguna función.
- `INFRAESTRUCTURA_lector_phantom_CS.md`, que cita ese mismo docstring.

**Ningún script las abre, ningún informe reporta números sacados de ellas, ningún CSV del proyecto
las usa.** Además su prefijo de volcado es `sphere_`, no `cosmog_`: son pruebas de la esfera de
Bonnor–Ebert / APR de julio-agosto, ajenas a la línea A2-B0-C2.

**Recomendación:** candidatas claras a borrar — **5,8 GB** de un tiro, más del 70% de lo que queda en
`phantom_cs073/`. **La decisión es de Alexis; yo no las borré.** En la misma bolsa, y por el mismo
criterio (prefijo `sphere_`, sin referencias, incompletas), están `test_BEsphere/`,
`test_sphereinbox/`, `test_apr*/`.

## 7. El arreglo hacia adelante — implementado

**Se investigó desacoplar la frecuencia de volcado sin tocar la física.** Phantom *sí* tiene el
parámetro: `nout` en el `.in` ("write dumpfile every n dtmax, -ve=ignore"). Leyendo el fuente en
`phantom/src/main/evolve_utils.F90:289`:

```fortran
writedump = ((nout <= 0) .or. (mod(noutput,nout)==0))
```

Con `nout=50` Phantom escribiría 10 volcados en vez de 501, **sin tocar `tmax` ni `dtmax`**, o sea
sin cambiar el paso de tiempo ni la física. Suena perfecto. **Pero no sirve**, por una razón concreta:

- El nombre del volcado sale de `getnextfilename`, que **incrementa el número de a uno cada vez que
  se escribe un volcado** (`evolve_utils.F90:301`, `utils_filenames.f90:37`). Con `nout=50` los 10
  volcados se llamarían `cosmog_00001`…`cosmog_00010`: **el estado final dejaría de llamarse
  `cosmog_00500`**. Eso rompe el hardcodeo de `cosmog_00500` en `cn4_delimitacion_fof.py`,
  `cs079_delimitacion_cn4.py`, `cs090_fase6_o4a_observable_comun.py`,
  `cs090_fase6_o3f_extraer_gas.py`, `cs090_fase7_f702_analizar.py` y **el chequeo de idempotencia de
  todos los `*_correr.py`** (que se saltean una carpeta si ya existe `cosmog_00500`) — con lo cual
  las baterías se recomputarían enteras.
- Además, con `nout>0` la condición de volcado completo pasa a ser
  `mod(noutput, nout*nfulldump)==0` (`io_control.f90:183`): los intermedios serían todos *chicos*.

Cambiar `nout` cambiaría la convención de nombres del protocolo. **Así que no lo hice, y recomiendo
no hacerlo.** Queda documentado por si alguna vez se rediseña el protocolo a propósito.

**La alternativa, que es la implementada:** poda automática inmediatamente después de cada corrida.
En `infra_podar_volcados.py`:

```python
from infra_podar_volcados import podar_una_corrida
...
subprocess.run([phantom, "cosmog.in"], cwd=carpeta, check=True)
podar_una_corrida(carpeta)      # <- acá, apenas termina esa corrida
```

Es la misma poda que ya hacían *inline* `cs090_fase6_o3d_barrido_kcap.py` y
`cs090_fase7_f701_factorial.py`, pero con la verificación de completitud e integridad que aquellas
no tienen: **si la corrida quedó incompleta no borra nada y lo dice**. Es idempotente. Probada en
los tres casos:

```
[poda] A2-B0-C2-batch3-r100_f704_c2: NO se poda -- nada que podar (menos de 3 volcados)   # ya podada
[poda] pool_d_vel: NO se poda -- INCOMPLETA: último volcado 38 != final esperado 500      # incompleta
[poda] Cosmogenesis: no es una carpeta de corrida                                          # no es corrida
```

No modifiqué ningún script congelado: `podar_una_corrida` está para que la llamen los runners
**nuevos**. Para las baterías que corran con los runners actuales, la línea equivalente es, al
terminar la batería:

```bash
./venv/bin/python infra_podar_volcados.py --ejecutar --raiz /Users/alexis/phantom_cs073/bateria_NUEVA
```

**Con esto, una batería de 40 corridas a N=8000 pasa de ~3,6 GB a ~200 MB en disco** (~5 MB por
corrida: `cosmog_00000` 610 KB + `cosmog_00500` 610 KB + `.sink` + `.ev` + `.log` + `ic.txt`), con
un pico transitorio de ~90 MB mientras esa corrida está en curso. El disco deja de ser el límite.

## 8. Lo que falta para que el sistema operativo devuelva el espacio

`du` confirma que los 10,19 GB están fuera del árbol de archivos (18 G → 7,8 G), pero `df` sigue
diciendo 13 GB libres. La causa está identificada: hay **nueve snapshots locales de Time Machine
tomados hoy** (04:49, 05:49, 06:49, 07:49, 08:49, 09:49, 10:49, 11:49, 12:52) y todos son
posteriores a la escritura de las baterías, así que siguen referenciando los bloques borrados.

```
$ tmutil listlocalsnapshots /
com.apple.TimeMachine.2026-08-12-044900.local
… (9 en total)
```

Son *snapshots locales* (copias de conveniencia en el disco interno), **no** el respaldo real de
Time Machine en el volumen externo de 8 TB — ese no se toca. macOS los adelgaza solo bajo presión de
disco y los caduca a las ~24 h, así que **el espacio vuelve solo dentro de un día**. Para tenerlo ya:

```bash
sudo tmutil thinlocalsnapshots / 20000000000 4     # pide ~20 GB de vuelta, urgencia máxima
```

Requiere contraseña de administrador, por eso **no lo corrí yo**. Es el único paso pendiente para
que los 10,19 GB aparezcan en `df`.

## 9. Archivos de este trabajo

**Nuevos** (ninguno modifica nada congelado):

- `infra_podar_volcados.py` — la herramienta. Dry run por defecto; `--ejecutar` para borrar;
  `--verificar-sarracen` para integridad fuerte; `--raiz`, `--csv`, `--excluir`. Expone
  `podar_una_corrida()` como gancho post-corrida.
- `infra_verificar_poda_no_cambia_analisis.py` — el control de calidad: recomputa el análisis real y
  lo compara contra los CSV guardados. Sale con código 1 si algún número difiere.
- `infra_poda_detalle.csv` — 554 filas, una por corrida (auditoría completa, incluye las 22 intactas
  con su motivo).
- `infra_poda_por_bateria.csv` — resumen por batería.
- `INFRA_liberar_disco_alta_resolucion_CS.md` — este informe.

**Modificados:** ninguno. **Commits de git:** ninguno.

## 10. Lo que queda abierto (decisión de Alexis, no mía)

1. `sudo tmutil thinlocalsnapshots / 20000000000 4` — para que los 10,19 GB aparezcan en `df` hoy.
2. `test_massiva/` + `test_massiva_hires/` (5,8 GB) — nadie las referencia; ¿se borran?
3. Mismo criterio, sin evaluar en detalle: `test_BEsphere/`, `test_sphereinbox/`, `test_apr*/`.
4. Las 22 corridas incompletas: quedaron intactas a propósito. Si alguna es basura conocida, podarla
   a mano libera hasta 6,56 GB más (de los cuales 6,18 GB son las dos `test_massiva*`).
