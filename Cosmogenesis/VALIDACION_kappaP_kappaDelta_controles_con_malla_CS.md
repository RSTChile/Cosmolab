# VALIDACIÓN de κ_P y κ_Δ contra controles QUE CONSERVAN LA MALLA CAUSAL

**Fecha:** 13-ago-2026 · **Tipo:** re-análisis de archivos existentes · **No se corrió ninguna simulación,
no se generaron condiciones iniciales, no se llamó a Phantom.**
**Fuente:** archivos `.sink` de Phantom ya existentes en `/Users/alexis/phantom_cs073/`.
**Entregable acompañante:** `VALIDACION_kappaP_kappaDelta_por_corrida.csv` (una fila por corrida).
**Estado:** ⚠️ **NO ES UN CIERRE.** Se reportan números. Ningún veredicto es válido sin el director.

---

## 0 · PRE-REGISTRO (declarado ANTES de calcular; no se modificó después)

> - **κ_P sobrevive contra NULL-3** (z≥3) → el piso de persistencia es real y no depende de tener malla.
> - **κ_P se cae contra NULL-3** (z<3) → el z=6,53 medía "pasó por la malla", no un mínimo de persistencia.
> - **Intermedio** → se reporta la curva, sin veredicto.
>
> Lo mismo para **κ_Δ** por separado.

Se añadió antes de calcular, como control de definición, un segundo estimador de κ_P (`κ_P_abs`,
normalizado por `tmax=0,5` en vez de por el intervalo del archivo), porque la frase del documento
—"duración total de la corrida"— admite las dos lecturas. **Sólo una de las dos reproduce el 0,981
publicado** (ver §2). Ambas se reportan.

---

## 1 · Definiciones usadas (idénticas a la tabla publicada; ninguna se cambió)

El script original (`analisis_kappa_bloque28.py`) **no está en el repositorio ni en el disco** — no se pudo
localizar. Las definiciones se reconstruyeron desde el texto de
`DISENO_EXPERIMENTOS_NODOS_ABIERTOS_desde_2.5.5_CS.md` §2 y se validaron reproduciendo los números
publicados dígito a dígito (§2). Columnas del `.sink` de Phantom: `0 time · 4 mass · 11 macc · 18 sinkID`.

| Símbolo | Definición operativa |
|---|---|
| `n_sinks` | nº de IDs distintos de sumidero en el archivo |
| **κ_P** | media sobre sumideros de (t_última_aparición − t_nacimiento) / (T_fin − T_ini), con T_ini/T_fin = primer/último instante **del archivo** |
| `κ_P_abs` | misma vida, dividida por `tmax = 0,5` (lectura literal alternativa) |
| **κ_Δ** | media sobre sumideros de masa_final / masa_inicial |
| **κ_Δ_alt** | media sobre sumideros de (masa_final − masa_inicial) = masa acretada |
| κ_V | media sobre sumideros de acreción(último tercio de vida) / acreción(primer tercio) |
| `frac_sobrev` | fracción de sumideros nacidos presentes en el último instante del archivo |

---

## 2 · REPRODUCCIÓN DE LA TABLA PUBLICADA — ✅ sale, 5 de 6 dígito a dígito

REAL = `bateria_n2000/ic_real` (n=1) contra `ic_null1..8` (NULL original, n=8).

| Invariante | Publicado REAL | **Medido REAL** | Publicado NULL | **Medido NULL** | Publicado z | **Medido z** |
|---|---|---|---|---|---|---|
| U0 · nº sumideros | 8 | **8** | 7,88 ± 0,35 | **7,875 ± 0,354** | 0,35 | **0,35** ✅ |
| U1 · κ_P | 0,981 | **0,9811** | 0,764 ± 0,033 | **0,764 ± 0,033** | 6,53 | **6,53** ✅ |
| U2 · κ_Δ | 3,74× | **3,7388** | 1,53 ± 0,065 | **1,533 ± 0,065** | 33,79 | **33,79** ✅ |
| U2 · κ_Δ alt | 193,9 | **193,875** | 31,5 ± 3,7 | **31,54 ± 3,75** | 43,33 | **43,33** ✅ |
| U3 · κ_V | 0,832 | **0,818** | 0,511 ± 0,235 | **0,518 ± 0,226** | 1,37 | **1,33** ≈ |
| (masa en sumideros) | — | **2124,4** | — | **720,3 ± 28,8** | (48,69) | **48,69** ✅ |

**La tabla se reproduce.** κ_V difiere en el 3.º decimal (0,818 vs 0,832) porque "tercios de vida" admite
más de una discretización (aquí, interpolación lineal de la masa en t_nac+vida/3 y t_nac+2·vida/3); el
veredicto (débil) no cambia.

**Hallazgo de documentación:** la frase "duración total de la corrida" es **ambigua y el número publicado
corresponde a la lectura NO literal**. Normalizando por `tmax=0,5` (lectura literal) el REAL da **0,9065**,
no 0,981. Se reproduce sólo normalizando por el intervalo cubierto por el propio archivo `.sink` —que
empieza en el **primer nacimiento**, no en t=0. Esto no es cosmético: significa que **κ_P se mide contra un
origen que la propia corrida elige**, y ese origen es más tardío justamente en el brazo que se quiere
denigrar (NULL original nace en t≈0,14; REAL en t≈0,04).

---

## 3 · GUARDA 1 · TAUTOLOGÍA — **κ_P no mide persistencia. No puede.**

Esta es la conclusión más importante del re-análisis y es previa a cualquier z.

**De los 280 sumideros nacidos en las 35 corridas de los 7 brazos, 280 llegan al último instante. Cero se
apagan. Cero tienen huecos intermedios.**

| Brazo | corridas | sumideros nacidos | sobreviven a T_fin | fracción |
|---|---|---|---|---|
| REAL | 1 | 8 | 8 | **1,0000** |
| REAL_extra (5 semillas) | 5 | 40 | 40 | **1,0000** |
| NULL_orig | 8 | 63 | 63 | **1,0000** |
| NULL-3 (con malla) | 8 | 64 | 64 | **1,0000** |
| RANDOM_ER | 8 | 64 | 64 | **1,0000** |
| NULL-4 | 3 | 25 | 25 | **1,0000** |
| NULL-5 | 2 | 16 | 16 | **1,0000** |
| **TOTAL** | **35** | **280** | **280** | **1,0000** |

Consecuencia algebraica: si todo sumidero termina en T_fin, entonces vida_i ≡ T_fin − t_nacimiento_i, y

> **κ_P = media_i(T_fin − t_nac_i) / (T_fin − T_ini) — función EXCLUSIVA de los tiempos de nacimiento.**

Se verificó numéricamente: recalculando κ_P **usando sólo los tiempos de nacimiento** (sin mirar nunca si
el sumidero murió), el error máximo contra el κ_P medido en las 35 corridas es **0,00e+00 — exactamente
cero**.

**κ_P es un reloj de nacimiento disfrazado de tasa de supervivencia.** El z=6,53 dice "en REAL los
sumideros nacen antes y más juntos", no "en REAL los sumideros viven más". Nada vive menos en ningún lado.
El rango de nacimientos lo confirma:

| Brazo | t_nac primero | t_nac último | dispersión |
|---|---|---|---|
| REAL | 0,0380 | 0,0570 | 0,0190 |
| REAL_extra | 0,0316 | 0,0524 | 0,0208 |
| **NULL_orig** | **0,1405** | **0,3395** | **0,1990** |
| NULL-3 | 0,0366 | 0,0541 | 0,0175 |
| RANDOM_ER | 0,0526 | 0,1011 | 0,0485 |
| NULL-4 | 0,0347 | 0,1907 | 0,1560 |

---

## 4 · GUARDA 2 · IDENTIDADES ALGEBRAICAS — **κ_Δ ES la masa reescalada**

Correlación sobre las 35 corridas de todos los brazos, contra `masa_fin_tot` (masa total en sumideros al
final, el observable del z=48,69):

| Métrica | Spearman ρ | Pearson r | Lectura |
|---|---|---|---|
| **κ_Δ_alt** | **+0,976** | **+0,997** | 🔴 es la misma cantidad |
| **κ_Δ** | **+0,911** | **+0,992** | 🔴 es la misma cantidad |
| κ_P_abs | +0,882 | +0,830 | 🟠 fuertemente redundante |
| κ_V | +0,823 | +0,810 | 🟠 (ya se sabía: ρ=0,902 en la medición previa) |
| κ_P | +0,702 | +0,772 | 🟠 redundante |
| t_nac_medio | −0,883 | −0,830 | (nacer antes ⇒ más masa) |

No es sólo correlación: es **identidad por construcción**.
`κ_Δ_alt = (masa_fin_tot − masa_ini_tot) / n_sumideros`, y tanto `masa_ini_tot` (todas las masas son
múltiplos del mismo 9,4 = masa de partícula) como `n_sumideros` (8 en casi todas las corridas) son
prácticamente constantes entre brazos. **κ_Δ_alt es masa_fin_tot dividida por ~8 y corrida por una
constante. El z=43,33 y el z=48,69 son el mismo número escrito dos veces.**

κ_P es la única de las tres que aporta un eje distinto de la masa (ρ=0,70), pero por §3 ese eje es el
tiempo de nacimiento, no la persistencia.

---

## 5 · RESULTADO CONTRA CADA CONTROL — REAL original (n=1)

| Métrica | REAL | NULL_orig (n=8) | z | **NULL-3 (n=8)** | **z** | RANDOM_ER (n=8) | z | NULL-4 (n=3) | z |
|---|---|---|---|---|---|---|---|---|---|
| **κ_P** | 0,9811 | 0,764 ± 0,033 | **6,53** | **0,981 ± 0,005** | **−0,02** | 0,957 ± 0,023 | 1,07 | 0,943 ± 0,053 | 0,71 |
| κ_P_abs | 0,9065 | 0,548 ± 0,027 | 13,31 | 0,909 ± 0,004 | −0,77 | 0,856 ± 0,011 | 4,82 | 0,878 ± 0,055 | 0,52 |
| **κ_Δ** | 3,7388 | 1,533 ± 0,065 | **33,79** | **3,815 ± 0,215** | **−0,36** | 2,215 ± 0,072 | 21,19 | 3,675 ± 0,310 | 0,21 |
| **κ_Δ_alt** | 193,88 | 31,54 ± 3,75 | **43,33** | **201,07 ± 8,51** | **−0,85** | 78,14 ± 3,87 | 29,89 | 186,83 ± 15,54 | 0,45 |
| κ_V | 0,818 | 0,518 ± 0,226 | 1,33 | 0,862 ± 0,081 | −0,54 | 0,458 ± 0,066 | 5,47 | 0,847 ± 0,050 | −0,58 |
| masa_fin_tot | 2124,4 | 720,3 ± 28,8 | 48,69 | 2186,7 ± 53,2 | −1,17 | 1143,3 ± 32,5 | 30,15 | 2136,9 ± 33,0 | −0,38 |
| n_sinks | 8 | 7,875 ± 0,354 | 0,35 | 8,000 ± 0,000 | — | 8,000 ± 0,000 | — | 8,333 ± 0,577 | −0,58 |
| **frac_sobrev** | **1,000** | **1,000 ± 0** | — | **1,000 ± 0** | — | **1,000 ± 0** | — | **1,000 ± 0** | — |

---

## 6 · RESULTADO CON REAL n=6 (semillas adicionales SÍ existen)

Se encontraron las 5 semillas REAL adicionales en **`/Users/alexis/phantom_cs073/bateria_real_extra_n2000/`**
(`ic_real_s301..s305`), todas con `.sink` legible, todas llegando a `tmax=0,5`, y con condiciones iniciales
verificadamente distintas entre sí y de `ic_real` (md5 y comparación numérica: 2000/2000 partículas
difieren). Se incorporan. Contraste con **Mann-Whitney exacta por enumeración completa** (no aproximación
normal), dos colas.

| Métrica | REAL n=6 | vs NULL_orig | vs **NULL-3** | vs RANDOM_ER | vs NULL-4 |
|---|---|---|---|---|---|
| **κ_P** | 0,9764 ± 0,0093 | z=6,39 · **p=0,00067** | z=−0,96 · **p=0,49** | z=0,87 · p=0,020 | z=0,62 · p=0,26 |
| κ_P_abs | 0,9126 ± 0,0042 | z=13,54 · p=0,00067 | z=0,91 · p=0,15 | z=5,40 · p=0,00067 | z=0,63 · p=0,26 |
| **κ_Δ** | 3,8771 ± 0,1186 | z=35,90 · **p=0,00067** | z=0,29 · **p=0,57** | z=23,11 · p=0,00067 | z=0,65 · p=0,26 |
| **κ_Δ_alt** | 203,28 ± 10,43 | z=45,84 · p=0,00067 | z=0,26 · **p=0,69** | z=32,32 · p=0,00067 | z=1,06 · p=0,17 |
| κ_V | 0,8832 ± 0,1113 | z=1,62 · p=0,0027 | z=0,26 · p=0,75 | z=6,46 · p=0,00067 | z=0,72 · p=0,71 |
| masa_fin_tot | 2196,5 ± 96,0 | z=51,19 · p=0,00067 | z=0,18 · p=0,55 | z=32,37 · p=0,00067 | z=1,80 · p=0,33 |
| **frac_sobrev** | 1,000 ± 0 | idéntico | idéntico | idéntico | idéntico |

Con n=6 REAL el z contra el NULL original **sube** (6,53→6,39 en κ_P; 33,79→35,90 en κ_Δ): las semillas
adicionales confirman que REAL es reproducible y que la varianza intra-REAL es chica. **El punto débil del
contraste original no era el n=1 del REAL. Era el control.**

---

## 7 · GUARDA 3 · PISO DE RUIDO — cuál es el p mínimo alcanzable

Ningún p reportado aquí puede bajar del piso estructural de su diseño. Con Mann-Whitney exacta a dos colas
el mínimo es `2/C(n₁+n₂, n₁)`:

| Contraste | n_REAL | n_control | **p mínimo posible** |
|---|---|---|---|
| REAL n=1 vs cualquier brazo n=8 | 1 | 8 | 2/9 = **0,222** (una cola: 1/9 = 0,111) |
| REAL n=6 vs NULL_orig / NULL-3 / RANDOM_ER | 6 | 8 | 2/3003 = **0,000666** |
| REAL n=6 vs NULL-4 | 6 | 3 | 2/84 = **0,0238** |
| REAL n=6 vs NULL-5 | 6 | 2 | 2/28 = **0,0714** |

Todo `p=0,00067` de la §6 es **el piso, no una medida de fuerza**: significa "separación total de rangos",
nada más. Y el `p=0,49 / 0,57 / 0,69` de NULL-3 está muy lejos de cualquier piso: es solapamiento genuino.

---

## 8 · GUARDA 4 · CORRIDAS EXCLUIDAS Y ANOMALÍAS

**Exclusiones por ilegibilidad o corrida incompleta: ninguna.** Las 35 corridas de los 7 brazos tienen
`.sink` legible y **todas** terminan exactamente en `t = 0,500 = tmax`.

**⚠️ ANOMALÍA GRAVE — `bateria_null5_n2000` NO ES UN CONTROL.** Se detectó por la guarda de determinismo
(la lección de "verificar determinismo antes de asumir azar"). Las dos corridas NULL-5 dan κ_P, κ_Δ, κ_Δ_alt
y masa **idénticos entre sí y a REAL hasta el último dígito**. Verificación:

- `md5` de las IC: distintas. Comparación numérica de las IC contra REAL: las 2000 partículas difieren.
- Pero **ordenando las filas de la IC, `ic_null5_s801` e `ic_null5_s802` son bit a bit idénticas a `ic_real`**
  (`maxdif = 0,00e+00`), mientras que NULL-3 no lo es (`maxdif = 3,43`).
- Es decir: **NULL-5 es REAL con las partículas renumeradas.** SPH es invariante bajo relabeling, así que
  la trayectoria es la misma: la columna de masa del `.sink` es idéntica a REAL bit a bit y las posiciones
  difieren en 1e-10 (redondeo del formato de texto).

**NULL-5 no puede separarse de REAL porque es REAL.** Se excluye de todo veredicto y queda como cuestión
abierta para el director: revisar qué operación genera `bateria_null5_n2000`, porque no está perturbando
nada físico. (No se tocó ese material; sólo se leyó.)

**NULL-4** (n=3) es un control real (IC genuinamente distintas), pero con n=3 su p mínimo es 0,024 y su DE
es poco fiable; se reporta sin peso decisorio.

---

## 9 · SEGUNDA PREGUNTA (C-N1.3) — ¿la persistencia FILTRA?

**Respuesta cuantitativa: no filtra nada. Fracción que sobrevive = 1,0000 en los 7 brazos, en las 35
corridas, en los 280 sumideros.** Cero apagados, cero huecos, cero fusiones detectables en el `.sink`.

Ya estaba documentado que el filtro **no actúa en el nacimiento** (8 vs 7,88 ± 0,35, z=0,35). Este
re-análisis cierra la otra mitad: **tampoco actúa en la supervivencia** — no porque REAL y NULL empaten en
una tasa intermedia, sino porque **la tasa es 100% en ambos y no puede ser otra cosa** en esta
configuración de Phantom (un sumidero, una vez creado, no se destruye).

Lo único que difiere entre brazos es **cuándo nacen** y **cuánta masa comen**. Esas son dos cosas, y la
segunda es la que ya medía el z=48,69.

En los términos del nodo: **C-N1.3 no está puesto a prueba por este material.** Para que "la persistencia
filtra" sea una afirmación falsable haría falta un observable donde algo pueda apagarse — lo que este
material no ofrece. Es un límite del sustrato, no del análisis.

---

## 10 · SÍNTESIS FRENTE AL PRE-REGISTRO

Aplicando literalmente lo declarado en §0, sin reinterpretarlo:

| Invariante | Criterio pre-registrado | Medido vs NULL-3 | **Rama que se activa** |
|---|---|---|---|
| **κ_P** | z≥3 sobrevive / z<3 se cae | z = **−0,02** (n=1) · z = **−0,96**, p=0,49 (n=6) | 🔴 **SE CAE** |
| **κ_Δ** | z≥3 sobrevive / z<3 se cae | z = **−0,36** (n=1) · z = **+0,29**, p=0,57 (n=6) | 🔴 **SE CAE** |
| κ_Δ_alt | idem | z = **−0,85** (n=1) · z = **+0,26**, p=0,69 (n=6) | 🔴 **SE CAE** |

Contra el **Erdős-Rényi ajeno** el cuadro es distinto y merece registrarse aparte: κ_Δ sí se separa
(z=21–23, p en el piso), κ_P casi no (z≈0,9–1,1). Es decir, el ordenamiento es

> **NULL_orig (720) ≪ Erdős-Rényi (1143) < REAL (2124) ≈ NULL-3 (2187) ≈ NULL-4 (2137)**

REAL supera al Erdős-Rényi, pero **no supera al control que conserva la malla y sólo baraja las aristas**.
Lo que separa a REAL del ER no lo separa de NULL-3.

**Lectura de lo medido (números, no veredicto):** los tres invariantes se comportan como un único eje —
la masa acretada en sumideros— visto desde tres ángulos, más un cuarto observable (el tiempo de nacimiento)
que κ_P recodifica. Contra el control sin malla los cuatro se separan; contra el control con malla ninguno.
Es exactamente el patrón que el director planteó como hipótesis a validar: **el contraste original mide
"pasó por la malla", no "tiene ESTA estructura".**

**No se declara cierre.** Queda a decisión del director.

---

## 11 · Reproducibilidad

Script de re-análisis: `validacion_kappa_analisis.py` y `validacion_kappa_guardas.py` (en esta carpeta). Sólo lectura;
ninguna escritura sobre `/Users/alexis/phantom_cs073/`. Entradas: 35 archivos `cosmog01.sink`.
Salida tabular: `VALIDACION_kappaP_kappaDelta_por_corrida.csv`.
