# FASE VI · O3-B — Control de reconfiguración (double-edge-swap): el grafo original contra su gemelo de mismos grados

**Fecha:** 11-ago-2026 · **Ejecuta:** CC (Claude) · **Tarea:** O3-B del `FASE6_PLAN_EJECUCION_COMPLETA_CS.md`
**Propuesta original:** GPT-5.6 Sol (F6-04) — *"uno de los controles más importantes de toda la serie"*
**Phantom:** autorizado por Alexis para esta línea · **Diámetro:** medición oficial vigente
(`cs090_diam_corregido.diam_gigante`), NO el `_diam` de `cs055`

> No se declara cierre ni veredicto. Ningún script congelado fue modificado (todos sólo se importan).
> No se hicieron commits de git.

---

## 0. En simple, con analogía

En Fase V-B teníamos 40 parejas de "maquetas de alambre": una compacta (Clase I) y una extendida
(Clase III), con la misma cantidad de nodos y el mismo tope de vecinos por nodo. Llenábamos las dos de
arena y las sacudíamos con gravedad (Phantom). La extendida juntaba más arena en los grumos: ganaba
31 de 40 veces.

La objeción dura es: **¿gana por cómo está tejida, o simplemente porque tiene más alambre / más nudos
por nodo?** Comparar contra "una maqueta al azar" no sirve, porque cambia veinte cosas a la vez.

Este control cambia **una sola cosa**. A cada maqueta extendida le fabricamos un **gemelo
reconfigurado**: se desatan todos los alambres y se vuelven a atar al azar, pero cuidando que
**cada nudo termine con exactamente la misma cantidad de alambres que tenía**. Misma gente en la sala,
cada persona con el mismo número de amigos que antes — pero los amigos son otros, sorteados. Lo que se
destruye es el **tejido**: los triangulitos, los barrios, la forma de la trama. Lo que se conserva es
todo lo "trivial": nodos, aristas, grados nodo por nodo.

Después las dos maquetas —la original y su gemelo— se llenan de arena igual y se sacuden igual.

**Lo que salió, en una frase:** el original **sí conserva ventaja** sobre su gemelo — gana 9 de 12 en
masa acretada y 11 de 12 en κ_V — y esa ventaja es de la mitad del tamaño de la ventaja
Clase III-vs-Clase I original. O sea: **el efecto no se explica sólo por "cuánto alambre hay"**, aunque
tampoco desaparece entero cuando se destruye el tejido. Y hay un detalle que apunta a un mecanismo:
**cuanto más "triangulitos" tenía el original (y por tanto más perdía al barajar), más grande era su
ventaja** (ρ=0.77, p=0.003).

---

## 1. Qué se hizo, con qué archivos

| Archivo nuevo | Qué hace |
|---|---|
| `cs090_fase6_o3b_rewiring.py` | Selecciona los 12 grafos, fabrica los gemelos por double-edge-swap, verifica grados nodo por nodo, mide estructura y pendiente corregida, escribe las condiciones iniciales de Phantom |
| `cs090_fase6_o3b_correr.py` | Corre Phantom (mismo protocolo exacto de toda la línea) |
| `cs090_fase6_o3b_analizar.py` | Verificación cruzada contra `meta_regla.json`, extrae métricas y hace el test pareado |

| CSV de salida | Contenido |
|---|---|
| `cs090_fase6_o3b_seleccion.csv` | los 12 grafos elegidos y por qué |
| `cs090_fase6_o3b_estructura.csv` | solapamiento, clustering, triángulos, gigante, pendientes (datos crudos de estructura) |
| `cs090_fase6_o3b_phantom_crudo.csv` | 24 corridas de Phantom, una fila por corrida |
| `cs090_fase6_o3b_phantom_pares.csv` | 12 pares, diferencias pareadas |
| `cs090_fase6_o3b_estadistica.csv` | test de signos + Wilcoxon |

Baterías de Phantom: `/Users/alexis/phantom_cs073/bateria_fase6_o3b_rewiring/<rule_id>_orig` y
`<rule_id>_rewire` — **prefijo/sufijo nuevo, jamás usado antes en la línea** (lección del bug de
colisión de nombres de Fase V-B).

Scripts congelados sólo importados, nunca tocados: `cs090_fase5_generador.py`, `cs090_fase5_motor.py`,
`cs090_fase5_clasificador.py`, `cs090_fase5b_phantom_adaptador.py`, `cs090_fase5b_analizar.py`,
`cs090_diam_corregido.py`, `cs080_renormalizacion.py`, `cg003_diagnostico_gromov.py`,
`cs072_modulos/piezas/p_semilla_causal.py` (de ahí sale el swap), `null3_generar_ic.py` /
`null3_investigacion_preliminar.py` (leídos como referencia, no modificados ni copiados).

---

## 2. Selección de los 12 grafos — criterio y por qué

**Universo:** las 40 filas con rol `III` de `cs090_fase5b_TOTAL_40pares.csv` (las que ya pasaron por
Phantom en Fase V-B), unidas con la **pendiente corregida** de `cs090_fase6_remedicion_430.csv`.

**Criterio:** las **3 de mayor pendiente corregida dentro de cada uno de los 4 lotes de `seed_base`**
(271828 / 371828 / 471828 / 571828) = 12 grafos.

*Por qué repartido por lote y no simplemente "los 12 de mayor pendiente":* `FASE6_O2F_N_efectivo_fase5b_CS.md`
señala que el techo del diseño de Fase V-B es **experimental, no estadístico** — faltan `seed_base`
distintos. Los 12 de mayor pendiente caerían casi todos en los lotes v3/v4 (17 y 14 candidatos
respectivamente, contra 3 y 3 en v1/v2), y serían menos independientes entre sí que 3+3+3+3.

Los 39 grafos del universo siguen siendo Clase III con la medición de diámetro corregida (ninguno se
cae de clase). Uno de los 40, `A2-B0-C2-r2v1fix`, no figura en la remedición y quedó fuera.

### 2.1 — Un bug de unión encontrado al construir esta tarea (importante para toda la línea)

Los lotes **v1 y v2 de Fase V-B usaron el mismo patrón de nombre `A2-B0-C2-r{idx}` sin prefijo de
lote**. Existen por lo tanto dos reglas DISTINTAS llamadas, por ejemplo, `A2-B0-C2-r14`: una con
`seed=273187` (lote 271828) y otra con `seed=373187` (lote 371828). Indexar cualquier CSV por
`rule_id` solo hace que una pise a la otra.

La primera versión de la selección de esta tarea cayó exactamente en eso (leyó pendiente 0.7729 para
una regla cuya pendiente real es 0.4674). **La clave de unión correcta es `(rule_id, seed)`** — es la
misma lección del bug de colisión de nombres de Fase V-B, ahora del lado del análisis y no de la
generación. Está corregido en `cs090_fase6_o3b_rewiring.py` y documentado en el código. **Conviene
revisar si algún análisis anterior de la línea unió por `rule_id` solo.**

---

## 3. El gemelo reconfigurado: qué se conserva y qué se destruye

**Operación:** `barajar_aristas` de `cs072_modulos/piezas/p_semilla_causal.py` (double-edge-swap de
Maslov-Sneppen), **importada tal cual** — es la misma que usa NULL-3 de la Fase II de CS073. Toma dos
aristas (a-b) y (c-d) y las reconecta como (a-d) y (c-b): cada nodo pierde un vecino y gana otro, así
que su grado no cambia. `factor_swaps=10` (convención estándar: 10 × nº de aristas intentos de swap).
Semilla derivada `seed*9100+7` (multiplicador nuevo; los ya usados en la línea son 1000/2000/5000/6000/
7000/7500/8000, no colisiona).

Se usó a propósito la versión **SIN** filtro geométrico de longitud (la variante con filtro de
`null3_investigacion_preliminar.py` preserva la escala local, y acá queremos destruirla).

### 3.1 — Verificación numérica de lo que se conserva

| Chequeo | Resultado |
|---|---|
| Secuencia de grados idéntica **nodo por nodo** (`np.array_equal` sobre los 2000 grados) | **12/12 pares: True** |
| Nº de nodos con grado distinto | **0 en los 12** |
| Nº de aristas idéntico | **12/12** |
| Sin bucles i-i en el gemelo | **12/12** |

No se asumió: se comprobó con un `assert` que aborta el par si falla.

### 3.2 — Cuánto cambió efectivamente la estructura

| regla | lote | K | kcap | aristas | ⟨k⟩ | **solape aristas** | triáng. orig→gemelo | **clustering local orig→gemelo** | transitividad orig→gemelo | gigante orig→gemelo |
|---|---|---|---|---|---|---|---|---|---|---|
| `A2-B0-C2-r14` | 271828 | 8 | 5 | 3296 | 3.30 | 0.24% | 19 → 3 | 0.0050 → 0.0012 (4.1×) | 0.0059 → 0.0009 | 1926 → 1934 |
| `A2-B0-C2-r17` | 271828 | 5 | 5 | 3165 | 3.17 | 0.38% | 29 → 4 | 0.0119 → 0.0014 (8.8×) | 0.0092 → 0.0013 | 1844 → 1854 |
| `A2-B0-C2-r19` | 271828 | 5 | 7 | 3679 | 3.68 | 0.27% | 6 → 7 | 0.0010 → 0.0012 (0.8×) | 0.0014 → 0.0016 | 1955 → 1952 |
| `A2-B0-C2-r20` | 371828 | 5 | 6 | 3706 | 3.71 | 0.16% | 12 → 3 | 0.0023 → 0.0008 (3.1×) | 0.0029 → 0.0007 | 1968 → 1959 |
| `A2-B0-C2-r39` | 371828 | 5 | 6 | 4083 | 4.08 | 0.32% | 20 → 9 | 0.0032 → 0.0016 (2.0×) | 0.0040 → 0.0018 | 1966 → 1968 |
| `A2-B0-C2-r28` | 371828 | 6 | 6 | 3947 | 3.95 | 0.28% | 37 → 10 | 0.0105 → 0.0018 (5.9×) | 0.0076 → 0.0021 | 1947 → 1951 |
| `A2-B0-C2-batch3-r0` | 471828 | 5 | 4 | 2308 | 2.31 | 0.52% | 27 → **0** | 0.0170 → 0.0000 (∞) | 0.0164 → 0.0000 | 1681 → 1730 |
| `A2-B0-C2-batch3-r60` | 471828 | 6 | 5 | 3068 | 3.07 | 0.26% | 41 → 6 | 0.0190 → 0.0018 (10.6×) | 0.0137 → 0.0020 | 1837 → 1843 |
| `A2-B0-C2-batch3-r111` | 471828 | 6 | 5 | 3017 | 3.02 | 0.36% | 37 → 1 | 0.0171 → 0.0002 (93×) | 0.0129 → 0.0003 | 1849 → 1865 |
| `A2-B0-C2-batch4-r36` | 571828 | 8 | 4 | 2343 | 2.34 | 0.17% | 30 → 1 | 0.0213 → 0.0003 (64×) | 0.0179 → 0.0006 | 1708 → 1744 |
| `A2-B0-C2-batch4-r10` | 571828 | 5 | 5 | 2994 | 2.99 | 0.27% | 25 → 6 | 0.0116 → 0.0012 (9.8×) | 0.0087 → 0.0021 | 1821 → 1832 |
| `A2-B0-C2-batch4-r62` | 571828 | 8 | 5 | 3299 | 3.30 | 0.30% | 15 → 2 | 0.0053 → 0.0005 (10.9×) | 0.0046 → 0.0006 | 1915 → 1913 |

**Resumen:**
- **Solapamiento de aristas: 0.16% – 0.52%, media 0.30%.** Es decir, **entre el 99.5% y el 99.8% de las
  aristas son otras**. El barajado es prácticamente total.
- **Clustering local medio: 0.0104 → 0.0010** (baja ~10× en promedio; en 11 de 12 baja, en 1 sube).
- **Transitividad global: 0.0088 → 0.0012.**
- **Triángulos: 297 en total → 52** (en `batch3-r0` bajan de 27 a **cero**).
- **Componente gigante: prácticamente igual** (1681–1968 → 1730–1968; el gemelo tiende a ser
  levemente MÁS conexo, +0.3% a +2.9%). Ni el original ni el gemelo se fragmentan.
- **Excepción a mirar:** `A2-B0-C2-r19` casi no tenía clustering para empezar (6 triángulos,
  C=0.0010) y el barajado **le subió** el clustering a 7 triángulos. Es, literalmente, un par donde el
  control no controló nada — y es también uno de los 2 pares donde el gemelo ganó. Ver §6.

### 3.3 — Pendiente corregida antes y después

| regla | pendiente corr. **original** | pendiente corr. **gemelo** | Δ |
|---|---|---|---|
| `batch3-r0` | 1.089 | 1.045 | −0.044 |
| `batch4-r36` | 1.139 | 1.008 | −0.131 |
| `batch4-r10` | 1.003 | 0.870 | −0.133 |
| `batch3-r60` | 0.927 | 0.884 | −0.043 |
| `batch4-r62` | 0.894 | 0.617 | −0.277 |
| `r17` | 0.856 | 0.793 | −0.063 |
| `batch3-r111` | 0.789 | 0.867 | **+0.078** |
| `r14` | 0.789 | 0.772 | −0.017 |
| `r39` | 0.741 | 0.529 | −0.212 |
| `r20` | 0.708 | 0.596 | −0.112 |
| `r28` | 0.686 | 0.559 | −0.127 |
| `r19` | 0.673 | 0.776 | **+0.103** |
| **media** | **0.858** | **0.776** | −0.082 |

**Hallazgo secundario, y bastante incómodo para el observable "pendiente":** el gemelo reconfigurado
**sigue siendo geométricamente extenso**. La pendiente baja en 10 de 12 pero sólo −0.08 en promedio, y
en 2 casos **sube**. Con estos grafos (⟨k⟩ ≈ 2.3–4.1, muy ralos) la pendiente log(diám)-vs-log(N_cajas)
está dominada por la secuencia de grados y la ralez, **no** por el tejido local. Dicho en simple: si lo
único que uno mira es "qué tan estirada" está la maqueta, barajar los alambres casi no se nota. Lo que
sí se nota — y mucho — son los triángulos.

Esto tiene una consecuencia que conviene anotar: **la pendiente NO es un buen resumen de "organización
relacional"** para esta familia de grafos, porque sobrevive casi intacta a la destrucción de la
organización local.

### 3.4 — Nota técnica: el orden de los conjuntos de vecinos afecta la medición

Al construir esta tarea se detectó que `cs080.cajas_bfs` recorre `for v in adj[u]`, y el **orden de
iteración de un `set` de Python depende de cómo se llenó**. Reconstruir un `set` con los mismos
elementos en otro orden puede dar otra partición en cajas y, por ahí, otra pendiente (verificado:
`A2-B0-C2-r14` pasaba de 0.7729 a 0.7517 sólo por eso).

Se resolvió midiendo dos veces:
- **medición histórica** — con el objeto nativo que devuelve el motor: **reproduce exactamente**
  (Δ<1e-9) la `pendiente_corregida` de `cs090_fase6_remedicion_430.csv` en **12/12** grafos;
- **medición canónica** (vecinos insertados de menor a mayor) — la que aparece en la tabla de §3.3,
  para que original y gemelo reciban trato idéntico y la diferencia sea atribuible al grafo.

---

## 4. Phantom — protocolo y verificación cruzada

Idéntico al de toda la línea, sin un parámetro cambiado: N=2000, masa total fija=18800 (masa por
partícula 9.4), lado de caja fijo 2000^(1/3), `layout_resortes` con `seed_layout=12345`, dilatación
`Expansion` de 60 pasos, turbulencia Mach=3 seed=42, `icreate_sinks=1`, `rho_crit_cgs=1000`,
`r_crit=0.6`, `h_acc=0.3`, `tmax=0.500`, `dtmax=0.001`. **24 corridas** (12 originales + 12 gemelos),
todas con `exit_run=0` y dump final `cosmog_00500`.

**Verificaciones cruzadas hechas antes de confiar en ningún par** (todas en el código, no a mano):

1. Contra el `meta_regla.json` de cada carpeta: variante correcta (`orig` / `rewire`), mismo `rule_id`,
   mismo `seed`, mismas aristas, misma `seed_layout`, y que la carpeta declarada dentro del meta sea la
   carpeta donde está el meta. → **12/12 pares pasan, 0 problemas.**
2. Contra el `meta_regla.json` de la carpeta de Fase V-B correspondiente: mismo `rule_id` y `seed`.
   → **12/12 True.**
3. **Réplica**: la mitad "original" de cada par es una re-corrida de algo ya informado, así que su
   fracción de masa debería reproducir la de Fase V-B. → **9/12 la reproducen EXACTAMENTE** (Δ=0);
   3 no: `batch3-r111` (+0.0025), `r19` (+0.0035), `r20` (−0.0005).

**Por qué esos 3 no replican bit a bit** (auditado, no adivinado): el grafo es el mismo (mismo
`n_aristas` en el encabezado del archivo de condición inicial), pero el layout Fruchterman-Reingold
divergió. En 8 de 12 casos la diferencia máxima de coordenada entre el archivo de Fase V-B y el nuevo
es ~1e-15 (ruido de punto flotante); en 4 casos (`r14`, `r19`, `r20`, `batch3-r111`) es de 15–23
unidades, o sea que la relajación amplificó una perturbación de nivel 1e-16 hasta un layout distinto.
Es la conocida sensibilidad caótica del FR, no un error de identidad de grafo.

**Esto NO invalida la comparación pareada de esta tarea** (los dos brazos de cada par se generaron en
la misma corrida, con el mismo código y el mismo entorno), pero **sí da un piso de ruido medible**:
ver §5.2.

---

## 5. Resultado

### 5.1 — Comparación pareada, 12 pares

| regla | lote | frac. masa **original** | frac. masa **gemelo** | Δ (orig − gemelo) | κ_V orig | κ_V gemelo | Δκ_V |
|---|---|---|---|---|---|---|---|
| `batch4-r36` | 571828 | 0.1535 | 0.1440 | **+0.0095** | 1.408 | 1.314 | +0.094 |
| `batch3-r0` | 471828 | 0.1505 | 0.1480 | **+0.0025** | 1.294 | 1.229 | +0.064 |
| `batch4-r10` | 571828 | 0.1280 | 0.1195 | **+0.0085** | 0.903 | 0.789 | +0.114 |
| `batch3-r60` | 471828 | 0.1255 | 0.1165 | **+0.0090** | 0.910 | 0.649 | +0.262 |
| `batch3-r111` | 471828 | 0.1245 | 0.1120 | **+0.0125** | 0.980 | 0.626 | +0.354 |
| `r17` | 271828 | 0.1210 | 0.1075 | **+0.0135** | 0.797 | 0.571 | +0.226 |
| `batch4-r62` | 571828 | 0.1025 | 0.0990 | **+0.0035** | 0.567 | 0.547 | +0.020 |
| `r14` | 271828 | 0.0990 | 0.1000 | −0.0010 | 0.617 | 0.463 | +0.154 |
| `r19` | 271828 | 0.0905 | 0.0910 | −0.0005 | 0.413 | 0.564 | −0.151 |
| `r28` | 371828 | 0.0895 | 0.0830 | **+0.0065** | 0.429 | 0.400 | +0.029 |
| `r20` | 371828 | 0.0860 | 0.0885 | −0.0025 | 0.500 | 0.480 | +0.020 |
| `r39` | 371828 | 0.0785 | 0.0760 | **+0.0025** | 0.472 | 0.420 | +0.052 |

**Fracción de masa acretada (observable principal):**

| | valor |
|---|---|
| n pares | 12 |
| media original | 0.11242 |
| media gemelo | 0.10708 |
| media de la diferencia | **+0.00533** (relativo: **+5.0%**) |
| mediana de la diferencia | +0.00500 |
| el original gana | **9 / 12** |
| test de signos (binomial exacto, 2 colas) | **p = 0.146** |
| Wilcoxon de rangos con signo (2 colas) | **p = 0.0103** |

**κ_V agregado (observable secundario):**

| | valor |
|---|---|
| media original | 0.7741 |
| media gemelo | 0.6710 |
| media de la diferencia | **+0.1031** |
| el original gana | **11 / 12** |
| test de signos | **p = 0.0063** |
| Wilcoxon | **p = 0.0122** |

Número de sumideros: 8 en 22 de las 24 corridas (9 en dos). **La diferencia no está en cuántos grumos
se forman sino en cuánto come cada grumo.**

### 5.2 — Robustez: sacando los 3 pares que no replicaron su corrida de Fase V-B

| observable | n | Δ medio | gana original | signos | Wilcoxon |
|---|---|---|---|---|---|
| fracción de masa | 9 | +0.00606 | **8 / 9** | **p = 0.039** | **p = 0.0078** |
| κ_V agregado | 9 | +0.1127 | **9 / 9** | p = 0.0039 | **p = 0.0039** |

Con el subconjunto limpio el resultado **se refuerza**, no se debilita.

**Piso de ruido:** la magnitud media del efecto (|Δ| medio = 0.0060) es **≈11× la magnitud media de la
deriva de réplica** (|Δréplica| medio = 0.00054). Pero el máximo de la deriva (0.0035) es del mismo
orden que los 4 Δ más chicos de la tabla, así que los pares con |Δ| ≲ 0.0035 (`batch3-r0`, `batch4-r62`,
`r39`, y los tres negativos) **no deberían leerse individualmente**; sólo el conjunto.

### 5.3 — Contra qué se compara: la ventaja Clase III-vs-Clase I de Fase V-B

Recalculado sobre los mismos 40 pares para tener la escala:

| comparación | n | Δ medio (frac. masa) | gana el "extendido" | Wilcoxon |
|---|---|---|---|---|
| **Fase V-B**: Clase III vs Clase I (mismo K, mismo kcap) | 40 | +0.00925 | 31/40 | p = 9.2e-06 |
| **O3-B**: original vs gemelo de mismos grados | 12 | +0.00533 | 9/12 | p = 0.0103 |

| comparación | n | Δ medio (κ_V) | gana el "extendido" | Wilcoxon |
|---|---|---|---|---|
| **Fase V-B**: III vs I | 40 | +0.0911 | 28/40 | p = 0.0032 |
| **O3-B**: original vs gemelo | 12 | +0.1031 | 11/12 | p = 0.0122 |

En fracción de masa el control de reconfiguración recupera **≈58%** de la magnitud del contraste
III-vs-I. En κ_V lo recupera entero (y con mayor consistencia de signo, aunque con n más chico).

### 5.4 — Qué predice el tamaño de la ventaja (exploratorio, no pre-registrado)

| relación (Spearman, n=12) | ρ | p |
|---|---|---|
| Δ frac. masa **vs. caída de clustering** (C_orig − C_gemelo) | **+0.769** | **0.0034** |
| Δ frac. masa vs. fracción de masa del original | +0.601 | 0.039 |
| Δ frac. masa **vs. caída de pendiente** (pend_orig − pend_gemelo) | +0.042 | 0.897 |

Lectura literal: **lo que predice cuánta ventaja pierde un grafo al ser barajado es cuánto clustering
tenía para perder**, no cuánto cambió su pendiente. Los tres pares con Δ negativo o nulo
(`r14`, `r19`, `r20`) son precisamente los de clustering original más bajo del lote (0.0050, 0.0010,
0.0023 — contra 0.0105–0.0213 en los que ganan claro). Y `r19`, el de clustering más bajo de todos, es
el único par donde el barajado **aumentó** el clustering… y es también donde el gemelo ganó.

Esto es correlacional y con n=12; **no** establece que el clustering sea la causa. Pero es un candidato
mucho más concreto que "organización relacional" a secas, y es directamente testeable (p. ej. un
control que conserve grados **y** número de triángulos).

---

## 6. Qué NO dice este resultado

- **No dice que el efecto sea "todo tejido".** El gemelo, con el tejido destruido, sigue acretando el
  95% de lo que acreta el original. La mayor parte de la masa acretada la explican N, aristas y grados.
  Lo que este control aísla es el **margen** de ~5% que sobrevive al barajado.
- **No dice que la "geometría extensa" (pendiente) sea el mecanismo.** Al contrario: la pendiente
  sobrevive casi intacta al barajado (§3.3) y no predice el Δ (§5.4). Si la pendiente fuera el
  mecanismo, el gemelo debería haber empatado.
- **El test de signos sobre la fracción de masa NO alcanza el 0.05 con los 12 pares** (p=0.146). Sólo
  Wilcoxon (que pesa por cuánto ganó cada uno) baja de 0.05, y el subconjunto de 9 pares limpios llega
  a p=0.039 en signos. Con n=12 la potencia del test de signos es baja por construcción.
- **Los 12 grafos no son 12 muestras independientes en el sentido fuerte.** Vienen de 4 `seed_base`
  (3 por lote), que es exactamente la limitación que señala `FASE6_O2F_N_efectivo_fase5b_CS.md`.
- **No se declara cierre.** La lectura de hasta dónde llega esta evidencia es de Alexis.

---

## 7. Lo que quedó fabricado y disponible sin gastar cómputo nuevo

- 24 corridas de Phantom completas en `/Users/alexis/phantom_cs073/bateria_fase6_o3b_rewiring/`
  (dumps, `.sink`, `run.log`, `meta_regla.json` de cada una).
- Los 27 grafos Clase III restantes del universo de 40 están listos para gemelizar con el mismo script
  (`cs090_fase6_o3b_rewiring.py <índices>`) si se quiere escalar el control.
- El script acepta cambiar `FACTOR_SWAPS` para hacer una **curva dosis-respuesta de barajado**
  (0.1×, 0.5×, 1×, 10× aristas): permitiría ver si la ventaja se pierde de a poco con el barajado o de
  golpe. No se hizo por presupuesto.

**Costo real medido de esta tarea:** ~360 s por grafo para generar las 2 condiciones iniciales (con la
máquina cargada por otras tareas en paralelo; el costo esperado en solitario era ~50-85 s por IC),
4 shards en paralelo → ~35 min; Phantom 16–52 s por corrida → ~12 min; análisis <1 min.
