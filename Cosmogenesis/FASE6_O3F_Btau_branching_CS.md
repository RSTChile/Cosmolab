# FASE VI — O3-F: $B_\tau$, "branching efectivo de futuros" sobre el gas no colapsado

**Fecha:** 11-ago-2026 · **Origen de la propuesta:** GPT-5.6 Sol, Fase VIII (frente #13 del
`FASE6_PLAN_EJECUCION_COMPLETA_CS.md`) · **Naturaleza:** reanálisis de corridas de Phantom ya existentes.
**No se corrió Phantom nuevo. No se modificó ningún archivo previo. No se declara cierre ni veredicto.**

---

## 0. Resumen de una pantalla

| pregunta | número |
|---|---|
| ¿$B_\tau$(III) > $B_\tau$(I), tal como está definido? | Sí en 3 de 6 umbrales (24-28 de 37 pares, Wilcoxon 3.8e-4 a 1.3e-2); no en los otros 3 |
| ¿De dónde viene esa diferencia? | Del **denominador**: 97.6 % a 99.9 % de $\Delta\log B_\tau$ es $\Delta\log$(masa); 0.1 % a 2.4 % es $\Delta\log H$ |
| Con el numerador **apagado** (H := constante), ¿sobrevive el patrón? | Sí, **idéntico** (24/37 → 24/37, p 0.0121 → 0.0121; 28/37 → 27/37, p 3.8e-4 → 5.2e-4) |
| La entropía sola (numerador, con N igualado por rarefacción) | III **no** supera a I en ninguno de los 6 umbrales; en la rejilla completa (72 celdas) hay 20 con III por encima pero **ninguna con p < 0.05**, y las 11 celdas significativas apuntan **todas** a favor de I |
| ¿III tiene menos gas, como decía la premisa? | Sí: masa de gas restante III < I en 29 de 37 pares (Wilcoxon 9.2e-6) |
| ¿$B_\tau$ es una medida nueva? | Con |Ω_gas| = todo el gas restante, $\rho_{Spearman}$ = **0.968** con la fracción de masa acretada ya conocida; el signo pareado coincide en 35/37 |
| Hallazgos laterales que **sí** sobreviven a N fijo | dispersión de velocidad del gas difuso mayor en III (26/37, p=0.0059) y fracción en el mayor grupo FoF mayor en III (25/37, p=3.1e-4) |

---

## 1. Qué se quería probar y con qué límite

La propuesta de GPT-5.6 Sol es un observable distinto de los que ya se probaron: en vez de mirar cuánta masa
cayó en los sumideros, mirar **el gas que todavía NO colapsó** y preguntarle cuánta variedad de movimiento le
queda, por unidad de gas.

$$B_\tau \;\approx\; \frac{H(v_{\text{gas difuso}})}{|\Omega_{\text{gas}}|}$$

**Predicción a poner a prueba:** $B_\tau$(Clase III) > $B_\tau$(Clase I), **aunque a III le quede menos gas**
(porque acreta más). La idea es "restricción exaptativa": menos grados de libertad, más futuros posibles.

**El límite teórico, respetado en todo el documento.** Hay una decisión formal ya adoptada
(`FASE5_especificacion_universalidad_CS.md` §6): **Cosmogénesis prueba sólo el ANTECEDENTE de O-N7.7** —
"¿existe restricción histórica capaz de reducir el espacio interno de una regla sin aniquilar su capacidad
futura?". La Libertad Funcional completa pertenece a ANIMA/Célula Madre, otro plano. Nada de lo que sigue
habla de LF ni de exaptación genuina. Además, el arco previo de O-N7.7
(`ON77_sistemaAB_cierre_CS.md`) terminó con la distinción **falsada** tal como estaba operacionalizada por
$\eta_{LF}$; **esto es un intento distinto, no una repetición** de aquel.

**Falsación declarada por el analista, y encuadre exacto que corresponde reportar:** si
$B_\tau$(III) ≤ $B_\tau$(I), entonces A2-B0-C2 es **geometría extensa pero no exaptativa** — **eso no refuta
O-N7.7**, sólo dice que C2 no es el nodo exaptativo.

---

## 2. Cómo se definió $B_\tau$ — las tres piezas y sus decisiones

Analogía general: si la corrida fuera una orquesta, los sumideros son los músicos que ya se sentaron y
dejaron de tocar. $B_\tau$ pregunta cuántas melodías distintas siguen sonando entre los que quedan de pie,
dividido por cuántos quedan de pie.

Se lee, de cada corrida, el volcado final `cosmog_00500` (t = 0.5 en unidades de código) **y** la condición
inicial `cosmog_00000` (sólo como ancla de densidad). Se reusó `leer_volcado_phantom.py` (congelado) con
`sarracen` 1.3.1 del `venv` del proyecto.

### 2.1 Separación gas difuso / gas colapsado — SEIS umbrales, a propósito

Los sumideros ya son partículas aparte en Phantom; el corte de densidad saca además el gas que está cayendo
dentro de los grumos. La lección de tareas anteriores (`FASE6_O1A_kappaV_umbral_CS.md`, el borde de kcap)
es que **un umbral fijo puede fabricar el resultado**, así que se usaron tres familias:

| familia | umbral | qué controla | N_difuso (mín–máx sobre las 76 corridas) |
|---|---|---|---|
| **A** absoluta, común a todas | rho < 0.2004 (P50 agrupado) | mismo corte físico para todos | 414 – 1623 |
| | rho < 0.4962 (P75 agrupado) | | 826 – 1831 |
| | rho < 1.022 (P90 agrupado) | | 1220 – 1864 |
| **B** anclada a la propia IC | rho < 1 × mediana(rho en t=0 de esa corrida) | "difuso" = no más denso de lo que arrancó | 990 – 1093 |
| | rho < 3 × mediana(rho en t=0) | | 1207 – 1640 |
| **C** conteo fijo | las **1000** partículas menos densas de cada corrida | **N idéntico en ambos brazos por construcción**: cualquier diferencia NO puede venir de que a III le quede menos gas | 1000 – 1000 |

Los percentiles agrupados se calcularon una sola vez sobre la densidad de las 76 corridas juntas
(≈ 137.000 partículas), no corrida por corrida.

### 2.2 Entropía de la distribución de velocidades

Shannon en **bits**, con **bordes de bin fijos y globales** (percentiles de la distribución agrupada de las 76
corridas, calculados una vez por umbral), para que dos corridas sean comparables bin a bin.

- **Espacios:** (i) el vector $(v_x,v_y,v_z)$ en una grilla **4×4×4 = 64 celdas** equiprobables globalmente;
  (ii) el módulo $\log_{10}|v|$ en **16 bins**.
- **Escala:** dos versiones. `abs` sobre la velocidad cruda (mezcla dispersión + forma) y `std` sobre la
  velocidad **estandarizada por corrida** $(v-\bar v)/\sigma$ eje a eje (**sólo la forma**). La versión `std`
  es el control contra el artefacto obvio de que "si III colapsa más, su gas va más rápido y llena más bins"
  — eso sería más velocidad, no más futuros.
- **Sesgo de muestra pequeña:** la entropía estimada con pocas partículas está sesgada hacia abajo, y los dos
  brazos tienen distinto número de partículas. Se aplicó (a) corrección de **Miller-Madow** y (b)
  **rarefacción**: recalcular la entropía submuestreando *todas* las corridas al mismo N (el mínimo de la
  tanda para ese umbral), promediando **200 remuestreos** sin reemplazo. La versión rarefacción (`Hrar`) es
  la única realmente comparable entre brazos y es la que se usa en las tablas principales.

### 2.3 Número de filamentos por friends-of-friends

FoF clásico sobre las **posiciones** del gas difuso: dos partículas son amigas si están a menos de la longitud
de enlace; los filamentos son las componentes conexas. Longitud de enlace **absoluta e igual para todas las
corridas**, fijada en múltiplos $c \in \{1.0,\,1.5,\,2.5\}$ de la mediana agrupada de la distancia al vecino
más cercano del gas difuso de ese umbral (p.ej. 4.404 en A_P50, 1.330 en A_P90). Se cuenta como filamento todo
grupo con **≥ 5 miembros**, y se reporta además la **fracción de partículas en el grupo mayor** — si es ≈ 1 el
FoF percoló y el conteo pierde sentido.

### 2.4 El cociente

$|\Omega_{\text{gas}}|$ = masa del gas difuso = (nº de partículas difusas) × 9.4. La masa por partícula
(`massoftype` = 9.4) y el N inicial (2000) son **idénticos en las 76 corridas** — verificado, no supuesto.
Se reporta además una variante con $|\Omega_{\text{gas}}|$ = **todo** el gas restante (§5.3).

---

## 3. Datos: qué corridas entraron

- **40 pares** Clase I vs Clase III de A2-B0-C2 de Fase V-B, de las cinco tandas:
  `bateria_fase5b_a2b0c2_piloto` (6), `_escala_v2`, `_escala_v3`, `_escala_v4`,
  `bateria_fase6_outliers_negativos`.
- **76 corridas únicas** (algunas reglas aparecen en más de un par), todas con `cosmog_00000` y
  `cosmog_00500` presentes. Ninguna faltó; **no hizo falta correr Phantom**.
- **Se usan los 37 pares con `estado_contraste = valido`** de
  `cs090_fase6_reanalisis_40pares_corregido.csv` (tras la corrección de diámetro): de los 40, 2 quedaron
  `roto_misma_clase` y 1 `invertido`. **Sensibilidad reportada en §5.4: los 40 crudos dan el mismo cuadro.**
- Gas restante en el volcado final: entre 1693 y 1876 partículas (de 2000); 8 sumideros en las 76.

---

## 4. Resultado pareado III vs I

Δ = III − I. Positivo = la dirección predicha. Test de signos (binomial exacto) y Wilcoxon de rangos con
signo, dos colas, n = 37.

### 4.1 $B_\tau$ tal como fue propuesto (entropía rarefacción, forma 3D, / masa difusa)

| umbral | media I | media III | III>I | p signos | p Wilcoxon |
|---|---|---|---|---|---|
| A_P50_abs | 6.787e-4 | 7.222e-4 | **24/37** | 0.0989 | **0.0127** |
| A_P75_abs | 4.527e-4 | 4.695e-4 | **24/37** | 0.0989 | **0.0116** |
| A_P90_abs | 3.798e-4 | 3.900e-4 | **28/37** | **0.0026** | **3.79e-4** |
| B_IC_k1 | 6.065e-4 | 6.030e-4 | 14/37 | 0.188 | 0.0800 (signo **negativo**) |
| B_IC_k3 | 4.057e-4 | 4.043e-4 | 20/37 | 0.743 | 0.541 |
| C_N1000_fijo | 6.327e-4 | 6.323e-4 | 16/37 | 0.511 | 0.149 (signo **negativo**) |

Leído en crudo: la predicción se cumple en la familia A (umbral absoluto) y **se apaga o se invierte
levemente** en cuanto el umbral se ancla a la IC o se fija el conteo. Eso ya es una alerta de umbral; la §5
explica por qué.

### 4.2 La entropía SOLA — el numerador, sin dividir por nada

Con N igualado por rarefacción, que es la comparación honesta:

| umbral | H media I | H media III | III>I | p signos | p Wilcoxon | p permutación pareada |
|---|---|---|---|---|---|---|
| A_P50_abs | 5.8833 | 5.8827 | 17/37 | 0.743 | 0.644 | 0.866 |
| A_P75_abs | 5.9276 | 5.9259 | 15/37 | 0.324 | 0.394 | 0.282 |
| A_P90_abs | 5.9365 | 5.9325 | 15/37 | 0.324 | 0.153 | 0.197 |
| B_IC_k1 | 5.9448 | 5.9416 | 13/37 | 0.0989 | 0.121 | 0.087 |
| B_IC_k3 | 5.9320 | 5.9309 | 18/37 | 1.000 | 0.988 | 0.868 |
| C_N1000_fijo | 5.9477 | 5.9440 | 16/37 | 0.511 | 0.149 | 0.198 |

**En los 6 umbrales la media de III es menor que la de I**, y en ninguno la diferencia alcanza significación
en la dirección predicha. La permutación pareada de la etiqueta de clase (10.000 barajadas dentro de cada
par) confirma: p entre 0.087 y 0.87, todos los deltas negativos.

**La rejilla completa (6 umbrales × 4 espacios de velocidad × 3 estimadores = 72 celdas)** dice lo mismo, y
conviene decirlo con precisión en vez de redondearlo:

- **20 de 72 celdas tienen media de III por encima de la de I**, pero **ninguna de esas 20 alcanza p < 0.05**
  (la mejor llega a 21/37 pares y p = 0.21). Las 20 se concentran casi todas en el espacio `mod_abs`
  ($\log_{10}|v|$ sobre la velocidad **cruda**) — precisamente la variante que mezcla "más variedad" con
  "más velocidad", y el gas difuso de III efectivamente va más rápido (§6.1). En el espacio `std`
  (velocidad estandarizada por corrida, que es forma pura) esas celdas positivas se reducen a 5 de 24.
- **11 celdas sí alcanzan p_Wilcoxon < 0.05, y las 11 apuntan en la dirección contraria** (I > III):
  A_P50/A_P75/A_P90 `v3d_abs`, B_IC_k1 `v3d_abs`, C_N1000_fijo `v3d_abs`, y sus versiones Miller-Madow.

Detalle del estimador principal (rarefacción, N igualado), 24 celdas: 19 con delta negativo, 5 positivo; las
5 positivas están todas en `mod_abs`/`mod_std` con p ≥ 0.21; las 4 con p < 0.05 son todas a favor de I.

### 4.3 Filamentos (FoF)

| umbral | n_fil (c=1.5) I → III | III>I | p Wilcoxon |
|---|---|---|---|
| A_P50_abs | 48.7 → 44.1 | 13/37 | 0.021 (**a favor de I**) |
| A_P75_abs | 40.2 → 40.9 | 17/37 | 0.473 |
| A_P90_abs | 67.5 → 67.0 | 17/37 | 0.729 |
| B_IC_k1 | 45.2 → 43.9 | 15/37 | 0.258 |
| B_IC_k3 | 53.8 → 58.1 | 23/37 | 0.010 (a favor de III) |
| C_N1000_fijo | 45.7 → 44.8 | 15/37 | 0.317 |

El conteo de filamentos **cambia de signo con el umbral y con la longitud de enlace** (con c=1.0 y c=2.5 el
cuadro vuelve a moverse). No hay dirección estable. No se saca conclusión de este componente.

### 4.4 $|\Omega_{gas}|$ por brazo — la premisa "III tiene menos gas"

**Se confirma.** Masa total de gas restante (todo el gas, no sólo el difuso):

| medida | media I | media III | III>I | p signos | p Wilcoxon |
|---|---|---|---|---|---|
| masa de gas restante | 17078.3 | 16892.3 | 8/37 | 7.5e-4 | **9.2e-6** |
| nº de partículas de gas | 1816.8 | 1797.1 | 8/37 | 7.5e-4 | 9.2e-6 |
| fracción de masa en sumideros | 0.0916 | 0.1015 | 29/37 | 7.5e-4 | 8.9e-6 |

Y la masa de gas **difuso** también es menor en III bajo la familia A (13/37, p_Wilcoxon 0.0069 / 0.0088;
10/37, p 3.9e-4), pero **no** bajo la familia B (22/37, p 0.13; 15/37, p 0.51) ni, por construcción, bajo C.
Ese detalle es la bisagra de todo lo que sigue.

---

## 5. El control central: ¿es $B_\tau$ un artefacto de la normalización?

Analogía: si mido "libros por estante" y a un estante le saco estantes, el cociente sube sin que aparezca un
solo libro nuevo. La pregunta es si aparecieron libros.

### 5.1 Descomposición $\log B_\tau = \log H - \log M$

Diferencia pareada media (III − I):

| umbral | Δlog $B_\tau$ | Δlog H | Δlog M | aporte del numerador | aporte del denominador |
|---|---|---|---|---|---|
| A_P50_abs | +7.60e-2 | −1.08e-4 | −7.61e-2 | **0.1 %** | **99.9 %** |
| A_P75_abs | +3.94e-2 | −2.86e-4 | −3.97e-2 | **0.7 %** | **99.3 %** |
| A_P90_abs | +2.68e-2 | −6.78e-4 | −2.75e-2 | **2.4 %** | **97.6 %** |
| B_IC_k1 | −5.69e-3 | −5.33e-4 | +5.16e-3 | 9.4 % | 90.6 % |
| B_IC_k3 | −2.27e-3 | −1.92e-4 | +2.08e-3 | 8.5 % | 91.5 % |
| C_N1000_fijo | −6.32e-4 | −6.32e-4 | 0 | 100 % | 0 % (y el signo es **negativo**) |

### 5.2 $B_\tau$ postizo: apagar el numerador

Se reemplaza H por su **media global constante** y se rehace el test pareado. Si el patrón sobrevive con el
numerador apagado, el observable no estaba midiendo entropía:

| umbral | $B_\tau$ real III>I | p Wilcoxon | $B_\tau$ **postizo** III>I | p Wilcoxon |
|---|---|---|---|---|
| A_P50_abs | 24/37 | 0.0127 | **24/37** | **0.0121** |
| A_P75_abs | 24/37 | 0.0116 | **24/37** | **0.0121** |
| A_P90_abs | 28/37 | 3.79e-4 | **27/37** | **5.21e-4** |
| B_IC_k1 | 14/37 | 0.080 | 15/37 | 0.121 |
| B_IC_k3 | 20/37 | 0.541 | 22/37 | 0.502 |
| C_N1000_fijo | 16/37 | 0.149 | (indefinido: denominador constante) | — |

**El patrón sobrevive intacto con el numerador congelado.** En la familia A, $B_\tau$ es 1/masa disfrazado.

### 5.3 $B_\tau$ contra lo que ya se sabía

Correlación de $B_\tau$ con la **fracción de masa acretada**, un punto por brazo (74 observaciones):

| umbral | Pearson r | Spearman ρ |
|---|---|---|
| A_P50_abs | +0.801 (p 1.1e-17) | +0.783 |
| A_P75_abs | +0.807 (p 3.8e-18) | +0.776 |
| A_P90_abs | +0.871 (p 5.7e-24) | +0.843 |
| B_IC_k1 | −0.563 | −0.541 |
| C_N1000_fijo | −0.505 | −0.502 |

Y con $|\Omega_{gas}|$ = **todo** el gas restante (la lectura más literal de "gas no colapsado"):
$B_\tau$(III) > $B_\tau$(I) en 28-31 de 37 pares, p_Wilcoxon 5e-7 a 5e-6 — pero **Spearman = 0.968** entre
Δ$B_\tau$ y Δ(fracción de masa acretada), con **el mismo signo en 35 de 37 pares**. Es decir: así definido,
$B_\tau$ **no es una medida nueva**, es una reescritura monótona del resultado de masa-en-sumideros que ya
estaba en `FASE5B_escala_40pares_CS.md`.

### 5.4 Dos chequeos más

- **ANCOVA H ~ N_difuso + clase:** el coeficiente de clase III es **negativo en los 6 umbrales**, con
  p entre 0.12 y 0.79. No queda efecto de clase sobre la entropía una vez descontado el tamaño.
- **37 válidos vs 40 crudos:** mismo cuadro (A_P90: 28/37 p 3.8e-4 → 30/40 p 4.0e-4; C_N1000: 16/37 p 0.149 →
  18/40 p 0.197). La corrección de diámetro no cambia nada acá.

---

## 6. Dos cosas que sí sobreviven al control de N fijo (no son $B_\tau$)

No estaban en la pregunta, pero aparecen limpias y con signo estable, así que quedan anotadas como números,
sin interpretación:

1. **Dispersión de velocidad del gas difuso, σ_v: mayor en III.** C_N1000_fijo 26/37 (p_Wilcoxon 0.0059,
   permutación 0.084); B_IC_k1 27/37 (p 0.0030, permutación 0.030); B_IC_k3 (p 0.0145, permutación 0.029).
   O sea: al gas que le queda a III se mueve **más rápido**, pero — y ése es el punto — esa velocidad extra
   **no se traduce en más variedad** (la entropía de la forma, `std`, no sube).
2. **Concentración local del gas difuso: mayor en III.** La fracción de partículas en el mayor grupo FoF con
   la longitud de enlace corta (c = 1.0) es mayor en III en 4 de 6 umbrales: C_N1000_fijo 25/37 (p 3.1e-4),
   B_IC_k1 31/37 (p 1.3e-5), A_P75 25/37 (p 0.0051), B_IC_k3 22/37 (p 0.031). Con enlaces más largos el signo
   se pierde.

---

## 7. Encuadre exacto que pidió el analista

Con la definición operacional de esta tarea:

- La entropía de velocidades del gas difuso — **el numerador, que es la parte que pretende medir "futuros
  posibles"** — no alcanza significación a favor de Clase III en **ninguna** de las 72 celdas
  (6 umbrales × 4 espacios de velocidad × 3 estimadores), ni con corrección de sesgo, ni con N igualado por
  rarefacción, ni bajo permutación pareada de la etiqueta de clase. Las 11 celdas que sí alcanzan p < 0.05
  apuntan todas en la dirección opuesta.
- El cociente $B_\tau$ **sí** sale mayor en III bajo umbral absoluto, pero el 97.6-99.9 % de esa diferencia
  viene del denominador, sobrevive intacta al apagar el numerador, y correlaciona ρ = 0.968 con la fracción
  de masa acretada ya conocida.

Por lo tanto, **por el criterio de falsación que declaró el analista**: $B_\tau$(III) ≤ $B_\tau$(I) una vez
descontada la normalización ⇒ **A2-B0-C2 es geometría extensa pero no exaptativa** — y, con el encuadre
exacto que él mismo fijó, **eso no refuta O-N7.7; sólo dice que C2 no es el nodo exaptativo.**

Se subraya, además, el límite de §6 de `FASE5_especificacion_universalidad_CS.md`: acá se estaba probando
**sólo el antecedente** de O-N7.7. Este resultado dice que **esta operacionalización particular del
antecedente** (entropía de velocidad del gas residual, normalizada por masa) no lo encuentra. No dice nada
sobre la Libertad Funcional, que pertenece a ANIMA/Célula Madre.

**No se declara cierre. La interpretación es de Alexis.**

---

## 8. Lo que queda abierto (anotado, no resuelto)

- La propuesta original de §6 de la especificación de Fase V hablaba de **generar un pequeño ensamble de
  continuaciones bajo perturbaciones mínimas** desde el snapshot y medir cuántos futuros distinguibles
  quedan. Eso es un $B_\tau$ **dinámico** y requiere correr Phantom de nuevo (N réplicas perturbadas por
  brazo desde `cosmog_00500`). Lo de esta tarea es el proxy **estático** que propuso GPT-5.6 Sol para
  hacerlo con datos existentes. El proxy estático no encuentra la señal; **si eso es propiedad del proxy o de
  la Clase III, no se puede distinguir con estos datos.** Correr el $B_\tau$ dinámico sería el paso natural,
  y no se hizo acá porque la tarea pedía explícitamente reanálisis.
- σ_v mayor en III con entropía igual: es la firma de "más caliente, no más variado". No se sabe si eso es
  informativo o sólo energía de caída.
- La concentración local mayor en III (§6.2) apunta en dirección contraria al "branching": más agrupado, no
  más ramificado.

---

## 9. Archivos producidos (todos nuevos)

| archivo | qué es |
|---|---|
| `cs090_fase6_o3f_extraer_gas.py` | paso 1: lee los 152 volcados (IC + final) de las 76 corridas y cachea el crudo |
| `cs090_fase6_o3f_cache/*.npz` | caché de posiciones/velocidades/densidades por corrida (76 archivos) |
| `cs090_fase6_o3f_corridas_leidas.csv` | inventario de lo leído (N gas, sumideros, masa acretada) |
| `cs090_fase6_o3f_btau.py` | paso 2: define los 6 umbrales, calcula H (4 variantes × 3 correcciones), FoF y $B_\tau$; tests pareados |
| `cs090_fase6_o3f_btau_crudo.csv` | **CSV crudo**: una fila por (corrida × umbral), todas las métricas |
| `cs090_fase6_o3f_btau_pares.csv` | una fila por (par × umbral): brazo I, brazo III |
| `cs090_fase6_o3f_btau_tests.csv` | test de signos + Wilcoxon por (umbral × métrica) |
| `cs090_fase6_o3f_correlaciones.csv` | correlaciones con la fracción de masa acretada |
| `cs090_fase6_o3f_control_normalizacion.py` | paso 3: descomposición log, $B_\tau$ postizo, ANCOVA, permutación pareada, sensibilidad 37 vs 40 |
| `cs090_fase6_o3f_control_normalizacion.csv` | resultados del paso 3 |
| `cs090_fase6_o3f_figura.py` / `cs090_fase6_o3f_btau.png` | figura de 3 paneles |

**Reproducibilidad:** semilla fija 20260811 para rarefacción (200 remuestreos) y permutaciones (10.000).
Intérprete: `./venv/bin/python` (sarracen 1.3.1).
