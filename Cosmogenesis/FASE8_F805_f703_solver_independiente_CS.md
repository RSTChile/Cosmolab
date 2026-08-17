# FASE VIII · F8-05 — F7-03 en un integrador **independiente** de Phantom

**12 de agosto de 2026** · **Ejecuta:** CC (Claude) · Ola 1 del `FASE8_PLAN_EJECUCION_CS.md`
**Lo que se valida:** `FASE7_F703_grados_y_triangulos_fijos_CS.md` — con grados idénticos nodo por nodo
**y el mismo número de triángulos**, `solap` supera a `disj` en **+0.01433 de fracción de masa
(+28.7 partículas, 12/12 grafos, Wilcoxon p = 4.9e-04)**.
**El motor usado:** `cs090_fase6_o4a_nbody.py`, el integrador validado en `FASE6_O4A_solver_independiente_CS.md`
— importado tal cual, sin tocar una línea.

> **No se corrió Phantom.** Las 60 corridas de F7-03 ya existían y se leyeron de disco.
> No se modificó ningún script existente. No se declara cierre ni veredicto. Sin commits.

---

## 0. En simple, con analogía

F7-03 dijo: con la misma cantidad de nudos, el mismo número de alambres en cada nudo y **exactamente los
mismos triangulitos**, la maqueta que los tiene **apilados compartiendo varilla** junta más arena que la
que los tiene **sueltos y separados**. Doce maquetas de doce.

Ese resultado salió de un solo simulador. Esta tarea lo vuelve a preguntar con **otro simulador**: uno
escrito desde cero, con otra física (gravedad y nada más — sin presión, sin viscosidad, sin sumideros que
se traguen partículas), otro algoritmo y otro lenguaje. Si dos relojes hechos por relojeros distintos
dicen que ganó el mismo corredor, la victoria es del corredor.

**Y salió algo que no estaba en el guion.** Resulta que la pregunta *"¿coinciden los motores?"* estaba mal
planteada: lo que casi decide la respuesta no es **qué motor** se usa, sino **con qué vara se mide**.
Cuando se le aplica al **estado final del propio Phantom** la vara del motor independiente, **Phantom se
contradice a sí mismo**. Y cuando se le aplica al motor independiente la vara que Phantom realmente usa —
un umbral de densidad altísimo— los dos motores coinciden.

---

## 1. Qué se hizo, con qué archivos

| Archivo nuevo | Qué hace |
|---|---|
| `cs090_fase8_f805_correr.py` | Localiza y verifica las 24 condiciones iniciales de F7-03, las integra con el motor independiente, mide el observable análogo en t=0 y en t=0.5 |
| `cs090_fase8_f805_analizar.py` | Une con las corridas de Phantom por `(rule_id, seed, brazo)`, concordancia par por par, correlaciones, tamaños de efecto, y la **misma vara aplicada al estado final de Phantom** |
| `cs090_fase8_f805_umbrales.py` | Barrido de umbrales de densidad de 10× a 49.453× la media (= el umbral real de sumidero de Phantom) |
| `cs090_fase8_f805_figura.py` | Los 6 paneles del dibujo |

| CSV / PNG | Contenido |
|---|---|
| `cs090_fase8_f805_corridas_nbody.csv` | **CSV crudo pedido**: una fila por corrida (24), diagnóstico + 17 observables en t=0 y en t=0.5 |
| `cs090_fase8_f805_unido.csv` | lo mismo, unido a las métricas de Phantom y del grafo |
| `cs090_fase8_f805_pares.csv` | una fila por grafo, con `solap`, `disj` y su diferencia en cada observable |
| `cs090_fase8_f805_robustez_grilla.csv` | la cuenta repetida en las 17 definiciones de "región densa" |
| `cs090_fase8_f805_correlaciones.csv` | Phantom contra cada observable propio, 24 corridas |
| `cs090_fase8_f805_efectos.csv` | signos, Δ, %, Wilcoxon y t pareado de cada observable |
| `cs090_fase8_f805_vara_comun.csv` | **la misma vara sobre el estado final de Phantom** (gas vivo + sumideros con su masa) |
| `cs090_fase8_f805_umbrales{,_crudo}.csv` | la escalera de umbrales de densidad |
| `cs090_fase8_f805_comparacion.png` | los 6 paneles |

**Sólo importados, nunca modificados:** `cs090_fase6_o4a_nbody.py`, `cs090_fase6_o4a_correr.py` (indirecto).
Logs: `cs090_fase8_f805_{correr,analisis,umbrales}.log`.

---

## 2. Las condiciones iniciales: se reusaron, no se regeneraron

**Ruta:** `/Users/alexis/phantom_cs073/bateria_fase7_f703_organizacion/<rule_id>_s<seed>_f703_{solap,disj}/cosmogenesis_ic.txt`

Son **exactamente los mismos archivos que consumió Phantom**. Ese es todo el punto del experimento: los
dos motores tienen que recibir el mismo input byte a byte, si no la comparación no dice nada del motor.

Verificaciones hechas **antes** de integrar nada (todas con `assert`, el programa aborta si fallan):

| control | resultado |
|---|---|
| el `meta_regla.json` de cada carpeta declara la tarea `FASE7_F703_organizacion_triangulos` | 24/24 |
| el brazo y el `(rule_id, seed)` del meta coinciden con el nombre de la carpeta | 24/24 |
| la carpeta declarada **dentro** del meta es la carpeta donde está el meta | 24/24 |
| el meta declara `grados_identicos_al_original = true` | 24/24 |
| los 12 grafos tienen **los dos** brazos | 12/12 |
| `npart = 2000` y masa por partícula `= 9.4` en la cabecera del IC | 24/24 |
| **md5 distintos** (no se corrió 24 veces lo mismo) | **24 md5 distintos de 24** |

Se corrieron los **12 grafos × 2 brazos = 24** corridas. Los otros tres brazos de F7-03 (`libre`, `conc`,
`disp`) no entran: el contraste que la tarea manda validar es `solap` − `disj`.

### 2.1 Salud numérica de las 24 corridas

Parámetros: los ya validados en O4-A — `dt = 2e-3` (250 pasos), `t_final = 0.5` (el `tmax` de Phantom),
suavizado de Plummer `eps = 0.6` (= `r_crit` de Phantom).

| diagnóstico | resultado |
|---|---|
| deriva relativa de energía | **−8.2e−06 … −8.2e−07** en las 24 (el leapfrog de paso fijo es simpléctico: oscila, no deriva) |
| error relativo del momento lineal | **máx 1.8e−14** — o sea, exacto al redondeo, como debe ser en gravedad aislada |
| tiempo de reloj | 515 s las 24 corridas con 6 procesos (77–186 s cada una, la máquina con carga ~9) |

---

## 3. El resultado pre-declarado: **NO reproduce** — y por qué eso no es lo que parece

El observable principal se declaró de antemano, copiado de O4-A: **friends-of-friends con `ell = 1.0` y
grupos de al menos 5 partículas**, medido en t = 0.5.

| | Δ (`solap` − `disj`) | en partículas | signos | Wilcoxon |
|---|---|---|---|---|
| **PHANTOM** (fracción de masa en sumideros) | **+0.01433** | **+28.7** | **12/12** | **4.9e−04** |
| **motor independiente**, FoF ell=1.0 n≥5 (pre-declarado) | **−0.01458** | **−29.2** | **4/12** | 0.064 |
| **IC sin integrar**, FoF ell=1.0 n≥5 | −0.01813 | −36.3 | 4/12 | 0.129 |

**Concordancia de orden par por par: 4 de 12** (binomial p = 0.39). Con ese observable, el motor
independiente no sólo no reproduce el orden: lo **invierte**.

### 3.1 El control que reencuadra todo: la misma vara sobre el estado final de **Phantom**

Antes de concluir nada sobre motores, se hizo el control que O4-A §5.2 dejó armado: reconstruir el estado
final **del propio Phantom** como nube de masas puntuales (gas vivo a 9.4 c/u **más** los sumideros con la
masa que acretaron) y aplicarle **exactamente el mismo friends-of-friends**.

| vara aplicada al **mismo estado final de Phantom** | Δ (`solap` − `disj`) | signos | Pearson Phantom↔motor (24 corridas) | orden Phantom↔motor |
|---|---|---|---|---|
| **masa en sumideros** (la vara oficial de F7-03) | **+0.01433** | **12/12** | — | — |
| FoF `ell = 0.3` | +0.01325 | **12/12** | +0.816 | 4/12 |
| FoF `ell = 0.45` | +0.00587 | 7/12 | +0.898 | 8/12 |
| FoF `ell = 0.6` | −0.00267 | 5/12 | +0.930 | 9/12 |
| **FoF `ell = 1.0`** (el pre-declarado) | **−0.02275** | **2/12** | **+0.956** | 8/12 |
| FoF `ell = 2.0` | −0.01263 | 1/12 | +0.982 | 12/12 |

**Léalo despacio: es la misma corrida de Phantom, los mismos archivos, y el signo se da vuelta según con
qué vara se la mida.** Con `ell = 1.0` **Phantom se contradice a sí mismo**: 12/12 a favor de `solap`
medido como masa en sumideros, 2/12 a favor de `solap` medido como masa en grupos FoF.

Y en esa misma vara, los dos motores **sí** coinciden: Pearson **+0.956** sobre los valores absolutos de
las 24 corridas, y el orden en 8/12. O sea: **el desacuerdo del §3 no es entre motores, es entre reglas de
medición.**

### 3.2 Por qué la vara `ell = 1.0` mide otra cosa

La separación media entre partículas es 7.75, así que `ell = 1.0` parece exigente. No lo es en estas nubes:
en t = 0 el FoF a `ell = 1.0` ya agrupa el **40–42 %** de la masa (estas condiciones iniciales nacen mucho
más apelotonadas que las de Fase V-B). A esa longitud de enlace el observable no mide "grumos": mide
**la telaraña que percola**.

Los conteos de grupos lo dicen sin ambigüedad:

| definición | nº de grupos `solap` | nº de grupos `disj` | qué significa |
|---|---|---|---|
| FoF ell=1.0, **n≥3** | 195.3 | **208.4** | `disj` tiene **más** grumitos chicos |
| FoF ell=1.0, **n≥5** | 70.7 | **79.8** | idem |
| FoF ell=1.0, **n≥10** | **9.00** | 9.42 | los grumos **grandes** son los mismos ~9 en ambos |

`solap` fragmenta más el grafo (componente gigante 1843 contra ~1880 en F7-03) y concentra: **menos
grumitos, pero más apretados**. `disj` reparte: **más grumitos, más chicos**. Si uno cuenta "masa en
cualquier cosa que sea al menos un trío", gana `disj`. Si uno cuenta "masa en las cosas grandes y densas",
gana `solap`.

**Phantom cuenta lo segundo.** Un sumidero no es un trío: nace cuando una región cruza `rho_crit`, y el
número de sumideros **no cambia entre brazos** (8.08 contra 8.08 en F7-03) — cambia cuánto come cada uno.

---

## 4. Con la vara correcta, el motor independiente **sí** reproduce el orden

Todas las definiciones que exigen **grumos grandes** o **densidad alta** — que es lo que Phantom exige —
van en la dirección de Phantom:

| observable del motor independiente (t = 0.5) | Δ | en partículas | % | signos | Wilcoxon | orden vs Phantom | r dentro de grafo |
|---|---|---|---|---|---|---|---|
| **densidad local > 1000× la media** | +0.00883 | **+17.7** | +9.0 % | 9/12 | **0.0049** | **9/12** | **+0.946** |
| **densidad local > 100× la media** | +0.01088 | **+21.8** | +8.0 % | 9/12 | 0.049 | **9/12** | **+0.841** |
| **FoF ell=1.0, n≥10** | +0.01354 | **+27.1** | +10.6 % | 10/12 | 0.012 | **10/12** | **+0.853** |
| FoF ell=0.6, n≥10 | +0.00821 | +16.4 | +8.7 % | 10/12 | 0.042 | 10/12 | +0.805 |
| FoF ell=0.45, n≥10 | +0.00650 | +13.0 | +8.1 % | 10/12 | 0.0039 | 10/12 | +0.792 |
| — *el pre-declarado* — FoF ell=1.0, n≥5 | −0.01458 | −29.2 | −3.3 % | 4/12 | 0.064 | 4/12 | −0.360 |
| **PHANTOM, referencia** | **+0.01433** | **+28.7** | **+13.8 %** | **12/12** | **4.9e−04** | — | — |

(«% » con la misma convención de F7-03: media de los porcentajes grafo por grafo. Como cociente de medias,
Phantom da +12.6 %.)

**Y no es sólo el signo: los tamaños de efecto se siguen entre sí par por par.**

| | Spearman entre el Δ de Phantom y el Δ del motor, a través de los 12 pares |
|---|---|
| densidad > 100× | **+0.827** (p = 0.0009) |
| densidad > 1000× | **+0.776** (p = 0.0030) |
| FoF ell=1.0, n≥10 | +0.694 (p = 0.012) |

Los grafos donde Phantom ve más diferencia son los mismos donde el motor independiente ve más diferencia.
Panel B de la figura.

### 4.1 Las dos firmas secundarias de F7-03 también se repiten

1. **No se forman más grumos, cada grumo come más.** En el motor independiente, con criterio n≥10, los
   grupos son **8.00 vs 7.83** (ell=0.3), **8.25 vs 8.08** (ell=0.45), **8.33 vs 8.42** (ell=0.6) —
   indistinguibles, igual que los 8.08 vs 8.08 sumideros de Phantom. Lo que cambia es la masa dentro.
2. **El efecto crece con T\*** (los triángulos disponibles para repartir). Phantom: ρ = +0.818. Motor
   independiente con densidad >100×: **ρ = +0.778** (p = 0.0029). Con FoF ell=1.0 n≥10: **ρ = +0.876**
   (p = 0.0002). Panel D.

---

## 5. **El control que exige O4-A: los tres números**

Este es el punto que la consigna pide decir "con todas las letras".

| | Δ (`solap` − `disj`) | en partículas | signos | Wilcoxon |
|---|---|---|---|---|
| **① IC SIN INTEGRAR** (t = 0, densidad > 100×) | **+0.01904** | **+38.1** | **12/12** | **4.9e−04** |
| **② MOTOR INDEPENDIENTE** (t = 0.5, densidad > 100×) | +0.01088 | +21.8 | 9/12 | 0.049 |
| **③ PHANTOM** (masa en sumideros) | +0.01433 | +28.7 | **12/12** | 4.9e−04 |

Con las varas "sueltas", **la advertencia de O4-A se confirma y se agrava**: la geometría de partida no
sólo ya contiene el ordenamiento — lo contiene **más fuerte** que el resultado integrado. Los números que
lo dicen:

| relación (24 corridas) | Pearson |
|---|---|
| Phantom ↔ **IC sin integrar** (FoF ell=0.45, n≥5) | **+0.992** |
| Phantom ↔ **IC sin integrar** (densidad > 100×) | **+0.984** |
| Phantom ↔ motor independiente (FoF ell=1.0, n≥10) | +0.962 |
| Phantom ↔ motor independiente (densidad > 1000×) | +0.957 |
| motor independiente ↔ IC (mismo observable) | +0.943 |
| **parcial: Phantom ↔ motor independiente, descontando la IC** | **+0.111 (p = 0.61)** |

Y en concordancia de signo, la IC sola da **12/12** en varias definiciones (densidad>100×: 12/12;
FoF ell=0.6 n≥10: 12/12; FoF ell=0.45 n≥5: 12/12) — **tan bien o mejor que integrando**.

### 5.1 Pero hay un piso a esa lectura, y es el número más interesante de la tarea

Phantom no crea sumideros a "100× la densidad media". El `cosmog.in` de esta línea dice
`rho_crit_cgs = 1000`, y la densidad media de estas cajas es 18800/97.6³ = 0.0202. Es decir:
**el umbral real de Phantom está 49.453 veces por encima de la densidad media.** Ninguno de los
observables de arriba llegaba ni cerca.

Se barrió la escalera completa sobre las mismas 24 corridas (`cs090_fase8_f805_umbrales.py`):

| umbral (× densidad media) | masa sobre el umbral | **IC sin integrar** Δ / signos | **motor independiente** Δ / signos | lo que puso la dinámica | orden vs Phantom (fin) |
|---|---|---|---|---|---|
| 10× | 0.540 | −0.0338 / **1**/12 | −0.0282 / 0/12 | +0.0056 | 0/12 |
| 30× | 0.271 | +0.0038 / 7/12 | −0.0053 / 5/12 | −0.0091 | 5/12 |
| **100×** | 0.179 | **+0.0190 / 12/12** | +0.0109 / 9/12 | **−0.0082** | 9/12 |
| 300× | 0.143 | +0.0061 / 12/12 | +0.0127 / 10/12 | +0.0066 | 10/12 |
| 1.000× | 0.114 | +0.0020 / 10/12 | +0.0088 / 9/12 | +0.0069 | 9/12 |
| 3.000× | 0.087 | +0.0035 / 9/12 | +0.0038 / 9/12 | +0.0003 | 9/12 |
| 10.000× | 0.064 | **0.0000 / 0 de masa** | +0.0048 / 9/12 | **+0.0048** | 9/12 |
| 30.000× | 0.040 | **0.0000 / 0 de masa** | +0.0044 / 8/12 | **+0.0044** | 8/12 |
| **49.453× = el umbral real de Phantom** | **0.028** | **0.0000 / 0 de masa** | **+0.0070 / 9/12** (p = 0.0088) | **+0.0070 (el 100 %)** | **9/12** |

**En las condiciones iniciales no existe ni una sola partícula por encima de 10.000× la densidad media —
en las 24, `solap` y `disj` por igual.** A la densidad que Phantom exige para encender un sumidero, la
geometría de partida **no tiene nada que decir**: el contraste entero lo pone la gravedad.

Traducido con analogía: si uno pregunta "¿cuánta harina quedó en montoncitos?", la respuesta ya estaba
escrita antes de amasar (①). Si uno pregunta "¿cuánta harina quedó **tan apretada que ya es masa**", en el
bol crudo la respuesta es **cero en las dos recetas**, y toda la diferencia aparece amasando. Phantom
pregunta lo segundo.

El Δ a ese umbral es **+14.0 partículas** contra un grano de 1 — modesto comparado con las +28.7 de
Phantom (que además acreta, y por eso mueve más masa), pero **catorce veces la resolución del
instrumento**, con 9/12 y Wilcoxon p = 0.0088, y correlacionado con el Δ de Phantom (Spearman +0.586,
p = 0.045) y con T\* (ρ = +0.579, p = 0.049).

---

## 6. Los cuatro números que pidió la consigna

1. **Concordancia par por par:** depende de la vara, y esa dependencia es el hallazgo.
   - con el observable pre-declarado (FoF ell=1.0, n≥5): **4/12** — pero **Phantom medido con esa misma
     vara da 2/12**, o sea que la vara, no el motor, es lo que falla;
   - con observables de densidad o de grumo grande (los análogos reales del criterio de sumidero de
     Phantom): **9/12 y 10/12**, en 5 definiciones distintas;
   - a **49.453× la densidad media** — el umbral literal de Phantom: **9/12**, p = 0.0088.
2. **Correlación entre motores** (24 corridas, valores absolutos): **+0.96** (FoF ell=1.0 n≥10),
   **+0.957** (densidad >1000×), **+0.956** con la vara común sobre el estado final de Phantom. Con el
   observable pre-declarado, +0.804. En tamaño de efecto par por par, Spearman **+0.83**.
3. **Tamaño del efecto en el motor independiente:** **+17.7 a +27.1 partículas** (+8 % a +10.6 %) según la
   definición, contra las **+28.7 partículas (+13.8 %)** de Phantom. Es decir, **entre el 62 % y el 94 %
   del efecto de Phantom**, con un motor que no tiene sumideros ni acreción — los dos ingredientes que en
   Phantom amplifican la masa capturada.
4. **Las IC sin integrar:** en las varas sueltas **ya contienen el ordenamiento completo (12/12, +38.1
   partículas, más fuerte que el resultado integrado)**, y la parcial Phantom↔motor descontando la IC cae a
   **+0.11 (p = 0.61)**. Pero **a la densidad que Phantom realmente usa las IC no contienen absolutamente
   nada (0 partículas por encima del umbral, en las 24)** y el 100 % del contraste lo produce la dinámica.

---

## 7. Las dos lecturas, las dos sobre la mesa

**Lectura A — el +13.8 % no es un artefacto de Phantom.**
Un integrador escrito desde cero, con otra física (sin presión, sin viscosidad, sin sumideros), otro
algoritmo de gravedad (suma directa contra árbol), otro esquema temporal (paso global fijo contra pasos
individuales) y otro lenguaje, alimentado con **los mismos archivos de entrada**, ordena `solap` por encima
de `disj` en 9 o 10 de 12 grafos, en cinco definiciones distintas de "región densa", con un tamaño de
efecto que sigue par por par al de Phantom (Spearman +0.83) y que reproduce sus dos firmas secundarias
(mismo número de grumos, efecto que crece con T\*). Y en el punto donde la comparación es más exigente —el
umbral de densidad literal de Phantom— coincide 9/12 con p = 0.0088 y **sin nada heredado del punto de
partida**.

**Lectura B — parte de lo que parecía dinámica es geometría de partida.**
Con las varas más sueltas (densidad >100×, FoF de grumo grande a ell ≤ 0.6), las condiciones iniciales por
sí solas ya separan `solap` de `disj` 12/12 con un efecto **mayor** que el integrado, correlacionan con
Phantom a r = +0.98/+0.99, y la parcial Phantom↔motor descontando la IC cae a +0.11. En ese régimen, "dos
motores coinciden" no habla de la gravedad: habla de que ambos heredaron la misma nube inicial, y la nube
inicial ya venía ordenada por el `layout_resortes`.

**Y una advertencia que no estaba en el guion, que aplica a toda la línea:**
el observable elegido puede **invertir el signo del resultado sobre las mismísimas corridas de Phantom**.
`solap` − `disj` sobre el dump `cosmog_00500` es **+0.01433 (12/12)** si se mide como masa en sumideros,
**+0.01325 (12/12)** si se mide con FoF a ell = 0.3, y **−0.02275 (2/12)** si se mide con FoF a ell = 1.0.
Esto no cuestiona F7-03 — la vara de F7-03 (masa en sumideros) es la que corresponde a la pregunta física
y la que se viene usando en toda la línea — pero sí dice que **"cuánta masa quedó apelotonada" no es una
cantidad, son varias**, y que `solap` gana en la parte densa mientras pierde en la parte suelta.

Cuál pesa más depende de qué se esté afirmando:
- *"el contraste `solap`/`disj` sobrevive al cambio de motor"* → **esta tanda lo apoya**, sobre todo en el
  régimen de densidad que importa;
- *"la gravedad convierte la organización de los triángulos en masa colapsada"* → **lo apoya sólo en el
  régimen denso** (donde las IC no tienen nada y toda la separación es dinámica), y **no lo apoya** en el
  régimen suelto, donde el ordenamiento ya venía puesto en el layout.

No declaro cierre. La interpretación es de Alexis.

---

## 8. Lo que esta tanda NO puede decidir

1. **El observable principal se declaró antes, y falló.** El honesto es decirlo así: la elección
   pre-declarada (FoF ell=1.0, n≥5, copiada de O4-A) da 4/12. Que las otras definiciones den 9-10/12 es
   evidencia **posterior a mirar la grilla**. Lo que la rescata de ser cherry-picking es (a) que la grilla
   completa estaba escrita en el script antes de correr nada, (b) que el fracaso de la vara pre-declarada
   se explica por un control independiente —Phantom medido con esa vara también falla (2/12)— y (c) que las
   varas que funcionan son las que corresponden al criterio físico que Phantom usa (umbral de densidad),
   no las que dan el número más lindo. Aun así, **un pre-registro limpio requeriría repetir esto con
   "densidad > umbral de Phantom" declarado de antemano en una batería nueva.**
2. **`eps = 0.6` no se barrió.** El suavizado de Plummer pone un piso a cuánto pueden colapsar los grumos,
   y por lo tanto acota directamente el observable de densidad alta. Un `eps` menor dejaría llegar más
   arriba en la escalera. Es la limitación más obvia de la §5.1.
3. **El motor no acreta.** Phantom se lleva masa a sumideros; acá los grumos son grumos. Los valores
   absolutos no son comparables y el efecto del motor propio es sistemáticamente menor. Eso es esperable,
   no es evidencia en contra.
4. **12 grafos, todos del linaje A2-B0-C2, Clase III.** Igual que F7-03. No se sabe si vale afuera.
5. **`solap` fragmenta más el grafo** (gigante 1843 vs 1880). F7-03 controló ese confound con parciales y
   con el brazo `conc` (que no fragmenta y aun así gana); acá no se repitió ese control, porque el diseño
   sólo trajo `solap` y `disj`.
6. **No se probó ninguna hipótesis sobre *por qué***. Que el contraste sólo exista arriba de 10.000× la
   densidad media es un dato nuevo y sugerente —parece que el apretamiento local no cambia cuánta materia
   se junta, sino **cuán lejos llega el colapso de la que ya se juntó**— pero es conjetura, no medición.

---

## 9. Costos

24 corridas × 250 pasos, O(N²) a N=2000: **515 s** de reloj con 6 procesos (77–186 s cada una, máquina con
carga ~9). El barrido de umbrales re-integró las mismas 24 (el integrador es determinista): otros ~600 s.
Análisis + vara común (lectura de 24 dumps con sarracen): ~90 s. Nada de esto tocó Phantom.

---

## Archivos

**Nuevos (esta tarea):** `cs090_fase8_f805_correr.py`, `cs090_fase8_f805_analizar.py`,
`cs090_fase8_f805_umbrales.py`, `cs090_fase8_f805_figura.py`,
`cs090_fase8_f805_corridas_nbody.csv`, `cs090_fase8_f805_unido.csv`, `cs090_fase8_f805_pares.csv`,
`cs090_fase8_f805_robustez_grilla.csv`, `cs090_fase8_f805_correlaciones.csv`,
`cs090_fase8_f805_efectos.csv`, `cs090_fase8_f805_vara_comun.csv`,
`cs090_fase8_f805_umbrales.csv`, `cs090_fase8_f805_umbrales_crudo.csv`,
`cs090_fase8_f805_comparacion.png`, `cs090_fase8_f805_{correr,analisis,umbrales}.log`.

**Sólo importados, nunca modificados:** `cs090_fase6_o4a_nbody.py`.

**Leídos, nunca escritos:** `cs090_fase7_f703_phantom_crudo.csv` y la batería
`/Users/alexis/phantom_cs073/bateria_fase7_f703_organizacion/` (24 `cosmogenesis_ic.txt`,
24 `meta_regla.json`, 24 dumps `cosmog_00500`).

> Sin cierre, sin veredicto, sin commits.
