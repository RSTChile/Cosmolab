# F7-05 — La mediación rehecha: qué camino sobrevive cuando los demás quedan condicionados

**Fecha:** 12-ago-2026 · Ejecuta: CC (Claude) · Tarea **F7-05** de Fase VII (propuesta de GPT-5.6 Sol)
**Reanálisis puro:** no se corrió Phantom, no se generó ningún grafo, no se creó ninguna regla nueva.
Todo sale de CSV y de archivos que ya estaban en el disco.
**Este informe deja los números. No declara cierre ni veredicto.**

---

## 0. Qué se preguntó, en simple

O3-C dejó una deuda explícita: encontró que el 75% del efecto de la condición sobre la masa viajaba por la
pendiente, **pero declaró un confound que no pudo desenredar** — el grado medio era el correlato más fuerte
de la masa (ρ=−0.951) y estaba enredado con la pendiente (r=−0.755). El informe del equipo agregó otra
pieza: la densidad domina, y el residual que queda está ligado al *clustering* (ρ=0.77), no a la pendiente
(ρ=0.04).

La pregunta de F7-05 **no es "quién explica más R²"**. Con variables tan enredadas, esa pregunta la gana
siempre la más colineal y no significa nada. La pregunta es: **cuando a cada camino se le descuenta todo lo
que los demás pueden explicar, ¿cuál sigue en pie?**

*Analogía:* cuatro personas empujan el mismo carro, apretadas hombro con hombro. Preguntar "quién empuja
más" es inútil. Lo que se hace acá es sacar a tres del carro por vez y ver si el que queda todavía lo mueve.

---

## 1. El dataset unificado — qué entró, qué se descartó y por qué

**Archivo:** `cs090_fase7_f705_dataset_unificado.csv` — **254 filas × 54 columnas**, una fila por corrida
de Phantom. Script: `cs090_fase7_f705_unificar.py`. Registro completo: `cs090_fase7_f705_unificar.log`.

| Experimento de origen | filas | diseño | qué varía |
|---|---:|---|---|
| `F5B_40pares` (Fase V-B, 40 pares) | 76 | pareado | densidad libre |
| `O3A_N4000` (resolución) | 26 | pareado | mismas reglas a N=4000 |
| `O3B_rewiring` (gemelos de grados idénticos) | 24 | pareado | **densidad fijada por construcción** |
| `O3C_factorial` (mecanismo, 4 condiciones) | 47 | pareado | rigidez × criterio |
| `O3D_kcap` (barrido de kcap + controles Erdős-Rényi) | 38 | no pareado | kcap libre, 6 controles ER |
| `O3D_control_hist` | 2 | no pareado | control histórico |
| `O3E_memoria` (memoria vs sin memoria) | 30 | pareado | poda con/sin historia |
| `OUT_pendNEG` (outliers de pendiente negativa) | 11 | no pareado | reglas extremas |

**Cuentas de la unificación:**
- **258** filas leídas en bruto de 8 fuentes.
- **4 descartadas** por ser exactamente la misma corrida contada dos veces: en
  `cs090_fase5b_TOTAL_40pares.csv` cuatro reglas (`r6`, `r9`, `r19`, `r39`) participan en dos parejas
  distintas, con la misma `frac_masa` idéntica al quinto decimal. Quedan auditables en
  `cs090_fase7_f705_filas_descartadas.csv`.
- **254 filas finales**, todas con desenlace (`frac_masa`) — cero filas sin masa.

**Verificación de la clave `(rule_id, seed)`** — la advertencia de `FASE6_O3B_control_rewiring_CS.md`:
- Dentro del propio dataset de Phantom, ningún `rule_id` tiene más de un `seed` (0 colisiones internas).
- **Contra los lotes históricos de grafos sí colisiona:** cruzando con
  `cs090_fase5_mapa_transicion_grid.csv`, **12 filas coinciden por `rule_id` solo y 0 por `(rule_id, seed)`**.
  Unir por `rule_id` habría pegado 12 filas de grafos **equivocados**. El bug es real y está reproducido.

**Cobertura por variable** (de 254): masa 254 · nº aristas 254 · grado medio 254 · **geometría inicial 254**
· pendiente corregida 217 · diámetro 222 · holonomía 135 · **clustering 24**.

> **Limitación dura, declarada:** el **clustering existe medido en 24 corridas de las 254** (sólo O3-B). Los
> grafos no se guardan en disco (sólo las condiciones iniciales), y recalcular el clustering exigiría
> **regenerar los grafos**, que está fuera del alcance de esta tarea. Todo lo que este informe dice sobre
> clustering descansa sobre n=24 (12 pares).

### 1.bis. Un dato que estaba en el disco y nadie había pesado

Cada carpeta de Phantom guarda su `cosmogenesis_ic.txt`: **la nube de gas tal como nació, antes de integrar
un solo paso**. Estaba ahí desde siempre; sólo se había medido en 26 casos (O3-A) y 20 (O4-A).

Se midieron **las 360 condiciones iniciales del disco** con la misma vara de O4-A/O3-A (`fof_masa` importada
sin tocar), más dos descriptores locales que no dependen de umbral. Resultado: **la geometría inicial pasa a
estar disponible en las 254 filas** (antes: 62).

*Analogía: es pesar el bollo crudo. El bollo crudo estaba guardado en la heladera de cada experimento.*

Script: `cs090_fase7_f705_geometria_ic_todas.py` → `cs090_fase7_f705_geometria_ic_todas.csv` (360 filas).
**Verificación cruzada:** el nº de aristas anotado en la cabecera de cada IC coincide **254/254** con el del
CSV del experimento correspondiente. La unión IC↔corrida se hizo por `(carpeta, npart)`, no por nombre.

Las variables de geometría inicial:

| nombre | qué mide |
|---|---|
| `geoIC_fof_b0.20/0.30/0.50` | fracción de masa que ya nace en grumos FoF (escala **global**) |
| `geoIC_knn8_cv` | cuán desparejo está repartido el gas (CV de la densidad a 8 vecinos) |
| `geoIC_knn8_p90_med` | **cuán alto llega el pico local** respecto de lo típico (p90/mediana) |

---

## 2. El hallazgo que reordena todo lo demás: densidad y geometría FoF son **una sola variable**

```
densidad (grado medio 2E/N)  vs  geometría inicial FoF b=0.30      r = −0.9945   (n=254)
                                                                  ρ = −0.985
```

No son dos cosas. Es **la misma cosa medida dos veces**. Un PCA de las dos deja **99.73% de la varianza en un
solo eje**; el segundo eje (la discrepancia entre ellas) se queda con **0.27%**.

*Analogía: querer decidir si el peso de una persona lo explica mejor su altura en centímetros o en pulgadas.
El programa te da un número. El número es basura.*

**Consecuencia directa y declarada:** la cadena `densidad → geometría → masa` **no se puede estimar como
mediación**, porque el eslabón `a` (densidad→geometría) es prácticamente una identidad (a = −0.995). Cuando
se la fuerza igual, el programa devuelve:

| bloque | a | b | c | c′ | indirecto a·b | IC95% | prop. mediada |
|---|---:|---:|---:|---:|---:|---|---:|
| global | −0.995 | −1.307 | −0.829 | −2.129 | +1.300 | [+0.755, +1.851] | **−156.8%** |
| N=2000 | −0.995 | −2.178 | −0.938 | −3.105 | +2.167 | [+1.791, +2.555] | **−230.9%** |

Una "proporción mediada" de −157% o −231% **no es un resultado, es el síntoma** de que el modelo está
dividiendo por una diferencia que no existe. Se reportan porque están calculados y para que nadie los
recalcule creyendo que dicen algo. **No dicen nada.**

---

## 3. Regresión múltiple: qué coeficientes quedaron ininterpretables por colinealidad

Regla de lectura declarada **antes** de mirar los números (como hizo O3-D): VIF<5 se lee · 5-10 con reserva ·
**>10 ININTERPRETABLE**.

| modelo | n | R² | coeficientes y VIF |
|---|---:|---:|---|
| **M1** masa ~ dens + exp | 254 | 0.850 | dens β=−0.892 (p=3e-79) **VIF 1.6 — legible** |
| **M2** + geo | 254 | 0.895 | dens β=−3.04 **VIF 106.5 ✗** · geo β=−2.17 **VIF 106.5 ✗** |
| **M3** + pend | 217 | 0.949 | dens **VIF 132.0 ✗** · geo **VIF 117.6 ✗** · pend β=+0.261 **VIF 11.4 ✗** |
| **M4** + geo_cv | 217 | 0.949 | geo_cv β=−0.019 (p=0.67) VIF 8.2 (reserva) · los otros 3 ✗ |
| **M5** + kcap | 217 | 0.949 | kcap β=−0.010 (p=0.81) VIF 7.0 (reserva) · los otros 3 ✗ |
| **M7** O3B: masa ~ dens+geo+clus | 24 | 0.978 | dens ✗ (61.0) · geo ✗ (59.8) · **clus β=+0.182 (p=7e-5) VIF 1.2 — legible** |
| **M8** O3B: + pend | 24 | 0.981 | **clus β=+0.150 (p=0.001) VIF 1.5 — legible** · pend VIF 6.1 (reserva) · dens/geo ✗ |

**Lista explícita de coeficientes ININTERPRETABLES por colinealidad** (no se usan para concluir nada):
`dens` y `geo` en M2-M5 (VIF 96-137), `pend` en M3-M5 (VIF 11.4-11.5), la dummy `exp=OUT_pendNEG` en M3-M5
(VIF 12.4), y `dens`/`geo` en M7-M8 (VIF 60-65).

**Lo único legible de esta tabla:** con densidad sola y ajuste por experimento, R²=0.850 y VIF=1.6; y en
O3-B, el **clustering entra con VIF ≈ 1.2 y coeficiente significativo** aun conviviendo con dos variables
podridas de colinealidad — precisamente porque el clustering *no* está enredado con la densidad (r=−0.36).

**Reemplazo limpio de M2** (ejes ortogonales, VIF = 1.0 por construcción):

```
masa ~ eje_común(99.7% var) + eje_"geometría-que-no-es-densidad"(0.3% var)     R² = 0.706
   eje_común              β = −0.821 ± 0.034   p = 8e-67
   eje_geo_no_dens        β = −0.180 ± 0.034   p = 3e-07
   + dummies de experimento:  R² = 0.895;  eje_geo_no_dens β = −0.273 ± 0.022 (p=6e-27)
```
El segundo eje —el 0.27% de varianza donde densidad y geometría *no* coinciden— **no es ruido**: aporta
señal significativa. Pero su signo (β negativo) es **opuesto** al de la correlación cruda de la geometría con
la masa (ρ=+0.916). Ese vuelco de signo es la firma clásica de la supresión en régimen casi-singular:
**se informa, no se interpreta como mecanismo.**

---

## 4. ¿Artefacto de Simpson por mezclar diseños? — se probó, y disparó una vez

|  | ρ global | por experimento | efectos fijos (dentro de experimento) |
|---|---:|---|---:|
| densidad | −0.928 | −0.927, −0.930, −0.965, −0.956, −0.989, −0.971, −0.938 (**todos igual signo**) | r = −0.875 |
| geometría FoF | +0.916 | +0.845 … +0.975 (**todos igual signo**) | r = +0.836 |
| pendiente | +0.693 | +0.586, +0.922, +0.800, +0.972, +0.950, **−0.642 (OUT_pendNEG)** | r = +0.811 |

Para densidad y geometría **no hay Simpson**: la relación es la misma dentro de cada diseño y en el conjunto.
La única inversión es `OUT_pendNEG`, que **es la definición de ese subconjunto** (reglas seleccionadas por
pendiente negativa), no una sorpresa.

**Pero el Simpson sí apareció donde no se lo esperaba — en la resolución.** Ver §5.

---

## 5. Lo que sobrevive al condicionar (y lo que se cae)

Método: a cada variable se le descuenta la densidad (residualización), y se mide la correlación con la masa
igualmente residualizada — **primero mezclando todo, después sin mezclar resoluciones, después dentro de cada
diseño por separado**. Tabla completa en `cs090_fase7_f705_robustez.csv`.

| camino | mezclando todo (n=254) | sólo N=2000 (n=228) | dentro de cada experimento |
|---|---:|---:|---|
| **pico local del gas inicial** (p90/mediana) | +0.303 (p=9e-7) | **+0.713 (p=1e-36)** | **+0.64 · +0.66 · +0.79 · +0.84 · +0.87 · +0.90 — los 6, p<0.001** |
| **pendiente corregida** | −0.068 (p=0.32) | −0.068 (p=0.32) | **+0.47 · +0.50 · +0.59 · +0.59 · +0.74 — los 5 significativos** |
| geometría FoF (global) | −0.244 | −0.632 | −0.08 · −0.33 · −0.50 · −0.58 · −0.62 · −0.72 (**signo invertido vs. el crudo**) |
| CV de densidad local | +0.637 (p=2e-30) | **+0.071 (p=0.28)** | −0.40 · −0.21 · −0.17 · −0.12 · +0.21 · +0.25 — **ninguno significativo** |

### 5.1 Un Simpson real, cazado: el "CV local" era mezclar resoluciones

El CV de la densidad local parecía el ganador absoluto (r parcial +0.637, p=2×10⁻³⁰, y ρ medio +0.85 dentro
de bandas de densidad igualada). **Se cae entero** al no mezclar N=2000 con N=4000: r = +0.071, p=0.28. Y
dentro de cada experimento por separado, ninguno de los seis alcanza significancia y **cuatro cambian de
signo**. Era la paradoja de Simpson operando entre dos resoluciones. Queda anotado como falso positivo
encontrado y descartado, no como hallazgo.

*(El mismo control aplicado a las "bandas de densidad igualada" muestra por qué: el `geo` FoF dentro de banda
daba ρ medio +0.66, pero al descontar además la densidad **residual** de cada banda queda r parcial ≈ 0:
z=+0.10 (p=0.92), z=−0.07 (p=0.95), z=−0.82 (p=0.41) con 4/6/8 bandas. Era densidad disfrazada.)*

### 5.2 La pendiente: no muere, pero sólo se ve dentro de su propio diseño

Agrupando todo, la pendiente condicionada a la densidad da r = −0.068 (p=0.32): nada. **Dentro de cada
experimento da +0.47 a +0.74, los cinco significativos.** Con dummies de experimento en el modelo, la parcial
global vuelve a aparecer: r = +0.325 (p=1×10⁻⁶), ρ parcial = +0.402 (p=8×10⁻¹⁰).

Traducción: la pendiente **sí** aporta algo que la densidad no explica, pero sólo comparando corridas del
mismo diseño. Comparando entre diseños, el desnivel entre experimentos se la come. Esto **matiza** el ρ=0.04
del informe de equipo (que era el residual dentro de O3-B, n=12) sin contradecirlo: en O3-B la parcial de la
pendiente contra la densidad es +0.468 (p=0.021), pero en el análisis **pareado** de O3-B la Δpendiente **no**
ordena quién gana (ρ = +0.063, p=0.85). Son dos preguntas distintas y dan respuestas distintas.

---

## 6. El único lugar donde la densidad está fijada de verdad: O3-B

El rewiring de O3-B conserva los grados nodo por nodo. Verificado acá: **máxima diferencia de grado medio
entre gemelos = 0.000e+00, y 12/12 pares con el mismo número exacto de aristas.** Es el único punto del
corpus donde "a igual densidad" no es una corrección estadística sino un hecho de construcción.

**Diferencias original − gemelo rebarajado (12 pares):**

| Δ | mediana | gana el original | p signos | p Wilcoxon |
|---|---:|---:|---:|---:|
| **fracción de masa** | **+0.00500** | 9/12 | 0.146 | **0.0093** |
| clustering local | +0.00955 | **11/12** | **0.0063** | **0.0010** |
| transitividad | +0.00608 | **11/12** | **0.0063** | **0.0010** |
| nº de triángulos | +22 | **11/12** | **0.0063** | **0.0010** |
| **pico local del gas inicial** | +1.209 | 10/12 | **0.0386** | **0.0425** |
| pendiente corregida | +0.0877 | 10/12 | **0.0386** | **0.0269** |
| geometría FoF b=0.30 | −0.00475 | 3/12 | 0.146 | 0.424 |
| CV de densidad local | −0.0123 | 6/12 | 1.000 | 0.519 |

**Qué diferencia predice quién gana masa** (ρ de Spearman entre las Δ, n=12):

```
Δ nº de triángulos      ρ = +0.746   (p = 0.0054)
Δ clustering            ρ = +0.746   (p = 0.0053)
Δ pico local del gas    ρ = +0.725   (p = 0.0076)
Δ transitividad         ρ = +0.701   (p = 0.0112)
Δ geometría FoF         ρ = +0.151   (p = 0.64)      <- no ordena nada
Δ pendiente             ρ = +0.063   (p = 0.85)      <- no ordena nada
Δ CV local              ρ = −0.291   (p = 0.36)
```

El residual del +5% se reproduce exactamente: mediana Δ=+0.0050 sobre una masa media de ~0.108, es decir
**+4.6% relativo**, con el original ganando en 9/12 pares.

### 6.1 La mediación pareada, con la densidad ya fijada

Con dos candidatos a eslabón del medio:

| cadena | a (X→M) | b (M→Y\|X) | c total | c′ directo | indirecto a·b | IC95% bootstrap |
|---|---:|---:|---:|---:|---:|---|
| Δclus → Δgeo **FoF global** → Δmasa | −0.025 | +0.127 | +0.712 | +0.715 | −0.0031 | [−0.101, +0.141] **incluye 0** |
| Δclus → Δ**pico local** → Δmasa | +0.418 | +0.491 | +0.712 | +0.507 | **+0.205** | [−0.032, +0.476] **incluye 0** |

La versión con **geometría local** es la única que tiene forma de cadena (a=+0.42 con ρ=+0.559, p=0.059;
proporción mediada 28.8%), pero **con n=12 el intervalo bootstrap incluye el cero por poco**. No alcanza para
afirmar la mediación; alcanza para decir que **si hay una cadena, es por la geometría LOCAL y no por la
global**, y para señalar exactamente qué experimento la resolvería.

---

## 7. La hipótesis de F7-05, punto por punto

> **Hipótesis propuesta:** que el fenómeno se descomponga en `densidad → geometría → masa` (efecto grande) y
> `clustering → geometría local → +5%` (residual).

**Primera mitad: NO se sostiene como cadena — no porque falle, sino porque no es separable.**
Densidad y geometría inicial (FoF) correlacionan a **r=−0.9945**; son la misma variable medida en dos
unidades. La flecha `densidad → geometría` no se puede distinguir de una identidad, y cualquier mediación
estimada sobre ella devuelve proporciones sin sentido (−157%, −231%). Lo que **sí** se sostiene es el bloque
conjunto: `[densidad ≡ geometría global] → masa`, con **R²=0.687** usando densidad sola (0.850 con ajuste por
experimento), β=−0.829 (p=2×10⁻⁶⁵), y **el mismo signo en los 7 subconjuntos**. Es el efecto grande, sí; pero
como **un solo eslabón**, no como cadena de dos.

**Segunda mitad: se sostiene en la forma, se corrige en el detalle, y le falta potencia.**
- El residual del **+4.6%** a densidad exactamente idéntica existe y se reproduce (9/12, Wilcoxon p=0.0093).
- El **clustering es lo que mejor ordena quién gana** (ρ=+0.746, p=0.005), junto con el nº de triángulos
  (idéntico ρ) y la transitividad — las tres son la misma familia de medida.
- **Pero el eslabón "geometría local" hay que reescribirlo.** La geometría que media *no* es la global (la
  masa en grumos FoF: ρ=+0.151, p=0.64, indirecto −0.003), sino **la altura del pico local de densidad del
  gas al nacer** (ρ=+0.725, p=0.008; indirecto +0.205, prop. mediada 28.8%). Con n=12 el IC95% roza el cero
  ([−0.032, +0.476]): **la forma de la cadena aparece, la significancia no llega.**

**Lo que emerge, que no estaba en la hipótesis:** hay **dos ejes de geometría, no uno**.

```
   EJE 1 — geometría GLOBAL  ≡  densidad          r = −0.995 con el grado medio
           (cuánta masa nace en grumos grandes)   efecto grande, un solo eslabón, R² = 0.69
                                                  NO aporta nada al condicionar (r parcial ≈ 0 dentro de banda)

   EJE 2 — geometría LOCAL   (altura del pico)    r parcial con la masa, descontada la densidad:
           independiente de la densidad             +0.64 a +0.90 DENTRO DE CADA UNO de los 6 experimentos
                                                    +0.713 con sólo N=2000
                                                  y es el eje que el CLUSTERING mueve (Δ: ρ=+0.559)
```

**El eje 2 es el único camino que sobrevive en todos los subconjuntos, en todos los diseños y en las dos
resoluciones.** Y no lo captura ninguna de las variables con las que se venía trabajando: ni la pendiente
(que sólo sobrevive dentro del propio diseño), ni el FoF global (que es densidad disfrazada), ni el CV local
(que era un Simpson entre resoluciones).

---

## 8. Advertencias que hay que leer junto con estos números

1. **Todo esto es correlacional.** Ningún camino de acá prueba causalidad. La única intervención genuina del
   corpus es el rewiring de O3-B (n=12 pares), y ahí la potencia es baja. Los experimentos que **sí**
   intervienen (F7-02, F7-04) corren en paralelo.
2. **Clustering: n=24 de 254.** Todo lo que este informe dice sobre clustering se apoya en 12 pares. No se
   puede extender al resto sin regenerar grafos.
3. **El "pico local" no es una intervención.** Es una medida hecha sobre las IC. Que sobreviva a condicionar
   la densidad en 6/6 experimentos es fuerte, pero sigue siendo una asociación observada.
4. **Diseños heterogéneos.** Se mezclaron 8 experimentos con diseños distintos (pareados y no, densidad libre
   y grados fijos, dos resoluciones). El control de Simpson se hizo y **disparó una vez** (§5.1). Cualquier
   número global de este informe debe leerse contra su versión por-experimento.
5. **La geometría inicial se pegó por `(carpeta, npart)`, no por nombre de regla**, y se verificó contra el
   nº de aristas de la cabecera del IC (254/254). La geometría de O4-A que sólo tenía `rule_id` sin `seed`
   quedó marcada como `geo_o4a_sin_verificar_seed` y **no se usó** en ningún análisis.
6. **La columna `giant` no se homogeneizó a la fuerza:** en `remedicion_430` es un conteo de nodos, en otros
   CSV una fracción, y en `o3b_estructura` un booleano. Se dejó vacía donde las unidades no coincidían, en
   vez de mezclar tres cosas distintas en una columna.

---

## 9. Lo que estos números señalan como próximo paso (no es un cierre)

- **El experimento que falta ahora es distinto del que se venía pidiendo.** No es sólo "barrer nº de aristas
  a kcap fijo" (que rompe `kcap`↔densidad): es **fijar la densidad y mover el clustering** — exactamente el
  control con grados **y triángulos** fijos que el informe de equipo ya había anotado como pendiente. F7-02
  (escalera de clustering) es ese experimento.
- **El observable a medir en él debería ser el pico local del gas inicial** (`p90/mediana` de la densidad a 8
  vecinos), no la masa en grumos FoF ni la pendiente. Es la única variable que sobrevivió a todos los
  controles de esta tanda, se mide gratis sobre las IC antes de correr Phantom, y es la que el clustering
  mueve.
- **Con n=12 la mediación local roza el cero.** Doce pares más de rewiring bastarían para saber si el
  intervalo se despega, sin necesidad de un diseño nuevo.

---

## 10. Archivos producidos (todos nuevos; no se modificó ni un CSV ni un script previo)

| archivo | qué es |
|---|---|
| `cs090_fase7_f705_unificar.py` | unificación de las 8 fuentes por `(rule_id, seed)` con verificaciones |
| `cs090_fase7_f705_dataset_unificado.csv` | **el dataset: 254 corridas × 54 columnas** |
| `cs090_fase7_f705_filas_descartadas.csv` | las 4 filas duplicadas, para auditoría |
| `cs090_fase7_f705_geometria_ic_todas.py` | mide la geometría inicial de las 360 IC del disco |
| `cs090_fase7_f705_geometria_ic_todas.csv` | esa geometría |
| `cs090_fase7_f705_mediacion.py` | correlaciones, parciales/semiparciales, VIF, mediación, Simpson |
| `cs090_fase7_f705_separar.py` | ejes ortogonales, estratos, gemelos O3-B, robustez, figura |
| `cs090_fase7_f705_{correlaciones,parciales,modelos,mediacion,simpson,homogeneos}.csv` | tablas del paso 3 |
| `cs090_fase7_f705_{ejes_ortogonales,estratos_densidad,o3b_pareado,robustez}.csv` | tablas del paso 4 |
| `cs090_fase7_f705_{mediacion,separar,unificar}.log` | registro completo, número por número |
| `cs090_fase7_f705_caminos.png` | los cuatro paneles: efecto, colinealidad, gemelos, camino superviviente |

*Reproducción:* `./venv/bin/python cs090_fase7_f705_unificar.py` → `..._geometria_ic_todas.py` →
`..._mediacion.py` → `..._separar.py`.
