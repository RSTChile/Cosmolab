# FASE VI — O3-E: A2-B0-C2 **con memoria** vs **sin memoria** (Phantom, N=2000, masa fija)

*11-ago-2026 · tarea **O3-E** de `FASE6_PLAN_EJECUCION_COMPLETA_CS.md` (origen: 2do analista, §4.2).
15 pares emparejados, 30 corridas de Phantom, N=2000, masa fija, protocolo estándar de la línea.
No declara cierre ni veredicto: la interpretación es de Alexis.*

> **En una línea:** la memoria del costo, en A2-B0-C2, resulta ser *cortar ~80 aristas de más* (el grafo
> con memoria es subconjunto estricto del sin memoria en los 15 pares); el brazo con memoria acreta más
> masa en 10 de 15 pares (Wilcoxon p=0.027) pero **menos de lo que esa poda extra ya predice por sí sola**
> (residuo negativo en 11 de 15, Wilcoxon p=0.048), y **no** cambia la geometría (pendiente corregida
> 8/15, p=1). El confound de densidad **está presente y es sistemático** — ver §5.3.

---

## 0. Qué se preguntó y qué NO se preguntó

La propuesta del 2do analista (§4.2) observa que A2-B0-C2 **tiene memoria por construcción**: parte de lo
que decide qué relaciones sobreviven no es el estado presente del sistema sino su historia. La pregunta
operativa es si esa historia **sirve para algo físico**: ¿acreta más masa en Phantom un universo cuya
poda recuerda, comparado con uno idéntico cuya poda sólo mira el presente?

Esto prueba **sólo el ANTECEDENTE de O-N7.7** — *"restricción histórica que reduce sin destruir capacidad
futura"* — tal como fijó la decisión formal ya adoptada (`FASE5_especificacion_universalidad_CS.md` §6 y
el informe del equipo del 11-ago). **No** prueba Libertad Funcional, que pertenece a otro plano teórico
(ANIMA / Célula Madre). Este informe no cruza ese límite.

---

## 1. Dónde está la memoria en A2-B0-C2 (leído del motor congelado)

En `cs090_fase5_motor.py`, `dinamica_B0` lleva un contador `flip_count`: en **cada** sweep se compara la
topología con la del sweep anterior y toda arista que apareció o desapareció suma +1. Ese contador se
**acumula** sobre los 14 sweeps y al final entra como uno de los dos componentes del costo con el que C1/C2
poda las aristas más caras (`_costo_y_podar`, percentil P70):

```
costo = 0.5 · z( inconsistencia HISTÓRICA )   +   0.5 · z( conflicto de holonomía INSTANTÁNEO )
        └── cuántas veces cambió esa arista ──┘       └── el estado final de los triángulos ──┘
            a lo largo de TODA la corrida                    que contienen esa arista
```

**Analogía.** Al terminar la partida el sistema decide qué relaciones cortar mirando dos cosas: cuánto
conflicto hay *ahora* (holonomía) y cuántas veces esa relación fue inestable *a lo largo de toda la
partida* (historia). La variante "sin memoria" le tapa el segundo ojo: sólo puede mirar el ahora.

---

## 2. Qué se quitó EXACTAMENTE para la variante sin memoria

Un único cambio, parametrizado por `ventana_memoria` en `cs090_fase6_o3e_memoria.py`:

| brazo | `ventana_memoria` | término histórico del costo |
|---|---|---|
| **`-mem`** (con memoria) | `None` | `flip_count` **acumulado** sobre los 14 sweeps — idéntico al motor congelado |
| **`-nomem`** (sin memoria) | `1` | flips ocurridos **sólo en el último sweep**, sin ninguna acumulación previa |

Todo lo demás es idéntico y así se verifica en el código:

- mismo `p` (mismos K, J, ruido, meandeg, kcap), misma semilla, mismo `np.random.default_rng(seed*5000+N)`;
- mismo `construir_A2`, mismo `_circ_mean_update`, mismo calendario de recableo (`step % 3 == 0`) y de
  límite de escala kcap (`step % 4 == 0`), mismos 14 sweeps;
- mismo `_costo_y_podar` (mismo percentil **P70**, mismos pesos 0.5/0.5, mismo z-score), mismo
  `_muestrear_triangulos`, mismo `medir()`;
- **mismo consumo del generador aleatorio**: contar flips no consume azar y el costo/poda tampoco. Por
  construcción los dos brazos recorren trayectorias **bit a bit idénticas** hasta el instante final de la
  poda; lo único que difiere es **cuáles** aristas condena el costo.

**Verificación de fidelidad (hecha, no asumida).** `verificar_identidad_con_motor_congelado()` corre la
cadena congelada (`cs090_fase5b_phantom_adaptador.reconstruir_regla_a2b0c2`) y este módulo con
`ventana_memoria=None`, y exige igualdad **nodo por nodo, vecino por vecino** de la adyacencia final, más
igualdad de `n_aristas`, `diam` y holonomía. Pasó.

**Un hecho medido que hay que decir de frente.** Con el calendario del motor congelado y `n_sweeps=14`, en
el último sweep (`step=13`) no hay ni recableo (13 % 3 = 1) ni kcap (13 % 4 = 1): la topología no cambia. O
sea que la ventana instantánea de un sweep **no contiene ningún cambio**, el término histórico queda
idénticamente 0, `z(0) = 0`, y el costo del brazo sin memoria pasa a ser **puramente el conflicto de
holonomía del presente**. El script lo verifica corrida por corrida (`hist_usado_suma = 0` en todos los
`-nomem`) en vez de darlo por supuesto. Es la lectura literal de "actualización instantánea, sin
acumulación de historia".

---

## 3. Pipeline (reusado, sin tocar nada)

```
regla A2-B0-C2 (seed)
   ├─ brazo -mem   ─┐
   └─ brazo -nomem ─┤→ grafo final N=2000  →  pendiente corregida (coarse-graining b=1,2,4,8,16 con
                     │                          cs090_diam_corregido.diam_gigante)
                     └→ layout de resortes + masa fija + turbulencia
                        (cs090_fase5b_phantom_adaptador, lado 2000^(1/3), masa total 18800,
                         seed_layout=12345, Mach=3 seed=42)
                        → Phantom (cs090_fase5b_correr.correr_una: icreate_sinks=1, rho_crit=1000,
                           r_crit=0.6, h_acc=0.3, tmax=0.500, dtmax=0.001)
                        → métricas (cs090_fase5b_analizar.analizar_carpeta)
```

**Selección de reglas base.** Determinista y estratificada: se ordenan las 430 reglas A2-B0-C2 ya
remedidas (`cs090_fase6_remedicion_430.csv`) por **pendiente corregida** y se toman posiciones igualmente
espaciadas. No se elige por resultado en Phantom ni al azar sin registro; se cubre todo el rango de
geometría, que es justamente la variable que después hay que descontar.

**Prefijo de lote nuevo, sin colisión** con ninguno previo (`r0-r19`, `r0-r39`, `batch3-*`, `batch4-*`,
`*v1fix`, `*v2fix`, `*pendNEG`): **`A2-B0-C2-o3e-s{seed}-mem` / `-nomem`**.

**Verificación cruzada contra `meta_regla.json`** (la lección del bug de colisión de nombres): tras
escribir cada meta se relee del disco y se exige que `seed`, `K`, `kcap`, `brazo` y `ventana_memoria`
coincidan con lo pedido y que el nombre de la carpeta coincida con el `rule_id` de adentro. Además, para
cada par se exige que los dos metas declaren **exactamente la misma regla** (seed, K, J, noise, meandeg,
kcap, seed_layout, N). Cualquier discrepancia aborta.

---

## 4. Costo medido antes de comprometer la batería

Piloto de 2 pares (4 corridas): grafo + geometría ≈ 5 s, layout + IC ≈ 275 s, Phantom ≈ 30 s por corrida;
8 min los 2 pares con 4 procesos en paralelo. Durante la batería completa la máquina quedó saturada por
otras tareas del equipo (load average > 500), lo que multiplicó por ~3 el tiempo del layout; se solapó la
etapa de Phantom con la de generación de condiciones iniciales para recuperar tiempo.

---

## 5. Resultados — 15 pares, 30 corridas de Phantom

### 5.1 Qué tan grande resultó ser la manipulación

Antes de mirar la masa conviene saber **cuánto** cambia el grafo al taparle un ojo al costo. Se
reconstruyeron los dos grafos de cada par y se comparó arista por arista:

| medida | valor (15 pares) |
|---|---|
| Jaccard entre los dos conjuntos de aristas | mediana **0.975** (rango 0.966–0.994) |
| aristas presentes **sólo** en el brazo con memoria | **0 en los 15 pares** |
| aristas presentes sólo en el brazo sin memoria | mediana 80 (rango 27–149) |
| fracción de aristas conservada por la poda P70 | mem 0.963 · nomem 0.987 |

**El resultado más limpio de toda la tarea está en esa segunda fila**: el grafo con memoria es un
**subconjunto estricto** del grafo sin memoria, en los 15 pares sin excepción. La memoria no reordena
nada, no crea ninguna relación: lo único que hace es **cortar ~80 aristas de más** (≈2.5%), y son
exactamente las que tienen historial de inestabilidad. Es, literalmente y no por metáfora, una
*restricción histórica que reduce*.

(Se registró también que el corte a percentil P70 no poda el 30% sino apenas el 1–7%: el costo toma pocos
valores distintos —mediana 51 en `mem`, 46 en `nomem`— y el umbral cae dentro de un empate enorme. Es una
propiedad del motor congelado, no de esta tarea, pero explica por qué la palanca es chica.)

### 5.2 Tests pareados (Δ = brazo con memoria − brazo sin memoria)

| observable | mem>nomem | mediana Δ | signos p | Wilcoxon p |
|---|---|---|---|---|
| **fracción de masa acretada (principal)** | **10 / 15** | **+0.0010** | 0.302 | **0.0267** |
| masa absoluta en sumideros | 10 / 15 | +18.8 | 0.302 | 0.0286 |
| κ_V agregado | 11 / 15 | +0.0118 | 0.119 | 0.188 |
| nº de sumideros | 0 / 15 (**15 empates**) | 0 | — | — |
| pendiente corregida (geometría) | 8 / 15 | +0.0005 | 1.000 | 0.525 |
| nº de aristas *(confound)* | **0 / 15** | **−80** | **6.1e-05** | **6.5e-04** |
| grado medio *(confound)* | 0 / 15 | −0.081 | 6.1e-05 | 6.5e-04 |

Escala de referencia: la fracción de masa acretada va de 0.060 a 0.158 entre las 30 corridas, y **las 30
corridas formaron exactamente 8 sumideros** (ni uno más ni uno menos). O sea que el Δ mediano de +0.0010
es ~1% del valor típico: un empujón, no un salto.

### 5.3 El confound de densidad **está presente y es sistemático**

Los dos brazos **no** quedan con la misma densidad: el brazo con memoria tiene menos aristas en **los 15
pares** (mediana **−2.48%**, rango −0.63% a −3.37%). Y en este pipeline la densidad no es un detalle:
sobre las 30 corridas, la fracción de masa acretada y el número de aristas están casi perfectamente
anti-correlacionados,

> Spearman ρ = **−0.971** (p = 6e-19) · ajuste lineal `fracción_masa = −3.31e-05 · n_aristas + 0.217`, R² = 0.915

**Analogía:** menos hilos que sostengan la red ⇒ el material cae más fácil al centro. Es un efecto casi
mecánico, y es mucho más grande que cualquier cosa que estemos midiendo acá.

Descontándolo dentro de cada par:

| | media | mediana |
|---|---|---|
| Δmasa **observado** | +0.00157 | +0.00100 |
| Δmasa **predicho por la sola diferencia de aristas** | +0.00268 | +0.00265 |
| **residuo** (observado − predicho) | **−0.00111** | −0.00139 |

El residuo es **negativo en 11 de 15 pares** (signos p = 0.119; Wilcoxon p = 0.048). La regresión
independiente `Δmasa ~ 1 + Δpendiente + Δaristas` da lo mismo por otro camino: β_aristas = −3.47e-05
(prácticamente idéntico al −3.31e-05 del ajuste global, lo que da confianza en el descuento) e intercepto
**−0.00138** — el Δmasa que quedaría si los dos brazos tuvieran igual densidad e igual geometría.

Es decir: **el brazo con memoria gana masa, pero gana MENOS de lo que su propia poda extra ya explicaba.**

### 5.4 ¿Pasa por la geometría?

No. La pendiente corregida no se mueve sistemáticamente entre brazos (8/15, mediana +0.0005, signos p = 1,
Wilcoxon p = 0.52) y Δmasa vs Δpendiente da ρ = +0.29 (p = 0.29), mientras Δmasa vs Δaristas da ρ = −0.51
(p = 0.053). Ojo con una trampa de lectura: sobre las 30 corridas la pendiente corregida y la fracción de
masa correlacionan fortísimo (ρ = +0.950), pero eso es porque **ambas** son función del número de aristas
— *entre brazos del mismo par*, que es donde se controla la regla, la geometría no se mueve.

Traducido: la memoria **no** produce grafos más extendidos. Lo único que produce es un grafo un poco más
ralo.

### 5.5 Lo que queda abierto (y el control que falta)

El contraste, tal como quedó operacionalizado, **no es limpio en densidad** — hay que decirlo antes que
cualquier otra cosa. El control decisivo que no se corrió (costo: ~15 condiciones iniciales más, y el
layout de resortes es lo caro) sería: tomar el grafo `nomem` y quitarle **las mismas ~80 aristas pero
elegidas al azar** en vez de por historial de inestabilidad, y pasar eso por Phantom. Eso separaría
"podar 80 aristas" de "podar LAS 80 aristas que la historia señala". Con los números de arriba, la
predicción del ajuste global es que el azar daría **+0.0027** de masa, y el brazo con memoria dio
**+0.0016** — o sea que el azar ganaría; pero eso es una extrapolación de una relación entre reglas, no
una medición dentro del par, y hasta que no se corra sigue siendo una conjetura.

Otros dos límites del diseño, declarados: (a) con `n_sweeps=14` la ventana instantánea de un sweep resulta
vacía, así que "sin memoria" es aquí "el costo pierde su término histórico" en su forma más fuerte —
convendría repetirlo con un calendario de recableo donde la ventana de un sweep sí contenga cambios;
(b) n = 15 pares da poca potencia para efectos de este tamaño (el test de signos no llega, sólo Wilcoxon,
y sólo apenas).

### 5.6 Los 15 pares, uno por uno

| seed | clase histórica | Δ masa | Δ % aristas | Δ pendiente | Δ κ_V |
|---|---|---|---|---|---|
| 273381 | III | −0.00150 | −2.66% | −0.0693 | +0.016 |
| 375224 | I | +0.00200 | −3.10% | +0.0677 | +0.024 |
| 475321 | I | +0.00650 | −3.37% | −0.0021 | +0.019 |
| 477455 | III | +0.00400 | −2.48% | −0.0286 | +0.163 |
| 478716 | III | +0.00350 | −2.71% | +0.0275 | +0.033 |
| 484633 | III | −0.00100 | −1.56% | +0.0375 | −0.023 |
| 485506 | I | +0.00350 | −3.01% | +0.0301 | +0.076 |
| 573963 | I | +0.00400 | −2.38% | +0.0758 | −0.006 |
| 575709 | I | −0.00100 | −2.66% | −0.0048 | −0.081 |
| 576679 | I | +0.00100 | −1.78% | −0.0313 | +0.007 |
| 577261 | III | +0.00100 | −1.43% | +0.0050 | +0.007 |
| 582111 | I | −0.00050 | −0.63% | +0.0005 | +0.061 |
| 584342 | I | +0.00150 | −2.51% | −0.0021 | +0.012 |
| 586573 | III | +0.00100 | −1.87% | +0.1501 | +0.008 |
| 588998 | I | −0.00050 | −1.07% | −0.0349 | −0.097 |

---

## 6. Archivos

| archivo | qué es |
|---|---|
| `cs090_fase6_o3e_memoria.py` | variante del motor con `ventana_memoria` + verificación de fidelidad + pendiente corregida |
| `cs090_fase6_o3e_correr.py` | batería emparejada: selección de reglas, IC, Phantom, métricas |
| `cs090_fase6_o3e_analizar.py` | estadística pareada (signos + Wilcoxon), confound de densidad, ¿pasa por la geometría? |
| `cs090_fase6_o3e_memoria_crudo.csv` | CSV crudo, una fila por corrida |
| `cs090_fase6_o3e_diferencias_por_par.csv` | diferencias mem − nomem por par |
| `/Users/alexis/phantom_cs073/bateria_fase6_o3e_memoria/` | 30 corridas de Phantom (IC, dumps, `.sink`, `meta_regla.json`) |
| `cs090_fase6_o3e_correr.log` · `o3e_solapamiento.log` | log de la batería y del cálculo de solapamiento de grafos |

Ningún script congelado fue modificado; ningún commit fue hecho.
