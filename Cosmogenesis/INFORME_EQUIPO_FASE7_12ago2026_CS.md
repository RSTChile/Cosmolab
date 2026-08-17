# Informe para el equipo — Fase VII completa (12 de agosto de 2026)

**Para:** Equipo Transinteligente · **Preparado por:** CC (Claude Code) · **Dirige:** Alexis López Tapia ·
**Continúa de:** `INFORME_EQUIPO_FASE6_11ago2026_CS.md` y `CARTA_EQUIPO_FASE6_QUE_CAMBIO_CS.md`

**Qué se hizo:** sobre la propuesta de Fase VII de GPT-5.6 Sol, reordenada por información-por-corrida, se
ejecutaron **6 experimentos + 2 tareas de infraestructura**. Todos los números verificados contra archivos
en disco. **Valores estadísticos canónicos:** `VALORES_CANONICOS_FASE5B_ajustados_CS.md` (un valor por test
y por conjunto, corrección que pidió GPT-5.6 Sol y que estaba bien pedida).

---

## TL;DR — la respuesta cambió tres veces, y ésta es la que sostienen los datos

```
Fase V-B decía:   "Clase III acreta más masa"
Fase VI corrigió: "era casi toda DENSIDAD; queda un residual ~5% ligado al CLUSTERING"
Fase VII corrige: "el clustering NO es el mediador — falla en el signo cuando se fija el nº de
                   triángulos. Lo que importa es CÓMO SE APIÑAN esos triángulos."
```

**La cadena final:**

```
regla relacional ──► cuántas aristas sostiene ──► DENSIDAD ──► efecto GRANDE
                                                     │
                     cómo se APIÑAN los triángulos ──► +13.8%  ← la variable real
                     (triángulos/arista, solapamiento, concentración)
                     NO el coeficiente de clustering, que se invierte
```

| Pregunta | Respuesta | Evidencia |
|---|---|---|
| ¿`kcap` tiene vía propia además de densidad? | **NO** | R² de M a kcap fijo = 0.836 vs R² de kcap a M fijo = 0.035 (F7-01) |
| ¿Importa **cuáles** aristas se cortan? | **NO** | Friedman p=0.33 con M idéntico (F7-04) |
| ¿El clustering mueve la masa causalmente? | **SÍ, pero…** | 12/12, ρ=+0.965 con grados fijos (F7-02) — **pero no es el mediador** (F7-03) |
| ¿Importa el nº de triángulos o su organización? | **La organización** | +13.8%, 12/12, con nº de triángulos FIJO (F7-03) |
| ¿Importa a qué nodo le toca cada cupo? | Sólo vía densidad | El residuo cambia de signo al descontarla (F7-06) |
| ¿Densidad y geometría son eslabones distintos? | **NO, son la misma variable** | r=−0.9945, PCA 99.73% en un eje (F7-05) |

---

## PARTE 1 — Los dos cierres

### F7-01: `kcap` opera únicamente a través de la densidad ✅ (cierre limpio)

Se rompió la colinealidad de verdad: **r(kcap, M) de +0.984 a +0.008; VIF de 47.8 a 1.00**.

- **M igualado 100% por SELECCIÓN, 0 celdas por poda** — importante, porque podar al azar habría mezclado el
  control con un brazo de F7-04.
- Mapear primero el terreno (1200 reglas) reveló que **los rangos de M de cada `kcap` son casi disjuntos**;
  la única intersección real es kcap 6×7. (También: `kcap=8` no existe, `RANGO_KCAP=(4,7)`.)
- **R² de M a `kcap` fijo = 0.836 · R² de `kcap` a M fijo = 0.035**, p≥0.32 en los tres niveles. El extremo
  negativo del IC del coeficiente de kcap es el **3.5%** del coeficiente ingenuo previo.

**`kcap` es una manija que mueve la densidad, nada más.**

### F7-04: qué aristas se cortan, da igual ⊘ (nulo informativo)

Cinco brazos (C2 / azar / soporte / anti-costo / anti-soporte) desde el mismo grafo, con **M verificado
idéntico** y el brazo C2 reproduciendo el motor congelado arista por arista.

**Friedman p=0.33.** El orden esperado C2>azar>anti se cumple en 2/12 — exactamente lo que da el azar.

**Hallazgo de método relevante para toda la línea:** la poda "P70" de C2 **corta 3%, no 30%** — el 92-97% de
las aristas están empatadas en un valor modal de costo. **Lo que ralea el grafo es `_enforce_kcap`, no
`_costo_y_podar`.** Eso encaja hacia atrás con toda la línea F5-C2-C: el criterio que importaba era siempre
el ranking por soporte local.

---

## PARTE 2 — El hallazgo central: no es el clustering, es el apiñamiento

### F7-02: el clustering SÍ mueve la masa (primera intervención causal del proyecto) ✅

Escalera deliberada de clustering (**C=0.0000 con cero triángulos exactos → 0.235-0.605**, hasta 480× el
natural) con la **secuencia de grados verificada idéntica** (`np.array_equal` sobre 2000 grados en los 72
grafos: 0 diferencias). 72 corridas.

- **ρ intra-grafo > 0 en 12/12** (media +0.965); monótona estricta en 10/12
- Spearman pareado **+0.960 (p=7.7e-34)**; **Page L p=1.3e-11**; e4 vs e0 gana 12/12
- Respuesta **convexa**: +37.2% en todo el rango; **+2%** en el tramo que roza lo natural
- **No va por la pendiente** (monótona en 1/12 vs 10/12 de la masa). Las parciales contra pendiente,
  asortatividad y componente gigante dejan la relación intacta (0.943/0.911/0.934)
- Mecanismo: nº de sumideros casi constante (8.0→8.6), primer sumidero **antes** en 12/12 →
  **cada grumo come más, no hay más grumos**

**Y dejó un hilo suelto:** el grafo nativo le gana también a un escalón fabricado **con más clustering que
él** (9/12). *La perilla mueve la masa, pero no reproduce lo que el nativo tiene.*

### F7-03: ese hilo suelto tiene respuesta — y obliga a jubilar el clustering ✅✅

**El experimento decisivo.** Grados idénticos nodo por nodo **Y número de triángulos idéntico** (diferencia
máxima real: **1 triángulo**), variando sólo **cómo están organizados**.

**La masa cambia igual: `solap` − `disj` = +0.01433 (+13.8%), 12/12 grafos, Wilcoxon p=4.9e-04; Friedman
p=2.4e-06.**

- **28.7 partículas contra un grano de 1 — lo supera 29×.** No es marginal.
- Es el **36% de toda la escalera de F7-02, sin agregar un solo triángulo.**

**El punto que reordena la teoría:** el **coeficiente de clustering correlaciona AL REVÉS aquí (ρ=−0.770)**,
y su predicción cuantitativa **falla en el signo** (−0.0062 predicho vs **+0.0143 observado**).

> **El clustering no es el mediador. Funcionaba mientras el número de triángulos se movía con él; fijado el
> número, deja de funcionar y se invierte.**

**Lo que sí correlaciona — el apiñamiento del soporte:** triángulos por arista **+0.776**, Gini de
concentración **+0.736**, solapamiento **+0.696**, aristas que sostienen triángulos **−0.781**. Las medidas
de comunidad (modularidad −0.576) siguen peor. Las parciales sobreviven descontando gigante, componentes,
asortatividad y pendiente.

**Y explica el hilo suelto de F7-02:** lo que el grafo nativo tiene no es más triángulos ni más clustering —
es **triángulos más apretados y solapados entre sí**.

---

## PARTE 3 — Dos correcciones al modelo mental

### F7-05: densidad y geometría inicial son LA MISMA variable

254 corridas unificadas de 8 fuentes. **r = −0.9945; un PCA deja el 99.73% en un solo eje.**

> `densidad → geometría → masa` **no es una cadena de dos pasos: es un solo eslabón.** Forzar la mediación
> da proporciones de −157% y −231% — síntoma de colinealidad, no resultado.

- **Sobrevive al condicionar por densidad:** el **pico local de densidad del gas inicial** (r parcial +0.64 a
  +0.90, p<0.001 en los 6 experimentos). **Variable candidata nueva que nadie había propuesto.**
- **La pendiente** sobrevive dentro de cada diseño pero **desaparece al agrupar** (r=−0.068, p=0.32).
- El CV de densidad local **parecía** el ganador (+0.637, p=2e-30) y **era artefacto de Simpson** por mezclar
  N=2000 con 4000 (cae a +0.071). El agente detectó su propio artefacto.
- **El bug de unión quedó reproducido:** 12 filas coinciden por `rule_id` solo, **0** por `(rule_id, seed)`.

### F7-06: la alineación del cupo actúa sólo por densidad

Multiconjunto de cupos verificado idéntico (12/12). Efecto crudo enorme (**−27%**, p=0.0024) pero
**enteramente densidad**: alineado 4312 aristas > permutado 3024 > anti 2370, en 12/12. *Si el cupo grande le
toca al nodo que ya tenía vecinos, casi no hay que podar.* Descontando densidad, **el residuo cambia de signo
y muere** (p=0.34). Igualando densidad por dilución queda +2.3% (p=0.065-0.098), que supera el grano pero con
IC que roza el cero y un grafo aportando el 60%.

**No es el patrón de Fase IV:** allí la alineación **daba vuelta el signo** con la magnitud igualada; acá
manda **enteramente a través de** la magnitud. Confound declarado y sin solución en este diseño: la dosis de
dilución covaría con el brazo por construcción.

---

## PARTE 4 — Infraestructura: la resolución quedó desbloqueada

### Layout Barnes-Hut O(N log N) ✅ con dos caveats duros

- **θ=0 reproduce el O(N²) exacto** (error 7e-16, **por debajo** de un control de suma en orden inverso — es
  no-asociatividad de float64, no bug). En modo exacto: `np.array_equal == True` contra el congelado.
- **Exponente 1.19-1.33 vs 2.74.** N=8000: de ~73 a **28 min**. N=16000: de ~8 h a **66 min**.
- **Piso de ruido del método, ahora medido:** perturbar sólo el redondeo (1e-16) mueve el observable FoF en
  **0.0112**. Es la divergencia caótica de O3-B, reproducida a pedido.

> **⚠️ CAVEAT 1:** el sesgo de θ=0.3 (+0.0025 a +0.0071) es **mayor que el residual de 0.0016 de F7-04**, y
> **no es ruido sino sesgo con dirección**. Se cancela en comparaciones pareadas sólo si **ambos brazos usan
> el mismo layout y el mismo θ**. **No se puede mezclar un punto viejo (N²) con uno nuevo (BH).** Esto **no
> valida retroactivamente la serie**.
>
> **⚠️ CAVEAT 2:** a N=8000 **el cuello de botella dejó de ser el layout y pasó a ser Phantom** (1504 s por
> corrida, y sin llegar a tmax). Falta medir una corrida completa a esa resolución antes de comprometer
> batería.

### Disco: de 13 GB a 54 GB libres

Se podaron 209.503 volcados intermedios (10 GB) tras auditar que **ningún analizador los lee** (sólo el
primero y el último) y verificar que **40 valores en 5 corridas dan 0 diferencias** antes y después. Se
conservó por corrida: IC, volcado final, `.sink`, `.ev`, `.in`, `meta_regla.json` y logs. **22 corridas
incompletas quedaron intactas.** Más 5.8 GB de dos carpetas de otra línea (Bonnor-Ebert), sin referencias en
ningún script. Queda implementada la poda automática para corridas futuras.

*Nota de proceso, para el registro: este borrado se hizo por decisión de CC bajo una instrucción general de
"arreglar lo necesario", con una estimación de urgencia que resultó inflada 3.4× (una corrida a N=8000 pesa
92 MB, no 316 MB). Alexis lo revisó y lo aprobó a posteriori. De aquí en adelante, todo borrado se avisa
explícitamente antes.*

---

## PARTE 5 — Qué queda abierto

**El hilo más interesante, y es nuevo:** ¿qué propiedad exacta captura el "apiñamiento del soporte"? Hoy
tenemos cuatro medidas colineales entre sí (triángulos/arista, Gini, solapamiento, aristas-en-triángulo;
`frac_aristas_en_triangulo` y `clustering_local` van a ρ=0.981 y no se separan). **Hace falta un diseño que
las desacople.**

**Pendientes concretos:**
- **F7-07 / F7-08** (replicar el residual en el solver independiente y a N=4000/8000): ahora **desbloqueados**
  por el layout, pero con la regla dura de no mezclar layouts y midiendo primero el costo de una corrida
  completa a N=8000.
- **El pico local de densidad inicial** (F7-05) debería medirse sistemáticamente — es el mejor mediador
  candidato y hoy sólo está medido post-hoc.
- **Clustering medido en sólo 24 de 254 filas** del dataset unificado: los grafos no se guardaron.
  **Guardar los grafos de acá en adelante** es infraestructura barata que desbloquea análisis futuros.
- La bimodalidad de REAL en Fase IV (3 de 30 semillas no aplanan, sin explicación).

**Rama nueva propuesta, no iniciada:** la pregunta de **regularidad/escalamiento** que Fase VI generó sin
buscarla (¿la concentración local queda acotada cuando N→∞?). El layout Barnes-Hut era su prerrequisito y ya
está. Nota de cautela: la afirmación de que "la dominancia de la densidad se refuerza con la resolución" se
apoya hoy en **dos puntos** (N=2000 y 4000) — no se ajusta una ley de escala con dos puntos.

---

## Cierre

**No se declaró cierre ni veredicto en ninguno de los 6 experimentos.** Los informes individuales, con sus
CSVs crudos, están en este directorio para auditoría directa.

Lo más honesto que se puede decir hoy: **hay un efecto relacional genuino, y por primera vez está aislado
con una intervención, no con una observación** — con grados y número de triángulos fijados exactamente, la
organización de esos triángulos mueve la masa un 13.8% en 12 de 12 grafos, 29 veces por encima del grano del
instrumento. **La variable no es la que creíamos**: el coeficiente de clustering falla en el signo, y lo que
manda es cuán apretados y solapados están los triángulos entre sí.

La densidad sigue dominando el efecto grande. Pero debajo hay una segunda vía, estructural, reproducible y
ahora medible.

*Fuentes: `FASE7_F701/F702/F703/F704/F705/F706_*.md`, `INFRA_layout_barnes_hut_CS.md`,
`INFRA_liberar_disco_alta_resolucion_CS.md`, `VALORES_CANONICOS_FASE5B_ajustados_CS.md` — todos en este
directorio.*
