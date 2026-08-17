# FASE VIII · F8-02 — El pico local de densidad inicial, **manipulado a propósito**

**12 de agosto de 2026** · **Ejecuta:** CC (Claude) · Ola 2 del `FASE8_PLAN_EJECUCION_CS.md`
**Lo que se interviene:** el único camino que sobrevivió a todos los controles de
`FASE7_F705_mediacion_nueva_CS.md` — el **pico local de densidad del gas inicial** (`p90/mediana` de la
densidad a 8 vecinos), r parcial **+0.64 a +0.90** en los 6 experimentos, **pero hasta hoy sólo medido
post-hoc, nunca manipulado.**

> **60 corridas nuevas de Phantom** (12 condiciones iniciales base × 5 niveles de pico).
> No se regeneró ningún grafo. No se modificó ningún script, CSV ni carpeta existente.
> **No se declara cierre ni veredicto. Sin commits.**

---

## 0. En simple, con analogía

Hasta ahora sabíamos esto: las nubes de gas que **nacen con grumos más picudos** terminan con más masa
en sumideros. Pero lo sabíamos **mirando**, no **tocando**. Y mirar no alcanza, porque hay un sospechoso
obvio: la topología del grafo fabrica el picudo *y* fabrica la masa, y entonces el pico sería un simple
acompañante — el copiloto que va en el auto pero no maneja.

Esta tarea agarra el volante. Toma nubes que **ya existían**, deja **todo** igual —la misma cantidad de
gas, la misma masa, la misma caja, las mismas velocidades y **exactamente el mismo grafo**— y les
**aprieta o afloja los grumos con los dedos**. Cinco niveles de apretón por cada nube. Después las mete
al horno (Phantom) con el protocolo de siempre y mira cuánta masa se enciende.

Si al apretar sale más masa y al aflojar sale menos, **con todo lo demás clavado**, el pico maneja.
Si no cambia nada, era copiloto.

---

## 1. El observable, declarado ANTES de correr nada

**Principal: la fracción de masa en sumideros** — es decir, el **criterio de densidad real de Phantom**
(`rho_crit_cgs = 1000`, que `FASE8_F805_f703_solver_independiente_CS.md` midió en **49.453 × la densidad
media** de estas cajas).

**No se usa FoF laxo.** F8-05 dejó una advertencia dura que esta tarea respeta al pie de la letra: sobre
los mismísimos dumps, el FoF a `ell = 1.0` **invierte el signo** del resultado respecto de la masa en
sumideros (+0.01433 con 12/12 contra −0.02275 con 2/12). El observable se eligió antes, por corresponder
a la pregunta física, no por dar el número más lindo.

**Secundarios, también declarados de antemano:** el pico local logrado en la condición inicial, el
tiempo al primer sumidero y κ_V.

**Grano del instrumento:** 1 partícula = **0.0005** de fracción de masa a N=2000. **Piso práctico de un
pareado: ~5 partículas = 0.0025** (medido en F8-01). Por debajo de eso se dice "por debajo del piso".

---

## 2. Cómo se manipuló el pico: una compresión radial suave, y por qué ésa

De las dos vías sugeridas se eligió la **transformación radial suave**. Para cada burbuja de centro `c`
y radio `R`, con `u = |x − c| / R`:

```
        x'  =  c + (x − c) · g(u)          g(u) = 1 − a · (1 − u²)²
```

| `a` | qué hace | efecto sobre el pico |
|---|---|---|
| `a = 0`   | identidad exacta | nivel de control interno |
| `a > 0`   | el núcleo se comprime (cerca del centro, un factor 1−a en cada eje) | el pico **sube** |
| `a < 0`   | el núcleo se expande | el pico **baja** |

Las tres propiedades por las que se eligió ésta, y no otra:

1. **Conserva N y la masa por identidad, no por suerte.** No mueve masa entre partículas: mueve
   posiciones. Masa total = N × m_particula, y ninguno de los dos cambia.
2. **Es local de verdad.** Fuera del radio `R` **no se mueve ni un dígito**. No es "casi intacto": es
   intacto, exactamente, y se verifica partícula por partícula.
3. **No deja escalón de densidad en el borde.** El mapa es C¹ en `r = R` (posición *y* derivada
   coinciden con la identidad), así que no queda una cáscara artificial. Además es monótono
   (`dr'/dr = 1 + a(1−u²)(5u²−1) > 0` para |a| ≤ 0.8) y cumple `u·g(u) ≤ 1`: **ninguna partícula sale
   de su burbuja**.

**Dónde están las burbujas:** en los **máximos locales de densidad**. Se ordenan las partículas por
densidad a 8 vecinos y se toman las más densas de a una, salteando las que caigan a menos de `2R` de un
centro ya aceptado. Así **las burbujas son disjuntas** — verificado con `assert`: ninguna partícula
pertenece a dos.

**Los centros y `R` se calculan UNA sola vez por condición inicial, sobre la nube original, y se usan
idénticos en los cinco niveles.** Lo único que cambia entre niveles de una misma IC es el número `a`.
Es una familia de un solo parámetro que pasa por la identidad: el pareado más limpio que se puede armar.

Parámetros (fijados en una calibración previa sobre 2 IC, antes de comprometer la batería):
`R = 1.0 × separación media` (= 7.75 en una caja de 97.6), **30 centros**, `a ∈ {−0.35, 0, +0.20, +0.35,
+0.50}` → niveles **L0 · L1 · L2 · L3 · L4**.

---

## 3. Las 12 condiciones iniciales base: se reusaron, no se regeneraron

Salen todas de **`F5B_40pares`** (Fase V-B, N=2000) — **un solo experimento a propósito**: F7-05 §5.1
cazó una paradoja de Simpson por mezclar diseños y resoluciones, y este diseño no la quiere cerca. Se
eligieron 12 espaciadas por cuantiles del pico ya medido, para cubrir el rango del observable.

| regla | clase | pico original | aristas | grado medio | masa en sumideros (histórica) |
|---|---|---:|---:|---:|---:|
| A2-B0-C2-batch4-r47 | III | 6.03 | 3282 | 3.282 | 0.0930 |
| A2-B0-C2-batch3-r120 | I | 7.25 | 3396 | 3.396 | 0.0925 |
| A2-B0-C2-r2 | I | 7.53 | 4094 | 4.094 | 0.0760 |
| A2-B0-C2-r14 | III | 7.87 | 3296 | 3.296 | 0.0990 |
| A2-B0-C2-batch4-r31 | I | 8.16 | 3332 | 3.332 | 0.0985 |
| A2-B0-C2-r39 | III | 8.34 | 4083 | 4.083 | 0.0785 |
| A2-B0-C2-batch4-r12 | III | 8.79 | 3340 | 3.340 | 0.1005 |
| A2-B0-C2-batch3-r143 | I | 10.08 | 3349 | 3.349 | 0.1000 |
| A2-B0-C2-batch4-r1 | III | 11.43 | 3224 | 3.224 | 0.1075 |
| A2-B0-C2-r1 | I | 12.26 | 4078 | 4.078 | 0.0870 |
| A2-B0-C2-batch3-r111 | III | 14.90 | 3017 | 3.017 | 0.1220 |
| A2-B0-C2-r17 | III | 18.35 | 3165 | 3.165 | 0.1210 |

**El grafo no se regeneró ni una vez.** Se copió el `.grafo.gz` que F8-00 dejó sellado con sha256 y se
verificó el sello al leerlo y al releerlo ya copiado. Los cinco niveles de una IC **comparten el mismo
sello** — es la afirmación central del diseño y está verificada como dato, no como intención.

---

## 4. Lo pedido contra lo LOGRADO: cuánto se movió de verdad el pico (y qué NO se movió)

**Lo que quedó exacto** (verificado releyendo del disco, no confiando en la memoria del programa):

| control | resultado |
|---|---|
| nº de partículas | **2000 en las 60**, un solo valor |
| masa total | **18800.0 en las 60**, un solo valor (m_particula = 9.4, sin tocar) |
| velocidades y `h` de cada partícula | **idénticas al original en las 60** (`np.array_equal`) |
| cabecera y línea de `phantomsetup` | copiadas verbatim, idénticas |
| grafo (sello sha256) | **1 solo sello por IC a través de sus 5 niveles**, 12/12 |
| nº de aristas | 1 solo valor por IC a través de sus 5 niveles, 12/12 |
| nivel identidad (L1) | **byte a byte idéntico al archivo original**, 12/12 (md5) |
| burbujas disjuntas | ninguna partícula en dos burbujas (assert), 60/60 |

**Lo que se movió, medido:**

| nivel | `a` | pico logrado / pico original (mediana) | rango | partículas movidas | desplazamiento máx. |
|---|---:|---:|---|---:|---:|
| **L0** | −0.35 | **×0.852** | ×0.68 a ×1.04 | 625–797 (36.2% medio) | 0.78 |
| **L1** | 0.00 | ×1.000 | — | 0 (identidad) | 0.00 |
| **L2** | +0.20 | **×1.414** | ×1.17 a ×1.58 | 625–797 (36.2% medio) | 0.44 |
| **L3** | +0.35 | **×1.923** | ×1.48 a ×2.22 | 625–797 (36.2% medio) | 0.78 |
| **L4** | +0.50 | **×2.904** | ×2.21 a ×3.38 | 625–797 (36.2% medio) | 1.11 |

(Las burbujas son las mismas en los cinco niveles de una IC — de ahí que el conjunto de partículas
tocadas no cambie; en L1 el mapa es la identidad y ninguna se mueve.)

En valor absoluto el pico recorre **de 5.77 a 62.0** — el rango del corpus entero de 254 corridas (5.6 a
34) y un poco más arriba. Dentro de **una misma** condición inicial, el pico recorre **×2.21 a ×4.85**.

**Y lo que NO se movió, que es la mitad del punto:** la **geometría GLOBAL** — la masa que nace en grumos
FoF b=0.30, que F7-05 mostró que es la densidad disfrazada (r = −0.9945 con el grado medio) — se mueve
**entre ×0.9878 y ×1.0038**. Dentro de una IC, como mucho **×1.013**.

> **En una frase: el eje-2 de F7-05 se movió hasta ×4.85 y el eje-1 se quedó quieto dentro del 1.3%.**
> Eso es exactamente la disección que hasta hoy no se había hecho.

Un dato honesto de la calibración que los números confirman: **bajar el pico es mucho más difícil que
subirlo.** Aflojar un grumo baja el p90 pero también baja la mediana, y el cociente se mueve poco: con
a = −0.35 el pico bajó en **10 de 12** IC (hasta ×0.68) y en 2 subió levemente (×1.02 y ×1.04). Por eso
la escalera es asimétrica y por eso **la monotonía se juzga siempre contra el pico logrado, nunca contra
`a`**.

---

## 5. El resultado principal: la masa sigue al pico, monótonamente, en todas las nubes

**Observable: fracción de masa en sumideros (criterio de densidad de Phantom).**
Estadística de bloques sobre las **11 IC con los 5 niveles completos** (la que falta, en §8).

```
   Page (L) para tendencia ordenada:   z = +6.63    p = 1.6e-11
   Friedman (¿alguna diferencia?):     χ² = 44.0    p = 6.4e-09
   ESTRICTAMENTE CRECIENTE EN LOS 5 NIVELES:  11 de 11 condiciones iniciales
```

Contrastes pareados contra el nivel identidad (1 partícula = 0.0005; **piso práctico = 5 partículas**):

| contraste | Δ mediano | **en partículas** | signos | Wilcoxon | ¿supera el piso? |
|---|---:|---:|---:|---:|---|
| **L0 − L1** (pico bajado) | **−0.01500** | **−30.0** | **0/11** | 9.8e−04 | **sí, ×6** |
| **L2 − L1** | +0.01050 | **+21.0** | **11/11** | 9.8e−04 | **sí, ×4** |
| **L3 − L1** | +0.02250 | **+45.0** | **11/11** | 9.8e−04 | **sí, ×9** |
| **L4 − L1** | **+0.06200** | **+124.0** | **11/11** | 9.8e−04 | **sí, ×25** |
| **L4 − L0** (extremo a extremo) | **+0.07750** | **+155.0** | **11/11** | 9.8e−04 | **sí, ×31** |

(9.8e−04 es el **mínimo p alcanzable** con el Wilcoxon pareado a n=11: 11/11 signos iguales.)

**Dentro de cada condición inicial**, la correlación entre el pico logrado y la masa:

```
   Spearman por IC:  mediana = +1.000     11 de 11 positivos     rango +0.90 a +1.00
   agrupado, centrando por IC (efectos fijos):   ρ = +0.935  (p = 1.3e-25),  r = +0.865
```

Nueve de las once IC dan Spearman **exactamente +1** (orden perfecto en los cinco niveles); las dos que
dan +0.90 son las mismas dos a las que bajarles el pico no funcionó (§4) — y aun así su masa ordena
perfecto respecto del pico logrado en los otros cuatro niveles.

**El tamaño, en las unidades de la línea:** la pendiente mediana es **+0.128 de fracción de masa por
década de pico** = **+256 partículas por década**, con rango +0.088 a +0.179 entre las 11 IC.

### 5.1 ¿La intervención reproduce la pendiente que ya se veía observando?

Es la comparación que más dice sobre si el camino es el mismo:

| de dónde sale la pendiente | pendiente (frac. de masa por década de pico) | en partículas |
|---|---:|---:|
| **manipulando** (esta tarea, dentro de IC, mediana de 11) | **+0.128** | +256 |
| observando, `F5B_40pares` (n=76, las mismas reglas de donde salen las IC) | +0.092 | +183 |
| observando, `O3B_rewiring` (n=24, la única otra intervención del corpus) | +0.120 | +240 |
| observando, todo N=2000 (n=228) | +0.084 | +168 |
| observando, `F5B_40pares` **descontando la densidad** | +0.053 | +107 |

La pendiente manipulada es **del mismo orden** que la observada y algo más empinada (1.4× la de F5B
cruda, 2.4× la de F5B con la densidad descontada). Traducción: la intervención no está inventando una
física nueva ni un efecto exagerado — cae **encima de la relación que ya se veía**, un poco por arriba.

---

## 6. Los secundarios: qué más se movió, y todos en la misma dirección

| secundario | L0 (mediana) | L1 | L2 | L3 | L4 | Page (L) |
|---|---:|---:|---:|---:|---:|---|
| **tiempo al primer sumidero** | 0.0635 | 0.0375 | 0.0245 | 0.0160 | **0.0080** | tendencia **decreciente**, p = 1.7e−11 |
| **κ_V agregado** | 0.5634 | 0.5076 | 0.3825 | 0.2946 | **0.2669** | tendencia **decreciente**, p = 1.8e−10 |
| **nº de sumideros** | 8.5 | 8 | 8 | 9 | **16.5** | tendencia creciente, p = 1.6e−05 |
| geometría global FoF b=0.30 (eje-1) | — | — | — | — | — | Δ mediano L4−L1 = **−0.0015** (0.2%) |

El **tiempo al primer sumidero cae un factor 8** de punta a punta: con el pico alto, la nube enciende
casi ocho veces antes. Es la lectura más física de todo el experimento — apretar el grumo no cambia
"cuánta materia hay", cambia **cuándo y cuánto colapsa la que ya estaba**.

**Un contraste con F7-03/F8-05 que conviene anotar:** allá, el número de sumideros **no cambiaba** entre
brazos (8.08 contra 8.08) y sólo cambiaba cuánto comía cada uno. Acá, en el nivel más apretado, **cambian
las dos cosas**: hay el doble de sumideros *y* más masa. Manipular el pico a mano no es lo mismo que
mover la organización de los triángulos: es una palanca más gruesa.

---

## 7. El control que decide si esto es física o trampa: ¿le regalamos los sumideros a Phantom?

Si el gas apretado ya nace por encima del umbral de sumidero, el experimento no diría nada de la
gravedad: le habríamos puesto los sumideros ya prendidos en la mano.

Se hizo la cuenta con **el mismo estimador de densidad que usó F8-05**
(`rho = k·m/((4/3)πr_k³)`, k=8) y el umbral real de Phantom (`rho_crit = 1000` = **49.442× la densidad
media** de estas cajas):

| nivel | densidad máxima en la IC (× la media, mediana) | partículas sobre el umbral (mediana / máx) | IC con alguna |
|---|---:|---:|---:|
| L0 | 3.285 | **0 / 0** | **0 de 12** |
| L1 | 8.035 | **0 / 0** | **0 de 12** |
| L2 | 15.603 | **0 / 0** | **0 de 12** |
| L3 | 28.899 | **0 / 0** | **0 de 12** |
| **L4** | 62.825 | **4.5 / 13** | **11 de 12** |

**Lectura, sin adornos:**
- En **L0, L1, L2 y L3 no hay ni una sola partícula** por encima del umbral en ninguna de las 12 nubes —
  igual que lo que F8-05 encontró en las IC de F7-03. Y sin embargo L2 ya da **+21 partículas** y L3
  **+45**, ambas muy por encima del piso. **Ese efecto es 100% dinámica.**
- En **L4 sí hay**: entre 1 y 13 partículas por nube (mediana 4.5), o sea **hasta el 0.65% de la masa**.
  El efecto de L4 es **+124 partículas**. Aunque toda esa masa "pre-encendida" terminara en sumideros,
  explicaría **como mucho el 10%** del efecto de L4. Pero está, y se declara: **L4 es el único nivel
  contaminado por ese mecanismo, y el resto de la escalera no lo está.**

---

## 8. Lo que no se pudo medir, y por qué (nada se esconde)

**Una corrida de 60 no llegó a t = 0.500: `A2-B0-C2-batch3-r143` en el nivel L4.** Phantom la abortó él
mismo en t = 0.4525 con su guardián de conservación:

```
ERROR! evolve: Large error in angular momentum conservation : err = 1.152E-01
FATAL ERROR! evolve: Conservation errors too large to continue simulation
```

**No se forzó** (existe la variable de entorno `I_WILL_NOT_PUBLISH_CRAP=yes` para saltear ese guardián:
no se usó). Esa IC queda **fuera de la estadística de bloques** — por eso los tests van sobre 11 IC, no
12 — y **queda en el CSV crudo con su dump final marcado**. Sus cinco valores, para que se vean:

```
   L0 = 0.0890   L1 = 0.1000   L2 = 0.1090   L3 = 0.1220   L4 = 0.1375*  (*t = 0.4525, no comparable)
```

Es decir: iba creciendo igual que las otras once, y su L4 truncado ya superaba a su L3. Que el nivel más
apretado sea el que rompe el integrador **es en sí un dato**: la escalera de compresión tiene un techo
numérico y lo tocamos con `a = +0.50`.

---

## 9. Las cinco limitaciones que hay que leer junto con estos números

1. **La manipulación mueve el pico, pero no SÓLO el pico.** El CV de la densidad local viaja con él
   (agrupado, centrando por IC: ρ = +0.959 con la masa, contra +0.935 del pico). Este diseño **no puede
   separar** "cuán alto llega el pico" de "cuán desparejo quedó todo el reparto local". Lo que se
   manipuló, dicho con precisión, es **cuán apretados están los grumos**; `p90/mediana` es un resumen de
   eso, no la única cosa que cambió.
2. **Muestra que el pico ALCANZA, no que sea NECESARIO ni que sea EL mediador.** Que empujando el pico
   se mueva la masa dice que la física responde a esa palanca. **No** dice que en el corpus observacional
   la variación del pico sea lo que transporta el efecto de la topología. Ese es exactamente el control
   complementario: **F8-03 (mismo pico, distinta topología)**.
3. **La transformación es exógena.** El `layout_resortes` no puede fabricar estas nubes; se las
   fabricamos nosotros. Son nubes físicamente legítimas (masa, N, caja y velocidades intactas, mapa
   monótono y C¹), pero **no son nubes que el generador produzca**.
4. **Las velocidades se heredaron partícula por partícula.** Esa fue la elección: conserva la energía
   cinética y el momento total exactamente. El precio es que dentro de un grumo comprimido el campo
   turbulento queda muestreado en posiciones corridas hasta 1.11 en una caja de 97.6 (≈1%); la caja
   envolvente crece a lo sumo un 0.31% en el nivel expandido.
5. **12 condiciones iniciales, todas A2-B0-C2, todas N=2000, todas de `F5B_40pares`.** No se sabe si vale
   afuera, ni a otra resolución.

---

## 10. Costos

- Generación de las 60 condiciones iniciales (con todas las verificaciones y la vara FoF): **32 s**.
- Phantom: 60 corridas, **~124 s por IC** cronometrados en el piloto de 1 IC (5 niveles, serial, máquina
  descargada). En la batería completa, con 3–5 procesos en paralelo y la máquina con carga ~900 por otros
  experimentos de la fase corriendo a la vez, cada corrida tardó **18 a 60 s** y el total fue
  **~35 min de reloj**.
- Análisis + figura: ~60 s. Control de umbral sobre las 60 IC: ~30 s.

---

## 11. Archivos

**Nuevos (esta tarea):**

| archivo | qué es |
|---|---|
| `cs090_fase8_f802_pico.py` | selecciona las 12 IC base, fabrica los 5 niveles, verifica y copia el grafo sellado |
| `cs090_fase8_f802_correr.py` | Phantom sobre las 60 carpetas, protocolo estándar CS073 |
| `cs090_fase8_f802_analizar.py` | verificación cruzada, Page/Friedman/Wilcoxon, correlaciones, figura |
| `cs090_fase8_f802_umbral_ic.py` | el control del §7: partículas sobre el umbral real de Phantom en la IC |
| `cs090_fase8_f802_ic_transformadas.csv` | **una fila por (IC × nivel)**: lo pedido y lo logrado, con todas las verificaciones |
| `cs090_fase8_f802_crudo.csv` | **CSV crudo**: las 60 corridas, IC + Phantom |
| `cs090_fase8_f802_por_ic.csv` | una fila por IC base: Spearman, pendiente por década, Δ extremos |
| `cs090_fase8_f802_estadistica.csv` | Page, Friedman, contrastes pareados, correlaciones |
| `cs090_fase8_f802_umbral_ic.csv` | densidad máxima y partículas sobre `rho_crit` en cada IC |
| `cs090_fase8_f802_pico.png` | los cuatro paneles |
| `cs090_fase8_f802_{pico,analisis,umbral_ic}.log`, `cs090_fase8_f802_shard_L*.log` | registro completo |

**Sólo importados, nunca modificados:** `cs090_fase8_f800_grafos.py`, `cs090_fase6_o4a_observable_comun.py`,
`cs090_fase5b_analizar.py`.
**Leídos, nunca escritos:** `cs090_fase8_f800_dataset_enriquecido.csv`, los 12 `.grafo.gz` de
`grafos_f800/F5B_40pares/`, y las 12 `cosmogenesis_ic.txt` de las baterías de Fase V-B.
**Batería nueva:** `/Users/alexis/phantom_cs073/bateria_fase8_f802_pico/` (60 carpetas, 1.2 GB).

---

## 12. Qué dicen estos números sobre la pregunta central (sin cerrar nada)

La pregunta era: **¿el pico local causa más masa acretada, o sólo la acompaña?**

Lo que hay sobre la mesa: con el grafo clavado (mismo sello sha256), la masa clavada (18800), N clavado,
la caja clavada y las velocidades clavadas partícula por partícula, **mover el pico mueve la masa**:
11 de 11 nubes ordenan perfecto a través de los cinco niveles, el efecto va de **−30 a +124 partículas**
contra un piso de 5, el tiempo al primer sumidero cae un factor 8, y la pendiente que sale de manipular
(+0.128 por década) **cae encima de la que ya se veía observando** (+0.092 en las mismas reglas).
Y en los tres niveles donde **ninguna** partícula nace por encima del umbral de Phantom, el efecto
igual está (+21 y +45 partículas).

Lo que estos números **no** deciden: si en el corpus observacional el pico es el eslabón por donde viaja
el efecto de la topología, o si es una palanca que existe pero que la topología no usa. Esa es la
pregunta de **F8-03** — mismo pico, distinta topología —, y con este resultado en la mano ese control
pasa a ser el que puede cerrar (o romper) la cadena `topología → pico local → masa`.

> No declaro cierre. La interpretación es de Alexis.
