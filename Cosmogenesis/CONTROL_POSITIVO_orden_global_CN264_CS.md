# CONTROL POSITIVO del instrumento — ¿los jueces de "orden global" ven orden global cuando lo hay?

**Nodo bajo control:** C-N2.6.4 ("la acumulación de constricción produce organización a gran escala"),
anotado como **refutado en su mitad global** por CS067B + CS068 (Mundo B) + CS069B + Fase III.

**Esto NO es un intento de rescatar el nodo.** Es un control positivo del INSTRUMENTO. La pregunta no es
"¿aparece orden global en una retícula?" (la retícula se construye ordenada, sería una identidad, no una
medición). La pregunta es: **¿el observable con el que se declaró "no hay orden global" es capaz de
DETECTAR orden global cuando sabemos que lo hay?**

**Codea/ejecuta:** CC. **Fecha:** 13-ago-2026. **No se declara cierre.** La lectura final es de Alexis.

---

## 0. Analogía de arranque

Es como preguntarse si el detector de humo de la casa funciona. No alcanza con que nunca haya sonado: hay
que acercarle un fósforo encendido. Si suena, el silencio de los últimos meses significa "no hubo fuego".
Si no suena ni con el fósforo en la boca, el silencio no significaba nada — había que cambiar el detector.
Acá el fósforo son retículas 1D/2D/3D/4D y grafos con dirección sembrada a mano; el detector son los tres
jueces con los que el arco CS067-068-069 y Fase III declararon "no hay orden global".

---

## 1. PRE-REGISTRO — escrito ANTES de calcular, no modificado después

### 1.1 Los observables identificados (leídos en el código, no en los informes)

Del rastreo de `ADJUDICACION_CS067_bifurcacion_ab_CS.md`, `ADJUDICACION_CS068_paso2b_cierre_arco_CS.md`,
`ADJUDICACION_CS069_tanda_cierre_CS.md`, `FASE3_renormalizacion_resultado_CS.md` y de los scripts
`cs067_habitacion_completa.py`, `cs068_paso2b_diametro.py`, `cs069_quantum_graph.py`, `cs069_tanda.py`,
`cs080_renormalizacion.py`: **son TRES jueces distintos, no uno.** Se controlan por separado.

| id | nombre en el arco | función exacta | criterio de "detecta" usado por el arco |
|---|---|---|---|
| **J-A** | Juez A — cedazo de π | `cs069_quantum_graph.cedazo_pi()` → `pi_cv` | π_local(r)=\|S(r)\|/(2r); **CV BAJO** = "π se congela" = espacio métrico plano coherente. En el arco real dio 1.0–1.1 = "estalla" |
| **J-B** | Juez B — pendiente log-log del diámetro | `_pendiente_loglog(log N, log diam)`; `H._diam_robusto` (clásico) / `Q.diam_q_robusto` (cuántico) | **pendiente > 0.3** = "hay lejos real"/métrico. Umbral pre-inscrito por CS en `cs068_paso2b_diametro.py`, calibrado contra retícula2D=0.52, anillo-de-cliques=1.01, small-world=0.14 |
| **J-B'** | magnitud absoluta del diámetro | mismo `diam` a N fijo | El ruling de CS068 Paso 2b dice explícitamente que **la magnitud, no la pendiente, fue el juez decisivo**: residual 6.0–7.5 vs métrico-2D ~58–96 → "13x menor" |
| **J-C** | Juez C — gap espectral + candado de picado | `Q.juez_gap_espectral()` = `H.cuenta_ejes_gap()` + `H.picado_por_nodo()` sobre embedding MDS | **n_ejes ≥ 1 con gap limpio Y `pico_medio > 0.85` (`certificado=True`)**. En las 96 corridas de CS069: n_ejes=0, certificado 0% |

Fase III (`cs080_renormalizacion.py`) usó **J-B bajo otro protocolo** (variar N_b agrupando el MISMO grafo
en cajas, no generando grafos nuevos) más `dim_volumen`. El propio informe de Fase III ya reportó que bajo
ese protocolo el umbral 0.3 **no discrimina** (el piso Erdős-Rényi da 0.406). Ese hallazgo se toma como
dado y se anota; el control positivo se hace sobre el protocolo ORIGINAL (grafos nuevos por N), que es el
que sostiene el veredicto de CS068/CS069.

### 1.2 Criterios de detección — absolutos, y por lo tanto FALLABLES

Se fijan en valores absolutos (no relativos a un NULL) para que el control pueda fallar:

- **J-A detecta** si `pi_cv < 0.5`. Justificación: el sustrato real dio 1.0–1.1 en las 96 corridas de
  CS069; una reducción a menos de la mitad es una separación inequívoca. (Si un sustrato perfectamente
  ordenado no baja de 1.0, el juez es ciego.)
- **J-B detecta** si `pendiente > 0.3` (el umbral literal pre-inscrito por CS, sin tocar).
- **J-B' detecta** si el diámetro a N≈2500 es **≥ 3×** la referencia small-world que CS068 usó a esa N
  (~13) — es decir, `diam(N=2500) ≥ 39`. (CS068 declaró Mundo B porque el residual daba 7.5.)
- **J-C detecta** si `certificado == True` (`pico_medio > 0.85`) **y** `n_ejes ≥ 1`.

### 1.3 Sustratos (el "fósforo")

**Con orden global CONOCIDO (positivos — el instrumento DEBE sonar):**
1. `anillo1d` — anillo/cadena 1D periódica. Orden global máximo, dimensión 1.
2. `reticula2d` — retícula cuadrada periódica (toro). Orden global evidente, d=2.
3. `reticula3d` — retícula cúbica periódica, d=3.
4. `reticula4d` — retícula hipercúbica periódica, d=4. (Se agrega para probar la GUARDA 1: ver si el
   umbral 0.3 de J-B es alcanzable por una geometría de dimensión alta.)
5. `aniso2d` — retícula 2D anisótropa (enlaces en y sobreviven con p=0.25): dirección global sembrada a mano.
6. `flujo_capas` — grafo en capas (nodo → capa siguiente): flujo neto / gradiente impuesto a propósito.

**SIN orden global (negativos — el instrumento NO debe sonar):**
7. `er` — Erdős-Rényi, mismo N y grado medio.
8. `real_barajado` — barajado con grados preservados (`_double_edge_swap`) del grafo real.
9. `real` — el sustrato del arco (`E._sustrato`), como referencia de lo que efectivamente se midió.

### 1.4 Dos vías de aplicación (el observable NO se modifica en ninguna)

- **Vía Q (instrumento completo, tal cual):** se corre `Q.brazo_completo()` sobre la adyacencia de cada
  sustrato → matriz D_q de la integral de camino → J-A, J-B, J-C exactamente como en `cs069_tanda.py`.
  Esto controla el instrumento COMO SE USÓ.
- **Vía M (observable sobre la métrica desnuda):** las mismas funciones J-A y J-C aplicadas a la matriz de
  distancias BFS del grafo, y J-B con `H._diam_robusto`. Esto separa "el observable es ciego" de "la
  cañería cuántica que lo alimenta es ciega".

### 1.5 Lecturas pre-inscritas

- **Detecta en los ordenados y no en los desordenados** → instrumento sano; **la refutación de C-N2.6.4 en
  su mitad global queda CONFIRMADA.**
- **No detecta ni donde sabemos que hay orden** → instrumento ciego; **la refutación queda ANULADA** y el
  nodo vuelve a "sin medir".
- **Detecta en algunos ordenados y en otros no** → se dice en cuáles y qué TIPO de orden global no puede
  ver; la refutación queda **ACOTADA** a ese tipo de orden.

### 1.6 Guardas declaradas de antemano (se reportan las cuatro, salgan como salgan)

1. **Identidad algebraica.** Se busca activamente si algún juez es constante *por construcción* en alguna
   dimensión/densidad/tamaño, como pasó hoy con π(r)=\|S(r)\|/2r ~ r^(d−2).
2. **¿El número podía salir distinto?** Para cada juez se verifica que el criterio sea alcanzable.
3. **Bug `_diam`.** Todo diámetro se mide sobre la componente GIGANTE (`cs090_diam_corregido.diam_gigante`
   además de `H._diam_robusto`, y se comparan).
4. **El barajado debe destruir algo medible** y no ser isomorfo al real: se verifica con números
   (clustering, triángulos, solapamiento de aristas).

---

*(Nada de lo escrito arriba se modificó después de correr.)*

---

## 2. Qué se corrió

- `cs092_control_positivo_orden_global.py` — 9 sustratos × 3 escalas de N × 2 semillas = **54 corridas**,
  cada una medida por las DOS vías (M y Q) con los tres jueces. 7.4 min. Crudo:
  `cs092_control_positivo_crudo.json`.
- `cs092_guardas.py` — las cuatro guardas + una quinta que apareció al correr (G1b, horizonte L=8).
- `cs092_tabla.py` — agregación → `cs092_control_positivo_tabla.csv` + `cs092_control_positivo.png`.
- Figura de guardas: `cs092_guardas.png`.

Ninguna función de juez se reescribió: `Q.cedazo_pi`, `Q.diam_q_robusto`, `Q.juez_gap_espectral`,
`H.cuenta_ejes_gap`, `H.picado_por_nodo`, `H._diam_robusto`, `Q.brazo_completo` se importan tal cual.

---

## 3. LA TABLA — observable × sustrato

Valores a la N más grande de cada escalera, media de 2 semillas. `M` = observable sobre la métrica desnuda
(BFS). `Q` = el instrumento completo tal cual CS069 (integral de camino con L=8). "NO MEDIBLE" = el juez
devuelve NaN, no un número.

### 3.1 J-A (π-CV) y J-B (pendiente log-log del diámetro)

| sustrato | orden global | π-CV (M) | J-A vía M | π-CV (Q) | J-A vía Q | pend (M) | J-B vía M | pend (Q) | J-B vía Q |
|---|---|---|---|---|---|---|---|---|---|
| anillo1d | **SÍ** | 0.716 | ✗ no detecta | — | NO MEDIBLE | **1.000** | ✅ DETECTA | — | NO MEDIBLE |
| reticula2d | **SÍ** | 0.328 | ✅ DETECTA | 0.242 | ✅ DETECTA | **0.500** | ✅ DETECTA | — | NO MEDIBLE |
| reticula3d | **SÍ** | 0.510 | ✗ no detecta | 0.411 | ✅ DETECTA | **0.368** | ✅ DETECTA | 0.077 | ✗ no |
| reticula4d | **SÍ** | 0.494 | ✅ DETECTA | 0.556 | ✗ no | **0.309** | ✅ DETECTA (por 0.009) | 0.146 | ✗ no |
| aniso2d | **SÍ** | 0.403 | ✅ DETECTA | 0.470 | ✅ DETECTA | **0.333** | ✅ DETECTA | — | NO MEDIBLE |
| flujo_capas | **SÍ** | 1.271 | ✗ no detecta | 0.560 | ✗ no | **0.508** | ✅ DETECTA | 0.051 | ✗ no |
| er | no | 0.972 | ✗ (correcto) | 1.102 | ✗ (correcto) | 0.166 | ✗ (correcto) | 0.157 | ✗ |
| **real** | no | 0.641 | ✗ (correcto) | 0.654 | ✗ (correcto) | **0.236** | ✗ (correcto) | 0.153 | ✗ |
| real_barajado | no | 1.009 | ✗ (correcto) | 1.040 | ✗ (correcto) | 0.141 | ✗ (correcto) | 0.214 | ✗ |

### 3.2 J-B' (magnitud absoluta del diámetro) y J-C (gap + candado de picado)

| sustrato | orden global | diam (M) | J-B' vía M | diam_q (Q) | J-B' vía Q | n_ejes (M) | pico (M) | J-C vía M | n_ejes (Q) | pico (Q) | J-C vía Q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| anillo1d | **SÍ** | 1250 | ✅ DETECTA | — | NO MEDIBLE | 2 | 0.832 | ✗ no | 0 | 0.557 | ✗ no |
| reticula2d | **SÍ** | 50 | ✅ DETECTA | — | NO MEDIBLE | 0 | 0.732 | ✗ no | 0 | 0.589 | ✗ no |
| reticula3d | **SÍ** | 18 | ✗ **no detecta** | 19.1 | ✗ no | 6 | 0.660 | ✗ no | 0 | 0.689 | ✗ no |
| reticula4d | **SÍ** | 12 | ✗ **no detecta** | 19.0 | ✗ no | 0 | 0.629 | ✗ no | 0 | 0.656 | ✗ no |
| aniso2d | **SÍ** | 72.5 | ✅ DETECTA | — | NO MEDIBLE | 2 | 0.788 | ✗ no | 0 | 0.690 | ✗ no |
| flujo_capas | **SÍ** | 50.5 | ✅ DETECTA | 18.5 | ✗ no | 1 | 0.968 | ✅ DETECTA | 0 | 0.652 | ✗ no |
| er | no | 9 | ✗ (correcto) | 16.7 | ✗ | 0 | 0.652 | ✗ | 0 | 0.644 | ✗ |
| **real** | no | 14 | ✗ (correcto) | **36.3** | ✗ | 0 | 0.705 | ✗ | 0 | 0.690 | ✗ |
| real_barajado | no | 7.5 | ✗ (correcto) | 23.9 | ✗ | 0 | 0.690 | ✗ | 0 | 0.683 | ✗ |

Detalle que conviene mirar dos veces: en la vía Q el **blob real tiene el diámetro MÁS GRANDE de los nueve
sustratos** (36.3), por encima de la retícula 3D (19.1) y del grafo en capas (18.5). El "diámetro cuántico"
no ordena los sustratos por lejanía; ordena por otra cosa.

---

## 4. LAS GUARDAS — las cuatro pedidas, más una que apareció al correr

### GUARDA 1 — SÍ hay identidad algebraica, y es la misma que la de π

`cedazo_pi` calcula π_local(r)=|S(r)|/(2r). En una retícula d-dimensional |S(r)| ~ r^(d−1), luego
**π(r) ~ r^(d−2)**. Medido sobre retículas de dimensión conocida (`cs092_guarda1_perfil_pi.csv`, panel
izquierdo de `cs092_guardas.png`):

| d | exponente ajustado de π(r) | predicho (d−2) | π-CV | ¿J-A detecta? |
|---|---|---|---|---|
| 1 | **−1.000** | −1 | 0.716 | no |
| 2 | **−0.048** | 0 | 0.328 | SÍ |
| 3 | +0.766 | 1 | 0.510 | no |
| 4 | +1.360 | 2 | 0.494 | SÍ (marginal) |

En la retícula cuadrada π(r) es una **meseta exactamente en 2.000** en todo el rango. **J-A no mide
"planitud" ni "orden global": mide BIDIMENSIONALIDAD.** Un CV bajo es la firma de d=2 y de nada más. Por eso
el anillo 1D —el sustrato con orden global MÁXIMO— reprueba J-A (0.716), y reprueba **peor que el blob
real** (0.641). El juez califica al ovillo como más "ordenado" que a una recta perfecta.

Esto es la misma degeneración que se encontró hoy con π(r)=|S(r)|/2r ~ r^(d−2) en la otra línea. No es un
caso análogo: es literalmente la misma función.

### GUARDA 1b (no estaba pedida; apareció sola y es la más grave) — el horizonte L=8 de la vía cuántica

`Q._K_y_Dq` suma caminos hasta longitud **L=8** (fijo, `G-NO-CALIBRAR`). Todo par a más de 8 pasos queda
NaN. `diam_q_robusto` exige ≥30% de la fila finita; por debajo devuelve NaN. Fracción de pares que la
integral de camino alcanza (N≈2500, panel derecho de `cs092_guardas.png`):

| sustrato | diám BFS | pares alcanzados (L=8) | diam_q |
|---|---|---|---|
| anillo1d | 1250 | **0.7 %** | NaN |
| aniso2d | 73 | **2.8 %** | NaN |
| reticula2d | 50 | **5.8 %** | NaN |
| flujo_capas | 50 | 30.8 % | 18.5 |
| reticula3d | 18 | 36.3 % | 19.1 |
| **real** | 15 | **74.5 %** | 36.5 |
| er | 8 | **99.0 %** | 15.8 |

**El instrumento cuántico de CS069 tiene un horizonte de 8 pasos.** Sólo puede ver entero un sustrato cuyo
diámetro sea ≲8 — es decir, sólo puede ver mundos pequeños. Sobre cualquier sustrato métrico devuelve NaN o
un número construido con el 3-6% de los pares. **La vía Q no podía dar otra respuesta que "mundo pequeño":
está calibrada al tamaño del mundo pequeño.** Nótese que el diseño de CS069 justificó L=8 diciendo
"margen sobre el diámetro típico mundo-pequeño (~4-6)" — el parámetro se eligió midiendo el objeto que se
iba a juzgar.

### GUARDA 2 — ¿el número podía salir distinto?

**J-B (pendiente > 0.3):** en una retícula d-dim, diam ~ N^(1/d) ⇒ pendiente = 1/d.

| d | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| pendiente esperada | 1.000 | 0.500 | 0.333 | 0.250 | 0.200 | 0.167 |
| vs umbral 0.3 | pasa | pasa | pasa (por 0.033) | **REPRUEBA** | **REPRUEBA** | **REPRUEBA** |

El umbral 0.3 es **inalcanzable por construcción para cualquier orden métrico de dimensión ≥ 4**, y en d=3
pasa por 0.033. En la corrida real la retícula 4D dio 0.309 — pasó por **0.009**, es decir por efecto de
tamaño finito, no por margen. El umbral fue calibrado (dice el propio `cs068_paso2b_diametro.py`) contra
tres referencias, todas de dimensión ≤2: retícula 2D, anillo de cliques, small-world. **Nunca se lo
controló contra una geometría de dimensión alta.**

**J-C (pico_medio > 0.85):** el criterio SÍ es alcanzable — un tensor plantado con k ejes ortogonales
exactos da `n_ejes=k`, `PR=k`, `pico_medio=1.000`, `certificado=True` para k=2,3,5. J-C no es un criterio
imposible; es un criterio que sólo se satisface con **dominios discretos sobre ejes canónicos** (pozos
tipo Potts). Ninguna geometría continua lo satisface: retícula 2D 0.732, 3D 0.660, 4D 0.629, anillo 0.832.

**J-B' (magnitud):** sí podía salir distinto, y de hecho sale distinto — pero las referencias que CS068 usó
("métrico 2D predice ~96 a N=2500") son **específicas de d=2**. Un tejido métrico 3D perfecto da 18 y uno
4D da 12, ambos **indistinguibles del small-world de referencia (~13)** que CS068 usó como piso.

### GUARDA 3 — bug `_diam` / componente gigante

Se midió todo con `C90.diam_gigante` (doble BFS desde la componente **más grande**) y en paralelo con
`H._diam_robusto`. **Difieren en 5 de 9 sustratos**, siempre con `_diam_robusto` por debajo:
aniso2d 67 vs 72, flujo_capas 40 vs 50, er 7 vs 8, real 11 vs 14, real_barajado 6 vs 8. Es un sesgo
sistemático **hacia abajo**, o sea hacia "mundo pequeño". Ahora bien: recalculadas TODAS las pendientes de
J-B con una y con otra medida, **ningún veredicto cambia** (anillo 1.000/1.000, ret2D 0.500/0.500, ret3D
0.368/0.368, ret4D 0.309/0.309, aniso 0.333/0.431, capas 0.508/0.500, er 0.166/0.101, real 0.236/0.228,
barajado 0.141/0.132). La conclusión de J-B es robusta a este bug; la magnitud absoluta J-B' no lo es tanto
(el real pasa de 11 a 14).

### GUARDA 4 — el barajado destruye algo medible y no es isomorfo

| | N=900 | N=2500 |
|---|---|---|
| aristas | 2975 → 2975 | 8094 → 8094 |
| **solapamiento de aristas con el real** | **71/2975 = 2.4 %** | **58/8094 = 0.7 %** |
| secuencia de grados idéntica | True (config-model, debe serlo) | True |
| clustering | 0.5036 → 0.0104 (**−97.9 %**) | 0.5019 → 0.0029 (**−99.4 %**) |
| triángulos | 2862 → 104 (**−96.4 %**) | 7659 → 87 (**−98.9 %**) |

El barajado conserva los grados y destruye ~99% del clustering y de los triángulos, con <2.5% de aristas en
común. No es isomorfo al real ni por asomo. Guarda cumplida.

---

## 5. LECTURA — cuál juez pasa, cuál no, y qué queda de la refutación

### 5.1 Juez por juez

| juez | ¿detecta orden global donde lo hay? | diagnóstico |
|---|---|---|
| **J-B, pendiente log-log del diámetro, sobre la métrica desnuda (vía M)** | **SÍ — 6 de 6 sustratos ordenados; 0 de 3 desordenados** | **SANO.** Único juez que pasa el control limpio, sin un solo falso negativo ni un solo falso positivo |
| J-B', magnitud absoluta del diámetro | **4 de 6** — falla en retícula 3D y 4D | **ACOTADO a d ≤ 2.** Sus referencias son 2D; en d≥3 un cristal perfecto es indistinguible de un mundo pequeño |
| J-A, π-CV | **2 de 6** (2D y, marginalmente, 4D) — falla en el anillo 1D, en 3D y en el grafo de flujo | **DEGENERADO.** Identidad algebraica π~r^(d−2): mide bidimensionalidad, no orden. Califica al blob real (0.641) como más ordenado que a un anillo perfecto (0.716) |
| J-C, gap espectral + candado picado>0.85 | **1 de 6** (sólo el grafo en capas, vía M) | **CIEGO al orden continuo por construcción.** Sólo certifica dominios discretos sobre ejes canónicos (verificado: el plantado da pico=1.000). Ninguna retícula lo pasa |
| **cualquier juez por la vía Q (integral de camino, L=8)** | **prácticamente ninguno** | **CIEGO POR HORIZONTE.** L=8 sólo abarca mundos pequeños; en los sustratos métricos devuelve NaN o mide con el 3-6% de los pares |

### 5.2 Y lo que dice el juez que SÍ pasa, sobre el sustrato real

Esto es lo que decide, y es limpio: con **J-B vía M** —el único observable que detecta orden global en los
seis sustratos ordenados y en ninguno de los tres desordenados— el sustrato real cae **con los
desordenados**:

```
ordenados:    anillo1d 1.000 | ret2D 0.500 | flujo_capas 0.508 | ret3D 0.368 | aniso2d 0.333 | ret4D 0.309
                                        --------- umbral 0.3 ---------
desordenados: real 0.236 | er 0.166 | real_barajado 0.141
```

No hay solape. El real (0.236) está por debajo del umbral, más cerca del Erdős-Rényi que del sustrato
ordenado más débil, y a 0.073 del peor de los ordenados. El instrumento que funciona dice lo mismo que
decía el arco.

### 5.3 Veredicto sobre la refutación de C-N2.6.4 (mitad global)

**ACOTADA — no anulada, y no confirmada en bloque.** En detalle:

- **Se sostiene** la parte apoyada en la **pendiente del diámetro medida sobre la métrica del grafo**: ese
  observable pasó el control positivo sin fallos y coloca al sustrato real junto a sus NULL. Es decir:
  **en esta familia de sustratos no hay orden global de tipo métrico-direccional de dimensión d ≤ 3**, y
  eso ahora es un resultado sobre el mundo, no sobre el aparato.
- **Queda ANULADA la contribución de CS069 (el frente cuántico)**: sus tres jueces corrieron sobre D_q con
  horizonte L=8, que no puede ver más allá de 8 pasos. "(B) se extiende al régimen cuántico" se apoyó en un
  Juez B que devuelve NaN sobre cualquier retícula, un Juez A que sólo reconoce d=2 y un Juez C que sólo
  reconoce pozos de Potts. Ese tramo hay que volver a correrlo con L ≫ diámetro o declararlo sin medir.
- **Queda ANULADO el argumento de MAGNITUD de CS068 Paso 2b** — que el propio ruling declaró *decisivo* por
  encima de la pendiente ("la magnitud decide antes"). Sus referencias son de d=2; una retícula cúbica
  perfecta da 18 a N=2197 y una 4D da 12 a N=2401, ambos **por debajo** del small-world de referencia (~13)
  o a su nivel. Con ese juez, un cristal es un ovillo. Lo que salva a CS068 es justamente la pendiente que
  el ruling declaró "frágil e irrelevante": ahí el residual daba 0.000-0.355, y la parte baja de ese rango
  es el resultado que sobrevive.
- **Queda RE-ETIQUETADA la contribución de CS067**: J-C no es un juez de "organización a gran escala" en
  general — es un juez de **dominios discretos tipo Potts**, y como tal funciona (el plantado certifica).
  El negativo de CS067 vale para "no emergen dominios/ejes discretos", no para "no hay orden global".
- **Fase III** ya había reportado por su cuenta que bajo coarse-graining el umbral 0.3 no discrimina
  (piso ER = 0.406) y había pasado a comparar real-vs-NULL directamente. Esa corrección es exactamente la
  lógica del juez sano; su hallazgo (real no se separa de sus NULL) es consistente con lo de acá.
- **No se controló, y por lo tanto sigue sin medir:** cualquier orden global que NO se manifieste como
  crecimiento del diámetro con N. Un sustrato puede tener orientación global, jerarquía o flujo neto
  macroscópico y seguir siendo compacto en pasos. Ninguno de los cuatro jueces del arco mide eso: J-B/J-B'
  sólo ven "hay lejos", J-A sólo ve d=2, J-C sólo ve pozos discretos. **Ese tipo de orden global es el que
  el instrumento no puede ver, y es donde C-N2.6.4 sigue literalmente sin medir.**

### 5.4 En simple

Le acercamos el fósforo al detector de humo. Resultado: de los cuatro detectores que tenía la casa, **uno
suena bien** (la pendiente del diámetro medida sobre el grafo) — y ése, cuando lo apuntamos al sustrato
real, sigue en silencio, así que el silencio ahora significa algo. Los otros tres estaban rotos de distinta
manera: uno sólo suena si la habitación es exactamente cuadrada; otro compara contra una regla hecha para
habitaciones cuadradas y llama "ovillo" a un cubo perfecto; el tercero sólo suena si el humo viene en
cajitas separadas. Y el detector cuántico entero tiene un alcance de 8 metros en una casa de 50 — nunca
podía oler nada que no estuviera al lado.

**No se declara cierre.** La lectura final es de Alexis.

---

## 6. Archivos

| archivo | qué es |
|---|---|
| `cs092_control_positivo_orden_global.py` | batería: 9 sustratos × 3 N × 2 semillas × 2 vías × 3 jueces |
| `cs092_guardas.py` | las guardas G1, G1b, G2, G3, G4 |
| `cs092_tabla.py` | agregación y veredicto por criterio pre-registrado |
| `cs092_control_positivo_tabla.csv` | **la tabla observable × sustrato** |
| `cs092_control_positivo_crudo.json` | las 54 corridas, todos los campos |
| `cs092_guarda1_perfil_pi.csv` | perfil π(r) por dimensión (la identidad algebraica) |
| `cs092_guarda1b_horizonte.json` | fracción de pares alcanzados con L=8 |
| `cs092_control_positivo.png` | matriz juez × sustrato (verde detecta / rojo no / gris no medible) |
| `cs092_guardas.png` | π(r)~r^(d−2) y el horizonte L=8 |
| `cs092_control_positivo.log` | log completo de la batería |
