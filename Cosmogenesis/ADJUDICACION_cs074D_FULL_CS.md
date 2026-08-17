# ADJUDICACIÓN cs074-D FULL — CC acertó el diagnóstico; el experimento NO se cierra

**Adjudica:** Claude Science · **Ejecutó:** CC · **Director:** Alexis López Tapia · 29-jul-2026
**Estado:** el experimento **NO se cierra.** Espera decisión explícita del director.
**Verificado en disco**, no de palabra: `resultados_cs074D_barrido_fino/cs074D_result_FULL.json`
(1,28 MB), `cs074D_full_stderr.log`, `RESULTADO_cs074D_barrido_fino_PARA_CS.md`, y lectura
directa de `cs073_cierre_holistico.py` y `cs074_energia_holistica.py`.

---

## 1. Lo primero: CC hizo lo correcto

CC **paró y reportó en vez de arreglar por su cuenta.** Eso era exactamente lo pedido: un
desacuerdo es un dato, no un problema a resolver en silencio. Y su diagnóstico coincide con
el que esta sesión había emitido de forma independiente el 27-jul sobre el smoke de 20
configuraciones (`ADJUDICACION_cs074D_NULL_sin_fuerza_CS.md`), antes de que el barrido
completo terminara. Dos análisis separados, misma conclusión, por el mismo mecanismo.

**Corresponde decirlo con claridad: el diagnóstico de CC es correcto y está confirmado.**

---

## 2. Lo que se verificó del barrido completo

| lectura | valor en disco |
|---|---|
| configuraciones corridas | 2000 de 2000 (completó) |
| tiempo total | 219.738 s = **61,0 horas** |
| corridas totales | **48.000** |
| configuraciones válidas | **1647** (82,4 %) |
| configuraciones sin ningún átomo | **353** (17,6 %) |
| **configuraciones con z > 2** | **0 de 1647 (0,0 %)** |
| z máximo | **+0,895** |
| z mínimo | −1,026 |
| z medio | −0,023 |
| conectividad | `sin_hits_z2` — no calculable |

Las tres lecturas pre-inscritas del protocolo §6 presuponen al menos un acierto z>2. Hubo
cero en 1647. **Ninguna de las tres se puede emitir.**

### 2.1 El diagnóstico, ahora con 1647 configuraciones

En el smoke (16 configuraciones válidas) el factor faltante era ~20. Con el barrido completo:

- dispersión entre semillas, mediana: **0,0566**
- efecto necesario para z=2 (= 2·sd): **0,1132**
- efecto real−NULL observado, mediana: **0,0012**
- **factor faltante: 95×**

El criterio z es un tamaño de efecto: **no crece agregando semillas ni configuraciones.** Las
61 horas confirmaron el diagnóstico con cien veces más estadística, y no lo movieron.

**Matiz sobre una afirmación de CC.** CC dice "algunas filas con real y NULL idénticos hasta
el último decimal". Verificado: a precisión de máquina (<1e-15) hay **0 filas**; idénticas a
1e-9 hay **3**; a 1e-6 hay **478 (29,0 %)**. La afirmación es correcta en espíritu — casi un
tercio del barrido es indistinguible a 1e-6, y eso es ruido de redondeo, no física — pero
"hasta el último decimal" es más fuerte de lo que muestran los datos. Se anota porque el
pacto pide precisión en las dos direcciones.

### 2.2 La causa raíz, confirmada en el código

Verificado leyendo el motor: la masa bariónica es **constante** (un solo valor único) y
`posiciones_escenario` (`p_gravedad_general.py`, l.29-35) sortea posiciones uniformes con
semilla fija, con su propio comentario diciendo que las posiciones *"no cargan información,
son el contenedor neutro"*. Barajar densidades entre posiciones que nunca estuvieron
correlacionadas con ellas no puede destruir una correlación que no existe.

---

## 3. El arreglo que CC propone: verificado, existe

CC afirma que `cs073_cierre_holistico.py` ya tiene un modo de posiciones sembradas por
densidad. **Verificado en disco y es cierto:**

- `cs073_cierre_holistico.py` l.29 importa `malla_causal_atomos, layout_resortes` de
  `cs072_modulos/piezas/p_semilla_causal.py`
- l.76 expone el parámetro `semilla="uniforme"` con alternativa `"causal"`
- l.86-87 documenta: posiciones sembradas por layout-de-resortes sobre la malla causal REAL,
  donde pares causalmente cercanos quedan espacialmente cercanos
- l.114-128 implementa ambas ramas, con error explícito si el modo es desconocido

Y verificado también que **`cs074_energia_holistica.py` NO tiene ese modo**: la búsqueda de
`causal`/`semilla=` en el motor de cs074 solo devuelve comentarios sobre otra cosa (el "gate
causal" de la Regla 4, que es un cobro de energía, no un layout).

**Conclusión: el arreglo es real, está probado en otra rama del proyecto, y CC identificó
correctamente que falta portarlo.** Con el modo causal, barajar densidades sí destruiría algo
— porque la posición pasaría a depender de la densidad.

---

## 4. Una afirmación de CC que NO se pudo verificar

CC informa "0 fallas de conservación". **Ese campo no existe en el JSON entregado.** Las
claves por fila son: `idx, cfg, ok, ok_reales, ok_nulls, frac_masa_ligada_real_media/std,
frac_masa_ligada_null_media/std, z, n_clusters_finales_media, frac_masa_en_mayor_cluster_media`.
No hay ningún campo de fuga, deriva o balance energético.

Puede ser cierto —el motor tiene su propio control de deriva interno y quizá CC lo vio en
otra salida— pero **en el archivo entregado no está impreso, así que no se adjudica como
verificado.** Es la misma regla que me apliqué a mí mismo en la vuelta del Camino B.

Las 353 fallidas tampoco traen nota (`nota` ausente en todas), aunque el motivo se
reprodujo corriendo el motor directamente en la sesión del 27-jul: *"sólo 0 átomos reales
(<8): sin masa suficiente"*.

---

## 5. El hallazgo que SÍ queda de las 61 horas

Con 2000 configuraciones el borde de expansión pasó de sospecha (4 casos) a resultado
caracterizado:

| tasa de expansión | configuraciones | % sin ningún átomo |
|---|---|---|
| 0,0010 – 0,0015 | 153 | **100 %** |
| 0,0015 – 0,0020 | 109 | **100 %** |
| 0,0020 – 0,0025 | 83 | **100 %** |
| 0,0025 – 0,0030 | 69 | 11,6 % |
| 0,0030 – 0,2 | 1586 | **0 %** |

**Hay un piso de expansión en ≈0,0026 por debajo del cual el modelo no forma ni un solo
átomo**, y la transición es abrupta pero no un corte perfecto: hay una banda estrecha
(0,0025–0,0030) donde algunas configuraciones sí forman átomos y otras no. Por encima de
0,003, cero fallas en 1586 configuraciones.

Esto es del mismo tipo que el hallazgo del Experimento A (demasiada asimetría destruye
estructura): un régimen donde el proceso no arranca. **No estaba en el diseño, no se buscó, y
es independiente del defecto del NULL** — no depende del brazo de control, solo de si el
motor produce bariones. Por eso sobrevive intacto al problema del experimento.

Es, hasta ahora, el único resultado que las 61 horas dejan en firme.

---

## 6. Veredicto

**El barrido completo NO contesta la pregunta del director**, por defecto de control, no de
ejecución. CC implementó el protocolo correctamente, sin tocar el motor, y paró cuando
detectó el problema.

**Lo que NO se puede escribir:** "la estructura no vive en una banda estrecha". Cero aciertos
con un control 95 veces demasiado débil significa *no se midió*, no *no existe*.

**Lo que SÍ queda establecido:**
1. En este motor, **la fracción de masa ligada es prácticamente insensible a cuál partícula
   lleva qué densidad** (1647 configuraciones, factor 95 por debajo del umbral). La densidad
   #23, en este régimen y con posiciones uniformes, casi no es un canal causal.
2. **Piso de expansión ≈0,0026:** por debajo no hay materia. Caracterizado con 2000 puntos.

---

## 7. Recomendación

**Camino A del análisis previo, ahora con el arreglo identificado por CC.** Portar el modo
`semilla="causal"` de `cs073_cierre_holistico.py` al motor de cs074, **con un calibrador
previo obligatorio**: una configuración donde se sabe que hay estructura, verificando que el
NULL nuevo produce z>2. **Si el NULL no pasa su propio calibrador, no se usa** — y eso se
sabe en ~2 horas, no en 61.

Sobre el alcance del re-barrido, mi recomendación es **no repetir las 2000 configuraciones**:

- las 353 con tasa <0,0026 no forman átomos y son tiempo perdido → arrancar el rango de
  expansión en 0,003 recupera **17,6 % del cómputo**
- con el calibrador aprobado, unas 300–400 configuraciones bastan para ver si aparecen
  aciertos; solo si aparecen tiene sentido el barrido grande para medir conectividad

Eso baja la próxima corrida de 61 horas a aproximadamente 10-12.

**Alternativa, si el director prefiere cerrar la línea:** los dos resultados del §6 quedan
como hallazgos honestos, y el pendiente "¿la estructura vive en una banda estrecha?" se
declara **abierto y no medido** — no negado.

---

*Nada se cierra aquí. Nada se corre sin autorización explícita del director.*
