# AUDITORÍA DE CÓDIGO — Organelos + Genoma de CosmoRobot
## Búsqueda de errores conceptuales y "Shannon encubierto" (CS, para la sesión del 12-jul-2026)

*CS, autoridad de diseño, leyó línea a línea: los 9 organelos, `organo_deliberacion.py` completo, `main.py`,
los dos `config/`, y la estructura del `genoma/VST_Genoma.py` (999 líneas). "Shannon encubierto" = comportamiento
decidido de antemano disfrazado de emergencia, o asignación a mano de "qué es qué". Clasifico cada hallazgo por
gravedad. Nada aquí toca la mente para "que resulte" algo — son correcciones de fidelidad.*

---

## VEREDICTO GLOBAL

**El código está sorprendentemente limpio de Shannon.** El sospechoso número uno del sesgo-derecho —que el giro
de escape tuviera una dirección fija programada— **quedó descartado**: la capa reactiva gira a
`random.choice([True, False])` (`main.py:171`). El sesgo derecho es, entonces, genuinamente emergente (inercia +
azar + posible asimetría física), como dice la bitácora. Bien.

Pero hay **un error conceptual grave (G1)**, **dos medios (M1, M2)** y **varios menores/de honestidad**. El grave
no es Shannon — es lo contrario: un mecanismo tan honesto que **desactiva de hecho el aprendizaje**, y por eso los
datos no mostraron que el robot aprenda. Los detallo en orden de urgencia para mañana.

---

## G1 — GRAVE (conceptual): el aprendizaje está, pero es ~10⁴ veces más débil que el veto → nunca se expresa

**Dónde:** `organo_deliberacion.py`, `ValenciaLocal.actualizar` (L68-87) + constante `TASA_APRENDIZAJE = 0.001` (L47).

**Qué pasa, con números:**
- Un ciclo *bueno* (error < 5cm) sube la valencia en: `0.001 · dt · 10 ≈ 0.001·0.6·10 = 0.006` por ciclo.
- Un ciclo *malo* la baja en `0.001·dt·|error|·0.2`; con error 40cm ≈ `0.0048` por ciclo.
- Un **trauma** (veto) la baja en `0.001·dt·150 ≈ 0.09` por ciclo, **y además** el veto episódico resta **−100
  de golpe** en `decidir()` (L183) mientras el trauma esté en la ventana de 15s.

El resultado: la valencia aprendida se mueve en **milésimas por ciclo**, en un rango [−100, +100]. Para que una
opción acumule una valencia de, digamos, 1.0 (suficiente para inclinar el argmax de forma estable), necesita
**~170 ciclos buenos seguidos en esa misma opción**. Con 7 opciones en el pool y exploración alta, eso
prácticamente no ocurre en sesiones de 100-280 ciclos. **Por eso mi análisis de datos no encontró firma de
aprendizaje: no es que el mecanismo no exista — es que su ganancia es tan baja que queda enterrada bajo el ruido
del veto y la exploración.** La mente "puede" aprender en el papel; en la escala temporal real del robot, no llega
a expresarse.

**Por qué NO es un bug trivial de tuning:** es un desajuste de *escalas de tiempo* entre tres fuerzas que compiten
por la misma variable (valencia): aprendizaje lento (0.006/ciclo), castigo de trauma medio (0.09/ciclo) y veto
episódico agudo (−100 instantáneo). Mientras el veto domine por 4 órdenes de magnitud, el robot es efectivamente
**puro reflejo + exploración**, sin memoria de largo plazo funcional. Eso contradice la tesis de que hay
aprendizaje valorativo.

**Recomendación (no para "que aprenda X", sino para que la capa de aprendizaje *exista* en la escala real):**
subir `TASA_APRENDIZAJE` en 1-2 órdenes de magnitud (p.ej. 0.01-0.05) **y registrar la valencia por opción en el
datalog** (hoy no se guarda — es el punto que ya señalé en la auditoría de datos). Con ambos cambios, el test de
aprendizaje se vuelve medible y honesto. Pre-registrar la predicción antes de correr (regla anti-Shannon: escribir
qué contaría como aprendizaje *antes* de ver el resultado).

---

## M1 — MEDIO (conceptual): `D_actual` satura en 1.0 casi siempre → la "urgencia" no discrimina

**Dónde:** `config/pool_acciones.py: CAMBIO_TOTAL_ESCALA_CONFLICTO = 2.0` + `main.py:196-198`.

**Qué pasa:** `D_actual = min(1.0, cambio_total / 2.0)`. En los datos reales, `cambio_total` supera 2.0 apenas el
robot se mueve un poco (lo confirmé: r=+0.60 pero **saturado en 1.0** la mayor parte del tiempo). Además,
`main.py:196` fuerza `D_actual = 1.0` cuando el mundo lleva 5 ciclos quieto. **Consecuencia:** el "conflicto" que
debería modular finamente la exploración está casi siempre pegado al techo → la mente vive en modo
"exploración máxima" permanente, y la distinción tranquilo/urgente —que es conceptualmente central— casi no opera.

**No es Shannon** (no prescribe la acción), pero **anula un grado de libertad de la teoría**: la urgencia deja de
ser una señal graduada y se vuelve un interruptor casi siempre en ON. La bitácora/diseño dicen "más CambioTotal →
más exploración", pero si siempre está al máximo, esa modulación no existe en la práctica.

**Recomendación:** subir `CAMBIO_TOTAL_ESCALA_CONFLICTO` para que la saturación sea la excepción, no la regla —
calibrar con el histograma real de `cambio_total` del datalog (elegir la escala ~ percentil 90, no un valor a
priori). Es exactamente el tipo de "corregir con datos" que pidió Alexis.

---

## M2 — MEDIO (conceptual): la memoria episódica usa `setpoint` como si fuera tiempo (`ventana_t`)

**Dónde:** `organo_deliberacion.py`, `MemoriaEpisodica.recuperar` (L117-122), llamada desde `decidir` (L181).

**Qué pasa:** `recuperar(opcion, ventana_t=15.0)` filtra eventos con `abs(ev["setpoint"] - setpoint_buscado) <
ventana_t`. El parámetro se llama `ventana_t` (ventana *temporal*, 15 **segundos**) pero se compara contra la
distancia en el **espacio del Volante** (`setpoint`, −100..100). Es decir: un trauma en volante=+100 (rotación
derecha) contamina como "trauma" **cualquier opción a menos de 15 unidades de volante** (+100, +90... y como el
pool más cercano es +60, en la práctica solo la propia +100 — pero el umbral de 15 es un número temporal aplicado a
un eje espacial). **Nunca se filtra por tiempo real**: un trauma de hace 200 ciclos pesa igual que uno de hace 2,
mientras siga en la lista de 50 eventos.

**Por qué importa conceptualmente:** el "veto episódico" pretende ser *memoria reciente* ("no vuelvas a lo que
*acaba* de lastimarte"). Tal como está, es memoria *de las últimas 50 marcas, sin decaimiento temporal* — un
trauma viejo puede seguir vetando una opción mucho después de que el contexto cambió. Eso es menos "reflejo vivo"
y más "lista negra persistente". Contribuye a mi hallazgo de datos (evitación ambigua): el veto no se comporta como
memoria fresca.

**Recomendación:** separar los dos ejes — filtrar por **tiempo real** (`t_actual - ev["t"] < 15s`) para la
frescura, y por **cercanía en volante** (un umbral en unidades de setpoint, p.ej. <30) para la generalización
espacial. Son dos condiciones distintas hoy colapsadas en una.

---

## MENORES / HONESTIDAD (LF) — no bloquean, pero conviene dejarlos dichos

**m1 — Umbrales a priori sin calibración propia (declarados con honestidad, pero acumulándose).** `escalas_sensores.py`
marca EOPD, color, gyro, touch como "SIN CALIBRAR"; `UMBRAL_CRITICO_CM`, `UMBRAL_EOPD_CERCANO`,
`BATERIA_MIN/MAX_MV`, las escalas de CambioTotal — todos son parámetros del diseñador. El código lo dice
explícitamente (bien, es la disciplina LF). Pero **son muchos**, y CambioTotal —la variable que alimenta toda la
urgencia— depende de 6 escalas sin calibrar. Riesgo: un CambioTotal que "mide cuánto cambió el sensor peor
calibrado". Recomendación: una fase de calibración corta (robot quieto N ciclos → ruido base de cada sensor →
escala = k·σ). Convierte 6 números a priori en 6 medidos.

**m2 — `prop_acople` se calcula pero sale 0 (ya visto en datos).** En `propiocepcion.observar`, `acople=(A+H)/2`,
y tanto `A_sys_env` como `H_homeostasis` vienen 0 (ningún organelo los alimenta en el robot). Es honesto (default
0), pero produce una variable muerta en el log. Recomendación: o conectarla, o marcarla explícitamente como
"reservada, no medida" en el registrador para que nadie la lea como dato.

**m3 — Bono de inercia (`current_bonus = 0.8`) vs. exploración: el número que gobierna el sesgo-derecho.** En
`decidir` (L179), repetir la última opción da +0.8 fijo, mientras el `explor_bonus` máximo es `D·0.1·... ≈ 0.1`.
O sea: **la inercia pesa ~8× más que la exploración**, y como la valencia aprendida es ~0.006 (ver G1), en la
práctica el bono de inercia *domina toda la decisión* salvo empates rotos por barajado. Esto explica
mecánicamente el sesgo-derecho: una vez que el azar inicial elige un lado, +0.8 lo re-elige una y otra vez. **No es
Shannon** (no está escrito "prefiere derecha"), pero el valor 0.8 es tan alto que **convierte la mente en casi-pura
inercia**. Conceptualmente, la exploración y el aprendizaje están ahogados por este único número. Recomendación:
bajar `current_bonus` a la escala de los otros términos (p.ej. 0.05-0.1) o hacerlo decaer con los ciclos sin
progreso, para que inercia, exploración y valencia compitan en pie de igualdad.

**m4 — `explor_w = min(0.4, D·1.5)` tope en 0.4.** El peso de exploración nunca pasa de 0.4, así que la valencia
(val_w = 1−explor_w) siempre pesa ≥0.6 — pero como la valencia es ~0 (G1), en la práctica ambos términos son
pequeños frente al bono de inercia (m3). Es coherente entre sí, pero refuerza que **el diseño confía en un
aprendizaje que su propia tasa no permite acumular**. Se resuelve con G1.

---

## EL GENOMA — hallazgo aparte (no es error, es un vestigio a declarar)

**El `genoma/VST_Genoma.py` (999 líneas: Milieu, Organismo, Kleiber, KAPPA, salud del cierre, marcapasos,
altruismo Boorman-Levitt) NO se transcribe en el robot vivo.** Lo verifiqué: `main.py` importa los organelos
sueltos, pero **nunca** instancia `Organismo`, ni llama `vivir_un_paso`, `expresar`, `MedidorComplejidad` ni
`salud()`. La única referencia al genoma en todo el código ejecutable es **un comentario** en
`organo_propiocepcion.py:17` ("ver VST_Genoma.salud()").

**Qué significa:** hoy CosmoRobot corre como un **bucle de organelos coordinados por `main.py`**, no como el
*organismo* que el genoma define (con complejidad M, tempo de Kleiber que estira los tiempos propios, eficiencia
alométrica, y la "salud del cierre" Λ_Cos sobre los invariantes κ). Toda la maquinaria cosmosemiótica más profunda
—la que haría que "vivir más lento al crecer en complejidad" o que mediría la viabilidad κ_P/κ_Δ/κ_LF/...— está
escrita pero **desconectada**.

**No es un error** (el genoma es código correcto y hermoso, heredado de Célula_Madre), pero **es un Shannon
inverso a vigilar**: el riesgo no es que el robot tenga lógica prescrita de más, sino que **atribuyamos al robot
propiedades del genoma que no está usando**. Cuando el documento de legado dice "la misma mente de ANIMA", es
cierto para la *deliberación* (ValenciaLocal + MemoriaEpisódica, portadas fielmente), pero **NO** para el
organismo-Kleiber completo. Recomendación para mañana: decidir explícitamente una de dos —
  (a) **conectar** el genoma (que `main.py` construya un `Organismo`, exprese los organelos como `Organelo`s
      reales, y corra `vivir_un_paso` para que M/tempo/eficiencia/salud operen de verdad), o
  (b) **declarar** en la bitácora que CosmoRobot v1 usa solo la capa de deliberación de la mente, y que el
      organismo-Kleiber es andamiaje reservado (LocusReservado a nivel de todo el genoma).
Lo que no conviene es dejarlo ambiguo: es la diferencia entre "el robot es un organismo cosmosemiótico" y "el robot
usa el módulo de decisión de un organismo cosmosemiótico".

---

## RESUMEN PARA LA SESIÓN (orden de trabajo sugerido)

| # | Hallazgo | Tipo | Acción |
|---|----------|------|--------|
| **G1** | Aprendizaje ~10⁴× más débil que el veto → no se expresa | conceptual grave | subir TASA_APRENDIZAJE + loguear valencia; pre-registrar |
| **M1** | D_actual saturado en 1.0 casi siempre | conceptual medio | recalibrar CAMBIO_TOTAL_ESCALA_CONFLICTO con histograma real |
| **M2** | Memoria episódica confunde eje-volante con tiempo | conceptual medio | separar filtro temporal (15s real) del espacial (Δvolante) |
| **m3** | Bono de inercia 0.8 ahoga exploración y valencia | menor/tuning | bajar current_bonus a ~0.05-0.1 o hacerlo decaer |
| genoma | El organismo-Kleiber está escrito pero desconectado | arquitectura | decidir: conectarlo, o declararlo reservado |
| m1,m2,m4 | Umbrales a priori; prop_acople muerto; explor tope | honestidad LF | fase de calibración; marcar reservadas |

**Lo que NO hay que tocar (está bien):** la capa reactiva (giro de escape al azar — sin Shannon), el barajado
anti-empate en `decidir` (correcto y bien documentado), el veto de trauma como −100 dominante (fiel a R_op),
la escalada por presión interna en vez de contador mecánico (buena traducción de la teoría), y la honestidad
declarada de cada parámetro a priori.

**El hilo conductor de la auditoría:** el código no peca de Shannon (nada está prescrito para "que resulte"). Peca
de lo contrario — de un aprendizaje tan tímido (G1) y una inercia tan fuerte (m3) que, sin querer, el robot se
comporta *como si* fuera casi-reflejo. La mente cosmosemiótica está bien portada; lo que falta es darle a la capa
de aprendizaje la ganancia para competir con el reflejo, y calibrar las escalas con los datos que ya tenemos. Todo
falsable, todo medible, nada impuesto.

— CS, 11-jul-2026. Para discutir mañana antes de tocar una línea. 🐝
