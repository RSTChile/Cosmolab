# PROTOCOLO E5.6-4 — Sensibilidad de X a la definición de equilibrio de referencia

**Congelado (pre-registro):** 2026-07-24 20:46 (America/Santiago, UTC-4)
**Ejecutor:** CC (agente E5.6-4, batería Enfoque 5, corrida EN PARALELO con 29 agentes más — prefijo propio `E5_6_4_`, no se tocó nada fuera de esta carpeta)
**Base de código leída (NO editada):** `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py`
**Documento madre:** `BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md`, sección "E5.6-4"

Este documento se escribe y congela ANTES de tocar el motor (`E5_6_4_motor.py`). Cualquier
desviación respecto de lo aquí escrito se reporta como desviación explícita al terminar —
no se edita retroactivamente (T3, regla de ejecución #1 y #7).

---

## 1. Pregunta

La exergía X se mide siempre como una **desviación respecto de un "equilibrio" de
referencia**. ¿Cambia el veredicto de persistencia (persiste/no persiste bajo la razón
r=H/D) si se cambia la definición de esa referencia? Se implementan ≥3 definiciones
razonables y se recalcula X con cada una, sobre la MISMA trayectoria física simulada
(mismo φ(t), misma expansión), para que la comparación sea limpia (no confunda "cambiar
la física" con "cambiar la vara con la que se mide").

## 2. Modelo físico (heredado sin editar de cs074_rcruz.py)

Campo escalar φ en un anillo de N=200 sitios, idéntico a la base:
- `campo_inicial(N, eps, rng)`: fondo φ=1 + ε·(5 armónicos, fase aleatoria, normalizado a
  σ=1).
- `paso_difusion`: relajación local hacia el promedio de vecinos, SOLO por aristas vivas.
- `paso_expansion`: cada arista viva se corta con probabilidad de Bernoulli H por paso.
- `medir_D`: D = fracción de contraste borrada en un paso de difusión pura (H=0), MEDIDA.
- `medir_pasos_lavado`: pasos calibrados (mediana×1.15 de margen) para que a H=0 la
  persistencia caiga bajo P_thr=0.05 — MEDIDO, igual que en la base, reusado como
  `pasos_fijo` único para toda la grilla (mismo método que el modo "produccion" de
  cs074_rcruz.py: la difusión es lineal, el tiempo de lavado relativo no depende de ε).
- r = H/D es el eje primario; H = min(r_target·D, 1.0), D medido primero.

Todas estas funciones se **importan** desde `cs074_rcruz.py` (sys.path), no se copian ni
se reescriben — la física es exactamente la de la base.

## 3. LAS TRES (+ implícita) DEFINICIONES DE REFERENCIA — EXACTAS, congeladas aquí

Sea φ₀ el campo inicial, φ_f el campo al final de la corrida (tras `pasos_fijo` pasos de
difusión+expansión). Para cada definición se construye un campo de referencia `ref` (un
valor por sitio, puede ser uniforme o no) y se mide la desviación d = φ − ref.

### (A) REF_GLOBAL — media global fija del campo
`ref_global = vector constante = mean(φ₀)` (el promedio espacial del campo ENTERO al
inicio, difundido a los N sitios). Es el "estado muerto" clásico de la exergía
termodinámica: una única referencia de bulto, fija, igual para t=0 y para t=final.
- d₀ = φ₀ − ref_global
- d_f = φ_f − ref_global (la MISMA referencia fija, no se recalcula con φ_f)

### (B) REF_LOCAL — media móvil espacial (ventana circular)
`ref_local(φ) = suavizado_circular(φ, W)`, media móvil de ventana W con kernel uniforme y
envoltura circular (anillo). **W = 21** (fórmula congelada: W = round(N/10) forzado a
impar más cercano; N=200 → 20 → 21). A diferencia de (A), esta referencia se RECALCULA
sobre el propio estado que se está midiendo (equilibrio "local" de cada instante, no un
número fijo de t=0):
- d₀ = φ₀ − suavizado_circular(φ₀, 21)
- d_f = φ_f − suavizado_circular(φ_f, 21)

### (C) REF_DINÁMICA — media móvil exponencial en el TIEMPO
Arranca en el mismo punto que (A) — `ref_dyn(0) = mean(φ₀)` (vector constante) — y luego
seguido paso a paso DURANTE la simulación, alcanzando (con retraso) al campo real:
`ref_dyn(t) = (1−α)·ref_dyn(t−1) + α·φ(t)`, actualizada en CADA paso de evolución.
**α = 20 / pasos_fijo** (fórmula congelada: memoria efectiva ≈ pasos_fijo/20 pasos — un
valor derivado de la calibración medida, no puesto a mano). Requiere trayectoria completa
(no solo el estado final), a diferencia de (A) y (B):
- d₀ = φ₀ − ref_dyn(0) = φ₀ − mean(φ₀)  (idéntico a (A) en t=0, por construcción)
- d_f = φ_f − ref_dyn(pasos_fijo)

### Fórmula de X — igual forma funcional para las 3, solo cambia `ref`
    c = max(0, corr(d_f, roll(d_f, 1)))     [estructura espacial de la desviación]
    v = Var(d_f) / Var(d₀)                  [fracción de la desviación inicial que sobrevive]
    X_ref = c · v
Si Var(d₀) ≤ 1e-18 (caso ε=0, sin diferencia inicial) → X_ref = 0 por definición (mismo
guardián que `persistencia()` de la base con contraste0≤0).

Nota: esta es la MISMA forma funcional que `persistencia()` de cs074_rcruz.py (coherencia
espacial × varianza retenida), generalizada para aceptar cualquier `ref` en vez de asumir
implícitamente ref=0 (correlación cruda). Con REF_GLOBAL y ε chico, X_global ≈ P de la
base (deben coincidir en el límite N→∞, ref→plano).

## 4. NULL

Se permuta φ_f al final (idéntico en espíritu a `evolucionar(..., null=True)` de la base:
"permutar φ al final"), usando una permutación pseudoaleatoria derivada
determinísticamente de la semilla de la corrida (`seed + 500_000`), aplicada al MISMO
φ_f ya simulado — NO se re-simula (ahorra cómputo; la trayectoria pre-permutación es
idéntica por construcción, no aleatoria de nuevo). Para REF_DINÁMICA, `ref_dyn` proviene
de la trayectoria REAL (no se recalcula sobre el campo permutado) — igual que en la base,
donde el NULL solo baraja el estado final, no la historia. Para REF_LOCAL, el suavizado
SÍ se recalcula sobre el campo permutado (por construcción, (B) siempre se mide sobre el
estado que se le da). Declarado y congelado aquí antes de correr.

Desviación explícita respecto de la base: la base recicla el mismo objeto `rng` para
generar la permutación (siguiente extracción del stream), aquí se usa un generador
derivado de la semilla (`seed+500_000`) para poder computar REAL y NULL a partir de UNA
sola simulación en vez de dos. Funcionalmente equivalente (permutación aleatoria del
campo final), más barato.

## 5. Barrido (sobredimensionado, regla del director)

| Eje | Valores | N puntos |
|---|---|---|
| ε | {0, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.3, 1.0} | 8 |
| r = H/D | {0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000} | 14 (6 décadas, cruza r=1) |
| tipo_referencia | {global, local, dinámica} | 3 |
| semillas | 0..11 | 12 |
| N | 200 (fijo, modo "produccion" de la base) | — |
| pasos | calibrado UNA vez con `medir_pasos_lavado(N=200, eps=1e-3, semillas=12)`, reusado en toda la grilla | — |

Combinaciones físicas (ε,r,semilla) = 8×14×12 = **1344 simulaciones** (cada una da 1 φ_f +
1 ref_dyn(final) + su NULL derivado). De cada simulación se extraen las 3 X (una por
definición de referencia) × {real, null} = 6 números → **8064 valores de X** en total.
Cómputo largo autorizado (regla de ejecución #8); no se recorta el barrido si tarda.

## 6. Observable, juez y veredicto de persistencia (T2: separados)

Por cada (ε, r, tipo_referencia), sobre las 12 semillas:
- X_real_media, X_real_std; X_null_media, X_null_std.
- z = (X_real_media − X_null_media) / max(√((Var_real+Var_null)/2), 1/12)  [mismo diseño
  de z-score que `barrido_rcruz` de la base].
- **Veredicto de persistencia** (juez, congelado): "persiste" si z > 2.0 Y
  X_real_media > X_null_media; si no, "no_persiste". (Z_THR=2.0 declarado aquí, no
  ajustado después de ver resultados.)

**Invariancia (la pregunta de este experimento):** para cada (ε,r), se comparan los 3
veredictos (global/local/dinámica). "Invariante" si los 3 coinciden; si no, se marca
"DIVERGE" y se reporta cuál definición difiere y en qué dirección — sin ocultarlo (regla
del director: "si depende, se reporta cuál difiere y por qué").

## 7. PASS / criterios de lectura (congelados antes de correr)

- ε=0 → X=0 en las 3 referencias, a todo r (control trivial, debe ser invariante).
- r=0, ε>0 → las 3 referencias deberían mostrar "no_persiste" (la difusión lava; control
  de validez, análogo a `control_r0_ok` de la base).
- r≫1 → se espera "persiste" en las 3 si el mecanismo de aislamiento por expansión es
  robusto a la vara de medir.
- **PASS de la pieza:** el veredicto persiste/no-persiste es invariante a la referencia en
  ≥90% de las celdas (ε,r) no triviales (excluyendo ε=0). Si hay divergencia sistemática
  (no aislada), se reporta EXACTAMENTE en qué zona del barrido y con qué definición,
  como hallazgo — no se promedia para ocultarlo.
- No se ajusta ningún coeficiente (W=21, α=20/pasos_fijo, Z_THR=2.0) para forzar
  invariancia. Estos quedaron fijados en este documento antes de ejecutar.

## 8. Verificación cruzada (regla de ejecución #4)

1. NULL propio por celda y por tipo de referencia (arriba).
2. Segundo método: además de X (coherencia×varianza), se reporta `var_ratio` puro
   (Var(d_f)/Var(d₀), sin el factor de correlación c) para las 3 referencias — permite ver
   si una eventual divergencia viene del factor de estructura o ya está en la varianza
   cruda.
3. Auditoría en disco: resultado crudo completo (`E5_6_4_resultado_crudo.json`) con las
   8064 celdas + dispersión entre semillas, para revisión por quien no escribió el motor.

## 9. Salidas

- `E5_6_4_motor.py` — motor (escrito DESPUÉS de este pre-registro).
- `E5_6_4_resultado_crudo.json` — todas las filas (ε, r, tipo_referencia, X_real por
  semilla, X_null por semilla, medias/std, z, veredicto, var_ratio).
- `E5_6_4_invariancia.json` — tabla de invariancia por (ε,r): los 3 veredictos y si
  coinciden o divergen (derivada del crudo, calculada tras correr, sin tocar el motor).

## 10. Trampas explícitamente evitadas

- T0/T1: N=200, pasos_fijo (medido), W=21 y α=20/pasos_fijo (fórmulas fijadas aquí, sus
  valores numéricos se derivan de la calibración medida) — nada puesto a mano para
  acercar un resultado.
- T2: X es una fórmula fija; el veredicto lo da z vs NULL con umbral declarado, no el
  observable crudo.
- T3: este documento se congela ANTES de escribir el motor.
- T4: el NULL se mide en las 3 referencias, debe morder en las 3 por separado.
- T5: se reporta la curva completa X(ε,r) por referencia, no solo un gate binario.
- T6: (no aplica conservación de E aquí — es el Tema 6, no el Tema 2; el motor SÍ reporta
  la deriva de Σφ inicio→fin como auditoría secundaria, heredada de la física de la base).
- T7: la expansión usa sorteos de Bernoulli por arista en CADA paso (ya es perturbación
  dinámica, no solo semilla; heredado de `paso_expansion` de la base).

No se corre nada del motor hasta que este archivo esté guardado en disco.

---

## ADENDA — Definición común de X (ARREGLO 3), 2026-07-25

**No se edita el texto original arriba (T3): esta sección se agrega, no reemplaza.**

Por instrucción del director (re-correr 5 experimentos ya completados con una definición
homologada de exergía para hacer comparables los 30 experimentos de Enfoque 5): este motor
ahora calcula, EN PARALELO a las 3 referencias ya congeladas (global/local/dinámica, §3),
una **CUARTA** forma de medir exergía — la definición canónica del proyecto, importada de
`BATERIA_ENFOQUE5/_observables_homologadas.py::exergia_X` (NO reimplementada a mano):

    Xh(φ) = (1/N) · Σᵢ (φᵢ − 1)²      [mean-square-deviation, referencia FIJA φ_eq=1]

A diferencia de (A)/(B)/(C), Xh se calcula sobre el φ CRUDO (φ_f real y φ_null permutado),
SIN restar ninguna referencia local/dinámica — φ_eq=1 ya está incorporado dentro de la
fórmula misma, como una referencia global implícita (mismo espíritu que REF_GLOBAL, pero
con una forma funcional distinta: cuadrática pura, sin el factor de autocorrelación c que
usan las 3 referencias existentes).

Se calcula por celda (ε, r_target), sobre las mismas 12 semillas, mismo NULL (permutación
de φ_f con `seed+500_000`, ya usado en las 3 referencias), y el MISMO criterio de veredicto
(`z > Z_THR=2.0 AND X_real.mean() > X_null.mean()` → "persiste", si no "no_persiste").
Nada de la grilla, semillas, NULL, ni las 3 referencias existentes cambia — Xh_real/Xh_null
son un cálculo paralelo, adicional, sobre la MISMA trayectoria física ya simulada.

**Comparación clave que pide el director:** ¿el veredicto con la definición canónica
(`veredicto_canonica`) coincide, celda por celda, con el veredicto de REF_GLOBAL
(`veredicto_global`, la única de las 3 referencias que reprodujo la física esperada en la
corrida original)? Se reporta la fracción de coincidencia, y en qué región (ε, r) diverge
si diverge.

**Predicción pre-registrada (declarada ANTES de correr, T3):** la fórmula Xh es una
estadística de "bolsa" (depende solo del multiconjunto de valores {φᵢ}, no de su orden
espacial) — es literalmente invariante bajo cualquier permutación de φ. El NULL de este
experimento (§4) es EXACTAMENTE una permutación de φ_f. Por construcción algebraica,
`Xh(φ_null) = Xh(φ_f)` de forma EXACTA (hasta error de punto flotante) para cada semilla,
sin excepción — no es una predicción estadística sino una identidad matemática (verificada
antes de lanzar la corrida completa: `exergia_X(φ)` y `exergia_X(permutación de φ)`
coinciden a 1e-15 en una prueba directa). Por lo tanto se predice, ANTES de correr el
barrido completo: `z_canonica ≈ 0` en TODAS las celdas y `veredicto_canonica = "no_persiste"`
en el 100% de las celdas no triviales — no porque la física de persistencia canónica no
exista, sino porque el NULL por-permutación-espacial usado en este diseño (heredado de §4,
congelado para las otras 3 referencias) no tiene ningún poder para detectar una estadística
que ya es invariante a esa misma permutación. Si esta predicción se confirma, la
"divergencia" entre `veredicto_canonica` y `veredicto_global` NO debe leerse como que la
física cambió, sino como un desajuste ESTRUCTURAL entre la definición canónica (bolsa,
sin estructura espacial) y el método de NULL de este experimento (permutación espacial) —
se reporta así, explícitamente, no se oculta ni se reinterpreta después de ver el resultado.

**Persistencia de detalle crudo (regla del director, "para no volver a simular"):** además
de los resúmenes agregados, `celda()` ahora también devuelve φ0, φ_f, φ_null. `barrido()`
guarda, por cada celda y cada una de las 12 semillas: `sum(φ)` y `sum(φ²)` de los tres
arrays (suficiente para reconstruir E y Xh canónica sin re-simular), y además los arrays
COMPLETOS (φ0, φ_f, φ_null, redondeados a 6 decimales) de la semilla representativa
`seed=2000` (la primera de la grilla) por celda. Esto permite auditar o recomputar
cualquier definición futura de X sobre el mismo φ sin re-correr la física.

**Archivos previos conservados (no se borran, ver regla del proyecto):**
- `E5_6_4_resultado_crudo_DEFINICION_VIEJA_pre_ARREGLO3.json`
- `E5_6_4_invariancia_DEFINICION_VIEJA_pre_ARREGLO3.json`

Este experimento se re-corre desde cero (mismo barrido, mismas semillas, mismo NULL, las
mismas 3 referencias intactas) agregando únicamente esta cuarta medición en paralelo.

No se corre el motor modificado hasta que esta adenda esté guardada en disco.
