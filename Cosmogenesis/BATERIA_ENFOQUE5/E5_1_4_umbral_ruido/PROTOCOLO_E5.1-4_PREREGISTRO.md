# PROTOCOLO E5.1-4 — Pre-registro
## "Umbral de exergía frente al ruido dinámico, barrido de 8 décadas"

**Fecha/hora de pre-registro:** 2026-07-24 16:37 (America/Santiago, -04)
**Ejecutor:** CC (agente E5.1-4, Enfoque 5, ejecución paralela 30 experimentos)
**Estado:** escrito ANTES de correr el motor (T3 — juez congelado antes de correr). No se edita tras ver resultados; si algo falla, se reporta tal cual.

---

## 1. Pregunta

¿Cuánto ruido dinámico (forzamiento estocástico aplicado en CADA paso, no solo en la
condición inicial) aguanta la exergía (persistencia de estructura frente al equilibrio
uniforme) antes de disolverse? ¿La curva X_final(amplitud_ruido) decae de forma suave, o
hay un salto abrupto (umbral discreto)?

## 2. Código base (NO editado)

`/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py` — se **importa** por
`importlib` (archivo cargado tal cual, sin tocar una línea) y se reutilizan sus funciones
puras: `campo_inicial`, `paso_difusion`, `paso_expansion`, `persistencia`, `medir_D`,
`medir_pasos_lavado`, `temperatura_fisica`. El motor de este experimento
(`E5_1_4_motor.py`) añade el forzamiento estocástico por paso que el código base NO tiene
(el código base solo perturba la condición inicial vía `eps` en `campo_inicial`; la
difusión y la expansión no reciben ruido dinámico).

## 3. Mecanismo añadido: forzamiento estocástico por paso

En cada paso de la evolución, **después** de `paso_difusion` + `paso_expansion` (orden
Euler-Maruyama: primero la dinámica determinista/topológica, luego la fluctuación), se
suma al campo un vector de ruido blanco gaussiano i.i.d. por sitio:

```
phi_{t+1} = paso_expansion(paso_difusion(phi_t, activo), H) 
phi_{t+1} = phi_{t+1} + amplitud_ruido * xi_t,   xi_t ~ N(0,1) por sitio, i.i.d. en t
```

El ruido actúa sobre TODOS los sitios (no depende de qué aristas están vivas — es una
fluctuación dinámica del valor del campo, no de la topología). `amplitud_ruido` está en
las mismas unidades que `eps` (perturbación relativa al fondo=1), lo que permite comparar
directamente con la escala de `eps` ya usada en el código base.

**No se impone conservación de E1 en este experimento.** E1/E2 son axiomas del Tema 2
(la contabilidad, ver E5.2-3 "Conservación bajo forzamiento estocástico" — ese es el
experimento que audita el balance). Aquí, Tema 1, el objetivo es la persistencia de
estructura, no el balance energético. Como diagnóstico de transparencia (regla 5,
"conservación de E verificada cada paso"), se registra la deriva de `sum(phi)` entre
inicio y fin de cada corrida, SIN pretender que esté acotada — se reporta cruda.

## 4. Barrido (T7 — barrido + perturbación dinámica, nunca un punto)

- **amplitud_ruido** ∈ [1e-8 … 1] — 8 décadas, 17 puntos log-espaciados (2 por década):
  `10**k` para `k = -8, -7.5, -7, ..., -0.5, 0`.
- **r** = H/D (razón expansión/difusión, definición idéntica al código base) ∈
  {0, 0.1, 0.3, 1, 3, 10, 30, 100} — 8 valores, rango extremo (factor 1000 entre el mínimo
  no nulo y el máximo), consistente con `R_TARGETS` del código base.
- **semillas** ≥16 → se usan **16** semillas independientes por combinación
  (amplitud_ruido, r), cada semilla gobierna tanto la condición inicial como la secuencia
  completa de ruido dinámico + cortes de expansión (T7: perturbación dinámica real, no
  solo la semilla de arranque).
- **eps (contraste inicial), FIJO y pre-registrado:** eps_real = 1e-3 (escala media ya
  usada en el código base, modo "producción"). No se barre eps en este experimento — el
  barrido de eps es el objeto de E5.3-1/E5.5-1; aquí el eje sobredimensionado es
  exclusivamente `amplitud_ruido` (8 décadas) × r, tal como pide la ficha E5.1-4.
- **N = 200**, igual que `cs074_rcruz.py` modo "produccion".
- **pasos:** medidos (no puestos a mano) vía `medir_pasos_lavado(N=200, eps=1e-3,
  semillas=8)` del código base — el mismo procedimiento de calibración que usa
  `cs074_rcruz.py`. Se fija un único valor de `pasos` para TODAS las corridas (real y
  NULL), de modo que la única diferencia entre real y NULL sea `eps` (control apareado).
- **H(r):** se mide `D` UNA vez (promedio de `medir_D` sobre varias semillas, con
  eps_real=1e-3) y se deriva `H = min(r*D, 1.0)` para cada r del grid. Ese mismo grid de
  H(r) se usa tanto para las corridas reales como para las NULL (ver §5) — así ambas
  comparten exactamente la misma topología/dinámica determinista y solo difieren en `eps`
  y en la realización de ruido.

Total de corridas: 17 amplitudes × 8 r × 16 semillas × 2 (real + NULL) = **4352 corridas**.

## 5. NULL (T4 — el NULL debe morder)

Tal como especifica la ficha: **NULL = ruido con ε=0.** Es decir: misma H(r), mismo
`pasos`, misma secuencia de ruido dinámico y de cortes de expansión (misma semilla), pero
`campo_inicial` se genera con `eps=0` (fondo perfectamente uniforme, sin perturbación
inicial). Es un control apareado (misma semilla real/NULL) — no un barajado post-hoc.

## 6. Observables (T2 — el observable no es su propio juez)

**Observable primario — X_final, definición heredada literal de la ficha E5.1-1** ("fracción
de E que puede hacer trabajo, desviación del equilibrio uniforme"): se usa la función
`persistencia()` del código base tal cual (correlación con el vecino × varianza
normalizada respecto del contraste INICIAL `contraste0` medido antes de que empiece la
dinámica). Para `eps=0`, `contraste0=0` y la función retorna 0.0 por construcción (rama
degenerada ya presente en el código base) — se reporta así, sin editar la función.

**Observable secundario — X_alt (segundo método independiente, regla 4: "segundo
observable/método"):** dado que X_final es estructuralmente 0 para todo el NULL (por la
rama `contraste0<=0`), NO permite verificar si el ruido dinámico por sí solo genera
estructura correlacionada por encima del piso de ruido blanco. Se define un segundo
observable que SÍ puede "morder" en el NULL: la misma fórmula de persistencia
(correlación×varianza normalizada) pero usando como referencia la escala del propio ruido
inyectado, `contraste0_alt = amplitud_ruido` (o, si amplitud_ruido=0, se usa contraste0
real). Esto mide si la dinámica (difusión+expansión) organiza el ruido puro en estructura
correlacionada de vecino, más allá de ruido blanco no correlacionado (que da X_alt≈0 por
construcción, ya que el ruido blanco i.i.d. tiene correlación esperada ≈0 con el vecino).

Ambos observables se reportan para real y NULL, en toda la grilla.

**Diagnóstico auxiliar (no es observable de PASS/FAIL):** deriva de `sum(phi)` inicio→fin
(transparencia, ver §3).

## 7. Criterio de PASS/FAIL (congelado antes de correr)

- **PASS-forma:** la curva X_final(amplitud_ruido), agregada por semilla en cada r, es
  MONÓTONA NO CRECIENTE dentro de tolerancia de ruido estadístico (una caída con pendiente
  que cambia de signo más de una vez fuera de las barras de dispersión entre semillas
  cuenta como no-monótona) y **suave** — se define "salto abrupto" como: existe un par de
  puntos consecutivos en el eje log-amplitud (separados por 0.5 década) donde
  |ΔX_final| > 3× la desviación estándar entre semillas promedio de la curva en esa zona,
  Y ese salto es mayor que el cambio total acumulado en las 2 décadas adyacentes. Si eso
  ocurre, se reporta explícitamente el punto y NO se aliza (regla del director: "si hay
  salto abrupto, repórtalo, no lo alises").
- **Control de validez (gate, no ajuste):** en NULL (eps=0), X_final debe ser ≈0 en TODA
  la grilla (por construcción, ver §6) — se verifica que en efecto lo sea (auditoría de
  que la implementación no tiene un bug que rompa esa garantía).
- **Lectura de X_alt:** si X_alt(NULL) > 0 de forma sistemática y creciente con
  amplitud_ruido, es evidencia de que el ruido dinámico por sí solo, mediado por la
  dinámica de difusión+expansión, genera correlación vecino-a-vecino (un tipo de
  estructura, aunque no sea la "exergía" del campo original) — se reporta como hallazgo,
  no se fuerza a que sea 0.
- **eps=0 → X_final=0 a todo ruido:** ya garantizado por construcción (§6); se reporta el
  chequeo, no se re-interpreta.
- Ningún coeficiente se mueve para acercar el resultado a ninguna expectativa (regla de
  oro 2 y 6).

## 8. Verificación cruzada (regla 4: tres verificaciones)

1. NULL apareado (§5).
2. Segundo observable X_alt, método independiente (§6).
3. Auditoría en disco: el JSON crudo con las 4352 filas se entrega completo (curva
   entera, sin agregar de más) para que quien no escribió el motor pueda re-verificar.

## 9. Archivos de salida (prefijo E5_1_4_, carpeta propia — no se toca nada fuera de ella)

- `E5_1_4_motor.py` — motor (este documento se escribe ANTES).
- `E5_1_4_resultado.json` — barrido crudo completo (4352 filas) + metadatos de
  calibración (D, pasos, tiempos de lavado).
- `E5_1_4_resultado_stdout.txt` / `E5_1_4_log.txt` — log de ejecución con timestamps.

## 10. Axiomas de diseño declarados (no física real)

**E1** (conservación del presupuesto total) — **NO se impone** en este experimento
(alcance de Tema 2, ver E5.2-3). **E2** (la expansión redistribuye E latente en exergía,
no la crea) — el mecanismo de corte de aristas (`paso_expansion`) es el mismo del código
base, sin modificación; el ruido dinámico es un forzamiento ADICIONAL, no forma parte de
E2.

---

## ADENDA — Arreglo 2 (ruido calibrado) + Arreglo 3 (Xh canónica), 2026-07-25

**No se edita el texto original arriba (T3): esta sección se agrega, no reemplaza.**

Este experimento había quedado MUERTO A MEDIO CAMINO (pausa de la batería): la corrida
original murió a mano (o se dejó morir intencionalmente al ver el síntoma) a 832/4352
combinaciones (19.1%), log conservado en `E5_1_4_stderr.txt` (NO se borra, es evidencia).
Ese log muestra el síntoma clásico del bug de "ruido mal calibrado" (Arreglo 2, ya
corregido hoy en otros 4 experimentos hermanos): a `r=0.1, amplitud=1.000e-04`,
`X_final_real` ya subía a 7.4; extrapolando la progresión geométrica visible en el log
(cada década de amplitud multiplica X_final_real por ~10×, ver líneas
`amplitud=1.000e-03 → X_final_real=27.8`, `amplitud=1.000e-02 → X_final_real=2727`,
..., `amplitud=1.000e+00 → X_final_real=27,221,303` en `r=0.0`), el motor iba camino a
producir números sin sentido físico en el resto de la grilla — una explosión numérica
por acumulación de ruido sin control, no una señal real de "el ruido arrasa con la
estructura".

### Diagnóstico exacto del bug (confirmado leyendo `E5_1_4_motor.py` original)

En `paso_con_ruido()` / `corrida_con_ruido()`, cada uno de los `pasos=6095` pasos FIJOS
(calibrados una sola vez por `medir_pasos_lavado`) sumaba:

```python
phi = phi + amplitud * rng.standard_normal(phi.shape)     # amplitud CONSTANTE, cada paso
```

Esto es un paseo aleatorio sin amortiguar por sitio: su varianza acumulada tras `pasos`
pasos es ≈ `pasos · amplitud²` — CRECE SIN TOPE con el número de pasos, y `pasos` está
fijo en 6095 (grande) para TODA la grilla. Es exactamente el mismo mecanismo del "ruido
mal calibrado" ya diagnosticado y corregido en `E5_1_1`, `E5_1_2`, `E5_2_3`, `E5_5_2`
(ver `_ruido_calibrado.py`), solo que aquí el eje barrido ES la amplitud de ruido misma
(17 puntos, 1e-8…1, 8 décadas) — el bug no es incidental al experimento, es el corazón
de la pregunta que se estaba tratando de responder, así que su presencia invalidaba el
experimento entero, no solo una esquina de la grilla.

### Arreglo 2 aplicado — reinterpretación de `amplitud_ruido` como presupuesto TOTAL

`AMPLITUDES_RUIDO` (17 puntos, 1e-8…1, 8 décadas), `R_GRID` (8 valores) y `SEMILLAS=16`
**NO se tocan** — son el eje central de la pregunta y una regla explícita del proyecto.
Lo que cambia es la interpretación física de cada punto `amplitud` del barrido: deja de
ser "amplitud de ruido por paso" (como estaba, y como generaba la explosión) y pasa a
ser el **presupuesto TOTAL de ruido** a repartir en los `pasos` pasos de la corrida. Se
usa el módulo ya escrito y verificado `BATERIA_ENFOQUE5/_ruido_calibrado.py`:

```python
amplitud_por_paso = ruido_por_paso(NOISE_REL=amplitud, eps=1.0, pasos_fijo=pasos)
                   = amplitud / sqrt(pasos)
```

como la amplitud real aplicada en cada paso (en vez de `amplitud` directamente). Así la
varianza acumulada total al final de la corrida es ≈ `amplitud²`, **independiente de
`pasos`** — que es precisamente lo que el barrido de 8 décadas pretendía medir desde el
principio (un presupuesto total de perturbación estocástica, no un ruido por paso sin
control). `X_alt` sigue usando `amplitud` (el presupuesto total, no `amplitud_por_paso`)
como referencia de escala — es la cantidad físicamente comparable entre puntos del
barrido, y es coherente con la definición original de §6.

Verificado con smoke-test (grid reducido, r∈{0,1}, amplitud∈{1e-8,1e-3,1}, 3 semillas)
ANTES de lanzar la corrida completa: `max|phi_final|` en toda la grilla reducida quedó
en 4.24 (vs. ~2.7×10⁷ que alcanzaba `X_final` — no `phi`, pero del mismo orden de
magnitud de fuera-de-control — en la corrida vieja a un solo punto de la grilla real).

### Arreglo 3 aplicado — Xh canónica en paralelo

Se agrega, sin reemplazar `X_final` ni `X_alt`, el cálculo de `Xh_final =
exergia_X(phi)` (definición homologada de la batería,
`BATERIA_ENFOQUE5/_observables_homologadas.py`, `Xh = (1/N)·Σ(φᵢ-1)²`) sobre el mismo
φ final de cada corrida (real y NULL). Se reporta como tercer observable, en paralelo,
para las 4352 combinaciones.

**Diferencia importante respecto de los otros experimentos ya corregidos hoy:** el NULL
de E5.1-4 es `eps=0` (una corrida física DISTINTA, sin perturbación inicial — el campo
arranca perfectamente uniforme y solo el ruido dinámico lo mueve), NO una permutación
del campo final. En los experimentos donde el NULL sí era `rng.permutation(phi_final)`,
ya se confirmó que `exergia_X` es matemáticamente ciega (invariante bajo permutación,
por construcción: `Σ(φᵢ-1)²` no depende del orden espacial) y por lo tanto da
EXACTAMENTE el mismo valor en real y NULL. Aquí el NULL es una condición inicial
físicamente distinta, así que es *plausible* — se verifica, no se asume — que `Xh` sí
pueda discriminar real de NULL en este experimento en particular. El resultado (si
discrimina o no) se reporta tal cual salga, sin forzarlo en ninguna dirección.

### Detalle de auditoría añadido (paso 3 del encargo)

Para las 4352 combinaciones individuales (no solo el agregado por fila r×amplitud), se
guarda `sum(phi)` y `sum(phi²)` del φ FINAL de cada corrida en el JSON principal
(`E5_1_4_resultado.json`, listas `suma_phi_final_*_por_semilla` /
`suma_phi2_final_*_por_semilla` por fila) — suficiente para reconstruir `Xh_final` sin
recomputar, porque `exergia_X(φ) = Σφ²/N − 2·Σφ/N + 1`. Además, el array φ FINAL crudo
completo (N=200 valores, redondeado a 6 decimales) de las 4352 corridas se guarda en un
archivo aparte comprimido, `E5_1_4_phi_final_crudo.npz` (arrays `phi_final_real` y
`phi_final_null`, forma `(136, 16, 200)` = 136 filas r×amplitud × 16 semillas × N),
para no inflar el JSON principal con ~870k floats. Los índices de fila (`fila_idx_npz`
en cada fila del JSON) mapean 1:1 al primer eje de esos arrays.

### Qué NO cambia

`R_GRID`, `AMPLITUDES_RUIDO`, `SEMILLAS`, `EPS_REAL`, `N`, el método de calibración de
`pasos` (`medir_pasos_lavado`), el mecanismo del NULL (§5, eps=0 apareado por semilla),
los criterios de PASS/FAIL de §7, y `cs074_rcruz.py` (no se edita, regla del proyecto).

Este experimento se corre por primera vez de verdad con esta corrección — la corrida
anterior nunca llegó a producir un JSON de resultados (murió a 19.1%, sin escribir
`E5_1_4_resultado.json`), así que no hay un resultado "buggy" previo que conservar aparte
de `E5_1_4_stderr.txt` (que se deja intacto como evidencia del bug).
