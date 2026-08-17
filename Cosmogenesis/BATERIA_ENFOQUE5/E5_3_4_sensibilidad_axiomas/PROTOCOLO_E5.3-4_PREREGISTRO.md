# PROTOCOLO E5.3-4 — Sensibilidad de la eficiencia a los dos axiomas (E1 on/off, E2 on/off)

**Pre-registrado antes de escribir el motor.** Timestamp de creación: 2026-07-24 20:45 UTC
/ 2026-07-24 16:45 -04 (America/Santiago, `date` del sistema al momento de escribir este
archivo).

Agente: E5.3-4 (Enfoque 5, Tema 3). Corre EN PARALELO con otros ~29 agentes, prefijo propio
`E5_3_4_`, carpeta propia `BATERIA_ENFOQUE5/E5_3_4_sensibilidad_axiomas/`. No se toca nada
fuera de esta carpeta. No se edita `cs074_rcruz.py` (se importa como librería de solo
lectura, igual que hicieron E5.2-2 y E5.3-3).

---

## 0. Estado de E5.3-1 al momento de escribir esto (y de sus hermanos de Tema 3)

Verificado en disco justo antes de escribir este protocolo:

```
find BATERIA_ENFOQUE5 -maxdepth 1
```

`E5_3_1_eficiencia_12decadas/` **no existe todavía** (no aparece en el listado de
`BATERIA_ENFOQUE5/`). Sí existen, con contenido:

- `E5_3_2_eficiencia_vs_ligadura/` (vacía, sin archivos)
- `E5_3_3_estabilidad_temporal/PROTOCOLO_E5.3-3_PREREGISTRO.md` (con contenido)
- `E5_3_5_falsacion_externa/` (vacía)
- `E5_2_2_anticorrelacion_X_S/E5_2_2_motor.py` + protocolo (con contenido)

**Decisión (siguiendo instrucción explícita del orquestador: revisar disco y reusar lo que
haya):** como `E5.3-1` no existe, se adopta la definición de eficiencia que **ya construyó
E5.3-3** (mismo Tema 3, hermano más cercano, definición ya congelada y escrita en su
protocolo) en vez de inventar una nueva desde cero. Esto maximiza la consistencia dentro de
Tema 3 tal como pide el orquestador. Cito textual de
`E5_3_3_estabilidad_temporal/PROTOCOLO_E5.3-3_PREREGISTRO.md`:

```
E_total ≡ contraste0**2 = var(phi_0)
eficiencia(t) := persistencia(phi_t, contraste0)
              = max(0, corr(phi_t, roll(phi_t,1))) * [var(phi_t) / contraste0**2]
E_ligada(t) ≡ corr_local(t) * var(phi_t)
eficiencia(t) = E_ligada(t) / E_total
```

Esto reutiliza literalmente `persistencia()` de `cs074_rcruz.py` (no se reinventa la
fórmula). E5.3-3 declaró explícitamente que en SU experimento el axioma E2 "no aplica"
(N/A) porque no toca enfriamiento — precisamente el hueco que **este** experimento (E5.3-4)
llena: aquí SÍ se implementan y se barren E1 y E2 como interruptores explícitos sobre esa
misma fórmula base, con la definición exacta de "apagar" cada uno dada en la sección 1.

Si al momento de reportar a CS ya existe una definición registrada por E5.3-1 que difiera de
la de E5.3-3, CS puede cotejar las tres; esta corrida no se detiene ni se reajusta a
posteriori (T3).

---

## 1. Definición operacional: qué significa exactamente "apagar" E1 y E2 aquí

### 1.1 Línea base (dinámica sin modificar, igual a `cs074_rcruz.py`)

`campo_inicial`, `paso_difusion`, `paso_expansion` se importan tal cual (sin editar). Por
construcción de `paso_difusion` (promedio con vecinos vivos, una contracción) y
`paso_expansion` (solo corta aristas, nunca toca valores de φ), la dinámica *sin ningún
axioma explícito activo* JAMÁS puede hacer que `var(phi_t) > var(phi_0)` — la "conservación"
en el sentido de "nada emerge de la nada" ya está incorporada en la física base, tal como
notó E5.3-3. Esta es la dinámica que corre cuando **E2 está OFF**, sin importar E1 (ver 1.3).

### 1.2 Axioma E2 (la expansión redistribuye E latente en exergía) — el mecanismo explícito

Cuando **E2 está ON**, después de cada `paso_expansion` se aplica un paso adicional de
**inyección**, que es la única fuente posible de estructura "desde fuera" de la dinámica
base:

1. `delta_frac(t)` = fracción de aristas que estaban vivas y quedaron cortadas **en este
   paso concreto** (medida directamente de `activo_antes & ~activo_despues`, no la tasa
   nominal `H` — así la conversión sigue la expansión REALIZADA, estocástica, no un número
   puesto a mano; T1/T7).
2. `E_lat(t)` = reservorio latente, inicializado en `E_lat0 := mean(phi_0)**2` (medido del
   propio campo — el nivel de fondo uniforme `≈1` de `campo_inicial`, igual para todo ε
   porque `fondo = np.ones(N)` siempre; no es un número puesto a mano, es lo que el campo
   mismo reporta como su nivel de fondo).
3. `delta_iny(t) = delta_frac(t) * E_lat(t)` — cantidad convertida este paso (tasa atada a
   la fracción de expansión medida, sin coeficiente libre).
4. Se inyecta ese excedente como AMPLITUD de la desviación existente:
   `phi_nuevo = media + (phi - media) * sqrt(1 + delta_iny / var_actual)` si `var_actual >
   1e-15`; si el campo ya está plano (`var_actual ≈ 0`, típicamente tras mucha difusión, o
   en el caso degenerado ε=0), la inyección se deposita en los nodos recién aislados
   (`n_nb=0`, es decir protegidos de más difusión por la propia expansión) con signo
   aleatorio (`rng`) y amplitud `sqrt(delta_iny / n_aislados)`. Si no hay ningún nodo
   aislado ese paso, la inyección de ese paso se pierde (`delta_iny := 0`, se reporta).
5. `E_lat(t+1) = E_lat(t) - delta_iny(t)` **si E1 está ON** (la fuente se paga); `E_lat(t+1)
   = E_lat(t)` **si E1 está OFF** (la fuente NO se paga: el mismo `delta_iny` se crea todos
   los pasos sin agotar nunca el reservorio — energía gratis, la violación anti-Shannon que
   E1 existe para impedir).

Cuando **E2 está OFF**: este paso de inyección nunca se ejecuta (`delta_iny ≡ 0` todo el
tiempo); la dinámica es exactamente 1.1.

### 1.3 Axioma E1 (conservación de E impuesta como invariante VERIFICADO)

En cada paso, **siempre** (con E2 on o off) se mide `var(phi_t)` y se compara contra el
presupuesto `contraste0_sq = var(phi_0)`:

- `exceso(t) := max(0, var(phi_t) - contraste0_sq)` — se calcula y se registra SIEMPRE,
  con E1 on o off (T6: "la conservación de E se verifica cada paso", en ambos regímenes).
- **E1 ON:** si `var(phi_t) > contraste0_sq`, se **reescala** la desviación (`phi - media`)
  para que `var(phi_t) := contraste0_sq` exactamente (el guardián actúa: el presupuesto
  nunca se excede, se impone y se verifica). Con E2 off esto nunca se dispara (1.1
  garantiza `var ≤ contraste0_sq` por construcción); con E2 on, SÍ se dispara y es lo que
  limita el efecto de la inyección.
- **E1 OFF:** nunca se reescala. `exceso(t)` puede crecer sin límite mientras dure la
  inyección (con E2 on); se reporta como serie completa, no se oculta ni se corrige (T6:
  "toda etapa puede fallar" — aquí, si E1 está apagado, la etapa SÍ falla, y eso es
  exactamente el dato que se está midiendo).

### 1.4 Las 4 variantes, resumidas

| Variante | E2 (inyección) | E1 (tope/pago) | Comportamiento esperado |
|---|---|---|---|
| E1 on, E2 on (default) | sí | sí, con tope | inyecta pero nunca excede `var(phi_0)`; `E_lat` se agota |
| E1 off, E2 on | sí | no | inyecta SIN tope y SIN agotar `E_lat` → estructura gratis, posible `eficiencia > 1` |
| E1 on, E2 off | no | sí (tope nunca se activa) | idéntico bit-a-bit a la física base de `cs074_rcruz.py` |
| E1 off, E2 off | no | no (nada que verificar falle) | idéntico bit-a-bit a la física base de `cs074_rcruz.py` |

**Predicción pre-registrada (falsable):** con E2 off, alternar E1 no debería cambiar
`eficiencia` en absoluto (idéntico hasta precisión de punto flotante), porque sin el canal
de inyección no hay nada que el tope de E1 pueda llegar a recortar. Es decir, se predice que
**E1 solo es "load-bearing" en presencia de E2** — un efecto de interacción, no un efecto
principal aislado. Si esto NO se cumple (alguna diferencia >1e-9 entre E1on/E1off con E2
off), es un hallazgo inesperado y se reporta como tal, sin forzar la narrativa.

### 1.5 Caso degenerado ε=0 (a diferencia de E5.3-3, aquí SÍ se simula)

E5.3-3 calculó ε=0 analíticamente (eficiencia≡0 para ahorrar cómputo) porque en su marco
nada podía crear estructura desde cero. **Aquí eso NO es cierto**: E2 (con E1 off) reclama
poder crear estructura "desde la nada". ε=0 es, por lo tanto, el caso más agudo para probar
esa afirmación — si con `phi_0` perfectamente plano (`contraste0_sq=0`) el motor con E2=on,
E1=off termina con `var(phi_final) > 0`, eso es evidencia directa y medida de creación de
estructura sin pagar. Por eso ε=0 **sí se simula** en este experimento (no se salta), y se
reporta aparte: como `contraste0_sq=0` hace indefinida la razón `eficiencia = E_ligada /
E_total` (0/0), para esa fila se reporta `eficiencia := None` (indefinida, no se inventa un
valor) junto con el diagnóstico crudo `var_phi_final` y `exceso_final` — la pregunta " ¿E2
crea var>0 de la nada? " se responde con ese número, no con la razón normalizada.

---

## 2. Observable, NULL y PASS (congelados ANTES de correr)

- **Observable primario:** `eficiencia_final = max(0, corr(phi_final, roll(phi_final,1))) *
  var(phi_final) / contraste0_sq`, calculado al final de una corrida de `pasos` fijos
  (calibrados, ver sección 3), para cada combinación (E1, E2, ε, r, semilla).
- **Observable secundario (diagnóstico de conservación, T6):** serie `exceso(t)` completa
  por corrida (se guarda min/mediana/max, no solo el final), y `E_lat_final`.
- **NULL:** al terminar la evolución, se permuta espacialmente `phi_final`
  (`rng.permutation`, exactamente igual que el NULL de `cs074_rcruz.py` y de E5.3-3) ANTES
  de medir `eficiencia`. La permutación no cambia `var(phi_final)` (invariante a
  reordenamiento) pero destruye la correlación vecino-a-vecino → si `eficiencia_NULL ≈
  eficiencia_REAL`, no hay estructura genuina atrapada, solo varianza cruda (T4: el NULL
  debe morder — y muerde exactamente la parte de correlación, no la de varianza,
  replicando el diseño ya validado por E5.3-3/cs074).
- **PASS (no binario, es una medida de sensibilidad — T5, según el enunciado exacto de
  E5.3-4):** se reportan, para cada (ε, r), los cuatro promedios de eficiencia sobre las 12
  semillas (una por combinación E1×E2), más:
  - `ΔE1(ε,r,E2) := eficiencia(E1=on,E2) − eficiencia(E1=off,E2)` (para E2 on y para E2 off
    por separado).
  - `ΔE2(ε,r,E1) := eficiencia(E2=on,E1) − eficiencia(E2=off,E1)` (para E1 on y para E1 off
    por separado).
  - Dispersión entre semillas (std) de cada uno de los 4 promedios, como piso de ruido para
    juzgar si un `Δ` es mayor que el ruido intrínseco.
  - Ningún umbral fijo separa "pasa/no pasa" — se entrega la distribución completa de
    `ΔE1` y `ΔE2` sobre toda la grilla (ε,r) como la medida de "cuán load-bearing" es cada
    axioma, tal como pide el enunciado.

---

## 3. Barrido (sobredimensionado, regla del director)

- **N** = 200 (mismo tamaño que `modo="produccion"` de `cs074_rcruz.py`, y que usan
  E5.2-2/E5.3-3 — D y comportamiento ya caracterizados ahí).
- **ε** ∈ {0.0, 1e-12, 1e-9, 1e-6, 1e-4, 1e-2, 1e-1, 1.0} — 8 valores, mismo rango de 12
  décadas usado por E5.2-2 (consistente entre hermanos de la batería).
- **r** = H/D ∈ {0.0, 1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0} — 12
  valores, mismo `R_LIST` que usó E5.2-2, log-espaciado de 1e-3 a 1e3 (rango pedido por la
  regla de oro), cruzando r≈1. `H = min(r·D, 1.0)`, D medido por `medir_D()` (no impuesto).
- **E1** ∈ {ON, OFF} × **E2** ∈ {ON, OFF} — las 4 combinaciones (núcleo de este
  experimento).
- **semillas** = 12 (mínimo exacto pedido por el enunciado de E5.3-4).
- **Perturbación dinámica (T7):** `paso_expansion` es Bernoulli por arista con `rng`
  avanzando cada paso (estocástico dentro de la corrida, no solo en la condición inicial);
  el propio mecanismo de inyección de E2 depende de qué aristas se cortaron ESE paso
  (estocástico también). Ambas fuentes de ruido dinámico, más las 12 semillas
  independientes, satisfacen T7.
- **pasos:** calibrado UNA vez (igual método que E5.2-2/cs074:
  `medir_pasos_lavado(N, eps=1e-2, semillas)`), fijo para las 4×8×12×12 combinaciones —
  a propósito NO se recalibra por variante de axioma, para que la comparación entre
  variantes sea a igualdad de duración (si `pasos` variara por axioma, cualquier diferencia
  en eficiencia sería confundible con duración distinta, no con el axioma).
- **Total de corridas:** 4 combos × 8 ε × 12 r × 12 semillas = 4608 corridas reales + 4608
  NULL (barajado final, cómputo casi gratis) = 9216 evaluaciones.

## 4. Verificación cruzada (regla de ejecución #4)

1. Su propio NULL (barajado espacial final, sección 2).
2. Segundo observable/método: la serie `exceso(t)` (violación cruda del presupuesto,
   independiente de la fórmula de correlación) reportada en paralelo a `eficiencia` — si
   `ΔE1` aparece en `eficiencia` pero `exceso(t)` es ≈0 en ambas variantes de E1, la
   diferencia sería sospechosa (artefacto de la fórmula, no del mecanismo real).
3. Chequeo interno de la predicción 1.4: se verifica numéricamente que
   `eficiencia(E1=on,E2=off) == eficiencia(E1=off,E2=off)` hasta 1e-9 en TODAS las filas
   (si falla, se reporta como hallazgo inesperado, no se oculta).
4. Auditoría en disco: JSON crudo completo (4608 filas reales + 4608 NULL) queda para que
   CS o un tercero lo revise sin re-correr nada.

## 5. T0–T7 — checklist

- **T0** nada discreto/dimensional puesto a mano: N, pasos, ε, r son parámetros de barrido
  declarados aquí, no ajustados a un resultado.
- **T1** ningún número puesto a mano: `D` medido; `contraste0_sq` medido; `E_lat0` medido
  del propio campo (`mean(phi_0)**2`); `delta_frac` medido (no `H` nominal); la única
  constante literal es el umbral de lavado `P_LAVADO` heredado sin cambios de
  `cs074_rcruz.py` (ya validado por los hermanos).
- **T2** observable ≠ juez: el juez es la fórmula de `eficiencia` (congelada en sección 1,
  heredada de E5.3-3), el observable es la curva/grilla completa de `ΔE1`/`ΔE2`.
- **T3** juez congelado antes de correr: secciones 1 y 2, escritas en este archivo antes de
  ejecutar el motor.
- **T4** el NULL debe morder: barajado espacial mata correlación sin tocar varianza.
- **T5** curva entera, no gate binario: se reporta la grilla completa (ε×r×4 combos), no un
  solo número "pasa/no pasa" (así lo pide el enunciado explícitamente para E5.3-4).
- **T6** conservación de E verificada cada paso, en las 4 variantes (con E1 on: verificada Y
  impuesta; con E1 off: verificada Y NO impuesta, reportando el exceso sin ocultarlo).
- **T7** barrido + perturbación dinámica: sección 3.

## 6. Cómputo

Benchmark de referencia de un hermano (E5.3-3, mismo N=200, misma clase de operaciones
vectorizadas): ~16 000 pasos/s. 9216 evaluaciones (4608 reales, con `pasos` de la
calibración de lavado — comparable a los ~3-6k pasos típicos de E5.2-2/E5.3-3 en este rango
de ε) más el costo casi nulo de sus 4608 NULL. Orden de magnitud esperado: decenas de
millones de pasos-corrida, minutos de cómputo. Autorizado (regla de ejecución #8, cómputo
largo autorizado). Se corre y se reporta el tiempo real transcurrido.

## 6bis. Addendum de rendimiento (escrito ANTES de la corrida de producción, T3 no se toca)

Un primer smoke-test del motor (loop por semilla, 1:1 con las funciones 1D de
`cs074_rcruz.py`) proyectó >3h para la grilla completa (demasiado lento para la sesión).
Se reescribió `E5_3_4_motor.py` para vectorizar las 12 semillas simultáneamente (batch
`(S=12, N=200)`), reimplementando `campo_inicial`, `paso_difusion`, `paso_expansion` en
versión batched **matemáticamente idéntica fila-por-fila** a las funciones originales de
`cs074_rcruz.py` (verificado con `_verificar_equivalencia()`, que corre automáticamente al
importar el motor: compara `campo_inicial_batch` y `paso_difusion_batch` contra las
funciones 1D originales, `atol=1e-12`; `paso_expansion` es estocástica por Bernoulli, se
deja documentado que el algoritmo es idéntico en vez de comparar valores exactos).
`cs074_rcruz.py` NO se edita ni se deja de usar (sigue siendo la fuente de `medir_D` y
`medir_pasos_lavado`, que no son el cuello de botella). **Ninguna definición de la sección
1 ni el criterio de la sección 2 cambió** — solo la implementación interna del loop
temporal. Esto se documenta aquí, antes de la corrida de producción, precisamente para
cumplir T3 (nada se ajusta después de ver resultados).

## 6ter. Bug detectado y corregido en el smoke-test (ANTES de producción)

El primer smoke-test de la versión batched-por-grilla completa (grid reducido, 4×4×12) dio
`chequeo_prediccion_E1_solo_con_E2.se_cumple = False` con `max|delta|=0.088` — la predicción
de la sección 1.4 (E1 no debería mover nada con E2 apagado) parecía fallar. Investigado
ANTES de correr producción: la causa era que `seed_axioma` se calculaba como `SEED_BASE +
1_000_000*axioma_idx`, es decir **cada una de las 4 variantes de axioma usaba una semilla
distinta** — el field `phi_0` y los sorteos de `paso_expansion` eran realizaciones
aleatorias DIFERENTES entre variantes, confundiendo cualquier comparación E1-on-vs-off con
ruido de muestreo, no con el efecto del axioma. Corregido: las 4 variantes ahora comparten
`seed_axioma = SEED_BASE` (misma semilla, ver comentario en `main()`), así que `phi_0` y el
flujo de números aleatorios de `paso_expansion` son idénticos entre variantes hasta el punto
en que el axioma en sí introduce una diferencia real (el clamp de E1 disparando, o el
consumo extra de `rng` de la inyección de E2 divergiendo el stream). Esto es exactamente el
tipo de error que la regla de ejecución #7 pide reportar y no ocultar — se documenta aquí
por transparencia, corregido antes de generar el resultado que se entrega a CS.

## 7. Archivos que este experimento va a producir

- `E5_3_4_motor.py` — motor (importa funciones de `cs074_rcruz.py`, no lo edita).
- `E5_3_4_resultado.json` — crudo completo (4608 combinaciones reales + 4608 NULL, series
  `exceso(t)` resumidas, metadatos de calibración).
- Reporte final se entrega en el mensaje de cierre al orquestador (CS), no en un .md nuevo
  de "hallazgos" (regla del entorno: no autoadjudicar veredictos, T3, regla de ejecución
  #7/#9).
