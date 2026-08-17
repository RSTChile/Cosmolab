# PROTOCOLO E5.3-3 — Estabilidad temporal de la eficiencia (¿congela o deriva?)

**Pre-registrado antes de escribir el motor.** Timestamp de creación: 2026-07-24 16:40 -04
(America/Santiago, `date` del sistema al momento de escribir este archivo).

Agente: E5.3-3 (Enfoque 5, Tema 3). Corre EN PARALELO con 29 agentes más — prefijo propio
`E5_3_3_`, carpeta propia `BATERIA_ENFOQUE5/E5_3_3_estabilidad_temporal/`. No se toca nada
fuera de esta carpeta. No se edita `cs074_rcruz.py` (se importa como librería de solo
lectura).

---

## 0. Estado de E5.3-1 al momento de escribir esto

Se buscó en disco `BATERIA_ENFOQUE5/E5_3_1_eficiencia_12decadas/` (definición de eficiencia
a reutilizar, según instrucción). **No existe todavía** — el único directorio presente en
`BATERIA_ENFOQUE5/` al momento de arrancar es `E5_1_1_supervivencia_exergia` (más
`E5_3_3`, `E5_3_4`, `E5_3_5`, las carpetas ya creadas por agentes hermanos de Tema 3, sin
contenido de definición). No hay `find -iname "*eficiencia*"` con resultados.

**Decisión:** como no hay definición de E5.3-1 en disco para reutilizar, se construye una
definición de eficiencia PROPIA pero explícitamente diseñada para ser consistente con:
(a) el enunciado de TEMA 3 en el documento autoritativo (`eficiencia = E_ligada/E_total`,
SALIDA, no se fija a mano), y (b) la física YA IMPLEMENTADA en `cs074_rcruz.py`, que es el
código base común de todo Enfoque 5 (campo φ en anillo, difusión solo por aristas vivas,
expansión = corte de aristas, NULL = barajado). Se reutilizan tal cual las funciones de
`cs074_rcruz.py` (`campo_inicial`, `paso_difusion`, `paso_expansion`, `medir_D`,
`persistencia`) — no se reinventa la dinámica, solo se la observa en el tiempo.

Si al momento de reportar a CS ya existe una definición registrada por E5.3-1, CS puede
cotejar ambas; esta corrida no se detiene ni se reajusta a posteriori (T3).

---

## 1. Definición operacional de eficiencia(t)

En `cs074_rcruz.py`, `persistencia(phi, contraste0)` ya mide exactamente la cantidad que
Tema 3 llama eficiencia: la fracción del presupuesto ESTRUCTURAL inicial (contraste/varianza
del campo, generado por ε) que sigue **atrapada de forma coherente** (correlación espacial
> 0) en vez de haberse perdido en ruido difusivo:

```
eficiencia(t) := persistencia(phi_t, contraste0)
              = max(0, corr(phi_t, roll(phi_t,1))) * [var(phi_t) / contraste0**2]
```

- `E_total` ≡ `contraste0**2` = `var(phi_0)` — el presupuesto de estructura fijado por ε al
  arrancar (nunca se toca durante la corrida: la difusión SOLO puede bajar la varianza, la
  expansión SOLO corta aristas; ninguna operación puede aumentar `var(phi)` por encima de
  `var(phi_0)` → E1 (conservación, en el sentido de "no se fabrica estructura de la nada")
  se cumple por construcción de la dinámica ya validada en CS074).
- `E_ligada(t)` ≡ `corr_local(t) * var(phi_t)` — la parte de lo que queda de varianza que
  además está organizada (vecinos correlacionados), es decir, "atrapada como estructura" y
  no como ruido residual sin orden.
- `eficiencia(t) = E_ligada(t) / E_total ∈ [0, 1]` por construcción (corr≤1, var(phi_t)≤
  var(phi_0) siempre, porque cada paso de difusión promedia con vecinos y no puede aumentar
  la varianza global).

E2 (redistribución adiabática) no aplica a este experimento — E5.3-3 no toca enfriamiento
ni temperatura; se declara N/A y no se mide aquí (le corresponde a Tema 4).

**Por qué esta y no otra:** es la única cantidad que el propio motor de Enfoque 5 ya
calcula para "cuánto de la diferencia inicial sigue siendo estructura real"; no se inventa
ninguna fórmula nueva ni se ajusta ningún coeficiente. T1 (ningún número puesto a mano): D,
contraste0, corr y var son todos MEDIDOS del campo mismo.

---

## 2. Observable, NULL y PASS (congelados ANTES de correr)

- **Observable:** curva completa `eficiencia(t)` por cada combinación (ε, r, semilla),
  muestreada en pasos log-espaciados dentro de `[1e2 … 1e5]` (10 checkpoints:
  100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000), dentro de UNA sola
  trayectoria continua (no se reinicia el campo en cada checkpoint — así "estabilidad
  temporal" se mide sobre la misma historia, no sobre corridas independientes de distinta
  duración).
- **NULL:** en cada checkpoint, se baraja (`rng.permutation`) el campo φ_t ANTES de medir
  persistencia — mismo φ, mismo `var(phi_t)`, pero sin orden espacial. Si `eficiencia_real(t)`
  no se separa de `eficiencia_NULL(t)`, no hay estructura genuina atrapada en ese punto (T4:
  el NULL debe morder — y en este caso SÍ puede morder, porque barajar destruye la
  correlación vecino-a-vecino sin tocar la varianza, a diferencia de un NULL que preservara
  ambas cantidades).
- **Criterio de congelamiento (fijado ANTES de correr, T3):** para una combinación (ε, r,
  semilla) dada, sea `e_k` la eficiencia real en el checkpoint k (k=1..10, pasos
  crecientes). Se dice que la curva **congela en el checkpoint k\*** si, para TODO k ≥ k\*,
  `|e_k − e_{k-1}| / max(e_{k-1}, 1e-6) < 0.02` (cambio relativo <2% entre checkpoints
  consecutivos, sostenido hasta el final de la corrida — no basta un solo punto quieto). Si
  ninguna k satisface esto hasta pasos=1e5, se reporta **"no congela en el rango"**, sin
  extrapolar. El valor de eficiencia en el punto de congelamiento NO se fija de antemano —
  es lo que salga.
- **PASS:** se reporta SI hay congelamiento y a qué paso (mediana y dispersión entre las 12
  semillas, por cada ε×r), separado de si sigue derivando. Ambos resultados (congela /
  deriva) son hallazgos válidos, no hay "éxito" fijado de antemano (T3, regla 9 de
  ejecución: "cualquier NEGATIVO es hallazgo, no fracaso").

---

## 3. Barrido (sobredimensionado, regla del director)

- **N** = 200 (tamaño de anillo, igual al `modo="produccion"` de `cs074_rcruz.py` — D y
  pasos_lavado ya caracterizados ahí para este N).
- **pasos:** checkpoints log-espaciados en `[1e2, 1e5]`, 10 puntos (arriba). Cubre las 3
  décadas pedidas por el enunciado íntegro.
- **ε** ∈ {0.0 (control puro), 1e-6, 1e-4, 1e-2, 0.1, 1.0} — 6 valores, de "sin diferencia"
  a "perturbación total", varias décadas.
- **r** = H/D ∈ {0.0, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0} — 8 valores, log-espaciado
  de 1e-3 a 1e3 (rango pedido explícitamente por la regla de oro de esta batería), cruzando
  r≈1. `H = min(r · D, 1.0)`, con D MEDIDO por `medir_D(N, eps, seed)` (no impuesto).
- **semillas** = 12 (mínimo pedido por el enunciado E5.3-3).
- **Perturbación dinámica (T7):** la propia dinámica de corte de aristas
  (`paso_expansion`) es estocástica paso a paso (Bernoulli por arista con `rng` avanzando en
  cada paso), no solo en la condición inicial — esto ya provee la perturbación dinámica
  exigida por T7, además de las 12 semillas independientes.
- **Total de trayectorias:** 6 ε × 8 r × 12 semillas = 576 corridas completas a pasos=1e5
  (con 10 checkpoints cada una, real + NULL). El caso ε=0 es degenerado por construcción
  (φ constante, var=0 desde el origen → eficiencia≡0 en todo t, sin necesidad de simular
  1e5 pasos reales): se calcula analíticamente para ahorrar cómputo, y se documenta como tal
  en el resultado (no afecta ninguna otra fila).
- **NULL adicional de todo el experimento:** además del NULL por checkpoint (barajado
  espacial), se reporta la curva NULL agregada por (ε,r) para comparar contra REAL en cada
  paso de la sección 2.

## 4. Verificación cruzada (regla de ejecución #4)

1. Su propio NULL (barajado por checkpoint, sección 2).
2. Segundo observable: `var(phi_t)/var(phi_0)` (parte "cruda" de la eficiencia, sin el
   factor de correlación) reportado en paralelo — si el congelamiento aparece en
   `eficiencia` pero NO en el ratio de varianza crudo, se marca como artefacto de la
   correlación, no de energía real congelada.
3. Auditoría en disco: JSON crudo completo con las 576×10 filas (real y NULL) queda para
   que CS o un tercero lo revise sin tener que re-correr nada.

## 5. T0–T7 — checklist

- T0 nada discreto/dimensional puesto a mano: N y pasos son parámetros de barrido
  declarados aquí, no ajustados a un resultado.
- T1 ningún número puesto a mano: D medido; contraste0 medido; eficiencia sale del campo.
- T2 observable ≠ juez: el juez es el criterio de congelamiento (sección 2, umbral 2%
  fijado antes de correr), el observable es la curva completa.
- T3 juez congelado antes de correr: sección 2, escrita en este archivo antes de ejecutar
  el motor.
- T4 el NULL debe morder: barajado espacial mata la correlación sin tocar la varianza.
- T5 curva entera, no gate binario: se reporta eficiencia(t) completa por (ε,r), no un
  solo número.
- T6 conservación de "energía" (var(phi)≤var(phi_0)) verificada en cada checkpoint —
  cualquier violación (por error numérico) se reporta como falla, no se oculta.
- T7 barrido + perturbación dinámica: sección 3.

## 6. Cómputo

Benchmark previo: ~16 000 pasos/s para N=200 (difusión + expansión vectorizadas, un solo
hilo). 576 trayectorias × 1e5 pasos ≈ 5.76e7 pasos-corrida ≈ 50–60 min de cómputo en un
solo proceso. Autorizado (regla de ejecución #8: "ejecutar completo, que tarde lo que
tarde"). Se corre en background y se reporta el tiempo real transcurrido al final.

## 7. Archivos que este experimento va a producir

- `E5_3_3_motor.py` — motor (importa funciones de `cs074_rcruz.py`, no lo edita).
- `E5_3_3_resultado.json` — crudo completo (576 combinaciones × 10 checkpoints × real+NULL).
- Reporte final se entrega en el mensaje de cierre al orquestador (CS), no en un .md nuevo
  de "hallazgos" (regla del entorno: no autoadjudicar veredictos).
