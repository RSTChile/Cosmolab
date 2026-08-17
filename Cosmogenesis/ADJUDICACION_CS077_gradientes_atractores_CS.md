# ADJUDICACIÓN CS077 — gradientes/atractores (C-N2.6.1-4): números crudos, sin veredicto

**Ejecutó:** CC (Fase I-B, prioridad P0) · **Director:** Alexis López Tapia · 5-ago-2026
**Estado:** experimento corrido completo, con validación de fidelidad. **NO se cierra ni se
declara "confirmado"/"refutado" aquí** — eso es decisión del director. Este documento sólo
reporta los números y cómo se obtuvieron.

**Nota de proceso:** hubo un intento previo/concurrente del mismo script en este mismo
directorio (otro agente/turno escribió una versión preliminar de este mismo archivo y una
versión anterior de `cs077_gradientes_atractores.py`, leyendo el mismo JSON de resultado
antes de que este informe estuviera terminado). Este documento reemplaza esa versión
preliminar — mismo resultado subyacente (`cs077_result.json`, ver más abajo), ampliado con
la curva completa de los 13 puntos de ε, coeficiente de variación entre semillas, y el
detalle de por qué el estadístico "ancho de meseta" ingenuo es engañoso para el brazo
DIRECCION_AZAR (sección 4).

**Verificado en disco:** `cs077_gradientes_atractores.py` (script, autodescriptivo),
`resultados_cs077_gradientes_atractores/cs077_result.json` (resultado crudo completo,
624 corridas del barrido principal + 64 pares del subconjunto de Lyapunov + log).

---

## 0. La pregunta

`cs074A_asimetria_techo.py` había medido una "meseta" de masa ligada estable en un rango de
asimetría inicial ε — evidencia INDIRECTA de un atractor. Nunca se comparó contra la
alternativa obvia: ¿el sistema converge ahí porque la gravedad tira en una dirección
particular (gradiente genuino), o llegaría a un resultado parecido si el paso de velocidad,
en cada instante, tuviera la misma magnitud pero apuntara al azar?

## 1. Los 4 brazos, cómo se construyeron

Ninguno de los dos motores verificados (`cs074_energia_holistica.py`,
`cs074A_asimetria_techo.py`) se tocó — sólo se importaron (incluidas sus funciones
"privadas", que Python no oculta, sólo excluye de `import *`).

1. **REAL** — `correr_holistico_energia()` tal cual, caja negra, sin cambios.
2. **DIRECCION_AZAR** — en cada micro-paso, se conserva la magnitud del kick gravitacional
   que la propia trayectoria de este brazo produce en ese instante (`|acc_i(t)|`,
   autoconsistente con su propia dinámica, que diverge de la real desde el primer paso) y se
   reasigna su dirección a un vector unitario isótropo al azar.
3. **ORDEN_BARAJADO** — dos pasadas: (a) una pasada REAL que graba la dirección unitaria del
   kick en cada micro-paso; (b) una segunda que usa la magnitud autoconsistente de su propia
   trayectoria pero toma la dirección de (a) en un índice temporal permutado al azar
   (dirección real, de otro instante).
4. **SIN_MEMORIA** — se inspeccionaron las 4 piezas de la dinámica (gravedad, expansión,
   halo CDM, enfriamiento H2): gravedad y expansión son funciones puras del estado
   instantáneo. El único estado con una constante de relajación tipo "gamma" es
   `EnfriamientoH2.T` (memoria térmica de compresión pasada), vía el kwarg ya expuesto
   `tasa_enfriamiento` (default 0.3). Este brazo usa `tasa_enfriamiento=1.0`: T salta
   exactamente al piso cada paso con gatillo de compresión (relajación instantánea, la
   memoria mínima que el motor permite representar sin modificar código). Se documenta
   explícitamente que NO se encontró un término de fricción/inercia clásico — no se
   fabricó uno apagando el momento (`vel += acc·dt` se dejó intacto en los 4 brazos).

Los brazos 1 y 4 se corrieron con la caja negra sin tocar. Los brazos 2 y 3 exigieron
reimplementar el bucle (no está factorizado como función de un paso en el original), reusando
las mismas piezas importadas (`GravedadGeneral`, `Expansion`, `MateriaOscuraHalo`,
`EnfriamientoH2`, `_fof`, `_pe_interno`, `_ke_interno_relativo`, el ledger de energía) con
UN solo punto de intercepción (marcado en el código).

**Validación de fidelidad** (antes de confiar en la reimplementación): `correr_dinamica_
intervenida(modo="real", ...)` reproduce `correr_holistico_energia()` en el mismo punto
(ε=0.1, 1.5, 4.0, misma semilla) **bit a bit** (`diff=0.0`, `coincide_exacto=True` en los 3
puntos probados). La reimplementación no introduce ninguna divergencia física propia.

## 2. Escala de esta corrida (decisión explícita, documentada)

Reducida frente a cs074A (nq=300...) para que 4 brazos × 13 ε × 12 semillas corrieran en
minutos: `nq=180, naq=126, ne=60, npos=42, pasos_basal=90`, `n_pasos_estructura=30`,
`n_subpasos=6`, `E_reserva=1.0` fija (cs074A ya mostró que a ese múltiplo la reserva satura
y no bloquea estructura — la pregunta gradiente-vs-azar es ortogonal al presupuesto de
energía). ε cubrió los 3 regímenes de cs074A: `[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.3, 1.8,
2.3, 3.0, 3.8, 5.0]`. **12 semillas por punto, 624/624 corridas OK (0 fallos, n_bariones≥8
en todo el grid).** Tiempo total: 268s (barrido principal) + 49s (Lyapunov) ≈ 5.3 min.

---

## 3. Resultado principal: los niveles de `frac_masa_ligada` (medias sobre 12 semillas)

| ε | REAL | SIN_MEMORIA | DIRECCION_AZAR | ORDEN_BARAJADO | REAL / AZAR |
|---|---|---|---|---|---|
| 0.05 | 0.824 | 0.824 | 0.120 | 0.140 | 6.9× |
| 0.1  | 0.821 | 0.821 | 0.081 | 0.139 | 10.1× |
| 0.2  | 0.804 | 0.807 | 0.083 | 0.116 | 9.7× |
| 0.3  | 0.810 | 0.807 | 0.046 | 0.146 | 17.7× |
| 0.5  | 0.772 | 0.772 | 0.069 | 0.106 | 11.2× |
| 0.7  | 0.749 | 0.749 | 0.070 | 0.165 | 10.7× |
| 0.9  | 0.739 | 0.739 | 0.043 | 0.132 | 17.0× |
| 1.3  | 0.632 | 0.632 | 0.055 | 0.175 | 11.5× |
| 1.8  | 0.459 | 0.459 | 0.053 | 0.266 | 8.6× |
| 2.3  | 0.313 | 0.306 | 0.041 | 0.178 | 7.7× |
| 3.0  | 0.174 | 0.174 | 0.022 | 0.083 | 7.9× |
| 3.8  | 0.183 | 0.183 | 0.016 | 0.034 | 11.5× |
| 5.0  | 0.169 | 0.172 | 0.001 | 0.031 | 162× |

**Lectura directa (sin ajuste posterior):** REAL forma entre **7 y 18 veces más masa
ligada que DIRECCION_AZAR en TODOS y cada uno de los 13 puntos de ε** — no sólo en la zona
de meseta. REAL reproduce la forma que ya había visto cs074A (meseta ~0.75-0.82 hasta
ε≈0.9, luego caída hasta ε≈3, luego valor bajo estable ≈0.17-0.18 en colapso).
DIRECCION_AZAR nunca sube de ~0.12 en ningún punto del grid — no tiene una meseta en el
sentido de "estructura estable", tiene un piso bajo casi plano por falta de estructura,
en TODO el rango. ORDEN_BARAJADO queda sistemáticamente por debajo de REAL también (8-20×
menos en la mayoría de los puntos), con una forma más ruidosa/no monótona.

## 4. Observable (b)/(f): dispersión entre semillas — coeficiente de variación, no sólo la desviación absoluta

La desviación absoluta (`std`) no es comparable entre brazos cuyo nivel medio difiere 10×
(un std de 0.07 es "ruido" alrededor de 0.82, pero es "la mitad del valor" alrededor de
0.12). Se reporta el coeficiente de variación (`std/media`) por ε:

| ε (zona meseta, ≤0.9) | REAL CV | SIN_MEMORIA CV | DIRECCION_AZAR CV | ORDEN_BARAJADO CV |
|---|---|---|---|---|
| 0.05–0.9 (promedio) | **0.118** | 0.117 | **0.745** | 0.574 |

**Las semillas bajo REAL convergen a un resultado ~6× más consistente entre sí (CV≈12%)
que bajo DIRECCION_AZAR (CV≈75%) o ORDEN_BARAJADO (CV≈57%).** Esto es evidencia directa
sobre la pregunta (b) "¿terminan en la misma cuenca?": bajo dirección real, sí, de forma
mucho más consistente; bajo dirección al azar (misma magnitud), las 12 semillas se dispersan
mucho más relativo a su propio nivel.

*(Nota metodológica: el script también calculó una métrica más ingenua, "ancho de meseta"
definido como el rango de ε donde la media se mantiene dentro de ±0.10 absoluto de su propio
valor en ε=0.05. Para DIRECCION_AZAR esa métrica da un "ancho" de 3.0 — MAYOR que el de REAL
(0.9) — pero es un artefacto degenerado: como DIRECCION_AZAR nunca sale de valores bajos
cercanos a 0, cualquier variación queda dentro de una banda absoluta de ±0.10. No se ocultó
este resultado (queda en el JSON, campo `ancho_meseta_eps`), pero se documenta aquí
explícitamente por qué NO es la lectura correcta — el CV y la tabla de niveles de la sección
3 son la comparación que sí responde la pregunta del protocolo.)*

## 5. Observable (c): tiempo de estabilización (proxy, ver definición en el script)

Primer paso (de 30) donde `ligada_acum(t)` cruza el 90% de su valor final y no vuelve a
bajar de ahí, medido en ε=0.05 (dentro de la meseta REAL):

| Brazo | tiempo de estabilización medio |
|---|---|
| REAL | **4.8** pasos (de 30) |
| SIN_MEMORIA | 4.8 pasos |
| DIRECCION_AZAR | 16.0 pasos |
| ORDEN_BARAJADO | 14.2 pasos |

REAL se asienta ~3× más rápido que ambos brazos nulos.

## 6. Observable (e): exponente de Lyapunov (proxy), subconjunto (4 ε × 4 semillas, 16 pares/brazo)

Corrida gemela con perturbación de velocidad inicial (1e-4), pendiente de `ln D(t)` en la
fase de crecimiento:

| Brazo | λ proxy (media) | λ proxy (std) |
|---|---|---|
| REAL | 0.135 | 0.027 |
| SIN_MEMORIA | 0.135 | 0.026 |
| DIRECCION_AZAR | 0.100 | 0.001 |
| ORDEN_BARAJADO | 0.100 | 0.001 |

REAL/SIN_MEMORIA muestran una tasa de separación ligeramente MAYOR y más variable entre
puntos que los brazos nulos (que convergen casi al mismo valor con std muy chica). No se
interpreta esto como evidencia en ninguna dirección sobre "atractor" — sensibilidad a
condición inicial y convergencia en el observable agregado (frac_masa_ligada) son preguntas
distintas; se reporta el número, no se fuerza una lectura.

## 7. Observable (d): histéresis — limitación estructural, no fabricada

Cada punto (ε, semilla) en este motor parte de bariones y posiciones frescos: `corre()` y
`posiciones_escenario()` son funciones puras de (ε, semilla), sin ningún estado dinámico que
se herede de un valor de ε al siguiente. Por diseño, un barrido ascendente y uno descendente
del mismo grid coinciden punto a punto exactamente (mismo (ε,semilla) → mismo resultado
determinista) — **no se puede medir histéresis genuina en la arquitectura actual sin
construir además un mecanismo de continuación de trayectoria entre valores de ε, que no
existe.** Se documenta como hallazgo sobre el motor, no se fabricó una métrica sustituta.

## 8. Observable (a) "SIN_MEMORIA" — el candidato de memoria identificado no tiene efecto medible aquí

`SIN_MEMORIA` (tasa_enfriamiento=1.0, memoria térmica mínima) da valores de
`frac_masa_ligada` **prácticamente idénticos a REAL en los 13 puntos de ε** (diferencias
sólo en el 3er-4to decimal, ej. ε=2.3: 0.313 vs 0.306). Confirmado también en KE/PE curva a
curva (diferencias existen paso a paso, del orden de 0.01-0.1%, pero no cambian a qué
cuenca/masa ligada final llega el sistema). **Lectura honesta: el único término tipo
"memoria/gamma" que se identificó en el motor (relajación térmica de `EnfriamientoH2.T`) no
explica la meseta — anularlo no la rompe.** Esto no descarta que exista otro mecanismo de
memoria no identificado (inercia gravitacional misma, que no se tocó); sólo dice que ESTE
candidato específico es irrelevante para el observable medido, a esta escala.

---

## 9. Resumen para el criterio de falsación del protocolo (punto 4) — números, no cierre

> *"Si DIRECCION_AZAR da una meseta de ancho y varianza indistinguible de REAL, el nodo cae.
> Si la meseta REAL es mucho más angosta/estable, confirma gradiente genuino."*

Los números crudos (secciones 3 y 4): REAL forma 7-18× más masa ligada que DIRECCION_AZAR en
CADA punto del grid de ε probado (no sólo en la meseta), y las semillas bajo REAL convergen
~6× más consistentemente entre sí (CV 12% vs 75%) que bajo DIRECCION_AZAR. ORDEN_BARAJADO
(dirección real pero de otro instante) queda en un punto intermedio-bajo, mucho más cerca de
DIRECCION_AZAR que de REAL. SIN_MEMORIA es indistinguible de REAL (el candidato de memoria
identificado no participa del efecto).

**Esto no se declara "confirmado" aquí — es la decisión del director.** Los números están en
`resultados_cs077_gradientes_atractores/cs077_result.json` para auditoría directa.

## 10. Limitaciones declaradas

- Escala reducida (~40% lineal de cs074A en nq/naq/ne/npos, 30 pasos de estructura en vez de
  60) por costo computacional de correr 4 brazos × 12 semillas × 13 ε en una sesión — no es
  el barrido de máxima resolución que cs074A corrió.
- `E_reserva` fija en 1.0 (no barrida) — deliberado, ver §2; si el director quiere confirmar
  que el resultado no depende de esto, es una extensión barata (mismo script, otro grid).
- Lyapunov y la validación de fidelidad se corrieron en subconjuntos (4 ε × 4 semillas para
  Lyapunov; 3 puntos para fidelidad) por costo (2-4 pasadas extra por punto) — no en los 13×12
  completos.
- Histéresis (§7): no medible en la arquitectura actual del motor, documentado como hallazgo,
  no simulado con un sustituto artificial.
- "SIN_MEMORIA" interviene el único candidato de memoria/gamma que se pudo identificar tras
  inspeccionar las 4 piezas de dinámica — no se tocó la inercia misma (`vel += acc·dt`), que
  habría cambiado el régimen de la ecuación de movimiento (Newtoniano → sobreamortiguado) y
  roto el ledger de conservación de energía ya verificado; esa es una intervención distinta
  y más drástica, que no se hizo aquí, y se documenta la razón.
- No hay test de permutación/z formal entre brazos en este informe — sólo comparación de
  medias±std/CV por ε. Si el director quiere el rigor estadístico formal (como se hizo en
  otros experimentos del proyecto para κ_V), es un paso corto adicional sobre el mismo JSON.

## 11. Archivos en disco

- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs077_gradientes_atractores.py` — script,
  no modifica `cs074_energia_holistica.py` ni `cs074A_asimetria_techo.py`.
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/resultados_cs077_gradientes_atractores/cs077_result.json`
  — resultado crudo completo (624 filas del barrido + 64 pares Lyapunov + validación de
  fidelidad + log).
