# DISEÑO CS065 — El ingrediente anti-colapso REAL: exclusión de Pauli. ¿Sostiene varios ejes sin fundirlos?
## CS065 — al sistema completo de CS064 se le añade la exclusión de Pauli entre fermiones (regla física fija, NO perilla). Pregunta: ¿la exclusión impide el colapso-a-1-eje y sostiene direcciones ortogonales múltiples, o el colapso aguanta igual?

**Diseña:** CS · **Fecha:** 9-jul-2026 · **A codear/ejecutar:** CC · **Hipótesis del ingrediente:** CS (recomendado), endosado por Alexis
**Origen:** CS064 dio su primer resultado real (barrido N∈{1500,2500,3500}, ~280 parches). Dos hallazgos:
  (1) FIRME — la exaptación tiene firma medible y robusta: congelar la orientación → 0 ejes en todo N; dejarla
      co-evolucionar → siempre >1. El "espacio es una exaptación" pasó su test O-N8.3 ("reutiliza estructura
      previa", medido, no declarado). Este resultado NO se toca — está cerrado.
  (2) ABIERTO — desenlace B': la inercia-de-la-mayoría colapsa a UN solo eje (73-81% de parches dan 1 eje). No
      emergió 3D. El experimento nombró su propio ingrediente faltante: algo que impida que todas las
      orientaciones se fundan en una — "no todos pueden apuntar igual".
CS064 también mostró que ni el catálogo del Modelo Estándar ni la mediación cambiaban nada (null_tipos ≈
null_mediado ≈ completo): a ese régimen, la dirección salía SOLO de la inercia de alineamiento. CS065 corrige
la omisión que explica ambas cosas.

---

## 0. LA REGLA DE ORO (que separa ciencia de trampa)
El resultado de CS064 dice "falta un ingrediente anti-colapso". La tentación es meter una fuerza calibrada para
que no colapse y celebrar el 3D. **ESO ESTÁ PROHIBIDO** — es jugar a Dios dentro de la cancha, parametrizar
hasta que salga 3. El ingrediente anti-colapso SOLO vale si cumple las tres condiciones:
  (a) es FÍSICA REAL del Modelo Estándar, no un invento ad-hoc;
  (b) lo OMITIMOS por error en CS064 — es una corrección, no una adición oportunista;
  (c) entra como REGLA FIJA (encendido/apagado), NO como parámetro que se afina mirando el resultado.
La exclusión de Pauli cumple las tres. Cualquier "fuerza anti-colapso" que no las cumpla queda fuera.

## 1. EL INGREDIENTE: exclusión de Pauli (por qué es honesto y no una perilla)
- **Es física real:** dos fermiones idénticos no pueden ocupar el mismo estado cuántico. Es el principio que da
  volumen a la materia, estructura a la tabla periódica, y estabilidad a las estrellas de neutrones. No es
  discutible ni ajustable — es un hecho del Modelo Estándar.
- **Lo omitimos:** quarks y leptones SON fermiones. En el inventario honesto que hicimos al cerrar CS062
  reconocimos que el motor modela la orientación como INERCIA DE ALINEAMIENTO (cada nodo tiende a apuntar donde
  apunta la mayoría — dinámica tipo imán/bandada). Nunca metimos lo contrario: que ciertos estados se RECHACEN,
  que no toleren ser idénticos. Esa es precisamente la exclusión. Estaba, y no la pusimos.
- **Por qué ataca justo el colapso:** el colapso-a-1 de CS064 es el desenlace típico de "todos se alinean con la
  mayoría" (consenso único). Pauli es la fuerza EXACTAMENTE OPUESTA a nivel de orientación: prohíbe que dos
  fermiones vecinos ocupen la MISMA orientación-estado. Si el alineamiento empuja a fundir todo en un eje, la
  exclusión empuja a DIFERENCIAR — y de esa tensión podrían sostenerse varios ejes ortogonales sin colapsar.
  Ortogonal = "lo más distinto posible" = lo que la exclusión favorece. La hipótesis es que el nº de ejes que
  sobrevive es el nº de orientaciones mutuamente excluyentes que caben.

## 2. QUÉ CAMBIA vs CS064 (mínimo — un solo actor nuevo, todo lo demás intacto)
Se parte del motor de CS064 SIN tocar nada (mismos ingredientes, mismas 4 fuerzas mediadas, misma expansión/
enfriamiento, misma medición de ejes/dimensión/holonomía). Se añade UNA regla en el paso de co-evolución del
marco:
- **CS064 (inercia sola):** cada nodo ajusta su orientación s_i hacia la media de sus vecinos (alineamiento).
- **CS065 (inercia + exclusión):** el mismo ajuste, MENOS un término de repulsión de orientación entre
  fermiones vecinos que penaliza que s_i quede paralelo (o antiparalelo, según spin) a un vecino fermión en el
  mismo estado. Formalmente: un costo que crece cuando dos fermiones vecinos comparten orientación-estado, y
  que el sistema minimiza empujando sus orientaciones a ser distintas/ortogonales.
- **Los bosones NO sienten exclusión** (fotón, gluón, W/Z, Higgs son bosones — pueden compartir estado). Esto
  es fiel a la física y da un discriminante interno gratis: si la dirección emerge de la exclusión, debería
  depender de la fracción de fermiones, no de bosones. Se registra para leerlo.
- **CRÍTICO — la fuerza de la exclusión NO se calibra:** su magnitud sale de la misma escala que las demás
  interacciones del motor (no un valor elegido para que salga 3). Si hay que barrerla, se barre en un rango
  amplio SORTEADO junto con el resto, nunca fijada a mano. Guardián G-NO-CALIBRAR.

## 3. LOS BRAZOS (anti-Shannon; la exclusión debe ganarle a su propio control)
- **excl (real):** sistema completo de CS064 + exclusión de Pauli entre fermiones, marco co-evolucionando.
- **sin_excl (= CS064):** EL CONTROL CENTRAL. Idéntico pero sin exclusión — es el CS064 que colapsó a 1 eje.
  Si `excl` no se separa de `sin_excl`, la exclusión no hace nada y B' se confirma como definitivo.
- **excl_barajada:** la exclusión se aplica pero entre pares AL AZAR (no entre fermiones-vecinos reales) —
  misma "cantidad" de repulsión, colocada donde no corresponde. Aísla si importa la ESTRUCTURA de la exclusión
  (quién excluye a quién) o solo meter repulsión genérica. Es la cuerda anti-Shannon fina: rompe el "asignar a
  mano qué excluye a qué".
- **excl_bosones (falso):** se aplica exclusión a los BOSONES en vez de a los fermiones — físicamente falso,
  control de placebo. Debe NO ayudar (o ayudar distinto). Si "cualquier cosa que excluya" da el mismo efecto,
  la exclusión no es el mecanismo — es repulsión genérica, y hay que decirlo.
- **marco_congelado:** como en CS064, orientación congelada → debe seguir dando 0 ejes (G-CONTINUIDAD, ancla).

## 4. EL BARRIDO
- **N ∈ {1500, 2500, 3500}** (mismas escalas que CS064, comparación directa) + si el motor lo permite, un punto
  mayor (N=5000 o más) para extender la curva N-dependiente que CS064 insinuó (19%→27% de parches con 2+ ejes).
- **Fracción de fermiones sorteada** por parche (no fija): permite leer si más fermiones ⇒ más ejes (predicción
  de que la exclusión es el motor).
- **~100+ parches por (N, brazo)** para tener la misma estadística de distribución que CS064 (no solo media:
  la DISTRIBUCIÓN de nº de ejes — % de parches con 1, 2, 3, 4+ ejes — es lo que de verdad se lee).
- Checkpoint por parche (como CS062/CS064). SMOKE primero (N=1000, ~10 parches, 5 brazos) validando que la
  exclusión discrimina en los calibradores antes de la tanda grande.

## 5. LO QUE SE MIDE (idéntico a CS064 + un discriminante nuevo)
- **nº de ejes** (espectro del tensor de orientación) — la variable central. MEDIA y DISTRIBUCIÓN por brazo/N.
- **dimensión espectral d_s** (paseo aleatorio) — fraccionaria y de escala, se reporta como tal (lección CDT).
- **δ/escala de Gromov** (planitud) y **holonomía** (consistencia global del marco).
- **NUEVO — correlación nº-ejes ↔ fracción-de-fermiones:** si la exclusión es el mecanismo, más fermiones ⇒
  más ejes sostenidos. Es una predicción interna que el propio dato puede confirmar o matar.

## 6. SALIDAS PRE-INSCRITAS (blind — se leen contra esto, NO se acomodan)
- **(A) LA EXCLUSIÓN ABRE LA DIMENSIÓN:** `excl` sostiene claramente MÁS ejes que `sin_excl` (p.ej. media 2-3
  vs 1.2), la distribución se corre a 2-3 ejes, `excl` > `excl_barajada` > `excl_bosones` ≈ `sin_excl`, y el nº
  de ejes correlaciona con la fracción de fermiones. ⇒ el ingrediente anti-colapso ERA la exclusión de Pauli;
  la dimensión múltiple emerge de la tensión alineamiento↔exclusión. Predicción más fuerte de la teoría, ganada
  con física real y sin calibrar. (Si además tiende a ~3 y se aplana ahí, es enorme — pero eso no se espera ni
  se fuerza; se lee si aparece.)
- **(B) ABRE EJES PERO NO 3:** `excl` sostiene 2+ ejes de forma robusta pero el número no es 3 (o no se
  estabiliza). ⇒ la exclusión SÍ es un ingrediente anti-colapso real (rompe el B' de CS064), pero no basta para
  fijar la dimensión en 3 — hallazgo fuerte, reorienta hacia qué más falta.
- **(C) NO CAMBIA NADA:** `excl` ≈ `sin_excl`. ⇒ la exclusión no era el ingrediente; el colapso-a-1 es más
  profundo. B' de CS064 se confirma como definitivo en el régimen accesible, y hay que buscar el faltante en
  otra parte (o aceptar que la inercia relacional sola no genera dimensión múltiple).
- **(D) PLACEBO:** `excl` ≈ `excl_bosones` ≈ `excl_barajada` > `sin_excl`. ⇒ ayuda meter CUALQUIER repulsión,
  no la exclusión específica. Entonces no es "Pauli" — es repulsión genérica; honestidad obliga a decirlo y a
  no vender la física.
- **(E) DEPENDE DE N:** el efecto de la exclusión crece con N (más ejes a más escala). ⇒ confirma la grieta
  N-dependiente de CS064 y que el régimen de "números enormes" es donde vive la respuesta.

## 7. GUARDIANES
- **G-NO-CALIBRAR:** la magnitud de la exclusión NUNCA se fija mirando el resultado; sale de la escala del motor
  o se sortea en rango amplio. (El guardián central — sin esto, es trampa.)
- **G-INTRÍNSECO:** ser fermión/bosón es propiedad fija al nacer, no reasignada por la geometría.
- **G-SIN-COORDENADAS:** ningún nodo tiene posición; nº de ejes, d_s, δ, holonomía son todos relacionales.
- **G-CONTINUIDAD:** `sin_excl` DEBE reproducir CS064 (colapso a ~1.2 ejes, 0 con marco congelado). Si no lo
  hace, el motor cambió y el experimento no es comparable — abortar y revisar.
- **G-PLACEBO:** los brazos `excl_barajada` y `excl_bosones` existen para que "meter repulsión" no se confunda
  con "meter Pauli". Sin ellos, un (A) no distingue mecanismo de artefacto.
- **G-SMOKE-ANTES:** no correr la tanda grande hasta que el smoke pase y CS adjudique el andamiaje.

## 8. LO QUE NO HACE / LÍMITES
- No prueba que la dimensión "sea" 3 — prueba si la exclusión rompe el colapso-a-1 y cuántos ejes sostiene. Que
  salga 3 sería un regalo, no el criterio de éxito. El éxito es que el test DISCRIMINE (excl vs sus controles).
- La exclusión aquí es un análogo relacional del principio de Pauli (repulsión de orientación-estado entre
  fermiones vecinos), no la mecánica cuántica completa. La fidelidad está en la ESTRUCTURA (fermiones se
  excluyen, bosones no; el rechazo empuja a la diferenciación), no en resolver la ecuación de Dirac.
- Sigue sin fijarse ningún parámetro para forzar un desenlace. El azar y la física fija juzgan.

---
**PREGUNTA (ninguna abierta de diseño):** el diseño está cerrado. La única decisión —qué ingrediente
anti-colapso— ya la tomamos: la exclusión de Pauli, por ser física real omitida y no una perilla. Listo para
que CC lo codee sobre el motor de CS064, con SMOKE antes de la tanda grande.

— CS. CS064 nombró su propio faltante; CS065 mete la única cosa que cumple la regla de oro (real + omitida +
no-calibrada). Si la dimensión múltiple emerge de la tensión alinear↔excluir, es un hallazgo con física de
verdad. Si no, el negativo es igual de limpio. No sé qué dará — y ese es el punto. El azar juzga, no nosotros.
