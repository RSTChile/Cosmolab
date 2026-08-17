# Informe para el equipo — Fase V-B y cierre de la línea A2-B0-C2 (10-11 de agosto de 2026)

**Para:** Equipo Transinteligente · **Preparado por:** CC (Claude Code) · **Dirige:** Alexis López Tapia ·
**Continúa de:** `INFORME_EQUIPO_FASE5_10ago2026_CS.md` (Fase V-A completa, candidato A2-B0-C2 identificado,
recomendado para validación física). Este documento cubre todo lo que pasó DESPUÉS: la batería de estrés
que pidió el equipo sobre A2-B0-C2, y la primera validación física real en Phantom.

## TL;DR — árbol de decisión, para no perderse en el detalle

```
¿A2-B0-C2 (grafo dinámico que se recablea solo, con límite de escala) sigue en pie como candidato?
│
├─ ¿Es un artefacto de escala escondida (violaría el filtro de admisión)? ────────── NO. Auditado y limpio.
│
├─ ¿La bimodalidad Clase I / Clase III es una transición de fase real? ──────────── PROBABLEMENTE NO.
│                                                                                     Es más probable que sea
│                                                                                     un umbral de clasificación
│                                                                                     cortando una variable
│                                                                                     continua que se mueve suave.
│
├─ ¿QUÉ mecanismo produce la geometría extendida (Clase III)?
│   ├─ ¿Es la señal de costo (historia+holonomía+compatibilidad)? ─────────────────  NO.
│   ├─ ¿Es agregarle "vecinos compartidos" a esa señal? ───────────────────────────  NO.
│   └─ ¿Es la RIGIDEZ del corte (cupo exacto, sin excepción) + el CRITERIO
│      correcto (soporte local) juntos? ────────────────────────────────────────── SÍ, evidencia fuerte.
│      (ninguno de los dos solo alcanza)
│
├─ ¿El patrón se sostiene con "redes sociales" (genealogías) genuinamente distintas? ── SÍ, aparentemente
│                                                                                        (poca muestra, n=4)
│
├─ ¿Y bajo GRAVEDAD REAL (Phantom), la Clase III de verdad acumula más masa? ────── SÍ, con significancia
│                                                                                     estadística (n=40 pares,
│                                                                                     p<0.001) — efecto MODESTO,
│                                                                                     no una frontera dura.
│
└─ ¿Se declaró cierre o veredicto en algún punto? ─────────────────────────────────  NO. Nunca. Todo son
                                                                                       números — la lectura
                                                                                       final es de Alexis.
```

## Cómo leer el resto del informe

Mismos símbolos que los informes anteriores: ✅ separación estadística sólida · ⊘ señal parcial o con caveat
real · ❌ no se separa del control / hipótesis no sostenida · ⚠️ falla o bug encontrado y corregido en el
camino · 🔧 infraestructura. Todo verificado contra archivos en disco (informes `.md`, CSVs de datos crudos,
carpetas reales de Phantom con sus dumps) — no contra lo que un agente narró haber hecho.

---

## PARTE 1 — Blindaje del candidato antes de Phantom (roadmap del equipo, 8 pasos)

El equipo (dos analistas, incluido GPT-5.6 Sol) propuso una batería de 8 pasos para estresar A2-B0-C2 antes
de gastar cómputo en Phantom. Se ejecutaron los primeros 4 (más 3 tareas adicionales "de piso" antes del
checkpoint):

### 1.1 — Auditoría de C2/kcap ✅

**Pregunta:** ¿`kcap` (el límite de escala) esconde geometría/distancia, lo cual invalidaría al candidato
por violar el propio filtro de admisión de Fase V?

**Resultado: PASS limpio.** `kcap` decide qué relación cortar mirando sólo cuántos vecinos comparten dos
nodos — cero coordenadas, cero distancia, en ningún lado del código. Invariante a reescalar el costo
0.01x-1000x, invariante a N. Único hallazgo menor (desempate por índice de nodo) resultó del mismo orden que
el ruido estocástico normal cerca de la frontera bimodal, no geometría oculta.

### 1.2 — Mapa de transición kcap×K ⊘

**Pregunta:** ¿la bimodalidad Clase I/Clase III es una transición de fase genuina?

**Resultado: probablemente NO, aunque no cerrado.** Grilla completa (4×5×20=400 reglas). A primera vista
había un "borde nítido" en P(Clase III) entre kcap=5 y kcap=6. Pero mirando los observables continuos de
fondo (no la clase, el número del que depende la clase): la pendiente baja SUAVE con kcap, sin ningún salto
propio — lo que pasa es que esa pendiente cruza el umbral fijo de clasificación (0.7) justo en ese tramo.
Es el mismo efecto que un "% de aprobados" que se desploma cuando el promedio de un examen cruza la nota de
corte, aunque nadie tuvo una caída individual brusca. Test de histéresis: sin memoria de estado (como corre
el motor real), cero diferencia entre direcciones — confirmado tanto por código como empíricamente.

### 1.3 a 1.6 — La línea del MECANISMO (el hallazgo más importante de esta tanda) ✅/❌

Cuatro experimentos consecutivos, cada uno construido sobre lo que dejó abierto el anterior, para entender
QUÉ hace que el cupo fijo (`kcap`) produzca 45% de Clase III mientras que cualquier alternativa "más
inteligente" se queda muy por debajo:

| experimento | qué probó | %Clase III | resultado |
|---|---|---|---|
| **F5-C2-C** | presupuesto de costo emergente (historia+holonomía+compatibilidad) en vez de cupo fijo | 15% vs. 45% del cupo fijo | ❌ indistinguible de podar al azar |
| **F5-C2-C2** | agregar "vecinos compartidos" (el criterio que SÍ usa el cupo fijo) a ese presupuesto | 10-15% | ❌ tampoco cerró la brecha |
| **F5-C2-C3** | separar RIGIDEZ (corte exacto) de UNIFORMIDAD (mismo número para todos) — cupo exacto pero que varía por nodo | **35%**, casi empatado con el 45% del cupo fijo | ✅ la rigidez SÍ importa mucho |
| **F5-C2-C3 (control)** | mismo cupo variable, pero cortando al azar en vez de por criterio | 5% | ❌ sin el criterio correcto, la rigidez sola tampoco alcanza |
| **F5-C2-C4** | cerrar la matriz 2×2 completa (rigidez × uniformidad) | celda final (elástico+variable) = 10%, IGUAL al elástico+uniforme | El MECANISMO domina 2.5x-∞ sobre la uniformidad, en ambas direcciones |
| **F5-C2-C5** | ¿el criterio importa DENTRO del mecanismo elástico también? | con criterio 10-15% vs. sin criterio (azar) 10% — **empate estadístico** | Hallazgo final: el criterio SÓLO importa si el corte ya es rígido |

**Conclusión de las 5 tareas juntas:** lo que reproduce la fuerza geométrica del cupo fijo NO es la
sofisticación de la señal de costo, ni si el número de cupo es igual para todos o varía — es la combinación
de **corte rígido y sin excepciones** + **criterio correcto (vecinos compartidos)**. Sacar cualquiera de los
dos ingredientes derrumba el efecto. Un hallazgo mecanístico limpio, no sólo un resultado negativo.

### 1.7 — Réplicas de genealogías independientes ⊘

4 "redes sociales" completamente separadas (semillas de partida bien distintas, no sólo distintos días de
la misma red). El patrón bimodal se sostiene en las 4 (cupo fijo: 45-75%, media 58.75%; nunca 0% ni 100%).
La varianza ENTRE genealogías no resultó mayor que la varianza YA CONOCIDA dentro de una sola genealogía por
puro azar de qué 20 semillas te tocan — pero con sólo 4 genealogías es poca muestra para probarlo con
fuerza. Hallazgo lateral honesto: un intento de "control de consistencia" resultó, por un desfase de
semilla no documentado antes, ser una 5ª muestra independiente en vez de una repetición exacta — quedó
documentado con transparencia, no escondido.

## PARTE 2 — Tres pendientes resueltos antes del checkpoint de Phantom

### 2.1 — A0 con métricas nativas de campo continuo ⊘

**Pregunta:** ¿el 27% de reglas A0 (sin grafo) que caían en "Clase II" en el barrido de 180 era un
artefacto del "grafo de medición" derivado (que muestrea pares al azar, receta de mundo-pequeño por
construcción)?

**Resultado:** se midió el campo directamente (correlación entre vecinos físicos inmediatos, sin ningún
grafo derivado) — las métricas nuevas SÍ distinguen campo real de ruido (firma de difusión clara). Pero las
2 reglas "Clase II" del método viejo caen EXACTAMENTE dentro del rango de las 33 "Clase I" en las métricas
nativas, sin agruparse ni siquiera entre sí. Y el propio método viejo, mirado de cerca, no tiene dos
poblaciones — es una nube continua apretada contra el umbral. Consistente con artefacto de medición, aunque
n=2 casos es chico para ser concluyente.

### 2.2 — Fase IV: refinar el 92%/8% (control local vs. global) ✅ — REETIQUETA un hallazgo previo

**Pregunta:** el 92% "consenso global" de Fase IV robustecido — ¿es realmente dispersión pareja sobre todo
el grafo, o es concentración en grupos de 3 (tríos), aunque estén mal armados?

**Resultado — cambia cómo hay que leer el 92%:** un mecanismo maximalmente disperso (cada arista recibe la
misma fuerza, sin ningún trío) resultó **indistinguible de ruido puro** (p=0.52) — cero aplanamiento. En
cambio, tríos concentrados —incluso mal armados (del triángulo equivocado, o completamente sueltos)—
preservan 80-92% del efecto de REAL. **Lo que ordena la red no es "hablar más fuerte para todos" — es
"hablar en grupos chicos de a tres", sea o no el grupo correcto.** El 8% que sí es "el grupo correcto"
se mantiene igual que antes. Hallazgo raro sin explicar: el trío "parecido pero equivocado" funcionó peor
que el trío "totalmente al azar" — pista abierta.

### 2.3 — CS073: factorial q_E×q_T (identidad de arista vs. orden de formación) ✅/🔧

**Resultado:** el orden de formación (q_T) no se ve en el instrumento usado (espectro del laplaciano) — no
por debilidad del efecto, sino por identidad matemática (el laplaciano depende sólo del grafo final, nunca
del orden de inserción). La identidad de las conexiones (q_E) domina el 77% de la varianza detectable.
Matiz importante: el orden SÍ deja huella real en las posiciones finales tras `layout_resortes` (38% de
diferencia, ya documentado antes) — el instrumento usado acá simplemente es ciego a esa huella específica,
no significa que no exista.

## PARTE 3 — Fase V-B: primera validación física real (Phantom)

Alexis autorizó explícitamente Phantom el 10-ago ("Sí, corremos Phantom"). Se siguió disciplina
piloto-primero en cada escalada, con checkpoints de costo antes de comprometer más cómputo.

### 3.1 — Piloto (n=3 pares) y la sospecha del "8 sumideros" ⚠️→🔧

Primer piloto: 3 pares Clase III vs. Clase I, N=2000, masa fija. Costo bajo (~15s/corrida Phantom). Las 6
corridas dieron EXACTAMENTE 8 sumideros — Alexis sospechó fallo de instrumento. Se investigó a fondo:

- **Descartado:** tope de configuración (revisado el código de Phantom, `maxptmass=1000`, muy por encima).
- **Descartado:** la semilla de turbulencia (probado con 2 semillas alternativas sobre el mismo grafo — el
  número no cambió).
- **Causa real encontrada:** la RESOLUCIÓN (N) domina el conteo de sumideros con margen enorme (8→29→122 al
  subir N a masa fija, patrón SPH conocido, no exclusivo de este proyecto). A N=2000, "8" es una moda muy
  fuerte (87% de 53 corridas históricas con grafos MUY distintos) pero no rígida (rango real 7-10). El
  conteo de sumideros satura a esta resolución — no sirve para distinguir nada. Las métricas que sí cargan
  información son fracción de masa acretada y κ_V.

### 3.2 — Escalada progresiva: n=3 → 8 → 20 → 40 pares ✅ (con un bug real encontrado y corregido) ⚠️

Al escalar de 3 a 8 pares se encontró y corrigió un bug real (colisión de nombres de regla entre lotes
generados en momentos distintos, que había mezclado 3 pares con el grafo equivocado) — detectado
comparando cada corrida contra sus metadatos de origen, documentado con transparencia total, no escondido.
Se blindó con doble verificación cruzada (antes y después de generar cada condición inicial) en todas las
escaladas siguientes — 0 errores en las ~65-90 min de cada corrida posterior.

**Resultado agregado final, n=40 pares (37 con parámetros de fondo K y kcap exactamente iguales):**

| subconjunto | n | %Clase III > Clase I (fracción de masa) | %Clase III > Clase I (κ_V) |
|---|---|---|---|
| n=3 (piloto) | 3 | 2/3 (66%) | 2/3 |
| n=8 | 8 | 6/8 (75%) / 5/5 en exactos (100%) | 6/8 |
| n=20 | 20 | 17/20 (85%) / 16/17 exactos (94%) | 16/20 |
| **n=40 (actual)** | 40 | **31/40 (77.5%) / 30/37 exactos (81.1%)** | **28/40 (70.0%)** |

La tendencia **Clase III > Clase I** en masa acretada bajo gravedad real se sostiene consistentemente al
escalar — no se diluye ni se revierte, aunque los 20 pares más recientes fueron algo menos unánimes (70%)
que la tanda anterior (85%).

### 3.3 — Primer test estadístico formal de la línea ✅

Sobre las 40 diferencias pareadas (`scipy.stats`):

| métrica | test de signos | Wilcoxon signed-rank |
|---|---|---|
| Fracción de masa (n=40) | **p=0.00068** | **p=0.00001** |
| Fracción de masa (n=37 exactos) | **p=0.00019** | **p=0.00001** |
| κ_V (n=40) | p=0.0166 | p=0.0032 |
| κ_V (n=37 exactos) | p=0.0201 | p=0.0087 |

**Interpretación honesta, sin sobre-vender:** estos p-valores dicen que el patrón observado NO se parece a
ruido/moneda al aire — es evidencia real de que hay algo sistemático. **No** establecen causalidad, **no**
confirman la teoría Cosmosemiótica, y **no** descartan confounds no controlados en el diseño pareado (mismo
K/kcap/N/protocolo, pero no necesariamente controla todo lo demás). El tamaño del efecto sigue siendo
MODESTO (Δfracción media ≈+0.01 sobre fracciones típicas de 0.06-0.15) — no hay una frontera dura tipo
"todas las Clase III muy por encima de todas las Clase I".

---

## SÍNTESIS FINAL — qué se sostiene, qué se cayó, qué queda abierto

**Lo que se sostiene con evidencia fuerte:**
- A2-B0-C2 pasó la auditoría de geometría escondida (no es un artefacto de escala disfrazada).
- El mecanismo que produce la geometría extendida está identificado con precisión: rigidez del corte +
  criterio de soporte local, juntos — no la señal de costo, no la uniformidad del número.
- Bajo gravedad real (Phantom), Clase III acumula más masa que Clase I de forma estadísticamente
  significativa (p<0.001 en fracción de masa, n=40 pares emparejados).

**Lo que se cayó o se revirtió:**
- La bimodalidad I/III probablemente no es una transición de fase física — es más consistente con un
  artefacto del umbral de clasificación sobre una variable continua.
- Ningún presupuesto "emergente" (con la señal de costo probada) reprodujo la fuerza del cupo fijo — hizo
  falta encontrar que era el MECANISMO, no la señal, lo que importaba.
- El "92% consenso global" de Fase IV se reetiqueta: no es dispersión pareja, es concentración en tríos
  (aunque mal armados) — sólo la dispersión pura, sin ningún trío, no aplana nada.

**Lo que queda genuinamente abierto:**
- El tamaño del efecto en Phantom es modesto, no una frontera dura — la pregunta de si esto "confirma" la
  teoría o es un efecto real pero menor sigue abierta.
- El artefacto de medición de A0 (n=2 casos, no concluyente).
- El hallazgo raro de Fase IV (trío-equivocado peor que trío-suelto) sin explicar.
- Si escalar Phantom más allá de n=40, o diseñar un control adicional (ej. pares emparejados al azar en vez
  de por clase genuina, para confirmar que el efecto desaparece sin la distinción I/III real).

**No se declaró cierre ni veredicto en ningún punto de esta tanda.** Todos los números están en los
informes individuales (`FASE5_auditoria_C2_resultado_CS.md`, `FASE5_mapa_transicion_C2_resultado_CS.md`,
`FASE5_presupuesto_emergente_CS.md`, `FASE5_presupuesto_soporte_local_CS.md`,
`FASE5_mecanismo_aislado_CS.md`, `FASE5_matriz_2x2_completa_CS.md`, `FASE5_control_azar_elastico_CS.md`,
`FASE5_genealogias_independientes_CS.md`, `FASE5_A0_metricas_nativas_CS.md`,
`FASE4_control_local_global_CS.md`, `CS073_factorial_qE_qT_CS.md`, `FASE5B_phantom_A2B0C2_piloto_CS.md`,
`FASE5B_investigacion_8sumideros_y_escala_CS.md`, `FASE5B_escala_20pares_CS.md`,
`FASE5B_escala_40pares_CS.md` — todos en este directorio, con metodología completa y CSVs de datos crudos
para auditoría directa). La lectura final de qué tan lejos llega esta evidencia es de Alexis.
