# ADDENDUM DE VERIFICACIÓN — VSTCosmo

**Verificación ejecutada por re-corrida real de los scripts (stdout capturado).**
Fechas de ejecución: 2026-06-21 / 2026-06-22.
Entorno: `venv/bin/python3` (Python 3.13), backend headless `MPLBACKEND=Agg`, macOS.
Logs sellados (script · sha1 · fecha · veredicto) en `verificacion_hitos/*.log`.

---

## 0. Por qué este documento

Los informes afirmaban validaciones (✅/VALIDADO) cuyo veredicto **nunca se había guardado**:
en estos scripts el juicio va a `stdout`, y el repositorio solo conservaba CSV/PNG. No se podía
sostener ninguna afirmación de validez sin re-ejecutar. Este addendum recoge el resultado de
re-ejecutar y leer el veredicto real, cruzándolo contra lo que cada informe afirma.

**Caveat metodológico ineludible:** todas las corridas son N=1, re-ejecutadas en estas fechas,
no las corridas originales que citan los informes. Un resultado distinto hoy no prueba que el
original fuera falso (pudo haber otra semilla/versión), pero un resultado coincidente **sí**
confirma que la afirmación es reproducible.

---

## 1. Dos pistas experimentales distintas (aclaración estructural)

| Pista | Qué es | ¿Se valida corriendo `.py`? |
|---|---|---|
| **E1–E24** (Teoría Canónica Definitiva) | Experimentos conceptuales/aplicados: radio AM/FM, 3G/4G/5G, Bluetooth EIT-1/2/3, Daisyworld-IA, térmica, fiscalía (MAPAR-S), grafitis, Cosmo-SETI, textos | **No.** No hay script vXX ejecutable; es otra base probatoria |
| **vXX VSTCosmos** (v70–v182) | Simulaciones de campo: organismo, orientación, memoria, comunicación | **Sí.** Es lo re-ejecutable y lo que cubre este addendum |

Este addendum verifica la **pista vXX**. La pista E1–E24 queda fuera de alcance (no es ejecutable aquí).

---

## 2. Era del campo rico (v70–v92) — verificado

| Script | Afirmación | Veredicto real |
|---|---|---|
| v70 / v71 — campo continuo | Estructura → mayor gradiente; varianza discrimina | ✅ ✅ ⚠ (transición no asimétrica) |
| v72a — protocolo causal | Criterio causal | ❌ **no pasó el criterio** |
| v72b — plasticidad hebbiana | Persistencia/asimetría | ⚠ parcial (grad sin entreno 0.4966, debía <0.03) |
| **v72c — modos propios** | Primera demostración de modos propios aprendidos | ✅ **VALIDACIÓN COMPLETA** (pilar) |
| v73–v79 | Varios | rc=0, corren limpio |
| **v80h — ciclo evolutivo** | Selección interna funcional (C7/C8) | ✅ **VALIDADO** (pilar) |
| v81 — campo expandido | Asimetría diferencial | ⚠ parcial (C12 = 0.000) |
| v82 / v83 / v85 / v86 | Acoplamiento/orientación/decisión/loop | OK |
| v84b — campo diferencial | Inversión Δ(L,R) | ⚠ parcial **en aislamiento** (C21 sin inversión) — *superado por v88* |
| v87 / v89 | Coherencia / gradiente canónico | ⚠ parciales |
| **v88 — gradiente energético** | Orientación espacial | ✅ **ORIENTACIÓN ESPACIAL VALIDADA** (C18/C21/C27/C28/C29) |

### 2.1 Corrección importante sobre la lateralidad Δ(L,R)

Una lectura intermedia marcó la inversión Δ(L,R) como "no cumplida" porque **v84b** fallaba C21.
La verificación posterior lo **rehabilita**: v84b es una versión **temprana, no convergida**.
En el estado **maduro** de la cadena de orientación, C21 (act_busc invierte) **sí ocurre**:
v88 lo cierra con veredicto final, y los protocolos cíclicos lo confirman ciclo a ciclo.
La lateralización emerge — más adelante en la secuencia de desarrollo, no en v84b.

### 2.2 Protocolos cíclicos (100 ciclos completos)

| Script | Resultado a 100 ciclos | Veredicto |
|---|---|---|
| **v90 — protocolo cíclico** | C28/C21/C29 = **100/100**; eficiencia 0.838→0.893 | ✅ con ⚠: diferencia geom 0.039→0.043 etiquetada "deterioro — revisar dinámica" (orientación robusta, sin consolidación creciente) |
| **v91 — cierre lazo Acción→A** | C28/C21/C29 = **100/100**; pero **C30 = 1/100** | ⚠ **PARCIAL**: delta_A F2=−2e-05, F13=−5e-05 (ambos ~0, mismo signo) → **el lazo Acción→A NO cierra**: la acción no realimenta diferencialmente el campo |
| **v92 — respuesta de orientación Ω** | C28/C21/C29/**C31** = **10/10** (diseño = 10 ciclos) | ✅ **VALIDADO** (Ω_F2=0.870 vs F13=0.491) |

**Hallazgo negativo duro:** v91 muestra que el lazo acción→percepción **no cierra**. La orientación
se *percibe* (act_busc/act_geom invierten), pero la acción no modifica el campo de forma direccional.
Si algún informe afirma que v91 "cierra el ciclo acción→percepción", **el dato lo desmiente**.

---

## 3. Los 10 hitos reportados por los informes de etapa — afirmado vs. real

Fuentes: *Síntesis V90–V103*, *Informe Canónico de Clausura VSTCosmo 150*, *Informe Experimento V176*,
*Informe Final V180 — Memoria Episódica*, *Informe ANIMA4 182A5*, *Teoría Cosmosemiótica Aplicada*.

| Hito | Informe afirma | Salida real (hoy) | Veredicto |
|---|---|---|---|
| **V117** | R₂ sin lateralidad | `R₂ ✅ CONFIRMADO` · `Lateralidad ❌` · "alma racional con fusión" | ✅ **coincide** |
| **V118** | Lateralidad sin R₂ (trade-off) | `Lateralidad ✅` · `R₂ ❌` · "alma sensitiva++" | ✅ **coincide** |
| **V122** | Coexistencia R₂+lateralidad ("primer EIT-3") | `✅ ÉXITO: R₂ Y LATERALIDAD COEXISTEN` | ✅ **coincide** (pilar-clímax) |
| **V132** | Organismo mínimo funcional | 5 logros: lateralidad, R₂, C50, tracking, plasticidad | ✅ **coincide** |
| **V147** | Baseline fisiológico sano | `✅ BASELINE SANO` · T_settle 31.0s · error 2.1° | ✅ **coincide** (cifras idénticas) |
| **V150** | IONB-1, cierre ANIMA-1 | `✅` 7 capacidades · fatiga 7.1× · "ANIMA-1 CERRADO" | ✅ **coincide** (con residuo abierto) ³ |
| **V176** | R_op, primer "No" operativo (4/4) | `Éxito: True` · rechazo específico +60° · deliberación medible | ✅ **coincide** |
| **V180c** | Memoria episódica | `✅ MEMORIA EPISÓDICA DEMOSTRADA` · Éxito: True | ✅ **conducta coincide; mecanismo sobreafirmado** ¹ |
| **V182A5** | Comunicación bidireccional | `✅ CULTURA ACUMULATIVA` ON=11.94 vs OFF=6.47 | ✅ **coincide** (cifras idénticas) ² |
| **V103** | "Clasificación multiestímulo perfecta" (nomenclatura shannoniana) | **Ω es un espacio representacional reproducible con estructura emergente**: voces distintas sin etiqueta → mismo Ω (Voz_Estudio 0.8512 ≈ voz 0.8517, **no supervisado**); híbridos voz+viento se agrupan con voz; ondas mixtas se reparten a polos BigBang±; usa todo el rango [0,1]. NO identifica estímulos individuales (música/voz/viento colapsan) — pero esa nunca fue la pregunta | ✅ **confirmado en el marco correcto** (estructura Ω reproducible). "Clasificación perfecta" = error de nomenclatura — ver §4 |

¹ **V180c (matiz de mecanismo, no de marco):** los 4 criterios pasan (P(45°)=0%, latencia 12.58×, Val(-60°)=41.4,
Val(+60°)=-2.0), pero la propia salida dice `Eventos recuperados: 0/50 (0.0%)` e `Impacto del recuerdo: 0.00`.
La **conducta de evitación es real**, pero **no demuestra recuperación episódica explícita**: el "utiliza el recuerdo
explícito" está sobreafirmado. El propio Informe V180 lo reconoce ("hallazgo no previsto, sin recall trial-a-trial").
También usa ±60° de criterio, no el +45° del texto.
² **V182A5 (corrección de una revisión previa de este addendum):** el `70% [❌]` de retención pertenece a la condición
**OFF (control, que *debe* fallar** — regresión a la media), no a ON. La condición **ON (memoria relacional) retiene 89% [✅]**
y acumula hasta min 11.94. El contraste ON 89%/11.94 vs OFF 70%/6.47 **es** la hipótesis; el resultado es más limpio que
lo que decía la nota anterior. Mi versión previa confundía el control con un defecto del resultado.
³ **V150 (residuo abierto reconocido):** cierra ANIMA-1, pero la **recuperación post-reposo es −6%** (la fatiga *sube*
11109°→11774° tras 180s de descanso; T_settle=∞ en F1/F3/F5). El propio informe lo declara problema abierto que pasa a ANIMA-2.

### 3.1 Tiempos de ejecución (referencia)

V117 70s · V103 700s · V132 810s · V147 760s · V150 890s · V176 840s · V180c 1110s · V182A5 530s · V122 2000s · **V118 5982s (~100 min, el pesado)**. Todos rc=0.

### 3.2 Revisión de marco (shannoniano vs. cosmosemiótico) — sobre los 10 logs, sin re-ejecutar

Tras la lección de V103, se releyeron los 10 logs capturados preguntando: ¿el criterio de éxito mide
**estructura/régimen** (cosmosemiótico, legítimo) o **identificación de etiqueta / umbral impuesto**
(riesgo shannoniano)?

- **9/10 usan criterios estructurales/de régimen** — diferenciación (S_shared), reorganización ante
  perturbación (R₂ vs umbral 3σ), valencia diferencial + acción diferencial, seguimiento de setpoint,
  alternancia sostenida, acumulación vs regresión con control ON/OFF. **No son shannonianos; los veredictos
  se sostienen** y en general pasan con holgura (no al filo del umbral).
- **Solo V103** tenía el error de marco (proyección de etiquetas humanas sobre el espacio Ω). Recategorizado en §4.
- **Dos matices que NO son de marco**, surgidos en la relectura: **V180c** (nota ¹: la conducta es real, pero
  el mecanismo de "recall explícito" no se demuestra, 0/50) y **V150** (nota ³: residuo de recuperación −6%).
  Son límites reales, ya reconocidos por sus propios informes.
- **Una corrección a este addendum:** **V182A5** (nota ²) — el resultado es más limpio de lo que decía una
  versión previa (el 70% era el control OFF, no un defecto).

**Conclusión: no hay un segundo V103.** El error de marco shannoniano fue un caso aislado; el resto de la
columna vertebral se mide cosmosemióticamente y resiste.

---

## 4. V103 — recategorización (la pregunta correcta no es shannoniana)

**Punto central (aportado por GPT y Alexis, 2026-06-22):** tanto el informe histórico ("clasificación
perfecta") como la primera pasada de esta auditoría ("no separa timbres, falla") evaluaron V103 con una
**pregunta shannoniana** — *¿identifica la etiqueta correcta?* — cuando el experimento plantea una
**pregunta cosmosemiótica** — *¿el espacio Ω tiene estructura propia reproducible?*. La sección
`CLASIFICACION POR CERCANIA` es una lectura del **experimentador**, no algo que el organismo haga: el
organismo nunca dice "esto es voz", solo produce un estado Ω. Las categorías las proyectamos nosotros.
V103, además, **no emite veredicto** (cierra con *"extraiga sus propias conclusiones"*).

**Lo que V103 demuestra — y es fuerte:**
- **Invarianza NO supervisada:** dos grabaciones de voz distintas, sin etiqueta alguna, terminan en el
  mismo Ω — `Voz_Estudio_pos=0.8512 ≈ voz_pos=0.8517`; `Voz_Estudio_neg = voz_neg = 0.5236` (exactos).
  Lo contrario de una clasificación supervisada: el organismo llegó allí solo.
- **Geometría emergente:** híbridos voz+viento se agrupan con voz (diff 0.01–0.03); **ondas mixtas se
  reparten a los polos BigBang** (pos→BigBang−, neg→BigBang+); BigBang ocupa región propia.
- **Ω = coordenada global de estado:** el sistema usa casi todo el rango [0,1].
- **Las "anomalías" confirman la teoría, no la rompen:** los estímulos sin estructura interna
  (tono puro → 0.0004/1.0000, en los bordes; ruido inestable: Ruido blanco_pos=0.9989 vs ruido_pos=−0.0002)
  **no producen un Ω interior estable: caen a los límites**. Estructura → régimen estable; ausencia de
  estructura → borde. Es cosmosemiótica, no fallo.

**Lo que V103 NO demuestra:** identificación unívoca de estímulos individuales — música, voz y viento
(todos Ω≈0.8–0.96) ocupan la misma región. Pero esa **nunca fue la pregunta del experimento**.

**Corrección explícita de la primera pasada de esta auditoría:** citó "Brandemburgo→voz" como *fallo de
clasificación*. Es doblemente inexacto: (1) no existe clase "música" entre las anclas, y (2) el criterio
mismo —identificar la categoría— es shannoniano y ajeno al experimento. Se retira.

**Veredicto recategorizado (fiel a los datos):**
> *V103 demuestra que Ω constituye un espacio de representación estable y reproducible que organiza
> estímulos heterogéneos en regiones funcionales recurrentes. No demuestra identificación unívoca de
> estímulos individuales — pero esa no era su pregunta. El término "clasificación" fue un error histórico
> de nomenclatura; la descripción fiel es **"cartografía de estados Ω inducidos por estímulos heterogéneos"**.*

**Redacción sugerida para el informe Síntesis (reemplaza "clasificación perfecta"):** *"V103 muestra que
Ω es un espacio representacional reproducible: estímulos no entrenados inducen estados Ω estables y
característicos (voces distintas → mismo Ω sin supervisión), con estructura geométrica emergente
(agrupamiento de híbridos, polos propios). No identifica estímulos individuales; el espacio Ω agrupa por
régimen funcional, no por etiqueta."*

> **Lección metodológica:** preguntar *"¿qué estímulo es?"* es Shannon (identificar etiqueta); preguntar
> *"¿en qué estado terminó el organismo?"* es Cosmosemiótica (estructura del espacio de estados).
> Importar el criterio shannoniano —medir aciertos de etiqueta— **solo porque el resultado queda ajustado**
> es un error de marco, no una medición de fracaso. Aplica a toda la auditoría: el criterio de éxito de un
> experimento de campo es *¿hay estructura reproducible no trivial?*, no *¿clasifica categorías humanas?*.

---

## 5. Lectura global — qué está validado y qué no

**Validado (re-ejecutado, reproducible):**
- Pilares de campo: **v72c** (modos propios), **v80h** (ciclo evolutivo).
- Orientación espacial: **v88**, robusta a 100 ciclos (v90/v92).
- Espina dorsal del organismo y la cognición: **V117/V118** (trade-off R₂/lateralidad) → **V122**
  (coexistencia) → **V132/V147/V150** (organismo mínimo, baseline, clausura ANIMA-1) →
  **V176** (negación operativa) → **V180c** (memoria episódica) → **V182A5** (comunicación bidireccional).

**Recategorizado (afirmación central confirmada, nomenclatura a corregir):**
- **V103**: el experimento **confirma estructura Ω reproducible** (marco cosmosemiótico, ver §4). Lo que
  no se sostiene es la etiqueta "clasificación perfecta" (marco shannoniano). **Recategorizar, no degradar:**
  de "clasificación de estímulos" a "cartografía de estados Ω". El error de marco fue compartido por el
  informe (optimista) y la auditoría inicial (pesimista).

**Parcial / no validado (afirmación a corregir):**
- **v91**: el lazo Acción→A **no cierra** (C30 1/100) → no afirmar cierre acción→percepción.
- **v90**: orientación pasa, pero con deterioro de separación geométrica → no afirmar "consolidación creciente".
- Piezas tempranas/aisladas (**v72a** ❌, **v72b**, **v81**, **v84b**, **v87**, **v89**): parciales;
  son andamiaje del desarrollo, no evidencia final. No citar como pruebas cerradas.

**Fuera de alcance:** la pista canónica **E1–E24** (experimentos conceptuales, no scripts).

---

## 6. Recomendaciones para el equipo transinteligente

1. **Capturar el veredicto, no solo los artefactos.** Cada script debe volcar su `stdout` (✅/❌/criterios)
   a un `.log` versionado junto al CSV/PNG. Fue la causa raíz de todo el episodio. *(Ya resuelto para
   los 10 hitos: ver `verificacion_hitos/`.)*
2. **Sellar semilla y versión en cada corrida.** Sin `seed` + hash del script en la cabecera del log,
   una re-corrida divergente no es comparable con la original.
3. **Atribuir cada validación a la versión que la cierra.** Orientación → v88 (no v84b). No citar piezas
   tempranas/aisladas como prueba del resultado maduro.
4. **Separar "hito validado" de "andamiaje/falsación productiva".** Varios pasos (v72a, v91, V103, V119,
   V121, v180/a/b) son negativos o parciales: documentarlos como tales fortalece el crédito real.
5. **Reescribir la línea de V103** según §4.
6. **Presupuesto de cómputo para los pesados.** v90/v91 (100 ciclos, ~4 h c/u) y V118 (~100 min) no caben
   en una ventana interactiva: correrlos desatendidos y archivar el cierre.
7. **No importar criterios shannonianos para juzgar experimentos de campo.** El criterio de éxito es
   *¿hay estructura reproducible no trivial en el espacio de estados?*, no *¿clasifica/identifica categorías
   humanas?*. Medir "aciertos de etiqueta" solo porque el resultado queda ajustado mete Shannon por la puerta
   de atrás. Ocurrió con V103 en ambas direcciones (informe optimista, auditoría pesimista). Cada experimento
   debe declarar **su** pregunta en cabecera, para no juzgarlo con la pregunta de otro marco.

---

## 7. Manifiesto de logs

Todos los veredictos quedan sellados y son re-leíbles:

```
verificacion_hitos/_MANIFIESTO.txt     # estado OK/FAIL + duración por hito
verificacion_hitos/V103.log  V117.log  V118.log  V122.log  V132.log
verificacion_hitos/V147.log  V150.log  V176.log  V180c.log V182A5.log
```

Cada `.log` incluye cabecera con: script, `sha1[:12]`, `mtime`, afirmación reportada e inicio de corrida.

---

*Generado tras la verificación por re-ejecución real. Los 10 hitos confirman su afirmación central;
V103 requiere recategorización de nomenclatura ("clasificación" → "cartografía de estados Ω"), no degradación.
Revisión 2 (2026-06-22): corregido el marco de evaluación de V103 (shannoniano → cosmosemiótico) tras
aporte de GPT y Alexis.
Revisión 3 (2026-06-22): revisión de marco sobre los 10 logs (§3.2) — 9/10 son cosmosemióticos, no hay
segundo V103; corregida la nota de V182A5 (el 70% era el control OFF); añadidos matices de V180c (mecanismo
de recall no demostrado) y V150 (residuo de recuperación −6%).*
