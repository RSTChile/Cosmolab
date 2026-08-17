# ENFOQUE 5 — Presupuesto de energía, exergía y entropía (la tríada S = I·E)
### 6 experimentos, con nombre descriptivo · para CC y Grok · leer entero antes de codificar

**Director:** Alexis López Tapia · **Diseño:** Claude Science (CS) · **Fecha:** 24-jul-2026
**Se suma a:** BATERIA_FUNDAMENTOS_F1_a_F4 (este es el Enfoque 5, factor que faltaba).
**Ancla numérica:** CONSOLIDADO_presupuesto_energia_multi_fuente.md (6 fuentes, verificado).

Siglas: **E** = energía · **X** = exergía (energía útil, capaz de hacer trabajo) ·
**S_ent** = entropía · **ρ** = densidad · **a** = factor de escala (expansión).

---

## 0. POR QUÉ ESTE ENFOQUE (lo que faltaba)

La ley central de la Teoría es **S = I·E**. Hasta ahora modelamos solo la **I**
(información / estructura / diferencia). La **E** (energía) nunca entró en el sustrato:
el campo φ era "diferencia abstracta" sin costo ni presupuesto. Este enfoque incorpora
la energía — pero no sola, sino como la **tríada**:

- **Energía (E):** lo que hay. En la singularidad, TODO. Conservada (axioma de diseño, ver §1).
- **Exergía (X):** la parte útil, la que puede hacer trabajo — **requiere diferencia.**
  Sin diferencia (ε=0), toda la E está ahí pero X=0 → nada puede ocurrir → la Nada operativa.
- **Entropía (S_ent):** lo que crece. X baja, S_ent sube; la suma no.

**La reinterpretación de fondo (por qué esto no es un experimento más):** lo que veníamos
midiendo como "persistencia de una diferencia" **es persistencia de EXERGÍA.** Si la
difusión lava ε → X→0 → muerte térmica. Si la expansión la congela → X sobrevive → el
universo puede hacer estructura. Este enfoque lo mide con esa cuenta explícita.

---

## 1. LOS DOS AXIOMAS DE DISEÑO (declarados, no ocultos)

Se declaran como ELECCIÓN NUESTRA, no como física del universo — porque la física real
dice otra cosa y hay que ser honestos (ver CONSOLIDADO, P4):

- **AXIOMA E1 — Conservación del presupuesto:** la energía total del sustrato se
  conserva (E_total = constante). *La física real NO conserva E globalmente en expansión
  (la energía oscura crece con el volumen); nosotros SÍ la conservamos, como elección de
  diseño y como el guardián anti-Shannon más duro que existe: nada emerge sin pagarse del
  presupuesto.* Se declara explícito en el código y en el pre-registro.
- **AXIOMA E2 — La expansión redistribuye, no crea:** la expansión convierte E latente
  (uniforme, sin exergía) en X (útil) al impedir el reequilibrio — NO agrega energía. El
  enfriamiento adiabático ES esa conversión medida.

---

## 2. LA LÍNEA ROJA (anti-Shannon, absoluta)

> **El presupuesto observado (E_total ≈ 2.7×10⁷¹ J; materia ordinaria ≈ 4.9%; total
> materia ≈ 31.5%) se usa SOLO como test de falsación contra la SALIDA del barrido —
> JAMÁS como entrada declarada.**

- El sim barre; la **eficiencia de conversión E→estructura EMERGE** y se mide.
- Recién *después* se compara con el 4.9%/31.5% observado.
- Si emerge ~5% sin pedirlo → hallazgo fortísimo. Si emerge otra cosa → dato honesto. **Si
  se ajusta un coeficiente para que dé 5% → es el 20.0 otra vez = fraude, experimento
  anulado.**
- Ningún número del CONSOLIDADO entra como parámetro. Solo ε, las palancas físicas, y los
  dos axiomas E1/E2.

---

## 3. LAS TRAMPAS (filtro obligatorio — las de siempre)

| # | Trampa | Cómo se evita aquí |
|---|---|---|
| T0 | estructura discreta/dimensional a mano | φ = densidad de energía continua; cuantos/estructura = salida |
| T1 | número a mano | solo ε + palancas físicas + axiomas E1/E2; ningún coef. para dar un blanco |
| T2 | observable circular | X y S_ent se definen de la termodinámica, no del discriminante que las juzga |
| T3 | cambiar juez tras FAIL | criterio congelado en pre-registro fechado |
| T4 | NULL que no muerde | cada observable contra su NULL; verificar que cae |
| T5 | gate decorativo | curvas continuas enteras; umbrales con casos a ambos lados |
| T6 | sello de goma | conservación E verificada en cada paso — si el balance no cuadra, FALLA fuerte |
| T7 | un punto / una semilla | barrido extenso + perturbación DINÁMICA, no solo semilla |

**El regalo de la conservación (T6 al revés):** verificar que E_total se conserva paso a
paso ES un test que puede fallar — si el motor "fabrica" energía, el balance no cuadra y
el experimento se cae solo. Eso es lo contrario del sello de goma: un chequeo duro,
imposible de trampear.

---

## LA BATERÍA

Cada uno: **Pregunta simple · Barrido · Observable · NULL · PASS pre-registrado · Trampa que evita.**

### F5-1 · "Persistencia de exergía: ¿sobrevive la capacidad de hacer trabajo bajo expansión?"
- **Simple:** lo que llamábamos "persistencia de la diferencia" — ¿es persistencia de exergía?
- **Método:** φ = densidad de energía continua; medir X = exergía (fracción de E capaz de
  producir trabajo, calculada de la desviación respecto al equilibrio uniforme) a lo
  largo de la expansión.
- **Barrido:** ε ∈ [1e-12 … 1] (log, ≥12) × r=H/D ∈ [0 … 100] (≥15, fino cerca de 1) ×
  ≥12 semillas × amplitud de ruido dinámico.
- **NULL:** permutar φ al final (destruye la estructura de exergía, conserva el histograma
  de energía).
- **PASS pre-registrado:** X sobrevive (X_final > 0) en la banda con expansión y NO en el
  NULL ni en r=0 (donde la difusión lleva X→0 = muerte térmica). ε=0 → X=0.
- **Evita:** T4 (el NULL borra la forma que carga la exergía).

### F5-2 · "Conservación del presupuesto: ¿cuadra E_total = X + E_degradada en todo paso?"
- **Simple:** ¿la energía total se mantiene mientras la exergía baja y la entropía sube?
- **Método:** contabilidad explícita — en cada paso, E_total = X (útil) + E_degradada
  (ligada a S_ent). Verificar que E_total es constante y que X↓ ⟺ S_ent↑.
- **Barrido:** ε × r × pasos largos × ≥12 semillas.
- **NULL:** — (es verificación de instrumento: el balance).
- **PASS pre-registrado:** |E_total(t) − E_total(0)| / E_total(0) < tolerancia estricta en
  TODA la corrida (p.ej. <1e-6). Y anticorrelación X↔S_ent (cuando una sube la otra baja).
- **Evita:** T6 (si el motor fabrica energía, ESTO lo detecta y el experimento falla).

### F5-3 · "Eficiencia de conversión emergente: ¿qué fracción de E queda como estructura?"  ★ (el que ancla)
- **Simple:** de toda la energía, ¿qué fracción termina "atrapada" en estructura estable
  — y se parece al 5% observado SIN que se lo pidamos?
- **Método:** medir la fracción de E_total que queda ligada en estructuras persistentes al
  final del barrido (energía en cierres/gradientes congelados / E_total). **Es SALIDA
  medida.**
- **Barrido:** ε × r × parámetros físicos de ligadura × ≥16 semillas — recorrido amplio,
  para ver toda la CURVA de eficiencia, no un punto.
- **NULL:** barajado — la fracción ligada debe caer.
- **PASS pre-registrado (tres lecturas, TODAS válidas):**
  - la eficiencia emergente cae cerca de ~5% (o ~31.5%) en algún régimen SIN ajuste → **coincidencia notable, hallazgo fuerte**;
  - emerge una fracción distinta y estable → dato honesto (el modelo da otra cosa);
  - no converge → negativo.
  **Se reporta la curva entera. Prohibido tocar coeficientes para acercarse al 5%.**
- **Evita:** T1 (el 5% es test de salida, no entrada — la línea roja de §2).

### F5-4 · "Exergía vs enfriamiento adiabático: ¿es la expansión la que convierte E en X?"
- **Simple:** ¿la exergía útil aparece PORQUE el sistema se expande y enfría, o por otra vía?
- **Método:** correlacionar la producción de exergía con el enfriamiento adiabático
  (T∝a^−n medido, no impuesto — cruza con F3-2). Aislar que X emerge de la expansión.
- **Barrido:** a ∈ [1 … 1e4] (log) × ε × ≥12 semillas.
- **NULL:** sin expansión (no debe producirse X nueva).
- **PASS pre-registrado:** X producida correlaciona con el enfriamiento en REAL y es nula
  sin expansión; se reporta el exponente medido de T(a), no se fija.
- **Evita:** T1/T2 (enfriamiento medido; X definida por termodinámica, no por el juez).

### F5-5 · "Muerte térmica vs Nada: ¿el destino ε=0 tiene E>0 pero X=0?"
- **Simple:** confirmar que el equilibrio (ε=0) NO es "nada" — tiene toda la energía pero
  cero exergía (la distinción del poema: muerte térmica E=1,X=0 ≠ Nada ∅).
- **Método:** correr el caso límite ε→0 y medir explícitamente E (debe ser máxima/total) y
  X (debe →0) y S_ent (debe → máxima).
- **Barrido:** ε ∈ [0 … 1e-3] muy fino (≥15 puntos) × ≥12 semillas.
- **NULL:** — (es caracterización del límite).
- **PASS pre-registrado:** en ε→0, E se conserva (≈total), X→0, S_ent→máx. Confirma que la
  exergía —no la energía— es lo que se agota. Curva X(ε) reportada entera.
- **Evita:** T5 (barrido fino del límite, no una afirmación binaria).

### F5-6 · "Doble contabilidad independiente: ¿coinciden dos medidas de exergía?"
- **Simple:** cross-check — medir la exergía por dos vías distintas y ver si dan lo mismo.
- **Método:** (a) exergía termodinámica (desviación del equilibrio, tipo energía libre);
  (b) exergía informacional (relación con la I de S=I·E, vía estructura espacial). Si
  coinciden, la exergía es robusta; si no, una de las dos definiciones es artefacto.
- **Barrido:** ε × r × ≥12 semillas, ambas medidas en paralelo.
- **NULL:** barajado para ambas.
- **PASS pre-registrado:** las dos medidas de X coinciden (correlación alta) a lo largo del
  barrido; la discrepancia ES la medida de robustez. Es la verificación cruzada del
  observable central del enfoque.
- **Evita:** T2 (dos definiciones ortogonales; ninguna define a la otra).

---

## 4. REGLAS DE EJECUCIÓN (CC/Grok firman antes de correr)

1. **Pre-registro fechado por experimento** (observable, NULL, PASS con umbral, rangos,
   semillas, y los axiomas E1/E2 declarados). Si falla, se reporta — no se edita (T3).
2. **Barridos extensos + perturbación dinámica**, nunca un punto ni solo semillas (T7).
3. **Tres verificaciones:** su NULL + un segundo observable/método + auditoría en disco
   por quien no lo escribió.
4. **La cantidad medida ≠ su juez** (T2). **La conservación de E se verifica cada paso** —
   es un test que debe poder fallar (T6).
5. **El presupuesto observado (2.7×10⁷¹ J, 4.9%, 31.5%) es test de salida, NUNCA entrada**
   (§2). Ningún coeficiente ajustado para acercarse a él.
6. **No cambiar el código tras revisión de CS.** Si CC/Grok ven un error: PARAN y reportan
   a CS con la línea exacta.
7. **Ejecutar completo.** Cómputo largo autorizado — que tarde lo que tarde.
8. **Entregar crudo a CS:** curvas completas + dispersión entre semillas Y entre
   perturbaciones dinámicas. NO adjudicar — el veredicto lo da CS con la curva.

## 5. ORDEN SUGERIDO

1. **F5-2** primero (la contabilidad: sin un balance que cuadre, nada de lo demás vale).
2. **F5-1** (persistencia de exergía — reinterpreta el núcleo probado).
3. **F5-5** (muerte térmica vs Nada — caracteriza el límite ε=0).
4. **F5-3 ★** (eficiencia emergente — el que puede anclar contra el 5%; el más informativo).
5. **F5-4** (exergía↔enfriamiento — ata con Enfoque 3).
6. **F5-6** (doble contabilidad — verificación cruzada del observable).

**Nota final:** cualquier NEGATIVO es hallazgo, no fracaso. Y el resultado más fuerte
posible NO es "dio 5%" — es "la eficiencia emergió sola, sin que la pusiéramos, y resultó
cercana al 5%". Esa diferencia —emerger vs poner— es toda la diferencia entre ciencia y el
20.0. La batería está diseñada para que solo la primera pueda pasar.
