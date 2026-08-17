# ENFOQUE 5 — Energía · Exergía · Entropía (S = I·E) — BATERÍA EXHAUSTIVA
### 6 temas × 5 experimentos = 30 · nombre descriptivo · barridos deliberadamente sobredimensionados

**Director:** Alexis López Tapia · **Diseño:** Claude Science (CS) · **Fecha:** 24-jul-2026
**Ancla:** CONSOLIDADO_presupuesto_energia_multi_fuente.md (E_total≈2.7×10⁷¹ J; ordinaria 4.9%; materia 31.5%).
**Se suma a:** BATERIA_FUNDAMENTOS_F1_a_F4. Este es el Enfoque 5, exhaustivo.

Siglas: **E**=energía · **X**=exergía (energía útil, capaz de trabajo) · **S_ent**=entropía · **a**=factor de expansión · **ρ**=densidad.

---

## 0. LAS TRES REGLAS DE ORO DE ESTA BATERÍA

1. **Barrido sobredimensionado (regla del director):** cada parámetro se barre en un rango
   **mucho mayor** que donde suponemos que está la respuesta. Si esperamos ~5%, barremos
   de 0% a 100%. Si esperamos un umbral en r≈1, barremos r de 1e-3 a 1e3. **Nunca se
   centra el barrido en el resultado esperado** — eso ya sería medio-Shannon.
2. **El presupuesto observado (4.9% / 31.5% / 2.7×10⁷¹ J) es TEST DE SALIDA, JAMÁS
   entrada.** La eficiencia emerge del barrido; recién después se compara. Ajustar un
   coeficiente para acercarse al blanco = el 20.0 = experimento anulado.
3. **Dos axiomas de diseño, declarados como elección nuestra** (no como física real, que
   NO conserva E globalmente): **E1** = el presupuesto energético total se conserva (el
   guardián anti-Shannon más duro: nada emerge sin pagarse); **E2** = la expansión
   redistribuye E latente en exergía, no la crea (enfriamiento adiabático = esa conversión).

**Trampas prohibidas (filtro de cada experimento):** T0 nada discreto/dimensional a mano ·
T1 ningún número puesto a mano · T2 observable ≠ su juez · T3 juez congelado antes de
correr · T4 el NULL debe morder · T5 curva entera, no gate binario · T6 toda etapa puede
fallar (la conservación de E se verifica cada paso) · T7 barrido + perturbación dinámica,
nunca un punto ni solo semillas.

---

# TEMA 1 — PERSISTENCIA DE LA EXERGÍA
*¿Sobrevive la capacidad de hacer trabajo cuando el universo se expande?*

### E5.1-1 · "Supervivencia de exergía frente a la razón expansión/difusión, rango extremo"
- **Simple:** ¿la exergía sobrevive a la expansión, y a partir de qué r?
- **Barrido sobredim.:** r=H/D ∈ [1e-3 … 1e3] (log, ≥25 pts — 6 décadas, aunque esperamos el cruce cerca de 1) × ε ∈ [1e-12 … 1] × ≥16 semillas × ruido dinámico.
- **Observable:** X_final = fracción de E que puede hacer trabajo (desviación del equilibrio uniforme).
- **NULL:** permutar φ al final. **PASS:** X_final>0 con expansión, →0 sin ella y en NULL; ε=0→X=0.

### E5.1-2 · "Vida media de la exergía: ¿cuántos pasos tarda X en decaer sin expansión?"
- **Simple:** sin expansión, ¿cuán rápido muere la exergía?
- **Barrido:** difusividad D ∈ [1e-4 … 1e2] (6 décadas) × ε × ≥16 semillas; medir τ (tiempo a X/2).
- **Observable:** τ(D) — vida media de la exergía. **NULL:** — (caracterización). **PASS:** τ decrece monótono con D; se reporta la ley τ(D) entera.

### E5.1-3 · "Exergía persistente en 2D: ¿es artefacto del anillo 1D?"
- **Simple:** ¿lo de 1D pasa en 2D?
- **Barrido:** ε × r (rango extremo) × malla 2D L∈{32,64,128,256} × ≥8 semillas (el caro; de noche).
- **Observable:** X_final 2D. **NULL:** permutación 2D. **PASS:** mismo comportamiento cualitativo que 1D.

### E5.1-4 · "Umbral de exergía frente al ruido dinámico, barrido de 8 décadas"
- **Simple:** ¿cuánto ruido aguanta la exergía antes de disolverse?
- **Barrido:** amplitud de ruido dinámico ∈ [1e-8 … 1] (8 décadas) × r × ≥16 semillas.
- **Observable:** X_final(ruido). **NULL:** ruido con ε=0. **PASS:** curva X(ruido) entera; decaimiento suave, no salto.

### E5.1-5 · "Persistencia de exergía bajo expansión no monótona (historias H(t) variadas)"
- **Simple:** si la expansión acelera y frena, ¿la exergía aguanta?
- **Barrido:** 6+ perfiles H(t) (acelerante, frenante, ráfagas) × r efectivo × ≥12 semillas.
- **Observable:** X_final por perfil. **NULL:** barajado. **PASS:** X depende del r efectivo integrado, no del perfil; si un perfil rompe eso, se reporta.

---

# TEMA 2 — CONSERVACIÓN DEL PRESUPUESTO (LA CONTABILIDAD)
*¿Cuadra E_total = X + E_degradada en todo momento?*

### E5.2-1 · "Balance de energía paso a paso: deriva del total sobre corridas largas"
- **Simple:** ¿el total se mantiene o el motor fabrica/pierde energía?
- **Barrido:** pasos ∈ [1e2 … 1e5] (corridas muy largas) × ε × r × ≥12 semillas.
- **Observable:** |E_total(t)−E_total(0)|/E_total(0). **NULL:** —. **PASS:** deriva < 1e-6 en toda la corrida (T6: si no cuadra, FALLA).

### E5.2-2 · "Anticorrelación exergía↔entropía: ¿X baja exactamente lo que S_ent sube?"
- **Simple:** ¿cuando la exergía baja, la entropía sube en la misma cuenta?
- **Barrido:** ε × r (rango extremo) × ≥16 semillas.
- **Observable:** correlación temporal X(t) vs S_ent(t). **NULL:** barajado temporal. **PASS:** anticorrelación fuerte (r<−0.9) en REAL, ausente en NULL.

### E5.2-3 · "Conservación bajo forzamiento estocástico: ¿el ruido rompe el balance?"
- **Simple:** con ruido dinámico fuerte, ¿sigue cuadrando la cuenta?
- **Barrido:** amplitud de ruido ∈ [1e-6 … 1] (6 décadas) × pasos largos × ≥12 semillas.
- **Observable:** deriva de E_total vs amplitud de ruido. **NULL:** —. **PASS:** el balance se mantiene aun con ruido (el ruido redistribuye, no crea); si se rompe, se localiza dónde.

### E5.2-4 · "Presupuesto por componentes: ¿en qué se reparte E a lo largo del barrido?"
- **Simple:** ¿cuánta E queda útil, cuánta degradada, cuánta ligada — y cómo cambia con r?
- **Barrido:** r ∈ [1e-3 … 1e3] × ε × ≥12 semillas; descomponer E_total en {X, degradada, ligada}.
- **Observable:** las tres fracciones vs r. **NULL:** barajado. **PASS:** la suma = E_total (T6) y las fracciones varían con r de forma medible (curvas enteras).

### E5.2-5 · "Robustez del balance a la resolución: ¿depende del tamaño de paso/malla?"
- **Simple:** ¿la conservación es física o artefacto de discretización?
- **Barrido:** Δt ∈ [1e-4 … 1e-1] × N ∈ {128 … 2048} × ≥8 semillas.
- **Observable:** deriva de E_total vs Δt y N. **NULL:** —. **PASS:** la deriva → 0 al refinar (converge); si crece al refinar, el balance es numérico, no físico.

---

# TEMA 3 — EFICIENCIA DE CONVERSIÓN EMERGENTE ★ (el que ancla contra el 5%)
*De toda la energía, ¿qué fracción queda atrapada como estructura — y se parece al 5% SIN pedirlo?*

### E5.3-1 · "Eficiencia estructura/total barriendo ε de 12 décadas (0% a 100% posible)"
- **Simple:** ¿qué fracción de E queda ligada, y dónde cae en el rango 0–100%?
- **Barrido sobredim.:** ε ∈ [1e-12 … 1] × r ∈ [1e-3 … 1e3] × ≥16 semillas. La eficiencia puede salir en CUALQUIER punto de [0,1] — no se centra en 5%.
- **Observable:** E_ligada/E_total (SALIDA). **NULL:** barajado. **PASS:** curva de eficiencia entera; se anota si algún régimen cae cerca de 4.9%/31.5% SIN ajuste.

### E5.3-2 · "Eficiencia vs intensidad de ligadura, rango que cubre nula-a-total"
- **Simple:** al variar cuán fuerte liga la estructura, ¿cambia la fracción atrapada?
- **Barrido:** intensidad de ligadura ∈ [1e-3 … 1e2] (5 décadas) × ε × ≥12 semillas.
- **Observable:** eficiencia(ligadura). **NULL:** barajado. **PASS:** curva entera; ningún coef. tocado para acercar al 5%.

### E5.3-3 · "Estabilidad temporal de la eficiencia: ¿se congela o sigue cambiando?"
- **Simple:** la fracción atrapada, ¿queda fija (congelada) o deriva?
- **Barrido:** pasos ∈ [1e2 … 1e5] × ε × r × ≥12 semillas.
- **Observable:** eficiencia(t) — ¿mesetea? **NULL:** barajado. **PASS:** se reporta si hay congelamiento y a qué paso; sin fijar el valor.

### E5.3-4 · "Sensibilidad de la eficiencia a los dos axiomas (E1 on/off, E2 on/off)"
- **Simple:** ¿la eficiencia depende de imponer conservación / redistribución?
- **Barrido:** {E1 on/off} × {E2 on/off} × ε × r × ≥12 semillas.
- **Observable:** eficiencia por combinación. **NULL:** barajado. **PASS:** se reporta cuánto mueve cada axioma la eficiencia (mide cuán load-bearing es cada supuesto).

### E5.3-5 · "Test de falsación externo: distancia emergente al 4.9%/31.5%, sin ajuste"
- **Simple:** ¿la eficiencia emergente cae cerca de los valores observados, o lejos?
- **Barrido:** el grid completo de E5.3-1 a E5.3-2, agregando la distancia |eficiencia_emergente − 0.049| y |− 0.315|.
- **Observable:** distribución de la eficiencia emergente contra los dos blancos. **NULL:** barajado. **PASS (tres lecturas):** cae cerca sin ajuste → hallazgo fuerte; cae en otro valor estable → dato honesto; no converge → negativo. **Prohibido mover coeficientes hacia el blanco.**

---

# TEMA 4 — EXERGÍA Y ENFRIAMIENTO ADIABÁTICO
*¿Es la expansión (vía enfriamiento) la que convierte energía latente en exergía útil?*

### E5.4-1 · "Producción de exergía vs enfriamiento medido, expansión de 1 a 1e4"
- **Simple:** ¿la exergía aparece al enfriarse por expandirse?
- **Barrido:** a ∈ [1 … 1e4] (log) × ε × ≥12 semillas; T(a) medida, no impuesta.
- **Observable:** X producida vs caída de T. **NULL:** sin expansión (no debe producir X). **PASS:** X correlaciona con enfriamiento en REAL, nula sin expansión.

### E5.4-2 · "Exponente de enfriamiento emergente: ¿T∝a^−n, con qué n?"
- **Simple:** ¿con qué ley baja la temperatura al expandir?
- **Barrido:** a ∈ [1 … 1e6] (6 décadas) × ε × ≥12 semillas.
- **Observable:** n medido en T∝a^−n (SALIDA). **NULL:** sin expansión. **PASS:** n emerge y se reporta; NO se fija a n=2 ni n=3 (aunque la física los sugiera).

### E5.4-3 · "Reversibilidad: si se detiene la expansión, ¿la exergía se re-degrada?"
- **Simple:** parando la expansión a mitad, ¿la difusión mata la exergía ganada?
- **Barrido:** tiempo de parada ∈ ≥10 puntos a lo largo de la corrida × ε × ≥12 semillas.
- **Observable:** X tras parar vs seguir. **NULL:** nunca parar. **PASS:** curva "re-degradación vs tiempo de parada"; existe (o no) un punto de no-retorno.

### E5.4-4 · "Exergía por escalas espectrales: ¿qué longitudes de onda la retienen?"
- **Simple:** ¿la exergía se congela antes en estructuras grandes o chicas?
- **Barrido:** a × banda espectral completa × ≥12 semillas.
- **Observable:** exergía por escala vs a. **NULL:** densidad fija. **PASS:** espectro de retención reportado; se compara con "escalas grandes primero" sin imponerlo.

### E5.4-5 · "Control negativo: enfriamiento con baño externo (lo prohibido) vs adiabático"
- **Simple:** ¿nuestro enfriamiento es adiabático de verdad, o un baño térmico oculto?
- **Barrido:** acople a baño externo ∈ [0 … fuerte] (≥8 pts) × a × ≥12 semillas.
- **Observable:** X y T(a) con baño vs sin baño. **NULL:** —. **PASS:** el caso adiabático (acople=0) difiere claramente del baño (T→T_baño); confirma que no hay baño encubierto.

---

# TEMA 5 — MUERTE TÉRMICA vs NADA (el límite ε→0)
*¿El equilibrio tiene toda la energía pero cero exergía — E=1, X=0, distinto de la Nada ∅?*

### E5.5-1 · "Barrido fino de ε→0: curvas E, X, S_ent en el límite"
- **Simple:** al desaparecer la diferencia, ¿qué pasa con las tres cantidades?
- **Barrido:** ε ∈ [0 … 1e-2] MUY fino (≥20 pts, incluye 0 estricto) × ≥16 semillas.
- **Observable:** E(ε), X(ε), S_ent(ε). **NULL:** —. **PASS:** en ε→0, E≈total (constante), X→0, S_ent→máx. Curvas enteras.

### E5.5-2 · "Tiempo a la muerte térmica: ¿cuánto tarda X→0 según ε y r?"
- **Simple:** ¿cuánto vive el universo-modelo antes del equilibrio, según ε?
- **Barrido:** ε ∈ [1e-9 … 1] × r ∈ [1e-3 … 1] (sin expansión suficiente) × ≥12 semillas.
- **Observable:** tiempo a X<umbral. **NULL:** —. **PASS:** se reporta t_muerte(ε,r); diverge cuando r cruza el umbral de congelamiento.

### E5.5-3 · "Reversibilidad de la muerte térmica: ¿re-inyectar ε la revierte?"
- **Simple:** una vez en equilibrio, ¿una nueva diferencia revive la exergía o ya no?
- **Barrido:** momento de re-inyección de ε × amplitud re-inyectada ∈ [1e-6 … 1] × ≥12 semillas.
- **Observable:** X recuperada. **NULL:** sin re-inyección. **PASS:** se mide si el equilibrio es recuperable o absorbente (informa la naturaleza de la muerte térmica).

### E5.5-4 · "Muerte térmica vs Nada operativa: caracterización de los dos estados"
- **Simple:** confirmar E=máx/X=0 (muerte térmica) como estado CON energía, distinto de ∅.
- **Barrido:** ε→0 × ≥16 semillas; medir E, X, S_ent y capacidad de trabajo residual.
- **Observable:** vector (E, X, S_ent) en el límite. **NULL:** —. **PASS:** E>0 y X=0 y S_ent=máx simultáneamente — el estado tiene energía pero no puede hacer nada (la distinción del poema, empírica).

### E5.5-5 · "Universalidad del límite: ¿todos los ε→0 llegan al mismo estado?"
- **Simple:** distintas formas iniciales de diferencia, ¿mueren todas en el mismo equilibrio?
- **Barrido:** 6+ familias de forma inicial × ε→0 × ≥12 semillas.
- **Observable:** estado final por familia. **NULL:** —. **PASS:** convergencia al mismo (E,X,S_ent) independiente de la forma (o se reporta la dispersión).

---

# TEMA 6 — DEFINICIÓN Y VERIFICACIÓN CRUZADA DE LA EXERGÍA
*¿La exergía es una cantidad robusta, medible por vías independientes que coinciden?*

### E5.6-1 · "Doble medida: exergía termodinámica vs informacional, mismo barrido"
- **Simple:** ¿dos definiciones distintas de exergía dan lo mismo?
- **Barrido:** ε × r (rango extremo) × ≥16 semillas; medir X_termo (desviación del equilibrio) y X_info (estructura espacial, ligada a la I).
- **Observable:** correlación X_termo vs X_info. **NULL:** barajado (ambas deben caer). **PASS:** coinciden (corr>0.9); la discrepancia = medida de robustez.

### E5.6-2 · "Exergía como energía libre: ¿se comporta como una energía libre real?"
- **Simple:** ¿la exergía sigue las propiedades termodinámicas esperadas (F=E−T·S)?
- **Barrido:** T efectiva (de la expansión) × ε × ≥12 semillas.
- **Observable:** X vs (E − T·S_ent) medidos por separado. **NULL:** —. **PASS:** X ≈ E − T·S_ent dentro de tolerancia (verifica que la construcción es termodinámicamente coherente).

### E5.6-3 · "Invariancia de X a la escala del sistema (N barrido amplio)"
- **Simple:** ¿la exergía por unidad es la misma a distinto tamaño, o es efecto de borde?
- **Barrido:** N ∈ [64 … 4096] (6 duplicaciones) × ε × r × ≥8 semillas.
- **Observable:** X/N vs N. **NULL:** barajado. **PASS:** X/N estable con N (intensiva) o se reporta la dependencia.

### E5.6-4 · "Sensibilidad de X a la definición de equilibrio de referencia"
- **Simple:** la exergía se mide contra un "equilibrio" — ¿cambia si movemos esa referencia?
- **Barrido:** distintas referencias de equilibrio (media global, local, móvil) × ε × ≥12 semillas.
- **Observable:** X según referencia. **NULL:** —. **PASS:** el veredicto (persiste/no) es invariante a la referencia razonable; si depende, se reporta cuál y por qué.

### E5.6-5 · "Exergía informacional y la I de S=I·E: ¿la relación es medible?"
- **Simple:** ¿la exergía informacional se conecta con la I (información/estructura) de la ley central?
- **Barrido:** ε × r × ≥12 semillas; medir X_info y una medida independiente de I (entropía estructural).
- **Observable:** relación X_info ↔ I. **NULL:** barajado. **PASS:** se reporta la relación empírica entre exergía informacional e I, contra NULL — sin forzar que sea S=I·E, solo midiendo qué sale.

---

## REGLAS DE EJECUCIÓN (CC/Grok firman antes de correr)

1. **Pre-registro fechado por experimento** (observable, NULL, PASS con umbral, rangos,
   semillas, axiomas E1/E2). Si falla, se reporta — no se edita (T3).
2. **Barrido sobredimensionado siempre** (regla del director): rango ≫ resultado esperado.
3. **Perturbación dinámica además de semilla** — nunca "rate 10/10" cosmético (T7).
4. **Tres verificaciones por experimento:** su NULL + segundo observable/método + auditoría
   en disco por quien no lo escribió.
5. **La cantidad medida ≠ su juez** (T2). **Conservación de E verificada cada paso** (T6).
6. **Presupuesto observado = test de salida, NUNCA entrada.** Ningún coef. hacia el blanco.
7. **No cambiar código tras revisión de CS.** Error visto → PARAR y reportar línea exacta.
8. **Ejecutar completo.** Cómputo largo autorizado — que tarde lo que tarde.
9. **Entregar crudo a CS:** curvas completas + dispersión entre semillas Y perturbaciones.
   NO adjudicar.

## ORDEN SUGERIDO

- **Tema 2** primero (la contabilidad — sin balance que cuadre, nada vale).
- **Tema 1** (persistencia de exergía — reinterpreta el núcleo).
- **Tema 5** (límite ε→0 — caracteriza la muerte térmica vs Nada).
- **Tema 3 ★** (eficiencia emergente — el ancla contra el 5%).
- **Tema 4** (exergía↔enfriamiento — ata con Enfoque 3).
- **Tema 6** (definición robusta de X — verificación cruzada del observable central).

**Nota final:** cualquier NEGATIVO es hallazgo, no fracaso. El resultado más fuerte no es
"dio 5%" — es "la eficiencia emergió sola, sin ponerla, y resultó cercana al 5%".
Barremos rangos enormes precisamente para que, si algo aparece cerca del blanco, no se
pueda decir que miramos solo donde convenía. Esa es la diferencia entre ciencia y el 20.0.
