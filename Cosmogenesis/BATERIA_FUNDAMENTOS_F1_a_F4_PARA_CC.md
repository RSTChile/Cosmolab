# BATERÍA DE FUNDAMENTOS — Enfoques 1, 2 y 3 (campo · expansión · enfriamiento)
### 18 experimentos (6 por enfoque), múltiples métodos, verificación cruzada, barridos extensos

**Director:** Alexis López Tapia · **Diseño:** Claude Science (CS) · **Fecha:** 24-jul-2026
**Para:** CC (y Grok si el director lo asigna). Leer entero antes de tocar código.

---

## ⚠ INTERPRETACIÓN DE "1, 2 y 3" (corregir aquí si me equivoqué)

Los enfoques son tus factores de la singularidad. **Corrección del director (24-jul):
la densidad NO se deja fuera** — el campo pierde densidad al expandirse (ρ∝a⁻³), así que
es un factor físico real, no separable. Se incluye como Enfoque 4.

- **ENFOQUE 1 — Campo caliente + asimetría puntual ε:** ¿persiste una diferencia ínfima?
- **ENFOQUE 2 — Expansión supralumínica (r = expansión/reabsorción):** ¿congela la diferencia?
- **ENFOQUE 3 — Gradientes estirados + enfriamiento adiabático:** ¿enfriar es expandir, sin baño externo?
- **ENFOQUE 4 — Descenso de densidad por expansión (ρ∝a⁻³):** ¿es la densidad una palanca causal propia, o solo expansión con otro nombre?

**24 experimentos, 6 por enfoque.** Si querías otra cosa (p.ej. las emergencias de masa
①②), pará aquí y corrígeme antes de que CC ejecute.

---

## 0. POR QUÉ ESTA BATERÍA

La rama de masa (CF-4/4b) chocó con el muro del campo: cinco observables de masa no
dieron señal limpia. Antes de seguir ahí, **blindamos el fundamento físico que SÍ
funciona** — los tres factores basales — pero con el rigor que hasta ahora faltó:
múltiples métodos (no un solo observable), verificación cruzada (no una sola semilla),
y barridos extensos (no un punto). El director autorizó cómputo largo: **cada
experimento puede tardar toda la noche; no se ahorra cómputo.**

---

## 1. LAS TRAMPAS (prohibido repetirlas — filtro obligatorio de cada experimento)

| # | Trampa | Cómo se evita en TODA la batería |
|---|---|---|
| T0 | estructura discreta/dimensional a mano | sustrato = campo continuo; cuantos/dimensión = salida medida |
| T1 | número puesto a mano | ningún coeficiente elegido para dar un resultado; solo ε y palancas físicas barridas |
| T2 | observable circular | la cantidad medida ≠ las variables que la juzgan |
| T3 | cambiar el juez tras el FAIL | criterio de PASS congelado en pre-registro fechado |
| T4 | NULL que no muerde | verificar que el NULL cae; si REAL=NULL, reportarlo |
| T5 | gate decorativo | umbrales con casos a ambos lados; curva entera reportada |
| T6 | sello de goma | toda etapa debe poder FALLAR |
| T7 | un punto / una semilla | barrido de rango + perturbación DINÁMICA (no solo semilla) |

**Lección específica de CF-2 (grabar):** en una ecuación en derivadas parciales casi
determinista, muchas semillas dan el mismo número → "rate 10/10" NO es robustez.
**Robustez = perturbar la dinámica** (amplitud de ruido, parámetro físico, condición
inicial), no la semilla. Cada experimento de esta batería incluye esa perturbación.

**Múltiples sistemas de verificación (regla nueva, obligatoria):** cada experimento se
valida por TRES vías independientes — (a) su NULL, (b) un **segundo observable o método
distinto** que debería dar el mismo veredicto, y (c) auditoría en disco por quien NO
escribió el código (código + JSON crudo, no de palabra).

---

## ENFOQUE 1 — ¿PERSISTE UNA DIFERENCIA ÍNFIMA EN UN CAMPO CONTINUO CALIENTE?

Núcleo ya probado (CS074/CF-1); aquí se blinda con métodos independientes.

### F1-1 · "Persistencia por autocorrelación de forma contra NULL barajado"
- **Simple:** ¿la forma espacial de la diferencia sobrevive y le gana al azar?
- **Método:** observable forma×magnitud (autocorrelación a primer vecino × varianza).
- **Barrido extenso:** ε ∈ [1e-12 … 1] (log, ≥12 puntos) × r=H/D ∈ [0 … 100] (≥15 puntos, FINO cerca de r≈1) × N ∈ {200, 400, 800, 1600} × ≥12 semillas.
- **NULL:** permutar el campo al final (destruye forma, conserva histograma).
- **Verificación cruzada:** (a) NULL cae a ~0; (b) control ε=0 → P=0; (c) auditoría en disco.
- **PASS pre-registrado:** P_real ≫ P_null en la banda con expansión; ε=0 y r=0 dan P≈0.
- **Evita:** T4 (el observable hace morder al NULL).

### F1-2 · "Persistencia por información mutua espacial (observable independiente)"
- **Simple:** ¿la persistencia aparece también con un medidor completamente distinto?
- **Método:** en vez de autocorrelación, **información mutua** entre mitades del dominio / entropía espacial de bloques. Si persiste con un estimador no relacionado, no es artefacto del observable.
- **Barrido:** mismo grid de ε y r que F1-1; ≥12 semillas.
- **NULL:** permutación (la info mutua espacial debe colapsar al barajar).
- **Verificación cruzada:** el veredicto debe COINCIDIR con F1-1 punto por punto (ésa es la verificación: dos observables, mismo mapa). Si difieren, uno es artefacto.
- **PASS:** el mapa de persistencia (ε, r) de F1-2 coincide con el de F1-1 dentro de tolerancia.
- **Evita:** T2/T4 (dos observables ortogonales; ninguno define al otro).

### F1-3 · "Umbral de amplitud: ¿existe un ε mínimo bajo el cual nada persiste?"
- **Simple:** ¿basta cualquier diferencia por ínfima que sea, o hay un piso?
- **Método:** barrer ε en MUCHAS décadas finamente (1e-15 … 1e-1, ≥20 puntos log), a r fijo en régimen de congelamiento, buscando un umbral o confirmando que es scale-free.
- **Barrido:** ε (20+ puntos) × ≥16 semillas × perturbación de ruido dinámico.
- **NULL:** ε=0 estricto (debe dar P=0).
- **Verificación cruzada:** (a) NULL; (b) repetir con el observable de F1-2; (c) disco.
- **PASS pre-registrado (tres lecturas):** P(ε) plano hasta ε→0 = "S>0 basta" (scale-free); P(ε) con umbral = hay piso mínimo; ambos son hallazgo, se reporta la curva.
- **Evita:** T1 (no se fija ε; se recorre).

### F1-4 · "Independencia de la forma inicial: barrido de familias de perturbación"
- **Simple:** ¿el resultado depende de la forma de mancha que sembramos (arbitraria)?
- **Método:** sembrar la diferencia con **familias distintas** — un solo modo de Fourier, multi-modo, bulto localizado gaussiano, campo aleatorio con distintos espectros de potencia (blanco, rojo, azul). Si el veredicto es el mismo, la forma inicial no manda.
- **Barrido:** 5+ familias × ε × r × ≥12 semillas.
- **NULL:** permutación por familia.
- **Verificación cruzada:** el mapa (ε, r) debe ser invariante a la familia. La dispersión entre familias ES la medida de robustez.
- **PASS:** persistencia presente en TODAS las familias; si una familia difiere, se reporta (D1 vivo).
- **Evita:** T0/T1 (la forma no se privilegia).

### F1-5 · "Robustez frente a ruido dinámico (no cosmético de semilla)"
- **Simple:** ¿la persistencia sobrevive si el campo tiene ruido real durante la dinámica, no solo al inicio?
- **Método:** añadir forzamiento estocástico de amplitud variable EN CADA PASO (no solo condición inicial). Esta es la prueba que CF-1/CF-2 no tenían.
- **Barrido:** amplitud de ruido dinámico ∈ [1e-6 … 1e-1] (≥8 puntos log) × ε × r × ≥16 semillas.
- **NULL:** permutación; y control con ruido pero ε=0.
- **Verificación cruzada:** (a) NULL; (b) que la persistencia decaiga suavemente con el ruido (no salto artificial); (c) disco.
- **PASS pre-registrado:** existe un rango de ruido bajo el cual la persistencia se mantiene; se reporta la curva P(amplitud_ruido) entera.
- **Evita:** T7 (perturba la DINÁMICA — la robustez real).

### F1-6 · "Persistencia en 2D: ¿es un artefacto del anillo 1D?"
- **Simple:** ¿lo que vimos en 1D pasa también en 2D?
- **Método:** repetir F1-1 en malla 2D (mismo observable adaptado). Robustez dimensional.
- **Barrido:** ε × r × tamaños 2D (L=32,64,128) × ≥8 semillas. (El caro; corre de noche.)
- **NULL:** permutación 2D.
- **Verificación cruzada:** el umbral en r debe caer en el mismo régimen que 1D.
- **PASS:** persistencia presente en 2D con el mismo comportamiento cualitativo que 1D.
- **Evita:** T0 (dimensión no impuesta; se prueba en otra geometría).

---

## ENFOQUE 2 — ¿LA EXPANSIÓN CONGELA LA DIFERENCIA? (competencia expansión vs reabsorción)

Núcleo probado (CS074-rcruz); aquí se resuelve el r* fino y se aísla el mecanismo.

### F2-1 · "Umbral de congelamiento: barrido fino de r cruzando 1, con r* resuelto"
- **Simple:** ¿dónde exactamente enciende el congelamiento al subir la expansión?
- **Método:** grid MUY FINO de r alrededor del cruce (0.01 … 3, ≥25 puntos), para resolver el r* continuo que rcruz dejó en "mismo bin".
- **Barrido:** r fino × N ∈ {200,400,800,1600} × ≥16 semillas × perturbación dinámica.
- **NULL:** barajado del acople al final.
- **Verificación cruzada:** (a) NULL cae; (b) r* estable al subir N (ley de escala r*(N)); (c) disco.
- **PASS pre-registrado:** curva P(r) con transición resuelta; se reporta r* y su dependencia de N (no se afirma "invariante" sin la curva — lección del hallazgo previo).
- **Evita:** T5/T7 (curva fina entera, no un bin).

### F2-2 · "D emergente multi-paso: ¿el r* nominal es el crítico real?"
- **Simple:** medimos la reabsorción (D) en un paso, pero borra más en muchos pasos — ¿corregir D mueve el cruce a r≈1?
- **Método:** medir D sobre múltiples escalas de tiempo (1, 2, 5, 10, 50 pasos), redefinir r con la D multi-paso correcta, ver si el cruce se alinea a r≈1.
- **Barrido:** escala de medición de D × r × ≥12 semillas.
- **NULL:** — (es verificación de instrumento).
- **Verificación cruzada:** el r* de F2-1 recalculado con D multi-paso debe acercarse a 1; si no, el "r=1" no es el crítico físico y se dice.
- **PASS pre-registrado:** se reporta cómo se mueve el cruce al corregir D; sin afirmar r*=1 salvo que la curva lo muestre.
- **Evita:** T1 (D medida, no fijada).

### F2-3 · "Irreversibilidad del corte: ¿el congelamiento requiere cortes permanentes?"
- **Simple:** ¿la diferencia se congela porque los cortes NO vuelven? Si dejamos que reconecten, ¿desaparece la persistencia?
- **Método:** introducir probabilidad de reconexión de aristas cortadas, barrida de 0 (irreversible = nuestro modelo) a 1 (totalmente reversible). Aísla el MECANISMO.
- **Barrido:** prob. reconexión ∈ [0 … 1] (≥10 puntos) × r × ≥12 semillas.
- **NULL:** barajado del acople.
- **Verificación cruzada:** predicción fuerte — P debe caer monótonamente al subir la reconexión; si no cae, el mecanismo no es la irreversibilidad.
- **PASS pre-registrado:** P(reconexión) decreciente, con P alto en 0 y P≈NULL en 1.
- **Evita:** T2 (prueba el porqué, no solo el qué).

### F2-4 · "Expansión y reabsorción como ejes independientes: ¿colapsa todo en r?"
- **Simple:** ¿de verdad manda solo la razón r, o H y D por separado?
- **Método:** barrer H y D en un grid 2D INDEPENDIENTE (no solo su cociente), ver si los puntos colapsan sobre una curva única de r=H/D.
- **Barrido:** H (≥8) × D (≥8) grid 2D × ≥12 semillas.
- **NULL:** barajado.
- **Verificación cruzada:** el colapso sobre r ES la verificación (si no colapsa, r no es la variable correcta).
- **PASS pre-registrado:** los puntos (H,D) colapsan sobre P(r) dentro de tolerancia.
- **Evita:** T1/T7 (dos ejes barridos, no un cociente asumido).

### F2-5 · "Congelamiento bajo expansión no uniforme (historia temporal variable)"
- **Simple:** la expansión real no es a tasa constante — ¿el congelamiento aguanta si la tasa cambia en el tiempo?
- **Método:** barrer perfiles H(t) — acelerando, desacelerando, ráfaga-luego-lento — con la misma expansión total.
- **Barrido:** 4+ perfiles H(t) × r medio × ≥12 semillas.
- **NULL:** barajado.
- **Verificación cruzada:** el resultado debe depender del r efectivo integrado, no del perfil; si un perfil rompe eso, se reporta.
- **PASS pre-registrado:** persistencia robusta al perfil a r efectivo fijo.
- **Evita:** T1 (no se fija una historia; se barren varias).

### F2-6 · "NULL alternativo: barajar la historia de cortes, no el acople final"
- **Simple:** ¿el REAL≻NULL depende de cómo construimos el azar de comparación?
- **Método:** un NULL distinto — barajar la SECUENCIA temporal de qué aristas se cortan, no el grafo final. Cross-check del NULL.
- **Barrido:** r × ≥12 semillas, con ambos NULLs.
- **Verificación cruzada:** REAL debe ganar a AMBOS NULLs; si gana a uno y no al otro, el resultado es frágil.
- **PASS pre-registrado:** REAL≻NULL con los dos nulos independientes.
- **Evita:** T4 (robustez del NULL mismo).

---

## ENFOQUE 3 — ¿ENFRIAR ES EXPANDIR? (enfriamiento adiabático, sin baño externo)

Núcleo tocado (CF-2, pero no robusto). Aquí se hace robusto y con métodos múltiples.

### F3-1 · "Estiramiento geométrico del gradiente, barrido amplio de a + ruido dinámico"
- **Simple:** ¿al crecer el espacio el gradiente físico se suaviza solo?
- **Método:** ∇_fis = ∇_comov / a; el núcleo de CF-2 pero con la corrección de robustez.
- **Barrido:** factor de expansión a ∈ [1 … 1e4] (log, ≥12 puntos) × **amplitud de ruido inicial** ∈ [1e-4 … 1e-1] (≥6 puntos — la perturbación DINÁMICA que CF-2 no tenía) × ≥12 semillas.
- **NULL:** densidad fija (sin dilución) — el gradiente comóvil debe seguir erosionándose.
- **Verificación cruzada:** (a) NULL de ρ fija muerde; (b) pendiente ≈ −1 (estiramiento puro); (c) disco.
- **PASS pre-registrado:** suavizado monótono con a en REAL, ausente en NULL, ESTABLE al variar la amplitud de ruido (no solo la semilla).
- **Evita:** T7 (robustez dinámica, la falla de CF-2).

### F3-2 · "Enfriamiento como consecuencia medida, no impuesta: T leída del estado"
- **Simple:** ¿la temperatura baja por la expansión sola, sin poner un término de enfriamiento?
- **Método:** NO hay término de cooling; se mide T del propio campo (varianza/energía) mientras se expande, y se ve si T cae con a de forma emergente.
- **Barrido:** a (amplio) × ≥12 semillas × amplitud de ruido.
- **NULL:** sin expansión (T no debe caer).
- **Verificación cruzada:** T(a) emergente debe seguir una ley de potencia; se mide el exponente, no se impone.
- **PASS pre-registrado:** T cae monótona con a en REAL y no en el control sin expansión; se reporta el exponente medido.
- **Evita:** T1 (enfriamiento no puesto a mano — emerge).

### F3-3 · "Tasa de dilución: ¿es a⁻³ especial o solo monótona?"
- **Simple:** la densidad cae como a⁻³ — ¿ese número concreto importa, o cualquier caída sirve?
- **Método:** barrer el exponente de dilución ρ∝a^(−n) con n ∈ {1,2,3,4,5}, ver cómo cambia el destino del gradiente.
- **Barrido:** n (5 valores) × a × ≥12 semillas.
- **NULL:** n=0 (densidad fija).
- **Verificación cruzada:** el efecto debe ser monótono en n; a⁻³ no debe ser un punto singular salvo que la física lo exija.
- **PASS pre-registrado:** se reporta la familia de curvas por n; ninguna se privilegia a mano.
- **Evita:** T1 (no se fija n=3; se barre).

### F3-4 · "Reversibilidad térmica: ¿se re-homogeneiza si se detiene la expansión?"
- **Simple:** si paramos la expansión a mitad, ¿el gradiente se vuelve a borrar (la difusión lo alcanza) o queda congelado?
- **Método:** detener la expansión en distintos tiempos y dejar correr solo difusión; medir si el gradiente re-aplana.
- **Barrido:** tiempo de parada (≥8 puntos a lo largo de la corrida) × ≥12 semillas.
- **NULL:** nunca parar (expansión continua).
- **Verificación cruzada:** predicción — parar temprano permite re-homogeneizar; parar tarde (ya aislado) no. Debe haber un tiempo de no-retorno.
- **PASS pre-registrado:** se reporta la curva "re-aplanamiento vs tiempo de parada"; existe (o no) un punto de no-retorno.
- **Evita:** T2 (prueba la irreversibilidad del adiabático).

### F3-5 · "Enfriamiento en el espectro: ¿qué escalas se congelan primero?"
- **Simple:** ¿se congelan antes las estructuras grandes o las chicas?
- **Método:** análisis de Fourier — seguir qué longitudes de onda dejan de difundir a medida que a crece. Lente de verificación totalmente distinta (espectral).
- **Barrido:** a × banda de escalas (todo el espectro) × ≥12 semillas.
- **NULL:** densidad fija.
- **Verificación cruzada:** predicción física — las escalas grandes se congelan primero (no pueden re-homogeneizarse a través del horizonte creciente). Si sale al revés, es dato en contra.
- **PASS pre-registrado:** se reporta el espectro de congelamiento vs a; se compara con la predicción de escalas-grandes-primero.
- **Evita:** T7 (método independiente — espectral, no de gradiente).

### F3-6 · "Control negativo: enfriamiento CON baño externo (lo prohibido)"
- **Simple:** para probar que NO estamos simulando un baño térmico sin darnos cuenta, metemos uno a propósito y mostramos que el resultado es distinto.
- **Método:** añadir explícitamente un baño externo (acoplamiento a reservorio a T fija) como control NEGATIVO; comparar con el caso adiabático (sin baño).
- **Barrido:** intensidad del acople al baño ∈ [0 … fuerte] (≥8 puntos) × a × ≥12 semillas.
- **NULL:** — (es control de método).
- **Verificación cruzada:** con baño, T→T_baño (no ley de potencia de a); sin baño, T∝a^(−n). Deben diferir claramente.
- **PASS pre-registrado:** el caso adiabático (acople=0) difiere del caso con baño; confirma que nuestro enfriamiento NO es un baño encubierto.
- **Evita:** T1/T0 (control explícito de lo que NO debe estar).

---

---

## ENFOQUE 4 — ¿EL DESCENSO DE DENSIDAD POR EXPANSIÓN ES UNA PALANCA CAUSAL PROPIA?

La densidad cae con la expansión (ρ∝a⁻³) — es consecuencia, no causa aparte. Pero como
ρ y el factor de escala a están atados, hay que MEDIR si la densidad tiene efecto propio
o si es solo la expansión disfrazada. Ése es el espinazo anti-Shannon de este enfoque.

### F4-1 · "Densidad emergente: ¿ρ cae sola al expandir, sin imponerlo?"
- **Simple:** ¿la densidad del campo baja por sí misma al expandirse, o hay que forzarla?
- **Método:** NO imponer ρ∝a⁻³; medir la densidad efectiva del campo (masa/energía por volumen comóvil) mientras se expande y ver qué ley de potencia emerge.
- **Barrido:** factor de expansión a ∈ [1 … 1e4] (log, ≥12 puntos) × ≥12 semillas × amplitud de ruido dinámico.
- **NULL:** sin expansión (ρ no debe caer).
- **Verificación cruzada:** (a) control sin expansión; (b) el exponente medido se compara con −3 (esperado en 3D) SIN imponerlo — se reporta el que salga; (c) disco.
- **PASS pre-registrado:** ρ cae monótona con a en REAL y no en el control; se reporta el exponente medido, no se fija.
- **Evita:** T1 (la dilución emerge, no se pone a mano).

### F4-2 · "Densidad vs expansión DESACOPLADAS: ¿efecto causal propio de ρ?" ★ (el clave)
- **Simple:** ¿la densidad hace algo por sí sola, o todo lo que vemos es la expansión?
- **Método:** desacoplar artificialmente las dos — (a) expandir manteniendo ρ fija (compensando), (b) bajar ρ sin expandir (a fijo), (c) ambas juntas (natural). Ver cuál de las tres controla el destino de la diferencia.
- **Barrido:** grid de {expansión ON/OFF} × {dilución ON/OFF} × r × ≥12 semillas.
- **NULL:** barajado del acople en cada rama.
- **Verificación cruzada:** predicción a decidir por el dato — si (b) "solo ρ baja" congela la diferencia tanto como (c), la densidad es causal propia; si (b) no hace nada y solo (a)+(c) con expansión congelan, la densidad es un proxy de la expansión.
- **PASS pre-registrado (tres lecturas):** densidad causal propia / densidad = proxy de expansión / interacción de ambas — se reporta cuál, con la curva. Es el experimento que responde la pregunta del enfoque.
- **Evita:** T2/T1 (aísla la variable en vez de asumir que r lo explica todo).

### F4-3 · "Dilución y reabsorción: ¿la caída de densidad apaga la difusión?"
- **Simple:** menos densidad = menos encuentros = ¿menos re-homogeneización? ¿La difusión se frena sola al diluirse el campo?
- **Método:** medir la difusividad efectiva D como función de la densidad ρ (a densidad alta, media, baja), sin tocar nada más.
- **Barrido:** ρ ∈ varias décadas (≥10 puntos log) × ≥12 semillas.
- **NULL:** — (medición de instrumento: D(ρ)).
- **Verificación cruzada:** predicción física — D debe caer al bajar ρ (menos acoplamientos activos); si D es independiente de ρ, la dilución no actúa por esta vía.
- **PASS pre-registrado:** se reporta la curva D(ρ); se dice si la dilución frena la difusión o no.
- **Evita:** T7 (barrido de ρ, no un punto).

### F4-4 · "Densidad crítica de congelamiento: ¿hay un ρ bajo el cual la diferencia se bloquea?"
- **Simple:** ¿existe una densidad tan baja que la diferencia ya no puede borrarse (queda congelada por dilución)?
- **Método:** barrer la densidad inicial/final y buscar un umbral ρ_c bajo el cual la persistencia se dispara.
- **Barrido:** ρ ∈ varias décadas (≥15 puntos) × r × ≥12 semillas × ruido dinámico.
- **NULL:** barajado del acople.
- **Verificación cruzada:** (a) NULL cae; (b) el umbral ρ_c debe ser estable al variar N y la perturbación (no de una semilla); (c) disco.
- **PASS pre-registrado:** curva P(ρ) entera; existe (o no) un ρ_c de congelamiento — ambos son hallazgo.
- **Evita:** T5 (curva continua, no gate binario).

### F4-5 · "Rugosidad de densidad: ¿importa el gradiente de ρ, no solo su valor medio?"
- **Simple:** el fondo no es uniforme — ¿que unas zonas sean más densas que otras cambia dónde persiste la diferencia?
- **Método:** sembrar un campo de densidad con estructura espacial (no uniforme) y medir si la persistencia se correlaciona con las zonas densas vs. las diluidas.
- **Barrido:** amplitud de la rugosidad de densidad ∈ [0 … fuerte] (≥8 puntos) × r × ≥12 semillas.
- **NULL:** rugosidad de densidad barajada (misma estadística, sin estructura).
- **Verificación cruzada:** si la persistencia se correlaciona con la estructura de densidad REAL y no con la barajada, la rugosidad de ρ es causal.
- **PASS pre-registrado:** se reporta la correlación persistencia↔densidad, REAL vs NULL.
- **Evita:** T0/T2 (la estructura de densidad se mide contra su propio barajado).

### F4-6 · "Control de dilución: ρ∝a⁻ⁿ barrido + doble NULL"
- **Simple:** cross-check de que el efecto de densidad no es artefacto de cómo modelamos la dilución.
- **Método:** barrer el exponente n en ρ∝a⁻ⁿ (n ∈ 1…5, cruza con F3-3 pero aquí el observable es la persistencia, no el gradiente) y confirmar con dos NULLs independientes.
- **Barrido:** n (5 valores) × a × ≥12 semillas, con NULL-barajado-acople y NULL-densidad-fija.
- **Verificación cruzada:** el efecto de densidad debe sobrevivir a AMBOS nulos; n=3 no debe ser singular salvo que la física lo exija.
- **PASS pre-registrado:** familia de curvas por n; efecto robusto a los dos nulos.
- **Evita:** T1/T4 (n barrido, doble NULL).

---

## 2. REGLAS DE EJECUCIÓN (CC firma antes de correr)

1. **Un `PROTOCOLO_<nombre>_PREREGISTRO.md` fechado por experimento**, ANTES del motor:
   observable exacto, NULL, criterio de PASS con umbral, rangos del barrido, semillas.
   Si falla, se reporta el FAIL — no se edita el protocolo (T3).
2. **Barridos extensos + perturbación dinámica**, nunca un punto ni solo semillas (T7).
3. **Tres verificaciones por experimento:** su NULL + un segundo observable/método +
   auditoría en disco por quien no lo escribió.
4. **La cantidad medida ≠ su juez** (T2). **Todo gate debe poder fallar** (T5, T6).
5. **Nada a mano:** solo ε y las palancas físicas se barren; ningún coeficiente elegido
   para dar un resultado (T1). Cuantos/dimensión = salida, no entrada (T0).
6. **No cambiar el código tras la revisión de CS.** Si CC ve un error: PARA y reporta a
   CS con la línea exacta. No "arregla" a criterio propio.
7. **Ejecutar completo.** El director autorizó cómputo largo — que tarde lo que tarde.
8. **Entregar crudo a CS:** curvas completas + dispersión real entre semillas Y entre
   perturbaciones dinámicas. **No adjudicar** — el veredicto lo da CS con la curva.

## 3. ORDEN SUGERIDO (no obligatorio)

- **Enfoque 1** primero (F1-1 es re-verificación del núcleo; F1-2/1-3 los más informativos).
- **Enfoque 2** después (F2-1 resuelve el r* que quedó abierto; F2-3 aísla el mecanismo).
- **Enfoque 3** después (F3-1 arregla la no-robustez de CF-2; F3-4/3-5 son los nuevos).
- **Enfoque 4** con el 2/3 — F4-2 (desacople densidad/expansión) es el más informativo de
  toda la batería: responde si la densidad es causa propia o proxy de la expansión.
- Reparto CC/Grok a criterio del director; F1-6, F3-5, F4-5 (2D, espectral, rugosidad) son los más caros.

**Nota final (la de siempre):** cualquier NEGATIVO es un hallazgo, no un fracaso. Esta
batería está diseñada para poder fallar en cada punto — por eso sirve. El objetivo no es
"hacer que los factores pasen", es medir honestamente cuán robustos son.
