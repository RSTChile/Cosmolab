# DISEÑO CS052 — El MARCO (espín) como el "hacia dónde" que genera consistencia local, con la masa como lectura relacional

**Número:** CS052 (secuencia CS, correlativo — ver REGISTRO_EXPERIMENTOS_CS.md)
**Dimensión técnica:** extiende CG002/r7g (primitivo de 3 puntos) + conecta con el cierre CG004/CG005
(el marco, no la adyacencia) + incorpora el sector Higgs (CG002/r7f) como término subordinado.
**Diseño:** Claude Science (CS) · **Ingrediente, rumbo y correcciones de fondo:** Alexis López Tapia.
**Estado:** DISEÑO. No corrido. A codear por CC con disciplina P-antes-de-B. Guardianes pre-registrados.
**Fecha:** 5-jul-2026 · **Fundamento:** `origen_era_la_relacion` · `PUERTA_R7_espin_como_marco`.

---

## 0. UNA LÍNEA
Meter en el EDS una **orientación intrínseca por nodo (análogo del espín)** que se ALINEA al ligarse
—consistencia de marco, no solo neutralidad de color— y medir con el **Burgers de CG004** si el medio
relacional sale PLANO. Es el ataque directo al hueco que los dos arcos localizaron: la adyacencia ya se
probó (no basta); el marco nunca se tocó.

## 1. POR QUÉ ESTE EXPERIMENTO, Y POR QUÉ AHORA (la genealogía completa)
Todo lo conversado converge aquí. En orden:
1. **El fundamento** (`archē`/`lógos`): no hay geometría preestablecida; todo emerge de diferencias que
   persisten y se relacionan. Corrigió el error de los nodos vacíos.
2. **CG005** puso identidad real (color) → el lógos CONFINA (CS047, 100% vs 82%). Pero confinar liga
   QUIÉN con quién (adyacencia), y eso solo da gas o blob (CS048–CS050), nunca plano.
3. **CG004** cerró en paralelo (CS039–CS046): el pegado PRESERVA lo plano, no lo GENERA. Frontera κ=0⁺.
4. **El hueco, localizado por dos caminos:** todos los mecanismos operan sobre la ADYACENCIA
   (quién-con-quién); ninguno sobre el MARCO (con-qué-orientación-relativa). La planitud vive en el marco.
5. **La pregunta de Alexis (el espín):** los quarks (½) y gluones (1) tienen espín SIEMPRE, intrínseco.
   El espín ES una dirección — el "hacia dónde" que el color (escalar) no captura. Es el marco que faltaba.
6. **La corrección de la masa (Alexis):** la masa de la materia es 99% RELACIONAL (energía de
   confinamiento, E=mc²) y solo 1% Higgs. La frontera "no tocar la masa" era falsa: el confinamiento YA
   contiene la masa. Y el Higgs es una etiqueta intrínseca (1%), el vértice de 3 puntos ya bloqueado
   (CS031) — el ingrediente MENOS central. Entra subordinado, no como protagonista.

**Conclusión de diseño:** el protagonista es el MARCO (espín), no la masa ni el Higgs. La masa es una
LECTURA del confinamiento que ya tenemos; el Higgs es un término pequeño opcional. El experimento se
juega en si la alineación de marcos genera planitud.

## 2. EL OBJETO NUEVO (lo único que se agrega al andamio EDS)
Cada nodo, además de su **color inmutable** {R,V,A} (identidad, ya existe), lleva ahora una
**ORIENTACIÓN intrínseca** `θ_i` — un marco local, el análogo del espín. Es un atributo DIRECCIONAL de
la diferencia, no un vínculo.
- **Representación:** para poder medir con el Burgers (que vive en giros de π/3, enteros de Eisenstein),
  la orientación se discretiza en el MISMO retículo: `θ_i ∈ {0,1,2,3,4,5}` (múltiplos de 60°). Coherente
  con CG004f3.
- **Origen del valor inicial (CRÍTICO — guardián duro):** `θ_i` NO se asigna desde ninguna posición ni
  coordenada. Emerge, igual que la energía de congelamiento emergió en CS050. Opción canónica: `θ_i`
  inicial aleatorio simétrico (sin dirección privilegiada) y se DEJA que la dinámica de alineación lo
  ordene. La ruptura de simetría la hace la relación, no una mano.

## 3. LA REGLA NUEVA (marco-consistencia, sobre el confinamiento intacto)
Dos capas, la vieja intacta y la nueva encima:
- **Capa 1 — confinamiento (SIN CAMBIOS, CS047):** el lógos liga por neutralidad de color RVA saturante.
  Decide QUIÉN se liga. Sigue igual.
- **Capa 2 — alineación de marco (NUEVA):** cuando dos nodos se ligan, sus orientaciones θ NO son
  independientes: la energía premia que los marcos vecinos sean CONSISTENTES. Término:
  `E_marco = −μ · Σ_aristas f(θ_i, θ_j, dir_ij)`
  donde la consistencia se evalúa por transporte (el mismo de CG004f3): el marco de i, transportado a lo
  largo de la arista, debe coincidir con el de j. Alinear = el giro relativo acumulado es coherente.
  **Esto es un vértice de RELACIÓN DIRECCIONAL — es la extensión del primitivo pareado a algo que carga
  orientación (el paso que r7g/CS032 tanteó sin el ingrediente físico correcto).**
- **μ (peso del marco) FIJADO por criterio físico ANTES de correr, no tuneado.** (Análogo: relación
  espín-órbita / rigidez de marco. Fijar un valor y NO moverlo buscando planitud.)

## 4. EL JUEZ (une los dos arcos — sin juez nuevo, sin perilla nueva)
El criterio de éxito NO es una métrica inventada: es el **Burgers de CG004f3** (CS046) medido sobre el
medio relacional que emerge.
- **Plano ⟺ Burgers = 0 a TODO radio** (multi-radio, Eisenstein exacto, con el estadístico Burgers_max
  sobre familia radio×centro para blindar la cancelación por simetría — ya diseñado en CS046).
- Además el arnés de CG005 (δ Gromov, dim, turn, %gig, anclas lattice2D/árbol) como cross-check.
- **CG005 GENERA el sustrato (con marco); CG004 MIDE si es plano.** Los dos arcos se enchufan por el marco.

## 5. LOS BRAZOS (control es todo — anti-Shannon, anti-circularidad)
| brazo | color | marco (θ) | qué prueba |
|---|---|---|---|
| **REGLA_M** | neutralidad RVA | alineación de marco ON | ¿marco-consistencia genera plano? |
| **NULL_M** (control 1) | neutralidad RVA | θ presente pero SIN premio de alineación | aísla el marco: si REGLA_M ≠ NULL_M, el mérito es la alineación |
| **NULL_θrand** (control 2) | neutralidad RVA | premio de alineación pero a θ BARAJADOS (marco sin acoplar a estructura) | anti-relabel (la lección de CS050): el marco debe estar acoplado, no ser etiqueta |
| **base CS047** | neutralidad RVA | sin θ | reproduce el gas/blob previo (línea base) |

## 6. GUARDIANES PRE-REGISTRADOS (antes de correr — son la validez del resultado)
1. **G-COORD (el duro del fundamento, un nivel más arriba):** θ NUNCA se lee de una posición/coordenada.
   Assert: la función que inicializa y actualiza θ recibe SOLO relaciones (color, aristas, θ de vecinos),
   jamás x,y. Si toca coordenada → espacio horneado → INVÁLIDO.
2. **G-PLANO (éxito real, no separarse-de-null):** REGLA_M debe ACERCARSE al ancla lattice2D
   (Burgers→0 multi-radio Y turn↓ hacia 1.15, δ↑ hacia 2.18, dim→~2). Un blob (conexo pero curvo) NO es
   éxito.
3. **G-ANTIRELABEL (la lección de CS050):** el θ que "funciona" debe estar CORRELACIONADO con la
   estructura relacional (co-alineados comparten estructura más que el azar). Si REGLA_M = NULL_θrand,
   el marco no aportó — fue relabel. Mídelo, como el G4 de CS050 (ratio ≫ 1).
4. **G-CONFINA (anti-colapso de 2º orden, la de CS049):** el marco NO debe romper el confinamiento —
   tríadas-neutras/nodo tras CS052 ≈ las de CS047 (~3.3–5.7). Si alinear marcos funde hadrones, inválido.
5. **G-NOTUNE (cuerda dura de siempre):** μ (y cualquier peso) fijado por física antes de correr,
   reportado, y NO movido buscando que salga plano. Cada perilla que se mueve es Shannon.

## 7. EL HIGGS / LA MASA — subordinado y OPCIONAL (fase B, solo si REGLA_M da algo)
La masa NO es el protagonista y NO es un experimento aparte: es una LECTURA. Dos cosas, en este orden:
- **Fase A (el experimento real): SIN Higgs.** Solo marco. Porque el 99% de la masa es relacional
  (confinamiento), que YA está en el modelo. Si el marco genera plano, la masa relacional se lee del
  propio medio (energía de enlace del confinamiento) — sin añadir nada.
- **Fase B (opcional, solo si A es positivo): el Higgs como el 1%.** Un término de masa INTRÍNSECA
  pequeño por nodo (etiqueta escalar, ~1% de la escala de energía de enlace), fijado por física. NO es
  un vértice de 3 puntos hospedado (eso quedó bloqueado en CS031 y NO se fuerza aquí): es solo un peso
  intrínseco pequeño que modula qué nodos frenan primero. Sirve para ver si el 1% de Higgs cambia algo
  sobre el 99% relacional — la predicción honesta es que casi no. Si cambia mucho, es sospechoso (revisar).
- **Guardián Higgs:** el término de masa intrínseca NO puede ser el que genera la planitud (sería
  contrabandear el resultado en una etiqueta). Debe ser subordinado: REGLA_M sin Higgs y con Higgs deben
  dar esencialmente lo MISMO en planitud; el Higgs solo modula, no genera.

## 8. DESENLACES (cuerda honesta — los tres, pre-escritos)
- **REGLA_M genera plano y los controles no:** PRIMER POSITIVO DE GENERACIÓN de espacio del arco entero.
  El marco (espín) era el lever. Enorme. Auditar el cuádruple (sobre todo G-ANTIRELABEL y G-COORD).
- **REGLA_M conecta pero da blob (curvo, no plano):** el marco liga con orientación pero no la ALINEA
  hacia planitud — falta la regla correcta de alineación, o el plano necesita algo más que consistencia
  local de marco. Hueco más estrecho, no muro.
- **REGLA_M sigue gas o = base:** el marco tampoco basta localmente → confirmación TRIPLE (CG004,
  CG005-adyacencia, CG005-marco) de que la planitud es aguas arriba. Sería el negativo más fuerte
  posible, y cerraría que NINGUNA regla local —ni de adyacencia ni de marco— genera plano. También es
  resultado grande.

## 9. LO QUE NO SE RECLAMA (disciplina del equipo)
- No se reclama haber "derivado el espín" — se USA el espín como orientación, motivado por la física real.
- No se reclama hospedar el vértice gauge de 3 puntos fielmente (CS027/CS031 siguen bloqueados); se
  prueba un vértice de RELACIÓN DIRECCIONAL, que es lo que el marco necesita, no el gluón-entidad.
- No se reclama que el Higgs sea central — se muestra, por diseño, que es el 1% subordinado.

## 10. RESUMEN OPERATIVO PARA CC
- Andamio: EDS de CS047 (color inmutable, confinamiento saturante, arnés, anclas). NO reconstruir.
- Agregar: θ_i por nodo (6 estados, emergente, sin coordenadas) + término E_marco de alineación por
  transporte (μ fijo por física).
- Medir: Burgers de CS046 (multi-radio, Eisenstein) + arnés CG005.
- Brazos: REGLA_M / NULL_M / NULL_θrand / base. Guardianes G-COORD, G-PLANO, G-ANTIRELABEL, G-CONFINA,
  G-NOTUNE, todos pre-registrados.
- Fase A sin Higgs. Fase B (Higgs 1%) SOLO si A es positivo, y con su guardián.
- Traer el resultado a CS para adjudicación. No tunear. No fabricar aguas abajo.

— Diseño CS052 por Claude Science. El ingrediente (espín=marco), la corrección de la masa (99%
relacional) y la detección de la frontera falsa ("no tocar la masa") son de Alexis López Tapia.
Registrado en REGISTRO_EXPERIMENTOS_CS.md. El próximo número libre tras este es CS053.
