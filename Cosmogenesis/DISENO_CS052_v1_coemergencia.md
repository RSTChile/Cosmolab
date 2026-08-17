# DISEÑO CS052-v1 — La CO-EMERGENCIA del espacio: ni la entidad sola ni el vínculo libre; solo el vínculo ATADO a sus extremos

**Número:** CS052-v1 (misma línea CS052, versión v1 — ver REGISTRO_EXPERIMENTOS_CS.md)
**Dimensión técnica:** teoría gauge de retículo (LGT) sobre el EDS · conexión por-link (gluón) + plaqueta
de Wilson = Burgers de CG004f3 · extiende CG002/r7g.
**Tesis a probar (Alexis):** "No hay espacio posible sin un *entre*, y ese entre es el gluón — pero el
espacio co-emerge del vínculo ATADO a sus extremos, no de la entidad sola ni del vínculo libre."
**Diseño:** Claude Science (CS) · **Tesis e ingredientes:** Alexis López Tapia.
**Estado:** DISEÑO. No corrido. A codear por CC con disciplina P-antes-de-B. Guardianes pre-registrados.
**Fecha:** 5-jul-2026 · **Fundamento:** `origen_era_la_relacion` · `adjudicacion_CS052_v0_LGT_CS`.

---

## 0. LA IDEA EN UNA LÍNEA
Montar los TRES casos en el mismo motor y ver cuál genera espacio plano: (A) el "hacia dónde" en la
ENTIDAD sola, (B) el "hacia dónde" en el VÍNCULO libre, (C) el vínculo ATADO a la entidad. La tesis
predice: A→0, B→0, C→plano. Si sale así, el espacio co-emerge del entre-atado, y queda probado que
NI la cosa sola NI el vínculo suelto bastan.

## 1. POR QUÉ LA ESTRUCTURA DE TRES BRAZOS *ES* LA PRUEBA
La tesis de Alexis no es "el vínculo genera el espacio" a secas — es una afirmación de CO-IMPLICACIÓN:
el espacio necesita el entre Y los extremos, atados. Una tesis de co-implicación se prueba mostrando que
CADA MITAD SOLA FALLA y solo el PAR ATADO funciona. Por eso el experimento no es un brazo con premio —
son tres brazos que se contrastan, y el patrón entre ellos es el resultado:

| brazo | qué es | dónde vive el "hacia dónde" | predicción | ya sabemos |
|---|---|---|---|---|
| **A — ENTIDAD sola** | marco por-NODO (espín del quark), θ_i | en las cosas | **Burgers ≡ 0** (gradiente, se cancela al dar la vuelta) | ✅ CS052-v0 lo mostró (gauge puro) |
| **B — VÍNCULO libre** | conexión por-LINK ω_ij LIBRE, sin atar | en el entre, pero suelto | **Burgers → 0** (se gauge-aplana en cualquier grafo) | ⏳ la trampa espejo, a confirmar |
| **C — VÍNCULO atado** | ω_ij ligado al marco (sin-torsión) | en el entre, atado a los extremos | **Burgers 0 ⟺ grafo plano** (frustración geométrica real) | ⏳ EL TEST |

**Lectura del patrón:**
- Si A=0, B=0, C discrimina (0 en plano, ≠0 en curvo) → **tesis CONFIRMADA**: ni la entidad sola ni el
  vínculo libre generan espacio; solo el vínculo atado a sus extremos. La co-emergencia es real.
- Si C tampoco discrimina → el entre-atado tampoco basta localmente → la planitud es aguas arriba
  (confirmación por tercer camino). También resultado.
- A y B son CONTROLES con predicción de CERO — no son "fracasos", son las dos mitades que DEBEN fallar
  para que la tesis de co-implicación signifique algo. Su cero ES parte del resultado positivo.

## 2. EL MECANISMO DE C (el vínculo atado — la pieza física)
Sobre el confinamiento de CS047 (intacto: el lógos liga quiénes por neutralidad de color):
- **Variable de LINK:** cada arista (i,j) lleva una rotación ω_ij ∈ {0..5} (la conexión del gluón).
  Init aleatorio (G-COORD: NUNCA de coordenada).
- **La atadura (lo que distingue C de B):** ω_ij no evoluciona libre. Se restringe a ser COMPATIBLE con
  el marco de las tríadas que une — el análogo discreto de "conexión sin torsión / compatible con la
  métrica" (Regge). En la práctica: el premio no es "plaqueta trivial a secas" (eso es B, gauge-libre),
  sino **"plaqueta trivial BAJO la rotación geométrica que el sustrato impone"** — exactamente el
  transporte triángulo→triángulo de CG004f3, donde el giro π/3 por paso lo FIJA la geometría, no un DoF
  suelto.
- **Premio de Wilson = −μ·|Burgers de la plaqueta|** (Eisenstein exacto) sobre cada tríada neutra.
  Frustra donde hay curvatura, se anula donde es plano. NO hornea coordinación-6: la planitud emerge del
  requisito de que los lazos de Wilson cierren, no se impone.
- Por qué C funciona y B no: en el plano {3,6} la conexión compatible ES plana → Wilson cierra → premio
  máximo. En {3,q>6} Gauss-Bonnet FUERZA déficit ≠ 0 → la conexión compatible NO puede ser plana →
  Wilson frustrado → penalizado. **Conexión plana posible ⟺ grafo plano.** La atadura es lo que hace que
  el premio mida el GRAFO (el espacio) y no solo la conexión.

## 3. EL JUEZ (une los dos arcos, sin métrica nueva)
El criterio es el **Burgers-Eisenstein multi-radio de CG004f3** (= el lazo de Wilson de la LGT), con el
estadístico Burgers_max sobre familia radio×centro para blindar la cancelación por simetría. Más el arnés
de CG005 (δ, dim, turn, %gig, anclas lattice2D/árbol) como cross-check. CG005 GENERA el medio; CG004 MIDE
si es plano. Los arcos se enchufan por la conexión, no por analogía.

## 4. GUARDIANES PRE-REGISTRADOS (la validez del resultado)
1. **G-COORD:** ni θ (brazo A) ni ω (B,C) se leen de coordenada. Assert de diseño — solo relaciones.
2. **G-NO-GAUGE-LIBRE (el que decide, va ANTES de leer nada):** test directo sobre grafos CONOCIDOS —
   corre el premio de Wilson de C sobre {3,6} plano Y sobre {3,7}/{3,8} curvos. DEBE dar Burgers 0 en
   {3,6} y ≠0 en {3,7},{3,8}. Si da 0 en los tres → la atadura no quedó atada (es B disfrazado de C) →
   el experimento no mide nada, corregir la ligadura ANTES de seguir. Este es el guardián que separa la
   tesis real de la trampa espejo.
3. **G-PLANO:** el brazo C ganador debe ACERCARSE al ancla lattice2D (Burgers→0 multi-radio Y turn↓ hacia
   1.15, δ↑ hacia 2.18, dim→~2), no solo "no ser cero". Un blob conexo pero curvo NO es éxito.
4. **G-ANTIRELABEL (sobre ω):** la conexión que funciona en C debe estar acoplada a la estructura, no ser
   una etiqueta permutable. Control ω-relabelado, como el G4 de CS050.
5. **G-CONFINA:** el gluón-conexión NO funde hadrones (tri/nodo se mantiene ≈ CS047).
6. **G-NOTUNE:** μ fijo por física antes de correr, reportado, NO movido buscando plano.

## 5. LO QUE CADA BRAZO APORTA A LA TESIS (por qué los tres, no solo C)
- **A (entidad sola) DEBE dar 0** → prueba que la cosa aislada no genera espacio. (Ya visto en v0.)
- **B (vínculo libre) DEBE dar 0** → prueba que el entre SUELTO tampoco basta — el vínculo necesita estar
  atado. Este brazo es nuevo y crítico: sin él, alguien diría "bastaba poner el DoF en el link". B
  demuestra que no: el link libre se desenrosca. Es la otra mitad del "ni... ni...".
- **C (vínculo atado) es el TEST** → si discrimina plano/curvo, el espacio co-emerge del entre-atado.
El resultado no es un número de C: es el PATRÓN A=0, B=0, C discrimina. Ese patrón ES la tesis de
co-emergencia hecha dato.

## 6. DESENLACES (cuerda honesta)
- **A=0, B=0, C discrimina y G-NO-GAUGE-LIBRE pasa:** tesis CONFIRMADA — el espacio co-emerge del vínculo
  atado a sus extremos; ni la entidad ni el vínculo libre bastan. PRIMER POSITIVO DE GENERACIÓN del arco,
  y con la forma exacta que Alexis predijo. Auditar el séxtuple (sobre todo G-NO-GAUGE-LIBRE y
  G-ANTIRELABEL).
- **C no discrimina (da 0 en plano y curvo):** o la atadura no quedó atada (G-NO-GAUGE-LIBRE lo caza →
  arreglar ligadura, no premio), o —si la atadura está bien— la planitud es aguas arriba aun con la
  conexión atada → negativo fuerte por tercer camino. Distinguir con G-NO-GAUGE-LIBRE cuál de los dos.
- **C discrimina pero da blob (curvo conexo):** el entre-atado genera medio pero no plano → falta el
  ingrediente que fuerza planitud vs curvatura. Hueco más estrecho, no muro.
- En todos: NO tunear μ. El patrón entre brazos es el resultado, no el valor de un brazo.

## 7. LO QUE NO SE RECLAMA (disciplina)
- No se reclama derivar el gluón — se USA una conexión gauge por-link, motivada por la física real
  (Alexis nombró los gluones desde el inicio; v0 mostró que el nodo solo es gauge puro).
- No se reclama hospedar el vértice gauge de 3 puntos fielmente (CS027/CS031 siguen bloqueados) — se
  prueba una conexión de retículo (LGT), que es el objeto correcto para el marco, no el gluón-entidad.
- No se reclama que la tesis esté probada por ser evidente — se prueba por el patrón de tres brazos.

## 8. RESUMEN OPERATIVO PARA CC
- Andamio: EDS de CS047 (color inmutable, confinamiento saturante, arnés, anclas) + Burgers de CS046. NO
  reconstruir.
- Tres brazos en el mismo motor: A (θ por nodo, = v0), B (ω por link LIBRE), C (ω por link ATADO/sin-
  torsión). Mismos controles NULL y misma semilla entre brazos.
- Juez: Burgers-Eisenstein multi-radio (= Wilson) + arnés CG005.
- Guardianes 1–6 pre-registrados; G-NO-GAUGE-LIBRE testeado sobre {3,6}/{3,7}/{3,8} ANTES de leer el
  medio emergente.
- Traer el PATRÓN A/B/C a CS para adjudicación. No tunear. No fabricar aguas abajo.
- Registrar el resultado como CS052-v1 en el registro. El siguiente número libre sigue siendo CS053.

— Diseño CS052-v1 por Claude Science. La tesis (el espacio es el entre; co-emerge del vínculo atado a
sus extremos) y los ingredientes (quark+gluón, nodo+link acoplados) son de Alexis López Tapia.
