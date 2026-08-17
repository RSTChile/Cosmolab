# Informe CC → CS — CG005 v0 (EDS): el lógos CONFINA pero NO extiende. Falta el "al lado de" (= Fase I)

**De:** CC · **Para:** CS · **Fecha:** 4-jul-2026
**Marco:** Protocolo CG005 (Alexis) · reemplazo canónico de CG004 · fundamento `origen_era_la_relacion`.
**Script:** `cg005_eds_v0.py` · **Log:** `cg005_run.log`
**Modo:** Alexis autorizó desarrollo autónomo ("avanza hasta que sea imprescindible que yo apruebe"). Me detengo en el punto donde la aprobación ES imprescindible (§4).

---

## 1. Lo que construí (andamio validado, Fase II con regla v0)
EDS = nodos con color INMUTABLE {R,V,A} + vínculos condicionados por neutralidad local, enfriados por
un filtro Metropolis (`S=I×E`): parte de vínculos al azar (caos, S→0) y disuelve/forma hacia la
neutralidad. **Regla v0 (local, NO horneada — la planitud NO es la función-costo):** energía
`E = Σ_i [ c_bond·grado_i − λ·(1−e^{−t_i/τ}) ]`, con `t_i` = nº de tríadas de color NEUTRO {R,V,A}
(barión=blanco) del nodo i. La **saturación** `(1−e^{−t/τ})` encarna el confinamiento: un quark se
neutraliza en UN hadrón y se satura (más vínculos sólo cuestan). Arnés de medición = el calibrado de
CG004 (δ Gromov, dim, diam-pend, turn, %gig; anclas lattice2D+/árbol−). Guardianes cableados: identidad
inmutable, %gig, y **brazo NULL del lógos** (color-ciego: premia CUALQUIER triángulo).

**Primer intento colapsó** (falsabilidad a): premio `λ·(nº triángulos)` sin saturar crece como grado²
→ grafo casi completo (grado 205, diam 2, "agujero negro topológico"). El guardián lo cazó. Lo corregí
con el premio SATURANTE (física real: el confinamiento satura). Lo dejo escrito porque es la prueba de
que el guardián funciona.

## 2. Resultado v0 (N=450, 4 semillas, REGLA vs NULL)

| brazo | tríadas-neutras/nodo | %confinados | δ_med | turn | diam | %gig |
|---|---|---|---|---|---|---|
| **REGLA** (neutralidad de color) | **5.7** | **100%** | 0.29 | 12.6 | 3 | 100 |
| **NULL** (color-ciego) | 1.9 | 82% | 0.28 | 10.9 | 3 | 100 |
| ancla lattice2D (plano) | — | — | **2.19** | **1.15** | **57** | 100 |
| ancla árbol_b3 (hiperbólico) | — | — | 0.00 | 1.97 | 12 | 100 |

**Guardianes OK:** %gig=100 (no colapsó ni fragmentó, falsabilidad a NO se dispara ahora); identidad
inmutable (assert OK, falsabilidad b descartada — no se diluyó a nodos vacíos).

## 3. Dos hechos, los dos honestos
1. **El LÓGOS confina, y se SEPARA del NULL.** REGLA mete al **100%** de los nodos en tríadas de color
   neutro (~5.7/nodo); el control color-ciego sólo al 82% con 1.9. La regla de neutralidad hace
   **hadrones reales**, y el control lo confirma: es el CONTENIDO (color/lógos), no el mero formar
   triángulos. El confinamiento del §2 del protocolo FUNCIONA y es del lógos, no de la identidad sola.
2. **Pero NO hay EXTENSIÓN.** REGLA y NULL son **ambos mundo-pequeño/hiperbólicos** (diam 3, δ≈0.29 ≈
   árbol 0.00, turn≈12 ≫ plano 1.15). El espacio plano extendido NO emergió — ni cerca del ancla plana.

## 4. Diagnóstico y por qué me detengo (aprobación imprescindible)
La neutralidad es **necesaria pero no suficiente**: el lógos LIGA unidades (hadrones) pero no las
arregla "al lado de". La razón es mecánica y exacta: **el modelo v0 no tiene LOCALIDAD** — cualquier
nodo puede vincularse con cualquiera, así que los hadrones cuajan como un *blob* de hubs (diam 3), no
como una retícula extendida. Y "local" presupone métrica, que es lo que queremos que EMERJA
(huevo-y-gallina).

Ese ingrediente faltante —el principio que fuerza PROXIMIDAD/extensión— es **literalmente el objetivo
declarado de tu Fase I**: *"identificar en qué punto el congelamiento genera la primera restricción
topológica que fuerza la noción de al-lado-de"*. El confinamiento da las unidades; el "al lado de" es
un principio APARTE, aguas arriba. Es coherente con todo el arco: en CG004 la dimensión se decidía en
el CRECIMIENTO (frente/attach), no en la reparación; aquí la extensión se decidiría en el arreglo, no
en el confinamiento.

**No fabrico ese mecanismo solo** — sería el error del fundamento (fabricar el diseño sin la relación,
que se coló dos veces). Es tu dominio (Fase I) y una decisión de física.

## 5. Candidatos que veo para el "al lado de" (para TU adjudicación, no para que yo elija)
1. **Localidad de vínculo por CRECIMIENTO (frente/accreción):** los hadrones se forman incrementalmente
   pegándose al borde de la red ya cuajada, no por Metropolis global. CG004 mostró que la dimensión se
   decide al crecer. Riesgo: ¿de dónde sale el "borde" sin métrica previa? (¿del orden de
   congelamiento?).
2. **Saturación de VALENCIA que fuerza dispersión:** cada hadrón tiene enlaces residuales contados
   (fuerza nuclear residual entre hadrones) → los bonos entre hadrones se reparten en vez de agruparse
   → extensión. Mapea a la física real (residual strong force liga nucleones en núcleos).
3. **Orden temporal del congelamiento como el "primer al-lado-de":** lo que se confina JUNTO en el
   tiempo queda adyacente — la historia (Nivel III) genera la proximidad. Esto es lo más fiel a
   "traducir el confinamiento como Nivel I→II→III".

**Pregunta directa:** ¿cuál de estos (o cuál otro) mapea el confinamiento→proximidad que Fase I debe
fijar? En cuanto lo adjudiques, lo codeo como v1 sobre este mismo andamio (que ya confina y ya mide).

Andamio validado, confinamiento real (REGLA≠NULL), guardianes funcionando, y un negativo PRECISO que
localiza el hueco: falta el "al lado de", y es tuyo de mapear. Espero Fase I.

— CC
