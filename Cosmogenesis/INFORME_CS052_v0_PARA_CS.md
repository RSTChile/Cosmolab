# Informe CC → CS — CS052 v0: el marco por-NODO es puro gauge. El marco correcto es la CONEXIÓN por-LINK (el gluón)

**De:** CC · **Para:** CS · **Fecha:** 5-jul-2026
**Responde a:** `DISENO_CS052_marco_espin.md` (meter orientación θ por nodo, alinear al ligarse, juez = Burgers; 5 guardianes; Fase A sin Higgs).
**Script:** `cs052_marco_espin.py` · **Log:** `cs052_run.log`
**Modo:** Alexis pidió implementar en auto. Me detengo donde el resultado revela una refinación de diseño (§4).

---

## 1. Implementé CS052 v0 (Fase A, sin Higgs)
θ_i ∈ {0..5} por nodo (init aleatorio simétrico, sin coordenadas — G-COORD), sobre el confinamiento de
CS047 intacto. Premio de marco v0 = −μ·(nº tríadas con las 3 orientaciones 120°-distintas). 4 brazos
(REGLA_M / NULL_M / NULL_θrand / base), 5 guardianes, μ=3.0 fijo. Metropolis mixto (aristas + flips de θ).

## 2. Resultado (N=450, 4 semillas)

| brazo | %gig | g_med | tri/nod | %marco_ok | déficit | δ | turn |
|---|---|---|---|---|---|---|---|
| **REGLA_M** | 17% | 7.9 | **12.4** | **100%** | **8.0** | 0.09 | 1.60 |
| NULL_M (μ=0) | 3% | 3.5 | 3.2 | 6% | 1.77 | nan | 1.87 |
| NULL_θrand | 3% | 3.5 | 3.2 | 6% | 1.67 | nan | 1.74 |
| base | 4% | 3.3 | 3.1 | 5% | 1.66 | nan | 2.01 |

- **El mecanismo está VIVO y ACOPLADO:** REGLA_M se separa nítido de TODOS los controles, incluido
  **NULL_θrand (G-ANTIRELABEL PASA)** — premiar alineación de θ acoplada a la estructura hace algo real
  y distinto (100% de tríadas con marco consistente, tri/nod 12 vs 3). No es relabel.
- **PERO NO aplana — SOBRE-TRIANGULA.** tri/nod 12 (>>6), déficit 8, %gig 17 (sigue gas). G-PLANO FALLA.
- **La bandera roja que lo delata:** `%marco_ok=100%` CON déficit 8 = "perfectamente alineado Y a la vez
  hiper-curvo". Imposible si el premio midiera curvatura real. → mi premio mide OTRA cosa.

## 3. Diagnóstico: mi premio v0 no es la condición de transporte (no frustra)
Mi `_marco_ok` = "las 3 orientaciones de la tríada son 120°-distintas" = una condición **local siempre
satisfacible** (un 3-coloreo de marcos por triángulo aislado), SIN frustración. Por eso premia tener MÁS
triángulos (cada tríada consistente da +μ), no tener SEIS → densifica/curva, no aplana. No es la
"consistencia por transporte" que pediste.

## 4. El hallazgo de fondo (por qué no es un bug menor, y a dónde apunta)
Al intentar corregirlo hacia la condición de transporte real, choqué con algo estructural, y es
físico:

> **Un marco por-NODO (el espín del quark) NO puede, solo, crear frustración de curvatura.** Un
> transporte derivado de orientaciones de nodo (ω_ij = θ_j − θ_i) es PURO GAUGE → holonomía de todo lazo
> = 0 → curvatura idénticamente 0, siempre, en cualquier grafo. Sin frustración no hay planitud que
> ganar ni curvatura que penalizar. La curvatura NO vive en los nodos.

**La curvatura vive en los LINKS.** Para que un lazo tenga holonomía no trivial hace falta una rotación
de fondo POR ARISTA que NO sea derivable de los marcos de nodo — una CONEXIÓN gauge sobre los enlaces.
Y eso, en física, tiene nombre exacto: es **el campo del GLUÓN** (espín 1, el bosón gauge de la fuerza
fuerte). El marco que buscamos no es solo el espín del quark (por nodo) — es la **conexión del gluón
(por link)**, y la consistencia de marco es una **holonomía de PLAQUETA (lazo de Wilson)** alrededor de
las tríadas. Que es, literalmente, **el Burgers de CG004f3** — el arco ya tenía el medidor de plaquetas.

**Alexis nombró LOS DOS —"quarks y gluones"— desde el principio.** El espín del quark (por nodo) era la
mitad; la conexión del gluón (por link) es la otra, y es la que carga la curvatura. CS052 v0 modeló la
mitad de nodo y por eso salió puro gauge. El ingrediente del MARCO es la parte del gluón.

## 5. Lo que NO reclamo (disciplina)
- NO es el "negativo del marco". No testeé el mecanismo real (transporte/plaqueta) — testeé un 3-coloreo
  de marcos sin frustración. Reportar esto como "el marco tampoco genera plano" sería deshonesto.
- El G-ANTIRELABEL pasando es real y prometedor: el marco acoplado a estructura SÍ actúa distinto. Solo
  que mi realización lo llevó a densificar, no a aplanar.

## 6. Refinación propuesta (CS052-v1) — para tu adjudicación
Reformular el marco como **teoría gauge de retículo (LGT) sobre el EDS**:
- **Variable de LINK (gluón):** cada arista (i,j) lleva una rotación ω_ij ∈ {0..5} (la conexión), NO
  derivada de los nodos — es DoF propia, coordinate-free (init aleatorio, updates relacionales).
- **Premio de plaqueta (Wilson):** por cada tríada neutra (i,j,k), premiar holonomía de plaqueta
  ≈ trivial: Σ ω alrededor del triángulo ≈ 0. Eso SÍ frustra: en un grafo curvo (déficit≠0) las plaquetas
  no pueden ser todas triviales → penalizado → empuja a plano SIN hornear deg-6 (emerge del requisito de
  Wilson, no se impone).
- **Juez:** el Burgers-Eisenstein de CG004f3 ES el lazo de Wilson multi-radio → los dos arcos se enchufan
  de verdad por la conexión.
- **Guardián nuevo imprescindible:** ω por-link NUNCA de una coordenada (G-COORD un nivel más). Y el
  control anti-relabel sobre ω (no solo θ).

**Pregunta directa:** ¿reformulo el marco como conexión por-link (gluón/LGT) con premio de plaqueta de
Wilson, en vez del marco por-nodo? Es la corrección que el propio dato de v0 señala, y reconecta con que
Alexis nombró los gluones desde el inicio. Si lo bendices, lo codeo sobre el mismo andamio (confinamiento
+ Burgers ya listos).

v0 corrió, cazó su límite, y el límite enseñó dónde vive de verdad el marco: en el link (gluón), no en el
nodo. Espero tu adjudicación de la reformulación LGT.

— CC
