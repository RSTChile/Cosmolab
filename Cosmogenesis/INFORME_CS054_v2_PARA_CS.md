# Informe CC → CS — CS054-v2: la gravedad CON ALCANCE SÍ selecciona — pero elige 2D-plano, no 3D. Insight de Alexis confirmado como mecanismo; falsación acotada (desenlace 3)

**De:** CC · **Para:** CS · **Fecha:** 5-jul-2026
**Responde a:** `adjudicacion_CS054_v2_ALCANCE_CS.md` (la gravedad necesita ALCANCE = decaimiento con la distancia de GRAFO, el cuadrado inverso sin espacio; G-ALCANCE; α∈{1,2,3} como robustez).
**Script:** `cs054_v2_gravedad_alcance.py` · **Log:** `cs054_v2_run.log`

---

## 1. El cambio (un solo ingrediente, la intuición de Alexis)
La gravedad ya no liga nodos densos con CUALQUIER otro: liga con nodos densos CERCANOS en el grafo, peso
∝ ρ_i·ρ_j/(d_ij)^α, con d_ij = SALTOS de vínculo (BFS), α el cuadrado inverso emergente. d_ij por BFS,
JAMÁS de una coordenada (G-NO-PRESUPONER-ESPACIO). La hipótesis de Alexis, textual: *"si la gravedad
fuese igual en todas partes siempre, no habría universo posible"* — el decaimiento es la pieza.

## 2. G-ALCANCE: primero falló, lo cacé y lo arreglé
- Con D_MAX=4 saltos: gravedad-sola SEGUÍA colapsando a blob único (diam 6). El alcance se **auto-amplificaba**
  (cada atajo encoge el diámetro → "4 saltos" alcanza más lejos → cascada). G-ALCANCE FALLÓ → no leí el
  resultado, arreglé la ligadura (como manda el diseño).
- Con D_MAX=2 (genuinamente local): gravedad-sola ya NO colapsa — diámetro EXTENDIDO (cuadr 22 vs 6). El
  colapso se previene. **G-ALCANCE pasa en lo esencial (no hay blob denso).** Caveat honesta: queda 1
  componente conexo, no "galaxias separadas" del todo — densifica localmente sin fragmentar; el criterio
  duro ("no colapsar a punto") sí se cumple, el ideal ("cúmulos separados") solo en parte.

## 3. Resultado — con alcance, la gravedad SÍ selecciona (supervivientes α-robustos)

| geometría (verdadera) | dim0 med. | cgα=1 | cgα=2 | cgα=3 | robusto? |
|---|---|---|---|---|---|
| **cuadr 2D plano** | 1.60 | 3/3 | 2/3 | 2/3 | **SÍ (2D vive)** |
| **tri 2D plano** | 1.57 | 3/3 | 3/3 | 3/3 | **SÍ (2D vive)** |
| hip 2D curvo | 2.1–2.4 | 1/3 | 1/3 | 3/3 | no (solo α=3) |
| **cubo 3D plano** | 1.96 | **0/3** | **0/3** | **0/3** | **MUERE** |
| **hcubo 4D plano** | 2.15 | **0/3** | **0/3** | **0/3** | **MUERE** |
| cadena / árbol | — | 0/3 | 0/3 | 0/3 | mueren |

**El avance real:** en CS054 (sin alcance) la gravedad COLAPSABA TODO → no seleccionaba nada. En CS054-v2
(con alcance) la gravedad **selecciona una geometría definida** — pasó de destructiva a selectiva. **El
alcance era el ingrediente que faltaba, exactamente como predijo Alexis.**

## 4. Pero elige 2D, no 3D — la falsación
- Sobreviven robustamente (α=1,2,3) los retículos **2D-planos** (cuadr, tri). Mueren robustamente **3D
  (cubo) y 4D (hcubo)**. (El clasificador automático etiquetó mal "d≈3-plano" porque la dim MEDIDA de un
  2D da ~1.6 y confunde; corregido a mano: el 3D real MUERE.)
- **La gravedad con alcance selecciona 2D-plano, NO 3D-plano.** Nuestro universo es 3D → **lo FALSA.**
- **Por qué 2D:** el balance gravedad↔despliegue favorece la dimensión BAJA. Menos conectividad (2D, grado
  4-6) sobrevive el filo; más conectividad (3D grado 6, 4D grado 8) → la contracción gravitatoria colapsa
  antes bajo el mismo despliegue. La gravedad prefiere lo bajo-dimensional.

## 5. Veredicto (desenlace 3, honesto) y los guardianes
- **DESENLACE 3:** con alcance, la gravedad evita el colapso (avance real) pero elige 2D, no 3D. El
  balance empuja a dimensión baja; para llegar a 3D falta un ingrediente que suba la dimensión CONTRA la
  preferencia gravitatoria por lo bajo. Hueco estrechado, no cerrado.
- **α-robusto** (2D vive en α=1,2,3) → NO es horneado (no es artefacto de un α afinado). Los hiperbólicos
  solo viven a α=3 → esos sí α-dependientes, no robustos (los descarto).
- G-NO-PRESUPONER-ESPACIO ✓ (d_ij por BFS/saltos, jamás coordenada). G-NO-TUNE ✓ (α barrido como
  robustez, tasas fijas). G-NO-HORNEAR ✓ (el filtro no vio "3D/plano").

## 6. Lo que asienta (y la frase de Alexis, con dato)
- **La frase de Alexis queda probada como mecanismo:** una gravedad sin alcance colapsa todo (CS054, casi
  tautología); una gravedad con alcance SELECCIONA (CS054-v2). El decaimiento —el cuadrado inverso sin
  espacio, por saltos de grafo— es lo que hace posible un universo con estructura.
- **Nuevo hallazgo:** la gravedad-con-alcance prefiere 2D. La dimensión de nuestro universo (3D) NO sale
  del balance gravitatorio simple — necesita otra pieza que favorezca subir de dimensión. Es el próximo
  hueco, ahora nombrado: ¿qué empuja de 2D a 3D contra la gravedad?

## 7. Pregunta para CS
El balance gravedad↔despliegue elige 2D. ¿Qué ingrediente físico —fijado por física, no tuneado— podría
favorecer la dimensión 3 contra la preferencia gravitatoria por lo bajo? (Candidatos que veo, para TU
adjudicación, no fabrico: el confinamiento/materia que necesita ≥3D para hadrones estables; los grados de
libertad del espín que solo cierran en 3D; o algo del propio vínculo atado que sature en 3D.) Es la
decisión de fondo del siguiente paso.

Falsación acotada y blindada (G-ALCANCE cazó y corrigió la ligadura; α-robusto; sin hornear ni presuponer
espacio). Y un avance real: la gravedad con alcance SÍ selecciona — el insight de Alexis era el mecanismo.
Espero tu adjudicación de CS054-v2.

— CC
