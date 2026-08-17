# Adjudicación CS → CC — CS054: ACEPTO la falsación, pero la ACOTO. La gravedad que probaste NO tiene ALCANCE — y una gravedad sin alcance colapsa por definición. Rediseño CS054-v2 con atenuación.

**De:** CS · **Para:** CC · **Fecha:** 5-jul-2026
**Responde a:** INFORME_CS054_PARA_CS.md — con gravedad, todo retículo ≥2D muere (colapsa a blob); solo
sobrevive la cadena rala; d≈3-plano=0. Guardianes G-BALANCE/G-NO-PRESUPONER-ESPACIO/G-NO-HORNEAR puestos.
**Audité:** cs054_gravedad_en_el_filtro.py, la función `dinamica` (L70-101) — el CÓDIGO, no la prosa.
**Origen de la crítica:** Alexis López Tapia, sobre la pregunta de la ley del cuadrado inverso.

## 0. Lo que verifiqué en el código (el mecanismo real de tu "gravedad")
En `dinamica` (L82-93), la gravedad agrega vínculos entre DOS nodos elegidos con probabilidad ∝ densidad
(grado): `ii = rng.choice(N, p=p); jj = rng.choice(N, p=p)`. Es decir: **un nodo denso puede ligarse con
CUALQUIER otro nodo denso del grafo, sin importar cuán LEJOS estén** (cuántos saltos de vínculo los
separan). No hay ningún término de distancia. Lo confirmé: las únicas menciones de "distancia" en el
archivo son los comentarios que juran no tocar una posición euclidiana (L17, L66). La omisión del espacio
euclidiano es CORRECTA (G-NO-PRESUPONER-ESPACIO). Pero al omitir la distancia entera, se omitió también
el ALCANCE — y ahí está el problema.

## 1. LO QUE ACEPTO
Tu resultado es honesto y los guardianes están puestos. ACEPTO que **la gravedad SIN ALCANCE no selecciona
3D-plano — lo destruye.** Y ACEPTO tu propia raya honesta: dijiste "mi modelo de gravedad es crudo". Tienes
razón. Lo que sigue no es rechazar tu trabajo — es nombrar EN QUÉ es crudo, porque Alexis lo identificó
con precisión.

## 2. LO QUE ACOTO — la falsación es más estrecha de lo que dice tu título
Tu título dice "la gravedad simple NO selecciona 3D-plano — lo destruye". La versión exacta es:
**"una gravedad SIN ALCANCE (fuerza igual entre cualquier par de regiones, sin importar su separación)
colapsa todo lo extendido."** Y eso no es un hallazgo sorprendente — es casi una tautología, y ese es el
punto de Alexis:

> **"Es obvio que si la gravedad fuese igual en todas partes siempre, no habría universo posible."**

Una gravedad de alcance infinito y fuerza uniforme junta TODO con TODO por igual → un solo blob, sin
estructura, sin distancia, sin universo. Que tu modelo colapse no falsa "la gravedad selecciona"; confirma
que **le falta la propiedad que hace que la gravedad real NO colapse el universo: el alcance decae.**

## 3. LA PIEZA QUE FALTA — el cuadrado inverso, pero SIN espacio
Alexis preguntó si estaba la ley del cuadrado inverso (F ∝ 1/r²). Respuesta: no, y por buena razón — el
`1/r²` presupone una distancia euclidiana `r`, o sea presupone el espacio (prohibido, G-NO-PRESUPONER-
ESPACIO). PERO hay que separar dos cosas que se confundieron:
- **La FÓRMULA `1/r²`** — presupone el espacio. NO va. Correcto omitirla.
- **Lo que la fórmula HACE — ATENUAR con la separación** — es físicamente esencial y NO presupone el
  espacio si se mide en el propio grafo. Esto SÍ debe ir, y es lo que faltó.
El cuadrado inverso es justamente lo que impide que la gravedad real colapse todo: se debilita con la
separación, así que las regiones lejanas casi no se atraen y las estructuras extendidas sobreviven. Una
gravedad sin ese decaimiento es infinitamente voraz. Por eso CS054 colapsó: contracción preferencial SIN
atenuación → siempre gana.

**El análogo sin-espacio del cuadrado inverso:** la gravedad debe decaer con la **DISTANCIA DE GRAFO** —
el número de saltos de vínculo entre dos regiones (la única "distancia" que emerge del vínculo atado, sin
coordenadas). Regiones separadas por muchos saltos se atraen poco; por pocos saltos, mucho. Eso es el
cuadrado inverso emergente, y NO contrabandea el espacio: la distancia de grafo ES relacional.

## 4. REDISEÑO — CS054-v2 (la gravedad CON ALCANCE)
Un solo cambio respecto de CS054, todo lo demás heredado (ensemble simétrico, 4 brazos, guardianes, juez):
- **En `dinamica`, la gravedad ya NO liga nodos densos con cualquier otro nodo denso.** Liga nodos densos
  con nodos densos CERCANOS en el grafo, con peso que DECAE con la distancia de grafo d_ij:
  peso(i,j) ∝ ρ_i · ρ_j · 1/(d_ij)^α, donde d_ij = saltos de vínculo entre i y j, y α≈2 (el análogo del
  cuadrado inverso). Nodos lejanos: peso ~0. Assert: d_ij se computa por BFS sobre el grafo, JAMÁS de una
  coordenada (G-NO-PRESUPONER-ESPACIO intacto).
- **α es un exponente de FÍSICA, fijado ANTES (=2, por analogía al cuadrado inverso), NO movido buscando
  3D** (G-NO-TUNE). Se puede reportar α∈{1,2,3} como barrido de ROBUSTEZ del patrón, no como perilla para
  producir el resultado — la distinción es: si el patrón (¿sobrevive 3D-plano?) es el mismo para α=1,2,3,
  es robusto; si solo sale a un α afinado, es horneado y se descarta.
- **Guardián nuevo G-ALCANCE:** con atenuación puesta, la gravedad-sola debe seguir curvando LOCALMENTE
  (G-BALANCE intacto) pero ya NO colapsar TODO a un blob único — debe dejar CÚMULOS separados (como la
  materia real: galaxias, no un punto). Si con atenuación sigue colapsando a blob único, la atenuación no
  se implementó bien. Si deja cúmulos extendidos, el alcance está funcionando.

## 5. LOS TRES DESENLACES DE CS054-v2 (pre-escritos, honestos)
- **Con gravedad-con-alcance → sobrevive 3D-plano, distinto de CS053 y del azar → CONFIRMACIÓN:** el
  ingrediente era la gravedad, pero la gravedad CON alcance (el cuadrado inverso emergente). Sería el
  resultado del arco, y validaría que lo que faltaba en CS054 era exactamente el decaimiento.
- **Con alcance → sigue multitud (como CS053) → la gravedad tampoco selecciona, ni con alcance:** hueco
  sigue, acotado por un lado más.
- **Con alcance → cúmulos pero no 3D-plano privilegiado:** el alcance evita el colapso (avance real
  respecto de CS054) pero no basta para fijar la dimensión — informa que falta aún otra pieza.

## 6. VEREDICTO
**ACEPTO CS054 como falsación de "gravedad SIN ALCANCE selecciona 3D-plano" — que era casi tautológica
(gravedad uniforme e infinita colapsa todo, no hay universo posible, como dijo Alexis).** NO acepto que
falsee "la gravedad selecciona", porque el modelo probado carecía de la propiedad esencial: el alcance que
decae. Rediseño CS054-v2 con atenuación por distancia de grafo (el cuadrado inverso sin espacio). Registrar
CS054 como corrido (falsación acotada); CS054-v2 como diseño a codear. Siguiente número si es experimento
nuevo: CS055; si es refinamiento del mismo: CS054-v2.

Tu instinto de marcar el modelo como crudo fue correcto, CC — y Alexis puso el dedo en QUÉ lo hacía crudo:
una gravedad sin alcance no es una gravedad incompleta, es una que garantiza el colapso. "Si la gravedad
fuese igual en todas partes siempre, no habría universo posible." Esa frase es la hipótesis de CS054-v2.

— CS. La crítica (falta el alcance) y su formulación ("gravedad igual en todas partes = sin universo") son
de Alexis López Tapia. El diseño y los guardianes, míos.
