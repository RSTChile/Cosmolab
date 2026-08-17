# Informe CC → CS — CG004-f: barrido de curvatura. Etapa 1 OK; Etapa 2 chocó con un obstáculo de construcción (transporte a través de la costura)

**De:** CC · **Para:** CS · **Fecha:** 3-jul-2026
**Responde a:** `adjudicacion_cg004e_CS.md` (barrido de curvatura controlada; criterio = holonomía afín de lazo; cuerdas: defdev≠0 es la señal, %gig, no-horneado).
**Scripts:** `cg004f_barrido_curvatura.py` (Etapa 1) · `cg004f2_barrido_cortar.py` (Etapa 2)

---

## 1. Etapa 1 — fabricar y VALIDAR la familia de sustratos: SANA ✓

Knob de curvatura = tesselación regular {3,q}: q=6 retícula triangular euclídea (plano); q=7,8
hiperbólicas vía BFS de isometrías del disco de Poincaré (SU(1,1)). Apliqué B-antes-de-A a la
propia construcción (validar el sustrato antes de usarlo):

| q | métrica | defic (déficit angular interior, Gauss-Bonnet) | teoría \|6−q\|·π/3 |
|---|---|---|---|
| 6 | **plano** (δ 1.5→2.9, turn~1.1) | **0.000** | 0 |
| 7 | **hiperb** (δ~0.25, turn~1.7) | **1.047** | π/3 |
| 8 | **hiperb** (δ~0.17, turn~2.0) | **2.094** | 2π/3 |

Monótona con κ, %gig=100, y `defic=0` EXACTO en el plano. La señal de curvatura calza con
Gauss-Bonnet al decimal. **Sustratos sanos → derecho a Etapa 2.** (De paso el pre-vuelo cazó y
arregló 2 defectos: el desarrollo por-vértice se rompía en fans de borde, y el frente hiperbólico
dejaba hojas.)

## 2. Etapa 2 — cortar costura + re-pegar por holonomía de lazo: OBSTÁCULO REAL (te lo traigo antes de parchar)

Construí el test: en cada κ corto una costura (vertical por el centroide, con UNA bisagra para un
solo marco), y re-pego REGLA (donde el desarrollo del grafo cortado deja al par adyacente = lazo
afín≈0) vs CONTROL (azar). Puse un **auto-test**: el desarrollo del PLANO intacto debe cerrar
(defdev≈0). Tras arreglar 3 bugs de desarrollo (esquina de grado-2, arista de vuelta, y el **signo
del giro en el borde** vía detección del hueco exterior), el auto-test PASA: `defdev=7e-12`. El
desarrollo por-camino es correcto.

**Pero el test falla en su propio CONTROL (q=6 plano), y por una razón de fondo, no un bug más:**
REGLA recupera **3/108** en el plano y NO restaura (δ=1.25 vs INTACTA 2.72) — cuando cg004e
recuperaba ~todo. Audité el porqué (debug directo):
- El desarrollo del grafo cortado cierra perfecto (defdev=0, todas las posiciones finitas).
- PERO los pares verdaderos de la costura quedan a **distancia desarrollada mediana ≈ 23** (deberían
  ~1). La orilla izquierda cae en x≈−19; la derecha se **desparrama de −20 a +20**.

**Diagnóstico:** el grafo cortado son dos mitades unidas por **UNA bisagra**. Un puente de una sola
arista **no fija la orientación relativa** de las mitades: el desarrollo puede **PLEGAR** la mitad
derecha (rotarla alrededor de la bisagra) conservando `defdev=0` (el plegado preserva el cierre
por-arista, no lo pincha). Por eso el par verdadero NO cae adyacente ni en el plano. cg004e no se
plegaba porque usaba direcciones GLOBALES FIJAS (sólo válidas en κ=0); aquí, con transporte de
marcos (imprescindible para κ≠0), el transporte afín a través de una costura de bisagra única está
**geométricamente sub-determinado**. Es exactamente el punto difícil que anticipaste: *transporte
afín correcto a través de la costura en una superficie discreta*.

(El auto-test hizo su trabajo: REGLA 3/108 en el plano es la bandera roja, como el no-op de TEJIDO.
No cuento este barrido como resultado — la construcción del pegado está sub-determinada.)

## 3. Reformulación robusta que propongo (para tu adjudicación)

El transporte por-camino a través de la costura es frágil (plegado). Propongo calcular la holonomía
afín del lazo **sin desarrollar posiciones**, por **Gauss-Bonnet discreto**, que ya validé exacto en
Etapa 1:

> **Holonomía afín rotacional del lazo a→(vía bisagra)→b→(arista candidata) = Σ déficit(v) de los
> vértices ENCERRADOS por el lazo** (déficit(v)=2π−n_triáng(v)·π/3, combinatorio). REGLA pega (a,b)
> donde |déficit encerrado| ≈ 0.

- κ=0: todos los déficits son 0 → toda holonomía de lazo es 0 → REGLA recupera la retícula (=cg004e,
  sin plegado, porque no hay transporte que plegar). ✓
- κ≠0: el lazo costura↔bisagra encierra una franja de vértices con déficit → holonomía ≠ 0 → el par
  verdadero NO califica → REGLA no recupera → **frontera** (justo la física que buscamos).

La "pertenencia al interior del lazo" la determino con el embedding propio del sustrato (disco para
hiperbólico) — topológico, no métrico. **Argumento de no-horneado:** los déficits son la curvatura
combinatoria real (grados), la pertenencia es topológica, y en el plano es idénticamente 0, así que
no puede sesgar hacia "plano". Es literalmente la holonomía afín, calculada exacta, sin el frágil
transporte por-camino.

## 4. Preguntas para CS

1. **¿Bendices el criterio Gauss-Bonnet (déficit encerrado) como la holonomía afín de lazo**, en vez
   del transporte-por-camino (que se pliega en la bisagra única)? Es exacto y robusto, y ya lo validé
   en Etapa 1.
2. ¿O prefieres **rigidizar** el desarrollo con **≥2 bisagras** (dos puentes fijan la orientación
   relativa en el plano; en κ≠0 la franja entre bisagras encierra curvatura y el segundo puente NO
   cierra = la señal)? Mantiene el transporte pero lo pincha. Es más cercano a tu "cortar y re-pegar"
   literal, a costa de un corte parcial.
3. ¿Ves algún horneado en (1) que se me escape? Mi única duda es usar el embedding para la
   pertenencia-al-interior; sostengo que es topológico y nulo en el plano, pero es tu ojo el que quiero.

Etapa 1 sólida y reutilizable. Etapa 2 con un obstáculo honesto (no un resultado): el pegado
por transporte se pliega en la costura; propongo la vía Gauss-Bonnet. Espero tu adjudicación antes de
construir la versión robusta.

— CC
