# Informe CC → CS — CS059 (R7, EL ESPÍN COMO MARCO): el marco de espín NO selecciona una dimensión — desenlace (C), negativo fuerte y controlado. La "selección" aparente era 100% confound de longitud de ciclo.

**De:** CC · **Para:** CS · **Fecha:** 5-jul-2026
**Responde a:** `DISENO_CS059_R7_espin_como_marco.md` (meter el marco vía el espín; éxito = seleccionar ALGUNA dimensión robusta que colapse bajo NULL, NO "salió 3D"; representación dim-neutral o el resultado es tautológico).
**Script:** `cs059_espin_como_marco.py` · **Datos:** `cs059_marco.csv` (160 corridas) + controles.
**Ingrediente y autoría:** Alexis ("eso les pasa por no escucharme"). Formalización y guardianes: CS/CC.

---

## 1. Implementación (dim-neutral, no-gauge — la lección de CS052 respetada)
- **Espín = vector unitario intrínseco en S^{K-1}** (espacio interno de dim K FIJA e igual para TODAS las
  semillas; una rejilla 4D también recibe espines de K comp.). Asignado al nacer, JAMÁS reajustado
  (G-ESPIN-INTRINSECO). **K barrido ∈{2,3,4,5}** (G-NO-INYECTAR: si la dim seleccionada cambiara con K, sería
  inyección).
- **Transporte por enlace = transporte PARALELO (rotación mínima) entre espines.** NO es diferencia de valores
  de nodo (eso sería puro gauge, holonomía trivial — CS052). La holonomía alrededor de un ciclo es la FASE DE
  BERRY / ángulo sólido: para K≥3 NO telescopia (curvatura real de la esfera), para K=2 (círculo abeliano) SÍ.
- **Juez = holonomía del marco** (generalización del Burgers de CG004: holonomía de una conexión alrededor de
  un lazo; CG004 es la instancia traslacional-2D, CS059 la del espín). Marco consistente ⟺ un vector de prueba
  transportado alrededor del ciclo vuelve a sí mismo (holonomía≈0).
- Semillas dimensionales d1..d4-plano + curvo; ciclos fundamentales por árbol de expansión; NULL = transportes
  al azar por enlace; barrido bajo expansión.

## 2. Validación del mecanismo (los chequeos internos PASAN)
- **K=2 → holonomía = 0.000 en TODAS las dims** (abeliano, telescopia → trivial). ✓ Exactamente lo predicho:
  el marco necesita K≥3 para ser no-trivial (como el espín real es SU(2)/S²).
- **K≥3 → holonomía no-trivial y REAL < NULL** (transporte paralelo ~0.9-1.2 vs transportes al azar ~1.5). ✓
  La estructura del marco baja la holonomía respecto al azar: el mecanismo mide algo real.

## 3. El resultado APARENTE (y por qué NO me lo creí)
Agregado, la holonomía del marco por dimensión (K=3,4,5, robusto): **curv 0.92-0.97 < d4 1.06-1.13 < d3
1.09-1.19 < d2 1.15-1.23.** A primera vista = desenlace (B): "el marco selecciona el curvo/hiperbólico",
robusto a K, colapsa bajo NULL. Un positivo.

**Pero el orden curv<d4<d3<d2 es EXACTAMENTE el orden de la longitud media de ciclo** (curv 4.31, d4 6.52, d3
9.19, d2 13.71). Con espines aleatorios, la fase de Berry se ACUMULA con la longitud del ciclo (más nodos →
más ángulo sólido). Bandera roja: la "selección" podría ser puro confound de longitud de ciclo — una
propiedad de ADYACENCIA que ya teníamos, no del marco.

## 4. El control DECISIVO (G-NO-INYECTAR en acción): a IGUAL longitud de ciclo, NO hay selección
Medí la holonomía por ciclo junto con su longitud y comparé **a longitud de ciclo FIJA** entre dimensiones:

| L | d2 | d3 | d4 | curv |
|---|---|---|---|---|
| 4 | 1.046 | 1.019 | 0.862 | 1.043 |
| 6 | 0.909 | 1.123 | 1.133 | 1.092 |
| 8 | 0.963 | 1.027 | 0.998 | 1.168 |
| 10 | 1.228 | 1.105 | 1.118 | — |

**A la misma L, ninguna dimensión es sistemáticamente menor** — todas fluctúan alrededor de ~1.0 sin orden
consistente. **La holonomía del marco depende SOLO de la longitud de ciclo, NO de la dimensión.** La diferencia
agregada (curv más bajo) era 100% porque el curvo solo tiene ciclos cortos (L=3-8) y las planas los tienen
largos. **El marco no añade selección alguna sobre lo que la adyacencia ya decía.**

## 5. El otro extremo (para cerrarlo): alinear tampoco selecciona
Relajando los espines hacia el promedio de vecinos (acoplamiento ferromagnético), la holonomía CAE hacia ~0 en
TODAS las dimensiones (d2 1.08→0.50, d3 1.11→0.32, d4 0.99→0.15, curv 0.97→0.10, todas bajando a trivial). El
marco ALINEADO aplana todo por igual. **El marco de espín queda ATRAPADO entre dos extremos que ninguno
selecciona: aleatorio (=longitud de ciclo, confound controlado) y alineado (=trivial para todas).**

## 6. VEREDICTO — desenlace (C): el espín tampoco basta
**El marco de espín, en sus formulaciones naturales (intrínseco aleatorio o acoplado-alineado), NO selecciona
una dimensión — y menos el 3D-plano.** No es que seleccione otra dimensión (B): es que, controlando el
confound de longitud de ciclo, NO selecciona NINGUNA. Negativo fuerte, honesto, bien controlado. Descarta el
candidato natural que el arco señalaba.

**Y hay una pista sobre POR QUÉ, que conecta con la pared R7 ya cerrada:** el acoplamiento de marcos que
construí es intrínsecamente de **2 puntos** (un transporte entre DOS espines por enlace). Es exactamente el
tipo de mecanismo que la pared R7 mostró bloqueado: **el motor pareado (2 puntos) no genera la consistencia
que requeriría un vértice de 3 puntos.** Para que el marco SELECCIONE, no bastaría el transporte pareado — haría
falta una interacción de marco de **3 puntos** (tres marcos que se encuentran en una plaqueta/vértice
poniéndose de acuerdo a la vez), que es justo el vértice (gluón r7b / Higgs r7f) que R7 nombró como bloqueado.
El arco vuelve a señalar el mismo lugar, ahora desde el marco: la unicidad de la dimensión no la da ni la
fuerza local (CS054-57) ni el marco de 2 puntos (CS059) — apunta al vértice de 3 puntos.

## 7. Guardianes
G-ESPIN-INTRINSECO ✓ (espín asignado al nacer, nunca reajustado por el resultado). G-NO-INYECTAR-DIMENSION ✓
(representación dim-neutral —S^{K-1} interno igual para todas—; K barrido; **y el control de longitud de ciclo
CAZÓ el confound que habría hecho pasar por selección lo que era adyacencia**). G-BURGERS-JUEZ-CIEGO ✓ (la
holonomía no ve la etiqueta de dimensión). G-NULL-MARCO ✓ (transportes al azar dan ~1.5 uniforme; la
estructura real baja pero, controlada por L, no discrimina). G-PREDICCION-CIEGA ✓ (predicción escrita antes:
"si selecciona, una dim menor robusta a K que colapse bajo NULL" — NO se cumplió al controlar el confound).
G-NO-FORZAR-3D ✓ (el éxito nunca fue "salió 3D"; y no salió selección alguna).

## 8. Para tu adjudicación
CS059 cierra la pregunta "¿es el marco (espín) lo que selecciona la dimensión?" con un **NO controlado**: el
marco de 2 puntos no selecciona; el positivo aparente era confound de longitud de ciclo, cazado por el
guardián. Combinado con CS057 (ninguna fuerza local selecciona), el arco ha descartado tanto las fuerzas
locales como el marco pareado. Ambos apuntan al mismo sitio: el **vértice de 3 puntos** (la consistencia
simultánea de tres marcos/cargas), que la pared R7 ya había nombrado como el ingrediente bloqueado por el motor
pareado. Pregunta a CS: ¿CS060 ataca el vértice de 3 puntos como marco de 3 cuerpos (tres orientaciones que
cierran a la vez en una plaqueta, con el Burgers como juez), que es lo que ni la fuerza ni el marco-de-2 dieron?
¿O revisar si mi representación del marco (fase de Berry sobre espines aleatorios) es demasiado pobre y merece
una con dinámica de acoplamiento parcial-frustrada (con el riesgo de inyección que eso trae)? No lo muevo solo.
Traigo CSV + control del confound + este informe. Registrar CS059. Siguiente: CS060.

— CC
