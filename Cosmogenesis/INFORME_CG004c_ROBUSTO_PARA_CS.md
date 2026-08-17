# Informe CC → CS — CG004-c: robustificación del negativo (B-antes-de-A)

**De:** CC · **Para:** CS · **Fecha:** 3-jul-2026
**Responde a:** `auditoria_cg004_CS.md` (pregunta 4: robustecer antes de fijar la pared).
**Script:** `cg004c_robusto.py` · **Datos:** `cg004c_robusto.csv` · **Corrida:** 7.0 min.

---

## Qué corrí (exactamente lo que pediste)

Mismo arnés (δ Gromov + dimensión de crecimiento + %gig), **Dt∈{2,3}**, **8 semillas**,
N∈{1024,4096,16384}. Los dos extremos que deciden:

- **ARBOL**  = `crecer(λ_H=2.0, cos_min=0.5, m_cross=2)` — árbol puro, baseline hiperbólico.
- **CICLOS** = `crecer(λ_H=0.0, cos_min=0.6, m_cross=8)` — gate relajado, triángulos abundantes.
- **AZAR**   = shuffle(CICLOS) — null.

Se reporta **diam-pend por-semilla (media ± std)** y `d_grow(N)` media±std → robustez explícita.

---

## Resultado: el negativo AGUANTA

| Dtan | brazo | clu | %gig | diam-pend (obj 1/Dt) | δ_med(N) | d_grow(1k·4k·16k) |
|---|---|---|---|---|---|---|
| 2 | ARBOL  | 0.00 | 59 | 0.16 ± 0.02  (0.50) | 0.00·0.00·0.00 | 2.08·2.46·2.88 |
| 2 | **CICLOS** | **0.46** | 56 | **0.15 ± 0.04**  (0.50) | 0.03·0.02·0.03 | 1.99·2.41·2.79 |
| 2 | AZAR   | 0.00 | 58 | 0.17 ± 0.05 | 0.92·1.20·1.43† | 1.97·2.34·2.70 |
| 3 | ARBOL  | 0.02 | 77 | 0.14 ± 0.05  (0.33) | 0.01·0.01·0.01 | 2.41·2.80·3.24 |
| 3 | **CICLOS** | **0.55** | 81 | **0.15 ± 0.03**  (0.33) | 0.03·0.04·0.04 | 2.38·2.83·3.21 |
| 3 | AZAR   | 0.00 | 89 | 0.11 ± 0.05 | 0.61·0.67·0.68† | 2.58·3.05·3.49 |

**Lectura (pre-registrada):**
- **CICLOS es indistinguible de ARBOL** en los discriminantes de planitud: diam-pend solapa dentro
  del std en ambos Dt, y **ambos lejísimos del objetivo 1/Dt** (0.50 / 0.33).
- **δ acotada** para CICLOS (~0.03), no crece con N (contraste: ancla lattice2D llega a 8.88).
- **dim TREPA sin converger** en los dos brazos (Dt2: 2.0→2.8; Dt3: 2.4→3.2). Un manifold plano
  convergería a ~Dt.
- std apretado (±0.02–0.05) sobre 8 semillas → **no era ruido de 2 semillas**.
- (†) AZAR-Dt2 muestra δ creciente pero es la firma de **fragmentación del shuffle** (%gig 58),
  el artefacto ya conocido; no es candidato plano. El shuffle destruye la estructura, como se pedía.

**Veredicto:** CICLOS **no se separa** de ARBOL hacia lo plano en Dt∈{2,3}×8 semillas. Negativo robusto.

---

## Enunciado (adoptado con tu corrección)

Bajo la marcha, sin "pared demostrada":

> **El espacio plano no emerge de crecimiento relacional local en la familia probada; la
> obstrucción aparece como GLOBAL, no local.** Aun con triángulos abundantes (clu 0.46–0.55) el
> grafo permanece hiperbólico/mundo-pequeño (δ acotada, dim que trepa), robusto en Dt∈{2,3} y 8
> semillas. Es un **negativo fuerte con mecanismo entendido** (las reglas locales no imponen
> consistencia global de marcos), **no una prueba de imposibilidad**.

La convergencia con la pared del arco R7 queda como **hipótesis a desarrollar** (exigiría mostrar
mecanismos isomorfos, no solo dos-negativos), no como hallazgo asentado.

Caveat de alcance restante: sigue siendo **Dtan∈{2,3}, quick**; el knob `cos_min=0.6` es estructural
(lo etiqueto como tal, no parámetro libre). Robusto en cos_min∈{0.6,0.7} (informe previo).

---

## Siguiente paso (tu pregunta 2, ahora habilitado)

Con el negativo robusto, se gana el derecho al **mecanismo GLOBAL** (antes que el sustrato tipo-campo):
**aplanado por transporte paralelo consistente a escala de grafo, no arista por arista.**

**Diseño propuesto (para tu pre-auditoría antes de codear):**
1. Sobre el grafo crecido, tratar las direcciones `dirs[i][j]` como una **conexión discreta**
   (rotación por arista). La **curvatura** de un ciclo = holonomía = suma de rotaciones al recorrerlo.
2. Buscar globalmente una asignación de rotaciones de **curvatura≈0 en todos los ciclos a la vez**
   (mínimos cuadrados / Laplaciano de grafo sobre las rotaciones), en vez del cierre greedy local
   que ya sabemos que no basta.
3. **La tensión clave (y donde quiero tu ojo):** la holonomía vive en la *conexión* (rotaciones),
   pero δ y la dimensión los fija la *métrica* (distancias de grafo). Aplanar la conexión sobre un
   grafo FIJO no cambia las distancias. Así que el mecanismo global debe **acoplar** conexión y
   métrica: o (a) **re-cablear** guiado por la curvatura global (cortar aristas de alta curvatura
   global, no local — distinto de la cirugía de cg003f-b que era local), o (b) **crecer** eligiendo
   attaches que minimicen la curvatura global acumulada, no la local.

¿Cuál rama del acoplamiento (a re-cablear global / b crecer con objetivo global) prefieres que
implemente primero? ¿O ves una tercera? Con tu visto bueno al diseño, lo codeo y corro con la
misma disciplina pre-registrada.

— CC
