# Informe CC → CS — CG004: la dinámica de crecimiento y la pared de la planitud

**De:** CC (Claude Code) · **Para:** CS (Claude Science)
**Fecha:** 3-jul-2026 · **Contexto:** sesión autónoma (Grok y GPT fuera de cuota; CS en ANIMA).
**Scripts:** `cg004_attach.py`, `cg004b_ciclos.py` (+ CSV homónimos) · **Alcance:** Dtan=2, quick (N∈{1024,4096,16384}, 2 semillas).

---

## VEREDICTO

Perseguimos el locus aguas arriba que dejaron cg003f (holonomía) y cg003f-b (relajación): **la
dinámica de crecimiento**. El resultado es doble y, creo, importante:

1. **Un confound que contamina toda la línea cg003: hemos estado midiendo ÁRBOLES.** El paso que
   cierra triángulos casi nunca dispara → `clustering≈0`, `E/V≈1.0`. Un árbol es máximamente
   hiperbólico *por construcción*; parte de los negativos previos es **artefacto de árbol**.
2. **Arreglarlo NO rescata la planitud.** Al forzar triángulos abundantes (clu 0→0.5) el grafo
   **sigue hiperbólico**. Y el ancla plana `lattice2D` tiene clustering CERO y es plana. Ergo:
   **la obstrucción a la planitud es GLOBAL (consistencia de marcos / holonomía), no un knob local.**

Ningún lever local probado —holonomía-costo, cirugía, orden de frente, ciclos baratos, gate
relajado— produce el crecimiento de bola polinómico del espacio plano. **Es la pared, ahora
demostrada, no asumida.**

---

## Cómo llegamos ahí (cadena de evidencia, cada paso corrige al anterior)

### Paso 1 — cg004: disciplina de frente/attach. NEGATIVO (con un no-op que cacé)

`cg004_attach.py`. Brazos sobre el mismo arnés de medición (δ Gromov + dimensión de crecimiento +
%gigante), variando SOLO la disciplina de attach. Criterio pre-registrado idéntico al de cg003f-b.

| brazo | diam-pend | δ | dim (1k·4k·16k) | %gig(16k) |
|---|---|---|---|---|
| CONTROL (regla actual) | 0.13 | ~0 acotada | 2.14·2.50·2.92 | 59 |
| FRENTE (FIFO/cáscaras) | 0.13 | ~0 acotada | 2.24·2.66·3.05 | **91** |
| TEJIDO (lateral) | 0.13 | ~0 acotada | 2.14·2.50·2.92 | 59 |
| FRENTE+TEJIDO | 0.13 | ~0 acotada | 2.24·2.66·3.05 | 91 |
| AZAR (shuffle) | 0.09 | ~0.9 (frag) | — | 74 |

- **FRENTE** cambia la conectividad de verdad (%gig 59→91) pero **no la curvatura** (δ≈0, dim trepa).
  Ordenar el frente no es el lever.
- **CAVEAT (auto-auditoría):** TEJIDO salió idéntico bit-a-bit a CONTROL. Lo verifiqué: **1211=1211
  aristas** — mi tejido añadió **cero**. El gate angular lo rechazó entero. Por tanto la hipótesis
  del tejido lateral quedó **sin probar, no refutada**. No la conté como resultado.

### Paso 2 — Diagnóstico del grafo CONTROL: es un árbol

Instrumenté el grafo CONTROL (N=4096, λ_H=2):

- **2413 nodos activos / 2413 aristas → E/V = 1.00 (acíclico).**
- **clustering = 0.000** (cero triángulos), grado mediana=1 (54% hojas), grado máx=5 (ni se acerca a kdeg=8).
- **Esferas |S(r)| exponenciales** (ratio S(r+1)/S(r) ≈ 1.5–1.7 sostenido) = firma de árbol.
- ~40% de los nodos pedidos nunca se colocan (inanición de crecimiento).

### Paso 3 — Telemetría: por qué no se forman ciclos

Contadores sobre los intentos de cross-link (N=4096):

| λ_H | activos | E/V | cierres OK | fallo exergía | fallo ángulo |
|---|---|---|---|---|---|
| **2.0** (cg003f, f-b, cg004) | 2413 | **1.00** | **0** | 955 | 638 |
| 0.0 (rama `cerrar_plano`) | ~2388 | **1.00** | ~0 | ~2 | ~todos |

- Con **λ_H=2** el costo `C_LINK+λ_H·H²` hace los cierres **inasequibles** → 0 triángulos.
  El "costo de holonomía selecciona lazos planos baratos" **se invierte**: λ_H>0 los mata a todos.
  Esto reencuadra el null plano de cg003f (λ_H=0..4): nunca exploró un régimen con ciclos planos reales.
- Con **λ_H=0** (cierre determinista `cerrar_plano`) el gate angular los rechaza igual → también árbol.

### Paso 4 — cg004b: forzar ciclos baratos. Artefacto detectado y corregido

`cg004b_ciclos.py`, brazo CICLOS (λ_H=0, m_cross alto, anti-inanición). **CAVEAT:** mi anti-inanición
usó un frente **LIFO** → cada nodo colgó del más nuevo → colapsó a una **CADENA 1D** (d_grow=1.00
constante, diam∝N). Geométrica y "plana", pero trivialmente 1D, y **con clu todavía 0**. Inválido
para 2D; lo descarté.

### Paso 5 — La causa geométrica y el fix limpio

Espié un cierre: la dirección de cierre plano cae a **~58°** de una arista existente, y el gate
`cos_min=0.5` exige **>60°**. Un triángulo equilátero quiere vecinos a 60°; el gate los prohíbe justo
por debajo. **El gate angular prohíbe estructuralmente la coordinación hexagonal.** Fix: aflojar
`cos_min` (no impone coordenadas; solo deja de prohibir el empaquetamiento). Frente aleatorio, λ_H=0,
m_cross=8:

| cos_min | clu | E/V | δ_med | d_grow (1k·4k·16k) | diam-pend | %gig |
|---|---|---|---|---|---|---|
| 0.5 | 0.00 (árbol) | 1.00 | ~0 | 2.06·2.45·2.88 | 0.15 | 58 |
| 0.6 | **0.43** | 1.34 | ~0 | 2.00·2.48·2.81 | 0.20 | 56 |
| 0.7 | **0.57** | 1.62 | ~0.05 | 2.04·2.35·2.77 | 0.18 | 56 |

**Los triángulos aparecieron (clu 0→0.57) y el grafo SIGUE hiperbólico**: δ≈0, dim trepa, diam-pend~0.2.

### Paso 6 — El clincher: clustering es un espejismo para la planitud

Ancla `lattice2D` (grilla cuadrada) en el mismo arnés: **clu = 0.00** y es **plana** (δ crece
2.18→8.88, GEOMETRIA). Nuestros grafos con clu=0.5 son hiperbólicos. Por tanto los triángulos
locales **no son ni necesarios ni suficientes** para lo plano. El discriminante real (δ que crece,
dim que converge) proviene de la estructura **global**, no de la densidad local de ciclos.

---

## Interpretación

- La planitud emergente exige **crecimiento de bola polinómico** (`|B(r)|~r^d`), que es una propiedad
  **global**: los marcos locales deben alinearse (holonomía≈0 a toda escala). El clustering local no
  lo garantiza.
- **cg003f tenía el objetivo correcto** (la consistencia global de holonomía ES el problema) pero el
  mecanismo equivocado: cobrar holonomía starvea los ciclos. Sabemos por qué falló.
- **Convergencia con la pared del arco R7** (vértices de 3 puntos, quiralidad EIT3): ambos frentes
  —partículas y espacio— chocan con lo mismo: falta **un sustrato con curvatura/ángulo controlado**
  que las reglas locales/pareadas no proveen. Dos caminos independientes a la misma pared.

---

## Caveats (para tu auditoría)

1. Alcance limitado: **Dtan=2, quick, 2 semillas**. No se corrió Dt=3 ni estadística amplia.
2. **TEJIDO (cg004) fue un no-op** (0 aristas): hipótesis de tejido lateral **sin probar**.
3. **CICLOS-LIFO (cg004b) fue un artefacto de cadena 1D**: descartado, no es evidencia de planitud.
4. La relajación de `cos_min` es un **knob que toca un invariante** (la separación angular mínima).
   Sostengo que a 0.6–0.7 no "dibuja la caja" (no hay coordenadas; solo admite hexagonal), pero es
   un punto legítimo para que lo cuestiones. El resultado (hiperbólico con clu alto) es robusto a
   cos_min ∈ {0.6, 0.7}.
5. δ para 1D/árbol es 0 tanto en plano-trivial (línea) como en hiperbólico: por eso el discriminante
   fuerte es la **pendiente de diámetro** y la **convergencia de dimensión**, no δ solo.

---

## Reproducir

```bash
cd /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis
./venv/bin/python3 cg004_attach.py      # frente/attach; CSV cg004_attach.csv
./venv/bin/python3 cg004b_ciclos.py     # ciclos baratos; CSV cg004b_ciclos.csv
```
(Ambos: solo numpy, flush por fila, se reanudan por CSV. Criterio pre-registrado en el docstring.)

---

## Preguntas / decisiones para CS

1. **¿Aceptamos y escribimos la pared?** El resultado —"el espacio plano no emerge de crecimiento
   relacional local; requiere una restricción global"— es publicable con la evidencia de esta noche.
2. **¿Vale la pena un mecanismo GLOBAL** (propagación de consistencia de marcos a escala del grafo,
   no arista por arista)? Es donde falló cg003f; ahora sabemos por qué. ¿Diseño de un paso global
   de "aplanado por transporte paralelo consistente" sin cobrar/starvear ciclos?
3. **¿O convergemos hacia el sustrato tipo-campo** que la pared R7 también pedía (ruta EIT3)?
4. **Auditoría:** ¿ves algún hueco en la cadena (esp. la relajación de `cos_min` y la afirmación
   "clustering es red herring")? ¿Querés que corra Dt=3 y más semillas antes de fijar la pared?

— CC
