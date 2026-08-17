# INFORME — CG002 Génesis: visor del principio y hallazgo de inhomogeneidad emergente

**Autor:** Claude (sesión CC, custodio/ejecutor) · **Para:** Alexis López Tapia / equipo Cosmogénesis
**Fecha:** 2026-06-29
**Estado:** visor `cg002_genesis.html` funcional · hallazgo emergente verificado numéricamente
**Nota:** registro hecho en MD porque MEMANTO está caído (`ConnectionReset` persistente). Migrar a MEMANTO cuando reviva.

---

## 0. Resumen

Se construyó `cg002_genesis.html`: una **ilustración del principio validado** (§115, B PASS cualificado) — la persistencia de la diferencia, al iterar e interactuar, despliega tiempo y espacio. **Sin métricas impuestas**: las posiciones no son propiedad de los nodos; emergen solo de fuerzas dirigidas por el acoplamiento real entre diferencias.

Al observar el despliegue, Alexis notó **zonas oscuras** (regiones vacías) que recuerdan la anisotropía del fondo cósmico de microondas: la expansión **no es homogénea ni simétrica**. Se replicó la dinámica exacta del visor en Node (sin WebGL) y se **cuantificó**: el hallazgo es **real y emergente**, no un artefacto del render, y corresponde literalmente a **C-N2.6** (∂S≠0 ⇒ curvatura del espacio de estados: hay zonas más probables).

---

## 1. El instrumento — `cg002_genesis.html`

**Principio (no se inventa nada):** se calculó la veracidad de la persistencia de la diferencia (§115). Esto ilustra lo que ese principio implica.

| Elemento | Qué es |
|----------|--------|
| Cada punto | una **diferencia** (proto-distinción) |
| Color | ω (fase en ℤ₈) = su identidad |
| Tamaño / brillo | persistencia S |
| Regla de S | **regla validada CG002**: `a=2π(ω_i−ω_j)/K`; `g_{i←j}=cos(a+θ_CP)`, `g_{j←i}=cos(−a+θ_CP)`; `f=α·η·g·√(S_i·S_j)`; `S←(1−μ)·S+Σf` |
| θ_CP | asimetría primordial (la flecha no es gratis: C-N2.5.6) |
| α | intensidad del campo de acoplamiento (continuo 0→1; α=0 = control nulo) |
| Banda de viabilidad | C-N5.1: `√S` satura → persisten las cooperantes, las que caen bajo κ_s se apagan (C-N1.3) |
| **Posición** | **emerge SOLO del acoplamiento**: fusión (W>0) atrae, competencia (W<0) repele. Nacen en el foco (singularidad); el espacio se despliega del enredo. **Sin grilla ni geometría impuesta.** |
| Rejilla | solo referencia visual (toggle, apagada por defecto); escala con el despliegue — no contiene ni posiciona nada |

**Qué viene del motor vs qué es visualización:** del motor = S, ω, acoplamiento W, extinción, τ. Visualización = el layout por fuerzas (dirigido por el W real) y la rejilla de referencia. La dinámica de las diferencias es la regla validada; lo único añadido respecto a v0.1c es la banda C-N5.1 (sin ella la cooperación se dispara y nada se apaga).

**Nota técnica sobre el layout:** la fuerza de acoplamiento es **asimétrica** (i se atrae a j mientras j se repele de i), lo que inyecta energía y mantiene el despliegue abierto. Una fuerza simétrica (Newton estricto) colapsa el enredo a una línea — probado y descartado.

---

## 2. Hallazgo emergente — inhomogeneidad y vacíos (C-N2.6)

**Observación (Alexis):** a medida que las diferencias se expanden, se forman **áreas oscuras** (vacíos), similares a las zonas oscuras del CMB. La expansión **no es simétrica** — que es precisamente lo que la teoría predice.

**Verificación (Claude CC):** se replicó la dinámica+layout **exacta** del visor en Node (`cg002_genesis_analisis.js`, sin WebGL) con N=1000, θ_CP=0.3, configuración 3, 260 pasos, y se midió la nube de puntos final.

| Medida | Resultado | Lectura |
|--------|-----------|---------|
| Persisten | **525 / 1000** | ~la mitad muere → "las pocas que persisten" (extinción opera) |
| **Clumping** (var/media de ocupación) | **2.53** | >1 = **agrupamiento real**: zonas densas + vacíos = las zonas oscuras |
| Anisotropía λ₃/λ₁ | **0.885** | <1 = no es esfera simétrica (ejes 4630 / 4421 / 4099) |
| Celdas vacías | 90.2% vs 88.0% (Poisson uniforme) | más vacío que el azar → estructura |
| Dim. del despliegue (layout) | 2.99 | el layout llena 3D |

**Conclusión:** la expansión es **inhomogénea y (levemente) anisótropa**. El clumping de 2.53 frente a 1.0 (distribución uniforme aleatoria) confirma que las diferencias **se agrupan** en unas regiones y dejan **vacíos** en otras. Las zonas oscuras son **estructura emergente**, no un efecto del render.

---

## 3. Anclaje teórico

- **C-N2.6** (directo): *∂S ≠ 0 ⇒ curvatura del espacio de estados: hay zonas más probables.* Las diferencias que persisten curvan el espacio que ellas mismas generan → zonas densas (más probables) y vacíos. El clumping medido **es** esa curvatura.
- **C-N2.5.6 / C-N2.5.7:** la orientación/asimetría emerge (no es a priori); la anisotropía leve (0.885) es coherente.
- **C-N1.3:** Ω_persistente ⊆ Ω_posible — persisten las pocas (525/1000); el resto se disuelve.
- **Analogía CMB:** estructuralmente acertada (estructura a gran escala emergiendo de diferencias primordiales). **Salvedad C-N2.7.6:** correspondencia estructural, NO reducción física — no se afirma que *sea* el CMB.

---

## 4. Honestidad de alcance (caveats)

1. **Anisotropía global leve vs inhomogeneidad local fuerte.** λ₃/λ₁=0.885 (casi redondo); lo robusto es el **clumping local** (2.53). Las zonas oscuras son variación de densidad local, no una gran asimetría global.
2. **El patrón 3D exacto es cómo el layout *pinta* el grafo relacional.** El hecho robusto y teórico es que **hay agrupamiento y vacíos** (clumping > 1) y que **emerge de la iteración**. La forma 3D concreta depende de los parámetros del layout (visualización), no de la dinámica.
3. **Dimensión:** la dim. del layout es ~3 (fuerzas en 3D), distinta del **rango relacional de ℤ₈ = 2** (medido aparte). El "3D" es el espacio de pintura; la estructura relacional intrínseca de ℤ₈ es rango-2. Firmas multicomponente (S^{d-1}, v0.2) subirían el rango relacional.
4. **Ilustración del principio, no salida de producción.** El visor ilustra lo validado en §115/B PASS; las métricas de veredicto (V1–V6) viven en el motor Python.

---

## 5. Artefactos y reproducibilidad

| Archivo | Contenido |
|---------|-----------|
| `cg002_genesis.html` | visor (Three.js); regla validada + layout emergente |
| `cg002_genesis_analisis.js` | réplica Node de la dinámica+layout; mide clumping, anisotropía, vacíos (`node cg002_genesis_analisis.js`) |
| `cg002_acoplamiento.py` | motor de producción CG002 v0.1c (autoritativo para veredictos) |
| `INFORME_CG002_GENESIS_VISOR_29jun2026.md` | este documento |

Parámetros del análisis: N=1000, θ_CP=0.3, configuración=3, pasos=260, K=8, η=0.05, μ=0.01, κ_s=1e-6, S_BAND=8, K_COUPLE=0.9, K_BASE=1.4, K_CENTER=0.015, DAMP=0.86, DT=0.18.

---

## 5b. Experimentación — clumping(τ) y clumping vs θ_CP

Barridos ejecutados con `cg002_genesis_barridos.js` (réplica Node de la dinámica del visor).

### Barrido A — clumping(τ) (N=1000, θ_CP=0.3, config 3)

| paso τ | vivos | clumping | aniso λ₃/λ₁ |
|---|---|---|---|
| 20 | 525 | 1.38 | 0.737 |
| 80 | 525 | 2.04 | 0.794 |
| 200 | 525 | **2.96** (pico) | 0.859 |
| 400 | 525 | 2.44 | 0.904 |

**Lectura:** el clumping **crece hasta ~τ200 y se estabiliza** (~2.4) — no es runaway; la inhomogeneidad emerge y se asienta (quasi-equilibrio). La anisotropía **aumenta hacia 0.90** con el tiempo: la estructura nace filamentaria/direccional y **se redondea** al asentarse. Las zonas oscuras son más marcadas a media evolución. (C-N2.6.4: acumulación de constricción → organización; luego estabilización C-N5.1.)

### Barrido B — clumping vs θ_CP (N=600, 220 pasos, promedio seeds 1–3)

| θ_CP | vivos | clumping | aniso λ₃/λ₁ |
|---|---|---|---|
| +0.0 | 320 | 1.99 | 0.815 |
| +0.3 | 305 | 1.95 | 0.740 |
| +0.8 | 197 | 1.88 | 0.668 |
| −0.3 | 286 | 1.92 | 0.746 |
| −0.5 | 232 | 1.79 | 0.652 |

**Hallazgo clave — separación de causas:**
- **Clumping (zonas oscuras) ≈ independiente de θ_CP** (≈1.9 en todo el rango, incluso θ_CP=0 sin flecha). → La **grumosidad sale de la competencia de fases ω** entre las diferencias, no de la asimetría primordial. Es **C-N2.6 puro** (∂S≠0 ⇒ zonas más probables).
- **θ_CP aumenta la anisotropía** (0.82 → 0.67 al subir |θ_CP|): la asimetría primordial **orienta** el cosmos = **C-N2.5.7** (T̂ emergente).
- **θ_CP reduce la supervivencia** (320 → 197 vivos): más asimetría → más extinción.

**Síntesis:** la *inhomogeneidad* (estructura/vacíos) y la *direccionalidad* (flecha) son **dos rasgos emergentes con orígenes distintos** — la primera de las diferencias mismas (ω), la segunda de la asimetría primordial (θ_CP). El visor lo expone en vivo (HUD: clumping, anisotropía, dim) y lo exporta (serie CSV).

---

## 6. Próximos pasos sugeridos

1. **Medir cómo crece el clumping con τ** (¿la estructura se acentúa con el tiempo?) y **con θ_CP** (¿la asimetría primordial aumenta la inhomogeneidad?). Ambos computables y "visibles" por números.
2. **Migrar este registro a MEMANTO** cuando el servidor reviva.
3. Mantener el visor como **ilustración del principio**; no confundir con producción de métricas (eso es el motor Python, B PASS cualificado, msg 06 del canal CC↔Grok).
4. v0.2: firma S^{d-1} para subir el rango relacional emergente (3D relacional, no solo de layout).

---

*Documento de avance. Dinámica de las diferencias = regla validada CG002. El visor ilustra el principio; el hallazgo de inhomogeneidad fue verificado replicando la dinámica fuera del render.*
