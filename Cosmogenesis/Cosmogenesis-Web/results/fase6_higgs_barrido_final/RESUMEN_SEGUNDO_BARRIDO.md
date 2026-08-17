# Segundo barrido Fase 6 + diagnóstico: ¿datos o no funciona?

**Fecha:** 2026-07-22  
**Scripts:** `Fase6-Higgs-A-V3-r0alto.py`, `Fase6-Higgs-B-V3-alpha_alto.py`  
**Logs:** mismos nombres en esta carpeta

---

## 1. Resultados del segundo barrido (Mac)

### A — r0 ∈ {5, 8, 12, 20} × u ∈ {0.1, 0.3, 0.7}

| | |
|--|--|
| VIVO (⟨\|Φ\|⟩>0.15) | **0/12** |
| ⟨\|Φ\|⟩ | **0.004–0.006** (peor o igual que r0≤3) |
| k1/k3 | sostenidos (~170–190 / ~13–27) |
| ratio | **0.26–0.42** O(1) |

**Subir r0 no encendió el VEV** en este integrador (difusión + ruido + forma de r(T) actuales).

### B — α ∈ {2, 5, 8, 10} en ventana D/K0 viva + NULL

| | |
|--|--|
| NULL (η=0) | **siempre ~0.333** cuando hay k1 y k3 |
| REAL α↑ | ratio **no baja a 0.00054**; a menudo **sube** o se invierte (>>1) |
| Mejor REAL α alto | ~0.41 (α=2); con α=5–10: 0.5–15 (inestable / GEOM) |

**α grande no produce jerarquía 10⁻³; el NULL sigue demostrando que sin η el ratio es geométrico.**

---

## 2. ¿Falla por nuestros datos o porque “no funciona”?

### Respuesta en una frase

> **Está fallando el mecanismo tal como lo instrumentamos** (y en parte la pregunta “¿perim-cluster + un acople débil da 1/1836?”).  
> **No** es un veredicto de que “la Teoría es falsa”, ni solo “malos datos al azar”.

### Tres capas distintas

| Capa | ¿Qué significa “falla”? | ¿Es nuestro caso? |
|------|-------------------------|-------------------|
| **1. Datos / numerica / grilla** | Seed, L, bugs, k=0, nan | **Parcialmente resuelto** (L=30, F0 PASS, k3 vivo, B/C con NULL). Segundo barrido **no** es ruido: A y B son **reproducibles y sistemáticos**. |
| **2. Instrumento / modelo de juguete** | No implementamos el Higgs real del MS ni el holismo físico pleno; m sigue mezclada con tamaño k; Φ no sostiene VEV; η solo O(1) | **Sí — principal** |
| **3. Hipótesis física fuerte** | “Cualquier asimetría + topología + un medio simple → masa leptón/protón” | **No refutada en general**; **sí acotada**: *esta familia de codificaciones no produce jerarquía 10³* |

### Por qué no es “simplemente no funciona la idea de Higgs”

El MS no dice “el perímetro de un dominio 2D da m_e/m_p”.  
Dice: **fondo ⟨Φ⟩ ≠ 0** + **acoples g distintos** (o dinámicos) → masas.

Nosotros medimos, en la práctica:

| Lo que el MS necesita | Lo que el código hace |
|----------------------|------------------------|
| VEV estable | A: ⟨Φ⟩ ~ 0.005–0.01 y **cae** al subir r0 |
| g que discrimine especies | Una y0 global; m∝ v_k·Σρ o exp(-α η)·Σρ |
| Discriminación fuerte de g o de ⟨Φ⟩_local | v_k1 ≈ v_k3; η1 solo un poco > η3 |
| Resultado: m1 ≪ m3 | Residual **geométrico** ~1/3 o O(1) |

Mientras **Σρ ~ k** domine y el factor “Higgs” sea **casi igual** en k1 y k3, el ratio **tiene** que ser O(1). Eso es aritmética, no mala suerte de seed.

### Por qué no es “solo malos datos”

- F0 dilución: **PASS** estable Meta=Mac.  
- k3/perim8: **reproducible**.  
- B NULL→0.333: **reproducible** → el acople al medio **sí se detecta**.  
- Segundo barrido: r0 alto y α alto **no** abren magia; a veces empeoran.  
Eso es **evidencia negativa del instrumento actual**, no un fluke.

### Entonces, ¿qué está “roto”?

1. **A (Higgs de libro):** el potencial + integración **no mantienen VEV** bajo expansión/difusión. Fallo de **dinámica de Φ**, no de “faltó un CSV”.  
2. **B (fricción):** el medio **sí** cambia el ratio respecto al NULL, pero el contraste η es **débil** (factores O(1), no 10³). Fallo de **potencia del canal**, no de ausencia total de mecanismo.  
3. **Pregunta 1/1836 como éxito de clusters 2D:** con m∝tamaño·(factor O(1)), **estructuralmente no puede** salir 10⁻³ sin Shannon (g distintos a mano o if k).

---

## 3. Cómo leerlo sin desánimo ni autoengaño

| Afirmación honesta | |
|--------------------|--|
| “Nuestros datos están mal” | **No** como explicación principal del segundo barrido |
| “El Higgs del MS no existe en la naturaleza” | **No** lo hemos testeado |
| “Este modelo de juguete + esta m no da jerarquía leptónica” | **Sí** — resultado sólido del barrido |
| “Hay germen de acople al medio (B/C NULL)” | **Sí** — no tirar a la basura |
| “Falta VEV estable + g emergente + respuesta inercial + tríada holística de verdad” | **Sí** — siguiente diseño, no más α a ciegas hacia 1/1836 |

---

## 4. Conclusión operativa

**Falla el “cómo lo codificamos e interrogamos”** (instrumento + observable de masa), de forma **sistemática**.  
**No** falla “porque los números salieron raros una vez”.  
**No** prueba aún que “un medio tipo Higgs en la Teoría sea imposible”.

Siguiente paso útil (no más del mismo barrido fino de α):

1. **A:** ecuación de Φ que **fije** ⟨\|Φ\|⟩≠0 (baño / pozo / ruido equilibrado) y NULL Φ=0 → ratio geométrico bien definido.  
2. **Masa = respuesta a empujón**, misma fórmula, no y0·|Φ|·Σρ.  
3. Dejar de usar **1/1836 como objetivo de sintonía**; usar “Rm se separa del NULL y de 1/3 de forma robusta”.
