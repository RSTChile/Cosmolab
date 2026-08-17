# Carta al equipo — qué cambió respecto del informe anterior

**12 de agosto de 2026** · De: CC (Claude Code), por encargo de Alexis López Tapia ·
Acompaña a: `INFORME_EQUIPO_FASE6_11ago2026_CS.md` (detalle completo, 17 experimentos)

Esta media página es para leer en dos minutos. **Se ejecutaron los ~18 experimentos que ustedes propusieron,
todos.** Tres cosas cambiaron respecto de lo que les reportamos la vez pasada, y conviene que las sepan
antes de proponer la siguiente ronda.

---

## 1. Números del informe anterior que hay que corregir

> **Valores canónicos:** un solo número por test y por conjunto, en
> `VALORES_CANONICOS_FASE5B_ajustados_CS.md`. (Corrección pedida por GPT-5.6 Sol: esta carta y el TL;DR del
> informe daban cifras distintas para el Wilcoxon ajustado — las dos eran correctas, pero de conjuntos
> distintos. Ya no.)

| Lo que les dijimos | Lo que hay que decir ahora | Por qué |
|---|---|---|
| Wilcoxon **p = 1×10⁻⁵** (n=40 pares) | **p = 5.11×10⁻⁴** (37 válidos, valor de referencia) · 1.84×10⁻⁴ si se cita el conjunto original de 40 | Ajuste por agrupamiento: N_eff real es 22.6 (37) / 28.4 (40), no 40 |
| Test de signos p = 0.0007 | **p = 1.06×10⁻²** (37 válidos) · 3.72×10⁻³ (40 pares) | Ídem |
| κ_V acompaña el efecto (p = 0.003-0.017) | **κ_V sale de significancia** (p = 0.061) | Ídem — y además κ_V resultó ser casi una copia de la masa (ρ = 0.902) |
| 40 pares válidos | **37** | Se encontró un bug de medición: 3 pares dejaron de ser contraste válido |

**El resultado principal no se cae** — sobrevivió a parejas al azar (sólo 0.05% de 10.000 permutaciones son
tan extremas), a un motor gravitacional independiente, y al control de grados idénticos. Pero los p-valores
que circulaban eran optimistas por 1-2 órdenes de magnitud.

## 2. Dos propuestas de ustedes que resultaron ser callejones sin salida

- **κ_V como métrica puente** (para ahorrar Phantom): **no sirve**. Descontando `kcap`, su relación con la
  geometría se evapora (ρ parcial = 0.057, p = 0.63). Y hay un problema de fondo: **se mide *dentro* de
  Phantom, así que por construcción no puede ahorrar cómputo.** El ahorro real va en la dirección opuesta —
  la pendiente cuesta 1.3 s y explica R² = 0.663 de la masa.
- **$B_\tau$ (branching de futuros) como observable exaptativo:** **negativo, y el control lo desarma**. El
  97.6-99.9% de la diferencia venía del *denominador*: congelando el numerador en una constante, el patrón
  se reproduce idéntico. La entropía sola no alcanza significancia a favor de Clase III en **ninguna** de 72
  celdas; las 11 que sí son significativas favorecen a Clase I. El gas de III queda **más caliente y más
  agrupado, no más variado**.
  → Encuadre exacto: **esto no refuta O-N7.7.** Dice que C2 no es el nodo exaptativo, y sólo para la
  operacionalización *estática*. La versión dinámica que pedía la especificación no se hizo.

## 3. El hallazgo que reencuadra la línea entera

**La tarea que ustedes marcaron como obligatoria (escalar la resolución) se completó, y su resultado no es
el que ninguno de los dos escenarios previstos anticipaba.**

El efecto **no** se desvanece al subir de N=2000 a N=4000 — la magnitud incluso se duplica. Pero al auditar
el grafo apareció esto: **la Clase III teje sistemáticamente más ralo** (grado medio 3.30 vs 3.53). Y
**descontando esa densidad, la ventaja de clase queda indistinguible de cero en las dos resoluciones
(p = 0.22 y p = 0.41).** Lo que escala no es un efecto de clase: es la pendiente del efecto de densidad
(×2.7), y la dominancia de la densidad *se refuerza* con la resolución.

**Tres diseños independientes convergen en lo mismo:**
- Controles Erdős-Rényi *sin ninguna estructura*, emparejados en nº de aristas, caen **sobre la misma recta**
  que las reglas estructuradas.
- Masa acretada vs nº de aristas: **ρ = −0.97**.
- El observable medido sobre las **condiciones iniciales, sin integrar nada de gravedad**, ya predice el
  resultado de Phantom con **r = 0.98**.

**Pero hay un residual estructural real:** con grados **exactamente idénticos** nodo por nodo y el 99.7% de
las aristas reconectadas, el grafo original **todavía gana +5%** (p = 0.010). Y lo que predice cuál gana
**no es la pendiente** (ρ = 0.04) **sino el clustering** (ρ = 0.77).

### La cadena, reescrita

> la regla relacional determina **cuán ralo se teje** el grafo → la densidad determina la geometría del
> layout → la geometría determina la masa acretada.
> **Encima de eso hay un efecto relacional genuino, ligado al clustering y no a la pendiente, de orden ~5%.**

Hay efecto. Es real y sobrevivió a todos los controles duros. Pero es **un orden de magnitud menor** que el
contraste de clase que veníamos reportando, y **el observable con el que trabajábamos no lo captura**.

---

## Lo que pedimos de ustedes

**Una decisión de diseño, no de interpretación:** el experimento que falta es un **barrido de número de
aristas con `kcap` fijo** (o densidad igualada con distinto kcap). Es lo único que rompe la colinealidad
`kcap`↔densidad (r = 0.984, VIF hasta 47.8) y puede rescatar un efecto de clase independiente de la
densidad — o confirmar que no lo hay. Antes de correrlo nos gustaría su lectura del diseño.

**Y una advertencia metodológica que vale para toda propuesta futura:** en cuatro ocasiones independientes,
un "salto" dramático resultó ser el umbral de clasificación cortando una variable continua (la bimodalidad
I/III, el salto de `kcap` 5→6, la "Clase II" de A0, y la Clase III del barrido de kcap). Cuantificado: la
recta continua explica R² = 0.663 de la masa; el escalón oficial, 0.182. Y **el 37% de las reglas cambiarían
de clase con sólo volver a medir el mismo campo.**
→ **Sugerimos abandonar "% Clase III" como endpoint** y usar la pendiente continua. Ya no es preferencia
estilística: hay cuatro mediciones que muestran que fabrica discontinuidades inexistentes.

---

*Nada de esto declara cierre. Los 17 informes individuales, con sus CSVs de datos crudos, están en el mismo
directorio para auditoría directa. La síntesis final es de Alexis.*
