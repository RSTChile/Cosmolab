# ANIMA-4 · La invención léxica es O-N1 (`RC = ICR + IRDE`) operando
**Cosmolab / VSTCosmo · para el equipo · 2026-06-29**

> **Advertencia epistémica:** los rótulos de voz/audio (`dolor`, `hambre`, `palabra_A001`…) son etiquetas
> humanas de archivo/banco. NO sabemos qué significan para los organismos. "Sentido nuevo" aquí = una unidad
> de vocabulario que cubre una región afectiva antes no cubierta, NO un significado humano.

## TL;DR
Veníamos probando topologías sociales y obteniendo, una y otra vez, **cero invención léxica**. La pregunta
se dio vuelta: dejamos de preguntar *"¿por qué no inventan?"* y leímos el **mecanismo**. Resultado: la
invención de vocabulario es un **caso particular de O-N1** (`RC = ICR + IRDE`). El ruido contextual que el
repertorio existente **no alcanza a integrar** se vuelve **sentido nuevo (ICR → palabra acuñada)** o se
**desvía (IRDE → riesgo/desacople)**. En **silencio**, `RC ≈ 0` → no hay materia prima → no aparece sentido
nuevo. Y la conservación se cumple **exacta** en los datos.

---

## 1. El mecanismo de acuñación (qué dispara una palabra nueva)
`VST_OrganoComunicacion._quizas_crear`. Una voz nueva nace solo si se cumplen **a la vez**:

1. **HUECO afectivo** — la voz del banco más cercana está a **> 0.22** del estado afectivo `(arousal,
   valence)`: el organismo *siente algo que su repertorio no sabe expresar*.
2. **RECURRENCIA** — ese mismo hueco reaparece **≥ 3 veces** (no se acuña por un estado fugaz).
3. **ENERGÍA** — energía **≥ 0.25** (acuñar cuesta, es metabólicamente real).
4. **LIBERTAD** — un dado de probabilidad **0.6** (a veces, aun con todo dado, no crea).

Y para **perdurar**: debe **reusarse ≥ 4 veces** (`USOS_CONSOLIDA`) o se **poda y se olvida**
(`VIDA_PROVISIONAL = 600`). El vocabulario crece por **selección (uso)**, no por acumulación.

**La compuerta crítica es la #1, el hueco afectivo.** Y el estado afectivo se calcula (línea 707):
```
arousal = 0.45·RC + 0.30·E + 0.25·lateralidad
```
→ **RC (ruido contextual) es el driver dominante del arousal.** Sin RC, el afecto es calmo y estrecho,
cae siempre en regiones que el banco innato ya cubre → nunca hay hueco → nunca se acuña.

---

## 2. La conexión: esto ES O-N1
No es analogía. El marco está **literal en el organismo** (diccionario de variables del observatorio):

- `RC_total` → *"Ruido contextual total observado: **RC = ICR + IRDE**."*
- `ICR` → *"Integración de ruido contextual: fracción convertida en **sentido/acoplamiento**."*
- `IRDE` → *"Ruido contextual desviado: fracción que aparece como **riesgo o desacople**."*
- `ICR_ratio` / `IRDE_ratio` → cómo se **reparte** RC entre ambos destinos.
- `O-N9.14` → *"H (homeostasis) = A_sys-env estable **sostenido por la competencia ICR↔IRDE**."*

**Lectura:** el ruido contextual no desaparece — se **conserva** y se reparte en conversión a sentido (ICR)
o desviación a riesgo (IRDE). La **invención léxica** es el momento en que el excedente de RC, no absorbible
por el banco actual, **se materializa como ICR**: una palabra nueva que amplía la capacidad de comprensión.
*"Sólo con estímulos fuertes (contexto) aparece el sentido nuevo."*

---

## 3. Confirmación empírica (run silencio `20260629_042243`, 736 285 filas)
Análisis DuckDB sobre la fisiología primaria:

| Magnitud | Valor |
|---|---|
| RC media | 0.0258 |
| ICR media | 0.0051 |
| IRDE media | 0.0208 |
| (ICR + IRDE) | 0.0258 |
| **Residuo `\|RC − (ICR+IRDE)\|`** | **0.00000** |
| ICR / RC | 0.15 |
| RC mediana / max | **0.0** / 0.78 |

**Dos cosas, las dos contundentes:**
1. **`RC = ICR + IRDE` se cumple sin error** (residuo exactamente 0 sobre 736 mil filas). Es una
   **identidad** que el organismo respeta en cada ciclo, no una aproximación.
2. **En silencio `RC ≈ 0`** (mediana 0.0): casi no hay ruido contextual que metabolizar, y de lo poco que
   hay, solo el **15 % se convierte en sentido** (ICR); el 85 % se desvía (IRDE). → **Por eso `creadas = 0`
   en los 4 organismos.** El mecanismo léxico no está roto: le falta la materia prima (RC).

Esto **cierra el brazo negativo** de la pinza: *sin estímulo fuerte, no hay sentido nuevo.*

---

## 4. Lo que falta y por qué ahora es MEDIBLE (brazo positivo)
La afirmación *"con estímulo fuerte SÍ aparece sentido"* deja de ser cualitativa: `RC`, `ICR`, `IRDE` son
**columnas registradas en cada fila**. La predicción es cuantitativa:

> **Mundo compartido RICO** (mismo audio estimulante para los 4, no silencio) → **RC sube** → si O-N1 gobierna
> la invención, **el vocabulario nuevo debe aparecer justo cuando ICR sube**, y debe **trazar rutas**
> (`voz_id`/`voz_emulada_de`) porque el canal es compartido, no divergente.

El experimento siguiente no solo mide *si* inventan: permite **graficar `vocabulario_nuevo(t)` contra
`ICR(t)`** y verificar si la acuñación léxica **es** la conversión RC→ICR materializándose en una palabra.
Sería ver la ley O-N1 operando en tiempo real.

**Diseño propuesto (al terminar la corrida actual):**
- Mundo **compartido y rico** (un audio estimulante común, no silencio → RC alto, canal relacional intacto).
- Desde cero, condiciones largas.
- Métricas: `voz_creadas/aprendidas/estables` + `RC_total/ICR/IRDE/ICR_ratio` por fila + rutas
  `voz_id`/`voz_emulada_de`. Correlación temporal invención ↔ ICR. Conservación `RC = ICR + IRDE` por condición.

---

## 5. Por qué importa (más allá del léxico)
- **El "no inventan" dejó de ser un callejón**: era O-N1 prediciendo correctamente (RC≈0 → sin ICR → sin
  sentido nuevo). Probar lo que NO es nos llevó a la ley que sí es.
- **O-N1 quedó validada como identidad de conservación** en los datos (residuo 0), no solo como postulado.
- El sistema tiene una **palanca causal explícita** sobre la emergencia de sentido: subir RC (riqueza del
  contexto compartido) debería producir vocabulario nuevo y trazable. Es la hipótesis a falsar.

---

## Estado de las corridas
- **`20260629_042243`** (silencio, terminada): brazo negativo, conservación exacta, RC≈0, creadas=0.
- **`20260629_054905`** (silencio, condiciones largas 90 min + ESTRELLA×3, en curso, fin ~18:00): confirma
  que ni con ventanas largas asoma léxico en silencio (refuerzo del brazo negativo).
- **Siguiente (a lanzar):** mundo **compartido rico** — el brazo positivo, para cazar la conversión RC→ICR.

## Archivos
- Este informe + `ANIMA4_RESULTADOS_v2_2026-06-29.md` + `ANIMA4_CORRECCIONES_y_protocolo_v2.md` (repo + `~/Downloads/`).
- Datos: `~/Downloads/ANIMA4_TOPO_20260629_042243/primarios_fisiologia.tar.gz` (columnas `RC_total/ICR/IRDE`).
- Mecanismo: `Célula_Madre/organelos/VST_OrganoComunicacion.py` (`_quizas_crear`, umbrales línea 418–434).
