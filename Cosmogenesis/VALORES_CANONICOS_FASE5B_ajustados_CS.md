# Valores canónicos — Fase V-B ajustados por agrupamiento

**12 de agosto de 2026** · Corrección documental pedida por GPT-5.6 Sol · Fuente única de verdad:
`FASE6_O2F_resultados_neff.csv`, fila `unidad_agrupamiento = lote_seed_base`.

**Por qué existe este archivo:** la carta al equipo daba el Wilcoxon ajustado como `5×10⁻⁴–1.5×10⁻³` y el
TL;DR del informe como `~10⁻⁴`. **Los dos eran correctos, pero de conjuntos distintos** — y esa ambigüedad
es justamente lo que hay que eliminar. De aquí en adelante, **un solo valor por test y por conjunto de
datos**, tomado de esta tabla.

---

## Tabla canónica

Unidad de agrupamiento: `lote_seed_base` (4-5 lotes). Observable: fracción de masa acretada.

| Test | 40 pares (original) | 37 pares (válidos tras corrección de diámetro) |
|---|---|---|
| **Signos, crudo** | 6.80×10⁻⁴ | 7.53×10⁻⁴ |
| **Signos, ajustado por N_eff** | **3.72×10⁻³** | **1.06×10⁻²** |
| **Wilcoxon, crudo** | 9.17×10⁻⁶ | 8.87×10⁻⁶ |
| **Wilcoxon, deflactado por N_eff** | **1.84×10⁻⁴** | **5.11×10⁻⁴** |
| **Cluster-robusto (CR1)** | 8.44×10⁻³ | 1.41×10⁻² |
| **Bootstrap de lotes** | 5.0×10⁻⁴ | 4.0×10⁻⁴ |
| ICC / deff / N_eff | 0.058 / 1.41 / **28.4** | 0.099 / 1.64 / **22.6** |
| Signos (conteo) | 31/40 | 29/37 |

**Variante estricta** (`lote_seed_base_sin_mixto`, reasignando los 1-2 pares de lote mixto): Wilcoxon
deflactado 6.10×10⁻⁴ (40 pares) y **1.50×10⁻³** (37 válidos), N_eff 23.9 y 18.9. *De aquí venía el
`1.5×10⁻³` de la carta* — es el extremo conservador de la variante estricta sobre 37 pares, no el valor
estándar.

---

## Cómo citar, según el caso

- **Valor de referencia por defecto** (el más honesto para comunicar el resultado principal):
  **Wilcoxon deflactado sobre los 37 pares válidos = 5.11×10⁻⁴.** Usa el conjunto depurado tras la
  corrección de diámetro y el ajuste por agrupamiento.
- **Si se cita el conjunto original de 40:** Wilcoxon deflactado = 1.84×10⁻⁴.
- **Rango defendible al comunicar incertidumbre de método:** de 1.84×10⁻⁴ (40 pares, estándar) a 1.50×10⁻³
  (37 pares, criterio estricto) — pero **declarando siempre a qué conjunto y variante corresponde cada
  extremo**, no como un rango sin atribución.
- **El test más conservador de todos** es el sign-flip de lote entero (p=0.125), pero **tiene piso
  0.0625-0.125 con 4-5 lotes**: es límite de resolución del test, no ausencia de efecto. No usarlo como
  valor de referencia.

## Dato que no cambia con el ajuste

`kcap` está **balanceado dentro de cada par por diseño** (K=kcap exacto), y la ICC de las *diferencias*
por kcap es negativa (−0.10) → **deff = 1.00, N_eff = 40/40**. La fuente de agrupamiento dominante a nivel
de reglas individuales **no contamina el contraste pareado** — medido, no asumido.

---

*Los documentos `CARTA_EQUIPO_FASE6_QUE_CAMBIO_CS.md` e `INFORME_EQUIPO_FASE6_11ago2026_CS.md` fueron
corregidos para apuntar a esta tabla.*
