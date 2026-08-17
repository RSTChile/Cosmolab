# Higgs_TEST_REAL_v4_rho_stretch — resultado

**Fecha:** 2026-07-22

## Pregunta

¿El germen v3 (medio→tejido→masa) sobrevive cuando ρ∝a⁻³ está activa y los gradientes de Φ se estiran en físico?

## Veredicto global

**`ROBUST_PARTIAL_higgs_with_rho_stretch`**

### Agregados multi-seed REAL

- rate señal: **1.00** (5 seeds)
- rate stretch: **1.00**
- rate señal+stretch: **1.00**
- sep mediana: **0.1257** (umbral 0.08)
- A_phys ratio mediana: **0.0017**

### Brazos seed=2025

| brazo | sep Rm−NULL | A_phys× | Rm | stretch | signal |
|-------|-------------|---------|----|---------|--------|
| REAL | 0.1257 | 0.0017 | 0.2079 | True | True |
| NULL_RHO_FIXED | 0.1271 | 0.0025 | 0.2082 | True | True |
| NULL_NO_MEDIUM | 0.0186 | 0.0017 | 0.3153 | True | False |

### Lectura

- medium_beats_blind: **True** (sep REAL 0.1257 vs blind 0.0186) → la señal **es del medio Φ**, no del estiramiento solo.
- rho_changes_signal: **False** (RHO_FIXED sep ≈ REAL) → con este sello, la separación de masa es casi igual con/sin rarefacción de D; el **estiramiento físico de muros sí ocurre** (A_phys×~0.002) en ambos brazos que expanden. ρ activa congela transporte de Φ más temprano, pero el claim de masa ya estaba en v3; v4 muestra que **no se destruye** al estirar.
- NULL_NO_MEDIUM: stretch sí, herencia/masa no → ρ+a no fabrican el germen Higgs sin acople medio→tejido.

No es claim SM/1/1836. Es: orden Φ + rarefacción + estiramiento → ¿masa ≠ geometría? **Sí, de forma parcial y robusta.**

## Artefactos

- `codigo/fase6_higgs_barrido_final/Higgs_TEST_REAL_v4_rho_stretch.py`
- `results/fase6_higgs_barrido_final/Higgs_TEST_REAL_v4_rho_stretch_result.json`
