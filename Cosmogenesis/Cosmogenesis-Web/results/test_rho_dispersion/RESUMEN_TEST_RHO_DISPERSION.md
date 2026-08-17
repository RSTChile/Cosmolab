# TEST_RHO_DISPERSION — resultado
**Fecha:** 2026-07-22
## Hipótesis (Alexis)
Con la expansión la densidad se **estira**/enrarece: de una caída de temperatura **abrupta** se pasa a una caída **suave** porque la escala se agranda.
## Veredicto
**`TEST_PASS_stretch_and_rho`**
### Flags
- `stretch_pure_ok`: **True**
- `smooth_pure_ok`: **True**
- `real_stretch_ok`: **True**
- `real_smooth_ok`: **True**
- `rho_effect_ok`: **True**
- `a_fixed_control_ok`: **True**

### Ratios (final/init)
| brazo | A_phys | A_comov | w_phys | a_final | ρ_final |
|-------|--------|---------|--------|---------|--------|
| REAL | ×0.0019 | ×0.7625 | ×6690.052 | 403.4 | 1.523e-08 |
| NULL_RHO_FIXED | ×0.0006 | ×0.2582 | ×6640.969 | 403.4 | 1.000e+00 |
| NULL_A_FIXED | ×0.2582 | ×0.2582 | ×16.461 | 1.0 | 1.000e+00 |
| NULL_STRETCH | ×0.0025 | ×1.0000 | ×403.429 | 403.4 | 1.523e-08 |

### Contraste densidad
- A_comov final REAL = 0.2331
- A_comov final RHO_FIXED = 0.0789
- rho_contrast = 0.6614 (umbral 0.08)

## Lectura
1. **Estiramiento:** ∇_phys = ∇_comov/a — a mayor a, la caída es más suave en espacio físico aunque el perfil comóvil esté congelado.
2. **Densidad:** ρ∝a⁻³ apaga D; sin eso (ρ fija) el frente se erosiona más en coordenadas comóviles.
3. No es claim de Higgs ni de 1/1836; es el eslabón basal expansión → densidad → dispersión del gradiente térmico.

## Artefactos
- `codigo/test_rho_dispersion/TEST_RHO_DISPERSION.py`
- `results/test_rho_dispersion/TEST_RHO_DISPERSION_result.json`
