# Higgs_TEST_REAL_v3 — tejido condicionado por medio (2026-07-22)

## Pregunta (igual v1/v2)

¿VEV + fórmula única `m = y0 · factor · Σρ` produce separación REAL vs NULL?  
Éxito ≠ 1/1836.

## Delta v3

1. Hard-freeze Φ (de v2).  
2. **Rotura de enlaces ponderada por |Φ| del borde**: muro (medio débil) se corta antes.  
3. **Mezcla de densidad ∝ fuerza del medio**: bulk difunde más; muros aíslan.  
4. Sin `if k`, sin gate 1/1836.

Sello: `ALPHA_CUT=2.5 MIX0=0.35 G_RHO=0.8 FREEZE_TNORM=0.40 Y0=0.3 seed=2025`.

## Resultado

| métrica | valor |
|---------|-------|
| **veredicto** | **TEST_PARTIAL_medium_coupling** |
| ⟨\|Φ\|⟩ | 0.277 ✓ |
| wall_frac | 0.397 ✓ |
| v_k1 / v_k3 | **0.176 / 0.285** (herencia ✓, contrast≈0.47) |
| Rm_SSB | **0.210** |
| NULL_SSB | **0.334** (geometría limpia) |
| \|Rm−NULL\| | **0.124** > SEP_THR=0.08 ✓ |
| hierarchy Rm&lt;0.1 | no (parcial, no PASS pleno) |

Trayectoria post-freeze: v1 se queda en muros (~0.17), v3 sube hacia bulk (~0.32); Rm baja de ~0.27 a ~0.18 mientras NULL se queda en ~1/3.

## Cadena completa TEST_REAL

| versión | hallazgo | veredicto |
|---------|----------|-----------|
| v1 | VEV, Φ uniforme | `FAIL_VEV_but_no_mass_signal` |
| v2 soft | freeze malo (dV borra muros) | `FAIL_VEV_still_uniform` |
| v2 hard | muros vivos, sin herencia | `FAIL_structure_but_no_mass_signal` |
| **v3** | **herencia + señal vs NULL** | **`PARTIAL_medium_coupling`** |

## Lectura

En lenguaje de la Teoría: el **medio** solo no basta; hace falta que el **tejido** (enlaces / dominios) esté **condicionado por el medio**. Entonces aparece **diferencia estructurada de masa** respecto al control geométrico.

No es 1/1836. Es el primer germen anti-Shannon de mecanismo tipo medio en este juguete:  
**Rm se separa del NULL sin poner la jerarquía a mano.**

## Artefactos

- `codigo/fase6_higgs_barrido_final/Higgs_TEST_REAL_v3_test.py`
- `results/fase6_higgs_barrido_final/Higgs_TEST_REAL_v3_result.json`
- `results/fase6_higgs_barrido_final/Higgs_TEST_REAL_v3.log`
