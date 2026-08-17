# Higgs_TEST_REAL_v2 — medio no uniforme (2026-07-22)

## Pregunta (igual que v1)

¿VEV + fórmula única `m = y0 · factor · Σρ` produce separación REAL vs NULL geométrico (~1/3)?  
Éxito ≠ 1/1836.

## Delta de diseño vs v1

| | v1 | v2 |
|--|----|----|
| Φ tras SSB | evoluciona libre → homogéneo | quench Z2 + **hard-freeze** al cruzar `FREEZE_TNORM` |
| Pozo | `r = R0(T−TC)` global | `r_loc = R0(T−TC) − G_RHO(ρ̂−1)` pre-freeze |
| Lección soft-freeze | — | dV local **borra muros** aunque D≈0; hay que **bloquear Φ** |

Sello hard-freeze: `R0=2 U=0.5 TC=0.55 D_PHI=0.05 G_RHO=0.8 FREEZE_TNORM=0.40 Y0=0.3 seed=2025`.

## Intentos

### A) Soft-freeze (archivado)

`Higgs_TEST_REAL_v2_softfreeze_FAIL_*`  
**TEST_FAIL_VEV_still_uniform**: ⟨|Φ|⟩→1, wall→0, Rm≈NULL.  
Causa: potencial local empuja cada sitio a ±v; bajar D no preserva muros.

### B) Hard-freeze (resultado principal)

`Higgs_TEST_REAL_v2_result.json` / `.log`

| métrica | valor |
|---------|-------|
| **veredicto** | **TEST_FAIL_structure_but_no_mass_signal** |
| ⟨\|Φ\|⟩_SSB | 0.293 (VEV ✓) |
| std \|Φ\| | 0.175 |
| wall_frac | **0.368** (estructura ✓) |
| v_k1 / v_k3 | 0.312 / 0.303 (casi iguales) |
| Rm_SSB | 0.342 |
| NULL_SSB | 0.331 |
| \|Rm−NULL\| | **0.011** (≪ SEP_THR=0.08) |

Φ queda fijo con muros; k1 y k3 del campo de densidad **muestrean casi el mismo** ⟨|Φ|⟩.

## Lectura en capas

1. **v1:** “no hay señal” porque el medio era **uniforme**.  
2. **v2:** el medio **sí** puede ser no uniforme (muros congelados).  
3. Aun así **no hay señal de masa**: los dominios de densidad no están **acoplados al contraste del medio**.  
4. No es refutación de “medio da masa” en abstracto; es cota fuerte del instrumento:
   - VEV solo → no basta  
   - VEV + muros espaciales → no basta  
   - Hace falta que la **formación / muestreo de dominios** herede la estructura de Φ (dinámica de tejido condicionada por el medio), sin `if k` ni gate 1/1836.

## Siguiente diseño natural (v3, no ejecutado)

Dinámica de φ o de enlaces `ar/ad` que dependa de `|Φ|` local (p.ej. difusión o rotura de enlaces más fácil en muros), de modo que k=1 tienda a muros (bajo |Φ|) y k=3 a bulk (alto |Φ|) — o al revés — y re-medir la misma pregunta.

## Artefactos

- `codigo/fase6_higgs_barrido_final/Higgs_TEST_REAL_v2_test.py`
- `results/.../Higgs_TEST_REAL_v2_result.json` (hard-freeze)
- `results/.../Higgs_TEST_REAL_v2_softfreeze_FAIL_result.json`
- `results/.../Higgs_TEST_REAL_result.json` (v1)
