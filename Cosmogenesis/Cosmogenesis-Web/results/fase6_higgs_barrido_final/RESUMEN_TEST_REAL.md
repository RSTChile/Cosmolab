# Higgs_TEST_REAL — veredicto (2026-07-22)

## Pregunta pre-registrada

¿Un campo de medio Φ con VEV no nulo y **una sola** ley de acoplamiento produce masas de dominio distintas de la geometría pura (~1/3), de modo que el NULL (m sin Φ) colapse a geometría?

Éxito ≠ 1/1836.  
Éxito = VEV vivo + separación REAL vs NULL + (opcional) Rm estructurado.

## Sello (constantes fijas, no retocadas post-resultado)

| const | valor |
|-------|-------|
| L | 30 |
| PASOS | 400 |
| R0 | 2.0 |
| U | 0.5 |
| TC | 0.55 |
| D_PHI | 0.05 |
| SIGMA0 | 0.08 |
| Y0 | 0.3 |
| SEED | 2025 |
| VEV_THR | 0.15 |
| SEP_THR | 0.08 |

Fórmula única: `m = y0 · factor · sum_ρ`  
- REAL: `factor = ⟨|Φ|⟩_dom`  
- NULL: `factor = 1`

## Resultado

| métrica | valor |
|---------|-------|
| **veredicto** | **TEST_FAIL_VEV_but_no_mass_signal** |
| ⟨|Φ|⟩_SSB | **0.926** (VEV vivo ✓) |
| Rm_SSB | 0.3335 |
| NULL_SSB | 0.3332 |
| \|Rm − NULL\| | **0.00028** (≪ SEP_THR=0.08) |
| null_ok (≈1/3) | True |
| hierarchy Rm&lt;0.1 | False |

### Trayectoria cualitativa

- step 0: Φ ~0.39, Rm≈NULL≈0.36–0.38  
- step 50: cruce r&lt;0, Φ cae un poco (~0.20)  
- steps 100–399: Φ → ~1.0 (VEV saturado), v_k1 ≈ v_k3 ≈ 1, **Rm ≈ NULL ≈ 1/3**

## Lectura (instrumento vs teoría)

1. **No es “la teoría no funciona” en abstracto.**  
   Este test **sí** puso un medio Φ con SSB y VEV estable. Eso cierra el fallo previo de A-V3 (“no hay VEV”).

2. **Tampoco es refutación de la idea de medio.**  
   Con Φ **casi uniforme** espacialmente, `|Φ|` multiplica **por igual** a k=1 y k=3 → el ratio se cancela y queda el residual geométrico `sum_ρ(k=1)/sum_ρ(k=3) ≈ 1/3`.

3. **Diagnóstico fino (sello cumplido):**  
   - Dinámica de VEV: **pasa**  
   - NULL geométrico: **pasa** (control limpio)  
   - Señal de masa vs geometría: **falla** porque el acople actual no genera **diferencia estructurada de medio entre dominios**

4. Afirmación anterior “simplemente no funciona (Higgs abstracto)” queda **demasiado fuerte**.  
   Lo medido: *hay VEV; no hay discriminación de masa bajo fórmula única con Φ homogéneo*.

## Qué haría falta para el siguiente test (no ejecutado; no sintonía a 1/1836)

Sin mover el sello de este JSON: un diseño donde Φ **no** sea uniforme a escala de dominio — p.ej. Φ acoplado a ρ local, o muros de dominio entre pozos ±v que dejen v_k1 ≠ v_k3 de forma estable — y re-medir la misma pregunta pre-registrada.

## Artefactos

- `codigo/fase6_higgs_barrido_final/Higgs_TEST_REAL_test.py`
- `results/fase6_higgs_barrido_final/Higgs_TEST_REAL_result.json`
- `results/fase6_higgs_barrido_final/Higgs_TEST_REAL.log`
