# Higgs_TEST_REAL_v3 — robustez multi-seed + L (2026-07-22)

## Pregunta

¿El germen `TEST_PARTIAL_medium_coupling` de v3 es estable bajo seed y tamaño de grilla, **sin** retocar la física hacia 1/1836?

## Batería pre-registrada

| bloque | L | n seeds | seeds |
|--------|---|---------|-------|
| A | 30 | 10 | 2025, 7, 42, 99, 123, 777, 1024, 3141, 8191, 99991 |
| B | 45 | 5 | 2025, 42, 777, 3141, 99991 |
| C | 60 | 3 | 2025, 42, 777 |

Física = sello v3 intacto (`ALPHA_CUT=2.5`, hard-freeze, etc.).

Criterio robustez: `rate_signal_ok ≥ 0.7` en L=30 y mediana `|Rm−NULL| > 0.08`.

## Resultado

### Veredicto global

**`ROBUST_PARTIAL_medium_coupling`**

### Agregados

| bloque | rate signal | partial/pass | sep mediana | NULL mediana |
|--------|-------------|--------------|-------------|--------------|
| L=30 | **0.90** (9/10) | 0.90 | **0.115** | 0.333 |
| L=45 | **1.00** (5/5) | 1.00 | **0.109** | 0.334 |
| L=60 | **1.00** (3/3) | 1.00 | **0.109** | 0.333 |

Única falla: seed=123 en L=30 → `TEST_FAIL_inherit_but_no_mass_signal` (|Δ|=0.077, justo bajo el umbral 0.08). Herencia presente, señal borderline.

### Rangos típicos (corridas con señal)

- Rm ≈ 0.19–0.23 (nunca ~1/1836; no se buscó)
- NULL ≈ 0.33 (control geométrico estable)
- v_k1/v_k3 ≈ 0.17/0.27 (herencia estable)
- |Rm−NULL| ≈ 0.10–0.14

## Lectura

1. **No es fluke de seed 2025.** 9/10 en L=30; 100% en L=45 y L=60.  
2. **Escala con L:** la señal no se diluye al subir 30→45→60.  
3. **Techo del juguete:** PARTIAL estable, no PASS de jerarquía fuerte.  
4. Para la mesa: el germen “medio condiciona tejido → masa ≠ geometría” es **reproducible** en este instrumento.

## Artefactos

- `codigo/fase6_higgs_barrido_final/Higgs_TEST_REAL_v3_robustez.py`
- `results/.../Higgs_TEST_REAL_v3_robustez_result.json`
- `results/.../Higgs_TEST_REAL_v3_robustez.log`
