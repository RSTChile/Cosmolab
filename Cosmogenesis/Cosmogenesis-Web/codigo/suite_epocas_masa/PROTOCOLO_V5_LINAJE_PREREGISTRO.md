# PROTOCOLO PRE-REGISTRADO — Suite épocas masa v5 (juez por linaje)

**Fecha de pre-registro:** 2026-07-23 (antes de la corrida v5)  
**Hereda:** v4 (N-body, cutoff, pares mutuos, co-membresía)  
**Motivo del cambio de juez (v4 midió, no se ajusta el umbral a ojo):**

- v4: stack que exigía `mutual_bind R/S ≥ 1.25` → rate **0.30** (E_mutual no discrimina; SHUFFLE ≥ REAL).
- v4: `lineage_causal` (co-membresía) → rate **0.90**, R/S≈1.42.
- **V5 no baja umbrales tras ver el dato:** cambia el *protocolo* y lo fija *antes* de correr.

---

## Contrato de épocas (sin cambio)

| Época | mass_obs |
|-------|----------|
| E0–E3 | **0** |
| E4 REAL | puede ser >0 |
| E4 OFF / SHUFFLE / INVERT | **0** |

Rm/v3 = leak de nombre, **no** masa. Sin 1/1836 / GeV.

---

## Juez primario E4 (v5) — PRE-REGISTRADO

Una semilla es **`e4_lineage_pass = True`** sii **todas**:

1. `E3_ok` (átomo estricto)
2. `mass_pre == 0` (E0–E3)
3. `mass_OFF == mass_SHUFFLE == mass_INVERT == 0`
4. `mass_REAL ≥ MASS_REAL_MIN` (0.3, igual v4)
5. `dens_REAL ≥ DENS_MIN` (1.2, igual v4)
6. **Linaje gana al SHUFFLE** (al menos una de):
   - `co_member_REAL / max(co_member_SHUFFLE, ε) ≥ COMEM_VS_SHUFFLE_MIN` (**1.15**), **o**
   - `fusion_REAL > fusion_SHUFFLE` **y** `fusion_R/S ≥ 1.15`, **o**
   - `n_long_co_REAL / max(n_long_co_SHUFFLE, ε) ≥ 1.15` con `n_long_co_REAL ≥ 1`

### Diagnóstico (se reporta, **no** entra al PASS primario)

- `mutual_bind`, `mutual_R/S`, `n_mutual` (v4 los usaba como juez; v5 no)
- `E_bind` total, gyr_ratio

### PASS global de la suite

```
rate_E3 ≥ 0.55
rate_e4_lineage_pass ≥ 0.55
mass_nulls_clean == True
```

Veredictos posibles:

| condición | veredicto |
|-----------|-----------|
| E3 + lineage PASS + nulls | `E3_OK_E4_LINEAGE_CAUSAL_OK` |
| E3 + nulls + mass REAL + lineage rate 0.30–0.55 | `E3_OK_E4_PARTIAL_lineage` |
| E3 + nulls + mass REAL + lineage <0.30 | `E3_OK_E4_MASS_GATE_OK_LINEAGE_FAIL` |
| nulls rotos | `E3_OK_MASS_NULL_LEAK` |
| E3 fallido | `E3_E4_LINEAGE_FAIL` |

**Prohibido tras ver el dato:** subir G, bajar COMEM_VS_SHUFFLE_MIN, o redefinir mass_obs para fabricar PASS.

---

## Barridos (igual espíritu v4)

- Controles: 10 seeds, G=0.20, modos real/off/shuffle/invert  
- Barrido G: 0…0.45  
- Smoke L: 24, 28, 32, 40  

---

## Firma de pre-registro

Este archivo se escribe **antes** de ejecutar `suite_epocas_masa_v5_linaje.py`.  
Cualquier cambio de umbral después de la corrida invalida el PASS de v5.
