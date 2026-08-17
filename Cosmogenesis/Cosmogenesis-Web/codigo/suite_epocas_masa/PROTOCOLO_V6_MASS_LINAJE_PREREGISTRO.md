# PROTOCOLO PRE-REGISTRADO — Suite épocas masa v6 (mass_obs acoplada al linaje)

**Fecha de pre-registro:** 2026-07-23 (**antes** de la corrida v6)  
**Hereda:** v5 (juez primario = linaje; mutual = diagnóstico)  
**Hallazgo v5 que motiva el cambio de protocolo (no de umbral a ojo):**

- rate lineage_ok ≈ 0.80, co_member R/S ≈ 1.42  
- rate e4_lineage_pass = **0.40** porque muchas semillas con linaje bueno tenían **mass_obs=0**  
- mass_obs v4/v5 dependía de **E_mutual** (que no discrimina REAL≻SHUFFLE)

---

## Cambio pre-registrado: definición de `mass_obs`

### v5 (retirada como fórmula de masa)
```
mass_obs ∝ (-E_mutual) × dens × gyr × n_groups   # solo si n_mutual≥1
```

### v6 (nueva, fija antes de correr)
Solo si `grav_mode == "real"` y `G_GRAV > 0`:

```
mass_obs = dens_enhance
         × max(n_groups, 1)
         × ( n_long_co_pairs + 1e-6 )
         × ( co_member_score + 1e-6 )
         × gyr_factor
         / max(N_atoms, 1)
```

donde:
- `n_long_co_pairs` = pares co-miembros ≥ MUTUAL_MIN_STEPS pasos en E4  
- `co_member_score` = fracción media de co-membresía en E4  
- `gyr_factor` = compactación (igual espíritu v4; ≥1 si el grupo se aprieta)  
- OFF / SHUFFLE / INVERT: **mass_obs ≡ 0** (por modo, no por fórmula)

**Lectura:** la masa del relato se asigna solo cuando hay gravedad REAL **y** hay **linaje de fusión medible**; no cuando hay pegamento energético all-to-all.

---

## Juez E4 (igual v5 — no se afloja)

`e4_lineage_pass` requiere **todas**:

1. E3_ok  
2. mass_pre == 0  
3. mass OFF/SHUFFLE/INVERT == 0  
4. mass_REAL ≥ MASS_REAL_MIN (**0.3**, sin bajar)  
5. dens_REAL ≥ DENS_MIN (**1.2**)  
6. linaje gana (co_member R/S ≥ **1.15** **o** fusión R/S ≥ 1.15 con REAL>SHUFFLE **o** n_long_co R/S ≥ 1.15 con n_long≥1)

PASS global: rate_E3 ≥ 0.55 **y** rate_e4_lineage_pass ≥ **0.55** **y** nulls limpios.

Veredictos: mismos nombres que v5 (`E3_OK_E4_LINEAGE_CAUSAL_OK`, `PARTIAL_lineage`, etc.).

---

## Prohibido tras ver el dato

- Bajar MASS_REAL_MIN o COMEM_VS_SHUFFLE_MIN  
- Volver a meter E_mutual en el gate de PASS  
- Subir G solo en seeds fallidas  

---

## Firma

Este archivo se escribe **antes** de ejecutar `suite_epocas_masa_v6_mass_linaje.py`.
