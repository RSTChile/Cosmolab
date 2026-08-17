# PROTOCOLO PRE-REGISTRADO — Motor unificado etapas 1→7

**Fecha de pre-registro:** 2026-07-23  
**Objetivo:** un solo pipeline con estado compartido que encadena el enfoque:

```
1 campo continuo + ε  →  2 expansión (r)  →  3 estiramiento/adiabático
→  4 ρ∝a⁻³  →  5 orden/VEV (sin masa)  →  6 átomo E3  →  7 masa por linaje E4
```

Hoy existían **dos laboratorios** (CS074-rcruz en Cosmogenesis; suite épocas en Cosmogenesis-Web).  
Este motor **orquesta** ambos con un JSON de estado único; no reescribe la física adjudicada.

---

## Etapas y jueces (pre-registrados)

| # | Etapa | Motor | PASS si |
|---|-------|-------|---------|
| 1 | Campo + ε | CS074-rcruz (campo) | ε>0 produce contraste; ε=0 → P=0 |
| 2 | Expansión r | CS074-rcruz | r=0 lava (P<0.15); r≥0.1 REAL≻NULL (z>2) |
| 3 | Estiramiento | TEST_RHO_DISPERSION o proxy | stretch_ok / real_stretch (si se invoca) |
| 4 | Densidad ρ∝a⁻³ | TEST_RHO o a³ del juguete épocas | ρ cae con a |
| 5 | Orden/VEV sin masa | suite épocas E0–E1 | mass_obs=0 pre-E4; VEV post-Tc |
| 6 | Átomo E3 | suite épocas | rate E3 ≥ 0.55 |
| 7 | Masa linaje E4 | **suite v5** (pre-registro original) | rate e4_lineage_pass ≥ 0.55; nulls limpios |

**Cierre de cadena 1→7 (PASS global):** etapas 1–2 y 5–7 PASS; 3–4 reportadas (pueden ser smoke).  
No se exige 1/1836 ni GeV.

### Addendum 2026-07-23 (auditoría director) — no altera el pre-registro; lo aplica

- La corrida con el juez **pre-registrado (v5)** dio rate 0.40 → **FAIL etapa 7** → chain_pass false.  
- La sustitución posterior por **v6** y un chain_pass true **no cuenta** como PASS del protocolo de este archivo.  
- v6 además define mass_obs con las mismas variables del linaje → tautología (ver `HALLAZGO_ABIERTO_etapa7_v6_masa_es_linaje_CS.md`).  
- **Etapa 7 permanece ABIERTA** hasta nuevo pre-registro con masa **independiente** del juez de linaje, o hasta renunciar al claim “masa”.

---

## Estado compartido

Ver `estado.py`: `EstadoMotor1a7` serializable a JSON en `resultados/`.

---

## Modos

- `smoke`: N campo 100, épocas seeds reducidas, G fijo  
- `produccion`: usa resultados ya corridos si existen + re-ejecuta faltantes  

**Prohibido:** retocar umbrales de v5 o r-cruz tras ver el pipeline.
