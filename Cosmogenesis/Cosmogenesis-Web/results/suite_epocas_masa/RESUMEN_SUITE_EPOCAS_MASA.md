# SUITE ÉPOCAS MASA — resultados

**Fecha:** 2026-07-22  
**Código:** `Cosmogenesis-Web/codigo/suite_epocas_masa/suite_epocas_masa.py`  
**JSON:** `results/suite_epocas_masa/suite_epocas_masa_result.json`  
**Canon:** `15_CRONOLOGIA_MASA_NO_ANTES_DE_ATOMO_CS.md`  
**Tiempo:** ~521 s · barridos amplios · sin 1/1836

---

## Contrato (Alexis)

```text
E0  pre-Higgs simétrico     →  PROHIBIDA masa
E1  ruptura / VEV           →  PROHIBIDA masa
E2  post-Higgs pre-átomo    →  PROHIBIDA masa
E3  primer átomo (H)        →  aún sin claim de masa
E4  gravedad sobre H +
    densidad de H sube      →  AHÍ nace la masa
```

Dos canales de medición (no confundir):

| canal | qué es | rol |
|-------|--------|-----|
| **`mass_obs`** | densidad de H densificada por gravedad | **masa legítima** (solo E4) |
| **`leak_sep`** | fórmula precoz tipo v3 `m∝‖Φ‖·Σφ`, ratio k1/k3 | **instrumento ilegal** si se llama masa en E0–E3 |

---

## Veredicto global

### `MASS_E4_OK_BUT_LEAK_INSTRUMENT`

| claim | estado |
|-------|--------|
| Masa legítima **solo** en E4 con gravedad | **OK (robusto)** |
| Masa legítima = 0 en E0–E3 | **OK (100%)** |
| Instrumento v3 **no** finge masa en E0–E3 | **FAIL (fuga sistemática)** |
| Cadena completa E0→E4 sin fuga de instrumento | **no** (por el leak v3) |

---

## Resultado 1 — La masa del relato (E4)

### Control gravedad ON vs OFF (6 seeds)

| seed | mass ON | mass OFF | solo con gravedad |
|------|---------|----------|-------------------|
| 2025 | 12.32 | **0** | sí |
| 42 | 9.45 | **0** | sí |
| 777 | 16.62 | **0** | sí |
| 3141 | 11.39 | **0** | sí |
| 99991 | 29.06 | **0** | sí |
| 12345 | 15.02 | **0** | sí |

- **rate mass_only_with_grav = 1.00**
- media ON ≈ **15.64** · media OFF = **0**
- `mass_obs_max_pre_E4 = 0` en ablaciones full (por construcción del proceso: no se calcula sin gravedad)

### Barrido G_GRAV

- G=0 → E4 masa off  
- G>0 → E4 masa on en **~89%** del barrido (36 corridas)

### Ablaciones

| ablación | mass_pre | mass_E4 | rate E4 |
|----------|----------|---------|---------|
| full | **0** | ~12.4 | 1.00 |
| no_gravity | **0** | **0** | **0** |
| no_medium | 0 | ~14.2 | 1.00 |
| early_gravity | 0 | ~136 (más clumping) | 1.00 |

**Lectura:** en este juguete, la **masa legítima** cumple tu norma:  
cero hasta que hay gravedad actuando sobre H densificable; sin gravedad, cero.

(Nota: `no_medium` aún forma átomos/densidad en esta definición laxa de H — el medio no es el interruptor de la masa E4; la **gravedad** sí lo es.)

---

## Resultado 2 — Kill-switch del instrumento precoz (v3)

Si se sigue calculando `leak_sep = |Rm−NULL|` con la fórmula de “masa” v3 **en E0–E3**:

| bloque | rate KILL (leak≤0.05) | mean leak_max |
|--------|----------------------|---------------|
| K1 seeds | **0.00** | **0.65** |
| K2 potencial amplio | **0.00** | **0.25** |
| E01 / E3 / E4 barridos | **0.00** | ~0.21–0.28 |

**Lectura (la tuya, confirmada por máquina):**  
ese instrumento **no puede llamarse masa**. Fuga en todo el barrido.  
Era un **falso positivo de nomenclatura**, no un fenómeno de masa del relato.

---

## Resultado 3 — Épocas tempranas (sin masa legítima)

| check | rate |
|-------|------|
| E0 simétrico (Φ bajo pre-Tc) | **1.00** (seeds / H_EXP / átomos) |
| E1 VEV tras Tc | **1.00** (baseline; 0.63 en barrido potencial extremo) |
| E3 átomo H análogo | **1.00** (barrido L×MIX y seeds) |
| mass_obs pre-E4 | **0** siempre |

---

## Qué queda admitido / suspendido / enterrado

| claim | estado |
|-------|--------|
| “No hay `mass_obs` antes de E4” | **Admitido** |
| “`mass_obs` requiere gravedad sobre H” | **Admitido** |
| “v3 Rm es masa” | **Enterrado** (fuga / nombre ilegal) |
| “Higgs = masa al instante” | **Suspendido / rechazado** en este arco |
| “Átomo H análogo aparece post-freeze” | **Admitido** (definición operativa de la suite) |
| 1/1836, 125 GeV | **No reclamados** |

---

## Implicación para el programa

1. **Borrar o renombrar** todo juez `Rm`/`masa` pre-E4 → p. ej. `order_inheritance_sep` (si se mide, **no** es masa).  
2. **Conservar** la cadena: E0 simetría → E1 VEV → E3 átomo → E4 gravedad+densidad→masa.  
3. Endurecer E3 (átomo menos laxo) y E4 (masa solo si dens_enhance y N_H suben **por** gravedad, con NULL shuffle de pozos).  
4. No reabrir claim Higgs-as-mass.

---

## Artefactos

- `codigo/suite_epocas_masa/suite_epocas_masa.py`
- `results/suite_epocas_masa/suite_epocas_masa_result.json`
- `results/suite_epocas_masa/suite_run.log`
- este resumen
