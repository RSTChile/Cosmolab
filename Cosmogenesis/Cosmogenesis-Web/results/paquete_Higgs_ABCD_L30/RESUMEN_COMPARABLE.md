# Higgs ABCD L=30 + NULL — ejecución Mac

**Zip:** `/Users/alexis/Downloads/paquete_Higgs_ABCD_L30.zip`  
**Código:** `codigo/paquete_Higgs_ABCD_L30/`  
**Logs:** `results/paquete_Higgs_ABCD_L30/*.log`  
**Fecha:** 2026-07-22  
**Todos EXIT: 0** (B sin fixed: EXIT 0 con nan)

---

## Tabla maestra Mac L=30

| Brazo | k sostenido | ratio REAL (típ.) | ratio NULL | ⟨Φ⟩/notas | Veredicto |
|-------|-------------|-------------------|------------|-----------|-----------|
| **A** | k1→348 k3→37 | **0.15–0.37** (fin ~0.37) | **0.000** | ⟨Φ⟩→**0** tras step 0; r cruza 0 ~a=11 | SSB r sí; VEV no; NULL m=0 no geometría |
| **B** (orig) | sí (nan) | **nan** | — | φ overflow | **FAIL numérico** |
| **B fixed** | k1→355 k3→36 | **0.333–0.340** | (no en print) | ⟨φ⟩~1, lam0=0.001 | **FAIL** jerarquía (~1/3) |
| **C** | k1→320 k3→46 | **0.81–4.75** (fin ~0.81) | **0.333** | ⟨η⟩~1.8 | **NULL discrimina**; Rm O(1) no 10⁻³ |
| **D** | k1→334 k3→41 | **0.500** | **0.500** | var~1e-3 | **FAIL** NULL; perim 4 vs 8 |

---

## Detalle por brazo

### A — Scalar Φ (L=30 + NULL)

| step | a | r | ⟨Φ⟩ | k1 | k3 | ratio | ratio_NULL |
|-----:|--:|--:|----:|---:|---:|------:|-----------:|
| 0 | 1.0 | +9.00 | 0.062 | 29 | 6 | 0.361 | 0.000 |
| 150 | 6.0 | +0.65 | 0.000 | 154 | 22 | 0.262 | 0.000 |
| 200 | 11.0 | **−0.09** | 0.000 | 214 | 21 | 0.257 | 0.000 |
| 350 | 66.7 | −0.85 | 0.000 | 348 | 37 | 0.350 | 0.000 |
| 450 | 221.4 | −0.95 | 0.000 | 348 | 37 | 0.369 | 0.000 |

Alineado con tu Meta: k sostenido, r&lt;0 tras a~11, Φ→0, Rm O(1), NULL→0 (no 0.333).

### B — original vs fixed

- **B:** lam~1e5 → φ~924 → **nan** (igual diagnóstico Meta).  
- **B_fixed:** ratio clavado **~0.333** (geometría; ⟨φ⟩ uniforme ~1).

### C — fricción fase + NULL

| step | a | ratio | ratio_NULL |
|-----:|--:|------:|-----------:|
| 0 | 1.0 | 1.427 | **0.333** |
| 100 | 3.3 | 2.059 | **0.333** |
| 200 | 11.0 | 1.671 | **0.333** |
| 350 | 66.7 | 0.813 | **0.333** |
| 450 | 221.4 | 0.809 | **0.333** |

NULL **siempre 0.333** cuando hay k3 → mecanismo de acople **operativo**.  
REAL O(1) e incluso **&gt;1** (inversión k1 “más pesado” a veces) — no jerarquía 10⁻³.

### D — d²E + NULL

ratio = ratio_NULL = **0.500** en toda la trayectoria con k3 → **no Higgs**.

---

## Comparación Meta ↔ Mac

| Claim Meta | Mac L=30 |
|------------|----------|
| A: k1~348 k3~37, Φ→0, r&lt;0, Rm~0.36, NULL 0 | **Coincide** (348/37, Φ=0, r→−0.95, Rm~0.37, NULL 0) |
| B: nan sin fix; fix λ₀ pequeño | **Coincide** |
| C: NULL 0.333; REAL ~0.8–2 | **Coincide** (0.81–4.75 según step) |
| D: 0.5 = NULL 0.5 | **Coincide** |

**Doble ejecución L=30: alineada con tu reporte Meta.**

---

## Veredicto Higgs (sello)

1. **A** = único VEV-potencial legítimo; **VEV no se sostiene** → siguiente: anclaje térmico de ⟨\|Φ\|⟩.  
2. **C** = único **NULL que valida acople al medio**; jerarquía débil/invertida O(1).  
3. **B fixed / D** = no pasan como mecanismo de masa tipo Higgs.  
4. **F0** sigue siendo la plataforma k; **Higgs A2** es el siguiente experimento prioritario.

---

## Archivos

| | |
|--|--|
| Código | `codigo/paquete_Higgs_ABCD_L30/` |
| Logs | `results/paquete_Higgs_ABCD_L30/*.log` |
| Veredicto previo | `results/paquete_Higgs_ABCD/VEREDICTO_META_L30_vs_MAC.md` |
