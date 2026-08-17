# MF-1 — Masa térmica (post F0 cerrado)

**Fecha:** 2026-07-21  
**Código:** `codigo/mf1_masa_termica.py`  
**Log:** `results/mf1_masa_termica_run.log`  
**JSON:** `results/mf1_masa_termica_result.json`

## Contexto

- Meta = Mac F0 **PASS** (2.22e-16); k1→333, k3→36, perim8=k3, mk1/mk3=**0.3333**.
- Diagnóstico: ratio **geométrico** (dens≈1/celda → m∝k → 1/3).

## Fórmula implementada

\[
m = \frac{\sum_{i\in\mathrm{dom}} \rho_{\mathcal{E},i}\, e^{-\mathrm{var}_{in,i}/T_{\mathrm{dimless},i}}}{P_{\mathrm{norm}}\, a^{\beta}}
\]

- \(T_{\mathrm{local}}=T_0/a\cdot(1+\varepsilon\cdot\mathrm{pert})\), \(\varepsilon=10^{-5}\) fijo  
- \(\mathrm{var}_{in}\) = rugosidad **local en malla** (phi+theta+fluct T), no var trivial de cluster k=1  
- Constantes selladas; no ajuste a 1/1836  

## Resultado local

| step | a | k1 | k3 | var1 | var3 | Rm_mean | geo |
|-----:|--:|---:|---:|-----:|-----:|--------:|----:|
| 50 | 1.82 | 11 | 4 | 0.433 | 0.378 | **0.313** | 0.333 |
| 100 | 3.32 | 72 | 12 | 0.380 | 0.292 | **0.308** | 0.333 |
| 250 | 20.1 | 253 | 31 | 0.315 | 0.244 | **0.312** | 0.333 |
| 450 | 221 | 333 | 36 | 0.307 | 0.220 | **0.308** | 0.333 |

- **SMOKE:** PASS  
- **MF-1:** `MF1-FAIL-geometric` — Rm ≈ **0.31** (se separa poco de 1/3; **no** cae a ~0.01 ni 0.00054)

## Lectura

1. El peso \(e^{-\mathrm{var}/T}\) diferencia poco k1 vs k3 (var1 solo ~30% mayor que var3).  
2. El factor de tamaño **×3 celdas** domina → ratio sigue O(1/3).  
3. phi casi plano (1+1e-9·pert) → poca ℰ atrapada diferencial.  
4. **No es FAIL de smoke F0**; es **no quiebre térmico de la jerarquía** con esta definición de var/ℰ.

## Siguiente (sin contrabando)

Opciones honestas (elegir en preregistro, no tras ver 1/1836):

- A) Enriquecer \(\rho_ℰ\) con contraste real de dominio (no phi≈1).  
- B) Definir coherencia por **persistencia temporal** del dominio, no solo var espacial.  
- C) Declarar MF-1 **negativo** con esta instrumentación y pasar a tríada con ℰ/P dinámicos más fuertes (EOS acoplada).
