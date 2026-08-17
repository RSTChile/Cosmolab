# Paquete FINAL 7 experimentos — ejecución Mac completa

**Zip:** `/Users/alexis/Downloads/paquete_FINAL_7_experimentos.zip`  
**Código:** `codigo/paquete_FINAL_7_experimentos/` (11 scripts)  
**Logs:** `results/paquete_FINAL_7_experimentos/*.log`  
**Fecha Mac:** 2026-07-21  
**Todos EXIT: 0**

---

## Tabla maestra

| # | Script | EXIT | Resultado clave (Mac) |
|---|--------|------|------------------------|
| 01 | `01_f0_final_unico.py` | 0 | **SMOKE PASS** 2.22e-16; k1→333 k3→36 perim8=k3; ratio **0.3333** |
| 02 | `02_mf1_v2_formula_unica.py` | 0 | ratio **0.3333** (var~1e-20); FAIL jerarquía |
| 03 | `03_mf1_v3_contraste_O1.py` | 0 | ratio **0.32–0.36**; var O(1e-3); aún O(1) |
| 04 | `04_mf2_K_debye_crossover.py` | 0 | K 0.368→0.995; \|M\|~0.12–0.13 estable |
| 05 | `05_mf3_sigma_tension.py` | 0 | σ=P·a **decrece** (~a⁻²); desconfinamiento |
| 06 | `06_MF4_rhoE_real.py` | 0 | ratio 0.24→**0.43** (empeora hacia O(1), no 0.00054) |
| 07 | `07_MF5_beta_running.py` | 0 | g↓, K↑ 0.008→0.466; σ↓ (script: necesita Λ creciente) |
| 08 | `08_MF6_topological_charge.py` | 0 | Q_k3 inestable; k1/k3 a menudo 0 tras step 0 |
| 09 | `09_MF7_holographic.py` | 0 | k1=k3=0 en samples; FIN dice perim ratio 0.5 O(1) |
| 10 | `10_MF4_alpha.py` | 0 | α⁻¹ 372→1194 (no se acerca a 137; se aleja) |
| 11 | `11_MF5_entropia.py` | 0 | S crece ~a^{2.5}; 2º principio OK (relato script) |

---

## Detalle por script (números Mac)

### 01 F0
PASS dilución; k3/perim8 como corridas previas; ratio geométrico 1/3.

### 02 MF-1 v2
ratio fijo 0.3333.

### 03 MF-1 v3
| step | ratio |
|-----:|------:|
| 50 | 0.3409 |
| 250 | 0.3258 |
| 450 | 0.3573 |

### 04 MF-2
K: 0.368 → 0.995 · \|M\|: 0.021 → ~0.125

### 05 MF-3
σ: 3.0e46 → 6.1e41 (decrece)

### 06 MF4 ρ_E real
| step | a | k1 | k3 | ratio |
|-----:|--:|---:|---:|------:|
| 100 | 3.3 | 84 | 12 | 0.245 |
| 200 | 11.0 | 235 | 15 | 0.277 |
| 400 | 121.5 | 345 | 24 | **0.430** |

### 07 MF5 beta running
| step | a | g | K | σ |
|-----:|--:|--:|--:|--:|
| 0 | 1.0 | 0.455 | 0.008 | 2.07e14 |
| 200 | 11.0 | 0.195 | 0.092 | 1.04e12 |
| 400 | 121.5 | 0.104 | 0.466 | 8.05e9 |

### 08 MF6 Q topológico
step 0: k1=28 k3=7 Q_k3=0.353; luego k a menudo 0.

### 09 MF7 holográfico
sin clusters contados en samples (0/0); mensaje FIN: ratio perímetro O(1).

### 10 MF4 α_EM
| step | α⁻¹ | obj ~137 |
|-----:|-----:|----------|
| 0 | 372.5 | lejos |
| 100 | 299.3 | mínimo local aún ~2× |
| 450 | **1193.6** | se aleja |

### 11 MF5 entropía
S: 1.33e21 → 9.73e26 (crece)

---

## Lectura global (Mac)

| Dominio | Estado |
|---------|--------|
| **F0** (tríada dilución + k3) | **PASS / cerrado** |
| **MF-1** (v2, v3, ρ_E) | **No rompe** jerarquía a 10⁻³–10⁻⁴; ratio O(1) |
| **MF-2/K, beta** | K crece con a; no es el crossover T 0.391→0.191 |
| **MF-3 σ** | **No** confinamiento creciente con σ=P·a, w=1/3 |
| **α_EM** | **No** converge a 1/137 |
| **Entropía** | Crece (consistente con relato del script) |
| **Q / holografía** | Débil / sin señal de jerarquía |

Listo para comparación 1:1 con logs Meta del mismo zip.
