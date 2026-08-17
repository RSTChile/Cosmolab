# Paquete F0+MF — ejecución Mac completa

**Zip:** `/Users/alexis/Downloads/paquete_f0_mf_final.zip`  
**Extraído en:** `codigo/paquete_f0_mf_final/`  
**Logs:** `results/paquete_f0_mf_final/*.log`  
**Python:** Cosmogenesis venv (numpy 2.5)  
**Todos EXIT: 0**

---

## Tabla de veredictos

| # | Script | Smoke Ta | k3 | ratio m_k1/m_k3 | Nota |
|---|--------|----------|----|-----------------|------|
| **01** | `01_f0_final_unico.py` | **PASS 2.22e-16** | 0→36, perim8=k3 | **0.3333** geométrico | F0 entregable OK |
| **02** | `02_mf1_v2_formula_unica.py` | **PASS** | 0→36 | **0.3333** | var~1e-20; FAIL jerarquía (honesto) |
| **03** | `03_mf1_v3_contraste_O1.py` | (implícito T∝1/a) | 0→36 | **0.31–0.36** | var O(10⁻³); aún O(1), no 0.00054 |
| **04** | `04_mf2_K_debye_crossover.py` | — | — | — | K 0.368→0.995; \|M\|~0.12–0.13 estable |
| **05** | `05_mf3_sigma_tension.py` | — | E_k3 reportado | — | σ=P·a **decrece** (~a⁻²); no confinamiento creciente |

---

## 01 — F0 final único

| step | a | k1 | k3 | perim8 | mk1/mk3 | err |
|-----:|--:|---:|---:|-------:|--------:|----:|
| 0 | 1.0 | 0 | 0 | 0 | 0 | 0 |
| 50 | 1.8 | 11 | 4 | 4 | 0.3333 | 0 |
| 200 | 11.0 | 215 | 26 | 26 | 0.3333 | ~0 |
| 450 | 221.4 | 333 | 36 | 36 | 0.3333 | ~0 |

**SMOKE PASS** · alineado con corridas previas Meta=Mac.

---

## 02 — MF-1 v2 fórmula única

ratio **0.3333** en todos los samples con k3>0.  
var_k1 ≈ var_k3 ~ 10⁻²⁰ (phi casi plano).

**Veredicto:** FAIL jerarquía (anti-Shannon correcto).

---

## 03 — MF-1 v3 contraste O(1) (`phi=1+0.5*pert`)

| step | a | k1 | k3 | var_k1 | var_k3 | ratio |
|-----:|--:|---:|---:|-------:|-------:|------:|
| 50 | 1.8 | 11 | 4 | 6.3e-3 | 1.2e-2 | 0.3409 |
| 100 | 3.3 | 72 | 12 | 5.9e-3 | 2.7e-3 | 0.3267 |
| 250 | 20.1 | 253 | 31 | 5.6e-3 | 5.0e-3 | 0.3258 |
| 450 | 221.4 | 333 | 36 | 5.3e-3 | 4.7e-3 | **0.3573** |

var ya es O(10⁻³), no 10⁻²⁰, pero **ratio sigue O(1)** (~1/3).  
No hay quiebre a 0.00054.

**Veredicto:** FAIL jerarquía con fórmula única + contraste O(1).

---

## 04 — MF-2 K Debye / crossover

| step | a | K | \|M\| |
|-----:|--:|--:|-----:|
| 0 | 1.0 | 0.368 | 0.021 |
| 50 | 1.8 | 0.578 | 0.102 |
| 100 | 3.3 | 0.740 | 0.134 |
| 250 | 20.1 | 0.951 | 0.124 |
| 450 | 221.4 | 0.995 | 0.125 |

K sube 0.368→0.995 (como exp(-1/a)).  
\|M\| se estabiliza ~0.12–0.13 (no el 0.391→0.191 del dominio T; **otro observable/setup**).

**Veredicto:** K crece con a; \|M\| no muestra el crossover T de referencia (definición distinta).

---

## 05 — MF-3 σ = P·a

| step | a | P | σ=P·a | E_k3 |
|-----:|--:|--:|------:|-----:|
| 0 | 1.0 | 3.0e46 | 3.0e46 | 7.7e46 |
| 100 | 3.3 | 8.2e44 | 2.7e45 | 2.3e45 |
| 250 | 20.1 | 3.7e42 | 7.4e43 | 1.1e43 |
| 450 | 221.4 | 2.8e39 | 6.1e41 | 8.0e39 |

Script mismo concluye: P~a⁻³ ⇒ **σ~a⁻² decrece** → desconfinamiento, no V=σr creciente.

**Veredicto:** FAIL de confinamiento tipo cuerda con σ=P·a bajo EOS w=1/3.

---

## Síntesis del paquete (Mac)

| Hito | Estado |
|------|--------|
| F0 dilución + k3/perim8 | **PASS / cerrado** |
| MF-1 v2 (φ~1) | ratio **0.3333** FAIL |
| MF-1 v3 (φ O(1)) | ratio **~0.32–0.36** FAIL jerarquía |
| MF-2 K(a) | K sube; \|M\| plano ~0.12 |
| MF-3 σ | **decrece** con a |

**Mensaje para comparar con Meta:** los cinco scripts corrieron EXIT 0; números arriba. Si Meta reporta los mismos prints, 100% alineados en el paquete.
