# F0 — Comparación Meta vs Local (doble ejecución)

**Fecha:** 2026-07-21  
**Código Meta (sin modificar en la 1ª corrida):** `data/f0_triada_holistica.py`  
**Log local:** `results/f0_meta_local_run.log`  
**Diagnóstico:** `results/f0_local_diagnostico.json`  
**Semilla:** `np.random.default_rng(2025)`

---

## 1. Resultado de la corrida (idéntica al script Meta)

| step | t_g (s) | a | T (K) | rho | λ_D (m) | K_phys | k1 | k3 | m_k1/m_k3 |
|-----:|--------:|--:|------:|-----:|--------:|-------:|---:|---:|----------:|
| 0 | 0 | 1.000 | 1.00e15 | 1.00e30 | 6.90e-20 | 0.000 | 0 | 0 | 0 |
| 40 | 4.00e-37 | 1.041 | 4.58e14 | 8.87e29 | 4.86e-20 | 0.000 | 0 | 0 | 0 |
| 80 | 8.00e-37 | 1.051 | 6.72e13 | 8.62e29 | 1.88e-20 | 0.000 | 0 | 0 | 0 |
| 120 | 1.20e-36 | 1.051 | 9.28e12 | 8.61e29 | 6.99e-21 | 0.000 | 0 | 0 | 0 |
| 160 | 1.60e-36 | 1.051 | 1.26e12 | 8.60e29 | 2.58e-21 | 0.000 | 0 | 0 | 0 |

EXIT 0. Reproduce el comportamiento esperado del script (misma lógica).

---

## 2. Criterio preregistro F0 (§7 / doc 14)

```
PASS  ⇔  max |T·a/T0 − 1| ≤ 5%  y  max |ρ·a³/ρ0 − 1| ≤ 5%
FAIL  ⇔  alguno > 20%
```

| Observable | max error | Umbral | Estado |
|------------|----------:|--------|--------|
| **T·a / T0** | **99.98%** | 5% / 20% | **FAIL** |
| **ρ·a³ / ρ0** | **0.00%** | 5% / 20% | PASS |

# Veredicto 1ª ejecución: **F0-FAIL**

- Densidad media: **correcta** (ρ = ρ₀/a³ cada paso).  
- Temperatura: **no adiabática** en el sentido del contrato (T·a ≠ const).

---

## 3. Errores detectados (código Meta)

### B1 — Crítico: enfriamiento T (rompe F0)

```python
T_new = T / a   # cada paso
```

Efecto: tras n pasos, \(T \sim T_0 / \prod_i a_i\), no \(T_0/a(t)\).  
Con a≈1.05, el producto acumula ~6×10³ → T cae ~10⁴ de más.  
**Fix contrato:** `T = T_amp / a` con `T_amp` fijo (CI), o `T *= a_old/a_new`.

### B2 — Expansión a(t) a trozos

- `step < 50`: `a = exp(H_inf * t_g)`  
- `step ≥ 50`: `a *= (1 + H_inf*dt*0.01)` → crecimiento casi nulo (a se congela ~1.05)

No es el a(t)=e^{H_inf t} continuo del preregistro en todo t_g.

### B3 — H_fisico no es H_inf de la tríada

```python
H_fisico = H_topo * (T_mean/T0)  # 0.0025 * T/T0
```

El corte de aristas usa **H topológico**, no H_inf físico. La “expansión” del grafo se desacopla del a(t) real cuando T cae (y tras el bug de T, H_fisico→0 y dejan de cortarse aristas: edges 1796→1632 y se estabiliza).

### B4 — K_phys ≈ 0 siempre

λ_D ~ 10⁻²⁰ m → `exp(-1/λ_D)` ≈ 0.  
No hay dinámica de fase útil en esta escala con unidades SI crudas sin adimensionalizar.

### B5 — k1=k3=0 en todos los samples

A step 0 aún no hay contraste de dominios estable / o el BFS con media no cuenta tamaños 1 y 3 de forma útil en ese instante; en pasos posteriores el campo se suaviza y el corte se detiene. **No se puede leer MF-1** sobre esta corrida.

### B6 — m_fisica_ratio

Con k3=0 queda 0; además E_k3 = n_k3 * Tc (97 MeV) es un proxy ad hoc, no ∫ρ_ℰ dV del preregistro.

### B7 — Menor: `nr` si ar vacío

Si `sum(ar)==0`, `nr` no se define y `rem=nc-(nr if 'nr' in locals() else 0)` depende de scopes previos (frágil).

### B8 — Alcance vs smoke F0

El script mezcla F0 + trozos MF-1/2/topología. Para el **smoke F0 sellado** basta dilución a–T–ρ; el resto contamina el veredicto.

---

## 4. Qué coincide Meta ↔ Local

Misma trayectoria cualitativa y números en la tabla de prints (misma semilla y lógica).  
**Doble ejecución: alineada.** El FAIL no es divergencia Meta/Mac: es **bug de física de T + diseño vs preregistro**.

---

## 5. Corrección propuesta (solo contrato F0)

Archivo: `codigo/f0_smoke/f0_smoke_preregistro.py`  
- a(t) = exp(H_inf * t) continuo en el tramo  
- T_mean = T0 / a  (y campo T_amp/a si hay ε)  
- ρ = ρ0 / a³  
- Métricas Ta, ρa³ → PASS/FAIL  
- **Sin** retocar constantes para 1/1836  
- Opcional: guardar JSON de resultado

Siguiente ciclo: ejecutar corrección → si F0-PASS → entonces reintegrar grafo/MF sobre T correcta.

---

## 6. 2ª ejecución — corrección mínima preregistro

**Código:** `codigo/f0_smoke/f0_smoke_preregistro.py`  
**Log:** `results/f0_smoke_preregistro_run.log`  
**JSON:** `results/f0_smoke_preregistro_result.json`

| Cambio vs Meta | Detalle |
|----------------|---------|
| T | `T = T_amp / a` (T_amp CI fija) |
| a(t) | `exp(H_inf t)` continuo |
| Alcance | Solo dilución (sin clusters/Debye) |

### Resultado 2ª ejecución

| max\|Ta/T0−1\| | max\|ρ a³/ρ0−1\| | Veredicto |
|---------------:|-----------------:|-----------|
| ~0 (numérico) | ~0 | **F0-PASS** |

Muestra:

| step | a | T | Ta/T0−1 | ρa³/ρ0−1 |
|-----:|--:|--:|--------:|---------:|
| 0 | 1.000 | 1.00e15 | 0 | 0 |
| 40 | 1.041 | 9.61e14 | ~0 | 0 |
| 199 | 1.220 | 8.20e14 | ~0 | 0 |

### Ciclo recursivo — estado

| Paso | Estado |
|------|--------|
| 1. Código Meta ejecutado local | Hecho — **F0-FAIL** (bug T) |
| 2. Comparación Meta↔Local | Alineados |
| 3. Corrección mínima preregistro | Hecho — F0-PASS dilución |
| 4. Reintegrar grafo/MF sobre T correcta | **Pendiente** (siguiente: parche a `f0_triada_holistica.py` o fork controlado) |
