# Paquete Higgs ABCD — ejecución Mac

**Zip:** `/Users/alexis/Downloads/paquete_Higgs_ABCD.zip`  
**Código:** `codigo/paquete_Higgs_ABCD/`  
**Logs:** `results/paquete_Higgs_ABCD/*.log`  
**Fecha:** 2026-07-21  
**Todos EXIT: 0**

---

## Qué prueba cada brazo (intención)

| ID | Script | Análogo Higgs | Fórmula de m (única) | NULL declarado |
|----|--------|---------------|----------------------|----------------|
| **A** | `12_Higgs_A_scalar.py` | Campo Φ, V=rΦ²+uΦ⁴, r∝(T/Tc−1) SSB | m = y0 · \|⟨Φ⟩\|_k · Σρ | Φ=0 → ratio ~1/3 |
| **B** | `13_Higgs_B_density.py` | Orden φ con λ(ρ) | m = y0 · \|⟨φ⟩\|_k · Σρ | — |
| **C** | `14_Higgs_C_phase_friction.py` | Fricción de fase η=∫\|∇θ\|² | m = y0 · Σρ/(η+ε) | η=0 → 1/3 |
| **D** | `15_Higgs_D_effective.py` | Curvatura d²E (perim+var)/T | m = y0 · d²E | — |

Común: a=exp(6·tg), T=T0/a, L=24, seed 2025, sin if k1≠k3 en potencias opuestas.

---

## Resultados Mac

### A — Scalar Φ

| step | a | r | ⟨Φ⟩ | k1 | k3 | mk1 | mk3 | ratio |
|-----:|--:|--:|----:|---:|---:|----:|----:|------:|
| 0 | 1.0 | 0.00 | −0.008 | 32 | 6 | 2.1e-2 | 6.0e-2 | **0.355** |
| 100 | 4.5 | −0.78 | −0.047 | **0** | **0** | 0 | 0 | 0 |
| 200–300 | … | r→−1 | ⟨Φ⟩~−0.05 | 0 | 0 | 0 | 0 | 0 |

- En step 0 (r=0, sin SSB aún): ratio **O(1)** ~ geometría.  
- Luego aparece r<0 (T<Tc) y ⟨Φ⟩ crece un poco, pero **k1/k3 medibles desaparecen** → no hay curva Rm(t) usable.  
- NULL Φ=0 **no se ejecutó** en el script (solo mensaje).

### B — Densidad / φ

| step | a | ⟨φ⟩ | k1 | k3 | ratio |
|-----:|--:|----:|---:|---:|------:|
| 0–300 | 1→90 | ~0.99 | **0** | **0** | 0 |

Sin clusters k1/k3 contados → **no hay test de masa**.

### C — Fricción de fase

| step | a | ⟨η⟩ | k1 | k3 | ratio |
|-----:|--:|----:|---:|---:|------:|
| 0 | 1.0 | 6.28 | 33 | 6 | **0.488** |
| 100–300 | … | ~0.9–1.0 | **0** | **0** | 0 |

Step 0: ratio **0.49 > 1/3** (k1 más “pesados” con m∝1/η — coherente si k1 tiene más gradiente).  
Luego sin k3 medible.

### D — Potencial efectivo

| step | a | k1 | k3 | ratio |
|-----:|--:|---:|---:|------:|
| 0 | 1.0 | 33 | 6 | **0.503** |
| 100–300 | … | 0 | 0 | 0 |

Otra vez solo step 0 útil; ratio O(1).

---

## Veredictos

| Brazo | ¿Campo de fondo / acoplamiento? | ¿Rm estructura? | ¿NULL corrido? | Estado |
|-------|----------------------------------|-----------------|----------------|--------|
| **A** | Sí (Φ + V(T)) | No medible tras t>0 (k=0) | No | **Andamiaje parcial** — SSB de r sí; masa no testable |
| **B** | Orden φ + λ(ρ) | No (k=0 siempre) | No | **FAIL de lectura** (clusters) |
| **C** | Fricción fase (análogo débil de “resistencia”) | Solo t=0, O(1) | No | **Indiciario t=0**; sin trayectoria |
| **D** | Curvatura efectiva (no VEV) | Solo t=0, O(1) | No | **No es Higgs** (es proxy d²E); sin trayectoria |

**Ningún brazo A–D produce, en esta corrida, una jerarquía Rm ≪ O(1) sostenida ni NULL que demuestre mecanismo tipo Higgs.**

---

## Diagnóstico de instrumentación (por qué k→0)

1. **L=24**, corte H_fis agresivo → grafo se fragmenta y el detector k=3∧perim=8 deja de ver objetos.  
2. Samples solo cada 100 pasos → se pierde la ventana intermedia.  
3. F0 “cerrado” usaba L=30 y prints cada 50; aquí se pierde continuidad con esa plataforma.  
4. A: m∝|⟨Φ⟩|·Σρ sigue siendo **∝ tamaño** si ⟨Φ⟩ es similar en k1 y k3 → ratio ~1/3 cuando hay clusters.

---

## Relación con “equivalente del Higgs”

| Requisito anti-Shannon | A | B | C | D |
|------------------------|---|---|---|---|
| Campo de fondo Φ | ✓ | parcial (φ es el orden del dominio) | ✗ (η es gradiente, no VEV) | ✗ |
| Misma fórmula m para todo k | ✓ | ✓ | ✓ | ✓ |
| g no fijado al MS | ✓ (y0 global) | ✓ | ✓ | ✓ |
| NULL Φ=0 / g=0 **corrido** | ✗ | ✗ | ✗ | ✗ |
| Inercia por **respuesta** (empujón) | ✗ | ✗ | ✗ | ✗ |
| Rm estable y kill-switch | ✗ | ✗ | ✗ | ✗ |

Conclusión: el paquete **empieza** el camino correcto (sobre todo **A**), pero aún no es un experimento Higgs completo en el sentido acordado (fondo + acoplamiento + respuesta + NULL).

---

## Siguiente ciclo recomendado (sin Shannon)

1. **A+NULL:** misma corrida con Phi≡0; exigir que Rm vuelva a geometría.  
2. **Preservar k3:** L=30, H_topo como F0, sample cada 50, no filtrar solo perim==8 al inicio.  
3. **Observable de respuesta:** empujar dominio y medir retraso ∝ |⟨Φ⟩| o g.  
4. No gate 1/1836.

---

## Archivos

| Ruta |
|------|
| `codigo/paquete_Higgs_ABCD/12_…15_….py` |
| `results/paquete_Higgs_ABCD/*.log` |
| Este resumen |
