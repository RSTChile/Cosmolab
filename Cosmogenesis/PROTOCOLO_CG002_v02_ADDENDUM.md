# PROTOCOLO CG002 — Addendum v0.2: firma multicomponente S^{d-1}

**Autor:** Claude (sesión CC) · **Para:** Alexis / Grok · **Fecha:** 2026-06-29
**Estado:** IMPLEMENTADO + smoke verificado (`cg002_multicomponente.py`)
**Relación:** extiende `PROTOCOLO_CG002_ACOPLAMIENTO_ORIGINARIO.md` §2.5. **No** modifica el motor verificado v0.1c (`cg002_acoplamiento.py`, B PASS cualificado) — v0.2 es módulo aparte.

---

## 0. Motivación

En v0.1c la firma es 1 fase (ω∈ℤ_K). La matriz de compatibilidad `C = cos(2π(ω_i−ω_j)/K)` tiene **rango 2 SIEMPRE** (Gram de cos/sin) → la dimensión relacional emergente está **topada en 2**. El "3D" del visor génesis era solo *layout*. v0.2 sube el techo para que la dimensión 3D sea **intrínseca** (output medido), no pintada — coherente con la Teoría: las dimensiones **no están predeterminadas**, emergen.

## 1. Decisiones de diseño (pre-registradas)

| Elemento | Definición |
|----------|------------|
| **Firma** | `U_i ∈ S^{d-1}` (vector unitario en R^d), sorteado por semilla (Gaussiano normalizado). `d` = parámetro (default 3). |
| **Compatibilidad base** (simétrica) | `c_ij = U_i · U_j ∈ [−1,1]` (cooperación si alineados, competencia si opuestos). Rango(C) hasta `d`. |
| **Acoplamiento dirigido** (θ_CP) | rotación `R(θ_CP)` en el plano (e₀,e₁): `g_{i←j} = U_i · R(θ_CP) · U_j`, `g_{j←i} = U_i · R(−θ_CP) · U_j`. |
| **Dinámica de S** | idéntica a v0.1c: `f = η·g·√(sat S_i · sat S_j)`, `S ← (1−μ)S + Σf`, extinción κ_s, banda C-N5.1. |

**Propiedades garantizadas:**
- `θ_CP=0 ⇒ R=I ⇒ g simétrico` → **control G′ exacto** (asimetría = 0).
- Asimetría dirigida `= g_{i←j}−g_{j←i} = 2·sinθ·(u_{i0}u_{j1} − u_{i1}u_{j0})` — antisimétrica en (i,j).
- Recupera v0.1c (ℤ₈ continuo) con `d=2`.
- **d (riqueza) ⟂ θ_CP (flecha):** ortogonales (no confundir dimensión con dirección).

## 2. Smoke verificado (Claude CC)

`cg002_multicomponente.py`, N=300, θ_CP=0.3, promedio seeds 1–3:

| d | vivos | rango(C) | dim_efectiva | asym(θ=0.3) |
|---|---|---|---|---|
| 2 | 136 | **2.0** | 1.93 | 0.593 |
| 3 | 155 | **3.0** | **2.92** | 0.474 |
| 4 | 145 | 4.0 | 3.89 | 0.428 |
| 5 | 166 | 5.0 | 4.9 | 0.368 |

**Control G′ (θ_CP=0):** d=3 → `asym=0.000`, rango=3, dim=2.97; d=4 → `asym=0.000`, rango=4, dim=3.94.

**Veredicto:** el rango relacional emergente **= d** (3D intrínseco con d=3), y es **ortogonal a θ_CP** (apagar la flecha no baja el rango). Confirmado.

## 3. Caveat honesto

`asym` decrece al subir d (0.59 → 0.37): la flecha vive solo en el plano (e₀,e₁), así que con firmas más ricas representa una fracción menor de la estructura total. No es fallo — es consecuencia del diseño (un solo plano de rotación). **Pendiente v0.3 (opcional):** generalizar el plano/eje de la flecha (p.ej. θ_CP como rotación en un plano elegido, o vector CP `n` en S^{d-1}).

## 4. Reproducir

```
cd Cosmogenesis && ./venv/bin/python cg002_multicomponente.py
```

## 5. Pendiente

- Integrar firma S^{d-1} en el **visor** (posiciones desde embedding relacional 3D real, no solo layout).
- Bump del protocolo principal a v0.2 cuando se consolide.
- Migrar a MEMANTO cuando reviva.
