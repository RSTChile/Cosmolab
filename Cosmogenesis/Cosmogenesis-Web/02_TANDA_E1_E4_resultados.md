# Tanda E1–E8 y resultados E1 / E2 / E4

**Línea:** Cosmogenesis-Web  
**Origen:** consolidado del equipo Web (Grok, Qwen, DeepSeek, Gemini, Alexis) + corridas Meta  
**Fecha de registro local:** 2026-07-21  

> Esta tanda **no** está en el share de campo/clusters (`G4lrPl47yF`) con plots.  
> Los números de E1/E2/E4 provienen del reporte de Meta en el hilo de trabajo (texto).  
> Cuando existan PNG/CSV de esta tanda, van a `assets/` y se enlazan aquí.

---

## 1. Tabla comparativa de huecos (equipo Web)

| Hueco | Grok | Qwen | DeepSeek | Gemini | Alexis (datos 100×100) | **Consenso** |
|---|---|---|---|---|---|---|
| **Masa / jerarquía** | Definición adimensional no separa >10–100× | m_e/m_p O(1) vs 1/1836, factor ×2000 lejos | m_k1≈0.46 m_k3; m_k2≈2.6 m_k3 (invertida) | m=H·perim/f propone invertir f | k3 más ligero que k2 2.6×; k1/k3=0.46 O(1) ≠ 0.00054 | **CRÍTICO — bloqueo #1** |
| **K_c termodinámico** | Orden diluye con N, necesita reescalado | 0.259→0.062 crossover finito, no fase | ¿KT? no Ising | K(N)=K0·√N o ln N | 0.259→0.093 (K=0.1); 0.512→0.112 (K=0.25); N=900→10000 | **CRÍTICO — 4/5 crossover** |
| **Dinámica interacción** | Fusión/fisión, ligados, σ no medidos | F=4H=0.01 teórico, no dinámico | V(r) k3-k3 y k3-k1 no medido | Movilidad Boltzmann P∝e^{−ΔE/T} | Vecino k3 dist 5.43 medido, no desplazamiento real | **ALTO — falta potencial** |
| **Grados internos Z3/U(1)** | No conservado bajo interacción | No SU(3)×SU(2)×U(1); solo U(1) | Carga no discreta robusta | Circulación ± → Pauli emergente (hip.) | Nulls cosθ 0.613 vs 0.06 solo forma | **MEDIO** |
| **Escala dimensional** | Todo adimensional, falta MeV | Falta espín/estadística | T_c=1.13×10¹² K = 97 MeV | Masa como impedancia relacional | H=0.0025–0.005 banda; dens k3 0.0078–0.0367 | **MEDIO** |
| **Pre-átomos** | No medio colectivo; nudos aislados | No H, He ligados | Estado ligado k2+1 hipotético | Química pre-atómica k1+k2 “electrón vestido” | k1 mean 37.2; k2 11.7; k3 4.2 @ H=0.0025 | **ALTO** |

---

## 2. Propuesta integradora (Alexis / convergencia)

1. **Masa** no es `H×perim×f`; es **costo de reorganización / impedancia**:  
   - `m = H×perim / f` o  
   - `m = H×perim×(1−f)` (buffer; requiere f acotada).  
   k=3 con f≈0.45 estable → más masa si `/f`; k=1 con f≈1.09 → más ligero.  
   Invierte jerarquía hacia ~0.2, **no** a 0.00054 → hace falta estado ligado k2+1.

2. **K_c:** reescalado termodinámico `K(N)=K0·ln N` (K0 fijo, N variable) — prueba si |M| deja de caer 0.259→0.062.

3. **F vector:** medir dinámico con λ=1.5 forzando 14→10, trackear vecino k3 (dx, v) — cierra honestamente si F es real o papel.

---

## 3. Tanda de 8 experimentos (diseño)

| Exp | Qué mide | Protocolo (resumen) | Éxito (diseño) | Tiempo | Riesgo |
|---|---|---|---|---|---|
| **E1** | m=H·perim/**f** | 30×30, H=0.0025, 3 semillas; f=std/mean | ratio k1/k3 &lt;0.2 | 2h | Bajo |
| **E2** | m=H·perim·(1−f) o variante buffer | mismos datos f | ratio &lt;0.1 | 2h | Bajo |
| **E3** | Estado ligado k2+1 | dist&lt;3, coexistencia &gt;96% 400 pasos; perim_conj | m_par/m_k3 &lt;0.1 | 6h | Medio |
| **E4** | F dinámico λ=1.5 | 50×50; forzar 14→10; track k3; F=m·a | dx,v,F medidos ≠ solo teórico | 4h | Bajo* |
| **E5** | K(N)=K0·ln N | N=900,2500,10000; |M| | |M| no cae / colapso escala | 6h | Bajo |
| **E6** | V(r) k3-k3, k3-k1 | r=2–10; Δperim·H | lineal vs 1/r | 8h | Medio |
| **E7** | Pre-átomo H/He | k3+k1; E_bind | E_bind&gt;0 estable | 6h | Medio |
| **E8** | T_c→MeV | ancla 97 MeV; m→MeV | factor &lt;10 de 938 | 2h | **Alto** |

\*E4 resultó alto en práctica (ver §4).

**Orden de diseño:** E1→E2→E4→E5→E3→E6→E7→E8.  
**Criterio cierre puente (diseño):** E1/E2 ratio&lt;0.1 **y** (E5 estabiliza |M| **o** E4 confirma F).

---

## 4. Resultados corridos: E1 + E2 + E4

### 4.1 Parámetros de f (Fase 2)

| especie | f |
|---|---:|
| k1 | 1.09 |
| k2 | 1.274 |
| k3 fluctuante | 1.18 |
| k3 stable | **0.45** |

> **Nota:** el diseño pedía E2 = `H·perim·(1−f)`. Con f&gt;1 en k1/k2, `(1−f)` es negativo.  
> La corrida reportó **`m = H·perim/(1+f)`** — variante distinta del diseño original.

### 4.2 E1 y E2 — masas y ratios

Grid **30×30** (no 100×100).

| Definición | H=0.0025 k1 | k2 | k3 stable | **ratio k1/k3** | k2/k3 | H=0.005 ratio k1/k3 |
|---|---:|---:|---:|---:|---:|---:|
| **E1** `H·perim/f` | 0.00917 | 0.01177 | 0.04444 | **0.206** | 0.265 | **0.206** |
| **E2** `H·perim/(1+f)` | 0.00478 | 0.00660 | 0.01379 | **0.347** | 0.478 | **0.347** |
| E0 `H·perim·f` (ref.) | — | — | — | ~0.46–0.80 | — | — |

**Lectura del equipo:**

- E0 daba O(1) (0.46–0.80).  
- **E1 mejora a 0.206** (~factor 4× hacia 1/1836) pero aún **~380×** lejos de 0.00054.  
- **E2 (0.347) peor que E1.**  
- Inversión `/f` **no suficiente** → se prioriza estado ligado (E3).

**Estatus formal (libro único):**

| Exp | Estatus | Nota |
|---|---|---|
| E1 | **PASS débil (lectura)** | Definición provisional de m; **no** “masa emergente” |
| E2 | **FAIL** | Peor ratio; fórmula ≠ diseño (1−f) |

### 4.3 E4 — F dinámico

**Setup:** 50×50, H=0.0025, 400 pasos, track k=3 perim=8, λ=1.5 + placa 10H.

| intervalo | observación |
|---|---|
| 200→250 | k3 en (14.33,7.33) dist 7.09; (20,31) dist 4.48 — **aparición/desaparición**, no desplazamiento continuo |
| 250→300 | dist 0.00 — **estable, no se mueve** |
| 300→350 | dist 0.00 — estable |

**Energía del forzamiento:**

\[
E_\lambda = \lambda \cdot[(14-10)^2 - 0] = 1.5 \cdot 16 = 24.0
\]

frente a \(F\cdot dx \sim 0.01\) → λ domina **~2400×**; el empuje real del protocolo es 24, no 0.01.

**Conclusión del equipo:**

- F=0.01 **teórico no se observa** dinámicamente.  
- k3 **no se desplaza espontáneamente** sin forzar 14→10.  
- λ=1.5 es brutal; si se reabre, λ∈0.1–0.3 + trigger térmico \(P\propto e^{-\Delta E/T}\).  
- Track-ID estable es prerequisito.

**Estatus formal:**

| Exp | Estatus | Nota |
|---|---|---|
| E4 | **FAIL limpio (valioso)** | Sin trayectoria de cuerpo; F de papel no confirmada |

*(Sin gráficos Meta de E4 en esta captura.)*

---

## 5. Próximo paso acordado en el hilo

1. **E3** — estado ligado k2+1 (co-localización 96%, masa de conjunto).  
2. **E5** — K(N)=K0·ln N y |M| vs N.

**No prioritario aún:** E6 V(r), E7 H/He, E8 MeV (hasta ver E3/E5 y no apilar sobre F inexistente).

---

## 6. Criterio de puente (actualizado post E1/E2/E4)

| Condición de diseño | Estado |
|---|---|
| E1/E2 ratio &lt; 0.1 | **No** (mejor: 0.206 con E1) |
| E4 confirma F dinámico | **No** (FAIL) |
| E5 estabiliza \|M\| | Pendiente |
| E3 m_par/m_k3 &lt; 0.1 vs null | Pendiente |

El puente de diseño **no está cerrado**. E1 aporta techo de lectura; E4 cierra en negativo el canal F suave con λ=1.5.

---

## 7. Checklist anti-repetición Cosmo (ancla)

Antes de firmar cualquier claim de esta tanda:

1. ¿Apagar el actor mata el observable?  
2. ¿El contador lee dinámica o forma/catálogo?  
3. ¿REAL/NULL son el mismo universo menos un factor?  
4. ¿El nombre (masa, F, barión, MeV) tiene el ingrediente?  
5. ¿El gate es un número de nuestro universo? → solo post-hoc.  
6. ¿Una redefinición de m mejoró el ratio sin dinámica nueva? → solo “lectura”.

Ver análisis comparativo en la sesión de trabajo Grok (no duplicado aquí como veredicto final del equipo).
