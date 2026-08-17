# 08 — Tanda E1–E10, 3D, quiral y cierre del equipo

**Fuentes:** PDF sesión Web + reporte de corridas E1/E2/E4 (texto Meta)  
**Detalle numérico E1/E2/E4:** también [02_TANDA_E1_E4_resultados.md](02_TANDA_E1_E4_resultados.md)  
**Nota:** en el PDF muchas celdas salen “vacías” por export; donde hay números claros del hilo, se citan.

---

## 1. Diseño de la tanda (8 + 2 de ruptura)

| Exp | Objetivo | Gate de diseño (original) |
|-----|----------|---------------------------|
| E1 | m = H·perim/**f** | ratio k1/k3 &lt; 0.2 |
| E2 | buffer (1−f) o /(1+f) | ratio &lt; 0.1 |
| E3 | ligado k1–k2 “electrón vestido” | m_par/m_k3 &lt; 0.1 |
| E4 | F dinámico λ=1.5 | dx, v, F medidos |
| E5 | K(N)=K0·ln N | \|M\| no cae / fase |
| E6 | V(r) k3-k3 / k3-k1 | lineal σ vs 1/r |
| E7 | Pre-átomo H/He | E_bind&gt;0 estable |
| E8 | Ancla T_c → MeV | factor &lt;10 de 938 |
| E9 | Reescalado termodinámico alternativo | estabilizar \|M\| |
| E10 | Masa exponencial / buffer fuerte | jerarquía fuerte |

---

## 2. Resultados

### E1 — `m = H·perim/f` (30×30)

| | k1 | k2 | k3 stable | k1/k3 | k2/k3 |
|--|---:|---:|---:|---:|---:|
| m | 0.00917 | 0.01177 | 0.04444 | **0.206** | 0.265 |

- E0 (`×f`) daba ~0.46–0.80.  
- E1 **mejora ~4×**, sigue **~380×** lejos de 1/1836.  
- Mismo ratio a H=0.0025 y 0.005.

**Estatus:** PASS débil de **lectura** · FAIL de jerarquía física.

### E2 — `m = H·perim/(1+f)`

| k1/k3 |
|------:|
| **0.347** (peor que E1) |

Nota: diseño pedía `(1−f)`; con f&gt;1 era inválido → se usó `/(1+f)`.

**Estatus:** **FAIL**.

### E3 — Estado ligado k1–k2

- Co-localización **frecuente** (cientos de eventos en frames).  
- Masa de conjunto: ratio vs k3 **1.543 → más pesado**, no más ligero.  
- Buffer no da “electrón”.

**Estatus:** **FAIL** del gate de ligereza · **dato útil** (ligado no implica liviano).

### E4 — F dinámico

- k3: aparición/desaparición, **dx=0** en tramos largos.  
- λ=1.5 → E_λ=24 ≫ F·dx~0.01 (~2400×).  
- F teórica **no observada**.

**Estatus:** **FAIL limpio** (valioso).

### E5 — K(N)·ln N

- Con reescalado: mejora parcial de \|M\|, **sigue cayendo**.  
- No hay fase termodinámica estable reportada.

**Estatus:** **FAIL** / crossover finito.

### E6 — V(r) por perímetro

| r (celdas) | Δ perim / energía (relato) |
|------------|----------------------------|
| 1 | atractivo (contacto) |
| 2 | menos atractivo |
| 3+ | ~0 |

**No** potencial lineal de confinamiento.

**Estatus:** **FAIL** de QCD lineal · contacto sí.

### E7 / E8 / E10

- En el cierre del equipo: **no cierran** química/escala MeV/jerarquía exponencial de forma convincente.  
- E8 con T_c=97 MeV tratado como **alto riesgo** de destino Shannon; no se usa como victoria.

**Estatus:** **no cruce** (o no prioritario / no salvado).

### E9 — Reescalado alternativo

- Mejora vs sin reescalar; **sigue cayendo** con N.

**Estatus:** **FAIL parcial**.

### 3D

- mean/perim por k reportados en PDF (k1 grande en mean, etc.).  
- **Ratios de masa idénticos a 2D**.  
- Ventana de atractor **más estrecha** en 3D.

**Estatus:** 3D **no salva** la jerarquía.

### Circulación quiral / “Pauli impar”

| Regla | Pares misma fase (relato) |
|-------|---------------------------|
| Kuramoto simétrico | se repelen / no co-localizan como se esperaba de “solo quiral” |
| horario / anti | igual: **repulsión ya existe sin quiralidad** |

**Estatus:** exclusión **no** es efecto de fase quiral en este protocolo.

---

## 3. Diagnóstico E1–E5 (intermedio del hilo)

- E1 mejor, lejos  
- E2 peor  
- E3 ligado más pesado  
- E4 no dinámico  
- E5 no estabiliza  

→ masa y termodinámica **no cierran** con definiciones actuales.

---

## 4. Cierre total (12 experimentos / resumen ejecutivo)

### Hechos robustos (anti cherry-picking)

1. **Atractor topológico k=3** — banda H — recuperación en protocolos.  
2. **Forma emerge** vs nulls (no contrabando de contador solo).  
3. **3 valores / generaciones análogas** con acoplamiento + placa Gauss.  
4. **F vector teórico** no medido dinámicamente.

### Fallos que bloquean pre-átomos

1. **Masa no emerge** (ratios O(1); ligado pesado; 3D=2D).  
2. **K_c crossover** no fase (reescalados insuficientes).  
3. **Contacto ≠ confinamiento lineal**.  
4. **F dinámico no observado**.  
5. **Pauli/quiral** no añade exclusión nueva.

### Conclusión del equipo (párrafo final del PDF)

> Topología da plataforma de salida sólida (k=3, forma, 3 gen…), pero el **puente no cruza** a física pre-átomos con 2D/3D actuales. La masa requiere cancelación no topológica, o espín/circulación con exclusión real no medida aquí.  
> **Próximo paso honesto:** documentar límites (no cruce masa / F / V lineal / termodinámica) — o rediseñar (p.ej. 3D con buffer bien definido), sin vender victoria.

---

## 5. Criterio de cierre del puente (diseño vs realidad)

| Criterio de diseño | Realidad |
|--------------------|----------|
| E1/E2 ratio &lt; 0.1 | No (mejor 0.206) |
| E4 o E5 salvan dinámica/orden | **Ambos FAIL** |
| E3 ligereza | **FAIL** (más pesado) |
| E6 lineal | **FAIL** |

**Puente topología → pre-átomos: NO CRUZADO.**

→ Libro de claims: [09_LIBRO_DE_CLAIMS.md](09_LIBRO_DE_CLAIMS.md)
