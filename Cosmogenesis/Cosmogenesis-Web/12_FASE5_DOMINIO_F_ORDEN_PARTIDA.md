# 12 — Fase 5 · Dominio F (físico pleno) · Orden de partida

**Fecha:** 2026-07-21  
**Base:** sistematización objeción Alexis + consenso 6/6 IAs  
**Precedente:** [11_RECHAZO_CIERRE_Y_REENCUADRE_DOMINIOS.md](11_RECHAZO_CIERRE_Y_REENCUADRE_DOMINIOS.md)

> ⚠️ **SUPERSEDIDO EN PARTE** por [13_FASE5_MF0_TRIADA_HOLISTICA.md](13_FASE5_MF0_TRIADA_HOLISTICA.md)  
> Alexis (y 7/7): no plasma solo ni MF-1→2→3 secuencial. **Tríada** Plasma + Expansión supralumínica + Tiempo genético desde t=0; m, K, V **simultáneos**.  
> Este doc 12 se conserva como historia del debate y por kill-switches/anti-Shannon aún válidos.

---

## 0. Decisión de marco (cerrada)

| Pregunta | Estado |
|----------|--------|
| A — Topológica: S>0 → persistencia | **Respondida** (dominio T cartografiado) |
| B — Física: ∇T plasma → pre-átomos | **No respondida** (instrumento era T, no F) |
| Opción A “cerrar Teoría” | **Rechazada** |
| Documentar T + límites como fase | **Sí** |
| Abrir dominio F | **Sí** |
| Más E con `m = H·perim·…` hacia 1/1836 | **No** |

**Capa 0 (control, no retocable para “arreglar” masa):**  
atractor k=3 / perim8 / banda H / U(1) forma / nulls del dominio T.

---

## 1. Los tres MF (consenso) y qué bloqueo atacan

| ID | Nombre | Bloqueo T que ataca | Riesgo Shannon principal |
|----|--------|---------------------|---------------------------|
| **MF-1** | Masa = inercia térmica / energía del dominio | Masa O(1), 380×, ligado más pesado | Definir ℰ, P sin meter m_e |
| **MF-2** | Carga Debye / K∝ρ_térmica | K_c crossover, no fase | Meter e, ε₀, λ_D de libro como perillas |
| **MF-3** | Confinamiento hidrodinámico V~σr | V(r) solo contacto r≤2 | Imponer σ o ecuación de estado “hasta que confinen” |

**Criterio de éxito F (igual que 11):**  
no 938 MeV exacto; sí — desplazamiento espontáneo, V_eff no solo contacto, orden extensivo, ligadura con kill-switch; anti-Shannon; sin gates MS.

---

## 2. Orden de partida recomendado

### Respuesta corta

```
F0  Sustrato termodinámico mínimo (estado del “plasma” del modelo)
  →  MF-1  Masa inercial térmica          ← PARTIR AQUÍ
  →  MF-2  Carga / apantallamiento / K(ρ_T)  (después de F0+MF-1)
  →  MF-3  Hidrodinámica / tensión / V(r)    (después de P,ρ estables)
```

**No** partir en paralelo los tres.  
**No** partir por MF-2 o MF-3 sin F0: ambos **presuponen** presión, densidad y temperatura como variables de **estado**, no como etiquetas de reporte de H.

---

### Por qué este orden (no otro)

| Orden tentador | Por qué no primero |
|----------------|-------------------|
| MF-2 Debye | Fórmula de libro con e, ε₀, n; fácil **importar** la física en lugar de emergerla; además K∝ρ_T exige ρ_T ya bien definida (F0) |
| MF-3 Hydro V=σr | Infraestructura pesada; sin EOS (P(ρ,T)) es teatro de símbolos; V_eff se mide **después** de tener P |
| MF-1 sin F0 | “∫ρ_ℰ dV / P” es circular si ℰ y P no existen como campos del modelo |
| Todo a la vez | Repite el error Cosmo: 23 piezas, contador confuso, firma prematura |

| Por qué **MF-1 primero** (tras F0) |
|-------------------------------------|
| Ataca el **bloqueo #1** (masa 380×) que unió a las 6 IAs |
| No exige U(1) de Maxwell completo ni fluidos 3D de entrada |
| Usa la plataforma T: dominios k=3 con var_in≈0 vs k=1 con var_in alta = **hipótesis de ℰ_int distinta** ya sugerida por T |
| Kill-switch claro: apagar gradiente de energía / P → la “masa” no puede ser inercia térmica |
| Si MF-1 **no** quiebra O(1)→jerarquía estructurada, MF-2/3 no salvan la tesis de pre-átomos por carga o burbujas |

---

## 3. F0 — Sustrato termodinámico mínimo (antes de MF-1)

**Objetivo:** que T, ℰ, P dejen de ser solo **mapeo de reporte** y pasen a ser **variables de estado** del modelo (aunque el modelo sea aún 2D de red o fluido de red).

### Contenido mínimo (borrador de contrato)

1. **Campo de temperatura** T(x) o equivalente de energía interna por celda/vínculo — evoluciona (no solo `T_fin(H)` cosmético).  
2. **Densidad de energía** ℰ (o u) y **presión** P ligadas por una EOS **declarada y barrida** (familia pre-registrada: p.ej. radiación-like P=ℰ/3, o politrópica con índice fijo a priori — no ajustada a 1/1836).  
3. **Expansión** que actúe sobre el estado (dilución / enfriamiento) de forma **consistente** con la EOS elegida (adiabasis como ley del modelo, no adorno).  
4. **Asimetría inicial** ∇T o ε_ℰ mínima — la misma pregunta fundacional, ahora sobre **estado energético**.  
5. **Capa T opcional como control:** se puede **sembrar** o **leer** dominios coherentes (ex-k=3) **sin** redefinir m por perim/f.

### Guardianes F0

| Guardián | Contenido |
|----------|-----------|
| G-ESTADO | T, ℰ, P entran en la dinámica; assert: cambiar EOS cambia evolución |
| G-NO-REPORTE | Ningún observable de éxito lee solo H etiquetado como “10¹² K” |
| G-NULL-ASIM | REAL = ε_ℰ>0; NULL = barajado / ε=0 |
| G-T-CONTROL | Si se usa k/perim de la capa T, es **lectura**, no motor de masa |

**Éxito F0 (andamiaje, no física final):**  
persistencia de asimetría energética bajo expansión (reformulación de W-01 en variables de estado); ℰ y P no triviales; NULL colapsa la señal.

**Tiempo estimado:** corto (diseño + smoke + 1 barrido ε_ℰ × H_phys).

---

## 4. MF-1 — Masa inercial térmica (primera física de F)

### Hipótesis (lenguaje de la Teoría + físico)

Un dominio coherente (ex-k=3, var_in≈0) almacena **energía interna** que **cuesta reorganizar** frente a una presión de medio; un dominio fluctuante (ex-k=1) disipa.  
La “masa” no es perímetro: es **impedancia energética** / inercia efectiva:

\[
m_{\mathrm{fis}} \;\sim\; \frac{\mathcal{E}_{\mathrm{int}}(\mathrm{dominio})}{P_{\mathrm{plasma}}/\mathrm{escala}} 
\quad\text{(forma exacta a fijar en pre-registro; no la del MS)}
\]

La integral del equipo \( \int \rho_ℰ\,dV / P \) es la **familia** correcta; la normalización se elige **ciega** y se barre.

### Protocolo mínimo

| Pieza | Spec |
|-------|------|
| Input | Estado F0 + dominios (de T o redetectados por coherencia energética) |
| Observable A | m_fis(k≈3) / m_fis(k≈1) — ¿jerarquía **estructurada** (&lt;0.1 o al menos ≪ O(1) de T)? |
| Observable B | Respuesta a empujón: desplazamiento / retraso del centroide del dominio (¿hay inercia?) |
| NULL | barajar ℰ dentro del dominio; o apagar gradiente de P |
| Kill-switch | sin P o sin ℰ_int diferenciada → no hay jerarquía de m_fis |
| **No gate** | 1/1836, 938 MeV |

### Desenlaces pre-escritos

| Código | Resultado | Consecuencia |
|--------|-----------|--------------|
| F1-A | Jerarquía estructurada + respuesta inercial + kill-switch | **Abrir MF-2** (carga sobre medio que ya tiene masa térmica) |
| F1-B | Inercia sí, jerarquía aún O(1) | Documentar; **no** forzar f; decidir si EOS/familia de m_fis o pasar a MF-3 con P |
| F1-C | Como T (solo geometría renombrada) | F0 mal implementado o masa térmica no es el canal; **no** empujar 1/1836 |

---

## 5. MF-2 — Carga / Debye (segundo)

**Solo después** de F0 (+ idealmente MF-1 al menos F1-B no catastrófico).

### Contenido legítimo vs contrabando

| Legítimo | Contrabando |
|----------|-------------|
| Densidad de “carga” **conservada** emergente o de la asimetría de fase ya vista en T | Fijar e, ε₀, n del MS para sacar λ_D “bonito” |
| Longitud de apantallamiento **medida** del decaimiento de correlaciones | Escribir K = e²/(4πε₀λ_D kT) con constantes de libro no derivadas |
| K_eff(ρ_T) que **colapsa curvas** \|M\|(N) | Ajustar K hasta \|M\|&gt;0.2 en N=100 |

### Éxito MF-2

- Orden extensivo o colapso de escala de \|M\| con variable reducida **emergente**  
- Neutralidad / continuidad de carga con kill-switch  
- No “Kuramoto con otro nombre”

---

## 6. MF-3 — Hidrodinámica / V(r) (tercero)

**Después** de tener P(ρ,T) creíble (F0) y preferible con inercia (MF-1).

### Idea

Romper un dominio coherente = crear “vacío” o interfaz en medio denso → costo ∝ perímetro o ∝ r si hay tensión efectiva; V_eff se **mide** (trabajo para separar), no se impone σr.

### Éxito MF-3

- V_eff(r) **no** plana para r&gt;2  
- Separar k=3 cuesta de forma monótona en r (familia lineal u otra — pre-registrar familias)  
- Kill-switch: P→0 o sin interfaz → V_eff colapsa a contacto T

---

## 7. Hoja de ruta operativa (partida)

| Paso | Qué | Gate de salida | Siguiente si PASS |
|------|-----|----------------|-------------------|
| **0** | Acta dominio T (ya casi en 08–11) | Un solo libro T | F0 |
| **1** | **F0** sustrato ℰ,P,T dinámicos + ε_ℰ | Persistencia energética vs NULL | **MF-1** |
| **2** | **MF-1** m_fis + empujón | F1-A o F1-B documentado | MF-2 |
| **3** | **MF-2** carga + apantallamiento medido | Colapso escala / conservación | MF-3 |
| **4** | **MF-3** V_eff hidrodinámico | V_eff no solo contacto | Informe contraste T vs F |
| **∥** | Paper dominio T en paralelo | — | No bloquea F |

### Orden de partida en una línea

> **F0 → MF-1 → (MF-2 ∥ diseño) → MF-3.**  
> Primera corrida de física: **MF-1**. Primera línea de código nueva: **F0**.

---

## 8. Qué no hacer en Fase 5

1. Reabrir E1–E10 con otra fórmula de perímetro.  
2. Gate 1/1836 o 938 MeV.  
3. E_bind = 0.9995·(m1+m2).  
4. Meter e, G, α_s, m_p como constantes libres “porque el plasma real las tiene” sin derivación o barrido ciego con NULL.  
5. Firmar “plasma QCD” porque T_mapeado dice 10¹² K.  
6. Romper la capa T de control para que MF-1 “salga”.

---

## 9. Respuesta directa a “¿Orden partida Fase 5 plasma?”

| Prioridad | Módulo | Rol |
|-----------|--------|-----|
| **1º** | **F0** estado termodinámico mínimo | Sin esto no hay plasma del modelo |
| **2º** | **MF-1** masa inercial térmica | Rompe (o no) el bloqueo 380× con kill-switch |
| **3º** | **MF-2** carga / Debye / K(ρ_T) | Solo con ρ_T y conservación reales |
| **4º** | **MF-3** hydro / V_eff | Solo con P de interfaz |

**Partida mañana (si se implementa):** especificación F0 + pre-registro MF-1 (observables, NULL, desenlaces F1-A/B/C) — **sin** codear aún Debye ni σr.

---

## 10. Mensaje equipo (copiable)

> Consenso 6/6 sobre la objeción Alexis: aceptado y canónico.  
> Dominio T = fase documentable (plataforma + límites). Teoría **no** cerrada.  
> Fase 5 orden: **F0 → MF-1 → MF-2 → MF-3**.  
> Primera física: **masa inercial térmica**, no más perim/f, no gates MS.  
> Topología T queda como **condición inicial / control**, no como objeto a forzar a 1/1836.
