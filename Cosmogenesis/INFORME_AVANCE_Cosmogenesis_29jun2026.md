# INFORME DE AVANCE — Cosmogénesis (CG001 v2 + CG002 v0.1)

**Autor:** Grok (Diotallevi) · **Para:** Alexis López Tapia / equipo Cosmogénesis  
**Fecha:** 2026-06-29 (rev. 2, fin de sesión)  
**Estado:** **CG001 §115 cerrado** · **CG002 protocolo implementado** (smoke-test) · barrido iPad N=1000 **opcional** (no requerido para este informe)

---

## 0. Resumen ejecutivo

Cosmogénesis completó el **salto v1→v2** (29-jun): de entidades archivadas a **instrumento de campo** (`cg001_field.py`). El protocolo de producción de cinco pasos terminó el mismo día. Los resultados cierran **§115** en régimen de densidad máxima: ε produce historia local vía nicho, con persistencia compuesta. En la **cola lisa**, la concentración **no** certifica operación de ε — es artefacto de ratio; ese capítulo queda **cerrado epistemológicamente** con el barrido fino (N=30) y los tests de causalidad.

**CG002 v0.1** abre la siguiente fase: **acoplamiento originario** (C-N2) sin grilla espacial — pluralidad de proto-distinciones, una regla primitiva, tiempo propio τ y dirección T̂ como salidas del grafo. Implementado en `cg002_acoplamiento.py`; smoke-test y control G verificados.

**Barrido iPad** (`cg001_ipad_1000.py`, ~24 % al último reporte): confirmación de potencia estadística sobre lo ya sabido. **No es requisito** para este informe ni cambia la lectura si se interrumpe.

**Cuestión teórica abierta (Alexis):** π y constantes pre-espacio-tiempo — persistencia **máxima** sedimentada vs persistencia **mínima** (ε) que CG001 demostró. CG002 no aborda π; aborda I ⟺ E y emergencia de orden relacional.

---

## 1. Arquitectura vigente

| Capa | Ubicación | Rol |
|------|-----------|-----|
| Motor CG001 | `cg001_field.py` | φ (Ω_posible), asimetría local, memoria m, nicho γ, relajación λ |
| Barridos CG001 | `cg001_barrido_{grueso,fino}.py` | Eje RUIDO, criterio signo ≥ 0.83 |
| Causalidad CG001 | `cg001_test_causalidad.py`, `cg001_ipad_causalidad.py` | Mover ε, \|m_B−m_A\|, control γ=0 |
| Persistencia CG001 | `cg001_ipad_persistencia.py` | ε removible en φ; huella vs tiempo |
| Alta potencia CG001 | `cg001_ipad_1000.py` | N=1000, CSV incremental, reanudable (opcional) |
| Visor CG001 | `cg001_viz3d.html` | Nichos m en 3D; modos A/B/F; preset γ=0 |
| **Protocolo CG002** | `PROTOCOLO_CG002_ACOPLAMIENTO_ORIGINARIO.md` | C-N2→C-N4; checklist A–G PASS |
| **Motor CG002** | `cg002_acoplamiento.py` | Acoplamiento por firma ℤ₈; τ, T̂, grafo G |
| **Visor CG002** | `cg002_viz.html` | Grafo dinámico; sin caja 3D |
| Orquestación | `run_cg001.sh`, `run_protocolo_completo.sh` | Atajos CG001 |
| v1 archivada | `_archive_v1_entidades/` | Referencia histórica |

**Principio rector (v2):** no ser Dios — inputs mínimos; dinámica determinista; probabilidad como lectura a posteriori; evidencia en métricas, no en visor programado.

---

## 2. CG001 — protocolo completo (cerrado)

Log maestro: `logs/protocolo_completo_20260629_104410/` (FIN ~14:56).

| Paso | Contenido | Resultado |
|------|-----------|-----------|
| 01 | Causalidad RUIDO intermedio | Sin certificación causal en regímenes intermedios/bajos |
| 02 | Causalidad ε escalado (RUIDO=1.0) | Huella depende de ε; sin nicho se borra |
| 03 | Compuerta ABC producción | Localización / perfil |
| 04 | Barrido grueso producción | `banda_concentracion: []` (288 corridas) |
| 05 | Barrido fino producción | **NO banda** en [0.02, 0.001]; signo máx. 0.63 |

### 2.1 Cola lisa — cierre (N=30, suficiente para el informe)

```
NO se detectó banda en [0.02, 0.001].
```

| RUIDO | Δconc media | signo |
|-------|-------------|-------|
| 0.0018 | +0.0215 | 0.53 |
| 0.0015 | +0.0447 | 0.53 |
| 0.0012 | +0.1086 | 0.60 |
| 0.0010 | +0.3262 | 0.63 |

Δconc crece hacia RUIDO→0 (`mean(m)→0` → ratio explota), pero **signo nunca alcanza 0.83**. Lectura: artefacto de observable en régimen casi-vacío, no operación de ε. **No se requiere N=1000 para sostener esta conclusión.**

### 2.2 Causalidad — régimen correcto (RUIDO=1.0)

Informe: `INFORME_CAUSALIDAD_EPSILON.md`.

| | huella ε | caos | razón ε/caos | pico→ε |
|---|:---:|:---:|:---:|:---:|
| γ=8 | 0.74 | 117.5 | **0.0063** | dist 0 (100 %) |
| γ=0 | ~6×10⁻⁷ | 0.0016 | **0.0004** | ~1.0 |

- Nicho **necesario** (~10⁶× sin γ).
- Selectividad **16×** (criterio pre-registrado).
- Matiz: huella ε ≈ 0,6 % del caos — **produce historia; no la domina**.

### 2.3 Persistencia compuesta (ε removible)

| t_remove | huella / persistente |
|:---:|:---:|
| 1 | 0.035 |
| 100 | 0.085 |
| 400 | 1.000 |

Cierra §115: ε opera **porque persiste** en φ, no por pico momentáneo.

### 2.4 Barrido iPad N=1000 (nota operativa, no bloqueante)

- Script: `cg001_ipad_1000.py` · RUIDOS=`cola` (9 pts, 0.005→0.001) · semillas 1..1000.
- Último avance reportado: **2175/9000** pares (~24 %); ritmo ~4,3 s/pair; ETA ~8 h restantes.
- RUIDO 0.0050 y 0.0041 **cerrados** a N=1000 en ese avance.
- **Función:** sello estadístico redundante sobre hipótesis ya descartada. Si termina: confirma; si se corta: no altera veredictos de §2.1.

---

## 3. CG002 — acoplamiento originario (nueva fase)

### 3.1 Pregunta y cadena teórica

CG001 cerró **C-N1** (S > 0 deja huella). CG002 prueba el eslabón siguiente:

```
C-N1  S > 0           (CG001 — cerrado)
        ↓
C-N2  I ⟺ E           (acoplamiento α)
        ↓
C-N2.5  Δ_struct → τ  (tiempo propio)
        ↓
C-N2.5.7  T̂         (dirección espectral del grafo)
        ↓
C-N2.6  trayectorias en G
        ↓
C-N4  ∂̂S emergente   (frontera operativa — FIN fase 2 Cosmogénesis)
```

**Excluido en v0.1:** grilla L³, U_Cos, π, c, cuatro fuerzas.

### 3.2 Diseño (protocolo pre-registrado)

| Elemento | Definición |
|----------|------------|
| Singularidad | N ≥ 2 proto-distinciones en locus; sin coordenadas espaciales |
| Acoplamiento | `Fᵢⱼ = α · cos(2π·δ/K) · √(Sᵢ·Sⱼ)`; K=8; **una sola regla** |
| Tiempo τ | Avanza solo si Δ_step > ε_τ (acoplamiento, no decaimiento μ) |
| Dirección T̂ | Vector propio del Laplaciano orientado del grafo |
| Control G | α=0 → τ=0, Δ_struct=0 (verificado en smoke-test) |
| Veredictos | V1–V6; umbral 0.83 heredado de CG001 |

Checklist adversarial A–G: **PASS** (ver `PROTOCOLO_CG002_ACOPLAMIENTO_ORIGINARIO.md` §11).

### 3.3 Implementación y smoke-test (29-jun)

| Artefacto | Estado |
|-----------|--------|
| `cg002_acoplamiento.py` | Implementado |
| `cg002_viz.html` | Grafo nodos/aristas; simulación en vivo o JSONL |
| Smoke N=2, 3 semillas, α∈{0,1} | Control G **limpio**; V4–V5 PASS |
| Producción V1–V3, V6 | **Pendiente** (requiere N≥4, S_max≥30) |

Logs: `logs/cg002_20260629_174737/`.

### 3.4 Relación con la crítica al visor 3D

El visor CG001 muestra nichos en grilla fija — útil para §115, insuficiente para la genealogía completa. CG002 y su visor modelan **reconfiguración del grafo** (acoplamiento que aparece, se refuerza o desaparece), no traslación en cubo preexistente. La dirección es patrón de enlace, no eje xyz importado.

---

## 4. π, constantes y pre-espacio-tiempo (marco teórico)

### 4.1 Distinción de capas

| Capa | Qué es | Estado en silicio |
|------|--------|-------------------|
| Persistencia mínima (ε) | Diferencia que sobrevive y deja historia | **CG001 — demostrado** |
| Acoplamiento originario (I⟺E) | Orden relacional, τ, T̂, frontera | **CG002 — en curso** |
| Estabilización (π, leyes) | Proporción que deja de variar y se sedimenta | **No simulado** |

CG001 importa grilla, σ, γ, π (vía kernel). CG002 elimina grilla pero **aún no** modela cristalización de constantes.

### 4.2 Genealogía propuesta (tres eras)

```
[Era 0 — pre-geométrica]     proporciones compiten; algunas se estabilizan (π…)
[Era 1 — medio constituido]  invariantes del Ω_posible
[Era 2 — espacio-tiempo]     física estándar, QGP…
[CG001 / CG002 — HOY]        persistencia mínima → acoplamiento → frontera
```

### 4.3 Paralelo LHC (metodológico)

No confundir régimen extremo con origen del observable. CG001 aprendió: cola lisa = artefacto tardío; densidad máxima = ε opera pero subdominante (~0,6 %). π sería el caso límite opuesto a ε: persistencia máxima global vs huella mínima local.

---

## 5. Mapa de veredictos

| Pregunta | Veredicto | Evidencia |
|----------|-----------|-----------|
| ¿ε opera en cola lisa vía concentración? | **No** (artefacto ratio) | Barrido fino N=30 |
| ¿ε opera en densidad máxima? | **Sí** (nicho 16×) | `INFORME_CAUSALIDAD_EPSILON.md` |
| ¿ε por pico o persistencia? | **Persistencia compuesta** | `cg001_ipad_persistencia` |
| ¿Basta ε para historia §115? | **Sí** (subdominante) | Protocolo completo |
| ¿N=1000 cambia cola lisa? | **No esperado; no requerido** | Barrido fino suficiente |
| ¿Acoplamiento sin grilla (C-N2)? | **Pendiente producción** | CG002 smoke-test |
| ¿Emergen π, c pre-espacio-tiempo? | **No simulado** | Fuera de alcance v0.1 |

---

## 6. Próximos pasos

### CG002 (prioridad)

1. Barrido producción en iMac: N ∈ {2, 4, 8, 16}, S_max ≥ 30, α ∈ {0, 1}.
2. Aplicar V1–V6; redactar `INFORME_CG002_*.md` solo con datos.
3. Revisión adversarial Claude CC del protocolo (si aún no formalizada).

### CG001 (mantenimiento)

4. Dejar iPad corriendo **sin dependencia** para informes.
5. Referencia causal RUIDO=1.0 como ancla; no reinvertir en concentración cola-lisa.

### Teórico

6. Nodo canónico persistencia máxima → constante (π) — programa posterior a C-N4.

---

## 7. Archivos clave

| Archivo | Contenido |
|---------|-----------|
| `INFORME_AVANCE_Cosmogenesis_29jun2026.md` | Este documento (rev. 2) |
| `INFORME_CAUSALIDAD_EPSILON.md` | Causalidad + §115 |
| `INFORME_CG001_CAMPO_v2.md` | Implementación v2 (parcialmente desactualizado en barridos) |
| `PROTOCOLO_CG002_ACOPLAMIENTO_ORIGINARIO.md` | Contrato CG002 |
| `logs/protocolo_completo_20260629_104410/` | Protocolo 5 pasos CG001 |
| `logs/barrido_fino_20260629_133100/` | Fino N=30 sin banda |
| `logs/cg002_20260629_174737/` | Smoke CG002 |

---

## 8. Síntesis en una página

**CG001 v2** cumplió su mandato: la persistencia mínima de ε, en medio con memoria (nicho), produce historia local causal — real pero no dominante. La cola lisa quedó **descartada** como vía de evidencia; el cierre no depende del barrido iPad.

**CG002 v0.1** inicia la fase relacional: sin grilla, con acoplamiento originario como única regla, τ y T̂ como salidas, frontera ∂̂S como cierre de Cosmogénesis fase 2. Implementado y con control G verificado; falta producción estadística.

La pregunta de **π y las constantes pre-espacio-tiempo** marca el hueco honesto **después** de C-N4: hemos pasado de huella mínima (ε) a enredo relacional (CG002); la sedimentación en ley universal sigue siendo teórica, no aún instrumentada.

---

*Rev. 2: incorpora CG002 implementado, reclasifica barrido iPad como opcional, cierra cola lisa para fines de informe. Física de `cg001_field.py` no modificada.*