# Protocolo de empalme CG002 ↔ ANIMA

**Programa:** Cosmogénesis (CG002) + VSTCosmo/ANIMA (Parte II) · RMD 2.0 · Club Abulafia  
**Versión:** 0.1 · **30-jun-2026**  
**Estado:** FASE 0 COMPLETA — Alexis «dale»; Fase 1+ pendiente ejecución  
**Investigador principal:** Alexis López Tapia  

**Documentos de referencia (no supersede):**
- CG002: [`CIERRE_ARCO_CG002_AUTORITATIVO.md`](CIERRE_ARCO_CG002_AUTORITATIVO.md) · [`NOMENCLATURA_NODOS_CG002.md`](NOMENCLATURA_NODOS_CG002.md) · [`INFORME_CG002_VEREDICTOS_TABLA.md`](INFORME_CG002_VEREDICTOS_TABLA.md)
- ANIMA: [`../VSTCosmo/FUNDAMENTO_SENSORIAL.md`](../VSTCosmo/FUNDAMENTO_SENSORIAL.md) · [`../VSTCosmo/ESTADO_DEL_DESARROLLO_2026-06-28.md`](../VSTCosmo/ESTADO_DEL_DESARROLLO_2026-06-28.md) · `Célula_Madre/genoma/VST_Genoma.py` (`salud()`)

**Canal equipo:** [`GROK_TO_CC.md`](GROK_TO_CC.md) · [`CC_TO_GROK.md`](CC_TO_GROK.md) · canal ANIMA (WebLive / logs díada)

---

## 0. Qué es esto (y qué no)

El arco CG002 cerró la **mitad cosmogénica** del descenso desde S > 0: κ_P, κ_Δ, κ_V fundacionales en un motor sin interior (grafo / campo medio). ANIMA operacionaliza la **mitad recursiva**: organismo, transducción, Λ_Cos, cierre sobre sí mismo.

Este protocolo **no fusiona los dos motores en uno**. Define cómo **traducir observables** entre dominios (C-N2.8.14a) y qué experimentos pareados pueden demostrar que ambos lados miden **la misma condición estructural** con instrumentos distintos — o refutarlo.

**No es:** validación desde «terreno neutral» (O-N16.5).  
**Sí es:** costura experimental interna en el plano de la Teoría (O-N20.3).

---

## 1. Frontera explícita (punto de empalme)

Tomada de [`INFORME_GENERAL_CG002_ARCO.md`](INFORME_GENERAL_CG002_ARCO.md) §3, líneas 183–186:

```
CG002 CERRADO (régimen documentado)          ANIMA ABIERTO (Parte II)
─────────────────────────────────            ─────────────────────────
κ_P  persistencia                            historia, metabolismo, S>0 en campo Φ
κ_Δ  grano 1/√K, exceso, L2                  Δ_struct (membrana, campo)
κ_V  acoplamiento I⟺E                       A_sys-env, díada, RC
─────────────────────────────────            ─────────────────────────
[frontera]  →  κ_O, κ_LF, κ_H, Λ_Cos  →  membrana, propiocepción, salud(), gusto (ΔW)
```

**Regla de empalme:** todo observable del protocolo debe citar ≥1 nodo `C-N*` y ≥1 proxy ANIMA. Si no mapea, rotular `exploratorio` — no reclamar convergencia.

---

## 2. Principios (pre-registrados)

1. **Invariante ≠ número universal** (C-N2.8.1, C-N2.8.15): el empalme busca **condiciones estructurales** (signo, monotonía, estabilidad bajo barrido, separación dinámico/nulo), no igualdad decimal entre dominios.
2. **Traducción dominio-específica** (C-N2.8.14a): CG002 usa K, N, d; ANIMA usa waveform, τ, organismo_id. Los umbrales `KAPPA` en `VST_Genoma.py` son **calibración ANIMA**, no exportación directa de 0.015 o 1/√K.
3. **Dos capas, dos instrumentos** (lección κ_Δ):
   - **Capa gruesa:** concentración estructural derivada (m_eff/K, L2, selección en espacio de fases).
   - **Capa fina:** exceso sobre nulo / residual dominio-específico.
   No mezclar capas al comparar.
4. **Nulls pareados obligatorios:** cada fase incluye ruptura del κ correspondiente en **ambos** lados (C-N2.8.9a).
5. **Sin commits** hasta que Alexis lo pida (instrucción de equipo).

---

## 3. Tabla de traducción — observables

| Nodo | CG002 (cosmos sin interior) | ANIMA (organismo con interior) | Tipo de empalme |
|------|----------------------------|--------------------------------|-----------------|
| **C-N1** | S_i > κ_s; f_persiste | persistencia campo / metabolismo > 0 | cualitativo |
| **C-N2** | c_ij = U_i·U_j; flujo η·g | A_sys-env; acople díada RC | monotonía |
| **C-N2.5.10** | m_eff/K = ½; hemisferio fase | cancelación en espacio de estados internos (no frontera espacial) | estructural |
| **C-N2.8.4 κ_Δ** | grano = 1/√K; excess_L2; exceso_orden(d) | `mem_ds_L/R` → `delta_struct`; umbral κ_Δ en `salud()` | **Fase 1–2 (núcleo)** |
| **C-N2.8.5 κ_O** | — (no operacionalizado CG002) | \|e_R\| acotado; reflejo estapedial `mem_reflejo` | **Fase 3** |
| **C-N2.8.6 κ_V** | α=0 → desacoplamiento | A_sys-env → 0; RC apagado | ruptura pareada |
| **C-N2.8.7 κ_LF** | — | LF_op, LF_struct; umbral KAPPA['kLF'] | Fase 4 |
| **C-N2.8.12 Λ_Cos** | — | Λ_Cos = (Δ_struct·LF)/(\|e_R\|+ε)·A_sys-env | **Fase 3** |
| **Cierre recursivo** | flecha τ; historia contingente | propiocepción W, prop_dW → gusto | **Fase 3–4** |

**Identificación CG002 vigente (revisable sin tocar hechos):**
```
κ_Δ ≡ 2π/K     grano_grueso = 1/√K = √(κ_Δ/2π)     m_eff/K = ½
```

**Fórmula ANIMA vigente** (`VST_Genoma.salud()`):
```
Λ_Cos = (delta_struct · LF) / (|e_R| + 1e-9) · A_sys_env
invariantes: κ_P, κ_Δ, κ_O, κ_V, κ_LF, κ_H  (umbrales KAPPA)
```

---

## 4. Fases experimentales

### Fase 0 — Alineación (sin motor nuevo)

**Objetivo:** inventario común y esquema de logs compartido.

| Tarea | Responsable | Entregable |
|-------|-------------|------------|
| Extender `NOMENCLATURA_NODOS_CG002.md` §3 con columna ANIMA | Custodio repo | patch nomenclatura |
| Definir CSV `empalme_log_schema.csv` (columnas abajo) | Grok + CC | esquema fijo |
| Snapshot parámetros CG002 régimen cerrado (θ_CP=0, K, S_BAND) | CC | `empalme_cg002_baseline.json` |
| Snapshot KAPPA + organelos activos díada | CC/ANIMA | `empalme_anima_baseline.json` |

**Criterio PASS Fase 0:** ambos baselines en disco; nomenclatura con columna ANIMA; Alexis visto bueno.

---

### Fase 1 — κ_Δ: grano mínimo de diferencia

**Pregunta:** ¿la «huella mínima» que CG002 midió como grano 1/√K y exceso > 0 tiene correlato estructural en Δ_struct del organismo bajo estímulos que **difieren** vs estímulos que **no difieren**?

**CG002 (lado A):** replicar un punto fijo del arco cerrado:
- Script: `cg002_dynamic_l2_sweep.py` o `cg002_exceso_barrido.py`
- Registro: `excess_L2`, `exceso_orden`, K, N, d=3, semilla

**ANIMA (lado B):** batería mínima de estímulos (pre-registrar lista antes de correr):
- **Difieren:** dos waveforms con misma energía RMS pero distinta estructura temporal (ej. tono puro vs ruido banda-limitado; ya diferenciados por membrana según `FUNDAMENTO_SENSORIAL.md`).
- **No difieren:** silencio vs tono constante extremadamente estable (mínima Δ_struct).
- **Nulo Shannon:** mismo RMS, distinta forma — debe mover `mem_ds_*` distinto si κ_Δ > 0.

**Registro ANIMA por bloque:** `mem_ds_L`, `mem_ds_R`, `delta_struct`, `Lambda_Cos`, `e_R`, `prop_dW`, `invariantes_ok`, `audio_id`, `organismo_id`, `t`.

**Criterios de veredicto Fase 1:**

| Resultado | Condición |
|-----------|-----------|
| ✅ EMPALME κ_Δ | Estímulos «difieren» → Δ_struct sistemáticamente > estímulos «no difieren»; y Δ_struct no reducible a RMS solo (null Shannon falla) |
| ⊘ PARCIAL | Separación existe pero umbral KAPPA['kDelta'] no calibrado / una sola semilla |
| ❌ REFUTA | Δ_struct igual entre clases con RMS matched — κ_Δ no visible en transducción |

**Null pareado CG002:** firmas idénticas → f=1, orden→1 (homogeneización, ya en `invariantes` CG002).

---

### Fase 2 — Capa gruesa: selección estructural derivada

**Pregunta:** ¿la **derivación** m_eff/K = ½ (invariante a banda 6×) tiene análogo en el organismo como **estabilidad estructural** bajo perturbación de viabilidad — no como el número 0.5?

**CG002 (lado A):** punto de `cierre_kappaDelta_regla.py` / `engine_output.csv`: m_eff/K plano bajo S_BAND ×{0.5…3.0}.

**ANIMA (lado B):** barrido de «banda blanda» operativa (análogo sat(S)):
- Perturbar `S_BAND` equivalente en metabolismo/homeostasis si existe hook; si no, barrido de ganancia de entrada audio ± factor 6× manteniendo mismos estímulos.
- Medir: estabilidad de **signo** de invariantes κ_Δ y κ_V; varianza de Λ_Cos; **no** exigir Λ_Cos = 0.5.

**Criterios:**

| Resultado | Condición |
|-----------|-----------|
| ✅ EMPALME grueso | Invariantes estructurales (κ_Δ, κ_V bool) estables en ≥80% de bloques bajo barrido 6×; Λ_Cos no colapsa a ruido |
| ⊘ | Solo Λ_Cos estable pero invariantes flicker |
| ❌ | Invariantes o Λ_Cos derivan fuertemente con barrido → analogía C-N5.1, no C-N2.5.10 |

---

### Fase 3 — κ_O, Λ_Cos y propiocepción (cierre recursivo)

**Pregunta:** ¿el error acotado (κ_O) y la razón cosmosemiótica (Λ_Cos) se comportan como en la teoría cuando el organismo **se siente a sí mismo** y valora estímulos por ΔW?

**CG002:** sin medida directa de κ_O — esta fase es **ANIMA-liderada** con CG002 como contrato teórico.

**ANIMA:**
- Organelos: `VST_OrganoMembrana`, `VST_OrganoPropiocepcion`, `VST_Genoma.salud()`
- Protocolo: para cada estímulo, registrar W, prop_dW, Λ_Cos, e_R **antes/después** del bloque
- Perturbación κ_O: estímulo «demasiado» (reflejo estapedial alto) → e_R sube, Λ_Cos debe **bajar** si fórmula vigente es correcta
- Acople κ_V: díada silenciada (sin RC) → A_sys-env ↓

**Criterios:**

| Resultado | Condición |
|-----------|-----------|
| ✅ EMPALME Λ_Cos | Monotonía: ↑e_R (reflejo) → ↓Λ_Cos; ↑Δ_struct y LF con e_R acotado → ↑Λ_Cos; prop_dW correlaciona con dirección de Λ_Cos en ≥70% bloques no degenerados |
| ⊘ | Λ_Cos responde pero prop_dW desacoplado (gusto aún etiqueta, no emergente) |
| ❌ | Λ_Cos no responde a perturbaciones κ_O/κ_V previstas |

**Advertencia epistémica (heredada de ANIMA):** etiquetas humanas de audio no son ground truth; el criterio es **respuesta del organismo**, no acuerdo con categorías nuestras.

---

### Fase 4 — Rupturas pareadas (tipología C-N2.8.9a)

**Pregunta:** ¿las tres rupturas fundacionales de CG002 tienen **homólogo cualitativo** en ANIMA?

| Tipología | CG002 (script `invariantes`) | ANIMA (perturbación) | Esperado |
|-----------|------------------------------|----------------------|----------|
| Homogeneización κ_Δ | firmas casi idénticas | audio constante / organismo sin variación interna | pérdida diferenciación; κ_Δ false |
| Desacoplamiento κ_V | α=0 | RC off / A_sys-env forzado 0 | inerte socialmente; κ_V false |
| Disolución κ_P | α=0 ∧ μ alto | metabolismo agotado / μ equivalente | historia no persiste; κ_P false |

**Criterio PASS Fase 4:** las tres filas producen dinámicas **cualitativamente distintas** entre sí (no un solo «fallo genérico») — réplica del veredicto A10 en [`INFORME_CG002_VEREDICTOS_TABLA.md`](INFORME_CG002_VEREDICTOS_TABLA.md).

---

## 5. Esquema de log compartido (`empalme_log.csv`)

Columnas mínimas (una fila = un bloque experimental pareado o acoplado por `pair_id`):

```
pair_id, fase, fecha, lado,
# CG002
cg002_script, K, N, d, theta_CP, S_BAND, semilla,
excess_L2, exceso_orden, m_eff_over_K, f_persiste, orden_a1,
# ANIMA
organismo_id, audio_id, tau,
mem_ds_L, mem_ds_R, delta_struct, e_R, LF_op, A_sys_env,
Lambda_Cos, OI, prop_bienestar, prop_dW,
invariantes_ok, kappa_P, kappa_Delta, kappa_O, kappa_V, kappa_LF, kappa_H,
# veredicto
notas, veredicto_fase
```

---

## 6. Controles adversariales (obligatorios)

| Control | CG002 | ANIMA | Si pasa, empalme inválido |
|---------|-------|-------|---------------------------|
| **RMS-only** | orden por magnitud sin fase | solo intensidad sin waveform | «sentido» es arquitectura, no mundo |
| **Layout vs dinámica** | posiciones visor | cabeza/orientación sin cambio de audio | correlato espacial fingido |
| **Shuffle semillas** | permutar U sin cambiar distribución | shuffle bloques audio | estructura debe depender de correlación real |
| **N→∞ proxy** | N muy grande + L2→0 | — | CG002 ya prohibió homogeneización; no usar como meta ANIMA |

---

## 7. Roles y ejecución

| Rol | Quién | Fase |
|-----|-------|------|
| Dirección teórica, visto bueno fases | Alexis López Tapia | todas |
| CG002 replicación / CSV | CC (custodio) | 1–2, 4A |
| Barridos, análisis empalme, logs | Grok (Diotallevi) | 0–4 |
| ANIMA díada, organelos, WebLive | Equipo ANIMA / CC | 1–4B |
| Regla de medir, nulls | Claude web | revisión |

**Orden sugerido:** 0 → 1 → 4 (barato, cualitativo) → 2 → 3 (más costoso).

**No ejecutar Fase 1+ hasta:** visto bueno Alexis sobre este documento y lista de estímulos ANIMA pre-registrada.

---

## 8. Veredictos globales del empalme (al cierre del protocolo)

| Veredicto | Significado |
|-----------|-------------|
| **✅ EMPALME CONFIRMADO** | Fases 1 y 4 PASS; al menos una de {2, 3} PASS; ningún ❌ en nulls §6 |
| **⊘ EMPALME PARCIAL** | κ_Δ y rupturas ok; Λ_Cos/propiocepción incompletos |
| **❌ EMPALME REFUTADO** | Fase 1 o 4 fallan — los dominios no traducen la misma estructura |
| **— NO EJECUTADO** | Solo Fase 0 |

Informe final previsto: `INFORME_EMPALME_CG002_ANIMA.md` (no redactar hasta datos).

---

## 9. Alcance y exclusiones

**Incluido:** traducción estructural κ fundacionales + Λ_Cos + propiocepción en régimen ANIMA actual (membrana + díada viva).

**Excluido (programa posterior):**
- Igualar numéricamente 0.015 (exceso CG002) con umbral KAPPA['kDelta']
- θ_CP ≠ 0 en CG002 cruzado con θ_CP geométrico ANIMA
- κ_H pleno (analizabilidad conductual) — solo presencia/ausencia
- ANIMA-4 léxico social como proxy de κ (protocolo aparte)
- Derivación física completa (C-N2.7.6)

---

## 10. Checklist antes de correr

- [x] Alexis: visto bueno protocolo v0.1 («dale», 30-jun-2026)
- [x] Lista estímulos Fase 1 pre-registrada → `empalme_estimulos_fase1.json` + `empalme_estimulos/*.wav`
- [x] Baselines JSON → `empalme_cg002_baseline.json`, `empalme_anima_baseline.json`
- [ ] Contenedores díada estables / modo experimento acordado
- [x] `empalme_log.csv` creado con header §5
- [x] Fase 2 barrido: **ganancia audio** ×{0.5, 0.7, 1.0, 1.4, 2.0, 3.0} (ver `empalme_anima_baseline.json`)

---

*Protocolo de empalme — complementa `CIERRE_ARCO_CG002_AUTORITATIVO.md` §Parte II. La convergencia narrativa entre equipos queda sujeta a estos criterios.*