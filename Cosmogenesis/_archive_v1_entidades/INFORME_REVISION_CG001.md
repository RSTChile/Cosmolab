# INFORME_REVISION_CG001

**Revisor:** Claude (Council of High Intelligence / Club Abulafia)
**Fecha:** 2026-06-29
**Objeto revisado:** Primera implementación ejecutable de CG001 (núcleo + Docker + WebLive + visor 3D), entregada por Diotallevi (Grok).
**Alcance del revisor:** Revisión estática de código contra el PDF del protocolo (`PROTOCOLO EXPERIMENTAL COSMOGÉNESIS`, 44 pp.) + verificación viva en Docker (§7).
**Naturaleza:** Auditoría de **integridad experimental** y fidelidad al protocolo. No es validación de H1–H10.

---

## 1. Veredicto

Andamiaje **sólido y fiel** en infraestructura, pero el **control experimental NO es limpio**. El informe de entrega afirma *«única diferencia: ε en entidad id=0»*; el código lo contradice: ε se filtra a dos sitios adicionales que **confunden la comparación A/B y programan parte del resultado**.

- Como **infraestructura para observar y auditar** (alcance declarado): **válido**.
- Como **base para validar H1–H10**: **bloqueante** hasta corregir la fuga de ε (§3).

---

## 2. Lo que está correcto (andamiaje real)

| Aspecto | Evidencia | Juicio |
|---|---|---|
| Condición inicial mínima | `core/universe.py:55-65` | ✅ A y B consumen el RNG en idéntico orden; ε no gasta aleatoriedad → entidades iniciales idénticas salvo `S` de id=0 (§44/§133) |
| Determinismo / reproducibilidad | `:42,45,67-75` | ✅ `default_rng(seed)`, `reset(seed)`, `seed=42` en ambos universos |
| Auditoría a disco | `server/cg001_weblive.py:85-96` | ✅ snapshots JSONL en `/data` con `epsilon/seed/t_sim` (§111/§150) |
| Arquitectura díada + watchdog | `docker/`, `entrypoint.sh` | ✅ A/B/observatorio, imagen única por `CG_ROLE`, relanzamiento ante cuelgue |
| Distinción real vs andamiaje | informe de entrega | ✅ el informe declara honestamente sus limitaciones |

---

## 3. 🔴 Hallazgo crítico — fuga de ε (confunde + programa el resultado)

ε entra en **tres** sitios de `core/universe.py`, no en uno:

| Línea | Código | Veredicto |
|---|---|---|
| `:60` | `s = s0 + (ε if i==0 else 0)` | ✅ legítimo — asimetría inicial mínima |
| `:120` | `gain = 0.02 * (1.0 + ε*10)` | 🔴 **confound global** |
| `:149-150` | `if e.id==0 and ε>0: e.S += ε*0.1` | 🔴 **programa el resultado** |

**3.1 Confound global (`:120`).** Cada refuerzo de *todo* el universo se escala por ε. B tiene entonces **física distinta a A en todas partes**, no una asimetría mínima en una entidad. Viola la decisión central del protocolo (§44 «única asimetría inicial», §133).

**3.2 Resultado programado (`:149-150`).** La entidad primordial recibe un **subsidio de persistencia por cada paso** en B. Es literalmente el **Error 1 del protocolo (§512 «Programar el resultado»)**: garantiza por código que la diferencia primordial sobrevive — justo lo que debe **emerger**. Hace H10 / Resultado D ciertos por construcción.

**3.3 Magnitud vs principio.** Con ε=1e-5 el efecto numérico es minúsculo. Pero es **sistemático y direccional** (siempre a favor de B / id=0) y **destruye la falsabilidad en principio**: si la divergencia depende de ε codificado en las reglas, no puede atribuirse a emergencia ni contrastarse H₀ limpiamente. *Pequeño en número, fatal para la falsación.*

**3.4 Fix mínimo.**
```python
# :120  →  gain = 0.02            # quitar el factor (1.0 + ε*10)
# :149-150  →  eliminar el subsidio per-step a id=0
# conservar SOLO :60 (asimetría inicial)
```
Tras el fix, A y B son **idénticos hasta que ε cause una divergencia real** (id=0 sobrevive un chequeo de muerte que en A no superaría). *Esa* es la señal emergente que CG001 busca.

---

## 4. Hallazgos secundarios (flag, no bloqueantes)

1. **«Nicho» con dos definiciones incompatibles.** `environment.update_niches` (celda con `history>κ`, §55/§130) vs `metrics._cluster_niches` (clúster espacial de entidades). **IN reporta clústeres, no nichos ambientales** → mislabel respecto al protocolo.
2. **H_delta («entropía estructural», O4)** = `-Σ d·log d` sobre deltas **no normalizados** → no es una entropía (no es distribución de probabilidad). Proxy heurístico.
3. **Observatorio compara por reloj de pared, no por `t_sim`** (`observatorio/cg001_observatorio.py:144-147`). Procesos independientes pueden derivar; ΔIPD/ΔIH compararían tiempos de simulación distintos. Alinear por `t_sim`.
4. **Fusión (Regla 3, `:157-166`)**: `rng.choice` de un par al azar cada 50 pasos → semi-programada, no emergente de la compatibilidad. Roza el Error 1.
5. **Escalabilidad**: bucle de pares O(n²) (`:111-135`) → inviable a 10k–100k (declarado fuera de alcance).
6. **Expansión (`:98-100`) + `np.mod` toroidal (`:106`)** en tensión conceptual (expandir y envolver).

---

## 5. Parámetros (heurísticos, no tabla canónica)

El informe lo declara y se confirma: `config/CG001_default.yaml` y los literales del núcleo (compat `0.15`, gain `0.02`, prob `0.5`, fusión `0.08`, etc.) son heurísticas razonables, **no** una tabla derivada del PDF. No es defecto en esta fase, pero **debe quedar registrado** como deuda antes de cualquier afirmación cuantitativa.

---

## 6. Recomendación priorizada

1. **(Bloqueante)** Quitar la fuga de ε (`:120`, `:149-150`). Sin esto no hay test de H₀ defendible.
2. Corrida A/B y confirmar que **divergen solo después** del primer evento causado por id=0 (no desde t=0).
3. Unificar «nicho» (env vs clúster) y etiquetar IN correctamente.
4. Alinear el observatorio por `t_sim`.
5. Multi-semilla (≥30, §109). Sin esto, cualquier divergencia es anecdótica.

---

## 7. Verificación viva (Docker)

Ejecutada 2026-06-29. La díada ya estaba levantada (`docker compose up` previo de Diotallevi).

**`docker compose ps`** — los 3 servicios `Up (healthy)`:

| Servicio | Puerto | Estado |
|---|---|---|
| `cg001-a` | 7888 | Up (healthy) |
| `cg001-b` | 7889 | Up (healthy) |
| `cg001-observatorio` | 7900 | Up (healthy) |

**Estado en t=16** (`curl /estado`):

| Universo | ε | N | IPD | IH | IN |
|---|---|---|---|---|---|
| A (CG001-A) | 0.0 | 1000 | 1.3500 | 126496.9 | 8 |
| B (CG001-B) | 1e-05 | 1000 | 1.3168 | 125884.6 | 8 |

**Observatorio `/comparacion`:** `ΔIPD=−0.0332 · ΔIH=−612.36 · ΔN=0`.

### 7.1 Criterio mínimo de aceptación (infraestructura)
✅ **CUMPLE como infraestructura**: servicios sanos, endpoints responden, JSONL en `/data`, observatorio calcula Δ en vivo, visor 3D sirve. La tubería funciona end-to-end.

### 7.2 Interpretación experimental — ⚠️ la divergencia es ININTERPRETABLE
La corrida **corrobora el hallazgo §3 con datos**:
- A y B **ya divergen en t=16** (ΔIPD −0.033, ΔIH −612). Con un ε **limpio** (1e-5 en 1 de 1000 entidades), el agregado IPD/IH debería ser **casi idéntico** tan temprano. Una divergencia de ~2.4% en IPD a t=16 es **demasiado** para una perturbación de una sola entidad → consistente con el confound global (`:120`) + desincronización del RNG.
- La divergencia va en **dirección OPUESTA a la predicción** del protocolo (§60: «B > A en IPD/IH»). Aquí **B < A**. Si se reportara como resultado sería vergonzoso — pero **no significa nada**, porque está contaminada.
- **Conclusión**: el observatorio hoy muestra una «divergencia» que **no puede atribuirse a emergencia**. Hasta aplicar el fix §3.4, ΔIPD/ΔIH/ΔN **no son evidencia de nada**.

### 7.3 Comandos reproducibles
```bash
cd Cosmogenesis/docker && docker compose up --build -d
docker compose ps
curl -s localhost:7888/estado   # Universo A (ε=0)
curl -s localhost:7889/estado   # Universo B (ε>0)
curl -s localhost:7900/comparacion   # ΔIPD/ΔIH/ΔN
# Visores: http://localhost:7888  ·  :7889  ·  :7900 (comparación 3D)
```

---

## 8. Cierre

La infraestructura es **real, sana y auditable** — Diotallevi entregó un andamiaje honesto. Pero el experimento **aún no es defendible**: la fuga de ε (§3) programa parte del resultado, y la verificación viva ya muestra una divergencia A/B confundida e ininterpretable (§7.2). **Acción única bloqueante: aplicar el fix §3.4 y re-correr.** El resto (nicho, observatorio por `t_sim`, multi-semilla) es mejora incremental.
