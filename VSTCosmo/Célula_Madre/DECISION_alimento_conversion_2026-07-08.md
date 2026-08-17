# DECISIÓN — Alimento canónico = conversión (ICR/ICES)

**Fecha:** 2026-07-08  
**Estado:** **ADOPTADO** (basal / canónico)  
**Ámbito:** organismos A/B/C/D (Docker) y E-Planta en Pi nativa (`OrganeloMetabolismo`; `ANIMA_MET_ALIMENTO=conversion`)  
**Autores de la verificación:** Grok + medición en vivo + revisión Teoría CS  
**Para:** Club Abulafia / equipo ANIMA (CS, CC, Codex, humanos)

---

## 1. Decisión en una frase

**El organismo se alimenta de lo que ya convirtió en sentido** (`ICR` / `ICES` endógeno), no de un duelo impuesto `ICR > IRDE`.

```text
canónico:  nutricion = ICR_ratio · es_norm · (1 − saciedad)
legacy:    nutricion = max(0, IM − im_piso) · es_norm      # IM = EMA(ICR_ratio − IRDE_ratio)
```

Toxicidad en canónico: **disipar cuesta** (`∝ IRDE_ratio · es_norm`), no **veta** la conversión.

---

## 2. Por qué se abrió el caso

Síntoma reportado y medido: organismos con **hambre crónica** (`met_hambre ≈ 1`, `met_energia ≈ 0`, `met_ingesta = 0`) aunque hubiera mundo sonoro y `RC_total > 0`.

Documento previo relacionado: `NOTA_hambre_mundo_2026-07-02.md` (mundo mudo, IM negativo, knob `im_piso`).  
Esa nota diagnosticó bien el **régimen**; la presente decisión corrige la **definición de alimento** respecto de la teoría.

---

## 3. Revisión desde la Teoría Cosmosemiótica

Fuente: *Teoría_Cosmosemiotica_Integrada_FINAL_.pdf* (O-N1, NE23, ley de conservación).

| Teoría | Significado |
|--------|-------------|
| `RC = ICR + IRDE` ≡ `ES = ICES + IDES` | Toda la magnitud disponible se resuelve en **conversión en sentido** o **disipación**. Nada desaparece. |
| **ICR / ICES** | Fracción / índice de **conversión de ruido/contexto (ES) en sentido**. |
| **IRDE / IDES** | Fracción **no convertida** (disipación / riesgo residual). |
| Tesis operativa | El ruido no se purga: se **incorpora y se convierte en estructura**. |
| Alimento semiótico | **Lo convertido** — no “ganar por mayoría a la disipación”. |

Implicación directa:

- Si hay `ICR > 0` y hay ES en acto, **hubo sentido producido** → eso es alimento.
- Que `IRDE` sea alto es el **otro destino** de la misma ES (costo / riesgo), no un veto ontológico de nutrición.
- Exigir `ICR_ratio > IRDE_ratio` para comer es una **regla de diseño externa**, no la ley de conservación.

---

## 4. Qué estaba mal en el metabolismo (duelo)

### Fórmula histórica

```text
IM = EMA(ICR_ratio − IRDE_ratio)
nutricion = max(0, IM − im_piso) · es_norm
```

Con `im_piso = 0` (canónico CS del 2-jul para el lazo):

```text
solo hay ingesta si ICR_ratio > IRDE_ratio
```

### Efectos medidos (8-jul-2026)

1. **Conservación RC OK** en runtime: `|ICR + IRDE − RC_total| = 0`.
2. **Trampa de desarrollo:** organismo joven (soporte bajo → IRDE gana) → `IM < 0` → `ingesta = 0` → `E → 0` → hambre 1 → peor soporte.
3. **Biografía A (fisiología 18:00):** ~**70%** de filas con `met_hambre == 1`; en **~52%** de filas había `ICR_ratio > 0.01` y aun así `met_IM ≤ 0` (conversión sin comida).
4. El comentario del código decía “nutre lo que CONVIERTE (ICR>IRDE)” — el paréntesis **reescribía** la teoría.

El organelo **no estaba roto**; la **puerta de nutrición** no era la de la teoría.

---

## 5. Experimento A/B (8-jul-2026)

### Diseño

| Modo | Env | Nutrición | Toxicidad |
|------|-----|-----------|-----------|
| **duelo** (control) | `ANIMA_MET_ALIMENTO=duelo` | `max(0, IM)·es_norm` | `∝ max(0, −IM)·es` |
| **conversion** (tratamiento) | `ANIMA_MET_ALIMENTO=conversion` | `ICR_ratio·es_norm` | `∝ IRDE_ratio·es` |

Misma instrumentación, mismos organismos Docker, flag reversible.

### Resultado clave (fase conversion, ~70 s post-arranque, t≈2–6 s)

| Criterio de aceptación (teoría) | Resultado |
|---------------------------------|-----------|
| Si `ICR > 0` y hay ES → alguna nutrición aunque `IM ≤ 0` | **100%** de las filas “trampa” (`ICR_ratio>0.01` ∧ `met_IM≤0`) tuvieron `met_ingesta > 0` (A: 11/11, B: 10/10, C: 11/11, D: 16/16) |
| Misma señal bajo duelo (offline) | **hyp_ingesta = 0** en A–D (IM negativo en todo el tramo) |
| Conservación RC | Intacta (no se tocó OrganoRC) |

Lectura: el experimento **cierra la contradicción** “hay conversión pero no come”.

En arranque joven el balance puede seguir ligeramente negativo (ICR_ratio bajo + disipación cara): eso es coherente — **poco convertido + mucha disipación = come poco y paga disipar**, no el veto absoluto del duelo.

---

## 6. Auditoría anti-Shannon (condición de adopción)

**Pregunta del autor:** adoptar conversion solo si no es Shannon encubierto (parámetros nuestros que dictan el contenido).

| Factor | Origen | ¿Shannon encubierto? |
|--------|--------|----------------------|
| `ICR_ratio` | OrganoRC: competencia endógena (sin ponderaciones manuales) | **No** |
| `RC_total` / ES | Energía/novedad del acto | **No** |
| `es_norm`, `k_ingesta`, `k_toxico` | Constantes **cinéticas** (ritmo) | **No** (escala; ya existían) |
| Setpoint de `E` | Ausente | **No** |
| Codebook de comidas que abre nutrición | Ausente | **No** |
| Puerta `IM > 0` (duelo) | Regla externa de validez | **Sí — legacy** |
| `im_piso` | Parche de umbral del duelo | **No se usa para nutrir en conversion** |

**Veredicto:** conversion es **más anti-Shannon** que el duelo: no inventa qué es comida; multiplica lo que el campo ya partió en ICR vs IRDE.

---

## 7. Implementación (estado del repo)

| Pieza | Detalle |
|-------|---------|
| Código | `organelos/VST_Metabolismo.py` — default `ANIMA_MET_ALIMENTO=conversion` |
| Compose | `docker/docker-compose.yml` — env canónico en A/B/C/D + bind-mount del organelo |
| Telemetría | columna / campo `met_alimento_modo` ∈ {`conversion`, `duelo`} |
| Legacy | `ANIMA_MET_ALIMENTO=duelo` (o `legacy`) restaura el comportamiento histórico |

### Revertir (si el equipo lo pide)

```bash
cd Célula_Madre/docker
ANIMA_MET_ALIMENTO=duelo docker compose up -d --no-deps --force-recreate \
  anima-a anima-b anima-c anima-d
```

### Confirmar en vivo

```bash
# debe mostrar conversion
curl -s http://127.0.0.1:7788/ultima_fila | python3 -c \
  "import sys,json; f=json.load(sys.stdin).get('fila') or {}; print(f.get('met_alimento_modo'), f.get('met_ingesta'), f.get('ICR_ratio'), f.get('met_IM'))"
```

---

## 8. Complemento basal (mismo día) — costo de vocalizar

**Problema:** con conversion, si `ICR` es alto la ingesta puede ≫ gasto y `E` se clampa en 1. El gasto de **hablar del banco** era `COSTO_USAR = 0` (solo se cobraba **acuñar**). Hablar no vaciaba la reserva.

**Cierre (basal, anti-Shannon):**

| Concepto | Valor |
|----------|--------|
| `COSTO_USAR` | `0.010` por paso de **emisión** (antes 0) |
| `COSTO_CREAR` | `0.04` una vez al acuñar/emular (sin cambio) |
| Silencio (`voz_emitida='-'`) | costo 0 |
| Escala | endógena: arousal del estado + `g_intensidad` del gesto (no la etiqueta del sample) |
| Cableado | `registrar_costo_emision` → `met_costo_extra` → gasto metabólico (lag 1 paso) |
| Env opcional | `ANIMA_COSTO_VOZ_USAR` |

Archivos: `organelos/VST_OrganoComunicacion.py`, `web/VST_CelulaMadre_WebLive_{A,B,C,D}.py`.

Así la energía **se usa** al hablar/escuchar (perm) / disipar / vivir — alineado a “no saciedad permanente ociosa”.

---

## 9. Qué **no** cambia

- OrganoRC / ley `RC = ICR + IRDE` (sin tocar).
- Membrana, genoma, autostart social.
- Anti-setpoint de energía: `E` sigue emergiendo de balance.
- `im_piso` permanece disponible **solo** para el modo legacy `duelo`.

---

## 10. Seguimiento recomendado

1. Series longitudinales: `%` de filas con `ICR_ratio>0.01` y `met_ingesta==0` (debe ≈ 0 en conversion, salvo ES≈0).
2. `met_balance` medio en vida basal (¿se estabiliza E sin forzar setpoint?).
3. Con costo de voz: `met_costo_extra > 0` cuando `voz_emitida ≠ '-'`; E no debe quedar en 1 con bal≥0 eterno si hay emisión/perm.
4. Si la toxicidad por IRDE se ve demasiado agresiva en arranque: calibrar **solo** `k_toxico` (cinética), no reintroducir el veto.

---

## 11. Referencias

- Teoría: `Teoría_Cosmosemiotica_Integrada_FINAL_.pdf` — O-N1, NE23 (`RC≡ES`, `ICR≡ICES`, `IRDE≡IDES`).
- Antecedente hambre/mundo: `NOTA_hambre_mundo_2026-07-02.md`.
- Código: `organelos/VST_Metabolismo.py`, `organelos/VST_RC_A.py` / `VST_RC_B.py`.
- Orquestación: `docker/docker-compose.yml` (`ANIMA_MET_ALIMENTO`).

---

## 12. Resumen para el equipo

| Antes (duelo) | Ahora (conversion) |
|---------------|-------------------|
| Come solo si gana el duelo ICR vs IRDE | Come lo **convertido** (ICR), pague o no disipar |
| Hambre crónica frecuente con RC>0 | Trampa “convertí y no como” **cerrada** en experimento |
| Umbral de validez externo | Magnitudes endógenas del OrganoRC |
| Riesgo Shannon (imposición) | Más alineado a “ruido → sentido = alimento” |

**Decisión:** `conversion` es el **basal canónico** de ANIMA a partir de 2026-07-08.
