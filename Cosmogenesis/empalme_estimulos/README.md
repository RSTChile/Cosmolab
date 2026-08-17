# `empalme_estimulos/` — estímulos de audio para el puente CG002 ↔ ANIMA

**Qué es:** archivos WAV generados por `empalme_generar_estimulos_fase1.py` (raíz del proyecto) como
Fase 1 del `PROTOCOLO_EMPALME_CG002_ANIMA.md` — el protocolo que traduce observables entre el arco
Cosmogénesis (CG002: S>0, κ_P/κ_Δ/κ_V, motor sin interior) y ANIMA (VSTCosmo/Célula_Madre: organismo,
transducción sensorial, Λ_Cos). **No fusiona los dos motores**; define experimentos pareados que
prueban si ambos lados miden la misma condición estructural con instrumentos distintos.

**Estado (según el protocolo):** Fase 0 completa, Alexis dio luz verde; Fase 1+ pendiente de ejecución.
**Confirmado con Alexis (19-jul-2026): en espera A PROPÓSITO — se retoma cuando Cosmogénesis (el arco
CS072/CS073) cierre, no antes.** No es una tarea olvidada ni abandonada.

## Los 9 estímulos (nomenclatura `<letra><nº>_<descripción>`)

| prefijo | rol | archivos |
|---|---|---|
| `E` — estímulo | las condiciones experimentales centrales | `E01_tono_220_sostenido` (tono puro, estructura estable), `E02_ruido_banda_amplia` (ruido blanco, RMS igualado a E01 post-hoc), `E03_pulsos_220_01s` (pulsos, no sostenido) |
| `N` — control negativo/nulo | condiciones sin la estructura que se prueba | `N01_silencio`, `N02_tono_ultra_estable` |
| `S` — control de igualación (matched) | aísla si el efecto es sólo RMS/energía, no estructura | `S01_rms_match_tono_vs_ruido`, `S02_rms_match_440_vs_220` |
| `R` — extremos de ruptura | condiciones límite (saturación, homogeneización) | `R01_saturacion_colapso`, `R02_homogeneizacion_dc` |

Esta es la misma lógica REAL/NULL/control que gobierna el resto del programa (incluido CS072/CS073):
cada estímulo "positivo" (E) tiene su contraparte de control (N, S) para que el efecto medido en ANIMA
no pueda explicarse por energía/RMS a secas.

**Para el diseño experimental completo (qué mide cada par, qué observable de ANIMA se traduce a qué
observable de CG002)**, ver `PROTOCOLO_EMPALME_CG002_ANIMA.md` en la raíz — esta carpeta sólo contiene
el audio ya renderizado.
