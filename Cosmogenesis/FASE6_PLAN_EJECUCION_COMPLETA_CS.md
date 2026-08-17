# Fase VI — plan de ejecución completa (todos los experimentos pendientes, automático)

**Fecha:** 11-ago-2026 · **Encargo de Alexis:** *"haz un flujo de trabajo para ejecutarlos todos,
incluyendo los pendientes, automáticamente, con agentes, sin detenerte a analizar nada antes de terminar
y luego, una vez ejecutado se analiza"*.

**Regla de esta tanda:** se ejecuta todo primero, se analiza al final. Ningún experimento declara cierre ni
veredicto — cada uno deja su informe + CSVs en disco, y el análisis consolidado se hace cuando termine todo.

**Regla metodológica nueva, vigente desde ahora:** toda medición de diámetro usa `cs090_diam_corregido.py`
(componente gigante), NO el `_diam` de `cs055` (que queda sólo para reproducir resultados históricos).
Ver `FASE6_adopcion_diam_corregido_CS.md`.

---

## OLA 1 — Baratos, sin Phantom (4 agentes en paralelo)

| id | experimento | origen |
|---|---|---|
| **O1-A** | κ_V como métrica puente (#15) + recalibración del umbral 0.7 con Phantom como verdad de campo (#17) | 2do analista 4.1 + 5.1 |
| **O1-B** | Auditoría del bug de diámetro en Fase IV/`cs082` y línea CS07x-CS08x (no auditadas) + reinstalar `sarracen` | pendiente nuevo |
| **O1-C** | Cierre de A0: métricas nativas sobre TODAS las Clase II conocidas, no sólo 2 (#3) | los 3 analistas |
| **O1-D** | Factorial sistemático del trío: correcto / casi-correcto / al azar / sin tríos, n=30 (#4) | 2do + 3er analista |

## OLA 2 — Motor liviano, sin Phantom (4-5 agentes en paralelo)

| id | experimento | origen |
|---|---|---|
| **O2-A** | q_E×q_T con instrumento POSICIONAL (post-`layout_resortes`), que sí puede ver el orden (#6/#11) | GPT-5.6 Sol VII-A/F6-07 |
| **O2-B** | Genealogías independientes escaladas a 10-20 (hoy sólo 4) (#7) | los 3 analistas |
| **O2-C** | ¿`kcap` es un número especial o cualquier capacidad finita sirve? (#10) | GPT-5.6 Sol F6-06 |
| **O2-D** | Campo continuo 2D/3D (reacción-difusión) para A0, más allá del anillo 1D (#18) | 3er analista A4 |
| **O2-E** | Re-correr el barrido de Fase V-A guardando `seed` (cierra la única inferencia hoy indirecta) | pendiente nuevo |

## OLA 3 — Phantom (secuencial por pares, para no saturar CPU)

| id | experimento | origen | prioridad |
|---|---|---|---|
| **O3-A** | **Escalar resolución N=2000→4000→8000** sobre pares ya emparejados (#5) | GPT-5.6 Sol VI-B/F6-02 — *"obligatoria"* | **la más urgente** |
| **O3-B** | Control de rewiring preservando grado/N/aristas exactas (#9) | GPT-5.6 Sol F6-04 | alta |
| **O3-C** | Factorial mecanístico completo → Phantom (cadena mecanismo→geometría→gravedad) (#8) | GPT-5.6 Sol F6-03 | alta |
| **O3-D** | Barrido kcap/K directo en Phantom (#14) | 2do analista 1.1 | media |
| **O3-E** | O-N7.7: variante con memoria vs. sin memoria (#16) | 2do analista 4.2 | media |
| **O3-F** | Observable $B_\tau$ (branching de futuros) sobre gas no colapsado (#13) | GPT-5.6 Sol Fase VIII | media |

## OLA 4 — Checkpoint final

| id | experimento | origen |
|---|---|---|
| **O4-A** | Replicación con un integrador gravitacional INDEPENDIENTE de Phantom (#12) | GPT-5.6 Sol F6-08 — *"checkpoint final"* |

---

## Análisis consolidado (después de todo)

Cuando las 4 olas terminen: informe único para el equipo integrando todos los resultados, con árbol de
decisión al inicio (formato de `INFORME_EQUIPO_FASE5B_11ago2026_CS.md`, que funcionó bien), y la lectura de
qué se sostiene / qué se cae / qué queda abierto tras la batería completa.

## Reglas de la casa, vigentes en TODOS los experimentos de esta tanda

- No modificar ningún script congelado ni de tareas anteriores — código nuevo en archivos nuevos.
- Usar `cs090_diam_corregido.py` para toda medición de diámetro nueva.
- Phantom autorizado (Alexis lo autorizó explícitamente y pidió ejecutar todo).
- Verificación cruzada obligatoria contra `meta_regla.json` en cualquier tarea que genere pares para Phantom
  (lección del bug de colisión de nombres).
- Nombres de regla con prefijo de lote único, sin colisión con lotes previos
  (`r0-r19`, `r0-r39`, `batch3-*`, `batch4-*`, `*v1fix`, `*v2fix`, `*pendNEG` ya usados).
- No declarar cierre ni veredicto — sólo números; la interpretación es de Alexis, al final.
- No hacer commits de git.
- Explicaciones en lenguaje simple con analogías; código autodescriptivo.
