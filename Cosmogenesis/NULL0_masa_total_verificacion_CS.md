# NULL-0 — masa total (Bloque 2.8, jerarquía de controles para blindar CS073)

**Fecha:** 5-ago-2026 · **Verificado por:** orquestador, directo, sin agente (chequeo trivial sobre datos existentes).

**Qué es NULL-0:** el primer escalón de la jerarquía de 6 controles propuesta por el roadmap multi-IA para
blindar el resultado de CS073 (z=48.69). Conserva únicamente la masa total del sistema; destruye todo lo demás
(posiciones, densidad, estructura). Sirve como chequeo de sanidad: si REAL y los NULL ni siquiera arrancan con
la misma masa total, cualquier comparación posterior está confundida desde el origen.

**Resultado:** las 9 corridas de la batería N=2000 (`ic_real` + `ic_null1`..`ic_null8`) arrancan con
**exactamente la misma masa total inicial: 2000 partículas × 9.4 = 18800.00**, sin excepción. Verificado
leyendo el volcado `cosmog_00000` de cada corrida con `leer_volcado_phantom.py` (campo `massoftype` en los
parámetros del dump, más conteo de partículas de gas).

**Lectura:** U0 pasa — no hay confusión de masa total entre REAL y NULL. La batería es comparable. No aporta
información nueva más allá de validar que las comparaciones anteriores (κ_P, κ_Δ, κ_V, z=48.69) parten de un
terreno parejo.

**Siguiente paso:** NULL-1 (distribución radial y densidad) requiere construir una condición inicial sintética
nueva — no está entre los datos existentes. Ver piloto en curso.
