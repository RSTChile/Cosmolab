# cs074-D — ¿La estructura vive en una banda estrecha no azarosa?

**Fecha:** 2026-07-29 · 2000 configuraciones LHS × 12 semillas × (REAL+NULL) = 48.000 corridas,
219.738 s (~61 horas), 1647/2000 configuraciones válidas (82,4%).

---

## PARÁ — esto es un desacuerdo con el diseño, no una respuesta a la pregunta

El barrido corrió completo y limpio, pero el resultado no se puede leer como "banda
estrecha / sin banda / disperso" (protocolo §6) porque **el control NULL no puede detectar
nada, en ningún punto del espacio, por una razón estructural — no porque no haya señal.**

## Lo que se encontró

`z` (real vs. NULL barajado) se quedó entre **−1,03 y +0,89** en las 1647 configuraciones
válidas (desviación estándar de toda la distribución de z: **0,18** — extremadamente
angosta). **Cero configuraciones cruzaron z>2, en cualquier dirección.** Varias filas
muestran REAL y NULL **idénticos bit a bit**:

```
z=0.00  real=0.1292±0.0286  null=0.1292±0.0286
z=-0.00 real=0.8083±0.0829  null=0.8083±0.0829
z=-0.00 real=0.8611±0.0661  null=0.8611±0.0661
```

## Por qué — verificado en el código, no es interpretación

`correr_holistico_energia()` genera las posiciones de los bariones con
`posiciones_escenario()` — un escenario 3D **uniforme al azar**, en una llamada separada,
con una semilla separada, **sin ninguna conexión con la densidad #23**. El barajado de
densidad (`seed_dens_null`) reasigna qué partícula tiene qué peso de masa, pero esas
partículas YA estaban en posiciones al azar independientes de la densidad. **Nunca hubo
coherencia espacial entre posición y densidad que el barajado pudiera destruir** — REAL y
NULL son, por construcción, dos muestras del mismo proceso puramente aleatorio.

Es el mismo patrón que ya encontramos en Enfoque 5 (la regla común ciega al barajado del
campo): el instrumento no puede ver lo que se le pide, sin importar cuánto barrido se
corra. `cs073_cierre_holistico.py` ya tiene la solución construida —
`semilla="causal"`, que siembra las posiciones desde la propia densidad vía una malla
causal (`malla_causal_atomos` + `layout_resortes`) — pero **nunca se incorporó al motor de
cs074_energia_holistica.py**. El diseño de este experimento asumía que el mecanismo NULL
ya validado en A/B alcanzaba; no alcanza para ESTA pregunta específica (coherencia
posición↔densidad), aunque sí sirvió para las preguntas de A/B (que nunca dependían de esa
coherencia).

## Lo que SÍ se puede rescatar de las 61 horas de cómputo

- El mapa completo de `frac_masa_ligada`, `n_clusters_finales` y `frac_masa_en_mayor_cluster`
  sobre las 1647 configuraciones válidas queda en disco — es un recurso real para
  responder OTRAS preguntas (p.ej. "¿qué tan seguido el motor basal falla en formar
  átomos?": 353/2000 configuraciones, 17,7%, fallaron por completo — concentradas, como en
  el smoke test, en `tasa_expansion` muy baja).
- El barrido confirma que el motor es estable en las 6 dimensiones a esta escala — 0
  fallas de conservación, 0 crashes, en 48.000 corridas.

## Qué se necesita para contestar la pregunta real

Re-correr (no todo el barrido, se puede acotar) con posiciones sembradas por la malla
causal (`semilla="causal"`, ya existe en `cs072_modulos/piezas/p_semilla_causal.py`) en vez
de posiciones uniformes — recién ahí el barajado de densidad destruye algo real, y el
z-score mide lo que el diseño pide medir.

**No se re-corrió por cuenta propia** (regla explícita del diseño: "un desacuerdo es un
dato, PARÁ y reportá"). Queda tu decisión: incorporar el modo causal y re-correr (con qué
alcance), o cerrar esta línea acá.

**Archivos:** `PROTOCOLO_cs074D_barrido_fino_banda_PREREGISTRO.md`,
`cs074D_barrido_fino_banda.py`, `resultados_cs074D_barrido_fino/cs074D_result_FULL.json`.
