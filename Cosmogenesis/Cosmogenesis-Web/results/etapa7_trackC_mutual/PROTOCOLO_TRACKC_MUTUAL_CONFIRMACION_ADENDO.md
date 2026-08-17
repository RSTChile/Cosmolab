# ADENDO de pre-registro — confirmación fuera de muestra de `mean_instant` (H4)

**Timestamp:** 2026-07-23 05:09 -04 (ANTES de correr la confirmación; H1/H2/H3 aún en curso al momento de escribir esto)
**Motivo:** el director señaló, correctamente, que probar 6 variantes de agregación de la misma métrica contra el mismo baseline de 10 semillas y quedarse con la única que cruza el umbral (`mean_instant`, mean R/S=1.218, rate=0.60) es indistinguible de "probar variantes hasta que una pase" — el mismo patrón de defecto ya auditado en el hallazgo v6 de masa/linaje, aquí trasladado a la forma del estadístico. `mean_instant` NO cuenta como RESUELVE hasta que replique fuera de muestra.

## Definición congelada (sin más ajustes)

`mean_instant` = promedio temporal, sobre los pasos E4, de la energía instantánea de TODOS los pares próximos del paso (mismo umbral de proximidad `GROUP_LINK_R=4.5` y `FORCE_CUTOFF=8.0` que v6, energía simetrizada `-0.5·G·(m_i·src_j+m_j·src_i)/r_soft`), **sin** el gate de persistencia de 5 pasos consecutivos que usa v6. Parámetros físicos: `FORCE_CUTOFF=8.0`, `SOFTENING=1.2`, `GRAV_START_FRAC=0.65`, `pasos=400`, `G=0.20` (idénticos al baseline v6; no se tocan aquí — eso es competencia de H1/H2).

## Semillas de confirmación (declaradas AHORA, nunca usadas antes en este proyecto ni en el set estándar de 10)

```
111, 222, 333, 444, 555, 666, 777777, 13, 31, 271828
```

## Criterio de decisión (idéntico al umbral pre-registrado original, sin relajar)

- **RESUELVE (confirmado):** mean(R/S) sobre las 10 semillas de confirmación ≥ 1.15 **Y** rate(R/S>1) ≥ 0.60.
- **NO REPLICA / candidato no confirmado (posible artefacto de selección de estadístico):** cualquier otro resultado. Esto se reportará así de claro, sin suavizarlo, si ocurre.

## Regla anti-Shannon aplicada aquí

No se prueban más variantes de agregación en este paso. Solo se corre `mean_instant`, tal cual quedó fija, sobre las 10 semillas nuevas. Si no replica, el veredicto de H4 pasa a "candidato no confirmado", y el veredicto global de Track C no cuenta a `mean_instant` como una reparación del canal E_mutual.
