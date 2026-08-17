# Observaciones de campo — rezago lluvia → floración (real, en curso)

Registro simple de observaciones de campo de Alexis, para poner a prueba
(no para forzar) el rezago de ~90 días que el instrumento usa hoy
(`objetivoFloracionEmpirico`/`computeFloracion`, He et al. 2017) contra
datos reales de primera mano, con respaldo fotográfico. Primer punto de
control con dato de campo propio del proyecto, en tiempo real — no
in-sample, no retrospectivo.

**Cómo se usa este archivo**: cada evento es una entrada nueva, con los
hitos que se vayan confirmando (lluvia → primer botón → floración plena)
agregados a medida que pasan, nunca completados de antemano. Fotos como
respaldo verificable de cada hito.

---

## Evento 1 — Llay Llay, río Aconcagua (zona de transición)

- **Zona**: Llay Llay, en la zona de transición pegada al río Aconcagua
  (Región de Valparaíso) — **fuera de la ZHCS** (Huintil/Choapa) contra la
  que está calibrado el instrumento hoy. Referencia comparable, no prueba
  directa del modelo.
- **Lluvia intensa**: ~3 semanas antes del 09-ago-2026 → **c. 19-jul-2026**
  (fecha aproximada, a precisar).
- **Primer botón por abrir**: 09-ago-2026 (observado directamente,
  respaldo fotográfico) → **~21 días desde la lluvia**.
- **Floración plena (pendiente)**: por confirmar. Hipótesis de Alexis: la
  zona es más húmeda que Copiapó pese a ser estepa subdesértica, así que
  basta una lluvia — floración plena esperada para **septiembre 2026**
  (~45-60 días desde la lluvia), como ya se observa habitualmente en la
  zona central/Santiago.
- **Hipótesis de fondo de Alexis, real y testeable**: el rezago y la
  cantidad de lluvia necesaria varían con la aridez de la zona — en zonas
  más húmedas (Llay Llay) basta una lluvia y el rezago es más corto; más
  al norte (ZHCS, más árido) se necesita más lluvia acumulada y el
  desfase podría ser mayor. Esto conecta con H2 de Cosmoclima (la
  magnitud/distribución del pulso de lluvia importa, no solo el total) —
  ver `hipotesis_y_modelo_formal.md`.
- **Qué comparar cuando se confirme el hito de floración plena**: el
  modelo predice, para un pulso fuerte y sostenido, floración = 11,8% del
  techo a los 21 días, 35,8% a los 90 días (medido corriendo
  `computeFloracion()` en el motor Node el 09-ago-2026 — ver conversación
  de esa fecha). Si la floración PLENA de Llay Llay llega bien antes de
  los 90 días (ej. mediados de septiembre, ~55-60 días), sería evidencia
  real de que el rezago de 90 días (calibrado con He et al. 2017,
  probablemente sobre zonas más áridas) es demasiado lento para zonas más
  húmedas — no necesariamente que esté mal para la ZHCS.

---

## Viaje planificado — Desierto Florido, octubre 2026 (ZHCS)

Alexis va a ir al Desierto Florido real en octubre 2026 — esta vez sí
dentro (o cerca) de la ZHCS, la zona que el instrumento modela
directamente. Reporte de campo real pendiente. Ver conversación de
09-ago-2026 para la propuesta de qué instrumentos/muestras interesan.
