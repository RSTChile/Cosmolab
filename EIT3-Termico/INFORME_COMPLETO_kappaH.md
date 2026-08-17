# EIT-3 Térmico · κ_H — Informe de batería (tres rondas + cierre)

Estadística descriptiva únicamente. La interpretación queda para el investigador principal.

## Resumen

| Pieza | Estado | Nota |
|---|---|---|
| Batería 1 (v7.3) · Exp. A, B, C | **INVALIDADA** | El barrido arrastraba estado entre paradas; la "frontera invariante" era un artefacto del método. |
| Batería 2 (v7.4→v7.5) · validación | PASÓ | Reproducibilidad byte a byte, motor Node validado bit a bit, ambas rondas. |
| Batería 3 (v7.6.1) · eje real 0,60–1,40 | Corrida | Con el instrumento más corregido, la correlación huella↔entropía se desplomó a ≈0. La frontera con parámetros se sostiene. |
| Cierre · confound acoplamiento/huella | Resuelto | No es identidad forzada: depende de t_óptima. |
| Cierre · mecanismo del −0,756 original | Confirmado, parcial | El recorte de rango es real y mueve la correlación en la dirección correcta, sin reproducir el número exacto. |

*(Para el detalle completo de batería 1 y 2 —metodología común, Paso 0/1, throttling térmico, tablas de D/A'/B'/C de la batería 2— ver la versión anterior de este informe en `Old/INFORME_COMPLETO_kappaH.md`. Este documento se enfoca en lo nuevo: batería 3 y el cierre de investigación.)*

## Batería 3 (v7.6.1) — el eje real y tres correcciones más

**Qué cambió en el instrumento:**
- Réplicas de verdad: las 5 repeticiones de recuperación ahora parten del mismo estado exacto (snapshot/restore) — antes se encadenaban y sesgaban el tiempo medido.
- Asentar hasta el equilibrio real, en vez de un número fijo de pasos.
- Azar separado por parada y por fase (calibración/preasentamiento/recuperación/asentamiento/medición) — antes un solo flujo continuo corría el ruido de una parada a la siguiente.
- Eje corregido: el control de luminosidad siempre estuvo limitado a 0,60–1,40 por el `<input>` real; pedir valores de afuera hacía que el navegador los recortara en silencio mientras el CSV anotaba el valor pedido. v7.6.1 lo declara explícito.

**Topes rebajados con respaldo de datos**: la medición de recuperación agregó ~1,9× el costo de v7.5. Se midió la distribución real en 480 puntos de muestra antes de decidir: 0 casos de asentamiento/recuperación genuinamente lentos por encima de los topes reducidos propuestos (6.000/3.000 en vez de 20.000/20.000). Se bajaron los topes con ese respaldo y se corrió la grilla completa de 108 combinaciones igual.

**Resultados, comparados contra la batería 2:**

| | Batería 2 (eje 0,25–1,95) | Batería 3 (eje real 0,60–1,40) |
|---|---|---|
| D · correlación (parada/inicio) | 0,367 / 0,509 | 0,044 / 0,061 (≈0) |
| A' · correlación (30 semillas) | 0,375 ± 0,039 | 0,008 ± 0,134 (cruza de signo) |
| A' · percentil en barajado | 99,72 (100% fuera del 95%) | 51,6 (6,7% fuera del 95%) |
| B' · combinaciones saturadas | 44,4% | 1,9% |
| B' · frontera con t_óptima | rango ≈0,264 | rango ≈0,320 (más marcado) |

Tercera lectura distinta de la correlación central en tres rondas (−0,236 inválida → +0,375 → ≈0), cada vez que se corrigió algo del método. La frontera con t_óptima se sostiene, incluso más marcada que antes.

## Cierre de investigación

**¿Acoplamiento y huella son la misma medida disfrazada?** No es identidad algebraica forzada. Sobre 2.770 series (las tres baterías completas): r va de −0,39 a 0,998 (media 0,458±0,317). El factor que decide es **t_óptima**: con tOpt=22 son casi la misma variable (r≈0,998); con tOpt=28 se separan casi por completo (r≈0). Una variable derivada probada para independizarlas no funcionó (salió más redundante) — documentado para no repetir el intento.

**Material para definir κ_H** (sin proponer el número): distribución completa de H_absLocal/H_rel/H_noiseLocal en zonas "vivo" vs "colapsado", por batería. En la batería 1 esas zonas se distinguían con claridad; con el método corregido (2 y 3) la diferencia es mucho más chica. H_noiseLocal no discrimina en ningún caso. Tabla completa con percentiles en `bateria3/distribucion_H_vivo_vs_colapsado.csv`, para que el investigador principal elija el umbral.

**¿El recorte de rango explica el −0,756 original?** Se implementó el recorte real de un `<input type="range" min="0,6" max="1,4">` (que ningún motor/shim de las tres baterías reprodujo nunca) sobre el script real de v7.3 con el bug de arrastre intacto, eje 0,25→1,95:

| corrida | r |
|---|---|
| Referencia original | −0,756 |
| Esta prueba (arrastre + recorte, semilla 7) | **−0,3179** |
| Batería 1 (arrastre, sin recorte) | −0,236 ± 0,073 |
| Batería 2 (arrastre corregido, sin recorte) | +0,375 ± 0,039 |
| Batería 3 (todo corregido, eje real) | +0,008 ± 0,134 |

Confirmación **direccional, no exacta**: el recorte mueve la correlación hacia el mismo signo y más lejos de cero que el arrastre solo, consistente con haber sido parte de la causa — pero no reproduce el número exacto (esa corrida no dejó semilla ni configuración registrada; no se puede auditar más).

## Lo que queda genuinamente abierto

1. **La correlación central no tiene una lectura estable** — tres rondas, tres respuestas distintas. Con el instrumento ya validado a fondo, falta decidir si "≈0" es la respuesta final o si hace falta una cuarta verificación.
2. **El −0,756 solo parcialmente explicado** — mecanismo confirmado, número exacto no reproducido, y no se puede ir más lejos sin la semilla original (nunca registrada).
3. **La meseta >1,40 ahora es inalcanzable** — el eje válido del instrumento termina ahí. Reabrir el rango para investigarla implicaría deshacer la corrección reciente.
4. **κ_H todavía no tiene un número** — el material descriptivo está listo, la definición del umbral es decisión del investigador principal.
5. **Nunca se validó contra un navegador real** — solo contra Node simulando el DOM. El único bug de plataforma (el recorte) se encontró probando a mano en el navegador, no por la validación automatizada. Puede haber otros sin detectar.
6. Acoplamiento/huella: confound **resuelto** (depende de t_óptima), pero no hay todavía una variable derivada que los independice.

## Archivos

- `EIT3-Termico/bateria/` — batería 1 (invalidada).
- `EIT3-Termico/bateria2/` — batería 2 completa.
- `EIT3-Termico/bateria3/` — batería 3 (D/A'/B'/C, mismo formato que batería 2) + `topes_investigacion.md`, `cierre_investigacion.md`, `confound_huella_acoplamiento_por_serie.csv` (2.770 filas), `distribucion_H_vivo_vs_colapsado.csv`, `cierre_recorte/cierre_recorte.md`, `motor_v76.mjs`.
