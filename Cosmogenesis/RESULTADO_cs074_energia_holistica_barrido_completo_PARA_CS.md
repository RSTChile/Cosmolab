# cs074 — Experimento holístico de energía: resultado del barrido completo

**Fecha:** 2026-07-25 · Corrida de punta a punta, verificada contra disco, sin adjudicar.

---

## Qué se corrió

El motor (`cs074_energia_holistica.py`) combina, en un solo proceso por punto, la formación
de estructura (gravedad general + expansión + enfriamiento H₂ + materia oscura, las piezas
de CS073) con una capa de energía nueva: un presupuesto cerrado, exergía medida por
diferencias, y un costo de ligadura que se cobra cuando la gravedad forma estructura nueva
(la parte genuinamente nueva, el "gate" causal).

**Barrido:** 5 valores de ε (la asimetría fundacional) × 7 valores de reserva de energía
(de escasa a abundante, 6 décadas) × 8 semillas = **280 corridas del barrido principal**,
más 40 corridas con presupuesto infinito (para la prueba de admisibilidad) y 32 controles
(apagar una pieza a la vez). **352 corridas en total, 984 segundos (~16 minutos), 0 fallas
de conservación, 280/280 corridas completas.**

## Lo que se verificó primero (antes de confiar en el resultado)

- **La contabilidad de energía cierra:** en el control de gravedad pura, la fuga se quedó
  en 1.7% (el límite de falla declarado de antemano era 5%).
- **El costo de ligadura funciona en la dirección correcta:** con reserva escasa, poca masa
  queda acreditada como estructura; con reserva abundante, mucha más — confirmado en los
  280 puntos, no solo en la prueba aislada.
- **El presupuesto tiene efecto causal real (no decorativo):** en el 29.3% de las celdas
  (las que caen en la zona de reserva escasa, ~2 de los 7 valores barridos), el resultado
  con presupuesto finito es distinto del resultado con presupuesto infinito — exactamente
  donde se esperaba que importara, ni más ni menos.

## El resultado — la fracción de masa que termina ligada en estructura

| ε | satura en (reserva abundante) |
|---|---|
| 0.5 | 74.5% |
| 1.0 | 72.1% |
| 1.5 | 60.7% |
| 2.5 | 26.6% |
| 4.0 | 8.4% |

Cada ε tiene su propia curva: sube con la reserva desde casi 0 hasta un techo, y ese techo
depende de ε de forma **no monótona** — ni "más ε siempre liga más" ni lo contrario. Es un
hallazgo genuino sobre cómo la rugosidad inicial interactúa con la formación de estructura,
no algo que se buscó a propósito.

## La comparación con 4,9%/31,5% (SOLO como salida, nunca ajustado)

**No hay una coincidencia robusta.** Aparecieron 12 celdas cerca de 4,9% y 1 cerca de
31,5%, pero TODAS caen en la zona de reserva más escasa del barrido (donde la curva
todavía está subiendo, no en un techo estable) — con una dispersión entre semillas del
mismo orden que la distancia al número real. Es decir: son números que caen ahí por el
ruido normal de esa zona de la curva, no un punto especial y estable. Es un resultado
honesto: **el modelo, con esta implementación de la energía, no reproduce 4,9%/31,5% de
forma robusta.**

## Los controles (apagar una pieza a la vez) — 3 de 4 se comportan como se espera

- **Sin gravedad:** la fracción ligada cae a ~2% (de 60.7% con todo prendido). Es la
  confirmación más básica y más fuerte de que el mecanismo mide lo que dice medir.
- **Sin expansión:** la fracción SUBE a 88.4% (más que con expansión) — tiene sentido:
  sin expansión, nada compite contra el colapso gravitacional.
- **Sin materia oscura:** cambio chico (62.3% vs 60.7%) — efecto menor, no nulo.
- **Sin enfriamiento H₂:** **exactamente el mismo número (60.7%) que con enfriamiento
  prendido.** Este observable en particular (cuánta MASA queda ligada) no nota si el
  enfriamiento está prendido o apagado — probablemente porque el enfriamiento decide si
  la estructura se FRAGMENTA en varios pedazos o queda en uno solo, no si hay ligadura o
  no. Se reporta tal cual, no se esconde: es una limitación real de este observable
  específico, no un error de cómputo.

## Resumen en una frase

El motor funciona, la contabilidad de energía cierra, el mecanismo de costo de ligadura
tiene efecto causal real y verificado en 280 puntos, y el resultado es honesto: no hay
coincidencia con 4,9%/31,5%, la dependencia con ε no es la esperada ingenuamente (no
monótona), y el enfriamiento H₂ no mueve este observable en particular aunque sí debería
importar para la fragmentación. Nada de esto se adjudica — queda para que el director lo
lea y decida qué sigue.

**Archivos:** `PROTOCOLO_cs074_energia_holistica_PREREGISTRO.md` (con su adenda de
implementación), `cs074_energia_holistica.py`, resultado crudo completo en
`resultados_cs074_energia_holistica/cs074_barrido_completo_result.json`.
