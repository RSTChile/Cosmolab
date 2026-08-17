# v7.7 — física real de Daisyworld (Stefan-Boltzmann) y recalibración

Cambio de fondo: la temperatura planetaria dejó de ser una recta (`12+34*absorbido`)
y pasó a ser la ley real de Watson & Lovelock 1983 (*Tellus* 35B): `σ(Te+273)⁴ = S·L·(1−A)`.
Archivo nuevo: `ET3-Termico_v7.7.html`. `v7.6.1` queda intacto para comparar.

## 1. Derivación de constantes (verificadas con Node, no de memoria)

```
S = 9,17×10⁵ erg·cm⁻²·s⁻¹        (constante solar del paper)
σ = 5,6704×10⁻⁵ erg·cm⁻²·s⁻¹·K⁻⁴  (Stefan-Boltzmann, física universal)
K = S/σ = 1,61732×10¹⁰

Te(absorbido) = (K·absorbido)^0,25 − 273        [°C]
```

Verificación puntual: `Te(0,5) = 26,87°C` — sensato para L=1, A=0,5 (planeta
estéril de referencia).

**Offset de temperatura local** (reemplaza el "14" fijo, sin origen
documentado, de v7.6.1 y anteriores): el paper linealiza la temperatura local
como `T_local = q'·(A_global−A_propia) + Te`, con `q' = q/[4·(273+22,5)³]` y
acota `q < 0,2·S·L/σ`. Usando la cota superior en L=1: **q' = 31,3367**.

## 2. Rango de temperatura resultante — por qué esto no era un cambio de una línea

| | recta vieja (v7.6.1) | física real (v7.7) |
|---|---|---|
| más frío (L=0,6, A=0,75) | ~17°C | **−51,1°C** |
| más caliente (L=1,4, A=0,25) | ~48°C | **+88,0°C** |

Casi 5× más ancho. Todo lo demás del instrumento estaba calibrado para la
banda angosta.

## 3. Qué más se tocó, y por qué

- **`ptcResponse()` — ratio en Kelvin, no Celsius.** Con Celsius, `ratio=temp/ptcTc`
  se degenera apenas `temp` se acerca a 0 o se vuelve negativo (todo el medio
  eje frío quedaba pegado en el piso 0,2 del clamp, sin discriminar nada; y si
  se intentara mover `ptcTc` a un valor negativo, el signo del ratio se
  invierte). En Kelvin, `(temp+273)/(ptcTc+273)`, ambos términos son siempre
  positivos y el sensor discrimina en todo el rango nuevo. Misma forma
  (ratio^ptcSharp), solo la referencia de escala cambia.
- **Rango del slider `ptcTc`**: de [18,35] a [**−60,100**], para poder cubrir
  el nuevo espectro. Default se mantiene en 25°C (sigue siendo representativo
  del punto de operación típico de la zona fértil, ver tabla del punto 4).
- **`computeCoupling` (constante 8,0)**: se revisó, NO se cambió. Medido
  empíricamente: A_sys_env sigue discriminando (0 a ~0,73 en el barrido de
  prueba, ver tabla), no queda pegado en un extremo. No hacía falta tocarlo.
- **NO se tocó**: curva de crecimiento (0,003265, tOpt, 5-40°C), albedos
  (0,25/0,5/0,75), muerte (0,28+ruido), arquitectura de semilla por
  parada/fase, `medirRecuperacion`/`asentarHastaEquilibrio` (operan sobre
  `black`/`white`, no sobre la escala de Tf, así que no dependen de este
  cambio).

## 4. Validación

**Bit a bit (render vs no-render), 2 semillas × 1500 pasos: IDÉNTICO, 0
diferencias en ambos casos** (usando el script real de v7.7 corrido en
sandbox, no una reimplementación).

**Barrido cualitativo** (24 puntos, luminosidad 0,6→1,4, reinicio por parada,
semilla=1, parámetros por defecto, settle=400/measure=150):

| lum | Tf (vivo) | abiótico | huella | black | white | A_sys_env |
|---|---|---|---|---|---|---|
| 0,60–0,67 | — | −9,1 a 3,6 | — | 0 | 0 | 0 (extinto) |
| 0,70 | 29,0 | 1,7 | 27,3 | 0,640 | 0,000 | 0,05 |
| 0,81 | 34,9 | 11,4 | 23,6 | 0,549 | 0,014 | 0,22 |
| 1,05 | 30,7 | 30,7 | 0,07 | 0,185 | 0,366 | 0,10 |
| 1,30 | 28,9 | 46,9 | 18,0 | 0,002 | 0,569 | 0,05 |
| 1,33–1,40 | 51,8–55,3 | 49,1–53,2 | ~2–3 | 0 | 0 | 0,64–0,73 (extinto) |

Sin `NaN`/`Infinity`. 6 de 24 puntos extintos (los dos bordes del eje,
esperable). **El fenómeno central de Daisyworld se ve MÁS marcado que antes,
no roto**: mientras la curva abiótica sube de forma casi lineal de −9°C a
+47°C a lo largo de la zona fértil, la temperatura real con vegetación se
mantiene entre 29°C y 35°C — una meseta de ~6°C contra una excursión abiótica
de ~56°C. Con la recta vieja esa meseta también existía pero el contraste era
mucho menor (la curva abiótica solo recorría ~30°C en total). El traspaso
suave de margaritas negras→blancas a medida que sube la luminosidad también
se preserva intacto (mismo patrón que siempre).

## 5. Decisiones que quedan documentadas, no aplicadas en silencio

- q' se calculó con la cota SUPERIOR de q que da el paper (evaluada en L=1),
  no con un valor intermedio — es la lectura más defendible dado que el paper
  no da un valor único de q, solo una cota.
- `computeCoupling` se dejó igual tras confirmar que no hacía falta tocarlo
  — evité un cambio innecesario.
- El offset local usa L=1 como referencia fija para q' (el paper sugiere que
  el propio q podría depender de L, pero no hay evidencia de que lo variaran
  dinámicamente en sus corridas — se trató como constante, igual que el "14"
  que reemplaza).

No se corrió ninguna batería completa (D/A'/B'/C) — eso queda para cuando se
autorice, dado que compararla contra `bateria2`/`bateria3` requiere decidir
primero si tiene sentido comparar directamente (la escala de `huella` cambió
de orden de magnitud: antes tope ~9, ahora tope ~27).
