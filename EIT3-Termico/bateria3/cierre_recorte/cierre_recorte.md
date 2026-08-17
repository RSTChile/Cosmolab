# Cierre — ¿el recorte de rango explica el −0,756 original?

Prueba de mecanismo, no una repetición exacta de la corrida original (esa
corrida usó `Math.random()` sin semilla, en una versión anterior a que
existiera control de semilla — no se puede reproducir bit a bit).

## Método

Se tomó el `<script>` REAL de `EIT3_Termico_kappaH_v7.3.html` (versión que
todavía tiene el bug de arrastre entre paradas, tal como debió estar la
versión que produjo la corrida original) corriendo dentro de Node, con el
elemento `luminosity` implementando el recorte real de un
`<input type="range" min="0.6" max="1.4">` — exactamente el comportamiento
que un navegador de verdad aplica y que ningún shim ni motor de las tres
baterías reprodujo nunca.

Barrido de 60 puntos, luminosidad 0,25→1,95 (el rango de Experimento A de la
batería 1), settle=300, measure=120, semilla=7, mismos parámetros fijos que
Experimento A (tc_ptc=18, exponente_ptc=4,1, resto default).

## Resultado

**La curva queda visiblemente deformada por el recorte**, tal como predice el
mecanismo: footprint se aplana en ~0,55–0,68 para todo x<0,6 (todos esos
puntos corren en realidad la misma luminosidad clampeada a 0,6, con el estado
arrastrándose de una parada a la siguiente) y se aplana de nuevo en ~6,8–6,9
para x>1,4 (misma lógica, clampeado a 1,4). La dinámica real —incluido el
colapso en V cerca de x≈0,88— solo aparece en el tramo 0,6–1,4, que es
angosto dentro del eje pedido de 0,25 a 1,95.

**Correlación huella↔entropía resultante: −0,3179.**

| corrida | r |
|---|---|
| Referencia original ("única corrida limpia") | −0,756 |
| Esta prueba (arrastre + recorte simulado, semilla=7) | **−0,3179** |
| Batería 1 (arrastre, SIN recorte, promedio de 30 semillas) | −0,236 ± 0,073 |
| Batería 2 (arrastre corregido, sin recorte) | +0,375 ± 0,039 |
| Batería 3 (todo corregido + eje real 0,60-1,40) | +0,008 ± 0,134 |

## Lectura (sin concluir)

El recorte, sumado al bug de arrastre, mueve la correlación en la **misma
dirección** que el −0,756 original (negativa, más fuerte que el arrastre
solo) — a diferencia de la batería 2, donde arreglar SOLO el arrastre (sin
tocar el recorte, porque en el motor Node nunca existió) la manda para el
lado positivo. Esto es consistente con la hipótesis de que el recorte fue
parte de lo que producía ese número, pero **esta única corrida (una semilla,
sin poder replicar el `Math.random()` sin semilla del original) no alcanza el
−0,756 exacto** — es una confirmación direccional del mecanismo, no una
reproducción completa. No se puede llegar más lejos sin saber la semilla y
configuración exactas de la corrida original, que no quedaron registradas en
su momento (esa es, en sí, la falla de origen: sin bitácora de semilla no hay
manera de auditar una corrida después de hecha — exactamente lo que `v7.4`
en adelante vino a resolver).
