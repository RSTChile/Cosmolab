# Validación — EIT-3 Térmico κ_H, batería de corridas

## 1. Verificación obligatoria: física con dibujo vs sin dibujo (bit a bit)

Método: `shim_html.mjs` ejecuta el `<script>` REAL del HTML v7.3 (ya modificado
con la separación `pasoFisica()`/`updateSimulation()`/`stepHeadless()` y el rng
seedeado) dentro de una sandbox `vm` de Node, con un DOM falso mínimo (elementos
`$(id)` con `.value`/`.getContext()`/etc., `Chart` stub no-op, `requestAnimationFrame`
no-op). Esto permite correr tanto `updateSimulation()` (física + `renderAll()`
completo, incluyendo `drawField`, `updateMetrics`, `updateTable`, `updateCharts`,
`drawQuadrant`) como `stepHeadless()` (solo `pasoFisica()`) sobre el MISMO código
fuente, no una reimplementación.

Prueba (`test_bit_identico.mjs`): misma semilla, 2000 pasos, comparando cada
campo numérico de `state` y un checksum completo de `field` (64×64).

| semilla | pasos | estado idéntico | campo idéntico |
|---|---|---|---|
| 42  | 2000 | ✅ | ✅ |
| 777 | 2000 | ✅ | ✅ |

Resultado: **idéntico bit a bit** en ambos casos (0 mismatches). Confirma que
`renderAll()` no toca el estado físico — la separación es segura. Como
subproducto, distintas semillas producen `field` distinto (fieldSum 102471.95
vs 102650.03 a igual número de pasos), lo que confirma que el offset de semilla
inyectado en `pseudoNoise` (ver `defectos_encontrados.md`, punto "pseudoNoise no
usa Math.random") sí varía el ruido del campo entre semillas.

## 2. Validación cruzada: motor.mjs (extracción Node limpia) vs shim_html.mjs (script real)

Método (`test_cruzado.mjs`): misma semilla y parámetros de Experimento A,
1200 pasos, en dos puntos distintos de luminosidad (0.6 y 1.2) más una segunda
semilla, comparando `state` completo y `field`.

| caso | semilla | luminosidad | pasos | resultado |
|---|---|---|---|---|
| 1 | 1  | 0.6 | 1200 | ✅ idéntico bit a bit |
| 2 | 1  | 1.2 | 1200 | ✅ idéntico bit a bit |
| 3 | 17 | 0.6 | 1200 | ✅ idéntico bit a bit |

Resultado: **motor.mjs reproduce exactamente** al script real del HTML. Se usa
como motor de la batería completa (Experimentos A/B).

## 3. Velocidad medida

25.200 pasos de física pura (`motor.mjs`, sin shim): **21.99 s** en esta
máquina → ~1146 pasos/s. El encargo estimaba 9,6 s (2624 pasos/s) en la máquina
donde se midió originalmente; aquí sale ~2,3× más lento que ese número, pero
sigue siendo **~230× más rápido** que los 5 pasos/s del simulador con dibujo
(25.200 pasos a 5 pasos/s = 84 min; aquí, 22 s). La batería completa (510
barridos × ~30.000 pasos/barrido ≈ 15,3M pasos) se paralelizó en 14 procesos
(la máquina tiene 16 núcleos) vía `child_process.fork`, ver `run_bateria.mjs`.
Experimento A (30 barridos): 168 s de reloj. Experimento B (480 barridos):
ver tiempo real en `resumen_descriptivo.md`.
