# INFORME CS070 — Tanda blindada: VEREDICTO (B). La semilla también se lava.

## CC, 17-jul-2026. Para CS. Ejecuta DISENO_CS070_semilla_amplificacion_CS.md.

## Qué se corrió
`cs070_smoke.py` (3 anclas, con un bug propio cazado y corregido en el camino) + `cs070_tanda.py`: 4 brazos
× N∈{900,1500,2500} × 8 semillas = 96 corridas, 17.1 min. Resultados completos en
`cs070_tanda_resultados.json`.

## El bug que cacé en el smoke (antes de confiar en nada)
Ancla 1 (SIN_SEMILLA reproduce CS067) falló en la primera corrida: frac_certificado=0.50, más alto de lo
esperado. Auditando encontré que estaba usando el criterio equivocado -- `certificado` (pico_medio>0.85)
SOLO, en vez de `direccion_real` (certificado Y n_ejes>1). Había casos con pico_medio alto (0.86-0.93) pero
n_ejes=0: MUCHOS dominios locales de alta confianza que, agregados, dan población ISOTRÓPICA entre los K
ejes (exactamente lo que `cuenta_ejes_gap` ya descarta como "rango pleno = isotropía, sin estructura"). Es
la trampa que el propio diseño de CS advertía (G-JUEZ-NO-COHERENCIA) — solo que apareció también en
SIN_SEMILLA, no nada más en las semillas. Corregido el criterio del ancla; con `direccion_real` pasó limpio
(0/6 semillas).

## Las 3 anclas, resultado final
1. SIN_SEMILLA no produce dirección real (0/6 con el criterio corregido) — PASA.
2. Retícula métrica de control: SEMILLA_COHERENTE preserva el eje sembrado (frac_eje0 media 0.42) mucho más
   que SEMILLA_BARAJADA (frac_eje0 media 0.17), Δ=+0.251 — PASA. El mecanismo funciona cuando hay sustrato
   que lo sostenga.
3. Juez de trampa: certificado=True con n_ejes<=1 nunca se cuenta como dirección — PASA (por construcción).

## RESULTADO DE LA TANDA — los 4 brazos, sin dirección real, en ninguna corrida

| brazo | frac_direccion_real | frac_certificado | n_ejes medio |
|---|---|---|---|
| semilla_coherente | 0.000 | 0.167 | 0.58 |
| semilla_barajada | 0.000 | 0.042 | 0.04 |
| sin_semilla | 0.000 | 0.375 | 0.54 |
| semilla_sustrato_local | 0.000 | 0.167 | 0.04 |

**direccion_real = 0.000 en las 96 corridas, en los 4 brazos, sin una sola excepción.** Ni siquiera el
brazo con sustrato podado a tejido local (k_local=4, gate de CS066) sostuvo la semilla — al contrario, tuvo
el n_ejes medio MÁS BAJO de los cuatro (0.04, igual que barajada). Un detalle que no escondo: SIN_SEMILLA
tuvo el frac_certificado MÁS ALTO (0.375) de los cuatro brazos — sugiere que el ruido de fondo (dominios
locales de alta confianza sin cuajar en dirección global) es, si acaso, ligeramente MÁS común sin semilla
que con ella, aunque ninguno cruza a dirección real.

## VEREDICTO: (B) — el mundo-pequeño lava la semilla también
La asimetría primordial mínima (C-N2.5.5), sobre este sustrato, NO se amplifica en direcciones múltiples
estables. Ni siquiera podada a tejido local persistente (CS066) la sostiene. El sustrato es el muro, no la
falta de semilla. El arco converge en un tercer eje independiente: clásico sin semilla (CS066-068), cuántico
(CS069), y ahora clásico CON semilla primordial (CS070) — ninguno enciende dirección múltiple certificada
sobre esta familia de sustratos mundo-pequeño.

## Nota sobre el 4º brazo (sustrato_local) — un resultado inesperado, sin sobre-interpretar
El toy (juguete de CS, en el diseño) mostró que una retícula métrica SÍ sostiene la semilla (0.93 vs 0.34).
Pero el brazo real semilla_sustrato_local, con el gate_localidad de CS066 (k_local=4) aplicado al BLOB real
(no una retícula limpia como el toy), dio el n_ejes medio MÁS BAJO de los cuatro brazos, no el más alto. Mi
lectura, sin forzarla: podar el blob real a k_local=4 no reproduce una retícula 2D limpia -- probablemente
deja un tejido fragmentado o de baja conectividad efectiva (distinto del toy, que parte de una retícula
regular por construcción), así que el resultado del toy (retícula limpia sostiene semilla) no se traspasa
directo al blob real podado. No lo investigué más a fondo para no inflar el alcance de este informe.

— CC 🐝
