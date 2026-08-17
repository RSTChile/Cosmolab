# INFORME CS072-II — Puerta S (S0-S9) COMPLETA. Todas pasan. Listo el lector para la exploratoria NÚCLEO-II.

## CC, 17-jul-2026. Para CS. Ejecuta ADJUDICACION_CS072_II_puerta_s_CS.md ("Construir el módulo de filtración/jueces continuos AHORA").

## Qué construí
`cs072_ii_filtracion.py` — el lector de topología del sustrato sin-grafo (§7.1-7.3):
- **Jueces continuos sin umbral** (§7.1): log-dispersión de W, concentración nodal h_i=s_i/Σs (max_h como
  juez de hub), grado efectivo por participación k_eff=(Σw)²/Σw², rango efectivo del Laplaciano ponderado.
- **Filtración por bloques de empate** (§7.2): ordena TODOS los pares por peso, agrupa en bloques donde la
  diferencia relativa es < tolerancia (nunca desempata por índice/RNG); recorre TODA la filtración,
  registra frac_gigante en cada bloque (barato, union-find) y diam/d_s en checkpoints muestreados
  uniformemente + en el nivel exacto de "onset de persistencia" (primera vez que frac_gigante≥umbral —
  el mismo criterio para toda N, nunca el umbral que maximiza β, tal como exige §7.2).
- **Segundo sello** (§7.3): d_ij=−log(W_ij/maxW) (canónico) y d_ij=1/W_ij (alternativa), Dijkstra sobre el
  grafo COMPLETO ponderado (sin binarizar), δ-Gromov muestreado por landmarks sobre esa matriz de distancia.
- **Control positivo declarado** (Codex §8 brazo 5): grilla 2D genuina con coordenadas reales,
  W_ij=exp(−dist_euclídea/ξ) — instrumento de prueba, no participa de la afirmación de origen.

## S8 y S9 — RESULTADOS

| test | qué mide | resultado |
|---|---|---|
| **S8 (control positivo)** | el lector detecta métrica conocida (β + 2º sello) | β=0.570 (N∈{64,144,256,400}, diam onset=[10.5,16.5,25.5,28.5]) — no-degenerado, en el rango esperado de una métrica 2D genuina (referencia del arco: CS071 dio β=0.482 con jueces distintos). 2º sello: δ_log=0.138, δ_inv=1.260 — AMBOS finitos y >0 en las dos transformaciones (robusto) |
| **S9 (empate uniforme)** | W uniforme NO adquiere topología por filtración | n_bloques=**1** (todo el grafo entra en un único bloque de empate — no hay NINGÚN nivel intermedio) — log_dispersión=0, max_h=1/N exacto (sin concentración), 2º sello degenerado (δ_log=nan, δ_inv=0.0 — ambos indican "sin estructura métrica", el resultado honesto) |

**Contraste limpio**: la MISMA maquinaria da β=0.57 + δ finito en la retícula 2D conocida, y n_bloques=1 +
δ=0/nan en W uniforme. El lector distingue metricidad genuina de empate sin inventar nada.

## PUERTA S COMPLETA: S0-S9, TODAS PASAN

| S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 |
|---|---|---|---|---|---|---|---|---|---|
| ✓ | ✓ (exacto) | ✓ (exacto) | ✓ (exacto) | ✓ | ✓ | ✓ | ✓ (crítico, 1e-16 a 1e-19) | ✓ | ✓ |

Código: `cs072_ii_filtracion.py` (lector), `cs072_ii_puerta_s.py` (los 10 tests, actualizado con S8/S9).

## Declarado (para la exploratoria que sigue, per tu instrucción sobre la gravedad de ≥2 focos)
Tomo nota de tu adjudicación: con 1 solo foco la gravedad es idénticamente 0 (outer(cold,cold) se anula
fuera de la diagonal). Para la exploratoria NÚCLEO-II voy a incluir explícitamente en la tabla el caso
n_focos=1 como sub-control de "roce+expansión SIN gravedad", y usar n_focos≥2 para medir el balance
gravedad-vs-expansión real. Lo declaro ahora para no descubrirlo a mitad de la exploratoria.

## Siguiente paso (per tu instrucción)
Abrir la exploratoria NÚCLEO-II: barrer la expansión continua (p_t) junto con n_focos, fijar las anclas
P-COHESIÓN/P-BORDE/P-DISOLUCIÓN por persistencia-de-conectividad a través de la filtración (tengo
`F.persistencia_conectividad` ya construida para esto), ANTES de mirar TODO-II. No toco el fold de 5 brazos
hasta que las anclas queden congeladas. Empiezo con esto salvo que quieras auditar algo de la Puerta S
primero.

— CC 🐝
