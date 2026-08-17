# PROTOCOLO cs074-B — ¿Dónde actúa el enfriamiento? Medir la fragmentación, no la ligadura

**Congelado (pre-registro):** 2026-07-25 · **Ejecutor:** CC · **Director:** Alexis López Tapia
**Diseño base:** `DISENO_tres_experimentos_holistico_PARA_CC.md` (Experimento B, leído entero).
**Motor reusado (leído, no editado en su física):** `cs074_energia_holistica.py`.

Este documento se congela ANTES de escribir el script del experimento.

---

## 1. Pregunta

En el barrido original de cs074, apagar el enfriamiento H₂ (`cooling_on=False`) no cambió
`frac_masa_ligada` en absoluto (60.7% con y sin). Se reportó como limitación honesta: el
enfriamiento probablemente decide en cuántos PEDAZOS se parte la estructura, no si hay
estructura o no. Este experimento mide el observable que sí puede verlo.

## 2. Cambios aditivos al motor (declarados, no ocultos)

- `tasa_enfriamiento` (nuevo parámetro de `correr_holistico_energia`, default 0.3 = el
  mismo valor hardcodeado que usaba cs074 original — no cambia ningún resultado previo que
  no lo especifique explícitamente). Pasa directo a `EnfriamientoH2(tasa_enfriamiento=...)`.
  Es el dial CONTINUO de intensidad del canal H₂.
- `seed_dens_null` (nuevo parámetro, default `None` = comportamiento idéntico al original).
  Si se da una semilla, baraja la densidad #23 entre bariones ANTES de construir masa
  efectiva — mismo mecanismo NULL que ya usa `cs073_cierre_holistico.py`.
- Los campos de fragmentación (`n_clusters_finales`, `frac_masa_en_mayor_cluster`,
  `masas_clusters_finales`) ya se agregaron para el Experimento A — se reusan tal cual.

## 3. Observable nuevo: fragmentación

`frac_masa_en_mayor_cluster` (¿un grumo gigante o muchos chicos?) y `n_clusters_finales`
(cuántos grumos). Es una cantidad DISTINTA de `frac_masa_ligada` (mide la FORMA de la
estructura, no su cantidad) — no se deriva una de la otra.

## 4. Barrido (sobredimensionado)

| Eje | Rango | Puntos |
|---|---|---|
| `tasa_enfriamiento` | {0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0} (de nada a 10x el default) | 11 |
| ε (`amp_rugosidad`) | {0.5, 1.0, 1.5, 2.5, 4.0} (mismo grid que cs074 original) | 5 |
| `E_reserva` | {1e-2, 1.0, 1e2} (escasa/media/abundante — reducido de las 7 de cs074, foco en enfriamiento no en reserva) | 3 |
| semillas | 0..11 | 12 |

Total: 11×5×3×12 = **1980 corridas REAL**, más el control barajado (§5): mismo grid,
`seed_dens_null` derivado de la semilla → **1980 corridas NULL**. **3960 corridas totales.**

## 5. Controles

- **Admisibilidad (¿el enfriamiento actúa en ESTE observable?):** ya incluido en el barrido
  — `tasa_enfriamiento=0.0` es el punto de apagado completo, dentro de la misma grilla.
  Se compara contra el resto de la curva. Si `frac_masa_en_mayor_cluster` y
  `n_clusters_finales` NO cambian con `tasa_enfriamiento` → el enfriamiento tampoco actúa
  aquí (sería Shannon, se reporta tal cual, no se fuerza un hallazgo).
- **Barajado (significancia):** para cada combinación (tasa_enfriamiento, ε, E_reserva,
  semilla), una corrida gemela con `seed_dens_null = 90_000 + semilla` (misma física, misma
  semilla de layout, densidad #23 barajada). Se compara REAL vs NULL con z-score sobre las
  12 semillas, para `n_clusters_finales` (el observable primario de este experimento).

## 6. PASS pre-registrado

Más enfriamiento (`tasa_enfriamiento`↑) → más fragmentación: `n_clusters_finales` sube
Y/O `frac_masa_en_mayor_cluster` baja, de forma monótona o cuasi-monótona, CON separación
del NULL barajado (z>2 en al menos la mitad de las celdas (ε, E_reserva) al comparar el
extremo `tasa_enfriamiento=3.0` contra `tasa_enfriamiento=0.0`). Si sale así: el
enfriamiento SÍ actúa, solo que en un observable distinto al que cs074 original miraba. Si
NO sale (plano, o sin separación del NULL): el enfriamiento no fragmenta en este régimen —
también un hallazgo real, se reporta igual.

## 7. Trampas

- **T1:** ningún número a mano — tasa_enfriamiento, ε, E_reserva y semillas barridos.
- **T-conservación:** heredado de cs074 (5% en control de gravedad pura).
- **La cantidad medida ≠ su juez:** fragmentación (nueva) vs `frac_masa_ligada` (vieja) son
  observables independientes, ninguno se deriva del otro.
- **Perturbación dinámica + semillas:** 12 semillas por celda.
- **Control que muerde:** el NULL barajado (§5) — si el NULL también fragmenta igual que el
  REAL, el hallazgo se descarta como artefacto, no se reporta como positivo.

## 8. Qué se entrega a CS, sin adjudicar

Curva completa de fragmentación vs `tasa_enfriamiento` (por ε, por E_reserva), z-scores
contra el NULL, y el veredicto de si el PASS del §6 se cumple. No se cierra aquí.
