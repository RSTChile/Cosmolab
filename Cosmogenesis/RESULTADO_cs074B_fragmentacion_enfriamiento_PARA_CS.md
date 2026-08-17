# cs074-B — ¿Dónde actúa el enfriamiento?

**Fecha:** 2026-07-26 · 3960 corridas (1980 real + 1980 control barajado), 27298 s (~7,6 h),
1980/1980 válidas, verificado contra disco.

---

## Resultado — negativo, limpio y robusto: PASS = False

La sospecha era que el enfriamiento H₂ no cambiaba `frac_masa_ligada` (cs074 original)
porque decidía en cuántos PEDAZOS se parte la estructura, no si hay estructura. Se midió
exactamente eso — y tampoco.

| intensidad enfriamiento | grumos (real) | grumos (control barajado) | z |
|---|---|---|---|
| 0,00 (apagado) | 3,07 | 3,42 | −0,14 |
| 0,30 (default original) | 3,18 | 3,47 | −0,11 |
| 1,00 | 3,15 | 3,45 | −0,12 |
| 3,00 (10× el default) | 3,15 | 3,45 | −0,12 |

**La curva es plana de punta a punta.** Desde apagado hasta 10 veces la intensidad
original, el número de grumos no se mueve (3,07 a 3,22, sin tendencia), y en NINGUNO de
los 11 niveles probados el real se separa del control barajado (todos los z entre −0,11 y
−0,14 — ni siquiera cruzan cero en la dirección esperada). El reparto de masa entre grumos
(`frac_masa_en_mayor_cluster`) tampoco se mueve (0,062–0,065 siempre).

## Lectura

El enfriamiento H₂, en esta implementación, **no mueve ningún observable de estructura
que se le haya medido hasta ahora** — ni cuánta masa queda ligada (cs074 original) ni en
cuántos pedazos se parte (cs074-B). No es que el observable viejo fuera ciego y el nuevo
viera algo: los dos coinciden en que el canal no actúa, con 1980 corridas de respaldo y
control barajado que no muerde en ningún punto de un barrido de 10×.

Posibles razones (no verificadas aquí, quedan para quien quiera perseguirlas): la escala
de tiempo de la corrida (60 pasos) puede ser corta para que el enfriamiento tenga efecto
dinámico visible; o la agitación térmica de soporte (que sí está siempre activa, ver
`incluir_presion_termica` en cs074) puede estar dominando sobre la señal del canal de
enfriamiento específico.

## En una frase

Segundo negativo limpio sobre la misma pieza: el enfriamiento H₂ no fragmenta ni liga de
forma medible en este modelo, confirmado con control barajado en todo un barrido 10× de
intensidad. Hallazgo real, no artefacto de instrumento — se reporta tal cual.

**Archivos:** `PROTOCOLO_cs074B_fragmentacion_enfriamiento_PREREGISTRO.md`,
`cs074B_fragmentacion_enfriamiento.py`,
`resultados_cs074B_fragmentacion_enfriamiento/cs074B_result_FULL.json`.
