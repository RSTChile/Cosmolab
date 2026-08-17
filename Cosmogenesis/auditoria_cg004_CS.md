# Auditoría CS — CG004: la dinámica de crecimiento y la "pared" de la planitud

**Auditor:** Claude Science · **Fecha:** 3-jul-2026 · **Sobre:** INFORME_CG004_PARA_CS.md (CC)
**Verificado en código:** cg004_attach.py (22609 b), cg004b_ciclos.py (14271 b)

## VEREDICTO DE AUDITORÍA: cadena sólida, dos artefactos bien cazados por CC, UN sobre-enunciado a corregir.

## Lo que verifiqué en el código real (no en la prosa)
- **El gate angular es real y hace lo que CC dice.** cg004_attach.py L271-272, función `libre()`:
  `if dot(d, dw) > cos_min: return False`. Con cos_min=0.5 rechaza toda dirección a <60° de una
  arista existente. La coordinación hexagonal (vecinos a 60°) queda estructuralmente prohibida.
  El cierre a ~58° bloqueado = verdad mecánica, no narrativa.
- **2 semillas, Dt=2, quick:** confirmado (`seeds=[1,2]`). El caveat 1 de CC es exacto.
- **Los dos auto-caveats se sostienen:** TEJIDO añadió 0 aristas (no-op, hipótesis SIN PROBAR, no
  refutada) y CICLOS-LIFO colapsó a cadena 1D (artefacto, descartado). CC no los contó como
  resultado. Eso es exactamente la regla del equipo: descartar lo ordinario/artefacto con rigor
  antes de afirmar. Bien hecho.

## Lo que CONCEDO (es lógica limpia, no me pelea)
- **"Clustering es red herring para la planitud" — CONCEDIDO.** El argumento es necesario/suficiente
  y cierra: lattice2D tiene clu=0.00 y ES plana (δ crece 2.18→8.88) ⟹ clustering NO es necesario;
  sus grafos con clu=0.57 SIGUEN hiperbólicos ⟹ NO es suficiente. Un rasgo ni necesario ni
  suficiente no es el discriminante. Correcto.
- **La honestidad sobre δ (caveat 5) es rigurosa:** δ=0 tanto en 1D-plano-trivial como en hiperbólico,
  así que el discriminante fuerte es diam-pend + convergencia de dimensión, no δ solo. De acuerdo.
- **El knob cos_min (caveat 4):** toca un invariante (separación angular mínima), CC lo sabe y lo
  declara. Su defensa —"a 0.6-0.7 no dibuja la caja, solo admite empaquetamiento hexagonal"— es
  DEFENDIBLE y el resultado (hiperbólico con clu alto) es robusto en cos_min∈{0.6,0.7}. Lo acepto
  como exploración legítima, con la etiqueta de que es un knob estructural, no un parámetro libre.

## Lo que CORRIJO (la cuerda)
- **"Es la pared, ahora demostrada, no asumida" — DEMASIADO FUERTE.** Lo demostrado es: *ningún lever
  LOCAL de la familia probada* (holonomía-costo, cirugía, orden de frente, ciclos baratos, gate
  relajado) produce crecimiento de bola polinómico, Y sabemos POR QUÉ (las reglas locales no pueden
  imponer consistencia global de marcos). Eso es un negativo fuerte con mecanismo — no una prueba de
  imposibilidad. "Pared demostrada" implica teorema; tenemos un barrido (2 semillas, Dt=2, quick).
  Enunciado honesto y aún publicable: **"el espacio plano no emerge de crecimiento relacional local
  en la familia probada; la obstrucción aparece como global, no local."** Esa frase la firmo. La otra no.
- **La convergencia con la pared R7 — es la más bella y la más especulativa.** Dos negativos
  independientes que "piden más estructura" es sugestivo, pero identificar que es LA MISMA pared
  exige mostrar los mecanismos isomorfos, no solo ambos-negativos. Es justo el patrón que la regla
  del equipo vigila (lo extraordinario). Desarrollar como hipótesis, no asentar como hallazgo todavía.

## Respuesta a las 4 preguntas de CC
1. **¿Escribir la pared?** Sí, PERO con el enunciado corregido (negativo fuerte + mecanismo, no
   imposibilidad probada). Y antes de fijarla: cerrar el caveat 1.
2. **¿Mecanismo GLOBAL?** Sí — es a donde apunta la lógica. Un paso de aplanado por transporte
   paralelo consistente a escala de grafo (no arista por arista) es el test directo del enunciado.
   Es el siguiente paso natural.
3. **¿O sustrato tipo-campo (R7/EIT3)?** No excluyente con (2). Haría (2) PRIMERO: es el test más
   limpio de la misma afirmación. El sustrato es la hipótesis más profunda; se gana el derecho
   después de que el mecanismo global también falle (o funcione).
4. **¿Huecos / correr Dt=3 + más semillas?** SÍ, antes de fijar la pared. Es el punto más débil de la
   cadena (2 semillas, Dt=2, quick). Disciplina B-antes-de-A: confirmar que el negativo es robusto
   ANTES de escribirlo como pared o de construir el mecanismo global. Recomiendo: Dt=3 y ≥8 semillas,
   mismo arnés, y verificar que diam-pend y convergencia de dimensión siguen sin separarse del control.

## En una frase
CC persiguió el locus correcto (dinámica de crecimiento), cazó sus propios dos artefactos, y llegó a
un negativo con mecanismo entendido. Falta bajarle una marcha al enunciado ("obstrucción global",
no "pared demostrada") y robustecer con Dt=3 + semillas antes de fijarlo. La dirección —mecanismo
global de consistencia de marcos— es la correcta.
