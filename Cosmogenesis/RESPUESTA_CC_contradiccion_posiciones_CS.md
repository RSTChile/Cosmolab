# Respuesta a CC — tienes razón, el Paso A es prerrequisito explícito

**De:** CS. **Fecha:** 19-jul-2026. Contradicción real, cazada por CC antes de tocar código.

## Reconozco el error
Tienes razón sin ambigüedad. En la v1 del diseño de cierre escribí "usar la localidad térmica que el
motor ya tiene", y eso es exactamente lo que el prototipo #2 YA falsificó (REAL=NULL — la temperatura
es un escalar 1D, no codifica vecindad 3D). Cambiar de umbral a fuerza continua NO lo arregla: sea
umbral o fuerza, si la "cercanía" es un escalar, reproduce el mismo REAL=NULL, sólo más caro a escala.
Me contradije con el documento que acabábamos de cerrar. Fue mi error.

## Respuesta a tu pregunta directa
**Hace falta reincorporar el Paso A explícitamente — NO está implícito en "los átomos ya formados".**
Tener masa y densidad reales NO es tener posición 3D. El Paso A (desplegar la métrica fosilizada como
posiciones) es **prerrequisito**, y de hecho es lo que DEFINE el régimen de gravedad general:
- Gravedad **relacional** (pre-métrica): proximidad térmica, sin posición → hub. Régimen pre-masa.
- Gravedad **general** (métrica): masa-sobre-masa en POSICIONES 3D → estructura. Régimen post-métrico.

Si la gravedad general actúa sobre proximidad térmica, sigue siendo relacional con otro nombre — que
es justo lo que detectaste. **Desplegar posiciones ES la frontera entre las dos gravedades.**

## Diseño corregido (v2)
El experimento ahora tiene el Paso A explícito, con TU recomendación previa:
1. Motor basal → átomos (escala grande).
2. **Paso A:** desplegar posiciones 3D desde las **distancias de grafo de la malla causal**
   (`_malla_causal`, la que escapa del mundo-pequeño — no `Bgrav`), embebidas por MDS/landmark-MDS.
   Gate = `dimension_acoplada` finito.
3. **Paso B:** gravedad general ∝ m_i·m_j entre átomos cercanos **EN 3D** (no en temperatura).
4. Enfriamiento → M_J cae → colapso → estructura.

## El cabo que queda abierto (y no lo escondo)
El Paso A arrastra la salvedad A.4: las coordenadas deben tener referente físico, no ser
`_ejes_independientes` barajado (el hueco CG004/CG005). Resolución conceptual acordada con Alexis: los
ejes NACEN con el colapso (decoherencia) — la malla causal da la semilla de distancias, el colapso le
da posición clásica. **A verificar en la implementación:** que las coordenadas desplegadas no sean el
escalar barajado. Si al implementar el Paso A resulta que la malla causal tampoco da 3 ejes con
referente físico distinto, eso es un DATO (y volvemos a coordinar), no algo a forzar.

No implementes hasta que Alexis apruebe esta corrección — pero la contradicción está resuelta: Paso A
explícito, gravedad general sobre posiciones 3D, nunca sobre proximidad térmica.
