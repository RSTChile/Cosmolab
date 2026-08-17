# CORRECCIÓN CS072 v4 → v5 — DOS REPAROS (Codex, verificados por CS con código) ANTES DE MEDIR ESPACIO
## Director aprobó: limpiar estos dos antes del barrido de escalas. Ninguno es fatal; ambos son reales.

## CONTEXTO: el motor v4 (proceso único) está bien en lo esencial — un solo bucle, fuerzas actuando de verdad
## (CS verificó apagando gravedad/fuerte/EM: cambian el resultado; sin EM → 0 hidrógeno), y pasa la TESIS #1
## (simetría exacta materia=antimateria → universo VACÍO 0/0/0; asimetría mínima → nace materia). Eso queda.
## Pero Codex cazó dos cosas en el CONTEO de materia que CS confirmó con código. Arreglar ambas.

## ─── REPARO 1: la supervivencia se fija en 2 pasos (emparejador de un golpe), no emerge del proceso ───
HALLAZGO (dos corridas distintas, ambas trazables): (a) CS verificó a ESCALA CHICA (300q/270aq/100e/90p):
corriendo 1,2,3,5,10,20 pasos, los vivos caen 220→40 entre el paso 1 y 2, y de ahí quedan CLAVADOS en 40
hasta el paso 20 — la aniquilación resuelve casi todo de un golpe. (b) Codex, en su reproducción a la ESCALA
de CC (1500q/1350aq/500e/450p = 3.800 partículas), observó que los 150 quarks supervivientes / 50 bariones
quedan fijados por el emparejador determinista global. Son dos escalas diferentes, coinciden en lo mismo: la
cuenta queda GARANTIZADA por el emparejamiento, casi aritmética, NO emerge del forcejeo de las fuerzas a lo
largo del tiempo.
QUÉ HACER: la aniquilación debe ser un PROCESO en el tiempo, no un emparejamiento global instantáneo en el
paso 1. Un par materia-antimateria se aniquila cuando su ligadura (por EM/fuerte, que actúan en el bucle)
SUPERA un umbral físico — así, quién se aniquila y cuándo DEPENDE de la dinámica de las fuerzas, no de un
emparejador que decide todo antes de que las fuerzas actúen. El resultado final puede seguir siendo ~150
supervivientes (la asimetría lo fija), pero el CAMINO hasta ahí debe ser dinámico y distribuido en los pasos,
no un salto en el paso 1. VERIFICACIÓN que CS hará: la curva de vivos vs paso debe DECRECER gradual, no caer
de golpe en el paso 2.

## ─── REPARO 2: el umbral de "ligado" incluye partículas MUERTAS (infla el conteo de bariones) ───
HALLAZGO (CS verificó en cuenta_bariones_e_hidrogeno, línea ~368): w0_ef = W.sum(axis=1).mean() / (N-1) usa
N = TODAS las partículas, incluidas las ~3.600 MUERTAS (vivo=False). Las muertas tienen W≈0 y arrastran la
media hacia abajo → el umbral queda más BAJO → más fácil que un par cuente como "ligado" → puede INFLAR el
conteo de bariones. El conteo de tríos sí filtra por vivo, pero el UMBRAL que lo decide está contaminado.
QUÉ HACER: calcular w0_ef (y por tanto el umbral) SÓLO sobre las partículas VIVAS: usar
W[vivo][:,vivo].sum(axis=1).mean() / max(n_vivos - 1, 1). Las muertas no deben entrar en la media que fija el
umbral. VERIFICACIÓN que CS hará: recalcular con y sin muertas y confirmar que el conteo de bariones no
depende de incluirlas.

## CERO AZAR sigue vigente. NO tocar nada más del bucle (las fuerzas actúan bien). Sólo estos dos arreglos.

## DESPUÉS de los dos arreglos, y sólo después: correr el BARRIDO DE ESCALAS — la pregunta que de verdad
## falta y que nadie ha medido aún: ¿el DIÁMETRO de la red final CRECE con N (métrica, hay espacio) o se
## queda PLANO (grumo/hub, no hay espacio)? Medir diámetro (BFS real) para N creciente: p.ej. poblaciones
## que den ~50, ~200, ~800, ~3000 bariones, y ver la pendiente diámetro-vs-N. Real vs NULL determinista.
## ESA es la incógnita del espacio. Todo lo demás (materia, aniquilación, hidrógeno) ya sabíamos calcularlo.

## Dudas → PREGUNTA a CS antes de codificar. No inventes piezas nuevas.
— CS 🐝
