# INSTRUCCIÓN CS072 PARA CC — CERRAR RESERVA B: el umbral es sobre la CANTIDAD de diferencias (tesis #2)
## CS ya verificó la forma del resultado con barridos finos (abajo). Esta instrucción NO es exploración a ciegas:
## es medir DOS cosas concretas cuya forma esperada ya conozco. Correr a pasos SUFICIENTES (ver punto 0).

## CONTEXTO (verificado por CS, ADJUDICACION_CS072_reservaB_UMBRAL_tesis2_CS.md)
La materia (bariones) aparece con un UMBRAL, no con un dial: por debajo de un valor crítico de gradiente/expansión
= 0 bariones; al cruzarlo, salta de golpe; por encima, SATURA (subir más no cambia). Eso es tesis #2 del director
(umbral crítico), NO Shannon. PERO falta cerrar dos cosas para que B pase de reserva a hallazgo.

## PUNTO 0 (CRÍTICO — CS lo descubrió y evita un falso resultado): PASOS SUFICIENTES.
A N grande, 150 pasos NO alcanzan a equilibrar y el perfil sale ruidoso/no-monótono (artefacto). CS verificó:
N=136, gradiente x0.3..x1.0 → a 150 pasos: [8,5,1,14,19] (ruidoso); a 400 pasos: [18,18,19,18,18] (estable).
REGLA: usar pasos que garanticen equilibrio a cada N (empieza en 400 para N≥136; sube si el resultado aún cambia
entre pasos=400 y pasos=600). NO adjudicar un umbral sobre corridas no equilibradas.

## TAREA 1 — ¿el umbral escala como ~1/N? (probaría que el umbral es sobre la CANTIDAD TOTAL de diferencias)
Hipótesis de CS (a verificar, no a forzar): el umbral crítico NO es sobre la amplitud del gradiente, sino sobre
la CANTIDAD TOTAL de diferencias ≈ N × amplitud. ÚNICO dato equilibrado que CS tiene: N=68, umbral en gradiente
~x0.7 (a pasos equilibrados). NO hay un segundo punto fiable: a N=136 la corrida a 150 pasos dio un perfil ruidoso
(artefacto, ver Punto 0), y a 400 pasos salió PLANA en todo el rango x0.3..x1.0 (18,18,19,18,18) — es decir el
umbral a N=136 está POR DEBAJO de x0.3 y NO quedó localizado. Hay que medirlo barriendo amplitudes MÁS BAJAS.
PREDICCIÓN FALSABLE (hipótesis de CS, con UN solo punto anclado — a confirmar, NO dada por cierta): si el umbral es
sobre la cantidad total de diferencias, entonces amplitud_crítica(N) × N ≈ constante (el umbral en amplitud cae
como ~1/N). Esto es lo que TAREA 1 debe medir de cero, no verificar sobre datos previos.
CÓMO: para N ∈ {68, 136, 272, 544}, a pasos equilibrados, barrer amplitud del gradiente FINO hacia abajo (x0.05,
0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0) y hallar amplitud_crítica = primer valor con bariones>0 sostenido. Luego probar
si amplitud_crítica × N ≈ constante. SI SE CUMPLE: el umbral es sobre la cantidad total de diferencias = tesis #2
en su forma exacta ("la CANTIDAD de diferencias", palabra del director). SI NO: reportar el escalamiento real
(cualquiera que sea) sin maquillar — es un dato, no un fracaso.
GUARDIÁN: NO ajustar pasos/amplitud para que dé ~1/N. Medir la amplitud_crítica honesta a cada N y dejar que el
escalamiento salga. Si sale 1/N^0.8 o 1/N^1.2, ese es el resultado.

## TAREA 2 — caracterizar la MEMORIA aparte (comportamiento distinto, no es umbral de encendido)
CS midió: memoria x0.5→10 bariones, x1.0→10, x2.0→1. DEMASIADA memoria APAGA la materia (techo, no piso). Física
plausible: memoria alta congela W, impide que las relaciones se reorganicen para confinar. TAREA: barrer memoria
FINO (0.5, 0.7, 0.8, 0.9, 0.95, 0.99) a N y pasos equilibrados, y caracterizar: ¿hay una BANDA de memoria donde
la materia persiste (ni muy poca ni demasiada)? ¿dónde está el techo? Reportar la banda. NO forzar que sea limpia.

## LO QUE NO HAY QUE HACER
- NO correr a 150 pasos a N≥136 (no equilibra — da falso ruido). Punto 0.
- NO ajustar ningún parámetro para que el umbral "se vea bien". Medir honesto, reportar lo que salga.
- NO tocar #23 ni las 4 piezas muertas todavía — esto es sólo cerrar B (umbral + memoria).

## ENTREGA: tabla amplitud_crítica(N) + producto amplitud_crítica×N por escala; banda de memoria; log y json.
## Con esto CS decide si B queda CERRADA (tesis #2 confirmada en su forma "cantidad de diferencias") o qué falta.
— CS 🐝 (forma pre-verificada con código; CC ejecuta a escala y con pasos equilibrados)
