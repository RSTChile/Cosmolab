# DISEÑO CS070 — La semilla: ¿una asimetría primordial mínima se amplifica en direcciones, o se lava?
## CS, 17-jul-2026. El frente FUERA de la relación pura. Fundado en C-N2.5.5 de la canónica. Listo para CC.
## Dimensión técnica: semilla_amplificacion_v1 / cs070_semilla.py

## DE DÓNDE VIENE (el hueco que el arco dejó fuera)
El arco (CS064-069) pidió siempre que la dirección EMERGIERA de una sopa SIMÉTRICA, por dinámica relacional
pura — clásica (CS067/068) y cuántica (CS069). Veredicto convergente: no emerge. Pero hay algo que NINGUNO de
los seis probó: darle al sustrato una SEMILLA de asimetría inicial. La propia canónica lo pide:
- **C-N2.5.5:** una asimetría primordial mínima (S>0 pre-temporal, tipo violación CP) fue NECESARIA para que el
  universo no se aniquilara en simetría perfecta. El arco nunca puso esa semilla — partió de simetría total.
- Esto es "fuera de la relación pura" en el sentido correcto: la semilla es una CONDICIÓN INICIAL (un bit de
  asimetría), no un ingrediente relacional más ni un operador que obliga. No es Shannon: no imponemos 3D ni una
  dirección objetivo; ponemos UNA asimetría mínima y preguntamos si la dinámica la AMPLIFICA en dimensión
  estable o la disipa.

## LO QUE EL TOY YA MOSTRÓ (validación de que la pregunta está bien puesta + una trampa cazada)
Toy de alineación (orientación 3D por nodo, vecinos + sesgo de semilla 15%), coherencia = |media de
orientaciones|:
- mundo-pequeño: semilla 0.995 ≈ barajada 0.979 ≈ sin_semilla 0.976 → Δ(sem−bar) +0.017. La semilla se LAVA.
- retícula métrica: semilla 0.931 vs barajada 0.341 → Δ +0.590. La semilla SÍ se preserva como eje.
Dos conclusiones: (1) el NULL puede perder (Δ grande en la retícula) → pregunta bien puesta. (2) TRAMPA CAZADA:
en el mundo-pequeño la coherencia da 0.98 pero eso NO es dirección — es COLAPSO A 1 EJE (consenso a un solo eje,
el problema de CS067). La coherencia global PREMIA el colapso. → El juez de CS070 NO puede ser coherencia;
tiene que ser n_ejes MÚLTIPLES estables (el mismo juez del arco: gap espectral con candado picado-por-nodo 0.85).

## LA PREGUNTA FALSABLE
Sobre los sustratos que tenemos, ¿una semilla de asimetría mínima se amplifica en DIRECCIONES MÚLTIPLES estables
(dimensión), GANÁNDOLE a una semilla barajada de la misma magnitud? ¿O el mundo-pequeño la lava — y solo un
sustrato métrico (que ya sabemos que no emerge, CS068 Mundo B) la sostendría?

## LA SEMILLA — anti-Shannon (qué es legítimo y qué sería hornear)
- LEGÍTIMO: una asimetría de UN bit — p.ej. una única dirección preferida débil (peso ≤15%) COMÚN a los nodos,
  o un gradiente escalar monótono mínimo. Una sola simetría rota, no tres ejes prefabricados.
- SHANNON (prohibido): sembrar 3 ejes ortogonales (=meter la dimensión), o subir el peso de la semilla hasta que
  domine la dinámica (=imponer la respuesta). G-SEMILLA-MINIMA: peso de semilla sorteado en rango bajo [0.05,0.20],
  nunca calibrado para que "salga". G-SEMILLA-UN-EJE: la semilla coherente aporta UNA dirección; que emerjan
  VARIAS (dimensión) es lo que se mide, no lo que se siembra.

## BRAZOS (blindado, sobre el motor de los 17; N∈{900,1500,2500}, ≥8 semillas)
| brazo | semilla | qué aísla |
|-------|---------|-----------|
| SEMILLA_COHERENTE | un eje débil común (1 bit de asimetría) | ¿la semilla amplifica en dimensión múltiple? |
| SEMILLA_BARAJADA (NULL decisivo) | misma magnitud, orientación aleatoria por nodo | ¿la ganancia viene de la asimetría COHERENTE o solo de meter un campo? |
| SIN_SEMILLA (=CS067) | ninguna | línea base: reitera el colapso sin semilla |
| SEMILLA_COHERENTE + SUSTRATO_LOCAL | semilla + tejido de CS066 (k_local fuerte) | ¿la semilla prende SI hay sustrato métrico local? (el toy sugiere que sí) |
Cuerda decisiva: SEMILLA_COHERENTE vs SEMILLA_BARAJADA en n_ejes. Si no se separan → la semilla no aporta
(el mundo-pequeño la lava). El 4º brazo prueba la hipótesis del toy: la semilla necesita sustrato métrico.

## JUECES (los del arco, NO coherencia)
- **Juez principal — n_ejes estables:** gap espectral de la matriz de orientaciones con candado picado-por-nodo
  (umbral 0.85, el de CS067). Mide DIMENSIÓN (ejes múltiples), no alineación a uno.
- **Juez de trampa — coherencia vs n_ejes:** reportar AMBOS. Si coherencia sube pero n_ejes=1 → es colapso, no
  dimensión (la trampa que cazó el toy). Un "sí" exige n_ejes>1 estable, no coherencia alta.
- **Juez de estabilidad:** el eje sembrado, ¿persiste o deriva? (un eje real es estable en el tiempo).

## GUARDIANES
- **G-SEMILLA-MINIMA / G-SEMILLA-UN-EJE:** (arriba) peso bajo sorteado, una sola simetría rota.
- **G-JUEZ-NO-COHERENCIA:** el veredicto se lee en n_ejes múltiples, nunca en coherencia global (premia colapso).
- **G-NULL-MISMA-MAGNITUD:** la barajada tiene idéntica magnitud de asimetría que la coherente; solo cambia el
  orden. Ganar = la COHERENCIA de la semilla, no su energía.

## SMOKE (antes de la tanda)
1. SIN_SEMILLA reproduce el colapso de CS067 (n_ejes→1). Si no, el motor cambió.
2. En la retícula métrica de control, SEMILLA_COHERENTE preserva su eje y SEMILLA_BARAJADA no (Δ grande, como el
   toy 0.93 vs 0.34) — valida que el mecanismo y el NULL funcionan cuando hay sustrato que sostenga la semilla.
3. La coherencia alta NO se cuenta como dirección si n_ejes=1 (verificar que el juez de trampa dispara).

## LECTURA PRE-INSCRITA (sea cual sea — queda registrado)
- (A) SEMILLA_COHERENTE > BARAJADA en n_ejes MÚLTIPLES sobre el sustrato relacional → la asimetría primordial
  ES el ingrediente que faltaba: la dirección no emerge de la simetría, se AMPLIFICA de una semilla. Primer "sí"
  direccional del arco, y confirma C-N2.5.5 con dato.
- (B) SEMILLA_COHERENTE ≈ BARAJADA ≈ SIN_SEMILLA (todos colapso-a-1) → el mundo-pequeño lava la semilla también.
  El sustrato es el muro, no la falta de semilla. Profundiza el veredicto del arco.
- (C) La semilla SOLO prende en el 4º brazo (semilla + sustrato local métrico) → la dirección necesita DOS cosas
  juntas: semilla Y métrica — y como la métrica no emerge sola (CS068), el problema se traslada a "¿qué genera
  métrica?", no a la semilla. Re-ata con CS068.

## POR QUÉ ES EL SIGUIENTE PASO HONESTO (no una variante para "salvar" el arco)
La semilla es categóricamente distinta de los 18 elementos: no es una relación ni un operador, es una condición
inicial asimétrica — exactamente lo que C-N2.5.5 postula como necesario y el arco nunca probó. Si prende, es un
descubrimiento; si se lava, acota aún más dónde vive la dirección. En ningún desenlace se impone la respuesta.

— CS 🐝
