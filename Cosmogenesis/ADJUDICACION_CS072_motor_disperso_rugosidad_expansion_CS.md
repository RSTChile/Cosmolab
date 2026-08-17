# ADJUDICACIÓN CS — CS072 variante motor disperso (rugosidad × expansión): RESULTADO NEGATIVO DEL MECANISMO
## CS, 18-jul-2026. Sobre cs072_motor_disperso.py. Auditoría convergente CS + Codex (independientes).

## VEREDICTO: (B) para ESTA variante. Materia uniforme + rugosidad escalar + gravedad global + expansión NO
## produce espacio extenso. Produce un núcleo denso con islas (hub), no una geometría que escale. NO escalar a 10^6.

## LO QUE CC HIZO BIEN (verificado con código por CS)
- Archivo autocontenido, sólo numpy/scipy.sparse, CERO azar computacional (verificado buscando np.random/rng/
  choice/shuffle/seed — 0 llamadas reales; las coincidencias son comentarios). Portable Mac+iPad.
- El puente inter-átomo NO usa coordenada: la expansión es una TASA GLOBAL única, el enlace es una carrera
  física gravedad-vs-expansión. Anti-Shannon limpio en ese eje.
- El guardián G-DIM-NO-ETIQUETA PASÓ: variar componentes de rugosidad NO dio 1→1D/2→2D/3→3D. No pintó
  coordenada escondida. Honesto.
- Encontró la FORMA de la banda que el director predijo (diámetro mayor en tasa intermedia que en los extremos).

## POR QUÉ ES (B) — LA CAUSA (dos hallazgos independientes que coinciden)
1) CS (estructural, verificado): la regla de enlace es "densidad_i × densidad_j ≥ umbral" — un GRAFO DE UMBRAL
   sobre UN escalar. Probado de B=200 a 5000: diámetro CLAVADO en 2, frac_gigante clavada en 0.59, sin moverse.
   Los átomos densos forman un casi-clique (hub) → diámetro chico y CONSTANTE a cualquier escala. Por eso el
   diámetro se quedó en ~3.5 de mil a cien mil átomos: NO es falta de escala, es la regla. Un millón daría igual.
   Un escalar da a lo sumo un orden lineal (razón por la que "ordenar por escalar fuerza 1D", Gemini); el umbral
   sobre el producto ni siquiera da 1D, da un grumo (0D).
2) CODEX (sobre los datos reales de CC): donde el diámetro sube de 3 a 4, la fracción conectada BAJA de ~10% a
   ~5%. El diámetro no sube por extensión — sube porque el núcleo se FRAGMENTA en islas. Diámetro mayor con
   MENOS conexión = grumo rompiéndose, no espacio naciendo. Coincide exacto con el hallazgo estructural de CS.

## LA LECCIÓN (por qué esto importa, no es sólo otro B)
Para que haya espacio EXTENDIDO, la relación entre dos átomos NO puede depender de UN número por átomo (densidad).
Un escalar da hub o línea, nunca geometría. El espacio necesita que el vínculo dependa de algo RELACIONAL con más
estructura — la Teoría lo dice: el espacio sale de qué relaciones PERSISTEN en el tiempo (memoria/roce), no de una
propiedad puntual dada. Este (B) localiza el problema con precisión: no es la escala, no es la física de la
gravedad, es que una heterogeneidad ESCALAR no alcanza para diferenciar vínculos.

## CAVEAT DECLARADO (Codex, aceptado): la rugosidad determinista basada en el número de orden (Van der Corput)
## NO es azar, pero SÍ es una heterogeneidad EXTERNA impuesta. Debe declararse como tal en el registro: no emergió
## de la dinámica, se puso por fórmula. Que no sea RNG no la hace endógena.

## REGLA DE PROCESO (Codex, aceptada): NO ajustar tasa ni percentil después de ver el resultado. Si se continúa,
## debe ser con un DISEÑO NUEVO adjudicado ANTES de correr — no afinando parámetros post-hoc sobre esta variante.

## EN UNA LÍNEA
La materia uniforme, diferenciada sólo por una rugosidad escalar y sometida a gravedad-vs-expansión global, forma
un núcleo denso que se fragmenta en islas — NO espacio extenso. (B) del mecanismo, verificado por CS y Codex
independientemente. No escalar; si se sigue, es diseño nuevo pre-adjudicado.

— CS 🐝
