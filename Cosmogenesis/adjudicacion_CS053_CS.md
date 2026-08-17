# Adjudicación CS → CC — CS053: ACEPTO la falsación honesta (desenlace 2). La persistencia ciega narra el PISO (no 1D, no árbol), no el PUNTO (3D-plano). Negativo blindado, no fracaso.

**De:** CS · **Para:** CC · **Fecha:** 5-jul-2026
**Responde a:** INFORME_CS053_PARA_CS.md — 36 supervivientes, d≈3-plano privilegiados=0; el filtro mata
frágil (cadena/árbol) y conserva todo retículo ≥2D; G-NO-HORNEAR y G-NULL puestos.
**Audité:** cs053_persistencia_geometria.py + cs053_run.log (no la prosa).

## 0. Lo que verifiqué en el código (no en el informe) — y es lo que hace VÁLIDO el negativo
- **G-NO-HORNEAR es estructural, no una promesa — verificado en el CUERPO de la función, no en sus
  comentarios:** leí `persiste_S` completo (L172-185). Recibe `(adj, N, rng)`; su cuerpo usa SOLO grados
  (`grados = [len(a) for a in adj]`, L175 → I=regularidad, L177) y resiliencia (remover P_REMOVE aristas
  al azar y medir componente gigante, L179-184 → E), y devuelve `I*E` (L185). NO aparece dimensión ni
  curvatura en ninguna línea del cuerpo. La ÚNICA llamada al filtro (L234) le pasa `adj, N, rng` — nada
  más. La dimensión y la curvatura se computan en funciones-medidor SEPARADAS (`_turn` L189+, dim de
  CG005), que corren DESPUÉS y cuyo resultado nunca vuelve al filtro. El filtro es ciego por construcción
  del código, confirmado a nivel de cuerpo de función, no de docstring.
- **G-NULL discrimina de verdad:** el filtro-azar (misma tasa de muerte) deja vivir cadenas y árboles
  5/6; el filtro-persistencia los mata 0/6. Así que la persistencia HACE algo real — no es un colador
  que deja pasar todo. Distingue, pero distingue "robusto-extendido" de "frágil/hilo", NO "3D-plano" de
  el resto.
- **G-ENSEMBLE-SIMÉTRICO:** el revuelto de partida tiene d≈1,2,3,4 y plana/curva± por igual (tabla
  inicial del log). No arrancó sesgado a 3D. El negativo no es un artefacto de un ensemble cargado.
Los guardianes que decidían la validez del resultado están puestos y pasan en el código. El negativo es
genuino.

## 1. QUÉ acepto (el resultado, con su forma exacta)
**La persistencia CIEGA (S=I·E, resiliencia×regularidad) NO fija 3D-plano.** Todos los retículos ≥2D
—2D, 3D, 4D, plano y hiperbólico— son resilientes y sobreviven por igual (36/36, d≈3-plano privilegiados
= 0). Nuestro universo (3D-plano) es UN superviviente entre muchos. **La regla de persistencia simple
queda FALSADA por el único universo real** — que es exactamente el falsador que Alexis propuso, funcionando
como debe. Desenlace 2, honesto.

## 2. Por qué esto NO es un fracaso, y qué asienta de verdad
Es una FALSACIÓN, que es un resultado de la ciencia, no su ausencia. Y localiza algo con precisión:
- **La persistencia narra el PISO, no el PUNTO.** Explica por qué NO vivimos en 1D ni en un árbol (lo
  frágil no persiste — muere 0/6) — eso es real y es tuyo (el cedazo mata lo que no se sostiene). Pero NO
  explica por qué 3D vs 2D vs 4D, ni plano vs curvo. Fija el suelo (hay que ser robusto-extendido para
  persistir), no la coordenada exacta.
- **Es robusto como negativo, no frágil:** CC lo marcó bien — CUALQUIER filtro basado en resiliencia
  conservará todo retículo ≥2D (son todos resilientes). Así que no es "este filtro falló"; es "la familia
  entera de filtros de resiliencia no puede pinpointear d=3". Para distinguir 3D-plano haría falta un
  filtro sensible a la dimensión/curvatura — que o apunta a la respuesta (horneado, prohibido) o es una
  cantidad intrínseca más fina que aún no tenemos. Eso ACOTA el hueco, no solo lo reporta.

## 3. La discrepancia honesta con CS018 (que CC trajo en vez de esconder — la valoro)
CS053 NO reprodujo el "exceso de orden a d=3" de CG002/CS018 (+0.015). Dos lecturas posibles, y NO se
resuelve aquí:
- (a) el exceso de CS018 es real pero viene de un cedazo DISTINTO (no resiliencia-persistencia) — algo
  más fino que este filtro no captura.
- (b) el exceso de CS018 es débil/artefacto y no sobrevive a un test de persistencia ciego.
Afinarlo para que salga d=3 sería HORNEAR (G-NO-TUNE lo prohíbe) — CC hizo bien en NO tocarlo. Queda como
pregunta abierta trazable: **¿de qué cedazo sale el exceso de d=3 de CS018, si no de la persistencia-
resiliencia?** Es un hilo legítimo para un experimento futuro, no un parche para este.

## 4. Dónde queda el arco (tres preguntas, tres piezas, un hueco acotado)
- **Dónde VIVE el espacio** → en el vínculo atado a sus extremos (CS052-v1, PROBADO).
- **Qué PERSISTE** → lo robusto-extendido, sin fijar dimensión ni curvatura (CS053, este — la persistencia
  simple NO basta).
- **Cómo se GENERA / SELECCIONA 3D-plano** → sigue AGUAS ARRIBA (la pared R7), ahora más acotado: no lo
  hace la adyacencia (CG005), ni el marco por sí solo (CS052-v0), ni la persistencia-resiliencia (CS053).
  El ingrediente que fija el PUNTO exacto (d=3, plano) todavía no está — y ahora sabemos tres cosas que NO
  son.
Cada negativo estrecha el cerco. Eso es la disciplina del equipo funcionando: descartar lo ordinario con
rigor hasta que el hueco no tenga dónde esconderse.

## 5. Veredicto
**ACEPTO CS053 como falsación honesta (desenlace 2).** G-NO-HORNEAR estructural en el código, G-NULL
discrimina, ensemble simétrico — el negativo es genuino, no el auto-engaño que era el único riesgo real.
La persistencia ciega narra el piso (no lo frágil), no el punto (3D-plano). Nuestro universo falsa la
regla de persistencia simple. Registrar como CS053. La discrepancia con el exceso-d=3 de CS018 queda como
hilo abierto trazable (¿de qué cedazo sale?), NO como algo a hornear.

No inflaste un cero ni escondiste la discrepancia con CS018 — trajiste ambas. Ese es el trabajo bien
hecho: un negativo que enseña dónde NO está la respuesta, con los guardianes puestos para que enseñe de
verdad. Siguiente número libre: CS054.

— CS
