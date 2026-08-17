# ESPECIFICACIÓN CS — COMPONENTE 22: FLUCTUACIONES CUÁNTICAS QCD (desbloquea el bloqueo honesto de Codex)
## CS especifica la representación computacional física. Verificado con código ANTES de escribir (las 4 pruebas
## abajo tienen su output real). CC puede codificar esto; NO elige otra arquitectura por su cuenta.

## QUÉ ES (fiel a la física y a la definición del director "el gluón es la relación")
La fluctuación cuántica QCD NO es un RNG, NO son objetos virtuales con identificador, NO es ruido ni una tasa.
ES la ENERGÍA DE CAMPO que emerge de las RELACIONES (gluones) entre los quarks de un hadrón. Hecho físico real:
~99% de la masa del protón NO son los quarks de valencia (~1%); es la energía del campo gluónico + mar q-qbar.
Los gluones que "se crean, se dividen y se comparten" (director) = esa energía relacional DINÁMICA del enlace.
Por eso es un componente aparte de la "fase cuántica" CS069: aquella es coherencia de fase; ésta es energía de
campo del sector fuerte. No se cubren con una casilla.

## REPRESENTACIÓN PERMITIDA (la única que CS autoriza; CC la implementa tal cual)
Para cada hadrón (barión = 3 quarks ligados; mesón = 2), sea W_sub la submatriz de relaciones (gluones) entre
sus constituyentes. Entonces:
    E_campo_QCD = g_fuerte * suma de W_sub sobre PARES NO ORDENADOS (triángulo superior)
    masa_efectiva = masa_valencia + E_campo_QCD
donde g_fuerte es la constante de fuerza fuerte (CONSTANTE FÍSICA ESTRUCTURAL — permitida por
G-NO-PARAMETRO-FORMA, igual que una carga o una masa; NO es una perilla que dibuje el resultado).
La energía de campo sale de W (la estructura relacional que ya evoluciona en el bucle), NO de la nada, NO de un
sorteo. Es DINÁMICA: cuando las relaciones se refuerzan/debilitan paso a paso, E_campo cambia con ellas — eso ES
la fluctuación (gluones que se dividen y comparten), sin crear ni una sola partícula virtual clásica.

## CANTIDADES CONSERVADAS
- Energía total: E_campo_QCD se CONTABILIZA en el libro de energía del estado (sale de la energía relacional,
  no se inventa). Al aniquilar o decaer, esa energía va a radiación y se conserva en el total.
- Carga, color, número bariónico: los aporta la valencia (los 3 quarks), intactos. QCD sólo añade masa/energía.

## EL NULL — sin_fluct_qcd
Apaga ÚNICAMENTE E_campo_QCD (masa_efectiva = masa_valencia, sólo los quarks de valencia). Conserva valencia,
carga, color, número bariónico y todas las demás leyes. Sirve para medir qué cambia gracias a QCD — sobre todo
la GRAVEDAD, que depende de la masa: sin QCD la masa hadrónica cae ~100×, así que la gravedad entre hadrones
cae ~100×. El NULL mide exactamente ese aporte.

## VERIFICACIONES DE CS (output real, reproducible — no de memoria)
- V-A (RETIRADA — NO es verificación): el ejemplo con W=0.31 fue afinado a mano para reproducir el 99% conocido
  del protón — es circular, no valida nada. Lo que SÍ se puede afirmar: la FORMA masa=valencia+campo_relacional
  es la estructura correcta de QCD (la masa hadrónica es dominada por energía de campo, no por valencia), pero la
  FRACCIÓN exacta (99% u otra) debe EMERGER de la W que evoluciona en el motor, NO ponerse a mano. Es un
  OBSERVABLE a medir, no un número a reproducir. Si el motor da una fracción muy distinta, es un dato, no un error.
- V-B invariante a índice: reordenar los 3 quarks da masa idéntica (diferencia 0.00) — suma sobre pares NO
  ordenados, no depende del orden del array. SIN sesgo de índice.
- V-C NULL mide algo real: apagar QCD deja masa 0.009 vs 0.939 → gravedad ~104× menor. No es cosmético.
- V-D determinista: cero np.random; energía sale de W (contable), no de la nada.

## PROHIBIDO (repetido de Codex, ratificado por CS)
- NO np.random/ruido/sorteo llamado "fluctuación cuántica".
- NO miles de objetos virtuales clásicos con identificador individual.
- NO tasa de aparición / cupo de pares / vida media por conveniencia (G-NO-PARAMETRO-FORMA).
- NO hacer aparecer/desaparecer energía sin conservarla en el total.
- NO declarar que el W escalar "ya contiene todo QCD" sin este libro contable explícito.

## ACLARACIÓN DEL DIRECTOR (ratificada por CS, corrige una sobreafirmación previa de CS): la ε ORIGINAL es el
## GRADIENTE TÉRMICO, no el +1 poblacional materia-antimateria. Que ambos vengan de la misma raíz es HIPÓTESIS
## PENDIENTE, no un hecho. NO afirmar que "es la misma ε" hasta probarlo.

## Con esto CC puede marcar el componente 22 como especificado e implementarlo. Dudas → preguntar a CS.
— CS 🐝
