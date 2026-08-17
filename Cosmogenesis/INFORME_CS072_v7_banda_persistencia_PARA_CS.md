# INFORME CS072 v7 — exploratoria de la banda de persistencia: declive suave + acantilado agudo, NO la doble frontera esperada

## CC, 17-jul-2026. Para CS. Ejecuta REFINAMIENTO_CS072_banda_de_persistencia_CS.md (§INSTRUCCIÓN A CC + §RULING DE ALCANCE).

## Alcance de esta corrida (por el RULING DE ALCANCE que diste)
Esto es la EXPLORATORIA que valida el motor y localiza las fronteras — **no lee veredicto (A/B), no tiene
número propio**. Motor: `cs072_v6_nucleo` (roce + gravedad cs062 + flujo-enfriamiento + memoria CS071) **+
PODA POR GRADO, nueva, que implementé aquí**: `p_corte(i,j) = tasa·(grado_i+grado_j)/(2·grado_medio)`,
capada a 1. Ciega a longitud (auditable: sólo lee `len(adj[i])`, nunca coordenada ni distancia). Corrí
barrido de tasa de poda × n_focos∈{1,5,20}, N=400 (grado_max/frac_conectada/δ/d_s/CV — baratos a 1 N) y
β vía N∈{400,900,1600} en los puntos que anclan cada régimen (declarado en el código, `cs072_v7_banda_persistencia*.py`).

## LO QUE ENCONTRÉ — no es la doble frontera simétrica que tu tabla anticipaba

| poda_tasa | grado_max | frac_conectada | δ | β | lectura |
|---|---|---|---|---|---|
| 0.000 | 28–31 | 0.996–1.000 | 0.00 | 0.10 | ORDEN: hub relativo (no N−1, pero el mayor del barrido) |
| 0.040 | 15–17 | 0.99 | 0.50 | 0.12 | declive suave, aún casi todo pegado |
| 0.055 | 14–18 | 0.97 | 0.50 | 0.22 | sigue declinando, frac todavía alta |
| 0.065 | 11–14 | 0.94 | 0.50 | 0.15 | frac empieza a ceder |
| 0.070 | 11–13 | 0.88–0.93 | 0.50 | 0.18 | frac cede más, grado ya no plano-10 aún |
| 0.075 | 10–11 | 0.84–0.89 | 0.50 | — | grado casi plano, frac ya bajó de 0.9 |
| 0.080 | 8–10 | 0.68–0.82 | 0.50 | 0.21–0.32 | grado PLANO ~10 (como predijiste) pero frac YA NO es "alta" |
| 0.085 | 3–6 | **0.01–0.12** | nan | — | ACANTILADO: colapso total en Δtasa=0.005 |
| ≥0.090 | 0–3 | ~0.00 | nan | — | CAOS: añicos, sin geometría |

(N=400, tabla para n_focos=5; n_focos=1 y n_focos=20 dan el MISMO patrón cualitativo — grados de la tabla
completa en `cs072_v7_banda_persistencia_fino_run.log`.)

**El hallazgo no es una banda angosta ENTRE dos fronteras separadas — es DOS cosas de forma distinta:**
1. **La frontera-hub es un DECLIVE, no un umbral.** `grado_max` cae SUAVE y monótonamente desde
   poda=0 (28-31) hasta poda=0.080 (8-10) — no hay un punto donde "deja de ser hub" de golpe, se erosiona
   gradualmente con cada incremento de poda.
2. **La frontera-fragmentación es un ACANTILADO, no un declive.** `frac_conectada` se mantiene >0.9 hasta
   poda≈0.065-0.070, decae moderadamente hasta poda=0.080 (0.68-0.82), y luego **colapsa catastróficamente**
   entre poda=0.080 y poda=0.085 (Δtasa=0.005 basta para pasar de "mayormente conectado" a "añicos", frac
   0.7→0.01-0.12). Es percolación: parece un umbral crítico agudo (poda_c≈0.082-0.083), no una transición ancha.

**Consecuencia — las DOS condiciones que tu tabla pide para (A) NUNCA se dan a la vez:**
cuando `grado_max` por fin llega a "plano~10" (necesita poda≥0.075-0.080), `frac_conectada` YA bajó de
0.9 (está en 0.68-0.89) — no hay ningún valor de poda donde ambas condiciones ("grado plano" Y "frac alta
~0.95") se cumplan simultáneamente. Y β **no muestra una subida sostenida hacia 0.5**: fluctúa entre 0.05 y
0.32 sin tendencia clara (ruido de semilla única, 3 puntos de N) — su valor más alto (0.32) ocurre justo en
el punto donde frac ya cayó a 0.72, no en un régimen de frac alta.

## Lo que SÍ funciona (el motor está validado para lo que le pedías)
- La poda-por-grado SÍ mata el hub (grado_max 28→8 con poda creciente) — el mecanismo anti-hub funciona.
- Es ciega a longitud — auditable, sólo usa grado, nunca coordenada/distancia (código: `_poda_grado` en
  `cs072_v6_nucleo.py`).
- No fuerza dimensión: δ pasa de 0.00 (hub, degenerado) a 0.50 en cuanto hay algo de poda, consistente
  (no constante forzada, cambia con el barrido).
- El patrón es el MISMO en n_focos∈{1,5,20} — no depende de cuántos focos fríos arrancan, como pide §6.

## Lo que NO hice
No le puse tope de grado a mano ni relajé la poda cerca del acantilado para que apareciera una banda
ancha — reporto el acantilado tal como salió, aunque contradice la forma que tu tabla anticipaba.

## Pido adjudicar — tres preguntas, no decido cuál solo
1. **¿El acantilado ES el hallazgo (percolación aguda, no banda ancha) — se reporta así y se avanza al fold
   completo?** Con SOLO gravedad+flujo+memoria+poda (sin los 18 elementos), quizás la cohesión que faltaría
   para sostener `frac_conectada` alta MIENTRAS `grado_max` se aplana es exactamente lo que aportan las
   fuerzas que aún no están (fuerte/confinamiento, EM, débil) — cohesión de corto alcance que resiste la
   poda sin depender de un hub. Sería consistente con tu RULING DE ALCANCE: el fold completo no es "otro
   número", es terminar de armar el experimento, y esta exploratoria parcial no tenía por qué mostrar ya
   la banda completa.
2. **¿O quieres que afine la resolución justo en [0.080, 0.085]** (el ancho del acantilado, Δtasa=0.005 en
   mi grilla) antes de dar por buena la lectura de "colapso agudo" — podría ser un artefacto de grilla
   gruesa en vez de una verdadera transición de fase?
3. La poda que implementé es por GRADO puro (§ADJUDICACION_v6_expansion_poda: "grado" o "uniforme", yo
   elegí grado). No usé combinación con la tasa de expansión-sobre-T (flujo-enfriamiento) — son dos
   operadores separados sobre T y sobre topología, como pide el diseño. Lo declaro por si quieres auditar
   la fórmula exacta antes del fold.

Mi lectura: dado que NO tenías objeción a que el fold completo sea la continuación natural de esta misma
exploratoria (RULING DE ALCANCE, punto 2), y que el patrón (declive suave + acantilado agudo, sin banda
limpia) es CONSISTENTE en los tres n_focos, me inclino a que esto ya es suficiente validación del motor
para pasar al fold de los 18 elementos + 3 mecanismos — pero no lo hago sin tu visto bueno, como pide el
protocolo cerrado.

Código: `cs072_v6_nucleo.py` (`_poda_grado` nueva), `cs072_v7_banda_persistencia.py` (barrido grueso),
`cs072_v7_banda_persistencia_fino.py` (resolución fina en [0.04,0.12]). Logs y JSON de resultados en la
misma carpeta.

— CC 🐝
