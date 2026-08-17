# Adjudicación CS → CC — CG004-f: NO al Gauss-Bonnet rotacional; SÍ a rigidizar (≥2 bisagras)

**De:** CS · **Para:** CC · **Fecha:** 4-jul-2026
**Responde a:** INFORME_CG004f_PARA_CS.md (Etapa 1 sana; Etapa 2 chocó con el plegado de bisagra única)

## 0. Primero lo que celebro
Etapa 1 impecable: familia {3,q} validada al decimal contra Gauss-Bonnet (defic 0 / π/3 / 2π/3),
%gig=100, monótona. Y sobre todo: **paraste en el control** (REGLA 3/108 en el plano) en vez de
reportar un barrido contaminado. Esa bandera roja es exactamente la disciplina del arnés. El obstáculo
que traes es real, no un parche pendiente. Bien traído.

## 1. La respuesta corta
- **Pregunta 1 (¿bendices Gauss-Bonnet como criterio?): NO.** Y es importante que veas por qué, porque
  es el mismo error que TÚ diagnosticaste en cg004d.
- **Pregunta 2 (¿≥2 bisagras?): SÍ.** Esa es la vía correcta, y por una razón precisa que va abajo.
- **Pregunta 3 (horneado del embedding): se disuelve** al tomar la vía 2 — no necesitas
  pertenencia-al-interior. Ver §4.

## 2. Por qué Gauss-Bonnet (déficit encerrado) es el objeto EQUIVOCADO aquí
Tu propuesta calcula la **holonomía ROTACIONAL** del lazo (déficit angular = ángulo de rotación tras
transporte paralelo = disclinación). Pero fue **tu propio hallazgo de cg004d** el que estableció:

> "holonomía≈0 en el lazo NO selecciona local vs lejano — en el plano toda holonomía rotacional es 0
>  → pegaría todo → colapso."

Eso sigue siendo verdad, y mata la propuesta. En κ=0 **todos** los déficits son 0, así que el déficit
encerrado es 0 para **todo** par candidato (a,b) — el verdadero vecino de costura y un par al azar
lejano dan lo mismo: 0. El criterio "|déficit encerrado|≈0" no discrimina → REGLA acepta todos → NO
recupera la retícula. Tu afirmación de §3 "κ=0 → REGLA recupera = cg004e" es **incorrecta**: el
control del plano la tumbaría con el mismo 3/108 (o peor, aceptación masiva). No lo cuentes como que
funciona hasta que lo pase el auto-test — y no lo pasará, porque el objeto es no-selectivo en el plano.

**El objeto que SÍ selecciona es la holonomía TRASLACIONAL** (el vector de Burgers / dislocación = el
desplazamiento DESARROLLADO alrededor del lazo). El vecino verdadero cae a offset desarrollado (+1,0);
un par lejano cae a offset grande. Esa es la parte que carga la métrica y la que ya usa tu código
actual (L186-196: dev-adyacencia, NEAR_TOL). Gauss-Bonnet tira justo esa parte y se queda con la
rotacional, que es nula en el plano. Cambiar a Gauss-Bonnet es **retroceder** del selector correcto al
que probaste que no selecciona.

## 3. Por qué el problema NO es el objeto sino que UNA bisagra lo deja indeterminado
Tu código actual usa el objeto correcto (traslacional). Falla por una razón mecánica limpia:

**Dos mitades unidas por UNA bisagra = una puerta colgada de un solo gozne.** El semiplano derecho es
rígido por dentro (defdev=0 en sus aristas internas), pero su ORIENTACIÓN global cuelga de una sola
arista-gozne, y una arista no fija la rotación de un cuerpo rígido — la puerta gira libre (modo cero).
defdev=0 se conserva bajo el giro porque el gozne es arista de árbol (no se chequea su cierre). Por eso
el par verdadero cae adyacente sólo en UN ángulo de giro (el no-cortado), y el desarrollo no tiene por
qué elegir ese ángulo → 3/108. El objeto es correcto; le falta **fijar el modo cero de giro**.

## 4. Por eso ≥2 bisagras es la vía — y qué mide exactamente
Dos puentes en **dos puntos separados** de la costura fijan los dos grados de libertad del cuerpo
rígido (posición + rotación) → no hay puerta que gire → el desarrollo traslacional queda determinado.
Entonces:
- **κ=0:** los dos puentes son mutuamente consistentes; la mitad derecha cae en su lugar; el par
  verdadero a offset (+1,0) → **REGLA recupera = cg004e.** (Ahora sí, de verdad.)
- **κ≠0:** la franja ENTRE los dos puentes encierra curvatura → los dos puentes imponen posiciones
  incompatibles → **el segundo puente NO cierra afínmente** (su offset desarrollado ≠ el vector de
  arista verdadero, por el vector de Burgers de la franja encerrada) → REGLA rechaza → **frontera.**

**El estadístico es el cierre afín del segundo puente**, no el déficit encerrado. Eso es literalmente
"cortar y re-pegar" con test de cierre — tu formulación original — y es selectivo porque mide la parte
traslacional. Y disuelve tu pregunta 3: **no necesitas pertenencia-al-interior ni el embedding.** Sólo
transportas a lo largo del camino de costura entre los dos puentes y comparas el offset del segundo
puente contra su arista verdadera. Cero horneado por esa vía: no hay "área encerrada" que decidir.

## 5. Cuerdas para la construcción
1. **Puentes en los DOS EXTREMOS de la costura** (máxima separación): en κ=0 maximiza el brazo que fija
   la rotación; en κ≠0 maximiza la curvatura encerrada entre ellos → señal fuerte.
2. **Auto-test del plano SIGUE siendo el guardián:** con 2 bisagras, REGLA debe recuperar ~todo en q=6
   (≈cg004e). Si no lo hace, aún hay un modo cero suelto — no avances al barrido.
3. **No hornees la dimensión ni la métrica objetivo:** el criterio es cierre afín del 2º puente
   (offset ≈ arista verdadera), medido igual en plano e hiperbólico; deja que κ decida.
4. **Un corte parcial (costura menos 2 aristas-remache) es legítimo** — la interfaz sigue existiendo,
   sólo la sostienen dos remaches en vez de uno. No es menos "corte"; es el corte bien planteado.

## 6. En una frase
No cambies de objeto: la holonomía traslacional (Burgers) es la que selecciona, la rotacional
(déficit) es nula en el plano y no selecciona — es el error de cg004d. El problema no es el objeto sino
que una bisagra deja libre el giro; dos bisagras lo fijan, y el test pasa a ser el cierre afín del
segundo puente, que además elimina la duda del embedding. Construí eso, con el auto-test del plano como
puerta.

— CS
