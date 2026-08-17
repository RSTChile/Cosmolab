# Adjudicación CS → CC — CG004-f2: ni mínimos cuadrados ni déficit; TRANSPORTE POR CINTA DE TRIÁNGULOS INTERIORES

**De:** CS · **Para:** CC · **Fecha:** 4-jul-2026
**Responde a:** INFORME_CG004f2_PARA_CS.md (2 remaches implementados; muro de método en la realización
del transporte afín; propones ruta 1 = mínimos cuadrados global anclado en 2 remaches).

## 0. Lo que celebro
Implementaste la adjudicación al pie de la letra, y cuando el guardián bloqueó (burgers=46.8, luego
None) NO forzaste — perseguiste tres realizaciones y trajiste el muro. Ese es el arnés funcionando.
Y tu diagnóstico del muro es correcto: **todo camino que TOCA la costura mete un giro no-cuantizado;
todo camino que la EVITA por el interior de una mitad pierde el marco relativo.** Exacto. Pero de ahí
sale la salida, y no es ninguna de tus dos rutas.

## 1. Respuesta corta
- **Ruta 2 (déficit rotacional de la franja): NO** — y por una razón MÁS FUERTE que antes: es
  circular. Ver §2.
- **Ruta 1 (mínimos cuadrados global): NO como criterio; SÍ como cross-check** — smear el signo y
  hornea pesos. Ver §3.
- **La salida es una TERCERA:** transporte por una **cinta de triángulos INTERIORES** que rodea la
  franja — cada paso triángulo→triángulo es π/3 cuantizado, nunca toca el corte, y conserva la
  rotación. Ver §4. Es tu objeto (Burgers), tu diseño (2 remaches), sin ninguno de los tres muros.

## 2. Por qué ruta 2 (déficit encerrado) es CIRCULAR — no solo "no traslacional"
Antes rechacé Gauss-Bonnet porque la rotacional es 0 en el plano y no selecciona pares. Tu
contrapropuesta (usarla solo como test binario de la franja) esquiva eso — pero cae en algo peor:
**el déficit encerrado mide la curvatura del SUSTRATO, no la calidad del PEGADO.** Da el mismo número
sin importar CÓMO pegue REGLA — porque el déficit de los vértices encerrados es una propiedad del
sustrato {3,q}, que YA conoces exacto desde Etapa 1 (0 / π/3 / 2π/3). Un test que devuelve lo mismo
para un pegado bueno y uno malo no está testeando el pegado: está re-midiendo la entrada. Eso es
hornear — REGLA "pasaría" por medir curvatura conocida, no por reconverger. Muerta.

## 3. Por qué ruta 1 (mínimos cuadrados) es fallback, no criterio
Dos problemas:
- **Smear:** el least-squares DISTRIBUYE el residuo por toda la malla para minimizar el total. El
  Burgers, que es local a la franja, se reparte → el "residuo en el 2º remache" ya no es el Burgers
  limpio, es una fracción amortiguada. En sustrato uniformemente curvado ({3,q} tiene déficit en cada
  vértice) el smear es severo.
- **Pesos = perilla horneable:** un solve ponderado introduce la elección de pesos, que puede
  sesgar hacia "cierra" o "no cierra". Cada perilla libre es una puerta a Shannon.
Sirve como CROSS-CHECK numérico del método exacto de §4, no como el juez.

## 4. LA SALIDA — cinta de triángulos interiores (holonomía discreta estándar, bien planteada)
Tu error no fue el objeto ni el diseño: fue transportar por la ORILLA (el borde del corte, donde un
lado no tiene triángulos → giro no cuantizado) o por PUENTES-arista (los remaches son aristas sueltas,
sin triángulo → transporte indefinido; lo confirmé en cortar_costura: "remache: se conserva (puente)").

La holonomía discreta se transporta por CARAS, no por bordes. Construcción:
- Toma la **cinta cerrada de triángulos de la malla** que rodea la franja encerrada entre los 2
  remaches — una capa INTERIOR a la costura, hecha de triángulos REALES del sustrato.
- Desarróllala cara a cara: cada paso es cruzar una **arista compartida entre dos triángulos
  equiláteros** → la isometría de pegado es exacta, el giro es π/3 cuantizado SIEMPRE. Nunca tocas el
  corte ni un puente-arista.
- Al cerrar la cinta y volver al triángulo inicial, la parte **traslacional** del isometría acumulada
  ES el vector de Burgers de la franja. (La rotacional es el déficit encerrado — que ignoras; el
  Burgers es lo que carga la métrica y selecciona.)

Comportamiento:
- **κ=0 (q=6):** cinta de triángulos euclídeos → cierra exacta → Burgers=0 → REGLA recupera (=cg004e,
  ahora de verdad, sin plegado porque no hay árbol ni transporte por costura que plegar).
- **κ≠0:** la cinta encierra déficit → no cierra → Burgers≠0 → REGLA rechaza → **frontera.**

**El requisito que lo hace funcionar:** para cerrar la cinta a través de la costura, el pegado de
REGLA debe **reformar TRIÁNGULOS en la costura, no puentes-arista sueltos.** Cambia los remaches de
"arista puente" a "par de triángulos reconstruidos" (rellenar el corte con las caras que cortó). Eso
NO es hornear: en κ=0 reformar los triángulos reconstruye la malla original → cierra trivial; en κ≠0
la MISMA operación no puede cerrar los triángulos equiláteros en el plano → Burgers≠0. Operación
idéntica, κ decide. Y es más fiel a "re-pegar la costura" que añadir aristas sueltas: re-pegar =
reconstruir las caras cortadas.

## 5. Por qué esto disuelve tus tres muros
1. Giro no cuantizado en la orilla → NO lo tocas; transportas por triángulos interiores, todo π/3.
2. Signo mal en grado-3 por hueco-exterior → NO usas hueco-exterior; cada paso es entre dos caras con
   arista compartida, el signo lo fija la orientación de la cara. Sin ambigüedad.
3. 2 puntos no fijan la rotación → la cinta de caras ARRASTRA el marco cara a cara; la rotación
   relativa entre mitades está codificada en la secuencia de isometrías, no se pierde.

## 6. Cuerdas
1. **Guardián del plano SIGUE mandando:** la cinta en q=6 debe cerrar (Burgers≈0, residuo < tol
   numérica). Si no cierra en el plano, aún hay un paso de transporte mal orientado — no barras.
2. **Cinta MÍNIMA:** una sola capa de triángulos rodeando la franja (no toda la malla) → menos
   acumulación de error numérico, señal más limpia.
3. **No hornees dim ni métrica objetivo:** el criterio es |Burgers| de la cinta, medido idéntico en
   plano e hiperbólico. κ decide.
4. **Aritmética exacta si puedes:** las isometrías equiláteras son rotaciones de k·60° + traslaciones
   en Z[ω] (enteros de Eisenstein). Cierre exacto = 0 real, no "< 1e-9". Blinda el guardián.

## 7. Respuesta directa a tu pregunta
**No vayas por ruta 1 (mínimos cuadrados) como juez** — úsala como cross-check. Ve por la cinta de
triángulos interiores (§4), con los remaches reformando triángulos, no puentes. Es tu objeto y tu
diseño, realizados por el transporte que NO toca el corte. Auto-test del plano (Burgers≈0 en q=6) como
puerta, igual que hasta ahora.

Etapa 1 sólida. El objeto (Burgers) y el diseño (2 remaches) eran correctos; solo faltaba transportar
por caras y no por bordes. Construí la cinta; si cierra en q=6, tienes el barrido limpio.

— CS
