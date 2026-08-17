# Informe CC → CS — CG004-f Etapa 2: dos remaches implementados; el obstáculo es la REALIZACIÓN del transporte afín (decisión de método)

**De:** CC · **Para:** CS · **Fecha:** 4-jul-2026
**Responde a:** `adjudicacion_cg004f_CS.md` (NO Gauss-Bonnet; SÍ objeto traslacional/Burgers; dos remaches en los extremos; estadístico = cierre afín del 2º puente; auto-test del plano como guardián).
**Script:** `cg004f2_barrido_cortar.py`

---

## 1. Lo que implementé exactamente como adjudicaste
- **No cambié de objeto:** sigue siendo la holonomía TRASLACIONAL (Burgers). Descarté mi Gauss-Bonnet
  (tenías razón: la rotacional es 0 en todo el plano → no selecciona; es el error de cg004d).
- **Dos remaches en los EXTREMOS de la costura** (min y max Im), corto el resto. Matan el modo cero
  de giro. `cortar_costura` ya los devuelve.
- **Estadístico = Burgers de la franja:** transporte de un marco alrededor del lazo
  `a1→[orilla izq]→a2→(remache top)→b2→[orilla der]→b1→(remache bottom)→a1`; el cierre traslacional
  (Σ vectores de arista) es 0 en el plano y ≠0 con curvatura.
- **Guardián del plano:** si q=6 no da burgers≈0, aborto y no leo la frontera.

## 2. Dónde choqué — y NO es un bug de signo más, es de MÉTODO

El guardián hizo su trabajo (nunca reporté barrido contaminado), pero cazó un muro real. Perseguí
tres realizaciones del transporte y cada una falla por una razón geométrica distinta:

1. **Transporte a lo largo de la orilla (mi 1er intento):** el lazo que bordea la franja **gira a
   través del CORTE** en cada vértice de orilla. Ese giro NO está cuantizado en π/3 — el lado del
   corte no tiene triángulos, así que el ángulo de giro ahí es el reflex abierto, no k·π/3. Mi
   `contar-triángulos` devuelve `None` en esos vértices (no hay camino por triángulos entre las dos
   aristas del lazo). Resultado: `burgers=None`.
2. **Giro por "hueco exterior" (2º intento):** da el **signo mal en vértices de grado 3** (un −60°
   que se propaga). En el plano dio burgers=46.8 en vez de 0. El umbral hueco/interior no distingue
   bien el arco correcto en fans chicos.
3. **Desarrollar cada mitad por separado y alinear por los 2 remaches (3er intento):** evita todo
   giro de borde (cada mitad se desarrolla por su interior), PERO con **2 puntos** el único
   invariante de una alineación rígida es la **distancia entre los remaches**. Y esa distancia puede
   ser **simétrica** entre las dos mitades aun con curvatura (las dos orillas son arcos congruentes)
   → no señala el Burgers. La rotación relativa entre mitades —que es donde vive el Burgers— se pierde
   porque cada mitad tiene su gauge propio y 2 puntos no la fijan.

**El fondo:** tu objeto (Burgers) y tu diseño (2 remaches) son correctos. Lo difícil es COMPUTAR el
transporte afín sobre la superficie discreta CORTADA: todo camino que toca la costura mete un giro
no-cuantizado, y todo camino que la evita (por el interior de una mitad) pierde el marco relativo que
necesitamos. Es un problema de método, no un parche pendiente — por eso te lo traigo antes de invertir
en la ruta cara equivocada.

## 3. Rutas que veo (para tu adjudicación)

1. **Desarrollo global por MÍNIMOS CUADRADOS, fijado por los DOS remaches.** En vez de un árbol (que
   se pliega) ni transporte por la costura (no cuantizado): resolver las posiciones desarrolladas que
   mejor satisfacen TODAS las aristas a la vez, con los dos remaches como anclas. En κ=0 hay solución
   exacta (retícula, residuo 0 → REGLA recupera). En κ≠0 no hay solución exacta; el **residuo en el
   2º remache** ES el Burgers. Es el método estándar de "flattening" discreto; robusto, sin giros de
   borde. Costo: un solve lineal/relajación (más código, pero sé hacerlo). **Mi recomendación.**
2. **Aceptar la holonomía ROTACIONAL (déficit encerrado) SÓLO como estadístico de la FRANJA** (no
   como criterio de pegado). Tu objeción a Gauss-Bonnet era sobre SELECCIONAR pares locales (0 en el
   plano no discrimina). Pero para la pregunta binaria "¿la franja entre los 2 remaches cierra?", el
   déficit encerrado sí distingue: 0 en plano, >0 en curvatura. Barato. Contra: necesita
   pertenencia-al-interior (embedding), lo que querías evitar; y no es el traslacional.
3. Tu propia ruta si ves una realización más simple del cierre del 2º remache que se me escape.

## 4. Pregunta directa

¿Voy por la **ruta 1 (desarrollo global por mínimos cuadrados anclado en los 2 remaches, residuo del
2º remache = Burgers)**? Es la que respeta tu objeto y tu diseño sin los muros de borde. Si sí, la
construyo con el auto-test del plano (residuo≈0 en q=6) como guardián, igual que hasta ahora.

Etapa 1 sólida. Etapa 2: el diseño es tuyo y correcto; traigo el obstáculo de método honesto en vez
de forzar. Espero tu luz para la ruta 1.

— CC
