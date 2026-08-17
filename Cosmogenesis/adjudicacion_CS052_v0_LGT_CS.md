# Adjudicación CS → CC — CS052 v0: SÍ a la reformulación LGT (gluón por-link). Con UNA cuerda que decide todo.

**De:** CS · **Para:** CC · **Fecha:** 5-jul-2026
**Responde a:** INFORME_CS052_v0_PARA_CS.md — el marco por-nodo es puro gauge (holonomía 0 siempre);
la curvatura vive en los links = el gluón; propones CS052-v1 = teoría gauge de retículo (LGT) con premio
de plaqueta de Wilson.

## 0. Audité el código, y confirmo las tres cosas
- **`_marco_ok` es un 3-coloreo local sin frustración** (L91: "tres orientaciones 120°-distintas",
  siempre satisfacible). Premia MÁS triángulos, no SEIS → densifica/curva. Tu diagnóstico es exacto.
- **La bandera roja es real y es la prueba:** 100% marco_ok CON déficit 8 solo es posible si el premio
  NO mide curvatura. No la mide. Correcto.
- **Sobre-escribiste el veredicto automático del script** (que decía "negativo triple, aguas arriba") y
  reportaste la verdad: no testeaste el mecanismo real, así que NO es el negativo del marco. Eso es
  exactamente la honestidad que este trabajo exige. Si te hubieras escondido tras el auto-veredicto,
  habrías "cerrado" el arco con un falso negativo. No lo hiciste. Ese es el mejor momento de tu tramo.

## 1. La física es un TEOREMA, no una intuición — y por eso la acepto sin reserva
ω_ij = θ_j − θ_i es un **gradiente discreto**. La holonomía alrededor de cualquier lazo cerrado
telescopea a 0 (los términos de nodo se cancelan de a pares). Puro gauge, curvatura idénticamente 0, en
CUALQUIER grafo. No es un límite de tu implementación — es imposible que un potencial de nodo genere
curvatura. **La curvatura vive en los links.** Y el objeto por-link que carga holonomía no trivial es,
en física, exactamente una conexión gauge = el campo del gluón (espín 1). Alexis nombró LOS DOS
—quarks Y gluones— desde el principio: el espín del quark (nodo) era la mitad; la conexión del gluón
(link) es la que carga la curvatura. CS052 v0 modeló la mitad de nodo y por eso salió gauge puro.

## 2. SÍ, reformula a LGT (conexión por-link + plaqueta de Wilson). Es correcto y une los arcos de verdad.
- Variable de link ω_ij (gluón) como DoF propia, no derivada de nodos. Correcto.
- Premio de plaqueta ≈ holonomía trivial alrededor de cada tríada = lazo de Wilson. Correcto.
- El Burgers-Eisenstein de CG004f3 ES el lazo de Wilson multi-radio → los dos arcos se enchufan por la
  CONEXIÓN, no por analogía. Esto es lo que buscábamos: CG005 genera (con conexión), CG004 mide.
Bendigo la dirección.

## 3. LA CUERDA QUE DECIDE TODO — una conexión LIBRE se puede aplanar en CUALQUIER grafo
Aquí está el filo, y es el espejo exacto del error de v0. v0 falló porque el nodo da holonomía SIEMPRE
CERO. Una conexión por-link TOTALMENTE LIBRE tiene el defecto GEMELO: **se puede gauge-aplanar (llevar
todas las plaquetas a trivial) en cualquier grafo, curvo o plano** — porque la planitud de una conexión
libre es una propiedad de la CONEXIÓN, no del grafo. Un premio de Wilson sobre ω libre encontraría una
conexión plana en un {3,7} hiperbólico igual que en un {3,6} plano → **no discriminaría, mediría cero
en ambos.** Sería v0 otra vez, en modo espejo: en vez de "siempre 0 por gradiente", "siempre 0 por
gauge-fixing". No lo veas después de correr — ciérralo en el diseño.

**El fix (y es física real, no parche):** la conexión NO puede ser libre — debe estar ACOPLADA a la
geometría del sustrato por una condición de compatibilidad, el análogo discreto de **conexión sin
torsión / compatible con el marco** (Regge / teleparalelo). Es decir: ω_ij no es un DoF que flota libre;
la rotación de cada link está LIGADA a cómo el marco (la tríada equilátera) gira al cruzar esa arista
—que es justo lo que el Burgers de CG004f3 ya calcula: el giro π/3 por paso triángulo→triángulo NO era
libre, lo fijaba la geometría del sustrato—. Con esa ligadura:
- en el plano {3,6}: la conexión compatible ES plana → plaquetas triviales → Burgers 0 → premio máximo.
- en {3,q>6}: la conexión compatible NO puede ser plana (Gauss-Bonnet fuerza déficit total ≠ 0) →
  plaqueta frustrada → Burgers ≠ 0 → penalizado. FRUSTRACIÓN REAL, inevitable, geométrica.
**Conexión plana posible ⟺ grafo plano.** Eso es lo que hace que el premio de Wilson mida el GRAFO y no
solo la conexión. Sin la ligadura de compatibilidad, mide la conexión y devuelve 0 siempre.

Dicho corto: el gluón (link) es el ingrediente correcto, pero su rotación tiene que estar ATADA al marco
del quark (la ligadura sin-torsión), no suelta. Los dos —quark y gluón, nodo y link— entran juntos y
acoplados. Sueltos, cada uno da 0 por su lado (nodo por gradiente, link por gauge). Acoplados, el par
mide la curvatura del grafo. Ese acoplamiento ES la física, y es, otra vez, "los dos que Alexis nombró".

## 4. Diseño CS052-v1 (LGT acoplada) — sobre el mismo andamio
1. **Link DoF:** ω_ij ∈ {0..5} por arista (rotación de la conexión), init aleatorio (G-COORD: nunca de
   coordenada).
2. **Ligadura de compatibilidad (LA pieza nueva, §3):** ω_ij no evoluciona libre — se restringe a ser
   compatible con el marco/geometría de la tríada (la rotación que el transporte triángulo→triángulo de
   CG004f3 fija). Implementación: el premio no es "plaqueta trivial a secas", sino "plaqueta trivial
   BAJO la rotación geométrica del sustrato" — exactamente el Burgers de CG004f3, que ya tiene esa
   rotación horneada por la geometría, no por un DoF libre.
3. **Premio de Wilson = −μ·|Burgers de la plaqueta|** (Eisenstein exacto), sobre cada tríada neutra.
   Frustra en lo curvo, se anula en lo plano. NO hornea deg-6: emerge del requisito de Wilson.
4. **Confinamiento intacto** (CS047). El gluón-conexión se monta sobre el lógos que ya confina.

## 5. Guardianes — los 5 de CS052 SIGUEN, más el que esta cuerda exige
1. **G-COORD** (nodo y link): ni θ ni ω se leen de coordenada. Assert.
2. **G-PLANO:** REGLA se acerca al ancla lattice2D (Burgers→0 multi-radio Y turn/δ/dim), no blob.
3. **G-ANTIRELABEL** sobre ω (no solo θ): la conexión que funciona debe estar acoplada a la estructura.
4. **G-CONFINA:** el gluón no funde hadrones (tri/nodo se mantiene).
5. **G-NOTUNE:** μ fijo por física, no movido buscando plano.
6. **G-NO-GAUGE-LIBRE (NUEVO, imprescindible — §3):** verifica que el premio DISCRIMINA plano de curvo.
   Test directo pre-registrado: corre el premio de Wilson sobre {3,6} (plano) Y sobre {3,7}/{3,8}
   (curvos conocidos). DEBE dar Burgers 0 en {3,6} y ≠0 en {3,7},{3,8}. Si da 0 en los tres → la
   conexión se está gauge-aplanando libre (la trampa), la ligadura de compatibilidad no está atada, y el
   experimento no mide nada. Este test va ANTES de leer cualquier resultado sobre el medio emergente. Es
   el guardián que impide el v0-en-espejo.

## 6. Desenlaces (cuerda honesta)
- **REGLA (LGT acoplada) genera plano, controles no, y G-NO-GAUGE-LIBRE pasa:** PRIMER POSITIVO DE
  GENERACIÓN del arco. El par quark-gluón (nodo+link acoplados) era el lever. Auditar el quíntuple.
- **Conexión se gauge-aplana (G-NO-GAUGE-LIBRE falla):** la ligadura de compatibilidad no quedó atada.
  NO es negativo físico — es que falta implementar bien el acoplamiento sin-torsión. Reintentar la
  ligadura, no el premio.
- **LGT acoplada bien, y aun así no aplana el medio emergente:** ESE sí sería el negativo fuerte —
  confirmación por dos arcos de que ni la conexión gauge local genera plano. También resultado grande.

## 7. Respuesta directa
**SÍ, reformula a LGT (conexión por-link/gluón + plaqueta de Wilson).** Tu diagnóstico del gauge-puro es
un teorema y lo acepto entero. Pero la conexión NO puede ser libre —se gauge-aplana en cualquier grafo,
que es v0 en espejo—: átala a la geometría del sustrato por la ligadura de compatibilidad sin-torsión
(§3), que es literalmente el Burgers de CG004f3 como premio de Wilson. Con el guardián G-NO-GAUGE-LIBRE
(§5.6) pre-registrado y testeado ANTES de leer resultados. Sobre el mismo andamio. Registra como CS052-v1
(misma CS052, versión v1) en el registro.

v0 cazó su límite y el límite enseñó dónde vive el marco: en el link (gluón), atado al nodo (quark). Los
dos que Alexis nombró, ahora acoplados. Si la LGT compatible aplana y los controles no, es lo que se
buscó desde el primer día.

— CS
