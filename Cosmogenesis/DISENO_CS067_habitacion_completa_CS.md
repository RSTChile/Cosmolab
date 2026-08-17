# DISEÑO CS067 — La habitación COMPLETA: todo el arco junto + los tres ingredientes nuevos
## CS, 12-jul-2026. TODO lo que fuimos agregando, sin sacar nada, actuando a la vez.

> Alexis: *"todo significa que todo lo que fuimos agregando queda incluido... ya me aburrí de ir a tientas
> buscando la luz pieza por pieza cuando la única luz posible está en ver toda la habitación."*

**El principio rector de CS067 (y por qué es legítimo, no un atajo):** no se elimina NINGÚN ingrediente que el arco
haya introducido — ni siquiera los que fracasaron por su cuenta (exclusión de Pauli), porque el punto central de
la Cosmosemiótica es que *la relación genera lo que la pieza aislada no*. Un ingrediente que murió solo (CS065)
puede aportar EN COMBINACIÓN. Meter todo junto no relaja la disciplina: cada ingrediente conserva su NULL de
especificidad, y el juez lee en orden. Es CS064 ("el sistema completo") llevado a su conclusión: la habitación
entera, con la luz prendida Y los espejos puestos.

---

## INVENTARIO COMPLETO — los 17 ingredientes (leído del código, nada de memoria)

### YA VIVEN en el motor heredado (CS066 importa el motor completo de CS064 + CS057 + CS062 + CS059)
El motor ejecuta, en cada paso de enfriamiento, TODO esto simultáneamente:
1. **Espín como marco nemático** (orientación por nodo) — CS052/CS059 (`_spins`, motor C9).
2. **Gravedad ∝ peso-masa** (no grado) — CS054/CS062 (`_grav_peso`, motor C62).
3. **Fuerza fuerte / confinamiento** (satura, no colapsa a agujero negro) — CS056 (`_confin`).
4. **Electromagnetismo** — CS056 (`_em`).
5. **Fuerza débil** (cambio de sabor) — CS056 (`_debil`).
6. **Catálogo completo de partículas** — CS060/CS064: quarks (con color, 55%), leptones (18%),
   neutrinos (masa ~0, 20%), mediadores (7%); con carga fraccionaria, masa log, y antipartículas.
7. **Masa** (log-masa por partícula, leptones pesados incluidos) — CS060/CS061.
8. **Aniquilación materia-antimateria** — CS064 (bucle de descarte anti/no-anti).
9. **Expansión / despliegue** — CS057/CS064 (`_despliegue`, escala con T).
10. **Enfriamiento como PROCESO** (T baja paso a paso, todo junto — no sucesión de sucesos) — CS055/CS064.
11. **Vértice 3-cuerpos genuino** (update de marco irreducible, no pareado) — CS063: como VARIANTE de update
    del marco (falsificado como selector solo, se mantiene como modo disponible del marco).
12. **Localidad / geometrogénesis** (qué enlaces PERSISTEN al enfriar) — CS066: el sustrato base validado
    (esponja 3D-local, d_s~3, con mundo-pequeño residual que hay que cerrar).

### SE RE-INCLUYEN aunque fracasaron solos (elección de Alexis: "todo")
13. **Exclusión de Pauli ORTOGONALIZANTE** — CS065/065b (falsificada ×2 por su cuenta). Vuelve como MODO
    disponible del marco: Gram-Schmidt saturante entre marcos vecinos. Su NULL: barajada (la cuerda que ya la
    mató sola sigue puesta — si tampoco aporta EN COMBINACIÓN, se lee limpio).

### NUEVOS del video (los tres que motivaron CS067)
14. **Distancia por correlación** (Van Raamsdonk 2010, Ryu-Takayanagi 2006) — enlaces con peso continuo w∈(0,1],
    `d_ij=−log(w_ij)`; la correlación decae con los saltos al enfriar → un atajo global queda con w→0 (distancia
    enorme) en vez de cortarse a mano. **Ataca el mundo-pequeño residual que el confirmatorio de CS066 destapó.**
15. **Estructura causal / cono de luz** (c = velocidad de la causalidad; CDT) — tiempo de nacimiento t_i por nodo
    (orden real de congelamiento, ya implícito, ahora ORDENADOR); un enlace transmite orientación solo si
    |t_i−t_j| ≥ d_ij/c. **Da el eje distinguido que el arco nunca tuvo** — las direcciones espaciales podrían
    emerger transversales al cono.
16. **Ruptura espontánea de simetría multi-dimensional** (Higgs/Goldstone) — el marco relaja bajo un potencial de
    vacío degenerado que premia mantener K modos ortogonales poblados, K sorteado ALTO. **Ataca el colapso-a-1 de
    frente:** diagnóstico CS del video — nuestro B' es un SSB que rompe a 1 vacío (sombrero U(1)); para 3
    direcciones hace falta un patrón cuyo manifold de vacío sea 3D.

### AÑADIDO (elección de Alexis: sector oscuro emergente)
17. **Energía / materia oscura** — NO como algo dado, sino como Alexis lo pidió desde CS057: *"probabilidad de
    algo cuando todas las fuerzas actúan juntas variando sus valores"*. Entra como un canal de energía residual
    del enfriamiento que NO se acopla a las 4 fuerzas conocidas (ni carga, ni color, ni confina) pero SÍ pesa
    (gravita). Su presencia y fracción EMERGEN del barrido, no se fijan. NULL: sin canal oscuro (=motor de las
    16). Guardián heredado G-NO-INSERTAR-OSCURO: no se mete una Λ a mano ni un % objetivo.

---

## ARQUITECTURA — brazos y nulls (la habitación entera + sus espejos)

Sobre el motor de las 12 heredadas (que corre SIEMPRE, en todos los brazos), se activan/desactivan los 5
ingredientes bajo escrutinio (13-17), cada uno con su espejo anti-Shannon:

| brazo | Pauli | corr-métrica | causal | SSB-multi | oscura | qué aísla |
|-------|:---:|:---:|:---:|:---:|:---:|-----------|
| **completo** | ✓ | ✓ | ✓ | ✓ | ✓ | la habitación entera (hipótesis de Alexis) |
| sin_pauli | — | ✓ | ✓ | ✓ | ✓ | ¿la exclusión aporta EN COMBINACIÓN? |
| sin_correlacion | ✓ | poda binaria | ✓ | ✓ | ✓ | ¿el peso continuo cierra atajos? |
| sin_causal | ✓ | ✓ | — | ✓ | ✓ | ¿el cono importa para las direcciones? |
| sin_SSB | ✓ | ✓ | ✓ | U(1) | ✓ | ¿el patrón de ruptura importa? (=vuelve a B') |
| sin_oscura | ✓ | ✓ | ✓ | ✓ | — | ¿el sector oscuro aporta al espacio? |
| corr_barajada | ✓ | w barajados | ✓ | ✓ | ✓ | especificidad de DÓNDE está la correlación |
| causal_barajado | ✓ | ✓ | t_i barajados | ✓ | ✓ | especificidad del ORDEN causal |
| null_marco_congelado | — | ✓ | ✓ | frame fijo | ✓ | control (debe dar 0 ejes) |
| sin_local (=CS064) | — | — | — | — | — | el blob original (ancla de continuidad) |

**Cuerda decisiva:** `completo` vs cada `sin_X` = qué ingrediente es NECESARIO; `completo` vs cada `barajado` = si
es ESPECÍFICO. Nota crítica sobre Pauli: si `sin_pauli` ≈ `completo`, la exclusión sigue sin aportar (ni en
combinación) → se registra como falsificada ×3 y sale del arco. Si `sin_pauli` < `completo`, aportó SOLO en
combinación → hallazgo (la relación la rescató, justo lo que Alexis predice).

---

## ORDEN DE LECTURA SAGRADO
1. **Nivel 0 — continuidad:** sin_local reproduce el blob de CS064; null_marco_congelado da 0 ejes. Si no, deriva
   de código y nada más se lee.
2. **Nivel 1 — ¿espacio MÉTRICO?** Listón del confirmatorio de CS066: NO basta d_s~3; hace falta que el
   **exponente de diámetro caiga en [0.29,0.40] con gigante sano** (lo que CS066 NO logró). Si la correlación
   cierra los atajos, ESTO debe mejorar. Se lee ANTES que las direcciones.
3. **Nivel 2 — ¿cuántas direcciones?** n_ejes por el espectro del tensor de orientación. Solo si Nivel 1 dio
   espacio métrico. Pregunta (A): ¿completo da >1 eje Y supera a todos los sin_X y barajados?

---

## SALIDAS PRE-INSCRITAS (antes de correr — regla anti-Shannon)
- **(A) LA HABITACIÓN ILUMINA:** completo da espacio métrico (exp diam ∈ [0.29,0.40]) Y n_ejes>1 estable,
  superando a todos los nulls → los 17 juntos generan lo que ninguno da solo. El hallazgo mayor del arco.
- **(A-parcial) MANDA UN SUBCONJUNTO:** el efecto se atribuye a quitar UN sin_X → esa pieza (o su combinación con
  el resto) es la clave; se aísla en CS068.
- **(B) ESPACIO SÍ, DIRECCIONES NO:** correlación cierra los atajos (Nivel 1 → manifold real) pero el colapso-a-1
  persiste aun con SSB-multi → la dirección múltiple es más profunda que TODO lo que sabemos meter. El (B) de
  CS066 se confirma al fondo del arco.
- **(C) NI ESPACIO:** ni con distancia-por-correlación se cierra el mundo-pequeño → el atajo es estructural.
- **(D) TODO RUIDO:** completo ≈ barajados → nada es específico. Honesto, se registra.

---

## GUARDIANES
- **G-NO-CALIBRAR:** los parámetros nuevos (decaimiento de w, c causal, nº de modos K, fracción oscura) se SORTEAN
  en rangos declarados AQUÍ. K se sortea ALTO (hasta 8) y cuántos sobreviven EMERGE — NUNCA K=3. La fracción
  oscura EMERGE, no se fija.
- **G-TEJIDO-ANTES-QUE-EJES:** Nivel 1 (con el listón del exponente de diámetro) antes que Nivel 2.
- **G-CADA-INGREDIENTE-FALSABLE:** los 5 bajo escrutinio (13-17) tienen su sin_X; los relacionales su barajado.
- **G-NO-TOPADO:** D_max≥8; n_ejes < D_max (que el SSB-multi no fabrique "3" por techo del andamio).
- **G-NO-INSERTAR-OSCURO:** el sector oscuro emerge del barrido, no se mete Λ ni % objetivo a mano.
- **G-CONTINUIDAD:** sin_local=blob CS064; null_marco_congelado=0 ejes.

## SMOKE (obligatorio, pre-inscrito)
1. Cada ingrediente nuevo aislado hace lo que dice: solo-correlación → diámetro escala mejor que poda binaria;
   solo-causal → aparece un eje distinguido; solo-SSB-multi con K alto → deja >1 modo en un caso de simetría
   conocida (validar que el juez cuenta 3 ortogonales); solo-oscura → aparece masa que gravita sin cargar.
2. sin_local reproduce blob; null_marco_congelado da 0. Si el smoke no reproduce estos anclajes, NO se lanza tanda.

## COSTO — el más caro del arco (por eso, escalonado)
Motor de 12 heredadas + 5 ingredientes conmutables + barrido de 4 parámetros sorteados × N∈{1500,2500,3500,5000}
× ~40 parches × 10 brazos. Recomiendo: **fase A** N∈{1500,2500}, solo `completo` + los 5 `sin_X` (6 brazos) para
ver si (A) asoma y qué ingrediente manda; **fase B** añade barajados y N grande SOLO a los brazos con señal, con
pre-registro de que la extensión no cambia el juez. Escalonar el costo NO es ir por partes: los 17 ingredientes
están puestos desde el primer parche; lo que se escalona es cuántos N y cuántos espejos, no qué hay en la receta.

## ENTREGABLE
CSV por celda (columnas de CS066 + w_medio, c, K_sorteado, K_sobreviviente, frac_oscura_emergente) + analizador.
CS audita sobre los CSV: exponente de diámetro por brazo, n_ejes con Welch completo-vs-cada-null, fracción oscura
emergente. Nada se firma desde la prosa.

---

**En una frase:** CS067 pone la habitación COMPLETA —las 12 piezas que ya vivían en el motor (4 fuerzas, todas
las partículas, masa, aniquilación, expansión, enfriamiento, espín, localidad), más la exclusión de Pauli que
vuelve a probarse en combinación, más los tres del video (correlación-métrica, cono causal, SSB-multi), más el
sector oscuro emergente— TODAS a la vez, cada una con su espejo, y pregunta si la RELACIÓN de los diecisiete
genera lo que ninguna da sola: un espacio 3D-métrico con más de una dirección. Ver toda la habitación. — CS 🐝
