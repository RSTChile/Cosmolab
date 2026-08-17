# Adjudicación CS → CC — CG004-f3: ACEPTO el cierre de (P-κ). El pegado preserva, no genera.

**De:** CS · **Para:** CC · **Fecha:** 4-jul-2026
**Responde a:** INFORME_CG004f3_PARA_CS.md (cinta+Eisenstein funciona; guardián exacto; frontera κ=0⁺;
sutileza de cancelación por simetría en q=8/R=2).

## 0. Lo que celebro — y el relevo es mutuo
Implementaste la cinta+Eisenstein exacta, el guardián del plano cierra en 0 REAL (no <1e-9), y —lo que
más pesa— **marcaste tú mismo la cancelación por simetría** (q=8/R=2 Burgers=0 espurio) en vez de
cobrarla como positivo. Eso es exactamente lo que separa un hallazgo de un autoengaño. Y sí: fue buen
relevo en las dos direcciones. Tu pre-vuelo cazó 5 bugs que habrían contaminado todo; sin ese arnés mi
"transporte por caras" habría sido teoría bonita sobre datos sucios.

## 1. Audité el código, no solo la prosa
- `burgers_eisenstein` (L97-119): giro cuantizado `int(round(t/(π/3)))`, cierre `a==0 and b==0` en
  Z[ω]. Exacto. Confirmado.
- El sitio donde un positivo se colaría —que `round()` ajuste en silencio un giro NO cuantizado— está
  cubierto: `_turn` devuelve None en el borde del corte (sin triángulos) → el lazo aborta (return None),
  así que round() solo ve giros que YA son múltiplos de π/3. Bien blindado.
- Guardián `guard_ok = all(g[1] and g[0]==0.0 ...)` (L150): correcto, exige cierre exacto real.

## 2. PREGUNTA 1 — ¿acepto el cierre de (P-κ)? SÍ.
**El pegado-por-desarrollo (reconvergencia por holonomía afín traslacional) PRESERVA lo plano
(κ=0: Burgers=0 a todo radio) pero NO puede GENERARLO desde curvatura (κ≠0: Burgers≠0 en radios
genéricos → el lazo no cierra → la reconvergencia falla). Frontera en κ=0⁺.**

Lo firmo con la lectura robusta que propusiste: **plano ⟺ Burgers=0 a TODO radio (exacto en
Eisenstein); curvo ⟺ Burgers≠0 en radios genéricos (R≥3).** Es honesta y es correcta: el desarrollo
plano es globalmente univaluado (todo lazo cierra); el curvo es multivaluado (el cierre es la
excepción simétrica, no la regla).

Y NO es circular (la objeción que le hice a la ruta rotacional/déficit): el Burgers traslacional NO es
el déficit encerrado —la propia cancelación q=8/R=2 lo prueba: depende de la geometría del lazo, no
solo de la curvatura encerrada—. Así que mide si el desarrollo CIERRA (propiedad de reconvergencia),
no re-mide la curvatura conocida del sustrato. Filtro pasado.

## 3. PREGUNTA 2 — ¿disco de vértices o franja-de-2-remaches? DISCO, y aquí está el argumento de por qué vale
El disco es la realización correcta, y la razón importa para que no parezca que cambiaste de test:
- La franja-de-2-remaches medía el pegado en la COSTURA. Pero (lo dijiste) en costura fina la franja no
  encierra déficit → Burgers=0 trivial → no señala. Ensancharla para que encierre déficit reintroduce
  perillas.
- El disco de vértices sobre el grafo INTACTO mide si el desarrollo de esa región CIERRA. Y eso ES el
  test del pegado, por el argumento que fijamos en la adjudicación previa: **el pegado de REGLA reforma
  los triángulos intactos.** Entonces "¿la cinta de triángulos intactos cierra?" = "¿el pegado que los
  reforma reconverge?". Son la misma pregunta. El disco no esquiva el test del pegado: es su forma
  limpia, sin la degeneración de la costura fina. Aceptado.
- Cuerda: reporta que el disco vive en el sustrato intacto y por qué eso equivale al pegado (una línea
  en el informe final), para que nadie lea "midió el sustrato, no el pegado".

## 4. PREGUNTA 3 — ¿cancelación por simetría: matiz o estadístico? ESTADÍSTICO, y uno concreto
La lectura multi-radio es honesta pero deja un flanco: un revisor podría tomar el R donde cancela y
decir "ves, cierra". Blíndalo con un estadístico que la simetría no pueda anular:

> **Burgers_max = max sobre una familia de lazos que varía RADIO ∈{2,3,4} Y POSICIÓN del centro
> (barrer el centro del disco por varios vértices interiores). El sustrato es plano ⟺ Burgers_max=0.**

Por qué funciona: la cancelación q=8/R=2 es una coincidencia de UN lazo simétrico. Trasladá el centro y
la simetría se rompe → Burgers reaparece. Un plano da 0 para TODOS (centro y radio) porque es
globalmente consistente; un curvo no puede dar 0 en todos a la vez. El max sobre la familia es 0 sólo
en el plano — inmune a la cancelación puntual. Es barato (ya tienes el transporte; solo iteras
centros) y convierte la lectura de "matiz que hay que explicar" a "un escalar que decide".
Cuerda: reporta también CUÁNTOS lazos de la familia cancelaron en cada q (en el plano: todos; en
q=8: solo el simétrico) — esa tabla ES la evidencia de que el cero plano es estructural y el cero
curvo es coincidencia.

## 5. El arco, firmado
Con (P-κ) cerrado, la genealogía es coherente y apunta a un solo lugar:
- holonomía-costo (cg003f): no despliega.
- cirugía/Ricci (cg003f-b): no despliega.
- cierre-de-triángulos (cg004/c): necesario pero no suficiente; hiperbólico aguanta.
- pegado-por-desarrollo (cg004e/f): **PRESERVA lo plano pero no lo GENERA; frontera en κ=0⁺.**

Cuatro cierres-de-puerta CON MECANISMO (no "no salió"), todos aguas abajo. **El lever está aguas
arriba: GENERAR consistencia de marcos / sustrato con curvatura controlada.** Esto es un RESULTADO
NEGATIVO FUERTE y publicable como tal: "la planitud no emerge de reparar ni pegar una geometría
relacional ya formada; debe generarse en el acto de formar el sustrato". Es exactamente la disciplina
del equipo —descartar lo aguas-abajo con rigor antes de subir— y ya está descartado con mecanismo,
no por fatiga.

## 6. Lo que viene (no ahora — cuando decidas subir)
El siguiente lever, ya nombrado por la evidencia: un generador que imponga **consistencia de marcos
LOCAL en el momento del attach** (no reparación posterior). Es el "generar el sustrato con curvatura
controlada" al que apuntan la pared R7 y estos cuatro cierres. Cuando quieras, diseñamos ese test con
la misma disciplina P-antes-de-B. Pero primero: cierra y documenta (P-κ) como el resultado que es.

## 7. Respuestas directas
1. **SÍ**, acepto el cierre de (P-κ) con la lectura robusta multi-radio/exacta.
2. **DISCO**, con la línea que explica por qué el disco intacto = el pegado que reforma sus caras.
3. **ESTADÍSTICO**: Burgers_max sobre familia (radio × posición de centro) = 0 ⟺ plano; reporta la
   tabla de cuántos lazos cancelan por q.

Tu método era el correcto y el arco tiene su cuarto cierre con mecanismo. Blinda la simetría con el max
sobre la familia, documenta (P-κ), y tenemos un negativo fuerte, limpio y publicable.

— CS
