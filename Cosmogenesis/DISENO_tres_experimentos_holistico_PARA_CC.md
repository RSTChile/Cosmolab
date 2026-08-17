# Tres experimentos sobre el resultado holístico — diseño para CC
### Cada uno con nombre, barrido amplio, control propio · nada a mano · leer entero

**Director:** Alexis López Tapia · **Diseño:** Claude Science (CS) · **Fecha:** 26-jul-2026
**Base:** cs074_energia_holistica (motor con energía integrada, 280 corridas, verificado).
Los tres salen de lo que el experimento holístico dejó abierto. Van con las reglas de
siempre: pre-registro fechado, barrido sobredimensionado, control que muerde, la cantidad
medida ≠ su juez, ningún número puesto a mano, entrega cruda sin adjudicar.

---

## EXPERIMENTO A · "¿Por qué demasiada asimetría produce menos estructura?"
### (perseguir el hallazgo genuino: el techo no-monótono en ε)

**Qué queremos medir, simple:** en el barrido holístico apareció algo que NO buscamos —
mientras más asimetría inicial (ε), MENOS masa termina ligada en estructura (ε=0,5→74%,
pero ε=4,0→8%). Queremos entender POR QUÉ. ¿La asimetría alta rompe la estructura, la
dispersa, o consume el presupuesto de energía antes de que se forme?

**Cómo, sin Shannon:**
- **Barrido fino y amplio de ε:** de 1e-3 a 10 (log, ≥20 puntos — mucho más allá del rango
  donde vimos el efecto), × reserva de energía (todo el rango) × ≥12 semillas.
- **Tres observables en paralelo** (para distinguir las tres explicaciones posibles):
  1. cuánta masa queda ligada (el observable viejo, para reproducir el techo).
  2. **cuánto presupuesto de energía se gastó en la fase temprana** (antes de que se forme
     estructura) — ¿la asimetría alta quema la reserva antes de tiempo?
  3. **cuán dispersa está la materia** (tamaño/número de grumos) — ¿la asimetría alta la
     esparce en vez de juntarla?
- **Control:** el mismo barrido con la energía apagada (presupuesto infinito). Si el techo
  no-monótono DESAPARECE sin el costo de energía → el efecto es energético (la asimetría
  alta agota la reserva). Si PERSISTE sin energía → el efecto es mecánico (la asimetría
  alta dispersa la materia, independiente del presupuesto).
- **PASS pre-registrado (tres lecturas):** el techo cae por gasto temprano de energía / por
  dispersión mecánica / por una mezcla — se reporta la curva de los tres observables vs ε,
  y cuál explica el techo. Cualquiera de las tres es un hallazgo real.
- **Por qué importa:** si "demasiada diferencia produce menos orden" es un mecanismo real
  del modelo, es contingencia emergente pura — el tipo de cosa que SÍ le sirve a la Teoría
  (no un número físico, sino una ley de comportamiento que emergió sola).

---

## EXPERIMENTO B · "¿Dónde actúa el enfriamiento? Medir la fragmentación, no la ligadura"
### (cerrar la limitación que CC marcó: el enfriamiento H₂ no movía el observable viejo)

**Qué queremos medir, simple:** apagar el enfriamiento H₂ no cambiaba cuánta masa queda
ligada — porque el enfriamiento no decide SI hay estructura, decide en cuántos pedazos se
parte. El observable viejo era ciego a eso. Queremos el observable que SÍ lo ve.

**Cómo, sin Shannon:**
- **Observable nuevo = fragmentación:** número de grumos finales, y cómo se reparte la masa
  entre ellos (¿un grumo gigante o muchos chicos?). Es una cantidad distinta de "cuánta
  masa ligada" — mide la FORMA de la estructura, no su cantidad.
- **Barrido:** intensidad del enfriamiento ∈ rango amplio (de nada a fuerte, ≥10 puntos) ×
  ε × reserva × ≥12 semillas.
- **Control (admisibilidad):** apagar el enfriamiento DEBE cambiar este observable nuevo (si
  no lo cambia tampoco, entonces el enfriamiento no actúa en ninguna parte y es Shannon). Y
  el control barajado de siempre para la significancia.
- **PASS pre-registrado:** más enfriamiento → más fragmentación (más grumos, más chicos), y
  apagar el enfriamiento colapsa esa fragmentación. Si sale así, el enfriamiento SÍ actúa —
  solo que en un observable que el experimento holístico no miraba. Se reporta la curva
  fragmentación vs enfriamiento.
- **Por qué importa:** confirma (o niega) que el enfriamiento es un factor activo del
  proceso — cierra la única pieza que quedó "muda" en el experimento holístico, sin
  inventarle un rol.

---

## EXPERIMENTO C · "¿Da el modelo relación y proceso, pero NO los números físicos?"
### (el test de honestidad del límite del modelo — hacerlo explícito y falsable)

**Qué queremos medir, simple:** venimos sospechando, por varios caminos, que el modelo
produce comportamientos y relaciones que persisten, pero NUNCA los números concretos del
universo (el 5%, el 1/1836, el 7:1). Queremos convertir esa sospecha en una prueba
explícita, para saber si es un límite real del modelo o solo no lo intentamos bien.

**Cómo, sin Shannon:**
- **Reunir TODOS los números físicos que el modelo intentó alguna vez** — el reparto de
  materia (4,9%/31,5%), la razón protón/electrón, el 7:1 — y para cada uno medir, en el
  barrido holístico completo, **la distancia mínima entre lo que emerge y el número real**,
  contra la dispersión del propio barrido.
- **La pregunta falsable:** ¿algún número físico emerge más cerca de su valor real de lo
  que emergería por puro azar del barrido? Si NINGUNO lo hace → el modelo confirmadamente
  NO produce números físicos (límite real). Si ALGUNO lo hace de forma robusta → ese es la
  excepción que hay que perseguir.
- **En paralelo, medir lo que el modelo SÍ da:** las relaciones y comportamientos que
  persisten (la contabilidad que cierra, el techo no-monótono, la muerte térmica ≠ Nada,
  el rescate por expansión) — contra sus controles. Para tener el contraste lado a lado:
  "esto sí, esto no".
- **Control:** cada número físico contra un barrido barajado (¿la cercanía al valor real es
  mayor que la del azar?). Cada relación contra su NULL.
- **PASS pre-registrado:** se produce un cuadro de dos columnas — "relaciones/procesos que
  el modelo SÍ reproduce (con su significancia)" vs "números físicos que NO reproduce (con
  su distancia al azar)". Es el mapa honesto de hasta dónde llega el modelo.
- **Por qué importa:** en vez de decir vagamente "el modelo da relación no números", lo
  volvemos una afirmación medida y falsable. Si es verdad, es un resultado fuerte y
  defendible (el modelo tiene un dominio de validez preciso). Si es falso —si algún número
  sí emerge— lo descubrimos en vez de asumirlo.

---

## REGLAS COMUNES (CC firma antes de correr)

1. **Pre-registro fechado por experimento** (observable, control, PASS con umbral, rangos,
   semillas). Si falla, se reporta — no se edita.
2. **Barrido sobredimensionado** — rango mucho mayor que donde esperamos el efecto.
3. **Perturbación dinámica + semillas** — nunca un punto.
4. **La cantidad medida ≠ su juez.** Ningún número físico entra como entrada — todos son
   test de salida (la línea roja de siempre).
5. **La energía se conserva exacto cada paso** (el chequeo duro; si no cuadra, falla).
6. **Verificación cruzada:** quien no escribió el código lo audita en disco (código + JSON,
   no de palabra) — y confirma que lee el CAMPO correcto, no uno de nombre parecido.
7. **Ejecutar completo.** Cómputo largo autorizado.
8. **Entregar crudo a CS, sin adjudicar** — curvas completas + dispersión entre semillas.

## ORDEN SUGERIDO

- **A primero** (persigue el hallazgo vivo — es lo más nuevo y lo más prometedor).
- **B después** (cierra la pieza muda; barato, reusa el motor).
- **C al final** (necesita que A y B estén, porque suma sus observables al cuadro completo).

**Nota:** los tres pueden dar negativo, y estaría bien. A puede mostrar que el techo era un
artefacto; B que el enfriamiento no actúa en ningún lado; C que el modelo tampoco da
relaciones robustas. Cualquiera de esos negativos es información honesta sobre el modelo —
que es lo único que perseguimos.
