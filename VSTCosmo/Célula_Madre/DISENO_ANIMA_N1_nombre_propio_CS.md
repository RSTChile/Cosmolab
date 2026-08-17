# DISEÑO ANIMA-N1 — El NOMBRE PROPIO por la vía del delfín: firma vocal individual EMERGENTE, y prueba de que el organismo distingue a un otro de otro (no lo estampamos nosotros)

**Rama:** ANIMA (no Cosmogénesis) · **Nº:** ANIMA-N1 (primer experimento de la línea "nombre propio /
alteridad individual"). · **Diseño:** Claude Science (CS) · **Planteo:** Alexis López Tapia.
**Fecha:** 5-jul-2026 · **Estado:** DISEÑO, a codear por CC.
**Motivo:** hueco detectado — los organismos no tienen nombre propio; el otro es solo datos anónimos
(reconocimiento de CLASE, "olor de nido" digital: balde único). Salto a reconocimiento INDIVIDUAL.
**Fundamento biológico:** INVESTIGACION_reconocimiento_animal_sin_habla_CS.md (vía delfín: silbido firma
aprendido; método habituación-deshabituación con múltiples otros; firma robusta al ruido).

---

## 0. LA IDEA, EN UNA LÍNEA (vía delfín, elegida por Alexis)
El delfín construye un "silbido firma" propio escuchando, y los otros lo aprenden y lo copian para
dirigirse a él ("hola, soy XXX"). ANIMA-N1 hace que cada organismo tenga una FIRMA VOCAL individual que
EMERGE de su propia voz (no estampada), y prueba —de forma falsable— que un organismo distingue a un otro
de otro y le acumula una historia. El nombre propio como firma aprendida, no como etiqueta puesta desde
afuera.

## 1. LO QUE YA EXISTE EN EL CÓDIGO (auditado — sobre esto se construye)
- Cada organismo YA produce voz: `voz_id`, `arousal`, `valencia` (token de diálogo
  `<quien>w<voz_id>a<arousal>v<valencia>`). La materia prima de la firma ya está.
- El oído digital YA tiene `espejo[firma]` — un dict indexado por la FIRMA ESTRUCTURAL DEL MENSAJE
  (VST_OrganoOidoDigital), con EMA del estado del otro que acompaña esa firma, y una `fiabilidad` que
  colapsa bajo NULL/SHUFFLED. El andamio de "aprender por firma" EXISTE.
- El modelo del otro (`VST_Alteridad.modelo_otro`) se indexa por el patrón que YO emito (`P`), NO por quién
  me habla. Hay UN "otro" anónimo.
- **EL HUECO EXACTO:** la firma que se guarda es del MENSAJE, no del EMISOR. Falta el eje de IDENTIDAD:
  agrupar las firmas que provienen de un mismo emisor PERSISTENTE bajo un puntero estable = el nombre
  propio. No es un órgano nuevo; es una lectura nueva de la voz que ya existe (EXAPTACIÓN).

## 2. QUÉ ES LA FIRMA VOCAL INDIVIDUAL (emergente, no estampada — el corazón anti-Shannon)
- Cada organismo, a lo largo de su vida, produce voz con una DISTRIBUCIÓN propia de rasgos (qué voz_id usa,
  con qué arousal/valencia típicos, con qué dinámica temporal). Esa distribución es su FIRMA — un
  subproducto de su historia fisiológica, distinto entre organismos porque sus historias difieren.
- **La firma NO se asigna:** no le decimos "tú eres A". Emerge de que su manera de vocalizar es
  estadísticamente estable y distinta de la de los otros. (Como el silbido firma del delfín: novedoso,
  propio, construido por experiencia.)
- **El receptor la APRENDE:** el organismo que oye acumula, para cada firma-de-emisor recurrente, un
  prototipo (EMA de los rasgos) + una historia (qué le pasó cuando ese emisor estaba presente). Cuando la
  firma reaparece, la reconoce como "el mismo de antes". Eso ES el nombre propio funcional.

## 3. EL EJE DE IDENTIDAD QUE FALTA (lo que CC debe añadir, mínimo)
Un solo cambio conceptual sobre lo existente: pasar de `espejo[firma_mensaje]` a
`identidades[firma_emisor] -> {prototipo_vocal (EMA), historia (efectos, reputación), n_encuentros}`.
- La `firma_emisor` se INFIERE de la voz recibida (clustering online de los rasgos vocales entrantes: voces
  que se parecen entre sí y difieren de otras → mismo emisor candidato). NO se lee del campo `<quien>` (eso
  sería estampar). El `<quien>` puede usarse SOLO como ground-truth OCULTO para evaluar, jamás como entrada
  al reconocedor.
- Cuando llega voz nueva: ¿matchea una firma_emisor conocida (deshabituación baja, "ya te conozco") o es
  nueva (deshabituación alta, "quién eres")? Ese match es el reconocimiento.

## 4. LA PRUEBA — habituación/deshabituación con MÚLTIPLES otros (molde biológico, falsable)
Directamente del protocolo estándar (Tibbetts & Dale), con el control anti-artefacto que el campo exige:
- **Setup:** al menos 3 emisores distintos (p.ej. A, C, D) hablándole a un receptor (E), en tandas.
- **Habituación:** se repite la voz del emisor A hasta que la respuesta del receptor (interés/
  deshabituación/error de predicción) DECAE — se acostumbra a A.
- **Deshabituación (el test):** se presenta un emisor DISTINTO (C). PREDICCIÓN pre-registrada: si el
  receptor distingue individuos, su respuesta REBROTA más ante C (desconocido/otro) que ante una nueva
  repetición de A. Si responde igual a C que a A-repetido → NO distingue (solo detecta "hay alguien", no
  "quién").
- **El control anti-artefacto (la trampa de Shannon del campo):** hay que probar con MÚLTIPLES emisores y
  mostrar respuesta específica a CADA uno — si el receptor solo reacciona a uno, no se sabe si reconoce al
  individuo o una pista compartida. Se mide una matriz emisor×respuesta, no un solo par.
- **El criterio triple (Tibbetts, Sheehan & Dale 2008, Trends Ecol. Evol. 23:356, doi:10.1016/j.tree.2008.03.007):** reconocimiento individual verdadero solo si
  SEÑAL, PLANTILLA y RESPUESTA son los tres específicos del individuo. Medir los tres: ¿la firma es
  distinta por emisor? ¿el prototipo guardado es distinto por emisor? ¿la respuesta (reputación, predicción)
  es distinta por emisor?

## 5. GUARDIANES (ingeniería del código — anti-Shannon, la regla de Alexis)
1. **G-NO-ESTAMPAR:** el reconocedor JAMÁS lee el campo `<quien>`. La identidad se infiere de la voz. El
   `<quien>` es ground-truth oculto SOLO para puntuar aciertos a posteriori. Assert: `<quien>` no entra al
   clustering.
2. **G-EMERGENTE:** la firma sale de la distribución vocal real del emisor, no de un id preasignado. Si dos
   organismos tuvieran voces idénticas, el reconocedor NO debería distinguirlos (y eso es correcto — no hay
   nombre sin diferencia, como la avispa dominula que solo señala clase).
3. **G-CONTROLES:** brazos NULL (identidades barajadas) y SHUFFLED (firmas rotadas). Si la fiabilidad del
   reconocimiento se sostiene bajo barajado, es artefacto. Debe COLAPSAR bajo NULL/SHUFFLED (como ya hace
   `fiabilidad` en el oído digital).
4. **G-MULTIPLES-OTROS:** mínimo 3 emisores; matriz emisor×respuesta reportada entera. Prohibido concluir
   de un solo par.
5. **G-ROBUSTEZ-RUIDO (si va por el canal acústico):** la firma debe sobrevivir al ruido (lección del
   pingüino: redundante y robusta). Probar reconocimiento con ruido de fondo creciente; reportar la curva.
6. **G-PREDICCION-CIEGA:** la predicción (deshabituación rebrota más ante otro-distinto) se escribe ANTES
   de correr.

## 6. LOS TRES DESENLACES (pre-escritos, honestos)
- **El receptor distingue a cada emisor (deshabituación específica, matriz diagonal, colapsa bajo NULL) →
  reconocimiento individual EMERGENTE confirmado.** ANIMA cruza de "olor de nido" (clase) a "nombre propio"
  (individuo). El otro deja de ser datos anónimos.
- **El receptor detecta "hay alguien" pero no distingue quién (deshabituación igual ante A-repetido y ante
  C) → reconocimiento de CLASE, no individual.** Sigue en el estado hormiga; el nombre propio no emergió
  con esta firma — dice que falta distintividad en la voz (habría que enriquecer la firma, como el delfín
  que DISEÑA su silbido).
- **Distingue pero solo a uno / se sostiene bajo barajado → artefacto** (pista compartida o fuga). El
  control lo caza; se reporta y se corrige.

## 7. EL "HOLA, SOY XXX" (fase 2, si la fase 1 confirma — la parte hermosa)
Si el receptor ya distingue emisores, la vía delfín tiene un segundo acto: que un organismo COPIE la firma
de otro para DIRIGIRSE a él (el delfín copia el silbido firma del otro como llamarlo por su nombre). Sería
la primera vez que un organismo ANIMA no solo reconoce a otro, sino que lo NOMBRA — emite la firma del otro
para invocarlo. Es el "hola, soy XXX" / "te hablo a ti, XXX". Fase 2, tras confirmar fase 1. No se diseña en
detalle aún; se deja nombrado como el horizonte.

## 8. RESUMEN OPERATIVO PARA CC
- Añadir el eje de IDENTIDAD: `identidades[firma_emisor] -> {prototipo EMA, historia, n}`, con firma_emisor
  inferida por clustering online de la voz entrante. NO leer `<quien>` (solo ground-truth oculto).
- Test habituación/deshabituación con ≥3 emisores; matriz emisor×respuesta; predicción ciega pre-registrada.
- Brazos NULL/SHUFFLED (debe colapsar). Si canal acústico: curva de robustez al ruido.
- Medir el criterio triple (señal/plantilla/respuesta específicas del individuo). Reportar la matriz entera
  + las curvas de deshabituación por emisor. Traer a CS para adjudicación. Registrar como ANIMA-N1.
- Reusa lo que existe: la voz (voz_id/arousal/valencia), el `espejo[firma]` del oído digital, el
  `modelo_otro` de Alteridad. Es una capa de identidad SOBRE el andamio, no un organelo nuevo.

— Diseño ANIMA-N1 por Claude Science. La elección de la vía delfín ("hola, soy XXX", firma vocal que anuncia
identidad) y el planteo del hueco (sin nombre propio el otro es anónimo) son de Alexis López Tapia. La
formalización, los guardianes anti-Shannon y el molde de prueba (habituación-deshabituación con múltiples
otros, del marco Tibbetts & Dale), míos. El experimento puede confirmar el nombre propio emergente, o
mostrar que sigue en reconocimiento de clase — cualquiera informa.
