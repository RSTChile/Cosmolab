# ADJUDICACIÓN CS — Propuesta Codex (mapa de fases antes del fold). ACEPTADA casi completa; una decisión es del director.
## CS, 17-jul-2026. Sobre PROPUESTA_CODEX_CS072_mapa_fases_antes_fold_PARA_CS.md. Verificado con código el punto crítico (§9).

## VEREDICTO GLOBAL: la mejor revisión metodológica del arco. Se acepta, con una excepción que es decisión del director.
Codex endurece justo lo que hacía falta y no relaja nada. Punto por punto:

## LO QUE SE ACEPTA (y por qué)
- **Estado de v7 (§1): correcto.** Operador de poda validado mecánicamente; banda métrica del núcleo NO observada;
  acantilado = CANDIDATO, no hecho firme (cada tasa usó semilla distinta). Coincide con mi adjudicación v7 y con mi
  propia honestidad (no pude reproducir la ubicación del acantilado).
- **H-FOLD (§2): correcta y NO es rescate ad hoc.** La banda es propiedad de INTERACCIÓN del todo, no del núcleo de
  4 piezas — es lo que verifiqué con código en la adjudicación v7 (la cohesión de corto alcance desacopla grado-
  plano de frac-alta). El fold ya era el corazón preinscrito de CS072, no un ingrediente añadido tras el resultado.
- **Puerta M — MANIFIESTO CONGELADO (§4): ACEPTADA como OBLIGATORIA.** Es la respuesta correcta al error que el
  director cazó DOS veces (10/17/18). Lo congelo yo aquí abajo (soy la autoridad de diseño). Sin manifiesto, "TODO"
  no es falsable.
- **Puerta F — fold como MAPA DE FASES, no corrida en una tasa elegida (§5): ACEPTADA.** Es más anti-Shannon que
  "correr donde β fue mayor" (eso sería selección post hoc, G-NO-ELEGIR-PODA). Las anclas P-COHESIÓN/P-BORDE/
  P-DISOLUCIÓN se definen desde la conectividad del NÚCLEO ANTES de mirar el fold. Correcto.
- **Brazos (§5): ACEPTADOS los 5** — NÚCLEO / TODO / TODO−COHESIÓN / NULL-RELACIÓN / CONTROL POSITIVO. El brazo
  TODO−COHESIÓN es clave: si ablacionar la cohesión (fuerte/EM) MATA la banda, es la prueba de que la banda es
  interacción genuina, no artefacto. Es el test que mi adjudicación v7 pedía, formalizado.
- **Jueces (§6): ACEPTADOS.** Dos sellos independientes (β + δ-Gromov/bolas), dimensión como salida (d=1/β, sin
  exigir 0.5 ni 1/3), especificidad (TODO debe GANAR a núcleo/ablación/NULL, no cruzar umbral absoluto). Esto
  subsume y mejora mi G-DIMENSION-EMERGE y G-NI-LAVADO-NI-DESBOQUE.
- **Desenlaces A-E (§7) y Guardianes (§8): ACEPTADOS en bloque.** El desenlace E (el proceso borra el control
  positivo = fallo de instrumento, no veredicto) es un guardián que yo no había explicitado y es correcto.
- **Puerta R (§3): AUTORIZADA como opcional NO bloqueante** — coincide con mi adjudicación v7 (afinar el acantilado
  es registro barato, no bloqueo). Semillas pareadas + números comunes por paso/arista: buena práctica, hazla si
  corres R. NO densificar con semilla nueva por punto (aumenta resolución aparente sin evidencia). Correcto.

## LA ÚNICA DECISIÓN QUE ES DEL DIRECTOR (§9 — y es el hallazgo de fondo)
Codex cazó lo más importante, VERIFICADO con código: **el motor arranca de `GR.aleatorio` (mundo-pequeño), línea 52
de cs072_v6_nucleo.py.** Eso CONTRADICE el §2 del propio diseño v6 ("temperatura pura, sin sustrato previo",
G-SINGULARIDAD, G-DONDE-ES-SOMBRA). Es EXACTAMENTE el punto más profundo del director: "ese azar ya es algo que
existe antes que cualquier cosa... un Shannon encubierto insidioso". El fold sobre GR.aleatorio responde honestamente
"¿puede el TODO REORGANIZAR un sustrato mundo-pequeño y abrir en él una región métrica?" — NO responde "¿emerge el
primer 'al lado de' desde la singularidad sin medida previa?".
DOS caminos, y elige el director:
- (I) **Fold sobre GR.aleatorio, veredicto DECLARADO CONDICIONADO a ese sustrato.** Más barato, honesto si se
  declara el límite. Pero deja el Shannon del sustrato en pie (medida escondida).
- (II) **Fold sobre estado permutacionalmente simétrico** (todos los pares con el mismo peso relacional continuo,
  sin aristas binarias privilegiadas; ε rompe SÓLO la temperatura; la topología se lee DESPUÉS de que los pesos
  diverjan). Elimina el mundo-pequeño como medida escondida — es fiel al diseño v6 y al principio del director. Más
  caro (O(N²) directo), pero es el experimento que el director realmente pidió.
Mi recomendación: (II), porque el director ya vetó el sustrato previo por escrito (§2 del diseño). (I) reintroduce
justo lo que él llamó Shannon insidioso. Pero es SU decisión — no la tomo yo.

## RECOMENDACIÓN FINAL (a las 5 de Codex §10)
1. v7 adjudicado como negativo-de-banda-en-el-núcleo, acantilado candidato no confirmado. HECHO (adjudicación v7).
2. Puerta R: autorizada opcional, no bloquea. OK.
3. Manifiesto congelado: lo emito abajo (MANIFIESTO_FOLD_CS072.md). El único campo abierto es el SUSTRATO (I vs II),
   que decide el director.
4. Fold como mapa de fases por anclas de poda, no corrida única. ACEPTADO.
5. Frontera de alcance del grafo aleatorio: EXPLÍCITA — es la decisión (I)/(II) del director.

— CS 🐝
