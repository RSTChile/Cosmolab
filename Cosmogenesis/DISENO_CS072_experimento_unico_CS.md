# DISEÑO CS072 v6 — EL ORIGEN: TEMPERATURA + ASIMETRÍA + ENTROPÍA, y la VECINDAD como ROCE (no como parecido escalar).
## Diseño: CS (traducido de las imágenes del director) · Ejecuta: CC · PROTOCOLO CERRADO.
## Supersede v5. Integra todas las correcciones de la auditoría del 17-jul (director + Gemini + auditor + código de CS).

## 0. LA IMAGEN (director — es la fuente, no adorno)
Una niebla tibia lo llena todo, perfectamente pareja: la misma temperatura en cada parte = UNO (=1 normalizado),
entropía CERO (todo condensado, sin diferencia que disipar). La expansión (Big Bang) NO fue simétrica: una parte
—o muchas, da igual— quedó una fracción infinitesimal MÁS FRÍA (ε del 1, ε>0). Esa diferencia sigue siendo
temperatura, la misma sustancia, distinguida sólo por diferir del todo. El "ahí" nace CON la diferencia (no
preexiste). La niebla se expande, se enfría, y —por conservación (no hay reserva externa) + 2ª ley— la asimetría
fría no puede borrarse: persiste y se ahonda. De ese roce, si la Teoría acierta, nace el "al lado de", después las
zonas, después las direcciones. Del calor infinito a la muerte térmica: entropía = la medida de esa dispersión
irreversible (que en un sistema cerrado en expansión se manifiesta como enfriamiento).

## 1. QUÉ RESPONDE Y QUÉ NO
- **PREMISA (dada, no se explica):** hubo asimetría. Por qué —no se sabe, no se pregunta; sin ella no hay universo
  (es la razón de que la materia sobreviviera a la antimateria). Exigir que el experimento la EXPLIQUE = hornear.
- **PREGUNTA (falsable):** dado ε>0, ¿emerge espacio/dirección de la diferencia vía la relación I⟷E?

## 2. ESTADO INICIAL — temperatura, NO nodos, NO cuadrícula (G-SINGULARIDAD)
- Campo de temperatura: todas las parcelas = 1.0 salvo una fracción ε a (1−δ), δ→0⁺. Nada más: ni partículas ni
  color ni carga (eso condensa DESPUÉS si emerge). Al inicio "siempre es lo mismo: temperatura".
- **G-SINGULARIDAD:** el estado es UNO (homogéneo salvo ε). N (nº de parcelas) = sólo resolución del continuo, NO
  "nº de cosas"; el resultado debe ser invariante a N. Cuántas zonas distintas hay = RESULTADO.
- **G-DONDE-ES-SOMBRA:** las parcelas NO viven en grilla; la vecindad NO puede venir del índice del array.

## 3. LA VECINDAD — EL ROCE, no el parecido del escalar (corrección de Gemini + código de CS, la pieza clave de v6)
BUG FATAL de v5 (Gemini lo cazó, CS lo verificó): ordenar parcelas por VALOR de temperatura y conectar vecinas-en-
ese-orden FUERZA una cadena 1D (diám/N≈0.20 constante N=200→1600). Raíz: un ESCALAR vive en la recta ℝ; cualquier
"cercanía = proximidad de un solo número" es 1D por naturaleza. Y la dimensión TAMPOCO puede elegirse metiendo k
campos a mano (eso sería Shannon por el otro lado: elegir d=k).
**La salida está en la imagen del director:** la vecindad NO nace de que dos parcelas se PAREZCAN en temperatura,
sino de que se ROCEN — de la relación I⟷E entre ellas, que tiene más estructura que un escalar. Concretamente:
- Dos parcelas están en contacto si INTERCAMBIAN energía de forma MUTUA y ese intercambio PERSISTE (memoria de
  enlace, CS071: un roce transitado se refuerza; uno que se apaga, decae y se poda). El enlace es una RELACIÓN con
  historia, no una distancia entre dos números.
- El estado de cada parcela NO es sólo su T (escalar) — carga también su RELACIÓN I⟷E: con quién intercambia,
  cuánto, con qué persistencia (los dos polos de S=I⟷E: flujo de energía ⟷ registro de la diferencia). Eso es un
  vector/historia, no un número; por eso NO colapsa a una recta.
- **LA DIMENSIÓN SE MIDE, NO SE ELIGE (G-DIMENSION-EMERGE):** el nº de direcciones independientes de roce que una
  región sostiene = su dimensión local, LEÍDA del grafo (d_s, β, δ), nunca fijada. Si sale d=1, es un (B) honesto
  (la relación no bastó para más de una dimensión). Si sale d>1 y VARÍA con las condiciones, emergió. PROHIBIDO
  cualquier construcción que fije d por diseño (ni orden escalar → d=1, ni k campos → d=k).
- CC ELIGE la implementación concreta del roce, la DECLARA y la AUDITA contra: (a) no es un orden total sobre
  ningún escalar; (b) la dimensión es salida medida; (c) si la dimensión sale constante/forzada bajo variación de
  ε/N/expansión, es señal de Shannon oculto y la implementación es inválida. CS audita el código, no la promesa.

## 4. LA DINÁMICA — roce I⟷E (relación, NO producto) + los dos procesos que compiten
- **Intercambio I⟷E:** a lo largo de los enlaces de contacto, la energía fluye y con ella se acopla la
  información (cuál-difiere-de-cuál, con qué historia). S=I⟷E es RELACIÓN: el estado NO es un escalar "cantidad de
  S" por sitio (eso sería el producto I·E). Conserva la energía total (sistema cerrado, sin reserva; assert: ΣT
  sólo cae por enfriamiento).
- **Los DOS procesos que compiten (verificado con código por CS):**
  · DIFUSIÓN (roce que promedia) SIEMPRE suaviza → sola, LAVA la diferencia (ecuación del calor, es teorema).
  · INESTABILIDAD (gravedad: lo frío/denso atrae más) → sola, AMPLIFICA pero se DESBOCA (runaway sin cota,
    T diverge, = el colapso de CS064/CS067).
  El experimento REAL es el BALANCE de ambas contra la EXPANSIÓN que las frena: estructura métrica ACOTADA, ni
  lavado plano ni desboque a pozos. Esa competencia ES la formación de estructura del universo.
- **Enfriamiento por expansión:** monótono (nunca sube). La semilla es FRÍA (menor T): una caliente se disiparía,
  una fría se sostiene por el trinquete. Todo en el MISMO bucle por-paso (nada en secuencia).

## 3-bis. EL MOTOR IRREVERSIBLE — ENTROPÍA (necesario, no suficiente)
Entropía CERO en el origen (un solo microestado). La primera diferencia ε>0 abre el primer microestado accesible;
por la 2ª ley la entropía crece y no vuelve. Trinquete (cara energética): sin reserva externa no se puede
re-homogeneizar, y el todo enfría mientras tanto, así que la asimetría fría se ahonda. PERO —corrección de la
exploratoria v5— el trinquete impide la re-homogeneización EXTERNA; NO impide la difusión LOCAL (que lava sin
reserva). Por eso hace falta la mitad amplificante (§4). Esto SUBSUME el filtro de persistencia S=I·E de CS053:
la persistencia se DERIVA de la 2ª ley, no se impone como regla externa.
- Assert ΣT (conservación) + enfriamiento monótono + reportar entropía-proxy (CV=std/mean, invariante al
  reescalado uniforme) por paso. **G-ENTROPIA-MONOTONA:** la entropía-proxy no puede decrecer por un blanco fijo o
  reserva metida a mano; si decrece por eso, la implementación es inválida (Shannon).

## 4-bis. TODO LO ACUMULADO — LOS 18 ELEMENTOS COMPLETOS, PRESENTES COMO LEY DESDE t=0 (no como objeto inyectado)
Al inicio NO hay partículas — hay temperatura. Pero las fuerzas/propiedades del arco NO son objetos que falten:
son las LEYES según las cuales la temperatura se diferencia, activas desde el 1er paso, todas juntas. Los
PORTADORES (partículas con identidad) EMERGEN como diferencias que persisten al enfriarse. **NO se recorta NINGÚN
ingrediente del inventario validado del arco (18 elementos: 12 heredados + 5 de CS067 + inflación CS068, leídos del código, nada de memoria) — ése fue el error que este
experimento existe para no repetir.** Cada uno como ley sobre el grafo de ROCE (§3), NUNCA sobre índice/coordenada:
1. **Espín / marco nemático** (orientación por nodo) — cs052/cs059 (`_spins`).
2. **Gravedad ∝ peso-masa** (no grado) — cs054/cs062 (`_grav_peso`) — la inestabilidad amplificante de §4.
3. **Fuerte / confinamiento** (satura, no colapsa a agujero negro; por color) — cs056 (`_confin`).
4. **Electromagnetismo** (por carga) — cs056 (`_em`).
5. **Débil** (cambio de sabor) — cs056 (`_debil`).
6. **Catálogo completo de partículas** — cs060/cs064: quarks (color, 55%), leptones (18%), etc.
7. **Masa** (log-masa por partícula, leptones pesados incluidos) — cs060/cs061.
8. **Aniquilación materia-antimateria** — cs064; aquí vive la asimetría materia-antimateria por ε.
9. **Expansión / despliegue** — cs057/cs064 — AHORA como el operador de PODA de enlaces (§ poda), no sólo T.
10. **Enfriamiento como PROCESO** (T baja paso a paso, todo junto, no sucesión) — cs055/cs064/CS068.
11. **Vértice 3-cuerpos genuino** (update de marco irreducible, no pareado) — cs063.
12. **Localidad / geometrogénesis** (qué enlaces PERSISTEN al enfriar) — cs066 = el grafo de roce mismo de §3.
13. **Exclusión de Pauli ORTOGONALIZANTE** — cs065/065b. **FALSIFICADA ×2 en aislamiento, veredicto (C) "muere
    sin duelo".** Se incluye con su veredicto negativo; NO se presupone anti-colapso; si aportara DENTRO del todo
    co-emergente (no aislada), sería hallazgo nuevo a auditar aparte.
14. **Distancia por correlación** (Van Raamsdonk/Ryu-Takayanagi; w∈(0,1] continuo, no binario) — cs066.
15. **Estructura causal / cono de luz** (c = velocidad de la causalidad; tiempo de nacimiento t_i por nodo; CDT) — cs063.
16. **Ruptura espontánea de simetría multi-dimensional** (Higgs/Goldstone; el marco relaja bajo potencial) — cs067.
17. **Energía / materia oscura EMERGENTE** — NO dada, sino "probabilidad de algo cuando todas las fuerzas actúan
    juntas variando" (director, desde CS057). Guardián G-NO-INSERTAR-OSCURO: no se mete Λ a mano ni % objetivo.
18. **Análogo de inflación — estirar-y-enfriar / romper los atajos largos** (CS068) — el elemento 18 del arco.
    ES el operador de PODA de este diseño (§ poda): la expansión rompe enlaces para revelar/permitir tejido
    métrico. En CS068 aislado dio Mundo B (no había métrica latente bajo los atajos del blob de CS067); aquí
    entra como parte del TODO co-emergente, no sobre un blob ya formado. Poda CIEGA a la longitud (por grado o
    uniforme), NUNCA "enlaces largos" (eso leería la métrica = Shannon).
Propiedades (color/carga/masa/anti, #6-7): NO se asignan a posiciones; CONDENSAN cuando una diferencia persiste al
enfriarse. **Mecanismos NUEVOS post-CS067, también plegados:** Semilla (CS070 = la ε de §2, ahora en el origen);
Memoria de enlace (CS071 = la persistencia del roce de §3); Fase cuántica (CS069): FUERA salvo que CC la acople
sin grilla — su ausencia se DECLARA, no se omite en silencio.
**Total: 18 elementos del arco (12 heredados CS052-066 + 5 de CS067 + inflación/poda CS068) + 3 mecanismos nuevos
(semilla CS070, memoria CS071, [cuántico CS069 si acopla]).** Si CC no logra
acoplar alguno limpio al grafo de roce, lo REPORTA a CS ANTES de la tanda (como el §69 cuántico) — nunca se recorta
en silencio ni se aplaza a "otro número CS".

## 5. MEDICIÓN — sin coordenadas
d_s (crecimiento de bola sobre el grafo de contacto), δ de Gromov (plano vs mundo-pequeño), β (escalamiento del
diámetro: √N métrico vs log N mundo-pequeño), dirección emergente (n_ejes por inercia + holonomía). SIEMPRE sobre
el grafo leído del roce, jamás sobre el índice del array.

## 6. LOS TRES BARRIDOS QUE BLINDAN EL ANTI-SHANNON (parámetros de realidad, no perillas — parte del veredicto)
Ninguno se elige; los tres se BARREN y se lee cuál permite un universo:
- **ε (tamaño de la semilla):** {1e-6, 1e-4, 1e-2}. El resultado debe ser INVARIANTE a ε. Cuanto menor δ con
  resultado positivo, más fuerte (una millonésima no CONTIENE la dirección, sólo la desencadena).
- **Nº de focos (una o muchas):** {1, pocos, muchos}. El resultado debe ser invariante al número (director: "una o
  muchas es irrelevante"). Sólo importa que ε>0.
- **Tasa de expansión/enfriamiento (parámetro de realidad — director):** se BARRE, y se MIDE a qué tasa la primera
  diferencia se sostiene y condensa estructura. El umbral de condensación es SALIDA medida, NUNCA entrada. NINGUNA
  escala física (Hagedorn u otra) se escribe en el código; si aparece, se LEE del resultado. Es cómo el universo
  se dio a sí mismo su propia escala.
Si el resultado DEPENDE de ε o del nº de focos → se metió información por ahí, hallazgo positivo anulado.

## 7. BRAZOS
- **TODO (real):** campo temperatura + ε + roce I⟷E con memoria + inestabilidad + expansión + los 18 elementos del arco activos (+ 3 mecanismos nuevos).
- **NULL_BARAJADO (único):** mismas temperaturas, pero se BARAJAN las relaciones de roce (rompe la correlación
  qué-roza-con-qué, conserva la estadística de T). TODO vs NULL ES la tesis: si la emergencia vive en la relación,
  se separan; si vive en la mera estadística, no.
- **CONTROL POSITIVO (como CS071):** el mismo proceso sobre un campo con métrica GENUINA sembrada. Debe dar β≈0.5
  — valida que el juez detecta metricidad y que el proceso no la destruye. Sin él, un β bajo real sería ambiguo.

## 8. G-NI-LAVADO-NI-DESBOQUE — los tres destinos, los tres honestos
Reportar CV + β + δ por paso en todos los brazos:
- CV→0 (lava): difusión gana, las leyes no bastan. (B).
- CV→∞ / colapso a pozos: gravedad se desboca (runaway, T diverge). (B) tipo CS064.
- CV crece y se ESTABILIZA en régimen intermedio CON métrica (β→0.5, δ no plano, d>1 emergente): (A) — estructura
  acotada. Sólo éste es (A). El balance NO se sintoniza a mano (Shannon): parámetros heredados de cs062/CS068; si
  ninguno cae en el régimen, es (B).

## 9. PROTOCOLO CERRADO
EXPLORATORIA obligatoria primero: CC verifica que (a) el contacto-por-roce NO fuerza dimensión (barre ε/N/expansión
y comprueba que d NO es constante forzada), (b) no colapsa ni lava trivialmente, (c) NULL y control positivo se
comportan. Reporta a CS. Sólo tras visto bueno se corre la TANDA. Cambiar un juez/umbral/ingrediente = OTRO número
CS. Un experimento = un protocolo = un número, para siempre.

## 10. EN UNA LÍNEA
CS072 arranca donde la Teoría dice que arrancó el universo: temperatura uniforme (=1, entropía 0) con una
diferencia infinitesimal fría (ε>0, una o muchas), sin cuadrícula, sin N dado, sin sustrato previo; la VECINDAD
nace del ROCE I⟷E (relación con historia, no parecido de un escalar — por eso la dimensión EMERGE y no queda
forzada a 1); la dinámica es difusión (suaviza) vs inestabilidad (amplifica) contra la expansión (frena), con la
entropía como motor irreversible; los 18 elementos del arco activos desde t=0 (Pauli incluida pero falsada); y se
pregunta si de ahí emerge espacio y dirección — con tres barridos (ε, nº de focos, tasa de expansión) que
garantizan que, si emerge, emergió de la relación y no de lo que metimos. Es la habitación entera encendida a la
vez, desde el uno.

— CS 🐝
