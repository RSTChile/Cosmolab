# ADJUDICACIÓN CS — CS072 v5 exploratoria del núcleo. VEREDICTO (corregido): NO plegar aún — el contacto por escalar fuerza 1D (Gemini), primero se arregla la vecindad. [La opción (a) que este documento proponía queda RETRACTADA — ver §VEREDICTO CORREGIDO.]
## CS, 17-jul-2026. Sobre INFORME_CS072_v5_exploratoria_nucleo_PARA_CS.md + cs072_v5_nucleo.py. Auditado y verificado con toy propio.

## LO QUE CC HIZO BIEN (sin reservas)
- Resolvió el núcleo más difícil del diseño: contacto SIN índice de array (ordena por VALOR de temperatura,
  conecta vecinas-en-valor). G-DONDE-ES-SOMBRA respetado — verifiqué el código: `contacto_por_temperatura`
  usa argsort(T), reconstruye el grafo cada paso desde T, jamás toca el índice. Correcto.
- Conservación verificada (ΣT cae sólo por enfriamiento, 0.97^30≈0.40). Restricción #1 respetada al pie.
- Eligió CV (no varianza cruda) como proxy — correcto: CV es invariante al reescalado uniforme del
  enfriamiento, así que sólo se mueve por diferenciación REAL. Buen juicio anti-artefacto.
- NO ajustó nada para que el CV creciera. Reportó el núcleo desnudo tal como salió. Eso es exactamente la
  honestidad que G-ENTROPIA-MONOTONA exige. El lavado reportado es un resultado válido, no un fracaso oculto.

## EL DIAGNÓSTICO DE CC ES CORRECTO — y lo confirmé con código, no de memoria
La difusión pura SIEMPRE suaviza (ecuación del calor: la heterogeneidad decae monótonamente hacia 0). No es un
bug: es un teorema. El trinquete del §3-bis responde a UNA amenaza (reinyección EXTERNA — sin reserva no se
puede rellenar) y el núcleo la respeta. Pero CC expuso una SEGUNDA amenaza que el trinquete no cubría: la
difusión LOCAL borra diferencias sin reserva alguna, y el enfriamiento global uniforme ni ayuda ni estorba
(escala todo por igual). CC tiene toda la razón. El trinquete de conservación es necesario pero NO suficiente.

## LO QUE FALTABA — media física, no un parámetro (toy verificado, N=500, 40 pasos)
El intercambio del núcleo es difusión = la mitad SUAVIZANTE de la física. Falta la mitad AMPLIFICANTE: la
INESTABILIDAD (gravedad), donde una diferencia atrae más diferencia en vez de promediarla. Mismo núcleo, mismo
contacto, sólo cambiando el signo del acoplamiento (de promediar a acretar, conservando suma):
| ley | CV[0] | CV[40] | destino |
| difusión (núcleo de CC) | 1.96e-4 | 1.52e-4 | SE LAVA |
| gravedad (inestabilidad) | 1.96e-4 | 1.5e+2 | CRECE |
Esto NO es hornear: la inestabilidad gravitatoria (las sobredensidades crecen) es una ley real, YA en el
repertorio del arco (cs062 `_grav_peso`), que el diseño v5 §4 manda activar desde t=0. El núcleo simplemente
la dejó fuera y sólo puso el intercambio difusivo. La difusión es la ecuación del calor; la gravedad es la
inestabilidad de Jeans. Ambas son física estándar; el núcleo tenía una y le faltaba la otra.

## EL PELIGRO, TAMBIÉN VERIFICADO — la gravedad sola SE DESBOCA (= el colapso de CS064 en forma de temperatura)
Gravedad sola a 60 pasos: T ∈ [−214584, +1.86e6]; ~24% de parcelas caen por debajo de mediana/2, y de ésas más
de la mitad DIVERGEN a negativos grandes (hasta ~−214584), no convergen a cero — es runaway sin cota, no colapso
a un punto fijo. Amplifica sin freno = degeneración tipo hub/desboque, el fracaso de CS064/CS067. Por eso la respuesta NO es "activa
gravedad y listo": es que el experimento REAL de CS072 es el BALANCE — inestabilidad (ahonda) vs expansión
(frena) — buscando estructura ACOTADA con métrica, ni lavado plano (difusión gana) ni colapso a un pozo
(gravedad gana). Esa competencia ES la historia de formación de estructura del universo (δ crece, pero acotado
por el arrastre de Hubble). El régimen intermedio es donde puede vivir un espacio métrico.

## VEREDICTO — CORREGIDO (Gemini cazó un bug fatal que yo NO vi; retracto la opción (a))
ADVERTENCIA: mi primer veredicto decía "opción (a), plegar las leyes sobre el núcleo tal cual". ERROR MÍO.
Gemini auditó lo que yo pasé por alto: el mecanismo de contacto de CC (ordenar por VALOR de T y conectar
vecinas-en-ese-orden) **fuerza una cadena 1D por diseño** — un contrabando de Shannon topológico. Yo verifiqué que
el contacto no usa el índice del ARRAY (correcto) pero NO verifiqué qué DIMENSIÓN produce. La produce, y es d=1.

**Verificado con código (N=200→1600):** diám/N ≈ 0.20 CONSTANTE = cadena 1D (diámetro ∝ N). No es mundo-pequeño
ni 2D: es una recta. Plegar las leyes sobre esto sería medir "qué hacen las fuerzas a una cadena 1D" = el error
del arco entero un nivel más abajo. Gemini tiene razón; **la opción (a) queda RETRACTADA.**

**Raíz del problema (verificada, más profunda que "usar otra fórmula"):** un ESCALAR (T) es 1-dimensional por
naturaleza — vive en la recta ℝ. CUALQUIER regla "contacto = proximidad de valor de un solo escalar" da 1D, y
reconstruir el grafo cada paso desde T no lo salva (siempre es el mismo escalar). Toy: contacto por 1 escalar →
diám/N=0.21 (1D); contacto por 2 grados de libertad relacionales → diám/N=0.03 (~√N, 2D). **La dimensión >1 sólo
puede venir de que la RELACIÓN tenga más grados de libertad que un escalar.** Aquí Gemini apunta bien ("matriz de
acoplamiento, no ordenamiento de un escalar") pero su arreglo es VAGO y peligroso: si yo simplemente AÑADO un
segundo campo (p.ej. dT/dt) para tener 2 grados, estoy ELIGIENDO dimensión 2 = Shannon por el otro lado. La
dimensión no puede elegirse metiendo k campos; el NÚMERO de grados de libertad independientes debe EMERGER.

**Dónde esto toca la Teoría (y por qué NO lo decido solo — decisión del director):** el director fue explícito:
al inicio SÓLO temperatura, un escalar. Pero S = I⟷E NO es un escalar — es una RELACIÓN con dos polos
(flujo de energía ⟷ registro de la diferencia). La salida al 1D probablemente ya está en su propia fórmula: la
"cercanía" no debe leerse del VALOR de T (escalar, 1D), sino de la RELACIÓN I⟷E entre parcelas (qué intercambia
con qué, y con qué historia) — que carga más estructura que un número. Pero CÓMO cuaja eso, y de dónde emerge el
número de dimensiones en vez de elegirse, es pregunta de Teoría. NO la resuelvo unilateralmente (es justo el error
por el que me corrigieron en este arco).

## PRECISIÓN DEL DIRECTOR (17-jul) — el umbral como SALIDA es parámetro de realidad, NO Shannon.
Corrijo un exceso mío: marqué como Shannon la idea de un umbral de condensación (Hagedorn ~150 MeV). El director
distingue con razón dos cosas opuestas:
- **Shannon (prohibido):** meter el umbral como ENTRADA — "si T<150 MeV enciende confinamiento" — para que el
  motor condense donde sabemos que debe. Dato impuesto. Esto sigue prohibido.
- **Parámetro de realidad (legítimo, y es la pregunta correcta):** BARRER la expansión y MEDIR cuánta expansión
  de la singularidad hace que al menos un área baje su temperatura lo suficiente para que ocurra lo que ocurrió
  (condensación de estructura). El umbral es SALIDA del barrido, no entrada. Se conoce porque "es lo que existe"
  — se contrasta contra la realidad, no se gira para que salga 3D. Misma lógica que el barrido fuerzas-0-a-1 y
  que el barrido de ε ya en el diseño: no se elige el valor, se barre y se ve cuál permite un universo.
Más fuerte aún (y es la tesis del director, no una defensa): la escala concreta (temperatura de condensación,
tamaño de las partículas) NO está definida en la singularidad — no hay "150 MeV" porque no hay partículas ni
escala. La escala NACE cuando la expansión enfría lo bastante para que algo cuaje. Medirla como salida del barrido
es medir cómo el universo se dio a sí mismo su propia escala. Eso es el hallazgo, no contrabando.
→ Diseño: añadir al barrido (junto a ε y nº de focos) un BARRIDO DE LA TASA DE EXPANSIÓN/ENFRIAMIENTO, y reportar
a qué tasa la primera diferencia se sostiene y condensa estructura. El umbral resultante es dato medido. Ninguna
escala física (Hagedorn u otra) se escribe en el código como entrada; si aparece, se LEE del resultado.

## VEREDICTO OPERATIVO
1. **NO plegar las leyes todavía.** El núcleo tiene un bug de dimensionalidad que las leyes NO arreglarían
   (seguirían corriendo sobre una cadena 1D). Primero se arregla el contacto.
2. **El contacto NO puede derivar del orden de un escalar.** Debe derivar de la relación I⟷E, y el número de
   dimensiones debe EMERGER, no elegirse metiendo k campos. Esta es decisión de Teoría — la lleva el director.
3. Gemini acertó en el diagnóstico (1D forzado) y en el espíritu del arreglo (relación, no ordenamiento escalar);
   se rechaza sólo la versión que metería dimensión a mano. CC NO codea el arreglo hasta que el director defina de
   dónde salen los grados de libertad relacionales.

## GUARDIÁN NUEVO (obligatorio, por el desboque verificado)
**G-NI-LAVADO-NI-DESBOQUE:** la tanda debe reportar el CV (entropía-proxy) por paso en TODOS los brazos. Tres
destinos posibles, los tres son resultados honestos:
- CV→0 (lava): la difusión gana; las leyes no bastan para sostener la diferencia. (B) para el todo.
- CV→∞ / colapso a pocos pozos: la gravedad se desboca; NO hay métrica, hay hub. (B) tipo CS064.
- CV crece y se ESTABILIZA en un régimen intermedio CON métrica (β→0.5, δ no plano): (A) — estructura acotada.
Sólo el tercero es (A). Y el balance NO se sintoniza a mano para caer ahí (sería Shannon): los parámetros de
gravedad/expansión son los YA calibrados en cs062/CS068; si ninguna combinación heredada cae en el régimen
intermedio, se REPORTA como (B), no se ajusta hasta que salga.

## PARA CC
1. Pliega las 10 leyes (gravedad cs062, EM/confinamiento cs057, débil+aniquilación cs064, marco cs059,
   propiedades, CS068 enfriamiento-inflación, CS070 semilla=ε, CS071 memoria) sobre el núcleo v5, activas desde
   t=0, TODAS sobre el grafo de contacto (G-DONDE-ES-SOMBRA rige para todas). Pauli: incluida con su veredicto
   negativo, sin presuponer anti-colapso.
2. Exploratoria de nuevo ANTES de la tanda: reporta CV por paso + β + δ-Gromov, con G-NI-LAVADO-NI-DESBOQUE.
   Verifica que el brazo real no lava NI se desboca antes de correr el veredicto. Si se desboca, es señal de que
   falta el freno de expansión bien acoplado — repórtalo, no lo ajustes.
3. Usa los parámetros heredados (cs062/CS068), no nuevos. Cambiar uno = otro número CS.

## EN UNA LÍNEA
El núcleo difusivo se lava porque la difusión pura es sólo la mitad suavizante de la física (teorema, no bug, y
CC hizo bien en reportarlo sin tocarlo); la mitad que falta —la inestabilidad gravitatoria, ya en el repertorio—
la revierte (verificado: CV crece), pero sola se desboca a un pozo (verificado: el colapso de CS064). CS072 es el
BALANCE entre ambas contra la expansión: si de esa competencia emerge estructura métrica acotada, es (A); si lava
o se desboca, es (B). PERO antes de nada: el contacto por orden de un escalar fuerza 1D (Gemini, verificado por
CS con código, diám/N≈0.20 constante) → la opción (a) queda RETRACTADA. NO se pliegan las leyes hasta arreglar la
VECINDAD: debe nacer del ROCE I⟷E (relación con historia), no del parecido de un escalar, para que la dimensión
EMERJA en vez de quedar forzada a 1. Esto pasa al diseño CS072 v6.

— CS 🐝
