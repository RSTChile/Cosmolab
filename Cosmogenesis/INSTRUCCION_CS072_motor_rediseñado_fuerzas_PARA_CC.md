# INSTRUCCIÓN CS072 PARA CC — MOTOR REDISEÑADO: las FUERZAS ligan, no el campo térmico (CS ya lo probó)
## CS reescribió el motor DESDE CERO para eliminar el artefacto que invalidó la corrida anterior, y lo VERIFICÓ
## corriendo el test decisivo. CC ejecuta a escala; el diseño está cerrado y probado. Archivo: cs072_motor_fuerzas.py

## POR QUÉ HUBO QUE REDISEÑAR (la raíz del artefacto, verificada por CS)
El motor viejo (cs072_fold_completo.py) usaba UNA matriz W que mezclaba: (a) historia térmica DOMINANTE
(0.9*W acumulado cada paso) y (b) aportes pequeños de las fuerzas (dW_confin, dW_em...). El contador leía W total.
Como lo térmico dominaba, apagar el confinamiento NO cambiaba el conteo -> los "bariones" se formaban por cercanía
térmica, no por la fuerza fuerte. PRUEBA (CS, motor viejo, 600 pasos): apagar confinamiento+EM+gravedad+QCD+Pauli
seguía dando 9 bariones. El director lo diagnosticó por lógica: si no depende de las fuerzas, no es materia. Correcto.

## EL REDISEÑO (cs072_motor_fuerzas.py — DOS matrices separadas)
- T (campo térmico): SÓLO la condición inicial (gradiente + expansión). Crea la ASIMETRÍA que deja sobrevivir
  materia. La expansión enfría T. NUNCA liga. El contador NO la lee.
- B (matriz de LIGADURA): empieza en CERO. SÓLO las fuerzas la construyen (confinamiento, EM, gravedad), cada una
  con su regla física (color distinto, carga opuesta, masa). El campo térmico sólo HABILITA el confinamiento
  (actúa cuando el universo se enfría, T_ef < T_CONF), pero NO aporta a B.
- Contador: lee B (ligadura por fuerzas). Barión = 3 quarks color distinto mismo estatus LIGADOS EN B.

## LO QUE CS YA VERIFICÓ (test decisivo, reproducible)
Cuatro brazos (300 pasos): A homog=0, B homog+exp=0, C grad=0, D grad+exp=3. Contraste correcto.
PRUEBA DE ADMISIBILIDAD (brazo D, apagando fuerzas):
  con TODO                          : 3 bariones
  sin confinamiento                 : 0   <- LA FUERZA FUERTE LIGA. Sin ella, CERO materia. (viejo daba 9 = artefacto)
R_STRONG=0 -> 0 bariones. La materia DEPENDE de la fuerza fuerte. Determinista, estable a 300 y 600 pasos.

ANIQUILACIÓN SIN TASA (Motor B, constraint del director): NO es porcentual. Es RESTA de poblaciones -- por clase,
min(materia, antimateria) se aniquila (va a luz), sobrevive el EXCEDENTE. Verificado: 30 quarks - 21 antiquarks =
9 quarks sobreviven (0 antiquarks), sin ritmo/cupo/porcentaje. De esos 9 salen ~3 bariones (9/3, estequiometría
real del excedente). El conteo es más chico que el viejo (que inflaba por una tasa mal puesta) pero es el correcto.

CATÁLOGO (anti-Shannon): color/carga son COMPOSICIÓN intrínseca (tercios iguales de color, mitad up/down), NO
ruptura de simetría por índice. INVARIANCIA A PERMUTACIÓN VERIFICADA POR CS corriendo el test real (base vs 6
permutaciones del catálogo): base=3, permutaciones=[3,3,3,3,3,3], INVARIANTE=True (está en __main__ del motor,
reproducible con `python cs072_motor_fuerzas.py`). IMPORTANTE: en una versión previa el test FALLABA (3 vs
2,1,2...) porque la aniquilación mataba los primeros k por índice (mat[:k]) -> el orden decidía qué colores
sobrevivían -> Shannon. CORREGIDO: la aniquilación ahora resta POR COLOR (min materia/antimateria de cada color),
invariante al orden. Si CC modifica el motor y el test de invariancia vuelve a fallar, es Shannon reintroducido.

## CRITERIO DE ADMISIBILIDAD (lo que CC debe confirmar a escala antes de cualquier veredicto)
Apagar confinamiento DEBE llevar bariones a ~0 a TODA escala (N=68, 136, 272, 544). Si a alguna escala apagar
confinamiento NO baja los bariones, el artefacto reapareció y la corrida NO es admisible. Es la prueba de que las
fuerzas —no el campo térmico— hacen la materia.

## TAREAS PARA CC
1. Correr el test decisivo (apagar confinamiento -> 0) a N=68/136/272/544, pasos equilibrados (300+ ; subir si
   el conteo aún cambia entre 300 y 600). Confirmar 0 bariones sin confinamiento en TODAS las escalas.
2. Barrido de escala CON confinamiento: ¿los bariones crecen con N de forma física (≈ N_quarks/3)?
3. Auditoría de apagado de TODAS las fuerzas/piezas: cuáles cambian el conteo (actúan) y cuáles no. A diferencia
   del viejo, ahora apagar confinamiento DEBE cambiarlo. Reportar cuáles siguen sin actuar (reserva A real).
4. NO reintroducir la mezcla térmica-en-la-ligadura. B sólo la llenan las fuerzas. Es la corrección central.

## GUARDIANES
- CERO AZAR (no np.random en ningún punto). Constantes = sólo físicas (R_STRONG/R_EM/R_GRAV/T_CONF), no perillas.
- El campo térmico NO aporta a B. Si CC lo reconecta "para que dé más bariones", es volver al artefacto. Prohibido.
- Admisibilidad = apagar la fuerza que forma una estructura DEBE destruir esa estructura. Si no, no es física.

## ENTREGA: tabla (N × apagar-confinamiento) mostrando 0 sin confinamiento a toda escala; barrido N con bariones≈N/3;
## auditoría de qué fuerzas actúan; log y json. Con esto CS decide si el motor rediseñado es admisible.
— CS 🐝 (motor reescrito y test decisivo verificado con código; CC ejecuta a escala)
