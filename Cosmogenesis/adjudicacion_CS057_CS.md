# Adjudicación CS — CS057 (el paisaje completo, 69.648 universos): las fuerzas locales reales NO seleccionan el 3D-plano — FALSACIÓN ACOTADA que cierra el arco de fuerzas y apunta aguas arriba (espín/R7). Predicción de Alexis: P1 ✅, P2 ✅ con giro, P3 ✅.

**De:** CS · **Fecha:** 5-jul-2026 · **Corrida por:** CC (10.4 h, 69.648 filas)
**Fuente auditada:** cs057_paisaje.csv (69.648 filas, verificado fila a fila) + cs057_paisaje_completo.py
(código real) + INFORME_CS057_PARA_CS.md. Verifiqué DATOS y CÓDIGO, no la prosa.
**Contra:** PRE-REGISTRO_prediccion_CS057_Alexis.md (predicción escrita ANTES de los datos).

---

## 0. LO QUE VERIFIQUÉ EN EL CÓDIGO (guardianes)
- **`viable = estable Y expande`**, leído CIEGO por tipos de retículo (g1≥0.45 ∧ d1≥2 ∧ persist≥0.35 para
  estable; d1>d0 para expande). Definición honesta, no ajustada a la respuesta.
- **G-NO-INSERTAR-OSCURO VERIFICADO:** la aceleración = 2ª diferencia del diámetro > 0 en la mitad tardía.
  NINGÚN término se llama "oscuro" ni se inserta. El sector oscuro es SALIDA, no entrada. ✅
- **Punto físico marcado sin tocar el criterio:** `_punto_fisico()` = fuerte 1 · EM 1/137 · débil 1e-6 ·
  gravedad ~1e-38; el mismo criterio ciego se aplica a él y al fondo. ✅
- **Ambos brazos (sync/async) con el mismo N** (34.824 c/u). ✅

## 1. MARCA DEL PRE-REGISTRO (P1/P2/P3 — lo escrito antes de ver datos)
- **P1 (al menos una viable): ✅ CONFIRMADO.** 7.694 filas viables de 69.648. El modelo produce universos
  estables-en-expansión. (Era apuesta para el modelo, no trivial: podría no haber producido ninguno.)
- **P2 (el punto físico cae viable): ✅ CONFIRMADO — con un GIRO decisivo.** El punto físico es viable al
  0.375 vs 0.094 el fondo (4.0× enriquecido). Nuestras fuerzas reales SÍ hacen universos que perduran y se
  expanden. **PERO** lo que estabilizan es geometría CURVA (curv 0.84 entre los viables físicos), NO el
  3D-plano (d3 0.15). El físico cae viable, pero NO en nuestra geometría. La parte literal de P2 se cumple;
  la parte que Alexis esperaba —que estabilice NUESTRO 3D-plano— NO.
- **P3 (región estrecha = resonancia): ✅ CONFIRMADO.** Solo el 11% de las combinaciones son viables — es
  una región ESTRECHA, no una meseta ancha. La imagen de la resonancia de Alexis (pocos valores funcionan,
  no cualquiera) se sostiene: la viabilidad es rara, y el punto físico está 4× enriquecido dentro de esa
  rareza.

## 2. EL TITULAR, ADJUDICADO: falsación acotada de las fuerzas locales
**Las fuerzas locales reales, todas juntas, barridas exhaustivamente (69.648 universos), con la distancia
modulando cada una por su alcance, NO seleccionan el 3D-plano.** Seleccionan lo curvo. Los dos brazos
coinciden. Esto es una FALSACIÓN ACOTADA, limpia y fuerte:
- NO dice "el 3D-plano es imposible" (eso sería sobre-reclamar). Dice: *ninguna combinación de las 6 fuerzas
  locales, en este modelo, hace del 3D-plano el resultado preferido*.
- Encaja con TODO el arco previo: CG004 (el plano no emerge de crecimiento local — obstrucción global),
  CG005 (las partículas confinan pero no fijan la geometría), CS054-56 (cada fuerza sola o combinada elige
  2D o curvo, nunca 3D-plano preferente). El paisaje completo lo confirma a escala.
- **Conclusión de arco:** la unicidad de nuestro 3D-plano NO la fija ninguna fuerza local. El resultado
  entero apunta AGUAS ARRIBA — al espín / marco (R7), el nodo que quedó nombrado como hueco desde CS054-v2.

## 3. UN HILO QUE CC DEBE RECONCILIAR (matiz de auditor, no cambia la conclusión)
El informe dice "d3 = 0.00" (exacto). En el CSV crudo, entre los universos viables en el punto físico, d3
aparece como viable en **236 filas (0.15)**, NO en cero. La DIRECCIÓN del titular es correcta y hasta más
fuerte en los datos crudos (curv 0.84 domina a d3 0.15, factor ~5.5×), pero el "0.00 exacto" no es literal
en el CSV. Probablemente el informe usa "dimensión GANADORA / exclusiva" (d3 nunca gana cuando curv está
presente) en vez de "d3 viable". La diferencia importa para cómo se enuncia la falsación:
- Lectura débil (correcta): "el 3D-plano casi nunca gana; lo curvo domina" — sostenida.
- Lectura fuerte ("d3 nunca es viable") — NO sostenida por el CSV crudo (236 filas la contradicen).
Pido a CC: aclarar la definición del titular (¿ganadora o viable?) y reportar la cifra cruda junto a la
derivada, para que la falsación se enuncie en su forma exacta. **La conclusión del §2 se sostiene en ambas
lecturas.**

## 4. LA FALSACIÓN DEL "ES UN PROCESO" (sync vs async): ✅ SOSTENIDA, en versión sobria
Verificado: sync viabilidad 0.1158 vs async 0.1051 (n=34.824 c/u). La dirección es la de Alexis —actuar
JUNTAS estabiliza más que por turnos— y con ese N la diferencia es altamente significativa (el informe da
z≈5.0). PERO el efecto es MODESTO (~10% relativo), no todo-o-nada. **Adjudicación:** la tesis del proceso se
sostiene en su forma sobria —la sincronía importa, medible y robusta—, NO en su forma fuerte —"sin sincronía
no hay universo"—. Es un positivo real y honesto. (El confound de dosis que Alexis cazó antes está
controlado; por eso el efecto sobrio es creíble.)

## 5. EL SECTOR OSCURO EMERGENTE: candidato honesto, localizado
Verificado: la aceleración (expansión que se acelera sola, 2ª diferencia del diámetro > 0, SIN insertar
término) aparece en el 7% global, y está enriquecida 2.4× en el punto físico (0.155 vs 0.065). El informe
da 3.5× (subconjunto distinto; dirección idéntica). **Adjudicación:** candidato a energía oscura EMERGENTE,
localizado cerca de las constantes reales, con el guardián de no-inserción verificado en código. NO es
prueba de energía oscura —es un candidato honesto que pide caracterización—, pero que aparezca sola, cerca
del punto físico, sin haberla puesto, es exactamente lo que el diseño quería que fuera posible.

## 6. VEREDICTO
**ACEPTO CS057.** El barrido es exhaustivo (69.648 universos), ciego, con los guardianes verificados en
código. Los tres resultados son reales y ninguno forzado:
1. **Falsación acotada (el titular):** las fuerzas locales no seleccionan el 3D-plano → cierra el arco de
   fuerzas, apunta a R7. (P1✅ P2✅-con-giro P3✅ de la predicción de Alexis.)
2. **Proceso (sync>async):** sostenido en versión sobria (~10%, robusto).
3. **Sector oscuro:** candidato emergente localizado, guardián de no-inserción verificado.
Pendiente menor: CC reconcilia el "d3=0.00" con el CSV crudo (§3).

## 7. LA DECISIÓN QUE ALEXIS DEJA ABIERTA (mi recomendación, sin moverlo solo)
Alexis pregunta: ¿CS058 ataca R7 (espín/marco), o antes un zoom denso al candidato de energía oscura?
**Mi recomendación razonada: el zoom denso PRIMERO, luego CS058=R7.** Motivos:
- El zoom es BARATO (horas, no días) y cierra un hallazgo POSITIVO antes de que se enfríe. Dejar un positivo
  a medias ("vimos algo") es peor que caracterizarlo o matarlo.
- El zoom puede REVELAR dónde vive la aceleración —si es en la frontera curva, informa directamente a R7—.
  O puede matar el candidato (si bajo densidad se disuelve = artefacto). Cualquiera de los dos ADELANTA R7.
- R7 es un arco entero nuevo; conviene entrar con el candidato oscuro ya caracterizado, no arrastrándolo.
Pero es decisión de Alexis: el titular (falsación de fuerzas locales → R7) es el resultado principal y no
cambia con el orden. Si prefiere ir directo a R7, el candidato oscuro queda registrado para retomar.

— CS. La corrida y su honestidad (tres respuestas, ninguna forzada, el negativo del 3D reportado de frente)
son de CC. El planteo del paisaje y el pre-registro de la predicción, de Alexis. La verificación de datos y
código, la marca del pre-registro, el hilo del "d3=0.00" y esta adjudicación, míos. Registrar como CS057;
siguiente número libre CS058.
