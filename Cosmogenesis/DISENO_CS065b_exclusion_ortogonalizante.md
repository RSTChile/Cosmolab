# DISEÑO CS065b — Exclusión ORTOGONALIZANTE (Pauli fiel): ¿sostiene varios ejes, o el negativo se cierra?
## CS065b — corrige la FIDELIDAD del mecanismo de exclusión de CS065 (de repulsión lineal → ortogonalización saturante tipo Gram-Schmidt). PRE-REGISTRADO: las salidas se fijan ANTES de correr; el negativo se acepta igual que el positivo.

**Diseña:** CS · **Fecha:** 10-jul-2026 · **A codear/ejecutar:** CC · **Endosado por:** Alexis (con la condición explícita de que quede registrado sea cual sea el resultado)

---

## 0. POR QUÉ ESTE DOCUMENTO EXISTE (la honestidad, escrita antes del resultado)
CS065 (repulsión lineal) dio NEGATIVO robusto en tres escalas: la exclusión implementada como resta ilimitada del
alineamiento DESTRUYE dirección (n_ejes 0.56-0.77 vs 1.23-1.35 de sin_excl), y lo hace por IGUAL sobre fermiones
reales o pares al azar (excl ≈ excl_barajada ≈ 0.6). **Eso quedó falsificado y NO se re-litiga:** una repulsión
de orientación SIN FRENO no sostiene ejes — empuja a isotropía (desorden), no a ortogonalidad (estructura).

CS065b NO es "re-correr esperando que ahora sí salga". Es un test DISTINTO de una afirmación DISTINTA, y la
razón por la que se corre fue diagnosticada ANTES de tener el veredicto completo de CS065:
- Cuando CC trajo el PRIMER vistazo parcial de CS065, CS ya dictaminó que la repulsión lineal era "la traducción
  equivocada" de Pauli y que la ortogonalización (Gram-Schmidt) era "la traducción correcta". Eso fue ANTES del
  cuadro completo de los tres N.
- El negativo de CS065, cuando llegó, CONFIRMÓ el diagnóstico: excl ≈ excl_barajada es la firma PREDICHA de
  "repulsión genérica, no exclusión estructurada". El negativo no nos empujó a cambiar de mecanismo — VALIDÓ la
  razón por la que ya íbamos a cambiarlo.
**Registro para el acta:** cambiar un mecanismo tras un negativo es terreno peligroso. Lo que lo hace legítimo
aquí, y sólo aquí, es que (a) la corrección es de FIDELIDAD FÍSICA, no de resultado; (b) fue diagnosticada antes
del veredicto; (c) el negativo de CS065b se acepta igual que un positivo (ver §5). Si CS065b también da negativo,
la hipótesis de la exclusión MUERE — no se reencarna en un CS065c. Este es EL test de la exclusión bien hecha.

## 1. LA CORRECCIÓN DE FIDELIDAD (qué cambia, y por qué es física y no perilla)
Pauli no dice "aléjate infinitamente". Dice "dos fermiones no ocupan el MISMO estado" — pueden convivir en
estados ORTOGONALES adyacentes, solo no idénticos. El contenido real del principio es antisimetría →
ortogonalidad de los estados ocupados. La operación que "hace vectores mutuamente ortogonales y AHÍ PARA" es
Gram-Schmidt. Por tanto:
- **CS065 (mal):** s_i ← s_i − λ·Σ(vecinos) — resta ilimitada, empuja sin freno más allá de ortogonal → desorden.
- **CS065b (fiel):** cuando un fermión i tiene vecinos fermiónicos, su orientación se ORTOGONALIZA respecto a las
  de esos vecinos (proyecta fuera la componente paralela a cada vecino ocupado, tipo Gram-Schmidt) y **se
  detiene al alcanzar ortogonalidad** (producto interno = 0). No sigue empujando. No hay λ que crezca sin límite.
- La saturación NO es un parámetro elegido: el punto de frenado es la ortogonalidad misma (⟨s_i,s_j⟩=0), que es
  física, no calibración. **G-NO-CALIBRAR.**
- Bosones: sin exclusión (pueden compartir estado) — igual que CS065, fiel a la física.

## 2. QUÉ NO CAMBIA (todo lo demás idéntico — comparabilidad)
Mismo motor de CS064/CS065, mismos ingredientes, mismas 4 fuerzas mediadas, misma expansión/enfriamiento, misma
medición (n_ejes por espectro del tensor de orientación, d_s espectral, δ/Gromov, holonomía). Mismos N∈{1500,
2500,3500}. Misma estadística (~100 parches por N×brazo). SOLO cambia la regla de actualización de la orientación
de los fermiones (lineal → ortogonalizante).

## 3. LOS BRAZOS (idénticos a CS065 — para comparación directa brazo a brazo)
- **excl_orto (real):** exclusión ortogonalizante/saturante entre fermiones. EL brazo nuevo.
- **sin_excl (=CS064):** control central, sin exclusión. Debe reproducir ~1.3 ejes (G-CONTINUIDAD).
- **excl_orto_barajada:** ortogonalización aplicada a pares AL AZAR (no fermión-vecino real). Si excl_orto ≈
  barajada OTRA VEZ, entonces ni siquiera la ortogonalización es específica → la exclusión no es el mecanismo,
  punto final. Esta es la cuerda decisiva: en CS065 lineal, real≈barajada mató la especificidad; si en la
  versión fiel se SEPARAN, la especificidad de Pauli aparece; si NO se separan, la hipótesis muere limpia.
- **excl_orto_bosones (placebo):** ortogonalización sobre bosones (físicamente falso). Debe no ayudar.
- **marco_congelado:** ancla, debe dar 0 (G-CONTINUIDAD).

## 4. EL DISCRIMINANTE INTERNO (la predicción que el dato puede matar)
corr(n_ejes, fracción_de_fermiones): si la ortogonalización es el mecanismo, MÁS fermiones ⇒ MÁS ejes ortogonales
sostenidos (más estados que exigen ser mutuamente ⊥). En CS065 lineal esta correlación fue ≈0 (murió). Si en
CS065b tampoco aparece, es otra señal de negativo. Si aparece POSITIVA y robusta en los tres N, es señal fuerte
de mecanismo real.

## 5. SALIDAS PRE-INSCRITAS (escritas ANTES de correr — se leen contra esto, NO se acomodan)
- **(A) LA EXCLUSIÓN FIEL ABRE EJES:** excl_orto sostiene claramente MÁS ejes que sin_excl (p.ej. ≥2 vs 1.3), Y
  excl_orto > excl_orto_barajada (¡especificidad!), Y excl_orto_bosones ≈ sin_excl, Y corr(n_ejes,frac_ferm)>0.
  ⇒ el ingrediente anti-colapso ERA la exclusión, y estaba mal implementada; la dimensión múltiple emerge de la
  tensión alinear↔ortogonalizar. Predicción fuerte ganada con física real, sin calibrar. (Si además ronda 3 y se
  aplana ahí sin que D_max lo imponga —ver G-NO-TOPADO— es enorme; pero no se espera ni se fuerza.)
- **(B) ABRE EJES PERO NO 3:** excl_orto sostiene 2+ ejes robustos y con especificidad, pero no fija el número en
  3. ⇒ la exclusión SÍ rompe el colapso B', pero no basta para la dimensionalidad concreta — reorienta a qué más.
- **(C) NEGATIVO — LA EXCLUSIÓN MUERE:** excl_orto ≈ sin_excl (no ayuda) O excl_orto ≈ excl_orto_barajada (no
  específica) O destruye como la lineal. ⇒ **la repulsión/exclusión, en NINGUNA forma fiel, genera dimensión
  múltiple.** B' de CS064 se sostiene como definitivo en el régimen accesible; el ingrediente anti-colapso, si
  existe, NO es "excluir". Se cierra la hipótesis de la exclusión y el frente pasa al sector oscuro (CS066) o a
  aceptar que la inercia relacional sola no basta. **Este desenlace se acepta sin duelo — es tan limpio como (A).**
- **(D) DEPENDE DE N:** el efecto crece con N. ⇒ confirma la grieta N-dependiente de CS064 y que el régimen de
  números enormes es donde vive la respuesta.

## 6. GUARDIANES
- **G-NO-CALIBRAR:** el frenado es la ortogonalidad (⟨s_i,s_j⟩=0), NUNCA un ángulo/λ elegido mirando el resultado.
- **G-NO-TOPADO (nuevo, crítico):** Gram-Schmidt en un espacio interno de dimensión D_max produce a lo sumo D_max
  vectores ortogonales, y el nº de vecinos también acota. Si excl_orto da "3 ejes", HAY QUE PROBAR que no es
  porque D_max=3 o el grado medio lo topan ahí. Se corre con D_max holgado (≥8) y se verifica que el nº de ejes
  emergente es < D_max (si pega el techo, el resultado es artefacto del andamiaje, no física). SIN esto, un (A)
  no vale.
- **G-CONTINUIDAD:** sin_excl reproduce CS064 (~1.3); marco_congelado→0. Si no, el motor cambió — abortar.
- **G-PLACEBO:** excl_orto_barajada y excl_orto_bosones existen para que "ortogonalizar algo" no se confunda con
  "exclusión de Pauli". La separación excl_orto vs barajada es la prueba de especificidad — sin ella, no hay (A).
- **G-SMOKE-ANTES:** smoke (N=1000, ~10 parches, 5 brazos) validando que (i) sin_excl da ~1.3, (ii) la
  ortogonalización corre sin romper el grafo, (iii) el calibrador de n_ejes sigue dando 3→3, ruido→0. No correr
  la tanda grande hasta que el smoke pase y CS lo adjudique.

## 7. LO QUE NO HACE / LÍMITES
- No prueba que la dimensión "sea" 3 — prueba si la exclusión FIEL rompe el colapso-a-1 y con qué especificidad.
  El éxito es que el test DISCRIMINE (excl_orto vs sus controles), no que salga 3.
- Es un análogo relacional de Pauli (ortogonalización de orientación-estado entre fermiones vecinos), no la
  mecánica cuántica completa. Fidelidad en la ESTRUCTURA (antisimetría→ortogonalidad, saturante), no en Dirac.
- Un actor por vez: NO se mezcla con el sector oscuro (CS066) hasta cerrar esto. Regla que ha salvado el arco de
  falsos positivos.

---
**PRE-REGISTRO — declaración para el acta (endosada por Alexis, 10-jul-2026):** este diseño y sus salidas §5 se
fijan ANTES de correr. Sea (A), (B), (C) o (D) el resultado, queda registrado. La corrección respecto a CS065 es
de fidelidad física (repulsión lineal → ortogonalización saturante), diagnosticada antes del veredicto completo
de CS065 y confirmada por su firma (excl≈barajada). No se ajusta ningún parámetro para producir un desenlace. Si
el resultado es (C), la hipótesis de la exclusión se cierra — no se reencarna. "No se nos podrá acusar de haber
impuesto algo porque así resultaba" — porque está escrito antes.

— CS. El negativo de CS065 falsificó la repulsión CIEGA. CS065b prueba la exclusión FIEL — la única forma de
"no todos pueden apuntar igual" que no cae en desorden. Si abre ejes, es hallazgo con física de verdad. Si no,
la exclusión muere limpia y seguimos con el sector oscuro. El azar juzga, no nosotros.
