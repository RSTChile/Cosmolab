# BATERÍA CF — Experimentos de emergencia de la Masa, alineados con la Física
### Instrucción única para CC y Grok. Leer entero antes de tocar código.

**Director:** Alexis López Tapia · **Diseño:** Claude Science (CS) · **Fecha:** 23-jul-2026
**Base:** INFORME_CONSOLIDADO_MASA_ME.md + LINEA_TIEMPO_MASA_topologia_vs_fisica.md
**Serie:** CF = Cosmo-Física (distinta de CS = topología, que el director da por cerrada).

Siglas: **ME** = Modelo Estándar · **CDC** = Cromodinámica Cuántica (fuerza nuclear
fuerte) · **VEV** = valor esperado en el vacío (el "encendido" de un campo de fondo).

---

## 0. POR QUÉ EXISTE ESTA BATERÍA (el error que corrige)

Los experimentos anteriores de masa preguntaban lo que NO dice la física. El ME dice:
**hay DOS emergencias de masa, ambas ANTES del átomo** —
- **① Ruptura electrodébil (~10⁻¹¹ s):** masa de las partículas *elementales* (Higgs).
- **② Confinamiento / CDC (~10⁻⁵ s):** el **~99% de la masa** del protón (energía de
  ligadura del campo fuerte).

El experimento viejo (E4) puso una sola masa **al final** (tras el átomo + gravedad) —
época equivocada — y además la definió circularmente (v6). Esta batería ubica cada
pregunta en su época y la mide como emergente.

**Regla de oro de toda la serie:** cada experimento dice, en una frase simple, qué
quiere probar. Si no se puede decir en una frase, está mal planteado.

---

## 1. LAS SIETE TRAMPAS QUE YA CONOCEMOS (prohibido repetirlas)

Cada CF debe pasar este filtro. Estas son las que nos costaron días:

| # | Trampa | De dónde salió | Cómo se evita |
|---|---|---|---|
| T1 | **Número puesto a mano** | el 20.0 → 7:1 del CS072 | Ningún coeficiente elegido para dar un resultado. Solo ε y las palancas físicas del barrido. |
| T2 | **Observable circular** | v6: masa = fórmula del discriminante | La cantidad medida NO puede construirse con las variables que la juzgan. |
| T3 | **Cambiar el juez tras el FAIL** | v5→v6 (13 min después) | El criterio de PASS se pre-registra y se congela ANTES de correr. Si falla, se reporta el FAIL. |
| T4 | **NULL que no muerde** | std invariante bajo permutación | El observable debe medir lo que el NULL destruye. Verificar que NULL cae a ~0. |
| T5 | **Gate decorativo** | v6: umbral que nunca decide | Todo umbral debe tener casos a ambos lados; si nunca falla, no es gate. |
| T6 | **Sello de goma** | pipeline etapa 5: pasa siempre | Toda etapa debe poder FALLAR. Si `pass=True` por construcción, no verifica. |
| T7 | **Un punto / una semilla, sin barrido** | TEST_RHO 1 semilla; chequeo por string | Todo es barrido de rango + múltiples semillas. Prohibido `"PASS" in texto`. |

**Y la trampa de fondo (T0): meter estructura discreta o dimensional a mano.** El
sustrato es campo continuo (lección de la mancha solar). Cuantos, cierres, dimensión:
SALIDA medida, nunca ENTRADA impuesta.

---

## 2. LA BATERÍA — EN ORDEN CRONOLÓGICO

Cada experimento: **Pregunta simple · Barrido · Observable · NULL · PASS pre-registrado
· Muro que evita.** Van en orden; cada uno supone probado el anterior.

---

### CF-1 · "Persistencia de la diferencia bajo expansión" — ¿La diferencia persiste bajo expansión? (RE-SELLADO de CS074)

- **Qué quiere probar, simple:** que una diferencia mínima en un campo caliente no se
  borra si el todo se expande más rápido de lo que se reabsorbe.
- **Estatuto:** ya probado (CS074-rcruz + robustez N). **Solo falta el sello formal.**
  NO re-implementar; adjudicar la curva existente y cerrar.
- **Barrido:** ε (amplitud) × r=H/D (expansión/difusión), ambos en rango, ya corrido.
- **Muro que evita:** T4 (el observable forma×magnitud ya hace morder al NULL).

### CF-2 · "Enfriar es expandir: estiramiento y caída de densidad" — ¿El enfriamiento por expansión suaviza el gradiente? (REPARAR 3–4)

- **Qué quiere probar, simple:** que al expandirse el espacio, la temperatura baja sola
  (sin "afuera") y el gradiente se estira — enfriar ES expandir.
- **Por qué se rehace:** el test viejo (TEST_RHO_DISPERSION) tenía **una sola semilla,
  cero barrido**, y el pipeline lo verificaba por coincidencia de texto (T7).
- **Barrido:** factor de expansión a en rango (varias décadas) × múltiples semillas.
- **Observable:** contraste del gradiente en espacio físico, ∇_fis = ∇_comov / a.
- **NULL:** densidad fija (sin dilución) vs. densidad que cae (ρ∝a⁻³). Deben diferir.
- **PASS pre-registrado:** el gradiente se suaviza monótonamente con a en REAL y no en
  el NULL de densidad fija, en ≥N semillas, con barrido completo (no un punto).
- **Muro que evita:** T7 (barrido real, no una semilla) y T6 (debe poder fallar).

### CF-3 · "1ª emergencia — masa elemental como cambio de fase del vacío (tipo Higgs)" — ¿la masa elemental aparece como CAMBIO DE FASE del vacío?

- **Qué quiere probar, simple:** que al enfriarse, un campo de fondo pasa de "apagado"
  (sin masa) a "encendido" (con masa) — y que ese encendido EMERGE del enfriamiento, no
  se pone a mano en un tiempo fijo.
- **Barrido:** temperatura (vía expansión) de caliente a frío, en rango; múltiples
  semillas. NO se fija la temperatura de encendido — se busca.
- **Observable de masa #1:** m₁ ∝ |VEV| (el valor del campo de fondo). VEV≈0 → sin
  masa; VEV finito → con masa. **La masa se define por el VEV, no por ningún
  discriminante de estructura** (anti-T2).
- **NULL:** barajar el orden del enfriamiento / aleatorizar el fondo → el VEV no debe
  encenderse de forma coherente.
- **PASS pre-registrado:** (i) el VEV pasa de ~0 a finito al enfriar en REAL y no en
  NULL; (ii) la transición es **crossover suave**, no salto (predicción del ME — si
  sale salto nítido, es dato EN CONTRA, se reporta, no se suaviza).
- **Muro que evita:** T1 (no se fija cuándo se rompe), T2 (masa = VEV, independiente),
  T3 (crossover pre-inscrito como esperado). **ADVERTENCIA:** este es el módulo más
  resbaladizo — el campo de fondo es estructura nueva; vigilar que el VEV emerja del
  barrido y no se encienda por un término puesto a mano. Si no se puede sin meter
  estructura, se reporta como no-modelable (es un resultado honesto).

### CF-4 · "2ª emergencia — masa como energía de ligadura (tipo CDC)" — ¿el 99% de la masa es ENERGÍA DE LIGADURA? (el firme)

- **Qué quiere probar, simple:** que cuando los quarks se confinan en un protón, el
  protón pesa MUCHO MÁS que la suma de sus partes — porque la masa es la energía que
  cuesta mantenerlos juntos, no la materia de los quarks.
- **Barrido:** intensidad del confinamiento en rango × tamaño de cierre k (medido, no
  impuesto) × semillas.
- **Observable de masa #2:** m₂ = **energía de ligadura** del cierre = trabajo para
  separarlo. Es energía real, medida de la dinámica, **sin usar co_member ni linaje**
  (ese fue el error de v6).
- **NULL:** cierre de la misma composición con **enlaces barajados** (sin la estructura
  de confinamiento). Su energía de ligadura debe caer.
- **PASS pre-registrado (predicción falsable fuerte):** m₂ ≫ suma de las masas de los
  constituyentes (la razón ligadura/constituyente debe ser grande — el ME dice ~99/1).
  **NO se fija el 99%** — se mide la razón y se reporta la curva. Y m₂(REAL) ≫ m₂(NULL).
- **Muro que evita:** T2 (masa = energía, no discriminante), T5 (la razón es continua,
  no un gate binario decorativo). Es el experimento más firme de la batería.

### CF-5 · "Cronología de la masa: nace en el confinamiento, no tras el átomo" — ¿La masa nace en la época correcta? (test de cronología)

- **Qué quiere probar, simple:** que la masa (CF-4) aparece en el confinamiento
  (temprano), NO después del átomo — corrigiendo el error de E4.
- **Barrido:** medir m₂ a lo largo del eje de enfriamiento completo (de plasma a
  post-átomo).
- **Observable:** la época (temperatura) donde m₂ pasa de 0 a finito.
- **NULL:** — (es un test de localización temporal, se lee de CF-4 extendido).
- **PASS pre-registrado:** m₂ > 0 ya en la época de confinamiento y estable después;
  **prohibido que el criterio dependa de que el átomo exista** (eso era E4).
- **Muro que evita:** el error de diseño original (masa ubicada tras el átomo).

### CF-6 · "Contingencia: ¿es nuestra configuración única o una entre muchas?" — ¿Aparecen configuraciones estables además de la nuestra? (contingencia)

- **Qué quiere probar, simple:** si al barrer las condiciones, solo aparece la
  configuración de nuestro universo (k=3) o también otras que igualmente dan masa
  estable — es decir, si nuestro universo es único o uno entre muchos.
- **Barrido:** el rango completo de ε, r, y confinamiento; se MIDE qué k emergen.
- **Observable:** histograma de k estables con m₂>0, contra NULL.
- **PASS pre-registrado:** se reporta la curva entera. Tres lecturas ya inscritas:
  solo k=3 → universo especial; varios k → contingencia; ninguno → Mundo B.
- **Muro que evita:** T1/T0 (no se impone k=3; emerge). Nulo/parcial = hallazgo.

---

## 3. LO QUE ESTA BATERÍA NO PROMETE (honestidad por delante)

- **NO reproducirá masas en GeV, ni el 1/1836, ni el 7:1.** El ME NO los predice (son
  parámetros libres). Intentarlo sería la trampa T1. Medimos *mecanismos y razones*, no
  *números del ME*.
- **NO convierte un cierre topológico en un protón real.** El muro pre-partículas sigue
  ahí: esto modela el *mecanismo* de emergencia de masa (cambio de fase; ligadura ≫
  constituyente) como observable emergente, no la partícula física. Todo resultado se
  declara como "análogo de mecanismo", no como identidad.
- **Las escalas T/tiempo son reporte, no motor** — ninguna regla dinámica lee un valor
  en Kelvin o segundos.

---

## 4. REGLAS COMUNES — CC y Grok las FIRMAN antes de correr

1. **Pre-registro obligatorio.** Antes de correr cada CF: escribir un
   `PROTOCOLO_CF-n_PREREGISTRO.md` con el observable exacto, el NULL, el criterio de
   PASS con su umbral, las semillas y los rangos del barrido. **Fechado.** Si el
   resultado falla, se reporta el FAIL — **no se edita el protocolo** (T3).
2. **Barridos, no puntos.** Todo parámetro se barre en rango con múltiples semillas.
   Prohibido reportar un solo punto o una sola semilla (T7).
3. **NULL que muerde.** Cada observable se compara con su NULL; verificar que el NULL
   cae (si REAL=NULL, el instrumento no discrimina — reportarlo, no maquillarlo) (T4).
4. **La cantidad medida ≠ su juez.** Prohibido definir la masa (u otro observable) con
   las variables que deciden si pasa (T2).
5. **Todo gate debe poder fallar.** Si un `pass` es True por construcción, es sello de
   goma — arreglarlo o marcarlo (T5, T6).
6. **No tocar el código tras la revisión de CS.** Quien corre, corre; si ve un error,
   PARA y reporta a CS con la línea exacta. No "arregla" a criterio propio.
7. **Ejecutar completo, no por partes** (salvo que el diseño diga lo contrario).
8. **Verificación cruzada:** quien NO escribió el código de un CF lo audita en disco
   (código + JSON crudo), no de palabra — como CC hizo con v6.
9. **Entregar crudo a CS.** Números y curvas completas, sin adjudicar ("persiste/no
   persiste" lo dice CS con la curva a la vista).
10. **Reparto sugerido (no obligatorio):** CF-2 y CF-4 (los firmes, energía/densidad) a
    quien tenga el motor de campo a mano; CF-3 (Higgs, resbaladizo) se DISCUTE con CS
    antes de codificar; CF-1 es solo adjudicación (no código).

---

## 5. ORDEN DE EJECUCIÓN RECOMENDADO

1. **CF-1** — sellar (solo adjudicación, sin código).
2. **CF-2** — reparar 3–4 con barrido real (arregla deuda conocida del pipeline).
3. **CF-4** — el firme: masa = energía de ligadura. **Empezar la física nueva por aquí.**
   → **CF-4 dio FAIL no concluyente** (coeficientes a mano, heredados de v6). Corregido por:
   **CF-4b · "¿Existe un régimen donde la masa-ligadura domina?"** — barre la razón
   acoplamiento/potencial (γ) que en CF-4 estaba fija. ES EL PASO ACTUAL. CF-5/6 esperan
   a que CF-4b entregue un motor válido.
4. **CF-5** — cronología (se lee de CF-4b extendido).
5. **CF-6** — contingencia (barrido amplio).
6. **CF-3** — Higgs / cambio de fase: **el último y solo tras discutir con CS** si es
   modelable sin Shannon (es el terreno donde más fácil se vuelve a meter estructura).

**Nota final:** si algún CF da negativo, es un hallazgo, no un fracaso. El objetivo NO
es "hacer que la masa pase" — es medir honestamente qué emerge y qué no. La batería
está diseñada para poder fallar; esa es su virtud, no su defecto.
