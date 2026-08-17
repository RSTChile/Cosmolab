# INSTRUCCIÓN PARA CC — Experimento de cierre CS073, ejecución HOLÍSTICA

**De:** CS (diseño + adjudicación). **Para:** CC (implementación + corrida en el motor real).
**Regla del pacto:** CC implementa y corre; no rediseña. Un desacuerdo con el diseño es un DATO, se
coordina antes de tocar. Nota permanente vigente: no se cierra nada hasta que Alexis lo diga.

---

## REGLA 1 — Ejecutar COMPLETO de una sola vez, NO por partes
El experimento se corre con **TODOS los subsistemas operando simultáneamente** en la ventana
átomo→estrella, en un solo bucle temporal. NO correr una pieza, luego otra. Ya sabemos que por partes
no resulta — lo probamos 4 veces (densidad sola→grumo, red sola→hub, masa sola→ρ=T, expansión
sola→semillas suaves). El fenómeno (primera estrella) es la COMPETENCIA simultánea de las piezas; aislar
una rompe el equilibrio que ES el fenómeno.

## REGLA 2 — Depuración al revés del reduccionismo: primero el TODO, luego la parte
- Se corre el sistema COMPLETO primero.
- **Si falla completo**, recién ahí se baja a ver EN QUÉ MÓDULO falló. No se valida módulo por módulo
  antes de correr el todo.
- Primero miramos el todo; luego, si hace falta, las partes.

## REGLA 3 — Cada cosa nueva, en su MÓDULO (como el resto del experimento)
Igual que `p02_gravedad`, `p03_fuerte`, etc., las piezas nuevas van como módulos aislados, para poder
corregir uno sin tocar los demás:
- `p_gravedad_general.py` — fuerza G·m·m/r² sobre posiciones (NO el umbral térmico de Bgrav).
- `p_enfriamiento_H2.py` — canal de enfriamiento por H₂ (H+e⁻→H⁻+γ; H⁻+H→H₂+e⁻); libera calor →
  permite contracción Y fragmentación. SIN esto sale un solo grumo.
- `p_materia_oscura_halo.py` — CDM que colapsa ANTES que el gas y crea pozos de potencial (andamio).
- `p_expansion.py` — expansión del espacio: diluye fondo (ρ∝a⁻³) y separa regiones (crea el "lejos").
Cada módulo con su interruptor on/off, para que si el todo falla, se aísle y corrija el culpable.

---

## QUÉ CORRER (motor completo, ventana átomo→estrella)
Sobre el motor basal ya validado (S>0 → átomos H/He 75/25), añadir los 4 módulos nuevos y correr un
único bucle temporal donde en CADA paso actúan a la vez:
1. **CDM** colapsa primero → pozos de potencial (halos ~10⁶ M☉, z≈20-30).
2. **Gas H/He** cae en los pozos guiado por **gravedad general** (G·m·m/r², posiciones reales).
3. **Expansión** separa/diluye el fondo en cada paso (tensión contra la gravedad).
4. **EM** da presión térmica (choques) que se opone al colapso.
5. **Enfriamiento H₂** libera el calor del colapso → permite contracción y **fragmentación**.
6. Donde se cumplen los criterios de **Jeans** (M_J, λ_J, t_ff con constantes físicas reales
   G/k_B/h/m_p, μ≈1.22) → colapsa una estructura; la nube grande se fragmenta en varias.

Detalle completo de fórmulas y cantidades: INVENTARIO_atomo_a_estrella_CS.md.

## OBSERVABLE DE CIERRE (pre-registrado, antes de correr)
¿Nacen estructuras **MÚLTIPLES Y SEPARADAS** (fragmentación jerárquica), no un blob único?
- REAL (campo con coherencia) vs **NULL barajado** (misma distribución 1-punto, coherencia destruida).
- Discriminante: nº de estructuras ligadas separadas + que al menos una supere el umbral de Jeans.
- Gana al NULL con z-score sobre varias semillas, o no cuenta.

## ADJUDICACIÓN de las 3 preguntas de CC (coordinadas antes de tocar código)
**Q1 — constantes: ADIMENSIONAL, no SI.** CC tiene razón: δ_c=1.686 es número PURO (legítimo), pero
G/k_B/h/c/m_p en SI traen unidades de nuestro universo (π, α dentro) = contrabando que p24_tiempo.py
prohíbe. El observable de cierre (fragmenta vs blob) es topológico/adimensional — NO necesita SI.
Implementar M_J∝T^1.5/√ρ adimensional (como ya hace el motor). SI sólo si algún día se quiere predecir
masa en M☉, que NO es el observable. → sigue la forma de las fórmulas del inventario, SIN los números SI.

**Q2 — CDM EMERGE, no se planta (G-SIN-SIEMBRA, omitido por error, restaurado).** CC caza un error de
CS: G-SIN-SIEMBRA faltaba y es el que aplica. NADA de Press-Schechter ni estadística ΛCDM (= sembrar
centros a mano = Shannon). La materia oscura = SEGUNDA especie que sale del MISMO campo #23, corre bajo
la MISMA gravedad, pero DESACOPLADA de EM (no siente presión/radiación) → colapsa antes desde las mismas
fluctuaciones. La asimetría está en el ACOPLAMIENTO (sin EM), no en posiciones plantadas.

**Q3 — posiciones 3D = escenario post-fósil, NO re-derivar de la malla causal. [CONFIRMADO por Alexis
19-jul — luz verde.]** El Paso A (malla→MDS) dio negativo sólido: el sustrato relacional NO da embedding
3D. Re-discutirlo es regresivo — el tema de las posiciones YA está resuelto. Regla de Alexis, precisada:
**el espacio ya está en los átomos (radio de Bohr, escala CUÁNTICA), pero se vuelve DIMENSIONABLE en
términos MACROSCÓPICOS (no cuánticos) sólo con la expansión y la consolidación.** No nace aquí; la
métrica cuántica que ya existe se DESPLIEGA a escala macroscópica cuando la expansión separa y la
gravedad consolida. → el 3D es escenario legítimo: asignar posiciones 3D (D=3, dimensión fosilizada YA
probada) portando el campo #23; DEJAR de sacar posiciones de la malla causal (falló, bloquea). Las
posiciones son el ESCENARIO, no la claim. NO es Shannon: 3D uniforme + fluct. #23 no siembra estructura
(se mide vs NULL); el 3D no se decreta, es el resultado fosilizado probado. Todo adimensional. LA
EXPANSIÓN es la que vuelve ese 3D cuántico en 3D macroscópico dimensionable — por eso es imprescindible.

## GUARDIANES (anti-Shannon, heredados del arco)
- **G-DIFERENCIA-INTERNA:** toda diferencia = el campo consigo mismo (NULL barajado). Nada inyectado.
- **G-SIN-ENERGIA-NUEVA:** M_J depende sólo de T y ρ heredadas; el motor no inyecta energía para colapsar.
- **G-SIN-SIEMBRA:** cero centros de colapso impuestos; CDM y sobredensidades salen del campo #23 real.
- **G-PARAMETROS-ESTRUCTURALES:** fórmulas de Jeans en forma ADIMENSIONAL (sin constantes SI = contrabando);
  parámetros derivados de la física o barridos, nunca a ojo.
- **G-DOS-GRAVEDADES:** la gravedad general es módulo NUEVO sobre posiciones 3D, no un reajuste de Bgrav.
- **G-EXPANSION-ISOTROPA:** la expansión no impone dirección ni rejilla.

## COSTO
El motor es O(N²) hoy (2800 quarks ≈ 621s). La escala de "masa suficiente" puede requerir corrida larga
o en segundo plano en el entorno de CC — no en el kernel de CS (límite de RAM). Evaluar antes de lanzar.
