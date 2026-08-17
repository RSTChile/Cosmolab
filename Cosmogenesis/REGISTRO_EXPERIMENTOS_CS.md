# REGISTRO DE EXPERIMENTOS — Cosmogénesis (numeración secuencial CS)

**Director:** Alexis López Tapia · **Registro montado por:** Claude Science (CS) · **Fecha:** 5-jul-2026
**Regla de orden (instrucción de Alexis):** *"secuencia simple, única, reiterada — eso es orden."*

**Procedencia de cada estado (importante para trazar):** los experimentos los CORRE CC (u otro miembro del
equipo) en su terminal; CS NO ejecuta los scripts del equipo (regla de la sesión). Un estado "corrió" en
este registro significa: CC entregó su informe + el script y su `*_run.log` quedaron en la carpeta
Cosmogenesis, y CS los AUDITÓ contra el código y firmó una adjudicación. La evidencia de una corrida es el
par (informe de CC entregado a CS) + (`<exp>_run.log` en disco con el veredicto ejecutado), no una
ejecución dentro de la sesión de CS.

---

## 0. POR QUÉ EXISTE ESTE REGISTRO
El equipo (CC, Grok, GPT, CS) numeró los experimentos con etiquetas internas que se hablaban entre
ellos —CG001, CG002, R7a, r7b, cg004f3, EDS v0…— sin una secuencia única, y el director quedó fuera de
la nomenclatura. Peor: en el arco de estos días se corrigió sobre la marcha que se estaba MIDIENDO MAL
(nodos sin identidad en vez de diferencias persistentes; el error del fundamento). Sin trazabilidad,
"el enredo va a ser fenomenal".

**Regla desde ahora:**
1. **Todo experimento nuevo recibe un número CS correlativo único:** CS001, CS002, CS003…
   Nunca se reutiliza ni se salta. Un experimento = un número, para siempre.
2. **El diseño de cada experimento indica su etiqueta técnica** entre paréntesis, p.ej.:
   *"CS023 — aquí probamos el sector gluón del modelo estándar (dimensión técnica: CG002 / r7b)."*
3. **Las etiquetas viejas se conservan solo como alias** en la columna "etiqueta técnica", para poder
   rastrear los archivos que ya existen en disco. No se usan para nombrar experimentos nuevos.
4. **El veredicto se marca con el mismo léxico de siempre:** ✅ resultado real · ⊘ parcial/tensión ·
   ❌ no emerge/refutado · 🔒 cerrado · ⏳ abierto/en curso.

Este registro es retro-activo (numera lo ya hecho) y prospectivo (lo nuevo sigue desde el último CS).

---

## 1. TABLA MAESTRA (retroactiva — lo ya hecho, en orden de arco)

| CS | Etiqueta técnica (alias) | Archivo real | Qué se probó | Veredicto |
|----|--------------------------|--------------|--------------|-----------|
| **CS001** | CG001 campo | cg001_field.py | ¿Emerge un campo desde la regla mínima? | ⊘ base del arco |
| CS002 | CG001 barrido grueso | cg001_barrido_grueso.py | Barrido de parámetros del campo (grueso) | ⊘ soporte |
| CS003 | CG001 barrido fino | cg001_barrido_fino.py | Barrido fino del campo | ⊘ soporte |
| CS004 | CG001 iPad | cg001_ipad.py / _1000 | Corridas de producción del campo (iPad) | ⊘ soporte |
| CS005 | CG001 causalidad | cg001_test_causalidad.py / _ipad_causalidad | ¿Hay estructura causal? | ⊘ ver INFORME_CAUSALIDAD_EPSILON |
| CS006 | CG001 localización | cg001_test_localizacion.py | ¿Se localiza la diferencia? | ⊘ soporte |
| CS007 | CG001 persistencia | cg001_ipad_persistencia.py | ¿Persisten las diferencias? | ✅ C-N1 |
| **CS008** | CG002 A1 baryogénesis | cg002_baryogenesis.py / _experimentos_arco | ¿Exceso materia sin programarlo? | ✅ ~50% sobrevive |
| CS009 | CG002 A3 flecha | cg002_experimentos_arco.py (flecha) | ¿Flecha del tiempo emerge? | ✅ monótona |
| CS010 | CG002 A4 dimensión | cg002_experimentos_arco.py (dimension) | ¿Dimensión heredada del sustrato? | ✅ S¹→1…S⁴→3.4 |
| CS011 | CG002 A5 coexistencia | cg002_experimentos_arco.py (coexistencia) | ¿Dominios coexisten? | ✅ |
| CS012 | CG002 A6 criticidad | cg002_experimentos_arco.py (criticidad) | ¿Transición de fase por alcance? | ✅ pico rc≈3 |
| CS013 | CG002 A8 inercia | cg002_experimentos_arco.py (inercia) | ¿Duración da resistencia? | ⊘ tensión C-N5.1 |
| CS014 | CG002 A9 constantes-vs-historia | cg002_experimentos_arco.py (constantes) | ¿Razones convergen entre cosmos? | ✅ ley vs historia |
| CS015 | CG002 A10 invariantes | cg002_experimentos_arco.py (invariantes) | ¿Tipologías de ruptura? | ✅ κ_Δ,κ_V,κ_P |
| CS016 | CG002 A11 paredes de dominio | cg002_experimentos_arco.py (paredes) | ¿Fronteras nítidas? | ❌ no robusto |
| CS017 | CG002 B1 1000 cosmos | cg002_constantes_1000.py | ¿Constantes robustas a 1000 corridas? | ✅ CV<2% |
| CS018 | CG002 B3 exceso-barrido | cg002_exceso_barrido.py | ¿Exceso aguanta η,μ,S_BAND,d? | ✅ +0.015 a d=3 |
| CS019 | CG002 B4 exceso-caracteriza | cg002_exceso_caracteriza.py | ¿Exceso es sesgo finito? | ✅ 88-95% real |
| CS020 | CG002 B5 grain-null | grain_null_model.py | Línea base combinatoria 1/√N | ✅ nulo exacto |
| CS021 | CG002 B6 L2-dinámico | cg002_dynamic_l2_sweep.py | ¿El motor se aparta del nulo? | ✅ ratio 12× |
| CS022 | CG002 B7 m_eff/K vs banda | cierre_kappaDelta (engine_output.csv) | ¿½ es derivación o calibración? | ✅ DERIVACIÓN (0.0000) |
| CS023 | CG002 acoplamiento originario | cg002_acoplamiento.py | ¿B PASS dirección? | ✅ cualificado |
| CS024 | CG002 multicomponente | cg002_multicomponente.py | ¿Rango = dimensión? | ✅ 3D |
| CS025 | CG002 observables v0.2 | cg002_observables_v02.py | Observables del motor | ⊘ soporte |
| **CS026** | CG002 **r7a** color+spin | cg002_r7a_color_spin.py | Sector color SU(3), spin PASIVO | ✅ polarización real (abeliano) |
| CS027 | CG002 **r7b** gluón-entidad | cg002_r7b_gluon_entity.py | Sector gluón (octeto, no-abeliano) | ❌ BLOQUEADO (vértice 3-puntos) |
| CS028 | CG002 **r7c** leptones | cg002_r7c_leptons.py | Sector leptónico | ⊘ ver informe R7 |
| CS029 | CG002 **r7d** carga U(1) | cg002_r7d_charge_u1.py | Sector carga eléctrica (abeliano) | ✅ FUNCIONA |
| CS030 | CG002 **r7e** generaciones | cg002_r7e_generations.py | Tres generaciones fermiónicas | ✅ (abeliano) |
| CS031 | CG002 **r7f** Higgs/Yukawa | cg002_r7f_higgs.py | Sector Higgs (Yukawa) | ❌ BLOQUEADO (misma pared que r7b) |
| CS032 | CG002 **r7g** primitivo 3-puntos | cg002_r7g_vertex3.py / _primitivo_vertex3 | Extender la regla a 3 puntos | ⏳ auditado, pared NO cayó aún |
| CS033 | CG002 **r7h** quiral/Hubble | cg002_r7h_chiral.py / _primitivo_chiral | Medición con historia (tensión Hubble) | ⏳ θ=0 no colapsó limpio |
| **PARED R7** | (hallazgo de cierre CG002) | CIERRE_ARCO_CG002_AUTORITATIVO | El motor pareado (2-puntos) cubre lo abeliano; el gluón y el Higgs (3-puntos) quedan fuera | 🔒 frontera documentada |
| **CS034** | CG003 espacio relacional | cg003_espacio_relacional.py / _diagnostico_gromov | ¿Emerge espacio métrico? | ⊘ base arco espacio |
| CS035 | CG003b exergía primero | cg003b_exergia_primero.py | Orden por exergía | ⊘ soporte |
| CS036 | CG003c crecimiento | cg003c_crecimiento.py | ¿Crecimiento da geometría plana? | ❌ no plano (mundo-pequeño) |
| CS037 | CG003d campo angular | cg003d_campo_angular.py | Campo angular por dirección | ⊘ preludio a cg003f |
| CS038 | CG003f planitud-exergía | cg003f_planitud_exergia.py / _carnets / _b | ¿Holonomía-costo aplana? | ❌ no despliega (degenerado→corregido) |
| **CS039** | CG004 attach | cg004_attach.py | ¿El attach local genera plano? | ❌ no |
| CS040 | CG004b ciclos | cg004b_ciclos.py | ¿Cerrar ciclos aplana? | ❌ indistinguible de árbol |
| CS041 | CG004c robusto | cg004c_robusto.py | Robustez del negativo (8 semillas, N grande) | ✅ negativo aguanta |
| CS042 | CG004d dos-frentes | cg004d_dosfrentes.py | Cirugía/Ricci a dos frentes | ❌ mide rotacional (circular) |
| CS043 | CG004e retícula-cortada | cg004e_reticula_cortada.py | Desarrollo/holonomía afín | ✅ primer positivo (plano⟺univaluado) |
| CS044 | CG004f curvatura | cg004f_barrido_curvatura.py | Cortar+repegar por holonomía | ⊘ muro: bisagra única pliega |
| CS045 | CG004f2 barrido-cortar | cg004f2_barrido_cortar.py | Transporte afín sobre corte | ⊘ muro: 3 rutas fallan |
| CS046 | CG004f3 cinta-Eisenstein | cg004f3_cinta_eisenstein.py | Transporte por caras, Eisenstein exacto | ✅ **(P-κ) cerrado**: pega preserva, no genera |
| **CS047** | CG005 v0 (EDS) | cg005_eds_v0.py | Color inmutable: ¿el lógos confina? | ✅ confina 100% vs NULL 82% |
| CS048 | CG005 v1 (EDS temporal) | cg005_eds_v1.py | Orden temporal como "al lado de" | ⊘ confina mejor (89 vs 62) pero gas |
| CS049 | CG005 v2 (EDS residual) | cg005_eds_v2.py | + fuerza residual débil | ❌ gas (local) / blob (no-local) |
| CS050 | CG005 v3 (EDS energía) | cg005_eds_v3.py | Orden por energía emergente (t_freeze) | ❌ negativo, G4 anti-relabel PASA |
| **CIERRE ARCO** | CG004+CG005 | adjudicacion_cg005_v3_CIERRE_ARCO_CS | Ninguna regla local genera plano; el lever es el MARCO, no la adyacencia | 🔒 negativo convergente (2 caminos) |
| **CS051** | (puerta, dirección) | PUERTA_R7_espin_como_marco | El espín = "hacia dónde" ausente; candidato al marco | ⏳ HIPÓTESIS (no corrido) |
| **CS052** | marco/espín (diseño) | DISENO_CS052_marco_espin.md | Orientación intrínseca (espín) que se alinea al ligar; masa 99% relacional, Higgs 1% | ⏳ diseño → corrido en v0/v1 |
| CS052-v0 | marco por-NODO (espín quark) | cs052_marco_espin.py / INFORME_CS052_v0 | ¿El marco de nodo genera plano? | ❌ gauge puro (holonomía 0 siempre); reveló que la curvatura vive en el LINK (gluón) |
| CS052-v1 | co-emergencia (LGT: nodo+link) | cs052_v1_coemergencia.py / adjudicacion_CS052_v1_CS | A=entidad sola, B=vínculo libre, C=vínculo atado; ¿cuál carga geometría? | ✅ **A=0,B=0,C discrimina** — el espacio VIVE en el vínculo atado (ontológico). Generar plano sigue abierto |
| **CS053** | persistencia geometría/dimensión | cs053_persistencia_geometria.py / adjudicacion_CS053_CS | De todas las geometrías/dimensiones, ¿cuáles PERSISTEN? Filtro ciego S=I·E | ❌ **falsación honesta**: sobrevive TODO retículo ≥2D por igual; la persistencia mata lo frágil pero NO fija 3D-plano (G-NO-HORNEAR y G-NULL puestos) |
| **CS054** | gravedad en el filtro | cs054_gravedad_en_el_filtro.py / adjudicacion_CS054_v2_ALCANCE_CS | Añadir la gravedad (densidad↔curvatura, balance vs despliegue) al filtro de CS053 | ❌ **falsación ACOTADA**: la gravedad SIN alcance colapsa todo ≥2D a blob (d≈3-plano=0). Falsa "gravedad sin alcance selecciona", casi tautológico (Alexis: gravedad uniforme = sin universo). Falta el ALCANCE |
| **CS054-v2** | gravedad CON alcance | cs054_v2_gravedad_alcance.py / adjudicacion_CS054_v2_CIERRE_CS | La gravedad decae con la DISTANCIA DE GRAFO (cuadrado inverso sin espacio, α∈1,2,3, D_MAX=2) | ✅/❌ **positivo de mecanismo + falsación**: el alcance vuelve SELECTIVA a la gravedad (deja de colapsar, elige geometría) — intuición de Alexis probada. PERO elige **2D**-plano, no 3D (cubo 3D muere 0/3, α-robusto). Nuestro universo (3D) la falsa. Cuerda: el contador "d≈3" del clasificador NO es fiable, leer los TIPOS |

---

| **CS055** | proceso acoplado | cs055_proceso_acoplado.py / adjudicacion_CS055_CS | Enfriamiento + gravedad-con-caída + confinamiento + despliegue, TODO A LA VEZ | ✅/❌ **dos fuerzas visibles + falsación**: confinamiento-solo sostiene 3D (3/3, empuje ARRIBA), gravedad-sola colapsa a 2D (empuje ABAJO) — las dos fuerzas de Alexis, con dato. PERO a fuerza IGUAL (1:1) la gravedad domina → 3D no emerge (acoplado=2D, G-NULL lo confirma). La razón 1:1 NO es el valor físico real |
| **CS055-v2** | razón de fuerzas física | (subsumido en CS056) | Fijar la razón por su valor físico real con barrido y predicción ciega | ↪ SUBSUMIDO en CS056 (el barrido de 4 fuerzas incluye esta razón) |

| **CS056** | proceso TOTAL (4 fuerzas) | cs056_cuatro_fuerzas.py / adjudicacion_CS056_CS | Las CUATRO fuerzas juntas + enfriamiento + despliegue, intensidades físicas, barrido de razón | ✅/❌ **válido bajo su supuesto + hueco destapado**: el EM no rescata el 3D (a fuerza real inerte; a fuerza alta interfiere — hallazgo real: color y carga son DOS neutralidades independientes que se pelean). PERO gravedad y EM se corrieron con el MISMO alcance (D_MAX=2), tapando la asimetría física real (gravedad se acumula/largo, EM se cancela/corto) — pregunta de Alexis. Puerta EM NO cerrada |
| **CS056-v2** | alcances distintos grav/EM | (propuesto en adjudicacion_CS056_CS) | MISMA ley 1/d² pero ALCANCE distinto: gravedad largo (se acumula), EM corto (se cancela por neutralidad). Predicción ciega. ¿El EM de corto alcance sostiene la malla 3D? | ↪ SUBSUMIDO en CS057 (el eje de alcances distintos entra en el paisaje completo) |

| **CS057** | paisaje completo (todas las fuerzas + sector oscuro + sync/async) | cs057_paisaje_completo.py / cs057_paisaje.csv (69.648 filas) / adjudicacion_CS057_CS.md | Barrer TODAS las fuerzas 0→1 (muestreo del hipercubo, punto físico marcado), distancia modulando cada una por su alcance. Criterio: ¿qué combinaciones ESTABILIZAN un universo persistente en expansión, de CUALQUIER dimensión? (no cazar 3D). Sector oscuro (materia/energía) como SALIDA emergente, nunca insertado. Brazo sincrónico vs asincrónico = falsación del "es un proceso" | ✅ **corrió (CC, 10.4h) y ADJUDICADO — FALSACIÓN ACOTADA**: las fuerzas locales reales, barridas exhaustivamente, NO seleccionan el 3D-plano (el punto físico cae viable 0.375 vs 0.094 fondo, PERO estabiliza CURVO curv~0.84, no d3-plano ~0.15). Cierra el arco de fuerzas → apunta a R7. Predicción Alexis: P1✅ P2✅-con-giro P3✅ (región estrecha 11%, resonancia). Sync>async +10% z≈5 (proceso, sobrio). Sector oscuro emergente 2.4× cerca del físico (candidato honesto, no insertado). Pend menor: CC reconcilia "d3=0.00" (informe) vs 236 filas d3-viable (CSV) — dirección idéntica |

| **CS058** | zoom denso al candidato de energía oscura | DISENO_CS058_zoom_energia_oscura.md | Caracterizar (o matar) la aceleración emergente que CS057 vio 2.4× cerca del punto físico: ¿sobrevive a más resolución/semillas, tiene región contigua propia, y vive en la frontera CURVA (lo que la conectaría con R7)? Malla densa local leída del CSV de CS057 (no elegida a mano), brazo de resolución ×1/×2/×4, brazo NULL. G-NO-INSERTAR-OSCURO heredado | ✅ **corrió (CC, completo 1404 pts) y ADJUDICADO — REAL-PERO-DÉBIL** (adjudicacion_ARCO_CS058-061_CS.md §1, CORREGIDO desde "artefacto" del parcial): SUPERA al NULL (accrob 0.115 vs 0.069, ratio 1.66 → señal real) PERO decae con resolución (0.170→0.106→0.069, no robusta) y máx en d4 no curv (desacoplada de R7). Ni artefacto, ni candidato firme, ni puente a R7: fenómeno propio débil, línea aparte. **Lección: no declarar veredicto firme desde un parcial (CC lo destapó, CS lo cometió en v2 y lo asume)** |

| **CS059 (R7)** | el espín como MARCO — la orientación que ninguna fuerza local dio | DISENO_CS059_R7_espin_como_marco.md | EL experimento al que apunta el arco entero. Meter el espín (orientación intrínseca de quarks/gluones = marco de referencia local) sobre el EDS de CG005; regla de acoplamiento de marcos al ligar (holonomía guardada en el enlace); el Burgers de CG004 como juez ciego (plano→0, curvo≠0). ¿Es el MARCO —no la fuerza— lo que selecciona una dimensión? Éxito = selección consistente y falsable que colapsa bajo NULL, NO "salió 3D". Intuición y autoría del ingrediente: Alexis | ✅ **corrió (CC) y ADJUDICADO — NEGATIVO, desenlace C** (adjudicacion_ARCO_CS058-061_CS.md): el marco pareado (2 puntos) NO selecciona dimensión. CC cazó el FALSO POSITIVO — el orden de holonomía era idéntico al orden de longitud de ciclo; control a igual longitud → la "selección" se desintegró. Apunta al vértice de 3 puntos (→ CS061) |

| **CS060** | los tres leptones + la MASA (electrón=control, muón/tauón=generaciones) | DISENO_CS060_leptones_y_masa.md | El leptón = marco (espín ½) SIN color, el único que rompe el confound marco↔ligadura de CS059. Brazo CONTROL (electrón): ¿un marco que no se liga aporta geometría o vaga libre? Brazo GENERACIONES (e/μ/τ a razones de masa reales 1:207:3477): ¿la masa —mapeada a inercia de orientación + persistencia temporal— cambia qué dimensión se selecciona? Juez: Burgers de CG004. **Corrección asentada: el veto a la masa lo puso el EQUIPO, nunca Alexis.** 7 guardianes, críticos G-MASA-FÍSICA-FIJA, G-MASA-ES-INERCIA-NO-DIMENSIÓN, G-NO-NUMEROLOGÍA-DEL-TRES | ✅ **corrió (CC) y ADJUDICADO** (adjudicacion_ARCO_CS058-061_CS.md): **A (leptones)** = G2, la masa cambia la COHERENCIA del marco (umbral, no gradiente) pero NO selecciona dim. **B (gravedad∝masa vs grado, la de Alexis)** = FALSO POSITIVO cazado por el NULL: masa (viab 3D+4D 0.211) ≈ null (0.211) — el efecto NO es masa sino independencia-del-grado (el proxy de grado se auto-amplifica y sesga contra 3D). GRIETA POSITIVA: gravedad∝peso 2× > grado → relee el negativo de CS057 |

| **CS061** | la masa que EMERGE — vértice de 3 puntos tipo Higgs (convergencia del arco) | DISENO_CS061_masa_emergente_higgs.md / cs061_masa_emergente.py / cs061_masa.csv / adjudicacion_ARCO_CS058-061_CS.md | El experimento convergente: el vértice de 3 puntos que CS059 pidió = el del Higgs = el origen de la masa. Campo φ uniforme + defecto de tríada (3 cuerpos genuino en la MEDICIÓN) + inercia EMERGENTE m_i=φ·frustración. Juez doble: ¿el 3-puntos selecciona dim donde el 2-puntos no? + espectro de masas | ✅ **corrió (CC) y ADJUDICADO — NEGATIVO (C/D mezclado)** (adjudicacion_ARCO_CS058-061_CS.md): 3punto (0.567) ≈ null_campo (0.549) ≈ null_vértice (0.547) → COLAPSA bajo NULL; espectro trivial (razón 2.14 vs 3477 real, sin ceros tipo fotón). **CAVEAT verificado en código por CS:** el update del marco es campo-medio PAREADO (media de vecinos); la inercia de 3 cuerpos solo amortigua un escalar → el vértice de 3 cuerpos está en la MEDICIÓN, NO en la DINÁMICA. Por eso es (C) para "inercia amortigua relajación pareada" pero deja ABIERTO (D): update 3-cuerpos genuino (→ CS063) |

| **CS062** | el paisaje con gravedad ∝ PESO-INTRÍNSECO (no grado) | DISENO_CS062_paisaje_peso_intrinseco.md | ★ Re-correr el paisaje de CS057 cambiando SOLO el acople gravitatorio: de ρ=grado (que se auto-amplifica y sesga contra 3D) a m_i·m_j/d² con peso intrínseco fijo (Newton real). ¿El 3D emerge más en TODO el mapa con la gravedad correcta, o el negativo de CS057 se sostiene? Brazos PESO/GRADO(=CS057)/NULL-PESO. Origen: grieta positiva de CS060-B + observación de Alexis (gravedad-sin-masa) | ⏳ DISEÑO (a codear CC). **Prioridad 1 (barato: reusa CS057, cambio quirúrgico).** G-PESO-INTRÍNSECO-FIJO, G-NULL-PESO (separa peso-real de no-grado). Desenlaces A (relee CS057) / B (solo independencia-del-grado) / C (negativo se sostiene) |

| **CS063** | el vértice de 3 CUERPOS GENUINO (cerrar C/D de CS061) | DISENO_CS063_vertice_3cuerpos_genuino.md | Lo que CS061 NO hizo: un update donde los TRES marcos de una tríada se muevan JUNTOS (término irreducible de 3 cuerpos, ∂³E≠0), no cada nodo hacia la media de vecinos (campo medio pareado, el error de CS061 que CS verificó en código). ¿AHORA el 3-cuerpos selecciona dim donde el pareado no pudo? Juez: Burgers con control de longitud de ciclo | ✅ **corrió (CC) y ADJUDICADO — NEGATIVO (B)** (adjudicacion_ARCO_CS058-061_CS.md, §4-bis): G-IRREDUCIBLE VERIFICADO por CS (∂³E=1.96≠0, producto triple escalar, update mueve los tres marcos sin término pareado — es 3-cuerpos GENUINO). COLAPSA bajo NULL: 3cuerpos 1.050 ≈ null_marco 1.062 ≈ null_triada 1.053, casi plano por dim. **Ni el vértice de 3 cuerpos genuino selecciona la dimensión → cierra el (D) de CS061 → arco de eliminación local COMPLETO → la CONTINGENCIA se gana el derecho a ser la conclusión** |

| **CS064** | sistema completo / el espacio como exaptación | DISENO_CS064_sistema_completo_emergencia.md / cs064_sistema_completo.py / cs064_N{1500,2500,3500}.csv / cs064_barrido.log | El motor COMPLETO a la vez (4 fuerzas + aniquilación + co-evolución del marco), desde sopa caliente sin retícula. ¿emerge geometría Y dirección de la relación plena? null_marco = test de EXAPTACIÓN (O-N8.3) | ✅/❌ **EXAPTACIÓN del marco CONFIRMADA + blob destapado** (verificado por CC en datos; blob AUDITADO por CS sobre los CSV): congelar el marco (null_marco) MATA la dirección → n_ejes 0.00 vs completo 1.32 (las direcciones son REÚSO de la inercia del marco, no primitivo). PERO el sustrato es un blob ULTRA-mundo-pequeño: diam≈3.9 que NO crece con N (1500→3500), d_s se infla 4.8→5.6, colapso a ~1.3 ejes. Ese blob (auditado por CS sobre los CSV) FUNDA CS066 |

| **CS065** | exclusión de Pauli (anti-colapso), 1ª forma | DISENO_CS065_exclusion_pauli_anticolapso.md / cs065_exclusion_pauli.py / cs065_N{1500,2500,3500}.csv | ¿Una repulsión tipo Pauli entre marcos rompe el colapso-a-1-eje de CS064? Arms: excl/sin_excl/excl_barajada/excl_bosones/marco_congelado | ❌ **FALSIFICADO** (datos de CC; adjudicado por CS como árbitro — el diagnóstico "repulsión lineal ≠ Pauli" fue ANTES del veredicto completo): la exclusión NO sostiene ejes — los REDUCE (excl 0.62 < sin_excl 1.23) e indistinguible de su barajada (excl 0.62 ≈ barajada 0.68). Empuja a ISOTROPÍA, no a ortogonalidad. Cuerda anti-Shannon: real≈barajado ⇒ no específico. marco_congelado=0 (sanidad ok) |

| **CS065b** | exclusión ORTOGONALIZANTE (Pauli fiel), 2ª forma — PRE-REGISTRADA | DISENO_CS065b_exclusion_ortogonalizante.md / cs065b_exclusion_ortogonalizante.py / cs065b_N{1500,2500}.csv (100 c/u) + N3500 parcial 24/60 | La traducción FIEL de Pauli (Gram-Schmidt saturante: ortogonaliza y PARA). Decisivo pre-inscrito: excl_orto vs excl_orto_barajada — si no se separan, la exclusión muere sin duelo | ❌ **FALSIFICADO — salida (C) pre-inscrita "muere sin duelo"** (adjudicado por CC en datos, 10-jul): la cuerda decisiva es NULA en los 3 N: Δ(excl_orto−barajada)=−0.02/−0.09/−0.08 (t=−0.17/−0.75/−0.39); barajada ≥ excl_orto siempre. El espejismo de N=1500 (excl_orto>sin_excl t=+3.69) no salva: la barajada llega al mismo n_ejes ⇒ es ruido ortogonalizante, no fidelidad a Pauli. corr(frac_ferm,n_ejes) no positiva. n_ejes<D_max=8 (sin techo). **Cierra la exclusión en sus DOS formas** |

| **CS066** | LOCALIDAD primero / geometrogénesis — costo de no-localidad EN LA FORMACIÓN | DISENO_CS066_localidad_geometrogenesis.md / cs066_localidad_geometrogenesis.py / cs066_N{1500,2500,3500}.csv / cs066_barrido.log | ¿El blob de CS064 se vuelve tejido LOCAL con "lejos" (diam~N^(1/d), d_s finito) y recién ahí, cuántos ejes? Actor nuevo: la localidad gobierna qué enlaces PERSISTEN (cada nodo conserva sus k_local más locales; el smoke descartó el podador externo — sin punto fijo de tejido, o blob o gas). k_local sorteado (G-NO-CALIBRAR). Arms: local/sin_local(=CS064)/local_barajado/local_marco_congelado. G-TEJIDO-ANTES-QUE-EJES | ✅/❌ **corrió (CC) — salida (B): ESPACIO LOCAL SÍ, DIRECCIONES NO** (datos de CC, AUDITADO y confirmado por CS sobre los 3 CSV, 11-jul). **Nivel 1 (tejido):** con localidad fuerte (k_local 5-6) emerge tejido CON especificidad — d_s se ESTABILIZA ~3 (vs blob que se infla 4.8→5.6), clustering 0.41 vs 0.10 barajado, diam 12 vs blob 3.9, gigante sano (sin gas). Mata (D) placebo. **Nivel 2 (direcciones):** sobre ese tejido el colapso-a-1 PERSISTE, y la auditoría de CS lo endurece: local NO solo no supera a barajado — va PEOR con significancia (Δ n_ejes local−barajado = −0.00/−0.61/−0.67; Welch t=+0.00/−1.96/−2.31; p=1.00/0.059/**0.036** en N=1500/2500/3500). El tejido apretado SUPRIME ejes, no los crea. marco_congelado da tejido idéntico (clustering 0.43) con 0 ejes → falla (A) de forma limpia. **Espacio y direcciones son problemas SEPARADOS; reordena el arco.** G-NO-TOPADO ok (n_ejes máx=5 ≪ D_max=8). Caveat (lección CS058): el exponente diam~N^(1/d) NO salió limpio (diam local 11.0→12.8→12.1, baja a N=3500; bins de baja-k flacos, 9 parches en N=3500); "hay tejido" se apoya en 4 firmas convergentes (d_s-estabiliza~3 + clustering 4× + diam 3× + especificidad vs barajado), NO en una ley de potencia nítida → confirmatorio (malla k_local×N) pendiente. **Auditado y confirmado por CS sobre los 3 CSV (1040 parches).** ‖ **CONFIRMATORIO Nivel 1 (24 celdas k∈{3,4,5,6,8,10}×N∈{1500..5000}×40 parches, AUDITADO por CS con ajuste propio sobre los CSV, 12-jul):** el test pre-registrado del exponente NO confirma d≈3 — **0/6 celdas** en [0.29,0.40] (slopes k3-10 = 0.148/0.576/0.169/0.135/0.118/0.127; donde diam escala fuerte k4 slope 0.58 el gigante es 0.40=filamento roto; donde gigante sano k5-6 slope 0.13-0.17=mundo-pequeño residual). PERO el d_s espectral SÍ es un dial limpio, monótono y estable en N: pasa por ~3 entre k5-6 con gigante 0.90. **Veredicto CS: el tejido es LOCALMENTE geométrico (d_s~3, clustering alto, conexo) pero GLOBALMENTE compacto (atajos de largo alcance sobreviven — retícula-3D real diam ~27→39 con N; nuestro diam crece muy poco: k5 13.6→16.6, k6 6.9→8.1 al pasar N 1500→5000). Esponja 3D-local con mundo-pequeño residual, NO 3-manifold métrico.** El (B) global intacto; nuevo para CS067: falta cerrar los atajos globales del espacio, no solo levantar las direcciones. cs066conf_exponentes.md firmado |
| **CS067** | la HABITACIÓN COMPLETA — los 17 ingredientes juntos + los 3 del video (correlación-métrica, cono causal, SSB multi-dim) + sector oscuro emergente | DISENO_CS067_habitacion_completa_CS.md / cs067_gamma_sweep.py / cs067_gamma_sweep_blindaje.log / ADJUDICACION_CS067_SSB_juez / _bifurcacion_ab / ADDENDUM_pico_guarda_rango | ¿La RELACIÓN de los 17 enciende las direcciones que ninguna pieza sola pudo? Voto Potts pesado por correlación + cono causal + SSB con manifold de vacío K-dim. 4 iteraciones de diseño, cada fallo cazado (juez de gap no distingue K-discreto/continuo → candado de picado por nodo; SSB snap hornea K → realización Potts; Potts×cono colapsa a 1 por atajos mundo-pequeño → voto pesado por w; pico mal implementado + gap artefacto de borde → guardas) | ✅ **VEREDICTO (B) CANÓNICO, blindado y auditado por CS (16-jul)** — 160 corridas (16 sem/γ completo + 8/γ control, γ∈[0.5,2.5]), auditado sobre 80 semillas del brazo completo. En NINGÚN régimen el completo cumple las 3 condiciones: pico_medio nunca cruza 0.85 (IC95%_inf máx 0.789 @γ2.0; global 0.782); anti-especificidad limpia (controles igualan/superan a completo: γ1.5 sin_corr 0.789≈completo 0.786; γ2.5 sin_causal 0.804>completo 0.778); n_ejes disperso (CV 0.35-0.81) — dominios que se disputan, no que se asientan (smear). **La habitación completa NO enciende direcciones mientras el sustrato siga siendo mundo-pequeño. El cabo métrico de CS066 es PRECONDICIÓN, no detalle. Espacio y dirección RE-ATADOS. → CS068.** REGISTRO v26 |
| **CS068** | análogo de INFLACIÓN — estirar-y-enfriar: ¿el mundo-pequeño TAPA un tejido métrico latente que el enfriamiento revela? | DISENO_CS068_inflacion_estirar_enfriar_CS.md / cs068_paso1_sintetico.py / cs068_paso2_mundo_ab.py / cs068_paso2b_diametro.py / ADJUDICACION_CS068_etapa1/paso1/paso2/paso2b | Hipótesis (Alexis: mayor distancia=menor T=menor energía): los atajos largos cuestan energía de correlación; al enfriar se rompen PRIMERO y se destapa un tejido con "lejos" real. Juez IRC-seguro (aporte paper Schwartz). Anti-Shannon: NULL corte-azar también estira (discriminante=trayectoria+gradiente, no estado final); T se MIDE de la estructura, no se impone | ✅/❌ **VEREDICTO MUNDO B — CIERRE DEL ARCO DEL ESPACIO (16-jul, CC ejecutó, CS auditó cada paso)**. Paso 1 (sintético, verdad de fondo): el mecanismo FUNCIONA — separa vs NULL, robusto a escala (la magnitud modesta ~0.06 era del estimador por-nodo; por-cascarón radial da −0.28, techo geométrico ~0.96). Paso 2 (blob real, soporte vs configuration-model NULL): z=122-300, PERO CS auditó que el CM-null mide CLUSTERING, no metricidad (falsos +/− verificados). Paso 2b (juez correcto=escalamiento del diámetro): tejido residual diám 6-7.5, ~13× MENOR que métrica 2D real → **mundo-pequeño hasta el fondo, NO hay geometría latente**. CC cazó su propio falso positivo (1 semilla dio "Mundo A", blindaje 4 sem lo revirtió). **El sustrato jamás tuvo métrica bajo los atajos. CS066(B)+CS067(B)+CS068(Mundo B) CONVERGEN: la distancia emerge de la diferencia, la dirección NO emerge de la relación pura en esta familia de modelos.** |
| **CS069** | frente cuántico v1 — ¿la dirección emerge de una SUPERPOSICIÓN de grafos donde ningún grafo definido la tiene? | DISENO_CS069_frente_cuantico_CS.md / cs069_quantum_graph.py / cs069_tanda.py / cs069_spotcheck_L12.py / ADJUDICACION_CS069_smoke_regla_fase / _tanda_cierre_CS.md | Matriz de amplitud A_ij=ρ·e^(iφ) (ρ de −log w del motor; φ fase relacional ciega en el enlace); distancia efectiva D_q=−log\|K_ij(L)\|, integral de camino sobre topologías. Propuesta de Gemini, auditada por CS (hueco de Shannon: la interferencia NO distingue atajo de local por sí sola → reformulada). Cuerda decisiva: COMPLETO vs NULL_FASE_TOPO (configuration-model, misma redundancia, fases decoherentes). G-FASE-CIEGA + G-NULL-CONSERVA-TOPOLOGÍA | ✅/❌ **VEREDICTO (B) CANÓNICO (17-jul, CC ejecutó, CS auditó cada paso)**. Smoke: regla de fase "media de vecinos" NO distinguía atajo/local (los arrastra a ambos a sync local); CS reformuló a FRUSTRACIÓN ENTRE EXTREMOS (un atajo puentea dos dominios de fase que sincronizaron aparte → queda frustrado) validada AUC 0.80→0.843. Tanda blindada 4 brazos × 8 sem × 3 N = 96 corridas: los cuatro brazos INDISTINGUIBLES en los 3 jueces, sin un solo indicio parcial (Juez A π-CV ~1.0-1.1 estalla igual; Juez B pendiente 0.13-0.23 <0.3; Juez C gap 0% certificado, n_ejes=0 en las 96). Spot-check confirmatorio L=12 (diagnóstico CS: L=8 acumula 97.9%, caminos largos decoheren MÁS): idéntico a L=8 (pico_medio 0.716=0.716, 0% cert). CC cazó 4 bugs propios antes de confiar y señaló su propia sospecha de la regla. **Mundo B se extiende al régimen cuántico: la superposición de fases relacional-ciega TAMPOCO enciende la dirección. El arco CS066-069 converge por DOS rutas independientes (clásica y cuántica): la distancia sobrevive, la dirección no aparece en ninguna.** ACOTA, no clausura: sitúa el siguiente frente fuera de estas dos rutas, no cierra la pregunta. Consecuencia medida: π/geometría no son condiciones predeterminadas, son estado condensado — extraordinario a secas. |
| **CS070** | semilla / amplificación — ¿una asimetría primordial mínima (semilla tipo CP, C-N2.5.5) se AMPLIFICA en direcciones múltiples estables donde la sopa simétrica no pudo? | DISENO_CS070_semilla_amplificacion_CS.md / cs070_*.py + _run.log / INFORME_CS070_tanda_PARA_CS.md / ADJUDICACION_CS070_tanda_semilla_CS.md | 4 brazos: SEMILLA coherente / SEMILLA_BARAJADA / SIN_SEMILLA / semilla+sustrato-local. Toy: retícula limpia PRESERVA el eje sembrado (Δ=+0.59), mundo-pequeño lo LAVA (Δ≈+0.017). Anti-Shannon: juez exige n_ejes>1 estable, NO coherencia (colapso a 1 eje = trampa CS067). Guardián G-JUEZ-NO-COHERENCIA. | ✅/❌ **VEREDICTO (B) CANÓNICO (17-jul, CC ejecutó 17.1 min, CS auditó con código)**. direccion_real=0.000 en las 96 corridas, 4 brazos, sin excepción — incluido el brazo semilla+local (que dio el n_ejes MÁS BAJO, no el más alto). CC cazó bug propio en ancla 1 con el guardián preinscrito: G-JUEZ-NO-COHERENCIA disparó (pico 0.86-0.93 con n_ejes=0 = dominios locales de alta confianza que agregados dan población isótropa); corrigió a criterio combinado (certificado Y n_ejes>1) y 3 anclas pasaron limpias. No escondió que SIN_SEMILLA tuvo el frac_certificado MÁS ALTO (0.375). **Matiz auditado por CS (código):** el 4º brazo aplicó gate k_local=4 al BLOB real, pero el diámetro escala como log N (22/25/30 para N=400/900/1600) — SIGUE siendo mundo-pequeño, NO se vuelve métrico (retícula limpia del toy: 38/58/78 ~√N). Es decir el 4º brazo NUNCA entregó un sustrato métrico → su negativo suma al muro (cuarto negativo del MISMO tipo) pero deja la ruta semilla+MÉTRICA-genuina SIN testear, pre-inscrita para cuando un experimento haga emerger un sustrato métrico real. **Tercer eje independiente (clásico sin-semilla + cuántico + clásico con-semilla) converge en el mismo muro. La semilla se lava en las tres rutas.** ACOTA, no clausura. Si la dirección necesitara métrica pre-existente, heredaría la contingencia de la métrica — misma exaptación un peldaño más arriba (C-N2.7.8), no ingrediente nuevo. REGISTRO v29 |
| **CS071** | histéresis / memoria-de-enlace — ¿la asimetría que FABRICA el proceso (transitar refuerza, no-usado decae+poda) auto-organiza métrica donde la sopa simétrica, la superposición y la semilla no pudieron? | DISENO_CS071_histeresis_memoria_enlace_CS.md / cs071_histeresis.py / cs071_tanda.py / cs071_tanda_resultados.json + _run.log / INFORME_CS071_tanda_PARA_CS.md / ADJUDICACION_CS071_tanda_histeresis_CS.md | 4 brazos: HISTÉRESIS (homeostática, ciega a geometría) / NULL_BARAJADO (misma magnitud de toques, aleatoriza cuáles) / SIN_PROCESO / HISTÉRESIS_SOBRE_RETÍCULA (control +). Juez = escalamiento del diámetro β + δ-Gromov. Toy CS pre-registrado: el paseo ciego carga los atajos 3.9× → predice (B). Guardianes G-PASEO-CIEGO, G-NO-AJUSTAR-CRONOGRAMA, G-NULL-MISMA-MAGNITUD, G-ANTI-HUB, G-CONECTIVIDAD. | ✅/❌ **VEREDICTO (B) CANÓNICO (17-jul, CC ejecutó 1.5 min, CS auditó JSON crudo de 96 corridas + código)**. β recalculado (regresión log-log del diámetro sobre 3 N): histéresis 0.154 ≈ null 0.132 ≈ sin_proceso 0.141 — los tres lejos de 0.5, mundo-pequeño. Control + histeresis_sobre_reticula β=0.482 (a un paso del ideal métrico). **Doble sello del juez:** δ-Gromov confirma por vía independiente — PLANO en 0.5 a los 3 N en los brazos WS (hiperbólico) vs CRECIENTE 0.38→1.0→1.75 en el control (firma métrica, como retícula 2D del arco). β y δ coinciden brazo por brazo → el juez ve métrica cuando la hay y su ausencia cuando no. Los 5 guardianes VERIFICADOS en código (no solo declarados): `_elige_vecino` no recibe posición (G-PASEO-CIEGO); cronograma REFUERZO=0.04/DECAY=0.99/PRUNE_FRAC=0.15/PASOS=30 global único en los 4 brazos; grado_max 8.96-9.12 en WS (no hub); frac_gigante 0.995-1.000. CC cazó bug propio ANTES de la tanda (poda en cascada: umbral relativo al grado ya reducido → grado 6→<1; corrigió a umbral fijo sobre peso original, recalibró declarando "evitar colapso, NO buscar √N"). Detalle menor anotado: docstring L21 aún describe la poda vieja, código L148 correcto. **La memoria de enlace NO metriciza — cuarta fuente de asimetría (tras estructura/fase/semilla) que el muro absorbe.** ACOTA, no clausura. REGISTRO v30 |

## 2. EL PRÓXIMO NÚMERO ES **CS072**
Estado del arco del espacio al 17-jul-2026 (CS068 CIERRA el arco clásico; CS069 régimen cuántico; CS070 semilla
primordial; CS071 memoria de proceso — SEIS rutas independientes CS066-071, mismo muro; tramo CS064-066 añadido 11-jul):
- CS052 corrió en dos versiones (v0 gauge-puro → v1 co-emergencia confirmada: el espacio vive en el
  vínculo atado).
- CS053 **corrió** (por CC): falsación honesta — la persistencia ciega mata lo frágil pero NO fija
  3D-plano; sobrevive todo retículo ≥2D por igual. Adjudicado en adjudicacion_CS053_CS.md.
- CS054 **corrió** (por CC): falsación ACOTADA — la gravedad SIN alcance colapsa todo ≥2D a blob. Casi
  tautológico (Alexis: gravedad uniforme = sin universo); le faltaba el ALCANCE. Adjudicado en
  adjudicacion_CS054_v2_ALCANCE_CS.md.
- CS054-v2 **corrió** (por CC): la gravedad CON alcance (cuadrado inverso por saltos de grafo) deja de
  colapsar y SELECCIONA geometría (positivo de mecanismo, intuición de Alexis probada) — PERO elige
  2D-plano, no 3D (cubo 3D muere 0/3, α-robusto); nuestro universo la falsa. Adjudicado en
  adjudicacion_CS054_v2_CIERRE_CS.md. Cuerda: el contador "d≈3" del clasificador NO es fiable, leer TIPOS.
- CS055 **corrió** (por CC): el PROCESO acoplado hizo VISIBLES las dos fuerzas — confinamiento-solo
  sostiene 3D (3/3, empuje arriba), gravedad-sola colapsa a 2D (empuje abajo). Reencuadre de Alexis
  (proceso, no sucesión) confirmado como método. PERO a fuerza igual (1:1) la gravedad domina → 3D no
  emerge. Adjudicado en adjudicacion_CS055_CS.md.
- CS056 **corrió** (por CC): las 4 fuerzas a intensidad física se reducen al confinamiento; el EM no
  rescata el 3D (a fuerza real inerte, a fuerza alta interfiere — hallazgo real: color y carga son dos
  neutralidades INDEPENDIENTES que se pelean, no se corrige alineándolas). Adjudicado en
  adjudicacion_CS056_CS.md.
- Hueco destapado por Alexis: gravedad y EM se corrieron con el MISMO alcance (D_MAX=2). La asimetría
  física real NO es la ley (idéntica, 1/d²) sino el ALCANCE EFECTIVO — la gravedad se acumula (largo), el
  EM se cancela por neutralidad (corto). CS056-v2 (alcances distintos) queda subsumido en CS057.
- CS057 **corrió** (por CC, 10.4h, 69.648 universos) y **ADJUDICADO** (adjudicacion_CS057_CS.md): el
  paisaje completo. TRES resultados reales, ninguno forzado: (1) **FALSACIÓN ACOTADA — el titular:** las
  fuerzas locales reales, todas juntas, barridas exhaustivamente con distancia por alcance, NO seleccionan
  el 3D-plano. El punto físico cae viable (0.375 vs 0.094 fondo, 4×) PERO estabiliza geometría CURVA
  (curv~0.84), no el 3D-plano (d3~0.15). Los dos brazos coinciden. (2) **Proceso:** sync>async +10%, z≈5 —
  sostenido en versión sobria (la sincronía importa, no es todo-o-nada). (3) **Sector oscuro:** aceleración
  emergente 2.4× cerca del físico, con G-NO-INSERTAR verificado en código (candidato honesto a energía
  oscura, no insertado). Predicción pre-registrada de Alexis: P1✅ (existe viable), P2✅-con-giro (físico
  viable pero NO en 3D-plano), P3✅ (región estrecha 11% = resonancia). Pend menor: CC reconcilia el
  "d3=0.00" del informe con las 236 filas d3-viable del CSV crudo (dirección idéntica, enunciado a afinar).
- **CIERRE DEL ARCO DE FUERZAS:** la unicidad de nuestro 3D-plano NO la fija ninguna fuerza local. Todo el
  arco (CG004 obstrucción global · CG005 confinamiento sin geometría · CS054-56 cada fuerza elige 2D/curvo ·
  CS057 el paisaje entero) apunta AGUAS ARRIBA, al **espín/marco (R7)** — el nodo nombrado desde CS054-v2.
- Hueco actual (post-CS057): **¿el espín/marco (R7) es lo que selecciona el 3D-plano que ninguna fuerza
  local elige?** Alexis pidió DISEÑAR AMBOS: CS058 (zoom denso al candidato de energía oscura — caracterizar
  o matar) y CS059=R7 (el espín como marco — EL experimento del arco). Ambos diseñados y a codear por CC.
- CS058 **diseñado** (DISENO_CS058_zoom_energia_oscura.md): zoom local denso a la aceleración emergente de
  CS057; brazo de resolución ×1/×2/×4 como falsación directa (si no sobrevive a más pasos = artefacto), región
  leída del CSV no elegida a mano, cruce con dominio-curvo para ver si conecta con R7. A codear por CC.
- CS059=R7 **diseñado** (DISENO_CS059_R7_espin_como_marco.md): el ingrediente que faltaba — el espín como
  MARCO/orientación intrínseca sobre el EDS de CG005, con el Burgers de CG004 como juez ciego. Prueba si el
  MARCO (no la fuerza) selecciona una dimensión. Éxito = selección consistente que colapsa bajo NULL, NUNCA
  "salió 3D" (G-NO-FORZAR-3D). Autoría del ingrediente: Alexis. A codear por CC. Es el experimento al que
  converge todo el arco (CG004+CG005+CS054-57).

- CS060=leptones+masa **diseñado** (DISENO_CS060_leptones_y_masa.md): el leptón como marco SIN color
  (electrón=control que aísla marco vs ligadura, el confound que CS059 no puede separar solo), más el eje de
  MASA a las tres generaciones reales (e/μ/τ). **Corrección importante asentada: el veto a la masa/Higgs lo
  puso el equipo (CC/Grok/CS) desde el comienzo, NUNCA Alexis — queda corregido, la masa entra.** La masa se
  exapta como inercia de orientación + persistencia temporal del marco, no como número mágico ni como
  coordenada. A codear por CC, tras/junto a CS059.
- Hueco tras CS059/CS060: si el marco (espín) y/o la masa (inercia del marco) seleccionan geometría, el
  vínculo con el vértice de 3 puntos (gluón/Higgs, la pared R7 ya cerrada de CG002) se vuelve la pregunta
  siguiente.
- **CS058-CS061 CORRIERON (CC) y ADJUDICADOS EN BLOQUE** (adjudicacion_ARCO_CS058-061_CS.md): el arco de
  cierre. CS058 artefacto · CS059 marco-pareado NO (falso positivo cazado) · CS060-A masa toca coherencia no
  geometría · CS060-B falso positivo cazado por NULL · CS061 vértice 3-puntos colapsa bajo NULL, espectro
  trivial. **NEGATIVO GRANDE:** ni fuerza, ni marco pareado, ni masa (dada o emergente-con-dinámica-pareada)
  seleccionan la dimensión. DOS falsos positivos cazados por los controles del propio equipo (anti-Shannon
  funcionando dentro, no impuesto desde fuera).
- **CAVEAT que CS elevó a condición (verificado en el código de CS061):** el update del marco es campo-medio
  PAREADO (cada nodo → media de vecinos); la inercia de 3 cuerpos solo amortigua un escalar. El vértice de 3
  cuerpos está en la MEDICIÓN (defecto de tríada), NO en la DINÁMICA. Por eso CS061 NO cierra el 3-puntos: es
  (C) para "inercia amortigua relajación pareada" y deja ABIERTO (D) = un update 3-cuerpos genuino.
- **GRIETA POSITIVA (de Alexis, doble):** el proxy de grado de la gravedad de CS054-057 sesgaba contra el 3D;
  con gravedad ∝ peso-intrínseco (como la real) el 3D/4D es ~2× más viable → relee el negativo central.
- **DOS EXPERIMENTOS QUE EL ARCO ABRE (prioridad de CS):** CS062 (★ re-correr el paisaje de CS057 con
  gravedad ∝ peso-intrínseco, no grado — barato, ataca el hallazgo más concreto) y CS063 (el vértice
  3-cuerpos GENUINO — update donde la tríada se mueve junta, no cada nodo hacia la media — cierra C vs D de
  CS061). Hipótesis de fondo nombrada: la dimensión podría ser CONTINGENTE, no seleccionada (se gana el
  derecho a defenderla solo tras CS062 y CS063).

- CS062 y CS063 **diseñados** (los dos que la adjudicación del arco abrió): CS062=paisaje con gravedad∝peso
  (★ prioridad 1, barato, relee el negativo central) y CS063=vértice 3-cuerpos genuino (prioridad 2, caro,
  cierra el (D) que CS061 dejó abierto). A codear por CC.
- **CS063 CORRIÓ (CC) y ADJUDICADO — NEGATIVO (B), verificado por CS en código y datos:** G-IRREDUCIBLE pasa
  (∂³E=1.96≠0, producto triple escalar; update mueve los tres marcos, sin término pareado — 3-cuerpos GENUINO,
  no CS061 con otro nombre). Colapsa bajo NULL (3cuerpos≈null_marco≈null_triada). **Ni el vértice de 3 cuerpos
  genuino selecciona la dimensión.** Cierra el (D). **El ARCO DE ELIMINACIÓN LOCAL está COMPLETO:** ni fuerza,
  ni marco pareado, ni masa (dada/emergente), ni vértice 3-cuerpos genuino → la hipótesis **la dimensión es
  CONTINGENTE (no seleccionada por ningún ingrediente local, persistió una de muchas posibles)** se gana el
  derecho a ser la conclusión del arco. Encaja con la imagen de Alexis desde el principio (el cedazo de π).
- **CS058 corregido con la corrida completa (1404 pts):** de "artefacto firme" (parcial) a REAL-PERO-DÉBIL
  (supera NULL ratio 1.66, pero decae con resolución y desacoplada de R7). Lección asentada: no declarar
  veredicto firme desde un parcial — CC la destapó, CS la cometió en la v2 de su adjudicación y la asume.
- **CS062 CORRIÓ (CC) y ADJUDICADO — NEGATIVO-MATIZADO, verificado por CS en código y datos (52.248 filas):**
  gravedad ∝ peso-intrínseco (Newton m·m/d²) vs grado (=CS057) vs null_peso (masas barajadas). Guardián
  corr_ok=1 en el 100% (peso separado del grado). **Leído contra las dos salidas pre-inscritas:** (1) el proxy
  de grado SÍ inflaba el negativo — 3D/4D global sube de 11.0% (grado) a 16.2% (peso), grieta CS060-B
  confirmada; PERO no lo destapa (d4≳d3, punto físico 3D+4D=0% estabiliza curvo, muro de direcciones en pie).
  (2) el núcleo se sostiene: ningún acople local privilegia el 3D-plano. **Hallazgo decisivo: NO es la masa —
  null_peso≈peso (Δ≈0)**, cualquier peso fijo desacoplado del grado da lo mismo; lo que aliviaba era la FORMA
  del acople (no-auto-amplificar-por-grado), no la identidad de la masa. **El asterisco no se elimina, se
  transforma:** el negativo de CS057 estaba medido con un proxy que lo exageraba, pero corregido el proxy el
  fondo negativo se sostiene. **Arco de eliminación local COMPLETO, SIN salvedades vivas** → la contingencia de
  la dimensión queda FIRME. adjudicacion_CS062_CS.md.

- **TRAMO CS064-066 (añadido 11-jul-2026; datos de CC, AUDITADO por CS sobre los CSV — CS064 blob, CS065b y CS066 verificados con cómputo propio; CS065 adjudicado como árbitro):**
  CS064 corrió (blob ultra-mundo-pequeño + exaptación del marco confirmada por null_marco→0 ejes) → destapó
  que NO hay "lejos": el arco peleaba el nivel de las DIRECCIONES sin un espacio local contra el cual
  distinguirlas. CS065 y CS065b metieron el anti-colapso por EXCLUSIÓN (Pauli, en dos formas) → ambos
  FALSIFICADOS (excl≈barajada; la 065b por su salida (C) pre-inscrita, sin duelo). La exclusión se retira.
  CS066 dio el giro (Alexis: "el tejido primero"): la LOCALIDAD en la formación. Smoke decisivo: un podador
  externo NO tiene punto fijo de tejido (o blob o gas) → la localidad debe gobernar la PERSISTENCIA del
  enlace (Quantum Graphity). Resultado **(B)**: sobre un fondo local que SÍ emerge con especificidad (d_s~3
  estable, clustering 4× el placebo, diam 3× el blob), el colapso-a-1-eje PERSISTE. **La auditoría de CS
  endurece el (B):** sobre el tejido, local no solo NO supera a barajado — va significativamente PEOR
  (Δ n_ejes = −0.61/−0.67 en N=2500/3500, p=0.059/0.036); el tejido apretado SUPRIME ejes. El colapso-a-1 es
  más profundo que la falta de localidad: sobrevive —y se agrava— sobre un fondo local. **Espacio y
  direcciones son problemas SEPARADOS** — reordena el arco: el anti-colapso vuelve a tener sentido, pero ahora
  sobre un fondo local, no en el blob. El sector oscuro (antes rotulado 66) se DIFIERE: no se andamia lo que
  aún no tiene espacio local. Pendiente: confirmatorio de Nivel 1 de CS066 (malla k_local × N, más parches a
  baja k para clavar el exponente diam~N^(1/d) — hoy la única pata floja del "hay tejido") y el rediseño del
  anti-colapso Nivel 2 sobre fondo local (→ CS067, a diseñar con calma a la luz de (B), no improvisar).

Cualquier experimento nuevo posterior arranca en **CS067** y sigue correlativo. Su diseño debe abrir con
una línea del tipo:
> *"CS0NN — aquí probamos [X] (dimensión técnica: [etiqueta])."*

- **CS067 — "la habitación completa" — VEREDICTO (B), auditado y firmado por CS (16-jul):** los 17 ingredientes
  del arco juntos (12 heredados en el motor + Pauli re-incluido + los 3 del video [correlación-métrica, cono
  causal, SSB-discreto] + sector oscuro emergente), voto Potts pesado por correlación (rumbo (a),
  DISENO_CS067_habitacion_completa_CS.md). Recorrido de diseño (4 iteraciones, cada fallo cazado y adjudicado):
  (i) juez de gap NO distingue K-discreto de K-continuo → candado de picado por nodo (ADJUDICACION_CS067_SSB v3);
  (ii) SSB con snap a pozos fijos HORNEA K → realización Potts/reloj (mayoría de vecinos); (iii) Potts×cono
  colapsa a 1 porque los atajos de mundo-pequeño de CS066 dejan percolar el consenso → rumbo (a): voto pesado por
  w (ADJUDICACION_CS067_bifurcacion_ab); (iv) el pico "por nodo" estaba mal implementado (varianza global) y el
  gap_val era artefacto de borde-de-rango → candado real + guarda de rango + criterio de 3 condiciones sin gap
  (ADDENDUM_CS067_pico_guarda_rango v2). **Resultado final, blindado (160 corridas, 16 semillas/γ completo + 8/γ
  por control, γ∈[0.5,2.5], log cs067_gamma_sweep_blindaje.log, auditado por CS sobre 80 semillas del brazo
  completo):** en NINGÚN régimen de γ el brazo completo cumple las 3 condiciones. (ii) pico_medio nunca cruza el
  piso 0.85 — IC95%_inf máximo 0.789 (γ=2.0); pico global 0.782. (iii) anti-especificidad limpia: los controles
  igualan o SUPERAN a completo (γ=1.5 sin_correlacion 0.789≈completo 0.786; γ=2.5 sin_causal 0.804>completo
  0.778). n_ejes disperso (CV 0.35–0.81, 1–6/16 semillas colapsan a ≤1 eje por régimen): **dominios que se
  disputan, no que se asientan — smear, no direcciones.** **(B) CANÓNICO:** la habitación completa NO basta para
  encender direcciones múltiples mientras el sustrato siga siendo mundo-pequeño. El cabo métrico de CS066 no era
  detalle pendiente — es la PRECONDICIÓN de las direcciones (espacio y dirección quedaron RE-ATADOS: el mismo
  atajo que infla d_s mata los dominios Potts). Reorienta el arco → CS068 candidato: cerrar el cabo de
  mundo-pequeño (análogo de inflación, el #1 que Grok priorizó — estiramiento que abre "lejos" real).
- **CS068 — análogo de inflación (estirar-y-enfriar) — VEREDICTO MUNDO B, CIERRA EL ARCO DEL ESPACIO (16-jul,
  CC ejecutó, CS auditó cada paso):** hipótesis (Alexis: mayor distancia=menor T=menor energía) — los atajos
  largos cuestan energía de correlación, al enfriar se rompen primero y destaparían un tejido con "lejos" real.
  Paso 1 (sintético con verdad de fondo): el mecanismo FUNCIONA (separa vs NULL; la magnitud modesta ~0.06 era
  del estimador por-nodo, el por-cascarón radial da −0.28). Paso 2 (blob real, soporte vs configuration-model):
  z=122-300, PERO CS auditó que ese test mide CLUSTERING, no metricidad (falso positivo: small-world clusterizado
  lo pasa; falso negativo: retícula 2D pura lo reprueba). Paso 2b (juez correcto = escalamiento del diámetro del
  tejido residual): diám 6-7.5, ~13× MENOR que una métrica 2D real (~58-97) → mundo-pequeño hasta el fondo, NO hay
  geometría métrica latente que el enfriamiento pueda revelar. CC cazó su propio falso positivo (1 semilla dio
  "Mundo A", el blindaje de 4 semillas lo revirtió; pendiente 0.218<0.3, y la magnitud absoluta cierra donde la
  pendiente dudaba). **CIERRE DEL ARCO: CS066(B) + CS067(B) + CS068(Mundo B) CONVERGEN. La distancia emerge de la
  diferencia (hay "lejos"), pero NO cuaja en dimensión ni dirección — el "hacia dónde" no emerge de la relación
  pura en esta familia de modelos. Distancia y dirección son SEPARABLES.**
- **CONSECUENCIA — π CONTINGENTE (hallazgo medido, 16-jul, HALLAZGO_pi_contingente_y_rumbo_gravedad_cuantica_CS.md):**
  si la distancia existe pero no cuaja en dimensión/dirección, entonces π NO estaba predefinido. Medido (π como
  frontera-de-bola/diámetro en grafos): donde hay geometría π es CONSTANTE pero su valor depende de la red
  (cuadrada 2.0, triangular 2.99, hexagonal 1.5); donde NO la hay (mundo-pequeño) π ESTALLA (2.5→48, indefinido).
  π es una huella de la geometría que cuajó, no una ley previa — falsable contra la física estándar (que trata π
  como verdad necesaria anterior a todo universo).
- **CS069 — frente cuántico v1 — VEREDICTO (B) CANÓNICO, CERRADO (17-jul, CC ejecutó, CS auditó cada paso):**
  con los 18 elementos el universo no cuaja en geometría → el arco llegó, por su cuenta, al MISMO límite de la
  física actual: el punto que conecta Big Bang con agujeros negros, el régimen pre-geométrico donde
  tiempo/espacio/dimensión/dirección aún no emergen = un universo cuántico. La pregunta que el arco no podía
  formular con ingredientes clásicos: si la dirección no emerge de estados DEFINIDOS, ¿emerge de una SUPERPOSICIÓN
  de grafos? Propuesta de Gemini (matriz de amplitud + integral de camino sobre topologías), auditada por CS
  (hueco de Shannon cazado: la interferencia no distingue atajo de local sola → reformulada a frustración entre
  extremos, AUC 0.843). Tanda blindada (96 corridas): los 4 brazos INDISTINGUIBLES en los 3 jueces, sin indicio
  parcial; spot-check L=12 idéntico a L=8. **Mundo B se extiende al régimen cuántico: la superposición de fases
  relacional-ciega TAMPOCO enciende la dirección. El arco CS066-069 converge por DOS rutas independientes,
  clásica y cuántica.** Lenguaje de cierre (calibrado con Alexis): el muro ACOTA, no clausura — sitúa el
  siguiente frente FUERA de estas dos rutas (un ingrediente categóricamente ausente, o una capa donde la
  dirección se define en vez de emerger), NO afirma que nada exista tras él. Cautela mantenida: NO meter
  loop/cuerdas/CDT como ingrediente 19 para que "salga" 3D (=Shannon). **Lo positivo del arco (no lo que faltó):
  π y la geometría NO son condiciones predeterminadas — son estado condensado, emergen o quedan indefinidos según
  lo que persista. Hecho medido y falsable. Extraordinario a secas.**
- **APORTE DE GROK (12-jul, mapa del *después*, adjudicado por CS — NO altera CS067):** mapeo 17-ingredientes ↔
  relato del Big Bang estándar; huecos reales priorizados por relevancia a espacio+direcciones: #1 inflación
  (estiramiento que congela diferencias), #2 transición de fase con "antes/después" (que el SSB nazca EN una
  transición, no de adorno), #3 semillas de irregularidad, #4 bariogénesis/quiralidad, #5 era de radiación, #6
  Λ tardía. **Adjudicación CS:** (a) el #2 de Grok YA lo realiza el mecanismo Kibble-Zurek de CS067 (convergencia
  independiente video→CS y Big Bang→Grok a la misma pieza — señal fuerte); (b) el #1 (inflación) es el candidato
  natural a CS068 SOLO si el veredicto de CS067 es (B) y la distancia-por-correlación NO cierra los atajos
  globales; (c) #4/#5 son otra pregunta (identidad de la materia / termodinámica), CS0xx paralelo cuando el
  espacio esté firme; (d) nucleosíntesis/CMB/galaxias = validación mucho más abajo. Método confirmado: NO ampliar
  a 22 ahora; cerrar la habitación de 17 primero. **Cautela CS:** las afirmaciones de Grok de que bariogénesis
  "se exploró en CG002" y quiralidad "en R7" NO están auditadas contra este registro — verificar antes de
  apoyarse en ellas para un CS0xx. Candidatos CS068+ a redactar (en lenguaje de la Teoría) DESPUÉS del veredicto
  de CS067, no antes.
- **CS070 — semilla/amplificación — VEREDICTO (B) CANÓNICO, CERRADO (17-jul, CC ejecutó 17.1 min, CS auditó con
  código):** la Teoría ya tenía escrita la salida del muro — C-N2.5.5 dice que una asimetría primordial mínima
  (semilla tipo CP) fue necesaria. CS070 la probó: 4 brazos (semilla coherente / barajada / sin semilla /
  semilla+local). Resultado: direccion_real=0.000 en las 96 corridas, sin excepción. La semilla se lava en el
  mundo-pequeño igual que la sopa simétrica. CC cazó bug propio con el guardián preinscrito (G-JUEZ-NO-COHERENCIA:
  pico alto + n_ejes=0 = isotropía disfrazada de coherencia) y NO escondió que SIN_SEMILLA tuvo el
  frac_certificado más alto — el juez funcionando. **Matiz que CS auditó con código y NO hay que perder:** el 4º
  brazo aplicó un gate k_local=4 al blob real, pero el diámetro sigue escalando como log N (mundo-pequeño), no
  como √N (métrico) — nunca entregó un sustrato métrico. Su negativo suma al muro (tercer eje independiente:
  clásico-sin-semilla + cuántico + clásico-con-semilla, mismo muro) PERO deja la ruta semilla+MÉTRICA-genuina sin
  testear. Queda pre-inscrita: si algún experimento hace emerger un sustrato métrico real, la pregunta "¿la
  semilla prende ahí?" ya está formulada con su predicción del toy (retícula limpia preserva el eje, Δ=+0.59) y su
  NULL. Consecuencia teórica: si la dirección necesitara métrica pre-existente, heredaría la contingencia de la
  métrica — misma exaptación un peldaño arriba (C-N2.7.8), no ingrediente nuevo.
- **CS071 — histéresis/memoria-de-enlace — VEREDICTO (B) CANÓNICO, CERRADO (17-jul, CC 1.5 min, CS auditó JSON
  crudo + código):** era el candidato más original de la batería de Gemini (Test 2.2): sin semilla ni estructura
  privilegiada al inicio, la asimetría —si aparecía— la fabricaba el propio proceso (transitar refuerza,
  no-usar decae+poda). No apareció. β=0.154 (histéresis) ≈ 0.132 (null) ≈ 0.141 (sin_proceso), todos
  mundo-pequeño; control positivo sobre retícula limpia β=0.482 (métrico) valida que el juez ve métrica cuando
  existe. Doble sello: δ-Gromov confirma por vía independiente (plano 0.5 en WS, creciente 0.38→1.75 en el
  control). Los 5 guardianes verificados EN CÓDIGO. CC cazó bug propio (poda en cascada) antes de la tanda. CS
  pre-registró (B) con un toy (el paseo ciego carga los atajos 3.9× → refuerza justo lo que habría que podar);
  se sostuvo. **Cuarta fuente de asimetría —tras estructura, fase, semilla— que el muro absorbe. SEIS rutas
  independientes (CS066-071) al mismo muro.**
- **PRÓXIMO NÚMERO CS072 — candidatos vivos (de la auditoría de la batería topológica de Gemini; Test 2.2 ya
  ejecutado en CS071):**
  (b) **Test 1.2 — asimetría acoplada a exergía:** el costo de un enlace depende de la energía disponible. Riesgo
  Shannon ALTO (una "penalización por distancia" impone T(r) = mete la métrica a mano); solo con guardián que
  mida el costo de la estructura, no de una distancia pre-supuesta.
  (c) **Bloque 3 (APLAZADO) — conservación de carga/color/espín:** el más profundo (podría romper el muro con
  leyes físicas reales, ligado a la rigidez de 3-cuerpos de CS063) pero el más propenso a Shannon; necesita
  guardián dedicado para que el veto no lea la geometría-objetivo. NO antes de agotar (a)/(b).
  Regla heredada (Pauli CS065b): aislar cada mecanismo antes de cruzarlos (Bloque 4). CS070 (asimetría de
  condición inicial) ≠ Bloque 1 (asimetría estructural permanente) — complementarios, no se solapan.

## 3. NOTAS DE HONESTIDAD SOBRE ESTE REGISTRO
- El **orden CS es por arco** (CG001→002→003→004→005), que es el orden lógico de construcción; NO
  reconstruí la cronología exacta día-a-día (parte está detrás de sesiones que no tengo completas).
  Si el equipo tiene fechas exactas, se pueden añadir como columna sin cambiar los números.
- Los **veredictos** vienen de: INFORME_CG002_VEREDICTOS_TABLA.md (CS008-025), síntesis R7 en
  CC_TO_GROK.md (CS026-033), y mis propias adjudicaciones auditadas contra código (CS039-051).
- Algunos CS agrupan variantes del mismo experimento (p.ej. CG003f + _carnets + _b) — se anota en la
  columna archivo. Si prefieres un número por variante, se expande sin romper la secuencia.
- Este registro es EDITABLE y vive en Cosmogénesis/. Cuando el equipo cierre CS052+, se agrega la fila.

— CS. Numeración a pedido de Alexis: secuencia simple, única, reiterada.
