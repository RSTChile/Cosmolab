# Auditoria arqueologica funcional de VSTCosmo / Celula Madre

Fecha: 2026-06-24  
Modo: solo lectura / arqueologia funcional. No se modifico codigo de Cosmolab.

## 0. Alcance, metodo y evidencias

Se revisaron 325 scripts relevantes (`vXXX*.py`, `VXXX*.py`, `VST_*.py`) en `/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo`, mas informes internos y manifiestos actuales.

Evidencias principales:

- `build_vstcosmo_db.py`: cronologia por ciclos y veredictos de hitos.
- `INVENTARIO_RESCATE.md`: inventario previo de capacidades perdidas, subsumidas o inertadas.
- `INFORME_UNIFICADO_resuelto_perdido_olvidado.md`: diagnostico de lo resuelto, perdido y olvidado.
- `INFORME_CELULA_MADRE_Cosmosemiotica.md`: arquitectura del Genoma / organelos.
- `INFORME_BATERIA_Test.md`: estado funcional de la Celula Madre con audio real.
- `INFORME_LIVE_CelulaMadre.md`: interfaz live y entrada binaural / dos fuentes.
- `ROADMAP_ANIMA_V182_V183_rev1_verificado.md`: estado verificado de V176-V182.
- `_docscan/INFORME_FINAL_V180___MEMORIA_EPIS_DICA_pdf.txt`: V180c.
- `_docscan/Informe_VST_ANIMA4_182A5_Comunicaci_n_Bidireccional_pdf.txt`: V182A5.
- `VST_Genoma.py`, `Célula_Madre_Funcional_001.py`, `VST_Celula_Madre_001.py`, bloques 5/7/8 y Homeostasis.

Archivo auxiliar generado localmente, sin tocar el proyecto:

- `work/inventario_scripts_vstcosmo.tsv`: inventario mecanico de 325 scripts, con version, archivo, clases, funciones y tokens funcionales.

Leyenda de estado:

- OK: validado por informe, log o veredicto interno.
- Parcial: funciona una conducta, pero no todo el mecanismo reclamado.
- Superado: corregido por una version posterior.
- Subsumido: la funcion sobrevive dentro de otra estructura, no como organo propio.
- Inertado: existe nombre/codigo, pero ya no mueve conducta.
- Ausente: probado o explorado historicamente, no activo en el organismo actual.
- Desconocido: no hay evidencia suficiente en esta pasada.

Principio aplicado: toda capacidad historicamente validada se presume recuperable salvo evidencia explicita de poda deliberada.

---

## Fase 1 - Inventario historico funcional

### Tabla maestra resumida por ciclo

| Rango | Capacidad dominante | Resultado | Estado |
|---|---|---|---|
| V001-V005 | Validacion con grabaciones reales, umbrales y mapas iniciales | Se establecen entradas reales, correcciones de umbral y sensibilidad minima | OK / base |
| V006-V019 | Campo Phi minimo, acoplamiento, competencia global, barridos multidimensionales | Nace la dinamica de campo, inercia, competencia y supervivencia de configuraciones | OK / fundacional |
| V020-V027 | Memorias tempranas, restriccion, trayectoria de atencion, refuerzo por curvatura | Aparecen memoria multiple, restriccion y direccion atencional incipiente | Parcial / exploratorio |
| V028-V039 | Homeostasis, metabolismo de experiencias, preferencias, anclaje metastable | Se valida que el sistema pueda conservar rango, preferir y estabilizar configuraciones | OK / varias piezas subsumidas |
| V040-V058 | Miedo, valor, seleccion autonoma, conviccion, identidad Omega | Se desarrolla valencia, amenaza, seleccion, plasticidad asimetrica e identidad | OK / repartido en espina actual |
| V059-V069 | Campo espectral, zona fertil, oscilador, firma oscilatoria | Se exploran modos/frecuencias naturales y firma del sistema | Parcial; modos espectrales podados, oscilador recuperado |
| V070-V079 | Campo continuo C-N2.0, plasticidad hebbiana, HETA, LF activa, homeostasis estructural | Nace el campo continuo robusto y memoria de configuraciones | OK; varios mecanismos luego perdidos/recuperados |
| V080-V096 | Memorias duales, ganglio, actuadores, decision interna, identidad ciclica | W_prof/W_rec, exploracion de actuadores, orientacion y cierre de identidad | OK parcial; ganglio podado; dualW recuperada pero silenciada |
| V097-V117 | Cartografia Omega, dos agentes, clasificacion, R2 / sensibilidad relacional | Se valida sensibilidad relacional y cartografia de multiples estimulos | V117 OK; R2 como capacidad medida no sobrevive igual |
| V118-V121 | Arquitectura bihemisferica, atencion, cuerpo calloso, consolidacion doble canal | Coexistencia de hemisferios, acoplamiento selectivo y lateralidad | OK parcial; atencion/calloso escalar eliminados |
| V122-V150 | ANIMA-1: segregacion fisica, cabeza/orientacion, control, memoria, cierre de etapa | Coexistencia R2 + lateralidad; motor orientado; fatiga y cierre organismico | V122/V132/V147/V150 OK; maquinaria V122 luego parcialmente perdida |
| V151-V176 | ANIMA-2: ausencia, Cb, juego, ritual, meta, primer No | Se valida cuerpo individual con ausencia, ritual, metaobservacion y negacion operativa | V176 OK; CbGlobal perdida |
| V177-V181 | ANIMA-3: rechazo, extincion, conflicto, memoria episodica, afirmacion | V180c valida memoria episodico-valencial; V177-V179/V181 quedan diferidos segun roadmap | V180c OK parcial; otros diferidos/desconocidos |
| V182A | Relacional canonica sobre cuerpo V180: roles, transferencia, acumulacion | A.3 roles por valencia, A.4 transferencia, A.5 cultura acumulativa | OK |
| V182B | Comunicacion funcional A->B / bidireccional sin cuerpo V180 | B-v9 valida comunicacion con nulo por setpoint; B.1 no reverificado | OK / B.1 desconocido |
| V182C | Sentido compartido / convencion en modelo reducido | Comunicacion necesaria y suficiente; memoria no agrega en esa tarea | OK en modelo reducido |
| V182D-E-F/V183 | Alteridad, negociacion, empatia, irreductibilidad | Pendiente segun roadmap | Pendiente |
| VST_Celula_Madre_001 | Consolidacion monolitica del organismo individual | All-on viable; flags conmutables; campo v2 con piezas estables | OK parcial |
| VST_Genoma + bloques | Genoma de organelos, B5/B7/B8, Homeostasis, OI | Arquitectura modular presente; varias capacidades aun viven en monolito o manifesto | OK arquitectural |
| Celula_Madre_Funcional_001 | Audio real -> Soma -> organelos -> OI | Carga WAV, binaural, dos fuentes, ablacion, determinismo | OK; timbre no separado por Omega; lateralidad interna pequena |
| WebLive + RC + comunicacion | Laboratorio live, dos organismos, organo RC, voz interorganismo, cabeza 3D | Ya corre; RC y comunicacion integrados en interfaz; cabeza aun en calibracion experimental | OK experimental / en ajuste |

### Hitos con veredicto documental fuerte

| Version | Capacidad | Resultado | Estado |
|---|---|---|---|
| V103 | Clasificacion de multiples estimulos | Veredicto interno: parcial | Parcial |
| V117 | Sensibilidad relacional / tiempos completos | Veredicto interno: OK | OK |
| V118 | Atencion / hemisferios tempranos | Funcion historica divergente; no sobrevive como atencion real | Eliminada / reconstruir |
| V122 | ANIMA-1, lateralidad + R2 por segregacion funcional | Climax: coexistencia R2 + lateralidad; maquinaria eliminada en V123 | OK, parcialmente perdida |
| V132 | Motor/orientacion | Motor de orientacion validado | OK, presente |
| V147 | Control / correccion avanzada | Hito OK en clausura ANIMA-1 | OK |
| V150 | Cierre ANIMA-1 | OK con residuo de recuperacion de fatiga (-6%) | OK parcial |
| V176 | Negacion operativa R_op | Validada como cuerpo individual | OK |
| V180c | Memoria episodico-valencial | Rechazo +45: 0%; latencia 12.58x; recall explicito 0/50 | OK parcial |
| V182A3 | Roles emergentes por valencia | Roles aparecen con memoria/confianza; corregido respecto A.2 | OK |
| V182A4 | Transferencia mutua dirigida | Alumno aprende, maestro preserva mas en ON que OFF | OK |
| V182A5 | Cultura acumulativa | ON min final 11.94 vs OFF 6.47; retencion 89% vs 70% | OK |
| V182B-v9 | Comunicacion A->B | Efecto genuino > nulo por setpoint | OK |
| V182C | Sentido compartido | Convencion emergente en modelo reducido | OK en modelo reducido |
| Celula Madre funcional | Audio real y organelos | Diferenciacion estructural por XE/R2/OI; determinismo bit a bit | OK |

---

## Fase 2 - Extraccion de organos funcionales historicos

| Organo funcional | Origen | Mecanismo | Variables principales | Estado actual |
|---|---|---|---|---|
| Campo Phi | V006-V019, robustecido V070/V122 | Medio excitable por hemisferio; difusion + reaccion + forzamiento | Phi, omega, gradiente, A_sys-env | Presente y expresado via Soma/CM001 |
| Plasticidad hebbiana W | V072b | Ajuste estructural del campo | W, Phi, aprendizaje | Presente en manifesto/campo |
| Atractor de historia | V072c/V080h | Phi_int_historia reinyectado | Phi_int_historia, gamma | Presente en campo; expresion conductual aun limitada |
| Oscilador / modos naturales | V070 | Resonancia por nodo | frecuencias naturales, Phi_vel | Recuperado, presente en campo |
| W dual identidad/contexto | V080h | W_prof lenta + W_rec rapida | W_prof, W_rec, tau_prof/tau_rec | Presente pero por defecto silenciada |
| Olvido selectivo | V080h | Olvido modulado por eficiencia | eficiencia, W, GED | Ausente |
| Membrana sensorial | V111 | Preprocesamiento inst/envolvente/derivada/tanh | dS, envolvente, derivada | Ausente |
| Lateralidad bihemisferica | V118-V122 | Hemisferios L/R acoplados | omega_L/R, balance, coherencia | Presente; expresion conductual en calibracion |
| Cuerpo calloso escalar | V123 | Transferencia direccional rectificada | diff, max(0,diff) | Podado; funcion parcial inline |
| Inhibicion reciproca | V123 | Winner-take-all duro | inhibicion, Omega | Podada |
| Inhibicion lateral | V122 | Congela lento ante cambio rapido | dOmega/dt, buffer rapido | Presente parcial; atribucion no limpia |
| Lambda nativa / LF atractores | V122 | delta_struct * LF / e_R | historial_omega, LF, Lambda | Presente parcial; plano en sonda |
| Orientacion gradiente | V132/V147 | Driver unico por omega_A - omega_B | gradiente, Kp, orientacion | Presente; cabeza live lo usa/modula |
| Actuadores exploratorios | V081/V097 | act_busc, act_perm, act_geom, act_mant | alpha, asimetria, varianza | act_perm ausente; act_geom subsumido |
| Permeabilidad activa | V081 | Modula acople al estimulo desde campo | alpha, act_perm | Ausente candidata |
| Memoria de configuraciones | V023/V031/V072 | Persistencia de patrones | patrones, historial | Subsumida en campo/atractor |
| Memoria de ausencia | V153/V155 | Confianza decae en silencio | confianza, setpoint | Presente |
| Memoria de trabajo | V176/V180 | Deliberacion y opciones | opciones, valencias, costo | Presente/subsumida |
| Memoria episodico-valencial | V180c | Episodio marcado altera valencia local y veto | episodios, valencia, latencia | Presente parcial |
| Memoria contextual | V180a/b | Contexto A/B pretendido | contexto, setpoint | No validada; pendiente |
| Memoria relacional | V182A3-A5 | Confianza sigmoide por competencia del otro | val_i, val_j, conf | Ausente del Genoma activo; probada historicamente |
| Buffer de acoplamiento | V182A-v3 | Trayectoria del otro (val,Cb,D) | val, Cb, D, distancia | Ausente; version rica perdida |
| Memoria largo plazo V182 v10 | V182A-v10 | Media por audio con contador defectuoso | media, contador | Podada/trivial |
| Homeostasis | V028-V039, VST_Homeostasis | Mantener variable en rango viable | H_homeostasis, x_interna | Presente y expresada |
| CbGlobal | V174 | Presion global de desacople | CbGlobal, presion global | Ausente candidata |
| Fatiga | V140/V150/V155/V180c | Historia irreversible + fatiga recuperable | historia, fatiga_activa | Presente parcial |
| Saciedad / metabolismo de experiencias | V028-V039 | Regulacion por consumo/experiencia | energia, saciedad, metabolismo | Subsumida en homeostasis/fatiga |
| Preferencia / valencia | V035/V040-V058/V172 | Valor diferencial por experiencia | valencia, preferencia, recompensa/costo | Presente |
| Miedo / amenaza / alerta | V044-V045, V182 amenaza | Riesgo / arousal frente a amenaza | miedo, amenaza, alerta, IRDE | Parcial: RC/IRDE actual; historico disperso |
| Presion de desacople | V155/CM001 | e_R * (1-A) integrado | presion_desacople | Presente |
| Consciencia basica | Bloque 5 | C_b = |R1| | C_b, C_b_norm | Presente |
| Meta-representacion R2 | V097-V117, Bloque 5 | Representacion de representacion | R2, R2_meta | Presente |
| Self | Bloque 5 | Coherencia de auto-modelo | self_coherencia | Presente |
| Juego | V156/V165 | Accion como-si bajo desacople | juego_activo, INR | Presente |
| Ritual | V158-V167 | Patron repetible estabilizador | ritual_activo, cruces | Presente |
| Meta Rᴿ | V167 | Observa desajuste sin inhibir | desajuste, meta | Presente |
| Negacion operativa / No | V176 | Veto sobre representacion/accion | R_op, veto, INR | Presente |
| Afirmacion R_af | V181 | Afirmacion funcional | R_af | Diferida/desconocida |
| Mutacion | Bloque 8 | Variacion aleatoria sobre error no filtrado | mutacion | Presente |
| Adaptacion | Bloque 8 | Afinacion sin abrir dominio | adaptacion_activa | Presente |
| Exaptacion | Bloque 8 | Reutilizacion con reserva PRE; abre dominio | XE, reserva, Omega_op | Presente |
| Consciencia metacognitiva | Bloque 8 | Surge ante fallo sostenido con LF | C_m | Presente; observador |
| Activacion latente | Bloque 8 | Detecta deficit y dispara pluripotencia | activacion_latente | Presente |
| Altruismo / Boorman | VST_Genoma actual | beta_crit + Hamilton + psi_alma + simbiosis | disposicion_cooperar, coopera | Presente en Genoma; expresion colectiva por validar |
| Comunicacion funcional | V182B, VST_OrganoComunicacion | Voz/estado de un organismo como entrada del otro | voz, peer, full_state | Presente en interfaz; no organelo Genoma |
| RC = ICR + IRDE | VST_RC_A/B | Conservacion de ruido contextual entre integracion y riesgo | RC, ICR, IRDE, INR | Presente en interfaz; en calibracion |

---

## Fase 3 - Comparacion Genoma actual vs historia

### A) Presente y expresado

| Capacidad | Evidencia actual | Comentario |
|---|---|---|
| Soma / campo Phi con audio | `Célula_Madre_Funcional_001.py` expresa `OrganeloSoma` | Audio real entra al Milieu |
| Presion desacople | Genoma + Celula Funcional | Separada de C_b canonica |
| Consciencia basica | Bloque 5 expresado | C_b=|R1| |
| Meta-representacion R2 | Bloque 5 expresado | Funda LF |
| Self | Bloque 5 expresado | Coherencia identitaria |
| Fatiga | Genoma + CM001 | Parcial por residuo V150 |
| Memoria de ausencia | Genoma/CM001 | Presente en espina |
| Juego | Bloque 7 / CM001 | Expresado |
| Ritual | Bloque 7 / CM001 | Expresado |
| Negacion operativa | Bloque 7 / V176 | Expresada |
| LF | Bloque 7 | Expresada |
| Mutacion, adaptacion, exaptacion | Bloque 8 | Expresadas |
| C_m y activacion latente | Bloque 8 | Expresadas |
| Homeostasis | `VST_Homeostasis.py` | Expresada |
| Orientacion / actuador cabeza | WebLive actual | No como organelo Genoma puro; si en interfaz/actuador |
| RC | `VST_RC_A/B.py` | Integrado en interfaz; no en `GENOMA` como ficha historica |
| Comunicacion interorganismo | `VST_OrganoComunicacion.py` + WebLive | Integrada en interfaz; no ficha Genoma |
| Altruismo | `VST_Genoma.py` actual | Presente y expresado por `locus_altruismo_boorman()`, aunque conducta multicelular requiere validacion |

### B) Presente pero no expresado o expresion condicional

| Capacidad | Evidencia | Estado |
|---|---|---|
| W dual | `GENOMA`: presente, por defecto silenciada | Presente no expresada por default |
| Campo Lambda | `GENOMA`: parcial; sale plano | Presente, latente/no expresiva |
| Inhibicion lateral | `GENOMA`: parcial | Presente, atribucion no limpia |
| Campo atractor / memoria estructural | `GENOMA`: presente | Presente, expresion conductual incompleta |
| Multicelula / S_shared | `VST_Homeostasis.py` | Presente como estructura; cohesion voluntaria depende de altruismo |

### C) Ausente del organismo pero probado historicamente

| Capacidad | Validacion / origen | Observacion |
|---|---|---|
| Memoria relacional | V182A5 | Probada con cuerpo V180 real; ausente del organismo individual activo |
| Buffer de acoplamiento rico | V182A-v3 | Perdido al degradarse a escalar |
| Membrana sensorial | V111 | Ausente |
| Act_perm / permeabilidad activa | V081 | Ausente |
| Relajacion a centro | V134-V139 | Ausente; hoy solo decae confianza |
| CbGlobal | V174 | Ausente; candidato fuerte |
| Olvido selectivo | V080h | Ausente |

### D) Reemplazado por otra estructura

| Capacidad historica | Reemplazo aparente | Riesgo |
|---|---|---|
| Pastor | Homeostasis/economia intrinseca | No revivir como controlador externo |
| Cuerpo calloso escalar | Acoplamientos inline/vectoriales | Se perdio rectificacion direccional explicita |
| R_op clase | Valencia + deliberacion + negacion operativa | Funcion sobrevive, forma original no |
| Saciedad/metabolismo temprano | Homeostasis + fatiga + OI | Puede haber matices perdidos |
| Campo W simple | W/atractor/oscilador/W dual | Reorganizado |

### E) Eliminado explicitamente / podado con razon

| Capacidad | Razon documentada |
|---|---|
| Pastor | Regulador externo; Shannon encubierto |
| Modos espectrales riqueza/entropia | Diagnostico que no alimentaba criterio |
| Ganglio G | No coordinaba, solo elegia slice con mas aristas |
| Inhibicion reciproca | WTA duro; no sobrevive V146-V182 |
| Memoria largo plazo v10 | Contador siempre 1; trivial |

### F) Estado desconocido o evidencia conflictiva

| Capacidad | Por que queda abierta |
|---|---|
| Predictor trayectoria V135-V138 | Genoma lo marca validado; informe unificado advierte fallas de verificacion/umbral. Clasificacion: desarrollada, validacion disputada. |
| R_af V181 | Roadmap lo deja diferido/no ejecutado en capa individual |
| V177-V179 | Generalizacion/extincion/conflicto diferidos segun roadmap |
| V182B1 bidireccional | Existe registro, pero roadmap dice no reverificado |
| Subj_sem / alteridad V182D | Pendiente como gate interpretativo |

---

## Fase 4 - Perdidas evolutivas principales

| Capacidad perdida | Experimento de validacion/origen | Resultado original | Razon aparente de perdida | Impacto actual |
|---|---|---|---|---|
| Maquinaria V122 completa | V122 | Coexistencia R2 + lateralidad por segregacion funcional | V123 simplifica y elimina piezas | Lateralidad sobrevive, pero parte del cierre relacional se adelgaza |
| Membrana sensorial | V111 | Entrada preprocesada por envolvente/derivada/tanh | Eliminada V122 | Entrada actual llega mas cruda al campo |
| Atencion real a tendencias | V118 prometida | Nombre prometia atencion, implementacion divergente | Eliminada; reconstruir capacidad real | Falta organo que atienda cambios, no solo estados |
| Cuerpo calloso direccional | V123 | Transferencia rectificada max(0,diff) | Eliminado V125/subsumido | Se pierde direccion explicita del acoplamiento |
| CbGlobal | V174 | Presion global de desacople | Eliminado V176 | Juego/ritual dependen de senales mas locales |
| Relajacion a centro | V134-V139 | Setpoint vuelve a centro en ausencia | Eliminada V176 | Ausencia solo decae confianza; no hay tono de reposo pleno |
| Act_perm | V081 | Permeabilidad activa de membrana | Inertado/eliminado en ANIMA-1 | Campo no regula suficientemente su permeabilidad |
| Memoria relacional | V182A5 | Cultura acumulativa ON vs regresion OFF | No integrada en Celula Madre actual | Dos organismos pueden hablar, pero aun no tienen organo relacional canonico interno |
| Buffer acoplamiento rico | V182A-v3 | Registro de trayectoria del otro | Degradado a escalar | Se pierde estado temporal del otro |
| Olvido selectivo | V080h | Poda modulada por eficiencia | No portado con W dual | La memoria estructural puede acumular sin criterio organico fino |
| Modos espectrales funcionales | V072c | Diagnostico de riqueza/entropia | Podado con razon | No se debe rescatar como driver sin nuevo criterio |
| R2 como capacidad medida historica | V097-V117 | Sensibilidad relacional | Reescrita como meta-representacion B5 | Puede haberse perdido la prueba especifica R2-relacional |

---

## Fase 5 - Mapa genealogico

```text
V001-V005 grabaciones reales / umbrales
  -> entrada real y sensibilidad minima
  -> Celula Funcional: cargador WAV universal, audio real, binaural

V006-V019 campo Phi minimo
  -> V070 campo continuo
  -> V072b W hebbiana
  -> V072c Phi_int_historia / modos
  -> V080h W dual identidad/contexto
  -> V122 campo bihemisferico
  -> CM001 / Soma actual

V028-V039 homeostasis + metabolismo + preferencia
  -> fatiga / saciedad / valencia
  -> VST_Homeostasis + OI
  -> Celula Madre protoorganismo

V040-V058 miedo / valor / seleccion / identidad
  -> valencia diferencial
  -> V172 precursor No
  -> V176 negacion operativa
  -> Bloque 7 / deliberacion actual

V097-V117 R2 y sensibilidad relacional
  -> V118-V122 hemisferios + lateralidad
  -> V122 coexistencia R2+lateralidad
  -> Bloque 5 R2 canonico + Soma lateral
  -> cabeza live actual, aun calibrandose

V134-V139 memoria con relajacion + prediccion
  -> motor con inercia
  -> parte sobrevive en orientacion/fatiga
  -> predictor y relajacion a centro quedan ausentes

V151-V176 ausencia / Cb / juego / ritual / No
  -> CM001 all-on
  -> Bloque 7 organelizado
  -> VST_CelulaMadre funcional

V180c memoria episodico-valencial
  -> veto por trauma localizado + costo cognitivo
  -> presente parcial en Genoma/CM001
  -> falta recall explicito trial-a-trial y contexto A/B

V182A3-A5 relacion sobre cuerpo V180
  -> roles emergentes
  -> transferencia mutua
  -> cultura acumulativa
  -> candidato principal a organelo relacional futuro

V182B comunicacion sin cuerpo
  -> senal A->B con nulo
  -> organo comunicacion actual en interfaz
  -> falta integracion como organelo Genoma y fuente auditiva estable del otro

V182C convencion
  -> sentido compartido en modelo reducido
  -> base conceptual para S_shared / multicelula
  -> falta version plena sobre cuerpo actual

VST_Genoma
  -> Milieu + Organelo + GenEspec
  -> B5/B7/B8/Homeostasis
  -> Altruismo O-N22 desarrollado
  -> Celula Madre modular, pero con varias capacidades aun en manifiesto o monolito
```

---

## Fase 6 - Respuestas finales

### 1. Que sabe hacer realmente la Celula Madre hoy

La Celula Madre actual puede:

- Procesar audio real por un Soma de campo Phi con 4 hemisferios.
- Recibir entradas binaurales y dos fuentes independientes.
- Producir senales internas: omega_A/B, gradiente, Omega, e_R, A_sys-env, orientacion, INR, energia L/R, balance L/R, coherencia.
- Expresar organelos B5/B7/B8: C_b, R2, Self, juego, ritual, LF, No, mutacion, adaptacion, exaptacion, C_m, activacion latente.
- Mantener homeostasis y calcular OI / Lambda_Cos / invariantes.
- Diferenciar estimulos a igual energia por XE/R2/OI, aunque no por timbre fino en Omega.
- Ejecutar ablaciones por organelo y correr deterministamente.
- Orientar una cabeza/actuador en interfaz live, ahora bajo experimentos RC.
- Comunicar senales/voz de un organismo al otro en la interfaz.
- Tener RC = ICR + IRDE como organo operativo de lectura de ruido contextual.
- Tener altruismo O-N22 en Genoma como organo presente, aunque no hay aun demostracion completa de organismo pluricelular voluntario.

### 2. Que sabia hacer VSTCosmo que hoy ya no hace activamente

- Usar una membrana sensorial rica antes del campo.
- Aplicar permeabilidad activa del campo (`act_perm`).
- Usar CbGlobal como presion global.
- Relajar setpoints hacia centro en ausencia.
- Operar memoria relacional A.5 en una diada real con cuerpo V180.
- Mantener buffer rico de trayectoria del otro.
- Usar rectificacion direccional explicita del cuerpo calloso escalar.
- Aplicar olvido selectivo ligado a eficiencia.
- Explorar actuadores activamente.
- Posiblemente predecir trayectoria, aunque su validacion esta disputada.

### 3. Que organos probados existen en el Genoma pero no estan siendo expresados

- `campo_dualW`: presente, por defecto silenciado.
- `campo_lambda`: presente parcial, pero plano/no expresivo.
- `campo_inhib_lateral`: presente parcial, atribucion no limpia.
- Campo atractor/memoria estructural: presente, pero la evidencia de expresion conductual sigue incompleta.
- `Multicelula` / S_shared: estructura presente; voluntariedad depende de altruismo y pruebas futuras.
- `altruismo`: presente y expresable por compatibilidad, pero falta validacion conductual colectiva plena.

### 4. Capacidades criticas ausentes para reconstruir el organismo cosmosemiotico completo

| Prioridad | Capacidad candidata | Razon |
|---|---|---|
| 1 | Memoria relacional V182A5 | Es la condicion historicamente validada para cultura acumulativa entre organismos |
| 2 | Buffer de acoplamiento rico | Permite que el otro sea trayectoria, no solo escalar |
| 3 | Membrana sensorial | Mejora la forma de entrada sin meter controlador externo |
| 4 | CbGlobal | Devuelve estado fisiologico de conjunto |
| 5 | Relajacion a centro | Recupera tono de reposo real en ausencia |
| 6 | Act_perm | Da permeabilidad organica al campo |
| 7 | Atencion a derivadas | Atender cambios, no solo niveles |
| 8 | Olvido selectivo | Evita acumulacion no metabolizada de memoria estructural |
| 9 | Predictor trayectoria | Candidato si se resuelve conflicto de validacion |
| 10 | Alteridad V182D | Gate interpretativo para que V183 no sea solo acoplamiento dinamico |

### 5. Candidatas a rescate

Rescatar primero, salvo nueva evidencia de poda deliberada:

1. Memoria relacional A.5.
2. Buffer de acoplamiento rico.
3. CbGlobal.
4. Membrana sensorial.
5. Act_perm.
6. Relajacion a centro.
7. Olvido selectivo.
8. Atencion por derivadas.
9. Predictor trayectoria, solo despues de auditar la disputa de validacion.

No rescatar como tales:

- Pastor.
- Ganglio G.
- Memoria largo plazo v10.
- Modos espectrales como driver.
- Inhibicion reciproca WTA dura.

## Veredicto arqueologico

La Celula Madre actual no es un organismo empobrecido: es una arquitectura funcional real con audio, campo, consciencia, libertad, evolucion, homeostasis, RC, comunicacion y actuacion live. Pero tampoco contiene todavia todo el linaje expresado. Su Genoma funciona como manifiesto pluripotente: varias capacidades historicas estan registradas, algunas estan presentes pero latentes, y otras siguen fuera aunque fueron probadas.

La perdida evolutiva mas importante no es una pieza aislada: es la diferencia entre "dos organismos con entrada/salida" y "dos organismos con memoria relacional interna". V182A5 demostro cultura acumulativa; la Celula Madre live actual todavia esta construyendo el organo que haga que esa relacion sea propia del organismo y no solo de la interfaz experimental.

