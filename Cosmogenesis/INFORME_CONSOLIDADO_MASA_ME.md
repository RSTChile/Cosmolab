# La masa en el Modelo Estándar — informe consolidado y verificado cruzado
### Síntesis de tres fuentes: literatura primaria (CS), Perplexity, y verificación cruzada

**Preparado por:** Claude Science (CS) · **Fecha:** 23-jul-2026
**Método:** se cruzaron dos investigaciones independientes (la mía sobre fuentes primarias — papers de 1964, lattice / retículo, PDG — y la de Perplexity). Se marca dónde coinciden (✓✓), dónde una aporta lo que la otra no, y las correcciones de detalle.

### Glosario de siglas (traducción al castellano)
- **ME** — Modelo Estándar (de la física de partículas).
- **QCD → CDC** — *Quantum Chromodynamics* = **Cromodinámica Cuántica** (la teoría de la fuerza nuclear fuerte, entre quarks y gluones).
- **BEH** — mecanismo de **Brout–Englert–Higgs** (con Guralnik–Hagen–Kibble; a veces BEHGHK).
- **VEV** — *Vacuum Expectation Value* = **valor esperado en el vacío** (el valor no nulo que toma el campo de Higgs).
- **EWSB** — *Electroweak Symmetry Breaking* = **ruptura de la simetría electrodébil**.
- **BSM** — *Beyond the Standard Model* = **más allá del Modelo Estándar** (física nueva).
- **Yukawa** — el acoplamiento entre un fermión y el campo de Higgs que fija su masa.
- **Λ_QCD → Λ_CDC** — escala de energía de la Cromodinámica Cuántica (~200 MeV).

---

## RESUMEN EN UNA PÁGINA (lo verificado por ambas fuentes)

1. **✓✓ El Higgs NO da "toda la masa".** Da masa directa a las partículas elementales (bosones W/Z y fermiones vía Yukawa), pero **la mayor parte de la masa de protones y neutrones es energía de la Cromodinámica Cuántica (CDC)** — energía de gluones + energía cinética de quarks —, no Higgs. Ambas fuentes coinciden en que esta es "una de las incongruencias más frecuentes entre la divulgación popular y la formulación teórica estricta" (Perplexity), y la mía la cuantifica: **~1% Higgs, ~99% CDC** para la materia ordinaria.
2. **✓✓ El campo de Higgs no "aparece" en un instante.** Está en la teoría desde el inicio como parte del vacío; **lo que cambia es el estado del vacío**: a alta temperatura la simetría está restaurada (VEV≈0, sin masas electrodébiles), y al enfriarse ocurre la ruptura y el campo toma un VEV finito. "La masa emergente es una propiedad de fase del vacío y del estado térmico del universo, no una constante ya dada desde siempre" (Perplexity).
3. **✓✓ El bosón de Higgs existe (~125 GeV), confirmado en 2012** (ATLAS + CMS, >5σ). Pero el acoplamiento a fermiones está probado directo **solo para la 3ª generación** (top, bottom, tau).
4. **✓✓ El problema de jerarquía sigue abierto** — el ME no protege la masa del Higgs frente a correcciones cuánticas; no está resuelto ni confirmada la física BSM que lo resolvería.
5. **Aporte solo de la revisión CS (Perplexity no lo cubrió):** la cronología cósmica cuantitativa (dos épocas: ruptura electrodébil ~10⁻¹¹ s y transición de CDC ~10⁻⁵ s, **ambas crossovers suaves**), la descomposición numérica de la masa del protón, y que la **1ª generación (electrón, up, down) nunca se ha medido**.

---

## 1. QUÉ MASA EXPLICA EL HIGGS — Y QUÉ NO (convergencia total)

**Ambas fuentes coinciden, sin matices.** El mecanismo BEH (1964) rompe la simetría electrodébil y da masa a W⁺, W⁻, Z (el fotón queda sin masa), y a los fermiones elementales vía sus acoplamientos de Yukawa a un campo con VEV no nulo. Papers fundacionales, confirmados por las dos investigaciones:
- Englert & Brout, *Phys. Rev. Lett.* **13**, 321 (1964).
- Higgs, *Phys. Rev. Lett.* **13**, 508 (1964) — el único que menciona el bosón masivo.
- Guralnik, Hagen & Kibble, *Phys. Rev. Lett.* **13**, 585 (1964).

**La distinción crítica (ambas):** el Higgs da masa al *electrón* (fuertemente ligada a su Yukawa), pero la masa del *protón* está dominada por la energía de los gluones y la energía cinética de los quarks confinados — Cromodinámica Cuántica, no Higgs.

**Cuantificación (solo revisión CS, lattice / retículo):** descomposición de la masa del protón (Yang et al., *Phys. Rev. Lett.* 121, 212001, 2018): ~33% energía de quarks + ~37% energía de gluones + ~23% anomalía de traza + ~9% masa de quarks. La suma de masas de los quarks de valencia es apenas ~1% del protón. En el límite quiral (masa de quarks→0) el protón **conserva casi toda su masa**, proporcional a Λ_CDC.

---

## 2. CUÁNDO "APARECE" LA MASA — LA CORRECCIÓN CONCEPTUAL CLAVE

**Convergencia en el concepto (ambas):** no hay un instante en que el campo de Higgs "se enciende". El campo preexiste; **cambia el estado del vacío** al enfriarse el universo. Por encima de la temperatura crítica la simetría electrodébil está restaurada y no hay masas electrodébiles; por debajo, el vacío se rompe y aparecen.

**Cuantificación cronológica (solo revisión CS):** son **dos épocas distintas**, y —dato decisivo— **ambas son crossovers suaves, no transiciones de fase con salto**:

| Época | Tiempo tras Big Bang | Temperatura crítica | Qué genera | Tipo |
|---|---|---|---|---|
| Ruptura electrodébil | ~0.9 × 10⁻¹¹ s | 159 ± 1 GeV | masas de las partículas elementales | **crossover suave** (D'Onofrio+ 2014) |
| Transición de CDC (Cromodinámica Cuántica) | ~10⁻⁵ s | ≈ 155 MeV | **~99% de la masa del nucleón** | **crossover suave** (Aoki+ 2006, *Nature* 443, 675) |

**Consecuencia:** el grueso de la masa (la del protón/neutrón) se genera en la **transición de CDC (~10⁻⁵ s)** — muchísimo antes de que existan átomos (recombinación ~380.000 años). Y ninguna de las dos transiciones tiene un umbral nítido.

---

## 3. TABLA CONSOLIDADA — AFIRMACIÓN vs. TEORÍA vs. PRUEBA

Fusión de las dos tablas, con el estado experimental afinado. ✓✓ = ambas fuentes concuerdan.

| Afirmación | Qué dice el ME | Estado experimental | Nota crítica |
|---|---|---|---|
| Existe el campo de Higgs en todo el espacio | Parte del vacío electrodébil | Apoyado indirecto (bosón 2012) ✓✓ | No se "ve" el campo; se infiere por excitaciones y acoplamientos |
| El bosón de Higgs existe (~125 GeV) | Excitación del campo | **Sí, 2012, >5σ** ✓✓ | Evidencia fuerte y estándar |
| Higgs da masa a W, Z | Directo (BEH) | **Confirmado** ✓✓ | — |
| Higgs da masa a fermiones (Yukawa) | Directo | **Solo 3ª gen. a >5σ** (top, bottom, tau) | 2ª gen: muón ~3σ, charm solo cota. **1ª gen (e, u, d): NUNCA medida** (solo revisión CS) |
| "El Higgs da toda la masa del universo" | **Falso como formulación** ✓✓ | No probado (es incorrecto) | ~99% de la masa visible es CDC, no Higgs |
| Masa del protón ≈ energía de CDC | Sí (dinámica de gluones) | **Lattice/retículo lo confirma** | ~1% Higgs, ~99% CDC (solo revisión CS, cuantificado) |
| Masas de fermiones predichas por el ME | **No** — Yukawa son parámetros libres | — | Incluye el 1/1836 y toda la jerarquía de sabor |
| Higgs da masa al neutrino | **No en el ME mínimo** | Oscilaciones prueban masa ≠ 0 → BSM | Física más allá del ME, ya probada |
| BEH resuelve el problema de jerarquía | **No** ✓✓ | Problema abierto, sin confirmación BSM | Tensión central del ME |
| El sector Higgs está "cerrado"/completo | Mínimo consistente pero incompleto ✓✓ | **No** — se buscan desviaciones | Naturalidad, vacío, extensiones abiertas |

---

## 4. INCONGRUENCIAS TEÓRICAS (consolidadas)

1. **Semántica (✓✓):** "el Higgs da masa a todo" es pedagógicamente útil pero físicamente incompleto — da masa a *algunas partículas elementales*, no explica el origen dominante de la masa hadrónica (la de los núcleos).
2. **Problema de jerarquía / naturalidad (✓✓):** la masa del Higgs recibe correcciones cuánticas cuadráticas y, sin física nueva, parece extraordinariamente inestable frente a escalas mayores (hasta la de Planck). El ME no la protege. Sin resolver; hay incluso quien lo considera un pseudo-problema.
3. **Cosmológica (✓✓):** si la simetría se restaura a alta temperatura, las masas electrodébiles **no son propiedades eternas** — son propiedad de la fase del vacío y del estado térmico. La masa es emergente, no "dada desde siempre".
4. **El sabor (solo revisión CS):** las proporciones de masa (1/1836 protón/electrón; top ~10⁵ × electrón) son *inputs* ajustados, no salidas de la teoría.
5. **Neutrino (solo revisión CS):** el ME mínimo predice neutrino sin masa; las oscilaciones prueban lo contrario → es física BSM ya confirmada.

---

## 5. CORRECCIONES DE DETALLE (verificación cruzada)

Al cruzar, dos puntos a afinar en el material de Perplexity — menores, no cambian nada de fondo:
- **Masa del bosón:** Perplexity cita "126.0 GeV" del reporte ATLAS 2012 (correcto para ese anuncio inicial); el **valor combinado y actual es ≈125.25 GeV** (PDG). Ambos son el mismo descubrimiento; el número de consenso hoy es ~125.
- **Cita de Higgs:** Perplexity lista el primer paper de Higgs como *Physics Letters* **12**, 132 (1964) — correcto: ese es el primero (sin bosón), y el *Phys. Rev. Lett.* **13**, 508 (que sí menciona el bosón) es el segundo. Los dos son de Higgs, 1964; conviene no confundirlos.
- **Orden de nombres:** BEH / BEHGHK — variantes de orden, sin consecuencia.

Ninguna fuente contradice a la otra en sustancia. La de Perplexity es más fuerte en el **marco conceptual** ("el campo preexiste, cambia el vacío"; la incongruencia semántica); la mía es más fuerte en lo **cuantitativo** (porcentajes de CDC, cronología con temperaturas y tiempos, estado por generación).

---

## 6. RELEVANCIA PARA COSMOGÉNESIS (sin adjudicar el modelo)

Cuatro puntos, dichos con cuidado — esto orienta, no cierra:

- **A favor de nuestro instrumento:** que las transiciones reales sean **crossovers suaves, no saltos**, es *consistente* con CS074-rcruz, donde el cruce en r encendía monótono y temprano, no como escalón. La ausencia de umbral nítido no era defecto nuestro.
- **Tensión real con la premisa E4:** el enfoque del equipo pone la masa "solo en E4, tras el átomo + gravedad". Pero **el ME sitúa el grueso de la masa (el nucleón) en la transición de CDC (~10⁻⁵ s), mucho antes de los átomos** (~380.000 años). Físicamente, la masa del protón ya está fijada por la Cromodinámica Cuántica antes de que exista un átomo. Es una incongruencia sobre E4 que conviene mirar de frente. (Salvedad ya declarada: nuestra "masa" del juguete está en unidades propias, no en GeV.)
- **Valida el anti-Shannon:** que **ni la teoría de frontera prediga las masas de fermiones** (son parámetros libres) confirma que nunca intentáramos reproducir 1/1836 ni 7:1 como jueces — ponerlos a mano habría sido el error del 20.0.
- **Afinidad conceptual:** que ~99% de la masa del protón sea *energía del campo* (relación, no sustancia) y que **la masa sea emergente del estado del vacío, no dada** — ambas cosas son afines a la intuición cosmosemiótica (masa como relación/energía; lo que emerge no está pre-dado). Punto de contacto, no prueba.

---

## 7. REFERENCIAS (fuentes primarias, ambas investigaciones)

- Englert & Brout, *Phys. Rev. Lett.* **13**, 321 (1964).
- Higgs, *Physics Letters* **12**, 132 (1964) [primero] y *Phys. Rev. Lett.* **13**, 508 (1964) [con bosón].
- Guralnik, Hagen & Kibble, *Phys. Rev. Lett.* **13**, 585 (1964).
- Yang, Liang, Chen, Draper, Liu, Sun, *Phys. Rev. Lett.* **121**, 212001 (2018), arXiv:1808.08677 — descomposición de la masa del protón.
- Roberts & Schmidt, arXiv:2006.08782 — emergencia de la masa hadrónica.
- Aoki, Endrődi, Fodor, Katz, Szabó, *Nature* **443**, 675 (2006) — orden de la transición de CDC (crossover).
- HotQCD, *Phys. Lett. B* **795**, 15 (2019); Wuppertal-Budapest — temperatura crítica de CDC.
- D'Onofrio, Rummukainen, Tranberg, *Phys. Rev. Lett.* **113**, 141602 (2014), arXiv:1404.3565 — crossover electrodébil, Tc=159±1 GeV.
- ATLAS Collaboration (2012); CMS Collaboration, *Nature* **607**, 60 (2022).
- Particle Data Group, "Status of Higgs Boson Physics", *PTEP* **2022**, 083C01.

*Nota: no soy físico de partículas ni esto es asesoría profesional; es una síntesis de literatura acreditada, verificada cruzando dos investigaciones independientes, para orientar el trabajo experimental. Los valores y veredictos son los que reportan las fuentes citadas; toda decisión de fondo debe contrastarse con los papers primarios directamente.*
