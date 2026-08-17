# Síntesis del arco R7 — Partículas del Modelo Estándar en el marco cosmosemiótico

**Programa:** Cosmogénesis CG002 · RMD 2.0 · Club Abulafia
**Fechas:** 30-jun-2026
**Equipo:** Alexis López Tapia (dirección/diseño) · Grok/Diotallevi (implementación + barridos) · Claude CC (custodio/verificación)
**Estado:** arco R7 **cerrado por fase** · sector gauge-no-abeliano/Yukawa **bloqueado** (decisión de Mesa pendiente)
**Verificación:** todos los veredictos de esta síntesis fueron recomputados por CC contra los CSV en disco + auditoría de cada script (semilla + acoplamientos). Canal: `CC_TO_GROK.md` Msg 28, 30, 33, 35.

---

## 0. Qué se preguntó (motivación epistémica)

La pregunta la planteó Alexis tras la discusión sobre el estatuto de las propiedades (ver `Análisis con Google IA.txt` y `CC_TO_GROK.md` Msg 25):

> En vez de firmas cuyas propiedades **elegimos**, pongamos partículas **con las propiedades de los quarks/gluones/etc.** a través del **mismo proceso cosmosemiótico**, y veamos qué pasa.

Esto **no** es reducción física (lo prohíbe C-N2.7.6). Es la versión **experimental** de la *homología estructural entre marcos* (Msg 25, pregunta legítima #2): bajo S>0 no hay propiedad poseída en ninguna parte — ni en la firma ni en el quark; ambas son **determinaciones** dentro de un marco. No hay terreno neutral (O-N16.5). Entonces el careo no es «¿la sim reproduce la realidad?» sino **«¿el marco cosmosemiótico y el marco físico determinan la misma estructura mínima de la diferencia?»**.

---

## 1. Método

| | Qué |
|--|-----|
| **FIJO** (el proceso cosmosemiótico) | S>0; persistencia; extinción irreversible κ_s; acoplamiento campo medio O(N); banda sat C-N5.1; τ por Δ_struct; θ_CP=0 |
| **REEMPLAZADO** (la partícula) | Ω_firma + ley de compatibilidad: de `U·U` *elegido* → números cuánticos del Modelo Estándar |

**Disciplina «derivación ciega inversa»:** sólo **primitivos** como input (spin, color, carga, isospín, generación). **Prohibido en semilla:** masa, confinamiento, violación CP, baryogénesis. Deben **emerger** o quedar excluidos. Si lo que emerge ya estaba en la entrada, no prueba nada.

---

## 2. Tabla de veredictos por fase

| Fase | Adición | Ω_firma | Veredicto CC | Estatuto |
|------|---------|---------|--------------|----------|
| **R7a** | color SU(3) × spin (gluón = acoplamiento) | 12 microestados | ✅ limpio; señal real en **K=6** (exceso 4→17× + shuffle+); el ½@K12 era **trivial** | Resultado |
| **R7c** | leptones (singlete de color) | +leptones | subsumido y verificado dentro de R7e | — |
| **R7d** | carga eléctrica U(1) | +canal Q·Q | subsumido y verificado dentro de R7e | — |
| **R7e** | generaciones ×3 + U(1) + leptones | 84 micro | ✅ limpio; **polarización emergente real** (ley/historia); generación pasiva; sabor activo vía carga | **Resultado fuerte** |
| **R7b** | gluón como **entidad** (octeto) | 12q + 8g | ❌ **roto**: Gell-Mann malformado + colapso diagonal + gg ortonormal → gluones inertes | Artefacto |
| **R7f** | γ/W/Z + Higgs | 84f + 5b | ⚠️ masa OK (post-hoc); polarización persiste; **Higgs desacoplado** (vacuo) | Mixto |

---

## 3. Las dos líneas del arco

### 3.1 Línea abeliana / por carga — **FUNCIONA** ✅

Canales expresables como **producto interno pareado**: color SU(3) fundamental (`v_i·v_j`, antipartícula = −v) y carga U(1) (`Q_i·Q_j`).

**Resultado genuino (el más fuerte de todo R7):** polarización de carga **y** color **emergente**, con la firma del arco — **magnitud robusta (ley) + dirección contingente (historia)** — sin CP inyectada:

| (R7e, N=4000) | inicial | final | amplificación |
|--|--|--|--|
| orden_charge | 0.0008 | **0.27** | ~340× |
| orden_color | 0.0016 | **0.44** | ~270× |

`mean_Q` entre semillas ≈ 0 mientras `|mean Q|` por corrida ≈ 0.27 ⇒ cada cosmos se polariza con magnitud reproducible pero **signo azaroso por semilla**. Es la **baryogénesis-análoga** (mismo patrón ley/historia que el arco fundacional), ahora en los canales U(1)+color.

Control decisivo: **shuffle-E** (carga aleatoria) destruye la estructura (ΔL2_K3 grande) → la selección **la manda la carga**. K_eff = 3 (signo de carga) es la señal, análoga a K=6 de R7a.

### 3.2 Línea de 3 puntos (gauge no-abeliano self / Yukawa) — **BLOQUEADA** ❌

| | R7b (gluón entidad) | R7f (Higgs) |
|--|--|--|
| Síntoma | gluones **100% inertes** (no aniquilan); f=0.80 = conteo de tokens | Higgs **Σ\|c_ij\|=0** (desacoplado); φ nunca crece, decae |
| Bugs de implementación | Gell-Mann malformado (falta λ₃, λ₄ no Hermitiana); `Re(vᵀλv)` colapsa a la diagonal (quarks ↔ 1 de 8 gluones); gg=`eye(8)` ortonormal | γ/Z = glue ad-hoc `0.15·\|Q\|`,`0.15·\|T₃\|`; alza f 0.47→0.53 es ese artefacto |
| Conclusión «de física» | "octeto no reproduce el grano" = **vacua** (octeto nunca acoplado) | "Higgs no condensa" = **vacua** (token inerte decayendo) |

**Causa común (no son dos fallos, es uno):** el vértice **gauge** (q→q′+g) y el **Yukawa** (f–f–H) son **de 3 puntos** (el color/la identidad se intercambia). El motor cosmosemiótico usa compatibilidad **pareada** `c_ij` (2 puntos). **Un vértice de 3 puntos no se reduce a un escalar pareado** — esa reducción es lo que fuerza el colapso diagonal (R7b) y deja al Higgs sin dónde acoplarse (R7f).

> **El gluón-entidad y el Higgs son la MISMA puerta bloqueada.** Tocada dos veces.

---

## 4. Lección metodológica recurrente: la trampa del ½

`m_eff/K = (bins ocupados)/K`. Cuando **medio sobrevive** (geometría de hemisferio, C-N1.3), aparece un ½ que **no es invariante** — es la reescritura de la mitad trivial. Apareció tres veces y sedujo dos:

- **R7a:** m_eff/K=½ en K=12 ⟺ 6/12 microestados ⟺ trivial. `best_K` por «cercanía a ½» **gana K=12 por construcción**.
- **R7b:** m_eff/K₈=1.0 ⟺ L2_K₈=0 ⟺ todos los gluones (inertes) uniformes.
- **R7e:** los ½ de sabor/u-d son mitad-por-signo-de-carga.

**Regla del arco:** el invariante real **no** es `m_eff/K`, es el **exceso sobre el nulo finito + separación bajo shuffle**. Desconfiar de todo ½ celebrado. (Esta es la misma lección que ya degradó f≈½ y |orden|≈½ en el arco fundacional κ_Δ.)

---

## 5. La frontera y la decisión (de Mesa, no de ejecución)

El arco R7 deja una frontera **nítida**, no un fracaso:

- ✅ **Lo abeliano / por carga se determina igual en ambos marcos** — color y carga, expresables como producto interno pareado, producen polarización emergente con estructura ley/historia. Eso es homología estructural medida (no retórica, no reducción).
- ❌ **Lo gauge-no-abeliano-self y el Yukawa/Higgs NO caben en el primitivo pareado.**

**Decisión Mesa (Alexis, 30-jun-2026):** **extender el primitivo a vértices de 3 puntos.**

**R7g (apertura post-cierre):** `cg002_primitivo_vertex3.py` + `cg002_r7g_vertex3.py` — motor `coop_i = Σ_j c_ij m_j + Σ_{j,k} v_ijk m_j m_k`. Primer barrido: gauge (λ corregido, triplete q–g–q′) y Yukawa (f–H–f′). Ver `GROK_TO_CC.md` Msg 38.

---

## 6. Lo que NO se reclama (disciplina)

- **No** es reducción física (C-N2.7.6). Es homología estructural en el canal abeliano, no «el cosmos tiene estas partículas».
- **Sin masa, generación y sabor son siempre pasivos** — porque lo que las hace no-degeneradas es la jerarquía de masa, excluida por disciplina. (R7f lo confirma: m₂/m₀=1.000, null honesto.) Activar el sabor exigiría que la masa **emerja** vía un Yukawa real — bloqueado por §3.2.
- Los ½ triviales y los tokens inertes **no** son resultados; son síntomas de binning o de acoplamiento ausente.

---

## 7. Reproducibilidad

| Fase | Script | Datos | Verificación CC |
|------|--------|-------|-----------------|
| R7a | `cg002_r7a_color_spin.py` | `cg002_r7a_sweep.csv` (400) | Msg 28 |
| R7b | `cg002_r7b_gluon_entity.py` | `cg002_r7b_sweep.csv` (600) | Msg 30 |
| R7c | `cg002_r7c_leptons.py` | — | vía R7e |
| R7d | `cg002_r7d_charge_u1.py` | — | vía R7e |
| R7e | `cg002_r7e_generations.py` | `cg002_r7e_sweep.csv` (1200) | Msg 33 |
| R7f | `cg002_r7f_higgs.py` | `cg002_r7f_sweep.csv` (1000) | Msg 35 |

Visor: pestaña **Plasma QCD** (toggle R7a/R7b) en `UniversoCosmosemiotico.html`.

---

## 8. Frase de cierre

> El marco cosmosemiótico, dado un primitivo **pareado**, determina **lo mismo** que el marco físico allí donde la física es **abeliana / por carga**: polarización emergente con magnitud-ley y dirección-historia, sin asimetría inyectada. Donde la física es **no-abeliana-self o de Yukawa** —vértices de 3 puntos—, el primitivo pareado **no alcanza**, y lo honesto es nombrar la pared, no fabricar un resultado. El arco no falló: **midió su propia frontera.**

---

## Anexo — Aporte externo (Plano C)

Lectura **independiente** del arco computacional: Google IA propuso mapa Stirling ↔ canon (regenerador como O-N8.19, ES=ICES+IDES, OI Caso A/B). Documentado en **`APORTE_GOOGLE_IA_STIRLING_PLANO_C.md`** — valioso per se; no veredicto del equipo.

---

*Síntesis de verificación · Claude CC (custodio) · Cosmolab / VSTCosmo · 30-jun-2026 · sin commits hasta orden de Alexis.*
