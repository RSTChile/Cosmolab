# Inventario de rescate — mecanismos perdidos del linaje VSTCosmo
### Catálogo de destinos: qué entró, qué computa su CÓDIGO, dónde cayó, y si valía conservarlo

> **Por qué existe.** Las cronologías (`GENEALOGIA_EXPERIMENTAL_VSTCosmo.md`, `CRONOLOGIA_POR_SCRIPTS.md`)
> registran **qué entró**. Este documento registra **qué se perdió y por qué** — el ciclo de vida de
> cada mecanismo. Sin esto, se vuelve a redescubrir lo mismo cada mes.
>
> **Método (no negociable).** Por **mecanismo**, no por versión. Leído del **CÓDIGO**, no de informes.
> Se reporta **lo que la fórmula computa literalmente**, no lo que promete el nombre. Compilado 2026-06-20
> leyendo el código fuente (clases/funciones/firmas a lo largo de 292 scripts).
>
> **Tipos de pérdida:** `eliminado` (desapareció del código) · `subsumido` (colapsado dentro de otra cosa)
> · `inertado` (presente pero desconectado: se computa y no se usa).
> **Clasificación:** **(A)** deriva-de-foco · **(B)** poda-con-razón · **(?)** indeterminado · **FdA** fuera-de-alcance (relacional).
> **Validado:** "criterio ✅" = hay criterio explícito impreso; "resultado ⊘" = no se verificó contra log. ⊘ = no se puede determinar.
>
> ⚠️ **Esto es inventario, NO implementación.** Nada se restauró. La restauración se decide con esto en mano.

---

## Tabla maestra — mecanismos perdidos / subsumidos / inertados

| Mecanismo | Entró | Qué hace su CÓDIGO (literal) | Nombre vs función | ¿Validado? | Dónde/cómo cayó | Clasif. | Estado v1/v2 |
|---|---|---|---|---|---|---|---|
| **Pastor** (`PastorCosmosemiotico`) | v11 | Controlador proporcional bang-bang que ajusta 2 params (ganancia, intensidad) **entre corridas** para mantener rango_Φ∈[0.3,0.7] | OK | criterio ✅ (débil) | v11→**v12 subsumido**: la meta homeostática se absorbe en el campo permeable (target intra-paso desde la entrada); se pierde la *adaptación de parámetros* | **(B)** | perdido (subsumido) |
| **Modos espectrales** (riqueza/entropía) | v72c | FFT por banda → `riqueza_modal=Σ(perfil>media)`, `entropía=−Σp·log p`, frec. dominante | OK | criterio ✅ (resultado ⊘) | v72c→**v80h subsumido/podado**: GED sobrevive (`calcular_ged_entre`); riqueza/entropía **nunca alimentaron criterio** → descartadas | **(B)** | perdido |
| **Oscilador frec. naturales** (`term_osc`) | v70 | Oscilador armónico por banda `−ω²(Φ−Φeq)−amort·Φ̇`, ω log-espaciada | OK (pero "ω" se **recicla** desde V109 como media del campo) | ⊘ (estructural, sin criterio aislado) | v108→**V109 eliminado** al pasar Φ de 2D→1D (pierde el eje temporal sobre el que oscilar) | **(?)→(A)** | **rescatado en v2** ✅ |
| **W dual** `W_prof`/`W_rec` + olvido selectivo | v80 | Plasticidad hebbiana dual (identidad lenta / contexto rápido) con olvido modulado por eficiencia | OK | criterio ✅ (v80h "ciclo completo") | **discontinuidad V122** (linaje ANIMA-1 nunca la heredó); inertada ya en v81 | **(A)** | **rescatado en v2** (olvido selectivo ⊘) |
| **`Phi_int_historia`** (atractor) | v72c | EMA del campo interno, reinyectada `γ·(hist−Φ)` (viva en v80h) | OK | criterio ✅ (v80h) | **inertada en v81** (se computa, no se reinyecta); no llega a ANIMA-1 | **(A)** | **rescatado en v2** (versión viva v80h) ✅ |
| **Ganglio G** | v81 | NO coordina: es la **rebanada con más aristas** (grado 7) en una lista `VECINDADES` escrita a mano → recibe más promedio difusivo | **DIVERGE** | criterio C10 (resultado ⊘) | **inertado V120** (se borra `VECINDADES`, índice colgando) → **eliminado V121**; sin reemplazo (bihemisferio usa otra arquitectura) | **(A)** | perdido |
| **Actuadores** `act_busc/geom/perm/mant` | v81 | `act_busc`=diferencia espectral L-R; `act_perm`=permeabilidad α; `act_geom`=asimetría L/R; `act_mant`=detector de varianza baja | **DIVERGEN** (busc≠búsqueda, mant≠mantenimiento) | criterio C11-13 (resultado ⊘) | **inertados V120** (índices definidos, 0 lecturas) → **eliminados V121**. Lateralidad (geom/busc) reimplementada como **filtros de audio**; perm/mant **sin reemplazo** | geom/busc **(B parcial)** · perm/mant **(A)** | perdido |
| **`ExploradorActuadores`** | v97.2 | **Registrador pasivo** del argmax de eficiencia (`best-so-far`); NO explora ni perturba nada | **DIVERGE FUERTE** | ⊘ | **inertado ya en V109** (instanciado, **nunca llamado**) → eliminado V121 | **(A)** | perdido |
| **MembranaSensorial** | V111 | Preprocesa la entrada escalar dS=L−R: `0.4·inst+0.3·envolvente+0.2·derivada+0.1·tanh`, clip (V111); degradada a suma cruda sin pesos/clip (V121) | OK | ⊘ | degradada V121 → **eliminada V122** (entrada cruda a la frontera) | **(A)/(B parcial)** (cambio de dieta sensorial; descarte tácito) | perdido |
| **Atención** (`MecanismoAtencion`/`Atencion`) | V118 | V118: similitud coseno + softmax sobre 1000 estados → `max(pesos)`. V122b: mezcla Ω-por-Λ **cuya salida se descarta**. **Nunca atiende derivadas** | **DIVERGE** (el nombre promete dΩ/dt) | ⊘ (nunca load-bearing) | V118→**V122 eliminado**; V122b **inertado** (computa, no se usa) → eliminado | **(A)** | perdido |
| **CuerpoCalloso** (clase) | V123 | Transferencia escalar Ω **rectificada-direccional** (`max(0,diff)`: solo el dominante empuja), gated por umbral 0.5 | OK | ⊘ | **eliminada V125** (ramal escalar V123/124 abandonado; V125 vuelve a `from V122`). La función (acoplamiento gated) sobrevive **inline** pero en forma **vectorial-simétrica** (linaje V121/V122), perdiendo la rectificación direccional | **(B)** | perdido (función subsumida, rectificación no) |
| **InhibicionReciproca** | V123 | Winner-take-all duro sobre Ω: el hemisferio dominante recibe input pleno, **anula el del rival** | OK | ⊘ | **eliminada V125**. **Ningún término inhibitorio sobrevive en V146–V182** | **(?)→(A)** | perdido |
| **Inhibición lateral** (V122) | V122 | El rápido ('L') **congela su propio acoplamiento** cuando `|dΩ/dt|>0.1` (protege R₂) | OK | ⊘ | eliminada en V123 | **(?)** (poda temprana, sin medición) | **rescatada en v2** (sin efecto en sonda aislada) |
| **Λ** (`_calcular_Lambda`) | V122 | `Δ_struct·(LF+ε)/(e_R+ε)`, **LF = nº atractores distintos en omega reciente** (agitación temporal en ~½s) | **DIVERGE** (se vendía como "pluripotencia"/destinos latentes) | ⊘ | eliminada en V123 | **(?)** | **rescatada en v2** (medida; salió plana) |
| **Predictor de trayectoria** (`PredictorTrayectoria`) | V135 | Diferencia finita de posición + extrapolación lineal a horizonte 2s: `pos + vel·h` | OK | **criterio ✅** (MAE red>50%, Lead>0) | **eliminado V139** (reemplazo conceptual: la anticipación "emerge del sobrepaso por inercia") | **(A)** | perdido |
| **Predictor adaptativo / con aceleración** (V137/V138) | V137-138 | Horizonte adaptativo (`error/|vel|` o `|vel/acc|`); V138 predicción cuadrática `pos+vel·h+½·acc·h²` | OK | **criterio ✅** (MAE<12° en 4 fases) | **eliminados V139** | **(A)** | perdido |
| **MemoriaConRelajacion** | V134 | Confianza `exp(−t_silencio/τ)` + **relajación hacia centro**: `ángulo·c^α + centro·(1−c^α)` (desde V135) | OK (V134 original colapsa a 0; V135+ añade centro) | **criterio ✅** (O-N9.1, error relajación <10° a 60s) | **eliminada V176**: la **relajación-a-centro desaparece**; sobrevive solo el decaimiento de confianza en `MemoriaAusencia` (mantiene último setpoint) | **(A)** | perdido (parcial) |
| **R_op** (clase explícita) | V168 | Clase de veto/negación con umbral y supresión de acción | OK | criterio en docstring | **eliminada como clase**; función **subsumida** en `ValenciaLocal`+`MemoriaDeTrabajo` (veto por valencia negativa; en V180c veto episódico explícito `puntaje−=100`) | **(B)** | subsumido (vivo por otra vía) |
| **CbGlobal** | V174 | Integrador con fuga de la **presión global de desacople** `dCb/dt=e_R·(1−A_sys_env)−Cb/τ`; separado de la valencia local; gatea ritual/juego | OK (no es media de Cb locales) | indirecto (que la valencia local resista Cb alta) | **eliminada V176**: la capa **global** desaparece; la EDO de presión sobrevive **relabelizada como Cb local** por motor | **(A)** | perdido (capa global) |
| **MemoriaRelacional** (3 mecanismos homónimos) | V182A-v7 | v7=delta vs último resultado del otro; v10=delta-por-audio (sin uso); **A5=confianza sigmoide** `conf[banda]` (la madura) | nombre cubre 3 cosas | A5: **criterio ✅** ON/OFF pareado | inter-organismo (díada) | **FdA** | fuera-de-alcance (relacional) |
| **MemoriaLargoPlazo** (V182A-v10) | V182A-v10 | Media incremental por nombre de audio; **contador siempre 1** → nunca promedia | OK pero trivial | ⊘ | **inertada en su propia versión** | **(A) trivial** | perdido |
| **BufferAcoplamiento** (2 homónimos) | V182A-v3 | v3: buffer 3D de la **trayectoria del otro** (val,Cb,D) + distancia euclidiana + convergencia. v7: **colapsado a escalar** de reward sin estado | nombre cubre 2 cosas | ⊘ | **degradado v3→v7** (se elimina el historial de estados del otro) | **(A)** | perdido |

---

## Hallazgo transversal: nombre ≠ función (marcado por el código)

Estos mecanismos **prometen una cosa y computan otra**. Es la lección central de este inventario:

- **`ExploradorActuadores`** no explora — **registra el máximo** que le pasen (best-so-far). Además, inerte (nunca llamado) ya en V109.
- **Ganglio G** no coordina — es la **rebanada con más aristas** en una lista de adyacencia escrita a mano.
- **`act_busc`** no busca — es una **diferencia espectral L-R**. **`act_mant`** no mantiene — es un **umbral de varianza baja**.
- **Atención** nunca atendió derivadas — fue coseno-softmax (V118) y luego una mezcla **descartada** (V122b).
- **Λ** no mide pluripotencia — cuenta **agitación temporal de omega en ~½ segundo** (ya señalado en sesión previa).

> Regla que deja este inventario: **antes de rescatar un mecanismo, leer qué computa su fórmula, no su nombre.**

---

## Candidatos (A) que PARECEN valiosos — ordenados por valor aparente
*(Señalados, NO restaurados. "Valor aparente" = función real + validación, no el nombre.)*

1. **Predictores de trayectoria (V135–V138)** — **el más valioso.** Es la **única familia (A) con criterio de éxito explícito** (MAE red>50%, MAE<12° en 4 fases). Computa anticipación real (vel/acc por diferencia finita + extrapolación). Se eliminó en V139 por una apuesta conceptual ("la anticipación emerge del sobrepaso por inercia"), **sin test que comparara** predictor-explícito vs inercia. El organismo actual no anticipa: orienta al setpoint instantáneo. Si se quiere anticipación medible, esto ya existía y estaba validado.
2. **MemoriaConRelajacion → relajación-a-centro (V134–V139)** — validada (O-N9.1). En el organismo actual sobrevive el decaimiento de confianza pero **no** la relajación del setpoint hacia un centro: en ausencia, el organismo mantiene el último setpoint en vez de relajar. Capacidad validada, parcialmente perdida.
3. **`act_perm` — permeabilidad activa (v81)** — modulaba α (acoplamiento al estímulo) desde el estado del propio campo. Sin reemplazo. Conceptualmente cercano a la "permeabilidad estructural" fundacional (v12). Función real modesta pero coherente con el telos de campo-como-medio.
4. **CbGlobal — presión global de desacople (V174)** — una capa de Cb **global** (separada de la valencia local) que gateaba ritual/juego. El organismo actual solo tiene Cb **local** por motor. Si se busca un estado global del organismo (relevante a "célula madre"), esta capa existía.

> **No incluidos como valiosos** (son (A) por tipo de pérdida pero ⊘ validación o función trivial/divergente):
> `ExploradorActuadores` (registrador pasivo, inerte), Ganglio (rebanada-por-grado), Atención (nunca load-bearing),
> MemoriaLargoPlazo (contador siempre 1), BufferAcoplamiento (degradado, ⊘). **(A) ≠ valioso.**

---

## (B) confirmados — NO volver a tocar
*(Para que nadie los redescubra y los quiera revivir sin razón.)*

- **Pastor (regulador externo de parámetros)** — subsumido **con razón** en el campo permeable (v12). Revivir el regulador externo sería **regresión al control desde fuera**, justo lo que el proyecto abandonó (línea anti-Shannon). La meta homeostática ya vive en la dinámica del campo.
- **Modos espectrales: riqueza/entropía** — diagnóstico que **nunca alimentó un criterio de éxito**. GED (lo que sí entraba en validación) ya sobrevive subsumido.
- **CuerpoCalloso (clase escalar V123/124)** — su sustrato (EDO escalar de 1 g.d.l.) fue **descartado deliberadamente** (V125 volvió al campo vectorial V122). La función (acoplamiento gated por divergencia) ya sobrevive inline. *Matiz: la rectificación direccional `max(0,diff)` sí se perdió — registrable, no urgente.*
- **MemoriaRelacional / BufferAcoplamiento-trayectoria / MemoriaLargoPlazo** — **fuera de alcance** del organismo individual (son inter-organismo / díada). Pertenecen a la capa relacional V182, no a la célula.

---

## Cierre

- **Mecanismos perdidos catalogados:** ~21 (clases + señales de campo).
- **Nuevos (no estaban en la lista de los conocidos —campo rico/W dual, ganglio, Λ, inhibición lateral, hemisferios, `Phi_int_historia`—):** **~15**, entre ellos los **predictores de trayectoria validados**, la **relajación-a-centro validada**, `ExploradorActuadores`, la **Atención**, `MembranaSensorial`, `CuerpoCalloso`/`InhibicionReciproca` (clases), `CbGlobal`, el **Pastor**, y `act_perm/mant`.
- **Ya rescatados (en v2):** oscilador, W dual, `Phi_int_historia`, Λ, inhibición lateral.
- **Candidatos (A) por valor:** (1) **predictores de trayectoria** (validados, sin reemplazo medido), (2) relajación-a-centro (validada, parcial), (3) `act_perm`, (4) `CbGlobal`.
- **Guarda final:** ninguno "resuelve" nada; son candidatos clasificados con evidencia de código. La restauración se decide con este inventario en mano.

*Procedencia: lectura directa de v11, v12, v70, v72c, v80/v80h, v81, v97.2, V108, V109, V111, V118, V120, V121, V122/b, V123, V124, V125, V134–V139, V146, V168, V174, V176, V180c, V182A-v2/v7/v10/v13, V182A5, y el organismo actual v1/v2.*
