# INFORME · Experimento de la TRÍADA (ANIMA) — ¿de-a-tres produce diferencia estructurada?
### Con validación por réplica. Resultado: NEGATIVO honesto.

> **Fecha:** 30-jun / 1-jul-2026 · **Equipo:** Cosmolab (Alexis) + Claude, a partir de una pregunta del
> equipo **Cosmogénesis** y un aporte de **Gemini**.
> **Veredicto consolidado:** **NO (provisional)** — el resultado preliminar positivo **NO replicó**;
> fue un falso positivo. No hay evidencia robusta en este setup.

---

## 1. La pregunta (Cosmogénesis)

> ¿Puede una relación **de a tres** producir **diferencia estructurada**, sin que le metamos la diferencia
> de antemano?

En su física, un mediador "de a tres" **sostiene** pero **no diferencia** (pegamento tipo Higgs). La
frontera —dijeron— no la decide la física, la decide la Teoría corriendo en un sistema vivo. Así que la
respondimos con los organismos.

## 2. Traducción a ANIMA

- **De a dos** = la díada A↔B (se hablan por la voz): relación simétrica, refuerzo sin estructura.
- **De a tres** = un **tercero que media y PERSISTE y CAMBIA**: la **palabra acuñada** (S>0 propia, se
  re-usa, se re-acuña). A — palabra — B.

**Hipótesis:** si A y B parten idénticos compartiendo una **palabra-semilla neutra**, ¿la palabra se
**diferencia** (se usa desde estados internos distintos en A y B, o se especializa) por **pura historia**,
sin inyectar la diferencia?

**Mecanismo propuesto por Gemini (Plano C):** la diferencia emergería si **especializar la palabra le baja
el IRDE** (riesgo) a cada organismo — la estructura como atractor de menor energía.

## 3. Diseño

Vocabulario base reducido a **solo la semilla** (para que sea el tercero dominante); díada A↔B por canal
de voz; cada condición parte de **cero y simétrica** (down+wipe de los volúmenes de A y B).

| Cond. | Tercero | Control |
|---|---|---|
| **C1** TRIADA_VIVA | palabra que puede acuñar/emular (S>0, engendra ecosistema) | real |
| **C2** DIADA_SOLA | banco vacío (sin tercero) | real |
| **C3** PALABRA_CONGELADA | tercero **estéril** (solo la semilla; ablación `ANIMA_NO_ACUNAR`) | real |
| **C4** SHUFFLED | como C1, historia barajada | **falsador** |

**Observables (emergentes, anti-Shannon):** (1) especialización funcional (estado interno al emitir la
semilla, divergencia A vs B y si crece), (2) roles emergentes (asimetría de uso, quién acuña/emula),
(3) atractor IRDE (¿usar la palabra baja el riesgo?).

Nota de implementación: para un tercero **de verdad estéril** (C3) hubo que gatear **dos** vías —acuñación
(`palabra_X`) **y** emulación/imitación (`apr_X`)—; con una sola, el ecosistema se colaba por imitación.

## 4. Lo que pasó

### Corrida 1 (30-jun) — preliminar: parecía **SÍ**
- C1: asimetría de roles **0.515** (A emitió la semilla 78×, B 25×); atractor IRDE presente (usar la
  semilla dejaba a A en 4× menos riesgo: 0.0044 vs 0.0171); C4 shuffled aplanaba ambos.
- Verdicto preliminar: SÍ. **Marcado explícitamente como preliminar, pendiente de validación** (faltaba C3
  y una réplica).

### Corrida 2 (1-jul) — validación con C1 + C3 + C4: **NO replicó**

| cond | asim. roles | atractor IRDE | lectura |
|---|---|---|---|
| C1 (viva) | **0.011** (simétrico) | ausente | **sin estructura** |
| C3 (estéril) | −0.002 | ausente | sin estructura |
| C4 (shuffled) | **0.321** | ausente | la asimetría cayó en la condición **equivocada** |

## 5. Interpretación honesta

- La **asimetría de roles se comporta como RUIDO**, no como estructura: rompe la simetría al azar y
  aterriza en distinta condición cada corrida (en la 1 en C1; en la 2 en C4/shuffled, justo donde NO
  debería si el efecto fuera histórico).
- El **atractor IRDE de Gemini no reapareció** en la réplica.
- La **distancia-de-estado** crece en las tres condiciones → es deriva/ruido, no discrimina (por eso se
  descartó como criterio; un bug del veredicto automático inicial, corregido).
- **C3 estéril**: sin estructura, consistente con "no hay señal".

**El SÍ preliminar fue un FALSO POSITIVO.** No hay evidencia robusta de que una relación de a-tres con
palabra-viva produzca diferencia estructurada en este setup (historias de 900 s).

## 6. Dónde queda la frontera (dos lecturas, sin decidir)

1. **La frontera es honda:** ni un tercero vivo diferencia por sola historia — el de-a-tres, en ANIMA, es
   pegamento sin estructura, como el mediador pasivo de Cosmogénesis.
2. **El setup es débil:** 900 s es poco; A/B quizá demasiado simétricos para un *symmetry-breaking*
   estable; o el observable no capta la diferenciación real. Un diseño más fuerte (historias largas, test
   limpio de ruptura de simetría, muchas réplicas) podría cambiarlo.

## 7. Valor metodológico

La validación hizo exactamente su trabajo: **cazó el falso positivo**. Es *ciencia por resistencia, no por
confirmación* — se publica el fracaso como hallazgo. Un negativo verdadero vale más que un positivo bonito
y falso.

## 8. Artefactos

- Harness: `Célula_Madre/experimentos/experimento_triada_palabra.py` (+ `semilla_raiz.wav`).
- Análisis: `Célula_Madre/experimentos/analizar_triada.py`.
- Ablación: `ANIMA_NO_ACUNAR` en `VST_OrganoComunicacion.py` (gatea acuñar **y** emular) + compose.
- Datos: `~/Downloads/ANIMA_TRIADA_20260630_182417` (corrida 1) y `..._20260701_090604` (validación).

## 9. Pendiente (para retomar)

- **Decisión:** ¿cerrar como negativo honesto, o montar el diseño fuerte (réplicas + historias largas)
  para una última oportunidad limpia?
- **Mensaje a Cosmogénesis y Gemini** con el resultado honesto (lo probamos más duro y no aguantó).
