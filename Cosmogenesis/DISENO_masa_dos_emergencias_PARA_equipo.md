# Rediseño del experimento de Masa — las dos emergencias en su época correcta
### Diseño conceptual para revisión del equipo (NO correr aún)

**Preparado por:** Claude Science (CS) · **Fecha:** 23-jul-2026
**Base:** LINEA_TIEMPO_MASA_topologia_vs_fisica.md + INFORME_CONSOLIDADO_MASA_ME.md
**Estatuto:** diseño para que el equipo lo revise ANTES de implementar. No lo corro yo.

Siglas: **ME** = Modelo Estándar · **CDC** = Cromodinámica Cuántica (fuerza fuerte) ·
**VEV** = valor esperado en el vacío del campo de Higgs.

---

## 1. QUÉ ESTABA MAL, EN UNA LÍNEA

El experimento E4 puso **una** emergencia de masa **al final** (tras el átomo + la
gravedad). La física tiene **dos** emergencias, **ambas antes del átomo**:
- **① Higgs (~10⁻¹¹ s):** masa de las partículas elementales = el vacío cambia de fase.
- **② Confinamiento / CDC (~10⁻⁵ s):** el ~99% de la masa del nucleón = energía de
  ligadura del campo fuerte.

El rediseño ubica cada una donde va, y las mide como **emergentes**, no impuestas.

---

## 2. PRINCIPIO RECTOR (lo que NO se repite de v6)

El fracaso de v6 fue definir la "masa" con las mismas variables del discriminante de
linaje → circular. La regla de este diseño, absoluta:

> **La masa NO puede construirse a partir del observable que la juzga.** Debe ser una
> cantidad con significado físico propio (energía), medida, y el juez debe ser
> independiente de la fórmula de masa. Pre-registrado ANTES de correr.

Y el anti-Shannon de siempre: **cero números del ME como blanco** (nada de GeV, 1/1836,
7:1, 125). Las escalas de temperatura/tiempo son reporte, no perillas. Todo contra NULL.

---

## 3. DISEÑO — DOS MÓDULOS, EN ORDEN CRONOLÓGICO CORRECTO

### MÓDULO A — 1ª emergencia (tipo Higgs): la masa como CAMBIO DE FASE del vacío

**Qué modela:** en el ME la masa elemental aparece cuando un campo de fondo pasa de
VEV≈0 (simetría) a VEV finito (roto) al enfriarse. No es que "se cree masa de la nada":
es que un parámetro de orden se enciende.

**Cómo, sin Shannon:**
- Un campo de fondo φ_vac con un parámetro de orden que puede valer 0 o finito.
- Se **barre la temperatura** (enfriamiento = expansión, como en CS074). NO se impone
  cuándo se rompe.
- **Observable de masa #1:** la masa de un modo = su acoplamiento al VEV emergente,
  `m₁ ∝ |φ_vac|`. Con VEV=0 → m₁=0 (sin masa, fase simétrica); con VEV finito → m₁>0.
- **Se MIDE si la ruptura emerge del enfriamiento** y si es **crossover suave**
  (predicción del ME) o salto. La literatura dice crossover — si sale salto, es señal
  de que el modelo no captura la física.
- **NULL:** barajar el orden del enfriamiento / aleatorizar el fondo → el VEV no debe
  encenderse coherentemente. Si REAL enciende y NULL no, la ruptura es real.

**Candado:** `m₁` se define por el VEV, NO por ningún discriminante de estructura. El
juez ("¿emergió el VEV?") mide |φ_vac|, que es independiente de cómo se mida la masa.

### MÓDULO B — 2ª emergencia (tipo CDC): la masa como ENERGÍA DE LIGADURA

**Qué modela:** el ~99% de la masa del protón es energía del campo de gluones que
confina los quarks. La masa **es** energía de ligadura, no sustancia. Y ocurre en el
**confinamiento (~10⁻⁵ s)**, NO tras el átomo.

**Cómo, sin Shannon:**
- Cuando un cierre (k elementos) se forma por confinamiento, tiene una **energía de
  ligadura** = el trabajo para separarlo (cuánto "cuesta" romperlo).
- **Observable de masa #2:** `m₂ = energía_de_ligadura del cierre` — una cantidad
  física (energía), medible directamente de la dinámica de confinamiento, **totalmente
  independiente del linaje** (no usa co_member ni n_long_co — ese fue el error de v6).
- Se mide en la **época de confinamiento**, no en E4. La masa ya existe ahí.
- **Predicción falsable fuerte (del ME):** m₂ debe ser **mucho mayor** que la masa de
  los constituyentes sueltos (porque el 99% es ligadura, ~1% constituyente). Si el
  cierre pesa ≈ suma de sus partes, el modelo NO reproduce el mecanismo de CDC. Si pesa
  ≫ suma, sí. **Esto es un test, no un ajuste** — no fijamos el 99%, medimos la razón.
- **NULL:** un cierre con la misma composición pero **enlaces barajados** (sin la
  estructura de confinamiento). Su energía de ligadura debe caer. Si REAL ≫ NULL, la
  masa-ligadura es de la estructura de confinamiento, no del conteo.

**Candado:** `m₂` es energía (trabajo de separación), medida del campo, no un
reempaquetado del discriminante. El juez es la razón ligadura/constituyente, fijada
como criterio **antes** de correr.

---

## 4. LO QUE ESTE DISEÑO ADMITE QUE NO PUEDE HACER (honestidad)

- **NO reproducirá masas en GeV ni el 1/1836** — y no debe intentarlo (son parámetros
  libres que ni el ME predice). Mide *mecanismos* (cambio de fase; ligadura ≫
  constituyente), no *números*.
- **La topología sigue sin dar propiedades físicas** — el muro que ya conocemos. Estos
  módulos modelan el *mecanismo* de emergencia de masa como observable emergente; no
  convierten un cierre topológico en un protón real. Es un análogo de mecanismo, y debe
  declararse como tal.
- **Módulo A (Higgs) es el más especulativo:** requiere un campo de fondo con parámetro
  de orden, que es estructura nueva — hay que vigilar que no sea Shannon encubierto (el
  VEV debe emerger del enfriamiento, no encenderse a mano en un tiempo fijado).

---

## 5. GUARDIANES (para la revisión del equipo)

- **G-MASA-ES-ENERGIA-NO-DISCRIMINANTE:** la masa se define como energía (ligadura /
  acoplamiento a VEV), nunca a partir del observable que la juzga (lección de v6).
- **G-JUEZ-INDEPENDIENTE-PREREGISTRADO:** el criterio de PASS se fija y se escribe
  ANTES de correr; si falla, no se cambia el juez (lección de v5→v6).
- **G-DOS-EMERGENCIAS-EN-SU-EPOCA:** masa #1 en ruptura (~10⁻¹¹ s), masa #2 en
  confinamiento (~10⁻⁵ s); NINGUNA tras el átomo.
- **G-MECANISMO-NO-NUMERO:** se miden razones y encendidos (crossover, ligadura≫
  constituyente), nunca valores del ME como blanco.
- **G-CROSSOVER-NO-SALTO:** la predicción del ME es crossover suave; si el modelo da
  salto nítido, es un dato en contra, no algo a suavizar.
- **G-NULL-MUERDE:** cada masa contra su NULL (fondo barajado / enlaces barajados);
  nulo = hallazgo.

---

## 6. PREGUNTA PARA EL DIRECTOR ANTES DE IMPLEMENTAR

Tres opciones de alcance, para que el equipo no arranque a ciegas:
- **(a)** Implementar solo el **Módulo B** (masa = energía de ligadura en confinamiento)
  — es el más firme (energía real, NULL claro, en la época correcta) y ataca el 99% de
  la masa. El Módulo A (Higgs) queda para después.
- **(b)** Implementar **ambos módulos** encadenados en la época correcta.
- **(c)** Antes de codificar nada, discutir si el Módulo A (campo de fondo tipo Higgs)
  es siquiera modelable sin meter estructura a mano — porque roza el límite del muro.

Recomendación de CS: **(a) primero** — Módulo B es donde la física es más clara (masa =
energía de ligadura), la época es inequívoca (confinamiento), y el candado anti-v6 es
más fácil de sostener (energía ≠ discriminante). El Módulo A es el terreno resbaladizo.
