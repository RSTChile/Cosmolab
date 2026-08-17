# PROTOCOLO E5.1-2 — Vida media de la exergía (τ vs D, sin expansión)

**Experimento:** E5.1-2 · "Vida media de la exergía: ¿cuántos pasos tarda X en decaer sin expansión?"
**Base:** `Cosmogenesis/cs074_rcruz.py` (NO editado; se importa como módulo).
**Agente:** CC E5.1-2 (batería Enfoque 5, 30 experimentos en paralelo).
**Fecha de congelamiento:** 2026-07-24, ANTES de correr la producción.

---

## 1. Pregunta y observable

Sin expansión (H=0 permanente, r=0), solo difusión pura del campo φ del `cs074_rcruz.py`
(`paso_difusion`, sin `paso_expansion`). Se mide **τ(D)**: el número de pasos hasta que la
exergía cae a la mitad de su valor inicial.

**Definición operativa de X (exergía, proxy)**: `X(t) = var(φ_t) / var(φ_0)` — la fracción de
varianza (desviación del equilibrio uniforme) que sobrevive. Esto es exactamente el término
`v` que ya calcula `persistencia()` en el código base (`v = phi.var()/contraste0**2`), reusado
sin modificarlo. No se inventa un observable nuevo: se aísla el componente ya presente en el
código, congelado ANTES de correr (T2: el observable no es su juez — X se mide por varianza,
el juez es el umbral 0.5 fijado aquí, no ajustado después).

**τ**: primer paso t (resolución `check_every`, calibrada por punto, ver §4) tal que X(t) ≤ 0.5.
Si no se alcanza dentro de `max_steps`, se marca **censurado** (τ ≥ max_steps, cota inferior,
NO se extrapola ni se inventa un valor).

## 2. Desviación pre-registrada del rango nominal D∈[1e-4…1e2] — DECLARADA ANTES DE CORRER

El D de esta plantilla de barrido (genérica, reusada en las 30 fichas) es, en el código base
tal como está escrito, **`D := medir_D(N,eps,seed)` = fracción de contraste (std) borrada en UN
paso de difusión**. Por construcción esa cantidad está **acotada en [0,1]** — nunca puede llegar
a 10 ni a 100. Se verificó empíricamente (calibración, no ajuste de resultados) antes de
congelar el grid:

| N | D medido (eps=1e-3, prom. 4 semillas) |
|---|---|
| 4 | 0.562 (con aliasing — el campo inicial usa 5 modos fijos m=1..5; N=4 no los resuelve, Nyquist violado) |
| 16 | 0.1021 (N mínimo que respeta Nyquist para m≤5: N≥2·5=10) |
| 3000 | 3.75e-6 |
| 6000 | 9.37e-7 |

**Máximo D alcanzable, respetando Nyquist (N≥16), con el código sin editar: ≈0.10.** No existe
ningún N que produzca D=1e2 ni D=10 ni D=1 con este operador de difusión (peso fijo 0.5 por
paso) — es un techo estructural del código base, no un error de mi implementación ni algo
ajustable. Esto se reporta como hallazgo de diseño, no se oculta ni se fuerza.

**Resolución adoptada:** en vez de forzar 6 décadas centradas donde no hay física posible,
sobre-dimensiono en la dirección que SÍ es alcanzable (N grande → D arbitrariamente chico) y
declaro el techo real. Grid final cubre D ∈ [9.4e-7 … 0.10] (D_min ya extiende ~2.4 décadas por
debajo del piso pedido de 1e-4; D_max es el techo estructural verificado, ~0.10). Total ≈5.0
décadas del observable D "medido en el código", en vez de la plantilla genérica de 6 décadas de
un D conceptual sin techo — desviación declarada, no oculta, con la razón física exacta arriba.
Este hallazgo se reporta a CS como posible desajuste de la plantilla genérica del documento
madre para experimentos cuyo D está definido así (fracción por paso, acotado).

## 3. Ley de escala usada solo para DISEÑAR el grid (no para reportar resultados)

Calibración previa (no es el resultado del experimento, solo dimensiona cuántos pasos hacen
falta para no cortar la simulación antes de tiempo): se midió τ·D ≈ 0.668 (constante) en un
piloto de 5 puntos N∈{200,500,1000,2000,4000}, consistente con decaimiento difusivo estándar.
Se usa **solo** para fijar `max_steps` por punto (margen ×3 sobre la estimación) — el τ
reportado en producción siempre es el medido directo por simulación, nunca el de esta fórmula.

## 4. Grid de producción (congelado)

**Tier 1 — curva primaria τ(D), la que se reporta como resultado central:**
- N ∈ {16,23,32,46,65,92,130,184,261,370,524,743,1053,1493,2116,3000} (16 puntos, log-espaciado)
- eps = 1e-3 fijo (mismo valor de referencia que usa `cs074_rcruz.py` en su propia calibración)
- **16 semillas** por punto (cumple el mínimo ≥16 del documento madre)
- Sin ruido dinámico (línea base limpia)
- `max_steps` = ceil(3 × 0.668/D_medido), `check_every` = max(1, min(50, max_steps//60))

**Tier 2 — invariancia a ε (chequeo, no la curva central):**
- 5 anclas de N ∈ {16,130,524,1493,3000} (cubren todo el rango de Tier 1)
- eps ∈ {0, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.5, 1.0} (idéntico a `eps_list` de producción de
  `cs074_rcruz.py`, reusado por consistencia, no elegido a mano para este experimento)
- 8 semillas por combinación
- **Justificación de reducir semillas aquí:** `paso_difusion` es un operador LINEAL sobre φ; con
  φ=fondo+eps·pert, escalar eps escala linealmente la perturbación y por tanto var(t)/var(0) es
  invariante a eps por construcción algebraica (no es una hipótesis, es una propiedad del código
  tal como está escrito). Este tier verifica esa predicción, no la re-descubre con la misma
  potencia estadística que la curva central.

**Tier 3 — extensión de D hacia abajo (confirmatorio, pocas semillas, caro en cómputo):**
- N ∈ {4500, 6000} (extiende ~0.7 décadas más allá del techo de Tier 1)
- eps = 1e-3, 6 semillas
- Declarado de antemano como de MENOR potencia estadística — el costo por punto en N=6000 es
  ~150× el de N=200 (τ escala ~1/D ~ N², costo total ~N³)

**Tier 4 — perturbación dinámica (T7, obligatorio):**
- 3 anclas N ∈ {16, 524, 3000}
- eps = 1e-3, ruido gaussiano aditivo por paso: φ += N(0, σ), σ = frac·eps,
  frac ∈ {0.01, 0.1} (relativo a la amplitud de la perturbación misma — no un número absoluto
  puesto a mano)
- 8 semillas
- Compara τ con Tier 1 (mismo N, sin ruido) para ver si el ruido dinámico rompe/desplaza la ley

## 5. PASS / lectura pre-inscrita

- **PASS primario:** τ(D) decrece MONÓTONAMENTE con D en Tier 1 (dentro de la dispersión entre
  semillas). Se reporta la ley completa (forma funcional que mejor ajusta, sin forzar τ=k/D si
  los datos no lo sostienen — se prueba también log-log slope).
- **Tier 2:** PASS si τ no difiere significativamente entre valores de eps (mismo N) más allá de
  la dispersión entre semillas — confirma invariancia. Si difiere, se reporta como hallazgo
  (rompe la predicción algebraica, sería en sí mismo interesante/sospechoso de revisar).
- **Tier 3:** PASS si la ley de Tier 1 se extrapola correctamente a D más chico.
- **Tier 4:** se reporta cuánto mueve el ruido dinámico τ respecto del baseline limpio — no hay
  "éxito/fracaso", es caracterización (igual que el experimento entero: no tiene NULL propio,
  según el documento madre).
- **No hay NULL para este experimento** (documento madre: "NULL: —"), es caracterización pura.
- Censura: cualquier punto con τ≥max_steps se reporta explícitamente como cota inferior, no se
  descarta ni se imputa.

## 6. Verificación cruzada (regla 4 de "REGLAS DE EJECUCIÓN")

1. NULL: no aplica (declarado arriba, consistente con doc madre).
2. Segundo método/observable: además de X=var-ratio, se registra también `std_ratio` (=√X) y la
   correlación espacial `c` que ya usa `persistencia()` de `cs074_rcruz.py`, para verificar que
   el cruce de 0.5 en varianza es consistente con el cruce en amplitud (√0.5).
3. Auditoría en disco: todos los resultados crudos (por semilla, no solo promedios) se guardan
   en JSON para que quien no escribió el motor pueda re-verificar τ directamente de las curvas
   X(t) muestreadas.

## 7. Axiomas E1/E2

No aplican directamente (este experimento no tiene expansión, por tanto E2 —redistribución por
expansión— está desactivado por diseño; E1 —conservación del presupuesto total— no se mide aquí
explícitamente porque el observable es X normalizado, no el balance E_total; el balance se
verifica en el Tema 2 de la batería, no en este experimento).

---

**Firma de congelamiento:** este documento se escribe y guarda ANTES de ejecutar el motor de
producción (`E5_1_2_motor_vida_media.py`). No se edita después de ver resultados (T3).

---

## ADENDA — Definición común de exergía (ARREGLO 3) + fix de ruido calibrado (ARREGLO 2), 2026-07-25

**No se edita el texto original arriba (T3): esta sección se agrega, no reemplaza.**

Por decisión del director (`INSTRUCCION_recorrer_5_definicion_comun_PARA_CC.md`): este
experimento se re-corre desde cero para medir, EN PARALELO al `tau` histórico (definición
propia de E5.1-2, §1 arriba: `X(t) = var(φ_t)/var(φ_0)`, sin factor de autocorrelación),
un `tau_canonico` calculado sobre la definición común de exergía de
`BATERIA_ENFOQUE5/_observables_homologadas.py`:

    Xh(t) = exergia_X(φ_t) / exergia_X(φ_0),   exergia_X(φ) = (1/N)·Σ(φᵢ-1)²

normalizada a su PROPIO t=0 (no a un umbral absoluto), porque a diferencia de `X` —que
empieza en 1.0 por construcción algebraica— `exergia_X` cruda no necesariamente vale 1 en
t=0. El criterio de "vida media" (primer cruce bajo la mitad) es idéntico en espíritu al de
`tau`, aplicado a `Xh` en vez de a `X`, sobre la MISMA trayectoria simulada (no se duplica
la simulación: ambas métricas se miden sobre el mismo φ(t) en cada paso).

**Predicción pre-registrada (T3, declarada ANTES de correr):** bajo H=0 permanente (difusión
pura, grafo anillo siempre conectado), `paso_difusion` conserva Σφ exactamente (promedio
local lineal), y `campo_inicial` resta la media de la perturbación (`pert -= pert.mean()`)
antes de escalarla por ε — por lo tanto la media de φ es exactamente 1.0 en t=0 y se
mantiene exactamente 1.0 en todo t (salvo error de punto flotante). Bajo esa condición,
`exergia_X(φ) = (1/N)Σ(φᵢ-1)² = (1/N)Σ(φᵢ-mean(φ))² = var(φ)` (varianza poblacional, mismo
denominador N que usa `phi.var()` de numpy) — es decir, para ESTE experimento específico
(H=0, perturbación de media cero), `Xh(t)` y `X(t)` deberían ser ALGEBRAICAMENTE IDÉNTICAS
(hasta redondeo de punto flotante), y por tanto `tau_canonico == tau` en cada corrida. Esto
NO es un resultado forzado ni un atajo: es una propiedad estructural de este experimento en
particular (a diferencia de E5.5-3, donde la re-inyección de energía SÍ mueve la media de φ
y por eso `X` y `Xh` divergen ahí). Si la corrida real muestra una discrepancia entre `tau`
y `tau_canonico` mayor a ruido de punto flotante, se reporta como hallazgo genuino (posible
violación de conservación de Σφ, a auditar), no se descarta.

**Arreglo 2 (ruido calibrado) — aplica a Tier 4:** el ruido dinámico por paso
(`sigma_ruido = frac·eps`, CONSTANTE durante `max_steps` pasos) se reemplaza por
`ruido_por_paso(frac, eps, max_steps)` de `BATERIA_ENFOQUE5/_ruido_calibrado.py`
(`= frac·eps/√max_steps`), que mantiene la varianza acumulada total ≈`(frac·eps)²`
independiente de N y de `max_steps`, en vez de crecer sin control con N (Tier 4 barre
N∈{16,130,524}, con `max_steps` variando ~3600× entre esas anclas — el bug era severo aquí).
`frac` (0.01, 0.1) NO cambia de significado ni de valor — solo cómo se reparte en el tiempo.

**Réplica exacta de las combinaciones históricas (no del diseño preregistrado completo):**
el motor original (`E5_1_2_motor_vida_media.py`) fue matado (SIGTERM limpio) durante Tier 2
en N=1493/eps=1e-9, y el diseño preregistrado completo (Tier 2 con N=3000 incluido, Tier 3
con N∈{4500,6000}×6 semillas) NUNCA se terminó de correr — ver
`E5_1_2_motor_extension.py` y la sección "desviaciones_declaradas_del_pre_registro" de
`E5_1_2_resultado_CONSOLIDADO.json`. Para que la comparación lado-a-lado sea honesta (mismo
número de combinaciones, mismos N, mismos eps, mismas semillas que el resultado histórico
que se está comparando), esta re-corrida reproduce EXACTAMENTE las combinaciones que
terminaron guardadas en disco la vez anterior, no el diseño aspiracional abandonado:
- Tier 1: sin cambios (16 N × 16 semillas × eps=1e-3, completo, como preregistrado).
- Tier 2: N∈{16,130,524} × 8 eps completos (incl. eps=0 trivial) × 8 semillas; N=1493 con
  SOLO eps∈{0, 1e-9} (los 2 puntos que alcanzaron a terminar antes del SIGTERM); N=3000
  ausente (igual que en el histórico).
- Tier 3: solo N=4000 × 4 semillas, eps=1e-3 (el recorte ya documentado en
  `E5_1_2_motor_extension.py::tier3_recortado`, N∈{4500,6000} nunca corrió ni en el
  histórico ni aquí).
- Tier 4: N∈{16,130,524} (excluye N=3000, igual que
  `E5_1_2_motor_extension.py::tier4_completo`) × frac∈{0.01,0.1} × 8 semillas, con el fix de
  ruido calibrado aplicado.

`E5_1_2_motor_extension.py` queda como registro histórico de la corrida truncada del
2026-07-24 (no se ejecuta de nuevo); toda la lógica de Tier 3/Tier 4 recortados se
incorporó directamente a `E5_1_2_motor_vida_media.py` para tener un único motor
autocontenido para esta re-corrida.

**Almacenamiento ampliado:** Tier 1 (tier primario) ahora guarda la curva completa `X(t)` Y
`Xh(t)` (más `c(t)`) para las 16 semillas de cada N (antes solo `curva_ejemplo_seed0`).
Tiers 2-4 guardan `tau` y `tau_canonico` (con censura de cada uno) por cada corrida
individual, sin curvas completas (el tamaño para N grande en Tier 2/3 lo desaconseja —
criterio de la regla 3 del encargo).

Los JSON de resultado previos a este arreglo se conservan como
`E5_1_2_resultado_*_DEFINICION_VIEJA_pre_ARREGLO3.json` (no se borran).

Este documento se guarda ANTES de re-ejecutar el motor (T3).
