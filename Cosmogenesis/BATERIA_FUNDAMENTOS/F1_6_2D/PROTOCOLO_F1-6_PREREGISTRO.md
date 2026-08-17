# PROTOCOLO F1-6 — Persistencia en 2D: ¿es un artefacto del anillo 1D?

**Pre-registrado:** 2026-07-24, ANTES de escribir el motor (`F1_6_motor_2d.py`).
**Autor:** CC (ejecutor F1-6), sobre diseño de CS en
`BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md`, sección F1-6 (líneas 110-118).
**Regla T3:** si el experimento falla contra este criterio, se reporta el FAIL. Este
documento NO se edita después de ver resultados.

---

## 1. Pregunta

F1-1 (autocorrelación de forma × magnitud contra NULL barajado) se probó en un anillo
1D (`cs074_rcruz.py`). ¿El mismo mecanismo de persistencia (diferencia congelada por
expansión que corta aristas) aparece también en una malla 2D toroidal, con el mismo
observable adaptado? Evita la trampa T0 (la dimensión no se impone a mano — se prueba
en otra geometría y se mide qué pasa).

## 2. Sustrato físico (idéntico en espíritu a CS074-rcruz, generalizado a 2D)

- Malla `L×L`, condiciones periódicas en ambos ejes (toro), vía `np.roll` (sin
  topología nueva — es la generalización obvia del anillo 1D a 2D, como referencia de
  estilo usa `suite_epocas_masa_v6_mass_linaje.py`, SOLO por el patrón `np.roll`
  toroidal de aristas horizontales/verticales; no se copia su física de masa/átomos).
- Campo `φ` = fondo uniforme (=1) + `ε·perturbación` multi-modo (5 modos, números de
  onda enteros aleatorios `(kx,ky)∈{1,2,3}`, fase aleatoria, normalizada a std=1 antes
  de escalar por ε). Generalización directa del `campo_inicial` 1D de CS074.
- Dos arreglos de aristas booleanas: `ar` (horizontales, borde entre `(i,j)` y
  `(i,j+1 mod L)`) y `ad` (verticales, borde entre `(i,j)` y `(i+1 mod L, j)`). Ambas
  empiezan 100% vivas (malla 4-conexa completa).
- **Difusión:** cada celda se mezcla con el promedio de sus vecinos activos (hasta 4),
  mismo esquema vectorizado de CS074 (factor de mezcla 0.5), extendido a los 4 vecinos.
  Solo ocurre por aristas vivas.
- **Expansión:** cada arista viva (horizontal o vertical) se corta independiente con
  probabilidad Bernoulli `H` por paso (misma corrección de CS074 frente al
  `round(H·N)` roto del CS074 original — válido también cuando `H·L² ≪ 1`).
- `D` = fracción de contraste (std) borrada en UN paso de difusión pura (`H=0`),
  medida del propio campo, no impuesta.
- `pasos_lavado`: igual que CS074 — tiempo medido (mediana de semillas, ×1.15 de
  margen) para que a `H=0` la persistencia caiga bajo `P_LAVADO=0.05`. Es el gate de
  validez del control r=0 (T4).
- `r = H/D` (razón expansión/reabsorción). Eje primario que CRUZA r=1, igual grid que
  CS074-rcruz: `r_targets = [0, 0.1, 0.3, 0.5, 1, 2, 5, 10, 30, 100]`.

## 3. Observable (F1-1 adaptado a 2D — "mismo observable")

`P = c_isotropo · v`, donde:
- `c_isotropo = max(0, 0.5·(corr(φ, roll(φ,1,eje=1)) + corr(φ, roll(φ,1,eje=0))))`
  (autocorrelación a primer vecino, promediada entre eje horizontal y vertical —
  generalización isotrópica de la autocorrelación a primer vecino 1D de CS074).
- `v = var(φ_final) / var(φ_inicial)` (razón de varianza, magnitud del contraste
  sobrevivido).

La cantidad medida (correlación espacial + varianza del campo) es distinta de las
variables que la juzgan (ε, r, L) — evita T2.

## 4. NULL

Permutación 2D: `φ` se aplana, se permuta (`rng.permutation`), se reforma a `L×L`, al
FINAL de la evolución (igual criterio que CS074: destruye forma, conserva histograma).

## 5. Barrido pre-registrado

- `ε ∈ {0, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.5, 1.0}` (8 puntos, mismo grid que
  `cs074_rcruz.py modo=produccion`, para comparabilidad directa 1D↔2D).
- `r_targets` = los 10 puntos de arriba, cruzando r=1 (idéntico a CS074-rcruz).
- `L ∈ {32, 64, 128}` (según spec F1-6).
- `semillas ≥ 8` por punto (spec pide ≥8; se usan 8).
- `pasos`: calibrado por L a partir de `pasos_lavado` medido en ε=1e-3 (mismo criterio
  que CS074 `modo=produccion`), fijo para todo el grid de ese L.
- Orden de ejecución: **smoke L=32 primero** (grid reducido, para validar el motor)
  antes de escalar a producción L=32 → L=64 → L=128 (el más caro).

## 6. Verificación cruzada (triple, regla general de la batería)

1. **NULL:** debe caer cerca de 0 y separarse de REAL en la banda r≫1 (T4).
2. **Consistencia dimensional 1D↔2D:** el r* (umbral de congelamiento) medido en 2D
   debe caer en el MISMO RÉGIMEN que el r* conocido de CS074/CF-1 en 1D (r≈0.1 es
   donde enciende, según runs previos `cs074_rcruz_produccion_resultado.json` /
   `RESUMEN_CS074_rcruz_PARA_CS.md`). Esta comparación cross-dimensional cumple el rol
   del "segundo método" — la geometría distinta es la verificación independiente
   pedida por el enunciado propio de F1-6 (líneas 115 del documento madre).
3. **Auditoría en disco:** código (`F1_6_motor_2d.py`, sin editar tras revisión) +
   JSON crudo con todas las filas (P_real, P_null, z, D, H, r efectivo, pasos) para
   que CS audite sin depender de lo que este informe declare.
4. Control `eps=0 → P≈0` a todo r (control T1/T4 adicional, igual que CS074).
5. Control `r_target=0 ∧ eps>0 → P_real` bajo (la difusión debe lavar) — gate de
   validez del cruce, igual `control_r0_ok` de CS074.

## 7. Criterio de PASS pre-registrado (congelado, no se toca tras ver datos)

- **PASS (persistencia dimensionalmente robusta):** existe una banda de r donde
  `P_real ≫ P_null` (separación con z apreciable, mismo criterio cualitativo de
  CS074 — no gate binario duro, T5), Y esa banda aparece en el mismo régimen
  (r no lejos de ~0.1-1, no en r≫100 ni solo en r=0) que el r* 1D conocido, en LOS
  TRES tamaños L. Y los controles ε=0 y r=0(ε>0) caen como se espera.
- **FAIL / hallazgo negativo (a reportar igual, sin suavizar):** si la persistencia no
  aparece en 2D, o aparece en un régimen de r cualitativamente distinto al 1D
  (p.ej. solo en r≫100), o si el NULL no cae, o si el resultado es inestable entre
  L=32/64/128 (dispersión grande sin patrón).
- Ambos resultados (PASS o FAIL) se reportan con la curva completa — no se auto-
  adjudica veredicto; el veredicto final lo da CS con la curva cruda.

## 8. Qué NO se hace (límites explícitos)

- No se impone dimensión ni cuantos a mano (T0): la única dimensión introducida es la
  de la MALLA (2D, dato del experimento, no del observable).
- No se elige ningún coeficiente para forzar el resultado (T1): ε y r se barren; D y
  pasos_lavado se MIDEN.
- No se toca `cs074_rcruz.py` ni `suite_epocas_masa_v6_mass_linaje.py` (solo lectura,
  el segundo es referencia de estilo `np.roll` toroidal, no de física).
- No se abre topología nueva (grafo no-toroidal, dimensión >2, etc.) — fuera de scope
  de F1-6.
- No hay ruido dinámico por paso (eso es F1-5, otro experimento del paralelo; F1-6 solo
  pide robustez por L y semillas según el documento madre).
- No commits, no se editan archivos existentes.

## 9. Rutas de salida

- Motor: `F1_6_motor_2d.py`
- Resultados crudos: `F1_6_resultado_smokeL32.json`, `F1_6_resultado_L32.json`,
  `F1_6_resultado_L64.json`, `F1_6_resultado_L128.json`
- Este protocolo: `PROTOCOLO_F1-6_PREREGISTRO.md` (este archivo)
