# `cs072_modulos/piezas/` — las fuerzas, cada una en su módulo

**Regla de la carpeta (desde el diseño original de CS072, reafirmada en la instrucción de cierre
holístico CS073):** cada fuerza fundamental es un módulo AISLADO, con interruptor on/off por
nombre-clave (`apagar=frozenset(["3_fuerte", ...])`), para poder corregir o auditar una sin tocar las
demás — y para poder correr el "guardián anti-Shannon": apagar una fuerza debe destruir SÓLO la
estructura que esa fuerza produce.

## Piezas del Modelo Estándar (motor CS072, VALIDADAS — no tocar)

| módulo | nombre-clave | qué liga / mide | nivel |
|---|---|---|---|
| `p02_gravedad.py` | `2_gravedad` | teje la red ÁTOMO-ÁTOMO (`Bgrav`) por umbral de sobredensidad + localidad TÉRMICA (un escalar 1D). Es la gravedad **relacional-cuántica**: régimen pre-métrico, sin posición. Da un hub invariante (densidad de red ~0.500 en 4 escalas) — es lo CORRECTO para su régimen, no un bug (ver `CS073_prototipo_estructura_hallazgo_CS.md`: dos prototipos probaron que este régimen NUNCA fragmenta). | átomo |
| `p03_fuerte.py` | `3_fuerte` | confina quarks en tríos RGB (bariones); fuerte residual liga nucleón-nucleón en frío profundo (`Bnuc`, forma He). | quark / nucleón |
| `p04_em.py` | `4_em` | recombinación electrón+protón → H (`Bem`). Sin esta pieza no hay átomos → la geometría COLAPSA (`dim_acoplada=None`), verificado. | quark/electrón |
| `p08_aniquilacion.py` | `8_aniquilacion` | materia-antimateria se cancela por RESTA de poblaciones (color+sabor), no por tasa — deja el excedente bariónico. | quark |
| `p23_fluctuaciones.py` | `23_fluctuaciones` | fija el campo de densidad #23 (rugosidad del plasma) al inicio — la condición inicial de la que sale toda la estructura posterior. | quark (inicial) |
| `p24_tiempo.py` | (lector, no fuerza) | `tiempo_emergente()`: el tiempo nace CON el primer átomo neutro (transición irreversible), no antes. Trae la regla **anti-contrabando geométrico** que gobierna todo el arco (no cablear constantes de nuestro universo). | global |

## Experimento deprecado (registro histórico, no forma parte del motor)

| módulo | estado | por qué se conserva |
|---|---|---|
| `p02b_gravedad_general.py` | **DEPRECADO** (Paso A de CS073, superado) | Intentó desplegar posiciones 3D embebiendo (MDS) las distancias de la malla causal. Resultado: negativo estadísticamente sólido (z=-0.34 / z=0.76 contra distribución de 8 NULLs, a 750 átomos reales — ver `verificar_p02b_pasoA_escala.py` y su output). Adjudicación Q3 (`INSTRUCCION_CC_cierre_holistico.md` v3, 19-jul): las posiciones NO se derivan de la malla causal; el 3D es el ESCENARIO fosilizado (ya probado en CS072), no algo a re-derivar. Superado por `p_gravedad_general.py`. |

## Piezas del experimento de cierre CS073 (átomo → primera estrella) — NUEVAS, 19-jul-2026

Régimen MÉTRICO (post-fósil): actúan sobre posiciones 3D reales, nunca sobre un escalar como proxy de
cercanía. Orquestadas desde `cs073_cierre_holistico.py` (en la raíz) — ver ese archivo y
`ARQUITECTURA_EXPERIMENTO_CS.md` para cómo se combinan en un solo bucle temporal (Regla 1 de
`INSTRUCCION_CC_cierre_holistico.md`).

| módulo | qué hace | distinción clave |
|---|---|---|
| `p_gravedad_general.py` | `GravedadGeneral`: F=G·m·m/r² vectorizado (numpy) sobre posiciones reales. `posiciones_escenario()`: escenario 3D UNIFORME (D=3, semilla fija 12345, independiente de la densidad — el contenedor neutro). G_ADIM=1 (convención de unidad, no una medición), softening=0.3 (necesidad numérica, ajustado tras un chequeo de cordura que mostró inestabilidad con softening=0.1/dt grueso). | gravedad sobre MASA×POSICIÓN, no sobre umbral térmico |
| `p_materia_oscura_halo.py` | `MateriaOscuraHalo`: segunda especie que sale del MISMO generador `densidad_intrinseca` (catalogo.py, reusado), en el mismo escenario, bajo la MISMA gravedad — pero `siente_em=False`: nunca pasa por presión ni enfriamiento. Esa asimetría de ACOPLAMIENTO es lo que la hace colapsar antes, no una posición plantada (G-SIN-SIEMBRA). Masa 1:1 con bariones (no se importa el 5:1 real de ΛCDM). | CDM emergente, no sembrado |
| `p_enfriamiento_H2.py` | `EnfriamientoH2`: presión térmica (calentamiento por compresión, medido vía densidad local dinámica con KDTree) + canal de enfriamiento H₂ (interruptor `activa_cooling`) que relaja la temperatura hacia un piso SÓLO donde hay sobredensidad. Soporte de presión = agitación térmica isótropa (`kick_termico`), simplificación declarada (no es un solver SPH de gradiente de presión). SÓLO actúa sobre partículas con `siente_em=True`. | el canal que permite fragmentar, no sólo calentar |
| `p_expansion.py` | `Expansion`: factor de escala `a(t) = T0/T(t)`, derivado del MISMO reloj de enfriamiento que ya usa `nucleo.Estado.enfria` — ninguna ley nueva. Estira posiciones isótropamente cada paso (G-EXPANSION-ISOTROPA). | el "lejos" — y lo que vuelve el 3D cuántico en 3D macroscópico (adjudicación Q3) |

## Resultado registrado de la primera corrida holística (19-jul-2026)

REAL vs NULL (campo #23 barajado, n=8): sin diferencia significativa (z≈-0.35/-0.41) — diagnosticado
como un límite de DISEÑO, no de física: las posiciones son independientes de la densidad por
construcción (adjudicación Q3), así que barajar la densidad no destruye ninguna coherencia espacial que
nunca existió. Control positivo (masa real sin pesar por #23, más tiempo cosmológico): **0 estructuras,
no emerge estrella** a esta escala (250+250 partículas) — ver `cs073_cierre_holistico.py`
(`correr_real_vs_null`, `correr_control_positivo`) y el hilo de reporte a CS. Pendiente de adjudicación;
nada de esto está cerrado (`NOTA_PERMANENTE_CS.md`).
