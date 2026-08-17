# NULL-1 — piloto de distribución radial (Fase II CS073, escalón 1 de 6)

**Encargo:** primer escalón de la jerarquía de 6 controles propuesta por el análisis externo (5
analistas de IA) para blindar CS073 (z=48.69, REAL 2.95× la masa en sumideros del promedio NULL, batería
N=2000 en `/Users/alexis/phantom_cs073/bateria_n2000/`). Objetivo de NULL-1: aislar si la ventaja de REAL
viene de la correspondencia relacional específica (quién quedó causalmente cerca de quién) o si basta con
la forma global de la nube (distribución radial/perfil de densidad) para explicar la diferencia.

No se declara cierre ni veredicto sobre CS073 ni sobre este escalón — sólo se reportan números. La
lectura es de Alexis.

---

## Paso 1 — ¿Qué destruyen realmente los NULL1-8 existentes?

**Método de los NULL1-8 existentes** (leído en `cs072_modulos/piezas/p_semilla_causal.py`,
`fase1_traducir_a_phantom.py`, `campo_velocidad_turbulento.py`, confirmado con la huella exacta del
formato de cabecera en los `cosmogenesis_ic.txt` de `bateria_n2000/`): masa REAL de átomos de H
(`_extraer_bariones`, motor basal ya validado) → malla causal REAL (`malla_causal_atomos`, grafo de
vecindad de las dos fases de expansión) → layout de resortes Fruchterman-Reingold
(`layout_resortes`) que despliega esa malla en posiciones 3D → dilatación isótropa estática → escritura
del IC. **La única diferencia entre REAL y cada NULL-i es que, antes del layout, las ARISTAS de la malla
se barajan por double-edge-swap** (`barajar_aristas`, preserva el grado exacto de cada nodo, destruye la
topología específica). Confirmado con los propios `seed_null=5000,5002,...,5014` usados en la batería
N=2000 (documentados en `RESULTADO_bateria_ignicion_sumideros_N2000_CS.md`).

**Pregunta del encargo: ¿ese barajado de aristas conserva la distribución radial/perfil de densidad de la
nube, o destruye más que eso?**

Se leyeron directamente las posiciones ya escritas en
`bateria_n2000/ic_real/cosmogenesis_ic.txt` y `ic_null1..8/cosmogenesis_ic.txt` (las condiciones
iniciales REALES que Phantom recibió, antes de que la gravedad actúe — ningún cómputo nuevo, sólo
lectura) y se comparó la distancia de cada partícula al centro de masa (r = |pos - COM|):

| corrida | r_mean | r_std | r_max | KS(REAL, NULL) stat | KS p-valor |
|---|---|---|---|---|---|
| REAL | 72.78 | 8.20 | 84.78 | — | — |
| NULL 1 | 63.51 | 13.58 | 84.66 | 0.359 | 1.5e-114 |
| NULL 2 | 63.26 | 13.67 | 84.91 | 0.370 | 3.5e-122 |
| NULL 3 | 63.19 | 13.63 | 84.95 | 0.378 | 1.2e-127 |
| NULL 4 | 63.18 | 13.60 | 84.98 | 0.384 | 8.1e-132 |
| NULL 5 | 63.25 | 13.34 | 84.89 | 0.388 | 1.2e-134 |
| NULL 6 | 63.49 | 13.26 | 84.88 | 0.379 | 5.5e-128 |
| NULL 7 | 63.21 | 13.65 | 85.16 | 0.391 | 8.7e-137 |
| NULL 8 | 63.41 | 13.56 | 84.77 | 0.368 | 7.8e-121 |

**Respuesta: los NULL1-8 existentes destruyen MÁS que la correspondencia relacional.** No son el
equivalente de NULL-1. REAL forma una nube sistemáticamente más extendida y más ESTRECHA en su
distribución radial (r_mean alto, r_std bajo — más parecida a una cáscara) que los 8 NULL, que son más
concentradas hacia el centro pero con MÁS dispersión (r_mean bajo, r_std ~65% más alto). Las 8
comparaciones KS son estadísticamente inequívocas (p < 1e-113 en todos los casos, N=2000 en cada
muestra). La razón física es clara una vez que se mira el mecanismo: `layout_resortes` no es una
reasignación de etiquetas sobre un punto fijo — es una relajación física completa (Fruchterman-Reingold)
que vuelve a calcular el equilibrio espacial de TODAS las partículas bajo la topología dada. Preservar el
grado (double-edge-swap) no preserva la forma de equilibrio que esa topología produce bajo repulsión
universal + atracción sólo entre vecinos causales — cambiar la topología, aunque sea isodegree, cambia el
perfil radial resultante. Esto no es un defecto de los NULL1-8 para lo que fueron diseñados a probar
(coherencia relacional en general) — pero sí significa que **no sirven** como el control aislado que pide
la jerarquía de 6 (que necesita tocar SOLO la correspondencia relacional, dejando el perfil radial
intacto). Por lo tanto hizo falta construir NULL-1 de cero (Paso 2).

---

## Paso 2 — Construcción de NULL-1 (`null1_generar_ic.py`)

**Conserva -- exacto, no sólo en distribución:** se generó la condición REAL con la pieza congelada
`fase1_traducir_a_phantom.traducir_pool` (sin tocarla), se leyeron sus posiciones finales, se calculó
r_i = |pos_i − centro_de_masa| para cada partícula, y NULL-1 hereda ESE MISMO multiconjunto de radios —
mismo histograma de distancia al centro, mismo perfil ρ(r), por construcción (no por muestreo
estadístico: literalmente el mismo conjunto de números).

**Destruye:** la dirección angular de cada partícula se reasigna a un vector aleatorio isótropo
(muestreo uniforme sobre la esfera, método de Marsaglia), independiente de cualquier vecino, de la malla
causal o del layout de resortes. No hay grafo ni Fruchterman-Reingold en NULL-1 — es una permutación pura
del ángulo, a radio fijo.

**Por qué no "reasignar partículas a posiciones ya existentes":** con la convención de
`fase1_traducir_a_phantom.py`, la masa es idéntica para todas las partículas y la velocidad se calcula
puramente a partir de la posición final (`campo_velocidad_turbulento` interpola un campo en `pos`, no
depende de "qué partícula" ocupa ese punto). Reasignar identidades manteniendo las posiciones habría
producido un archivo de condición inicial BIT-IDÉNTICO al de REAL — ningún observable habría cambiado.
El único grado de libertad real, no trivial, es la posición misma; por eso NULL-1 actúa sobre el ángulo.

Script: `null1_generar_ic.py` (funciones `leer_ic_txt`, `radios_desde_real`, `generar_null1`) — importa
`traducir_pool`/`POLYK`/`HFACT` de la pieza congelada, no la reimplementa. Orquestador del piloto:
`null1_piloto_generar.py`.

---

## Paso 3 — Piloto chico

**Escala:** se probó primero N=300 (nq=1800,naq=1260,ne=600,npos=420 en `_extraer_bariones`) — corre
limpio a tmax=0.5 pero SIN formar sumideros (densidad máxima final 3.05 g/cm³, muy por debajo del umbral
rho_crit_cgs=1000). Al extender tmax a 3.0 para darle tiempo a colapsar, Phantom aborta con "Large error
in linear momentum conservation" en t=0.885 (densidad~215, aún bajo el umbral) — el guardián de
conservación del propio Phantom se dispara antes de que se forme ningún sumidero. No se usó
`I_WILL_NOT_PUBLISH_CRAP` para forzarlo (regla de la casa). Lectura: a N=300 el ruido de dos-cuerpos
entre partículas discretas (más masa por partícula que a N=2000) es numéricamente demasiado agresivo para
este pipeline. **Se subió a N=500** (el techo del rango 250-500 pedido) — ahí el pipeline SÍ corre limpio
y SÍ forma sumideros dentro de tmax=0.5 (ver abajo).

**Configuración final del piloto** (idéntica a la de `bateria_n2000`, sólo N distinto): N=500 átomos de H
(`_extraer_bariones(3000,2100,1000,700,150,1.5)`), 1 corrida REAL (`seed_layout=12345`) + 3 corridas
NULL-1 (semillas angulares 101, 102, 103), mismo campo de velocidad turbulento (Mach=3, semilla=42,
idéntico en las 4), Phantom con sumidero pragmático (`icreate_sinks=1, rho_crit_cgs=1000, h_acc=0.3,
r_crit=0.6`, mismos valores que la batería N=2000), `tmax=0.5`, binario `phantom_cosmogenesis_backup`
(la build previa a la incorporación de APR — la build actual `phantom`/`phantomsetup` añade refinamiento
adaptativo de partículas por defecto, que habría sido un confound frente a la metodología ya validada de
`bateria_n2000`; se usó el backup para igualar exactamente el método).

**Resultado:**

| corrida | exit | wall time | nptmass final | masa total en sumideros | n_accretadas | densidad máx. final |
|---|---|---|---|---|---|---|
| REAL | 0 | 4.09 s | 4 | **282.0** (56.4+65.8+75.2+84.6) | 30/500 | 1.88e2 g/cm³ |
| NULL-1 seed 101 | 0 | 2.31 s | 0 | 0 | 0/500 | 2.81e-2 g/cm³ |
| NULL-1 seed 102 | 0 | 2.84 s | 0 | 0 | 0/500 | 3.36e-2 g/cm³ |
| NULL-1 seed 103 | 0 | 2.94 s | 0 | 0 | 0/500 | 4.23e-2 g/cm³ |

Las 4 corridas terminaron completas a tmax=0.5 sin ningún error de conservación (no se necesitó
`I_WILL_NOT_PUBLISH_CRAP`).

**Objetivo (a) pipeline sin errores:** SÍ, en las 4 corridas (N=500, tmax=0.5). A N=300 el pipeline
se vuelve numéricamente frágil (ver arriba) — dato relevante para elegir escala en el futuro.

**Objetivo (b) formación de sumideros medible:** REAL forma 4 sumideros (masa total 282.0). Las 3
semillas de NULL-1 NO forman ningún sumidero — la densidad máxima alcanzada (0.028-0.042 g/cm³) queda
~4 órdenes de magnitud por debajo del umbral de creación (1000 g/cm³) y ~4 órdenes de magnitud por debajo
del máximo que alcanzó REAL durante su evolución. Es una diferencia cualitativa nítida entre las dos
condiciones a esta escala reducida (no hay z-score que calcular: NULL-1 dio cero en las 3 semillas
corridas). Se deja constancia de que esto es un piloto de N=500 con 3 semillas, no el diseño
estadístico completo (N=2000, ≥8 semillas) — no se declara conclusión sobre CS073 ni sobre la
jerarquía de 6 a partir de este número.

**Objetivo (c) estimación de tiempo de cómputo:** REAL a N=500 tardó 4.09s (vs. 31.45s que tardó REAL a
N=2000 en la batería original, mismo tmax=0.5 — factor ~7.7× por un aumento de N de 4×, escalamiento
moderadamente superlineal, consistente con más partículas acretadas/formación de sumideros a mayor N).
NULL-1 a N=500 fue más barato aún (2.3-2.9s) al no colapsar. Extrapolando de forma conservadora (usando
el costo de REAL, más caro, como cota superior también para NULL-1 a N=2000): una batería completa de
8 semillas NULL-1 a N=2000 costaría del orden de **8 × 30-40s ≈ 4-6 minutos de cómputo de Phantom**.
A esto se suma la extracción del pool de bariones -- pero para la batería completa **no hace falta
repetirla**: `bateria_n2000/masa_bar.npy` y `dens_bar.npy` ya están en disco (extracción de ~58 min ya
hecha, ~2 ago), y `ic_real/cosmogenesis_ic.txt` ya tiene las posiciones REALES de las que NULL-1 necesita
heredar los radios -- ambos de sólo lectura, reutilizables directamente por `null1_generar_ic.py` sin
volver a correr el motor basal. Es decir: la batería completa NULL-1 (N=2000 × 8 semillas), si Alexis la
autoriza, es una corrida de minutos, no de horas.

---

## Entregables de esta tarea

- `null1_generar_ic.py` -- generador de la condición NULL-1 (radio heredado exacto de REAL, ángulo
  aleatorio isótropo). No toca ninguna pieza congelada, sólo importa `traducir_pool`/`POLYK`/`HFACT`.
- `null1_piloto_generar.py` -- orquestador que extrae el pool a N=500 y escribe las 4 condiciones
  iniciales del piloto (1 REAL + 3 NULL-1) en `/Users/alexis/phantom_cs073/piloto_null1/` (carpeta nueva,
  `bateria_n2000/` no se tocó).
- `/Users/alexis/phantom_cs073/piloto_null1/{real,null1_s1,null1_s2,null1_s3}/` -- las 4 corridas de
  Phantom del piloto (IC, `cosmog.in`, `run.log`, `setup.log`, dumps, `.sink`).
- Este informe.
