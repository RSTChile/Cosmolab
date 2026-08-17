# NULL-2 — piloto de aleatorización de fases (Fase II CS073, escalón 2 de 6)

**Encargo:** segundo escalón de la jerarquía de 6 controles (roadmap multi-IA, 5-ago-2026) para blindar
CS073 (z=48.69, REAL 2.95× la masa en sumideros del promedio NULL, batería N=2000 en
`/Users/alexis/phantom_cs073/bateria_n2000/`). NULL-1 (ver `NULL1_piloto_distribucion_radial_CS.md`) ya
mostró que conservar sólo la distribución radial (1 número por partícula) NO alcanza para formar
sumideros. NULL-2 pregunta algo más fino: ¿alcanza con que la nube conserve la ESTADÍSTICA DE DOS PUNTOS
completa (espectro de potencia P(k) / función de correlación ξ(r)), o hace falta la estructura de orden
SUPERIOR (3+ puntos) que sólo la malla causal genuina produce?

No se declara cierre ni veredicto sobre CS073 ni sobre este escalón — sólo se reportan números. La
lectura es de Alexis.

---

## Paso 1 — Método elegido y por qué

**Para el MECANISMO de NULL-2** (obligatorio según el encargo): aleatorización de fases sobre un campo
de densidad gridizado — se leyeron las posiciones de gas de la corrida REAL original N=2000
(`bateria_n2000/ic_real/cosmog_00000`, el primer volcado = condición inicial antes de que la gravedad
de Phantom actúe, leído con `leer_volcado_phantom.py`, sólo lectura), se gridizaron en un cubo de
20³ celdas (NGP vía `numpy.histogramdd`), se calculó su FFT completa (`numpy.fft.fftn`, no `rfftn` —
necesario para poder controlar la simetría hermítica explícitamente, ver abajo), y se reasignó la fase
de cada modo conservando el módulo `|F(k)|` exacto.

**Truco de implementación** (evita re-derivar a mano qué modos deben quedar fijos por la simetría
hermítica que exige un campo real): se generó un campo de ruido blanco real independiente del mismo
tamaño de grilla, se tomó SU FFT, y se usó la FASE de ese campo de ruido (que por construcción, al venir
de un campo real, ya respeta la simetría hermítica exacta) combinada con el MÓDULO del campo de REAL.
Es el mismo truco que usa el método de "phase randomization" para subrogados en series no lineales
(Theiler et al. 1992), aplicado en 3D. Residuo imaginario tras la transformada inversa: ~7×10⁻¹⁶ en
relación a la escala del campo (~26) — error de punto flotante, no señal.

**Para la VERIFICACIÓN de dos puntos** (Paso 2 del encargo: P(k) vs ξ(r), la que tenga menor riesgo de
artefacto de muestreo con N=500-2000): se usó **ξ(r) vía conteo de pares directo sobre las partículas**
(distribución de distancias par-a-par, comparada con test de Kolmogorov-Smirnov de 2 muestras), NO P(k)
de partícula discreta. Razón: estimar P(k) directamente de un catálogo de N=500-2000 puntos exige
volver a grillar esas partículas en una malla FFT — reintroduce el mismo ruido de Poisson (shot noise)
que ya limita el campo NULL-2 en sí (ocupación media 0.03-0.5 partículas/celda según la resolución
probada). ξ(r) por pares usa las posiciones EXACTAS de cada partícula, sin volver a grillar — el único
ruido que queda es el de tamaño de muestra, mucho menor a este N.

---

## Paso 1 (cont.) — Verificación ANTES de gastar cómputo en Phantom

Se corrió `null2_disenar_verificar.py` sobre la nube REAL N=2000 original (sólo lectura de
`bateria_n2000/`). Dos verificaciones separadas, con un resultado importante en cada una:

**(a) P(k) de la GRILLA (implementación) — coincide EXACTO, como debía.** Diferencia relativa máxima
entre P(k) de REAL y P(k) de NULL-2 calculados sobre la misma grilla: **3.8×10⁻¹⁶** (mediana 0.0) —
error de punto flotante puro. Esto confirma que la implementación de `aleatorizar_fases()` no tiene
bugs: `|F(k)|` se preserva modo a modo, exactamente, tal como exige el método.

*(Nota de proceso: la primera corrida de esta verificación dio P(k)_sintético = 0 en TODOS los bins —
un bug real, no un artefacto del método: el campo aleatorizado en fase puede terminar con media
NEGATIVA en la grilla completa (la fase del modo k=0, que en el campo original es real por construcción,
se reasigna a 0 o π al azar, lo que puede voltear el signo de la suma total sin alterar su magnitud) —
`pk_radial()` tenía una guardia `if media > 0` pensada para evitar dividir por grilla vacía, que
incorrectamente también anulaba el cálculo para cualquier media negativa válida. Corregido a
`if media != 0` en `null2_generar_ic.py` antes de seguir, según pedía el encargo.)*

**(b) ξ(r) por pares de PARTÍCULA (el catálogo final que Phantom vería) — NO coincide bien.** Al
muestrear N=2000 partículas del campo NULL-2 sintético y comparar su distribución de distancias
par-a-par contra las 2000 partículas REALES: KS stat=0.495, p≈0 (estadísticamente muy distintas).
Concretamente: distancia media entre pares REAL=97.07, NULL-2=66.08 (−32%); desviación estándar
REAL=36.22, NULL-2=25.62. Se repitió con resoluciones de grilla ngrid=16,20,26,32,40 — el efecto es
persistente en todo el rango (r_mean reconstruido siempre entre 47.7 y 51.1, frente a un r_mean REAL de
72.78 medido desde el centro de masa).

**Interpretación honesta (no un veredicto sobre CS073, sólo una lectura del método):** la nube REAL,
según ya documentó NULL-1, tiene forma de cáscara (r_mean alto, r_std bajo — masa concentrada en un
radio específico). Para una fuente esféricamente simétrica, la fase de su transformada de Fourier no es
un ángulo libre cualquiera: al ser el campo real y simétrico bajo r→−r, la fase de cada modo sólo puede
tomar los valores 0 o π (nunca un continuo). Reasignar la fase a un valor UNIFORME en [0,2π) — como pide
el método estándar de aleatorización de fases — descarta esa restricción binaria, y con ella se pierde
gran parte de la coherencia radial que sostenía la cáscara, incluso mientras `|F(k)|` (y por lo tanto
P(k)/ξ(r) EN LA GRILLA) se conserva exacto. A esto se suma que, con N=500-2000 partículas en una grilla
de FFT razonable, la ocupación media es de sólo 0.03-0.5 partículas/celda — el campo real de partida ya
está dominado por ruido de Poisson de muestreo, no por señal suave — y que hubo que clipear a 0 la
densidad negativa del campo sintético (fracción de masa negativa clipeada: 0.53-1.9× la masa positiva,
según la semilla), lo que sesga aún más la reconstrucción. En criollo: **a esta escala de N, "aleatorizar
la fase preservando el espectro de potencia" no reproduce fielmente ni siquiera el perfil radial grueso
de REAL** — el catálogo de partículas que efectivamente llega a Phantom se parece más, en r_mean/r_std, a
una nube MÁS compacta que la de NULL-1 (que a su vez ya era más compacta y dispersa que REAL), no a REAL
mismo. Esto no invalida el método (P(k) de la grilla SÍ se preserva exacto, que es lo que el método
promete) — invalida la expectativa de que, a este N tan chico, la reconstrucción partícula-por-partícula
vaya a "verse como REAL" fuera de su espectro de potencia grillado. Se deja constancia explícita de esta
limitación, no se disimula.

---

## Paso 2 — Piloto chico en Phantom

Se reutilizó la condición REAL de N=500 ya en disco (`/Users/alexis/phantom_cs073/piloto_null1/real/`,
generada para el piloto de NULL-1: 4 sumideros, masa total 282.0 — no se volvió a correr, ver
`NULL1_piloto_distribucion_radial_CS.md` para el detalle completo de esa corrida). Sobre esas mismas 500
posiciones REAL se aplicó el método NULL-2 (grilla 14³, ocupación media 0.18 part/celda) con 3 semillas
de fase (201, 202, 203), MISMO campo de velocidad turbulento (Mach=3, semilla=42) y misma configuración
física de Phantom que toda la jerarquía (`icreate_sinks=1, rho_crit_cgs=1000, h_acc=0.3, r_crit=0.6`,
`tmax=0.5`, binario `phantom_cosmogenesis_backup`).

| corrida | r_mean (referencia) | r_std | exit | wall time | nptmass final | masa en sumideros | densidad máx. final |
|---|---|---|---|---|---|---|---|
| REAL (piloto NULL-1, ya en disco) | 45.50 | 5.70 | 0 | 4.09 s | 4 | **282.0** | 1.88e2 g/cm³ |
| NULL-2 seed 201 | 30.78 | 10.01 | 0 | 4.77 s | 0 | 0 | 1.69e-1 g/cm³ |
| NULL-2 seed 202 | 27.18 | 9.91 | 0 | 4.13 s | 0 | 0 | 2.13e-1 g/cm³ |
| NULL-2 seed 203 | 31.78 | 9.15 | 0 | 3.52 s | 0 | 0 | 1.32e-1 g/cm³ |

Las 3 corridas NULL-2 terminaron completas a tmax=0.5 sin ningún error de conservación (no se usó
`I_WILL_NOT_PUBLISH_CRAP`), sin NaN, y sin archivo `.sink` (ningún sumidero creado en ninguna).

**Objetivo (a) partículas físicamente sensatas:** SÍ — sin NaN, densidades siempre positivas (post-clip),
mismo pipeline de escritura de IC que REAL/NULL-1.

**Objetivo (b) pipeline sin errores:** SÍ, las 3 corridas (N=500, tmax=0.5) terminaron limpias.

**Objetivo (c) lectura preliminar de formación de sumideros:** **NULL-2 NO formó ningún sumidero en las
3 semillas.** Densidad máxima alcanzada (0.13-0.21 g/cm³) queda ~3 órdenes de magnitud por debajo del
umbral (1000 g/cm³) y ~3 órdenes de magnitud por debajo del máximo de REAL (188 g/cm³) — algo más alta
que la densidad máxima que alcanzó NULL-1 en su piloto (0.028-0.042 g/cm³), pero igualmente lejos del
umbral de creación de sumideros. Consistente con la lectura del Paso 1: el catálogo de partículas NULL-2
efectivo, aunque preserva P(k)/ξ(r) EXACTO a nivel de grilla, terminó con un perfil radial más compacto
y disperso que REAL (r_mean −30 a −40%, r_std +60 a +75%) — similar en espíritu al de NULL-1 — y no
alcanzó a colapsar en 3/3 semillas. No se declara conclusión sobre CS073 ni sobre la jerarquía de 6 a
partir de este número — es un piloto de N=500 con 3 semillas.

**Objetivo (d) estimación de tiempo para escalar a N=2000×8 semillas:** la parte de Python (grillar +
aleatorizar fases + muestrear partículas) es prácticamente gratis (~0.02-0.05 s por semilla incluso a
N=2000, medido en el diseño del Paso 1) y no requiere re-extraer el pool de bariones (se reusa
`bateria_n2000/ic_real/cosmog_00000`, ya en disco). El costo real es Phantom: las 3 corridas NULL-2 de
este piloto (N=500, sin colapso) tardaron 3.5-4.8 s cada una — más caras que las de NULL-1 (2.3-2.9 s)
pero del mismo orden. Extrapolando con el mismo factor de escalamiento superlineal ya observado para
corridas sin colapso entre N=500 y N=2000 (~7-8×, ver NULL1_piloto_distribucion_radial_CS.md), una
batería completa de 8 semillas NULL-2 a N=2000 costaría del orden de **8 × 30-40 s ≈ 4-6 minutos de
cómputo de Phantom**, similar a la estimación ya hecha para NULL-1. Si alguna semilla a N=2000 sí
colapsara (no ocurrió en este piloto, pero no puede descartarse con sólo 3 semillas), el costo por esa
corrida subiría al rango de REAL (~30 s), sin cambiar el orden de magnitud del total.

---

## Entregables de esta tarea

- `null2_generar_ic.py` — módulo con el método completo: `gridizar` (partículas→campo NGP),
  `pk_radial` (diagnóstico), `aleatorizar_fases` (el mecanismo NULL-2), `muestrear_particulas_de_campo`
  (campo→partículas por transformada inversa + jitter), `verificar_dos_puntos_particulas` (ξ por pares +
  KS), `escribir_ic_txt`.
- `null2_disenar_verificar.py` — Paso 1: lee REAL N=2000 (sólo lectura de `bateria_n2000/`), corre las
  verificaciones (a) y (b) descritas arriba.
- `null2_piloto_generar.py` — Paso 2: genera las 3 condiciones NULL-2 de N=500 a partir de la REAL ya en
  disco de `piloto_null1/`.
- `null2_piloto_correr.py` — corre Phantom sobre las 3 condiciones (mismo patrón que
  `null1_bateria_correr.py`/`real_extra_correr.py`).
- `/Users/alexis/phantom_cs073/piloto_null2/null2_s{1,2,3}/` — las 3 corridas de Phantom del piloto (IC,
  `cosmog.in`, `run.log`, `setup.log`, dumps). No se creó carpeta `real/` nueva — se reusó la de
  `piloto_null1/real/` como referencia, sin tocarla.
- Este informe.

No se tocó `bateria_n2000/`, `bateria_null1_n2000/`, ni `bateria_real_extra_n2000/` (sólo lectura). No se
tocó ningún script congelado del proyecto.
