# NULL-2 — mejora del método de conversión campo→partícula (desplazamiento de Zel'dovich)

**Encargo:** el informe anterior (`NULL2_piloto_espectro_potencia_CS.md`) documentó que el método de
NULL-2 (aleatorización de fases) preserva P(k) EXACTO a nivel de grilla, pero que el muestreo por
RECHAZO/INVERSIÓN usado para convertir el campo sintético en partículas destruye la estadística de
dos puntos a nivel de partícula (KS=0.495 frente a REAL, robusto a ngrid=16-40) — el diagnóstico fue
que a N=500-2000 el ruido de Poisson de resamplear partículas independientes domina sobre la señal
de 2 puntos que se quería trasladar. Alexis pidió mejorar el MÉTODO de conversión antes de escalar,
en vez de aceptar la limitación. Se implementó el desplazamiento de Zel'dovich, el método estándar de
generadores de condiciones iniciales cosmológicas (N-GenIC, 2LPTic), que no resamplea: desplaza
partículas ya existentes según un campo continuo. No se declara cierre ni veredicto sobre CS073 ni
sobre este escalón — sólo se reportan números. La lectura es de Alexis.

---

## Qué se implementó

`null2_zeldovich_generar_ic.py` (archivo nuevo, no toca ninguno de los anteriores):

1. **Reusa sin reescribir** `gridizar` y `aleatorizar_fases` de `null2_generar_ic.py` — la MISMA
   delta-hat de fase aleatorizada que ya usaba el método de rechazo.
2. **Campo de desplazamiento de Zel'dovich** (`campo_desplazamiento_zeldovich`): resuelve
   laplaciano(phi)=delta en Fourier y calcula Psi_hat_i(k) = i·k_i/|k|² · delta_hat(k) (modo k=0
   fijado a 0 — es sólo una traslación global de toda la nube, irrelevante para cualquier estadística
   relativa al centro de masa).
3. **Punto de partida NO perturbado** (`grilla_no_perturbada`): grilla regular jitterizada dentro de
   la misma caja que REAL, recortada a n partículas. Se eligió la opción más simple de las dos que
   sugería el encargo (en vez de "radios reales + ángulos uniformes") porque Zel'dovich asume que el
   punto de partida es HOMOGÉNEO — heredar la cáscara de REAL en el punto de partida habría
   contaminado la comparación (cualquier ξ(r) parecido a REAL sería un artefacto de esa herencia, no
   una prueba de que el desplazamiento por sí solo reproduce la estructura).
4. **Desplazamiento**: interpolación trilineal (`scipy.ndimage.map_coordinates`, `mode="wrap"` —
   el campo de Fourier es periódico por construcción, aproximación estándar de tipo N-GenIC aplicada
   aquí sobre una caja que en rigor no es periódica; se deja constancia) del campo continuo en la
   posición de cada partícula, x = q + Psi(q).

---

## Comparación ANTES / DESPUÉS del KS (mismo N=2000, mismo REAL, mismo ngrid=20, misma semilla de
## diseño 9001 que usó el informe anterior)

| método | KS stat | KS p | d_mean REAL | d_mean sintético | d_std REAL | d_std sintético |
|---|---|---|---|---|---|---|
| Rechazo/inversión (anterior) | **0.495** | ≈0 | 97.07 | 66.08 | 36.22 | 25.62 |
| Zel'dovich (este informe) | **0.220** | ≈0 | 97.07 | 86.75 | 36.22 | 36.59 |

**Mejora: KS bajó de 0.495 a 0.220 (−56%)** en las mismas condiciones. La distancia media par-a-par
del catálogo sintético pasó de subestimar REAL en −32% (rechazo) a solo −11% (Zel'dovich); la
desviación estándar de las distancias, que el rechazo subestimaba en −29%, ahora prácticamente
COINCIDE con REAL (36.59 vs 36.22, +1%).

**Robustez del resultado** (barrido de ngrid=16,20,26,32,40 con seed=9001, y barrido de 5 semillas de
fase con ngrid=20):

| ngrid | KS | d_mean sint. | d_std sint. |
|---|---|---|---|
| 16 | 0.233 | 84.70 | 35.47 |
| 20 | 0.220 | 86.75 | 36.59 |
| 26 | 0.377 | 73.04 | 30.65 |
| 32 | 0.190 | 88.88 | 35.40 |
| 40 | 0.200 | 88.32 | 36.04 |

| semilla | KS | d_mean sint. | d_std sint. |
|---|---|---|---|
| 9001 | 0.220 | 86.75 | 36.59 |
| 9002 | 0.269 | 82.70 | 36.81 |
| 9003 | 0.393 | 73.67 | 30.04 |
| 9004 | 0.187 | 89.22 | 36.78 |
| 9005 | 0.257 | 82.43 | 36.37 |

En todos los casos (10 combinaciones ngrid/semilla) KS quedó en el rango 0.19-0.39, sistemáticamente
por debajo del 0.495 del método de rechazo — mejora consistente, no accidente de una sola semilla.
**No obstante, sigue siendo estadísticamente distinto de REAL en todos los casos (p≈0)**: el método
mejora sustancialmente pero no reproduce perfectamente la estadística de dos puntos a nivel de
partícula a este N. Se reporta con la misma honestidad que el informe anterior.

---

## Piloto en Phantom (N=500, 3 semillas — mismo patrón que el piloto anterior)

Se reutilizó la condición REAL de N=500 ya en disco (`/Users/alexis/phantom_cs073/piloto_null1/real/`,
sin volver a correrla). Sobre esas mismas 500 posiciones REAL se aplicó Zel'dovich (grilla 14³,
misma resolución que usó el piloto de rechazo) con 3 semillas de fase (301, 302, 303), mismo campo de
velocidad turbulento (Mach=3, semilla=42) y misma configuración física de Phantom que toda la
jerarquía (`icreate_sinks=1, rho_crit_cgs=1000, h_acc=0.3, r_crit=0.6, tmax=0.5`, binario
`phantom_cosmogenesis_backup`).

Verificación de dos puntos ANTES de correr Phantom, a esta escala (N=500, ngrid=14):

| corrida | KS (vs REAL N=500) | r_mean | r_std | desplazamiento RMS |
|---|---|---|---|---|
| Zel'dovich seed 301 | 0.236 | 38.82 | 10.38 | 19.28 |
| Zel'dovich seed 302 | 0.094 | 44.13 | 11.40 | 19.80 |
| Zel'dovich seed 303 | 0.211 | 39.39 | 14.88 | 19.73 |
| (REAL, referencia) | — | 45.50 | 5.70 | — |

| corrida | exit | wall time | nptmass final | masa en sumideros | densidad máx. final |
|---|---|---|---|---|---|
| REAL (piloto NULL-1, ya en disco) | 0 | 4.09 s | 4 | **282.0** | 1.88e2 g/cm³ |
| Zel'dovich seed 301 | 0 | 1.93 s | 0 | 0 | 1.27e-1 g/cm³ |
| Zel'dovich seed 302 | 0 | 1.90 s | 0 | 0 | 8.99e-2 g/cm³ |
| Zel'dovich seed 303 | 0 | 1.88 s | 0 | 0 | 1.16e-1 g/cm³ |

Las 3 corridas terminaron completas a tmax=0.5 sin errores de conservación, sin NaN, y sin archivo
`.sink` (ningún sumidero creado en ninguna) — corrieron incluso más rápido que las del piloto de
rechazo (1.9 s vs 3.5-4.8 s, consistente con que Zel'dovich no requiere clipear/renormalizar densidad
negativa por celda del mismo modo).

**Lectura honesta:** aunque la verificación de dos puntos mejoró sustancialmente (KS 0.495→0.22 a
N=2000; 0.09-0.24 a N=500), **ninguna de las 3 semillas de Zel'dovich formó sumideros** — densidad
máxima 0.09-0.13 g/cm³, ~3 órdenes de magnitud por debajo del umbral (1000 g/cm³) y de la densidad
máxima de REAL (188 g/cm³), en el mismo orden que alcanzó el piloto de rechazo (0.13-0.21 g/cm³). La
mejora del método en la estadística de 2 puntos NO se tradujo, en este piloto chico de 3 semillas, en
formación de estructura colapsada. No se declara conclusión sobre CS073 ni sobre la jerarquía de 6 a
partir de este número — es un piloto de N=500 con 3 semillas.

---

## Entregables

- `null2_zeldovich_generar_ic.py` — método completo: `campo_desplazamiento_zeldovich`,
  `grilla_no_perturbada`, `interpolar_trilineal`, `generar_null2_zeldovich` (orquestador). Reusa
  `gridizar`/`aleatorizar_fases`/`verificar_dos_puntos_particulas` de `null2_generar_ic.py` sin
  reescribirlos.
- `null2_zeldovich_disenar_verificar.py` — verificación antes de Phantom, mismo formato/datos que
  `null2_disenar_verificar.py` para comparar KS antes/después en igualdad de condiciones.
- `null2_zeldovich_piloto_generar.py` / `null2_zeldovich_piloto_correr.py` — piloto Phantom N=500,
  3 semillas (301-303), mismo patrón que `null2_piloto_generar.py`/`null2_piloto_correr.py`.
- `/Users/alexis/phantom_cs073/piloto_null2_zeldovich/null2z_s{1,2,3}/` — carpeta NUEVA con las 3
  corridas de Phantom (IC, `cosmog.in`, `run.log`, `setup.log`, dumps). No se tocó
  `piloto_null2/` (método de rechazo, queda intacto como referencia) ni `piloto_null1/` (sólo
  lectura de `real/`).
- Este informe.

No se tocó `bateria_n2000/`, `bateria_null1_n2000/`, `bateria_real_extra_n2000/`, ni ningún script
congelado (`leer_volcado_phantom.py`, `null1_generar_ic.py`, `real_extra_generar_ic.py`,
`null2_generar_ic.py`, `null2_disenar_verificar.py`, `null2_piloto_generar.py`,
`null2_piloto_correr.py`) — sólo se importaron/leyeron.
