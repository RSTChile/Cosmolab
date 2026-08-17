# NULL-3 — investigación preliminar (punto de partida, NO un resultado), Fase II CS073, escalón 3 de 6

**Estado: esto NO es NULL-3 completo.** Es el paso de investigación que Alexis autorizó adelantar si
sobraba margen de tiempo tras cerrar la batería NULL-2 (ver `NULL2_bateria_completa_CS.md` — 58s de
cómputo real de un presupuesto de 20 min). No hay condiciones iniciales generadas, no corrió Phantom,
no hay comparación de sumideros. Es un diagnóstico a nivel de GRAFO únicamente. No se declara ningún
veredicto — sólo se documenta el punto de partida y los números de la verificación.

---

## El encargo de NULL-3

Palabras de Alexis: un control que conserva "grado de cada nodo y longitudes [de enlace]" y destruye
"motivos, ciclos e historia", identificando el "efecto de la topología de orden superior" — el
escalón siguiente a NULL-1 (conserva el radio/perfil de densidad de cada partícula, destruye el
ángulo) y NULL-2 (conserva P(k)/2-puntos del campo, aproxima ξ(r) de partícula vía Zel'dovich).

## Por qué el double-edge-swap YA EXISTENTE no sirve tal cual

`p_semilla_causal.barajar_aristas` (double-edge-swap de Maslov-Sneppen, congelado, ya usaba
`bateria_n2000/ic_null1..8`) preserva la SECUENCIA DE GRADOS exactamente, pero no restringe la
LONGITUD de las aristas nuevas. Ya se había documentado (en `null1_generar_ic.py`, motivo por el
que se construyó NULL-1 aislado) que esos NULL1-8 originales, al re-correr `layout_resortes` sobre
el grafo barajado sin restricción, cambian el perfil radial COMPLETO de la nube (r_mean/r_std) frente
a REAL — es decir, destruyen más que "sólo" la topología: también destruyen la escala espacial. Ese
swap sin restringir NO es el control aislado que pide NULL-3.

## Hipótesis de trabajo

Si el double-edge-swap se restringe para aceptar sólo intercambios cuyas dos aristas NUEVAS tengan
una longitud geométrica (medida sobre las posiciones REAL ya existentes) parecida a las dos aristas
VIEJAS que reemplazan, el grafo resultante debería, al pasar por `layout_resortes`, producir una nube
con un perfil radial mucho más parecido a REAL que los NULL1-8 originales — porque la escala local de
conexión (qué tan lejos tiende a estar cada partícula de sus vecinas EN EL GRAFO) es lo que determina
la escala global de la relajación de resortes. Si esto se confirma, la comparación de sumideros
NULL-3 vs REAL aislaría el efecto de la topología de orden superior (motivos/ciclos/triángulos
específicos — "quién-con-quién" más allá de la escala local) de forma mucho más limpia que el swap
sin restringir.

## Qué se hizo en este paso (`null3_investigacion_preliminar.py`)

1. Se reconstruyó el grafo causal REAL EXACTO — determinista, mismos parámetros que
   `traducir_pool` usó para `ic_real` (`dens_bar.npy` ya guardado, `seed_ejes=2000`, `D=3`, `k=4`):
   n=2000, 4945 aristas, grado min=4/max=10/mean=4.945.
2. Se calculó la longitud de cada arista sobre las posiciones REALES ya escritas en disco
   (`ic_real/cosmogenesis_ic.txt`): L_real mean=24.43, std=16.22.
3. Se implementó `barajar_aristas_preservando_longitud` — misma mecánica de Maslov-Sneppen que
   `barajar_aristas` (elige 2 aristas al azar, las reconecta cruzado), pero acepta el intercambio
   SÓLO si ambas longitudes nuevas quedan dentro de una tolerancia relativa (`tol_relativa=0.2`, 20%)
   de la longitud que reemplazan. El grado queda preservado exacto por construcción (idéntico al
   original — nunca se toca la secuencia de grados, sólo la aceptación de qué intercambios entran).
4. Se corrieron, sobre el MISMO grafo REAL (n=2000, seed=501), las dos versiones lado a lado:
   swap SIN restricción (el mecanismo original) y swap CON restricción de longitud.

## Resultado de la verificación (sólo grafo, sin layout_resortes ni Phantom)

| | grado preservado | L mean | L std | diff mean vs REAL | KS(L, L_real) | p |
|---|---|---|---|---|---|---|
| REAL | — | 24.43 | 16.22 | — | — | — |
| swap SIN restricción (Maslov-Sneppen original) | exacto | 97.49 | 36.22 | **+299.0%** | 0.818 | ≈0 |
| swap CON restricción de longitud (tol=0.2) | exacto | 24.47 | 16.33 | **+0.2%** | **0.0040** | 1.00 |

El swap sin restringir (el que ya se sabía problemático) casi cuadruplica la longitud media de las
aristas — consistente con por qué distorsiona el perfil radial completo cuando se re-corre
`layout_resortes`. El swap CON restricción de longitud, en cambio, deja la distribución de longitudes
prácticamente INDISTINGUIBLE de REAL (KS=0.004, la distribución más parecida a "igual" que se ha
visto en todo este arco) — mientras SÍ cambió 618 de las 4945 aristas (12.5%) respecto del grafo
REAL, es decir, sí barajó topología real, no fue un no-op. Con `factor_swaps=10` (10×nº de aristas =
49450 intentos), la tasa de aceptación fue baja (346/49450, 0.7% — el filtro de longitud es estricto,
como se esperaba), pero suficiente para cambiar >1 de cada 8 aristas mientras preserva grado exacto y
longitud casi exacta. Tiempo total del diagnóstico: 3.8 s (n=2000, un solo seed).

## Lectura de este paso (sin cerrar nada)

Este resultado es alentador para la hipótesis de trabajo — el filtro de longitud SÍ logra lo que se
buscaba a nivel de grafo (grado exacto + longitud ≈idéntica + topología parcialmente barajada) — pero
**no prueba nada todavía sobre NULL-3 como control de Phantom**. Falta, en orden:

1. **Verificar el perfil radial post-`layout_resortes`**: correr `layout_resortes(adj_null3, n, ...)`
   sobre el grafo con restricción y comparar r_mean/r_std/KS de las POSICIONES resultantes contra
   REAL (mismo tipo de verificación que `null2_zeldovich_disenar_verificar.py` hizo para NULL-2, o
   que el piloto de NULL-1 hizo antes de escalar) — la hipótesis predice que debería acercarse mucho
   más a REAL que los NULL1-8 originales (que fallaron esa misma verificación con KS<1e-113), pero
   esto NO se ha probado — `layout_resortes` no se corrió en este paso.
2. Si el perfil radial se confirma razonable, recién ahí correr un piloto chico (N=500, 2-3 semillas)
   en Phantom, mismo patrón que los pilotos anteriores (NULL-1, NULL-2), antes de escalar a la
   batería completa de 8 semillas.
3. Elegir/justificar `tol_relativa` (aquí 0.2 fue un valor de partida razonable, no barrido ni
   optimizado) y `factor_swaps` (aquí 10, igual que el original — con tasa de aceptación tan baja
   0.7%, valdría la pena verificar si más intentos cambian más aristas sin romper la longitud, o si
   0.7% ya satura el espacio de swaps válidos a esta tolerancia).
4. Definir el observable de "motivos/ciclos" que NULL-3 debería estar destruyendo (ej. conteo de
   triángulos, longitud de ciclos) para poder reportar, además del resultado en Phantom, la magnitud
   del cambio topológico de orden superior en sí — actualmente sólo se reportó "% de aristas
   distintas" (12.5%), que es un proxy crudo, no una medida de motivos/ciclos.

## Archivos de este paso

- `null3_investigacion_preliminar.py` — reconstruye el grafo causal REAL exacto (sólo lectura de
  `dens_bar.npy` y `ic_real/cosmogenesis_ic.txt`), implementa y prueba
  `barajar_aristas_preservando_longitud`, compara contra el swap sin restringir (importado tal cual
  de `p_semilla_causal.py`, no reescrito). No genera condiciones iniciales de Phantom. No toca ningún
  archivo de `bateria_n2000/`, `p_semilla_causal.py`, ni `fase1_traducir_a_phantom.py` — sólo los
  importa/lee.
- Este informe.

No se tocó ninguna carpeta de batería anterior. No se corrió Phantom en este paso. No se declara
ningún veredicto — el siguiente escalón (verificación de perfil radial post-layout, luego piloto) es
trabajo pendiente explícito, no forzado dentro de esta tarea.
