# CS075 — Informe completo v1→v4

**Ejecuta:** CC · **Para:** Claude Science · **Fecha:** 30-jul-2026 · **Estado:** nada
cerrado, sin autorización del director.

Arco completo: arquitectura, umbrales, barrido y los dos controles. Qué funcionó, qué NO
resultó, y qué queda abierto. (Versión visual del mismo contenido: artifact HTML
`cs075_informe_completo.html`, publicado por separado.)

---

## En una lectura

- La arquitectura de **23 agentes** sobre la física real (densidad/temperatura) quedó
  construida, verificada y sin bugs conocidos de divergencia.
- Con `amp_asimetria` ≥ 1,32 se activa el **inventario completo (23/23)**, hasta el
  nivel más alto (red conectada) — de forma consistente en 8 semillas.
- El borde de esa transición (entre 1,028 y 1,323) es **del modelo, no de la semilla ni
  de los umbrales declarados** — confirmado por dos controles independientes.
- Pero el control más duro (N2) mostró que **el observable usado no distingue una
  cascada causal genuina de un reparto arbitrario de precondiciones** — la tesis central
  queda sin evidencia, no refutada.
- Nada de esto se cerró. Sigue pendiente de decisión en varios puntos.

---

## 1. El arco, en orden

**v1 — arquitectura.** Los 23 agentes reimplementados sobre `EstadoFisico` (densidad/
temperatura real, no el campo Φ topológico anterior). Encontrado y corregido un bug de
divergencia numérica genuino: varias fórmulas de depósito tenían retroalimentación
positiva sin acotar; la más grave (`22_qcd`) era tipo Riccati y colapsaba el sistema en
NaN a los ~50.000 pasos.

**v2 — umbrales corregidos.** El anclaje original de `T_CONF`/`T_EW` (razón física 159
GeV/155 MeV) inventaba una escala donde el proyecto ya tenía la suya
(`cs072_motor_23.py` l.42-43). Corregido a los valores normalizados ya fijados en el
motor: confinamiento pasó de ~21 millones de pasos a 36. Además, encontrada y corregida
la causa real de que la exergía subiera en vez de bajar (E5).

**v3 — el barrido.** 160 configuraciones (20 amplitudes × 8 semillas). Encontrado un
escalón limpio entre incompleto (15–17/23) y completo (23/23), sin techo hasta 6,0, con
dispersión entre semillas prácticamente nula salvo en el punto exacto de transición.

**v4 — los dos controles.** N1 (sensibilidad de umbrales, 432 configs): el borde no se
mueve. N2 (puertas permutadas, 480 configs): 20 asignaciones arbitrarias de
precondición dieron el mismo resultado que la asignación real.

---

## 2. Qué funcionó

- **Batería E1–E6 completa en PASA.** Inventario cerrado en 23 (sin repetidos), nadie
  madruga (cero violaciones), cero exacto en dormidos, cero-turnos, las 4 direcciones
  termodinámicas correctas, orden esperado entre niveles.
- **El bug de divergencia, encontrado por medición, no por sospecha.** Diagnóstico
  agente-por-agente (covarianza del depósito con el exceso de densidad) localizó
  exactamente qué fórmulas rompían la conservación, en dos rondas de corrección.
- **Confinamiento: de ~35 horas de cómputo a 36 pasos.** Una vez anclados los umbrales
  correctos, cruzar la temperatura de confinamiento pasó de una corrida de casi un día y
  medio a una fracción de segundo.
- **Borde nítido y reproducible en el barrido.** Un solo punto de la grilla con
  resultado mixto; todo lo demás, 0/8 u 8/8 sin excepción. Desviación estándar entre
  semillas = 0 en 19 de 20 puntos.
- **N1: el borde resiste variar los umbrales declarados.** 9 combinaciones de
  `MIN_PERSISTENCIA`×`FACTOR_ATOMOS` (rango 3,3× y 4×) dan el mismo borde exacto.

---

## 3. Qué NO resultó

Con la misma claridad que lo positivo, tal como pide el protocolo del proyecto.

**N2 — la tesis del despertar ordenado queda sin evidencia.** Las 20 permutaciones de
qué agente espera qué hito (preservando el histograma exacto) dieron el MISMO
`n_despiertos` que la asignación real, en las tres amplitudes, sin excepción. Verificado
que no era un bug: el *override* sí cambia identidades reales de quién despierta cuándo.
Lo que no cambia es el conteo total. La cascada ordenada resulta **inevitable** con este
observable — no refuta que la asignación real tenga sentido causal, pero tampoco lo
sostiene. Hace falta buscar otro observable.

**E5 falló primero, y mi primera explicación también falló.** La exergía subía en vez de
bajar. Mi primera hipótesis (gravedad y fuerza fuerte generan estructura, compitiendo con
la difusión) fue **falsada directamente** por el control E5b: apagar esos dos agentes no
revertía la subida. La causa real era otra — dos agentes (`22_qcd`, `5_debil`) mal
diseñados como si reforzaran estructura cuando su propio docstring los describía como
fuentes uniformes.

**Sin control NULL de la física.** Excluido por decisión del director en esta etapa. Sin
él, "23/23" es un hecho de la arquitectura — cada agente encontró sus condiciones — no
una prueba de que lo formado sea materia en algún sentido físico fuerte.

**`hay_atomos` y `hay_red` nunca se anclaron en ningún archivo del proyecto.** Son
interpretaciones declaradas (persistencia + regiones conexas), no derivadas de una
fuente del proyecto — `EstadoFisico` no tiene noción de carga ni especies. Todo
resultado que involucra los niveles 5–6 hereda esa incertidumbre de diseño.

**Coincidencia sólo parcial con la banda de fragmentación de cs074A.** El borde cae
dentro del rango 0,9–2,3 que cs074A midió de forma independiente — pero sólo en el
extremo de entrada. El "colapso" que cs074A ubica por encima de 3,8 no tiene contraparte
acá: el inventario sigue completo, sin excepción, hasta 6,0.

**Malla fija en 16³ en todo el arco.** Ningún control probó si el borde depende de la
resolución de la grilla. Una malla mayor (24³ o 32³) es barata mecánicamente pero es otra
corrida y otra autorización.

**`14_correlacion` nunca se auditó individualmente.** Tiene el mismo patrón de riesgo
(fórmula sin saturar) que otros cuatro agentes que sí tuvieron que corregirse por
divergencia. No dio señales de falla en las corridas donde estuvo activo (v3, v4), pero
no se revisó su fórmula formalmente como se hizo con las otras.

---

## 4. El borde, en números

| `amp_asimetria` | n_despiertos | semillas en 23/23 | nivel máximo |
|---|---|---|---|
| 0,050 – 0,799 | 15,0 – 17,0 | 0/8 | 2 – 3 |
| **1,028** (borde) | 20,75 | 3/8 | 3 – 6 (mixto) |
| 1,323 – 6,000 | 23,0 | 8/8 | 6 (completo) |

### Los dos controles

| control | qué probó | resultado | veredicto |
|---|---|---|---|
| N1 | Sensibilidad a `MIN_PERSISTENCIA`/`FACTOR_ATOMOS` | 9/9 combinaciones, mismo borde | del modelo |
| N2 | 20 asignaciones agente↔hito permutadas | 20/20 idénticas al real | sin evidencia |

---

## 5. Qué queda abierto

1. Buscar un observable distinto de `n_despiertos` que pueda distinguir una asignación
   causal genuina de una arbitraria (candidatos no explorados: tiempo de residencia
   relativo, X/S finales bajo cada permutación, estructura espacial formada).
2. Correr un control NULL de la física, ahora que la arquitectura está completa y
   medida.
3. Probar independencia de malla (24³ o 32³) para ver si el borde se mueve con la
   resolución.
4. Localizar el borde con más precisión que "entre 1,028 y 1,323" (malla más fina), si
   vale la pena.
5. Auditar `14_correlacion` con el mismo método que encontró los otros cuatro bugs.

---

## Archivos del arco completo

Todos en `Cosmogenesis/Campo_Continuo_Estigmergico/`:

- `cs075_23_sobre_fisica.py` — el motor, 23 agentes
- `cs075_pruebas_23_sobre_fisica.py` + `cs075_resultado_pruebas_23_sobre_fisica.json` — batería E1–E6
- `cs075_smoke_23_sobre_fisica.py` + `cs075_resultado_23_sobre_fisica_v2.json` — smoke de 4 configuraciones
- `cs075_E5b_prueba_estructura_exergia.py` + json — control directo de exergía
- `cs075_diag_quien_sube_X.py` + json — diagnóstico agente-por-agente
- `cs075_barrido_v3.py` + `cs075_resultado_barrido_v3.json` — 160 configuraciones
- `cs075_N1_sensibilidad_umbrales.py` + json — 432 configuraciones
- `cs075_N2_puertas_permutadas.py` + json — 480 configuraciones
- `RESULTADO_cs075_23_sobre_fisica_PARA_CS.md` — el informe narrativo completo, 6 adendas

Nada de este arco está cerrado. Cada punto de "qué queda abierto" espera decisión
explícita del director.
