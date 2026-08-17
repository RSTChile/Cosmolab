# INSTRUCCIÓN PARA CC — Conectar el genoma (Kleiber + salud del cierre) a main.py
## CS, 11-jul-2026. Específica, copiable, anclada a las firmas reales del código.

**Objetivo:** que CosmoRobot deje de correr solo como un bucle de organelos sueltos y **transcriba su genoma**:
que cada ciclo mida su complejidad M, aplique las leyes de Kleiber (tempo s(M), eficiencia r(M)), y calcule su
"salud del cierre" (Λ_Cos, OI, invariantes κ). Hoy todo eso está escrito en `genoma/VST_Genoma.py` pero
**`main.py` nunca lo instancia** (verificado: no hay `Organismo(`, ni `vivir_un_paso`, ni `salud()` en el robot
vivo — la única referencia es un comentario en `organo_propiocepcion.py:17`).

Esta es la conexión **mínima y honesta**: NO reescribe los organelos de sensor/motor (esos siguen haciendo I/O
como están). Añade el organismo-Kleiber EN PARALELO, alimentado con las señales REALES que el robot ya computa.

---

## LA TRAMPA ANTI-SHANNON (léela antes de codear)

El genoma trae un `OrganeloMarcapasos` (VST_Genoma.py:654) que es un **stub de demostración**: inyecta valores
FALSOS fijos (`e_R=8.0`, `A_sys_env=0.4`, `delta_struct=0.30`...) para que la demo del genoma tenga de qué
alimentarse. **NO lo expreses en el robot.** Si lo expresas, el organismo "vivirá" de números inventados a mano
— exactamente el "asignar a mano qué es qué" que queremos evitar. El robot debe alimentar el milieu con sus
señales REALES (el error de distancia real, el CambioTotal real, el costo real). Esa es toda la diferencia entre
conectar el genoma honestamente y falsearlo.

---

## PASO 1 — Construir el organismo (una vez, en la inicialización de `main()`)

Junto a donde ya se crean los organelos (después de `propiocepcion = OrganoPropiocepcion()`, ~línea 83), añade:

```python
from genoma.VST_Genoma import Organismo, OrganeloPresionDesacople, OrganeloFatiga

# El organismo-Kleiber corre EN PARALELO al bucle de organelos de hardware.
# Se expresan SOLO los organelos abstractos cuyas entradas el robot produce de
# verdad (presión de desacople y fatiga). NO se expresa el marcapasos (stub que
# inyecta señales falsas — ver INSTRUCCION_*.md, sección anti-Shannon).
organismo = Organismo(nombre="cosmorobot", M0=1.0)
organismo.expresar(OrganeloPresionDesacople())   # lee e_R, A_sys_env → secreta presion_desacople
organismo.expresar(OrganeloFatiga())             # lee delta_real, costo_trabajo, en_reposo → historia, fatiga
```

Contrato verificado de cada organelo (VST_Genoma.py):
- `OrganeloPresionDesacople` (L549): `lee=["e_R","A_sys_env"]`, `secreta=["presion_desacople","presion_inst"]`,
  `costo_base=1.0`, τ Kleiber real (`tau*tempo`).
- `OrganeloFatiga` (L600): `lee=["delta_real","costo_trabajo","en_reposo"]`,
  `secreta=["historia","fatiga_activa","factor_gain"]`, `costo_base=1.0`, estado PARCIAL (residuo V150 documentado).

M arrancará en `max(M0, Σcosto_base) = max(1.0, 2.0) = 2.0` → `tempo s = (2.0/1.0)^0.25 ≈ 1.19`,
`eficiencia r = (2.0)^-0.25 ≈ 0.84`. Es decir: con 2 organelos ya se nota Kleiber (vive ~19% más lento). Cuando se
expresen más organelos (altruismo, marcapasos real, futuros), M sube y el tempo se estira solo.

---

## PASO 2 — Alimentar el milieu con señales REALES y vivir un paso (cada ciclo)

En el bucle de deliberación de `main.py`, **después** del bloque que calcula `error`, `costo` y llama
`mente.aprender(...)` (~línea 213, justo antes de `fila["e_R"] = ...`), inserta:

```python
# --- Transcribir el genoma: alimentar el milieu con lo REAL y vivir un paso ---
dt_real = fila["t"] - t_prev            # dt de reloj de pared del ciclo (ver PASO 3)
mil = organismo.milieu
# Señales REALES del robot (nada inventado):
mil.secretar("e_R", fila.get("error_post", 0.0))          # error real de distancia-objetivo
mil.secretar("delta_real", ct)                            # CambioTotal real (Δ experimentada)
mil.secretar("delta_struct", ct)                          # misma Δ, para κ_Δ y Λ_Cos
mil.secretar("costo_trabajo", costo)                      # costo real = duracion*potencia/100
mil.secretar("en_reposo", fila.get("veto_reactivo", 0) == 1)  # en veto no hubo trabajo motor
# A_sys_env y LF: SIN fuente real limpia todavía — NO inventar (ver nota honesta abajo).
parte = organismo.vivir_un_paso(dt_real)   # corre Kleiber + ciclo percibir/metabolizar/secretar
sal = organismo.salud()                    # Λ_Cos, OI, invariantes κ
# Loguear lo que el genoma ahora SÍ produce:
fila["M"]            = round(parte["M"], 3)
fila["tempo_s"]      = round(parte["tempo_s"], 3)
fila["eficiencia_r"] = round(parte["eficiencia_r"], 3)
fila["gasto_metab"]  = round(parte["gasto"], 3)
fila["Lambda_Cos"]   = round(sal["Lambda_Cos"], 4)
fila["OI"]           = round(sal["OI"], 4)
fila["nivel_OI"]     = sal["nivel_OI"]
fila["fatiga_activa"]= round(mil.leer("fatiga_activa", 0.0), 3)
fila["historia_bio"] = round(mil.leer("historia", 0.0), 3)
```

Firmas verificadas: `vivir_un_paso(dt)` (VST_Genoma.py:442) devuelve `{t, M, tempo_s, eficiencia_r, gasto}`;
`salud()` (L461) devuelve `{Lambda_Cos, OI, nivel_OI, invariantes}`.

---

## PASO 3 — dt real (necesario para que la fatiga y la τ signifiquen algo)

El genoma usa `dt` como tiempo transcurrido. Hoy `main.py` no lo pasa. Añade un `t_prev` que se actualice al
final de cada ciclo:

```python
# junto a las otras inicializaciones (~línea 105, con ultima_decision, ciclo...):
t_prev = 0.0
# ...
# al FINAL del cuerpo del ciclo (después de log.escribir(fila), ~línea 226):
t_prev = fila["t"]
```

---

## PASO 4 — Registrador: declarar las columnas nuevas

`datalog/registrador.py` escribe las claves de `fila`. Verifica que acepte columnas nuevas dinámicamente; si tiene
una cabecera fija, añade: `M, tempo_s, eficiencia_r, gasto_metab, Lambda_Cos, OI, nivel_OI, fatiga_activa,
historia_bio`. (Si escribe con `csv.DictWriter`, usa `extrasaction="ignore"` y una cabecera unión — no lo dejes
petar si falta una clave en un ciclo de veto.)

---

## NOTA HONESTA (LF) — qué queda en 0 a propósito, y por qué está BIEN

`A_sys_env` (acoplamiento sistema-entorno) y `LF` (libertad funcional) **no tienen una fuente real limpia
todavía** en el robot. NO los inventes. Consecuencias, todas correctas:
- `presion_desacople` interno usará `A_sys_env` default (1.0 en su `percibir`) → presión ≈ 0. Es honesto: sin
  medida de desacople, no se afirma tensión.
- `salud()` dará **OI bajo → "protoorganismo" o "no organismal"**. Esto es CORRECTO y el propio genoma lo dice
  (docstring de `salud`: "una célula madre mínima aún NO es organismo pleno"). Un OI alto aquí sería la señal de
  alarma de que algo se está inflando a mano.
- `Λ_Cos` saldrá chico (numerador Δ·LF con LF=0). También honesto.

**Lo valioso NO es que el OI salga alto** — es que ahora el organismo **mide y registra** su propia salud cada
ciclo con las leyes de Kleiber operando de verdad. Que salga bajo es el diagnóstico correcto del estado actual, y
da una barra REAL contra la cual medir progreso cuando se conecten H (homeostasis), A_sys_env, LF, etc.

Si más adelante quieren un proxy real de `A_sys_env`, pre-regístrenlo antes de correr (p.ej. "acople = fracción de
CambioTotal que sigue a una acción propia vs. ruido de fondo") — escribir la definición ANTES de ver el resultado,
regla anti-Shannon. No lo metan como un número a ojo.

---

## FASE 2 (NO ahora, requiere pre-registro): que Kleiber toque la CONDUCTA

Hoy el `tempo s(M)` solo estira las τ internas de los organelos abstractos — no cambia cómo se mueve el robot.
Para que Kleiber afecte la conducta (que el robot "viva más lento" al crecer en complejidad de verdad), habría que
usar `parte["tempo_s"]` para modular algo real: p.ej. el `dt_por_step` de la deliberación, o la cadencia del
bucle, o la duración de las acciones. **Eso cambia el comportamiento observable**, así que va aparte y con
predicción pre-registrada (qué esperamos que cambie y cómo lo mediríamos), no mezclado con esta conexión de
instrumentación. Esta instrucción (Pasos 1-4) es SOLO instrumentación: hace que el genoma corra y se registre, sin
alterar lo que el robot hace. Primero medir, después —con pre-registro— dejar que Kleiber muerda la conducta.

---

## SMOKE TEST antes de una tanda larga

Corre `main(max_ciclos=20)` y verifica en el datalog:
1. `M` constante = 2.0 (2 organelos expresados) y `tempo_s ≈ 1.19`, `eficiencia_r ≈ 0.84` — Kleiber operando.
2. `historia_bio` **monótona creciente** (nunca baja) — es el tiempo biológico irreversible; si baja, hay bug.
3. `fatiga_activa` sube en ciclos de trabajo, decae en vetos (`en_reposo=True`).
4. `OI` bajo y `nivel_OI` = "protoorganismo"/"no organismal" — honesto, no alarma.
5. `Lambda_Cos` finito, sin NaN/inf (la guarda anti-división-por-cero en e_R ya está en salud()).
Si los 5 pasan, la conexión es fiel. Reporta el smoke y esperamos veredicto antes de tanda larga (misma disciplina
que CS064/065/066).

---

## RESUMEN EN UNA FRASE

Construir un `Organismo` con los 2 organelos abstractos que consumen señales reales (NO el marcapasos-stub),
alimentar su milieu cada ciclo con e_R/CambioTotal/costo reales, correr `vivir_un_paso(dt_real)` y loguear
`M, tempo_s, eficiencia_r, Λ_Cos, OI`. Es instrumentación pura (no cambia la conducta); hace que el genoma-Kleiber
por fin corra y se mida, con OI honestamente bajo hasta que se conecten más señales. — CS 🐝
