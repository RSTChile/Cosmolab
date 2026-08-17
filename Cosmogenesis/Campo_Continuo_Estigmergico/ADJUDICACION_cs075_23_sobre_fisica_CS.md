# ADJUDICACIÓN cs075 — CC resolvió una ambigüedad que yo dejé abierta y encontró un bug que yo no vi

**Adjudica:** Claude Science · **Implementó y ejecutó:** CC · **Director:** Alexis López Tapia
**Fecha:** 30-jul-2026 · **Estado:** el experimento **NO se cierra.** El barrido **NO se corre**
sin autorización explícita del director.

**Verificado en disco**, no de palabra: `Campo_Continuo_Estigmergico/cs075_23_sobre_fisica.py`
(31,4 KB), `cs075_resultado_pruebas_23_sobre_fisica.json`, `cs075_resultado_23_sobre_fisica.json`
(201 KB), `RESULTADO_cs075_23_sobre_fisica_PARA_CS.md`, más `MANIFIESTO_FOLD_CS072.md` l.3 y
l.22-23 y `cs072_motor_23.py` l.29 leídos directamente.

---

## 0. Antes de adjudicar: dos errores míos que este trabajo expone

**Primero.** Cuando el informe llegó, busqué los archivos en `Cosmogenesis/` y en todo
`/Users/alexis`, no los encontré, y dije que no podía adjudicar — pero dije explícitamente que
eso NO significaba que no existieran, sino que no estaban en las rutas a las que tengo acceso.
Estaban en `Cosmogenesis/Campo_Continuo_Estigmergico/`, una subcarpeta que no existía cuando
escribí la instrucción. Esa vez la distinción la hice bien. La vez anterior, con
`VST_Celula_Madre_001.py`, la hice mal: afirmé que no existía habiéndolo leído yo mismo.

**Segundo, y es el que importa para este experimento:** mi instrucción listaba `#24 tiempo`
como elemento del inventario canónico. **Es falso.** `#24` aparece **cero veces** en
`MANIFIESTO_FOLD_CS072.md` (verificado con `grep -c`). Yo lo tomé de `p24_tiempo.py`, que es un
módulo del proyecto, y lo convertí en elemento numerado del inventario sin verificarlo — el
mismo error que había cometido con `#18` (que puse como "espacio" cuando es poda/dilución). Dos
veces el mismo error, en el mismo documento.

---

## 1. La ambigüedad del inventario: CC la resolvió, y contra mi versión

Mi instrucción admitía que su desglose sumaba 22 y no 23, y le pedía a CC verificar contra el
manifiesto antes de implementar. Lo hizo, y su lectura es correcta — verificada línea por línea:

- `MANIFIESTO_FOLD_CS072.md` l.3: *"18 ELEMENTOS + 3 MECANISMOS + 2 FLUCTUACIONES CUÁNTICAS
  (QCD #22 + campo #23) = 23"*.
- l.22-23: *"NOTA DE CUENTA — INVENTARIO CERRADO EN 23 (fijado por el director, 18-jul): 21 =
  18 ELEMENTOS DEL ARCO + 3 MECANISMOS DE ORIGEN"*.
- `cs072_motor_23.py` l.29: *"M3 fase cuántica (amplitud/fase; FUERA salvo acople sin grilla;
  ausencia declarada)"*.

**El elemento que faltaba es M3, no #24.** Los mecanismos de origen son tres (M1, M2, M3), no
dos como yo asumí. M3 cuenta en el inventario pero su efecto está arquitectónicamente ausente
en cualquier motor sin representación de fase cuántica — y éste trabaja con densidad y
temperatura, no con amplitud y fase.

**Adjudico la resolución como correcta.** Y CC hizo bien una distinción fina que yo no había
previsto: M3 deposita cero exacto **por ausencia arquitectónica declarada**, que no es lo mismo
que una casilla de falsación. Las casillas SÍ pueden actuar y su nulo es un resultado físico
medido; M3 no puede actuar porque el motor no tiene el sustrato. Son dos tipos distintos de
cero y conviene no confundirlos.

---

## 2. El bug que CC encontró y yo no: divergencia tipo Riccati

Esto es lo más valioso del trabajo, y no era lo que se le pidió buscar.

**Lo que reporta:** con los parámetros default, la corrida no hubiera terminado nunca — no por
lentitud, sino por divergencia numérica genuina. Colapsaba en NaN a los ~50.000 pasos (0,24 %
del camino a confinamiento), después de 48.000 pasos sin nada anómalo, explotando a 1e+289 en
2.000 pasos más.

**La causa:** `22_qcd` tenía un término `(ρ/media − 1)²` sin acotar. Ese es el diagnóstico
correcto: un término cuadrático en su propio exceso es una ecuación de Riccati, y las Riccati
divergen **en tiempo finito**. Explica exactamente el perfil observado — quieto mucho tiempo y
después vertical. No es un artefacto de paso de integración: es la ecuación.

**Lo que hizo bien, más allá del arreglo:** auditó el resto del archivo buscando la misma clase
de defecto y encontró tres más (`2_gravedad`, `17_oscuro`, `23_campo`), ninguno causante de este
colapso concreto pero todos con la misma falla latente. Eso es lo correcto: arreglar la clase
de error, no la instancia.

**Y no inventó técnica nueva.** Verificado en disco: el patrón `np.tanh((ρ − media)/std)`
aparece en las líneas 191, 220, 294, 408 y 491 de `cs075_23_sobre_fisica.py`, y los comentarios
del código lo declaran como *"el mismo patrón tanh que #2/#5/#22"*. Es la regla del proyecto
(`p_expansion.py`: "ninguna constante nueva") aplicada correctamente — y es justo la regla que
yo violé dos veces en el diseño de la base física.

---

## 3. El número empeoró, y eso es a favor de CC

**Antes:** ~20 h por configuración. **Ahora:** ~35,4 h. CC entregó un número que empeora su
propio reporte previo, con la razón correcta: el número anterior describía una corrida que iba a
fallar sin importar cuánto tiempo se le diera.

Verifiqué la aritmética de forma independiente, sin copiarla:

- 6,05 ms/paso × 35,4 h ⟹ **21.064.463 pasos**, es decir t ≈ 21.064 en unidades del modelo.
- 4 configuraciones en serie: 4 × 35,4 = **141,6 h**. CC reporta 141,5 h. Coincide.
- 50.000 pasos como fracción de 21,06 M: **0,24 %**. CC reporta 0,24 %. Coincide.

Los tres puntos cierran. **Adjudico la tasa y la estimación como verificadas.**

---

## 4. Las pruebas: 5 de 6 pasan, y la que falla es la interesante

| prueba | resultado |
|---|---|
| E1 — inventario completo (23, sin repetidos) | **PASA** |
| E2 — nadie madruga (**la tesis del director**) | **PASA** — cero violaciones |
| E3 — cero exacto dormido | **PASA** |
| E4 — cero turnos | **PASA** — diferencia relativa 0,0 |
| E5 — las 6 direcciones se mantienen | **FALLA** — X sube |
| E6 — orden entre niveles | **PASA** — Nivel 0→1→3 sin inversiones |

**E2 y E3 pasan, y eran las dos que yo marqué como no negociables.** La tesis del director —
que ningún agente actúa antes de que existan las condiciones que su elemento necesita— se
sostiene ahora sobre parámetros físicos reales, no sobre el campo topológico de mi primera
versión. Cero violaciones.

### La falla E5, y por qué la trato como hallazgo y no como bug

`T`, `ρ` y `S` mantienen su dirección. **La exergía X sube** (0,0099 → 0,0567) en vez de bajar.
CC **no la forzó a bajar tocando fórmulas**, que era la salida fácil y equivocada.

Su lectura: los agentes de estructura local (`2_gravedad`, `3_fuerte`) concentran densidad en
regiones sobredensas — **generan diferencia localmente**, compitiendo contra la homogeneización
de la difusión.

**Verifiqué el control que invoca, y existe:** en `cs075_resultado_base_fisica.json` la base sin
agentes va de X = 0,005528 a X = 0,000208 — **baja**. Con los 23 agentes, sube. La diferencia
entre las dos corridas son los agentes, y la explicación de CC es la única candidata sobre la
mesa.

**No lo adjudico como resultado físico.** Es plausible y el control apunta en su dirección, pero
"plausible con un control a favor" no es lo mismo que medido: haría falta apagar `2_gravedad` y
`3_fuerte` y ver si X vuelve a bajar. Eso es una corrida corta, no un barrido. **Lo adjudico
como hallazgo abierto con su prueba pendiente identificada.**

---

## 5. El hallazgo lateral, que puede ser el más importante

Con el bug corregido, las configuraciones de baja asimetría (0,01 y 0,1) **sí alcanzan
sobredensidad**, cuando antes nunca llegaban en la ventana de prueba.

**El bug estaba ocultando estructura real.** Eso no cambia solo la duración del barrido: cambia
qué esperar de él. Si la estructura aparece también a baja asimetría, el rango a barrer no es el
que yo había supuesto.

---

## 6. Lo que recomiendo, sin ejecutar nada

1. **Antes del barrido, cerrar E5 con una corrida corta:** apagar `2_gravedad` y `3_fuerte`, ver
   si X vuelve a bajar. Cuesta minutos y convierte una explicación plausible en un resultado
   medido. Es el mismo principio del calibrador que recomendé para cs074D, y por la misma razón:
   35,4 h × 4 configuraciones son 141,6 h, y no conviene gastarlas con una dirección
   termodinámica sin explicar.
2. **Revisar los umbrales de temperatura del §3.2 de mi instrucción.** Son la parte de mi diseño
   con menos anclaje: los derivé de la razón 159 GeV / 155 MeV ≈ 1026, y esa traducción a
   temperaturas del modelo no la verificó nadie todavía. Dado que ya me equivoqué con #18 y con
   #24 en el mismo documento, este punto merece desconfianza.
3. **Sobre el barrido:** 141,6 h en serie, ~35,4 h en paralelo de 4. Con el hallazgo lateral de
   §5, el rango de asimetría a barrer probablemente cambie — conviene fijarlo después de cerrar
   E5, no antes.

---

*Nada se cierra aquí. Nada se corre sin autorización explícita del director. Todos los números
de este documento están impresos en las salidas que miré o recalculados por mí; ninguno fue
puesto a mano.*
