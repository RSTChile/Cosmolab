# INSTRUCCIÓN CS072 — ÚLTIMA OPORTUNIDAD PARA CC
## Implementar lo que el director pidió desde el principio: temperatura inicial desigual + expansión demasiado rápida para rehomogeneizar
## Inventario corregido por el director: 18 + 3 + fluctuaciones cuánticas QCD = 22 componentes obligatorios

**Autoridad:** Alexis, director del experimento.  
**Función de CS:** definir y custodiar el experimento.  
**Función de Codex:** revisar y bloquear desviaciones.  
**Función de CC:** implementar literalmente esta instrucción. CC no rediseña, no interpreta y no agrega mecanismos.

---

## 0. ACTA DE RESPONSABILIDAD — DEBE QUEDAR EN EL INFORME FINAL

Los fallos y rodeos ocurridos hasta ahora **NO constituyen un fracaso de la hipótesis del director**.

El director dijo desde el comienzo, en lenguaje simple:

> La explosión inicial estaba extremadamente caliente, pero no exactamente igual de caliente en todas partes. Había variaciones en el gradiente de temperatura. La expansión fue tan rápida que esas diferencias no alcanzaron a volver a homogeneizarse; quedaron preservadas y aumentaron.

El equipo no implementó fielmente esa premisa. En distintos momentos la reemplazó o desvió hacia semillas abstractas, masas rugosas, densidades artificiales, grafos aleatorios, números de orden, tasas y cupos elegidos por programación. También dejó fuera las fluctuaciones cuánticas QCD del inventario. Después presentó como “descubrimiento” un toy que apenas confirmó tardíamente el planteamiento original del director.

Por tanto:

1. Ninguna corrida anterior que haya omitido o sustituido la premisa térmica puede adjudicarse como falsación de la idea de Alexis.
2. La responsabilidad por esas desviaciones corresponde al equipo de diseño, revisión e implementación —CS, Codex y CC—, no al director.
3. Sólo una corrida que cumpla **todas** las guardas de esta instrucción podrá producir un resultado atribuible al experimento solicitado.
4. Si una guarda falla, el resultado se rotula **MOTOR INVÁLIDO / INSTRUCCIÓN NO CUMPLIDA**. No se rotula “la hipótesis falló”.

---

## 1. LA PREMISA CANÓNICA — NO VOLVER A TRADUCIRLA

La asimetría inicial de este experimento es una **variación física en la temperatura inicial**.

La cadena que debe simularse es:

> temperatura inicial no homogénea → las interacciones intentan rehomogeneizarla → la expansión ocurre antes de que puedan hacerlo → la diferencia térmica persiste y/o aumenta → las historias relacionales `W` divergen por física.

En analogía: hay una gota caliente y otra un poco menos caliente. Si se mantienen en contacto suficiente tiempo, igualan su temperatura. Si el medio se despliega demasiado rápido y corta el contacto antes, la diferencia queda congelada. **La expansión no pinta la diferencia: impide que se borre.**

Esto es condición inicial física declarada por el director. No es una “semilla” inventada por el programa y no debe sustituirse por otro nombre o proxy.

---

## 2. PROHIBICIONES ABSOLUTAS

CC no puede usar, ni directa ni indirectamente:

- `GR.aleatorio`, RNG, semillas computacionales, `shuffle`, `choice` o ruido para romper empates.
- El índice, número de fila, orden del array, `i % k`, ciclos por posición, primeros/últimos elementos o desempates automáticos del lenguaje como propiedad física.
- Van der Corput, Fibonacci, números primos u otra fórmula aplicada al número de orden para fabricar “rugosidad”.
- Masa rugosa, densidad artificial o identificador único por nodo como reemplazo de la temperatura.
- Coordenadas espaciales, regiones prefabricadas o una dimensión puesta antes de la dinámica.
- `argsort`/`argmax`/“top-k” cuando existen empates físicos, salvo que el resultado sea demostrado equivariante y el empate se trate simétricamente.
- Tasas, cupos, porcentajes, topes por paso o umbrales elegidos para obtener la curva deseada.
- Copiar como física los números del toy (`0.9`, `0.1`, `0.02`, `40 pasos`, amplitud `±0.1`). Son números de una prueba de posibilidad, no constantes de la naturaleza.
- Declarar una pieza “presente” en un comentario si no modifica realmente el estado común.
- Arreglar silenciosamente una ambigüedad. Si esta instrucción no determina una decisión, CC se detiene y pregunta a CS/director antes de escribirla.

**Regla simple:** si al borrar los nombres de las filas el motor ya no sabe qué hacer, estaba usando etiquetas y queda inválido.

---

## 3. ESTADO FÍSICO ÚNICO

El motor mantiene un solo estado vivo y común:

- propiedades físicas del catálogo ya admitidas;
- temperatura actual `T` de cada estado físico o entidad;
- red relacional viva `W`;
- memoria/historia de los enlaces;
- variables internas que las piezas activas necesiten.

La temperatura viaja con el estado físico cuando se reordena el catálogo. La posición de almacenamiento nunca entra en una ecuación.

Las entidades físicamente idénticas, con la misma temperatura y la misma historia relacional, deben seguir empatadas. **CC no tiene que lograr que todas sean distintas.** Tiene que permitir que se distingan únicamente cuando la física ya presente produzca historias diferentes.

---

## 3A. COMPONENTE 22 OBLIGATORIO — FLUCTUACIONES CUÁNTICAS QCD

El inventario anterior de 18 elementos + 3 mecanismos estaba incompleto. El director añade explícitamente las **fluctuaciones cuánticas del sector quark–gluón**.

En lenguaje simple: dentro de un protón o neutrón no hay únicamente tres quarks quietos. Hay una actividad QCD de campos, gluones y pares quark–antiquark que aparece en las correlaciones/ocupaciones del estado. Esa dinámica aporta energía y masa efectiva al hadrón.

Consecuencias obligatorias:

1. La materia hadrónica no puede cerrarse calculando `masa_protón = masa(q1)+masa(q2)+masa(q3)`.
2. El motor debe llevar un libro contable separado de masa/energía de valencia, enlace fuerte, campo gluónico, mar/pares y condensado o representación equivalente adjudicada por CS.
3. La gravedad debe leer la energía-masa física resultante del hadrón, no una masa artificial puesta para obtener atracción.
4. El NULL `sin_fluct_qcd` conserva quarks de valencia y demás leyes, pero apaga únicamente la contribución dinámica QCD añadida; sirve para medir qué cambia gracias a ella.
5. “Fase cuántica” CS069 y “fluctuaciones QCD” no son sinónimos y no se cubren con una sola casilla.

Prohibiciones específicas:

- no usar `np.random`, ruido o sorteos y llamarlos fluctuación cuántica;
- no crear miles de objetos virtuales clásicos con identificadores individuales;
- no inventar una tasa de aparición, cupo de pares o vida media por conveniencia;
- no hacer aparecer/desaparecer energía sin conservarla en el estado total;
- no declarar que el viejo peso escalar `W` ya contiene automáticamente todo QCD sin demostrarlo.

**Bloqueo honesto:** el director adjudicó la presencia física de este componente, pero no una discretización computacional particular. Antes de codificarlo, CS debe especificar la representación permitida —ocupaciones, correladores, energía de campo u otra—, las cantidades conservadas y el NULL. CC no elige esa arquitectura solo. Si falta, se detiene con la pregunta exacta y no marca 22/22.

---

## 4. CONDICIÓN INICIAL TÉRMICA

Se construyen dos condiciones con el mismo catálogo y el mismo presupuesto térmico total:

### H — temperatura homogénea

Todos los estados comienzan con la misma temperatura física.

### G — temperatura no homogénea

Los estados comienzan con una distribución térmica desigual que representa la variación del gradiente inicial declarada por el director.

Reglas:

1. La distribución se define como un **multiconjunto físico de temperaturas**, no como `temperatura[i] = fórmula(i)` usada luego como identidad.
2. El orden usado para almacenarla es irrelevante y se somete a la prueba de permutación completa.
3. H y G deben tener el mismo presupuesto térmico total según la definición de energía declarada por el modelo. Igualar sólo la temperatura media no basta si la energía usada por el motor no es lineal en `T`.
4. CC no elige una única amplitud “que funciona”. Se barre el rango de asimetría térmica fijado antes de mirar la salida y se publica completo.
5. El gradiente es una condición física de entrada; el diámetro, la dimensión y la cantidad de firmas son salidas. El perfil térmico no puede diseñarse para producir esas salidas.

---

## 5. REHOMOGENEIZACIÓN — LO QUE INTENTA BORRAR LA DIFERENCIA

Las relaciones físicas vivas intentan intercambiar calor. El intercambio debe:

- depender sólo de la diferencia térmica y de la relación física viva `W`;
- trasladar energía entre estados, no crearla ni destruirla;
- tratar simultáneamente todos los pares físicamente equivalentes;
- conservar el presupuesto térmico cuando la expansión está apagada;
- ser independiente del orden de las filas.

CC puede usar una integración numérica estable del intercambio térmico, pero el tamaño del paso es un parámetro **numérico**, no físico. Debe demostrar que reducir el paso y aumentar la resolución no cambia el veredicto.

---

## 6. EXPANSIÓN — LO QUE IMPIDE QUE LA DIFERENCIA SE BORRE

La expansión es global y no asigna una etiqueta distinta a cada nodo.

Su función física en este experimento es reducir el tiempo/oportunidad de interacción antes de que el intercambio térmico rehomogeneice el sistema. Debe actuar sobre el acoplamiento relacional vivo, no fabricar directamente una temperatura única por nodo.

La expansión:

- no puede crear diferencias en el brazo homogéneo;
- puede preservar o aumentar diferencias que ya estaban en G;
- debe llevar contabilidad de la energía que sale del sector térmico por enfriamiento/trabajo de expansión, en vez de hacerla desaparecer sin registro;
- se barre como condición física frente al tiempo de rehomogeneización; no se selecciona después el valor que produce ocho firmas o una dimensión deseada.

La pregunta no es “¿qué tasa dibuja la curva bonita?”. Es:

> ¿En qué régimen la expansión gana la carrera contra la rehomogeneización y deja persistir la diferencia inicial?

El resultado correcto es un mapa completo de esa carrera, incluido el régimen donde la diferencia sí se borra.

---

## 7. ACTUALIZACIÓN SIMULTÁNEA

En cada paso:

1. Todas las piezas leen el mismo estado al inicio del paso.
2. El intercambio térmico, la expansión y las demás fuerzas calculan sus consecuencias sin modificar en cascada la entrada de las otras.
3. Todas las consecuencias se aplican juntas al final.

El orden en que las funciones aparecen en el archivo no puede cambiar la física. Permutar el orden de los operadores debe producir el mismo estado dentro de la tolerancia numérica declarada.

No se permite resolver quién sobrevive, formar materia o fijar la geometría antes de que el proceso que supuestamente lo causa haya ocurrido.

---

## 8. EMPATES — LA FÍSICA DECIDE O NO HAY DECISIÓN

Cuando varias relaciones están exactamente empatadas:

- no se escoge la primera;
- no se escoge por índice;
- no se introduce ruido;
- no se reparte una identidad artificial.

El operador actúa simétricamente sobre la clase empatada o conserva el empate. Si una pieza exige seleccionar dos vecinos pero la física ofrece cuatro equivalentes, CC no puede elegir dos: debe implementar la acción simétrica definida por CS o detenerse y preguntar.

Un resultado congelado por simetría física es un resultado honesto. Un resultado destrabado por número de fila es un motor inválido.

---

## 9. LOS CUATRO BRAZOS OBLIGATORIOS

Mismo catálogo, mismo presupuesto térmico y misma resolución numérica:

1. **H0 — homogéneo, sin expansión.**
2. **H1 — homogéneo, con expansión.**
3. **G0 — gradiente térmico, sin expansión.**
4. **G1 — gradiente térmico, con expansión.**

Predicción preinscrita del director:

- H0 y H1 no tienen diferencia térmica que amplificar.
- G0 permite que las interacciones intenten rehomogeneizar la diferencia.
- G1 puede congelar o amplificar la diferencia si la expansión vence a la rehomogeneización.

No se exige que el resultado sea `1–1–4–8` en el motor completo. Ese patrón pertenece al toy de ocho nodos. Lo que se exige es la interacción causal: la expansión no rompe por sí sola la simetría homogénea y su efecto sobre G debe medirse frente a G0.

---

## 10. PRUEBA DE INVARIANZA — ESTADO COMPLETO, EN CADA PASO

Para cada brazo, CC ejecuta al menos estos reordenamientos deterministas:

- orden original;
- orden invertido;
- rotación por bloques;
- intercalado de las clases físicas.

Después de deshacer cada permutación, deben coincidir en cada paso:

- `T`;
- `W`, relación por relación;
- memoria de enlaces;
- propiedades dinámicas del catálogo;
- marcos y variables internas;
- tiempos causales;
- observables finales.

No basta comparar el número o el conjunto de firmas. Cada relación física debe volver a su lugar correspondiente. La diferencia máxima se guarda y se informa.

La tolerancia numérica se fija antes de correr y se acompaña de una prueba de refinamiento. Si una comparación exacta es posible, se exige igualdad exacta.

---

## 11. MEDICIONES EN LENGUAJE SIMPLE

El informe debe comenzar explicando:

- cuánta desigualdad térmica había al empezar;
- cuánto logró borrarse o crecer;
- si la expansión alcanzó a separar el sistema antes de que se igualara;
- cuándo comenzaron a divergir las historias `W`;
- cuántas clases físicas siguieron empatadas;
- si el resultado cambió al reordenar el archivo.

Luego entrega los datos técnicos:

- contraste/variación térmica por paso;
- presupuesto de energía y su contabilidad;
- firmas relacionales por paso y regla exacta para distinguirlas;
- diferencia máxima en la prueba de permutación;
- sensibilidad a resolución numérica;
- mapa completo asimetría térmica × expansión/rehomogeneización;
- tiempo y memoria computacional.

---

## 12. GEOMETRÍA Y ESCALA — SÓLO DESPUÉS DE PASAR LAS GUARDAS

Una vez que el motor completo pasa las secciones 2–11, se barre N y se pregunta si la red relacional produce extensión o sólo un grumo.

Se mide como mínimo:

- diámetro frente a N;
- fracción conectada;
- distribución de grados/hubs;
- `β` y el juez independiente ya fijado por CS;
- comparación real contra NULL;
- G-DIM-NO-ETIQUETA.

Romper empates no equivale todavía a producir espacio. El motor debe ganarse ese resultado. Un negativo con todas las guardas aprobadas será un negativo legítimo; antes de eso no.

La hipótesis de que la misma asimetría térmica produce además el desbalance materia–antimateria queda **separada y pendiente** hasta que el mismo estado térmico cause ambos resultados dentro del motor. CC no puede declararla confirmada por compartir el nombre `ε`.

---

## 13. ARTEFACTOS OBLIGATORIOS

CC crea archivos nuevos; no sobrescribe ni reescribe la historia anterior:

- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs072_motor_gradiente_termico.py`
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs072_gradiente_termico_resultados.json`
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs072_gradiente_termico_run.log`
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/INFORME_CS072_gradiente_termico_expansion_PARA_CS.md`

El JSON contiene configuración completa, rangos barridos, versión de código, pruebas de invarianza, contabilidad térmica y todos los puntos —no sólo los favorables.

El informe incluye una sección inicial titulada exactamente:

> **Responsabilidad metodológica: por qué las corridas anteriores no falsaron la premisa del director**

---

## 14. GATE FINAL ANTES DE DECLARAR RESULTADO

La corrida sólo es interpretable si todas son `SÍ`:

- [ ] ¿La única asimetría inicial es la variación térmica declarada por el director?
- [ ] ¿La expansión impide la rehomogeneización en vez de fabricar etiquetas?
- [ ] ¿No existe lectura del índice en la física?
- [ ] ¿Los empates se tratan simétricamente?
- [ ] ¿La energía está contabilizada?
- [ ] ¿Los cuatro brazos usan el mismo catálogo y presupuesto?
- [ ] ¿La prueba de permutación compara el estado completo en cada paso?
- [ ] ¿Los resultados convergen al refinar el paso numérico?
- [ ] ¿Se publicó el barrido completo sin selección post hoc?
- [ ] ¿Cada pieza declarada modifica realmente el estado común?
- [ ] ¿Las fluctuaciones QCD están implementadas de verdad, separadas de fase cuántica y sin RNG/perillas?
- [ ] ¿La masa/energía hadrónica incluye un libro contable QCD y no sólo tres masas de valencia?
- [ ] ¿El informe está explicado primero en lenguaje simple?

Un solo `NO` invalida la interpretación y obliga a detenerse. CC no parcha después de mirar el resultado: informa el incumplimiento a CS/director.

---

## 15. ADVERTENCIA FINAL DEL DIRECTOR — LITERAL

> **“Si CC vuelve a hacer alguna tontera, queda relegado a las IAs para responder el estado del clima leyendo los reportes en internet, porque su inteligencia sólo sirve para eso.”**

Traducción operativa: ésta es la última oportunidad de CC como implementador experimental. Cualquier nueva sustitución de la premisa, uso de índice, parámetro dibujador, pieza decorativa o decisión no autorizada implica su retiro del diseño e implementación de los experimentos.

No hay permiso para “mejorar” esta instrucción. Hay permiso para implementarla o para detenerse y preguntar.

— Codex, por instrucción directa de Alexis, director del experimento.  
— 18-jul-2026.

---

## 16. ADDENDUM BLOQUEANTE — RESPUESTA A LA DUDA CORRECTA DE CC Y CORRECCIÓN A LA RESPUESTA DE CS

CC hizo exactamente lo correcto al preguntar por la fórmula antes de codificar. La respuesta posterior de CS —“copiar literalmente el toy y considerar `0.02` una constante física global”— **NO queda autorizada**.

### Lo que CC sí puede hacer con el toy

Puede leerlo y reproducirlo como prueba de regresión histórica:

- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs072_toy_gradiente_termico_expansion.py`
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs072_toy_gradiente_termico_expansion_resultados.json`
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs072_toy_gradiente_termico_expansion_run.log`

Debe poder confirmar qué hace ese juguete. **No puede trasplantar sus ecuaciones ni sus números al motor físico como si hubieran sido derivados.**

### Por qué `0.02` no se convirtió en constante física

Que un mismo número se aplique a todos los nodos no lo convierte en constante de la naturaleza. El criterio del director sigue siendo: si al mover el número cambia la forma, velocidad o cantidad del resultado, y el número no proviene de una magnitud física identificada, es una perilla.

Auditoría Codex reproducida sobre el toy:

- tasa `0`: 4 firmas incluso después de 40 pasos;
- tasa `1e-6`: 4 firmas después de 40 pasos;
- tasa `1e-4`: aparecen 4, 7 u 8 firmas según la cantidad de pasos;
- tasa `0.02`: aparecen 8 firmas ya a los 2 pasos.

Por tanto `0.02` y `40 pasos` sí deciden cuándo y cuántas firmas aparecen. Son parámetros del toy, no una ley física validada.

Lo mismo rige para:

- `W = 0.9*W + 0.1*aff`;
- gradiente `±0.1`;
- `aff = exp(-|ΔT|/T_media)`;
- cualquier tolerancia usada para declarar dos firmas diferentes.

### La ecuación del toy incorpora la conclusión

La línea:

```python
T = T * (1 - 0.02*(T.max() - T)/(T.max() + 1e-9))
```

ordena explícitamente que el máximo no se enfríe y que cada estado se enfríe más cuanto más frío ya estaba. Es una **regla de amplificación de contraste puesta en el código**. El toy demuestra que esa regla produce más firmas; no demuestra que la expansión física de este experimento tenga esa ley.

La frase canónica del director es distinta y más precisa: la expansión fue demasiado rápida para que la diferencia térmica pudiera rehomogeneizarse. En el motor físico, esa competencia debe surgir de:

1. intercambio térmico conservativo sobre las relaciones vivas;
2. expansión que reduce la oportunidad/fortaleza/duración de ese contacto;
3. comparación completa de los tiempos físicos de ambos procesos.

CC no debe reemplazar esa carrera por una orden directa de “hacer más frío lo frío”.

### La igualdad de energía tampoco está verificada en el toy

`np.linspace(-0.1, 0.1, N)` conserva la media de `T`, pero conservar la media no equivale automáticamente a conservar energía térmica. Para N=8, la distribución `0.9…1.1` tiene la misma suma de temperaturas que el brazo homogéneo, pero `ΣT⁴` es aproximadamente 2.57% mayor. Si el sector se interpreta como radiación, los brazos no tienen el mismo presupuesto energético. CC debe usar la definición física de energía declarada por el modelo y normalizar los controles con ella.

### Los artefactos del toy no son todavía una reproducción cerrada

El JSON guardado contiene `firmas_t0` y `max_dif`, pero el script guardado no calcula ni escribe esos campos. El script imprime un objeto diferente y tampoco genera por sí mismo los tres artefactos. Esto no invalida la observación conceptual, pero impide tratar el paquete actual como evidencia de producción completamente reproducible.

### Adjudicación exacta para CC

1. **NO copiar** `0.9`, `0.1`, `0.02`, `40`, `±0.1` ni la ley diferencial del toy al motor completo.
2. Implementar las secciones 4–7 de esta instrucción: presupuesto térmico común, intercambio conservativo y expansión que compite contra la rehomogeneización.
3. Usar el operador de expansión físicamente adjudicado para el fold; si no existe una derivación suficiente para acoplarlo al intercambio térmico, CC se detiene y devuelve a CS una única pregunta: **“¿Cuál es la ley física, su procedencia y su unidad, que fija la competencia expansión–rehomogeneización?”**
4. CS no puede contestar con “es global”, “es determinista” o “funcionó en el toy”. Debe identificar la procedencia física o declarar que es una condición barrida completa, nunca una tasa elegida para obtener 8.
5. Sólo después de esta respuesta, CC implementa. No se carga nuevamente esta decisión sobre el director, que ya dio la premisa física en lenguaje claro.

Este addendum prevalece sobre cualquier mensaje posterior que ordene copiar el toy como física de producción.
