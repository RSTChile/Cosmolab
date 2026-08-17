# PROPUESTA CODEX PARA CLAUDE SCIENCE — CS072: DEL “ACANTILADO” A UN MAPA DE FASES DEL TODO

**Autor de la propuesta:** Codex, revisor y colaborador propositivo  
**Deciden el diseño y la adjudicación:** Alexis López Tapia (director) + Claude Science  
**Implementa y ejecuta:** Claude Codex (CC)  
**Fecha:** 17-jul-2026  
**Estado:** propuesta para discusión; **no autoriza nuevas corridas**.

## 0. PROPUESTA EN UNA FRASE

La exploratoria v7 no encontró la banda métrica anticipada en el núcleo aislado: encontró una erosión suave del
hub y un posible acantilado de conectividad. Propongo que CS072 deje de buscar una tasa “correcta” de poda y pase
a probar una interacción falsable: **si el fold completo aporta cohesión local real, debe abrir una región finita
donde pluralidad, conectividad y escalamiento métrico coexistan; si sólo desplaza el acantilado o deja β bajo, no
emergió geometría.**

En lenguaje de la Teoría: la poda sola oscila entre dos silencios —todo unido sin diferencia y diferencias sin
relación—. La pregunta propositiva es si **el TODO**, no una parte aislada, sostiene una relación plural que no
colapse ni se disuelva.

---

## 1. QUÉ DICE HONESTAMENTE LA EXPLORATORIA V7

La implementación de poda por grado es auditable y ciega a longitud:

p_corte(i,j) = tasa · (grado_i + grado_j) / (2 · grado_medio), capada a 1.

Los datos muestran:

1. grado_max disminuye de forma gradual al aumentar la poda. Esto prueba que el operador anti-hub funciona,
   pero no es por sí solo un hallazgo emergente: la regla fue construida precisamente para castigar grado alto.
2. frac_conectada permanece alta hasta la vecindad de poda≈0.08 y luego cae bruscamente hacia fragmentación.
3. No apareció una región donde conectividad alta y escalamiento métrico fuerte coexistieran. En el barrido fino,
   β quedó aproximadamente entre 0.05 y 0.23 mientras frac_conectada era alta; cerca de poda=0.08 mejoró, pero
   la componente gigante ya había perdido una fracción importante.
4. El resultado cualitativo se repite para 1, 5 y 20 focos.
5. El salto 0.080→0.085 es **compatible con** percolación, pero aún no es una transición confirmada: cada tasa
   usó una semilla distinta. Además, poda=0.08 dio β distinto en el barrido grueso y el fino, señal de variabilidad
   entre realizaciones.

**Adjudicación sugerida del estado actual:**

- Operador de poda: **validado mecánicamente**.
- Banda métrica del núcleo aislado: **no observada**.
- Acantilado de percolación: **candidato**, no hecho firme todavía.
- Veredicto A/B de CS072: **no leído**, correctamente.

---

## 2. HIPÓTESIS NUEVA, DERIVADA DEL NEGATIVO

### H-FOLD

La banda de persistencia no tiene por qué existir en gravedad + flujo + memoria + poda aislados. Puede ser una
propiedad de interacción del sistema completo. En particular, el sector que crea o preserva cohesión local
(confinamiento/fuerza fuerte y cualquier otro mecanismo que CS identifique explícitamente como cohesivo en el
código) podría mantener relaciones locales mientras la expansión poda la sobreconexión global.

La predicción no es “saldrá 3D” ni “β será 0.5”. Es:

> Frente al núcleo y sus NULL, el fold completo abrirá —o no— una región de tasas de poda donde haya a la vez
> componente gigante alta, ausencia de hub extensivo y escalamiento métrico reproducible. La dimensión será una
> salida, nunca un objetivo.

### Por qué esta hipótesis no es un rescate ad hoc

- El fold completo ya es el corazón preinscrito de CS072; no se añade un ingrediente después del resultado.
- No se elige una tasa para hacer salir geometría: se comparan regímenes predefinidos que atraviesan cohesión,
  borde y fragmentación.
- La comparación decisiva es una **interacción**: TODO vs núcleo vs ablación cohesiva vs NULL relacional.
- Un mero desplazamiento del punto de fragmentación no contará como geometría.

---

## 3. PUERTA R — CONFIRMAR EL ACANTILADO SIN “AFINAR HASTA QUE SALGA”

Antes del fold costoso propongo un diagnóstico corto, sólo si CS lo autoriza. Su objeto no es hallar una tasa
óptima, sino separar transición real de variabilidad entre semillas.

### Diseño pareado

- poda_tasa ∈ {0.078, 0.080, 0.082, 0.084, 0.086}.
- N ∈ {400, 900, 1600}.
- n_focos = 5 como brazo principal; spot-check de 1 y 20 focos sólo en los extremos y el centro.
- Mínimo 12 semillas.
- **Las mismas semillas en todas las tasas**. Idealmente, usar números aleatorios comunes por paso+arista
  para que cambiar la poda no cambie también toda la historia estocástica por desalineación del RNG.
- Reportar distribuciones e IC, no una sola curva: frac_conectada, grado_max, diam, β y tiempo/paso de
  fragmentación.

### Qué decide esta puerta

- Si el salto se reproduce en la mayoría de semillas y se afila o desplaza sistemáticamente con N: llamarlo
  **transición percolativa del motor**.
- Si el punto varía ampliamente entre semillas: llamarlo **zona estocástica de fragmentación**, no acantilado.
- En ambos casos, no atribuir valor cosmológico al número p_c: depende de la fórmula y del cronograma del motor.

Esta puerta puede omitirse si CS considera que conocer con precisión la forma de la frontera no cambia el fold.
Lo que no recomiendo es densificar la grilla con una semilla nueva por punto: aumenta resolución aparente sin
aumentar evidencia.

---

## 4. PUERTA M — MANIFIESTO CANÓNICO DEL FOLD ANTES DE CODEAR

Los documentos alternan “10 leyes”, “17 ingredientes”, “18 elementos” y “18 + 3 mecanismos”. Antes de implementar,
CC debería recibir de CS un manifiesto congelado llamado MANIFIESTO_FOLD_CS072.md.

Para cada mecanismo:

| Campo | Contenido requerido |
|---|---|
| Nombre canónico | Una sola denominación |
| Fuente | Script y función de origen |
| Variable que modifica | T, enlace, identidad, marco, fase, etc. |
| Momento de acción | Debe estar dentro del mismo paso, activo desde t=0 |
| Parámetro | Valor y experimento del que se hereda |
| NULL | Cómo se rompe su correlación conservando magnitudes |
| Estatus previo | positivo, negativo, parcial o sólo disponible |

El manifiesto debe resolver explícitamente:

1. Si la fase cuántica CS069 está dentro o fuera.
2. Si semilla CS070, memoria CS071, flujo frío→tibio y poda se cuentan dentro de los 18 o como mecanismos de
   origen adicionales.
3. Qué mecanismos forman el sector **COHESIÓN LOCAL** que será ablacionado.
4. Qué orden computacional dentro del paso es inevitable y cómo se controla que ese orden no simule una sucesión
   ontológica.

Sin este manifiesto, un “TODO” no es falsable porque no tiene frontera exacta.

---

## 5. PUERTA F — EL FOLD COMO MAPA DE FASES, NO COMO CORRIDA ÚNICA EN UNA TASA ELEGIDA

### Regímenes de poda

No elegir poda≈0.08 porque allí β fue mayor: eso sería selección post hoc. Definir tres anclas desde la
distribución del núcleo, antes de mirar el fold:

- **P-COHESIÓN:** mayor tasa cuya mediana mantiene frac_conectada ≥ 0.95.
- **P-BORDE:** tasa cuya mediana queda más próxima a frac_conectada = 0.50.
- **P-DISOLUCIÓN:** menor tasa cuya mediana cumple frac_conectada ≤ 0.05.

Agregar poda=0 como control de hub. Estas anclas describen estados del núcleo; no contienen información sobre
la geometría deseada.

### Brazos mínimos

1. **NÚCLEO:** gravedad + flujo + memoria + poda.
2. **TODO:** manifiesto completo CS072.
3. **TODO−COHESIÓN:** mismo TODO, ablacionando sólo el sector cohesivo definido antes de correr.
4. **NULL-RELACIÓN:** mismas magnitudes y cronograma, pero barajando qué identidad/flujo refuerza qué relación.
5. **CONTROL POSITIVO:** proceso sobre sustrato métrico conocido, sólo para demostrar que los jueces siguen viendo
   metricidad y que el TODO no la destruye automáticamente.

### Escalas y semillas para el veredicto

- Al menos cinco tamaños de N, espaciados aproximadamente en log: por ejemplo {400, 700, 1000, 1400, 2000}.
- Al menos 8 semillas para exploración cerrada; 16 si aparece separación y se va a adjudicar A.
- Semillas pareadas entre brazos y tasas.
- Misma duración y mismos parámetros heredados en todos los brazos.

---

## 6. JUECES — MEDIR GEOMETRÍA SIN ELEGIR LA DIMENSIÓN

### Condiciones necesarias simultáneas

1. **Pluralidad conectada:** componente gigante alta y estable con N.
2. **No-hub:** grado_max/N → 0; reportar también distribución de grado, no sólo el máximo.
3. **Escalamiento métrico:** diam ∝ N^β con β positivo, estable entre escalas y con ajuste reproducible.
4. **Dimensión como salida:** si el ajuste es válido, reportar d_efectiva = 1/β; no exigir β=0.5 ni β=1/3.
5. **Segundo sello independiente:** escalamiento de δ-Gromov y/o crecimiento de bolas consistente con la misma
   lectura; no aceptar un β aislado.
6. **Especificidad:** TODO debe separarse de NÚCLEO, TODO−COHESIÓN y NULL, no sólo cruzar un umbral absoluto.

### Criterio propuesto de “región abierta”

Existe región de persistencia sólo si, para al menos una **ancla predefinida** de poda:

- TODO conserva conectividad alta;
- no desarrolla hub extensivo;
- β se separa de NÚCLEO/NULL con intervalo de confianza y no cae hacia 0 al crecer N;
- el segundo juez coincide;
- TODO−COHESIÓN pierde la ventaja.

El valor de d_efectiva se informa después. Que resulte cercano a 3 puede ser una comparación posterior, nunca
el criterio que decide el éxito.

---

## 7. DESENLACES PREINSCRITOS

### A — El TODO abre una región métrica específica

TODO satisface simultáneamente conectividad, no-hub y escalamiento; NÚCLEO, ablación y NULL no. Hallazgo fuerte:
la banda es propiedad de interacción del todo y depende del sector cohesivo.

### B — El TODO sólo desplaza el acantilado

La componente gigante sobrevive a más poda, pero β sigue como mundo-pequeño o no se separa de controles.
Hallazgo útil: las fuerzas aportan cohesión, **no geometría**.

### C — El TODO ensancha una zona conectada pero no define dimensión

Existe una banda topológica estable, pero los jueces métricos son inconsistentes o d_efectiva no converge.
Resultado parcial: persistencia plural sin “hacia dónde”.

### D — No hay rescate

TODO reproduce hub→fragmentación, o los controles lo igualan/superan. Negativo limpio para esta familia de
modelo y esta representación.

### E — El TODO destruye incluso el control positivo

El proceso no permite leer el juez porque borra una métrica conocida. Fallo del instrumento, no veredicto
cosmológico.

---

## 8. GUARDIANES PROPUESTOS

- **G-NO-ELEGIR-PODA:** ninguna tasa se selecciona por su β observado; se usan anclas definidas por conectividad
  del núcleo.
- **G-SEMILLAS-PAREADAS:** mismas realizaciones entre tasas y brazos.
- **G-DIMENSIÓN-SALIDA:** β y d_efectiva se miden; no se apunta a 2D/3D.
- **G-PODA-CIEGA:** la poda no lee longitud, coordenada, δ, β ni conectividad objetivo.
- **G-INTERACCIÓN-ESPECÍFICA:** TODO debe ganar a núcleo, ablación y NULL.
- **G-DOS-JUECES:** ningún positivo se adjudica con un solo estimador.
- **G-TRAZABILIDAD:** script, configuración, semillas, JSON/CSV, log, hash y manifiesto quedan juntos antes de
  redactar adjudicación.
- **G-NO-FÍSICA-PREMATURA:** “percolación”, “entropía”, “inflación” y “geometría” se usan como homologías del
  modelo hasta demostrar el puente; no se convierten automáticamente en afirmaciones sobre el universo físico.

---

## 9. LÍMITE DE ALCANCE QUE CS DEBE FIJAR

El motor vigente todavía arranca con GR.aleatorio, un sustrato mundo-pequeño preexistente. Por ello el fold
propuesto responde honestamente:

> ¿Puede el TODO reorganizar un sustrato mundo-pequeño y abrir en él una región métrica?

No responde todavía:

> ¿Emerge el primer “al lado de” desde la singularidad sin medida previa?

Para esa segunda pregunta propongo una línea posterior —o una bifurcación que CS puede integrar antes del
veredicto— con estado inicial **permutacionalmente simétrico**: todos los pares comienzan con el mismo peso
relacional continuo, sin aristas binarias privilegiadas; ε rompe sólo la temperatura, y la topología se lee
después de que los pesos diverjan. Es más costoso (O(N²) en la forma directa), pero elimina el mundo-pequeño
aleatorio como medida escondida. Si CS072 conserva GR.aleatorio, el veredicto debe declararse condicionado a
ese sustrato.

---

## 10. RECOMENDACIÓN FINAL A CS

1. Adjudicar v7 como **negativo de banda métrica en el núcleo**, con candidato de acantilado todavía no confirmado.
2. Decidir si la Puerta R aporta información necesaria; si sí, autorizar el pequeño estudio pareado. Si no,
   avanzar sin fingir precisión sobre p_c.
3. Congelar el Manifiesto del fold.
4. Implementar el fold como comparación de interacción a través de anclas de poda, no como una corrida en la
   tasa que pareció mejor.
5. Mantener explícita la frontera de alcance del grafo aleatorio.

La propuesta convierte el resultado inesperado en una pregunta más fuerte: **no si la poda sola tiene una banda,
sino si el TODO crea una región de persistencia que ninguna parte posee por separado.** Si aparece y colapsa al
ablacionar la cohesión o barajar la relación, sería una emergencia genuina de interacción. Si no aparece, el
negativo será más limpio y más profundo que seguir afinando la poda.

— Codex, revisión propositiva para Claude Science.
