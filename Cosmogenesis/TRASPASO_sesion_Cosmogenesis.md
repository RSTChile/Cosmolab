# TRASPASO — Continuación experimento Cosmogénesis (Cosmosemiótica)
### Para una nueva sesión de Claude Science · Director: Alexis López Tapia · 27-jul-2026

---

## 0. Cómo se trabaja (el pacto — leer primero, es inviolable)

- **Roles:** El director (Alexis, "CS" cuando adjudica) diseña y adjudica; **CC** y **Grok**
  implementan y CORREN el código; el asistente (Claude Science) **diseña experimentos y
  adjudica resultados verificando en disco**. El asistente NO corre producción salvo que se
  le pida; su rol es diseñar y juzgar.
- **Anti-Shannon (regla madre):** nada cuenta sin ganarle a su NULL; ningún número puesto a
  mano; ningún parámetro estructural salvo constantes físicas reales; los números físicos
  (5% materia, 7:1 p:n, 1836 masa p/e) son **test contra la SALIDA del barrido, jamás
  entrada**. Barrer siempre rangos MUCHO más amplios que el valor esperado.
- **VERIFICAR EN DISCO, NO DE PALABRA** (lección central de esta sesión, marcada 3× por el
  Auditor): antes de escribir "verifiqué X", el valor de X tiene que estar IMPRESO en la
  salida que estás mirando. "La celda no dio error" ≠ "se ejecutó" — un `if` puede saltarse
  en silencio. Esto vale en las dos direcciones: no afirmar "verificado" sin correr, y no
  dictaminar "falso" sin refutar con cálculo (me pasó con la conjetura del jacobiano hoy).
- **Es un PROCESO holístico, no una sucesión de sucesos.** No se le pide a una pieza sola el
  observable del todo. El experimento corre COMPLETO de una vez, no por partes; si falla,
  se aborda dónde falló.
- **NO cerrar ningún experimento hasta que Alexis lo autorice explícitamente.**
  (NOTA_PERMANENTE_CS.md, artifact 84d7f1bd-d43c-4b7a-a8d7-550cbee7dec5.)
- **Hablar en lenguaje simple**, sin jerga ni siglas en inglés — el director lo pidió
  reiteradamente, y su lectura en lenguaje llano CAZA errores que la jerga esconde.
- **NO reformular la Teoría** hasta que algo esté comprobado experimentalmente.
- **RMD 2.0 / MAPAR están FUERA de alcance** de Cosmosemiótica (rama propia del director).

## 1. Rutas y entorno

- **Desktop (fuente):** `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/`
- **Workspace:** el de la sesión (se sincroniza a Desktop con `cp` tras cada `save_artifacts`).
- **Python:** entorno `python` (numpy; hoy se agregaron `openpyxl` y `sympy`).
- **Motor holístico:** `cs072_modulos/` + `cs074_energia_holistica.py`, función
  `correr_holistico_energia`. NO se modifica; solo se llama.
- **Memanto** (memoria persistente del equipo): servicio en :8000 del Mac del director —
  NO alcanzable desde el sandbox; lo consulta CC y pasa el `recall`.

## 2. Estado del experimento — qué está probado y qué no

**Dos ramas, no confundirlas:**
- **Topológica (CS064–CS072), CERRADA:** probó que a nivel de grafos, **S>0 basta para
  generar estados topológicos diferenciados**, pero se agota en el momento pre-partículas
  (no hay entidades con propiedades físicas). π, dirección, geometría, dimensión: todas
  **contingentes** (Mundo B), inmunes al fraude del 7:1.
- **Física (Enfoque 5, holístico cs074, A/B/C), EN CURSO:** veredicto actual =
  **el modelo da RELACIÓN y PROCESO, no los NÚMEROS físicos del universo.**

**Lo que sostiene (con control real, verificado en disco):** conservación de energía (fuga
1,7%), muerte térmica ≠ Nada, la expansión rescata estructura (88,4% sin enfriar vs 60,7%),
gravedad indispensable (60,7%→2,0%), y un hallazgo emergente propio: **demasiada asimetría
inicial DESTRUYE estructura** (experimento A) en tres regímenes (meseta ~77% hasta ε≈0,5 /
fragmentación ε≈0,9–2,3 / colapso ε≳3,8), por mecanismo **mecánico no energético** (control
sin energía da curva idéntica, dif +0,0 en 20 valores de ε).

**Lo que NO sostiene:** la fracción de materia 4,9% NO emerge (z=1,37 bootstrap sobre 4180
puntos = no significativo, indistinguible del azar del barrido). El enfriamiento H₂ no actúa
sobre la estructura (experimento B, z≈−0,12 plano). Las constantes p:n y masa p/e no son
evaluables (son entrada del motor, no salida; la masa p/e da 18 en vez de 1836 porque usa
masas desnudas de quark, falta el ~99% que es energía de ligadura).

## 3. LO QUE SIGUE — el experimento diseñado, listo para correr

**DISEÑO_barrido_fino_banda_estrecha_PARA_CC.md** — contesta la pregunta abierta del
director: **¿la estructura vive en una banda estrecha NO azarosa del espacio completo de
configuraciones, o emerge en cualquier lado?** Barre TODAS las variables físicas juntas
(asimetría 1e-6→10, tasa expansión, reserva energía, poblaciones de partículas) por muestreo
Latin Hypercube (2000 configs × 12 semillas ≈ 24.000 corridas), con NULL por barajado de
densidades, y distingue banda de ruido midiendo **conexidad del cúmulo z>2** en el espacio de
parámetros. Tres lecturas pre-inscritas: banda estrecha / todo el espacio / disperso.
**Está listo para pasar a CC.** Cuando devuelva, adjudicar verificando en disco la métrica de
conexidad (es la que decide).

**Pendientes menores abiertos (honestos, no resueltos):** el borde INFERIOR de la asimetría
(A no lo encontró — a ε=0,001 aún hay 77% estructura; el barrido fino lo busca hasta 1e-6);
por qué el enfriamiento no actúa (¿60 pasos corto? ¿presión térmica domina?).

## 4. Documentos a revisar (enlaces)

**Adjudicaciones y resultados del arco físico:**
- [ADJUDICACION_cs074_ABC_CS.md]({{artifact:02711628-266f-48ea-836d-f34c054b4250}}) — veredicto A/B/C (v2, verificado en disco)
- [ADJUDICACION_ENFOQUE5_30de30_CS.md]({{artifact:8a8eadb3-5718-4bf3-959a-226b8b213c66}}) — arco Enfoque 5 (30 exp)
- [RESUMEN_sesion_Cosmogenesis_Web.md]({{artifact:a3837eec-d1a8-436b-add1-b2641b3cd3de}}) — resumen del transcript largo

**Diseño listo para correr:**
- [DISENO_barrido_fino_banda_estrecha_PARA_CC.md]({{artifact:41da629e-3b0a-4a63-b9ef-202f5a0dfda9}}) — **el próximo experimento**

**Contexto teórico (marco de esta sesión — conceptual):**
- [NODO_CANDIDATO_campos_sustratos_S0.md]({{artifact:0529af66-ab29-476e-b482-f5f033035b8e}}) — campos físicos como sustratos de S>0 + principio general ("en potencia" = filtro)
- [NOTA_LECTURA_persistencia_genealogia_S_I_E.md]({{artifact:8d06e75c-c840-4e1f-a195-809c8b404638}}) — persistencia ⟺ genealogía ⟺ S=I⟷E
- [NOTA_sustrato_comunicacion_EM_y_entorno.md]({{artifact:cf627eb2-2d00-4d79-9136-cd41c121a642}}) — dos planos (alma transustrato / comunicación por sustrato común) + especulación EM
- [PENDIENTE_campos_emergentes_ANIMA_Pinotsis_Miller.md]({{artifact:5fd97b31-b8f9-4714-868e-3947b1dd3f50}}) — validación externa (campo eléctrico del cerebro) + experimento ANIMA

**Literatura de masa (base del rediseño físico):**
- [INFORME_CONSOLIDADO_MASA_ME.md]({{artifact:c79f1c1d-39e7-4a84-8b5a-b8880bfb7253}}) — Higgs da ~1% de la masa; ~99% es ligadura QCD; dos emergencias (electrodébil ~10⁻¹¹s, QCD ~10⁻⁵s)
- [LINEA_TIEMPO_MASA_topologia_vs_fisica.md]({{artifact:ac9739f1-7c6b-404a-b7b8-64b2e4991821}}) — qué probó cada rama por época

## 5. Notas sueltas
- El nodo candidato de los campos espera decisión del director: DÓNDE se integra en la
  Canónica (respetando que cada nodo derive del anterior) y su FORMATO canónico. No tocar la
  Canónica sin su OK.
- Validación externa confirmada esta sesión: Pinotsis & Miller (*NeuroImage* 2022) — la
  memoria del cerebro vive en el campo eléctrico, no en las neuronas. Se llegó por
  Cosmosemiótica ANTES del paper (convergencia independiente).
- Anthropic AI-for-Science: ANIMA se postula por la puerta biológica (no publicar sin OK).

*Todo sincronizado en Desktop/Cosmogenesis y guardado como artefactos. Para continuar:
empezar por el barrido fino (sección 3), pasárselo a CC, y adjudicar verificando en disco.*