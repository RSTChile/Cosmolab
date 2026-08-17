# Arquitectura del experimento — Cosmogénesis

**Qué es este documento:** el mapa de la carpeta. Apunta a los informes/registros que ya existen (no
los duplica) y explica cómo encajan las piezas — generaciones del programa, roles del equipo, y dónde
vive cada cosa. Escrito el 19-jul-2026 (CC) tras documentar cada subcarpeta del proyecto.

---

## 1. La pregunta que organiza todo

Cosmogénesis prueba si **tiempo, espacio, dimensión, estructura, materia y las constantes que las
describen pueden EMERGER, medidas y reproducibles, de un único axioma relacional — S>0 (que una
diferencia persista lo suficiente para no anularse) — sin programarlas a mano.** El criterio de éxito no
es "que el modelo produzca algo interesante": es que cada resultado sea **falsable** (tiene su control
NULL/barajado), **invariante** (no depende del índice/orden — anti-Shannon), y que un desacuerdo entre
lo esperado y lo medido se reporte como DATO, nunca se tape.

**Regla permanente del programa** (`NOTA_PERMANENTE_CS.md`): ningún experimento se cierra —ni con
resultado positivo, ni negativo, ni con match perfecto— sin autorización explícita de Alexis (el
director). Ni CS, ni CC, ni Grok, ni Gemini pueden firmar un cierre por su cuenta.

## 2. Las generaciones del programa (de más vieja a más nueva)

| generación | qué probó | estado | dónde |
|---|---|---|---|
| **CG001** | primera implementación ejecutable (entidades, Docker, visor 3D) | **archivada** — tenía una fuga de Shannon (control A/B contaminado), se reconstruyó, nunca se relanzó a producción | `_archive_v1_entidades/` (ver su README) |
| **CG002** | S>0 → compatibilidad/acoplamiento dirigido → tiempo con flecha, espacio con dimensión heredada, estructura, criticidad, constantes vs. historia contingente | **CERRADO** (30-jun-2026, teórica y experimentalmente) | `CIERRE_ARCO_CG002_AUTORITATIVO.md`, `INFORME_GENERAL_CG002_ARCO.md` (raíz); datos en `logs/`, `demo/` (ver sus README) |
| **CG002 ↔ ANIMA** | puente de traducción de observables entre Cosmogénesis (motor sin interior) y ANIMA/VSTCosmo (organismo con interior) | Fase 0 completa, Fase 1+ **en espera a propósito** — se retoma cuando Cosmogénesis (CS072/CS073) cierre, confirmado con Alexis 19-jul | `PROTOCOLO_EMPALME_CG002_ANIMA.md`; estímulos en `empalme_estimulos/` (ver su README) |
| **CS057 → CS071** | hilos posteriores a CG002 (numeración secuencial única, ver abajo) — incluye el problema del marco/orientación (R7: ningún mecanismo local genera el marco 3D) que reaparece en CS072/CS073 | cerrados o en pausa según el registro | `REGISTRO_EXPERIMENTOS_CS.md` |
| **CS072** | motor integrado: plasma de quarks → bariones → H/He → tiempo → espacio → dimensión, TODO en una sola corrida determinista | **VALIDADO** — comprobación paralela CC↔CS, match exacto (18/19-jul-2026) | `cs072_modulos/` (ver su README); `INSTRUCCION_CS072_CORRER_motor_integrado_PARA_CC.md`, `verificar_cs072_output.txt` |
| **CS073** | del átomo a la primera estrella: gravedad general real, expansión, materia oscura, enfriamiento H₂, criterio de Jeans — ejecución HOLÍSTICA (todo junto, un solo bucle) | **EN CURSO, sin adjudicar** — primera corrida completa: REAL≈NULL (z≈-0.4, diagnosticado como límite de diseño, no de física) + control positivo con masa real: no emerge estrella a 250+250 partículas. Pendiente de que CS lo adjudique | `cs073_cierre_holistico.py` (raíz); piezas nuevas en `cs072_modulos/piezas/` (ver su README); `INSTRUCCION_CC_cierre_holistico.md` v3, `INVENTARIO_atomo_a_estrella_CS.md` |

## 3. Cómo orientarse en la raíz (cientos de archivos sueltos)

La raíz NO está organizada por carpeta — está organizada por **número de experimento**, indexado en:
- **`REGISTRO_EXPERIMENTOS_CS.md`** — el índice maestro. Todo experimento nuevo recibe un número CS
  correlativo único (nunca se reutiliza). Las etiquetas viejas (CG002, R7a, r7b, cg004f3...) se
  conservan como alias para rastrear archivos ya existentes, pero no se usan para nombrar nada nuevo.
- Patrón de nombres: `<TIPO>_<experimento>_<tema>_<autor>.md` — `INSTRUCCION_*` (una IA le pide a otra
  que corra algo), `ADJUDICACION_*` (veredicto firmado), `INFORME_*` (reporte de resultados),
  `RESPUESTA_*` (contestación a una objeción), `DISENO_*` (diseño pre-registrado antes de correr),
  `verificar_*.py` / `cs0NN_*.py` (los scripts ejecutables), `*_output.txt` / `*_log.txt` /
  `*_resultados.json` (evidencia cruda de la corrida).
- `Docs/` — dossier exportado (`.docx`/`.pdf`) del registro, para lectura fuera del repo (ver su README
  sobre por qué puede estar desactualizado).

## 4. Quiénes son quiénes (para leer los documentos sin confundirse)

- **Alexis (el director):** única autoridad para cerrar un experimento. Fija las reglas del método.
- **CS ("Claude Science"):** diseña experimentos, adjudica veredictos, prototipa en su propio entorno
  (nunca en el motor compartido sin coordinar).
- **CC ("Claude Code", este agente):** implementa en el motor real, corre las corridas grandes/costosas
  (en su propio entorno, no en el kernel de CS — límite de RAM), hace la **comprobación paralela**
  (correr exactamente lo que CS diseñó, sin tocar el diseño, y reportar coincidencia o discrepancia
  como dato).
- **Grok / Gemini / otros:** colaboradores puntuales (implementación de infraestructura en CG001,
  segundas opiniones).
- **Regla del pacto** (vigente desde CS073): "CC implementa y corre; no rediseña. Un desacuerdo con el
  diseño es un DATO, se coordina antes de tocar." — y a la inversa, cuando CC encuentra una
  contradicción real en un diseño (ocurrió dos veces en CS073: localidad térmica reintroducida por
  error, y el gate de escala multiplicando de más), se corrige ANTES de escribir código, no después.

## 5. Dónde retomar

CS073 (átomo → estrella) está donde quedó: dos corridas completas y honestas (REAL-vs-NULL nulo por
diseño; control positivo negativo con masa real), reportadas a CS, sin adjudicar. Ver
`cs072_modulos/piezas/README.md` §"Resultado registrado" para el estado técnico exacto, y
`NOTA_PERMANENTE_CS.md` — nada de esto es un cierre hasta que Alexis lo diga.
