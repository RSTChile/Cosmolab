# `_archive_v1_entidades/` — CG001, la primera implementación (archivada)

**Qué es:** la PRIMERA implementación ejecutable de Cosmogénesis (**CG001**, 29-jun-2026): un núcleo de
simulación por entidades con servidor Docker 24/7 y visualización 3D en vivo (WebGL/Three.js), análogo
a la arquitectura de ANIMA/Célula_Madre. Ejecutor técnico: Grok (agente Cursor); revisión: Claude
(Club Abulafia). **Archivada y superada** por el arco CG002 (raíz del proyecto) y, después, por la
serie numerada CS057→CS073 (`cs072_modulos/` es el motor vigente).

## Por qué está archivada, no borrada (la trazabilidad importa)
La primera entrega tenía una **fuga de Shannon real**: el informe afirmaba «única diferencia: ε en
entidad id=0», pero el código filtraba ε a sitios adicionales que confundían la comparación A/B y
**programaban parte del resultado por diseño** (`INFORME_REVISION_CG001.md`, veredicto: "el control
experimental NO es limpio"). Se reconstruyó cerrando dos canales que determinaban estructura por
diseño en vez de por interacción — geometría (paso de movimiento sesgado → gaussiano isótropo) y
métricas (`INFORME_RECONSTRUCCION_CG001.md`). Quedó verificada (smoke-test) pero **nunca se relanzó a
producción**: el programa avanzó a CG002 (modelo más simple, sin interior, más fácil de auditar) en su
lugar. Se conserva como registro de ese primer intento y de la lección (cómo se ve una fuga de Shannon
en código real, y cómo se cierra).

## Estructura interna
- `CG001/` — el código: `core/` (universo, entidades), `server/` (WebLive), `visualization/` (3D),
  `experiments/`, `config/`, `metrics/`, `observatorio/`, `logs/`.
- `docker/` — despliegue Docker 24/7.
- `tools/`, `venv_viz/` — utilidades y entorno de visualización.
- `INFORME_CG001_Implementacion_Docker_3D.md` — la entrega original (mandato, alcance, qué NO se
  implementó del protocolo de 44 págs.: VisPy/PyQt6, 100.000 entidades, grilla 512³, Unity/Godot).
- `INFORME_REVISION_CG001.md` — la auditoría que encontró la fuga de ε.
- `INFORME_RECONSTRUCCION_CG001.md` — el cierre de las dos fugas (geometría + métricas).
- `INFORME_FIX_VISUALIZACION_CG001.md` — un fix menor de visualización.

**No usar como base de código nueva** sin releer `INFORME_REVISION_CG001.md` — documenta exactamente el
tipo de error (control A/B contaminado) que todo el programa posterior (CS072/CS073 incluidos) se
cuida activamente de repetir.
