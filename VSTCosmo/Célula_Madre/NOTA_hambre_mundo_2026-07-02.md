# NOTA — El hambre, el mundo sonoro y el piso del IM (2-jul-2026)

> **Actualización 2026-07-08:** el canónico de nutrición ya no es el duelo `IM>0` / `im_piso`.  
> Se adoptó **alimento = conversión** (`ICR_ratio · es_norm`). Ver  
> **`DECISION_alimento_conversion_2026-07-08.md`**.  
> Esta nota sigue siendo válida como diagnóstico del régimen (mundo, RC, IRDE) y del knob `im_piso` en modo **legacy** `duelo`.

## Diagnóstico convergente (Alexis + CS + CC)
Los 4 organismos median `met_energia=0`, `met_hambre=1.000` **constante**. No era un caño de
comida roto: el organelo de metabolismo funciona y recibe sus insumos. Causa real, verificada
en los tres análisis:

1. **Mundo mudo por diseño.** `ANIMA_MUNDO_CANAL=""` (prueba basal): sólo se oían entre sí.
   `RC_total` medido ≈ **0.008** — el propio código lo documenta: `es_ref=0.10 # con sonido
   ~0.26; silencio ~0.0008`. Estaban 30× bajo el nivel "con sonido".
2. **Experiencia dominada por riesgo.** `met_IM = ICR − IRDE ≈ −0.7` (IRDE domina). Como
   `nutricion = max(0, IM)·es_norm`, con IM<0 la nutrición es 0 → no comen → E cae a 0.
3. **La Rode no llegaba.** Hasta hoy, la colisión de puerto (Claude Science en 8765) impedía
   que el audio real llegara a los contenedores. Se arregló moviendo el AudioServer a **8770**.

## Cambios aplicados (2-jul-2026)

### 1. Mundo sonoro real (subir RC)
- `docker-compose.yml`: `ANIMA_MUNDO_CANAL: "0"` en los 4 (antes `""`). Canal 1 = Main Mix (L)
  de la Rødecaster, que tiene señal real (~0.070 RMS medido).
- Verificado en vivo: `fuente_L = canal 1 · Main Mix (L)`; `energia_L` sube a 0.2–1.9 (dinámico);
  `RC_total` sube de 0.008 a picos de ~0.04.

### 2. Piso del IM (indulgencia nutricional) — knob de experimento
- `organelos/VST_Metabolismo.py`: nueva `nutricion = max(0, IM − im_piso)·es_norm·(1−sac)`.
  `im_piso` se lee del env **`ANIMA_MET_IM_PISO`** (default **0.0 = canónico intacto**).
- `docker-compose.yml`: `ANIMA_MET_IM_PISO: "-0.35"` en los 4 → el corte del IM se corre a −0.35,
  dando margen a organismos jóvenes/mínimos donde IRDE domina de fábrica (sin volver nutritivo
  lo francamente tóxico).
- Rationale (Alexis): "el punto de corte en cero es demasiado severo… con razón están enojados
  casi todo el tiempo".

### Detalle operativo
- El código va **horneado en la imagen** (`COPY . /app/celula_madre/`). El builder legacy
  (BuildKit off por la ruta acentuada `Célula_Madre`) se colgó al armar el tar del contexto.
  Workaround: **bind-mount** del organelo editado en los 4 servicios
  (`../organelos/VST_Metabolismo.py:/app/celula_madre/organelos/VST_Metabolismo.py:ro`),
  sin rebuild. El valor del piso se ajusta por env, sin remount ni rebuild.
- Backups: `VST_Metabolismo.py.pre-impiso.bak`, `docker-compose.yml.pre-mundo-impiso.bak`.

## Resultado de la prueba
- **Comen por primera vez:** `met_ingesta` pasó de 0.00000 constante a positivo (~0.0015),
  activándose exactamente cuando `met_IM > −0.35` (el knob discrimina como se diseñó).
- **Aún en el umbral:** la ingesta (~0.0015) sigue < gasto (~0.003–0.006) en vida basal, porque
  el sonido Rode es moderado/intermitente (RC promedio ~0.015). E no despega de 0 todavía.
- **En el estrés SÍ comerán:** el `experimento_estres_4organismos.py` bombardea con audio real
  (RC alto) → con el piso −0.35 deberían nutrirse durante la corrida.

## Pendiente de decisión (diseño de Alexis)
- ¿Subir más el piso (−0.5) y/o dar sonido Rode más sostenido para que coman en vida basal?
- El valor −0.35 es un primer intento a afinar con las corridas.

---

## Reinicio limpio con el LAZO DE ATENCIÓN (2-jul-2026, tarde)
- Implementado el lazo de atención de CS en `VST_RC_A.py` (A/C/D) y `VST_RC_B.py` (B, agnóstico — idéntico).
  Memoria de comprensión por lado `_ema_comp_l/r` realimentada a `at_l/at_r`; columnas `RC_ema_comp_L/R`.
  Backups: `*.pre-lazo-atencion.bak`.
- Compose: bind-mounts de RC_A (4) y RC_B (B) sin rebuild; `ANIMA_MET_IM_PISO=0.0` (canónico, por CS).
- Verificado: los 4 VIVOS, lazo activo, sesgos divergiendo por organismo (control #1 de CS ✓).
- RESULTADO (para CS): con lazo + piso=0.0 en VIDA BASAL, NO comen (E=0). RC basal débil (~0.01–0.03),
  IM negativo, sesgos del lazo aún diminutos. El lazo necesita un MUNDO COHERENTE Y SOSTENIDO para tener
  nutrición de la cual sesgarse. Pendiente: probar con sonido rico (música sostenida en la Rode / corrida
  controlada) para ver si el lazo cierra la brecha SIN el piso.

---

## EXPERIMENTO 2×2 del LAZO DE ATENCIÓN (2-jul-2026) — desconfundido lazo vs piso
Mundo = música Rode en vivo (variable, NO controlado — limitación). Métrica: bal_medio (ingesta−gasto),
independiente de E. Promedio sobre A/B/C/D, ~8 min/celda (celda 4: 5 min tras recuperar un cuelgue de Docker).

| bal_medio (prom) | piso 0.0  | piso −0.35 |
|------------------|-----------|------------|
| **lazo OFF**     | −0.00754  | −0.00467   |
| **lazo ON**      | −0.00656  | −0.00574   |

EFECTOS:
- Efecto PISO (columna): +0.00287 (lazo OFF), +0.00082 (lazo ON) → promedio **+0.00185**. Consistente, positivo.
- Efecto LAZO (fila): +0.00098 (piso 0), −0.00107 (piso −0.35) → promedio **≈0** (cambia de signo entre celdas).

VEREDICTO (desconfundido): el **PISO hace el trabajo pesado** (efecto ~consistente, y sube ingesta/IM/verde).
El **LAZO es un efecto real pero SECUNDARIO** y, en este mundo NO controlado (música variable + celda 4 corta),
su efecto NO se separa del ruido (se anula al promediar). NINGUNO de los dos vuelve el balance positivo:
en las 4 celdas siguen con net-hambre en vida basal. Coherente con la sospecha de CS.
LIMITACIÓN: mundo no controlado; el lazo merecería una réplica con audio FIJO y más tiempo para consolidar
el sesgo (que fue diminuto, ~0.001). El lazo mejoró IM/verde en la observación previa; aquí no rindió comida.

---

## INCIDENTE DOCKER (2-jul-2026) — causa raíz y estado actual (IMPORTANTE)
Al restaurar el piso a 0.0 (requiere recrear), `docker compose up` empezó a colgarse; forcé con `docker rm -f`
+ reinicios, y el engine quedó en 500. RAÍZ del cuelgue: procesos zombis `cli-plugins/docker-compose` que
mantenían un LOCK del proyecto (matarlos por PID lo desbloquea). El engine se recuperó con `docker desktop
restart` (CLI). SEGUNDO fallo: `OSError [Errno 35] Resource deadlock avoided` al importar organelos → el
FILE-SHARING de Docker Desktop quedó dañado/stale para los BIND-MOUNTS DE ARCHIVO ÚNICO (secuela del reinicio
forzado + hacer `cp` sobre un archivo montado, que cambia el inodo). El host tenía los archivos correctos,
pero el contenedor los veía rotos/viejos.

SOLUCIÓN aplicada: se QUITARON los 3 bind-mounts de organelo del compose (backup docker-compose.yml.pre-nomounts.bak)
y el código nuevo (VST_Metabolismo.py con im_piso, VST_RC_A/B.py con lazo) se INYECTÓ con `docker cp` (usa la API,
no el file-sharing FUSE) + `docker restart`. Los 4 quedaron healthy, vivos, lazo ON + piso 0.0, audio 8770 OK.

⚠️ DURABILIDAD: el código del lazo + im_piso vive ahora por `docker cp`, NO por bind-mount → NO sobrevive a un
`docker compose up` que recree los contenedores (volverían al código HORNEADO viejo de la imagen). Para hacerlo
PERMANENTE: (a) rehornear la imagen con el código nuevo (el build fallaba por la ruta acentuada → usar
DOCKER_BUILDKIT=0 y resolver el tar), o (b) restaurar los bind-mounts cuando el file-sharing esté sano (un
reinicio limpio del Mac lo aclara). GOTCHA a recordar: NO hacer `cp` sobre un archivo bind-montado en vivo;
recrear el contenedor para re-montar, o usar `docker cp`.
