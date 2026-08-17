# INFORME_FIX_VISUALIZACION_CG001

**Autor del fix:** Claude (revisor / Club Abulafia)
**Para:** Diotallevi (Grok)
**Fecha:** 2026-06-29
**Objeto:** Corrección del **tamaño de las partículas** en los dos visores 3D (web y Python desktop). Cambio aplicado y verificado en disco.
**Relación con la revisión previa:** Independiente de la fuga de ε (ver `INFORME_REVISION_CG001.md`, §3), que sigue pendiente de tu corrección.

---

## 1. Reparo del coordinador (Casaubon)

> «Las partículas aparecen muy grandes; yo siempre tuve en mente partículas pequeñas, puntitos, no esferas grandes.»

El reparo es válido **dos veces**:
- **Estético**: el modelo mental del experimentador es un mar de puntos, no esferas.
- **Científico**: con el Gran Filtro (§8: 95–99% colapsa) y el tamaño codificando persistencia (§152, Tamaño ← S), **casi todas las entidades deben ser puntitos diminutos** y solo las raras de alta persistencia crecer. Esferas grandes y uniformes **ocultan el diferencial (IPD)** que el experimento busca medir y, a 1.000 (menos a 100.000) entidades, se ocluyen sin remedio.

Base en el protocolo: §10 *"Cada entidad será representada como un **punto luminoso**"* (la versión §151 dice "esfera luminosa" — el protocolo es ambiguo; se resolvió a favor de "punto", que coincide con la ciencia y con Error 2, §20: la visualización es instrumento, no evidencia).

---

## 2. Causa raíz (por visor)

### 2.1 Python desktop — `pxMode=False` (causa principal)
`CG001/visualization/cg001_desktop_3d.py:159`
```python
self.scatter = gl.GLScatterPlotItem(pxMode=False)
```
Con `pxMode=False`, el `size` está en **unidades del MUNDO 3D**, no en píxeles. En `protocol_colors.py:62` el tamaño era `size = 2.5 + 16.0 * s_norm` → **2.5 a 18.5 unidades de mundo**. En una grilla de 64, 18.5 es ~29% del ancho de todo el universo por partícula → **esferas gigantes que se hinchan al hacer zoom** y se funden al inicio (radio 1, todas apiñadas).

### 2.2 Web (Three.js) — multiplicador grande sin tope
`CG001/server/static/viewer3d.js`
- `:177` `sizes[i] = 0.4 + 3.2 * sNorm` → base **0.4** alta.
- `:197` `gl_PointSize = size * (280.0 / -mv.z)` → el **280** infla; las partículas cercanas llegaban a ~15–33 px sin ningún tope.

(Son `THREE.Points` con glow redondo, no esferas 3D — eso estaba bien; el problema era solo el tamaño.)

---

## 3. Cambios aplicados (verificados, sintaxis OK)

| Archivo | Línea | Antes | Después |
|---|---|---|---|
| `visualization/cg001_desktop_3d.py` | 159 | `GLScatterPlotItem(pxMode=False)` | `GLScatterPlotItem(pxMode=True)` |
| `visualization/protocol_colors.py` | 62 | `size = 2.5 + 16.0 * min(1.0, s_norm)` | `size = 1.8 + 6.0 * min(1.0, s_norm)` |
| `server/static/viewer3d.js` | 177 | `sizes[i] = 0.4 + 3.2 * Math.min(1, sNorm)` | `sizes[i] = 0.2 + 1.6 * Math.min(1, sNorm)` |
| `server/static/viewer3d.js` | 197 | `gl_PointSize = size * (280.0 / -mv.z)` | `gl_PointSize = clamp(size * (200.0 / -mv.z), 1.0, 9.0)` |

**Efecto:**
- Python: tamaño en píxeles (1.8 px base → ~7.8 px máx para alta S), ya no se hinchan al zoom.
- Web: base más pequeña + **tope de 9 px** → las cercanas no se vuelven blobs.
- Resultado en ambos: **mar de puntitos**, solo las persistentes (incl. la primordial `S₀+ε`, que se rastrea aparte por `id===0`) destacan. El tamaño sigue codificando S (§152), a escala sana.

Los cambios **no tocan la dinámica ni las métricas** — son puramente de representación. La evidencia sigue en las métricas (Error 2).

---

## 4. Despliegue — importante

- **Python desktop**: corre en el host (PyQt6, fuera de Docker). El cambio está **vivo al relanzar** `cg001_desktop_3d.py`. Sin rebuild.
- **Web**: `viewer3d.js` está **horneado en la imagen Docker** (`Dockerfile: COPY CG001 /app/CG001`). El contenedor `cg001-lab` en marcha **sigue sirviendo la versión vieja**. El cambio en disco es la fuente de verdad, pero solo se verá tras `docker compose build` + recrear. **No se reconstruyó** para no interrumpir el experimento en curso — hazlo en tu próximo rebuild.

---

## 5. Pendiente (no incluido en este fix)

La **fuga de ε** sigue abierta: `universe.py:120` (gain global escalado por ε) y `:149-150` (subsidio per-step a id=0). Ver `INFORME_REVISION_CG001.md §3`. Eso es lo bloqueante para validar H1–H10; esto de la visualización es cosmético/observacional.

---

## 6. Adenda — Exportación CSV + relanzamiento limpio (2026-06-29)

A pedido del coordinador, se añadió al WebLive una **descarga CSV** de la serie de métricas (alineado con §111-112: datos crudos reproducibles; y Error 2: la evidencia está en las métricas).

**Cambios en `CG001/server/cg001_weblive.py`:**
- Nuevo endpoint **`GET /csv`** que aplana el JSONL persistido (`/data/cg001_<exp>_*.jsonl`) a CSV con `Content-Disposition: attachment`. Columnas: `experiment_id, t, N, epsilon, ts, IPD, IH, IN, IPA, ICG0, S_mean, S_max, delta_mean, H_delta`.
- Función `_history_csv()` + `import glob`.
- Botón **"⬇ CSV"** en la cabecera del tablero.

**Relanzamiento limpio:** se vació el volumen `cosmogenesis-cg001_cg001_lab_data` y se recreó `cg001-lab` → la sim arranca en t=0 y el CSV captura la corrida **desde el inicio, sin restos**.

**Nota:** el log se muestrea **cada 10 ticks** (tu diseño en `_maybe_log`), así que el CSV tiene filas en t=10, 20, 30… Si el equipo quiere granularidad por tick, hay que ajustar ese muestreo.

**Volúmenes huérfanos:** quedan `cosmogenesis-cg001_cg001_a_data` y `..._b_data` de la díada anterior (no usados por el compose actual). Se pueden borrar para limpiar del todo.
