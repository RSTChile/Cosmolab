# INFORME — IMPLEMENTACIÓN EJECUTABLE CG001 (Docker + WebLive 3D)

**Para:** Revisor (Claude) — Club Abulafia / equipo transinteligente  
**Solicitante:** Alexis López Tapia  
**Ejecutor técnico:** Grok (agente Cursor) — custodia + implementación infraestructural  
**Fecha:** 2026-06-29  
**Repositorio:** `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/`  
**Protocolo de referencia:** `PROTOCOLO EXPERIMENTAL COSMOGÉNESIS.pdf` (serie CG001, 44 págs.)

> Este informe documenta la **primera implementación ejecutable** de CG001: núcleo de simulación,
> servidor HTTP en vivo análogo a ANIMA, despliegue Docker 24/7, y visualización 3D en navegador.
> **No sustituye** el diseño experimental de Alexis; es infraestructura para observar y auditar corridas.

---

## 1. Mandato y alcance

### 1.1 Petición del usuario

1. Montar CosmoGénesis CG001 en **Docker como servidor**, análogo a la díada ANIMA (`Célula_Madre/docker/`).
2. Añadir **visualización 3D en tiempo real** mientras corren los experimentos (WebGL/Three.js en navegador).

### 1.2 Qué NO está en alcance (explícito)

| Elemento del protocolo | Estado |
|---|---|
| VisPy + PyQt6 (Parte IX §145) | **No implementado** — sustituido por Three.js en browser |
| 100.000 entidades (§6, §160 final) | **No** — primera corrida con **1.000** (§160 recomendación inicial) |
| Grilla 256³ / 512³ (§49) | **No** — grilla **64³** (capacidad Docker/iMac) |
| Unity / Godot (§11) | **No** |
| Clasificación automática de exaptación (§18.3) | **No** — solo métricas base |
| Modo auditoría por semilla reproducible UI (§157) | **Parcial** — `reset` con semilla vía API; sin UI dedicada |
| Tabla numérica canónica de κ, α, β (pendiente en revisión protocolo) | **Aproximación operativa** en YAML |

### 1.3 Rol del ejecutor

Conforme a `MEMORY.md` / `CLAUDE.md`: el agente actúa como **custodio + ejecutor de infraestructura**, no como autor del experimento. La lógica de reglas es una **traducción mínima** del protocolo para tener un laboratorio observable; Alexis y el equipo transinteligente definen la evolución del núcleo.

---

## 2. Resumen ejecutivo

Se creó desde cero el proyecto ejecutable `Cosmogenesis/CG001/` (antes solo existía el PDF del protocolo) y se desplegó en Docker con **tres servicios**:

| Servicio | Rol experimental | Puerto | URL |
|---|---|---|---|
| `cg001-a` | CG001-A — universo simétrico **ε=0** | 7888 | http://localhost:7888 |
| `cg001-b` | CG001-B — universo asimétrico **ε>0** | 7889 | http://localhost:7889 |
| `cg001-observatorio` | Comparación A vs B (métricas + 3D dual) | 7900 | http://localhost:7900 |

**Verificación (2026-06-29):** `docker compose build` OK · tres contenedores `running` · `/estado` responde · `/entidades` devuelve posiciones · `/static/viewer3d.js` servido · observatorio proxy A/B OK.

**Comando de arranque:**

```bash
cd /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/docker
docker compose up --build -d
```

---

## 3. Analogía con ANIMA (Célula Madre)

| Patrón ANIMA | Equivalente CG001 |
|---|---|
| Imagen única `anima-diada:latest` | `cosmogenesis-cg001:latest` |
| `ANIMA_ROLE` → a/b/c/d/mcp/conversacion | `CG_ROLE` → a / b / observatorio |
| `entrypoint.sh` + watchdog `/estado` | `docker/entrypoint.sh` + watchdog idéntico |
| `VST_PUERTO` + healthcheck HTTP | `CG_PUERTO` + healthcheck `/estado` |
| WebLive SSE `/stream` | SSE `/stream` + polling `/entidades` para 3D |
| `anima-conversacion` observa díada | `cg001-observatorio` compara ε=0 vs ε>0 |
| Volúmenes `/data` por organismo | Volúmenes `cg001_a_data`, `cg001_b_data` |
| `restart: unless-stopped` | Igual |

**Diferencia deliberada:** ANIMA consume audio del Mac vía `host.docker.internal`; CG001 no tiene entrada sensorial externa — el universo es autocontenido.

---

## 4. Arquitectura de software

```
Cosmogenesis/
├── PROTOCOLO EXPERIMENTAL COSMOGÉNESIS.pdf
├── requirements.txt                    # numpy, PyYAML
├── .dockerignore
├── INFORME_CG001_Implementacion_Docker_3D.md   # este archivo
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
└── CG001/
    ├── config/CG001_default.yaml
    ├── core/
    │   ├── entity.py          # E₀: S, Δ_struct, H, posición
    │   ├── environment.py     # Env(x,y,z) — historia ambiental
    │   └── universe.py        # motor de pasos + snapshot
    ├── metrics/
    │   └── persistence.py     # IPD, IH, IN, IPA, ICG₀
    ├── server/
    │   ├── cg001_weblive.py   # HTTP + dashboard 3D
    │   └── static/viewer3d.js # visor Three.js reutilizable
    └── observatorio/
        └── cg001_observatorio.py
```

---

## 5. Núcleo matemático (traducción operativa)

### 5.1 Entidad `E₀` (`entity.py`)

Variables por entidad (protocolo §48):

- `S` — persistencia / viabilidad
- `delta_struct` — diferencia estructural (Δ_struct)
- `H` — historia acumulada (solo si |Δ| ≥ κ_H)
- `pos` — (x, y, z) en grilla discreta
- `alive`, `t_hist`, `lineage`

**No hay tipos ni clases especiales** — coherente con §148.

### 5.2 Entorno (`environment.py`)

- Matriz 3D `history` + `stability` (float32)
- `deposit()` — toda interacción deja huella (§53, §54)
- `update_niches()` — nicho si `H_entorno > H_crit` (§55)
- `gradient_bias()` — constricción C-N2.6 (sesgo de movimiento)

### 5.3 Universo (`universe.py`) — reglas por paso

1. **Expansión** `R(t)` — separación experimental (§51), escala posiciones levemente.
2. **Movimiento** — random walk sesgado por gradiente del campo.
3. **Interacciones** si `d < r_int` (§52):
   - **Tipo I Refuerzo** — Δ y S suben si estructuras compatibles
   - **Tipo II Intercambio** — redistribución de Δ
   - **Tipo III Cancelación** — pérdida de Δ
4. **R₀** — cada paso consume `persist_cost` de S (§37).
5. **Muerte** — si `S ≤ κ_P`, entidad colapsa; huella va al entorno (§53).
6. **Fusión** (Regla 3, §52) — cada 50 pasos, par compatible puede fusionarse.
7. **Asimetría primordial** — entidad `id=0` recibe `S₀ + ε` en CG001-B (§44, §133).

### 5.4 Parámetros por defecto (`CG001_default.yaml`)

| Parámetro | Valor | Nota protocolo |
|---|---|---|
| `n_entities` | 1000 | §160 primera corrida |
| `grid_size` | 64 | §49 reducido |
| `s0` | 1.0 | persistencia base |
| `epsilon` (B) | 0.00001 | §44 “0.001%” aprox. vía env |
| `kappa_p` | 0.01 | κ_P |
| `kappa_h` | 0.05 | κ_H |
| `persist_cost` | 0.00012 | R₀ |
| `niche_h_crit` | 0.5 | H_crit nichos |
| `seed` | 42 | reproducibilidad |
| `tick_hz` | 10 | 10 pasos simulación / segundo |

### 5.5 Métricas (`persistence.py`)

| Métrica | Definición implementada | Protocolo |
|---|---|---|
| **IPD** | S_max / S_mean | §57, §135 |
| **IH** | Σ H_i + historia ambiental | §136 |
| **IN** | Clusters espaciales (greedy, eps=3, min=5) | §137 (aprox.) |
| **IPA** | media de `stability` ambiental | §138 (aprox.) |
| **ICG₀** | máximo de `history` ambiental | placeholder semántico |
| O1–O7 | N, S̄, Δ, H_Δ, nichos, H_total, S_max | §56 |

---

## 6. Servidor WebLive (`cg001_weblive.py`)

### 6.1 Endpoints

| Método | Ruta | Función |
|---|---|---|
| GET | `/` | Dashboard HTML con viewport 3D |
| GET | `/estado` | Healthcheck + métricas breves (Docker) |
| GET | `/metricas` | Snapshot completo |
| GET | `/entidades?limit=N` | Muestra de entidades vivas (máx. 1200) + `meta` |
| GET | `/stream` | SSE — un snapshot por tick de simulación |
| GET | `/static/viewer3d.js` | Módulo ES del visor 3D |
| POST | `/control` | `{action: pause\|resume\|reset, seed?: int}` |

**CORS:** `Access-Control-Allow-Origin: *` en respuestas (para consumo cruzado si se requiere).

### 6.2 Variables de entorno

| Variable | Default | Uso |
|---|---|---|
| `CG_PUERTO` | 7888 | Puerto HTTP |
| `CG_EPSILON` | 0 | Asimetría primordial |
| `CG_EXPERIMENT_ID` | CG001-A/B | Etiqueta experimental |
| `CG_SEED` | 42 | Semilla |
| `CG_AUTOSTART` | 1 | Arranque automático simulación |
| `CG_TICK_HZ` | 10 | Hz del loop |
| `CG_VISUAL_LIMIT` | 800 | Entidades enviadas al visor 3D |
| `CG_HISTORY_DIR` | /data | JSONL diario (cada 10 pasos) |
| `CG_HISTORY_ENABLE` | true | Activar log |

### 6.3 Persistencia

Archivo JSONL en volumen Docker: `/data/cg001_{CG001-A|CG001-B}_{YYYY-MM-DD}.jsonl`  
Campos: `t`, `N`, `metrics`, `epsilon`, `ts`.

---

## 7. Visualización 3D (`viewer3d.js`)

### 7.1 Tecnología

- **Three.js 0.160** vía CDN (import map)
- **OrbitControls** — rotación / zoom
- **Points + ShaderMaterial** — hasta ~800 entidades con buen rendimiento
- **LineBasicMaterial** — estelas de trayectoria

### 7.2 Codificación visual (protocolo §152)

| Canal visual | Variable |
|---|---|
| Tamaño del punto | S (persistencia) |
| Color (HSL) | H (historia) |
| Luminosidad / saturación | Δ_struct |
| Estela dorada | entidad id=0 (ε) |
| Caja + GridHelper | volumen 64³ |

### 7.3 Controles UI (WebLive)

- **Seguir ε (id=0)** — cámara sigue entidad primordial (§154 “Diferencia Primordial”)
- **Estelas ON/OFF** — trazas históricas
- Pausa / Reanudar / Reset

### 7.4 Actualización en vivo

- HUD (métricas, eventos): **SSE** `/stream`
- Posiciones 3D: **polling** `/entidades?limit=800` cada ~350 ms

### 7.5 Observatorio 3D (`cg001_observatorio.py`)

- Dos viewports lado a lado (modo comparación §158)
- Proxy `/proxy/a/entidades` y `/proxy/b/entidades` — evita CORS entre puertos
- B sigue automáticamente entidad ε
- Panel divergencia: ΔIPD, ΔIH, ΔN

---

## 8. Docker

### 8.1 Imagen

- Base: `python:3.12-slim`
- Deps: `numpy==2.2.6`, `PyYAML==6.0.2`
- `PYTHONPATH=/app`
- Healthcheck: GET `http://127.0.0.1:{CG_PUERTO}/estado`

### 8.2 Compose (`name: cosmogenesis-cg001`)

- A y B: misma semilla (42), mismo todo excepto ε
- Observatorio: `depends_on: service_healthy` en A y B
- Volúmenes nombrados para historia por universo

### 8.3 Watchdog (`entrypoint.sh`)

Réplica del patrón ANIMA: si `/estado` falla 3 veces (~60 s), mata y relanza el proceso Python.

---

## 9. Control experimental CG001-A vs CG001-B

| | CG001-A | CG001-B |
|---|---|---|
| ε | 0 | 0.00001 |
| Semilla | 42 | 42 |
| Entidades | 1000 | 1000 |
| Reglas | idénticas | idénticas |
| Única diferencia | entidad 0 sin bonus ε | entidad 0: S = S₀ + ε + bonus dinámico |

**Predicción cosmosemiótica a verificar (§60):** B debe mostrar mayor IPD, IH, nichos y persistencia diferencial que A. El observatorio expone ΔIPD, ΔIH, ΔN en tiempo real.

---

## 10. Inventario de archivos creados

| Archivo | Líneas aprox. | Función |
|---|---|---|
| `CG001/core/entity.py` | ~45 | Entidad E₀ |
| `CG001/core/environment.py` | ~65 | Campo ambiental |
| `CG001/core/universe.py` | ~210 | Motor simulación |
| `CG001/metrics/persistence.py` | ~75 | Métricas |
| `CG001/server/cg001_weblive.py` | ~310 | Servidor + HTML |
| `CG001/server/static/viewer3d.js` | ~175 | Visor 3D |
| `CG001/observatorio/cg001_observatorio.py` | ~175 | Comparador |
| `CG001/config/CG001_default.yaml` | ~27 | Parámetros |
| `docker/Dockerfile` | ~35 | Imagen |
| `docker/docker-compose.yml` | ~79 | Orquestación |
| `docker/entrypoint.sh` | ~55 | Roles + watchdog |
| `requirements.txt` | 2 | Dependencias |

**Total:** ~20 archivos nuevos bajo `Cosmogenesis/` (excl. `__pycache__`).

---

## 11. Criterios de falsación — estado actual

| Criterio | ¿Medible hoy? | Nota revisor |
|---|---|---|
| F1 Estructuras persistentes | Parcial | N(t), S_max, IPD en dashboard |
| F2 Atractores | **Débil** | No hay detector formal de atractores (§16.4) |
| F3 Reducción Ω por historia | **No** | No se calcula Ω_op |
| F4 Recombinación | Parcial | Fusión cada 50 pasos; no clasificada |
| F5 Nuevas unidades persistentes | Parcial | Fusión reduce N; no “nacimiento” explicítico |
| Éxito §23 (8 condiciones) | **No automatizado** | Requiere análisis offline multi-semilla |

---

## 12. Limitaciones conocidas (para revisión crítica)

1. **Simplificación fuerte del núcleo** — no implementa I·E acoplado (C-N2), ni t_hist ≠ t_sim con criterio riguroso, ni LF/Ω_op.
2. **IN (nichos)** — clustering espacial sobre entidades vivas, no sobre celdas `Env` con H > H_crit de forma separada.
3. **Muestreo visual** — el visor muestra hasta 800 de N entidades (submuestreo aleatorio); puede ocultar estructuras raras.
4. **Three.js por CDN** — requiere internet en el navegador; el cómputo sí es local/Docker.
5. **Sin tests unitarios** — verificación manual vía curl + inspección visual.
6. **Escalado** — 64³ / 1000 entidades; subir a 10k–100k requiere optimización (NumPy vectorizado, instancing, posiblemente C++ / Numba).
7. **Autoría experimental** — parámetros numéricos (persist_cost, r_int, ganancias de interacción) son **heurísticas iniciales**, no tabla canónica del PDF.

---

## 13. Checklist de verificación para el revisor

```bash
# 1. Contenedores
cd /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/docker && docker compose ps

# 2. Salud
curl -s http://localhost:7888/estado | python3 -m json.tool
curl -s http://localhost:7889/estado | python3 -m json.tool

# 3. Datos 3D
curl -s "http://localhost:7888/entidades?limit=5" | python3 -m json.tool

# 4. Estático
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:7888/static/viewer3d.js

# 5. Observatorio
curl -s http://localhost:7900/comparacion | python3 -m json.tool

# 6. UI
# Abrir http://localhost:7900 — dos cubos 3D, métricas, divergencia
```

**Criterio mínimo de aceptación infraestructural:** los seis pasos anteriores OK + simulación avanza (`t_sim` crece) con A y B en paralelo.

---

## 14. Próximos pasos sugeridos (decisión de Alexis / equipo)

1. **Validar o corregir parámetros** del YAML contra tabla numérica que falta en protocolo.
2. **Detector de atractores** formal (§16.4): persistencia espacial + acoplamiento mutuo.
3. **Ω_op y LF** — métricas para criterios de exaptación (§18.3).
4. **Batería multi-semilla** (§1577 menciona 1000 semillas) — script batch + informe estadístico.
5. **Escalar entidades** 10k → 100k con profiling.
6. **VisPy/PyQt6 opcional** — cliente desktop si se quiere fidelidad total a Parte IX.
7. **Integración cronológica** — entrada en `INFORME_CONSOLIDADO` y `MEMORY.md` del programa VSTCosmo.

---

## 15. Trazabilidad de decisiones en sesión

| Decisión | Fuente |
|---|---|
| Docker análogo ANIMA | Petición Alexis |
| 1000 entidades, no 100k | Protocolo §160 + pragmatismo |
| ε_B = 0.00001 | Protocolo §44 (perturbación mínima) |
| WebGL browser vs VisPy | Petición Alexis (ver 3D mientras corre) |
| Abulafia/Evolución/Genética NO movidos a VSTCosmo | Corrección Alexis (sesión previa) |

---

*Fin del informe. El revisor debe tratar este documento como descripción de **infraestructura experimental v0.1**, no como corroboración de la hipótesis cosmosemiótica H1–H10.*