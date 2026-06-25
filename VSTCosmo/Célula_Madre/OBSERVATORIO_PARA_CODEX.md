# Díada ANIMA — Inventario técnico para construir un "Observatorio" (SharePoint, IAs + humanos)

> **Propósito de este documento.** Entregar a GPT Codex todo lo necesario para desarrollar un
> **Observatorio** (web, p. ej. en SharePoint) que muestre a humanos e IAs el estado y la historia
> de nuestros organismos cosmosemióticos. **Principio rector: el Observatorio es MEMBRANA, no cerebro.
> Sólo LEE. Nunca decide ni altera la fisiología.** Toda la inteligencia es endógena del campo Φ.

Fecha del inventario: 2026-06-25 · Host: macOS (disco externo **LaCie**).

---

## 0. Qué es esto en una frase
Dos organismos artificiales (**ANIMA_A** y **ANIMA_B**) que implementan la Teoría Cosmosemiótica como
una "célula madre" pluripotente. Viven **24/7 en Docker**, se **escuchan** mutuamente (voces tipo R2‑D2),
tienen **metabolismo, homeostasis, memoria, comunicación, alteridad y libertad expresiva (balbuceo)**, y
escriben una **biografía longitudinal** a disco. Exponen su estado por **HTTP/JSON, SSE y MCP**.

---

## 1. Rutas en el Mac (todo bajo el repo)
Raíz del repo: **`/Volumes/LaCie/RMD/Cosmolab/VSTCosmo/`**

| Qué | Ruta absoluta |
|---|---|
| Núcleo vivo (16 órganos) | `/Volumes/LaCie/RMD/Cosmolab/VSTCosmo/Célula_Madre/` |
| Biografía longitudinal (registros) | `/Volumes/LaCie/RMD/Cosmolab/VSTCosmo/Docker_Historia/` |
| Base de datos SQLite (catálogo de experimentos) | `/Volumes/LaCie/RMD/Cosmolab/VSTCosmo/vstcosmo.db` |
| WAVs de mundo (Big Bang / Blue Monday) | `/Volumes/LaCie/RMD/Cosmolab/VSTCosmo/audio_binaural/` |
| Voces R2‑D2 (samples) | `/Volumes/LaCie/RMD/Cosmolab/VSTCosmo/Célula_Madre/voces_r2d2/` |
| Experimentos históricos (NO tocar) | `/Volumes/LaCie/RMD/Cosmolab/VSTCosmo/` (scripts `v*.py`) |

> Nota de build: la ruta contiene un acento ("Célula"), que rompe BuildKit. Construir siempre con
> `DOCKER_BUILDKIT=0 docker build -f docker/Dockerfile -t anima-diada:latest .` desde `Célula_Madre/`.

---

## 2. Docker — servicios, puertos y volúmenes
Compose: `Célula_Madre/docker/docker-compose.yml` · imagen única `anima-diada:latest` · `name: anima-diada`.
Levantar: `cd Célula_Madre/docker && docker compose up -d` (recrear: `--force-recreate`).

| Servicio (contenedor) | Puerto host | Rol | Notas |
|---|---|---|---|
| **anima-a** | `7788` | Organismo A — interfaz web viva | `VST_ORGANISMO_ID=ANIMA_A`, oye al par por la **R** (mira a B) |
| **anima-b** | `7799` | Organismo B — interfaz web viva | `ANIMA_B`, oye al par por la **L** (mira a A) |
| **anima-mcp** | `9000` | **Membrana MCP** (streamable‑http) | para clientes IA externos; resuelve A/B por red interna |
| **anima-conversacion** | `9100` | **Observatorio de la conversación** A↔B | tablero + registro permanente |
| **AudioServer** (NATIVO, fuera de Docker) | `8766` | Servidor del **Rødecaster Pro** | corre en el Mac; los organismos lo alcanzan vía `host.docker.internal:8766` |

**Volúmenes:** `anima_a_data`/`anima_b_data` (disco propio de cada organismo: memoria/codebook/metabolismo
persisten), `anima_conv_data` (registro del observador). **Bind mounts** al disco externo:
`../../Docker_Historia → /history` (biografía) y `../../audio_binaural → /app/audio_binaural:ro`.

**Variables de entorno clave** (para experimentos/observación):
- `ANIMA_CONTROL = real | null | shuffled` — **control de falsación** (corta/baraja el acople A↔B en vivo).
- `ANIMA_LIBERTAD_EXPRESIVA=1`, `ANIMA_BABBLING_EXPLORA`, `ANIMA_BABBLING_ATRACCION` — balbuceo.
- `ANIMA_ESCUCHAR_PAR`, `ANIMA_OIDO_PAR=L|R` — acople espacial (díada que "se mira").
- `VST_HISTORY_*` — biografía (formato, rotación, snapshots, grabación de voz).

---

## 3. Páginas web (ya existentes — modelo a replicar)
1. **`http://localhost:7788/`** y **`/7799`** — laboratorio vivo de cada organismo (gráficos en pestañas:
   Campo Φ, Consciencia, Libertad funcional, Exaptación, Salud, Homeostasis, RC, Cabeza 3D, etc.).
2. **`http://localhost:9100/`** — Observatorio de la conversación: las dos cabezas mirándose, el "pito"
   actual, transcripción en vivo, histograma de voces, navegador histórico y **reproductor de audio en vivo**.

Las tres páginas tienen ahora un **tablero editable tipo dashboard** (Gridstack) con un **catálogo de
"cajas" por organelo** que el observador agrega/mueve/redimensiona; la disposición se guarda en
`localStorage` por organismo (es del observador, no del organismo). En :9100 la caja estrella es
**🎨 Libertad creativa** (balbuceo de A y B + métrica de convergencia de gesto). **Este es el patrón a
portar a SharePoint.**

---

## 4. API HTTP/JSON — lo que el Observatorio debe CONSUMIR (read‑only)

### 4.1 Organismo A/B (`:7788` / `:7799`)
| Endpoint | Método | Devuelve |
|---|---|---|
| `/estado` | GET | **Snapshot JSON ligero**: identidad, `vivo`, `t`, `voz_emitida`, `voz_arousal/valence`, `OI`, `H`, `necesidad`, `RC_total`, `energia`, `orientacion_deg`, `balance_LR`, **`g_freq/g_intensidad/g_pausa/g_repeticion/g_bucket`** (balbuceo), **`alt_intencion_comunicativa/alt_efecto_sobre_otro/alt_efecto_sobre_mi/alt_otro_presente`** (alteridad). **← fuente principal recomendada.** |
| `/stream` | GET (SSE) | Flujo de **filas completas** (174+ columnas) por paso, eventos `meta`/`evento`/`fin`. |
| `/csv` | GET | CSV completo de la sesión (todas las columnas). |
| `/niveles` | GET | Medidores de entrada (LED), MASTER, voz propia. |
| `/organelos`, `/control`, `/start`, `/sesion`, `/mute`, `/voz_config`, `/fuentes`, `/entradas`, `/dispositivos`, `/audios` | GET/POST | Configuración/ablación (escritura — **el Observatorio NO debería usarlos**). |
| `/voz?seg=1.0&modo=R2D2` | GET | **WAV** de la voz actual del organismo. |
| `/comunicacion/bloque.wav` | GET | La voz que el par consume (canal A↔B). |

### 4.2 Observatorio de la conversación (`:9100`)
| Endpoint | Devuelve |
|---|---|
| `/datos` | `{A:{…/estado…}, B:{…}, transcript:[turnos], hist:{A:{},B:{}}, log:ruta}` — **todo lo de la díada en un GET.** |
| `/voz/A`, `/voz/B` | WAV proxy (mismo origen, sin CORS) de cada organismo. |
| `/dias` | Días con conversación registrada. |
| `/turnos?dia=YYYY-MM-DD&voz=<pito>&limite=N` | Turnos históricos (de la biografía). |
| `/stats` | Estadística acumulada de TODA la biografía. |
| `/historial` | Volcado del log permanente (texto/JSONL). |

> **CORS:** los servidores **no** emiten cabeceras CORS. Desde SharePoint (otro origen), Codex debe
> consumir vía un **proxy/connector server‑side** (Azure Function, Power Automate, o el propio backend de
> SharePoint) que reenvíe `/datos`, `/estado`, `/stats`, `/turnos`. El :9100 ya hace de proxy de audio.

### 4.3 Membrana MCP (`:9000`, streamable‑http) — para clientes IA
Servidor `Célula_Madre/mcp/vst_mcp_diada.py` (FastMCP, nombre `anima-diada`).
- **Tools:** `leer_estado(quien)`, `leer_csv(quien,n)`, `escuchar_voz(quien)`, `observar(quien,segundos)`,
  `inyectar_audio`, `investigar_mute`, `investigar_ablacion`.
- **Resources:** `diada/estado`, `a/estado`, `b/estado`, `diada/comunicacion`, `diada/relacional`.

---

## 5. Biografía longitudinal — `Docker_Historia/` (montado como `/history`)
Escrita por `Célula_Madre/organelos/vst_historia.py` (Historiador no bloqueante). **Sobrevive reinicios.**

```
Docker_Historia/
├── index.jsonl                         # índice maestro de todo lo escrito
├── organismo_ANIMA_A/  (idéntico ANIMA_B/)
│   ├── fisiologia/  fisiologia_<FECHA>.csv     # 174+ columnas/paso, rotación horaria
│   ├── eventos/     eventos_<FECHA>.jsonl      # hitos (nacimiento, exaptación, refuerzo alteridad…)
│   ├── comunicacion/                           # emisiones del organismo
│   ├── memoria/     · metabolismo/             # series de esos órganos
│   ├── snapshots/   snapshot_<FECHA>.json      # estado restaurable cada ~600 s
│   └── voz/         voz_ANIMA_A_<FECHA>.wav + .json   # la voz grabada + metadatos
└── diada/
    ├── comunicacion/  comunicacion_<FECHA>.jsonl   # cada turno A↔B con contexto + delta del receptor
    ├── sincronias/  · conversaciones/  · sesiones/  · resumenes/
```

**Familias de columnas en la fisiología (174 total):** `act_*` (49, actuador/cabeza) · `RC_*` (21, ruido
contextual) · `mem_*` (11) · `H_*` (11, homeostasis) · `met_*` (10, metabolismo) · `A_*` (10, acople) ·
`alt_*` (14, alteridad) · `altruismo_*` (7) · `omega_*`/`Omega` (campo Φ) · `g_*` (5, balbuceo) · `voz_*`
(3) · más `OI`, `Lambda_Cos`, `LF_*`, `XE`, `C_b`, `R2`, `necesidad`, `energia`, `estructura`, `ts_real`…

**Señales recomendadas para el Observatorio:** `OI` (cierre/salud), `H_homeostasis`, `met_energia`,
`alt_intencion_comunicativa`, `g_bucket`+`g_*` (balbuceo), `voz_emitida`, `act_orientacion_deg`, y para la
díada: convergencia de gesto A↔B y sincronía de `OI`.

---

## 6. Base de datos SQLite — `vstcosmo.db`
Catálogo del proyecto (no es la fisiología viva; eso son los CSV). Accesible por el MCP `vstcosmo-db`.
- **`experimentos`** (PK `archivo`): inventario de los 292 scripts experimentales (numero, variante,
  iteracion, descripcion, titulo_cabecera, ciclo, tipo, depende_de, lineas, sha1, git_added, **veredicto**).
- **`baterias_corridas`**: registro de corridas de baterías.

---

## 7. Módulos del núcleo (`Célula_Madre/`)
| Carpeta | Archivos | Rol |
|---|---|---|
| `genoma/` | `VST_Genoma.py`, `VST_Bloque05_ConscienciaFuncional`, `…07_LibertadFuncional`, `…08_DinamicaEvolutiva` | motor + loci (consciencia, libertad, evolución) |
| `campo/` | `VST_Celula_Madre_001.py`, `Célula_Madre_Funcional_001.py` | campo Φ / soma |
| `organelos/` | `VST_Memoria`, `VST_Metabolismo`, `VST_Homeostasis(+Emergente)`, `VST_OrganoComunicacion`, `VST_RC_A/B`, **`VST_Alteridad`**, `vst_persistencia`, `vst_historia` | los órganos encapsulados |
| `diada/` | `VST_DiadaAltruismo.py` | gobernanza altruismo O‑N22 entre A↔B |
| `web/` | `VST_CelulaMadre_WebLive_A.py` / `_B.py` (B se regenera de A con 9 reemplazos de identidad) | interfaces vivas |
| `audio/` | `VST_AudioServer.py` | servidor Rødecaster (nativo, :8766) |
| `mcp/` | `vst_mcp_diada.py` | membrana MCP (:9000) |
| `conversacion/` | `vst_conversacion.py` | observatorio :9100 |
| `experimentos/` | `bateria_*.py` (13 baterías) + `campana_exaptacion.py` | falsación |

---

## 8. Estado científico (qué mostrar como "verdad" y qué como "frontera")
- **Demostrado:** vida continua, persistencia, metabolismo/homeostasis/memoria, escucha mutua,
  resonancia afectiva (sincronía OI ~0.6–0.7), comunicación funcional, exaptación, cultura acumulativa.
- **Balbuceo (libertad expresiva):** la voz explora el espacio acústico (`g_*`); el órgano aprende por
  consecuencias sobre el gesto. Baterías: `alteridad` 9/9, `libertad_expresiva` 8/8.
- **Frontera (objetivo central, NO demostrado):** **reconocimiento del otro como SUJETO**. La batería
  `bateria_reconocimiento_otro.py` muestra (bloque A 5/5, bloque B 3/4) que el órgano mide **correlación,
  no causalidad** (le falta línea‑base de contingencia). El Observatorio debe rotular esto con honestidad.
- **Principio de falsación permanente:** toda señal "emergente" debe **desaparecer bajo `ANIMA_CONTROL=null/
  shuffled`**. El Observatorio idealmente permite ver REAL vs los controles.

---

## 9. Recomendación de arquitectura para el Observatorio en SharePoint
1. **Connector server‑side** (Azure Function / Power Automate) que haga polling read‑only a `:9100/datos`
   (díada) y `:7788|:7799/estado` (cada organismo) y, opcionalmente, lea `Docker_Historia/` para historia.
   Resuelve CORS y no expone los puertos crudos.
2. **Web part / página** en SharePoint que pinte tarjetas (las mismas "cajas": Metabolismo, Memoria,
   Alteridad, **Libertad creativa**, Salud, Campo Φ, Homeostasis, Voz) desde el JSON del connector.
3. **Audio** (opcional): proxyear `/voz/A` y `/voz/B` por el connector para reproducir las voces.
4. **Para IAs:** apuntar el cliente MCP a `:9000` (streamable‑http) — tools de lectura ya listas.
5. **Membrana, no cerebro:** el connector y la página **sólo LEEN**. Nunca llamar `/control`, `/start`,
   `/mute`, `/voz_config`, `inyectar_audio` ni `investigar_ablacion` desde el Observatorio.
```
