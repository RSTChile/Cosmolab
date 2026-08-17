# INFORME — INTERFAZ WEB DE LA CÉLULA MADRE (Opción B)

**Para:** Equipo transinteligente RMD 2.0
**Autor del proyecto:** Alexis López Tapia
**Fecha:** 2026-06-22 · **Archivo:** `VST_CelulaMadre_Web.py`
**Marco:** Teoría Cosmosemiótica — Versión Canónica (17-06-2026)

> Guía práctica para correr la célula madre funcional desde el navegador, experimentar
> aislando organelos y exportar los registros. Verificado end-to-end.

---

## 1. Qué es

Un **servidor local en Python** (stdlib, sin dependencias nuevas) que envuelve la célula
madre funcional (`Célula_Madre_Funcional_001.py`) y la expone como interfaz web al estilo
de los experimentos cosmosemióticos (Levitron / EIT3 / Dron): tema oscuro, paneles,
**chart.js** en vivo, narrador y descarga de **CSV por paso**.

**Por qué backend Python (Opción B) y no JS puro:** la célula madre validada es
Python/numpy (campo Φ, FFT, organelos). El backend la reutiliza *tal cual*; reescribirla
en JS duplicaría y arriesgaría divergencia con el código probado. El precio: necesita un
servidor local corriendo (no es un HTML de doble clic).

---

## 2. Cómo correrla

```bash
cd /Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo
venv/bin/python3 VST_CelulaMadre_Web.py
# → abre en el navegador:  http://localhost:7777
# Ctrl+C para detener.
```

Requisitos: el `venv` del proyecto (numpy) y conexión a internet **solo** para cargar
`chart.js` por CDN (la interfaz; el cómputo es 100% local).

---

## 3. La interfaz

**Panel izquierdo — controles**
- **Audio:** menú con señales de prueba (tono 440 Hz · ruido rosa · clicks Poisson) o
  **subir un `.wav`** propio (se convierte a mono y 48 kHz).
- **Segundos de simulación** (1–20).
- **Interruptores por organelo** — un checkbox por módulo, agrupados por bloque
  (Cuerpo · Base · B5 Consciencia · B7 Libertad · B8 Evolución · Homeostasis). Botones
  rápidos *Todos* / *Solo soma*. El **Soma** (campo Φ que procesa el audio) está siempre
  ON: es el cuerpo.
- **▶ Procesar** y **⬇ CSV**.

**Panel derecho — resultados**
- **Resumen con checks:** OI → nivel, ✅ campo Φ finito, ✅ invariantes κ (x/6), Ω medio,
  Λ_Cos, C_m pico, organelos apagados.
- **Tres gráficos en vivo:** (a) Ω — estado representacional; (b) OI & Λ_Cos; (c) organelos
  en el tiempo (LF_op, XE, H, C_m, R₂).
- **Narrador** con el registro de la corrida.

---

## 4. El uso científico clave: ABLACIÓN por interruptor

El valor real de la interfaz es **aislar procesos**. Cada interruptor = el flag `expresar`
del organelo: apagarlo lo saca del ciclo metabólico — **aislamiento real, no visual**.

**Protocolo de ablación recomendado:**
1. Corre con **todos los organelos ON** → exporta CSV (línea base).
2. Apaga **un** organelo → corre con el mismo audio → exporta CSV.
3. Compara los dos CSV: la diferencia atribuible *solo* a ese organelo.

**Ejemplo verificado (apagar la consciencia):** silenciar `meta_representacion` (R₂) hace
que la libertad colapse — `LF_op → 0.0` — y el OI baja (0.537 → 0.434). Es la cadena
canónica en acción: sin R₂ no hay LF (O-N13.8). *La consciencia funda la libertad, y se
puede medir apagándola.*

Esto es exactamente la metodología que pide el `ADDENDUM`: resultados **atribuibles y
reproducibles**, no afirmaciones sin verificar.

---

## 5. Esquema del CSV (una fila por paso metabólico, 22 columnas)

| Columna | Significado |
|---|---|
| `t` | tiempo de vida del organismo (s) |
| `Omega` | Ω — estado representacional [0,1] (cartografía tipo V103) |
| `omega_A`, `omega_B` | ω de sistema A (audio) y B (referencia) |
| `gradiente` | ω_A − ω_B (desajuste audio↔expectativa; driver) |
| `e_R` | error operativo |
| `A_sys_env` | acoplamiento sistema-entorno |
| `presion_desacople` | arousal (presión de desacople) |
| `C_b` | consciencia básica = nº de distinciones registradas (|R₁|) |
| `R2` | meta-representación [0,1] |
| `LF_op`, `lf_nivel` | libertad funcional efectiva y nivel (0–3) |
| `juego`, `ritual`, `negacion` | 1/0 — estadios de la libertad activos |
| `demanda_entorno` | demanda del entorno (energía del audio) |
| `Omega_op` | dominio operativo (crece con la exaptación) |
| `XE` | exaptación acumulada [0,1] |
| `C_m` | consciencia metacognitiva |
| `H_homeostasis` | calidad de regulación homeostática [0,1] |
| `OI` | Índice de Organismicidad (O-N9.14) |
| `Lambda_Cos` | razón cosmosemiótica / salud del cierre (C-N2.8.12) |

---

## 6. Cómo interpretar los resultados

- **Ω (estado representacional):** la coordenada [0,1] que induce cada estímulo. *Hoy es
  una medida coarse: no separa timbres todavía* (ver caveat). La diferenciación entre audios
  aparece sobre todo en **OI/exaptación**, no en Ω.
- **OI → nivel:** <0.4 no organismal · 0.4–0.7 protoorganismo · ≥0.7 organismo pleno. Una
  célula individual llega a protoorganismo; el organismo pleno requiere la escala
  multicelular (ME = memoria externalizada compartida), aún no en esta interfaz.
- **Invariantes κ (x/6):** condiciones de viabilidad (persistencia, diferencia, error
  acotado, acoplamiento, libertad, analizabilidad). 6/6 = viable.
- **Organelos en el tiempo:** cómo reacciona cada módulo al audio. Ej.: un tono sostenido
  (mucha energía) dispara más **exaptación** (XE↑) que clicks dispersos.

**Experimentos sugeridos para el equipo:**
1. Mismo audio, *todos ON* vs *sin un módulo* → medir el efecto aislado (ablación).
2. Tres audios distintos, todos ON → comparar Ω, OI y qué organelos se activan (cartografía).
3. Subir voz vs ruido vs música → ver la respuesta evolutiva (exaptación) por tipo.

---

## 7. Caveats honestos (para no sobre-afirmar al mostrarlo)

1. **Ω coarse:** la medida actual de Ω no separa timbres (los demos dan Ω≈0.49). El
   refinamiento tipo V103 (lector de Ω más rico) está pendiente. La diferenciación real hoy
   está en OI/exaptación.
2. **`juego`/`ritual` pueden no dispararse:** sus umbrales (28/40) vienen de la escala de
   CM001 y no siempre casan con las señales derivadas del audio. Es **calibración pendiente**,
   no un bug.
3. **El Soma es un envoltorio GRUESO** del campo Φ probado (no la espina motora completa
   portada a organelos nativos). La descomposición fina (validada contra v1) es posterior.
4. **Célula individual:** ME (memoria externalizada) = 0 → el OI no llega a pleno. Eso
   requiere la capa multicelular (S_shared), que existe en `VST_Homeostasis.py` pero no en
   esta interfaz todavía.
5. **CDN:** la interfaz carga `chart.js` por internet; el cómputo es local.

---

## 8. Archivos relacionados

| Archivo | Rol |
|---|---|
| `VST_CelulaMadre_Web.py` | **esta interfaz** (servidor + frontend) |
| `Célula_Madre_Funcional_001.py` | la célula funcional que procesa audio (motor) |
| `VST_Genoma.py` + `VST_Bloque05/07/08` + `VST_Homeostasis.py` | genoma de organelos |
| `INFORME_CELULA_MADRE_Cosmosemiotica.md` | informe de arquitectura completa |
| `CelulaMadre_logs/` | salidas JSON de las corridas por CLI |

---

## 9. Cierre

La interfaz cumple los tres objetivos que la motivaron: **(1)** modificar organelos en
código (el motor es el Python validado), **(2)** separar la ejecución de cada parte
(interruptor por organelo = aislamiento real), **(3)** obtener resultados sin arriesgar el
organismo base (cada corrida es independiente; ablación por comparación de CSV). Es lo
primero **funcional, visual e interactivo** que el equipo puede correr, y la base para
experimentar — incluida, más adelante, la multicelularidad voluntaria (locus de Boorman).
