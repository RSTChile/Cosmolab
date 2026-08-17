# INFORME — CG001 v2: Instrumento de Campo Cosmosemiótico

**Autor:** Grok (sesión Casaubon) · **Para:** equipo Cosmogenesis / RMD  
**Fecha:** 2026-06-29  
**Objeto:** Implementación desde cero del experimento CG001 como **instrumento de campo** (no entidades), según la reformulación acordada con Claude (PDF *Reformulación Experimental Cosmogénesis*).  
**Estado:** Código completo, smoke-tests ejecutados. **Barridos de producción pendientes** (288 + 960 corridas).

---

## 0. Principio rector

> No ser Dios: no fijar singularidad ni resultados. S>0 es trascendental; la restricción determina qué persiste. La evidencia vive en **métricas**, no en visualización programada.

La v1 modelaba **entidades** con atributos {S, Δ, H}, movimiento en 3D, visor WebLive y Docker. Esa arquitectura arrastraba canales por los que la estructura podía estar **predeterminada** (geometría de grilla, deriva obligatoria, estelas, expansión radial, métricas colapsadas).

La v2 **descarta** ese paradigma y lo reemplaza por:

- **Un campo continuo** φ ≡ Ω_posible en una grilla 3D L×L×L.
- **Una dinámica** determinista; la probabilidad **emerge** como lectura a posteriori de lo que persistió.
- **Dos únicos inputs manuales:** amplitud de la singularidad (eje **RUIDO**, liso↔rugoso) y la arruga ε en el centro (solo condición B).
- **Capa termodinámica de medición** que lee exergía, entropía, ICES/IDES y concentración — **no modifica** la dinámica.

---

## 1. Qué se archivó (v1)

Todo el árbol de entidades quedó en `_archive_v1_entidades/`:

| Contenido archivado | Descripción |
|---|---|
| `CG001/` | Motor de entidades (`universe.py`, `entity.py`, `environment.py`), métricas, visualización 3D, servidor web |
| `docker/` | Imagen y compose del experimento v1 |
| `venv_viz/`, `tools/` | Entorno y utilidades de visualización |
| `INFORME_*.md` (v1) | Informes de reconstrucción, Docker, fix visualización |

La v1 **no se borró**; queda como referencia histórica y para comparar decisiones de diseño.

---

## 2. Arquitectura v2 (raíz `Cosmogenesis/`)

| Archivo | Rol |
|---|---|
| `cg001_field.py` | Motor: campo φ, asimetría local, memoria ambiental, relajación, métricas termodinámicas |
| `cg001_barrido_grueso.py` | Barrido eje RUIDO 1.0 → 0.02 (24 puntos × 6 semillas × A/B) |
| `cg001_barrido_fino.py` | Barrido cola lisa 0.02 → 0.001 (16 puntos × 30 semillas × A/B) |
| `cg001_test_localizacion.py` | Compuerta paso 2: discrimina estructura en arruga vs artefacto de ratio |
| `run_cg001.sh` | Atajos: `demo`, `grueso`, `fino`, `grueso-quick`, `fino-quick` |
| `requirements.txt` | `numpy`, `scipy` |
| `venv/` | Entorno local (creado) |
| `logs/` | Salidas JSON + CSV con timestamp |
| `.gitignore` | `venv/`, `logs/`, `__pycache__/` |

**Dependencias mínimas:** sin Docker, sin visor, sin PyQt. El experimento corre en terminal sobre iMac/LaCie.

---

## 3. Dinámica del campo (nodo por nodo)

### 3.1 Estado

- **φ** — campo escalar 3D (Ω_posible).
- **m** — memoria ambiental acumulada (historia de asimetrías visitadas).

### 3.2 Inicialización (singularidad)

```text
φ ← N(0, RUIDO)   en todo el volumen
si B: φ[L/2, L/2, L/2] += ε
```

- **RUIDO** = amplitud de la singularidad (eje experimental liso↔rugoso). No se elige un valor “correcto”: se **barre**.
- **ε** = única arruga inicial (#44, #133). Solo en condición B; A tiene ε=0.

### 3.3 Paso de dinámica (determinista)

Por cada celda, en cada paso:

1. **Asimetría local** (C-N4, relacional):  
   `a = φ − gaussian_filter(φ, σ)` — kernel isotrópico, sin ejes privilegiados.

2. **Medición termodinámica** (solo lectura):  
   - Exergía = Σ|a|  
   - Disipado = Σ(λ_eff · |a|) → acumula entropía  
   - Convertido (ICES) = Σ|a| en celdas de nicho (m > cuantil 0.999)

3. **Memoria ambiental** (#126):  
   `m ← decay·m + |a|`

4. **Nicho history-dependiente** (#131):  
   `λ_eff = λ / (1 + γ·m)` — el medio protege asimetrías donde hubo historia.

5. **Relajación** (#119, #128):  
   `φ ← φ − λ_eff · a`

**No hay entidades, no hay probabilidad impuesta, no hay movimiento en 3D.**

### 3.4 Parámetros por defecto

| Parámetro | Valor | Rol |
|---|---|---|
| L | 48 (demo/smoke), **64 (producción)** | Tamaño del volumen |
| pasos | 300 (demo), **400 (producción)** | Duración de la corrida |
| λ (lam) | 0.50 | Costo de sostener diferencia |
| σ (sigma) | 1.0 | Vecindad del kernel gaussiano |
| γ (gamma) | 8.0 | Protección de nicho vía memoria |
| decay | 0.97 | Decaimiento de memoria ambiental |
| ε (eps) | 0.05 | Arruga inicial (solo B) |
| RUIDO | 1.0 (demo), **barrido** en experimento | Amplitud singularidad |

---

## 4. Capa termodinámica (medición, no dinámica)

| Magnitud | Definición | Interpretación |
|---|---|---|
| **Exergía** | Σ\|a\| al paso t | Asimetría total disponible |
| **Entropía** | acumulado de disipado | Exergía convertida (irreversible) |
| **Convertido** | Σ\|a\| en nicho (m > Q₀.₉₉₉) | ICES — estructura que el medio “recuerda” |
| **Concentración** | max(m) / mean(m) | Observable principal del barrido A/B |
| **n_nicho** | celdas con m > umbral | Tamaño del nicho |

La **flecha termodinámica** emerge de partir en exergía máxima / entropía cero (campo rugoso) y relajar — no se impone por reloj externo.

---

## 5. Diseño experimental

### 5.1 Condiciones A y B

| Condición | ε en centro | Interpretación |
|---|---|---|
| **CG001-A** | 0 | Control: singularidad sin arruga |
| **CG001-B** | 0.05 | Tratamiento: singularidad + arruga |

Misma semilla, mismo RUIDO, misma grilla → la **única** diferencia es ε.

### 5.2 Eje singularidad (RUIDO)

El PDF y la sesión acordaron **no elegir** el valor de singularidad (“No sé” → barrido):

1. **Barrido grueso:** RUIDO ∈ [1.0, 0.02], 24 puntos geométricos, 6 semillas.
2. **Barrido fino:** RUIDO ∈ [0.02, 0.001], 16 puntos, 30 semillas — solo en la banda que el grueso señale.

### 5.3 Observable principal y criterio de operación

Para cada punto RUIDO y cada semilla se calcula la divergencia **B − A** en:

- **Concentración** (principal)
- Convertido
- Exergía restante

Se agrega por semillas con `signo_estable()`:

- **Media** de las divergencias.
- **Fracción de semillas** con el mismo signo que la media.

**Criterio “OPERA”:** `signo ≥ 0.83` (≥83% semillas coherentes, ref. #109) **y** `|media| > 10⁻³`.

Esto certifica que B y A divergen de forma **reproducible**, no por ruido de una semilla.

---

## 6. Resultados de smoke-tests (2026-06-29)

### 6.1 Demo (`./run_cg001.sh demo`)

Config: L=48, pasos=300, RUIDO=1.0, 1 semilla.

| Magnitud | A | B | Δ(B−A) |
|---|---|---|---|
| Exergía final | 27 636 | 27 636 | +0.05 |
| Entropía final | 59 899 | 59 899 | 0 |
| Convertido | 290.0 | 290.0 | 0 |
| Concentración | 12.44 | 12.44 | 0 |

**Lectura:** Con RUIDO=1.0 (campo muy rugoso), ε=0.05 en **una celda** de 48³ ≈ 110 000 celdas queda **ahogada**. A ≡ B. Coherente con el diseño: en régimen de ruido dominante la arruga no opera.

Flecha termodinámica visible: exergía cae ~67% (83 316 → 27 636), entropía sube a 59 899.

### 6.2 Barrido grueso — quick

**Comando:** `./run_cg001.sh grueso-quick`  
**Config:** L=48, pasos=300, 4 puntos RUIDO, 2 semillas → 16 corridas  
**Log:** `logs/barrido_grueso_20260629_100726/`

| RUIDO | Δ concentración (media) | signo | OPERA |
|---|---|---|---|
| 1.000 | ~0 | 0.50 | no |
| 0.271 | ~0 | 0.50 | no |
| **0.074** | **+0.0205** | **1.00** | **sí** |
| **0.020** | **+0.0048** | **1.00** | **sí** |

**Banda preliminar:** RUIDO ∈ [0.074, 0.020] — señal en concentración cuando el campo deja de estar dominado por ruido puro.

### 6.3 Barrido fino — quick

**Comando:** `./run_cg001.sh fino-quick`  
**Config:** L=48, pasos=**100** (reducido), 4 puntos, 3 semillas → 24 corridas  
**Log:** `logs/barrido_fino_20260629_100840/`

| RUIDO | Δ concentración (media) | signo | OPERA |
|---|---|---|---|
| 0.020 | +0.0002 | 0.33 | no |
| **0.007** | **+5.80** | **1.00** | **sí** |
| **0.003** | **+28.95** | **1.00** | **sí** |
| **0.001** | **+89.79** | **1.00** | **sí** |

**Banda preliminar (smoke):** RUIDO ∈ [0.007, 0.001].

**Advertencia (ampliada — Claude web, 29-jun):** El fino-quick usa pocos pasos y semillas, pero el problema principal no es solo eso. `concentracion = max(m)/mean(m)` en la cola lisa (RUIDO→0) tiene `mean(m)→0` en A y B; la razón **explota por construcción** aunque la estructura sea débil. Δ=+89 puede ser división por casi-cero, no “ε opera fortísimo”. Es el mismo riesgo de métrica colapsada que motivó el salto v1→v2, reentrando por el observable.

**Test de localización ejecutado** (`cg001_test_localizacion.py`, seed=1):

| RUIDO | pasos | Δ conc. | argmax B (dist centro) | max(m) B/A | sum(núcleo) B/A | Veredicto |
|---|---|---|---|---|---|---|
| 0.074 | 300 | +0.003 | lejos (21) | ≈1.0 | 0.98 | NO PASA — A≡B espacialmente |
| 0.007 | 300 | +1.47 | **centro (0)** | 1.4× | 0.80 | NO PASA — localizado pero señal débil |
| 0.001 | 300 | +22.8 | **centro (0)** | **7.8×** | 1.89 | NO PASA (umbral 2×; borderline) |
| 0.001 | 100 | +95.1 | **centro (0)** | **20×** | **2.11** | **PASA** — estructura real + ratio amplificado |

**Lectura:** En la cola lisa, concentración **sola** no basta. El test discrimina: a RUIDO=0.001 con 100 pasos hay memoria alta **en la arruga** (no difusa como A) *y* el pico absoluto de B supera a A — la señal es mixta (estructura real + amplificación del ratio). A RUIDO=0.074 el barrido grueso mostró Δconc positivo pero **sin localización** → probable ruido de ratio con fondo aún activo.

---

## 7. Qué NO está hecho

| Ítem | Detalle |
|---|---|
| Barrido grueso producción | 288 corridas: L=64, pasos=400, 24×6 semillas — **no ejecutado** |
| Barrido fino producción | 960 corridas: L=64, pasos=400, 16×30 semillas — **bloqueado hasta pasar localización** |
| Localización multi-semilla | Solo seed=1; falta certificar en banda con ≥6 semillas antes de 960 corridas |
| Visualización | No hay visor v2; métricas van a CSV/JSON |
| Calibración de ε | ε=0.05 fijo según protocolo; no se barre en esta fase |

---

## 8. Cómo ejecutar

```bash
cd /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis

# Entorno (si hace falta)
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt

# Demostración de forma (1 semilla, RUIDO=1.0)
./run_cg001.sh demo

# Smoke-tests (~minutos)
./run_cg001.sh grueso-quick
./run_cg001.sh fino-quick

# Producción (horas; iMac/LaCie)
./run_cg001.sh grueso --production    # 288 corridas
./run_cg001.sh fino                 # 960 corridas
```

Salidas en `logs/barrido_{grueso|fino}_YYYYMMDD_HHMMSS/resultado.{json,csv}`.

---

## 9. Compuerta experimental (gate) — orden corregido

Antes de cualquier afirmación sobre el efecto de ε:

1. **Demo coherente** — flecha termodinámica visible; A≡B con RUIDO=1.0 ✅  
2. **Localización en punto de cola lisa** — `./run_cg001.sh localizacion-compare` (o `--ruido X`); criterio fijado:
   - **PASA:** en B, `argmax(m)` dentro de radio 2 del centro **y** `sum(m núcleo)_B / sum(m núcleo)_A ≥ 2` **y** `max(m)_B / max(m)_A ≥ 1.5`
   - **ARTEFACTO:** Δconcentración grande pero perfiles espaciales A≈B (solo ratio)
   - Ejecutado seed=1 ⏳ multi-semilla pendiente  
3. **Barrido grueso producción** — solo si el observable de concentración está validado en al menos un punto de la banda ⏳  
4. **Barrido fino producción** (960 corridas) — solo si paso 2 certifica estructura localizada, no ratio vacío ⏳  

**Regla:** no gastar 960 corridas para certificar con signo 1.00 un número que sea división por casi-cero.

Comando: `./run_cg001.sh localizacion` · `./run_cg001.sh localizacion-compare`

---

## 10. Diferencias clave v1 → v2

| Aspecto | v1 (entidades) | v2 (campo) |
|---|---|---|
| Unidad básica | Entidad con {S, Δ, H} | Campo φ + memoria m |
| Espacio | Posiciones 3D + deriva | Grilla L³ fija, wrap |
| Singularidad | Entidad id=0 con S alto | Ruido gaussiano global (RUIDO) |
| Intervención ε | S inicial id=0 | +ε en celda central (B) |
| Selección | Muerte/persistencia de entidades | Relajación + nicho history-dependent |
| Evidencia | IPD, IH, N(t), visor 3D | Concentración, exergía, entropía, CSV |
| Riesgo de artefacto | Geometría, estelas, expansión radial | Minimizado: kernel isotrópico, sin visor |

---

## 11. Próximos pasos recomendados

1. **Ejecutar barrido grueso producción** (`./run_cg001.sh grueso --production`).
2. Si hay banda → **barrido fino producción** (`./run_cg001.sh fino`).
3. En el RUIDO óptimo de la banda: corridas con `retornar_campos=True` para mapa de m y test de localización.
4. Redactar informe de resultados con banda certificada, tablas completas y decisión sobre ε / pasos adicionales.

---

## 12. Referencias

- `Reformulación Experimental Cosmogénesis: Creo que lo relevante, más….pdf` — diseño acordado  
- `_archive_v1_entidades/INFORME_RECONSTRUCCION_CG001.md` — último estado v1 antes del salto  
- Código fuente: `cg001_field.py` (docstring con nodos C-N2.x, #119, #126, #131, #133)

---

*Documento de implementación — no es informe de resultados experimentales. Los smoke-tests orientan; la certificación requiere barridos de producción.*