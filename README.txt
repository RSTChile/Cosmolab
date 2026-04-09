# EIT-3 Cosmosemiotic Audio Processor
## RMD 2.0 / Cosmosemiótica Canónica

Procesador de audio que implementa el módulo analógico EIT-3 Lite
en DSP digital. Voz mejorada por el contexto ambiental — no limpieza,
sino exaptación estructural.

---

## Requisitos

- macOS con Python 3 instalado (ya viene en Mac)
- numpy y scipy

Instalar dependencias (una sola vez):
```bash
pip3 install numpy scipy
```

---

## Uso

**1. Ejecutar el servidor:**
```bash
python3 eit3_server.py
```

El navegador se abre automáticamente en http://localhost:7373

**2. Cargar archivos:**
- Canal VOZ → el WAV de tu micrófono principal (RodeCaster)
- Canal CONTEXTO → el WAV del micrófono de ambiente

**3. Ajustar parámetros:**

| Parámetro | Descripción | Rango |
|---|---|---|
| LF | Libertad Funcional — cuánto ambiente opera sobre la voz | 0 (LF-0) → 1 (LF-3) |
| Umbral N9 | Sensibilidad del detector de error operativo | 0.05–0.5 |
| Attack ENV | Velocidad de detección de picos ambientales (ms) | 1–50 |
| Release ENV | Velocidad de decaimiento del envelope (ms) | 20–500 |

**4. Procesar y descargar el WAV resultante.**

---

## Indicadores (LEDs)

| LED | Nodo canónico | Significado |
|---|---|---|
| 🔴 Rojo | ERR_OUT / N9 | % del tiempo fuera de rango viable |
| 🔵 Azul | CTX_MOD | Nivel de exaptación contextual efectiva |
| 🟢 Verde | OUT_AUDIO | RMS de salida estructurada (S producida) |

---

## Uso con Adobe Audition

1. Grabar simultáneamente dos pistas en Audition con el RodeCaster:
   - Pista 1: MIC principal (voz)
   - Pista 2: MIC ambiente (contexto)
2. Exportar cada pista como WAV separado (File → Export → Multitrack Mixdown)
3. Cargar en el procesador EIT-3
4. Importar el WAV resultante de vuelta a Audition

---

## Correspondencia con el módulo analógico

| Bloque analógico | Implementación DSP |
|---|---|
| MIC1/Q1 | Canal de entrada VOZ |
| MIC2/Q2 | Canal de entrada CONTEXTO |
| LF/RV1 | Parámetro LF (0–1) |
| N_EX (D1/D2) | Suma no lineal via softplus |
| ENV_BUF (Q_BUFF) | Envelope follower attack/release |
| N9+R_HYS (LM393) | Comparador con histéresis |
| Λ/Q3 | Modulación por componente AC del envelope |
| LM358 sumador | Mezcla ponderada voz + CTX_MOD |

---

## Detener el servidor

Ctrl+C en la terminal.
