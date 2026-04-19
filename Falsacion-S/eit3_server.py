#!/usr/bin/env python3
"""
EIT-3 Cosmosemiotic Audio Processor
====================================
Implementación DSP del módulo EIT-3 Lite (RMD 2.0)
Servidor HTTP local + UI web. Sin dependencias externas.

Uso:
    python3 eit3_server.py
    Abrir: http://localhost:7373

Entradas: dos archivos WAV del RodeCaster Pro II
    - Canal VOZ (MIC principal)
    - Canal CONTEXTO (MIC ambiente)

Salida: WAV procesado por cosmosemiótica
"""

import http.server
import json
import base64
import io
import math
import threading
import webbrowser
import os
import struct
import wave
from urllib.parse import parse_qs

import numpy as np
from scipy import signal
from scipy.io import wavfile

# ─────────────────────────────────────────────────────────────
# NÚCLEO DSP — TRADUCCIÓN DEL CIRCUITO ANALÓGICO A DIGITAL
# ─────────────────────────────────────────────────────────────

def envelope_follower(x, sr, attack_ms=10, release_ms=100):
    """
    Bloque 5 — Detector ENV (C3/D3/C4/R13/Q_BUFF)
    Sigue la envolvente de amplitud de la señal.
    """
    attack  = 1.0 - math.exp(-1.0 / (sr * attack_ms  / 1000.0))
    release = 1.0 - math.exp(-1.0 / (sr * release_ms / 1000.0))
    env = np.zeros_like(x, dtype=np.float64)
    state = 0.0
    for i, s in enumerate(np.abs(x)):
        coef = attack if s > state else release
        state += coef * (s - state)
        env[i] = state
    return env


def lf_gate(ctx_signal, lf):
    """
    Bloque 3 — Libertad Funcional (RV1)
    Controla la permeabilidad del canal contextual.
    lf: 0.0 (LF-0, cerrado) → 1.0 (LF-3, máxima apertura)
    """
    return ctx_signal * lf


def n_ex_node(voice, ctx_gated):
    """
    Bloque 4 — Nodo de Exaptación N_EX (D1/D2/R12/R_BIAS_EX)
    Suma no lineal voz + contexto — ecotono electrónico.
    D1/D2 simulados como rectificación suave (softplus ≈ diodo).
    """
    def softplus(x, threshold=0.05):
        # Aproxima la conducción de un diodo 1N4148
        return np.where(x > threshold, x, threshold * np.log1p(np.exp((x - threshold) / threshold)))

    v_ex  = softplus(voice)
    c_ex  = softplus(ctx_gated)
    n_ex  = v_ex + c_ex
    # Bias DC (R_BIAS_EX): normalizar para mantener zona operativa
    n_ex  = n_ex / (np.max(np.abs(n_ex)) + 1e-9)
    return n_ex


def n9_comparator(env_buf, threshold, hysteresis=0.05):
    """
    Bloque 6 — Comparador N9 + Histéresis N10 (LM393 + R_HYS)
    Detecta cuándo A_sys-env está fuera de rango viable.
    Retorna señal binaria: 1 = en rango, 0 = error (N9 activo)
    """
    n9 = np.zeros(len(env_buf))
    state = 1  # empieza en rango
    for i, v in enumerate(env_buf):
        if state == 1 and v < (threshold - hysteresis):
            state = 0  # cruza hacia error
        elif state == 0 and v > (threshold + hysteresis):
            state = 1  # recupera rango
        n9[i] = state
    return n9


def lambda_operator(env_buf, n_ex, lf):
    """
    Bloque 8 — Operador Λ (Q3/BC547)
    Transforma ENV_BUF (medida pasiva) en CTX_MOD (modulación activa).
    Opera sobre componente dinámica (AC) de ENV_BUF — decisión canónica.
    lf controla la profundidad de modulación.
    """
    # C5: extraer componente AC (variación contextual, no nivel absoluto)
    from scipy.signal import butter, filtfilt
    b, a = butter(1, 0.01, btype='high')
    env_ac = filtfilt(b, a, env_buf)

    # Q3: modulación — el contexto dinámico modula la señal N_EX
    # Profundidad de modulación proporcional a LF
    mod_depth = lf * 0.8  # máximo 80% de modulación a LF-3
    ctx_mod = n_ex * (1.0 + mod_depth * env_ac)
    return ctx_mod


def lm358_summer(voice, ctx_mod, env_buf, n9_signal, lf, vref=0.5):
    """
    Bloque 9 — Sumador Final (LM358)
    Suma voz + contexto transformado referenciado a VREF.
    La ganancia de CTX_MOD escala con el acoplamiento viable (N9 activo).
    """
    # Ganancia de contexto: mayor cuando sistema está en rango viable
    # y proporcional a LF y al nivel de ENV_BUF
    ctx_gain = lf * n9_signal * env_buf

    # Suma ponderada con referencia a VREF (zona operativa central)
    result = voice + ctx_gain * ctx_mod

    # VREF: centrar la resultante en la zona operativa
    result = result - np.mean(result) + vref * np.std(result)

    # Normalizar sin clipear
    peak = np.max(np.abs(result))
    if peak > 1e-9:
        result = result / peak * 0.9

    return result


def compute_indicators(env_buf, ctx_mod, output, n9_signal):
    """
    LEDs del módulo:
    Rojo  → fracción del tiempo con N9 activo (error operativo)
    Azul  → nivel medio de exaptación efectiva (CTX_MOD)
    Verde → nivel RMS de salida estructurada
    """
    led_red   = 1.0 - np.mean(n9_signal)  # 0=todo OK, 1=todo error
    led_blue  = float(np.mean(np.abs(ctx_mod)))
    led_green = float(np.sqrt(np.mean(output**2)))

    # Normalizar a 0-1
    led_blue  = min(1.0, led_blue  * 5)
    led_green = min(1.0, led_green * 3)

    return {
        'red':   round(led_red,   3),
        'blue':  round(led_blue,  3),
        'green': round(led_green, 3)
    }


def process_eit3(voice_data, ctx_data, sr,
                 lf=0.7, n9_threshold=0.15,
                 attack_ms=10, release_ms=150):
    """
    Pipeline completo EIT-3 Lite
    ────────────────────────────
    Bloques 0-11 implementados como cadena DSP.
    """
    # Normalizar entradas
    voice = voice_data.astype(np.float64) / 32768.0
    ctx   = ctx_data.astype(np.float64)   / 32768.0

    # Igualar longitudes
    min_len = min(len(voice), len(ctx))
    voice = voice[:min_len]
    ctx   = ctx[:min_len]

    # Bloque 3 — LF gate
    ctx_gated = lf_gate(ctx, lf)

    # Bloque 4 — N_EX ecotono
    n_ex = n_ex_node(voice, ctx_gated)

    # Bloque 5 — ENV_BUF
    env_buf = envelope_follower(ctx_gated, sr, attack_ms, release_ms)
    env_buf = env_buf / (np.max(env_buf) + 1e-9)  # normalizar a 0-1

    # Bloque 6 — N9 + histéresis
    n9_signal = n9_comparator(env_buf, n9_threshold)

    # Bloque 8 — Operador Λ
    ctx_mod = lambda_operator(env_buf, n_ex, lf)

    # Bloque 9 — Sumador LM358
    output = lm358_summer(voice, ctx_mod, env_buf, n9_signal, lf)

    # LEDs
    indicators = compute_indicators(env_buf, ctx_mod, output, n9_signal)

    # Convertir a int16 para WAV
    output_int16 = (output * 32767).astype(np.int16)

    return output_int16, sr, indicators


# ─────────────────────────────────────────────────────────────
# LECTURA / ESCRITURA WAV ROBUSTA
# ─────────────────────────────────────────────────────────────

def read_wav_bytes(data_bytes):
    """Lee WAV desde bytes, retorna (samples_int16, sample_rate)"""
    buf = io.BytesIO(data_bytes)
    sr, samples = wavfile.read(buf)
    # Convertir a mono si es stereo
    if samples.ndim == 2:
        samples = samples[:, 0]
    # Convertir a int16 si es float o int32
    if samples.dtype != np.int16:
        samples = (samples / np.max(np.abs(samples) + 1e-9) * 32767).astype(np.int16)
    return samples, sr


def write_wav_bytes(samples, sr):
    """Escribe WAV a bytes"""
    buf = io.BytesIO()
    wavfile.write(buf, sr, samples)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# HTML DE LA INTERFAZ
# ─────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EIT-3 Cosmosemiotic Processor</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

  :root {
    --bg:       #0a0a0f;
    --surface:  #12121a;
    --border:   #1e1e2e;
    --accent:   #00e5ff;
    --red:      #ff3b3b;
    --blue:     #00b4ff;
    --green:    #00ff9d;
    --text:     #e0e0f0;
    --muted:    #5a5a7a;
    --mono:     'Space Mono', monospace;
    --sans:     'Syne', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px 20px;
  }

  /* HEADER */
  header {
    text-align: center;
    margin-bottom: 48px;
  }
  .logo-line {
    font-size: 11px;
    letter-spacing: 0.25em;
    color: var(--accent);
    font-family: var(--mono);
    text-transform: uppercase;
    margin-bottom: 12px;
  }
  h1 {
    font-size: clamp(28px, 5vw, 48px);
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #fff 0%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
  }
  .subtitle {
    color: var(--muted);
    font-size: 13px;
    font-family: var(--mono);
    margin-top: 10px;
  }

  /* MAIN GRID */
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    width: 100%;
    max-width: 900px;
    margin-bottom: 24px;
  }

  /* PANELS */
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
  }
  .panel-title {
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    font-family: var(--mono);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .panel-title::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
  }

  /* DROP ZONES */
  .drop-zone {
    border: 1.5px dashed var(--border);
    border-radius: 8px;
    padding: 32px 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    position: relative;
  }
  .drop-zone:hover, .drop-zone.drag-over {
    border-color: var(--accent);
    background: rgba(0,229,255,0.04);
  }
  .drop-zone.loaded {
    border-color: var(--green);
    border-style: solid;
  }
  .drop-icon {
    font-size: 28px;
    margin-bottom: 10px;
    display: block;
  }
  .drop-label {
    font-size: 12px;
    color: var(--muted);
    font-family: var(--mono);
  }
  .drop-filename {
    font-size: 11px;
    color: var(--green);
    font-family: var(--mono);
    margin-top: 8px;
    word-break: break-all;
  }
  input[type=file] {
    position: absolute; inset: 0;
    opacity: 0; cursor: pointer;
  }

  /* CHANNEL LABELS */
  .ch-voice { color: #a78bfa; }
  .ch-ctx   { color: #fb923c; }

  /* CONTROLS */
  .controls-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }
  .control {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .control label {
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    font-family: var(--mono);
    display: flex;
    justify-content: space-between;
  }
  .control label span {
    color: var(--accent);
  }
  input[type=range] {
    -webkit-appearance: none;
    width: 100%;
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    outline: none;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
    transition: transform 0.1s;
  }
  input[type=range]::-webkit-slider-thumb:hover {
    transform: scale(1.3);
  }

  .lf-regions {
    display: flex;
    justify-content: space-between;
    font-size: 9px;
    font-family: var(--mono);
    color: var(--muted);
    margin-top: 2px;
  }

  /* PROCESS BUTTON */
  .btn-process {
    width: 100%;
    max-width: 900px;
    padding: 18px;
    background: linear-gradient(135deg, rgba(0,229,255,0.15), rgba(0,180,255,0.08));
    border: 1px solid var(--accent);
    border-radius: 10px;
    color: var(--accent);
    font-family: var(--mono);
    font-size: 13px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.2s;
    margin-bottom: 24px;
  }
  .btn-process:hover:not(:disabled) {
    background: rgba(0,229,255,0.2);
    transform: translateY(-1px);
  }
  .btn-process:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }
  .btn-process.processing {
    animation: pulse 1s ease-in-out infinite;
  }
  @keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.5; }
  }

  /* INDICATORS */
  .indicators {
    display: flex;
    gap: 16px;
    width: 100%;
    max-width: 900px;
    margin-bottom: 24px;
  }
  .indicator {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .led {
    width: 12px; height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
    transition: all 0.5s;
  }
  .led-red   { background: #2a1010; box-shadow: none; }
  .led-blue  { background: #101a2a; box-shadow: none; }
  .led-green { background: #0a2a1a; box-shadow: none; }
  .led-red.on   { background: var(--red);   box-shadow: 0 0 12px var(--red); }
  .led-blue.on  { background: var(--blue);  box-shadow: 0 0 12px var(--blue); }
  .led-green.on { background: var(--green); box-shadow: 0 0 12px var(--green); }
  .ind-info { flex: 1; }
  .ind-name {
    font-size: 9px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-family: var(--mono);
    color: var(--muted);
    margin-bottom: 4px;
  }
  .ind-value {
    font-size: 18px;
    font-family: var(--mono);
    font-weight: 700;
  }
  .ind-bar {
    height: 2px;
    background: var(--border);
    border-radius: 1px;
    margin-top: 6px;
    overflow: hidden;
  }
  .ind-bar-fill {
    height: 100%;
    border-radius: 1px;
    transition: width 0.5s ease;
    width: 0%;
  }
  .fill-red   { background: var(--red); }
  .fill-blue  { background: var(--blue); }
  .fill-green { background: var(--green); }

  /* OUTPUT */
  .output-panel {
    width: 100%;
    max-width: 900px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    display: none;
  }
  .output-panel.visible { display: block; }
  .output-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }
  .btn-download {
    padding: 10px 20px;
    background: rgba(0,255,157,0.1);
    border: 1px solid var(--green);
    border-radius: 6px;
    color: var(--green);
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    cursor: pointer;
    text-decoration: none;
    display: inline-block;
    transition: all 0.2s;
  }
  .btn-download:hover {
    background: rgba(0,255,157,0.2);
  }
  .analysis {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    line-height: 1.8;
    border-top: 1px solid var(--border);
    padding-top: 16px;
    margin-top: 16px;
  }
  .analysis strong { color: var(--text); }

  /* STATUS */
  .status {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    text-align: center;
    margin-bottom: 8px;
  }

  @media (max-width: 600px) {
    .grid { grid-template-columns: 1fr; }
    .controls-grid { grid-template-columns: 1fr; }
    .indicators { flex-direction: column; }
  }
</style>
</head>
<body>

<header>
  <div class="logo-line">RMD 2.0 / Cosmosemiótica Canónica — EIT-3 Lite</div>
  <h1>Cosmosemiotic<br>Audio Processor</h1>
  <div class="subtitle">N_EX · ENV_BUF · Λ · LM358 · N9+HYS</div>
</header>

<!-- FILE INPUTS -->
<div class="grid">
  <div class="panel">
    <div class="panel-title ch-voice">Canal VOZ — MIC Principal</div>
    <div class="drop-zone" id="drop-voice">
      <span class="drop-icon">🎙</span>
      <div class="drop-label">Arrastra tu WAV de voz<br>o haz clic para seleccionar</div>
      <div class="drop-filename" id="fname-voice"></div>
      <input type="file" id="file-voice" accept=".wav">
    </div>
  </div>
  <div class="panel">
    <div class="panel-title ch-ctx">Canal CONTEXTO — MIC Ambiente</div>
    <div class="drop-zone" id="drop-ctx">
      <span class="drop-icon">🌐</span>
      <div class="drop-label">Arrastra tu WAV de ambiente<br>o haz clic para seleccionar</div>
      <div class="drop-filename" id="fname-ctx"></div>
      <input type="file" id="file-ctx" accept=".wav">
    </div>
  </div>
</div>

<!-- CONTROLS -->
<div class="panel" style="width:100%;max-width:900px;margin-bottom:24px;">
  <div class="panel-title">Parámetros Canónicos</div>
  <div class="controls-grid">
    <div class="control">
      <label>Libertad Funcional (LF) <span id="lf-val">0.70</span></label>
      <input type="range" id="ctrl-lf" min="0" max="1" step="0.01" value="0.70">
      <div class="lf-regions"><span>LF-0</span><span>LF-1</span><span>LF-2</span><span>LF-3</span></div>
    </div>
    <div class="control">
      <label>Umbral N9 (th_osc) <span id="n9-val">0.15</span></label>
      <input type="range" id="ctrl-n9" min="0.05" max="0.5" step="0.01" value="0.15">
    </div>
    <div class="control">
      <label>Attack ENV (ms) <span id="att-val">10</span></label>
      <input type="range" id="ctrl-att" min="1" max="50" step="1" value="10">
    </div>
    <div class="control">
      <label>Release ENV (ms) <span id="rel-val">150</span></label>
      <input type="range" id="ctrl-rel" min="20" max="500" step="10" value="150">
    </div>
  </div>
</div>

<!-- PROCESS BUTTON -->
<div class="status" id="status">Carga los dos archivos WAV para procesar</div>
<button class="btn-process" id="btn-process" disabled>
  ◈ Procesar — EIT-3 Cosmosemiotic Engine
</button>

<!-- INDICATORS -->
<div class="indicators">
  <div class="indicator">
    <div class="led led-red" id="led-red"></div>
    <div class="ind-info">
      <div class="ind-name">ERR_OUT / N9</div>
      <div class="ind-value" id="val-red" style="color:var(--red)">—</div>
      <div class="ind-bar"><div class="ind-bar-fill fill-red" id="bar-red"></div></div>
    </div>
  </div>
  <div class="indicator">
    <div class="led led-blue" id="led-blue"></div>
    <div class="ind-info">
      <div class="ind-name">CTX_MOD / Exaptación</div>
      <div class="ind-value" id="val-blue" style="color:var(--blue)">—</div>
      <div class="ind-bar"><div class="ind-bar-fill fill-blue" id="bar-blue"></div></div>
    </div>
  </div>
  <div class="indicator">
    <div class="led led-green" id="led-green"></div>
    <div class="ind-info">
      <div class="ind-name">OUT_AUDIO / S producida</div>
      <div class="ind-value" id="val-green" style="color:var(--green)">—</div>
      <div class="ind-bar"><div class="ind-bar-fill fill-green" id="bar-green"></div></div>
    </div>
  </div>
</div>

<!-- OUTPUT -->
<div class="output-panel" id="output-panel">
  <div class="output-header">
    <div class="panel-title" style="margin:0">Resultado — S estructurada</div>
    <a class="btn-download" id="btn-download" href="#" download="EIT3_output.wav">
      ↓ Descargar WAV
    </a>
  </div>
  <audio id="audio-preview" controls style="width:100%;margin-bottom:16px;accent-color:var(--accent);"></audio>
  <div class="analysis" id="analysis"></div>
</div>

<script>
let voiceFile = null, ctxFile = null;

// ── Sliders ─────────────────────────────────────────────────
function bindSlider(id, labelId) {
  const el = document.getElementById(id);
  const lb = document.getElementById(labelId);
  el.addEventListener('input', () => lb.textContent = parseFloat(el.value).toFixed(2));
}
bindSlider('ctrl-lf',  'lf-val');
bindSlider('ctrl-n9',  'n9-val');
document.getElementById('ctrl-att').addEventListener('input', e =>
  document.getElementById('att-val').textContent = e.target.value);
document.getElementById('ctrl-rel').addEventListener('input', e =>
  document.getElementById('rel-val').textContent = e.target.value);

// ── File handling ────────────────────────────────────────────
function setupDrop(dropId, fileInputId, fnameId, channel) {
  const drop  = document.getElementById(dropId);
  const input = document.getElementById(fileInputId);
  const fname = document.getElementById(fnameId);

  input.addEventListener('change', e => {
    const f = e.target.files[0];
    if (!f) return;
    if (channel === 'voice') voiceFile = f; else ctxFile = f;
    fname.textContent = f.name;
    drop.classList.add('loaded');
    checkReady();
  });

  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag-over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('drag-over'));
  drop.addEventListener('drop', e => {
    e.preventDefault();
    drop.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (!f || !f.name.endsWith('.wav')) return;
    if (channel === 'voice') voiceFile = f; else ctxFile = f;
    document.getElementById(fnameId).textContent = f.name;
    drop.classList.add('loaded');
    checkReady();
  });
}

setupDrop('drop-voice', 'file-voice', 'fname-voice', 'voice');
setupDrop('drop-ctx',   'file-ctx',   'fname-ctx',   'ctx');

function checkReady() {
  const ready = voiceFile && ctxFile;
  document.getElementById('btn-process').disabled = !ready;
  document.getElementById('status').textContent = ready
    ? 'Listo para procesar — ajusta los parámetros y presiona el botón'
    : 'Carga los dos archivos WAV para procesar';
}

// ── Process ──────────────────────────────────────────────────
document.getElementById('btn-process').addEventListener('click', async () => {
  const btn = document.getElementById('btn-process');
  btn.disabled = true;
  btn.classList.add('processing');
  btn.textContent = '◈ Procesando...';
  document.getElementById('status').textContent = 'Ejecutando pipeline EIT-3...';

  try {
    const [voiceB64, ctxB64] = await Promise.all([
      fileToBase64(voiceFile), fileToBase64(ctxFile)
    ]);

    const params = {
      voice:      voiceB64,
      ctx:        ctxB64,
      lf:         parseFloat(document.getElementById('ctrl-lf').value),
      n9:         parseFloat(document.getElementById('ctrl-n9').value),
      attack_ms:  parseInt(document.getElementById('ctrl-att').value),
      release_ms: parseInt(document.getElementById('ctrl-rel').value)
    };

    const resp = await fetch('/process', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(params)
    });

    const result = await resp.json();
    if (result.error) throw new Error(result.error);

    updateIndicators(result.indicators);
    showOutput(result.wav_b64, result.indicators, params);

  } catch(err) {
    document.getElementById('status').textContent = 'Error: ' + err.message;
  }

  btn.disabled = false;
  btn.classList.remove('processing');
  btn.textContent = '◈ Procesar de nuevo — EIT-3 Cosmosemiotic Engine';
});

function fileToBase64(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result.split(',')[1]);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
}

// ── Indicators ───────────────────────────────────────────────
function updateIndicators(ind) {
  function set(color, val) {
    const pct = Math.round(val * 100);
    document.getElementById(`val-${color}`).textContent = pct + '%';
    document.getElementById(`bar-${color}`).style.width = pct + '%';
    const led = document.getElementById(`led-${color}`);
    if (val > 0.15) led.classList.add('on'); else led.classList.remove('on');
  }
  set('red',   ind.red);
  set('blue',  ind.blue);
  set('green', ind.green);
}

// ── Output ───────────────────────────────────────────────────
function showOutput(wavB64, ind, params) {
  const blob = b64toBlob(wavB64, 'audio/wav');
  const url  = URL.createObjectURL(blob);

  document.getElementById('audio-preview').src = url;
  const dl = document.getElementById('btn-download');
  dl.href = url;
  dl.download = `EIT3_LF${params.lf}_N9${params.n9}.wav`;

  // Análisis canónico
  const lf = params.lf;
  const lfLabel = lf < 0.25 ? 'LF-0 (sistema cerrado)' :
                  lf < 0.50 ? 'LF-1 (umbral mínimo)' :
                  lf < 0.75 ? 'LF-2 (zona de tensión)' : 'LF-3 (exaptación máxima)';

  const n9status = ind.red < 0.1 ? 'Sistema en rango viable durante toda la sesión' :
                   ind.red < 0.4 ? `N9 activo ${Math.round(ind.red*100)}% del tiempo — régimen mixto` :
                   `N9 activo ${Math.round(ind.red*100)}% — contexto fuera de rango frecuente`;

  document.getElementById('analysis').innerHTML = `
    <strong>Análisis canónico de la sesión:</strong><br><br>
    LF aplicado: <strong>${params.lf} → ${lfLabel}</strong><br>
    ${n9status}<br>
    Exaptación contextual (CTX_MOD): <strong>${Math.round(ind.blue*100)}%</strong> — 
      ${ind.blue > 0.5 ? 'contexto operando activamente sobre la voz' : 'modulación contextual moderada'}<br>
    Salida estructurada (S): <strong>${Math.round(ind.green*100)}% RMS</strong><br><br>
    <strong>Nota:</strong> La voz resultante integra el ambiente como recurso estructural (N17/N18),
    no como ruido eliminado. El EIT-3 no limpia — transforma.
  `;

  document.getElementById('output-panel').classList.add('visible');
  document.getElementById('status').textContent = 'Proceso completo — S producida';
}

function b64toBlob(b64, type) {
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return new Blob([arr], {type});
}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────
# SERVIDOR HTTP
# ─────────────────────────────────────────────────────────────

class EIT3Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # silenciar logs de acceso

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != '/process':
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers['Content-Length'])
        body   = self.rfile.read(length)

        try:
            params     = json.loads(body)
            voice_wav  = base64.b64decode(params['voice'])
            ctx_wav    = base64.b64decode(params['ctx'])
            lf         = float(params.get('lf', 0.7))
            n9         = float(params.get('n9', 0.15))
            attack_ms  = int(params.get('attack_ms', 10))
            release_ms = int(params.get('release_ms', 150))

            voice_data, sr_v = read_wav_bytes(voice_wav)
            ctx_data,   sr_c = read_wav_bytes(ctx_wav)

            # Resamplear contexto si tiene diferente SR
            if sr_v != sr_c:
                from scipy.signal import resample
                target_len = int(len(ctx_data) * sr_v / sr_c)
                ctx_data   = resample(ctx_data, target_len).astype(np.int16)

            output, sr, indicators = process_eit3(
                voice_data, ctx_data, sr_v,
                lf=lf, n9_threshold=n9,
                attack_ms=attack_ms, release_ms=release_ms
            )

            wav_bytes = write_wav_bytes(output, sr)
            wav_b64   = base64.b64encode(wav_bytes).decode('ascii')

            response = json.dumps({'wav_b64': wav_b64, 'indicators': indicators})
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(response.encode('utf-8'))

        except Exception as e:
            error = json.dumps({'error': str(e)})
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(error.encode('utf-8'))


def main():
    PORT = 7373
    server = http.server.HTTPServer(('localhost', PORT), EIT3Handler)

    print()
    print('  ╔══════════════════════════════════════╗')
    print('  ║  EIT-3 Cosmosemiotic Audio Processor ║')
    print('  ║  RMD 2.0 / Cosmosemiótica Canónica   ║')
    print('  ╠══════════════════════════════════════╣')
    print(f' ║  http://localhost:{PORT}               ║')
    print('  ╚══════════════════════════════════════╝')
    print()
    print('  Abriendo navegador...')
    print('  Ctrl+C para detener el servidor')
    print()

    threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{PORT}')).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n  Servidor detenido.')


if __name__ == '__main__':
    main()
