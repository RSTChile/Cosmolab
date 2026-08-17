#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SINTETIZADOR DE FISIOLOGÍA — reduce CSV gigantes a síntesis analizable
================================================================================

USO
---
  python3 sintetizar_fisiologia.py <archivo.csv> [opciones]
  python3 sintetizar_fisiologia.py /Users/alexis/Desktop/RMD/...ANIMA_A/fisiologia_2026-06-27_00-00.csv \
    --ventana 100 --output síntesis.json --compress

QUÉ HACE
-------
1. Lee CSV de fisiología (~200 MB, 190k líneas)
2. Agrupa por ventana temporal (defecto: 100 pasos)
3. Calcula para cada ventana: media, min, max, std de columnas clave
4. Exporta a JSON (~5 MB) + CSV reducido (~10 MB) + resumen visual (PNG/HTML)
5. Extrae anomalías (picos, caídas, transiciones de modo)

COLUMNAS ANALIZADAS (13 clave de 16 original)
----------------------------------------------
Ómega, omega_A, omega_B, gradiente, e_R, A_sys_env, presion_desacople, C_b, R2,
LF_op, juego, xaptacion, C_m

RESULTADO
---------
  síntesis.json         — timeseries agregado (media/min/max/std) + metadatos
  síntesis.csv          — mismo en CSV (importable a Excel/Pandas)
  síntesis_anomalias.json  — transiciones, picos, caídas detectadas
  síntesis_resumen.html — gráficos interactivos (chart.js)
================================================================================
"""

import sys, os, json, csv, math
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Intenta numpy para rapidez; fallback a python puro si no existe.
try:
    import numpy as np
    HAS_NP = True
except:
    HAS_NP = False


def leer_csv_streaming(ruta, chunk=10000):
    """Lee CSV gigante por chunks (memoria O(chunk), no O(tamaño total)).
    Devuelve (header, iterator de dicts)."""
    # No usar 'with': el generador se consume después de que la función retorna,
    # así que el archivo debe seguir abierto y se cierra dentro del generador.
    f = open(ruta, 'r', encoding='utf-8', errors='replace')
    reader = csv.DictReader(f)
    header = reader.fieldnames or []
    def gen():
        try:
            buff = []
            for row in reader:
                buff.append(row)
                if len(buff) >= chunk:
                    for r in buff:
                        yield r
                    buff = []
            for r in buff:
                yield r
        finally:
            f.close()
    return header, gen()


def safe_float(v, defecto=0.0):
    """Convierte string a float robustamente."""
    try:
        return float(v)
    except:
        return defecto


def sintetizar(ruta_csv, ventana=100, columnas_clave=None, output_dir=None):
    """
    Lee CSV de fisiología, agrega por ventana, y devuelve síntesis.

    Args:
        ruta_csv: ruta al archivo CSV original
        ventana: número de pasos por ventana (defecto 100 = ~1 segundo a 100 Hz)
        columnas_clave: lista de columnas a analizar (defecto: preset)
        output_dir: directorio donde guardar síntesis (defecto: cwd)

    Devuelve: dict con síntesis, anomalías y metadata
    """
    if columnas_clave is None:
        columnas_clave = [
            "Omega", "omega_A", "omega_B", "gradiente", "e_R", "A_sys_env",
            "presion_desacople", "C_b", "R2", "LF_op", "juego", "exaptacion", "C_m"
        ]

    output_dir = Path(output_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)
    nombre_base = Path(ruta_csv).stem

    print(f"[síntesis] leyendo {ruta_csv}...")
    ruta_csv = Path(ruta_csv)
    if not ruta_csv.exists():
        raise FileNotFoundError(ruta_csv)

    header, filas_gen = leer_csv_streaming(str(ruta_csv))
    cols_presentes = [c for c in columnas_clave if c in header]
    print(f"[síntesis] columnas clave encontradas: {len(cols_presentes)}/{len(columnas_clave)}")

    # Agregación por ventana
    ventana_actual = []
    ventanas_datos = []
    anomalias = []
    n_filas = 0
    ts_inicio = None
    modo_vida_anterior = None

    for fila in filas_gen:
        n_filas += 1
        if n_filas % 50000 == 0:
            print(f"  ... {n_filas} filas procesadas")

        # Extrae timestamp y modo de vida
        ts = safe_float(fila.get("ts_real", 0))
        modo = fila.get("modo_vida", "desconocido")
        if ts_inicio is None:
            ts_inicio = ts

        # Extrae valores numéricos de columnas clave
        vals = {c: safe_float(fila.get(c, 0)) for c in cols_presentes}
        ventana_actual.append((ts, modo, vals))

        # Detecta transición de modo
        if modo != modo_vida_anterior and ventana_actual:
            anomalias.append({
                "tipo": "transición_modo",
                "desde": modo_vida_anterior,
                "a": modo,
                "fila": n_filas,
                "ts": ts
            })
            modo_vida_anterior = modo

        # Cierra ventana cuando alcanza el tamaño
        if len(ventana_actual) >= ventana:
            v_agg = _agregar_ventana(ventana_actual, cols_presentes)
            v_agg["fila_inicio"] = n_filas - len(ventana_actual)
            v_agg["fila_fin"] = n_filas
            ventanas_datos.append(v_agg)
            ventana_actual = []

    # Cierra ventana final si queda
    if ventana_actual:
        v_agg = _agregar_ventana(ventana_actual, cols_presentes)
        v_agg["fila_inicio"] = n_filas - len(ventana_actual)
        v_agg["fila_fin"] = n_filas
        ventanas_datos.append(v_agg)

    print(f"[síntesis] {n_filas} filas → {len(ventanas_datos)} ventanas")
    ts_fin = ventanas_datos[-1]["ts_fin"] if ventanas_datos else ts_inicio
    duracion_s = ts_fin - ts_inicio if ts_inicio else 0

    # Detecta anomalías: picos, caídas
    for i, v in enumerate(ventanas_datos):
        if i > 0 and i < len(ventanas_datos) - 1:
            v_ant = ventanas_datos[i - 1]
            v_sig = ventanas_datos[i + 1]
            for col in ["e_R", "presion_desacople", "gradiente"]:
                if col in v and col in v_ant and col in v_sig:
                    val_ant = v_ant[col + "_media"]
                    val = v[col + "_media"]
                    val_sig = v_sig[col + "_media"]
                    # Pico: sube y baja
                    if val > val_ant * 1.5 and val > val_sig * 1.5 and val_ant > 0:
                        anomalias.append({
                            "tipo": "pico", "columna": col, "ventana": i,
                            "valor": round(val, 4), "media_ant": round(val_ant, 4)
                        })
                    # Caída: baja fuerte
                    if val < val_ant * 0.5 and val_ant > 1e-2:
                        anomalias.append({
                            "tipo": "caída", "columna": col, "ventana": i,
                            "valor": round(val, 4), "media_ant": round(val_ant, 4)
                        })

    # Estadísticas globales
    stats_global = {}
    for col in cols_presentes:
        vals = [v.get(col + "_media", 0) for v in ventanas_datos if col + "_media" in v]
        if vals:
            stats_global[col] = {
                "media": round(sum(vals) / len(vals), 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "std": round(_std(vals), 4) if len(vals) > 1 else 0.0
            }

    resultado = {
        "metadata": {
            "archivo_original": str(ruta_csv),
            "filas_totales": n_filas,
            "ventanas": len(ventanas_datos),
            "tamaño_ventana": ventana,
            "duracion_s": round(duracion_s, 1),
            "timestamp_inicio": ts_inicio,
            "timestamp_fin": ts_fin,
            "columnas_analizadas": cols_presentes,
            "generado": datetime.now().isoformat()
        },
        "estadisticas_globales": stats_global,
        "ventanas": ventanas_datos,
        "anomalias": anomalias[:100],  # top 100 para no inflar el JSON
        "reduccion": {
            "filas_originales": n_filas,
            "filas_sintesis": len(ventanas_datos),
            "ratio": round(n_filas / max(1, len(ventanas_datos)), 1),
            "reduccion_aprox_mb": round(Path(ruta_csv).stat().st_size / (1024*1024) / max(1, len(ventanas_datos)) * 100, 1)
        }
    }

    # Exporta JSON
    ruta_json = output_dir / f"{nombre_base}_sintesis.json"
    with open(ruta_json, 'w') as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False, default=str)
    print(f"  ✓ {ruta_json}")

    # Exporta CSV reducido
    ruta_csv_red = output_dir / f"{nombre_base}_sintesis.csv"
    if ventanas_datos:
        todas_cols = sorted(set(k.replace("_media", "").replace("_min", "").replace("_max", "").replace("_std", "")
                                 for d in ventanas_datos for k in d.keys()))
        with open(ruta_csv_red, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=todas_cols)
            w.writeheader()
            for v in ventanas_datos:
                fila_simplificada = {}
                for col in todas_cols:
                    # Para cada columna, prioriza media > valor directo > 0
                    clave_media = col + "_media"
                    fila_simplificada[col] = v.get(clave_media, v.get(col, 0))
                w.writerow(fila_simplificada)
    print(f"  ✓ {ruta_csv_red}")

    # Exporta anomalías
    ruta_anom = output_dir / f"{nombre_base}_anomalias.json"
    with open(ruta_anom, 'w') as f:
        json.dump(anomalias[:200], f, indent=2, ensure_ascii=False)
    print(f"  ✓ {ruta_anom} ({len(anomalias)} anomalías)")

    # Exporta HTML con gráficos (placeholder)
    _generar_html_resumen(resultado, output_dir / f"{nombre_base}_resumen.html")

    return resultado


def _agregar_ventana(filas, cols_presentes):
    """Agrega una ventana de filas: calcula media/min/max/std por columna."""
    agg = {"ts_inicio": filas[0][0], "ts_fin": filas[-1][0]}
    modos = defaultdict(int)

    for ts, modo, vals in filas:
        modos[modo] += 1
        for col, val in vals.items():
            if col not in agg:
                agg[col] = []
            agg[col].append(val)

    agg["modo_dominante"] = max(modos, key=modos.get) if modos else "desconocido"

    for col in cols_presentes:
        vals = agg.pop(col, [])
        if vals:
            agg[col + "_media"] = round(sum(vals) / len(vals), 4)
            agg[col + "_min"] = round(min(vals), 4)
            agg[col + "_max"] = round(max(vals), 4)
            agg[col + "_std"] = round(_std(vals), 4) if len(vals) > 1 else 0.0

    return agg


def _std(vals):
    """Calcula desviación estándar (fallback python puro)."""
    if len(vals) < 2:
        return 0.0
    media = sum(vals) / len(vals)
    var = sum((x - media) ** 2 for x in vals) / (len(vals) - 1)
    return math.sqrt(max(0, var))


def _generar_html_resumen(resultado, ruta_html):
    """Genera HTML interactivo con gráficos (chart.js)."""
    meta = resultado["metadata"]
    stats = resultado["estadisticas_globales"]
    ventanas = resultado["ventanas"][:1000]  # limita a 1k ventanas para no saturar

    # Construye series de datos para chart.js
    labels = [f"v{i}" for i in range(len(ventanas))]
    series_omega = [v.get("Omega_media", 0) for v in ventanas]
    series_grad = [v.get("gradiente_media", 0) for v in ventanas]
    series_lf = [v.get("LF_op_media", 0) for v in ventanas]

    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Síntesis Fisiología</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    body {{ margin: 20px; background: #0a0e14; color: #dfe7f0; font-family: monospace; }}
    h1 {{ color: #e8b86d; }}
    .stat {{ display: inline-block; margin: 10px 20px 10px 0; padding: 10px; background: #121925; border-radius: 6px; }}
    .stat .label {{ color: #8aa0b8; font-size: 11px; }}
    .stat .value {{ color: #e8b86d; font-size: 18px; font-weight: bold; }}
    .chart-container {{ position: relative; height: 200px; margin-bottom: 30px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
    th, td {{ border: 1px solid #243246; padding: 8px; text-align: left; }}
    th {{ background: #121925; color: #e8b86d; }}
  </style>
</head>
<body>
  <h1>📊 Síntesis Fisiología</h1>
  <p><strong>Archivo:</strong> {meta['archivo_original']}</p>
  <p><strong>Reducción:</strong> {meta['filas_totales']:,} filas → {meta['ventanas']} ventanas ({resultado['reduccion']['ratio']}x)</p>
  <p><strong>Duración:</strong> {meta['duracion_s']}s · <strong>Generado:</strong> {meta['generado']}</p>

  <h2>Estadísticas Globales</h2>
  <div>
"""
    for col in sorted(stats.keys()):
        s = stats[col]
        html += f"""
    <div class="stat">
      <div class="label">{col}</div>
      <div class="value">{s['media']:.3f}</div>
      <div style="font-size:10px;color:#8aa0b8">min={s['min']:.3f} max={s['max']:.3f}</div>
    </div>
"""

    html += f"""
  </div>
  <h2>Series Temporales</h2>
  <div class="chart-container"><canvas id="cOmega"></canvas></div>
  <div class="chart-container"><canvas id="cGrad"></canvas></div>
  <div class="chart-container"><canvas id="cLF"></canvas></div>

  <h2>Anomalías ({len(resultado['anomalias'])} detectadas)</h2>
  <table><tr><th>Tipo</th><th>Detalles</th></tr>
"""
    for anom in resultado["anomalias"][:20]:
        detalles = ", ".join(f"{k}={v}" for k, v in anom.items() if k != "tipo")
        html += f"<tr><td>{anom['tipo']}</td><td>{detalles}</td></tr>"
    html += "</table>"

    html += f"""
  <script>
    const labels = {json.dumps(labels)};
    const omega = {json.dumps(series_omega)};
    const grad = {json.dumps(series_grad)};
    const lf = {json.dumps(series_lf)};
    new Chart(document.getElementById('cOmega'), {{
      type: 'line', data: {{ labels, datasets: [{{ label: 'Omega', data: omega, borderColor: '#e8b86d', borderWidth: 2 }}] }},
      options: {{ animation: false, responsive: true, maintainAspectRatio: false, scales: {{
        x: {{ grid: {{ color: '#1a2330' }} }}, y: {{ grid: {{ color: '#1a2330' }} }}
      }}, plugins: {{ legend: {{ labels: {{ color: '#dfe7f0' }} }} }} }}
    }});
    new Chart(document.getElementById('cGrad'), {{
      type: 'line', data: {{ labels, datasets: [{{ label: 'Gradiente', data: grad, borderColor: '#6db6ff', borderWidth: 2 }}] }},
      options: {{ animation: false, responsive: true, maintainAspectRatio: false, scales: {{
        x: {{ grid: {{ color: '#1a2330' }} }}, y: {{ grid: {{ color: '#1a2330' }} }}
      }}, plugins: {{ legend: {{ labels: {{ color: '#dfe7f0' }} }} }} }}
    }});
    new Chart(document.getElementById('cLF'), {{
      type: 'line', data: {{ labels, datasets: [{{ label: 'LF', data: lf, borderColor: '#5fd38a', borderWidth: 2 }}] }},
      options: {{ animation: false, responsive: true, maintainAspectRatio: false, scales: {{
        x: {{ grid: {{ color: '#1a2330' }} }}, y: {{ grid: {{ color: '#1a2330' }} }}
      }}, plugins: {{ legend: {{ labels: {{ color: '#dfe7f0' }} }} }} }}
    }});
  </script>
</body>
</html>
"""
    with open(ruta_html, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  ✓ {ruta_html}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("USO: python3 sintetizar_fisiologia.py <archivo.csv> [--ventana N] [--output DIR]")
        sys.exit(1)

    ruta = sys.argv[1]
    ventana = 100
    output_dir = None

    # Parsea argumentos
    for i, arg in enumerate(sys.argv[2:]):
        if arg == "--ventana" and i + 3 < len(sys.argv):
            ventana = int(sys.argv[i + 3])
        if arg == "--output" and i + 3 < len(sys.argv):
            output_dir = sys.argv[i + 3]

    print(f"\nSINTETIZADOR DE FISIOLOGÍA")
    print(f"  archivo: {ruta}")
    print(f"  ventana: {ventana} pasos")
    print(f"  output: {output_dir or 'cwd'}\n")

    resultado = sintetizar(ruta, ventana=ventana, output_dir=output_dir)
    print(f"\n✓ Síntesis completada.")
    print(f"  Reducción: {resultado['reduccion']['ratio']}x")
    print(f"  Anomalías detectadas: {len(resultado['anomalias'])}")
