#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor de síntesis fisiológica (contenedor).
Corre en background en anima-a / anima-b / anima-conversacion.
De forma automática: detecta CSV nuevos, sintetiza, reduce 100x, y los sube al estado compartido.
"""

import os, sys, json, csv, time, glob
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ESTADO_DIR = Path(os.environ.get("ANIMA_ESTADO_DIR", "/data"))
SINTESIS_DIR = ESTADO_DIR / "sintesis"
CSV_LOGS_DIR = ESTADO_DIR / "logs"

SINTESIS_DIR.mkdir(parents=True, exist_ok=True)
CSV_LOGS_DIR.mkdir(parents=True, exist_ok=True)

PROCESADOS = set()  # csvs ya sintetizados

def sintetizar_csv(ruta_csv, ventana=100):
    """Lee CSV, agrega por ventana, devuelve síntesis pequeña."""
    print(f"[sintesis] procesando {Path(ruta_csv).name}...")
    
    with open(ruta_csv, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        
        cols_clave = ["Omega", "omega_A", "omega_B", "gradiente", "e_R", "A_sys_env",
                      "presion_desacople", "C_b", "R2", "LF_op", "juego", "exaptacion", "C_m"]
        cols_presentes = [c for c in cols_clave if c in header]
        
        n_filas = 0
        ventanas = []
        ventana_actual = []
        ts_inicio = None
        modo_stats = defaultdict(int)
        
        for fila in reader:
            n_filas += 1
            ts = float(fila.get("ts_real", 0))
            if ts_inicio is None:
                ts_inicio = ts
            modo = fila.get("modo_vida", "")
            modo_stats[modo] += 1
            
            vals = {c: float(fila.get(c, 0)) for c in cols_presentes}
            ventana_actual.append((ts, vals))
            
            if len(ventana_actual) >= ventana:
                agg = {"ts_inicio": ventana_actual[0][0], "ts_fin": ventana_actual[-1][0]}
                for col in cols_presentes:
                    vals_col = [v[col] for _, v in ventana_actual]
                    agg[col] = round(sum(vals_col) / len(vals_col), 3)
                ventanas.append(agg)
                ventana_actual = []
        
        # ventana final
        if ventana_actual:
            agg = {"ts_inicio": ventana_actual[0][0], "ts_fin": ventana_actual[-1][0]}
            for col in cols_presentes:
                vals_col = [v[col] for _, v in ventana_actual]
                agg[col] = round(sum(vals_col) / len(vals_col), 3)
            ventanas.append(agg)
        
        ts_fin = ventanas[-1]["ts_fin"] if ventanas else ts_inicio
        ratio = n_filas / max(1, len(ventanas))
        
        resultado = {
            "metadata": {
                "archivo_original": Path(ruta_csv).name,
                "filas_totales": n_filas,
                "ventanas": len(ventanas),
                "tamaño_ventana": ventana,
                "reduccion_ratio": round(ratio, 1),
                "duracion_s": round(ts_fin - ts_inicio, 1) if ts_inicio else 0,
                "modos": dict(modo_stats),
                "generado": datetime.now().isoformat()
            },
            "ventanas": ventanas
        }
        
        print(f"  ✓ {n_filas} filas → {len(ventanas)} ventanas ({ratio:.1f}x reducción)")
        return resultado

def monitor():
    """Corre en loop: detecta CSVs nuevos, sintetiza, y guarda."""
    print(f"[monitor] iniciado. observando {CSV_LOGS_DIR}")
    while True:
        try:
            # Busca CSVs nuevos en logs/
            csvs = glob.glob(str(CSV_LOGS_DIR / "*.csv"))
            for csv_path in csvs:
                if csv_path in PROCESADOS:
                    continue
                
                PROCESADOS.add(csv_path)
                try:
                    resultado = sintetizar_csv(csv_path, ventana=100)
                    
                    # Guarda síntesis en estado/sintesis/
                    nombre_syn = Path(csv_path).stem + "_syntesis.json"
                    ruta_syn = SINTESIS_DIR / nombre_syn
                    with open(ruta_syn, 'w') as f:
                        json.dump(resultado, f, indent=2)
                    print(f"  guardado: {ruta_syn}")
                    
                    # Limpia el CSV original si es muy grande (opcional)
                    if os.path.getsize(csv_path) > 200 * 1024 * 1024:  # > 200 MB
                        print(f"  [limpieza] {Path(csv_path).name} era {os.path.getsize(csv_path)/(1024*1024):.0f} MB")
                        # Aquí podrías borrarlo o archivarlo; por ahora solo log
                
                except Exception as e:
                    print(f"  ❌ error: {e}")
        
        except Exception as e:
            print(f"  ❌ monitor error: {e}")
        
        time.sleep(10)  # chequea cada 10 segundos

if __name__ == "__main__":
    # si se llama directamente (dev), procesa un CSV
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        resultado = sintetizar_csv(csv_path, ventana=int(sys.argv[2]) if len(sys.argv) > 2 else 100)
        print(json.dumps(resultado, indent=2))
    else:
        # en contenedor, corre el monitor
        monitor()
