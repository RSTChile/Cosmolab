#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento Cosmosemiótico v2 - Validación con grabaciones reales
Correcciones:
- Calibración inicial con Voz_Estudio como referencia
- Acoplamiento por correlación coseno con perfil de referencia
- Supervivencia solo evaluada durante actividad vocal
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN
# ============================================================

FS = 48000  # Frecuencia de muestreo (Hz)
VENTANA_MS = 30
VENTANA_MUESTRAS = int(FS * VENTANA_MS / 1000)
HOP_MS = 10
HOP_MUESTRAS = int(FS * HOP_MS / 1000)

# Bandas de frecuencia
N_BANDAS = 32  # Reducido para estabilidad
F_MIN = 50
F_MAX = FS // 2
BANDAS = np.logspace(np.log10(F_MIN), np.log10(F_MAX), N_BANDAS + 1)
BANDAS_VOZ = (BANDAS[:-1] >= 100) & (BANDAS[:-1] <= 4000)
BANDAS_VIENTO = (BANDAS[:-1] <= 200)

# Umbrales
UMBRAL_ACOPLAMIENTO = 0.4
UMBRAL_ENERGIA_VOZ = 0.05  # Energía mínima para considerar que hay voz

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def cargar_wav_seguro(nombre):
    resultado = wav.read(nombre)
    if len(resultado) == 2:
        sr, data = resultado
    else:
        sr, data, _ = resultado
    
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    
    if data.ndim == 2:
        data = data.mean(axis=1)
    
    return sr, data

def espectrograma(audio, ventana, hop):
    n_ventanas = (len(audio) - ventana) // hop + 1
    espectro = np.zeros((N_BANDAS, n_ventanas))
    
    for i in range(n_ventanas):
        inicio = i * hop
        fragmento = audio[inicio:inicio + ventana]
        ventana_hann = np.hanning(len(fragmento))
        fragmento = fragmento * ventana_hann
        fft = np.fft.rfft(fragmento)
        potencia = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(fragmento), 1/FS)
        
        for b in range(N_BANDAS):
            mask = (freqs >= BANDAS[b]) & (freqs < BANDAS[b+1])
            if np.any(mask):
                espectro[b, i] = np.mean(potencia[mask])
    
    return espectro

def correlacion_coseno(a, b):
    """Correlación coseno normalizada entre dos vectores."""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ============================================================
# SISTEMA COSMOSEMIÓTICO
# ============================================================

class SistemaCosmosemiotico:
    def __init__(self, alfa=0.5, energia_computacion=0.9, perfil_voz_referencia=None):
        self.alfa = alfa
        self.energia_computacion = energia_computacion
        self.perfil_voz_referencia = perfil_voz_referencia
        
        # Estado
        self.campo_actual = None
        self.campo_anterior = None
        self.delta_struct = 0.0
        self.acoplamiento = 0.5
        self.acoplamiento_anterior = 0.5
        self.error = 0.5
        self.lf = 0.0
        self.icr = 0.5
        self.irde = 0.5
        self.poder_base = 0.5
        self.poder_vivo = 0.5
        
        # Modo
        self.modo_actual = "DESCONOCIDO"
        self.modo_candidato = "DESCONOCIDO"
        self.persistencia = 0
        
        # Mezcla
        self.mezcla_nada = 1.0
        self.mezcla_preservar = 0.0
        self.mezcla_contexto = 0.0
        
        self.historial = []
        self.tiempo_exposicion = 0
    
    def actualizar_campo(self, espectro):
        self.campo_anterior = self.campo_actual
        self.campo_actual = espectro[:, 0] if espectro.ndim > 1 else espectro
        
        if self.campo_anterior is not None:
            self.delta_struct = np.std(self.campo_actual - self.campo_anterior)
        else:
            self.delta_struct = 0.0
    
    def calcular_emergentes(self):
        if self.campo_actual is None or self.perfil_voz_referencia is None:
            return
        
        # Acoplamiento por correlación con referencia de voz
        self.acoplamiento = max(0.0, min(1.0, 
            correlacion_coseno(self.campo_actual, self.perfil_voz_referencia)))
        
        # Suavizado
        self.acoplamiento = 0.9 * self.acoplamiento_anterior + 0.1 * self.acoplamiento
        self.error = 1.0 - self.acoplamiento
        
        # Poder vivo y LF
        self.poder_vivo = 0.9 * self.poder_vivo + 0.1 * self.acoplamiento
        desviacion = abs(self.poder_vivo - self.poder_base)
        inercia = np.exp(-desviacion * 4.0)
        self.lf = min(1.0, desviacion * (1.0 - inercia))
        
        # ICR e IRDE
        self.icr = self.acoplamiento * (1.0 - self.error)
        self.irde = (1.0 - self.acoplamiento) * self.error
        
        self.acoplamiento_anterior = self.acoplamiento
    
    def calcular_signos(self):
        if self.campo_actual is None:
            return {}
        
        energia_bandas = self.campo_actual
        
        # Zona voz
        energia_voz = np.sum(energia_bandas[BANDAS_VOZ]) + 0.01
        energia_total = np.sum(energia_bandas) + 0.01
        s_zona_voz = min(1.0, energia_voz / energia_total)
        
        # Perturbación
        energia_viento = np.sum(energia_bandas[BANDAS_VIENTO]) + 0.01
        s_perturbacion = min(1.0, energia_viento / energia_total)
        
        s_no_intervenir = self.acoplamiento * (1.0 - self.delta_struct)
        s_contexto = 1.0 - s_perturbacion
        
        return {
            'S_zona_voz': s_zona_voz,
            'S_perturbacion_critica': s_perturbacion,
            'S_no_intervenir': s_no_intervenir,
            'S_contexto_integrable': s_contexto
        }
    
    def evaluar_modo(self, signos):
        sz = signos['S_zona_voz']
        sp = signos['S_perturbacion_critica']
        sn = signos['S_no_intervenir']
        sc = signos['S_contexto_integrable']
        
        score_zona_voz = 0.5 * sz + 0.3 * self.acoplamiento + 0.2 * (1.0 - sp)
        score_discrepo = 0.4 * sp + 0.3 * self.delta_struct + 0.3 * sc
        score_no_se = 0.5 * sp + 0.3 * (1.0 - self.acoplamiento) + 0.2 * self.error
        score_y_si = 0.5 * self.lf + 0.5 * (1.0 - abs(sz - sp))
        
        scores = {
            'ZONA_VOZ': score_zona_voz,
            'DISCREPO': score_discrepo,
            'NO_SE': score_no_se,
            'Y_SI': score_y_si
        }
        candidato = max(scores, key=scores.get)
        
        if candidato == self.modo_candidato:
            self.persistencia += 1
        else:
            self.modo_candidato = candidato
            self.persistencia = 1
        
        if self.persistencia >= 3:
            self.modo_actual = candidato
        
        return self.modo_actual
    
    def plan_permitido(self, plan, modo):
        if modo == 'ZONA_VOZ':
            return plan in ['NADA', 'PRESERVAR']
        elif modo == 'DISCREPO':
            return True
        elif modo == 'NO_SE':
            return plan == 'PRESERVAR'
        elif modo == 'Y_SI':
            return plan in ['NADA', 'PRESERVAR']
        return plan == 'NADA'
    
    def calcular_viabilidad(self, plan, signos):
        if plan == 'NADA':
            v = 1.0 - 0.05 * self.tiempo_exposicion / 50.0
            if self.acoplamiento > 0.7:
                v = min(1.0, v + 0.2)
        elif plan == 'PRESERVAR':
            costo = 0.3 + 0.5 * (1.0 - self.energia_computacion)
            v = max(0.0, 1.0 - costo)
            if self.modo_actual == 'NO_SE':
                v = min(1.0, v + 0.3)
        else:  # CONTEXTO
            v = 0.8 * self.energia_computacion
            v += 0.2 * signos['S_contexto_integrable']
            v -= 0.3 * signos['S_perturbacion_critica']
            v = max(0.0, min(1.0, v))
        return v
    
    def calcular_coherencia(self, plan, signos, modo):
        if plan == 'NADA':
            c = 1.0 - 0.3 * signos['S_no_intervenir']
            if modo == 'ZONA_VOZ':
                c = min(1.0, c + 0.2)
        elif plan == 'PRESERVAR':
            c = 1.0 - 0.2 * signos['S_perturbacion_critica']
            if modo == 'NO_SE':
                c = min(1.0, c + 0.3)
        else:  # CONTEXTO
            c = 1.0 - 0.7 * signos['S_zona_voz'] - 0.5 * signos['S_perturbacion_critica']
            c = max(0.0, min(1.0, c))
        return c
    
    def elegir_plan(self, signos, modo):
        mejor_j = -1.0
        mejor_plan = 'NADA'
        
        for plan in ['NADA', 'PRESERVAR', 'CONTEXTO']:
            if not self.plan_permitido(plan, modo):
                continue
            v = self.calcular_viabilidad(plan, signos)
            c = self.calcular_coherencia(plan, signos, modo)
            j = self.alfa * v + (1.0 - self.alfa) * c
            if j > mejor_j:
                mejor_j = j
                mejor_plan = plan
        
        return mejor_plan, mejor_j
    
    def actualizar_mezcla(self, plan):
        obj_nada = 1.0 if plan == 'NADA' else 0.0
        obj_pres = 1.0 if plan == 'PRESERVAR' else 0.0
        obj_ctx = 1.0 if plan == 'CONTEXTO' else 0.0
        
        k = 0.1 - 0.05 * self.lf
        k = max(0.02, min(0.1, k))
        
        self.mezcla_nada += k * (obj_nada - self.mezcla_nada)
        self.mezcla_preservar += k * (obj_pres - self.mezcla_preservar)
        self.mezcla_contexto += k * (obj_ctx - self.mezcla_contexto)
        
        suma = self.mezcla_nada + self.mezcla_preservar + self.mezcla_contexto + 1e-6
        self.mezcla_nada /= suma
        self.mezcla_preservar /= suma
        self.mezcla_contexto /= suma
    
    def procesar_ventana(self, espectro):
        self.actualizar_campo(espectro)
        self.calcular_emergentes()
        signos = self.calcular_signos()
        modo = self.evaluar_modo(signos)
        plan, j = self.elegir_plan(signos, modo)
        self.actualizar_mezcla(plan)
        
        if self.error > 0.5:
            self.tiempo_exposicion += 1
        else:
            self.tiempo_exposicion = max(0, self.tiempo_exposicion - 1)
        
        self.historial.append({
            'ventana': len(self.historial),
            'acoplamiento': self.acoplamiento,
            'error': self.error,
            'lf': self.lf,
            'delta_struct': self.delta_struct,
            'icr': self.icr,
            'irde': self.irde,
            'modo': modo,
            'plan': plan,
            'j': j,
            'mezcla_nada': self.mezcla_nada,
            'mezcla_preservar': self.mezcla_preservar,
            'mezcla_contexto': self.mezcla_contexto,
            'S_zona_voz': signos['S_zona_voz'],
            'S_perturbacion': signos['S_perturbacion_critica']
        })
    
    def esta_vivo(self):
        if len(self.historial) < 20:
            return True
        
        # Solo considerar ventanas con actividad vocal
        ventanas_con_voz = []
        for h in self.historial[-50:]:
            if h['S_zona_voz'] > 0.2:
                ventanas_con_voz.append(h)
        
        if len(ventanas_con_voz) < 5:
            return True  # Poca voz, no se puede evaluar colapso
        
        acops = [h['acoplamiento'] for h in ventanas_con_voz]
        if np.mean(acops) < 0.35:
            return False
        return True
    
    def resumen(self):
        if not self.historial:
            return {'vivo': False, 'modo_final': 'DESCONOCIDO', 'plan_final': 'NADA'}
        
        return {
            'vivo': self.esta_vivo(),
            'acoplamiento_final': self.historial[-1]['acoplamiento'],
            'error_final': self.historial[-1]['error'],
            'modo_final': self.historial[-1]['modo'],
            'plan_final': self.historial[-1]['plan']
        }

# ============================================================
# EXPERIMENTO
# ============================================================

def main():
    print("=" * 60)
    print("EXPERIMENTO COSMOSEMIÓTICO v2")
    print("=" * 60)
    
    # Cargar archivos
    print("\n[1] Cargando archivos...")
    sr, voz_estudio = cargar_wav_seguro('Voz_Estudio.wav')
    _, voz_viento1 = cargar_wav_seguro('Voz+Viento_1.wav')
    _, voz_viento2 = cargar_wav_seguro('Voz+Viento_2.wav')
    _, viento_puro = cargar_wav_seguro('Viento.wav')
    
    print(f"  Frecuencia: {sr} Hz")
    
    # Recortar a la misma longitud
    min_len = min(len(voz_estudio), len(voz_viento1), len(voz_viento2), len(viento_puro))
    voz_estudio = voz_estudio[:min_len]
    voz_viento1 = voz_viento1[:min_len]
    voz_viento2 = voz_viento2[:min_len]
    viento_puro = viento_puro[:min_len]
    
    # Calcular perfil de referencia desde voz limpia
    print("\n[2] Calibrando referencia de voz...")
    espectro_ref = espectrograma(voz_estudio, VENTANA_MUESTRAS, HOP_MUESTRAS)
    perfil_voz = np.mean(espectro_ref, axis=1)
    print(f"  Perfil de voz calculado ({len(perfil_voz)} bandas)")
    
    # Experimentos
    print("\n[3] Ejecutando experimentos...")
    resultados = []
    
    configs = [
        ("Viento Puro (control)", viento_puro, 0.5, 0.9),
        ("Voz + Viento 1", voz_viento1, 0.5, 0.9),
        ("Voz + Viento 2", voz_viento2, 0.5, 0.9),
        ("Voz + Viento 2 α=0.0", voz_viento2, 0.0, 0.9),
        ("Voz + Viento 2 α=1.0", voz_viento2, 1.0, 0.9),
        ("Voz + Viento 2 E=0.2", voz_viento2, 0.5, 0.2),
    ]
    
    for nombre, audio, alfa, energia in configs:
        print(f"\n--- {nombre} | α={alfa} | E={energia} ---")
        
        espectro = espectrograma(audio, VENTANA_MUESTRAS, HOP_MUESTRAS)
        n_ventanas = espectro.shape[1]
        
        sistema = SistemaCosmosemiotico(alfa=alfa, energia_computacion=energia,
                                        perfil_voz_referencia=perfil_voz)
        
        for i in range(n_ventanas):
            sistema.procesar_ventana(espectro[:, i:i+1])
        
        res = sistema.resumen()
        resultados.append({
            'nombre': nombre,
            'vivo': res['vivo'],
            'modo': res['modo_final'],
            'plan': res['plan_final']
        })
        
        print(f"  Resultado: {'VIVO' if res['vivo'] else 'COLAPSADO'}")
        print(f"  Modo final: {res['modo_final']}, Plan final: {res['plan_final']}")
        
        # Gráfico
        h = sistema.historial
        t = np.arange(len(h)) * HOP_MS / 1000
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        axes[0].plot(t, [x['acoplamiento'] for x in h], 'g-', label='Acoplamiento')
        axes[0].axhline(y=0.35, color='r', linestyle='--')
        axes[0].set_ylabel('Acoplamiento')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(t, [x['mezcla_preservar'] for x in h], 'b-', label='Preservar')
        axes[1].plot(t, [x['mezcla_contexto'] for x in h], 'r-', label='Contexto')
        axes[1].plot(t, [x['mezcla_nada'] for x in h], 'g-', label='Nada')
        axes[1].set_ylabel('Mezcla')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{nombre} (α={alfa}, E={energia})')
        plt.tight_layout()
        plt.savefig(f'resultado_{nombre.replace(" ", "_").replace("=", "")}.png', dpi=150)
        plt.close()
    
    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    for r in resultados:
        vivo = "✅ VIVO" if r['vivo'] else "❌ COLAPSADO"
        print(f"{r['nombre']:35} {vivo}  Modo:{r['modo']:10} Plan:{r['plan']}")
    
    print("\nArchivos generados: 6 gráficos resultado_*.png")

if __name__ == "__main__":
    main()