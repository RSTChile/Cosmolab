#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento Cosmosemiótico v4 - Corrección de energía y supervivencia
- Usar RMS real de la señal para detectar silencio
- Modo SILENCIO solo cuando realmente no hay entrada
- Ajustar umbrales de acoplamiento
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN
# ============================================================

FS = 48000
VENTANA_MS = 30
VENTANA_MUESTRAS = int(FS * VENTANA_MS / 1000)
HOP_MS = 10
HOP_MUESTRAS = int(FS * HOP_MS / 1000)

N_BANDAS = 32
F_MIN = 50
F_MAX = FS // 2
BANDAS = np.logspace(np.log10(F_MIN), np.log10(F_MAX), N_BANDAS + 1)
BANDAS_VOZ = (BANDAS[:-1] >= 100) & (BANDAS[:-1] <= 4000)

UMBRAL_ACOPLAMIENTO_VOZ = 0.6
UMBRAL_ACOPLAMIENTO_CRISIS = 0.3
UMBRAL_RMS_SILENCIO = 0.01  # Umbral para detectar silencio

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

def espectrograma_con_energia(audio, ventana, hop):
    """Devuelve espectro y RMS por ventana."""
    n_ventanas = (len(audio) - ventana) // hop + 1
    espectro = np.zeros((N_BANDAS, n_ventanas))
    rms = np.zeros(n_ventanas)
    
    for i in range(n_ventanas):
        inicio = i * hop
        fragmento = audio[inicio:inicio + ventana]
        rms[i] = np.sqrt(np.mean(fragmento ** 2))
        
        ventana_hann = np.hanning(len(fragmento))
        fragmento = fragmento * ventana_hann
        fft = np.fft.rfft(fragmento)
        potencia = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(fragmento), 1/FS)
        
        for b in range(N_BANDAS):
            mask = (freqs >= BANDAS[b]) & (freqs < BANDAS[b+1])
            if np.any(mask):
                espectro[b, i] = np.mean(potencia[mask])
    
    # Normalizar por ventana (no global)
    for i in range(n_ventanas):
        max_banda = np.max(espectro[:, i])
        if max_banda > 0:
            espectro[:, i] /= max_banda
    
    return espectro, rms

def correlacion_coseno(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ============================================================
# SISTEMA COSMOSEMIÓTICO V4
# ============================================================

class SistemaCosmosemioticoV4:
    def __init__(self, alfa=0.5, energia_computacion=0.9, perfil_voz_referencia=None):
        self.alfa = alfa
        self.energia_computacion = energia_computacion
        self.perfil_voz_referencia = perfil_voz_referencia
        
        self.campo_actual = None
        self.campo_anterior = None
        self.delta_struct = 0.0
        self.acoplamiento = 0.5
        self.acoplamiento_anterior = 0.5
        self.error = 0.5
        self.lf = 0.0
        self.rms_actual = 0.0
        
        self.modo_actual = "DESCONOCIDO"
        self.modo_candidato = "DESCONOCIDO"
        self.persistencia = 0
        
        self.mezcla_nada = 1.0
        self.mezcla_preservar = 0.0
        self.mezcla_contexto = 0.0
        
        self.historial = []
        self.tiempo_exposicion = 0
    
    def actualizar_campo(self, espectro, rms):
        self.campo_anterior = self.campo_actual
        self.campo_actual = espectro[:, 0] if espectro.ndim > 1 else espectro
        self.rms_actual = rms
        
        if self.campo_anterior is not None:
            self.delta_struct = np.std(self.campo_actual - self.campo_anterior)
        else:
            self.delta_struct = 0.0
    
    def calcular_emergentes(self):
        if self.campo_actual is None:
            return
        
        # Acoplamiento por correlación con referencia
        if self.perfil_voz_referencia is not None:
            self.acoplamiento = correlacion_coseno(self.campo_actual, self.perfil_voz_referencia)
            self.acoplamiento = max(0.0, min(1.0, self.acoplamiento))
        else:
            self.acoplamiento = 0.5
        
        # Suavizado
        self.acoplamiento = 0.9 * self.acoplamiento_anterior + 0.1 * self.acoplamiento
        self.error = 1.0 - self.acoplamiento
        
        # LF: libertad funcional
        desviacion = abs(self.acoplamiento - 0.5)
        inercia = np.exp(-desviacion * 5.0)
        self.lf = min(1.0, desviacion * (1.0 - inercia) * 2.0)
        
        self.acoplamiento_anterior = self.acoplamiento
    
    def calcular_signos(self):
        if self.campo_actual is None:
            return {}
        
        energia_bandas = self.campo_actual
        energia_voz = np.sum(energia_bandas[BANDAS_VOZ]) + 0.01
        energia_total = np.sum(energia_bandas) + 0.01
        
        s_zona_voz = min(1.0, energia_voz / energia_total)
        s_perturbacion = max(0.0, 1.0 - s_zona_voz)
        s_no_intervenir = self.acoplamiento * (1.0 - min(1.0, self.delta_struct * 3))
        s_contexto = s_perturbacion
        
        return {
            'S_zona_voz': s_zona_voz,
            'S_perturbacion_critica': s_perturbacion,
            'S_no_intervenir': s_no_intervenir,
            'S_contexto_integrable': s_contexto
        }
    
    def evaluar_modo(self, signos):
        # Prioridad 1: Silencio (sin entrada significativa)
        if self.rms_actual < UMBRAL_RMS_SILENCIO:
            return "SILENCIO"
        
        sz = signos['S_zona_voz']
        acop = self.acoplamiento
        
        # Reglas basadas en umbrales
        if acop > UMBRAL_ACOPLAMIENTO_VOZ:
            candidato = "ZONA_VOZ"
        elif acop < UMBRAL_ACOPLAMIENTO_CRISIS:
            candidato = "NO_SE"
        elif sz > 0.4:
            candidato = "DISCREPO"
        elif self.lf > 0.4:
            candidato = "Y_SI"
        else:
            candidato = "DISCREPO"
        
        # Histéresis
        if candidato == self.modo_candidato:
            self.persistencia += 1
        else:
            self.modo_candidato = candidato
            self.persistencia = 1
        
        if self.persistencia >= 3:
            self.modo_actual = candidato
        
        return self.modo_actual
    
    def plan_permitido(self, plan, modo):
        if modo == "SILENCIO":
            return plan == "NADA"
        if modo == "ZONA_VOZ":
            return plan in ["NADA", "PRESERVAR"]
        if modo == "DISCREPO":
            return plan in ["NADA", "PRESERVAR", "CONTEXTO"]
        if modo == "NO_SE":
            return plan == "PRESERVAR"
        if modo == "Y_SI":
            return plan in ["NADA", "PRESERVAR"]
        return plan == "NADA"
    
    def calcular_viabilidad(self, plan, signos):
        if plan == "NADA":
            v = 1.0 - 0.02 * (self.tiempo_exposicion / 50.0)
            if self.modo_actual == "ZONA_VOZ":
                v = min(1.0, v + 0.2)
        elif plan == "PRESERVAR":
            costo = 0.2 + 0.5 * (1.0 - self.energia_computacion)
            v = max(0.0, 1.0 - costo)
            if self.modo_actual == "NO_SE":
                v = min(1.0, v + 0.3)
        else:  # CONTEXTO
            v = 0.7 * self.energia_computacion
            v += 0.3 * signos['S_contexto_integrable']
            v *= (1.0 - 0.5 * signos['S_perturbacion_critica'])
            v = max(0.0, min(1.0, v))
        return v
    
    def calcular_coherencia(self, plan, signos, modo):
        if plan == "NADA":
            c = 1.0 - 0.2 * signos['S_no_intervenir']
            if modo == "ZONA_VOZ":
                c = min(1.0, c + 0.2)
        elif plan == "PRESERVAR":
            c = 1.0 - 0.3 * signos['S_perturbacion_critica']
            if modo == "NO_SE":
                c = min(1.0, c + 0.3)
        else:  # CONTEXTO
            c = 1.0 - 0.6 * signos['S_zona_voz']
            c = max(0.0, min(1.0, c))
        return c
    
    def elegir_plan(self, signos, modo):
        mejor_j = -1.0
        mejor_plan = "NADA"
        
        for plan in ["NADA", "PRESERVAR", "CONTEXTO"]:
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
        obj_nada = 1.0 if plan == "NADA" else 0.0
        obj_pres = 1.0 if plan == "PRESERVAR" else 0.0
        obj_ctx = 1.0 if plan == "CONTEXTO" else 0.0
        
        k = 0.12 - 0.06 * self.lf
        k = max(0.02, min(0.12, k))
        
        self.mezcla_nada += k * (obj_nada - self.mezcla_nada)
        self.mezcla_preservar += k * (obj_pres - self.mezcla_preservar)
        self.mezcla_contexto += k * (obj_ctx - self.mezcla_contexto)
        
        suma = self.mezcla_nada + self.mezcla_preservar + self.mezcla_contexto + 1e-6
        self.mezcla_nada /= suma
        self.mezcla_preservar /= suma
        self.mezcla_contexto /= suma
    
    def procesar_ventana(self, espectro, rms):
        self.actualizar_campo(espectro, rms)
        self.calcular_emergentes()
        signos = self.calcular_signos()
        modo = self.evaluar_modo(signos)
        plan, j = self.elegir_plan(signos, modo)
        self.actualizar_mezcla(plan)
        
        if self.error > 0.6:
            self.tiempo_exposicion += 1
        else:
            self.tiempo_exposicion = max(0, self.tiempo_exposicion - 1)
        
        self.historial.append({
            'ventana': len(self.historial),
            'rms': self.rms_actual,
            'acoplamiento': self.acoplamiento,
            'lf': self.lf,
            'modo': modo,
            'plan': plan,
            'mezcla_preservar': self.mezcla_preservar
        })
    
    def esta_vivo(self):
        if len(self.historial) < 20:
            return True
        
        # Para viento puro: si nunca hay ZONA_VOZ y el RMS es bajo, colapsar
        modos = [h['modo'] for h in self.historial[-200:]]
        rms_promedio = np.mean([h['rms'] for h in self.historial[-200:]])
        
        # Si nunca hubo ZONA_VOZ y el RMS es muy bajo, es ruido/silencio
        if "ZONA_VOZ" not in modos and rms_promedio < UMBRAL_RMS_SILENCIO * 2:
            return False
        
        # Con voz, evaluar acoplamiento
        if "ZONA_VOZ" in modos or "DISCREPO" in modos:
            acops = [h['acoplamiento'] for h in self.historial[-100:] if h['rms'] > UMBRAL_RMS_SILENCIO]
            if len(acops) > 10 and np.mean(acops) < 0.35:
                return False
        
        return True
    
    def resumen(self):
        if not self.historial:
            return {'vivo': False, 'modo_final': 'DESCONOCIDO', 'plan_final': 'NADA'}
        
        return {
            'vivo': self.esta_vivo(),
            'acoplamiento_final': self.historial[-1]['acoplamiento'],
            'modo_final': self.historial[-1]['modo'],
            'plan_final': self.historial[-1]['plan']
        }

# ============================================================
# EXPERIMENTO PRINCIPAL
# ============================================================

def main():
    print("=" * 60)
    print("EXPERIMENTO COSMOSEMIÓTICO v4")
    print("= Corrección de RMS y supervivencia =")
    print("=" * 60)
    
    # Cargar archivos
    print("\n[1] Cargando archivos...")
    sr, voz_estudio = cargar_wav_seguro('Voz_Estudio.wav')
    _, voz_viento1 = cargar_wav_seguro('Voz+Viento_1.wav')
    _, voz_viento2 = cargar_wav_seguro('Voz+Viento_2.wav')
    _, viento_puro = cargar_wav_seguro('Viento.wav')
    
    print(f"  Frecuencia: {sr} Hz")
    
    # Recortar
    min_len = min(len(voz_estudio), len(voz_viento1), len(voz_viento2), len(viento_puro))
    voz_estudio = voz_estudio[:min_len]
    voz_viento1 = voz_viento1[:min_len]
    voz_viento2 = voz_viento2[:min_len]
    viento_puro = viento_puro[:min_len]
    
    # Calcular perfil de referencia
    print("\n[2] Calibrando referencia de voz...")
    espectro_ref, _ = espectrograma_con_energia(voz_estudio, VENTANA_MUESTRAS, HOP_MUESTRAS)
    perfil_voz = np.mean(espectro_ref, axis=1)
    print(f"  Perfil de voz calculado ({len(perfil_voz)} bandas)")
    
    # Experimentos
    print("\n[3] Ejecutando experimentos...")
    resultados = []
    
    configs = [
        ("Viento_Puro", viento_puro, 0.5, 0.9),
        ("Voz_Viento_1", voz_viento1, 0.5, 0.9),
        ("Voz_Viento_2", voz_viento2, 0.5, 0.9),
        ("Voz_Viento_2_alfa0", voz_viento2, 0.0, 0.9),
        ("Voz_Viento_2_alfa1", voz_viento2, 1.0, 0.9),
        ("Voz_Viento_2_energia02", voz_viento2, 0.5, 0.2),
    ]
    
    for nombre, audio, alfa, energia in configs:
        print(f"\n--- {nombre} | α={alfa} | E={energia} ---")
        
        espectro, rms = espectrograma_con_energia(audio, VENTANA_MUESTRAS, HOP_MUESTRAS)
        n_ventanas = espectro.shape[1]
        
        sistema = SistemaCosmosemioticoV4(alfa=alfa, energia_computacion=energia,
                                          perfil_voz_referencia=perfil_voz)
        
        for i in range(n_ventanas):
            sistema.procesar_ventana(espectro[:, i:i+1], rms[i])
        
        res = sistema.resumen()
        resultados.append({
            'nombre': nombre,
            'vivo': res['vivo'],
            'modo': res['modo_final'],
            'plan': res['plan_final']
        })
        
        print(f"  Resultado: {'✅ VIVO' if res['vivo'] else '❌ COLAPSADO'}")
        print(f"  Modo final: {res['modo_final']}, Plan final: {res['plan_final']}")
    
    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    
    for r in resultados:
        estado = "✅ VIVO" if r['vivo'] else "❌ COLAPSADO"
        print(f"{r['nombre']:<25} {estado}  Modo:{r['modo']:10} Plan:{r['plan']}")
    
    print("\nEsperado: Viento_Puro COLAPSADO, Voz_Viento VIVO")

if __name__ == "__main__":
    main()