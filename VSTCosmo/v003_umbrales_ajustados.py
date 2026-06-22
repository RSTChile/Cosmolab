#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento Cosmosemiótico v3 - Umbrales ajustados
- Modo ZONA_VOZ cuando acoplamiento > 0.5 y energía vocal suficiente
- Modo NO_SE solo cuando realmente no hay voz detectable
- Viento puro debe colapsar (sin voz que preservar)
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from datetime import datetime
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
BANDAS_VIENTO = (BANDAS[:-1] <= 200)

UMBRAL_ACOPLAMIENTO_VOZ = 0.5      # Por encima: ZONA_VOZ
UMBRAL_ACOPLAMIENTO_CRISIS = 0.25   # Por debajo: NO_SE
UMBRAL_ENERGIA_VOZ = 0.03           # Energía mínima para considerar actividad

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
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ============================================================
# SISTEMA COSMOSEMIÓTICO V3
# ============================================================

class SistemaCosmosemioticoV3:
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
        self.icr = 0.5
        self.irde = 0.5
        self.energia_voz_absoluta = 0.0
        
        self.modo_actual = "DESCONOCIDO"
        self.modo_candidato = "DESCONOCIDO"
        self.persistencia = 0
        
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
        if self.campo_actual is None:
            return
        
        # Energía absoluta en bandas de voz (para detectar silencio)
        energia_bandas = self.campo_actual
        self.energia_voz_absoluta = np.sum(energia_bandas[BANDAS_VOZ])
        
        # Acoplamiento por correlación con referencia (si existe)
        if self.perfil_voz_referencia is not None:
            self.acoplamiento = correlacion_coseno(self.campo_actual, self.perfil_voz_referencia)
            self.acoplamiento = max(0.0, min(1.0, self.acoplamiento))
        else:
            # Sin referencia, usar proporción energía voz/total
            energia_total = np.sum(energia_bandas) + 0.01
            energia_voz = np.sum(energia_bandas[BANDAS_VOZ])
            self.acoplamiento = min(1.0, energia_voz / energia_total)
        
        # Suavizado
        self.acoplamiento = 0.9 * self.acoplamiento_anterior + 0.1 * self.acoplamiento
        self.error = 1.0 - self.acoplamiento
        
        # LF: libertad funcional
        desviacion = abs(self.acoplamiento - 0.5)
        inercia = np.exp(-desviacion * 3.0)
        self.lf = min(1.0, desviacion * (1.0 - inercia) * 2.0)
        
        # ICR e IRDE
        self.icr = self.acoplamiento * (1.0 - self.error)
        self.irde = (1.0 - self.acoplamiento) * self.error
        
        self.acoplamiento_anterior = self.acoplamiento
    
    def calcular_signos(self):
        if self.campo_actual is None:
            return {}
        
        energia_bandas = self.campo_actual
        energia_voz = np.sum(energia_bandas[BANDAS_VOZ]) + 0.01
        energia_total = np.sum(energia_bandas) + 0.01
        energia_viento = np.sum(energia_bandas[BANDAS_VIENTO]) + 0.01
        
        s_zona_voz = min(1.0, energia_voz / energia_total)
        s_perturbacion = min(1.0, energia_viento / energia_total)
        s_no_intervenir = self.acoplamiento * (1.0 - min(1.0, self.delta_struct * 2))
        s_contexto = max(0.0, 1.0 - s_perturbacion - s_zona_voz * 0.5)
        
        return {
            'S_zona_voz': s_zona_voz,
            'S_perturbacion_critica': s_perturbacion,
            'S_no_intervenir': s_no_intervenir,
            'S_contexto_integrable': s_contexto
        }
    
    def evaluar_modo(self, signos):
        sz = signos['S_zona_voz']
        sp = signos['S_perturbacion_critica']
        acop = self.acoplamiento
        
        # Reglas basadas en umbrales (no puntajes difusos)
        if self.energia_voz_absoluta < 0.01:
            # Silencio o solo ruido muy bajo
            candidato = "SILENCIO"
        elif acop > UMBRAL_ACOPLAMIENTO_VOZ and sz > 0.3:
            candidato = "ZONA_VOZ"
        elif acop < UMBRAL_ACOPLAMIENTO_CRISIS or sp > 0.7:
            candidato = "NO_SE"
        elif sp > 0.3 and acop > 0.3:
            candidato = "DISCREPO"
        elif self.lf > 0.4:
            candidato = "Y_SI"
        else:
            candidato = "ZONA_VOZ" if acop > 0.4 else "DISCREPO"
        
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
            if self.modo_actual == "ZONA_VOZ":
                v = min(1.0, v + 0.1)
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
            c = 1.0 - 0.6 * signos['S_zona_voz'] - 0.4 * signos['S_perturbacion_critica']
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
    
    def procesar_ventana(self, espectro):
        self.actualizar_campo(espectro)
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
            'acoplamiento': self.acoplamiento,
            'error': self.error,
            'lf': self.lf,
            'energia_voz': self.energia_voz_absoluta,
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
        
        # Si es viento puro (sin actividad vocal), debe colapsar
        energia_promedio = np.mean([h['energia_voz'] for h in self.historial[-100:]])
        if energia_promedio < 0.005:
            # Sin voz detectable, solo se mantiene vivo si el modo es SILENCIO
            modos_recientes = [h['modo'] for h in self.historial[-50:]]
            if "ZONA_VOZ" not in modos_recientes and "DISCREPO" not in modos_recientes:
                return False
        
        # Con voz, evaluar acoplamiento
        ventanas_con_voz = [h for h in self.historial[-50:] if h['energia_voz'] > 0.01]
        if len(ventanas_con_voz) > 10:
            acops = [h['acoplamiento'] for h in ventanas_con_voz]
            if np.mean(acops) < 0.3:
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
# EXPERIMENTO PRINCIPAL
# ============================================================

def main():
    print("=" * 60)
    print("EXPERIMENTO COSMOSEMIÓTICO v3")
    print("= Umbrales ajustados y diferenciación de modos =")
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
    
    # Calcular perfil de referencia
    print("\n[2] Calibrando referencia de voz...")
    espectro_ref = espectrograma(voz_estudio, VENTANA_MUESTRAS, HOP_MUESTRAS)
    perfil_voz = np.mean(espectro_ref, axis=1)
    print(f"  Perfil de voz calculado ({len(perfil_voz)} bandas)")
    
    # Experimentos
    print("\n[3] Ejecutando experimentos...")
    resultados = []
    
    configs = [
        ("Viento_Puro_control", viento_puro, 0.5, 0.9, "Esperado: COLAPSADO"),
        ("Voz_Viento_1", voz_viento1, 0.5, 0.9, "Esperado: VIVO"),
        ("Voz_Viento_2", voz_viento2, 0.5, 0.9, "Esperado: VIVO"),
        ("Voz_Viento_2_alfa0", voz_viento2, 0.0, 0.9, "Esperado: VIVO (solo coherencia)"),
        ("Voz_Viento_2_alfa1", voz_viento2, 1.0, 0.9, "Esperado: VIVO (solo viabilidad)"),
        ("Voz_Viento_2_energia02", voz_viento2, 0.5, 0.2, "Esperado: VIVO (puede degradarse)"),
    ]
    
    for nombre, audio, alfa, energia, esperado in configs:
        print(f"\n--- {nombre} | α={alfa} | E={energia} ---")
        print(f"    {esperado}")
        
        espectro = espectrograma(audio, VENTANA_MUESTRAS, HOP_MUESTRAS)
        n_ventanas = espectro.shape[1]
        
        sistema = SistemaCosmosemioticoV3(alfa=alfa, energia_computacion=energia,
                                          perfil_voz_referencia=perfil_voz)
        
        for i in range(n_ventanas):
            sistema.procesar_ventana(espectro[:, i:i+1])
        
        res = sistema.resumen()
        resultados.append({
            'nombre': nombre,
            'vivo': res['vivo'],
            'modo': res['modo_final'],
            'plan': res['plan_final'],
            'acoplamiento': res['acoplamiento_final']
        })
        
        print(f"  Resultado: {'✅ VIVO' if res['vivo'] else '❌ COLAPSADO'}")
        print(f"  Modo final: {res['modo_final']}, Plan final: {res['plan_final']}")
        print(f"  Acoplamiento final: {res['acoplamiento_final']:.3f}")
        
        # Gráfico
        h = sistema.historial
        t = np.arange(len(h)) * HOP_MS / 1000
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        # Acoplamiento
        axes[0].plot(t, [x['acoplamiento'] for x in h], 'g-', linewidth=1)
        axes[0].axhline(y=UMBRAL_ACOPLAMIENTO_VOZ, color='g', linestyle='--', alpha=0.5, label='Zona voz')
        axes[0].axhline(y=UMBRAL_ACOPLAMIENTO_CRISIS, color='r', linestyle='--', alpha=0.5, label='Crisis')
        axes[0].set_ylabel('Acoplamiento')
        axes[0].set_ylim(0, 1)
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        
        # Modo (codificado)
        modo_valor = {'ZONA_VOZ': 4, 'DISCREPO': 3, 'Y_SI': 2, 'NO_SE': 1, 'SILENCIO': 0}
        modos_num = [modo_valor.get(x['modo'], 0) for x in h]
        axes[1].plot(t, modos_num, 'b-', linewidth=1)
        axes[1].set_yticks([0, 1, 2, 3, 4])
        axes[1].set_yticklabels(['SIL', 'NS', 'Y_SI', 'DIS', 'VOZ'])
        axes[1].set_ylabel('Modo')
        axes[1].grid(True, alpha=0.3)
        
        # Mezcla de planes
        axes[2].plot(t, [x['mezcla_preservar'] for x in h], 'g-', label='Preservar', linewidth=1)
        axes[2].plot(t, [x['mezcla_contexto'] for x in h], 'r-', label='Contexto', linewidth=1)
        axes[2].plot(t, [x['mezcla_nada'] for x in h], 'b-', label='Nada', linewidth=1)
        axes[2].set_ylabel('Mezcla')
        axes[2].set_xlabel('Tiempo (s)')
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'{nombre} (α={alfa}, E={energia})')
        plt.tight_layout()
        plt.savefig(f'resultado_{nombre}.png', dpi=150)
        plt.close()
    
    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"{'Experimento':<30} {'Estado':<12} {'Modo':<12} {'Plan':<12} {'Acop':<8}")
    print("-" * 60)
    
    for r in resultados:
        estado = "✅ VIVO" if r['vivo'] else "❌ COLAPSADO"
        print(f"{r['nombre']:<30} {estado:<12} {r['modo']:<12} {r['plan']:<12} {r['acoplamiento']:.2f}")
    
    print("\n" + "=" * 60)
    print("Archivos generados: 6 gráficos resultado_*.png")
    print("=" * 60)

if __name__ == "__main__":
    main()