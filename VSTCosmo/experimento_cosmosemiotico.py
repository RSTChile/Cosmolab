#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento Cosmosemiótico - Validación con grabaciones reales
Autor: Equipo Transinteligente
Fecha: Abril 2026

Ejecutar: python3 experimento_cosmosemiotico.py
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

# Parámetros del sistema
FS = 48000  # Frecuencia de muestreo (Hz)
VENTANA_MS = 30  # Ventana de análisis (ms)
VENTANA_MUESTRAS = int(FS * VENTANA_MS / 1000)
HOP_MS = 10  # Desplazamiento (ms)
HOP_MUESTRAS = int(FS * HOP_MS / 1000)

# Bandas de frecuencia (64 bandas logarítmicas)
N_BANDAS = 64
F_MIN = 20  # Hz
F_MAX = FS // 2  # Hz
BANDAS = np.logspace(np.log10(F_MIN), np.log10(F_MAX), N_BANDAS + 1)
BANDAS_VOZ = (BANDAS[:-1] >= 100) & (BANDAS[:-1] <= 4000)
BANDAS_VIENTO = (BANDAS[:-1] <= 200)

# Umbrales para supervivencia
UMBRAL_ACOPLAMIENTO = 0.3
UMBRAL_ERROR = 0.6

# Parámetros modales
PERSISTENCIA_CAMBIO = 3  # ciclos para confirmar cambio de modo
PERSISTENCIA_ESTABLE = 4

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def a_mono(audio):
    """Convierte estéreo a mono promediando canales."""
    if audio.ndim == 1:
        return audio.astype(np.float32)
    elif audio.ndim == 2:
        return audio.mean(axis=1).astype(np.float32)
    return audio

def normalizar(audio):
    """Normaliza a rango [-1, 1]."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio

def espectrograma(audio, ventana, hop):
    """Calcula espectrograma de potencia."""
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
        
        # Integrar por bandas logarítmicas
        for b in range(N_BANDAS):
            mask = (freqs >= BANDAS[b]) & (freqs < BANDAS[b+1])
            if np.any(mask):
                espectro[b, i] = np.mean(potencia[mask])
            else:
                espectro[b, i] = 0
    
    # Normalizar por banda
    for b in range(N_BANDAS):
        max_b = np.max(espectro[b, :])
        if max_b > 0:
            espectro[b, :] /= max_b
    
    return espectro

# ============================================================
# SISTEMA COSMOSEMIÓTICO
# ============================================================

class SistemaCosmosemiotico:
    def __init__(self, alfa=0.5, energia_computacion=0.9, referencia_voz=None):
        self.alfa = alfa
        self.energia_computacion = energia_computacion
        self.referencia_voz = referencia_voz
        
        # Estado del sistema
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
        
        # Modo actual
        self.modo_actual = "DESCONOCIDO"
        self.modo_candidato = "DESCONOCIDO"
        self.persistencia = 0
        
        # Mezcla de planes
        self.mezcla_nada = 1.0
        self.mezcla_preservar = 0.0
        self.mezcla_contexto = 0.0
        
        # Historial
        self.historial = []
        
        # Contadores
        self.tiempo_exposicion = 0
        
        # Perfiles (señales 1D, no espectros)
        self.perfil_viento_1d = None
        self.perfil_voz_1d = None
    
    def calibrar_viento(self, viento_puro_audio):
        """Calibra el perfil de perturbación a partir de la señal de viento puro."""
        # Guardar la señal completa para referencia
        self.perfil_viento_1d = viento_puro_audio[:min(FS, len(viento_puro_audio))]
    
    def calibrar_voz(self, voz_limpia_audio):
        """Calibra el perfil de voz a partir de la señal de voz limpia."""
        self.perfil_voz_1d = voz_limpia_audio[:min(FS, len(voz_limpia_audio))]
    
    def actualizar_campo(self, espectro):
        """Actualiza el campo acústico."""
        self.campo_anterior = self.campo_actual
        self.campo_actual = espectro
        
        if self.campo_anterior is not None:
            self.delta_struct = np.std(self.campo_actual - self.campo_anterior)
        else:
            self.delta_struct = 0.0
    
    def calcular_emergentes(self):
        """Calcula variables emergentes: acoplamiento, error, LF, etc."""
        if self.campo_actual is None:
            return
        
        # Energía en bandas de voz vs total
        energia_voz = np.mean(self.campo_actual[BANDAS_VOZ, :])
        energia_total = np.mean(self.campo_actual)
        
        # Acoplamiento como coherencia de la voz
        if energia_total > 0:
            self.acoplamiento = min(1.0, energia_voz / (energia_total + 0.01))
        else:
            self.acoplamiento = 0.5
        
        # Filtro suave
        self.acoplamiento = 0.9 * self.acoplamiento_anterior + 0.1 * self.acoplamiento
        
        # Error como pérdida de acoplamiento
        self.error = 1.0 - self.acoplamiento
        
        # Poder vivo (desviación de la consigna)
        self.poder_vivo = 0.9 * self.poder_vivo + 0.1 * self.acoplamiento
        desviacion = abs(self.poder_vivo - self.poder_base)
        
        # Liberdad funcional: capacidad de desacoplar respuesta automática
        inercia = np.exp(-desviacion * 4.0)
        self.lf = min(1.0, desviacion * (1.0 - inercia))
        
        # ICR e IRDE estimados
        self.icr = self.acoplamiento * (1.0 - self.error)
        self.irde = (1.0 - self.acoplamiento) * self.error
        
        # Guardar histórico
        self.acoplamiento_anterior = self.acoplamiento
    
    def calcular_signos(self):
        """Calcula signos acústicos (S_zona_voz, S_perturbacion_critica, etc.)."""
        if self.campo_actual is None:
            return {}
        
        energia_bandas = np.mean(self.campo_actual, axis=1)
        
        # Zona voz (energía en bandas vocales)
        energia_voz = np.sum(energia_bandas[BANDAS_VOZ])
        energia_total = np.sum(energia_bandas) + 0.01
        s_zona_voz = min(1.0, energia_voz / energia_total)
        
        # Perturbación crítica (energía en bandas de viento vs total)
        energia_viento = np.sum(energia_bandas[BANDAS_VIENTO])
        s_perturbacion = min(1.0, energia_viento / energia_total)
        
        # No intervenir (cuando el acoplamiento es bueno y estable)
        estabilidad = 1.0 - abs(self.delta_struct)
        s_no_intervenir = self.acoplamiento * estabilidad
        
        # Contexto integrable (inverso de perturbación)
        s_contexto = 1.0 - s_perturbacion
        
        return {
            'S_zona_voz': s_zona_voz,
            'S_perturbacion_critica': s_perturbacion,
            'S_no_intervenir': s_no_intervenir,
            'S_contexto_integrable': s_contexto
        }
    
    def evaluar_modo(self, signos):
        """Evalúa el modo funcional actual (ZONA_VOZ, DISCREPO, NO_SE, Y_SI)."""
        sz = signos['S_zona_voz']
        sp = signos['S_perturbacion_critica']
        sn = signos['S_no_intervenir']
        sc = signos['S_contexto_integrable']
        
        # Puntajes por modo (funciones de membresía suaves)
        score_zona_voz = 0.4 * sz + 0.3 * self.acoplamiento + 0.2 * (1.0 - self.error) + 0.1 * (1.0 - sp)
        score_discrepo = 0.3 * sp + 0.3 * self.delta_struct + 0.2 * self.acoplamiento + 0.2 * sc
        score_no_se = 0.5 * sp + 0.3 * self.error + 0.2 * (1.0 - self.acoplamiento)
        score_y_si = 0.4 * self.lf + 0.3 * (1.0 - abs(sz - sp)) + 0.3 * sn
        
        # Modo candidato
        scores = {
            'ZONA_VOZ': score_zona_voz,
            'DISCREPO': score_discrepo,
            'NO_SE': score_no_se,
            'Y_SI': score_y_si
        }
        candidato = max(scores, key=scores.get)
        confianza = scores[candidato]
        
        # Histéresis
        if candidato == self.modo_candidato:
            self.persistencia += 1
        else:
            self.modo_candidato = candidato
            self.persistencia = 1
        
        # Persistencia suficiente para cambiar
        requerida = PERSISTENCIA_ESTABLE if candidato == 'ZONA_VOZ' else PERSISTENCIA_CAMBIO
        if candidato == 'NO_SE':
            requerida = 2  # crisis entra más rápido
        
        if self.persistencia >= requerida:
            self.modo_actual = candidato
        
        return self.modo_actual, confianza
    
    def calcular_viabilidades(self, signos):
        """Calcula viabilidad de cada plan."""
        # PLAN_NADA (dejar evolucionar)
        costo_nada = 0.05
        ratio_energia = min(1.0, self.energia_computacion / max(costo_nada, 0.01))
        riesgo_artefacto = 0.05
        tiempo_penalizacion = min(0.5, self.tiempo_exposicion / 100.0)
        v_nada = max(0.0, 1.0 - (1.0 - ratio_energia) - riesgo_artefacto - 0.5 * tiempo_penalizacion)
        if self.acoplamiento > 0.7:
            v_nada = min(1.0, v_nada + 0.2)
        
        # PLAN_PRESERVAR (preservar estructura vocal)
        costo_preservar = 0.15 + 0.35 * self.delta_struct
        costo_requerido = costo_preservar * 1.3  # margen de seguridad
        if self.energia_computacion >= costo_requerido:
            v_preserva = 1.0
        else:
            v_preserva = self.energia_computacion / max(costo_requerido, 0.01)
        riesgo_artefacto_p = 0.1 + 0.3 * (1.0 - self.lf)
        latencia_penalizacion = min(0.3, self.tiempo_exposicion / 50.0)
        v_preserva = max(0.0, v_preserva - riesgo_artefacto_p - latencia_penalizacion)
        if self.modo_actual == 'NO_SE':
            v_preserva = min(1.0, v_preserva + 0.2)
        
        # PLAN_CONTEXTO (dejar pasar contexto)
        costo_contexto = 0.08 + 0.1 * signos['S_perturbacion_critica']
        ratio_contexto = min(1.0, self.energia_computacion / max(costo_contexto, 0.01))
        riesgo_contexto = 0.15 * self.irde
        exposicion_penalizacion = min(0.4, self.tiempo_exposicion / 40.0)
        v_contexto = max(0.0, ratio_contexto - riesgo_contexto - exposicion_penalizacion)
        v_contexto += 0.25 * signos['S_contexto_integrable']
        v_contexto -= 0.4 * signos['S_perturbacion_critica']
        
        return {
            'NADA': min(1.0, v_nada),
            'PRESERVAR': min(1.0, v_preserva),
            'CONTEXTO': min(1.0, max(0.0, v_contexto))
        }
    
    def calcular_coherencias(self, signos, modo):
        """Calcula coherencia de cada plan."""
        base = 1.0
        
        # Plan NADA
        c_nada = base
        c_nada -= signos['S_no_intervenir'] * 0.5  # moderado
        if modo == 'ZONA_VOZ':
            c_nada = min(1.0, c_nada + 0.2)
        
        # Plan PRESERVAR
        c_preserva = base
        c_preserva -= signos['S_perturbacion_critica'] * 0.2
        if modo == 'NO_SE':
            c_preserva = min(1.0, c_preserva + 0.3)
        if modo == 'ZONA_VOZ':
            c_preserva = min(1.0, c_preserva + 0.1)
        
        # Plan CONTEXTO
        c_contexto = base
        c_contexto -= signos['S_zona_voz'] * 0.8
        c_contexto -= signos['S_perturbacion_critica'] * 0.6
        c_contexto += signos['S_contexto_integrable'] * 0.3
        
        return {
            'NADA': max(0.0, min(1.0, c_nada)),
            'PRESERVAR': max(0.0, min(1.0, c_preserva)),
            'CONTEXTO': max(0.0, min(1.0, c_contexto))
        }
    
    def plan_permitido(self, plan, modo):
        """Verifica si un plan está permitido en el modo actual."""
        if modo == 'ZONA_VOZ':
            return plan in ['NADA', 'PRESERVAR']
        elif modo == 'DISCREPO':
            return True  # todos permitidos
        elif modo == 'NO_SE':
            return plan == 'PRESERVAR'
        elif modo == 'Y_SI':
            return plan in ['NADA', 'PRESERVAR']
        else:
            return plan == 'NADA'
    
    def elegir_plan(self, viabilidades, coherencias):
        """Elige el plan que maximiza J = α·V + (1-α)·C entre los permitidos."""
        mejor_j = -1.0
        mejor_plan = 'NADA'
        
        for plan in ['NADA', 'PRESERVAR', 'CONTEXTO']:
            if not self.plan_permitido(plan, self.modo_actual):
                continue
            j = self.alfa * viabilidades[plan] + (1.0 - self.alfa) * coherencias[plan]
            if j > mejor_j:
                mejor_j = j
                mejor_plan = plan
        
        return mejor_plan, mejor_j
    
    def actualizar_mezcla(self, plan_seleccionado):
        """Actualiza la mezcla continua entre planes."""
        objetivo_nada = 1.0 if plan_seleccionado == 'NADA' else 0.0
        objetivo_preservar = 1.0 if plan_seleccionado == 'PRESERVAR' else 0.0
        objetivo_contexto = 1.0 if plan_seleccionado == 'CONTEXTO' else 0.0
        
        # Velocidad de transición (menor LF = más lenta)
        k = 0.08 - 0.04 * self.lf
        k = max(0.01, min(0.08, k))
        
        self.mezcla_nada += k * (objetivo_nada - self.mezcla_nada)
        self.mezcla_preservar += k * (objetivo_preservar - self.mezcla_preservar)
        self.mezcla_contexto += k * (objetivo_contexto - self.mezcla_contexto)
        
        # Normalizar
        suma = self.mezcla_nada + self.mezcla_preservar + self.mezcla_contexto + 1e-6
        self.mezcla_nada /= suma
        self.mezcla_preservar /= suma
        self.mezcla_contexto /= suma
    
    def procesar_ventana(self, espectro):
        """Procesa una ventana del espectrograma."""
        self.actualizar_campo(espectro)
        self.calcular_emergentes()
        signos = self.calcular_signos()
        modo, confianza = self.evaluar_modo(signos)
        viabilidades = self.calcular_viabilidades(signos)
        coherencias = self.calcular_coherencias(signos, modo)
        plan, j = self.elegir_plan(viabilidades, coherencias)
        self.actualizar_mezcla(plan)
        
        # Actualizar tiempo de exposición (acumulado si el error es alto)
        if self.error > 0.4:
            self.tiempo_exposicion += 1
        else:
            self.tiempo_exposicion = max(0, self.tiempo_exposicion - 2)
        
        # Registrar histórico
        registro = {
            'ventana': len(self.historial),
            'delta_struct': self.delta_struct,
            'acoplamiento': self.acoplamiento,
            'error': self.error,
            'lf': self.lf,
            'icr': self.icr,
            'irde': self.irde,
            'modo': modo,
            'plan': plan,
            'j': j,
            'mezcla_nada': self.mezcla_nada,
            'mezcla_preservar': self.mezcla_preservar,
            'mezcla_contexto': self.mezcla_contexto,
            'S_zona_voz': signos['S_zona_voz'],
            'S_perturbacion': signos['S_perturbacion_critica'],
            'S_no_intervenir': signos['S_no_intervenir'],
            'S_contexto': signos['S_contexto_integrable'],
            'V_nada': viabilidades['NADA'],
            'V_preservar': viabilidades['PRESERVAR'],
            'V_contexto': viabilidades['CONTEXTO'],
            'C_nada': coherencias['NADA'],
            'C_preservar': coherencias['PRESERVAR'],
            'C_contexto': coherencias['CONTEXTO']
        }
        self.historial.append(registro)
        
        return registro
    
    def esta_vivo(self):
        """Verifica si el sistema está vivo (no colapsado)."""
        if len(self.historial) < 10:
            return True
        
        # Promedio de acoplamiento en últimas 10 ventanas (1 segundo)
        acoplamientos = [h['acoplamiento'] for h in self.historial[-10:]]
        errores = [h['error'] for h in self.historial[-10:]]
        
        acoplamiento_promedio = np.mean(acoplamientos)
        error_promedio = np.mean(errores)
        
        if acoplamiento_promedio < UMBRAL_ACOPLAMIENTO:
            return False
        if error_promedio > UMBRAL_ERROR:
            return False
        
        return True
    
    def resumen_supervivencia(self):
        """Genera resumen de supervivencia."""
        if len(self.historial) == 0:
            return {'vivo': False, 'razon': 'Sin datos'}
        
        vivo = self.esta_vivo()
        acoplamiento_final = self.historial[-1]['acoplamiento'] if vivo else 0
        error_final = self.historial[-1]['error'] if vivo else 1
        
        return {
            'vivo': vivo,
            'duracion_ventanas': len(self.historial),
            'acoplamiento_final': acoplamiento_final,
            'error_final': error_final,
            'modo_final': self.historial[-1]['modo'],
            'plan_final': self.historial[-1]['plan']
        }

# ============================================================
# FUNCIONES DE EXPERIMENTO Y VISUALIZACIÓN
# ============================================================

def ejecutar_experimento(entrada, nombre, alfa, energia, referencia_voz=None, viento_puro_audio=None):
    """Ejecuta un experimento con una configuración dada."""
    print(f"\n--- Ejecutando: {nombre} | α={alfa} | E={energia} ---")
    
    # Calcular espectrograma
    espectro = espectrograma(entrada, VENTANA_MUESTRAS, HOP_MUESTRAS)
    n_ventanas = espectro.shape[1]
    print(f"  Ventanas procesadas: {n_ventanas}")
    
    # Inicializar sistema
    sistema = SistemaCosmosemiotico(alfa=alfa, energia_computacion=energia, referencia_voz=referencia_voz)
    
    # Calibrar con viento puro (usar la señal de audio, no el espectro)
    if viento_puro_audio is not None:
        sistema.calibrar_viento(viento_puro_audio)
    
    # Procesar ventana por ventana
    for i in range(n_ventanas):
        sistema.procesar_ventana(espectro[:, i:i+1])
    
    # Resumen
    resumen = sistema.resumen_supervivencia()
    print(f"  Resultado: {'VIVO' if resumen['vivo'] else 'COLAPSADO'}")
    print(f"  Modo final: {resumen['modo_final']}")
    print(f"  Plan final: {resumen['plan_final']}")
    
    return sistema, resumen

def graficar_resultados(sistema, titulo, archivo_salida):
    """Genera gráficos de evolución temporal."""
    if len(sistema.historial) == 0:
        return
    
    h = sistema.historial
    t = np.arange(len(h)) * HOP_MS / 1000  # tiempo en segundos
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle(titulo, fontsize=14, fontweight='bold')
    
    # Gráfico 1: Acoplamiento y error
    ax = axes[0, 0]
    ax.plot(t, [x['acoplamiento'] for x in h], 'g-', label='Acoplamiento (A_sys_env)', linewidth=1)
    ax.plot(t, [x['error'] for x in h], 'r-', label='Error (e_R)', linewidth=1)
    ax.axhline(y=UMBRAL_ACOPLAMIENTO, color='g', linestyle='--', alpha=0.5)
    ax.axhline(y=UMBRAL_ERROR, color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel('Valor')
    ax.legend(loc='upper right')
    ax.set_title('Acoplamiento y Error')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Δ_struct y LF
    ax = axes[0, 1]
    ax.plot(t, [x['delta_struct'] for x in h], 'b-', label='Δ_struct', linewidth=1)
    ax.plot(t, [x['lf'] for x in h], 'm-', label='Libertad Funcional (LF)', linewidth=1)
    ax.set_ylabel('Valor')
    ax.legend(loc='upper right')
    ax.set_title('Diferencia Estructural y Libertad Funcional')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Modo (codificado como número)
    ax = axes[1, 0]
    modo_dict = {'ZONA_VOZ': 0, 'DISCREPO': 1, 'NO_SE': 2, 'Y_SI': 3}
    modos_num = [modo_dict.get(x['modo'], -1) for x in h]
    ax.plot(t, modos_num, 'o-', markersize=2, linewidth=0.5)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['ZONA_VOZ', 'DISCREPO', 'NO_SE', 'Y_SI'])
    ax.set_ylabel('Modo')
    ax.set_title('Modo Funcional')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: Plan seleccionado
    ax = axes[1, 1]
    plan_dict = {'NADA': 0, 'PRESERVAR': 1, 'CONTEXTO': 2}
    planes_num = [plan_dict.get(x['plan'], -1) for x in h]
    ax.plot(t, planes_num, 's-', markersize=2, linewidth=0.5, color='orange')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['NADA', 'PRESERVAR', 'CONTEXTO'])
    ax.set_ylabel('Plan')
    ax.set_title('Plan Seleccionado')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 5: Mezcla entre planes
    ax = axes[2, 0]
    ax.plot(t, [x['mezcla_nada'] for x in h], 'b-', label='NADA', linewidth=1)
    ax.plot(t, [x['mezcla_preservar'] for x in h], 'g-', label='PRESERVAR', linewidth=1)
    ax.plot(t, [x['mezcla_contexto'] for x in h], 'r-', label='CONTEXTO', linewidth=1)
    ax.set_ylabel('Peso')
    ax.legend(loc='upper right')
    ax.set_title('Mezcla Continua entre Planes')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 6: ICR e IRDE
    ax = axes[2, 1]
    ax.plot(t, [x['icr'] for x in h], 'c-', label='ICR', linewidth=1)
    ax.plot(t, [x['irde'] for x in h], 'm-', label='IRDE', linewidth=1)
    ax.set_ylabel('Valor')
    ax.legend(loc='upper right')
    ax.set_title('Conversión (ICR) vs Desviación (IRDE)')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 7: Signos principales
    ax = axes[3, 0]
    ax.plot(t, [x['S_zona_voz'] for x in h], 'g-', label='Zona Voz', linewidth=1)
    ax.plot(t, [x['S_perturbacion'] for x in h], 'r-', label='Perturbación Crítica', linewidth=1)
    ax.set_ylabel('Intensidad')
    ax.legend(loc='upper right')
    ax.set_title('Signos Acústicos')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 8: Viabilidades
    ax = axes[3, 1]
    ax.plot(t, [x['V_nada'] for x in h], 'b-', label='V_NADA', linewidth=1, alpha=0.7)
    ax.plot(t, [x['V_preservar'] for x in h], 'g-', label='V_PRESERVAR', linewidth=1, alpha=0.7)
    ax.plot(t, [x['V_contexto'] for x in h], 'r-', label='V_CONTEXTO', linewidth=1, alpha=0.7)
    ax.set_ylabel('Viabilidad')
    ax.legend(loc='upper right')
    ax.set_title('Viabilidad por Plan')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(archivo_salida, dpi=150)
    plt.close()
    print(f"  Gráfico guardado: {archivo_salida}")

def exportar_csv(sistema, nombre_archivo):
    """Exporta historial a CSV."""
    if len(sistema.historial) == 0:
        return
    
    import csv
    with open(nombre_archivo, 'w', newline='') as f:
        campos = list(sistema.historial[0].keys())
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(sistema.historial)
    print(f"  CSV guardado: {nombre_archivo}")

# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("=" * 60)
    print("EXPERIMENTO COSMOSEMIÓTICO - VALIDACIÓN CON GRABACIONES REALES")
    print("=" * 60)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Directorio: {os.getcwd()}")
    
    # 1. Cargar archivos
    print("\n[1] Cargando archivos...")
    
    sr_v, voz_estudio = wav.read('Voz_Estudio.wav')
    sr_v1, voz_viento1 = wav.read('Voz+Viento_1.wav')
    sr_v2, voz_viento2 = wav.read('Voz+Viento_2.wav')
    sr_w, viento_puro = wav.read('Viento.wav')
    
    # Convertir a mono y normalizar
    voz_estudio = normalizar(a_mono(voz_estudio))
    voz_viento1 = normalizar(a_mono(voz_viento1))
    voz_viento2 = normalizar(a_mono(voz_viento2))
    viento_puro = normalizar(a_mono(viento_puro))
    
    # Verificar frecuencias de muestreo
    print(f"  Frecuencias: Voz_Estudio={sr_v}, Voz+Viento1={sr_v1}, Voz+Viento2={sr_v2}, Viento={sr_w}")
    
    # 2. Experimentos
    print("\n[2] Ejecutando experimentos...")
    
    resultados = []
    
    # Experimento 1: Viento puro (control negativo)
    sistema, res = ejecutar_experimento(viento_puro, "Control: Viento Puro", 0.5, 0.9, None, viento_puro)
    resultados.append({'nombre': 'viento_puro', 'sistema': sistema, 'resumen': res})
    graficar_resultados(sistema, "Control: Viento Puro (α=0.5, E=0.9)", "resultado_viento_puro.png")
    exportar_csv(sistema, "historial_viento_puro.csv")
    
    # Experimento 2: Voz+Viento 1 (menos viento) con α=0.5
    sistema, res = ejecutar_experimento(voz_viento1, "Voz+Viento 1", 0.5, 0.9, None, viento_puro)
    resultados.append({'nombre': 'voz_viento1_alfa05', 'sistema': sistema, 'resumen': res})
    graficar_resultados(sistema, "Voz + Viento 1 (α=0.5)", "resultado_voz_viento1_alfa05.png")
    exportar_csv(sistema, "historial_voz_viento1_alfa05.csv")
    
    # Experimento 3: Voz+Viento 2 (más viento) con α=0.5
    sistema, res = ejecutar_experimento(voz_viento2, "Voz+Viento 2", 0.5, 0.9, None, viento_puro)
    resultados.append({'nombre': 'voz_viento2_alfa05', 'sistema': sistema, 'resumen': res})
    graficar_resultados(sistema, "Voz + Viento 2 (α=0.5)", "resultado_voz_viento2_alfa05.png")
    exportar_csv(sistema, "historial_voz_viento2_alfa05.csv")
    
    # Experimento 4: Voz+Viento 2 con α=0.0 (solo coherencia)
    sistema, res = ejecutar_experimento(voz_viento2, "Voz+Viento 2 α=0.0", 0.0, 0.9, None, viento_puro)
    resultados.append({'nombre': 'voz_viento2_alfa00', 'sistema': sistema, 'resumen': res})
    graficar_resultados(sistema, "Voz + Viento 2 (α=0.0 - Solo Coherencia)", "resultado_voz_viento2_alfa00.png")
    exportar_csv(sistema, "historial_voz_viento2_alfa00.csv")
    
    # Experimento 5: Voz+Viento 2 con α=1.0 (solo viabilidad)
    sistema, res = ejecutar_experimento(voz_viento2, "Voz+Viento 2 α=1.0", 1.0, 0.9, None, viento_puro)
    resultados.append({'nombre': 'voz_viento2_alfa10', 'sistema': sistema, 'resumen': res})
    graficar_resultados(sistema, "Voz + Viento 2 (α=1.0 - Solo Viabilidad)", "resultado_voz_viento2_alfa10.png")
    exportar_csv(sistema, "historial_voz_viento2_alfa10.csv")
    
    # Experimento 6: Voz+Viento 2 con energía computacional baja (0.2)
    sistema, res = ejecutar_experimento(voz_viento2, "Voz+Viento 2 E=0.2", 0.5, 0.2, None, viento_puro)
    resultados.append({'nombre': 'voz_viento2_energia02', 'sistema': sistema, 'resumen': res})
    graficar_resultados(sistema, "Voz + Viento 2 (E=0.2 - Baja Energía)", "resultado_voz_viento2_energia02.png")
    exportar_csv(sistema, "historial_voz_viento2_energia02.csv")
    
    # 3. Resumen general
    print("\n[3] RESUMEN GENERAL DE RESULTADOS")
    print("-" * 60)
    print(f"{'Experimento':<35} {'Vivo':<8} {'Modo Final':<12} {'Plan Final':<12}")
    print("-" * 60)
    
    for r in resultados:
        vivo = "SÍ" if r['resumen']['vivo'] else "NO"
        print(f"{r['nombre']:<35} {vivo:<8} {r['resumen']['modo_final']:<12} {r['resumen']['plan_final']:<12}")
    
    print("-" * 60)
    
    # 4. Archivo de resumen
    with open('resumen_experimento.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EXPERIMENTO COSMOSEMIÓTICO - RESUMEN\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for r in resultados:
            f.write(f"Experimento: {r['nombre']}\n")
            f.write(f"  Vivo: {'SÍ' if r['resumen']['vivo'] else 'NO'}\n")
            f.write(f"  Duración (ventanas): {r['resumen']['duracion_ventanas']}\n")
            f.write(f"  Acoplamiento final: {r['resumen']['acoplamiento_final']:.3f}\n")
            f.write(f"  Error final: {r['resumen']['error_final']:.3f}\n")
            f.write(f"  Modo final: {r['resumen']['modo_final']}\n")
            f.write(f"  Plan final: {r['resumen']['plan_final']}\n\n")
    
    print("\nArchivos generados:")
    print("  - resumen_experimento.txt")
    print("  - resultado_*.png (6 gráficos)")
    print("  - historial_*.csv (6 archivos con datos por ventana)")
    print("\n¡Experimento completado!")

if __name__ == "__main__":
    main()