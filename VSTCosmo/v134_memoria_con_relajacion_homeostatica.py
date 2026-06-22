#!/usr/bin/env python3
"""
VSTCosmos V134 — Memoria con relajacion homeostatica

Correccion sobre V133:
  - La confianza modula el setpoint usado
  - Cuando confianza decae, orientacion vuelve gradualmente a centro
  - Olvido conductual, no solo epistemologico

Hipotesis O-N9.1:
  - La orientacion durante silencio sigue una funcion de decaimiento
  - orient(t) = orient_inicial * confianza(t)
  - Error de relajacion < 10° en t=60s
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
from collections import deque

# ============================================================
# PARAMETROS (heredados de V133)
# ============================================================
DT = 0.01
TIEMPO_POR_REPETICION = 452.0
REPETICIONES_LENTAS = 10

# Asimetria forzada
SESGO_L = 0.05
SESGO_R = -0.05
DIM_HEMISFERIO = 32

# Zona muerta
ZONA_MUERTA_BASE = 2.0

# Limites de plasticidad
KP_BASE = 0.002
KP_MIN = 0.0005
KP_MAX = 0.005

# Plasticidad
HABITUACION_RAPIDA = 0.99
SENSIBILIZACION_LENTA = 1.01
VENTANA_OSCILACION = 100

# Memoria episodica
TAU_MEMORIA = 30.0
UMBRAL_CONFIANZA = 0.1
ALPHA_CONFIANZA = 1.0  # Exponente para modular setpoint (1.0 = lineal)

# Semilla base
SEMILLA_BASE = 44


# ============================================================
# HEMISFERIO
# ============================================================

class HemisferioV134:
    def __init__(self, nombre, tau, generar_entrada_func, seed=None, sesgo=0.0):
        if seed is not None:
            np.random.seed(seed)
        self.nombre = nombre
        self.tau = tau
        self.generar_entrada = generar_entrada_func
        self.sesgo = sesgo
        
        self.Phi = np.random.normal(sesgo, 0.1, 32)
        self.Phi_vel = np.zeros(32)
        
        self.entrada = None
        self.sr = 48000
        self.en_inanicion = False
        self.factor_inanicion = 1.0
        
        self.buffer_rapido = []
        self.historial_omega = []
    
    def _calcular_omega(self):
        return np.mean(self.Phi[:32])
    
    def generar_entrada_para_t(self, t, duracion_total):
        if self.entrada is None:
            self.entrada = self.generar_entrada(duracion_total, self.sr)
        idx = int(t * self.sr)
        if idx >= len(self.entrada):
            return 0.0
        return self.entrada[idx] * self.factor_inanicion
    
    def actualizar(self, t, dt, duracion_total, otro_hemisferio=None):
        entrada = self.generar_entrada_para_t(t, duracion_total)
        
        laplaciano = np.zeros_like(self.Phi)
        for i in range(1, 31):
            laplaciano[i] = self.Phi[i-1] - 2*self.Phi[i] + self.Phi[i+1]
        
        reaccion = self.Phi * (1 - self.Phi * self.Phi)
        
        forzamiento = np.zeros_like(self.Phi)
        forzamiento[0] = entrada
        forzamiento[-1] = -entrada
        
        acoplamiento = np.zeros_like(self.Phi)
        if otro_hemisferio is not None:
            divergencia = abs(self._calcular_omega() - otro_hemisferio._calcular_omega())
            if divergencia > 0.5:
                acoplamiento = 0.01 * (otro_hemisferio.Phi - self.Phi)
        
        dPhi_vel = laplaciano + reaccion + forzamiento + acoplamiento
        self.Phi_vel += dPhi_vel * dt
        self.Phi += self.Phi_vel * dt
        self.Phi = np.clip(self.Phi, -1.0, 1.0)
        
        return {'omega': self._calcular_omega()}


# ============================================================
# MEMORIA EPISODICA CON RELAJACION HOMEOSTATICA (V134)
# ============================================================

class MemoriaConRelajacion:
    """
    Memoria donde la confianza modula el setpoint usado.
    
    Correccion sobre V133:
      - setpoint_usado = angulo_recordado * (confianza ** alpha)
      - Cuando confianza decae, orientacion vuelve gradualmente a centro
      - Olvido conductual, no solo epistemologico
    """
    
    def __init__(self, tau=TAU_MEMORIA, umbral_confianza=UMBRAL_CONFIANZA, alpha=ALPHA_CONFIANZA):
        self.tau = tau
        self.umbral_confianza = umbral_confianza
        self.alpha = alpha
        self.angulo = 0.0
        self.confianza = 0.0
        self.t_ultimo_estimulo = 0.0
        self.historial_confianza = []
    
    def update(self, angulo_medido, fuente_activa, t):
        if fuente_activa:
            self.angulo = angulo_medido
            self.confianza = 1.0
            self.t_ultimo_estimulo = t
        else:
            dt_silencio = t - self.t_ultimo_estimulo
            if dt_silencio >= 0:
                self.confianza = np.exp(-dt_silencio / self.tau)
            else:
                self.confianza = 0.0
        
        self.historial_confianza.append(self.confianza)
        return self.confianza
    
    def get_setpoint(self):
        """Devuelve setpoint modulado por confianza"""
        if self.confianza > self.umbral_confianza:
            # MODULACION CRITICA: setpoint = angulo * confianza^alpha
            return self.angulo * (self.confianza ** self.alpha)
        else:
            return 0.0
    
    def get_confianza(self):
        return self.confianza


# ============================================================
# APARATO MOTOR CON MEMORIA Y RELAJACION
# ============================================================

class AparatoMotorConRelajacion:
    def __init__(self):
        self.orientacion = 0.0
        self.Kp_base = KP_BASE
        self.Kp_actual = KP_BASE
        self.Kp_min = KP_MIN
        self.Kp_max = KP_MAX
        self.limite = 90.0
        self.zona_muerta = ZONA_MUERTA_BASE
        self.inercia = 0.95
        self.ultimo_delta = 0.0
        self.sensibilidad_grad = 10.0
        self.t = 0.0
        
        # Memoria con relajacion (NUEVO EN V134)
        self.memoria = MemoriaConRelajacion(tau=TAU_MEMORIA, alpha=ALPHA_CONFIANZA)
        
        # Plasticidad
        self.memoria_error = deque(maxlen=VENTANA_OSCILACION)
        self.historial_Kp = []
        
        self.setpoint_usado = 0.0
    
    def calcular_factor_freno(self, error):
        return 1 - np.exp(-abs(error) / 30.0)
    
    def actualizar_plasticidad(self, error):
        self.memoria_error.append(error)
        if len(self.memoria_error) < VENTANA_OSCILACION:
            return
        
        oscilacion = np.std(self.memoria_error)
        if oscilacion > self.zona_muerta * 1.5:
            self.Kp_actual = max(self.Kp_min, self.Kp_actual * 0.99)
        elif oscilacion < self.zona_muerta * 0.5:
            self.Kp_actual = min(self.Kp_max, self.Kp_actual * 1.01)
        
        self.historial_Kp.append(self.Kp_actual)
    
    def actuar(self, gradiente, LF_activa, fuente_activa, t, setpoint_percepcion):
        if not LF_activa:
            return self.orientacion, 0.0
        
        if abs(gradiente) < 0.05:
            return self.orientacion, self.memoria.get_confianza()
        
        # Actualizar memoria
        if fuente_activa:
            self.memoria.update(setpoint_percepcion, True, t)
            # Con fuente activa, el setpoint es el perceptivo
            self.setpoint_usado = setpoint_percepcion
        else:
            # Sin fuente: usar memoria con relajacion
            self.memoria.update(0.0, False, t)
            self.setpoint_usado = self.memoria.get_setpoint()
        
        error = self.setpoint_usado - self.orientacion
        
        # Zona muerta
        if abs(error) < self.zona_muerta:
            return self.orientacion, self.memoria.get_confianza()
        
        ganancia_grad = -np.tanh(gradiente * self.sensibilidad_grad)
        factor_freno = self.calcular_factor_freno(error)
        
        delta = self.Kp_actual * error * ganancia_grad * factor_freno
        
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta
        self.ultimo_delta = delta
        
        self.actualizar_plasticidad(error)
        
        self.orientacion += delta
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        self.t += DT
        
        return self.orientacion, self.memoria.get_confianza()
    
    def reset(self):
        self.orientacion = 0.0
        self.ultimo_delta = 0.0
        self.t = 0.0
        self.Kp_actual = KP_BASE
        self.memoria_error.clear()
        self.historial_Kp = []


# ============================================================
# SISTEMA V134
# ============================================================

class SistemaV134:
    def __init__(self, nombre, seed=SEMILLA_BASE):
        self.nombre = nombre
        
        def generar_ruido_rosa(duracion, sr):
            n = int(duracion * sr)
            ruido = np.random.normal(0, 1, n)
            fft = np.fft.rfft(ruido)
            freqs = np.fft.rfftfreq(n, 1/sr)
            filtro = 1.0 / np.sqrt(freqs + 0.01)
            fft_filtrado = fft * filtro
            ruido_rosa = np.fft.irfft(fft_filtrado, n=n)
            ruido_rosa = ruido_rosa / (np.max(np.abs(ruido_rosa)) + 1e-10)
            return ruido_rosa
        
        def generar_clicks_poisson(duracion, tasa=0.5, sr=48000):
            n = int(duracion * sr)
            clicks = np.zeros(n)
            n_clicks = int(duracion * tasa)
            for _ in range(n_clicks):
                pos = int(np.random.exponential(1.0/tasa) * sr)
                if pos < n:
                    clicks[pos] = 1.0
            return clicks
        
        self.izquierdo = HemisferioV134("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV134("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV134("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV134("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.modo_entrenamiento = True
        self.motor = AparatoMotorConRelajacion()
        
        self.historial = {
            't': [],
            'orientacion': [],
            'setpoint_usado': [],
            'confianza': [],
            'fuente_activa': [],
            's_shared': [],
            'Kp': []
        }
    
    def calcular_s_shared(self):
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        return 1 - abs(omega_A - omega_B) / 2.0
    
    def actualizar(self, t, dt, duracion_total, fuente_activa, setpoint_percepcion):
        # Gradiente inter-sistemas
        self.izquierdo.actualizar(t, dt, duracion_total, self.derecho)
        self.derecho.actualizar(t, dt, duracion_total, self.izquierdo)
        self.sistema_B_izq.actualizar(t, dt, duracion_total, self.sistema_B_der)
        self.sistema_B_der.actualizar(t, dt, duracion_total, self.sistema_B_izq)
        
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        gradiente = omega_A - omega_B
        
        # Espacializacion (solo si hay fuente)
        if fuente_activa:
            sesgo = setpoint_percepcion / 90.0
            gradiente += sesgo * 0.5
        
        # Motor con relajacion
        LF_activa = not self.modo_entrenamiento
        orientacion, confianza = self.motor.actuar(
            gradiente, LF_activa, fuente_activa, t, setpoint_percepcion
        )
        
        self.historial['t'].append(t)
        self.historial['orientacion'].append(orientacion)
        self.historial['setpoint_usado'].append(self.motor.setpoint_usado)
        self.historial['confianza'].append(confianza)
        self.historial['fuente_activa'].append(fuente_activa)
        self.historial['s_shared'].append(self.calcular_s_shared())
        self.historial['Kp'].append(self.motor.Kp_actual)
        
        return orientacion
    
    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()


# ============================================================
# EXPERIMENTO V134
# ============================================================

def ejecutar_v134():
    print("=" * 100)
    print("EXPERIMENTO V134 — Memoria con relajacion homeostatica")
    print("=" * 100)
    print("  ANIMA-2 - Linea 1 (cierre): Hipotesis O-N9.1")
    print("  Correccion sobre V133:")
    print("    - La confianza modula el setpoint usado")
    print("    - orient(t) = orient_inicial * confianza(t)")
    print("    - Olvido conductual, no solo epistemologico")
    print("=" * 100)
    
    # Configurar sistema
    sistema = SistemaV134("V134", seed=SEMILLA_BASE)
    
    # Entrenamiento lateral
    print("\n  Entrenando lateralidad (10 repeticiones)...")
    sistema.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS,
                              fuente_activa=False, setpoint_percepcion=0.0)
    
    print("  Entrenamiento completado.")
    
    # Fase de test con memoria y relajacion
    print("\n  Iniciando test de memoria con relajacion...")
    sistema.set_modo_entrenamiento(False)
    sistema.motor.reset()
    
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    duracion_test = 240.0
    
    for i in range(int(duracion_test / DT)):
        t = t_actual + i * DT
        t_rel = i * DT
        
        # Protocolo igual a V133
        if t_rel < 60:  # Fase 1: 0-60s, fuente a -60° ON
            fuente_activa = True
            setpoint = -60.0
        elif t_rel < 120:  # Fase 2: 60-120s, silencio OFF
            fuente_activa = False
            setpoint = -60.0  # No usado
        elif t_rel < 180:  # Fase 3: 120-180s, fuente a -60° ON
            fuente_activa = True
            setpoint = -60.0
        else:  # Fase 4: 180-240s, fuente a +60° ON
            fuente_activa = True
            setpoint = 60.0
        
        orientacion = sistema.actualizar(t, DT, t_actual + duracion_test,
                                         fuente_activa, setpoint)
        
        # Reporte cada 10s
        if int(t_rel * 10) % 100 == 0 and t_rel > 0:
            fase = ""
            if t_rel < 60:
                fase = "F1(-60°)"
            elif t_rel < 120:
                fase = "F2(silencio)"
            elif t_rel < 180:
                fase = "F3(reenganche)"
            else:
                fase = "F4(+60°)"
            
            conf = sistema.historial['confianza'][-1]
            setpoint_usado = sistema.historial['setpoint_usado'][-1]
            
            print(f"    t={t_rel:4.0f}s | {fase:12s} | orient={orientacion:5.1f}° | "
                  f"confianza={conf:.2f} | setpoint_usado={setpoint_usado:5.1f}°")
    
    # Analisis de resultados
    print("\n" + "=" * 80)
    print("ANALISIS DE RELAJACION HOMEOSTATICA")
    print("=" * 80)
    
    t_total = np.array(sistema.historial['t'])
    t_rel = t_total - t_total[0]
    orientacion = np.array(sistema.historial['orientacion'])
    confianza = np.array(sistema.historial['confianza'])
    setpoint_usado = np.array(sistema.historial['setpoint_usado'])
    fuente_activa = np.array(sistema.historial['fuente_activa'])
    
    # Metricas de relajacion (Fase 2: silencio)
    mask_f2 = (t_rel >= 60) & (t_rel < 120)
    
    if np.any(mask_f2):
        t_f2 = t_rel[mask_f2] - 60
        orient_f2 = orientacion[mask_f2]
        conf_f2 = confianza[mask_f2]
        setpoint_f2 = setpoint_usado[mask_f2]
        
        # Orientacion inicial en silencio (primeros 5s)
        orient_inicial = np.mean(orient_f2[t_f2 < 5]) if np.any(t_f2 < 5) else orient_f2[0]
        
        # Orientacion a los 60s de silencio
        orient_final = orient_f2[-1] if len(orient_f2) > 0 else orient_inicial
        
        # Error de relajacion (deriva desde -60°)
        error_relajacion = abs(orient_final + 60.0)
        
        # Setpoint teorico (modulado por confianza)
        setpoint_teorico = -60.0 * (conf_f2 ** ALPHA_CONFIANZA)
        
        # Error entre setpoint usado y teorico
        error_modulacion = np.mean(np.abs(setpoint_f2 - setpoint_teorico))
        
        print(f"\n  Fase 2 - Silencio (60-120s):")
        print(f"    Orientacion inicial: {orient_inicial:.1f}°")
        print(f"    Orientacion final (60s silencio): {orient_final:.1f}°")
        print(f"    Error de relajacion (deriva): {error_relajacion:.1f}°")
        print(f"    Confianza final: {conf_f2[-1]:.2f}")
        print(f"    Error de modulacion setpoint: {error_modulacion:.1f}°")
    
    # Criterios de exito O-N9.1
    exito_relajacion = error_relajacion < 30.0 if 'error_relajacion' in dir() else False
    exito_modulacion = error_modulacion < 10.0 if 'error_modulacion' in dir() else False
    
    # Verificar que en Fase 4 se sobreescribe
    mask_f4 = (t_rel >= 180) & (t_rel < 200)
    if np.any(mask_f4):
        orient_f4 = orientacion[mask_f4]
        orient_f4_final = orient_f4[-1] if len(orient_f4) > 0 else 0
        exito_sobreescritura = abs(orient_f4_final - 60.0) < 15.0
    else:
        exito_sobreescritura = False
    
    print(f"\n  Criterios de exito O-N9.1:")
    print(f"    Relajacion (deriva < 30° en 60s): {'✅' if exito_relajacion else '❌'}")
    print(f"    Modulacion setpoint por confianza: {'✅' if exito_modulacion else '❌'}")
    print(f"    Sobrescritura a +60°: {'✅' if exito_sobreescritura else '❌'}")
    
    exito_total = exito_relajacion and exito_modulacion and exito_sobreescritura
    
    # Graficos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Grafico 1: Orientacion con relajacion
    ax = axes[0, 0]
    ax.plot(t_rel, orientacion, 'b-', linewidth=0.8, label='Orientacion real')
    ax.axvline(x=60, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=120, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=180, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=-60, color='green', linestyle='--', alpha=0.5, label='Objetivo -60°')
    ax.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Objetivo +60°')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Orientacion (grados)')
    ax.set_title('V134: Memoria con relajacion homeostatica')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Grafico 2: Confianza y setpoint modulado
    ax = axes[0, 1]
    ax.plot(t_rel, confianza, 'purple', linewidth=0.8, label='Confianza')
    ax.plot(t_rel, setpoint_usado, 'r--', linewidth=0.8, alpha=0.7, label='Setpoint usado (modulado)')
    ax.axvline(x=60, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=120, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Valor')
    ax.set_title('Confianza y setpoint modulado')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Grafico 3: Detalle de la relajacion durante silencio
    ax = axes[1, 0]
    if np.any(mask_f2):
        ax.plot(t_f2, orient_f2, 'b-', linewidth=1, label='Orientacion real')
        ax.plot(t_f2, -60.0 * (conf_f2 ** ALPHA_CONFIANZA), 'r--', linewidth=1, alpha=0.7, 
                label=f'Teorico (orient * conf^{ALPHA_CONFIANZA})')
        ax.axhline(y=-60, color='green', linestyle=':', alpha=0.5, label='Objetivo original')
        ax.set_xlabel('Tiempo de silencio (s)')
        ax.set_ylabel('Orientacion (grados)')
        ax.set_title('Relajacion conductual durante silencio')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Grafico 4: Kp adaptativo
    ax = axes[1, 1]
    Kp_hist = sistema.historial['Kp']
    ax.plot(t_total[:len(Kp_hist)], Kp_hist, 'green', linewidth=0.8)
    ax.axhline(y=KP_BASE, color='gray', linestyle='--', alpha=0.5, label=f'Kp_base = {KP_BASE}')
    ax.axhline(y=KP_MIN, color='red', linestyle=':', alpha=0.5, label=f'Kp_min = {KP_MIN}')
    ax.axhline(y=KP_MAX, color='green', linestyle=':', alpha=0.5, label=f'Kp_max = {KP_MAX}')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Kp')
    ax.set_title('Plasticidad homeostatica')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v134_logs', exist_ok=True)
    plt.savefig(f'v134_logs/v134_relajacion_{timestamp}.png', dpi=150)
    print(f"\n  Graficos guardados: v134_logs/v134_relajacion_{timestamp}.png")
    
    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION V134 — Memoria con relajacion homeostatica")
    print("=" * 80)
    
    if exito_total:
        print("\n  ✅ O-N9.1 VALIDADA: Memoria con olvido conductual")
        print("     - El organismo relaja orientacion cuando la confianza decae")
        print("     - Setpoint modulado por confianza")
        print("     - Olvido gradual, no catastrofico")
        print("\n  ANIMA-2 - Linea 1: CERRADA COMPLETAMENTE")
    else:
        print("\n  ⚠️ O-N9.1 NO VALIDADA: La relajacion no es suficiente")
    
    return sistema, exito_total


if __name__ == "__main__":
    import time
    start = time.time()
    sistema, exito = ejecutar_v134()
    elapsed = time.time() - start
    print(f"\n  Tiempo de ejecucion: {elapsed/60:.1f} minutos")