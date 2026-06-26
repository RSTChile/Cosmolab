"""
OrganoFonador — aparato fonador paramétrico para los organismos ANIMA.

Segunda vía de emisión: NO reemplaza el banco de 40 palabras (.wav), se suma
como capacidad generativa. El organismo conserva su vocabulario fijo y, además,
puede producir sonido propio cuando lo decida.

Motor: emulación del ARP 2600 — el sintetizador con que Ben Burtt creó la voz de
R2-D2 — por síntesis FM y barridos de oscilador. Mientras toda vocalización pase
por este motor, el carácter R2-D2 queda garantizado por construcción.

Mapeo de ejes al ARP 2600:
  - barrido de frecuencia (freq_puntos / sirena) ......... VCO + portamento (glide)
  - fm_ratio, fm_index ................................... FM / ring-mod (timbre metálico)
  - vibrato (rate, profundidad) .......................... LFO
  - tension (limpio -> saturado/ruidoso) ................. calidad de la fuente glotal
  - resonancia ........................................... VCF resonante (formante / color)
  - ataque, caida, sostiene .............................. VCA + envolvente (EG)

PRINCIPIO RECTOR: este organelo es APARATO FONADOR, no intención. Recibe
parámetros y produce sonido. QUÉ vocalizar y CUÁNDO lo decide el organismo desde
su fisiología (arousal, valencia, lo que el balbuceo explore); ese nervio NO está
aquí — se cablea aparte, sobre el cuerpo real.

Salida idéntica al banco de palabras: 44.1 kHz, mono, PCM 16 bits, pico 0.9, con
micro fade anti-clic. Así las vocalizaciones generadas se concatenan sin fricción
con los .wav del vocabulario existente.
"""

import os
import json
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import iirpeak, lfilter


class OrganoFonador:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------ #
    #  PRIMITIVAS                                                         #
    # ------------------------------------------------------------------ #
    def _t(self, duracion):
        return np.linspace(0, duracion, int(self.sample_rate * duracion), endpoint=False)

    def _silencio(self, duracion):
        return np.zeros(int(self.sample_rate * duracion))

    def _aplicar_envolvente(self, senal, ventana_fade=220):
        """Micro fade-in/out anti-clic (idéntico al banco de palabras)."""
        if len(senal) < (ventana_fade * 2):
            return senal
        env = np.ones(len(senal))
        env[:ventana_fade] = np.linspace(0.0, 1.0, ventana_fade)
        env[-ventana_fade:] = np.linspace(1.0, 0.0, ventana_fade)
        return senal * env

    def _fase(self, freq_inst):
        """Fase integrada de una frecuencia instantánea (glide sin glitches)."""
        return 2 * np.pi * np.cumsum(freq_inst) / self.sample_rate

    def _contorno(self, n, puntos):
        """Interpola un contorno de frecuencia a partir de puntos (pos[0..1], Hz)."""
        xs = np.array([p[0] for p in puntos], dtype=float)
        ys = np.array([p[1] for p in puntos], dtype=float)
        return np.interp(np.linspace(0, 1, n), xs, ys)

    def _envolvente_vca(self, n, ataque, caida, sostiene):
        """Envolvente de amplitud (VCA): ataque -> sostén -> caída final."""
        env = np.ones(n)
        na = min(int(self.sample_rate * ataque), n)
        if na > 0:
            env[:na] = np.linspace(0.0, 1.0, na)
        nc = min(int(self.sample_rate * caida), n - na) if caida > 0 else 0
        if nc > 0:
            env[-nc:] = np.linspace(1.0, sostiene, nc)
        return env

    # ------------------------------------------------------------------ #
    #  NÚCLEO: una vocalización paramétrica estilo ARP 2600               #
    # ------------------------------------------------------------------ #
    def vocalizar(self, duracion=0.4,
                  freq_puntos=None, f_ini=600, f_fin=None,
                  sirena=None,                 # (f_a, f_b, rate_Hz)
                  fm_ratio=0.0,                # modulador = fm_ratio * frecuencia base
                  fm_index=0.0,                # escalar o (ini, fin): cantidad de FM
                  vibrato=(0.0, 0.0),          # (rate_Hz, profundidad_Hz)
                  tension=0.0,                 # 0 limpio -> 1 saturado/ruidoso
                  resonancia=0.0,              # 0..1 filtro resonante (color/formante)
                  res_centro=1400,             # Hz del realce resonante
                  ataque=0.01, caida=0.0, sostiene=1.0):
        """Devuelve una vocalización (float) sin renderizar a 16 bits."""
        t = self._t(duracion)
        n = len(t)

        # --- frecuencia base: barrido de oscilador (VCO + portamento) ---
        if sirena is not None:
            f_a, f_b, rate = sirena
            f_base = np.where(np.sin(2 * np.pi * rate * t) >= 0, float(f_a), float(f_b))
        elif freq_puntos is not None:
            f_base = self._contorno(n, freq_puntos)
        else:
            f_base = self._contorno(n, [(0, f_ini), (1, f_ini if f_fin is None else f_fin)])

        # --- vibrato (LFO) sobre la frecuencia instantánea ---
        vib_rate, vib_depth = vibrato
        f_inst = f_base + (vib_depth * np.sin(2 * np.pi * vib_rate * t) if vib_rate > 0 else 0.0)
        fase_port = self._fase(f_inst)

        # --- FM: modulador proporcional a la base (timbre metálico ARP) ---
        if isinstance(fm_index, (tuple, list)):
            idx = np.linspace(fm_index[0], fm_index[1], n)
        else:
            idx = np.full(n, float(fm_index))
        if fm_ratio > 0:
            fase_mod = self._fase(fm_ratio * f_base)
            senal = np.sin(fase_port + idx * np.sin(fase_mod))
        else:
            senal = np.sin(fase_port)

        # --- tensión: calidad de la fuente glotal (limpia -> ruidosa/tensa) ---
        if tension > 0:
            ruido = np.random.default_rng(0).standard_normal(n) * 0.6
            senal = (1 - tension) * senal + tension * (senal * (1 + 0.5 * ruido))
            senal = np.tanh((1 + 4 * tension) * senal)

        # --- resonancia: VCF resonante (formante / color) ---
        if resonancia > 0:
            w0 = min(max(res_centro / (self.sample_rate / 2), 1e-3), 0.99)
            Q = 1.0 + resonancia * 15.0
            b, a = iirpeak(w0, Q)
            wet = lfilter(b, a, senal)
            senal = (1 - 0.5 * resonancia) * senal + 0.7 * resonancia * wet

        # --- VCA + envolvente ---
        senal = senal * self._envolvente_vca(n, ataque, caida, sostiene)
        return senal

    def frase(self, gestos, gap=0.02):
        """Concatena varias vocalizaciones (cada gesto = dict de parámetros)."""
        partes = []
        for i, g in enumerate(gestos):
            partes.append(self.vocalizar(**g))
            if i < len(gestos) - 1:
                partes.append(self._silencio(gap))
        return np.concatenate(partes)

    # ------------------------------------------------------------------ #
    #  SALIDA (mismo formato que el banco de palabras)                   #
    # ------------------------------------------------------------------ #
    def render(self, senal, fade=220):
        """float -> PCM 16 bits, mono, pico 0.9, con micro fade anti-clic."""
        senal = self._aplicar_envolvente(senal, fade)
        pico = np.max(np.abs(senal))
        norm = senal / pico * 0.9 if pico > 0 else senal
        return np.int16(norm * 32767)

    def escribir(self, senal, ruta):
        wav.write(ruta, self.sample_rate, self.render(senal))

    # ------------------------------------------------------------------ #
    #  DEMOSTRACIÓN: el espacio continuo CONTIENE los arquetipos del banco
    #  (no los reemplaza; verifica que el sintetizador los cubre)        #
    # ------------------------------------------------------------------ #
    def presets_demostracion(self):
        """Arquetipos que cubren los contornos de las 40 palabras del banco."""
        return {
            # representa: pregunta, asombro, hallazgo (ascendentes)
            "interrogacion": lambda: self.vocalizar(
                duracion=0.45, freq_puntos=[(0, 600), (1, 1400)],
                fm_ratio=2.0, fm_index=(0.5, 2.0)),
            # representa: negacion, tristeza, despedida (descendentes)
            "negativa": lambda: self.vocalizar(
                duracion=0.4, f_ini=240, f_fin=110,
                fm_ratio=1.5, fm_index=(1.5, 0.4), caida=0.2, sostiene=0.2),
            # representa: alerta, urgencia (FM estridente, como el ARP original)
            "alarma": lambda: self.vocalizar(
                duracion=0.4, f_ini=1500, f_fin=1500,
                fm_ratio=32 / 1500, fm_index=12),
            # representa: peligro (sirena: alternancia de osciladores)
            "sirena": lambda: self.vocalizar(
                duracion=0.5, sirena=(1200, 750, 11)),
            # representa: bateria_baja, fatiga (wobble: vibrato profundo cayendo)
            "wobble": lambda: self.vocalizar(
                duracion=0.7, f_ini=300, f_fin=80, vibrato=(8, 40)),
            # representa: ternura, calma (suave, swell, vibrato leve)
            "ternura": lambda: self.vocalizar(
                duracion=0.5, freq_puntos=[(0, 400), (1, 520)],
                fm_ratio=2.0, fm_index=0.3, vibrato=(5, 8),
                ataque=0.15, caida=0.15, sostiene=0.4),
            # representa: frustracion, dolor (grave, tenso/saturado, resonante)
            "frustracion": lambda: self.vocalizar(
                duracion=0.4, f_ini=260, f_fin=170,
                fm_ratio=3.0, fm_index=4, tension=0.8, resonancia=0.4),
            # representa: curiosidad, exploracion (FM sutil medio)
            "curioso": lambda: self.vocalizar(
                duracion=0.35, f_ini=1000, f_fin=1000,
                fm_ratio=5 / 1000, fm_index=8),
            # representa: novedad (brillante, vibrato rápido)
            "novedad": lambda: self.vocalizar(
                duracion=0.3, f_ini=1500, f_fin=1500,
                fm_ratio=2.0, fm_index=1, vibrato=(22, 20)),
            # representa: saludo (arco sube-baja, amable)
            "saludo": lambda: self.vocalizar(
                duracion=0.45, freq_puntos=[(0, 500), (0.5, 800), (1, 600)],
                fm_ratio=2.0, fm_index=0.8),
            # representa: afirmacion, acuerdo, alegria (frase de dos gestos)
            "afirmativa": lambda: self.frase([
                dict(duracion=0.12, f_ini=880, fm_ratio=2.0, fm_index=1.0),
                dict(duracion=0.12, f_ini=1200, fm_ratio=2.0, fm_index=1.0),
            ], gap=0.02),
            # representa: atencion, insistencia (pulsos repetidos)
            "atencion": lambda: self.frase([
                dict(duracion=0.06, f_ini=1100, fm_ratio=2.0, fm_index=0.6),
                dict(duracion=0.06, f_ini=1100, fm_ratio=2.0, fm_index=0.6),
                dict(duracion=0.06, f_ini=1100, fm_ratio=2.0, fm_index=0.6),
            ], gap=0.05),
        }

    def barridos_del_espacio(self):
        """Recorre un eje continuo para EVIDENCIAR el espacio (lo que un banco
        fijo no puede: variar de forma continua y, así, converger o divergir)."""
        barridos = {}

        # eje TIMBRE: fm_index de 0 (puro) a 12 (metálico), tono fijo
        seg = []
        for k in np.linspace(0, 12, 6):
            seg.append(self.vocalizar(duracion=0.22, f_ini=700, f_fin=700,
                                      fm_ratio=2.0, fm_index=float(k)))
            seg.append(self._silencio(0.04))
        barridos["espacio_timbre"] = np.concatenate(seg)

        # eje TONO: de grave a agudo, timbre fijo
        seg = []
        for f in np.linspace(250, 1600, 7):
            seg.append(self.vocalizar(duracion=0.2, f_ini=float(f), f_fin=float(f),
                                      fm_ratio=2.0, fm_index=2.0))
            seg.append(self._silencio(0.04))
        barridos["espacio_tono"] = np.concatenate(seg)

        return barridos

    def exportar_demostracion(self, directorio_salida="voces_organelo"):
        if not os.path.exists(directorio_salida):
            os.makedirs(directorio_salida)

        manifiesto = {"sample_rate": self.sample_rate, "presets": [], "barridos": []}
        print("Generando demostración del organelo fonador (ARP 2600 / FM)...\n")

        print("[arquetipos: el espacio continuo contiene los contornos del banco]")
        for nombre, gen in self.presets_demostracion().items():
            senal = gen()
            ruta = os.path.join(directorio_salida, f"{nombre}.wav")
            self.escribir(senal, ruta)
            dur = len(senal) / self.sample_rate
            manifiesto["presets"].append({"nombre": nombre, "archivo": f"{nombre}.wav",
                                          "duracion_s": round(dur, 3)})
            print(f"   - {nombre+'.wav':<22} ({dur:.2f}s)")

        print("\n[barridos: el continuo que un banco fijo no puede recorrer]")
        for nombre, senal in self.barridos_del_espacio().items():
            ruta = os.path.join(directorio_salida, f"{nombre}.wav")
            self.escribir(senal, ruta)
            dur = len(senal) / self.sample_rate
            manifiesto["barridos"].append({"nombre": nombre, "archivo": f"{nombre}.wav",
                                           "duracion_s": round(dur, 3)})
            print(f"   - {nombre+'.wav':<22} ({dur:.2f}s)")

        ruta_manifiesto = os.path.join(directorio_salida, "organelo_demo.json")
        with open(ruta_manifiesto, "w", encoding="utf-8") as f:
            json.dump(manifiesto, f, ensure_ascii=False, indent=2)
        print(f"\nManifiesto: {ruta_manifiesto}")
        return manifiesto


if __name__ == "__main__":
    organo = OrganoFonador()
    organo.exportar_demostracion()
