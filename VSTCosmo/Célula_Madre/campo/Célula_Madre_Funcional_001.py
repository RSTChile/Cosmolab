#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Célula_Madre_Funcional_001 — LA CÉLULA MADRE QUE PROCESA AUDIO  ·  "QUIÉN SOY"
================================================================================

QUÉ ES ESTE ARCHIVO
-------------------
La PRIMERA versión FUNCIONAL de la Célula Madre Cosmosemiótica: integra la máquina
probada (el campo Φ de 4 hemisferios de CM001, que procesa audio como en los
experimentos VST iniciales — V103 "cartografía de estados Ω") DENTRO de la arquitectura
de organelos del genoma. Le pasas uno o varios audios y observas qué hace cada organelo,
con resultados y CHECKS al estilo de los experimentos VST anteriores.

CÓMO INTEGRA (la respuesta a "¿se hace invocando los otros scripts?")
---------------------------------------------------------------------
SÍ — pero limpio: un ORGANELO ADAPTADOR (`OrganeloSoma`) ENVUELVE la maquinaria probada
(`Hemisferio` de `VST_Celula_Madre_001.py`) y vuelca sus señales REALES al `Milieu`. El
andamiaje sintético (FuenteEntornoDemo/FuenteDemandaDemo) queda REEMPLAZADO por el soma:
ahora los bloques 5/7/8 + homeostasis reaccionan al procesamiento REAL del audio.

  AUDIO ─► OrganeloSoma (4 hemisferios Φ; audio = forzamiento de borde)
        ─► ω_A, ω_B, gradiente, Ω(estado), e_R, A_sys-env, orientación, Δ_struct, demanda, INR ─► Milieu
        ─► [B5 consciencia] [B7 libertad] [B8 evolución] [homeostasis] reaccionan
        ─► salud(): Λ_Cos, OI, invariantes κ — sobre el audio real

GRANULARIDAD: este es el enfoque GRUESO (envolver lo probado para tener funcionalidad YA).
La descomposición fina (portar campo/motor a organelos nativos validando contra v1) es la
mejora progresiva posterior.

USO
---
  venv/bin/python3 "Célula_Madre_Funcional_001.py"            # pregunta interactivamente
  venv/bin/python3 "Célula_Madre_Funcional_001.py" a.wav b.wav   # audios por argumento
  venv/bin/python3 "Célula_Madre_Funcional_001.py" demo        # señales de prueba (tono/rosa/clicks)

Formatos: WAV PCM (16/32 bit). Se convierte a mono y se remuestrea a 48 kHz.
================================================================================
"""

from __future__ import annotations
import sys, os, json, wave, struct
from datetime import datetime
import numpy as np

# soundfile (libsndfile) = ruta principal para cargar CUALQUIER WAV común. Import robusto.
try:
    import soundfile as sf
    SF_OK = True
except Exception:                       # si faltara, se usa el fallback wave (stdlib)
    sf = None; SF_OK = False
_FMT_ULTIMO = ""                        # formato/ruta de la última carga (para reporte)

# --- Maquinaria probada (campo Φ que procesa audio) ---
from VST_Celula_Madre_001 import Hemisferio, SESGO_L, SESGO_R
# --- Motor del genoma + organelos de cada bloque ---
from VST_Genoma import Organelo, Estado, Organismo, Milieu, locus_altruismo_boorman
from VST_Genoma import OrganeloPresionDesacople, OrganeloFatiga
from VST_Bloque05_ConscienciaFuncional import (
    OrganeloConscienciaBasica, OrganeloMetaRepresentacion, OrganeloSelf)
from VST_Bloque07_LibertadFuncional import (
    OrganeloJuego, OrganeloRitual, OrganeloNegacionOperativa, OrganeloLibertadFuncional)
from VST_Bloque08_DinamicaEvolutiva import (
    OrganeloMutacion, OrganeloAdaptacion, OrganeloExaptacion,
    OrganeloConscienciaMetacognitiva, OrganeloActivacionLatente)
from VST_Homeostasis import OrganeloHomeostasis

SR = 48000           # frecuencia de muestreo de trabajo (como los experimentos VST)
DT = 0.1             # paso del ciclo metabólico del genoma
SUBPASO = 0.01       # paso interno del campo Φ (estabilidad numérica, como CM001)
SIM_CAP_S = 20.0     # tope de simulación por audio (segundos)


# ==============================================================================
# CARGA DE AUDIO (real o sintético tipo experimentos iniciales)
# ==============================================================================
def _pink_noise(n, seed=44):
    rng = np.random.RandomState(seed)
    x = rng.normal(0, 1, n)
    f = np.fft.rfft(x); freqs = np.fft.rfftfreq(n, 1 / SR)
    x = np.fft.irfft(f / np.sqrt(freqs + 0.01), n=n)
    return x / (np.max(np.abs(x)) + 1e-9)

def _clicks_poisson(n, tasa=2.0, seed=44):
    rng = np.random.RandomState(seed); c = np.zeros(n)
    for _ in range(int(n / SR * tasa)):
        p = int(rng.exponential(1.0 / tasa) * SR)
        if p < n: c[p] = 1.0
    return c

def _tono(n, hz=440.0):
    t = np.arange(n) / SR
    return 0.8 * np.sin(2 * np.pi * hz * t)

def _leer_wav_stdlib(path: str):
    """FALLBACK (sin soundfile): parser RIFF/WAVE manual. Soporta PCM 8/16/24/32 entero,
    IEEE float 32/64 (wFormatTag=3) y cabeceras EXTENSIBLE (0xFFFE, subformato en el GUID).
    Devuelve (a_interleaved_float64, sr, ch, maxv, formato). Para PCM int devuelve las muestras
    crudas (sin dividir) + maxv, de modo que la aritmética de _load_wav sea IDÉNTICA a la previa."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"no es un WAV RIFF válido (cabecera {raw[:4]!r}/{raw[8:12]!r})")
    fmt_tag = ch = sr = bits = None; data = None; i = 12
    while i + 8 <= len(raw):
        cid = raw[i:i + 4]; csz = struct.unpack("<I", raw[i + 4:i + 8])[0]; body = raw[i + 8:i + 8 + csz]
        if cid == b"fmt ":
            fmt_tag, ch, sr, _br, _ba, bits = struct.unpack("<HHIIHH", body[:16])
            if fmt_tag == 0xFFFE and len(body) >= 26:        # EXTENSIBLE → subformato real en el GUID
                fmt_tag = struct.unpack("<H", body[24:26])[0]
        elif cid == b"data":
            data = body
        i += 8 + csz + (csz & 1)                              # chunks alineados a byte par
    if fmt_tag is None or data is None:
        raise ValueError("WAV sin chunk 'fmt ' o 'data'")
    if fmt_tag == 1:                                          # PCM entero
        if bits == 8:
            a = np.frombuffer(data, dtype=np.uint8).astype(np.float64) - 128.0; maxv = 128.0
        elif bits == 16:
            a = np.frombuffer(data, dtype="<i2").astype(np.float64); maxv = 2.0 ** 15
        elif bits == 24:
            b = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            v = np.where(v & 0x800000, v - 0x1000000, v)      # signo a 24 bits
            a = v.astype(np.float64); maxv = 2.0 ** 23
        elif bits == 32:
            a = np.frombuffer(data, dtype="<i4").astype(np.float64); maxv = 2.0 ** 31
        else:
            raise ValueError(f"PCM entero {bits}-bit no soportado")
        fmt = f"wave-fallback:PCM_{bits}"
    elif fmt_tag == 3:                                        # IEEE float
        if bits == 32:
            a = np.frombuffer(data, dtype="<f4").astype(np.float64)
        elif bits == 64:
            a = np.frombuffer(data, dtype="<f8").astype(np.float64)
        else:
            raise ValueError(f"IEEE float {bits}-bit no soportado")
        maxv = 1.0; fmt = f"wave-fallback:FLOAT_{bits}"
    else:
        raise ValueError(f"formato WAV no soportado (wFormatTag={fmt_tag}, bits={bits})")
    return a, int(sr), int(ch), maxv, fmt


def _leer_wav_universal(path: str):
    """Lee CUALQUIER WAV común → (a_interleaved_float64, sr, ch, maxv, formato).
    Ruta principal: soundfile (libsndfile) → PCM 8/16/24/32, float 32/64, EXTENSIBLE.
    Para PCM 16/32 se leen las muestras ENTERAS crudas (+maxv) para que _load_wav reproduzca
    la aritmética histórica BIT A BIT; el resto se lee ya normalizado en [-1,1] (maxv=1)."""
    global _FMT_ULTIMO
    if SF_OK:
        try:
            info = sf.info(path); st = (info.subtype or "").upper()
            sr, ch = int(info.samplerate), int(info.channels)
            if st == "PCM_16":                                   # sf.read → (data, sr): desempaquetar
                data, _ = sf.read(path, dtype="int16", always_2d=True)
                a = data.reshape(-1).astype(np.float64)
                _FMT_ULTIMO = f"soundfile:{st}"; return a, sr, ch, 2.0 ** 15, _FMT_ULTIMO
            if st == "PCM_32":
                data, _ = sf.read(path, dtype="int32", always_2d=True)
                a = data.reshape(-1).astype(np.float64)
                _FMT_ULTIMO = f"soundfile:{st}"; return a, sr, ch, 2.0 ** 31, _FMT_ULTIMO
            data, _ = sf.read(path, dtype="float64", always_2d=True)         # 24/8/float/extensible
            a = data.reshape(-1)
            _FMT_ULTIMO = f"soundfile:{st or info.format}"; return a, sr, ch, 1.0, _FMT_ULTIMO
        except Exception:
            pass                                              # cae al fallback wave
    a, sr, ch, maxv, fmt = _leer_wav_stdlib(path)
    _FMT_ULTIMO = fmt; return a, sr, ch, maxv, fmt


def _load_wav(path: str, binaural: bool = False):
    """Carga universal de WAV (cualquier formato común vía soundfile/fallback), normaliza a
    float64 [-1,1] y remuestrea a 48 kHz.
    - mono (binaural=False): devuelve un vector = promedio de canales (comportamiento histórico).
    - binaural=True: devuelve (izq, der, lateralidad_real); si la fuente es mono, duplica (izq=der).
    INVARIANTE: para PCM 16/32 la aritmética es la histórica → resultado IDÉNTICO bit a bit."""
    a, sr, ch, maxv, _fmt = _leer_wav_universal(path)

    def _resample(x):                               # remuestreo lineal a 48 kHz (idéntico al previo)
        if sr != SR:
            m = int(len(x) * SR / sr)
            x = np.interp(np.linspace(0, len(x), m, endpoint=False), np.arange(len(x)), x)
        return x

    if not binaural:
        # --- MONO: ORDEN EXACTO del comportamiento histórico (invariante de ingeniería) ---
        if ch > 1:
            a = a.reshape(-1, ch).mean(axis=1)      # estéreo -> mono
        a = a / maxv                                 # normaliza a [-1, 1]
        return _resample(a)

    # --- BINAURAL: dos canales por separado ---
    if ch == 1:
        canal = _resample(a / maxv)
        return canal, canal.copy(), False           # fuente mono: L=R, sin lateralidad real
    m = a.reshape(-1, ch)
    return _resample(m[:, 0] / maxv), _resample(m[:, 1] / maxv), True

def cargar_audio(spec: str, binaural: bool = False):
    """Devuelve (nombre, audio). En MONO, audio = vector [float,48kHz] (histórico).
    En BINAURAL, audio = (izq, der). Las señales 'demo' son mono de origen: en binaural se
    duplica el canal (izq=der) y se anota '(mono duplicado)' (no hay lateralidad real)."""
    dur_demo = 3.0; n = int(dur_demo * SR)
    demos = {"demo:tono": ("tono_440Hz", _tono),
             "demo:rosa": ("ruido_rosa", _pink_noise),
             "demo:clicks": ("clicks_poisson", _clicks_poisson)}
    if spec in demos:
        nombre, gen = demos[spec]; sig = gen(n)
        return (nombre + " (mono duplicado)", (sig, sig.copy())) if binaural else (nombre, sig)
    nombre = os.path.splitext(os.path.basename(spec))[0]
    if binaural:
        izq, der, real = _load_wav(spec, binaural=True)
        return (nombre if real else nombre + " (mono duplicado)", (izq, der))
    return (nombre, _load_wav(spec))


# ==============================================================================
# ORGANELO SOMA — envuelve el campo Φ probado; AUDIO -> señales reales al milieu
# ==============================================================================
class OrganeloSoma(Organelo):
    """El cuerpo: 4 hemisferios de campo Φ (CM001) que procesan el audio como
    forzamiento de borde, y vuelcan sus señales REALES al citoplasma.

    QUÉ HACE: el audio entra al sistema A (hemisferios L/R); el sistema B (B_L/B_R)
    sostiene una referencia (silencio = expectativa interna). El gradiente ω_A−ω_B es
    el desajuste audio↔expectativa, del que se derivan e_R, A_sys-env, orientación, la
    demanda del entorno (energía del audio) e Ω (estado representacional, como V103).
    CÓMO: cada ciclo metabólico (DT) ejecuta 10 subpasos del campo (SUBPASO=0.01) para
    estabilidad numérica, avanzando por el audio a 48 kHz.
    PROCEDENCIA: `Hemisferio` de VST_Celula_Madre_001.py (campo Φ robustecido, V122/v70/v72b/v80h).
    """

    SECRETA = ["omega_A", "omega_B", "gradiente", "Omega", "e_R", "A_sys_env", "orientacion",
               "delta_struct", "delta_real", "costo_trabajo", "en_reposo", "setpoint_presente",
               "INR", "demanda_entorno", "valencia_opcion", "representacion_valida",
               # --- binaural (señales NUEVAS y adicionales; no tocan ω_B ni el gradiente) ---
               "omega_L", "omega_R", "gradiente_lateral", "lateralidad_real",
               # --- binaural ampliado v2 (Req1): per-oído, energías, balance, coherencia. REAL ---
               "omega_A_L", "omega_A_R", "energia_L", "energia_R", "balance_LR",
               "lateralidad", "coherencia_biaural"]

    def __init__(self, audio, binaural: bool = False, seed: int = 44) -> None:
        super().__init__(
            nombre="soma", organelo_analogo="cuerpo sensorial (campo Φ / 4 hemisferios)",
            procedencia="VST_Celula_Madre_001.py::Hemisferio (campo Φ robustecido)",
            nodo_canonico="C-N7 interpretación · O-N3.1 R:D→S · campo=medio (V122/v70/v72b/v80h)",
            descripcion=("Procesa el audio como forzamiento de borde en 4 hemisferios Φ; deriva "
                         "ω/gradiente/Ω y las señales del entorno. BINAURAL: L=canal izq, R=canal der "
                         "→ gradiente_lateral=ω_L−ω_R (señal NUEVA; ω_A y el gradiente audio-vs-expectativa intactos)."),
            lee=[], secreta=list(self.SECRETA), depende_de=[],
            costo_base=3.0, estado=Estado.PRESENTE,
            criterio="campo finito; S>0 sostenido (ω varía, A_sys-env>0)",
        )
        self.binaural = bool(binaural)
        if self.binaural:
            izq, der = audio                                   # (izq, der)
            self._L = np.asarray(izq, dtype=np.float64)
            self._R = np.asarray(der, dtype=np.float64)
            self.lateralidad_real = not bool(np.array_equal(self._L, self._R))
            self._audio = (self._L + self._R) / 2.0            # mezcla SOLO para energía/demanda
        else:
            # MONO: comportamiento histórico EXACTO (L y R comparten el MISMO audio = mezcla mono)
            self._audio = audio.astype(np.float64)
            self._L = self._R = self._audio
            self.lateralidad_real = False
        # L recibe canal izquierdo, R canal derecho (en mono ambos = la mezcla mono)
        genL = lambda d, s: self._L
        genR = lambda d, s: self._R
        gen_sil = lambda d, s: np.zeros(int(d * s) + 16)
        # SIN SESGO (Alexis): los oídos no nacen con un sesgo opuesto (eso causaba el "oído bueno/malo").
        # Hemisferios SIMÉTRICos (mismo τ, sesgo 0) → acople simétrico que sigue al sonido, no al hemisferio;
        # y sustrato para hemisferios cerebrales que luego se especialicen por FUNCIÓN, no por nacimiento.
        self.L  = Hemisferio("L",   30.0,  genL,    seed=seed,     sesgo=0.0)
        self.R  = Hemisferio("R",   30.0,  genR,    seed=seed+100, sesgo=0.0)
        self.BL = Hemisferio("B_L", 30.0,  gen_sil, seed=seed+200, sesgo=0.0)
        self.BR = Hemisferio("B_R", 30.0,  gen_sil, seed=seed+300, sesgo=0.0)
        self.dur = len(self._L) / SR
        self.t = 0.0
        self.orient = 0.0
        # Cable A (PROPIOCEPCIÓN): la orientación de la cabeza (actuador) re-pondera el acople del soma.
        self.orient_ext = 0.0          # orientación del actuador, realimentada el paso previo (lag 1)
        self.propiocepcion = True      # ON = la orientación importa; OFF = comportamiento histórico EXACTO
        self.k_propio = 1.0            # (legacy Cable A, ya no se usa: reemplazado por la física del volumen)
        self.grad_atendido = 0.0
        self.k_vol = 0.8              # FÍSICA DEL VOLUMEN: cuánto sube/baja el oído al acercarse/alejarse de la fuente
        # Cable B (act_perm→α): abrir la membrana AFILA el acople (rigidez); perm_ext del paso previo.
        self.perm_ext = 0.0            # act_perm del actuador/circuito, realimentada (lag 1)
        # NEGATIVO HONESTO (bateria_conducta): en este soma A=calma/bajo-desajuste, así que ABRIR la
        # membrana sólo añade perturbación → A baja SIEMPRE (no afila). La palanca de acople aquí es la
        # ORIENTACIÓN (Cable A), no la permeabilidad. B2 queda DESACTIVADO (k_perm=0); hook documentado.
        self.k_perm = 0.0
        # observables del paso
        self.omega_A = self.omega_B = self.grad = 0.0
        self.omega_L = self.omega_R = self.grad_lateral = 0.0   # NUEVO (lateralidad binaural)
        self.omega_A_L = self.omega_A_R = 0.0                   # per-oído audio-vs-expectativa
        self.energia_L = self.energia_R = self.balance_LR = 0.0 # energías/balance por canal
        self.lateralidad = self.coherencia_biaural = 0.0
        self.Omega = 0.5; self.eR = 0.0; self.A = 1.0
        self.estructura = 0.0          # SENTIDO experiencial (mezcla): orden RECONOCIDO de la historia
        self.estructura_L = 0.0; self.estructura_R = 0.0   # por OÍDO → para forrajear hacia el sentido reconocido
        self._n_bands = 32
        self._K = 8                    # tamaño del REPERTORIO (codebook): voces, canciones, diálogos…
        self._codebook = np.zeros((self._K, self._n_bands))   # prototipos espectrales APRENDIDOS
        self.beta_spec = 0.03          # tasa de aprendizaje del repertorio (con el tiempo/experiencia)
        self.umbral_nuevo = 0.6        # match < umbral ∧ hay slot vacío → estructura NOVEDOSA = nuevo prototipo
        self.finito = True

    # ----- PERSISTENCIA: el codebook es lo que el organismo APRENDIÓ a reconocer (incremento 1) -----
    def snapshot(self) -> dict:
        """Estado APRENDIDO del soma a guardar: el REPERTORIO (codebook) de estructuras reconocidas.
        El campo Φ NO se guarda: se re-forma al despertar. Lo que persiste es la historia (lo aprendido),
        no el estado instantáneo del campo — como despertar recordando, con el cuerpo volviendo a su ritmo."""
        return {"codebook": self._codebook.tolist(), "_K": int(self._K), "_n_bands": int(self._n_bands)}

    def restore(self, d: dict) -> None:
        if not d:
            return
        cb = d.get("codebook")
        if cb is not None:
            arr = np.asarray(cb, dtype=np.float64)
            if arr.shape == self._codebook.shape:      # mismo repertorio (K × bandas) → restaurar
                self._codebook = arr

    def realimentar(self, audio, binaural=None) -> None:
        """CAPTURA CONTINUA: reemplaza el bloque de audio MANTENIENDO el estado del campo (Phi,
        Phi_vel) y la historia del organismo. Solo reinicia el tiempo LOCAL del bloque y la entrada
        cacheada de cada hemisferio (los generadores devuelven self._L/_R, que aquí se actualizan).
        El campo NO se reinicia → el organismo vive de forma continua a través de bloques sucesivos."""
        if binaural is None:
            binaural = self.binaural
        if binaural:
            izq, der = audio
            self._L = np.asarray(izq, dtype=np.float64); self._R = np.asarray(der, dtype=np.float64)
            self.lateralidad_real = not bool(np.array_equal(self._L, self._R))
            self._audio = (self._L + self._R) / 2.0
        else:
            self._audio = np.asarray(audio, dtype=np.float64); self._L = self._R = self._audio
            self.lateralidad_real = False
        self.L.entrada = self.R.entrada = None        # forzar regeneración → toman el nuevo bloque
        self.BL.entrada = self.BR.entrada = None
        self.dur = len(self._L) / SR
        self.t = 0.0                                  # tiempo LOCAL del bloque (Phi/Phi_vel PERSISTEN)

    def _estr_rec(self, win):
        """Membrana sensorial: tonalidad (orden) + RECONOCIMIENTO contra el repertorio (codebook).
        Devuelve (tonalidad, bins_normalizados | None, reconocimiento, índice_del_mejor_prototipo)."""
        if win.size < 512 or float(np.sqrt(np.mean(win * win))) < 1e-4:
            return 0.0, None, 0.0, 0                              # silencio / sin ventana → sin estructura
        X = np.abs(np.fft.rfft(win * np.hanning(win.size))) + 1e-12
        flat = float(np.exp(np.mean(np.log(X))) / np.mean(X))
        tonalidad = max(0.0, min(1.0, 1.0 - flat))                # ORDEN físico (¿hay estructura?)
        Xlow = X[:min(len(X), 600)]                               # rango audible bajo (tonos/voz)
        bins = np.array([b.sum() for b in np.array_split(Xlow, self._n_bands)])
        bins = bins / (np.linalg.norm(bins) + 1e-9)
        normas = np.linalg.norm(self._codebook, axis=1)           # coseno contra cada prototipo aprendido
        sims = np.where(normas > 1e-6, (self._codebook @ bins) / (normas + 1e-9), 0.0)
        bi = int(np.argmax(sims)); rec = max(0.0, float(sims[bi]))
        return tonalidad, bins, rec, bi

    def metabolizar(self, dt: float, tempo: float) -> None:
        n_sub = max(1, int(round(dt / SUBPASO)))
        # FÍSICA DEL VOLUMEN (reemplaza el Cable A artificial): el oído que se ACERCA a la fuente (según la
        # orientación de la cabeza) la oye MÁS FUERTE; el otro, más débil — como girar la cabeza para oír
        # mejor con un oído. Es una ley física (acústica direccional), NO Shannon: no hay premio ni target,
        # sólo que orientar cambia QUÉ tan fuerte entra cada canal. Centro (orient=0) → ambos 1 (invariante).
        f = max(-1.0, min(1.0, self.orient_ext / 90.0))           # facing: <0 izquierda, >0 derecha
        self.L.gain_vol = max(0.0, 1.0 - self.k_vol * f)          # facing izq (f<0) → L sube
        self.R.gain_vol = max(0.0, 1.0 + self.k_vol * f)          # facing der (f>0) → R sube
        for _ in range(n_sub):
            self.L.actualizar(self.t, SUBPASO, self.dur, self.R)
            self.R.actualizar(self.t, SUBPASO, self.dur, self.L)
            self.BL.actualizar(self.t, SUBPASO, self.dur, self.BR)
            self.BR.actualizar(self.t, SUBPASO, self.dur, self.BL)
            self.t += SUBPASO
        oL = self.L._calcular_omega(); oR = self.R._calcular_omega()
        oBL = self.BL._calcular_omega(); oBR = self.BR._calcular_omega()
        oB = (oBL + oBR) / 2.0
        oA = (oL + oR) / 2.0                                    # ω_A INTOCABLE: media L,R (como antes)
        self.omega_A, self.omega_B, self.grad = float(oA), float(oB), float(oA - oB)
        # NUEVO: lateralidad como señal ADICIONAL (NO reemplaza el gradiente audio-vs-expectativa)
        self.omega_L, self.omega_R = float(oL), float(oR)
        self.grad_lateral = float(oL - oR)                     # gradiente_lateral = ω_L − ω_R
        # binaural v2 (Req1): per-oído audio-vs-expectativa + coherencia. REAL, adicional.
        self.omega_A_L = float(oL - oBL); self.omega_A_R = float(oR - oBR)
        self.lateralidad = float(abs(oL - oR))
        pL, pR = self.L.Phi, self.R.Phi                        # coherencia = coseno(Φ_L, Φ_R)
        self.coherencia_biaural = float(np.dot(pL, pR) / (np.linalg.norm(pL) * np.linalg.norm(pR) + 1e-9))
        self.Omega = float(np.clip((oA + 1.0) / 2.0, 0.0, 1.0))     # estado representacional [0,1]
        self.dstruct = float(np.var(self.L.Phi))
        self.eR = abs(self.grad) * 30.0                             # error = desajuste audio↔expectativa
        # A = ACOPLE REAL. La orientación ya influye el campo por la FÍSICA DEL VOLUMEN (gain_vol, arriba):
        # el oído que se acerca a la fuente la oye más fuerte → cambia su perturbación. Sin truco de acople.
        self.grad_atendido = abs(self.grad)                        # (nombre conservado para el resto del código)
        self.A = float(np.clip(1.0 / (1.0 + 5.0 * abs(self.grad)), 0.05, 1.0))
        prev = self.orient
        self.orient = float(np.clip(self.orient + self.grad * 10.0 * dt, -90.0, 90.0))
        self._dorient = abs(self.orient - prev)
        idx = int(self.t * SR); w = self._audio[max(0, idx - 2400):idx + 2400]
        energia = float(np.sqrt(np.mean(w ** 2))) if w.size else 0.0
        self.demanda = 1.0 + energia * 3.0                         # energía del audio = demanda del entorno
        # MEMBRANA SENSORIAL — ESTRUCTURA RECONOCIDA (el SENTIDO nace del ORDEN y de la HISTORIA): el organismo
        # APRENDE un REPERTORIO de estructuras (codebook: voces, canciones, diálogos) y RECONOCE lo ya oído.
        # estructura = tonalidad·(0.4 + 0.6·reconocimiento). Se computa en la MEZCLA (para aprender el repertorio)
        # y por OÍDO (para forrajear hacia el sentido reconocido). Sin Shannon: aprende de lo que vive.
        ton_m, bins_m, rec_m, bi_m = self._estr_rec(w)
        if bins_m is not None and ton_m > 0.3:                     # aprende SÓLO de lo ordenado (no del ruido)
            normas = np.linalg.norm(self._codebook, axis=1)
            if rec_m < self.umbral_nuevo and float(normas.min()) < 1e-6:
                self._codebook[int(np.argmin(normas))] = bins_m    # estructura NOVEDOSA → nuevo prototipo (recuerda varias)
            else:
                self._codebook[bi_m] += self.beta_spec * (bins_m - self._codebook[bi_m])   # refuerza lo conocido
        self.estructura = max(0.0, min(1.0, ton_m * (0.4 + 0.6 * rec_m)))
        tL, _, rL, _ = self._estr_rec(self._L[max(0, idx - 2400):idx + 2400])
        tR, _, rR, _ = self._estr_rec(self._R[max(0, idx - 2400):idx + 2400])
        self.estructura_L = max(0.0, min(1.0, tL * (0.4 + 0.6 * rL)))   # estructura reconocida por oído → brújula del forrajeo
        self.estructura_R = max(0.0, min(1.0, tR * (0.4 + 0.6 * rR)))
        # binaural v2 (Req1): energías por canal y balance L/R. REAL, adicional.
        wL = self._L[max(0, idx - 2400):idx + 2400]; wR = self._R[max(0, idx - 2400):idx + 2400]
        self.energia_L = float(np.sqrt(np.mean(wL ** 2))) if wL.size else 0.0
        self.energia_R = float(np.sqrt(np.mean(wR ** 2))) if wR.size else 0.0
        _tot = self.energia_L + self.energia_R
        self.balance_LR = float((self.energia_L - self.energia_R) / (_tot + 1e-9))
        self.INR = float(np.clip(abs(self.grad) * 2.0, 0.0, 1.0))
        self.finito = bool(np.all(np.isfinite(self.L.Phi)) and np.isfinite(oA) and np.isfinite(oB))

    def secretar(self, milieu: "Milieu") -> None:
        m = milieu.secretar
        m("omega_A", self.omega_A); m("omega_B", self.omega_B); m("gradiente", self.grad)
        m("Omega", self.Omega, guardar_historial=True)
        m("e_R", self.eR); m("A_sys_env", self.A); m("orientacion", self.orient)
        m("delta_struct", self.dstruct); m("delta_real", self._dorient)
        m("costo_trabajo", self._dorient * 0.5); m("en_reposo", self._dorient < 1e-3)
        m("setpoint_presente", True); m("INR", self.INR); m("demanda_entorno", self.demanda)
        m("valencia_opcion", -self.eR); m("representacion_valida", self.A)
        # --- NUEVO: señales de lateralidad binaural (adicionales; no afectan lo anterior) ---
        m("omega_L", self.omega_L); m("omega_R", self.omega_R)
        m("estructura", self.estructura)                           # orden RECONOCIDO → convertibilidad en sentido
        m("estructura_L", self.estructura_L); m("estructura_R", self.estructura_R)   # por oído → forrajeo
        m("gradiente_lateral", self.grad_lateral)
        m("lateralidad_real", 1.0 if self.lateralidad_real else 0.0)
        # binaural v2 (Req1) — REAL, adicionales
        m("omega_A_L", self.omega_A_L); m("omega_A_R", self.omega_A_R)
        m("energia_L", self.energia_L); m("energia_R", self.energia_R)
        m("balance_LR", self.balance_LR); m("lateralidad", self.lateralidad)
        m("coherencia_biaural", self.coherencia_biaural)


# ==============================================================================
# TRANSCRIPCIÓN — CÉLULA MADRE FUNCIONAL (soma real + todos los bloques)
# ==============================================================================
def celula_madre_funcional(audio, binaural: bool = False) -> Organismo:
    o = Organismo(nombre="celula_madre_funcional", M0=1.0)
    o.expresar(OrganeloSoma(audio, binaural=binaural))   # cuerpo: procesa el audio -> señales reales
    o.expresar(OrganeloPresionDesacople())     # arousal (de e_R/A reales)
    o.expresar(OrganeloConscienciaBasica())    # B5: C_b=|R₁|
    o.expresar(OrganeloMetaRepresentacion())   # B5: R₂
    o.expresar(OrganeloSelf())                 # B5: Self
    o.expresar(OrganeloFatiga())               # tiempo biológico
    o.expresar(OrganeloRitual())               # B7: estadio 2
    o.expresar(OrganeloJuego())                # B7: estadio 1
    o.expresar(OrganeloLibertadFuncional())    # B7: LF
    o.expresar(OrganeloNegacionOperativa())    # B7: el "No"
    o.expresar(OrganeloMutacion())             # B8: variación
    o.expresar(OrganeloAdaptacion())           # B8: adaptación
    o.expresar(OrganeloExaptacion())           # B8: exaptación (XE)
    o.expresar(OrganeloConscienciaMetacognitiva())  # B8: C_m
    o.expresar(OrganeloActivacionLatente())    # B8: activación latente
    o.expresar(OrganeloHomeostasis(variable="x_interna", x0=0.5, perturb="e_R", perturb_escala=0.01))
    o.expresar(locus_altruismo_boorman())      # locus reservado (PRE)
    return o


# ==============================================================================
# PROCESAR UN AUDIO Y EMITIR RESULTADOS CON CHECKS (estilo experimentos VST)
# ==============================================================================
def procesar(nombre: str, audio, binaural: bool = False) -> dict:
    M = {"ok": "✅", "fail": "❌", "neutro": "·"}
    cel = celula_madre_funcional(audio, binaural=binaural)
    soma = cel.organelos["soma"]
    dur = soma.dur                                   # soma.dur sirve para mono y binaural (tupla)
    sim = min(dur, SIM_CAP_S)

    Omega_hist, A_hist = [], []
    cm_peak = 0.0; juego_on = ritual_on = neg_on = exapto = False
    pasos = max(1, int(sim / DT))
    for _ in range(pasos):
        cel.vivir_un_paso(DT)
        Omega_hist.append(soma.Omega); A_hist.append(soma.A)
        cm_peak = max(cm_peak, cel.organelos["consciencia_metacognitiva"].C_m)
        juego_on = juego_on or cel.organelos["juego"].activo
        ritual_on = ritual_on or cel.organelos["ritual"].active
        neg_on = neg_on or cel.organelos["negacion_operativa"].negacion
        exapto = exapto or cel.organelos["exaptacion"].activa

    s = cel.salud()
    cb = cel.organelos["consciencia_basica"]; r2 = cel.organelos["meta_representacion"]
    lf = cel.organelos["LF"]; ex = cel.organelos["exaptacion"]
    fat = cel.organelos["fatiga"]
    Omega_med = float(np.mean(Omega_hist)); A_min = float(np.min(A_hist))
    S_sostenido = (fat.historia > 0.0) and (A_min > 0.0) and (np.std(Omega_hist) > 1e-6)
    inv_ok = sum(1 for v in s["invariantes"].values() if v)

    print("=" * 78)
    print(f"AUDIO: '{nombre}'  ·  {len(audio)} muestras · {dur:.2f}s @ {SR//1000}kHz · sim {sim:.1f}s")
    print("=" * 78)
    print(f"  {M['ok'] if soma.finito else M['fail']} campo Φ finito (sin NaN/Inf)")
    print(f"  · Ω (estado representacional) = {Omega_med:.4f}   [rango 0..1]   "
          f"ω_A={soma.omega_A:.3f} ω_B={soma.omega_B:.3f} grad={soma.grad:.3f}")
    print(f"  {M['ok'] if S_sostenido else M['fail']} S>0 sostenido  "
          f"(historia={fat.historia:.1f}, A_sys-env_min={A_min:.3f}, Ω varía)")
    print(f"  {M['ok'] if inv_ok==6 else M['fail']} invariantes de viabilidad κ: {inv_ok}/6")
    print(f"  · ORGANELOS que reaccionaron al audio:")
    print(f"      consciencia: C_b=|R₁|={cb.C_b}  R₂={r2.R2:.3f}")
    print(f"      libertad:    {lf.ESCALA[lf.nivel].split('(')[0].strip()}   "
          f"juego={M['ok'] if juego_on else M['neutro']}  "
          f"ritual={M['ok'] if ritual_on else M['neutro']}  "
          f"negación('No')={M['ok'] if neg_on else M['neutro']}")
    print(f"      evolución:   exaptación={M['ok'] if exapto else M['neutro']} "
          f"(Ωop={ex.Omega_op:.2f}, XE={min(1.0,ex.XE):.3f})   C_m_pico={cm_peak:.2f}")
    if binaural:
        print(f"      binaural:    ω_L={soma.omega_L:.3f} ω_R={soma.omega_R:.3f} "
              f"grad_lateral={soma.grad_lateral:.3f}  lateralidad_real={soma.lateralidad_real}")
    print(f"  → Λ_Cos={s['Lambda_Cos']:.3f}   OI={s['OI']:.3f} → {s['nivel_OI'].upper()}")
    out = {"audio": nombre, "duracion_s": round(dur, 3), "Omega_medio": round(Omega_med, 4),
           "omega_A": round(soma.omega_A, 4), "omega_B": round(soma.omega_B, 4),
           "campo_finito": bool(soma.finito), "S_sostenido": bool(S_sostenido),
           "invariantes_ok": f"{inv_ok}/6", "C_b": cb.C_b, "R2": round(r2.R2, 3),
           "LF_nivel": lf.nivel, "juego": juego_on, "ritual": ritual_on, "negacion": neg_on,
           "exaptacion": exapto, "XE": round(min(1.0, ex.XE), 3), "C_m_pico": round(cm_peak, 3),
           "Lambda_Cos": round(s["Lambda_Cos"], 4), "OI": round(s["OI"], 4), "nivel_OI": s["nivel_OI"]}
    if binaural:                                      # campos nuevos solo en modo binaural
        out.update({"binaural": True, "omega_L": round(soma.omega_L, 4),
                    "omega_R": round(soma.omega_R, 4),
                    "gradiente_lateral": round(soma.grad_lateral, 4),
                    "lateralidad_real": bool(soma.lateralidad_real)})
    return out


# ==============================================================================
# ENTRADA INTERACTIVA + ORQUESTACIÓN
# ==============================================================================
def pedir_specs(argv: list[str]) -> list[str]:
    if argv:
        specs = argv
    else:
        print("Célula Madre Funcional — ¿qué audio(s) cargar?")
        raw = input("  rutas .wav separadas por coma (o 'demo' para señales de prueba): ").strip()
        specs = [x.strip() for x in raw.split(",") if x.strip()] or ["demo"]
    out: list[str] = []
    for s in specs:
        out += ["demo:tono", "demo:rosa", "demo:clicks"] if s == "demo" else [s]
    return out


def main(argv: list[str]) -> None:
    specs = pedir_specs(argv)
    binaural = "binaural" in specs                   # token 'binaural' activa el modo estéreo L/R
    specs = [s for s in specs if s != "binaural"]
    print("\n" + "#" * 78)
    print(f"#  CÉLULA MADRE FUNCIONAL 001 — procesando audio(s)  [{'BINAURAL L/R' if binaural else 'MONO'}]")
    print("#" * 78 + "\n")
    resultados = []
    for spec in specs:
        try:
            nombre, audio = cargar_audio(spec, binaural=binaural)
        except Exception as e:
            print(f"  ❌ no se pudo cargar '{spec}': {e}\n"); continue
        resultados.append(procesar(nombre, audio, binaural=binaural))
        print()

    if len(resultados) > 1:                 # cartografía comparativa (como V103)
        print("=" * 78)
        print("CARTOGRAFÍA Ω — comparación entre estímulos (estado representacional inducido)")
        print("=" * 78)
        print(f"  {'audio':<18}{'Ω':>8}{'OI':>8}  {'nivel':<16}{'organelos activos'}")
        for r in resultados:
            act = ",".join(k for k in ("juego", "ritual", "negacion", "exaptacion") if r[k])
            print(f"  {r['audio']:<18}{r['Omega_medio']:>8.4f}{r['OI']:>8.3f}  "
                  f"{r['nivel_OI']:<16}{act or '—'}")
        print("=" * 78)

    if resultados:
        os.makedirs("CelulaMadre_logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta = f"CelulaMadre_logs/funcional_{ts}.json"
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump({"timestamp": ts, "resultados": resultados}, f, indent=2, ensure_ascii=False)
        print(f"\n  📁 Resultados guardados: {ruta}")


if __name__ == "__main__":
    main(sys.argv[1:])
