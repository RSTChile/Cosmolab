#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_CelulaMadre_WebLive — LABORATORIO DE OBSERVACIÓN EN VIVO DE LA CÉLULA MADRE
================================================================================
Siguiente versión de la interfaz (Req 1-10). Backend Python (stdlib, sin deps nuevas
salvo OPCIONAL `sounddevice` para audio en vivo). NO reemplaza `VST_CelulaMadre_Web.py`
(que queda intacto) ni el motor validado: lo ENVUELVE.

QUÉ APORTA vs la interfaz anterior
  · Entrada BIAURAL real: dos canales (un archivo por oído, o estéreo, o mono duplicado).
  · Selector directo de audios del proyecto (/audios) — canal L y canal R.
  · Audio EN VIVO (micrófono / sistema vía BlackHole) — OPCIONAL, requiere sounddevice.
  · Audio EN VIVO vía SERVIDOR TCP (VST_AudioServer.py): este laboratorio actúa como CONSUMIDOR
    (igual que VST_AudioCliente.py). Los canales del servidor aparecen como fuentes '📡 device —
    canal N' con los MISMOS nombres que la conexión directa, pero el dato viaja por red desde el
    Mac (que sí ve la Rødecaster) en vez de abrir el dispositivo directo. La elección de canal L/R
    es del consumidor (esta Web). Soluciona que Docker/este host no capturen CoreAudio directo.
  · Gráficos EN TIEMPO REAL por streaming SSE (no al final).
  · 8 ventanas fisiológicas + timeline de eventos.
  · Monitor compacto / Laboratorio completo. Ablación por interruptor (real).
  · CSV con columnas antiguas (compat) + binaurales + de observación, y metadatos de apagados.

QUÉ ES REAL vs ANDAMIAJE (trazabilidad, Req 8)
  · REAL: el motor (campo Φ + organelos), las señales binaurales (energías/coherencia/
    lateralidad), la ablación (expresar=False saca del ciclo), el streaming de cada paso.
  · ANDAMIAJE/SIMULACIÓN: "biauralizar mono" (delay+gain interaural) — claramente marcado
    como simulación, no biaural real. El audio del SISTEMA en macOS requiere un dispositivo
    de loopback (BlackHole/Loopback): sin él, macOS NO entrega el audio del sistema al programa.

CÓMO CORRER
    venv/bin/python3 VST_CelulaMadre_WebLive.py   → http://localhost:7788
================================================================================
"""
from __future__ import annotations
import os, sys, json, base64, tempfile, glob, threading, queue, time
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# --- arranque de rutas: la Célula Madre vive en subcarpetas (genoma/campo/organelos/diada/web/audio).
# Pone cada órgano en sys.path ANTES de importar el núcleo. Aditivo y guardado por isdir: inocuo si
# todavía está plano (las subcarpetas no existen → no se añaden; la raíz sigue en el path). ---
_CM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../Célula_Madre/ tras mudar
for _d in ("genoma", "campo", "organelos", "diada", "web", "audio"):
    _p = os.path.join(_CM, _d)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from VST_RC_B import OrganoRC, COLS_RC
from VST_HomeostasisEmergente import (HomeostasisEmergente, soporte_A_sys_env,
                                      permeabilidad_activa, COLS_HOMEO_EMERGENTE)
from VST_Memoria import OrganeloMemoria, COLS_MEM
from VST_Metabolismo import OrganeloMetabolismo, COLS_MET

# Reutiliza el motor validado y el catálogo de organelos de la interfaz anterior
from VST_CelulaMadre_Web import cmf, ORG_UI

PUERTO = int(os.environ.get("VST_PUERTO", "7799"))  # Organismo B por defecto
SR = cmf.SR
DT = cmf.DT
AQUI = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.abspath(os.path.join(_CM, "..", "audio_binaural"))  # audio_binaural NO se mueve: queda en la raíz VSTCosmo
if not os.path.isdir(AUDIO_DIR):
    AUDIO_DIR = os.path.join(AQUI, "audio_binaural")                     # fallback (estructura plana)

# --- Audio en vivo: OPCIONAL (sounddevice). Si no está, se degrada con instrucciones. ---
try:
    import sounddevice as _sd
    SD_OK = True; SD_ERR = ""
except Exception as e:                                  # ImportError u OSError (PortAudio)
    _sd = None; SD_OK = False; SD_ERR = f"{type(e).__name__}: {e}"

# --- Puente de audio por red (VST_AudioServer.py): consumir el stream TCP como fuente. ---
# Quién soy: el enlace que convierte a ESTE laboratorio en un CONSUMIDOR del servidor (igual que
# VST_AudioCliente.py). El servidor corre NATIVO en el Mac y sí ve la Rødecaster; aquí elegimos el
# canal del lado cliente y recibimos el dato por TCP. Solo necesita numpy+socket (stdlib): el import
# NO arrastra sounddevice (VST_AudioServer lo importa con try/except), así que funciona aunque falte.
try:
    from VST_AudioServer import AudioStreamClient
    SERV_OK = True; SERV_ERR = ""
except Exception as e:
    AudioStreamClient = None; SERV_OK = False; SERV_ERR = f"{type(e).__name__}: {e}"

SERVIDOR_HOST = os.environ.get("VST_SERVIDOR_HOST", "127.0.0.1")   # en Docker = host.docker.internal (el AudioServer corre NATIVO en el Mac)
SERVIDOR_PORT = int(os.environ.get("VST_SERVIDOR_PORT", "8765"))          # puerto por defecto del servidor de audio
AUDIO_VIVO_DIRECTO_DESHABILITADO = os.environ.get("VST_DISABLE_DIRECT_AUDIO", "1") == "1"  # usar VST_AudioServer.py para A/B

# --- Organo de comunicacion: la voz del organismo como fuente consumible por su par. ---
try:
    from VST_OrganoComunicacion import OrganoComunicacion, audio_desde_url
    COM_OK = True; COM_ERR = ""
except Exception as e:
    OrganoComunicacion = None; audio_desde_url = None
    COM_OK = False; COM_ERR = f"{type(e).__name__}: {e}"

ORGANISMO_ID = os.environ.get("VST_ORGANISMO_ID", "ANIMA_B")
COMUNICACION_PEER_PORT = int(os.environ.get("VST_COMUNICACION_PEER_PORT", "7788"))
COMUNICACION_PEER_URL = os.environ.get(
    "VST_COMUNICACION_PEER",
    f"http://127.0.0.1:{COMUNICACION_PEER_PORT}/comunicacion/bloque.wav",
)
ORGANO_COMUNICACION = OrganoComunicacion(ORGANISMO_ID, SR) if COM_OK else None
ORGANO_RC = OrganoRC()
# H canónica fila-level (O-N9.14): A_sys-env estable sostenido por la competencia ICR↔IRDE (de RC).
# Banda viable EMERGENTE (no setpoint). Se resetea por corrida en _start.
HOMEO_EMERGENTE = HomeostasisEmergente()
# Historia interna del organismo (6 capas). Lee soma+milieu+fila; conecta Cb→act_perm=necesidad.
# Se resetea por corrida en _start (cada corrida = una vida).
MEMORIA = OrganeloMemoria()
# Economía energética: come la experiencia (nutritiva/tóxica), paga el costo de vivir/actuar, se
# degrada (basal) y se repone. Secreta met_nutricion → la memoria la usa para saciar la necesidad.
METABOLISMO = OrganeloMetabolismo()

# --- PERSISTENCIA (incremento 1): la HISTORIA del organismo sobrevive al apagón. El espacio en disco
# (futuro VOLUMEN Docker, vía ANIMA_ESTADO_DIR) lo da vst_persistencia; aquí sólo coordinamos
# DESPERTAR (restaurar al nacer la célula) y GUARDAR (autosave). Aditivo: si el módulo no está, el
# organismo sigue vivo igual — sólo no recordaría entre reinicios. ---
try:
    from vst_persistencia import (guardar as _persist_guardar, cargar as _persist_cargar,
                                  restaurar as _persist_restaurar)
    PERSIST_OK = True; PERSIST_ERR = ""
except Exception as _e:
    PERSIST_OK = False; PERSIST_ERR = f"{type(_e).__name__}: {_e}"
AUTOSAVE_S = float(os.environ.get("ANIMA_AUTOSAVE_S", "15"))   # cada cuánto el organismo graba su estado

def _organelos_persistibles(soma):
    """Los organelos cuya historia se guarda: memoria + metabolismo (globals) + el codebook del soma."""
    orgs = {"memoria": MEMORIA, "metabolismo": METABOLISMO}
    if soma is not None:
        orgs["soma"] = soma
    return orgs

def _despertar(soma):
    """Al nacer la célula: restaura de disco la historia previa de ESTE organismo (ORGANISMO_ID).
    Devuelve los organelos restaurados ([] si nace por primera vez o no hay persistencia)."""
    if not PERSIST_OK:
        return []
    try:
        return _persist_restaurar(_organelos_persistibles(soma), _persist_cargar(ORGANISMO_ID))
    except Exception:
        return []

def _guardar_estado(soma):
    """Graba la historia actual a disco (atómico). Silencioso ante fallos: nunca debe matar la vida."""
    if not PERSIST_OK or soma is None:
        return None
    try:
        return _persist_guardar(ORGANISMO_ID, _organelos_persistibles(soma))
    except Exception:
        return None

def _autosave_daemon():
    """Hilo de fondo: graba el estado del organismo VIVO cada AUTOSAVE_S s (también tras Detener),
    para que un apagón brusco del Mac pierda como mucho ese intervalo."""
    while True:
        time.sleep(AUTOSAVE_S)
        r = RUN
        if r is not None and getattr(r, "soma", None) is not None:
            _guardar_estado(r.soma)

COMUNICACION_VOICE_GAIN = float(os.environ.get("VST_VOICE_GAIN", "20.0"))
ORGANISMO_LABEL = os.environ.get("VST_ORGANISMO_LABEL", "Organismo B")

# --- Gobernanza de ALTRUISMO de la díada: aplica el locus del genoma (O-N22) a la comunicación A↔B.
#     Lee el estado del par, conduce el locus, y MODULA la voz como señal costosa (∝ disposición). ---
try:
    from VST_DiadaAltruismo import GobernanzaAltruismo, leer_estado_par, url_estado_desde_voz
    GOB_ALTRUISMO = GobernanzaAltruismo(base_voice_rms=float(os.environ.get("VST_VOICE_TARGET_RMS", "0.40")))
    COMUNICACION_PEER_ESTADO_URL = url_estado_desde_voz(COMUNICACION_PEER_URL)
    DIADA_OK = True; DIADA_ERR = ""
except Exception as e:
    GOB_ALTRUISMO = None; COMUNICACION_PEER_ESTADO_URL = None
    DIADA_OK = False; DIADA_ERR = f"{type(e).__name__}: {e}"
_PAR_ESTADO = {"t": 0.0, "data": None}   # caché throttled del estado del par (no se pide en cada paso)

# Tabla REAL de canales multitrack de la Rødecaster Pro (verificada por Alexis 23-jun-2026, 1-based en
# la consola → 0-based aquí). Permite elegir el canal por su nombre sin contar posiciones. Pares L/R:
#   1-2 Main Mix · 3-4 Combo 1 · 5-6 Combo 2 · 7-8 Combo 3 · 9-10 Bluetooth ·
#   11-12 USB 2 · 13-14 USB Main · 15-16 SMART Pads · 17-18 USB Chat
# Para canales fuera de tabla (la interfaz expone 20) o dispositivos que no sean Rødecaster → 'canal N'.
RODE_CANALES = [
    "Main Mix (L)", "Main Mix (R)",
    "Combo 1 (L)", "Combo 1 (R)", "Combo 2 (L)", "Combo 2 (R)", "Combo 3 (L)", "Combo 3 (R)",
    "Bluetooth (L)", "Bluetooth (R)",
    "USB 2 (L)", "USB 2 (R)", "USB Main (L)", "USB Main (R)",
    "SMART Pads (L)", "SMART Pads (R)", "USB Chat (L)", "USB Chat (R)",
]


def _nombre_canal(device: str, idx0: int) -> str:
    """Nombre legible de un canal: 'canal 3 · Combo 1' para la Rødecaster; 'canal N' para el resto."""
    n = idx0 + 1
    if "rode" in (device or "").lower() and 0 <= idx0 < len(RODE_CANALES):
        return f"canal {n} · {RODE_CANALES[idx0]}"
    return f"canal {n}"

# --- Columnas del CSV: 22 ANTIGUAS (compat) + binaurales + observación ---
COLS_BASE = ["t", "Omega", "omega_A", "omega_B", "gradiente", "e_R", "A_sys_env",
             "presion_desacople", "C_b", "R2", "LF_op", "lf_nivel", "juego", "ritual",
             "negacion", "demanda_entorno", "Omega_op", "XE", "C_m", "H_homeostasis",
             "OI", "Lambda_Cos"]
COLS_BIN = ["omega_L", "omega_R", "omega_A_L", "omega_A_R", "energia_L", "energia_R",
            "balance_LR", "lateralidad", "coherencia_biaural"]
COLS_OBS = ["LF_struct", "self_coherencia", "x_interna", "en_rango", "mutacion",
            "adaptacion_activa", "exaptacion_activa", "activacion_latente", "invariantes_ok",
            "estructura", "estructura_L", "estructura_R"]
COLS_ACT = ["act_orientacion_deg", "act_objetivo_deg", "act_delta_deg", "act_confianza", "act_fatiga", "act_zona_muerta", "act_temblor_rms", "act_lateralidad_dw", "act_atencion_L", "act_atencion_R", "act_comprension_L", "act_comprension_R", "act_riesgo_L", "act_riesgo_R", "act_consenso_RC", "act_conflicto_RC", "act_freno_RC", "act_rc_mix", "act_presencia_L", "act_presencia_R", "act_propuesta_atencional", "act_decision_RC", "act_bloqueo_IRDE", "act_permiso_decisional", "act_evidencia_L", "act_evidencia_R", "act_razon_L", "act_razon_R", "act_necesidad_cierre", "act_decision_organismica", "act_soporte_sentido", "act_vulnerabilidad_riesgo", "act_base_sentido", "act_base_riesgo", "act_peso_sentido", "act_peso_riesgo",
    "act_comp_gain_eff", "act_k_motor_eff", "act_persistencia_decision", "act_claridad_estimulo",
    "act_error_motor", "act_mejora_motor", "act_adaptacion_motor", "act_adaptacion_comprension"]
COLS = COLS_BASE + COLS_BIN + COLS_OBS + COLS_ACT + COLS_RC + COLS_HOMEO_EMERGENTE + COLS_MET + COLS_MEM



class ActuadorEsferaV122:
    """Actuador observacional V122+.

    La esfera NO sigue volumen. La energía sólo funciona como compuerta de presencia:
    si un oído está cortado o en silencio, ese oído no puede capturar atención.
    La dirección viene de saliencia fisiológica lateral, no de RMS bruto.

    Regla V122–V150:
      - Lateralidad interna propone dirección.
      - Presencia de canal evita atender un oído apagado.
      - R2/LF/H/A_sys_env/XE modulan confianza/intensidad.
      - Inercia, zona muerta, freno y fatiga evitan giro reflejo.
    """
    def __init__(self):
        self.theta = 0.0
        self.ultimo_delta = 0.0
        self.fatiga = 0.0
        self.temblores = []
        self.k_lateralidad = 65.0        # antes 1500: saturaba siempre a ±90°
        self.inercia = 0.88
        self.k_motor = 0.060
        self.K_HAMBRE = 1.5              # Cable B: cuánto ENERGIZA el hambre el movimiento (SEEKING); no dirige
        self.K_EXPLOR = 60.0            # Cable B/SEEKING: amplitud (°) del barrido exploratorio bajo hambre
        self.SCAN_VEL = 0.05            # velocidad del barrido (rad/paso); UNDIRECTED, no esteerado por A
        self._scan_phase = 0.0
        # Cable C — APRENDIZAJE de la orientación nutritiva (sesga el escaneo por MEMORIA, no por reward):
        self.K_BIAS = 1.0              # cuánto sesga la memoria el centro del escaneo hacia lo nutritivo
        self.ETA_ORIENT = 0.15        # tasa de aprendizaje del valor por orientación (rápida: pocas visitas bastan)
        # extinción ACTIVA: la EMA-al-visitar baja el valor cuando esa orientación deja de nutrir (regla 7);
        # el decay global es LENTO (sólo olvida lo nunca revisitado) para NO erosionar buckets nutritivos raros.
        self.DECAY_ORIENT = 0.9995
        self.CONF_REF = 0.10          # nutrición de referencia para la "confianza" del recuerdo (arranca el sesgo)
        self.valor_orient = {}        # mapa orientación(bucket 15°) → valor nutritivo aprendido
        self._theta_prev = 0.0
        self.DWELL = 40               # saccade-and-fixate (sólo Cable C): pasos que muestrea una orientación
        self.SCAN_STEP = 1.3          # salto (rad) entre orientaciones de muestreo (spread, no contiguo)
        self._dwell = 0
        self.forrajeo_c = False       # Cable C (aprendizaje+sesgo+hambre-domina): EXPERIMENTAL, NO converge aún
        #   → default OFF (la conducta default = Cable B validado: hambre energiza barrido alrededor de RC).
        self.zona_base = 2.0
        self.zona_por_fatiga = 0.004
        self.fatiga_decay = 0.9985
        self.temblor_base = 0.06
        self.energy_floor = 0.003        # bajo esto el canal se considera ausente
        self.energy_scale = 0.050        # escala de compuerta, no de dirección
        # --- ADAPTACIÓN ORGANÍSMICA (anti-Shannon): la ganancia de comprensión y la respuesta motora
        #     NO son constantes elegidas para "girar"; EMERGEN del cierre del propio organismo. Las
        #     constantes de abajo son FISIOLÓGICAS y declaradas (techos, tasas, escalas), no ajustes
        #     de orientación: acotan y temporizan la adaptación, no la dirigen a un resultado visual.
        self.comp_gain_eff = 1.0         # ganancia de comprensión EFECTIVA; parte en 1 (sin amplificar)
        self.GAIN_MIN = 1.0              # piso: nunca por debajo de la comprensión cruda (no inhibir el sentido)
        self.GAIN_MAX = 8.0              # techo fisiológico de amplificación de la comprensión
        self.tau_gain = 0.02             # tasa de re-ponderación de la comprensión (aprendizaje LENTO)
        self.escala_error = 5.0          # grados sobre los que una mejora motora cuenta como "significativa"
        self.sens_irde = 4.0             # sensibilidad del castigo al aumento de desviación (IRDE)
        self.sens_aenv = 4.0             # sensibilidad del castigo a la caída de acoplamiento (A_sys_env)
        self.ema_dec = 0.06              # memoria de la decisión (para medir persistencia de signo)
        self._dec_ema = 0.0; self._absdec_ema = 0.0
        self._Aenv_prev = None; self._irde_prev = None
        self._conf_prev = 0.0; self._error_prev = 0.0

    @staticmethod
    def _clamp(x, a, b):
        try:
            x = float(x)
        except Exception:
            x = 0.0
        if not np.isfinite(x):
            x = 0.0
        return max(a, min(b, x))

    def _presencia(self, energia):
        e = max(0.0, float(energia or 0.0) - self.energy_floor)
        # Comp puerta suave: presencia, no volumen. A partir de ~0.05 ya casi satura.
        return self._clamp(1.0 - np.exp(-e / self.energy_scale), 0.0, 1.0)

    def actualizar(self, fila: dict) -> dict:
        wL = float(fila.get("omega_L", 0.0) or 0.0)
        wR = float(fila.get("omega_R", 0.0) or 0.0)
        wA = float(fila.get("omega_A", 0.0) or 0.0)
        eL = float(fila.get("energia_L", 0.0) or 0.0)
        eR = float(fila.get("energia_R", 0.0) or 0.0)

        pL = self._presencia(eL)
        pR = self._presencia(eR)

        # Saliencia fisiológica por oído: cuánto se separa cada oído del estado integrado.
        # La energía sólo habilita la existencia del canal; no decide el lado por volumen.
        salL = abs(wL - wA) * pL
        salR = abs(wR - wA) * pR
        dw = salR - salL

        R2 = self._clamp(fila.get("R2", 0.0), 0.0, 1.0)
        LF = self._clamp(fila.get("LF_op", 0.0), 0.0, 1.0)
        H = self._clamp(fila.get("H_homeostasis", 0.0), 0.0, 1.0)
        OI = self._clamp(fila.get("OI", 0.0), 0.0, 1.0)
        Aenv = self._clamp(fila.get("A_sys_env", 0.0), 0.0, 1.0)
        XE = self._clamp(fila.get("XE", 0.0), 0.0, 1.0)
        eR = self._clamp(abs(fila.get("e_R", 0.0)), 0.0, 1.0)

        at_l = self._clamp(fila.get("RC_atencion_L", salL), 0.0, 1.0) * pL
        at_r = self._clamp(fila.get("RC_atencion_R", salR), 0.0, 1.0) * pR
        # Comprensión EFECTIVA: amplificada por la ganancia ADAPTATIVA (emergente del cierre, no fija).
        comp_l = self._clamp(self.comp_gain_eff * fila.get("RC_comprension_L", 0.0), 0.0, 1.0) * pL
        comp_r = self._clamp(self.comp_gain_eff * fila.get("RC_comprension_R", 0.0), 0.0, 1.0) * pR
        riesgo_l = self._clamp(fila.get("RC_riesgo_L", 0.0), 0.0, 1.0) * pL
        riesgo_r = self._clamp(fila.get("RC_riesgo_R", 0.0), 0.0, 1.0) * pR
        icr_ratio = self._clamp(fila.get("ICR_ratio", 0.5), 0.0, 1.0)
        irde_ratio = self._clamp(fila.get("IRDE_ratio", 0.5), 0.0, 1.0)
        freno_rc = self._clamp(fila.get("RC_freno_riesgo", irde_ratio), 0.0, 1.0)
        comprension_rc = self._clamp(fila.get("RC_confianza_comprension", icr_ratio), 0.0, 1.0)

        media = lambda vals: self._clamp(sum(float(v) for v in vals) / max(1, len(vals)), 0.0, 1.0)

        # Cadena E018:
        # presencia sensorial -> habilita evidencia
        # lateralidad -> evidencia, no causa
        # RC / ICR / IRDE -> razon organismica
        # fatiga / inercia -> encarnan
        evidencia_l = media([at_l, salL])
        evidencia_r = media([at_r, salR])
        propuesta_atencional = self._clamp(evidencia_r - evidencia_l, -1.0, 1.0)
        evidencia_total = self._clamp(evidencia_l + evidencia_r, 0.0, 1.0)
        riesgo_total = self._clamp(riesgo_l + riesgo_r, 0.0, 1.0)
        comprension_total = self._clamp(comp_l + comp_r, 0.0, 1.0)
        integracion = self._clamp((max(R2, 1e-6) * max(LF, 1e-6) * max(H, 1e-6) * max(OI, 1e-6) * max(Aenv, 1e-6)) ** 0.2, 0.0, 1.0)

        C_m = self._clamp(fila.get("C_m", 0.0), 0.0, 1.0)
        fatiga_norm = self._clamp(self.fatiga / (1.0 + self.fatiga), 0.0, 1.0)

        necesidad_cierre = media([1.0 - OI, 1.0 - H, 1.0 - Aenv, eR, C_m, 1.0 - LF])
        necesidad_relacional = self._clamp(necesidad_cierre * media([R2, evidencia_l]), 0.0, 1.0)
        necesidad_externa = self._clamp(necesidad_cierre * media([Aenv, evidencia_r]), 0.0, 1.0)

        # E017: la ponderacion sentido/riesgo nace del estado del organismo, no de un coeficiente fijo.
        soporte_sentido = media([OI, H, C_m, LF, integracion, comprension_rc])
        vulnerabilidad_riesgo = media([1.0 - OI, 1.0 - H, 1.0 - LF, eR, fatiga_norm, freno_rc])
        base_sentido = self._clamp(icr_ratio * soporte_sentido * media([comprension_total, evidencia_total, comprension_rc]), 0.0, 1.0)
        base_riesgo = self._clamp(irde_ratio * vulnerabilidad_riesgo * media([riesgo_total, freno_rc, 1.0 - integracion]), 0.0, 1.0)
        den_pesos = base_sentido + base_riesgo
        if den_pesos > 1e-9:
            peso_sentido = self._clamp(base_sentido / den_pesos, 0.0, 1.0)
            peso_riesgo = self._clamp(base_riesgo / den_pesos, 0.0, 1.0)
        else:
            peso_sentido = 0.0
            peso_riesgo = 0.0

        captura_l = media([comp_l, necesidad_relacional, comprension_rc * evidencia_l])
        captura_r = media([comp_r, necesidad_externa, comprension_rc * evidencia_r])
        amenaza_l = media([riesgo_l, freno_rc * evidencia_l, irde_ratio * riesgo_l])
        amenaza_r = media([riesgo_r, freno_rc * evidencia_r, irde_ratio * riesgo_r])

        razon_l = self._clamp(evidencia_l * (peso_sentido * captura_l - peso_riesgo * amenaza_l), -1.0, 1.0)
        razon_r = self._clamp(evidencia_r * (peso_sentido * captura_r - peso_riesgo * amenaza_r), -1.0, 1.0)

        decision_organismica = self._clamp(razon_r - razon_l, -1.0, 1.0)
        conflicto_rc = self._clamp(abs(propuesta_atencional - decision_organismica) * 0.5, 0.0, 1.0)
        bloqueo_irde = self._clamp(irde_ratio * media([riesgo_total, freno_rc, conflicto_rc, 1.0 - integracion]), 0.0, 1.0)
        permiso_decisional = self._clamp((max(comprension_rc, 0.0) * max(comprension_total, 0.0) * max(integracion, 0.0)) ** (1.0 / 3.0), 0.0, 1.0)
        decision_rc = self._clamp(decision_organismica * permiso_decisional * (1.0 - bloqueo_irde), -1.0, 1.0)

        if pL < 0.05 and pR < 0.05:
            objetivo = 0.0
            propuesta_atencional = 0.0
            decision_organismica = 0.0
            decision_rc = 0.0
            conflicto_rc = 0.0
            bloqueo_irde = 0.0
            permiso_decisional = 0.0
            necesidad_cierre = 0.0
            soporte_sentido = 0.0
            vulnerabilidad_riesgo = 0.0
            base_sentido = 0.0
            base_riesgo = 0.0
            peso_sentido = 0.0
            peso_riesgo = 0.0
            razon_l = 0.0
            razon_r = 0.0
        else:
            # BRÚJULA POR ENERGÍA SEMIÓTICA (Alexis): la dirección emerge de DÓNDE se CONVIERTE más energía
            # semiótica en sentido (ICES) — la comprensión por oído (que SÍ sigue al sonido). Orientar hacia
            # ahí es ir a donde el organismo VIVE más (enérgeia), no a un compás arbitrario de RC. RC gatea
            # (confianza/permiso en k_motor_eff; bloqueo_irde veta el riesgo). Cable A/volumen lo hace físico.
            # BRÚJULA HACIA EL SENTIDO RECONOCIDO: orienta hacia el oído con más ESTRUCTURA reconocida de la
            # historia (estructura_L/R) — voces/melodías/diálogos que el organismo conoce. Si no hay estructura
            # reconocida, cae a la comprensión por oído. RC gatea; Cable A/volumen lo hace físico.
            eL = self._clamp(fila.get("estructura_L", 0.0), 0.0, 1.0)
            eR = self._clamp(fila.get("estructura_R", 0.0), 0.0, 1.0)
            cL = self._clamp(fila.get("RC_comprension_L", 0.0), 0.0, 1.0)
            cR = self._clamp(fila.get("RC_comprension_R", 0.0), 0.0, 1.0)
            sL = eL + cL; sR = eR + cR                            # sentido reconocido por oído (estructura + comprensión)
            compas = (sR - sL) / (sR + sL + 1e-6)                 # >0 der, <0 izq (hacia el sentido)
            objetivo = self._clamp(self.k_lateralidad * compas * (1.0 - bloqueo_irde), -90.0, 90.0)

        confianza_base = media([R2, LF, H, Aenv, XE])
        confianza = self._clamp(confianza_base * media([permiso_decisional, 1.0 - bloqueo_irde, soporte_sentido]), 0.0, 1.0)

        # --- VARIABLES ADAPTATIVAS (emergentes, no fijas) ---
        # claridad del estímulo = asimetría NORMALIZADA de la evidencia (alta=blanco lateral claro; ~0=simétrico).
        claridad_estimulo = self._clamp(abs(evidencia_l - evidencia_r) / (evidencia_l + evidencia_r + 1e-6), 0.0, 1.0)
        # persistencia de la decisión = estabilidad de SIGNO de decision_rc (1=signo estable; 0=errático/cancela).
        self._dec_ema = (1.0 - self.ema_dec) * self._dec_ema + self.ema_dec * decision_rc
        self._absdec_ema = (1.0 - self.ema_dec) * self._absdec_ema + self.ema_dec * abs(decision_rc)
        persistencia_decision = self._clamp(abs(self._dec_ema) / (self._absdec_ema + 1e-6), 0.0, 1.0)
        # tasa motora EFECTIVA = máxima fisiológica gateada por confianza/permiso/no-bloqueo/no-fatiga/persistencia.
        # PULSIÓN (Cable B): el HAMBRE energiza el movimiento (SEEKING) — sube la GANANCIA, NO la dirección
        # (el objetivo lo fija RC). Saciado → se aquieta. No es premio ni setpoint: sólo inquietud por déficit.
        hambre = self._clamp(fila.get("met_hambre", 0.0), 0.0, 1.0)
        pulsion_hambre = 1.0 + self.K_HAMBRE * hambre
        k_motor_eff = self.k_motor * media([confianza, permiso_decisional, 1.0 - bloqueo_irde,
                                            1.0 - fatiga_norm, persistencia_decision]) * pulsion_hambre

        # Cable C — APRENDER la orientación nutritiva (sesgo por MEMORIA, NO por reward ni target_A):
        # consolida valor en la orientación ACTUAL ∝ lo que se COMIÓ ahí (met_nutricion, que SÓLO es >0 si
        # hubo acople real → regla 3), y decae (extinción → regla 7 si el alimento se mueve). Luego SESGA
        # el centro del escaneo hacia la orientación mejor recordada. El organismo va a la comida porque la
        # RECUERDA, no porque persiga A: convergencia por APRENDIZAJE, no gradient-ascent.
        nutricion = self._clamp(fila.get("met_nutricion", 0.0), 0.0, 1.0)
        if self.forrajeo_c:
            # === Cable C (EXPERIMENTAL, gateado — NO converge aún; ver bateria_cable_c) ===
            # SACCADE-AND-FIXATE + APRENDER orientación nutritiva (∝ met_nutricion real, regla 3) + SESGAR
            # el muestreo hacia lo recordado + el HAMBRE domina sobre RC. Sin reward, sin target_A, sin
            # esteerar por A. Diagnóstico del no-converge: (1) RC apunta en contra del lado que acopla;
            # (2) inercia del campo: en movimiento la señal orientación↔nutrición se invierte vs estático;
            # (3) metabolismo demasiado disipativo (E→0 siempre) → no hay saciedad que aquiete la búsqueda.
            self._dwell += 1
            asentado = self._dwell > (self.DWELL // 2)
            if self._dwell >= self.DWELL:
                self._dwell = 0
                self._scan_phase += self.SCAN_STEP
            for b in list(self.valor_orient):
                self.valor_orient[b] *= self.DECAY_ORIENT
            if asentado:
                b_now = round(self.theta / 15.0) * 15.0
                vb = self.valor_orient.get(b_now, 0.0)
                self.valor_orient[b_now] = vb + self.ETA_ORIENT * (nutricion - vb)
            centro_pref = 0.0
            if self.valor_orient:
                b_best = max(self.valor_orient, key=self.valor_orient.get)
                conf_pref = self._clamp(self.valor_orient[b_best] / self.CONF_REF, 0.0, 1.0)
                centro_pref = self.K_BIAS * conf_pref * b_best
            # hambre domina sobre RC (regla 2: RC manda saciado). Barrido amplio + sesgo por memoria.
            # NOTA: aprende el lado correcto (IM suavizado) pero AÚN no converge — explotar fuerte lo bloquea
            # en el primer lado visitado (causa 1: RC apunta en contra). Pendiente: reconciliar el compás RC.
            scan_offset = 90.0 * float(np.sin(self._scan_phase))
            objetivo = self._clamp((1.0 - hambre) * objetivo + hambre * (centro_pref + scan_offset), -90.0, 90.0)
        else:
            # Cable B (VALIDADO): el HAMBRE energiza un barrido UNDIRECTED alrededor de la dirección RC.
            self._scan_phase += self.SCAN_VEL
            objetivo = self._clamp(objetivo + self.K_EXPLOR * hambre * float(np.sin(self._scan_phase)), -90.0, 90.0)
        self._theta_prev = self.theta

        error = objetivo - self.theta
        if error > 180.0:
            error -= 360.0
        elif error < -180.0:
            error += 360.0

        zona = min(18.0, self.zona_base + self.zona_por_fatiga * self.fatiga + 4.0 * bloqueo_irde + 3.0 * conflicto_rc)
        if abs(error) < zona:
            delta_raw = 0.0
        else:
            freno = 1.0 - np.exp(-abs(error) / 30.0)
            factor_fatiga = max(0.20, np.exp(-0.004 * self.fatiga) * (1.0 - 0.35 * bloqueo_irde))
            delta_raw = k_motor_eff * error * freno * factor_fatiga   # tasa motora ADAPTATIVA (confianza ya va dentro)

        delta = self.inercia * self.ultimo_delta + (1.0 - self.inercia) * delta_raw
        temblor = float(np.random.normal(0.0, self.temblor_base * (1.0 + min(2.0, self.fatiga / 120.0) + 1.6 * bloqueo_irde + 0.8 * conflicto_rc)))
        delta_total = delta + temblor * 0.005

        self.theta = self._clamp(self.theta + delta_total, -90.0, 90.0)
        self.ultimo_delta = delta
        self.fatiga = self.fatiga * self.fatiga_decay + abs(delta_total) * (1.0 + 0.9 * bloqueo_irde + 0.4 * conflicto_rc)

        # --- CIERRE: ¿el movimiento AYUDÓ al organismo? → adapta la comprensión (anti-Shannon).
        #     SUBE sólo si: decisión persistente ∧ estímulo claro ∧ el movimiento redujo el error ∧
        #     el acoplamiento NO cayó ∧ la desviación (IRDE) NO aumentó.  BAJA en caso contrario.
        if self._Aenv_prev is None:
            self._Aenv_prev = Aenv; self._irde_prev = irde_ratio
        d_aenv = Aenv - self._Aenv_prev           # ¿el acoplamiento NO empeoró este paso? (cierre)
        d_irde = irde_ratio - self._irde_prev     # ¿la desviación (IRDE) NO aumentó?
        d_conf = conflicto_rc - self._conf_prev
        error_motor = abs(error)
        mejora_motor = self._error_prev - error_motor               # >0 si el movimiento redujo el error
        # Bootstrap anti-deadlock SIN Shannon: la ganancia sube ante un blanco PERSISTENTE y CLARO cuya
        # atención NO daña el cierre (acoplamiento estable ∧ IRDE no sube) — aunque aún no haya movimiento
        # (el movimiento llega cuando la comprensión crece lo suficiente). El movimiento que EMPEORA el
        # error o el cierre entra por el CASTIGO, no como requisito de subida. Anti-Shannon: no se sube
        # "para girar", se sube porque hay algo claro y persistente que comprender sin dañarse.
        gate_cierre = 1.0 if (d_aenv >= -1e-4 and d_irde <= 1e-4) else 0.0
        recompensa = persistencia_decision * claridad_estimulo * icr_ratio * gate_cierre
        castigo = ((1.0 - persistencia_decision)
                   + self.sens_irde * max(0.0, d_irde)
                   + self.sens_aenv * max(0.0, -d_aenv)
                   + max(0.0, d_conf)
                   + (0.5 if mejora_motor < 0.0 else 0.0))
        adaptacion_comprension = self.tau_gain * (recompensa - castigo)
        self.comp_gain_eff = self._clamp(self.comp_gain_eff + adaptacion_comprension, self.GAIN_MIN, self.GAIN_MAX)
        adaptacion_motor = k_motor_eff - self.k_motor               # cuánto se gateó la tasa vs su máximo fisiológico
        self._Aenv_prev = Aenv; self._irde_prev = irde_ratio
        self._conf_prev = conflicto_rc; self._error_prev = error_motor

        self.temblores.append(temblor)
        if len(self.temblores) > 120:
            self.temblores.pop(0)
        trem_rms = float(np.sqrt(np.mean(np.asarray(self.temblores, dtype=np.float64) ** 2))) if self.temblores else 0.0

        return {
            "act_orientacion_deg": round(self.theta, 3),
            "act_objetivo_deg": round(objetivo, 3),
            "act_delta_deg": round(delta_total, 5),
            "act_confianza": round(confianza, 4),
            "act_fatiga": round(self.fatiga, 4),
            "act_zona_muerta": round(zona, 3),
            "act_temblor_rms": round(trem_rms, 4),
            "act_lateralidad_dw": round(dw, 5),
            "act_atencion_L": round(at_l, 5),
            "act_atencion_R": round(at_r, 5),
            "act_comprension_L": round(float(fila.get("RC_comprension_L", 0.0)), 5),
            "act_comprension_R": round(float(fila.get("RC_comprension_R", 0.0)), 5),
            "act_riesgo_L": round(float(fila.get("RC_riesgo_L", 0.0)), 5),
            "act_riesgo_R": round(float(fila.get("RC_riesgo_R", 0.0)), 5),
            "act_consenso_RC": round(decision_organismica, 5),
            "act_conflicto_RC": round(conflicto_rc, 5),
            "act_freno_RC": round(freno_rc, 5),
            "act_rc_mix": round(permiso_decisional, 5),
            "act_presencia_L": round(pL, 5),
            "act_presencia_R": round(pR, 5),
            "act_propuesta_atencional": round(propuesta_atencional, 5),
            "act_decision_RC": round(decision_rc, 5),
            "act_bloqueo_IRDE": round(bloqueo_irde, 5),
            "act_permiso_decisional": round(permiso_decisional, 5),
            "act_evidencia_L": round(evidencia_l, 5),
            "act_evidencia_R": round(evidencia_r, 5),
            "act_razon_L": round(razon_l, 5),
            "act_razon_R": round(razon_r, 5),
            "act_necesidad_cierre": round(necesidad_cierre, 5),
            "act_decision_organismica": round(decision_organismica, 5),
            "act_soporte_sentido": round(soporte_sentido, 5),
            "act_vulnerabilidad_riesgo": round(vulnerabilidad_riesgo, 5),
            "act_base_sentido": round(base_sentido, 5),
            "act_base_riesgo": round(base_riesgo, 5),
            "act_peso_sentido": round(peso_sentido, 5),
            "act_peso_riesgo": round(peso_riesgo, 5),
            # --- adaptación organísmica (anti-Shannon): ganancia/tasa EMERGENTES + sus señales de cierre ---
            "act_comp_gain_eff": round(self.comp_gain_eff, 5),
            "act_k_motor_eff": round(k_motor_eff, 6),
            "act_persistencia_decision": round(persistencia_decision, 5),
            "act_claridad_estimulo": round(claridad_estimulo, 5),
            "act_error_motor": round(error_motor, 4),
            "act_mejora_motor": round(mejora_motor, 5),
            "act_adaptacion_motor": round(adaptacion_motor, 6),
            "act_adaptacion_comprension": round(adaptacion_comprension, 6),
        }


def _fila(cel, actuador=None) -> dict:
    """Una fila (dict col->valor) leída del milieu + salud, para CSV y streaming."""
    m = cel.milieu; s = cel.salud()
    g = lambda k, d=0.0: m.leer(k, d)
    inv = sum(1 for v in s["invariantes"].values() if v)
    d = {
        "t": round(cel.t, 3), "Omega": round(g("Omega"), 4), "omega_A": round(g("omega_A"), 4),
        "omega_B": round(g("omega_B"), 4), "gradiente": round(g("gradiente"), 4),
        "e_R": round(g("e_R"), 4), "A_sys_env": round(g("A_sys_env"), 4),
        "presion_desacople": round(g("presion_desacople"), 3), "C_b": int(g("C_b", 0)),
        "R2": round(g("R2"), 4), "LF_op": round(g("LF_op"), 4), "lf_nivel": int(g("lf_nivel", 0)),
        "juego": int(bool(g("juego_activo", False))), "ritual": int(bool(g("ritual_activo", False))),
        "negacion": int(bool(g("negacion_activa", False))), "demanda_entorno": round(g("demanda_entorno", 1.0), 4),
        "Omega_op": round(g("Omega_op", 1.0), 4), "XE": round(min(1.0, g("XE")), 4),
        "C_m": round(g("C_m"), 4), "H_homeostasis": round(g("H_homeostasis"), 4),
        "OI": round(s["OI"], 4), "Lambda_Cos": round(s["Lambda_Cos"], 5),
        # binaurales
        "omega_L": round(g("omega_L"), 4), "omega_R": round(g("omega_R"), 4),
        "omega_A_L": round(g("omega_A_L"), 4), "omega_A_R": round(g("omega_A_R"), 4),
        "energia_L": round(g("energia_L"), 5), "energia_R": round(g("energia_R"), 5),
        "balance_LR": round(g("balance_LR"), 4), "lateralidad": round(g("lateralidad"), 4),
        "coherencia_biaural": round(g("coherencia_biaural"), 4),
        # observación
        "LF_struct": round(g("LF_struct"), 4), "self_coherencia": round(g("self_coherencia"), 4),
        "estructura": round(g("estructura"), 4),
        "estructura_L": round(g("estructura_L"), 4), "estructura_R": round(g("estructura_R"), 4),
        "x_interna": round(g("x_interna", 0.5), 4), "en_rango": int(bool(g("x_interna_en_rango", False))),
        "mutacion": round(g("mutacion"), 5), "adaptacion_activa": int(bool(g("adaptacion_activa", False))),
        "exaptacion_activa": int(bool(g("exaptacion_activa", False))),
        "activacion_latente": int(bool(g("demanda_activacion", False))), "invariantes_ok": inv,
    }
    _rc_observar(d)
    # ── HOMEOSTASIS EMERGENTE (canónica, fila-level) ──────────────────────────────────────────
    # O-N9.14: H = A_sys-env en rango estable, SOSTENIDO por la competencia ICR↔IRDE (literal, de
    # RC). Cierra el circuito A_sys-env→H_real→OI. NO toca la orientación ni la cabeza 3D: sólo
    # añade columnas (H_*, A_soporte_*) y realimenta el OI con la H canónica. H_homeostasis (proxy
    # del organelo del genoma) se conserva como diagnóstico; H_homeostasis_real es la canónica.
    d.update(HOMEO_EMERGENTE.actualizar(d))
    d.update(soporte_A_sys_env(d, dA_sys_env=d.get("H_dA_sys_env", 0.0)))
    d.update(permeabilidad_activa(d))                         # act_perm DEFINIDA por el circuito (latente)
    _soma_p = cel.organelos.get("soma")                       # Cable B2: act_perm afila el acople el próximo
    if _soma_p is not None: _soma_p.perm_ext = float(d.get("act_perm", 0.0))   # paso (apertura de membrana, lag 1)
    # ── METABOLISMO (economía energética): come la experiencia, paga el costo, se degrada y repone.
    # Corre ANTES de la memoria para que su met_nutricion pueda SACIAR la necesidad (cierre del lazo). ──
    d.update(METABOLISMO.actualizar(d, dt=DT))
    # ── MEMORIA (historia interna): 6 capas + necesidad (Cb→act_perm). Lee soma (capa 5 implícita),
    # milieu (presion/fatiga/vida) y la fila (RC/act_perm/H_real). NO cierra el lazo conductual. ──
    d.update(MEMORIA.actualizar(d, dt=DT, milieu=m, soma=cel.organelos.get("soma")))
    m.secretar("H_homeostasis", d["H_homeostasis_real"])      # el OI usa H_real (canónica) si existe
    s2 = cel.salud()
    d["OI"] = round(s2["OI"], 4); d["Lambda_Cos"] = round(s2["Lambda_Cos"], 5)
    if actuador is not None:
        d.update(actuador.actualizar(d))
        _soma = cel.organelos.get("soma")            # Cable A: la orientación de la cabeza re-pondera el
        if _soma is not None:                          # acople del soma el PRÓXIMO paso (propiocepción, lag 1)
            _soma.orient_ext = float(d.get("act_orientacion_deg", 0.0))
    else:
        d.update({k: 0.0 for k in COLS_ACT})
    return d


# ==============================================================================
# CARGA DE AUDIO (mono, dos canales, demo, upload, biauralización, en vivo)
# ==============================================================================
def _mono_de_spec(spec: str):
    """Devuelve (nombre, vector mono 48kHz) desde 'demo:...' o un .wav (de audio_binaural)."""
    if spec.startswith("demo:"):
        return cmf.cargar_audio(spec, binaural=False)
    path = spec if os.path.isabs(spec) else os.path.join(AUDIO_DIR, spec)
    return os.path.splitext(os.path.basename(path))[0], cmf._load_wav(path, binaural=False)

def _mono_de_upload(b64: str, etiqueta: str):
    raw = b64.split(",", 1)[1] if "," in b64 else b64
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tmp.write(base64.b64decode(raw)); tmp.close()
    try:
        return etiqueta, cmf._load_wav(tmp.name, binaural=False)
    finally:
        os.unlink(tmp.name)

def _carga_upload_binaural(b64: str, etiqueta: str):
    """Carga un upload usando sus PROPIOS canales L/R (binaural real; duplica si es mono)."""
    raw = b64.split(",", 1)[1] if "," in b64 else b64
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tmp.write(base64.b64decode(raw)); tmp.close()
    try:
        _, audio = cmf.cargar_audio(tmp.name, binaural=True)
        return etiqueta, audio
    finally:
        os.unlink(tmp.name)

def biauralizar_mono(mono, delay_ms=0.3, gain_L=1.0, gain_R=0.95):
    """SIMULACIÓN biaural (NO real): de un mono genera L/R con microdiferencia interaural
    (delay + ganancia). Marcado como simulación. Valores prudentes por defecto."""
    d = int(round(delay_ms * 1e-3 * SR))
    izq = mono * float(gain_L)
    der = (np.concatenate([np.zeros(d), mono[:len(mono) - d]]) if d > 0 else mono.copy()) * float(gain_R)
    return izq, der

def _alinear(a, b, criterio="min"):
    if len(a) == len(b):
        return a, b, "igual_duracion"
    if criterio == "pad":
        n = max(len(a), len(b))
        return (np.concatenate([a, np.zeros(n - len(a))]),
                np.concatenate([b, np.zeros(n - len(b))]), "relleno_silencio_a_max")
    n = min(len(a), len(b))
    return a[:n], b[:n], "truncado_a_duracion_menor"


# ==============================================================================
# DOS FUENTES INDEPENDIENTES por oído (Req: selector L / selector R)
# Cada oído resuelve a un vector mono desde: archivo, demo, upload, o UN CANAL de un
# dispositivo de entrada (p.ej. un canal individual de la Rødecaster Pro II 16ch).
# ==============================================================================
def _extraer_canal(rec, ci: int):
    """Extrae SOLO el canal `ci` de un buffer (N x canales). NO mezcla los canales.
    Testeable sin hardware: dado un buffer multicanal, devuelve la columna pedida."""
    rec = np.asarray(rec, dtype=np.float64)
    if rec.ndim == 1:
        return rec
    return rec[:, ci]

def _grabar_dispositivo(device_index: int, channels: int, seg: float):
    """Graba `seg` s de un dispositivo (N x ch) y lo remuestrea a 48 kHz si hace falta.
    Robustez: usa el sample-rate NATIVO del dispositivo (no fuerza 48k → evita fallos de la
    Rødecaster), recorta `channels` al máximo real, y detecta SILENCIO TOTAL (señal típica de
    falta de permiso de Micrófono en macOS) con un mensaje claro."""
    if not SD_OK:
        raise RuntimeError("Audio en vivo NO disponible (sounddevice no instalado: "
                           f"{SD_ERR}). Instala: venv/bin/pip install sounddevice (brew install portaudio).")
    info = _sd.query_devices(device_index)
    maxin = int(info.get("max_input_channels", 0)); nom = info.get("name", f"dev{device_index}")
    if maxin <= 0:
        raise RuntimeError(f"el dispositivo {device_index} '{nom}' no tiene canales de entrada.")
    nch = min(maxin, max(1, channels))                          # no pedir más canales de los que hay
    dev_sr = int(round(info.get("default_samplerate") or SR)) or SR
    try:
        rec = _sd.rec(int(seg * dev_sr), samplerate=dev_sr, channels=nch, device=device_index)
        _sd.wait()
    except Exception as e:
        raise RuntimeError(
            f"no se pudo capturar de '{nom}' (sr={dev_sr}, canales={nch}): {e}. "
            "Posibles causas: (1) PERMISO de Micrófono de macOS — Ajustes del Sistema → Privacidad y "
            "seguridad → Micrófono → habilita Terminal (o tu app)/Python; (2) la Rødecaster no está a "
            f"{dev_sr} Hz o no expone {nch} canales en 'Configuración de Audio MIDI'; (3) otro programa "
            "tiene el dispositivo tomado.")
    rec = np.asarray(rec, dtype=np.float64)
    if rec.size and float(np.max(np.abs(rec))) < 1e-7:
        raise RuntimeError(
            f"se capturó SILENCIO TOTAL de '{nom}'. Lo más probable: FALTA EL PERMISO DE MICRÓFONO de "
            "macOS para el proceso (Ajustes del Sistema → Privacidad y seguridad → Micrófono → habilita "
            "Terminal/Python y reinicia el laboratorio). También verifica que el canal elegido reciba señal.")
    if dev_sr != SR:                                            # remuestrear cada canal a 48 kHz
        m = int(rec.shape[0] * SR / dev_sr); base = np.arange(rec.shape[0]); idx = np.linspace(0, rec.shape[0], m, endpoint=False)
        rec = np.stack([np.interp(idx, base, rec[:, c]) for c in range(rec.shape[1])], axis=1)
    return rec


class _LectorServidor:
    """Consume el stream TCP de VST_AudioServer.py y entrega bloques de `seg` segundos de DOS
    canales elegidos (oído L = canal iL, oído R = canal iR).

    Quién soy: el ADAPTADOR DE IMPEDANCIA del lado del laboratorio (idéntico en espíritu al de
    VST_AudioCliente.py). El servidor entrega TODOS los canales en frames pequeños (~1024 samples);
    aquí elijo MIS dos canales (la elección de canal es del consumidor, no del servidor) y los
    acumulo hasta juntar el bloque que la célula consume. Mantiene la conexión abierta entre
    lecturas si quien me usa me reutiliza (vida continua)."""

    def __init__(self, host: str, port: int, iL: int, iR: int) -> None:
        if not SERV_OK:
            raise RuntimeError(f"puente TCP no disponible (no pude importar AudioStreamClient: {SERV_ERR}).")
        self.iL, self.iR = int(iL), int(iR)
        try:
            self.cli = AudioStreamClient(host=host, port=port, timeout=5.0)
        except OSError as e:
            raise RuntimeError(f"no hay servidor de audio en {host}:{port} ({e}). "
                               "¿Está corriendo VST_AudioServer.py en el Mac?")
        self.hs = self.cli.handshake()
        self.nch = int(self.hs.get("channels", 0))
        self.sr = int(self.hs.get("sample_rate", SR))
        self._gen = self.cli.frames()
        self._bufL = np.zeros(0, dtype=np.float64)
        self._bufR = np.zeros(0, dtype=np.float64)
        if not (0 <= self.iL < self.nch and 0 <= self.iR < self.nch):
            self.cerrar()
            raise RuntimeError(f"el servidor entrega {self.nch} canales; pediste L={self.iL + 1}/R={self.iR + 1}. "
                               f"Usa canales 1..{self.nch}.")

    def leer_bloque(self, seg: float):
        """Devuelve (L, R) con seg·SR samples cada uno, esperando a que lleguen del stream en vivo.
        NOTA: no remuestrea; el servidor (y la Rødecaster) trabajan a 48 kHz = SR de la célula."""
        need = max(1, int(round(seg * SR)))
        while len(self._bufL) < need:
            try:
                blk = next(self._gen)                       # (frames, nch) float32, tamaño variable
            except StopIteration:
                raise RuntimeError("el servidor de audio cerró la conexión (¿se detuvo VST_AudioServer.py?).")
            self._bufL = np.concatenate([self._bufL, blk[:, self.iL].astype(np.float64)])
            self._bufR = np.concatenate([self._bufR, blk[:, self.iR].astype(np.float64)])
        L = self._bufL[:need]; R = self._bufR[:need]
        self._bufL = self._bufL[need:]; self._bufR = self._bufR[need:]
        return L, R

    def set_canales(self, iL, iR):
        """Cambia EN VIVO qué canales son L/R (el servidor ya entrega todos). Limpia los buffers
        para no mezclar muestras del canal viejo con el nuevo. El campo Φ de la célula NO se toca."""
        self.iL, self.iR = int(iL), int(iR)
        self._bufL = np.zeros(0, dtype=np.float64); self._bufR = np.zeros(0, dtype=np.float64)

    def cerrar(self):
        try:
            self.cli.cerrar()
        except Exception:
            pass


_COM_AUDIO_CACHE = {}
_COM_AUDIO_METER = {"nombre": "organismo", "rms": 0.0, "pico": 0.0, "ok": False,
                    "reserva": False, "updated": 0.0, "canales": {}}


def _rms_pico_audio(audio: np.ndarray) -> tuple[float, float]:
    x = np.asarray(audio, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    pico = float(np.max(np.abs(x))) if x.size else 0.0
    return rms, pico


def _actualizar_medidor_comunicacion(audio: np.ndarray, nombre: str, reserva: bool = False) -> None:
    rms, pico = _rms_pico_audio(audio)
    _COM_AUDIO_METER.update({"nombre": nombre or "organismo", "rms": rms, "pico": pico,
                             "ok": True, "reserva": bool(reserva), "updated": time.time(),
                             "canales": {"?": {"rms": rms, "pico": pico, "nombre": nombre or "organismo"}}})


def _actualizar_medidor_comunicacion_canales(canales, reserva: bool = False) -> None:
    datos = {}
    nombres = []
    for lado, nombre, audio in canales:
        rms, pico = _rms_pico_audio(audio)
        datos[str(lado)] = {"rms": rms, "pico": pico, "nombre": nombre or "organismo"}
        nombres.append(nombre or f"organismo {lado}")
    rms = max([v["rms"] for v in datos.values()] or [0.0])
    pico = max([v["pico"] for v in datos.values()] or [0.0])
    _COM_AUDIO_METER.update({"nombre": " + ".join(nombres) if nombres else "organismo",
                             "rms": rms, "pico": pico, "ok": True,
                             "reserva": bool(reserva), "updated": time.time(), "canales": datos})


def _master_input_level() -> dict:
    """Nivel MAESTRO = lo que el organismo OYE AHORA (RMS y pico del bloque de entrada que procesa),
    sea cual sea la fuente: Rødecaster, wav (Big Bang/Blue Monday), demo o silencio. Responde de un
    vistazo '¿está sonando?' incluso cuando NO hay canales de servidor (p.ej. un wav)."""
    r = RUN
    soma = getattr(r, "soma", None) if r is not None else None
    if soma is None:
        return {"rms": 0.0, "pico": 0.0, "ok": False}
    try:
        L = np.asarray(getattr(soma, "_L", []), dtype=np.float64)
        R = getattr(soma, "_R", None)
        R = L if R is None else np.asarray(R, dtype=np.float64)
        if L.size == 0:
            return {"rms": 0.0, "pico": 0.0, "ok": False}
        n = min(L.size, R.size)
        rms = float(np.sqrt(np.mean((L[:n] ** 2 + R[:n] ** 2) / 2.0)))
        pico = float(max(np.max(np.abs(L)) if L.size else 0.0, np.max(np.abs(R)) if R.size else 0.0))
        return {"rms": round(rms, 5), "pico": round(pico, 5), "ok": True}
    except Exception:
        return {"rms": 0.0, "pico": 0.0, "ok": False}


def _snapshot_comunicacion_entrante() -> dict:
    updated = float(_COM_AUDIO_METER.get("updated") or 0.0)
    age = None if updated <= 0 else max(0.0, time.time() - updated)
    canales = dict(_COM_AUDIO_METER.get("canales") or {})
    mute_l = bool(globals().get("RUN") and getattr(globals().get("RUN"), "mute_L", False))
    mute_r = bool(globals().get("RUN") and getattr(globals().get("RUN"), "mute_R", False))
    rms_vals = []
    pico_vals = []
    for lado, dato in canales.items():
        if lado == "L" and mute_l:
            continue
        if lado == "R" and mute_r:
            continue
        rms_vals.append(float(dato.get("rms") or 0.0))
        pico_vals.append(float(dato.get("pico") or 0.0))
    if not canales:
        rms_vals = [float(_COM_AUDIO_METER.get("rms") or 0.0)]
        pico_vals = [float(_COM_AUDIO_METER.get("pico") or 0.0)]
    rms = max(rms_vals or [0.0])
    pico = max(pico_vals or [0.0])
    canales_out = {}
    for lado, dato in canales.items():
        crms = float(dato.get("rms") or 0.0)
        cpico = float(dato.get("pico") or 0.0)
        if lado == "L" and mute_l:
            crms = cpico = 0.0
        if lado == "R" and mute_r:
            crms = cpico = 0.0
        canales_out[lado] = {"rms": crms, "pico": cpico, "nombre": dato.get("nombre", "organismo")}
    if age is not None and age > 1.0:
        decay = max(0.0, 1.0 - min(1.0, (age - 1.0) / 4.0))
        rms *= decay
        pico *= decay
        for dato in canales_out.values():
            dato["rms"] *= decay
            dato["pico"] *= decay
    return {"ok": bool(_COM_AUDIO_METER.get("ok")), "nombre": _COM_AUDIO_METER.get("nombre", "organismo"),
            "rms": round(rms, 6), "pico": round(pico, 6),
            "age_s": None if age is None else round(age, 3),
            "reserva": bool(_COM_AUDIO_METER.get("reserva")),
            "mute_L": mute_l, "mute_R": mute_r,
            "canales": {k: {"rms": round(v["rms"], 6), "pico": round(v["pico"], 6), "nombre": v["nombre"]}
                         for k, v in canales_out.items()}}


def _audio_comunicacion_resiliente(url: str, seg: float, nombre: str) -> np.ndarray:
    n = max(1, int(round(float(seg) * SR)))
    key = str(url or COMUNICACION_PEER_URL)
    timeout = max(4.0, min(18.0, float(seg) + 2.0))
    try:
        audio = np.asarray(audio_desde_url(key, seg=seg, sr=SR, timeout=timeout), dtype=np.float64)
        if audio.size:
            if audio.size < n:
                audio = np.pad(audio, (0, n - audio.size))
            elif audio.size > n:
                audio = audio[:n]
            _COM_AUDIO_CACHE[key] = audio.copy()
            _actualizar_medidor_comunicacion(audio, nombre, reserva=False)
            return audio
    except Exception as e:
        print(f"[comunicacion] bloque no disponible para {nombre}: {e}; usando reserva.")

    previo = _COM_AUDIO_CACHE.get(key)
    if previo is not None and previo.size:
        if previo.size >= n:
            y = previo[:n].copy()
        else:
            rep = int(np.ceil(n / previo.size))
            y = np.tile(previo, rep)[:n].astype(np.float64)
        _actualizar_medidor_comunicacion(y, nombre, reserva=True)
        return y
    y = np.zeros(n, dtype=np.float64)
    _actualizar_medidor_comunicacion(y, nombre, reserva=True)
    return y


def _resolver_fuente(src: dict, seg: float = 10.0):
    if (src or {}).get("tipo") == "comunicacion":
        if not COM_OK:
            raise RuntimeError(f"organo de comunicacion no disponible: {COM_ERR}")
        url = src.get("url") or COMUNICACION_PEER_URL
        nombre = src.get("nombre") or "organo de comunicacion"
        return (nombre, _audio_comunicacion_resiliente(url, seg, nombre))
    """Resuelve UN descriptor de fuente → (nombre, vector mono 48kHz).
    src.tipo ∈ {archivo, demo, upload, dispositivo}. Para 'dispositivo' graba ese device
    y extrae SOLO src.channel_index (no mezcla canales)."""
    t = (src or {}).get("tipo")
    if t == "archivo":
        return _mono_de_spec(src["nombre"])
    if t == "demo":
        return cmf.cargar_audio(src.get("spec", "demo:tono"), binaural=False)
    if t == "upload":
        return _mono_de_upload(src["b64"], src.get("name", "subido"))
    if t == "dispositivo":
        di = int(src["device_index"]); ci = int(src["channel_index"])
        rec = _grabar_dispositivo(di, ci + 1, seg)   # graba hasta el canal pedido; extrae esa columna
        nom = (_sd.query_devices(di).get("name", f"dev{di}") if SD_OK else f"dev{di}")
        return (f"{nom} — canal {ci + 1}", _extraer_canal(rec, ci))
    if t == "servidor":                                # UN canal traído por TCP del servidor de audio
        host = src.get("host", SERVIDOR_HOST); port = int(src.get("port", SERVIDOR_PORT))
        ci = int(src["channel_index"])
        lec = _LectorServidor(host, port, ci, ci)      # mismo canal en L y R; tomamos solo L (mono)
        try:
            L, _ = lec.leer_bloque(seg)
        finally:
            lec.cerrar()
        return (src.get("nombre_canal") or _nombre_canal(lec.hs.get("device", "servidor"), ci), L)
    raise ValueError(f"tipo de fuente desconocido: {t!r}")

def _build_audio_por_oido(cfg: dict):
    """Modelo NUEVO: left_src / right_src son descriptores independientes por oído.
    Caso especial: si ambos son canales del MISMO dispositivo, se graba UN solo stream
    sincronizado y se extraen las dos columnas (no dos grabaciones desfasadas)."""
    seg = float(cfg.get("segundos", 10)); crit = cfg.get("criterio_duracion", "min")
    ls, rs = cfg.get("left_src"), cfg.get("right_src")
    meta = {"fuente": "por_oido", "criterio_duracion": "-", "simulacion_biaural": False}

    # ambos canales del MISMO servidor TCP → UN stream sincronizado, dos columnas (lateralidad real).
    # Análogo a "mismo dispositivo, dos canales", pero el dato viaja por red desde el servidor que sí
    # ve la Rødecaster. La elección de qué canal es L y cuál R es del consumidor (esta Web).
    if (ls and rs and ls.get("tipo") == "servidor" and rs.get("tipo") == "servidor"
            and ls.get("host", SERVIDOR_HOST) == rs.get("host", SERVIDOR_HOST)
            and int(ls.get("port", SERVIDOR_PORT)) == int(rs.get("port", SERVIDOR_PORT))):
        host = ls.get("host", SERVIDOR_HOST); port = int(ls.get("port", SERVIDOR_PORT))
        cL = int(ls["channel_index"]); cR = int(rs["channel_index"])
        lec = _LectorServidor(host, port, cL, cR)
        try:
            L, R = lec.leer_bloque(seg)
        finally:
            lec.cerrar()
        dev = lec.hs.get("device", "servidor")
        meta.update(izquierdo=_nombre_canal(dev, cL), derecho=_nombre_canal(dev, cR),
                    fuente=f"servidor TCP {host}:{port}")
        return (L, R), True, meta

    # mismo dispositivo, dos canales → un stream sincronizado, dos columnas
    if (ls and rs and ls.get("tipo") == "dispositivo" and rs.get("tipo") == "dispositivo"
            and ls.get("device_index") == rs.get("device_index")):
        di = int(ls["device_index"]); cL = int(ls["channel_index"]); cR = int(rs["channel_index"])
        rec = _grabar_dispositivo(di, max(cL + 1, cR + 1), seg)   # graba hasta el canal mayor
        nom = (_sd.query_devices(di).get("name", f"dev{di}") if SD_OK else f"dev{di}")
        meta.update(izquierdo=f"{nom} — canal {cL+1}", derecho=f"{nom} — canal {cR+1}",
                    fuente="dispositivo (1 stream, 2 canales)")
        return (_extraer_canal(rec, cL), _extraer_canal(rec, cR)), True, meta

    if ls and rs:                                    # dos fuentes independientes
        nL, L = _resolver_fuente(ls, seg); nR, R = _resolver_fuente(rs, seg)
        L, R, c = _alinear(L, R, crit)
        canales_com = []
        if (ls or {}).get("tipo") == "comunicacion":
            canales_com.append(("L", nL, L))
        if (rs or {}).get("tipo") == "comunicacion":
            canales_com.append(("R", nR, R))
        if canales_com:
            _actualizar_medidor_comunicacion_canales(canales_com)
        meta.update(izquierdo=nL, derecho=nR, criterio_duracion=c); return (L, R), True, meta

    src = ls or rs                                   # una sola fuente
    nom, mono = _resolver_fuente(src, seg)
    if (src or {}).get("tipo") == "comunicacion":
        _actualizar_medidor_comunicacion_canales([("L", nom, mono), ("R", nom, mono)] if cfg.get("binaural") else [("?", nom, mono)])
    if cfg.get("binaural"):                          # duplicar a L/R (sin lateralidad real)
        meta.update(izquierdo=nom, derecho=nom + " (duplicado)"); return (mono, mono.copy()), True, meta
    meta.update(izquierdo=nom, derecho=nom); return mono, False, meta


def build_audio(cfg: dict):
    """Resuelve la configuración de entrada → (audio, binaural, meta). audio = vector mono
    o (izq, der). meta documenta fuente, canales, criterio de duración y si la lateralidad es real."""
    # MODELO NUEVO (dos selectores independientes): si hay left_src/right_src, úsalo.
    if cfg.get("left_src") or cfg.get("right_src"):
        return _build_audio_por_oido(cfg)
    # ---- MODELO PREVIO (preservado intacto): fuente/left/right ----
    bia = cfg.get("biauralizar") or {}
    fuente = cfg.get("fuente", "demo")
    crit_dur = cfg.get("criterio_duracion", "min")
    meta = {"fuente": fuente, "criterio_duracion": "-", "simulacion_biaural": False}

    # ---------- AUDIO EN VIVO (grabar N s y procesar; REAL si hay dispositivo) ----------
    if fuente == "vivo":
        if not SD_OK:
            raise RuntimeError("Audio en vivo NO disponible (sounddevice no instalado: "
                               f"{SD_ERR}). Instala:  venv/bin/pip install sounddevice  "
                               "(requiere PortAudio: brew install portaudio). Para AUDIO DEL SISTEMA "
                               "en macOS necesitas BlackHole (brew install blackhole-2ch) y elegirlo "
                               "como dispositivo de entrada.")
        v = cfg.get("vivo", {}); seg = float(v.get("segundos", 10)); dev = v.get("device")
        rec = np.asarray(_sd.rec(int(seg * SR), samplerate=SR, channels=2, device=dev), dtype=np.float64)
        _sd.wait()
        if rec.ndim == 2 and rec.shape[1] >= 2 and not np.array_equal(rec[:, 0], rec[:, 1]):
            meta.update(izquierdo="vivo:canal1", derecho="vivo:canal2"); return (rec[:, 0], rec[:, 1]), True, meta
        mono = rec.reshape(-1) if rec.ndim == 1 else rec[:, 0]
        if bia.get("on"):
            izq, der = biauralizar_mono(mono, bia.get("delay_ms", 0.3), bia.get("gain_L", 1.0), bia.get("gain_R", 0.95))
            meta.update(izquierdo="vivo (simulado L)", derecho="vivo (simulado R)", simulacion_biaural=True)
            return (izq, der), True, meta
        meta.update(izquierdo="vivo (mono)", derecho="vivo (mono)"); return mono, False, meta

    # ---------- ¿DOS fuentes distintas? -> monoizar cada una -> L/R (fuentes diferentes) ----------
    if fuente == "upload":
        left_b64 = cfg.get("left_b64"); right_b64 = cfg.get("right_b64"); dos = bool(right_b64)
    else:
        left = cfg.get("left") or "demo:tono"; right = cfg.get("right")
        dos = bool(right and right not in ("__mismo", "", None) and right != left)
    if dos:
        if fuente == "upload":
            nomL, Lm = _mono_de_upload(left_b64, cfg.get("left_name", "subido_L"))
            nomR, Rm = _mono_de_upload(right_b64, cfg.get("right_name", "subido_R"))
        else:
            nomL, Lm = _mono_de_spec(left); nomR, Rm = _mono_de_spec(right)
        Lm, Rm, crit = _alinear(Lm, Rm, crit_dur)
        meta.update(izquierdo=nomL, derecho=nomR, criterio_duracion=crit)
        return (Lm, Rm), True, meta

    # ---------- UNA sola fuente ----------
    spec = None if fuente == "upload" else (cfg.get("left") or "demo:tono")
    b64 = cfg.get("left_b64") if fuente == "upload" else None
    nomU = cfg.get("left_name", "subido") if fuente == "upload" else None

    # SIMULACIÓN biaural desde mono (delay+gain interaural) — marcada como simulación
    if bia.get("on"):
        nom, mono = (_mono_de_upload(b64, nomU) if b64 is not None else _mono_de_spec(spec))
        izq, der = biauralizar_mono(mono, bia.get("delay_ms", 0.3), bia.get("gain_L", 1.0), bia.get("gain_R", 0.95))
        meta.update(izquierdo=nom + " (sim L)", derecho=nom + " (sim R)", simulacion_biaural=True)
        return (izq, der), True, meta

    # DEMO (mono de origen): mono, o duplicado a L/R si se pide binaural
    if spec and spec.startswith("demo:"):
        if cfg.get("binaural"):
            nom, audio = cmf.cargar_audio(spec, binaural=True)        # demo mono -> duplicado
            meta.update(izquierdo=nom, derecho=nom + " (duplicado)"); return audio, True, meta
        nom, mono = cmf.cargar_audio(spec, binaural=False)
        meta.update(izquierdo=nom, derecho=nom); return mono, False, meta

    # ARCHIVO / UPLOAD de UN archivo: usar sus PROPIOS canales L/R (BINAURAL REAL; dup si es mono).
    # Clave: un .wav espacializado (p.ej. *_pos60deg) trae la lateralidad EN sus canales; NO monoizar.
    if b64 is not None:
        nom, audio = _carga_upload_binaural(b64, nomU)
    else:
        path = spec if os.path.isabs(spec) else os.path.join(AUDIO_DIR, spec)
        nom, audio = cmf.cargar_audio(path, binaural=True)
    meta.update(izquierdo=str(nom) + " (canal L)", derecho=str(nom) + " (canal R)")
    return audio, True, meta


def _aplicar_ablacion(cel, toggles: dict):
    """Apaga (expresar=False) los organelos cuyo interruptor está en False; NUNCA el soma.
    Devuelve la lista de apagados. Aislado para reutilizar desde build_cell y la ruta en vivo."""
    apagados = []
    for name, org in cel.organelos.items():
        if name == "soma":
            continue
        if not toggles.get(name, True):
            org.expresar = False; apagados.append(name)
    return apagados


def build_cell(cfg: dict, toggles: dict):
    """Construye la célula desde la config, aplica ablación, devuelve (cel, soma, meta)."""
    audio, binaural, meta = build_audio(cfg)
    cel = cmf.celula_madre_funcional(audio, binaural=binaural)
    apagados = _aplicar_ablacion(cel, toggles)
    soma = cel.organelos["soma"]
    meta.update(binaural=bool(binaural), lateralidad_real=bool(getattr(soma, "lateralidad_real", False)),
                apagados=apagados, dur=round(soma.dur, 3), muestras=int(len(soma._L)))
    return cel, soma, meta


# ==============================================================================
# CORRIDA — streaming (Run/worker) y no-streaming (para pruebas)
# ==============================================================================
def _csv_de_rows(rows, meta, eventos=None):
    """CSV con metadatos comentados + BITÁCORA de sesión (eventos con t: inicio, cambios de entrada,
    niveles de todos los canales, exaptación, pausas, fin) + cabecera + filas. Así el CSV contiene
    TODO lo que pasó en la sesión y no hay que inferir nada."""
    cab = (f"# organismo: {meta.get('organismo', ORGANISMO_LABEL)} | organismo_id: {meta.get('organismo_id', ORGANISMO_ID)}\n"
           f"# apagados: {','.join(meta.get('apagados') or []) or 'ninguno'}\n"
           f"# binaural: {meta.get('binaural')} | lateralidad_real: {meta.get('lateralidad_real')} | "
           f"fuente: {meta.get('fuente')} | criterio_duracion: {meta.get('criterio_duracion')}\n"
           f"# izquierdo: {meta.get('izquierdo')} | derecho: {meta.get('derecho')}\n"
           f"# mute_L: {meta.get('mute_L', False)} | mute_R: {meta.get('mute_R', False)}\n")
    if eventos:
        cab += "# === BITÁCORA DE SESIÓN (orden cronológico) ===\n"
        for e in eventos:
            linea = f"# [t={e.get('t_vida', 0)}s] {e.get('tipo')}: {e.get('detalle', '')}"
            if e.get("niveles"):
                linea += " | niveles: " + " ".join(f"{k}={v}" for k, v in e["niveles"].items())
            cab += linea + "\n"
        cab += "# === FIN BITÁCORA ===\n"
    return cab + ",".join(COLS) + "\n" + "\n".join(",".join(str(r[c]) for c in COLS) for r in rows)


def correr(cfg: dict, toggles: dict, sim_s: float):
    """Corrida NO-streaming (para pruebas/CLI). Devuelve dict con cols, csv, rows, meta."""
    cel, soma, meta = build_cell(cfg, toggles)
    sim = min(soma.dur, float(sim_s))
    pasos = max(1, int(sim / DT))
    rows = []
    actuador = ActuadorEsferaV122()
    for _ in range(pasos):
        cel.vivir_un_paso(DT)
        rows.append(_fila(cel, actuador)); _com_observar(rows[-1], meta)
    meta["sim_s"] = round(sim, 2); meta["pasos"] = pasos
    return {"cols": COLS, "csv": _csv_de_rows(rows, meta), "rows": rows, "meta": meta}


def _servidor_src(cfg: dict):
    """Devuelve {host,port,iL,iR,dup} SÓLO para el caso ÓPTIMO de captura continua en vivo: AMBOS
    oídos del MISMO servidor TCP (un stream sincronizado, lateralidad real), o uno solo (derecha
    «igual que izquierda», duplicado). En CUALQUIER OTRO caso devuelve None → se usa la ruta estándar
    por-oído (_build_audio_por_oido), que SÍ mezcla LIBREMENTE cualquier fuente en L y R (servidor +
    archivo/demo/dispositivo, o servidores distintos), resolviendo cada oído por separado en bloques.
    Es decir: mezclar ya NO es un error; sólo renuncia a la captura continua (pasa a modo bloque)."""
    ls, rs = cfg.get("left_src"), cfg.get("right_src")
    presentes = [s for s in (ls, rs) if s]
    serv = [s for s in presentes if s.get("tipo") == "servidor"]
    if not serv:
        return None
    if len(serv) != len(presentes):
        return None     # MEZCLA servidor + (archivo/demo/dispositivo) → ruta estándar por-oído (mezcla libre)
    host = serv[0].get("host", SERVIDOR_HOST); port = int(serv[0].get("port", SERVIDOR_PORT))
    for s in serv:
        if s.get("host", SERVIDOR_HOST) != host or int(s.get("port", SERVIDOR_PORT)) != port:
            return None  # servidores DISTINTOS en cada oído → ruta estándar por-oído (cada uno por su lado)
    iL = int(ls["channel_index"]) if ls else None
    iR = int(rs["channel_index"]) if rs else None
    dup = iR is None                                   # derecha «igual» → duplicar el canal izquierdo
    if dup:
        iR = iL
    return {"host": host, "port": port, "iL": iL, "iR": iR, "dup": dup}



def _aplicar_mute_soma(soma, mute_L=False, mute_R=False):
    """Corta/restaura canales de entrada sin destruir el audio original.

    Funciona con el soma binaural nuevo (_L/_R) y degrada con seguridad si sólo existe
    _audio mono. Para fuentes en vivo por servidor, además se aplica al bloque antes de
    realimentar. Esto permite hacer cortes experimentales durante la corrida.
    """
    try:
        if hasattr(soma, "_L") and hasattr(soma, "_R"):
            if not hasattr(soma, "_orig_L"):
                soma._orig_L = np.array(soma._L, dtype=np.float64).copy()
                soma._orig_R = np.array(soma._R, dtype=np.float64).copy()
            soma._L = np.zeros_like(soma._orig_L) if mute_L else soma._orig_L.copy()
            soma._R = np.zeros_like(soma._orig_R) if mute_R else soma._orig_R.copy()
            if hasattr(soma, "_audio"):
                soma._audio = (soma._L + soma._R) / 2.0
            return True
        if hasattr(soma, "_audio"):
            if not hasattr(soma, "_orig_audio"):
                soma._orig_audio = np.array(soma._audio, dtype=np.float64).copy()
            soma._audio = np.zeros_like(soma._orig_audio) if (mute_L or mute_R) else soma._orig_audio.copy()
            return True
    except Exception:
        return False
    return False


def _aplicar_mute_bloque(L, R, mute_L=False, mute_R=False):
    if mute_L:
        L = np.zeros_like(L)
    if mute_R:
        R = np.zeros_like(R)
    return L, R


class Run:
    """Corrida en streaming. El hilo worker CONSTRUYE la célula (incluida la captura en vivo,
    que puede tardar varios segundos) y LUEGO empuja una fila por paso. La construcción ocurre
    AQUÍ (no en /start) para NO bloquear la petición HTTP — eso causaba BrokenPipe cuando la
    grabación de un dispositivo tardaba y el navegador cerraba. Los errores se emiten por la
    COLA (evento SSE 'error'), no por la respuesta de /start."""
    def __init__(self, cfg, toggles, sim_s):
        self.cfg, self.toggles, self.sim_s = cfg, toggles, sim_s
        self.q = queue.Queue(); self.rows = []; self.meta = {}
        self.stop = False; self.paused = False; self.done = False; self.error = None
        self.mute_L = bool(cfg.get("mute_L", False)); self.mute_R = bool(cfg.get("mute_R", False))
        self.cel = None; self.soma = None
        self.actuador = ActuadorEsferaV122()
        self.eventos = []                       # BITÁCORA de sesión (todo lo que pasa, con t)
        self._ev_lock = threading.Lock()
        self._pending_canales = None            # (iL,iR) a aplicar en vivo (solo fuente servidor)
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self): self.thread.start()

    # ---- BITÁCORA: registrar TODO lo que pasa en la sesión (para no inferir nada) ----
    def _log_evento(self, tipo, detalle, extra=None):
        """Anota un evento con el tiempo de vida actual y lo emite por SSE (timeline en vivo)."""
        ev = {"t_vida": round(self.rows[-1]["t"], 2) if self.rows else 0.0, "tipo": tipo, "detalle": detalle}
        if extra:
            ev.update(extra)
        with self._ev_lock:
            self.eventos.append(ev)
        self.q.put({"__evento__": ev})

    def _snapshot_niveles(self):
        """Niveles (RMS) de TODOS los canales del servidor en este instante (para la bitácora)."""
        try:
            snap = _monitor().snapshot()
            if snap.get("ok"):
                return {c["nombre"]: c["rms"] for c in snap["canales"] if c["rms"] > 0.0015}
        except Exception:
            pass
        return {}

    def cambiar_entradas(self, iL, iR):
        """Pide cambiar L/R EN VIVO (lo aplica el worker en el próximo bloque; solo fuente servidor)."""
        self._pending_canales = (int(iL), int(iR))

    def set_mute(self, left=None, right=None):
        """Corta/restaura canal izquierdo y/o derecho durante la corrida."""
        if left is not None:
            self.mute_L = bool(left)
        if right is not None:
            self.mute_R = bool(right)
        if self.soma is not None:
            _aplicar_mute_soma(self.soma, self.mute_L, self.mute_R)
        _snapshot_comunicacion_entrante()
        self._log_evento("corte_audio", f"L={'OFF' if self.mute_L else 'ON'} · R={'OFF' if self.mute_R else 'ON'}")

    def _loop(self):
        # ¿Fuente en vivo del servidor TCP? → ruta dedicada (conexión persistente, trozos pequeños).
        try:
            sv = _servidor_src(self.cfg)
        except Exception as e:
            self.error = str(e); self.done = True
            self.q.put({"__error__": str(e)}); self.q.put(None); return
        if sv is not None:
            self._loop_servidor(sv); return

        try:
            cel, soma, meta = build_cell(self.cfg, self.toggles)   # construye/CAPTURA en el worker
        except Exception as e:
            self.error = str(e); self.done = True
            self.q.put({"__error__": str(e)}); self.q.put(None); return
        self.cel = cel; self.soma = soma
        _aplicar_mute_soma(soma, self.mute_L, self.mute_R)
        meta["organismo"] = ORGANISMO_LABEL; meta["organismo_id"] = ORGANISMO_ID
        meta["mute_L"] = self.mute_L; meta["mute_R"] = self.mute_R
        self.cel = cel; self.soma = soma
        _aplicar_mute_soma(soma, self.mute_L, self.mute_R)
        meta["organismo"] = ORGANISMO_LABEL; meta["organismo_id"] = ORGANISMO_ID
        meta["mute_L"] = self.mute_L; meta["mute_R"] = self.mute_R
        _rest = _despertar(self.soma)                     # DESPERTAR: restaura la historia previa de disco
        if _rest:
            self._log_evento("despertar", f"historia restaurada de disco: {', '.join(_rest)}")
        self.meta = meta
        continuo = bool(self.cfg.get("continuo"))
        meta["continuo"] = continuo

        if continuo:
            # ---- CAPTURA CONTINUA (sin límite): procesa el bloque, captura el siguiente,
            #      realimenta el soma (campo persiste) → vive hasta que se pulse Detener ----
            meta["sim_s"] = "∞ (continuo)"; meta["pasos_total"] = "∞"
            self.q.put({"__meta__": meta})
            CAP = 300000                                      # tope de filas en memoria (~8 h); ring
            while not self.stop:
                for _ in range(max(1, int(soma.dur / DT))):
                    if self.stop:
                        break
                    while self.paused and not self.stop:
                        time.sleep(0.05)
                    cel.vivir_un_paso(DT); fila = _fila(cel, self.actuador)
                    self.rows.append(fila)
                    if len(self.rows) > CAP:
                        self.rows.pop(0)
                    _com_observar(fila, self.meta)
                    self.q.put(fila); time.sleep(0.004)
                if self.stop:
                    break
                try:
                    audio, binaural, _m = build_audio(self.cfg)   # captura el SIGUIENTE bloque
                except Exception as e:
                    self.q.put({"__error__": "captura de bloque: " + str(e)}); break
                if binaural and isinstance(audio, tuple):
                    audio = _aplicar_mute_bloque(audio[0], audio[1], self.mute_L, self.mute_R)
                elif self.mute_L or self.mute_R:
                    audio = np.zeros_like(audio)
                soma.realimentar(audio, binaural)             # nuevo bloque, campo continúa
            self.done = True; self.q.put(None); return

        # ---- modo de duración FIJA (existente) ----
        sim = min(soma.dur, float(self.sim_s)); pasos = max(1, int(sim / DT))
        meta["sim_s"] = round(sim, 2); meta["pasos_total"] = pasos
        self.q.put({"__meta__": meta})                            # 1er evento: metadatos (L/R, etc.)
        for _ in range(pasos):
            if self.stop:
                break
            while self.paused and not self.stop:
                time.sleep(0.05)
            cel.vivir_un_paso(DT)
            fila = _fila(cel, self.actuador)
            self.rows.append(fila)
            _com_observar(fila, self.meta)
            self.q.put(fila)
            time.sleep(0.004)                  # ritmo observable (no inunda el SSE)
        self.done = True
        self.q.put(None)                       # centinela de fin

    def _loop_servidor(self, sv):
        """Ruta EN VIVO desde el servidor TCP. Diferencias clave que resuelven la lentitud:
          · CONEXIÓN PERSISTENTE: un solo _LectorServidor para toda la sesión (no reconecta por
            bloque → sin cortes ni esperas de re-handshake).
          · TROZOS PEQUEÑOS (LIVE_CHUNK): lee 0.5 s por vez y emite filas de inmediato. El arranque
            es ~0.5 s y la actualización es fluida, INDEPENDIENTE de la duración total pedida (antes,
            pedir N s leía un bloque de N s antes de mostrar nada → 'se demora muchísimo' con N>10).
          · Duración: 'continuo' = hasta Detener; fijo = self.sim_s segundos en total (no por bloque)."""
        LIVE_CHUNK = 0.5                                    # s por lectura del stream (responsividad)
        lector = None
        try:
            lector = _LectorServidor(sv["host"], sv["port"], sv["iL"], sv["iR"])
            L, R = lector.leer_bloque(LIVE_CHUNK)
            L, R = _aplicar_mute_bloque(L, R, self.mute_L, self.mute_R)
            audio = (L, L.copy()) if sv["dup"] else (L, R)
            cel = cmf.celula_madre_funcional(audio, binaural=True)
            apagados = _aplicar_ablacion(cel, self.toggles)
            soma = cel.organelos["soma"]
            dev = lector.hs.get("device", "servidor")
            der = (f"{_nombre_canal(dev, sv['iL'])} (duplicado)" if sv["dup"]
                   else _nombre_canal(dev, sv["iR"]))
            meta = {"fuente": f"servidor TCP {sv['host']}:{sv['port']}", "criterio_duracion": "-",
                    "simulacion_biaural": False, "izquierdo": _nombre_canal(dev, sv["iL"]), "derecho": der,
                    "binaural": True, "lateralidad_real": bool(getattr(soma, "lateralidad_real", False)),
                    "apagados": apagados, "dur": round(soma.dur, 3), "muestras": int(len(soma._L))}
            if lector.sr != SR:
                meta["aviso"] = f"servidor a {lector.sr} Hz ≠ {SR} Hz de la célula (sin remuestreo)"
        except Exception as e:
            self.error = str(e); self.done = True
            self.q.put({"__error__": str(e)}); self.q.put(None)
            if lector is not None:
                lector.cerrar()
            return

        self.cel = cel; self.soma = soma                  # exponer al autosave (y a set_mute)
        _rest = _despertar(soma)                          # DESPERTAR: restaura la historia previa de disco
        if _rest:
            self._log_evento("despertar", f"historia restaurada de disco: {', '.join(_rest)}")
        self.meta = meta
        continuo = bool(self.cfg.get("continuo"))
        meta["continuo"] = continuo
        total_pasos = None if continuo else max(1, int(round(max(0.5, float(self.sim_s)) / DT)))
        meta["sim_s"] = "∞ (continuo)" if continuo else round(max(0.5, float(self.sim_s)), 2)
        meta["pasos_total"] = "∞" if continuo else total_pasos
        self.q.put({"__meta__": meta})                         # 1er evento: metadatos (L/R, etc.)
        self._log_evento("inicio", f"L={meta['izquierdo']} · R={meta['derecho']} · "
                         f"{'continuo' if continuo else str(meta['sim_s'])+'s'}",
                         {"niveles": self._snapshot_niveles()})

        CAP = 300000                                           # tope de filas en memoria (~8 h); ring
        hechos = 0; ult_niv = time.time(); exapto_prev = False
        try:
            while not self.stop:
                # procesa el trozo ya cargado (el primero ya está; los siguientes por realimentar)
                for _ in range(max(1, int(soma.dur / DT))):
                    if self.stop:
                        break
                    while self.paused and not self.stop:
                        time.sleep(0.05)
                    cel.vivir_un_paso(DT); fila = _fila(cel, self.actuador)
                    self.rows.append(fila)
                    if len(self.rows) > CAP:
                        self.rows.pop(0)
                    _com_observar(fila, self.meta)
                    self.q.put(fila); hechos += 1; time.sleep(0.004)
                    if fila.get("exaptacion_activa") and not exapto_prev:        # onset de exaptación
                        self._log_evento("exaptacion", f"XE={fila.get('XE')} Ω_op={fila.get('Omega_op')}")
                    exapto_prev = bool(fila.get("exaptacion_activa"))
                    if total_pasos is not None and hechos >= total_pasos:
                        break
                if self.stop or (total_pasos is not None and hechos >= total_pasos):
                    break
                if self._pending_canales:                       # CAMBIO DE ENTRADAS EN VIVO
                    niL, niR = self._pending_canales; self._pending_canales = None
                    dev = lector.hs.get("device", "servidor")
                    antes = f"L={_nombre_canal(dev, lector.iL)} · R={_nombre_canal(dev, lector.iR)}"
                    lector.set_canales(niL, niR)
                    sv["iL"], sv["iR"], sv["dup"] = niL, niR, (niL == niR)
                    meta["izquierdo"] = _nombre_canal(dev, niL)
                    meta["derecho"] = _nombre_canal(dev, niR) + (" (duplicado)" if sv["dup"] else "")
                    self._log_evento("cambio_entrada",
                                     f"{antes}  →  L={meta['izquierdo']} · R={meta['derecho']}",
                                     {"niveles": self._snapshot_niveles()})
                    self.q.put({"__meta__": meta})              # refresca L/R en la interfaz
                if time.time() - ult_niv >= 2.0:                # NIVELES de todos los canales c/2s
                    con = self._snapshot_niveles()
                    self._log_evento("niveles", f"{len(con)} canal(es) con señal", {"niveles": con})
                    ult_niv = time.time()
                try:
                    L, R = lector.leer_bloque(LIVE_CHUNK)       # siguiente trozo (campo persiste)
                except Exception as e:
                    self.q.put({"__error__": "captura en vivo: " + str(e)}); break
                L, R = _aplicar_mute_bloque(L, R, self.mute_L, self.mute_R)
                soma.realimentar((L, L.copy()) if sv["dup"] else (L, R), True)
        finally:
            lector.cerrar()
        self._log_evento("fin", f"{len(self.rows)} pasos · {round(self.rows[-1]['t'],1) if self.rows else 0}s de vida")
        self.done = True; self.q.put(None)

    def csv(self): return _csv_de_rows(self.rows, self.meta, self.eventos)


RUN: "Run|None" = None
RUN_LOCK = threading.Lock()


class MonitorNiveles:
    """Monitor de NIVELES por canal (VU). Mantiene UNA conexión propia al servidor de audio y
    calcula el RMS suavizado + pico de CADA canal en vivo, para ver qué canales reciben señal
    antes/durante la captura. Independiente de la célula y de la selección L/R (es otro cliente
    más del servidor; el servidor difunde a todos). Se activa solo cuando alguien mira la ventana:
    si nadie consulta /niveles por unos segundos, suelta la conexión (no la mantiene en vano)."""

    IDLE = 8.0   # s sin que nadie consulte → soltar la conexión

    def __init__(self, host=SERVIDOR_HOST, port=SERVIDOR_PORT) -> None:
        self.host, self.port = host, port
        self.rms = []; self.pico = []; self.nch = 0; self.device = ""; self.sr = SR
        self.ok = False; self.mensaje = "iniciando…"
        self._lock = threading.Lock()
        self._stop = False
        self._last_poll = time.time()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop:
            if not SERV_OK:
                with self._lock:
                    self.ok = False; self.mensaje = f"puente TCP no disponible: {SERV_ERR}"
                time.sleep(2.0); continue
            if time.time() - self._last_poll > self.IDLE:          # nadie mirando → reposo
                with self._lock:
                    self.ok = False; self.mensaje = "en reposo (sin observadores)"
                time.sleep(0.4); continue
            try:
                cli = AudioStreamClient(host=self.host, port=self.port, timeout=2.0)
                hs = cli.handshake()
            except Exception as e:
                with self._lock:
                    self.ok = False; self.mensaje = f"sin servidor en {self.host}:{self.port} ({e})"
                time.sleep(1.5); continue
            nch = int(hs.get("channels", 0))
            with self._lock:
                self.nch = nch; self.device = hs.get("device", "servidor")
                self.sr = int(hs.get("sample_rate", SR))
                self.rms = [0.0] * nch; self.pico = [0.0] * nch
                self.ok = True; self.mensaje = "ok"
            try:
                for blk in cli.frames():
                    if self._stop or time.time() - self._last_poll > self.IDLE:
                        break
                    b = blk.astype(np.float64)
                    inst = np.sqrt(np.mean(b ** 2, axis=0)); pk = np.max(np.abs(b), axis=0)
                    with self._lock:
                        for c in range(min(nch, inst.shape[0])):
                            self.rms[c] = 0.8 * self.rms[c] + 0.2 * float(inst[c])   # EMA (VU)
                            self.pico[c] = max(0.85 * self.pico[c], float(pk[c]))     # pico con caída
            except Exception:
                pass
            finally:
                cli.cerrar()
            with self._lock:
                self.ok = False; self.mensaje = "reconectando…"

    def snapshot(self) -> dict:
        self._last_poll = time.time()                              # marca que hay observador
        with self._lock:
            return {"ok": self.ok, "device": self.device, "nch": self.nch, "sample_rate": self.sr,
                    "mensaje": self.mensaje,
                    "canales": [{"channel_index": c, "nombre": _nombre_canal(self.device, c),
                                 "rms": round(self.rms[c], 5), "pico": round(self.pico[c], 5)}
                                for c in range(self.nch)]}


MONITOR: "MonitorNiveles|None" = None
MONITOR_LOCK = threading.Lock()


def _monitor() -> MonitorNiveles:
    """Devuelve el monitor de niveles (lo crea al primer uso; un hilo de fondo perezoso)."""
    global MONITOR
    with MONITOR_LOCK:
        if MONITOR is None:
            MONITOR = MonitorNiveles()
    return MONITOR


# ==============================================================================
# FRONTEND (tema oscuro · pestañas · 8 ventanas · compacto/completo · SSE)
# ==============================================================================
HTML = r"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Célula Madre — Organismo B — Laboratorio en vivo</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js"}}</script>
<script type="module">
import * as THREE from 'three';

function clamp01(x){ x=Number(x||0); return Math.max(0, Math.min(1, x)); }

// Cabeza 3D real estética: un solo grupo orgánico; ojos hundidos, orejas suaves y cabeza clara.
window.drawVSTCabeza3DReal = function(container, thetaDeg, params={}){
  if(!container) return;

  if(!container._vstHeadReal){
    container.innerHTML = '';

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
    camera.position.set(0, 0.08, 4.45);

    const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true, powerPreference:'high-performance'});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.25;
    container.appendChild(renderer.domElement);

    // Iluminación clara y satinada, cercana al look aprobado.
    const hemi = new THREE.HemisphereLight(0xddeeff, 0x182231, 1.7);
    scene.add(hemi);

    const key = new THREE.DirectionalLight(0xffffff, 2.6);
    key.position.set(3.8, 4.2, 5.0);
    scene.add(key);

    const fill = new THREE.DirectionalLight(0xb7d7ff, 1.1);
    fill.position.set(-3.5, 1.8, 2.2);
    scene.add(fill);

    const rim = new THREE.DirectionalLight(0x66cfff, 1.2);
    rim.position.set(-4.0, 1.2, -3.0);
    scene.add(rim);

    const warm = new THREE.PointLight(0xffd7b0, 1.5, 6);
    warm.position.set(1.0, 1.2, 2.3);
    scene.add(warm);

    // Grupo único: TODO gira junto como una sola cabeza.
    const cabeza = new THREE.Group();
    scene.add(cabeza);

    // Esfera base clara, porcelana/metal satinado. No oscura.
    const headGeo = new THREE.SphereGeometry(1, 96, 96);
    const headMat = new THREE.MeshPhysicalMaterial({
      color:0xe6dfd5,
      metalness:0.14,
      roughness:0.32,
      clearcoat:0.38,
      clearcoatRoughness:0.24,
      reflectivity:0.55
    });
    const head = new THREE.Mesh(headGeo, headMat);
    head.scale.set(1.05, 1.0, 1.02);
    cabeza.add(head);

    // Ojos como la referencia: cuenca hundida, blanco visible, iris azul profundo y brillo grande.
    function crearOjo(x){
      const grupo = new THREE.Group();

      const cuencaGeo = new THREE.SphereGeometry(0.215, 56, 56);
      const cuencaMat = new THREE.MeshPhysicalMaterial({
        color:0xc9c2b8,
        metalness:0.05,
        roughness:0.36,
        clearcoat:0.25,
        clearcoatRoughness:0.20
      });
      const cuenca = new THREE.Mesh(cuencaGeo, cuencaMat);
      cuenca.scale.set(1.15, 1.0, 0.34);
      cuenca.position.z = -0.035;
      grupo.add(cuenca);

      const blancoGeo = new THREE.SphereGeometry(0.165, 64, 64);
      const blancoMat = new THREE.MeshPhysicalMaterial({
        color:0xf3f6fb,
        roughness:0.10,
        clearcoat:1.0,
        clearcoatRoughness:0.04,
        reflectivity:0.65
      });
      const blanco = new THREE.Mesh(blancoGeo, blancoMat);
      blanco.scale.set(1.0, 1.0, 0.45);
      blanco.position.z = 0.018;
      grupo.add(blanco);

      const irisGeo = new THREE.SphereGeometry(0.094, 48, 48);
      const irisMat = new THREE.MeshPhysicalMaterial({
        color:0x063f78,
        emissive:0x031a3a,
        emissiveIntensity:0.22,
        roughness:0.04,
        clearcoat:1.0,
        clearcoatRoughness:0.015,
        reflectivity:0.9
      });
      const iris = new THREE.Mesh(irisGeo, irisMat);
      iris.scale.set(1.0, 1.0, 0.30);
      iris.position.z = 0.090;
      grupo.add(iris);

      const irisLuzGeo = new THREE.SphereGeometry(0.062, 32, 32);
      const irisLuzMat = new THREE.MeshBasicMaterial({
        color:0x0d8cff,
        transparent:true,
        opacity:0.42
      });
      const irisLuz = new THREE.Mesh(irisLuzGeo, irisLuzMat);
      irisLuz.scale.set(1.0, 1.0, 0.18);
      irisLuz.position.set(-0.012, 0.012, 0.112);
      grupo.add(irisLuz);

      const pupilaGeo = new THREE.SphereGeometry(0.045, 32, 32);
      const pupilaMat = new THREE.MeshBasicMaterial({color:0x000711});
      const pupila = new THREE.Mesh(pupilaGeo, pupilaMat);
      pupila.scale.set(1.0, 1.0, 0.22);
      pupila.position.z = 0.132;
      grupo.add(pupila);

      const brilloGeo = new THREE.SphereGeometry(0.029, 16, 16);
      const brilloMat = new THREE.MeshBasicMaterial({color:0xffffff});
      const brillo = new THREE.Mesh(brilloGeo, brilloMat);
      brillo.position.set(-0.045, 0.052, 0.154);
      grupo.add(brillo);

      grupo.position.set(x, 0.16, 0.925);
      grupo.rotation.x = -0.04;
      return grupo;
    }

    const ojoL = crearOjo(-0.23);
    const ojoR = crearOjo(0.23);
    cabeza.add(ojoL, ojoR);

    // Sonrisa sutil, sin mejillas/pecas de color.
    const smileCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.15, -0.285, 0.94),
      new THREE.Vector3(-0.06, -0.325, 0.965),
      new THREE.Vector3(0.06, -0.325, 0.965),
      new THREE.Vector3(0.15, -0.285, 0.94)
    ]);
    const smileGeo = new THREE.TubeGeometry(smileCurve, 32, 0.007, 8, false);
    const smileMat = new THREE.MeshBasicMaterial({color:0x4b4745, transparent:true, opacity:0.48});
    const sonrisa = new THREE.Mesh(smileGeo, smileMat);
    cabeza.add(sonrisa);

    // Orejas tipo sensor suave, no cilindros. Integradas al grupo.
    function crearOreja(colorNeon, lado){
      const grupo = new THREE.Group();

      const baseGeo = new THREE.SphereGeometry(0.34, 56, 56);
      baseGeo.scale(0.54, 0.95, 0.28);
      const baseMat = new THREE.MeshPhysicalMaterial({
        color:0xded9cf,
        metalness:0.18,
        roughness:0.24,
        clearcoat:0.65,
        clearcoatRoughness:0.12,
        reflectivity:0.55
      });
      const base = new THREE.Mesh(baseGeo, baseMat);
      grupo.add(base);

      const innerGeo = new THREE.SphereGeometry(0.235, 48, 48);
      innerGeo.scale(0.40, 0.80, 0.16);
      const innerMat = new THREE.MeshPhysicalMaterial({
        color:0xc6c5c0,
        metalness:0.20,
        roughness:0.17,
        clearcoat:0.85,
        clearcoatRoughness:0.05
      });
      const inner = new THREE.Mesh(innerGeo, innerMat);
      inner.position.z = 0.045;
      grupo.add(inner);

      const ringGeo = new THREE.TorusGeometry(0.225, 0.020, 16, 96);
      const ringMat = new THREE.MeshStandardMaterial({
        color:colorNeon,
        emissive:colorNeon,
        emissiveIntensity:0.45,
        roughness:0.16,
        metalness:0.12
      });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.scale.set(0.68, 1.0, 1.0);
      grupo.add(ring);

      const glowGeo = new THREE.TorusGeometry(0.225, 0.044, 12, 96);
      const glowMat = new THREE.MeshBasicMaterial({
        color:colorNeon,
        transparent:true,
        opacity:0.12,
        depthWrite:false
      });
      const glow = new THREE.Mesh(glowGeo, glowMat);
      glow.scale.set(0.68, 1.0, 1.0);
      grupo.add(glow);

      grupo.position.set(lado * 1.04, 0.02, 0.02);
      grupo.rotation.y = lado > 0 ? Math.PI / 2 : -Math.PI / 2;
      grupo.ringMat = ringMat;
      grupo.glowMat = glowMat;
      return grupo;
    }

    const orejaL = crearOreja(0x33eaff, -1);
    const orejaR = crearOreja(0xff5264, 1);
    cabeza.add(orejaL, orejaR);

    function resize(){
      const r = container.getBoundingClientRect();
      const w = Math.max(320, r.width || 560);
      const h = Math.max(260, r.height || 420);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }
    resize();
    window.addEventListener('resize', resize);

    container._vstHeadReal = {scene, camera, renderer, cabeza, orejaL, orejaR, resize};
  }

  const h = container._vstHeadReal;
  h.resize();

  // Rotación real del sólido completo: esfera, ojos, sonrisa y sensores giran juntos.
  h.cabeza.rotation.y = THREE.MathUtils.degToRad(Number(thetaDeg || 0));

  const energiaL = clamp01(params.energiaL ?? params.energyL ?? 0);
  const energiaR = clamp01(params.energiaR ?? params.energyR ?? 0);
  if(h.orejaL.ringMat) h.orejaL.ringMat.emissiveIntensity = 0.45 + energiaL * 2.8;
  if(h.orejaR.ringMat) h.orejaR.ringMat.emissiveIntensity = 0.45 + energiaR * 2.8;
  if(h.orejaL.glowMat) h.orejaL.glowMat.opacity = 0.10 + Math.min(0.35, energiaL * 0.60);
  if(h.orejaR.glowMat) h.orejaR.glowMat.opacity = 0.10 + Math.min(0.35, energiaR * 0.60);

  h.renderer.render(h.scene, h.camera);
};
</script>
<style>
:root{--bg:#0a0e14;--panel:#121925;--ink:#dfe7f0;--mut:#8aa0b8;--gold:#e8b86d;--ok:#5fd38a;--bad:#ff6b6b;--line:#243246;--blue:#6db6ff;--org:#ff8c6b;--pur:#b58cff;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.4 'Helvetica Neue',Arial,sans-serif}
h1{font-size:17px;margin:0;color:var(--gold)}h2{font-size:12px;color:var(--gold);text-transform:uppercase;letter-spacing:.05em;margin:0 0 6px}
.wrap{display:grid;grid-template-columns:330px 1fr;gap:12px;padding:12px;max-width:1500px;margin:auto}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;margin-bottom:12px}
label.fld{display:block;color:var(--mut);font-size:10.5px;margin:8px 0 3px;text-transform:uppercase;letter-spacing:.04em}
select,input[type=number],input[type=text]{background:#0c121b;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px;width:100%}
button{background:#1b2636;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:7px 10px;cursor:pointer;font-size:12px}
button:hover{border-color:var(--gold)}button.go{background:var(--gold);color:#1a1206;font-weight:bold;border:none}
button.csv{border-color:var(--gold);color:var(--gold)}button.sm{padding:5px 8px;font-size:11px}
.row{display:flex;gap:6px;align-items:center;margin-top:8px;flex-wrap:wrap}
.tog{display:flex;align-items:center;gap:7px;padding:2px 0}.tog input{accent-color:var(--gold);width:15px;height:15px}
.tog.req label{color:var(--mut)}.gt{color:var(--mut);font-size:10px;text-transform:uppercase;margin:7px 0 3px}
.tabs{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:8px}
.tab{padding:5px 9px;border:1px solid var(--line);border-radius:6px;cursor:pointer;font-size:11.5px;background:#0c121b;color:var(--mut)}
.tab.on{background:var(--gold);color:#1a1206;border:none;font-weight:bold}
.win{display:none}.win.on{display:block}
.canwrap{position:relative;height:200px;background:#0c121b;border:1px solid var(--line);border-radius:6px;padding:4px;margin-bottom:8px}
.status{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--mut);margin-bottom:8px}
.status b{color:var(--ink)}
.big{font-size:22px;color:var(--gold);font-weight:bold}
#ev{font:11px/1.5 'SF Mono',Menlo,monospace;background:#070a0f;border:1px solid var(--line);border-radius:6px;padding:8px;height:240px;overflow:auto}
.fuente-box{display:none}.fuente-box.on{display:block}
.mut{color:var(--mut)}.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--gold)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px}

.actgrid{display:grid;grid-template-columns:1.45fr .9fr;gap:12px}
.actscene{height:430px;position:relative;background:radial-gradient(circle at 50% 35%,#172231,#080b10 75%);border:1px solid var(--line);border-radius:8px;overflow:hidden;perspective:900px}
.actscene:before{content:"";position:absolute;inset:52% -20% -25% -20%;background:
 linear-gradient(rgba(109,182,255,.07) 1px,transparent 1px),
 linear-gradient(90deg,rgba(109,182,255,.07) 1px,transparent 1px);
 background-size:42px 42px;transform:perspective(600px) rotateX(62deg);transform-origin:50% 0}
.acttitle{position:absolute;top:12px;left:0;right:0;text-align:center;font-size:11px;color:#dfe7f0;text-transform:uppercase}
.actangle{position:absolute;top:30px;left:0;right:0;text-align:center;font-size:28px;color:#69e35f;font-weight:bold}
.actdir{position:absolute;top:63px;left:0;right:0;text-align:center;font-size:11px;color:#dfe7f0}
.actbody{position:absolute;left:50%;top:52%;width:246px;height:246px;margin:-123px 0 0 -123px;transform-style:preserve-3d;transition:transform .08s linear;transform:rotateY(0deg);}
.head3d{position:absolute;left:50%;top:50%;width:220px;height:220px;margin:-110px 0 0 -110px;transform-style:preserve-3d;border-radius:50%;filter:drop-shadow(0 30px 38px rgba(0,0,0,.62));}
.ear{position:absolute;top:76px;width:54px;height:88px;border-radius:50%;z-index:1;opacity:.96;transform-style:preserve-3d;filter:drop-shadow(0 14px 18px rgba(0,0,0,.45));}
.ear:before{content:"";position:absolute;inset:-7px;border-radius:50%;background:inherit;filter:blur(6px);opacity:.30;}
.ear:after{content:"";position:absolute;inset:8px;border-radius:50%;background:radial-gradient(circle at 38% 30%,rgba(255,255,255,.70),rgba(255,255,255,.12) 42%,rgba(0,0,0,.30) 88%);box-shadow:inset -8px -8px 16px rgba(0,0,0,.28), inset 7px 5px 12px rgba(255,255,255,.22);}
.earL{left:-1px;background:radial-gradient(circle at 35% 28%,#d8ffff 0%,#73e7f2 30%,#2aa9c3 60%,#0d526b 100%);transform:translateZ(-34px) rotateY(-28deg) scaleX(.82);}
.earR{right:-1px;background:radial-gradient(circle at 35% 28%,#ffe2e4 0%,#ff9aa0 32%,#e65461 62%,#823139 100%);transform:translateZ(-34px) rotateY(28deg) scaleX(.82);}
.sphere{position:absolute;left:13px;top:13px;width:220px;height:220px;border-radius:50%;z-index:3;transform:translateZ(24px);background:
 radial-gradient(circle at 37% 31%,rgba(255,255,255,.96) 0%,rgba(255,255,255,.76) 9%,rgba(242,229,222,.72) 18%,rgba(190,174,166,.72) 43%,rgba(102,104,108,.88) 72%,rgba(34,38,45,.98) 100%),
 radial-gradient(circle at 50% 50%,#e9eef3 0%,#9ea7b1 48%,#343a44 100%);box-shadow:inset -42px -28px 52px rgba(0,0,0,.48),inset 20px 14px 34px rgba(255,255,255,.24),0 26px 58px rgba(0,0,0,.62);overflow:hidden;}
.sphere:before{content:"";position:absolute;left:20%;top:13%;width:44%;height:38%;border-radius:50%;background:radial-gradient(circle,rgba(255,255,255,.95) 0%,rgba(255,255,255,.55) 25%,rgba(255,255,255,.10) 60%,rgba(255,255,255,0) 78%);filter:blur(2px);}
.sphere:after{content:"";position:absolute;inset:0;border-radius:50%;background:linear-gradient(90deg,rgba(88,215,243,.18) 0%,rgba(88,215,243,0) 22%,rgba(255,255,255,0) 50%,rgba(255,107,107,0) 76%,rgba(255,107,107,.18) 100%),radial-gradient(circle at 66% 62%,rgba(255,140,107,.20),rgba(0,0,0,0) 40%);mix-blend-mode:screen;pointer-events:none;}
.eye{position:absolute;top:102px;width:38px;height:48px;border-radius:50%;z-index:4;background:radial-gradient(circle at 33% 25%,#ffffff 0%,#a8c6de 11%,#24415f 37%,#07111c 74%,#02050a 100%);box-shadow:0 0 0 4px rgba(255,255,255,.46),0 0 0 6px rgba(0,0,0,.28),inset -8px -10px 18px rgba(0,0,0,.68),0 10px 20px rgba(0,0,0,.35);}
.eye:before{content:"";position:absolute;left:10px;top:7px;width:12px;height:12px;border-radius:50%;background:rgba(255,255,255,.86);filter:blur(.4px);}
.eye:after{content:"";position:absolute;right:7px;bottom:8px;width:15px;height:12px;border-radius:50%;background:rgba(255,112,112,.24);filter:blur(3px);}
.eyeL{left:80px;transform:translateZ(48px) rotateY(8deg);}
.eyeR{right:80px;transform:translateZ(48px) rotateY(-8deg);}
.orbit{position:absolute;left:50%;top:64%;width:450px;height:128px;margin:-64px 0 0 -225px;border-radius:50%;border:1px solid rgba(151,182,213,.26);box-shadow:0 0 28px rgba(109,182,255,.08),inset 0 0 35px rgba(109,182,255,.05);transform:perspective(780px) rotateX(70deg);pointer-events:none;}
.orbit:before{content:"";position:absolute;inset:16px;border-radius:50%;border:1px dashed rgba(151,182,213,.15);}
.orbit:after{content:"";position:absolute;left:50%;top:-7px;width:1px;height:142px;background:linear-gradient(transparent,rgba(151,182,213,.35),transparent);}
.axisline{position:absolute;left:50%;top:24%;height:56%;border-left:1px dashed rgba(255,255,255,.55)}
.actarrow{position:absolute;left:50%;bottom:62px;width:5px;height:105px;background:#68d94f;transform-origin:50% 100%;box-shadow:0 0 12px rgba(104,217,79,.6);border-radius:4px}
.actarrow:before{content:"";position:absolute;top:-12px;left:-9px;border-left:12px solid transparent;border-right:12px solid transparent;border-bottom:22px solid #68d94f}
.headBoxLive{position:absolute;left:50%;top:50%;width:420px;height:320px;margin:-160px 0 0 -210px;z-index:4;filter:drop-shadow(0 30px 38px rgba(0,0,0,.62));}
@media(max-width:1100px){.headBoxLive{width:340px;height:260px;margin:-130px 0 0 -170px}}
.earinfo{position:absolute;top:126px;width:180px;font-size:12px;line-height:1.7}.earinfo b{font-size:12px;text-transform:uppercase}
.earinfo.left{left:28px}.earinfo.right{right:28px;text-align:right}
.bars{display:flex;gap:3px;margin-top:5px}.right .bars{justify-content:flex-end}
.bar{width:14px;height:14px;border:1px solid currentColor;background:transparent;opacity:.75}.bar.on{background:currentColor}
.actcards{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:8px}
.actcard{background:#0c121b;border:1px solid var(--line);border-radius:7px;padding:10px;text-align:center}.actcard .k{font-size:10px;color:#8aa0b8;text-transform:uppercase}.actcard .v{font-size:20px;color:var(--gold)}
.formula{background:#0c121b;border:1px solid var(--line);border-radius:7px;padding:12px;margin-bottom:8px;color:#b8c7d9}
.formula h3{margin:0 0 8px;color:#58d7f3;font-size:12px;text-transform:uppercase}
.formula code{display:block;background:#071019;border:1px solid #223147;border-radius:6px;padding:9px;margin:6px 0;color:#dfe7f0;white-space:pre-wrap;font-family:'SF Mono',Menlo,monospace;font-size:11px}
.actnote{font-size:11px;color:#8aa0b8;line-height:1.55}
@media(max-width:1100px){.actgrid{grid-template-columns:1fr}.actscene{height:360px}.earinfo{display:none}.actcards{grid-template-columns:repeat(2,1fr)}}

</style></head><body>
<div class="wrap">
  <!-- ================= CONTROLES ================= -->
  <div>
    <div class="panel">
      <h1>🧬 Célula Madre — Organismo B — Laboratorio en vivo</h1>
      <div class="mut">Entrada biaural · observación en tiempo real · ablación real</div>

      <div class="mut" style="font-size:10.5px">Dos entradas INDEPENDIENTES. Cada una = un canal de
        dispositivo (p.ej. Rødecaster 16ch) o un archivo. Pueden ser distintas.</div>

      <label class="fld">🟦 Entrada izquierda → hemisferio L</label>
      <select id="selL"></select>
      <input type="file" id="upL" accept=".wav" style="display:none;margin-top:6px">

      <label class="fld">🟥 Entrada derecha → hemisferio R</label>
      <select id="selR"></select>
      <input type="file" id="upR" accept=".wav" style="display:none;margin-top:6px">

      <div id="fuentesMsg" class="mut" style="font-size:10.5px;margin-top:6px"></div>
      <div class="panel" style="background:#0c121b;margin:8px 0 4px;padding:8px">
        <h2>🔇 Cortes experimentales</h2>
        <div class="mut" style="font-size:10px;margin-bottom:4px">Corta/restaura cada oído durante la corrida. Queda registrado en la bitácora y CSV.</div>
        <div class="tog"><input type="checkbox" id="muteL"><label for="muteL">Cortar canal izquierdo / voz-relación</label></div>
        <div class="tog"><input type="checkbox" id="muteR"><label for="muteR">Cortar canal derecho / mundo externo</label></div>
      </div>
      <div class="tog" style="margin-top:8px"><input type="checkbox" id="continuo"><label for="continuo">🔴 Captura CONTINUA (sin límite — hasta Detener)</label></div>
      <div class="grid2">
        <div><label class="fld" id="lblCap">Captura en vivo (s)</label><input type="number" id="capSeg" value="10" min="1" max="120"></div>
        <div><label class="fld">&nbsp;</label><button class="sm" id="bAuto" style="width:100%">Auto-par pos/neg (R)</button></div>
      </div>

      <label class="fld" id="lblSim">Segundos de simulación</label>
      <input type="number" id="sim" value="6" min="1" max="120" step="1">
      <div class="row">
        <button class="go" id="bStart" style="flex:1">▶ Iniciar</button>
        <button class="sm" id="bPause">⏸</button><button class="sm" id="bStop">⏹</button>
      </div>
      <div class="row">
        <button class="sm" id="bClear">Limpiar</button>
        <button class="csv sm" id="bCsv" style="flex:1">⬇ CSV (parcial/final)</button>
      </div>
      <div class="row">
        <button class="sm" id="bAplicar" style="flex:1" title="Cambia L/R sin reiniciar; queda en la bitácora">↻ Aplicar entradas (en vivo)</button>
        <button class="sm" id="bSesion" title="Descarga la bitácora de la sesión (JSON)">⬇ Bitácora</button>
      </div>
    </div>

    <div class="panel">
      <h2>🎚 Niveles de entrada (LED) · en vivo</h2>
      <div class="mut" style="font-size:10px;margin-bottom:4px"><b>MASTER</b> = lo que el organismo OYE (cualquier fuente: Rødecaster, wav, demo). Debajo, cada canal del servidor.</div>
      <div id="master" class="mut" style="margin-bottom:6px">—</div>
      <div id="leds" class="mut">cargando…</div>
      <div style="margin-top:8px;border-top:1px solid #243246;padding-top:7px">
        <div style="display:flex;align-items:center;gap:8px;font-size:10.5px">
          <span style="width:104px;color:#9fb1c6">🗣 Volumen de voz</span>
          <input type="range" id="vozVol" min="0.10" max="0.95" step="0.01" value="0.40" style="flex:1">
          <span id="vozVolVal" style="width:40px;text-align:right;color:#8aa0b8">0.40</span>
        </div>
        <div class="mut" style="font-size:9px;margin-top:2px">RMS objetivo de la voz (equipara con Rode/wav, que suenan ~1.0). Aplica en vivo y afecta lo que el par oye.</div>
      </div>
    </div>

    <div class="panel">
      <h2>🔊 Voz del organismo · escuchar / grabar</h2>
      <div class="mut" style="font-size:10px;margin-bottom:6px">Reproduce la voz por las <b>bocinas del Mac</b> (vía navegador → funciona igual en nativo y en Docker). Grabar descarga un .wav de lo que dijo.</div>
      <div class="row">
        <button class="sm" id="bEscuchar" style="flex:1" onclick="_vozEscuchar()">🔊 Escuchar voz</button>
        <button class="sm" id="bGrabar" onclick="_vozGrabar()">⏺ Grabar</button>
      </div>
      <div style="display:flex;align-items:center;gap:8px;font-size:10.5px;margin-top:6px">
        <span style="width:104px;color:#9fb1c6">🔊 Volumen escucha</span>
        <input type="range" id="vozMon" min="0" max="25" step="0.5" value="8" style="flex:1">
        <span id="vozMonVal" style="width:40px;text-align:right;color:#8aa0b8">8×</span>
      </div>
      <div class="mut" style="font-size:9px;margin-top:2px">Sólo amplifica TU escucha (no cambia la voz real ni lo que el par oye). La voz es señal costosa: en soledad suena bajo.</div>
      <div id="vozStat" class="mut" style="font-size:9.5px;margin-top:4px">en silencio</div>
    </div>

    <div class="panel">
      <h2>🤝 Altruismo / Cooperación (díada) · en vivo</h2>
      <div class="mut" style="font-size:10px;margin-bottom:4px">Gobernanza del locus O-N22 entre A↔B: atractor, β_crit, disposición, Ψ_alma, costo de desacople.</div>
      <div id="altru" class="mut">esperando datos…</div>
    </div>

    <div class="panel">
      <h2>Organelos (interruptores · ablación real)</h2>
      <div id="toggles"></div>
      <div class="row"><button class="sm" id="tAll">Todos</button><button class="sm" id="tNone">Solo soma</button></div>
    </div>
  </div>

  <!-- ================= OBSERVACIÓN ================= -->
  <div>
    <div class="panel">
      <div class="row" style="justify-content:space-between">
        <div><button class="tab on" id="mCompacto" onclick="setModo('compacto')">Monitor compacto</button>
             <button class="tab" id="mCompleto" onclick="setModo('completo')">Laboratorio completo</button></div>
        <div id="estado" class="big">listo</div>
      </div>
      <div class="status" id="status"></div>
      <div class="tabs" id="winTabs"></div>
      <div id="wins"></div>
    </div>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
const SR_DT=0.1;
let cols=[], buf={}, charts={}, es=null, paused=false, modo='compacto', tabActual=null;
let nrec=0, t0=0, evPrev={};

// ---- definición de ventanas (Req 5) ----
// GLOSARIO canónico: qué significa cada sigla según la Teoría Cosmosemiótica (para no tener que recordar).
const GLOSARIO={
 Omega:'Estado representacional del organismo [0,1]: nivel de organización del campo Φ.',
 omega_A:'Promedio de los hemisferios de entrada (ω_L+ω_R)/2: lo que el organismo PERCIBE.',
 omega_A_L:'Percibido-vs-esperado del oído izquierdo.',
 omega_A_R:'Percibido-vs-esperado del oído derecho.',
 omega_B:'Referencia/expectativa interna (sistema B en silencio): lo que ESPERA.',
 gradiente:'ω_A−ω_B: desajuste entre lo percibido y lo esperado (la "sorpresa" que fuerza al sistema).',
 omega_L:'Estado del hemisferio izquierdo.', omega_R:'Estado del hemisferio derecho.',
 energia_L:'Volumen real (RMS) de la entrada izquierda.', energia_R:'Volumen real (RMS) de la entrada derecha.',
 balance_LR:'Energía L↔R: + domina izquierda, − domina derecha, 0 parejo.',
 lateralidad:'|ω_L−ω_R|. OJO: incluye el sesgo estructural de los hemisferios, NO solo el input (para el input mira energía/balance).',
 coherencia_biaural:'Coseno(Φ_L,Φ_R): 1 idénticos, −1 opuestos.',
 C_b:'Consciencia básica: magnitud de la representación de 1er orden |R₁| (O-N5.1).',
 R2:'Meta-representación R(R): el modelo que el organismo hace de su propia representación (2º orden).',
 self_coherencia:'Coherencia del sí-mismo: consistencia de su auto-modelo.',
 C_m:'Consciencia metacognitiva (O-N8.4): emerge cuando C_b FALLA; convoca la reorganización (sube en crisis, baja al resolverse).',
 LF_op:'Libertad funcional operativa: capacidad de operar sobre {competencia≠operación}.',
 LF_struct:'Libertad funcional estructural (capacidad latente).',
 lf_nivel:'Genealogía LF: 0 ninguno · 1 juego · 2 ritual · 3 negación operativa (O-N7.2).',
 juego:'Exploración libre (1er escalón de LF).', ritual:'Patrón repetido estabilizado (2º escalón de LF).',
 negacion:'Negación operativa: operar sobre la propia representación, "decir No" (2º orden, ≠ inhibición).',
 OI:'Organismicidad Integrada (O-N9.14): cuánto ES organismo (H+memoria+exaptación+LF−ética). ≥0.7 pleno · 0.4–0.7 proto · <0.4 no.',
 Lambda_Cos:'Razón cosmosemiótica (Δ_struct·LF)/|e_R|·A_sys-env: "salud del cierre" (en el código es el Λ nativo V122).',
 invariantes_ok:'Cuántos de los 6 invariantes de viabilidad se cumplen (κ_P,κ_Δ,κ_O,κ_V,κ_LF,κ_H).',
 A_sys_env:'Acoplamiento sistema–entorno (viable si ≥κ_V): cuán ajustado está al entorno.',
 e_R:'Error de representación / oscilación reciente (viable si 0<|e_R|<κ_O).',
 Omega_op:'Ω operativo = 1+XE: estado ampliado cuando hay exaptación.',
 XE:'EXAPTACIÓN: uso NUEVO de una capacidad existente en un dominio nuevo (ΔLF>0 ∧ ΔA_sys-env≥0). El motor de la evolución cosmosemiótica.',
 mutacion:'Variación interna (cambio del parámetro).',
 adaptacion_activa:'Adaptación: mejora del ajuste SIN ganar libertad (ΔLF≈0). Distinta de exaptación.',
 exaptacion_activa:'Flag: en este paso se cumplió la condición de exaptación.',
 activacion_latente:'Activación de una capacidad/locus reservado (PRE).',
 H_homeostasis:'Homeostasis: salud del equilibrio interno (variables en rango). Componente del OI.',
 x_interna:'Variable interna regulada por la homeostasis.', en_rango:'1 si x_interna está dentro del rango viable.',
 RC_total:'Ruido contextual total observado: RC = ICR + IRDE.',
 RC_relacional:'RC que entra por el oído izquierdo: voz/estado del organismo par.',
 RC_externo:'RC que entra por el oído derecho: música, video o mundo externo.',
 ICR:'Integración de ruido contextual: fracción convertida en sentido/acoplamiento.',
 IRDE:'Ruido contextual desviado: fracción que aparece como riesgo o desacople.',
 ICR_ratio:'Proporción de RC metabolizada como ICR.',
 IRDE_ratio:'Proporción de RC metabolizada como IRDE.',
 RC_delta_salud:'Cambio ponderado de salud semio-organísmica usado para repartir ICR/IRDE.',
 RC_atencion_L:'Atención lateral izquierda: cuánto reclama orientación la voz/oreja L.',
 RC_atencion_R:'Atención lateral derecha: cuánto reclama orientación el mundo/oreja R.',
 RC_comprension_L:'Comprensión funcional atribuida al canal izquierdo.',
 RC_comprension_R:'Comprensión funcional atribuida al canal derecho.',
 RC_riesgo_L:'Riesgo/desacople atribuido al canal izquierdo.',
 RC_riesgo_R:'Riesgo/desacople atribuido al canal derecho.',
 RC_consenso_orientacion:'Consenso mínimo RC para orientar: negativo izquierda, positivo derecha.',
 RC_confianza_comprension:'Permiso de giro dado por comprensión funcional.',
 RC_freno_riesgo:'Freno corporal dado por IRDE/riesgo contextual.',
 act_rc_mix:'Compatibilidad histórica: alias de act_permiso_decisional en E015.',
 act_presencia_L:'Presencia sensorial izquierda: habilita o no la oreja L.',
 act_presencia_R:'Presencia sensorial derecha: habilita o no la oreja R.',
 act_propuesta_atencional:'Diferencia atencional R-L: lo que propone la escucha lateral.',
 act_decision_RC:'Decisión final RC/ICR/IRDE que se transforma en objetivo de giro.',
 act_bloqueo_IRDE:'Bloqueo por riesgo/desacople: IRDE no gira al lado opuesto, degrada la decisión.',
 act_permiso_decisional:'Permiso endógeno para que la comprensión oriente la cabeza.',
 act_evidencia_L:'Evidencia sensorial izquierda: lateralidad como dato, no causa del giro.',
 act_evidencia_R:'Evidencia sensorial derecha: lateralidad como dato, no causa del giro.',
 act_razon_L:'Razón organísmica para orientar hacia la izquierda: comprensión + cierre - riesgo.',
 act_razon_R:'Razón organísmica para orientar hacia la derecha: comprensión + cierre - riesgo.',
 act_necesidad_cierre:'Necesidad de recuperar cierre/acoplamiento del organismo.',
 act_decision_organismica:'Diferencia razón_R-razón_L antes de permiso/bloqueo corporal.',
 act_soporte_sentido:'Soporte endógeno para privilegiar captura de sentido: OI, homeostasis, memoria, libertad, integración y comprensión.',
 act_vulnerabilidad_riesgo:'Vulnerabilidad endógena que permite a IRDE frenar: desorganización, baja homeostasis, baja libertad, error, fatiga y freno RC.',
 act_base_sentido:'Potencia ICR/INR antes de normalizar: soporte de sentido por comprensión/evidencia disponible.',
 act_base_riesgo:'Potencia IRDE antes de normalizar: vulnerabilidad por riesgo/freno/desintegración.',
 act_peso_sentido:'Peso emergente de captura de sentido: base_sentido/(base_sentido+base_riesgo).',
 act_peso_riesgo:'Peso emergente de riesgo: base_riesgo/(base_sentido+base_riesgo).',
};

function bars(v,color){
  const n=6; let lit=Math.max(0,Math.min(n,Math.round((Math.abs(v)||0)*n)));
  let h='<div class="bars" style="color:'+color+'">';
  for(let i=0;i<n;i++)h+='<span class="bar '+(i<lit?'on':'')+'"></span>';
  return h+'</div>';
}
function fmt(x,n=3){x=Number(x||0);return x.toFixed(n);}
function updateActuador(r){
  const theta=Number(r.act_orientacion_deg||0), obj=Number(r.act_objetivo_deg||0), conf=Number(r.act_confianza||0);
  const side=theta>2?'hacia derecha':(theta<-2?'hacia izquierda':'centro');
  const arrow=$('actArrow');
  const head=$('head3dLive');
  if(window.drawVSTCabeza3DReal && head){
    window.drawVSTCabeza3DReal(head, theta, {energiaL:Number(r.energia_L||0), energiaR:Number(r.energia_R||0)});
  }
  if(arrow)arrow.style.transform='translateX(-50%) rotate('+theta+'deg)';
  if($('actTheta'))$('actTheta').textContent=(theta>=0?'+':'')+theta.toFixed(1)+'°';
  if($('actDir'))$('actDir').textContent='('+side+')';
  if($('actWL'))$('actWL').textContent=fmt(r.omega_L,3);
  if($('actWR'))$('actWR').textContent=fmt(r.omega_R,3);
  if($('actEL'))$('actEL').textContent=fmt(r.energia_L,3);
  if($('actER'))$('actER').textContent=fmt(r.energia_R,3);
  if($('barsL'))$('barsL').innerHTML=bars(r.energia_L,'#58d7f3');
  if($('barsR'))$('barsR').innerHTML=bars(r.energia_R,'#ff6b6b');
  const set=(id,val,suf='')=>{const e=$(id);if(e)e.textContent=val+suf;};
  set('actConf',fmt(conf,2));
  set('actObj',(obj>=0?'+':'')+obj.toFixed(1),'°');
  set('actFat',fmt(r.act_fatiga,2));
  set('actZone',fmt(r.act_zona_muerta,1),'°');
  set('actTrem',fmt(r.act_temblor_rms,2),'°');
  set('actDW',(Number(r.act_lateralidad_dw||0)>=0?'+':'')+fmt(r.act_lateralidad_dw,4));
}


// Monitor liviano E010: el actuador conserva datos en CSV, pero no redibuja gráficos históricos en vivo.
const actCharts={};
function mkActCharts(){}
function updateActCharts(){}


const VENTANAS=[
 {id:'actuador', tit:'Actuador: Cabeza 3D real', comp:true, actuador:true},
 {id:'campo', tit:'Campo Φ / Soma', comp:true, desc:'Lo que el organismo percibe (ω_A) frente a lo que espera (ω_B); el gradiente es la sorpresa.', series:[['Omega','Ω','#e8b86d'],['omega_A_L','ω_A_L','#6db6ff'],['omega_A_R','ω_A_R','#ff8c6b'],['omega_B','ω_B','#8aa0b8'],['gradiente','gradiente','#5fd38a']]},
 {id:'conciencia', tit:'Consciencia', comp:true, desc:'De la representación básica (C_b) a la meta-representación (R₂) y la metacognición de crisis (C_m).', series:[['C_b','C_b','#e8b86d'],['R2','R₂','#6db6ff'],['self_coherencia','Self','#5fd38a'],['C_m','C_m','#b58cff']]},
 {id:'libertad', tit:'Libertad funcional', comp:true, desc:'Genealogía juego→ritual→negación (O-N7.2): capacidad de operar sobre la propia representación.', series:[['LF_struct','LF_struct','#8aa0b8'],['LF_op','LF_op','#e8b86d'],['lf_nivel','nivel','#6db6ff'],['juego','juego','#5fd38a'],['ritual','ritual','#b58cff'],['negacion','No','#ff6b6b']]},
 {id:'exaptacion', tit:'★ Exaptación', comp:true, desc:'El motor evolutivo: cuándo el organismo da un uso NUEVO a lo que ya tiene (ΔLF>0 ∧ ΔA_sys-env≥0).', series:[['XE','XE','#5fd38a'],['Omega_op','Ω_op','#e8b86d'],['exaptacion_activa','exapt activa','#ff8c6b'],['adaptacion_activa','adapt','#6db6ff'],['mutacion','mutación','#8aa0b8'],['activacion_latente','latente','#b58cff']]},
 {id:'salud', tit:'Salud del cierre', comp:true, desc:'Cuán organismo es como un todo (OI), su razón cosmosemiótica (Λ_Cos) y los invariantes de viabilidad κ.', series:[['OI','OI','#5fd38a'],['Lambda_Cos','Λ_Cos','#6db6ff'],['invariantes_ok','κ ok','#e8b86d'],['A_sys_env','A_env','#8aa0b8']]},
 {id:'biaural', tit:'Entrada biaural', comp:false, desc:'Volumen real por oído (energía) y balance. Para juzgar el input usa estas, no la "lateralidad".', series:[['energia_L','energía_L','#6db6ff'],['energia_R','energía_R','#ff8c6b'],['balance_LR','balance','#e8b86d'],['coherencia_biaural','coherencia','#5fd38a'],['lateralidad','lateralidad','#b58cff']]},
{id:'rc', tit:'RC = ICR + IRDE', comp:false, desc:'Destino del ruido contextual: conversión en sentido/acoplamiento o desviación/riesgo.', series:[['RC_total','RC','#e8b86d'],['ICR','ICR','#5fd38a'],['IRDE','IRDE','#ff6b6b'],['RC_relacional','RC_rel','#6db6ff'],['RC_externo','RC_ext','#b58cff'],['ICR_ratio','ICR%','#8aa0b8']]},
 {id:'rc_cabeza', tit:'RC: atención y comprensión', comp:false, desc:'Vías segregadas por oído: atención, comprensión y riesgo antes del consenso de orientación.', series:[['RC_atencion_L','at_L','#6db6ff'],['RC_atencion_R','at_R','#ff8c6b'],['RC_comprension_L','comp_L','#5fd38a'],['RC_comprension_R','comp_R','#b58cff'],['RC_riesgo_L','riesgo_L','#8aa0b8'],['RC_riesgo_R','riesgo_R','#ff6b6b'],['RC_consenso_orientacion','consenso','#e8b86d']]},
 {id:'cabeza_decision', tit:'Cabeza: razón organísmica', comp:false, desc:'Cadena E017: presencia habilita, lateralidad aporta evidencia, ICR/INR e IRDE se ponderan endógenamente, fatiga encarna.', series:[['act_evidencia_L','evid_L','#6db6ff'],['act_evidencia_R','evid_R','#ff8c6b'],['act_razon_L','razón_L','#5fd38a'],['act_razon_R','razón_R','#b58cff'],['act_decision_organismica','razón_R-L','#e8b86d'],['act_decision_RC','decisión','#ffffff'],['act_bloqueo_IRDE','bloqueo','#ff6b6b'],['act_peso_sentido','peso sentido','#64f0c8'],['act_peso_riesgo','peso riesgo','#ff8c8c']]},
 {id:'homeostasis', tit:'Homeostasis', comp:false, desc:'Salud del equilibrio interno (H) y la variable regulada x dentro de su rango viable.', series:[['H_homeostasis','H','#5fd38a'],['x_interna','x','#e8b86d'],['en_rango','en_rango','#6db6ff'],['e_R','e_R','#ff6b6b']]},
 {id:'eventos', tit:'Bitácora', comp:false, eventos:true},
];

function setModo(m){modo=m;$('mCompacto').classList.toggle('on',m==='compacto');$('mCompleto').classList.toggle('on',m==='completo');buildTabs();}
function buildTabs(){
  const tabs=$('winTabs'), wins=$('wins'); tabs.innerHTML='';
  const visibles=VENTANAS.filter(v=>modo==='completo'||v.comp||v.eventos);
  visibles.forEach((v,i)=>{
    const b=document.createElement('div');b.className='tab'+((tabActual===v.id||(tabActual===null&&i===0))?' on':'');
    b.textContent=v.tit;b.onclick=()=>{tabActual=v.id;buildTabs();};tabs.appendChild(b);
  });
  if(!visibles.find(v=>v.id===tabActual)) tabActual=visibles[0].id;
  wins.querySelectorAll('.win').forEach(w=>w.classList.remove('on'));
  let w=$('win-'+tabActual); if(w) w.classList.add('on');
}
function buildWins(){
  const wins=$('wins'); wins.innerHTML='';
  VENTANAS.forEach(v=>{
    const d=document.createElement('div');d.className='win';d.id='win-'+v.id;
    if(v.actuador){d.innerHTML=`<h2>ACTUADOR: CABEZA 3D REAL (V122+)</h2>
      <div class="mut" style="font-size:12px;color:var(--gold);margin-bottom:8px"><b>El giro no sigue el volumen. El giro sigue la diferencia relevante.</b></div>
      <div class="actgrid">
        <div>
          <div class="actscene">
            <div class="acttitle">Orientación actual</div><div class="actangle" id="actTheta">+0.0°</div><div class="actdir" id="actDir">(centro)</div>
            <div class="earinfo left" style="color:#58d7f3"><b>Canal izquierdo (L)</b><br>ωL = <span id="actWL">0.000</span><br>EL = <span id="actEL">0.000</span><div id="barsL"></div></div>
            <div class="earinfo right" style="color:#ff8c8c"><b>Canal derecho (R)</b><br>ωR = <span id="actWR">0.000</span><br>ER = <span id="actER">0.000</span><div id="barsR"></div></div>
            <div class="axisline"></div><div class="actarrow" id="actArrow"></div>
            <div class="orbit"></div><div class="headBoxLive" id="head3dLive"></div>
          </div>
          <div class="actcards">
            <div class="actcard"><div class="k">Confianza</div><div class="v" id="actConf">0.00</div></div>
            <div class="actcard"><div class="k">Objetivo</div><div class="v" id="actObj">0.0°</div></div>
            <div class="actcard"><div class="k">Fatiga motor</div><div class="v" id="actFat">0.00</div></div>
            <div class="actcard"><div class="k">Zona muerta</div><div class="v" id="actZone">2.0°</div></div>
            <div class="actcard"><div class="k">Temblor RMS</div><div class="v" id="actTrem">0.00°</div></div>
          </div>
        </div>
        <div>
          <div class="formula"><h3>1) Dirección: lateralidad interna</h3><code>pL/pR = presencia suave del canal
salL = |ωL − ωA|·pL
salR = |ωR − ωA|·pR
Δω = salR − salL
θobj = clamp(k_lat × Δω, −90°, +90°)</code><div class="actnote">La energía L/R sólo actúa como compuerta de presencia: si cortas un oído, ese oído no puede seguir capturando atención.</div></div>
          <div class="formula"><h3>2) Confianza organismal</h3><code>Conf = clamp(w1·R2 + w2·LFop + w3·H
             + w4·Aenv + w5·XE, 0, 1)</code><div class="actnote">R₂ habilita confianza, LF permite reorientar, H evita giro patológico y XE aporta reorganización.</div></div>
          <div class="formula"><h3>3) Dinámica motora</h3><code>err = wrap(θobj − θ)
Δθ = I × F × Conf × err</code><div class="actnote">Inercia, freno exponencial, zona muerta, fatiga y temblor vienen del aprendizaje V122–V150.</div></div>
          <div class="formula"><h3>Δω actual</h3><div style="font-size:28px;color:#69e35f;text-align:center" id="actDW">+0.0000</div></div>
        </div>
      </div>
      <div class="actnote" style="margin-top:10px;color:#8aa0b8">Monitor liviano E010: cabeza 3D real con Three.js; los históricos del actuador se siguen guardando completos en el CSV, pero no se grafican aquí para evitar congelamientos del navegador.</div>`;}
    else if(v.niveles){d.innerHTML=`<h2>${v.tit}</h2><div id="vu" class="mut">cargando niveles…</div>`;}
    else if(v.eventos){d.innerHTML=`<h2>${v.tit}</h2><div class=mut style="font-size:10px;margin-bottom:4px">Todo lo que pasa en la sesión (inicio, cambios de entrada, exaptación, pausas). Se guarda completo en el CSV y en ⬇ Bitácora.</div><div id="ev"></div>`;}
    else{
      const ley=v.series.map(s=>`<div style="margin:1px 0"><b style="color:${s[2]}">${s[1]}</b> — ${GLOSARIO[s[0]]||''}</div>`).join('');
      d.innerHTML=`<h2>${v.tit}</h2>`+(v.desc?`<div class=mut style="font-size:10.5px;margin-bottom:6px">${v.desc}</div>`:'')
        +`<div class="canwrap"><canvas id="c-${v.id}"></canvas></div>`
        +`<div style="font-size:10px;color:#9fb1c6;margin-top:6px;line-height:1.5">${ley}</div>`;
    }
    wins.appendChild(d);
  });
}
function mkChart(v){
  const ds=v.series.map(s=>({label:s[1],data:[],borderColor:s[2],borderWidth:1.6,tension:.2,pointRadius:0}));
  charts[v.id]=new Chart($('c-'+v.id),{type:'line',data:{labels:[],datasets:ds},
    options:{animation:false,responsive:true,maintainAspectRatio:false,
      scales:{x:{ticks:{color:'#8aa0b8',maxTicksLimit:8},grid:{color:'#1a2330'}},y:{ticks:{color:'#8aa0b8'},grid:{color:'#1a2330'}}},
      plugins:{legend:{labels:{color:'#dfe7f0',boxWidth:10,font:{size:10}}}}}});
}
function ev(msg,c){const d=document.createElement('div');if(c)d.className=c;d.textContent=msg;const e=$('ev');if(e)e.prepend(d);}

// ---- carga de listas (audios, organelos, dispositivos) ----
let FUENTES=[];   // descriptores unificados: demos + subir + canales de dispositivo + archivos
fetch('/fuentes').then(r=>r.json()).then(d=>{
  // Solo: canales del SERVIDOR TCP (entrada en vivo) + archivos + demos + subir.
  // Los dispositivos físicos DIRECTOS se omiten a propósito: en este host la captura directa
  // fallaba; la entrada en vivo va siempre por el servidor (VST_AudioServer.py).
  FUENTES=[
    {tipo:'demo',spec:'demo:tono',label:'▮ demo — tono 440Hz'},
    {tipo:'demo',spec:'demo:rosa',label:'▮ demo — ruido rosa'},
    {tipo:'demo',spec:'demo:clicks',label:'▮ demo — clicks Poisson'},
    {tipo:'subir',label:'⬆ subir archivo .wav…'},
    ...(d.comunicacion||[]),             // 🗣 voz del organismo par
    ...(d.servidor||[]),                 // 📡 canales en vivo vía servidor TCP
    ...(d.archivos||[]),
  ];
  const opt=FUENTES.map((s,i)=>`<option value="${i}">${s.label}</option>`).join('');
  $('selL').innerHTML=opt;
  $('selR').innerHTML='<option value="__igual">— igual que izquierda —</option>'+opt;
  const na=(d.archivos||[]).length, sv=d.servidor_info||{}, ci=d.comunicacion_info||{};
  const lineaSrv = sv.ok
    ? `<span class=ok>📡 servidor TCP: ${sv.device} · ${sv.canales} canales</span> <span class=mut>(${sv.host}:${sv.port})</span>`
    : `<span class=warn>📡 sin servidor TCP</span> <span class=mut>${sv.mensaje||''} — corre VST_AudioServer.py para entrada en vivo</span>`;
  const lineaCom = ci.ok ? `<span class=ok>🗣 órgano comunicación</span> <span class=mut>${ci.peer_url}</span>` : `<span class=warn>🗣 sin órgano comunicación</span> <span class=mut>${ci.mensaje||''}</span>`;
  $('fuentesMsg').innerHTML = lineaSrv + '<br>' + lineaCom + '<br>' + `<span class=mut>${na} archivos · 3 demos</span>`;
});
fetch('/organelos').then(r=>r.json()).then(list=>{
  const byG={};list.forEach(o=>{(byG[o.grupo]=byG[o.grupo]||[]).push(o);});
  const c=$('toggles');
  for(const g in byG){const h=document.createElement('div');h.className='gt';h.textContent=g;c.appendChild(h);
    byG[g].forEach(o=>{const w=document.createElement('div');w.className='tog'+(o.req?' req':'');
      w.innerHTML=`<input type="checkbox" id="t_${o.name}" ${o.req?'checked disabled':'checked'}><label for="t_${o.name}">${o.label}</label>`;c.appendChild(w);});}
});
// mostrar el input de archivo de cada oído solo si su fuente elegida es 'subir'
function syncUp(sel,up){const d=FUENTES[+sel.value];up.style.display=(d&&d.tipo==='subir')?'block':'none';}
$('selL').onchange=()=>syncUp($('selL'),$('upL'));
$('selR').onchange=()=>{if($('selR').value==='__igual')$('upR').style.display='none';else syncUp($('selR'),$('upR'));};
$('bAuto').onclick=()=>{const d=FUENTES[+$('selL').value];
  if(!d||d.tipo!=='archivo'){ev('Auto-par: elige un ARCHIVO en la entrada izquierda','bad');return;}
  const par=d.nombre.replace('pos60deg','neg60deg').replace('left','right').replace('_L','_R');
  const i=FUENTES.findIndex(s=>s.tipo==='archivo'&&s.nombre===par);
  if(i>=0){$('selR').value=i;$('upR').style.display='none';ev('Auto-par derecha: '+par,'ok');}else ev('No encontré par de '+d.nombre,'bad');};
$('tAll').onclick=()=>document.querySelectorAll('#toggles input:not([disabled])').forEach(c=>c.checked=true);
$('tNone').onclick=()=>document.querySelectorAll('#toggles input:not([disabled])').forEach(c=>c.checked=false);
$('continuo').onchange=()=>{const c=$('continuo').checked;
  $('lblCap').textContent=c?'Tamaño de bloque (s)':'Captura en vivo (s)';
  $('sim').disabled=c; $('lblSim').style.opacity=c?0.4:1;
  ev(c?'Captura continua ON — corre hasta que pulses ⏹ Detener':'Captura continua OFF','mut');};

// ---- construir config: dos fuentes independientes (left_src / right_src) ----
async function fileB64(inp){const f=inp.files[0];if(!f)return null;return await new Promise(r=>{const x=new FileReader();x.onload=()=>r(x.result);x.readAsDataURL(f);});}
async function srcDe(sel,up){const d=FUENTES[+sel.value];
  if(d.tipo==='subir')return {tipo:'upload',b64:await fileB64(up),name:(up.files[0]||{}).name};
  return d;}
async function buildCfg(){
  const left_src=await srcDe($('selL'),$('upL'));
  const right_src=($('selR').value==='__igual')?null:await srcDe($('selR'),$('upR'));
  const cont=$('continuo').checked;   // captura continua sin límite (segundos = tamaño de bloque)
  const cfg={left_src,right_src,binaural:(right_src===null),segundos:+$('capSeg').value,continuo:cont,criterio_duracion:'min',
             mute_L:$('muteL').checked, mute_R:$('muteR').checked};
  const toggles={};document.querySelectorAll('#toggles input').forEach(c=>toggles[c.id.slice(2)]=c.checked);
  return {cfg,toggles,sim_s:+$('sim').value};
}

// ---- ciclo de vida de la corrida (SSE) ----
function limpiar(){buf={};nrec=0;cols.forEach(c=>buf[c]=[]);buf.t=[];evPrev={};
  Object.values(charts).forEach(ch=>{ch.data.labels=[];ch.data.datasets.forEach(d=>d.data=[]);ch.update('none');});Object.values(actCharts).forEach(ch=>{ch.data.labels=[];ch.data.datasets.forEach(d=>d.data=[]);ch.update('none');});$('ev').innerHTML='';}
async function start(){
  if(es){es.close();es=null;}
  const body=await buildCfg();
  $('estado').textContent='iniciando…';
  let r;try{r=await fetch('/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});}catch(e){$('estado').textContent='error';ev('Error: '+e,'bad');return;}
  const d=await r.json();
  if(d.error){$('estado').textContent='error';ev('Error: '+d.error,'bad');return;}
  // /start responde YA; la captura/construcción ocurre en el worker → llega por SSE (evento meta o fallo)
  cols=d.cols; limpiar(); paused=false; t0=performance.now(); nrec=0;
  ev('… preparando entrada (si es en vivo, capturando '+(+$('capSeg').value)+'s)…','mut');
  $('estado').textContent='capturando/preparando…';
  es=new EventSource('/stream');
  es.addEventListener('meta',e=>{const m=JSON.parse(e.data);pintarStatus(m);t0=performance.now();
    ev('▶ inicio · '+(m.izquierdo||'')+'  |  '+(m.derecho||'')+(m.simulacion_biaural?'  [SIM biaural]':''),'ok');});
  es.addEventListener('fallo',e=>{const m=JSON.parse(e.data);$('estado').textContent='error';
    ev('❌ '+m.error,'bad');es.close();es=null;});
  es.onmessage=e=>{const row=JSON.parse(e.data);recibir(row);};
  es.addEventListener('fin',e=>{$('estado').textContent='terminado';ev('⏹ fin · '+nrec+' pasos','ok');es.close();es=null;});
  es.addEventListener('evento',e=>{const m=JSON.parse(e.data);
    if(m.tipo==='niveles')return;               // los niveles van a la bitácora; en vivo ya están en el VU/LED
    const ic={cambio_entrada:'🔀',exaptacion:'✦',inicio:'▶',fin:'⏹',pausa:'⏸',reanuda:'▶',detener:'⏹'}[m.tipo]||'◆';
    let txt=ic+' '+m.tipo.toUpperCase()+(m.detalle?': '+m.detalle:'');
    if(m.niveles&&Object.keys(m.niveles).length)txt+=' | '+Object.entries(m.niveles).map(([k,v])=>k+'='+v).join('  ');
    ev('t='+m.t_vida+'  '+txt, m.tipo==='exaptacion'?'ok':(m.tipo==='cambio_entrada'?'warn':'mut'));});
  es.onerror=()=>{$('estado').textContent='(conexión cerrada)';};
}
// Cambiar L/R EN VIVO (sin reiniciar): solo entre canales 📡 del servidor; queda en la bitácora.
$('bAplicar').onclick=async()=>{
  const dL=FUENTES[+$('selL').value], dR=($('selR').value==='__igual')?dL:FUENTES[+$('selR').value];
  if(!dL||dL.tipo!=='servidor'||!dR||dR.tipo!=='servidor'){ev('Cambio en vivo: elige canales 📡 del servidor en L y R','bad');return;}
  try{const r=await fetch('/entradas',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({iL:dL.channel_index,iR:dR.channel_index})});
    const d=await r.json();ev(d.ok?'↻ entradas aplicadas en vivo (en la bitácora)':'no se pudo: '+(d.error||''),d.ok?'ok':'bad');}catch(e){ev('error: '+e,'bad');}};
$('bSesion').onclick=()=>{fetch('/sesion').then(r=>r.text()).then(txt=>{if(!txt||txt==='[]'){ev('bitácora vacía','bad');return;}
  const b=new Blob([txt],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);
  a.download='bitacora_sesion_'+new Date().toISOString().slice(0,19).replace(/[:.]/g,'-')+'.json';a.click();ev('bitácora descargada','ok');});};
function recibir(row){
  nrec++; buf.t.push(row.t); cols.forEach(c=>{(buf[c]=buf[c]||[]).push(row[c]);});
  detectarEventos(row); updateActuador(row); updateActCharts(); renderAltruismo(row);
  // actualizar solo la ventana visible (rendimiento)
  const v=VENTANAS.find(x=>x.id===tabActual);
  if(v&&!v.eventos&&charts[v.id]){const ch=charts[v.id];ch.data.labels=buf.t;
    v.series.forEach((s,i)=>{ch.data.datasets[i].data=buf[s[0]]||[];});ch.update('none');}
  if(nrec%5===0){const fps=(nrec/((performance.now()-t0)/1000)).toFixed(1);
    $('estado').textContent=row.OI!=null?('OI '+row.OI):'corriendo';
    pintarStatus(null,{pasos:nrec,fps});}
}
function detectarEventos(r){
  const mk=(k,cond,msg,cl)=>{if(cond&&!evPrev[k]){ev('t='+r.t+'  '+msg,cl);}evPrev[k]=cond;};
  mk('juego',r.juego>0,'▷ inicio JUEGO','warn');
  mk('ritual',r.ritual>0,'▷ inicio RITUAL','warn');
  mk('neg',r.negacion>0,'▷ inicio NEGACIÓN ("No")','warn');
  mk('exapt',r.exaptacion_activa>0,'▷ EXAPTACIÓN activa (XE='+r.XE+')','ok');
  mk('cm',r.C_m>0.3,'△ C_m sobre umbral ('+r.C_m+')','ok');
  mk('hbaja',r.H_homeostasis<0.3,'▽ H bajo umbral ('+r.H_homeostasis+')','bad');
  mk('lbaja',r.Lambda_Cos<0.005,'▽ Λ_Cos bajo umbral','bad');
  if(evPrev.lf!==r.lf_nivel){ev('t='+r.t+'  ⇅ nivel LF → '+r.lf_nivel,'warn');evPrev.lf=r.lf_nivel;}
}
function pintarStatus(meta,live){
  const s=$('status'); if(meta){s.dataset.izq=meta.izquierdo||'-';s.dataset.der=meta.derecho||'-';
    s.dataset.fte=meta.fuente;s.dataset.dur=meta.sim_s+'s';s.dataset.ap=(meta.apagados&&meta.apagados.length)?meta.apagados.join(','):'ninguno';
    s.dataset.crit=meta.criterio_duracion||'-';s.dataset.latr=meta.lateralidad_real;}
  const p=live?live.pasos:0, fps=live?live.fps:'0';
  s.innerHTML=`<span>org: <b>Organismo B</b></span><span>L: <b>${s.dataset.izq||'-'}</b></span><span>R: <b>${s.dataset.der||'-'}</b></span>
    <span>fuente: <b>${s.dataset.fte||'-'}</b></span><span>lat.real: <b>${s.dataset.latr}</b></span>
    <span>dur: <b>${s.dataset.dur||'-'}</b></span><span>pasos: <b>${p||s.dataset.pasos||0}</b></span>
    <span>FPS: <b>${fps}</b></span><span>apagados: <b>${s.dataset.ap||'-'}</b></span>`;
  if(live){s.dataset.pasos=p;}
}
async function aplicarMute(){
  try{
    const r=await fetch('/mute',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({left:$('muteL').checked,right:$('muteR').checked})});
    const d=await r.json();
    ev('🔇 corte actualizado: L='+($('muteL').checked?'OFF':'ON')+' · R='+($('muteR').checked?'OFF':'ON'), d.ok?'warn':'bad');
  }catch(e){ev('error corte: '+e,'bad');}
}
$('muteL').onchange=aplicarMute;
$('muteR').onchange=aplicarMute;
$('bStart').onclick=start;
$('bPause').onclick=()=>{paused=!paused;fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:paused?'pause':'resume'})});$('bPause').textContent=paused?'▶':'⏸';ev(paused?'⏸ pausa':'▶ reanuda','mut');};
$('bStop').onclick=()=>{fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'stop'})});ev('⏹ detener solicitado','mut');};
$('bClear').onclick=limpiar;
$('bCsv').onclick=()=>{fetch('/csv').then(r=>r.text()).then(txt=>{if(!txt){ev('sin datos','bad');return;}
  const b=new Blob([txt],{type:'text/csv'});const a=document.createElement('a');a.href=URL.createObjectURL(b);
  a.download='Celula madre live - Organismo B - '+new Date().toISOString().slice(0,19).replace(/[:.]/g,'-')+'.csv';a.click();ev('CSV descargado','ok');});};

buildWins(); VENTANAS.forEach(v=>{if(!v.eventos&&!v.niveles&&!v.actuador)mkChart(v);}); mkActCharts(); buildTabs(); updateActuador({act_orientacion_deg:0,act_objetivo_deg:0,act_confianza:0,act_fatiga:0,act_zona_muerta:2,act_temblor_rms:0,act_lateralidad_dw:0,omega_L:0,omega_R:0,energia_L:0,energia_R:0});

// ---- VU por canal (lee /niveles del servidor; solo mientras la ventana está visible) ----
function renderNiveles(d){
  const box=$('vu'); if(!box) return;
  if(!d.ok){box.innerHTML='<span class=warn>📡 '+(d.mensaje||'sin servidor')+'</span>'
    +'<div class=mut style="margin-top:4px">Para ver niveles, corre VST_AudioServer.py.</div>';return;}
  const cs=(d.canales||[]).filter(c=>!/^canal \d+$/.test((c.nombre||'').trim()));  // oculta canales SIN nombre (#19/#20 vacíos)
  const con=cs.filter(c=>c.rms>0.002).length;
  let h='<div class=mut style="margin-bottom:6px">'+d.device+' · '+cs.length+' canales · '
       +'<span class=ok>'+con+' con señal</span> · en vivo</div>';
  cs.forEach(c=>{
    const r=c.rms, on=r>0.002, w=Math.min(100,r*400), wp=Math.min(100,c.pico*400);
    const col=on?(r>0.05?'#5fd38a':'#e8b86d'):'#33414f';
    h+='<div style="display:flex;align-items:center;gap:8px;margin:2px 0;font-size:11px">'
      +'<span style="width:160px;color:'+(on?'#dfe7f0':'#8aa0b8')+'">'+(on?'● ':'○ ')+c.nombre+'</span>'
      +'<div style="flex:1;position:relative;height:12px;background:#0c121b;border:1px solid #243246;border-radius:3px;overflow:hidden">'
      +'<div style="position:absolute;left:0;top:0;bottom:0;width:'+w+'%;background:'+col+'"></div>'
      +'<div style="position:absolute;top:0;bottom:0;left:'+wp+'%;width:2px;background:#ff6b6b"></div></div>'
      +'<span style="width:60px;text-align:right;color:#8aa0b8">'+r.toFixed(4)+'</span></div>';
  });
  box.innerHTML=h;
}
// ---- Medidor de LEDs en el panel lateral: SIEMPRE visible, todos los canales ----
function ledStrip(rms,pico){
  const N=10, lit=Math.round(Math.min(1,rms/0.25)*N), pk=Math.round(Math.min(1,pico/0.25)*N);
  let s='<span style="display:inline-flex;gap:1px">';
  for(let i=0;i<N;i++){const on=i<lit, isPk=(i+1===pk);
    let col=on?(i<6?'#5fd38a':(i<9?'#e8b86d':'#ff6b6b')):'#1a2330';
    if(isPk&&!on)col='#7a3b3b';
    s+='<span style="width:6px;height:9px;border-radius:1px;background:'+col+'"></span>';}
  return s+'</span>';}
function organismoRows(d){
  const o=(d&&d.organismo)||{ok:false,canales:{},reserva:false};
  const chans=o.canales||{}, stale=o.age_s!=null&&o.age_s>2.0;
  const mk=(idx,lado)=>{
    const c=chans[lado]||{rms:0,pico:0,nombre:'organismo '+lado};
    const on=o.ok&&c.rms>0.0015, label='#'+idx+' organismo '+lado+(o.reserva?'*':'');
    return '<div style="display:flex;align-items:center;gap:6px;margin:1px 0">'
      +'<span title="'+(c.nombre||label)+'" style="width:86px;font-size:9.5px;color:'+(on?'#dfe7f0':'#6b7d92')+';white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
      +(on?'●':'○')+' '+label+'</span>'+ledStrip(c.rms||0,c.pico||0)
      +(stale?'<span class=mut style="font-size:9px">pausa</span>':'')+'</div>';
  };
  return mk(19,'L')+mk(20,'R');
}
function renderAltruismo(row){
  const box=$('altru'); if(!box||row.disposicion_cooperar==null) return;
  const atr=row.altruismo_atractor||'mudo';
  const col={comunicando:'#5fd38a',emergiendo:'#e8b86d',mudo:'#8aa0b8'}[atr]||'#8aa0b8';
  const disp=+row.disposicion_cooperar||0, beta=+row.altruismo_beta_crit||0,
        psi=+row.altruismo_psi_alma||0, costo=+row.altruismo_costo_desacople||0,
        tau=+row.altruismo_tau||0, coop=(+row.altruismo_coopera||0)>0, sup=disp>beta;
  const bar=(v,c)=>'<div style="flex:1;height:9px;background:#0c121b;border:1px solid #243246;border-radius:3px;overflow:hidden">'
    +'<div style="height:100%;width:'+Math.min(100,Math.max(0,v*100))+'%;background:'+c+'"></div></div>';
  const fila=(et,v,c,extra)=>'<div style="display:flex;align-items:center;gap:6px;margin:2px 0;font-size:10.5px">'
    +'<span style="width:80px;color:#9fb1c6">'+et+'</span>'+bar(v,c)
    +'<span style="width:62px;text-align:right;color:#8aa0b8">'+(extra!=null?extra:v.toFixed(2))+'</span></div>';
  box.innerHTML=
     '<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">'
    +'<span style="font-weight:bold;color:'+col+';font-size:12px">● '+atr.toUpperCase()+'</span>'
    +(coop?'<span class=ok style="font-size:9.5px">V′=A⊕B</span>':'')+'</div>'
    +fila('disposición',disp,'#6db6ff')
    +fila('β_crit',beta,'#b58cff',beta.toFixed(2)+(sup?' ✓':''))
    +fila('Ψ_alma',psi,'#5fd38a',psi.toFixed(2)+(psi>0?' sujeto':' no-suj'))
    +fila('costo desac.',costo,'#ff8c6b')
    +'<div class=mut style="font-size:9.5px;margin-top:3px">τ simbiosis: '+tau.toFixed(1)+'s · coopera: '
    +(coop?'<span class=ok>SÍ</span>':'<span class=mut>aún no</span>')+'</div>';
}
// ---- MASTER: lo que el organismo OYE (cualquier fuente). Responde "¿está sonando?" ----
function renderMaster(d){
  const box=$('master'); if(!box) return;
  const m=(d&&d.master)||{rms:0,pico:0,ok:false};
  const r=+m.rms||0, on=r>0.0015, w=Math.min(100,r*300), wp=Math.min(100,(+m.pico||0)*300);
  const col=on?(r>0.15?'#5fd38a':(r>0.04?'#e8b86d':'#6db6ff')):'#33414f';
  box.innerHTML='<div style="display:flex;align-items:center;gap:8px;font-size:11px">'
    +'<span style="width:104px;color:'+(on?'#dfe7f0':'#8aa0b8')+';font-weight:bold">'+(on?'● ':'○ ')+'MASTER (oído)</span>'
    +'<div style="flex:1;position:relative;height:15px;background:#0c121b;border:1px solid #243246;border-radius:3px;overflow:hidden">'
    +'<div style="position:absolute;left:0;top:0;bottom:0;width:'+w+'%;background:'+col+'"></div>'
    +'<div style="position:absolute;top:0;bottom:0;left:'+wp+'%;width:2px;background:#ff6b6b"></div></div>'
    +'<span style="width:60px;text-align:right;color:#8aa0b8">'+r.toFixed(4)+'</span></div>';
}
function renderLeds(d){
  const box=$('leds'); if(!box) return;
  if(!d.ok){box.innerHTML='<span class=warn>📡 sin servidor</span> <span class=mut style=font-size:9.5px>— corre VST_AudioServer.py</span>'+organismoRows(d);return;}
  const cs=(d.canales||[]).filter(c=>!/^canal \d+$/.test((c.nombre||'').trim()));  // oculta canales SIN nombre (#19/#20 vacíos)
  const con=cs.filter(c=>c.rms>0.002).length;
  let h='<div class=mut style="font-size:9.5px;margin-bottom:4px"><span class=ok>'+con+'</span>/'+cs.length+' con señal</div>';
  cs.forEach(c=>{const on=c.rms>0.002;
    h+='<div style="display:flex;align-items:center;gap:6px;margin:1px 0">'
      +'<span style="width:86px;font-size:9.5px;color:'+(on?'#dfe7f0':'#6b7d92')+';white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
      +(on?'●':'○')+' '+c.nombre.replace('canal ','#')+'</span>'+ledStrip(c.rms,c.pico)+'</div>';});
  h+=organismoRows(d);
  box.innerHTML=h;}

// Un solo sondeo alimenta el MASTER, el LED lateral (siempre) y la ventana VU detallada (si visible)
setInterval(()=>{fetch('/niveles').then(r=>r.json()).then(d=>{renderMaster(d);renderLeds(d);if(tabActual==='niveles')renderNiveles(d);}).catch(()=>{});},350);

// ---- Volumen de voz (voice_target_rms): lee el actual y lo aplica EN VIVO ----
fetch('/comunicacion/estado').then(r=>r.json()).then(s=>{
  const tr=(s&&(s.voice_target_rms))||null, v=$('vozVol');
  if(tr!=null&&v){v.value=tr;$('vozVolVal').textContent=(+tr).toFixed(2);}
}).catch(()=>{});
(function(){const v=$('vozVol'); if(!v) return;
  v.addEventListener('input',()=>{$('vozVolVal').textContent=(+v.value).toFixed(2);});
  v.addEventListener('change',()=>{fetch('/voz_config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({target_rms:+v.value})}).then(r=>r.json()).then(()=>ev('🗣 volumen de voz → '+(+v.value).toFixed(2),'ok')).catch(()=>{});});
})();

// ---- ESCUCHAR / GRABAR la voz del organismo (Web Audio → bocinas del Mac; graba a .wav) ----
(function(){
  let ac=null, monGain=null, playing=false, nextT=0, timer=null, rec=false, recBufs=[], recSR=48000, nblk=0;
  const SEG=1.0;
  async function tick(){
    if(!playing) return;
    const st=$('vozStat');
    try{
      const ab0=await fetch('/voz?seg='+SEG+'&modo=R2D2').then(r=>r.arrayBuffer());
      const ab=await ac.decodeAudioData(ab0.slice(0));
      const s=ac.createBufferSource(); s.buffer=ab; s.connect(monGain);   // → GainNode de monitoreo → bocinas
      const t=Math.max(ac.currentTime+0.02, nextT); s.start(t); nextT=t+ab.duration; nblk++;
      const d0=ab.getChannelData(0); let sm=0,nn=0; for(let i=0;i<d0.length;i+=64){sm+=d0[i]*d0[i];nn++;}
      const rms=Math.sqrt(sm/Math.max(1,nn));
      if(rec){recSR=ab.sampleRate; recBufs.push(d0.slice());
        const segs=recBufs.reduce((a,b)=>a+b.length,0)/recSR;
        if(st) st.textContent='⏺ grabando · '+segs.toFixed(1)+'s · señal voz '+rms.toFixed(3);}
      else if(st) st.textContent='🔊 reproduciendo · '+nblk+' bloques · señal voz rms '+rms.toFixed(3)+' · vol '+(monGain?monGain.gain.value:8)+'× · estado AudioContext: '+ac.state;
    }catch(e){ if(st) st.textContent='⚠ error de audio: '+((e&&e.message)||e); }
    timer=setTimeout(tick, SEG*1000*0.9);
  }
  window._vozEscuchar=function(){
    if(playing){playing=false; clearTimeout(timer); $('bEscuchar').textContent='🔊 Escuchar voz';
      if(!rec){const st=$('vozStat'); if(st) st.textContent='en silencio';} return;}
    ac=ac||new (window.AudioContext||window.webkitAudioContext)(); ac.resume();
    if(!monGain){monGain=ac.createGain(); const _mv=$('vozMon'); monGain.gain.value=_mv?(+_mv.value):8; monGain.connect(ac.destination);}  // respeta 0× (silencio)
    playing=true; nblk=0; nextT=ac.currentTime; $('bEscuchar').textContent='⏸ Detener';
    const st=$('vozStat'); if(st&&!rec) st.textContent='conectando…'; tick();
  };
  // volumen de escucha (sólo monitoreo, no toca la voz real)
  {const v=$('vozMon'); if(v) v.addEventListener('input',()=>{$('vozMonVal').textContent=(+v.value).toFixed(0)+'×'; if(monGain) monGain.gain.value=+v.value;});}
  window._vozGrabar=function(){
    if(rec){ rec=false; $('bGrabar').textContent='⏺ Grabar';
      if(recBufs.length) descargarWav(recBufs, recSR);
      recBufs=[]; const st=$('vozStat'); if(st) st.textContent=playing?'🔊 sonando por las bocinas':'en silencio';
    }else{ if(!playing) window._vozEscuchar();
      rec=true; recBufs=[]; $('bGrabar').textContent='⏹ Detener y descargar'; }
  };
  function descargarWav(bufs, sr){
    let n=bufs.reduce((a,b)=>a+b.length,0); const pcm=new Float32Array(n); let o=0;
    bufs.forEach(b=>{pcm.set(b,o); o+=b.length;});
    const org=(document.title.indexOf('Organismo B')>=0)?'B':'A';
    const blob=encodeWav(pcm,sr), url=URL.createObjectURL(blob), a=document.createElement('a');
    a.href=url; a.download='voz_Organismo'+org+'_'+new Date().toISOString().slice(0,19).replace(/[:.]/g,'-')+'.wav';
    a.click(); setTimeout(()=>URL.revokeObjectURL(url),2000); ev('🔊 voz grabada ('+(n/sr).toFixed(1)+'s) descargada','ok');
  }
  function encodeWav(s,sr){
    const buf=new ArrayBuffer(44+s.length*2), v=new DataView(buf);
    const w=(o,t)=>{for(let i=0;i<t.length;i++)v.setUint8(o+i,t.charCodeAt(i));};
    w(0,'RIFF'); v.setUint32(4,36+s.length*2,true); w(8,'WAVE'); w(12,'fmt '); v.setUint32(16,16,true);
    v.setUint16(20,1,true); v.setUint16(22,1,true); v.setUint32(24,sr,true); v.setUint32(28,sr*2,true);
    v.setUint16(32,2,true); v.setUint16(34,16,true); w(36,'data'); v.setUint32(40,s.length*2,true);
    let o=44; for(let i=0;i<s.length;i++){let x=Math.max(-1,Math.min(1,s[i])); v.setInt16(o,x<0?x*0x8000:x*0x7FFF,true); o+=2;}
    return new Blob([buf],{type:'audio/wav'});
  }
})();
ev('Laboratorio listo. Elige entrada y pulsa Iniciar.');
</script></body></html>"""


# ==============================================================================
# SERVIDOR
# ==============================================================================
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, body, ctype="application/json"):
        b = body.encode("utf-8") if isinstance(body, str) else body
        try:                                            # el cliente puede haber cerrado (BrokenPipe)
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        u = urlparse(self.path); path = u.path; qs = parse_qs(u.query)
        if path in ("/", "/index.html"):
            self._send(200, HTML, "text/html")
        elif path == "/audios":
            wavs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
            self._send(200, json.dumps(wavs, ensure_ascii=False))
        elif path == "/organelos":
            self._send(200, json.dumps([{"name": n, "grupo": g, "label": l, "req": rq}
                                        for n, g, l, rq in ORG_UI], ensure_ascii=False))
        elif path == "/dispositivos":
            self._send(200, json.dumps(_dispositivos(), ensure_ascii=False))
        elif path == "/fuentes":
            self._send(200, json.dumps(_fuentes(), ensure_ascii=False))
        elif path == "/niveles":
            snap = _monitor().snapshot()
            snap["organismo"] = _snapshot_comunicacion_entrante()
            snap["master"] = _master_input_level()      # lo que el organismo OYE (cualquier fuente)
            self._send(200, json.dumps(snap, ensure_ascii=False))
        elif path == "/csv":
            self._send(200, RUN.csv() if RUN else "", "text/csv")
        elif path == "/sesion":
            self._send(200, json.dumps(RUN.eventos if RUN else [], ensure_ascii=False))
        elif path in ("/comunicacion/estado", "/voz_estado"):
            self._send(200, json.dumps(ORGANO_COMUNICACION.estado() if ORGANO_COMUNICACION else {"ok": False, "error": COM_ERR}, ensure_ascii=False))
        elif path in ("/comunicacion/bloque.wav", "/voz"):
            seg = float((qs.get("seg") or ["0.5"])[0])
            modo_voz = (qs.get("modo") or ["R2D2"])[0]
            if ORGANO_COMUNICACION is None:
                self._send(503, json.dumps({"ok": False, "error": COM_ERR}))
            else:
                gain_voz = float((qs.get("gain") or [str(COMUNICACION_VOICE_GAIN)])[0])
                try:
                    wav = ORGANO_COMUNICACION.wav_bytes(seg, modo=modo_voz, gain=gain_voz)
                except TypeError:
                    wav = ORGANO_COMUNICACION.wav_bytes(seg)
                self._send(200, wav, "audio/wav")
        elif path == "/stream":
            self._stream()
        else:
            self._send(404, json.dumps({"error": "no encontrado"}))

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if self.path == "/start":
                self._start(self._body())
            elif path == "/control":
                self._control(self._body())
            elif path == "/entradas":
                self._entradas(self._body())
            elif path == "/mute":
                self._mute(self._body())
            elif path == "/voz_config":
                self._voz_config(self._body())
            else:
                self._send(404, json.dumps({"error": "no encontrado"}))
        except Exception as e:
            import traceback; traceback.print_exc()
            self._send(200, json.dumps({"error": str(e)}))

    # ---- iniciar una corrida ----
    def _start(self, req):
        _nacer(req.get("cfg", {}), req.get("toggles", {}), req.get("sim_s", 6))
        self._send(200, json.dumps({"cols": COLS, "ok": True}))   # responde al instante

    # ---- pausa / reanuda / detiene ----
    def _control(self, req):
        a = req.get("action")
        if RUN:
            if a == "pause": RUN.paused = True; RUN._log_evento("pausa", "captura en pausa")
            elif a == "resume": RUN.paused = False; RUN._log_evento("reanuda", "captura reanudada")
            elif a == "stop": RUN.stop = True; RUN._log_evento("detener", "detención solicitada")
        self._send(200, json.dumps({"ok": True}))

    # ---- cortar/restaurar canales L/R EN VIVO (queda en la bitácora) ----
    def _mute(self, req):
        if RUN and not RUN.done:
            RUN.set_mute(req.get("left"), req.get("right"))
            self._send(200, json.dumps({"ok": True, "mute_L": RUN.mute_L, "mute_R": RUN.mute_R}))
        else:
            self._send(200, json.dumps({"ok": False, "error": "no hay una corrida activa"}))

    # ---- VOLUMEN DE LA VOZ en vivo (voice_target_rms): las voces suenan tan bajo porque se normalizan
    #      a un RMS objetivo pequeño; subirlo las equipara con el R#de/wav. Afecta lo que el par oye. ----
    def _voz_config(self, req):
        tr = req.get("target_rms")
        v = None
        if tr is not None and ORGANO_COMUNICACION is not None:
            v = max(0.05, min(0.98, float(tr)))
            ORGANO_COMUNICACION.voice_target_rms = v          # cuando NO hay gobernanza (solo)
            try:
                if GOB_ALTRUISMO is not None:
                    GOB_ALTRUISMO.base_voice_rms = v          # en díada: escala la voz costosa modulada
            except Exception:
                pass
            if RUN is not None:
                RUN._log_evento("voz_volumen", f"voice_target_rms → {v:.2f}")
        self._send(200, json.dumps({"ok": True, "voice_target_rms": v}))

    # ---- cambiar entradas L/R EN VIVO (solo fuente servidor; queda en la bitácora) ----
    def _entradas(self, req):
        if RUN and not RUN.done:
            RUN.cambiar_entradas(req.get("iL", 0), req.get("iR", 0))
            self._send(200, json.dumps({"ok": True}))
        else:
            self._send(200, json.dumps({"ok": False, "error": "no hay una corrida activa"}))

    # ---- streaming SSE: una fila por paso, en vivo ----
    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        run = RUN
        try:
            while run is not None:
                try:
                    fila = run.q.get(timeout=1.0)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n"); self.wfile.flush(); continue
                if fila is None:                       # centinela de fin
                    self.wfile.write(b"event: fin\ndata: {}\n\n"); self.wfile.flush(); break
                if isinstance(fila, dict) and "__meta__" in fila:      # metadatos (L/R, etc.)
                    self.wfile.write(("event: meta\ndata: " + json.dumps(fila["__meta__"]) + "\n\n").encode("utf-8")); self.wfile.flush(); continue
                if isinstance(fila, dict) and "__error__" in fila:     # error de construcción/captura
                    self.wfile.write(("event: fallo\ndata: " + json.dumps({"error": fila["__error__"]}) + "\n\n").encode("utf-8")); self.wfile.flush(); continue
                if isinstance(fila, dict) and "__evento__" in fila:    # bitácora (cambio entrada, niveles, etc.)
                    self.wfile.write(("event: evento\ndata: " + json.dumps(fila["__evento__"], ensure_ascii=False) + "\n\n").encode("utf-8")); self.wfile.flush(); continue
                self.wfile.write(("data: " + json.dumps(fila) + "\n\n").encode("utf-8")); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass                                       # el cliente cerró la pestaña


def _fuentes_servidor(host=SERVIDOR_HOST, port=SERVIDOR_PORT):
    """Pregunta a VST_AudioServer.py (si está corriendo) y lista UN descriptor por canal, con los
    MISMOS nombres que la conexión directa ('device — canal N'), pero el dato viaja por TCP desde el
    servidor —que sí ve la Rødecaster— en vez de abrir el dispositivo directo (que fallaba en este
    host). Conexión efímera: handshake y cierre. Si no hay servidor, lista vacía + motivo."""
    if not SERV_OK:
        return [], {"ok": False, "host": host, "port": port,
                    "mensaje": f"puente TCP no disponible: {SERV_ERR}"}
    try:
        cli = AudioStreamClient(host=host, port=port, timeout=1.5)
    except OSError as e:
        return [], {"ok": False, "host": host, "port": port,
                    "mensaje": f"sin servidor en {host}:{port} ({e}). Corre VST_AudioServer.py."}
    try:
        hs = cli.handshake()
    except Exception as e:
        cli.cerrar()
        return [], {"ok": False, "host": host, "port": port, "mensaje": f"handshake falló: {e}"}
    cli.cerrar()
    dev = hs.get("device", "servidor"); nch = int(hs.get("channels", 0)); sr = int(hs.get("sample_rate", SR))
    canales = [{"tipo": "servidor", "host": host, "port": port, "channel_index": c,
                "nombre": dev, "canales": nch, "sample_rate": sr,
                "nombre_canal": _nombre_canal(dev, c),
                "label": f"📡 {_nombre_canal(dev, c)}"} for c in range(nch)]
    return canales, {"ok": True, "host": host, "port": port, "device": dev, "canales": nch, "sample_rate": sr}


def _fuentes_comunicacion():
    if not COM_OK:
        return [], {"ok": False, "mensaje": COM_ERR}
    modos = ["R2D2", "FULL_STATE_NOTES", "FULL_STATE_OSC", "PHYSIO_VOICE", "NULL_STATE", "SHUFFLED_STATE", "NOISE_MATCHED"]
    fuentes = []
    sep = "&" if "?" in COMUNICACION_PEER_URL else "?"
    for modo in modos:
        fuentes.append({"tipo": "comunicacion",
                        "url": f"{COMUNICACION_PEER_URL}{sep}modo={modo}&gain={COMUNICACION_VOICE_GAIN}",
                        "nombre": f"voz del organismo par · {modo}",
                        "modo": modo,
                        "label": f"🗣 voz par ({COMUNICACION_PEER_PORT}) · {modo}"})
    return (fuentes, {"ok": True, "peer_url": COMUNICACION_PEER_URL,
                      "organismo_id": ORGANISMO_ID, "organismo": ORGANISMO_LABEL, "voice_gain": COMUNICACION_VOICE_GAIN})


def _rc_observar(fila):
    ORGANO_RC.observar(fila)


def _par_estado_throttled(intervalo=0.5):
    """Estado del par (su fila) cacheado; se refresca como mucho cada `intervalo` s (no en cada paso)."""
    if not DIADA_OK or COMUNICACION_PEER_ESTADO_URL is None:
        return None
    ahora = time.time()
    if ahora - _PAR_ESTADO["t"] >= intervalo:
        _PAR_ESTADO["t"] = ahora
        d = leer_estado_par(COMUNICACION_PEER_ESTADO_URL, timeout=1.2)
        if d is not None:
            _PAR_ESTADO["data"] = d
    return _PAR_ESTADO["data"]


def _com_observar(fila, meta=None):
    # GOBERNANZA DE ALTRUISMO de la díada (VST_DiadaAltruismo): conduce el locus del genoma con la
    # fila propia + el estado del par; MEZCLA la disposición en la fila (para que el par la lea y
    # para los monitores) y MODULA la voz como SEÑAL COSTOSA (voice_target_rms ∝ disposicion_cooperar).
    if GOB_ALTRUISMO is not None:
        try:
            res = GOB_ALTRUISMO.paso(fila, _par_estado_throttled(), dt=DT)
            fila["disposicion_cooperar"] = res["disposicion_cooperar"]
            fila["altruismo_coopera"] = 1.0 if res["coopera"] else 0.0
            fila["altruismo_beta_crit"] = res["beta_crit"]
            fila["altruismo_psi_alma"] = res["psi_alma"]
            fila["altruismo_tau"] = res["tau_simbiosis"]
            fila["altruismo_costo_desacople"] = res["costo_desacople"]
            fila["altruismo_S_shared"] = res["S_shared"]
            fila["altruismo_atractor"] = res["atractor"]
            if ORGANO_COMUNICACION is not None:
                ORGANO_COMUNICACION.voice_target_rms = res["voice_rms"]   # voz = señal costosa modulada
        except Exception:
            fila.setdefault("disposicion_cooperar", 0.0)
    if ORGANO_COMUNICACION is not None:
        ORGANO_COMUNICACION.observar(fila, meta)


def _fuentes():
    """Lista UNIFICADA de fuentes seleccionables por oído: canales del SERVIDOR TCP (recomendado en
    este host) + cada CANAL de cada dispositivo de entrada DIRECTO (max_input_channels>0) + los .wav
    de la carpeta del proyecto. La Rødecaster Pro II aparece como N entradas 'nombre — canal n', tanto
    vía servidor (📡, el dato viaja por TCP) como directa (🎙, abre el dispositivo)."""
    archivos = [{"tipo": "archivo", "nombre": os.path.basename(p),
                 "label": "📄 " + os.path.basename(p)}
                for p in sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))]
    dispositivos, vivo = [], {"disponible": SD_OK}
    if SD_OK:
        try:
            for i, d in enumerate(_sd.query_devices()):
                nch = int(d.get("max_input_channels", 0))
                for c in range(nch):
                    dispositivos.append({"tipo": "dispositivo", "device_index": i, "channel_index": c,
                                         "nombre": d["name"], "canales": nch,
                                         "label": f"🎙 {d['name']} — canal {c + 1}"})
        except Exception as e:
            vivo = {"disponible": False, "mensaje": f"error consultando dispositivos: {e}"}
    else:
        vivo.update(mensaje=f"sounddevice no instalado ({SD_ERR})",
                    instrucciones="venv/bin/pip install sounddevice (brew install portaudio). "
                                  "Audio del SISTEMA en macOS: brew install blackhole-2ch y elígelo como entrada.")
    servidor, servidor_info = _fuentes_servidor()
    comunicacion, comunicacion_info = _fuentes_comunicacion()
    return {"servidor": servidor, "servidor_info": servidor_info,
            "comunicacion": comunicacion, "comunicacion_info": comunicacion_info,
            "dispositivos": ([] if AUDIO_VIVO_DIRECTO_DESHABILITADO else dispositivos), "archivos": archivos, "audio_vivo": vivo}


def _dispositivos():
    """Lista dispositivos de entrada (sounddevice) o explica por qué no está disponible."""
    if not SD_OK:
        return {"disponible": False,
                "mensaje": f"sounddevice no instalado ({SD_ERR})",
                "instrucciones": "Instala:  venv/bin/pip install sounddevice  (requiere PortAudio: "
                                 "brew install portaudio). Para AUDIO DEL SISTEMA en macOS necesitas un "
                                 "dispositivo de loopback: brew install blackhole-2ch, y elige 'BlackHole 2ch' "
                                 "como entrada (macOS no entrega el audio del sistema directamente al programa)."}
    devs = []
    try:
        for i, d in enumerate(_sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                devs.append({"index": i, "name": d["name"], "canales": d["max_input_channels"]})
    except Exception as e:
        return {"disponible": False, "mensaje": f"error consultando dispositivos: {e}", "instrucciones": ""}
    return {"disponible": True, "dispositivos": devs,
            "nota": "Para audio del SISTEMA elige un dispositivo de loopback (BlackHole/Loopback)."}


def _nacer(cfg, toggles=None, sim_s=6):
    """Hace NACER/renacer una vida del organismo: crea el Run, reinicia los organelos (el _despertar
    del worker restaura de disco si hay historia previa) y lo arranca. Usado por POST /start y por el
    AUTOARRANQUE al boot (incremento 2: el organismo vive aunque nadie pulse 'start')."""
    global RUN
    with RUN_LOCK:
        if RUN and not RUN.done:
            RUN.stop = True                       # detener vida previa
        RUN = Run(cfg or {}, toggles or {}, sim_s)
        HOMEO_EMERGENTE.reset(); MEMORIA.reset(); METABOLISMO.reset()   # el _despertar restaura tras esto
        RUN.start()
    return RUN


# Vida en SOLEDAD: qué percibe el organismo cuando nadie lo alimenta (incremento 2). Por defecto
# 'demo:silencio' = sigue vivo (metabolismo, memoria, dinámica endógena del campo) sin estímulo externo.
ANIMA_AUTOSTART = os.environ.get("ANIMA_AUTOSTART", "0").lower() in ("1", "true", "yes", "on")
ANIMA_FUENTE_DEFECTO = os.environ.get("ANIMA_FUENTE_DEFECTO", "demo:silencio")
# ACOPLE DE LA DÍADA: que el organismo nazca ESCUCHANDO la voz del par (oído de relación = L),
# y el mundo en el otro (R). Así A↔B se oyen entre sí 24/7 → la voz puede subir SOLA con la
# cooperación (señal costosa), y es el sustrato para que emerja comunicación.
ANIMA_ESCUCHAR_PAR = os.environ.get("ANIMA_ESCUCHAR_PAR", "0").lower() in ("1", "true", "yes", "on")
ANIMA_VOZ_PAR_MODO = os.environ.get("ANIMA_VOZ_PAR_MODO", "R2D2")


def _autoarranque_vida():
    """Si ANIMA_AUTOSTART, el organismo NACE y vive en continuo al arrancar el servidor (no espera
    a que nadie pulse 'start'). Es lo que lo vuelve un organismo-servidor 24/7, no una app.
    Si ANIMA_ESCUCHAR_PAR, nace ACOPLADO: oye la voz del par en el oído de relación (L).
    Arranca en un HILO con un RETARDO (ANIMA_AUTOSTART_DELAY): así el servidor HTTP ya está sirviendo
    y el par ya está vivo cuando intentan oírse — evita la carrera de arranque (A se quedaba mudo)."""
    if not ANIMA_AUTOSTART:
        return
    delay = float(os.environ.get("ANIMA_AUTOSTART_DELAY", "6" if ANIMA_ESCUCHAR_PAR else "0"))

    def _arrancar():
        if delay > 0:
            time.sleep(delay)            # deja que el servidor (y el par) estén listos antes de oírse
        if ANIMA_ESCUCHAR_PAR and COM_OK and COMUNICACION_PEER_URL:
            sep = "&" if "?" in COMUNICACION_PEER_URL else "?"
            url = f"{COMUNICACION_PEER_URL}{sep}modo={ANIMA_VOZ_PAR_MODO}&gain={COMUNICACION_VOICE_GAIN}"
            izq = {"tipo": "comunicacion", "url": url, "modo": ANIMA_VOZ_PAR_MODO, "nombre": "voz del par"}
            cfg = {"left_src": izq, "right_src": {"tipo": "demo", "spec": "demo:silencio"},
                   "binaural": True, "segundos": 2, "continuo": True, "criterio_duracion": "min"}
            _nacer(cfg, {}, 6)
            print(f"  AUTOARRANQUE ACOPLADO (tras {delay:.0f}s): oye la voz del par ({COMUNICACION_PEER_URL}) en el oído de relación (L)")
        else:
            cfg = {"left_src": {"tipo": "demo", "spec": ANIMA_FUENTE_DEFECTO}, "right_src": None,
                   "binaural": False, "segundos": 2, "continuo": True, "criterio_duracion": "min"}
            _nacer(cfg, {}, 6)
            print(f"  AUTOARRANQUE: el organismo nace y vive en CONTINUO · en soledad percibe '{ANIMA_FUENTE_DEFECTO}'")

    threading.Thread(target=_arrancar, daemon=True).start()


def main():
    srv = ThreadingHTTPServer((os.environ.get("ANIMA_BIND", "127.0.0.1"), PUERTO), Handler)  # 0.0.0.0 en Docker
    print("=" * 66)
    print("  CÉLULA MADRE — LABORATORIO EN VIVO")
    print(f"  → abre:  http://localhost:{PUERTO}")
    print(f"  audio en vivo: {'DISPONIBLE' if SD_OK else 'no (sounddevice ausente)'}")
    if PERSIST_OK:
        threading.Thread(target=_autosave_daemon, daemon=True).start()
        print(f"  persistencia: ON · autosave cada {AUTOSAVE_S:.0f}s · {os.environ.get('ANIMA_ESTADO_DIR','(disco por defecto)')}")
    else:
        print(f"  persistencia: OFF ({PERSIST_ERR})")
    _autoarranque_vida()
    print("  Ctrl+C para detener.")
    print("=" * 66)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  detenido."); srv.shutdown()


if __name__ == "__main__":
    main()
