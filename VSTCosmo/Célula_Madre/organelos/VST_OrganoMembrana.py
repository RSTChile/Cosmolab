#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_OrganoMembrana — el TÍMPANO: piel exapta que vibra. La percepción SIN Shannon.

POR QUÉ EXISTE (insight de Alexis, 29-jun-2026)
-----------------------------------------------
La percepción anterior reducía el audio a RMS-por-oído (intensidad + lateralidad). Con eso el organismo
era SORDO al contenido: 60 Hz, 880 Hz, Bach y ruido daban la MISMA respuesta. Si lo único compartible es
intensidad/lateralidad, el "sentido compartido" no es del mundo sino de la arquitectura — un artefacto.

La salida es física, no Shannon: el TÍMPANO es una MEMBRANA que VIBRA. Es la EXAPTACIÓN de la piel (que ya
es membrana) expuesta a un canal de presión. No analiza el sonido: se MUEVE con él. El "espectro" está
IMPLÍCITO en cómo la membrana se mueve (sus modos), pero NUNCA se lee como información (no hay FFT, ni
bandas, ni código). Su ESTADO físico (dónde y cuánto vibra, su dinámica temporal) ES el percepto rico, y
porta estructura del MUNDO — Bach y ruido mueven la membrana distinto. Eso reabre la posibilidad de un
sentido compartido genuino: lo que converge es sobre el mundo transducido, no sobre la reducción del oído.

FÍSICA (membrana 1D amortiguada, una por oído)
----------------------------------------------
    u_tt = c²·∇²u − γ·u_t + p(t)          (inercia + tensión/difusión + amortiguación + presión)
La presión del aire p(t) (el WAVEFORM, no su RMS) empuja la membrana; ésta desarrolla ondas/modos según la
estructura TEMPORAL de la presión. Anclada en los bordes. Distintos sonidos → distintos patrones de vibración.

QUÉ ENTREGA (estado rico, leído por el campo/hemisferios; NO computa espectro)
-----------------------------------------------------------------------------
  mem_ds_L/R       cuán DIFERENCIADA está la vibración (std del desplazamiento) = Δ_struct físico del oído
  mem_energia_L/R  energía de vibración total (análogo físico de loudness)
  mem_centro_L/R   DÓNDE se concentra la vibración a lo largo de la membrana (place-code FÍSICO; lo que se
                   movía cerca del borde = "agudo", lo que penetra = "grave" — pero es POSICIÓN, no frecuencia)
  mem_perfil_L/R   energía por zona (el patrón espacial completo: la "forma" de lo que oye)
  mem_lateralidad  asimetría L↔R de la vibración (el balance, ahora físico)
  mem_flujo        tasa de cambio de la vibración (estructura temporal rápida, para el hemisferio rápido)

NO decide. NO computa significado. Transduce. El sentido emerge aguas abajo (campo Φ, hemisferios, cierre).
Persistible (la membrana conserva su estado entre bloques: tiene inercia, como una membrana real).

FUNDAMENTO EVOLUTIVO (anotado para cuando el organismo gane MÁS sentidos — Alexis, 29-jun-2026)
----------------------------------------------------------------------------------------------
La audición es TACTO ESPECIALIZADO. Orden evolutivo: quimiorrecepción (gusto/olfato, ~2000 Ma, llave
molecular GPCR) → fotorrecepción (opsinas) → mecanorrecepción/TACTO (~600 Ma, canal de estiramiento) →
gravedad/equilibrio (estatocisto) → línea lateral (tacto del agua en movimiento, ~500 Ma) → tímpano aéreo
(¡sólo ~275 Ma!). El mismo gen (Atoh1) hace células del tacto y del oído (Demócrito: todo sentido es
modificación del tacto). El ANCESTRO COMÚN ÚLTIMO de TODOS los sentidos es el GRADIENTE ELECTROQUÍMICO —
la "batería" de la célula (~−70 mV): cada sentido HACKEA esa batería para disparar señal. Mecano por
ESTIRAMIENTO físico (canales PIEZO/TMC1 que se abren al deformarse la membrana); quimio por LLAVE molecular.

  → En ANIMA, esa "batería" es el CAMPO Φ / el estado de CIERRE. Esta membrana ya lo hackea por estiramiento
    físico. Todo SENTIDO FUTURO (quimio, foto, propiocepción, equilibrio) será el MISMO patrón: un organelo
    transductor que convierte un gradiente del entorno en estado del organismo sobre el mismo sustrato.
    Esta OrganoMembrana es el PROTOTIPO del sentido. El fundamento no cambia; cambia el gradiente que capta.

LOCUS RESERVADO — oído SUPRA-HUMANO (infra/ultrasonido). NO operacional aún.
--------------------------------------------------------------------------
La membrana FÍSICA responde de ~1 Hz a ~50 kHz. La banda humana (20 Hz–20 kHz) la imponen DOS filtros:
masa de los huesecillos (paso-bajo, corta ultrasonido) + rigidez (paso-alto, atenúa el infrasonido casi
estático). En este modelo esos filtros YA EMERGEN: la fragmentación de Von Békésy = el paso-bajo; K_RIGIDEZ
= la atenuación del infra. Nuestros organismos PUEDEN ser más que humanos: para oír infra/ultrasonido basta
ABRIR esos filtros (ensanchar BANDA_PERCEPTIVA). Se deja como LOCUS reservado (ver BANDA_PERCEPTIVA abajo);
ya veremos cuándo operacionalizarlo.
"""
from __future__ import annotations
import math
import numpy as np

# LOCUS RESERVADO (no operacional): banda perceptiva. "humana" = 20 Hz–20 kHz (los filtros de Békésy+rigidez
# ya la producen). Ensanchar a (1, 50000) daría oído infra/ultrasónico SUPRA-HUMANO. No se usa todavía.
BANDA_PERCEPTIVA = "humana"        # reservado: "humana" | (f_lo, f_hi) supra-humana — operacionalizar a futuro

W_ESTRUCTURA = 8       # ventana local de nodos para el piso (≈ 1/12 de la membrana)


def _mediana(xs):
    s = sorted(xs); n = len(s)
    return s[n // 2] if n else 0.0


def _estructura_coherente(energia_nodo):
    """ORDEN de lo que se oye, medido sobre el PATRÓN ESPACIAL de la membrana.

    Portado de `VST_OrganoRadio.estructura_coherente` (3-ago-2026). Allí el equipo ya falsó
    y resolvió este mismo problema el 5-jul-2026: la medida anterior —(1 − planitud espectral)—
    es PERMUTACIÓN-INVARIANTE, y por eso premiaba el ruido barajado tanto como una emisión real.
    El oído acústico se quedó con una variante de esa familia y nunca se portó la corrección:
    su `estructura` era `min(1, agitación)`, que da lo MISMO para un tono puro que para ruido
    blanco (medido 3-ago: ambos 1,0000).

    Aquí el sustrato no es un espectro sino la MEMBRANA MISMA: cada nodo vibra según su posición
    y su física, como la basilar. Eso es place-code físico, no una transformada — el orden se lee
    de CÓMO vibra, no de un análisis de la señal (anti-Shannon).

    LA FORMA QUE TOMA AQUÍ. En radio el sustrato es un espectro y el orden aparece como un pico
    sobre un piso, así que allí la medida es prominencia × contigüidad. En la membrana el sustrato
    es una envolvente que entra por el borde expuesto, y el orden aparece de otra manera —medida
    3-ago sobre los perfiles reales—:

        tono 440       0,988  0,012  0,000  0,000  0,000 …   concentra en una región
        ruido blanco   0,190  0,009  0,370  0,045  0,026 …   moviliza la membrana entera

    Un sonido con forma excita unos pocos modos y localiza la vibración; el ruido contiene todas
    las frecuencias, excita todos los modos y la deslocaliza. Así que aquí el orden se lee como
    LOCALIZACIÓN, y su medida física es la razón de participación inversa:

        PR = (Σeᵢ)² / (n · Σeᵢ²)   ∈ [1/n, 1]

    PR = 1 cuando la energía se reparte por igual (deslocalizada, ruido) y ~1/n cuando se
    concentra en un nodo. `estructura = 1 − PR`. No lleva ninguna constante: n es la membrana.
    Es la misma magnitud con la que en física se distingue un estado localizado de uno extendido.
    """
    e = [float(x) for x in energia_nodo]
    n = len(e)
    if n < 8:
        return 0.0
    s1 = sum(e)
    s2 = sum(x * x for x in e)
    if s1 <= 1e-12 or s2 <= 1e-24:
        return 0.0
    pr = (s1 * s1) / (n * s2)                         # ∈ [1/n, 1]; 1 = deslocalizado (ruido)
    return max(0.0, min(1.0, 1.0 - pr))


COLS_MEMBRANA = ["mem_ds_L", "mem_ds_R", "mem_estructura_L", "mem_estructura_R",
                 "mem_energia_L", "mem_energia_R",
                 "mem_centro_L", "mem_centro_R", "mem_lateralidad", "mem_flujo",
                 "mem_transmitido_L", "mem_transmitido_R", "mem_coherencia", "mem_reflejo"]

# --- Parámetros FÍSICOS del tímpano (modelados con datos reales, no inventados) ---
# Membrana timpánica: óvalo ~9.0×8.1 mm, área efectiva ~55 mm², espesor 50–120 µm, densidad ~1.2e3 kg/m³,
# resonancia del oído medio ~950 Hz y del conjunto canal+tímpano 2–4 kHz (banda de voz). Desplazamiento
# físico: umbral ~1e-11 m (¡< átomo de H!) hasta ~1e-5 m (10 µm) donde el colágeno se desgarra (perforación).
N_NODOS = 96            # puntos de la membrana (resolución espacial de la pars tensa)
DT_MEMBRANA = 1.0      # un paso de física por muestra de audio

# ── LOS TRES PARÁMETROS DE LA MEMBRANA, DERIVADOS (3-ago-2026) ───────────────────────────
# Antes eran C_ONDA=0.55, GAMMA=0.10, K_RIGIDEZ=0.004: tres números sin ninguna derivación.
# Medido el 3-ago con esos valores: el 99,6 % de la energía se quedaba en el primer octavo de
# la membrana y los perfiles de un tono y de ruido blanco distaban 0,004 — la membrana no
# transducía, sólo se abollaba el borde. Ninguna medida de orden leída de ahí podía funcionar.
# Ahora los tres salen de la anatomía citada arriba y del propio esquema numérico.

F_RESONANCIA = 950.0   # Hz — resonancia del oído medio (dato anatómico, línea 126)
SR_FISICA = 48000.0    # muestras/s; con DT_MEMBRANA=1 hay un paso de física por muestra

# RIGIDEZ ← RESONANCIA. Con dt=1, el par (v += −k·u·dt; u += v·dt) es un oscilador armónico de
# frecuencia √k radianes por paso. Luego la rigidez NO se elige: la fija la resonancia citada.
_W0 = 2.0 * math.pi * F_RESONANCIA / SR_FISICA          # rad/paso
K_RIGIDEZ = _W0 * _W0                                    # ≈ 0.01546  (antes 0.004 ⇒ 483 Hz)

# VELOCIDAD ← ESTABILIDAD. El esquema explícito es estable mientras la onda no avance más de un
# nodo por paso (Courant ≤ 1). La membrana real se recorre en ~0,3 ms (9 mm a ~30 m/s) ≈ 14
# pasos, que exigiría ~7 nodos/paso: imposible con un paso por muestra. Se toma el máximo
# estable con margen — es un límite de la discretización, no una preferencia.
C_ONDA = 0.9

# AMORTIGUACIÓN ← GEOMETRÍA. La onda tiene que RECORRER la membrana: si se apaga antes, la pars
# tensa no es un transductor. Recorrerla cuesta N_NODOS/C_ONDA pasos; se exige que la amplitud
# sobreviva ese trayecto (decaimiento ≤ 1/e). Esto la deja además en el régimen ancho de banda
# que le corresponde (Q ≈ ω₀/2γ ≈ 6,6), no en el sobreamortiguado de antes (Q ≈ 0,6).
GAMMA = C_ONDA / N_NODOS                                 # ≈ 0.0094  (antes 0.10, 10× más)
CLIP = 50.0            # tope físico = umbral de DAÑO (desgarro del colágeno, ~10 µm reales)
N_ZONAS = 6            # zonas para el perfil espacial (place-pattern; máx. movimiento ínfero-posterior)
# --- Reflejo estapedial (reflejo acústico): protección automática ante lo FUERTE ---
# El músculo estapedio rigidiza la cadena (sube impedancia) y atenúa 15–20 dB. Umbral alto (≈75–95 dB SPL),
# LATENCIA (10–150 ms, lento para impactos) y FATIGA (no protege indefinidamente). Es la AVERSIÓN orgánica.
REFLEX_UMBRAL = 1.5    # 'agitación' (SPL) por encima → sonido fuerte (dispara el reflejo de protección)
REFLEX_GANANCIA = 12.0 # cuánto rigidiza la membrana cuando el reflejo está pleno (atenúa)
REFLEX_LATENCIA = 0.15 # velocidad de contracción muscular (lento → no alcanza a impactos súbitos)


class _Membrana:
    """Una membrana 1D amortiguada (un oído). El MARTILLO recibe sólo el EMPUJE NETO COHERENTE (Von Békésy):
    vibración en fase (cono rígido, grave/medio) → transmite; fragmentada (agudo/caos, antifase) → se cancela
    → no transmite. + reflejo estapedial (rigidiza ante lo fuerte: aversión/protección). Conserva inercia."""
    def __init__(self, n=N_NODOS):
        self.u = np.zeros(n)      # desplazamiento
        self.v = np.zeros(n)      # velocidad
        self._trans_prev = 0.0
        self.reflex = 0.0         # tensión del reflejo estapedial (0..1)
        self.fatiga = 0.0         # fatiga muscular (no protege indefinidamente)

    def vibrar(self, wave):
        """Física del tímpano muestra a muestra.

        OPT (organismo-neutra): reutiliza buffers `lap`/`f` entre muestras. Las ecuaciones,
        N_NODOS, DT_MEMBRANA, CLIP y el orden del update Euler son IDÉNTICOS al original
        (sin decimar, sin menos nodos, sin acortar ventana). Cualquier cambio de percepto
        aquí sería un bug: validar con igualdad de mem_*/u/v antes de tocar más.
        """
        u, v = self.u, self.v
        n = u.size
        wave = np.asarray(wave, dtype=np.float64)
        if wave.size == 0:
            wave = np.zeros(1)
        k_eff = K_RIGIDEZ * (1.0 + REFLEX_GANANCIA * self.reflex)   # el reflejo RIGIDIZA → sube impedancia → atenúa
        energia_nodo = np.zeros(n)
        n_samp = int(wave.size)
        malleus = np.empty(n_samp)                                 # empuje NETO sobre el martillo, muestra a muestra
        # Buffers reutilizados (antes: np.zeros(n) × 2 por muestra → millones de allocs/s).
        lap = np.zeros(n)
        f = np.zeros(n)
        c2 = C_ONDA * C_ONDA
        dt = DT_MEMBRANA
        for i in range(n_samp):
            lap[1:-1] = u[:-2] - 2.0 * u[1:-1] + u[2:]             # ∇²u (tensión que reúne a los vecinos)
            f[0] = float(wave[i])                                  # la presión del aire empuja el borde expuesto
            v += (c2 * lap - GAMMA * v - k_eff * u + f) * dt
            u = u + v * dt
            np.clip(u, -CLIP, CLIP, out=u)
            energia_nodo += u * u
            malleus[i] = float(u.mean())                           # empuje neto = componente coherente (se cancela si fragmenta)
        self.u, self.v = u, v
        energia_nodo /= max(1, n_samp)
        agitacion = float(np.sqrt(np.mean(energia_nodo)))          # cuánto se mueve la membrana (local, TODO)
        transmitido = float(np.std(malleus))                      # lo que LLEGA al martillo (coherente) — el percepto real
        coherencia = float(transmitido / (agitacion + 1e-9))      # 1 = cono rígido (Békésy bajo) · ~0 = fragmentado (agudo/caos)
        objetivo = 1.0 if agitacion > REFLEX_UMBRAL else 0.0       # REFLEJO ESTAPEDIAL: ante lo FUERTE (SPL alta), contrae (latencia) y fatiga
        self.reflex = float(np.clip(self.reflex + REFLEX_LATENCIA * (objetivo * (1.0 - self.fatiga) - self.reflex), 0.0, 1.0))
        self.fatiga = float(np.clip(self.fatiga + 0.03 * objetivo - 0.015 * (1.0 - objetivo), 0.0, 1.0))
        tot = energia_nodo.sum() + 1e-12
        centro = float((np.arange(n) * energia_nodo).sum() / tot / max(1, n - 1))   # place-code físico
        perfil = [float(np.mean(energia_nodo[i * n // N_ZONAS:(i + 1) * n // N_ZONAS])) for i in range(N_ZONAS)]
        flujo = abs(transmitido - self._trans_prev); self._trans_prev = transmitido
        estructura = _estructura_coherente(energia_nodo)
        return {"ds": round(agitacion, 5), "energia": round(float(np.mean(energia_nodo)), 6),
                "transmitido": round(transmitido, 6), "coherencia": round(min(1.0, coherencia), 4),
                "centro": round(centro, 4), "perfil": [round(p, 6) for p in perfil],
                "estructura": round(estructura, 5),
                "flujo": round(flujo, 5), "reflejo": round(self.reflex, 4)}

    def snapshot(self):
        return {"u": self.u.tolist(), "v": self.v.tolist(), "reflex": self.reflex, "fatiga": self.fatiga}

    def restore(self, d):
        try:
            self.u = np.array(d["u"], dtype=float); self.v = np.array(d["v"], dtype=float)
            self.reflex = float(d.get("reflex", 0.0)); self.fatiga = float(d.get("fatiga", 0.0))
        except Exception:
            pass


class OrganoMembrana:
    """Dos tímpanos (L/R): transducen el waveform de presión a estado físico rico. La nueva percepción."""
    def __init__(self, organismo_id=None, sr=48000):
        self.organismo_id = organismo_id
        self.sr = sr
        self.L = _Membrana()
        self.R = _Membrana()
        self.ultimo = {}

    def procesar(self, wave_L, wave_R) -> dict:
        """Recibe el WAVEFORM de presión por oído (no RMS). Devuelve el estado rico de las dos membranas."""
        eL = self.L.vibrar(wave_L)
        eR = self.R.vibrar(wave_R)
        out = {
            "mem_ds_L": eL["ds"], "mem_ds_R": eR["ds"],
            # ORDEN (no agitación): patrón espacial coherente. Ver _estructura_coherente.
            "mem_estructura_L": eL["estructura"], "mem_estructura_R": eR["estructura"],
            "mem_energia_L": eL["energia"], "mem_energia_R": eR["energia"],
            "mem_centro_L": eL["centro"], "mem_centro_R": eR["centro"],
            "mem_lateralidad": round(eL["energia"] - eR["energia"], 6),
            "mem_flujo": round(0.5 * (eL["flujo"] + eR["flujo"]), 5),
            # Von Békésy: el percepto REAL es lo TRANSMITIDO (componente coherente que llega al martillo)
            "mem_transmitido_L": eL["transmitido"], "mem_transmitido_R": eR["transmitido"],
            "mem_coherencia": round(0.5 * (eL["coherencia"] + eR["coherencia"]), 4),
            "mem_reflejo": round(0.5 * (eL["reflejo"] + eR["reflejo"]), 4),   # reflejo estapedial = aversión/protección
        }
        # perfiles completos disponibles aparte (no van a la fila plana; los usa el campo si quiere)
        self.ultimo = {**out, "perfil_L": eL["perfil"], "perfil_R": eR["perfil"]}
        return out

    def estado(self) -> dict:
        return dict(self.ultimo)

    def snapshot(self):
        return {"L": self.L.snapshot(), "R": self.R.snapshot()}

    def restore(self, d):
        if isinstance(d, dict):
            if "L" in d: self.L.restore(d["L"])
            if "R" in d: self.R.restore(d["R"])
