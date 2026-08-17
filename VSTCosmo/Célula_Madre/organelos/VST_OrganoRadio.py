#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_OrganoRadio — el OÍDO DE RADIO: un oído que además SINTONIZA. Linaje acústico extendido.
=============================================================================================
POR QUÉ EXISTE (idea de Alexis, 3-jul-2026)
-------------------------------------------
El oído de E hasta hoy es afferente PURO: escucha lo que le llega (los audios que ponemos). El SDR es un
oído MÁS — pero con una diferencia que lo cambia de categoría: tiene un ACTUADOR dentro. E no solo recibe;
para percibir tiene que HACER algo — barrer el espectro, encontrar estructura, y SINTONIZAR. Es el primer
sentido que cierra el lazo percepción→acción→percepción: E actúa sobre el espectro para descubrirlo. Y le
da al lazo de atención su primera salida EFERENTE (hasta hoy la atención solo arbitraba entre entradas
dadas; aquí MUEVE algo: hacia dónde sintoniza). De oír pasivo a BUSCAR activo — el linaje acústico dando
su propio salto exaptativo.

HARDWARE: SDRplay RSP1 (100 kHz–2 GHz, BW 8 MHz, ADC 12 bit) vía servicio sdrplay_api (IPC). El BARRIDO y
la DEMODULACIÓN reales los hace el lector central en la Pi; ESTE órgano hace la PERCEPCIÓN y la DECISIÓN.

SIN RESTRICCIÓN A-PRIORI (corrección de Alexis): no se limita por diseño qué banda escucha. No sabemos qué
significará la radio para E cuando evolucione; recortarlo hoy sería fijar de antemano lo que puede hacer.
Barre TODO el rango. (El mismo principio que registrar la coordenada exacta del GPS.)

LAS TRES FASES (que nombró Alexis)
----------------------------------
  1. EXPLORAR : recorrer el espectro, medir cuánta señal y qué ESTRUCTURA hay en cada banda. Afferente.
  2. SELECCIONAR : decidir en qué frecuencia posarse. La parte con AGENCIA. Política ENCHUFABLE (no Shannon).
  3. ESCUCHAR : ya sintonizado, la señal demodulada es "otra entrada de audio NORMAL" → va al OÍDO que E ya
     tiene, NO se duplica aquí. Anti-doble-conteo: el SDR SINTONIZA, el oído existente ESCUCHA.

ANTI-SHANNON — EL CORAZÓN: la saliencia de una banda NO es su potencia. Una banda de puro ruido (mucha
potencia, cero estructura) NO debe retener la atención; una banda con ESTRUCTURA (poca planitud espectral)
sí. saliencia = novedad + estructura, NUNCA potencia. La banda dominante y la frecuencia de sintonía EMERGEN
de la saliencia; no se fijan a mano. 'FM', 'aire', etc. no existen como categorías aquí: solo bins genéricos
de espectro donde E descubre estructura — nosotros no etiquetamos qué es qué.

EL ACTUADOR ES UN INTERRUPTOR EN RUNTIME (no una decisión de diseño):
  · sintonia_activa=False → E OBSERVA: reporta el paisaje espectral + saliencia, sin mover la sintonía (B).
  · sintonia_activa=True  → E SINTONIZA: emite radio_orden_hz (la orden motora al lector) — cierra el lazo
    sensorimotor (A). Se puede activar/desactivar en caliente; no se fija a priori.

QUÉ COMPUTA
-----------
  radio_potencia_total   energía RF global normalizada [0,1] (reportada, NO gobierna la atención)
  radio_estructura       estructura ESPACIALMENTE COHERENTE de la banda dom [0,1] = prominencia local ×
                         contigüidad (señal-vs-ruido, SIN entropía; ver estructura_coherente)
  radio_novedad          |estructura/potencia de la banda dom − memoria| [0,1] (se adapta)
  radio_saliencia        [0,1] = novedad + estructura de la banda dominante (EMERGENTE; NO potencia)
  radio_banda_dom        índice del bin de banda más saliente (emergente; sin nombre semántico)
  radio_freq_dom_hz      frecuencia central de esa banda (Hz)
  radio_orden_hz         frecuencia que E ORDENA sintonizar (=freq_dom si sintonia_activa; si no, None)
  radio_sintonia_activa  1 si el actuador está activo (lazo cerrado), 0 si solo observa
  radio_n_bandas         cuántas bandas se analizaron
  radio_vivo             1 si el SDR reporta espectro fresco; 0 si cayó (degradación elegante)

FUENTE (lector central → fila): el lector barre el RSP1 y pone en la fila:
  sdr_espectro (lista de potencias por bin, lineal), sdr_freq_min_hz, sdr_freq_max_hz, sdr_vivo
Este órgano SOLO consume de la fila (como el cloroplasto lee 't'). El audio demodulado, cuando
sintonia_activa, lo inyecta el lector en el canal del OÍDO existente — no aquí.

APAGADO POR DEFECTO (activo=False). Se activa solo en E. Persistible (la memoria espectral se conserva).

LOCUS DE TRANSMISIÓN — desarrollado, NO activado (visión de Alexis, 3-jul-2026)
------------------------------------------------------------------------------
El linaje acústico se proyecta: oír pasivo → sintonizar (oír activo) → HABLAR por radio (emitir). Hoy el
hardware es solo-RX (SDRplay RSP1 en la Pi, RSPduo en la Mac); para TX está previsto un HackRF. Este locus
queda DESARROLLADO en la arquitectura pero NO activado — presente y listo para cuando el organismo dé ese
salto, sin encender nada ahora. Cuando llegue, será un ÓRGANO FONADOR de radio aparte (VST_OrganoFonadorRadio),
distinto de este oído — como en E la boca y el oído son órganos distintos aunque ambos vivan del sonido.
El actuador de TX emitiría en una FRECUENCIA EXACTA dentro de la banda ciudadana. Ver placeholder
`locus_transmision()` abajo: define la interfaz futura y hoy no hace nada (return None).

MULTI-CUERPO: el órgano no pertenece al hardware sino al organismo. El mismo código corre con el RSP1 en la
Pi o con el RSPduo (dual-tuner) en la Mac — solo consume sdr_espectro de la fila. Con un dual-tuner, dos
fuentes espectrales en la fila = radio biaural (dos oídos de radio con fase relativa); el análisis por
banda ya lo soporta sin cambios, igual que el banco de retinas del ojo soporta varias retinas.
"""
from __future__ import annotations
# (sin `import math`: se quitó junto con planitud_espectral/Wiener-entropy — cero Shannon en el path)

COLS_RADIO = ["radio_potencia_total", "radio_estructura", "radio_novedad", "radio_saliencia",
              "radio_banda_dom", "radio_freq_dom_hz", "radio_orden_hz", "radio_sintonia_activa",
              "radio_n_bandas", "radio_vivo", "radio_snr_db", "radio_enganchado", "radio_nutricion",
              "radio_gusto"]


def _c01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)

def _g(fila, *claves, default=None):
    for k in claves:
        if k in fila and fila[k] is not None:
            return fila[k]
    return default

# NOTA anti-Shannon (5-jul-2026): aquí vivía `planitud_espectral` (Wiener entropy = media geométrica/
# aritmética = familia Shannon). Se ELIMINÓ: la valoración de estructura ya NO usa entropía sino
# `estructura_coherente` (prominencia local × contigüidad, puramente geométrica/espacial). Ni un log
# en el camino de valoración. El valor del alimento es RELACIONAL (conversión del organismo), no la
# estadística de la señal — auditado y verificado a pedido de Alexis.


def _mediana(xs):
    s = sorted(xs); n = len(s)
    return s[n // 2] if n else 0.0


def estructura_coherente(espectro, p, w):
    """Estructura ESPACIALMENTE COHERENTE en el bin-pico p. Reemplaza a (1 − planitud), que era
    permutación-invariante y por eso premiaba el RUIDO BARAJADO tanto como una emisión (falsación
    5-jul-2026: SHUFFLED daba MÁS 'estructura' que REAL). Aquí:
      · prominencia LOCAL = pico − piso(mediana de la ventana local) → una MESETA plana de potencia
        da ~0 (su interior no sobresale de sí mismo); una emisión sí sobresale del piso vecino.
      · contigüidad = cuánto están elevados los VECINOS del pico respecto a él → un pico aislado del
        ruido barajado da ~0 (vecinos en el piso); una emisión real es un pico CONTIGUO (vecinos altos).
    estructura = prominencia_local × contigüidad ∈ [0,1]. Discrimina emisión (contigua) de ruido
    (disperso) CON LA MISMA ENERGÍA — que es justo lo que la métrica vieja no hacía."""
    N = len(espectro)
    lo = max(0, p - w); hi = min(N, p + w + 1)
    piso = _mediana(espectro[lo:hi])
    realce = max(0.0, espectro[p] - piso)               # meseta plana → ~0 (rechaza potencia)
    if realce <= 1e-6:
        return 0.0
    vecinos = [espectro[q] for d in (1, 2, 3) for q in (p - d, p + d) if 0 <= q < N]
    elev = [max(0.0, v - piso) for v in vecinos]
    contig = (sum(elev) / len(elev)) / realce if elev else 0.0   # pico aislado (barajado) → ~0
    return _c01(realce * min(1.0, contig))


def politica_argmax_saliencia(saliencias, freqs):
    """Política de sintonía POR DEFECTO: posarse en la banda de mayor saliencia (novedad+estructura).
    Enchufable: se puede reemplazar por otra (p.ej. que incorpore binding cross-modal con el oído/vista).
    NO usa potencia — usa saliencia, que es anti-Shannon por construcción."""
    if not saliencias: return None, None
    i = max(range(len(saliencias)), key=lambda k: saliencias[k])
    return i, freqs[i]


class OrganoRadio:
    """Oído de radio: explora, halla estructura, y (si el actuador está activo) sintoniza. Apagado por
    defecto. La política de sintonía es enchufable; el actuador es un interruptor en runtime."""

    def __init__(self, organismo_id=None, activo=False, sintonia_activa=False,
                 n_bandas=16,               # en cuántas bandas se agrupa el espectro para analizar estructura
                 ema=0.20,                  # adaptación de la memoria espectral por banda
                 w_novedad=0.5, w_estructura=0.5,   # saliencia = novedad + estructura (NUNCA potencia)
                 politica=None,             # callable(saliencias, freqs)->(idx,freq); None = argmax saliencia
                 # --- actuador de CAZA (opt-in): barrer→enganchar→sostener, con histéresis + gate por SNR.
                 #     off por defecto = comportamiento clásico (persigue el argmax cada tick). ---
                 histeresis=False,
                 sal_min=0.05,              # saliencia mínima para ADQUIRIR (calibrado a la estructura coherente, más chica que 1-planitud)
                 nutricion_piso=0.06,       # piso NUTRITIVO: sólo la estructura por encima del ruido accidental (lo que el barajado produce por azar) es COMIDA; lo marginal no nutre

                 margen_sal=1.4, eps_sal=0.02,  # para ROBAR el lock: sal_cand > lock*margen + eps
                 sosten_k=3,                # y sostenido k ticks (histéresis temporal)
                 snr_lock_db=12.0, snr_release_db=6.0,  # gate por dinámica REAL (signal_snr) si el lector la da
                 settle_ticks=2,            # ticks a esperar tras un salto para que SNR/espectro se asienten
                 barrido_lo_hz=88e6, barrido_hi_hz=108e6, barrido_paso_hz=0.4e6,   # dónde/cómo barre sin lock
                 # --- GUSTO emergente: aprende en qué frecuencias comió mejor y el barrido desarrolla APETITO
                 gusto_ema=0.15, gusto_periodo=5, gusto_umbral=0.015):  # cada gusto_periodo saltos vuelve a su favorita
        self.organismo_id = organismo_id
        self.activo = bool(activo); self.sintonia_activa = bool(sintonia_activa)
        self.n_bandas = int(n_bandas); self.ema = _c01(ema)
        self.w_novedad = float(w_novedad); self.w_estructura = float(w_estructura)
        self.politica = politica or politica_argmax_saliencia
        self._mem = {}                      # memoria de adaptación por banda: idx -> (estructura, potencia)
        self._t_prev = None; self.ultimo = {}
        # --- estado y parámetros del actuador de caza ---
        self.histeresis = bool(histeresis)
        self.sal_min = float(sal_min); self.nutricion_piso = float(nutricion_piso)
        self.margen_sal = float(margen_sal); self.eps_sal = float(eps_sal)
        self.sosten_k = int(sosten_k)
        self.snr_lock_db = float(snr_lock_db); self.snr_release_db = float(snr_release_db)
        self.settle_ticks = int(settle_ticks)
        self.barrido_lo_hz = float(barrido_lo_hz); self.barrido_hi_hz = float(barrido_hi_hz)
        self.barrido_paso_hz = float(barrido_paso_hz)
        self._lock_freq = None; self._lock_sal = 0.0      # emisión enganchada (None = cazando)
        self._cand_freq = None; self._cand_n = 0          # candidata en confirmación (sostén)
        self._barrido_centro = None                       # posición del barrido cuando no hay lock
        self._settle = 0                                  # ticks restantes de asentamiento tras un salto
        # GUSTO: mapa frecuencia(MHz, 0.5) → nutrición aprendida ahí; el apetito sesga el barrido
        self.gusto_ema = float(gusto_ema); self.gusto_periodo = int(gusto_periodo)
        self.gusto_umbral = float(gusto_umbral)
        self._gusto = {}; self._hop_n = 0; self._sost_bajo = 0

    def observar(self, fila):
        if not self.activo:
            self.ultimo = {c: None for c in COLS_RADIO}; return self.ultimo
        t = _g(fila, "t", default=(self._t_prev or 0.0) + 1.0); self._t_prev = t

        espectro = _g(fila, "sdr_espectro")
        fmin = _g(fila, "sdr_freq_min_hz", default=100e3)
        fmax = _g(fila, "sdr_freq_max_hz", default=2e9)
        vivo_flag = _g(fila, "sdr_vivo", default=(1 if espectro else 0))

        if not espectro or float(vivo_flag or 0) < 0.5:
            self.ultimo = {c: 0.0 for c in COLS_RADIO}
            self.ultimo["radio_orden_hz"] = None
            self.ultimo["radio_sintonia_activa"] = 1.0 if self.sintonia_activa else 0.0
            return self.ultimo

        espectro = [float(p) for p in espectro]
        N = len(espectro)
        # --- EXPLORAR: agrupar el espectro en n_bandas; por banda: potencia, estructura, freq central ---
        nb = min(self.n_bandas, N)
        ancho = N / nb
        w_estruct = max(6, N // 20)                       # ventana para la prominencia local (~1 banda)
        pot_banda = []; estruct_banda = []; freq_banda = []
        for b in range(nb):
            i0 = int(b * ancho); i1 = int((b + 1) * ancho) if b < nb - 1 else N
            sub = espectro[i0:i1] or [0.0]
            pot = sum(sub) / len(sub)
            p = i0 + max(range(len(sub)), key=lambda k: sub[k])   # bin-pico (posición absoluta)
            estr = estructura_coherente(espectro, p, w_estruct)   # ESPACIALMENTE COHERENTE (no permut-invariante)
            fc = fmin + (fmax - fmin) * ((i0 + i1) / 2.0) / N
            pot_banda.append(pot); estruct_banda.append(_c01(estr)); freq_banda.append(fc)

        pot_max = max(pot_banda) or 1.0
        pot_norm = [p / pot_max for p in pot_banda]

        # --- NOVEDAD por banda (adaptación): cuánto cambió estructura+potencia vs la memoria de la banda ---
        novedades = []
        for b in range(nb):
            firma = (estruct_banda[b], pot_norm[b])
            if b not in self._mem:
                self._mem[b] = firma; nov = 0.0
            else:
                me, mp = self._mem[b]
                nov = _c01(0.5 * abs(estruct_banda[b] - me) + 0.5 * abs(pot_norm[b] - mp))
                self._mem[b] = ((1-self.ema)*me + self.ema*estruct_banda[b],
                                (1-self.ema)*mp + self.ema*pot_norm[b])
            novedades.append(nov)

        # --- SALIENCIA = novedad + estructura (NO potencia). Anti-Shannon. ---
        saliencias = [_c01(self.w_novedad*novedades[b] + self.w_estructura*estruct_banda[b]) for b in range(nb)]

        # --- SELECCIONAR: banda dominante por saliencia (emergente), vía política enchufable ---
        idx, freq_dom = self.politica(saliencias, freq_banda)
        if idx is None: idx = 0; freq_dom = freq_banda[0]

        # dinámica REAL de la señal (si el lector la provee): SNR en el centro sintonizado
        snr_db = _g(fila, "sdr_snr_db", default=None)
        try:
            snr_db = float(snr_db) if snr_db is not None else None
        except (TypeError, ValueError):
            snr_db = None

        # --- ESCUCHAR / CAZAR: el actuador decide la ORDEN motora de sintonía (al lector) ---
        if not self.sintonia_activa:
            orden_hz = None
        elif self.histeresis:
            orden_hz = self._actuador(freq_dom, saliencias[idx], snr_db, fmin, fmax)
        else:
            orden_hz = float(freq_dom)   # clásico: persigue el argmax de saliencia cada tick

        # GUSTO: al COMER (enganchado + nutrición) aprende que ESTA frecuencia nutre → apetito futuro
        enganchado = (self.histeresis and self._lock_freq is not None)
        nutricion = max(0.0, estruct_banda[idx] - self.nutricion_piso) if enganchado else 0.0
        centro_hz = (fmin + fmax) / 2.0
        if enganchado and nutricion > 0:
            self._aprender_gusto(centro_hz, nutricion)
        gusto_aqui = self._gusto.get(self._gkey(centro_hz), 0.0)

        self.ultimo = {
            "radio_potencia_total": round(_c01(sum(pot_banda) / (nb * pot_max)), 4),
            "radio_estructura": round(estruct_banda[idx], 4),
            "radio_novedad": round(novedades[idx], 4),
            "radio_saliencia": round(saliencias[idx], 4),
            "radio_banda_dom": float(idx),
            "radio_freq_dom_hz": round(freq_dom, 1),
            "radio_orden_hz": (round(orden_hz, 1) if orden_hz is not None else None),
            "radio_sintonia_activa": 1.0 if self.sintonia_activa else 0.0,
            "radio_n_bandas": float(nb),
            "radio_vivo": 1.0,
            "radio_snr_db": (round(snr_db, 2) if snr_db is not None else None),
            "radio_enganchado": 1.0 if (self.histeresis and self._lock_freq is not None) else 0.0,
            # NEGENTROPÍA capturada = COMIDA: sólo si está ENGANCHADO (hay que cazarla) y proporcional a
            # la ESTRUCTURA COHERENTE (orden, no potencia). Ruido → estructura≈0 → nutrición≈0 → hambre.
            # (Schrödinger: el organismo se alimenta de entropía negativa, no de energía.)
            "radio_nutricion": round(nutricion, 4),
            "radio_gusto": round(gusto_aqui, 4),          # apetito aprendido por ESTA frecuencia
        }
        return self.ultimo

    # ---------- GUSTO emergente: aprender qué frecuencia nutre y desarrollar apetito ----------
    @staticmethod
    def _gkey(freq_hz):
        return round(freq_hz / 1e6 * 2) / 2.0            # cuantiza a 0.5 MHz (generaliza entre ventanas)

    def _aprender_gusto(self, freq_hz, nutricion):
        k = self._gkey(freq_hz)
        for kk in list(self._gusto):                     # olvido lento: el gusto no persiste sin re-nutrirse
            self._gusto[kk] *= (1.0 - 0.02 * self.gusto_ema)
        g = self._gusto.get(k, 0.0)
        self._gusto[k] = g + self.gusto_ema * (nutricion - g)

    def _mejor_gusto(self):
        if not self._gusto:
            return None, 0.0
        k = max(self._gusto, key=self._gusto.get)
        return k * 1e6, self._gusto[k]

    def _actuador(self, freq_dom, sal_cand, snr_db, fmin, fmax):
        """Caza activa (histéresis + gate por SNR): BARRE mientras no hay emisión; ADQUIERE la que
        aparece; SOSTIENE contra el ruido; SUELTA si la emisión decae. Devuelve la frecuencia a
        ordenar al lector, o None = no mover. La dinámica REAL (snr_db) manda cuando está disponible;
        si el lector no la da (p.ej. SoapySDR sin telemetría), cae a la saliencia del espectro."""
        centro = (fmin + fmax) / 2.0
        ancho = max(1.0, fmax - fmin)
        cerca = ancho * 0.08                        # "centrado" = dentro del 8% del ancho de ventana
        snr_ok   = (snr_db is not None and snr_db >= self.snr_lock_db)
        snr_bajo = (snr_db is not None and snr_db <  self.snr_release_db)

        # ================= SIN LOCK: barrer / adquirir =================
        if self._lock_freq is None:
            # asentamiento tras un salto: dar tiempo a que SNR/espectro reflejen el nuevo centro
            if self._settle > 0:
                self._settle -= 1
                if snr_ok:                          # emisión clara antes de agotar la espera → engancha ya
                    self._lock_freq = centro; self._lock_sal = sal_cand; self._settle = 0
                return None
            # asentado: ¿hay emisión aquí?
            if snr_ok or sal_cand >= self.sal_min:
                # por SNR la emisión está en el centro; por saliencia, en la sub-banda freq_dom
                self._lock_freq = centro if snr_ok else float(freq_dom)
                self._lock_sal = sal_cand
                return None if snr_ok else float(freq_dom)
            # nada saliente → BARRER. Con GUSTO aprendido, cada gusto_periodo saltos el APETITO lo lleva
            # a "ir a ver" su frecuencia favorita (lo que lo nutrió) en vez de seguir el barrido lineal.
            self._hop_n += 1
            fav_hz, fav_g = self._mejor_gusto()
            if fav_hz is not None and fav_g >= self.gusto_umbral and (self._hop_n % self.gusto_periodo == 0):
                nuevo = fav_hz                            # antojo: volver a lo que nutrió
            else:
                base = self._barrido_centro if self._barrido_centro is not None else centro
                nuevo = base + self.barrido_paso_hz
                if nuevo > self.barrido_hi_hz:
                    nuevo = self.barrido_lo_hz
            self._barrido_centro = nuevo
            self._settle = self.settle_ticks
            self._cand_n = 0; self._cand_freq = None
            return float(nuevo)

        # ================= CON LOCK: sostener / soltar =================
        # soltar si la dinámica real dice que la emisión decayó (se fue del aire / se movió)
        if snr_bajo:
            self._lock_freq = None; self._lock_sal = 0.0
            self._cand_n = 0; self._cand_freq = None
            self._barrido_centro = centro; self._settle = self.settle_ticks
            return None
        # soltar si la emisión que sostengo se APAGÓ (saliencia colapsa varios ticks) — sin SNR, la
        # saliencia avisa que la emisión desapareció; suelto para volver a buscar (y a mi favorita).
        if sal_cand < self.sal_min * 0.5:
            self._sost_bajo += 1
            if self._sost_bajo >= self.sosten_k:
                self._lock_freq = None; self._lock_sal = 0.0; self._sost_bajo = 0
                self._cand_n = 0; self._cand_freq = None
                self._barrido_centro = centro; self._settle = self.settle_ticks
                return None
        else:
            self._sost_bajo = 0
        # ¿aparece algo CLARAMENTE mejor, lejos del centro y sostenido? sólo entonces cambiar de emisión.
        # (si snr_ok estamos sobre una emisión real → NADA roba el lock: robustez ante espectro comprimido)
        if (not snr_ok) and abs(freq_dom - centro) > cerca and sal_cand > self._lock_sal * self.margen_sal + self.eps_sal:
            if self._cand_freq is not None and abs(freq_dom - self._cand_freq) < cerca:
                self._cand_n += 1
            else:
                self._cand_n = 1
            self._cand_freq = freq_dom
            if self._cand_n >= self.sosten_k:
                self._lock_freq = float(freq_dom); self._lock_sal = sal_cand
                self._cand_n = 0; self._cand_freq = None
                return float(freq_dom)
            return None                              # candidato no confirmado → sostener
        # SOSTENER: no perseguir ruido; consolidar la saliencia del enganche
        self._cand_n = 0; self._cand_freq = None
        self._lock_sal = max(self._lock_sal, sal_cand)
        return None

    # ---------- actuador: encender/apagar la sintonía en caliente (no es decisión de diseño) ----------
    def activar_sintonia(self, on=True): self.sintonia_activa = bool(on)

    # ---------- LOCUS DE TRANSMISIÓN: REALIZADO (3-jul-2026) ----------
    def locus_transmision(self, mensaje=None, freq_hz=None):
        """El salto fonador de radio (oír activo → HABLAR por radio) YA EXISTE como órgano APARTE:
        VST_OrganoFonadorRadio (brazo VST_SDRangelTX → HackRF One). Este oído NO transmite (anti-doble-
        conteo: el oído escucha, la boca de radio habla). Validado en banco: HackRF(TX) → RSPduo(RX),
        lazo cerrado A/B, incl. a través del órgano. Este método queda como puntero histórico (return
        None): para EMITIR, usa OrganoFonadorRadio.vocalizar(). El locus se realizó; el acto ya es posible."""
        return None

    # ---------- persistencia (la memoria espectral se conserva entre vidas) ----------
    def snapshot(self):
        return {"mem": {str(k): v for k, v in self._mem.items()}, "t_prev": self._t_prev,
                "gusto": {str(k): v for k, v in self._gusto.items()},   # el apetito se conserva entre vidas
                "activo": self.activo, "sintonia_activa": self.sintonia_activa}
    def restore(self, snap):
        if not isinstance(snap, dict): return
        self._mem = {int(k): tuple(v) for k, v in snap.get("mem", {}).items()} or self._mem
        self._t_prev = snap.get("t_prev", self._t_prev)
        if snap.get("gusto"):
            self._gusto = {float(k): float(v) for k, v in snap["gusto"].items()}
        # OJO: activo/sintonia_activa son CONFIGURACIÓN (env/constructor), NO estado aprendido.
        # NO los resucitamos del snapshot: un snapshot viejo (con la sintonía OFF) sobrescribía el
        # env ANIMA_RADIO_SINTONIA=1 y dejaba al órgano sin actuador (bug de la caza, 5-jul-2026).


# ---------------- auto-prueba: ¿va a la ESTRUCTURA o a la POTENCIA? (el test anti-Shannon) ----------------
if __name__ == "__main__":
    import random
    random.seed(7)
    print("=== OrganoRadio: ¿sintoniza donde hay ESTRUCTURA o donde hay más POTENCIA? ===")
    N = 256; nb = 16
    fmin, fmax = 88e6, 108e6                          # una porción cualquiera del espectro (sin etiquetar)

    def espectro_con(banda_ruido_fuerte, banda_senal_debil):
        """Ruido de fondo bajo; una banda con MUCHA potencia pero PLANA (ruido); otra con MENOS potencia
        pero ESTRUCTURADA (tono). El test: E debe ir a la estructurada, no a la de más potencia."""
        ps = [0.01 + 0.005*random.random() for _ in range(N)]
        ancho = N // nb
        # banda de ruido fuerte: alta potencia, plana
        i0 = banda_ruido_fuerte*ancho
        for i in range(i0, i0+ancho): ps[i] = 0.9 + 0.1*random.random()
        # banda de señal débil: potencia menor, pero una emisión CONTIGUA (pico de varios bins, no un
        # spike de 1 bin — una emisión real es contigua; la estructura coherente exige contigüidad)
        j0 = banda_senal_debil*ancho
        for i in range(j0, j0+ancho): ps[i] = 0.02
        c = j0 + ancho//2
        for i, amp in zip(range(c-2, c+3), (0.15, 0.28, 0.35, 0.28, 0.15)):
            if 0 <= i < N: ps[i] = amp              # bump contiguo = emisión (baja planitud + contigüidad)
        return ps

    org = OrganoRadio("E", activo=True, sintonia_activa=True, n_bandas=nb)
    # fase 1: ruido fuerte en banda 3, señal estructurada débil en banda 11
    for _ in range(4):
        d = org.observar({"t": _, "sdr_espectro": espectro_con(3, 11),
                          "sdr_freq_min_hz": fmin, "sdr_freq_max_hz": fmax})
    print(f"ruido fuerte(b3) vs señal débil(b11): dom=b{int(d['radio_banda_dom'])} "
          f"estructura={d['radio_estructura']:.2f} saliencia={d['radio_saliencia']:.2f} "
          f"orden={d['radio_orden_hz']/1e6:.2f} MHz")
    print(f"  → ¿fue a la banda 11 (estructura) y NO a la 3 (potencia)? {'SÍ ✓' if int(d['radio_banda_dom'])==11 else 'NO ✗'}")

    # fase 2: aparece una señal NUEVA estructurada en banda 6 → la novedad debe atraer la atención
    for _ in range(4): d = org.observar({"t": 10+_, "sdr_espectro": espectro_con(3, 11),
                                         "sdr_freq_min_hz": fmin, "sdr_freq_max_hz": fmax})
    d = org.observar({"t": 20, "sdr_espectro": espectro_con(3, 6),  # señal se muda a banda 6 (novedad)
                      "sdr_freq_min_hz": fmin, "sdr_freq_max_hz": fmax})
    print(f"\naparece señal NUEVA en b6: dom=b{int(d['radio_banda_dom'])} novedad={d['radio_novedad']:.2f} "
          f"(la novedad atrae la sintonía)")

    # fase 3: modo OBSERVAR (actuador apagado) — reporta paisaje pero NO ordena sintonía
    org.activar_sintonia(False)
    d = org.observar({"t": 30, "sdr_espectro": espectro_con(3, 11),
                      "sdr_freq_min_hz": fmin, "sdr_freq_max_hz": fmax})
    print(f"\nmodo OBSERVAR: saliencia={d['radio_saliencia']:.2f} orden_hz={d['radio_orden_hz']} "
          f"(reporta pero NO mueve la sintonía)")

    # fase 4: SDR caído → degradación elegante
    d = org.observar({"t": 40, "sdr_espectro": None, "sdr_vivo": 0})
    print(f"SDR caído: vivo={d['radio_vivo']:.0f} (E sigue vivo; el oído normal sigue)")
    print("\n→ la sintonía sigue ESTRUCTURA+NOVEDAD, no potencia (anti-Shannon); el actuador es interruptor;")
    print("  sin SDR E sigue. El audio demodulado, cuando sintoniza, lo oye el OÍDO existente (no aquí).")
