#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_ALTERIDAD — Órgano de ALTERIDAD / INTENCIÓN COMUNICATIVA (a DEMOSTRAR, no diseñar)
================================================================================
QUIÉN SOY
  Hoy A y B vocalizan su estado (estado→voz). Eso es EXPRESIÓN fisiológica, no lenguaje.
  Yo NO impongo lenguaje. NO tengo diccionario, gramática, ni "P17=comida", ni reward externo.
  Sólo APRENDO una regularidad por su HISTORIA DE CONSECUENCIAS:

      emito un patrón P  →  ¿el OTRO cambia?  →  ¿ese cambio me BENEFICIA?  →  ¿conviene repetir P?

  Un patrón sólo adquiere valor por sus consecuencias (anti-Shannon). El sistema lo nombra
  internamente (P = la voz emitida: 'chat','worried',…), nunca lo interpreta como palabra humana.

  Esta primera versión MIDE (no sesga la conducta): computa si el organismo DESCUBRE que puede
  afectar al otro (alt_intencion_comunicativa). El sesgo de emisión aprendido es un hook OPCIONAL
  (sesgar_emision, OFF por defecto) para no meter Shannon mientras medimos la emergencia.

  Boorman ↔ aquí: en el altruismo, mis acciones modifican el futuro del otro. El lenguaje hace lo
  mismo: no transmite por transmitir, modifica el comportamiento futuro del otro = cooperación diferida.
================================================================================
"""
from __future__ import annotations
import os, math, hashlib
from collections import deque
import numpy as np

# El patrón «comparar contra lo habitual del propio organismo» se IMPORTA, no se reescribe (escala.py).
#   rel(x, escala) -> 0,5 cuando x es lo de siempre para este organismo. Sin parámetro libre.
# `escala` vive en celula_madre/; esto permite importar el organelo suelto (pruebas y smokes)
# además de dentro del organismo. Unificado el 5-ago-2026: la revisión encontró CUATRO
# variantes del mismo arranque, que es el problema contra el que existe el módulo compartido.
import os as _os, sys as _sys
_RAIZ_CM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _RAIZ_CM not in _sys.path:
    _sys.path.insert(0, _RAIZ_CM)
from escala import Escala, rel as _rel, NEUTRO

COLS_ALT = [
    "alt_otro_presente", "alt_modelo_otro", "alt_prediccion_respuesta", "alt_error_prediccion",
    "alt_efecto_sobre_otro", "alt_efecto_sobre_mi", "alt_valor_emision", "alt_intencion_comunicativa",
    "alt_patron_emitido", "alt_patron_repetido", "alt_confianza_relacional",
    "alt_contacto_presencia", "alt_contacto_recuperado", "alt_turno_detectado",
    # AGENCIA del otro (reconocimiento de SUJETO, O-N3.4): separa presencia de causalidad.
    "alt_efecto_basal", "alt_contingencia_social", "alt_agencia_otro",
    # LIBERTAD EXPRESIVA (balbuceo): gesto vocal explorado (parámetros ACÚSTICOS, no etiquetas)
    "g_freq", "g_intensidad", "g_pausa", "g_repeticion", "g_bucket",
]


def _seed(s):
    return int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16)


def _num(v, d=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else d
    except Exception:
        return d


class OrganeloAlteridad:
    """Aprende (emisión propia → respuesta del otro → efecto sobre mí) por consecuencias.
    NO decide significados. Mide la INTENCIÓN COMUNICATIVA emergente. Simétrico en A y B."""

    def __init__(self, organismo_id: str, ventana: float = 1.0, lr: float = 0.06, ema: float = 0.03):
        self.id = str(organismo_id)
        self.ventana = float(ventana)     # s para medir la respuesta del otro tras una emisión
        self.lr = float(lr)               # aprendizaje del valor de emisión / modelo del otro
        self.ema = float(ema)             # suavizado de los escalares
        self.pendientes = deque(maxlen=4000)   # emisiones esperando que pase su ventana
        self.valor = {}                   # (P, ctx) -> valor de emisión (beneficio aprendido)
        self.modelo_otro = {}             # P -> efecto esperado sobre el otro (EMA) = "modelo del otro"
        self.n_emis = {}                  # (P, ctx) -> conteo (para 'patrón repetido')
        self.intencion = 0.0              # alt_intencion_comunicativa (EMA del [otro respondió ∧ me ayudó])
        self.efecto_otro_ema = 0.0
        self.efecto_mi_ema = 0.0
        self.error_pred_ema = 0.0
        # LÍNEA-BASE DE CONTINGENCIA (reconocimiento del otro como SUJETO, no sólo presencia):
        # baseline = cuánto cambia el otro POR SU CUENTA, en ventanas SIN emisión mía. La contingencia
        # social = cuánto MÁS cambia el otro tras mi emisión que ese basal. La agencia = qué fracción del
        # cambio del otro depende de MI acto (no de compartir ambiente). Anti-confound: presencia ≠ agencia.
        # Lo HABITUAL que cambia el otro, aprendido de la propia experiencia. Sustituye al umbral fijo
        # `efecto_otro > 0.05` que decidía si "el otro respondió" (ver el comentario largo en observar()).
        self.esc_efecto = Escala()
        self.otro_hist = deque(maxlen=512)   # (t, resumen_otro) para medir el cambio espontáneo del otro
        self.ult_emis_t = -1e9               # t de la última emisión (para excluir ventanas con emisión del basal)
        self.baseline_ema = 0.0              # cambio del otro JUSTO ANTES de emitir (línea-base pre-emisión)
        self.contingencia = 0.0              # exceso de cambio atribuible a mi acto (mediaPost − mediaPre)
        self.agencia = 0.0                   # fracción del cambio del otro que depende de mí (0..1)
        self._P_prev = None
        self._presente_prev = 1.0
        self._llamando = None             # (t, P) de una emisión hecha con el otro AUSENTE (un "¿sigues ahí?")
        self.contacto_recuperado = 0.0    # pulso cuando el otro vuelve tras una llamada
        self.eventos = []                 # bitácora a drenar por WebLive

        # --- LIBERTAD EXPRESIVA (balbuceo) -------------------------------------------------
        # La voz deja de ser estado→patrón FIJO y pasa a estado+exploración→patrón. El organismo
        # explora pequeñas variaciones ACÚSTICAS espontáneas (NO elige etiquetas, NO hay diccionario).
        # Espacio explorado = [frecuencia, intensidad, pausa, repetición], pequeño/continuo/reversible.
        # El 'patrón' que el órgano aprende por consecuencias pasa a ser el GESTO (bucket acústico),
        # no la etiqueta de afecto. Anti-Shannon: el significado, si emerge, será sólo de la historia.
        self.libertad = os.environ.get("ANIMA_LIBERTAD_EXPRESIVA", "1") not in ("0", "false", "no", "off")
        self.explora = _num(os.environ.get("ANIMA_BABBLING_EXPLORA", "0.10"), 0.10)   # magnitud del paso espontáneo
        self.atraccion = _num(os.environ.get("ANIMA_BABBLING_ATRACCION", "0.05"), 0.05)  # leve tirón a gestos útiles
        self._rng = np.random.RandomState(_seed(organismo_id))   # exploración espontánea, distinta por organismo (arbitrariedad)
        self._gesto = np.zeros(4, dtype=float)   # gesto vocal actual (neutro = voz fisiológica pura)
        self._P_gesto = "fisio"                  # firma del gesto actual (lo que el órgano aprende)

    # --------------------------------------------------------- helpers
    @staticmethod
    def _ctx(fila):
        """Contexto GRUESO (no semántico): nivel de necesidad × nivel de OI. Para agrupar emisiones."""
        return (int(round(_num(fila.get("necesidad")) * 2)), int(round(_num(fila.get("OI")) * 3)))

    @staticmethod
    def _resumen_otro(otro):
        """El estado del par puede venir PLANO (fila) o anidado en ['fila'] (estado de comunicación)."""
        otro = otro or {}
        f = otro.get("fila") if isinstance(otro.get("fila"), dict) else otro
        return {
            "OI": _num(f.get("OI")),
            "nec": _num(f.get("necesidad", f.get("necesidad_efectiva"))),
            "orient": _num(f.get("act_orientacion_deg", f.get("orientacion_deg"))),
            "voz": f.get("voz_emitida"),
            "vivo": bool(otro.get("ok", otro.get("vivo", True))),
        }

    @staticmethod
    def _mio(fila):
        return {"OI": _num(fila.get("OI")), "nec": _num(fila.get("necesidad")),
                "A": _num(fila.get("A_sys_env")), "ener": _num(fila.get("energia"))}

    @staticmethod
    def _cambio_otro(o0, o1):
        """Magnitud del cambio del otro entre dos instantes (misma métrica que efecto_sobre_otro)."""
        return min(1.0, abs(o1["OI"] - o0["OI"]) + abs(o1["nec"] - o0["nec"])
                   + abs(o1["orient"] - o0["orient"]) / 90.0 + 0.25 * (1.0 if o1["voz"] != o0["voz"] else 0.0))

    # --------------------------------------------------------- LIBERTAD EXPRESIVA (balbuceo)
    @staticmethod
    def _bucket(g):
        """Firma GRUESA del gesto (buckets de 0.5) — el 'patrón' que se aprende por consecuencias."""
        return "g%+d%+d%+d%+d" % tuple(int(round(float(x) * 2)) for x in g[:4])

    @staticmethod
    def _desbucket(b):
        """Centro (vector) de un bucket de gesto: 'g+1-0+2-1' -> [0.5,0,1.0,-0.5]."""
        try:
            import re
            nums = [int(x) for x in re.findall(r"[+-]\d+", str(b))]
            return (np.array(nums[:4], dtype=float) / 2.0) if len(nums) >= 4 else np.zeros(4)
        except Exception:
            return np.zeros(4)

    def _centro_mejor_gesto(self, ctx):
        """Centro del gesto con MAYOR valor aprendido en este contexto (para el leve tirón). Neutro si no hay."""
        mejores = [(k, v) for k, v in self.valor.items() if k[1] == ctx and v > 1e-4]
        if not mejores:
            return np.zeros(4)
        bucket = max(mejores, key=lambda kv: kv[1])[0][0]
        return self._desbucket(bucket)

    def gesto_actual(self, fila: dict) -> dict:
        """CÓDIGO MUERTO HOY — comprobado el 4-ago-2026: nadie llama a este método, ni en el repositorio
        ni en la copia desplegada (C:\\...\\Programs\\ANIMA\\app). Las columnas g_freq / g_intensidad /
        g_pausa / g_repeticion / g_bucket que SÍ aparecen en el CSV (101.340 pasos, g_freq sd = 0,180) las
        produce VST_Expresion.py, no esto; `observar()` las lee de la fila ya escritas por aquél. Por eso
        `explora = 0,10`, `atraccion = 0,05` y el factor `1 + 0,5·necesidad` de aquí NO deciden nada sobre
        el organismo y no se han tocado: corregirlos sería maquillar código que no corre. Si algún día se
        vuelve a enchufar este balbuceo, hay que auditarlos ANTES, porque ninguno tiene origen escrito.

        Explora una pequeña variación ACÚSTICA espontánea (balbuceo): random walk pequeño y reversible
        + leve atracción al mejor gesto histórico de este contexto. NO elige etiquetas. Fija el patrón
        (bucket acústico) que 'observar' aprenderá por consecuencias. La exploración domina; el tirón es
        leve (no se acelera el aprendizaje ni se fija convención). Devuelve los parámetros físicos del gesto."""
        if not self.libertad:
            self._gesto = np.zeros(4); self._P_gesto = "fisio"
            return {"g_freq": 0.0, "g_intensidad": 0.0, "g_pausa": 0.0, "g_repeticion": 0.0, "g_bucket": "fisio"}
        ctx = self._ctx(fila)
        nec = _num(fila.get("necesidad"))
        bias = self._centro_mejor_gesto(ctx)
        paso = self.explora * (1.0 + 0.5 * nec)            # explora un poco más con necesidad (como SEEKING)
        self._gesto = self._gesto + paso * self._rng.randn(4) + self.atraccion * (bias - self._gesto)
        self._gesto = np.clip(self._gesto, -1.0, 1.0)
        self._gesto[2] = abs(self._gesto[2]); self._gesto[3] = abs(self._gesto[3])   # pausa, repetición ≥ 0
        self._P_gesto = self._bucket(self._gesto)
        return {"g_freq": round(float(self._gesto[0]), 3), "g_intensidad": round(float(self._gesto[1]), 3),
                "g_pausa": round(float(self._gesto[2]), 3), "g_repeticion": round(float(self._gesto[3]), 3),
                "g_bucket": self._P_gesto}

    # --------------------------------------------------------- ciclo
    def observar(self, fila: dict, otro: dict | None, dt: float = 0.1) -> dict:
        t = _num(fila.get("t"))
        # el 'patrón' aprendido es el GESTO acústico (libertad expresiva) si está activo; si no, la etiqueta.
        # Sólo cuenta como emisión cuando el organismo DE VERDAD vocaliza (no en silencio).
        _voz = fila.get("voz_emitida", "-")
        _vocaliza = _voz not in (None, "-", "")
        P = (fila.get("g_bucket") or _voz) if _vocaliza else "-"
        ctx = self._ctx(fila)
        otro_r = self._resumen_otro(otro)
        # ¿está el otro AHÍ? (vivo + da señal: OI o voz)
        presente = 1.0 if (otro_r["vivo"] and (otro_r["OI"] > 1e-4 or (otro_r["voz"] not in (None, "-", "")))) else 0.0
        conf = _num(fila.get("mem_relacional_confianza"))
        self.otro_hist.append((t, otro_r))    # historia del otro (para la línea-base de contingencia)

        # un TURNO/acto emisor = cuando CAMBIA el patrón emitido (no cada paso)
        emis = (P != self._P_prev) and P not in (None, "-", "")
        contacto_presencia = 0.0
        if emis:
            mio0 = self._mio(fila)
            # LÍNEA-BASE DE CONTINGENCIA (pre/post): cuánto cambió el otro en la ventana JUSTO ANTES de
            # emitir. Si tras emitir cambia MÁS que antes → es agencia mía; si igual → ambiente compartido.
            o_pre = next((oo for (tt, oo) in self.otro_hist if t - tt <= self.ventana), otro_r)
            efecto_pre = self._cambio_otro(o_pre, otro_r)
            self.pendientes.append({"t": t, "P": P, "ctx": ctx, "otro0": otro_r, "mio0": mio0,
                                    "pres0": presente, "efecto_pre": efecto_pre})
            self.ult_emis_t = t
            self.n_emis[(P, ctx)] = self.n_emis.get((P, ctx), 0) + 1
            self.eventos.append(("alteridad_emision", f"emite {P}", {"ctx": ctx, "presente": presente}))
            # ¿es una LLAMADA? (emite cuando el otro está ausente o acaba de irse) = "¿sigues ahí?"
            # QUÉ ESTABA MAL: `(self._presente_prev - presente) > 0.3` parecía un umbral, pero `presente`
            # sólo vale 0 ó 1 (se calcula justo arriba con un `1.0 if ... else 0.0`), así que CUALQUIER
            # número entre 0 y 1 daba el mismo resultado: el 0,3 no decidía nada, sólo disfrazaba de
            # umbral la pregunta «¿acabo de perder al otro?». Se escribe la pregunta directamente, sin
            # número: así nadie tiene que averiguar de dónde salía el 0,3. Comportamiento idéntico.
            if presente < 0.5 or presente < self._presente_prev:
                self._llamando = (t, P); contacto_presencia = 1.0
                self.eventos.append(("alteridad_contacto", f"llamada {P} (otro ausente/lejano)", None))
            self.eventos.append(("alteridad_turno", f"turno: {P}", None))
        self._P_prev = P

        # ¿se recuperó el contacto tras una llamada? (el otro volvió en una ventana)
        self.contacto_recuperado = 0.0
        if self._llamando is not None:
            tll, _ = self._llamando
            if presente >= 0.5 and self._presente_prev < 0.5:
                self.contacto_recuperado = 1.0
                self.eventos.append(("alteridad_contacto", "contacto RECUPERADO tras llamada", None))
                self._llamando = None
            elif t - tll > 5.0:
                # SE DEJA A PROPÓSITO, con la medición delante. Este 5,0 s es el plazo que el organismo
                # espera respuesta tras llamar, y sí es un número puesto a mano. Pero MEDIDO sobre
                # 101.340 pasos (dt = 0,1 s) del organismo ANIMA_5Z934MWHNNRH resulta INERTE: hubo 6
                # episodios de ausencia del otro y los 6 se registraron como contacto recuperado — el
                # plazo no descartó ninguno. La razón es que `self._llamando` se re-arma en CADA emisión
                # hecha con el otro ausente (18.253 pasos de llamada), así que el reloj se reinicia solo
                # y casi nunca llega a vencer. Cambiarlo sería tocar algo que no ha decidido nada.
                # PARA CUANDO SÍ IMPORTE: la duración real de las ausencias medidas fue mediana 3,70 s
                # (p25 2,12 · p75 7,60 · máx 74,30), así que 5,0 s está en el orden de magnitud correcto
                # por casualidad, no por medida. Con más episodios lo autorregulado sería una Escala de
                # duraciones de ausencia; con 6 episodios en toda la historia no maduraría nunca (el
                # mínimo de escala.py son 20), y fabricar una escala con 6 datos sería inventar.
                self._llamando = None
        self._presente_prev = presente

        # procesar emisiones cuya VENTANA ya pasó → medir efecto en el otro y en mí
        mio_now = self._mio(fila)
        while self.pendientes and (t - self.pendientes[0]["t"]) >= self.ventana:
            ev = self.pendientes.popleft()
            o0 = ev["otro0"]
            dOI_o = otro_r["OI"] - o0["OI"]
            dnec_o = otro_r["nec"] - o0["nec"]
            dor_o = (otro_r["orient"] - o0["orient"]) / 90.0
            voz_cambio = 1.0 if (otro_r["voz"] != o0["voz"]) else 0.0
            efecto_otro = min(1.0, abs(dOI_o) + abs(dnec_o) + abs(dor_o) + 0.25 * voz_cambio)
            # beneficio propio = subió mi OI + subió mi acople − subió mi necesidad
            m0 = ev["mio0"]
            efecto_mi = (mio_now["OI"] - m0["OI"]) + (mio_now["A"] - m0["A"]) - (mio_now["nec"] - m0["nec"])

            # CONTINGENCIA (medida NUEVA, diagnóstica — NO altera la conducta): cuánto cambió el otro tras
            # emitir POR ENCIMA de lo que ya cambiaba JUSTO ANTES (línea-base pre-emisión). Separa AGENCIA
            # (me responde a MÍ) de PRESENCIA (cambiaba igual). intención/valor SIGUEN siendo nivel-presencia
            # (sobreviven a shuffle = correcto); la AGENCIA es la señal que debe COLAPSAR bajo shuffle.
            efecto_pre = ev.get("efecto_pre", 0.0)
            # PROMEDIAR pre y post por separado y rectificar el PROMEDIO (no cada emisión): si se rectifica
            # por emisión, el ruido decorrelacionado suma siempre positivo y la agencia no colapsa nunca.
            self.baseline_ema = (1 - self.ema) * self.baseline_ema + self.ema * efecto_pre
            P_e = ev["P"]; k = (P_e, ev["ctx"])
            pred = self.modelo_otro.get(P_e, 0.0)
            self.error_pred_ema = (1 - self.ema) * self.error_pred_ema + self.ema * abs(efecto_otro - pred)
            self.modelo_otro[P_e] = (1 - self.lr) * pred + self.lr * efecto_otro     # modelo del otro: efecto esperado de P

            # ¿RESPONDIÓ EL OTRO? — antes: `efecto_otro > 0.05`, escrito tres veces en este bloque.
            # QUÉ ESTABA MAL: ese 0,05 comparaba el cambio del otro contra una escala que nadie midió, y
            # MEDIDO sobre 101.340 pasos del organismo ANIMA_5Z934MWHNNRH no discriminaba nada:
            # alt_efecto_sobre_otro vale exactamente 0 el 42,6 % del tiempo y supera 0,05 el 56,1 %, con
            # mediana de los no nulos ≈ 0,28. Es decir: entre los pasos en que el otro cambia ALGO,
            # prácticamente todos pasaban el umbral. Era un «> 0» con aspecto de criterio.
            # POR QUÉ LA CORRECCIÓN ES AUTORREGULADA: el cambio del otro se compara con LO HABITUAL QUE
            # CAMBIA EL OTRO para este organismo (su propia escala, misma magnitud y mismas unidades).
            # Y POR QUÉ NO SE USA `baseline_ema`, que sería la comparación aún más fuerte: porque este
            # órgano tiene un diseño falsable explícito —intención y valor son NIVEL-PRESENCIA y DEBEN
            # sobrevivir al shuffle; sólo la AGENCIA debe colapsar—. Usar el pre/post aquí convertiría la
            # intención en agencia y borraría esa distinción. Con la Escala el diseño se conserva: al
            # barajar, la distribución marginal de efecto_otro no cambia, así que este criterio tampoco.
            # Mientras no haya historia, se abstiene (no cuenta como respuesta) en vez de inventar.
            self.esc_efecto.observar(efecto_otro)
            respondio = self.esc_efecto.madura and _rel(efecto_otro, self.esc_efecto) > NEUTRO

            # VALOR de emisión: cuenta el beneficio SI el otro respondió (anti-Shannon: por consecuencia)
            valor_obs = efecto_mi if respondio else 0.0
            self.valor[k] = (1 - self.lr) * self.valor.get(k, 0.0) + self.lr * valor_obs
            # INTENCIÓN (nivel-presencia): el otro respondió Y me benefició
            contrib = (min(1.0, efecto_otro) if respondio else 0.0) * (1.0 if efecto_mi > 0 else 0.0)
            self.intencion = (1 - self.ema) * self.intencion + self.ema * contrib
            self.efecto_otro_ema = (1 - self.ema) * self.efecto_otro_ema + self.ema * efecto_otro
            self.efecto_mi_ema = (1 - self.ema) * self.efecto_mi_ema + self.ema * efecto_mi
            if respondio and efecto_mi > 0:
                self.eventos.append(("alteridad_refuerzo", f"{P_e}: el otro cambió y me ayudó", {"valor": round(self.valor[k], 4)}))
            elif not respondio:
                self.eventos.append(("alteridad_fallo", f"{P_e}: no movió al otro", None))
            else:
                self.eventos.append(("alteridad_respuesta", f"{P_e}: movió al otro", {"efecto": round(efecto_otro, 3)}))

        # AGENCIA del otro: exceso del cambio POST sobre el PRE (promedios), no compartir ambiente.
        # max(0, mediaPost − mediaPre): bajo decorrelación pre≈post → contingencia→0 (colapsa, como debe).
        self.contingencia = max(0.0, self.efecto_otro_ema - self.baseline_ema)
        self.agencia = max(0.0, min(1.0, self.contingencia / (self.efecto_otro_ema + 1e-6)))

        return {
            "alt_otro_presente": round(presente, 3),
            "alt_efecto_basal": round(self.baseline_ema, 4),
            "alt_contingencia_social": round(self.contingencia, 4),
            "alt_agencia_otro": round(self.agencia, 4),
            "alt_modelo_otro": round(self.modelo_otro.get(P, 0.0), 4),
            "alt_prediccion_respuesta": round(self.modelo_otro.get(P, 0.0), 4),
            "alt_error_prediccion": round(self.error_pred_ema, 4),
            "alt_efecto_sobre_otro": round(self.efecto_otro_ema, 4),
            "alt_efecto_sobre_mi": round(self.efecto_mi_ema, 4),
            "alt_valor_emision": round(self.valor.get((P, ctx), 0.0), 4),
            "alt_intencion_comunicativa": round(self.intencion, 4),
            "alt_patron_emitido": P,
            "alt_patron_repetido": 1.0 if self.n_emis.get((P, ctx), 0) > 1 else 0.0,
            "alt_confianza_relacional": round(conf, 4),
            "alt_contacto_presencia": round(contacto_presencia, 3),
            "alt_contacto_recuperado": round(self.contacto_recuperado, 3),
            "alt_turno_detectado": 1.0 if emis else 0.0,
        }

    # --------------------------------------------------------- hook OPCIONAL (OFF por defecto)
    def sesgar_emision(self, P_fisiologico: str, fila: dict, repertorio: list, explorar: float = 0.3):
        """CÓDIGO MUERTO HOY — comprobado el 4-ago-2026: nadie llama a este hook (está OFF por diseño, y
        no aparece ni una llamada en el repositorio ni en la copia desplegada). Por eso `explorar = 0,3` y
        el `necesidad > 0.6` de abajo NO deciden nada sobre el organismo y se dejan intactos. Para cuando
        se encienda, queda medido: `necesidad > 0,6` se cumple el 12,99 % de 101.340 pasos (mediana real
        de la necesidad 0,2297), o sea que ese umbral SÍ discriminaría — pero sigue sin tener origen
        escrito y habría que compararlo contra la escala propia de necesidad del organismo antes de usarlo.

        Sesgo comunicativo APRENDIDO (capa nueva, separada de la voz fisiológica). Devuelve el P a
        emitir: con prob. (1-explorar) el de MAYOR valor aprendido en este contexto; si no, explora
        (variación, más alta si la necesidad es alta o no hay valor aprendido). NO usa etiquetas
        semánticas. SÓLO se usa si el organismo lo activa (anti-Shannon: la conducta también emerge)."""
        ctx = self._ctx(fila)
        cands = [(P, self.valor.get((P, ctx), 0.0)) for P in (repertorio or [P_fisiologico])]
        mejor = max(cands, key=lambda kv: kv[1]) if cands else (P_fisiologico, 0.0)
        nec = _num(fila.get("necesidad"))
        # determinista por estado (sin Math.random): explora si la necesidad es alta o no hay valor
        explora_ahora = (mejor[1] <= 1e-4) or (nec > 0.6)
        return P_fisiologico if explora_ahora else mejor[0]
