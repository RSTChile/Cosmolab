#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_EXPECTATIVA — Nodo de EXPECTATIVA (NO lenguaje, NO decisión, NO predictor perfecto)
================================================================================
QUIÉN SOY
  Diagnóstico previo: la voz del otro se oye (presencia ✅) pero NO afecta al otro (agencia ❌) ni ayuda
  a mi persistencia (valor ecológico ❌). Conectar voz→persistencia es demasiado directo (riesga Shannon).
  La biología no obtiene beneficio de una SEÑAL, sino de la CONDUCTA que realiza tras interpretarla:

      voz  →  EXPECTATIVA  →  exploración  →  resultado  →  persistencia

  Mi ÚNICA función es aprender, por historia de consecuencias:
      "después de esta firma acústica del otro, ¿las EXPLORACIONES posteriores tendieron a mejorar o
       empeorar mi situación?"
  No aprendo objetos ni etiquetas. Aprendo SÓLO capacidad predictiva. NO hay diccionario, ni reward
  externo, ni significado. Mi única SALIDA es muy leve: "vale un poco más la pena seguir observando"
  (disposición a explorar / segunda observación). Nunca oriento, nunca decido, nunca toco
  metabolismo/energía/RC/persistencia directamente.

  Falsable: bajo NULL no aparece; bajo SHUFFLED fuerte no se consolida; en REAL sólo si hay contingencia
  histórica. Hipótesis genealógica: expectativa ANTES que agencia, agencia ANTES que intención, intención
  ANTES que convención, convención ANTES que lenguaje.

  FRASE GUÍA: la voz del otro todavía no debe cambiar mi vida; primero debe enseñarme que, históricamente,
  después de escuchar ciertos patrones, vale la pena seguir explorando.
================================================================================
"""
from __future__ import annotations
import os, math
from collections import deque

# El patrón «comparar contra lo habitual del propio organismo» se IMPORTA, no se reescribe (escala.py).
# La auditoría del 4-ago-2026 avisó: si este cálculo se escribe 168 veces, en tres meses hay 168 variantes.
#   rel(x, escala)      -> 0,5 cuando x es lo de siempre para este organismo. Sin parámetro libre.
#   rel_contra(x, ref)  -> 0,5 cuando x iguala a OTRA magnitud del organismo con las MISMAS unidades.
# `escala` vive en celula_madre/; esto permite importar el organelo suelto (pruebas y smokes)
# además de dentro del organismo. Unificado el 5-ago-2026: la revisión encontró CUATRO
# variantes del mismo arranque, que es el problema contra el que existe el módulo compartido.
import os as _os, sys as _sys
_RAIZ_CM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _RAIZ_CM not in _sys.path:
    _sys.path.insert(0, _RAIZ_CM)
from escala import Escala, rel as _rel, rel_contra as _rel_contra, NEUTRO

COLS_EXP = [
    "expectativa", "expectativa_confianza", "expectativa_error", "expectativa_historia",
    "expectativa_utilidad", "expectativa_exploracion", "expectativa_confirmaciones", "expectativa_falsaciones",
]


def _num(v, d=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else d
    except Exception:
        return d


class OrganeloExpectativa:
    """Aprende si EXPLORAR tras una firma acústica del otro tiende a mejorar la situación. Salida única y
    leve: disposición a seguir observando. Falsable (NULL/SHUFFLED). NO significado, NO control. A/B simétrico."""

    def __init__(self, organismo_id: str, ventana: float = 1.5, lr: float = 0.05, ema: float = 0.04,
                 k_expl: float = 0.5, cap_expl: float = 0.20):
        self.id = str(organismo_id)
        self.ventana = float(ventana)        # s tras la firma para ver si la exploración mejoró la situación
        self.lr = float(lr)                  # aprendizaje LENTO de la expectativa
        self.ema = float(ema)
        # ------------------------------------------------------------------------------------------
        # DESAPARECIDOS AQUÍ (4-ago-2026), y por qué:
        #  · `umbral_voz = 0.02`: era un RMS de audio ABSOLUTO que decidía si el organismo OYE al otro,
        #    contra un ambiente acústico que nadie midió. Ahora el fondo se aprende (self.esc_e).
        #  · `thr = 0.005`: era el umbral de "mejora real". MEDIDO sobre 101.340 pasos del organismo
        #    ANIMA_5Z934MWHNNRH: sólo el 1,74 % de los pasos cae en la banda (0 ; 0,005], es decir que el
        #    umbral NO discriminaba nada — era un "> 0" disfrazado de criterio. Y la tasa de confirmación
        #    que producía, 1.170/(1.170+1.254) = 48,3 %, es indistinguible de una moneda al aire. Ahora la
        #    "mejora real" se mide contra el propio error de predicción (self.error_ema), misma unidad.
        # ------------------------------------------------------------------------------------------
        self.k_expl = float(k_expl)          # cuánto la expectativa empuja la exploración
        # TOPE, no criterio: en 101.340 pasos reales expectativa_exploracion llegó como máximo a 0,1838 y
        # NUNCA tocó este 0,20 (0 veces). Es una barandilla estructural declarada ("la salida es leve"),
        # no un número que decida: no ha decidido nada todavía. Se deja a propósito.
        self.cap_expl = float(cap_expl)
        self.esc_e = Escala()                # lo habitual de la energía de voz RECIBIDA = el ruido de fondo
        self.esc_expect = Escala()           # lo habitual de las propias expectativas (para "¿esta firma vale?")
        self.expect = {}                     # firma -> expectativa (exceso PROMEDIO post−pre de la situación)
        self.modelo = {}                     # firma -> mejora POST media (EMA) = utilidad esperada
        self.modelo_pre = {}                 # firma -> mejora PRE media (línea-base; rectificar el PROMEDIO)
        self.n_firma = {}                    # firma -> conteo (novedad/repetición)
        self.confianza = 0.0
        self.error_ema = 0.0
        self.historia = 0.0                  # beneficio histórico atribuido a explorar tras la voz
        self.confirmaciones = 0.0            # veces que explorar tras la voz mejoró
        self.falsaciones = 0.0               # veces que NO mejoró
        self.exploracion = 0.0               # salida actual (disposición a seguir observando)
        self.pendientes = deque(maxlen=4000)
        self.sit_hist = deque(maxlen=512)    # (t, situacion) para la línea-base pre-firma
        self._e_hist = deque(maxlen=24)
        self._firma_prev = "·"
        self.eventos = []

    @staticmethod
    def _situacion(fila):
        """Situación del organismo (no etiqueta): ICES/ICR + acople − necesidad + homeostasis."""
        ICR = _num(fila.get("ICR", fila.get("RC_total")))
        A = _num(fila.get("A_sys_env"))
        nec = _num(fila.get("necesidad", fila.get("necesidad_efectiva")))
        H = _num(fila.get("H_homeostasis", fila.get("H_real", fila.get("H"))))
        return 0.5 * ICR + A - nec + 0.5 * H

    def _firma(self, e):
        """ESTRUCTURA de la voz oída (no etiqueta): nivel sobre el fondo × estabilidad. Silencio = '·'.

        QUÉ ESTABA MAL — `if e <= self.umbral_voz` con umbral_voz = 0,02. Ese 0,02 es un RMS de audio
        absoluto y decidía LO MÁS GRAVE de este organelo: si hay voz o no. Por debajo no hay firma, no hay
        evento, no hay expectativa y el órgano entero queda mudo; por encima, cualquier zumbido de la sala
        cuenta como "el otro me habla". NO SE PUDO MEDIR su efecto directamente: `energia_voz_otro` no se
        publica al CSV (mapa_organismo.py: «la produce NADIE»); la única evidencia indirecta es que la
        columna `expectativa` es no nula el 29,8 % de 101.340 pasos, o sea que el umbral sí se cruza a
        menudo — pero eso no dice si 0,02 es el sitio correcto, sólo que no está apagado.

        POR QUÉ LA CORRECCIÓN ES AUTORREGULADA — el umbral auditivo de un animal no es un número: flota
        con el ruido de fondo (por eso se oye un susurro de noche y no se oye un grito en una cascada).
        Aquí el fondo es la energía habitual RECIBIDA por este organismo, aprendida siempre —también en
        silencio— por `self.esc_e`. Hay voz cuando la energía supera lo habitual: rel(e, esc_e) > 0,5.
        Es una PERCEPCIÓN («¿el otro está sonando?»), no una condición de viabilidad, así que
        relativizarla es legítimo (advertencia 2 de la auditoría). Mientras la escala no tenga historia,
        se ABSTIENE devolviendo silencio, en vez de inventarse un fondo.
        """
        self.esc_e.observar(e)                        # el fondo se aprende SIEMPRE, también en silencio
        if (not self.esc_e.madura) or _rel(e, self.esc_e) <= NEUTRO:
            return "·"
        self._e_hist.append(e)
        if len(self._e_hist) >= 3:
            m = sum(self._e_hist) / len(self._e_hist)
            desv = (sum((x - m) ** 2 for x in self._e_hist) / len(self._e_hist)) ** 0.5
            # QUÉ ESTABA MAL: `var > 0.02**2` comparaba la varianza de la energía oída contra un número
            # sin origen (y además dependía de la ganancia del micrófono: doblar el volumen de entrada
            # convertía toda voz "estable" en "inestable" sin que pasara nada en el mundo).
            # AUTORREGULADO: desviación y media tienen LAS MISMAS UNIDADES, así que se comparan entre sí
            # con rel_contra —la forma que la auditoría llama preferible—: la voz es estable cuando su
            # dispersión es menor que su nivel. Adimensional y sin parámetro libre.
            estable = 0 if _rel_contra(desv, m) > NEUTRO else 1
        else:
            estable = 1
        # QUÉ ESTABA MAL: `min(1.0, e) * 4` daba por supuesto que la energía ya venía normalizada a 0..1;
        # con un RMS de audio real (décimas o centésimas) casi todo caía en el mismo cubo y la "estructura"
        # de la voz era una sola. AUTORREGULADO: el nivel se mide como cuánto SUPERA la voz al fondo
        # propio —(rel−0,5)·2 vive en 0..1 por construcción, sin suponer nada del micrófono—. El 4 sigue
        # siendo granularidad declarada (5 cubos), no un criterio: no decide qué le pasa al organismo.
        nivel = (_rel(e, self.esc_e) - NEUTRO) * 2.0
        return "e%d_%d" % (int(round(nivel * 4)), estable)

    def observar(self, fila: dict, energia_voz_otro: float, dt: float = 0.1) -> dict:
        t = _num(fila.get("t"))
        e = max(0.0, _num(energia_voz_otro))
        sit = self._situacion(fila)
        self.sit_hist.append((t, sit))

        firma = self._firma(e)
        evento = (firma != "·") and (firma != self._firma_prev)
        if evento:
            sit_pre = next((s for (tt, s) in self.sit_hist if t - tt <= self.ventana), sit)
            mejora_pre = sit - sit_pre        # cuánto mejoraba YA mi situación antes de oír la firma
            self.pendientes.append({"t": t, "firma": firma, "sit0": sit, "mejora_pre": mejora_pre})
            self.n_firma[firma] = self.n_firma.get(firma, 0) + 1
            self.eventos.append(("expectativa_firma", f"firma {firma}", {"e": round(e, 4)}))
        self._firma_prev = firma

        # cuando pasa la ventana de EXPLORACIÓN: ¿mejoró la situación POR ENCIMA de mi línea-base?
        while self.pendientes and (t - self.pendientes[0]["t"]) >= self.ventana:
            ev = self.pendientes.popleft()
            f = ev["firma"]
            mejora_post = sit - ev["sit0"]
            mejora_cont = mejora_post - ev["mejora_pre"]
            pred = self.modelo.get(f, 0.0)
            self.error_ema = (1 - self.ema) * self.error_ema + self.ema * abs(mejora_post - pred)
            self.modelo[f] = (1 - self.lr) * pred + self.lr * mejora_post
            self.modelo_pre[f] = (1 - self.lr) * self.modelo_pre.get(f, 0.0) + self.lr * ev["mejora_pre"]
            # EXPECTATIVA = exceso del PROMEDIO post sobre el PROMEDIO pre (rectifica el PROMEDIO, no cada
            # evento → bajo SHUFFLED post≈pre y la expectativa NO converge; el ruido coincidente no se acumula).
            self.expect[f] = max(0.0, self.modelo[f] - self.modelo_pre[f])

            # ¿VALE ESTA FIRMA? — antes: `self.expect[f] > 0.005`. Ese 0,005 no discriminaba: MEDIDO
            # sobre 101.340 pasos, sólo el 1,74 % cae en la banda (0 ; 0,005], así que el criterio real
            # que se estaba aplicando era «> 0» y el 0,005 sólo daba aspecto de umbral. AUTORREGULADO:
            # una firma vale cuando su expectativa está por encima de LO HABITUAL EN LAS EXPECTATIVAS DE
            # ESTE ORGANISMO (su propia escala). Es una comparación entre lo que el organismo espera hoy
            # y lo que suele esperar; y mientras no tenga historia se abstiene (util=False) en vez de
            # regalar confianza recién nacido.
            self.esc_expect.observar(self.expect[f])
            util = self.esc_expect.madura and _rel(self.expect[f], self.esc_expect) > NEUTRO

            # ¿MEJORÓ DE VERDAD? — antes: `mejora_cont > 0.005`, con el mismo problema, y con una
            # consecuencia peor: producía una tasa de confirmación de 1.170/(1.170+1.254) = 48,3 %, que
            # es exactamente lo que daría una moneda. Un criterio que acierta el 48 % no es un criterio.
            # AUTORREGULADO: `mejora_cont` y `error_ema` están en LAS MISMAS UNIDADES (unidades de
            # situación), así que se comparan entre sí (rel_contra): la mejora es real cuando supera el
            # propio error de predicción del organismo — detección de señal sobre el ruido de uno mismo,
            # sin escala externa. Recuérdese la norma del proyecto: poco es suficiente (el guepardo caza
            # el 17 % de las veces); lo que no sirve es contar como éxito el ruido.
            hay_ruido_medido = self.error_ema > 1e-9
            real = hay_ruido_medido and _rel_contra(mejora_cont, self.error_ema) > NEUTRO
            if real:
                self.confirmaciones += 1.0
                if util:
                    self.historia += mejora_cont
                self.eventos.append(("expectativa_confirma", f"{f}: explorar tras la voz mejoró", {"exp": round(self.expect[f], 4)}))
            elif hay_ruido_medido:
                self.falsaciones += 1.0
            # si el organismo todavía no ha medido su propio ruido, NO cuenta ni confirmación ni
            # falsación: abstenerse es lo que manda escala.py, y así el recuento no arranca sesgado.
            self.confianza = (1 - self.ema) * self.confianza + self.ema * (1.0 if util else 0.0)

        # SALIDA ÚNICA Y LEVE: "vale un poco más la pena seguir observando" (disposición a explorar).
        # `k_expl = 0,5` SE DEJA A PROPÓSITO y no es legítimo del todo: es una ganancia puesta a mano. Se
        # deja porque (a) es lo único de este organelo que sale al organismo —mapa_organismo.py:
        # expectativa_exploracion la consumen los cuatro WebLive, en el bus de absorción de la voz del
        # par— y (b) MEDIDO sobre 101.340 pasos su salida vive en [0 ; 0,1838], nunca toca el tope y ya
        # es "leve" como promete el docstring: no hay evidencia de que 0,5 esté mal. Para poder auditarla
        # de verdad hace falta publicar `energia_voz_otro` al CSV. Ver informe.
        exp_act = self.expect.get(firma, 0.0) if firma != "·" else 0.0
        self.exploracion = max(0.0, min(self.cap_expl, self.k_expl * exp_act))

        return {
            "expectativa": round(exp_act, 4),
            "expectativa_confianza": round(self.confianza, 4),
            "expectativa_error": round(self.error_ema, 4),
            "expectativa_historia": round(self.historia, 4),
            "expectativa_utilidad": round(self.modelo.get(firma, 0.0), 4),
            "expectativa_exploracion": round(self.exploracion, 4),
            "expectativa_confirmaciones": round(self.confirmaciones, 1),
            "expectativa_falsaciones": round(self.falsaciones, 1),
        }

    def expect_max(self):
        return max(self.expect.values()) if self.expect else 0.0
