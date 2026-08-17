#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_VALOR_ECOLOGICO_VOZ — ¿la voz del OTRO importa para MI persistencia? (a DEMOSTRAR)
================================================================================
QUIÉN SOY
  El control fuerte mostró: presencia ✅, resonancia ✅, dependencia del otro ✅, pero AGENCIA ❌.
  La voz del otro se OYE, pero todavía no cambia mi economía vital (metabolismo, acople, permeabilidad).
  El cuello de botella NO es "subir el volumen". Es que la voz del otro aún no tiene RELEVANCIA BIOLÓGICA.

  Yo NO programo significado, ni "voz = comida", ni "voz = sujeto". NO hay diccionario, ni reward externo,
  ni control directo (no oriento, no decido). Sólo APRENDO, por HISTORIA DE CONSECUENCIAS, si una clase de
  voz recibida (su ESTRUCTURA: energía, estabilidad, repetición) ANTECEDE una mejora REAL de mi persistencia:

      recibo voz-estructura X  →  ¿mejora mi A_sys-env / ICR / energía / necesidad / H, MÁS que sin ella?

  Si X predice mejora real (contingente, por encima de mi línea-base), su VALOR ECOLÓGICO sube MUY lento, y
  ese valor sólo puede MODULAR LEVEMENTE mi permeabilidad / disposición a escuchar (facilidad de absorber
  energía semiótica). Nunca controla. Si deja de predecir mejora, decae. Bajo NULL (sin voz) no aparece;
  bajo SHUFFLED fuerte (voz presente pero descorrelacionada) no se consolida. Anti-Shannon puro.

  FRASE GUÍA: la alteridad no nace porque exista otro; nace cuando lo que hace el otro empieza a IMPORTAR
  para mi propia persistencia.
================================================================================
"""
from __future__ import annotations
import os, math
from collections import deque

# El patrón «comparar contra lo habitual del propio organismo» se IMPORTA, no se reescribe (escala.py).
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

COLS_VOZECO = [
    "voz_otro_valor_ecologico", "voz_otro_relevancia_metabolica", "voz_otro_relevancia_acople",
    "voz_otro_relevancia_permeabilidad", "voz_otro_predice_mejora", "voz_otro_error_prediccion",
    "voz_otro_historia_beneficio", "voz_otro_confianza_ecologica", "voz_otro_modulacion_aplicada",
    "voz_otro_efecto_real",
]


def _num(v, d=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else d
    except Exception:
        return d


class OrganeloValorEcologicoVoz:
    """Aprende si la voz del otro (como ESTRUCTURA recibida) tiene VALOR ECOLÓGICO: si antecede mejora real
    de la persistencia del receptor, por encima de su línea-base. Modula levemente la permeabilidad. Falsable:
    cae bajo NULL/SHUFFLED. NO impone significado, NO controla orientación ni acción. Simétrico en A y B."""

    def __init__(self, organismo_id: str, ventana: float = 1.5, lr: float = 0.05, ema: float = 0.04,
                 k_mod: float = 0.6, cap_mod: float = 0.25):
        self.id = str(organismo_id)
        self.ventana = float(ventana)        # s para ver si la persistencia mejora tras recibir la voz
        self.lr = float(lr)                  # aprendizaje LENTO del valor ecológico
        self.ema = float(ema)
        # ------------------------------------------------------------------------------------------
        # DESAPARECIDO AQUÍ (4-ago-2026): `umbral_voz = 0.02`, un RMS de audio ABSOLUTO que decidía si el
        # organismo OYE al otro contra un ambiente acústico que nadie midió. Ahora el fondo se aprende
        # (self.esc_e) igual que en VST_Expectativa, que recibe exactamente la misma entrada.
        # ------------------------------------------------------------------------------------------
        self.k_mod = float(k_mod)            # cuánto el valor ecológico modula la permeabilidad
        # TOPE, no criterio: MEDIDO sobre 101.340 pasos, voz_otro_modulacion_aplicada llegó como máximo a
        # 1,2095 y NUNCA tocó este tope (1+0,25 = 1,25; 0 veces). Es una barandilla estructural declarada
        # («leve: nunca controla»), no un número que haya decidido nada. Se deja a propósito.
        self.cap_mod = float(cap_mod)
        self.esc_e = Escala()                # lo habitual de la energía de voz RECIBIDA = el ruido de fondo
        self.esc_valor = Escala()            # lo habitual de los propios valores ecológicos aprendidos
        self.valor = {}                      # firma_voz -> valor ecológico (exceso post−pre de persistencia)
        self.modelo = {}                     # firma_voz -> mejora POST media (EMA) = "modelo de utilidad"
        self.modelo_pre = {}                 # firma_voz -> mejora PRE media (línea-base, para rectificar el PROMEDIO)
        self.n_firma = {}                    # firma_voz -> conteo (repetición)
        self.historia_beneficio = 0.0        # beneficio acumulado atribuido a la voz del otro
        self.confianza = 0.0                 # EMA: ¿la voz valorada de verdad precede mejora?
        self.error_pred_ema = 0.0
        self.rel_metab = 0.0                 # relevancia metabólica (EMA de la mejora de energía/necesidad)
        self.rel_acople = 0.0                # relevancia de acoplamiento (EMA de ΔA_sys-env)
        self.rel_perm = 0.0                  # relevancia de permeabilidad (EMA de Δpermeabilidad)
        self.efecto_real = 0.0               # última mejora medida tras la voz
        self.pendientes = deque(maxlen=4000)
        self.persist_hist = deque(maxlen=512)  # (t, persistencia) para la línea-base pre-voz
        self._e_hist = deque(maxlen=24)        # energía recibida reciente (para estabilidad)
        self._firma_prev = "·"
        self.eventos = []

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _vitales(fila):
        """Componentes de la PERSISTENCIA del receptor (no etiquetas): acople, ICR, energía, necesidad, H, perm."""
        return {
            "A": _num(fila.get("A_sys_env")),
            "ICR": _num(fila.get("ICR", fila.get("RC_total"))),
            "E": _num(fila.get("met_energia", fila.get("energia"))),
            "nec": _num(fila.get("necesidad", fila.get("necesidad_efectiva"))),
            "H": _num(fila.get("H_homeostasis", fila.get("H_real", fila.get("H")))),
            "perm": _num(fila.get("act_perm", fila.get("permeabilidad"))),
        }

    @staticmethod
    def _persistencia(v):
        """Escalar de persistencia: sostenerse como organismo = acople + ICR + energía − necesidad + H."""
        return v["A"] + 0.5 * v["ICR"] + v["E"] - v["nec"] + 0.5 * v["H"]

    def _firma(self, e):
        """ESTRUCTURA de la voz recibida (no etiqueta): nivel sobre el fondo × estabilidad. Silencio='·'.

        QUÉ ESTABA MAL — `if e <= self.umbral_voz` con umbral_voz = 0,02: un RMS de audio absoluto que
        decidía si hay voz o no, y con ello si este organelo existe. NO SE PUDO MEDIR directamente porque
        `energia_voz_otro` no se publica al CSV (mapa_organismo.py: «la produce NADIE»); la evidencia
        indirecta es que `voz_otro_valor_ecologico` es no nulo el 30,4 % de 101.340 pasos, o sea que el
        umbral se cruza, pero eso no dice que 0,02 esté en el sitio correcto.

        POR QUÉ LA CORRECCIÓN ES AUTORREGULADA — el umbral auditivo de un animal flota con el ruido de
        fondo. Aquí el fondo es la energía habitual recibida por ESTE organismo, aprendida siempre (también
        en silencio) por `self.esc_e`; hay voz cuando la energía la supera: rel(e, esc_e) > 0,5. Es una
        PERCEPCIÓN, no una condición de viabilidad, así que relativizar es legítimo. Sin historia, se
        abstiene (silencio) en vez de inventarse un fondo.
        """
        self.esc_e.observar(e)                        # el fondo se aprende SIEMPRE, también en silencio
        if (not self.esc_e.madura) or _rel(e, self.esc_e) <= NEUTRO:
            return "·"
        self._e_hist.append(e)
        if len(self._e_hist) >= 3:
            m = sum(self._e_hist) / len(self._e_hist)
            desv = (sum((x - m) ** 2 for x in self._e_hist) / len(self._e_hist)) ** 0.5
            # QUÉ ESTABA MAL: `var > 0.02**2` comparaba la varianza de la energía oída contra un número
            # sin origen, y encima dependía de la ganancia del micrófono. AUTORREGULADO: desviación y
            # media tienen LAS MISMAS UNIDADES y se comparan entre sí (rel_contra, la forma preferida
            # por la auditoría): la voz es estable cuando su dispersión es menor que su nivel.
            estable = 0 if _rel_contra(desv, m) > NEUTRO else 1
        else:
            estable = 1
        # QUÉ ESTABA MAL: `min(1.0, e) * 4` suponía que la energía ya venía normalizada a 0..1; con un RMS
        # real casi todo caía en el mismo cubo. AUTORREGULADO: el nivel es cuánto SUPERA la voz al fondo
        # propio; (rel−0,5)·2 vive en 0..1 por construcción. El 4 es granularidad declarada (5 cubos).
        nivel = (_rel(e, self.esc_e) - NEUTRO) * 2.0
        return "v%d_%d" % (int(round(nivel * 4)), estable)

    # ---------------------------------------------------------------- ciclo
    def observar(self, fila: dict, energia_voz_otro: float, dt: float = 0.1) -> dict:
        t = _num(fila.get("t"))
        e = max(0.0, _num(energia_voz_otro))
        vit = self._vitales(fila)
        persist = self._persistencia(vit)
        self.persist_hist.append((t, persist, vit))

        firma = self._firma(e)
        # un EVENTO de voz = llega una voz (o cambia su estructura). En silencio no hay evento.
        evento = (firma != "·") and (firma != self._firma_prev)
        if evento:
            # LÍNEA-BASE pre-voz: cuánto mejoraba YA mi persistencia en la ventana ANTES de oír esta voz.
            p_pre = next((p for (tt, p, _vv) in self.persist_hist if t - tt <= self.ventana), persist)
            mejora_pre = persist - p_pre
            self.pendientes.append({"t": t, "firma": firma, "persist0": persist, "vit0": vit, "mejora_pre": mejora_pre})
            self.n_firma[firma] = self.n_firma.get(firma, 0) + 1
            self.eventos.append(("vozeco_voz", f"recibe voz {firma}", {"e": round(e, 4)}))
        self._firma_prev = firma

        # procesar voces cuya VENTANA ya pasó → ¿mejoró mi persistencia POR ENCIMA de mi línea-base?
        while self.pendientes and (t - self.pendientes[0]["t"]) >= self.ventana:
            ev = self.pendientes.popleft()
            f = ev["firma"]; v0 = ev["vit0"]
            mejora_post = persist - ev["persist0"]
            mejora_cont = mejora_post - ev["mejora_pre"]      # CONTINGENTE: mejora atribuible a la voz, no a mi deriva
            pred = self.modelo.get(f, 0.0)
            self.error_pred_ema = (1 - self.ema) * self.error_pred_ema + self.ema * abs(mejora_post - pred)
            self.modelo[f] = (1 - self.lr) * pred + self.lr * mejora_post                         # mejora POST media (por firma)
            self.modelo_pre[f] = (1 - self.lr) * self.modelo_pre.get(f, 0.0) + self.lr * ev["mejora_pre"]  # mejora PRE media
            # VALOR ECOLÓGICO = exceso del PROMEDIO post sobre el PROMEDIO pre (rectificar el PROMEDIO, no cada
            # evento → bajo SHUFFLED post≈pre y el valor COLAPSA; el ruido coincidente ya no se acumula como sí).
            self.valor[f] = max(0.0, self.modelo[f] - self.modelo_pre[f])
            # ¿VALE ESTA VOZ? — antes: `self.valor[f] > 0.005`, un número escrito a mano dentro del
            # cuerpo, sin siquiera ser parámetro. MEDIDO sobre 101.340 pasos: sólo el 1,66 % de los pasos
            # cae en la banda (0 ; 0,005], o sea que el umbral NO discriminaba — el criterio real era
            # «> 0» y el 0,005 sólo tenía aspecto de criterio. AUTORREGULADO: una voz vale cuando su
            # valor ecológico está por encima de LO HABITUAL EN LOS VALORES DE ESTE ORGANISMO (su propia
            # escala), y mientras no haya historia se abstiene en vez de regalar confianza.
            self.esc_valor.observar(self.valor[f])
            util = self.esc_valor.madura and _rel(self.valor[f], self.esc_valor) > NEUTRO
            self.efecto_real = mejora_post
            if util:
                self.historia_beneficio += max(0.0, mejora_cont)
            self.confianza = (1 - self.ema) * self.confianza + self.ema * (1.0 if util else 0.0)
            # ¿QUÉ vital ayudó? (relevancia por componente, para entender SIN imponer significado)
            self.rel_metab = (1 - self.ema) * self.rel_metab + self.ema * ((vit["E"] - v0["E"]) - (vit["nec"] - v0["nec"]))
            self.rel_acople = (1 - self.ema) * self.rel_acople + self.ema * (vit["A"] - v0["A"])
            self.rel_perm = (1 - self.ema) * self.rel_perm + self.ema * (vit["perm"] - v0["perm"])
            # HITO "la voz precedió mejora REAL" — antes: `mejora_cont > 0.005`. Este evento es el que se
            # escribe a la bitácora del organismo, así que el umbral decidía qué se cuenta como hallazgo.
            # AUTORREGULADO: `mejora_cont` y `error_pred_ema` están en LAS MISMAS UNIDADES (unidades de
            # persistencia), así que se comparan entre sí: la mejora es real cuando supera el propio error
            # de predicción del organismo — detección de señal sobre el ruido de uno mismo. Si el
            # organismo todavía no ha medido su ruido, no se declara ningún hito (abstención).
            if self.error_pred_ema > 1e-9 and _rel_contra(mejora_cont, self.error_pred_ema) > NEUTRO:
                self.eventos.append(("vozeco_util", f"{f}: la voz precedió mejora real", {"valor": round(self.valor[f], 4)}))

        # MODULACIÓN LEVE de la permeabilidad: la voz ACTUAL, según su valor ecológico aprendido, facilita
        # un poco más (o no) la absorción. Nunca controla. Neutro (1.0) en silencio o si no tiene valor.
        # `k_mod = 0,6` SE DEJA A PROPÓSITO y no es del todo legítimo: es una ganancia puesta a mano. Se
        # deja porque es de las tres únicas salidas de este grupo que llegan al organismo
        # (mapa_organismo.py: voz_otro_modulacion_aplicada la consumen los cuatro WebLive, en el bus de
        # absorción; voz_otro_valor_ecologico la consume VST_Expresion.py) y porque MEDIDO sobre 101.340
        # pasos la modulación vive en [1,0000 ; 1,2095], nunca toca su tope y ya es leve como promete el
        # docstring: no hay evidencia de que 0,6 esté mal. Para auditarla hace falta publicar
        # `energia_voz_otro` al CSV. Ver informe.
        val_actual = self.valor.get(firma, 0.0) if firma != "·" else 0.0
        modulacion = 1.0 + max(-self.cap_mod, min(self.cap_mod, self.k_mod * val_actual))

        return {
            "voz_otro_valor_ecologico": round(val_actual, 4),
            "voz_otro_relevancia_metabolica": round(self.rel_metab, 4),
            "voz_otro_relevancia_acople": round(self.rel_acople, 4),
            "voz_otro_relevancia_permeabilidad": round(self.rel_perm, 4),
            "voz_otro_predice_mejora": round(self.modelo.get(firma, 0.0), 4),
            "voz_otro_error_prediccion": round(self.error_pred_ema, 4),
            "voz_otro_historia_beneficio": round(self.historia_beneficio, 4),
            "voz_otro_confianza_ecologica": round(self.confianza, 4),
            "voz_otro_modulacion_aplicada": round(modulacion, 4),
            "voz_otro_efecto_real": round(self.efecto_real, 4),
        }

    def valor_max(self):
        return max(self.valor.values()) if self.valor else 0.0
