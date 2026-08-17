#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_ORGANO_OIDO_DIGITAL — el OÍDO del canal digital nRF24 (boca sin oído → oído)
================================================================================
QUIÉN SOY
  Soy el OÍDO del canal digital. Entre A (Mac/Docker) y E (Raspberry Pi) hay una radio DIGITAL nRF24
  (canal Shannon: paquetes con ACK). La radio ya ENTREGA los mensajes del otro a la fila cada tick —
  columnas `nrf_last_rx` (último mensaje digital recibido), `nrf_rx_delta` (cuántos paquetes NUEVOS
  llegaron), `nrf_vivo`/`nrf_connected` (estado del enlace) — pero el mensaje se MUESTRA y no ENTRA en la
  cognición. Un test (5-jul-2026) lo demostró: el canal entrega el 100% de los paquetes pero A y E NO se
  acoplan (correlación de arousal r=-0.31). Es una BOCA SIN OÍDO.

  Soy el ANÁLOGO digital de OrganeloValorEcologicoVoz + OrganeloExpectativa, que son el oído del canal de
  AUDIO. Igual que OrganoComunicacion mantiene un modelo `voz_otro_*` de la voz del otro por audio, yo
  mantengo un modelo `oido_dig_*` de los SÍMBOLOS del otro por radio digital.

POR QUÉ EXISTO
  Para que el símbolo digital recibido MODULE emergentemente al organismo, de modo que A↔E puedan
  acoplarse A TRAVÉS del canal digital — sin decodificación semántica y de forma falsable.

CÓMO FUNCIONO (anti-Shannon, emergente, falsable)
  Trato cada paquete como una SEGUNDA MODALIDAD SENSORIAL OPACA del otro. NO lo parseo, NO hay tabla
  "si recibo X entonces siento Y". Aprendo por HISTORIA DE CO-OCURRENCIA dos cosas:

    (a) ESPEJO / forward-model (informativo): a qué ESTADO del otro (arousal/valence sensado por AUDIO)
        tiende a acompañar cada FIRMA estructural del mensaje. `espejo[firma]` es una EMA revisable con
        olvido: NO es un diccionario fijo, el mismo string se remapea si la relación cambia. De aquí sale
        `fiabilidad` = ganancia predictiva del canal sobre la línea-base (colapsa bajo NULL/SHUFFLED).

    (b) VALOR ECOLÓGICO / consecuencia (útil): si ATENDER el canal digital ANTECEDE una mejora REAL de mi
        persistencia por encima de mi línea-base (pre/post, idéntico patrón a ValorEcologicoVoz). Este es
        el GATE PRIMARIO y funciona DIGITAL-ONLY (no necesita audio co-presente), resolviendo el
        bootstrapping del espejo. Colapsa bajo NULL/SHUFFLED por el rectificado del PROMEDIO.

  Sólo cuando lo digital VALE (por consecuencia y/o por poder predictivo confirmado) abro LEVEMENTE la
  absorción de la voz del par por el bus YA EXISTENTE (_VOZ_ECO_MOD, el mismo que usan VALOR_ECO y
  EXPECTATIVA). NUNCA toco arousal directamente: el acople A↔E EMERGE como CONSECUENCIA de abrir el oído
  (más absorción → entra más energía semiótica del otro → el arousal de A empieza a seguir al de E por vía
  SENSORIAL), no de copiar un byte. El estimado `oido_dig_arousal_otro_est` se EXPONE como columna
  observable pero JAMÁS se inyecta (esa vía se descarta por fuga Shannon).

  LIBERTAD FUNCIONAL de NO atender: la disposición a atender = act_perm·(1−necesidad). Con alta
  necesidad / baja energía el organismo puede ignorar lo digital. Nada fuerza uptake. OLVIDO doble: el
  espejo decae hacia la línea-base y la fiabilidad hacia 0; un canal que deja de predecir/servir se apaga
  solo.

FALSACIÓN (reusa ANIMA_CONTROL = real|null|shuffled, gestionado por WebLive vía estado_otro/nrf)
  NULL:     sin paquetes (nrf vacío) → cero eventos → fiabilidad=0, valor_eco=0 → modulacion=1.0.
  SHUFFLED: paquetes REALES en los mismos tiempos pero contenido/estado descorrelacionado → espejo→base
            (fiabilidad→0) y post≈pre (valor_eco no consolida) → modulacion→1.0.
  ÉXITO:    r(arousal_A, arousal_E) en REAL ≫ SHUFFLED ≈ NULL ≈ 0, con oido_dig_valor_ecologico y
            oido_dig_fiabilidad siguiendo el mismo orden.

  FRASE GUÍA: el mensaje digital no significa nada por sí mismo; empieza a importar sólo cuando su
  historia me enseña que atenderlo reduce mi incertidumbre sobre el otro y mejora mi persistencia.
================================================================================
"""
from __future__ import annotations
import os, math, re
from collections import deque

# ── Columnas que este organelo aporta a la fila. Todas NEUTRAS (0.0/1.0/"") en silencio. Prefijo oido_dig_*.
COLS_OIDO_DIG = [
    "oido_dig_activo",            # 0/1  llegó paquete NUEVO este tick (rx_delta>0 & vivo)
    "oido_dig_simbolo",          # str  firma estructural actual ("·" en silencio) — NO semántica
    "oido_dig_arousal_otro_est", # float estimación-espejo del arousal del otro (OBSERVABLE, NO se inyecta)
    "oido_dig_valence_otro_est", # float ídem valencia
    "oido_dig_fiabilidad",       # 0..1 ganancia predictiva del canal sobre la línea-base. COLAPSA NULL/SHUFFLED
    "oido_dig_concordancia",     # -1..1 acuerdo estimación-digital vs otro sensado por audio
    "oido_dig_valor_ecologico",  # 0..1 ¿atender lo digital precede mejora de persistencia? (GATE PRIMARIO)
    "oido_dig_confianza",        # 0..1 EMA de utilidad consolidada
    "oido_dig_atender",          # 0..1 disposición LF a atender (gate act_perm·(1−necesidad))
    "oido_dig_modulacion",       # ~1.0 factor DÉBIL aplicado a la absorción del par (neutro 1.0)
    "oido_dig_error_pred",       # float error actual del espejo
    "oido_dig_n_simbolos",       # int  tamaño del repertorio digital aprendido
    "oido_dig_eventos",          # int  conteo de paquetes-evento procesados
]

# salida neutra (flag OFF nunca la usa; con flag ON se devuelve tal cual en silencio/sin evento)
_NEUTRO = {
    "oido_dig_activo": 0.0, "oido_dig_simbolo": "·", "oido_dig_arousal_otro_est": 0.0,
    "oido_dig_valence_otro_est": 0.0, "oido_dig_fiabilidad": 0.0, "oido_dig_concordancia": 0.0,
    "oido_dig_valor_ecologico": 0.0, "oido_dig_confianza": 0.0, "oido_dig_atender": 0.0,
    "oido_dig_modulacion": 1.0, "oido_dig_error_pred": 0.0, "oido_dig_n_simbolos": 0.0,
    "oido_dig_eventos": 0.0,
}


def _num(v, d=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else d
    except Exception:
        return d


def _clip01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class OrganoOidoDigital:
    """OÍDO del canal digital nRF24. Lee nrf_last_rx/nrf_rx_delta como huella OPACA (jamás decodifica),
    aprende (a) un ESPEJO del estado del otro por firma (fiabilidad, informativo) y (b) el VALOR ECOLÓGICO
    de atender el canal (consecuencia, gate primario, digital-only), y con eso modula LEVEMENTE la absorción
    de la voz del par por el bus _VOZ_ECO_MOD (nunca arousal). Emergente, con olvido, LF de NO atender,
    falsable (NULL/SHUFFLED). Simétrico en A y E. NO impone significado, NO controla."""

    def __init__(self, organismo_id: str, activo: bool = True,
                 ventana: float = 1.5, lr: float = 0.05, ema: float = 0.04,
                 lr_espejo: float = 0.05, olvido: float = 0.015,
                 k_mod: float = 0.6, cap_mod: float = 0.15, beta: float = 0.5,
                 n_bucket: int = 32, n_sym_min: int = 3, thr: float = 0.005):
        self.id = str(organismo_id)
        self.activo = bool(activo)
        # --- constantes del diseño ganador ---
        self.ventana = float(ventana)        # s para pre/post de consecuencia y de espejo
        self.lr = float(lr)                  # aprendizaje LENTO del valor ecológico (consecuencia)
        self.ema = float(ema)                # EMA de confianza/error
        self.lr_espejo = float(lr_espejo)    # aprendizaje del forward-model del otro
        self.olvido = float(olvido)          # decaimiento por tick (espejo→base, fiabilidad→0, valor→erosión)
        self.k_mod = float(k_mod)            # ganancia de la modulación
        self.cap_mod = float(cap_mod)        # tope DURO de |efecto| (≈±15 %)
        self.beta = float(beta)              # peso del término confirmatorio del espejo en el gate
        self.n_bucket = int(n_bucket)        # nº de buckets si el repertorio de firmas explota
        self.n_sym_min = int(n_sym_min)      # nº mínimo de repeticiones antes de dar peso a una firma
        self.thr = float(thr)                # umbral de "mejora real"
        # --- estado interno (aprendido, revisable, con olvido) ---
        self.espejo = {}                     # firma -> [a_otro, v_otro] EMA: estado del otro que acompaña la firma
        self.base_otro = [0.0, 0.0]          # EMA global (a,v) del otro sin condicionar = línea-base del espejo
        self.n_sym = {}                      # firma -> conteo (repetición); poda por desuso
        self.err_cond = {}                   # firma -> error condicionado (EMA)
        self.fiabilidad = 0.0                # 0..1 ganancia predictiva sobre la línea-base
        self.valor = {}                      # firma -> valor ecológico (exceso PROMEDIO post−pre)
        self.modelo = {}                     # firma -> mejora POST media (EMA)
        self.modelo_pre = {}                 # firma -> mejora PRE media (línea-base para rectificar el PROMEDIO)
        self.confianza = 0.0                 # EMA de utilidad consolidada
        self.error_pred_ema = 0.0            # error actual del espejo (observable)
        self.eventos_n = 0                   # conteo de paquetes-evento procesados
        # --- ventanas / buffers ---
        self.pendientes = deque(maxlen=512)  # eventos esperando cierre de ventana (consecuencia + espejo)
        self.persist_hist = deque(maxlen=512)  # (t, persistencia) para la línea-base pre-evento
        self.otro_hist = deque(maxlen=512)   # (t, (a,v)) del otro sensado por audio, para el target del espejo
        self._ring = deque(maxlen=256)       # ring de (a,v) del otro (soporte de diagnóstico shuffled interno)
        self._rx_prev = None                 # último nrf_last_rx visto (dedup defensivo además de rx_delta)
        # --- salida cacheada ---
        self.mod_actual = 1.0
        self.eventos = []                    # hitos para la bitácora (patrón hermano)

    # ---------------------------------------------------------------- helpers de persistencia
    @staticmethod
    def _vitales(fila):
        """Componentes de la PERSISTENCIA del receptor (idéntico a ValorEcologicoVoz)."""
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
        """Escalar de persistencia = acople + ½ICR + energía − necesidad + ½H (igual que los hermanos)."""
        return v["A"] + 0.5 * v["ICR"] + v["E"] - v["nec"] + 0.5 * v["H"]

    # ---------------------------------------------------------------- firma estructural (NO semántica)
    _re_digits = re.compile(r"\d+")
    _re_ws = re.compile(r"\s+")

    def _firma(self, rx):
        """ESTRUCTURA del mensaje (no valor decodificado): strip/lower, COLAPSA campos monótonos (dígitos:
        contadores/timestamps → '#'), normaliza espacios, trunca. Si el repertorio explota, bucket por hash
        coarse con N pequeño. La firma es identidad NOMINAL ('este TIPO de mensaje vs. otro'), nunca un dato."""
        if rx is None:
            return "·"
        s = str(rx).strip().lower()
        if s in ("", "-", "none"):
            return "·"
        s = self._re_digits.sub("#", s)          # contadores/timestamps → mismo símbolo (evita explosión)
        s = self._re_ws.sub(" ", s)
        s = s[:16]                               # trunca: sólo el prefijo estructural importa
        # si aún hay demasiadas firmas vivas, buckea por hash coarse a n_bucket
        if len(self.n_sym) >= self.n_bucket and s not in self.n_sym:
            return "#b%d" % (hash(s) % self.n_bucket)
        return s

    # ---------------------------------------------------------------- estado del otro (audio) → target espejo
    @staticmethod
    def _estado_av(estado_otro):
        """arousal/valence del otro sensado por AUDIO. El par puede venir PLANO (fila) o anidado en ['fila'].
        Devuelve None si no hay percepción (NULL) → el espejo no tiene target y no aprende de audio."""
        if not estado_otro:
            return None
        f = estado_otro.get("fila") if isinstance(estado_otro.get("fila"), dict) else estado_otro
        a = f.get("voz_arousal", f.get("arousal"))
        v = f.get("voz_valence", f.get("valence"))
        if a is None and v is None:
            return None
        return (_num(a), _num(v))

    # ---------------------------------------------------------------- ciclo
    def observar(self, fila: dict, estado_otro=None, dt: float = 0.1) -> dict:
        """Lee nrf_* de la fila, aprende espejo+consecuencia, aplica la modulación DÉBIL sobre
        `oido_dig_modulacion` (que WebLive compone en _VOZ_ECO_MOD) y devuelve las columnas oido_dig_*.
        Si activo=False, salida neutra sin tocar nada (idempotente)."""
        if not self.activo:
            return dict(_NEUTRO)
        try:
            return self._observar(fila, estado_otro, dt)
        except Exception:
            # jamás romper el tick (patrón de los hermanos: el organelo falla silencioso y neutro)
            return dict(_NEUTRO)

    def _observar(self, fila, estado_otro, dt):
        t = _num(fila.get("t"))
        vit = self._vitales(fila)
        persist = self._persistencia(vit)
        self.persist_hist.append((t, persist, vit))

        av_otro = self._estado_av(estado_otro)   # None bajo NULL / sin co-presencia acústica
        if av_otro is not None:
            self.otro_hist.append((t, av_otro))
            self._ring.append(av_otro)
            # línea-base global del espejo (a,v del otro sin condicionar)
            self.base_otro[0] = (1 - self.lr_espejo) * self.base_otro[0] + self.lr_espejo * av_otro[0]
            self.base_otro[1] = (1 - self.lr_espejo) * self.base_otro[1] + self.lr_espejo * av_otro[1]

        # --- EVENTO discreto: SÓLO si llegó paquete NUEVO y el enlace está vivo. nrf_last_rx PERSISTE entre
        #     ticks; sin esta puerta se re-aprendería el mismo paquete cada tick e inflaría la línea-base.
        rx = fila.get("nrf_last_rx", "")
        delta = _num(fila.get("nrf_rx_delta"))
        vivo = _num(fila.get("nrf_vivo", fila.get("nrf_connected")))
        firma = self._firma(rx)
        # dedup defensivo: aunque delta>0, si el string es idéntico al previo y no cambió, no es un símbolo nuevo
        nuevo_str = (rx != self._rx_prev)
        evento = (delta > 0.0) and (vivo >= 0.5) and (firma != "·")
        self._rx_prev = rx

        if evento:
            self.eventos_n += 1
            self.n_sym[firma] = self.n_sym.get(firma, 0) + 1
            # línea-base pre-evento de la persistencia (cuánto mejoraba YA antes de este paquete)
            p_pre = next((p for (tt, p, _v) in self.persist_hist if t - tt <= self.ventana), persist)
            mejora_pre = persist - p_pre
            # predicción del espejo ANTES de ver el target (para medir ganancia)
            pred = self.espejo.get(firma, list(self.base_otro))
            self.pendientes.append({"t": t, "firma": firma, "persist0": persist,
                                    "mejora_pre": mejora_pre, "pred": list(pred)})
            self.eventos.append(("oido_dig_rx", f"paquete {firma}", {"delta": int(delta)}))

        # --- cerrar ventanas vencidas → aprender ESPEJO (informativo) y VALOR ECOLÓGICO (consecuencia) ---
        while self.pendientes and (t - self.pendientes[0]["t"]) >= self.ventana:
            ev = self.pendientes.popleft()
            f = ev["firma"]
            # (a) ESPEJO: el estado del otro (por audio) que acompañó la ventana de esta firma
            av_win = self._av_en_ventana(ev["t"])
            if av_win is not None:
                err_base = abs(ev["pred"][0] - av_win[0]) + abs(ev["pred"][1] - av_win[1]) + 1e-9
                em = self.espejo.get(f, list(self.base_otro))
                em[0] = (1 - self.lr_espejo) * em[0] + self.lr_espejo * av_win[0]
                em[1] = (1 - self.lr_espejo) * em[1] + self.lr_espejo * av_win[1]
                self.espejo[f] = em
                err_cond = abs(em[0] - av_win[0]) + abs(em[1] - av_win[1]) + 1e-9
                self.err_cond[f] = (1 - self.ema) * self.err_cond.get(f, err_base) + self.ema * err_cond
                self.error_pred_ema = (1 - self.ema) * self.error_pred_ema + self.ema * err_cond
                err_base_glob = abs(self.base_otro[0] - av_win[0]) + abs(self.base_otro[1] - av_win[1]) + 1e-9
                # ganancia predictiva normalizada: cuánto MEJOR predice el espejo que la línea-base global
                ganancia = _clip01((err_base_glob - err_cond) / err_base_glob)
                # sólo firmas con soporte suficiente aportan a la fiabilidad (evita ruido de firmas raras)
                if self.n_sym.get(f, 0) >= self.n_sym_min:
                    self.fiabilidad = (1 - self.ema) * self.fiabilidad + self.ema * ganancia
            # (b) VALOR ECOLÓGICO: ¿atender el canal ANTECEDIÓ mejora real de persistencia? (rectifica PROMEDIO)
            mejora_post = persist - ev["persist0"]
            mejora_cont = mejora_post - ev["mejora_pre"]
            self.modelo[f] = (1 - self.lr) * self.modelo.get(f, 0.0) + self.lr * mejora_post
            self.modelo_pre[f] = (1 - self.lr) * self.modelo_pre.get(f, 0.0) + self.lr * ev["mejora_pre"]
            self.valor[f] = max(0.0, self.modelo[f] - self.modelo_pre[f])
            util = self.valor[f] > self.thr
            self.confianza = (1 - self.ema) * self.confianza + self.ema * (1.0 if util else 0.0)
            if mejora_cont > self.thr and util:
                # HITO (no inundar): una firma digital PRECEDIÓ mejora contingente real
                self.eventos.append(("oido_dig_util", f"{f}: el símbolo digital precedió mejora real",
                                     {"valor": round(self.valor[f], 4)}))

        # --- OLVIDO doble cada tick: el espejo deriva a la línea-base y la fiabilidad hacia 0. Un canal que
        #     deja de predecir/servir se apaga solo. También se erosiona el valor ecológico sin refresco.
        self.fiabilidad *= (1.0 - self.olvido)
        if self.espejo:
            for f, em in self.espejo.items():
                em[0] += self.olvido * (self.base_otro[0] - em[0])
                em[1] += self.olvido * (self.base_otro[1] - em[1])
        for f in list(self.valor.keys()):
            self.valor[f] *= (1.0 - self.olvido)
        # poda de firmas sin soporte y ya olvidadas (mantiene el repertorio pequeño)
        if len(self.n_sym) > self.n_bucket:
            for f in list(self.n_sym.keys()):
                if self.n_sym[f] < self.n_sym_min and self.valor.get(f, 0.0) < 1e-4:
                    self.n_sym.pop(f, None); self.espejo.pop(f, None)
                    self.valor.pop(f, None); self.modelo.pop(f, None)
                    self.modelo_pre.pop(f, None); self.err_cond.pop(f, None)

        # --- estimación observable del estado del otro para la firma ACTUAL (NO se inyecta) ---
        est = self.espejo.get(firma) if firma != "·" else None
        a_est, v_est = (est[0], est[1]) if est is not None else (0.0, 0.0)
        # concordancia estimación-digital vs otro sensado por audio (−1..1); 0 si no hay audio o silencio
        concord = 0.0
        if est is not None and av_otro is not None:
            d = abs(a_est - av_otro[0]) + abs(v_est - av_otro[1])
            concord = max(-1.0, 1.0 - d)     # cerca → +1, lejos → hasta −1

        # --- LIBERTAD FUNCIONAL de NO atender: alta necesidad / baja energía ⇒ puede ignorar lo digital ---
        atender = _clip01(vit["perm"]) * (1.0 - _clip01(vit["nec"]))

        # --- GATE del diseño ganador: valor_eco (consecuencia, primario, digital-only) + término
        #     confirmatorio del espejo (fiabilidad·concordancia, sólo cuando hay audio). Ambos falsables. ---
        val_actual = self.valor.get(firma, 0.0) if firma != "·" else 0.0
        gate = _clip01(val_actual) + self.beta * self.fiabilidad * max(0.0, concord)
        efecto = self.k_mod * atender * gate
        efecto = max(-self.cap_mod, min(self.cap_mod, efecto))
        self.mod_actual = 1.0 + efecto       # DÉBIL, capado a ≈±15 %; neutro (1.0) en silencio/sin valor

        return {
            "oido_dig_activo": 1.0 if evento else 0.0,
            "oido_dig_simbolo": firma,
            "oido_dig_arousal_otro_est": round(a_est, 4),
            "oido_dig_valence_otro_est": round(v_est, 4),
            "oido_dig_fiabilidad": round(_clip01(self.fiabilidad), 4),
            "oido_dig_concordancia": round(concord, 4),
            "oido_dig_valor_ecologico": round(val_actual, 4),
            "oido_dig_confianza": round(self.confianza, 4),
            "oido_dig_atender": round(atender, 4),
            "oido_dig_modulacion": round(self.mod_actual, 4),
            "oido_dig_error_pred": round(self.error_pred_ema, 4),
            "oido_dig_n_simbolos": float(len(self.n_sym)),
            "oido_dig_eventos": float(self.eventos_n),
        }

    def _av_en_ventana(self, t0):
        """estado (a,v) del otro sensado por audio dentro de la ventana que sigue a t0 (target del espejo)."""
        cand = [av for (tt, av) in self.otro_hist if t0 <= tt <= t0 + self.ventana]
        if not cand:
            return None
        n = len(cand)
        return (sum(a for a, _ in cand) / n, sum(v for _, v in cand) / n)

    def valor_max(self):
        return max(self.valor.values()) if self.valor else 0.0

    # ---------------------------------------------------------------- persistencia (snapshot/restore)
    def snapshot(self) -> dict:
        """Estado serializable para vst_persistencia (el oído no se reinicia en cada apagón)."""
        return {
            "espejo": {k: list(v) for k, v in self.espejo.items()},
            "base_otro": list(self.base_otro),
            "n_sym": dict(self.n_sym), "err_cond": dict(self.err_cond),
            "fiabilidad": self.fiabilidad, "valor": dict(self.valor),
            "modelo": dict(self.modelo), "modelo_pre": dict(self.modelo_pre),
            "confianza": self.confianza, "error_pred_ema": self.error_pred_ema,
            "eventos_n": self.eventos_n,
        }

    def restore(self, snap: dict):
        if not isinstance(snap, dict):
            return
        try:
            self.espejo = {k: list(v) for k, v in (snap.get("espejo") or {}).items()}
            self.base_otro = list(snap.get("base_otro") or [0.0, 0.0])[:2] or [0.0, 0.0]
            self.n_sym = dict(snap.get("n_sym") or {})
            self.err_cond = dict(snap.get("err_cond") or {})
            self.fiabilidad = _num(snap.get("fiabilidad"))
            self.valor = dict(snap.get("valor") or {})
            self.modelo = dict(snap.get("modelo") or {})
            self.modelo_pre = dict(snap.get("modelo_pre") or {})
            self.confianza = _num(snap.get("confianza"))
            self.error_pred_ema = _num(snap.get("error_pred_ema"))
            self.eventos_n = int(_num(snap.get("eventos_n")))
        except Exception:
            pass

    def reset(self):
        """Reinicio suave (el _despertar restaura después de esto, patrón de los hermanos)."""
        self.espejo.clear(); self.base_otro = [0.0, 0.0]; self.n_sym.clear(); self.err_cond.clear()
        self.fiabilidad = 0.0; self.valor.clear(); self.modelo.clear(); self.modelo_pre.clear()
        self.confianza = 0.0; self.error_pred_ema = 0.0; self.eventos_n = 0
        self.pendientes.clear(); self.persist_hist.clear(); self.otro_hist.clear(); self._ring.clear()
        self._rx_prev = None; self.mod_actual = 1.0; self.eventos.clear()
