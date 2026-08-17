#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_BitacoraSocial · resumen de interacciones para la Bitácora
==============================================================

POR QUÉ EXISTE
--------------
La columna `as_esc_objetivo` que el organelo de Atención Social escribe al CSV
parpadea: los episodios de atención continuada tienen mediana 0,0 s y p90 0,1 s,
porque la moneda de enganche (`_disposicion`) se re-sortea en CADA tick y, al
salir cara-abajo, borra el objetivo comprometido y reinicia el reloj de dwell.
Medido a ~29 Hz sobre los organismos del laboratorio.

En consecuencia:

  1. El **objetivo instantáneo no es la unidad de análisis**. Lo que significa algo
     es la DISTRIBUCIÓN en una ventana ("Alfa dedicó 58% de su atención a Duno").
  2. El hilo aplicador lee `objetivo_escucha` una vez por segundo: toma 1 de cada
     ~29 valores de un atributo que parpadea. Lo que de verdad gobernó el audio es
     `_AS_APLICADO["oid"]`, **que hoy no se guarda en ninguna parte**.

Este módulo resuelve las dos cosas sin tocar el organelo: observa de forma pasiva,
agrega en ventanas y empuja el resumen a la Bitácora.

QUÉ NO HACE
-----------
* No decide nada ni modifica el comportamiento del organismo.
* No bloquea nunca al llamador: acumular es O(1) y el envío va en un hilo aparte.
* Si la Bitácora no responde, se rinde con backoff y descarta. El organismo vive
  igual. Nunca propaga una excepción hacia arriba.

CONFIGURACIÓN
-------------
  ANIMA_BITACORA          1 (por defecto) | 0 para apagarlo
  ANIMA_BITACORA_URL      http://192.168.86.205:9110/api/resumen
  ANIMA_BITACORA_VENTANA_S  30
  ANIMA_BITACORA_CLAVE    opcional; sólo hace falta si se llega desde fuera de la LAN
"""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import Counter

__all__ = ["BitacoraSocial", "COLS_BITACORA"]

# Columna extra que este módulo aporta al CSV: a quién escuchó DE VERDAD el
# organismo (el objetivo que el hilo aplicador llevó a la fuente de audio).
COLS_BITACORA = ["as_aplicado"]

_UA = "ANIMA-Organismo/1.2 (+https://cosmosemiotica.cl)"


def _num(v):
    try:
        f = float(v)
        return f if f == f else None          # descarta NaN
    except (TypeError, ValueError):
        return None


class BitacoraSocial:
    """Acumula la ventana social del organismo y la empuja a la Bitácora."""

    def __init__(self, organismo_id: str, nombre: str | None = None) -> None:
        self.oid = organismo_id
        self.nombre = nombre or organismo_id
        self.activo = os.environ.get("ANIMA_BITACORA", "1") not in ("0", "false", "no")
        self.url = os.environ.get(
            "ANIMA_BITACORA_URL", "http://192.168.86.205:9110/api/resumen").strip()
        self.clave = os.environ.get("ANIMA_BITACORA_CLAVE", "").strip()
        try:
            self.ventana_s = max(5.0, float(os.environ.get("ANIMA_BITACORA_VENTANA_S", "30")))
        except ValueError:
            self.ventana_s = 30.0

        self._lock = threading.Lock()
        self._reiniciar_acumulador()
        self._enviados = 0
        self._fallos = 0
        self._ultimo_error = ""
        self._espera = 0.0            # backoff tras un fallo

        if self.activo and self.url:
            threading.Thread(target=self._bucle, daemon=True,
                             name="anima-bitacora-social").start()

    # ── acumulación (se llama en cada tick; O(1)) ───────────────────────────
    def _reiniciar_acumulador(self) -> None:
        self._t0 = time.time()
        self._n = 0
        self._escucha = Counter()
        self._habla = Counter()
        self._aplicado = Counter()
        self._sesgos = Counter()
        self._n_atendiendo = 0
        self._n_explorando = 0
        self._dwell: list[float] = []
        self._candidatos: list[int] = []
        # Léxico: contadores del repertorio vocal. Son monótonos, así que basta el
        # último valor de la ventana; su serie da la curva de crecimiento léxico.
        # No viajan en el heartbeat de plaza (el Observatorio sólo reenvía 11 campos),
        # así que ésta es la única vía por la que se pueden observar desde fuera.
        self._lexico: dict[str, float] = {}

    def anotar(self, fila: dict, aplicado: str | None = None) -> None:
        """Registra un tick. `fila` es lo que devolvió OrganoAtencionSocial.observar();
        `aplicado` es el objetivo que el hilo aplicador tiene puesto en el audio."""
        if not self.activo:
            return
        try:
            with self._lock:
                self._n += 1
                esc = fila.get("as_esc_objetivo")
                hab = fila.get("as_habla_objetivo")
                if esc:
                    self._escucha[esc] += 1
                    self._n_atendiendo += 1
                if hab:
                    self._habla[hab] += 1
                if aplicado:
                    self._aplicado[aplicado] += 1
                sesgo = fila.get("as_esc_sesgo")
                if sesgo and sesgo != "-":
                    self._sesgos[sesgo] += 1
                if _num(fila.get("as_esc_explorando")):
                    self._n_explorando += 1
                d = _num(fila.get("as_esc_dwell_s"))
                if d is not None:
                    self._dwell.append(d)
                c = _num(fila.get("as_n_candidatos"))
                if c is not None:
                    self._candidatos.append(int(c))
                self._tomar_lexico(fila)
        except Exception:
            pass                                  # jamás estorbar al organismo

    def _tomar_lexico(self, fila: dict) -> None:
        for k in ("voz_creadas", "voz_propias", "voz_estables",
                  "voz_aprendidas", "voz_aprendidas_forma", "voz_arousal", "voz_valence"):
            v = _num(fila.get(k))
            if v is not None:
                self._lexico[k] = v

    def anotar_lexico(self, fila: dict) -> None:
        """Anota SÓLO el repertorio vocal, desde una fila ya completa.

        `anotar()` se llama al principio del ciclo de observación, antes de que se
        rellenen las columnas de voz: ahí el léxico todavía no existe. Este método
        se llama al final, cuando la fila ya está entera.
        """
        if not self.activo:
            return
        try:
            with self._lock:
                self._tomar_lexico(fila)
        except Exception:
            pass

    # ── envío ───────────────────────────────────────────────────────────────
    def _cerrar_ventana(self) -> dict | None:
        with self._lock:
            if self._n == 0:
                self._reiniciar_acumulador()
                return None
            n = self._n
            dur = max(1e-6, time.time() - self._t0)

            def reparto(c: Counter) -> dict:
                tot = sum(c.values()) or 1
                return {k: round(v / tot, 4) for k, v in c.most_common(12)}

            resumen = {
                "oid": self.oid,
                "nombre": self.nombre,
                "ts": time.time(),
                "ventana_s": round(dur, 2),
                "n_ticks": n,
                "frac_atendiendo": round(self._n_atendiendo / n, 4),
                "frac_explorando": round(self._n_explorando / n, 4),
                "n_candidatos": int(statistics.median(self._candidatos)) if self._candidatos else 0,
                "sesgo_dominante": self._sesgos.most_common(1)[0][0] if self._sesgos else None,
                "dwell_mediana_s": round(statistics.median(self._dwell), 3) if self._dwell else None,
                "dwell_p90_s": round(sorted(self._dwell)[int(len(self._dwell) * 0.9)], 3)
                if self._dwell else None,
                "reparto_escucha": reparto(self._escucha),
                "reparto_habla": reparto(self._habla),
                "sesgos": reparto(self._sesgos),
                # lo que DE VERDAD sonó: el objetivo que el aplicador puso en el audio
                "reparto_aplicado": reparto(self._aplicado),
                # estado del repertorio vocal al cerrar la ventana
                "lexico": dict(self._lexico),
            }
            self._reiniciar_acumulador()
            return resumen

    def _empujar(self, resumen: dict) -> bool:
        cuerpo = json.dumps(resumen, ensure_ascii=False).encode("utf-8")
        cab = {"Content-Type": "application/json", "User-Agent": _UA}
        if self.clave:
            cab["X-Bitacora-Clave"] = self.clave
        pet = urllib.request.Request(self.url, data=cuerpo, headers=cab, method="POST")
        with urllib.request.urlopen(pet, timeout=5) as r:
            return 200 <= r.status < 300

    def _bucle(self) -> None:
        time.sleep(self.ventana_s)                # no enviar una ventana a medio llenar
        while True:
            try:
                time.sleep(self.ventana_s + self._espera)
                resumen = self._cerrar_ventana()
                if resumen is None:
                    continue
                if self._empujar(resumen):
                    self._enviados += 1
                    self._espera = 0.0
                else:
                    raise RuntimeError("respuesta no OK")
            except Exception as exc:
                self._fallos += 1
                self._ultimo_error = f"{type(exc).__name__}: {exc}"
                # backoff exponencial hasta 5 min: si la Bitácora está caída, no insistir
                self._espera = min(300.0, (self._espera or self.ventana_s) * 2)

    # ── diagnóstico ─────────────────────────────────────────────────────────
    def estado(self) -> dict:
        return {"activo": self.activo, "url": self.url, "ventana_s": self.ventana_s,
                "enviados": self._enviados, "fallos": self._fallos,
                "ultimo_error": self._ultimo_error, "espera_s": self._espera,
                "ticks_en_curso": self._n}
