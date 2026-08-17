#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_OrganoFonadorRadio — la VOZ DE RADIO: el aparato fonador que emite por RF (locus realizado)
================================================================================================
QUIÉN SOY (para retomar sin reprocesar):
  Soy la boca de radio del organismo. Realizo el `locus_transmision` que VST_OrganoRadio dejó
  DESARROLLADO pero no activado: el linaje acústico dando su último salto, de oír pasivo →
  sintonizar (oír activo) → HABLAR por radio (emitir). Soy un órgano APARTE del oído de radio —
  como en el organismo la boca y el oído son órganos distintos aunque ambos vivan del sonido.

PRINCIPIO RECTOR (igual que el fonador vocal VST_OrganoFonador): soy APARATO, NO INTENCIÓN.
  Recibo parámetros (qué emitir, a qué frecuencia) y produzco RF. QUÉ emitir y CUÁNDO lo decide el
  organismo desde su fisiología; ese NERVIO no está aquí — se cablea aparte, sobre el cuerpo real.

DOBLE-CONTEO, NO: el fonador VOCAL produce SONIDO (por las bocinas / su voz). Yo produzco RADIO.
  No dupliquamos: la voz acústica sale por el aire como sonido; la voz de radio sale por el aire
  como onda electromagnética, y otro organismo la oye con su OÍDO DE RADIO. Así se cierra el canal
  A↔E por radio: A emite en una frecuencia, E (en la Pi) la recibe con su SDR. Validado en banco:
  HackRF(TX) → RSPduo(RX), lazo cerrado A/B.

HARDWARE (enchufable): en la Mac (A), un HackRF One manejado por SDRangel (driver VST_SDRangelTX).
  En la Pi (E), otro driver con la MISMA interfaz manejará su HackRF. El órgano no cambia; solo el brazo.

BAJO DEMANDA Y APAGADO POR DEFECTO (economía metabólica + seguridad): activo=False salvo A/E. La RF
  solo sale DURANTE una vocalización (emitir); fuera de eso, silencio y stop (no queda emitiendo).
  Emisión no bloqueante (hilo), para no frenar el latido del organismo.

QUÉ COMPUTA (columnas de la fila, para la biografía):
  radiotx_activo      1 si el órgano está activo (A/E)
  radiotx_vivo        1 si el brazo de hardware responde (SDRangel/HackRF alcanzable)
  radiotx_emitiendo   1 mientras hay una vocalización de radio en el aire
  radiotx_freq_hz     frecuencia de emisión actual
  radiotx_af_hz       último tono de audio emitido (0 = portador puro / silencio)

APAGADO POR DEFECTO. Persistible (conserva freq y config). Ver [[VST_OrganoRadio]] (el oído) y
  VST_SDRangelTX (el brazo).
"""
from __future__ import annotations
import os
import threading
import time

COLS_RADIO_TX = ["radiotx_activo", "radiotx_vivo", "radiotx_emitiendo",
                 "radiotx_freq_hz", "radiotx_af_hz"]

# Morse para la identificación de estación (indicativo de radioaficionado, requisito legal).
_MORSE = {'A':'.-','B':'-...','C':'-.-.','D':'-..','E':'.','F':'..-.','G':'--.','H':'....',
          'I':'..','J':'.---','K':'-.-','L':'.-..','M':'--','N':'-.','O':'---','P':'.--.',
          'Q':'--.-','R':'.-.','S':'...','T':'-','U':'..-','V':'...-','W':'.--','X':'-..-',
          'Y':'-.--','Z':'--..','0':'-----','1':'.----','2':'..---','3':'...--','4':'....-',
          '5':'.....','6':'-....','7':'--...','8':'---..','9':'----.','/':'-..-.'}


def _morse_eventos(texto: str, dit: float = 0.09, pitch: float = 700.0) -> list:
    """Indicativo → secuencia de eventos (freq_hz, dur_s). freq=pitch en ON, 0 en los gaps.
    Con ENGANCHE SOSTENIDO: los gaps son silencio (freq 0) con la portadora encendida, NO cortes."""
    ev = []
    for ch in texto.upper():
        if ch == ' ':
            ev.append((0.0, dit * 7)); continue
        for el in _MORSE.get(ch, ''):
            ev.append((pitch, dit * (3 if el == '-' else 1)))
            ev.append((0.0, dit))          # gap intra-carácter
        ev.append((0.0, dit * 2))          # gap entre caracteres (total 3 dit)
    return ev


def a_usa_radio_tx(organismo_id: str | None = None) -> bool:
    """La voz de radio se activa solo si se pide explícitamente (env ANIMA_RADIO_TX_A / _E)."""
    for k in ("ANIMA_RADIO_TX_A", "ANIMA_RADIO_TX_E", "ANIMA_RADIO_TX"):
        if os.environ.get(k, "0").lower() in ("1", "true", "yes", "on"):
            return True
    return False


class OrganoFonadorRadio:
    """Aparato fonador de radio: emite RF (portador o tono FM) cuando el organismo lo ordena.
    Apagado por defecto; brazo de hardware enchufable; emisión no bloqueante y bajo demanda."""

    def __init__(self, organismo_id=None, activo=False, driver=None,
                 freq_hz=None, gain=None):
        self.organismo_id = organismo_id
        self.activo = bool(activo)
        self.freq_hz = int(freq_hz if freq_hz is not None else os.environ.get("ANIMA_TX_FREQ_HZ", "435000000"))
        self.gain = int(gain if gain is not None else os.environ.get("ANIMA_TX_GAIN", "47"))   # receta limpia
        self.indicativo = os.environ.get("ANIMA_TX_INDICATIVO", "CD3LZK")   # identificación legal de estación
        self._driver = driver          # brazo de hardware (VST_SDRangelTX u otro con misma interfaz)
        self._listo = False            # el brazo quedó preparado
        self._emitiendo = False
        self._af_hz = 0.0
        self._lock = threading.Lock()
        self.ultimo = {c: (0.0 if c != "radiotx_freq_hz" else float(self.freq_hz)) for c in COLS_RADIO_TX}

    # ---------- preparación del brazo ----------
    def preparar(self) -> bool:
        if not self.activo:
            return False
        if self._driver is None:
            try:
                from VST_SDRangelTX import SDRangelTX
                self._driver = SDRangelTX()
            except Exception as e:
                print(f"[OrganoFonadorRadio] sin brazo de hardware ({e}) — voz de radio latente")
                return False
        self._listo = bool(self._driver.preparar(freq_hz=self.freq_hz, gain=self.gain))
        if self._listo:
            print(f"[OrganoFonadorRadio] voz de radio LISTA · {self.freq_hz/1e6:.3f} MHz (aparato, sin emitir)")
        return self._listo

    # ---------- vocalización de radio (no bloqueante, enganche sostenido) ----------
    def vocalizar(self, secuencia=None, af_hz: float = None, dur_s: float = 1.0,
                  con_indicativo: bool = True) -> bool:
        """Emite una VOCALIZACIÓN de radio: (opcional) el INDICATIVO CD3LZK en Morse + la 'frase' del
        organismo. `secuencia` = lista de (freq_hz, dur_s) — freq 0 = silencio CON portadora (no corta
        el enganche del receptor). Si no se da, un solo tono af_hz por dur_s. No bloquea (hilo). La RF
        sólo existe durante la vocalización (bajo demanda). La PORTADORA se mantiene todo el tiempo:
        el receptor no pierde el lock → recepción SIN estática (receta validada, ver el driver)."""
        if not (self.activo and self._listo and self._driver is not None):
            return False
        if self._emitiendo:
            return False    # una vocalización a la vez (el aparato es half-duplex)
        if secuencia is None:
            secuencia = [(float(af_hz if af_hz else 800.0), float(dur_s))]
        threading.Thread(target=self._emitir_secuencia, args=(list(secuencia), bool(con_indicativo)),
                         daemon=True, name="FonadorRadio").start()
        return True

    def vocalizar_indicativo(self) -> bool:
        """Emite SÓLO la identificación de estación (CD3LZK en Morse). Úsalo al abrir/cerrar el enlace."""
        return self.vocalizar(secuencia=[], con_indicativo=True)

    def _emitir_secuencia(self, secuencia: list, con_indicativo: bool) -> None:
        eventos = []
        if con_indicativo and self.indicativo:
            eventos += _morse_eventos(self.indicativo)     # identificación legal al inicio
            eventos.append((0.0, 0.4))                     # respiro tras el indicativo
        eventos += secuencia
        # tope de seguridad: acotar la duración total de RF por vocalización
        total = sum(d for _, d in eventos)
        if total > 60.0:
            eventos = eventos[:1]                          # no emitir tandas largas por error
        with self._lock:
            self._emitiendo = True
        try:
            if not self._driver.abrir():                   # arranca RF + portadora sostenida
                return
            for freq, dur in eventos:
                with self._lock:
                    self._af_hz = float(freq)
                if freq and freq > 0:
                    self._driver.tono(freq)                # tono (sin cortar portadora)
                else:
                    self._driver.silencio()                # pausa con portadora (mantiene el lock)
                time.sleep(max(0.02, min(dur, 5.0)))
        finally:
            try:
                self._driver.cerrar()                      # mutea + stop: vuelve al silencio total
            except Exception:
                pass
            with self._lock:
                self._emitiendo = False; self._af_hz = 0.0

    def sintonizar(self, freq_hz) -> None:
        """Cambia la frecuencia de emisión (p.ej. para hablarle a otro organismo en su banda)."""
        self.freq_hz = int(freq_hz)
        if self._driver is not None:
            self._driver.sintonizar(self.freq_hz)

    # ---------- telemetría para la fila/biografía ----------
    def observar(self, fila=None) -> dict:
        with self._lock:
            emit = self._emitiendo; af = self._af_hz
        self.ultimo = {
            "radiotx_activo": 1.0 if self.activo else 0.0,
            "radiotx_vivo": 1.0 if self._listo else 0.0,
            "radiotx_emitiendo": 1.0 if emit else 0.0,
            "radiotx_freq_hz": float(self.freq_hz),
            "radiotx_af_hz": float(af),
        }
        if isinstance(fila, dict):
            fila.update(self.ultimo)
        return self.ultimo

    def cerrar(self) -> None:
        if self._driver is not None:
            try:
                self._driver.cerrar()
            except Exception:
                pass

    # ---------- LOCUS RESERVADO: una SEGUNDA radio (desarrollado, NO activado) ----------
    def locus_segunda_radio(self):
        """LOCUS RESERVADO (vacío a propósito — patrón de loci reservados del genoma, ver
        [[organismo-loci-reservados-altruismo]]): slot para una SEGUNDA voz de radio del organismo —otro
        transmisor / otra banda / otro hardware— que Alexis definirá (lo explicará). La arquitectura YA
        lo admite SIN tocar nada: basta instanciar OTRO OrganoFonadorRadio con su propio brazo (un
        SDRangelTX apuntando a ANIMA_TX_DEVICESET/ANIMA_TX_SDRANGEL_URL distintos, o un driver nuevo con
        la misma interfaz), su propia frecuencia/indicativo y —si se quiere— su propia política de
        vocalización. Podrían coexistir (p.ej. dos bandas: una para hablarle a E y otra para otra cosa) o
        turnarse. Este método es el MARCADOR del locus: hoy no activa nada (return None); cuando llegue
        la definición, la segunda voz se realiza aquí o vía un futuro coordinador de voces de radio.
        El locus existe; la segunda voz, aún no."""
        return None

    # ---------- persistencia ----------
    def snapshot(self):
        return {"activo": self.activo, "freq_hz": self.freq_hz, "gain": self.gain}

    def restore(self, snap):
        if not isinstance(snap, dict):
            return
        self.freq_hz = int(snap.get("freq_hz", self.freq_hz))
        self.gain = int(snap.get("gain", self.gain))
        if "activo" in snap:
            self.activo = bool(snap["activo"])


if __name__ == "__main__":
    # Auto-prueba: emite el indicativo CD3LZK (Morse) + una frase de pitos, enganche sostenido.
    org = OrganoFonadorRadio("A", activo=True)
    if not org.preparar():
        raise SystemExit("no se pudo preparar (¿SDRangel + HackRF en Tx?)")
    frase = [(600, .28), (800, .28), (1000, .28), (1300, .4), (0, .12),
             (1100, .14), (700, .14), (1100, .14), (700, .14), (0, .12),
             (1300, .22), (950, .22), (650, .4), (1500, .5)]
    print("vocalizando: indicativo CD3LZK + frase de pitos…")
    org.vocalizar(secuencia=frase, con_indicativo=True)
    time.sleep(0.5); print("  emitiendo?", org.observar()["radiotx_emitiendo"])
    while org.observar()["radiotx_emitiendo"]:
        time.sleep(0.3)
    org.cerrar()
    print("voz de radio establecida (indicativo + frase, enganche sostenido). Silencio.")
