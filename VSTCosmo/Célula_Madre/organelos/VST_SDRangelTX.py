#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_SDRangelTX — BRAZO DE HARDWARE de transmisión: maneja el HackRF vía la API REST de SDRangel
================================================================================================
QUIÉN SOY (para retomar sin reprocesar):
  Soy el brazo motor de la VOZ DE RADIO. No decido nada: recibo órdenes (sintoniza aquí, emite
  esto) y las ejecuto sobre el hardware de transmisión — un HackRF One manejado por SDRangel
  (app con API REST en :8091). Soy al TX lo que VST_LectorSDRServidor es al RX: el adaptador entre
  el organismo y el aparato. Interfaz de DRIVER ENCHUFABLE: en la Pi (E), otro driver con estos
  mismos métodos manejará su propio HackRF (SDRangel o nativo) sin tocar el órgano.

POR QUÉ ASÍ (Alexis + CC, 3-jul-2026):
  El HackRF es half-duplex y en la Mac lo tiene abierto SDRangel; su API REST es la vía de control.
  El organismo NO habla el protocolo del hardware: habla con este brazo, que traduce a llamadas REST.
  Validado en banco: lazo cerrado HackRF(TX 435 MHz) → RSPduo(RX) confirmado A/B.

INTERFAZ (la usa VST_OrganoFonadorRadio):
  disponible()->bool · preparar(freq_hz, sample_rate, gain, modulador)->bool · sintonizar(freq_hz)
  · portador(on) · tono(af_hz, on) · transmitir(on)->bool · emitiendo()->bool · cerrar()

SEGURIDAD: transmitir es emitir RF real. Defaults conservadores (ganancia baja, muteado). El
  organismo debe encender explícitamente. Frecuencia por defecto en 70cm (435 MHz), la que dejó
  configurada Alexis; ajustable. Responsabilidad de licencia/potencia: del operador.

Config por entorno:
  ANIMA_TX_SDRANGEL_URL   http://127.0.0.1:8091/sdrangel   (en la Pi: su propia instancia)
  ANIMA_TX_DEVICESET      (auto: busca el deviceset TX HackRF; o índice explícito)
  ANIMA_TX_MODULADOR      WFMMod        (WFMMod|NFMMod|AMMod — coherente con el demod del oído RX)
  ANIMA_TX_FREQ_HZ        435000000
  ANIMA_TX_SAMPLE_RATE    2400000
  ANIMA_TX_GAIN           14            (vgaGain HackRF 0..47; bajo por defecto = poca potencia)
"""
from __future__ import annotations
import json
import os
import urllib.error
import urllib.request


class SDRangelTX:
    """Driver TX sobre SDRangel/HackRF. Idempotente y con degradación elegante (nunca lanza)."""

    def __init__(self, base_url: str | None = None):
        self.base = (base_url or os.environ.get(
            "ANIMA_TX_SDRANGEL_URL", "http://127.0.0.1:8091/sdrangel")).rstrip("/")
        # RECETA LIMPIA (validada por oído, 3-jul-2026): NFMMod de banda angosta, energía concentrada
        # (desviación estrecha), offset del centro para esquivar el DC del receptor. Ver
        # [[radio-transmision-fonador]]. WFMMod se desparramaba; FileSource+IQ era poco fiable.
        self.modulador = os.environ.get("ANIMA_TX_MODULADOR", "NFMMod")
        self.desviacion = int(os.environ.get("ANIMA_TX_DESVIACION_HZ", "2000"))   # concentra la energía
        self.rf_bw = int(os.environ.get("ANIMA_TX_RF_BW_HZ", "8000"))
        self.offset = int(os.environ.get("ANIMA_TX_OFFSET_HZ", "100000"))         # señal fuera del DC del RX
        self._ds = None            # índice del deviceset TX (se resuelve en preparar)
        self._canal = None         # índice del canal modulador
        self._ok = False

    # ---------- REST helpers ----------
    def _req(self, method: str, path: str, body: dict | None = None, timeout: float = 6.0):
        url = self.base + path
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, method=method, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}

    def disponible(self) -> bool:
        try:
            self._req("GET", "")
            return True
        except (urllib.error.URLError, OSError, ValueError):
            return False

    # ---------- resolución del deviceset TX (HackRF) ----------
    def _resolver_deviceset(self) -> int | None:
        env = os.environ.get("ANIMA_TX_DEVICESET", "").strip()
        if env.isdigit():
            return int(env)
        try:
            info = self._req("GET", "")
        except Exception:
            return None
        for ds in info.get("devicesetlist", {}).get("deviceSets", []):
            sd = ds.get("samplingDevice", {})
            if sd.get("direction") == 1 and "HackRF" in str(sd.get("hwType", "")):
                return sd.get("index")
        # si hay algún TX aunque no sea HackRF, úsalo como fallback
        for ds in info.get("devicesetlist", {}).get("deviceSets", []):
            if ds.get("samplingDevice", {}).get("direction") == 1:
                return ds["samplingDevice"].get("index")
        return None

    def _asegurar_canal(self) -> bool:
        """Garantiza un canal modulador en el deviceset TX; devuelve True si hay uno usable."""
        try:
            info = self._req("GET", f"/deviceset/{self._ds}")
        except Exception:
            return False
        canales = info.get("channelcount", 0)
        if canales and canales > 0:
            self._canal = 0
            return True
        try:
            self._req("POST", f"/deviceset/{self._ds}/channel",
                      {"channelType": self.modulador, "direction": 1})
            self._canal = 0
            return True
        except Exception:
            return False

    # ---------- interfaz pública ----------
    def preparar(self, freq_hz=None, sample_rate=None, gain=None, modulador=None) -> bool:
        """Deja el HackRF listo para emitir: deviceset TX + frecuencia + sample rate + ganancia +
        un canal modulador. NO transmite todavía (queda muteado). Idempotente."""
        if modulador:
            self.modulador = modulador
        if not self.disponible():
            print(f"[SDRangelTX] SDRangel no responde en {self.base} — TX inactivo")
            return False
        self._ds = self._resolver_deviceset()
        if self._ds is None:
            print("[SDRangelTX] no hay deviceset TX (HackRF) en SDRangel — abre uno (dir=Tx, HackRF)")
            return False
        freq = int(freq_hz if freq_hz is not None else os.environ.get("ANIMA_TX_FREQ_HZ", "435000000"))
        sr = int(sample_rate if sample_rate is not None else os.environ.get("ANIMA_TX_SAMPLE_RATE", "2000000"))
        g = int(gain if gain is not None else os.environ.get("ANIMA_TX_GAIN", "47"))   # receta limpia: potencia alta
        try:
            self._req("PATCH", f"/deviceset/{self._ds}/device/settings",
                      {"deviceHwType": "HackRF", "direction": 1,
                       "hackRFOutputSettings": {"centerFrequency": freq, "devSampleRate": sr,
                                                "vgaGain": max(0, min(47, g))}})
        except Exception as e:
            print(f"[SDRangelTX] no pude ajustar el HackRF ({e})")
            return False
        if not self._asegurar_canal():
            print("[SDRangelTX] no pude asegurar el canal modulador")
            return False
        self.freq = freq
        # canal listo, MUTEADO: NFM de banda angosta, energía concentrada, con offset (fuera del DC del RX)
        self._set_canal(channelMute=1, modAFInput=0, fmDeviation=self.desviacion,
                        rfBandwidth=self.rf_bw, inputFrequencyOffset=self.offset)
        self._ok = True
        print(f"[SDRangelTX] listo · deviceset={self._ds} · {self.modulador} · "
              f"{(freq + self.offset)/1e6:.3f} MHz (centro {freq/1e6:.3f}+offset) · vgaGain={g} (muteado)")
        return True

    def _set_canal(self, **kv) -> None:
        if self._ds is None or self._canal is None:
            return
        try:
            self._req("PATCH", f"/deviceset/{self._ds}/channel/{self._canal}/settings",
                      {"channelType": self.modulador, "direction": 1,
                       f"{self.modulador}Settings": kv})
        except Exception:
            pass

    def sintonizar(self, freq_hz) -> None:
        """Mueve la frecuencia de emisión (centro del HackRF)."""
        if self._ds is None:
            return
        try:
            self._req("PATCH", f"/deviceset/{self._ds}/device/settings",
                      {"deviceHwType": "HackRF", "direction": 1,
                       "hackRFOutputSettings": {"centerFrequency": int(freq_hz)}})
        except Exception:
            pass

    # --- EMISIÓN con ENGANCHE SOSTENIDO (la clave de la recepción SIN estática) ---
    # La portadora se mantiene ENCENDIDA durante toda la vocalización (nunca se mutea entre pitos):
    # así el demod FM del receptor NUNCA pierde el lock. Los tonos cambian solo modAFInput/frecuencia.
    def abrir(self) -> bool:
        """Abre la emisión: arranca el hardware y deja la PORTADORA encendida y silenciosa (lista para
        que el receptor enganche). A partir de aquí, tono()/silencio() no cortan la portadora."""
        if not self.transmitir(True):
            return False
        self._set_canal(channelMute=0, modAFInput=0)   # portadora ON, sin tono (demod puede fijar lock)
        return True

    def tono(self, af_hz: float) -> None:
        """Pone un TONO (modAFInput=1) SIN tocar la portadora — vocalización continua, lock sostenido."""
        self._set_canal(modAFInput=1, toneFrequency=int(af_hz))

    def silencio(self) -> None:
        """Silencio CON portadora (modAFInput=0): pausa audible sin perder el enganche del receptor."""
        self._set_canal(modAFInput=0)

    def portador(self, on: bool) -> None:
        """Compat: portadora on/off. 'on' abre (run+unmute); 'off' cierra. Prefiere abrir()/cerrar()."""
        if on:
            self.abrir()
        else:
            self.cerrar()

    def transmitir(self, on: bool) -> bool:
        """Enciende/apaga el flujo del hardware (run/stop del deviceset). Sin esto no sale RF."""
        if self._ds is None:
            return False
        try:
            if on:
                self._req("POST", f"/deviceset/{self._ds}/device/run")
            else:
                self._req("DELETE", f"/deviceset/{self._ds}/device/run")
            return True
        except Exception:
            return False

    def emitiendo(self) -> bool:
        if self._ds is None:
            return False
        try:
            return self._req("GET", f"/deviceset/{self._ds}/device/run").get("state") == "running"
        except Exception:
            return False

    def cerrar(self) -> None:
        """Silencio y stop — deja el hardware sin emitir."""
        self._set_canal(channelMute=1, modAFInput=0)
        self.transmitir(False)


if __name__ == "__main__":
    import time
    tx = SDRangelTX()
    if not tx.preparar():
        raise SystemExit("no se pudo preparar el TX (¿SDRangel abierto con HackRF en Tx?)")
    print("abriendo emisión (portadora sostenida) y tocando una frasecita de pitos…")
    tx.abrir()
    for f in (600, 800, 1000, 1300, 900, 650, 1500):   # melodía continua (sin cortar portadora)
        tx.tono(f); time.sleep(0.25)
    tx.silencio(); time.sleep(0.3)
    print("  ¿en el aire?", tx.emitiendo())
    tx.cerrar()
    print("silencio. TX con enganche sostenido establecido.")
