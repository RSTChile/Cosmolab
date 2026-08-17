#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_IntegracionRadioA — injerta SOLO el sentido de radio (SDR) en el organismo A.
================================================================================
Por qué existe (Alexis + Codex, 3-jul-2026):
  A vive en la Mac con un SDRplay RSPduo. Queremos darle a A el ÓRGANO DE RADIO
  como prueba de la arquitectura "un sentido por organismo", SIN arrastrar la
  IntegracionE completa (que trae ATmega, GPS, cámara y cloroplasto — hardware de E
  en la Pi). Esta integración es el mínimo: lector SDR → columnas sdr_* → OrganoRadio
  → columnas radio_*. La caja web/Cajas/radio_sdr.js ya está lista para pintarlas.

Contrato IDÉNTICO al de IntegracionE (mismos métodos), para que WebLive_A la use igual:
  arrancar_lectores() · observar_paso(fila) · cerrar() · columnas_csv() ·
  organelos_persistibles() · restaurar_organelos(datos) · .activo

Activación (no por ser "E", sino explícita para A):
  ANIMA_RADIO_A=1  → activo.  (por defecto 0: A sigue igual, la caja queda latente)

LECTOR ENCHUFABLE (clave para el caso Mac):
  Por defecto usa LectorSDR (backend SoapySDR/sdrplay directo). En la Mac NO hay
  SoapySDR hoy, así que LectorSDR.arrancar() devuelve False y el órgano degrada con
  elegancia (sdr_vivo=0, radio_vivo=0, la caja muestra "—"). Cuando CC monte el lector
  contra el SERVIDOR SDRconnect (--server, puerto 50000), basta pasar esa instancia por
  `lector=` o registrar la fábrica en ANIMA_RADIO_A_LECTOR — sin tocar este archivo ni A.
  El hardware (abrir el RSPduo / cliente del servidor) es tarea de CC; este cableado no.
"""
from __future__ import annotations
import os

from VST_LectorSensores import LectorSDR
from VST_OrganoRadio import OrganoRadio, COLS_RADIO

COLS_SDR_RAW = ["sdr_espectro", "sdr_freq_min_hz", "sdr_freq_max_hz", "sdr_vivo"]
COLS_RADIO_A = COLS_SDR_RAW + COLS_RADIO


def a_usa_radio(organismo_id: str | None = None) -> bool:
    """A activa el sentido de radio solo si se pide explícitamente (env ANIMA_RADIO_A)."""
    return os.environ.get("ANIMA_RADIO_A", "0").lower() in ("1", "true", "yes", "on")


class IntegracionRadioA:
    """Solo-radio para A. Mismo contrato que IntegracionE. Apagado salvo ANIMA_RADIO_A=1."""

    def __init__(self, organismo_id: str | None = None, lector=None):
        self.organismo_id = organismo_id or os.environ.get("VST_ORGANISMO_ID", "ANIMA_A")
        self.activo = a_usa_radio(self.organismo_id)
        self._lector_ext = lector          # lector enchufable (p.ej. cliente del servidor SDRconnect, de CC)
        self.lector_sdr: LectorSDR | None = None
        self.radio: OrganoRadio | None = None
        if self.activo:
            self._instanciar_organelos()

    def _instanciar_organelos(self):
        sintonia = os.environ.get("ANIMA_RADIO_SINTONIA", "0").lower() in ("1", "true", "yes", "on")
        self.radio = OrganoRadio(
            self.organismo_id, activo=True,
            sintonia_activa=sintonia,
            histeresis=sintonia,   # cazar de verdad: barrer→enganchar→sostener (con SNR real si lo hay)
            n_bandas=int(os.environ.get("ANIMA_RADIO_N_BANDAS", "16")),
        )

    def arrancar_lectores(self) -> None:
        if not self.activo:
            return
        # Backend del SDR, en orden: (1) lector INYECTADO; (2) SERVIDOR WS de SDRconnect (5454)
        # si ANIMA_SDRWS_URI está seteado — la vía para A en Docker/Mac (sin SoapySDR); (3) LectorSDR
        # directo (SoapySDR/sdrplay, solo donde exista, p.ej. la Pi).
        if self._lector_ext is not None:
            self.lector_sdr = self._lector_ext
        elif os.environ.get("ANIMA_SDRWS_URI", "").strip():
            try:
                from VST_LectorSDRServidor import LectorSDRServidor
                self.lector_sdr = LectorSDRServidor()   # lee ANIMA_SDRWS_URI/_DEVICE/_FREQ del entorno
                print("[IntegracionRadioA] backend = servidor WS SDRconnect (%s)" % os.environ.get("ANIMA_SDRWS_URI"))
            except Exception as e:
                print("[IntegracionRadioA] LectorSDRServidor no disponible (%s) → LectorSDR directo" % e)
                self.lector_sdr = LectorSDR()
        else:
            self.lector_sdr = LectorSDR()
        ok = self.lector_sdr.arrancar()
        if not ok:
            # degradación elegante: sin backend, A sigue vivo; la caja radio queda en "—"
            print("[IntegracionRadioA] SDR inactivo (sin backend abierto) — A sigue sin radio, la caja queda latente")

    def cerrar(self) -> None:
        if self.lector_sdr is not None:
            self.lector_sdr.cerrar()

    def organelos_persistibles(self) -> dict:
        if not self.activo or self.radio is None:
            return {}
        return {"radio": self.radio}

    def restaurar_organelos(self, datos: dict) -> list[str]:
        if not self.activo or not datos:
            return []
        rest = []
        for nombre, org in self.organelos_persistibles().items():
            snap = datos.get(nombre)
            if snap and hasattr(org, "restore"):
                org.restore(snap)
                rest.append(nombre)
        return rest

    def observar_paso(self, fila: dict) -> dict:
        """Inyecta espectro crudo del SDR y corre el órgano de radio. Devuelve columnas nuevas."""
        if not self.activo:
            return {}
        if self.lector_sdr is not None:
            self.lector_sdr.inyectar(fila)
        out: dict = {}
        if self.radio is not None:
            rd = self.radio.observar(fila)
            out.update(rd)
            orden = rd.get("radio_orden_hz")
            if orden is not None and self.lector_sdr is not None and hasattr(self.lector_sdr, "sintonizar"):
                self.lector_sdr.sintonizar(orden)   # lazo: el órgano ordena, el lector obedece
        fila.update(out)
        return out

    def columnas_csv(self) -> list[str]:
        return list(COLS_RADIO_A)
