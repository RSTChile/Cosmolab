#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apariencia de la cabeza 3D — personalización al instalar el organismo."""

from __future__ import annotations

import os
import re

NOMBRE_MAX_LEN = 14
_PAT_NOMBRE = re.compile(r"[^0-9A-Za-záéíóúÁÉÍÓÚñÑüÜ ]")

TONOS_VALIDOS = frozenset({"blanco", "celeste", "rosado", "amarillo", "cafe", "negro"})
GENEROS_VALIDOS = frozenset({"masculino", "femenino"})

_ALIASES_GENERO = {
    "m": "masculino", "masculino": "masculino", "male": "masculino",
    "f": "femenino", "femenino": "femenino", "female": "femenino",
}
_ALIASES_TONO = {
    "blanco": "blanco", "white": "blanco",
    "celeste": "celeste", "azul": "celeste", "sky": "celeste",
    "rosado": "rosado", "rosa": "rosado", "pink": "rosado",
    "amarillo": "amarillo", "yellow": "amarillo",
    "cafe": "cafe", "café": "cafe", "marron": "cafe", "marrón": "cafe", "brown": "cafe",
    "negro": "negro", "black": "negro", "oscuro": "negro", "moreno": "cafe",
}


def normalizar_nombre_organismo(raw: str | None, *, default: str = "Animalito") -> str:
    """Nombre visible al instalar: letras, números y espacios; máx. 14 caracteres."""
    s = (raw or default).strip()
    s = _PAT_NOMBRE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        s = default
    return s[:NOMBRE_MAX_LEN]


def normalizar_genero(raw: str | None) -> str:
    g = (raw or "masculino").strip().lower()
    return _ALIASES_GENERO.get(g, g if g in GENEROS_VALIDOS else "masculino")


def normalizar_tono(raw: str | None) -> str:
    t = (raw or "blanco").strip().lower()
    return _ALIASES_TONO.get(t, t if t in TONOS_VALIDOS else "blanco")


def apariencia_desde_identity(identity: dict | None = None) -> dict[str, str]:
    ident = identity if isinstance(identity, dict) else {}
    app = ident.get("appearance") or ident.get("cara") or ident.get("aspecto") or {}
    if not isinstance(app, dict):
        app = {}
    genero = app.get("genero") or app.get("gender") or ident.get("cara_genero")
    tono = app.get("tono") or app.get("color") or app.get("tono_piel") or ident.get("cara_tono")
    return apariencia_cara(genero=genero, tono=tono)


def apariencia_cara(*, genero: str | None = None, tono: str | None = None) -> dict[str, str]:
    g = normalizar_genero(genero or os.environ.get("ANIMA_CARA_GENERO", "masculino"))
    t = normalizar_tono(tono or os.environ.get("ANIMA_CARA_TONO", "blanco"))
    return {
        "schema": "anima.appearance.v1",
        "genero": g,
        "tono": t,
        "cara_genero": g,
        "cara_tono": t,
    }


def config_instalacion_organismo(
    *,
    nombre: str | None = None,
    genero: str | None = None,
    tono: str | None = None,
) -> dict[str, str]:
    """Paquete de instalación: nombre + apariencia de cabeza."""
    n = normalizar_nombre_organismo(
        nombre or os.environ.get("VST_ORGANISMO_NOMBRE") or os.environ.get("VST_ORGANISMO_LABEL"),
    )
    app = apariencia_cara(genero=genero, tono=tono)
    return {"name": n, "nombre": n, **app}