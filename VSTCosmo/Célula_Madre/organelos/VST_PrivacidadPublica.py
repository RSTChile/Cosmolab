#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Privacidad pública del organismo / Observatorio.

- Asa opaca de organismo (no derivada de máquina).
- Categoría de fuente de audio (no nombre del dispositivo del SO).
- Sanitización de payloads expuestos por HTTP público.

No altera la captura real de audio ni la UI local de selección de entradas:
solo lo que se *publica* hacia el campo / Observatorio.
"""

from __future__ import annotations

import re
import secrets
from typing import Any

# Alfabeto sin A–F ni 0/1/I/O: evita rachas que disparen el detector de hex del auditor.
_ALPH_ASA = "GHJKLMNPQRSTUVWXYZ23456789"
_ASA_RE = re.compile(r"^ANIMA_[GHJKLMNPQRSTUVWXYZ23456789]{10,16}$")
# Patrón del auditor: OID con tramo hex largo (p. ej. ANIMA_DESKTOP_E2C1C454D013488E8E90)
_OID_HEX_LARGO = re.compile(r"ANIMA_[A-Z0-9_]*[0-9A-F]{12,}", re.I)

# IDs de laboratorio fijos (máquinas de Cosmolab): se conservan legibles.
_OID_LAB = re.compile(
    r"^ANIMA_(?:A|B|C|D|E|E_PI|ANIMA_PI|PI)$"
    r"|^ANIMA_[A-D]$",
    re.I,
)

_RE_RUTA = re.compile(r"/(?:home|Users|Volumes|mnt)/[A-Za-z0-9_\-./]{3,80}")
_RE_IP_PRIV = re.compile(
    r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"
)
_RE_HOST_LOCAL = re.compile(
    r"host\.docker\.internal[:\d]*|\banima-[a-z]:\d+|localhost:\d{2,5}|127\.0\.0\.1:\d{2,5}"
    r"|\b[A-Za-z0-9\-]{3,30}\.local\b",
    re.I,
)
_RE_AUDIO_DEV = re.compile(
    r"\biMac\b|\bMacBook\b|\bMac ?mini\b|alsa[_a-z0-9.\-]{4,60}|\bhw:\d+"
    r"|RODECaster|Scarlett|micr[oó]fono\s*[—\-:(]",
    re.I,
)

_CLAVES_SENSIBLES = frozenset({
    "base_url", "host", "hostname", "ip", "local_ip", "bind", "address",
    "lan_url", "public_url_raw", "device_name", "device_index",
    "fuente_L_local", "fuente_R_local", "izquierdo", "derecho",
})


def generar_asa_opaca() -> str:
    """Asa estable no derivada de hostname/MAC/GUID/ruta. Formato ANIMA_ + 12 chars."""
    return "ANIMA_" + "".join(secrets.choice(_ALPH_ASA) for _ in range(12))


def oid_es_sensible(oid: str | None) -> bool:
    """True si el OID parece derivado de máquina / hex largo (no lab legible)."""
    s = (oid or "").strip()
    if not s:
        return False
    if _OID_LAB.match(s):
        return False
    if _ASA_RE.match(s):
        return False
    if _OID_HEX_LARGO.search(s):
        return True
    if s.upper().startswith("ANIMA_DESKTOP_"):
        return True
    return False


def oid_es_lab(oid: str | None) -> bool:
    return bool(_OID_LAB.match((oid or "").strip()))


def normalizar_oid_publico(oid: str | None, *, mapa: dict[str, str] | None = None) -> str:
    """Devuelve OID publicable. Si es sensible y hay mapa, usa el asa; si no, genera una."""
    s = (oid or "").strip()
    if not s:
        return generar_asa_opaca()
    if not oid_es_sensible(s):
        return s
    if mapa is not None and s in mapa:
        return mapa[s]
    nuevo = generar_asa_opaca()
    if mapa is not None:
        mapa[s] = nuevo
    return nuevo


def categorizar_fuente(raw: Any) -> str:
    """Tipo de fuente para el campo público (nunca el nombre del SO)."""
    if raw is None:
        return "silencio"
    s = str(raw).strip()
    if not s or s in ("—", "-", "–", "none", "null", "None"):
        return "silencio"
    low = s.lower()
    if "silencio" in low or low in ("mute", "mudo", "off"):
        return "silencio"
    if any(k in low for k in ("organismo", "otros organismos", "otro organismo", "par ", "voz del par", "vecino")):
        return "otros_organismos"
    if any(k in low for k in (".wav", ".mp3", ".flac", ".ogg", "archivo", "upload", "demo", "banco", "brandemburgo", "pista")):
        return "archivo"
    # Mezcla / interfaces multicanal (incl. RodeCaster como *tipo*, no como marca en salida)
    if any(
        k in low
        for k in (
            "mix", "mezcla", "loopback", "stereo mix", "main mix", "main multitrack",
            "what u hear", "cable", "rodecaster", "rode ", "multitrack", "virtual cable",
            "blackhole", "soundflower", "vb-audio",
        )
    ):
        return "mezcla_equipo"
    if any(
        k in low
        for k in (
            "mic", "micrófono", "microfono", "entrada", "input", "alsa", "imac",
            "macbook", "mac mini", "canal", "channel", "array", "webcam", "headset",
        )
    ):
        return "microfono"
    # Fallback: capturas de host sin etiqueta clara → micrófono genérico
    return "microfono"


def _scrub_str(s: str) -> str:
    t = s
    t = _RE_RUTA.sub("[ruta]", t)
    t = _RE_IP_PRIV.sub("[red]", t)
    t = _RE_HOST_LOCAL.sub("[host]", t)
    t = _RE_AUDIO_DEV.sub("[audio]", t)
    return t


def sanitizar_estado_publico(estado: dict | None, *, oid_publico: str | None = None) -> dict:
    """Copia del estado apta para /datos, /estado público y exportaciones."""
    if not isinstance(estado, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in estado.items():
        if k in _CLAVES_SENSIBLES:
            continue
        if k in ("fuente_L", "fuente_R", "fuente"):
            out[k] = categorizar_fuente(v)
            continue
        if isinstance(v, str):
            out[k] = _scrub_str(v)
        elif isinstance(v, dict):
            # no anidar base_url internos
            out[k] = sanitizar_estado_publico(v)
        else:
            out[k] = v
    if oid_publico:
        out["organism_id"] = oid_publico
        out["organismo_id"] = oid_publico
    elif "organism_id" in out and oid_es_sensible(str(out.get("organism_id"))):
        # no inventar aquí sin mapa: el caller debe pasar oid_publico
        pass
    # Nunca re-exponer URL LAN
    out.pop("base_url", None)
    return out


def sanitizar_texto_export(texto: str) -> str:
    """CSV / bitácora / cabeceras: mismas reglas de no-fuga."""
    if not texto:
        return texto
    t = _scrub_str(texto)
    # líneas de cabecera con izquierdo/derecho literales
    def _fix_src(m: re.Match) -> str:
        return m.group(1) + categorizar_fuente(m.group(2))

    t = re.sub(
        r"(?im)^(#\s*(?:izquierdo|derecho|fuente(?:_L|_R)?)\s*:\s*)(.+)$",
        _fix_src,
        t,
    )
    t = re.sub(
        r"(?i)(fuente_[LR]\s*[=:]\s*)([^,\n\r]+)",
        lambda m: m.group(1) + categorizar_fuente(m.group(2)),
        t,
    )
    return t
