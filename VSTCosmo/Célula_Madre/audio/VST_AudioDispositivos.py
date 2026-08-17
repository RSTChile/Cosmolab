#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptador tecnológico de audio local (entrada/salida del sistema).

Principio del instalable público
--------------------------------
El inventario es un ESPEJO DINÁMICO de lo que el SO / PortAudio reporta AHORA
en el equipo host. No hay blacklist por marca (RØDE, Voicemeeter, etc.).
Si un dispositivo está conectado y el host lo expone, se lista; si se desconecta
y el host deja de exponerlo, desaparece en el siguiente refresco.

Capa biológica: entrada_local, salida_local, voz → ecosistema.
Backends: sounddevice/PortAudio + PipeWire/Pulse (Linux) vía pactl/parec/paplay.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np

SR = 48000

# Únicos nombres que NO son dispositivos del host: placeholders del propio SO
# (sesión RDP / "Remote Audio") que PortAudio a veces enumera sin hardware real.
# Nunca marcas comerciales (RØDE, CABLE, Voicemeeter, BlackHole, …).
_PLACEHOLDERS_SO = (
    "audio remoto",
    "remote audio",
)

# Mics virtuales de cámara/streaming: se LISTAN siempre; solo se evitan como
# default automático (suelen devolver silencio total).
_MICS_VIRTUALES = (
    "camo",
    "obs virtual",
    "virtual cam",
    "droidcam",
    "iriun",
    "epoccam",
    "snap camera",
    "manycam",
    "xsplit",
    "nvidia broadcast",
    "mmhmm",
)

# Alias genéricos que no identifican hardware y pueden redirigir a Audio remoto.
# Se muestran en el inventario, pero no se eligen automáticamente si existe una
# entrada concreta. El usuario todavía puede seleccionarlos de forma explícita.
_MAPPERS_GENERICOS = (
    "asignador de sonido microsoft",
    "microsoft sound mapper",
    "controlador primario de captura",
    "primary sound capture",
)

# Prioridad de host API en Windows al deduplicar el mismo endpoint físico
# (MME + DirectSound + WASAPI + WDM-KS repiten el mismo mic 4 veces).
_HOSTAPI_RANK = {
    "windows wasapi": 0,
    "windows wdm-ks": 1,
    "windows directsound": 2,
    "mme": 3,
    "core audio": 0,
    "pulseaudio": 0,
    "alsa": 1,
}

_last_pa_refresh = 0.0
_PA_REFRESH_MIN_S = 5.0
_PORTAUDIO_LOCK = threading.RLock()
_PA_INVENTORY_REFRESH_S = 5.0
_PA_INPUT_CACHE: list[dict[str, Any]] | None = None
_PA_OUTPUT_CACHE: list[dict[str, Any]] | None = None
_last_pa_input_query = 0.0
_last_pa_output_query = 0.0
_last_pa_capture_activity = 0.0
_pa_capture_active = 0


def _es_placeholder_so(nombre: str) -> bool:
    """True solo para placeholders del SO, nunca por marca de hardware."""
    low = (nombre or "").lower()
    return any(token in low for token in _PLACEHOLDERS_SO)


# Compat: código antiguo importa _es_fuente_heredada
def _es_fuente_heredada(nombre: str) -> bool:
    return _es_placeholder_so(nombre)


def _es_bluetooth_hf(nombre: str) -> bool:
    """Metadata: mic Hands-Free Bluetooth (baja tasa). Se lista; no se oculta."""
    low = (nombre or "").lower()
    return (
        "bthhfenum" in low
        or "hands-free" in low
        or "hands free" in low
        or "hf audio" in low
    )


def _nombre_ui(nombre: str) -> str:
    """Limpia nombres Windows crípticos (bthhfenum, %1, saltos de línea) para la UI."""
    raw = (nombre or "").replace("\r", " ").replace("\n", " ").strip()
    if not raw:
        return raw
    m = re.search(r"\(\s*;\s*\(([^)]+)\)\s*\)\s*$", raw)
    if not m:
        m = re.search(r";\s*\(([^)]+)\)\s*\)\s*$", raw)
    if m and ("bthhfenum" in raw.lower() or "hands-free" in raw.lower()):
        device = m.group(1).strip()
        if device:
            return f"{device} · Bluetooth Hands-Free"
    clean = re.sub(r"@System32\\drivers\\[^\s,)]+", "", raw, flags=re.I)
    clean = re.sub(r"%\d+", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip(" -;,()")
    return clean or raw


def _es_mic_virtual(nombre: str) -> bool:
    """True si parece micrófono de cámara/streaming (Camo, OBS, etc.)."""
    low = (nombre or "").lower()
    return any(token in low for token in _MICS_VIRTUALES)


def _es_mapper_generico(nombre: str) -> bool:
    low = (nombre or "").lower()
    return any(token in low for token in _MAPPERS_GENERICOS)


def _refresh_portaudio(*, force: bool = False) -> None:
    """Actualiza el estado de consulta sin reinicializar PortAudio en uso normal.

    ``sounddevice._terminate()`` invalida streams y consultas de otros hilos. En
    Windows eso termina en una violación de acceso dentro de libportaudio64bit.dll,
    que no puede capturarse como excepción Python. La reenumeración destructiva se
    reserva para una llamada de mantenimiento explícita con ``force=True``.
    """
    global _last_pa_refresh
    if not SD_OK:
        return
    now = time.monotonic()
    if not force and (now - _last_pa_refresh) < _PA_REFRESH_MIN_S:
        return
    if not force:
        _last_pa_refresh = now
        return
    if not _PORTAUDIO_LOCK.acquire(blocking=False):
        return
    try:
        try:
            if hasattr(sd, "_terminate") and hasattr(sd, "_initialize"):
                sd._terminate()  # noqa: SLF001 — solo mantenimiento explícito
                sd._initialize()  # noqa: SLF001
            _last_pa_refresh = now
        except Exception:
            _last_pa_refresh = now
    finally:
        _PORTAUDIO_LOCK.release()


def _hostapi_name(hostapi_index: int) -> str:
    if not SD_OK:
        return ""
    try:
        apis = sd.query_hostapis()
        if 0 <= int(hostapi_index) < len(apis):
            return str(apis[int(hostapi_index)].get("name") or "")
    except Exception:
        pass
    return ""


def _hostapi_rank(hostapi_index: int) -> int:
    name = _hostapi_name(hostapi_index).lower()
    for key, rank in _HOSTAPI_RANK.items():
        if key in name:
            return rank
    return 50


def _dedupe_portaudio(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Un endpoint físico aparece 3–4 veces (MME/DS/WASAPI/WDM). Nos quedamos
    con la mejor API. No elimina dispositivos distintos (p. ej. dos mics)."""
    best: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for e in entries:
        nom = (e.get("nombre") or "").strip().lower()
        kind = "in" if int(e.get("canales") or e.get("max_input") or 0) else "out"
        # clave: nombre + si es entrada o el n de canales de salida
        key = f"{kind}|{nom}|{int(e.get('canales') or 0)}"
        rank = _hostapi_rank(int(e.get("hostapi") or 0))
        prev = best.get(key)
        if prev is None:
            best[key] = e
            order.append(key)
            e["_api_rank"] = rank
        else:
            if rank < int(prev.get("_api_rank", 99)):
                e["_api_rank"] = rank
                best[key] = e
    out = []
    for k in order:
        e = dict(best[k])
        e.pop("_api_rank", None)
        out.append(e)
    return out


# Entradas que capturan lo que el PC REPRODUCE (Spotify, navegador, juegos…).
_ENTRADAS_SISTEMA = (
    "mezcla est",
    "stereo mix",
    "stereomix",
    "what u hear",
    "what you hear",
    "wave out mix",
    "loopback",
    "altavoz de pc",
    "speakers (",  # a veces aparece como entrada loopback
    "monitor de",
)


def _es_entrada_sistema(nombre: str) -> bool:
    """True si el dispositivo es mezcla/loopback del audio que sale del host."""
    low = (nombre or "").lower()
    return any(token in low for token in _ENTRADAS_SISTEMA)

try:
    import sounddevice as sd
    SD_OK = True
    SD_ERR = ""
    if not hasattr(sd, "_anima_query_devices_raw"):
        sd._anima_query_devices_raw = sd.query_devices
        sd._anima_query_hostapis_raw = sd.query_hostapis

        def _query_devices_serialized(*args, **kwargs):
            with sd._anima_portaudio_lock:
                return sd._anima_query_devices_raw(*args, **kwargs)

        def _query_hostapis_serialized(*args, **kwargs):
            with sd._anima_portaudio_lock:
                return sd._anima_query_hostapis_raw(*args, **kwargs)

        # sounddevice es un módulo compartido: WebLive lo importa por separado.
        # Sustituir aquí sus consultas hace que también esos accesos directos respeten
        # el mismo candado que protege sd.rec()/sd.wait().
        sd.query_devices = _query_devices_serialized
        sd.query_hostapis = _query_hostapis_serialized
    sd._anima_portaudio_lock = _PORTAUDIO_LOCK
    _SD_QUERY_DEVICES_RAW = sd._anima_query_devices_raw
    _SD_QUERY_HOSTAPIS_RAW = sd._anima_query_hostapis_raw
except Exception as e:
    sd = None
    SD_OK = False
    SD_ERR = f"{type(e).__name__}: {e}"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _linux_audio() -> bool:
    return sys.platform.startswith("linux")


def _pactl(args: list[str], timeout: float = 2.0) -> str:
    if not shutil.which("pactl"):
        return ""
    try:
        r = subprocess.run(
            ["pactl", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return (r.stdout or "").strip()
    except Exception:
        return ""


def _pulse_despertar() -> None:
    if not _linux_audio():
        return
    _pactl(["suspend-sink", "@DEFAULT_SINK@", "0"])
    _pactl(["suspend-source", "@DEFAULT_SOURCE@", "0"])


def _parse_pactl_short(lines: str, kind: str) -> list[dict[str, Any]]:
    out = []
    default_name = ""
    if kind == "source":
        default_name = _pactl(["get-default-source"]) or ""
    else:
        default_name = _pactl(["get-default-sink"]) or ""
    for line in (lines or "").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            parts = line.split(maxsplit=1)
        if len(parts) < 2:
            m = re.match(r"^(\d+)(\S.+)$", line.strip())
            if not m:
                continue
            idx, name = m.group(1), m.group(2).strip()
        else:
            idx = parts[0].strip()
            name = parts[1].strip()
        if not name or name.endswith(".monitor") and kind == "sink":
            continue
        es_monitor = name.endswith(".monitor") or ".monitor" in name
        label_name = name
        if len(parts) >= 4 and parts[3] and "Hz" not in parts[3]:
            label_name = parts[3]
        if re.search(r"\d+ch\s+\d+Hz", label_name):
            label_name = name
        if name.startswith("alsa_"):
            short = name.replace("alsa_output.platform-fe00b840.mailbox.stereo-fallback", "Parlantes Pi")
            short = short.replace(".monitor", " (monitor)")
            label_name = short
        out.append({
            "backend": "pulse",
            "pulse_name": name,
            "index": idx,
            "nombre": label_name,
            "es_default": name == default_name,
            "es_monitor": es_monitor,
            "kind": kind,
        })
    return out


def listar_entradas_portaudio(*, refresh: bool = True) -> list[dict[str, Any]]:
    """Todas las entradas que el host reporta ahora (sin blacklist de marcas)."""
    with _PORTAUDIO_LOCK:
        return _listar_entradas_portaudio_locked(refresh=refresh)


def _listar_entradas_portaudio_locked(*, refresh: bool = True) -> list[dict[str, Any]]:
    global _PA_INPUT_CACHE, _last_pa_input_query
    if not SD_OK:
        return []
    now = time.monotonic()
    capture_recent = _pa_capture_active > 0 or (now - _last_pa_capture_activity) < _PA_INVENTORY_REFRESH_S
    cache_fresh = (now - _last_pa_input_query) < _PA_INVENTORY_REFRESH_S
    if _PA_INPUT_CACHE is not None and (not refresh or capture_recent or cache_fresh):
        return [dict(item) for item in _PA_INPUT_CACHE]
    if capture_recent:
        return []
    if refresh:
        _refresh_portaudio()
    out: list[dict[str, Any]] = []
    try:
        try:
            default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
            default_in = int(default_in) if default_in is not None else -1
        except Exception:
            default_in = -1
        for i, d in enumerate(sd.query_devices()):
            nch = int(d.get("max_input_channels", 0) or 0)
            nom = d.get("name", f"dev{i}")
            if nch <= 0:
                continue
            # Solo omitir placeholders del SO (RDP), nunca hardware por marca.
            if _es_placeholder_so(nom):
                continue
            out.append({
                "backend": "portaudio",
                "device_index": i,
                "nombre": nom,
                "nombre_ui": _nombre_ui(nom),
                "canales": nch,
                "sample_rate": int(round(d.get("default_samplerate") or SR)),
                "hostapi": int(d.get("hostapi", 0)),
                "hostapi_name": _hostapi_name(int(d.get("hostapi", 0))),
                "es_default": i == default_in,
                "bluetooth_hf": _es_bluetooth_hf(nom),
            })
    except Exception:
        return []
    # Una fila por dispositivo físico (WASAPI preferido frente a MME/DS duplicados).
    result = _dedupe_portaudio(out)
    _PA_INPUT_CACHE = [dict(item) for item in result]
    _last_pa_input_query = now
    return result


def listar_salidas_portaudio(*, refresh: bool = True) -> list[dict[str, Any]]:
    """Todas las salidas que el host reporta ahora (sin blacklist de marcas)."""
    with _PORTAUDIO_LOCK:
        return _listar_salidas_portaudio_locked(refresh=refresh)


def _listar_salidas_portaudio_locked(*, refresh: bool = True) -> list[dict[str, Any]]:
    global _PA_OUTPUT_CACHE, _last_pa_output_query
    if not SD_OK:
        return []
    now = time.monotonic()
    capture_recent = _pa_capture_active > 0 or (now - _last_pa_capture_activity) < _PA_INVENTORY_REFRESH_S
    cache_fresh = (now - _last_pa_output_query) < _PA_INVENTORY_REFRESH_S
    if _PA_OUTPUT_CACHE is not None and (not refresh or capture_recent or cache_fresh):
        return [dict(item) for item in _PA_OUTPUT_CACHE]
    if capture_recent:
        return []
    if refresh:
        _refresh_portaudio()
    out: list[dict[str, Any]] = []
    try:
        for i, d in enumerate(sd.query_devices()):
            nch = int(d.get("max_output_channels", 0) or 0)
            nom = d.get("name", f"dev{i}")
            if nch <= 0:
                continue
            if _es_placeholder_so(nom):
                continue
            out.append({
                "backend": "portaudio",
                "device_index": i,
                "nombre": nom,
                "nombre_ui": _nombre_ui(nom),
                "canales": nch,
                "sample_rate": int(round(d.get("default_samplerate") or SR)),
                "hostapi": int(d.get("hostapi", 0)),
                "hostapi_name": _hostapi_name(int(d.get("hostapi", 0))),
            })
    except Exception:
        return []
    result = _dedupe_portaudio(out)
    _PA_OUTPUT_CACHE = [dict(item) for item in result]
    _last_pa_output_query = now
    return result


def listar_entradas_pulse() -> list[dict[str, Any]]:
    if not _linux_audio():
        return []
    _pulse_despertar()
    raw = _pactl(["list", "short", "sources"])
    items = _parse_pactl_short(raw, "source")
    entradas = []
    for it in items:
        nom = it.get("nombre") or it.get("pulse_name") or ""
        if _es_placeholder_so(nom):
            continue
        entradas.append({
            **it,
            "canales": 2,
            "tipo_entrada": "monitor_sistema" if it["es_monitor"] else "microfono",
        })
    return entradas


def listar_salidas_pulse() -> list[dict[str, Any]]:
    if not _linux_audio():
        return []
    _pulse_despertar()
    raw = _pactl(["list", "short", "sinks"])
    items = _parse_pactl_short(raw, "sink")
    return [{**it, "canales": 2, "tipo_salida": "parlantes"} for it in items]


def listar_entradas() -> list[dict[str, Any]]:
    """Entradas del host en este instante: PortAudio + Pulse/PipeWire (Linux)."""
    pa = listar_entradas_portaudio(refresh=True)
    pulse = listar_entradas_pulse()
    out = []
    for d in pa:
        out.append({**d, "id": f"pa:{d['device_index']}", "prioridad": -1 if d.get("es_default") else 0})
    for d in pulse:
        pr = 1 if d.get("es_monitor") else 0
        if d.get("es_default"):
            pr -= 1
        out.append({**d, "id": f"pulse:{d['pulse_name']}", "prioridad": pr + 2})
    out.sort(key=lambda x: (x.get("prioridad", 9), x.get("nombre", "")))
    return out


def listar_salidas() -> list[dict[str, Any]]:
    """Salidas del host en este instante: PortAudio + Pulse/PipeWire (Linux)."""
    # refresh con throttle: si listar_entradas() acaba de reenumerar, no repite.
    pa = listar_salidas_portaudio(refresh=True)
    pulse = listar_salidas_pulse()
    out = []
    for d in pa:
        out.append({**d, "id": f"pa_out:{d['device_index']}", "prioridad": 0})
    for d in pulse:
        pr = 0 if d.get("es_default") else 1
        out.append({**d, "id": f"pulse_out:{d['pulse_name']}", "prioridad": pr + 1})
    out.sort(key=lambda x: (x.get("prioridad", 9), x.get("nombre", "")))
    return out


def _match_policy(name: str, policy: str, match: str) -> bool:
    low = (name or "").lower()
    if match:
        return match.lower() in low
    if policy == "prefer-usb":
        return any(t in low for t in ("usb", "mic", "microphone", "webcam"))
    if policy in ("prefer-system-input", "auto", ""):
        if any(t in low for t in ("usb", "mic", "microphone")):
            return True
        if any(t in low for t in ("pulse", "pipewire", "default", "monitor")):
            return True
    return False


def entrada_local_recomendada(
    *,
    policy: str | None = None,
    match: str | None = None,
    pi_device: str | None = None,
) -> dict[str, Any] | None:
    """Selecciona la entrada recomendada sin competir con una captura activa."""
    with _PORTAUDIO_LOCK:
        return _entrada_local_recomendada_locked(
            policy=policy,
            match=match,
            pi_device=pi_device,
        )


def _entrada_local_recomendada_locked(
    *,
    policy: str | None = None,
    match: str | None = None,
    pi_device: str | None = None,
) -> dict[str, Any] | None:
    policy = policy or _env("ANIMA_AUDIO_LOCAL_POLICY", "prefer-system-input")
    match = match or _env("ANIMA_AUDIO_LOCAL_MATCH")
    pi_device = pi_device or _env("ANIMA_PI_AUDIO_DEVICE")

    entradas = listar_entradas()
    if not entradas:
        return None
    # Precargar también salidas/loopbacks antes de que empiece la captura
    # continua; /fuentes no debe abrir una consulta nativa nueva entre bloques.
    listar_salidas()

    def _nom(e: dict[str, Any]) -> str:
        return e.get("nombre") or e.get("pulse_name") or ""

    # Preferir entradas concretas del host; mappers genéricos, BT Hands-Free y
    # mics virtuales se listan, pero no son el default si hay otra alternativa.
    preferidos = [
        e for e in entradas
        if not _es_mic_virtual(_nom(e))
        and not e.get("bluetooth_hf")
        and not _es_bluetooth_hf(_nom(e))
    ]
    bluetooth = [
        e for e in entradas
        if e.get("bluetooth_hf") or _es_bluetooth_hf(_nom(e))
    ]
    virtuales = [e for e in entradas if _es_mic_virtual(_nom(e))]
    concretos = [e for e in preferidos if not _es_mapper_generico(_nom(e))]
    candidatos = concretos or preferidos or bluetooth or virtuales or entradas

    if pi_device:
        for e in entradas:
            if pi_device.lower() in _nom(e).lower():
                return _entrada_a_fuente(e)

    if match:
        for e in entradas:
            if match.lower() in _nom(e).lower() or match.lower() in (e.get("pulse_name") or "").lower():
                return _entrada_a_fuente(e)

    for e in candidatos:
        if e.get("es_default"):
            return _entrada_a_fuente(e)

    for e in candidatos:
        nombre = _nom(e)
        if _match_policy(nombre, policy, ""):
            return _entrada_a_fuente(e)

    if candidatos:
        return _entrada_a_fuente(candidatos[0])
    if bluetooth:
        return _entrada_a_fuente(bluetooth[0])
    if virtuales:
        return _entrada_a_fuente(virtuales[0])
    return _entrada_a_fuente(entradas[0])


def salida_local_recomendada() -> dict[str, Any] | None:
    salidas = listar_salidas()
    if not salidas:
        return None
    for s in salidas:
        if s.get("es_default"):
            return s
    return salidas[0]


def _entrada_a_fuente(entrada: dict[str, Any]) -> dict[str, Any]:
    if entrada.get("backend") == "pulse":
        nom = entrada.get("nombre") or entrada.get("pulse_name")
        tipo = "monitor_sistema" if entrada.get("es_monitor") else "microfono"
        return {
            "tipo": "pulse",
            "pulse_name": entrada["pulse_name"],
            "channel_index": 0,
            "nombre": nom,
            "backend": "pulse",
            "tipo_entrada": tipo,
            "label": f"🎙 {nom}" + (" · monitor sistema" if entrada.get("es_monitor") else ""),
        }
    di = int(entrada["device_index"])
    nom = entrada.get("nombre", f"dev{di}")
    ui = entrada.get("nombre_ui") or _nombre_ui(nom)
    bt = bool(entrada.get("bluetooth_hf") or _es_bluetooth_hf(nom))
    return {
        "tipo": "dispositivo",
        "device_index": di,
        "channel_index": 0,
        "nombre": nom,
        "backend": "portaudio",
        "tipo_entrada": "bluetooth_hf" if bt else "microfono",
        "bluetooth_hf": bt,
        "label": (
            f"🎧 {ui} — mic Bluetooth (baja tasa)"
            if bt else
            f"🎙 {ui} — canal 1"
        ),
    }


def _parece_captura_de_reproduccion(nombre: str) -> bool:
    """Heurística neutra: entradas del SO que capturan lo reproducido (no un micrófono)."""
    low = (nombre or "").lower()
    if _es_entrada_sistema(nombre):
        return True
    tokens = (
        "voicemeeter out",
        "cable output",
        "vb-audio point)",  # CABLE Output / Input (VB-Audio Point)
        "stereo mix",
        "mezcla est",
        "what u hear",
        "what you hear",
        "wave out",
        "loopback",
    )
    return any(t in low for t in tokens)


def fuentes_dispositivo_ui(entradas: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Descriptores para el selector: TODAS las entradas del host (un canal por entrada).

    Inventario genérico — no prioriza ni oculta mezclas/loopbacks por app concreta.
    Cada dispositivo listado por PortAudio/Pulse aparece y es elegible en vivo.
    """
    entradas = entradas if entradas is not None else listar_entradas()
    out: list[dict[str, Any]] = []
    for e in entradas:
        if e.get("backend") == "pulse":
            nom = e.get("nombre") or e.get("pulse_name")
            mon = bool(e.get("es_monitor") or e.get("tipo_entrada") == "monitor_sistema")
            nch = max(1, int(e.get("canales", 2)))
            # Estéreo: L/R. Multicanal: una entrada (mezcla mono de todos).
            canales_ui = [0, 1] if nch == 2 else ([0] if nch != 1 else [0])
            if nch > 2:
                canales_ui = [0]
            for c in canales_ui:
                multi = nch > 2
                item = {
                    "tipo": "pulse",
                    "pulse_name": e["pulse_name"],
                    "channel_index": c,
                    "nombre": nom,
                    "canales": nch,
                    "backend": "pulse",
                    "es_default": e.get("es_default", False),
                    "tipo_entrada": "monitor_sistema" if mon else "microfono",
                    "mix_to_mono": bool(mon or multi),
                    "label": (
                        f"🔊 {nom} — monitor de reproducción"
                        if mon and multi else
                        f"🔊 {nom} — monitor · canal {c + 1}"
                        if mon else
                        f"🎙 {nom} — {nch} ch (mezcla)"
                        if multi else
                        f"🎙 {nom} — canal {c + 1}"
                    ) + (" ★" if e.get("es_default") and c == 0 else ""),
                }
                out.append(item)
        else:
            di = int(e["device_index"])
            nch = max(1, int(e.get("canales", 1)))
            nom = e.get("nombre", f"dev{di}")
            ui = e.get("nombre_ui") or _nombre_ui(nom)
            bt = bool(e.get("bluetooth_hf") or _es_bluetooth_hf(nom))
            # Clasificación neutra: mezcla/loopback del SO = reproducción; resto = entrada.
            repro = (not bt) and (_es_entrada_sistema(nom) or _parece_captura_de_reproduccion(nom))
            # Estéreo: ambos canales. Multicanal (>2): una sola opción mono-mezcla.
            if nch > 2:
                canales_ui = [0]
            elif nch == 2:
                canales_ui = [0, 1]
            else:
                canales_ui = [0]
            multi = nch > 2
            for c in canales_ui:
                if bt:
                    tipo_in = "bluetooth_hf"
                    label = f"🎧 {ui} — mic Bluetooth (baja tasa)" + (
                        " ★" if e.get("es_default") and c == 0 else ""
                    )
                elif repro and multi:
                    tipo_in = "reproduccion"
                    label = f"🔊 {ui} — reproducción ({nch} ch)"
                elif repro:
                    tipo_in = "reproduccion"
                    label = f"🔊 {ui} — reproducción · canal {c + 1}"
                elif multi:
                    tipo_in = "microfono"
                    label = f"🎙 {ui} — {nch} ch (mezcla)"
                else:
                    tipo_in = "microfono"
                    label = f"🎙 {ui} — canal {c + 1}"
                if e.get("es_default") and c == 0 and "★" not in label:
                    label += " ★"
                item = {
                    "tipo": "dispositivo",
                    "device_index": di,
                    "channel_index": c,
                    "nombre": nom,
                    "nombre_ui": ui,
                    "canales": nch,
                    "backend": "portaudio",
                    "hostapi": e.get("hostapi"),
                    "es_default": e.get("es_default", False),
                    "tipo_entrada": tipo_in,
                    "bluetooth_hf": bt,
                    "mix_to_mono": bool(repro or multi or bt),
                    "label": label,
                }
                out.append(item)
    return out


def fuentes_loopback_ui(salidas: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Salidas del host como fuentes capturables vía WASAPI loopback (si el SO lo permite).

    Si el sounddevice instalado no soporta loopback, devuelve []. En ese caso el
    inventario se apoya en las entradas reales de reproducción (mezcla, cables…).
    """
    if not SD_OK or _wasapi_loopback_settings() is None:
        return []
    salidas = salidas if salidas is not None else listar_salidas()
    out: list[dict[str, Any]] = []
    vistos: set[str] = set()
    for s in salidas:
        if s.get("backend") != "portaudio":
            continue
        nom = (s.get("nombre") or "").strip()
        if not nom or _es_placeholder_so(nom):
            continue
        key = nom.lower()
        if key in vistos:
            continue
        vistos.add(key)
        di = int(s["device_index"])
        ui = s.get("nombre_ui") or _nombre_ui(nom)
        out.append({
            "tipo": "dispositivo",
            "device_index": di,
            "channel_index": 0,
            "nombre": nom,
            "nombre_ui": ui,
            "canales": int(s.get("canales", 2) or 2),
            "backend": "wasapi_loopback",
            "tipo_entrada": "reproduccion",
            "loopback": True,
            "mix_to_mono": True,
            "label": f"🔊 Reproducción · {ui}",
        })
    return out


def fuentes_audio_host_ui() -> list[dict[str, Any]]:
    """Inventario dinámico del host: todo lo que PortAudio/Pulse reporta ahora.

    - Todas las entradas reales (mics, BT, mezclas, cables, virtuales…)
    - Loopbacks WASAPI de salidas, si el backend lo soporta
    Sin blacklist de marcas. El usuario elige cualquiera en cualquier momento.
    """
    # listar_entradas() reenumera PortAudio (throttle) y trae el mapa actual del host.
    entradas = fuentes_dispositivo_ui()
    # Marcar capturas de reproducción con tipo neutro (sin renombrar a una app)
    for d in entradas:
        nom = d.get("nombre") or ""
        if d.get("tipo_entrada") not in ("reproduccion", "monitor_sistema") and _parece_captura_de_reproduccion(nom):
            d["tipo_entrada"] = "reproduccion"
            d["mix_to_mono"] = True
            if not str(d.get("label") or "").startswith("🔊"):
                ch = int(d.get("channel_index", 0)) + 1
                d["label"] = f"🔊 {nom} — reproducción · canal {ch}"
    nombres_entrada = {(d.get("nombre") or "").strip().lower() for d in entradas}
    loopbacks = []
    for lb in fuentes_loopback_ui():
        nom = (lb.get("nombre") or "").strip().lower()
        if nom and nom in nombres_entrada:
            continue
        loopbacks.append(lb)
    return entradas + loopbacks


def _resample_canales(rec: np.ndarray, dev_sr: int, sr: int) -> np.ndarray:
    rec = np.asarray(rec, dtype=np.float64)
    if rec.ndim == 1:
        rec = rec.reshape(-1, 1)
    if int(dev_sr) == int(sr) or rec.shape[0] < 2:
        return rec
    m = max(1, int(round(rec.shape[0] * float(sr) / float(dev_sr))))
    base = np.arange(rec.shape[0], dtype=np.float64)
    idx = np.linspace(0, rec.shape[0] - 1, m)
    return np.stack([np.interp(idx, base, rec[:, c]) for c in range(rec.shape[1])], axis=1)


def _wasapi_loopback_settings():
    """extra_settings WASAPI loopback si el sounddevice del host lo soporta.

    sounddevice 0.5.x en Windows a menudo NO expone loopback=True; en ese caso
    devolvemos None y el inventario se apoya en entradas reales del SO
    (Mezcla estéreo, Voicemeeter Out, CABLE Output, etc.).
    """
    if not SD_OK or not hasattr(sd, "WasapiSettings"):
        return None
    # Probar APIs conocidas sin asumir la firma
    for kwargs in (
        {"loopback": True},
        {"loopback": True, "auto_convert": True},
    ):
        try:
            return sd.WasapiSettings(**kwargs)
        except TypeError:
            continue
        except Exception:
            continue
    return None


def _indice_salida_default() -> int | None:
    if not SD_OK:
        return None
    try:
        out = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else None
        if out is not None and int(out) >= 0:
            return int(out)
    except Exception:
        pass
    try:
        for i, d in enumerate(sd.query_devices()):
            nom = d.get("name") or ""
            if int(d.get("max_output_channels", 0) or 0) <= 0 or _es_placeholder_so(nom):
                continue
            low = nom.lower()
            if any(t in low for t in ("realtek", "speakers", "altavoz", "headphones", "auricular")):
                return i
        for i, d in enumerate(sd.query_devices()):
            nom = d.get("name") or ""
            if int(d.get("max_output_channels", 0) or 0) > 0 and not _es_placeholder_so(nom):
                return i
    except Exception:
        pass
    return None


def grabar_wasapi_loopback(seg: float, sr: int = SR, device_index: int | None = None) -> np.ndarray:
    """Captura lo que el PC está reproduciendo (WASAPI loopback sobre la salida).

    Es la vía fiable para Spotify/navegador cuando la 'Mezcla estéreo' llega en silencio.
    """
    global _pa_capture_active, _last_pa_capture_activity
    with _PORTAUDIO_LOCK:
        _pa_capture_active += 1
        _last_pa_capture_activity = time.monotonic()
        try:
            return _grabar_wasapi_loopback_locked(seg, sr=sr, device_index=device_index)
        finally:
            _pa_capture_active -= 1
            _last_pa_capture_activity = time.monotonic()


def _grabar_wasapi_loopback_locked(
    seg: float,
    sr: int = SR,
    device_index: int | None = None,
) -> np.ndarray:
    if not SD_OK:
        raise RuntimeError(f"sounddevice no disponible: {SD_ERR}")
    extra = _wasapi_loopback_settings()
    if extra is None:
        raise RuntimeError("WASAPI loopback no disponible en este host")
    di = int(device_index) if device_index is not None else _indice_salida_default()
    if di is None:
        raise RuntimeError("no hay dispositivo de salida para loopback")
    info = sd.query_devices(di)
    nom = info.get("name", f"out{di}")
    nch = max(1, min(2, int(info.get("max_output_channels", 2) or 2)))
    dev_sr = int(round(info.get("default_samplerate") or sr)) or sr
    frames = max(1, int(round(float(seg) * dev_sr)))
    try:
        rec = sd.rec(
            frames,
            samplerate=dev_sr,
            channels=nch,
            device=di,
            dtype="float32",
            extra_settings=extra,
        )
        sd.wait()
    except Exception as e:
        raise RuntimeError(f"loopback WASAPI falló en '{nom}': {e}") from e
    rec = _resample_canales(rec, dev_sr, sr)
    return rec


def grabar_portaudio(
    device_index: int,
    channels: int,
    seg: float,
    sr: int = SR,
    *,
    loopback: bool = False,
    mix_to_mono: bool = False,
) -> np.ndarray:
    """Captura mediante PortAudio sin solapar operaciones sobre su estado global."""
    global _pa_capture_active, _last_pa_capture_activity
    with _PORTAUDIO_LOCK:
        _pa_capture_active += 1
        _last_pa_capture_activity = time.monotonic()
        try:
            return _grabar_portaudio_locked(
                device_index,
                channels,
                seg,
                sr,
                loopback=loopback,
                mix_to_mono=mix_to_mono,
            )
        finally:
            _pa_capture_active -= 1
            _last_pa_capture_activity = time.monotonic()


def _grabar_portaudio_locked(
    device_index: int,
    channels: int,
    seg: float,
    sr: int = SR,
    *,
    loopback: bool = False,
    mix_to_mono: bool = False,
) -> np.ndarray:
    if not SD_OK:
        raise RuntimeError(f"sounddevice no disponible: {SD_ERR}")

    # Ruta loopback: device_index es de SALIDA (o se usa la salida por defecto).
    if loopback:
        rec = grabar_wasapi_loopback(seg, sr=sr, device_index=device_index if device_index >= 0 else None)
        if mix_to_mono and rec.ndim > 1:
            return rec.mean(axis=1).reshape(-1, 1)
        return rec

    info = sd.query_devices(device_index)
    maxin = int(info.get("max_input_channels", 0))
    nom = info.get("name", f"dev{device_index}")
    if maxin <= 0:
        # Si parece salida y pedimos sistema, intentar loopback
        if int(info.get("max_output_channels", 0)) > 0:
            return grabar_portaudio(device_index, channels, seg, sr, loopback=True, mix_to_mono=mix_to_mono)
        raise RuntimeError(f"'{nom}' no tiene canales de entrada.")
    # Para mezcla/loopback del sistema pedimos todos los canales y luego promediamos
    # (Spotify a veces llega solo a un canal o con poca energía por canal).
    sistema = _es_entrada_sistema(nom)
    want = max(1, int(channels))
    if sistema:
        want = max(want, maxin)
    nch = min(maxin, want)
    dev_sr = int(round(info.get("default_samplerate") or sr)) or sr
    frames = max(1, int(round(float(seg) * dev_sr)))
    rec = None
    last_err = None
    # Intentos: normal → con latency alta (dispositivos Realtek a veces fallan en exclusive)
    for kwargs in (
        dict(dtype="float32"),
        dict(dtype="float32", latency="high"),
    ):
        try:
            rec = sd.rec(frames, samplerate=dev_sr, channels=nch, device=device_index, **kwargs)
            sd.wait()
            break
        except Exception as e:
            last_err = e
            rec = None
    if rec is None:
        raise RuntimeError(f"no se pudo capturar de '{nom}': {last_err}")
    rec = _resample_canales(rec, dev_sr, sr)

    # Si la mezcla clásica viene muda, caer a WASAPI loopback de la salida por defecto.
    pk = float(np.max(np.abs(rec))) if rec.size else 0.0
    if sistema and pk < 1e-5:
        try:
            lb = grabar_wasapi_loopback(seg, sr=sr)
            if float(np.max(np.abs(lb))) > pk:
                rec = lb
                pk = float(np.max(np.abs(rec))) if rec.size else 0.0
        except Exception:
            pass

    if mix_to_mono or sistema:
        if rec.ndim > 1 and rec.shape[1] > 1:
            mono = rec.mean(axis=1)
        else:
            mono = rec.reshape(-1)
        # Ganancia suave y limitada: la mezcla puede llegar baja, pero no amplificar
        # de forma agresiva (evita saturación y artefactos tipo "eco" percibido).
        pk = float(np.max(np.abs(mono))) if mono.size else 0.0
        if 1e-5 < pk < 0.05:
            mono = mono * min(0.6 / pk, 6.0)
        return mono.reshape(-1, 1)

    return rec


def grabar_pulse(pulse_name: str, seg: float, channels: int = 1, sr: int = SR) -> np.ndarray:
    if not shutil.which("parec"):
        raise RuntimeError("parec no instalado (pulseaudio-utils).")
    _pulse_despertar()
    ch = max(1, min(2, int(channels)))
    cmd = [
        "parec",
        f"--device={pulse_name}",
        "--format=float32le",
        f"--rate={sr}",
        f"--channels={ch}",
    ]
    nbytes = int(seg * sr * ch * 4)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        raw = proc.stdout.read(nbytes) if proc.stdout else b""
    finally:
        proc.kill()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
    if not raw:
        err = (proc.stderr.read() if proc.stderr else b"").decode("utf-8", errors="replace")[:200]
        if err:
            raise RuntimeError(f"no se pudo capturar de '{pulse_name}': {err}")
        return np.zeros(int(seg * sr), dtype=np.float64)
    arr = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    if ch > 1:
        n = len(arr) // ch
        arr = arr[: n * ch].reshape(n, ch)
    else:
        arr = arr[: int(seg * sr)]
    return arr


def reproducir_pulse(pulse_sink: str, audio: np.ndarray, sr: int = SR) -> None:
    """Solo monitoreo local del mundo — NO la voz del organismo hacia el ecosistema."""
    if not shutil.which("paplay"):
        return
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return
    cmd = ["paplay", f"--device={pulse_sink}", f"--format=float32le", f"--rate={sr}", "--channels=1", "-"]
    subprocess.run(cmd, input=mono.tobytes(), timeout=max(mono.size / sr + 1.0, 2.0), check=False)


def fuente_audio_sistema() -> dict[str, Any] | None:
    """Compat: primera fuente de reproducción del host (loopback genérico).

    Preferir `fuentes_audio_host_ui()` para el inventario completo.
    """
    lbs = fuentes_loopback_ui()
    if lbs:
        return dict(lbs[0])
    for d in fuentes_dispositivo_ui():
        if d.get("tipo_entrada") in ("reproduccion", "monitor_sistema", "sistema"):
            return dict(d)
    return None


def snapshot_dispositivos() -> dict[str, Any]:
    _pulse_despertar()
    entradas = listar_entradas()
    salidas = listar_salidas()
    entrada_rec = entrada_local_recomendada()
    salida_rec = salida_local_recomendada()
    voz_destino = _env("ANIMA_VOZ_DESTINO", "ecosistema") or "ecosistema"
    return {
        "ok": SD_OK or bool(entradas) or bool(salidas),
        "sounddevice": {"ok": SD_OK, "error": SD_ERR},
        "pulse": {"ok": _linux_audio() and bool(shutil.which("pactl")), "parec": bool(shutil.which("parec"))},
        "entradas": entradas,
        "salidas": salidas,
        "entrada_recomendada": entrada_rec,
        "salida_recomendada": salida_rec,
        "voz_destino": voz_destino,
        "voz_nota": "La voz del organismo sale por HTTP (/comunicacion/bloque.wav) al ecosistema; no se enruta a parlantes locales.",
        "monitor_local": _env("ANIMA_AUDIO_MONITOR", "0").lower() in ("1", "true", "yes", "on"),
    }
