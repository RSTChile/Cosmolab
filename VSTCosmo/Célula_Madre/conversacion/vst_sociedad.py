#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_SOCIEDAD — Observatorio dinámico de organismos presentes (membrana, solo lectura).

Página NUEVA, separada del observatorio de la díada (:9100). No lo modifica.

Fuente de verdad del roster (v1):
  · Semillas ANIMA_SEED_URLS (p. ej. la Pi con ANIMA)
  · GET /identidad y GET /presencia de cada semilla
  · Los vecinos descubiertos se añaden al campo

Primer caso: ANIMA en la Pi anuncia su presencia → aparece en la pestaña Sociedad.

    python conversacion/vst_sociedad.py   →  http://localhost:9101
"""
from __future__ import annotations

import json
import math
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Privacidad pública (asa opaca + categorías de fuente; no altera interacción).
try:
    from VST_PrivacidadPublica import (  # type: ignore
        categorizar_fuente,
        generar_asa_opaca,
        oid_es_sensible,
        sanitizar_estado_publico,
        sanitizar_texto_export,
    )
except ImportError:
    try:
        import sys as _sys
        _org = Path(__file__).resolve().parents[1] / "celula_madre" / "organelos"
        if str(_org) not in _sys.path:
            _sys.path.insert(0, str(_org))
        from VST_PrivacidadPublica import (  # type: ignore
            categorizar_fuente,
            generar_asa_opaca,
            oid_es_sensible,
            sanitizar_estado_publico,
            sanitizar_texto_export,
        )
    except ImportError:
        def categorizar_fuente(raw):  # type: ignore
            return "silencio" if not raw else "microfono"

        def generar_asa_opaca():  # type: ignore
            import secrets
            return "ANIMA_" + "".join(secrets.choice("GHJKLMNPQRSTUVWXYZ23456789") for _ in range(12))

        def oid_es_sensible(oid):  # type: ignore
            s = (oid or "").upper()
            return "DESKTOP_" in s or (len(s) > 20 and any(c in "0123456789ABCDEF" for c in s[-12:]))

        def sanitizar_estado_publico(estado, oid_publico=None):  # type: ignore
            out = dict(estado or {})
            out.pop("base_url", None)
            for k in ("fuente_L", "fuente_R", "fuente"):
                if k in out:
                    out[k] = categorizar_fuente(out.get(k))
            if oid_publico:
                out["organism_id"] = out["organismo_id"] = oid_publico
            return out

        def sanitizar_texto_export(texto):  # type: ignore
            return texto or ""


# Mapa estable real→público para OID sensibles (solo en el servidor del Observatorio).
_OID_MAP_PATH = Path(
    os.environ.get(
        "ANIMA_SOCIEDAD_OID_MAP",
        str(Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".anima") / "ANIMA" / "sociedad_oid_map.json"),
    )
)
_OID_REAL_TO_PUB: dict[str, str] = {}
_OID_PUB_TO_REAL: dict[str, str] = {}


def _oid_map_load() -> None:
    global _OID_REAL_TO_PUB, _OID_PUB_TO_REAL
    try:
        if _OID_MAP_PATH.is_file():
            raw = json.loads(_OID_MAP_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                _OID_REAL_TO_PUB = {str(k): str(v) for k, v in raw.items() if k and v}
                _OID_PUB_TO_REAL = {v: k for k, v in _OID_REAL_TO_PUB.items()}
    except Exception:
        _OID_REAL_TO_PUB, _OID_PUB_TO_REAL = {}, {}


def _oid_map_save() -> None:
    try:
        _OID_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _OID_MAP_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(_OID_REAL_TO_PUB, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, _OID_MAP_PATH)
    except Exception as ex:
        print(f"[sociedad] oid map save: {ex}", flush=True)


_oid_map_load()


def _oid_publico(real: str) -> str:
    """OID expuesto en superficies públicas. Lab legible; desktop sensible → asa opaca estable."""
    r = (real or "").strip()
    if not r:
        return r
    if not oid_es_sensible(r):
        return r
    if r in _OID_REAL_TO_PUB:
        return _OID_REAL_TO_PUB[r]
    pub = generar_asa_opaca()
    _OID_REAL_TO_PUB[r] = pub
    _OID_PUB_TO_REAL[pub] = r
    _oid_map_save()
    print(f"[sociedad] OID sensible mapeado a asa pública: {r[:24]}… → {pub}", flush=True)
    return pub


def _oid_real(pub_or_real: str) -> str:
    s = (pub_or_real or "").strip()
    return _OID_PUB_TO_REAL.get(s, s)


def _json_safe(value):
    """Convierte valores no finitos a null para producir JSON web válido."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _json_dumps(value, **kwargs) -> str:
    return json.dumps(_json_safe(value), allow_nan=False, **kwargs)

# Defaults instalables: solo el organismo local. Sin IPs de laboratorio Cosmolab.
# Override de despliegue (lab / servidor sociedad): ANIMA_SEED_URLS=...
_LOCAL_PORT = os.environ.get("VST_PUERTO") or os.environ.get("ANIMA_LOCAL_PORT") or "7788"
_DEFAULT_SEEDS = f"http://127.0.0.1:{_LOCAL_PORT}"
SEED_URLS = [
    u.strip().rstrip("/")
    for u in os.environ.get("ANIMA_SEED_URLS", _DEFAULT_SEEDS).split(",")
    if u.strip()
]
# Banda desktop instalable (9100+): el organismo local puede no estar en SEED_URLS
# si eligió otro puerto sticky (p.ej. 9103). Escanear en cada refresh.
_LOCAL_SCAN = os.environ.get("ANIMA_SOCIEDAD_LOCAL_SCAN", "1").strip().lower() in (
    "1", "true", "yes", "on",
)
try:
    _LOCAL_SCAN_FROM = int(os.environ.get("ANIMA_SOCIEDAD_SCAN_FROM", "9100"))
    _LOCAL_SCAN_TO = int(os.environ.get("ANIMA_SOCIEDAD_SCAN_TO", "9119"))
except ValueError:
    _LOCAL_SCAN_FROM, _LOCAL_SCAN_TO = 9100, 9119
_COLS_CONOCIDOS = {
    "ANIMA_A": "#6db6ff", "ANIMA_B": "#ffd479", "ANIMA_C": "#8ef0c0",
    "ANIMA_D": "#ff8c6b", "ANIMA_E_PI": "#c9a0ff", "ANIMA_ANIMA_PI": "#a797c9",
}
PORT = int(os.environ.get("ANIMA_SOCIEDAD_PORT", "9101"))
POLL_S = float(os.environ.get("POLL_S", "0.5"))
ROSTER_TTL_S = float(os.environ.get("ANIMA_ROSTER_TTL_S", "8.0"))
SOC_COLS = max(1, min(12, int(os.environ.get("ANIMA_SOCIEDAD_COLS", "6"))))
SOC_HEAD_AR = os.environ.get("ANIMA_SOCIEDAD_HEAD_AR", "2/3")  # ancho/alto → rectángulo vertical
SOC_HEAD_W = os.environ.get("ANIMA_SOCIEDAD_HEAD_W", "80%")
SOC_HEAD_FIT = float(os.environ.get("ANIMA_SOCIEDAD_HEAD_FIT", "0.78"))
SOCIEDAD_PUBLIC = os.environ.get("ANIMA_SOCIEDAD_PUBLIC", "0").strip().lower() in ("1", "true", "yes", "on")
DEFAULT_COUNTRY = (os.environ.get("ANIMA_DEFAULT_COUNTRY", "CL") or "CL").upper()[:2]

_PAISES_NOMBRE: dict[str, str] = {
    "CL": "Chile", "AR": "Argentina", "PE": "Perú", "BO": "Bolivia", "UY": "Uruguay",
    "BR": "Brasil", "CO": "Colombia", "EC": "Ecuador", "MX": "México", "ES": "España",
    "US": "Estados Unidos", "DE": "Alemania", "FR": "Francia", "GB": "Reino Unido",
    "JP": "Japón", "AU": "Australia", "CA": "Canadá", "IT": "Italia", "PT": "Portugal",
}
_DEFAULT_PAISES_OID: dict[str, str] = {
    "ANIMA_A": "CL", "ANIMA_B": "CL", "ANIMA_C": "CL", "ANIMA_D": "CL",
    "ANIMA_ANIMA_PI": "CL", "ANIMA_E_PI": "CL",
}


_DEFAULT_NOMBRES_OID: dict[str, str] = {
    "ANIMA_A": "Alfa",
    "ANIMA_B": "Brisa",
    "ANIMA_C": "Coral",
    "ANIMA_D": "Duno",
    "ANIMA_ANIMA_PI": "ANIMA",
    "ANIMA_E_PI": "Esmeralda",
}
_DEFAULT_ASPECTO_OID: dict[str, dict] = {
    "ANIMA_A": {"genero": "masculino", "tono": "blanco"},
    "ANIMA_B": {"genero": "femenino", "tono": "celeste"},
    "ANIMA_C": {"genero": "femenino", "tono": "rosado"},
    "ANIMA_D": {"genero": "masculino", "tono": "amarillo"},
    "ANIMA_ANIMA_PI": {"genero": "femenino", "tono": "blanco"},
    "ANIMA_E_PI": {"genero": "femenino", "tono": "cafe"},
}


def _load_nombres_oid() -> dict[str, str]:
    out = dict(_DEFAULT_NOMBRES_OID)
    raw = os.environ.get("ANIMA_ORG_NOMBRES", "").strip()
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                for k, v in data.items():
                    if k and v:
                        out[str(k).strip()] = str(v).strip()[:14]
        except Exception:
            pass
    return out


def _load_aspecto_oid() -> dict[str, dict]:
    out = {k: dict(v) for k, v in _DEFAULT_ASPECTO_OID.items()}
    raw = os.environ.get("ANIMA_ORG_ASPECTO", "").strip()
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                for k, v in data.items():
                    if k and isinstance(v, dict):
                        out[str(k).strip()] = v
        except Exception:
            pass
    return out


_NOMBRES_OID = _load_nombres_oid()
_ASPECTO_OID = _load_aspecto_oid()


def _aspecto_meta(entry: dict | None = None, ident: dict | None = None, oid: str = "") -> dict[str, str]:
    entry = entry or {}
    app: dict = {}
    if ident:
        raw = ident.get("appearance") or ident.get("cara") or {}
        if isinstance(raw, dict):
            app.update(raw)
    if isinstance(entry.get("appearance"), dict):
        app.update(entry["appearance"])
    for k in ("genero", "tono", "cara_genero", "cara_tono"):
        if entry.get(k):
            app[k] = entry[k]
    if not app and oid in _ASPECTO_OID and isinstance(_ASPECTO_OID[oid], dict):
        app.update(_ASPECTO_OID[oid])
    g = str(app.get("genero") or app.get("cara_genero") or "masculino").lower()
    t = str(app.get("tono") or app.get("cara_tono") or "blanco").lower()
    if g in ("f", "femenino", "female"):
        g = "femenino"
    else:
        g = "masculino"
    if t not in ("blanco", "celeste", "rosado", "amarillo", "cafe"):
        t = "blanco"
    return {"genero": g, "tono": t}


def _flag_emoji(cc: str) -> str:
    cc = (cc or "").upper()
    if len(cc) != 2 or not cc.isalpha():
        return ""
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in cc)


def _load_paises_oid() -> dict[str, str]:
    out = dict(_DEFAULT_PAISES_OID)
    raw = os.environ.get("ANIMA_ORG_PAISES", "").strip()
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                for k, v in data.items():
                    if k and v:
                        out[str(k).strip()] = str(v).strip().upper()[:2]
        except Exception:
            pass
    return out


_PAISES_OID = _load_paises_oid()


def _pais_desde_ident(ident: dict | None) -> str | None:
    if not ident:
        return None
    for key in ("location", "ubicacion"):
        loc = ident.get(key)
        if isinstance(loc, dict):
            cc = loc.get("country") or loc.get("pais")
            if cc:
                return str(cc).strip().upper()[:2]
    return None


def _pais_meta(code: str | None, oid: str = "") -> dict[str, str]:
    cc = (code or _PAISES_OID.get(oid) or DEFAULT_COUNTRY).upper()[:2]
    return {
        "country_code": cc,
        "country_name": _PAISES_NOMBRE.get(cc, cc),
        "country_flag": _flag_emoji(cc),
    }


def _paises_campo(roster: list[dict]) -> list[dict]:
    counts: dict[str, int] = {}
    names: dict[str, str] = {}
    flags: dict[str, str] = {}
    for e in roster:
        cc = e.get("country_code") or DEFAULT_COUNTRY
        counts[cc] = counts.get(cc, 0) + 1
        names[cc] = e.get("country_name") or _PAISES_NOMBRE.get(cc, cc)
        flags[cc] = e.get("country_flag") or _flag_emoji(cc)
    return [
        {"country_code": cc, "country_name": names[cc], "country_flag": flags[cc], "count": counts[cc]}
        for cc in sorted(counts, key=lambda c: (-counts[c], names[c]))
    ]


def _roster_cliente(roster: list[dict]) -> list[dict]:
    """Vista pública del roster: OID opaco si aplica, sin base_url LAN, fuentes ya no van aquí."""
    out: list[dict] = []
    for e in roster:
        item = dict(e)
        real = (item.get("organism_id") or "").strip()
        pub = _oid_publico(real)
        item["organism_id"] = pub
        item.pop("base_url", None)
        item["observatorio_url"] = f"/puerta?oid={urllib.parse.quote(pub, safe='')}"
        item["voz_proxy"] = f"/voz?oid={urllib.parse.quote(pub, safe='')}"
        # No filtrar por SOCIEDAD_PUBLIC: el JSON /datos no debe filtrar LAN ni OID de máquina
        # ni siquiera en modo lab si se publica por túnel.
        out.append(item)
    return out


def _estados_cliente(roster: list[dict]) -> dict[str, dict]:
    with _LOCK:
        raw = dict(_ESTADOS)
    out: dict[str, dict] = {}
    for e in roster:
        real = (e.get("organism_id") or "").strip()
        if not real:
            continue
        pub = _oid_publico(real)
        est = raw.get(real) or {"vivo": False, "organism_id": real}
        out[pub] = sanitizar_estado_publico(est, oid_publico=pub)
    return out


_LOCK = threading.Lock()
_ROSTER: list[dict] = []          # [{organism_id, name, base_url, estado_presencia, ...}]
_ROSTER_T = 0.0
_ESTADOS: dict[str, dict] = {}    # organism_id → último estado mergeado
_PRESENCIA_CAMPO: dict = {}


# Catálogo fantasma de lab (Alfa/Brisa/…): solo con ANIMA_SOCIEDAD_CATALOGO=1.
# En instalable el roster es SOLO organismos realmente descubiertos (semilla + presencia).
SOCIEDAD_CATALOGO = os.environ.get("ANIMA_SOCIEDAD_CATALOGO", "0").strip().lower() in (
    "1", "true", "yes", "on",
)
_SEED_BASES: dict[str, str] = {
    "ANIMA_A": f"http://127.0.0.1:{_LOCAL_PORT}",
}


def _get_json(base_url: str, path: str, timeout: float = 1.5) -> dict | None:
    url = base_url.rstrip("/") + path
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _color_hash(organism_id: str) -> str:
    h = 0
    for c in organism_id or "?":
        h = (h * 31 + ord(c)) & 0xFFFFFF
    r = 80 + (h & 0x7F)
    g = 80 + ((h >> 8) & 0x7F)
    b = 80 + ((h >> 16) & 0x7F)
    return f"#{r:02x}{g:02x}{b:02x}"


def _upsert_roster(entry: dict, seen: set[str], out: list[dict]) -> None:
    oid = (entry.get("organism_id") or "").strip()
    base = (entry.get("base_url") or "").strip().rstrip("/")
    if not oid or not base or oid in seen:
        return
    seen.add(oid)
    pais = _pais_meta(entry.get("country_code"), oid)
    aspecto = _aspecto_meta(entry, oid=oid)
    name = _NOMBRES_OID.get(oid) or entry.get("name") or oid
    out.append({
        "organism_id": oid,
        "name": name,
        "base_url": base,
        "estado_presencia": entry.get("estado_presencia", "desconocido"),
        "confianza": entry.get("confianza"),
        "frescura": entry.get("frescura"),
        "source": entry.get("source", "semilla"),
        "capabilities": entry.get("capabilities") or [],
        "color": _COLS_CONOCIDOS.get(oid) or _color_hash(oid),
        "voz_proxy": f"/voz?oid={urllib.parse.quote(oid, safe='')}",
        "observatorio_url": base + "/",
        **pais,
        **aspecto,
    })


def _tcp_abierto(host: str, port: int, timeout: float = 0.25) -> bool:
    """Probe rápido: evita gastar 1.5s de HTTP en puertos cerrados."""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False


def _seed_urls_efectivas() -> list[str]:
    """Semillas configuradas + hosts locales de la banda desktop (si el scan está activo)."""
    import socket as _socket  # noqa: F401 — used via _tcp_abierto

    urls: list[str] = []
    seen: set[str] = set()
    for u in SEED_URLS:
        uu = (u or "").strip().rstrip("/")
        if uu and uu not in seen:
            seen.add(uu)
            urls.append(uu)
    if _LOCAL_SCAN:
        lo = min(_LOCAL_SCAN_FROM, _LOCAL_SCAN_TO)
        hi = max(_LOCAL_SCAN_FROM, _LOCAL_SCAN_TO)
        for p in range(lo, hi + 1):
            if p == PORT:
                continue
            if not _tcp_abierto("127.0.0.1", p, 0.2):
                continue
            uu = f"http://127.0.0.1:{p}"
            if uu not in seen:
                seen.add(uu)
                urls.append(uu)
    return urls


def _descubrir_roster() -> tuple[list[dict], dict]:
    """Roster vivo: semillas + vecinos por presencia."""
    out: list[dict] = []
    seen: set[str] = set()
    campo: dict = {}

    for seed in _seed_urls_efectivas():
        # Saltar semillas locales muertas sin esperar timeout HTTP largo
        try:
            parsed = urllib.parse.urlparse(seed)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            if host in ("127.0.0.1", "localhost", "::1") and not _tcp_abierto(host, port, 0.25):
                continue
        except Exception:
            pass

        ident = _get_json(seed, "/identidad", timeout=1.0)
        if ident and not ident.get("error") and ident.get("organism_id"):
            eps = ident.get("local_endpoints") or ident.get("endpoints") or {}
            # base_url del peer puede ser IP de red Docker (172.x) inalcanzable desde LAN.
            # Preferir la semilla (URL pública de host:puerto) si el peer reporta red privada.
            base_peer = (eps.get("base_url") or "").rstrip("/")
            base_seed = seed.rstrip("/")
            base = base_seed
            if base_peer:
                try:
                    ph = urllib.parse.urlparse(base_peer).hostname or ""
                except Exception:
                    ph = ""
                private = (
                    ph.startswith("172.")
                    or ph.startswith("10.")
                    or ph.startswith("127.")
                    or ph.endswith(".internal")
                    or ph in ("localhost",)
                )
                if not private:
                    base = base_peer
            _upsert_roster({
                "organism_id": ident.get("organism_id", ""),
                "name": ident.get("name") or ident.get("organism_id", "organismo"),
                "base_url": base or base_seed,
                "estado_presencia": ident.get("state", "activo"),
                "source": "semilla",
                "capabilities": ident.get("capabilities") or [],
                "country_code": _pais_desde_ident(ident),
                "appearance": ident.get("appearance") or ident.get("cara"),
            }, seen, out)
        else:
            est = _get_json(seed, "/estado", timeout=1.0)
            if est and (est.get("organismo_id") or est.get("organism_id")):
                oid = est.get("organismo_id") or est.get("organism_id") or ""
                _upsert_roster({
                    "organism_id": oid,
                    "name": est.get("organismo") or oid or "organismo",
                    "base_url": seed.rstrip("/"),
                    "estado_presencia": "activo" if est.get("vivo") else "reposo",
                    "source": "semilla",
                    "capabilities": [],
                }, seen, out)

        pres = _get_json(seed, "/presencia", timeout=1.0)
        if pres:
            campo = pres.get("campo_presencia") or campo
            for v in pres.get("vecinos") or []:
                base = (v.get("base_url") or "").rstrip("/")
                entry = {
                    "organism_id": v.get("organism_id", ""),
                    "name": v.get("name") or v.get("organism_id", ""),
                    "base_url": base,
                    "estado_presencia": v.get("estado_presencia", "presente"),
                    "confianza": v.get("confianza"),
                    "frescura": v.get("frescura"),
                    "source": v.get("source", "presencia"),
                    "capabilities": v.get("capabilities") or [],
                    "country_code": v.get("country") or v.get("pais"),
                }
                # Apariencia del vecino: primero lo que trae /presencia; si no, /identidad del peer.
                app = v.get("appearance") if isinstance(v.get("appearance"), dict) else None
                if not app and (v.get("genero") or v.get("tono") or v.get("cara_genero") or v.get("cara_tono")):
                    app = {
                        "genero": v.get("genero") or v.get("cara_genero"),
                        "tono": v.get("tono") or v.get("cara_tono"),
                        "cara_genero": v.get("cara_genero") or v.get("genero"),
                        "cara_tono": v.get("cara_tono") or v.get("tono"),
                    }
                if not app and base:
                    peer_ident = _get_json(base, "/identidad", timeout=1.0)
                    if peer_ident and not peer_ident.get("error"):
                        app = peer_ident.get("appearance") or peer_ident.get("cara")
                        if peer_ident.get("name"):
                            entry["name"] = peer_ident.get("name")
                        if not entry.get("country_code"):
                            entry["country_code"] = _pais_desde_ident(peer_ident)
                if app:
                    entry["appearance"] = app
                for k in ("genero", "tono", "cara_genero", "cara_tono"):
                    if v.get(k):
                        entry[k] = v[k]
                _upsert_roster(entry, seen, out)

    out.sort(key=lambda e: (0 if e.get("source") == "semilla" else 1, e.get("name", "")))
    return out, campo


def _catalogo_inicial() -> list[dict]:
    """Organismos de laboratorio ficticios (Alfa, Brisa, …).

    Desactivado por defecto: el observatorio instalable solo muestra presencia real.
    Activar con ANIMA_SOCIEDAD_CATALOGO=1 en despliegues de lab/debug.
    """
    if not SOCIEDAD_CATALOGO:
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for oid, nombre in _NOMBRES_OID.items():
        _upsert_roster({
            "organism_id": oid,
            "name": nombre,
            "base_url": _SEED_BASES.get(oid, ""),
            "estado_presencia": "catalogo",
            "source": "catalogo",
        }, seen, out)
    return out


def _refresh_roster() -> None:
    global _ROSTER, _ROSTER_T, _PRESENCIA_CAMPO
    descubierto, campo = _descubrir_roster()
    # Solo organismos reales (+ catálogo de lab si está explícitamente activado).
    por_id = {e["organism_id"]: e for e in _catalogo_inicial()}
    for e in descubierto:
        por_id[e["organism_id"]] = e
    roster = sorted(
        por_id.values(),
        key=lambda e: (0 if e.get("source") in ("semilla", "catalogo") else 1, e.get("name", "")),
    )
    with _LOCK:
        _ROSTER = roster
        _PRESENCIA_CAMPO = campo
        _ROSTER_T = time.time()


def _roster_vivo() -> list[dict]:
    with _LOCK:
        return list(_ROSTER)


def _entry_por_id(oid: str) -> dict | None:
    oid = (oid or "").strip()
    real = _oid_real(oid)
    for e in _roster_vivo():
        rid = e.get("organism_id") or ""
        if rid == oid or rid == real or e.get("name") == oid or _oid_publico(rid) == oid:
            return e
    return None


def _poller():
    global _LAST_POLL_TS
    while True:
        try:
            _refresh_roster()
            _LAST_POLL_TS = time.time()
        except Exception as exc:
            print(f"[sociedad] poll error: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(2.0)
            continue
        roster = _roster_vivo()
        for entry in roster:
            oid = entry["organism_id"]
            base = entry["base_url"]
            est = _get_json(base, "/estado") or {}
            fila = {}
            try:
                uf = _get_json(base, "/ultima_fila") or {}
                fila = uf.get("fila") or {}
            except Exception:
                pass
            merged = dict(fila)
            merged.update(est)
            merged["organism_id"] = oid
            merged["organismo"] = merged.get("organismo") or entry.get("name")
            merged["estado_presencia"] = entry.get("estado_presencia")
            merged["base_url"] = base
            merged["vivo"] = bool(merged.get("vivo")) or bool(fila)
            # Sincronizar apariencia del roster con /estado o /ultima_fila
            # (evita que vecinos UDP queden en masculino/blanco por defecto).
            has_real = any(
                merged.get(k) for k in ("genero", "tono", "cara_genero", "cara_tono")
            )
            with _LOCK:
                _ESTADOS[oid] = merged
                if has_real:
                    aspecto = _aspecto_meta(merged, oid=oid)
                    for i, e in enumerate(_ROSTER):
                        if e.get("organism_id") == oid:
                            _ROSTER[i] = {**e, **aspecto}
                            break
        time.sleep(POLL_S)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if "json" in ctype or "html" in ctype else ""))
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.end_headers()
        try:
            self.wfile.write(b)
        except Exception:
            pass

    def do_GET(self):
        u = urlparse(self.path)
        path = u.path
        qs = parse_qs(u.query)

        if path in ("/", "/index.html"):
            self._send(200, HTML, "text/html")
        elif path == "/vendor/three.min.js":
            fp = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web", "Cajas", "vendor", "three.min.js"))
            try:
                with open(fp, "rb") as fh:
                    self._send(200, fh.read(), "application/javascript")
            except OSError:
                self._send(404, "three.min.js no encontrado", "text/plain")
        elif path == "/datos":
            roster = _roster_vivo()
            roster_out = _roster_cliente(roster)
            estados = _estados_cliente(roster)
            payload: dict = {
                "ok": True,
                "public": SOCIEDAD_PUBLIC,
                "roster": roster_out,
                "estados": estados,
                "campo_presencia": _PRESENCIA_CAMPO,
                "paises_campo": _paises_campo(roster),
                "ts": time.time(),
            }
            # Semillas (IPs de lab) solo en modo no-público y nunca en túnel público.
            if not SOCIEDAD_PUBLIC:
                payload["semillas"] = SEED_URLS
            self._send(200, _json_dumps(payload, ensure_ascii=False))
        elif path == "/salud":
            # Health check for WinSW watchdog / install_services.ps1.
            # Must return HTTP 200 + {"ok": true} or the watchdog restarts forever.
            roster = _roster_vivo()
            with _LOCK:
                poll_ts = float(globals().get("_LAST_POLL_TS", 0) or 0)
            poll_age = (time.time() - poll_ts) if poll_ts > 0 else None
            self._send(
                200,
                _json_dumps(
                    {
                        "ok": True,
                        "port": PORT,
                        "public": SOCIEDAD_PUBLIC,
                        "roster_n": len(roster),
                        "poll_age_s": poll_age,
                        "ts": time.time(),
                    },
                    ensure_ascii=False,
                ),
            )
        elif path == "/puerta":
            oid = (qs.get("oid") or [""])[0]
            entry = _entry_por_id(oid)
            public = globals().get("SOCIEDAD_PUBLIC", False)
            if not entry:
                with _LOCK:
                    snap = _ESTADOS.get(oid) or {}
                if snap:
                    name = snap.get("organismo") or snap.get("name") or oid
                    html = (
                        "<!doctype html><meta charset=utf-8><title>Organismo no alcanzable</title>"
                        f"<body style='font-family:system-ui;background:#0b1018;color:#dfe7f0;padding:24px'>"
                        f"<h1>{name}</h1>"
                        f"<p>Este organismo no es alcanzable desde el Observatorio en este momento.</p>"
                        f"<p style='color:#8aa0b8'>id={oid}</p>"
                        f"<p><a href='/' style='color:#6db6ff'>← Volver</a></p></body>"
                    )
                    self._send(200, html, "text/html")
                    return
                self._send(
                    404,
                    f"<!doctype html><meta charset=utf-8><title>No encontrado</title>"
                    f"<body style='font-family:system-ui;background:#0b1018;color:#dfe7f0;padding:24px'>"
                    f"<h1>Organismo no está en el roster</h1><p>id={oid}</p>"
                    f"<p><a href='/' style='color:#6db6ff'>← Volver</a></p></body>",
                    "text/html",
                )
                return
            base = (entry.get("base_url") or "").rstrip("/")
            if not base:
                self._send(404, json.dumps({"ok": False, "error": "sin base_url"}))
                return
            if public:
                try:
                    with urllib.request.urlopen(base + "/", timeout=2.0) as r:
                        body = r.read()
                        ctype = r.headers.get("Content-Type", "text/html")
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    try:
                        self.wfile.write(body)
                    except Exception:
                        pass
                    return
                except Exception:
                    name = entry.get("name") or oid
                    html = (
                        "<!doctype html><meta charset=utf-8><title>Organismo (snapshot)</title>"
                        f"<body style='font-family:system-ui;background:#0b1018;color:#dfe7f0;padding:24px'>"
                        f"<h1>{name}</h1>"
                        f"<p>El organismo está en el roster pero su dirección local no es alcanzable "
                        f"desde este Observatorio (LAN / NAT).</p>"
                        f"<p><a href='/' style='color:#6db6ff'>← Volver</a></p></body>"
                    )
                    self._send(200, html, "text/html")
                    return
            self.send_response(302)
            self.send_header("Location", base + "/")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
        elif path == "/roster":
            self._send(200, _json_dumps({"ok": True, "roster": _roster_cliente(_roster_vivo())}, ensure_ascii=False))
        elif path in ("/estado", "/csv", "/bitacora"):
            # Proxies de lectura con el mismo scrub de privacidad que /datos.
            oid = (qs.get("oid") or [""])[0]
            entry = _entry_por_id(oid)
            if not entry:
                self._send(404, json.dumps({"ok": False, "error": "organismo no en roster"}))
                return
            real = entry.get("organism_id") or ""
            pub = _oid_publico(real)
            base = (entry.get("base_url") or "").rstrip("/")
            if path == "/estado":
                with _LOCK:
                    est = dict(_ESTADOS.get(real) or {})
                if not est:
                    est = _get_json(base, "/estado") or {}
                self._send(200, _json_dumps(sanitizar_estado_publico(est, oid_publico=pub), ensure_ascii=False))
                return
            # csv / bitácora: reenviar y scrubear texto
            sub = "/export/csv" if path == "/csv" else "/export/bitacora"
            # rutas alternativas usadas por WebLive
            for cand in (sub, "/csv", "/bitacora", "/exportar/csv", "/exportar/bitacora"):
                try:
                    with urllib.request.urlopen(base + cand, timeout=6.0) as r:
                        raw = r.read()
                        ctype = r.headers.get("Content-Type", "text/plain")
                    text = raw.decode("utf-8", "replace")
                    text = sanitizar_texto_export(text)
                    # reescribir OID sensible en cabeceras
                    if real and pub and real != pub:
                        text = text.replace(real, pub)
                    self._send(200, text.encode("utf-8"), ctype.split(";")[0])
                    return
                except Exception:
                    continue
            self._send(404, json.dumps({"ok": False, "error": "export no disponible"}))
        elif path == "/voz":
            oid = (qs.get("oid") or [""])[0]
            seg = (qs.get("seg") or ["1.0"])[0]
            entry = _entry_por_id(oid)
            if not entry:
                self._send(404, json.dumps({"ok": False, "error": "organismo no en roster"}))
                return
            url = f"{entry['base_url']}/comunicacion/bloque.wav?seg={seg}&modo=R2D2"
            try:
                with urllib.request.urlopen(url, timeout=4.0) as r:
                    self._send(200, r.read(), "audio/wav")
            except Exception as e:
                self._send(502, json.dumps({"ok": False, "error": str(e)}))
        else:
            self._send(404, json.dumps({"error": "no encontrado"}))


def _load_cabeza3d_snippet() -> str:
    """Misma cabeza que :9100, con Three.js servido por este Observatorio."""
    conv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vst_conversacion.py")
    try:
        with open(conv, encoding="utf-8") as f:
            txt = f.read()
        marker = '<script id="anima-three-head">'
        i = txt.find(marker)
        if i < 0:
            return ""
        i += len(marker)
        j = txt.find("</script>", i)
        if j < 0:
            return ""
        code = txt[i:j]
        return '<script src="/vendor/three.min.js"></script>\n<script>' + code + "</script>"
    except Exception:
        return ""


_CABEZA3D = _load_cabeza3d_snippet()

HTML = r"""<!doctype html><html lang=es><head><meta charset=utf-8>
<title>Sociedad ANIMA — organismos presentes</title>
""" + _CABEZA3D + r"""
<style>
:root{--bg:#0b1018;--panel:#141e2e;--bord:#243246;--mut:#8aa0b8;--gold:#e8b86d;--ok:#5fd38a;--soc-cols:""" + str(SOC_COLS) + r""";--soc-head-ar:""" + SOC_HEAD_AR + r""";--soc-head-w:""" + SOC_HEAD_W + r"""}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#dfe7f0;font-family:system-ui,sans-serif}
h1{font-size:16px;margin:8px 12px 2px;font-weight:600}
.mut{color:var(--mut);font-size:10px}
.socbar{display:flex;align-items:center;gap:8px;padding:6px 12px;background:var(--panel);border-bottom:1px solid var(--bord);flex-wrap:wrap}
.socgrid{display:grid;grid-template-columns:repeat(var(--soc-cols),minmax(0,1fr));gap:7px;padding:8px 10px;width:100%;max-width:none;margin:0}
@media (max-width:1100px){.socgrid{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media (max-width:640px){.socgrid{grid-template-columns:repeat(2,minmax(0,1fr))}}
.soccard{background:linear-gradient(180deg,#22324f,#1a2840);border:1px solid var(--bord);border-radius:9px;padding:5px 6px 6px;text-align:center;box-shadow:0 2px 8px #00000030;transition:opacity .3s;min-width:0}
.soccard.off{opacity:.45;filter:grayscale(.55)}
.soccard h3{margin:0 0 3px;font-size:10.5px;line-height:1.15;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:flex;align-items:center;justify-content:center;gap:3px}
.socname{overflow:hidden;text-overflow:ellipsis;min-width:0}
.socflag{font-size:11px;line-height:1;flex-shrink:0}
.socpaises{display:inline-flex;gap:8px;flex-wrap:wrap;align-items:center}
.socpais{display:inline-flex;align-items:center;gap:3px;font-size:10px;color:var(--mut)}
.socpais b{color:#cfe0f5;font-weight:500}
.socbadge{display:inline-block;font-size:7px;padding:1px 4px;border-radius:5px;margin-left:3px;vertical-align:middle;background:#1d2c44;color:#9fb1c6;max-width:52px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.socbadge.presente{background:#1a3d2a;color:#8ef0c0}
.socbadge.silencioso{background:#3a3214;color:#e8b86d}
.sochead-slot{width:100%;display:flex;justify-content:center;margin:2px 0 4px}
.sochead{width:var(--soc-head-w);max-width:var(--soc-head-w);aspect-ratio:var(--soc-head-ar);height:auto;flex-shrink:0;position:relative;filter:drop-shadow(0 4px 8px rgba(0,0,0,.45))}
.sochead canvas{width:100%!important;height:100%!important;display:block}
.socvoz{font-size:8.5px;margin:2px 0;min-height:13px;line-height:1.15;color:#e9f1fb;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.socvit-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:2px;font-size:7.5px;color:#9fb1c6;margin:2px 0;padding-top:2px;border-top:1px solid #243246}
.socvit-row span{display:flex;flex-direction:column;align-items:center;gap:1px}
.socvit-row b{color:#cfe0f5;font-variant-numeric:tabular-nums;font-size:8px;font-weight:600}
.socoidos{display:flex;flex-direction:column;gap:2px;margin:3px 1px;padding:3px 4px;background:#16223a;border:1px solid #243246;border-radius:6px;text-align:left}
.socoid-row{display:flex;align-items:center;gap:3px;font-size:7.5px;color:#9fb1c6}
.socoid-side{width:10px;font-weight:700;font-size:7px;color:#7d8ea6;text-align:center;flex-shrink:0}
.socoid-src{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#cfe0f5}
.socoid-bar{width:28px;height:4px;flex-shrink:0;background:#0e1726;border-radius:3px;overflow:hidden}
.socoid-bar>div{height:100%;width:0;border-radius:3px;transition:width .15s}
.socactions{display:flex;gap:4px;justify-content:center;margin-top:5px}
.btn{padding:3px 7px;border:1px solid #35506e;background:#1d2c44;color:#dfe7f0;border-radius:6px;font-size:10px;line-height:1.2;cursor:pointer;text-decoration:none}
.btn:hover{background:#26395a}
.btn.on{background:#1a3d2a;border-color:#3a6b4a}
.empty{padding:40px 20px;text-align:center;color:var(--mut)}
</style></head><body>
<h1>🌐 Sociedad ANIMA <span class=mut>· organismos que se anuncian en el campo</span></h1>
<div class=socbar>
  <b style=color:var(--gold)>Campo de presencia</b>
  <span class=socpaises id=paisesCampo>—</span>
  <span class=mut id=socN>buscando…</span>
  <span style=flex:1></span>
  <button class=btn id=bAudio onclick="audioToggle()">🔊 Escuchar campo</button>
  <span class=mut>vol</span>
  <input type=range id=vol min=0 max=8 step=0.5 value=3 style=width:80px oninput="setVol(this.value)">
</div>
<div id=socGrid class=socgrid></div>
<div id=socEmpty class=empty style=display:none>
  <p>Ningún organismo en el campo todavía.</p>
  <p class=mut>No hay organismos vivos en el campo. Arranca ANIMA en este PC u otro de la red local.</p>
</div>
<script>
const $=id=>document.getElementById(id);
const emo=v=>({screaming:'😱',worried:'😟',excited:'🤩',sing:'🎶',acknowledged:'👍',alegria:'😊',calma:'😌',ternura:'🥰',hambre:'🍽️',compania:'🫂'}[v]||'🗣️');
function w(v){return Math.min(100,Math.sqrt(Math.max(0,+v||0))*70);}

const HEAD_FIT=""" + str(SOC_HEAD_FIT) + r""";
let built=false, lastRosterKey='', audioOn=false, ac=null, gMaster=null, players={}, heads={};

function renderPaises(paises){
  const el=$('paisesCampo');
  if(!el) return;
  if(!paises||!paises.length){el.textContent='—';return;}
  el.innerHTML=paises.map(p=>`<span class=socpais title="${p.country_name}"><span>${p.country_flag||''}</span><b>${p.country_name}</b><span class=mut>(${p.count})</span></span>`).join('');
}

function build(d){
  const roster=d.roster||[], grid=$('socGrid'), empty=$('socEmpty');
  renderPaises(d.paises_campo);
  $('socN').textContent=roster.length+' organismo(s) · campo ρ='+(+(d.campo_presencia&&d.campo_presencia.densidad)||0).toFixed(2);
  if(!roster.length){grid.innerHTML='';empty.style.display='';built=false;return;}
  empty.style.display='none';
  const key=roster.map(r=>r.organism_id).join('|');
  if(built && key===lastRosterKey) return;
  lastRosterKey=key; built=true; heads={};
  grid.innerHTML=roster.map(r=>{
    const st=(r.estado_presencia||'').toLowerCase();
    const badge=st==='presente'?'presente':(st==='silencioso'?'silencioso':'');
    heads[r.organism_id]={th:0,tt:0,eL:0,tL:0,eR:0,tR:0,cara:0};
    return `<div class=soccard id=card_${r.organism_id} data-oid="${r.organism_id}">
      <h3 style=color:${r.color}><span class=socname>${r.name}</span>${r.country_flag?`<span class=socflag title="${r.country_name||''}">${r.country_flag}</span>`:''}<span class="socbadge ${badge}">${r.estado_presencia||r.source}</span></h3>
      <div class=sochead-slot><div class=sochead id=sochead_${r.organism_id}></div></div>
      <div class=socoidos>
        <div class=socoid-row><span class=socoid-side>L</span><span class=socoid-src id=srcL_${r.organism_id}>—</span><div class=socoid-bar><div id=barL_${r.organism_id}></div></div></div>
        <div class=socoid-row><span class=socoid-side>R</span><span class=socoid-src id=srcR_${r.organism_id}>—</span><div class=socoid-bar><div id=barR_${r.organism_id}></div></div></div>
      </div>
      <div class=socvoz id=voz_${r.organism_id}>—</div>
      <div class=socvit-row>
        <span>OI<b id=oi_${r.organism_id}>—</b></span>
        <span>Ω<b id=omega_${r.organism_id}>—</b></span>
        <span>OVE<b id=cara_${r.organism_id}>—</b></span>
      </div>
      <div class=socactions>
        <a class=btn href="${r.observatorio_url}" target=_blank rel=noopener title="observatorio">🔬</a>
        <button class=btn data-play="${r.organism_id}" title="escuchar">🔊</button>
      </div>
    </div>`;
  }).join('');
  grid.querySelectorAll('[data-play]').forEach(btn=>{
    btn.onclick=()=>togglePlay(btn.dataset.play, btn);
  });
}

function update(d){
  const roster=d.roster||[], est=d.estados||{};
  if(!built || roster.length!==document.querySelectorAll('.soccard').length) build(d);
  roster.forEach(r=>{
    const e=est[r.organism_id]||{}, card=$('card_'+r.organism_id);
    if(!card) return;
    const vivo=!!e.vivo;
    card.classList.toggle('off', !vivo);
    const set=(id,v)=>{const el=$(id); if(el) el.textContent=v;};
    set('srcL_'+r.organism_id, vivo?(e.fuente_L||'—'):'—');
    set('srcR_'+r.organism_id, vivo?(e.fuente_R||'—'):'—');
    const bL=$('barL_'+r.organism_id); if(bL){bL.style.width=(vivo?w(e.energia_L):0)+'%';bL.style.background='#6db6ff';}
    const bR=$('barR_'+r.organism_id); if(bR){bR.style.width=(vivo?w(e.energia_R):0)+'%';bR.style.background='#ff8c6b';}
    set('voz_'+r.organism_id, vivo?(emo(e.voz_emitida)+' '+(e.voz_titulo||e.voz_emitida||'—')):'· en reposo ·');
    set('oi_'+r.organism_id, (+e.OI||0).toFixed(3));
    set('omega_'+r.organism_id, (+e.Omega||0).toFixed(2));
    const c=+e.cara_valoracion||0;
    set('cara_'+r.organism_id, c>0.05?'😊 favorable':(c<-0.05?'☹️ desfavorable':'😐 neutra'));
  });
}

async function tick(){
  try{
    const d=await fetch('/datos').then(r=>r.json());
    build(d); update(d);
    window._ultD=d;
  }catch(e){ console.error('[Sociedad ANIMA] /datos inválido o inaccesible:', e); }
}
setInterval(tick, 500); tick();

function animHeads(){
  requestAnimationFrame(animHeads);
  const roster=(window._ultD&&window._ultD.roster)||[];
  const est=(window._ultD&&window._ultD.estados)||{};
  if(!window.drawVSTCabeza3DReal) return;
  roster.forEach(r=>{
    const e=est[r.organism_id]||{}, h=heads[r.organism_id], c=$('sochead_'+r.organism_id);
    if(!h||!c) return;
    const vivo=!!e.vivo;
    const _oid=+e.oao_oido||0, _ar=+e.voz_arousal||0, _eb=Math.max(_oid,_ar*0.6);
    h.tt=vivo?(+e.orientacion_deg||0):0;
    h.tL=vivo?(e.energia_L!=null?+e.energia_L:_eb):0;
    h.tR=vivo?(e.energia_R!=null?+e.energia_R:_eb):0;
    h.cara=vivo?Math.sign(+e.cara_valoracion||0):0;
    h.th+=(h.tt-h.th)*0.12;
    h.eL+=(h.tL-h.eL)*0.18;
    h.eR+=(h.tR-h.eR)*0.18;
    // Preferir apariencia viva del /estado (Predator rosado, etc.); roster solo como fallback.
    const gen=e.cara_genero||e.genero||r.genero||'masculino';
    const ton=e.cara_tono||e.tono||r.tono||'blanco';
    window.drawVSTCabeza3DReal(c,h.th,{energiaL:h.eL,energiaR:h.eR,cara:h.cara,fitScale:HEAD_FIT,genero:gen,tono:ton});
  });
}
requestAnimationFrame(animHeads);

function setVol(v){ if(gMaster) gMaster.gain.value=+v; }

function togglePlay(oid, btn){
  if(players[oid]&&players[oid].on){
    players[oid].on=false; btn.classList.remove('on'); btn.textContent='🔊'; return;
  }
  if(!ac){ ac=new (window.AudioContext||window.webkitAudioContext)(); gMaster=ac.createGain(); gMaster.gain.value=+($('vol').value||3); gMaster.connect(ac.destination); ac.resume(); }
  if(!players[oid]) players[oid]={on:true, next:0, gain:ac.createGain()};
  const p=players[oid]; p.on=true; p.gain.gain.value=1; p.gain.connect(gMaster); p.next=ac.currentTime;
  btn.classList.add('on'); btn.textContent='⏸';
  (async function loop(){
    if(!p.on) return;
    try{
      const ab=await fetch('/voz?oid='+encodeURIComponent(oid)).then(r=>r.arrayBuffer());
      const buf=await ac.decodeAudioData(ab.slice(0));
      const s=ac.createBufferSource(); s.buffer=buf; s.connect(p.gain);
      const t=Math.max(ac.currentTime+0.02,p.next); s.start(t); p.next=t+buf.duration;
      setTimeout(loop, Math.max(80,(p.next-ac.currentTime-0.1)*1000));
    }catch(e){ setTimeout(loop, 900); }
  })();
}

function audioToggle(){
  const roster=(window._ultD&&window._ultD.roster)||[];
  if(audioOn){
    audioOn=false; Object.values(players).forEach(p=>p.on=false);
    document.querySelectorAll('[data-play]').forEach(b=>{b.classList.remove('on');b.textContent='🔊';});
    $('bAudio').classList.remove('on'); $('bAudio').textContent='🔊 Escuchar campo'; return;
  }
  audioOn=true; $('bAudio').classList.add('on'); $('bAudio').textContent='⏸ Detener campo';
  roster.forEach(r=>{ const b=document.querySelector('[data-play="'+r.organism_id+'"]'); if(b) togglePlay(r.organism_id,b); });
}
</script></body></html>"""


def main():
    global _ROSTER
    _ROSTER = _catalogo_inicial()  # vacío salvo ANIMA_SOCIEDAD_CATALOGO=1
    threading.Thread(target=_poller, daemon=True).start()
    print(f"[sociedad] observatorio dinámico en http://0.0.0.0:{PORT}", flush=True)
    print(f"[sociedad] semillas: {', '.join(SEED_URLS)}", flush=True)
    print(f"[sociedad] catálogo lab: {'ON' if SOCIEDAD_CATALOGO else 'OFF (solo presencia real)'}", flush=True)
    print(f"[sociedad] grid: {SOC_COLS} columnas · cabeza {SOC_HEAD_W} de caja × {SOC_HEAD_AR} · fit {SOC_HEAD_FIT}", flush=True)
    print(f"[sociedad] modo: {'público (sin IPs)' if SOCIEDAD_PUBLIC else 'desarrollo'}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
