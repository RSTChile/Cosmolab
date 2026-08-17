#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_OrganoPresencia — descubrimiento local de otros organismos ANIMA (Fase 2 spec).

Capa biológica: campo de presencia, aislamiento, hambre social, foco comunicativo.
Capa tecnológica intercambiable: mDNS (_anima._tcp) + UDP 47788 fallback.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlparse

try:
    from VST_AparienciaCara import apariencia_desde_identity, normalizar_nombre_organismo
except Exception:
    def apariencia_desde_identity(identity=None):  # type: ignore[misc]
        return {"schema": "anima.appearance.v1", "genero": "masculino", "tono": "blanco",
                "cara_genero": "masculino", "cara_tono": "blanco"}
    def normalizar_nombre_organismo(raw=None, **kw):  # type: ignore[misc]
        return (raw or "Animalito").strip()[:14]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _local_ip() -> str:
    """IP de la ruta por defecto (preferida para base_url)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _local_ipv4s() -> list[str]:
    """Todas las IPv4 no-loopback del host (multi-homing)."""
    found: list[str] = []
    seen: set[str] = set()
    # Preferir primero la de la ruta por defecto
    primary = _local_ip()
    if primary and not primary.startswith("127."):
        found.append(primary)
        seen.add(primary)
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_STREAM):
            ip = info[4][0]
            if not ip or ip.startswith("127.") or ip in seen:
                continue
            seen.add(ip)
            found.append(ip)
    except Exception:
        pass
    # Interfaces vía getaddrinfo de hostname vacío / 0.0.0.0 a veces no listan todo;
    # completar con nombres conocidos de host.
    try:
        for info in socket.getaddrinfo(socket.getfqdn(), None, socket.AF_INET, socket.SOCK_DGRAM):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and ip not in seen:
                seen.add(ip)
                found.append(ip)
    except Exception:
        pass
    return found or ([primary] if primary else ["127.0.0.1"])


def _same_subnet(a: str, b: str, mask_bits: int = 24) -> bool:
    try:
        pa = socket.inet_aton(a)
        pb = socket.inet_aton(b)
        if mask_bits == 24:
            return pa[:3] == pb[:3]
        if mask_bits == 16:
            return pa[:2] == pb[:2]
        return pa == pb
    except OSError:
        return False


def _prefer_host(candidates: list[str], local_ips: list[str] | None = None) -> str:
    """Elige IP del vecino preferiblemente en la misma subred que alguna NIC local."""
    cands = [c for c in candidates if c and not c.startswith("127.")]
    if not cands:
        return (candidates[0] if candidates else "") or ""
    locals_ = local_ips or _local_ipv4s()
    for lip in locals_:
        for c in cands:
            if _same_subnet(c, lip, 24):
                return c
    return cands[0]


def _load_identity(path: str | None = None) -> dict[str, Any]:
    path = path or os.path.join(os.path.expanduser("~"), ".anima", "identity.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@dataclass
class Vecino:
    organism_id: str
    installation_id: str
    name: str
    base_url: str
    host: str
    port: int
    state: str = "atento"
    visibility: str = "local"
    capabilities: list[str] = field(default_factory=list)
    public_key: dict[str, Any] | None = None
    first_seen: float = 0.0
    last_seen: float = 0.0
    confianza: float = 0.7
    source: str = "mdns"

    def fresco(self, ttl_s: float) -> float:
        if self.last_seen <= 0:
            return 0.0
        edad = max(0.0, time.time() - self.last_seen)
        return max(0.0, 1.0 - edad / max(1.0, ttl_s))

    def estado_presencia(self, ttl_s: float) -> str:
        edad = max(0.0, time.time() - self.last_seen) if self.last_seen else 1e9
        if edad <= ttl_s:
            return "presente"
        if edad <= 3 * ttl_s:
            return "silencioso"
        return "ausente"


class OrganoPresencia:
    SERVICE_TYPE = "_anima._tcp.local."
    UDP_PORT = 47788

    def __init__(
        self,
        organism_id: str,
        organism_name: str,
        http_port: int,
        *,
        bind_host: str = "0.0.0.0",
        voice_gain: float = 1.0,
    ):
        self.organism_id = organism_id
        self.organism_name = normalizar_nombre_organismo(organism_name)
        self.http_port = int(http_port)
        self.bind_host = bind_host
        self.voice_gain = voice_gain
        self.mode = os.environ.get("ANIMA_PRESENCE_MODE", "local").strip().lower()
        self.visibility = os.environ.get("ANIMA_VISIBILITY", "local").strip().lower()
        self.ttl_s = _env_float("ANIMA_PRESENCE_TTL_S", 30.0)
        self.aislamiento_sat = _env_float("ANIMA_AISLAMIENTO_TIEMPO_SATURACION", 300.0)
        self.identity = _load_identity()
        self.installation_id = self.identity.get("installation_id", "")
        # Plaza pública: POST heartbeat al Observatorio (colaboradores fuera de la LAN).
        # Default: observatorio.cosmosemiotica.cl salvo ANIMA_PLAZA=0 o URL vacía forzada.
        plaza_flag = os.environ.get("ANIMA_PLAZA", "1").strip().lower()
        self.plaza_url = (os.environ.get("ANIMA_PLAZA_URL") or "").strip().rstrip("/")
        if plaza_flag in ("0", "false", "off", "no"):
            self.plaza_url = ""
        elif not self.plaza_url:
            # Activar plaza si visibility lo pide, o si el modo es both/plaza, o perfil instalable.
            want_plaza = (
                self.visibility in ("plaza_publico", "plaza_latente", "experimento")
                or self.mode in ("plaza", "both", "all", "local+plaza")
                or os.environ.get("ANIMA_UI_PERFIL", "").strip().lower() == "limpio"
            )
            if want_plaza:
                self.plaza_url = "https://observatorio.cosmosemiotica.cl"
        self.plaza_interval_s = _env_float("ANIMA_PLAZA_INTERVAL_S", 20.0)
        self.plaza_ttl_s = _env_float("ANIMA_PLAZA_HEARTBEAT_TTL_S", 90.0)
        self._plaza_last_ok = 0.0
        self._plaza_last_err = ""
        # Fase 6: relay de voz por el Observatorio (organismos fuera de la LAN se escuchan).
        self._mi_oid_publico = ""
        self.plaza_voz_seg = _env_float("ANIMA_PLAZA_VOZ_SEG", 1.0)
        self.plaza_voz_interval_s = _env_float("ANIMA_PLAZA_VOZ_INTERVAL_S", 1.0)
        self.plaza_roster_interval_s = _env_float("ANIMA_PLAZA_ROSTER_INTERVAL_S", 2.0)
        # Amable con enlaces lentos (celular): empujar solo al vocalizar + aligerar el WAV.
        self.plaza_voz_solo_vocalizando = os.environ.get(
            "ANIMA_PLAZA_VOZ_SOLO_VOCALIZANDO", "1").strip().lower() in ("1", "true", "yes", "on")
        self.plaza_voz_mono = os.environ.get(
            "ANIMA_PLAZA_VOZ_MONO", "1").strip().lower() in ("1", "true", "yes", "on")
        self.plaza_voz_sr = _env_int("ANIMA_PLAZA_VOZ_SR", 16000)
        self.plaza_voz_max_bytes = _env_int("ANIMA_PLAZA_VOZ_MAX_BYTES", 200000)
        self._plaza_voz_last_ok = 0.0
        self._plaza_voz_last_err = ""
        self._plaza_roster_last_err = ""
        # Fase 2 Atención Social: foco social actual (a quién atiende / a quién habla) → grafo del Observatorio.
        self.atiende_a: str | None = None
        self.habla_a: str | None = None
        self._last_signals: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._vecinos: dict[str, Vecino] = {}
        self._known_installations: set[str] = set()
        self._sin_vecinos_desde: float | None = None
        self._novedad_until = 0.0
        self._retorno_until = 0.0
        self._campo = {
            "densidad": 0.0,
            "proximidad": 0.0,
            "persistencia": 0.0,
            "confianza": 0.0,
            "novedad": 0.0,
            "aislamiento": 0.0,
        }
        self._zeroconf = None
        self._browser = None
        self._service_info = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._udp_sock: socket.socket | None = None
        # mode=off apaga mDNS/UDP y plaza; mode=plaza solo plaza (sin LAN).
        self._activo = self.mode not in ("off", "0", "false", "no")
        self._local_discovery = self._activo and self.mode not in ("plaza",)

    @property
    def activo(self) -> bool:
        return self._activo

    def _base_url(self, host: str | None = None, port: int | None = None) -> str:
        # En Docker la IP del contenedor no es alcanzable desde la LAN/Observatorio.
        # Preferir URL/host publicable por env (p.ej. ANIMA_PUBLIC_BASE_URL=http://192.168.86.83:7788).
        explicit = (os.environ.get("ANIMA_PUBLIC_BASE_URL") or os.environ.get("ANIMA_ADVERTISE_URL") or "").strip()
        if explicit and host is None and port is None:
            return explicit.rstrip("/")
        h = host or (os.environ.get("ANIMA_ADVERTISE_HOST") or "").strip() or _local_ip()
        p = port or self.http_port
        return f"http://{h}:{p}"

    def _capabilities(self) -> list[str]:
        caps = ["audio_local", "voz", "comunicacion"]
        if os.environ.get("ANIMA_SDR_ENABLE", "0").lower() in ("1", "true", "yes", "on"):
            caps.append("sdr")
        return caps

    def identidad(self) -> dict[str, Any]:
        pub = self.identity.get("public_key") if self.identity else None
        return {
            "schema": "anima.identity.v1",
            "organism_id": self.organism_id,
            "installation_id": self.installation_id,
            "name": self.organism_name,
            "species": "ANIMA",
            "state": "activo",
            "visibility": self.visibility,
            "capabilities": self._capabilities(),
            "public_key": pub,
            "local_endpoints": {
                "base_url": self._base_url(),
                "identity": "/identidad",
                "state": "/estado",
                "voice": "/comunicacion/bloque.wav",
                "presence": "/presencia",
            },
            "appearance": apariencia_desde_identity(self.identity),
        }

    def _registrar_vecino(self, data: dict[str, Any], source: str = "mdns") -> None:
        oid = (data.get("organism_id") or "").strip()
        iid = (data.get("installation_id") or "").strip()
        if not oid or oid == self.organism_id:
            return
        if iid and iid == self.installation_id:
            return
        host = (data.get("host") or data.get("address") or "").strip()
        port = int(data.get("port") or self.http_port)
        if not host:
            ep = data.get("local_endpoints") or {}
            base = ep.get("base_url") or data.get("base_url") or ""
            if base:
                pu = urlparse(base)
                host = pu.hostname or ""
                port = pu.port or port
        # Candidatos multi-IP (mDNS puede anunciar varias)
        candidates: list[str] = []
        for key in ("hosts", "addresses", "ips"):
            raw = data.get(key)
            if isinstance(raw, (list, tuple)):
                candidates.extend(str(x).strip() for x in raw if x)
        if host:
            candidates.insert(0, host)
        host = _prefer_host(candidates, _local_ipv4s()) if candidates else host
        if not host:
            return
        base_url = (data.get("base_url") or "").strip().rstrip("/")
        if not base_url or urlparse(base_url).hostname in (None, ""):
            base_url = f"http://{host}:{port}"
        else:
            # Si base_url trae host, preferir el host ya filtrado por subred
            pu = urlparse(base_url)
            if pu.hostname and pu.hostname != host and candidates:
                base_url = f"http://{host}:{port}"
        now = time.time()
        # Validar con HTTP /identidad cuando sea posible: frescura real, no solo anuncio
        conf_base = 0.75 if source == "mdns" else 0.55
        if source in ("mdns", "udp", "refresh"):
            ident = self._fetch_identidad(base_url)
            if ident and ident.get("organism_id"):
                oid = (ident.get("organism_id") or oid).strip() or oid
                iid = (ident.get("installation_id") or iid).strip() or iid
                if oid == self.organism_id:
                    return
                if iid and iid == self.installation_id:
                    return
                data = {**data, **ident}
                conf_base = min(0.95, conf_base + 0.2)
                eps = ident.get("local_endpoints") or {}
                if eps.get("base_url"):
                    # Mantener host preferido si la identidad trae otra IP
                    alt = urlparse(str(eps["base_url"])).hostname or ""
                    if alt and _prefer_host([host, alt], _local_ipv4s()) == alt:
                        host = alt
                        base_url = f"http://{host}:{port}"
            # Si falla HTTP, aún registramos el anuncio pero con menor confianza
            elif source == "refresh":
                return
        with self._lock:
            prev = self._vecinos.get(oid)
            nuevo = Vecino(
                organism_id=oid,
                installation_id=iid,
                name=data.get("name") or (prev.name if prev else oid),
                base_url=base_url,
                host=host,
                port=port,
                state=data.get("state", "atento"),
                visibility=data.get("visibility", "local"),
                capabilities=list(data.get("capabilities") or (prev.capabilities if prev else [])),
                public_key=data.get("public_key"),
                first_seen=prev.first_seen if prev else now,
                last_seen=now,
                confianza=conf_base,
                source=source if source != "refresh" else (prev.source if prev else "mdns"),
            )
            if iid and iid not in self._known_installations:
                self._known_installations.add(iid)
                self._novedad_until = now + 10.0
            elif prev and prev.estado_presencia(self.ttl_s) == "ausente":
                self._retorno_until = now + 10.0
            self._vecinos[oid] = nuevo
            self._sin_vecinos_desde = None

    def _fetch_identidad(self, base_url: str) -> dict[str, Any] | None:
        try:
            with urllib.request.urlopen(f"{base_url.rstrip('/')}/identidad", timeout=1.5) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            return None

    def _purge_stale(self) -> None:
        now = time.time()
        with self._lock:
            muertos = [
                k for k, v in self._vecinos.items()
                if v.estado_presencia(self.ttl_s) == "ausente"
            ]
            for k in muertos:
                del self._vecinos[k]

    def _vecinos_vivos(self) -> list[Vecino]:
        self._purge_stale()
        with self._lock:
            return [
                v for v in self._vecinos.values()
                if v.estado_presencia(self.ttl_s) != "ausente"
            ]

    def presencia(self) -> dict[str, Any]:
        vecinos = self._vecinos_vivos()
        campo = dict(self._campo)
        return {
            "ok": True,
            "mode": self.mode,
            "ttl_s": self.ttl_s,
            "organism_id": self.organism_id,
            "vecinos_n": len(vecinos),
            "vecinos": [
                {
                    "organism_id": v.organism_id,
                    "installation_id": v.installation_id,
                    "name": v.name,
                    "base_url": v.base_url,
                    "state": v.state,
                    "estado_presencia": v.estado_presencia(self.ttl_s),
                    "confianza": round(v.confianza, 3),
                    "frescura": round(v.fresco(self.ttl_s), 3),
                    "capabilities": v.capabilities,
                    "source": v.source,
                }
                for v in vecinos
            ],
            "campo_presencia": campo,
        }

    def observar(self, fila: dict[str, Any], dt: float = 0.1) -> dict[str, Any]:
        vecinos = self._vecinos_vivos()
        n = len(vecinos)
        now = time.time()
        if n == 0:
            if self._sin_vecinos_desde is None:
                self._sin_vecinos_desde = now
            seg = now - self._sin_vecinos_desde
            aislamiento = min(1.0, seg / max(1.0, self.aislamiento_sat))
        else:
            aislamiento = max(0.0, self._campo.get("aislamiento", 0.0) - dt / 60.0)
            self._sin_vecinos_desde = None
        confianza = 0.0
        if vecinos:
            confianza = sum(v.confianza * v.fresco(self.ttl_s) for v in vecinos) / len(vecinos)
        # DENSIDAD SOCIAL — hipérbola saturante, nunca 1. Era min(1, n/4): con cuatro vecinos
        # valía exactamente 1,0 y ahí se quedaba, así que en una plaza de ocho marcaba 1,000 el
        # 100 % del tiempo en los ocho organismos (medido el 11-ago-2026 sobre 24 h). Un indicador
        # que no se mueve no informa de nada. La forma correcta no tiene tope, porque no existe un
        # número máximo de otros: n/(n+1) crece con cada vecino y se acerca a 1 sin alcanzarlo
        # jamás — 0 solo · 0,500 con uno · 0,667 con dos · 0,875 con siete. Es la misma forma con
        # que la naturaleza resuelve toda recepción de capacidad limitada (Michaelis-Menten, la
        # respuesta funcional de tipo II): el primer otro cambia el mundo más que el octavo, pero
        # el octavo todavía cuenta. El 1,0 del denominador no es un umbral —no decide nada— sino
        # la unidad en que se cuenta: la mitad de la escala está en el primer vecino.
        densidad = n / (n + 1.0)
        proximidad = min(1.0, sum(v.fresco(self.ttl_s) for v in vecinos) / max(1, n))
        novedad = 1.0 if now < self._novedad_until else 0.0
        retorno = 1.0 if now < self._retorno_until else 0.0
        hambre_social = max(0.0, min(1.0, 0.65 * aislamiento + 0.35 * float(fila.get("necesidad", 0.0) or 0.0)))
        self._campo = {
            "densidad": densidad,
            "proximidad": proximidad,
            "persistencia": proximidad,
            "confianza": confianza,
            "novedad": novedad,
            "aislamiento": aislamiento,
        }
        out = {
            "presencia_vivo": 1.0 if (self._activo or bool(self.plaza_url)) else 0.0,
            "presencia_vecinos_n": n,
            "presencia_local_n": n,
            "presencia_global_n": 0,
            "presencia_confiable_n": sum(1 for v in vecinos if v.confianza >= 0.6),
            "presencia_confianza": round(confianza, 4),
            "presencia_aislamiento": round(aislamiento, 4),
            "presencia_retorno": retorno,
            "presencia_novedad": novedad,
            "presencia_densidad": round(densidad, 4),
            "presencia_proximidad": round(proximidad, 4),
            "hambre_social": round(hambre_social, 4),
            "comunicacion_foco": round(max(confianza, novedad, retorno), 4),
        }
        with self._lock:
            self._last_signals = {
                "arousal": float(fila.get("voz_arousal") or fila.get("arousal") or 0.0) if fila else 0.0,
                "valence": float(fila.get("voz_valence") or fila.get("valence") or 0.0) if fila else 0.0,
                "energy": float(fila.get("energia") or 0.0) if fila else 0.0,
                "voice_active": bool((fila or {}).get("voz_titulo") and str((fila or {}).get("voz_titulo")) not in ("-", "")),
                "presencia_densidad": round(densidad, 4),
                "hambre_social": round(hambre_social, 4),
            }
        return out

    def urls_comunicacion(self) -> list[dict[str, str]]:
        out = []
        for v in self._vecinos_vivos():
            if v.source == "plaza" and self.plaza_url:
                # Vecino fuera de la LAN: su voz se sirve por el relay del Observatorio.
                base = self.plaza_url.rstrip("/")
                url = f"{base}/voz?oid={quote(v.organism_id, safe='')}&seg={self.plaza_voz_seg}"
            else:
                sep = "&" if "?" in v.base_url else "?"
                url = f"{v.base_url.rstrip('/')}/comunicacion/bloque.wav{sep}modo=R2D2&gain={self.voice_gain}"
            letra = v.name[:1].upper() if v.name else v.organism_id.split("_")[-1][:1].upper()
            out.append({
                "tipo": "comunicacion",
                "url": url,
                "nombre": f"voz de {v.name}",
                "modo": "R2D2",
                "label": f"🗣 Escuchar {v.name}",
                "organism_id": v.organism_id,
                "letra": letra,
            })
        return out

    def fuente_otros_organismos(self) -> list[dict[str, Any]]:
        urls = [x["url"] for x in self.urls_comunicacion()]
        if not urls:
            return []
        return [{
            "tipo": "otros_organismos",
            "urls": urls,
            "nombre": "otros organismos",
            "label": f"\U0001f9d1\u200d\U0001f91d\u200d\U0001f9d1 Otros organismos \u2014 {len(urls)} en el campo",
        }]

    def _udp_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", self.UDP_PORT))
        except OSError as e:
            print(f"[presencia] UDP bind :{self.UDP_PORT} falló: {e}", flush=True)
            return
        self._udp_sock = sock
        sock.settimeout(1.0)
        print(f"[presencia] UDP escuchando :{self.UDP_PORT}", flush=True)
        while not self._stop.is_set():
            try:
                data, addr = sock.recvfrom(65535)
                msg = json.loads(data.decode("utf-8"))
                if isinstance(msg, dict) and msg.get("schema") == "anima.presence.udp.v1":
                    # Preferir host declarado en payload; fallback al origen del datagrama
                    if not (msg.get("host") or msg.get("base_url")):
                        msg["host"] = addr[0]
                    elif not msg.get("host"):
                        try:
                            msg["host"] = urlparse(str(msg.get("base_url") or "")).hostname or addr[0]
                        except Exception:
                            msg["host"] = addr[0]
                    self._registrar_vecino(msg, source="udp")
            except socket.timeout:
                continue
            except Exception:
                continue

    def _udp_beacon(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while not self._stop.is_set():
            ips = _local_ipv4s()
            primary = ips[0] if ips else _local_ip()
            payload = {
                "schema": "anima.presence.udp.v1",
                "organism_id": self.organism_id,
                "installation_id": self.installation_id,
                "name": self.organism_name,
                "host": primary,
                "hosts": ips,
                "port": self.http_port,
                "base_url": self._base_url(primary),
                "visibility": self.visibility,
                "capabilities": self._capabilities(),
            }
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            try:
                sock.sendto(data, ("<broadcast>", self.UDP_PORT))
                # Broadcast dirigido a /24 de cada NIC (más fiable multi-homing Windows)
                for ip in ips:
                    try:
                        parts = ip.split(".")
                        if len(parts) == 4 and not ip.startswith("127."):
                            bcast = f"{parts[0]}.{parts[1]}.{parts[2]}.255"
                            sock.sendto(data, (bcast, self.UDP_PORT))
                    except Exception:
                        pass
            except Exception as e:
                print(f"[presencia] UDP beacon: {e}", flush=True)
            self._stop.wait(5.0)
        sock.close()

    def _mdns_register(self) -> None:
        try:
            from zeroconf import IPVersion, ServiceInfo, Zeroconf
        except ImportError:
            print("[presencia] zeroconf no instalado; solo UDP", flush=True)
            return
        ips = _local_ipv4s()
        packed: list[bytes] = []
        for ip in ips:
            try:
                packed.append(socket.inet_aton(ip))
            except OSError:
                continue
        if not packed:
            print("[presencia] mDNS: sin IPv4 para anunciar", flush=True)
            return
        props = {
            b"schema": b"anima.presence.v1",
            b"organism_id": self.organism_id.encode("utf-8"),
            b"installation_id": (self.installation_id or "").encode("utf-8"),
            b"name": self.organism_name.encode("utf-8"),
            b"visibility": self.visibility.encode("utf-8"),
            b"identity_path": b"/identidad",
            b"presence_path": b"/presencia",
            b"base_url": self._base_url(ips[0]).encode("utf-8"),
        }
        name = f"{self.organism_id}.{self.SERVICE_TYPE}"
        info = ServiceInfo(
            self.SERVICE_TYPE,
            name,
            addresses=packed,
            port=self.http_port,
            properties=props,
            server=f"{self.organism_id}.local.",
        )
        zc = Zeroconf(ip_version=IPVersion.V4Only)
        zc.register_service(info)
        self._zeroconf = zc
        self._service_info = info
        print(f"[presencia] mDNS registrado en {', '.join(ips)}:{self.http_port}", flush=True)

    def _mdns_browse(self) -> None:
        try:
            from zeroconf import ServiceBrowser, Zeroconf
        except ImportError:
            return

        class _Listener:
            def __init__(self, organo: OrganoPresencia):
                self.organo = organo

            def add_service(self, zc, type_, name):
                self.organo._on_mdns_service(zc, name)

            def remove_service(self, zc, type_, name):
                pass

            def update_service(self, zc, type_, name):
                self.organo._on_mdns_service(zc, name)

        zc = self._zeroconf or __import__("zeroconf").Zeroconf()
        if self._zeroconf is None:
            self._zeroconf = zc
        self._browser = ServiceBrowser(zc, self.SERVICE_TYPE, _Listener(self))
        self._mdns_listener = _Listener(self)

    def _on_mdns_service(self, zc, name: str) -> None:
        try:
            info = zc.get_service_info(self.SERVICE_TYPE, name, timeout=1500)
        except Exception:
            return
        if not info:
            return
        props = {}
        for k, v in (info.properties or {}).items():
            key = k.decode("utf-8") if isinstance(k, bytes) else str(k)
            val = v.decode("utf-8") if isinstance(v, bytes) else str(v)
            props[key] = val
        addrs: list[str] = []
        for raw in (info.addresses or []):
            try:
                addrs.append(socket.inet_ntoa(raw))
            except Exception:
                continue
        host = _prefer_host(addrs, _local_ipv4s()) if addrs else ""
        data = {
            "organism_id": props.get("organism_id", ""),
            "installation_id": props.get("installation_id", ""),
            "name": props.get("name", props.get("organism_id", "")),
            "host": host,
            "hosts": addrs,
            "port": info.port,
            "visibility": props.get("visibility", "local"),
            "base_url": props.get("base_url") or (f"http://{host}:{info.port}" if host else ""),
        }
        self._registrar_vecino(data, source="mdns")

    def _refresh_loop(self) -> None:
        """Re-query de vecinos conocidos + re-browse mDNS periódico (multi-homing / late join)."""
        interval = _env_float("ANIMA_PRESENCE_REFRESH_S", 60.0)
        while not self._stop.is_set():
            self._stop.wait(max(15.0, interval))
            if self._stop.is_set():
                break
            # Refrescar vecinos ya conocidos con HTTP /identidad
            with self._lock:
                snapshot = list(self._vecinos.values())
            for v in snapshot:
                try:
                    self._registrar_vecino({
                        "organism_id": v.organism_id,
                        "installation_id": v.installation_id,
                        "name": v.name,
                        "host": v.host,
                        "port": v.port,
                        "base_url": v.base_url,
                        "visibility": v.visibility,
                        "capabilities": v.capabilities,
                    }, source="refresh")
                except Exception:
                    pass
            # Re-registrar mDNS por si cambió la IP de la NIC
            try:
                if self._zeroconf and self._service_info:
                    ips = _local_ipv4s()
                    packed = []
                    for ip in ips:
                        try:
                            packed.append(socket.inet_aton(ip))
                        except OSError:
                            continue
                    if packed:
                        self._service_info.addresses = packed
                        try:
                            self._zeroconf.update_service(self._service_info)
                        except Exception:
                            pass
            except Exception as e:
                print(f"[presencia] refresh mDNS: {e}", flush=True)

    def set_foco_social(self, atiende_a: str | None, habla_a: str | None) -> None:
        """El organelo de Atención Social publica aquí su decisión; el heartbeat la reporta al Observatorio."""
        self.atiende_a = atiende_a or None
        self.habla_a = habla_a or None

    def _plaza_payload(self) -> dict[str, Any]:
        """Snapshot mínimo + señales para el Observatorio público."""
        app = apariencia_desde_identity(self.identity)
        country = (os.environ.get("ANIMA_DEFAULT_COUNTRY") or "CL").upper()[:2]
        # Intentar enriquecer con /estado local (mismo proceso).
        estado: dict[str, Any] = {}
        vivo = True
        try:
            url = f"http://127.0.0.1:{int(self.http_port)}/estado"
            with urllib.request.urlopen(url, timeout=1.2) as r:
                estado = json.loads(r.read().decode("utf-8"))
            if isinstance(estado, dict) and "vivo" in estado:
                vivo = bool(estado.get("vivo"))
        except Exception:
            estado = {}
            vivo = True  # proceso vivo aunque /estado falle un instante
        # Subconjunto seguro (sin IPs / rutas)
        keep = (
            "vivo", "t", "voz_titulo", "voz_emitida", "voz_origen", "OI", "H",
            "energia", "necesidad", "RC_total", "cara_valoracion", "cara_modo",
            "orientacion_deg", "pitch_deg", "ove_valoracion_actual",
            "organismo", "genero", "tono", "cara_genero", "cara_tono",
            "fuente_L", "fuente_R",   # a qué escucha cada oído (para las cajas del Observatorio)
        )
        est_pub = {k: estado[k] for k in keep if isinstance(estado, dict) and k in estado}
        est_pub["vivo"] = vivo
        with self._lock:
            signals = dict(self._last_signals)
        return {
            "schema": "anima.plaza.heartbeat.v1",
            "protocol_version": "1.0.0",
            "organism_id": self.organism_id,
            "installation_id": self.installation_id,
            "name": self.organism_name,
            "species": "ANIMA",
            "state": "activo" if vivo else "silencioso",
            "visibility": self.visibility if self.visibility.startswith("plaza") else "plaza_publico",
            "vivo": vivo,
            "ttl_s": int(max(30, min(self.plaza_ttl_s, 3600))),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "capabilities": self._capabilities(),
            "appearance": app,
            "genero": app.get("genero"),
            "tono": app.get("tono"),
            "cara_genero": app.get("cara_genero"),
            "cara_tono": app.get("cara_tono"),
            "country_code": country,
            "atiende_a": self.atiende_a,   # a quién ESCUCHA (grafo de atención → prestigio/conformidad)
            "habla_a": self.habla_a,       # a quién HABLA (dirección intencional, modelo A)
            "signals": signals,
            "estado": est_pub,
        }

    def _plaza_send_once(self) -> bool:
        if not self.plaza_url:
            return False
        body = json.dumps(self._plaza_payload(), ensure_ascii=False).encode("utf-8")
        url = self.plaza_url.rstrip("/") + "/plaza/heartbeat"
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                # UA tipo navegador: Cloudflare a veces responde 403 a clientes “bot”.
                "User-Agent": "Mozilla/5.0 (compatible; ANIMA-Presencia/1.0; +https://cosmosemiotica.cl)",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=8.0) as r:
                raw = r.read().decode("utf-8", "replace")
            try:
                resp = json.loads(raw) if raw else {}
            except Exception:
                resp = {}
            if isinstance(resp, dict) and resp.get("ok"):
                self._plaza_last_ok = time.time()
                self._plaza_last_err = ""
                pid = (resp.get("public_id") or "").strip()
                if pid:
                    self._mi_oid_publico = pid
                return True
            self._plaza_last_err = str(resp.get("error") or raw[:160] or f"http {getattr(r, 'status', '?')}")
            return False
        except Exception as e:
            self._plaza_last_err = f"{type(e).__name__}: {e}"
            return False

    def _plaza_loop(self) -> None:
        # Primer envío casi inmediato (descubrimiento rápido en el Observatorio).
        if self._stop.wait(2.0):
            return
        while not self._stop.is_set():
            try:
                ok = self._plaza_send_once()
                if ok:
                    # Log ocasional (cada ~2 min)
                    if int(time.time()) % 120 < self.plaza_interval_s:
                        print(f"[presencia] plaza OK → {self.plaza_url}/plaza/heartbeat", flush=True)
                else:
                    print(f"[presencia] plaza fail: {self._plaza_last_err}", flush=True)
            except Exception as e:
                print(f"[presencia] plaza loop: {e}", flush=True)
            if self._stop.wait(max(5.0, float(self.plaza_interval_s))):
                return

    # --- Fase 6: relay de voz por el Observatorio -------------------------------

    def _fetch_voz_local(self) -> bytes | None:
        """Bloque de voz actual del propio organismo (servidor HTTP local)."""
        try:
            url = (
                f"http://127.0.0.1:{int(self.http_port)}/comunicacion/bloque.wav"
                f"?seg={self.plaza_voz_seg}&modo=R2D2&gain={self.voice_gain}"
            )
            with urllib.request.urlopen(url, timeout=5.0) as r:
                data = r.read()
            return data if data and len(data) >= 64 else None
        except Exception as e:
            self._plaza_voz_last_err = f"local voz: {type(e).__name__}"
            return None

    def _voz_activa(self) -> bool:
        """¿El organismo está vocalizando ahora? Evita subir silencio por enlaces lentos."""
        try:
            url = f"http://127.0.0.1:{int(self.http_port)}/estado"
            with urllib.request.urlopen(url, timeout=1.5) as r:
                est = json.loads(r.read().decode("utf-8", "replace"))
        except Exception:
            # Si no puedo consultarlo, uso la última señal de presencia (o asumo activo).
            with self._lock:
                return bool(self._last_signals.get("voice_active", True))
        for k in ("voz_emitida", "voz_titulo"):
            v = est.get(k)
            if v is not None and str(v).strip() not in ("", "-"):
                return True
        return False

    def _aligerar_wav(self, wav: bytes) -> bytes:
        """Reduce el WAV a mono + baja tasa para que quepa en enlaces lentos (celular).
        Solo stdlib (wave + audioop). Si algo falla, devuelve el original intacto."""
        if not self.plaza_voz_mono and int(self.plaza_voz_sr) <= 0:
            return wav
        try:
            import io
            import wave
            import audioop
            with wave.open(io.BytesIO(wav), "rb") as w:
                ch, sw, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
                frames = w.readframes(w.getnframes())
            if sw != 2:
                return wav  # solo PCM16
            if self.plaza_voz_mono and ch == 2:
                frames = audioop.tomono(frames, 2, 0.5, 0.5)
                ch = 1
            target_sr = int(self.plaza_voz_sr)
            if target_sr > 0 and target_sr != sr:
                frames, _ = audioop.ratecv(frames, 2, ch, sr, target_sr, None)
                sr = target_sr
            out = io.BytesIO()
            with wave.open(out, "wb") as w2:
                w2.setnchannels(ch)
                w2.setsampwidth(2)
                w2.setframerate(sr)
                w2.writeframes(frames)
            return out.getvalue()
        except Exception:
            return wav

    def _plaza_voz_send_once(self) -> bool:
        if not self.plaza_url:
            return False
        # Empujar solo cuando el organismo realmente vocaliza (ahorra enlace en celular).
        if self.plaza_voz_solo_vocalizando and not self._voz_activa():
            return True  # nada que enviar; no es un fallo
        wav = self._fetch_voz_local()
        if not wav:
            return False
        wav = self._aligerar_wav(wav)
        url = f"{self.plaza_url.rstrip('/')}/plaza/voz?oid={quote(self.organism_id, safe='')}"
        req = urllib.request.Request(
            url, data=wav, method="POST",
            headers={
                "Content-Type": "audio/wav",
                "User-Agent": "Mozilla/5.0 (compatible; ANIMA-Presencia/1.0; +https://cosmosemiotica.cl)",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=8.0) as r:
                r.read()
            self._plaza_voz_last_ok = time.time()
            self._plaza_voz_last_err = ""
            return True
        except Exception as e:
            self._plaza_voz_last_err = f"{type(e).__name__}: {e}"
            return False

    def _plaza_voz_loop(self) -> None:
        # Esperar a que el heartbeat registre al organismo en la plaza antes de empujar voz.
        if self._stop.wait(4.0):
            return
        base = max(0.4, float(self.plaza_voz_interval_s))
        espera = base
        while not self._stop.is_set():
            ok = True
            try:
                ok = self._plaza_voz_send_once()
            except Exception as e:
                ok = False
                print(f"[presencia] plaza voz loop: {e}", flush=True)
            # Backoff exponencial si el enlace falla: nunca ahogar el heartbeat en celular.
            espera = base if ok else min(espera * 2, 15.0)
            if self._stop.wait(espera):
                return

    def _plaza_roster_fetch(self) -> list[dict[str, Any]]:
        if not self.plaza_url:
            return []
        url = self.plaza_url.rstrip("/") + "/roster"
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (compatible; ANIMA-Presencia/1.0; +https://cosmosemiotica.cl)",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=8.0) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            self._plaza_roster_last_err = f"{type(e).__name__}: {e}"
            return []
        if isinstance(data, dict):
            return [e for e in (data.get("roster") or []) if isinstance(e, dict)]
        return []

    def _registrar_vecino_plaza(self, entry: dict[str, Any]) -> None:
        """Registra un organismo remoto del roster del Observatorio como vecino de relay."""
        oid = (entry.get("organism_id") or "").strip()  # OID público (opaco) del Observatorio
        if not oid:
            return
        # No registrarme a mí mismo (identificador público aprendido del heartbeat).
        if self._mi_oid_publico and oid == self._mi_oid_publico:
            return
        iid = (entry.get("installation_id") or "").strip()
        if iid and iid == self.installation_id:
            return
        name = entry.get("name") or oid
        now = time.time()
        with self._lock:
            prev = self._vecinos.get(oid)
            # No pisar un vecino LAN real (mDNS/UDP) vigente con el relay remoto.
            if prev and prev.source in ("mdns", "udp") and prev.estado_presencia(self.ttl_s) != "ausente":
                return
            self._vecinos[oid] = Vecino(
                organism_id=oid,
                installation_id=iid,
                name=name,
                base_url=self.plaza_url.rstrip("/"),
                host="",
                port=0,
                state=entry.get("estado_presencia") or "activo",
                visibility="plaza_publico",
                capabilities=list(entry.get("capabilities") or ["voz", "comunicacion"]),
                first_seen=prev.first_seen if prev else now,
                last_seen=now,
                confianza=0.6,
                source="plaza",
            )
            if iid and iid not in self._known_installations:
                self._known_installations.add(iid)
                self._novedad_until = now + 10.0
            self._sin_vecinos_desde = None

    def _plaza_roster_loop(self) -> None:
        if self._stop.wait(6.0):
            return
        while not self._stop.is_set():
            try:
                for entry in self._plaza_roster_fetch():
                    if (entry.get("source") or "") != "plaza":
                        continue  # solo remotos de plaza; los LAN llegan por mDNS/UDP
                    self._registrar_vecino_plaza(entry)
            except Exception as e:
                print(f"[presencia] plaza roster loop: {e}", flush=True)
            if self._stop.wait(max(2.0, float(self.plaza_roster_interval_s))):
                return

    def arrancar(self) -> None:
        if not self._activo and not self.plaza_url:
            return
        self._stop.clear()
        if self._local_discovery:
            try:
                self._mdns_register()
            except Exception as e:
                print(f"[presencia] mDNS registro: {e}", flush=True)
            try:
                self._mdns_browse()
            except Exception as e:
                print(f"[presencia] mDNS browse: {e}", flush=True)
            t_udp = threading.Thread(target=self._udp_loop, daemon=True, name="anima-presencia-udp")
            t_bcn = threading.Thread(target=self._udp_beacon, daemon=True, name="anima-presencia-beacon")
            t_ref = threading.Thread(target=self._refresh_loop, daemon=True, name="anima-presencia-refresh")
            t_udp.start(); t_bcn.start(); t_ref.start()
            self._threads.extend([t_udp, t_bcn, t_ref])
            ips = ", ".join(_local_ipv4s())
            print(
                f"[presencia] ON · modo={self.mode} · vis={self.visibility} · "
                f"ttl={self.ttl_s:.0f}s · UDP :{self.UDP_PORT} · nics={ips}",
                flush=True,
            )
        else:
            print(f"[presencia] LAN off · modo={self.mode}", flush=True)
        if self.plaza_url:
            t_pl = threading.Thread(target=self._plaza_loop, daemon=True, name="anima-presencia-plaza")
            t_pl.start()
            self._threads.append(t_pl)
            t_voz = threading.Thread(target=self._plaza_voz_loop, daemon=True, name="anima-presencia-plaza-voz")
            t_voz.start()
            self._threads.append(t_voz)
            t_ros = threading.Thread(target=self._plaza_roster_loop, daemon=True, name="anima-presencia-plaza-roster")
            t_ros.start()
            self._threads.append(t_ros)
            print(
                f"[presencia] plaza → {self.plaza_url}/plaza/heartbeat cada {self.plaza_interval_s:.0f}s "
                f"· voz cada {self.plaza_voz_interval_s:.1f}s · roster cada {self.plaza_roster_interval_s:.0f}s",
                flush=True,
            )

    def cerrar(self) -> None:
        self._stop.set()
        if self._udp_sock:
            try:
                self._udp_sock.close()
            except Exception:
                pass
        if self._zeroconf and self._service_info:
            try:
                self._zeroconf.unregister_service(self._service_info)
                self._zeroconf.close()
            except Exception:
                pass