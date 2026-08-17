# -*- coding: utf-8 -*-
"""
Bootstrap de escritorio: arranca VST_CelulaMadre_WebLive_A con ganchos de
actualización (/api/desktop/*) y banner JS, sin alterar el motor del organismo.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def _install_root() -> Path:
    # .../app/celula_madre/web/bootstrap_desktop.py → root = parents[3]
    # web → celula_madre → app → root
    here = Path(__file__).resolve()
    # Prefer explicit env from launcher
    env_root = os.environ.get("ANIMA_INSTALL_ROOT")
    if env_root:
        return Path(env_root)
    # bootstrap lives in app/celula_madre/web/
    return here.parents[3]


def _patch_web_live() -> None:
    import VST_CelulaMadre_WebLive_A as live  # noqa: WPS433 — path ya en sys.path

    root = _install_root()
    updater_py = root / "updater" / "anima_updater.py"

    def _run_updater(cmd: str) -> dict:
        if not updater_py.is_file():
            return {"ok": False, "message": "Updater no instalado."}
        # Importar por path sin contaminar global
        import importlib.util

        spec = importlib.util.spec_from_file_location("anima_updater", updater_py)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        if cmd == "check":
            return mod.check_update(root)
        if cmd == "apply":
            return mod.apply_update(root, yes=True)
        if cmd == "status":
            return {
                "ok": True,
                "version": mod.load_version(root),
                "last": mod.load_status(root),
            }
        return {"ok": False, "message": f"comando desconocido: {cmd}"}

    docs_dir = root / "docs"

    def _peer_entry() -> dict | None:
        """Interlocutor realmente elegido como fuente; presencia sola no basta."""
        run = getattr(live, "RUN", None)
        cfg = getattr(run, "cfg", None) if run is not None else None
        if not isinstance(cfg, dict):
            return None
        selected_urls: list[str] = []
        for key in ("left_src", "right_src"):
            src = cfg.get(key) or {}
            if src.get("tipo") == "comunicacion" and src.get("url"):
                modo = str(src.get("modo") or "").upper()
                if modo not in ("NULL_STATE", "SHUFFLED_STATE"):
                    selected_urls.append(str(src["url"]))
            elif src.get("tipo") == "otros_organismos":
                urls = [str(u) for u in (src.get("urls") or []) if u]
                if len(urls) == 1:
                    selected_urls.extend(urls)
        if not selected_urls:
            return None
        roster = live._ecosistema_roster_cached(0.5)
        for raw in selected_urls:
            target = raw.split("?", 1)[0].rstrip("/").lower()
            for entry in roster:
                bases = {
                    str(entry.get("base_url") or "").split("?", 1)[0].rstrip("/").lower(),
                    str(entry.get("voz_url") or "").split("?", 1)[0].rstrip("/").lower(),
                }
                if target and any(base and (target.startswith(base) or base.startswith(target)) for base in bases):
                    return entry
        return None

    def _peer_json(path: str) -> dict:
        entry = _peer_entry()
        vacio = ({"corriendo": False, "cols": [], "fila": None, "n": 0}
                 if path == "/ultima_fila"
                 else {"vivo": False, "organismo": "Sin interlocutor", "organismo_id": ""})
        if not entry or not entry.get("base_url"):
            return vacio
        base = entry["base_url"].rstrip("/")
        oid = entry.get("organism_id", "") or ""
        es_relay = bool(oid) and "voz?oid=" in (entry.get("voz_url") or "")
        def _fetch(url):
            # UA de navegador: Cloudflare responde 403 a Python-urllib en el Observatorio público.
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; ANIMA-Desktop/1.0; +https://cosmosemiotica.cl)",
                "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=2.5) as r:
                return json.loads(r.read().decode("utf-8"))
        try:
            if es_relay:
                # Par por RELAY del Observatorio: su estado se lee por proxy con ?oid=
                # (base_url = Observatorio, no alcanzable como par LAN directo).
                from urllib.parse import quote as _q
                d = _fetch(f"{base}/comunicacion/estado?oid={_q(oid, safe='')}")
                fila = d.get("fila") if isinstance(d, dict) else None
                if path == "/ultima_fila":
                    return {"corriendo": bool(fila and fila.get("vivo")), "cols": [],
                            "fila": fila, "n": 1 if fila else 0}
                return fila or vacio   # /estado: estado plano del par (lo que espera vst_conversacion)
            return _fetch(base + path)
        except Exception:
            return vacio

    def _mount_original_observatory():
        """Monta el código original como rutas del MISMO servidor del organismo."""
        cm_dir = Path(live.__file__).resolve().parent.parent
        conv_dir = cm_dir / "conversacion"
        state_dir = Path(os.environ.get("ANIMA_STATE_DIR", str(Path.home() / ".anima")))
        os.environ["ANIMA_A_URL"] = f"http://127.0.0.1:{live.PUERTO}"
        os.environ["ANIMA_B_URL"] = f"http://127.0.0.1:{live.PUERTO}/ecosistema/peer"
        os.environ.setdefault("ANIMA_CONV_DIR", str(state_dir / "conversacion"))
        os.environ.setdefault("VST_HISTORY_DIR", str(state_dir / "history"))
        conv_path = str(conv_dir)
        if conv_path not in sys.path:
            sys.path.insert(0, conv_path)
        try:
            import vst_conversacion as conv  # noqa: WPS433 — fuente original del Observatorio

            # Adaptación de membrana: oculta su navegación duplicada dentro del iframe,
            # acepta la vista pedida y cambios por postMessage. El HTML/CSS/Three/circuito
            # siguen siendo literalmente los del archivo original.
            conv.HTML = conv.HTML.replace(
                '<div style="display:flex;align-items:center;gap:10px;padding:6px 14px',
                '<div id=obsTop style="display:flex;align-items:center;gap:10px;padding:6px 14px',
                1,
            )
            conv.HTML = conv.HTML.replace(
                "vista('vivo');",
                (
                    "const _embedQ=new URLSearchParams(location.hash.slice(1));\n"
                    "if(_embedQ.get('embed')==='1'){const _h=document.querySelector('body>h1'),_t=document.getElementById('obsTop');if(_h)_h.style.display='none';if(_t)_t.style.display='none';}\n"
                    "const _embedVista=_embedQ.get('vista')||'vivo';vista(_embedVista);\n"
                    "window.addEventListener('message',e=>{const v=e&&e.data&&e.data.animaVista;if(['vivo','hist','circ'].includes(v))vista(v);});"
                ),
                1,
            )
            threading.Thread(target=conv._poller, daemon=True, name="observatorio-poller").start()
            threading.Thread(target=conv._stats_precompute, daemon=True, name="observatorio-stats").start()
            return conv
        except Exception as exc:
            print(f"[desktop] Observatorio original no disponible: {exc}", file=sys.stderr)
            return None

    original_observatory = _mount_original_observatory()

    def _list_manuals() -> list[dict]:
        items = []
        catalog = [
            {
                "id": "pagina_organismo",
                "file": "MANUAL_pagina_organismo_ANIMA.pdf",
                "title": "Manual — Página del organismo",
                "desc": "Tu animalito en el PC: cara, cajas, voz y lectura básica.",
            },
            {
                "id": "observatorio",
                "file": "MANUAL_observatorio_sociedad_ANIMA.pdf",
                "title": "Manual — Observatorio / Sociedad",
                "desc": "Varios organismos del campo a la vez (vista pública).",
            },
        ]
        for c in catalog:
            p = docs_dir / c["file"]
            if p.is_file():
                items.append({**c, "url": f"/docs/{c['file']}", "bytes": p.stat().st_size})
        return items

    # --- HTML: scripts de escritorio (updates + ayuda) ---
    if hasattr(live, "_render_html"):
        _orig_render = live._render_html

        def _render_html_patched():
            html = _orig_render()
            inject = ""
            if "desktop_update.js" not in html:
                inject += '<script src="/Cajas/desktop_update.js"></script>'
            if "desktop_help.js" not in html:
                inject += '<script src="/Cajas/desktop_help.js"></script>'
            if "desktop_section_help.js" not in html:
                inject += '<script src="/Cajas/desktop_section_help.js"></script>'
            if "desktop_portal.js" not in html:
                inject += '<script src="/Cajas/desktop_portal.js"></script>'
            if inject:
                html = html.replace("</body>", inject + "</body>")
            return html

        live._render_html = _render_html_patched

    # --- HTTP API + manuales PDF ---
    Handler = live.Handler
    _orig_get = Handler.do_GET
    _orig_post = getattr(Handler, "do_POST", None)

    def do_GET(self):  # type: ignore[no-untyped-def]
        u = urlparse(self.path)
        path = u.path
        if path == "/observatorio-original":
            if original_observatory is None:
                self._send(503, "Observatorio original no disponible", "text/plain")
            else:
                self._send(200, original_observatory.HTML, "text/html")
            return
        if path == "/api/desktop/observatorio/datos":
            # Proxy del mismo origen: el navegador local no debe depender de
            # CORS para comprobar el Observatorio público.
            try:
                req = urllib.request.Request(
                    "https://observatorio.cosmosemiotica.cl/datos",
                    headers={"User-Agent": "ANIMA-Desktop/0.3.9"},
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=8.0) as response:
                    raw = response.read()
                # Python acepta NaN por compatibilidad, JavaScript no. Normalizar
                # también un servidor antiguo mientras termina de reiniciarse.
                data = json.loads(
                    raw.decode("utf-8"),
                    parse_constant=lambda _token: None,
                )
                clean = json.dumps(
                    data,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                self._send(200, clean, "application/json")
            except Exception as exc:
                self._send(
                    502,
                    json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
                    "application/json",
                )
            return
        if original_observatory is not None and path in (
            "/datos", "/dias", "/turnos", "/stats", "/historial", "/voz/A", "/voz/B"
        ):
            original_observatory.Handler.do_GET(self)
            return
        if path == "/ecosistema/peer/estado":
            try:
                self._send(200, json.dumps(_peer_json("/estado"), ensure_ascii=False), "application/json")
            except Exception:
                self._send(200, json.dumps({"vivo": False, "organismo": "Sin interlocutor"}, ensure_ascii=False), "application/json")
            return
        if path == "/ecosistema/peer/ultima_fila":
            try:
                self._send(200, json.dumps(_peer_json("/ultima_fila"), ensure_ascii=False), "application/json")
            except Exception:
                self._send(200, json.dumps({"corriendo": False, "cols": [], "fila": None, "n": 0}), "application/json")
            return
        if path == "/ecosistema/peer/voz":
            entry = _peer_entry()
            if not entry or not entry.get("voz_url"):
                self._send(503, b"", "audio/wav")
                return
            try:
                query = parse_qs(u.query)
                seg = (query.get("seg") or ["1.0"])[0]
                voz_url = str(entry["voz_url"])
                sep = "&" if "?" in voz_url else "?"
                full = voz_url if "seg=" in voz_url else f"{voz_url}{sep}seg={seg}"
                with urllib.request.urlopen(full, timeout=5.0) as response:
                    self._send(200, response.read(), "audio/wav")
            except Exception:
                self._send(503, b"", "audio/wav")
            return
        if path in ("/api/desktop/version", "/api/desktop/update/status"):
            try:
                data = _run_updater("status" if path.endswith("status") else "status")
                if path.endswith("version"):
                    data = {"ok": True, **(data.get("version") or {})}
                self._send(200, json.dumps(data, ensure_ascii=False), "application/json")
            except Exception as e:
                self._send(500, json.dumps({"ok": False, "message": str(e)}), "application/json")
            return
        if path == "/api/desktop/update/check":
            try:
                data = _run_updater("check")
                self._send(200, json.dumps(data, ensure_ascii=False), "application/json")
            except Exception as e:
                self._send(500, json.dumps({"ok": False, "message": str(e)}), "application/json")
            return
        if path == "/api/desktop/manuals":
            self._send(
                200,
                json.dumps(
                    {
                        "ok": True,
                        "manuals": _list_manuals(),
                        "links": [
                            {
                                "title": "Observatorio público",
                                "url": "https://observatorio.cosmosemiotica.cl/",
                            },
                            {"title": "Cosmolab", "url": "https://cosmosemiotica.cl"},
                        ],
                    },
                    ensure_ascii=False,
                ),
                "application/json",
            )
            return
        if path.startswith("/docs/"):
            name = path[len("/docs/") :].lstrip("/").replace("\\", "/")
            if ".." in name or name.startswith("/"):
                self._send(400, "ruta inválida", "text/plain")
                return
            fp = (docs_dir / name).resolve()
            try:
                fp.relative_to(docs_dir.resolve())
            except ValueError:
                self._send(403, "prohibido", "text/plain")
                return
            if not fp.is_file():
                self._send(404, "manual no encontrado", "text/plain")
                return
            ext = fp.suffix.lower()
            ctype = {
                ".pdf": "application/pdf",
                ".txt": "text/plain; charset=utf-8",
                ".md": "text/markdown; charset=utf-8",
            }.get(ext, "application/octet-stream")
            try:
                data = fp.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Disposition", f'inline; filename="{fp.name}"')
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self._send(500, str(e), "text/plain")
            return
        return _orig_get(self)

    def do_POST(self):  # type: ignore[no-untyped-def]
        u = urlparse(self.path)
        path = u.path
        if original_observatory is not None and path == "/cortar":
            original_observatory.Handler.do_POST(self)
            return
        if path == "/ecosistema/peer/mute":
            entry = _peer_entry()
            if not entry or not entry.get("base_url"):
                self._send(200, json.dumps({"ok": False, "error": "sin interlocutor"}), "application/json")
                return
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(length) if length else b"{}"
                request = urllib.request.Request(
                    entry["base_url"].rstrip("/") + "/mute",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=4.0) as response:
                    payload = response.read()
                self._send(200, payload, "application/json")
            except Exception as exc:
                self._send(200, json.dumps({"ok": False, "error": str(exc)}), "application/json")
            return
        if path == "/api/desktop/update/apply":
            try:
                # Aplicar en hilo para no bloquear el HTTP demasiado tiempo en UI
                result_box: dict = {}

                def work() -> None:
                    result_box["r"] = _run_updater("apply")

                t = threading.Thread(target=work, daemon=True)
                t.start()
                t.join(timeout=300)
                data = result_box.get("r") or {
                    "ok": False,
                    "message": "Tiempo de espera agotado aplicando la actualización.",
                }
                self._send(200, json.dumps(data, ensure_ascii=False), "application/json")
            except Exception as e:
                self._send(500, json.dumps({"ok": False, "message": str(e)}), "application/json")
            return
        if _orig_post is not None:
            return _orig_post(self)
        self._send(404, "not found", "text/plain")

    Handler.do_GET = do_GET
    Handler.do_POST = do_POST

    live.main()


def _load_organismo_env(root: Path) -> None:
    """Carga app/config/organismo.env.

    Por defecto no pisa variables ya definidas por el launcher.
    Excepción: claves de presencia/plaza del instalable (colaboradores deben
    poder empujar heartbeat al Observatorio aunque el launcher traiga local).
    """
    path = root / "app" / "config" / "organismo.env"
    if not path.is_file():
        return
    # Claves que el producto desktop debe tomar del config del instalable.
    override_keys = {
        "ANIMA_PRESENCE_MODE",
        "ANIMA_VISIBILITY",
        "ANIMA_PLAZA",
        "ANIMA_PLAZA_URL",
        "ANIMA_PLAZA_INTERVAL_S",
        "ANIMA_PLAZA_HEARTBEAT_TTL_S",
        "ANIMA_PLAZA_HEARTBEAT_TTL",
    }
    try:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if not key:
                continue
            if key in override_keys or key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass


def _port_free(port: int) -> bool:
    """True solo si el puerto está libre en 0.0.0.0 y 127.0.0.1 (Windows dual-stack)."""
    import socket

    # Si algo ya acepta conexiones en localhost, no reutilizar (p. ej. LGHub en :9100).
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return False
    except OSError:
        pass

    for addr in ("0.0.0.0", "127.0.0.1"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((addr, port))
        except OSError:
            return False
    return True


def _resolve_desktop_port(root: Path) -> int:
    """Puerto del organismo instalable: banda 9100+ (fuera del lab Mac 7788–7820).

    - Preferencia: ANIMA_PORT_BASE / VST_PUERTO si ya es >= 9100
    - Si el launcher aún manda 7788 (lab), se ignora y se usa 9100+
    - Si está ocupado, busca el siguiente libre (salta 9101 observatorio, etc.)
    - Persiste el puerto en ~/.anima/desktop_port.json
    """
    raw_env = os.environ.get("VST_PUERTO") or os.environ.get("ANIMA_PORT_BASE") or "9100"
    try:
        env_port = int(raw_env)
    except ValueError:
        env_port = 9100
    # Forzar banda pública: el lab Mac se queda en 7788–7820.
    base = env_port if env_port >= 9100 else int(os.environ.get("ANIMA_PORT_BASE") or "9100")
    if base < 9100:
        base = 9100
    span = int(os.environ.get("ANIMA_PORT_SPAN") or "100")
    # 9101 = observatorio Abraxas; 9100 a veces lo ocupa LGHub en 127.0.0.1
    reserved = {9101}
    for tok in (os.environ.get("ANIMA_RESERVE_PORTS") or "").split(","):
        tok = tok.strip()
        if tok.isdigit():
            reserved.add(int(tok))

    state_dir = Path(os.environ.get("ANIMA_STATE_DIR") or (Path.home() / ".anima"))
    port_file = state_dir / "desktop_port.json"
    sticky: int | None = None
    try:
        if port_file.is_file():
            data = json.loads(port_file.read_text(encoding="utf-8"))
            sticky = int(data.get("port") or 0) or None
    except Exception:
        sticky = None

    candidates: list[int] = []
    if sticky and sticky >= 9100:
        candidates.append(sticky)
    if base not in candidates:
        candidates.append(base)
    for p in range(base, base + max(1, span)):
        if p not in candidates:
            candidates.append(p)

    chosen = base
    for p in candidates:
        if p in reserved or p < 9100:
            continue
        if _port_free(p):
            chosen = p
            break

    os.environ["VST_PUERTO"] = str(chosen)
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        port_file.write_text(
            json.dumps(
                {"port": chosen, "base": base, "product": "anima-desktop", "band": "9100+"},
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass
    return chosen


def _try_auto_update(root: Path) -> None:
    """Actualización automática al arrancar (colaboradores). Antes de cargar el motor.

    Si ANIMA_AUTO_UPDATE=0, solo registra el check. Fallos de red no impiden el arranque.
    """
    if os.environ.get("ANIMA_AUTO_UPDATE", "1").strip().lower() in ("0", "false", "no", "off"):
        print("[anima-desktop] auto-update desactivado (ANIMA_AUTO_UPDATE=0)", flush=True)
        return
    updater_py = root / "updater" / "anima_updater.py"
    if not updater_py.is_file():
        return
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("anima_updater_boot", updater_py)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        result = mod.auto_update(root)
        if result.get("update_available") is False and result.get("ok"):
            print(f"[anima-desktop] updates: {result.get('message', 'ok')}", flush=True)
        elif result.get("ok") and result.get("new_version"):
            print(
                f"[anima-desktop] actualizado a {result.get('new_version')} — "
                f"{result.get('message', '')}",
                flush=True,
            )
        elif not result.get("ok"):
            print(f"[anima-desktop] updates: {result.get('message', 'sin canal')}", flush=True)
    except Exception as exc:
        print(f"[anima-desktop] updates omitido: {type(exc).__name__}: {exc}", flush=True)


def _ensure_desktop_data_dirs() -> None:
    """Rutas escribibles del instalable ANTES de importar el motor.

    El lab Docker usa VST_HISTORY_DIR=/history. En desktop eso cae en la raíz
    del FS (solo lectura) y tumba el servidor → 'Failed to fetch' en el browser.
    """
    home = Path.home()
    state = Path(os.environ.get("ANIMA_STATE_DIR") or (home / ".anima"))
    estado = Path(
        os.environ.get("ANIMA_ESTADO_DIR")
        or (home / "Library" / "Application Support" / "ANIMA" / "data")
        if sys.platform == "darwin"
        else (Path(os.environ.get("LOCALAPPDATA", str(home))) / "ANIMA" / "data")
    )
    # Corregir si quedó el valor de Docker
    hist_env = (os.environ.get("VST_HISTORY_DIR") or "").strip()
    if not hist_env or hist_env in ("/history", "/data", "history"):
        hist = state / "history"
    else:
        hist = Path(hist_env)
    conv_env = (os.environ.get("ANIMA_CONV_DIR") or "").strip()
    if not conv_env or conv_env.startswith("/history"):
        conv = state / "conversacion"
    else:
        conv = Path(conv_env)
    for d in (state, hist, conv):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    os.environ["ANIMA_STATE_DIR"] = str(state)
    os.environ["VST_HISTORY_DIR"] = str(hist)
    os.environ["ANIMA_CONV_DIR"] = str(conv)
    os.environ.setdefault("VST_HISTORY_ENABLE", "true")
    if "ANIMA_ESTADO_DIR" not in os.environ:
        os.environ["ANIMA_ESTADO_DIR"] = str(estado)


def main() -> None:
    # Evitar UnicodeEncodeError en Windows (consola/log cp1252 vs flechas Unicode del WebLive)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    root = _install_root()
    _load_organismo_env(root)
    # CRITICO: paths de datos antes de importar WebLive/historia
    _ensure_desktop_data_dirs()
    # Colaboradores: actualizar app/ antes de importar el motor (archivos no bloqueados aún)
    _try_auto_update(root)
    # Escritorio público: puertos altos (9100+), no 7788 del lab Mac
    port = _resolve_desktop_port(root)
    os.environ.setdefault("ANIMA_BIND", "0.0.0.0")
    print(f"[anima-desktop] puerto organismo = {port} (banda 9100+; lab Mac usa 7788–7820)")
    print(f"[anima-desktop] history = {os.environ.get('VST_HISTORY_DIR')}")

    # Asegurar paths de la célula madre
    web_dir = Path(__file__).resolve().parent
    cm = web_dir.parent
    for d in (cm, cm / "organelos", cm / "audio", cm / "web", cm / "genoma", cm / "campo", cm / "diada"):
        s = str(d)
        if d.is_dir() and s not in sys.path:
            sys.path.insert(0, s)
    os.chdir(str(cm))
    _patch_web_live()


if __name__ == "__main__":
    main()
