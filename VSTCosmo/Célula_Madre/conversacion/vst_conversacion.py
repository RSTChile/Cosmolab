#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_CONVERSACION — OBSERVATORIO PERMANENTE DE LA CONVERSACIÓN DE LA DÍADA
================================================================================
QUIÉN SOY
  Los organismos viven 24/7, no sólo durante los experimentos. Yo capturo su
  CONVERSACIÓN de forma PERMANENTE: sondeo a A y B, registro qué 'pito' (voz R2-D2)
  usa cada uno en secuencia, lo guardo a disco (un volumen que sobrevive reinicios),
  y lo MUESTRO en vivo — ambos organismos "mirándose" y qué se están diciendo.

  NO toco la fisiología (sólo leo /estado por HTTP, como la membrana MCP). Soy un
  OBSERVADOR: si caigo, los organismos siguen conversando igual.

QUÉ HAGO
  · Sondeo A (ANIMA_A_URL) y B (ANIMA_B_URL) cada POLL_S y leo su voz emitida + afecto
    + orientación de la cabeza.
  · Cuando un organismo CAMBIA de pito = un TURNO: lo escribo en el log permanente
    (JSONL en ANIMA_CONV_DIR) con timestamp.
  · Sirvo un tablero: las dos cabezas girando hacia el otro (espacialidad), el pito
    actual de cada uno, la transcripción en vivo y un histograma de voces.

CÓMO CORRER
    venv/bin/python Célula_Madre/conversacion/vst_conversacion.py   → http://localhost:9100
  Config por entorno: ANIMA_A_URL, ANIMA_B_URL, ANIMA_CONV_DIR, ANIMA_CONV_PORT, POLL_S.
================================================================================
"""
from __future__ import annotations
import os, sys, json, time, threading, urllib.request, copy as _copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Historiador de la DÍADA (biografía de la conversación A↔B en disco externo). Infra, no cerebro.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "organelos"))
try:
    from vst_historia import Historiador
    _HIST_DIADA = Historiador("diada")
except Exception as _e:
    _HIST_DIADA = None
    sys.stderr.write(f"[conversacion] historia díada OFF: {_e}\n")
_CTX_PREV = {"A": {}, "B": {}}   # estado del receptor en el turno anterior (para delta)

# Carpeta del registro PERMANENTE de la conversación (para el navegador histórico).
_HIST_DIR = os.environ.get("VST_HISTORY_DIR", "/history")
_HIST_COMM_DIR = os.path.join(_HIST_DIR, "diada", "comunicacion")


def _dias_disponibles():
    """Fechas con conversación registrada (de los archivos comunicacion_YYYY-MM-DD.jsonl)."""
    out = []
    try:
        for n in sorted(os.listdir(_HIST_COMM_DIR)):
            if n.startswith("comunicacion_") and n.endswith(".jsonl"):
                out.append(n[len("comunicacion_"):-len(".jsonl")])
    except Exception:
        pass
    return out


def _leer_turnos(dia, voz=None, limite=500):
    """Lee los turnos registrados de un día (o el más reciente), opcionalmente filtrados por pito."""
    dias = _dias_disponibles()
    if not dias:
        return []
    if not dia or dia not in dias:
        dia = dias[-1]
    path = os.path.join(_HIST_COMM_DIR, f"comunicacion_{dia}.jsonl")
    turnos = []
    try:
        with open(path, encoding="utf-8") as f:
            for ln in f:
                try:
                    t = json.loads(ln)
                except Exception:
                    continue
                if voz and t.get("prototipo_codebook") != voz:
                    continue
                turnos.append(t)
    except Exception:
        pass
    return turnos[-int(limite):]


# ---- /stats: caché + single-flight + actualización incremental ----
# Antes: cada GET /stats releía TODOS los JSONL (~2M turnos, ~1 GB) y, con varios
# clientes, apilaba hilos hasta saturar CPU/RAM. Ahora: un solo scan (streaming),
# caché en memoria, contadores incrementales al registrar turnos, y single-flight
# para que N requests concurrentes esperen el mismo cálculo.
_STATS_LOCK = threading.Lock()
_STATS_COND = threading.Condition(_STATS_LOCK)
_STATS_CACHE = None          # {"dias", "total_turnos", "hist"} o None
_STATS_BUILDING = False
_STATS_BUILT_AT = 0.0
_STATS_FULL_REBUILD_S = float(os.environ.get("ANIMA_STATS_REBUILD_S", "3600"))  # self-heal opcional


def _compute_stats_historial():
    """Scan streaming de los JSONL de la díada (sin materializar la lista completa)."""
    hist = {}
    total = 0
    dias = _dias_disponibles()
    for dia in dias:
        path = os.path.join(_HIST_COMM_DIR, f"comunicacion_{dia}.jsonl")
        try:
            with open(path, encoding="utf-8") as f:
                for ln in f:
                    try:
                        t = json.loads(ln)
                    except Exception:
                        continue
                    em = t.get("emisor", "?")
                    voz = t.get("prototipo_codebook", "-")
                    bucket = hist.setdefault(em, {})
                    bucket[voz] = bucket.get(voz, 0) + 1
                    total += 1
        except Exception:
            continue
    return {"dias": list(dias), "total_turnos": total, "hist": hist}


def _stats_snapshot():
    """Copia del caché bajo lock (el caller ya no debe mutarla)."""
    return _copy.deepcopy(_STATS_CACHE) if _STATS_CACHE is not None else None


def _stats_note_turno(emisor, voz):
    """Actualización O(1) del histograma cuando nace un turno nuevo (evita re-scan)."""
    if not emisor:
        emisor = "?"
    if not voz:
        voz = "-"
    today = time.strftime("%Y-%m-%d")
    with _STATS_LOCK:
        if _STATS_CACHE is None:
            return  # aún no hay base; el primer /stats o el precompute la arman
        hist = _STATS_CACHE["hist"]
        bucket = hist.setdefault(emisor, {})
        bucket[voz] = bucket.get(voz, 0) + 1
        _STATS_CACHE["total_turnos"] = int(_STATS_CACHE.get("total_turnos", 0)) + 1
        dias = _STATS_CACHE.get("dias") or []
        if today not in dias:
            _STATS_CACHE["dias"] = list(dias) + [today]
        global _STATS_BUILT_AT
        _STATS_BUILT_AT = time.time()


def _stats_historial(force=False):
    """Estadística acumulada. Sirve de caché; un solo builder a la vez (single-flight)."""
    global _STATS_CACHE, _STATS_BUILDING, _STATS_BUILT_AT

    with _STATS_COND:
        now = time.time()
        stale = (
            _STATS_CACHE is not None
            and _STATS_FULL_REBUILD_S > 0
            and (now - _STATS_BUILT_AT) > _STATS_FULL_REBUILD_S
        )
        if _STATS_CACHE is not None and not force and not stale:
            return _stats_snapshot()

        # Esperar a un builder en curso (no lanzar otro scan de 2M líneas).
        while _STATS_BUILDING:
            _STATS_COND.wait(timeout=120.0)
            if _STATS_CACHE is not None and not force:
                # Tras el wait: si ya hay caché, servirla (aunque force, el builder la dejó fresca).
                return _stats_snapshot()
            if not _STATS_BUILDING:
                break

        if _STATS_CACHE is not None and not force and not (
            _STATS_FULL_REBUILD_S > 0 and (time.time() - _STATS_BUILT_AT) > _STATS_FULL_REBUILD_S
        ):
            return _stats_snapshot()

        _STATS_BUILDING = True

    try:
        t0 = time.time()
        result = _compute_stats_historial()
        dt = time.time() - t0
        sys.stderr.write(
            f"[conversacion] /stats rebuild: {result.get('total_turnos', 0)} turnos · "
            f"{len(result.get('dias') or [])} días · {dt:.1f}s\n"
        )
        with _STATS_COND:
            _STATS_CACHE = result
            _STATS_BUILT_AT = time.time()
            _STATS_BUILDING = False
            _STATS_COND.notify_all()
        return _stats_snapshot()
    except Exception as e:
        with _STATS_COND:
            _STATS_BUILDING = False
            _STATS_COND.notify_all()
        sys.stderr.write(f"[conversacion] /stats rebuild ERROR: {e}\n")
        # Si había caché vieja, servirla; si no, respuesta vacía (no colgar al cliente).
        snap = _stats_snapshot()
        if snap is not None:
            return snap
        return {"dias": [], "total_turnos": 0, "hist": {}, "error": str(e)}


def _stats_precompute():
    """Calienta la caché al arrancar (hilo daemon) para que el 1er /stats no pague el scan."""
    try:
        _stats_historial(force=True)
    except Exception as e:
        sys.stderr.write(f"[conversacion] precompute /stats falló: {e}\n")

A_URL = os.environ.get("ANIMA_A_URL", "http://127.0.0.1:7788")
B_URL = os.environ.get("ANIMA_B_URL", "http://127.0.0.1:7799")
C_URL = os.environ.get("ANIMA_C_URL", "http://127.0.0.1:7810")
D_URL = os.environ.get("ANIMA_D_URL", "http://127.0.0.1:7820")
E_URL = os.environ.get("ANIMA_E_URL", "")   # E vive en la Pi (por IP); vacío = no aparece
# ROSTER configurable: A y B siempre; C, D y E si sus URLs están definidas (se adapta a N organismos).
ROSTER = [("A", A_URL), ("B", B_URL)]
for _L, _u in (("C", C_URL), ("D", D_URL), ("E", E_URL)):
    if _u and _u.strip():
        ROSTER.append((_L, _u.strip()))
CONV_DIR = os.environ.get("ANIMA_CONV_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "registro"))
PORT = int(os.environ.get("ANIMA_CONV_PORT", "9100"))
POLL_S = float(os.environ.get("POLL_S", "0.4"))
MAX_TRANSCRIPT = 400          # turnos recientes en memoria para el tablero

os.makedirs(CONV_DIR, exist_ok=True)
LOG_PATH = os.path.join(CONV_DIR, "conversacion.jsonl")

_EST = {L: {} for L, _ in ROSTER}         # último estado de cada organismo (roster, no solo A/B)
_TRANSCRIPT = []                          # turnos recientes (dicts) para el tablero
_HIST = {L: {} for L, _ in ROSTER}        # histograma de voces por organismo
_ULTIMA = {L: None for L, _ in ROSTER}    # última voz vista (para detectar cambios = turnos)
_LEXICO = []                              # feed de eventos de léxico: creación / emulación
_LEXPREV = {L: None for L, _ in ROSTER}   # (voz_creadas, voz_aprendidas) previas, por organismo
_LOCK = threading.Lock()


def _get_json(url, path, timeout=3.0):
    with urllib.request.urlopen(url + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _registrar_turno(lado, est):
    """Escribe un turno (cambio de pito) al log permanente y al transcript en memoria."""
    voz = est.get("voz_emitida", "-")
    turno = {"ts": round(time.time(), 2), "lado": lado, "organismo": est.get("organismo", lado),
             "voz": voz, "titulo": est.get("voz_titulo", voz),   # título en castellano (etiqueta, no significado)
             "arousal": est.get("voz_arousal"), "valence": est.get("voz_valence"),
             "OI": est.get("OI"), "necesidad": est.get("necesidad"), "t_vida": est.get("t")}
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:    # PERMANENTE: sobrevive reinicios (volumen)
            f.write(json.dumps(turno, ensure_ascii=False) + "\n")
    except Exception:
        pass
    with _LOCK:
        _TRANSCRIPT.append(turno)
        if len(_TRANSCRIPT) > MAX_TRANSCRIPT:
            _TRANSCRIPT.pop(0)
        # histograma agrupado por TÍTULO (reposo/descanso/…), no por la etiqueta interna (dolor/…);
        # guarda la etiqueta para que el frontend siga sabiendo el emoji/color de cada voz.
        _tit = turno.get("titulo") or voz
        _h = _HIST[lado].setdefault(_tit, {"n": 0, "label": voz})
        _h["n"] += 1; _h["label"] = voz
    # BIOGRAFÍA de la díada: cada emisión como evento A↔B con contexto + delta del receptor.
    if _HIST_DIADA is not None:
        otro = "B" if lado == "A" else "A"
        rec = _EST.get(otro, {}); rec_prev = _CTX_PREV.get(otro, {})
        def _d(k):
            try:
                return round(float(rec.get(k, 0) or 0) - float(rec_prev.get(k, 0) or 0), 4)
            except Exception:
                return None
        _emisor = est.get("organismo", lado)
        _HIST_DIADA.comunicacion({
            "ts": turno["ts"], "emisor": _emisor, "receptor": rec.get("organismo", otro),
            "prototipo_codebook": voz, "duracion": None, "energia": est.get("energia"),
            "t_vida_emisor": est.get("t"), "t_vida_receptor": rec.get("t"),
            "contexto_emisor": {"necesidad": est.get("necesidad"), "OI": est.get("OI"),
                                "ICES": est.get("RC_total"), "arousal": est.get("voz_arousal"),
                                "valence": est.get("voz_valence"), "orientacion": est.get("orientacion_deg")},
            "contexto_receptor": {"necesidad": rec.get("necesidad"), "OI": rec.get("OI"),
                                  "orientacion": rec.get("orientacion_deg"), "voz": rec.get("voz_emitida")},
            "delta_receptor": {"delta_OI": _d("OI"), "delta_necesidad": _d("necesidad"),
                               "delta_orientacion": _d("orientacion_deg")},
        })
        _stats_note_turno(_emisor, voz)   # contadores /stats en O(1); no re-scan de JSONL
        _CTX_PREV[otro] = dict(rec)


def _detectar_lexico(lado, m):
    # CREACIÓN (voz_creadas↑) = inventó una palabra; EMULACIÓN (voz_aprendidas↑) = el otro la imita → léxico compartido.
    try:
        cre = int(m.get("voz_creadas") or 0); apr = int(m.get("voz_aprendidas") or 0)
    except Exception:
        return
    org = m.get("organismo", lado); prev = _LEXPREV.get(lado)
    if prev is not None:
        pcre, papr = prev
        if cre > pcre:
            _LEXICO.append({"ts": round(time.time(), 2), "lado": lado, "org": org, "tipo": "creada", "n": cre})
        if apr > papr:
            _LEXICO.append({"ts": round(time.time(), 2), "lado": lado, "org": org, "tipo": "emulada", "n": apr})
        del _LEXICO[:-80]
    _LEXPREV[lado] = (cre, apr)


def _poller():
    while True:
        for lado, url in ROSTER:
            try:
                est = _get_json(url, "/estado")
                # FILA COMPLETA (256 cols) para las cajas que usan met_*/mem_*/Omega/salud/etc.
                try:
                    uf = _get_json(url, "/ultima_fila"); fila = (uf or {}).get("fila") or {}
                except Exception:
                    fila = {}
                merged = dict(fila); merged.update(est or {})   # el estado breve (curado) gana sobre la fila cruda
                with _LOCK:
                    _EST[lado] = merged
                    _detectar_lexico(lado, merged)
                voz = merged.get("voz_emitida", "-")
                if merged.get("vivo") and voz != _ULTIMA[lado]:   # CAMBIÓ de pito = un turno
                    _ULTIMA[lado] = voz
                    _registrar_turno(lado, merged)
            except Exception:
                with _LOCK:
                    _EST[lado] = {"vivo": False, "organismo": lado}
        time.sleep(POLL_S)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if "json" in ctype or "html" in ctype else ""))
        self.send_header("Content-Length", str(len(b)))
        # Sin caché: el observatorio cambia seguido; el navegador debe traer siempre lo último (no versión vieja).
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        try:
            self.wfile.write(b)
        except Exception:
            pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML, "text/html")
        elif self.path.startswith("/datos"):
            with _LOCK:
                _d = {L: _EST.get(L, {}) for L, _ in ROSTER}
                _d.update({"roster": [L for L, _ in ROSTER],
                           "transcript": _TRANSCRIPT[-120:], "lexico": _LEXICO[-50:],
                           "hist": _HIST, "log": LOG_PATH})
                self._send(200, json.dumps(_d, ensure_ascii=False))
        elif self.path.startswith("/voz/"):
            # PROXY de voz (mismo origen → sin CORS): /voz/A o /voz/B.
            # Con ?label= / ?titulo= / ?palabra= → esa palabra del repertorio del organismo.
            from urllib.parse import urlparse, parse_qs, quote
            pu = urlparse(self.path)
            lado = (pu.path.split("/voz/", 1)[1][:1] or "").upper()
            try:
                url = dict(ROSTER).get(lado)
            except Exception:
                url = None
            if not url:
                url = A_URL if lado == "A" else B_URL
            if not url:
                self._send(404, b"", "audio/wav")
            else:
                q = parse_qs(pu.query)
                clave = ((q.get("label") or q.get("titulo") or q.get("palabra") or [""])[0] or "").strip()
                if clave:
                    dest = url.rstrip("/") + "/voz?label=" + quote(clave, safe="")
                else:
                    dest = url.rstrip("/") + "/voz?seg=1.0&modo=R2D2"
                try:
                    with urllib.request.urlopen(dest, timeout=5) as r:
                        self._send(200, r.read(), "audio/wav")
                except Exception:
                    self._send(503, b"", "audio/wav")
        elif self.path.startswith("/dias"):
            self._send(200, json.dumps(_dias_disponibles(), ensure_ascii=False))
        elif self.path.startswith("/turnos"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            dia = (q.get("dia") or [None])[0]; voz = (q.get("voz") or [None])[0]
            lim = int((q.get("limite") or ["500"])[0])
            self._send(200, json.dumps(_leer_turnos(dia, voz, lim), ensure_ascii=False))
        elif self.path.startswith("/stats"):
            self._send(200, json.dumps(_stats_historial(), ensure_ascii=False))
        elif self.path.startswith("/historial"):
            try:
                with open(LOG_PATH, encoding="utf-8") as f:
                    self._send(200, f.read(), "text/plain")
            except Exception:
                self._send(200, "", "text/plain")
        else:
            self._send(404, json.dumps({"error": "no encontrado"}))

    def do_POST(self):
        # PROXY de CORTE (mismo origen :9100 → sin CORS): el observatorio reenvía al organismo su /mute.
        # Permite cortar/restaurar UN canal (L o R) desde la página de los 4 sin tocar el otro (set_mute
        # ignora el lado no enviado). El estado real vive en el organismo (RUN.mute_L/R): fuente única de
        # verdad, por eso es BIDIRECCIONAL — los botones del observatorio leen mute_L/R de /datos y la
        # página individual sincroniza sus checkboxes desde /niveles.
        if self.path.split("?")[0] == "/cortar":
            try:
                ln = int(self.headers.get("Content-Length", 0) or 0)
                req = json.loads(self.rfile.read(ln) or b"{}")
            except Exception:
                req = {}
            org = str(req.get("org", "")).upper()
            canal = str(req.get("canal", "")).upper()
            cortar = bool(req.get("cortar"))
            url = dict(ROSTER).get(org)
            if not url or canal not in ("L", "R"):
                self._send(400, json.dumps({"ok": False, "error": "org/canal inválido"})); return
            payload = {"left": cortar} if canal == "L" else {"right": cortar}
            try:
                pr = urllib.request.Request(url + "/mute", data=json.dumps(payload).encode("utf-8"),
                                            headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(pr, timeout=4) as r:
                    out = json.loads(r.read() or b"{}")
                self._send(200, json.dumps({"ok": True, "org": org, "canal": canal, "cortar": cortar, "resp": out}, ensure_ascii=False))
            except Exception as e:
                self._send(200, json.dumps({"ok": False, "error": str(e)}))
        else:
            self._send(404, json.dumps({"error": "no encontrado"}))


HTML = """<!doctype html><html lang=es><head><meta charset=utf-8>
<title>Observatorio de la conversación — Díada ANIMA</title>
<link href="https://cdn.jsdelivr.net/npm/gridstack@10.3.1/dist/gridstack.min.css" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/gridstack@10.3.1/dist/gridstack-all.js"></script>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js"}}</script>
<script type="module">
import * as THREE from 'three';

function clamp01(x){ x=Number(x||0); return Math.max(0, Math.min(1, x)); }

// Cabeza 3D real estética: un solo grupo orgánico; ojos hundidos, orejas suaves y cabeza clara.
window.drawVSTCabeza3DReal = function(container, thetaDeg, params={}){
  if(!container) return;
  // Paleta alineada con WebLive del organismo (celeste visible, no pastel casi-blanco; incluye negro).
  const TONOS_CARA={
    blanco:{head:0xe6dfd5,cuenca:0xc9c2b8,oreja:0xded9cf,orejaIn:0xc6c5c0},
    celeste:{head:0x5daeea,cuenca:0x3c82bd,oreja:0x67b4e9,orejaIn:0x2e6f9f},
    rosado:{head:0xe2789a,cuenca:0xb84e70,oreja:0xdb7896,orejaIn:0xa53e60},
    amarillo:{head:0xe6c84d,cuenca:0xb99d31,oreja:0xddc24d,orejaIn:0xa88d29},
    cafe:{head:0x9a623c,cuenca:0x714329,oreja:0x8c5738,orejaIn:0x603820},
    negro:{head:0x20242b,cuenca:0x11151b,oreja:0x292f38,orejaIn:0x0b0e13}
  };
  const normGenero=g=>{g=(g||'masculino').toLowerCase();return(g==='f'||g==='femenino'||g==='female')?'femenino':'masculino';};
  const normTono=t=>{t=(t||'blanco').toLowerCase();return TONOS_CARA[t]?t:'blanco';};

  if(!container._vstHeadReal){
    container.innerHTML = '';

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
    camera.position.set(0, 0.08, 4.45);

    const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true, powerPreference:'high-performance'});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.25;
    container.appendChild(renderer.domElement);

    // Iluminación clara y satinada, cercana al look aprobado.
    const hemi = new THREE.HemisphereLight(0xddeeff, 0x182231, 1.7);
    scene.add(hemi);

    const key = new THREE.DirectionalLight(0xffffff, 2.6);
    key.position.set(3.8, 4.2, 5.0);
    scene.add(key);

    const fill = new THREE.DirectionalLight(0xb7d7ff, 1.1);
    fill.position.set(-3.5, 1.8, 2.2);
    scene.add(fill);

    const rim = new THREE.DirectionalLight(0x66cfff, 1.2);
    rim.position.set(-4.0, 1.2, -3.0);
    scene.add(rim);

    const warm = new THREE.PointLight(0xffd7b0, 1.5, 6);
    warm.position.set(1.0, 1.2, 2.3);
    scene.add(warm);

    // Grupo único: TODO gira junto como una sola cabeza.
    const cabeza = new THREE.Group();
    scene.add(cabeza);

    // Esfera base clara, porcelana/metal satinado. No oscura.
    const headGeo = new THREE.SphereGeometry(1, 96, 96);
    const headMat = new THREE.MeshPhysicalMaterial({
      color:0xe6dfd5,
      metalness:0.14,
      roughness:0.32,
      clearcoat:0.38,
      clearcoatRoughness:0.24,
      reflectivity:0.55
    });
    const head = new THREE.Mesh(headGeo, headMat);
    head.scale.set(1.05, 1.0, 1.02);
    cabeza.add(head);

    // Ojos como la referencia: cuenca hundida, blanco visible, iris azul profundo y brillo grande.
    function crearOjo(x){
      const grupo = new THREE.Group();

      const cuencaGeo = new THREE.SphereGeometry(0.215, 56, 56);
      const cuencaMat = new THREE.MeshPhysicalMaterial({
        color:0xc9c2b8,
        metalness:0.05,
        roughness:0.36,
        clearcoat:0.25,
        clearcoatRoughness:0.20
      });
      const cuenca = new THREE.Mesh(cuencaGeo, cuencaMat);
      cuenca.scale.set(1.15, 1.0, 0.34);
      cuenca.position.z = -0.035;
      grupo.add(cuenca);

      const blancoGeo = new THREE.SphereGeometry(0.165, 64, 64);
      const blancoMat = new THREE.MeshPhysicalMaterial({
        color:0xf3f6fb,
        roughness:0.10,
        clearcoat:1.0,
        clearcoatRoughness:0.04,
        reflectivity:0.65
      });
      const blanco = new THREE.Mesh(blancoGeo, blancoMat);
      blanco.scale.set(1.0, 1.0, 0.45);
      blanco.position.z = 0.018;
      grupo.add(blanco);

      const irisGeo = new THREE.SphereGeometry(0.094, 48, 48);
      const irisMat = new THREE.MeshPhysicalMaterial({
        color:0x063f78,
        emissive:0x031a3a,
        emissiveIntensity:0.22,
        roughness:0.04,
        clearcoat:1.0,
        clearcoatRoughness:0.015,
        reflectivity:0.9
      });
      const iris = new THREE.Mesh(irisGeo, irisMat);
      iris.scale.set(1.0, 1.0, 0.30);
      iris.position.z = 0.090;
      grupo.add(iris);

      const irisLuzGeo = new THREE.SphereGeometry(0.062, 32, 32);
      const irisLuzMat = new THREE.MeshBasicMaterial({
        color:0x0d8cff,
        transparent:true,
        opacity:0.42
      });
      const irisLuz = new THREE.Mesh(irisLuzGeo, irisLuzMat);
      irisLuz.scale.set(1.0, 1.0, 0.18);
      irisLuz.position.set(-0.012, 0.012, 0.112);
      grupo.add(irisLuz);

      const pupilaGeo = new THREE.SphereGeometry(0.045, 32, 32);
      const pupilaMat = new THREE.MeshBasicMaterial({color:0x000711});
      const pupila = new THREE.Mesh(pupilaGeo, pupilaMat);
      pupila.scale.set(1.0, 1.0, 0.22);
      pupila.position.z = 0.132;
      grupo.add(pupila);

      const brilloGeo = new THREE.SphereGeometry(0.029, 16, 16);
      const brilloMat = new THREE.MeshBasicMaterial({color:0xffffff});
      const brillo = new THREE.Mesh(brilloGeo, brilloMat);
      brillo.position.set(-0.045, 0.052, 0.154);
      grupo.add(brillo);

      grupo.position.set(x, 0.16, 0.925);
      grupo.rotation.x = -0.04;
      grupo.cuencaMat = cuencaMat;
      return grupo;
    }

    function crearPestanas(ojo){
      const grp=new THREE.Group();
      const mat=new THREE.MeshBasicMaterial({color:0x120e0c, depthTest:false});
      [[-0.052,0.125,0.105,0.20],[-0.026,0.142,0.112,0.10],[0,0.150,0.118,0.02],[0.026,0.142,0.112,-0.10],[0.052,0.125,0.105,-0.20]].forEach(([x,y,z,rz])=>{
        const lash=new THREE.Mesh(new THREE.BoxGeometry(0.034,0.008,0.012),mat);
        lash.position.set(x,y,z); lash.rotation.z=rz; lash.renderOrder=4; grp.add(lash);
      });
      ojo.add(grp); grp.visible=false; return grp;
    }

    const ojoL = crearOjo(-0.23);
    const ojoR = crearOjo(0.23);
    const pestanasL = crearPestanas(ojoL);
    const pestanasR = crearPestanas(ojoR);
    cabeza.add(ojoL, ojoR);

    // BOCA EVALUATIVA — indicador del OVE (valoración experiencial), NO emoción humana.
    // 3 formas DISCRETAS: favorable=arco arriba(U) · neutra=recta · desfavorable=arco abajo(∩).
    // Sólo cambia la boca; nada de ojos/orejas/cabeza/rotación. Sin interpolación.
    const bocaMat = new THREE.MeshBasicMaterial({color:0x3a3633, transparent:true, opacity:0.7, depthTest:false});
    let _boca = null, _generoInit = 'masculino';
    // z de la SUPERFICIE de la esfera (escala 1.05,1.0,1.02) en (x,y), + adelanto para ir SIEMPRE por delante.
    function _zSup(x,y){ const r2 = 1 - (x/1.05)*(x/1.05) - y*y; return 1.02*Math.sqrt(Math.max(0.02, r2)) + 0.03; }
    function construirBoca(cara, genero){              // cara ∈ {-1,0,+1} = desfavorable/neutra/favorable
      genero = normGenero(genero || _generoInit);
      const dy = (cara>0 ? -0.055 : (cara<0 ? 0.055 : 0.0));   // U(+1) / recta(0) / ∩(-1)
      const xs = [-0.16, -0.055, 0.055, 0.16], ys = [-0.305, -0.305+dy, -0.305+dy, -0.305];
      const pts = xs.map((x,i)=> new THREE.Vector3(x, ys[i], _zSup(x, ys[i])));   // sobre la cara, nunca dentro
      const radio = genero==='femenino' ? 0.017 : 0.012;
      const geo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 28, radio, 8, false);
      if(_boca){ _boca.geometry.dispose(); _boca.geometry = geo; }   // SWAP: la boca nunca sale de la escena
      else { _boca = new THREE.Mesh(geo, bocaMat); _boca.renderOrder = 5; cabeza.add(_boca); }
    }
    construirBoca(0);                          // nace neutra (recta, siempre visible)

    // Orejas tipo sensor suave, no cilindros. Integradas al grupo.
    function crearOreja(colorNeon, lado){
      const grupo = new THREE.Group();

      const baseGeo = new THREE.SphereGeometry(0.34, 56, 56);
      baseGeo.scale(0.54, 0.95, 0.28);
      const baseMat = new THREE.MeshPhysicalMaterial({
        color:0xded9cf,
        metalness:0.18,
        roughness:0.24,
        clearcoat:0.65,
        clearcoatRoughness:0.12,
        reflectivity:0.55
      });
      const base = new THREE.Mesh(baseGeo, baseMat);
      grupo.add(base);

      const innerGeo = new THREE.SphereGeometry(0.235, 48, 48);
      innerGeo.scale(0.40, 0.80, 0.16);
      const innerMat = new THREE.MeshPhysicalMaterial({
        color:0xc6c5c0,
        metalness:0.20,
        roughness:0.17,
        clearcoat:0.85,
        clearcoatRoughness:0.05
      });
      const inner = new THREE.Mesh(innerGeo, innerMat);
      inner.position.z = 0.045;
      grupo.add(inner);

      const ringGeo = new THREE.TorusGeometry(0.225, 0.020, 16, 96);
      const ringMat = new THREE.MeshStandardMaterial({
        color:colorNeon,
        emissive:colorNeon,
        emissiveIntensity:0.45,
        roughness:0.16,
        metalness:0.12
      });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.scale.set(0.68, 1.0, 1.0);
      grupo.add(ring);

      const glowGeo = new THREE.TorusGeometry(0.225, 0.044, 12, 96);
      const glowMat = new THREE.MeshBasicMaterial({
        color:colorNeon,
        transparent:true,
        opacity:0.12,
        depthWrite:false
      });
      const glow = new THREE.Mesh(glowGeo, glowMat);
      glow.scale.set(0.68, 1.0, 1.0);
      grupo.add(glow);

      grupo.position.set(lado * 1.04, 0.02, 0.02);
      grupo.rotation.y = lado > 0 ? Math.PI / 2 : -Math.PI / 2;
      grupo.ringMat = ringMat;
      grupo.glowMat = glowMat;
      grupo.baseMat = baseMat;
      grupo.innerMat = innerMat;
      return grupo;
    }

    const orejaL = crearOreja(0x33eaff, -1);
    const orejaR = crearOreja(0xff5264, 1);
    cabeza.add(orejaL, orejaR);

    function aplicarApariencia(st, genero, tono){
      genero = normGenero(genero); tono = normTono(tono);
      const pal = TONOS_CARA[tono];
      if(st.headMat) st.headMat.color.setHex(pal.head);
      if(st.cuencaMatL) st.cuencaMatL.color.setHex(pal.cuenca);
      if(st.cuencaMatR) st.cuencaMatR.color.setHex(pal.cuenca);
      if(st.orejaBaseMatL) st.orejaBaseMatL.color.setHex(pal.oreja);
      if(st.orejaBaseMatR) st.orejaBaseMatR.color.setHex(pal.oreja);
      if(st.orejaInnerMatL) st.orejaInnerMatL.color.setHex(pal.orejaIn);
      if(st.orejaInnerMatR) st.orejaInnerMatR.color.setHex(pal.orejaIn);
      const fem = genero==='femenino';
      if(st.pestanasL) st.pestanasL.visible = fem;
      if(st.pestanasR) st.pestanasR.visible = fem;
      if(st.bocaMat){
        st.bocaMat.color.setHex(fem ? 0xc84858 : 0x3a3633);
        st.bocaMat.opacity = fem ? 0.88 : 0.7;
      }
      _generoInit = genero;
      st._genero = genero; st._tono = tono;
      if(st.construirBoca) st.construirBoca(st._cara||0, genero);
    }

    function resize(){
      const r = container.getBoundingClientRect();
      // Respetar el tamaño real del contenedor (caja padre); no inflar el canvas por encima del CSS.
      const w = Math.max(1, Math.round(r.width || 240));
      const h = Math.max(1, Math.round(r.height || 264));
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
      renderer.domElement.style.width = '100%';
      renderer.domElement.style.height = '100%';
      renderer.domElement.style.display = 'block';
    }
    resize();
    window.addEventListener('resize', resize);

    // Apariencia desde params en el primer frame (no forzar blanco).
    const g0 = params.genero ?? params.cara_genero ?? 'masculino';
    const t0 = params.tono ?? params.cara_tono ?? 'blanco';
    container._vstHeadReal = {
      scene, camera, renderer, cabeza, orejaL, orejaR, resize, construirBoca, aplicarApariencia,
      headMat, bocaMat, cuencaMatL:ojoL.cuencaMat, cuencaMatR:ojoR.cuencaMat,
      orejaBaseMatL:orejaL.baseMat, orejaBaseMatR:orejaR.baseMat,
      orejaInnerMatL:orejaL.innerMat, orejaInnerMatR:orejaR.innerMat,
      pestanasL, pestanasR, _cara:0, _genero:null, _tono:null
    };
    aplicarApariencia(container._vstHeadReal, g0, t0);
  }

  const h = container._vstHeadReal;
  h.resize();

  // Encuadre opcional (p. ej. Sociedad): cabeza completa con orejas dentro del contenedor.
  const fit = Math.max(0.45, Math.min(1.0, Number(params.fitScale ?? 1.0)));
  h.cabeza.scale.setScalar(fit);

  const genero = params.genero ?? params.cara_genero ?? h._genero ?? 'masculino';
  const tono = params.tono ?? params.cara_tono ?? h._tono ?? 'blanco';
  if(h.aplicarApariencia && (genero !== h._genero || tono !== h._tono)){
    h.aplicarApariencia(h, genero, tono);
  }

  // Rotación real del sólido completo: esfera, ojos, sonrisa y sensores giran juntos.
  h.cabeza.rotation.y = THREE.MathUtils.degToRad(Number(thetaDeg || 0));

  const energiaL = clamp01(params.energiaL ?? params.energyL ?? 0);
  const energiaR = clamp01(params.energiaR ?? params.energyR ?? 0);
  if(h.orejaL.ringMat) h.orejaL.ringMat.emissiveIntensity = 0.45 + energiaL * 2.8;
  if(h.orejaR.ringMat) h.orejaR.ringMat.emissiveIntensity = 0.45 + energiaR * 2.8;
  if(h.orejaL.glowMat) h.orejaL.glowMat.opacity = 0.10 + Math.min(0.35, energiaL * 0.60);
  if(h.orejaR.glowMat) h.orejaR.glowMat.opacity = 0.10 + Math.min(0.35, energiaR * 0.60);

  // BOCA evaluativa: estado DISCRETO del OVE. Sólo reconstruye la boca cuando el estado cambia.
  const cara = Math.sign(Number(params.cara || 0));   // -1/0/+1 (lectura visual, sin causalidad)
  if(h.construirBoca && cara !== h._cara){ h.construirBoca(cara, h._genero); h._cara = cara; }

  h.renderer.render(h.scene, h.camera);
};
</script>
<style>
:root{--bg:#16223a;--panel:#1e2c44;--bord:#34486b;--mut:#a4b6d0;--ok:#56d6a0;--gold:#f0c47b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#dfe7f0;font:13px system-ui,sans-serif}
h1{font-size:16px;margin:10px 14px}.mut{color:var(--mut)}
.escena{display:flex;align-items:center;justify-content:center;gap:60px;padding:18px;background:var(--panel);
  border-bottom:1px solid var(--bord);min-height:200px}
.org{text-align:center;width:230px}
.cabeza{width:120px;height:120px;border-radius:50%;margin:0 auto;position:relative;
  background:radial-gradient(circle at 40% 35%,#2a3a4f,#16202e);border:2px solid var(--bord);transition:transform .25s}
.nariz{position:absolute;top:50%;left:50%;width:54px;height:6px;border-radius:3px;background:var(--gold);
  transform-origin:left center;transform:translateY(-50%)}
.ojo{position:absolute;width:12px;height:12px;border-radius:50%;background:#cfe0f0;top:38px}
.pito{margin-top:10px;font-size:20px;font-weight:bold;min-height:26px}
.afecto{font-size:11px}.vmed{height:8px;background:#0c121b;border:1px solid var(--bord);border-radius:4px;overflow:hidden;margin-top:5px}
.vmed>div{height:100%;background:var(--ok)}
.entre{font-size:30px;color:var(--gold)}
.cols{display:flex;gap:0}
.trans{flex:1;height:46vh;overflow:auto;padding:8px 14px;border-right:1px solid var(--bord)}
.linea{padding:2px 0;border-bottom:1px solid #1a2330;font-family:ui-monospace,monospace;font-size:12px}
.A{color:#6db6ff}.B{color:#ffd479}
.hist{width:300px;padding:10px 14px}
/* Mini-reproductor por palabra (▶ en En vivo / Historia) */
.playw{display:inline-flex;align-items:center;justify-content:center;width:17px;height:17px;margin:0 5px 0 0;padding:0;border:1px solid #35506e;background:#1d2c44;color:#9fb1c6;border-radius:50%;font-size:9px;line-height:1;cursor:pointer;vertical-align:middle;flex:none}
.playw:hover{background:#26395a;color:#dfe7f0;border-color:#6db6ff}
.playw.playing{background:#1a3a2a;color:#8ef0c0;border-color:#5fd38a}
.playw:disabled{opacity:.35;cursor:default}
.barra{display:flex;align-items:center;gap:6px;margin:2px 0;font-size:11px}
.barra .b{flex:1;height:10px;background:#0c121b;border:1px solid var(--bord);border-radius:3px;overflow:hidden}
.barra .b>div{height:100%}
.tab{background:#16202e;color:#dfe7f0;border:1px solid var(--bord);border-radius:6px;padding:5px 11px;font-size:12px;cursor:pointer}
.tab.on{background:var(--gold);color:#10171f;font-weight:bold}
select{background:#16202e;color:#dfe7f0;border:1px solid var(--bord);border-radius:5px;padding:3px 6px;font-size:12px}
.sm{background:#16202e;color:#dfe7f0;border:1px solid var(--bord);border-radius:6px;padding:4px 9px;font-size:11px;cursor:pointer}
.panel{background:var(--panel);border:1px solid var(--bord);border-radius:9px;padding:9px;margin:8px 14px}
.grid-stack{background:transparent}
.obscaja{background:#0e1622;border:1px solid var(--bord);border-radius:10px;height:100%;overflow:auto;display:flex;flex-direction:column}
.obscaja h3{margin:0;padding:7px 10px;font-size:12px;border-bottom:1px solid #1d2940;color:var(--gold);display:flex;align-items:center;gap:6px}
.obscaja .obsbody{padding:8px 10px;font-size:11px;flex:1}
.obscaja .obsdel{margin-left:auto;cursor:pointer;color:#ff8c8c;font-size:13px;display:none}
#obsZona.editando .obscaja .obsdel{display:inline}
#obsZona.editando .obscaja{border-color:#3a557a;box-shadow:0 0 0 1px #3a557a44}
.obsrow{display:flex;justify-content:space-between;gap:8px;margin:3px 0}
.obsk{color:#9fb1c6}.obsv{color:#e6eefb;font-variant-numeric:tabular-nums}
.obsgauge{height:6px;border-radius:4px;background:#1b2740;overflow:hidden;margin:2px 0 5px}
.obsgauge>i{display:block;height:100%;border-radius:4px}
.obschip{font-size:10px;border:1px solid #2a3a55;border-radius:6px;padding:3px 7px;cursor:pointer;background:#101a28;color:#cfe0f5}
.obschip:hover{border-color:#4a6da0}.obschip.puesta{opacity:.4;cursor:default}

/* ===== REDISEÑO escena: dos cabezas 3D mirándose, piso en perspectiva ===== */
.escena{position:relative;min-height:320px;gap:80px;overflow:hidden;
  background:radial-gradient(ellipse at 50% 28%,#16222f 0%,#0a1018 62%,#070b11 100%)}
.escena:before{content:"";position:absolute;inset:56% -25% -35% -25%;
  background:linear-gradient(rgba(109,182,255,.09) 1px,transparent 1px),
            linear-gradient(90deg,rgba(109,182,255,.09) 1px,transparent 1px);
  background-size:46px 46px;transform:perspective(640px) rotateX(60deg);transform-origin:50% 0;pointer-events:none}
.escena:after{content:"";position:absolute;left:0;right:0;top:0;height:80px;
  background:linear-gradient(#070b11,transparent);pointer-events:none}
.org{position:relative;z-index:1;width:240px}
.head3d{width:214px;height:214px;margin:0 auto;filter:drop-shadow(0 14px 26px rgba(0,0,0,.55))}
.nameplate{display:flex;align-items:center;justify-content:center;gap:8px;font-size:13px;font-weight:600;
  letter-spacing:.05em;margin-bottom:2px;color:#e9f1fb}
.npdot{width:9px;height:9px;border-radius:50%}
.npA{background:#6db6ff;box-shadow:0 0 9px #6db6ff}
.npB{background:#ffd479;box-shadow:0 0 9px #ffd479}
.oilab{font-size:9px;text-transform:uppercase;letter-spacing:.09em;margin-top:3px;opacity:.75}
.entre{font-size:34px;filter:drop-shadow(0 0 10px rgba(232,184,109,.5));transition:opacity .3s}
.pito{margin-top:8px}
/* histograma compacto: nunca empuja el tablero */
.hist{max-height:46vh;overflow:auto}

/* ===== TEMA MODERNO: más claro y aireado (menos ominoso) ===== */
body{background:linear-gradient(180deg,#22344f 0%,#1a2840 55%,#141f33 100%) fixed;color:#e9f0f8}
h1,h2{text-shadow:0 1px 2px #00000055}
.panel{background:linear-gradient(180deg,#24344f,#1d2a41);box-shadow:0 6px 18px #00000030;border-color:#34486b;border-radius:11px}
.obscaja{background:linear-gradient(180deg,#23334f,#1c2942);border:1px solid #34486b;box-shadow:0 5px 16px #00000038;border-radius:12px}
.obscaja h3{color:#f3cb84;border-bottom-color:#2e4163}
.obsgauge{background:#2b3c5c}
.obschip{background:#23334f;border-color:#3a517a;color:#dbe7f7}
.obschip:hover{border-color:#6f97cf;background:#27395a}
.trans{border-right-color:#2e4163}.linea{border-bottom-color:#22324c}
.tab{background:#23334f;box-shadow:0 2px 7px #0000002e}
.tab.on{box-shadow:0 2px 9px #e8b86d44}
.escena{background:radial-gradient(ellipse at 50% 30%,#243b5e 0%,#172642 58%,#101b2e 100%)!important}
select,.sm{background:#23334f;border-color:#3a517a}

/* ===== CIRCUITO VIVO ===== */
.cbar{display:flex;align-items:center;gap:10px;padding:8px 14px;background:var(--panel);border-bottom:1px solid var(--bord);flex-wrap:wrap}
.cwrap{position:relative;display:flex;gap:0;align-items:stretch}
#csvg{flex:1;width:100%;height:calc(100vh - 150px);background:radial-gradient(ellipse at 50% 18%,#22375c 0%,#16233c 55%,#0f1a2c 100%)}
.cinfo{width:310px;min-width:270px;max-width:34vw;padding:15px;background:linear-gradient(180deg,#22324f,#1a2840);border-left:1px solid var(--bord);overflow:auto}
.cinfo h4{margin:0 0 2px;color:var(--gold);font-size:15px}
.cinfo .sub{color:#9fb1c6;font-size:10px;text-transform:uppercase;letter-spacing:.06em}
.cinfo .q{margin:9px 0;font-size:12px;line-height:1.55;color:#dbe7f7}.cinfo .q b{color:#8fd0ff}
.cinfo .vbig{font-size:24px;color:var(--gold);font-weight:bold;margin-top:6px}
.cedge{fill:none;stroke:#3c577f;stroke-width:2;stroke-dasharray:5 7;animation:cflow 1.1s linear infinite}
.cedge.bridge{stroke:var(--gold);stroke-width:2.6;stroke-dasharray:3 10;animation-duration:.9s}
@keyframes cflow{to{stroke-dashoffset:-24}}
.cnode{cursor:pointer}
.cnode text{pointer-events:none;font-family:system-ui,sans-serif;text-anchor:middle}
.cnode .ic{font-size:17px}
.cnode .nm{font-size:10.5px;fill:#eaf1f9;font-weight:600}
.cnode .vl{font-size:9.5px;fill:#a9bdd8}
.cnode .glow{fill:none;opacity:0;transition:opacity .12s}
.cnode.flash .glow{opacity:1!important}
.clabel{fill:#7f93ad;font-size:11px;text-anchor:middle;font-family:system-ui;letter-spacing:.08em;text-transform:uppercase}

/* ===== célula madre: membrana + núcleo + categorías ===== */
.membrana{animation:cbreathe 5.5s ease-in-out infinite}
@keyframes cbreathe{0%,100%{stroke-opacity:.34}50%{stroke-opacity:.72}}
.nucleo{animation:cpulse 3.2s ease-in-out infinite}
@keyframes cpulse{0%,100%{opacity:.45}50%{opacity:.95}}
.cedge.stim{stroke:#9fd0ff;stroke-dasharray:7 6}
.cedge.read{stroke:#8071b3;stroke-dasharray:2 8}
.stimlab{fill:#9fd0ff;font-size:10px;font-family:system-ui;text-anchor:start;letter-spacing:.06em;text-transform:uppercase}
.leglab{fill:#9fb1c6;font-size:11px;font-family:system-ui}
.catdot{filter:drop-shadow(0 0 3px currentColor)}
.facehelp{font-size:11.5px;color:#b9c8de;background:var(--panel);border-bottom:1px solid var(--bord);padding:7px 16px;line-height:1.5}
.facehelp b{color:#8fd0ff}

/* ===== eventos de léxico (palabra propia / imitada) ===== */
.lexbanner{max-height:0;overflow:hidden;opacity:0;transition:all .4s ease;text-align:center;font-size:13px;color:#dfe7f0}
.lexbanner.show{max-height:52px;opacity:1;padding:10px 14px;border-bottom:1px solid var(--bord)}
.lexbanner.cre{background:linear-gradient(90deg,#3a2f14,#241d0d);color:#f3cb84;box-shadow:inset 0 0 34px #e8b86d22}
.lexbanner.emu{background:linear-gradient(90deg,#123626,#0f2a20);color:#8ef0c0;box-shadow:inset 0 0 34px #5fd38a22;animation:lexpulse 1.1s ease 2}
@keyframes lexpulse{50%{box-shadow:inset 0 0 54px #5fd38a55}}

/* ===== pestaña Sociedad ===== */
.socbar{display:flex;align-items:center;gap:10px;padding:8px 14px;background:var(--panel);border-bottom:1px solid var(--bord);flex-wrap:wrap}
.socgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px;padding:16px;max-width:1500px;margin:auto}
.soccard{background:linear-gradient(180deg,#22324f,#1a2840);border:1px solid var(--bord);border-radius:14px;padding:12px;text-align:center;box-shadow:0 5px 16px #00000038;transition:opacity .3s}
.soccard.off{opacity:.4;filter:grayscale(.65)}
.soccard h3{margin:0 0 2px;font-size:14px}
.sochead{width:100%;height:264px;margin:0 auto;overflow:hidden;filter:drop-shadow(0 10px 18px rgba(0,0,0,.5))}
.socvoz{font-size:13px;margin:5px 0 7px;min-height:20px;color:#e9f1fb}
.socvit{display:flex;justify-content:space-between;font-size:10.5px;color:#9fb1c6;margin:2px 6px;border-top:1px solid #243246;padding-top:2px}
.socvit b{color:#cfe0f5;font-variant-numeric:tabular-nums}
/* ENTRADA POR OÍDO (L/R) + REACCIÓN al sonido. Un oído lleva la voz del par; el otro, el audio de
   contexto que el observador define en la página de cada organismo. El emoji = la cara que pone el
   organismo (propiocepción: 😊 le gusta · 😐 indiferente · 🙁 le disgusta). */
.socoidos{display:flex;flex-direction:column;gap:6px;margin:6px 4px 8px;padding:8px 7px;background:#16223a;border:1px solid #243246;border-radius:10px}
.socoid-row{display:flex;align-items:center;gap:7px;font-size:10.5px;color:#9fb1c6}
.socoid-side{width:14px;flex:none;font-weight:700;color:#7d8ea6;text-align:center}
.socoid-src{flex:1;min-width:0;text-align:left;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#cfe0f5}
.socoid-bar{width:58px;flex:none;height:8px;background:#0e1726;border-radius:5px;overflow:hidden}
.socoid-bar>div{height:100%;width:0;border-radius:5px;transition:width .15s,background .3s}
.socoid-emo{width:24px;flex:none;text-align:center;font-size:19px;line-height:1;filter:drop-shadow(0 1px 3px #0007)}
.soccut{flex:none;width:20px;height:18px;padding:0;border:1px solid #35506e;background:#1d2c44;color:#9fb1c6;border-radius:5px;font-size:11px;line-height:1;cursor:pointer;transition:background .15s,color .15s,border-color .15s}
.soccut:hover{background:#26395a;color:#dfe7f0}
.soccut.cut{background:#5a1f1f;border-color:#b34;color:#ff9a9a}
</style></head><body>
<h1>🛰️ Observatorio de la conversación · <span class=mut>díada ANIMA — captura permanente</span></h1>
<div style="display:flex;align-items:center;gap:10px;padding:6px 14px;background:var(--panel);border-bottom:1px solid var(--bord);flex-wrap:wrap">
  <button class=tab id=tabVivo onclick="vista('vivo')">🟢 En vivo</button>
  <button class=tab id=tabHist onclick="vista('hist')">🕮 Historia</button>
  <button class=tab id=tabCirc onclick="vista('circ')">🫀 Circuito vivo</button>
  <button class=tab id=tabSoc onclick="vista('soc')">🌐 Sociedad</button>
  <span style="flex:1"></span>
  <button class=tab id=bAudio onclick="audioToggle()">🔊 Escuchar conversación</button>
  <span class=mut style=font-size:11px>vol</span>
  <input type=range id=vol min=0 max=8 step=0.5 value=4 style=width:90px oninput="setVol(this.value)">
  <span id=volV class=mut style=font-size:11px>4×</span>
  <span class=mut style="font-size:10.5px">A↗izquierda · B↗derecha</span>
</div>
<div id=vivo>
<div id=lexbanner class=lexbanner></div>
<div class=escena>
  <div class=org>
    <div class=nameplate><span class="npdot npA"></span><span id=nomA>Organismo A</span></div>
    <div class=head3d id=head3dA></div>
    <div class=pito id=pitoA>—</div>
    <div class=afecto mut id=afA></div>
    <div class=vmed><div id=oiA style=width:0%></div></div>
    <div class=oilab>acople OI ↔ par</div>
  </div>
  <div class=entre id=entre>↔</div>
  <div class=org>
    <div class=nameplate><span class="npdot npB"></span><span id=nomB>Organismo B</span></div>
    <div class=head3d id=head3dB></div>
    <div class=pito id=pitoB>—</div>
    <div class=afecto mut id=afB></div>
    <div class=vmed><div id=oiB style=width:0%></div></div>
    <div class=oilab>acople OI ↔ par</div>
  </div>
</div>
<div class=facehelp>🧬 <b>Las cabezas son las células vivas:</b> giran hacia donde &laquo;miran&raquo; (orientación), los <b>anillos de oído</b> laten con lo que oyen, y la <b>boca</b> es la valoración del OVE — <b>sonríe</b> 😊 si la experiencia le es favorable, se aplana 😐 o se invierte ☹️ si no. Es expresión, no cambia su conducta.</div>
<div class=cols>
  <div class=trans id=trans></div>
  <div class=hist>
    <div class=mut style=margin-bottom:6px>Pitos usados (histograma)</div>
    <div id=histA></div><div style=height:8px></div><div id=histB></div>
    <div class=mut style="font-size:10px;margin-top:10px">Registro permanente: <span id=logp></span><br><a href=/historial style=color:var(--gold)>descargar historial completo</a></div>
  </div>
</div>
<!-- ===== OBSERVATORIO DE LA DÍADA (tablero editable, aditivo · misma lógica que las páginas de organismo) ===== -->
<div id=obsZona style="margin:6px 0 24px">
  <div class=panel style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
    <b style="color:var(--gold)">🧩 Observatorio de la díada</b>
    <span class=mut style="font-size:10px;flex:1;min-width:160px">Agrega y mueve cajas para observar la díada. La caja ⭐ <b>Libertad creativa</b> muestra el balbuceo de ambos.</span>
    <button class=sm id=obsAdd style=display:none>➕ Agregar caja</button>
    <button class=sm id=obsReset style=display:none>↺ Restaurar (vaciar)</button>
    <button class=sm id=obsEdit>✏️ Editar tablero</button>
  </div>
  <div id=obsPaleta class=panel style=display:none>
    <div class=mut style="font-size:10px;margin-bottom:6px">Catálogo · click para agregar al tablero:</div>
    <div id=obsCatalogo style="display:flex;flex-wrap:wrap;gap:6px"></div>
  </div>
  <div class="grid-stack" id=obsGrid></div>
  <div id=obsVacio class=mut style="font-size:11px;text-align:center;padding:14px;opacity:.7">Tablero vacío. Pulsa <b>✏️ Editar tablero</b> → <b>➕ Agregar caja</b> (prueba ⭐ Libertad creativa).</div>
</div>
</div><!-- /vivo -->
<div id=historia style=display:none>
  <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;flex-wrap:wrap;border-bottom:1px solid var(--bord)">
    <span class=mut>Día</span><select id=selDia onchange=cargarHist()></select>
    <span class=mut>Pito</span><select id=selVoz onchange=cargarHist()><option value="">todos</option></select>
    <span id=histInfo class=mut style=font-size:11px></span>
  </div>
  <div class=cols>
    <div class=trans id=transH style=height:54vh></div>
    <div class=hist><div class=mut style=margin-bottom:6px>Acumulado de TODA la biografía</div><div id=statH></div></div>
  </div>
</div>
<div id=circuito style=display:none>
 <div class=cbar>
   <b style="color:var(--gold)">🫀 Circuito vivo de la díada</b>
   <span class=mut style="font-size:11px;flex:1;min-width:200px">Cada organelo late con su valor real; las flechas muestran el flujo del campo. La voz de cada organismo altera el Soma del otro: el lazo se cierra.</span>
   <button class=sm id=cTrace>✴ Trazar el campo</button>
 </div>
 <div class=cwrap>
   <svg id=csvg viewBox="0 0 1280 750" preserveAspectRatio="xMidYMid meet"></svg>
   <div id=cinfo class=cinfo><div class=mut style="font-size:11.5px;line-height:1.5">Pasa el cursor (o haz click) sobre un organelo para ver <b>qué hace</b> y <b>para qué sirve</b>.<br><br>El flujo va: <b>Mundo → Soma → Campo Φ → órganos internos → órganos relacionales → Expresión → Voz</b>, y la voz cierra el lazo en el otro.</div></div>
 </div>
</div>
<div id=sociedad style=display:none>
 <div class=socbar>
   <b style="color:var(--gold)">🌐 Sociedad</b>
   <span class=mut style="font-size:11px;flex:1;min-width:200px">Los organismos activos. Cada uno con su cabeza 3D, su voz y signos vitales. Se adapta a cuántos estén corriendo (los apagados quedan en gris).</span>
   <span id=socN class=mut style="font-size:11px"></span>
 </div>
 <div id=socGrid class=socgrid></div>
</div>
<script>
const $=id=>document.getElementById(id);
const COL={screaming:'#ff5b5b',shout:'#ff8c6b',worried:'#e8b86d',excited:'#5fd38a','excited-2':'#5fd38a',sing:'#8ef0c0',acknowledged:'#6db6ff',chat:'#9fb1c6'};
const emo=v=>({screaming:'😱',shout:'😨',worried:'😟',excited:'🤩','excited-2':'😃',sing:'🎶',acknowledged:'👍',chat:'💬',
  alegria:'😊',miedo:'😨',calma:'😌',curiosidad:'🤔',asombro:'😮',dolor:'😣',tristeza:'😢',ternura:'🥰',
  alerta:'⚠️',peligro:'🚨',urgencia:'⏰',atencion:'👁️',hambre:'🍽️',fatiga:'🥱',saludo:'👋',despedida:'🫡',
  pregunta:'❓',respuesta:'💡',duda:'🤨',confusion:'😵',acuerdo:'🤝',desacuerdo:'🙅',negacion:'🚫',afirmacion:'✅',
  llamada:'📣',compania:'🫂',exploracion:'🧭',hallazgo:'✨',novedad:'🆕',despertar:'🌅',satisfaccion:'😋',frustracion:'😤'}[v]||'🤖');
const _HEAD={
  A:{th:0,tt:0,eL:0,tL:0,eR:0,tR:0,cara:0,genero:'masculino',tono:'blanco'},
  B:{th:0,tt:0,eL:0,tL:0,eR:0,tR:0,cara:0,genero:'masculino',tono:'blanco'}
};
function cabeza(pre, est){
  const h=_HEAD[pre];                                   // objetivos para la animación suave (lerp en _animHeads)
  h.tt=(est&&+est.orientacion_deg)||0;                  // hacia dónde mira
  var _oid=(est&&+est.oao_oido)||0,_ar=(est&&+est.voz_arousal)||0,_eb=Math.max(_oid,_ar*0.6);   // proxy de energía oída si el estado breve no trae per-canal
  h.tL=(est&&est.energia_L!=null)?+est.energia_L:_eb; h.tR=(est&&est.energia_R!=null)?+est.energia_R:_eb;   // anillos de oído laten
  h.cara=Math.sign((est&&+est.cara_valoracion)||0);     // boca evaluativa (OVE): -1/0/+1
  // Look&feel del organismo (Abraxas=celeste, Predator=negro, …) desde /estado
  h.genero=(est&&(est.cara_genero||est.genero))||h.genero||'masculino';
  h.tono=(est&&(est.cara_tono||est.tono))||h.tono||'blanco';
  const v=(est&&est.voz_emitida)||'—', tit=(est&&est.voz_titulo)||v, vivo=est&&est.vivo;
  $('pito'+pre).textContent=vivo?(emo(v)+' '+tit):'· dormido ·';   // título en castellano
  $('pito'+pre).style.color=COL[v]||'#dfe7f0';
  $('af'+pre).textContent=vivo?('aro '+(+est.voz_arousal||0).toFixed(2)+' · val '+(+est.voz_valence||0).toFixed(2)+' · OI '+(+est.OI||0).toFixed(2)):'';
  $('oi'+pre).style.width=Math.min(100,((+((est||{}).OI))||0)*100)+'%';
  if(est&&est.organismo)$('nom'+pre).textContent=est.organismo;
}
function _animHeads(){                                  // las MISMAS cabezas 3D de las páginas de organismo
  for(const pre of ['A','B']){
    const h=_HEAD[pre], c=$('head3d'+pre);
    if(c && window.drawVSTCabeza3DReal){
      h.th+=(h.tt-h.th)*0.12; h.eL+=(h.tL-h.eL)*0.18; h.eR+=(h.tR-h.eR)*0.18;  // giro y latido suaves
      window.drawVSTCabeza3DReal(c, h.th, {
        energiaL:h.eL, energiaR:h.eR, cara:h.cara,
        genero:h.genero||'masculino', tono:h.tono||'blanco',
        cara_genero:h.genero||'masculino', cara_tono:h.tono||'blanco'
      });
    }
  }
  requestAnimationFrame(_animHeads);
}
if(window.requestAnimationFrame) requestAnimationFrame(_animHeads);
function hist(pre, h){
  // h: { titulo: {n, label} }  → muestra el TÍTULO (reposo/…), emoji/color por la etiqueta interna
  const ent=Object.entries(h||{}).map(([t,o])=>[t,(o&&o.n)||0,(o&&o.label)||t]).sort((a,b)=>b[1]-a[1]);
  const tot=ent.reduce((a,b)=>a+b[1],0)||1;
  const _top=ent.slice(0,6), _resto=ent.length-_top.length;   // compacto: solo los 6 más usados
  $('hist'+pre).innerHTML='<div class=mut style=font-size:10px>'+pre+'</div>'+_top.map(([t,n,lab])=>
    '<div class=barra><span style=width:88px>'+emo(lab)+' '+t+'</span><div class=b><div style="width:'+(100*n/tot)+'%;background:'+(COL[lab]||'#5fd38a')+'"></div></div><span style="width:34px;text-align:right" class=mut>'+n+'</span></div>').join('')
    +(_resto>0?'<div class=mut style="font-size:9px;margin-top:4px;opacity:.7">+'+_resto+' pitos más</div>':'');
}
let nT=0;
let _lexSeen=-1;
function lexBanner(feed){
  const el=$('lexbanner'); if(!el)return;
  if(_lexSeen<0){_lexSeen=feed.length;return;}            // baseline: no mostrar el histórico al cargar
  if(feed.length>_lexSeen){
    const x=feed[feed.length-1]; _lexSeen=feed.length;
    const cre=x.tipo==='creada';
    el.innerHTML=cre?('🗣️✨ <b>'+x.org+'</b> inventó una palabra nueva (su banco no la cubría)')
                    :('🗣️↔ <b>'+x.org+'</b> emuló una palabra del otro — <b>léxico compartido</b>');
    el.className='lexbanner show '+(cre?'cre':'emu');
    clearTimeout(window._lexT); window._lexT=setTimeout(()=>{el.className='lexbanner';},9000);
  }
}
async function tick(){
  let d; try{ d=await fetch('/datos').then(r=>r.json()); }catch(e){ return; }
  cabeza('A', d.A); cabeza('B', d.B);
  window._ultD=d; if(window.renderCajas)renderCajas(d);   // OBSERVATORIO de la díada (cajas)
  window._ultLex=d.lexico||[]; lexBanner(window._ultLex);   // eventos de léxico (creación / imitación de palabra)
  // ¿se miran? (A mira a la derecha y B a la izquierda)
  const oa=(d.A&&d.A.orientacion_deg)||0, ob=(d.B&&d.B.orientacion_deg)||0;
  $('entre').textContent=(oa>5 && ob<-5)?'👁️‍🗨️':'↔';
  $('logp').textContent=d.log||'';
  hist('A', (d.hist||{}).A); hist('B', (d.hist||{}).B);
  if(d.transcript && d.transcript.length!==nT){
    nT=d.transcript.length;
    $('trans').innerHTML=d.transcript.slice(-120).map(t=>{
      const hh=new Date(t.ts*1000).toLocaleTimeString();
      const lab=t.voz||t.titulo||'';
      const btn=lab&&lab!=='-'?('<button type=button class=playw data-lado="'+(t.lado||'A')+'" data-label="'+encodeURIComponent(lab)+'" title="escuchar «'+(t.titulo||lab)+'»">▶</button>'):'';
      return '<div class=linea>'+btn+'<span class=mut>'+hh+'</span> <span class='+t.lado+'>'+t.lado+'</span> '+emo(t.voz)+' <b>'+(t.titulo||t.voz)+'</b> <span class=mut>(aro '+(+t.arousal||0).toFixed(2)+', val '+(+t.valence||0).toFixed(2)+')</span></div>';
    }).join('');
    $('trans').scrollTop=$('trans').scrollHeight;
  }
}
// ---- pestañas En vivo / Historia ----
function vista(v){
  $('vivo').style.display=v==='vivo'?'':'none'; $('historia').style.display=v==='hist'?'':'none';
  var _c=$('circuito'); if(_c)_c.style.display=v==='circ'?'':'none';
  $('tabVivo').classList.toggle('on', v==='vivo'); $('tabHist').classList.toggle('on', v==='hist');
  var _tc=$('tabCirc'); if(_tc)_tc.classList.toggle('on', v==='circ');
  var _cs=$('sociedad'); if(_cs)_cs.style.display=v==='soc'?'':'none';
  var _ts=$('tabSoc'); if(_ts)_ts.classList.toggle('on', v==='soc');
  if(v==='hist') initHist();
  if(v==='circ' && window.initCircuito) window.initCircuito();
  if(v==='soc' && window.initSociedad) window.initSociedad();
}
vista('vivo');
// CORTAR/RESTAURAR un canal (L o R) de un organismo desde la página de los 4. Conmuta según el estado
// real (mute_L/R de /datos) y lo manda al proxy /cortar (mismo origen → sin CORS), que reenvía al /mute
// del organismo. Feedback inmediato; el próximo sondeo confirma desde la fuente de verdad (el organismo).
window.cortarCanal=async function(org,side){
  const e=(window._ultD||{})[org]||{};
  const cur=(side==='L')?!!e.mute_L:!!e.mute_R;
  const b=document.getElementById('soc'+side+'cut_'+org); if(b)b.classList.toggle('cut',!cur);
  try{ await fetch('/cortar',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({org:org,canal:side,cortar:!cur})}); }catch(err){}
};
// ===================== PESTAÑA SOCIEDAD (N organismos) =====================
(function(){
  let built=false; const heads={};
  const COL={A:'#6db6ff',B:'#ffd479',C:'#8ef0c0',D:'#ff8c6b',E:'#c9a0ff'};
  function build(d){
    const grid=document.getElementById('socGrid'); if(!grid)return;
    const roster=(d&&d.roster)||['A','B'];
    grid.innerHTML=roster.map(L=>`<div class=soccard id=soc_${L}>
      <h3 style="color:${COL[L]||'#dfe7f0'}">● Organismo ${L}</h3>
      <div class=sochead id=sochead_${L}></div>
      <div class=socoidos id=socoidos_${L}>
        <div class=socoid-row title="entrada IZQUIERDA — la fuente que definas en la página del organismo">
          <button class=soccut id=socLcut_${L} title="cortar / restaurar este canal" onclick="cortarCanal('${L}','L')">✂</button>
          <span class=socoid-side>L</span><span class=socoid-src id=socLsrc_${L}>—</span>
          <div class=socoid-bar><div id=socLbar_${L}></div></div><span class=socoid-emo id=socLemo_${L}>·</span>
        </div>
        <div class=socoid-row title="entrada DERECHA — la fuente que definas en la página del organismo">
          <button class=soccut id=socRcut_${L} title="cortar / restaurar este canal" onclick="cortarCanal('${L}','R')">✂</button>
          <span class=socoid-side>R</span><span class=socoid-src id=socRsrc_${L}>—</span>
          <div class=socoid-bar><div id=socRbar_${L}></div></div><span class=socoid-emo id=socRemo_${L}>·</span>
        </div>
      </div>
      <div class=socvoz id=socvoz_${L}>—</div>
      <div class=socvit><span>imitación</span><b id=socimit_${L}>—</b></div>
      <div class=socvit><span>¿oye?</span><b id=socoye_${L}>—</b></div>
      <div class=socvit><span>Ω campo</span><b id=socomega_${L}>—</b></div>
      <div class=socvit><span>cara (OVE)</span><b id=soccara_${L}>—</b></div>
    </div>`).join('');
    heads.length=0; roster.forEach(L=>{heads[L]={th:0,tt:0,e:0,te:0,cara:0};});
    const sn=document.getElementById('socN'); if(sn)sn.textContent=roster.length+' organismos';
    built=true;
  }
  const cara=c=>c>0.05?'😊 favorable':(c<-0.05?'☹️ desfavorable':'😐 neutra');
  window.initSociedad=function(){ if(!built && window._ultD) build(window._ultD); };
  function tick(){
    requestAnimationFrame(tick);
    const soc=document.getElementById('sociedad');
    if(!soc||soc.style.display==='none'||!window._ultD)return;
    const d=window._ultD, roster=d.roster||['A','B'];
    if(!built || (document.getElementById('socGrid')||{}).children.length!==roster.length){ build(d); return; }
    roster.forEach(L=>{
      const e=d[L]||{}, card=document.getElementById('soc_'+L); if(!card)return;
      const vivo=!!e.vivo; card.classList.toggle('off',!vivo);
      const h=heads[L], c=document.getElementById('sochead_'+L);
      if(h&&c&&window.drawVSTCabeza3DReal){
        h.tt=(+e.orientacion_deg)||0; h.te=Math.max((+e.oao_oido)||0,(+e.voz_arousal||0)*0.6); h.cara=Math.sign((+e.cara_valoracion)||0);
        h.th+=(h.tt-h.th)*0.12; h.e+=(h.te-h.e)*0.18;
        const gen=e.cara_genero||e.genero||'masculino', ton=e.cara_tono||e.tono||'blanco';
        window.drawVSTCabeza3DReal(c,h.th,{energiaL:h.e,energiaR:h.e,cara:h.cara,genero:gen,tono:ton,cara_genero:gen,cara_tono:ton});
      }
      const set=(p,v)=>{const el=document.getElementById(p+'_'+L);if(el)el.textContent=v;};
      // ENTRADA POR OÍDO: nombre de la fuente REAL que el observador definió en la página del organismo
      // (NO fija) + barra de nivel (energia, escala sqrt: susurro ~0.1 y fuerte ~5 ambos visibles) + EMOJI
      // de reacción POR OÍDO (appraisal del organismo en esa oreja: comprensión vs riesgo del RC). Vivo y
      // sincronizado con la cabeza; sólo el emoji de la reacción actual. '·' = ese oído no recibe nada
      // (silencio o entrada apagada en la página del organismo) → no hay a qué reaccionar.
      const w=v=>Math.min(100,Math.sqrt(Math.max(0,v))*70);
      const emoR=r=>(r==null?'·':(r>0.1?'😊':(r<-0.1?'🙁':'😐')));
      const colR=r=>(r==null?'#33445e':(r>0.1?'#5fd38a':(r<-0.1?'#ff6b6b':'#6db6ff')));
      const oido=(side,ener,src,reac)=>{
        const s=document.getElementById('soc'+side+'src_'+L); if(s)s.textContent=vivo?((src&&(''+src))||'—'):'—';
        const b=document.getElementById('soc'+side+'bar_'+L); if(b){b.style.width=(vivo?w(ener):0)+'%';b.style.background=colR(reac);}
        const em=document.getElementById('soc'+side+'emo_'+L); if(em)em.textContent=vivo?emoR(reac):'·';
      };
      const rL=(e.reaccion_L==null?null:+e.reaccion_L), rR=(e.reaccion_R==null?null:+e.reaccion_R);
      oido('L',+e.energia_L||0,e.fuente_L,rL); oido('R',+e.energia_R||0,e.fuente_R,rR);
      // BOTÓN DE CORTE: refleja el estado REAL del organismo (mute_L/R, fuente única de verdad). Si lo
      // cortaste desde la página individual, el botón aparece cortado aquí en el próximo sondeo (bidireccional).
      const cutBtn=(side,muted)=>{const b=document.getElementById('soc'+side+'cut_'+L);if(b)b.classList.toggle('cut',!!muted);};
      cutBtn('L',e.mute_L); cutBtn('R',e.mute_R);
      set('socvoz', vivo?(emo(e.voz_emitida)+' '+(e.voz_titulo||e.voz_emitida||'—')):'· dormido ·');
      set('socimit',(+e.oao_imitacion_mag||0).toFixed(3));
      set('socoye',(+e.oao_oido||0).toFixed(2));
      set('socomega',(+e.Omega||0).toFixed(2));
      set('soccara',cara(+e.cara_valoracion||0));
    });
  }
  requestAnimationFrame(tick);
})();
// ===================== CIRCUITO VIVO DE LA DÍADA =====================
(function(){
  const NS='http://www.w3.org/2000/svg';
  const HEART='#b58cff', LIVE='#5fd38a', SIDECOL={A:'#6db6ff',B:'#ffd479'};
  const CAT={mundo:'sens',soma:'sens',hemi_rap:'nucleo',hemi_len:'nucleo',campo:'nucleo',metabolismo:'int',homeostasis:'int',memoria:'int',alteridad:'rel',expectativa:'rel',valoreco:'rel',oao:'rel',ove:'rel',expresion:'out',voz:'out'};
  const CATCOL={sens:'#6db6ff',nucleo:'#b58cff',int:'#5fd38a',rel:'#e8b86d',out:'#ff8c6b'};
  const ORG=[
    {id:'mundo',ic:'🌍',nom:'Mundo',row:0,hace:'Inyecta el sonido del entorno (audio guardado o Rode en vivo) por los oídos L/R.',para:'Fuente de perturbación: lo que el organismo oye y que alterará su campo.',sig:e=>({v:Math.min(1,(+e.energia||0)*3),t:''})},
    {id:'soma',ic:'👂',nom:'Soma',row:1,hace:'Capta la energía binaural (L/R) y la realimenta al campo Φ de la célula.',para:'La membrana sensible: traduce sonido en perturbación del campo.',sig:e=>({v:Math.min(1,(+e.oao_oido||0)*1.5),t:'oye '+(+e.oao_oido||0).toFixed(2)})},
    {id:'hemi_rap',ic:'⚡',nom:'Hemisferio',nom2:'Izquierdo',row:2,col:0,hace:'Hemisferio IZQUIERDO (rápido: τ corto, filtro temporal tipo highpass = transitorios). Recibe el oído DERECHO (cruce contralateral). Computa R₂ (predicción rápida).',para:'Procesa la NOVEDAD / lo temporal EN PARALELO, sin que la lateralidad lo anule.',sig:e=>({v:Math.min(1,Math.abs(+e.hemi_rapido_omega||0)),t:'R₂ '+(+e.hemi_R2||0).toFixed(3)})},
    {id:'campo',ic:'🌀',nom:'Campo Φ',row:2,heart:1,hace:'Núcleo dinámico: Ω (orden) y gradiente. Aquí se REÚNEN los hemisferios (vía cuerpo calloso) y se integra la perturbación.',para:'El cuerpo del organismo: su estado interno del que TODO lo demás depende.',sig:e=>({v:Math.min(1,+e.Omega||0),t:'Ω '+(+e.Omega||0).toFixed(2)})},
    {id:'hemi_len',ic:'🐢',nom:'Hemisferio',nom2:'Derecho',row:2,col:1,hace:'Hemisferio DERECHO (lento: τ largo, filtro temporal tipo lowpass = sostenido). Recibe el oído IZQUIERDO (cruce contralateral). Sostiene la lateralidad.',para:'Procesa lo ESPECTRAL / sostenido EN PARALELO. El cruce contralateral, como en el cerebro humano.',sig:e=>({v:Math.min(1,Math.abs(+e.hemi_lento_omega||0)),t:'lat '+(+e.hemi_lateralidad_func||0).toFixed(3)})},
    {id:'metabolismo',ic:'⚡',nom:'Metabolismo',row:3,col:0,hace:'Come la experiencia (nutritiva/tóxica) y paga el costo de vivir.',para:'Mantiene la energía: define hambre y saciedad.',sig:e=>({v:Math.min(1,+e.met_energia||0),t:''})},
    {id:'homeostasis',ic:'⚖️',nom:'Homeostasis',row:3,col:1,hace:'Vigila que las variables internas se mantengan en rango vital.',para:'Autorregulación: corrige desviaciones para no morir.',sig:e=>({v:Math.min(1,+e.H_homeostasis||0),t:(+e.en_rango>=.5?'✓ rango':'⚠ fuera')})},
    {id:'memoria',ic:'🧠',nom:'Memoria',row:3,col:2,hace:'Guarda episodios y reconoce lo familiar frente a lo nuevo.',para:'Continuidad: el pasado pesa en cómo interpreta el presente.',sig:e=>({v:Math.min(1,+e.mem_familiaridad||0),t:''})},
    {id:'alteridad',ic:'🗣',nom:'Alteridad',row:4,col:0,hace:'Detecta si el otro está presente y si su emisión lo mueve (agencia).',para:'Reconoce al otro como sujeto, no como ruido.',sig:e=>({v:Math.min(1,+e.alt_intencion_comunicativa||0),t:''})},
    {id:'expectativa',ic:'🔭',nom:'Expectativa',row:4,col:1,hace:'Evalúa si vale la pena seguir explorando tras la voz del otro.',para:'Primer eslabón de la genealogía del sentido.',sig:e=>({v:Math.min(1,(+e.expectativa||0)*8),t:''})},
    {id:'valoreco',ic:'🌱',nom:'Valor ecológico',row:4,col:2,hace:'Mide si la voz del otro ayuda a su persistencia.',para:'Convierte la señal social en valor para vivir.',sig:e=>({v:Math.min(1,(+e.voz_otro_valor_ecologico||0)*8),t:''})},
    {id:'oao',ic:'🧠',nom:'Aprendizaje·OAO',row:4,col:3,hace:'Memoria ecoica de lo oído → sesgo de imitación de la voz del par.',para:'Hace que su voz CONVERJA con la del otro (lenguaje compartido).',sig:e=>({v:Math.min(1,(+e.oao_imitacion_mag||0)*2),t:'imit '+(+e.oao_imitacion_mag||0).toFixed(2)})},
    {id:'expresion',ic:'🎙',nom:'Expresión',row:5,col:0,hace:'Decide SI vocalizar y con qué gesto (frecuencia/intensidad/pausa/repetición).',para:'La conducta vocal emergente: la salida del organismo.',sig:e=>({v:(+e.expr_vocalizando>=.5?1:0.15),t:(+e.expr_vocalizando>=.5?'habla':'calla')})},
    {id:'voz',ic:'🔊',nom:'Voz',row:6,hace:'Emite la vocalización (señal costosa) hacia el otro organismo.',para:'Cierra el lazo: su voz altera el Soma del otro.',sig:e=>({v:(+e.expr_vocalizando>=.5?1:0.1),t:(e.voz_titulo||e.voz_emitida||'—')})},
    {id:'ove',ic:'🙂',nom:'OVE · Cara',row:5,col:1,face:1,hace:'Valora la experiencia (favorable/neutra/desfavorable) y la EXPRESA en la cara de la cabeza 3D: la sonrisa, la boca recta o invertida.',para:'Es la lectura afectiva del organismo. NO cambia su conducta — es expresión pura, sin influencia causal (un espejo, no un timón).',sig:e=>{var c=+e.cara_valoracion||0;return {v:Math.min(1,Math.abs(c)*1.6+0.18),t:(c>0.05?'😊 favorable':(c<-0.05?'☹️ desfavorable':'😐 neutra')),face:(c>0.05?'😊':(c<-0.05?'☹️':'😐'))};}}
  ];
  const FLOW=[['mundo','soma'],['soma','hemi_rap'],['soma','hemi_len'],['hemi_rap','campo'],['hemi_len','campo'],
    ['campo','metabolismo'],['campo','homeostasis'],['campo','memoria'],
    ['campo','alteridad'],['campo','expectativa'],['campo','valoreco'],['campo','oao'],
    ['metabolismo','expresion'],['homeostasis','expresion'],['memoria','expresion'],
    ['alteridad','expresion'],['expectativa','expresion'],['valoreco','expresion'],['oao','expresion'],
    ['expresion','voz']];
  const READ=[['campo','ove']];   // lectura read-only (sin causalidad): el campo se valora y se expresa en la cara
  const ORDER=['mundo','soma','hemi_rap','hemi_len','campo','metabolismo','homeostasis','memoria','alteridad','expectativa','valoreco','oao','ove','expresion','voz'];
  const baseX={A:320,B:960}, rowY={0:52,1:124,2:218,3:336,4:456,5:566,6:642};
  const off2=[-150,150], off3=[-150,0,150], off4=[-198,-66,66,198], off5=[-96,96];
  function pos(side,o){let x=baseX[side],y=rowY[o.row];if(o.row===2&&o.col!==undefined)x+=off2[o.col];if(o.row===3)x+=off3[o.col];if(o.row===4)x+=off4[o.col];if(o.row===5)x+=off5[o.col];return {x:x,y:y};}

  let built=false; const N={}, E=[];
  function mk(tag,attrs){const el=document.createElementNS(NS,tag);for(const k in attrs)el.setAttribute(k,attrs[k]);return el;}
  function path(a,b){const dx=b.x-a.x,dy=b.y-a.y;const c1y=a.y+dy*0.45,c2y=b.y-dy*0.45;return `M ${a.x} ${a.y} C ${a.x+dx*0.1} ${c1y}, ${b.x-dx*0.1} ${c2y}, ${b.x} ${b.y}`;}
  function bridge(a,b){return `M ${a.x} ${a.y} C ${(a.x+b.x)/2} ${a.y+90}, ${(a.x+b.x)/2} ${b.y-90}, ${b.x} ${b.y}`;}

  function grad(id,stops,cx,cy,r){const g=mk('radialGradient',{id:id,cx:cx,cy:cy,r:r});stops.forEach(s=>g.appendChild(mk('stop',{offset:s[0],'stop-color':s[1],'stop-opacity':s[2]})));return g;}
  function build(){
    const svg=document.getElementById('csvg'); if(!svg||built)return; built=true;
    const defs=mk('defs',{});
    defs.appendChild(grad('cytoA',[['0%','#244a7e','.40'],['70%','#16294a','.26'],['100%','#0f1d35','.05']],'50%','42%','62%'));
    defs.appendChild(grad('cytoB',[['0%','#5c4a1e','.34'],['70%','#3a2f14','.22'],['100%','#2a230f','.05']],'50%','42%','62%'));
    defs.appendChild(grad('nucleo',[['0%','#b58cff','.55'],['100%','#b58cff','0']],'50%','50%','50%'));
    svg.appendChild(defs);
    // MEMBRANA (look de célula madre) + halo del NÚCLEO (Campo Φ)
    const memb={A:{x:320,fill:'url(#cytoA)'},B:{x:960,fill:'url(#cytoB)'}};
    for(const side of ['A','B']){const m=memb[side];
      // membrana: tope en y~88 → el Mundo (row0,y52) queda FUERA/arriba; Soma(y124)..Voz(y642) dentro
      svg.appendChild(mk('ellipse',{class:'membrana',cx:m.x,cy:396,rx:272,ry:308,fill:m.fill,stroke:SIDECOL[side],'stroke-width':2.2}));
      svg.appendChild(mk('circle',{class:'nucleo',cx:m.x,cy:rowY[2],r:98,fill:'url(#nucleo)'}));
    }
    svg.appendChild(mk('text',{x:320,y:24,class:'clabel'})).textContent='🔵 Organismo A · célula';
    svg.appendChild(mk('text',{x:960,y:24,class:'clabel'})).textContent='🟡 Organismo B · célula';
    for(const side of ['A','B']){N[side]={};ORG.forEach(o=>{N[side][o.id]=Object.assign({},o,{side:side},pos(side,o));});}
    // edges del flujo (mundo→soma = ESTÍMULO, engrosado y etiquetado)
    for(const side of ['A','B']) FLOW.forEach(([a,b])=>{
      const na=N[side][a],nb=N[side][b];const stim=(a==='mundo'&&b==='soma');
      const p=mk('path',{class:'cedge'+(stim?' stim':''),d:path(na,nb)});svg.appendChild(p);E.push({p:p,src:na,stim:stim});
      if(stim){const t=mk('text',{x:na.x+26,y:(na.y+nb.y)/2+3,class:'stimlab'});t.textContent='estímulo';svg.appendChild(t);}
    });
    // edges de LECTURA (read-only, sin causalidad): campo → OVE/cara
    for(const side of ['A','B']) READ.forEach(([a,b])=>{const na=N[side][a],nb=N[side][b];
      const p=mk('path',{class:'cedge read',d:path(na,nb)});svg.appendChild(p);E.push({p:p,src:na,read:1});});
    // puentes de comunicación (acople): voz A → soma B, voz B → soma A
    [['A','B'],['B','A']].forEach(([s,o])=>{const na=N[s]['voz'],nb=N[o]['soma'];
      const p=mk('path',{class:'cedge bridge',d:bridge(na,nb)});svg.appendChild(p);E.push({p:p,src:na,bridge:1});});
    // CUERPO CALLOSO: arco que une los dos hemisferios sobre el Campo Φ; pulsa al REUNIRSE (hemi_calloso_activo)
    for(const side of ['A','B']){const na=N[side]['hemi_rap'],nb=N[side]['hemi_len'];
      const dd=`M ${na.x+60} ${na.y} Q ${(na.x+nb.x)/2} ${na.y-46} ${nb.x-60} ${nb.y}`;
      const p=mk('path',{class:'cedge calloso',stroke:'#b58cff','stroke-dasharray':'5 4',fill:'none',d:dd});svg.appendChild(p);E.push({p:p,src:na,calloso:1});
      const lx=(na.x+nb.x)/2;const tl=mk('text',{x:lx,y:na.y-52,class:'stimlab','text-anchor':'middle',fill:'#c4a4ff'});tl.textContent='cuerpo calloso';svg.appendChild(tl);}
    // nodos
    for(const side of ['A','B']) ORG.forEach(o=>{
      const n=N[side][o.id];const w=o.heart?156:120,h=o.heart?64:54;const col=o.heart?HEART:SIDECOL[side];
      const g=mk('g',{class:'cnode'});g.dataset.side=side;g.dataset.id=o.id;
      const glow=mk('rect',{class:'glow',x:n.x-w/2-3,y:n.y-h/2-3,width:w+6,height:h+6,rx:14,stroke:LIVE,'stroke-width':3});
      const body=mk('rect',{class:'body',x:n.x-w/2,y:n.y-h/2,width:w,height:h,rx:12,fill:'#172339',stroke:col,'stroke-width':1.6});
      const dot=mk('circle',{class:'catdot',cx:n.x-w/2+13,cy:n.y-h/2+13,r:5,fill:CATCOL[CAT[o.id]]||'#8aa0b8'});
      const ic=mk('text',{class:'ic',x:n.x,y:n.y-h/2+21});ic.textContent=o.ic;
      const nm=mk('text',{class:'nm',x:n.x,y:o.nom2?n.y-5:n.y+2});nm.textContent=o.nom;
      if(o.nom2){const t2=mk('tspan',{x:n.x,dy:14});t2.textContent=o.nom2;nm.appendChild(t2);}
      const vl=mk('text',{class:'vl',x:n.x,y:n.y+h/2-7});
      g.appendChild(glow);g.appendChild(body);g.appendChild(dot);g.appendChild(ic);g.appendChild(nm);g.appendChild(vl);
      g._glow=glow;g._body=body;g._vl=vl;g._ic=ic;g._col=col;
      g.addEventListener('mouseenter',()=>showInfo(side,o));
      g.addEventListener('click',()=>showInfo(side,o));
      svg.appendChild(g);n._g=g;
    });
    // leyenda de categorías
    const leg=[['sens','sensorial'],['nucleo','núcleo'],['int','interno'],['rel','relacional'],['out','salida']];
    let lx=455;leg.forEach(([k,lab])=>{svg.appendChild(mk('circle',{cx:lx,cy:716,r:5,fill:CATCOL[k]}));const t=mk('text',{x:lx+10,y:720,class:'leglab'});t.textContent=lab;svg.appendChild(t);lx+=lab.length*7+40;});
  }
  function showInfo(side,o){
    const e=(window._ultD&&window._ultD[side])||{};const s=o.sig(e);
    const ci=document.getElementById('cinfo');if(!ci)return;
    ci.innerHTML=`<div class=sub>Organismo ${side} · organelo</div><h4>${o.ic} ${o.nom}${o.nom2?' '+o.nom2:''}</h4>`
      +`<div class=q><b>Qué hace:</b> ${o.hace}</div><div class=q><b>Para qué sirve:</b> ${o.para}</div>`
      +`<div class=sub>valor actual</div><div class=vbig>${s.t||(s.v*100).toFixed(0)+'%'}</div>`;
  }
  let _last=0, _lexCircN=-1;
  function fmt(g,v,txt){
    g._glow.setAttribute('opacity',(0.12+0.88*Math.max(0,Math.min(1,v))).toFixed(2));
    g._body.setAttribute('stroke-width',(1.4+2.4*v).toFixed(2));
    if(txt!==undefined)g._vl.textContent=txt;
  }
  function tickCirc(ts){
    requestAnimationFrame(tickCirc);
    const cir=document.getElementById('circuito');
    if(!built||!cir||cir.style.display==='none'||!window._ultD)return;
    const upd=(ts-_last)>220; if(upd)_last=ts;
    if(window._ultLex){if(_lexCircN<0)_lexCircN=window._ultLex.length;
      else if(window._ultLex.length>_lexCircN){const x=window._ultLex[window._ultLex.length-1];_lexCircN=window._ultLex.length;
        flash(x.lado,'voz',1500);flash(x.lado,x.tipo==='emulada'?'oao':'expresion',1500);}}
    for(const side of ['A','B']){const e=window._ultD[side]||{};
      ORG.forEach(o=>{const n=N[side][o.id];if(!n||!n._g)return;const s=o.sig(e);
        fmt(n._g,s.v,upd?s.t:undefined);
        if(o.face&&s.face&&n._g._ic)n._g._ic.textContent=s.face;});}   // la cara del OVE cambia en vivo
    E.forEach(ed=>{const e=window._ultD[ed.src.side]||{};
      if(ed.calloso){const on=+e.hemi_calloso_activo||0,dv=Math.min(1,(+e.hemi_divergencia||0)*3);
        ed.p.style.opacity=(0.5+0.45*on).toFixed(2);ed.p.style.strokeWidth=(2.2+2.8*dv).toFixed(2);return;}
      const s=ed.src.sig(e);
      const v=Math.max(0.05,Math.min(1,s.v));
      ed.p.style.opacity=((ed.read?0.5:0.22)+(ed.read?0.3:0.78)*v).toFixed(2);
      const base=ed.bridge?2.4:(ed.stim?3.4:(ed.read?1.1:1.6));
      ed.p.style.strokeWidth=(base+(ed.read?0.4:2.2)*v).toFixed(2);});
  }
  // ✴ Trazar el campo: enciende los organelos en orden de flujo (A → puente → B)
  function flash(side,id,ms){const n=N[side]&&N[side][id];if(!n||!n._g)return;n._g.classList.add('flash');setTimeout(()=>n._g.classList.remove('flash'),ms||650);}
  function trace(){let t=0;const step=230;
    ORDER.forEach(id=>{setTimeout(()=>flash('A',id,700),t);t+=step;});
    t+=step; ORDER.forEach(id=>{setTimeout(()=>flash('B',id,700),t);t+=step;});
  }
  window.initCircuito=function(){build();const bt=document.getElementById('cTrace');if(bt&&!bt._w){bt._w=1;bt.addEventListener('click',trace);}};
  requestAnimationFrame(tickCirc);
})();

// ---- REPRODUCTOR EN VIVO: A por la IZQUIERDA, B por la DERECHA (como se ven) ----
let ac=null, audioOn=false, gMaster=null, lados={};
function audioToggle(){
  if(audioOn){ audioOn=false; $('bAudio').textContent='🔊 Escuchar conversación'; $('bAudio').classList.remove('on'); return; }
  ac=ac||new (window.AudioContext||window.webkitAudioContext)(); ac.resume();
  if(!gMaster){
    gMaster=ac.createGain(); gMaster.gain.value=+($('vol').value||4); gMaster.connect(ac.destination);
    for(const [lado,panv] of [['A',-0.85],['B',0.85]]){
      const g=ac.createGain();
      if(ac.createStereoPanner){ const p=ac.createStereoPanner(); p.pan.value=panv; g.connect(p); p.connect(gMaster); }
      else g.connect(gMaster);
      lados[lado]={g:g, next:0};
    }
  }
  audioOn=true; $('bAudio').textContent='⏸ Detener audio'; $('bAudio').classList.add('on');
  for(const l of ['A','B']){ lados[l].next=ac.currentTime; reproducir(l); }
}
function setVol(v){ $('volV').textContent=(+v).toFixed(1)+'×'; if(gMaster) gMaster.gain.value=+v; }
async function reproducir(lado){
  if(!audioOn) return;
  let ahead=0.9;
  try{
    const ab=await ac.decodeAudioData((await fetch('/voz/'+lado).then(r=>r.arrayBuffer())).slice(0));
    const s=ac.createBufferSource(); s.buffer=ab; s.connect(lados[lado].g);
    const t=Math.max(ac.currentTime+0.02, lados[lado].next); s.start(t); lados[lado].next=t+ab.duration;
    ahead=Math.max(0.1, lados[lado].next-ac.currentTime-0.12);
  }catch(e){}
  setTimeout(()=>reproducir(lado), ahead*1000);
}

// ---- NAVEGADOR HISTÓRICO (lee el registro permanente del Docker) ----
let histInit=false;
async function initHist(){
  if(!histInit){ histInit=true;
    try{
      const dias=await fetch('/dias').then(r=>r.json());
      $('selDia').innerHTML=(dias.length?dias:['(sin registro)']).map(d=>'<option>'+d+'</option>').join('');
      if(dias.length) $('selDia').value=dias[dias.length-1];
      const st=await fetch('/stats').then(r=>r.json());
      const voces=new Set(); Object.values(st.hist||{}).forEach(h=>Object.keys(h).forEach(v=>voces.add(v)));
      $('selVoz').innerHTML='<option value="">todos</option>'+[...voces].sort().map(v=>'<option>'+v+'</option>').join('');
      renderStat(st);
    }catch(e){}
  }
  cargarHist();
}
function renderStat(st){
  const C={screaming:'#ff5b5b',shout:'#ff8c6b',worried:'#e8b86d',excited:'#5fd38a','excited-2':'#5fd38a',sing:'#8ef0c0',acknowledged:'#6db6ff',chat:'#9fb1c6'};
  let h='<div class=mut style=font-size:10px>'+(st.total_turnos||0)+' turnos · '+((st.dias||[]).length)+' día(s)</div>';
  for(const [org,hh] of Object.entries(st.hist||{})){
    const tot=Object.values(hh).reduce((a,b)=>a+b,0)||1;
    h+='<div class=mut style=margin-top:6px>'+org+'</div>'+Object.entries(hh).sort((a,b)=>b[1]-a[1]).map(([v,n])=>
      '<div class=barra><span style=width:92px>'+emo(v)+' '+v+'</span><div class=b><div style="width:'+(100*n/tot)+'%;background:'+(C[v]||'#5fd38a')+'"></div></div><span style="width:34px;text-align:right" class=mut>'+n+'</span></div>').join('');
  }
  $('statH').innerHTML=h;
}
async function cargarHist(){
  const dia=$('selDia').value, voz=$('selVoz').value;
  try{
    const ts=await fetch('/turnos?dia='+encodeURIComponent(dia)+'&voz='+encodeURIComponent(voz)+'&limite=600').then(r=>r.json());
    $('histInfo').textContent=ts.length+' turnos'+(voz?(' de "'+voz+'"'):'');
    $('transH').innerHTML=ts.map(t=>{
      const hh=new Date((t.ts||0)*1000).toLocaleTimeString();
      const em=(''+(t.emisor||''));
      let lado='A';
      if(/[Bb]/.test(em)&&!/[Aa]/.test(em)) lado='B';
      else if(em.indexOf('B')>=0) lado='B';
      const lab=t.prototipo_codebook||t.voz||t.titulo||'';
      const btn=lab&&lab!=='-'?('<button type=button class=playw data-lado="'+lado+'" data-label="'+encodeURIComponent(lab)+'" title="escuchar «'+lab+'»">▶</button>'):'';
      const dOI=((t.delta_receptor||{}).delta_OI);
      return '<div class=linea>'+btn+'<span class=mut>'+hh+'</span> <span class='+lado+'>'+(t.emisor||'?')+'→'+(t.receptor||'?')+'</span> '+emo(t.prototipo_codebook)+' <b>'+t.prototipo_codebook+'</b> <span class=mut>ΔOI_rec '+(dOI==null?'·':dOI)+'</span></div>';
    }).join('');
    $('transH').scrollTop=$('transH').scrollHeight;
  }catch(e){}
}
// ---- REPRODUCTOR POR PALABRA (▶ en transcript En vivo / Historia) ----
let _pwAc=null, _pwSrc=null;
async function playPalabra(lado, label, btn){
  if(!label||label==='-'||!lado) return;
  try{
    if(_pwSrc){ try{_pwSrc.stop();}catch(_){ } _pwSrc=null; }
    document.querySelectorAll('.playw.playing').forEach(b=>{b.classList.remove('playing'); b.textContent='▶';});
    _pwAc=_pwAc||new (window.AudioContext||window.webkitAudioContext)();
    await _pwAc.resume();
    if(btn){ btn.classList.add('playing'); btn.textContent='…'; }
    const r=await fetch('/voz/'+lado+'?label='+encodeURIComponent(label));
    if(!r.ok) throw new Error('palabra no disponible');
    const ab=await r.arrayBuffer();
    const buf=await _pwAc.decodeAudioData(ab.slice(0));
    const s=_pwAc.createBufferSource(); s.buffer=buf;
    const g=_pwAc.createGain();
    g.gain.value=+(($('vol')&&$('vol').value)||4);
    s.connect(g); g.connect(_pwAc.destination);
    _pwSrc=s;
    if(btn) btn.textContent='■';
    s.onended=()=>{ if(btn){ btn.classList.remove('playing'); btn.textContent='▶'; } if(_pwSrc===s) _pwSrc=null; };
    s.start();
  }catch(e){
    if(btn){ btn.classList.remove('playing'); btn.textContent='▶'; btn.title='no se pudo reproducir (¿aún en repertorio?)'; }
  }
}
function _bindPlayw(el){
  if(!el||el._playw) return; el._playw=1;
  el.addEventListener('click', e=>{
    const b=e.target.closest&&e.target.closest('.playw'); if(!b) return;
    e.preventDefault();
    const lab=decodeURIComponent(b.getAttribute('data-label')||'');
    const lado=b.getAttribute('data-lado')||'A';
    if(b.classList.contains('playing')&&_pwSrc){
      try{_pwSrc.stop();}catch(_){} _pwSrc=null;
      b.classList.remove('playing'); b.textContent='▶'; return;
    }
    playPalabra(lado, lab, b);
  });
}
if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',()=>{_bindPlayw($('trans'));_bindPlayw($('transH'));});
else {_bindPlayw($('trans'));_bindPlayw($('transH'));}

setInterval(tick, 500); tick();

// ============================ OBSERVATORIO DE LA DÍADA (cajas editables) ============================
// Misma lógica que las páginas de organismo: cajas declarativas que el observador agrega/mueve.
// Disposición en localStorage (membrana, no cerebro). Datos = último /datos (d.A, d.B).
(function(){
  const cjN=(x,d=2)=>{x=Number(x);return isFinite(x)?x.toFixed(d):'—';};
  const cjRow=(k,v)=>`<div class="obsrow"><span class="obsk">${k}</span><span class="obsv">${v}</span></div>`;
  const cjGauge=(v,col='#5fd38a')=>{v=Math.max(0,Math.min(1,Number(v)||0));return `<div class="obsgauge"><i style="width:${(v*100).toFixed(0)}%;background:${col}"></i></div>`;};
  const cjBip=(v,col='#6db6ff')=>{v=Math.max(-1,Math.min(1,Number(v)||0));const w=Math.abs(v)*50,left=v>=0?50:50-w;return `<div class="obsgauge"><i style="margin-left:${left}%;width:${w}%;background:${col}"></i></div>`;};
  const G=(d,L)=>{const x=(d&&d[L])||{};return {f:+x.g_freq||0,i:+x.g_intensidad||0,p:+x.g_pausa||0,r:+x.g_repeticion||0,b:x.g_bucket||'—',int:+x.alt_intencion_comunicativa||0,eo:+x.alt_efecto_sobre_otro||0,oi:+x.OI||0,aro:+x.voz_arousal||0,val:+x.voz_valence||0,voz:x.voz_emitida||'—'};};

  const CAJAS=[
   {id:'libertad_creativa',tit:'🎙 Libertad expresiva (balbuceo)',w:6,h:5,render:(b,d)=>{
     const A=G(d,'A'),B=G(d,'B');
     const dist=Math.sqrt((A.f-B.f)**2+(A.i-B.i)**2+(A.p-B.p)**2+(A.r-B.r)**2), conv=Math.max(0,1-dist/2.0);
     const col=(lab,g)=>`<div style="flex:1;min-width:120px"><div class="obsk" style="margin-bottom:3px">${lab} · <b style="color:#cfe0f5">${g.b}</b></div>`
       +`<span class="obsk">frecuencia</span>`+cjBip(g.f,'#e8b86d')+`<span class="obsk">intensidad</span>`+cjBip(g.i,'#6db6ff')
       +`<span class="obsk">pausa</span>`+cjGauge(g.p,'#b58cff')+`<span class="obsk">repetición</span>`+cjGauge(g.r,'#ff8c6b')
       +`<div class="obsk" style="margin-top:4px">intención ${g.int.toFixed(3)}</div>`+cjGauge(g.int,'#64f0c8')+`</div>`;
     b.innerHTML=`<div style="display:flex;gap:12px;flex-wrap:wrap">${col('🔵 A',A)}${col('🟡 B',B)}</div>`
       +`<div style="margin-top:8px;border-top:1px solid #1d2940;padding-top:7px">`
       +cjRow('convergencia de gesto',(conv*100).toFixed(0)+'%')+cjGauge(conv,'#8ef0c0')
       +`<div class="obsk" style="font-size:9.5px;margin-top:3px">${conv>0.72?'⚠ gestos MUY similares — ¿convención emergente? (confirmar con control NULL/SHUFFLED en vivo)':'exploran su espacio expresivo con libertad (sin convención fijada)'}</div></div>`;}},
   {id:'intencion',tit:'🗣 Alteridad / Intención (A/B)',w:3,h:3,render:(b,d)=>{const A=G(d,'A'),B=G(d,'B');b.innerHTML=
     cjRow('A intención',cjN(A.int,3))+cjGauge(A.int,'#6db6ff')+cjRow('A efecto→otro',cjN(A.eo,3))+cjGauge(A.eo,'#6db6ff')
    +cjRow('B intención',cjN(B.int,3))+cjGauge(B.int,'#ffd479')+cjRow('B efecto→otro',cjN(B.eo,3))+cjGauge(B.eo,'#ffd479');}},
   {id:'afecto',tit:'🔊 Voz / Comunicación (A/B)',w:3,h:3,render:(b,d)=>{const A=G(d,'A'),B=G(d,'B');b.innerHTML=
     cjRow('A',A.voz)+`<span class="obsk">arousal</span>`+cjGauge(A.aro,'#ff8c6b')+`<span class="obsk">valencia</span>`+cjBip(A.val,'#6db6ff')
    +cjRow('B',B.voz)+`<span class="obsk">arousal</span>`+cjGauge(B.aro,'#ff8c6b')+`<span class="obsk">valencia</span>`+cjBip(B.val,'#ffd479');}},
   {id:'acople',tit:'❤️ Acople (OI A↔B)',w:3,h:2,render:(b,d)=>{const A=G(d,'A'),B=G(d,'B');b.innerHTML=
     cjRow('OI · A',cjN(A.oi,3))+cjGauge(A.oi,'#6db6ff')+cjRow('OI · B',cjN(B.oi,3))+cjGauge(B.oi,'#ffd479')
    +cjRow('|diferencia|',cjN(Math.abs(A.oi-B.oi),3));}},
   {id:'agencia',tit:'🧭 Alteridad / Agencia (A/B)',w:3,h:3,render:(b,d)=>{const q=(L,k)=>(+(((d&&d[L])||{})[k])||0);b.innerHTML=
     cjRow('A contingencia',cjN(q('A','alt_contingencia_social'),3))+cjGauge(Math.min(1,q('A','alt_contingencia_social')*8),'#64f0c8')
    +cjRow('A agencia',cjN(q('A','alt_agencia_otro'),3))+cjGauge(q('A','alt_agencia_otro'),'#6db6ff')
    +cjRow('B contingencia',cjN(q('B','alt_contingencia_social'),3))+cjGauge(Math.min(1,q('B','alt_contingencia_social')*8),'#64f0c8')
    +cjRow('B agencia',cjN(q('B','alt_agencia_otro'),3))+cjGauge(q('B','alt_agencia_otro'),'#ffd479')
    +`<div class="obsk" style="font-size:9px;margin-top:3px">¿la emisión mueve al otro sobre su basal? (≈0 hoy)</div>`;}},
   {id:'vozeco',tit:'🌱 Valor ecológico de la voz (A/B)',w:3,h:3,render:(b,d)=>{const q=(L,k)=>(+(((d&&d[L])||{})[k])||0);b.innerHTML=
     cjRow('A valor ecológico',cjN(q('A','voz_otro_valor_ecologico'),3))+cjGauge(Math.min(1,q('A','voz_otro_valor_ecologico')*8),'#8ef0c0')
    +cjRow('A confianza',cjN(q('A','voz_otro_confianza_ecologica'),3))+cjGauge(q('A','voz_otro_confianza_ecologica'),'#e8b86d')
    +cjRow('B valor ecológico',cjN(q('B','voz_otro_valor_ecologico'),3))+cjGauge(Math.min(1,q('B','voz_otro_valor_ecologico')*8),'#8ef0c0')
    +cjRow('B confianza',cjN(q('B','voz_otro_confianza_ecologica'),3))+cjGauge(q('B','voz_otro_confianza_ecologica'),'#ffd479')
    +`<div class="obsk" style="font-size:9px;margin-top:3px">¿la voz del otro ayuda a persistir? cae bajo NULL/SHUFFLED</div>`;}},
   {id:'expectativa',tit:'🔭 Expectativa (A/B)',w:3,h:3,render:(b,d)=>{const q=(L,k)=>(+(((d&&d[L])||{})[k])||0);b.innerHTML=
     cjRow('A expectativa',cjN(q('A','expectativa'),3))+cjGauge(Math.min(1,q('A','expectativa')*8),'#b58cff')
    +cjRow('A explora · confianza',cjN(q('A','expectativa_exploracion'),3)+' · '+cjN(q('A','expectativa_confianza'),2))
    +cjRow('B expectativa',cjN(q('B','expectativa'),3))+cjGauge(Math.min(1,q('B','expectativa')*8),'#b58cff')
    +cjRow('B explora · confianza',cjN(q('B','expectativa_exploracion'),3)+' · '+cjN(q('B','expectativa_confianza'),2))
    +`<div class="obsk" style="font-size:9px;margin-top:3px">¿vale la pena explorar tras la voz? (1er eslabón de la genealogía)</div>`;}},
   {id:'expresion',tit:'🎙 Expresión vocal (conducta) (A/B)',w:3,h:3,render:(b,d)=>{const q=(L,k)=>(+(((d&&d[L])||{})[k])||0);
     const est=(L)=>(q(L,'expr_vocalizando')>=.5?'🗣 habla':(q(L,'expr_silencio')>=.5?'🤫 calla':'·'));
     window._exprT=window._exprT||{A:{h:0,n:0},B:{h:0,n:0}};
     for(const L of ['A','B']){if(q(L,'expr_vocalizando')>=.5)window._exprT[L].h++;else if(q(L,'expr_silencio')>=.5)window._exprT[L].n++;}
     const med=(L)=>{const t=window._exprT[L],tot=(t.h+t.n)||1,fh=t.h/tot;
       return cjRow(L+' · '+est(L), '🗣 '+(fh*100).toFixed(0)+'% · '+((1-fh)*100).toFixed(0)+'% 🤫')
         +`<div class="obsgauge" style="display:flex"><i style="width:${(fh*100).toFixed(1)}%;background:#5fd38a"></i><i style="width:${((1-fh)*100).toFixed(1)}%;background:#8aa0b8"></i></div>`;};
     b.innerHTML = med('A') + med('B')
    +`<div class="obsk" style="font-size:9px;margin-top:3px">el 1er acto es decidir SI hablar; el silencio es una conducta (acumulado de la sesión)</div>`;}},
   {id:'imitacion',tit:'🧠 Aprendizaje (OAO: ecoica + imitación) (A/B)',w:3,h:3,render:(b,d)=>{const q=(L,k)=>(+(((d&&d[L])||{})[k])||0);b.innerHTML=
     cjRow('A oye · ecoica',cjN(q('A','oao_oido'),3)+' · '+cjN(q('A','oao_echoica_n'),0))
    +cjRow('A imitación',cjN(q('A','oao_imitacion_mag'),3))+cjGauge(Math.min(1,q('A','oao_imitacion_mag')*2),'#8ef0c0')
    +cjRow('B oye · ecoica',cjN(q('B','oao_oido'),3)+' · '+cjN(q('B','oao_echoica_n'),0))
    +cjRow('B imitación',cjN(q('B','oao_imitacion_mag'),3))+cjGauge(Math.min(1,q('B','oao_imitacion_mag')*2),'#8ef0c0')
    +`<div class="obsk" style="font-size:9px;margin-top:3px">lo oído sesga la voz futura por historia (imitación, no copia)</div>`;}},
   {id:'lexico',tit:'🔤 Léxico: palabras propias ↔ compartidas',w:6,h:5,render:(b,d)=>{const q=(L,k)=>(+(((d&&d[L])||{})[k])||0);
     const cnt=(L,em)=>`<div style="flex:1;min-width:150px"><div class="obsk" style="margin-bottom:3px">${em} <b style="color:#cfe0f5">${(d&&d[L]&&d[L].organismo)||L}</b></div>`
       +cjRow('✨ inventó',q(L,'voz_creadas'))+cjRow('↔ emuló del otro',q(L,'voz_aprendidas'))
       +cjRow('estables · activas',q(L,'voz_estables')+' · '+q(L,'voz_propias'))+`</div>`;
     const comp=q('A','voz_aprendidas')+q('B','voz_aprendidas');
     const feed=(window._ultLex||[]);
     const ev=feed.slice(-6).reverse().map(x=>{const hh=new Date((x.ts||0)*1000).toLocaleTimeString();
       const cre=x.tipo==='creada';const ic=cre?'🗣️✨':'🗣️↔';const col=cre?'#e8b86d':'#8ef0c0';
       const txt=cre?(x.org+' INVENTÓ una palabra (#'+x.n+')'):(x.org+' EMULÓ la del otro → compartida (#'+x.n+')');
       return `<div style="font-size:10px;margin:2px 0;color:${col}"><span class="obsk">${hh}</span> ${ic} ${txt}</div>`;}).join('')
       ||'<div class="obsk" style="font-size:10px;opacity:.6">aún sin eventos (esperando que inventen o emulen)…</div>';
     b.innerHTML=`<div style="display:flex;gap:12px;flex-wrap:wrap">${cnt('A','🔵')}${cnt('B','🟡')}</div>`
       +`<div style="margin-top:7px;border-top:1px solid #2e4163;padding-top:6px">`
       +cjRow('🌐 palabras compartidas (emuladas A+B)',comp)+cjGauge(Math.min(1,comp/4),'#8ef0c0')
       +`<div class="obsk" style="font-size:9px;margin:6px 0 2px">eventos recientes:</div>${ev}</div>`;}},
   {id:'metabolismo',tit:'⚡ Metabolismo (A/B)',w:3,h:3,render:(b,d)=>{const q=(L,k)=>(+(((d&&d[L])||{})[k])||0);
     const f=(L,c)=>cjRow(L+' energía',cjN(q(L,'met_energia'),3))+cjGauge(q(L,'met_energia'),'#56d6a0')
       +cjRow(L+' hambre · saciedad',cjN(q(L,'met_hambre'),2)+' · '+cjN(q(L,'met_saciedad'),2))+cjGauge(q(L,'met_hambre'),'#ff8c6b')
       +cjRow(L+' balance',cjN(q(L,'met_balance'),3))+cjBip(Math.max(-1,Math.min(1,q(L,'met_balance')*20)),'#f0c47b');
     b.innerHTML=f('A','#6db6ff')+'<div style="height:6px;border-top:1px solid #2e4163;margin-top:4px"></div>'+f('B','#ffd479');}},
   {id:'memoria',tit:'🧠 Memoria (A/B)',w:3,h:3,render:(b,d)=>{const q=(L,k)=>(+(((d&&d[L])||{})[k])||0);
     const f=(L)=>cjRow(L+' episodios',cjN(q(L,'mem_episodios_n'),0))
       +cjRow(L+' familiaridad',cjN(q(L,'mem_familiaridad'),2))+cjGauge(q(L,'mem_familiaridad'),'#6db6ff')
       +cjRow(L+' novedad',cjN(q(L,'mem_novedad'),2))+cjGauge(q(L,'mem_novedad'),'#ff8c6b')
       +cjRow(L+' recall · conf. relacional',cjN(q(L,'mem_recall'),2)+' · '+cjN(q(L,'mem_relacional_confianza'),2));
     b.innerHTML=f('A')+'<div style="height:6px;border-top:1px solid #2e4163;margin-top:4px"></div>'+f('B');}},
   {id:'salud',tit:'❤️ Salud del cierre (A/B)',w:3,h:3,render:(b,d)=>{const q=(L,k)=>(+(((d&&d[L])||{})[k])||0);
     const f=(L)=>cjRow(L+' invariantes ok',cjN(q(L,'invariantes_ok'),0))
       +cjRow(L+' autonomía A_sys/env',cjN(q(L,'A_sys_env'),3))+cjGauge(Math.min(1,q(L,'A_sys_env')),'#56d6a0')
       +cjRow(L+' Λ_Cos',cjN(q(L,'Lambda_Cos'),4))
       +cjRow(L+' OI (acople)',cjN(q(L,'OI'),3))+cjGauge(q(L,'OI'),'#ff6b6b');
     b.innerHTML=f('A')+'<div style="height:6px;border-top:1px solid #2e4163;margin-top:4px"></div>'+f('B')
       +`<div class="obsk" style="font-size:9px;margin-top:3px">¿el organismo sostiene su clausura operacional? (invariantes vivos)</div>`;}},
   {id:'campo',tit:'🌀 Campo Φ / Soma (A/B)',w:3,h:3,render:(b,d)=>{const q=(L,k)=>(+(((d&&d[L])||{})[k])||0);
     const f=(L)=>cjRow(L+' Ω (orden)',cjN(q(L,'Omega'),3))+cjGauge(q(L,'Omega'),'#b58cff')
       +cjRow(L+' gradiente',cjN(q(L,'gradiente'),3))+cjBip(Math.max(-1,Math.min(1,q(L,'gradiente'))),'#64f0c8')
       +cjRow(L+' ωA  L · R',cjN(q(L,'omega_A_L'),2)+' · '+cjN(q(L,'omega_A_R'),2));
     b.innerHTML=f('A')+'<div style="height:6px;border-top:1px solid #2e4163;margin-top:4px"></div>'+f('B');}},
   {id:'homeostasis',tit:'⚖️ Homeostasis (A/B)',w:3,h:3,render:(b,d)=>{const q=(L,k)=>(+(((d&&d[L])||{})[k])||0);
     const enr=(L)=>(q(L,'en_rango')>=.5?'✅ en rango':'⚠ fuera de rango');
     const f=(L)=>cjRow(L+' '+enr(L),cjN(q(L,'H_homeostasis'),3))+cjGauge(q(L,'H_homeostasis'),'#56d6a0')
       +cjRow(L+' x interna',cjN(q(L,'x_interna'),3))+cjGauge(Math.min(1,Math.abs(q(L,'x_interna'))),'#f0c47b');
     b.innerHTML=f('A')+'<div style="height:6px;border-top:1px solid #2e4163;margin-top:4px"></div>'+f('B');}},
  ];

  const LSKEY='obs_v1_diada';
  let gs=null, activas=new Map(), editando=false, paletaAbierta=false, cargando=false;
  function renderUna(c){const b=document.getElementById('obsbody_'+c.id);if(b&&window._ultD)c.render(b,window._ultD);}
  window.renderCajas=function(d){activas.forEach(c=>{const b=document.getElementById('obsbody_'+c.id);if(b)c.render(b,d);});};
  function guardar(){if(cargando)return;try{localStorage.setItem(LSKEY,JSON.stringify(gs.save(false)));}catch(e){}}
  function chequearVacio(){const e=document.getElementById('obsVacio');if(e)e.style.display=activas.size?'none':'';}
  function refrescarCatalogo(){const cat=document.getElementById('obsCatalogo');if(!cat)return;cat.innerHTML='';
    CAJAS.forEach(c=>{const chip=document.createElement('span');chip.className='obschip'+(activas.has(c.id)?' puesta':'');chip.textContent=c.tit;
      if(!activas.has(c.id))chip.onclick=()=>addCaja(c.id);cat.appendChild(chip);});}
  function addCaja(cid,pos){if(activas.has(cid))return;const c=CAJAS.find(x=>x.id===cid);if(!c)return;
    const content=`<div class="obscaja"><h3>${c.tit}<span class="obsdel" data-cid="${cid}" title="quitar">✕</span></h3><div class="obsbody" id="obsbody_${cid}">—</div></div>`;
    gs.addWidget({id:cid,content,w:(pos&&pos.w)||c.w,h:(pos&&pos.h)||c.h,x:pos&&pos.x,y:pos&&pos.y});
    activas.set(cid,c);renderUna(c);refrescarCatalogo();chequearVacio();guardar();}
  function setEdit(on){editando=on;document.getElementById('obsZona').classList.toggle('editando',on);if(gs)gs.setStatic(!on);
    document.getElementById('obsAdd').style.display=on?'':'none';document.getElementById('obsReset').style.display=on?'':'none';
    document.getElementById('obsEdit').textContent=on?'✓ Listo':'✏️ Editar tablero';
    if(!on){paletaAbierta=false;document.getElementById('obsPaleta').style.display='none';}}
  function initObs(){if(!window.GridStack){console.warn('Gridstack no disponible');return;}
    gs=GridStack.init({column:12,cellHeight:54,margin:6,float:true,staticGrid:true,handle:'.obscaja h3'},'#obsGrid');
    gs.on('change',guardar);
    document.getElementById('obsGrid').addEventListener('click',e=>{const x=e.target.closest('.obsdel');if(!x)return;
      const cid=x.dataset.cid,item=x.closest('.grid-stack-item');if(item)gs.removeWidget(item);activas.delete(cid);refrescarCatalogo();chequearVacio();guardar();});
    document.getElementById('obsEdit').onclick=()=>setEdit(!editando);
    document.getElementById('obsAdd').onclick=()=>{paletaAbierta=!paletaAbierta;document.getElementById('obsPaleta').style.display=paletaAbierta?'':'none';};
    document.getElementById('obsReset').onclick=()=>{if(!confirm('¿Vaciar el tablero y borrar lo guardado?'))return;gs.removeAll();activas.clear();try{localStorage.removeItem(LSKEY);}catch(e){}refrescarCatalogo();chequearVacio();};
    refrescarCatalogo();cargando=true;let s=null;try{s=JSON.parse(localStorage.getItem(LSKEY)||'null');}catch(e){}
    if(s&&s.length)s.forEach(n=>addCaja(n.id,n));cargando=false;chequearVacio();}
  if(document.readyState!=='loading')initObs();else document.addEventListener('DOMContentLoaded',initObs);
})();
</script></body></html>"""


def main():
    threading.Thread(target=_poller, daemon=True).start()
    # Precompute /stats en background: el 1er click a "hist" no paga el scan de ~2M turnos.
    threading.Thread(target=_stats_precompute, daemon=True, name="stats-precompute").start()
    print(f"[conversacion] observatorio en http://0.0.0.0:{PORT}  ·  A={A_URL} B={B_URL}", flush=True)
    print(f"[conversacion] registro permanente: {LOG_PATH}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
