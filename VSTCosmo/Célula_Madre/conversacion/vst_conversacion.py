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
import os, sys, json, time, threading, urllib.request
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


def _stats_historial():
    """Estadística acumulada de TODA la conversación registrada (todos los días)."""
    hist = {}; total = 0; dias = _dias_disponibles()
    for dia in dias:
        for t in _leer_turnos(dia, limite=10 ** 9):
            em = t.get("emisor", "?"); voz = t.get("prototipo_codebook", "-")
            hist.setdefault(em, {}); hist[em][voz] = hist[em].get(voz, 0) + 1
            total += 1
    return {"dias": dias, "total_turnos": total, "hist": hist}

A_URL = os.environ.get("ANIMA_A_URL", "http://127.0.0.1:7788")
B_URL = os.environ.get("ANIMA_B_URL", "http://127.0.0.1:7799")
CONV_DIR = os.environ.get("ANIMA_CONV_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "registro"))
PORT = int(os.environ.get("ANIMA_CONV_PORT", "9100"))
POLL_S = float(os.environ.get("POLL_S", "0.4"))
MAX_TRANSCRIPT = 400          # turnos recientes en memoria para el tablero

os.makedirs(CONV_DIR, exist_ok=True)
LOG_PATH = os.path.join(CONV_DIR, "conversacion.jsonl")

_EST = {"A": {}, "B": {}}                 # último estado de cada organismo
_TRANSCRIPT = []                          # turnos recientes (dicts) para el tablero
_HIST = {"A": {}, "B": {}}                # histograma de voces por organismo (acumulado de la sesión del observador)
_ULTIMA = {"A": None, "B": None}          # última voz vista (para detectar cambios = turnos)
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
        _HIST[lado][voz] = _HIST[lado].get(voz, 0) + 1
    # BIOGRAFÍA de la díada: cada emisión como evento A↔B con contexto + delta del receptor.
    if _HIST_DIADA is not None:
        otro = "B" if lado == "A" else "A"
        rec = _EST.get(otro, {}); rec_prev = _CTX_PREV.get(otro, {})
        def _d(k):
            try:
                return round(float(rec.get(k, 0) or 0) - float(rec_prev.get(k, 0) or 0), 4)
            except Exception:
                return None
        _HIST_DIADA.comunicacion({
            "ts": turno["ts"], "emisor": est.get("organismo", lado), "receptor": rec.get("organismo", otro),
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
        _CTX_PREV[otro] = dict(rec)


def _poller():
    while True:
        for lado, url in (("A", A_URL), ("B", B_URL)):
            try:
                est = _get_json(url, "/estado")
                with _LOCK:
                    _EST[lado] = est
                voz = est.get("voz_emitida", "-")
                if est.get("vivo") and voz != _ULTIMA[lado]:   # CAMBIÓ de pito = un turno
                    _ULTIMA[lado] = voz
                    _registrar_turno(lado, est)
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
                self._send(200, json.dumps({"A": _EST["A"], "B": _EST["B"],
                                            "transcript": _TRANSCRIPT[-120:],
                                            "hist": _HIST, "log": LOG_PATH}, ensure_ascii=False))
        elif self.path.startswith("/voz/"):
            # PROXY de voz (mismo origen → sin CORS): el navegador pide /voz/A o /voz/B y suena.
            lado = self.path.split("/voz/", 1)[1][:1].upper()
            url = A_URL if lado == "A" else B_URL
            try:
                with urllib.request.urlopen(url + "/voz?seg=1.0&modo=R2D2", timeout=5) as r:
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


HTML = """<!doctype html><html lang=es><head><meta charset=utf-8>
<title>Observatorio de la conversación — Díada ANIMA</title>
<link href="https://cdn.jsdelivr.net/npm/gridstack@10.3.1/dist/gridstack.min.css" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/gridstack@10.3.1/dist/gridstack-all.js"></script>
<style>
:root{--bg:#0a0f16;--panel:#111927;--bord:#243246;--mut:#8aa0b8;--ok:#5fd38a;--gold:#e8b86d}
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
</style></head><body>
<h1>🛰️ Observatorio de la conversación · <span class=mut>díada ANIMA — captura permanente</span></h1>
<div style="display:flex;align-items:center;gap:10px;padding:6px 14px;background:var(--panel);border-bottom:1px solid var(--bord);flex-wrap:wrap">
  <button class=tab id=tabVivo onclick="vista('vivo')">🟢 En vivo</button>
  <button class=tab id=tabHist onclick="vista('hist')">🕮 Historia</button>
  <span style="flex:1"></span>
  <button class=tab id=bAudio onclick="audioToggle()">🔊 Escuchar conversación</button>
  <span class=mut style=font-size:11px>vol</span>
  <input type=range id=vol min=0 max=8 step=0.5 value=4 style=width:90px oninput="setVol(this.value)">
  <span id=volV class=mut style=font-size:11px>4×</span>
  <span class=mut style="font-size:10.5px">A↗izquierda · B↗derecha</span>
</div>
<div id=vivo>
<div class=escena>
  <div class=org>
    <div class=mut id=nomA>Organismo A</div>
    <div class=cabeza id=cabA><div class=nariz id=narA></div><div class=ojo style=left:36px></div><div class=ojo style=left:72px></div></div>
    <div class=pito id=pitoA>—</div>
    <div class=afecto mut id=afA></div>
    <div class=vmed><div id=oiA style=width:0%></div></div>
  </div>
  <div class=entre id=entre>↔</div>
  <div class=org>
    <div class=mut id=nomB>Organismo B</div>
    <div class=cabeza id=cabB><div class=nariz id=narB></div><div class=ojo style=left:36px></div><div class=ojo style=left:72px></div></div>
    <div class=pito id=pitoB>—</div>
    <div class=afecto mut id=afB></div>
    <div class=vmed><div id=oiB style=width:0%></div></div>
  </div>
</div>
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
<script>
const $=id=>document.getElementById(id);
const COL={screaming:'#ff5b5b',shout:'#ff8c6b',worried:'#e8b86d',excited:'#5fd38a','excited-2':'#5fd38a',sing:'#8ef0c0',acknowledged:'#6db6ff',chat:'#9fb1c6'};
const emo=v=>({screaming:'😱',shout:'😨',worried:'😟',excited:'🤩','excited-2':'😃',sing:'🎶',acknowledged:'👍',chat:'💬',
  alegria:'😊',miedo:'😨',calma:'😌',curiosidad:'🤔',asombro:'😮',dolor:'😣',tristeza:'😢',ternura:'🥰',
  alerta:'⚠️',peligro:'🚨',urgencia:'⏰',atencion:'👁️',hambre:'🍽️',fatiga:'🥱',saludo:'👋',despedida:'🫡',
  pregunta:'❓',respuesta:'💡',duda:'🤨',confusion:'😵',acuerdo:'🤝',desacuerdo:'🙅',negacion:'🚫',afirmacion:'✅',
  llamada:'📣',compania:'🫂',exploracion:'🧭',hallazgo:'✨',novedad:'🆕',despertar:'🌅',satisfaccion:'😋',frustracion:'😤'}[v]||'🤖');
function cabeza(pre, est){
  const o=(est&&est.orientacion_deg)||0;
  $('nar'+pre).style.transform='translateY(-50%) rotate('+(-o)+'deg)';  // gira la 'nariz' hacia donde mira
  $('cab'+pre).style.transform='rotate('+(o*0.15)+'deg)';
  const v=(est&&est.voz_emitida)||'—', tit=(est&&est.voz_titulo)||v, vivo=est&&est.vivo;
  $('pito'+pre).textContent=vivo?(emo(v)+' '+tit):'· dormido ·';   // título en castellano
  $('pito'+pre).style.color=COL[v]||'#dfe7f0';
  $('af'+pre).textContent=vivo?('aro '+(+est.voz_arousal||0).toFixed(2)+' · val '+(+est.voz_valence||0).toFixed(2)+' · OI '+(+est.OI||0).toFixed(2)):'';
  $('oi'+pre).style.width=Math.min(100,((+((est||{}).OI))||0)*100)+'%';
  if(est&&est.organismo)$('nom'+pre).textContent=est.organismo;
}
function hist(pre, h){
  const tot=Object.values(h||{}).reduce((a,b)=>a+b,0)||1;
  const ent=Object.entries(h||{}).sort((a,b)=>b[1]-a[1]);
  $('hist'+pre).innerHTML='<div class=mut style=font-size:10px>'+pre+'</div>'+ent.map(([v,n])=>
    '<div class=barra><span style=width:88px>'+emo(v)+' '+v+'</span><div class=b><div style="width:'+(100*n/tot)+'%;background:'+(COL[v]||'#5fd38a')+'"></div></div><span style="width:34px;text-align:right" class=mut>'+n+'</span></div>').join('');
}
let nT=0;
async function tick(){
  let d; try{ d=await fetch('/datos').then(r=>r.json()); }catch(e){ return; }
  cabeza('A', d.A); cabeza('B', d.B);
  window._ultD=d; if(window.renderCajas)renderCajas(d);   // OBSERVATORIO de la díada (cajas)
  // ¿se miran? (A mira a la derecha y B a la izquierda)
  const oa=(d.A&&d.A.orientacion_deg)||0, ob=(d.B&&d.B.orientacion_deg)||0;
  $('entre').textContent=(oa>5 && ob<-5)?'👁️‍🗨️':'↔';
  $('logp').textContent=d.log||'';
  hist('A', (d.hist||{}).A); hist('B', (d.hist||{}).B);
  if(d.transcript && d.transcript.length!==nT){
    nT=d.transcript.length;
    $('trans').innerHTML=d.transcript.slice(-120).map(t=>{
      const hh=new Date(t.ts*1000).toLocaleTimeString();
      return '<div class=linea><span class=mut>'+hh+'</span> <span class='+t.lado+'>'+t.lado+'</span> '+emo(t.voz)+' <b>'+(t.titulo||t.voz)+'</b> <span class=mut>(aro '+(+t.arousal||0).toFixed(2)+', val '+(+t.valence||0).toFixed(2)+')</span></div>';
    }).join('');
    $('trans').scrollTop=$('trans').scrollHeight;
  }
}
// ---- pestañas En vivo / Historia ----
function vista(v){
  $('vivo').style.display=v==='vivo'?'':'none'; $('historia').style.display=v==='hist'?'':'none';
  $('tabVivo').classList.toggle('on', v==='vivo'); $('tabHist').classList.toggle('on', v==='hist');
  if(v==='hist') initHist();
}
vista('vivo');

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
      const hh=new Date((t.ts||0)*1000).toLocaleTimeString(), lado=(''+(t.emisor||'')).indexOf('B')>=0?'B':'A';
      const dOI=((t.delta_receptor||{}).delta_OI);
      return '<div class=linea><span class=mut>'+hh+'</span> <span class='+lado+'>'+(t.emisor||'?')+'→'+(t.receptor||'?')+'</span> '+emo(t.prototipo_codebook)+' <b>'+t.prototipo_codebook+'</b> <span class=mut>ΔOI_rec '+(dOI==null?'·':dOI)+'</span></div>';
    }).join('');
    $('transH').scrollTop=$('transH').scrollHeight;
  }catch(e){}
}
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
   {id:'libertad_creativa',tit:'🎨 Libertad creativa (balbuceo)',w:6,h:5,render:(b,d)=>{
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
   {id:'intencion',tit:'🗣 Intención comunicativa (A/B)',w:3,h:3,render:(b,d)=>{const A=G(d,'A'),B=G(d,'B');b.innerHTML=
     cjRow('A intención',cjN(A.int,3))+cjGauge(A.int,'#6db6ff')+cjRow('A efecto→otro',cjN(A.eo,3))+cjGauge(A.eo,'#6db6ff')
    +cjRow('B intención',cjN(B.int,3))+cjGauge(B.int,'#ffd479')+cjRow('B efecto→otro',cjN(B.eo,3))+cjGauge(B.eo,'#ffd479');}},
   {id:'afecto',tit:'🔊 Afecto / Voz (A/B)',w:3,h:3,render:(b,d)=>{const A=G(d,'A'),B=G(d,'B');b.innerHTML=
     cjRow('A',A.voz)+`<span class="obsk">arousal</span>`+cjGauge(A.aro,'#ff8c6b')+`<span class="obsk">valencia</span>`+cjBip(A.val,'#6db6ff')
    +cjRow('B',B.voz)+`<span class="obsk">arousal</span>`+cjGauge(B.aro,'#ff8c6b')+`<span class="obsk">valencia</span>`+cjBip(B.val,'#ffd479');}},
   {id:'acople',tit:'❤️ Acople (OI A↔B)',w:3,h:2,render:(b,d)=>{const A=G(d,'A'),B=G(d,'B');b.innerHTML=
     cjRow('OI · A',cjN(A.oi,3))+cjGauge(A.oi,'#6db6ff')+cjRow('OI · B',cjN(B.oi,3))+cjGauge(B.oi,'#ffd479')
    +cjRow('|diferencia|',cjN(Math.abs(A.oi-B.oi),3));}},
   {id:'agencia',tit:'🧭 Agencia del otro (A/B)',w:3,h:3,render:(b,d)=>{const q=(L,k)=>(+(((d&&d[L])||{})[k])||0);b.innerHTML=
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
   {id:'expresion',tit:'🎙 Habla / No Habla (A/B)',w:3,h:3,render:(b,d)=>{const q=(L,k)=>(+(((d&&d[L])||{})[k])||0);
     const est=(L)=>(q(L,'expr_vocalizando')>=.5?'🗣 habla':(q(L,'expr_silencio')>=.5?'🤫 calla':'·'));
     window._exprT=window._exprT||{A:{h:0,n:0},B:{h:0,n:0}};
     for(const L of ['A','B']){if(q(L,'expr_vocalizando')>=.5)window._exprT[L].h++;else if(q(L,'expr_silencio')>=.5)window._exprT[L].n++;}
     const med=(L)=>{const t=window._exprT[L],tot=(t.h+t.n)||1,fh=t.h/tot;
       return cjRow(L+' · '+est(L), '🗣 '+(fh*100).toFixed(0)+'% · '+((1-fh)*100).toFixed(0)+'% 🤫')
         +`<div class="obsgauge" style="display:flex"><i style="width:${(fh*100).toFixed(1)}%;background:#5fd38a"></i><i style="width:${((1-fh)*100).toFixed(1)}%;background:#8aa0b8"></i></div>`;};
     b.innerHTML = med('A') + med('B')
    +`<div class="obsk" style="font-size:9px;margin-top:3px">el 1er acto es decidir SI hablar; el silencio es una conducta (acumulado de la sesión)</div>`;}},
   {id:'imitacion',tit:'🧠 Aprendizaje / Imitación (A/B)',w:3,h:3,render:(b,d)=>{const q=(L,k)=>(+(((d&&d[L])||{})[k])||0);b.innerHTML=
     cjRow('A oye · ecoica',cjN(q('A','oao_oido'),3)+' · '+cjN(q('A','oao_echoica_n'),0))
    +cjRow('A imitación',cjN(q('A','oao_imitacion_mag'),3))+cjGauge(Math.min(1,q('A','oao_imitacion_mag')*2),'#8ef0c0')
    +cjRow('B oye · ecoica',cjN(q('B','oao_oido'),3)+' · '+cjN(q('B','oao_echoica_n'),0))
    +cjRow('B imitación',cjN(q('B','oao_imitacion_mag'),3))+cjGauge(Math.min(1,q('B','oao_imitacion_mag')*2),'#8ef0c0')
    +`<div class="obsk" style="font-size:9px;margin-top:3px">lo oído sesga la voz futura por historia (imitación, no copia)</div>`;}},
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
    print(f"[conversacion] observatorio en http://0.0.0.0:{PORT}  ·  A={A_URL} B={B_URL}", flush=True)
    print(f"[conversacion] registro permanente: {LOG_PATH}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
