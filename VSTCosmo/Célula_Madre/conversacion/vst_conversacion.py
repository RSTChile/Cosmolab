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
             "voz": voz, "arousal": est.get("voz_arousal"), "valence": est.get("voz_valence"),
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
const emo=v=>({screaming:'😱',shout:'😨',worried:'😟',excited:'🤩','excited-2':'😃',sing:'🎶',acknowledged:'👍',chat:'💬'}[v]||'🤖');
function cabeza(pre, est){
  const o=(est&&est.orientacion_deg)||0;
  $('nar'+pre).style.transform='translateY(-50%) rotate('+(-o)+'deg)';  // gira la 'nariz' hacia donde mira
  $('cab'+pre).style.transform='rotate('+(o*0.15)+'deg)';
  const v=(est&&est.voz_emitida)||'—', vivo=est&&est.vivo;
  $('pito'+pre).textContent=vivo?(emo(v)+' '+v):'· dormido ·';
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
  // ¿se miran? (A mira a la derecha y B a la izquierda)
  const oa=(d.A&&d.A.orientacion_deg)||0, ob=(d.B&&d.B.orientacion_deg)||0;
  $('entre').textContent=(oa>5 && ob<-5)?'👁️‍🗨️':'↔';
  $('logp').textContent=d.log||'';
  hist('A', (d.hist||{}).A); hist('B', (d.hist||{}).B);
  if(d.transcript && d.transcript.length!==nT){
    nT=d.transcript.length;
    $('trans').innerHTML=d.transcript.slice(-120).map(t=>{
      const hh=new Date(t.ts*1000).toLocaleTimeString();
      return '<div class=linea><span class=mut>'+hh+'</span> <span class='+t.lado+'>'+t.lado+'</span> '+emo(t.voz)+' <b>'+t.voz+'</b> <span class=mut>(aro '+(+t.arousal||0).toFixed(2)+', val '+(+t.valence||0).toFixed(2)+')</span></div>';
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
</script></body></html>"""


def main():
    threading.Thread(target=_poller, daemon=True).start()
    print(f"[conversacion] observatorio en http://0.0.0.0:{PORT}  ·  A={A_URL} B={B_URL}", flush=True)
    print(f"[conversacion] registro permanente: {LOG_PATH}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
