#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_CelulaMadre_Web — INTERFAZ WEB DE LA CÉLULA MADRE (Opción B: backend Python)
================================================================================

QUÉ ES
------
Un servidor local (stdlib, SIN dependencias nuevas) que envuelve la célula madre
funcional (`Célula_Madre_Funcional_001.py`) y la expone como una interfaz web al
estilo de los experimentos cosmosemióticos (Levitron / EIT3 / Dron): tema oscuro,
paneles, chart.js en vivo, narrador, y descarga de CSV con el mismo idiom Blob.

POR QUÉ B (backend Python, no JS puro)
--------------------------------------
1) Modificar los organelos EN CÓDIGO: el motor es el Python validado; se edita el
   organelo y la web lo refleja, sin reescribir nada en JS.
2) Separar la ejecución de cada parte: un INTERRUPTOR por organelo (= `expresar`).
   Apagar un módulo lo saca del ciclo metabólico → aislamiento REAL (no solo visual).
3) Resultados sin arriesgar todo: cada corrida es independiente; se compara el CSV
   con y sin un módulo (ablación) sin tocar el organismo base.

CÓMO CORRER
-----------
    venv/bin/python3 VST_CelulaMadre_Web.py
    → abre http://localhost:7777 en el navegador.

Rutas: GET /  (interfaz) · GET /organelos (lista de módulos) · POST /run (corre).
================================================================================
"""

from __future__ import annotations
import os, sys, json, base64, tempfile, importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PUERTO = 7777
AQUI = os.path.dirname(os.path.abspath(__file__))

# --- Cargar la célula madre funcional (nombre de archivo con acento → importlib) ---
# El soma vive en Célula_Madre/campo/; este módulo en Célula_Madre/web/. Busca en campo/ y
# cae a AQUI si la estructura aún es plana.
_cmf_path = os.path.join(os.path.dirname(AQUI), "campo", "Célula_Madre_Funcional_001.py")
if not os.path.isfile(_cmf_path):
    _cmf_path = os.path.join(os.path.dirname(AQUI), "campo", "Celula_Madre_Funcional_001.py")
if not os.path.isfile(_cmf_path):
    _cmf_path = os.path.join(AQUI, "Célula_Madre_Funcional_001.py")
if not os.path.isfile(_cmf_path):
    _cmf_path = os.path.join(AQUI, "Celula_Madre_Funcional_001.py")
_spec = importlib.util.spec_from_file_location("cmf", _cmf_path)
cmf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmf)

# --- Catálogo de organelos para la UI (nombre interno, grupo, etiqueta) ---
# 'soma' es OBLIGATORIO (procesa el audio): no se puede apagar.
ORG_UI = [
    ("soma",                      "Cuerpo",       "Soma — campo Φ (procesa el audio)", True),
    ("presion_desacople",         "Base",         "Presión de desacople (arousal)", False),
    ("fatiga",                    "Base",         "Fatiga (tiempo biológico)", False),
    ("consciencia_basica",        "B5 Consciencia", "C_b = |R₁| (consciencia básica)", False),
    ("meta_representacion",       "B5 Consciencia", "R₂ = R(R) (meta-representación)", False),
    ("self",                      "B5 Consciencia", "Self = operador(R₂)", False),
    ("ritual",                    "B7 Libertad",  "Ritual (estadio 2)", False),
    ("juego",                     "B7 Libertad",  "Juego (estadio 1)", False),
    ("LF",                        "B7 Libertad",  "Libertad Funcional (medida)", False),
    ("negacion_operativa",        "B7 Libertad",  "Negación operativa (el 'No')", False),
    ("mutacion",                  "B8 Evolución", "Mutación (ΔR aleatoria)", False),
    ("adaptacion",                "B8 Evolución", "Adaptación (Ωop cte)", False),
    ("exaptacion",                "B8 Evolución", "Exaptación (ΔΩop, XE)", False),
    ("consciencia_metacognitiva", "B8 Evolución", "C_m (metacognición)", False),
    ("activacion_latente",        "B8 Evolución", "Activación latente", False),
    ("homeostasis::x_interna",    "Homeostasis",  "Homeostasis (componente H)", False),
]

# Columnas del CSV (una fila por paso metabólico) = señales clave del milieu + medidas.
COLS = ["t", "Omega", "omega_A", "omega_B", "gradiente", "e_R", "A_sys_env",
        "presion_desacople", "C_b", "R2", "LF_op", "lf_nivel", "juego", "ritual",
        "negacion", "demanda_entorno", "Omega_op", "XE", "C_m", "H_homeostasis",
        "OI", "Lambda_Cos"]


def _cargar(audio_spec: dict, binaural: bool = False):
    """Devuelve (nombre, audio). En mono, audio = vector; en binaural, audio = (izq, der).
    audio_spec = {'type':'demo','spec':'demo:tono'} o {'type':'upload','b64':...,'name':...}."""
    if audio_spec.get("type") == "upload":
        raw = audio_spec["b64"]
        if "," in raw:
            raw = raw.split(",", 1)[1]            # quita prefijo dataURL
        datos = base64.b64decode(raw)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(datos); tmp.close()
        try:
            nombre, audio = cmf.cargar_audio(tmp.name, binaural=binaural)
            return (audio_spec.get("name") or nombre, audio)
        finally:
            os.unlink(tmp.name)
    return cmf.cargar_audio(audio_spec.get("spec", "demo:tono"), binaural=binaural)


def run_sim(audio_spec: dict, toggles: dict, sim_s: float | None = None,
            binaural: bool = False) -> dict:
    """Corre la célula madre sobre un audio con los interruptores dados. En MONO, columnas
    y valores IDÉNTICOS al comportamiento previo (invariante). En BINAURAL, se AÑADEN las
    columnas omega_L, omega_R, gradiente_lateral (ω_B y el gradiente NO cambian)."""
    nombre, audio = _cargar(audio_spec, binaural)
    cel = cmf.celula_madre_funcional(audio, binaural=binaural)
    # --- aplicar interruptores: silenciar organelos apagados (soma siempre ON) ---
    apagados = []
    for name, org in cel.organelos.items():
        if name == "soma":
            continue
        if not toggles.get(name, True):
            org.expresar = False
            apagados.append(name)

    soma = cel.organelos["soma"]
    dur = soma.dur                               # soma.dur sirve para mono y binaural
    n_muestras = len(soma._L)
    sim = min(dur, float(sim_s) if sim_s else cmf.SIM_CAP_S)
    pasos = max(1, int(sim / cmf.DT))
    # columnas: las 22 de siempre; en binaural se AÑADEN 3 (mono queda igual)
    cols = list(COLS) + (["omega_L", "omega_R", "gradiente_lateral"] if binaural else [])

    rows = []
    cm_peak = 0.0
    for _ in range(pasos):
        cel.vivir_un_paso(cmf.DT)
        m = cel.milieu
        s = cel.salud()
        cm_peak = max(cm_peak, m.leer("C_m", 0.0))
        fila = [
            round(cel.t, 3), round(m.leer("Omega", 0.0), 4), round(m.leer("omega_A", 0.0), 4),
            round(m.leer("omega_B", 0.0), 4), round(m.leer("gradiente", 0.0), 4),
            round(m.leer("e_R", 0.0), 4), round(m.leer("A_sys_env", 0.0), 4),
            round(m.leer("presion_desacople", 0.0), 3), int(m.leer("C_b", 0)),
            round(m.leer("R2", 0.0), 4), round(m.leer("LF_op", 0.0), 4), int(m.leer("lf_nivel", 0)),
            int(bool(m.leer("juego_activo", False))), int(bool(m.leer("ritual_activo", False))),
            int(bool(m.leer("negacion_activa", False))), round(m.leer("demanda_entorno", 1.0), 4),
            round(m.leer("Omega_op", 1.0), 4), round(min(1.0, m.leer("XE", 0.0)), 4),
            round(m.leer("C_m", 0.0), 4), round(m.leer("H_homeostasis", 0.0), 4),
            round(s["OI"], 4), round(s["Lambda_Cos"], 4),
        ]
        if binaural:                             # columnas nuevas SOLO en binaural
            fila += [round(m.leer("omega_L", 0.0), 4), round(m.leer("omega_R", 0.0), 4),
                     round(m.leer("gradiente_lateral", 0.0), 4)]
        rows.append(fila)

    s = cel.salud()
    inv_ok = sum(1 for v in s["invariantes"].values() if v)
    csv_text = ",".join(cols) + "\n" + "\n".join(",".join(str(x) for x in r) for r in rows)
    series = {c: [r[i] for r in rows] for i, c in enumerate(cols)}
    resumen = {
        "audio": nombre, "muestras": int(n_muestras), "duracion_s": round(dur, 2),
        "sim_s": round(sim, 1), "pasos": pasos, "campo_finito": bool(soma.finito),
        "Omega_medio": round(sum(series["Omega"]) / len(series["Omega"]), 4),
        "OI": round(s["OI"], 4), "nivel_OI": s["nivel_OI"], "Lambda_Cos": round(s["Lambda_Cos"], 4),
        "invariantes": f"{inv_ok}/6", "C_m_pico": round(cm_peak, 3),
        "binaural": bool(binaural), "lateralidad_real": bool(getattr(soma, "lateralidad_real", False)),
        "apagados": apagados, "organelos_activos": [n for n in cel.organelos if cel.organelos[n].expresar],
    }
    return {"cols": cols, "csv": csv_text, "series": series, "resumen": resumen}


# ==============================================================================
# FRONTEND (estilo familia: tema oscuro, paneles, chart.js, narrador, ⬇CSV)
# ==============================================================================
HTML = r"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Célula Madre — Interfaz Funcional</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0a0e14;--panel:#121925;--ink:#dfe7f0;--mut:#8aa0b8;--gold:#e8b86d;--ok:#5fd38a;--bad:#ff6b6b;--line:#243246;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 'Helvetica Neue',Arial,sans-serif}
h1{font-size:18px;margin:0;color:var(--gold)}h2{font-size:13px;color:var(--gold);text-transform:uppercase;letter-spacing:.06em;margin:0 0 8px}
.wrap{display:grid;grid-template-columns:340px 1fr;gap:14px;padding:14px;max-width:1400px;margin:auto}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:14px}
.grp{margin:6px 0 10px}.grp .gt{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.05em;margin:8px 0 4px}
.tog{display:flex;align-items:center;gap:8px;padding:3px 0}.tog input{accent-color:var(--gold);width:16px;height:16px}
.tog label{cursor:pointer;font-size:13px}.tog.req label{color:var(--mut)}
button{background:#1b2636;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:8px 12px;cursor:pointer;font-size:13px}
button:hover{border-color:var(--gold)}button.go{background:var(--gold);color:#1a1206;font-weight:bold;border:none}
button.csv{border-color:var(--gold);color:var(--gold)}
select,input[type=number]{background:#0c121b;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px;width:100%}
label.fld{display:block;color:var(--mut);font-size:11px;margin:8px 0 3px;text-transform:uppercase;letter-spacing:.04em}
.row{display:flex;gap:8px;align-items:center;margin-top:10px}
.chk{font-size:13px;line-height:1.7}.ok{color:var(--ok)}.bad{color:var(--bad)}.mut{color:var(--mut)}
.big{font-size:26px;color:var(--gold);font-weight:bold}
#log{font:12px/1.5 'SF Mono',Menlo,monospace;background:#070a0f;border:1px solid var(--line);border-radius:6px;
     padding:8px;height:120px;overflow:auto}
.canwrap{position:relative;height:200px;margin-bottom:8px}
.muted{color:var(--mut);font-size:12px}
input[type=file]{font-size:12px;color:var(--mut)}
</style></head><body>
<div class="wrap">
  <!-- ===== CONTROLES ===== -->
  <div>
    <div class="panel">
      <h1>🧬 Célula Madre — Funcional</h1>
      <div class="muted">Procesa audio · interruptor por organelo · CSV por paso</div>
      <label class="fld">Audio</label>
      <select id="demo">
        <option value="demo:tono">demo — tono 440 Hz</option>
        <option value="demo:rosa">demo — ruido rosa</option>
        <option value="demo:clicks">demo — clicks Poisson</option>
        <option value="__upload">— subir archivo .wav —</option>
      </select>
      <input type="file" id="file" accept=".wav" style="display:none;margin-top:8px">
      <div class="tog" style="margin-top:8px"><input type="checkbox" id="binaural"><label for="binaural">Binaural (L = canal izq · R = canal der)</label></div>
      <label class="fld">Segundos de simulación (máx 20)</label>
      <input type="number" id="sim" value="3" min="1" max="20" step="1">
      <div class="row">
        <button class="go" id="run" style="flex:1">▶ Procesar</button>
        <button class="csv" id="dl" disabled>⬇ CSV</button>
      </div>
    </div>
    <div class="panel">
      <h2>Organelos (interruptores)</h2>
      <div id="toggles"></div>
      <div class="row"><button id="all">Todos</button><button id="none">Solo soma</button></div>
    </div>
  </div>
  <!-- ===== RESULTADOS ===== -->
  <div>
    <div class="panel" id="resumen"><h2>Resultado</h2><div class="muted">Pulsa «Procesar» para correr la célula madre.</div></div>
    <div class="panel">
      <h2>Ω — estado representacional</h2><div class="canwrap"><canvas id="cOmega"></canvas></div>
      <h2>Organismicidad: OI &amp; Λ_Cos</h2><div class="canwrap"><canvas id="cOI"></canvas></div>
      <h2>Organelos en el tiempo</h2><div class="canwrap"><canvas id="cOrg"></canvas></div>
    </div>
    <div class="panel"><h2>Narrador</h2><div id="log"></div></div>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
let lastCSV="", lastName="celula";
function log(m,c){const d=document.createElement('div');if(c)d.className=c;d.textContent='» '+m;$('log').prepend(d);}
// construir interruptores desde el backend
fetch('/organelos').then(r=>r.json()).then(list=>{
  const byG={};list.forEach(o=>{(byG[o.grupo]=byG[o.grupo]||[]).push(o);});
  const cont=$('toggles');
  for(const g in byG){
    const h=document.createElement('div');h.className='gt';h.textContent=g;cont.appendChild(h);
    byG[g].forEach(o=>{
      const w=document.createElement('div');w.className='tog'+(o.req?' req':'');
      w.innerHTML=`<input type="checkbox" id="t_${o.name}" ${o.req?'checked disabled':'checked'}><label for="t_${o.name}">${o.label}</label>`;
      cont.appendChild(w);
    });
  }
});
$('demo').onchange=e=>{$('file').style.display=e.target.value==='__upload'?'block':'none';};
$('all').onclick=()=>document.querySelectorAll('#toggles input:not([disabled])').forEach(c=>c.checked=true);
$('none').onclick=()=>document.querySelectorAll('#toggles input:not([disabled])').forEach(c=>c.checked=false);

let charts={};
function mkChart(id,labels,datasets){
  if(charts[id])charts[id].destroy();
  charts[id]=new Chart($(id),{type:'line',data:{labels,datasets},
    options:{animation:false,responsive:true,maintainAspectRatio:false,
      scales:{x:{ticks:{color:'#8aa0b8',maxTicksLimit:8},grid:{color:'#1a2330'}},
              y:{ticks:{color:'#8aa0b8'},grid:{color:'#1a2330'}}},
      plugins:{legend:{labels:{color:'#dfe7f0',boxWidth:12}}},elements:{point:{radius:0}}}});
}
function ds(label,data,color){return{label,data,borderColor:color,borderWidth:2,tension:.2};}

async function run(){
  const sel=$('demo').value; let audio;
  if(sel==='__upload'){
    const f=$('file').files[0]; if(!f){log('Elige un archivo .wav','bad');return;}
    audio={type:'upload',name:f.name,b64:await new Promise(res=>{const r=new FileReader();r.onload=()=>res(r.result);r.readAsDataURL(f);})};
  } else audio={type:'demo',spec:sel};
  const toggles={};document.querySelectorAll('#toggles input').forEach(c=>toggles[c.id.slice(2)]=c.checked);
  log('Procesando…');$('run').disabled=true;
  try{
    const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({audio,toggles,sim_s:+$('sim').value,binaural:$('binaural').checked})});
    const d=await r.json();
    if(d.error){log('Error: '+d.error,'bad');return;}
    pintar(d);
  }catch(e){log('Error: '+e,'bad');}
  finally{$('run').disabled=false;}
}
function pintar(d){
  const R=d.resumen, S=d.series, t=S.t;
  lastCSV=d.csv; lastName=R.audio; $('dl').disabled=false;
  const inv=R.invariantes==='6/6';
  $('resumen').innerHTML=`<h2>Resultado — ${R.audio}</h2>
    <div class="big">OI = ${R.OI} → ${R.nivel_OI.toUpperCase()}</div>
    <div class="chk">${R.campo_finito?'<span class=ok>✅</span>':'<span class=bad>❌</span>'} campo Φ finito ·
      Ω medio = ${R.Omega_medio} · Λ_Cos = ${R.Lambda_Cos}</div>
    <div class="chk">${inv?'<span class=ok>✅</span>':'<span class=bad>❌</span>'} invariantes κ ${R.invariantes} ·
      C_m pico = ${R.C_m_pico} · ${R.muestras} muestras / ${R.duracion_s}s (sim ${R.sim_s}s)</div>
    <div class="chk mut">apagados: ${R.apagados.length?R.apagados.join(', '):'ninguno'}</div>
    ${R.binaural?`<div class="chk">🎧 BINAURAL · lateralidad real en la fuente: ${R.lateralidad_real?'<span class=ok>sí</span>':'<span class=bad>no (mono duplicado)</span>'} · columnas ω_L/ω_R/gradiente_lateral añadidas al CSV</div>`:''}`;
  const dsOmega=[ds('Ω',S.Omega,'#e8b86d')];
  if(S.gradiente_lateral){dsOmega.push(ds('ω_L',S.omega_L,'#6db6ff'));dsOmega.push(ds('ω_R',S.omega_R,'#ff8c6b'));}
  mkChart('cOmega',t,dsOmega);
  mkChart('cOI',t,[ds('OI',S.OI,'#5fd38a'),ds('Λ_Cos',S.Lambda_Cos,'#6db6ff')]);
  mkChart('cOrg',t,[ds('LF_op',S.LF_op,'#e8b86d'),ds('XE',S.XE,'#ff8c6b'),
    ds('H',S.H_homeostasis,'#5fd38a'),ds('C_m',S.C_m,'#b58cff'),ds('R₂',S.R2,'#6db6ff')]);
  log(`Listo: ${R.pasos} pasos · OI ${R.OI} (${R.nivel_OI})`,'ok');
}
function descargar(){
  if(!lastCSV)return;
  const blob=new Blob([lastCSV],{type:'text/csv'});const url=URL.createObjectURL(blob);
  const a=document.createElement('a');a.href=url;
  a.download=`celula_madre_${lastName}_${new Date().toISOString().slice(0,19).replace(/[:.]/g,'-')}.csv`;
  a.click();URL.revokeObjectURL(url);log('CSV descargado','ok');
}
$('run').onclick=run;$('dl').onclick=descargar;
log('Interfaz lista. Backend Python = célula madre validada.');
</script></body></html>"""


# ==============================================================================
# SERVIDOR (stdlib)
# ==============================================================================
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):  # silenciar el log ruidoso por request
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML, "text/html")
        elif self.path == "/organelos":
            data = [{"name": n, "grupo": g, "label": l, "req": req} for n, g, l, req in ORG_UI]
            self._send(200, json.dumps(data, ensure_ascii=False))
        else:
            self._send(404, json.dumps({"error": "no encontrado"}))

    def do_POST(self):
        if self.path != "/run":
            self._send(404, json.dumps({"error": "no encontrado"})); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            out = run_sim(req.get("audio", {"type": "demo", "spec": "demo:tono"}),
                          req.get("toggles", {}), req.get("sim_s"), req.get("binaural", False))
            self._send(200, json.dumps(out, ensure_ascii=False))
        except Exception as e:
            import traceback; traceback.print_exc()
            self._send(200, json.dumps({"error": str(e)}))


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PUERTO), Handler)
    print("=" * 64)
    print("  CÉLULA MADRE — INTERFAZ WEB FUNCIONAL (backend Python)")
    print(f"  → abre:  http://localhost:{PUERTO}")
    print("  Ctrl+C para detener.")
    print("=" * 64)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  detenido.")
        srv.shutdown()


if __name__ == "__main__":
    main()
