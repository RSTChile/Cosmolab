import subprocess, time, os, hashlib, datetime
ROOT = "/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo"
os.chdir(ROOT)
PY3 = os.path.join(ROOT, "venv/bin/python3")
OUT = os.path.join(ROOT, "verificacion_hitos")
TIMEOUT = 7200  # 2 h por script

HITOS = [
    ("V103",  "V103.py",                 "Síntesis V90-103: clasificación multiestímulo Ω"),
    ("V117",  "V117.py",                 "Aplicada: R₂ sin lateralidad"),
    ("V118",  "V118.py",                 "Aplicada: lateralidad sin R₂ (trade-off)"),
    ("V122",  "V122.py",                 "Aplicada: coexistencia R₂+lateralidad (primer EIT-3)"),
    ("V132",  "V132.py",                 "Clausura: organismo mínimo funcional"),
    ("V147",  "V147.py",                 "Clausura: baseline fisiológico sano"),
    ("V150",  "V150.py",                 "Clausura: IONB-1, cierre ANIMA-1"),
    ("V176",  "V176.py",                 "V176: R_op, primer 'No' operativo"),
    ("V180c", "V180c.py",                "V180: memoria episódica"),
    ("V182A5","V182A5_acumulacion.py",   "ANIMA4: comunicación bidireccional"),
]

def file_hash(p):
    try:
        return hashlib.sha1(open(p,'rb').read()).hexdigest()[:12]
    except Exception:
        return "nohash"

env = dict(os.environ); env["MPLBACKEND"] = "Agg"
manifest = open(os.path.join(OUT, "_MANIFIESTO.txt"), "w", buffering=1)
manifest.write(f"# Verificación 10 hitos — runner lanzado {datetime.datetime.now().isoformat()}\n")
manifest.write(f"# python={PY3}\n\n")

CONC = 3
running = {}   # name -> (proc, fo, t0, path, log)
queue = list(HITOS)
done = []

def launch(item):
    name, path, desc = item
    log = os.path.join(OUT, f"{name}.log")
    fo = open(log, "w")
    stamp = datetime.datetime.now().isoformat()
    fo.write(f"==== VERIFICACIÓN HITO {name} ====\n")
    fo.write(f"script   : {path}\n")
    fo.write(f"sha1[:12]: {file_hash(os.path.join(ROOT,path))}\n")
    fo.write(f"mtime    : {datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat() if os.path.exists(path) else 'MISSING'}\n")
    fo.write(f"reporta  : {desc}\n")
    fo.write(f"inicio   : {stamp}\n")
    fo.write("="*70 + "\n\n"); fo.flush()
    if not os.path.exists(path):
        fo.write("ERROR: script no encontrado\n"); fo.close()
        manifest.write(f"MISSING :: {name} :: {path}\n")
        return None
    p = subprocess.Popen([PY3, path], stdout=fo, stderr=subprocess.STDOUT, env=env)
    return (p, fo, time.time(), path, log)

while queue or running:
    while queue and len(running) < CONC:
        item = queue.pop(0)
        r = launch(item)
        if r: running[item[0]] = (item, r)
    fin = []
    for name, (item, (p, fo, t0, path, log)) in running.items():
        rc = p.poll()
        if rc is not None:
            dt = int(time.time()-t0); fo.write(f"\n\n[fin rc={rc} {dt}s]\n"); fo.close()
            manifest.write(f"{'OK' if rc==0 else 'FAIL'} rc={rc} {dt}s :: {name} :: {path}\n")
            fin.append(name)
        elif time.time()-t0 > TIMEOUT:
            p.kill(); dt=int(time.time()-t0); fo.write(f"\n\n[TIMEOUT {dt}s]\n"); fo.close()
            manifest.write(f"TIMEOUT rc=-1 {dt}s :: {name} :: {path}\n")
            fin.append(name)
    for n in fin: running.pop(n)
    if queue or running: time.sleep(10)

manifest.write("\nALL_DONE\n"); manifest.close()
