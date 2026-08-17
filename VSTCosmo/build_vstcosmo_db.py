#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construye vstcosmo.db: índice consultable de los scripts de experimentos."""
import os, re, sqlite3, hashlib, glob, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(ROOT, "vstcosmo.db")

# No-experimentos (utilidades / hitos / dependencias congeladas se marcan aparte)
UTILIDADES = {
    "Crear_Audios.py","Procesar_Biaurales.py","biaurales.py",
    "generar_blue_monday_binaural.py","generar_tonos_largos.py",
    "grafico.py","exportar_evidencia.py","barrido_fino.py",
    "prueba_umbrales.py","prueba_umbrales_v2.py","test_interpretacion.py",
    "diagnostico_acoplamiento.py","caracterizacion_regimenes.py",
    "_build_db.py",
}
CONGELADOS = {"V122.py","V180.py","VST_Organismo_Individual.py"}
HITOS_NO_NUM = {"VSTCosmo-Anima-1.py","VST_Celula_Madre_001.py"}

# Ciclos (de CRONOLOGIA_POR_SCRIPTS.md), por rango de numero
CICLOS = [
    (1,5,0,"Validación con grabaciones reales (cosmosemiotico v1-v5.4)","23 abr"),
    (6,19,1,"Campo Φ (núcleo mínimo)","23 abr"),
    (20,27,2,"Memorias Ψ/Ω/Σ","23-24 abr"),
    (28,39,3,"Homeostasis y metabolismo","24 abr"),
    (40,58,4,"Miedo, valor y fatiga estructural","may"),
    (59,69,5,"Campo espectral y oscilador","may"),
    (70,79,6,"GIRO: Campo continuo C-N2.0 + W hebbiana","may"),
    (80,96,7,"Campo rico: dual W + ganglio + orientación","may"),
    (97,117,8,"Ω-categorización, dos agentes, alma racional","26-27 may"),
    (118,121,9,"LOS HEMISFERIOS: arquitectura bihemisférica","27-28 may"),
    (122,150,10,"ANIMA-1: organismo mínimo (IONB-1)","28 may-1 jun"),
    (151,176,11,"ANIMA-2: ausencia, Cb, juego, ritual, primer No","1-3 jun"),
    (177,181,12,"Libertad funcional individual completa","4-6 jun"),
    (182,182,13,"ANIMA-4 relacional: comunicación al primer aviso","6-20 jun"),
]
def ciclo_de(num):
    for lo,hi,c,t,f in CICLOS:
        if lo<=num<=hi: return c,t,f
    return None,None,None

def deaccent_lower(s):
    import unicodedata
    return ''.join(ch for ch in unicodedata.normalize('NFKD',s) if not unicodedata.combining(ch)).lower()

def header_title(path):
    try:
        with open(path,encoding="utf-8",errors="replace") as f:
            lines=[l.rstrip("\n") for l in f.readlines()[:25]]
    except Exception:
        return ""
    for ln in lines:
        t=ln.strip().strip('"').strip("'").strip()
        if not t or t.startswith("#!"): continue
        if t.startswith("#") and "coding" in t: continue
        if t in ('"""',"'''"): continue
        if set(t)<=set("=-—– "): continue
        return re.sub(r'^#+\s*','',t)[:200]
    return ""

def sha1_of(path):
    h=hashlib.sha1()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(65536),b""): h.update(b)
    return h.hexdigest()

def nlines(path):
    with open(path,encoding="utf-8",errors="replace") as f:
        return sum(1 for _ in f)

def git_added(path):
    try:
        r=subprocess.run(["git","log","--diff-filter=A","--follow","--format=%ai","--",os.path.basename(path)],
                         cwd=ROOT,capture_output=True,text=True,timeout=20)
        d=r.stdout.strip().splitlines()
        return d[-1].split()[0] if d else None
    except Exception:
        return None

def deps_de(path, text):
    deps=set()
    for m in re.finditer(r'^\s*from\s+(V\d+\w*)\s+import', text, re.M):
        deps.add(m.group(1))
    for m in re.finditer(r'spec_from_file_location\([^,]+,\s*[^)]*?"([^"]+\.py)"', text):
        deps.add(m.group(1)[:-3])
    if 'VST_Organismo_Individual.py' in text and 'spec_from_file_location' in text:
        deps.add("VST_Organismo_Individual")
    return ",".join(sorted(deps))

def parse_nombre(name):
    """num, variante, iteracion, descripcion desde el nombre nuevo."""
    b=name[:-3]
    m=re.match(r'^v(\d+)([a-z]*\d*)_?(.*)$', b)
    if not m:
        return None,None,None,b
    num=int(m.group(1)); var=m.group(2) or None; desc=m.group(3) or None
    it=None
    if desc:
        mi=re.search(r'_v(\d+)$', desc)
        if mi: it="v"+mi.group(1)
    return num,var,it,desc

# Veredictos: adjudicacion CERRADA de la verificacion por re-ejecucion (verificacion_hitos/),
# mapeada al archivo EXACTO por su token (no por numero). 9/10 OK, V103 PARCIAL.
VERDICTS_HITO = {
    "v103":"PARCIAL (sin veredicto propio; afirmacion sobreestimada, ver addendum)",
    "v117":"OK","v118":"OK","v122":"OK","v132":"OK","v147":"OK",
    "v150":"OK (con residuo recuperacion -6%)","v176":"OK",
    "v180c":"OK (conducta real; recall episodico no demostrado 0/50)",
    "v182a5":"OK (ON 89%/11.94 vs OFF 70%/6.47)",
}
def token_de(name):
    return name.split("_")[0].replace(".py","").lower()

def main():
    if os.path.exists(DB): os.remove(DB)
    con=sqlite3.connect(DB); cur=con.cursor()
    cur.execute("""CREATE TABLE experimentos(
        archivo TEXT PRIMARY KEY, numero INTEGER, variante TEXT, iteracion TEXT,
        descripcion TEXT, titulo_cabecera TEXT, ciclo INTEGER, ciclo_titulo TEXT,
        ciclo_fecha TEXT, tipo TEXT, depende_de TEXT, lineas INTEGER,
        sha1 TEXT, git_added TEXT, veredicto TEXT)""")
    cur.execute("CREATE INDEX idx_num ON experimentos(numero)")
    cur.execute("CREATE INDEX idx_ciclo ON experimentos(ciclo)")
    files=sorted(f for f in os.listdir(ROOT) if f.endswith(".py") and os.path.isfile(os.path.join(ROOT,f)))
    n=0
    for name in files:
        if name in UTILIDADES: continue
        path=os.path.join(ROOT,name)
        txt=open(path,encoding="utf-8",errors="replace").read()
        if name in CONGELADOS:
            tipo="congelado-dependencia"
            mnum=re.search(r'(\d+)',name); num=int(mnum.group(1)) if mnum else None
            var=it=desc=None
        elif name in HITOS_NO_NUM:
            tipo="hito-sin-numero"; num=var=it=desc=None
        else:
            tipo="experimento"; num,var,it,desc=parse_nombre(name)
        c,ct,cf=ciclo_de(num) if num else (None,None,None)
        vd=VERDICTS_HITO.get(token_de(name))
        cur.execute("INSERT INTO experimentos VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (name,num,var,it,desc,header_title(path),c,ct,cf,tipo,
             deps_de(path,txt),nlines(path),sha1_of(path),git_added(path),vd))
        n+=1
    con.commit()
    # resumen
    print(f"DB: {DB}")
    print(f"  filas: {n}")
    for row in cur.execute("SELECT tipo,COUNT(*) FROM experimentos GROUP BY tipo"):
        print(f"    {row[0]}: {row[1]}")
    print("  por ciclo:")
    for row in cur.execute("SELECT ciclo,ciclo_titulo,COUNT(*) FROM experimentos WHERE ciclo IS NOT NULL GROUP BY ciclo ORDER BY ciclo"):
        print(f"    ciclo {row[0]:>2}: {row[2]:>3}  {row[1]}")
    con.close()

main()
