#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analizar.py — Capa de CONSULTA sobre la historia longitudinal de VSTCosmo (DuckDB).

Por qué existe: los CSV de fisiología crecen sin límite (los organismos viven 24/7) y
llegan a cientos de MB. NADIE (ni un humano ni un agente LLM) debe ABRIR esos CSV crudos.
En su lugar se CONSULTAN con SQL y se lee SÓLO el resumen (unas filas), nunca los GB.

DuckDB lee los CSV directo del disco, sin cargarlos en memoria, y une los ~100 archivos
rotados por hora con `union_by_name` (el esquema creció de 193 a 241 columnas con el tiempo;
unir por NOMBRE rellena con NULL las columnas que falten en archivos viejos).

Uso (siempre con el python del venv):
    venv/bin/python Célula_Madre/analisis/analizar.py info
    venv/bin/python Célula_Madre/analisis/analizar.py cols A
    venv/bin/python Célula_Madre/analisis/analizar.py resumen A
    venv/bin/python Célula_Madre/analisis/analizar.py evolucion A Omega --cada hora
    venv/bin/python Célula_Madre/analisis/analizar.py evolucion A R2 --cada minuto --desde "2026-06-26 08:00" --hasta "2026-06-26 10:00"
    venv/bin/python Célula_Madre/analisis/analizar.py sql "SELECT modo_vida, count(*) FROM fisio_A GROUP BY 1"

Vistas SQL ya disponibles en el modo `sql`:  fisio_A , fisio_B  (y fisio_AB = ambas con columna 'org').
La columna ts_real es epoch (segundos); usa  ts  (TIMESTAMP ya convertido) en tus consultas.
"""
from __future__ import annotations
import argparse
import os
import sys

try:
    import duckdb
except ImportError:
    sys.exit("Falta DuckDB. Instálalo en el venv:  venv/bin/pip install duckdb")

# --- Localiza Docker_Historia (raíz de la historia) de forma robusta ---
_AQUI = os.path.dirname(os.path.abspath(__file__))
_DEFECTO = os.path.normpath(os.path.join(_AQUI, "..", "..", "Docker_Historia"))
HIST = os.environ.get("VST_HISTORY_BASE", _DEFECTO)

# Opciones de lectura que toleran el esquema cambiante y filas corruptas sueltas.
_LECTURA = "union_by_name=true, null_padding=true, ignore_errors=true, sample_size=-1"


def _glob(org: str, tipo: str = "fisiologia") -> str:
    return os.path.join(HIST, f"organismo_ANIMA_{org.upper()}", tipo, "*.csv")


def _con():
    """Conexión DuckDB con una vista fisio_<org> por cada organismo con historia
    (A, B, C, D, E) + fisio_TODOS (todos con columna 'org') + fisio_AB (compat)."""
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    presentes = []
    for org in ("A", "B", "C", "D", "E"):
        if not _existe_org(org):   # E vive en la Pi: sincroniza su organismo_ANIMA_E/ a Docker_Historia
            continue
        patron = _glob(org).replace("'", "''")
        # ts: epoch -> TIMESTAMP. Se expone junto al resto de columnas (SELECT *).
        con.execute(f"""
            CREATE VIEW fisio_{org} AS
            SELECT to_timestamp(ts_real) AS ts, *
            FROM read_csv('{patron}', {_LECTURA})
        """)
        presentes.append(org)
    if presentes:
        union = "\n        UNION ALL BY NAME\n".join(
            f"        SELECT '{o}' AS org, * FROM fisio_{o}" for o in presentes)
        con.execute(f"CREATE VIEW fisio_TODOS AS\n{union}")
        if "A" in presentes and "B" in presentes:
            con.execute("CREATE VIEW fisio_AB AS SELECT * FROM fisio_TODOS WHERE org IN ('A','B')")
    return con


def _existe_org(org: str) -> bool:
    import glob as _g
    return bool(_g.glob(_glob(org)))


# Variables fisiológicas clave (rótulo legible -> columna). Se filtran por las que EXISTAN.
_CLAVE = [
    ("Ω (acoplamiento)", "Omega"),
    ("ω_A", "omega_A"), ("ω_B", "omega_B"),
    ("gradiente", "gradiente"),
    ("R2 (razón)", "R2"),
    ("C_m (metabolismo)", "C_m"),
    ("C_b", "C_b"),
    ("H_homeostasis", "H_homeostasis"),
    ("A_sys_env", "A_sys_env"),
    ("presion_desacople", "presion_desacople"),
    ("LF_op", "LF_op"),
    ("XE (exaptación)", "XE"),
    ("OI", "OI"),
    ("energía_L", "energia_L"), ("energía_R", "energia_R"),
]


def _columnas(con, vista: str) -> set[str]:
    return {r[0] for r in con.execute(f"DESCRIBE {vista}").fetchall()}


def _imprimir(con, q, limite, nota_cola="acota con --desde/--hasta o sube --limite"):
    """Imprime el resultado de una consulta como tabla (sin pandas). Devuelve nº de filas."""
    cur = con.execute(q)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    if not rows:
        print("(sin resultados)")
        return 0

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return "—" if v is None else str(v)

    vis = rows[:limite]
    anchos = [max(len(cols[i]), *(len(fmt(r[i])) for r in vis)) for i in range(len(cols))]
    print("  " + "  ".join(cols[i].ljust(anchos[i]) for i in range(len(cols))))
    print("  " + "  ".join("-" * anchos[i] for i in range(len(cols))))
    for r in vis:
        print("  " + "  ".join(fmt(r[i]).ljust(anchos[i]) for i in range(len(cols))))
    if len(rows) > limite:
        print(f"\n  … {len(rows) - limite} filas más ({nota_cola})")
    return len(rows)


def cmd_info(_args):
    con = _con()
    print(f"Historia: {HIST}\n")
    for org in ("A", "B"):
        if not _existe_org(org):
            print(f"  ANIMA_{org}: (sin archivos de fisiología)")
            continue
        r = con.execute(f"""
            SELECT count(*) AS filas,
                   min(ts) AS desde, max(ts) AS hasta,
                   count(DISTINCT modo_vida) AS modos
            FROM fisio_{org}
        """).fetchone()
        import glob as _g
        nfiles = len(_g.glob(_glob(org)))
        print(f"  ANIMA_{org}:  {r[0]:,} filas  ·  {nfiles} archivos")
        print(f"             {r[1]}  →  {r[2]}")
    print("\nVistas SQL: fisio_A, fisio_B, fisio_AB  ·  usa la columna 'ts' (TIMESTAMP).")


def cmd_cols(args):
    con = _con()
    org = args.organismo.upper()
    cols = sorted(_columnas(con, f"fisio_{org}"))
    print(f"fisio_{org}: {len(cols)} columnas (unión de todos los archivos)\n")
    # imprime en columnas de 3
    for i in range(0, len(cols), 3):
        print("   " + "".join(f"{c:<28}" for c in cols[i:i + 3]))


def cmd_resumen(args):
    con = _con()
    org = args.organismo.upper()
    disp = _columnas(con, f"fisio_{org}")
    usar = [(lbl, c) for lbl, c in _CLAVE if c in disp]
    if not usar:
        sys.exit("Ninguna variable clave presente en los datos.")
    sel = ", ".join(
        f"avg({c}) AS a_{c}, min({c}) AS mn_{c}, max({c}) AS mx_{c}, stddev_samp({c}) AS sd_{c}"
        for _, c in usar
    )
    n = con.execute(f"SELECT count(*) FROM fisio_{org}").fetchone()[0]
    fila = con.execute(f"SELECT {sel} FROM fisio_{org}").fetchone()
    cols = [d[0] for d in con.description]
    val = dict(zip(cols, fila))
    print(f"Resumen fisiológico ANIMA_{org}  ·  {n:,} filas\n")
    print(f"  {'variable':<22}{'media':>12}{'min':>12}{'max':>12}{'desv':>12}")
    print("  " + "-" * 70)
    for lbl, c in usar:
        def f(x):
            return f"{x:12.4f}" if isinstance(x, (int, float)) and x is not None else f"{'—':>12}"
        print(f"  {lbl:<22}{f(val.get('a_'+c))}{f(val.get('mn_'+c))}{f(val.get('mx_'+c))}{f(val.get('sd_'+c))}")


def cmd_evolucion(args):
    con = _con()
    org = args.organismo.upper()
    var = args.variable
    if var not in _columnas(con, f"fisio_{org}"):
        sys.exit(f"La variable '{var}' no existe en fisio_{org}. Mira las disponibles con:  analizar.py cols {org}")
    bucket = {"minuto": "minute", "hora": "hour", "dia": "day", "segundo": "second"}.get(args.cada, "hour")
    donde = []
    if args.desde:
        donde.append(f"ts >= TIMESTAMP '{args.desde}'")
    if args.hasta:
        donde.append(f"ts < TIMESTAMP '{args.hasta}'")
    w = ("WHERE " + " AND ".join(donde)) if donde else ""
    q = f"""
        SELECT date_trunc('{bucket}', ts) AS t,
               count(*) AS n,
               avg({var}) AS media, min({var}) AS minimo, max({var}) AS maximo
        FROM fisio_{org} {w}
        GROUP BY 1 ORDER BY 1
    """
    print(f"Evolución de '{var}' en ANIMA_{org}  ·  por {args.cada}\n")
    _imprimir(con, q, args.limite, nota_cola="acota con --desde/--hasta o sube --limite")


def cmd_sql(args):
    con = _con()
    _imprimir(con, args.query, args.limite, nota_cola="limita en tu SQL o sube --limite")


# ---------------------------------------------------------------------------
# EXPORT: analiza un CSV DESCARGADO desde la página del organismo (botón ⬇ CSV).
# Esos archivos traen un encabezado de líneas '#' (metadatos + bitácora completa)
# antes de los datos, y usan 't' (segundos de VIDA) como eje temporal, no ts_real.
# ---------------------------------------------------------------------------
def _skip_comentarios(ruta: str) -> int:
    """Cuenta las líneas iniciales que empiezan con '#' (metadatos + bitácora)."""
    n = 0
    with open(ruta, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                n += 1
            else:
                break
    return n


def _con_archivo(ruta: str):
    """Conexión DuckDB con la vista 'd' sobre un CSV exportado (salta los '#')."""
    if not os.path.isfile(ruta):
        sys.exit(f"No existe el archivo: {ruta}")
    skip = _skip_comentarios(ruta)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    patron = ruta.replace("'", "''")
    con.execute(f"""
        CREATE VIEW d AS
        SELECT * FROM read_csv('{patron}', skip={skip}, header=true,
                               null_padding=true, ignore_errors=true)
    """)
    return con


# --- Ideas integradas de la síntesis de Gordon: rollup + anomalías + HTML ---
def _rollup_export(con, seg, varlist):
    """Síntesis por ventana de `seg` s de vida: avg de cada variable. Devuelve (ts, {var:media})."""
    sel = ", ".join(f"round(avg({c}),4) AS {c}" for c in varlist)
    q = f"SELECT (floor(t/{seg})*{seg})::INT AS t_s, {sel} FROM d GROUP BY 1 ORDER BY 1"
    cur = con.execute(q)
    names = [d[0] for d in cur.description]
    rows = [dict(zip(names, r)) for r in cur.fetchall()]
    return rows


def _anomalias_export(con, varlist, seg=10):
    """Detecta (sobre el rollup): picos (sube >1.5× y baja), caídas (<0.5×) y saltos de etiqueta de voz."""
    roll = _rollup_export(con, seg, varlist)
    anom = []
    for i in range(1, len(roll) - 1):
        for c in varlist:
            a, v, s = roll[i - 1].get(c), roll[i].get(c), roll[i + 1].get(c)
            if None in (a, v, s):
                continue
            if a and v > 1.5 * a and v > 1.5 * s:
                anom.append({"tipo": "pico", "var": c, "t_s": roll[i]["t_s"], "valor": round(v, 4), "previo": round(a, 4)})
            if a and abs(a) > 1e-2 and abs(v) < 0.5 * abs(a):
                anom.append({"tipo": "caida", "var": c, "t_s": roll[i]["t_s"], "valor": round(v, 4), "previo": round(a, 4)})
    # saltos de etiqueta de voz dominante por ventana
    if "voz_emitida" in _columnas(con, "d"):
        q = (f"SELECT t_s, arg_max(voz_emitida, n) FROM (SELECT (floor(t/{seg})*{seg})::INT t_s, voz_emitida, "
             f"count(*) n FROM d WHERE voz_emitida<>'-' GROUP BY 1,2) GROUP BY 1 ORDER BY 1")
        prev = None
        for t_s, dom in con.execute(q).fetchall():
            if prev is not None and dom != prev:
                anom.append({"tipo": "salto_etiqueta", "t_s": t_s, "de": prev, "a": dom})
            prev = dom
    return anom, roll


def _html_export(con, ruta_salida, nombre, disp):
    varlist = [c for _, c in _CLAVE if c in disp]
    anom, roll = _anomalias_export(con, varlist)
    n, tmin, tmax = con.execute("SELECT count(*), min(t), max(t) FROM d").fetchone()
    series = {c: [r.get(c) for r in roll] for c in varlist}
    labels = [r["t_s"] for r in roll]
    import json as _json
    canvases = "".join(f'<div class=cc><canvas id="c_{c}"></canvas></div>' for c in varlist)
    charts_js = "".join(
        f"new Chart(document.getElementById('c_{c}'),{{type:'line',data:{{labels:L,datasets:[{{label:'{c}',"
        f"data:{_json.dumps(series[c])},borderColor:'#e8b86d',borderWidth:1.5,pointRadius:0}}]}},"
        f"options:{{animation:false,responsive:true,maintainAspectRatio:false,"
        f"plugins:{{legend:{{labels:{{color:'#dfe7f0'}}}}}},scales:{{x:{{ticks:{{color:'#6b7d92'}}}},"
        f"y:{{ticks:{{color:'#6b7d92'}}}}}}}}}});" for c in varlist)
    filas_anom = "".join(
        f"<tr><td>{a['tipo']}</td><td>{', '.join(f'{k}={v}' for k,v in a.items() if k!='tipo')}</td></tr>"
        for a in anom[:60])
    html = f"""<!DOCTYPE html><html lang=es><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Síntesis · {nombre}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>body{{margin:18px;background:#0a0e14;color:#dfe7f0;font-family:monospace}}
h1{{color:#e8b86d}}.cc{{height:170px;margin:6px 0 22px}}
table{{border-collapse:collapse;width:100%;margin-top:10px}}td,th{{border:1px solid #243246;padding:5px;font-size:12px}}
th{{background:#121925;color:#e8b86d;text-align:left}}.mut{{color:#8aa0b8}}</style></head><body>
<h1>📊 Síntesis de fisiología</h1>
<p class=mut>Archivo: <b>{nombre}</b> · {n:,} filas · vida {tmin:.1f}s → {tmax:.1f}s (~{(tmax-tmin)/60:.1f} min)
· síntesis por ventanas de 10s · {len(varlist)} variables · {len(anom)} anomalías</p>
<p class=mut>⚠ Las etiquetas de voz son RÓTULOS, no significados.</p>
{canvases}
<h2>Anomalías detectadas ({len(anom)})</h2>
<table><tr><th>tipo</th><th>detalle</th></tr>{filas_anom}</table>
<script>const L={_json.dumps(labels)};{charts_js}</script></body></html>"""
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(html)
    return ruta_salida, len(anom)


def cmd_export(args):
    con = _con_archivo(args.ruta)
    disp = _columnas(con, "d")

    if args.html:
        ruta, n = _html_export(con, args.html, os.path.basename(args.ruta), disp)
        print(f"✓ informe HTML: {ruta}  ({n} anomalías detectadas)")
        return

    if args.anomalias:
        varlist = [c for _, c in _CLAVE if c in disp]
        anom, _ = _anomalias_export(con, varlist)
        print(f"Anomalías detectadas: {len(anom)} (picos, caídas, saltos de etiqueta)\n")
        for a in anom[:args.limite]:
            print("  " + a["tipo"].upper() + " · " + ", ".join(f"{k}={v}" for k, v in a.items() if k != "tipo"))
        if len(anom) > args.limite:
            print(f"\n  … {len(anom)-args.limite} más (sube --limite)")
        return

    if args.sql:
        _imprimir(con, args.sql, args.limite, nota_cola="limita en tu SQL o sube --limite")
        return

    if args.etiquetas:
        col = args.etiquetas
        if col not in disp:
            sys.exit(f"La columna '{col}' no existe. Mira las columnas con --sql \"DESCRIBE d\".")
        print(f"Frecuencia de etiquetas en '{col}'  (¡son RÓTULOS, no significados!)\n")
        _imprimir(con, f"SELECT {col} AS etiqueta, count(*) AS n, "
                       f"round(100.0*count(*)/sum(count(*)) OVER (),1) AS pct "
                       f"FROM d WHERE {col} IS NOT NULL AND {col} <> '-' "
                       f"GROUP BY 1 ORDER BY n DESC", args.limite, nota_cola="sube --limite")
        return

    if args.evolucion:
        var = args.evolucion
        if var not in disp:
            sys.exit(f"La variable '{var}' no existe. Mira las columnas con --sql \"DESCRIBE d\".")
        seg = max(1, args.cada)
        print(f"Evolución de '{var}' por tramos de {seg}s de vida\n")
        _imprimir(con,
                  f"SELECT (floor(t/{seg})*{seg})::INT AS t_vida_s, count(*) AS n, "
                  f"round(avg({var}),4) AS media, round(min({var}),4) AS minimo, "
                  f"round(max({var}),4) AS maximo "
                  f"FROM d GROUP BY 1 ORDER BY 1", args.limite,
                  nota_cola="sube --limite o usa --cada con tramos más grandes")
        return

    # Por defecto: panorama del archivo
    n, tmin, tmax = con.execute("SELECT count(*), min(t), max(t) FROM d").fetchone()
    print(f"Archivo: {os.path.basename(args.ruta)}")
    print(f"  {n:,} filas · {len(disp)} columnas · vida {tmin:.1f}s → {tmax:.1f}s "
          f"(~{(tmax - tmin) / 60:.1f} min)\n")
    usar = [(lbl, c) for lbl, c in _CLAVE if c in disp]
    if usar:
        sel = ", ".join(f"avg({c}) AS a_{c}, min({c}) AS mn_{c}, max({c}) AS mx_{c}" for _, c in usar)
        fila = con.execute(f"SELECT {sel} FROM d").fetchone()
        val = dict(zip([d2[0] for d2 in con.description], fila))
        print(f"  {'variable':<22}{'media':>12}{'min':>12}{'max':>12}")
        print("  " + "-" * 58)
        for lbl, c in usar:
            def f(x):
                return f"{x:12.4f}" if isinstance(x, (int, float)) and x is not None else f"{'—':>12}"
            print(f"  {lbl:<22}{f(val.get('a_'+c))}{f(val.get('mn_'+c))}{f(val.get('mx_'+c))}")
    if "voz_emitida" in disp:
        print(f"\n  Etiquetas de voz más frecuentes (RÓTULOS, no significados):")
        for r in con.execute("SELECT voz_emitida, count(*) n FROM d "
                             "WHERE voz_emitida IS NOT NULL AND voz_emitida <> '-' "
                             "GROUP BY 1 ORDER BY n DESC LIMIT 8").fetchall():
            print(f"    {r[0]:<22} {r[1]:>7,}")


def main():
    p = argparse.ArgumentParser(
        description="Consulta la historia longitudinal de VSTCosmo sin abrir CSV gigantes (DuckDB).")
    p.add_argument("--limite", type=int, default=60, help="máx. filas a imprimir (def: 60)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="resumen: filas, archivos y rango temporal por organismo")

    c = sub.add_parser("cols", help="lista las columnas disponibles de un organismo")
    c.add_argument("organismo", help="A o B")

    r = sub.add_parser("resumen", help="estadísticos (media/min/max/desv) de variables clave")
    r.add_argument("organismo", help="A o B")

    e = sub.add_parser("evolucion", help="evolución temporal de una variable (avg/min/max por tramo)")
    e.add_argument("organismo", help="A o B")
    e.add_argument("variable", help="nombre de columna (ej. Omega, R2, C_m)")
    e.add_argument("--cada", default="hora", choices=["segundo", "minuto", "hora", "dia"],
                   help="granularidad del tramo (def: hora)")
    e.add_argument("--desde", help="inicio, ej. '2026-06-26 08:00'")
    e.add_argument("--hasta", help="fin, ej. '2026-06-26 10:00'")

    s = sub.add_parser("sql", help="ejecuta SQL libre sobre fisio_A / fisio_B / fisio_AB")
    s.add_argument("query", help="consulta SQL (entre comillas)")

    x = sub.add_parser("export", help="analiza un CSV DESCARGADO de la página (salta el encabezado # automáticamente)")
    x.add_argument("ruta", help="ruta al .csv descargado (entre comillas si tiene espacios)")
    x.add_argument("--evolucion", metavar="VAR", help="evolución de una variable por tramos de vida")
    x.add_argument("--cada", type=int, default=60, help="tamaño del tramo en segundos de vida (def: 60)")
    x.add_argument("--etiquetas", metavar="COL", help="frecuencia de etiquetas de una columna (ej. voz_emitida)")
    x.add_argument("--html", metavar="RUTA", help="genera un informe HTML (gráficos + anomalías) en RUTA")
    x.add_argument("--anomalias", action="store_true", help="lista anomalías (picos, caídas, saltos de etiqueta)")
    x.add_argument("--sql", help="SQL libre sobre la vista 'd' (este archivo)")
    x.add_argument("--limite", type=int, default=60, help="máx. filas a imprimir (def: 60)")

    args = p.parse_args()
    {"info": cmd_info, "cols": cmd_cols, "resumen": cmd_resumen,
     "evolucion": cmd_evolucion, "sql": cmd_sql, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    main()
