"""
CS057 — genera el artifact visual del paisaje (HTML autocontenido con SVG inline) desde cs057_paisaje.csv.
Sin matplotlib (CSP de artifact bloquea CDNs): SVG hecho a mano desde los datos REALES. Tema claro/oscuro.
"""
import pandas as pd, numpy as np, os, html

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "cs057_paisaje.csv"))
CL = ["d1", "d2", "d3", "d4", "curv"]
ETIQ = {"d1": "1D", "d2": "2D", "d3": "3D·plano", "d4": "4D", "curv": "curvo"}
df["vt"] = df[[f"viable_{c}" for c in CL]].sum(axis=1)

# ---- agregados ----
sob = df[df.phys == 0]
fis = df[df.phys == 1]
den = df[df.phys == 2]
viable_dim = {c: float(sob[f"viable_{c}"].mean()) for c in CL}
sync = {c: float(sob[sob.arm == "sync"][f"viable_{c}"].mean()) for c in CL}
asyn = {c: float(sob[sob.arm == "async"][f"viable_{c}"].mean()) for c in CL}
acc_dim = {c: float(sob[f"acelera_{c}"].mean()) for c in CL}
fis_dim = {c: float(fis[f"viable_{c}"].mean()) for c in CL}
den_dim = {c: float(den[f"viable_{c}"].mean()) for c in CL}
s = sob[sob.arm == "sync"]["vt"]; a = sob[sob.arm == "async"]["vt"]
z = (s.mean() - a.mean()) / np.sqrt(s.var() / len(s) + a.var() / len(a))

# ---- heatmap w_grav (x) × w_exp (y) → viabilidad media (el MAPA) ----
NB = 20
gx = np.clip((sob.w_grav * NB).astype(int), 0, NB - 1)
gy = np.clip((sob.w_exp * NB).astype(int), 0, NB - 1)
heat = np.full((NB, NB), np.nan)
cnt = np.zeros((NB, NB))
acc = np.zeros((NB, NB))
for xi, yi, v in zip(gx, gy, sob["vt"]):
    acc[yi, xi] += v; cnt[yi, xi] += 1
heat = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
vmax = np.nanmax(heat)

# ---------- helpers SVG ----------
def lerp(a, b, t):
    return a + (b - a) * t

def col_viab(t):  # 0→fondo tenue, 1→cian luminoso
    t = max(0.0, min(1.0, t))
    r = lerp(28, 79, t); g = lerp(32, 214, t); b = lerp(48, 200, t)
    return f"rgb({r:.0f},{g:.0f},{b:.0f})"

def barras(datos, etiqs, colca, vmax=None, w=520, h=210, pad=38, dest=None, marca=None):
    vmax = vmax or max(datos) * 1.15 or 1
    n = len(datos); bw = (w - 2 * pad) / n * 0.62
    gap = (w - 2 * pad) / n
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" class="fig">']
    # ejes y
    for gv in [0, 0.25, 0.5, 0.75, 1.0]:
        y = h - pad - gv * (h - 2 * pad) / (vmax / (vmax or 1))
        yy = h - pad - (gv * vmax / vmax) * (h - 2 * pad) if False else h - pad - gv * (h - 2 * pad)
    for i, (v, e) in enumerate(zip(datos, etiqs)):
        x = pad + i * gap + (gap - bw) / 2
        bh = (v / vmax) * (h - 2 * pad)
        y = h - pad - bh
        c = colca[i] if isinstance(colca, list) else colca
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="2" fill="{c}"/>')
        out.append(f'<text x="{x+bw/2:.1f}" y="{h-pad+16:.1f}" class="lbl" text-anchor="middle">{e}</text>')
        out.append(f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" class="val" text-anchor="middle">{v:.3f}</text>')
    out.append(f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" class="axis"/>')
    out.append('</svg>')
    return "\n".join(out)

def barras_pareja(d1, d2, etiqs, c1, c2, w=560, h=230, pad=40):
    vmax = max(max(d1), max(d2)) * 1.18 or 1
    n = len(d1); grp = (w - 2 * pad) / n; bw = grp * 0.30
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" class="fig">']
    for i in range(n):
        x0 = pad + i * grp + grp / 2
        for j, (v, c) in enumerate([(d1[i], c1), (d2[i], c2)]):
            x = x0 + (j - 1) * bw - bw * 0.05 + (bw * 0.1)
            bh = (v / vmax) * (h - 2 * pad); y = h - pad - bh
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="2" fill="{c}"/>')
        out.append(f'<text x="{x0:.1f}" y="{h-pad+16:.1f}" class="lbl" text-anchor="middle">{etiqs[i]}</text>')
    out.append(f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" class="axis"/>')
    out.append('</svg>')
    return "\n".join(out)

def heatmap(H, vmax, w=430, h=430, pad=46):
    NB = H.shape[0]; cell = (w - 2 * pad) / NB
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" class="fig heat">']
    for yi in range(NB):
        for xi in range(NB):
            v = H[yi, xi]
            x = pad + xi * cell; y = pad + (NB - 1 - yi) * cell
            if np.isnan(v):
                fill = "var(--cell-empty)"
            else:
                fill = col_viab(v / vmax)
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell+0.6:.1f}" height="{cell+0.6:.1f}" fill="{fill}"/>')
    # punto físico: w_grav≈0 (izq), w_exp=0.5 (medio)
    px = pad + 0.02 * (w - 2 * pad); py = pad + (1 - 0.5) * (h - 2 * pad)
    out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="none" stroke="var(--amber)" stroke-width="2.5"/>')
    out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.5" fill="var(--amber)"/>')
    out.append(f'<text x="{px+12:.1f}" y="{py-8:.1f}" class="mk">punto físico</text>')
    # ejes
    out.append(f'<text x="{w/2:.1f}" y="{h-10:.1f}" class="axl" text-anchor="middle">gravedad  →</text>')
    out.append(f'<text x="14" y="{h/2:.1f}" class="axl" text-anchor="middle" transform="rotate(-90 14 {h/2:.1f})">expansión  →</text>')
    out.append('</svg>')
    return "\n".join(out)

# leyenda gradiente
def leyenda(vmax):
    stops = "".join(f'<stop offset="{o}%" stop-color="{col_viab(o/100)}"/>' for o in range(0, 101, 10))
    return f'''<svg viewBox="0 0 220 44" class="leg"><defs><linearGradient id="g" x1="0" x2="1">{stops}</linearGradient></defs>
    <rect x="6" y="6" width="208" height="12" rx="3" fill="url(#g)"/>
    <text x="6" y="34" class="lbl" text-anchor="start">0</text>
    <text x="214" y="34" class="lbl" text-anchor="end">{vmax:.2f} viab. media</text></svg>'''

COLS = ["#5b6172", "#6aa9c9", "#4fd6c8", "#57c8a8", "#8f7fd6"]  # por dim, sobrio

FIG1 = barras([viable_dim[c] for c in CL], [ETIQ[c] for c in CL], COLS)
FIG2 = barras_pareja([sync[c] for c in CL], [asyn[c] for c in CL], [ETIQ[c] for c in CL], "#4fd6c8", "#6a6f82")
FIG3 = heatmap(heat, vmax)
FIG_LEG = leyenda(vmax)
FIG4 = barras_pareja([den_dim[c] for c in CL], [viable_dim[c] for c in CL], [ETIQ[c] for c in CL], "#f5b545", "#5b6172")
FIG5 = barras([acc_dim[c] for c in CL], [ETIQ[c] for c in CL], ["#8f7fd6"] * 5)

# ---------- HTML ----------
PAGE = f"""<title>CS057 — El paisaje de universos posibles</title>
<style>
:root {{
  --ground:#0b0d14; --panel:#12151f; --ink:#e8eaf2; --muted:#9aa0b4; --faint:#5b6172;
  --line:#232838; --cyan:#4fd6c8; --amber:#f5b545; --violet:#8f7fd6; --cell-empty:#151925;
  --serif:'Iowan Old Style','Palatino Linotype',Palatino,Georgia,serif;
  --sans:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --mono:ui-monospace,'SF Mono',Menlo,'Cascadia Code',monospace;
}}
@media (prefers-color-scheme: light) {{
  :root {{ --ground:#f6f5f2; --panel:#ffffff; --ink:#1a1d29; --muted:#5a6070; --faint:#9aa0b4;
    --line:#e4e2dc; --cell-empty:#eeece7; }}
}}
:root[data-theme="dark"] {{ --ground:#0b0d14; --panel:#12151f; --ink:#e8eaf2; --muted:#9aa0b4; --faint:#5b6172; --line:#232838; --cell-empty:#151925; }}
:root[data-theme="light"] {{ --ground:#f6f5f2; --panel:#ffffff; --ink:#1a1d29; --muted:#5a6070; --faint:#9aa0b4; --line:#e4e2dc; --cell-empty:#eeece7; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink); font-family:var(--sans);
  line-height:1.6; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:820px; margin:0 auto; padding:clamp(28px,5vw,72px) clamp(20px,4vw,40px); }}
.eyebrow {{ font-family:var(--mono); font-size:12px; letter-spacing:.18em; text-transform:uppercase;
  color:var(--cyan); margin:0 0 14px; }}
h1 {{ font-family:var(--serif); font-weight:600; font-size:clamp(30px,5vw,46px); line-height:1.1;
  margin:0 0 18px; text-wrap:balance; letter-spacing:-.01em; }}
.lede {{ font-size:clamp(17px,2.3vw,20px); color:var(--muted); margin:0 0 8px; text-wrap:pretty; max-width:62ch; }}
.stats {{ display:flex; flex-wrap:wrap; gap:28px; margin:36px 0 10px; padding:22px 0; border-top:1px solid var(--line); border-bottom:1px solid var(--line); }}
.stat b {{ font-family:var(--mono); font-size:26px; font-weight:600; display:block; letter-spacing:-.02em; font-variant-numeric:tabular-nums; }}
.stat span {{ font-size:13px; color:var(--muted); }}
.stat.hl b {{ color:var(--amber); }}
section {{ margin:52px 0; }}
h2 {{ font-family:var(--serif); font-weight:600; font-size:clamp(21px,3vw,27px); margin:0 0 6px; letter-spacing:-.01em; }}
.tag {{ font-family:var(--mono); font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--faint); }}
p {{ max-width:64ch; }}
p.note {{ color:var(--muted); font-size:15px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:clamp(18px,3vw,30px); margin:18px 0; }}
.fig {{ width:100%; height:auto; display:block; }}
.fig .lbl {{ font-family:var(--mono); font-size:12px; fill:var(--muted); }}
.fig .val {{ font-family:var(--mono); font-size:11px; fill:var(--ink); font-variant-numeric:tabular-nums; }}
.fig .axis {{ stroke:var(--line); stroke-width:1; }}
.heat .axl {{ font-family:var(--mono); font-size:12px; fill:var(--muted); letter-spacing:.05em; }}
.heat .mk {{ font-family:var(--mono); font-size:11px; fill:var(--amber); }}
.leg {{ width:220px; height:44px; }} .leg .lbl {{ font-family:var(--mono); font-size:11px; fill:var(--muted); }}
.legend-row {{ display:flex; gap:20px; align-items:center; flex-wrap:wrap; margin-top:8px; }}
.key {{ display:flex; gap:7px; align-items:center; font-size:13px; color:var(--muted); font-family:var(--mono); }}
.key i {{ width:13px; height:13px; border-radius:3px; display:inline-block; }}
.verdict {{ border-left:3px solid var(--amber); padding:4px 0 4px 20px; margin:20px 0; }}
.verdict strong {{ color:var(--amber); }}
.good {{ color:var(--cyan); }}
table {{ width:100%; border-collapse:collapse; font-family:var(--mono); font-size:13px; margin-top:6px; font-variant-numeric:tabular-nums; }}
th,td {{ text-align:right; padding:7px 10px; border-bottom:1px solid var(--line); }}
th:first-child,td:first-child {{ text-align:left; color:var(--muted); }}
tr.mark td {{ color:var(--amber); }}
.foot {{ margin-top:60px; padding-top:24px; border-top:1px solid var(--line); color:var(--faint); font-size:13px; }}
.foot code {{ font-family:var(--mono); color:var(--muted); }}
</style>

<div class="wrap">
  <p class="eyebrow">Cosmosemiótica · CS057 · Claude Science × Alexis López Tapia</p>
  <h1>El paisaje de los universos posibles</h1>
  <p class="lede">69.648 universos simulados. Se barrieron las seis fuerzas de 0 a 1 y se preguntó, ciego:
  ¿qué combinaciones estabilizan un universo que persiste <em>y se expande</em> — de la dimensión que sea?
  Nuestro universo es un punto del mapa, no el objetivo.</p>

  <div class="stats">
    <div class="stat"><b>4 353</b><span>combinaciones de fuerzas (Sobol)</span></div>
    <div class="stat"><b>× 8 × 2</b><span>semillas × brazos (sync/async)</span></div>
    <div class="stat hl"><b>0.47</b><span>viabilidad en el punto físico</span></div>
    <div class="stat hl"><b>curvo</b><span>la geometría que estabiliza</span></div>
  </div>

  <section>
    <span class="tag">El titular</span>
    <h2>El punto físico es viable — pero curvo, no 3D-plano</h2>
    <p>Con las constantes reales (confinamiento fuerte, gravedad despreciable, expansión moderada) el punto
    físico cae en una zona <span class="good">muy viable</span>: 0.47 en su vecindad densa contra 0.17 del
    promedio. Nuestras fuerzas <em>sí</em> hacen universos que perduran y se expanden. Pero la geometría que
    estabilizan es la <strong style="color:var(--amber)">curva/hiperbólica</strong>, no la 3D-plana.</p>
    <div class="card">
      {FIG4}
      <div class="legend-row">
        <div class="key"><i style="background:var(--amber)"></i>vecindad del punto físico</div>
        <div class="key"><i style="background:#5b6172"></i>promedio global</div>
      </div>
    </div>
    <table>
      <tr><th>región</th><th>n</th><th>viab.</th><th>3D·plano</th><th>curvo</th></tr>
      <tr><td>global (Sobol)</td><td>65 536</td><td>0.17</td><td>0.040</td><td>0.064</td></tr>
      <tr class="mark"><td>punto físico exacto</td><td>16</td><td>0.75</td><td>0.000</td><td>0.688</td></tr>
      <tr class="mark"><td>vecindad densa</td><td>4 096</td><td>0.47</td><td>0.058</td><td>0.315</td></tr>
    </table>
    <div class="verdict"><strong>Falsación acotada:</strong> las fuerzas locales reales, todas juntas y
    barridas, <strong>no seleccionan el 3D-plano</strong> — favorecen lo curvo-expansivo. La unicidad de
    nuestro 3D-plano no la fija ninguna fuerza local; apunta aguas arriba (espín/marco, R7).</div>
  </section>

  <section>
    <span class="tag">El mapa</span>
    <h2>Dónde vive la viabilidad en el espacio de fuerzas</h2>
    <p>Viabilidad media proyectada sobre los dos ejes que más la deciden: <em>gravedad</em> (horizontal) y
    <em>expansión</em> (vertical). Lo luminoso = universos que prenden. La viabilidad se enciende con
    <strong>poca gravedad y mucha expansión</strong>; la gravedad la apaga.</p>
    <div class="card" style="display:flex; gap:24px; flex-wrap:wrap; align-items:center; justify-content:center;">
      {FIG3}
      {FIG_LEG}
    </div>
    <p class="note">El anillo ámbar marca el punto físico (gravedad ≈ 0). El cuello de botella en todo el
    paisaje no es persistir — casi todo se mantiene estable — sino <em>expandir</em> sin disolverse.</p>
  </section>

  <section>
    <span class="tag">Falsación del "es un proceso"</span>
    <h2>La simultaneidad ayuda — modesto pero robusto</h2>
    <p>El brazo <em>sincrónico</em> (las cuatro fuerzas juntas cada paso) contra el <em>asincrónico</em>
    (por turnos, nunca a la vez), con la dosis total de cada fuerza igualada. El sincrónico estabiliza más
    universos, en todas las dimensiones.</p>
    <div class="card">
      {FIG2}
      <div class="legend-row">
        <div class="key"><i style="background:var(--cyan)"></i>sincrónico (juntas)</div>
        <div class="key"><i style="background:#6a6f82"></i>asincrónico (por turnos)</div>
      </div>
    </div>
    <p class="note">Diferencia +0.023 ± 0.005 → <strong style="color:var(--ink)">z = {z:.1f}</strong>. Pequeña
    (~13% relativo) pero de varios sigma: la tesis de que <em>es un proceso, no una sucesión</em> se sostiene
    en su versión sobria. (El confound de dosis que inflaba async a 8× fue detectado y corregido antes de esta
    tanda.)</p>
  </section>

  <section>
    <span class="tag">Geometría</span>
    <h2>La viabilidad crece con la dimensión</h2>
    <p>Fracción de combinaciones que estabilizan un universo en expansión, por geometría de partida. La cadena
    1D nunca; las geometrías más conectadas prenden más fácil.</p>
    <div class="card">{FIG1}</div>
  </section>

  <section>
    <span class="tag">Sector oscuro · emergente, no insertado</span>
    <h2>La expansión que se acelera sola</h2>
    <p>El 7% de las combinaciones producen expansión que <em>acelera</em> (segunda diferencia del diámetro
    positiva) sin que se metiera ningún término de aceleración. De lo que acelera, es un universo viable el
    94–99% de las veces — y es <strong style="color:var(--amber)">3.5× más común en el punto físico</strong>
    (0.25 vs 0.07). Candidato honesto a un análogo de energía oscura, localizado cerca del valor real.</p>
    <div class="card">{FIG5}</div>
  </section>

  <div class="foot">
    <p><strong>Método.</strong> Muestreo Sobol de baja discrepancia sobre el hipercubo [0,1]⁷ de las seis
    fuerzas + eje de alcance; punto físico marcado y su vecindad resuelta densa. Toda distancia por saltos de
    grafo (nunca coordenada). Criterios <em>estable</em>/<em>expande</em>/<em>acelera</em> medidos ciegos a los
    pesos y a la dimensión. Sector oscuro solo como salida medida, jamás como término de entrada.</p>
    <p>Diseño CS057 de Claude Science; planteo físico y método de Alexis López Tapia; implementación, guardianes
    y auditoría de CC. Datos: <code>cs057_paisaje.csv</code> (69.648 corridas). Informe: <code>INFORME_CS057_PARA_CS.md</code>.</p>
  </div>
</div>
"""
open(os.path.join(HERE, "cs057_paisaje.html"), "w").write(PAGE)
print("escrito cs057_paisaje.html  ·  z=%.2f  vmax_heat=%.3f  fis_curv=%.3f den_curv=%.3f" %
      (z, vmax, fis_dim["curv"], den_dim["curv"]))
