# -*- coding: utf-8 -*-
import csv, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import numpy as np

DATA = os.path.expanduser("/Users/alexis/Downloads/ANIMA4_ESTRES_20260630_081345")
FIG = "figuras"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 16, "axes.titleweight": "bold",
                     "figure.dpi": 140, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
AZUL="#2c6fbb"; VERDE="#3aa66f"; ROJO="#d6553f"; AMBAR="#e8a13a"; GRIS="#7d8ea6"; MOR="#7b5cc4"

def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))
def num(x):
    try: return float(x)
    except: return None

# ---------- FIG 1: gradiente de estrés (W por ruteo) ----------
modos = {r["modo"]: r for r in load("por_modo.csv")}
orden = ["PAR","L","R","BOTH"]; etiq={"PAR":"PAR\n(se oyen\nentre ellos)","L":"L\n(audio 1 oído)","R":"R\n(audio 1 oído)","BOTH":"BOTH\n(audio 2 oídos)"}
W=[num(modos[m]["W"]) for m in orden]
cols=[VERDE,AZUL,AZUL,ROJO]
fig,ax=plt.subplots(figsize=(8,5))
b=ax.bar([etiq[m] for m in orden], W, color=cols, edgecolor="white", linewidth=1.5)
for bar,w in zip(b,W): ax.text(bar.get_x()+bar.get_width()/2, w+0.004, f"{w:.3f}", ha="center", fontweight="bold")
ax.set_ylabel("Bienestar  W  (0=mal, 1=pleno)"); ax.set_ylim(0,0.34)
ax.set_title("Cuánto se 'siente bien' el organismo según cómo entra el sonido")
ax.axhline(min(W), ls=":", c=GRIS, alpha=.5)
fig.text(0.5,-0.04,"Oírse entre ellos (PAR) es el estado más calmo; el bombardeo por ambos oídos (BOTH) es el más aversivo.",
         ha="center", style="italic", color=GRIS, fontsize=11)
fig.savefig(os.path.join(FIG,"fig1_gradiente_estres.png")); plt.close(fig)

# ---------- FIG 2: reacción por categoría de sonido ----------
cats=load("por_categoria.csv")
cats=[c for c in cats if num(c["W"]) is not None]
cats.sort(key=lambda c:num(c["W"]))
names=[c["categoria"] for c in cats]; Wc=[num(c["W"]) for c in cats]
fig,ax=plt.subplots(figsize=(9,5))
b=ax.barh(names, Wc, color=[ROJO if w<0.2 else (AMBAR if w<0.25 else VERDE) for w in Wc], edgecolor="white")
for bar,w in zip(b,Wc): ax.text(w+0.003, bar.get_y()+bar.get_height()/2, f"{w:.3f}", va="center", fontsize=11)
ax.set_xlabel("Bienestar  W"); ax.set_title("Qué tipos de sonido los hacen sentir mejor o peor")
fig.text(0.5,-0.03,"Las frecuencias puras y los tonos sostenidos estresan más; texturas/voz, menos.",
         ha="center", style="italic", color=GRIS, fontsize=11)
fig.savefig(os.path.join(FIG,"fig2_categorias.png")); plt.close(fig)

# ---------- FIG 3: vocabulario por organismo ----------
voc={r["org"]:r for r in load("lexico_vocabulario.csv")}
orgs=sorted(voc)
ac=[int(voc[o]["n_acunadas"]) for o in orgs]; ap=[int(voc[o]["n_aprendidas"]) for o in orgs]; fo=[int(voc[o]["n_fonador"]) for o in orgs]
fig,ax=plt.subplots(figsize=(8,5))
ax.bar(orgs,ac,color=AZUL,label="acuñadas (propias)",edgecolor="white")
ax.bar(orgs,ap,bottom=ac,color=VERDE,label="aprendidas (del otro)",edgecolor="white")
ax.bar(orgs,fo,bottom=[a+p for a,p in zip(ac,ap)],color=AMBAR,label="del aparato fonador",edgecolor="white")
ax.set_ylabel("nº de palabras"); ax.set_title("Vocabulario propio de cada organismo")
ax.legend()
fig.text(0.5,-0.03,"Cada organismo acuña palabras que nadie le dio, y algunas las aprende de los demás.",
         ha="center", style="italic", color=GRIS, fontsize=11)
fig.savefig(os.path.join(FIG,"fig3_vocabulario.png")); plt.close(fig)

# ---------- FIG 4: difusión léxica (quién copia a quién) ----------
dif=load("lexico_difusion.csv")
pos={"A":(0.15,0.8),"B":(0.85,0.8),"C":(0.85,0.2),"D":(0.15,0.2)}
fig,ax=plt.subplots(figsize=(7,6)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
colorg={"A":AZUL,"B":AMBAR,"C":VERDE,"D":ROJO}
for o,(x,y) in pos.items():
    ax.add_patch(Circle((x,y),0.09,color=colorg[o],zorder=3))
    ax.text(x,y,o,ha="center",va="center",color="white",fontsize=22,fontweight="bold",zorder=4)
for r in dif:
    s,d,n=r["origen"],r["copiador"],int(r["veces"])
    x1,y1=pos[s]; x2,y2=pos[d]
    ar=FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=22,lw=1+min(6,n/40),
                       color=MOR,alpha=.8,shrinkA=32,shrinkB=32,zorder=2,connectionstyle="arc3,rad=0.15")
    ax.add_patch(ar)
    mx,my=(x1+x2)/2,(y1+y2)/2
    ax.text(mx,my+0.05,f"{n}×",ha="center",color=MOR,fontweight="bold",fontsize=13)
ax.set_title("Transmisión cultural: una palabra viaja entre organismos\n(A → D → C)",fontsize=15)
fig.text(0.5,0.02,"Una palabra nacida en A fue copiada por D (226 veces) y de D pasó a C (28). Un linaje léxico.",
         ha="center", style="italic", color=GRIS, fontsize=11)
fig.savefig(os.path.join(FIG,"fig4_difusion.png")); plt.close(fig)

print("figuras de datos:", sorted(os.listdir(FIG)))
