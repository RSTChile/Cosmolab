# -*- coding: utf-8 -*-
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Wedge, Ellipse
import numpy as np
plt.rcParams.update({"font.family":"DejaVu Sans","savefig.dpi":150,"savefig.bbox":"tight"})
AZUL="#2c6fbb"; VERDE="#3aa66f"; ROJO="#d6553f"; AMBAR="#e8a13a"; GRIS="#7d8ea6"; MOR="#7b5cc4"; CREMA="#f4 efe9".replace(" ","")

def cabeza(ax,cx,cy,r,cara=1,color="#cfe0f5",label=None,eL=0.5,eR=0.5):
    ax.add_patch(Circle((cx,cy),r,color=color,ec="#33506e",lw=2,zorder=3))
    # orejas (anillos que laten)
    ax.add_patch(Circle((cx-r*1.05,cy),r*0.28,color=AMBAR,alpha=0.3+0.5*eL,ec=AMBAR,lw=2,zorder=2))
    ax.add_patch(Circle((cx+r*1.05,cy),r*0.28,color=AZUL,alpha=0.3+0.5*eR,ec=AZUL,lw=2,zorder=2))
    # ojos
    ax.add_patch(Circle((cx-r*0.35,cy+r*0.25),r*0.12,color="#22303f",zorder=4))
    ax.add_patch(Circle((cx+r*0.35,cy+r*0.25),r*0.12,color="#22303f",zorder=4))
    # boca según cara
    th=np.linspace(0,np.pi,40)
    if cara>0: y=-np.sin(th)
    elif cara<0: y=np.sin(th)
    else: y=np.zeros_like(th)
    ax.plot(cx+ (np.linspace(-1,1,40))*r*0.4, cy-r*0.35+y*r*0.18, color="#22303f",lw=3,zorder=4)
    if label: ax.text(cx,cy-r*1.5,label,ha="center",fontsize=13,fontweight="bold")

# ---------- ESQ 1: el organismo = cabeza + célula con organelos ----------
fig,(axL,axR)=plt.subplots(1,2,figsize=(13,6.2))
# panel izq: la cabeza (lo que se ve)
axL.set_xlim(0,1);axL.set_ylim(0,1);axL.axis("off")
cabeza(axL,0.5,0.55,0.22,cara=1,eL=0.7,eR=0.3,label=None)
axL.text(0.5,0.92,"LO QUE SE VE: el 'cuerpo'",ha="center",fontsize=14,fontweight="bold")
axL.annotate("oído izquierdo\n(late con lo que oye)",(0.27,0.55),(0.02,0.30),fontsize=10,color=AMBAR,
             arrowprops=dict(arrowstyle="->",color=AMBAR))
axL.annotate("oído derecho",(0.73,0.55),(0.72,0.30),fontsize=10,color=AZUL,
             arrowprops=dict(arrowstyle="->",color=AZUL))
axL.annotate("la BOCA = cómo se siente\n(sonríe si está bien)",(0.5,0.42),(0.55,0.12),fontsize=10,
             arrowprops=dict(arrowstyle="->",color="#22303f"))
axL.annotate("la cabeza GIRA hacia\ndonde 'mira'",(0.5,0.77),(0.1,0.85),fontsize=10,color=GRIS,
             arrowprops=dict(arrowstyle="->",color=GRIS))
# panel der: la célula y sus organelos
axR.set_xlim(0,1);axR.set_ylim(0,1);axR.axis("off")
axR.text(0.5,0.95,"LO QUE PASA ADENTRO: la célula y sus 'órganos'",ha="center",fontsize=14,fontweight="bold")
axR.add_patch(Circle((0.5,0.48),0.40,color="#eef4fb",ec=VERDE,lw=2.5,ls="--",zorder=1))
axR.add_patch(Circle((0.5,0.48),0.13,color=MOR,alpha=0.25,ec=MOR,lw=2,zorder=2))
axR.text(0.5,0.48,"MILIEU\n(medio\ncompartido Φ)",ha="center",va="center",fontsize=9,color=MOR,fontweight="bold",zorder=3)
orgs=[("Propio-\ncepción",0.5,0.82,VERDE),("Memoria",0.78,0.68,AZUL),("Metabo-\nlismo",0.85,0.45,AMBAR),
      ("Homeo-\nstasis",0.78,0.22,ROJO),("Hemis-\nferios",0.5,0.13,MOR),("Membrana\n(tímpano)",0.22,0.22,VERDE),
      ("Comuni-\ncación",0.15,0.45,AZUL),("Aparato\nfonador",0.22,0.68,AMBAR)]
for nom,x,y,c in orgs:
    axR.add_patch(FancyBboxPatch((x-0.085,y-0.05),0.17,0.10,boxstyle="round,pad=0.01",
                  fc="white",ec=c,lw=2,zorder=4))
    axR.text(x,y,nom,ha="center",va="center",fontsize=8,color=c,fontweight="bold",zorder=5)
    axR.add_patch(FancyArrowPatch((x,y),(0.5,0.48),arrowstyle="-",color=GRIS,alpha=0.4,
                  shrinkA=14,shrinkB=30,zorder=1))
fig.text(0.5,0.01,"Cada órgano (organelo) es un módulo independiente; no se 'hablan' por cables fijos, sino dejando señales en el medio compartido (el milieu). Así pueden reorganizarse y reinventarse.",
         ha="center",style="italic",color=GRIS,fontsize=10.5)
fig.savefig("esq1_organismo.png"); plt.close(fig)

# ---------- ESQ 2: el lazo semiótico anti-Shannon ----------
fig,ax=plt.subplots(figsize=(11,6)); ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
ax.text(0.5,0.95,"Cómo NACE el sentido (el lazo Anti-Shannon)",ha="center",fontsize=15,fontweight="bold")
pasos=[("EL MUNDO\nun sonido llega",0.12,0.5,GRIS),
       ("LO CAPTA\nsiente que hay algo,\ncuán fuerte (NO lo\n'descifra')",0.34,0.72,AZUL),
       ("LE AFECTA\ncambia su energía,\nequilibrio, necesidad",0.62,0.72,AMBAR),
       ("LO VALORA\n¿me deja mejor\no peor? (gusto)",0.84,0.5,VERDE),
       ("RESPONDE\nse orienta, vocaliza,\nacuña una palabra",0.62,0.22,ROJO),
       ("ALTERA EL MUNDO\ny el ciclo recomienza",0.34,0.22,MOR)]
for nom,x,y,c in pasos:
    ax.add_patch(FancyBboxPatch((x-0.11,y-0.08),0.22,0.16,boxstyle="round,pad=0.01",fc="white",ec=c,lw=2.5,zorder=3))
    ax.text(x,y,nom,ha="center",va="center",fontsize=9.5,color=c,fontweight="bold",zorder=4)
seq=[(0.12,0.5),(0.34,0.72),(0.62,0.72),(0.84,0.5),(0.62,0.22),(0.34,0.22),(0.12,0.5)]
for (x1,y1),(x2,y2) in zip(seq,seq[1:]):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=20,color=GRIS,
                 lw=2,shrinkA=40,shrinkB=40,zorder=1,connectionstyle="arc3,rad=0.1"))
ax.text(0.5,0.5,"EL SENTIDO\nse PRODUCE aquí dentro,\nno se lee en la señal",ha="center",va="center",
        fontsize=11,color=MOR,fontweight="bold",style="italic",
        bbox=dict(boxstyle="round",fc="#f3effb",ec=MOR))
fig.savefig("esq2_lazo_semiotico.png"); plt.close(fig)

# ---------- ESQ 3: la sociedad de 4 ----------
fig,ax=plt.subplots(figsize=(8.5,7)); ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
ax.text(0.5,0.96,"La sociedad: cuatro organismos que se oyen",ha="center",fontsize=15,fontweight="bold")
pos={"A":(0.27,0.72,"#cfe0f5"),"B":(0.73,0.72,"#ffe6b0"),"C":(0.73,0.28,"#c9f0db"),"D":(0.27,0.28,"#ffd2c4")}
caras={"A":1,"B":0,"C":-1,"D":1}
for o,(x,y,c) in pos.items():
    cabeza(ax,x,y,0.11,cara=caras[o],color=c,eL=0.5,eR=0.5)
    ax.text(x,y-0.20,f"Organismo {o}",ha="center",fontsize=12,fontweight="bold")
import itertools
for a,b in itertools.combinations(pos,2):
    x1,y1,_=pos[a]; x2,y2,_=pos[b]
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="<|-|>",mutation_scale=12,color=MOR,
                 alpha=0.35,lw=1.5,shrinkA=26,shrinkB=26,zorder=1))
ax.text(0.5,0.5,"se oyen\nentre todos",ha="center",va="center",fontsize=10,color=MOR,style="italic")
fig.text(0.5,0.02,"Cada uno oye por un oído al resto (la 'sociedad') y por el otro el sonido del entorno. De ese oírse mutuo emergen palabras compartidas.",
         ha="center",style="italic",color=GRIS,fontsize=10.5)
fig.savefig("esq3_sociedad.png"); plt.close(fig)

# ---------- ESQ 4: escala de Organismicidad (OI) ----------
fig,ax=plt.subplots(figsize=(10,2.6)); ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
ax.text(0.5,0.92,"El 'termómetro' de cuán organismo es (índice OI, de 0 a 1)",ha="center",fontsize=14,fontweight="bold")
ax.add_patch(FancyBboxPatch((0.05,0.35),0.9,0.22,boxstyle="round,pad=0.005",fc="white",ec=GRIS,lw=1))
ax.add_patch(plt.Rectangle((0.05,0.35),0.9*0.4,0.22,color="#f0d3cc"))
ax.add_patch(plt.Rectangle((0.05+0.9*0.4,0.35),0.9*0.3,0.22,color="#f6e7c6"))
ax.add_patch(plt.Rectangle((0.05+0.9*0.7,0.35),0.9*0.3,0.22,color="#cdeedd"))
for v,lab,c in [(0.4,"0,4",GRIS),(0.7,"0,7",GRIS)]:
    ax.plot([0.05+0.9*v]*2,[0.33,0.59],color=GRIS,lw=1.5)
    ax.text(0.05+0.9*v,0.27,lab,ha="center",fontsize=10,color=c)
ax.text(0.05+0.9*0.2,0.46,"aún no\norganismo",ha="center",va="center",fontsize=9)
ax.text(0.05+0.9*0.55,0.46,"protoorganismo\n(en borrador)",ha="center",va="center",fontsize=9)
ax.text(0.05+0.9*0.85,0.46,"organismo\npleno",ha="center",va="center",fontsize=9)
for v,lab,c in [(0.648,"célula sola\n0,648",AZUL),(0.858,"colonia\n0,858",VERDE)]:
    ax.plot([0.05+0.9*v]*2,[0.35,0.70],color=c,lw=2.5)
    ax.add_patch(Circle((0.05+0.9*v,0.70),0.012,color=c))
    ax.text(0.05+0.9*v,0.80,lab,ha="center",fontsize=9.5,color=c,fontweight="bold")
fig.savefig("esq4_OI.png"); plt.close(fig)

print("esquemas:", [f for f in sorted(__import__('os').listdir('.')) if f.startswith('esq')])
