"""
estado.py — EL ESTADO COMPARTIDO del cronograma CS072.

Es el contrato central: TODOS los módulos-pieza leen y modifican este objeto, nada más. Un módulo nunca habla
con otro directamente; sólo a través del Estado. Así cada fuerza toca SÓLO su parte y se puede apagar/auditar sola.

NIVELES de ligadura (separados para que una fuerza no contamine a otra -- lección del bug del confinamiento):
  Bq   : ligadura QUARK-QUARK  -> SÓLO la fuerza fuerte la construye (forma bariones).
  Bnuc : ligadura NUCLEÓN-NUCLEÓN -> fuerte residual en frío (forma núcleos: He).
  Bem  : ligadura ELECTRÓN-NUCLEÓN -> EM en frío (forma átomos: H).
  Bgrav: ligadura ÁTOMO-ÁTOMO -> gravedad (teje la red: el espacio).
El contador lee cada nivel por separado. Ninguna pieza escribe fuera de su nivel.
"""
import numpy as np

class Estado:
    def __init__(self, color, carga, es_anti, es_quark, masa, amp_asimetria, tasa_expansion, T0=3.0):
        N = len(color)
        # --- identidad de las partículas (catálogo, pieza #6) ---
        self.color=color; self.carga=carga; self.es_anti=es_anti; self.es_quark=es_quark; self.masa=masa
        self.N=N
        self.viva=np.ones(N)                       # 1=viva, 0=aniquilada
        # --- campo térmico (condición inicial: gradiente de asimetría) ---
        d=np.linspace(-amp_asimetria,amp_asimetria,N); self.d=d-d.mean()
        self.amp_asimetria=amp_asimetria; self.tasa_expansion=tasa_expansion
        self.T0=T0; self.T=T0                       # temperatura global (cae con el enfriamiento)
        # --- niveles de ligadura, cada uno tocado por SU fuerza ---
        self.Bq=np.zeros((N,N))                     # quark-quark (fuerte)
        self.Bnuc={}                                # nucleón-nucleón (fuerte residual): dict (a,b)->peso
        self.Bem={}                                 # electrón-nucleón (EM): dict (nucleon,electron)->peso
        self.Bgrav={}                               # átomo-átomo (gravedad): dict (a,b)->peso
        # --- productos de fase 1, que la fase 2 consume ---
        self.trios=[]; self.prot_trios=[]; self.neut_trios=[]
        self.nucleones=[]; self.es_prot={}
        self.elec=list(np.where((~es_anti)&(~es_quark)&(self.viva>0.5))[0])
        # --- bitácora de épocas (a qué T se activó cada fuerza) ---
        self.epocas={}
        # --- máscaras de pares reutilizables ---
        self.cd=(color[:,None]!=color[None,:])&(color[:,None]>=0)&(color[None,:]>=0); np.fill_diagonal(self.cd,False)
        self.me=(es_anti[:,None]==es_anti[None,:])  # misma clase materia/antimateria

    def enfria(self, step):
        """Ley de enfriamiento física: feroz al inicio, se frena (T~1/sqrt). Expansión del PLASMA, no del espacio."""
        self.T = self.T0/np.sqrt(1.0 + (self.tasa_expansion*50.0)*step)
        return self.T
