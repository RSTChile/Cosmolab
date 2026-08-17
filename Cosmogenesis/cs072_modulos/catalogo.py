"""
catalogo.py — PIEZA #6: CATÁLOGO DE PARTÍCULAS.

Qué hace, en simple: define QUÉ partículas hay y sus propiedades intrínsecas (color, carga, si es anti, masa).
color/carga son COMPOSICIÓN del catálogo (tercios de color, mitad up/down), NO ruptura de simetría por índice.
El índice sólo REPARTE la mezcla; ningún resultado depende del orden (verificado invariante a permutación).

masa: u=2.3, d=4.8 -> neutrón(udd)=11.9 > protón(uud)=9.4 (el neutrón pesa más, base del freeze-out).
"""
import numpy as np

MU, MD = 2.3, 4.8

def densidad_intrinseca(N, amp):
    """Campo de densidad INTRÍNSECO por partícula (rugosidad del plasma, pieza #23). Es una PROPIEDAD que viaja
    con la partícula (se permuta con el catálogo) -> el test de permutación puede detectar si el índice decide.
    Lo físico es la DISTRIBUCIÓN de densidades (el histograma), no qué partícula tiene cuál. amp=0 -> uniforme.
    Distribución tipo lognormal (fluctuaciones cosmológicas): pocos picos densos, muchas regiones tenues.
    Determinista (cero azar), declarada como heterogeneidad externa. NO es una coordenada espacial: es un escalar."""
    if amp <= 0:
        return np.ones(N)
    # cuantiles de una lognormal determinista: valor por rango, zero-mean en log. La ASIGNACIÓN a partículas
    # se hará por catálogo y luego se permuta; sólo el multiset (la forma de la distribución) es físico.
    r = (np.arange(N) + 0.5) / N               # rangos en (0,1)
    z = np.sqrt(2.0) * _erfinv(2*r - 1.0)      # normal estándar por cuantiles (Box-Muller inverso determinista)
    d = np.exp(amp * z)                          # lognormal: densidad positiva, sesgada (picos raros)
    return d / d.mean()                          # normalizar a media 1 (densidad relativa)

def _erfinv(y):
    # aproximación racional de erfinv (Winitzki), determinista, sin scipy
    a = 0.147
    ln = np.log(1 - y*y + 1e-300)
    t = 2/(np.pi*a) + ln/2
    return np.sign(y) * np.sqrt(np.sqrt(t*t - ln/a) - t)

def catalogo(nq, naq, ne, npos, amp_rugosidad=0.0):
    color,carga,es_anti,es_quark,masa=[],[],[],[],[]
    def add(n, anti, quark):
        for i in range(n):
            if quark:
                color.append(i%3); carga.append(2 if i%2==0 else -1); masa.append(MU if i%2==0 else MD)
            else:
                color.append(-1); carga.append(-3 if not anti else 3); masa.append(0.51)
            es_anti.append(anti); es_quark.append(quark)
    add(nq,False,True); add(naq,True,True); add(ne,False,False); add(npos,True,False)
    N = nq+naq+ne+npos
    dens = densidad_intrinseca(N, amp_rugosidad)   # densidad intrínseca (rugosidad), parte del catálogo
    # La temperatura de cada partícula ES su densidad (más densidad = más caliente): UNA sola asimetría, no dos.
    # HONESTIDAD (anti-Shannon): esta densidad/temperatura es HETEROGENEIDAD EXTERNA DECLARADA (determinista,
    # cero azar, pero impuesta como condición inicial), NO medida de la estructura. NO se afirma "medida no
    # impuesta" -- eso sería falso (ver docstring de densidad_intrinseca). Es la asimetría de partida del arco,
    # declarada como tal. temp deriva de dens -> hereda su invariancia a permutación (que sí es real).
    temp = dens.copy()                             # temperatura ≡ densidad local (heterogeneidad DECLARADA)
    return (np.array(color), np.array(carga,np.int8), np.array(es_anti,bool),
            np.array(es_quark,bool), np.array(masa), dens, temp)
