"""
p02_gravedad.py — PIEZA #2: GRAVEDAD.

Qué hace, en simple: liga ÁTOMO con ÁTOMO por su masa. Es la fuerza que teje la red de átomos ya formados
-> de esa red emerge el ESPACIO (su geometría). No forma átomos ni bariones; los CONECTA una vez que existen.
Sólo tiene con qué trabajar si ya hay átomos (anti-Shannon: sin átomos no hay red que tejer).

Observable: diámetro de la red de átomos. Nivel: átomo. Época: siempre (pero sólo actúa si hay átomos).
"""
import numpy as np
from cs072_modulos.pieza_base import Pieza

R_GRAV = 0.02

class Gravedad(Pieza):
    numero = 2
    nombre = "gravedad"
    nivel = "atomo"
    T_umbral = None
    observable = "diametro_red"

    def actua(self, estado, step):
        # liga pares de ÁTOMOS (nodos ya recombinados) con PESO por PRODUCTO DE MASAS (gravedad real: los
        # más masivos se atraen más). NO liga todos los pares: sólo aquellos cuyo producto de masas supera
        # el promedio (umbral relativo, no una perilla de forma) -> la estructura se ANCLA en los átomos más
        # pesados (He), no es un grafo completo. Sin masa NO hay gravedad (constraint del director).
        e = estado
        atomos = [n for (n, _) in e.Bem]          # protones recombinados = átomos con electrón (H o núcleo)
        if len(atomos) < 2:
            return
        m = e.masa_trio                            # masa total de cada átomo (suma de su trío)
        # peso efectivo de cada átomo = masa × densidad local (la rugosidad de #23). Sin rugosidad, densidad=1
        # para todos y esto vuelve a ser masa-uniforme (grafo trivial). CON rugosidad, las regiones densas pesan
        # más -> la gravedad las liga preferentemente y la red deja de ser trivial.
        dens = getattr(e, "densidad", None)
        def peso(a):
            ma = m.get(a, 1.0)
            da = float(dens[a]) if dens is not None else 1.0
            return ma * da
        # CRITERIO DE LIGADURA (sin constantes libres): un par de átomos se liga sólo si cumple DOS condiciones
        # físicas, ambas medidas de la propia distribución (no impuestas):
        #   (1) SOBREDENSIDAD: ambos en regiones con densidad > media (las tenues se dispersan, no colapsan).
        #   (2) LOCALIDAD TÉRMICA: sus temperaturas intrínsecas son cercanas (|ΔT| < mediana de las |ΔT| de los
        #       pares sobredensos). Dos átomos con T parecida son "vecinos" -- compartieron historia térmica.
        #       Esto NO es posición espacial: es proximidad en el gradiente térmico (la asimetría inicial que la
        #       expansión preservó). Es lo que da un "cerca/lejos" sin meter coordenadas -> rompe el grafo estrella.
        # Sin rugosidad: densidad uniforme -> nadie sobredenso -> sin red. Sin gradiente: T uniforme -> todos
        # vecinos -> vuelve al grafo completo (correcto: sin asimetría térmica no hay localidad, no hay métrica).
        dens = getattr(e, "densidad", None); temp = getattr(e, "temp", None)
        if dens is None or temp is None:
            return
        media_dens = float(np.mean([float(dens[a]) for a in atomos]))
        sobredensos = [a for a in atomos if float(dens[a]) > media_dens]
        if len(sobredensos) < 2:
            return
        # ventana de localidad térmica = mediana de |ΔT| entre pares sobredensos (medida, no una perilla)
        difs = [abs(float(temp[a]) - float(temp[b]))
                for i, a in enumerate(sobredensos) for b in sobredensos[i+1:]]
        if not difs:
            return
        ventana = float(np.median(difs))
        for i in range(len(sobredensos)):
            for j in range(i + 1, len(sobredensos)):
                a, b = sobredensos[i], sobredensos[j]
                if abs(float(temp[a]) - float(temp[b])) <= ventana:   # sólo vecinos térmicos
                    e.Bgrav[(a, b)] = e.Bgrav.get((a, b), 0.0) + R_GRAV * peso(a) * peso(b)
