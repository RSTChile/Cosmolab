"""
p04_em.py — PIEZA #4: ELECTROMAGNETISMO / RECOMBINACIÓN.

Qué hace, en simple: cuando el universo está MUY frío (T<T_REC), el EM liga un electrón a un protón y nace el
ÁTOMO de hidrógeno neutro. NO toca quarks (no tienen relevancia aquí: liga electrón-nucleón, otro nivel).
Es la época de la recombinación -- la que libera la luz del fondo de microondas.

Observable: nº de hidrógeno. Nivel: nucleón/electrón. Época: T < T_REC (la más fría).
"""
from cs072_modulos.pieza_base import Pieza

R_EM  = 0.10
T_REC = 0.15

class Electromagnetismo(Pieza):
    numero = 4
    nombre = "electromagnetismo/recombinación"
    nivel = "nucleon"
    T_umbral = T_REC
    observable = "hidrogeno"

    def actua(self, estado, step):
        # liga cada protón a un electrón (1:1), SÓLO en frío. La selección es FÍSICA, no por índice: recombinan
        # PRIMERO los protones en regiones de mayor densidad local (la recombinación arranca donde el plasma es
        # más denso). Como la densidad es intrínseca e invariante a permutación, esta selección también lo es
        # -> el índice deja de decidir qué materia se vuelve átomo (elimina el último residuo Shannon).
        e = estado
        if "recombinacion" not in e.epocas:
            e.epocas["recombinacion"] = round(float(e.T), 3)
        dens = getattr(e, "densidad", None)
        # protones ordenados por densidad DESCENDENTE (los más densos recombinan primero); tie-break por densidad
        prot_reps = [t[0] for t in e.prot_trios if t[0] not in {n for (n, _) in e.Bem}]
        if dens is not None:
            prot_reps.sort(key=lambda r: -float(dens[r]))
        libres_e = [x for x in e.elec if x not in {p for (_, p) in e.Bem}]
        if dens is not None:
            libres_e.sort(key=lambda r: -float(dens[r]))   # electrones densos primero (invariante)
        for rep in prot_reps:
            if libres_e:
                el = libres_e.pop(0)
                e.Bem[(rep, el)] = e.Bem.get((rep, el), 0.0) + R_EM
