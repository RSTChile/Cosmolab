"""
scratch_comparar_confinamiento.py -- comparación A/B: _detecta_trios ACTUAL (nucleo.py, sin exigir
ligadura real) vs Opción B (exige triángulo mutuamente ligado en Bq, desempate por densidad).

No toca nucleo.py. Replica exactamente la FASE 1 de corre() (mismo catálogo, mismo Estado, mismas
piezas 8_aniquilacion+3_fuerte, mismos pasos) para producir un Bq/color/carga/es_anti/viva/densidad
REAL e IDÉNTICO, y corre ambas funciones de detección de tríos sobre esos mismos datos -- comparación
limpia, ninguna diferencia de insumo, sólo diferencia de lógica de selección.
"""
import numpy as np
import time

from cs072_modulos.catalogo import catalogo
from cs072_modulos.estado import Estado
from cs072_modulos.piezas.p03_fuerte import FuerzaFuerte
from cs072_modulos.piezas.p08_aniquilacion import Aniquilacion
from cs072_modulos.piezas.p23_fluctuaciones import Fluctuaciones
from cs072_modulos.nucleo import _detecta_trios as detecta_trios_actual


def fase1_real(nq, naq, ne, npos, amp_asimetria=0.1, tasa_expansion=0.02, pasos=150,
               T0=3.0, amp_rugosidad=1.5, apagar=frozenset()):
    """Réplica EXACTA de la Fase 1 de nucleo.corre() -- mismo orden de piezas, mismos pasos."""
    color, carga, es_anti, es_quark, masa, densidad, temp = catalogo(nq, naq, ne, npos, amp_rugosidad)
    e = Estado(color, carga, es_anti, es_quark, masa, amp_asimetria, tasa_expansion, T0)
    e.densidad = densidad; e.temp = temp
    todas = {"3_fuerte": FuerzaFuerte(), "8_aniquilacion": Aniquilacion(),
             "23_fluctuaciones": Fluctuaciones(amp_rugosidad)}
    activas = {k: v for k, v in todas.items() if k not in apagar}
    for step in range(pasos):
        e.enfria(step)
        f = activas.get("23_fluctuaciones")
        if f: f.actua(e, step)
        for key in ("8_aniquilacion", "3_fuerte"):
            p = activas.get(key)
            if p and p.nivel == "quark" and p.activa(e): p.actua(e, step)
    return e, color, carga, es_anti


def detecta_trios_opcionB(Bq, color, carga, es_anti, viva, N, dens=None):
    """Exige TRIÁNGULO mutuamente ligado en Bq (i-j, i-k, j-k) -- confinamiento real, no estequiometría
    de población. Recorrido y desempate SIEMPRE por densidad descendente (magnitud física), nunca por
    índice/orden de aparición en el array."""
    b0 = max(float(Bq.sum(axis=1).mean()) / max(N - 1, 1), 1e-12)
    ligado = Bq > 1.5 * b0
    idxs = np.where((~es_anti) & (color >= 0) & (viva > 0.5))[0]
    if dens is None:
        dens = np.ones(N)
    orden = sorted(idxs, key=lambda i: -float(dens[i]))
    usado = np.zeros(N, bool)
    trios = []
    idxs_set_por_color = {c: set(int(i) for i in idxs if int(color[i]) == c) for c in (0, 1, 2)}

    for i in orden:
        if usado[i]:
            continue
        ci = int(color[i])
        vecinos = [j for j in np.where(ligado[i])[0]
                   if (not usado[j]) and int(color[j]) != ci and int(color[j]) in (0, 1, 2)
                   and (not es_anti[j]) and j in idxs_set_por_color[int(color[j])]]
        candidatos = []
        nv = len(vecinos)
        for a in range(nv):
            j = vecinos[a]; cj = int(color[j])
            for b in range(a + 1, nv):
                k = vecinos[b]; ck = int(color[k])
                if ck != cj and ligado[j, k]:
                    candidatos.append((min(float(dens[i]), float(dens[j]), float(dens[k])), j, k))
        if not candidatos:
            continue
        candidatos.sort(key=lambda t: -t[0])
        _, j, k = candidatos[0]
        usado[i] = usado[j] = usado[k] = True
        trios.append((i, j, k))
    return trios, ligado


def resumen(trios, carga, etiqueta, t0):
    prot = sum(1 for t in trios if int(carga[t[0]]) + int(carga[t[1]]) + int(carga[t[2]]) == 3)
    neut = sum(1 for t in trios if int(carga[t[0]]) + int(carga[t[1]]) + int(carga[t[2]]) == 0)
    otros = len(trios) - prot - neut
    print(f"{etiqueta}: bariones={len(trios)} protones={prot} neutrones={neut} "
          f"otros_carga={otros} tiempo={time.time()-t0:.2f}s")
    return dict(bariones=len(trios), protones=prot, neutrones=neut)


if __name__ == "__main__":
    nq, naq, ne, npos = 300, 210, 100, 70
    print(f"=== nq={nq} naq={naq} ne={ne} npos={npos} (misma escala que la prueba de admisibilidad) ===")
    e, color, carga, es_anti = fase1_real(nq, naq, ne, npos)
    print(f"Bq construido: suma={e.Bq.sum():.1f} media_no_cero={e.Bq[e.Bq>0].mean() if (e.Bq>0).any() else 0:.4f}")

    t0 = time.time()
    trios_actual, ligado = detecta_trios_actual(e.Bq, color, carga, es_anti, e.viva, e.N, dens=e.densidad)
    r_actual = resumen(trios_actual, carga, "ACTUAL (población, sin exigir ligadura)", t0)

    t0 = time.time()
    trios_B, ligado_B = detecta_trios_opcionB(e.Bq, color, carga, es_anti, e.viva, e.N, dens=e.densidad)
    r_B = resumen(trios_B, carga, "OPCION B (triángulo mutuamente ligado)", t0)

    # verificación cruzada: ¿los tríos de la Opción B son TODOS triángulos cerrados reales en Bq?
    todos_cerrados = all(ligado[t[0], t[1]] and ligado[t[0], t[2]] and ligado[t[1], t[2]] for t in trios_B)
    print(f"\n¿Todos los tríos de Opción B son triángulos mutuamente ligados en Bq? {todos_cerrados}")

    # ¿Cuántos de los tríos ACTUALES eran, de casualidad, triángulos cerrados también?
    cerrados_actual = sum(1 for t in trios_actual
                           if ligado[t[0], t[1]] and ligado[t[0], t[2]] and ligado[t[1], t[2]])
    print(f"De los {len(trios_actual)} tríos ACTUALES, {cerrados_actual} eran también triángulos "
          f"mutuamente ligados (el resto -- {len(trios_actual)-cerrados_actual} -- NUNCA estuvieron "
          f"realmente confinados).")

    print(f"\nresumen final: bariones actual={r_actual['bariones']}  bariones OpciónB={r_B['bariones']}  "
          f"caida={100*(1-r_B['bariones']/max(r_actual['bariones'],1)):.1f}%")
