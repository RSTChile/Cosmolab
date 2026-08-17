# test_confinamiento_real.py
# Comparación Opción A (actual) vs Opción B (rigurosa)

import numpy as np
import time
from cs072_modulos.nucleo import corre, _detecta_trios
from cs072_modulos.estado import Estado
from cs072_modulos.catalogo import catalogo

def _detecta_trios_opcionA(Bq, color, carga, es_anti, viva, N, dens=None):
    """Versión actual (contabilidad poblacional) - para referencia"""
    b0 = max(float(Bq.sum(axis=1).mean())/max(N-1,1), 1e-12)
    ligado = Bq > 1.5*b0
    idxs = np.where((~es_anti)&(color>=0)&(viva>0.5))[0]
    up_por_color = {c: [] for c in (0,1,2)}
    dn_por_color = {c: [] for c in (0,1,2)}
    for i in idxs:
        (up_por_color if int(carga[i])==2 else dn_por_color)[int(color[i])].append(i)
    if dens is not None:
        for c in (0,1,2):
            up_por_color[c].sort(key=lambda q:-float(dens[q]))
            dn_por_color[c].sort(key=lambda q:-float(dens[q]))
    bolsa = {c: [('u',q) for q in up_por_color[c]] + [('d',q) for q in dn_por_color[c]] for c in (0,1,2)}
    if dens is not None:
        for c in (0,1,2):
            bolsa[c].sort(key=lambda sq:-float(dens[sq[1]]))
    n_trios = min(len(bolsa[0]), len(bolsa[1]), len(bolsa[2]))
    trios = []
    for t in range(n_trios):
        picks = [bolsa[c][t] for c in (0,1,2)]
        idx3 = tuple(q for (_,q) in picks)
        trios.append(idx3)
    return trios, ligado

def _detecta_trios_opcionB(Bq, color, carga, es_anti, viva, N, dens=None):
    """Versión rigurosa (triángulo cerrado en grafo de confinamiento)"""
    b0 = max(float(Bq.sum(axis=1).mean())/max(N-1,1), 1e-12)
    ligado = Bq > 1.5*b0
    idxs = np.where((~es_anti)&(color>=0)&(viva>0.5))[0]
    
    # Orden por densidad DESCENDENTE (física, no índice)
    if dens is not None:
        orden = sorted(idxs, key=lambda i: -float(dens[i]))
    else:
        orden = list(idxs)  # fallback
    
    usado = np.zeros(N, dtype=bool)
    trios = []
    
    for i in orden:
        if usado[i]:
            continue
        ci = int(color[i])
        
        # Vecinos ligados de otros colores
        vecinos = [
            j for j in idxs 
            if not usado[j] and j != i 
            and ligado[i, j] 
            and int(color[j]) != ci
        ]
        
        if len(vecinos) < 2:
            continue
            
        candidatos = []
        for a in range(len(vecinos)):
            j = vecinos[a]
            cj = int(color[j])
            for b in range(a+1, len(vecinos)):
                k = vecinos[b]
                ck = int(color[k])
                # Verificar: colores distintos y ligadura mutua
                if ck != cj and ligado[j, k]:
                    # Score = densidad mínima (eslabón más débil)
                    score = min(dens[i], dens[j], dens[k]) if dens is not None else 1.0
                    candidatos.append((score, j, k))
        
        if not candidatos:
            continue
            
        # Elegir el trío más denso (score más alto)
        candidatos.sort(key=lambda t: -t[0])
        _, j, k = candidatos[0]
        
        usado[i] = usado[j] = usado[k] = True
        trios.append((i, j, k))
    
    return trios, ligado

def comparar_confinamiento(nq=300, naq=210, ne=100, npos=70, 
                          amp_rugosidad=0.5, T0=3.0, tasa_expansion=0.02,
                          pasos=400, verbose=True):
    """Compara ambas opciones con la misma semilla de catálogo"""
    
    if verbose:
        print("="*70)
        print(f"COMPARACIÓN CONFINAMIENTO: Opción A vs Opción B")
        print(f"nq={nq}, naq={naq}, ne={ne}, npos={npos}")
        print(f"T0={T0}, tasa_expansion={tasa_expansion}, pasos={pasos}")
        print("="*70)
    
    # Generar catálogo una sola vez (misma semilla)
    color, carga, es_anti, es_quark, masa, densidad, temp = catalogo(
        nq, naq, ne, npos, amp_rugosidad
    )
    N = len(color)
    
    # Inicializar estado (necesario para Bq después de la evolución)
    e = Estado(color, carga, es_anti, es_quark, masa, 0.1, tasa_expansion, T0)
    e.densidad = densidad
    e.temp = temp
    
    # Ejecutar evolución QUARK para obtener Bq
    from cs072_modulos.piezas.p03_fuerte import FuerzaFuerte
    from cs072_modulos.piezas.p08_aniquilacion import Aniquilacion
    from cs072_modulos.piezas.p23_fluctuaciones import Fluctuaciones
    
    fuerte = FuerzaFuerte()
    aniquilacion = Aniquilacion()
    fluct = Fluctuaciones(amp_rugosidad)
    
    for step in range(pasos):
        e.enfria(step)
        if fluct and fluct.activa(e):
            fluct.actua(e, step)
        if aniquilacion and aniquilacion.activa(e):
            aniquilacion.actua(e, step)
        if fuerte and fuerte.activa(e):
            fuerte.actua(e, step)
    
    # Aplicar ambas detecciones
    t_start = time.time()
    trios_A, lig_A = _detecta_trios_opcionA(
        e.Bq, color, carga, es_anti, e.viva, N, dens=e.densidad
    )
    t_A = time.time() - t_start
    
    t_start = time.time()
    trios_B, lig_B = _detecta_trios_opcionB(
        e.Bq, color, carga, es_anti, e.viva, N, dens=e.densidad
    )
    t_B = time.time() - t_start
    
    # Estadísticas
    n_A = len(trios_A)
    n_B = len(trios_B)
    
    # Clasificar por carga (protón = +3, neutrón = 0)
    def clasificar_trios(trios):
        protones = []
        neutrones = []
        for t in trios:
            qsum = int(carga[t[0]]) + int(carga[t[1]]) + int(carga[t[2]])
            if qsum == 3:
                protones.append(t)
            elif qsum == 0:
                neutrones.append(t)
        return protones, neutrones
    
    p_A, n_A_clas = clasificar_trios(trios_A)
    p_B, n_B_clas = clasificar_trios(trios_B)
    
    # Densidades promedio
    def densidad_promedio_trio(t):
        return (densidad[t[0]] + densidad[t[1]] + densidad[t[2]]) / 3.0
    
    dens_A = [densidad_promedio_trio(t) for t in trios_A] if trios_A else [0]
    dens_B = [densidad_promedio_trio(t) for t in trios_B] if trios_B else [0]
    
    # Verificar solapamiento (cuántos tríos de B están en A)
    set_A = set(trios_A)
    set_B = set(trios_B)
    overlap = set_A & set_B
    
    if verbose:
        print("\n" + "="*70)
        print("RESULTADOS")
        print("="*70)
        print(f"\n[TIEMPO] Opción A: {t_A*1000:.2f} ms")
        print(f"[TIEMPO] Opción B: {t_B*1000:.2f} ms")
        print(f"[RATIO]  B/A: {t_B/t_A:.2f}x")
        
        print(f"\n[BARIONES TOTALES]")
        print(f"  Opción A: {n_A}")
        print(f"  Opción B: {n_B}")
        print(f"  Diferencia: {n_A - n_B} ({100*(n_A-n_B)/n_A if n_A>0 else 0:.1f}%)")
        
        print(f"\n[PROTONES]")
        print(f"  Opción A: {len(p_A)}")
        print(f"  Opción B: {len(p_B)}")
        print(f"  Diferencia: {len(p_A)-len(p_B)}")
        
        print(f"\n[NEUTRONES]")
        print(f"  Opción A: {len(n_A_clas)}")
        print(f"  Opción B: {len(n_B_clas)}")
        print(f"  Diferencia: {len(n_A_clas)-len(n_B_clas)}")
        
        print(f"\n[DENSIDAD PROMEDIO DE TRIOS]")
        print(f"  Opción A: {np.mean(dens_A):.4f} ± {np.std(dens_A):.4f}")
        print(f"  Opción B: {np.mean(dens_B):.4f} ± {np.std(dens_B):.4f}")
        
        print(f"\n[SOLAPAMIENTO]")
        print(f"  Tríos en común: {len(overlap)}")
        if n_A > 0:
            print(f"  % de A que están en B: {100*len(overlap)/n_A:.1f}%")
        if n_B > 0:
            print(f"  % de B que están en A: {100*len(overlap)/n_B:.1f}%")
        
        # Verificación de triángulos cerrados en B
        if trios_B:
            lig_check = []
            for t in trios_B:
                i, j, k = t
                ok = lig_B[i,j] and lig_B[i,k] and lig_B[j,k]
                lig_check.append(ok)
            print(f"\n[VERIFICACIÓN TRIÁNGULOS CERRADOS]")
            print(f"  Opción B: {sum(lig_check)}/{len(trios_B)} tríos tienen los 3 enlaces")
            if sum(lig_check) != len(trios_B):
                print(f"  ⚠️ ALERTA: algunos tríos de B no tienen triángulo cerrado")
        
        # Verificar que A NO tiene triángulos cerrados (debe fallar)
        if trios_A:
            lig_check_A = []
            for t in trios_A[:min(10, len(trios_A))]:  # solo una muestra
                i, j, k = t
                ok = lig_A[i,j] and lig_A[i,k] and lig_A[j,k]
                lig_check_A.append(ok)
            print(f"\n[VERIFICACIÓN TRIÁNGULOS CERRADOS EN A (muestra)]")
            print(f"  Opción A: {sum(lig_check_A)}/{len(lig_check_A)} tríos tienen los 3 enlaces")
            if sum(lig_check_A) == 0:
                print(f"  ✅ Confirmado: A no exige triángulos cerrados")
        
        # Impacto en cascada (estimación)
        print(f"\n[IMPACTO ESTIMADO EN CADENA]")
        H_A = min(len(p_A), ne)  # hidrógeno = protones + electrones
        H_B = min(len(p_B), ne)
        He_A = min(len(p_A)//2, len(n_A_clas)//2, ne//2)
        He_B = min(len(p_B)//2, len(n_B_clas)//2, ne//2)
        print(f"  H (opción A): {H_A}")
        print(f"  H (opción B): {H_B}")
        print(f"  He (opción A): {He_A}")
        print(f"  He (opción B): {He_B}")
        print(f"  Reducción H: {H_A-H_B} ({100*(H_A-H_B)/H_A if H_A>0 else 0:.1f}%)")
        print(f"  Reducción He: {He_A-He_B} ({100*(He_A-He_B)/He_A if He_A>0 else 0:.1f}%)")
    
    return {
        'opcionA': {'trios': trios_A, 'protones': p_A, 'neutrones': n_A_clas, 'tiempo': t_A},
        'opcionB': {'trios': trios_B, 'protones': p_B, 'neutrones': n_B_clas, 'tiempo': t_B},
        'overlap': list(overlap),
        'parametros': {'nq': nq, 'naq': naq, 'ne': ne, 'npos': npos, 'T0': T0}
    }

if __name__ == "__main__":
    # Prueba con los parámetros de CC
    resultados = comparar_confinamiento(
        nq=300, naq=210, ne=100, npos=70,
        amp_rugosidad=0.5, T0=3.0,
        tasa_expansion=0.02, pasos=400,
        verbose=True
    )
    
    # Prueba adicional con diferentes parámetros
    print("\n" + "="*70)
    print("PRUEBA ADICIONAL: parámetros variados")
    print("="*70)
    
    for nq in [200, 400]:
        for ne in [50, 150]:
            res = comparar_confinamiento(
                nq=nq, naq=int(nq*0.7), ne=ne, npos=int(nq*0.2),
                amp_rugosidad=0.5, T0=3.0,
                tasa_expansion=0.02, pasos=200,
                verbose=False
            )
            nA = len(res['opcionA']['trios'])
            nB = len(res['opcionB']['trios'])
            print(f"nq={nq}, ne={ne}: A={nA}, B={nB}, diff={nA-nB} ({100*(nA-nB)/nA if nA>0 else 0:.1f}%)")