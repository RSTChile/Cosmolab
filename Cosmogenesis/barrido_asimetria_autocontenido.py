# barrido_asimetria_autocontenido.py
"""
BARRIDO COMPLETO DE ASIMETRÍA - VERSIÓN AUTOCONTENIDA
No depende de módulos externos de Cosmogenesis.
Genera su propio catálogo y ejecuta la simulación mínima.
"""

import numpy as np
import json
import time
from collections import defaultdict
import matplotlib.pyplot as plt

# ===================================================================
# 1. CATÁLOGO PLANO (sin densidad pre-asignada)
# ===================================================================

def catalogo_plano(nq, naq, ne, npos):
    """
    Genera catálogo SIN DENSIDAD ASIGNADA.
    Todas las partículas tienen densidad = 1.0 (plana).
    El orden de creación NO importa.
    """
    N = nq + naq + ne + npos
    
    # Arrays de números cuánticos
    color = np.zeros(N, dtype=int)
    carga = np.zeros(N, dtype=int)
    es_anti = np.zeros(N, dtype=bool)
    es_quark = np.zeros(N, dtype=bool)
    masa = np.zeros(N, dtype=float)
    
    idx = 0
    
    # Quarks
    for i in range(nq):
        es_quark[idx] = True
        color[idx] = i % 3
        carga[idx] = 2 if np.random.random() < 0.5 else -1  # up (+2) o down (-1)
        es_anti[idx] = False
        masa[idx] = 0.002 if carga[idx] == 2 else 0.005
        idx += 1
    
    # Antiquarks
    for i in range(naq):
        es_quark[idx] = True
        color[idx] = i % 3
        carga[idx] = -2 if np.random.random() < 0.5 else 1  # anti-up o anti-down
        es_anti[idx] = True
        masa[idx] = 0.002 if carga[idx] == -2 else 0.005
        idx += 1
    
    # Electrones
    for i in range(ne):
        es_quark[idx] = False
        color[idx] = -1
        carga[idx] = -1
        es_anti[idx] = False
        masa[idx] = 0.0005
        idx += 1
    
    # Positrones
    for i in range(npos):
        es_quark[idx] = False
        color[idx] = -1
        carga[idx] = 1
        es_anti[idx] = True
        masa[idx] = 0.0005
        idx += 1
    
    # DENSIDAD PLANA: todos 1.0
    densidad = np.ones(N)
    
    # Temperatura inicial: misma para todos
    temp = 3.0 * np.ones(N)
    
    return color, carga, es_anti, es_quark, masa, densidad, temp

# ===================================================================
# 2. NÚCLEO MÍNIMO (simulación simplificada)
# ===================================================================

def simular_bariones(color, carga, es_anti, es_quark, masa, densidad, temp, pasos=10):
    """
    Simulación mínima para detectar bariones confinados.
    Versión simplificada que no requiere los módulos completos.
    """
    N = len(color)
    
    # Estado de las partículas
    viva = np.ones(N, dtype=bool)
    
    # Matriz de interacción fuerte (Bq)
    # Simulación simplificada: Bq[i,j] = 1 si pueden interactuar
    Bq = np.zeros((N, N))
    
    # Solo quarks interactúan
    quarks = np.where(es_quark)[0]
    
    for i in quarks:
        for j in quarks:
            if i >= j:
                continue
            # Condiciones para interacción fuerte:
            # 1. Colores distintos
            # 2. No son ambos materia o ambos antimateria
            if color[i] != color[j] and color[i] >= 0 and color[j] >= 0:
                if es_anti[i] != es_anti[j]:
                    # Interacción fuerte
                    Bq[i,j] = 1.0
                    Bq[j,i] = 1.0
    
    # Detectar tríos confinados (Opción B - triángulo cerrado)
    umbral = 1e-6
    ligado = Bq > umbral
    
    # Índices de quarks vivos
    idxs = np.where((~es_anti) & (color >= 0) & viva)[0]
    
    if len(idxs) < 3:
        return [], Bq
    
    # Ordenar por densidad (que es plana = 1.0, pero mantenemos el mecanismo)
    orden = sorted(idxs, key=lambda i: -densidad[i])
    
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
                if ck != cj and ligado[j, k]:
                    score = min(densidad[i], densidad[j], densidad[k])
                    candidatos.append((score, j, k))
        
        if not candidatos:
            continue
            
        candidatos.sort(key=lambda t: -t[0])
        _, j, k = candidatos[0]
        
        usado[i] = usado[j] = usado[k] = True
        trios.append((i, j, k))
    
    return trios, Bq

# ===================================================================
# 3. CONTADORES
# ===================================================================

def contar_bariones(trios, carga, ne, verbose=False):
    """
    Clasifica tríos en protones y neutrones.
    """
    protones = []
    neutrones = []
    
    for t in trios:
        qsum = int(carga[t[0]]) + int(carga[t[1]]) + int(carga[t[2]])
        if qsum == 3:  # uud
            protones.append(t)
        elif qsum == 0:  # udd
            neutrones.append(t)
    
    # Hidrógeno: protones + electrones (simplificado)
    hidrogeno = min(len(protones), ne)
    
    # Helio: 2p + 2n (simplificado)
    helio = min(len(protones)//2, len(neutrones)//2, ne//2)
    
    if verbose:
        print(f"    Bariones totales: {len(trios)}")
        print(f"    Protones: {len(protones)}")
        print(f"    Neutrones: {len(neutrones)}")
        print(f"    Hidrógeno: {hidrogeno}")
        print(f"    Helio: {helio}")
    
    return {
        'bariones': len(trios),
        'protones': len(protones),
        'neutrones': len(neutrones),
        'hidrogeno': hidrogeno,
        'helio': helio,
        'relacion_pn': len(protones) / max(len(neutrones), 1)
    }

# ===================================================================
# 4. BARRIDO DE ASIMETRÍA
# ===================================================================

def barrido_asimetria(
    N_total=580,
    ne=100,
    npos=70,
    pasos=10,
    verbose=True,
    guardar_json=True
):
    """
    Barre epsilon desde 0 hasta 1 con resolución adaptativa.
    """
    
    # Valores de epsilon - todo el rango
    epsilons = [
        # Simetría casi perfecta (resolución fina)
        0.0, 
        1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 
        5e-6, 1e-5, 5e-5, 1e-4, 5e-4,
        # Transición (resolución media)
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
        # Alta asimetría (resolución gruesa)
        0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0
    ]
    
    resultados = []
    n_total_quarks = N_total - ne - npos
    
    for eps in epsilons:
        if verbose:
            print(f"\n{'='*60}")
            print(f"EPSILON = {eps:.2e}")
            print(f"{'='*60}")
        
        # Calcular nq y naq
        nq = int(n_total_quarks * (1 + eps) / 2)
        naq = n_total_quarks - nq
        
        if verbose:
            print(f"  nq = {nq}, naq = {naq}")
            if naq > 0:
                print(f"  ratio q/qbar = {nq/naq:.3f}")
        
        # Generar catálogo plano
        color, carga, es_anti, es_quark, masa, densidad, temp = catalogo_plano(
            nq, naq, ne, npos
        )
        
        # Verificar densidad plana
        assert np.allclose(densidad, 1.0), "¡La densidad no es plana!"
        
        # Simular
        start = time.time()
        trios, Bq = simular_bariones(
            color, carga, es_anti, es_quark, masa, densidad, temp, pasos
        )
        elapsed = time.time() - start
        
        # Contar
        conteo = contar_bariones(trios, carga, ne, verbose=verbose)
        
        # Registrar
        resultado = {
            'epsilon': eps,
            'nq': nq,
            'naq': naq,
            'bariones': conteo['bariones'],
            'protones': conteo['protones'],
            'neutrones': conteo['neutrones'],
            'relacion_pn': conteo['relacion_pn'],
            'hidrogeno': conteo['hidrogeno'],
            'helio': conteo['helio'],
            'tiempo_ejecucion': elapsed,
            'n_trios': len(trios)
        }
        
        if verbose:
            print(f"\n  ✅ RESULTADOS:")
            print(f"    Bariones: {conteo['bariones']}")
            print(f"    p/n: {conteo['relacion_pn']:.3f}")
            print(f"    H: {conteo['hidrogeno']}")
            print(f"    He: {conteo['helio']}")
            print(f"    Tiempo: {elapsed:.3f}s")
        
        resultados.append(resultado)
    
    # Guardar JSON
    if guardar_json:
        with open('barrido_asimetria_completo.json', 'w') as f:
            json.dump(resultados, f, indent=2)
        print(f"\n✅ Resultados guardados en barrido_asimetria_completo.json")
    
    return resultados

# ===================================================================
# 5. ANÁLISIS Y GRÁFICOS
# ===================================================================

def analizar_barrido(resultados):
    """Genera gráficos de la curva completa."""
    
    # Filtrar resultados válidos
    validos = [r for r in resultados if 'bariones' is not None]
    if not validos:
        print("No hay resultados válidos")
        return None
    
    eps = np.array([r['epsilon'] for r in validos])
    bariones = np.array([r['bariones'] for r in validos])
    pn = np.array([r['relacion_pn'] for r in validos])
    H = np.array([r['hidrogeno'] for r in validos])
    He = np.array([r['helio'] for r in validos])
    
    # Punto de transición
    umbral = 0.01  # 1% del máximo
    max_b = bariones.max()
    idx_transicion = np.where(bariones > umbral * max_b)[0]
    eps_transicion = None
    if len(idx_transicion) > 0:
        eps_transicion = eps[idx_transicion[0]]
        print(f"\n🔍 Punto de transición: ε ≈ {eps_transicion:.2e}")
    
    # Crear figura
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Barrido de Asimetría - Curva Completa', fontsize=14, fontweight='bold')
    
    # 1. Bariones
    axes[0,0].semilogx(eps, bariones, 'o-', color='blue', linewidth=2, markersize=8)
    axes[0,0].set_xlabel('ε (asimetría)')
    axes[0,0].set_ylabel('N_bariones')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axvline(x=1e-9, color='red', linestyle='--', alpha=0.5, label='ε ≈ 1e-9')
    if eps_transicion:
        axes[0,0].axvline(x=eps_transicion, color='green', linestyle='--', alpha=0.5, label=f'ε = {eps_transicion:.2e}')
    axes[0,0].legend()
    axes[0,0].set_title('Emergencia de Bariones')
    
    # 2. Relación p/n
    axes[0,1].semilogx(eps, pn, 'o-', color='green', linewidth=2, markersize=8)
    axes[0,1].set_xlabel('ε (asimetría)')
    axes[0,1].set_ylabel('p/n')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='p = n')
    axes[0,1].legend()
    axes[0,1].set_title('Relación Protón/Neutrón')
    
    # 3. Hidrógeno
    axes[0,2].semilogx(eps, H, 'o-', color='cyan', linewidth=2, markersize=8)
    axes[0,2].set_xlabel('ε (asimetría)')
    axes[0,2].set_ylabel('N_H')
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].set_title('Hidrógeno Emergente')
    
    # 4. Helio
    axes[1,0].semilogx(eps, He, 'o-', color='magenta', linewidth=2, markersize=8)
    axes[1,0].set_xlabel('ε (asimetría)')
    axes[1,0].set_ylabel('N_He')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_title('Helio Emergente')
    
    # 5. Eficiencia (bariones por quark)
    nq = np.array([r['nq'] for r in validos])
    eficiencia = bariones / nq
    axes[1,1].semilogx(eps, eficiencia, 'o-', color='purple', linewidth=2, markersize=8)
    axes[1,1].set_xlabel('ε (asimetría)')
    axes[1,1].set_ylabel('Bariones / nq')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_title('Eficiencia de Confinamiento')
    
    # 6. Comparación H/He
    axes[1,2].semilogx(eps, H, 'o-', color='cyan', linewidth=2, markersize=8, label='H')
    axes[1,2].semilogx(eps, He, 's-', color='magenta', linewidth=2, markersize=8, label='He')
    axes[1,2].set_xlabel('ε (asimetría)')
    axes[1,2].set_ylabel('Número de átomos')
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].legend()
    axes[1,2].set_title('Comparación H vs He')
    
    plt.tight_layout()
    plt.savefig('barrido_asimetria_completo.png', dpi=150)
    print(f"\n✅ Gráfico guardado en barrido_asimetria_completo.png")
    plt.show()
    
    return {
        'eps_transicion': eps_transicion,
        'max_bariones': int(max_b),
        'max_H': int(H.max()),
        'max_He': int(He.max()),
        'pn_en_max': float(pn[np.argmax(bariones)])
    }

# ===================================================================
# 6. MAIN
# ===================================================================

def main():
    print("="*70)
    print("🚀 BARRIDO COMPLETO DE ASIMETRÍA")
    print("="*70)
    print("Barriendo ε desde 0 hasta 1 con resolución adaptativa")
    print("Catálogo PLANO (densidad = 1.0 para todos)")
    print("Simulación simplificada para detección de bariones")
    print("="*70)
    
    # Ejecutar barrido
    resultados = barrido_asimetria(
        N_total=580,
        ne=100,
        npos=70,
        pasos=10,
        verbose=True,
        guardar_json=True
    )
    
    # Análisis
    print("\n" + "="*70)
    print("📊 ANÁLISIS DE LA CURVA COMPLETA")
    print("="*70)
    
    analisis = analizar_barrido(resultados)
    
    if analisis:
        print("\n" + "="*70)
        print("✅ BARRIDO COMPLETADO")
        print("="*70)
        print(f"Punto de transición: ε ≈ {analisis['eps_transicion']:.2e}")
        print(f"Máximo de bariones: {analisis['max_bariones']}")
        print(f"Máximo de H: {analisis['max_H']}")
        print(f"Máximo de He: {analisis['max_He']}")
        print(f"Relación p/n en el máximo: {analisis['pn_en_max']:.3f}")
    
    print("\n" + "="*70)
    print("📁 Archivos generados:")
    print("  - barrido_asimetria_completo.json (datos)")
    print("  - barrido_asimetria_completo.png (gráficos)")
    print("="*70)

if __name__ == "__main__":
    main()