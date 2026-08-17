# barrido_asimetria.py
import numpy as np
import json
import time
from cs072_modulos.catalogo_plano import catalogo_plano
from cs072_modulos.nucleo import corre

def barrido_completo(
    N_total=580,
    ne=100,
    npos=70,
    pasos=400,
    T0=3.0,
    tasa_expansion=0.02,
    amp_rugosidad=0.5,
    verbose=True,
    guardar_resultados=True
):
    """
    Barre epsilon desde 0 hasta 1.
    Usa catálogo plano (sin densidad pre-asignada).
    La densidad emerge de la dinámica.
    """
    
    # Valores de epsilon a barrer (todo el rango)
    epsilons = [
        # Zona de simetría casi perfecta (resolución fina)
        0.0, 
        1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 
        5e-6, 1e-5, 5e-5, 1e-4, 5e-4,
        # Zona de transición (resolución media)
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
        # Zona de alta asimetría (resolución gruesa)
        0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0
    ]
    
    resultados = []
    n_total_quarks = N_total - ne - npos
    
    for eps in epsilons:
        if verbose:
            print(f"\n{'='*60}")
            print(f"EPSILON = {eps:.2e}")
            print(f"{'='*60}")
        
        # Calcular número de quarks y antiquarks
        nq = int(n_total_quarks * (1 + eps) / 2)
        naq = n_total_quarks - nq
        
        if verbose:
            print(f"  nq = {nq}, naq = {naq}")
            if naq > 0:
                print(f"  ratio q/qbar = {nq/naq:.3f}")
        
        # Generar catálogo PLANO (sin densidad)
        color, carga, es_anti, es_quark, masa, densidad, temp = catalogo_plano(
            nq, naq, ne, npos
        )
        
        # Verificar que la densidad es plana
        assert np.all(densidad == 1.0), "La densidad no es plana!"
        
        # Ejecutar simulación
        try:
            start = time.time()
            obs = corre(
                nq, naq, ne, npos,
                amp_asimetria=0.0,
                tasa_expansion=tasa_expansion,
                pasos=pasos,
                T0=T0,
                amp_rugosidad=amp_rugosidad,
                devolver_estado=False
            )
            elapsed = time.time() - start
            
            # Extraer resultados
            bariones = obs.get('bariones', 0)
            protones = obs.get('protones', 0)
            neutrones = obs.get('neutrones', 0)
            hidrogeno = obs.get('hidrogeno', 0)
            helio = obs.get('helio', 0)
            diametro = obs.get('diametro_red', 0)
            geometria = obs.get('geometria', {})
            
            resultado = {
                'epsilon': eps,
                'nq': nq,
                'naq': naq,
                'bariones': bariones,
                'protones': protones,
                'neutrones': neutrones,
                'relacion_pn': protones / max(neutrones, 1),
                'hidrogeno': hidrogeno,
                'helio': helio,
                'diametro_red': diametro,
                'n_nodos_atomo': geometria.get('n_nodos_atomo', 0),
                'espacio_emergio': geometria.get('espacio_emergio', False),
                'tiempo_ejecucion': elapsed,
                'epocas': obs.get('epocas', {})
            }
            
            if verbose:
                print(f"\n  RESULTADOS:")
                print(f"    Bariones: {bariones}")
                print(f"    p/n: {resultado['relacion_pn']:.3f}")
                print(f"    H: {hidrogeno}")
                print(f"    He: {helio}")
                print(f"    Diámetro red: {diametro}")
                print(f"    Espacio emergió: {geometria.get('espacio_emergio', False)}")
                print(f"    Tiempo: {elapsed:.2f}s")
            
            resultados.append(resultado)
            
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            resultados.append({
                'epsilon': eps,
                'error': str(e),
                'bariones': 0
            })
    
    # Guardar resultados
    if guardar_resultados:
        with open('barrido_asimetria_completo.json', 'w') as f:
            json.dump(resultados, f, indent=2)
        print(f"\n✅ Resultados guardados en barrido_asimetria_completo.json")
    
    return resultados

def analizar_barrido(resultados):
    """Análisis de la curva completa."""
    import matplotlib.pyplot as plt
    
    # Filtrar resultados válidos
    validos = [r for r in resultados if 'error' not in r]
    if not validos:
        print("No hay resultados válidos")
        return
    
    eps = np.array([r['epsilon'] for r in validos])
    bariones = np.array([r['bariones'] for r in validos])
    pn = np.array([r['relacion_pn'] for r in validos])
    H = np.array([r['hidrogeno'] for r in validos])
    He = np.array([r['helio'] for r in validos])
    diametro = np.array([r['diametro_red'] for r in validos])
    
    # Encontrar punto de transición
    umbral = 0.01  # 1% del máximo
    max_b = bariones.max()
    idx_transicion = np.where(bariones > umbral * max_b)[0]
    if len(idx_transicion) > 0:
        eps_transicion = eps[idx_transicion[0]]
        print(f"\n🔍 Punto de transición: ε ≈ {eps_transicion:.2e}")
    
    # Gráficos
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Bariones
    axes[0,0].semilogx(eps, bariones, 'o-', color='blue', linewidth=2)
    axes[0,0].set_xlabel('ε (asimetría)')
    axes[0,0].set_ylabel('N_bariones')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axvline(x=1e-9, color='red', linestyle='--', alpha=0.5, label='ε ≈ 1e-9 (observado)')
    axes[0,0].legend()
    axes[0,0].set_title('Emergencia de Bariones')
    
    # 2. Relación p/n
    axes[0,1].semilogx(eps, pn, 'o-', color='green', linewidth=2)
    axes[0,1].set_xlabel('ε (asimetría)')
    axes[0,1].set_ylabel('p/n')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='p = n')
    axes[0,1].legend()
    axes[0,1].set_title('Relación Protón/Neutrón')
    
    # 3. Hidrógeno
    axes[0,2].semilogx(eps, H, 'o-', color='cyan', linewidth=2)
    axes[0,2].set_xlabel('ε (asimetría)')
    axes[0,2].set_ylabel('N_H')
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].set_title('Hidrógeno Emergente')
    
    # 4. Helio
    axes[1,0].semilogx(eps, He, 'o-', color='magenta', linewidth=2)
    axes[1,0].set_xlabel('ε (asimetría)')
    axes[1,0].set_ylabel('N_He')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_title('Helio Emergente')
    
    # 5. Diámetro de la red
    axes[1,1].semilogx(eps, diametro, 'o-', color='orange', linewidth=2)
    axes[1,1].set_xlabel('ε (asimetría)')
    axes[1,1].set_ylabel('Diámetro')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_title('Espacio Emergente (diámetro red)')
    
    # 6. Eficiencia (bariones por quark)
    nq = np.array([r['nq'] for r in validos])
    eficiencia = bariones / nq
    axes[1,2].semilogx(eps, eficiencia, 'o-', color='purple', linewidth=2)
    axes[1,2].set_xlabel('ε (asimetría)')
    axes[1,2].set_ylabel('Bariones / nq')
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].set_title('Eficiencia de Confinamiento')
    
    plt.tight_layout()
    plt.savefig('barrido_asimetria_completo.png', dpi=150)
    plt.show()
    
    return {
        'eps_transicion': eps_transicion if 'eps_transicion' in locals() else None,
        'max_bariones': max_b,
        'max_H': H.max(),
        'max_He': He.max(),
        'pn_en_max': pn[np.argmax(bariones)]
    }

if __name__ == "__main__":
    print("🚀 INICIANDO BARRIDO COMPLETO DE ASIMETRÍA")
    print("="*60)
    print("Barriendo ε desde 0 hasta 1 con resolución adaptativa")
    print("Catálogo PLANO (sin densidad pre-asignada)")
    print("="*60)
    
    resultados = barrido_completo(verbose=True)
    
    print("\n" + "="*60)
    print("📊 ANÁLISIS DE LA CURVA COMPLETA")
    print("="*60)
    
    analisis = analizar_barrido(resultados)
    
    print("\n" + "="*60)
    print("✅ BARRIDO COMPLETADO")
    print("="*60)
    print(f"Punto de transición: ε ≈ {analisis['eps_transicion']:.2e}")
    print(f"Máximo de bariones: {analisis['max_bariones']}")
    print(f"Relación p/n en el máximo: {analisis['pn_en_max']:.3f}")