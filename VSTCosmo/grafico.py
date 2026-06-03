import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Función Sigmoide (Logística)
def sigmoid(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))

# Parámetros del modelo (según el paper)
K = 120.0   # Capacidad de carga (Asíntota)
r = 0.3     # Tasa de crecimiento (Cladogénesis)
t0 = 2028.0 # Punto de inflexión

# Datos empíricos conocidos (Ejemplos acumulados por año)
# 2016: 1 (AlphaGo), 2020: ~10, 2022: ~20, 2024: ~33, 2025: ~45, 2026 (mid): 54
years_data = np.array([2016, 2018, 2020, 2022, 2024, 2025, 2026.5])
exaptations_data = np.array([1, 4, 10, 20, 33, 45, 54])

# Generar curva de proyección (2016 a 2035)
years_proj = np.linspace(2016, 2035, 500)
proj_values = sigmoid(years_proj, K, r, t0)

# Configurar el gráfico para paper científico (Escala de grises, alta resolución)
plt.figure(figsize=(8, 5), dpi=300)

# 1. Dibujar la curva de proyección
plt.plot(years_proj, proj_values, 'k-', linewidth=2.5, label='Sigmoidal Projection (Logistic Model)')

# 2. Sombrear la fase de estasis (Capacidad de carga)
plt.fill_between(years_proj, proj_values, where=(years_proj >= 2030), 
                 color='gray', alpha=0.3, label='Stasis Phase (Carrying Capacity $K \\approx 120$)')

# 3. Plotear los datos empíricos históricos
plt.scatter(years_data[:-1], exaptations_data[:-1], color='black', s=40, zorder=5, label='Historical Accumulation')

# 4. Plotear el punto empírico CRÍTICO de 2026 (Validación del paper)
plt.scatter([2026.5], [54], color='darkred', s=100, zorder=6, edgecolor='black', linewidth=1.5, label='Empirical Validation (Mid-2026: N=54)')

# Anotación para el punto de 2026
plt.annotate('Current Empirical\nData (N=54)', 
             xy=(2026.5, 54), xytext=(2022, 75),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, color='black'),
             fontsize=10, fontweight='bold', ha='center')

# Anotación para la meseta
plt.annotate('Predicted Stasis\n(Paradigm Limit)', 
             xy=(2033, 115), xytext=(2031, 90),
             arrowprops=dict(facecolor='gray', shrink=0.05, width=1.5, headwidth=8, color='gray'),
             fontsize=10, ha='center', color='dimgray')

# Formato y etiquetas
plt.title('Figure A1. Sigmoidal Projection of AI Exaptations (2016–2035)', fontsize=12, fontweight='bold')
plt.xlabel('Year', fontsize=11)
plt.ylabel('Cumulative Number of Validated Exaptations', fontsize=11)
plt.xlim(2016, 2035)
plt.ylim(0, 135)
plt.xticks(np.arange(2016, 2036, 2), fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', fontsize=9, framealpha=0.9)

# Guardar imagen
os.makedirs('lisci2026_submission', exist_ok=True)
plt.tight_layout()
plt.savefig('lisci2026_submission/Fig_A1_Sigmoid_Exaptations.png', dpi=300, bbox_inches='tight')
plt.savefig('lisci2026_submission/Fig_A1_Sigmoid_Exaptations.pdf', dpi=300, bbox_inches='tight') # Formato vectorial para la revista
print("✅ Gráfico generado y guardado en la carpeta 'lisci2026_submission' (PNG y PDF vectorial).")
plt.show()




López Tapia, A., & Transintelligent Research Team (GPT-OpenAI, Grok-xAI, DeepSeek, Meta AI, Qwen-Alibaba Cloud). (2026). ANIMA-1 Experimental Series (V122–V150): First Minimal Non-Biological Organic Intelligence (IONB-1) under Cosmosemiotic Constraints. Cosmolab Repository, GitHub. Available at: https://github.com/RSTChile/Cosmolab/tree/main/VSTCosmo
[Includes: Python source code, experimental logs, canonical closure report, and reproducibility protocol.]