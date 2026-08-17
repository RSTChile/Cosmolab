# CF-4b -- barrido de acoplamiento gamma (resultado crudo, sin adjudicar)

Protocolo: `CF4b_ACOPLAMIENTO_2026-07-24`

- rango gamma barrido: [0.0005, 50] (referencia CF-4: gamma=0.05)
- H_TOPO elegidos: [0.04, 0.1] (ver protocolo secc.2)
- semillas: [7, 42, 99, 777]
- n instancias de cierre medidas: **523614**
- corridas totales: **128**, divergieron: **32**
- max ratio_lig_median (CRUDO, incluye puntos 100% divergidos): **{'gamma': 50.0, 'H_TOPO': '0.040', 'ratio_lig_median': 109.47100707017007}**
- max ratio_lig_median (RELIABLE, excluye puntos donde TODAS las semillas divergieron numericamente): **{'gamma': 2.0, 'H_TOPO': '0.100', 'ratio_lig_median': 2.0913384425439996}**
- max ratio_null_median (CRUDO): **{'gamma': 50.0, 'H_TOPO': '0.040', 'ratio_null_median': 1.4317175896118695}**
- max ratio_null_median (RELIABLE): **{'gamma': 0.001, 'H_TOPO': '0.100', 'ratio_null_median': 1.1092350940414502}**
- bandas estables, criterio congelado CRUDO (>=3 puntos contiguos, ambos umbrales, incluye puntos divergidos): **{'0.040': [], '0.100': []}**
- bandas estables RELIABLE (excluye puntos donde el 100% de semillas divergio): **{'0.040': [], '0.100': []}**
- umbrales heredados (congelados): ratio_lig>=5.0, ratio_null>=1.25

Ver JSON completo para la curva punto a punto (incluye n_seeds_diverged y all_seeds_diverged por punto) y la tabla de estabilidad completa. No se adjudica aqui si existe o no el regimen -- lo decide CS con la curva a la vista.
