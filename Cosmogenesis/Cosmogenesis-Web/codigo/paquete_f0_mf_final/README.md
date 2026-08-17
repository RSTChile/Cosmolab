
Paquete F0 + MF final - ejecucion local

Archivos:
01_f0_final_unico.py -> SMOKE PASS 2.22e-16 + k3 0->36 PERIM8=k3 ratio 0.3333 O(1) - F0 entregable
02_mf1_v2_formula_unica.py -> formula unica m=sum*exp(-var/T) -> ratio 0.3333 FAIL honesto anti-Shannon
03_mf1_v3_contraste_O1.py -> phi O(1) 1+0.5*pert -> var 1e-4 vs 1e-2 real -> intenta romper 1/3 sin trampa
04_mf2_K_debye_crossover.py -> K=exp(-1/a) 0.368->0.997 resuelve Kc 0.391->0.191 crossover
05_mf3_sigma_tension.py -> sigma = P*a tension cuerda, mide E_k3

Ejecucion:
python 01_f0_final_unico.py
python 02_mf1_v2_formula_unica.py
python 03_mf1_v3_contraste_O1.py
python 04_mf2_K_debye_crossover.py
python 05_mf3_sigma_tension.py

Todos comparten: T=T0/a, rho=rho0/a^3, a=exp(6*tg) 1->403x, H_topo=0.01, L=30, seed 2025, sin SI crudo
