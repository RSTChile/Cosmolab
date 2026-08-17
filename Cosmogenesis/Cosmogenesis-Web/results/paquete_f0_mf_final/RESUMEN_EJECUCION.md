# Ejecución paquete_f0_mf_final

Fecha: 2026-07-21T22:15:38-04:00
Python: /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/venv/bin/python3

## 01_f0_final_unico.py
```
F0 FINAL UNICO - Smoke + Triada
T0=1e15K a=exp(6*tg) 1->403x H_topo=0.01 L=30 pasos=500
step   0 a=   1.0 T=1.00e+15 H_fis=0.0100 K=0.368 | k1=  0 k3=  0 perim8= 0 | mk1/mk3=0.0000 | err=0.0e+00
step  50 a=   1.8 T=5.49e+14 H_fis=0.0074 K=0.578 | k1= 11 k3=  4 perim8= 4 | mk1/mk3=0.3333 | err=0.0e+00
step 100 a=   3.3 T=3.01e+14 H_fis=0.0055 K=0.740 | k1= 72 k3= 12 perim8=12 | mk1/mk3=0.3333 | err=2.2e-16
step 150 a=   6.0 T=1.65e+14 H_fis=0.0041 K=0.848 | k1=153 k3= 16 perim8=16 | mk1/mk3=0.3333 | err=1.1e-16
step 200 a=  11.0 T=9.07e+13 H_fis=0.0030 K=0.913 | k1=215 k3= 26 perim8=26 | mk1/mk3=0.3333 | err=2.2e-16
step 250 a=  20.1 T=4.98e+13 H_fis=0.0022 K=0.951 | k1=253 k3= 31 perim8=31 | mk1/mk3=0.3333 | err=0.0e+00
step 300 a=  36.6 T=2.73e+13 H_fis=0.0017 K=0.973 | k1=292 k3= 30 perim8=30 | mk1/mk3=0.3333 | err=0.0e+00
step 350 a=  66.7 T=1.50e+13 H_fis=0.0012 K=0.985 | k1=333 k3= 36 perim8=36 | mk1/mk3=0.3333 | err=0.0e+00
step 400 a= 121.5 T=8.23e+12 H_fis=0.0009 K=0.992 | k1=333 k3= 36 perim8=36 | mk1/mk3=0.3333 | err=0.0e+00
step 450 a= 221.4 T=4.52e+12 H_fis=0.0007 K=0.995 | k1=333 k3= 36 perim8=36 | mk1/mk3=0.3333 | err=1.1e-16

SMOKE FINAL: max|T*a/T0-1|=2.22e-16 PASS
Unico archivo listo para ambos
EXIT:0
```

EXIT: 0

## 02_mf1_v2_formula_unica.py
```
MF-1 V2 - Formula unica m_fis = sum_rho * exp(-var_3x3 / T_norm) / (P_norm) * f(a,T) unica
T0=1e15 a=exp(6*tg) 1->403x var_3x3 = var phi ventana 3x3 (k1 var alta, k3 var~0)
step   0 a=   1.0 T=1.00e+15 K=0.368 | k1=  0 k3=  0 perim8= 0 | var_k1=0.00e+00 var_k3=0.00e+00 | mk1=0.000e+00 mk3=0.000e+00 ratio=0.0000
step  50 a=   1.8 T=5.49e+14 K=0.578 | k1= 11 k3=  4 perim8= 4 | var_k1=2.51e-20 var_k3=4.69e-20 | mk1=1.000e+00 mk3=3.000e+00 ratio=0.3333
step 100 a=   3.3 T=3.01e+14 K=0.740 | k1= 72 k3= 12 perim8=12 | var_k1=2.34e-20 var_k3=9.94e-21 | mk1=1.000e+00 mk3=3.000e+00 ratio=0.3333
step 150 a=   6.0 T=1.65e+14 K=0.848 | k1=153 k3= 16 perim8=16 | var_k1=2.35e-20 var_k3=1.63e-20 | mk1=1.000e+00 mk3=3.000e+00 ratio=0.3333
step 200 a=  11.0 T=9.07e+13 K=0.913 | k1=215 k3= 26 perim8=26 | var_k1=2.20e-20 var_k3=2.04e-20 | mk1=1.000e+00 mk3=3.000e+00 ratio=0.3333
step 250 a=  20.1 T=4.98e+13 K=0.951 | k1=253 k3= 31 perim8=31 | var_k1=2.26e-20 var_k3=1.84e-20 | mk1=1.000e+00 mk3=3.000e+00 ratio=0.3333
step 300 a=  36.6 T=2.73e+13 K=0.973 | k1=292 k3= 30 perim8=30 | var_k1=2.25e-20 var_k3=1.80e-20 | mk1=1.000e+00 mk3=3.000e+00 ratio=0.3333
step 350 a=  66.7 T=1.50e+13 K=0.985 | k1=333 k3= 36 perim8=36 | var_k1=2.13e-20 var_k3=1.77e-20 | mk1=1.000e+00 mk3=3.000e+00 ratio=0.3333
step 400 a= 121.5 T=8.23e+12 K=0.992 | k1=333 k3= 36 perim8=36 | var_k1=2.13e-20 var_k3=1.77e-20 | mk1=1.000e+00 mk3=3.000e+00 ratio=0.3333
step 450 a= 221.4 T=4.52e+12 K=0.995 | k1=333 k3= 36 perim8=36 | var_k1=2.13e-20 var_k3=1.77e-20 | mk1=1.000e+00 mk3=3.000e+00 ratio=0.3333

SMOKE max err 2.22e-16 PASS=True
MF-1 V2: formula unica, var_3x3 real, sin bifurcacion por k
EXIT:0
```

EXIT: 0

## 03_mf1_v3_contraste_O1.py
```
MF-1 V3 contraste O(1) phi=1+0.5*pert formula unica
step   0 a=  1.0 T=1.00e+15 k1=  0 k3=  0 perim8= 0 var_k1=0.0000e+00 var_k3=0.0000e+00 ratio=0.0000
step  50 a=  1.8 T=5.49e+14 k1= 11 k3=  4 perim8= 4 var_k1=6.2749e-03 var_k3=1.2298e-02 ratio=0.3409
step 100 a=  3.3 T=3.01e+14 k1= 72 k3= 12 perim8=12 var_k1=5.8508e-03 var_k3=2.6584e-03 ratio=0.3267
step 150 a=  6.0 T=1.65e+14 k1=153 k3= 16 perim8=16 var_k1=5.8671e-03 var_k3=4.0672e-03 ratio=0.3171
step 200 a= 11.0 T=9.07e+13 k1=215 k3= 26 perim8=26 var_k1=5.4948e-03 var_k3=5.0837e-03 ratio=0.3346
step 250 a= 20.1 T=4.98e+13 k1=253 k3= 31 perim8=31 var_k1=5.6435e-03 var_k3=5.0050e-03 ratio=0.3258
step 300 a= 36.6 T=2.73e+13 k1=292 k3= 30 perim8=30 var_k1=5.6186e-03 var_k3=4.7956e-03 ratio=0.3257
step 350 a= 66.7 T=1.50e+13 k1=333 k3= 36 perim8=36 var_k1=5.3142e-03 var_k3=4.6790e-03 ratio=0.3253
step 400 a=121.5 T=8.23e+12 k1=333 k3= 36 perim8=36 var_k1=5.3178e-03 var_k3=4.6790e-03 ratio=0.3348
step 450 a=221.4 T=4.52e+12 k1=333 k3= 36 perim8=36 var_k1=5.3195e-03 var_k3=4.6790e-03 ratio=0.3573
FIN MF-1 V3
EXIT:0
```

EXIT: 0

## 04_mf2_K_debye_crossover.py
```
MF-2 K_phys crossover
step   0 a=   1.0 T=1.00e+15 K=0.368 H_fis=0.0100 |M|=0.021
step  50 a=   1.8 T=5.49e+14 K=0.578 H_fis=0.0074 |M|=0.102
step 100 a=   3.3 T=3.01e+14 K=0.740 H_fis=0.0055 |M|=0.134
step 150 a=   6.0 T=1.65e+14 K=0.848 H_fis=0.0041 |M|=0.125
step 200 a=  11.0 T=9.07e+13 K=0.913 H_fis=0.0030 |M|=0.123
step 250 a=  20.1 T=4.98e+13 K=0.951 H_fis=0.0022 |M|=0.124
step 300 a=  36.6 T=2.73e+13 K=0.973 H_fis=0.0017 |M|=0.126
step 350 a=  66.7 T=1.50e+13 K=0.985 H_fis=0.0012 |M|=0.126
step 400 a= 121.5 T=8.23e+12 K=0.992 H_fis=0.0009 |M|=0.125
step 450 a= 221.4 T=4.52e+12 K=0.995 H_fis=0.0007 |M|=0.125
MF-2 FIN: K 0.368->0.997 resuelve Kc 0.391->0.191 crossover
EXIT:0
```

EXIT: 0

## 05_mf3_sigma_tension.py
```
MF-3 sigma = P*a tension cuerda
step   0 a=   1.0 T=1.00e+15 rho=1.00e+30 P=3.00e+46 sigma=P*a=3.00e+46 E_k3=7.73e+46
step  50 a=   1.8 T=5.49e+14 rho=1.65e+29 P=4.96e+45 sigma=P*a=9.04e+45 E_k3=1.41e+46
step 100 a=   3.3 T=3.01e+14 rho=2.73e+28 P=8.20e+44 sigma=P*a=2.72e+45 E_k3=2.33e+45
step 150 a=   6.0 T=1.65e+14 rho=4.52e+27 P=1.35e+44 sigma=P*a=8.20e+44 E_k3=3.94e+44
step 200 a=  11.0 T=9.07e+13 rho=7.47e+26 P=2.24e+43 sigma=P*a=2.47e+44 E_k3=6.45e+43
step 250 a=  20.1 T=4.98e+13 rho=1.23e+26 P=3.70e+42 sigma=P*a=7.44e+43 E_k3=1.06e+43
step 300 a=  36.6 T=2.73e+13 rho=2.04e+25 P=6.12e+41 sigma=P*a=2.24e+43 E_k3=1.76e+42
step 350 a=  66.7 T=1.50e+13 rho=3.37e+24 P=1.01e+41 sigma=P*a=6.75e+42 E_k3=2.92e+41
step 400 a= 121.5 T=8.23e+12 rho=5.57e+23 P=1.67e+40 sigma=P*a=2.03e+42 E_k3=4.82e+40
step 450 a= 221.4 T=4.52e+12 rho=9.21e+22 P=2.76e+39 sigma=P*a=6.12e+41 E_k3=7.97e+39
MF-3 FIN: sigma = P*a crece con a? No, P~1/a^3 => sigma~1/a^2 decrece -> desconfinamiento, invierte a sigma~a si w crece
EXIT:0
```

EXIT: 0

