# PASO A a escala (~750 átomos, f=15 sobre la línea base nq=300) -- NULL como distribución (n_null=8) + z-score.
# Gate desacoplado a la línea base barata (300,210,100,70) -- ver docstring de desplegar_posiciones.
# corre: PYTHONPATH=. venv/bin/python verificar_p02b_pasoA_escala.py
import time
from cs072_modulos.piezas.p02b_gravedad_general import desplegar_posiciones

t0 = time.time()
r = desplegar_posiciones(nq=4500, naq=3150, ne=1500, npos=1050, D_distinciones=3, pasos=150, n_null=8)
dt = time.time() - t0

print("tiempo total: %.1fs" % dt)
print("gate_ok:", r.get("gate_ok"))
print("nota:", r.get("nota"))
gate = r.get("gate") or {}
print("gate.dim_efectiva:", gate.get("dim_efectiva"), " gate.pendiente:", gate.get("pendiente"))

if r.get("gate_ok"):
    print("n_atomos:", r["n_atomos"], " n_null:", r["n_null"], " pares_desconectados:", r["pares_desconectados"])
    print("varianza_explicada (3 dims) REAL:", r["varianza_explicada_dims"])
    print("  NULL media:", r["null_varianza_explicada_dims_media"], " std:", r["null_varianza_explicada_dims_std"])
    print("  z:", r["z_varianza_explicada_dims"])
    print("dims_para_90pct REAL:", r["dims_para_90pct"])
    print("  NULL media:", r["null_dims_para_90pct_media"], " std:", r["null_dims_para_90pct_std"])
    print("  z:", r["z_dims_para_90pct"])
    print("A.4 distinguible del NULL (|z|>=%.1f):" % r["z_umbral"], r["a4_referente_fisico_distinguible_de_null"])
