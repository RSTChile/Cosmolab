# smoke test PASO A (despliegue de posiciones 3D) -- p02b_gravedad_general, escala chica.
# corre: PYTHONPATH=. venv/bin/python verificar_p02b_pasoA.py
from cs072_modulos.piezas.p02b_gravedad_general import desplegar_posiciones

r = desplegar_posiciones(nq=300, naq=210, ne=100, npos=70, D_distinciones=3, pasos=150)

print("gate_ok:", r.get("gate_ok"))
print("nota:", r.get("nota"))
gate = r.get("gate") or {}
print("gate.dim_efectiva:", gate.get("dim_efectiva"), " gate.pendiente:", gate.get("pendiente"))

if r.get("gate_ok"):
    print("n_atomos:", r["n_atomos"], " pares_desconectados:", r["pares_desconectados"])
    print("varianza_explicada (3 dims) REAL:", r["varianza_explicada_dims"],
          " vs NULL:", r["null_varianza_explicada_dims"])
    print("dims_para_90pct REAL:", r["dims_para_90pct"], " vs NULL:", r["null_dims_para_90pct"])
    print("A.4 distinguible del NULL:", r["a4_referente_fisico_distinguible_de_null"])
