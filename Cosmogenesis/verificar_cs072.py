# guarda esto como verificar_cs072.py en Cosmogenesis/ y corre:  PYTHONPATH=. python verificar_cs072.py
from cs072_modulos.proceso_sucesivo import proceso_sucesivo

print("=== 1) MOTOR COMPLETO D=3 (este universo) ===")
r = proceso_sucesivo(nq=300, naq=210, ne=100, npos=70, D_distinciones=3, pasos=150, medir_acoplada=True)
print("  bariones          =", r["bariones"],          " (esperado 100)")
print("  ratio p:n         =", r["ratio_pn_congelado"]," (esperado 7.1)")
print("  hidrogeno         =", r["hidrogeno"],         " (esperado 50)")
print("  helio             =", r["helio"],             " (esperado 25)")
print("  tiempo            =", r["tiempo"]["tiempo_emergente"], " (esperado 75 = H+He)")
print("  dim ACOPLADA      =", r["dimension_acoplada"]["dim_efectiva"], " (esperado ~2.0-2.4, átomos reales)")
print("  dim ENSEMBLE      =", r["dimension"]["dim_efectiva"],          " (esperado ~2.77, ley del régimen)")
print("  oscura_necesaria  =", r["materia_oscura"]["oscura_necesaria"], " (esperado True)")
print("  invariante        =", r["invariancia"]["invariante"],          " (esperado True)")

print("\n=== 2) BARRIDO DE BANDA: dimensión vs nº de distinciones (otros universos posibles) ===")
print("  D | dim_ensemble | invariante   (esperado: crece con D, invariante en las 5)")
for D in [1,2,3,4,5]:
    r = proceso_sucesivo(nq=300, naq=210, ne=100, npos=70, D_distinciones=D, pasos=100)  # sin acoplada = rápido
    print("  %d |    %-5s     | %s" % (D, r["dimension"]["dim_efectiva"], r["invariancia"]["invariante"]))
# esperado aprox: D1=1.0, D2=2.24, D3=2.77, D4=3.33, D5=3.41 ; invariante=True en todas

print("\n=== 3) GUARDIÁN anti-Shannon: apagar una fuerza destruye su estructura ===")
print("  (nombres-clave EXACTOS: 3_fuerte, 4_em, 8_aniquilacion, 2_gravedad, 23_fluctuaciones)")
r_fuerte = proceso_sucesivo(nq=300,naq=210,ne=100,npos=70,D_distinciones=3,pasos=150,apagar=frozenset(["3_fuerte"]))
print("  apagar 3_fuerte    : H=%s He=%s   (esperado He=0: sin fuerza fuerte no hay fusión)" % (r_fuerte["hidrogeno"], r_fuerte["helio"]))
r_em = proceso_sucesivo(nq=300,naq=210,ne=100,npos=70,D_distinciones=3,pasos=150,apagar=frozenset(["4_em"]),medir_acoplada=True)
print("  apagar 4_em        : H=%s He=%s   (esperado H=0: sin EM no se liga protón+electrón)" % (r_em["hidrogeno"], r_em["helio"]))
print("  apagar 4_em dim_acoplada =", r_em["dimension_acoplada"]["dim_efectiva"], " (esperado None: sin átomos, la geometría COLAPSA)")
