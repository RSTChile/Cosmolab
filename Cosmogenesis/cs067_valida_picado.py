"""
CS067 — revalidación de Guarda 1 (picado_por_nodo) contra la tabla plantada del ADDENDUM_CS067 (CS, 15-jul).
Corre ANTES de cualquier sweep real: si no reproduce el patrón discreto~1.00 / smear<0.85 / ortho+ruido>=0.85,
el candado está mal y nada se corre (regla explícita de CS).

Nota de honestidad: los generadores plantados son mi propia reconstrucción a partir de la DESCRIPCIÓN de CS
(no tengo su código exacto), así que los decimales no tienen por qué calzar bit a bit con su tabla — lo que
importa es el PATRÓN: discreto=1.00 exacto, smear/artefacto por debajo del piso 0.85, ortho+ruido por encima.
"""
import numpy as np
from cs067_habitacion_completa import picado_por_nodo, cuenta_ejes_gap
import cs064_smoke as SM

RNG = np.random.default_rng(1234)
D = 8
N = 4000


def _norm(V):
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)


def caso_colapso1():
    V = np.zeros((N, D)); V[:, 0] = RNG.choice([-1.0, 1.0], N)
    return V

def caso_3discreto():
    V = np.zeros((N, D)); ax = RNG.integers(0, 3, N); sg = RNG.choice([-1.0, 1.0], N)
    V[np.arange(N), ax] = sg
    return V

def caso_5continuo_smear():
    V = np.zeros((N, D)); V[:, :5] = RNG.standard_normal((N, 5))
    return _norm(V)

def caso_6desigual_2ceros():
    # varianza DESIGUAL entre 6 dims (no isotrópica) + 2 dims en cero exacto — sigue siendo smear continuo,
    # solo que la desigualdad de varianza empuja el pico_medio un poco por encima del caso isotrópico 5D,
    # sin cruzar el piso 0.85 (el patrón que importa: sigue siendo artefacto, no dominio discreto).
    stds = np.array([1.0, 0.85, 0.7, 0.6, 0.5, 0.4])
    V = np.zeros((N, D)); V[:, :6] = RNG.standard_normal((N, 6)) * stds
    return _norm(V)

def caso_3ortogonal_ruido():
    V = np.zeros((N, D)); ax = RNG.integers(0, 3, N); sg = RNG.choice([-1.0, 1.0], N)
    V[np.arange(N), ax] = sg * 0.92
    V += RNG.normal(0, 0.10, (N, D))
    return _norm(V)

def caso_isotropo_8d():
    V = RNG.standard_normal((N, D))
    return _norm(V)

def caso_5ortogonal_onehot():
    V = np.zeros((N, D)); ax = RNG.integers(0, 5, N); sg = RNG.choice([-1.0, 1.0], N)
    V[np.arange(N), ax] = sg
    return V


CASOS = [
    ("colapso-1 (rango 1)", caso_colapso1, 1.00),
    ("3-discreto balanceado", caso_3discreto, 1.00),
    ("5-continuo subespacio (SMEAR)", caso_5continuo_smear, 0.738),
    ("6-desigual + 2 ceros (artefacto)", caso_6desigual_2ceros, 0.771),
    ("3-ortogonal + ruido", caso_3ortogonal_ruido, 0.927),
]

print("=" * 88)
print("CS067 — revalidación Guarda 1: picado_por_nodo (mi implementación) vs patrón plantado de CS")
print("=" * 88)
print(f"{'caso':<34}{'pico_medio (mío)':>18}{'frac_picados':>15}{'ref. CS':>10}{'lectura':>12}")
ok1 = True
for nombre, gen, ref in CASOS:
    V = gen()
    pico_medio, frac_picados = picado_por_nodo(V)
    lectura = "dominio" if pico_medio >= 0.85 else "SMEAR"
    print(f"{nombre:<34}{pico_medio:>18.3f}{frac_picados:>15.3f}{ref:>10.3f}{lectura:>12}")
    if nombre.startswith("colapso") or nombre.startswith("3-discreto"):
        ok1 &= abs(pico_medio - 1.00) < 1e-6
    if nombre.startswith("5-continuo") or nombre.startswith("6-desigual"):
        ok1 &= pico_medio < 0.85
    if nombre.startswith("3-ortogonal"):
        ok1 &= pico_medio >= 0.85

print("\nPATRÓN ESPERADO: discreto=1.00 exacto, smear/artefacto <0.85, ortho+ruido >=0.85.")
print("VALIDACIÓN Guarda 1:", "PASA" if ok1 else "FALLA — revisar antes de correr nada más.")

print("\n" + "=" * 88)
print("CS067 — revalidación Guarda 2: cuenta_ejes_gap (n_ejes) vs tabla 1/3/0/3/5 de CS")
print("=" * 88)
TABLA = [
    ("todos en e_1 (one-hot)", caso_colapso1, 1),
    ("1/3 en e1,e2,e3 (one-hot)", caso_3discreto, 3),
    ("esfera uniforme 8D", caso_isotropo_8d, 0),
    ("3 ortogonales + ruido 15%", caso_3ortogonal_ruido, 3),
    ("5 ortogonales one-hot", caso_5ortogonal_onehot, 5),
]
print(f"{'planta':<30}{'n_ejes (mío)':>14}{'esperado':>10}{'PR':>7}{'gap_interno':>13}")
ok2 = True
for nombre, gen, esperado in TABLA:
    V = gen()
    ev = SM.tensor_orientacion(V)
    nej, PR, gap_interno, rthr = cuenta_ejes_gap(ev)
    print(f"{nombre:<30}{nej:>14d}{esperado:>10d}{PR:>7.2f}{gap_interno:>13.2f}")
    ok2 &= (nej == esperado)

print("\nVALIDACIÓN Guarda 2:", "PASA — reproduce 1/3/0/3/5" if ok2 else "FALLA — revisar antes de correr nada más.")
print("\nVEREDICTO GLOBAL:", "AMBAS GUARDAS PASAN, se puede correr el sweep." if (ok1 and ok2) else "NO CORRER NADA MÁS — hay una guarda rota.")
