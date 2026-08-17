"""
grafo_random_bateria_correr.py — corre Phantom sobre las 8 condiciones del control grafo-random de la
batería N=2000 (semillas 701-708). Mismo patrón EXACTO que `null3_bateria_correr.py`/
`null2_bateria_correr.py`/`null1_bateria_correr.py` (congelados, no se tocan directamente): por cada
carpeta `/Users/alexis/phantom_cs073/bateria_grafo_random_n2000/ic_random_s{701..708}/` (ya con su
`cosmogenesis_ic.txt` escrito por `grafo_random_bateria_generar.py`):
  1. corre `phantomsetup_cosmogenesis_backup cosmog`.
  2. reescribe el bloque de sumideros + tmax/dtmax de `cosmog.in` para que coincida EXACTO con
     `bateria_n2000/ic_real/cosmog.in` (icreate_sinks=1, rho_crit_cgs=1000, r_crit=0.6, h_acc=0.3,
     tmax=0.500, dtmax=0.001) -- la MISMA configuración física de toda la jerarquía.
  3. corre `phantom_cosmogenesis_backup cosmog.in`, mide el wall time real.

Salvaguarda de tiempo (mismo criterio que null3_bateria_correr.py): si el tiempo ACUMULADO de cómputo
Phantom supera TIEMPO_LIMITE_S, el script deja de lanzar corridas nuevas y termina limpio.

No toca `bateria_n2000/` (sólo se LEE `ic_real/cosmog.in` como plantilla de parámetros, vía el bloque
hardcodeado abajo -- igual que el resto de la jerarquía).
"""
import re
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/Users/alexis/phantom_cs073/bateria_grafo_random_n2000")
SEMILLAS = list(range(701, 709))
CARPETAS = [BASE / f"ic_random_s{s}" for s in SEMILLAS]

PHANTOMSETUP = "/Users/alexis/phantom_cs073/phantom/bin/phantomsetup_cosmogenesis_backup"
PHANTOM = "/Users/alexis/phantom_cs073/phantom/bin/phantom_cosmogenesis_backup"

TMAX = "0.500"
DTMAX = "0.001"

BLOQUE_SINKS_DEFAULT = """# options controlling sink particles
     isink_potential =           0    ! sink potential (0=1/r,1=surf)
       icreate_sinks =           0    ! allow automatic sink particle creation
     h_soft_sinksink =       0.000    ! softening length between sink particles
               f_acc =       0.800    ! particles < f_acc*h_acc accreted without checks"""

BLOQUE_SINKS_CS073 = """# options controlling sink particles
     isink_potential =           0    ! sink potential (0=1/r,1=surf)
       icreate_sinks =           1    ! allow automatic sink particle creation
        rho_crit_cgs =       1000.    ! density above which sink particles are created (g/cm^3)
              r_crit =       0.600    ! critical radius for point mass creation (no new sinks < r_crit from existing sink)
               h_acc =       0.300    ! accretion radius for new sink particles
      h_soft_sinkgas =       0.000    ! softening length for new sink particles
     h_soft_sinksink =       0.000    ! softening length between sink particles
               f_acc =       0.800    ! particles < f_acc*h_acc accreted without checks
      r_merge_uncond =       0.000    ! sinks will unconditionally merge within this separation
        r_merge_cond =       0.000    ! sinks will merge if bound within this radius"""

TIEMPO_LIMITE_S = 20 * 60


def editar_cosmog_in(ruta: Path) -> None:
    texto = ruta.read_text()
    assert BLOQUE_SINKS_DEFAULT in texto, (
        f"{ruta}: el bloque de sumideros por defecto no coincide con lo esperado -- "
        "¿cambió el binario phantomsetup_cosmogenesis_backup? no se edita a ciegas."
    )
    texto = texto.replace(BLOQUE_SINKS_DEFAULT, BLOQUE_SINKS_CS073)
    texto = re.sub(r"(?m)^(\s*tmax\s*=\s*)\S+(\s*!)", rf"\g<1>{TMAX}   \g<2>", texto)
    texto = re.sub(r"(?m)^(\s*dtmax\s*=\s*)\S+(\s*!)", rf"\g<1>{DTMAX}   \g<2>", texto)
    ruta.write_text(texto)


def correr_una(carpeta: Path) -> dict:
    ic = carpeta / "cosmogenesis_ic.txt"
    assert ic.exists(), f"falta {ic} -- correr grafo_random_bateria_generar.py primero"

    t0 = time.time()
    with open(carpeta / "setup.log", "w") as f:
        r_setup = subprocess.run([PHANTOMSETUP, "cosmog"], cwd=carpeta, stdin=subprocess.DEVNULL,
                                  stdout=f, stderr=subprocess.STDOUT)
    t_setup = time.time() - t0
    assert r_setup.returncode == 0, f"phantomsetup falló en {carpeta} (ver setup.log)"

    editar_cosmog_in(carpeta / "cosmog.in")

    t1 = time.time()
    with open(carpeta / "run.log", "w") as f:
        r_run = subprocess.run([PHANTOM, "cosmog.in"], cwd=carpeta, stdin=subprocess.DEVNULL,
                                stdout=f, stderr=subprocess.STDOUT)
    t_run = time.time() - t1

    return dict(carpeta=str(carpeta), exit_setup=r_setup.returncode, t_setup=t_setup,
                exit_run=r_run.returncode, t_run=t_run)


def main():
    t_inicio = time.time()
    resultados = []
    tiempo_phantom_acumulado = 0.0

    for carpeta in CARPETAS:
        if tiempo_phantom_acumulado > TIEMPO_LIMITE_S:
            faltan = len(CARPETAS) - len(resultados)
            print(f"\n[SALVAGUARDA DE TIEMPO] {tiempo_phantom_acumulado:.0f}s acumulados > "
                  f"{TIEMPO_LIMITE_S}s límite. Corridas hechas: {len(resultados)}/8. "
                  f"Faltan {faltan}. Deteniendo.", flush=True)
            break

        print(f"[{carpeta.name}] setup + run...", flush=True)
        info = correr_una(carpeta)
        tiempo_phantom_acumulado += info["t_setup"] + info["t_run"]
        resultados.append(info)
        print(f"[{carpeta.name}] exit_setup={info['exit_setup']} t_setup={info['t_setup']:.2f}s  "
              f"exit_run={info['exit_run']} t_run={info['t_run']:.2f}s  "
              f"(acumulado Phantom: {tiempo_phantom_acumulado:.1f}s)", flush=True)
        if info["exit_run"] != 0:
            tail = (carpeta / "run.log").read_text().splitlines()[-15:]
            print(f"  AVISO: exit_run != 0 en {carpeta.name}. Tail de run.log:", flush=True)
            print("  " + "\n  ".join(tail), flush=True)

    t_total = time.time() - t_inicio
    print(f"\n[TOTAL] {len(resultados)}/8 corridas completadas. "
          f"tiempo Phantom acumulado={tiempo_phantom_acumulado:.1f}s  "
          f"tiempo total script (incluye overhead)={t_total:.1f}s", flush=True)
    return resultados


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
