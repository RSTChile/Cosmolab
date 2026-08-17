"""
cs090_fase6_o3b_correr.py — FASE VI O3-B: corre Phantom sobre los pares (original, gemelo reconfigurado)
que fabricó `cs090_fase6_o3b_rewiring.py`.

MISMO patrón EXACTO que `cs090_fase5b_correr.py` (que a su vez copia el de `grafo_random_masa_fija_correr.py`):
se copia la FORMA del pipeline, no se importa ni se toca ninguno de los dos. Tres pasos por carpeta:

  1. `phantomsetup_cosmogenesis_backup cosmog` sobre el ASCII ya escrito,
  2. se reescribe el bloque de sumideros + tmax/dtmax de `cosmog.in` para que quede EXACTO al de toda
     la jerarquía CS073 / Fase V-B: icreate_sinks=1, rho_crit_cgs=1000, r_crit=0.6, h_acc=0.3,
     tmax=0.500, dtmax=0.001 — ni un parámetro distinto, porque el punto de esta tarea es que original y
     gemelo pasen por el MISMO tubo,
  3. `phantom_cosmogenesis_backup cosmog.in`, midiendo el wall time real.

No recomputa una carpeta que ya tenga `cosmog_00500`. Aborta una corrida individual que pase de 20 min
(mismo criterio de salvaguarda que CS073 y Fase V-B). No declara cierre ni veredicto.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/Users/alexis/phantom_cs073/bateria_fase6_o3b_rewiring")

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

TIEMPO_LIMITE_UNA_CORRIDA_S = 20 * 60


def editar_cosmog_in(ruta: Path) -> None:
    texto = ruta.read_text()
    assert BLOQUE_SINKS_DEFAULT in texto, (
        f"{ruta}: el bloque de sumideros por defecto no coincide con lo esperado -- "
        "¿cambió el binario phantomsetup_cosmogenesis_backup? no se edita a ciegas.")
    texto = texto.replace(BLOQUE_SINKS_DEFAULT, BLOQUE_SINKS_CS073)
    texto = re.sub(r"(?m)^(\s*tmax\s*=\s*)\S+(\s*!)", rf"\g<1>{TMAX}   \g<2>", texto)
    texto = re.sub(r"(?m)^(\s*dtmax\s*=\s*)\S+(\s*!)", rf"\g<1>{DTMAX}   \g<2>", texto)
    ruta.write_text(texto)


def correr_una(carpeta: Path) -> dict:
    ic = carpeta / "cosmogenesis_ic.txt"
    assert ic.exists(), f"falta {ic} -- correr cs090_fase6_o3b_rewiring.py primero"
    if (carpeta / "cosmog_00500").exists():
        return dict(carpeta=str(carpeta), exit_setup=0, t_setup=0.0, exit_run=0, t_run=0.0,
                    ya_corrida=True)

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
                               stdout=f, stderr=subprocess.STDOUT, timeout=TIEMPO_LIMITE_UNA_CORRIDA_S)
    t_run = time.time() - t1

    return dict(carpeta=str(carpeta), exit_setup=r_setup.returncode, t_setup=round(t_setup, 2),
                exit_run=r_run.returncode, t_run=round(t_run, 2))


def main(carpetas):
    t_inicio = time.time()
    for carpeta in carpetas:
        print(f"[{carpeta.name}] setup + run...", flush=True)
        info = correr_una(carpeta)
        if info.get("ya_corrida"):
            print(f"[{carpeta.name}] ya tenía cosmog_00500 -- se salta", flush=True)
            continue
        print(f"[{carpeta.name}] exit_setup={info['exit_setup']} t_setup={info['t_setup']}s  "
              f"exit_run={info['exit_run']} t_run={info['t_run']}s", flush=True)
        if info["exit_run"] != 0:
            tail = (carpeta / "run.log").read_text().splitlines()[-20:]
            print("  AVISO: exit_run != 0. Tail run.log:\n  " + "\n  ".join(tail), flush=True)
    print(f"\n[TOTAL] tiempo script={time.time()-t_inicio:.1f}s", flush=True)


if __name__ == "__main__":
    # uso: python3.9 cs090_fase6_o3b_correr.py [patron_glob]   (default: todas las carpetas)
    patron = sys.argv[1] if len(sys.argv) > 1 else "*"
    # se saltan las carpetas cuya condición inicial todavía no está escrita (el generador crea la
    # carpeta antes de terminar de escribir el IC; así este script se puede re-invocar mientras corre)
    carpetas = sorted(c for c in BASE.glob(patron)
                      if c.is_dir() and (c / "cosmogenesis_ic.txt").exists())
    print(f"patron={patron}: {len(carpetas)} carpetas -> {[c.name for c in carpetas]}")
    main(carpetas)
