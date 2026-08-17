"""
cs090_fase5b_correr.py — FASE V-B Paso 3/4: corre Phantom sobre las condiciones iniciales de
`cs090_fase5b_generar_pares.py` (grafo A2-B0-C2, masa fija, N=2000). MISMO patrón EXACTO que
`grafo_random_masa_fija_correr.py` (congelado, sólo se copia la FORMA del pipeline, no se toca ni se
importa código de él): (1) `phantomsetup_cosmogenesis_backup cosmog` sobre el ASCII ya escrito, (2)
reescribe el bloque de sumideros + tmax/dtmax de `cosmog.in` para que coincida EXACTO con
`bateria_n2000/ic_real/cosmog.in` (icreate_sinks=1, rho_crit_cgs=1000, r_crit=0.6, h_acc=0.3,
tmax=0.500, dtmax=0.001 -- MISMOS parámetros de Phantom que toda la jerarquía CS073, no se cambian sin
razón), (3) corre `phantom_cosmogenesis_backup cosmog.in`, MIDE el wall time real de cada paso.

DISCIPLINA PILOTO-PRIMERO (pedida por Alexis): `main(carpetas)` recibe una lista explícita de carpetas
a correr -- se invoca primero con SÓLO el PAR_A (2 corridas) para medir el costo real antes de decidir
si escalar a los otros 2 pares. Ver `if __name__ == "__main__"` abajo para cómo se invoca cada etapa.

No corre nada si ya existe un dump final (`cosmog_00500` o el último `cosmog_0*` con tmax alcanzado) en
la carpeta -- para poder re-invocar sin recomputar. Salvaguarda de tiempo: aborta si una corrida
individual supera `TIEMPO_LIMITE_UNA_CORRIDA_S` (20 min, mismo criterio que CS073 previo).
"""
import re
import sys
import time
from pathlib import Path

BASE = Path("/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_piloto")

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

import subprocess
TIEMPO_LIMITE_UNA_CORRIDA_S = 20 * 60


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
    assert ic.exists(), f"falta {ic} -- correr cs090_fase5b_generar_pares.py primero"

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
    resultados = []
    for carpeta in carpetas:
        print(f"[{carpeta.name}] setup + run...", flush=True)
        info = correr_una(carpeta)
        resultados.append(info)
        if info.get("ya_corrida"):
            print(f"[{carpeta.name}] ya tenía cosmog_00500 -- se salta (no se recomputa)", flush=True)
            continue
        print(f"[{carpeta.name}] exit_setup={info['exit_setup']} t_setup={info['t_setup']}s  "
              f"exit_run={info['exit_run']} t_run={info['t_run']}s", flush=True)
        if info["exit_run"] != 0:
            tail = (carpeta / "run.log").read_text().splitlines()[-20:]
            print(f"  AVISO: exit_run != 0. Tail run.log:\n  " + "\n  ".join(tail), flush=True)

    t_total = time.time() - t_inicio
    print(f"\n[TOTAL] {len(resultados)}/{len(carpetas)} corridas. tiempo script={t_total:.1f}s", flush=True)
    return resultados


if __name__ == "__main__":
    # etapa por línea de comando: "piloto" = sólo PAR_A (2 carpetas); "todo" = las 6 carpetas.
    etapa = sys.argv[1] if len(sys.argv) > 1 else "piloto"
    if etapa == "piloto":
        carpetas = sorted(BASE.glob("A2-B0-C2-r9_*")) + sorted(BASE.glob("A2-B0-C2-r19_*"))
    else:
        carpetas = sorted(BASE.iterdir())
        carpetas = [c for c in carpetas if c.is_dir()]
    print(f"Etapa={etapa}: {len(carpetas)} carpetas -> {[c.name for c in carpetas]}")
    main(carpetas)
