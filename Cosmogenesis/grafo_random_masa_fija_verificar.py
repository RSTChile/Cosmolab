"""
grafo_random_masa_fija_verificar.py — Paso 2 del encargo: verificar, leyendo el volcado BINARIO real de
Phantom (no el ASCII que nosotros escribimos), que la masa total del sistema generado por
`grafo_random_layout_generar_ic_masa_fija.py` es ~18800 para varios N, y que YA NO escala con N (a
diferencia del generador original, ver `MISTERIO_N500_vs_N2000_CS.md`).

Usa `leer_volcado_phantom.py` (script congelado, sólo se importa/lee) sobre el dump inicial
`cosmog_00000` de cada carpeta -- el volcado que escribe `phantomsetup_cosmogenesis_backup` a partir de
nuestro ASCII, ya en el formato binario propio de Phantom. A t=0 no hay sumideros todavía, así que la
masa total del sistema es sólo `sum(gas['m'] o npart*massoftype)`.

También, si existe el dump final (`cosmog_00500`) de cada carpeta, reporta cuántos sumideros hay ahí y
su masa -- esto alimenta también el Paso 3 (barrido de formación de sumideros), reusando la misma
lectura.
"""
from pathlib import Path

from leer_volcado_phantom import leer_dump

BASE = Path("/Users/alexis/phantom_cs073/bateria_grafo_random_masa_fija")
VALORES_N = (200, 500, 1000, 2000)
SEMILLAS = (1, 2)


def masa_total_dump(gas, sinks):
    m_gas = float(gas["m"].sum()) if "m" in gas.columns else float(gas.params.get("massoftype", 0.0)) * len(gas)
    m_sinks = float(sinks["m"].sum()) if sinks is not None else 0.0
    return m_gas, m_sinks


def main():
    print("=== Paso 2: masa total leída del volcado BINARIO (cosmog_00000, t=0) ===\n")
    filas_inicial = []
    for n in VALORES_N:
        for seed in SEMILLAS:
            carpeta = BASE / f"ic_masaFija_N{n}_s{seed}"
            dump0 = carpeta / "cosmog_00000"
            if not dump0.exists():
                print(f"[N={n} s{seed}] falta {dump0} -- ¿corriste grafo_random_masa_fija_correr.py?")
                continue
            gas, sinks = leer_dump(dump0)
            m_gas, m_sinks = masa_total_dump(gas, sinks)
            n_gas = len(gas)
            print(f"[N={n} s{seed}] t=0: n_gas={n_gas} m_gas={m_gas:.6g} m_sinks={m_sinks:.6g} "
                  f"masa_total={m_gas + m_sinks:.6g}")
            filas_inicial.append(dict(n=n, seed=seed, n_gas=n_gas, masa_total=m_gas + m_sinks))

    print("\n=== Resumen Paso 2: ¿la masa total escala con N o queda fija? ===")
    for n in VALORES_N:
        vals = [f["masa_total"] for f in filas_inicial if f["n"] == n]
        if vals:
            print(f"  N={n}: masa_total={vals}")

    print("\n=== Paso 3 (si existe dump final cosmog_00500): sumideros formados ===\n")
    filas_final = []
    for n in VALORES_N:
        for seed in SEMILLAS:
            carpeta = BASE / f"ic_masaFija_N{n}_s{seed}"
            dump_final = carpeta / "cosmog_00500"
            if not dump_final.exists():
                candidatos = sorted(carpeta.glob("cosmog_0*"))
                dump_final = candidatos[-1] if candidatos else None
            if dump_final is None or not dump_final.exists():
                print(f"[N={n} s{seed}] sin dump final todavía")
                continue
            gas, sinks = leer_dump(dump_final)
            n_sinks = 0 if sinks is None else len(sinks)
            m_sinks = 0.0 if sinks is None else float(sinks["m"].sum())
            print(f"[N={n} s{seed}] dump final={dump_final.name}: n_gas={len(gas)} "
                  f"n_sumideros={n_sinks} masa_sumideros={m_sinks:.6g}")
            filas_final.append(dict(n=n, seed=seed, dump=dump_final.name, n_sinks=n_sinks,
                                     m_sinks=m_sinks))

    print("\n=== Resumen Paso 3: sumideros por N (masa fija) ===")
    for n in VALORES_N:
        rs = [f for f in filas_final if f["n"] == n]
        if rs:
            resumen = ", ".join(f"s{f['seed']}: {f['n_sinks']} sumideros (masa {f['m_sinks']:.1f})"
                                 for f in rs)
            print(f"  N={n}: {resumen}")

    return filas_inicial, filas_final


if __name__ == "__main__":
    main()
