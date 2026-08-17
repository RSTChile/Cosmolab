#!/usr/bin/env python3
"""Build Windows installable zip (UI limpio: sin cajas de sensores remotos).

Output: dist/anima-desktop-runtime_<VERSION>_windows.zip
Run from any cwd:
  python3 packaging/anima-desktop-runtime/build_windows_zip.py
"""
from __future__ import annotations

import os
import shutil
import time
import zipfile
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent.parent
VER = os.environ.get("VERSION", "0.3.0-dev")
STAGE = Path(os.environ.get("ANIMA_STAGE", f"/tmp/anima_win_stage_{VER}"))
DIST = ROOT / "dist"
ZIP = DIST / f"anima-desktop-runtime_{VER}_windows.zip"

SKIP_DIR_NAMES = {"__pycache__", ".git", "node_modules", ".DS_Store"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".wav", ".mp3"}

DIRS = [
    "web",
    "organelos",
    "campo",
    "audio",
    "diada",
    "schemas",
    "lexico_comun",
    "conversacion",
    "genoma",
]


def should_skip(path: Path) -> bool:
    return path.name in SKIP_DIR_NAMES or path.suffix in SKIP_SUFFIXES


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"  skip missing {src.name}", flush=True)
        return
    print(f"  {src.name} ...", flush=True)
    t0 = time.time()
    for root, dirs, files in os.walk(src):
        root_p = Path(root)
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
        rel = root_p.relative_to(src)
        target_dir = dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            sp = root_p / f
            if should_skip(sp):
                continue
            shutil.copy2(sp, target_dir / f)
    print(f"    done {src.name} in {time.time() - t0:.2f}s", flush=True)


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    cm = STAGE / "celula_madre"
    cm.mkdir(parents=True)
    DIST.mkdir(exist_ok=True)

    print(f"ROOT={ROOT}", flush=True)
    print(f"STAGE={STAGE}", flush=True)

    for d in DIRS:
        copy_tree(ROOT / d, cm / d)

    for req in ("requirements-desktop.txt", "requirements.txt"):
        src = ROOT / req
        if src.exists():
            shutil.copy2(src, cm / req)
            print(f"  {req}", flush=True)

    shutil.copytree(PKG / "config", STAGE / "config")
    shutil.copy2(PKG / "install_windows.ps1", STAGE / "install_windows.ps1")
    bat = PKG / "Iniciar ANIMA.bat"
    if bat.exists():
        shutil.copy2(bat, STAGE / bat.name)

    (STAGE / "README.txt").write_text(
        f"ANIMA Desktop {VER} Windows\n"
        "UI limpio (sin sensores remotos: radio/GPS/camara/solar/nRF/PTZ)\n"
        "Install: .\\install_windows.ps1\n"
        "Luego: Iniciar ANIMA.bat -> http://127.0.0.1:7788/\n",
        encoding="utf-8",
    )

    man = cm / "web" / "Cajas" / "manifest.limpio.json"
    assert man.is_file(), f"missing {man}"
    weblive = (cm / "web" / "VST_CelulaMadre_WebLive_A.py").read_text(
        encoding="utf-8", errors="replace"
    )
    assert "ANIMA_UI_PERFIL" in weblive, "WebLive missing UI perfil"
    env = (STAGE / "config" / "organismo.env").read_text(encoding="utf-8")
    assert "ANIMA_UI_PERFIL=limpio" in env
    print("verify OK", flush=True)

    if ZIP.exists():
        ZIP.unlink()
    print(f"zipping -> {ZIP}", flush=True)
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, _dirs, files in os.walk(STAGE):
            for f in files:
                fp = Path(root) / f
                zf.write(fp, fp.relative_to(STAGE).as_posix())
    print(f"OK {ZIP} size={ZIP.stat().st_size}", flush=True)

    with zipfile.ZipFile(ZIP) as zf:
        names = zf.namelist()
    print(f"entries={len(names)}", flush=True)
    for key in (
        "manifest.limpio.json",
        "install_windows.ps1",
        "organismo.env",
        "VST_CelulaMadre_WebLive_A.py",
        "README.txt",
    ):
        hits = [n for n in names if n.endswith(key)]
        print(f"  {key}: {hits}", flush=True)


if __name__ == "__main__":
    main()
