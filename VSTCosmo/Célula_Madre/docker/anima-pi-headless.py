#!/usr/bin/env python3
"""Cabeza 3D exacta (Three.js /cabeza) en PiScreen vía Chromium headless → fb1."""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import time
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Falta Pillow: pip3 install --user Pillow")

URL = os.environ.get("ANIMA_CABEZA_URL", "http://127.0.0.1:7788/cabeza")
INTERVAL = float(os.environ.get("ANIMA_PI_HEADLESS_S", "0.5"))
CHROMIUM = os.environ.get(
    "ANIMA_CHROMIUM",
    shutil.which("chromium") or shutil.which("chromium-browser") or "/snap/bin/chromium",
)
# Snap chromium no escribe en /tmp; usar ruta bajo el proyecto.
_FRAME = Path(__file__).resolve().parent / "headless-frame.png"


def _fb_dev() -> str:
    for dev in ("/dev/fb1", "/dev/fb0"):
        if Path(dev).exists():
            return dev
    raise SystemExit("No hay framebuffer (/dev/fb1 ni /dev/fb0)")


def _fb_info(dev: str) -> tuple[int, int, int, int]:
    name = Path(dev).name
    base = Path(f"/sys/class/graphics/{name}")
    w, h = (int(x) for x in (base / "virtual_size").read_text().strip().split(","))
    bpp = int((base / "bits_per_pixel").read_text().strip())
    stride = int((base / "stride").read_text().strip()) if (base / "stride").exists() else w * bpp // 8
    return w, h, bpp, stride


def _screenshot(w: int, h: int) -> Image.Image | None:
    if not Path(CHROMIUM).exists():
        raise SystemExit(f"Chromium no encontrado: {CHROMIUM}")
    out = str(_FRAME)
    try:
        cmd = [
            CHROMIUM,
            "--headless=new",
            "--disable-gpu",
            "--enable-unsafe-swiftshader",
            f"--screenshot={out}",
            f"--window-size={w},{h}",
            "--virtual-time-budget=8000",
            "--hide-scrollbars",
            "--no-sandbox",
            URL,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45, check=False)
        if not _FRAME.exists() or _FRAME.stat().st_size < 1000:
            return None
        return Image.open(out).convert("RGB")
    except subprocess.TimeoutExpired:
        return None


def _write_fb(img: Image.Image, dev: str, w: int, h: int, bpp: int, stride: int) -> None:
    if img.size != (w, h):
        img = img.resize((w, h), Image.Resampling.LANCZOS)
    px = img.load()
    with open(dev, "r+b") as fb:
        if bpp == 32:
            for y in range(h):
                off = y * stride
                row = bytearray()
                for x in range(w):
                    r, g, b = px[x, y]
                    row.extend((b, g, r, 255))
                fb.seek(off)
                fb.write(row)
            return
        if bpp == 16:
            for y in range(h):
                off = y * stride
                row = bytearray()
                for x in range(w):
                    r, g, b = px[x, y]
                    v = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
                    row.extend(struct.pack("<H", v))
                fb.seek(off)
                fb.write(row)
            return
        fb.seek(0)
        fb.write(img.convert("RGB").tobytes()[: w * h * (bpp // 8)])


def main() -> None:
    dev = _fb_dev()
    w, h, bpp, stride = _fb_info(dev)
    print(f"[pi-headless] {dev} {w}x{h} @{bpp}bpp · {CHROMIUM}", flush=True)
    fails = 0
    while True:
        try:
            img = _screenshot(w, h)
            if img is None:
                fails += 1
                if fails % 10 == 1:
                    print(f"[pi-headless] screenshot vacío ({fails})", flush=True)
                time.sleep(INTERVAL)
                continue
            fails = 0
            _write_fb(img, dev, w, h, bpp, stride)
        except Exception as exc:
            print(f"[pi-headless] error: {exc}", flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()