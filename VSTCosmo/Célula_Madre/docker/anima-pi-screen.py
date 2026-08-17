#!/usr/bin/env python3
"""Solo la cabeza del organismo ANIMA — pantalla pequeña PiScreen (fb1)."""
from __future__ import annotations

import json
import math
import os
import struct
import time
import urllib.request
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    raise SystemExit("Falta Pillow: pip3 install --user Pillow")

URL = os.environ.get("ANIMA_ESTADO_URL", "http://127.0.0.1:7788/estado")
INTERVAL = float(os.environ.get("ANIMA_PI_SCREEN_S", "0.12"))


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


def _fetch() -> dict:
    with urllib.request.urlopen(URL, timeout=5) as r:
        return json.load(r)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _draw_background(draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
    for i in range(h):
        t = i / max(h - 1, 1)
        r = int(_lerp(23, 8, t))
        g = int(_lerp(34, 11, t))
        b = int(_lerp(49, 16, t))
        draw.line([(0, i), (w, i)], fill=(r, g, b))
    # suelo tipo observatorio
    gy = int(h * 0.72)
    step = max(14, w // 24)
    for x in range(-w, w * 2, step):
        draw.line([(x, gy), (x + w // 3, h)], fill=(40, 58, 72), width=1)


def _ear(draw: ImageDraw.ImageDraw, cx: int, cy: int, rx: int, ry: int, inner: tuple, outer: tuple, glow: float) -> None:
    bbox = (cx - rx, cy - ry, cx + rx, cy + ry)
    draw.ellipse(bbox, fill=outer, outline=None)
    draw.ellipse((cx - rx + 8, cy - ry + 10, cx + rx - 8, cy + ry - 10), fill=inner)
    if glow > 0.05:
        gr = int(rx + 6 + glow * 10)
        gb = (bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4)
        ring = (
            int(_lerp(outer[0], 255, glow * 0.35)),
            int(_lerp(outer[1], 255, glow * 0.35)),
            int(_lerp(outer[2], 255, glow * 0.35)),
        )
        draw.ellipse(gb, outline=ring, width=max(2, int(2 + glow * 3)))


def _eye(draw: ImageDraw.ImageDraw, cx: int, cy: int, scale: float) -> None:
    r = max(5, int(10 * scale))
    draw.ellipse((cx - r, cy - r, cx + r, cy + int(r * 1.15)), fill=(243, 246, 251))
    ir = max(3, int(5 * scale))
    draw.ellipse((cx - ir, cy - ir + 1, cx + ir, cy + ir + 1), fill=(6, 63, 120))
    draw.ellipse((cx - 2, cy - 3, cx, cy - 1), fill=(255, 255, 255))


def _mouth(draw: ImageDraw.ImageDraw, cx: int, cy: int, cara: float, scale: float) -> None:
    w = int(22 * scale)
    s = 1 if cara > 0.2 else (-1 if cara < -0.2 else 0)
    col = (58, 54, 51)
    if s == 0:
        draw.line([(cx - w, cy), (cx + w, cy)], fill=col, width=max(2, int(2 * scale)))
    else:
        dy = int(7 * scale) * s
        draw.arc((cx - w, cy - w // 2 - dy, cx + w, cy + w // 2 - dy), 200, 340, fill=col, width=max(2, int(2 * scale)))


def _head_sphere(draw: ImageDraw.ImageDraw, cx: int, cy: int, rx: int, ry: int) -> None:
    layers = [
        (1.00, (52, 56, 64)),
        (0.92, (102, 104, 108)),
        (0.82, (190, 174, 166)),
        (0.70, (233, 229, 222)),
        (0.55, (246, 240, 235)),
    ]
    for f, col in layers:
        ex, ey = int(rx * f), int(ry * f)
        draw.ellipse((cx - ex, cy - ey, cx + ex, cy + ey), fill=col)
    # brillo
    hx, hy = cx - rx // 3, cy - ry // 3
    draw.ellipse((hx - rx // 4, hy - ry // 4, hx + rx // 5, hy + ry // 5), fill=(255, 252, 248))


def _render_head(estado: dict, w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    _draw_background(draw, w, h)

    theta = math.radians(float(estado.get("orientacion_deg") or 0.0))
    e_l = min(1.0, float(estado.get("energia_L") or 0.0) * 4.0)
    e_r = min(1.0, float(estado.get("energia_R") or 0.0) * 4.0)
    cara = float(estado.get("cara_valoracion") or 0.0)
    vivo = bool(estado.get("vivo"))

    cx, cy = w // 2, int(h * 0.46)
    base = min(w, h)
    r = int(base * 0.22)
    depth = max(0.35, math.cos(theta))
    rx = max(8, int(r * depth))
    ry = r
    scale = base / 480.0

    # orejas (detrás) — giran con la cabeza
    ear_off = int(r * 0.72)
    ear_rx, ear_ry = int(r * 0.28), int(r * 0.42)
    sin_t = math.sin(theta)
    l_vis = 0.55 + 0.45 * max(0.0, -sin_t)
    r_vis = 0.55 + 0.45 * max(0.0, sin_t)

    if l_vis > 0.2:
        _ear(
            draw,
            cx - int(ear_off * depth) - ear_rx // 2,
            cy + int(sin_t * 6),
            int(ear_rx * l_vis),
            int(ear_ry * l_vis),
            (115, 231, 242),
            (13, 82, 107),
            e_l if vivo else 0.0,
        )
    if r_vis > 0.2:
        _ear(
            draw,
            cx + int(ear_off * depth) + ear_rx // 2,
            cy + int(sin_t * 6),
            int(ear_rx * r_vis),
            int(ear_ry * r_vis),
            (255, 154, 160),
            (130, 49, 57),
            e_r if vivo else 0.0,
        )

    _head_sphere(draw, cx, cy, rx, ry)

    # ojos — se desplazan con el giro
    eye_sep = int(r * 0.34 * depth)
    eye_y = cy - int(r * 0.06)
    shift = int(math.sin(theta) * r * 0.18)
    _eye(draw, cx - eye_sep + shift, eye_y, scale * depth)
    _eye(draw, cx + eye_sep + shift, eye_y, scale * depth)

    _mouth(draw, cx + shift, cy + int(r * 0.38), cara, scale)

    if not vivo:
        draw.rectangle((0, 0, w - 1, h - 1), outline=(180, 60, 60), width=3)

    return img


def _write_fb(img: Image.Image, dev: str, w: int, h: int, bpp: int, stride: int) -> None:
    px = img.convert("RGB").load()
    with open(dev, "r+b") as fb:
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
        if bpp == 32:
            for y in range(h):
                off = y * stride
                row = bytearray()
                for x in range(w):
                    r, g, b = px[x, y]
                    row.extend((b, g, r, 255))  # BGRA — formato típico en Pi fb
                fb.seek(off)
                fb.write(row)
            return
        data = img.convert("RGB").tobytes()
        fb.seek(0)
        fb.write(data[: w * h * (bpp // 8)])


def main() -> None:
    dev = _fb_dev()
    w, h, bpp, stride = _fb_info(dev)
    print(f"[pi-cabeza] {dev} {w}x{h} @{bpp}bpp stride={stride}", flush=True)
    while True:
        try:
            estado = _fetch()
            img = _render_head(estado, w, h)
            if img.size != (w, h):
                img = img.resize((w, h))
            _write_fb(img, dev, w, h, bpp, stride)
        except Exception as exc:
            print(f"[pi-cabeza] error: {exc}", flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()