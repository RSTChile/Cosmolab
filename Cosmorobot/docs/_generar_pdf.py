#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera el PDF de la bitácora a partir del HTML. Script de uso único,
no forma parte del robot en sí."""
from fpdf import FPDF
from pathlib import Path

AQUI = Path(__file__).resolve().parent
HTML = (AQUI / "bitacora_sesion_2026-07-10_11.html").read_text(encoding="utf-8")
SALIDA = AQUI / "CosmoRobot_Bitacora_2026-07-10_11.pdf"

FONTS = "C:/WINDOWS/Fonts"


class Bitacora(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Arial", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "CosmoRobot - Bitacora tecnica 2026-07-10/11", align="L")
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Pagina {self.page_no()}", align="C")


pdf = Bitacora(format="A4")
pdf.set_auto_page_break(auto=True, margin=20)
pdf.set_margins(20, 18, 20)
pdf.add_font("Arial", "", f"{FONTS}/arial.ttf")
pdf.add_font("Arial", "B", f"{FONTS}/arialbd.ttf")
pdf.add_font("Arial", "I", f"{FONTS}/ariali.ttf")
pdf.add_font("Arial", "BI", f"{FONTS}/arialbi.ttf")
pdf.set_font("Arial", size=11)

pdf.add_page()
pdf.write_html(HTML)

pdf.output(str(SALIDA))
print(f"Generado: {SALIDA} ({SALIDA.stat().st_size} bytes, {pdf.page_no()} paginas)")
