#!/usr/bin/env bash
# Lanzador escritorio: arregla SDRplay RSP1 (pide sudo en esta terminal).
set -euo pipefail

SCRIPT="/home/ubuntu/anima/celula_madre/docker/fix-sdr-hardware.sh"

echo "════════════════════════════════════════"
echo "  Arreglar SDRplay RSP1 en la Pi"
echo "════════════════════════════════════════"
echo ""
echo "Se pedirá la contraseña de ubuntu (sudo)."
echo ""

sudo bash "$SCRIPT"
echo ""
read -r -p "Listo. Pulsa Enter para cerrar…"