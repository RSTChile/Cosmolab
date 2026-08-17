# Plan — Organismos instalables (PC → Mac → Android → iPhone)

**Fecha:** 2026-07-08  
**Principio de madurez:** A–D en Mac = laboratorio. Solo tras validar se promueve a instalables / Pis / campo.

## 1. Perfiles de UI

| Perfil | `ANIMA_UI_PERFIL` | Cajas |
|--------|-------------------|--------|
| **Lab / Pi hardware** | `completo` (default en Docker lab) | Todas, incl. sensores remotos |
| **Instalable consumer** | `limpio` | Sin radio, GPS, cámara, solar, nRF, PTZ |

### Cajas excluidas en `limpio`

- `localizacion` (GPS)
- `cloroplasto_fisico` (panel solar)
- `vision`, `ptz` (cámara)
- `radio_sdr`, `radio_voz`, `nrf24` (radios)

Archivo: `web/Cajas/manifest.limpio.json`  
Servido por `WebLive_A` cuando `ANIMA_UI_PERFIL=limpio`.

## 2. Roadmap de plataformas

| Plataforma | Empaque | Estado |
|------------|---------|--------|
| **Windows PC** | zip + `install_windows.ps1` | **Listo para probar** (`dist/…_windows.zip`, UI limpio) |
| **macOS** | tar.gz + `install_mac.sh` | Existe (desktop-runtime) |
| **Linux amd64** | `.deb` | Existe |
| **Raspberry Pi** | `.deb` anima-pi-runtime | Existe (perfil hardware) |
| **Android** | APK / PWA + runtime | Pendiente |
| **iPhone** | PWA (sin sensores nativos) / TestFlight más adelante | Pendiente |

## 3. Windows PC (primer entregable)

```powershell
# En Mac o PC con el árbol del repo:
cd Célula_Madre
# Construir zip (PowerShell):
powershell -File packaging/anima-desktop-runtime/build_windows.ps1

# En el PC destino:
Expand-Archive dist/anima-desktop-runtime_*_windows.zip -DestinationPath $env:TEMP\anima-install
cd $env:TEMP\anima-install
$env:ANIMA_NOMBRE = "Nido"
.\install_windows.ps1
# Escritorio: Iniciar ANIMA.bat → http://127.0.0.1:7788/
```

Defaults:

- Audio local / silencio basal  
- Presencia local (mDNS/UDP)  
- Sin Rode, sin SDR, sin PTZ  
- UI limpia  

## 4. Relación con el lab

```text
Cambios de código/teoría
    → probar en A–D (Docker Mac)
    → si adoptado: bump packaging + rebuild instalables
    → actualizar Pis / PC campo solo entonces
```

El **observatorio público** (PC `.38` + Cloudflare) solo **lee** organismos; no es el canal de upgrades primarios.

## 5. Siguientes pasos técnicos

1. Probar zip Windows en Abraxas (`.38`) o PC limpio  
2. ~~Añadir pregunta género/tono en `install_windows.ps1`~~ (hecho)  
3. Mac: alinear `ANIMA_UI_PERFIL=limpio` en install_mac  
4. Android/iOS: shell PWA que abra el mismo WebLive (o embebido) con perfil limpio  
 
