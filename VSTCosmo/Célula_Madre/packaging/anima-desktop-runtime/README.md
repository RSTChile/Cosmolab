# ANIMA Desktop Runtime

Instalador del organismo ANIMA para **PC / Mac / Linux**, sin Docker ni Rode del laboratorio.

Objetivo:

- Instalar un organismo con audio local y silencio basal válido.
- Identidad persistente `~/.anima/identity.json` con clave Ed25519.
- **UI limpia** (`ANIMA_UI_PERFIL=limpio`): sin cajas de sensores remotos
  (radio SDR, nRF24, GPS, cámara/PTZ, cloroplasto/solar).
- Lab A–D sigue con UI completa; este paquete es el perfil *consumer/instalable*.

Ver también: `docs/PLAN_instalables_multiplataforma.md`.

## Windows (PC)

Construir zip (preferido en Mac — no depende de PowerShell):

```bash
python3 packaging/anima-desktop-runtime/build_windows_zip.py
# → dist/anima-desktop-runtime_0.3.0-dev_windows.zip
```

Alternativa PowerShell: `packaging/anima-desktop-runtime/build_windows.ps1`

Instalar en el PC:

```powershell
Expand-Archive .\anima-desktop-runtime_*_windows.zip -DestinationPath $env:TEMP\anima-inst
cd $env:TEMP\anima-inst
# Opcional no-interactivo:
$env:ANIMA_NOMBRE = "Nido"
$env:ANIMA_CARA_GENERO = "femenino"   # masculino|femenino
$env:ANIMA_CARA_TONO = "celeste"      # blanco|celeste|rosado|amarillo|cafe|moreno|negro
.\install_windows.ps1
# Escritorio: "Iniciar ANIMA.bat" → http://127.0.0.1:7788/
```

**UI limpia:** las cajas de radio, GPS, cámara/PTZ, solar y nRF **no aparecen** en la página
(`ANIMA_UI_PERFIL=limpio` + `web/Cajas/manifest.limpio.json`).

Requisito: Python 3.10+ en PATH (3.9 mínimo aceptable).

## macOS

Construir desde el Mac:

```bash
cd /Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Célula_Madre
VERSION=0.1.0-dev packaging/anima-desktop-runtime/build_mac.sh
```

Salida:

```text
dist/anima-desktop-runtime_0.1.0-dev_macos.tar.gz
```

Instalar (sin sudo):

```bash
tar -xzf dist/anima-desktop-runtime_0.1.0-dev_macos.tar.gz
cd anima-desktop-runtime_0.1.0-dev_macos
./install_mac.sh
```

Durante la instalación pregunta el nombre propio del animalito.

Instalación no interactiva:

```bash
ANIMA_NOMBRE="Nido" ./install_mac.sh
```

Qué instala:

| Ruta | Contenido |
|------|-----------|
| `~/Library/Application Support/ANIMA/celula_madre` | Runtime Python |
| `~/.config/anima/` | `organismo.env`, `organelos.yml`, `hardware.yml` |
| `~/.anima/identity.json` | Identidad Ed25519 |
| `~/Library/LaunchAgents/com.vstcosmo.anima-*.plist` | Arranque al login |
| `~/.local/bin/anima` | CLI |
| `~/Desktop/Iniciar ANIMA.command` | Doble-clic → observatorio |

Comandos:

```bash
anima status
anima start
anima stop
anima restart
anima open          # abre http://127.0.0.1:7788/
anima setup --perfil limpio --nombre "Nido"
anima-config show
```

Requisitos macOS:

- Python 3.10+ (`python3` en PATH)
- PortAudio (viene con el sistema; `sounddevice` lo usa vía CoreAudio)

## Linux amd64

Construir:

```bash
cd /Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Célula_Madre
VERSION=0.1.0-dev packaging/anima-desktop-runtime/build_deb_local.sh
```

Salida:

```text
dist/anima-desktop-runtime_0.1.0-dev_amd64.deb
```

Instalar en Ubuntu/Debian:

```bash
sudo apt install ./anima-desktop-runtime_0.1.0-dev_amd64.deb
```

No interactivo:

```bash
sudo ANIMA_NOMBRE="Nido" apt install ./anima-desktop-runtime_0.1.0-dev_amd64.deb
```

Comandos iguales que macOS. Servicios vía `systemd --user`.

## Diferencias con anima-pi-runtime

| | Pi | Desktop |
|---|-----|---------|
| Arquitectura | arm64 | macOS universal / amd64 |
| Arranque | systemd --user | LaunchAgent (Mac) / systemd (Linux) |
| Config | `/etc/anima` | `~/.config/anima` (Mac) o `/etc/anima` (Linux .deb) |
| Cabeza SPI/fb1 | opcional | desactivada |
| ID por defecto | `ANIMA_*_PI` | `ANIMA_*_PC` |

## Windows

Pendiente (roadmap): zip portable + script PowerShell. La arquitectura Python es compatible; falta empaquetar PortAudio/WASAPI y autostart.

## Red lines respetadas

- `ANIMA_AUDIO_MODE=local` por defecto
- `ANIMA_OTROS_URLS` vacío
- Sin LLM en Presencia
- Separación bio/técnica en configuración