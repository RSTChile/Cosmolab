# Build zip instalable ANIMA para Windows (PC).
# Ejecutar desde PowerShell en el Mac (con bash/rsync) o en Windows con Git Bash.
# Salida: dist/anima-desktop-runtime_<VERSION>_windows.zip
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Pkg = Join-Path $Root "packaging\anima-desktop-runtime"
$Version = if ($env:VERSION) { $env:VERSION } else { "0.3.0-dev" }
$Dist = Join-Path $Root "dist"
$Stage = Join-Path $Root "build\anima-desktop-runtime_${Version}_windows"
New-Item -ItemType Directory -Force -Path $Dist | Out-Null
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null

$Cm = Join-Path $Stage "celula_madre"
New-Item -ItemType Directory -Force -Path $Cm | Out-Null

# Núcleo (sin arduino/hardware pesado ni voces gigantes)
$dirs = @("web", "organelos", "genoma", "campo", "audio", "diada", "schemas", "lexico_comun", "conversacion")
foreach ($d in $dirs) {
  $src = Join-Path $Root $d
  if (Test-Path $src) {
    Copy-Item -Recurse -Force $src (Join-Path $Cm $d)
  }
}
Copy-Item -Force (Join-Path $Root "requirements-desktop.txt") (Join-Path $Cm "requirements-desktop.txt") -ErrorAction SilentlyContinue
Copy-Item -Force (Join-Path $Root "requirements.txt") (Join-Path $Cm "requirements.txt") -ErrorAction SilentlyContinue

# Config limpio + scripts Windows
Copy-Item -Recurse -Force (Join-Path $Pkg "config") (Join-Path $Stage "config")
Copy-Item -Force (Join-Path $Pkg "install_windows.ps1") (Join-Path $Stage "install_windows.ps1")
Copy-Item -Force (Join-Path $Pkg "Iniciar ANIMA.bat") (Join-Path $Stage "Iniciar ANIMA.bat") -ErrorAction SilentlyContinue
@"
ANIMA Desktop Runtime $Version (Windows)
Instalar: PowerShell como usuario → .\install_windows.ps1
Opcional: `$env:ANIMA_NOMBRE='Nido'; .\install_windows.ps1
UI: ANIMA_UI_PERFIL=limpio (sin radio/GPS/cámara/solar en cajas)
"@ | Set-Content -Path (Join-Path $Stage "README.txt") -Encoding UTF8

$Zip = Join-Path $Dist "anima-desktop-runtime_${Version}_windows.zip"
if (Test-Path $Zip) { Remove-Item $Zip -Force }
Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $Zip -Force
Write-Host "OK $Zip"
