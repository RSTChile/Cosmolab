# Instala ANIMA Desktop Runtime en el perfil del usuario (Windows).
# Uso: .\install_windows.ps1
#      $env:ANIMA_NOMBRE='Nido'; .\install_windows.ps1
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$HomeAnima = Join-Path $env:USERPROFILE "ANIMA"
$Cm = Join-Path $HomeAnima "celula_madre"
$Cfg = Join-Path $env:USERPROFILE ".config\anima"
$IdDir = Join-Path $env:USERPROFILE ".anima"

Write-Host "=== ANIMA Desktop Runtime (Windows) ==="
New-Item -ItemType Directory -Force -Path $HomeAnima, $Cfg, $IdDir | Out-Null

# Copiar runtime
if (Test-Path (Join-Path $Here "celula_madre")) {
  if (Test-Path $Cm) { Remove-Item -Recurse -Force $Cm }
  Copy-Item -Recurse -Force (Join-Path $Here "celula_madre") $Cm
} else {
  throw "No se encuentra celula_madre/ junto al instalador"
}

# Config
Copy-Item -Force (Join-Path $Here "config\organismo.env") (Join-Path $Cfg "organismo.env")
Copy-Item -Force (Join-Path $Here "config\organelos.yml") (Join-Path $Cfg "organelos.yml")
Copy-Item -Force (Join-Path $Here "config\hardware.yml") (Join-Path $Cfg "hardware.yml")

# Nombre
$nombre = $env:ANIMA_NOMBRE
if (-not $nombre) {
  $nombre = Read-Host "Nombre del animalito (max 14 caracteres)"
}
if (-not $nombre) { $nombre = "Animalito" }
$nombre = $nombre.Trim().Substring(0, [Math]::Min(14, $nombre.Trim().Length))

# Apariencia cara (sin sensores remotos; solo cara en UI limpia)
$genero = $env:ANIMA_CARA_GENERO
if (-not $genero) {
  $genero = Read-Host "Genero de la cara [masculino/femenino] (Enter=masculino)"
}
if (-not $genero) { $genero = "masculino" }
$genero = $genero.Trim().ToLower()
if ($genero -notin @("masculino", "femenino", "m", "f")) { $genero = "masculino" }
if ($genero -eq "m") { $genero = "masculino" }
if ($genero -eq "f") { $genero = "femenino" }

$tono = $env:ANIMA_CARA_TONO
if (-not $tono) {
  $tono = Read-Host "Tono de piel [blanco/celeste/rosado/amarillo/cafe/moreno/negro] (Enter=blanco)"
}
if (-not $tono) { $tono = "blanco" }
$tono = $tono.Trim().ToLower()

$envPath = Join-Path $Cfg "organismo.env"
$txt = Get-Content $envPath -Raw
$txt = $txt -replace 'VST_ORGANISMO_NOMBRE=.*', "VST_ORGANISMO_NOMBRE=$nombre"
$txt = $txt -replace 'VST_ORGANISMO_LABEL=.*', "VST_ORGANISMO_LABEL=$nombre"
if ($txt -notmatch 'ANIMA_UI_PERFIL=') { $txt += "`nANIMA_UI_PERFIL=limpio`n" }
if ($txt -match 'ANIMA_CARA_GENERO=') {
  $txt = $txt -replace 'ANIMA_CARA_GENERO=.*', "ANIMA_CARA_GENERO=$genero"
} else {
  $txt += "`nANIMA_CARA_GENERO=$genero`n"
}
if ($txt -match 'ANIMA_CARA_TONO=') {
  $txt = $txt -replace 'ANIMA_CARA_TONO=.*', "ANIMA_CARA_TONO=$tono"
} else {
  $txt += "ANIMA_CARA_TONO=$tono`n"
}
Set-Content -Path $envPath -Value $txt -Encoding UTF8

# venv + deps
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command py -ErrorAction SilentlyContinue).Source }
if (-not $py) { throw "Python no encontrado en PATH (instala Python 3.10+ de python.org)" }
$venv = Join-Path $HomeAnima "venv"
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
  & $py -m venv $venv
}
$pip = Join-Path $venv "Scripts\pip.exe"
$req = Join-Path $Cm "requirements-desktop.txt"
if (-not (Test-Path $req)) { $req = Join-Path $Cm "requirements.txt" }
& $pip install -q --upgrade pip
if (Test-Path $req) { & $pip install -q -r $req }

# Scripts de arranque
$startPs1 = @'
$ErrorActionPreference = "Stop"
$HomeAnima = Join-Path $env:USERPROFILE "ANIMA"
$Cm = Join-Path $HomeAnima "celula_madre"
$Cfg = Join-Path $env:USERPROFILE ".config\anima"
$venvPy = Join-Path $HomeAnima "venv\Scripts\python.exe"
# Cargar env
Get-Content (Join-Path $Cfg "organismo.env") | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
  $k,$v = $_.Split('=',2)
  if ($k -and $v -ne $null) { Set-Item -Path "Env:$k" -Value $v.Trim('"') }
}
$env:PYTHONPATH = "$Cm;$Cm\organelos;$Cm\audio;$Cm\web"
$env:PYTHONUNBUFFERED = "1"
if (-not $env:ANIMA_UI_PERFIL) { $env:ANIMA_UI_PERFIL = "limpio" }
Set-Location $Cm
& $venvPy "web\VST_CelulaMadre_WebLive_A.py"
'@
Set-Content -Path (Join-Path $HomeAnima "start_organismo.ps1") -Value $startPs1 -Encoding UTF8

$bat = @"
@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File `"$HomeAnima\start_organismo.ps1`"
"@
Set-Content -Path (Join-Path $env:USERPROFILE "Desktop\Iniciar ANIMA.bat") -Value $bat -Encoding ASCII
Set-Content -Path (Join-Path $HomeAnima "Iniciar ANIMA.bat") -Value $bat -Encoding ASCII

Write-Host ""
Write-Host "Instalado en $HomeAnima"
Write-Host "Nombre: $nombre"
Write-Host "Cara: $genero / $tono"
Write-Host "UI: limpia (sin radio/GPS/camara/solar/nRF/PTZ)"
Write-Host "Arranque: escritorio 'Iniciar ANIMA.bat' o:"
Write-Host "  powershell -File $HomeAnima\start_organismo.ps1"
Write-Host "Luego abre: http://127.0.0.1:7788/"
