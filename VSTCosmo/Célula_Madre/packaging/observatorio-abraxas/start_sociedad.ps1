# Arranca vst_sociedad (:9101) en Abraxas — observatorio.cosmosemiotica.cl
# Uso: .\packaging\observatorio-abraxas\start_sociedad.ps1
#      (desde la raíz Célula_Madre / C:\Users\adale\ANIMA\celula_madre)
$ErrorActionPreference = "Stop"

$PackDir = $PSScriptRoot
$CmRoot = (Resolve-Path (Join-Path $PackDir "..\..")).Path
Set-Location $CmRoot

. (Join-Path $PackDir "load_observatorio_env.ps1")

$py = $null
foreach ($c in @(
    (Join-Path $CmRoot ".venv\Scripts\python.exe"),
    (Join-Path $CmRoot "venv\Scripts\python.exe"),
    "python",
    "py"
)) {
    if ($c -match '\\') {
        if (Test-Path $c) { $py = $c; break }
    } else {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if ($cmd) { $py = $cmd.Source; break }
    }
}
if (-not $py) { throw "Python no encontrado. Instala Python 3.10+ o crea .venv en celula_madre." }

$port = if ($env:ANIMA_SOCIEDAD_PORT) { $env:ANIMA_SOCIEDAD_PORT } else { "9101" }
Write-Host "=== Observatorio ANIMA (Abraxas) ==="
Write-Host "Raíz: $CmRoot"
Write-Host "Puerto: $port"
Write-Host "Semillas: $($env:ANIMA_SEED_URLS)"
Write-Host "Salud: http://127.0.0.1:${port}/salud"
Write-Host ""

& $py (Join-Path $CmRoot "conversacion\vst_sociedad.py")