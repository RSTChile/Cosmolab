# Arranca observatorio (:9101) + túnel Cloudflare en ventanas separadas (Abraxas).
# Tras reinicio del PC: ejecutar este script (o programar en Inicio).
$ErrorActionPreference = "Stop"

$PackDir = $PSScriptRoot
$Sociedad = Join-Path $PackDir "start_sociedad.ps1"
$Tunnel = Join-Path $PackDir "start_tunnel.ps1"

if (-not (Test-Path $Sociedad)) { throw "Falta $Sociedad" }
if (-not (Test-Path $Tunnel)) { throw "Falta $Tunnel" }

Write-Host "=== Abraxas: observatorio + túnel ==="
Write-Host "1) Sociedad en nueva ventana"
Write-Host "2) Cloudflared en nueva ventana"
Write-Host ""

Start-Process powershell -ArgumentList @(
    "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$Sociedad`""
)
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList @(
    "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$Tunnel`""
)

$port = if ($env:ANIMA_SOCIEDAD_PORT) { $env:ANIMA_SOCIEDAD_PORT } else { "9101" }
Write-Host "Espera ~5s y prueba:"
Write-Host "  http://127.0.0.1:${port}/salud"
Write-Host "  https://observatorio.cosmosemiotica.cl/"