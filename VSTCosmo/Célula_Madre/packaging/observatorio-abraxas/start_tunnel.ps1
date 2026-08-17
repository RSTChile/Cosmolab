# Túnel Cloudflare → observatorio.cosmosemiotica.cl
# Requiere cloudflared instalado y credenciales en C:\Users\adale\ANIMA\cloudflared\
param(
    [string]$TunnelName = "anima-observatorio",
    [string]$CloudflaredDir = (Join-Path $env:USERPROFILE "ANIMA\cloudflared")
)

$ErrorActionPreference = "Stop"

$cf = Get-Command cloudflared -ErrorAction SilentlyContinue
if (-not $cf) {
    $local = Join-Path $CloudflaredDir "cloudflared.exe"
    if (Test-Path $local) { $cf = Get-Item $local }
}
if (-not $cf) { throw "cloudflared no encontrado. Instálalo o colócalo en $CloudflaredDir" }

$config = Join-Path $CloudflaredDir "config.yml"
if (-not (Test-Path $config)) {
    $config = Join-Path $env:USERPROFILE ".cloudflared\config.yml"
}
if (-not (Test-Path $config)) {
    throw "No se encuentra config.yml del túnel (buscado en $CloudflaredDir y ~/.cloudflared)"
}

Write-Host "=== Cloudflare tunnel: $TunnelName ==="
Write-Host "Config: $config"
Write-Host "Destino público: https://observatorio.cosmosemiotica.cl/"
Write-Host ""

if ($cf -is [System.IO.FileInfo] -or $cf.Source -match '\.exe$') {
    $exe = if ($cf -is [System.IO.FileInfo]) { $cf.FullName } else { $cf.Source }
    & $exe tunnel --config $config run $TunnelName
} else {
    & cloudflared tunnel --config $config run $TunnelName
}