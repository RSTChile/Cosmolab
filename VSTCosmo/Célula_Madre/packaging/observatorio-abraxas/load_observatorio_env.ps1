# Carga variables KEY=VALUE desde observatorio.env (sin ejecutar el archivo).
param(
    [string]$EnvFile = (Join-Path $PSScriptRoot "config\observatorio.env")
)

if (-not (Test-Path $EnvFile)) {
    Write-Warning "No se encuentra $EnvFile"
    return
}

Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#")) { return }
    $i = $line.IndexOf("=")
    if ($i -lt 1) { return }
    $key = $line.Substring(0, $i).Trim()
    $val = $line.Substring($i + 1).Trim()
    if ($key) { Set-Item -Path "Env:$key" -Value $val }
}