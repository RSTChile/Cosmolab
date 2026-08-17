# Ejecutar en PowerShell del PC (usuario Alexis, admin preferible)
$ErrorActionPreference = 'Stop'
$pub = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBrRWtb4E6pEX2HNE6yjAUncW4zhWw8EnDBz0wdtKv4I anima-mac-to-pc38'
$sshDir = Join-Path $env:USERPROFILE '.ssh'
$auth = Join-Path $sshDir 'authorized_keys'
New-Item -ItemType Directory -Force -Path $sshDir | Out-Null
if (-not (Test-Path $auth)) { New-Item -ItemType File -Path $auth | Out-Null }
$cur = Get-Content $auth -ErrorAction SilentlyContinue
if ($cur -notcontains $pub) { Add-Content -Path $auth -Value $pub }
# ACL: solo el usuario (OpenSSH en Windows es estricto)
icacls $sshDir /inheritance:r
icacls $sshDir /grant "${env:USERNAME}:(OI)(CI)F"
icacls $auth /inheritance:r
icacls $auth /grant "${env:USERNAME}:F"
# Administrators + SYSTEM a veces requeridos
icacls $auth /remove "NT AUTHORITY\Authenticated Users" 2>$null
Write-Host "authorized_keys listo:"
Get-Content $auth
Get-Service sshd | Format-List Status,Name
Write-Host "Prueba desde Mac: ssh -i ~/.ssh/id_anima_pc38 Alexis@192.168.86.38"
