@echo off
REM Tras install_windows.ps1, el acceso real es %USERPROFILE%\ANIMA\Iniciar ANIMA.bat
powershell -NoProfile -ExecutionPolicy Bypass -File "%USERPROFILE%\ANIMA\start_organismo.ps1"
