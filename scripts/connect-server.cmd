@echo off
REM Обёртка для connect-server.ps1 (удобно из cmd или проводника).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0connect-server.ps1" %*
