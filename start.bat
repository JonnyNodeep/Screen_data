@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo Starting DSData MVP...
set "FLAGS_use_mkldnn=0"
set "OMP_NUM_THREADS=1"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True"
call poetry run python main.py

echo.
echo Done.
pause

