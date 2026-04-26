@echo off
setlocal
set ROOT=%USERPROFILE%
set VENV=%ROOT%\moloch_pc_env
set HERE=%~dp0

if not exist "%VENV%" (
  echo [setup] creating venv at %VENV%
  python -m venv "%VENV%" || goto :err
)

call "%VENV%\Scripts\activate"
python -m pip install --upgrade pip || goto :err
pip install -r "%HERE%requirements.txt" || goto :err
pip install torch --index-url https://download.pytorch.org/whl/cpu || goto :err

if not exist "%ROOT%\moloch_samples"  mkdir "%ROOT%\moloch_samples"
if not exist "%ROOT%\moloch_adapters" mkdir "%ROOT%\moloch_adapters"

echo.
echo [setup] done.
echo   venv:     %VENV%
echo   samples:  %ROOT%\moloch_samples
echo   adapters: %ROOT%\moloch_adapters
echo.
echo Activate with:  call "%VENV%\Scripts\activate"
endlocal
exit /b 0

:err
echo [setup] FAILED
endlocal
exit /b 1
