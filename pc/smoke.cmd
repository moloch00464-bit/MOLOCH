@echo off
setlocal
set HERE=%~dp0
set VENV=%USERPROFILE%\moloch_pc_env

if exist "%VENV%\Scripts\python.exe" (
  set "PY=%VENV%\Scripts\python.exe"
) else (
  echo [smoke] venv not found at %VENV% - falling back to system python
  set "PY=python"
)

echo [smoke] imports
"%PY%" -c "import torch, peft, transformers, fastapi, uvicorn, pydantic" || goto :err

echo [smoke] trainer self-test
"%PY%" "%HERE%lora_trainer.py" --self-test || goto :err

echo [smoke] proxy self-test
"%PY%" "%HERE%adapter_inference_proxy.py" --self-test || goto :err

echo [smoke] OK
endlocal
exit /b 0

:err
echo [smoke] FAILED
endlocal
exit /b 1
