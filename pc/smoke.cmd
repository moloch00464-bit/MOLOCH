@echo off
setlocal
set HERE=%~dp0

echo [smoke] imports
python -c "import torch, peft, transformers, fastapi, uvicorn, pydantic" || goto :err

echo [smoke] trainer self-test
python "%HERE%lora_trainer.py" --self-test || goto :err

echo [smoke] proxy self-test
python "%HERE%adapter_inference_proxy.py" --self-test || goto :err

echo [smoke] OK
endlocal
exit /b 0

:err
echo [smoke] FAILED
endlocal
exit /b 1
