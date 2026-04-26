"""MOLOCH ThreeBrain Welle 3 - Adapter Inference Proxy (PC-Side).

FastAPI on :11600. Loads Qwen2.5-1.5B-Instruct base + latest LoRA adapter
from %USERPROFILE%/moloch_adapters/v{N}/. Exposes /infer /health /list /reload.

A single threading.Lock serializes both adapter swap and generate(). The
PC side is single-consumer (Pi remote calls) so this is fine, and it
prevents PEFT mid-swap corruption when /reload fires during /infer.
The base model is kept pristine: each load creates a fresh PeftModel
from the original base, so reloads don't stack adapters.
"""
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = os.environ.get("MOLOCH_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_DIR = Path(
    os.environ.get("MOLOCH_ADAPTERS", str(Path.home() / "moloch_adapters"))
)
HOST = os.environ.get("MOLOCH_PROXY_HOST", "0.0.0.0")
PORT = int(os.environ.get("MOLOCH_PROXY_PORT", "11600"))
SYSTEM_DEFAULT = "Du bist Moloch."

_lock = threading.Lock()
state: dict = {
    "tokenizer": None,
    "base_model": None,
    "model": None,
    "adapter_version": None,
}


class InferRequest(BaseModel):
    prompt: str
    system: Optional[str] = SYSTEM_DEFAULT
    max_tokens: int = 200


class InferResponse(BaseModel):
    response: str
    adapter_version: Optional[str]
    tokens: int
    duration_ms: int


def list_versions() -> list[str]:
    if not ADAPTER_DIR.exists():
        return []
    found = []
    for d in ADAPTER_DIR.iterdir():
        if (
            d.is_dir()
            and re.match(r"v\d+$", d.name)
            and (d / "adapter_config.json").exists()
        ):
            found.append(d.name)
    return sorted(found, key=lambda v: int(v[1:]))


def latest_version() -> Optional[str]:
    vs = list_versions()
    return vs[-1] if vs else None


def load_model(version: Optional[str]) -> None:
    with _lock:
        if state["base_model"] is None:
            print(f"[proxy] loading base {BASE_MODEL}")
            tok = AutoTokenizer.from_pretrained(BASE_MODEL)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            state["tokenizer"] = tok
            state["base_model"] = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, torch_dtype=torch.float32, device_map="cpu"
            )
        if version is None:
            print("[proxy] no adapter present, serving base only")
            state["model"] = state["base_model"]
            state["adapter_version"] = None
            return

        adapter_path = ADAPTER_DIR / version
        if not (adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(f"adapter not found: {adapter_path}")

        print(f"[proxy] loading adapter {version} on pristine base")
        state["model"] = PeftModel.from_pretrained(
            state["base_model"], str(adapter_path)
        )
        state["adapter_version"] = version


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        load_model(latest_version())
    except Exception as e:
        print(f"[proxy] startup load failed: {e!r} -- /health will report 503")
    yield


app = FastAPI(title="MOLOCH Adapter Inference Proxy", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "adapter": state["adapter_version"], "base": BASE_MODEL}


@app.get("/list")
def list_endpoint():
    return {"adapters": list_versions(), "active": state["adapter_version"]}


@app.post("/reload")
def reload_endpoint():
    v = latest_version()
    if v is None:
        raise HTTPException(404, "no adapters present in MOLOCH_ADAPTERS dir")
    load_model(v)
    return {"reloaded": True, "adapter": v}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    if state["model"] is None:
        raise HTTPException(503, "model not loaded")
    if not req.prompt.strip():
        raise HTTPException(422, "prompt is empty")

    tok = state["tokenizer"]
    msgs = [
        {"role": "system", "content": req.system or SYSTEM_DEFAULT},
        {"role": "user", "content": req.prompt},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")

    with _lock:
        t0 = time.time()
        with torch.no_grad():
            out = state["model"].generate(
                **inputs,
                max_new_tokens=max(1, min(req.max_tokens, 1024)),
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tok.pad_token_id,
            )
        duration_ms = int((time.time() - t0) * 1000)
        version_at_call = state["adapter_version"]

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    response_text = tok.decode(new_tokens, skip_special_tokens=True).strip()
    return InferResponse(
        response=response_text,
        adapter_version=version_at_call,
        tokens=int(new_tokens.shape[0]),
        duration_ms=duration_ms,
    )


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
