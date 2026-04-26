"""MOLOCH ThreeBrain Welle 3 - LoRA-Trainer (PC-Side).

Trains a LoRA adapter on Qwen2.5-1.5B-Instruct from approved Pi samples.
CPU-only by default (GTX 760 = Kepler, 32 GB RAM is plenty).
Threads capped to ~40 percent of a 24-thread Ryzen per Markus' rule.
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

CPU_THREADS = int(os.environ.get("MOLOCH_TRAIN_THREADS", "10"))
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))

if os.name == "nt":
    try:
        import ctypes
        BELOW_NORMAL_PRIORITY_CLASS = 0x4000
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS
        )
    except Exception:
        pass

import torch  # noqa: E402

torch.set_num_threads(CPU_THREADS)

from datasets import Dataset  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SAMPLES = 100
BATCH_SIZE = 2
EPOCHS = 3
LR = 2e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MAX_LEN = 512
KEEP_LAST = 5
SYSTEM_PROMPT = "Du bist Moloch."

# Per-sample weighting: critic-samples lehren "schlecht -> besser", thumbs_up
# verstaerken nur die bestehende Pi-Antwort. Bei kleinem oder gemischtem Pool
# sollen critic-samples den Lerneffekt dominieren.
WEIGHT_CRITIC = 3
WEIGHT_THUMBS_UP = 1


def load_samples(path: Path) -> list[dict]:
    """Parse approved samples from jsonl. No weighting, no cap."""
    pairs: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("approved"):
                continue
            situation = (row.get("situation") or "").strip()
            if not situation:
                continue
            source = row.get("source", "")
            if source == "critic":
                target = (row.get("better_response") or "").strip()
            elif source == "thumbs_up":
                target = (row.get("pi_response") or "").strip()
            else:
                continue
            if not target:
                continue
            pairs.append({"input": situation, "output": target, "source": source})
    return pairs


def apply_weighting_and_cap(pairs: list[dict]) -> list[dict]:
    """Multiplicate samples per WEIGHT_*, then truncate to MAX_SAMPLES.

    Critic-samples get repeated WEIGHT_CRITIC times (default 3), thumbs_up
    WEIGHT_THUMBS_UP times (default 1). The Trainer sees more critic-driven
    gradient updates per epoch, so the LoRA learns the corrective style
    instead of just reinforcing pi_response.
    """
    weighted: list[dict] = []
    for p in pairs:
        w = WEIGHT_CRITIC if p["source"] == "critic" else WEIGHT_THUMBS_UP
        weighted.extend([p] * w)
    return weighted[:MAX_SAMPLES]


def encode_pair(tokenizer, situation: str, response: str) -> dict:
    """Tokenize one (situation, response) pair with prompt tokens masked to -100.

    Loss is computed only on the assistant response, not on system+user prompt
    or padding. This is what makes the LoRA actually learn the target style
    rather than echoing the input.
    """
    prompt_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": situation},
    ]
    full_msgs = prompt_msgs + [{"role": "assistant", "content": response}]

    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(
        full_text, add_special_tokens=False, truncation=True, max_length=MAX_LEN
    )["input_ids"]

    labels = list(full_ids)
    mask_until = min(len(prompt_ids), len(labels))
    for i in range(mask_until):
        labels[i] = -100

    pad_len = MAX_LEN - len(full_ids)
    full_ids = full_ids + [tokenizer.pad_token_id] * pad_len
    labels = labels + [-100] * pad_len
    attention = [1] * (MAX_LEN - pad_len) + [0] * pad_len

    return {"input_ids": full_ids, "attention_mask": attention, "labels": labels}


def self_test() -> int:
    """Mock self-test without torch downloads. Verifies sample-filter and version-pick."""
    import tempfile

    fd = tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    try:
        fd.write(json.dumps({"approved": True, "situation": "x", "source": "critic", "better_response": "y"}) + "\n")
        fd.write(json.dumps({"approved": False, "situation": "x", "source": "critic", "better_response": "y"}) + "\n")
        fd.write(json.dumps({"approved": True, "situation": "x", "source": "thumbs_up", "pi_response": "z"}) + "\n")
        fd.write(json.dumps({"approved": True, "situation": "x", "source": "thumbs_down"}) + "\n")
        fd.write(json.dumps({"approved": True, "situation": "", "source": "critic", "better_response": "y"}) + "\n")
        fd.write("not json\n")
        path = Path(fd.name)
    finally:
        fd.close()
    try:
        raw = load_samples(path)
        assert len(raw) == 2, f"expected 2 raw pairs, got {len(raw)}"
        assert sorted(p["source"] for p in raw) == ["critic", "thumbs_up"]

        weighted = apply_weighting_and_cap(raw)
        # 1 critic * 3 + 1 thumbs_up * 1 = 4
        assert len(weighted) == 4, f"expected 4 weighted, got {len(weighted)}"
        c = sum(1 for p in weighted if p["source"] == "critic")
        t = sum(1 for p in weighted if p["source"] == "thumbs_up")
        assert (c, t) == (WEIGHT_CRITIC, WEIGHT_THUMBS_UP), f"weights wrong: {c=} {t=}"
    finally:
        path.unlink()

    with tempfile.TemporaryDirectory() as d:
        out = Path(d)

        def pick() -> int:
            existing = sorted(
                (x for x in out.iterdir() if x.is_dir() and re.match(r"v\d+$", x.name)),
                key=lambda x: int(x.name[1:]),
            )
            return int(existing[-1].name[1:]) + 1 if existing else 1

        assert pick() == 1
        for n in ("v1", "v3", "v10", "not-a-version"):
            (out / n).mkdir()
        assert pick() == 11

    print("[trainer] self-test OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--base", default=BASE_MODEL)
    args = ap.parse_args()

    if not args.samples.exists():
        print(f"[trainer] samples file not found: {args.samples}", file=sys.stderr)
        return 2

    raw_pairs = load_samples(args.samples)
    if not raw_pairs:
        print(
            "[trainer] no usable samples (need approved=true + critic/thumbs_up with non-empty target)",
            file=sys.stderr,
        )
        return 3
    pairs = apply_weighting_and_cap(raw_pairs)
    raw_breakdown = dict(Counter(p["source"] for p in raw_pairs))
    eff_breakdown = dict(Counter(p["source"] for p in pairs))
    print(
        f"[trainer] {len(raw_pairs)} raw approved {raw_breakdown} "
        f"-> {len(pairs)} effective (critic x{WEIGHT_CRITIC}, thumbs_up x{WEIGHT_THUMBS_UP}) "
        f"{eff_breakdown}, threads={CPU_THREADS}"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        (d for d in args.out.iterdir() if d.is_dir() and re.match(r"v\d+$", d.name)),
        key=lambda d: int(d.name[1:]),
    )
    version = int(existing[-1].name[1:]) + 1 if existing else 1
    target_dir = args.out / f"v{version}"
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[trainer] training v{version} -> {target_dir}")

    print(f"[trainer] loading base {args.base} on CPU")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    rows = [encode_pair(tokenizer, p["input"], p["output"]) for p in pairs]
    ds = Dataset.from_list(rows)

    train_args = TrainingArguments(
        output_dir=str(target_dir / "trainer_state"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        fp16=False,
        bf16=False,
        no_cuda=True,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        processing_class=tokenizer,
        data_collator=default_data_collator,
    )

    t0 = time.time()
    result = trainer.train()
    duration = time.time() - t0

    model.save_pretrained(str(target_dir))
    tokenizer.save_pretrained(str(target_dir))

    log = {
        "version": f"v{version}",
        "base": args.base,
        "samples_used_raw": len(raw_pairs),
        "samples_used_effective": len(pairs),
        "samples_breakdown_raw": raw_breakdown,
        "samples_breakdown_effective": eff_breakdown,
        "sample_weights": {"critic": WEIGHT_CRITIC, "thumbs_up": WEIGHT_THUMBS_UP},
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "cpu_threads": CPU_THREADS,
        "final_loss": float(result.training_loss),
        "duration_seconds": round(duration, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (target_dir / "training_log.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )
    print(
        f"[trainer] done v{version}, loss={log['final_loss']:.4f}, "
        f"{log['duration_seconds']}s"
    )

    versions = sorted(
        (d for d in args.out.iterdir() if d.is_dir() and re.match(r"v\d+$", d.name)),
        key=lambda d: int(d.name[1:]),
    )
    if len(versions) > KEEP_LAST:
        old = [d.name for d in versions[:-KEEP_LAST]]
        print(
            f"[trainer] note: more than {KEEP_LAST} versions present; "
            f"old (manual prune): {old}"
        )

    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(self_test())
    raise SystemExit(main())
