#!/usr/bin/env python3
"""Download FER2013 von HuggingFace und sortiere in Emotion-Ordner."""
import os
import sys
import numpy as np
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from PIL import Image

DEST = "/mnt/moloch-data/reference/emotions"
LABELS = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprised", 6: "neutral"}

def process_split(split_name):
    """Download and process one split (train/test/valid)."""
    print(f"\n--- {split_name} ---")
    path = hf_hub_download(
        repo_id="AutumnQiu/fer2013",
        filename=f"data/{split_name}-00000-of-00001.parquet",
        repo_type="dataset"
    )
    print(f"Downloaded: {path}")

    table = pq.read_table(path)
    df = table.to_pydict()

    # FER2013 hat 'pixels' (space-separated string) und 'label' (int)
    # Oder 'image' (dict mit bytes) und 'label'
    keys = list(df.keys())
    print(f"Columns: {keys}")
    n = len(df[keys[0]])
    print(f"Rows: {n}")

    count = 0
    for i in range(n):
        label = df.get("label", df.get("labels", [None]))[i]
        if label is None or label not in LABELS:
            continue

        emotion = LABELS[label]
        out_dir = os.path.join(DEST, emotion)

        # Try 'image' column (dict with 'bytes' or 'path')
        if "image" in df:
            img_data = df["image"][i]
            if isinstance(img_data, dict) and "bytes" in img_data and img_data["bytes"]:
                import io
                img = Image.open(io.BytesIO(img_data["bytes"]))
            elif isinstance(img_data, dict) and "path" in img_data:
                img = Image.open(img_data["path"])
            else:
                continue
        # Try 'pixels' column (space-separated string)
        elif "pixels" in df:
            pixels = df["pixels"][i]
            arr = np.array([int(p) for p in pixels.split()], dtype=np.uint8).reshape(48, 48)
            img = Image.fromarray(arr, mode="L")
        else:
            print(f"Unknown format! Keys: {keys}")
            sys.exit(1)

        fname = f"{split_name}_{i:05d}.png"
        img.save(os.path.join(out_dir, fname))
        count += 1

    print(f"Saved {count} images from {split_name}")
    return count

total = 0
for split in ["train", "test", "valid"]:
    try:
        total += process_split(split)
    except Exception as e:
        print(f"Error in {split}: {e}")

print(f"\n=== FER2013 TOTAL: {total} images ===")

# Report
for emotion in sorted(LABELS.values()):
    d = os.path.join(DEST, emotion)
    n = len([f for f in os.listdir(d) if f.endswith(".png")])
    print(f"  {emotion}: {n}")
