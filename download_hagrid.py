#!/usr/bin/env python3
"""Download HaGRID Classification (512p) und extrahiere relevante Gesten.

Laedt das ZIP (~3.8GB), extrahiert nur die 6 relevanten Klassen
mit max 200 Bildern pro Klasse, loescht den Rest.
"""
import os
import sys
import subprocess
import shutil
import random

DEST = "/mnt/moloch-data/reference/gestures"
TMP = "/mnt/moloch-data/reference/_tmp_hagrid"
ZIP_URL = "https://huggingface.co/datasets/cj-mills/hagrid-classification-512p-no-gesture-150k-zip/resolve/main/hagrid-classification-512p-no-gesture-150k.zip"
ZIP_PATH = "/mnt/moloch-data/reference/_hagrid.zip"
MAX_PER_CLASS = 200

# Mapping: HaGRID class -> our folder name
CLASS_MAP = {
    "like": "thumbs_up",
    "peace": "peace",
    "palm": "open_hand",
    "fist": "fist",
    "one": "pointing",
    "stop": "wave",  # stop = open hand halt, closest to wave
}

print("=== HaGRID Classification Download ===")
print(f"URL: {ZIP_URL}")
print(f"Ziel: {DEST}")
print(f"Max pro Klasse: {MAX_PER_CLASS}")

# Download ZIP
if not os.path.exists(ZIP_PATH):
    print("\nDownloading ZIP (~3.8GB)...")
    r = subprocess.run(
        ["wget", "-O", ZIP_PATH, "--progress=dot:giga", ZIP_URL],
        timeout=1800
    )
    if r.returncode != 0:
        print("DOWNLOAD FAILED!")
        sys.exit(1)
    print("Download OK")
else:
    print(f"\nZIP bereits vorhanden: {ZIP_PATH}")

# Extract only needed classes
os.makedirs(TMP, exist_ok=True)
print("\nExtrahiere relevante Klassen...")

for hagrid_class in CLASS_MAP:
    print(f"  Extrahiere {hagrid_class}/...")
    # unzip specific folder
    subprocess.run(
        ["unzip", "-q", "-o", ZIP_PATH,
         f"hagrid-classification-512p-no-gesture-150k/{hagrid_class}/*",
         "-d", TMP],
        timeout=300
    )

# Copy max N images per class to destination
print(f"\nKopiere max {MAX_PER_CLASS} Bilder pro Klasse...")
for hagrid_class, our_name in CLASS_MAP.items():
    src_dir = os.path.join(TMP, "hagrid-classification-512p-no-gesture-150k", hagrid_class)
    dst_dir = os.path.join(DEST, our_name)
    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.exists(src_dir):
        print(f"  SKIP {hagrid_class} - nicht gefunden!")
        continue

    all_files = [f for f in os.listdir(src_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.seed(42)  # Reproducible selection
    selected = random.sample(all_files, min(MAX_PER_CLASS, len(all_files)))

    for fname in selected:
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

    print(f"  {our_name}: {len(selected)}/{len(all_files)} Bilder kopiert")

# Cleanup
print("\nAufraeumen...")
shutil.rmtree(TMP, ignore_errors=True)
os.remove(ZIP_PATH)
print("ZIP + Temp geloescht")

# Report
print("\n=== HAGRID GESTURES ===")
for gesture in sorted(os.listdir(DEST)):
    d = os.path.join(DEST, gesture)
    if os.path.isdir(d):
        n = len([f for f in os.listdir(d)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  {gesture}: {n}")
