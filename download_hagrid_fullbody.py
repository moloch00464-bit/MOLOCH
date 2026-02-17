#!/usr/bin/env python3
"""Download HaGRID v1 Full-Body Bilder (100 pro Klasse).

Laedt jede Klasse einzeln von Sbercloud, extrahiert 100 Bilder, loescht ZIP.
Sequentiell um Disk-Platz zu sparen (~40GB Temp pro Klasse).
"""
import os
import sys
import subprocess
import shutil
import random
import zipfile

DEST = "/mnt/moloch-data/reference/gestures_fullbody"
TMP = "/mnt/moloch-data/reference/_tmp_fullbody"
MAX_PER_CLASS = 100

# HaGRID v1 Sbercloud URLs (Full-Body FullHD)
BASE_URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset"

# Mapping: Sbercloud ZIP name -> unser Ordnername
CLASS_MAP = {
    "like": "thumbs_up",
    "peace": "peace",
    "palm": "open_hand",
    "fist": "fist",
    "one": "pointing",
    "stop": "wave",
}

print("=== HaGRID Full-Body Download ===")
print(f"Ziel: {DEST}")
print(f"Max pro Klasse: {MAX_PER_CLASS}")
print(f"Klassen: {list(CLASS_MAP.keys())}")

os.makedirs(DEST, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

for hagrid_name, our_name in CLASS_MAP.items():
    dst_dir = os.path.join(DEST, our_name)

    # Skip wenn schon genug Bilder da sind
    if os.path.exists(dst_dir):
        existing = [f for f in os.listdir(dst_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(existing) >= MAX_PER_CLASS:
            print(f"\n--- {our_name}: {len(existing)} Bilder vorhanden, SKIP ---")
            continue

    zip_url = f"{BASE_URL}/{hagrid_name}.zip"
    zip_path = os.path.join(TMP, f"{hagrid_name}.zip")

    print(f"\n--- {hagrid_name} -> {our_name} ---")
    print(f"URL: {zip_url}")

    # Download
    if not os.path.exists(zip_path):
        print(f"Downloading {hagrid_name}.zip...")
        r = subprocess.run(
            ["wget", "-O", zip_path, "--progress=dot:giga",
             "--timeout=60", "--tries=3", zip_url],
            timeout=7200  # Max 2h pro Klasse
        )
        if r.returncode != 0:
            print(f"DOWNLOAD FAILED fuer {hagrid_name}!")
            # Cleanup und weiter
            if os.path.exists(zip_path):
                os.remove(zip_path)
            continue
    else:
        print(f"ZIP bereits vorhanden: {zip_path}")

    # Extrahiere 100 zufaellige Bilder aus dem ZIP
    print(f"Extrahiere {MAX_PER_CLASS} Bilder...")
    os.makedirs(dst_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Alle JPG/PNG Dateien im ZIP finden
            all_images = [n for n in zf.namelist()
                         if n.lower().endswith(('.jpg', '.jpeg', '.png'))
                         and not n.startswith('__MACOSX')]

            if not all_images:
                print(f"KEINE BILDER in {hagrid_name}.zip!")
                os.remove(zip_path)
                continue

            # Zufaellige Auswahl
            random.seed(42)
            selected = random.sample(all_images, min(MAX_PER_CLASS, len(all_images)))

            count = 0
            for img_path in selected:
                fname = os.path.basename(img_path)
                if not fname:
                    continue
                try:
                    data = zf.read(img_path)
                    with open(os.path.join(dst_dir, fname), 'wb') as f:
                        f.write(data)
                    count += 1
                except Exception as e:
                    print(f"  Skip {fname}: {e}")

            print(f"  {our_name}: {count}/{len(all_images)} Bilder extrahiert")

    except zipfile.BadZipFile:
        print(f"KORRUPTES ZIP: {zip_path}")
    except Exception as e:
        print(f"FEHLER: {e}")

    # ZIP loeschen (spart ~40GB)
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"  ZIP geloescht ({hagrid_name}.zip)")

# Cleanup
shutil.rmtree(TMP, ignore_errors=True)

# Report
print("\n=== HAGRID FULL-BODY GESTURES ===")
total = 0
for gesture in sorted(os.listdir(DEST)):
    d = os.path.join(DEST, gesture)
    if os.path.isdir(d):
        n = len([f for f in os.listdir(d)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total += n
        print(f"  {gesture}: {n}")
print(f"  GESAMT: {total}")
